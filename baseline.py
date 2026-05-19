import os
import re
import json
import urllib.request
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# =====================================================================
# Configuration & Setup
# =====================================================================

# Load local .env if it exists
if os.path.exists(".env"):
    with open(".env", "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                os.environ[key.strip()] = val.strip()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")

# Default validation split (first N items from train.csv)
VAL_SIZE = 50

# =====================================================================
# Helper Functions
# =====================================================================

def extract_answer(text):
    """
    Extracts the final answer from the LaTeX \boxed{...} notation.
    Supports simple and nested brackets.
    """
    if not text:
        return ""
    # Try to match one level of nested curly brackets inside \boxed{}
    match = re.search(r'\\boxed{([^{}]*(?:{[^{}]*}[^{}]*)*)}', text)
    if match:
        return match.group(1).strip()
    # Fallback to simple non-greedy regex
    match = re.search(r'\\boxed{(.*?)}', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ""

def clean_string(s):
    """
    Cleans output strings for exact matching.
    """
    if not s:
        return ""
    s = str(s).strip().lower()
    # Remove extra spaces, quotes, etc.
    return re.sub(r'\s+', ' ', s)

def compare_answers(pred, target):
    """
    Compares predicted and target answers.
    Supports exact matching and standard Kaggle tolerance for numbers.
    """
    pred_clean = clean_string(pred)
    target_clean = clean_string(target)
    
    if pred_clean == target_clean:
        return True
        
    # Attempt float comparison if both can be parsed as numbers
    try:
        pred_val = float(pred_clean)
        target_val = float(target_clean)
        # Kaggle tolerance: 1e-2 relative or absolute
        return abs(pred_val - target_val) <= 1e-2 or abs((pred_val - target_val) / (target_val + 1e-9)) <= 1e-2
    except ValueError:
        pass
        
    return False

# =====================================================================
# API Callers (urllib only - zero dependencies)
# =====================================================================

def call_gemini(prompt):
    """
    Calls the Gemini API using urllib.
    """
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY environment variable not found.")
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "temperature": 0.1,
            "maxOutputTokens": 2048
        }
    }
    
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    with urllib.request.urlopen(req) as response:
        res_data = json.loads(response.read().decode("utf-8"))
        return res_data["candidates"][0]["content"]["parts"][0]["text"]

def call_claude(prompt):
    """
    Calls the Anthropic Claude API using urllib.
    """
    if not ANTHROPIC_API_KEY:
        raise ValueError("ANTHROPIC_API_KEY environment variable not found.")
        
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "content-type": "application/json",
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": "2023-06-01"
    }
    
    payload = {
        "model": "claude-3-5-sonnet-20241022",
        "max_tokens": 2048,
        "temperature": 0.1,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }
    
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    with urllib.request.urlopen(req) as response:
        res_data = json.loads(response.read().decode("utf-8"))
        return res_data["content"][0]["text"]

def query_llm(prompt):
    """
    Tries Gemini API first, falls back to Anthropic Claude if Gemini is missing.
    """
    if GEMINI_API_KEY:
        return call_gemini(prompt)
    elif ANTHROPIC_API_KEY:
        return call_claude(prompt)
    else:
        raise ValueError("No API keys found. Please export GEMINI_API_KEY or ANTHROPIC_API_KEY.")

# =====================================================================
# Main Evaluation Loop
# =====================================================================

def evaluate_row(row):
    row_id = row['id']
    prompt = row['prompt']
    target = row['answer']
    
    # Append strict instructions to output LaTeX box format
    formatted_prompt = (
        f"{prompt}\n\n"
        "IMPORTANT: You must think step-by-step. "
        "At the very end of your response, you must output your final answer wrapped inside a LaTeX \\boxed{} command. "
        "For example: \\boxed{42} or \\boxed{True} or \\boxed{abc}. Only the value inside \\boxed{} will be graded."
    )
    
    try:
        raw_response = query_llm(formatted_prompt)
        pred = extract_answer(raw_response)
        is_correct = compare_answers(pred, target)
        return {
            "id": row_id,
            "prompt_snippet": prompt[:60].replace("\n", " "),
            "target": target,
            "prediction": pred,
            "is_correct": is_correct,
            "error": None
        }
    except Exception as e:
        return {
            "id": row_id,
            "prompt_snippet": prompt[:60].replace("\n", " "),
            "target": target,
            "prediction": "",
            "is_correct": False,
            "error": str(e)
        }

def main():
    train_csv_path = "train.csv"
    if not os.path.exists(train_csv_path):
        print(f"❌ Error: '{train_csv_path}' not found at the current path!")
        print("Please download the dataset from Kaggle and place 'train.csv' in the workspace directory.")
        return
    
    print(f"🚀 Loading dataset from {train_csv_path}...")
    df = pd.read_csv(train_csv_path)
    
    # Ensure we have enough rows
    eval_df = df.head(min(VAL_SIZE, len(df)))
    print(f"📊 Selected first {len(eval_df)} rows for local validation.")
    
    results = []
    print(f"🤖 Running evaluation via LLM API (using parallel threads)...")
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(evaluate_row, row): index for index, row in eval_df.iterrows()}
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            status = "✅" if res["is_correct"] else "❌"
            if res["error"]:
                status = "⚠️ Error"
            print(f"[{res['id']}] {status} Target: {res['target']} | Pred: {res['prediction']} | Snippet: {res['prompt_snippet']}")

    # Calculate metrics
    results_df = pd.DataFrame(results)
    total = len(results_df)
    correct = results_df["is_correct"].sum()
    errors = results_df["error"].notnull().sum()
    accuracy = (correct / total) * 100 if total > 0 else 0
    
    print("\n" + "="*60)
    print("📊 LOCAL VALIDATION REPORT")
    print("="*60)
    print(f"Total Evaluated: {total}")
    print(f"Correct Answers: {correct}")
    print(f"Failed/Errors:   {errors}")
    print(f"Local Accuracy:  {accuracy:.2f}%")
    print("="*60)
    
    # Save results
    results_df.to_csv("baseline_results.csv", index=False)
    print("📝 Detailed results saved to 'baseline_results.csv'")

if __name__ == "__main__":
    main()
