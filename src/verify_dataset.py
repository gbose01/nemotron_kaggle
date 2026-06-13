import os
import json
import re

class SftDatasetVerifier:
    def __init__(self, dataset_path="data/sft_reasoning_dataset_v2.jsonl"):
        self.dataset_path = dataset_path

    def verify(self):
        print(f"🔬 Running dataset verifier on '{self.dataset_path}'...")
        
        if not os.path.exists(self.dataset_path):
            print(f"❌ Error: Dataset file '{self.dataset_path}' not found!")
            return False
            
        total_rows = 0
        valid_rows = 0
        syntax_errors = 0
        schema_errors = 0
        cot_format_errors = 0
        
        stats = {
            "Roman": 0, "Linear": 0, "Gravity": 0,
            "Cipher": 0, "Bit": 0, "Equation": 0,
            "Synthetic": 0
        }
        
        with open(self.dataset_path, "r") as f:
            for line_idx, line in enumerate(f):
                total_rows += 1
                # 1. Check JSON syntax
                try:
                    data = json.loads(line)
                except Exception as e:
                    print(f"  Line {line_idx + 1}: JSON Syntax Error: {e}")
                    syntax_errors += 1
                    continue
                    
                # 2. Check Schema
                if "id" not in data or "prompt" not in data or "completion" not in data:
                    print(f"  Line {line_idx + 1}: Schema Error (missing id, prompt, or completion columns)")
                    schema_errors += 1
                    continue
                    
                prompt = data["prompt"]
                response = data["completion"]
                row_id = data["id"]
                
                # 3. Check CoT formatting
                has_think_start = response.startswith("<think>")
                has_think_end = "</think>" in response
                has_boxed = "\\boxed{" in response
                
                if not (has_think_start and has_think_end and has_boxed):
                    print(f"  Row {row_id}: CoT Formatting Error. Missing <think>, </think> or \\boxed{{}}.")
                    cot_format_errors += 1
                    continue
                    
                # 4. Tally stats
                if str(row_id).startswith("syn_"):
                    stats["Synthetic"] += 1
                else:
                    # Classify category based on prompt
                    prompt_lower = prompt.lower()
                    if "gravitational constant" in prompt_lower:
                        stats["Gravity"] += 1
                    elif "becomes" in prompt_lower:
                        stats["Linear"] += 1
                    elif "numeral system" in prompt_lower:
                        stats["Roman"] += 1
                    elif "decrypt the following text" in prompt_lower:
                        stats["Cipher"] += 1
                    elif "bit manipulation" in prompt_lower:
                        stats["Bit"] += 1
                    else:
                        stats["Equation"] += 1
                        
                valid_rows += 1
                
        success = syntax_errors == 0 and schema_errors == 0 and cot_format_errors == 0
        
        print("\n" + "="*60)
        print("📊 SFT DATASET HEALTH & VERIFICATION REPORT")
        print("="*60)
        print(f"File Tested:          {self.dataset_path}")
        print(f"Total Rows Scanned:   {total_rows}")
        print(f"Valid SFT Rows:       {valid_rows} ({(valid_rows/total_rows)*100:.2f}%)")
        print(f"Syntax/JSON Errors:   {syntax_errors}")
        print(f"Schema Errors:        {schema_errors}")
        print(f"CoT Format Errors:    {cot_format_errors}")
        print("-"*60)
        print("🧩 CATEGORY BREAKDOWN:")
        print(f"  - Gravity Puzzles:  {stats['Gravity']}")
        print(f"  - Linear Puzzles:   {stats['Linear']}")
        print(f"  - Roman Puzzles:    {stats['Roman']}")
        print(f"  - Cipher Puzzles:   {stats['Cipher']}")
        print(f"  - Bit Puzzles:      {stats['Bit']}")
        print(f"  - Equation Puzzles: {stats['Equation']}")
        print(f"  - Synthetic Puzzles: {stats['Synthetic']}")
        print("="*60)
        
        if success:
            print("🎉 SUCCESS: The SFT training dataset is 100% clean and formatted perfectly!")
            return True
        else:
            print("❌ FAILURE: Dataset contains validation errors. Please fix before training.")
            return False

if __name__ == "__main__":
    verifier = SftDatasetVerifier()
    verifier.verify()
