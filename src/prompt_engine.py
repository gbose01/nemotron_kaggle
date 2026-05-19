import re
import pandas as pd

class PromptEngine:
    def __init__(self, train_csv_path="train.csv"):
        self.df = pd.read_csv(train_csv_path)
        self.categorized_data = self._categorize_dataset()
        
    def _classify_prompt(self, prompt):
        """
        Categorizes a prompt into one of the predefined logical puzzle classes.
        """
        prompt_lower = prompt.lower()
        if "gravitational constant" in prompt_lower:
            return "Gravity"
        elif "becomes" in prompt_lower:
            return "Linear"
        elif "numeral system" in prompt_lower:
            return "Roman"
        elif "decrypt the following text" in prompt_lower:
            return "Cipher"
        elif "bit manipulation" in prompt_lower:
            return "Bit"
        else:
            return "Equation"
            
    def _categorize_dataset(self):
        """
        Indexes the entire training dataset by category for fast, relevant few-shot retrieval.
        """
        categorized = {
            "Gravity": [],
            "Linear": [],
            "Roman": [],
            "Cipher": [],
            "Bit": [],
            "Equation": []
        }
        for idx, row in self.df.iterrows():
            cat = self._classify_prompt(row['prompt'])
            categorized[cat].append(row)
        return categorized

    def get_system_prompt(self):
        return (
            "You are a world-class logical reasoning agent. Your task is to solve the logical puzzle provided below.\n\n"
            "CRITICAL GUIDELINE:\n"
            "1. You must think step-by-step to deduce the hidden rule of the puzzle. Write your logical analysis inside <think> ... </think> tags.\n"
            "2. Your deduction MUST be fully consistent with all the input-output examples provided.\n"
            "3. Once you have deduced the rule, apply it carefully to the final query.\n"
            "4. At the very end, you MUST output your final answer wrapped inside a LaTeX \\boxed{} command (e.g. \\boxed{42} or \\boxed{True} or \\boxed{abc}). Only the boxed string will be evaluated."
        )

    def build_prompt(self, query_prompt, num_shots=3):
        """
        Classifies the query, fetches the best matching solved examples from train.csv, and packages the final prompt.
        """
        category = self._classify_prompt(query_prompt)
        exemplars = self.categorized_data[category]
        
        # Filter out the current query if it happens to be in our exemplars (prevent exact overlap leakage)
        clean_exemplars = []
        for ex in exemplars:
            if query_prompt.strip()[:100] != ex['prompt'].strip()[:100]:
                clean_exemplars.append(ex)
                if len(clean_exemplars) >= num_shots:
                    break
                    
        # Format few-shot examples
        few_shot_text = ""
        if clean_exemplars:
            few_shot_text += "### EXAMPLES OF SIMILAR PUZZLES AND SOLUTIONS:\n\n"
            for idx, ex in enumerate(clean_exemplars):
                few_shot_text += f"--- Example {idx + 1} ---\n"
                # We strip out the final instruction from exemplars to keep them clean
                clean_ex_prompt = re.sub(r'Now, (?:determine|convert|write|decrypt).*', '', ex['prompt'], flags=re.DOTALL).strip()
                few_shot_text += f"Puzzle:\n{clean_ex_prompt}\n\n"
                few_shot_text += f"Step-by-Step Solution:\n<think>\n[Deducing rule: ...]\n</think>\n\\\\boxed{{{ex['answer']}}}\n\n"
            few_shot_text += "============================================================\n\n"
            
        # Formulate final prompt
        final_prompt = (
            f"{self.get_system_prompt()}\n\n"
            f"{few_shot_text}"
            "### YOUR PUZZLE TO SOLVE:\n\n"
            f"{query_prompt}\n\n"
            "Let's think step-by-step and write down the final answer inside \\\\boxed{}."
        )
        
        return final_prompt
