import os
import re
import pandas as pd

class DatasetSanitizer:
    def __init__(self, train_csv_path="train.csv"):
        self.df = pd.read_csv(train_csv_path)
        
    def check_bit_manipulation_mismatch(self, prompt, answer):
        """
        Detects if the prompt examples have intermediate mismatches, OR if the 
        final calculation in the prompt contradicts the target answer.
        """
        # Check for "Result: X" and compare it with the ground truth answer
        result_match = re.search(r'Result:\s*([^\n]+)', prompt, re.IGNORECASE)
        if result_match:
            computed = result_match.group(1).strip().lower()
            ans_clean = str(answer).strip().lower()
            # Strip out brackets or extra spaces
            computed = re.sub(r'[\\ boxed{}]', '', computed)
            if computed != ans_clean:
                return True # Mismatch!
                
        # Check all individual examples inside the prompt for internal logic mismatches
        # Examples format: "01010001 -> 11011101"
        examples = re.findall(r'([01]{8})\s*->\s*([01]{8})', prompt)
        # If it is a bit manipulation prompt, let's verify if we can find inconsistencies
        # (e.g. same input leading to different outputs, which violates function mapping)
        seen_inputs = {}
        for inp, out in examples:
            if inp in seen_inputs and seen_inputs[inp] != out:
                return True # Contradicting examples!
            seen_inputs[inp] = out
            
        return False

    def check_equation_transformation_quality(self, prompt):
        """
        Checks if equation transformation ciphers have character mapping bugs:
        1. Input/Output length mismatches (len(lhs) != len(rhs))
        2. Unmapped/Unknown characters in the query string.
        """
        # Extract examples like "LHS = RHS" or "LHS -> RHS" or cipher words
        # Format: "word1 -> word2" or "word1 = word2"
        lines = prompt.split('\n')
        examples = []
        query = ""
        for line in lines:
            if '->' in line:
                parts = line.split('->')
                examples.append((parts[0].strip(), parts[1].strip()))
            elif '=' in line and 'Observation' not in line and 't =' not in line:
                parts = line.split('=')
                examples.append((parts[0].strip(), parts[1].strip()))
            elif 'decrypt the following text:' in line:
                query = line.split('decrypt the following text:')[1].strip()
            elif 'determine the result for:' in line:
                query = line.split('determine the result for:')[1].strip()

        if not examples or not query:
            return "valid" # Not applicable or not parseable
            
        # 1. Check input/output length mismatches
        for lhs, rhs in examples:
            if len(lhs) != len(rhs):
                return "length_mismatch"
                
        # 2. Check for unknown characters in the query alphabet
        char_map = {}
        for lhs, rhs in examples:
            for j in range(min(len(lhs), len(rhs))):
                if lhs[j] not in char_map:
                    char_map[lhs[j]] = rhs[j]
                    
        result = "".join(char_map.get(c, "?") for c in query)
        if "?" in result:
            return "unknown_query_chars"
            
        return "valid"

    def _classify_prompt(self, prompt):
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

    def sanitize(self):
        print(f"🧹 Running dataset sanitizer on {len(self.df)} rows...")
        
        clean_rows = []
        buggy_rows = []
        
        stats = {
            "Total": 0, "Clean": 0, "Buggy": 0,
            "Bit_Buggy": 0, "Equation_Buggy": 0
        }
        
        for idx, row in self.df.iterrows():
            stats["Total"] += 1
            prompt = row['prompt']
            answer = row['answer']
            category = self._classify_prompt(prompt)
            
            is_buggy = False
            bug_reason = ""
            
            if category == "Bit":
                if self.check_bit_manipulation_mismatch(prompt, answer):
                    is_buggy = True
                    bug_reason = "bit_mismatch"
                    stats["Bit_Buggy"] += 1
            elif category == "Equation" or category == "Cipher":
                quality = self.check_equation_transformation_quality(prompt)
                if quality != "valid":
                    is_buggy = True
                    bug_reason = quality
                    stats["Equation_Buggy"] += 1
                    
            row_data = {
                "id": row['id'],
                "prompt": prompt,
                "answer": answer,
                "category": category,
                "bug_reason": bug_reason
            }
            
            if is_buggy:
                buggy_rows.append(row_data)
                stats["Buggy"] += 1
            else:
                clean_rows.append(row_data)
                stats["Clean"] += 1
                
        # Create output directories
        os.makedirs("data", exist_ok=True)
        
        clean_df = pd.DataFrame(clean_rows)
        buggy_df = pd.DataFrame(buggy_rows)
        
        clean_df.to_csv("data/cleaned_train.csv", index=False)
        buggy_df.to_csv("data/buggy_train.csv", index=False)
        
        print("\n" + "="*60)
        print("📊 DATASET SANITIZATION REPORT")
        print("="*60)
        print(f"Total Train Puzzles:   {stats['Total']}")
        print(f"Clean Puzzles Saved:   {stats['Clean']} ({(stats['Clean']/stats['Total'])*100:.2f}%)")
        print(f"Buggy Puzzles Saved:   {stats['Buggy']} ({(stats['Buggy']/stats['Total'])*100:.2f}%)")
        print(f"  - Broken Bit manipulation: {stats['Bit_Buggy']}")
        print(f"  - Broken Equations/Ciphers: {stats['Equation_Buggy']}")
        print("="*60)
        print("📝 Saved output files to 'data/cleaned_train.csv' and 'data/buggy_train.csv'.")

if __name__ == "__main__":
    sanitizer = DatasetSanitizer()
    sanitizer.sanitize()
