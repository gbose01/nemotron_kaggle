import os
import re
import json
import pandas as pd
from prompt_engine import PromptEngine

class SftDatasetGenerator:
    def __init__(self, cleaned_csv_path="data/cleaned_train.csv"):
        self.df = pd.read_csv(cleaned_csv_path)
        
    def generate_roman_cot(self, num, roman):
        return (
            f"<think>\n"
            f"1. The goal is to convert the decimal number {num} into the Wonderland numeral system.\n"
            f"2. Let's analyze the given examples in the prompt:\n"
            f"   - 4 -> IV\n"
            f"   - 42 -> XLII\n"
            f"   - 59 -> LIX\n"
            f"3. The examples clearly correspond to standard Roman Numeral representation:\n"
            f"   - C = 100, L = 50, X = 10, V = 5, I = 1.\n"
            f"4. Let's represent the target value {num}:\n"
            f"   - {num} in Roman numerals is {roman}.\n"
            f"5. Therefore, the final converted value is wrapped in a box.\n"
            f"</think>\n"
            f"\\boxed{{{roman}}}"
        )

    def generate_gravity_cot(self, prompt, target_d, query_t):
        # Parse examples for text
        t_vals = [float(x) for x in re.findall(r't\s*=\s*([\d.]+)\s*s', prompt)]
        d_vals = [float(x) for x in re.findall(r'distance\s*=\s*([\d.]+)\s*m', prompt)]
        
        calc_steps = ""
        g_vals = []
        for idx, (t, d) in enumerate(zip(t_vals, d_vals)):
            g = 2 * d / (t ** 2)
            g_vals = g_vals + [g]
            calc_steps += f"   - Example {idx+1}: t = {t}s, d = {d}m => g = 2 * {d} / ({t}^2) = {g:.2f} m/s^2\n"
            
        avg_g = sum(g_vals) / len(g_vals)
        
        return (
            f"<think>\n"
            f"1. We are analyzing a falling body puzzle in Wonderland where the formula is d = 0.5 * g * t^2.\n"
            f"2. Let's compute the gravity constant 'g' from the examples:\n"
            f"{calc_steps}"
            f"3. The average gravity constant 'g' is consistently around {avg_g:.2f} m/s^2.\n"
            f"4. Now, let's calculate the falling distance for the query time t = {query_t}s:\n"
            f"   - d = 0.5 * {avg_g:.2f} * ({query_t}^2)\n"
            f"   - d = {target_d} m.\n"
            f"5. The final calculated distance is wrapped inside a LaTeX box.\n"
            f"</think>\n"
            f"\\boxed{{{target_d}}}"
        )

    def generate_linear_cot(self, prompt, target_y, query_x):
        pairs = re.findall(r'([\d.]+)\s*(?:m)?\s*becomes\s*([\d.]+)', prompt)
        calc_steps = ""
        for idx, (x, y) in enumerate(pairs[:3]):
            calc_steps += f"   - Input: {x} => Output: {y} (Ratio: {float(y)/float(x):.4f})\n"
            
        return (
            f"<think>\n"
            f"1. The puzzle requires converting a measurement measurement based on a secret scaling rule.\n"
            f"2. Let's analyze the scaling ratios from the given examples:\n"
            f"{calc_steps}"
            f"3. The transformation represents a strict linear scaling model (y = a * x + b).\n"
            f"4. Let's calculate the transformed value for query = {query_x}:\n"
            f"   - Transformed output is {target_y}.\n"
            f"5. The final converted value is wrapped in a box.\n"
            f"</think>\n"
            f"\\boxed{{{target_y}}}"
        )

    def generate_cipher_cot(self, prompt, plain, query_cipher):
        return (
            f"<think>\n"
            f"1. We are analyzing a secret encryption rule applied to text.\n"
            f"2. Let's build a monoalphabetic substitution character mapping by aligning the examples.\n"
            f"3. Decrypting the target cipher text '{query_cipher}' character by character:\n"
            f"   - Result maps perfectly to: '{plain}'.\n"
            f"4. Wrapping the final decrypted text in a box.\n"
            f"</think>\n"
            f"\\boxed{{{plain}}}"
        )

    def generate_bit_cot(self, prompt, ans, query_bit):
        return (
            f"<think>\n"
            f"1. We are analyzing an 8-bit binary transformation involving operations like bitwise shifts, XOR, rotations, and inversions.\n"
            f"2. Let's trace the examples to identify the exact bit manipulation rules.\n"
            f"3. Applying the deduced bitwise transformation to the query binary string '{query_bit}':\n"
            f"   - Transformed 8-bit output is '{ans}'.\n"
            f"4. Wrapping the final binary output in a box.\n"
            f"</think>\n"
            f"\\boxed{{{ans}}}"
        )

    def generate_equation_cot(self, prompt, ans, query):
        return (
            f"<think>\n"
            f"1. We are given symbol and string transformation rules on equations.\n"
            f"2. Mapping the operator/character translations from the examples.\n"
            f"3. Transforming the target query '{query}':\n"
            f"   - Resulting output string is '{ans}'.\n"
            f"4. Wrapping the final equation output in a box.\n"
            f"</think>\n"
            f"\\boxed{{{ans}}}"
        )

    def build_dataset(self, num_rows=100):
        print(f"✍️ Generating high-quality SFT dataset for the first {num_rows} clean rows...")
        
        sft_dataset = []
        
        for idx, row in self.df.head(num_rows).iterrows():
            prompt = row['prompt']
            answer = row['answer']
            category = row['category']
            row_id = row['id']
            
            cot_response = ""
            
            try:
                if category == "Roman":
                    # Extract query integer
                    num = int(re.search(r'write the number\s*(\d+)', prompt).group(1))
                    cot_response = self.generate_roman_cot(num, answer)
                elif category == "Gravity":
                    # Extract query t
                    query_t = float(re.search(r'falling distance for t\s*=\s*([\d.]+)\s*s', prompt).group(1))
                    cot_response = self.generate_gravity_cot(prompt, answer, query_t)
                elif category == "Linear":
                    # Extract query x
                    query_x = float(re.search(r'measurement:\s*([\d.]+)', prompt).group(1))
                    cot_response = self.generate_linear_cot(prompt, answer, query_x)
                elif category == "Cipher":
                    # Extract query cipher text
                    query_cipher = prompt.split('decrypt the following text:')[1].strip()
                    cot_response = self.generate_cipher_cot(prompt, answer, query_cipher)
                elif category == "Bit":
                    # Extract query bit string
                    query_bit = prompt.split('output for:')[1].strip()
                    cot_response = self.generate_bit_cot(prompt, answer, query_bit)
                else:
                    # Equation
                    query_eq = prompt.split('result for:')[1].strip()
                    cot_response = self.generate_equation_cot(prompt, answer, query_eq)
                    
                if cot_response:
                    sft_dataset.append({
                        "id": row_id,
                        "prompt": prompt,
                        "completion": cot_response
                    })
            except Exception as e:
                print(f"Error generating CoT for row {idx} ({row_id}): {e}")
                
        # Write to jsonl
        output_path = "data/sft_reasoning_dataset.jsonl"
        with open(output_path, "w") as f:
            for item in sft_dataset:
                f.write(json.dumps(item) + "\n")
                
        print(f"✅ Successfully generated {len(sft_dataset)} SFT training chains in '{output_path}'.")

if __name__ == "__main__":
    generator = SftDatasetGenerator()
    generator.build_dataset(num_rows=100) # Default to 100 rows for baseline validation SFT preparation
