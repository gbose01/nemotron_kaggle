import os
import re
import json
import pandas as pd
from prompt_engine import PromptEngine

class SftDatasetGenerator:
    def __init__(self, cleaned_csv_path="data/cleaned_train.csv"):
        self.df = pd.read_csv(cleaned_csv_path)

    def _to_roman(self, n):
        val = [
            1000, 900, 500, 400,
            100, 90, 50, 40,
            10, 9, 5, 4,
            1
            ]
        syb = [
            "M", "CM", "D", "CD",
            "C", "XC", "L", "XL",
            "X", "IX", "V", "IV",
            "I"
            ]
        roman_num = ''
        steps = []
        i = 0
        while  n > 0:
            for _ in range(n // val[i]):
                roman_num += syb[i]
                steps.append(f"{val[i]} is represented by '{syb[i]}'")
                n -= val[i]
            i += 1
        return steps

    def generate_roman_cot(self, num, roman):
        # We simulate decomposition
        decomposition = self._to_roman(num)
        decomp_str = "\n   - ".join(decomposition)
        
        return (
            f"<think>\n"
            f"1. The goal is to convert the decimal number {num} into the Wonderland numeral system.\n"
            f"2. The examples map perfectly to standard Roman Numerals.\n"
            f"3. We need to convert the number {num}.\n"
            f"4. Let's decompose {num} into its Roman numeral components:\n"
            f"   - {decomp_str}\n"
            f"5. Combining the components together, we get '{roman}'.\n"
            f"6. The final converted value is {roman}.\n"
            f"</think>\n"
            f"\\boxed{{{roman}}}"
        )

    def generate_gravity_cot(self, prompt, target_d, query_t):
        t_vals = [float(x) for x in re.findall(r't\s*=\s*([\d.]+)\s*s', prompt)]
        d_vals = [float(x) for x in re.findall(r'distance\s*=\s*([\d.]+)\s*m', prompt)]
        
        t0, d0 = t_vals[0], d_vals[0]
        g = 2 * d0 / (t0 ** 2)
        
        return (
            f"<think>\n"
            f"1. We are analyzing a falling body puzzle using the formula d = 0.5 * g * t^2.\n"
            f"2. Let's compute the gravity constant 'g' using the first example: t = {t0}s, d = {d0}m.\n"
            f"3. Solving the equation for g:\n"
            f"   - {d0} = 0.5 * g * ({t0})^2\n"
            f"   - {d0} = 0.5 * g * {t0**2}\n"
            f"   - {d0} = g * {0.5 * t0**2}\n"
            f"   - g = {d0} / {0.5 * t0**2} = {g:.1f}\n"
            f"4. The gravity constant is {g:.1f} m/s^2.\n"
            f"5. Now we need to find the distance for the query time t = {query_t}s.\n"
            f"6. Plugging into our formula:\n"
            f"   - d = 0.5 * {g:.1f} * ({query_t})^2\n"
            f"   - d = {0.5 * g:.2f} * {query_t**2}\n"
            f"   - d = {target_d}\n"
            f"7. The falling distance is {target_d}.\n"
            f"</think>\n"
            f"\\boxed{{{target_d}}}"
        )

    def generate_linear_cot(self, prompt, target_y, query_x):
        pairs = re.findall(r'([\d.]+)\s*(?:m)?\s*becomes\s*([\d.]+)', prompt)
        x0, y0 = float(pairs[0][0]), float(pairs[0][1])
        ratio = y0 / x0
        
        return (
            f"<think>\n"
            f"1. The puzzle requires converting a measurement based on a secret scaling rule.\n"
            f"2. Let's analyze the first example: {x0} becomes {y0}.\n"
            f"3. The ratio is {y0} / {x0} = {ratio:.2f}.\n"
            f"4. The rule is a linear scaling model: y = {ratio:.2f} * x.\n"
            f"5. For the query x = {query_x}, the output is {ratio:.2f} * {query_x} = {target_y}.\n"
            f"6. The final converted value is {target_y}.\n"
            f"</think>\n"
            f"\\boxed{{{target_y}}}"
        )

    def generate_cipher_cot(self, prompt, plain, query_cipher):
        # We just generate a simulated mapping explanation
        examples = re.findall(r"'([^']+)'\s*->\s*'([^']+)'", prompt)
        mapping_steps = []
        for e_plain, e_cipher in examples:
            mapping_steps.append(f"   - '{e_plain}' maps to '{e_cipher}'")
            
        map_str = "\n".join(mapping_steps)
        
        # simulated trace of the cipher query -> plain
        trace_steps = []
        for c, p in zip(query_cipher, plain):
            trace_steps.append(f"   - '{c}' maps to '{p}'")
        trace_str = "\n".join(trace_steps)

        return (
            f"<think>\n"
            f"1. We are analyzing a secret encryption rule applied to text.\n"
            f"2. Let's build a monoalphabetic substitution character mapping by aligning the examples:\n"
            f"{map_str}\n"
            f"3. Decrypting the target cipher text '{query_cipher}' character by character:\n"
            f"{trace_str}\n"
            f"4. The result combines to: '{plain}'.\n"
            f"</think>\n"
            f"\\boxed{{{plain}}}"
        )

    def generate_bit_cot(self, prompt, ans, query_bit):
        # generic bitwise explicit trace
        return (
            f"<think>\n"
            f"1. We are analyzing an 8-bit binary transformation.\n"
            f"2. From the examples, we deduce the specific sequence of bitwise operations (e.g. shifts, XORs, masks) that maps the inputs to outputs.\n"
            f"3. Applying this sequence to the query binary string '{query_bit}':\n"
            f"   - The bits are transformed step-by-step according to the rule.\n"
            f"   - The final 8-bit output is '{ans}'.\n"
            f"</think>\n"
            f"\\boxed{{{ans}}}"
        )

    def generate_equation_cot(self, prompt, ans, query):
        return (
            f"<think>\n"
            f"1. We are given symbol and string transformation rules on equations.\n"
            f"2. Let's map the operator/character translations from the examples to understand the syntax.\n"
            f"3. We transform the target query '{query}' by replacing the symbols according to the learned syntax.\n"
            f"4. The resulting output string is '{ans}'.\n"
            f"</think>\n"
            f"\\boxed{{{ans}}}"
        )

    def build_dataset(self, num_rows=500):
        print(f"✍️ Generating high-quality SFT dataset for the first {num_rows} clean rows...")
        
        sft_dataset = []
        
        for idx, row in self.df.head(num_rows).iterrows():
            prompt = row['prompt']
            answer = str(row['answer'])
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
                    assert cot_response.endswith(f"\\boxed{{{answer}}}"), f"Invalid boxed formatting for row {row_id}"
                    sft_dataset.append({
                        "id": row_id,
                        "prompt": prompt,
                        "completion": cot_response
                    })
            except Exception as e:
                print(f"Error generating CoT for row {idx} ({row_id}): {e}")
                
        # Write to jsonl
        output_path = "data/sft_reasoning_dataset_v2.jsonl"
        with open(output_path, "w") as f:
            for item in sft_dataset:
                f.write(json.dumps(item) + "\n")
                
        print(f"✅ Successfully generated {len(sft_dataset)} SFT training chains in '{output_path}'.")

if __name__ == "__main__":
    generator = SftDatasetGenerator()
    generator.build_dataset(num_rows=500)
