import random
import re
import json
from generate_sft_dataset import SftDatasetGenerator

# Standard Roman numeral conversion helpers
def int_to_roman(num):
    val = [100, 90, 50, 40, 10, 9, 5, 4, 1]
    syb = ["C", "XC", "L", "XL", "X", "IX", "V", "IV", "I"]
    roman_num = ''
    i = 0
    while num > 0:
        for _ in range(num // val[i]):
            roman_num += syb[i]
            num -= val[i]
        i += 1
    return roman_num

class SyntheticGenerator:
    def __init__(self):
        self.sentences = [
            "the golden dragon dreams inside castle",
            "the secret wizard explores the mystical forest",
            "the brave knight protects the ancient queen",
            "student discovers near magical valley",
            "the clever mouse chases a fast rabbit",
            "princess reads the mysterious library book",
            "wise bird watches from deep cave",
            "turtle watches the gorgeous garden"
        ]

    def generate_roman_puzzle(self):
        ex_nums = random.sample(range(1, 99), 5)
        query_num = random.choice([n for n in range(1, 99) if n not in ex_nums])
        
        prompt = "In Alice's Wonderland, numbers are secretly converted into a different numeral system. Some examples are given below:\n"
        for n in ex_nums:
            prompt += f"{n} -> {int_to_roman(n)}\n"
        prompt += f"Now, write the number {query_num} in the Wonderland numeral system."
        
        answer = int_to_roman(query_num)
        
        # Create corresponding CoT
        sft = SftDatasetGenerator()
        cot = sft.generate_roman_cot(query_num, answer)
        return prompt, answer, cot

    def generate_linear_puzzle(self):
        a = round(random.uniform(0.5, 2.5), 2)
        b = round(random.uniform(-10, 10), 2)
        
        ex_x = [round(random.uniform(5, 50), 2) for _ in range(4)]
        query_x = round(random.uniform(5, 50), 2)
        
        prompt = "In Alice's Wonderland, a secret unit conversion is applied to measurements. For example:\n"
        for x in ex_x:
            y = round(a * x + b, 2)
            prompt += f"{x} m becomes {y:.2f}\n"
        prompt += f"Now, convert the following measurement: {query_x} m"
        
        answer = f"{round(a * query_x + b, 2):.2f}"
        
        sft = SftDatasetGenerator()
        cot = sft.generate_linear_cot(prompt, answer, query_x)
        return prompt, answer, cot

    def generate_gravity_puzzle(self):
        g = round(random.uniform(4.5, 19.5), 2)
        ex_t = [round(random.uniform(1.0, 5.0), 2) for _ in range(4)]
        query_t = round(random.uniform(1.0, 5.0), 2)
        
        prompt = "In Alice's Wonderland, the gravitational constant has been secretly changed. Here are some example observations:\n"
        for t in ex_t:
            d = round(0.5 * g * (t ** 2), 2)
            prompt += f"For t = {t}s, distance = {d:.2f} m\n"
        prompt += f"Now, determine the falling distance for t = {query_t}s given d = 0.5*g*t^2."
        
        answer = f"{round(0.5 * g * (query_t ** 2), 2):.2f}"
        
        sft = SftDatasetGenerator()
        cot = sft.generate_gravity_cot(prompt, answer, query_t)
        return prompt, answer, cot

    def generate_cipher_puzzle(self):
        # Create a random shift Caesar cipher
        shift = random.randint(1, 25)
        
        def encrypt(text):
            res = []
            for char in text:
                if char.isalpha():
                    base = ord('a')
                    res.append(chr((ord(char) - base + shift) % 26 + base))
                else:
                    res.append(char)
            return "".join(res)
            
        ex_sentences = random.sample(self.sentences, 4)
        query_sentence = random.choice([s for s in self.sentences if s not in ex_sentences])
        
        prompt = "In Alice's Wonderland, secret encryption rules are used on text. Here are some examples:\n"
        for s in ex_sentences:
            prompt += f"{encrypt(s)} -> {s}\n"
        prompt += f"Now, decrypt the following text: {encrypt(query_sentence)}"
        
        answer = query_sentence
        
        sft = SftDatasetGenerator()
        cot = sft.generate_cipher_cot(prompt, answer, encrypt(query_sentence))
        return prompt, answer, cot

    def generate_batch(self, size=50):
        print(f"🧬 Generating {size} high-quality synthetic puzzles (data augmentation)...")
        synthetic_puzzles = []
        
        categories = ["Roman", "Linear", "Gravity", "Cipher"]
        for i in range(size):
            cat = random.choice(categories)
            row_id = f"syn_{i:05d}"
            
            if cat == "Roman":
                p, ans, cot = self.generate_roman_puzzle()
            elif cat == "Linear":
                p, ans, cot = self.generate_linear_puzzle()
            elif cat == "Gravity":
                p, ans, cot = self.generate_gravity_puzzle()
            else:
                p, ans, cot = self.generate_cipher_puzzle()
                
            synthetic_puzzles.append({
                "id": row_id,
                "prompt": p,
                "completion": cot
            })
            
        # Append to sft_reasoning_dataset.jsonl
        output_path = "data/sft_reasoning_dataset.jsonl"
        with open(output_path, "a") as f:
            for item in synthetic_puzzles:
                f.write(json.dumps(item) + "\n")
                
        print(f"🧬 Successfully appended {len(synthetic_puzzles)} synthetic puzzles to '{output_path}'.")

if __name__ == "__main__":
    generator = SyntheticGenerator()
    generator.generate_batch(size=50) # Generate 50 custom logical puzzles
