Here are the **exact code snippets** you need to copy-paste into the cells of your Kaggle Notebook. 

I have structured them into 5 easy-to-run cells.

---

### 📦 Cell 1: Install ML Libraries
*Paste this in the first cell and run it. It installs the training libraries on the Kaggle server.*

```python
!pip install torch transformers peft trl datasets accelerate bitsandbytes --index-url https://pypi.org/simple
```

---

### 📁 Cell 2: Write the Training Configuration
*Paste this in the second cell and run it. It automatically creates our directory structure and writes the configuration parameters optimized for Kaggle T4 GPUs.*

```python
import os
os.makedirs("src", exist_ok=True)
os.makedirs("data", exist_ok=True)

config_content = """
model:
  base_model_name: "NVIDIA/Nemotron-3-Nano-30B-Base"
  precision: "bfloat16"
  max_seq_length: 4096
  max_gen_tokens: 2048

lora:
  r: 32
  alpha: 64
  dropout: 0.05
  bias: "none"
  target_modules:
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"

training:
  global_batch_size: 128
  micro_batch_size: 1
  learning_rate: 3e-6
  lr_scheduler_type: "constant"
  epochs: 3
  weight_decay: 0.01
  warmup_ratio: 0.03
  logging_steps: 1
  save_steps: 50
  output_dir: "outputs/nemotron_lora_adapter"
"""

with open("src/train_config.yaml", "w") as f:
    f.write(config_content.strip())
print("✅ SFT Config successfully written to 'src/train_config.yaml'!")
```

---

### 🧼 Cell 3: Clean the Dataset & Compile the SFT Chains
*Paste this in the third cell and run it. This loads the competition's raw data directly from Kaggle's filesystem, isolates the buggy examples, and generates our high-fidelity reasoning chains on the fly!*

```python
import re
import json
import random
import pandas as pd
from src.prompt_engine import PromptEngine # We define this below or import it

# Define the exact CoT generators
class SftDatasetGenerator:
    def generate_roman_cot(self, num, roman):
        return f"<think>\\n1. The goal is
<truncated 7667 bytes>
,
    num_train_epochs=config['training']['epochs'],
    weight_decay=config['training']['weight_decay'],
    warmup_ratio=config['training']['warmup_ratio'],
    logging_steps=config['training']['logging_steps'],
    save_steps=config['training']['save_steps'],
    bf16=True,
    logging_dir=os.path.join(output_dir, "logs"),
    report_to="none",
    remove_unused_columns=False,
    max_length=config['model']['max_seq_length'],
    completion_only_loss=False
)

print("🚂 Initializing SFT Trainer & starting fine-tuning...")
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['train'],
    peft_config=peft_config,
    processing_class=tokenizer,
    formatting_func=formatting_prompts_func,
    args=training_args
)

trainer.train()

print("📝 Saving fine-tuned LoRA adapter adapter...")
trainer.model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("🎉 Fine-tuning completed successfully!")
```

---

### 📦 Cell 5: Packaging the Submission ZIP
*Paste this in the fifth cell and run it. This compiles your submission folder into the strictly compliant Kaggle `submission.zip` format, ready to be downloaded and submitted!*

```python
import zipfile
import os

source_dir = "outputs/nemotron_lora_adapter"
output_zip = "submission.zip"

print(f"⏳ Packaging adapter files into '{output_zip}'...")
with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
    # Archive files strictly at the root level
    zipf.write(os.path.join(source_dir, "adapter_config.json"), arcname="adapter_config.json")
    if os.path.exists(os.path.join(source_dir, "adapter_model.safetensors")):
        zipf.write(os.path.join(source_dir, "adapter_model.safetensors"), arcname="adapter_model.safetensors")
    else:
        zipf.write(os.path.join(source_dir, "adapter_model.bin"), arcname="adapter_model.bin")

size_mb = os.path.getsize(output_zip) / (1024 * 1024)
print(f"🎉 SUCCESS: '{output_zip}' successfully packaged! ({size_mb:.2f} MB)")
print("Ready for download and submission on Kaggle!")
```