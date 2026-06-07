import os
import yaml
import argparse
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

def main():
    # 0. Blackwell/Triton Fix (Environment Check)
    if os.path.exists('/opt/nvidia/ptxas-blackwell') and "TRITON_PTXAS_PATH" not in os.environ:
        print("⚙️ Applying late-bind Triton ptxas fix for Blackwell...")
        os.makedirs('/tmp/bin', exist_ok=True)
        import shutil
        shutil.copy('/opt/nvidia/ptxas-blackwell', '/tmp/bin/ptxas')
        os.chmod('/tmp/bin/ptxas', 0o755)
        os.environ["TRITON_PTXAS_PATH"] = "/tmp/bin/ptxas"

    parser = argparse.ArgumentParser(description="Unsloth SFT for Blackwell 96GB GPUs")
    parser.add_argument("--config", type=str, default="src/train_config.yaml", help="Path to config YAML")
    parser.add_argument("--model_name", type=str, default=None, help="Override base model name")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    args = parser.parse_args()

    # 1. Load Configuration
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found at {args.config}")
        
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model_name = args.model_name or config['model']['base_model_name']
    output_dir = args.output_dir or config['training']['output_dir']
    max_seq_length = config['model']['max_seq_length']
    
    print(f"🦥 Starting Unsloth SFT on model: {model_name}")
    print(f"🚀 Optimized for Blackwell GPUs (96GB VRAM)")

    # 2. Load Model and Tokenizer via Unsloth
    # With 96GB VRAM, we can load in 4-bit for speed or 16-bit for quality.
    # We default to 4-bit as per Unsloth's recommendation for fast training.
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = None, # Auto detect (BF16 for Blackwell)
        load_in_4bit = True,
        trust_remote_code = True,
    )

    # 3. Add LoRA Adapters
    model = FastLanguageModel.get_peft_model(
        model,
        r = config['lora']['r'],
        target_modules = config['lora']['target_modules'],
        lora_alpha = config['lora']['alpha'],
        lora_dropout = 0, # Unsloth optimized LoRA works best with 0 dropout
        bias = "none",
        use_gradient_checkpointing = "unsloth", # Use Unsloth's optimized checkpointing
        random_state = 3407,
    )

    # 4. Load Dataset
    print("📊 Loading SFT training dataset...")
    dataset = load_dataset("json", data_files="data/sft_reasoning_dataset.jsonl")

    # Format prompt + completion (matches the reasoning format in our jsonl)
    def formatting_prompts_func(example):
        return f"{example['prompt']}\n\n{example['completion']}"

    # 5. Setup SFT Config
    t_cfg = config['training']
    training_args = SFTConfig(
        output_dir = output_dir,
        per_device_train_batch_size = 4, # Larger batch size possible on Blackwell
        gradient_accumulation_steps = 4,
        learning_rate = float(t_cfg['learning_rate']),
        lr_scheduler_type = t_cfg['lr_scheduler_type'],
        num_train_epochs = t_cfg['epochs'],
        weight_decay = t_cfg['weight_decay'],
        warmup_ratio = t_cfg['warmup_ratio'],
        logging_steps = t_cfg['logging_steps'],
        save_steps = t_cfg['save_steps'],
        bf16 = True, # Use BF16 for Blackwell!
        logging_dir = os.path.join(output_dir, "logs"),
        report_to = "none",
        max_length = max_seq_length,
        dataset_text_field = "text",
    )

    # 6. Initialize SFTTrainer
    trainer = SFTTrainer(
        model = model,
        train_dataset = dataset['train'],
        tokenizer = tokenizer,
        formatting_func = formatting_prompts_func,
        args = training_args,
    )

    # 7. Start Training
    print("🔥 Starting Unsloth SFT training loop...")
    trainer.train()

    # 8. Export Final Model
    print("📝 Exporting final LoRA adapter...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"🎉 Unsloth SFT completed! Saved at '{output_dir}'.")

if __name__ == "__main__":
    main()
