import os
import yaml
import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

def main():
    parser = argparse.ArgumentParser(description="QLoRA 4-bit SFT for Kaggle 2x T4 GPUs")
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
    
    print(f"🚀 Starting QLoRA 4-bit SFT Fine-Tuning on model: {model_name}")
    print(f"📂 Outputs will be saved to: {output_dir}")

    # 2. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Configure 4-bit Quantization (QLoRA)
    # NF4 quantization fits the 30B model weights in ~15GB VRAM, allowing training on free GPUs!
    print("⚙️ Configuring 4-bit BitsAndBytes Quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    # 4. Load Model with Quantization
    print("📥 Loading model sharded across GPUs in 4-bit...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto"  # Auto shards model weights across T4 GPU #1 and T4 GPU #2
    )

    # 5. Prepare model for k-bit (4-bit) gradient checkpoint training
    print("🔧 Preparing model for 4-bit training...")
    model = prepare_model_for_kbit_training(model)

    # 6. Setup LoRA Config
    print("🔧 Wrapping model with PEFT/LoRA Adapter...")
    lora_cfg = config['lora']
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_cfg['r'],
        lora_alpha=lora_cfg['alpha'],
        lora_dropout=lora_cfg['dropout'],
        bias=lora_cfg['bias'],
        target_modules=lora_cfg['target_modules']
    )
    
    # SFTTrainer in newer versions will handle get_peft_model automatically

    # 7. Load Dataset
    print("📊 Loading SFT training dataset...")
    dataset = load_dataset("json", data_files="data/sft_reasoning_dataset.jsonl")

    # Format prompt + completion
    def formatting_prompts_func(example):
        return f"{example['prompt']}\n\n{example['completion']}"

    # 8. Setup SFT Config (Optimized for T4 VRAM and gradient accumulation)
    t_cfg = config['training']
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=1,            # Micro-batch 1 fits T4 GPU
        gradient_accumulation_steps=16,          # Simulates larger batch size
        learning_rate=float(t_cfg['learning_rate']),
        lr_scheduler_type=t_cfg['lr_scheduler_type'],
        num_train_epochs=t_cfg['epochs'],
        weight_decay=t_cfg['weight_decay'],
        warmup_ratio=t_cfg['warmup_ratio'],
        logging_steps=t_cfg['logging_steps'],
        save_steps=t_cfg['save_steps'],
        bf16=True,                               # Uses highly stable bfloat16 compute
        logging_dir=os.path.join(output_dir, "logs"),
        report_to="none",
        remove_unused_columns=False,
        max_length=config['model']['max_seq_length'],
        completion_only_loss=False
    )

    # 9. Initialize SFTTrainer
    print("🚂 Initializing Trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        peft_config=peft_config,
        processing_class=tokenizer,
        formatting_func=formatting_prompts_func,
        args=training_args
    )

    # 10. Start Training
    print("🔥 Starting 4-bit QLoRA training loop on Kaggle GPUs...")
    trainer.train()

    # 11. Export Checkpoint & Adapter Config
    print("📝 Exporting final fine-tuned LoRA adapter...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"🎉 QLoRA SFT completed successfully! Final LoRA adapter saved at '{output_dir}'.")

if __name__ == "__main__":
    main()
