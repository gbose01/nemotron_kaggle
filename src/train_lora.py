import os
import yaml
import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from trl import SFTTrainer, SFTConfig

def main():
    parser = argparse.ArgumentParser(description="Supervised Fine-Tuning (SFT) with LoRA")
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
    
    print(f"🚀 Starting SFT Fine-Tuning on model: {model_name}")
    print(f"📂 Adapter checkpoints will be saved to: {output_dir}")

    # 2. Load Tokenizer & Model
    print("📥 Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure model loading precision
    torch_dtype = torch.bfloat16 if config['model']['precision'] == "bfloat16" else torch.float32
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto"
    )

    # 3. Setup LoRA Config
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
    # 4. Load Dataset
    print("📊 Loading SFT training dataset...")
    dataset = load_dataset("json", data_files="data/sft_reasoning_dataset.jsonl")

    # Format for SFTTrainer: combines prompt + completion for causal training
    def formatting_prompts_func(example):
        return f"{example['prompt']}\n\n{example['completion']}"

    # 5. Setup SFT Config
    t_cfg = config['training']
    training_args = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=t_cfg['micro_batch_size'],
        learning_rate=float(t_cfg['learning_rate']),
        lr_scheduler_type=t_cfg['lr_scheduler_type'],
        num_train_epochs=t_cfg['epochs'],
        weight_decay=t_cfg['weight_decay'],
        warmup_ratio=t_cfg['warmup_ratio'],
        logging_steps=t_cfg['logging_steps'],
        save_steps=t_cfg['save_steps'],
        bf16=(config['model']['precision'] == "bfloat16"),
        logging_dir=os.path.join(output_dir, "logs"),
        report_to="none",
        remove_unused_columns=False,
        max_length=config['model']['max_seq_length'], # set directly in SFTConfig as max_length
        completion_only_loss=False # disable to allow custom formatting_func
    )

    # 6. Initialize SFTTrainer
    print("🚂 Initializing Trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        peft_config=peft_config,
        processing_class=tokenizer,
        formatting_func=formatting_prompts_func,
        args=training_args
    )

    # 7. Start Training
    print("🔥 Starting training loop...")
    trainer.train()

    # 8. Export Checkpoint & Adapter Config
    print("📝 Exporting final fine-tuned LoRA adapter...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"🎉 SFT completed successfully! Final LoRA adapter saved at '{output_dir}'.")

if __name__ == "__main__":
    main()
