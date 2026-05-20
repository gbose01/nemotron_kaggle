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
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
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

    # 4. Custom Model Local Registration (Bypasses HF Sandbox Isolation completely!)
    import shutil
    import json
    import sys
    
    model_path = os.path.abspath(model_name)
    if os.path.isdir(model_path):
        print("⚙️ Copying custom modeling files to bypass HuggingFace sandboxing...")
        try:
            # Ensure the 'src' folder is explicitly in Python's search path
            src_dir = os.path.abspath("src")
            if src_dir not in sys.path:
                sys.path.append(src_dir)
                
            # Copy files into the 'src' folder so they can be imported as standard local modules
            shutil.copy(os.path.join(model_path, "modeling_nemotron_h.py"), os.path.join(src_dir, "modeling_nemotron_h.py"))
            shutil.copy(os.path.join(model_path, "configuration_nemotron_h.py"), os.path.join(src_dir, "configuration_nemotron_h.py"))
            
            # Import local classes directly
            from configuration_nemotron_h import NemotronHConfig
            from modeling_nemotron_h import NemotronHModel, NemotronHForCausalLM
            
            # Statically patch sharding layout rules on both local classes
            NemotronHModel._no_split_modules = ["NemotronHBlock"]
            NemotronHForCausalLM._no_split_modules = ["NemotronHBlock"]
            print("⚙️ Statically Patched Local Classes: _no_split_modules = ['NemotronHBlock']")
            
            # Extract custom model type from config
            with open(os.path.join(model_path, "config.json"), "r") as f:
                model_type = json.load(f).get("model_type", "nemotron_h")
                
            # Register custom config and model in HuggingFace Auto classes
            AutoConfig.register(model_type, NemotronHConfig)
            AutoModelForCausalLM.register(NemotronHConfig, NemotronHForCausalLM)
            print(f"⚙️ Registered custom model type '{model_type}' in HuggingFace registries!")
        except Exception as e:
            print(f"⚠️ Could not execute local model registration: {e}")

    # 5. Load Model with Quantization (Loads natively without trust_remote_code!)
    print("📥 Loading model sharded across GPUs in 4-bit with custom max_memory rules...")
    # Limit weight loading on GPU 0 to leave plenty of headroom for activations and gradients
    max_memory = {0: "9GiB", 1: "14GiB"}
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",  # Auto shards model weights across T4 GPU #1 and T4 GPU #2
        max_memory=max_memory,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True
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
