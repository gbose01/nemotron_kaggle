import os
import yaml
import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoConfig
from peft import LoraConfig, TaskType, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

def main():
    parser = argparse.ArgumentParser(description="QLoRA 4-bit SFT for Kaggle Dual T4 x2 GPUs")
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
    
    print(f"🚀 Starting SFT QLoRA 4-bit Fine-Tuning on model: {model_name}")
    print(f"📂 Outputs will be saved to: {output_dir}")

    # 2. Load Tokenizer
    print("📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 🚀 RAM Safety: Clear memory before heavy model loading
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    # 3. Configure 4-bit Quantization (QLoRA) - FP16 for T4 GPUs
    print("⚙️ Configuring 4-bit BitsAndBytes Quantization (NF4 + FP16 compute)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16  # Native FP16 for T4 (no native BF16!)
    )

    # 4. Dynamic Class Patching for Multi-GPU Sharding (CRITICAL for Kaggle 2x T4!)
    # Custom architectures require explicit block definitions so accelerate can shard them layer-by-layer.
    # We scan sys.modules at runtime so it patches whatever transformers package is loaded (global or utility script).
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    print("⚙️ Triggering dynamic module registration for sharding rules...")
    try:
        # Load the class once into sys.modules memory
        _ = get_class_from_dynamic_module("modeling_nemotron_h.NemotronHForCausalLM", model_name)
        
        # Scan and inject _no_split_modules directly into the live registered classes
        import sys
        patched_count = 0
        for mod_name, module in list(sys.modules.items()):
            if mod_name.endswith("modeling_nemotron_h") and module is not None:
                if hasattr(module, "NemotronHModel"):
                    getattr(module, "NemotronHModel")._no_split_modules = ["NemotronHBlock"]
                    print(f"⚙️ Dynamic Patch: Set {mod_name}.NemotronHModel._no_split_modules = ['NemotronHBlock']")
                    patched_count += 1
                if hasattr(module, "NemotronHForCausalLM"):
                    getattr(module, "NemotronHForCausalLM")._no_split_modules = ["NemotronHBlock"]
                    print(f"⚙️ Dynamic Patch: Set {mod_name}.NemotronHForCausalLM._no_split_modules = ['NemotronHBlock']")
                    patched_count += 1
        print(f"⚙️ Applied {patched_count} sharding patches in sys.modules.")
    except Exception as e:
        print(f"⚠️ Skip dynamic sharding patch (single-GPU/CPU run): {e}")

    # 5. Load Model with Quantization & Custom Max Memory Budget
    print("📥 Loading model sharded across T4 GPUs in 4-bit (FP16 storage)...")
    # Limit weight loading on GPU 0 to leave plenty of headroom for activations/gradients
    # We use a slightly more conservative GPU 0 limit to prevent OOM on the primary device
    max_memory = {0: "5GiB", 1: "13GiB"}
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",          # Auto shards model weights across GPU #0 and GPU #1
        max_memory=max_memory,
        torch_dtype=torch.float16,  # FP16 storage for stability on T4
        low_cpu_mem_usage=True,
        trust_remote_code=True
    )
    
    # 🚀 RAM Safety: Clear memory after loading weights
    gc.collect()
    torch.cuda.empty_cache()

    # 6. Prepare model for k-bit (4-bit) gradient checkpoint training
    print("🔧 Preparing model for 4-bit training...")
    model = prepare_model_for_kbit_training(model)

    # 7. Setup LoRA Config
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

    # 8. Load Dataset
    print("📊 Loading SFT training dataset...")
    dataset = load_dataset("json", data_files="data/sft_reasoning_dataset.jsonl")

    # Format prompt + completion
    def formatting_prompts_func(example):
        return f"{example['prompt']}\n\n{example['completion']}"

    # 9. Setup SFT Config (Optimized for T4 VRAM and gradient accumulation)
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
        fp16=True,                               # Native FP16 compute for T4
        gradient_checkpointing=True,             # Save VRAM
        logging_dir=os.path.join(output_dir, "logs"),
        report_to="none",
        remove_unused_columns=False,
        max_length=config['model']['max_seq_length'],
        completion_only_loss=False
    )

    # 10. Initialize SFTTrainer
    print("🚂 Initializing Trainer...")
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        peft_config=peft_config,
        processing_class=tokenizer,
        formatting_func=formatting_prompts_func,
        args=training_args
    )

    # 11. Start Training
    print("🔥 Starting 4-bit QLoRA SFT training loop on Kaggle T4 GPUs...")
    trainer.train()

    # 12. Export Checkpoint & Adapter Config
    print("📝 Exporting final fine-tuned LoRA adapter...")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"🎉 QLoRA SFT completed successfully! Final LoRA adapter saved at '{output_dir}'.")

if __name__ == "__main__":
    main()
