#!/usr/bin/env python3
"""Full BF16 LoRA SFT for Nemotron-3-Nano-30B on RunPod (A100/RTX6000).

Mirrors the proven reference notebook recipe:
  - Load the full 30B model in bfloat16 (NO 4-bit / NO Unsloth).
  - Rank-32 LoRA over all-linear modules, alpha = r, dropout 0.0.
  - NEFTune noise (alpha=5) instead of dropout, cosine LR, 2 epochs.
  - Our curated <think> CoT dataset, formatted through the chat template.

Run with:
  HF_HOME=/workspace/hf_cache PYTHONPATH=src python3 src/train_runpod.py \
      --config src/train_config.yaml \
      --model_name nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16
"""

from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def build_sft_block(config: dict) -> dict:
    """Read the `sft` block from config, falling back to RunPod defaults."""
    sft = config.get("sft", {}) if isinstance(config, dict) else {}
    return {
        "lora_rank": int(sft.get("lora_rank", 32)),
        "lora_alpha": int(sft.get("lora_alpha", 32)),
        "lora_dropout": float(sft.get("lora_dropout", 0.0)),
        "neftune_alpha": float(sft.get("neftune_noise_alpha", 5)),
        "max_seq_len": int(sft.get("max_seq_length", 2048)), # Increased to 2048 for full CoT
        "num_epochs": float(sft.get("epochs", 2)), # Increased to 2 for larger dataset
        "batch_size": int(sft.get("batch_size", 1)),
        "grad_accum": int(sft.get("grad_accum", 4)),
        "lr": float(sft.get("learning_rate", 2e-4)),
        "warmup_ratio": float(sft.get("warmup_ratio", 0.1)),
        "weight_decay": float(sft.get("weight_decay", 0.01)),
        "lr_scheduler": sft.get("lr_scheduler_type", "cosine"),
        "grad_ckpt": bool(sft.get("gradient_checkpointing", True)),
        "logging_steps": int(sft.get("logging_steps", 10)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Full BF16 LoRA SFT on RunPod")
    parser.add_argument("--config", default="src/train_config.yaml")
    parser.add_argument("--model_name", default="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16",
                        help="Local model path or HF id.")
    parser.add_argument("--data", default="data/sft_reasoning_dataset_v2.jsonl")
    parser.add_argument("--output_dir", default="outputs/nemotron_lora_adapter")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override num_train_epochs; useful for quick tests")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    hp = build_sft_block(config)

    model_name = args.model_name
    if not model_name:
        raise SystemExit("No model path. Pass --model_name.")

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"Model:   {model_name}")
    print(f"Data:    {args.data}")
    print(f"Output:  {output_dir}")
    print(f"Hyperparams: {hp}")

    # ---- Tokenizer ----
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = hp["max_seq_len"]

    # ---- Dataset: format prompt/completion via the chat template ----
    # Implementing 90/10 split
    dataset = load_dataset("json", data_files=args.data)["train"]
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    
    train_dataset = split_dataset["train"]
    val_dataset = split_dataset["test"]

    def build_training_text(example):
        user_msg = (
            f"{example['prompt']}\n\n"
            "Solve this step by step. Show your reasoning, then put your final "
            "answer in \\boxed{}."
        )
        assistant_msg = example["completion"]
        try:
            text = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_msg},
                 {"role": "assistant", "content": assistant_msg}],
                tokenize=False, add_generation_prompt=False,
            )
        except Exception:
            text = (
                f"<|im_start|>user\n{user_msg}<|im_end|>\n"
                f"<|im_start|>assistant\n{assistant_msg}<|im_end|>"
            )
        return {"text": text}

    train_dataset = train_dataset.map(build_training_text, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(build_training_text, remove_columns=val_dataset.column_names)
    
    print(f"Loaded {len(train_dataset)} training and {len(val_dataset)} validation examples.")
    print(f"Sample:\n{train_dataset[0]['text'][:400]}")

    # ---- Model (full BF16) ----
    print("Loading model in bfloat16 ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    if hp["grad_ckpt"]:
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=hp["lora_rank"],
        lora_alpha=hp["lora_alpha"],
        target_modules="all-linear",
        lora_dropout=hp["lora_dropout"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---- SFTConfig (guard for TRL version differences) ----
    sft_kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=hp["num_epochs"] if args.max_steps is None else 1,
        per_device_train_batch_size=hp["batch_size"],
        gradient_accumulation_steps=hp["grad_accum"],
        learning_rate=hp["lr"],
        lr_scheduler_type=hp["lr_scheduler"],
        warmup_ratio=hp["warmup_ratio"],
        bf16=True,
        fp16=False,
        optim="adamw_torch",
        weight_decay=hp["weight_decay"],
        max_grad_norm=1.0,
        neftune_noise_alpha=hp["neftune_alpha"],
        logging_steps=hp["logging_steps"],
        eval_strategy="epoch",
        save_strategy="epoch",
        gradient_checkpointing=hp["grad_ckpt"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=True,
    )
    optional = {
        "max_seq_length": hp["max_seq_len"],
        "dataset_text_field": "text",
        "packing": False,
    }
    valid = set(inspect.signature(SFTConfig.__init__).parameters)
    for k, v in optional.items():
        if k in valid:
            sft_kwargs[k] = v
    if args.max_steps is not None:
        sft_kwargs["max_steps"] = args.max_steps

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=SFTConfig(**sft_kwargs),
    )

    print("Starting training ...")
    trainer.train()
    print("Training complete.")

    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Adapter + tokenizer saved to {output_dir}")

    if torch.cuda.is_available():
        print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
