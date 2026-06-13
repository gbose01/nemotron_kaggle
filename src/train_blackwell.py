#!/usr/bin/env python3
"""Full BF16 LoRA SFT for Nemotron-3-Nano-30B on RTX PRO 6000 (Blackwell).

Mirrors the proven reference notebook recipe:
  - Load the full 30B model in bfloat16 (NO 4-bit / NO Unsloth).
  - Rank-32 LoRA over all-linear modules, alpha = r, dropout 0.0.
  - NEFTune noise (alpha=5) instead of dropout, cosine LR, 1 epoch.
  - Our curated <think> CoT dataset, formatted through the chat template.

Run AFTER restarting the kernel and AFTER the offline env is set up. Example:

  PYTHONPATH=src python3 src/train_blackwell.py \
      --config src/train_config.yaml \
      --model_name /kaggle/input/<bundle>/nemotron-base

If --model_name is omitted, the script auto-detects a local Nemotron path
(offline bundle or kagglehub download).
"""

from __future__ import annotations

import argparse
import inspect
import os
from pathlib import Path

import blackwell_env

# Apply Blackwell hardening BEFORE importing torch/transformers model code.
blackwell_env.apply()

import torch  # noqa: E402
import yaml  # noqa: E402
from datasets import load_dataset  # noqa: E402
from peft import LoraConfig, TaskType, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402


def autodetect_model_path() -> str | None:
    """Find Nemotron weights from competition Models input or offline bundle."""
    try:
        import kagglehub
        path = kagglehub.model_download("metric/nemotron-3-nano-30b-a3b-bf16/transformers/default")
        if path and Path(path).exists():
            return str(path)
    except Exception:
        pass

    for cfg in Path("/kaggle/input").rglob("config.json"):
        parent = cfg.parent
        if (parent / "model.safetensors.index.json").exists() or any(
            parent.glob("model-*.safetensors")
        ):
            return str(parent)

    for entry in sorted(Path("/kaggle/input").iterdir()):
        for candidate in (
            entry / "nemotron-base",
            entry / "nemotron-blackwell-offline" / "nemotron-base",
        ):
            if (candidate / "config.json").exists():
                return str(candidate)
    return None


def build_sft_block(config: dict) -> dict:
    """Read the `sft` block from config, falling back to notebook defaults."""
    sft = config.get("sft", {}) if isinstance(config, dict) else {}
    return {
        "lora_rank": int(sft.get("lora_rank", 32)),
        "lora_alpha": int(sft.get("lora_alpha", 32)),
        "lora_dropout": float(sft.get("lora_dropout", 0.0)),
        "neftune_alpha": float(sft.get("neftune_noise_alpha", 5)),
        "max_seq_len": int(sft.get("max_seq_length", 1536)),
        "num_epochs": float(sft.get("epochs", 1)),
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
    parser = argparse.ArgumentParser(description="Full BF16 LoRA SFT on Blackwell")
    parser.add_argument("--config", default="src/train_config.yaml")
    parser.add_argument("--model_name", default=None,
                        help="Local model path or HF id. Auto-detected if omitted.")
    parser.add_argument("--data", default="data/sft_reasoning_dataset.jsonl")
    parser.add_argument("--output_dir", default="outputs/nemotron_lora_adapter")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Override num_train_epochs; useful for quick tests")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    hp = build_sft_block(config)

    model_name = args.model_name or autodetect_model_path() \
        or config.get("model", {}).get("base_model_name")
    if not model_name:
        raise SystemExit("No model path. Pass --model_name or attach the offline bundle.")

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
    dataset = load_dataset("json", data_files=args.data)["train"]

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

    dataset = dataset.map(build_training_text, remove_columns=dataset.column_names)
    print(f"Loaded {len(dataset)} examples. Sample:\n{dataset[0]['text'][:400]}")

    # ---- Model (full BF16) ----
    print("Loading model in bfloat16 ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    n_fast = blackwell_env.disable_fast_path(model)
    print(f"Disabled fast path on {n_fast} module(s)")
    n_frozen = blackwell_env.freeze_routers(model)
    print(f"Froze {n_frozen} MoE router param(s)")

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
        save_strategy="no",
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
        train_dataset=dataset,
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
