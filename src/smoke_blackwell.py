#!/usr/bin/env python3
"""Minimal Blackwell pipeline smoke test — verify load → train → pack works.

Uses only Kaggle-mounted inputs (competition model under /kaggle/input).
Trains 2 steps on 2 inline examples with a tiny LoRA rank so it finishes fast.

  PYTHONPATH=src python3 src/smoke_blackwell.py
  PYTHONPATH=src python3 src/smoke_blackwell.py --model_name /kaggle/input/...
"""

from __future__ import annotations

import argparse
import inspect
import os
import subprocess
import sys
from pathlib import Path

import blackwell_env

blackwell_env.apply()

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from peft import LoraConfig, TaskType, get_peft_model  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

# Two tiny examples — no external data file required.
SMOKE_EXAMPLES = [
    {
        "prompt": (
            "What is 2 + 2? Put your final answer in \\boxed{}."
        ),
        "completion": (
            "<think>\n2 + 2 = 4\n</think>\n\\boxed{4}"
        ),
    },
    {
        "prompt": (
            "What is 3 + 5? Put your final answer in \\boxed{}."
        ),
        "completion": (
            "<think>\n3 + 5 = 8\n</think>\n\\boxed{8}"
        ),
    },
]


def autodetect_model_path() -> str | None:
    """Find Nemotron weights from Kaggle Models input or any attached bundle."""
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


def build_training_text(example, tokenizer) -> dict:
    user_msg = example["prompt"]
    assistant_msg = example["completion"]
    try:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg},
             {"role": "assistant", "content": assistant_msg}],
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception:
        text = (
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_msg}<|im_end|>"
        )
    return {"text": text}


def main() -> None:
    parser = argparse.ArgumentParser(description="Blackwell pipeline smoke test")
    parser.add_argument("--model_name", default=None, help="Local model path (auto-detected if omitted)")
    parser.add_argument("--output_dir", default="outputs/smoke_adapter")
    parser.add_argument("--max_steps", type=int, default=2)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--no-pack", action="store_true",
                        help="Skip building submission.zip")
    args = parser.parse_args()

    model_name = args.model_name or autodetect_model_path()
    if not model_name:
        raise SystemExit(
            "No model found under /kaggle/input. "
            "Add Input → Models → metric/nemotron-3-nano-30b-a3b-bf16"
        )

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print("=== Blackwell smoke test ===")
    print(f"Model:      {model_name}")
    print(f"Output:     {output_dir}")
    print(f"max_steps:  {args.max_steps}, lora_rank: {args.lora_rank}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.max_seq_len

    rows = [build_training_text(ex, tokenizer) for ex in SMOKE_EXAMPLES]
    dataset = Dataset.from_list(rows)
    print(f"Dataset: {len(dataset)} inline examples")
    print(dataset[0]["text"][:200], "...")

    print("Loading model (bfloat16) ...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    n_fast = blackwell_env.disable_fast_path(model)
    n_frozen = blackwell_env.freeze_routers(model)
    print(f"Disabled fast path on {n_fast} module(s); froze {n_frozen} router param(s)")
    model.gradient_checkpointing_enable()

    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            target_modules="all-linear",
            lora_dropout=0.0,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        ),
    )
    model.print_trainable_parameters()

    sft_kwargs = dict(
        output_dir=output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        bf16=True,
        fp16=False,
        optim="adamw_torch",
        neftune_noise_alpha=5,
        logging_steps=1,
        save_strategy="no",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=True,
    )
    optional = {
        "max_seq_length": args.max_seq_len,
        "dataset_text_field": "text",
        "packing": False,
    }
    valid = set(inspect.signature(SFTConfig.__init__).parameters)
    for k, v in optional.items():
        if k in valid:
            sft_kwargs[k] = v

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=SFTConfig(**sft_kwargs),
    )

    print("Training ...")
    trainer.train()
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Adapter saved to {output_dir}")

    if torch.cuda.is_available():
        print(f"Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")

    if not args.no_pack:
        zip_path = "/kaggle/working/submission.zip"
        repo_root = Path(__file__).resolve().parent.parent
        pack = repo_root / "src" / "pack_submission.py"
        verify = repo_root / "src" / "verify_submission.py"
        if pack.exists():
            subprocess.run(
                [sys.executable, str(pack),
                 "--source", output_dir,
                 "--output", zip_path],
                check=True,
            )
            if verify.exists():
                subprocess.run(
                    [sys.executable, str(verify), "--submission", zip_path],
                    check=True,
                )
            print(f"Submission ready: {zip_path}")
        else:
            print("pack_submission.py not found — skip zip (adapter dir is valid)")


if __name__ == "__main__":
    main()
