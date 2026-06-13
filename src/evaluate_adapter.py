#!/usr/bin/env python3
"""Evaluate a trained LoRA adapter locally on a stratified train.csv split.

Loads the base Nemotron model + the trained adapter, generates answers in the
same chat format used for training, extracts ``\\boxed{}`` with the exact
competition-style parser, and reports per-puzzle-type accuracy. This is the
signal to iterate on -- NOT training loss.

Run on the Blackwell notebook after training:

  PYTHONPATH=src python3 src/evaluate_adapter.py \
      --model_name /kaggle/input/<bundle>/nemotron-base \
      --adapter outputs/nemotron_lora_adapter \
      --train_csv /kaggle/input/nvidia-nemotron-3-reasoning-challenge/train.csv \
      --n_per_type 10
"""

from __future__ import annotations

import argparse
import re

import blackwell_env

blackwell_env.apply()

import pandas as pd  # noqa: E402
import torch  # noqa: E402
from peft import PeftModel  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from consensus_inference import extract_boxed_answer  # noqa: E402


def classify(prompt: str) -> str:
    p = prompt.lower()
    if "gravitational constant" in p:
        return "Gravity"
    if "becomes" in p:
        return "Linear"
    if "numeral system" in p:
        return "Roman"
    if "decrypt the following text" in p:
        return "Cipher"
    if "bit manipulation" in p:
        return "Bit"
    return "Equation"


def answers_match(pred: str, target: str) -> bool:
    pred_c = re.sub(r"\s+", " ", str(pred).strip().lower())
    tgt_c = re.sub(r"\s+", " ", str(target).strip().lower())
    if pred_c == tgt_c:
        return True
    try:
        pv, tv = float(pred_c), float(tgt_c)
        return abs(pv - tv) <= 1e-2 or abs((pv - tv) / (tv + 1e-9)) <= 1e-2
    except ValueError:
        return False


def stratified_sample(df: pd.DataFrame, n_per_type: int, seed: int) -> pd.DataFrame:
    df = df.copy()
    df["__type"] = df["prompt"].map(classify)
    parts = []
    for _, group in df.groupby("__type"):
        parts.append(group.sample(n=min(n_per_type, len(group)), random_state=seed))
    return pd.concat(parts).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True)
    ap.add_argument("--adapter", default="outputs/nemotron_lora_adapter")
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--n_per_type", type=int, default=10)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    blackwell_env.disable_fast_path(model)
    print("Attaching adapter ...")
    model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    df = pd.read_csv(args.train_csv)
    val = stratified_sample(df, args.n_per_type, args.seed)
    print(f"Evaluating {len(val)} examples across {val['__type'].nunique()} types")

    rows = []
    for _, row in val.iterrows():
        user_msg = (
            f"{row['prompt']}\n\n"
            "Solve this step by step. Show your reasoning, then put your final "
            "answer in \\boxed{}."
        )
        inputs = tokenizer.apply_chat_template(
            [{"role": "user", "content": user_msg}],
            add_generation_prompt=True, return_tensors="pt",
        ).to(model.device)
        with torch.no_grad():
            out = model.generate(
                inputs, max_new_tokens=args.max_new_tokens,
                do_sample=False, pad_token_id=tokenizer.pad_token_id,
            )
        gen = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
        pred = extract_boxed_answer(gen)
        ok = answers_match(pred, row["answer"])
        rows.append({"id": row["id"], "type": row["__type"],
                     "target": row["answer"], "pred": pred, "correct": ok})
        print(f"[{row['__type']:8s}] {'OK ' if ok else 'XX '} "
              f"target={row['answer']!r} pred={pred!r}")

    res = pd.DataFrame(rows)
    print("\n=== Per-type accuracy ===")
    for t, g in res.groupby("type"):
        print(f"  {t:10s} {g['correct'].mean()*100:5.1f}%  ({g['correct'].sum()}/{len(g)})")
    print(f"\nOverall: {res['correct'].mean()*100:.2f}%  "
          f"({res['correct'].sum()}/{len(res)})")
    res.to_csv("adapter_eval_results.csv", index=False)
    print("Saved adapter_eval_results.csv")


if __name__ == "__main__":
    main()
