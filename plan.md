# NVIDIA Nemotron Reasoning Challenge — From-Scratch Plan (v2)

This plan rebuilds our approach from the ground up, keeping **our own generated SFT
dataset** (`data/sft_reasoning_dataset.jsonl`) but adopting the proven training recipe
from the reference Blackwell notebook that successfully trained the full 30B model in
~1 hour (loss 11.2 → ~2.8, peak 75 GB VRAM, valid `submission.zip`).

The core strategic shift: **stop fighting the hardware with QLoRA/Unsloth, and instead
load the full BF16 model on the Blackwell GPU exactly the way the working notebook does.**

---

## 0. Competition Constraints (non-negotiable)

- Base model: `NVIDIA Nemotron-3-Nano-30B-A3B` (hybrid Mamba-Transformer sparse MoE).
- Submission: `submission.zip` containing a **LoRA adapter** (`adapter_config.json` +
  `adapter_model.safetensors`) at the **zip root** (no nested folders).
- LoRA rank `r <= 32`. Must load under vLLM at scoring time.
- Answers must be wrapped in `\boxed{...}`; scoring uses exact string match or numeric
  tolerance of `1e-2`.
- Max gen tokens 7680, max seq length 8192.
- Dates: entry/merge **June 8, 2026**; competition ends **June 15, 2026**.

---

## 1. What We Learned From the Reference Notebook

1. **Full BF16 beats QLoRA on Blackwell.** The RTX PRO 6000 (96 GB) loads the full 30B
   in bf16 with room for a rank-32 `all-linear` adapter (883M trainable, 2.7%). No
   bitsandbytes, no Unsloth — both add Blackwell-specific breakage.
2. **The environment hardening is the hard part, not the training.** The notebook spends
   most of its code on Blackwell/Triton/CUTLASS workarounds. These are mandatory:
   - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   - Pure-PyTorch `rmsnorm_fn` monkeypatch over loaded modules.
   - Mock `cutlass.*` / `quack.*` modules pre-registered in `sys.modules` + a meta-path
     finder, so Nemotron custom code imports cleanly without the CUDA kernels.
   - `is_fast_path_available = False` on every submodule (forces stable path).
   - Full Triton `ptxas-blackwell` fix: copy whole `bin/` dir, set
     `TRITON_PTXAS_PATH` + `TRITON_PTXAS_BLACKWELL_PATH`, stub `get_ptxas_version → (12,9,0)`.
   - Freeze any param with `router` in its name (MoE stability; harmless if none found).
3. **Hyperparameters are tuned for a 1-hour SFT, not GRPO.** `lr=2e-4`, 1 epoch,
   `max_seq_len=1024`, `alpha=r=32`, dropout 0.0 + **NEFTune α=5**, cosine schedule,
   `adamw_torch`, grad checkpointing with `use_reentrant=False`.
4. **Chat template matters.** They build text with `tokenizer.apply_chat_template(...)`
   so the model trains in the exact Nemotron `<|im_start|>` / `<think>` format used at
   inference. Training raw `prompt\n\ncompletion` strings (our current scripts) risks a
   train/inference format mismatch.
5. **Submission must include tokenizer files.** Their zip carries `tokenizer.json`,
   `tokenizer_config.json`, `chat_template.jinja`, `README.md` alongside the adapter.
   Our `pack_submission.py` currently ships only adapter config + weights.

---

## 2. Decision: Keep Our Dataset, Change Everything Else

We keep `data/sft_reasoning_dataset.jsonl` (curated `<think>`-style CoT with `\boxed{}`
answers, already filtered for the 50.5% bit-manip and 49% charmap bugs). This is our
differentiator — real reasoning traces, not the notebook's terse "The answer is X" format.

But we feed it through the notebook's **training and environment recipe**:
- Format each `{prompt, completion}` pair through `apply_chat_template` instead of naive
  concatenation, so user turn = prompt + "put final answer in \boxed{}" instruction, and
  assistant turn = our CoT completion (which already contains `<think>...</think>\boxed{}`).
- Train full BF16 + rank-32 `all-linear` + NEFTune, not 4-bit.

Open question to validate early (Phase 1): whether keeping full CoT (longer sequences,
needs `max_seq_len` ~2048) outperforms the notebook's terse direct-answer format for the
`\boxed{}` exact-match metric. We will A/B this.

---

## 3. Execution Phases

### Phase 0 — Environment Hardening (highest priority, most failure-prone)
Create `src/blackwell_env.py` consolidating every workaround the notebook uses, callable
as `import blackwell_env; blackwell_env.apply()` at the very top of training (before model
load). Contents:
- Set `PYTORCH_CUDA_ALLOC_CONF` env var.
- `rmsnorm_fn` pure-PyTorch replacement injected into all `sys.modules` exposing it.
- Cutlass/quack mock modules + `_MockImportFinder` meta-path hook.
- Triton ptxas full-dir fix (guarded by path existence; supports the Kaggle
  `nvidia-utility-script` source path).
- `disable_fast_path(model)` helper to set `is_fast_path_available=False` post-load.
- `freeze_routers(model)` helper.

Fold the same logic into `scripts/setup_blackwell_offline.py` so the offline path stays
in sync.

### Phase 1 — Dataset Prep & Validation
1. Audit `data/sft_reasoning_dataset.jsonl`:
   - Confirm every `completion` has exactly one `\boxed{...}` with a **single** backslash
     (the generator `generate_sft_dataset.py` emits `\\\\boxed` → verify the committed
     JSONL decodes to `\boxed`, not `\\boxed`). Add an assertion/cleaner step.
   - Drop any row where the boxed answer is empty or contradicts the reasoning.
2. Add a `build_training_text(example, tokenizer)` function (in the new training script)
   that wraps prompt/completion through `apply_chat_template`, mirroring the notebook but
   using our CoT completion as the assistant message.
3. Decide `max_seq_len`: measure token-length distribution of formatted examples; start at
   **1536–2048** to fit full CoT (notebook used 1024 for terse answers).

### Phase 2 — New Training Script `src/train_blackwell.py`
Mirror the reference notebook flow exactly:
- `AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16,
  device_map="auto", trust_remote_code=True)` — **no quantization**.
- `blackwell_env.apply()` before load; `disable_fast_path` + `freeze_routers` after.
- `model.gradient_checkpointing_enable()`.
- `LoraConfig(r=32, lora_alpha=32, target_modules="all-linear", lora_dropout=0.0,
  bias="none", task_type=CAUSAL_LM)`.
- `SFTConfig` with the notebook's args (see Phase 3), using the
  `inspect.signature` guard for TRL version differences (`max_seq_length`,
  `dataset_text_field`, `packing`).
- Load our JSONL via `datasets.load_dataset("json", ...)`, map through
  `build_training_text`, train on the `text` field.
- Save adapter **and** tokenizer to `outputs/nemotron_lora_adapter`.

Keep `train_qlora.py` / `train_unsloth.py` as fallbacks but make `train_blackwell.py` the
primary path in the guide.

### Phase 3 — Hyperparameters (starting point, then tune)
```
LORA_RANK     = 32
LORA_ALPHA    = 32          # = r (notebook), not 2*r
LORA_DROPOUT  = 0.0
NEFTUNE_ALPHA = 5           # replaces dropout
MAX_SEQ_LEN   = 1536        # raised from 1024 to fit our CoT
NUM_EPOCHS    = 1           # then try 2 if underfit
BATCH_SIZE    = 1
GRAD_ACCUM    = 4
LR            = 2e-4
LR_SCHEDULER  = cosine
WARMUP_RATIO  = 0.1
WEIGHT_DECAY  = 0.01
MAX_GRAD_NORM = 1.0
OPTIM         = adamw_torch
GRAD_CKPT     = True (use_reentrant=False)
save_strategy = "no"
```
Update `src/train_config.yaml` to these SFT values (current values are GRPO/Megatron
scale and not used by this run), or add a dedicated `sft` block so we don't lose the GRPO
reference config.

### Phase 4 — Submission Packaging
Fix `src/pack_submission.py` to optionally include tokenizer artifacts:
- Always: `adapter_config.json`, `adapter_model.safetensors` (root level).
- Also bundle if present: `tokenizer.json`, `tokenizer_config.json`,
  `chat_template.jinja`, `special_tokens_map.json`.
- Keep everything at zip root (no nested dirs) — verify with `verify_submission.py`.

### Phase 5 — Local Validation Loop
1. Hold out a stratified slice of `train.csv` by puzzle type (bit-manip, roman, gravity,
   linear, cipher) as a local val set.
2. Build a `\boxed{}` extractor matching Kaggle's regex + `1e-2` numeric tolerance.
3. After training, run the adapter (HF generate, or vLLM if available) on the val set and
   report per-type accuracy. This is the signal we iterate on — not training loss.

### Phase 6 — Iteration Levers (in priority order)
1. CoT vs terse direct-answer format (biggest unknown).
2. `max_seq_len` 1024 vs 1536 vs 2048.
3. Epochs 1 → 2, and LR 2e-4 → 1e-4.
4. Dataset size / balance per puzzle type.
5. Self-consistency at inference (multi-sample majority vote on boxed value) if scoring
   harness allows multiple generations.

---

## 4. Two-Notebook Offline Workflow (retained, Internet OFF on Blackwell)

Kaggle disables internet on the Blackwell accelerator, so keep the bootstrap split:
1. **Bootstrap (T4, Internet ON):** `scripts/bootstrap_blackwell.py` downloads wheels +
   model + code → publish as private dataset. (The reference notebook instead uses
   `kagglehub.model_download(...)` + a tiny offline install of `datasets`/`trl`; if the
   competition model is attachable directly, prefer that and skip the heavy bootstrap.)
2. **Train (Blackwell, Internet OFF):** attach dataset → `setup_blackwell_offline.py`
   (now including the Phase 0 hardening) → restart kernel → `train_blackwell.py` →
   `pack_submission.py`.

`kaggle_guide.md` will be updated to point Cell 4 at `train_blackwell.py` and Cell 5 at
the tokenizer-inclusive packaging.

---

## 5. Concrete File-Level TODO

- [ ] `src/blackwell_env.py` — new: all env/Triton/CUTLASS/RMSNorm/router patches.
- [ ] `src/train_blackwell.py` — new: full BF16 + rank-32 all-linear + NEFTune SFT, chat
      template formatting of our JSONL.
- [ ] `src/train_config.yaml` — add/replace an `sft` block with Phase 3 values.
- [ ] `data/sft_reasoning_dataset.jsonl` — validate single-backslash `\boxed{}`, drop bad
      rows; fix `generate_sft_dataset.py` if it emits double backslashes.
- [ ] `src/pack_submission.py` — include tokenizer files; keep root-level structure.
- [ ] `src/verify_submission.py` — confirm it still passes with the larger zip.
- [ ] Local eval script — stratified val split + boxed extractor + per-type accuracy.
- [ ] `kaggle_guide.md` — repoint to `train_blackwell.py` and new packaging.

---

## 6. Risks & Mitigations

- **Env patches drift between notebook cell and offline script.** → Single source of truth
  in `blackwell_env.py`, imported by both.
- **Full CoT overflows `max_seq_len`.** → Measure distribution first; truncate or raise
  seq len; consider trimming verbose `<think>` blocks.
- **Double-backslash `\boxed` in data → 0 score at eval.** → Hard assertion in Phase 1.
- **Adapter won't load under vLLM** (rank > 32, wrong target modules, missing tokenizer).
  → Keep `r=32`, ship tokenizer, dry-run load before submitting.
- **kagglehub model path differs from offline bundle path.** → `train_blackwell.py` takes
  `--model_name` and auto-detects both `/kaggle/input/.../nemotron-base` and the kagglehub
  download path.

---

## 7. The RunPod Pivot (Internet-ON Alternative)

Given the extreme fragility of the offline Kaggle Blackwell environment, we built a fully documented `runpod_guide.md` to train the model on a standard RunPod A100/H100 instance instead.

**Key Learnings from the RunPod Environment:**
1. **PyTorch ABI Mismatch:** Running `pip install causal-conv1d mamba-ssm` natively often crashes because `pip` tries to upgrade PyTorch to a CUDA 13.0 version while the RunPod system compiler is CUDA 12.4.
   * *Fix:* We explicitly pin `torch==2.4.1` (cu124) and build the extensions with `--no-deps` to prevent upgrades.
2. **PyTorch Inductor Crashes:** If `triton` gets upgraded to `>=3.5.0` (which `mamba-ssm` requests), PyTorch 2.4.1's `compile_worker` throws `triton_key` import errors.
   * *Fix:* We strictly pin `triton==3.0.0`. Since we disable the Mamba fast-path anyway (`blackwell_env.disable_fast_path`), `mamba-ssm` runs perfectly on the older Triton version.
3. **Container Disk Out-of-Memory:** Hugging Face downloads default to `~/.cache/huggingface`, which sits on the tiny temporary Container Disk, instantly crashing the pod.
   * *Fix:* We prepend `HF_HOME=/workspace/hf_cache` to the training command to force the 60GB model download onto the persistent Volume Disk.

The output `submission.zip` from RunPod has been verified to structure identically to the Kaggle notebook output and can be submitted directly to the leaderboard.
