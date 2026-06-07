# Blackwell Fine-Tuning on Kaggle (Offline Bootstrap Method)

Kaggle **disables Internet** when you select **RTX PRO 6000 (Blackwell)** for this competition.

So you must use a **2-notebook bootstrap**:

1. **Bootstrap notebook** (T4 + Internet ON) → download wheels, model, code → publish as Kaggle Dataset
2. **Train notebook** (Blackwell + Internet OFF) → install offline from dataset → train → package submission

---

## Prerequisites (do once)

1. Join the [NVIDIA Nemotron Model Reasoning Challenge](https://www.kaggle.com/competitions/nvidia-nemotron-model-reasoning-challenge).
2. Accept the license on [Nemotron-3-Nano-30B Base](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16).
3. In Kaggle notebook: **Add-ons → Secrets** → add `HF_TOKEN` (Read token).

---

## Notebook 1 — Bootstrap (T4, Internet ON)

### Settings
- **Accelerator:** `GPU T4 x2`
- **Internet:** `ON`

### Cell 1 — clone repo
```python
!rm -rf /kaggle/working/nemotron_kaggle
!git clone https://github.com/gbose01/nemotron_kaggle.git /kaggle/working/nemotron_kaggle
%cd /kaggle/working/nemotron_kaggle
```

### Cell 2 — run bootstrap downloader
```python
from kaggle_secrets import UserSecretsClient
import os

os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

!python scripts/bootstrap_blackwell.py
```

This creates:
```text
/kaggle/working/nemotron-blackwell-offline/
  wheels/
  nemotron-base/
  nemotron_kaggle/
  MANIFEST.txt
```

### Cell 3 — publish as Kaggle Dataset
1. In the file browser, zip or upload the folder `nemotron-blackwell-offline`.
2. Go to **Datasets → New Dataset**.
3. Upload the bundle as a **private** dataset.
4. Name it something like `nemotron-blackwell-offline`.

> Keep this dataset private. It contains model weights and your training code.

---

## Notebook 2 — Train on Blackwell (Internet OFF)

### Settings
- **Accelerator:** `RTX PRO 6000 (Blackwell)`
- **Internet:** `OFF` (forced)
- **Input:** attach dataset `nemotron-blackwell-offline`

### Cell 1 — offline install + Triton fix
```python
import os, shutil, subprocess, sys
from pathlib import Path

# Auto-detect mounted bundle
bundle = None
for p in Path("/kaggle/input").iterdir():
    if (p / "wheels").exists() and (p / "nemotron-base").exists():
        bundle = p
        break
    nested = p / "nemotron-blackwell-offline"
    if nested.exists() and (nested / "wheels").exists():
        bundle = nested
        break
assert bundle is not None, "Attach nemotron-blackwell-offline dataset first"
print("Bundle:", bundle)

# Copy training code to writable path
!rm -rf /kaggle/working/nemotron_kaggle
!cp -r {bundle}/nemotron_kaggle /kaggle/working/nemotron_kaggle
%cd /kaggle/working/nemotron_kaggle

# Run offline setup script from copied repo
!python scripts/setup_blackwell_offline.py --bundle-root {bundle}
```

### Cell 2 — RESTART KERNEL (required)
**Run → Restart Kernel**

### Cell 3 — verify torch + GPU
```python
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))
```

### Cell 4 — train (full BF16 LoRA, no internet)
All Blackwell/Triton/CUTLASS/RMSNorm patches now live in `src/blackwell_env.py`,
which `train_blackwell.py` imports and applies automatically before model load.
No manual ptxas copy needed here.

```python
from pathlib import Path

# Resolve local model path from attached dataset
model_path = None
for p in Path("/kaggle/input").iterdir():
    candidate = p / "nemotron-base"
    if candidate.exists():
        model_path = candidate
        break
    nested = p / "nemotron-blackwell-offline" / "nemotron-base"
    if nested.exists():
        model_path = nested
        break
assert model_path is not None
print("Using local model:", model_path)

%cd /kaggle/working/nemotron_kaggle
!PYTHONPATH=src python3 src/train_blackwell.py \
  --config src/train_config.yaml \
  --model_name {model_path} \
  --data data/sft_reasoning_dataset.jsonl \
  --output_dir outputs/nemotron_lora_adapter
```

> Recipe (from `sft:` block in `train_config.yaml`): full BF16, LoRA r=32,
> alpha=32, `all-linear`, dropout 0.0 + NEFTune α=5, lr=2e-4, cosine, 1 epoch,
> grad-ckpt. Expect loss ~11 → ~3 and ~75 GB peak VRAM.

### Cell 5 — (optional) local accuracy check
```python
!PYTHONPATH=src python3 src/evaluate_adapter.py \
  --model_name {model_path} \
  --adapter outputs/nemotron_lora_adapter \
  --train_csv /kaggle/input/nvidia-nemotron-3-reasoning-challenge/train.csv \
  --n_per_type 10
```

### Cell 6 — package submission (adapter + tokenizer)
```python
!python3 src/pack_submission.py \
  --source outputs/nemotron_lora_adapter \
  --output /kaggle/working/submission.zip

!python3 src/verify_submission.py --submission /kaggle/working/submission.zip

from IPython.display import FileLink
FileLink("/kaggle/working/submission.zip")
```

---

## Troubleshooting

### `Internet cannot be enabled with current accelerator`
Expected on Blackwell. Do **not** try to enable Internet there. Use Notebook 1 bootstrap first.

### `CUDA error: no kernel image is available`
- You skipped PyTorch cu128 nightly install.
- Fix: rebuild bootstrap dataset, rerun `setup_blackwell_offline.py`, restart kernel.

### `mamba-ssm` / `causal-conv1d` install fails offline
`setup_blackwell_offline.py` automatically retries with `--no-build-isolation` on Blackwell.
If still failing, rerun bootstrap on T4 and ensure sdists are present in `wheels/`.

### `401 Unauthorized` during bootstrap only
- Accept Nemotron license on Hugging Face with the same account as `HF_TOKEN`.
- Secret name must be exactly `HF_TOKEN`.

### Training dataset focus
Current SFT data (`data/sft_reasoning_dataset.jsonl`) targets symbolic reasoning puzzles:
bit rules, ciphers, numeral/unit transforms, boxed answers with `<think>`.

---

## Quick checklist

- [ ] Bootstrap notebook finished on T4 with Internet ON
- [ ] Private dataset `nemotron-blackwell-offline` uploaded
- [ ] Blackwell notebook has dataset attached, Internet OFF
- [ ] Ran setup, restarted kernel, verified torch/GPU
- [ ] Trained with `--model_name` pointing to local `nemotron-base`
- [ ] Produced `/kaggle/working/submission.zip`
