# Blackwell on Kaggle — Simple Smoke Test First

Start with the **smoke test** below to confirm load → train → `submission.zip` works.
No wheel bootstrap, no 60 GB model download — use Kaggle's built-in inputs.

Once the smoke test passes, run the full training (bottom of this doc) or the
two-notebook bootstrap if packages are missing on the image.

---

## Smoke test (single Blackwell notebook, Internet OFF)

### Notebook settings
- **Accelerator:** `RTX PRO 6000 (Blackwell)`
- **Internet:** OFF (forced)

### Inputs to attach
| Input | How |
|-------|-----|
| Competition data | Auto-attached when you create a competition notebook |
| Nemotron-30B weights | **Add Input → Models →** `metric/nemotron-3-nano-30b-a3b-bf16` |
| Training code | **Add Input → Datasets →** your `nemotron-kaggle-code` dataset (see one-time setup below) |

> The model comes from Kaggle Models — **not** a custom bootstrap dataset.
> The code dataset is tiny (~few MB). Publish it once from a T4 notebook.

### One-time: publish code-only dataset (T4, Internet ON)

```python
!git clone https://github.com/gbose01/nemotron_kaggle.git /kaggle/working/nemotron_kaggle
%cd /kaggle/working/nemotron_kaggle
!python scripts/stage_code_only.py
# Upload /kaggle/temp/nemotron-kaggle-code as private dataset "nemotron-kaggle-code"
```

Or zip the repo's `src/` + `scripts/` folders manually — same thing.

**Ready-made notebook:** upload [`notebooks/blackwell_smoke.ipynb`](notebooks/blackwell_smoke.ipynb)
to Kaggle (or import from repo), set Blackwell accelerator, attach inputs, run all cells.

---

### Cell 1 — copy code from attached dataset

```python
import shutil
from pathlib import Path

code_src = next(Path("/kaggle/input").rglob("src/blackwell_env.py")).parent.parent
model_dir = next(
    cfg.parent
    for cfg in Path("/kaggle/input").rglob("config.json")
    if (cfg.parent / "model.safetensors.index.json").exists()
    or list(cfg.parent.glob("model-*.safetensors"))
)

shutil.copytree(code_src, "/kaggle/working/nemotron_kaggle", dirs_exist_ok=True)
print("Model:", model_dir)
print("Code: ", "/kaggle/working/nemotron_kaggle")

!PYTHONPATH=src python3 /kaggle/working/nemotron_kaggle/scripts/setup_kaggle_inputs.py
```

### Cell 2 — verify GPU + imports

```python
import torch
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))

for pkg in ("transformers", "peft", "trl", "datasets", "mamba_ssm"):
    try:
        __import__(pkg)
        print(f"  {pkg}: OK")
    except ImportError as e:
        print(f"  {pkg}: MISSING — use full bootstrap (see bottom)")
```

If `mamba_ssm` or `torch` is missing/wrong version, skip to
[Full bootstrap (two notebooks)](#full-bootstrap-two-notebooks) at the bottom.

### Cell 3 — smoke test (2 steps, ~10 min)

Trains on 2 inline examples, saves a LoRA adapter, builds `submission.zip`.

```python
%cd /kaggle/working/nemotron_kaggle
!PYTHONPATH=src python3 src/smoke_blackwell.py --max_steps 2 --lora_rank 8
```

Expected:
- Model loads from `/kaggle/input/models/...` (auto-detected)
- Loss prints for 2 steps
- `/kaggle/working/submission.zip` created and passes verify

```python
from IPython.display import FileLink
FileLink("/kaggle/working/submission.zip")
```

### Cell 4 — submit smoke test to competition (optional)

Upload `submission.zip` via **Submit to Competition** to confirm the scoring
engine accepts the format. Score will be low — that's fine for a pipeline check.

---

## Full training (after smoke test passes)

Same notebook inputs (competition + model + code). No bootstrap needed if
imports passed in Cell 2.

```python
%cd /kaggle/working/nemotron_kaggle
!PYTHONPATH=src python3 src/train_runpod.py \
  --config src/train_config.yaml \
  --data data/sft_reasoning_dataset_v2.jsonl \
  --output_dir outputs/nemotron_lora_adapter
```

Pack and verify:

```python
!python3 src/pack_submission.py \
  --source outputs/nemotron_lora_adapter \
  --output /kaggle/working/submission.zip
!python3 src/verify_submission.py --submission /kaggle/working/submission.zip
```

---

## Full bootstrap (two notebooks)

Use this **only if** the smoke test fails on missing packages (`torch` cu128,
`mamba-ssm`, `causal-conv1d`, etc.).

### Notebook 1 — Bootstrap (T4, Internet ON)

Settings: `GPU T4 x2`, Internet ON.

```python
!rm -rf /kaggle/working/nemotron_kaggle
!git clone https://github.com/gbose01/nemotron_kaggle.git /kaggle/working/nemotron_kaggle
%cd /kaggle/working/nemotron_kaggle

from kaggle_secrets import UserSecretsClient
import os
os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")

!python scripts/bootstrap_blackwell.py --stage deps
# Publish /kaggle/temp/nemotron-blackwell-offline as private dataset
```

Attach **both** the deps dataset and the competition model on Blackwell.

### Notebook 2 — Bootstrap install (Blackwell, Internet OFF)

```python
from pathlib import Path

bundle = None
for p in Path("/kaggle/input").iterdir():
    if (p / "wheels").exists():
        bundle = p
        break
    nested = p / "nemotron-blackwell-offline"
    if nested.exists() and (nested / "wheels").exists():
        bundle = nested
        break
assert bundle, "Attach nemotron-blackwell-offline (deps) dataset"

!cp -r {bundle}/nemotron_kaggle /kaggle/working/nemotron_kaggle
%cd /kaggle/working/nemotron_kaggle
!python scripts/setup_blackwell_offline.py --bundle-root {bundle}
```

**Restart kernel**, then run smoke test or full training as above.

---

## Troubleshooting

### `Internet cannot be enabled with current accelerator`
Expected on Blackwell. Attach inputs instead of downloading.

### `No model found under /kaggle/input`
Add Input → Models → `metric/nemotron-3-nano-30b-a3b-bf16`.

### `CUDA error: no kernel image is available`
PyTorch build doesn't match Blackwell. Use the bootstrap deps path (cu128 nightly).

### `mamba_ssm` / `causal-conv1d` import fails
Use bootstrap `--stage deps` (sdists compile on Blackwell).

### Smoke test passes but full training OOM
Reduce `sft.max_seq_length` in `train_config.yaml` or use smaller batch.

---

## Quick checklist

**Smoke test path:**
- [ ] Competition notebook on Blackwell
- [ ] Model attached via Add Input → Models
- [ ] Code dataset attached (`nemotron-kaggle-code`)
- [ ] Imports OK in Cell 2
- [ ] `smoke_blackwell.py` finishes, `submission.zip` verifies

**Full training path:**
- [ ] Same inputs as smoke test
- [ ] `train_runpod.py` completes (~1 h)
- [ ] `submission.zip` includes adapter + tokenizer files
