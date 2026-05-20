# 🏆 NVIDIA Nemotron Model Reasoning Challenge: SFT Plan & Progress

This document outlines our active training setup, progress milestones, environment configurations, and the final verification/submission roadmap for fine-tuning the **NVIDIA Nemotron-3-Nano-30B** model.

---

## 📅 Current Progress & Milestones

*   [x] **Identify Data Bugs**: Curated dataset (`sft_reasoning_dataset.jsonl`) to eliminate noisy/mismatching traces (50.5% bit manipulation errors and 49% charmap mismatches).
*   [x] **Configure 4-bit QLoRA SFT**: Optimized hyperparameter config in `train_config.yaml` (Rank=32, Gradient Checkpointing, bfloat16 compute, micro-batch=1).
*   [x] **Resolve Environment Obstacles**:
    *   *Attempt 1 (Kaggle T4)*: Encountered VRAM OOM during weight sharding due to dynamic module class registration flaws.
    *   *Attempt 2 (RunPod 4090)*: Switched to RunPod but hit a severe CUDA Driver vs. PyTorch compilation ABI mismatch on the host template.
    *   *Attempt 3 (Kaggle T4 - PIVOT)*: Discovered the `nvidia-utility-script` providing pre-compiled Mamba/Cutlass CUDA kernels. Bypassed the 10-minute compilation phase.
*   [x] **Launch Active SFT Training**: Combined the instant pre-compiled imports from the utility script with our optimized 4-bit QLoRA trainer on **Kaggle Dual T4 GPUs**.
*   [🔄] **Training Active**: Model download, quantization sharding, and active training steps are currently executing on Kaggle.

---

## ⚙️ Active Environment Configuration (Kaggle Dual T4 x2)

Because TPUs do not support Mamba's CUDA kernels and single GPUs (P100/Single T4) lack VRAM, we are running on **Dual T4 x2 GPUs (30GB total VRAM)**.

We are executing the training in a Kaggle Notebook using the following 3-cell setup:

### Cell 1: Environment Setup (Instant Mamba Mount)
```python
import site
import os
import subprocess

print("⚙️ Mounting pre-compiled Mamba & Cutlass CUDA kernels...")
path_hyphen = "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/nvidia_cutlass_dsl/python_packages/"
path_underscore = "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script/nvidia_cutlass_dsl/python_packages/"

mounted = False
for path in [path_hyphen, path_underscore]:
    if os.path.exists(path):
        site.addsitedir(path)
        print(f"✅ Successfully mounted pre-compiled path: {path}")
        mounted = True
        break

import mamba_ssm
print("✅ mamba_ssm imported successfully (Instant!)")

print("📥 Installing high-level training libraries...")
subprocess.run("pip install -q peft datasets trl pyyaml bitsandbytes", shell=True, check=True)

# Apply custom PyTorch 2.4 schema parsing patch to transformers
try:
    file_path = '/usr/local/lib/python3.11/dist-packages/transformers/integrations/moe.py'
    if not os.path.exists(file_path):
        file_path = '/usr/local/lib/python3.12/dist-packages/transformers/integrations/moe.py'
    with open(file_path, 'r') as f:
        content = f.read()
    old_str = 'torch.library.custom_op(\"transformers::grouped_mm_fallback\", _grouped_mm_fallback, mutates_args=())'
    new_str = 'torch.library.custom_op(\"transformers::grouped_mm_fallback\", _grouped_mm_fallback, mutates_args=(), schema=\"(Tensor input, Tensor weight, Tensor offs) -> Tensor\")'
    if old_str in content:
        content = content.replace(old_str, new_str)
        with open(file_path, 'w') as f:
            f.write(content)
        print('✅ Successfully patched transformers moe.py!')
except Exception as e:
    print(f'⚠️ Patch skipped: {e}')
```

### Cell 2: Clone Code & Prepare Workspace
```python
!rm -rf nemotron_kaggle
!git clone https://github.com/gbose01/nemotron_kaggle.git
%cd nemotron_kaggle
```
*(Note: The dataset `data/sft_reasoning_dataset.jsonl` is fully tracked in Git and is pulled down automatically during this clone!)*

### Cell 3: Launch 4-bit QLoRA Trainer
```python
import os

env_paths = [
    "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script/nvidia_cutlass_dsl/python_packages/",
    "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script/",
    "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/nvidia_cutlass_dsl/python_packages/",
    "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/"
]
pythonpath_val = ":".join(env_paths)
os.environ["PYTHONPATH"] = f"{pythonpath_val}:{os.environ.get('PYTHONPATH', '')}"
os.environ["HF_TOKEN"] = "YOUR_HUGGINGFACE_READ_TOKEN"

!python3 src/train_qlora.py
```

---

## 📊 SFT Training Runtime Estimation

Given the compact size of our high-quality reasoning dataset, training is highly optimized:
*   **Dataset Size**: ~140 KB (approx. 100–150 curated reasoning pairs).
*   **Global Batch Size**: 16 (accumulated steps).
*   **Steps per Epoch**: ~9 steps.
*   **Epochs**: 3.
*   **Total Steps**: ~27 steps.
*   **Time per Step**: ~30 to 45 seconds on Dual T4s (due to 4-bit QLoRA VRAM-native storage).
*   **Estimated Completion**: **15 to 25 minutes** (excluding the 4-minute download time).

---

## 🏁 Next Steps & Post-Training Roadmap

Once the training cell completes successfully and outputs the message `🎉 QLoRA SFT completed successfully!`, we will execute the following verification and submission pipeline:

### 1. Verify Adapter Exports
Confirm that the adapter files were successfully saved in `/kaggle/working/nemotron_kaggle/outputs/nemotron_lora_adapter/`:
*   `adapter_config.json` (Adapter configuration).
*   `adapter_model.safetensors` (Fine-tuned LoRA weights).

### 2. Package the Submission
Run the packaging utility inside your Kaggle notebook (or via a new cell) to bundle the files into the root of your Kaggle working directory:
```bash
python3 src/pack_submission.py --source outputs/nemotron_lora_adapter --output /kaggle/working/submission.zip
```

### 3. Verify Submission Compliance
Before submitting to the leaderboard, run the compliance verification script to guarantee that Kaggle's autograder will parse your submission successfully:
```bash
python3 src/verify_submission.py --submission /kaggle/working/submission.zip
```
This will verify:
*   That the zip structure is correct (no nested folders).
*   That the adapter configuration parses.
*   That the weights match the custom Nemotron-3 model structure.

### 4. Download & Submit!
Generate a download link directly in your notebook outputs:
```python
from IPython.display import FileLink
FileLink('submission.zip')
```
Click the link, download the package, and upload it to the Kaggle competition page!
