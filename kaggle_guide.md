# Running Nemotron Challenge on Kaggle via GitHub

Since your corporate laptop blocks manual file uploads in the browser, the **best and most standard industry method** to get your files into Kaggle is using Git. 

Because your repository is already linked and successfully pushed to GitHub at `https://github.com/gbose01/nemotron_kaggle`, you can directly clone and execute everything inside a Kaggle Notebook in just **4 simple cells**.

---

## ⚙️ Prerequisites: Kaggle Settings
Before running the notebook, make sure to configure the following in the Kaggle settings panel on the right:
1. **Accelerator:** Select **GPU T4 x2** (Dual T4 GPUs are absolutely required to fit the 30B model).
2. **Internet on:** Toggle this **ON** (this is critical so Kaggle can clone from GitHub and download PyPI packages).
3. **Hugging Face Access (Gated Model):**
   * The `NVIDIA/Nemotron-3-Nano-30B-Base` model is **gated** on Hugging Face.
   * Go to [Hugging Face](https://huggingface.co/NVIDIA/Nemotron-3-Nano-30B-Base), log in, and **accept the terms** of the model.
   * Generate a **Read** Access Token at [Hugging Face Settings](https://huggingface.co/settings/tokens).
   * In your Kaggle Notebook, go to **Add-ons** -> **Secrets** in the top menu bar.
   * Add a secret with the label `HF_TOKEN` and paste your Hugging Face access token as the value. Keep the "Access" toggle checked.

---

## 📓 Step-by-Step Kaggle Notebook Cells

Copy and paste the following blocks into separate cells in your Kaggle Notebook:

### 📥 Cell 1: Clone Repository & Install Dependencies
*This cell cleans any previous runs, clones the repository directly from your GitHub, and installs optimized training packages.*

```python
# 1. Clean up and clone repository
!rm -rf nemotron_kaggle
!git clone https://github.com/gbose01/nemotron_kaggle.git

# 2. Change working directory
%cd nemotron_kaggle

# 3. Install optimized libraries
!pip install -q torch transformers peft trl datasets accelerate bitsandbytes --index-url https://pypi.org/simple
```

---

### 🏋️ Cell 2: Run QLoRA SFT Training
*This cell retrieves your Hugging Face access token securely from Kaggle Secrets, injects it into the environment, and executes the 4-bit NF4 QLoRA training loop. It automatically shards the 30B model across the two T4 GPUs (`device_map="auto"`), utilizes our curated SFT dataset, and runs for 3 epochs.*

```python
# 1. Retrieve the Hugging Face token securely from Kaggle Secrets
from kaggle_secrets import UserSecretsClient
import os

try:
    user_secrets = UserSecretsClient()
    hf_token = user_secrets.get_secret("HF_TOKEN")
    # Ingest token into Jupyter session env so subprocesses can inherit it
    %env HF_TOKEN=$hf_token
    print("✅ HF_TOKEN successfully loaded from Kaggle Secrets!")
except Exception as e:
    print("⚠️ Could not load HF_TOKEN from Kaggle Secrets. If you encounter a 401 error, verify your Secrets configuration.")

# 2. Start QLoRA training
!PYTHONPATH=src python3 src/train_qlora.py --config src/train_config.yaml
```

---

### 📦 Cell 3: Package Submission
*Once training completes, this packages the final adapter weights and config file into the strictly compliant root-level `submission.zip`.*

```python
# Package the trained adapter outputs
!python3 src/pack_submission.py --source outputs/nemotron_lora_adapter --output submission.zip
```

---

### 📥 Cell 4: Download Submission
*This displays a clickable link directly in the notebook outputs to download the completed `submission.zip` package directly to your machine for Kaggle submission.*

```python
from IPython.display import FileLink
# Generate download link for the submission file
FileLink('submission.zip')
```

---

## 💡 Tips & Troubleshooting

> [!IMPORTANT]
> **401 Unauthorized Error:** If you still get a `401 Unauthorized` error:
> 1. Make sure you accepted the NVIDIA license on the [NVIDIA/Nemotron-3-Nano-30B-Base HF Model Page](https://huggingface.co/NVIDIA/Nemotron-3-Nano-30B-Base) using the *same* Hugging Face account that generated your token.
> 2. Make sure the Kaggle Secret is named exactly `HF_TOKEN` (all-caps, no spaces).

> [!NOTE]
> **Dual GPU Utilization:** The script uses PyTorch and HuggingFace's automatic `device_map="auto"`, which will load and shard the model seamlessly across GPU `cuda:0` and GPU `cuda:1`. You do not need to run multi-GPU launch commands (`accelerate launch`).

> [!TIP]
> **Checkpoints:** If you want to resume training or modify hyperparameters, you can edit `src/train_config.yaml` locally, commit and push to GitHub (`git commit -am "update config" && git push`), and then run `!git pull` inside Cell 1 in Kaggle to update.