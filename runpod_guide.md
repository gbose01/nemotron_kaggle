# Running Nemotron Training on RunPod

This guide covers how to set up, train, and export your submission from a RunPod instance. Because RunPod gives you full internet access and root privileges, the setup is much simpler than the offline Kaggle environment.

## 1. Provisioning Your RunPod Instance

1. Select a GPU template that has the standard **PyTorch** image (e.g., RunPod PyTorch 2.x).
2. Choose a GPU with enough VRAM. The 30B model in BF16 requires at least an **A100 (80GB)** or **RTX 6000 Ada (48GB might be tight, 80GB+ is safer)**.
3. Make sure to provision enough disk space (at least 100 GB to hold the model weights and datasets).

## 2. Setup and Cloning

SSH into your RunPod instance or open the Jupyter Lab Terminal, then run the following:

```bash
# Clone your repository
git clone https://github.com/gbose01/nemotron_kaggle.git
cd nemotron_kaggle

# 1. Update pip and ensure PyTorch matches the RunPod CUDA compiler
pip install -U pip
pip uninstall -y torch torchvision torchaudio
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124

# 2. Install general dependencies
# Note: We specifically pin triton==3.0.0 to prevent a PyTorch inductor crash
pip install transformers peft trl datasets accelerate bitsandbytes pyyaml einops triton==3.0.0 huggingface_hub hf_transfer sentencepiece protobuf safetensors tokenizers numpy packaging filelock regex requests tqdm psutil

# 3. Build CUDA specific packages (causal-conv1d and mamba-ssm)
# We use --no-deps to prevent pip from accidentally breaking the PyTorch version we just pinned.
pip install causal-conv1d mamba-ssm --no-build-isolation --no-deps
```
*(Note: The CUDA compilation in step 3 takes about 5-10 minutes and won't show progress bars. This is completely normal!)*

## 3. Log in to Hugging Face

You will need your Hugging Face token to download the base model (`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16`).

```bash
hf auth login
```

## 4. Training

You can run the main training script. We pass the Hugging Face model ID directly via `--model_name`.

```bash
# We use HF_HOME=/workspace/hf_cache so the 60GB model downloads to the persistent volume 
# rather than filling up and crashing the temporary container disk.
HF_HOME=/workspace/hf_cache PYTHONPATH=src python3 src/train_runpod.py \
    --config src/train_config.yaml \
    --data data/sft_reasoning_dataset_v2.jsonl \
    --model_name "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16" \
    --output_dir outputs/nemotron_lora_adapter
```

## 5. Pack Submission

Once the training completes, pack the LoRA adapter and tokenizer into `submission.zip`:

```bash
python3 src/pack_submission.py \
    --source outputs/nemotron_lora_adapter \
    --output submission.zip
```

Verify it meets Kaggle's requirements:

```bash
python3 src/verify_submission.py --submission submission.zip
```

## 6. Upload back to Kaggle

Download the generated `submission.zip` from your RunPod instance to your local machine (using `scp` or downloading from JupyterLab interface), and submit it directly to the competition page on Kaggle.
