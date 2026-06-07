#!/usr/bin/env python3
"""
Run on Kaggle with GPU T4 x2 + Internet ON.

Kaggle /kaggle/working is only ~20GB. The 30B BF16 model is ~60GB.
Use /kaggle/temp for staging and run in two stages if needed:

  Stage deps  -> wheels + code   (~5-10 GB)  publish as dataset 1
  Stage model -> base weights    (~60 GB)    publish as dataset 2

Usage:
  python scripts/bootstrap_blackwell.py --stage deps
  python scripts/bootstrap_blackwell.py --stage model
  python scripts/bootstrap_blackwell.py --stage all   # only if you have enough disk
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-Base-BF16"
DEFAULT_REPO = "https://github.com/gbose01/nemotron_kaggle.git"
DEFAULT_ROOT = Path("/kaggle/temp/nemotron-blackwell-offline")

WHEEL_PACKAGES = [
    "unsloth",
    "transformers",
    "peft",
    "trl",
    "datasets",
    "accelerate",
    "bitsandbytes",
    "pyyaml",
    "einops",
    "triton",
    "causal-conv1d",
    "mamba-ssm",
    "huggingface_hub",
    "hf_transfer",
    "sentencepiece",
    "protobuf",
    "safetensors",
    "tokenizers",
    "numpy",
    "packaging",
    "filelock",
    "regex",
    "requests",
    "tqdm",
    "psutil",
]


def run(cmd: list[str], **kwargs) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True, **kwargs)


def disk_free_gb(path: str | Path) -> float:
    usage = shutil.disk_usage(str(path))
    return usage.free / (1024**3)


def print_disk_report() -> None:
    for label, path in [
        ("working", "/kaggle/working"),
        ("temp", "/kaggle/temp"),
        ("input", "/kaggle/input"),
    ]:
        if os.path.exists(path):
            free = disk_free_gb(path)
            total = shutil.disk_usage(path).total / (1024**3)
            print(f"disk {label:7s} {path}: {free:.1f} GB free / {total:.1f} GB total")


def configure_hf_cache(cache_root: Path) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_root)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_root / "hub")
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    # Avoid duplicate cache + local_dir copies on tight disks
    os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


def download_torch_wheels(wheels_dir: Path) -> None:
    wheels_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            "torch",
            "torchvision",
            "torchaudio",
            "--dest",
            str(wheels_dir),
            "--index-url",
            "https://download.pytorch.org/whl/nightly/cu128",
        ]
    )


def download_python_wheels(wheels_dir: Path) -> None:
    wheels_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            *WHEEL_PACKAGES,
            "--dest",
            str(wheels_dir),
        ]
    )


def download_model(model_id: str, model_dir: Path, token: str | None) -> None:
    from huggingface_hub import snapshot_download

    free = disk_free_gb(model_dir.parent)
    print(f"Free space before model download: {free:.1f} GB")
    if free < 55:
        raise RuntimeError(
            f"Not enough free disk ({free:.1f} GB). Need ~60 GB for BF16 30B weights.\n"
            "Run only --stage model in a fresh notebook session after clearing space,\n"
            "or publish deps first and delete local wheels before downloading model."
        )

    if model_dir.exists():
        shutil.rmtree(model_dir)

    configure_hf_cache(model_dir.parent / "hf-cache")
    model_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {model_id} -> {model_dir}")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(model_dir),
        token=token,
        local_dir_use_symlinks=False,
    )


def clone_repo(repo_url: str, code_dir: Path) -> None:
    if code_dir.exists():
        shutil.rmtree(code_dir)
    run(["git", "clone", "--depth", "1", repo_url, str(code_dir)])


def write_manifest(root: Path, model_id: str, stage: str) -> None:
    manifest = root / "MANIFEST.txt"
    manifest.write_text(
        "\n".join(
            [
                "nemotron-blackwell-offline bundle",
                f"stage={stage}",
                f"model_id={model_id}",
                "folders may include:",
                "  wheels/",
                "  nemotron-base/",
                "  nemotron_kaggle/",
                "",
                "Publish as private Kaggle Dataset(s).",
                "Blackwell notebook can attach one or two datasets:",
                "  - nemotron-blackwell-deps  (wheels + code)",
                "  - nemotron-blackwell-model (nemotron-base)",
            ]
        ),
        encoding="utf-8",
    )


def stage_deps(root: Path, repo_url: str) -> None:
    wheels_dir = root / "wheels"
    code_dir = root / "nemotron_kaggle"
    root.mkdir(parents=True, exist_ok=True)

    print("\n[deps 1/3] Downloading PyTorch cu128 nightly wheels...")
    download_torch_wheels(wheels_dir)
    print("\n[deps 2/3] Downloading Python dependency wheels...")
    download_python_wheels(wheels_dir)
    print("\n[deps 3/3] Cloning training repository...")
    clone_repo(repo_url, code_dir)


def stage_model(root: Path, model_id: str, token: str | None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    model_dir = root / "nemotron-base"
    print("\n[model] Downloading base model snapshot...")
    download_model(model_id, model_dir, token)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap offline assets for Blackwell Kaggle training")
    parser.add_argument("--root", type=str, default=str(DEFAULT_ROOT))
    parser.add_argument("--model-id", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--repo-url", type=str, default=DEFAULT_REPO)
    parser.add_argument("--hf-token", type=str, default=os.environ.get("HF_TOKEN"))
    parser.add_argument(
        "--stage",
        choices=["deps", "model", "all"],
        default="deps",
        help="deps=wheels+code, model=weights only, all=both (needs ~70GB free)",
    )
    args = parser.parse_args()

    print_disk_report()

    root = Path(args.root)
    if args.stage in ("deps", "all"):
        if not args.hf_token and args.stage == "all":
            pass
        print(f"\nStaging deps under {root}")
        stage_deps(root, args.repo_url)
        write_manifest(root, args.model_id, "deps")

    if args.stage in ("model", "all"):
        if not args.hf_token:
            print("ERROR: HF_TOKEN is required for model stage.")
            sys.exit(1)
        print("Installing huggingface_hub for model download...")
        run([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub", "hf_transfer"])
        print(f"\nStaging model under {root}")
        stage_model(root, args.model_id, args.hf_token)
        write_manifest(root, args.model_id, args.stage)

    print_disk_report()
    print("\nBootstrap stage complete.")
    print(f"Bundle root: {root}")
    if args.stage == "deps":
        print("Next: publish deps dataset, then run --stage model in a fresh session.")
    elif args.stage == "model":
        print("Next: publish model dataset and attach with deps on Blackwell notebook.")
    else:
        print("Next: publish bundle as private Kaggle Dataset(s).")


if __name__ == "__main__":
    main()
