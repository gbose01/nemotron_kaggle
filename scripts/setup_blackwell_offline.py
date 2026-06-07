#!/usr/bin/env python3
"""
Run on Kaggle with RTX PRO 6000 (Blackwell) + Internet OFF.

Expects a mounted dataset containing the bootstrap bundle:

  /kaggle/input/<your-dataset>/wheels/
  /kaggle/input/<your-dataset>/nemotron-base/
  /kaggle/input/<your-dataset>/nemotron_kaggle/

Copies code to /kaggle/working, installs wheels offline, applies Triton fix.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], **kwargs) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True, **kwargs)


def find_bundle_root(explicit: str | None) -> Path:
    if explicit:
        root = Path(explicit)
        if (root / "wheels").exists() and (root / "nemotron-base").exists():
            return root
        raise FileNotFoundError(f"Bundle not found at {root}")

    for entry in sorted(Path("/kaggle/input").iterdir()):
        candidate = entry
        if (candidate / "wheels").exists() and (candidate / "nemotron-base").exists():
            return candidate
        nested = candidate / "nemotron-blackwell-offline"
        if nested.exists() and (nested / "wheels").exists():
            return nested

    raise FileNotFoundError(
        "Could not find bootstrap bundle under /kaggle/input. "
        "Attach your offline dataset or pass --bundle-root."
    )


def apply_triton_fix() -> None:
    os.makedirs("/tmp/bin", exist_ok=True)
    src = "/opt/nvidia/ptxas-blackwell"
    if not os.path.exists(src):
        print("WARNING: /opt/nvidia/ptxas-blackwell not found; skipping Triton fix.")
        return
    dst = "/tmp/bin/ptxas"
    shutil.copy(src, dst)
    os.chmod(dst, 0o755)
    os.environ["TRITON_PTXAS_PATH"] = dst
    print(f"Triton PTXAS configured: {dst}")


def install_offline(wheels_dir: Path) -> None:
    # torch only — torchvision/torchaudio not needed for Nemotron SFT and their
    # nightly wheels pin exact torch dev versions that conflict across download days.
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            f"--find-links={wheels_dir}",
            "torch",
        ]
    )

    # Pure-python / generic wheels first
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            f"--find-links={wheels_dir}",
            "transformers",
            "peft",
            "trl",
            "datasets",
            "accelerate",
            "bitsandbytes",
            "pyyaml",
            "einops",
            "huggingface_hub",
            "safetensors",
            "tokenizers",
            "numpy",
            "packaging",
            "regex",
            "requests",
            "tqdm",
            "psutil",
            "sentencepiece",
            "protobuf",
        ]
    )

    # CUDA extensions: try wheel, then build from sdist on Blackwell
    for pkg in ("causal-conv1d", "mamba-ssm", "triton", "unsloth"):
        try:
            run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-index",
                    f"--find-links={wheels_dir}",
                    pkg,
                ]
            )
        except subprocess.CalledProcessError:
            print(f"Wheel install failed for {pkg}; trying sdist build on Blackwell...")
            run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-index",
                    f"--find-links={wheels_dir}",
                    "--no-build-isolation",
                    pkg,
                ]
            )


def copy_code(bundle_code: Path, working_code: Path) -> None:
    if working_code.exists():
        shutil.rmtree(working_code)
    shutil.copytree(bundle_code, working_code)
    print(f"Code copied to {working_code}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Blackwell environment setup")
    parser.add_argument("--bundle-root", type=str, default=None)
    parser.add_argument("--working-code", type=str, default="/kaggle/working/nemotron_kaggle")
    args = parser.parse_args()

    bundle_root = find_bundle_root(args.bundle_root)
    wheels_dir = bundle_root / "wheels"
    model_dir = bundle_root / "nemotron-base"
    code_dir = bundle_root / "nemotron_kaggle"
    working_code = Path(args.working_code)

    print(f"Using bundle: {bundle_root}")
    apply_triton_fix()
    copy_code(code_dir, working_code)
    install_offline(wheels_dir)

    env_file = Path("/kaggle/working/blackwell_env.sh")
    env_file.write_text(
        "\n".join(
            [
                f'export NEMOTRON_BUNDLE_ROOT="{bundle_root}"',
                f'export NEMOTRON_MODEL_PATH="{model_dir}"',
                f'export NEMOTRON_CODE_PATH="{working_code}"',
                'export TRITON_PTXAS_PATH="/tmp/bin/ptxas"',
            ]
        ),
        encoding="utf-8",
    )

    print("\nOffline setup complete.")
    print("IMPORTANT: Restart kernel now (Run -> Restart Kernel).")
    print(f"Model path: {model_dir}")
    print(f"Code path:  {working_code}")
    print(f"Saved env:  {env_file}")


if __name__ == "__main__":
    main()
