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


def find_competition_model() -> Path | None:
    """Find the attached Kaggle competition / Models input (shortcut path)."""
    for cfg in Path("/kaggle/input").rglob("config.json"):
        parent = cfg.parent
        if (parent / "model.safetensors.index.json").exists() or any(
            parent.glob("model-*.safetensors")
        ):
            return parent
    return None


def find_bundle_parts(explicit: str | None) -> tuple[Path, Path | None, Path]:
    """Locate wheels/, nemotron_kaggle/, and optionally nemotron-base/.

    For the competition-model shortcut, only wheels + code are required at setup
    time; the model is resolved from /kaggle/input/models/... at train time.
    """
    wheels_dir = model_dir = code_dir = None

    def scan(base: Path) -> None:
        nonlocal wheels_dir, model_dir, code_dir
        if not base.exists():
            return
        if wheels_dir is None and (base / "wheels").is_dir():
            wheels_dir = base / "wheels"
        if code_dir is None and (base / "nemotron_kaggle").is_dir():
            code_dir = base / "nemotron_kaggle"
        if model_dir is None and (base / "nemotron-base").is_dir():
            model_dir = base / "nemotron-base"
        if model_dir is None and (
            (base / "model.safetensors.index.json").exists()
            or any(base.glob("model-*.safetensors"))
        ):
            model_dir = base

    if explicit:
        scan(Path(explicit))
        nested = Path(explicit) / "nemotron-blackwell-offline"
        if nested.exists():
            scan(nested)
    else:
        for entry in sorted(Path("/kaggle/input").iterdir()):
            for base in (
                entry,
                entry / "nemotron-blackwell-offline",
                entry / "nemotron-blackwell-deps",
                entry / "nemotron-blackwell-model",
            ):
                scan(base)

    if model_dir is None:
        model_dir = find_competition_model()

    missing = [
        name for name, path in [("wheels", wheels_dir), ("nemotron_kaggle", code_dir)]
        if path is None
    ]
    if missing:
        raise FileNotFoundError(
            "Could not find bootstrap assets under /kaggle/input. Missing: "
            + ", ".join(missing)
            + ". Attach nemotron-blackwell-deps or pass --bundle-root."
        )
    return wheels_dir, model_dir, code_dir


def find_bundle_root(explicit: str | None) -> Path:
    """Legacy helper: return a virtual root when all parts live in one folder."""
    wheels_dir, model_dir, code_dir = find_bundle_parts(explicit)
    if model_dir and wheels_dir.parent == model_dir.parent == code_dir.parent:
        return wheels_dir.parent
    return wheels_dir.parent


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

    wheels_dir, model_dir, code_dir = find_bundle_parts(args.bundle_root)
    working_code = Path(args.working_code)

    print(f"Wheels: {wheels_dir}")
    print(f"Model:  {model_dir}")
    print(f"Code:   {code_dir}")
    apply_triton_fix()
    copy_code(code_dir, working_code)
    install_offline(wheels_dir)

    env_file = Path("/kaggle/working/blackwell_env.sh")
    model_path_line = (
        f'export NEMOTRON_MODEL_PATH="{model_dir}"'
        if model_dir is not None
        else "# NEMOTRON_MODEL_PATH set at train time via kagglehub"
    )
    env_file.write_text(
        "\n".join(
            [
                f'export NEMOTRON_WHEELS_PATH="{wheels_dir}"',
                model_path_line,
                f'export NEMOTRON_CODE_PATH="{working_code}"',
                'export TRITON_PTXAS_PATH="/tmp/bin/ptxas"',
            ]
        ),
        encoding="utf-8",
    )

    print("\nOffline setup complete.")
    print("IMPORTANT: Restart kernel now (Run -> Restart Kernel).")
    print(f"Model path: {model_dir or '(competition model — resolve at train time)'}")
    print(f"Code path:  {working_code}")
    print(f"Saved env:  {env_file}")


if __name__ == "__main__":
    main()
