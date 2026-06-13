#!/usr/bin/env python3
"""Minimal Blackwell setup using only Kaggle notebook inputs (no wheel bootstrap).

Expects:
  - Nemotron model attached via Add Input → Models (or nemotron-base in a dataset)
  - Training code under /kaggle/input/<dataset>/nemotron_kaggle/ (code-only dataset)

Copies code to /kaggle/working and prints paths. Does NOT pip-install anything —
use this when the Kaggle Blackwell image already has transformers/peft/trl/mamba-ssm.

  python scripts/setup_kaggle_inputs.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


def find_model() -> Path | None:
    for cfg in Path("/kaggle/input").rglob("config.json"):
        parent = cfg.parent
        if (parent / "model.safetensors.index.json").exists() or any(
            parent.glob("model-*.safetensors")
        ):
            return parent
    for entry in sorted(Path("/kaggle/input").iterdir()):
        for candidate in (
            entry / "nemotron-base",
            entry / "nemotron-blackwell-offline" / "nemotron-base",
        ):
            if (candidate / "config.json").exists():
                return candidate
    return None


def find_code() -> Path | None:
    for entry in sorted(Path("/kaggle/input").iterdir()):
        for candidate in (
            entry / "nemotron_kaggle",
            entry / "nemotron-kaggle-code" / "nemotron_kaggle",
            entry / "nemotron-blackwell-offline" / "nemotron_kaggle",
        ):
            if (candidate / "src" / "blackwell_env.py").exists():
                return candidate
    # Already running from a cloned working copy
    local = Path("/kaggle/working/nemotron_kaggle")
    if (local / "src" / "blackwell_env.py").exists():
        return local
    return None


def check_imports() -> list[str]:
    missing = []
    for pkg in ("torch", "transformers", "peft", "trl", "datasets", "mamba_ssm"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


def main() -> None:
    parser = argparse.ArgumentParser(description="Setup from Kaggle inputs only")
    parser.add_argument("--working-code", default="/kaggle/working/nemotron_kaggle")
    args = parser.parse_args()

    model = find_model()
    code = find_code()
    working = Path(args.working_code)

    print("=== Kaggle inputs setup (no wheel install) ===")
    print(f"Model:  {model or 'NOT FOUND'}")
    print(f"Code:   {code or 'NOT FOUND'}")

    if model is None:
        print("\nAttach the competition model:")
        print("  Add Input → Models → metric/nemotron-3-nano-30b-a3b-bf16")
        sys.exit(1)

    if code is None:
        print("\nAttach a code-only dataset containing nemotron_kaggle/ (src/, scripts/).")
        print("Or clone the repo on a T4 notebook and publish src/ as a private dataset.")
        sys.exit(1)

    if code != working:
        if working.exists():
            shutil.rmtree(working)
        shutil.copytree(code, working)
        print(f"Copied code → {working}")
    else:
        print(f"Using code at {working}")

    missing = check_imports()
    if missing:
        print(f"\nWARNING: missing packages: {missing}")
        print("If training fails on import, use the full bootstrap path in kaggle_guide.md.")
    else:
        print("\nCore packages import OK.")

    # Apply Triton fix early (safe before restart)
    sys.path.insert(0, str(working / "src"))
    import blackwell_env  # noqa: E402

    blackwell_env.apply()
    print(f"\nReady. Model path:  {model}")
    print(f"       Code path:   {working}")
    print("Next: PYTHONPATH=src python3 src/smoke_blackwell.py")


if __name__ == "__main__":
    main()
