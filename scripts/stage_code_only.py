#!/usr/bin/env python3
"""Package repo code for a one-time Kaggle dataset upload (no model, no wheels).

Run on T4 with Internet ON, then publish /kaggle/working/nemotron-kaggle-code
as a private dataset and attach it on the Blackwell notebook.

  python scripts/stage_code_only.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_ROOT = Path("/kaggle/temp/nemotron-kaggle-code")
DEFAULT_REPO = "https://github.com/gbose01/nemotron_kaggle.git"


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    root = DEFAULT_ROOT
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)

    dest = root / "nemotron_kaggle"
    run(["git", "clone", "--depth", "1", DEFAULT_REPO, str(dest)])

    manifest = root / "README.txt"
    manifest.write_text(
        "\n".join(
            [
                "Code-only bundle for Blackwell smoke test.",
                "Attach on Blackwell notebook alongside the competition model.",
                "Contents: nemotron_kaggle/src, scripts, data/",
            ]
        ),
        encoding="utf-8",
    )

    print(f"\nDone. Upload this folder as a Kaggle dataset:\n  {root}")
    print("Suggested name: nemotron-kaggle-code")


if __name__ == "__main__":
    main()
