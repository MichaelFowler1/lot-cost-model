"""Bundle the tool into one runnable file.

Somewhere that blocks executables will usually still run Python, so this
packages the three modules into a single ``.pyz``, which is a plain zip
archive rather than a compiled binary. A colleague copies one file and runs
it with the Python they already have:

    python lot-cost-model.pyz

What this does not do is bundle numpy, pandas or openpyxl. It solves
"several files and a git clone", not "no dependencies". Whoever runs it still
needs those three installed, and cost_core as well if they want the risk half.

Usage:
    python tools/build_pyz.py [--out DIR]
"""

from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
import tempfile
import zipapp

MODULES = ("lot_cost_model.py", "risk.py", "wbs.py")

ENTRY = '''"""Entry point when the tool runs as a single .pyz archive."""
import sys

from lot_cost_model import main

sys.exit(main())
'''


def build(root: pathlib.Path, out_dir: pathlib.Path) -> pathlib.Path:
    missing = [m for m in MODULES if not (root / m).exists()]
    if missing:
        raise SystemExit(
            f"Cannot build: {missing} not found in {root}. Run this from the "
            "repository, or pass the right path."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "lot-cost-model.pyz"

    with tempfile.TemporaryDirectory() as tmp:
        staging = pathlib.Path(tmp) / "app"
        staging.mkdir()
        for name in MODULES:
            shutil.copy2(root / name, staging / name)
        (staging / "__main__.py").write_text(ENTRY, encoding="utf-8")
        zipapp.create_archive(staging, target)

    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default="dist", help="where to put the archive"
    )
    args = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parent.parent
    target = build(root, pathlib.Path(args.out))
    size = target.stat().st_size

    print(f"Built {target} ({size:,} bytes)")
    print()
    print("To use it, copy that one file and run:")
    print("    python lot-cost-model.pyz")
    print()
    print("It still needs numpy, pandas and openpyxl on the machine that")
    print("runs it, and cost_core for the risk analysis.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
