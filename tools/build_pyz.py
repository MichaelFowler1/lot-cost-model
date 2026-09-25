# Copyright 2026 Michael Fowler
# SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""Bundle the tool into one runnable file.

Somewhere that blocks executables will usually still run Python, so this
packages the tool into a single ``.pyz``, which is a plain zip archive rather
than a compiled binary. A colleague copies one file and runs it with the
Python they already have:

    python lot-cost-model.pyz

The archive carries the window (``lot_cost_model.py`` and ``risk.py``) and a
copy of the ``cost_core`` package, because the engine, the WBS roll-up and the
workbook writers live in the library now and there is no fallback left in the
window. Vendoring it is what keeps the archive a single file: asking a
colleague to ``pip install`` from a private git remote before they can open
the window would undo the point of building one.

The copy comes from whichever ``cost_core`` the interpreter running this
script imports, found with :func:`importlib.util.find_spec` rather than by
guessing at a sibling checkout. That way the archive carries the version the
build environment actually holds, and the version stamped into every workbook
it writes is the version that did the arithmetic.

What this does not bundle is numpy, pandas, openpyxl and scipy. They are
compiled, so they cannot be zipped up and imported out of an archive, and
scipy is not optional: the intervals and the Monte Carlo are built on its
quantiles. matplotlib is not bundled either, and is not needed. It became an
optional extra of the library in 1.0.0 and nothing the window does requires
it. So this solves "several files and a git clone", not "no dependencies".
Whoever runs the archive still needs those four installed.

It also carries the licences of both halves. Passing the archive on passes on
the window and a copy of the library, and both licences oblige whoever does
that to pass their terms on with it, so the window's LICENSE and NOTICE go at
the top of the archive and the library's go beside its copy of ``cost_core``.
A library copy whose licence cannot be found stops the build, rather than going
out without it.

Usage:
    python tools/build_pyz.py [--out DIR]
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import shutil
import sys
import tempfile
import zipapp

#: The window's own modules. The engine used to be a third file here; it is
#: in the vendored library now.
MODULES = ("lot_cost_model.py", "risk.py")

#: Directory names never worth carrying. ``__pycache__`` holds bytecode for
#: one interpreter version and the archive has to run on several, ``tests``
#: is the library's own suite, and the rest is editor and tooling litter.
SKIP_DIRS = frozenset(
    {"__pycache__", "tests", "test", ".pytest_cache", ".mypy_cache", ".git"}
)

#: Suffixes never worth carrying. Compiled bytecode is version-locked and
#: the source sitting beside it is what actually gets imported.
SKIP_SUFFIXES = (".pyc", ".pyo", ".pyd", ".so", ".orig", ".rej", ".swp")

#: The licence files that travel with the window's half of the archive.
LICENSE_FILES = ("LICENSE", "NOTICE")

#: The library's, which from cost-core 2.3.1 include the additional permission
#: for Government Work. It widens the library's licence, and the permission
#: asks for it to go with any copy handed on, so the archive carries it too
#: when the vendored library has it.
LIBRARY_LICENSE_FILES = ("LICENSE", "NOTICE", "LICENSE-GOVERNMENT-WORK.md")

ENTRY = '''"""Entry point when the tool runs as a single .pyz archive."""
import sys

from lot_cost_model import main

sys.exit(main())
'''


def _skip(path: pathlib.Path) -> bool:
    """Is this file litter rather than library source?"""
    return path.suffix in SKIP_SUFFIXES or path.name.startswith(".")


def locate_cost_core() -> pathlib.Path:
    """Where the library this interpreter imports actually lives.

    ``find_spec`` answers for the environment rather than for the filesystem,
    so an editable install, a wheel in site-packages and a checkout on
    PYTHONPATH all give the right directory. Guessing at ``../cost_core``
    would silently vendor a stale sibling checkout instead.
    """
    spec = importlib.util.find_spec("cost_core")
    if spec is None or not spec.submodule_search_locations:
        raise SystemExit(
            "Cannot build: cost_core is not importable from this Python "
            f"({sys.executable}). The archive vendors the library, so it has "
            "to be installed here first:\n"
            "    pip install -r requirements.txt"
        )
    return pathlib.Path(list(spec.submodule_search_locations)[0]).resolve()


def library_licence_files(package: pathlib.Path) -> list[pathlib.Path]:
    """The licence files of the cost_core being vendored.

    A source checkout is read directly: its LICENSE and NOTICE sit beside its
    pyproject.toml. That has to win over the installed metadata, because an
    editable install copies the licence into its dist-info once, at install
    time, so a checkout that has changed licence since would vendor the old
    terms. Anything else, a wheel in site-packages included, is read from the
    distribution's own metadata, which is the licence that version shipped
    with.
    """
    checkout = package.parent
    pyproject = checkout / "pyproject.toml"
    found: dict[str, pathlib.Path] = {}
    if pyproject.is_file() and 'name = "cost-core"' in pyproject.read_text(encoding="utf-8"):
        for name in LIBRARY_LICENSE_FILES:
            if (checkout / name).is_file():
                found[name] = checkout / name
    else:
        from importlib.metadata import PackageNotFoundError, distribution

        try:
            dist = distribution("cost-core")
        except PackageNotFoundError:
            dist = None
        for entry in (dist.files or ()) if dist is not None else ():
            if entry.name in LIBRARY_LICENSE_FILES and entry.name not in found:
                path = pathlib.Path(dist.locate_file(entry))
                if path.is_file():
                    found[entry.name] = path
    if "LICENSE" not in found:
        raise SystemExit(
            f"Cannot build: no LICENSE found for the cost_core at {package}. "
            "The archive carries a copy of the library, and a copy that goes "
            "out without its licence breaks the terms it is distributed under."
        )
    return list(found.values())


def _vendor(package: pathlib.Path, staging: pathlib.Path) -> int:
    """Copy the package into the staging directory. Returns the file count."""
    copied = 0
    for src in sorted(package.rglob("*")):
        rel = src.relative_to(package)
        if any(part in SKIP_DIRS for part in rel.parts):
            continue
        dest = staging / package.name / rel
        if src.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
            continue
        if _skip(src):
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied += 1
    if copied == 0:
        raise SystemExit(f"Cannot build: nothing to vendor from {package}.")
    return copied


def build(root: pathlib.Path, out_dir: pathlib.Path) -> pathlib.Path:
    missing = [m for m in (*MODULES, *LICENSE_FILES) if not (root / m).exists()]
    if missing:
        raise SystemExit(
            f"Cannot build: {missing} not found in {root}. Run this from the "
            "repository, or pass the right path."
        )

    package = locate_cost_core()

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "lot-cost-model.pyz"

    with tempfile.TemporaryDirectory() as tmp:
        staging = pathlib.Path(tmp) / "app"
        staging.mkdir()
        for name in (*MODULES, *LICENSE_FILES):
            shutil.copy2(root / name, staging / name)
        _vendor(package, staging)
        for path in library_licence_files(package):
            shutil.copy2(path, staging / package.name / path.name)
        (staging / "__main__.py").write_text(ENTRY, encoding="utf-8")
        # Compressed, not stored. The archive is mostly Python source, which
        # deflates to roughly a third, and it is meant to be copied around by
        # people on a corporate share.
        zipapp.create_archive(staging, target, compressed=True)

    return target


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default="dist", help="where to put the archive"
    )
    args = parser.parse_args()

    root = pathlib.Path(__file__).resolve().parent.parent
    package = locate_cost_core()
    target = build(root, pathlib.Path(args.out))
    size = target.stat().st_size

    import cost_core  # importable: locate_cost_core just found its spec

    vendored = cost_core.__version__

    print(f"Built {target} ({size:,} bytes)")
    print(f"Vendored cost_core {vendored} from {package}")
    print()
    print("To use it, copy that one file and run:")
    print("    python lot-cost-model.pyz")
    print()
    print("The library rides along inside the archive. What the machine that")
    print("runs it still needs is numpy, pandas, openpyxl and scipy, because")
    print("those are compiled and cannot be imported out of a zip. matplotlib")
    print("is not needed: it is an optional extra of the library and nothing")
    print("the window does asks for it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
