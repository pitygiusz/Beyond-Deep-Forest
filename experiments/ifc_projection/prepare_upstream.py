"""Build the clean Deep Forest clone without changing its source checkout."""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "upstream" / "deep_forest_py310"
BUILD = ROOT / ".build" / "deep_forest_py310"
EXPECTED_REVISION = "1a724924a01d5dd966425dde9c38d99746a4039e"


def main() -> None:
    if not (SOURCE / ".git").is_dir():
        raise FileNotFoundError(f"Clone Deep Forest first: {SOURCE}")
    revision = subprocess.run(
        ["git", "-C", str(SOURCE), "rev-parse", "HEAD"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    if revision != EXPECTED_REVISION:
        raise RuntimeError(f"Expected Deep Forest revision {EXPECTED_REVISION}, found {revision}")

    BUILD.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SOURCE / "deepforest", BUILD / "deepforest", dirs_exist_ok=True)
    for name in ("setup.py", "README.rst"):
        shutil.copy2(SOURCE / name, BUILD / name)

    # The upstream fork targets an older NumPy API. NumPy 2.5 accepts `method`
    # and rejects the former `interpolation` keyword. Patch the build only.
    binner = BUILD / "deepforest" / "_binner.py"
    content = binner.read_text()
    old = 'interpolation="midpoint"'
    if content.count(old) != 1:
        raise RuntimeError("Unexpected upstream binner; inspect its percentile call")
    binner.write_text(content.replace(old, 'method="midpoint"'))

    subprocess.run(
        [sys.executable, "setup.py", "build_ext", "--inplace"],
        cwd=BUILD,
        check=True,
    )
    print(f"Built Deep Forest from the fresh clone in {BUILD}")


if __name__ == "__main__":
    main()
