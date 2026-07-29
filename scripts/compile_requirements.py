"""Generate or verify the committed core dependency lock with uv.

``requirements.in`` is the human-reviewed list of direct core dependencies.
``requirements.txt`` is generated output and must not be edited manually.
All uv cache and check-mode output are kept below the skill directory.
"""
from __future__ import annotations

import argparse
import filecmp
import os
import shutil
import subprocess
import sys
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parent.parent
INPUT = SKILL_ROOT / "requirements.in"
OUTPUT = SKILL_ROOT / "requirements.txt"
CHECK_OUTPUT = SKILL_ROOT / ".review-tmp" / "requirements" / "requirements.txt"


def _compile(output: Path) -> None:
    uv = shutil.which("uv")
    if not uv:
        raise SystemExit(
            "uv is required. Install it from https://docs.astral.sh/uv/ or run this "
            "script in CI after astral-sh/setup-uv."
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", str(SKILL_ROOT / ".cache" / "uv"))
    subprocess.run(
        [
            uv, "pip", "compile", str(INPUT), "--output-file", str(output),
            # The default header embeds the output path, so check-mode's
            # repository-local scratch output would never byte-match the
            # committed lock file.
            "--no-header",
            "--quiet",
        ],
        cwd=SKILL_ROOT,
        env=env,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="fail when requirements.txt is not the current generated output",
    )
    args = parser.parse_args()

    if not args.check:
        _compile(OUTPUT)
        print(f"generated {OUTPUT.relative_to(SKILL_ROOT)} from {INPUT.name}")
        return

    _compile(CHECK_OUTPUT)
    if not OUTPUT.exists() or not filecmp.cmp(OUTPUT, CHECK_OUTPUT, shallow=False):
        raise SystemExit(
            "requirements.txt is stale; run `python scripts/compile_requirements.py` "
            "and commit the generated file."
        )
    print("requirements.txt is current")


if __name__ == "__main__":
    main()
