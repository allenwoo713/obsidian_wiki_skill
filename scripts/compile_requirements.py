"""Generate or verify the committed core dependency lock with uv.

``requirements.in`` is the human-reviewed list of direct core dependencies.
``requirements.txt`` is generated output and must not be edited manually.

Generation intentionally uses one pinned uv release.  Verification does *not*
re-resolve PyPI: a lock is a reviewed snapshot, so recompiling unconstrained
transitive ranges on every CI run would make it drift whenever an upstream
package releases.  Instead, the generated file carries the SHA-256 of its
``requirements.in`` input and ``--check`` validates that contract offline.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


SKILL_ROOT = Path(__file__).resolve().parent.parent
INPUT = SKILL_ROOT / "requirements.in"
OUTPUT = SKILL_ROOT / "requirements.txt"
UV_VERSION = "0.12.0"
INPUT_HASH_PREFIX = "# requirements.in-sha256: "


def _input_hash() -> str:
    return hashlib.sha256(INPUT.read_bytes()).hexdigest()


def _uv_command() -> list[str]:
    """Use CI's uv release, falling back to uvx for local reproducibility."""
    uv = shutil.which("uv")
    if uv:
        version = subprocess.run(
            [uv, "--version"], check=True, capture_output=True, text=True
        ).stdout.split()[1]
        if version == UV_VERSION:
            return [uv]

    uvx = shutil.which("uvx")
    if uvx:
        return [uvx, "--from", f"uv=={UV_VERSION}", "uv"]

    raise SystemExit(
        f"uv {UV_VERSION} is required. Install uv or use a uv distribution that "
        "includes uvx so the script can run the pinned compiler."
    )


def _compile(output: Path) -> None:
    """Resolve into a fresh file, then atomically publish the reviewed lock.

    ``uv pip compile --output-file`` treats an existing output file as a
    constraint source.  Compiling directly into ``requirements.txt`` would
    therefore preserve old transitive pins, while CI's fresh check output would
    select newer candidates.  Always use a nonexistent temporary path so both
    paths resolve under the same rules.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("UV_CACHE_DIR", str(SKILL_ROOT / ".cache" / "uv"))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    temporary.unlink()
    try:
        subprocess.run(
            [
                *_uv_command(), "pip", "compile", str(INPUT), "--output-file", str(temporary),
                "--no-header",
                "--quiet",
            ],
            cwd=SKILL_ROOT,
            env=env,
            check=True,
        )
        generated = temporary.read_text(encoding="utf-8")
        temporary.write_text(
            f"{INPUT_HASH_PREFIX}{_input_hash()}\n{generated}", encoding="utf-8"
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _locked_input_hash() -> str | None:
    if not OUTPUT.exists():
        return None
    for line in OUTPUT.read_text(encoding="utf-8").splitlines()[:5]:
        match = re.fullmatch(r"# requirements\.in-sha256: ([0-9a-f]{64})", line)
        if match:
            return match.group(1)
    return None


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

    if _locked_input_hash() != _input_hash():
        raise SystemExit(
            "requirements.txt does not match requirements.in; run "
            "`python scripts/compile_requirements.py` and commit the generated file."
        )
    print("requirements.txt matches the current requirements.in fingerprint")


if __name__ == "__main__":
    main()
