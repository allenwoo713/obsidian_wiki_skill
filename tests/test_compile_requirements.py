"""Regression tests for the generated requirements lock contract."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parent.parent / "scripts" / "compile_requirements.py"


def _load_lock_module():
    spec = importlib.util.spec_from_file_location("compile_requirements_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_compile_uses_a_fresh_output_path_not_the_existing_lock(
    monkeypatch, tmp_path: Path
) -> None:
    lock = _load_lock_module()
    source = tmp_path / "requirements.in"
    source.write_text("demo>=1\n", encoding="utf-8")
    output = tmp_path / "requirements.txt"
    output.write_text("stale==0\n", encoding="utf-8")
    monkeypatch.setattr(lock, "INPUT", source)
    monkeypatch.setattr(lock, "OUTPUT", output)
    monkeypatch.setattr(
        lock,
        "_uv_command",
        lambda: [
            sys.executable,
            "-c",
            (
                "from pathlib import Path; import sys; "
                "Path(sys.argv[sys.argv.index('--output-file') + 1])"
                ".write_text('fresh==1\\n', encoding='utf-8')"
            ),
        ],
    )

    lock._compile(output)

    assert output.read_text(encoding="utf-8") == (
        f"{lock.INPUT_HASH_PREFIX}{lock._input_hash()}\nfresh==1\n"
    )


def test_check_uses_embedded_input_fingerprint_without_recompiling(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    lock = _load_lock_module()
    source = tmp_path / "requirements.in"
    source.write_text("demo>=1\n", encoding="utf-8")
    output = tmp_path / "requirements.txt"
    monkeypatch.setattr(lock, "INPUT", source)
    monkeypatch.setattr(lock, "OUTPUT", output)
    output.write_text(f"{lock.INPUT_HASH_PREFIX}{lock._input_hash()}\ndemo==1\n", encoding="utf-8")
    monkeypatch.setattr(lock, "_compile", lambda _output: pytest.fail("check must stay offline"))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--check"])

    lock.main()

    assert "matches the current requirements.in fingerprint" in capsys.readouterr().out
