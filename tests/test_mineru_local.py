from pathlib import Path
import sys
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from parsers.mineru_local import MineruLocalPdfParser


_FAKE_PYTHON_EXE = sys.executable  # use a real executable so existence check passes


def _fake_run_factory(md_content: str, image_bytes_map: dict):
    """Return a mock subprocess.run that materializes fake MinerU output files."""
    def fake_run(cmd, **kwargs):
        # cmd: [python_exe, runner_script, input_pdf, output_dir, backend, language]
        _, _, input_pdf, output_dir, _backend, _language = cmd
        output_dir = Path(output_dir)
        stem = Path(input_pdf).stem
        auto_dir = output_dir / stem / "auto"
        auto_dir.mkdir(parents=True, exist_ok=True)
        images_dir = auto_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        md_path = auto_dir / f"{stem}.md"
        md_path.write_text(md_content, encoding="utf-8")

        for name, data in image_bytes_map.items():
            (images_dir / name).write_bytes(data)

        return mock.Mock(returncode=0, stdout="", stderr="")
    return fake_run


def test_parse_success(tmp_path: Path):
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    parser = MineruLocalPdfParser(mineru_python_exe=_FAKE_PYTHON_EXE)
    md_content = "# Title\n\n![](images/abc.jpg)\n"
    fake_run = _fake_run_factory(md_content, {"abc.jpg": b"fake-image-bytes"})

    with mock.patch("parsers.mineru_local.subprocess.run", fake_run):
        result = parser.parse(pdf_path)

    assert len(result.images) == 1
    assert result._image_bytes[0] == b"fake-image-bytes"
    assert "abc.jpg" not in result.text
    assert "{{IMG|" in result.text


def test_parse_subprocess_failure(tmp_path: Path):
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    parser = MineruLocalPdfParser(mineru_python_exe=_FAKE_PYTHON_EXE)

    def fake_run(cmd, **kwargs):
        return mock.Mock(returncode=1, stdout="", stderr="MinerU failed")

    with mock.patch("parsers.mineru_local.subprocess.run", fake_run):
        with pytest.raises(RuntimeError, match="MinerU local parser failed"):
            parser.parse(pdf_path)


def test_missing_mineru_python_raises(tmp_path: Path):
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")

    parser = MineruLocalPdfParser(mineru_python_exe="C:/nonexistent/path/python.exe")

    with pytest.raises(FileNotFoundError):
        parser.parse(pdf_path)
