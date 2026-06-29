"""parse_sources.py 测试。"""
import hashlib
from pathlib import Path
import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from models import ParsedDoc
from parse_sources import parse_docx, parse_pdf, parse_markdown, parse_file, compute_sha256


def test_compute_sha256(tmp_path):
    f = tmp_path / "x.txt"
    f.write_text("hello", encoding="utf-8")
    expected = hashlib.sha256(b"hello").hexdigest()
    assert compute_sha256(f) == expected


def test_parse_markdown(tmp_path):
    f = tmp_path / "note.md"
    f.write_text("# Title\n\n正文段落。", encoding="utf-8")
    doc = parse_markdown(f)
    assert doc.doc_type == "md"
    assert doc.title == "Title"
    assert "正文段落" in doc.text
    assert doc.tables == []
    assert len(doc.sha256) == 64


def test_parse_file_dispatch(tmp_path):
    f = tmp_path / "n.md"
    f.write_text("# H\nbody", encoding="utf-8")
    doc = parse_file(f)
    assert doc.title == "H"


def test_parse_file_unsupported(tmp_path):
    f = tmp_path / "x.xyz"
    f.write_text("?", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported"):
        parse_file(f)
