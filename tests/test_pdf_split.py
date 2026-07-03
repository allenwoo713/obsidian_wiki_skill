from pathlib import Path
import sys

import fitz
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from parsers.pdf_split import split_pdf_into_chunks


def _make_pdf(tmp_path: Path, page_count: int) -> Path:
    pdf_path = tmp_path / "source.pdf"
    doc = fitz.open()
    for _ in range(page_count):
        page = doc.new_page()
        page.insert_text((50, 50), "page marker")
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def test_no_split_for_short_pdf(tmp_path: Path):
    src = _make_pdf(tmp_path, 5)
    chunks = split_pdf_into_chunks(src, tmp_path / "out", pages_per_chunk=200)
    assert len(chunks) == 1
    assert chunks[0] == src.resolve()


def test_split_long_pdf(tmp_path: Path):
    src = _make_pdf(tmp_path, 250)
    out_dir = tmp_path / "chunks"
    chunks = split_pdf_into_chunks(src, out_dir, pages_per_chunk=100)

    assert len(chunks) == 3
    assert all(p.parent == out_dir.resolve() for p in chunks)

    total = 0
    for p in chunks:
        with fitz.open(str(p)) as doc:
            assert doc.page_count <= 100
            total += doc.page_count
    assert total == 250


def test_chunk_naming(tmp_path: Path):
    src = _make_pdf(tmp_path, 10)
    out_dir = tmp_path / "chunks"
    chunks = split_pdf_into_chunks(src, out_dir, pages_per_chunk=3)

    assert [p.name for p in chunks] == ["chunk_001.pdf", "chunk_002.pdf", "chunk_003.pdf", "chunk_004.pdf"]
