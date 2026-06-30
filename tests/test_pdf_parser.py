"""PdfParser 测试：用 fitz 构造测试 PDF。"""
import sys
import hashlib
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def _make_pdf_with_image(tmp_path, caption="图1 测试示意图"):
    """用 PyMuPDF 构造含图片的 PDF。"""
    import fitz
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "文档开头正文。", fontsize=12)
    img_path = tmp_path / "test.png"
    img_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
    img_path.write_bytes(img_bytes)
    page.insert_image(fitz.Rect(72, 100, 200, 200), filename=str(img_path))
    if caption:
        page.insert_text((72, 220), caption, fontsize=10)
    pdf_path = tmp_path / "test.pdf"
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def test_pdf_parser_returns_parse_result(tmp_path):
    pdf_path = _make_pdf_with_image(tmp_path)
    from parsers.pdf_parser import PdfParser
    result = PdfParser().parse(pdf_path)
    assert "文档开头正文" in result.text
    assert len(result.images) >= 1


def test_pdf_parser_image_sha256(tmp_path):
    pdf_path = _make_pdf_with_image(tmp_path)
    from parsers.pdf_parser import PdfParser
    result = PdfParser().parse(pdf_path)
    for ref in result.images:
        assert len(ref.sha256) == 64
        assert ref.filename.endswith(".png") or ref.filename.endswith(".jpg")
        assert ref.page_or_section.startswith("page")


def test_pdf_parser_caption_extracted(tmp_path):
    pdf_path = _make_pdf_with_image(tmp_path, caption="图1 方位角 FOV ±45° 示意图")
    from parsers.pdf_parser import PdfParser
    result = PdfParser().parse(pdf_path)
    assert any("图1" in img.caption for img in result.images)


def test_pdf_parser_placeholder_in_text(tmp_path):
    pdf_path = _make_pdf_with_image(tmp_path)
    from parsers.pdf_parser import PdfParser
    result = PdfParser().parse(pdf_path)
    assert "{{IMG|" in result.text
