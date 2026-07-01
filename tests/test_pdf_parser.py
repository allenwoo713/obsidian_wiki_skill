"""PdfParser 测试：用 fitz 构造测试 PDF。"""
import sys
import hashlib
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def _minimal_valid_png():
    """生成最小合法 1x1 黑色 PNG。"""
    import struct, zlib
    def chunk(chunk_type, data):
        c = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + c + crc
    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)  # 1x1, 8-bit RGB
    idat = zlib.compress(b"\x00\x00\x00\x00")  # filter byte + RGB
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


def _make_pdf_with_image(tmp_path, caption="Figure 1 Test Diagram"):
    """用 PyMuPDF 构造含图片的 PDF。"""
    import fitz
    img_bytes = _minimal_valid_png()
    img_path = tmp_path / "test.png"
    img_path.write_bytes(img_bytes)
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Document body text.", fontsize=12)
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
    assert "Document body text" in result.text
    assert len(result.images) >= 1
    # 无表格 PDF：tables 应为空，且文本中不出现 [表格] 标记
    assert result.tables == []
    assert "[表格]" not in result.text


def test_pdf_parser_image_sha256(tmp_path):
    pdf_path = _make_pdf_with_image(tmp_path)
    from parsers.pdf_parser import PdfParser
    result = PdfParser().parse(pdf_path)
    for ref in result.images:
        assert len(ref.sha256) == 64
        assert ref.filename.endswith(".png") or ref.filename.endswith(".jpg")
        assert ref.page_or_section.startswith("page")


def test_pdf_parser_caption_extracted(tmp_path):
    pdf_path = _make_pdf_with_image(tmp_path, caption="Figure 1 FOV Diagram")
    from parsers.pdf_parser import PdfParser
    result = PdfParser().parse(pdf_path)
    assert any("Figure" in img.caption for img in result.images)


def test_pdf_parser_placeholder_in_text(tmp_path):
    pdf_path = _make_pdf_with_image(tmp_path)
    from parsers.pdf_parser import PdfParser
    result = PdfParser().parse(pdf_path)
    assert "{{IMG|" in result.text


def _make_pdf_with_table(tmp_path):
    """用 reportlab 构造含表格的 PDF。"""
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Table, Paragraph
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.platypus.tables import TableStyle
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(tmp_path / "tbl.pdf"), pagesize=letter)
    table = Table([
        ["Field", "Offset", "Length"],
        ["Header", "0", "4"],
        ["Payload", "4", "N"],
    ], colWidths=[100, 80, 80])
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.black),
    ]))
    doc.build([
        Paragraph("UDP Frame Format", styles["Normal"]),
        table,
    ])
    return tmp_path / "tbl.pdf"


def test_pdf_parser_extracts_tables(tmp_path):
    pdf_path = _make_pdf_with_table(tmp_path)
    from parsers.pdf_parser import PdfParser
    result = PdfParser().parse(pdf_path)
    assert len(result.tables) >= 1
    flat = [cell for row in result.tables[0] for cell in row]
    assert any("Field" in c or "Header" in c or "Offset" in c for c in flat)


def test_pdf_parser_tables_in_text(tmp_path):
    pdf_path = _make_pdf_with_table(tmp_path)
    from parsers.pdf_parser import PdfParser
    result = PdfParser().parse(pdf_path)
    assert "[表格]" in result.text
    assert "表 1:" in result.text


def _make_pdf_with_two_tables(tmp_path):
    """用 reportlab 构造含 2 个表格的 PDF。"""
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Table, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from reportlab.platypus.tables import TableStyle
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(str(tmp_path / "two_tbl.pdf"), pagesize=letter)

    def make_table(data):
        table = Table(data, colWidths=[100, 80, 80])
        table.setStyle(TableStyle([
            ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.black),
        ]))
        return table

    story = [
        Paragraph("First Section", styles["Normal"]),
        make_table([
            ["Field", "Offset", "Length"],
            ["Header", "0", "4"],
            ["Payload", "4", "N"],
        ]),
        Spacer(1, 24),
        Paragraph("Second Section", styles["Normal"]),
        make_table([
            ["Name", "Type", "Desc"],
            ["id", "int", "primary key"],
            ["ts", "float", "timestamp"],
        ]),
    ]
    doc.build(story)
    return tmp_path / "two_tbl.pdf"


def test_pdf_parser_multiple_tables(tmp_path):
    pdf_path = _make_pdf_with_two_tables(tmp_path)
    from parsers.pdf_parser import PdfParser
    result = PdfParser().parse(pdf_path)
    assert len(result.tables) >= 2
    assert "表 1:" in result.text
    assert "表 2:" in result.text
