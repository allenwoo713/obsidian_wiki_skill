"""extract_assets.py 测试：调度与落盘。"""
import io
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from extract_assets import extract, UnsupportedFormat
from models import ParsedDoc


def _valid_png_bytes():
    """用 Pillow 生成有效 PNG 用于测试。"""
    from PIL import Image
    img = Image.new("RGB", (4, 4), color=(0, 100, 200))
    img.putpixel((1, 1), (255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


_DEFAULT_PNG = _valid_png_bytes()


def _make_test_docx(tmp_path, image_bytes=None):
    from docx import Document
    from docx.shared import Inches
    if image_bytes is None:
        image_bytes = _DEFAULT_PNG
    doc = Document()
    doc.add_paragraph("正文")
    img = tmp_path / "t.png"
    img.write_bytes(image_bytes)
    doc.add_picture(str(img), width=Inches(1))
    doc.add_paragraph("图1 示意图")
    docx_path = tmp_path / "t.docx"
    doc.save(str(docx_path))
    return docx_path, img.read_bytes()


def test_extract_returns_parsed_doc(tmp_path):
    docx_path, _ = _make_test_docx(tmp_path)
    assets_dir = tmp_path / "assets"
    parsed = extract(docx_path, assets_dir)
    assert isinstance(parsed, ParsedDoc)
    assert len(parsed.images) == 1
    assert "正文" in parsed.text


def test_extract_writes_image_to_assets(tmp_path):
    docx_path, img_bytes = _make_test_docx(tmp_path)
    assets_dir = tmp_path / "assets"
    parsed = extract(docx_path, assets_dir)
    assert assets_dir.exists()
    written = list(assets_dir.iterdir())
    assert len(written) == 1
    assert written[0].read_bytes() == img_bytes


def test_extract_dedup_same_sha256(tmp_path):
    """同一 docx 两张相同图片只落盘一份。"""
    from docx import Document
    from docx.shared import Inches
    doc = Document()
    img_bytes = _valid_png_bytes()
    img = tmp_path / "same.png"
    img.write_bytes(img_bytes)
    doc.add_picture(str(img), width=Inches(1))
    doc.add_paragraph("图1")
    doc.add_picture(str(img), width=Inches(1))
    doc.add_paragraph("图2")
    docx_path = tmp_path / "dup.docx"
    doc.save(str(docx_path))
    assets_dir = tmp_path / "assets"
    parsed = extract(docx_path, assets_dir)
    assert len(parsed.images) == 2
    written = list(assets_dir.iterdir())
    assert len(written) == 1


def test_extract_unsupported_format(tmp_path):
    f = tmp_path / "x.xyz"
    f.write_text("?", encoding="utf-8")
    with pytest.raises(UnsupportedFormat):
        extract(f, tmp_path / "assets")
