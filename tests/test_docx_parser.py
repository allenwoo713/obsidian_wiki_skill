"""DocxParser 测试：用 python-docx 构造测试 docx。"""
import sys
import hashlib
import io
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from parsers.docx_parser import DocxParser
from parsers.base import ParseResult


def _valid_png_bytes(r=0, g=0, b=0):
    """用 Pillow 生成有效 PNG（基准彩图，双色）用于 md5/SHA 区分。"""
    from PIL import Image
    img = Image.new("RGB", (4, 4), color=(r, g, b))
    # 加对比像素确保编码内容不同
    img.putpixel((1, 1), (255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


_DEFAULT_PNG = _valid_png_bytes(0, 100, 200)


def _make_docx_with_image(tmp_path, image_bytes=None, caption="图1 测试示意图"):
    """用 python-docx 构造含一张图片+图注的测试 docx。"""
    from docx import Document
    from docx.shared import Inches
    if image_bytes is None:
        image_bytes = _DEFAULT_PNG
    doc = Document()
    doc.add_paragraph("文档开头正文。")
    img_path = tmp_path / "test_img.png"
    img_path.write_bytes(image_bytes)
    doc.add_picture(str(img_path), width=Inches(1))
    if caption:
        doc.add_paragraph(caption)
    doc.add_paragraph("图片后续正文。")
    docx_path = tmp_path / "test.docx"
    doc.save(str(docx_path))
    return docx_path, image_bytes


def test_docx_parser_returns_parse_result(tmp_path):
    docx_path, _ = _make_docx_with_image(tmp_path)
    parser = DocxParser()
    result = parser.parse(docx_path)
    assert isinstance(result, ParseResult)
    assert "文档开头正文" in result.text
    assert "图片后续正文" in result.text


def test_docx_parser_extracts_image(tmp_path):
    img_bytes = _valid_png_bytes(100, 0, 0)
    docx_path, _ = _make_docx_with_image(tmp_path, image_bytes=img_bytes)
    parser = DocxParser()
    result = parser.parse(docx_path)
    assert len(result.images) == 1
    ref = result.images[0]
    # 文件名含 _img
    assert "_img" in ref.filename or "img01" in ref.filename
    assert ref.rel_path.startswith("assets/")
    assert len(ref.sha256) == 64
    assert ref.source_media_name.startswith("image")
    assert ref.page_or_section == "body"
    # _image_bytes 与 images 同序
    assert len(result._image_bytes) == 1
    assert hashlib.sha256(result._image_bytes[0]).hexdigest() == ref.sha256


def test_docx_parser_image_placeholder_in_text(tmp_path):
    docx_path, _ = _make_docx_with_image(tmp_path, caption="图1 测试示意图")
    parser = DocxParser()
    result = parser.parse(docx_path)
    assert "{{IMG|" in result.text
    assert "图注:" in result.text


def test_docx_parser_caption_extracted(tmp_path):
    docx_path, _ = _make_docx_with_image(tmp_path, caption="图1 方位角 FOV ±45° 示意图")
    parser = DocxParser()
    result = parser.parse(docx_path)
    assert len(result.images) == 1
    assert "图1" in result.images[0].caption
    assert "±45°" in result.images[0].caption


def test_docx_parser_no_caption(tmp_path):
    docx_path, _ = _make_docx_with_image(tmp_path, caption="")
    parser = DocxParser()
    result = parser.parse(docx_path)
    assert len(result.images) == 1
    assert result.images[0].caption == ""


def test_docx_parser_multiple_images(tmp_path):
    """构造含两张图片的 docx。"""
    from docx import Document
    from docx.shared import Inches
    img1_bytes = _valid_png_bytes(50, 0, 0)
    img2_bytes = _valid_png_bytes(0, 50, 0)
    doc = Document()
    doc.add_paragraph("正文1")
    img1 = tmp_path / "i1.png"
    img1.write_bytes(img1_bytes)
    doc.add_picture(str(img1), width=Inches(1))
    doc.add_paragraph("图1 第一张")
    img2 = tmp_path / "i2.png"
    img2.write_bytes(img2_bytes)
    doc.add_picture(str(img2), width=Inches(1))
    doc.add_paragraph("图2 第二张")
    docx_path = tmp_path / "multi.docx"
    doc.save(str(docx_path))
    parser = DocxParser()
    result = parser.parse(docx_path)
    assert len(result.images) == 2
    assert result.images[0].sha256 != result.images[1].sha256
