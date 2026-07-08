import sys
import types
from pathlib import Path
from typing import List

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from extract_assets import extract, UnsupportedFormat
from models import ImageRef
from parsers.base import ParseResult


def _make_fake_module(parser_name: str, marker: str):
    """Create a fake parsers.<name> module with a Parser class returning a marked ParseResult."""
    module = types.ModuleType(f"parsers.{parser_name}")

    class Parser:
        def __init__(self, *args, **kwargs):
            pass

        def parse(self, path: Path) -> ParseResult:
            return ParseResult(text=f"{marker}:{path.suffix}", images=[], tables=[])

    module.__dict__[f"{parser_name.replace('_pdf', '').replace('_parser', '').replace('_', ' ').title().replace(' ', '')}Parser"] = Parser
    # 统一暴露为 Parser
    module.Parser = Parser
    return module


def _make_fake_docx_module():
    module = types.ModuleType("parsers.docx_parser")

    class DocxParser:
        def parse(self, path: Path) -> ParseResult:
            return ParseResult(text="docx parser", images=[], tables=[])

    module.DocxParser = DocxParser
    return module


def _make_fake_cloud_module():
    module = types.ModuleType("parsers.mineru_cloud")

    class MineruCloudParser:
        def __init__(self, *args, **kwargs):
            pass

        def parse(self, path: Path) -> ParseResult:
            return ParseResult(text="mineru_cloud", images=[], tables=[])

    module.MineruCloudParser = MineruCloudParser
    return module


def _make_fake_local_module():
    module = types.ModuleType("parsers.mineru_local")

    class MineruLocalPdfParser:
        def __init__(self, *args, **kwargs):
            pass

        def parse(self, path: Path) -> ParseResult:
            return ParseResult(text="mineru_local", images=[], tables=[])

    module.MineruLocalPdfParser = MineruLocalPdfParser
    return module


@pytest.fixture(autouse=True)
def _patch_parsers(monkeypatch):
    """Replace all parser modules with fakes so tests don't need real dependencies."""
    monkeypatch.setitem(sys.modules, "parsers.docx_parser", _make_fake_docx_module())
    monkeypatch.setitem(sys.modules, "parsers.mineru_cloud", _make_fake_cloud_module())
    monkeypatch.setitem(sys.modules, "parsers.mineru_local", _make_fake_local_module())


def test_docx_uses_local_parser(tmp_path: Path):
    docx_path = tmp_path / "test.docx"
    docx_path.write_bytes(b"fake docx")
    assets_dir = tmp_path / "assets"

    result = extract(docx_path, assets_dir)
    assert result.doc_type == "docx"
    assert result.text == "docx parser"


def test_pptx_non_sensitive_uses_cloud(tmp_path: Path):
    pptx_path = tmp_path / "test.pptx"
    pptx_path.write_bytes(b"fake pptx")
    assets_dir = tmp_path / "assets"

    result = extract(pptx_path, assets_dir, is_sensitive=False)
    assert result.doc_type == "pptx"
    assert result.text == "mineru_cloud"


def test_pptx_sensitive_uses_local(tmp_path: Path):
    pptx_path = tmp_path / "test.pptx"
    pptx_path.write_bytes(b"fake pptx")
    assets_dir = tmp_path / "assets"

    result = extract(pptx_path, assets_dir, is_sensitive=True)
    assert result.doc_type == "pptx"
    assert result.text == "mineru_local"


def test_pdf_default_uses_cloud(tmp_path: Path):
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"fake pdf")
    assets_dir = tmp_path / "assets"

    result = extract(pdf_path, assets_dir, is_sensitive=False)
    assert result.doc_type == "pdf"
    assert result.text == "mineru_cloud"


def test_pdf_sensitive_uses_local(tmp_path: Path):
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"fake pdf")
    assets_dir = tmp_path / "assets"

    result = extract(pdf_path, assets_dir, is_sensitive=True)
    assert result.doc_type == "pdf"
    assert result.text == "mineru_local"


def test_pdf_sensitive_env_var(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MINERU_PDF_SENSITIVE", "1")
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"fake pdf")
    assets_dir = tmp_path / "assets"

    result = extract(pdf_path, assets_dir)
    assert result.text == "mineru_local"


def test_xlsx_uses_cloud(tmp_path: Path):
    xlsx_path = tmp_path / "test.xlsx"
    xlsx_path.write_bytes(b"fake xlsx")
    assets_dir = tmp_path / "assets"

    result = extract(xlsx_path, assets_dir)
    assert result.doc_type == "xlsx"
    assert result.text == "mineru_cloud"


def test_html_uses_cloud(tmp_path: Path):
    html_path = tmp_path / "test.html"
    html_path.write_bytes(b"<html></html>")
    assets_dir = tmp_path / "assets"

    result = extract(html_path, assets_dir)
    assert result.doc_type == "html"
    assert result.text == "mineru_cloud"


def test_unsupported_format_raises(tmp_path: Path):
    txt_path = tmp_path / "test.txt"
    txt_path.write_bytes(b"plain text")
    assets_dir = tmp_path / "assets"

    with pytest.raises(UnsupportedFormat):
        extract(txt_path, assets_dir)


def test_image_deduplication_by_sha256(tmp_path: Path):
    """Two images with same sha256 should share one file on disk."""
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_bytes(b"fake pdf")
    assets_dir = tmp_path / "assets"

    img_bytes = b"same image bytes"
    images: List[ImageRef] = [
        ImageRef(filename="img1.png", rel_path="assets/img1.png", caption="", source_media_name="a", sha256="abc", page_or_section=""),
        ImageRef(filename="img2.png", rel_path="assets/img2.png", caption="", source_media_name="b", sha256="abc", page_or_section=""),
    ]

    cloud_module = sys.modules["parsers.mineru_cloud"]

    class CloudParserWithDup:
        def parse(self, path: Path) -> ParseResult:
            return ParseResult(text="", images=images, tables=[], _image_bytes=[img_bytes, img_bytes])

    cloud_module.MineruCloudParser = CloudParserWithDup

    result = extract(pdf_path, assets_dir)
    assert len(result.images) == 2
    assert result.images[0].filename == result.images[1].filename
    assert len(list(assets_dir.iterdir())) == 1
