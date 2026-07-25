"""models.py 测试：ImageRef 与 ParsedDoc.images。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from models import ParsedDoc, ImageRef


def test_image_ref_fields():
    ref = ImageRef(
        filename="acme-visioncam-front-datasheet-v1-6_img03.png",
        rel_path="assets/acme-visioncam-front-datasheet-v1-6_img03.png",
        caption="图3 方位角 FOV ±45° 示意图",
        source_media_name="image3.png",
        sha256="a1b2c3" * 21 + "a1",
        page_or_section="body",
    )
    assert ref.filename.endswith("_img03.png")
    assert ref.rel_path.startswith("assets/")
    assert "±45°" in ref.caption


def test_parsed_doc_has_images_field():
    doc = ParsedDoc(
        path=Path("/tmp/x.docx"), title="T", text="body", tables=[],
        sha256="abc", doc_type="docx",
    )
    assert doc.images == []


def test_parsed_doc_with_images():
    ref = ImageRef(
        filename="x_img01.png", rel_path="assets/x_img01.png",
        caption="", source_media_name="image1.png",
        sha256="d" * 64, page_or_section="body",
    )
    doc = ParsedDoc(
        path=Path("/tmp/x.docx"), title="T", text="body", tables=[],
        sha256="abc", doc_type="docx", images=[ref],
    )
    assert len(doc.images) == 1
    assert doc.images[0].filename == "x_img01.png"
