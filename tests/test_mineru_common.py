import hashlib
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from parsers.mineru_common import mineru_markdown_to_parse_result

def test_extract_image_refs_from_markdown():
    md = "# Title\n\n![](images/abc123.jpg)\n\nMore.\n\n![](images/def456.jpg)\n"
    result = mineru_markdown_to_parse_result(
        markdown=md, doc_slug="test-doc", images_dir=Path("/fake"),
        image_bytes_map={"abc123.jpg": b"x1", "def456.jpg": b"x2"},
    )
    assert len(result.images) == 2
    assert result.images[0].filename == "test-doc_img01.jpg"
    assert "{{IMG|" in result.text
    assert "images/abc123.jpg" not in result.text

def test_image_sha256_from_bytes():
    img_bytes = b"\x89PNG fake"
    expected_sha = hashlib.sha256(img_bytes).hexdigest()
    md = f"![](images/{expected_sha}.jpg)"
    result = mineru_markdown_to_parse_result(
        markdown=md, doc_slug="doc", images_dir=Path("/fake"),
        image_bytes_map={f"{expected_sha}.jpg": img_bytes},
    )
    assert result.images[0].sha256 == expected_sha
    assert result._image_bytes[0] == img_bytes

def test_extract_html_table():
    md = "Intro.\n\n<table><tr><td>A</td><td>B</td></tr><tr><td>1</td><td>2</td></tr></table>\n\nAfter.\n"
    result = mineru_markdown_to_parse_result(
        markdown=md, doc_slug="doc", images_dir=Path("/fake"), image_bytes_map={},
    )
    assert len(result.tables) == 1
    assert result.tables[0] == [["A", "B"], ["1", "2"]]
    assert "<table>" not in result.text
    assert "[table 1]" in result.text

def test_merge_multiple_segments():
    seg1 = "# Part 1\n\n![](images/img1.jpg)\n"
    seg2 = "\n# Part 2\n\n![](images/img2.jpg)\n"
    result = mineru_markdown_to_parse_result(
        markdown=seg1 + "\n\n---\n\n" + seg2, doc_slug="doc",
        images_dir=Path("/fake"),
        image_bytes_map={"img1.jpg": b"a", "img2.jpg": b"b"},
    )
    assert len(result.images) == 2
    assert "Part 1" in result.text and "Part 2" in result.text
