"""update_wiki.py 测试。"""
import json
from pathlib import Path
import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from update_wiki import scan_sources, diff_manifest, SourceState


def _write_src(raw, name, content="doc content"):
    f = raw / name
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(content, encoding="utf-8")


def test_scan_sources(tmp_path):
    raw = tmp_path / "Raw" / "sources"
    _write_src(raw, "a/a.md", "# A\nbody")
    _write_src(raw, "b/b.docx", "fake")
    _write_src(raw, "c/c.pdf", "fake")
    docs = scan_sources(raw)
    suffixes = sorted(d.suffix for d in docs)
    assert ".docx" in suffixes
    assert ".md" in suffixes
    assert ".pdf" in suffixes


def test_diff_new_file(tmp_path):
    raw = tmp_path / "Raw" / "sources"
    _write_src(raw, "a.md", "# A\nbody")
    docs = scan_sources(raw)
    manifest = {"entries": {}}
    diff = diff_manifest(docs, manifest)
    assert len(diff["new"]) == 1
    assert diff["new"][0].path.name == "a.md"
    assert len(diff["modified"]) == 0
    assert len(diff["unchanged"]) == 0


def test_diff_unchanged(tmp_path):
    raw = tmp_path / "Raw" / "sources"
    _write_src(raw, "a.md", "# A\nbody")
    docs = scan_sources(raw)
    from parse_sources import compute_sha256
    sha = compute_sha256(raw / "a.md")
    manifest = {"entries": {str(raw / "a.md"): {"sha256": sha, "status": "processed"}}}
    diff = diff_manifest(docs, manifest)
    assert len(diff["unchanged"]) == 1
    assert len(diff["new"]) == 0


def test_diff_modified(tmp_path):
    raw = tmp_path / "Raw" / "sources"
    _write_src(raw, "a.md", "# A\nnew body")
    docs = scan_sources(raw)
    manifest = {"entries": {str(raw / "a.md"): {"sha256": "old_sha", "status": "processed"}}}
    diff = diff_manifest(docs, manifest)
    assert len(diff["modified"]) == 1


def _create_minimal_png(path):
    """生成一个最小的合法 1x1 红色 PNG。"""
    import struct, zlib
    sig = b'\x89PNG\r\n\x1a\n'
    ihdr_data = struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0)
    ihdr_crc = zlib.crc32(b'IHDR' + ihdr_data) & 0xffffffff
    ihdr = struct.pack('>I', 13) + b'IHDR' + ihdr_data + struct.pack('>I', ihdr_crc)
    raw_data = b'\x00\xff\x00\x00'
    compressed = zlib.compress(raw_data)
    idat_crc = zlib.crc32(b'IDAT' + compressed) & 0xffffffff
    idat = struct.pack('>I', len(compressed)) + b'IDAT' + compressed + struct.pack('>I', idat_crc)
    iend_crc = zlib.crc32(b'IEND') & 0xffffffff
    iend = struct.pack('>I', 0) + b'IEND' + struct.pack('>I', iend_crc)
    path.write_bytes(sig + ihdr + idat + iend)


def test_extract_images_for_diff_new(tmp_path):
    from docx import Document; from docx.shared import Inches
    raw = tmp_path / "Raw"; raw.mkdir()
    doc = Document(); img = tmp_path / "t.png"
    _create_minimal_png(img)
    doc.add_picture(str(img), width=Inches(1)); doc.add_paragraph("图1")
    dp = raw / "t.docx"; doc.save(str(dp))
    assets = tmp_path / "assets"
    from update_wiki import extract_images_for_diff
    im = extract_images_for_diff([dp], [], assets)
    assert len(im) == 1; assert im[0]["filename"].endswith("_img01.png")
    # assets 目录被创建且含图片
    assert assets.exists()
