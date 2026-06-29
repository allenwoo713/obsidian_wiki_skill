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
