"""build_index.py 测试。"""
import json
from pathlib import Path
import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from build_index import WikiIndex


def _write_page(wiki: Path, name: str, title: str, body: str, sources=None):
    d = wiki / "concepts"
    d.mkdir(parents=True, exist_ok=True)
    fm = '---\ntype: concept\ntitle: "%s"\nsources: %s\ntags: []\nrelated: []\nupdated: 2026-06-29\n---\n\n' % (title, json.dumps(sources or []))
    (d / name).write_text(fm + body, encoding="utf-8")


def test_build_and_search_bm25(tmp_path):
    wiki = tmp_path / "Wiki"
    idx_dir = tmp_path / ".index"
    _write_page(wiki, "a.md", "Acme Front Radar", "频率 60fps 探测距离 200m", ["raw/a.docx"])
    _write_page(wiki, "b.md", "Vega Radar", "频率 76GHz 探测距离 150m", ["raw/b.docx"])
    wi = WikiIndex(idx_dir)
    wi.build(wiki)
    results = wi.search_bm25("Acme 60fps", k=2)
    assert len(results) > 0
    assert "Acme" in results[0].title or "Acme" in results[0].path.name


def test_manifest_written(tmp_path):
    wiki = tmp_path / "Wiki"
    idx_dir = tmp_path / ".index"
    _write_page(wiki, "a.md", "T1", "body text here", ["raw/a.docx"])
    wi = WikiIndex(idx_dir)
    wi.build(wiki)
    manifest = idx_dir / "manifest.json"
    assert manifest.exists()
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert len(data["pages"]) >= 1
    assert data["pages"][0]["sha256"]


def test_vector_search(tmp_path):
    wiki = tmp_path / "Wiki"
    idx_dir = tmp_path / ".index"
    _write_page(wiki, "a.md", "Radar Calibration", "radar calibration procedure angle alignment", ["raw/a.docx"])
    _write_page(wiki, "b.md", "UDP Protocol", "udp packet format diagnostic interface", ["raw/b.docx"])
    wi = WikiIndex(idx_dir)
    wi.build(wiki)
    results = wi.search_vector("calibration alignment", k=2)
    assert len(results) > 0
