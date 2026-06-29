"""query.py 测试。"""
import json
from pathlib import Path
import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from query import hybrid_search, rrf_fuse, budget_control


def _write(wiki, subdir, name, title, body, sources=None):
    d = wiki / subdir
    d.mkdir(parents=True, exist_ok=True)
    fm = '---\ntype: concept\ntitle: "%s"\nsources: %s\ntags: []\nrelated: []\nupdated: 2026-06-29\n---\n\n' % (
        title, json.dumps(sources or []))
    (d / name).write_text(fm + body, encoding="utf-8")


def test_rrf_fuse():
    from models import RetrievedPage
    bm25 = [RetrievedPage(Path("a"), "A", 1, "", [], "bm25"),
            RetrievedPage(Path("b"), "B", 1, "", [], "bm25")]
    vec = [RetrievedPage(Path("b"), "B", 1, "", [], "vector"),
           RetrievedPage(Path("c"), "C", 1, "", [], "vector")]
    fused = rrf_fuse(bm25, vec, [])
    titles = [r.title for r in fused]
    assert "B" in titles
    assert titles[0] == "B"


def test_budget_control():
    from models import RetrievedPage
    pages = [RetrievedPage(Path(f"p{i}"), f"T{i}", 1.0, "x" * 500, [], "fused") for i in range(20)]
    selected = budget_control(pages, max_tokens=2000, char_per_token=4)
    total = sum(len(p.snippet) for p in selected)
    assert total <= 2000 * 4


def test_hybrid_search(tmp_path):
    wiki = tmp_path / "Wiki"
    idx_dir = tmp_path / ".index"
    _write(wiki, "concepts", "a.md", "Acme Front Radar",
           "频率 60fps 探测距离 200m FOV ±60度", ["raw/acme.docx"])
    _write(wiki, "concepts", "b.md", "Vega Radar",
           "频率 76GHz Vega 探测距离 150m", ["raw/vega.docx"])
    from build_index import WikiIndex
    wi = WikiIndex(idx_dir)
    wi.build(wiki)
    results = hybrid_search(wi, "Acme 60fps 频率", k=5, wiki_dir=wiki)
    assert len(results) > 0
