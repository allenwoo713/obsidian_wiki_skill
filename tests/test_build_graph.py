"""build_graph.py 测试。"""
import json
from pathlib import Path
import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from build_graph import build_graph, compute_4_signals, detect_communities, render_html


def _write(wiki, name, title, sources, related, ptype="concept", subdir="concepts"):
    d = wiki / subdir
    d.mkdir(parents=True, exist_ok=True)
    fm = '---\ntype: %s\ntitle: "%s"\nsources: %s\ntags: []\nrelated: %s\nupdated: 2026-06-29\n---\n\nbody' % (
        ptype, title, json.dumps(sources), json.dumps(related))
    (d / name).write_text(fm, encoding="utf-8")


def test_build_graph_basic(tmp_path):
    wiki = tmp_path / "Wiki"
    _write(wiki, "a.md", "Page A", ["raw/x.docx"], ["[[Page B]]"])
    _write(wiki, "b.md", "Page B", ["raw/x.docx", "raw/y.docx"], ["[[Page A]]"])
    G = build_graph(wiki)
    assert G.number_of_nodes() == 2
    assert G.number_of_edges() >= 1


def test_source_overlap_creates_edge(tmp_path):
    wiki = tmp_path / "Wiki"
    _write(wiki, "a.md", "A", ["raw/shared.docx"], [])
    _write(wiki, "b.md", "B", ["raw/shared.docx"], [])
    G = build_graph(wiki)
    assert G.has_edge("A", "B") or G.has_edge("B", "A")


def test_communities(tmp_path):
    wiki = tmp_path / "Wiki"
    _write(wiki, "a.md", "A", ["s1"], ["[[B]]"])
    _write(wiki, "b.md", "B", ["s1"], ["[[A]]"])
    _write(wiki, "c.md", "C", ["s2"], ["[[D]]"])
    _write(wiki, "d.md", "D", ["s2"], ["[[C]]"])
    G = build_graph(wiki)
    comms = detect_communities(G)
    assert len(comms) >= 1


def test_render_html(tmp_path):
    wiki = tmp_path / "Wiki"
    _write(wiki, "a.md", "A", ["s1"], ["[[B]]"])
    _write(wiki, "b.md", "B", ["s1"], ["[[A]]"])
    G = build_graph(wiki)
    out = tmp_path / "graph.html"
    render_html(G, out)
    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert "vis-network" in content or "vis.js" in content or "<canvas" in content
