"""build_graph.py 测试。"""
import json
from pathlib import Path
import pytest
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from build_graph import build_graph, compute_4_signals, compute_adamic_adar, detect_communities, render_html


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
    # v2 (#5)：节点标识为 page_id（规范化绝对路径），非 title。
    # source_overlap 边确实创建（两页 sources 完全重叠，Jaccard=1.0>0），
    # 但存在于两个 page_id 之间，须按 page_id 查询。
    pid_a = str((wiki / "concepts" / "a.md").resolve())
    pid_b = str((wiki / "concepts" / "b.md").resolve())
    assert G.has_edge(pid_a, pid_b) or G.has_edge(pid_b, pid_a)


def test_source_overlap_uses_only_shared_source_candidates_and_exact_jaccard(tmp_path):
    """Source overlap must not scan unrelated page pairs at corpus scale."""
    wiki = tmp_path / "Wiki"
    _write(wiki, "a.md", "A", ["s1", "s2"], [], ptype="concept")
    _write(wiki, "b.md", "B", ["s1", "s3"], [], ptype="procedure")
    _write(wiki, "c.md", "C", ["s2"], [], ptype="reference")
    _write(wiki, "d.md", "D", ["s4"], [], ptype="example")
    G = build_graph(wiki)
    ids = {name: str((wiki / "concepts" / f"{name}.md").resolve()) for name in "abcd"}
    assert G[ids["a"]][ids["b"]]["weight"] == pytest.approx(0.6 / 3)
    assert G[ids["a"]][ids["c"]]["weight"] == pytest.approx(0.3)
    assert not G.has_edge(ids["a"], ids["d"])
    assert "source_overlap" not in G[ids["b"]][ids["c"]]["signals"]


def test_adamic_adar_never_enumerates_networkx_global_non_edges(monkeypatch):
    """AA candidates are two-hop pairs only; 30k isolated nodes stay O(E), not O(N²)."""
    G = __import__("networkx").Graph()
    G.add_edges_from((("a", "b"), ("b", "c")))

    def forbidden(*_args, **_kwargs):
        pytest.fail("global nx.non_edges enumeration is forbidden at Phase07 scale")

    monkeypatch.setattr(__import__("networkx"), "non_edges", forbidden)
    compute_adamic_adar(G)
    assert G.has_edge("a", "c")
    assert "adamic_adar" in G["a"]["c"]["signals"]


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
