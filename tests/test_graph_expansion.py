"""issue #9：图谱扩展单元测试 —— 覆盖 build_graph 节点/边、graph_expand 邻居召回。"""
import json
from pathlib import Path

from build_graph import build_graph, detect_communities
from query import graph_expand


def test_build_graph_uses_page_id_nodes(tiny_kb):
    wi, wiki, _ = tiny_kb
    G = build_graph(wiki)
    # 节点标识为 page_id（规范化路径），非 title
    assert G.number_of_nodes() >= 3
    for n, d in G.nodes(data=True):
        assert "cam_x200" in n or "radar_" in n


def test_source_overlap_creates_edge(tiny_kb):
    """两台 Columbus 雷达共享 raw/radar.docx → 源重叠建边。"""
    wi, wiki, _ = tiny_kb
    G = build_graph(wiki)
    radar_ids = [n for n in G.nodes if "radar_cfr100" in n or "radar_ccr100" in n]
    assert len(radar_ids) == 2
    assert G.has_edge(radar_ids[0], radar_ids[1]) or G.has_edge(radar_ids[1], radar_ids[0])


def test_graph_expand_returns_neighbors(tiny_kb):
    """graph_expand 从种子 page_id 出发 1-hop 扩展，应召回源重叠邻居。"""
    wi, wiki, _ = tiny_kb
    # 先写 graph.json（hybrid_search 的 graph_expand 读取它）
    G = build_graph(wiki)
    graph_json = {
        "nodes": [{"id": n, **{k: v for k, v in d.items() if k != "signals"}}
                  for n, d in G.nodes(data=True)],
        "edges": [{"source": u, "target": v,
                   "weight": round(d.get("weight", 1.0), 4),
                   "signal": "unknown", "signals": []}
                  for u, v, d in G.edges(data=True)],
        "signals": {}, "communities": [],
    }
    (wiki.parent / ".index" / "graph.json").write_text(
        json.dumps(graph_json, ensure_ascii=False), encoding="utf-8")
    seed = [n for n in G.nodes if "radar_cfr100" in n][0]
    expanded = graph_expand(wi, [seed], wiki, k=10, hop=1)
    # 应召回 radar_ccr100（源重叠邻居）
    assert any("radar_ccr100" in c.page_id for c in expanded)


def test_detect_communities(tiny_kb):
    wi, wiki, _ = tiny_kb
    G = build_graph(wiki)
    comms = detect_communities(G)
    assert isinstance(comms, list)
