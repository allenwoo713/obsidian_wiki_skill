"""4 信号知识图谱 + Louvain 社区 + pyvis HTML 可视化（Retrieval v2）。

节点标识改为 page_id（规范化绝对路径，与索引 chunks 表一致，issue #5），
使 query.py 的 1-hop 扩展能用 page_id 精确匹配。用法：
    python build_graph.py <project_root>
"""
from __future__ import annotations
import json
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Dict, List

import _config  # noqa: F401  加载 <skill_dir>/.env，保持入口一致
import networkx as nx

_FM_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


def _read_title(proj: Path) -> str:
    purpose = proj / "purpose.md"
    if purpose.exists():
        for line in purpose.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
            if line.startswith("title:"):
                return line.split(":", 1)[1].strip().strip('"\'')
    return "Wiki"


def _page_id(path: Path) -> str:
    return str(Path(path).resolve())


def _load_pages(wiki_dir: Path) -> List[dict]:
    pages = []
    for md in sorted(wiki_dir.rglob("*.md")):
        if ".graph" in md.parts:
            continue
        raw = md.read_text(encoding="utf-8", errors="replace")
        m = _FM_RE.match(raw)
        if not m:
            continue
        import yaml
        fm = yaml.safe_load(m.group(1)) or {}
        links = [l.strip() for l in re.findall(r"\[\[([^\]]+)\]\]", m.group(2))]
        pid = _page_id(md)
        pages.append({
            "page_id": pid,
            "path": str(md),
            "title": fm.get("title", md.stem),
            "type": fm.get("type", "concept"),
            "sources": fm.get("sources", []) or [],
            "links": links,
        })
    return pages


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = a & b
    union = a | b
    return len(inter) / len(union) if union else 0.0


def _merge_signal(G: nx.Graph, u: str, v: str, *, weight: float, signal: str) -> None:
    """Apply one graph signal while preserving direct/inferred merge semantics."""
    if G.has_edge(u, v):
        G[u][v]["weight"] += weight
        G[u][v]["signals"].add(signal)
    else:
        G.add_edge(u, v, weight=weight, signals={signal})


def _source_overlap_candidates(src_sets: Dict[str, set]) -> list[tuple[str, str]]:
    """Generate only page pairs sharing a source, in a stable order.

    A full page-pair scan turns a 30k corpus into ~450M comparisons.  The
    inverted source map leaves exact Jaccard scoring intact but makes unique
    synthetic sources contribute no candidates at all.
    """
    inverted: dict[str, list[str]] = defaultdict(list)
    for page_id in sorted(src_sets):
        for source in sorted(src_sets[page_id]):
            inverted[source].append(page_id)
    candidates: set[tuple[str, str]] = set()
    for members in inverted.values():
        for u, v in combinations(members, 2):
            candidates.add((u, v) if u < v else (v, u))
    return sorted(candidates)


def _adamic_adar_candidates(G: nx.Graph) -> list[tuple[str, str]]:
    """Return two-hop non-edges only, skipping complete components in O(V+E)."""
    candidates: set[tuple[str, str]] = set()
    for component in nx.connected_components(G):
        nodes = tuple(sorted(component))
        node_count = len(nodes)
        if sum(G.degree(node) for node in nodes) // 2 == node_count * (node_count - 1) // 2:
            continue
        for pivot in nodes:
            neighbors = sorted(G.neighbors(pivot))
            for u, v in combinations(neighbors, 2):
                if not G.has_edge(u, v):
                    candidates.add((u, v) if u < v else (v, u))
    return sorted(candidates)


def build_graph(wiki_dir: Path) -> nx.Graph:
    pages = _load_pages(wiki_dir)
    G = nx.Graph()
    title_to_pid: Dict[str, str] = {p["title"]: p["page_id"] for p in pages}
    slug_to_pid: Dict[str, str] = {Path(p["path"]).stem: p["page_id"] for p in pages}
    for p in pages:
        G.add_node(p["page_id"], id=p["page_id"], title=p["title"], path=p["path"],
                   page_type=p["type"], sources=p["sources"], degree=0)
    # Signal 1: 直接链接
    for p in pages:
        for link_raw in p["links"]:
            link = link_raw.split("|")[0].strip() if "|" in link_raw else link_raw
            resolved = title_to_pid.get(link) or slug_to_pid.get(link)
            if not resolved or resolved == p["page_id"]:
                continue
            if G.has_edge(p["page_id"], resolved):
                G[p["page_id"]][resolved]["weight"] += 1.0
                G[p["page_id"]][resolved]["signals"].add("direct_link")
            else:
                G.add_edge(p["page_id"], resolved, weight=1.0, signals={"direct_link"})
    # Signal 2: 源重叠
    src_sets = {p["page_id"]: set(p["sources"]) for p in pages}
    for u, v in _source_overlap_candidates(src_sets):
        ov = _jaccard(src_sets[u], src_sets[v])
        if ov > 0:
            _merge_signal(G, u, v, weight=0.6 * ov, signal="source_overlap")
    # Signal 3: Adamic-Adar
    compute_adamic_adar(G)
    # Signal 4: 类型亲和力
    type_map = {p["page_id"]: p["type"] for p in pages}
    for u, v in G.edges():
        if type_map.get(u) == type_map.get(v) and "type_affinity" not in G[u][v]["signals"]:
            G[u][v]["weight"] += 0.3
            G[u][v]["signals"].add("type_affinity")
    for n in G.nodes():
        G.nodes[n]["degree"] = G.degree(n)
    return G


def compute_adamic_adar(G: nx.Graph, top_n_per_node: int = 5, min_score: float = 0.0):
    if G.number_of_edges() == 0 or G.number_of_nodes() < 2:
        return
    candidates = _adamic_adar_candidates(G)
    if not candidates:
        return
    preds = [(u, v, score) for u, v, score in nx.adamic_adar_index(G, ebunch=candidates)
             if score > min_score]
    preds.sort(key=lambda item: (-item[2], item[0], item[1]))
    added_count: Dict[str, int] = {}
    for u, v, score in preds:
        if added_count.get(u, 0) >= top_n_per_node and added_count.get(v, 0) >= top_n_per_node:
            continue
        _merge_signal(G, u, v, weight=0.4 * score, signal="adamic_adar")
        added_count[u] = added_count.get(u, 0) + 1
        added_count[v] = added_count.get(v, 0) + 1


def compute_4_signals(G: nx.Graph) -> Dict:
    stats = {"direct_link": 0, "source_overlap": 0, "adamic_adar": 0, "type_affinity": 0}
    for u, v, d in G.edges(data=True):
        for s in d.get("signals", set()):
            if s in stats:
                stats[s] += 1
    return stats


def _mark_community_reports_stale(project_root: Path) -> None:
    """Publish graph freshness diagnostics only after graph.json is durable."""
    from obsidian_wiki.infrastructure.filesystem_community_reports import FilesystemCommunityReportStore

    FilesystemCommunityReportStore(project_root / ".index").mark_stale(
        producer="build_graph", reason="graph_published"
    )


def detect_communities(G: nx.Graph) -> List[List[str]]:
    try:
        import community as community_louvain
        partition = community_louvain.best_partition(G)
        comms: Dict[int, List[str]] = {}
        for node, cid in partition.items():
            comms.setdefault(cid, []).append(node)
        comm_list = list(comms.values())
    except ImportError:
        comm_list = [list(G.nodes())]
    isolated = [n for n in G.nodes() if G.degree(n) == 0]
    if isolated:
        comm_list = [[n for n in comm if G.degree(n) > 0] for comm in comm_list]
        comm_list = [c for c in comm_list if c]
        comm_list.append(isolated)
    return comm_list


def render_html(G: nx.Graph, out_path: Path, title: str = "Wiki"):
    from pyvis.network import Network
    comms = detect_communities(G)
    node_comm: Dict[str, int] = {}
    for i, comm in enumerate(comms):
        for n in comm:
            node_comm[n] = i
    palette = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
               "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080"]
    isolated_comm_id = len(comms) - 1 if comms and any(G.degree(n) == 0 for n in comms[-1]) else -1
    net = Network(height="900px", width="100%", bgcolor="#1a1a2e",
                  font_color="white", directed=False, notebook=False)
    net.toggle_physics(True)
    net.set_options('{"physics": {"barnesHut": {"gravitationalConstant": -8000, "springLength": 150, "springConstant": 0.04, "damping": 0.4, "avoidOverlap": 0.2}, "stabilization": {"iterations": 300, "fit": true}}}')
    for n, d in G.nodes(data=True):
        ptype = d.get("page_type", "concept")
        deg = d.get("degree", 0)
        size = 15 + min(deg * 3, 35)
        cid = node_comm.get(n, 0)
        color = "#808080" if cid == isolated_comm_id else palette[cid % len(palette)]
        net.add_node(n, label=d.get("title", n), title=f"type: {ptype}\ndegree: {deg}",
                     size=size, color=color)
    for u, v, d in G.edges(data=True):
        sigs = ", ".join(sorted(d.get("signals", set())))
        net.add_edge(u, v, value=d.get("weight", 1.0), title=sigs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(out_path), notebook=False, open_browser=False)
    html = out_path.read_text(encoding="utf-8", errors="replace")
    html = html.replace('<script src="lib/bindings/utils.js"></script>', '')
    html = re.sub(r'<link[^>]*cdnjs\.cloudflare\.com[^>]*vis-network[^>]*>', '<link rel="stylesheet" href="lib/vis-network.min.css">', html)
    html = re.sub(r'<script[^>]*cdnjs\.cloudflare\.com[^>]*vis-network[^>]*></script>', '<script src="lib/vis-network.min.js"></script>', html)
    html = re.sub(r'<link\s+href="https://cdn\.jsdelivr\.net/npm/bootstrap[^"]*"[^>]*/>', '', html)
    html = re.sub(r'<script\s+src="https://cdn\.jsdelivr\.net/npm/bootstrap[^"]*"[^>]*></script>', '', html)
    header = f'<div style="color:#fff;padding:10px;font-family:sans-serif;background:#16213e;"><h2>{title} 知识图谱</h2><p>节点颜色 = Louvain 社区 | 边粗细 = 4信号加权 | 悬停看详情 | 拖拽节点 | 滚轮缩放</p></div>'
    html = html.replace("<body>", f"<body>{header}", 1)
    out_path.write_text(html, encoding="utf-8")


def main():
    import argparse
    p = argparse.ArgumentParser(
        prog="build_graph.py",
        description="构建 4 信号知识图谱（page_id 节点）+ Louvain 社区 + pyvis 可视化",
    )
    p.add_argument("project_root", help="知识库项目根目录（含 Wiki/）")
    args = p.parse_args()
    proj = Path(args.project_root)
    wiki = proj / "Wiki"
    G = build_graph(wiki)
    stats = compute_4_signals(G)
    comms = detect_communities(G)
    graph_json = {
        "nodes": [{"id": n, **{k: v for k, v in d.items() if k != "signals"}}
                  for n, d in G.nodes(data=True)],
        "edges": [{"source": u, "target": v,
                   "weight": round(d.get("weight", 1.0), 4),
                   "signal": sorted(d.get("signals", set()))[0] if d.get("signals") else "unknown",
                   "signals": sorted(d.get("signals", set()))}
                  for u, v, d in G.edges(data=True)],
        "signals": stats,
        "communities": comms,
    }
    idx = proj / ".index"
    idx.mkdir(exist_ok=True)
    (idx / "graph.json").write_text(
        json.dumps(graph_json, ensure_ascii=False, indent=2, default=list), encoding="utf-8")
    _mark_community_reports_stale(proj)
    render_html(G, wiki / ".graph" / "index.html", title=_read_title(proj))
    print(f"图谱构建完成: {G.number_of_nodes()} 节点, {G.number_of_edges()} 边, {len(comms)} 社区")
    print(f"信号分布: {stats}")
    print(f"HTML → {wiki / '.graph' / 'index.html'}")


if __name__ == "__main__":
    main()
