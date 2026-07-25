"""社区报告构建（issue #10 GraphRAG global）。

从 page graph → Louvain 社区 → 为每个社区生成结构化报告 →
``.index/community_reports.jsonl``，供 ``query.py`` global intent 的
Global Search 消费（与普通 local retrieval 完全分离）。

设计说明：
- 与 local retrieval 独立——普通查询不会加载 community reports；只有
  ``intent=global`` 才路由到本管道（``query.py::_global_retrieve``）。
- 本实现为**离线 / 无 LLM** 版本：报告 ``summary`` 为成员页面标题、类型、
  来源与内部关系的结构化聚合，而非 LLM 自然语言摘要。LLM 摘要可通过
  注入 ``CommunityReportProvider``（未来扩展）增强；当前无 LLM 时仍可用。
- 报告保存成员页面 hash 集合（``content_hash``），成员变化时可识别 stale。
- 没有构建报告时，global 查询明确提示运行本命令，不静默退化为 local search。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import List

from build_graph import build_graph, detect_communities

logger = logging.getLogger(__name__)


def _token_count(text: str) -> int:
    return max(1, len(text) // 4)


def _make_summary(titles, ptypes, sources, intra_edges, n_members) -> str:
    parts = [f"社区包含 {n_members} 个页面，类型: {', '.join(ptypes) if ptypes else '未知'}。"]
    parts.append("主要页面: " + ", ".join(titles[:12]) + ("..." if len(titles) > 12 else ""))
    if sources:
        parts.append("来源文档: " + ", ".join(sources[:8]) + ("..." if len(sources) > 8 else ""))
    if intra_edges:
        sig_kinds = sorted({s for e in intra_edges for s in e.get("signals", [])})
        parts.append(f"内部关系 {len(intra_edges)} 条（{', '.join(sig_kinds)}）。")
    return "\n".join(parts)


def build_community_reports(project_root: Path, max_communities: int = 50) -> List[dict]:
    """构建社区报告列表（不写盘）。"""
    wiki = project_root / "Wiki"
    if not wiki.exists():
        return []
    G = build_graph(wiki)
    comms = detect_communities(G)
    node_attr = {n: d for n, d in G.nodes(data=True)}
    reports: List[dict] = []
    for cid, members in enumerate(comms):
        if not members:
            continue
        mset = set(members)
        titles = sorted({node_attr[m].get("title", m) for m in members if m in node_attr})
        ptypes = sorted({node_attr[m].get("page_type", "concept") for m in members if m in node_attr})
        sources = sorted({s for m in members if m in node_attr
                          for s in (node_attr[m].get("sources") or [])})
        # 社区内部显式/推断关系
        intra_edges = []
        for u, v, d in G.edges(data=True):
            if u in mset and v in mset:
                intra_edges.append({
                    "source": u, "target": v,
                    "weight": round(float(d.get("weight", 1.0)), 4),
                    "signals": sorted(d.get("signals", set())),
                })
        # key entities = 成员标题，按 degree 降序取前 8
        deg_sorted = sorted(members, key=lambda m: -node_attr.get(m, {}).get("degree", 0))
        key_entities = [node_attr[m].get("title", m) for m in deg_sorted[:8] if m in node_attr]
        summary = _make_summary(titles, ptypes, sources, intra_edges, len(members))
        rep = {
            "community_id": cid,
            "level": 0,
            "member_page_ids": members,
            "title": key_entities[0] if key_entities else f"Community {cid}",
            "summary": summary,
            "key_entities": key_entities,
            "key_relationships": intra_edges[:20],
            "source_pages": sources,
            "token_count": _token_count(summary),
            "content_hash": hashlib.sha256(summary.encode("utf-8")).hexdigest()[:16],
        }
        reports.append(rep)
        if len(reports) >= max_communities:
            break
    return reports


def write_reports(project_root: Path, reports: List[dict]) -> Path:
    idx = project_root / ".index"
    idx.mkdir(exist_ok=True)
    out = idx / "community_reports.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for r in reports:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    # communities.json：社区结构（供可视化 / stale 检测）
    (idx / "communities.json").write_text(
        json.dumps([{"community_id": r["community_id"],
                     "member_page_ids": r["member_page_ids"],
                     "title": r["title"],
                     "content_hash": r["content_hash"]}
                    for r in reports], ensure_ascii=False, indent=2),
        encoding="utf-8")
    return out


def is_stale(project_root: Path) -> bool:
    """社区报告是否过期（成员页面变化）。简化版：graph.json 与报告成员集不一致即 stale。"""
    idx = project_root / ".index"
    cr = idx / "community_reports.jsonl"
    gj = idx / "graph.json"
    if not cr.exists():
        return True
    try:
        reported = set()
        for line in cr.read_text(encoding="utf-8").splitlines():
            if line.strip():
                reported |= set(json.loads(line).get("member_page_ids", []))
        if gj.exists():
            gnodes = {n["id"] for n in json.loads(gj.read_text(encoding="utf-8")).get("nodes", [])}
            return reported != gnodes
    except (json.JSONDecodeError, OSError):
        return True
    return False


def main():
    logging.basicConfig(level=logging.WARNING)
    p = argparse.ArgumentParser(
        prog="build_community_reports.py",
        description="构建社区报告（issue #10 GraphRAG global）",
    )
    p.add_argument("project_root", help="知识库项目根目录（含 Wiki/）")
    p.add_argument("--max-communities", type=int, default=50)
    args = p.parse_args()
    reports = build_community_reports(Path(args.project_root), args.max_communities)
    out = write_reports(Path(args.project_root), reports)
    print(f"社区报告构建完成: {len(reports)} 个社区 → {out}")


if __name__ == "__main__":
    main()
