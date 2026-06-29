"""Hybrid FTS+RAG 检索：BM25 + 向量 + 图谱 → RRF 融合 → 预算控制。
用法：python query.py <project_root> "<query>" [--k 5] [--max-tokens 4096] [--json] [--read-full]
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path
from typing import List

from models import RetrievedPage


_FM_RE = re.compile(r"^---\n.*?\n---\n(.*)$", re.DOTALL)


def rrf_fuse(bm25_results: List[RetrievedPage], vector_results: List[RetrievedPage],
             graph_results: List[RetrievedPage], k_rrf: int = 60) -> List[RetrievedPage]:
    scores: dict = {}
    meta: dict = {}
    for results in [bm25_results, vector_results, graph_results]:
        for rank, r in enumerate(results, 1):
            key = str(r.path)
            scores[key] = scores.get(key, 0) + 1.0 / (k_rrf + rank)
            meta[key] = r
    fused = []
    for key, score in sorted(scores.items(), key=lambda x: -x[1]):
        r = meta[key]
        fused.append(RetrievedPage(
            path=r.path, title=r.title, score=score,
            snippet=r.snippet, sources=r.sources, retrieval_method="fused",
        ))
    return fused


def budget_control(pages: List[RetrievedPage], max_tokens: int = 4096,
                   char_per_token: int = 4) -> List[RetrievedPage]:
    budget_chars = max_tokens * char_per_token
    selected = []
    used = 0
    for p in pages:
        snippet_len = len(p.snippet) + len(p.title) + 100
        if used + snippet_len > budget_chars:
            break
        selected.append(p)
        used += snippet_len
    return selected


def graph_expand(wi, top_paths: List[Path], wiki_dir: Path, k: int = 10) -> List[RetrievedPage]:
    idx_file = wiki_dir.parent / ".index" / "graph.json"
    if not idx_file.exists():
        return []
    data = json.loads(idx_file.read_text(encoding="utf-8"))
    edges = data.get("edges", [])
    nodes = {n["title"]: n for n in data.get("nodes", [])}
    neighbors: dict = {}
    for e in edges:
        neighbors.setdefault(e["source"], []).append((e["target"], e.get("weight", 1)))
        neighbors.setdefault(e["target"], []).append((e["source"], e.get("weight", 1)))
    top_titles = set()
    for p in top_paths:
        stem = p.stem
        for t, n in nodes.items():
            if t == stem or n.get("path", "").replace("\\", "/").endswith(p.name.replace("\\", "/")):
                top_titles.add(t)
    seen = set(top_titles)
    expanded = []
    for title in top_titles:
        for nbr, w in neighbors.get(title, []):
            if nbr not in seen:
                seen.add(nbr)
                n = nodes.get(nbr, {})
                expanded.append(RetrievedPage(
                    path=Path(n.get("path", "")), title=nbr, score=w,
                    snippet="", sources=n.get("sources", []),
                    retrieval_method="graph",
                ))
            if len(expanded) >= k:
                break
    return expanded[:k]


def read_full_content(path: Path, max_chars: int = 8000) -> str:
    """读取 wiki 页面完整内容（去 frontmatter），截断到 max_chars。"""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
        m = _FM_RE.match(raw)
        content = m.group(1).strip() if m else raw
        if len(content) > max_chars:
            content = content[:max_chars] + "\n...[截断，完整内容见原文件]"
        return content
    except Exception:
        return ""


def hybrid_search(wi, query: str, k: int = 5, max_tokens: int = 4096,
                  wiki_dir: Path = None, read_full: bool = False) -> List[RetrievedPage]:
    bm25_results = wi.search_bm25(query, k=20)
    vector_results = wi.search_vector(query, k=20)
    top_paths = [r.path for r in bm25_results[:5]]
    graph_results = graph_expand(wi, top_paths, wiki_dir, k=10) if wiki_dir else []
    fused = rrf_fuse(bm25_results, vector_results, graph_results)
    results = budget_control(fused[:k * 3], max_tokens=max_tokens)[:k]
    if read_full:
        for r in results:
            full = read_full_content(r.path)
            if full:
                r.snippet = full
    return results


def format_for_agent(results: List[RetrievedPage], read_full: bool = False) -> str:
    if not results:
        return "[无检索结果]"
    label = "全文" if read_full else "片段"
    lines = [f"## 检索结果（hybrid FTS+RAG，{label}模式）\n"]
    for i, r in enumerate(results, 1):
        lines.append(f"### [{i}] {r.title}")
        lines.append(f"- 路径: {r.path}")
        lines.append(f"- 相关度: {r.score:.4f} ({r.retrieval_method})")
        lines.append(f"- 源文档: {', '.join(r.sources) if r.sources else 'N/A'}")
        lines.append(f"- {label}:\n```\n{r.snippet}\n```")
        lines.append("")
    return "\n".join(lines)


def main():
    if len(sys.argv) < 3:
        print("用法: python query.py <project_root> <query> [--k 5] [--max-tokens 4096] [--json] [--read-full]")
        sys.exit(1)
    proj = Path(sys.argv[1])
    query = sys.argv[2]
    k = 5
    max_tokens = 4096
    as_json = False
    read_full = False
    for i, arg in enumerate(sys.argv[3:], 3):
        if arg == "--k":
            k = int(sys.argv[i + 1])
        elif arg == "--max-tokens":
            max_tokens = int(sys.argv[i + 1])
        elif arg == "--json":
            as_json = True
        elif arg == "--read-full":
            read_full = True
    from build_index import WikiIndex
    wi = WikiIndex(proj / ".index")
    wi.load()
    results = hybrid_search(wi, query, k=k, max_tokens=max_tokens, wiki_dir=proj / "Wiki", read_full=read_full)
    if as_json:
        print(json.dumps([{
            "path": str(r.path), "title": r.title, "score": r.score,
            "snippet": r.snippet, "sources": r.sources, "method": r.retrieval_method,
            "read_full": read_full,
        } for r in results], ensure_ascii=False, indent=2))
    else:
        print(format_for_agent(results, read_full=read_full))


if __name__ == "__main__":
    main()
