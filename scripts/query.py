"""Hybrid FTS+RAG 检索：BM25 + 向量 + 图谱 → RRF 融合 → 预算控制。
用法：python query.py <project_root> "<query>" [--k 5] [--max-tokens 4096] [--json] [--read-full]
"""
from __future__ import annotations
import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import _config  # noqa: F401  # 加载 <skill_dir>/.env（ISSUE-01），须在 build_index 的向量检索触发前执行

from models import RetrievedPage

# ISSUE-07：异常不再静默吞没，至少 warning 到 stderr（不污染 stdout 管道）
logger = logging.getLogger(__name__)


_FM_RE = re.compile(r"^---\n.*?\n---\n(.*)$", re.DOTALL)


@dataclass
class SearchResults:
    """检索结果，分离 text 与 image chunks。"""
    text: List[RetrievedPage]
    images: List[RetrievedPage]


def split_text_image(results: List[RetrievedPage]) -> Tuple[List[RetrievedPage], List[RetrievedPage]]:
    """按 path 推断：含 'assets/' 的归 image，其余归 text。"""
    text, images = [], []
    for r in results:
        ps = str(r.path).replace("\\", "/")
        if "assets/" in ps:
            images.append(r)
        else:
            text.append(r)
    return text, images


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
    """读取 wiki 页面完整内容（去 frontmatter），截断到 max_chars。

    ISSUE-07：异常不再静默吞没。读文件失败时 warning 到 stderr 并返回空串，
    让上层（hybrid_search）按"无内容可读"处理，而非把异常当作"文件无内容"。
    两者语义不同：前者是 I/O 故障，后者是合法空页。
    """
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        logger.warning("read_full_content: 读取失败 %s: %s", path, e)
        return ""
    m = _FM_RE.match(raw)
    content = m.group(1).strip() if m else raw
    if len(content) > max_chars:
        content = content[:max_chars] + "\n...[截断，完整内容见原文件]"
    return content


def hybrid_search(wi, query: str, k: int = 5, max_tokens: int = 4096,
                  wiki_dir: Path = None, read_full: bool = False) -> SearchResults:
    bm25_results = wi.search_bm25(query, k=20)
    vector_results = wi.search_vector(query, k=20)
    top_paths = [r.path for r in bm25_results[:5]]
    graph_results = graph_expand(wi, top_paths, wiki_dir, k=10) if wiki_dir else []
    fused = rrf_fuse(bm25_results, vector_results, graph_results)
    selected = budget_control(fused[:k * 3], max_tokens=max_tokens)[:k]
    if read_full:
        for r in selected:
            ps = str(r.path).replace("\\", "/")
            if "assets/" not in ps:
                full = read_full_content(r.path)
                if full:
                    r.snippet = full
    text_chunks, image_chunks = split_text_image(selected)
    return SearchResults(text=text_chunks, images=image_chunks)


def format_for_agent(results: SearchResults, read_full: bool = False) -> str:
    label = "全文" if read_full else "片段"
    lines = [f"## 检索结果（hybrid FTS+RAG，{label}模式）\n"]
    if results.text:
        lines.append("### 文本片段\n")
        for i, r in enumerate(results.text, 1):
            lines.append(f"### [{i}] {r.title}")
            lines.append(f"- 路径: {r.path}")
            lines.append(f"- 相关度: {r.score:.4f} ({r.retrieval_method})")
            lines.append(f"- 源文档: {', '.join(r.sources) if r.sources else 'N/A'}")
            lines.append(f"- {label}:\n```\n{r.snippet}\n```")
            lines.append("")
    if results.images:
        lines.append("### 相关图片（caption 命中）\n")
        for i, r in enumerate(results.images, 1):
            lines.append(f"### [图{i}] {r.title}")
            lines.append(f"- 路径: {r.path}")
            lines.append(f"- 相关度: {r.score:.4f} ({r.retrieval_method})")
            lines.append(f"- 源文档: {', '.join(r.sources) if r.sources else 'N/A'}")
            lines.append(f"- 图注/caption:\n```\n{r.snippet}\n```")
            lines.append(f"- 嵌入建议: ![[{r.path.name}]]")
            lines.append("")
    if not results.text and not results.images:
        return "[无检索结果]"
    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    """ISSUE-06：用 argparse 替代手写 sys.argv 循环，自动生成 --help、参数校验、
    避免 --k/--max-tokens/--out 后无值时 IndexError。"""
    p = argparse.ArgumentParser(
        prog="query.py",
        description="Hybrid FTS+RAG 检索：BM25 + 向量 + 图谱 → RRF 融合 → 预算控制",
    )
    p.add_argument("project_root", help="知识库项目根目录（含 Wiki/ 与 .index/）")
    p.add_argument("query", help="检索查询串（建议先做关键词提取与中英互译扩展）")
    p.add_argument("--k", type=int, default=5, help="返回 top-K 结果（默认 5）")
    p.add_argument("--max-tokens", type=int, default=4096, help="预算控制上限（默认 4096 tokens）")
    p.add_argument("--json", dest="as_json", action="store_true", help="输出 JSON 格式（默认输出 markdown）")
    p.add_argument("--read-full", action="store_true", help="读取命中页面全文（适合问具体数值/流程/对比）")
    p.add_argument("--out", dest="out_path", default=None, help="输出落盘路径（大输出必须用，绕开沙箱 stdout 拦截段错误）")
    return p


def main():
    # ISSUE-07：stderr 日志（不污染 stdout 管道），WARNING 级别即打印
    logging.basicConfig(stream=sys.stderr, level=logging.WARNING, format="[%(levelname)s] %(name)s: %(message)s")

    args = _build_arg_parser().parse_args()
    proj = Path(args.project_root)
    from build_index import WikiIndex
    wi = WikiIndex(proj / ".index")
    wi.load()
    results = hybrid_search(wi, args.query, k=args.k, max_tokens=args.max_tokens,
                             wiki_dir=proj / "Wiki", read_full=args.read_full)
    payload = json.dumps({
        "text": [{"path": str(r.path), "title": r.title, "score": r.score,
                  "snippet": r.snippet, "sources": r.sources, "method": r.retrieval_method}
                 for r in results.text],
        "images": [{"path": str(r.path), "title": r.title, "score": r.score,
                    "snippet": r.snippet, "sources": r.sources, "method": r.retrieval_method,
                    "embed": f"![[{r.path.name}]]"}
                   for r in results.images],
        "read_full": args.read_full,
    }, ensure_ascii=False, indent=2) if args.as_json else format_for_agent(results, read_full=args.read_full)
    if args.out_path:
        op = Path(args.out_path)
        op.parent.mkdir(parents=True, exist_ok=True)
        op.write_text(payload, encoding="utf-8")
        # 仅向 stdout 输出一行小确认（管道安全）；大 payload 落盘，绕开沙箱 stdout 拦截段错误
        print(f"wrote {op} ({len(payload)} bytes)")
    else:
        print(payload)


if __name__ == "__main__":
    main()
