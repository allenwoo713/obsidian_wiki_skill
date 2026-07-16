"""Wiki 索引构建：BM25 + LanceDB 向量 + manifest.json。
用法：python build_index.py <project_root>
"""
from __future__ import annotations
import json
import os
import pickle
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import _config  # noqa: F401  # 加载 <skill_dir>/.env（ISSUE-01），须在下方 setdefault 之前执行

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from models import WikiPage, RetrievedPage, ManifestEntry


# ISSUE-15：向量检索 metric contract —— 固定配置，索引侧与查询侧一致
# 默认 cosine：不受向量模长影响，语义稳定，0~1 score 含义明确
VECTOR_METRIC = "cosine"
# embedding 不做 L2 归一化（保持 MiniLM 原始输出，cosine metric 已处理模长）
NORMALIZE_EMBEDDINGS = False


_FM_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)
_LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


def parse_wiki_page(path: Path, project_root: Path) -> Optional[WikiPage]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    m = _FM_RE.match(raw)
    if not m:
        return None
    fm_text, body = m.group(1), m.group(2)
    import yaml
    fm = yaml.safe_load(fm_text) or {}
    links = [l.strip() for l in _LINK_RE.findall(body)]
    import hashlib
    sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    sources = fm.get("sources", []) or []
    if isinstance(sources, str):
        sources = [sources]
    return WikiPage(
        path=path,
        title=fm.get("title", path.stem),
        page_type=fm.get("type", "concept"),
        content=body.strip(),
        sources=[str(s) for s in sources],
        links=links,
        sha256=sha,
    )


def scan_wiki(wiki_dir: Path, project_root: Path) -> List[WikiPage]:
    pages = []
    for md in sorted(wiki_dir.rglob("*.md")):
        if ".graph" in md.parts:
            continue
        p = parse_wiki_page(md, project_root)
        if p:
            pages.append(p)
    return pages


class WikiIndex:
    """BM25 + LanceDB 向量索引，支持增量构建。"""

    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.bm25 = None
        self.pages: List[WikiPage] = []
        self._page_paths: List[str] = []
        self._embedder = None
        self._lance_table = None

    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            # 优先本地模型路径（已从 modelscope 预下载），回退 HF 在线下载
            local_path = os.environ.get("WIKI_EMBEDDER_LOCAL_PATH") or \
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))))),
                    "binaries", "python", "envs", "default", "models",
                    "paraphrase-multilingual-MiniLM-L12-v2")
            # 也检查 venv 标准位置（跨平台，基于 home 目录展开）
            candidate_paths = [
                local_path,
                os.path.expanduser("~/.workbuddy/binaries/python/envs/default/models/paraphrase-multilingual-MiniLM-L12-v2"),
            ]
            for p in candidate_paths:
                if os.path.isdir(p) and os.path.exists(os.path.join(p, "model.safetensors")):
                    self._embedder = SentenceTransformer(p)
                    return self._embedder
            # 回退：HF 在线下载（需网络）
            self._embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        return self._embedder

    def _get_lance(self, dim: int = None):
        """获取 LanceDB 表。首次建表时需提供 dim（向量维度）。

        ISSUE-10：维度从硬编码 384 改为运行时从 embedder.get_sentence_embedding_dimension()
        动态获取，换模型（如 bge-m3 是 1024 维）不再崩。
        """
        if self._lance_table is None:
            import lancedb
            db = lancedb.connect(str(self.index_dir / "lance_db"))
            try:
                self._lance_table = db.open_table("wiki_pages")
            except Exception:
                # 首次建表：用占位空行初始化 schema，dim 必须由调用方传入
                vec_dim = dim or 384  # 兜底 MiniLM-L12-v2 维度，避免无 dim 调用建错 schema
                self._lance_table = db.create_table("wiki_pages", data=[
                    {"path": "", "title": "", "page_type": "", "content": "",
                     "sources": "", "vector": [0.0] * vec_dim}
                ])
                self._lance_table.delete('path = ""')
        return self._lance_table

    def _embedding_dim(self) -> int:
        """ISSUE-10：从当前 embedder 动态获取向量维度，避免硬编码。"""
        emb = self._get_embedder()
        # sentence-transformers 新版重命名为 get_embedding_dimension，旧版用 get_sentence_embedding_dimension
        if hasattr(emb, "get_embedding_dimension"):
            return emb.get_embedding_dimension()
        return emb.get_sentence_embedding_dimension()

    def build(self, wiki_dir: Path):
        self.pages = scan_wiki(wiki_dir, wiki_dir.parent)
        # 追加图片 caption 作为虚拟页（page_type=image_caption）
        idx_dir = wiki_dir.parent / ".index"
        image_pages = self._load_image_caption_pages(idx_dir)
        self.pages.extend(image_pages)
        self._page_paths = [str(p.path) for p in self.pages]
        self._build_bm25()
        self._build_vector()
        self._write_manifest()

    def _load_image_caption_pages(self, idx_dir: Path) -> List[WikiPage]:
        """从 manifest.json 读 images，caption_text 非空的转为 WikiPage。

        ISSUE-07：manifest 解析失败不再静默吞没。schema 漂移会 warning 到 stderr，
        让维护者知道 manifest 格式出问题，而不是误判"知识库无 caption"。
        ISSUE-12：字段访问统一 .get() + 跳过不完整条目并 warning。
        """
        manifest_file = idx_dir / "manifest.json"
        if not manifest_file.exists():
            return []
        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            import logging
            logging.getLogger(__name__).warning(
                "_load_image_caption_pages: manifest 解析失败 %s: %s", manifest_file, e
            )
            return []
        wiki_dir = idx_dir.parent / "Wiki"
        pages = []
        skipped = 0
        for img in manifest.get("images", []):
            # caption_text 是主检索字段；为空时兜底读 vlm_caption.description，
            # 确保已生成 VLM 描述的图不会因 caption_text 漏填而缺席检索（defense-in-depth）。
            caption = (img.get("caption_text") or "").strip()
            if not caption:
                vlm = img.get("vlm_caption") or {}
                caption = (vlm.get("description") or "").strip()
            if not caption:
                continue
            # ISSUE-12：rel_path 缺失则跳过（否则 KeyError 崩溃整个 build）
            rel_path = img.get("rel_path")
            if not rel_path:
                skipped += 1
                continue
            img_path = wiki_dir / rel_path
            pages.append(WikiPage(
                path=img_path,
                title=img.get("figure_caption") or img.get("filename") or rel_path,
                page_type="image_caption",
                content=caption,
                sources=[img.get("source_doc", "")],
                links=[],
                sha256=img.get("sha256", ""),
            ))
        if skipped:
            import logging
            logging.getLogger(__name__).warning(
                "_load_image_caption_pages: 跳过 %d 条 rel_path 缺失的图片条目", skipped
            )
        return pages

    def _tokenize(self, text: str) -> List[str]:
        text = re.sub(r"[^\w\u4e00-\u9fff]", " ", text.lower())
        tokens = re.split(r"\s+", text.strip())
        cjk = re.findall(r"[\u4e00-\u9fff]", text)
        bigrams = [cjk[i] + cjk[i + 1] for i in range(len(cjk) - 1)]
        return [t for t in tokens if t] + bigrams

    def _build_bm25(self):
        from rank_bm25 import BM25Plus
        corpus = [self._tokenize(p.title + " " + p.content) for p in self.pages]
        self.bm25 = BM25Plus(corpus) if corpus else None
        with open(self.index_dir / "bm25_index.pkl", "wb") as f:
            pickle.dump({"bm25": self.bm25, "paths": self._page_paths}, f)

    def _build_vector(self):
        embedder = self._get_embedder()
        texts = [p.title + "\n" + p.content for p in self.pages]
        # ISSUE-15：索引侧与查询侧必须用相同 normalize_embeddings 配置
        vectors = embedder.encode(
            texts, show_progress_bar=False,
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
        ).tolist()
        # ISSUE-10：建表时传入动态维度（而非硬编码 384），换模型时 schema 正确
        table = self._get_lance(dim=self._embedding_dim())
        try:
            table.delete("path != ''")
        except Exception as e:
            # 首次建表时 delete 可能因空表 / SQL 方言差异失败，属预期情况，debug 级记录即可
            import logging
            logging.getLogger(__name__).debug("_build_vector: 跳过 delete（可能首次建表）: %s", e)
        data = [
            {
                "path": str(p.path),
                "title": p.title,
                "page_type": p.page_type,
                "content": p.content[:2000],
                "sources": json.dumps(p.sources, ensure_ascii=False),
                "vector": v,
            }
            for p, v in zip(self.pages, vectors)
        ]
        if data:
            table.add(data)

    def _write_manifest(self):
        manifest_file = self.index_dir / "manifest.json"
        # 读取现有 manifest（保留 images/entries 等非 pages 字段）
        existing = {}
        if manifest_file.exists():
            try:
                existing = json.loads(manifest_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                # manifest 损坏时 warning，从空重建（不崩整个 build）
                import logging
                logging.getLogger(__name__).warning(
                    "_write_manifest: 现有 manifest 解析失败，将覆盖重建: %s", e
                )
        existing["built_at"] = datetime.now().isoformat()
        existing["page_count"] = len(self.pages)
        # ISSUE-15：持久化向量 metric 配置，load 时校验
        existing["vector_config"] = {
            "metric": VECTOR_METRIC,
            "normalize_embeddings": NORMALIZE_EMBEDDINGS,
            "score_mapping": "cosine_linear_v1",
            "schema_version": 1,
        }
        existing["pages"] = [
            {
                "path": str(p.path),
                "sha256": p.sha256,
                "page_type": p.page_type,
                "title": p.title,
                "sources": p.sources,
                "links": p.links,
            }
            for p in self.pages
        ]
        manifest_file.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def load(self):
        with open(self.index_dir / "bm25_index.pkl", "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self._page_paths = data["paths"]
        self._get_lance()
        # 从 manifest 重建 pages（供 search_bm25 获取 title/sources/snippet）
        manifest_file = self.index_dir / "manifest.json"
        if manifest_file.exists():
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
            # ISSUE-15：检测 legacy 索引（无 vector_config），避免 metric 语义不确定
            vc = manifest.get("vector_config")
            if vc is None:
                raise RuntimeError(
                    "Legacy vector index detected (no vector_config in manifest). "
                    "Old index was built with LanceDB default L2 metric, which has "
                    "ambiguous score semantics. Rebuild the index: "
                    "python build_index.py <project_root>"
                )
            manifest_metric = vc.get("metric")
            if manifest_metric != VECTOR_METRIC:
                raise RuntimeError(
                    f"Vector metric mismatch: manifest has '{manifest_metric}', "
                    f"current code expects '{VECTOR_METRIC}'. "
                    "Rebuild the vector index before querying."
                )
            page_map = {p["path"]: p for p in manifest.get("pages", [])}
            self.pages = []
            for path in self._page_paths:
                p = page_map.get(path, {})
                self.pages.append(WikiPage(
                    path=Path(path),
                    title=p.get("title", Path(path).stem),
                    page_type=p.get("page_type", "concept"),
                    content="",  # snippet 从文件读取
                    sources=p.get("sources", []),
                    links=p.get("links", []),
                    sha256=p.get("sha256", ""),
                ))

    def search_bm25(self, query: str, k: int = 20) -> List[RetrievedPage]:
        if self.bm25 is None:
            return []
        tokens = self._tokenize(query)
        scores = self.bm25.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: -x[1])[:k]
        results = []
        for idx, score in ranked:
            if score <= 0:
                continue
            p = self.pages[idx] if idx < len(self.pages) else None
            path = self._page_paths[idx]
            title = p.title if p else Path(path).stem
            sources = p.sources if p else []
            snippet = (p.content[:200] if p and p.content else "")[:200]
            if not snippet and p:
                # 从文件读取 snippet（I/O 异常时 warning，不当作"页面无内容"）
                try:
                    raw = p.path.read_text(encoding="utf-8", errors="replace")
                    import re as _re
                    m = _re.search(r"^---\n.*?\n---\n(.*)$", raw, _re.DOTALL)
                    snippet = (m.group(1).strip()[:200] if m else raw[:200])
                except OSError as e:
                    import logging
                    logging.getLogger(__name__).warning(
                        "search_bm25: 读取 snippet 失败 %s: %s", p.path, e
                    )
                    snippet = ""
            results.append(RetrievedPage(
                path=Path(path), title=title, score=float(score),
                snippet=snippet, sources=sources, retrieval_method="bm25",
            ))
        return results

    def search_vector(self, query: str, k: int = 20) -> List[RetrievedPage]:
        """ISSUE-15：显式指定 cosine metric，用 normalize_vector_score 替换旧 1/(1+d)。"""
        embedder = self._get_embedder()
        qv = embedder.encode(
            [query], show_progress_bar=False,
            normalize_embeddings=NORMALIZE_EMBEDDINGS,
        )[0].tolist()
        table = self._get_lance()
        from vector_scoring import apply_vector_metric, normalize_vector_score
        query_builder = apply_vector_metric(table.search(qv), VECTOR_METRIC)
        rows = query_builder.limit(k).to_list()
        results = []
        for r in rows:
            # _distance 必须存在，缺失则报错（不伪造默认值）
            if "_distance" not in r:
                raise RuntimeError(
                    f"LanceDB result missing '_distance' field: {r}"
                )
            distance = float(r["_distance"])
            score = normalize_vector_score(
                distance, VECTOR_METRIC,
                vectors_are_unit_normalized=NORMALIZE_EMBEDDINGS,
            )
            results.append(RetrievedPage(
                path=Path(r["path"]), title=r["title"], score=score,
                distance=distance, vector_metric=VECTOR_METRIC,
                snippet=r["content"][:200], sources=json.loads(r.get("sources", "[]")),
                retrieval_method="vector",
            ))
        return results


def main():
    # ISSUE-06：argparse 替代手写 argv
    import argparse
    p = argparse.ArgumentParser(
        prog="build_index.py",
        description="构建 BM25 + LanceDB 向量索引",
    )
    p.add_argument("project_root", help="知识库项目根目录（含 Wiki/）")
    args = p.parse_args()
    proj = Path(args.project_root)
    wiki = proj / "Wiki"
    idx_dir = proj / ".index"
    wi = WikiIndex(idx_dir)
    wi.build(wiki)
    print(f"索引构建完成: {len(wi.pages)} 页, manifest → {idx_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
