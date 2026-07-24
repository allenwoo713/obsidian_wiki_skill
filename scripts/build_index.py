"""Wiki 索引构建：分层分块 + LanceDB 原生 FTS + 自适应向量索引 + manifest。

Retrieval v2（GitHub issues #1/#2/#8）：
- #1 分层分块：scripts/chunking.py 的 ChunkRecord（Page→Section→Sparse/Dense）。
- #2 FTS：LanceDB 原生 FTS（`tokenizer_name="whitespace"`）+ 应用层
  lexical_tokenizer 预分词，彻底规避 LANCE_LANGUAGE_MODEL_HOME 运行时模型下载。
- #8 向量索引自适应：exact → IVF_HNSW_FLAT → IVF_HNSW_SQ（按数据量自动选择）。

用法：python build_index.py <project_root>
"""
from __future__ import annotations
import json
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import _config  # noqa: F401  # 加载 <skill_dir>/.env（ISSUE-01），须在下方 setdefault 之前执行

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from models import (
    WikiPage, RetrievedPage, ManifestEntry,
    IndexState, ChunkHit, PageCandidate,
)
import chunking
from chunking import chunk_page, CHUNK_SCHEMA_VERSION
from lexical_tokenizer import fts_terms, extract_exact_terms, load_lexicon
from vector_scoring import apply_vector_metric, normalize_vector_score


# ISSUE-16（关键）：pyarrow 必须早于 torch 导入，否则在「已加载 torch 的进程里再
# import pyarrow（经 lancedb）」会触发 Windows access violation 段错误（RC=139）。
# 故在模块导入期 *先* 引入 lancedb（间接加载 pyarrow），再让下方 torch 配置导入 torch。
try:
    import lancedb  # noqa: F401  # 仅为固定「pyarrow 先于 torch」的导入顺序
except Exception:
    lancedb = None  # 无 lancedb 时向量索引不可用；延后到使用点报错


def _configure_torch_threads():
    """ISSUE-16：固定 torch CPU intra-op 线程数，须在模块导入期（任何 torch 并行区
    初始化之前）调用。默认 1；稳定的大机器可用 WIKI_TORCH_THREADS 调高提速。
    """
    try:
        import torch
        n = int(os.environ.get("WIKI_TORCH_THREADS", "1") or "1")
        torch.set_num_threads(max(1, n))
        torch.set_grad_enabled(False)  # 推理无需梯度，省内存并减少线程活动
    except Exception:
        pass


_configure_torch_threads()


# ISSUE-15：向量检索 metric contract —— 固定配置，索引侧与查询侧一致
VECTOR_METRIC = "cosine"
NORMALIZE_EMBEDDINGS = False
VECTOR_ENCODE_BATCH = 64   # 每次 encode 的切片数，控制内存峰值

# #8 自适应向量索引阈值（按数据量自动选择索引类型）
EXACT_INDEX_MAX_ROWS = 4096        # 低于此：不建 ANN 索引，LanceDB 暴力精确检索（recall=1）
SQ_INDEX_MIN_ROWS = 200_000        # 高于此：IVF_HNSW_SQ 标量量化省内存（recall≥0.98）

# page-level RRF 常量
RRF_K = 60


def page_id_of(path) -> str:
    """稳定 page 标识：解析后的绝对路径（保留真实大小写，见工作区 memory norm_key 修复）。"""
    return str(Path(path).resolve())


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
    """分层分块 + LanceDB FTS + 自适应向量索引，支持增量构建。

    表结构（LanceDB ``chunks`` 表）：
      chunk_id, page_id, path, title, page_type, section_path(json), heading,
      chunk_kind('dense'|'sparse'), chunk_index, parent_section_id, text,
      fts_text(应用层预分词), token_count, content_hash, vector
    - dense leaf chunk：带向量 + fts_text（主检索单元）
    - sparse section chunk：仅 fts_text（向量列填零向量，向量检索由 chunk_kind 过滤）
    """

    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.pages: List[WikiPage] = []
        self._page_by_id: Dict[str, WikiPage] = {}
        self._embedder = None
        self._lance_table = None
        self._lexicon = set()
        self._project_root: Optional[Path] = None

    # ---- embedder ----
    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            local_path = os.environ.get("WIKI_EMBEDDER_LOCAL_PATH") or \
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.dirname(os.path.abspath(__file__))))),
                    "binaries", "python", "envs", "default", "models",
                    "paraphrase-multilingual-MiniLM-L12-v2")
            candidate_paths = [
                local_path,
                os.path.expanduser("~/.workbuddy/binaries/python/envs/default/models/paraphrase-multilingual-MiniLM-L12-v2"),
            ]
            for p in candidate_paths:
                if os.path.isdir(p) and os.path.exists(os.path.join(p, "model.safetensors")):
                    self._embedder = SentenceTransformer(p)
                    return self._embedder
            self._embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        return self._embedder

    def count_tokens(self, text: str) -> int:
        """用 embedding 模型的 tokenizer 估算 token 数；缺省回退 char//4。"""
        try:
            emb = self._get_embedder()
            tok = getattr(emb, "tokenizer", None)
            if tok is not None:
                return len(tok.encode(text))
        except Exception:
            pass
        return max(1, len(text) // 4)

    def _embedding_dim(self) -> int:
        emb = self._get_embedder()
        if hasattr(emb, "get_embedding_dimension"):
            return emb.get_embedding_dimension()
        return emb.get_sentence_embedding_dimension()

    # ---- LanceDB ----
    def _get_lance_table(self, create_if_missing: bool = False, dim: int = None,
                         sample: dict = None):
        if self._lance_table is None:
            import lancedb
            db = lancedb.connect(str(self.index_dir / "lance_db"))
            try:
                self._lance_table = db.open_table("chunks")
            except Exception:
                if not create_if_missing or sample is None:
                    raise
                vec_dim = dim or 384
                row = dict(sample)
                row["vector"] = [0.0] * vec_dim
                self._lance_table = db.create_table("chunks", data=[row])
                self._lance_table.delete("chunk_id != ''")
        return self._lance_table

    # ---- build ----
    def build(self, wiki_dir: Path):
        self._project_root = wiki_dir.parent
        self._lexicon = load_lexicon(self._project_root)
        self.pages = scan_wiki(wiki_dir, self._project_root)
        image_pages = self._load_image_caption_pages(wiki_dir.parent / ".index")
        self.pages.extend(image_pages)
        self._page_by_id = {page_id_of(p.path): p for p in self.pages}
        self._build_chunks()
        self._write_manifest()

    def _load_image_caption_pages(self, idx_dir: Path) -> List[WikiPage]:
        manifest_file = idx_dir / "manifest.json"
        if not manifest_file.exists():
            return []
        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            import logging
            logging.getLogger(__name__).warning("_load_image_caption_pages: manifest 解析失败 %s: %s", manifest_file, e)
            return []
        wiki_dir = idx_dir.parent / "Wiki"
        pages = []
        skipped = 0
        for img in manifest.get("images", []):
            caption = (img.get("caption_text") or "").strip()
            if not caption:
                vlm = img.get("vlm_caption") or {}
                caption = (vlm.get("description") or "").strip()
            if not caption:
                continue
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
                "_load_image_caption_pages: 跳过 %d 条 rel_path 缺失的图片条目", skipped)
        return pages

    def _chunk_rows_for_page(self, p: WikiPage, dim: int):
        """为单页生成 chunks 表的行（dense leaf + sparse section）。"""
        pid = page_id_of(p.path)
        rows = []
        try:
            chunks = chunk_page(
                page_id=pid, path=p.path, title=p.title, page_type=p.page_type,
                content=p.content, tokenizer=None,  # char 估算兜底；不在此加载 embedder（规避早启 torch/pyarrow 冲突）
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("chunk_page 失败 %s: %s", p.path, e)
            return rows
        for cr in chunks:
            fts = " ".join(fts_terms(cr.text, self._lexicon) + extract_exact_terms(cr.text))
            is_dense = (cr.chunk_kind == "dense")
            rows.append({
                "chunk_id": f"{CHUNK_SCHEMA_VERSION}:{pid}:{cr.chunk_kind}:{cr.chunk_index}",
                "page_id": pid,
                "path": str(p.path),
                "title": p.title,
                "page_type": p.page_type,
                "section_path": json.dumps(cr.section_path, ensure_ascii=False),
                "heading": cr.heading,
                "chunk_kind": cr.chunk_kind,
                "chunk_index": cr.chunk_index,
                "parent_section_id": cr.parent_section_id or "",
                "text": cr.text,
                "fts_text": fts,
                "token_count": cr.token_count,
                "content_hash": cr.content_hash,
                # dense 带真实向量；sparse 填零向量（向量检索由 chunk_kind 过滤）
                "vector": [0.0] * dim,
            })
        return rows

    def _build_chunks(self):
        import numpy as np
        import gc
        embedder = self._get_embedder()
        dim = self._embedding_dim()

        # 1) 生成所有 chunk 元数据行（向量暂填零，稍后回填 dense 行）
        all_rows: List[dict] = []
        dense_texts: List[str] = []
        dense_row_idx: List[int] = []   # all_rows 中对应 dense 行的下标
        for p in self.pages:
            rows = self._chunk_rows_for_page(p, dim)
            for r in rows:
                if r["chunk_kind"] == "dense":
                    dense_texts.append(r["text"])
                    dense_row_idx.append(len(all_rows))
                all_rows.append(r)
        if not all_rows:
            return

        # 2) crash-safe 断点续的分批 encode（ISSUE-16：仅 encode dense 行）
        n_batches = (len(dense_texts) + VECTOR_ENCODE_BATCH - 1) // VECTOR_ENCODE_BATCH
        ckpt = self.index_dir / ".vec_ckpt"
        ckpt.mkdir(parents=True, exist_ok=True)
        done_path = ckpt / "done.json"
        meta_path = ckpt / "meta.json"
        sig = {"n_dense": len(dense_texts), "batch": VECTOR_ENCODE_BATCH,
               "dim": dim, "model": os.environ.get("WIKI_EMBEDDER_LOCAL_PATH", "")}
        done = set()
        prev_sig = None
        if meta_path.exists():
            try:
                prev_sig = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:
                prev_sig = None
        if prev_sig == sig and done_path.exists():
            try:
                done = {int(x) for x in json.loads(done_path.read_text(encoding="utf-8"))}
            except Exception:
                done = set()
        else:
            for old in ckpt.glob("batch_*.npy"):
                try:
                    old.unlink()
                except Exception:
                    pass
        meta_path.write_text(json.dumps(sig, ensure_ascii=False), encoding="utf-8")
        for bi in range(n_batches):
            if bi in done:
                continue
            s = bi * VECTOR_ENCODE_BATCH
            batch = dense_texts[s:s + VECTOR_ENCODE_BATCH]
            v = embedder.encode(batch, show_progress_bar=False,
                                 normalize_embeddings=NORMALIZE_EMBEDDINGS)
            np.save(ckpt / f"batch_{bi:05d}.npy", np.asarray(v, dtype="float32"))
            done.add(bi)
            done_path.write_text(json.dumps(sorted(done)), encoding="utf-8")

        # 写 lance 前释放 embedder（内存卫生）
        self._embedder = None
        embedder = None
        gc.collect()

        # 从 checkpoint 回载向量，回填 dense 行
        vectors = []
        for bi in range(n_batches):
            vectors.extend(np.load(ckpt / f"batch_{bi:05d}.npy").tolist())
        for row_pos, vec in zip(dense_row_idx, vectors):
            all_rows[row_pos]["vector"] = vec

        # 3) 写 LanceDB chunks 表（建表用首行 schema，再 delete 占位，分批 add）
        table = self._get_lance_table(create_if_missing=True, dim=dim,
                                      sample=all_rows[0])
        try:
            table.delete("chunk_id != ''")
        except Exception as e:
            import logging
            logging.getLogger(__name__).debug("_build_chunks: 跳过 delete（可能首次建表）: %s", e)
        for i in range(0, len(all_rows), 2000):
            table.add(all_rows[i:i + 2000])

        # 4) FTS 索引（#2：whitespace 预分词，规避中文模型下载）
        try:
            table.create_fts_index("fts_text", tokenizer_name="whitespace", replace=True)
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("create_fts_index 失败（FTS 检索不可用）: %s", e)

        # 5) 自适应向量索引（#8）
        self._build_vector_index(table, len(all_rows), dim)

        # 清理 checkpoint
        try:
            import shutil
            shutil.rmtree(ckpt, ignore_errors=True)
        except Exception:
            pass

    def _build_vector_index(self, table, n_rows: int, dim: int):
        """#8 自适应向量索引：exact → IVF_HNSW_FLAT → IVF_HNSW_SQ。"""
        if n_rows <= EXACT_INDEX_MAX_ROWS:
            return  # 数据量小：LanceDB 暴力精确检索，recall=1
        index_type = "IVF_HNSW_SQ" if n_rows >= SQ_INDEX_MIN_ROWS else "IVF_HNSW_FLAT"
        num_partitions = max(2, int(math.sqrt(n_rows)))
        try:
            table.create_index(
                "vector", metric=VECTOR_METRIC, index_type=index_type,
                num_partitions=num_partitions,
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(
                "create_index(%s) 失败，回退默认索引: %s", index_type, e)
            try:
                table.create_index("vector", metric=VECTOR_METRIC)
            except Exception as e2:
                logging.getLogger(__name__).warning("默认 create_index 也失败: %s", e2)

    def _write_manifest(self):
        manifest_file = self.index_dir / "manifest.json"
        existing = {}
        if manifest_file.exists():
            try:
                existing = json.loads(manifest_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                import logging
                logging.getLogger(__name__).warning(
                    "_write_manifest: 现有 manifest 解析失败，将覆盖重建: %s", e)
        existing["built_at"] = datetime.now().isoformat()
        existing["page_count"] = len(self.pages)
        # #1/#2/#8 索引状态契约（manifest v2 `index_state`）
        try:
            emb = self._get_embedder()
            model_name = getattr(emb, "model_name", "") or str(
                os.environ.get("WIKI_EMBEDDER_LOCAL_PATH", "paraphrase-multilingual-MiniLM-L12-v2"))
            dim = self._embedding_dim()
        except Exception:
            model_name = os.environ.get("WIKI_EMBEDDER_LOCAL_PATH", "")
            dim = 384
        state = IndexState(
            embedding_model=model_name,
            embedding_dimension=dim,
            vector_metric=VECTOR_METRIC,
            fts_config_hash="whitespace+" + ("jieba" if _jieba_available() else "bigram"),
            chunk_config_hash=f"v{CHUNK_SCHEMA_VERSION}:{chunking.DENSE_TARGET_TOKENS}:{chunking.DENSE_OVERLAP_TOKENS}",
        )
        existing["index_state"] = {
            "schema_version": state.schema_version,
            "chunk_schema_version": state.chunk_schema_version,
            "tokenizer_schema_version": state.tokenizer_schema_version,
            "embedding_model": state.embedding_model,
            "embedding_dimension": state.embedding_dimension,
            "vector_metric": state.vector_metric,
            "fts_config_hash": state.fts_config_hash,
            "chunk_config_hash": state.chunk_config_hash,
        }
        existing["pages"] = [
            {
                "path": str(p.path),
                "page_id": page_id_of(p.path),
                "sha256": p.sha256,
                "page_type": p.page_type,
                "title": p.title,
                "sources": p.sources,
                "links": p.links,
            }
            for p in self.pages
        ]
        manifest_file.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---- load ----
    def load(self):
        manifest_file = self.index_dir / "manifest.json"
        if not manifest_file.exists():
            raise RuntimeError("索引未找到，请先运行 build_index.py")
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        st = manifest.get("index_state")
        if st is None:
            raise RuntimeError(
                "Legacy index detected (no index_state in manifest). "
                "Rebuild the index: python build_index.py <project_root>")
        if st.get("vector_metric") != VECTOR_METRIC:
            raise RuntimeError(
                f"Vector metric mismatch: manifest has '{st.get('vector_metric')}', "
                f"current code expects '{VECTOR_METRIC}'. Rebuild the index.")
        self._project_root = Path(self.index_dir).parent
        self._lexicon = load_lexicon(self._project_root)
        self._page_by_id = {}
        self.pages = []
        for p in manifest.get("pages", []):
            wp = WikiPage(
                path=Path(p["path"]), title=p.get("title", Path(p["path"]).stem),
                page_type=p.get("page_type", "concept"), content="",
                sources=p.get("sources", []), links=p.get("links", []),
                sha256=p.get("sha256", ""),
            )
            self.pages.append(wp)
            self._page_by_id[p.get("page_id", page_id_of(p["path"]))] = wp
        # 触发 LanceDB 表打开
        self._get_lance_table()

    # ---- search ----
    def search_fts(self, query: str, k: int = 20) -> List[ChunkHit]:
        """LanceDB 原生 FTS（whitespace 预分词）。返回 chunk 级命中。"""
        q = " ".join(fts_terms(query, self._lexicon) + extract_exact_terms(query))
        table = self._get_lance_table()
        try:
            rows = table.search(q, query_type="fts").limit(k * 4).to_list()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("search_fts 失败: %s", e)
            return []
        return [self._hit_from_row(r, "fts") for r in rows]

    def search_vector(self, query: str, k: int = 20) -> List[ChunkHit]:
        """向量检索（仅 dense 行）。返回 chunk 级命中（按 page_id 归并前）。"""
        embedder = self._get_embedder()
        qv = embedder.encode([query], show_progress_bar=False,
                             normalize_embeddings=NORMALIZE_EMBEDDINGS)[0].tolist()
        table = self._get_lance_table()
        qb = apply_vector_metric(table.search(qv), VECTOR_METRIC)
        qb = qb.where("chunk_kind = 'dense'")
        try:
            rows = qb.limit(k * 4).to_list()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning("search_vector 失败: %s", e)
            return []
        return [self._hit_from_row(r, "vector") for r in rows]

    def search(self, query: str, k: int = 5) -> List[PageCandidate]:
        """端到端：chunk 级 FTS + 向量 → page-level RRF → 返回 PageCandidate。"""
        from fusion import page_level_rrf
        fts = self.search_fts(query, k=20)
        vec = self.search_vector(query, k=20)
        return page_level_rrf(fts, vec, k=k, k_rrf=RRF_K)

    def _hit_from_row(self, r, channel: str) -> ChunkHit:
        if channel == "fts":
            score = float(r.get("_score", 0.0))
            distance = None
        else:
            if "_distance" not in r:
                raise RuntimeError(f"LanceDB result missing '_distance' field: {r}")
            distance = float(r["_distance"])
            score = normalize_vector_score(
                distance, VECTOR_METRIC,
                vectors_are_unit_normalized=NORMALIZE_EMBEDDINGS)
        return ChunkHit(
            chunk_id=r["chunk_id"], page_id=r["page_id"], path=r["path"],
            title=r["title"], page_type=r["page_type"],
            section_path=json.loads(r.get("section_path") or "[]"),
            heading=r.get("heading", ""), chunk_kind=r["chunk_kind"],
            text=r["text"], channel=channel, score=score, distance=distance,
        )


def _jieba_available() -> bool:
    try:
        import lexical_tokenizer as _lt
        return _lt._HAS_JIEBA
    except Exception:
        return False


def main():
    import argparse
    p = argparse.ArgumentParser(
        prog="build_index.py",
        description="构建 分层分块 + LanceDB FTS + 自适应向量索引",
    )
    p.add_argument("project_root", help="知识库项目根目录（含 Wiki/）")
    args = p.parse_args()
    proj = Path(args.project_root)
    wiki = proj / "Wiki"
    idx_dir = proj / ".index"
    wi = WikiIndex(idx_dir)
    wi.build(wiki)
    print(f"索引构建完成: {len(wi.pages)} 页 → chunks 表, manifest → {idx_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
