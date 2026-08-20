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
import hashlib
import math
import os
import re
import sys
import time
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from types import SimpleNamespace


# The embedding model is an asset of this skill.  Keep it beside the skill
# instead of falling back to an unrelated user-level cache: that makes builds
# reproducible and lets a copied skill remain self-contained.
SKILL_ROOT = Path(__file__).resolve().parent.parent
EMBEDDING_MODEL_ID = "paraphrase-multilingual-MiniLM-L12-v2"
SKILL_EMBEDDER_DIR = SKILL_ROOT / "models" / EMBEDDING_MODEL_ID

# ISSUE-16 + 0xC0000005 修复：
#  - pyarrow 必须早于 torch 导入（在已加载 torch 的进程里再 import pyarrow 会触发
#    Windows access violation，RC=139）。
#  - torch 必须在任何可能拉起后台 asyncio 事件循环线程的导入（如 lancedb）之前完成
#    原生模块加载。否则在部分宿主（如 PowerShell 启动的 managed-python）下，torch 原生
#    加载会与宿主注入的后台事件循环线程时序 race → 0xC0000005（无 traceback、~8s 崩溃）。
#    故把 pyarrow + torch 提到模块最顶部（仅晚于标准库），并立即固定线程数。
import pyarrow  # noqa: F401  # 先于 torch（ISSUE-16）
import torch
# 多线程 encode 实测安全且 ~2.7x 更快（5120 条压测：8 线程 88s vs 1 线程 ~240s，
# faulthandler 80 批次无段错误；完整 build 路径 8 线程 test_index_safety 全过）。
# 早期「强制单线程」是对 0xC0000005 的误判兜底——真正根因是 torch 原生加载与宿主
# asyncio 后台线程的时序 race，已由上方「torch 先行 import」修复；多线程 encode
# 本身稳定。默认 min(cpu, 8)，WIKI_TORCH_THREADS 可覆盖（设 1 回退单线程）。
_default_threads = min(os.cpu_count() or 4, 8)
_wiki_threads = int(os.environ.get("WIKI_TORCH_THREADS") or _default_threads)
torch.set_num_threads(max(1, _wiki_threads))
torch.set_grad_enabled(False)  # 推理无需梯度，省内存并减少线程活动

import _config  # noqa: F401  # 加载 <skill_dir>/.env（ISSUE-01），须在下方 setdefault 之前执行

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

from models import (
    WikiPage, RetrievedPage, ManifestEntry,
    IndexState, ChunkHit, PageCandidate,
)
import chunking
from chunking import (chunk_page, CHUNK_SCHEMA_VERSION, EmbeddingTokenizer,
                      ChunkBuildError, count_token_ids)
from chunk_plan import (
    chunk_records_to_sparse,
    page_metadata_from_pages,
    plan_sparse_chunks,
)
from lexical_tokenizer import fts_terms, extract_exact_terms, load_lexicon
from vector_scoring import apply_vector_metric, normalize_vector_score
from obsidian_wiki.application.index_build_service import CandidateQueryPolicy


def _compose_storage_services(
    index_dir: Path,
    *,
    candidate_query_policy: CandidateQueryPolicy | None = None,
):
    """Compose real storage adapters once per public build request.

    This is deliberately the only production seam that knows the concrete LanceDB
    repository and filesystem journal implementations used by online indexing.
    """
    from obsidian_wiki.application.index_build_service import IndexBuildService
    from obsidian_wiki.application.index_publication_service import IndexPublicationService
    from obsidian_wiki.application.incremental_index_service import IncrementalIndexService
    from obsidian_wiki.infrastructure.filesystem_incremental_journal import FilesystemIncrementalJournal
    from obsidian_wiki.infrastructure.filesystem_index_manifest import FilesystemIndexManifest
    from obsidian_wiki.infrastructure.filesystem_post_commit_journal import FilesystemPostCommitJournal
    from obsidian_wiki.infrastructure.lancedb_index_repository import LanceDbIndexRepository

    repository_kwargs = (
        {"eval_candidate_policy": candidate_query_policy}
        if candidate_query_policy is not None else {}
    )
    repository_factory = lambda lance_dir: LanceDbIndexRepository(lance_dir, **repository_kwargs)
    journal = FilesystemPostCommitJournal(Path(index_dir))
    publication_service = IndexPublicationService()
    manifest_store = FilesystemIndexManifest()
    incremental_executor_factory = lambda: IncrementalIndexService(
        repository_factory=repository_factory,
        journal_factory=FilesystemIncrementalJournal,
        manifest_store=manifest_store,
        post_commit_journal_factory=FilesystemPostCommitJournal,
        publication_service=publication_service,
    )
    IncrementalIndexService.configure_legacy_dependencies(
        repository_factory=repository_factory,
        journal_factory=FilesystemIncrementalJournal,
        manifest_store=manifest_store,
        post_commit_journal_factory=FilesystemPostCommitJournal,
        publication_service=publication_service,
    )
    service = IndexBuildService(
        repository_factory(index_dir),
        reopen_storage=repository_factory,
        manifest_store=manifest_store,
        post_commit_journal=journal,
        candidate_query_policy=candidate_query_policy,
        publication_service=publication_service,
        incremental_executor_factory=incremental_executor_factory,
    )
    return SimpleNamespace(
        service=service,
        journal=journal,
        publication_service=publication_service,
        incremental_executor_factory=incremental_executor_factory,
    )


def build_storage_contract(wiki_dir: Path, index_dir: Path, *, embed, sparse_chunks=None,
                           page_metadata=None, image_metadata=None, ctx=None,
                           tokenizer=None, lexicon=None,
                           candidate_query_policy: CandidateQueryPolicy | None = None,
                           build_mode: str = "snapshot",
                           build_mode_policy=None,
                           outer_lock_held: bool = False):
    """Direct-script facade for the D-01/D-04 storage-contract build path.

    The public script remains the entry point while orchestration and storage are
    delegated to their package tiers.  ``embed`` is injected so the persisted
    tracer can exercise real LanceDB without loading a model in its tiny test.
    page_metadata / image_metadata are injected into the manifest before the
    single publish_pointer call (review #3: no second publish from facade).
    ``ctx`` is the outermost BuildContext (#34); generated once if omitted.

    Issue #39: when ``tokenizer`` (a ``callable[[str], int]``) is supplied, the
    chunk plan is produced with the tokenizer-aware ``plan_sparse_chunks`` so
    large pages are split into token-bounded dense leaves instead of stored as
    one whole-page dense chunk. ``sparse_chunks`` passed by the caller wins;
    otherwise the plan is computed here (the ``main()`` production path). The
    tokenizer-less ``sparse_chunks=None`` path keeps the legacy whole-page plan
    for callers/tests that do not inject a tokenizer.

    Returns ``IndexBuildOutcome`` (#37): the pointer publish is the commit point;
    subsequent community-report invalidation is post-commit work whose failure
    only surfaces as pending/warning and never masks an already-published build.
    """
    """Direct-script facade for the D-01/D-04 storage-contract build path.

    The public script remains the entry point while orchestration and storage are
    delegated to their package tiers.  ``embed`` is injected so the persisted
    tracer can exercise real LanceDB without loading a model in its tiny test.
    page_metadata / image_metadata are injected into the manifest before the
    single publish_pointer call (review #3: no second publish from facade).
    ``ctx`` 是最外层生成的 BuildContext（#34）；未传时自行生成一次。

    返回 ``IndexBuildOutcome``（#37）：pointer 发布即 commit point，其后的
    community report 失效是 post-commit 工作，失败只体现为 pending/warning，
    不会把已发布的 build 伪装成失败。
    """
    from obsidian_wiki.application.build_lock import new_build_context
    from obsidian_wiki.domain.index_models import IndexBuildOutcome

    # #34：ctx 贯穿 lock metadata、build 目录、manifest、ACTIVE_INDEX pointer 与
    # 返回 artifact；service 不再独立生成 ID。
    ctx = ctx or new_build_context()
    # Issue #39 (review)：tokenizer 注入时不再在此预计算 chunk 计划。分块必须在
    # BUILD.lock 持锁后、针对已加锁的 Wiki 快照执行（防止对并发写入中的快照分块），
    # 因此把 planner 作为回调下沉到 service._build（持锁区）内运行。planner 同时
    # 返回 canonical pages，用于生成「每源文件一逻辑页 + 全文件 SHA-256」的 manifest
    # 元数据（不再每 chunk 一行）。sparse_chunks 调用方显式传入时优先。
    plan_provider = None
    if sparse_chunks is None and tokenizer is not None:
        project_root = Path(wiki_dir).parent
        resolved_lexicon = lexicon if lexicon is not None else load_lexicon(project_root)

        def plan_provider(wiki_snapshot: Path):
            import build_index as _self  # 模块级符号，供测试 monkeypatch plan_sparse_chunks
            chunks = _self.plan_sparse_chunks(
                Path(wiki_snapshot), project_root,
                tokenizer=tokenizer, lexicon=resolved_lexicon,
            )
            # manifest：每 canonical 源文件一逻辑页（全文件 SHA-256），与 chunks 同一快照。
            pages = _self.scan_wiki(Path(wiki_snapshot), project_root)
            return chunks, page_metadata_from_pages(pages)

    composed = _compose_storage_services(
        Path(index_dir), candidate_query_policy=candidate_query_policy,
    )
    artifact = composed.service.build(
        Path(wiki_dir), Path(index_dir), embed=embed, sparse_chunks=sparse_chunks,
        page_metadata=page_metadata, image_metadata=image_metadata, ctx=ctx,
        plan_provider=plan_provider, build_mode=build_mode,
        build_mode_policy=build_mode_policy, outer_lock_held=outer_lock_held)
    # #37：pointer commit 之后执行 post-commit（可观察、可重试；失败保留 PREPARED）。
    post_commit_status, warnings = _run_post_commit(Path(index_dir), composed.journal, artifact)
    # #34：outcome 的 build_id/generation 必须来自 artifact（单一事实来源），
    # 不再从 manifest 二次解析。
    return IndexBuildOutcome(
        artifact=artifact,
        build_id=artifact.build_id,
        generation=artifact.generation,
        published=True,
        post_commit_status=post_commit_status,
        warnings=warnings,
    )


def _run_post_commit(index_dir: Path, journal, artifact):
    """执行 journal 中本 build 的 PREPARED 任务（#37）；失败保留 pending 供 retry_pending 重放。"""
    from obsidian_wiki.domain.index_models import PostCommitStatus
    from obsidian_wiki.infrastructure.filesystem_community_reports import FilesystemCommunityReportStore

    tasks = [
        t for t in journal.pending()
        if t.build_id == artifact.build_id and t.task_type == "community_report_invalidation"
    ]
    if not tasks:
        return PostCommitStatus.COMPLETE, ()
    try:
        FilesystemCommunityReportStore(index_dir).mark_stale(
            producer="build_index", reason="index_published"
        )
        for task in tasks:
            journal.complete(task.task_id)
        return PostCommitStatus.COMPLETE, ()
    except Exception as exc:
        logging.getLogger(__name__).warning(
            "community report 失效失败（post-commit pending，可重试，不影响已发布索引）：%s", exc
        )
        return PostCommitStatus.COMMUNITY_REPORT_INVALIDATION_PENDING, (
            "community report 失效未完成（post-commit pending，可重试）；索引已发布。",
        )


# 仅固定「pyarrow 先于 torch」的导入顺序（ISSUE-16）；torch 已于上方最顶部加载完毕。
try:
    import lancedb  # noqa: F401
except Exception:
    lancedb = None  # 无 lancedb 时向量索引不可用；延后到使用点报错


# ISSUE-15：向量检索 metric contract —— 固定配置，索引侧与查询侧一致
VECTOR_METRIC = "cosine"
NORMALIZE_EMBEDDINGS = False
VECTOR_ENCODE_BATCH = 64   # 每次 encode 的切片数，控制内存峰值

# #8 自适应向量索引阈值（按数据量自动选择索引类型）

# page-level RRF 常量
RRF_K = 60


def page_id_of(path) -> str:
    """稳定 page 标识：解析后的绝对路径（保留真实大小写，见工作区 memory norm_key 修复）。"""
    return str(Path(path).resolve())


_FM_RE = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)
_LINK_RE = re.compile(r"\[\[([^\]]+)\]\]")


def _read_page_content(path: Path) -> str:
    """Read complete Markdown body for the read-only ContextRepository port."""
    try:
        raw = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    match = _FM_RE.match(raw)
    return (match.group(2) if match else raw).strip()


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
    # #39 (review)：page 身份哈希必须锚定磁盘原始字节，而非 read_text 归一化后的
    # 文本（Windows 上 read_text 会把 CRLF 折成 LF，再 encode 会丢 \r，导致同一
    # 文件产生与 sha256(read_bytes()) 不一致的指纹）。用原始字节保证可复现、跨平台。
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    sources = fm.get("sources", []) or []
    if isinstance(sources, str):
        sources = [sources]
    aliases = fm.get("aliases", []) or []
    if isinstance(aliases, str):
        aliases = [aliases]
    return WikiPage(
        path=path,
        title=fm.get("title", path.stem),
        page_type=fm.get("type", "concept"),
        content=body.strip(),
        sources=[str(s) for s in sources],
        links=links,
        sha256=sha,
        aliases=[str(alias) for alias in aliases],
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
    """Compatibility facade over the D-01 split sparse/dense index contract.

    Public callers keep using ``WikiIndex``.  Storage does not: sparse retrieval
    is always native FTS against ``sparse_chunks`` and vector retrieval is always
    against ``dense_chunks``.  The retired single ``chunks`` artifact is rejected
    on load and is never created by this facade.
    """

    def __init__(self, index_dir: Path):
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.pages: List[WikiPage] = []
        self._page_by_id: Dict[str, WikiPage] = {}
        self._embedder = None
        self._lance_table = None
        self._repository = None
        self._lexicon = set()
        self._project_root: Optional[Path] = None
        # #11 索引原子发布：build 期指向新建 builds/<id>/lance_db（及 manifest 目标）；
        # 否则为 None，查询时经 _resolve_active_lance_dir() 走 ACTIVE_INDEX 指针。
        self._lance_dir: Optional[Path] = None
        # #7 增量：页级向量缓存元状态（构建期填写，供 _write_manifest 复用免加载 torch）
        self._built_dim: Optional[int] = None
        self._built_model_name: Optional[str] = None
        self._force_encode: bool = False  # --full-rebuild 时置 True，忽略页缓存强制重编码
        self._vector_index_mode: str = "auto"
        self._candidate_query_policy: CandidateQueryPolicy | None = None
        # #12 多模态：图片父文档回溯元数据（rel_path → meta），由 _load_image_meta 填充
        self._image_meta: Dict[str, dict] = {}

    # ---- embedder ----
    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            configured_path = os.environ.get("WIKI_EMBEDDER_LOCAL_PATH")
            candidate_paths = ([configured_path] if configured_path else []) + [
                str(SKILL_EMBEDDER_DIR),
            ]
            for p in candidate_paths:
                if os.path.isdir(p) and os.path.exists(os.path.join(p, "model.safetensors")):
                    self._embedder = SentenceTransformer(p)
                    return self._embedder
            raise RuntimeError(
                "本地 embedding 模型不存在。请运行 "
                f"`python scripts/download_embedding_model.py` 将 {EMBEDDING_MODEL_ID} "
                f"部署到 {SKILL_EMBEDDER_DIR}，或设置 WIKI_EMBEDDER_LOCAL_PATH。"
            )
        return self._embedder

    def count_tokens(self, text: str) -> int:
        """用 embedding 模型的 tokenizer 估算 token 数；缺省回退 char//4。"""
        try:
            emb = self._get_embedder()
            tok = getattr(emb, "tokenizer", None)
            if tok is not None:
                return count_token_ids(tok, text)
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
        raise RuntimeError(
            "The retired chunks-table accessor is unavailable. "
            "Use the split-table repository through WikiIndex search methods."
        )

    def _get_repository(self):
        """Open the active D-01 repository only after validating its manifest."""
        if self._repository is None:
            from obsidian_wiki.infrastructure.lancedb_index_repository import LanceDbIndexRepository

            lance_dir = self._resolve_active_lance_dir()
            LanceDbIndexRepository.require_current_layout(lance_dir.parent / "manifest.json")
            self._repository = LanceDbIndexRepository(lance_dir)
        return self._repository

    # ---- 活动索引解析（#11/#21 指针方案） ----
    def _resolve_active_lance_dir(self) -> Path:
        """经 ACTIVE_INDEX 指针解析当前活动 lance 目录；指针损坏时回退最近已验证 build。"""
        from obsidian_wiki.application.active_index_pointer import resolve_active_lance_dir
        return resolve_active_lance_dir(self.index_dir)

    def _resolve_active_manifest(self) -> Path:
        return self._resolve_active_lance_dir().parent / "manifest.json"

    def _content_hashes(self):
        """#11 内容签名：tokenizer / 分块配置哈希；变化时作废旧向量、强制全量重 encode。

        issue #47：sparse / dense schema 版本独立，sparse-only 改动（如 block-first
        重构）不使未变的 dense 向量缓存失效 → 第二元组仍用 DENSE_CHUNK_SCHEMA_VERSION。
        """
        return (
            "whitespace+" + ("jieba" if _jieba_available() else "bigram") + f":v{chunking.SPARSE_CHUNK_SCHEMA_VERSION}",
            f"v{chunking.DENSE_CHUNK_SCHEMA_VERSION}:{chunking.DENSE_TARGET_TOKENS}:{chunking.DENSE_OVERLAP_TOKENS}",
        )

    # ---- build ----
    def build(self, wiki_dir: Path, full_rebuild: bool = False,
               allow_partial_index: bool = False,
               candidate_query_policy: CandidateQueryPolicy | None = None,
               build_mode: str = "snapshot",
               build_mode_policy_path: Path | None = None,
               build_mode_policy=None):
        """Build a snapshot, staged incremental candidate, or evidence-gated auto mode.

        无论增量与否都写入全新 builds/<id>/lance_db（删除页自然不入表 → 无残留），
        校验通过后经 #11 指针原子发布。增量的加速点在于跳过未变页的 torch 编码。

        ``allow_partial_index``：默认 False（fail-fast）。任一页面分块失败或缺页/0-chunk
        都会中止 staging build 并保留旧活动索引（#13 review Gap 1）。置 True 仅用于实验，
        降级为 warning，禁止用于生产发布。

        #21 单写者：整个构建（含 embed 阶段）持有 .index/BUILD.lock，并发构建只有一个写者。
        #34：最外层 facade 生成一次不可变 BuildContext，贯穿两层锁 metadata、build
        目录、manifest、ACTIVE_INDEX pointer 与返回 artifact；内层不再独立生成 ID。

        Phase 06（issue #49）：``vector_index_mode`` 已移除——生产构建固定使用
        批准策略（IVF_HNSW_SQ / ef=100）；``candidate_query_policy`` 仅用于显式
        eval comparator 构建。
        """
        from obsidian_wiki.application.build_lock import BuildLock, new_build_context
        from obsidian_wiki.application.incremental_policy import load_build_mode_policy

        if build_mode not in {"snapshot", "incremental", "auto"}:
            raise ValueError("build_mode must be snapshot, incremental, or auto")
        project_root = Path(wiki_dir).parent
        # Only auto reads the policy: explicit snapshot ignores even an invalid path,
        # while explicit incremental remains policy-independent.
        policy_load = build_mode_policy if build_mode == "auto" else None
        if build_mode == "auto" and policy_load is None:
            policy_load = load_build_mode_policy(project_root, build_mode_policy_path)

        ctx = new_build_context()
        lock = BuildLock(self.index_dir, ctx=ctx)
        lock.acquire()
        try:
            return self._build(wiki_dir, full_rebuild=full_rebuild,
                               allow_partial_index=allow_partial_index, ctx=ctx,
                               candidate_query_policy=candidate_query_policy,
                               build_mode=build_mode, build_mode_policy=policy_load)
        finally:
            lock.release()

    def _build(self, wiki_dir: Path, full_rebuild: bool = False,
               allow_partial_index: bool = False, ctx=None,
               candidate_query_policy: CandidateQueryPolicy | None = None,
               build_mode: str = "snapshot", build_mode_policy=None):
        """build() 的锁内主体。"""
        # Keep the long-standing public call signature, but route every actual
        # build through the D-01 service.  #22 incremental/publisher work and
        # #23 ranking remain outside this compatibility migration.
        embedder = self._get_embedder()

        def embed(texts):
            vectors = embedder.encode(
                list(texts), show_progress_bar=False,
                normalize_embeddings=NORMALIZE_EMBEDDINGS,
            )
            return [list(vector) for vector in vectors]

        from obsidian_wiki.domain.index_models import SparseChunk

        self._project_root = Path(wiki_dir).parent
        self._lexicon = load_lexicon(self._project_root)
        # Preserve the established image-caption ingestion path.  The source
        # manifest lives at the index root before the first D-01 publication;
        # its image metadata is copied into the staged manifest below.
        source_manifest = self.index_dir / "manifest.json"
        try:
            source_images = json.loads(source_manifest.read_text(encoding="utf-8")).get("images", [])
        except (OSError, json.JSONDecodeError):
            source_images = []
        self.pages = scan_wiki(Path(wiki_dir), self._project_root)
        self.pages.extend(self._load_image_caption_pages(self.index_dir))
        self._page_by_id = {page_id_of(page.path): page for page in self.pages}
        tokenizer = EmbeddingTokenizer(getattr(embedder, "tokenizer", None))
        canonical_chunks = []
        for page in self.pages:
            page_id = page_id_of(page.path)
            try:
                records = list(chunk_page(
                    page_id=page_id, path=page.path, title=page.title,
                    page_type=page.page_type, content=page.content, tokenizer=tokenizer.count,
                ))
            except Exception as exc:
                self._mark_preflight_failure(exc)
                raise ChunkBuildError(
                    f"chunk_page 失败: page_id={page_id}, path={page.path}"
                ) from exc
            kinds = {record.chunk_kind for record in records}
            if page.content.strip() and (not records or kinds != {"dense", "sparse"}):
                message = (
                    f"索引完整性校验失败：非空页面 {page_id} 的 retrieval kinds="
                    f"{sorted(kinds)}，期望 ['dense', 'sparse']"
                )
                if allow_partial_index:
                    logging.getLogger(__name__).warning(message)
                    continue
                self._mark_preflight_failure(RuntimeError(message))
                raise ChunkBuildError(message)
            # issue #39：复用共享映射，避免与 build_storage_contract 路径重复分块逻辑。
            canonical_chunks.extend(chunk_records_to_sparse(records, self._lexicon))
        dense_sources = [chunk for chunk in canonical_chunks if chunk.chunk_kind == "dense"]
        vectors_by_position = self._cached_dense_vectors(
            dense_sources, embedder, full_rebuild=full_rebuild
        )

        def embed_cached(texts):
            if len(texts) != len(vectors_by_position):
                raise RuntimeError("Dense chunk plan changed during cached embedding")
            return vectors_by_position

        page_metadata = [
            {
                "page_id": page_id_of(page.path), "path": str(page.path),
                "title": page.title, "page_type": page.page_type,
                "sources": page.sources, "links": page.links,
                "aliases": page.aliases, "sha256": page.sha256,
            }
            for page in self.pages
        ] if self.pages else None

        # #34：facade 返回 IndexBuildOutcome（不再是 None），真实 CLI/facade 契约可测。
        outcome = build_storage_contract(
            Path(wiki_dir), self.index_dir, embed=embed_cached,
            sparse_chunks=canonical_chunks,
            page_metadata=page_metadata,
            image_metadata=source_images if source_images else None,
            ctx=ctx,
            candidate_query_policy=candidate_query_policy,
            build_mode=build_mode,
            build_mode_policy=build_mode_policy,
            outer_lock_held=True,
        )
        published_manifest = json.loads(
            outcome.artifact.manifest_path.read_text(encoding="utf-8")
        )
        # Phase 06：固定策略——manifest 记录 ann_policy / candidate_query_policy，
        # 不再有运行时 mode 状态。
        self._vector_index_mode = (
            candidate_query_policy.candidate
            if candidate_query_policy is not None
            else published_manifest.get("ann_policy", {}).get("selected_index_type", "ivf-hnsw-sq")
        )
        self._candidate_query_policy = candidate_query_policy
        # #21 review #3: single publication — service publishes once with
        # complete manifest (pages/images injected before publish_pointer).
        self._repository = None
        self._lance_table = None
        return outcome

    def _mark_preflight_failure(self, exc: Exception) -> None:
        """Keep the fail-fast forensic marker even when chunking fails pre-stage."""
        build_dir = self.index_dir / "builds" / f"build_{time.time_ns()}_{uuid.uuid4().hex}"
        try:
            build_dir.mkdir(parents=True, exist_ok=False)
            (build_dir / ".failed").write_text(
                f"chunk plan failed before persistence: {type(exc).__name__}: {exc}",
                encoding="utf-8",
            )
        except OSError:
            pass

    def _cached_dense_vectors(self, dense_chunks, embedder, *, full_rebuild: bool):
        """Retain the existing snapshot-build page cache without touching D-01 storage.

        Cache files are a local build acceleration only.  The published artifact
        remains a fresh, fully validated pair of LanceDB tables on every build.
        """
        from collections import OrderedDict
        import numpy as np

        model_tag = os.path.basename(
            os.environ.get("WIKI_EMBEDDER_LOCAL_PATH", "") or SKILL_EMBEDDER_DIR.name
        )
        namespace = re.sub(r"[^0-9A-Za-z._-]", "_", f"{model_tag}__v{CHUNK_SCHEMA_VERSION}")
        cache_dir = self.index_dir / "vec_cache" / namespace
        cache_dir.mkdir(parents=True, exist_ok=True)
        by_page = OrderedDict()
        for position, chunk in enumerate(dense_chunks):
            by_page.setdefault(chunk.page_id, []).append(position)
        result = [None] * len(dense_chunks)
        current_files = set()
        misses = []
        hits = 0
        dimension = self._embedding_dim()
        for _page_id, positions in by_page.items():
            digest = hashlib.sha256()
            for position in positions:
                digest.update(dense_chunks[position].content_hash.encode("utf-8"))
                digest.update(b"\x1f")
            cache_file = cache_dir / f"{digest.hexdigest()}.npy"
            current_files.add(cache_file.name)
            if not full_rebuild and cache_file.exists():
                try:
                    cached = np.load(cache_file)
                    if cached.shape == (len(positions), dimension):
                        for position, vector in zip(positions, cached.tolist()):
                            result[position] = vector
                        hits += 1
                        continue
                except Exception:
                    pass
            misses.append((cache_file, positions))
        if misses:
            miss_positions = [position for _file, positions in misses for position in positions]
            encoded = embedder.encode(
                [dense_chunks[position].text for position in miss_positions],
                show_progress_bar=False, normalize_embeddings=NORMALIZE_EMBEDDINGS,
            )
            for position, vector in zip(miss_positions, encoded):
                result[position] = list(vector)
            cursor = 0
            for cache_file, positions in misses:
                count = len(positions)
                np.save(cache_file, np.asarray(encoded[cursor:cursor + count], dtype="float32"))
                cursor += count
        for cache_file in cache_dir.glob("*.npy"):
            if cache_file.name not in current_files:
                try:
                    cache_file.unlink()
                except OSError:
                    pass
        logging.getLogger(__name__).info(
            "#7 增量向量缓存: 命中 %d 页, 需编码 %d 页 / %d dense chunks（force=%s）",
            hits, len(misses), len(dense_chunks), full_rebuild,
        )
        return result

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

    def _load_image_meta(self):
        """#12 多模态：从 manifest images[] 加载图片父文档回溯元数据。

        每条记录按 rel_path 索引，含 source_doc/source_page/source_section/
        parent_page_id/nearby_text 等可选字段（由 parser 填充，缺失则回退标记）。
        查询期 assemble_context 据 get_image_meta() 回溯父文档/页码/附近正文。
        """
        try:
            manifest_file = self._resolve_active_manifest()
        except Exception:
            manifest_file = self.index_dir / "manifest.json"
        if not manifest_file.exists():
            self._image_meta = {}
            return
        try:
            manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            self._image_meta = {}
            return
        meta: Dict[str, dict] = {}
        for img in manifest.get("images", []):
            rel = img.get("rel_path")
            if not rel:
                continue
            meta[rel] = {
                "source_doc": img.get("source_doc", ""),
                "source_page": img.get("source_page"),
                "source_section": img.get("source_section"),
                "parent_page_id": img.get("parent_page_id"),
                "nearby_text": img.get("nearby_text", ""),
                "figure_caption": img.get("figure_caption", ""),
                "caption_text": img.get("caption_text", ""),
            }
        self._image_meta = meta

    def get_image_meta(self, path) -> Optional[dict]:
        """#12：按图片路径（绝对路径或 rel_path）查父文档回溯元数据。"""
        if not self._image_meta:
            return None
        p = str(path).replace("\\", "/")
        if p in self._image_meta:
            return self._image_meta[p]
        for rel, m in self._image_meta.items():
            if p.endswith(rel):
                return m
        return None

    def _chunk_rows_for_page(self, p: WikiPage, dim: int, tokenizer):
        """为单页生成 chunks 表的行（dense leaf + sparse section）。

        ``tokenizer`` 必须是真实 embedding tokenizer（``EmbeddingTokenizer.count``），
        由构建路径注入；禁止传 ``None`` 回退字符估算（issue #13）。
        """
        pid = page_id_of(p.path)
        rows = []
        if tokenizer is None:
            raise RuntimeError(
                f"chunk_page 必须注入真实 tokenizer（{pid}）；禁止 char 估算静默回退")
        # #13 review (Gap 1)：单页分块失败必须让 staging build 失败并保留旧活动索引，
        # 不得静默吞异常、返回 [] 导致漏页发布。包装为领域异常携带 page_id/path。
        try:
            chunks = chunk_page(
                page_id=pid, path=p.path, title=p.title, page_type=p.page_type,
                content=p.content, tokenizer=tokenizer,
            )
        except Exception as exc:
            raise ChunkBuildError(
                f"chunk_page 失败: page_id={pid}, path={p.path}") from exc
        for cr in chunks:
            fts = " ".join(fts_terms(cr.text, self._lexicon) + extract_exact_terms(cr.text))
            rows.append({
                # #13：直接使用 ChunkRecord.chunk_id（内容哈希，稳定），不再改写成
                # schema:page_id:kind:index（位置相关，会漂移）。
                "chunk_id": cr.chunk_id,
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
                # #13 review (Gap 2)：sparse 超长强制切片元数据（硬上限守卫的显式记录）
                "forced_split": bool(getattr(cr, "forced_split", False)),
                "continuation_index": int(getattr(cr, "continuation_index", -1)),
                # #13：保留真实原文 span（映射回原文、覆盖 chunk 实际正文）
                "start_char": cr.start_char,
                "end_char": cr.end_char,
                # dense 带真实向量；sparse 填零向量（向量检索由 chunk_kind 过滤）
                "vector": [0.0] * dim,
            })
        return rows



    # ---- load ----
    def load(self):
        manifest_file = self._resolve_active_manifest()
        if not manifest_file.exists():
            raise RuntimeError("索引未找到，请先运行 build_index.py")
        from obsidian_wiki.infrastructure.lancedb_index_repository import LanceDbIndexRepository
        LanceDbIndexRepository.require_current_layout(manifest_file)
        self._project_root = Path(self.index_dir).parent
        self._lexicon = load_lexicon(self._project_root)
        self._load_image_meta()  # #12 多模态：加载图片父文档回溯元数据
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        # Phase 06：固定策略加载——eval candidate manifest 绑定 candidate 策略，
        # 生产 manifest 绑定批准策略；exact/auto 运行时路由已移除。
        self._vector_index_mode = (
            manifest.get("ann_policy", {}).get(
                "selected_index_type", "ivf-hnsw-sq"
            )
        )
        candidate_policy = manifest.get("candidate_query_policy")
        self._candidate_query_policy = (
            CandidateQueryPolicy(**candidate_policy)
            if isinstance(candidate_policy, dict)
            else None
        )
        self._page_by_id = {}
        self.pages = []
        for page in manifest.get("pages", []):
            wiki_page = WikiPage(
                path=Path(page["path"]), title=page.get("title", Path(page["path"]).stem),
                page_type=page.get("page_type", "concept"), content="",
                sources=page.get("sources", []), links=page.get("links", []),
                sha256=page.get("sha256", ""), aliases=page.get("aliases", []),
            )
            self.pages.append(wiki_page)
            self._page_by_id[page.get("page_id", page_id_of(wiki_page.path))] = wiki_page
        # Phase 06：repository 绑定策略——eval candidate manifest 绑定 candidate，
        # 其余绑定批准生产策略；ef 由仓库内部决定，facade 不传。
        repository_kwargs = (
            {"eval_candidate_policy": self._candidate_query_policy}
            if self._candidate_query_policy is not None else {}
        )
        self._repository = LanceDbIndexRepository(
            self._resolve_active_lance_dir(), **repository_kwargs
        )

    # ---- search ----
    def search_fts(self, query: str, k: int = 20) -> List[ChunkHit]:
        """LanceDB 原生 FTS（whitespace 预分词）。返回 chunk 级命中。"""
        q = " ".join(fts_terms(query, self._lexicon) + extract_exact_terms(query))
        rows = self._get_repository().search_sparse(q, limit=k * 4)
        return [self._hit_from_row(r, "fts") for r in rows]

    def search_fts_terms(self, lexical_terms, exact_terms, k: int = 20) -> List[ChunkHit]:
        """#6：直接消费 Query Planner 的通道专用 FTS 词项（不再二次分词）。

        ``lexical_terms``（来自 #2 ``fts_terms``）+ ``exact_terms``（型号/错误码/路径/
        数字/单位/CLI 参数）以空格拼接后交给 LanceDB whitespace FTS，与索引端
        ``fts_text`` 的构造完全一致（共享同一 tokenizer schema 与词典）。
        """
        q = " ".join(list(lexical_terms) + list(exact_terms))
        if not q.strip():
            return []
        rows = self._get_repository().search_sparse(q, limit=k * 4)
        return [self._hit_from_row(r, "fts") for r in rows]

    def search_vector(self, query: str, k: int = 20) -> List[ChunkHit]:
        """向量检索（仅 dense 行）。返回 chunk 级命中（按 page_id 归并前）。

        Phase 06：普通 dense 检索固定走批准 ANN + 批准 ef（仓库内部绑定）；
        exact 只保留为诊断 API（search_dense_exact），facade 不可达。
        """
        embedder = self._get_embedder()
        qv = embedder.encode([query], show_progress_bar=False,
                             normalize_embeddings=NORMALIZE_EMBEDDINGS)[0]
        repository = self._get_repository()
        rows = repository.search_dense(list(qv), metric=VECTOR_METRIC, limit=k * 4)
        return [self._hit_from_row(r, "vector") for r in rows]

    def search_page(self, page_id: str, plan, sparse_k: int = 20,
                    dense_k: int = 20) -> List[ChunkHit]:
        """Retrieve evidence under one page predicate on *both* retrieval paths.

        This is intentionally the only graph-validation adapter: callers cannot
        accidentally validate a graph recommendation with a similarly named
        chunk from another page.
        """
        repository = self._get_repository()
        out: List[ChunkHit] = []
        terms = " ".join(list(plan.lexical_terms) + list(plan.exact_terms))
        if terms.strip():
            rows = repository.search_sparse_for_page(terms, page_id, limit=sparse_k)
            out.extend(self._hit_from_row(row, "fts") for row in rows)
        try:
            embedder = self._get_embedder()
            for query in plan.semantic_queries:
                vector = embedder.encode([query], show_progress_bar=False,
                                          normalize_embeddings=NORMALIZE_EMBEDDINGS)[0]
                rows = repository.search_dense(
                    list(vector),
                    metric=VECTOR_METRIC,
                    limit=dense_k,
                    where=repository.page_predicate(page_id),
                )
                out.extend(self._hit_from_row(row, "vector") for row in rows)
        except Exception as exc:
            logging.getLogger(__name__).warning("restricted vector search failed: %s", exc)
        best: Dict[tuple, ChunkHit] = {}
        for hit in out:
            key = (hit.chunk_id, hit.channel)
            if hit.page_id == page_id and (key not in best or hit.score > best[key].score):
                best[key] = hit
        return list(best.values())

    # ---- Context repository port (issue #14) ----
    def _rows_where(self, predicate: str) -> List[dict]:
        """Small read-only adapter for context assembly; never exposes LanceDB to fusion."""
        try:
            return list(self._get_repository().context_rows(predicate))
        except Exception as exc:
            logging.getLogger(__name__).warning("context repository read failed: %s", exc)
            return []

    @staticmethod
    def _sql(value: str) -> str:
        return str(value).replace("'", "''")

    def get_chunk(self, chunk_id: str) -> Optional[ChunkHit]:
        rows = self._rows_where(f"chunk_id = '{self._sql(chunk_id)}'")
        return self._hit_from_row(rows[0], "fts") if rows else None

    def get_neighbors(self, chunk_id: str) -> List[ChunkHit]:
        anchor = self.get_chunk(chunk_id)
        if anchor is None:
            return []
        rows = self._rows_where(f"page_id = '{self._sql(anchor.page_id)}' AND chunk_kind = 'dense'")
        hits = [self._hit_from_row(row, "fts") for row in rows]
        hits.sort(key=lambda hit: (
            hit.chunk_index is None,
            hit.chunk_index if hit.chunk_index is not None else 0,
            hit.chunk_id,
        ))
        for pos, hit in enumerate(hits):
            if hit.chunk_id == chunk_id:
                return hits[max(0, pos - 1):pos] + hits[pos + 1:pos + 2]
        return []

    def get_parent_section(self, chunk_id: str) -> List[ChunkHit]:
        anchor = self.get_chunk(chunk_id)
        if anchor is None:
            return []
        rows = self._rows_where(f"page_id = '{self._sql(anchor.page_id)}'")
        hits = [self._hit_from_row(row, "fts") for row in rows]
        section_hits = [hit for hit in hits if hit.section_path == anchor.section_path] or [anchor]
        section_hits.sort(key=lambda hit: (
            hit.chunk_index is None,
            hit.chunk_index if hit.chunk_index is not None else 0,
            hit.chunk_id,
        ))
        return section_hits

    def get_page_sources(self, page_id: str) -> List[str]:
        page = self._page_by_id.get(page_id)
        return list(page.sources) if page else []

    def read_page(self, page_id: str) -> str:
        page = self._page_by_id.get(page_id)
        return _read_page_content(page.path) if page else ""

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
            title=r["title"], page_type=r.get("page_type", "concept"),
            section_path=json.loads(r.get("section_path") or "[]"),
            heading=r.get("heading", ""), chunk_kind=r.get("chunk_kind", "dense"),
            text=r["text"], channel=channel, score=score, distance=distance,
            chunk_index=(int(r["chunk_index"]) if r.get("chunk_index") is not None else None),
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
        description="构建 分层分块 + LanceDB FTS + 固定向量索引（批准策略 IVF_HNSW_SQ/ef=100）",
    )
    p.add_argument("project_root", help="知识库项目根目录（含 Wiki/）")
    p.add_argument("--full-rebuild", action="store_true",
                   help="忽略页级向量缓存，强制全量重编码（模型/分块配置变更或缓存疑似损坏时使用）")
    p.add_argument("--incremental", action="store_true",
                   help="（已废弃）仅复用 embedding 缓存；仍会发布全新的 sparse/dense 表和索引快照。")
    p.add_argument("--build-mode", choices=["snapshot", "incremental", "auto"], default="snapshot",
                   help="存储构建模式：snapshot（默认）、真实 staged incremental，或证据驱动 auto。")
    p.add_argument("--build-mode-policy", default=None,
                   help="auto 模式的项目内 JSON policy 相对路径；无效策略安全回退到 snapshot。")
    p.add_argument("--vector-index", default=None,
                   choices=["auto", "exact", "ivf-hnsw-flat", "ivf-hnsw-sq"],
                   help="（已废弃）Phase 06 起生产向量索引固定为批准策略 IVF_HNSW_SQ/ef=100；"
                        "传入任何值都会在构建前被拒绝。旧索引需全量重建。")
    p.add_argument("--allow-partial-index", action="store_true",
                   help="实验用：容忍缺页/0-chunk（降级为 warning）。默认关闭 fail-fast，禁止用于生产发布")
    p.add_argument("--retry-pending", action="store_true",
                   help="重放 post-commit 任务（#37）：只处理已发布 generation 的 PREPARED intent")
    args = p.parse_args()
    proj = Path(args.project_root)
    wiki = proj / "Wiki"
    idx_dir = proj / ".index"

    if args.retry_pending:
        if args.incremental or args.full_rebuild or args.build_mode != "snapshot" or args.build_mode_policy is not None:
            p.error("--retry-pending cannot be combined with build-mode or cache flags")
        from obsidian_wiki.application.post_commit_service import retry_pending
        from obsidian_wiki.infrastructure.filesystem_community_reports import FilesystemCommunityReportStore
        from obsidian_wiki.infrastructure.filesystem_post_commit_journal import FilesystemPostCommitJournal

        summary = retry_pending(
            idx_dir,
            journal=FilesystemPostCommitJournal(idx_dir),
            invalidator=FilesystemCommunityReportStore(idx_dir),
        )
        print(
            f"post-commit retry: completed={summary.completed}, "
            f"still_pending={summary.still_pending}"
        )
        return

    if args.vector_index is not None:
        # Phase 06（issue #49）：一次性兼容 shim——任何显式 mode 在 embedding/
        # storage mutation 之前拒绝；运行时类型选择已从生产路径移除。
        print(
            f"--vector-index={args.vector_index} 已废弃：生产向量索引固定为批准策略 "
            "(IVF_HNSW_SQ, ef=100)。请去掉该参数重新构建；旧 mode-ambiguous 索引需全量重建。",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.incremental and args.build_mode != "snapshot":
        p.error("--incremental is legacy embedding-cache reuse and conflicts with --build-mode")

    from obsidian_wiki.application.incremental_policy import load_build_mode_policy
    # Parse policy before model construction or any storage mutation.  Explicit
    # snapshot intentionally bypasses policy validation entirely.
    policy_load = (
        load_build_mode_policy(proj, Path(args.build_mode_policy))
        if args.build_mode == "auto" else None
    )

    from obsidian_wiki.application.build_lock import BuildLockHeldError
    from obsidian_wiki.domain.index_models import PostCommitStatus
    try:
        outcome = WikiIndex(idx_dir).build(
            wiki, full_rebuild=args.full_rebuild,
            build_mode=args.build_mode,
            build_mode_policy_path=(Path(args.build_mode_policy) if args.build_mode == "auto" else None),
            build_mode_policy=policy_load,
        )
    except BuildLockHeldError as exc:
        print(f"索引构建被拒：{exc}", file=sys.stderr)
        sys.exit(1)
    artifact = outcome.artifact
    telemetry = json.loads(artifact.manifest_path.read_text(encoding="utf-8"))["build_telemetry"]
    print(
        "索引构建完成 "
        f"requested={telemetry['mode_requested']} selected={telemetry['mode_selected']} "
        f"reason={telemetry['selection_reason']} policy_digest="
        f"{telemetry.get('build_mode_policy_sha256')} cache={'bypass' if args.full_rebuild else 'reuse'}: "
        f"sparse={artifact.sparse_count}, dense={artifact.dense_count} → {artifact.lance_dir}"
    )
    # #37：pending outcome 的 CLI exit code 必须为 0；stderr 输出明确 warning 与 retry action。
    if outcome.post_commit_status != PostCommitStatus.COMPLETE:
        print(
            f"警告：{outcome.warnings[0] if outcome.warnings else 'post-commit 任务待重试'}",
            file=sys.stderr,
        )
        print(
            "修复：运行 `python build_index.py <project_root> --retry-pending` 重试。",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
