"""Small orchestration layer for the first D-01/D-04 persisted tracer."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import os
import statistics
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Sequence

import chunking  # 叶子模块（仅 stdlib），提供 SPARSE/DENSE_CHUNK_SCHEMA_VERSION

from obsidian_wiki.application.active_index_pointer import (
    publish_pointer,
    record_building,
    record_validated,
)
from obsidian_wiki.application.build_lock import BuildLock, new_build_context
from obsidian_wiki.application.durable_filesystem import CommitUncertainError
from obsidian_wiki.domain.index_models import (
    BenchmarkObservation,
    BuildContext,
    CANDIDATE_PUBLICATION_EVIDENCE_SCHEMA_VERSION,
    CandidatePublicationEvidence,
    CandidateQueryPolicy,
    DenseChunk,
    FtsIndexConfig,
    INDEX_LAYOUT_VERSION,
    INDEX_MANIFEST_FORMAT_VERSION,
    IndexStats,
    PostCommitTask,
    PostCommitTaskState,
    ProductionAnnPolicy,
    SparseChunk,
    StorageArtifact,
    VectorIndexConfig,
    VectorPolicyDecision,
)
from obsidian_wiki.domain.incremental_models import BuildTelemetry, BuildTiming, TableRowCounts
from obsidian_wiki.domain.index_policy import (
    PolicyError,
    load_ann_policy_file,
    production_policy_sha256,
    validate_candidate_publication_evidence,
)
from obsidian_wiki.ports.chunk_repository import ChunkRepository
from obsidian_wiki.ports.index_manifest import IndexManifestStore
from obsidian_wiki.ports.post_commit import PostCommitJournal
from obsidian_wiki.application.incremental_policy import compatibility_digest_from_manifest
from obsidian_wiki.application.incremental_policy import select_auto_build_mode
from obsidian_wiki.application.index_publication_service import IndexPublicationService
from obsidian_wiki.domain.incremental_models import BuildModePolicyLoad, BuildModeSelection
from obsidian_wiki.ports.incremental_index import (
    IncrementalExecutorFactory,
    IncrementalFallbackEligible,
)


Embedder = Callable[[Sequence[str]], Sequence[Sequence[float]]]
BenchmarkObserver = Callable[[IndexStats], BenchmarkObservation]
# #39 (review)：持锁后运行的分块回调，返回 (sparse_chunks, page_metadata)。
PlanProvider = Callable[
    [Path], tuple[Sequence["SparseChunk"], "list[dict] | None"]
]
# #41→Phase 06：构建期 held-out 验证的最大 query 数。只决定
# ``min(BENCHMARK_MAX_PROBES, actual_dense_rows)`` 的验证采样规模，
# 不能影响索引类型 / 查询 ef / 任何生产策略值。
BENCHMARK_MAX_PROBES = 256
# Phase 06：held-out 验证 query 的确定性流种子（跨平台 Mersenne Twister 可复现）。
_VALIDATION_QUERY_SEED = 20260817
_VALIDATION_QUERY_SOURCE = "deterministic_disjoint_unit_v1"

# 兼容 re-export：CandidateQueryPolicy 已迁至 domain（infrastructure 的 eval
# 绑定需要它）；旧导入路径继续可用。
__all__ = ["IndexBuildService", "CandidateQueryPolicy", "BENCHMARK_MAX_PROBES"]


class IndexBuildService:
    """Partition canonical Markdown into physically separate sparse/dense rows."""

    def __init__(
        self,
        storage: ChunkRepository,
        *,
        reopen_storage: Callable[[Path], ChunkRepository],
        manifest_store: IndexManifestStore,
        post_commit_journal: PostCommitJournal,
        fts_config: FtsIndexConfig | None = None,
        benchmark_observer: BenchmarkObserver | None = None,
        benchmark_max_probes: int = BENCHMARK_MAX_PROBES,
        candidate_query_policy: CandidateQueryPolicy | None = None,
        ann_policy: ProductionAnnPolicy | None = None,
        publication_service: IndexPublicationService | None = None,
        incremental_executor_factory: IncrementalExecutorFactory | None = None,
    ):
        """#37：``post_commit_journal`` 为必填依赖——每个发布路径都必须在 pointer
        commit 前 durable prepare 失效 intent；不需要 invalidation 的调用方必须显式
        注入 deliberate no-op port，遗漏绝不能静默禁用契约。

        #41→Phase 06：``benchmark_max_probes`` 在 storage mutation 之前验证（构造时
        即拒绝非法值）；bool 是 int 子类，须显式拒绝。它只决定 held-out 验证的
        query 数，不影响任何生产策略值。

        Phase 06（issue #49）：``vector_index_mode`` 已移除——生产构建固定使用
        ``ann_policy``（默认加载 ``eval/ann-policy.json`` 的批准策略）。
        ``candidate_query_policy`` 仅用于 eval comparator 构建，不经过发布门禁。
        """
        if (
            isinstance(benchmark_max_probes, bool)
            or not isinstance(benchmark_max_probes, int)
            or benchmark_max_probes <= 0
        ):
            raise ValueError(
                f"benchmark_max_probes must be a positive integer, got {benchmark_max_probes!r}"
            )
        self._storage = storage
        self._reopen_storage = reopen_storage
        self._manifest_store = manifest_store
        self._fts_config = fts_config or FtsIndexConfig()
        self._benchmark_observer = benchmark_observer
        self._post_commit_journal = post_commit_journal
        self._benchmark_max_probes = benchmark_max_probes
        self._candidate_query_policy = candidate_query_policy
        self._ann_policy = ann_policy if ann_policy is not None else load_ann_policy_file()
        self._publication_service = publication_service or IndexPublicationService()
        self._publication_service.bind(
            next_generation=self._next_generation,
            exact_term=self._exact_term,
            disk_bytes=self._disk_bytes,
            candidate_validation=self._publication_validation,
            manifest=self._manifest,
            ann_policy=self._ann_policy,
        )
        self._incremental_executor_factory = incremental_executor_factory

    def build(
        self, wiki_dir: Path, index_dir: Path, *, embed: Embedder,
        sparse_chunks: Sequence[SparseChunk] | None = None,
        page_metadata: list[dict] | None = None,
        image_metadata: list[dict] | None = None,
        ctx: BuildContext | None = None,
        plan_provider: PlanProvider | None = None,
        build_mode: str = "snapshot",
        build_mode_policy: BuildModePolicyLoad | None = None,
        outer_lock_held: bool = False,
    ) -> StorageArtifact:
        """#21/#34 单写者构建：最外层传入或创建一次 BuildContext，锁 metadata、
        build 目录、manifest、pointer 与返回 artifact 共用同一个 build_id；
        service 不再独立生成 ID。

        #39 (review)：``plan_provider`` 是持锁后运行的分块回调
        ``(wiki_dir) -> (sparse_chunks, page_metadata)``。分块必须在 BUILD.lock
        获取之后、针对已加锁的 Wiki 快照执行；调用方传入的 ``sparse_chunks``
        （显式计划）仍然优先。"""
        if build_mode not in {"snapshot", "incremental", "auto"}:
            raise ValueError("build_mode must be snapshot, incremental, or auto")
        ctx = ctx or new_build_context()
        lock = None
        if not outer_lock_held:
            lock = BuildLock(index_dir, ctx=ctx)
            lock.acquire()
        try:
            if sparse_chunks is None and plan_provider is not None:
                sparse_chunks, planned_pages = plan_provider(wiki_dir)
                if page_metadata is None and planned_pages is not None:
                    page_metadata = planned_pages
            sparse_chunks = tuple(sparse_chunks) if sparse_chunks is not None else self._sparse_plan(wiki_dir)
            selection = self._select_build_mode(
                index_dir, build_mode=build_mode, policy_load=build_mode_policy,
            )
            if selection.selected_mode == "incremental":
                if self._incremental_executor_factory is None:
                    if build_mode == "incremental":
                        raise RuntimeError("incremental_executor_unavailable")
                    selection = BuildModeSelection(
                        "snapshot", "incremental_executor_unavailable",
                        selection.policy_sha256, selection.compatibility_digest,
                        selection.evidence_observation_ids, selection.calculated_values,
                    )
                else:
                    executor = self._incremental_executor_factory()
                    try:
                        result = executor.build_staged(
                            wiki_dir, index_dir, canonical_chunks=sparse_chunks, embed=embed,
                            page_metadata=page_metadata, ctx=ctx,
                            mode_requested=build_mode, selection_reason=selection.reason,
                            build_mode_policy_sha256=selection.policy_sha256,
                            outer_lock_held=True,
                        )
                        return result.artifact
                    except IncrementalFallbackEligible as exc:
                        if build_mode != "auto" or exc.reason not in {
                            "incompatible_active_contract", "shallow_clone_unavailable",
                            "index_catch_up_unproven",
                        }:
                            raise
                        selection_reason = (
                            exc.selection_reason
                            if exc.contract_drift is not None
                            else f"incremental_runtime_fallback:{exc.reason}"
                        )
                        selection = BuildModeSelection(
                            "snapshot", selection_reason,
                            selection.policy_sha256, selection.compatibility_digest,
                            selection.evidence_observation_ids, selection.calculated_values,
                        )
                        replacement = new_build_context()
                        ctx = BuildContext(replacement.build_id, ctx.started_at, ctx.owner_nonce)
            return self._build(
                wiki_dir, index_dir, embed=embed,
                sparse_chunks=sparse_chunks, ctx=ctx,
                page_metadata=page_metadata, image_metadata=image_metadata,
                plan_provider=None, mode_requested=build_mode,
                selection_reason=selection.reason,
                build_mode_policy_sha256=selection.policy_sha256,
            )
        finally:
            if lock is not None:
                lock.release()

    def _build(
        self, wiki_dir: Path, index_dir: Path, *, embed: Embedder,
        sparse_chunks: Sequence[SparseChunk] | None = None, ctx: BuildContext | None = None,
        page_metadata: list[dict] | None = None,
        image_metadata: list[dict] | None = None,
        plan_provider: PlanProvider | None = None,
        mode_requested: str = "snapshot",
        selection_reason: str = "explicit_snapshot",
        build_mode_policy_sha256: str | None = None,
    ) -> StorageArtifact:
        build_started = time.perf_counter()
        # #39 (review)：持锁后再分块。显式 sparse_chunks > plan_provider > 回退整页 plan。
        if sparse_chunks is None and plan_provider is not None:
            planned_chunks, planned_pages = plan_provider(wiki_dir)
            sparse_chunks = planned_chunks
            if page_metadata is None and planned_pages is not None:
                page_metadata = planned_pages
        sparse_chunks = tuple(sparse_chunks) if sparse_chunks is not None else self._sparse_plan(wiki_dir)
        if not sparse_chunks:
            raise RuntimeError("No canonical Wiki Markdown pages were available to index")
        # issue #47：canonical 计划同时含 sparse+dense 两种 kind；严格分离后再落两表。
        lexical_chunks = tuple(chunk for chunk in sparse_chunks if chunk.chunk_kind == "sparse")
        dense_sources = tuple(chunk for chunk in sparse_chunks if chunk.chunk_kind == "dense")
        if not lexical_chunks:
            raise RuntimeError("Canonical chunk plan contains no sparse (lexical) retrieval chunks")
        if not dense_sources:
            raise RuntimeError("Canonical chunk plan contains no dense retrieval chunks")
        embed_started = time.perf_counter()
        vectors = embed([chunk.text for chunk in dense_sources])
        embedding_cache_miss_ms = (time.perf_counter() - embed_started) * 1000
        if len(vectors) != len(dense_sources):
            raise RuntimeError("Embedder returned a vector count different from the dense chunk plan")
        dense_chunks = tuple(
            DenseChunk(
                chunk_id=chunk.chunk_id,
                page_id=chunk.page_id,
                path=chunk.path,
                title=chunk.title,
                text=chunk.text,
                vector=tuple(float(value) for value in vector),
                page_type=chunk.page_type,
                section_path=chunk.section_path,
                heading=chunk.heading,
                chunk_kind=chunk.chunk_kind,
                chunk_index=chunk.chunk_index,
                parent_section_id=chunk.parent_section_id,
                token_count=chunk.token_count,
                content_hash=chunk.content_hash,
                forced_split=chunk.forced_split,
                continuation_index=chunk.continuation_index,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
            )
            for chunk, vector in zip(dense_sources, vectors)
        )
        if not all(chunk.vector for chunk in dense_chunks):
            raise RuntimeError("Dense chunks require non-empty vectors")

        generation = self._publication_service.allocate_generation(index_dir)
        build_dir = index_dir / "builds" / ctx.build_id
        lance_dir = build_dir / "lance_db"
        build_dir.mkdir(parents=True, exist_ok=False)
        # #35：build 目录创建后立即耐久写 BUILDING（missing → building）。
        record_building(build_dir, build_id=ctx.build_id, generation=generation)
        try:
            write_started = time.perf_counter()
            self._storage.persist(lance_dir, lexical_chunks, dense_chunks, self._fts_config)
            serialization_write_ms = (time.perf_counter() - write_started) * 1000
            # #36 follow-up：所有 storage mutation（persist / create_vector_index）都
            # 必须在最终 seal 之前完成——vector index 创建会改写 LanceDB，故 seal
            # 不能放在 persist 后（当前已修复：先建索引，再最终 seal）。
            reopened = self._reopen_storage(lance_dir)
            dimension = len(dense_chunks[0].vector)
            # Phase 06（issue #49）：唯一的 candidate 选择——批准策略。
            # 无 corpus 大小 / probe / 运行时输入分支；eval candidate 构建由
            # 调用方显式传入 eval-bound repository。
            if self._candidate_query_policy is not None:
                eval_types = {
                    "ivf-hnsw-flat": "hnsw_flat",
                    "ivf-hnsw-sq": "hnsw_sq",
                }
                candidate_index_type = eval_types[self._candidate_query_policy.candidate]
            else:
                if dimension != self._ann_policy.dimensions:
                    raise RuntimeError(
                        f"embedding dimension {dimension} does not match the approved "
                        f"ann policy dimension {self._ann_policy.dimensions}"
                    )
                candidate_index_type = self._ann_policy.lancedb_index_type
            vector_config = VectorIndexConfig(
                index_type=candidate_index_type,
                metric="cosine", num_partitions=1,
                m=16, ef_construction=300,
                dense_chunks_count=len(dense_chunks),
            )
            index_started = time.perf_counter()
            reopened.create_vector_index(vector_config)
            index_build_ms = (time.perf_counter() - index_started) * 1000
            # issue #47: the exact-term validation probes the FTS (sparse) table,
            # so the sampled term must come from the lexical corpus that is
            # actually indexed there — not from a dense chunk that never enters FTS.
            exact_term = self._publication_service.canonical_exact_term(lexical_chunks)
            validation_started = time.perf_counter()
            counts, vector_stats, fts_stats = reopened.validate_reopened(
                dimension=dimension, exact_term=exact_term,
                vector_index_name=vector_config.index_name,
            )
            if (
                counts.sparse_chunks_count != len(lexical_chunks)
                or counts.dense_chunks_count != len(dense_chunks)
            ):
                raise RuntimeError(
                    "staging 持久化完整性校验失败："
                    f"sparse={counts.sparse_chunks_count}/{len(lexical_chunks)} "
                    f"dense={counts.dense_chunks_count}/{len(dense_chunks)}"
                )
            if self._candidate_query_policy is not None:
                # Eval candidate 构建：不经过生产发布门禁（FLAT/低 ef 达不到
                # 生产 floors 是预期），benchmark 记录 synthetic observer 证据。
                benchmark, benchmark_evidence = self._observer_benchmark(
                    dense_chunks
                )
                policy_decision = VectorPolicyDecision(
                    selected_mode="ann",
                    reason="evaluation candidate build; production publication gate not applicable",
                    benchmark=benchmark,
                    index_stats=vector_stats,
                    benchmark_scope=benchmark_evidence["probe_scope"],
                    benchmark_probe_count=benchmark_evidence["probe_count"],
                    benchmark_probe_total=benchmark_evidence["probe_total"],
                )
                publication_evidence = None
            else:
                # Phase 06：发布门禁——真实 staged candidate 的 held-out
                # CandidatePublicationEvidence 验证（fail-closed，任何失败都不
                # publish、不改写旧 ACTIVE_INDEX）。使用重开连接（HNSW 可见性）。
                publication_evidence, benchmark = self._publication_service.validate_candidate(
                    self._reopen_storage(lance_dir),
                    dense_chunks,
                    vector_stats=vector_stats,
                    actual_dense_rows=counts.dense_chunks_count,
                    unindexed_dense_rows=vector_stats.unindexed_dense_rows,
                    build_time_ms=index_build_ms,
                    disk_bytes=self._publication_service.staged_disk_bytes(build_dir),
                )
                policy_decision = VectorPolicyDecision(
                    selected_mode="ann",
                    reason=(
                        "approved fixed ann policy validated by held-out "
                        "publication evidence"
                    ),
                    benchmark=benchmark,
                    index_stats=vector_stats,
                    benchmark_scope="held_out",
                    benchmark_probe_count=publication_evidence.validation_query_count,
                    benchmark_probe_total=counts.dense_chunks_count,
                )
            validation_ms = (time.perf_counter() - validation_started) * 1000
            # #36 follow-up：最终 seal = 最后 storage mutation 之后的耐久边界。
            self._storage.seal(lance_dir)
            manifest_kwargs = dict(
                counts=counts.to_json(), vector_stats=vector_stats.to_json(),
                fts_stats=fts_stats.to_json(), vector_config=vector_config,
                benchmark=benchmark.to_json(),
                policy=policy_decision.to_json(),
                sparse_chunks=sparse_chunks, generation=generation, build_id=ctx.build_id,
                page_metadata=page_metadata, image_metadata=image_metadata,
                candidate_query_policy=self._candidate_query_policy,
                ann_policy=self._ann_policy,
                publication_evidence=publication_evidence,
            )
            manifest = self._publication_service.construct_manifest(**manifest_kwargs)
            telemetry = BuildTelemetry(
                schema_version=1, observation_id=ctx.build_id,
                mode_requested=mode_requested, mode_selected="snapshot",
                selection_reason=selection_reason,
                compatibility_digest=compatibility_digest_from_manifest(manifest),
                completed_at_epoch_seconds=time.time(),
                timings=BuildTiming(
                    scan_parse_ms=(time.perf_counter() - build_started) * 1000,
                    chunking_ms=0.0,
                    embedding_cache_hit_ms=0.0,
                    embedding_cache_miss_ms=embedding_cache_miss_ms,
                    serialization_write_ms=serialization_write_ms,
                    fts_catch_up_ms=0.0,
                    vector_catch_up_ms=index_build_ms,
                    validation_ms=validation_ms,
                    publication_ms=0.0,
                    index_rebuild_ms=index_build_ms,
                ),
                sparse_rows=TableRowCounts(
                    inserted=len(lexical_chunks), updated=0, deleted=0, unchanged=0,
                    physically_written=len(lexical_chunks),
                ),
                dense_rows=TableRowCounts(
                    inserted=len(dense_chunks), updated=0, deleted=0, unchanged=0,
                    physically_written=len(dense_chunks),
                ),
                embedding_cache_hits=0, embedding_cache_misses=len(dense_chunks),
                peak_staged_disk_bytes=self._publication_service.staged_disk_bytes(build_dir), completed=True,
                build_mode_policy_sha256=build_mode_policy_sha256,
            )
            manifest = self._publication_service.construct_manifest(**manifest_kwargs, build_telemetry=telemetry)
            # #34：发布前身份断言——build 目录、manifest、ctx 必须同一 build_id。
            if build_dir.name != ctx.build_id or manifest.get("build_id") != ctx.build_id:
                raise RuntimeError(
                    f"build 身份不一致：dir={build_dir.name} manifest="
                    f"{manifest.get('build_id')!r} ctx={ctx.build_id}"
                )
            manifest_path = build_dir / "manifest.json"
            self._manifest_store.write(manifest_path, manifest)
            # #35：manifest 完整落盘后写入 validated 生命周期记录——manifest 写后、
            # pointer 发布前中断的 staging generation 绝不被 recovery 选中。
            record_validated(
                build_dir, generation=generation, build_id=ctx.build_id,
                manifest_sha256=self._sha256_file(manifest_path),
            )
            # #37：pointer commit 前 durable prepare post-commit intent——进程在
            # prepare 与 invalidate 之间退出也不会永久丢失任务。journal 为必填依赖。
            self._post_commit_journal.prepare(PostCommitTask(
                task_id=uuid.uuid4().hex,
                task_type="community_report_invalidation",
                build_id=ctx.build_id,
                generation=generation,
                state=PostCommitTaskState.PREPARED,
                prepared_at=datetime.now(timezone.utc).isoformat(),
            ))
            publish_pointer(index_dir, build_dir, generation=generation, build_id=ctx.build_id)
            return StorageArtifact(
                lance_dir, manifest_path, len(lexical_chunks), len(dense_chunks),
                build_id=ctx.build_id, generation=generation,
            )
        except CommitUncertainError:
            # #37 follow-up：pointer 可能已替换（commit point 已过）——绝不能写
            # .failed 伪装成从未发布；向上传播让调用方按状态不确定处理。
            raise
        except Exception as exc:
            message = str(exc).lower()
            invariant = "manifest" if "manifest" in message else "validation"
            (build_dir / ".failed").write_text(
                f"{invariant} failed before publication: {type(exc).__name__}: {exc}",
                encoding="utf-8",
            )
            raise

    def _select_build_mode(
        self, index_dir: Path, *, build_mode: str,
        policy_load: BuildModePolicyLoad | None,
    ) -> BuildModeSelection:
        """Choose only from injected policy evidence while the one writer lock is held."""
        placeholder = "0" * 64
        if build_mode == "snapshot":
            return BuildModeSelection("snapshot", "explicit_snapshot", None, placeholder, ())
        if build_mode == "incremental":
            return BuildModeSelection("incremental", "explicit_incremental", None, placeholder, ())
        if policy_load is None:
            raise RuntimeError("auto build mode requires a parsed policy load")
        try:
            from obsidian_wiki.application.active_index_pointer import resolve_active_lance_dir
            active_manifest = json.loads(
                (resolve_active_lance_dir(index_dir).parent / "manifest.json").read_text(encoding="utf-8")
            )
            compatibility_digest = compatibility_digest_from_manifest(active_manifest)
            from obsidian_wiki.application.incremental_policy import policy_contract_drift
            drift = policy_contract_drift(policy_load, active_manifest)
            if drift is not None:
                return BuildModeSelection(
                    "snapshot", f"incompatible_active_contract:{drift.value}",
                    policy_load.policy_sha256, compatibility_digest, (),
                )
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError, TypeError):
            return BuildModeSelection("snapshot", "active_contract_unavailable", policy_load.policy_sha256, placeholder, ())
        observations: list[BuildTelemetry] = []
        for manifest_path in (index_dir / "builds").glob("*/manifest.json") if (index_dir / "builds").is_dir() else ():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                raw = manifest["build_telemetry"]
                observations.append(BuildTelemetry(
                    schema_version=raw["schema_version"], observation_id=raw["observation_id"],
                    mode_requested=raw["mode_requested"], mode_selected=raw["mode_selected"],
                    selection_reason=raw["selection_reason"], compatibility_digest=raw["compatibility_digest"],
                    completed_at_epoch_seconds=raw["completed_at_epoch_seconds"],
                    timings=BuildTiming(**raw["timings"]), sparse_rows=TableRowCounts(**raw["sparse_rows"]),
                    dense_rows=TableRowCounts(**raw["dense_rows"]), embedding_cache_hits=raw["embedding_cache_hits"],
                    embedding_cache_misses=raw["embedding_cache_misses"], peak_staged_disk_bytes=raw["peak_staged_disk_bytes"],
                    completed=raw["completed"], build_mode_policy_sha256=raw.get("build_mode_policy_sha256"),
                ))
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
                continue
        return select_auto_build_mode(
            policy_load, observations, current_compatibility_digest=compatibility_digest,
            now_epoch_seconds=time.time(),
        )

    def _manifest(self, *, counts: dict, vector_stats: dict, fts_stats: dict,
                  vector_config: VectorIndexConfig, benchmark: dict, policy: dict,
                  sparse_chunks: Sequence[SparseChunk], generation: int = 0,
                  build_id: str = "",
                  page_metadata: list[dict] | None = None,
                  image_metadata: list[dict] | None = None,
                  candidate_query_policy: CandidateQueryPolicy | None = None,
                  ann_policy: ProductionAnnPolicy | None = None,
                  publication_evidence: CandidatePublicationEvidence | None = None,
                  build_telemetry: BuildTelemetry | None = None) -> dict:
        fts_config = self._fts_config.to_json()
        vector_config_json = vector_config.to_json()
        # facade 提供的 page_metadata 含完整 page_type/sources/links/aliases/sha256；
        # 否则从 sparse_chunks 生成最小兼容元数据。
        if page_metadata is not None:
            pages = page_metadata
        else:
            pages = [
                {
                    "page_id": chunk.page_id,
                    "path": chunk.path,
                    "title": chunk.title,
                    "page_type": "concept",
                    "sources": [],
                    "links": [],
                    "aliases": [],
                    "sha256": hashlib.sha256(chunk.text.encode("utf-8")).hexdigest(),
                }
                for chunk in sparse_chunks
            ]
        manifest = {
            # Phase 06（issue #49）：v5 → v6——manifest 绑定固定批准策略与（生产
            # 构建的）held-out 发布证据；不再有 requested_vector_index_mode。
            # v5 / benchmark evidence-v2 是 mode-ambiguous，加载时须拒绝重建。
            "format_version": INDEX_MANIFEST_FORMAT_VERSION,
            "layout": "sparse_chunks+dense_chunks",
            "generation": generation,
            "build_id": build_id,
            # issue #47 F：布局/版本元信息，供 require_current_layout 拒绝旧构建。
            "index_layout_version": INDEX_LAYOUT_VERSION,
            "sparse_chunk_schema_version": chunking.SPARSE_CHUNK_SCHEMA_VERSION,
            "dense_chunk_schema_version": chunking.DENSE_CHUNK_SCHEMA_VERSION,
            "canonical_chunks_count": len(sparse_chunks),
            "fts_rows_count": sum(1 for c in sparse_chunks if c.chunk_kind == "sparse"),
            "fts_config": fts_config,
            "vector_config": vector_config_json,
            "ann_policy": {
                "selected_index_type": ann_policy.selected_index_type,
                "lancedb_index_type": ann_policy.lancedb_index_type,
                "query_ef": ann_policy.query_ef,
                "metric": ann_policy.metric,
                "dimensions": ann_policy.dimensions,
                "recall_at_10_floor": ann_policy.recall_at_10_floor,
                "recall_at_20_floor": ann_policy.recall_at_20_floor,
                "policy_sha256": production_policy_sha256(ann_policy),
                "comparator_sha256": ann_policy.comparator_sha256,
            } if ann_policy is not None else None,
            "config_hashes": {
                "fts_config": self._stable_hash(fts_config),
                "vector_config": self._stable_hash(vector_config_json),
            },
            "sdk_versions": {
                package: importlib.metadata.version(package)
                for package in ("lancedb", "pyarrow", "sentence-transformers")
            },
            "validation": {
                "schema_counts": counts, "vector_index": vector_stats,
                "fts_index": fts_stats, "exact_term_validated": True,
            },
            "benchmark": benchmark,
            "policy": policy,
            "pages": pages,
        }
        if image_metadata is not None:
            manifest["images"] = image_metadata
        if candidate_query_policy is not None:
            manifest["candidate_query_policy"] = candidate_query_policy.to_json()
        if publication_evidence is not None:
            manifest["candidate_publication_evidence"] = publication_evidence.to_json()
        if build_telemetry is not None:
            manifest["build_telemetry"] = build_telemetry.to_json()
        return manifest

    @staticmethod
    def _next_generation(index_dir: Path) -> int:
        """扫描 builds/ 分配单调递增 generation（review #2：持锁后分配）。"""
        builds_dir = index_dir / "builds"
        if not builds_dir.is_dir():
            return 1
        max_gen = 0
        for entry in builds_dir.iterdir():
            if not entry.is_dir() or entry.name.startswith(".") or entry.name.startswith("_old"):
                continue
            manifest = entry / "manifest.json"
            if not manifest.is_file():
                continue
            try:
                data = json.loads(manifest.read_text(encoding="utf-8"))
                gen = data.get("generation", 0) if isinstance(data, dict) else 0
                if isinstance(gen, (int, float)) and gen > max_gen:
                    max_gen = int(gen)
            except (OSError, json.JSONDecodeError, TypeError):
                continue
        return max_gen + 1

    def _observer_benchmark(
        self, dense_chunks: Sequence[DenseChunk]
    ) -> tuple[BenchmarkObservation, dict]:
        """Synthetic observer benchmark for eval candidate builds only.

        Eval candidate（FLAT / 低 ef）达不到生产 floors 是预期——它们不经过
        发布门禁；benchmark 记录 synthetic 证据以保持 manifest 结构一致。
        """
        observation = (
            self._benchmark_observer(IndexStats(index_name="dense_hnsw", indexed_rows=len(dense_chunks), unindexed_dense_rows=0))
            if self._benchmark_observer is not None
            else BenchmarkObservation(
                recall_at_10=0.0, recall_at_20=0.0,
                latency_p50_ms=0.0, latency_p95_ms=0.0,
                build_time_ms=0.0, disk_bytes=0,
            )
        )
        evidence = {
            "evidence_schema_version": 2,
            "evidence_source": "observer",
            "probe_scope": "synthetic",
            "sampling_method": "synthetic",
            "probe_count": 0,
            "probe_total": len(dense_chunks),
            "probe_coverage": 0.0,
            "result_limit": 20,
            "recall_aggregation": "none",
            "benchmark_duration_ms": 0.0,
            "exact_result_ids": [],
            "candidate_result_ids": [],
        }
        return observation, evidence

    def _validation_queries(self, count: int) -> tuple[tuple[float, ...], ...]:
        """独立确定性流的单位向量验证 query（与 corpus 行零重叠）。

        种子固定 → 跨平台/跨 runner 可复现（Mersenne Twister）；向量由
        gauss 流归一化生成，连续随机分布与任何 corpus 行都不相等（调用处
        再显式校验一次重叠 = 0）。
        """
        import random as _random

        rng = _random.Random(_VALIDATION_QUERY_SEED)
        queries: list[tuple[float, ...]] = []
        for _ in range(count):
            while True:
                raw = [rng.gauss(0.0, 1.0) for _ in range(self._ann_policy.dimensions)]
                norm = math.sqrt(sum(value * value for value in raw))
                if norm > 1e-6:
                    queries.append(tuple(value / norm for value in raw))
                    break
        return tuple(queries)

    def _publication_validation(
        self,
        repository: ChunkRepository,
        dense_chunks: Sequence[DenseChunk],
        *,
        vector_stats: IndexStats,
        actual_dense_rows: int,
        unindexed_dense_rows: int,
        build_time_ms: float,
        disk_bytes: int,
    ) -> tuple[CandidatePublicationEvidence, BenchmarkObservation]:
        """Phase 06 发布门禁：held-out 验证真实 staged candidate（fail-closed）。

        - query 数 = ``min(benchmark_max_probes, actual_dense_rows)``，来自独立
          确定性流且与 corpus 零重叠；
        - exact truth 走 #41 streamed batch-exact；candidate 走普通端口（固定
          批准 ef）；
        - 聚合 recall@10/20 与 floors 比较——任何失败抛 PolicyError，构建标记
          failed，旧 ACTIVE_INDEX 保持不变。
        """
        policy = self._ann_policy
        benchmark_started = time.perf_counter()
        count = min(self._benchmark_max_probes, actual_dense_rows)
        queries = self._validation_queries(count)
        corpus_vectors = {
            tuple(float(value) for value in chunk.vector) for chunk in dense_chunks
        }
        corpus_query_overlap = sum(1 for query in queries if query in corpus_vectors)
        if corpus_query_overlap:
            raise PolicyError(
                f"{corpus_query_overlap} validation queries overlap indexed corpus rows"
            )
        query_selection_sha256 = hashlib.sha256(
            "\n".join(
                ",".join(format(value, ".17g") for value in query) for query in queries
            ).encode("utf-8")
        ).hexdigest()

        exact_batch = repository.search_dense_exact_batch(
            list(queries), metric=policy.metric, limit=20
        )
        exact_ids = [list(row_ids) for row_ids in exact_batch.result_ids]
        if len(exact_ids) != count:
            raise RuntimeError("Batch exact result count does not match validation query count")

        ann_started = time.perf_counter()
        candidate_ids: list[list[str]] = []
        candidate_durations: list[float] = []
        for query in queries:
            started = time.perf_counter()
            rows = repository.search_dense(list(query), metric=policy.metric, limit=20)
            candidate_durations.append((time.perf_counter() - started) * 1000)
            candidate_ids.append([str(row["chunk_id"]) for row in rows])
        ann_verification_ms = (time.perf_counter() - ann_started) * 1000

        recalls: dict[int, float] = {}
        for recall_limit in (10, 20):
            hits = total = 0
            for truth, observed in zip(exact_ids, candidate_ids, strict=True):
                truth_prefix = set(truth[:recall_limit])
                hits += len(truth_prefix & set(observed[:recall_limit]))
                total += len(truth_prefix)
            if total <= 0:
                raise PolicyError("publication validation produced no truth IDs to score")
            recalls[recall_limit] = hits / total

        def percentile_95(samples: Sequence[float]) -> float:
            ordered = sorted(samples)
            return ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]

        observation = BenchmarkObservation(
            recall_at_10=recalls[10],
            recall_at_20=recalls[20],
            latency_p50_ms=statistics.median(candidate_durations),
            latency_p95_ms=percentile_95(candidate_durations),
            build_time_ms=build_time_ms,
            disk_bytes=disk_bytes,
        )
        evidence = CandidatePublicationEvidence(
            evidence_schema_version=CANDIDATE_PUBLICATION_EVIDENCE_SCHEMA_VERSION,
            actual_dense_rows=actual_dense_rows,
            dimensions=policy.dimensions,
            metric=policy.metric,
            index_type=policy.selected_index_type,
            query_ef=policy.query_ef,
            policy_sha256=production_policy_sha256(policy),
            decision_evidence_sha256=policy.comparator_sha256,
            benchmark_max_probes=self._benchmark_max_probes,
            validation_query_count=count,
            query_source=_VALIDATION_QUERY_SOURCE,
            query_selection_sha256=query_selection_sha256,
            corpus_query_overlap=corpus_query_overlap,
            exact_result_ids=tuple(tuple(row_ids) for row_ids in exact_ids),
            candidate_result_ids=tuple(tuple(row_ids) for row_ids in candidate_ids),
            recall_at_10=recalls[10],
            recall_at_20=recalls[20],
            unindexed_dense_rows=unindexed_dense_rows,
            exact_verification_ms=exact_batch.elapsed_ms,
            ann_verification_ms=ann_verification_ms,
            benchmark_duration_ms=(time.perf_counter() - benchmark_started) * 1000,
        )
        validate_candidate_publication_evidence(evidence, policy)
        return evidence, observation


    @staticmethod
    def _sha256_file(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _stable_hash(value: dict) -> str:
        import json
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    @staticmethod
    def _exact_term(chunks: Sequence[SparseChunk]) -> str:
        for token in reversed(chunks[0].fts_text.split()):
            token = token.strip()
            if token and token.isalnum():
                return token
        raise RuntimeError("No exact FTS token is available for staged validation")

    @staticmethod
    def _disk_bytes(build_dir: Path) -> int:
        return sum(path.stat().st_size for path in build_dir.rglob("*") if path.is_file())

    @staticmethod
    def _sparse_plan(wiki_dir: Path) -> tuple[SparseChunk, ...]:
        chunks: List[SparseChunk] = []
        for path in sorted(wiki_dir.rglob("*.md")):
            if ".graph" in path.parts:
                continue
            raw = path.read_text(encoding="utf-8", errors="replace")
            front_matter, body = ("", raw.strip())
            if raw.startswith("---"):
                parts = raw.split("---", 2)
                front_matter, body = parts[1], parts[-1].strip()
            if not body:
                continue
            title = path.stem
            for line in front_matter.splitlines():
                if line.startswith("title:"):
                    title = line.partition(":")[2].strip().strip("\"'") or title
                    break
            digest = hashlib.sha256(f"{path.resolve()}\0{body}".encode("utf-8")).hexdigest()
            page_id = str(path.resolve())
            chunks.append(SparseChunk(
                chunk_id=f"sparse:{digest}", page_id=page_id, path=str(path),
                title=title, text=body, fts_text=body, chunk_kind="sparse",
                end_char=len(body),
            ))
            # issue #47：legacy 无 tokenizer 路径（build_storage_contract 不带 tokenizer）
            # 仍需产出 dense chunk，否则 _build 的「两种 kind 都必须存在」守卫会拒绝。
            # 该 dense 是整页 chunk（仅用于 legacy fallback；生产路径走 tokenizer 计划）。
            chunks.append(SparseChunk(
                chunk_id=f"dense:{digest}", page_id=page_id, path=str(path),
                title=title, text=body, fts_text=body, chunk_kind="dense",
                end_char=len(body),
            ))
        return tuple(chunks)
