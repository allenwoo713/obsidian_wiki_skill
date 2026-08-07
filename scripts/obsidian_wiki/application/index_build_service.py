"""Small orchestration layer for the first D-01/D-04 persisted tracer."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import statistics
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Sequence

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
    DenseChunk,
    FtsIndexConfig,
    IndexStats,
    PostCommitTask,
    PostCommitTaskState,
    SparseChunk,
    StorageArtifact,
    VectorIndexConfig,
)
from obsidian_wiki.domain.index_policy import select_vector_policy
from obsidian_wiki.ports.chunk_repository import ChunkRepository
from obsidian_wiki.ports.index_manifest import IndexManifestStore
from obsidian_wiki.ports.post_commit import PostCommitJournal


Embedder = Callable[[Sequence[str]], Sequence[Sequence[float]]]
BenchmarkObserver = Callable[[IndexStats], BenchmarkObservation]
# #39 (review)：持锁后运行的分块回调，返回 (sparse_chunks, page_metadata)。
PlanProvider = Callable[
    [Path], tuple[Sequence["SparseChunk"], "list[dict] | None"]
]
# #41：构建期 ANN 自检的最大 probe 数。recall 语义从「全量最小」变为
# 「bottom-k SHA-256 样本最小」；evidence 与 policy 必须自证该口径。
BENCHMARK_MAX_PROBES = 256


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
    ):
        """#37：``post_commit_journal`` 为必填依赖——每个发布路径都必须在 pointer
        commit 前 durable prepare 失效 intent；不需要 invalidation 的调用方必须显式
        注入 deliberate no-op port，遗漏绝不能静默禁用契约。

        #41：``benchmark_max_probes`` 在 storage mutation 之前验证（构造时即拒绝
        非法值）；bool 是 int 子类，须显式拒绝。
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

    def build(
        self, wiki_dir: Path, index_dir: Path, *, embed: Embedder,
        sparse_chunks: Sequence[SparseChunk] | None = None,
        page_metadata: list[dict] | None = None,
        image_metadata: list[dict] | None = None,
        ctx: BuildContext | None = None,
        plan_provider: PlanProvider | None = None,
    ) -> StorageArtifact:
        """#21/#34 单写者构建：最外层传入或创建一次 BuildContext，锁 metadata、
        build 目录、manifest、pointer 与返回 artifact 共用同一个 build_id；
        service 不再独立生成 ID。

        #39 (review)：``plan_provider`` 是持锁后运行的分块回调
        ``(wiki_dir) -> (sparse_chunks, page_metadata)``。分块必须在 BUILD.lock
        获取之后、针对已加锁的 Wiki 快照执行；调用方传入的 ``sparse_chunks``
        （显式计划）仍然优先。"""
        ctx = ctx or new_build_context()
        lock = BuildLock(index_dir, ctx=ctx)
        lock.acquire()
        try:
            return self._build(
                wiki_dir, index_dir, embed=embed,
                sparse_chunks=sparse_chunks, ctx=ctx,
                page_metadata=page_metadata, image_metadata=image_metadata,
                plan_provider=plan_provider,
            )
        finally:
            lock.release()

    def _build(
        self, wiki_dir: Path, index_dir: Path, *, embed: Embedder,
        sparse_chunks: Sequence[SparseChunk] | None = None, ctx: BuildContext | None = None,
        page_metadata: list[dict] | None = None,
        image_metadata: list[dict] | None = None,
        plan_provider: PlanProvider | None = None,
    ) -> StorageArtifact:
        # #39 (review)：持锁后再分块。显式 sparse_chunks > plan_provider > 回退整页 plan。
        if sparse_chunks is None and plan_provider is not None:
            planned_chunks, planned_pages = plan_provider(wiki_dir)
            sparse_chunks = planned_chunks
            if page_metadata is None and planned_pages is not None:
                page_metadata = planned_pages
        sparse_chunks = tuple(sparse_chunks) if sparse_chunks is not None else self._sparse_plan(wiki_dir)
        if not sparse_chunks:
            raise RuntimeError("No canonical Wiki Markdown pages were available to index")
        dense_sources = tuple(chunk for chunk in sparse_chunks if chunk.chunk_kind == "dense")
        if not dense_sources:
            raise RuntimeError("Canonical chunk plan contains no dense retrieval chunks")
        vectors = embed([chunk.text for chunk in dense_sources])
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

        generation = self._next_generation(index_dir)
        build_dir = index_dir / "builds" / ctx.build_id
        lance_dir = build_dir / "lance_db"
        build_dir.mkdir(parents=True, exist_ok=False)
        # #35：build 目录创建后立即耐久写 BUILDING（missing → building）。
        record_building(build_dir, build_id=ctx.build_id, generation=generation)
        try:
            self._storage.persist(lance_dir, sparse_chunks, dense_chunks, self._fts_config)
            # #36 follow-up：所有 storage mutation（persist / create_vector_index）都
            # 必须在最终 seal 之前完成——vector index 创建会改写 LanceDB，故 seal
            # 不能放在 persist 后（当前已修复：先建索引，再最终 seal）。
            reopened = self._reopen_storage(lance_dir)
            dimension = len(dense_chunks[0].vector)
            vector_config = VectorIndexConfig(
                index_type="hnsw_flat", metric="cosine", num_partitions=1,
                m=16, ef_construction=300, dense_chunks_count=len(dense_chunks),
            )
            index_started = time.perf_counter()
            reopened.create_vector_index(vector_config)
            index_build_ms = (time.perf_counter() - index_started) * 1000
            exact_term = self._exact_term(sparse_chunks)
            counts, vector_stats, fts_stats = reopened.validate_reopened(
                dimension=dimension, exact_term=exact_term
            )
            if (
                counts.sparse_chunks_count != len(sparse_chunks)
                or counts.dense_chunks_count != len(dense_chunks)
            ):
                raise RuntimeError(
                    "staging 持久化完整性校验失败："
                    f"sparse={counts.sparse_chunks_count}/{len(sparse_chunks)} "
                    f"dense={counts.dense_chunks_count}/{len(dense_chunks)}"
                )
            benchmark, benchmark_evidence = self._benchmark(
                reopened,
                dense_chunks,
                vector_stats,
                build_time_ms=index_build_ms,
                disk_bytes=self._disk_bytes(build_dir),
                wiki_dir=wiki_dir,
            )
            policy = select_vector_policy(benchmark, vector_stats, evidence=benchmark_evidence)
            # #36 follow-up：最终 seal = 最后 storage mutation 之后的耐久边界。
            self._storage.seal(lance_dir)
            manifest = self._manifest(
                counts=counts.to_json(), vector_stats=vector_stats.to_json(),
                fts_stats=fts_stats.to_json(), vector_config=vector_config,
                benchmark={**benchmark.to_json(), **benchmark_evidence}, policy=policy.to_json(),
                sparse_chunks=sparse_chunks, generation=generation, build_id=ctx.build_id,
                page_metadata=page_metadata, image_metadata=image_metadata,
            )
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
                lance_dir, manifest_path, len(sparse_chunks), len(dense_chunks),
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

    def _manifest(self, *, counts: dict, vector_stats: dict, fts_stats: dict,
                  vector_config: VectorIndexConfig, benchmark: dict, policy: dict,
                  sparse_chunks: Sequence[SparseChunk], generation: int = 0,
                  build_id: str = "",
                  page_metadata: list[dict] | None = None,
                  image_metadata: list[dict] | None = None) -> dict:
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
            # #41：v4 → v5——recall 语义从「全量最小」变为「样本最小」（或显式
            # synthetic observer），v4-shaped record 不得静默携带 sampled 语义。
            "format_version": 5,
            "layout": "sparse_chunks+dense_chunks",
            "generation": generation,
            "build_id": build_id,
            "fts_config": fts_config,
            "vector_config": vector_config_json,
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

    @staticmethod
    def _benchmark_probe_keys(
        dense_chunks: Sequence[DenseChunk], wiki_dir: Path
    ) -> tuple[str, ...]:
        """#41 portable probe keys：与 checkout root / 平台无关。

        chunk_id 含绝对 page_id（换根目录或 Windows runner 会选出不同 probe），
        因此 key 只用 wiki-relative POSIX path + chunk-id content suffix +
        chunk_kind + chunk_index 构造，跨机器可复现。``_make_chunk_id`` 的
        chunk_id 形如 ``{absolute_page_id}::{16位hash}``，取 ``::`` 后片段即可。
        """
        resolved_wiki = Path(wiki_dir).resolve()
        keys: list[str] = []
        for chunk in dense_chunks:
            chunk_path = Path(chunk.path).resolve()
            try:
                relative = os.path.relpath(chunk_path, resolved_wiki)
            except ValueError:  # 不同盘符（Windows）无公共根
                relative = str(chunk_path)
            relative = relative.replace(os.sep, "/")
            suffix = (
                chunk.chunk_id.rsplit("::", 1)[-1]
                if "::" in chunk.chunk_id
                else chunk.content_hash
            )
            keys.append(f"{relative}::{suffix}::{chunk.chunk_kind}::{chunk.chunk_index}")
        return tuple(keys)

    def _benchmark(
        self,
        repository: ChunkRepository,
        dense_chunks: Sequence[DenseChunk],
        stats: IndexStats,
        *,
        build_time_ms: float,
        disk_bytes: int,
        wiki_dir: Path,
    ) -> tuple[BenchmarkObservation, dict]:
        """Measure the candidate against exact bypass with the same query contract.

        Latency is evidence only.  The policy consumes only deterministic recall
        and coverage observations, so runner variance cannot change publication.

        #41：probe 数受 ``benchmark_max_probes`` 上限约束——total ≤ cap 走全量
        （scope=full，顺序不变），超过则按 (sha256(key), key) 排序取 bottom-k
        （scope=sampled，确定性、与输入顺序无关）。evidence 显式记录采样口径，
        observer 分支输出 synthetic evidence 且不查询 repository。
        """
        benchmark_started = time.perf_counter()
        if self._benchmark_observer is not None:
            observation = self._benchmark_observer(stats)
            return observation, {
                "evidence_schema_version": 1,
                "evidence_source": "observer",
                "probe_scope": "synthetic",
                "sampling_method": "synthetic",
                "sampling_key_schema": "wiki_relative_path+chunk_suffix+kind+chunk_index:v1",
                "probe_keys": [],
                "probe_selection_sha256": hashlib.sha256(b"").hexdigest(),
                "probe_count": 0,
                "probe_total": len(dense_chunks),
                "probe_coverage": 0.0,
                "result_limit": 20,
                "recall_aggregation": "minimum",
                "benchmark_duration_ms": 0.0,
                "exact_result_ids": [],
                "candidate_result_ids": [],
            }

        keys = self._benchmark_probe_keys(dense_chunks, wiki_dir)
        total = len(dense_chunks)
        if total <= self._benchmark_max_probes:
            probe_indices = list(range(total))
            probe_keys = list(keys)
            scope = "full"
            sampling_method = "full"
        else:
            ranked = sorted(
                (hashlib.sha256(key.encode("utf-8")).hexdigest(), key, index)
                for index, key in enumerate(keys)
            )[: self._benchmark_max_probes]
            probe_indices = [index for _digest, _key, index in ranked]
            probe_keys = [key for _digest, key, _index in ranked]
            scope = "sampled"
            sampling_method = "bottom_k_sha256_v1"

        exact_ids: list[list[str]] = []
        candidate_ids: list[list[str]] = []
        candidate_durations: list[float] = []
        recalls: dict[int, list[float]] = {10: [], 20: []}
        for index in probe_indices:
            chunk = dense_chunks[index]
            exact = repository.search_dense_exact(
                chunk.vector, metric="cosine", limit=20, where=None
            )
            started = time.perf_counter()
            candidate = repository.search_dense(
                chunk.vector, metric="cosine", limit=20, where=None
            )
            candidate_durations.append((time.perf_counter() - started) * 1000)
            exact_row_ids = [str(row["chunk_id"]) for row in exact]
            candidate_row_ids = [str(row["chunk_id"]) for row in candidate]
            if not exact_row_ids:
                raise RuntimeError("Exact benchmark query returned no dense rows")
            exact_ids.append(exact_row_ids)
            candidate_ids.append(candidate_row_ids)
            for limit in recalls:
                truth = set(exact_row_ids[:limit])
                observed = set(candidate_row_ids[:limit])
                recalls[limit].append(len(truth & observed) / len(truth))

        def percentile_95(samples: Sequence[float]) -> float:
            ordered = sorted(samples)
            return ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]

        observation = BenchmarkObservation(
            recall_at_10=min(recalls[10]),
            recall_at_20=min(recalls[20]),
            latency_p50_ms=statistics.median(candidate_durations),
            latency_p95_ms=percentile_95(candidate_durations),
            build_time_ms=build_time_ms,
            disk_bytes=disk_bytes,
        )
        probe_count = len(probe_indices)
        evidence = {
            "evidence_schema_version": 1,
            "evidence_source": "measured",
            "probe_scope": scope,
            "sampling_method": sampling_method,
            "sampling_key_schema": "wiki_relative_path+chunk_suffix+kind+chunk_index:v1",
            "probe_keys": probe_keys,
            "probe_selection_sha256": hashlib.sha256(
                "\n".join(probe_keys).encode("utf-8")
            ).hexdigest(),
            "probe_count": probe_count,
            "probe_total": total,
            "probe_coverage": probe_count / total,
            "result_limit": 20,
            "recall_aggregation": "minimum",
            "benchmark_duration_ms": (time.perf_counter() - benchmark_started) * 1000,
            "exact_result_ids": exact_ids,
            "candidate_result_ids": candidate_ids,
        }
        return observation, evidence

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
                title=title, text=body, fts_text=body, end_char=len(body),
            ))
        return tuple(chunks)
