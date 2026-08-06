"""Small orchestration layer for the first D-01/D-04 persisted tracer."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import statistics
import time
import uuid
from pathlib import Path
from typing import Callable, List, Sequence

from obsidian_wiki.application.active_index_pointer import publish_pointer, record_validated
from obsidian_wiki.application.build_lock import BuildLock, new_build_context
from obsidian_wiki.domain.index_models import (
    BenchmarkObservation,
    BuildContext,
    DenseChunk,
    FtsIndexConfig,
    IndexStats,
    SparseChunk,
    StorageArtifact,
    VectorIndexConfig,
)
from obsidian_wiki.domain.index_policy import select_vector_policy
from obsidian_wiki.ports.chunk_repository import ChunkRepository
from obsidian_wiki.ports.index_manifest import IndexManifestStore


Embedder = Callable[[Sequence[str]], Sequence[Sequence[float]]]
BenchmarkObserver = Callable[[IndexStats], BenchmarkObservation]


class IndexBuildService:
    """Partition canonical Markdown into physically separate sparse/dense rows."""

    def __init__(
        self,
        storage: ChunkRepository,
        *,
        reopen_storage: Callable[[Path], ChunkRepository],
        manifest_store: IndexManifestStore,
        fts_config: FtsIndexConfig | None = None,
        benchmark_observer: BenchmarkObserver | None = None,
    ):
        self._storage = storage
        self._reopen_storage = reopen_storage
        self._manifest_store = manifest_store
        self._fts_config = fts_config or FtsIndexConfig()
        self._benchmark_observer = benchmark_observer

    def build(
        self, wiki_dir: Path, index_dir: Path, *, embed: Embedder,
        sparse_chunks: Sequence[SparseChunk] | None = None,
        page_metadata: list[dict] | None = None,
        image_metadata: list[dict] | None = None,
        ctx: BuildContext | None = None,
    ) -> StorageArtifact:
        """#21/#34 单写者构建：最外层传入或创建一次 BuildContext，锁 metadata、
        build 目录、manifest、pointer 与返回 artifact 共用同一个 build_id；
        service 不再独立生成 ID。"""
        ctx = ctx or new_build_context()
        lock = BuildLock(index_dir, ctx=ctx)
        lock.acquire()
        try:
            return self._build(
                wiki_dir, index_dir, embed=embed,
                sparse_chunks=sparse_chunks, ctx=ctx,
                page_metadata=page_metadata, image_metadata=image_metadata,
            )
        finally:
            lock.release()

    def _build(
        self, wiki_dir: Path, index_dir: Path, *, embed: Embedder,
        sparse_chunks: Sequence[SparseChunk] | None = None, ctx: BuildContext | None = None,
        page_metadata: list[dict] | None = None,
        image_metadata: list[dict] | None = None,
    ) -> StorageArtifact:
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

        build_dir = index_dir / "builds" / ctx.build_id
        lance_dir = build_dir / "lance_db"
        build_dir.mkdir(parents=True, exist_ok=False)
        try:
            self._storage.persist(lance_dir, sparse_chunks, dense_chunks, self._fts_config)
            # Reopen through a new adapter instance: inputs and an open write handle are not evidence.
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
            )
            policy = select_vector_policy(benchmark, vector_stats)
            generation = self._next_generation(index_dir)
            manifest = self._manifest(
                counts=counts.to_json(), vector_stats=vector_stats.to_json(),
                fts_stats=fts_stats.to_json(), vector_config=vector_config,
                benchmark={**benchmark.to_json(), **benchmark_evidence}, policy=policy.to_json(),
                sparse_chunks=sparse_chunks, generation=generation,
                page_metadata=page_metadata, image_metadata=image_metadata,
            )
            manifest_path = build_dir / "manifest.json"
            self._manifest_store.write(manifest_path, manifest)
            # #35：manifest 完整落盘后写入 validated 生命周期记录——manifest 写后、
            # pointer 发布前中断的 staging generation 绝不被 recovery 选中。
            record_validated(
                build_dir, generation=generation, build_id=ctx.build_id,
                manifest_sha256=self._sha256_file(manifest_path),
            )
            publish_pointer(index_dir, build_dir, generation=generation, build_id=ctx.build_id)
            return StorageArtifact(lance_dir, manifest_path, len(sparse_chunks), len(dense_chunks))
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
            "format_version": 4,
            "layout": "sparse_chunks+dense_chunks",
            "generation": generation,
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

    def _benchmark(
        self,
        repository: ChunkRepository,
        dense_chunks: Sequence[DenseChunk],
        stats: IndexStats,
        *,
        build_time_ms: float,
        disk_bytes: int,
    ) -> tuple[BenchmarkObservation, dict]:
        """Measure the candidate against exact bypass with the same query contract.

        Latency is evidence only.  The policy consumes only deterministic recall
        and coverage observations, so runner variance cannot change publication.
        """
        if self._benchmark_observer is not None:
            observation = self._benchmark_observer(stats)
            return observation, {"exact_result_ids": [], "candidate_result_ids": []}

        exact_ids: list[list[str]] = []
        candidate_ids: list[list[str]] = []
        candidate_durations: list[float] = []
        recalls: dict[int, list[float]] = {10: [], 20: []}
        for chunk in dense_chunks:
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
        return observation, {
            "exact_result_ids": exact_ids,
            "candidate_result_ids": candidate_ids,
        }

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
