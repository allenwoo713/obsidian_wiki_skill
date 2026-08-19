"""Staged, stable-ID online mutations for the immutable ACTIVE_INDEX contract."""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

from obsidian_wiki.application.active_index_pointer import (
    publish_pointer, record_building, record_validated, resolve_active_lance_dir,
)
from obsidian_wiki.application.build_lock import BuildLock, new_build_context
from obsidian_wiki.application.index_build_service import IndexBuildService
from obsidian_wiki.domain.incremental_models import (
    CoverageObservation, IncrementalBuildResult, TableDelta,
)
from obsidian_wiki.domain.index_models import (
    DenseChunk, FtsIndexConfig, INDEX_LAYOUT_VERSION, INDEX_MANIFEST_FORMAT_VERSION,
    PostCommitTask, PostCommitTaskState, SparseChunk, VectorIndexConfig,
)
from obsidian_wiki.infrastructure.filesystem_index_manifest import FilesystemIndexManifest
from obsidian_wiki.infrastructure.filesystem_post_commit_journal import FilesystemPostCommitJournal
from obsidian_wiki.infrastructure.lancedb_index_repository import LanceDbIndexRepository


Embedder = Callable[[Sequence[str]], Sequence[Sequence[float]]]


class IncrementalIndexService:
    """Clone a verified active generation, mutate only staging, then publish once."""

    def __init__(self, *, fts_config: FtsIndexConfig | None = None) -> None:
        self._fts_config = fts_config or FtsIndexConfig()

    @staticmethod
    def _fingerprint(row: Mapping[str, object]) -> str:
        def normalize(value: object) -> object:
            if isinstance(value, float):
                return round(value, 6)
            if isinstance(value, list):
                return [normalize(item) for item in value]
            if isinstance(value, tuple):
                return [normalize(item) for item in value]
            if isinstance(value, dict):
                return {str(key): normalize(item) for key, item in value.items()}
            return value
        canonical = {key: normalize(value) for key, value in row.items() if key != "_distance"}
        return hashlib.sha256(json.dumps(canonical, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()

    @classmethod
    def _delta(cls, table_name: str, source: Sequence[Mapping[str, object]],
               planned: Sequence[Mapping[str, object]]) -> tuple[TableDelta, tuple[Mapping[str, object], ...], tuple[Mapping[str, object], ...]]:
        source_by_id = {str(row["chunk_id"]): row for row in source}
        planned_by_id = {str(row["chunk_id"]): row for row in planned}
        if len(source_by_id) != len(source) or len(planned_by_id) != len(planned):
            raise ValueError(f"{table_name} stable IDs must be unique")
        added = tuple(sorted(set(planned_by_id) - set(source_by_id)))
        deleted = tuple(sorted(set(source_by_id) - set(planned_by_id)))
        shared = set(source_by_id) & set(planned_by_id)
        updated = tuple(sorted(chunk_id for chunk_id in shared if cls._fingerprint(source_by_id[chunk_id]) != cls._fingerprint(planned_by_id[chunk_id])))
        unchanged = tuple(sorted(shared - set(updated)))
        return (
            TableDelta(table_name, added, updated, deleted, unchanged),
            tuple(planned_by_id[chunk_id] for chunk_id in added),
            tuple(planned_by_id[chunk_id] for chunk_id in updated),
        )

    @staticmethod
    def _assert_current_manifest(active_lance: Path) -> dict[str, object]:
        manifest_path = active_lance.parent / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError("incremental source manifest is unreadable; snapshot required") from exc
        if not isinstance(manifest, dict) or (
            manifest.get("layout") != "sparse_chunks+dense_chunks"
            or manifest.get("format_version") != INDEX_MANIFEST_FORMAT_VERSION
            or manifest.get("index_layout_version") != INDEX_LAYOUT_VERSION
        ):
            raise RuntimeError("incremental source contract mismatch; snapshot required")
        ann = manifest.get("ann_policy")
        if not isinstance(ann, dict) or ann.get("selected_index_type") != "ivf-hnsw-sq" or ann.get("query_ef") != 100:
            raise RuntimeError("incremental source ANN contract mismatch; snapshot required")
        return manifest

    @staticmethod
    def _dense_plan(chunks: Sequence[SparseChunk], embed: Embedder) -> tuple[DenseChunk, ...]:
        dense_sources = tuple(chunk for chunk in chunks if chunk.chunk_kind == "dense")
        vectors = embed([chunk.text for chunk in dense_sources])
        if len(vectors) != len(dense_sources):
            raise RuntimeError("embedder returned a vector count different from the dense plan")
        return tuple(DenseChunk(
            chunk_id=chunk.chunk_id, page_id=chunk.page_id, path=chunk.path,
            title=chunk.title, text=chunk.text, vector=tuple(float(value) for value in vector),
            page_type=chunk.page_type, section_path=chunk.section_path, heading=chunk.heading,
            chunk_kind=chunk.chunk_kind, chunk_index=chunk.chunk_index,
            parent_section_id=chunk.parent_section_id, token_count=chunk.token_count,
            content_hash=chunk.content_hash, forced_split=chunk.forced_split,
            continuation_index=chunk.continuation_index, start_char=chunk.start_char,
            end_char=chunk.end_char,
        ) for chunk, vector in zip(dense_sources, vectors))

    def build(self, wiki_dir: Path, index_dir: Path, *, canonical_chunks: Sequence[SparseChunk],
              embed: Embedder, page_metadata: list[dict] | None = None) -> IncrementalBuildResult:
        del wiki_dir  # Canonical planning occurs under the caller's existing writer boundary.
        ctx = new_build_context()
        lock = BuildLock(index_dir, ctx=ctx)
        lock.acquire()
        try:
            active_lance = resolve_active_lance_dir(index_dir)
            self._assert_current_manifest(active_lance)
            lexical = tuple(chunk for chunk in canonical_chunks if chunk.chunk_kind == "sparse")
            if not lexical or len(lexical) + sum(1 for chunk in canonical_chunks if chunk.chunk_kind == "dense") != len(canonical_chunks):
                raise RuntimeError("canonical plan must contain only non-empty sparse and dense populations")
            dense = self._dense_plan(canonical_chunks, embed)
            if not dense or len({chunk.chunk_id for chunk in lexical}) != len(lexical) or len({chunk.chunk_id for chunk in dense}) != len(dense):
                raise ValueError("canonical sparse/dense populations require unique stable IDs")
            if {chunk.chunk_id for chunk in lexical} & {chunk.chunk_id for chunk in dense}:
                raise ValueError("cross-kind stable chunk IDs are forbidden")

            source = LanceDbIndexRepository(active_lance)
            generation = IndexBuildService._next_generation(index_dir)
            build_dir = index_dir / "builds" / ctx.build_id
            lance_dir = build_dir / "lance_db"
            build_dir.mkdir(parents=True, exist_ok=False)
            record_building(build_dir, build_id=ctx.build_id, generation=generation)
            try:
                source_tables = source.clone_tables(lance_dir)
                sparse_delta, sparse_added, sparse_updated = self._delta(
                    "sparse_chunks", source.table_rows("sparse_chunks"),
                    tuple(chunk.__dict__ for chunk in lexical),
                )
                dense_delta, dense_added, dense_updated = self._delta(
                    "dense_chunks", source.table_rows("dense_chunks"),
                    tuple(chunk.__dict__ for chunk in dense),
                )
                staging = LanceDbIndexRepository(lance_dir)
                sparse_result = staging.apply_delta("sparse_chunks", added=sparse_added, updated=sparse_updated, deleted_ids=sparse_delta.deleted_ids)
                dense_result = staging.apply_delta("dense_chunks", added=dense_added, updated=dense_updated, deleted_ids=dense_delta.deleted_ids)
                if sparse_result.physically_written != len(sparse_delta.physically_written_ids) or dense_result.physically_written != len(dense_delta.physically_written_ids):
                    raise RuntimeError("adapter mutation accounting does not reconcile with delta")
                sparse_coverage = staging.catch_up(self._fts_config)
                vector_config = VectorIndexConfig(
                    index_type="hnsw_sq", metric="cosine", num_partitions=1, m=16,
                    ef_construction=300, dense_chunks_count=len(dense),
                )
                staging.create_vector_index(vector_config)
                dense_stats = staging.vector_index_stats(vector_config.index_name)
                dense_coverage = CoverageObservation("dense_chunks", len(dense), dense_stats.indexed_rows, dense_stats.unindexed_dense_rows)
                if (sparse_coverage.unindexed_rows is None or sparse_coverage.unindexed_rows != 0
                    or sparse_coverage.indexed_rows != len(lexical)
                    or dense_coverage.unindexed_rows is None or dense_coverage.unindexed_rows != 0
                    or dense_coverage.indexed_rows != len(dense)):
                    raise RuntimeError("staged index coverage is incomplete")
                reopened = LanceDbIndexRepository(lance_dir)
                counts, vector_stats, fts_stats = reopened.validate_reopened(
                    dimension=len(dense[0].vector), exact_term=IndexBuildService._exact_term(lexical),
                    vector_index_name=vector_config.index_name,
                )
                publisher = IndexBuildService(
                    staging, reopen_storage=LanceDbIndexRepository,
                    manifest_store=FilesystemIndexManifest(),
                    post_commit_journal=FilesystemPostCommitJournal(index_dir),
                    fts_config=self._fts_config,
                )
                index_build_started = time.perf_counter()
                publication_evidence, benchmark = publisher._publication_validation(
                    reopened, dense, vector_stats=vector_stats,
                    actual_dense_rows=counts.dense_chunks_count,
                    unindexed_dense_rows=vector_stats.unindexed_dense_rows,
                    build_time_ms=(time.perf_counter() - index_build_started) * 1000,
                    disk_bytes=publisher._disk_bytes(build_dir),
                )
                staging.seal(lance_dir)
                manifest = publisher._manifest(
                    counts=counts.to_json(), vector_stats=vector_stats.to_json(), fts_stats=fts_stats.to_json(),
                    vector_config=vector_config, benchmark=benchmark.to_json(),
                    policy={"selected_mode": "ann", "reason": "approved fixed ann policy validated by held-out publication evidence", "benchmark": benchmark.to_json(), "index_stats": vector_stats.to_json(), "benchmark_scope": "held_out", "benchmark_probe_count": publication_evidence.validation_query_count, "benchmark_probe_total": counts.dense_chunks_count},
                    sparse_chunks=canonical_chunks, generation=generation, build_id=ctx.build_id,
                    page_metadata=page_metadata, ann_policy=publisher._ann_policy,
                    publication_evidence=publication_evidence,
                )
                manifest_path = build_dir / "manifest.json"
                FilesystemIndexManifest().write(manifest_path, manifest)
                record_validated(build_dir, generation=generation, build_id=ctx.build_id,
                                 manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
                FilesystemPostCommitJournal(index_dir).prepare(PostCommitTask(
                    task_id=uuid.uuid4().hex, task_type="community_report_invalidation",
                    build_id=ctx.build_id, generation=generation, state=PostCommitTaskState.PREPARED,
                    prepared_at=datetime.now(timezone.utc).isoformat(),
                ))
                publish_pointer(index_dir, build_dir, generation=generation, build_id=ctx.build_id)
                from obsidian_wiki.domain.index_models import StorageArtifact
                return IncrementalBuildResult(
                    StorageArtifact(lance_dir, manifest_path, len(lexical), len(dense), ctx.build_id, generation),
                    source_tables, sparse_delta, dense_delta, sparse_coverage, dense_coverage,
                )
            except Exception as exc:
                (build_dir / ".failed").write_text(
                    f"incremental failed before publication: {type(exc).__name__}: {exc}", encoding="utf-8",
                )
                raise
        finally:
            lock.release()
