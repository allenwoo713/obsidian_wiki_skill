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
    publish_pointer, read_generation_record, reconcile_committed_record, record_building,
    record_validated, resolve_active_lance_dir,
)
from obsidian_wiki.application.build_lock import BuildLock, new_build_context
from obsidian_wiki.application.durable_filesystem import CommitUncertainError
from obsidian_wiki.application.index_build_service import IndexBuildService
from obsidian_wiki.domain.incremental_models import (
    BuildTelemetry, BuildTiming, CoverageObservation, IncrementalBuildResult,
    IncrementalJournalRecord, IncrementalJournalState, MutationResult,
    SourceTableIdentity, TableDelta, TableRowCounts,
)
from obsidian_wiki.domain.index_models import (
    DenseChunk, FtsIndexConfig, INDEX_LAYOUT_VERSION, INDEX_MANIFEST_FORMAT_VERSION,
    PostCommitTask, PostCommitTaskState, SparseChunk, VectorIndexConfig,
)
from obsidian_wiki.domain.index_publication_models import GenerationState
from obsidian_wiki.infrastructure.filesystem_index_manifest import FilesystemIndexManifest
from obsidian_wiki.infrastructure.filesystem_incremental_journal import FilesystemIncrementalJournal
from obsidian_wiki.infrastructure.filesystem_post_commit_journal import FilesystemPostCommitJournal
from obsidian_wiki.infrastructure.lancedb_index_repository import LanceDbIndexRepository
from obsidian_wiki.application.incremental_policy import compatibility_digest_from_manifest


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

    @staticmethod
    def _digest(value: object) -> str:
        if isinstance(value, bytes):
            payload = value
        else:
            payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @classmethod
    def _plan_digest(cls, lexical: Sequence[SparseChunk], dense: Sequence[DenseChunk]) -> str:
        return cls._digest({
            "sparse": [chunk.to_json() for chunk in lexical],
            "dense": [chunk.to_json() for chunk in dense],
        })

    @staticmethod
    def _source_match(
        record: IncrementalJournalRecord, *, pointer_digest: str,
        source_build_id: str,
        source_tables: tuple[SourceTableIdentity, ...], plan_digest: str,
        config_digest: str, policy_digest: str, index_dir: Path,
    ) -> bool:
        target = index_dir / record.target_build
        return (
            record.prior_pointer_sha256 == pointer_digest
            and record.source_build_id == source_build_id
            and record.source_tables == source_tables
            and record.plan_sha256 == plan_digest
            and record.config_sha256 == config_digest
            and record.policy_sha256 == policy_digest
            and target == index_dir / "builds" / record.build_id
            and target.is_dir() and not target.is_symlink()
        )

    def recover(self, index_dir: Path) -> tuple[str, ...]:
        """Reconcile only possibly committed pointer writes under the normal build lock."""
        ctx = new_build_context()
        lock = BuildLock(index_dir, ctx=ctx)
        lock.acquire()
        try:
            journal = FilesystemIncrementalJournal(index_dir)
            reconciled: list[str] = []
            for record in journal.nonterminal():
                if record.state is not IncrementalJournalState.VALIDATED:
                    continue
                build_dir = index_dir / record.target_build
                if build_dir != index_dir / "builds" / record.build_id:
                    journal.abort(record.build_id, "target path containment mismatch; snapshot required")
                    continue
                committed = reconcile_committed_record(
                    index_dir, build_dir, build_id=record.build_id, generation=record.generation,
                )
                lifecycle = read_generation_record(build_dir)
                if not committed and lifecycle is not None and lifecycle.state is GenerationState.PUBLISHED:
                    try:
                        committed = resolve_active_lance_dir(index_dir) == build_dir / "lance_db"
                    except RuntimeError:
                        committed = False
                if committed:
                    journal.transition(record.build_id, IncrementalJournalState.PUBLISHED, boundary="pointer_reconciled")
                    reconciled.append(record.build_id)
            return tuple(reconciled)
        finally:
            lock.release()

    @staticmethod
    def required_ancestor_build_ids(index_dir: Path) -> frozenset[str]:
        """Return source generations reachable from ACTIVE_INDEX or unfinished staging.

        Absence/corruption is intentionally handled by ``assert_cleanup_allowed`` as a
        fail-closed condition; this method only returns proven edges.
        """
        try:
            pointer = json.loads((Path(index_dir) / "ACTIVE_INDEX").read_text(encoding="utf-8"))
            active_build = str(pointer["build_id"])
        except (OSError, json.JSONDecodeError, KeyError, TypeError):
            return frozenset()
        journal = FilesystemIncrementalJournal(index_dir)
        edges = {record.build_id: record.source_build_id for record in journal.records()}
        roots = [active_build, *(record.build_id for record in journal.nonterminal())]
        required: set[str] = set()
        for root in roots:
            current = root
            seen: set[str] = set()
            while current in edges:
                if current in seen:
                    return frozenset()
                seen.add(current)
                current = edges[current]
                required.add(current)
        return frozenset(required)

    @classmethod
    def assert_cleanup_allowed(cls, index_dir: Path, build_id: str, *, probe_verified: bool = False) -> None:
        """Fail closed: Phase 4 records retention evidence but enables no pruning."""
        if not probe_verified:
            raise RuntimeError("lineage cleanup probe evidence is missing; retain generation")
        required = cls.required_ancestor_build_ids(index_dir)
        if not required or build_id in required:
            raise RuntimeError("lineage retention guard blocks reachable or unproven generation cleanup")

    def build(self, wiki_dir: Path, index_dir: Path, *, canonical_chunks: Sequence[SparseChunk],
              embed: Embedder, page_metadata: list[dict] | None = None) -> IncrementalBuildResult:
        del wiki_dir  # Canonical planning occurs under the caller's existing writer boundary.
        build_started = time.perf_counter()
        ctx = new_build_context()
        lock = BuildLock(index_dir, ctx=ctx)
        lock.acquire()
        try:
            active_lance = resolve_active_lance_dir(index_dir)
            source_manifest = self._assert_current_manifest(active_lance)
            lexical = tuple(chunk for chunk in canonical_chunks if chunk.chunk_kind == "sparse")
            if not lexical or len(lexical) + sum(1 for chunk in canonical_chunks if chunk.chunk_kind == "dense") != len(canonical_chunks):
                raise RuntimeError("canonical plan must contain only non-empty sparse and dense populations")
            embedding_started = time.perf_counter()
            dense = self._dense_plan(canonical_chunks, embed)
            embedding_cache_miss_ms = (time.perf_counter() - embedding_started) * 1000
            if not dense or len({chunk.chunk_id for chunk in lexical}) != len(lexical) or len({chunk.chunk_id for chunk in dense}) != len(dense):
                raise ValueError("canonical sparse/dense populations require unique stable IDs")
            if {chunk.chunk_id for chunk in lexical} & {chunk.chunk_id for chunk in dense}:
                raise ValueError("cross-kind stable chunk IDs are forbidden")

            source = LanceDbIndexRepository(active_lance)
            source_tables = source.source_table_identities()
            pointer_payload = (index_dir / "ACTIVE_INDEX").read_bytes()
            pointer_digest = self._digest(pointer_payload)
            source_build_id = str(json.loads(pointer_payload.decode("utf-8"))["build_id"])
            plan_digest = self._plan_digest(lexical, dense)
            config_digest = self._digest(source_manifest)
            policy_digest = self._digest(source_manifest["ann_policy"])
            journal = FilesystemIncrementalJournal(index_dir)
            if journal.has_invalid_records():
                raise RuntimeError("incremental recovery journal is malformed; snapshot required")
            pending = journal.nonterminal()
            if len(pending) > 1:
                raise RuntimeError("multiple incremental recovery candidates; snapshot required")
            if pending:
                record = pending[0]
                if not self._source_match(
                    record, pointer_digest=pointer_digest, source_tables=source_tables,
                    source_build_id=source_build_id,
                    plan_digest=plan_digest, config_digest=config_digest, policy_digest=policy_digest,
                    index_dir=index_dir,
                ):
                    journal.abort(record.build_id, "source identity mismatch; snapshot required")
                    raise RuntimeError("incremental recovery identity mismatch; snapshot required")
                build_id = record.build_id
                generation = record.generation
                build_dir = index_dir / record.target_build
            else:
                build_id = ctx.build_id
                generation = IndexBuildService._next_generation(index_dir)
                build_dir = index_dir / "builds" / build_id
                build_dir.mkdir(parents=True, exist_ok=False)
                record_building(build_dir, build_id=build_id, generation=generation)
                record = journal.prepare(IncrementalJournalRecord(
                    schema_version=1, build_id=build_id, generation=generation,
                    state=IncrementalJournalState.PREPARED,
                    prior_pointer_sha256=pointer_digest, source_tables=source_tables,
                    source_build_id=source_build_id,
                    plan_sha256=plan_digest, config_sha256=config_digest, policy_sha256=policy_digest,
                    target_build=f"builds/{build_id}", last_completed_boundary="prepared",
                ))
            lance_dir = build_dir / "lance_db"
            pointer_committed = False
            try:
                sparse_delta, sparse_added, sparse_updated = self._delta(
                    "sparse_chunks", source.table_rows("sparse_chunks"),
                    tuple(chunk.__dict__ for chunk in lexical),
                )
                dense_delta, dense_added, dense_updated = self._delta(
                    "dense_chunks", source.table_rows("dense_chunks"),
                    tuple(chunk.__dict__ for chunk in dense),
                )
                if record.state is IncrementalJournalState.PREPARED:
                    cloned = source.clone_tables(lance_dir)
                    if cloned != source_tables:
                        raise RuntimeError("source table versions changed during clone; snapshot required")
                    record = journal.transition(build_id, IncrementalJournalState.CLONED, boundary="clone")
                staging = LanceDbIndexRepository(lance_dir)
                write_started = time.perf_counter()
                if record.state is IncrementalJournalState.CLONED and record.last_completed_boundary != "sparse_mutated":
                    sparse_result = staging.apply_delta("sparse_chunks", added=sparse_added, updated=sparse_updated, deleted_ids=sparse_delta.deleted_ids)
                    journal.checkpoint(build_id, boundary="sparse_mutated")
                    record = journal.load(build_id)
                    assert record is not None
                else:
                    sparse_result = MutationResult("sparse_chunks", len(sparse_delta.added_ids), len(sparse_delta.updated_ids), len(sparse_delta.deleted_ids))
                if record.state is IncrementalJournalState.CLONED:
                    dense_result = staging.apply_delta("dense_chunks", added=dense_added, updated=dense_updated, deleted_ids=dense_delta.deleted_ids)
                    record = journal.transition(build_id, IncrementalJournalState.MUTATED, boundary="dense_mutated")
                else:
                    dense_result = MutationResult("dense_chunks", len(dense_delta.added_ids), len(dense_delta.updated_ids), len(dense_delta.deleted_ids))
                if sparse_result.physically_written != len(sparse_delta.physically_written_ids) or dense_result.physically_written != len(dense_delta.physically_written_ids):
                    raise RuntimeError("adapter mutation accounting does not reconcile with delta")
                serialization_write_ms = (time.perf_counter() - write_started) * 1000
                vector_config = VectorIndexConfig(
                    index_type="hnsw_sq", metric="cosine", num_partitions=1, m=16,
                    ef_construction=300, dense_chunks_count=len(dense),
                )
                if record.state is IncrementalJournalState.MUTATED:
                    fts_started = time.perf_counter()
                    sparse_coverage = staging.catch_up(self._fts_config)
                    fts_catch_up_ms = (time.perf_counter() - fts_started) * 1000
                    vector_started = time.perf_counter()
                    staging.create_vector_index(vector_config)
                    vector_catch_up_ms = (time.perf_counter() - vector_started) * 1000
                    record = journal.transition(build_id, IncrementalJournalState.CAUGHT_UP, boundary="index_catch_up")
                else:
                    fts_catch_up_ms = 0.0
                    vector_catch_up_ms = 0.0
                    fts_stats = staging.fts_index_stats()
                    sparse_coverage = CoverageObservation(
                        "sparse_chunks", len(staging.table_rows("sparse_chunks")),
                        fts_stats.indexed_rows, fts_stats.unindexed_rows,
                    )
                dense_stats = staging.vector_index_stats(vector_config.index_name)
                dense_coverage = CoverageObservation("dense_chunks", len(dense), dense_stats.indexed_rows, dense_stats.unindexed_dense_rows)
                if (sparse_coverage.unindexed_rows is None or sparse_coverage.unindexed_rows != 0
                    or sparse_coverage.indexed_rows != len(lexical)
                    or dense_coverage.unindexed_rows is None or dense_coverage.unindexed_rows != 0
                    or dense_coverage.indexed_rows != len(dense)):
                    raise RuntimeError("staged index coverage is incomplete")
                validation_started = time.perf_counter()
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
                validation_ms = (time.perf_counter() - validation_started) * 1000
                if record.state is IncrementalJournalState.CAUGHT_UP:
                    staging.seal(lance_dir)
                    manifest_kwargs = dict(
                        counts=counts.to_json(), vector_stats=vector_stats.to_json(), fts_stats=fts_stats.to_json(),
                        vector_config=vector_config, benchmark=benchmark.to_json(),
                        policy={"selected_mode": "ann", "reason": "approved fixed ann policy validated by held-out publication evidence", "benchmark": benchmark.to_json(), "index_stats": vector_stats.to_json(), "benchmark_scope": "held_out", "benchmark_probe_count": publication_evidence.validation_query_count, "benchmark_probe_total": counts.dense_chunks_count},
                        sparse_chunks=canonical_chunks, generation=generation, build_id=build_id,
                        page_metadata=page_metadata, ann_policy=publisher._ann_policy,
                        publication_evidence=publication_evidence,
                    )
                    manifest = publisher._manifest(**manifest_kwargs)
                    telemetry = BuildTelemetry(
                        schema_version=1, observation_id=build_id,
                        mode_requested="incremental", mode_selected="incremental",
                        selection_reason="explicit_incremental",
                        compatibility_digest=compatibility_digest_from_manifest(manifest),
                        completed_at_epoch_seconds=time.time(),
                        timings=BuildTiming(
                            scan_parse_ms=(time.perf_counter() - build_started) * 1000,
                            chunking_ms=0.0, embedding_cache_hit_ms=0.0,
                            embedding_cache_miss_ms=embedding_cache_miss_ms,
                            serialization_write_ms=serialization_write_ms,
                            fts_catch_up_ms=fts_catch_up_ms, vector_catch_up_ms=vector_catch_up_ms,
                            validation_ms=validation_ms, publication_ms=0.0,
                            index_rebuild_ms=vector_catch_up_ms,
                        ),
                        sparse_rows=TableRowCounts(
                            inserted=sparse_result.inserted, updated=sparse_result.updated,
                            deleted=sparse_result.deleted, unchanged=len(sparse_delta.unchanged_ids),
                            physically_written=sparse_result.physically_written,
                        ),
                        dense_rows=TableRowCounts(
                            inserted=dense_result.inserted, updated=dense_result.updated,
                            deleted=dense_result.deleted, unchanged=len(dense_delta.unchanged_ids),
                            physically_written=dense_result.physically_written,
                        ),
                        embedding_cache_hits=0, embedding_cache_misses=len(dense),
                        peak_staged_disk_bytes=publisher._disk_bytes(build_dir), completed=True,
                    )
                    manifest = publisher._manifest(**manifest_kwargs, build_telemetry=telemetry)
                    manifest_path = build_dir / "manifest.json"
                    FilesystemIndexManifest().write(manifest_path, manifest)
                    record_validated(build_dir, generation=generation, build_id=build_id,
                                     manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
                    record = journal.transition(build_id, IncrementalJournalState.VALIDATED, boundary="manifest_validated")
                else:
                    manifest_path = build_dir / "manifest.json"
                FilesystemPostCommitJournal(index_dir).prepare(PostCommitTask(
                    task_id=uuid.uuid4().hex, task_type="community_report_invalidation",
                    build_id=build_id, generation=generation, state=PostCommitTaskState.PREPARED,
                    prepared_at=datetime.now(timezone.utc).isoformat(),
                ))
                publish_pointer(index_dir, build_dir, generation=generation, build_id=build_id)
                pointer_committed = True
                journal.transition(build_id, IncrementalJournalState.PUBLISHED, boundary="pointer_published")
                from obsidian_wiki.domain.index_models import StorageArtifact
                return IncrementalBuildResult(
                    StorageArtifact(lance_dir, manifest_path, len(lexical), len(dense), build_id, generation),
                    source_tables, sparse_delta, dense_delta, sparse_result, dense_result,
                    sparse_coverage, dense_coverage,
                )
            except CommitUncertainError:
                # replace may have committed; restart must reconcile pointer before journal classification.
                raise
            except Exception as exc:
                if not pointer_committed:
                    journal.abort(build_id, f"{type(exc).__name__}: {exc}")
                raise
        finally:
            lock.release()
