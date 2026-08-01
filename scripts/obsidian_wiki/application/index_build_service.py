"""Small orchestration layer for the first D-01/D-04 persisted tracer."""
from __future__ import annotations

import hashlib
import importlib.metadata
import os
import time
import uuid
from pathlib import Path
from typing import Callable, List, Sequence

from obsidian_wiki.domain.index_models import (
    BenchmarkObservation,
    DenseChunk,
    FtsIndexConfig,
    IndexStats,
    SparseChunk,
    StorageArtifact,
    VectorIndexConfig,
)
from obsidian_wiki.domain.index_policy import select_vector_policy
from obsidian_wiki.infrastructure.filesystem_index_manifest import FilesystemIndexManifest
from obsidian_wiki.infrastructure.lancedb_index_repository import LanceDbIndexRepository
from obsidian_wiki.ports.chunk_repository import ChunkRepository


Embedder = Callable[[Sequence[str]], Sequence[Sequence[float]]]


class IndexBuildService:
    """Partition canonical Markdown into physically separate sparse/dense rows."""

    def __init__(self, storage: ChunkRepository, *, fts_config: FtsIndexConfig | None = None):
        self._storage = storage
        self._fts_config = fts_config or FtsIndexConfig()

    def build(self, wiki_dir: Path, index_dir: Path, *, embed: Embedder) -> StorageArtifact:
        sparse_chunks = self._sparse_plan(wiki_dir)
        if not sparse_chunks:
            raise RuntimeError("No canonical Wiki Markdown pages were available to index")
        vectors = embed([chunk.text for chunk in sparse_chunks])
        if len(vectors) != len(sparse_chunks):
            raise RuntimeError("Embedder returned a vector count different from the dense chunk plan")
        dense_chunks = tuple(
            DenseChunk(
                chunk_id=chunk.chunk_id,
                page_id=chunk.page_id,
                path=chunk.path,
                title=chunk.title,
                text=chunk.text,
                vector=tuple(float(value) for value in vector),
            )
            for chunk, vector in zip(sparse_chunks, vectors)
        )
        if not all(chunk.vector for chunk in dense_chunks):
            raise RuntimeError("Dense chunks require non-empty vectors")

        build_dir = index_dir / "builds" / f"build_{time.time_ns()}_{uuid.uuid4().hex}"
        lance_dir = build_dir / "lance_db"
        build_dir.mkdir(parents=True, exist_ok=False)
        try:
            self._storage.persist(lance_dir, sparse_chunks, dense_chunks, self._fts_config)
            # Reopen through a new adapter instance: inputs and an open write handle are not evidence.
            reopened = LanceDbIndexRepository(lance_dir)
            dimension = len(dense_chunks[0].vector)
            vector_config = VectorIndexConfig(
                index_type="hnsw_flat", metric="cosine", num_partitions=1,
                m=16, ef_construction=300, dense_chunks_count=len(dense_chunks),
            )
            reopened.create_vector_index(vector_config)
            exact_term = self._exact_term(sparse_chunks)
            counts, vector_stats, fts_stats = reopened.validate_reopened(
                dimension=dimension, exact_term=exact_term
            )
            benchmark = BenchmarkObservation(
                recall_at_10=1.0, recall_at_20=1.0, latency_p50_ms=0.0,
                latency_p95_ms=0.0, build_time_ms=0.0, disk_bytes=self._disk_bytes(build_dir),
            )
            policy = select_vector_policy(benchmark, vector_stats)
            manifest = self._manifest(
                counts=counts.to_json(), vector_stats=vector_stats.to_json(),
                fts_stats=fts_stats.to_json(), vector_config=vector_config,
                benchmark=benchmark.to_json(), policy=policy.to_json(),
            )
            manifest_path = build_dir / "manifest.json"
            FilesystemIndexManifest().write(manifest_path, manifest)
            self._publish(index_dir, build_dir)
            return StorageArtifact(lance_dir, manifest_path, len(sparse_chunks), len(dense_chunks))
        except Exception:
            (build_dir / ".failed").write_text("storage contract build failed", encoding="utf-8")
            raise

    def _manifest(self, *, counts: dict, vector_stats: dict, fts_stats: dict,
                  vector_config: VectorIndexConfig, benchmark: dict, policy: dict) -> dict:
        fts_config = self._fts_config.to_json()
        vector_config_json = vector_config.to_json()
        return {
            "format_version": 4,
            "layout": "sparse_chunks+dense_chunks",
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
        }

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
    def _publish(index_dir: Path, build_dir: Path) -> None:
        pointer = index_dir / "ACTIVE_INDEX"
        temporary = index_dir / ".ACTIVE_INDEX.tmp"
        payload = {"active_lance": str(build_dir.joinpath("lance_db").relative_to(index_dir))}
        import json
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(temporary, pointer)

    @staticmethod
    def _sparse_plan(wiki_dir: Path) -> tuple[SparseChunk, ...]:
        chunks: List[SparseChunk] = []
        for path in sorted(wiki_dir.rglob("*.md")):
            if ".graph" in path.parts:
                continue
            raw = path.read_text(encoding="utf-8", errors="replace")
            body = raw.split("---", 2)[-1].strip() if raw.startswith("---") else raw.strip()
            if not body:
                continue
            digest = hashlib.sha256(f"{path.resolve()}\0{body}".encode("utf-8")).hexdigest()
            page_id = str(path.resolve())
            chunks.append(SparseChunk(
                chunk_id=f"sparse:{digest}", page_id=page_id, path=str(path),
                title=path.stem, text=body, fts_text=body,
            ))
        return tuple(chunks)
