"""Small orchestration layer for the first D-01/D-04 persisted tracer."""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Callable, List, Sequence

from obsidian_wiki.domain.index_models import (
    DenseChunk,
    FtsIndexConfig,
    SparseChunk,
    StorageArtifact,
)
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
            manifest = {
                "schema_version": 3,
                "layout": "sparse_chunks+dense_chunks",
                "sparse_count": len(sparse_chunks),
                "dense_count": len(dense_chunks),
                "fts_config": self._fts_config.to_json(),
            }
            manifest_path = build_dir / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            return StorageArtifact(lance_dir, manifest_path, len(sparse_chunks), len(dense_chunks))
        except Exception:
            (build_dir / ".failed").write_text("storage contract build failed", encoding="utf-8")
            raise

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
