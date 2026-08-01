"""SDK-neutral persistence and retrieval port for D-01 chunk tables."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Protocol, Sequence

from obsidian_wiki.domain.index_models import (
    DenseChunk,
    FtsIndexConfig,
    FtsIndexStats,
    IndexStats,
    SparseChunk,
    VectorIndexConfig,
)


class ChunkRepository(Protocol):
    """Owns separate sparse/dense storage without exposing SDK table objects."""

    def persist(
        self,
        lance_dir: Path,
        sparse_chunks: Sequence[SparseChunk],
        dense_chunks: Sequence[DenseChunk],
        fts_config: FtsIndexConfig,
    ) -> None: ...

    def search_sparse(self, query: str, *, limit: int = 10) -> list[Mapping[str, object]]: ...

    def search_dense(
        self, vector: Sequence[float], *, metric: str, limit: int = 10, where: str | None = None
    ) -> list[Mapping[str, object]]: ...

    def search_dense_exact(
        self, vector: Sequence[float], *, metric: str, limit: int = 10, where: str | None = None
    ) -> list[Mapping[str, object]]: ...

    def create_vector_index(self, config: VectorIndexConfig) -> IndexStats: ...

    def vector_index_stats(self, index_name: str) -> IndexStats: ...

    def fts_index_stats(self, index_name: str = "fts_text_idx") -> FtsIndexStats: ...
