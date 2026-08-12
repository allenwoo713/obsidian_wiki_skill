"""SDK-neutral persistence and retrieval port for D-01 chunk tables."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Protocol, Sequence

from obsidian_wiki.domain.index_models import (
    DenseChunk,
    ExactBatchResult,
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

    def context_rows(self, predicate: str) -> list[Mapping[str, object]]: ...

    def search_dense(
        self, vector: Sequence[float], *, metric: str, limit: int = 10,
        where: str | None = None, ef: int | None = None,
    ) -> list[Mapping[str, object]]: ...

    def search_dense_exact(
        self, vector: Sequence[float], *, metric: str, limit: int = 10, where: str | None = None
    ) -> list[Mapping[str, object]]: ...

    def search_dense_exact_batch(
        self,
        vectors: Sequence[Sequence[float]],
        *,
        metric: str,
        limit: int = 20,
        row_batch_size: int = 8192,
        query_batch_size: int = 32,
    ) -> ExactBatchResult: ...

    def create_vector_index(self, config: VectorIndexConfig) -> IndexStats: ...

    def vector_index_stats(self, index_name: str) -> IndexStats: ...

    def fts_index_stats(self, index_name: str = "fts_text_idx") -> FtsIndexStats: ...

    def seal(self, lance_dir: Path) -> None:
        """#36：建立「数据已可耐久读取」的发布前边界——关闭写入 connection、
        逐个 fsync 存储文件、自底向上同步目录。无法证明关闭/提交边界时必须失败。"""
