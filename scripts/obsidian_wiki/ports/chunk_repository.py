"""SDK-neutral persistence and retrieval port for D-01 chunk tables."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Protocol, Sequence

from obsidian_wiki.domain.index_models import (
    DenseChunk,
    ExactBatchResult,
    FtsIndexConfig,
    FtsIndexStats,
    IndexSchemaCounts,
    IndexStats,
    SparseChunk,
    VectorIndexConfig,
)
from obsidian_wiki.domain.incremental_models import (
    CoverageObservation,
    MutationResult,
    SourceTableIdentity,
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
        where: str | None = None,
    ) -> list[Mapping[str, object]]:
        """Normal dense retrieval — always the approved ANN type at the approved ef.

        Phase 06（issue #49）：生产端口不接受运行时算法或任意 ef 选择；
        ``ef`` 由仓库绑定的策略决定（eval candidate 绑定除外，见 adapter）。
        """
        ...

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

    def source_table_identities(self) -> tuple[SourceTableIdentity, ...]: ...

    def clone_tables(self, target_lance_dir: Path) -> tuple[SourceTableIdentity, ...]: ...

    def table_rows(self, table_name: str) -> tuple[Mapping[str, object], ...]: ...

    def apply_delta(
        self, table_name: str, *, added: Sequence[Mapping[str, object]],
        updated: Sequence[Mapping[str, object]], deleted_ids: Sequence[str],
    ) -> MutationResult: ...

    def catch_up(self, fts_config: FtsIndexConfig) -> CoverageObservation: ...

    def validate_reopened(
        self, *, dimension: int, exact_term: str, vector_index_name: str,
    ) -> tuple[IndexSchemaCounts, IndexStats, FtsIndexStats]: ...

    def seal(self, lance_dir: Path) -> None:
        """#36：建立「数据已可耐久读取」的发布前边界——关闭写入 connection、
        逐个 fsync 存储文件、自底向上同步目录。无法证明关闭/提交边界时必须失败。"""
