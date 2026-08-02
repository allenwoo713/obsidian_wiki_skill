"""Storage port for the D-01 two-table artifact; no SDK types cross this seam."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, Sequence

from obsidian_wiki.domain.index_models import DenseChunk, FtsIndexConfig, SparseChunk


class IndexStorage(Protocol):
    def persist(
        self,
        lance_dir: Path,
        sparse_chunks: Sequence[SparseChunk],
        dense_chunks: Sequence[DenseChunk],
        fts_config: FtsIndexConfig,
    ) -> None: ...
