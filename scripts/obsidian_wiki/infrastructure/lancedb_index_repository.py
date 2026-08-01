"""LanceDB/PyArrow implementation of the D-01 storage port."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import lancedb
import pyarrow as pa

from obsidian_wiki.domain.index_models import (
    DenseChunk,
    FtsIndexConfig,
    RebuildRequiredError,
    SparseChunk,
)


class LanceDbIndexRepository:
    """The sole location where D-01 domain values become LanceDB calls."""

    def __init__(self, lance_dir: Path):
        self._lance_dir = Path(lance_dir)

    def persist(
        self,
        lance_dir: Path,
        sparse_chunks: Sequence[SparseChunk],
        dense_chunks: Sequence[DenseChunk],
        fts_config: FtsIndexConfig,
    ) -> None:
        if not sparse_chunks or not dense_chunks:
            raise RuntimeError("Both D-01 physical tables require at least one row")
        dimensions = {len(chunk.vector) for chunk in dense_chunks}
        if len(dimensions) != 1 or 0 in dimensions:
            raise RuntimeError("Dense vectors must have one non-zero shared dimension")
        db = lancedb.connect(str(lance_dir))
        sparse_schema = pa.schema([
            pa.field("chunk_id", pa.string()), pa.field("page_id", pa.string()),
            pa.field("path", pa.string()), pa.field("title", pa.string()),
            pa.field("text", pa.string()), pa.field("fts_text", pa.string()),
        ])
        dense_schema = pa.schema([
            pa.field("chunk_id", pa.string()), pa.field("page_id", pa.string()),
            pa.field("path", pa.string()), pa.field("title", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), list_size=dimensions.pop())),
        ])
        sparse_table = db.create_table("sparse_chunks", schema=sparse_schema, mode="create")
        dense_table = db.create_table("dense_chunks", schema=dense_schema, mode="create")
        sparse_table.add(pa.Table.from_pylist([chunk.__dict__ for chunk in sparse_chunks], schema=sparse_schema))
        dense_table.add(pa.Table.from_pylist([chunk.__dict__ for chunk in dense_chunks], schema=dense_schema))
        sparse_table.create_fts_index(
            "fts_text", replace=True, base_tokenizer=fts_config.base_tokenizer,
            lower_case=fts_config.lower_case, stem=fts_config.stem,
            remove_stop_words=fts_config.remove_stop_words,
            ascii_folding=fts_config.ascii_folding,
            max_token_length=fts_config.max_token_length,
        )
        if "fts_text_idx" not in {index.name for index in sparse_table.list_indices()}:
            raise RuntimeError("Native FTS index was not created for sparse_chunks")

    def search_sparse(self, query: str) -> list[dict]:
        return lancedb.connect(str(self._lance_dir)).open_table("sparse_chunks").search(
            query, query_type="fts"
        ).limit(10).to_list()

    @staticmethod
    def require_current_layout(manifest_path: Path) -> None:
        try:
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RebuildRequiredError("Index manifest cannot be interpreted; rebuild required") from exc
        if manifest.get("layout") != "sparse_chunks+dense_chunks":
            raise RebuildRequiredError("Legacy index layout detected; rebuild required")
