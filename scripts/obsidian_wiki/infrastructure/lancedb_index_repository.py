"""LanceDB/PyArrow implementation of the SDK-free #17 storage ports."""
from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

import lancedb
import pyarrow as pa
from lancedb.index import HnswFlat

from obsidian_wiki.domain.index_models import (
    DenseChunk,
    FtsIndexConfig,
    FtsIndexStats,
    IndexSchemaCounts,
    IndexStats,
    RebuildRequiredError,
    SparseChunk,
    VectorIndexConfig,
)


class LanceDbIndexRepository:
    """The sole location where #17 domain values become LanceDB calls."""

    def __init__(self, lance_dir: Path):
        self._lance_dir = Path(lance_dir)

    @staticmethod
    def validate_dense_chunks(dense_chunks: Sequence[DenseChunk]) -> int:
        """Reject tampered dense data before it crosses the storage boundary."""
        if not dense_chunks:
            raise ValueError("Dense persistence requires at least one row")
        ids = [chunk.chunk_id for chunk in dense_chunks]
        if len(ids) != len(set(ids)):
            raise ValueError("Dense persistence rejects duplicate chunk IDs")
        dimensions = {len(chunk.vector) for chunk in dense_chunks}
        if len(dimensions) != 1 or 0 in dimensions:
            raise ValueError("Dense vectors must have one non-zero shared dimension")
        for chunk in dense_chunks:
            if not all(math.isfinite(value) for value in chunk.vector):
                raise ValueError("Dense vectors must contain only finite values")
        return dimensions.pop()

    def persist(
        self,
        lance_dir: Path,
        sparse_chunks: Sequence[SparseChunk],
        dense_chunks: Sequence[DenseChunk],
        fts_config: FtsIndexConfig,
    ) -> None:
        if not sparse_chunks:
            raise ValueError("Sparse persistence requires at least one row")
        dimensions = self.validate_dense_chunks(dense_chunks)
        if len({chunk.chunk_id for chunk in sparse_chunks}) != len(sparse_chunks):
            raise ValueError("Sparse persistence rejects duplicate chunk IDs")
        db = lancedb.connect(str(lance_dir))
        sparse_schema = pa.schema([
            pa.field("chunk_id", pa.string()), pa.field("page_id", pa.string()),
            pa.field("path", pa.string()), pa.field("title", pa.string()),
            pa.field("text", pa.string()), pa.field(fts_config.column, pa.string()),
            pa.field("page_type", pa.string()), pa.field("section_path", pa.string()),
            pa.field("heading", pa.string()), pa.field("chunk_kind", pa.string()),
            pa.field("chunk_index", pa.int64()), pa.field("parent_section_id", pa.string()),
            pa.field("token_count", pa.int64()), pa.field("content_hash", pa.string()),
            pa.field("forced_split", pa.bool_()), pa.field("continuation_index", pa.int64()),
            pa.field("start_char", pa.int64()), pa.field("end_char", pa.int64()),
        ])
        dense_schema = pa.schema([
            pa.field("chunk_id", pa.string()), pa.field("page_id", pa.string()),
            pa.field("path", pa.string()), pa.field("title", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), list_size=dimensions)),
            pa.field("page_type", pa.string()), pa.field("section_path", pa.string()),
            pa.field("heading", pa.string()), pa.field("chunk_kind", pa.string()),
            pa.field("chunk_index", pa.int64()), pa.field("parent_section_id", pa.string()),
            pa.field("token_count", pa.int64()), pa.field("content_hash", pa.string()),
            pa.field("forced_split", pa.bool_()), pa.field("continuation_index", pa.int64()),
            pa.field("start_char", pa.int64()), pa.field("end_char", pa.int64()),
        ])
        sparse_table = db.create_table("sparse_chunks", schema=sparse_schema, mode="create")
        dense_table = db.create_table("dense_chunks", schema=dense_schema, mode="create")
        sparse_table.add(pa.Table.from_pylist([chunk.__dict__ for chunk in sparse_chunks], schema=sparse_schema))
        dense_table.add(pa.Table.from_pylist([chunk.__dict__ for chunk in dense_chunks], schema=dense_schema))
        sparse_table.create_fts_index(
            fts_config.column, replace=True, base_tokenizer=fts_config.base_tokenizer,
            lower_case=fts_config.lower_case, stem=fts_config.stem,
            remove_stop_words=fts_config.remove_stop_words,
            ascii_folding=fts_config.ascii_folding,
            max_token_length=fts_config.max_token_length,
        )
        if f"{fts_config.column}_idx" not in {index.name for index in sparse_table.list_indices()}:
            raise RuntimeError("Native FTS index was not created for sparse_chunks")

    def seal(self, lance_dir: Path) -> None:
        """#36：发布前耐久边界——逐个 fsync 全部存储文件 + 目录自底向上同步。

        LanceDB 每次 persist/reopen 都使用独立 connection（无跨调用写入 handle），
        写入完成后 connection 已随 persist 返回释放；此处显式把全部文件落到磁盘。
        任一 open/fsync 失败向上传播（不允许跳过/降级为 warning）——无法证明的
        耐久边界必须阻止发布。
        """
        root = Path(lance_dir)
        if not root.is_dir():
            raise OSError(f"seal 失败：{root} 不存在")
        for path in sorted(root.rglob("*")):
            if path.is_file():
                fd = os.open(os.fspath(path), os.O_RDWR)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
        if os.name == "posix":
            dirs = sorted(
                (p for p in root.rglob("*") if p.is_dir()),
                key=lambda p: len(p.parts), reverse=True,
            )
            for directory in [root, *dirs]:
                fd = os.open(os.fspath(directory), os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)

    def create_vector_index(self, config: VectorIndexConfig) -> IndexStats:
        table = self._dense_table()
        if table.count_rows() != config.dense_chunks_count:
            raise ValueError("Vector index config dense_chunks_count does not match dense table")
        # `VectorIndexConfig` permits only hnsw_flat; keep current SDK objects adapter-local.
        table.create_index(
            "vector",
            config=HnswFlat(
                distance_type=config.metric,
                num_partitions=config.num_partitions,
                m=config.m,
                ef_construction=config.ef_construction,
            ),
            replace=True,
            name=config.index_name,
        )
        return self.vector_index_stats(config.index_name)

    def vector_index_stats(self, index_name: str) -> IndexStats:
        stats = self._dense_table().index_stats(index_name)
        if stats is None:
            raise RuntimeError(f"Vector index statistics unavailable for {index_name}")
        return IndexStats(
            index_name=index_name,
            indexed_rows=stats.num_indexed_rows,
            unindexed_dense_rows=stats.num_unindexed_rows,
        )

    def fts_index_stats(self, index_name: str = "fts_text_idx") -> FtsIndexStats:
        stats = self._sparse_table().index_stats(index_name)
        if stats is None:
            raise RuntimeError(f"FTS index statistics unavailable for {index_name}")
        return FtsIndexStats(index_name=index_name, indexed_rows=stats.num_indexed_rows)

    def validate_reopened(self, *, dimension: int, exact_term: str) -> tuple[IndexSchemaCounts, IndexStats, FtsIndexStats]:
        """Inspect persisted data through a new connection, never cached input rows."""
        if not exact_term or not exact_term.strip() or any(char.isspace() for char in exact_term.strip()):
            raise ValueError("Exact-term validation requires one non-empty token")
        db = lancedb.connect(str(self._lance_dir))
        if set(db.table_names()) != {"sparse_chunks", "dense_chunks"}:
            raise RuntimeError("Persisted artifact must contain exactly sparse_chunks and dense_chunks")
        sparse = db.open_table("sparse_chunks")
        dense = db.open_table("dense_chunks")
        context_columns = {
            "page_type", "section_path", "heading", "chunk_kind", "chunk_index",
            "parent_section_id", "token_count", "content_hash", "forced_split",
            "continuation_index", "start_char", "end_char",
        }
        required_sparse = {"chunk_id", "page_id", "path", "title", "text", "fts_text", *context_columns}
        required_dense = {"chunk_id", "page_id", "path", "title", "text", "vector", *context_columns}
        if set(sparse.schema.names) != required_sparse or set(dense.schema.names) != required_dense:
            raise RuntimeError("Persisted table schemas do not satisfy the two-table contract")
        sparse_rows = sparse.to_arrow().to_pylist()
        dense_rows = dense.to_arrow().to_pylist()
        if not sparse_rows or not dense_rows:
            raise RuntimeError("Persisted sparse and dense tables must both contain rows")
        for rows, name in ((sparse_rows, "sparse"), (dense_rows, "dense")):
            ids = [str(row["chunk_id"]) for row in rows]
            if len(ids) != len(set(ids)):
                raise RuntimeError(f"Persisted {name} table contains duplicate chunk IDs")
        for row in dense_rows:
            vector = row["vector"]
            if len(vector) != dimension or not all(math.isfinite(float(value)) for value in vector):
                raise RuntimeError("Persisted dense vectors must be finite and fixed-dimension")
        fts_names = {index.name for index in sparse.list_indices()}
        if "fts_text_idx" not in fts_names:
            raise RuntimeError("Persisted sparse table is missing the native FTS index")
        fts_stats = self.fts_index_stats()
        if fts_stats.indexed_rows < len(sparse_rows):
            raise RuntimeError("Native FTS statistics show unindexed sparse rows")
        if not self.search_sparse(exact_term, limit=1):
            raise RuntimeError("Native FTS exact-term validation failed")
        vector_stats = self.vector_index_stats("dense_hnsw")
        return (
            IndexSchemaCounts(len(sparse_rows), len(dense_rows)),
            vector_stats,
            fts_stats,
        )

    def search_sparse(self, query: str, *, limit: int = 10) -> list[Mapping[str, object]]:
        """Route every sparse request to the native FTS index, never a fallback."""
        return self._sparse_table().search(query, query_type="fts").limit(limit).to_list()

    def context_rows(self, predicate: str) -> list[Mapping[str, object]]:
        """Read canonical chunk text/metadata from sparse_chunks for context assembly.

        This deliberately never opens the retired ``chunks`` table.  Sparse rows
        retain every ChunkRecord (including dense leaf metadata), whereas the
        dense table is intentionally limited to vector-bearing leaves.
        """
        return self._sparse_table().search().where(predicate).to_list()

    def search_dense(
        self, vector: Sequence[float], *, metric: str, limit: int = 10, where: str | None = None
    ) -> list[Mapping[str, object]]:
        return self._search_dense(vector, metric=metric, limit=limit, where=where, exact=False)

    def search_dense_exact(
        self, vector: Sequence[float], *, metric: str, limit: int = 10, where: str | None = None
    ) -> list[Mapping[str, object]]:
        return self._search_dense(vector, metric=metric, limit=limit, where=where, exact=True)

    def search_sparse_for_page(self, query: str, page_id: str, *, limit: int = 10) -> list[Mapping[str, object]]:
        return self._sparse_table().search(query, query_type="fts").where(
            self.page_predicate(page_id)
        ).limit(limit).to_list()

    @staticmethod
    def page_predicate(page_id: str) -> str:
        """Build the only page-id predicate here, escaping Lance SQL literals."""
        return "page_id = '{}'".format(page_id.replace("'", "''"))

    def _search_dense(
        self, vector: Sequence[float], *, metric: str, limit: int, where: str | None, exact: bool
    ) -> list[Mapping[str, object]]:
        if metric not in {"cosine", "l2", "dot"}:
            raise ValueError("Vector metric must be cosine, l2, or dot")
        if limit <= 0 or not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError("Dense query requires a positive limit and finite vector")
        try:
            query = self._dense_table().search(list(vector)).distance_type(metric)
            if exact:
                query = query.bypass_vector_index()
            if where is not None:
                query = query.where(where)
            return query.limit(limit).to_list()
        except Exception as exc:  # compatibility read path: warn and return no hits
            logging.getLogger(__name__).warning("dense LanceDB query failed: %s", exc)
            return []

    def _sparse_table(self):
        return lancedb.connect(str(self._lance_dir)).open_table("sparse_chunks")

    def _dense_table(self):
        return lancedb.connect(str(self._lance_dir)).open_table("dense_chunks")

    @staticmethod
    def require_current_layout(manifest_path: Path) -> None:
        try:
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RebuildRequiredError("Index manifest cannot be interpreted; rebuild required") from exc
        if manifest.get("layout") != "sparse_chunks+dense_chunks":
            raise RebuildRequiredError("Legacy index layout detected; rebuild required")
