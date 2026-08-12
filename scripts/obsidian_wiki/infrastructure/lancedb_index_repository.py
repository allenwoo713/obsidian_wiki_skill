"""LanceDB/PyArrow implementation of the SDK-free #17 storage ports."""
from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Mapping, Sequence

import lancedb
import numpy as np
import pyarrow as pa
from lancedb.index import HnswFlat, HnswSq, IvfFlat

from obsidian_wiki.domain.index_models import (
    DenseChunk,
    ExactBatchResult,
    FtsIndexConfig,
    FtsIndexStats,
    INDEX_LAYOUT_VERSION,
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
        # issue #47：两表严格语义分离，写入前 fail-closed 拒绝混入的另一种 kind。
        if any(c.chunk_kind != "sparse" for c in sparse_chunks):
            raise ValueError("sparse_chunks accepts only chunk_kind='sparse' (issue #47)")
        if any(c.chunk_kind != "dense" for c in dense_chunks):
            raise ValueError("dense_chunks accepts only chunk_kind='dense' (issue #47)")
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
            pa.field("structure_kind", pa.string()),
            pa.field("table_header_text", pa.string()),
            pa.field("table_header_start_char", pa.int64()),
            pa.field("table_header_end_char", pa.int64()),
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
        if config.index_type == "ivf_flat":
            index_config = IvfFlat(
                distance_type=config.metric,
                num_partitions=config.num_partitions,
            )
        elif config.index_type == "hnsw_sq":
            index_config = HnswSq(
                distance_type=config.metric,
                num_partitions=config.num_partitions,
                m=config.m,
                ef_construction=config.ef_construction,
            )
        else:
            index_config = HnswFlat(
                distance_type=config.metric,
                num_partitions=config.num_partitions,
                m=config.m,
                ef_construction=config.ef_construction,
            )
        table.create_index(
            "vector",
            config=index_config,
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

    def validate_reopened(
        self, *, dimension: int, exact_term: str, vector_index_name: str | None = "dense_hnsw"
    ) -> tuple[IndexSchemaCounts, IndexStats, FtsIndexStats]:
        """Inspect persisted data through a new connection, never cached input rows."""
        if not exact_term or not exact_term.strip() or any(char.isspace() for char in exact_term.strip()):
            raise ValueError("Exact-term validation requires one non-empty token")
        db = lancedb.connect(str(self._lance_dir))
        if set(db.table_names()) != {"sparse_chunks", "dense_chunks"}:
            raise RuntimeError("Persisted artifact must contain exactly sparse_chunks and dense_chunks")
        sparse = db.open_table("sparse_chunks")
        dense = db.open_table("dense_chunks")
        # 两表共享的上下文列；structure_kind/table_header_* 为 sparse 表独有
        # （dense 表 schema 不含），仅计入 required_sparse 校验（issue #47）。
        context_columns = {
            "page_type", "section_path", "heading", "chunk_kind", "chunk_index",
            "parent_section_id", "token_count", "content_hash", "forced_split",
            "continuation_index", "start_char", "end_char",
        }
        sparse_only = {
            "structure_kind", "table_header_text",
            "table_header_start_char", "table_header_end_char",
        }
        required_sparse = {"chunk_id", "page_id", "path", "title", "text", "fts_text",
                          *context_columns, *sparse_only}
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
        vector_stats = (
            self.vector_index_stats(vector_index_name)
            if vector_index_name is not None
            else IndexStats(
                index_name="exact_scan",
                indexed_rows=len(dense_rows),
                unindexed_dense_rows=0,
            )
        )
        return (
            IndexSchemaCounts(len(sparse_rows), len(dense_rows)),
            vector_stats,
            fts_stats,
        )

    def search_sparse(self, query: str, *, limit: int = 10) -> list[Mapping[str, object]]:
        """Route every sparse request to the native FTS index, never a fallback."""
        return self._sparse_table().search(query, query_type="fts").limit(limit).to_list()

    def context_rows(self, predicate: str) -> list[Mapping[str, object]]:
        """Union sparse + dense retrieval rows for context assembly (issue #47).

        The two physical tables are strictly separated; context assembly reads
        both and deduplicates by ``chunk_id`` (dense wins on collision, since it
        carries the vector-bearing leaf).  Sorted by
        (page_id, chunk_index, chunk_id) for deterministic ordering.
        """
        def _normalize(row: Mapping[str, object]) -> dict:
            return {**row}

        sparse = self._sparse_table().search().where(predicate).to_list()
        dense = self._dense_table().search().where(predicate).to_list()
        rows = {row["chunk_id"]: _normalize(row) for row in sparse}
        rows.update({row["chunk_id"]: _normalize(row) for row in dense})
        return sorted(rows.values(), key=lambda r: (r["page_id"], r["chunk_index"], r["chunk_id"]))

    def search_dense(
        self, vector: Sequence[float], *, metric: str, limit: int = 10,
        where: str | None = None, ef: int | None = None,
    ) -> list[Mapping[str, object]]:
        return self._search_dense(
            vector, metric=metric, limit=limit, where=where, exact=False, ef=ef
        )

    def search_dense_exact(
        self, vector: Sequence[float], *, metric: str, limit: int = 10, where: str | None = None
    ) -> list[Mapping[str, object]]:
        return self._search_dense(
            vector, metric=metric, limit=limit, where=where, exact=True, ef=None
        )

    def search_dense_exact_batch(
        self,
        vectors: Sequence[Sequence[float]],
        *,
        metric: str,
        limit: int = 20,
        row_batch_size: int = 8192,
        query_batch_size: int = 32,
    ) -> ExactBatchResult:
        """#41: one streamed cosine top-k scan over the dense table for many probes.

        Replaces the 256 independent scalar ``bypass_vector_index`` full scans that
        dominated the build-time benchmark. The dense table is read exactly once via
        ``to_arrow`` (``to_lance`` needs the external ``lance`` package, absent from CI
        deps); rows stream in ``row_batch_size`` batches and scores are materialized
        only as ``query_batch_size x row_batch_size`` blocks so memory stays bounded
        regardless of corpus size. Ground-truth failures propagate and never silently
        fall back to a slower scalar path.
        """
        if metric != "cosine":
            raise ValueError("Batch exact verification currently supports cosine only")
        if not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0:
            raise ValueError("limit must be a positive integer")
        if row_batch_size <= 0 or query_batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        if not vectors:
            raise ValueError("Batch exact verification requires at least one query")

        queries = np.asarray(
            [tuple(float(value) for value in vector) for vector in vectors],
            dtype=np.float32,
        )
        if queries.ndim != 2 or queries.shape[1] == 0:
            raise ValueError("Batch exact queries must share one non-zero dimension")
        if not np.isfinite(queries).all():
            raise ValueError("Batch exact queries must contain only finite values")
        eps = np.finfo(np.float32).eps
        query_norms = np.linalg.norm(queries, axis=1, keepdims=True)
        if np.any(query_norms <= eps):
            raise ValueError("Cosine exact verification rejects zero-norm queries")
        queries = queries / query_norms

        # ponytail: read the dense table once; process it in row-batch slices so the
        # score block stays bounded at query_batch_size x row_batch_size. (RecordBatch
        # column .values returns the whole underlying buffer, not the batch's logical
        # rows, so we slice the materialized matrix by index instead of streaming.)
        table = self._dense_table()
        arrow = table.to_arrow().select(["chunk_id", "vector"])
        if arrow.num_rows <= 0:
            raise RuntimeError("Batch exact verification scanned no dense rows")
        vector_column = arrow.column("vector")
        if hasattr(vector_column, "combine_chunks"):
            vector_column = vector_column.combine_chunks()
        rows = np.asarray(
            vector_column.values.to_numpy(zero_copy_only=False),
            dtype=np.float32,
        ).reshape(arrow.num_rows, -1)
        if rows.shape[1] != queries.shape[1]:
            raise RuntimeError("Dense table/query dimensions do not match")
        if not np.isfinite(rows).all():
            raise RuntimeError("Dense table contains non-finite vectors")
        row_norms = np.linalg.norm(rows, axis=1, keepdims=True)
        if np.any(row_norms <= eps):
            raise RuntimeError("Cosine exact verification rejects zero-norm stored vectors")
        rows = rows / row_norms
        row_ids = np.asarray(arrow.column("chunk_id").to_pylist(), dtype=object)

        best_ids: list[np.ndarray] = [np.empty(0, dtype=object) for _ in range(len(queries))]
        best_scores: list[np.ndarray] = [np.empty(0, dtype=np.float32) for _ in range(len(queries))]

        started = time.perf_counter()
        scan_rows = arrow.num_rows
        scan_batches = 0
        for start in range(0, arrow.num_rows, row_batch_size):
            end = min(start + row_batch_size, arrow.num_rows)
            batch_rows = rows[start:end]
            batch_ids = row_ids[start:end]
            scan_batches += 1
            for q0 in range(0, len(queries), query_batch_size):
                q1 = min(q0 + query_batch_size, len(queries))
                scores = queries[q0:q1] @ batch_rows.T
                for offset in range(q1 - q0):
                    query_index = q0 + offset
                    scores_row = scores[offset]
                    local_k = min(limit, scores_row.shape[0])
                    kth = scores_row.shape[0] - local_k
                    positions = np.argpartition(scores_row, kth)[kth:]
                    local_scores = scores_row[positions]
                    candidate_scores = np.concatenate((best_scores[query_index], local_scores))
                    candidate_ids = np.concatenate((best_ids[query_index], batch_ids[positions]))
                    # ponytail: ties are broken by chunk_id, which is deterministic and
                    # reproducible across runners but NOT the order lancedb's own scan
                    # returns. Ceiling: on a corpus with exact score ties straddling the
                    # k-th rank, recall@k is computed on ordered prefixes and will
                    # under-report (fail-safe: ANN promotion is refused, never granted).
                    # Real float32 embeddings do not tie; upgrade path if they ever do is
                    # score-aware recall (count a hit when score >= the k-th exact score).
                    order = np.lexsort((candidate_ids.astype(str), -candidate_scores))[:limit]
                    best_scores[query_index] = candidate_scores[order]
                    best_ids[query_index] = candidate_ids[order]

        if scan_rows <= 0:
            raise RuntimeError("Batch exact verification scanned no dense rows")
        elapsed_ms = (time.perf_counter() - started) * 1000

        return ExactBatchResult(
            result_ids=tuple(
                tuple(str(chunk_id) for chunk_id in query_ids)
                for query_ids in best_ids
            ),
            elapsed_ms=elapsed_ms,
            scan_rows=scan_rows,
            scan_batches=scan_batches,
            method="streamed_numpy_cosine_v1",
        )

    def search_sparse_for_page(self, query: str, page_id: str, *, limit: int = 10) -> list[Mapping[str, object]]:
        return self._sparse_table().search(query, query_type="fts").where(
            self.page_predicate(page_id)
        ).limit(limit).to_list()

    @staticmethod
    def page_predicate(page_id: str) -> str:
        """Build the only page-id predicate here, escaping Lance SQL literals."""
        return "page_id = '{}'".format(page_id.replace("'", "''"))

    def _search_dense(
        self, vector: Sequence[float], *, metric: str, limit: int,
        where: str | None, exact: bool, ef: int | None,
    ) -> list[Mapping[str, object]]:
        if metric not in {"cosine", "l2", "dot"}:
            raise ValueError("Vector metric must be cosine, l2, or dot")
        if limit <= 0 or not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError("Dense query requires a positive limit and finite vector")
        try:
            query = self._dense_table().search(list(vector)).distance_type(metric)
            if exact:
                query = query.bypass_vector_index()
            else:
                # #41: lancedb 0.34 HNSW default ef (=1.5*limit) is too low for
                # build-time self-probe recall (recall@10/20 == 1.0 gate). Floor
                # ef at 100; but never go BELOW lancedb's own default (1.5*limit),
                # since large-k production queries (e.g. limit=80) would otherwise
                # regress from ef=120 to 100. Floor = max(100, ceil(1.5*limit)).
                query = query.ef(ef if ef is not None else max(100, (limit * 3 + 1) // 2))
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
        # issue #47：旧构建（被污染的两表 / 旧 schema）缺少 index_layout_version，
        # 据此明确要求重建，而非在污染的稀疏表上继续服务。
        if manifest.get("index_layout_version") != INDEX_LAYOUT_VERSION:
            raise RebuildRequiredError("Index layout version mismatch; rebuild required")
