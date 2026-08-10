"""Adapter correctness for #41 streamed batch exact verification.

These tests do NOT re-assert the application contract (that is
``test_benchmark_sampling.py``); they pin the NumPy streaming top-k against the
LanceDB scalar bypass on a real local table, so a numerical or batch-boundary
regression is caught independently of the benchmark harness.
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import lancedb
import pytest

from obsidian_wiki.infrastructure.lancedb_index_repository import LanceDbIndexRepository


def _table(vectors: np.ndarray) -> pa.Table:
    dimensions = int(vectors.shape[1])
    column = pa.FixedSizeListArray.from_arrays(
        pa.array(vectors.reshape(-1), type=pa.float32()), dimensions
    )
    return pa.table({
        "chunk_id": pa.array([f"c{i:05d}" for i in range(len(vectors))]),
        "vector": column,
    })


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, np.finfo(np.float32).eps)


def _scalar_topk(repo: LanceDbIndexRepository, vector, *, limit: int = 20) -> set[str]:
    rows = repo.search_dense_exact(np.asarray(vector).tolist(), metric="cosine", limit=limit)
    return {str(row["chunk_id"]) for row in rows}


@pytest.fixture
def table_repo(tmp_path):
    rng = np.random.default_rng(41)
    vectors = _normalize(rng.standard_normal((257, 32)).astype(np.float32))
    db = lancedb.connect(str(tmp_path))
    db.create_table("dense_chunks", data=_table(vectors))
    return LanceDbIndexRepository(tmp_path), vectors


def test_batch_exact_matches_scalar_bypass(table_repo) -> None:
    repo, vectors = table_repo
    probes = [vectors[index] for index in (0, 17, 64, 128, 256)]
    batch = repo.search_dense_exact_batch(
        probes, metric="cosine", limit=20, row_batch_size=64, query_batch_size=2
    )
    assert batch.scan_rows == 257
    assert batch.scan_batches > 0
    assert batch.method == "streamed_numpy_cosine_v1"
    assert len(batch.result_ids) == len(probes)
    for probe, batch_ids in zip(probes, batch.result_ids, strict=True):
        assert set(batch_ids) == _scalar_topk(repo, probe, limit=20)


@pytest.mark.parametrize("rows", [63, 64, 65, 127, 128, 129])
def test_batch_exact_row_block_boundaries(tmp_path, rows) -> None:
    rng = np.random.default_rng(rows)
    vectors = _normalize(rng.standard_normal((rows, 32)).astype(np.float32))
    db = lancedb.connect(str(tmp_path))
    db.create_table("dense_chunks", data=_table(vectors))
    repo = LanceDbIndexRepository(tmp_path)
    probe = vectors[0]
    batch = repo.search_dense_exact_batch(
        [probe], metric="cosine", limit=20, row_batch_size=64, query_batch_size=1
    )
    assert set(batch.result_ids[0]) == _scalar_topk(repo, probe, limit=20)


@pytest.mark.parametrize("queries", [1, 31, 32, 33, 65])
def test_batch_exact_query_block_boundaries(tmp_path, queries) -> None:
    rng = np.random.default_rng(7)
    vectors = _normalize(rng.standard_normal((257, 32)).astype(np.float32))
    db = lancedb.connect(str(tmp_path))
    db.create_table("dense_chunks", data=_table(vectors))
    repo = LanceDbIndexRepository(tmp_path)
    probes = [vectors[index] for index in range(queries)]
    batch = repo.search_dense_exact_batch(
        probes, metric="cosine", limit=20, row_batch_size=8192, query_batch_size=32
    )
    assert len(batch.result_ids) == queries
    for probe, ids in zip(probes, batch.result_ids, strict=True):
        assert set(ids) == _scalar_topk(repo, probe, limit=20)


def test_batch_exact_rejects_zero_norm_query(tmp_path) -> None:
    rng = np.random.default_rng(3)
    vectors = _normalize(rng.standard_normal((64, 32)).astype(np.float32))
    db = lancedb.connect(str(tmp_path))
    db.create_table("dense_chunks", data=_table(vectors))
    repo = LanceDbIndexRepository(tmp_path)
    zero = np.zeros(32, dtype=np.float32)
    with pytest.raises(ValueError, match="zero-norm"):
        repo.search_dense_exact_batch([zero], metric="cosine", limit=20)


def test_batch_exact_rejects_unsupported_metric(tmp_path) -> None:
    rng = np.random.default_rng(5)
    vectors = _normalize(rng.standard_normal((64, 32)).astype(np.float32))
    db = lancedb.connect(str(tmp_path))
    db.create_table("dense_chunks", data=_table(vectors))
    repo = LanceDbIndexRepository(tmp_path)
    with pytest.raises(ValueError, match="cosine only"):
        repo.search_dense_exact_batch([vectors[0]], metric="dot", limit=20)
