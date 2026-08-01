"""Immutable, SDK-free values for the #17 storage/index contract."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Tuple


class RebuildRequiredError(RuntimeError):
    """Raised when a persisted index belongs to the retired ``chunks`` layout."""


class _JsonRecord:
    """Provide a stable, JSON-safe representation for persisted domain records."""

    def to_json(self) -> dict[str, Any]:
        return json.loads(json.dumps(asdict(self), default=str, ensure_ascii=False))


@dataclass(frozen=True)
class FtsIndexConfig(_JsonRecord):
    """The single immutable D-04 native-FTS configuration."""

    column: str = "fts_text"
    base_tokenizer: str = "whitespace"
    lower_case: bool = False
    stem: bool = False
    remove_stop_words: bool = False
    ascii_folding: bool = False
    max_token_length: int = 256

    def __post_init__(self) -> None:
        if not self.column or not self.base_tokenizer:
            raise ValueError("FTS column and tokenizer must be non-empty")
        if not isinstance(self.max_token_length, int) or self.max_token_length <= 0:
            raise ValueError("FTS max_token_length must be a positive integer")


@dataclass(frozen=True)
class VectorIndexConfig(_JsonRecord):
    """Candidate ANN configuration; dense population is its sole sizing input."""

    index_type: str
    metric: str
    num_partitions: int
    m: int
    ef_construction: int
    dense_chunks_count: int
    index_name: str = "dense_hnsw"

    def __post_init__(self) -> None:
        if self.index_type != "hnsw_flat":
            raise ValueError("Only the hnsw_flat candidate index type is supported")
        if self.metric not in {"cosine", "l2", "dot"}:
            raise ValueError("Vector metric must be cosine, l2, or dot")
        for field_name in ("num_partitions", "m", "ef_construction", "dense_chunks_count"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if not self.index_name:
            raise ValueError("Vector index name must be non-empty")


@dataclass(frozen=True)
class SparseChunk(_JsonRecord):
    chunk_id: str
    page_id: str
    path: str
    title: str
    text: str
    fts_text: str


@dataclass(frozen=True)
class DenseChunk(_JsonRecord):
    chunk_id: str
    page_id: str
    path: str
    title: str
    text: str
    vector: Tuple[float, ...]


@dataclass(frozen=True)
class StorageArtifact(_JsonRecord):
    lance_dir: Path
    manifest_path: Path
    sparse_count: int
    dense_count: int


@dataclass(frozen=True)
class IndexSchemaCounts(_JsonRecord):
    """Persisted sparse/dense row counts observed after an adapter write."""

    sparse_chunks_count: int
    dense_chunks_count: int


@dataclass(frozen=True)
class IndexStats(_JsonRecord):
    """Vector index coverage observed from the storage adapter."""

    index_name: str
    indexed_rows: int
    unindexed_dense_rows: int


@dataclass(frozen=True)
class FtsIndexStats(_JsonRecord):
    """Native FTS index observation returned by the sparse-table adapter."""

    index_name: str
    indexed_rows: int


@dataclass(frozen=True)
class BenchmarkObservation(_JsonRecord):
    """Exact-versus-candidate ANN measurements; timings are reporting-only."""

    recall_at_10: float
    recall_at_20: float
    latency_p50_ms: float
    latency_p95_ms: float
    build_time_ms: float
    disk_bytes: int


@dataclass(frozen=True)
class ValidationObservation(_JsonRecord):
    schema_counts: IndexSchemaCounts
    vector_index: IndexStats
    fts_index: FtsIndexStats
    exact_term_validated: bool


@dataclass(frozen=True)
class SdkVersions(_JsonRecord):
    lancedb: str
    pyarrow: str
    sentence_transformers: str


@dataclass(frozen=True)
class VectorPolicyDecision(_JsonRecord):
    selected_mode: str
    reason: str
    benchmark: BenchmarkObservation
    index_stats: IndexStats


@dataclass(frozen=True)
class IndexManifest(_JsonRecord):
    """Stable persisted record of the #17 storage and policy outcome."""

    layout: str
    fts_config: FtsIndexConfig
    vector_config: VectorIndexConfig
    validation: ValidationObservation
    benchmark: BenchmarkObservation
    policy: VectorPolicyDecision
    sdk_versions: SdkVersions
    metadata: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.layout != "sparse_chunks+dense_chunks":
            raise ValueError("Index manifests must use the D-01 two-table layout")
