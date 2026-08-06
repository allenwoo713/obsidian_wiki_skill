"""Immutable, SDK-free values for the #17 storage/index contract."""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
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
    # The sparse table is also the SDK-free context-read source.  Keep the
    # ChunkRecord fields needed by the compatibility facade alongside its FTS
    # payload; vector data remains exclusively in DenseChunk.
    page_type: str = "concept"
    section_path: str = "[]"
    heading: str = ""
    chunk_kind: str = "dense"
    chunk_index: int = 0
    parent_section_id: str = ""
    token_count: int = 0
    content_hash: str = ""
    forced_split: bool = False
    continuation_index: int = -1
    start_char: int = 0
    end_char: int = 0


@dataclass(frozen=True)
class DenseChunk(_JsonRecord):
    chunk_id: str
    page_id: str
    path: str
    title: str
    text: str
    vector: Tuple[float, ...]
    page_type: str = "concept"
    section_path: str = "[]"
    heading: str = ""
    chunk_kind: str = "dense"
    chunk_index: int = 0
    parent_section_id: str = ""
    token_count: int = 0
    content_hash: str = ""
    forced_split: bool = False
    continuation_index: int = -1
    start_char: int = 0
    end_char: int = 0


@dataclass(frozen=True)
class StorageArtifact(_JsonRecord):
    """构建产物：路径 + count + 身份（#34：build_id/generation 贯穿 manifest/pointer/record）。"""

    lance_dir: Path
    manifest_path: Path
    sparse_count: int
    dense_count: int
    build_id: str
    generation: int


@dataclass(frozen=True)
class BuildContext:
    """不可变构建身份（#34）：最外层 facade 创建一次，贯穿 lock metadata、
    build 目录、manifest、ACTIVE_INDEX pointer、日志与返回 artifact。
    build_id 必须是 UTC 微秒时间戳 + 完整随机 UUID（原 #21 约定）。
    """

    build_id: str
    started_at: str
    owner_nonce: str


class PostCommitStatus(str, Enum):
    """#37 提交点之后的衍生工作状态：report 失效是可观察、可重试的 post-commit 任务。"""

    COMPLETE = "complete"
    COMMUNITY_REPORT_INVALIDATION_PENDING = "community_report_invalidation_pending"


class PostCommitTaskState(str, Enum):
    PREPARED = "prepared"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class PostCommitTask(_JsonRecord):
    """#37 提交点之前的 durable intent：post-commit 工作必须先 prepare 再执行，
    进程在任何一步退出都不会永久丢失任务。"""

    task_id: str
    task_type: str
    build_id: str
    generation: int
    state: PostCommitTaskState
    prepared_at: str
    completed_at: str | None = None


@dataclass(frozen=True)
class IndexBuildOutcome:
    """#37 构建对外结果：区分「索引提交结果」与「提交后任务状态」。

    ``published=True`` 表示 ACTIVE_INDEX 已耐久推进（commit point）；
    其后的 report 失效失败只体现在 ``post_commit_status`` 与 ``warnings``，
    绝不把已发布的 build 伪装成失败。
    """

    artifact: StorageArtifact
    build_id: str
    generation: int
    published: bool
    post_commit_status: PostCommitStatus
    warnings: Tuple[str, ...] = ()


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
