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


# issue #47 F：index 布局版本。旧构建（被污染的两表 / 旧 schema）缺少该字段，
# ``require_current_layout`` 据此拒绝并要求重建。
INDEX_LAYOUT_VERSION = 6

# Phase 06（issue #49）：固定生产 ANN 契约的兼容版本。format-5 / benchmark
# evidence-v2 的 mode-ambiguous 构建必须被拒绝重建；新 manifest 绑定批准策略。
ANN_POLICY_SCHEMA_VERSION = 2
ANN_DECISION_EVIDENCE_SCHEMA_VERSION = 2
CANDIDATE_PUBLICATION_EVIDENCE_SCHEMA_VERSION = 3
# Phase 06 起 manifest 携带固定策略与发布证据；旧 format-5 是 mode-ambiguous。
INDEX_MANIFEST_FORMAT_VERSION = 6


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
        if self.index_type not in {"hnsw_flat", "hnsw_sq", "ivf_flat"}:
            raise ValueError(
                "Only hnsw_flat, hnsw_sq, and ivf_flat candidate index types are supported"
            )
        if self.metric not in {"cosine", "l2", "dot"}:
            raise ValueError("Vector metric must be cosine, l2, or dot")
        for field_name in ("num_partitions", "m", "ef_construction", "dense_chunks_count"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        if not self.index_name:
            raise ValueError("Vector index name must be non-empty")


@dataclass(frozen=True)
class ProductionAnnPolicy(_JsonRecord):
    """Phase 06 批准的唯一生产 ANN 契约（issue #49 / Plan 06-02 决策）。

    值来自源码控制的 ``eval/ann-policy.json``；运行时不可选择类型/ef/exact。
    """

    policy_schema_version: int
    selected_index_type: str          # "ivf-hnsw-sq"
    lancedb_index_type: str           # "hnsw_sq"
    metric: str                       # "cosine"
    dimensions: int                   # 384
    num_partitions: int               # 1
    m: int                            # 16
    ef_construction: int              # 300
    query_ef: int                     # 100
    recall_at_10_floor: float         # 0.19
    recall_at_20_floor: float         # 0.17
    comparator_sha256: str            # Plan 02 决策证据 digest
    candidate_hybrid_sha256: str
    reconciliation_sha256: str
    evidence_run_url: str
    retention_days: int               # 90

    def __post_init__(self) -> None:
        if self.policy_schema_version != ANN_POLICY_SCHEMA_VERSION:
            raise ValueError(
                f"ann policy schema version must be {ANN_POLICY_SCHEMA_VERSION}, "
                f"got {self.policy_schema_version!r}"
            )
        if self.selected_index_type != "ivf-hnsw-sq" or self.lancedb_index_type != "hnsw_sq":
            raise ValueError("the only approved production ANN type is ivf-hnsw-sq (hnsw_sq)")
        if self.metric != "cosine":
            raise ValueError("the approved production vector metric is cosine")
        for field_name in (
            "dimensions", "num_partitions", "m", "ef_construction", "query_ef",
            "retention_days",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{field_name} must be a positive integer")
        for field_name in ("recall_at_10_floor", "recall_at_20_floor"):
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be between 0.0 and 1.0")
        for field_name in (
            "comparator_sha256", "candidate_hybrid_sha256", "reconciliation_sha256",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"{field_name} must be a SHA-256 hex digest")
        if not isinstance(self.evidence_run_url, str) or not self.evidence_run_url:
            raise ValueError("evidence_run_url must be non-empty")


@dataclass(frozen=True)
class AnnDecisionEvidence(_JsonRecord):
    """唯一能授权生产 ANN 策略的锁定 A/B 决策工件摘要。

    只绑定 77,348 x 384 / 恰好 256 held-out query 的两候选决策矩阵；
    正常 staged build 绝不能把它当发布证据使用（那是 CandidatePublicationEvidence）。
    """

    evidence_schema_version: int
    corpus_rows: int
    dimensions: int
    held_out_queries: int
    candidates: Tuple[str, ...]
    ef_grid: Tuple[int, ...]
    approved_index_type: str
    approved_query_ef: int
    approved_recall_at_10_floor: float
    approved_recall_at_20_floor: float
    comparator_sha256: str
    candidate_hybrid_sha256: str
    reconciliation_sha256: str
    evidence_run_url: str
    approved_by: str
    approved_at: str


@dataclass(frozen=True)
class CandidatePublicationEvidence(_JsonRecord):
    """一次 staged candidate 的发布前质量证据。

    ``actual_dense_rows`` 是该 candidate 重开后的真实 dense 行数（与 77,348 的
    决策语料库无关）；本记录不能授权或重选策略——策略值只能来自批准记录。
    """

    evidence_schema_version: int
    actual_dense_rows: int
    dimensions: int
    metric: str
    index_type: str
    query_ef: int
    policy_sha256: str
    decision_evidence_sha256: str
    benchmark_max_probes: int
    validation_query_count: int
    query_source: str
    query_selection_sha256: str
    corpus_query_overlap: int
    exact_result_ids: Tuple[Tuple[str, ...], ...]
    candidate_result_ids: Tuple[Tuple[str, ...], ...]
    recall_at_10: float
    recall_at_20: float
    unindexed_dense_rows: int
    exact_verification_ms: float
    ann_verification_ms: float
    benchmark_duration_ms: float


@dataclass(frozen=True)
class CandidateQueryPolicy(_JsonRecord):
    """Evaluation-only ANN binding applied below hybrid orchestration.

    The policy intentionally carries only the candidate index construction type
    and the ordinary dense-query ``ef``.  It is never exposed to query planning,
    hybrid orchestration, or fusion.  （Phase 06：迁至 domain 层，infrastructure
    的 eval 绑定需要它；application 保留 re-export 兼容旧导入路径。）
    """

    candidate: str
    query_ef: int

    def __post_init__(self) -> None:
        if self.candidate not in {"ivf-hnsw-flat", "ivf-hnsw-sq"}:
            raise ValueError("candidate must be ivf-hnsw-flat or ivf-hnsw-sq")
        if (
            isinstance(self.query_ef, bool)
            or not isinstance(self.query_ef, int)
            or self.query_ef <= 0
        ):
            raise ValueError("query_ef must be a positive integer")

    def to_json(self) -> dict[str, object]:
        return {"candidate": self.candidate, "query_ef": self.query_ef}


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
    # issue #47：结构感知 provenance（原样保留真实 source span，不伪造连续切片）。
    structure_kind: str = "paragraph"   # paragraph|quote|list|code|table
    table_header_text: str = ""         # 大表窗口重复 header 的真实文本
    table_header_start_char: int = -1   # header 真实起始 offset
    table_header_end_char: int = -1     # header 真实结束 offset


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
    # A missing native observation is unsafe for staged publication.  The
    # incremental service rejects it rather than guessing that FTS is caught up.
    unindexed_rows: int | None = None


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
class ExactBatchResult(_JsonRecord):
    """Ground truth from one streamed cosine top-k scan over the dense table.

    #41: replaces the 256 independent scalar ``bypass_vector_index`` full scans
    that dominated the build-time benchmark. The application layer consumes only
    IDs plus instrumentation; NumPy/Arrow objects never cross the storage port.
    """

    result_ids: Tuple[Tuple[str, ...], ...]
    elapsed_ms: float
    scan_rows: int
    scan_batches: int
    method: str


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
    # #41：policy 必须复述 benchmark evidence 的采样口径，禁止 v4-shaped record
    # 静默携带 sampled 语义。evidence 缺失的调用方（旧两参签名）保持空默认。
    benchmark_scope: str = ""
    benchmark_probe_count: int = 0
    benchmark_probe_total: int = 0


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
