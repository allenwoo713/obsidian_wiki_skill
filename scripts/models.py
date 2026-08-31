"""共享数据结构，所有脚本统一引用。"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set


@dataclass
class ImageRef:
    """提取的图片引用。"""
    filename: str          # acme-visioncam-front-datasheet-v1-6_img03.png
    rel_path: str          # assets/acme-visioncam-front-datasheet-v1-6_img03.png
    caption: str           # 图注全文（"图3 方位角..."），可能为空
    source_media_name: str # image3.png（docx）/ xref=17（pdf），溯源用
    sha256: str            # 图片内容哈希，去重用
    page_or_section: str   # "page 3" / "body"，定位用


@dataclass
class ParsedDoc:
    """解析后的源文档。"""
    path: Path
    title: str
    text: str
    tables: List[List[List[str]]]  # [table][row][cell]
    sha256: str
    doc_type: str  # 'docx' | 'pdf' | 'md' | 'txt'
    images: List[ImageRef] = field(default_factory=list)


@dataclass
class WikiPage:
    """解析后的 wiki 页面（供索引/图谱消费）。"""
    path: Path
    title: str
    page_type: str  # 'product' | 'specs' | 'installation' | 'calibration' | 'diagnostics' | 'interface' | 'source-summary' | 'comparison' | 'concept'
    content: str
    sources: List[str]
    links: List[str]  # [[wikilink]] 目标（无方括号）
    sha256: str
    aliases: List[str] = field(default_factory=list)


@dataclass
class RetrievedPage:
    """检索结果。

    ISSUE-15：score 与 distance 分离。
    - score: 0~1 展示相似度（转换后），便于人类理解与跨 metric 比较。
    - distance: LanceDB 返回的原始距离（仅 vector 检索有值），便于调试。
    - vector_metric: 本次向量检索使用的 metric（仅 vector/fused 有值）。
    """
    path: Path
    title: str
    score: float
    snippet: str
    sources: List[str]
    retrieval_method: str  # 'bm25' | 'vector' | 'graph' | 'fused'
    distance: Optional[float] = None       # 原始距离（仅 vector 检索）
    vector_metric: Optional[str] = None     # 向量 metric 名称（仅 vector/fused）


@dataclass
class ManifestEntry:
    """manifest.json 单条记录。"""
    path: str
    sha256: str
    mtime: float
    status: str  # 'new' | 'processed' | 'modified' | 'deleted'
    wiki_pages: List[str]
    last_processed: Optional[str]  # ISO datetime


# --------------------------------------------------------------------------
# Retrieval v2 shared data model (GitHub issues #1/#3/#4/#5)
# These define the cross-module contract consumed by chunking, fusion,
# context packing and graph expansion.
# --------------------------------------------------------------------------
IMAGE_PAGE_TYPE = "image_caption"
_IMAGE_FILE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp")


def is_image_page(page_type: Optional[str], path: object = "",
                  page_id: str = "") -> bool:
    """Return whether a retrieval object represents an image-caption page.

    Persisted ``page_type`` is authoritative. Path/suffix detection exists only
    for backward compatibility with hand-built test objects or legacy callers
    whose page type is absent/unknown.
    """
    normalized_type = (page_type or "").strip().lower()
    if normalized_type and normalized_type != "unknown":
        return normalized_type == IMAGE_PAGE_TYPE

    normalized_path = str(path or "").replace("\\", "/").lower()
    normalized_id = str(page_id or "").replace("\\", "/").lower()
    return (
        "/assets/" in f"/{normalized_path.lstrip('/')}"
        or normalized_path.endswith(_IMAGE_FILE_SUFFIXES)
        or normalized_id.endswith(_IMAGE_FILE_SUFFIXES)
    )


@dataclass
class GraphPath:
    """A relation path from a seed page to a graph candidate."""
    source_id: str
    target_id: str
    edge_type: str          # explicit: wikilink|derived_from_source|same_source
                            # inferred: adamic_adar|type_affinity
    is_inferred: bool
    weight: float
    hop: int
    edge_signals: List[str] = field(default_factory=list)


@dataclass
class EvidenceHit:
    """A single chunk hit on one retrieval channel."""
    chunk_id: str
    channel: str            # 'sparse' | 'dense'
    rank: int
    raw_score: float
    text: str
    section_path: List[str]


@dataclass
class PageCandidate:
    """A fused page candidate carrying evidence from both channels."""
    page_id: str
    path: Path
    title: str
    rrf_score: float
    sparse_rank: Optional[int]
    dense_rank: Optional[int]
    sparse_evidence: List[EvidenceHit] = field(default_factory=list)
    dense_evidence: List[EvidenceHit] = field(default_factory=list)
    graph_paths: List[GraphPath] = field(default_factory=list)
    page_type: str = "unknown"


@dataclass
class ContextItem:
    """A page/source included in the final LLM context bundle."""
    page_id: str
    path: str
    title: str
    inclusion_reason: str
    scope: str              # chunk|adjacent|section|full_page|full_source
    evidence: List[EvidenceHit] = field(default_factory=list)
    text: str = ""
    sources: List[str] = field(default_factory=list)
    graph_paths: List[GraphPath] = field(default_factory=list)
    token_count: int = 0
    truncated: bool = False
    truncation_reason: Optional[str] = None
    omitted_ranges: List[dict] = field(default_factory=list)
    page_type: str = "unknown"


@dataclass
class ContextBundle:
    """Final retrieval output — directly consumable by the LLM.

    Budget contract (formalised as data, not as a rule duplicated by callers):

    - ``requested_base_budget_tokens`` — the base budget the caller asked for
      (``query.py --max-tokens``). It is an *input*, not a cap.
    - ``budget_multiplier`` / ``budget_policy`` — how the intent-driven budget
      policy expanded (or kept) that base budget.
    - ``hard_max_tokens`` — optional absolute ceiling supplied by the caller
      (model window / cost limit). ``None`` means "no hard ceiling".
    - ``effective_budget_tokens`` — the budget actually applied this run::

          effective = min(base × multiplier, hard_max_tokens if supplied)

    - ``max_context_tokens`` — kept for backwards compatibility; its meaning is
      fixed as "the cap actually in force", i.e. it always mirrors
      ``effective_budget_tokens``.

    Invariants (see :meth:`budget_contract_violations`)::

        max_context_tokens == effective_budget_tokens
        token_count        <= effective_budget_tokens
        effective_budget_tokens <= hard_max_tokens   # when hard cap supplied
    """
    query: str
    mode: str
    items: List[ContextItem] = field(default_factory=list)
    context_text: str = ""
    token_count: int = 0
    max_context_tokens: int = 0
    omitted_items: List[dict] = field(default_factory=list)
    # --- budget contract (issue #14 follow-up) ---
    requested_base_budget_tokens: int = 0
    effective_budget_tokens: int = 0
    budget_multiplier: float = 1.0
    budget_policy: str = "context_mode_multiplier_v1"
    hard_max_tokens: Optional[int] = None

    def __post_init__(self):
        # Legacy construction sites only pass ``max_context_tokens``; treat that
        # value as the effective budget so the invariant holds by construction.
        if not self.effective_budget_tokens:
            self.effective_budget_tokens = self.max_context_tokens
        if not self.requested_base_budget_tokens:
            self.requested_base_budget_tokens = self.effective_budget_tokens

    def apply_budget(self, *, base_tokens: int, multiplier: float,
                     effective_tokens: int, hard_max_tokens: Optional[int] = None,
                     policy: str = "context_mode_multiplier_v1") -> "ContextBundle":
        """Stamp the resolved budget onto the bundle (single writer: query.py)."""
        self.requested_base_budget_tokens = int(base_tokens)
        self.budget_multiplier = float(multiplier)
        self.effective_budget_tokens = int(effective_tokens)
        self.max_context_tokens = int(effective_tokens)
        self.hard_max_tokens = hard_max_tokens
        self.budget_policy = policy
        return self

    def budget_contract_violations(self) -> List[str]:
        """Return human-readable contract breaches (empty list == healthy).

        Consumers (eval / hosts) should call this instead of re-deriving the
        budget rules themselves.
        """
        bad: List[str] = []
        if self.max_context_tokens != self.effective_budget_tokens:
            bad.append(f"max_context_tokens={self.max_context_tokens} != "
                       f"effective_budget_tokens={self.effective_budget_tokens}")
        if self.token_count > self.effective_budget_tokens:
            bad.append(f"token_count={self.token_count} > "
                       f"effective_budget_tokens={self.effective_budget_tokens}")
        if self.hard_max_tokens is not None and self.effective_budget_tokens > self.hard_max_tokens:
            bad.append(f"effective_budget_tokens={self.effective_budget_tokens} > "
                       f"hard_max_tokens={self.hard_max_tokens}")
        return bad

    def budget_to_json(self) -> dict:
        return {
            "requested_base_budget_tokens": self.requested_base_budget_tokens,
            "budget_multiplier": self.budget_multiplier,
            "effective_budget_tokens": self.effective_budget_tokens,
            "hard_max_tokens": self.hard_max_tokens,
            "budget_policy": self.budget_policy,
            "max_context_tokens": self.max_context_tokens,
        }


@dataclass
class ChunkHit:
    """A single chunk-level retrieval hit returned by WikiIndex.search_fts /
    search_vector. query.py fuses these into PageCandidates via page-level RRF.
    """
    chunk_id: str
    page_id: str
    path: str
    title: str
    page_type: str
    section_path: List[str]
    heading: str
    chunk_kind: str        # 'dense' | 'sparse'
    text: str
    channel: str           # 'fts' | 'vector'
    score: float
    distance: Optional[float] = None
    chunk_index: Optional[int] = None  # persisted document order; content hash is position-independent


@dataclass
class IndexState:
    """manifest v2 `index_state` (issues #1/#2/#7/#8/#11)."""
    schema_version: int = 2
    chunk_schema_version: int = 3  # #13：chunk_id 改为内容哈希（page_id::{sha256}），与位置无关
    tokenizer_schema_version: int = 1
    embedding_model: str = ""
    embedding_model_revision: str = ""
    embedding_dimension: int = 384
    vector_metric: str = "cosine"
    fts_config_hash: str = ""
    chunk_config_hash: str = ""
