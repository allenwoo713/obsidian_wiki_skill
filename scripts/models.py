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


@dataclass
class ContextBundle:
    """Final retrieval output — directly consumable by the LLM."""
    query: str
    mode: str
    items: List[ContextItem] = field(default_factory=list)
    context_text: str = ""
    token_count: int = 0
    max_context_tokens: int = 0
    omitted_items: List[dict] = field(default_factory=list)


@dataclass
class IndexState:
    """manifest v2 `index_state` (issues #1/#2/#7/#8/#11)."""
    schema_version: int = 2
    chunk_schema_version: int = 2
    tokenizer_schema_version: int = 1
    embedding_model: str = ""
    embedding_model_revision: str = ""
    embedding_dimension: int = 384
    vector_metric: str = "cosine"
    fts_config_hash: str = ""
    chunk_config_hash: str = ""
