"""共享数据结构，所有脚本统一引用。"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional


@dataclass
class ParsedDoc:
    """解析后的源文档。"""
    path: Path
    title: str
    text: str
    tables: List[List[List[str]]]  # [table][row][cell]
    sha256: str
    doc_type: str  # 'docx' | 'pdf' | 'md' | 'txt'


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
    """检索结果。"""
    path: Path
    title: str
    score: float
    snippet: str
    sources: List[str]
    retrieval_method: str  # 'bm25' | 'vector' | 'graph' | 'fused'


@dataclass
class ManifestEntry:
    """manifest.json 单条记录。"""
    path: str
    sha256: str
    mtime: float
    status: str  # 'new' | 'processed' | 'modified' | 'deleted'
    wiki_pages: List[str]
    last_processed: Optional[str]  # ISO datetime
