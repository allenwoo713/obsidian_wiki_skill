"""共享数据结构，所有脚本统一引用。"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional


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
