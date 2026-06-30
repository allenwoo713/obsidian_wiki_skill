"""DocumentParser 抽象接口与 ParseResult 数据结构。"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from models import ImageRef


@dataclass
class ParseResult:
    """Parser 内部返回结构（不落盘，由 extract_assets 转换为 ParsedDoc）。"""
    text: str                          # 带图片占位符的全文
    images: List[ImageRef] = field(default_factory=list)
    tables: List[List[List[str]]] = field(default_factory=list)

    # 内部字段：=None 使得默认不生成 _image_bytes 也能实例化
    _image_bytes: List[bytes] = field(default_factory=list, repr=False)


class DocumentParser(ABC):
    """文档解析器接口。子类实现 parse()，返回 ParseResult（内存对象，不写盘）。"""

    @abstractmethod
    def parse(self, path: Path) -> ParseResult:
        ...
