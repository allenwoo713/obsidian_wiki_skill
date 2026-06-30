"""PptxParser 骨架：YAGNI，实现待 Raw 出现 pptx 文件后。"""
from __future__ import annotations
from pathlib import Path
from parsers.base import DocumentParser, ParseResult

class PptxParser(DocumentParser):
    def parse(self, path: Path) -> ParseResult:
        raise NotImplementedError(
            "PPTX解析待实现。按OOXML标准：ppt/slides/slideN.xml中<p:pic>定位图片，ppt/media/提取。"
        )
