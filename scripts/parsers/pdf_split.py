"""PDF 拆分工具。

仅按页数拆分 PDF，不做任何内容解析。
用于 MinerU Cloud API 的 200 页/文件限制场景。
"""
from __future__ import annotations
from pathlib import Path
from typing import List

import fitz  # PyMuPDF


def split_pdf_into_chunks(
    input_path: Path,
    output_dir: Path,
    pages_per_chunk: int = 200,
) -> List[Path]:
    """将 PDF 拆分为每段不超过 pages_per_chunk 页的子文件。

    若总页数 <= pages_per_chunk，直接返回原文件路径（不复制）。
    否则在 output_dir 下生成 chunk_001.pdf, chunk_002.pdf, ...。

    Args:
        input_path: 源 PDF 路径。
        output_dir: 拆分文件输出目录。
        pages_per_chunk: 每段最大页数。

    Returns:
        拆分后的 PDF 路径列表（按页序）。
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with fitz.open(str(input_path)) as src_doc:
        total_pages = src_doc.page_count
        if total_pages <= pages_per_chunk:
            return [input_path.resolve()]

        chunk_paths: List[Path] = []
        chunk_index = 1
        for start in range(0, total_pages, pages_per_chunk):
            end = min(start + pages_per_chunk, total_pages)
            chunk_doc = fitz.open()
            chunk_doc.insert_pdf(src_doc, from_page=start, to_page=end - 1)
            chunk_path = output_dir / f"chunk_{chunk_index:03d}.pdf"
            chunk_doc.save(str(chunk_path))
            chunk_doc.close()
            chunk_paths.append(chunk_path.resolve())
            chunk_index += 1

        return chunk_paths
