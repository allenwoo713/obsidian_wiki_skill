"""MinerU Cloud v4 API 文档解析器（本地文件上传 + 批量精准解析）。"""
from __future__ import annotations

import os
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import requests

from parsers.base import DocumentParser, ParseResult
from parsers.mineru_common import mineru_markdown_to_parse_result
from parsers.pdf_split import split_pdf_into_chunks
from parsers.utils import slugify


BASE_URL = "https://mineru.net/api/v4"

SUPPORTED_EXTS = {
    ".pdf",
    ".doc",
    ".docx",
    ".ppt",
    ".pptx",
    ".xls",
    ".xlsx",
    ".html",
    ".htm",
}


class MineruCloudParser(DocumentParser):
    """调用 MinerU Cloud 精准解析 API (v4) 解析本地文档。

    支持 PDF/Office/HTML 等格式；PDF 超过 max_pages_per_file 时自动拆分后批量上传。
    """

    def __init__(
        self,
        api_token: Optional[str] = None,
        model_version: str = "vlm",
        language: str = "en",
        enable_table: bool = True,
        enable_formula: bool = True,
        max_pages_per_file: int = 200,
        poll_interval: int = 5,
        max_poll: int = 120,
    ):
        if api_token is None:
            api_token = os.environ.get("MINERU_API_TOKEN")
        if not api_token:
            raise ValueError("必须提供 api_token 参数或设置 MINERU_API_TOKEN 环境变量")
        self.api_token = api_token
        self.model_version = model_version
        self.language = language
        self.enable_table = enable_table
        self.enable_formula = enable_formula
        self.max_pages_per_file = max_pages_per_file
        self.poll_interval = poll_interval
        self.max_poll = max_poll

    def parse(self, path: Path) -> ParseResult:
        """解析单个本地文件，返回带图片占位符与表格的 ParseResult。"""
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"文件不存在：{path}")
        ext = path.suffix.lower()
        if ext not in SUPPORTED_EXTS:
            raise ValueError(f"不支持的文件格式：{ext}")

        doc_slug = slugify(path.stem)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            # 仅 PDF 需要按页数拆分
            if ext == ".pdf":
                file_paths = split_pdf_into_chunks(
                    path, tmp_path / "chunks", pages_per_chunk=self.max_pages_per_file
                )
            else:
                file_paths = [path]

            files_meta = [{"name": p.name, "data_id": p.stem} for p in file_paths]
            batch_id, file_urls = self._request_upload_urls(files_meta)

            # 依次上传到 MinerU 返回的 OSS 预签名 URL
            for file_path, upload_url in zip(file_paths, file_urls):
                self._upload_file(file_path, upload_url)

            # 轮询批量解析结果
            results = self._poll_batch_results(batch_id, len(file_paths))

            # 下载每个文件对应的 zip 结果并汇总
            markdown_parts: List[str] = []
            image_bytes_map: Dict[str, bytes] = {}
            for idx, file_path in enumerate(file_paths):
                file_name = file_path.name
                result = results.get(file_name)
                if result is None:
                    raise RuntimeError(f"MinerU 返回结果中缺少文件：{file_name}")

                extract_dir = tmp_path / f"extract_{idx:03d}"
                self._download_and_extract_zip(result["full_zip_url"], extract_dir)

                md_path = extract_dir / "full.md"
                markdown_parts.append(md_path.read_text(encoding="utf-8"))

                images_dir = extract_dir / "images"
                if images_dir.is_dir():
                    for img_path in images_dir.iterdir():
                        if img_path.is_file():
                            image_bytes_map[img_path.name] = img_path.read_bytes()

            full_markdown = "\n\n---\n\n".join(markdown_parts)
            return mineru_markdown_to_parse_result(
                markdown=full_markdown,
                doc_slug=doc_slug,
                images_dir=tmp_path / "images",
                image_bytes_map=image_bytes_map,
            )

    def _request_upload_urls(self, files: List[Dict[str, str]]) -> tuple[str, List[str]]:
        """批量获取 OSS 上传 URL 与 batch_id。"""
        url = f"{BASE_URL}/file-urls/batch"
        headers = {
            "Authorization": f"Bearer {self.api_token}",
            "Content-Type": "application/json",
        }
        payload = {
            "files": files,
            "model_version": self.model_version,
            "enable_formula": self.enable_formula,
            "enable_table": self.enable_table,
            "language": self.language,
        }
        resp = requests.post(url, headers=headers, json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"MinerU 获取上传 URL 失败 [{resp.status_code}]: {resp.text[:200]}")

        data = resp.json()
        if data.get("code", -1) != 0:
            raise RuntimeError(f"MinerU 获取上传 URL 返回错误: {data}")

        return data["data"]["batch_id"], data["data"]["file_urls"]

    def _upload_file(self, file_path: Path, upload_url: str) -> None:
        """直接 PUT 文件二进制到 OSS 预签名 URL。"""
        with open(file_path, "rb") as f:
            resp = requests.put(upload_url, data=f)
        if resp.status_code not in (200, 201, 204):
            raise RuntimeError(f"上传文件到 OSS 失败 [{resp.status_code}]: {resp.text[:200]}")

    def _poll_batch_results(self, batch_id: str, expected_count: int) -> Dict[str, Dict]:
        """轮询批量解析结果，完成后按 file_name 建立映射。"""
        url = f"{BASE_URL}/extract-results/batch/{batch_id}"
        headers = {"Authorization": f"Bearer {self.api_token}"}

        for _ in range(self.max_poll):
            resp = requests.get(url, headers=headers)
            if resp.status_code != 200:
                raise RuntimeError(f"MinerU 轮询结果失败 [{resp.status_code}]: {resp.text[:200]}")

            data = resp.json()
            if data.get("code", -1) != 0:
                raise RuntimeError(f"MinerU 轮询结果返回错误: {data}")

            extract_result = data["data"].get("extract_result", [])
            if len(extract_result) >= expected_count:
                failed = [item for item in extract_result if item.get("state") == "failed"]
                if failed:
                    details = "; ".join(
                        f"{item['file_name']}: {item.get('err_msg', '')}" for item in failed
                    )
                    raise RuntimeError(f"MinerU 解析失败: {details}")

                if all(item.get("state") == "done" for item in extract_result):
                    return {item["file_name"]: item for item in extract_result}

            time.sleep(self.poll_interval)

        raise TimeoutError(f"MinerU 解析结果轮询超时，batch_id={batch_id}")

    def _download_and_extract_zip(self, zip_url: str, extract_dir: Path) -> None:
        """下载结果 zip 并解压到 extract_dir。"""
        resp = requests.get(zip_url)
        if resp.status_code != 200:
            raise RuntimeError(f"下载结果 zip 失败 [{resp.status_code}]: {resp.text[:200]}")

        extract_dir.mkdir(parents=True, exist_ok=True)
        zip_path = extract_dir / "result.zip"
        zip_path.write_bytes(resp.content)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
