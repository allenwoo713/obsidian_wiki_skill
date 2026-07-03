"""MineruCloudParser 单元测试（不调用真实 API）。"""
from __future__ import annotations

import io
import sys
import tempfile
import zipfile
from pathlib import Path
from unittest.mock import patch

import fitz
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from parsers.mineru_cloud import MineruCloudParser
from parsers.pdf_split import split_pdf_into_chunks


class MockResponse:
    def __init__(self, status_code=200, json_data=None, content=b"", text=""):
        self.status_code = status_code
        self._json = json_data
        self.content = content
        self.text = text

    def json(self):
        return self._json


@pytest.fixture
def tmp_dir() -> Path:
    """使用系统临时目录，避免项目内 .pytest_tmp 受沙箱限制。"""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


def _make_pdf(tmp_path: Path, page_count: int) -> Path:
    """用 fitz 创建指定页数的 PDF。"""
    pdf_path = tmp_path / "source.pdf"
    doc = fitz.open()
    for _ in range(page_count):
        page = doc.new_page()
        page.insert_text((50, 50), "page marker")
    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


def _make_zip_bytes(markdown: str, images: dict | None = None) -> bytes:
    """构造内存中的假结果 zip。"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("full.md", markdown)
        if images:
            for name, data in images.items():
                zf.writestr(f"images/{name}", data)
    return buf.getvalue()


def _mock_requests_factory(batch_id, file_urls, extract_result, zip_map):
    """返回 (fake_post, fake_put, fake_get)，用于 patch requests 三个方法。"""

    def fake_post(url, **kwargs):
        if "file-urls/batch" in url:
            return MockResponse(
                json_data={
                    "code": 0,
                    "data": {"batch_id": batch_id, "file_urls": file_urls},
                }
            )
        return MockResponse(status_code=404, text="not found")

    def fake_put(url, **kwargs):
        return MockResponse()

    def fake_get(url, **kwargs):
        if f"/extract-results/batch/{batch_id}" in url:
            return MockResponse(
                json_data={
                    "code": 0,
                    "data": {
                        "batch_id": batch_id,
                        "extract_result": extract_result,
                    },
                }
            )
        if url in zip_map:
            return MockResponse(content=zip_map[url])
        return MockResponse(status_code=404, text="not found")

    return fake_post, fake_put, fake_get


def test_parse_single_pdf_no_split(tmp_dir: Path):
    """5 页 PDF 不拆分，验证返回 ParseResult 含 1 张图片和 1 个表格占位。"""
    pdf_path = _make_pdf(tmp_dir, 5)
    parser = MineruCloudParser(api_token="test")

    markdown = "# Title\n\n![](images/abc123.jpg)\n\n<table><tr><td>A</td></tr></table>\n"
    zip_bytes = _make_zip_bytes(markdown, {"abc123.jpg": b"imgdata"})

    file_urls = ["https://oss.example.com/upload/1"]
    extract_result = [
        {
            "file_name": "source.pdf",
            "state": "done",
            "full_zip_url": "https://result.example.com/1.zip",
            "err_msg": "",
        }
    ]
    zip_map = {"https://result.example.com/1.zip": zip_bytes}

    fake_post, fake_put, fake_get = _mock_requests_factory(
        "batch_single", file_urls, extract_result, zip_map
    )

    with (
        patch("requests.post", side_effect=fake_post),
        patch("requests.put", side_effect=fake_put),
        patch("requests.get", side_effect=fake_get),
    ):
        result = parser.parse(pdf_path)

    assert len(result.images) == 1
    assert result.images[0].source_media_name == "abc123.jpg"
    assert len(result.tables) == 1
    assert result.tables[0] == [["A"]]
    assert "{{IMG|" in result.text
    assert "<table>" in result.text  # HTML 表格保留在 text 中


def test_parse_pdf_split(tmp_dir: Path):
    """250 页 PDF 拆成 3 段，验证上传 3 个文件且合并 markdown 含 3 段分隔。"""
    pdf_path = _make_pdf(tmp_dir, 250)
    parser = MineruCloudParser(api_token="test", max_pages_per_file=100)

    file_urls = [
        "https://oss.example.com/upload/1",
        "https://oss.example.com/upload/2",
        "https://oss.example.com/upload/3",
    ]
    extract_result = [
        {
            "file_name": "chunk_001.pdf",
            "state": "done",
            "full_zip_url": "https://result.example.com/1.zip",
            "err_msg": "",
        },
        {
            "file_name": "chunk_002.pdf",
            "state": "done",
            "full_zip_url": "https://result.example.com/2.zip",
            "err_msg": "",
        },
        {
            "file_name": "chunk_003.pdf",
            "state": "done",
            "full_zip_url": "https://result.example.com/3.zip",
            "err_msg": "",
        },
    ]
    zip_map = {
        "https://result.example.com/1.zip": _make_zip_bytes("# Part 1"),
        "https://result.example.com/2.zip": _make_zip_bytes("# Part 2"),
        "https://result.example.com/3.zip": _make_zip_bytes("# Part 3"),
    }

    fake_post, fake_put, fake_get = _mock_requests_factory(
        "batch_split", file_urls, extract_result, zip_map
    )

    with (
        patch("requests.post", side_effect=fake_post),
        patch("requests.put", side_effect=fake_put),
        patch("requests.get", side_effect=fake_get),
    ):
        result = parser.parse(pdf_path)

    # 3 段 markdown 用 2 个分隔线连接
    assert result.text.count("---") == 2
    assert "# Part 1" in result.text
    assert "# Part 2" in result.text
    assert "# Part 3" in result.text


def test_parse_xlsx(tmp_dir: Path, monkeypatch):
    """非 PDF 文件直接上传，不调用 PDF 拆分。"""
    xlsx_path = tmp_dir / "report.xlsx"
    xlsx_path.write_bytes(b"fake xlsx content")
    parser = MineruCloudParser(api_token="test")

    split_called = False
    original_split = split_pdf_into_chunks

    def tracked_split(*args, **kwargs):
        nonlocal split_called
        split_called = True
        return original_split(*args, **kwargs)

    monkeypatch.setattr(
        "parsers.mineru_cloud.split_pdf_into_chunks", tracked_split
    )

    file_urls = ["https://oss.example.com/upload/1"]
    extract_result = [
        {
            "file_name": "report.xlsx",
            "state": "done",
            "full_zip_url": "https://result.example.com/report.zip",
            "err_msg": "",
        }
    ]
    zip_map = {
        "https://result.example.com/report.zip": _make_zip_bytes("# Report")
    }

    fake_post, fake_put, fake_get = _mock_requests_factory(
        "batch_xlsx", file_urls, extract_result, zip_map
    )

    with (
        patch("requests.post", side_effect=fake_post),
        patch("requests.put", side_effect=fake_put),
        patch("requests.get", side_effect=fake_get),
    ):
        result = parser.parse(xlsx_path)

    assert not split_called
    assert "# Report" in result.text


def test_missing_token_raises(monkeypatch):
    """未传 token 且环境变量为空时，初始化即抛错。"""
    monkeypatch.delenv("MINERU_API_TOKEN", raising=False)
    with pytest.raises(ValueError):
        MineruCloudParser()


def test_api_error_raises(tmp_dir: Path):
    """上传 URL 接口返回 code != 0，应抛 RuntimeError。"""
    xlsx_path = tmp_dir / "report.xlsx"
    xlsx_path.write_bytes(b"fake xlsx content")
    parser = MineruCloudParser(api_token="test")

    def fake_post(url, **kwargs):
        return MockResponse(json_data={"code": 1001, "msg": "invalid token", "data": None})

    with patch("requests.post", side_effect=fake_post):
        with pytest.raises(RuntimeError):
            parser.parse(xlsx_path)
