"""Issue #13 review (Gap 2): sparse hard limit must be a TRUE hard limit.

Pure-chunking unit tests (char tokenizer, no model load). For every fixture we
assert that NO sparse chunk exceeds ``SPARSE_HARD_MAX_CHARS`` (even after the
section prefix is prepended), that over-long content is force-split with
``forced_split`` metadata, that each chunk body is a real substring of the
source (no fabricated/garbled text, no silent char loss), and that the
un-bypassable hard-limit guard is actually wired.
"""
from pathlib import Path
from unittest import mock

import pytest

import chunking
from chunking import (
    chunk_page,
    ChunkBuildError,
    SPARSE_HARD_MAX_CHARS,
)

# 1 token per char → exact char budgets for assertions
char_tok = lambda s: len(s)
PREFIX = "A\n"  # section_path=["A"] → prefix "A\n"


def _sparse(records):
    return [r for r in records if r.chunk_kind == "sparse"]


def _body(r):
    return r.text[len(PREFIX):] if r.text.startswith(PREFIX) else r.text


def _covered_source(records, content):
    """把所有稀疏 chunk 的真实源区间拼接（去换行），用于覆盖度校验：若实现
    静默丢字，目标超长串将不完整/缺失。与 r.text 解耦，不受 section 前缀或
    表格管道重建干扰。"""
    sp = _sparse(records)
    return "".join(content[r.start_char:r.end_char] for r in sp).replace("\n", "")


def _assert_hard_limit(records, content):
    """鲁棒的稀疏硬上限不变量。

    重建会在原子单元之间插入 ``\\n`` 分隔符、并为表格重新对齐 ``|`` 管道，因此
    chunk 正文**不是**源文本的字面子串——用 ``body in content`` 判定会误报。改为校验：
      1) 真·硬上限：含前缀的完整 chunk 文本 ``len(r.text) <= SPARSE_HARD_MAX_CHARS``；
      2) span 合法（索引落在源区间内）；
      3) chunk 文本**包含**其真实源区间（去掉分隔符后），即无编造、无静默丢字。
    返回所有稀疏 chunk 拼接（去换行）后的字符串，供调用方做覆盖度校验。
    """
    sp = _sparse(records)
    assert sp, "expected sparse chunks"
    for r in sp:
        assert len(r.text) <= SPARSE_HARD_MAX_CHARS, (len(r.text), r.text[:60])
        assert 0 <= r.start_char <= r.end_char <= len(content), \
            (r.start_char, r.end_char, len(content))
        src_region = content[r.start_char:r.end_char].replace("\n", "")
        assert src_region and src_region in r.text.replace("\n", ""), \
            f"source region not embedded in chunk text: {src_region[:40]!r}"
    return "".join(r.text for r in sp).replace("\n", "")


def test_sparse_single_overlong_sentence():
    long = "数据库持久化异常恢复机制确保写入不丢失" * 120   # ~1920 字符，无标点无换行
    content = "# A\n" + long
    recs = chunk_page("p", Path("Wiki/a.md"), "A", "concept", content, tokenizer=char_tok)
    _assert_hard_limit(recs, content)
    sp = _sparse(recs)
    assert any(r.forced_split for r in sp), "overlong sentence must be force-split"
    # 无静默丢字：各 chunk 正文顺序拼接 == 原句
    assert "".join(_body(r) for r in sp) == long


def test_sparse_overlong_code_line():
    code = "x" * 2000
    content = "# A\n```\n" + code + "\n```"
    recs = chunk_page("p", Path("Wiki/a.md"), "A", "concept", content, tokenizer=char_tok)
    _assert_hard_limit(recs, content)
    sp = _sparse(recs)
    assert any(r.forced_split for r in sp)
    bodies = "".join(_body(r) for r in sp)
    assert "x" * 2000 in bodies, "code line content lost during forced split"


def test_sparse_overlong_table_cell():
    longcell = "数据持久化机制" * 300   # ~7*300=2100 字符的单 cell
    content = ("# A\n| 列1 | 列2 |\n|---|---|\n| 短 | " + longcell + " |\n| a | b |\n")
    recs = chunk_page("p", Path("Wiki/a.md"), "A", "concept", content, tokenizer=char_tok)
    all_text = _assert_hard_limit(recs, content)
    assert longcell in _covered_source(recs, content), "overlong table cell content lost during force-split"
    assert any(r.forced_split for r in _sparse(recs)), "overlong table cell must be force-split"
    # 强制切片片段的 span 必须与片段文本逐字符一致（真实 span）
    for r in _sparse(recs):
        if r.forced_split:
            assert r.text[len(PREFIX):] == content[r.start_char:r.end_char], \
                f"table frag span mismatch: {r.text[len(PREFIX):]!r} vs {content[r.start_char:r.end_char]!r}"


def test_sparse_overlong_table_header():
    longhead = "标题层级字段" * 300
    content = ("# A\n| " + longhead + " | 说明 |\n|---|---|\n| a | b |\n")
    recs = chunk_page("p", Path("Wiki/a.md"), "A", "concept", content, tokenizer=char_tok)
    all_text = _assert_hard_limit(recs, content)
    assert longhead in _covered_source(recs, content), "overlong table header content lost during force-split"
    assert any(r.forced_split for r in _sparse(recs)), "overlong header row must be force-split"
    # 强制切片片段的 span 必须与片段文本逐字符一致（真实 span）
    for r in _sparse(recs):
        if r.forced_split:
            assert r.text[len(PREFIX):] == content[r.start_char:r.end_char], \
                f"table frag span mismatch: {r.text[len(PREFIX):]!r} vs {content[r.start_char:r.end_char]!r}"


def test_sparse_overlong_list_item():
    item = "列表条目内容" * 600
    content = "# A\n* " + item + "\n* 其它项\n"
    recs = chunk_page("p", Path("Wiki/a.md"), "A", "concept", content, tokenizer=char_tok)
    _assert_hard_limit(recs, content)
    assert any(r.forced_split for r in _sparse(recs))


def test_sparse_overlong_quote():
    q = "引用段落内容" * 600
    content = "# A\n> " + q + "\n"
    recs = chunk_page("p", Path("Wiki/a.md"), "A", "concept", content, tokenizer=char_tok)
    _assert_hard_limit(recs, content)
    assert any(r.forced_split for r in _sparse(recs))


def test_sparse_prefix_budget_respected():
    # 深层标题使 prefix 超过硬上限(800) → 压缩前缀，正文预算 >= MIN_BODY_RESERVE
    h = "章节标题字段内容" * 40   # 单段 320 字符，三段 + " / " 分隔 > 800 → 触发截断
    content = f"# {h}\n## {h}\n### {h}\n" + ("正文内容用于检索与召回测试样本。" * 200)
    recs = chunk_page("p", Path("Wiki/a.md"), "A", "concept", content, tokenizer=char_tok)
    all_text = _assert_hard_limit(recs, content)
    # 前缀被压缩（含省略标记），正文预算未被前缀吞没
    assert "(section path truncated)" in all_text, \
        "prefix should be truncated when exceeding hard limit"
    # 正文全量保留（无静默丢字）：逐句 span 拼接后等于原文段落
    assert ("正文内容用于检索与召回测试样本。" * 200) in _covered_source(recs, content), \
        "body content lost under truncated prefix"


def test_sparse_overlong_wikilink():
    wl = "[[" + "链接目标标识符名称" * 200 + "]]"   # 超长 wikilink（单原子 > 硬上限）
    content = "# A\n" + wl
    recs = chunk_page("p", Path("Wiki/a.md"), "A", "concept", content, tokenizer=char_tok)
    _assert_hard_limit(recs, content)
    sp = _sparse(recs)
    assert any(r.forced_split for r in sp), "single overlong wikilink must be force-split"
    bodies = "".join(_body(r) for r in sp)
    assert "[" in bodies and "]" in bodies, "wikilink brackets must survive forced split"



def test_sparse_hard_limit_guard_wired():
    """不可绕过的最终守卫：任一 fallback 产生超长 chunk 必须抛 ChunkBuildError。"""
    content = "# A\n" + "x" * 5000
    real = chunking._char_slice_text

    def _bad(text, budget, base):
        # 故意返回超长片段，验证守卫会捕获
        return [("x" * (budget + 50), base, base + budget + 50)]

    with mock.patch.object(chunking, "_char_slice_text", _bad):
        with pytest.raises(ChunkBuildError):
            chunk_page("p", Path("Wiki/a.md"), "A", "concept", content, tokenizer=char_tok)
