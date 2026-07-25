"""Unit tests for chunking.py (GitHub issue #1).

Uses a 1-token-per-char tokenizer so token-budget assertions are exact and
the suite runs without loading the heavy embedding model.
"""
from pathlib import Path

from chunking import (
    chunk_page,
    DENSE_HARD_MAX_TOKENS,
)


def char_tok(s):
    return len(s)


def _dense(recs):
    return [r for r in recs if r.parent_section_id is not None]


def _sparse(recs):
    return [r for r in recs if r.parent_section_id is None]


def test_heading_assignment_into_correct_section():
    content = "# A\nintro\n## B\nmid\n### C\ndeep answer\n# A2\nother"
    recs = chunk_page("p1", Path("Wiki/x.md"), "X", "concept", content,
                      tokenizer=char_tok)
    dense = _dense(recs)
    deep = [r for r in dense if "deep answer" in r.text]
    assert deep, "expected a dense chunk containing 'deep answer'"
    assert deep[0].section_path == ["A", "B", "C"], deep[0].section_path
    assert deep[0].heading == "C"
    # must not be mis-attributed to an ancestor heading
    assert all(r.heading == "C" for r in deep)


def test_dense_token_cap_on_long_paragraph():
    para = "数据库异常增长导致WAL文件膨胀需紧急处理。" * 200  # no newline, very long
    content = "# 主题\n" + para
    recs = chunk_page("p2", Path("Wiki/y.md"), "Y", "concept", content,
                      tokenizer=char_tok)
    dense = _dense(recs)
    assert len(dense) > 1, "long paragraph must split into multiple chunks"
    for r in dense:
        assert r.token_count <= DENSE_HARD_MAX_TOKENS, (r.token_count, r.text[:40])


def test_wikilink_not_split_midway():
    para = ("关于配置请参考" + "上下文信息" * 80
            + "[[CapabilityAccessManager 配置说明]]" + "完成部署" + "后续步骤说明" * 80)
    content = "# T\n" + para
    recs = chunk_page("p3", Path("Wiki/z.md"), "Z", "concept", content,
                      tokenizer=char_tok)
    joined = " ".join(r.text for r in _dense(recs))
    assert "[[CapabilityAccessManager 配置说明]]" in joined


def test_code_block_intact_when_small():
    code = "```python\n" + "\n".join(f"x{i} = {i}" for i in range(5)) + "\n```"
    content = "# Code\n" + code + "\nsome text after"
    recs = chunk_page("p4", Path("Wiki/c.md"), "C", "concept", content,
                      tokenizer=char_tok)
    assert any(code.strip() in r.text for r in _dense(recs))


def test_stable_ids_across_runs():
    content = "# A\npara one\n## B\npara two"
    r1 = chunk_page("p5", Path("Wiki/a.md"), "A", "concept", content,
                    tokenizer=char_tok)
    r2 = chunk_page("p5", Path("Wiki/a.md"), "A", "concept", content,
                    tokenizer=char_tok)
    assert [r.chunk_id for r in r1] == [r.chunk_id for r in r2]


def test_sparse_section_present():
    content = "# A\npara\n## B\nmore"
    recs = chunk_page("p6", Path("Wiki/b.md"), "B", "concept", content,
                      tokenizer=char_tok)
    sparse = _sparse(recs)
    assert sparse, "expected sparse section chunks"
    assert any("para" in r.text for r in sparse)
