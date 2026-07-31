"""Unit tests for chunking.py (GitHub issue #1).

Uses a 1-token-per-char tokenizer so token-budget assertions are exact and
the suite runs without loading the heavy embedding model.
"""
import re
from pathlib import Path

from chunking import (
    chunk_page,
    DENSE_HARD_MAX_TOKENS,
    EmbeddingTokenizer,
    _norm,
)


def char_tok(s):
    return len(s)


def _body(r):
    """The chunk body = stored text with its section-path prefix stripped."""
    prefix = (" / ".join(r.section_path) + "\n") if r.section_path else ""
    return r.text[len(prefix):] if r.text.startswith(prefix) else r.text


class _FakeHF:
    """Minimal HF-tokenizer stand-in: 1 token per character."""
    def encode(self, t):
        return list(range(len(t)))


class _FastHF:
    def __init__(self):
        self.calls = []

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return {"input_ids": list(range(len(text) + 2))}

    def encode(self, _):
        raise AssertionError("counting must not use encode")


def test_embedding_tokenizer_real_count():
    et = EmbeddingTokenizer(_FakeHF())
    assert et.count("abc") == 3


def test_embedding_tokenizer_count_disables_truncation_without_model_length_warning():
    hf = _FastHF()
    assert EmbeddingTokenizer(hf).count("x" * 500) == 502
    assert hf.calls == [("x" * 500, {
        "add_special_tokens": True,
        "truncation": False,
        "return_attention_mask": False,
        "return_token_type_ids": False,
        "verbose": False,
    })]


def test_embedding_tokenizer_rejects_none():
    import pytest
    with pytest.raises(ValueError):
        EmbeddingTokenizer(None)


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


def test_stable_ids_under_insertion():
    """Issue #13: inserting an unrelated section at the top must NOT change
    the chunk IDs of unchanged chunks (IDs are content-derived, not positional).
    """
    base = "# A\npara one\n## B\npara two\n### C\ndeep answer\n# A2\nother"
    recs_base = chunk_page("p1", Path("Wiki/x.md"), "X", "concept", base,
                           tokenizer=char_tok)
    inserted = "## UNRELATED\nirrelevant stuff here\n\n" + base
    recs_ins = chunk_page("p1", Path("Wiki/x.md"), "X", "concept", inserted,
                          tokenizer=char_tok)
    base_ids = {(r.chunk_kind, _norm(_body(r))): r.chunk_id for r in recs_base}
    ins_ids = {(r.chunk_kind, _norm(_body(r))): r.chunk_id for r in recs_ins}
    for key, cid in base_ids.items():
        assert ins_ids.get(key) == cid, f"unchanged chunk {key} changed ID after insertion"


def test_sparse_respects_block_boundaries():
    """Issue #13: sparse chunks must not cut Markdown tables, lists, code
    blocks or wikilinks — they split only on block boundaries."""
    table = "| a | b |\n|---|---|\n" + "\n".join(
        f"| {i} | v{i} |" for i in range(40))
    content = (
        "# Big\n" + table + "\n"
        "## List\n"
        "- item one with [[WL A 说明]]\n"
        "- item two with [[WL B 说明]]\n"
        "```\ncode line x\ncode line y\n```\n"
    )
    recs = chunk_page("p", Path("Wiki/b.md"), "B", "concept", content,
                      tokenizer=char_tok)
    sparse = _sparse(recs)
    bodies = [r.text for r in sparse]
    joined = "\n".join(bodies)
    for wl in re.findall(r"\[\[.*?\]\]", content):
        assert wl in joined, f"wikilink {wl} cut across sparse chunks"
    for i in range(40):
        row = f"| {i} | v{i} |"
        assert any(row in b for b in bodies), f"table row {i} cut across sparse chunks"


def test_dense_span_maps_to_original():
    """Issue #13: dense chunk start_char/end_char must map back to the original
    text and cover the chunk's actual body (real original span)."""
    content = ("# A\nfirst sentence here. second sentence here.\n"
               "## B\ndeep answer paragraph text.\n")
    recs = chunk_page("p", Path("Wiki/c.md"), "C", "concept", content,
                      tokenizer=char_tok)
    for r in _dense(recs):
        s, e = r.start_char, r.end_char
        assert 0 <= s < e <= len(content), (s, e, len(content))
        body = _body(r)
        seg = content[s:e]
        # every character of the body appears in order within the spanned text
        it = iter(seg)
        assert all(c in it for c in body), \
            f"body not a subsequence of span: {body!r} vs {seg!r}"
