"""Unit tests for lexical_tokenizer.py (GitHub issue #2).

Runs with or without Jieba installed (Jieba only sharpens precision; the
CJK-bigram fallback keeps exact-match and bigram behaviour testable).
"""
from lexical_tokenizer import (
    extract_exact_terms,
    fts_terms,
    load_lexicon,
    tokenize_doc,
    tokenize_query,
)


def test_extract_exact_terms():
    text = ("型号 ARS540 报错 ERR-102，路径 C:/foo/bar "
            "详见 https://docs.x.com/y 延迟 12.5 mm 使用 --mode fast")
    terms = extract_exact_terms(text)
    assert "ARS540" in terms
    assert "ERR-102" in terms
    assert any(t.startswith("C:") for t in terms)
    assert any(t.startswith("https://") for t in terms)
    assert "12.5 mm" in terms
    assert "--mode" in terms


def test_fts_terms_chinese_bigrams():
    terms = set(fts_terms("数据库异常增长"))
    assert {"数据", "据库", "异常", "增长"}.issubset(terms)


def test_fts_terms_english_subwords():
    terms = set(fts_terms("CapabilityAccessManager"))
    assert "CapabilityAccessManager" in terms
    assert {"capability", "access", "manager"}.issubset(terms)


def test_tokenize_query_returns_both():
    fts, exact = tokenize_query("ARS540 错误码 ERR-102")
    assert isinstance(fts, list) and isinstance(exact, list)
    assert "ERR-102" in exact


def test_tokenize_doc_returns_fts_terms():
    assert isinstance(tokenize_doc("WAL 文件异常"), list)


def test_lexicon_loaded():
    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        lex = Path(d) / "lexicon.txt"
        lex.write_text("# comment\nMyProductX\n", encoding="utf-8")
        terms = load_lexicon(Path(d))
    assert "MyProductX" in terms
    assert "# comment" not in terms
