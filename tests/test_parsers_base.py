"""parsers/base.py 测试：DocumentParser ABC。"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from parsers.base import DocumentParser, ParseResult


def test_document_parser_is_abstract():
    with pytest.raises(TypeError):
        DocumentParser()


def test_parse_result_fields():
    r = ParseResult(text="hello", images=[], tables=[])
    assert r.text == "hello"
    assert r.images == []
    assert r.tables == []
