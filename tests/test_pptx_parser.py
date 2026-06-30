"""PptxParser骨架测试：应抛NotImplementedError。"""
import sys; from pathlib import Path; import pytest
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from parsers.pptx_parser import PptxParser

def test_pptx_parser_raises_not_implemented(tmp_path):
    fake = tmp_path / "f.pptx"; fake.write_bytes(b"PK")
    with pytest.raises(NotImplementedError, match="PPTX"):
        PptxParser().parse(fake)
