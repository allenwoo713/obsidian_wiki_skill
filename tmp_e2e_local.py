"""End-to-end: parse Vega PDF with MinerU Local backend."""
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from parsers.mineru_local import MineruLocalPdfParser

pdf_path = Path("<project_root>/Raw/sources/Datasheet/Vega/Vega_Standalone_Camera_UDP_User_Manual_1.3.pdf")
parser = MineruLocalPdfParser(backend="pipeline", language="en")

print(f"Parsing {pdf_path.name} with MinerU Local...")
t0 = time.time()
result = parser.parse(pdf_path)
elapsed = time.time() - t0

print(f"Done in {elapsed:.1f}s")
print(f"Text length: {len(result.text)} chars")
print(f"Images: {len(result.images)}")
print(f"Tables: {len(result.tables)}")
print("--- First 500 chars ---")
print(result.text[:500])
