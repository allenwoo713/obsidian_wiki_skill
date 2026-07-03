"""End-to-end: parse Vega PDF via parse_file with is_sensitive=True."""
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from parse_sources import parse_file

pdf_path = Path("<project_root>/Raw/sources/Datasheet/Vega/Vega_Standalone_Camera_UDP_User_Manual_1.3.pdf")
assets_dir = Path("<project_root>/tmp/e2e_assets")
assets_dir.mkdir(parents=True, exist_ok=True)

print(f"Parsing {pdf_path.name} via parse_file (sensitive=True)...")
t0 = time.time()
result = parse_file(pdf_path, assets_dir=assets_dir, is_sensitive=True)
elapsed = time.time() - t0

print(f"Done in {elapsed:.1f}s")
print(f"Title: {result.title}")
print(f"Doc type: {result.doc_type}")
print(f"Text length: {len(result.text)} chars")
print(f"Images: {len(result.images)}")
print(f"Tables: {len(result.tables)}")
print(f"Assets written: {len(list(assets_dir.iterdir()))}")
