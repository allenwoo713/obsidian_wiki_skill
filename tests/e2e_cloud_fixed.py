"""MinerU Cloud API E2E — fixed output with tables inline."""
import os, sys, time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from parsers.mineru_cloud import MineruCloudParser

PDF_PATH = Path("<project_root>/Raw/sources/Datasheet/Vega/Vega_Standalone_Camera_UDP_User_Manual_1.3.pdf")
OUTPUT_DIR = Path("<project_root>/tmp/mineru_cloud_e2e_fixed")

print("=" * 60)
print("MinerU Cloud API E2E Test (fixed: tables kept inline)")
print("=" * 60)

token = os.environ.get("MINERU_API_TOKEN")
parser = MineruCloudParser(api_token=token, model_version="vlm", language="en")

start = time.time()
result = parser.parse(PDF_PATH)
elapsed = time.time() - start

print(f"Parse complete: {elapsed:.1f}s")
print(f"  Text length: {len(result.text)} chars")
print(f"  Images: {len(result.images)}")
print(f"  Tables: {len(result.tables)}")

# 写完整结果（text 现在包含 HTML 表格）
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / "cloud_result.md").write_text(result.text, encoding="utf-8")

img_dir = OUTPUT_DIR / "images"
img_dir.mkdir(exist_ok=True)
for ref, img_bytes in zip(result.images, result._image_bytes):
    (img_dir / ref.filename).write_bytes(img_bytes)

print(f"  Output: {OUTPUT_DIR / 'cloud_result.md'}")
