"""MinerU Cloud API 真实端到端测试。

用法：
    cd <home>/.workbuddy/skills/obsidian_wiki_skill
    C:/Python313/python.exe tests/e2e_cloud_real.py
"""
import os
import sys
import time
from pathlib import Path

# 加载 .env
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from parsers.mineru_cloud import MineruCloudParser

PDF_PATH = Path("<project_root>/Raw/sources/Datasheet/Vega/Vega_Standalone_Camera_UDP_User_Manual_1.3.pdf")
OUTPUT_DIR = Path("<project_root>/tmp/mineru_cloud_e2e")

print("=" * 60)
print("MinerU Cloud API E2E Test")
print(f"  Input:  {PDF_PATH}")
print(f"  Output: {OUTPUT_DIR}")
print(f"  Token:  {'YES' if os.environ.get('MINERU_API_TOKEN') else 'NO'}")
print("=" * 60)

if not PDF_PATH.exists():
    print(f"ERROR: PDF not found: {PDF_PATH}")
    sys.exit(1)

token = os.environ.get("MINERU_API_TOKEN")
if not token:
    print("ERROR: MINERU_API_TOKEN not set")
    sys.exit(1)

start = time.time()
try:
    parser = MineruCloudParser(
        api_token=token,
        model_version="vlm",
        language="en",
        max_pages_per_file=200,
        poll_interval=5,
        max_poll=120,
    )
    print(f"\n[1/3] Parser initialized. model_version=vlm, language=en")
    print(f"[2/3] Calling parse()... (this uploads to OSS + polls MinerU Cloud)")

    result = parser.parse(PDF_PATH)
    elapsed = time.time() - start

    print(f"[3/3] Parse complete in {elapsed:.1f}s")
    print(f"\n--- Result Summary ---")
    print(f"  Text length:     {len(result.text)} chars")
    print(f"  Images:          {len(result.images)}")
    print(f"  Tables:          {len(result.tables)}")
    print(f"  Image bytes:     {len(result._image_bytes)}")

    # 落盘结果
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    md_path = OUTPUT_DIR / "cloud_result.md"
    md_path.write_text(result.text, encoding="utf-8")
    print(f"\n  Markdown saved:  {md_path}")

    # 落盘图片
    img_dir = OUTPUT_DIR / "images"
    img_dir.mkdir(exist_ok=True)
    for ref, img_bytes in zip(result.images, result._image_bytes):
        (img_dir / ref.filename).write_bytes(img_bytes)
    print(f"  Images saved:    {img_dir} ({len(result.images)} files)")

    # 打印前 80 行 markdown
    print(f"\n--- Markdown Preview (first 80 lines) ---")
    for i, line in enumerate(result.text.split("\n")[:80], 1):
        print(f"  {i:3d} | {line}")

    print(f"\n{'='*60}")
    print(f"E2E Cloud Test: SUCCESS ({elapsed:.1f}s)")
    print(f"{'='*60}")

except Exception as e:
    elapsed = time.time() - start
    import traceback
    print(f"\n{'='*60}")
    print(f"E2E Cloud Test: FAILED ({elapsed:.1f}s)")
    print(f"{'='*60}")
    traceback.print_exc()
    sys.exit(1)
