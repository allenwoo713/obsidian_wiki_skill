"""Dump raw MinerU Cloud API response: full.md before mineru_common processing,
plus all extracted tables."""
import os, sys, tempfile, zipfile, json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from parsers.mineru_cloud import MineruCloudParser
from parsers.mineru_common import mineru_markdown_to_parse_result
from parsers.utils import slugify
import requests

PDF_PATH = Path("<project_root>/Raw/sources/Datasheet/Vega/Vega_Standalone_Camera_UDP_User_Manual_1.3.pdf")
OUTPUT_DIR = Path("<project_root>/tmp/mineru_cloud_debug")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

token = os.environ.get("MINERU_API_TOKEN")
parser = MineruCloudParser(api_token=token, model_version="vlm", language="en")

# Replicate parse() internals but save raw data
doc_slug = slugify(PDF_PATH.stem)
with tempfile.TemporaryDirectory() as tmp_dir:
    tmp_path = Path(tmp_dir)
    file_paths = [PDF_PATH]
    files_meta = [{"name": p.name, "data_id": p.stem} for p in file_paths]
    batch_id, file_urls = parser._request_upload_urls(files_meta)
    print(f"batch_id: {batch_id}")
    for fp, url in zip(file_paths, file_urls):
        parser._upload_file(fp, url)
        print(f"uploaded: {fp.name}")

    results = parser._poll_batch_results(batch_id, len(file_paths))
    print(f"poll done: {len(results)} results")

    for idx, fp in enumerate(file_paths):
        result = results[fp.name]
        extract_dir = tmp_path / f"extract_{idx:03d}"
        parser._download_and_extract_zip(result["full_zip_url"], extract_dir)

        # List all files in the zip
        print(f"\n=== Files in zip ===")
        for f in sorted(extract_dir.rglob("*")):
            if f.is_file():
                print(f"  {f.relative_to(extract_dir)} ({f.stat().st_size} bytes)")

        # Save raw full.md
        raw_md = (extract_dir / "full.md").read_text(encoding="utf-8")
        (OUTPUT_DIR / "raw_full.md").write_text(raw_md, encoding="utf-8")
        print(f"\nRaw full.md saved ({len(raw_md)} chars)")

        # Check if there's a layout.json
        json_files = list(extract_dir.glob("*.json"))
        for jf in json_files:
            print(f"JSON file: {jf.name} ({jf.stat().st_size} bytes)")
            (OUTPUT_DIR / jf.name).write_text(jf.read_text(encoding="utf-8"), encoding="utf-8")

        # Now process through mineru_common
        image_bytes_map = {}
        images_dir = extract_dir / "images"
        if images_dir.is_dir():
            for img in images_dir.iterdir():
                if img.is_file():
                    image_bytes_map[img.name] = img.read_bytes()

        parse_result = mineru_markdown_to_parse_result(
            markdown=raw_md,
            doc_slug=doc_slug,
            images_dir=tmp_path / "images",
            image_bytes_map=image_bytes_map,
        )

        # Save processed text
        (OUTPUT_DIR / "processed_text.md").write_text(parse_result.text, encoding="utf-8")

        # Save tables
        print(f"\n=== Extracted Tables ({len(parse_result.tables)} total) ===")
        tables_text = ""
        for i, table in enumerate(parse_result.tables):
            print(f"\n--- Table {i+1} ({len(table)} rows x {len(table[0]) if table else 0} cols) ---")
            for row in table:
                print(f"  {row}")
            tables_text += f"\n\n## [table {i+1}]\n"
            for row in table:
                tables_text += "| " + " | ".join(row) + " |\n"

        (OUTPUT_DIR / "tables.md").write_text(tables_text, encoding="utf-8")

        # Save images
        img_out = OUTPUT_DIR / "images"
        img_out.mkdir(exist_ok=True)
        for ref, img_bytes in zip(parse_result.images, parse_result._image_bytes):
            (img_out / ref.filename).write_bytes(img_bytes)

        print(f"\n=== Summary ===")
        print(f"  Raw markdown:   {len(raw_md)} chars")
        print(f"  Processed text: {len(parse_result.text)} chars")
        print(f"  Images:         {len(parse_result.images)}")
        print(f"  Tables:         {len(parse_result.tables)}")
        print(f"  Output dir:     {OUTPUT_DIR}")
