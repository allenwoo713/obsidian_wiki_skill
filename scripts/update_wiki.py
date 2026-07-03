"""增量更新调度：扫描 Raw/sources → diff manifest → 输出 new/modified/deleted 列表。
用法：python update_wiki.py <project_root> [--apply]
"""
from __future__ import annotations
import json
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Dict, Tuple

from parse_sources import compute_sha256, parse_file


class SourceState(Enum):
    NEW = "new"
    MODIFIED = "modified"
    UNCHANGED = "unchanged"
    DELETED = "deleted"


@dataclass
class SourceDiff:
    path: Path
    sha256: str
    state: SourceState


SUPPORTED_EXT = {
    ".docx",
    ".pdf",
    ".pptx",
    ".doc",
    ".ppt",
    ".xls",
    ".xlsx",
    ".html",
    ".htm",
    ".md",
    ".markdown",
    ".txt",
}


def scan_sources(raw_sources_dir: Path) -> List[Path]:
    docs = []
    for f in sorted(raw_sources_dir.rglob("*")):
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXT:
            docs.append(f)
    return docs


def diff_manifest(current_docs: List[Path], manifest: dict) -> Dict[str, List[SourceDiff]]:
    entries: Dict[str, dict] = manifest.get("entries", {})
    current_paths = {str(d) for d in current_docs}
    new, modified, unchanged, deleted = [], [], [], []
    for d in current_docs:
        key = str(d)
        sha = compute_sha256(d)
        entry = entries.get(key)
        if entry is None:
            new.append(SourceDiff(d, sha, SourceState.NEW))
        elif entry.get("sha256") != sha:
            modified.append(SourceDiff(d, sha, SourceState.MODIFIED))
        else:
            unchanged.append(SourceDiff(d, sha, SourceState.UNCHANGED))
    for key, entry in entries.items():
        if key not in current_paths and entry.get("status") == "processed":
            deleted.append(SourceDiff(Path(key), "", SourceState.DELETED))
    return {"new": new, "modified": modified, "unchanged": unchanged, "deleted": deleted}


def update_manifest(diff: Dict[str, List[SourceDiff]], manifest: dict, wiki_pages_map: Dict[str, List[str]]):
    now = datetime.now().isoformat()
    entries = manifest.setdefault("entries", {})
    for d in diff["new"] + diff["modified"]:
        entries[str(d.path)] = {
            "sha256": d.sha256,
            "status": "processed",
            "wiki_pages": wiki_pages_map.get(str(d.path), []),
            "last_processed": now,
        }
    for d in diff["deleted"]:
        if str(d.path) in entries:
            entries[str(d.path)]["status"] = "deleted"


def main():
    if len(sys.argv) < 2:
        print("用法: python update_wiki.py <project_root> [--apply]")
        sys.exit(1)
    proj = Path(sys.argv[1])
    apply = "--apply" in sys.argv
    raw_sources = proj / "Raw" / "sources"
    idx_dir = proj / ".index"
    idx_dir.mkdir(exist_ok=True)
    manifest_file = idx_dir / "manifest.json"
    manifest = {"entries": {}}
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    docs = scan_sources(raw_sources)
    diff = diff_manifest(docs, manifest)
    print(f"扫描完成: {len(docs)} 源文档")
    print(f"  新增: {len(diff['new'])}")
    print(f"  修改: {len(diff['modified'])}")
    print(f"  未变: {len(diff['unchanged'])}")
    print(f"  删除: {len(diff['deleted'])}")
    if diff["new"]:
        print("\n新增文档（需 Box 生成 wiki 页）:")
        for d in diff["new"]:
            print(f"  + {d.path}")
    if diff["modified"]:
        print("\n修改文档（需 Box 重新生成 wiki 页）:")
        for d in diff["modified"]:
            print(f"  * {d.path}")
    if diff["deleted"]:
        print("\n删除文档（需 Box 清理关联 wiki 页）:")
        for d in diff["deleted"]:
            print(f"  - {d.path}")
    if not (diff["new"] or diff["modified"] or diff["deleted"]):
        print("\n无变更，零开销跳过。")

    assets_dir = proj / "Wiki" / "assets"
    existing_images = manifest.get("images", [])
    new_or_mod = diff["new"] + diff["modified"]
    image_manifest = extract_images_for_diff(new_or_mod, diff["unchanged"], assets_dir, existing_images)
    need_caption = [img for img in image_manifest if not img.get("caption_text")]
    if need_caption:
        print(f"\n需 Box 生成 caption 的图片: {len(need_caption)} 张")
        for img in need_caption[:5]:
            print(f"  - {img['rel_path']} (源: {img['source_doc']})")
    manifest["images"] = image_manifest
    (idx_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def extract_images_for_diff(new_or_modified, unchanged, assets_dir, existing_images=None):
    from parse_sources import parse_file
    image_manifest = []
    existing_by_path = {e["path"]: e for e in (existing_images or [])}
    for d in new_or_modified:
        p = d.path if hasattr(d, 'path') else d
        try:
            parsed = parse_file(p, assets_dir=assets_dir)
        except Exception as e:
            print(f"  [WARN] 图片提取跳过 {p}: {e}")
            continue
        for ref in parsed.images:
            image_manifest.append({
                "filename": ref.filename, "rel_path": ref.rel_path,
                "sha256": ref.sha256, "source_doc": str(p),
                "source_media": ref.source_media_name,
                "page_or_section": ref.page_or_section,
                "figure_caption": ref.caption, "vlm_caption": None, "caption_text": "",
            })
    for d in unchanged:
        p = d.path if hasattr(d, 'path') else d
        existing = existing_by_path.get(str(p))
        if existing:
            for img in existing.get("images", []):
                image_manifest.append({
                    "filename": img["filename"], "rel_path": img.get("rel_path", f"assets/{img['filename']}"),
                    "sha256": img["sha256"], "source_doc": str(p),
                    "source_media": img.get("source_media", ""),
                    "page_or_section": img.get("page_or_section", ""),
                    "figure_caption": img.get("figure_caption", ""),
                    "vlm_caption": img.get("vlm_caption"),
                    "caption_text": img.get("caption_text", ""),
                })
    return image_manifest


if __name__ == "__main__":
    main()
