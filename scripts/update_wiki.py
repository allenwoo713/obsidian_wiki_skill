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


SUPPORTED_EXT = {".docx", ".pdf", ".md", ".markdown", ".txt"}


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


if __name__ == "__main__":
    main()
