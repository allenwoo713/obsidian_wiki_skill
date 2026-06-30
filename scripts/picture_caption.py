"""图片 caption 工具：列出待标注图片、写入 caption 到 manifest。

用法：
  python picture_caption.py <project_root> list [--limit N]
      输出 JSON 格式的待标注图片清单（每项含 filename/rel_path/figure_caption/source_doc）
      Box 读图后填充 vlm_caption 和 caption_text 字段。

  python picture_caption.py <project_root> apply <captions_json>
      将 captions JSON 中的 caption 写回 manifest.json。

设计原则：Box 不直接改 manifest。Box 读 list 输出 + 读每张图 → 生成 captions JSON → apply 写回。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from typing import List, Dict


def load_manifest(project_root: Path) -> Dict:
    manifest_file = project_root / ".index" / "manifest.json"
    with open(manifest_file, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(project_root: Path, manifest: Dict):
    manifest_file = project_root / ".index" / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def list_pending(project_root: Path, limit: int = None) -> List[Dict]:
    """列出所有 caption_text 为空的图片，输出简略 JSON 供 Box 填充。"""
    manifest = load_manifest(project_root)
    pending = []
    for img in manifest.get("images", []):
        if not img.get("caption_text", "").strip():
            pending.append({
                "filename": img["filename"],
                "rel_path": img.get("rel_path", f"assets/{img['filename']}"),
                "figure_caption": img.get("figure_caption", ""),
                "source_doc": img.get("source_doc", ""),
                # 以下由 Box 填充
                "vlm_caption": {
                    "description": "",
                    "key_values": [],
                    "category": "",
                },
                "caption_text": "",
            })

    if limit:
        pending = pending[:limit]

    print(json.dumps(pending, ensure_ascii=False, indent=2))
    return pending


def apply_captions(project_root: Path, captions_file: Path):
    """将 captions JSON 中的 caption 写回 manifest 对应的 image 条目。"""
    with open(captions_file, "r", encoding="utf-8") as f:
        captions = json.load(f)

    manifest = load_manifest(project_root)
    by_filename = {c["filename"]: c for c in captions}

    updated = 0
    for img in manifest.get("images", []):
        cap = by_filename.get(img["filename"])
        if cap:
            img["vlm_caption"] = cap.get("vlm_caption", {})
            img["caption_text"] = cap.get("caption_text", "")
            updated += 1

    save_manifest(project_root, manifest)

    # 统计
    total = len(manifest.get("images", []))
    done = sum(1 for i in manifest["images"] if i.get("caption_text", "").strip())
    print(f"更新 {updated} 条。总计: {total}, 已标注: {done}, 待标注: {total - done}")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    proj = Path(sys.argv[1])
    cmd = sys.argv[2]

    if cmd == "list":
        limit = None
        for i, arg in enumerate(sys.argv):
            if arg == "--limit" and i + 1 < len(sys.argv):
                limit = int(sys.argv[i + 1])
        list_pending(proj, limit=limit)

    elif cmd == "apply":
        if len(sys.argv) < 4:
            print("用法: python picture_caption.py <project_root> apply <captions.json>")
            sys.exit(1)
        captions_path = Path(sys.argv[3])
        apply_captions(proj, captions_path)


if __name__ == "__main__":
    main()
