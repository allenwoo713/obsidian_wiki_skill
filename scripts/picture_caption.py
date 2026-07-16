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

import _config  # noqa: F401  # 加载 <skill_dir>/.env（ISSUE-01），统一入口脚本行为


def load_manifest(project_root: Path) -> Dict:
    manifest_file = project_root / ".index" / "manifest.json"
    with open(manifest_file, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(project_root: Path, manifest: Dict):
    manifest_file = project_root / ".index" / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def list_pending(project_root: Path, limit: int = None) -> List[Dict]:
    """列出所有 caption_text 为空的图片，输出简略 JSON 供 Box 填充。

    stdout 输出 pending JSON 数组（供 apply 消费）；stderr 输出 total/done/pending
    统计 + 按 source_doc 分组，防止调用方把 pending 切片误当成全集。
    """
    manifest = load_manifest(project_root)
    all_imgs = manifest.get("images", [])
    pending = []
    per_doc: Dict[str, int] = {}
    for img in all_imgs:
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
            doc = img.get("source_doc", "(unknown)")
            # 取文件名部分，避免绝对路径刷屏
            short = doc.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] or doc
            per_doc[short] = per_doc.get(short, 0) + 1

    total = len(all_imgs)
    done = total - len(pending)
    # 统计走 stderr，不污染 stdout 的 JSON
    print(f"[caption 统计] 总图: {total}, 已标注: {done}, 待标注: {len(pending)}", file=sys.stderr)
    if per_doc:
        print("[caption 待标注按文档]:", file=sys.stderr)
        for doc, n in sorted(per_doc.items(), key=lambda x: -x[1]):
            print(f"    {n:>3}  {doc}", file=sys.stderr)

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
    healed = 0
    for img in manifest.get("images", []):
        cap = by_filename.get(img["filename"])
        if cap:
            img["vlm_caption"] = cap.get("vlm_caption", {})
            # caption_text 是 build_index 唯一读取的检索字段；为空时自动从
            # vlm_caption.description 回填，避免"填了 VLM 却没进检索"静默发生。
            ct = (cap.get("caption_text") or "").strip()
            if not ct:
                vlm = cap.get("vlm_caption") or {}
                desc = (vlm.get("description") or "").strip()
                if desc:
                    parts = [desc]
                    kvs = vlm.get("key_values") or []
                    if kvs:
                        parts.append("关键词: " + ", ".join(str(k) for k in kvs))
                    ct = "\n".join(parts)
                    healed += 1
            img["caption_text"] = ct
            updated += 1

    save_manifest(project_root, manifest)

    # 统计
    total = len(manifest.get("images", []))
    done = sum(1 for i in manifest["images"] if i.get("caption_text", "").strip())
    print(f"更新 {updated} 条（其中 caption_text 自愈回填 {healed} 条）。总计: {total}, 已标注: {done}, 待标注: {total - done}")


def main():
    # ISSUE-06：argparse 子命令替代手写 argv
    import argparse
    p = argparse.ArgumentParser(
        prog="picture_caption.py",
        description="图片 caption 管理：列出待标注 / 写入 caption 到 manifest",
    )
    p.add_argument("project_root", help="知识库项目根目录")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="输出待标注图片 JSON 清单")
    p_list.add_argument("--limit", type=int, default=None, help="最多列出 N 张")

    p_apply = sub.add_parser("apply", help="将 captions JSON 写回 manifest")
    p_apply.add_argument("captions_json", help="captions JSON 文件路径")

    args = p.parse_args()
    proj = Path(args.project_root)

    if args.cmd == "list":
        list_pending(proj, limit=args.limit)
    elif args.cmd == "apply":
        apply_captions(proj, Path(args.captions_json))


if __name__ == "__main__":
    main()
