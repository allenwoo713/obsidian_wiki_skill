"""全库图片注册审计（常驻诊断工具，只读不改 manifest 除非 --fix）。

设计目标：
- 扫描 Wiki/**/*.md 下所有 ![[ref]] 引用（含实体页/概念页/源页，比仅扫 Wiki/sources 更全）
- 与磁盘 Wiki/assets/ 及 manifest.images 对账
- 报告：已注册 / 引用但未注册(磁盘有) / 引用但磁盘缺失 / 未引用资产(孤儿)
- status 回填：老条目无 status 时按 caption_text 推导 captioned / pending_vlm
- --fix 时把"引用但未注册(磁盘有)"写入 manifest.images（幂等，status=pending_vlm）

用法（默认报告模式，不改任何文件）：
  python audit_images.py <project_root>
  python audit_images.py <project_root> --fix

注意：本工具只补"注册缺口"，不填 caption、不进索引。VLM 解读由
picture_caption.py 完成，索引由 build_index.py 完成。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List
from image_provenance import normalize_embed, resolve_image_references

WIKILINK = re.compile(r"!\[\[([^\]]+)\]\]")


def norm_key(p) -> str:
    s = str(Path(p).resolve())
    if len(s) >= 2 and s[1] == ":":
        s = s[0].upper() + s[1:]
    return s.replace("/", "\\")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description="全库图片注册审计（只读，--fix 才补登）")
    ap.add_argument("project_root", nargs="?", default=".",
                    help="知识库项目根目录（含 Wiki/ 与 .index/），默认当前目录")
    ap.add_argument("--fix", action="store_true",
                    help="兼容别名：等同 --repair-registration（显式写模式）")
    ap.add_argument("--repair-registration", action="store_true",
                    help="显式写模式：补登记引用但未注册且磁盘存在的图片")
    ap.add_argument("--migrate-status", action="store_true",
                    help="显式写模式：迁移 legacy 条目的 status 字段")
    args = ap.parse_args()

    proj = Path(args.project_root).resolve()
    manifest_path = proj / ".index" / "manifest.json"
    assets_dir = proj / "Wiki" / "assets"

    if not manifest_path.exists():
        print(f"[ERROR] 找不到 manifest: {manifest_path}")
        raise SystemExit(2)
    if not assets_dir.is_dir():
        print(f"[ERROR] 找不到 assets 目录: {assets_dir}")
        raise SystemExit(2)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    images = manifest.get("images", [])

    # status 回填：老条目无 status 时按 caption_text 推导
    #   "captioned"   -> caption_text 非空（已 VLM/已 caption）
    #   "pending_vlm" -> caption_text 为空且已注册（待 VLM 解读）
    # 回填确保 picture_caption.py list 可按 status filter 找出待处理项
    migrated = 0
    for img in images:
        if args.migrate_status and not img.get("status"):
            img["status"] = "captioned" if (img.get("caption_text") or "").strip() else "pending_vlm"
            migrated += 1

    resolutions = resolve_image_references(proj, images)
    registered = {normalize_embed(e.get("rel_path") or e.get("filename", "")) for e in images}

    # 扫描 Wiki/**/*.md（排除 .obsidian / .graph 维护目录）
    md_files = [p for p in (proj / "Wiki").rglob("*.md")
                if ".obsidian" not in p.parts and ".graph" not in p.parts]
    refs: Dict[str, List[str]] = {}
    for md in md_files:
        try:
            txt = md.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for ref in WIKILINK.findall(txt):
            fn = normalize_embed(ref)
            if fn:
                refs.setdefault(fn, []).append(md.relative_to(proj).as_posix())

    disk = {p.relative_to(proj / "Wiki").as_posix() for p in assets_dir.rglob("*") if p.is_file()}
    # Windows 大小写不敏感：建小写映射，避免 6t8r_brief vs 6T8R_Brief 误报缺失
    disk_ci = {n.lower(): n for n in disk}
    reg_ci = {n.lower(): n for n in registered}

    referenced = set(refs.keys())
    referenced_ci = {f.lower() for f in referenced}
    unreg_present = sorted(f for f in referenced
                           if f.lower() in disk_ci and f.lower() not in reg_ci)
    missing_disk = sorted(f for f in referenced if f.lower() not in disk_ci)
    case_mismatch = sorted(f for f in referenced
                           if f.lower() in disk_ci and disk_ci[f.lower()] != f)
    orphan = sorted(f for f in disk
                    if f.lower() not in reg_ci and f.lower() not in referenced_ci)

    print(f"manifest 已注册图片 : {len(images)}")
    print(f"扫描 md 文件数      : {len(md_files)}")
    print(f"去重引用图片数      : {len(referenced)}")
    print(f"  已注册            : {len(referenced_ci & set(reg_ci.keys()))}")
    print(f"  引用但未注册(磁盘有): {len(unreg_present)}   <-- 可补登记")
    print(f"  引用但磁盘缺失     : {len(missing_disk)}   <-- 真断链，需人工")
    print(f"  仅大小写不符(Windows下正常): {len(case_mismatch)}   <-- 误报过滤")
    print(f"未引用资产(磁盘有, 未注册未引用): {len(orphan)}   <-- 需人工判断用途(logo/重复提取/真孤儿, 不可一概而论)")
    pending_vlm = sum(1 for e in images if e.get("status") == "pending_vlm")
    captioned = sum(1 for e in images if e.get("status") == "captioned")
    print(f"status 分布  : pending_vlm={pending_vlm}  captioned={captioned}  (回填 {migrated})")
    print("-" * 60)
    if unreg_present:
        print(f"[未注册-磁盘有] 共 {len(unreg_present)} 张:")
        for f in unreg_present:
            print(f"    {f}  <-  {', '.join(refs[f][:2])}{' ...' if len(refs[f]) > 2 else ''}")
    if missing_disk:
        print(f"[真断链-磁盘缺失] 共 {len(missing_disk)} 张:")
        for f in missing_disk[:50]:
            print(f"    {f}  <-  {', '.join(refs[f][:2])}")
    if case_mismatch:
        print(f"[仅大小写不符] 共 {len(case_mismatch)} 张 (前 10):")
        for f in case_mismatch[:10]:
            print(f"    md引用 {f}  -> 磁盘 {disk_ci[f.lower()]}")
    if orphan:
        print(f"[孤儿资产] 共 {len(orphan)} 张 (前 30):")
        for f in orphan[:30]:
            print(f"    {f}")

    if (args.fix or args.repair_registration) and unreg_present:
        added = 0
        for f in unreg_present:
            resolution = resolutions[f]
            source_doc = resolution.source_doc
            source_media = Path(source_doc).stem
            asset = proj / "Wiki" / f
            entry = {
                "filename": Path(f).name,
                "rel_path": f,
                "sha256": sha256_file(asset),
                "source_doc": source_doc,
                "source_media": source_media,
                "page_or_section": "",
                "figure_caption": "",
                "vlm_caption": None,
                "caption_text": "",
                "status": "pending_vlm",
                "parent_page_id": resolution.parent_page_id,
                "referring_wiki_pages": list(resolution.referring_wiki_pages),
                "source_candidates": list(resolution.source_candidates),
            }
            images.append(entry)
            added += 1
        manifest["images"] = images
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[FIX] 已补登记 {added} 张 -> manifest.images 现 {len(images)}")
    elif args.migrate_status and migrated > 0:
        # 即使未 --fix，也把回填后的 status 落盘（幂等：无新增则跳过）
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[MIGRATE] status 回填 {migrated} 条已落盘")


if __name__ == "__main__":
    main()
