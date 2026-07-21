"""自动重建 Wiki/index.md（MOC 内容地图）。

根因修复（2026-07-21）：index.md 此前由人工维护，skill 管线不触碰它，
导致长期遗漏与结构漂移（最近一次手工更新停在 2026-06-29）。
本脚本作为确定性生成器，扫描 Wiki/ 下全部 *.md（排除受保护目录与自身），
按 frontmatter `type` 分组，重建 [[slug|title]] 链接列表。

设计对齐 wiki 既有语义：wiki 的原生分类轴就是 `type`（build_index/build_graph
均以 type 为一级分类键），故按 type 分组而非人工策展主题，保证零遗漏、完全确定。

用法：
    python build_index_md.py <project_root>
也可被 update_wiki.py 在增量更新末尾调用：build_wiki_index_md(project_root)
"""
from __future__ import annotations

import os
import re
import sys
from datetime import datetime
from pathlib import Path

import yaml

# 确保同目录下的 check_tags 可被 import（被 update_wiki.py 间接调用时也成立）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# type -> (中文分区标题, 排序权重)；权重小者靠前
_SECTION_ORDER = [
    ("product", "产品实体", 0),
    ("source-summary", "源摘要", 1),
    ("concept", "概念知识", 2),
    ("comparison", "对比分析", 3),
    ("overview", "元信息", 4),
    ("log", "元信息", 4),
]
_LABEL = {t: lbl for t, lbl, _ in _SECTION_ORDER}
_DEFAULT_LABEL = "其他"

_FM_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def _parse_frontmatter(text: str) -> dict:
    m = _FM_RE.match(text)
    if not m:
        return {}
    try:
        return yaml.safe_load(m.group(1)) or {}
    except Exception:
        return {}


def build_wiki_index_md(project_root) -> Path:
    """扫描 Wiki/ 下全部页面，按 type 分组重建 index.md。返回 index.md 路径。"""
    root = Path(project_root)
    wiki_dir = root / "Wiki"
    if not wiki_dir.is_dir():
        raise NotADirectoryError(f"Wiki 目录不存在: {wiki_dir}")

    # ISSUE: tag-hygiene —— 重建索引前自动修复非法标签（含空格等 Obsidian 拒绝的
    # 标签名，如 "Euro NCAP" / "AD 001" / "Assisted Driving"）。幂等；任何批次新增或
    # 编辑页面后跑 build_index_md.py 即顺带修干净，避免再出现"不被允许的标签名"。
    try:
        import check_tags
        fixes = check_tags.fix_invalid_tags(project_root)
        if fixes:
            print(f"[tag-hygiene] 自动修复 {len(fixes)} 处非法标签")
    except Exception as e:
        print(f"[WARN] tag 检查跳过: {e}")

    # type -> [(slug, display), ...]
    groups: dict[str, list[tuple[str, str]]] = {}
    warnings: list[str] = []

    for md in sorted(wiki_dir.rglob("*.md"), key=lambda p: str(p).lower()):
        parts = md.relative_to(wiki_dir).parts
        if ".obsidian" in parts or ".graph" in parts:
            continue
        if md.name == "index.md":
            continue
        text = md.read_text(encoding="utf-8")
        fm = _parse_frontmatter(text)
        ptype = fm.get("type")
        if not ptype:
            warnings.append(f"  缺 type: {md.relative_to(wiki_dir)}")
            ptype = "未分类"
        ptype = str(ptype)
        slug = md.stem
        display = str(fm.get("title") or slug)
        groups.setdefault(ptype, []).append((slug, display))

    # 组装分区：按既定顺序，同标签 type 合并（如 overview/log 同归 元信息）；
    # 未知 type 与 未分类 统一归入 其他 并置后。分区内按 slug 确定性排序。
    ordered_types = [t for t, _, _ in _SECTION_ORDER]
    section_map: dict[str, list[tuple[str, str]]] = {}
    for t in ordered_types:
        if t not in groups:
            continue
        lbl = _LABEL.get(t, t)
        section_map.setdefault(lbl, []).extend(groups[t])
    for t, items in groups.items():
        if t in ordered_types or t == "未分类":
            continue
        section_map.setdefault(_DEFAULT_LABEL, []).extend(items)
    if "未分类" in groups:
        section_map.setdefault(_DEFAULT_LABEL, []).extend(groups["未分类"])
    section_items = [
        (lbl, sorted(items, key=lambda x: x[0].lower()))
        for lbl, items in section_map.items()
        if items
    ]

    today = datetime.now().strftime("%Y-%m-%d")
    lines = [
        "---",
        "type: index",
        'title: "Wiki 索引"',
        f"updated: {today}",
        "---",
        "",
        "# Obsidian Wiki Builder 索引",
        "",
        "欢迎来到雷达产品知识库。本索引按页面类型（type）分组，列出全部 Wiki 页面。"
        "本文件由脚本 build_index_md.py 自动生成，请勿手改——任何页面增删后运行该脚本即可重建。",
        "",
    ]
    for label, items in section_items:
        lines.append(f"## {label}")
        lines.append("")
        for slug, display in items:
            lines.append(f"- [[{slug}|{display}]]")
        lines.append("")
    out = "\n".join(lines).rstrip() + "\n"

    index_path = wiki_dir / "index.md"
    index_path.write_text(out, encoding="utf-8")

    total = sum(len(v) for v in groups.values())
    print(f"index.md 已重建: {total} 页, 分区 {len(section_items)} 个")
    for label, items in section_items:
        print(f"  - {label}: {len(items)}")
    if warnings:
        print("警告（缺 type 字段，已归入 其他）:")
        for w in warnings:
            print(w)
    return index_path


def main():
    if len(sys.argv) < 2:
        print("用法: python build_index_md.py <project_root>")
        sys.exit(1)
    build_wiki_index_md(sys.argv[1])


if __name__ == "__main__":
    main()
