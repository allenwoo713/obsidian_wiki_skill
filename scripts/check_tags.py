"""检查并自动修复 Wiki 页面 frontmatter `tags` 中的非法标签名。

Obsidian 规则：单个 tag 值不能含空格（也不能含 `#` 等）。本脚本扫描
`Wiki/**/*.md` 的 `tags:` 行，检测任何含空白字符的 tag 值，将其中的空白符
统一替换为连字符 `-`，并应用已知别名规范化（如 `c-ncap` -> `C-NCAP`）。
幂等，可重复运行；不触碰正文/标题里的普通空格短语。

用法：
    python check_tags.py <project_root> [--check]   # --check 只报告不修改
也可被 build_index_md.py 在重建索引前调用：fix_invalid_tags(project_root)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# 已知别名规范化（小写/不一致写法 -> 规范形式）
_ALIASES = {
    "c-ncap": "C-NCAP",
    "euro-ncap": "Euro-NCAP",
}

# 匹配 tags 行（允许前导缩进），捕获缩进与数组体
_FM_TAGS_RE = re.compile(r'^(\s*)tags:\s*(.*)$')
# 提取 tags 行内各 "..." 或 '...' 标记值
_TOKEN_RE = re.compile(r'"((?:[^"\\]|\\.)*)"|\'((?:[^\'\\]|\\.)*)\'')


def _normalize_tag(tag: str) -> str:
    # 1) 任意空白符（空格/制表/全角空格等）统一 -> 连字符
    t = re.sub(r'\s+', '-', tag)
    # 2) 去除 Obsidian 不允许的字符（如 #）
    t = t.replace('#', '')
    # 3) 已知别名规范化
    t = _ALIASES.get(t, t)
    return t


def fix_invalid_tags(project_root, autofix: bool = True):
    """扫描并（可选）修复非法标签。返回 [(相对路径, 原行, 新行), ...]。"""
    root = Path(project_root) / "Wiki"
    if not root.is_dir():
        raise NotADirectoryError(f"Wiki 目录不存在: {root}")
    changes = []
    for p in sorted(root.rglob("*.md")):
        if ".obsidian" in p.parts or ".graph" in p.parts:
            continue
        text = p.read_text(encoding="utf-8")
        lines = text.split("\n")
        new_lines = []
        file_changed = False
        for line in lines:
            m = _FM_TAGS_RE.match(line)
            if not m:
                new_lines.append(line)
                continue
            indent, body = m.group(1), m.group(2)
            toks = _TOKEN_RE.findall(body)
            if not toks:
                new_lines.append(line)
                continue
            new_vals = []
            line_changed = False
            for dq, sq in toks:
                raw = dq if dq != "" else sq
                norm = _normalize_tag(raw)
                if norm != raw:
                    line_changed = True
                new_vals.append(norm)
            if line_changed:
                new_line = indent + "tags: [" + ", ".join(f'"{v}"' for v in new_vals) + "]"
                changes.append((str(p.relative_to(root)), line.strip(), new_line))
                file_changed = True
                line = new_line
            new_lines.append(line)
        if file_changed and autofix:
            p.write_text("\n".join(new_lines), encoding="utf-8")
    return changes


def main():
    if len(sys.argv) < 2:
        print("用法: python check_tags.py <project_root> [--check]")
        sys.exit(1)
    check_only = "--check" in sys.argv
    proj = sys.argv[1]
    changes = fix_invalid_tags(proj, autofix=not check_only)
    if not changes:
        print("OK: 无非法标签，无需修复。")
        return
    verb = "（仅报告，未修改）" if check_only else "（已自动修复）"
    print(f"发现 {len(changes)} 处标签需修复{verb}:")
    for rel, old, new in changes:
        print(f"  {rel}")
        print(f"    - {old}")
        print(f"    + {new}")


if __name__ == "__main__":
    main()
