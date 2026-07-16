"""增量更新调度：扫描 Raw/sources → diff manifest → 输出 new/modified/deleted 列表。
用法：python update_wiki.py <project_root> [--apply]
"""
from __future__ import annotations
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Dict, Tuple

import _config  # noqa: F401  # 加载 <skill_dir>/.env（ISSUE-01，统一收口，替代下方原手写 load_dotenv）

from parse_sources import compute_sha256, parse_file


def norm_key(p) -> str:
    """归一化 manifest 键（Windows 路径大小写/分隔符不敏感）。

    根因修复：不同运行传入的项目根盘符大小写不一（D:\\ vs d:\\），
    导致 str(Path) 键逐字不匹配，误判全部源文档为 new、旧条目为 deleted，
    触发整库重解析。统一经 normcase+normpath 归一，消解大小写/分隔符差异。
    """
    return os.path.normcase(os.path.normpath(str(p)))


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
        if not f.is_file():
            continue
        # 跳过 Office 临时锁文件（Word/Excel 打开文档时生成的 ~$ 前缀文件）
        if f.name.startswith("~$") or f.name.startswith(".~"):
            continue
        if f.suffix.lower() in SUPPORTED_EXT:
            docs.append(f)
    return docs


def diff_manifest(current_docs: List[Path], manifest: dict) -> Dict[str, List[SourceDiff]]:
    entries: Dict[str, dict] = manifest.get("entries", {})
    current_paths = {norm_key(d) for d in current_docs}
    new, modified, unchanged, deleted = [], [], [], []
    for d in current_docs:
        key = norm_key(d)
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
    for d in diff["new"] + diff["modified"] + diff["unchanged"]:
        entries[norm_key(d.path)] = {
            "sha256": d.sha256,
            "status": "processed",
            "wiki_pages": wiki_pages_map.get(norm_key(d.path), []),
            "last_processed": now,
        }
    for d in diff["deleted"]:
        if norm_key(d.path) in entries:
            entries[norm_key(d.path)]["status"] = "deleted"


def build_wiki_pages_map(proj: Path) -> Dict[str, List[str]]:
    """扫描 Wiki/sources，建立 源文档路径 -> [wiki 页路径] 映射（依据 frontmatter sources[]）。"""
    wiki_dir = proj / "Wiki"
    mapping: Dict[str, List[str]] = {}
    if not wiki_dir.exists():
        return mapping
    _fm = re.compile(r"^---\n(.*?)\n---\n", re.DOTALL)
    import yaml
    for md in sorted(wiki_dir.rglob("*.md")):
        if ".graph" in md.parts:
            continue
        raw = md.read_text(encoding="utf-8", errors="replace")
        m = _fm.match(raw)
        if not m:
            continue
        try:
            fm = yaml.safe_load(m.group(1)) or {}
        except Exception:
            continue
        sources = fm.get("sources", []) or []
        if isinstance(sources, str):
            sources = [sources]
        for s in sources:
            key = norm_key(Path(proj / s))
            mapping.setdefault(key, []).append(str(md))
    return mapping


def main():
    # .env 已由顶部 `import _config` 统一加载（ISSUE-01）
    # ISSUE-06：argparse 替代手写 argv
    import argparse
    p = argparse.ArgumentParser(
        prog="update_wiki.py",
        description="增量更新调度：扫描 Raw/sources → diff manifest → 输出 new/modified/deleted 列表",
    )
    p.add_argument("project_root", help="知识库项目根目录（含 Raw/sources/ 与 .index/）")
    p.add_argument("--apply", action="store_true", help="执行全文落盘与 manifest 更新（默认仅 dry-run）")
    args = p.parse_args()
    proj = Path(args.project_root)
    apply = args.apply
    raw_sources = proj / "Raw" / "sources"
    idx_dir = proj / ".index"
    idx_dir.mkdir(exist_ok=True)
    manifest_file = idx_dir / "manifest.json"
    manifest = {"entries": {}}
    if manifest_file.exists():
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
        # 加载时按 norm_key 合并历史遗留的大小写重复键（如 D:\ 与 d:\ 指向同一文件），
        # 保留 processed 状态条目，避免 13 条 phantom deleted 残留
        _raw = manifest.get("entries", {})
        _collapsed: Dict[str, dict] = {}
        for _k, _v in _raw.items():
            _nk = norm_key(_k)
            _cur = _collapsed.get(_nk)
            if _cur is None or (_v.get("status") == "processed" and _cur.get("status") != "processed"):
                _collapsed[_nk] = _v
        manifest["entries"] = _collapsed
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
    # 确定性全文落盘：对 new/modified 文档，把 ParsedDoc.text 机械写入 source-summary 页
    # （脚本管理区），不经过 LLM，保证数据完整性
    if new_or_mod:
        print(f"\n全文落盘（确定性，非 LLM）:")
        for d in new_or_mod:
            try:
                parsed = parse_file(d.path, assets_dir=assets_dir)
                page = write_source_fulltext(proj, d.path, parsed)
                print(f"  -> {page.name}: {len(parsed.text)} chars, {len(parsed.images)} imgs")
            except Exception as e:
                print(f"  [WARN] 全文落盘失败 {d.path}: {e}")
    # 回填 entries（含 unchanged），使未来增量可用
    wiki_pages_map = build_wiki_pages_map(proj)
    update_manifest(diff, manifest, wiki_pages_map)
    print(f"\nmanifest entries: {len(manifest.get('entries', {}))}")
    (idx_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def extract_images_for_diff(new_or_modified, unchanged, assets_dir, existing_images=None):
    """重建图片清单，但保留全部旧条目的 caption（使历史标注不丢失）。

    - 以现有 manifest.images 为基础全量保留（含 caption_text/vlm_caption）。
    - new/modified 文档：重新解析并追加新条目，按 (source_doc, 去扩展名文件名)
      匹配旧条目合并 caption（兼容扩展名变化，如 .png→.jpg）。
    - unchanged 文档：旧条目已在基础中保留，无需额外处理。
    - 末尾按 (source_doc, 去扩展名) 去重，优先保留有 caption 的条目。
    """
    from parse_sources import parse_file
    import re as _re

    def _norm(fn: str) -> str:
        return _re.sub(r"\.\w+$", "", fn) if fn else ""

    # 1) 全量保留旧条目（含 caption）
    image_manifest = list(existing_images or [])

    # 2) 旧条目按 去扩展名文件名 索引（文件名全局唯一，规避 source_doc 绝对/相对路径格式不一致）
    old_index: Dict[str, dict] = {}
    for e in (existing_images or []):
        fn = e.get("filename") or Path(e.get("rel_path", "")).name
        if fn:
            old_index[_norm(fn)] = e

    # 3) new/modified：追加新条目并合并旧 caption
    for d in new_or_modified:
        p = d.path if hasattr(d, 'path') else d
        try:
            parsed = parse_file(p, assets_dir=assets_dir)
        except Exception as e:
            print(f"  [WARN] 图片提取跳过 {p}: {e}")
            continue
        for ref in parsed.images:
            prev = old_index.get(_norm(ref.filename))
            image_manifest.append({
                "filename": ref.filename, "rel_path": ref.rel_path,
                "sha256": ref.sha256, "source_doc": norm_key(p),
                "source_media": ref.source_media_name,
                "page_or_section": ref.page_or_section,
                "figure_caption": ref.caption or (prev.get("figure_caption", "") if prev else ""),
                "vlm_caption": prev.get("vlm_caption") if prev else None,
                "caption_text": prev.get("caption_text", "") if prev else (ref.caption or ""),
            })

    # 4) unchanged：旧条目已在第 1 步保留，跳过

    # 5) 去重：同一 (source_doc, 去扩展名) 仅保留一条，优先有 caption
    seen: Dict[tuple, int] = {}
    deduped = []
    for e in image_manifest:
        fn = e.get("filename") or Path(e.get("rel_path", "")).name
        key = (norm_key(e.get("source_doc", "")), _norm(fn))
        if key in seen:
            idx = seen[key]
            # 优先保留有 caption 的；若都有 caption，优先新条目（与重生成页面文件一致）
            if e.get("caption_text"):
                deduped[idx] = e
        else:
            seen[key] = len(deduped)
            deduped.append(e)
    return deduped


# === 确定性全文落盘（source-summary 页）===
# source-summary 页分两个管理区：
#   - 脚本管理区（AUTO-GENERATED 标记内）：全文内容 + 图片列表，由 ParsedDoc 机械写入
#   - LLM 管理区（标记外）：frontmatter / 核心内容摘要 / related 链接，由 Box 生成
# 数据完整性由脚本保证，LLM 批量处理的漏错风险隔离在衍生层。

_AUTO_FULLTEXT_BEGIN = "<!-- BEGIN AUTO-GENERATED FULLTEXT -->"
_AUTO_FULLTEXT_END = "<!-- END AUTO-GENERATED FULLTEXT -->"
_AUTO_IMAGES_BEGIN = "<!-- BEGIN AUTO-GENERATED IMAGES -->"
_AUTO_IMAGES_END = "<!-- END AUTO-GENERATED IMAGES -->"


def _slug_from_source(source_path: Path) -> str:
    from parsers.utils import slugify
    # slugify 保留 _（\w 含下划线），但 source-summary 页历史命名用全连字符
    # 归一化 _ → - 以匹配现有文件名（图片文件名仍用 slugify 原样，两套约定独立）
    return slugify(source_path.stem).replace("_", "-")


def _replace_auto_block(content: str, begin: str, end: str, new_block: str) -> str:
    """替换 begin...end 标记之间的内容为 new_block（含标记本身）。标记不存在则追加到末尾。"""
    pattern = re.compile(re.escape(begin) + r".*?" + re.escape(end), re.DOTALL)
    if pattern.search(content):
        # new_block 含文档原文，可能含反斜杠（如 C:\ 路径），若作为替换模板会被
        # re 解析转义而抛 `bad escape`。用 lambda 返回值作为字面量，跳过模板解析。
        return pattern.sub(lambda m: new_block, content)
    return content.rstrip() + "\n\n" + new_block + "\n"


def write_source_fulltext(proj: Path, source_path: Path, parsed, page_path: Path = None) -> Path:
    """将 ParsedDoc.text 确定性写入 source-summary 页的 AUTO-GENERATED 标记区。

    - 页面路径：page_path 或 Wiki/sources/{slug}.md
    - 全文区由脚本覆盖写入；frontmatter/摘要/related 由 LLM 管理，原样保留
    - 页面不存在时生成骨架（frontmatter 占位 + 标记区），Box 后续补摘要/related
    """
    wiki_sources = proj / "Wiki" / "sources"
    wiki_sources.mkdir(parents=True, exist_ok=True)
    if page_path is None:
        slug = _slug_from_source(source_path)
        page_path = wiki_sources / f"{slug}.md"

    fulltext_body = parsed.text.strip()
    fulltext_block = (
        f"{_AUTO_FULLTEXT_BEGIN}\n## 全文内容\n\n{fulltext_body}\n\n{_AUTO_FULLTEXT_END}"
    )

    today = datetime.now().strftime("%Y-%m-%d")

    if page_path.exists():
        content = page_path.read_text(encoding="utf-8")
        # 替换 FULLTEXT 区
        content = _replace_auto_block(content, _AUTO_FULLTEXT_BEGIN, _AUTO_FULLTEXT_END, fulltext_block)
        # 移除旧的 AUTO IMAGES 标记区（FULLTEXT 区已嵌入图片，IMAGES 区不再需要）
        content = re.sub(
            r"\n*<!-- BEGIN AUTO-GENERATED IMAGES -->.*?<!-- END AUTO-GENERATED IMAGES -->\n*",
            "\n\n",
            content,
            flags=re.DOTALL,
        )
        # 移除 LLM 管理区旧的"## 文档内嵌图片"段落（被 FULLTEXT 区嵌入替代）
        content = re.sub(
            r"## 文档内嵌图片\n\n!\[\[[^\]]+\]\][^\n]*\n",
            "",
            content,
        )
        # 清理多余空行
        content = re.sub(r"\n{3,}", "\n\n", content)
        # 更新日期
        content = re.sub(r"(updated:\s*)\d{4}-\d{2}-\d{2}", rf"\g<1>{today}", content)
        page_path.write_text(content, encoding="utf-8")
    else:
        rel_source = source_path.relative_to(proj).as_posix()
        title = parsed.title or source_path.stem
        skeleton = (
            f"---\n"
            f"type: source-summary\n"
            f'title: "{title}"\n'
            f'sources: ["{rel_source}"]\n'
            f"products: []\n"
            f"tags: []\n"
            f"related: []\n"
            f"updated: {today}\n"
            f"---\n\n"
            f"# {title}\n\n"
            f"## 文档信息\n\n"
            f"- **路径**: `{rel_source}`\n"
            f"- **格式**: {source_path.suffix.lstrip('.')}\n\n"
            f"## 核心内容摘要\n\n"
            f"<!-- TODO: Box 补充导航性摘要 -->\n\n"
            f"{fulltext_block}\n"
        )
        page_path.write_text(skeleton, encoding="utf-8")
    return page_path


if __name__ == "__main__":
    main()
