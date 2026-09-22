"""Check Wiki Markdown against the verified active index; no embedding/model load.

Issue #65 退出码约定（与 check_ann_drift.py 一致）：
  0 = 检查范围内可证明一致（未构建的可选本地图谱为 not_built，不阻塞纯文本检查）
  1 = 已确认内容/路径变化（stale）
  2 = 不能证明一致（unknown：旧 manifest 无 provenance、schema 不支持、扫描失败、
      graph 损坏或 active index 无法解析；聚合检查时 unknown 优先于 stale）

用法：
  python scripts/check_index_staleness.py <project_root> --json
  python scripts/check_index_staleness.py <project_root> --json --no-graph

不要为了检查旧索引而就地补写 provenance；缺少信息返回 2 是设计行为，需要正常 rebuild。
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

from obsidian_wiki.application.wiki_freshness import (
    collect_reports, diagnostic_exit_code, read_graph_payload, unknown,
)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("project_root", type=Path)
    p.add_argument("--json", action="store_true", dest="as_json")
    p.add_argument("--no-graph", action="store_true",
                   help="Only check the text index; do not claim graph freshness")
    args = p.parse_args(argv)
    root = args.project_root.resolve()
    try:
        # Same validated resolver/recovery path as production, not hand-parsed ACTIVE_INDEX.
        from obsidian_wiki.application.active_index_pointer import resolve_active_lance_dir
        lance_dir = resolve_active_lance_dir(root / ".index")
        manifest = json.loads((lance_dir.parent / "manifest.json").read_bytes())
        if not isinstance(manifest, dict):
            raise ValueError("manifest_not_object")
        graph, error = read_graph_payload(root / ".index") if not args.no_graph else (None, None)
        reports = collect_reports(
            root / "Wiki", manifest, graph, graph_error=error,
            include_graph=not args.no_graph,
        )
    except (OSError, RuntimeError, ValueError, UnicodeError) as exc:
        reports = {"index": unknown(f"active_index_unavailable:{type(exc).__name__}:{exc}")}
    code = diagnostic_exit_code(reports)
    output = {"exit_code": code, "components": reports,
              "scope": "wiki_markdown_only",
              "check_semantics": "observed content equality, not filesystem snapshot isolation"}
    print(json.dumps(output, ensure_ascii=False, indent=2))
    if code and not args.as_json:
        print("Rebuild affected artifacts; unknown requires diagnosis/rebuild, not a fresh stamp.",
              file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
