"""Build and atomically publish validated version-2 community reports."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import _config  # noqa: F401 - load the configured local model path before composition

from obsidian_wiki.application.community_report_service import CommunityReportService
from obsidian_wiki.infrastructure.filesystem_community_reports import FilesystemCommunityReportStore
from obsidian_wiki.infrastructure.filesystem_graph_snapshot import FilesystemGraphSnapshot
from obsidian_wiki.infrastructure.production_token_counter import LocalReportTokenCounter, TokenCounterUnavailable


SKILL_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TOKENIZER_DIR = SKILL_ROOT / "models" / "paraphrase-multilingual-MiniLM-L12-v2"


def compose_report_service(project_root: Path) -> CommunityReportService:
    """Compose only the real filesystem and strict-local-tokenizer adapters."""
    root = Path(project_root)
    tokenizer_dir = Path(os.environ.get("WIKI_EMBEDDER_LOCAL_PATH") or DEFAULT_TOKENIZER_DIR)
    return CommunityReportService(
        FilesystemCommunityReportStore(root / ".index"),
        FilesystemGraphSnapshot(root),
        LocalReportTokenCounter(tokenizer_dir),
    )


def build_community_reports(project_root: Path):
    """Build a complete staged set, reopen it, then atomically select it."""
    return compose_report_service(Path(project_root)).build()


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="build_community_reports.py", description="构建并原子发布 schema-v2 社区报告"
    )
    parser.add_argument("project_root", help="知识库项目根目录（含 Wiki/ 和 .index/graph.json）")
    args = parser.parse_args()
    try:
        manifest = build_community_reports(Path(args.project_root))
    except TokenCounterUnavailable as exc:
        print(str(exc))
        return 2
    except RuntimeError as exc:
        print(f"community report build failed: {exc}")
        return 1
    print(f"社区报告构建完成: {manifest.report_count} 个社区 → {manifest.build_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
