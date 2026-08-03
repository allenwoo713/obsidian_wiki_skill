"""Filesystem graph and Markdown facts for the community-report service."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from obsidian_wiki.domain.community_report_models import GraphEdge, GraphSnapshotState, PageSnapshot


class FilesystemGraphSnapshot:
    def __init__(self, project_root: Path):
        self._project_root = Path(project_root)

    def read(self) -> GraphSnapshotState:
        graph_path = self._project_root / ".index" / "graph.json"
        try:
            payload = json.loads(graph_path.read_text(encoding="utf-8"))
            nodes, edges, communities = payload["nodes"], payload["edges"], payload["communities"]
            if not isinstance(nodes, list) or not isinstance(edges, list) or not isinstance(communities, list):
                raise ValueError("invalid graph snapshot")
            pages = tuple(PageSnapshot(str(node["id"]), self._content_hash(Path(str(node["id"])))) for node in nodes)
            graph_edges = tuple(GraphEdge(
                str(edge["source"]), str(edge["target"]), tuple(sorted(str(signal) for signal in edge.get("signals", []))),
                float(edge["weight"]),
            ) for edge in edges)
            graph_communities = tuple((index, tuple(sorted(str(member) for member in members))) for index, members in enumerate(communities))
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"graph snapshot is unavailable: {exc}") from exc
        return GraphSnapshotState(pages=pages, edges=graph_edges, communities=graph_communities)

    @staticmethod
    def _content_hash(path: Path) -> str:
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError as exc:
            raise ValueError(f"graph page is unavailable: {path}") from exc
