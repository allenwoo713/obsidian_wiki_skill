"""Contained, staged persistence for version-2 community-report artifacts."""
from __future__ import annotations

import json
import os
from dataclasses import fields
from pathlib import Path
from typing import Any, Sequence

from obsidian_wiki.domain.community_report_models import CommunityReport, CommunityReportManifest


class FilesystemCommunityReportStore:
    """Persist immutable sets below ``.index`` and publish one relative pointer."""

    def __init__(self, index_dir: Path):
        self._index_dir = Path(index_dir)
        self._builds_dir = self._index_dir / "community_report_builds"
        self._pointer = self._index_dir / "ACTIVE_COMMUNITY_REPORTS"

    def stage(self, build_id: str, reports: Sequence[CommunityReport], manifest: CommunityReportManifest) -> None:
        target = self._build_path(build_id)
        if target.exists():
            raise RuntimeError(f"community report build already exists: {build_id}")
        target.mkdir(parents=True, exist_ok=False)
        try:
            (target / "reports.jsonl").write_text(
                "".join(json.dumps(report.to_json(), ensure_ascii=False, sort_keys=True) + "\n" for report in reports),
                encoding="utf-8",
            )
            (target / "manifest.json").write_text(
                json.dumps(manifest.to_json(), ensure_ascii=False, sort_keys=True), encoding="utf-8"
            )
        except Exception as exc:
            (target / ".failed").write_text(f"stage failed: {type(exc).__name__}: {exc}", encoding="utf-8")
            raise

    def read_staged(self, build_id: str) -> tuple[tuple[CommunityReport, ...], CommunityReportManifest] | None:
        try:
            target = self._build_path(build_id)
        except ValueError:
            return None
        return self._read_set(target)

    def activate(self, build_id: str) -> None:
        target = self._build_path(build_id)
        if self._read_set(target) is None:
            raise RuntimeError("cannot activate an unreadable staged community-report set")
        self._index_dir.mkdir(parents=True, exist_ok=True)
        temporary = self._index_dir / ".ACTIVE_COMMUNITY_REPORTS.tmp"
        payload = {"active_build": str(target.relative_to(self._index_dir))}
        temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(temporary, self._pointer)

    def record_failure(self, build_id: str, reason: str) -> None:
        target = self._build_path(build_id)
        target.mkdir(parents=True, exist_ok=True)
        (target / ".failed").write_text(reason, encoding="utf-8")

    def read_active(self) -> tuple[tuple[CommunityReport, ...], CommunityReportManifest] | None:
        if not self._pointer.is_file():
            return None
        try:
            pointer = json.loads(self._pointer.read_text(encoding="utf-8"))
            if not isinstance(pointer, dict) or set(pointer) != {"active_build"}:
                return None
            raw = pointer["active_build"]
            if not isinstance(raw, str) or not raw:
                return None
            candidate = Path(raw)
            if candidate.is_absolute():
                return None
            target = (self._index_dir / candidate).resolve()
            builds = self._builds_dir.resolve()
            if target.parent != builds or not target.is_dir():
                return None
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        return self._read_set(target)

    def _build_path(self, build_id: str) -> Path:
        if not build_id or Path(build_id).name != build_id or build_id in {".", ".."}:
            raise ValueError("invalid community report build id")
        return self._builds_dir / build_id

    @staticmethod
    def _read_set(target: Path) -> tuple[tuple[CommunityReport, ...], CommunityReportManifest] | None:
        reports_path = target / "reports.jsonl"
        manifest_path = target / "manifest.json"
        if not reports_path.is_file() or not manifest_path.is_file():
            return None
        try:
            manifest = _manifest_from_json(json.loads(manifest_path.read_text(encoding="utf-8")))
            lines = reports_path.read_text(encoding="utf-8").splitlines()
            if any(not line.strip() for line in lines):
                return None
            reports = tuple(_report_from_json(json.loads(line)) for line in lines)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return None
        return reports, manifest


def _report_from_json(value: Any) -> CommunityReport:
    if not isinstance(value, dict) or set(value) != {field.name for field in fields(CommunityReport)}:
        raise ValueError("invalid community report record")
    if not isinstance(value["member_page_ids"], list) or not all(isinstance(item, str) for item in value["member_page_ids"]):
        raise ValueError("invalid member ids")
    value = dict(value)
    value["member_page_ids"] = tuple(value["member_page_ids"])
    return CommunityReport(**value)


def _manifest_from_json(value: Any) -> CommunityReportManifest:
    if not isinstance(value, dict) or set(value) != {field.name for field in fields(CommunityReportManifest)}:
        raise ValueError("invalid community report manifest")
    return CommunityReportManifest(**value)
