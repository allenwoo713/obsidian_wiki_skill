"""Wiki Markdown provenance and freshness. Standard library only.

Issue #65：L2（Wiki/*.md 真值）与 L3（索引/图谱派生缓存）之间此前没有任何
staleness 检测——手改 Wiki 后 query 会静默返回旧内容。本模块提供：

- ``capture_wiki``：单次读取的 Wiki 树内容快照（per-file sha256 + tree digest）；
- ``compare_snapshot`` / ``inspect_snapshot``：当前树 vs 发布时 provenance 的差异报告；
- ``attach_snapshot`` / ``require_current``：发布边界校验（构建期间 Wiki 变更则拒绝发布）；
- ``collect_reports`` / ``diagnostic_exit_code``：查询期 index/graph 独立新鲜度报告，
  退出码约定与 ``check_ann_drift.py`` 一致（0=一致，1=stale，2=unknown）；
- ``GuardedContextRepository``：请求内全文读取——hash 与正文解码来自同一份字节。

不导入 build_index、Torch 或 LanceDB；不使用可能隐藏目录访问错误的 glob 来判断
「空树即 fresh」，目录错误明确返回 unknown。读取时的 stat 仅用于检测读过程发生
变化，不作为替代内容 hash 的缓存依据。
"""
from __future__ import annotations

import fnmatch
import hashlib
import io
import json
import os
import re
import stat
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Mapping

SCHEMA = 1
SCOPE = "wiki-markdown-excluding-dotgraph-v1"
DIGEST = re.compile(r"[0-9a-f]{64}")


class SnapshotError(RuntimeError):
    pass


@dataclass(frozen=True)
class WikiSnapshot:
    root: Path
    hashes: Mapping[str, str]
    raw: Mapping[str, bytes]
    scan_ms: float
    bytes_read: int

    def to_json(self) -> dict:
        files = dict(sorted(self.hashes.items()))
        return {
            "schema_version": SCHEMA,
            "scope": SCOPE,
            "wiki_root": str(self.root),
            "files": files,
            "tree_sha256": tree_digest(files),
        }


def tree_digest(files: Mapping[str, str]) -> str:
    payload = json.dumps(
        sorted(files.items()), ensure_ascii=True, separators=(",", ":")
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def decode_markdown(raw: bytes) -> str:
    # Same universal-newline semantics as Path.read_text(..., errors="replace").
    with io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8", errors="replace",
                          newline=None) as stream:
        return stream.read()


def _raise_walk_error(exc: OSError) -> None:
    raise exc


def _no_link(path: Path) -> os.stat_result:
    st = path.lstat()
    # Reject Windows junctions/reparse points too (works on Python 3.10).
    if stat.S_ISLNK(st.st_mode) or getattr(st, "st_file_attributes", 0) & 0x400:
        raise SnapshotError(f"unsupported_link:{path}")
    return st


def _paths(root: Path) -> tuple:
    result = []
    if not stat.S_ISDIR(root.stat().st_mode):
        raise SnapshotError(f"wiki_not_directory:{root}")
    for directory, dirs, files in os.walk(
        root, topdown=True, onerror=_raise_walk_error, followlinks=False
    ):
        dirs[:] = sorted(name for name in dirs if name != ".graph")
        for name in dirs:
            _no_link(Path(directory) / name)
        for name in sorted(files):
            # Mirrors native-platform "*.md" matching; do not lowercase paths.
            if not fnmatch.fnmatch(name, "*.md"):
                continue
            path = Path(directory) / name
            relative = path.relative_to(root).as_posix()
            if "\\" in relative:
                raise SnapshotError(f"nonportable_path:{relative}")
            if not stat.S_ISREG(_no_link(path).st_mode):
                raise SnapshotError(f"not_regular_file:{path}")
            result.append(path)
    return tuple(sorted(result))


def _signature(st: os.stat_result) -> tuple:
    # Windows: NTFS lazily updates the creation time of a just-truncated file, so
    # st_ctime_ns can differ between lstat and the opened handle's fstat (60/60
    # reproduced) and must not take part in replacement detection. File identity
    # is carried by dev+inode; size+mtime guard same-inode in-place rewrites.
    return (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns)


def _read_file(path: Path, retain_bytes: bool) -> tuple:
    before = _no_link(path)
    digest = hashlib.sha256()
    parts = [] if retain_bytes else None
    size = 0
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if _signature(before) != _signature(opened):
            raise SnapshotError(f"file_replaced_before_read:{path}")
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                break
            size += len(block)
            digest.update(block)
            if parts is not None:
                parts.append(block)
        after_fd = os.fstat(stream.fileno())
    after_path = _no_link(path)
    if (_signature(before) != _signature(after_fd)
            or _signature(before) != _signature(after_path)
            or size != before.st_size):
        raise SnapshotError(f"file_changed_during_read:{path}")
    return digest.hexdigest(), b"".join(parts) if parts is not None else None, size


def capture_wiki(wiki_dir: Path, *, retain_bytes: bool = False) -> WikiSnapshot:
    started = time.perf_counter()
    # Root aliases are resolved once; page IDs continue to use canonical paths.
    root = Path(wiki_dir).resolve(strict=True)
    paths = _paths(root)
    hashes, raw = {}, {}
    total = 0
    for path in paths:
        key = path.relative_to(root).as_posix()
        digest, content, size = _read_file(path, retain_bytes)
        hashes[key] = digest
        total += size
        if content is not None:
            raw[key] = content
    if paths != _paths(root):
        raise SnapshotError("wiki_membership_changed_during_scan")
    return WikiSnapshot(
        root, MappingProxyType(hashes), MappingProxyType(raw),
        (time.perf_counter() - started) * 1000, total,
    )


def validate_saved(saved: object) -> dict:
    if not isinstance(saved, dict):
        raise ValueError("missing_wiki_snapshot")
    required = {"schema_version", "scope", "wiki_root", "files", "tree_sha256"}
    if set(saved) != required:
        raise ValueError("invalid_wiki_snapshot_fields")
    if type(saved["schema_version"]) is not int or saved["schema_version"] != SCHEMA:
        raise ValueError("unsupported_wiki_snapshot_schema")
    if saved["scope"] != SCOPE:
        raise ValueError("unsupported_wiki_snapshot_scope")
    if not isinstance(saved["wiki_root"], str) or not saved["wiki_root"]:
        raise ValueError("invalid_wiki_root")
    files = saved["files"]
    if not isinstance(files, dict):
        raise ValueError("invalid_wiki_snapshot_files")
    for key, value in files.items():
        if not isinstance(key, str) or not key:
            raise ValueError("invalid_relative_path")
        path = PurePosixPath(key)
        if (path.is_absolute() or path.as_posix() != key
                or ".." in path.parts or "\\" in key or ".graph" in path.parts
                or not key.lower().endswith(".md")):
            raise ValueError("invalid_relative_path")
        if not isinstance(value, str) or DIGEST.fullmatch(value) is None:
            raise ValueError("invalid_page_sha256")
    if (not isinstance(saved["tree_sha256"], str)
            or saved["tree_sha256"] != tree_digest(files)):
        raise ValueError("invalid_tree_sha256")
    return saved


def unknown(reason: str) -> dict:
    return {
        "status": "unknown", "reasons": [reason],
        "added": [], "modified": [], "deleted": [],
    }


def compare_snapshot(saved: object, current: WikiSnapshot) -> dict:
    try:
        baseline = validate_saved(saved)
    except ValueError as exc:
        return unknown(str(exc))
    old, new = baseline["files"], current.hashes
    added = sorted(new.keys() - old.keys())
    deleted = sorted(old.keys() - new.keys())
    modified = sorted(key for key in old.keys() & new.keys() if old[key] != new[key])
    reasons = []
    # Absolute page IDs mean a relocated vault must be rebuilt even if bytes match.
    if baseline["wiki_root"] != str(current.root):
        reasons.append("wiki_root_changed")
    if added or modified or deleted:
        reasons.append("wiki_content_changed")
    return {
        "status": "stale" if reasons else "fresh",
        "reasons": reasons, "added": added, "modified": modified, "deleted": deleted,
        "expected_tree_sha256": baseline["tree_sha256"],
        "observed_tree_sha256": tree_digest(new),
        "page_count": len(new), "bytes_hashed": current.bytes_read,
        "scan_ms": round(current.scan_ms, 3), "scope": SCOPE,
    }


def inspect_snapshot(wiki_dir: Path, saved: object) -> dict:
    try:
        return compare_snapshot(saved, capture_wiki(wiki_dir))
    except (OSError, SnapshotError, ValueError) as exc:
        return unknown(f"wiki_scan_failed:{type(exc).__name__}:{exc}")


def require_current(wiki_dir: Path, saved: object) -> None:
    result = inspect_snapshot(wiki_dir, saved)
    if result["status"] != "fresh":
        raise SnapshotError("wiki_changed_before_publish:" + json.dumps(result))


def attach_snapshot(manifest: dict, wiki_dir: Path, saved: object) -> None:
    # Explicit injected/synthetic plans without provenance remain unknown.
    if saved is None:
        manifest["wiki_snapshot"] = None
        return
    baseline = validate_saved(saved)
    root = Path(wiki_dir).resolve(strict=True)
    if baseline["wiki_root"] != str(root):
        raise SnapshotError("planned_wiki_root_mismatch")
    for page in manifest.get("pages", []):
        path = Path(page["path"])
        if path.suffix.lower() != ".md":  # image-caption pseudo pages are out of scope
            continue
        path = path if path.is_absolute() else root.parent / path
        try:
            key = path.resolve().relative_to(root).as_posix()
        except ValueError as exc:
            raise SnapshotError("manifest_page_outside_wiki") from exc
        if baseline["files"].get(key) != page.get("sha256"):
            raise SnapshotError(f"manifest_page_hash_mismatch:{key}")
    require_current(root, baseline)
    # Detach from mutable caller-owned metadata before durable serialization.
    manifest["wiki_snapshot"] = json.loads(json.dumps(baseline))


def diagnostic_exit_code(reports: Mapping[str, dict]) -> int:
    states = {report["status"] for report in reports.values()}
    if "unknown" in states:
        return 2
    if "stale" in states:
        return 1
    return 0


class FreshnessError(RuntimeError):
    def __init__(self, reports: Mapping[str, dict]):
        self.reports = dict(reports)
        self.exit_code = diagnostic_exit_code(self.reports)
        super().__init__(json.dumps(self.reports, ensure_ascii=False))


def enforce_freshness(reports: Mapping[str, dict], policy: str) -> None:
    if policy not in {"warn", "strict", "allow"}:
        raise ValueError("freshness policy must be warn, strict, or allow")
    if policy == "strict" and diagnostic_exit_code(reports):
        raise FreshnessError(reports)


def read_graph_payload(index_dir: Path) -> tuple:
    try:
        payload = json.loads((Path(index_dir) / "graph.json").read_bytes())
    except FileNotFoundError:
        return None, None
    except (OSError, UnicodeError, ValueError) as exc:
        return None, f"graph_unreadable:{type(exc).__name__}:{exc}"
    if not isinstance(payload, dict):
        return None, "graph_not_object"
    # Only what the retrieval path consumes is required; communities is optional
    # here (the community-report gate validates it via FilesystemGraphSnapshot).
    if any(not isinstance(payload.get(key), list)
           for key in ("nodes", "edges")):
        return None, "invalid_graph_structure"
    return payload, None


def collect_reports(
    wiki_dir: Path, manifest: dict, graph_payload: dict | None = None,
    *, graph_error: str | None = None, include_graph: bool = True,
) -> dict:
    try:
        current = capture_wiki(wiki_dir)
        report = compare_snapshot(manifest.get("wiki_snapshot"), current)
    except (OSError, SnapshotError, ValueError) as exc:
        current = None
        report = unknown(f"wiki_scan_failed:{type(exc).__name__}:{exc}")
    report.update(build_id=manifest.get("build_id"), generation=manifest.get("generation"))
    reports = {"index": report}
    if include_graph:
        if graph_error:
            reports["graph"] = unknown(graph_error)
        elif graph_payload is None:
            # Optional local graph absent: no graph evidence will be served.
            reports["graph"] = {"status": "not_built", "reasons": ["optional_graph_missing"]}
        elif current is None:
            reports["graph"] = unknown("wiki_scan_failed")
        else:
            reports["graph"] = compare_snapshot(graph_payload.get("wiki_snapshot"), current)
            reports["graph"]["graph_build_id"] = graph_payload.get("graph_build_id")
    return reports


class GuardedContextRepository:
    """Request-local full-page reads: hash and render the very same bytes."""

    def __init__(self, repository, manifest: dict, wiki_dir: Path):
        self._repository = repository
        self._root = Path(wiki_dir).resolve()
        self._pages = {p.get("page_id"): p for p in manifest.get("pages", [])}
        self._cache = {}
        self.reports = {}

    def __getattr__(self, name):
        return getattr(self._repository, name)

    def read_page(self, page_id: str) -> str:
        if page_id in self._cache:
            return self._cache[page_id]
        page = self._pages.get(page_id)
        if page is None or Path(page["path"]).suffix.lower() != ".md":
            return self._repository.read_page(page_id)
        try:
            path = Path(page["path"])
            if not path.is_absolute():
                path = self._root.parent / path
            path.resolve().relative_to(self._root)
            digest, raw, _ = _read_file(path, True)
            if digest != page.get("sha256"):
                self.reports[page_id] = {
                    "status": "stale", "reasons": ["full_page_bytes_differ_from_index"],
                    "path": str(path),
                }
            text = decode_markdown(raw)
            match = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
            body = (match.group(2) if match else text).strip()
        except (OSError, SnapshotError, ValueError) as exc:
            self.reports[page_id] = unknown(f"full_page_read_failed:{exc}")
            body = ""
        self._cache[page_id] = body
        return body
