"""#11/#21 ACTIVE_INDEX 指针契约：原子发布 + 校验解析（纯 stdlib）。

指针 payload（schema_version=3）::

    {
      "active_lance": "builds/<id>/lance_db",
      "published_at": "<UTC ISO>",
      "schema_version": 3,
      "manifest_sha256": "<构建 manifest 的 sha256>"
    }

约定：
- 发布只允许 ``os.replace`` 原子替换；失败保留旧指针并上抛，绝不原位覆盖。
- 解析先验证 JSON schema、目标目录与 manifest checksum（v2 旧指针无 checksum 字段则跳过校验，兼容）。
- 指针损坏/校验失败时回退到最近一份已验证 build（有 manifest、无 .failed），
  并排除指针曾指向的失败目标；全部不可用时才回退 legacy 顶层 ``lance_db``。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger(__name__)

POINTER_NAME = "ACTIVE_INDEX"
_TEMP_NAME = ".ACTIVE_INDEX.tmp"
SCHEMA_VERSION = 3


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publish_pointer(index_dir: Path, build_dir: Path) -> None:
    """原子翻转 ACTIVE_INDEX 指向 ``build_dir``；失败时旧指针保持可用。"""
    pointer = index_dir / POINTER_NAME
    tmp = index_dir / _TEMP_NAME
    manifest = build_dir / "manifest.json"
    if not manifest.is_file():
        raise RuntimeError(f"拒绝发布：{manifest} 不存在（未 validated 的构建不可被引用）")
    payload = {
        "active_lance": str(build_dir.joinpath("lance_db").relative_to(index_dir)),
        "published_at": datetime.now(timezone.utc).isoformat(),
        "schema_version": SCHEMA_VERSION,
        "manifest_sha256": _sha256(manifest),
    }
    try:
        tmp.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        os.replace(tmp, pointer)
    except OSError as exc:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise RuntimeError(
            f"ACTIVE_INDEX 发布失败（旧活动索引保持可用）：{type(exc).__name__}: {exc}。"
            f"请确认 {pointer} 未被 Obsidian/杀软独占后重试。"
        ) from exc


def resolve_active_lance_dir(index_dir: Path) -> Path:
    """解析当前活动 lance 目录；指针损坏时回退最近已验证 build，最后回退 legacy 顶层。"""
    pointer = index_dir / POINTER_NAME
    rejected: Path | None = None  # 指针目标经校验被拒时，回退扫描中排除它
    if pointer.is_file():
        try:
            data = json.loads(pointer.read_text(encoding="utf-8"))
            rel = data.get("active_lance")
            if isinstance(rel, str) and rel:
                cand = Path(rel) if os.path.isabs(rel) else index_dir / rel
                manifest = cand.parent / "manifest.json"
                if cand.is_dir() and manifest.is_file():
                    digest = data.get("manifest_sha256")
                    if digest is None or digest == _sha256(manifest):
                        return cand
                    log.warning("ACTIVE_INDEX checksum 不匹配，忽略 %s", cand)
                    rejected = cand.parent
                else:
                    log.warning("ACTIVE_INDEX 指向的构建不完整，忽略 %s", cand)
                    rejected = cand.parent
        except (json.JSONDecodeError, OSError, ValueError):
            log.warning("ACTIVE_INDEX 损坏/被截断，回退已验证构建")
    builds_dir = index_dir / "builds"
    best: Path | None = None
    if builds_dir.is_dir():
        for entry in builds_dir.iterdir():
            if not entry.is_dir() or entry.name.startswith(".") or entry == rejected:
                continue
            if (entry / ".failed").exists():
                continue
            if (entry / "lance_db").is_dir() and (entry / "manifest.json").is_file():
                if best is None or entry.stat().st_mtime > best.stat().st_mtime:
                    best = entry
    if best is not None:
        log.info("回退到最近已验证构建：%s", best.name)
        return best / "lance_db"
    return index_dir / "lance_db"
