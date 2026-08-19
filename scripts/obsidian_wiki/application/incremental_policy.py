"""Strict, SDK-free evidence policy for auto build-mode selection.

This policy never authorizes an explicit incremental request.  It only decides
whether an ``auto`` request has supplied current, published snapshot evidence.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from obsidian_wiki.domain.incremental_models import (
    BuildModeCriterion,
    BuildModePolicy,
    BuildModePolicyLoad,
    BuildModeSelection,
    BuildTelemetry,
)


_DISABLED_FIELDS = frozenset({"schema_version", "enabled"})
_ENABLED_FIELDS = frozenset({
    "schema_version", "enabled", "compatibility_digest", "evidence_observation_ids",
    "minimum_compatible_observations", "max_evidence_age_seconds", "match", "criteria",
})
_IDENTITY_FIELDS = (
    "layout", "index_layout_version", "sparse_chunk_schema_version",
    "dense_chunk_schema_version", "fts_config", "vector_config", "ann_policy", "sdk_versions",
)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def compatibility_digest_from_manifest(manifest: Mapping[str, object]) -> str:
    """Bind mode evidence to every storage/ANN/SDK identity input that affects rows."""
    if not isinstance(manifest, Mapping):
        raise ValueError("manifest compatibility input must be a mapping")
    missing = [name for name in _IDENTITY_FIELDS if name not in manifest]
    if missing:
        raise ValueError(f"manifest compatibility fields missing: {missing}")
    ann_policy = manifest["ann_policy"]
    if not isinstance(ann_policy, Mapping) or not isinstance(ann_policy.get("policy_sha256"), str):
        raise ValueError("manifest approved ANN policy identity is missing")
    identity = {name: manifest[name] for name in _IDENTITY_FIELDS}
    return _canonical_sha256(identity)


def _failed_load(reason: str) -> BuildModePolicyLoad:
    return BuildModePolicyLoad(policy=None, policy_sha256=None, reason=reason)


def _contained_policy_path(project_root: Path, policy_path: Path | None) -> tuple[Path | None, str | None]:
    root = Path(project_root).resolve()
    raw_path = Path(".index") / "build-mode-policy.json" if policy_path is None else Path(policy_path)
    if raw_path.is_absolute():
        return None, "policy_path_outside_project"
    candidate = root / raw_path
    if candidate.is_symlink():
        return None, "policy_path_symlink"
    try:
        resolved = candidate.resolve()
        resolved.relative_to(root)
    except (OSError, ValueError):
        return None, "policy_path_outside_project"
    return candidate, None


def _parse_policy(data: object) -> BuildModePolicy:
    if not isinstance(data, dict) or isinstance(data.get("schema_version"), bool):
        raise ValueError("policy must be an object")
    if data.get("schema_version") != 1 or not isinstance(data.get("enabled"), bool):
        raise ValueError("policy schema version is invalid")
    if data["enabled"] is False:
        if set(data) != _DISABLED_FIELDS:
            raise ValueError("disabled policy fields are invalid")
        return BuildModePolicy(schema_version=1, enabled=False)
    if set(data) != _ENABLED_FIELDS:
        raise ValueError("enabled policy fields are invalid")
    raw_criteria = data["criteria"]
    if not isinstance(raw_criteria, list):
        raise ValueError("policy criteria must be a list")
    criteria: list[BuildModeCriterion] = []
    for item in raw_criteria:
        if not isinstance(item, dict) or set(item) != {"metric", "operator", "threshold"}:
            raise ValueError("policy criterion fields are invalid")
        criteria.append(BuildModeCriterion(**item))
    return BuildModePolicy(
        schema_version=data["schema_version"], enabled=data["enabled"],
        compatibility_digest=data["compatibility_digest"],
        evidence_observation_ids=tuple(data["evidence_observation_ids"]),
        minimum_compatible_observations=data["minimum_compatible_observations"],
        max_evidence_age_seconds=data["max_evidence_age_seconds"],
        match=data["match"], criteria=tuple(criteria),
    )


def load_build_mode_policy(project_root: Path, policy_path: Path | None = None) -> BuildModePolicyLoad:
    """Read one contained regular JSON policy and return a snapshot-safe result.

    Every input failure is represented as a stable reason code instead of raising
    into a path that could accidentally default to online mutation.
    """
    candidate, failure = _contained_policy_path(Path(project_root), policy_path)
    if failure is not None:
        return _failed_load(failure)
    assert candidate is not None
    if not candidate.exists():
        return _failed_load("policy_missing")
    if candidate.is_symlink():
        return _failed_load("policy_path_symlink")
    if not candidate.is_file():
        return _failed_load("policy_path_not_regular")
    try:
        raw = candidate.read_text(encoding="utf-8")
    except OSError:
        return _failed_load("policy_read_failed")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return _failed_load("policy_parse_failed")
    try:
        policy = _parse_policy(data)
    except (TypeError, ValueError):
        return _failed_load("policy_schema_invalid")
    digest = _canonical_sha256(data)
    if not policy.enabled:
        return BuildModePolicyLoad(policy=policy, policy_sha256=digest, reason="policy_disabled")
    return BuildModePolicyLoad(policy=policy, policy_sha256=digest, reason="policy_loaded")


def _p95(values: Sequence[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)]


def select_auto_build_mode(
    loaded: BuildModePolicyLoad,
    observations: Sequence[BuildTelemetry],
    *, current_compatibility_digest: str, now_epoch_seconds: float,
) -> BuildModeSelection:
    """Select incremental only from all policy-named, compatible snapshots."""
    if not isinstance(now_epoch_seconds, (int, float)) or isinstance(now_epoch_seconds, bool) or not math.isfinite(now_epoch_seconds):
        raise ValueError("selection clock must be finite")
    policy = loaded.policy
    if policy is None:
        return BuildModeSelection("snapshot", loaded.reason, None, current_compatibility_digest, ())
    if not policy.enabled:
        return BuildModeSelection("snapshot", "policy_disabled", loaded.policy_sha256, current_compatibility_digest, ())
    assert policy.compatibility_digest is not None
    if policy.compatibility_digest != current_compatibility_digest:
        return BuildModeSelection("snapshot", "policy_compatibility_mismatch", loaded.policy_sha256, current_compatibility_digest, ())
    by_id = {observation.observation_id: observation for observation in observations}
    named: list[BuildTelemetry] = []
    for observation_id in policy.evidence_observation_ids:
        observation = by_id.get(observation_id)
        if observation is None:
            return BuildModeSelection("snapshot", "evidence_missing", loaded.policy_sha256, current_compatibility_digest, ())
        if not observation.completed or observation.mode_selected != "snapshot":
            return BuildModeSelection("snapshot", "evidence_not_published_snapshot", loaded.policy_sha256, current_compatibility_digest, ())
        if observation.compatibility_digest != current_compatibility_digest:
            return BuildModeSelection("snapshot", "evidence_compatibility_mismatch", loaded.policy_sha256, current_compatibility_digest, ())
        if now_epoch_seconds - observation.completed_at_epoch_seconds > policy.max_evidence_age_seconds:
            return BuildModeSelection("snapshot", "evidence_stale", loaded.policy_sha256, current_compatibility_digest, ())
        named.append(observation)
    if len(named) < policy.minimum_compatible_observations:
        return BuildModeSelection("snapshot", "evidence_insufficient", loaded.policy_sha256, current_compatibility_digest, tuple(item.observation_id for item in named))
    values = {
        "snapshot_p95_ms": _p95([
            item.timings.scan_parse_ms + item.timings.chunking_ms
            + item.timings.embedding_cache_hit_ms + item.timings.embedding_cache_miss_ms
            + item.timings.serialization_write_ms + item.timings.fts_catch_up_ms
            + item.timings.vector_catch_up_ms + item.timings.validation_ms
            + item.timings.publication_ms
            for item in named
        ]),
        "peak_staged_disk_bytes": float(max(item.peak_staged_disk_bytes for item in named)),
        "index_rebuild_ms": _p95([item.timings.index_rebuild_ms for item in named]),
    }
    matched = tuple(values[item.metric] >= item.threshold for item in policy.criteria)
    criteria_met = any(matched) if policy.match == "any" else all(matched)
    reason = "policy_criteria_met" if criteria_met else "policy_criteria_unmet"
    return BuildModeSelection(
        "incremental" if criteria_met else "snapshot", reason, loaded.policy_sha256,
        current_compatibility_digest, tuple(item.observation_id for item in named),
        tuple(sorted(values.items())),
    )
