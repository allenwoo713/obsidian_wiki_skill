"""Strict, evidence-bound build mode and telemetry contract gates."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pytest

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def _digest(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _compatibility() -> dict[str, object]:
    return {
        "layout": "sparse_chunks+dense_chunks",
        "index_layout_version": 6,
        "sparse_chunk_schema_version": 1,
        "dense_chunk_schema_version": 1,
        "fts_config": {"column": "fts_text"},
        "vector_config": {"index_type": "hnsw_sq", "metric": "cosine"},
        "ann_policy": {"policy_sha256": "a" * 64},
        "sdk_versions": {"lancedb": "0.34.0", "pyarrow": "25.0.0"},
    }


def _observation(*, observation_id: str = "snapshot-a", compatibility_digest: str | None = None,
                 completed_at: float = 1000.0, mode_selected: str = "snapshot"):
    from obsidian_wiki.domain.incremental_models import BuildTelemetry, BuildTiming, TableRowCounts

    return BuildTelemetry(
        schema_version=1,
        observation_id=observation_id,
        mode_requested="snapshot",
        mode_selected=mode_selected,
        selection_reason="explicit_snapshot",
        compatibility_digest=compatibility_digest or _digest(_compatibility()),
        completed_at_epoch_seconds=completed_at,
        timings=BuildTiming(
            scan_parse_ms=1.0, chunking_ms=2.0, embedding_cache_hit_ms=3.0,
            embedding_cache_miss_ms=4.0, serialization_write_ms=5.0,
            fts_catch_up_ms=6.0, vector_catch_up_ms=7.0, validation_ms=8.0,
            publication_ms=9.0, index_rebuild_ms=10.0,
        ),
        sparse_rows=TableRowCounts(inserted=1, updated=2, deleted=3, unchanged=4, physically_written=3),
        dense_rows=TableRowCounts(inserted=5, updated=6, deleted=7, unchanged=8, physically_written=11),
        embedding_cache_hits=12,
        embedding_cache_misses=13,
        peak_staged_disk_bytes=14,
        completed=True,
    )


def _policy(compatibility_digest: str, observation_ids: list[str], **overrides: object) -> dict[str, object]:
    policy = {
        "schema_version": 1,
        "enabled": True,
        "compatibility_digest": compatibility_digest,
        "evidence_observation_ids": observation_ids,
        "minimum_compatible_observations": len(observation_ids),
        "max_evidence_age_seconds": 100.0,
        "match": "all",
        "criteria": [{"metric": "snapshot_p95_ms", "operator": "gte", "threshold": 1.0}],
    }
    policy.update(overrides)
    return policy


def _write_policy(project_root: Path, policy: object, relative: str = ".index/build-mode-policy.json") -> Path:
    path = project_root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(policy), encoding="utf-8")
    return path


def _select(project_root: Path, observations, *, now: float = 1001.0, policy_path: Path | None = None):
    from obsidian_wiki.application.incremental_policy import (
        load_build_mode_policy,
        select_auto_build_mode,
    )

    loaded = load_build_mode_policy(project_root, policy_path)
    return select_auto_build_mode(
        loaded, observations, current_compatibility_digest=_digest(_compatibility()), now_epoch_seconds=now
    )


def test_selection_uses_explicit_compatible_snapshot_evidence(tmp_path):
    compatibility_digest = _digest(_compatibility())
    _write_policy(tmp_path, _policy(compatibility_digest, ["snapshot-a"]))

    selection = _select(tmp_path, [_observation(compatibility_digest=compatibility_digest)])

    assert selection.selected_mode == "incremental"
    assert selection.reason == "policy_criteria_met"
    assert selection.policy_sha256 == _digest(_policy(compatibility_digest, ["snapshot-a"]))
    assert selection.evidence_observation_ids == ("snapshot-a",)


@pytest.mark.parametrize(
    ("policy", "reason"),
    [
        (None, "policy_missing"),
        ({"schema_version": 1, "enabled": False}, "policy_disabled"),
        ({"schema_version": 2, "enabled": False}, "policy_schema_invalid"),
        ({"schema_version": 1, "enabled": True}, "policy_schema_invalid"),
    ],
)
def test_selection_falls_back_for_missing_disabled_or_invalid_policy(tmp_path, policy, reason):
    if policy is not None:
        _write_policy(tmp_path, policy)

    selection = _select(tmp_path, [_observation()])

    assert selection.selected_mode == "snapshot"
    assert selection.reason == reason


def test_policy_loader_rejects_escape_symlink_and_non_regular_override(tmp_path):
    compatibility_digest = _digest(_compatibility())
    outside = tmp_path.parent / "outside-policy.json"
    outside.write_text(json.dumps(_policy(compatibility_digest, ["snapshot-a"])), encoding="utf-8")
    _write_policy(tmp_path, _policy(compatibility_digest, ["snapshot-a"]), "policies/valid.json")
    symlink = tmp_path / "policies" / "link.json"
    symlink.symlink_to(outside)
    directory = tmp_path / "policies" / "directory.json"
    directory.mkdir()

    for override, expected in ((Path("../outside-policy.json"), "policy_path_outside_project"),
                               (Path("policies/link.json"), "policy_path_symlink"),
                               (Path("policies/directory.json"), "policy_path_not_regular")):
        selection = _select(tmp_path, [_observation(compatibility_digest=compatibility_digest)], policy_path=override)
        assert selection.selected_mode == "snapshot"
        assert selection.reason == expected


def test_evidence_requires_published_completed_current_compatible_named_snapshots(tmp_path):
    compatibility_digest = _digest(_compatibility())
    _write_policy(tmp_path, _policy(compatibility_digest, ["snapshot-a", "snapshot-b"], minimum_compatible_observations=2))
    valid = _observation(observation_id="snapshot-a", compatibility_digest=compatibility_digest)
    wrong_mode = _observation(observation_id="snapshot-b", compatibility_digest=compatibility_digest, mode_selected="incremental")

    selection = _select(tmp_path, [valid, wrong_mode])

    assert selection.selected_mode == "snapshot"
    assert selection.reason == "evidence_not_published_snapshot"


@pytest.mark.parametrize(
    ("observation", "now", "reason"),
    [
        (_observation(compatibility_digest="b" * 64), 1001.0, "evidence_compatibility_mismatch"),
        (_observation(completed_at=1.0), 1001.0, "evidence_stale"),
    ],
)
def test_evidence_compatibility_and_age_fail_closed(tmp_path, observation, now, reason):
    compatibility_digest = _digest(_compatibility())
    _write_policy(tmp_path, _policy(compatibility_digest, ["snapshot-a"]))

    selection = _select(tmp_path, [observation], now=now)

    assert selection.selected_mode == "snapshot"
    assert selection.reason == reason


def test_unmet_explicit_threshold_is_snapshot_without_any_builtin_threshold(tmp_path):
    compatibility_digest = _digest(_compatibility())
    policy = _policy(compatibility_digest, ["snapshot-a"], criteria=[{
        "metric": "peak_staged_disk_bytes", "operator": "gte", "threshold": 15.0,
    }])
    _write_policy(tmp_path, policy)

    selection = _select(tmp_path, [_observation(compatibility_digest=compatibility_digest)])

    assert selection.selected_mode == "snapshot"
    assert selection.reason == "policy_criteria_unmet"


def test_telemetry_rejects_unreconciled_rows_and_nonfinite_measurements():
    from obsidian_wiki.domain.incremental_models import BuildTiming, TableRowCounts

    with pytest.raises(ValueError, match="physically_written"):
        TableRowCounts(inserted=1, updated=1, deleted=0, unchanged=0, physically_written=1)
    with pytest.raises(ValueError, match="finite"):
        BuildTiming(
            scan_parse_ms=math.nan, chunking_ms=0, embedding_cache_hit_ms=0,
            embedding_cache_miss_ms=0, serialization_write_ms=0, fts_catch_up_ms=0,
            vector_catch_up_ms=0, validation_ms=0, publication_ms=0, index_rebuild_ms=0,
        )


def test_compatibility_digest_changes_for_all_locked_identity_inputs():
    from obsidian_wiki.application.incremental_policy import compatibility_digest_from_manifest

    base = _compatibility()
    changed = json.loads(json.dumps(base))
    changed["ann_policy"]["policy_sha256"] = "b" * 64

    assert compatibility_digest_from_manifest(base) != compatibility_digest_from_manifest(changed)


def _tiny_plan(text: str):
    from obsidian_wiki.domain.index_models import SparseChunk

    common = dict(
        page_id="notes/telemetry.md", path="notes/telemetry.md", title="Telemetry", text=text,
        content_hash=text, end_char=len(text),
    )
    return (
        SparseChunk(**common, chunk_id="notes/telemetry.md::sparse", fts_text=text, chunk_kind="sparse"),
        SparseChunk(**common, chunk_id="notes/telemetry.md::dense", fts_text=text, chunk_kind="dense"),
    )


def _tiny_embed(texts):
    vectors = []
    for offset, _text in enumerate(texts):
        raw = [float((offset + index) % 13 + 1) for index in range(384)]
        norm = math.sqrt(sum(value * value for value in raw))
        vectors.append([value / norm for value in raw])
    return vectors


def test_telemetry_in_snapshot_and_incremental_manifests_reconciles_physical_rows(tmp_path):
    from build_index import build_storage_contract
    from obsidian_wiki.application.incremental_index_service import IncrementalIndexService

    wiki_dir = tmp_path / "Wiki"
    wiki_dir.mkdir()
    index_dir = tmp_path / ".index"
    initial = _tiny_plan("initial telemetry payload")
    snapshot = build_storage_contract(wiki_dir, index_dir, embed=_tiny_embed, sparse_chunks=initial)
    snapshot_manifest = json.loads(snapshot.artifact.manifest_path.read_text(encoding="utf-8"))

    snapshot_telemetry = snapshot_manifest["build_telemetry"]
    assert snapshot_telemetry["mode_selected"] == "snapshot"
    assert set(snapshot_telemetry["timings"]) == {
        "scan_parse_ms", "chunking_ms", "embedding_cache_hit_ms", "embedding_cache_miss_ms",
        "serialization_write_ms", "fts_catch_up_ms", "vector_catch_up_ms", "validation_ms",
        "publication_ms", "index_rebuild_ms",
    }
    for table in ("sparse_rows", "dense_rows"):
        rows = snapshot_telemetry[table]
        assert rows["physically_written"] == rows["inserted"] + rows["updated"]

    incremental = IncrementalIndexService().build(
        wiki_dir, index_dir, canonical_chunks=_tiny_plan("changed telemetry payload"), embed=_tiny_embed,
    )
    incremental_manifest = json.loads(incremental.artifact.manifest_path.read_text(encoding="utf-8"))
    incremental_telemetry = incremental_manifest["build_telemetry"]
    assert incremental_telemetry["mode_selected"] == "incremental"
    assert incremental_telemetry["sparse_rows"]["physically_written"] == incremental.sparse_mutation.physically_written
    assert incremental_telemetry["dense_rows"]["physically_written"] == incremental.dense_mutation.physically_written
