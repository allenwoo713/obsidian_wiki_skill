"""Production-route snapshot versus online-incremental evaluation gates."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest


SKILL_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKILL_ROOT / "eval"))

from compare_build_modes import (  # noqa: E402
    REQUIRED_SCENARIOS,
    _artifact_digest,
    run_diagnostic_comparison,
    run_mode_comparison,
    validate_diagnostic_comparison_artifact,
    validate_comparison_artifact,
)
from obsidian_wiki.application.incremental_policy import load_build_mode_policy  # noqa: E402


def _run(tmp_path: Path) -> dict:
    output = tmp_path / "equivalence.json"
    artifact = run_mode_comparison(
        work_dir=tmp_path / "work",
        output=output,
    )
    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    return artifact


def _run_diagnostic(tmp_path: Path, *, scenario: str) -> dict:
    output = tmp_path / "diagnostic.json"
    artifact = run_diagnostic_comparison(
        work_dir=tmp_path / "work", output=output, scenario=scenario,
    )
    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    return artifact


def _refresh_digest(artifact: dict) -> dict:
    """Return a digest-valid mutation so validation reaches scenario integrity."""
    artifact["artifact_sha256"] = _artifact_digest(artifact)
    return artifact


@pytest.fixture(scope="module")
def full_acceptance_artifact(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Build the complete public facade/query artifact once for mutation cases."""
    return _run(tmp_path_factory.mktemp("full-comparison"))


def test_public_snapshot_incremental_retrieval_equivalence(tmp_path: Path) -> None:
    """An edit uses public build/load/hybrid_search in both build modes."""
    artifact = _run_diagnostic(tmp_path, scenario="page_edit")

    validate_diagnostic_comparison_artifact(artifact)
    with pytest.raises(ValueError, match="comparison artifact kind: diagnostic"):
        validate_comparison_artifact(artifact)
    assert artifact["verdict"] == "pass"
    scenario = artifact["scenarios"]["page_edit"]
    assert scenario["verdict"] == "pass"
    assert scenario["snapshot"]["manifest"]["build_mode_selected"] == "snapshot"
    assert scenario["incremental"]["manifest"]["build_mode_selected"] == "incremental"
    assert scenario["equivalence"]["page_recall_at_5"]
    assert scenario["equivalence"]["evidence_recall_at_10"]
    assert scenario["equivalence"]["citations"]
    assert scenario["equivalence"]["context"]
    assert scenario["equivalence"]["graph"]
    assert scenario["equivalence"]["sparse"]
    assert scenario["equivalence"]["dense"]


def test_scenario_matrix_and_artifact_fail_closed(full_acceptance_artifact: dict) -> None:
    """Every logical mutation and public fallback leaves complete evidence."""
    baseline = (SKILL_ROOT / "eval" / "baselines.json").read_bytes()
    ann_policy = (SKILL_ROOT / "eval" / "ann-policy.json").read_bytes()
    artifact = full_acceptance_artifact

    validate_comparison_artifact(artifact)
    assert artifact["verdict"] == "pass"
    assert artifact["artifact_kind"] == "acceptance"
    assert tuple(artifact["scenario_names"]) == REQUIRED_SCENARIOS
    assert set(artifact["scenarios"]) == set(REQUIRED_SCENARIOS)
    assert artifact["scenarios"]["page_deletion"]["deleted_page_absent"] is True
    assert artifact["scenarios"]["unchanged_rebuild"]["incremental"]["telemetry"]["written_rows"] == 0
    assert artifact["scenarios"]["configuration_drift"]["incremental_calls"] == {
        "clone_table": 0, "delta": 0, "catch_up": 0,
    }
    drift = artifact["scenarios"]["configuration_drift"]["incremental"]["manifest"]
    assert drift["build_mode_requested"] == "auto"
    assert drift["build_mode_selected"] == "snapshot"
    assert drift["selection_reason"] == "incompatible_active_contract:fts_config"
    assert drift["build_mode_policy_sha256"]
    assert artifact["scenarios"]["failure_recovery"]["recovery_preserved_active"] is True
    assert (SKILL_ROOT / "eval" / "baselines.json").read_bytes() == baseline
    assert (SKILL_ROOT / "eval" / "ann-policy.json").read_bytes() == ann_policy

    for mutate in (
        lambda value: value.pop("inputs"),
        lambda value: value["scenarios"]["page_edit"].pop("incremental"),
        lambda value: value["scenarios"]["page_edit"]["snapshot"]["manifest"].update({"unindexed_rows": 1}),
        lambda value: value["scenarios"]["page_edit"]["equivalence"].update({"dense": False}),
        lambda value: value["inputs"].update({"ann_policy_sha256": "0" * 64}),
    ):
        invalid = copy.deepcopy(artifact)
        mutate(invalid)
        with pytest.raises(ValueError):
            validate_comparison_artifact(invalid)


@pytest.mark.parametrize("missing", REQUIRED_SCENARIOS)
def test_acceptance_validator_rejects_every_digest_valid_missing_scenario(
    full_acceptance_artifact: dict, missing: str,
) -> None:
    """D-09: each canonical scenario is independently required for acceptance."""
    artifact = copy.deepcopy(full_acceptance_artifact)
    # Keep an explicit declaration before mutating so old artifacts cannot fail
    # only because the new field is absent.
    artifact.setdefault("scenario_names", list(REQUIRED_SCENARIOS))
    artifact["scenario_names"].remove(missing)
    artifact["scenarios"].pop(missing)
    _refresh_digest(artifact)

    with pytest.raises(ValueError, match=f"comparison acceptance scenarios missing: {missing}"):
        validate_comparison_artifact(artifact)


def test_acceptance_validator_rejects_digest_valid_duplicate_unexpected_and_disagreement(
    full_acceptance_artifact: dict,
) -> None:
    """The ordered declaration and scenario records are one exact matrix."""
    duplicate = copy.deepcopy(full_acceptance_artifact)
    duplicate.setdefault("scenario_names", list(REQUIRED_SCENARIOS))
    duplicate["scenario_names"].append("page_edit")
    _refresh_digest(duplicate)
    with pytest.raises(ValueError, match="comparison acceptance scenarios duplicated: page_edit"):
        validate_comparison_artifact(duplicate)

    unexpected = copy.deepcopy(full_acceptance_artifact)
    unexpected.setdefault("scenario_names", list(REQUIRED_SCENARIOS))
    unexpected["scenario_names"].append("unexpected_scenario")
    unexpected["scenarios"]["unexpected_scenario"] = copy.deepcopy(
        unexpected["scenarios"]["page_edit"]
    )
    _refresh_digest(unexpected)
    with pytest.raises(ValueError, match="comparison acceptance scenarios unexpected: unexpected_scenario"):
        validate_comparison_artifact(unexpected)

    disagreement = copy.deepcopy(full_acceptance_artifact)
    disagreement.setdefault("scenario_names", list(REQUIRED_SCENARIOS))
    disagreement["scenarios"].pop("page_edit")
    _refresh_digest(disagreement)
    with pytest.raises(ValueError, match="comparison acceptance scenarios declaration mismatch"):
        validate_comparison_artifact(disagreement)

    malformed = copy.deepcopy(full_acceptance_artifact)
    malformed["scenario_names"] = "page_edit"
    _refresh_digest(malformed)
    with pytest.raises(ValueError, match="comparison acceptance scenarios declaration"):
        validate_comparison_artifact(malformed)


def test_v1_policy_contract_extension_keeps_legacy_policy_and_rejects_bad_digest(tmp_path: Path) -> None:
    """The typed drift extension is backward compatible and fail closed."""
    legacy = {
        "schema_version": 1,
        "enabled": True,
        "compatibility_digest": "a" * 64,
        "evidence_observation_ids": ["snapshot-a"],
        "minimum_compatible_observations": 1,
        "max_evidence_age_seconds": 60.0,
        "match": "all",
        "criteria": [{"metric": "snapshot_p95_ms", "operator": "gte", "threshold": 1.0}],
    }
    policy = tmp_path / "legacy.json"
    policy.write_text(json.dumps(legacy), encoding="utf-8")
    loaded = load_build_mode_policy(tmp_path, policy.name)
    assert loaded.policy is not None
    assert loaded.policy.compatibility_contract is None

    bad = dict(legacy, compatibility_contract={"fts_config": {}})
    policy.write_text(json.dumps(bad), encoding="utf-8")
    rejected = load_build_mode_policy(tmp_path, policy.name)
    assert rejected.policy is None
    assert rejected.reason == "policy_schema_invalid"
