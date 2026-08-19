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
    run_mode_comparison,
    validate_comparison_artifact,
)


def _run(tmp_path: Path, *, scenarios: tuple[str, ...]) -> dict:
    output = tmp_path / "equivalence.json"
    artifact = run_mode_comparison(
        work_dir=tmp_path / "work",
        output=output,
        scenarios=scenarios,
    )
    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    return artifact


def test_public_snapshot_incremental_retrieval_equivalence(tmp_path: Path) -> None:
    """An edit uses public build/load/hybrid_search in both build modes."""
    artifact = _run(tmp_path, scenarios=("page_edit",))

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


def test_scenario_matrix_and_artifact_fail_closed(tmp_path: Path) -> None:
    """Every logical mutation and public fallback leaves complete evidence."""
    baseline = (SKILL_ROOT / "eval" / "baselines.json").read_bytes()
    ann_policy = (SKILL_ROOT / "eval" / "ann-policy.json").read_bytes()
    artifact = _run(tmp_path, scenarios=REQUIRED_SCENARIOS)

    validate_comparison_artifact(artifact)
    assert artifact["verdict"] == "pass"
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
