"""Phase 07 red contracts for paired ANN frontier statistics (D-02..D-04, D-20)."""
from __future__ import annotations

import importlib
import json
from copy import deepcopy
from pathlib import Path

import pytest


FIXTURE = Path(__file__).parent / "fixtures" / "ann_statistics_v1.json"


def _statistics():
    """Load the production statistics boundary; Phase 07 must provide it."""
    return importlib.import_module("eval.ann_frontier_statistics")


def _d04_family() -> list[dict[str, object]]:
    return [
        {"m": m, "metric": metric, "baseline_ef": 200, "candidate_ef": 300}
        for m in (16, 20, 32)
        for metric in ("recall_at_10", "recall_at_20")
    ]


def test_d02_d04_statistics_keep_effect_ci_raw_p_and_holm_separate() -> None:
    """D-02/D-04: six ordered paired comparisons use the production utility."""
    statistics = _statistics()
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    result = statistics.evaluate_paired_family(
        family=_d04_family(),
        paired_samples=fixture["paired_samples"],
        method=fixture["method"],
    )
    assert result["family_size"] == 6
    assert [record["comparison"] for record in result["comparisons"]] == _d04_family()
    for record in result["comparisons"]:
        assert set(record) >= {"mean_effect", "basic_ci_95", "raw_permutation_p", "holm_adjusted_p"}
        assert record["basic_ci_95"][0] <= record["basic_ci_95"][1]
        assert 0 <= record["raw_permutation_p"] <= 1
        assert 0 <= record["holm_adjusted_p"] <= 1


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["paired_samples"].update({"recall_at_10": [[0.1, 0.2], [float("nan"), 0.3]]}),
        lambda value: value["paired_samples"].update({"recall_at_20": [[0.1, 0.2]]}),
        lambda value: value["method"].update({"resamples": 999}),
    ],
    ids=("non-finite", "mismatched-pairs", "changed-method"),
)
def test_d02_rejects_malformed_pooled_or_nonfinite_paired_evidence(mutation) -> None:
    """D-02/D-03: independent rebuilds cannot be pooled or silently repaired."""
    statistics = _statistics()
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    mutation(fixture)
    with pytest.raises(ValueError):
        statistics.evaluate_paired_family(
            family=_d04_family(), paired_samples=fixture["paired_samples"], method=fixture["method"],
        )


def test_d04_and_d20_families_are_distinct_and_have_locked_cardinalities() -> None:
    """D-04 uses six ef comparisons; D-20 is a separate at-most-four baseline family."""
    statistics = _statistics()
    d20_family = [
        {"m": 16, "metric": "recall_at_10", "baseline": "production-sq"},
        {"m": 16, "metric": "recall_at_20", "baseline": "production-sq"},
    ]
    statistics.validate_declared_family(_d04_family(), family_name="d04_ef_300_vs_200", expected_size=6)
    statistics.validate_declared_family(d20_family, family_name="d20_production_baseline", max_size=4)
    with pytest.raises(ValueError):
        statistics.validate_declared_family(_d04_family() + d20_family, family_name="d04_ef_300_vs_200", expected_size=6)


def test_d03_d04_confirmation_requires_three_distinct_runs_and_fresh_builds() -> None:
    """D-03/D-04: screening evidence cannot authorize Stage 2 or reuse a hosted allocation."""
    statistics = _statistics()
    confirmation = {
        "screening_only": False,
        "replicates": [
            {"run_id": "run-1", "run_attempt": 1, "job_id": "job-1", "build_id": "build-1", "positive_metric": "recall_at_10", "other_non_regressing": True},
            {"run_id": "run-2", "run_attempt": 1, "job_id": "job-2", "build_id": "build-2", "positive_metric": "recall_at_10", "other_non_regressing": True},
            {"run_id": "run-3", "run_attempt": 1, "job_id": "job-3", "build_id": "build-3", "positive_metric": "recall_at_10", "other_non_regressing": True},
        ],
    }
    assert statistics.confirm_stage_two_continuation(confirmation) is True
    for field in ("run_id", "job_id", "build_id"):
        broken = deepcopy(confirmation)
        broken["replicates"][2][field] = broken["replicates"][1][field]
        with pytest.raises(ValueError):
            statistics.confirm_stage_two_continuation(broken)
    with pytest.raises(ValueError):
        statistics.confirm_stage_two_continuation({"screening_only": True, "replicates": confirmation["replicates"]})


def test_locked_scipy_golden_fixture_is_self_sealed_but_not_locally_generated() -> None:
    """D-02/D-18: CI numerics are generated only by locked SciPy 1.15.3 CI."""
    statistics = _statistics()
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert fixture["method"]["scipy_version"] == "1.15.3"
    assert fixture["authoritative_generation"] == "locked-ci-only"
    assert statistics.validate_statistics_fixture(fixture) is fixture
