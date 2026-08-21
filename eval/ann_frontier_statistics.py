"""Pure, fail-closed statistics for the Phase 7 ANN evidence campaign.

This module deliberately produces *evidence* only.  It neither imports nor
selects ``ProductionAnnPolicy``; the approval checkpoint remains outside this
utility.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


_METHOD = {
    "ci": "paired_basic_bootstrap_two_sided_95",
    "permutation": "paired_two_sided",
    "holm": "step_down",
    "resamples": 99_999,
    "batch_size": 4_096,
    "seed": "phase07-ann-statistics-v1",
    "scipy_version": "1.15.3",
}
_D04_M = (16, 20, 32)
_METRICS = ("recall_at_10", "recall_at_20")


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Mapping[str, object]) -> str:
    payload = dict(value)
    payload.pop("record_self_sha256", None)
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def _finite_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _paired(values: object, name: str) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values:
        raise ValueError(f"{name} must be a non-empty sequence of pairs")
    left: list[float] = []
    right: list[float] = []
    for index, pair in enumerate(values):
        if not isinstance(pair, Sequence) or isinstance(pair, (str, bytes)) or len(pair) != 2:
            raise ValueError(f"{name}[{index}] must contain baseline and candidate")
        left.append(_finite_number(f"{name}[{index}][0]", pair[0]))
        right.append(_finite_number(f"{name}[{index}][1]", pair[1]))
    return np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)


def _seed_for(comparison: Mapping[str, object]) -> int:
    text = f"{_METHOD['seed']}:{_canonical(dict(comparison))}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "big")


def paired_basic_effect(
    pairs: object, *, comparison: Mapping[str, object] | None = None,
    resamples: int = 99_999, batch_size: int = 4_096,
) -> dict[str, object]:
    """Return paired mean effect and a deterministic two-sided 95% basic CI."""
    if resamples != _METHOD["resamples"] or batch_size != _METHOD["batch_size"]:
        raise ValueError("Phase 7 statistics require 99,999 resamples in batches of 4,096")
    baseline, candidate = _paired(pairs, "paired_samples")
    effects = candidate - baseline
    observed = float(effects.mean())
    rng = np.random.default_rng(_seed_for(comparison or {}))
    boot = np.empty(resamples, dtype=np.float64)
    offset = 0
    while offset < resamples:
        count = min(batch_size, resamples - offset)
        indices = rng.integers(0, effects.size, size=(count, effects.size))
        boot[offset:offset + count] = effects[indices].mean(axis=1)
        offset += count
    low_q, high_q = np.quantile(boot, (0.025, 0.975), method="linear")
    return {"mean_effect": observed, "basic_ci_95": [float(2 * observed - high_q), float(2 * observed - low_q)]}


def paired_permutation_p(
    pairs: object, *, comparison: Mapping[str, object] | None = None,
    resamples: int = 99_999, batch_size: int = 4_096,
) -> float:
    """Deterministic paired, two-sided sign-flip permutation p-value."""
    if resamples != _METHOD["resamples"] or batch_size != _METHOD["batch_size"]:
        raise ValueError("Phase 7 statistics require 99,999 resamples in batches of 4,096")
    baseline, candidate = _paired(pairs, "paired_samples")
    effects = candidate - baseline
    observed = abs(float(effects.mean()))
    rng = np.random.default_rng(_seed_for({"permutation": dict(comparison or {})}))
    extreme = 0
    offset = 0
    while offset < resamples:
        count = min(batch_size, resamples - offset)
        signs = rng.integers(0, 2, size=(count, effects.size), dtype=np.int8) * 2 - 1
        extreme += int(np.count_nonzero(np.abs((signs * effects).mean(axis=1)) >= observed))
        offset += count
    return float((extreme + 1) / (resamples + 1))


def holm_adjust(p_values: Sequence[object]) -> list[float]:
    """Holm step-down adjustment for one declared family (and no other)."""
    if not p_values:
        raise ValueError("Holm family must not be empty")
    values = [_finite_number(f"p_values[{i}]", value) for i, value in enumerate(p_values)]
    if any(value < 0 or value > 1 for value in values):
        raise ValueError("p-values must be between zero and one")
    ordered = sorted(enumerate(values), key=lambda pair: pair[1])
    adjusted = [0.0] * len(values)
    running = 0.0
    size = len(values)
    for rank, (index, value) in enumerate(ordered):
        running = max(running, min(1.0, (size - rank) * value))
        adjusted[index] = running
    return adjusted


def validate_declared_family(
    family: Sequence[Mapping[str, object]], *, family_name: str,
    expected_size: int | None = None, max_size: int | None = None,
) -> None:
    if expected_size is not None and max_size is not None:
        raise ValueError("family cardinality is ambiguous")
    if expected_size is not None and len(family) != expected_size:
        raise ValueError(f"{family_name} must contain exactly {expected_size} comparisons")
    if max_size is not None and not 1 <= len(family) <= max_size:
        raise ValueError(f"{family_name} must contain between one and {max_size} comparisons")
    if not family:
        raise ValueError("family must not be empty")
    canonical = [_canonical(dict(item)) for item in family]
    if len(set(canonical)) != len(canonical):
        raise ValueError("family contains duplicate comparisons")
    if family_name == "d04_ef_300_vs_200":
        expected = [
            {"m": m, "metric": metric, "baseline_ef": 200, "candidate_ef": 300}
            for m in _D04_M for metric in _METRICS
        ]
        if list(map(dict, family)) != expected:
            raise ValueError("D-04 family must use its fixed six-member canonical order")
    elif family_name == "d20_production_baseline":
        for item in family:
            if item.get("baseline") != "production-sq" or item.get("metric") not in _METRICS:
                raise ValueError("D-20 requires a separate production-SQ Recall family")
    else:
        raise ValueError("unknown declared family")


def _validate_method(method: object) -> None:
    if not isinstance(method, Mapping):
        raise ValueError("method must be an object")
    for key, expected in _METHOD.items():
        if method.get(key) != expected:
            raise ValueError(f"method.{key} must equal its locked value")


def evaluate_paired_family(
    *, family: Sequence[Mapping[str, object]], paired_samples: Mapping[str, object], method: object,
) -> dict[str, object]:
    """Evaluate one explicitly declared family without pooling independent runs."""
    _validate_method(method)
    name = "d04_ef_300_vs_200" if len(family) == 6 else "d20_production_baseline"
    validate_declared_family(family, family_name=name, expected_size=6 if name.startswith("d04") else None,
                             max_size=4 if name.startswith("d20") else None)
    if not isinstance(paired_samples, Mapping) or set(paired_samples) != set(_METRICS):
        raise ValueError("paired_samples must contain exactly both recall metrics")
    sample_sizes = []
    for metric in _METRICS:
        baseline, candidate = _paired(paired_samples[metric], str(metric))
        if baseline.shape != candidate.shape:
            raise ValueError("mismatched paired samples")
        sample_sizes.append(baseline.size)
    if len(set(sample_sizes)) != 1:
        raise ValueError("metrics must describe the same unpooled query pairs")
    records: list[dict[str, object]] = []
    raw: list[float] = []
    for comparison in family:
        metric = comparison["metric"]
        if metric not in paired_samples:
            raise ValueError("comparison metric has no paired samples")
        effect = paired_basic_effect(paired_samples[metric], comparison=comparison)
        p_value = paired_permutation_p(paired_samples[metric], comparison=comparison)
        records.append({"comparison": dict(comparison), **effect, "raw_permutation_p": p_value})
        raw.append(p_value)
    for record, adjusted in zip(records, holm_adjust(raw), strict=True):
        record["holm_adjusted_p"] = adjusted
    result: dict[str, object] = {
        "schema_version": 1, "family_name": name, "family_size": len(records),
        "method": dict(_METHOD), "comparisons": records,
    }
    result["record_self_sha256"] = _digest(result)
    return result


def confirm_three_runs(confirmation: Mapping[str, object]) -> bool:
    """Require three fresh builds and the D-04 direction/non-regression rule."""
    if confirmation.get("screening_only") is not False:
        raise ValueError("screening evidence cannot authorize continuation")
    replicates = confirmation.get("replicates")
    if not isinstance(replicates, Sequence) or len(replicates) != 3:
        raise ValueError("exactly three independent replicate records are required")
    seen = {"run_id": set(), "job_id": set(), "build_id": set()}
    for replicate in replicates:
        if not isinstance(replicate, Mapping):
            raise ValueError("replicate must be an object")
        for field, values in seen.items():
            value = replicate.get(field)
            if not isinstance(value, str) or not value or value in values:
                raise ValueError(f"replicates require distinct {field}")
            values.add(value)
        if not isinstance(replicate.get("run_attempt"), int) or replicate["run_attempt"] < 1:
            raise ValueError("replicate run_attempt must be positive")
        if "metrics" in replicate:
            metrics = replicate["metrics"]
            if not isinstance(metrics, Mapping) or set(metrics) != set(_METRICS):
                raise ValueError("replicate metrics must contain both recall metrics")
            positive = False
            for metric in _METRICS:
                item = metrics[metric]
                if not isinstance(item, Mapping):
                    raise ValueError("metric evidence must be an object")
                effect = _finite_number("mean_effect", item.get("mean_effect"))
                ci = item.get("basic_ci_95")
                if not isinstance(ci, Sequence) or len(ci) != 2:
                    raise ValueError("basic_ci_95 must be a pair")
                lower, upper = _finite_number("ci lower", ci[0]), _finite_number("ci upper", ci[1])
                adjusted = _finite_number("holm_adjusted_p", item.get("holm_adjusted_p"))
                positive |= effect > 0 and lower > 0 and adjusted <= 0.05
                if not (positive or upper >= 0):
                    raise ValueError("other recall metric regressed")
            if not positive:
                return False
        elif replicate.get("positive_metric") not in _METRICS or replicate.get("other_non_regressing") is not True:
            return False
    return True


def confirm_stage_two_continuation(confirmation: Mapping[str, object]) -> bool:
    """Compatibility spelling used by the campaign caller and its RED contracts."""
    return confirm_three_runs(confirmation)


def select_stage2(configurations: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Return the bounded, non-authorizing D-21 Stage 2 nomination and ceiling state."""
    if not configurations:
        raise ValueError("no configurations supplied")
    required = {"m", "recall_at_10", "recall_at_20", "p95_ms", "index_bytes", "build_time_s"}
    items = [dict(item) for item in configurations]
    if any(set(item) < required for item in items):
        raise ValueError("configuration is missing D-21 measurements")
    def dominates(a: Mapping[str, object], b: Mapping[str, object]) -> bool:
        better = (float(a["recall_at_10"]) >= float(b["recall_at_10"]) and float(a["recall_at_20"]) >= float(b["recall_at_20"])
                  and float(a["p95_ms"]) <= float(b["p95_ms"]) and float(a["index_bytes"]) <= float(b["index_bytes"]))
        strict = any((float(a[k]) != float(b[k])) for k in required - {"m", "build_time_s"})
        return better and strict
    retained = [item for item in items if not any(dominates(other, item) for other in items if other is not item)]
    def recall_key(item: Mapping[str, object]) -> tuple[float, float, float, float, int]:
        return (-(float(item["recall_at_10"]) + float(item["recall_at_20"])), float(item["p95_ms"]),
                float(item["index_bytes"]), float(item["build_time_s"]), int(item["m"]))
    if len(retained) > 2:
        highest = min(retained, key=recall_key)
        lowest_p95 = min(retained, key=lambda item: (float(item["p95_ms"]), float(item["index_bytes"]), float(item["build_time_s"]), int(item["m"])))
        chosen = [highest] if highest is lowest_p95 else [highest, lowest_p95]
        if len(chosen) == 1:
            chosen.append(min((item for item in retained if item is not highest), key=recall_key))
    else:
        chosen = retained
    return {"schema_version": 1, "stage2_candidates": sorted(chosen, key=lambda item: int(item["m"])),
            "raw_sq_ceiling_open_at_ef": 500, "authorization": "none"}


def rank_raw_ceiling(configurations: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Choose the D-22 refine diagnostic only; this never chooses production policy."""
    if not configurations:
        raise ValueError("no raw SQ configurations supplied")
    candidates = [dict(item) for item in configurations]
    needed = {"m", "recall_at_10", "recall_at_20", "p95_ms", "index_bytes", "build_time_s"}
    if any(set(item) < needed for item in candidates):
        raise ValueError("raw ceiling configuration missing measurements")
    winner = min(candidates, key=lambda item: (-(float(item["recall_at_10"]) + float(item["recall_at_20"])),
                                                float(item["p95_ms"]), float(item["index_bytes"]),
                                                float(item["build_time_s"]), int(item["m"])))
    return {"schema_version": 1, "raw_sq_ceiling": winner, "refine_factors": [2, 5, 10], "authorization": "none"}


def validate_statistics_fixture(fixture: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(fixture, Mapping) or fixture.get("schema_version") != 1:
        raise ValueError("statistics fixture schema_version")
    if fixture.get("authoritative_generation") != "locked-ci-only":
        raise ValueError("statistics fixture must be generated in locked CI")
    _validate_method(fixture.get("method"))
    samples = fixture.get("paired_samples")
    if not isinstance(samples, Mapping) or set(samples) != set(_METRICS):
        raise ValueError("statistics fixture pairs")
    for metric in _METRICS:
        _paired(samples[metric], metric)
    digest = fixture.get("record_self_sha256")
    if digest not in (None, "locked-ci-populates-canonical-payload-digest"):
        if not isinstance(digest, str) or digest != _digest(fixture):
            raise ValueError("statistics fixture self digest")
    return fixture
