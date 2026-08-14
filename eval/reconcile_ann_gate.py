"""Fail-closed PR reconciliation for held-out ANN decision artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from benchmark_ann_build import validate_evidence
from run_eval import validate_candidate_decision_records


ARCHITECTURE_REQUIRED_CHECKS = (
    "Architecture (ubuntu-latest, Python 3.10)",
    "Architecture (ubuntu-latest, Python 3.13)",
    "Architecture (windows-latest, Python 3.10)",
    "Architecture (windows-latest, Python 3.13)",
)


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"artifact unavailable or invalid: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must be an object: {path}")
    return payload


def _canonical_sha256(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def reconcile(
    *, scale_evidence: Path, model_records: Path, conclusions: dict[str, str], expected_head: str
) -> dict:
    """Validate one same-head numeric-success evidence set before reporting it."""
    for name in ("test-and-eval", "scale", "model-backed"):
        if conclusions.get(name) != "success":
            raise ValueError(f"required job {name} != \"success\"")
    if not isinstance(expected_head, str) or len(expected_head) != 40:
        raise ValueError("expected head")

    scale = validate_evidence(_read_json(scale_evidence))
    source = scale["source"]
    if source["head_sha"] != expected_head:
        raise ValueError("scale artifact head")
    model = validate_candidate_decision_records(_read_json(model_records), scale)
    if model.get("head_sha") != expected_head:
        raise ValueError("model artifact head")

    return {
        "schema_version": 1,
        "all_required_jobs_numeric_success": True,
        "head_sha": expected_head,
        "comparator_sha256": _canonical_sha256(scale),
        "scale_lock_identity": source["lock_identity"],
        "scale_runtime_identity": scale["environment"]["runtime"],
        "job_conclusions": conclusions,
        "branch_protection_requirements": list(ARCHITECTURE_REQUIRED_CHECKS),
        "artifacts": {
            "scale_evidence": str(scale_evidence),
            "model_records": str(model_records),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale-evidence", type=Path, required=True)
    parser.add_argument("--model-records", type=Path, required=True)
    parser.add_argument("--test-and-eval", required=True)
    parser.add_argument("--scale", required=True)
    parser.add_argument("--model-backed", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = reconcile(
            scale_evidence=args.scale_evidence,
            model_records=args.model_records,
            conclusions={
                "test-and-eval": args.test_and_eval,
                "scale": args.scale,
                "model-backed": args.model_backed,
            },
            expected_head=args.expected_head,
        )
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"[FAIL] ANN reconciliation: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print("[PASS] ANN reconciliation", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
