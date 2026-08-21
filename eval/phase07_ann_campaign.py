"""Typed, non-authorizing request runner for the bounded Phase 7 ANN campaign.

The CLI owns its output directory: callers never need a pre-created checkout
artifact.  It intentionally has no GitHub, installer, downloader, or policy
selection capability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable

from eval.ann_frontier_statistics import validate_declared_family

STAGES = frozenset({"screening", "confirmation", "continuation"})
SECRET_MARKERS = ("token", "secret", "password", "authorization", "private_key", "ghp_", "github_pat_")
SCREENING = {"schema_version", "stage", "request_id", "environment", "model_manifest_sha256", "corpus_manifest_sha256"}
CONFIRMATION = SCREENING | {"prior_screening_sha256", "nominated_m", "run_ordinal", "run_identity"}
CONTINUATION = SCREENING | {"mode", "prior_evidence_sha256"}
CONTINUATION_MODES = frozenset({"stage2_sq", "flat_diagnostic", "refinement", "representative_ann", "hybrid_non_regression"})


def canonical_digest(value: dict[str, Any]) -> str:
    payload = dict(value); payload.pop("record_self_sha256", None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _reject_secrets(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if any(marker in str(key).lower() for marker in SECRET_MARKERS): raise ValueError("secret-like request field")
            _reject_secrets(item)
    elif isinstance(value, list):
        for item in value: _reject_secrets(item)
    elif isinstance(value, str) and any(marker in value.lower() for marker in ("ghp_", "github_pat_", "bearer ")):
        raise ValueError("secret-like request value")


def validate_request(request: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(request, dict) or request.get("schema_version") != 1 or request.get("stage") not in STAGES:
        raise ValueError("typed Phase 7 request")
    _reject_secrets(request)
    allowed = SCREENING if request["stage"] == "screening" else CONFIRMATION if request["stage"] == "confirmation" else CONTINUATION
    if set(request) != allowed: raise ValueError("unknown or missing request field")
    if not isinstance(request["request_id"], str) or not request["request_id"]: raise ValueError("request identity")
    for name in ("model_manifest_sha256", "corpus_manifest_sha256"):
        if not isinstance(request[name], str) or len(request[name]) != 64: raise ValueError("manifest binding")
    if request["stage"] == "screening":
        return request
    if not isinstance(request["prior_screening_sha256" if request["stage"] == "confirmation" else "prior_evidence_sha256"], str): raise ValueError("immutable prior evidence binding")
    if request["stage"] == "confirmation":
        nominated = request["nominated_m"]
        if not isinstance(nominated, list) or not 1 <= len(nominated) <= 2 or any(value not in (16, 20, 32) for value in nominated): raise ValueError("at most two nominated m values")
        if not isinstance(request["run_ordinal"], int) or request["run_ordinal"] not in (1, 2, 3): raise ValueError("deterministic run ordinal")
        if not isinstance(request["run_identity"], dict) or not {"run_id", "run_attempt", "job_id", "job_allocation_nonce"} <= set(request["run_identity"]): raise ValueError("fresh hosted allocation identity")
    else:
        if request["mode"] not in CONTINUATION_MODES: raise ValueError("unsupported bounded continuation mode")
    return request


def screening_plan() -> dict[str, Any]:
    family = [{"m": m, "metric": metric, "baseline_ef": 200, "candidate_ef": 300} for m in (16,20,32) for metric in ("recall_at_10","recall_at_20")]
    validate_declared_family(family, family_name="d04_ef_300_vs_200", expected_size=6)
    return {"corpus": {"rows": 77348, "dimensions": 384, "queries": 256, "truth": "seeded_vector_exact"}, "index": {"type": "hnsw_sq", "m": [16,20,32], "ef_construction": 300}, "query_ef": [100,150,200,300], "replicates": 3, "per_build_max_seconds": 180, "authorization": "none"}


def _write(path: Path, value: dict[str, Any]) -> None:
    value["record_self_sha256"] = canonical_digest(value)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def execute(request: dict[str, Any], output_dir: Path, *, runner: Callable[[dict[str, Any]], dict[str, Any]] | None = None) -> dict[str, Any]:
    request = validate_request(request); output_dir.mkdir(parents=True, exist_ok=True)
    _write(output_dir / f"{request['stage']}-request.json", dict(request))
    ledger = {"schema_version": 1, "stage": request["stage"], "request_sha256": canonical_digest(request), "authorization": "none"}
    _write(output_dir / f"{request['stage']}-ledger.json", ledger)
    try:
        payload = screening_plan() if request["stage"] == "screening" else {"mode": request.get("mode", "confirmation"), "authorization": "none"}
        result = runner(payload) if runner else payload
        if not isinstance(result, dict) or result.get("authorization") != "none": raise ValueError("campaign results are evidence only")
        record = {"schema_version": 1, "stage": request["stage"], "request_sha256": canonical_digest(request), "result": result, "authorization": "none"}
        _write(output_dir / f"{request['stage']}-result.json", record); return record
    except Exception as exc:
        rejected = {"schema_version": 1, "stage": request["stage"], "status": "reject-evidence", "reason": f"{type(exc).__name__}: {exc}", "authorization": "none"}
        _write(output_dir / f"{request['stage']}-rejection.json", rejected); raise


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--request-file", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); args = parser.parse_args()
    try:
        request = json.loads(args.request_file.read_text(encoding="utf-8")); execute(request, args.output_dir); return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[FAIL] Phase 7 campaign: {exc}", file=os.sys.stderr); return 1

if __name__ == "__main__": raise SystemExit(main())
