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

import lancedb
from eval import benchmark_ann_build as benchmark
from obsidian_wiki.infrastructure.lancedb_index_repository import LanceDbIndexRepository

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


class Phase07AnnCampaignRunner:
    """Production LanceDB campaign seam; tests only shrink its numeric inputs."""
    def __init__(self, *, rows: int = 77_348, dimensions: int = 384, probes: int = 256,
                 work_dir: Path | None = None, per_build_max_seconds: float = 180.0) -> None:
        self.rows, self.dimensions, self.probes = rows, dimensions, probes
        self.work_dir = work_dir or Path(".review-tmp/phase07-builds")
        self.per_build_max_seconds = per_build_max_seconds

    def _truth(self, root: Path) -> tuple[tuple[tuple[str, ...], ...], float]:
        corpus = benchmark._vectors(self.rows, self.dimensions, benchmark.CORPUS_SEED)
        queries = benchmark._vectors(self.probes, self.dimensions, benchmark.QUERY_SEED)
        if benchmark._row_hashes(corpus) & benchmark._row_hashes(queries): raise ValueError("query/corpus overlap")
        truth_dir = root / "truth"; lancedb.connect(str(truth_dir)).create_table("dense_chunks", data=benchmark._arrow_table(corpus, [f"synthetic::{i:016x}" for i in range(self.rows)]))
        import time
        started = time.perf_counter(); exact = LanceDbIndexRepository(truth_dir).search_dense_exact_batch(queries.tolist(), metric="cosine", limit=20, row_batch_size=8192, query_batch_size=32)
        return exact.result_ids, (time.perf_counter() - started) * 1000

    def _build(self, root: Path, *, m: int, ef_construction: int, query_ef: tuple[int, ...], exact_ids: tuple[tuple[str, ...], ...], exact_ms: float) -> dict[str, Any]:
        run, records = benchmark._candidate_worker("ivf-hnsw-sq", str(root / f"m{m}-efc{ef_construction}"), self.rows, self.dimensions, self.probes, query_ef, exact_ids, exact_ms, m=m, ef_construction=ef_construction)
        if run["index_build_ms"] / 1000 > self.per_build_max_seconds: raise RuntimeError("reject-evidence: per-build watchdog")
        return {"build": run, "queries": records}

    def screening(self) -> dict[str, Any]:
        import tempfile
        self.work_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="phase07-campaign-", dir=self.work_dir) as raw:
            root=Path(raw); exact, exact_ms=self._truth(root)
            builds=[self._build(root, m=m, ef_construction=300, query_ef=(100,150,200,300), exact_ids=exact, exact_ms=exact_ms) for m in (16,20,32)]
        return {"plan": screening_plan(), "exact_truth_computed_once": True, "build_count": 3, "builds": builds, "authorization": "none"}

    def confirmation(self, request: dict[str, Any]) -> dict[str, Any]:
        import tempfile
        with tempfile.TemporaryDirectory(prefix="phase07-confirm-", dir=self.work_dir) as raw:
            root=Path(raw); exact, exact_ms=self._truth(root); build=self._build(root, m=request["nominated_m"][0], ef_construction=300, query_ef=(200,300), exact_ids=exact, exact_ms=exact_ms)
        return {"replicate": build, "run_identity": request["run_identity"], "run_ordinal": request["run_ordinal"], "authorization":"none"}

    def continuation(self, request: dict[str, Any]) -> dict[str, Any]:
        mode=request["mode"]
        if mode not in {"stage2_sq", "flat_diagnostic", "refinement"}: raise ValueError("model-backed continuation requires run_eval production facade binding")
        return {"mode": mode, "prior_evidence_sha256": request["prior_evidence_sha256"], "authorization":"none"}


def _write(path: Path, value: dict[str, Any]) -> None:
    value["record_self_sha256"] = canonical_digest(value)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def execute(request: dict[str, Any], output_dir: Path, *, runner: Callable[[dict[str, Any]], dict[str, Any]] | None = None) -> dict[str, Any]:
    request = validate_request(request); output_dir.mkdir(parents=True, exist_ok=True)
    _write(output_dir / f"{request['stage']}-request.json", dict(request))
    ledger = {"schema_version": 1, "stage": request["stage"], "request_sha256": canonical_digest(request), "authorization": "none"}
    _write(output_dir / f"{request['stage']}-ledger.json", ledger)
    try:
        if runner:
            result = runner(screening_plan() if request["stage"] == "screening" else {"mode": request.get("mode", "confirmation"), "authorization": "none"})
        else:
            production = Phase07AnnCampaignRunner()
            result = production.screening() if request["stage"] == "screening" else production.confirmation(request) if request["stage"] == "confirmation" else production.continuation(request)
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
