"""Bounded, non-authorizing Phase 7 ANN campaign runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import lancedb

from eval import benchmark_ann_build as benchmark
from eval.ann_frontier_statistics import (
    holm_adjust,
    paired_basic_effect,
    paired_permutation_p,
    validate_declared_family,
)
from eval.ann_corpus_manifest import PHASE07_CURRENT_BASELINE, phase07_current_baseline_sha256
from obsidian_wiki.infrastructure.lancedb_index_repository import LanceDbIndexRepository

SECRET_MARKERS = ("token", "secret", "password", "authorization", "private_key", "ghp_", "github_pat_")
BASE = {"schema_version", "stage", "request_id", "environment", "model_manifest_sha256", "corpus_manifest_sha256"}
STAGES = frozenset({"screening", "confirmation", "continuation"})
MODES = frozenset({"stage2_sq", "flat_diagnostic", "refinement", "representative_ann", "hybrid_non_regression"})


def canonical_digest(value: dict[str, Any]) -> str:
    payload = dict(value); payload.pop("record_self_sha256", None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _digest(name: str, value: object) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _reject_secrets(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if any(marker in str(key).lower() for marker in SECRET_MARKERS): raise ValueError("secret-like request field")
            _reject_secrets(item)
    elif isinstance(value, list):
        for item in value: _reject_secrets(item)
    elif isinstance(value, str) and any(marker in value.lower() for marker in ("ghp_", "github_pat_", "bearer ")):
        raise ValueError("secret-like request value")


def _identity(value: object) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {"run_id", "run_attempt", "job_id", "job_allocation_nonce"}:
        raise ValueError("fresh hosted allocation identity")
    if not all(isinstance(value[name], (str, int)) and not isinstance(value[name], bool) and str(value[name]) for name in value):
        raise ValueError("fresh hosted allocation identity")
    return value


def _validate_continuation_config(mode: str, config: dict[str, Any]) -> None:
    if mode == "stage2_sq":
        if set(config) != {"approved_d04_sha256", "m"} or config["m"] not in (16, 20, 32): raise ValueError("stage2 requires D-04 approval and one nominated m")
        _digest("approved_d04_sha256", config["approved_d04_sha256"])
    elif mode == "flat_diagnostic":
        if set(config) != {"no_confirmed_sq_sha256", "m", "query_ef"} or config["m"] not in (16, 20, 32) or config["query_ef"] not in (200, 300): raise ValueError("FLAT requires explicit no_confirmed_sq proof")
        _digest("no_confirmed_sq_sha256", config["no_confirmed_sq_sha256"])
    elif mode == "refinement":
        if set(config) != {"ceiling_sha256", "m", "ef_construction", "query_ef"} or config["m"] not in (16, 20, 32) or config["ef_construction"] not in (300, 500) or config["query_ef"] not in (100, 150, 200, 300, 500): raise ValueError("refinement requires one bounded raw ceiling configuration")
        _digest("ceiling_sha256", config["ceiling_sha256"])
    else:
        if set(config) != {"size", "baseline", "baseline_sha256", "finalist", "finalist_sha256"} or config["size"] not in (1000, 10000, 30000): raise ValueError("representative runs allow only pinned 1k/10k/30k corpora")
        for name in ("baseline", "finalist"):
            candidate = config[name]
            if not isinstance(candidate, dict) or set(candidate) != {"candidate", "m", "ef_construction", "query_ef", "refine_factor"} or candidate["candidate"] != "ivf-hnsw-sq" or candidate["m"] not in (16, 20, 32) or candidate["ef_construction"] not in (300, 500) or candidate["query_ef"] not in (100, 150, 200, 300, 500) or candidate["refine_factor"] not in (None, 2, 5, 10): raise ValueError("immutable representative candidate configuration")
            if canonical_digest(candidate) != config[f"{name}_sha256"]: raise ValueError("representative candidate digest mismatch")
        if config["baseline"] != PHASE07_CURRENT_BASELINE or config["baseline_sha256"] != phase07_current_baseline_sha256():
            raise ValueError("representative baseline must equal current Phase 6 identity")


def validate_request(request: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(request, dict) or request.get("schema_version") != 1 or request.get("stage") not in STAGES: raise ValueError("typed Phase 7 request")
    _reject_secrets(request); stage = request["stage"]
    extra = set() if stage == "screening" else {"prior_screening_sha256", "nominated_m", "run_ordinal", "run_identity"} if stage == "confirmation" else {"mode", "prior_evidence_sha256", "config"}
    if set(request) != BASE | extra: raise ValueError("unknown or missing request field")
    if not isinstance(request["request_id"], str) or not request["request_id"] or not isinstance(request["environment"], dict): raise ValueError("request identity/environment")
    for name in ("model_manifest_sha256", "corpus_manifest_sha256"): _digest(name, request[name])
    if stage == "confirmation":
        _digest("prior_screening_sha256", request["prior_screening_sha256"])
        # Two nominees require two requests/jobs, never one reused allocation.
        if not isinstance(request["nominated_m"], list) or len(request["nominated_m"]) != 1 or request["nominated_m"][0] not in (16, 20, 32): raise ValueError("confirmation requires exactly one separately allocated nominated m")
        if request["run_ordinal"] not in (1, 2, 3) or isinstance(request["run_ordinal"], bool): raise ValueError("deterministic run ordinal")
        _identity(request["run_identity"])
    elif stage == "continuation":
        if request["mode"] not in MODES or not isinstance(request["config"], dict): raise ValueError("unsupported bounded continuation mode")
        _digest("prior_evidence_sha256", request["prior_evidence_sha256"]); _validate_continuation_config(request["mode"], request["config"])
    return request


def screening_plan() -> dict[str, Any]:
    family = [{"m": m, "metric": metric, "baseline_ef": 200, "candidate_ef": 300} for m in (16, 20, 32) for metric in ("recall_at_10", "recall_at_20")]
    validate_declared_family(family, family_name="d04_ef_300_vs_200", expected_size=6)
    return {"corpus": {"rows": 77348, "dimensions": 384, "queries": 256, "truth": "seeded_vector_exact"}, "index": {"type": "hnsw_sq", "m": [16, 20, 32], "ef_construction": 300}, "query_ef": [100, 150, 200, 300], "per_build_max_seconds": 180, "authorization": "none"}


@dataclass(frozen=True)
class CampaignConfig:
    """Trusted Python dependency seam; the CLI/request cannot change it."""
    rows: int = 77_348; dimensions: int = 384; probes: int = 256
    per_build_max_seconds: float = 180.0; work_dir: Path = Path(".review-tmp/phase07-builds")


def _build_index_child(lance_dir: str, candidate: str, m: int, ef_construction: int,
                       rows: int, result_queue) -> None:
    """Child owns *only* the mutable LanceDB index build operation."""
    try:
        from obsidian_wiki.domain.index_models import CandidateQueryPolicy, VectorIndexConfig
        started = time.perf_counter()
        repository = LanceDbIndexRepository(
            Path(lance_dir), eval_candidate_policy=CandidateQueryPolicy(candidate=candidate, query_ef=100),
        )
        stats = repository.create_vector_index(VectorIndexConfig(
            index_type="hnsw_sq" if candidate == "ivf-hnsw-sq" else "hnsw_flat",
            metric="cosine", num_partitions=1, m=m, ef_construction=ef_construction,
            dense_chunks_count=rows,
        ))
        result_queue.put({"status": "complete", "index_build_ms": (time.perf_counter() - started) * 1000,
                          "unindexed_dense_rows": stats.unindexed_dense_rows})
    except BaseException as exc:  # child error crosses a strict serializable boundary
        result_queue.put({"status": "crashed", "reason": f"{type(exc).__name__}: {exc}"})


class Phase07AnnCampaignRunner:
    def __init__(self, config: CampaignConfig = CampaignConfig()) -> None:
        if config.rows <= config.probes or config.dimensions <= 0 or config.probes <= 0 or not 0 < config.per_build_max_seconds <= 180: raise ValueError("trusted campaign configuration")
        self.config = config

    def _truth(self, root: Path) -> tuple[tuple[tuple[str, ...], ...], float]:
        corpus = benchmark._vectors(self.config.rows, self.config.dimensions, benchmark.CORPUS_SEED); queries = benchmark._vectors(self.config.probes, self.config.dimensions, benchmark.QUERY_SEED)
        if benchmark._row_hashes(corpus) & benchmark._row_hashes(queries): raise ValueError("query/corpus overlap")
        truth_dir = root / "truth"; lancedb.connect(str(truth_dir)).create_table("dense_chunks", data=benchmark._arrow_table(corpus, [f"synthetic::{i:016x}" for i in range(self.config.rows)]))
        started = time.perf_counter(); exact = LanceDbIndexRepository(truth_dir).search_dense_exact_batch(queries.tolist(), metric="cosine", limit=20, row_batch_size=8192, query_batch_size=32)
        return exact.result_ids, (time.perf_counter() - started) * 1000

    def _build(self, root: Path, *, candidate: str = "ivf-hnsw-sq", m: int, ef_construction: int, query_ef: tuple[int, ...], exact_ids: tuple[tuple[str, ...], ...], exact_ms: float) -> dict[str, Any]:
        build_root = root / f"{candidate}-m{m}-efc{ef_construction}"
        lance_dir = build_root / "lance"
        corpus = benchmark._vectors(self.config.rows, self.config.dimensions, benchmark.CORPUS_SEED)
        queries = benchmark._vectors(self.config.probes, self.config.dimensions, benchmark.QUERY_SEED)
        lancedb.connect(str(lance_dir)).create_table("dense_chunks", data=benchmark._arrow_table(
            corpus, [f"synthetic::{index:016x}" for index in range(self.config.rows)]))
        pre_index_bytes = benchmark._directory_bytes(lance_dir)
        queue = multiprocessing.get_context("spawn").Queue()
        child = multiprocessing.get_context("spawn").Process(
            target=_build_index_child,
            args=(str(lance_dir), candidate, m, ef_construction, self.config.rows, queue),
        )
        child.start(); child.join(self.config.per_build_max_seconds)
        if child.is_alive():
            child.terminate(); child.join()
            raise RuntimeError("reject-evidence: per-build watchdog timed out")
        if child.exitcode != 0 or queue.empty():
            raise RuntimeError("reject-evidence: build worker crashed or emitted no result")
        child_result = queue.get_nowait()
        if not isinstance(child_result, dict) or child_result.get("status") != "complete":
            raise RuntimeError("reject-evidence: malformed or failed build result")
        build_ms = child_result.get("index_build_ms")
        if not isinstance(build_ms, (int, float)) or build_ms < 0 or build_ms / 1000 > self.config.per_build_max_seconds:
            raise RuntimeError("reject-evidence: invalid per-build watchdog result")
        # Parent reopens only after the child has successfully exited.  Queries
        # are intentionally after (and outside) the bounded build interval.
        from obsidian_wiki.domain.index_models import CandidateQueryPolicy
        repository = LanceDbIndexRepository(lance_dir, eval_candidate_policy=CandidateQueryPolicy(candidate=candidate, query_ef=query_ef[0]))
        records = []
        for ef in query_ef:
            samples, latencies = [], []
            for ordinal, vector in enumerate(queries):
                started = time.perf_counter(); result = repository.search_dense_eval(vector.tolist(), metric="cosine", limit=20, ef=ef)
                latencies.append((time.perf_counter() - started) * 1000)
                ids = [str(row.get("chunk_id", "")) for row in result]
                truth = list(exact_ids[ordinal])
                samples.append({"query_index": ordinal, "exact_top_10": truth[:10], "exact_top_20": truth,
                                "candidate_top_10": ids[:10], "candidate_top_20": ids,
                                "recall_at_10": len(set(truth[:10]) & set(ids[:10])) / 10,
                                "recall_at_20": len(set(truth) & set(ids)) / 20})
            records.append({"candidate": candidate, "m": m, "ef_construction": ef_construction,
                            "query_ef": ef, "build_time_ms": build_ms, "exact_time_ms": exact_ms,
                            "total_bytes": benchmark._directory_bytes(lance_dir),
                            "index_delta_bytes": benchmark._directory_bytes(lance_dir) - pre_index_bytes,
                            "latency_p50_ms": benchmark._percentile(latencies, 50), "latency_p95_ms": benchmark._percentile(latencies, 95),
                            "recall_at_10": sum(row["recall_at_10"] for row in samples) / len(samples),
                            "recall_at_20": sum(row["recall_at_20"] for row in samples) / len(samples), "queries": samples})
        build_id = hashlib.sha256(f"{lance_dir}:{build_ms}".encode()).hexdigest()
        return {"build_id": build_id, "build": {"candidate": candidate, "m": m, "ef_construction": ef_construction,
                "index_build_ms": build_ms, "index_bytes": benchmark._directory_bytes(lance_dir),
                "unindexed_dense_rows": child_result["unindexed_dense_rows"], "reopen_verified": True,
                "dense_table_open_count": repository.dense_table_open_count,
                "normal_ann_request_count": len(query_ef) * self.config.probes, "watchdog": {"owner": "parent", "cap_seconds": self.config.per_build_max_seconds, "child_exitcode": child.exitcode}}, "queries": records}

    def _with_truth(self, operation: Callable[[Path, tuple[tuple[str, ...], ...], float], dict[str, Any]]) -> dict[str, Any]:
        self.config.work_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="phase07-campaign-", dir=self.config.work_dir) as raw:
            root = Path(raw); exact, exact_ms = self._truth(root); return operation(root, exact, exact_ms)

    def screening(self) -> dict[str, Any]:
        def operation(root, exact, exact_ms):
            builds = [self._build(root, m=m, ef_construction=300, query_ef=(100, 150, 200, 300), exact_ids=exact, exact_ms=exact_ms) for m in (16, 20, 32)]
            statistics = self._screening_statistics(builds)
            nominees = sorted({record["comparison"]["m"] for record in statistics["comparisons"]
                               if record["mean_effect"] > 0 and record["basic_ci_95"][0] > 0 and record["holm_adjusted_p"] <= 0.05})[:2]
            return {"plan": screening_plan(), "exact_truth_computed_once": True, "build_count": 3,
                    "builds": builds, "d04_statistics": statistics, "nominated_m": nominees,
                    "authorization": "none"}
        return self._with_truth(operation)

    @staticmethod
    def _screening_statistics(builds: list[dict[str, Any]]) -> dict[str, Any]:
        comparisons, raw = [], []
        for build in builds:
            by_ef = {group["query_ef"]: group for group in build["queries"]}
            for metric in ("recall_at_10", "recall_at_20"):
                comparison = {"m": build["build"]["m"], "metric": metric, "baseline_ef": 200, "candidate_ef": 300}
                pairs = [[left[metric], right[metric]] for left, right in zip(by_ef[200]["queries"], by_ef[300]["queries"], strict=True)]
                effect = paired_basic_effect(pairs, comparison=comparison)
                p_value = paired_permutation_p(pairs, comparison=comparison)
                comparisons.append({"comparison": comparison, **effect, "raw_permutation_p": p_value, "paired_rows": pairs})
                raw.append(p_value)
        validate_declared_family([record["comparison"] for record in comparisons], family_name="d04_ef_300_vs_200", expected_size=6)
        for record, adjusted in zip(comparisons, holm_adjust(raw), strict=True): record["holm_adjusted_p"] = adjusted
        result = {"schema_version": 1, "family_name": "d04_ef_300_vs_200", "family_size": 6,
                  "comparisons": comparisons, "authorization": "none"}
        result["record_self_sha256"] = canonical_digest(result)
        return result

    def confirmation(self, request: dict[str, Any]) -> dict[str, Any]:
        m = request["nominated_m"][0]
        return self._with_truth(lambda root, exact, exact_ms: {"run_ordinal": request["run_ordinal"], "run_identity": request["run_identity"], "build_count": 1, "replicate": self._build(root, m=m, ef_construction=300, query_ef=(200, 300), exact_ids=exact, exact_ms=exact_ms), "authorization": "none"})

    def continuation(self, request: dict[str, Any]) -> dict[str, Any]:
        mode, config = request["mode"], request["config"]
        if mode == "stage2_sq": return self._with_truth(lambda root, exact, exact_ms: {"mode": mode, "build_count": 1, "stage2": self._build(root, m=config["m"], ef_construction=500, query_ef=(300, 500), exact_ids=exact, exact_ms=exact_ms), "ceiling_open_at_ef_500": True, "authorization": "none"})
        if mode == "flat_diagnostic": return self._with_truth(lambda root, exact, exact_ms: {"mode": mode, "diagnostic_only": True, "build_count": 1, "flat": self._build(root, candidate="ivf-hnsw-flat", m=config["m"], ef_construction=300, query_ef=(config["query_ef"],), exact_ids=exact, exact_ms=exact_ms), "authorization": "none"})
        if mode == "refinement": return self._refinement(config)
        from eval.run_eval import run_phase07_representative_campaign
        return run_phase07_representative_campaign(mode=mode, size=config["size"], baseline=config["baseline"], finalist=config["finalist"], work_dir=self.config.work_dir, authorization="none")

    def _refinement(self, config: dict[str, Any]) -> dict[str, Any]:
        def operation(root: Path, exact: tuple[tuple[str, ...], ...], exact_ms: float) -> dict[str, Any]:
            built = self._build(root, m=config["m"], ef_construction=config["ef_construction"], query_ef=(config["query_ef"],), exact_ids=exact, exact_ms=exact_ms)
            lance_dir = root / f"ivf-hnsw-sq-m{config['m']}-efc{config['ef_construction']}" / "lance"; repository = LanceDbIndexRepository(lance_dir); queries = benchmark._vectors(self.config.probes, self.config.dimensions, benchmark.QUERY_SEED); observations = []
            for factor in (2, 5, 10):
                started = time.perf_counter(); rows = []
                for ordinal, vector in enumerate(queries):
                    result = repository._dense_table().search(vector.tolist()).distance_type("cosine").ef(config["query_ef"]).refine_factor(factor).limit(20).to_list()
                    ids = [str(row.get("chunk_id", "")) for row in result]; rows.append({"query_index": ordinal, "recall_at_10": len(set(ids[:10]) & set(exact[ordinal][:10])) / 10, "recall_at_20": len(set(ids) & set(exact[ordinal])) / 20})
                observations.append({"refine_factor": factor, "query_count": len(rows), "total_query_ms": (time.perf_counter() - started) * 1000, "queries": rows})
            return {"mode": "refinement", "build_count": 1, "raw_build": built, "refinement": observations, "exact_fallback_used": False, "authorization": "none"}
        return self._with_truth(operation)

    def run(self, request: dict[str, Any]) -> dict[str, Any]:
        return self.screening() if request["stage"] == "screening" else self.confirmation(request) if request["stage"] == "confirmation" else self.continuation(request)


def _write(path: Path, value: dict[str, Any]) -> None:
    sealed = dict(value); sealed["record_self_sha256"] = canonical_digest(sealed); path.write_text(json.dumps(sealed, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def execute(request: dict[str, Any], output_dir: Path, *, runner: Callable[[dict[str, Any]], dict[str, Any]] | None = None) -> dict[str, Any]:
    request = validate_request(request); output_dir.mkdir(parents=True, exist_ok=True); _write(output_dir / f"{request['stage']}-request.json", request); _write(output_dir / f"{request['stage']}-ledger.json", {"schema_version": 1, "stage": request["stage"], "request_sha256": canonical_digest(request), "authorization": "none"})
    try:
        result = (runner or Phase07AnnCampaignRunner().run)(request)
        if not isinstance(result, dict) or result.get("authorization") != "none": raise ValueError("campaign results are evidence only")
        record = {"schema_version": 1, "stage": request["stage"], "request_sha256": canonical_digest(request), "result": result, "authorization": "none"}; _write(output_dir / f"{request['stage']}-result.json", record); return record
    except Exception as exc:
        _write(output_dir / f"{request['stage']}-rejection.json", {"schema_version": 1, "stage": request["stage"], "status": "reject-evidence", "reason": f"{type(exc).__name__}: {exc}", "authorization": "none"}); raise


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--request-file", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); args = parser.parse_args()
    try: execute(json.loads(args.request_file.read_text(encoding="utf-8")), args.output_dir); return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc: print(f"[FAIL] Phase 7 campaign: {exc}", file=os.sys.stderr); return 1


if __name__ == "__main__": raise SystemExit(main())
