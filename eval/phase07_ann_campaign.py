"""Bounded, non-authorizing Phase 7 ANN campaign runner."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import lancedb

# GitHub Actions invokes this production entry point as a file.  In that mode
# Python places ``eval/`` rather than the repository root on sys.path, so the
# package imports below would otherwise fail before the campaign can seal a
# rejection artifact.  Module execution and normal imports already include
# the repository root and therefore remain unchanged.
if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval import benchmark_ann_build as benchmark
from eval.ann_frontier_statistics import (
    holm_adjust,
    paired_basic_effect,
    paired_permutation_p,
    validate_declared_family,
)
from eval.ann_corpus_manifest import PHASE07_CURRENT_BASELINE, phase07_current_baseline_sha256
from eval.phase07_operator_gate import CONFIRMATION_WORKFLOW_INPUT_FIELDS, canonical_digest as operator_digest, validate_confirmation_workflow_input, validate_stage1_screening_runtime
from obsidian_wiki.infrastructure.lancedb_index_repository import LanceDbIndexRepository

SECRET_MARKERS = ("token", "secret", "password", "authorization", "private_key", "ghp_", "github_pat_")
BASE = {"schema_version", "stage", "request_id", "environment", "model_manifest_sha256", "corpus_manifest_sha256"}
STAGES = frozenset({"screening", "confirmation", "continuation"})
MODES = frozenset({"stage2_sq", "flat_diagnostic", "refinement", "representative_ann", "hybrid_non_regression"})
CONFIRMATION_RUNTIME_IDENTITY = {
    "python": "3.13", "lancedb": "0.34.0", "numpy": "2.2.6", "pyarrow": "25.0.0",
    "omp_num_threads": 2, "openblas_num_threads": 2, "mkl_num_threads": 2,
}
CONFIRMATION_ENVIRONMENT_FIELDS = frozenset({"head_sha", "runtime", "source_digests", "host"})
CONFIRMATION_SOURCE_DIGEST_FIELDS = frozenset({
    "requirements_sha256", "model_manifest_sha256", "corpus_manifest_sha256",
})
CONFIRMATION_HOST_FIELDS = frozenset({"os", "architecture", "image", "hostname", "cpu_count", "cpu_model"})


def canonical_digest(value: dict[str, Any]) -> str:
    payload = dict(value); payload.pop("record_self_sha256", None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _digest(name: str, value: object) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")
    return value


def _confirmation_source_digests() -> dict[str, str]:
    """Return the actual source-file identities the hosted request must record."""
    root = Path(__file__).resolve().parent.parent
    return {
        "requirements_sha256": hashlib.sha256((root / "requirements.txt").read_bytes()).hexdigest(),
        "model_manifest_sha256": hashlib.sha256((root / "eval" / "model-manifest.json").read_bytes()).hexdigest(),
        "corpus_manifest_sha256": hashlib.sha256((root / "eval" / "personal-wiki-corpus-manifest.json").read_bytes()).hexdigest(),
    }


def validate_confirmation_execution(environment: object, *, model_manifest_sha256: str | None = None,
                                    corpus_manifest_sha256: str | None = None) -> dict[str, Any]:
    """Validate the immutable runtime, source, and host identity before a build.

    Host values are measurements rather than a cross-run equality oracle, but every
    field is mandatory and the source values must describe the actual checkout.
    """
    if not isinstance(environment, dict) or set(environment) != CONFIRMATION_ENVIRONMENT_FIELDS:
        raise ValueError("strict confirmation locked execution identity")
    head = environment["head_sha"]
    if not isinstance(head, str) or len(head) != 40 or any(char not in "0123456789abcdef" for char in head):
        raise ValueError("confirmation source head")
    if environment["runtime"] != CONFIRMATION_RUNTIME_IDENTITY:
        raise ValueError("confirmation locked runtime/thread identity")
    source = environment["source_digests"]
    actual = _confirmation_source_digests()
    if not isinstance(source, dict) or set(source) != CONFIRMATION_SOURCE_DIGEST_FIELDS \
            or any(not isinstance(value, str) or len(value) != 64 for value in source.values()) \
            or source != actual:
        raise ValueError("confirmation source digest identity")
    if model_manifest_sha256 is not None and source["model_manifest_sha256"] != model_manifest_sha256:
        raise ValueError("confirmation model source digest binding")
    if corpus_manifest_sha256 is not None and source["corpus_manifest_sha256"] != corpus_manifest_sha256:
        raise ValueError("confirmation corpus source digest binding")
    host = environment["host"]
    if not isinstance(host, dict) or set(host) != CONFIRMATION_HOST_FIELDS \
            or host.get("os") != "Linux" or host.get("architecture") != "X64" \
            or not all(isinstance(host[name], str) and host[name] for name in ("image", "hostname", "cpu_model")) \
            or not isinstance(host.get("cpu_count"), int) or isinstance(host["cpu_count"], bool) or host["cpu_count"] <= 0:
        raise ValueError("confirmation host/CPU identity")
    return environment


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
    extra = {"lock_identity"} if stage == "screening" else {"workflow_inputs", "run_identity"} if stage == "confirmation" else {"mode", "prior_evidence_sha256", "config"}
    if set(request) != BASE | extra: raise ValueError("unknown or missing request field")
    if not isinstance(request["request_id"], str) or not request["request_id"] or not isinstance(request["environment"], dict): raise ValueError("request identity/environment")
    for name in ("model_manifest_sha256", "corpus_manifest_sha256"): _digest(name, request[name])
    if stage == "screening":
        validate_stage1_screening_runtime(request["environment"])
        _digest("lock_identity", request["lock_identity"])
    if stage == "confirmation":
        inputs = request["workflow_inputs"]
        if not isinstance(inputs, dict) or set(inputs) != CONFIRMATION_WORKFLOW_INPUT_FIELDS or inputs.get("record_self_sha256") != operator_digest(inputs):
            raise ValueError("confirmation requires a sealed generated workflow input")
        slot = inputs.get("slot")
        if inputs.get("campaign_stage") != "confirmation" or not isinstance(slot, dict) or set(slot) != {"m", "ordinal"} or (slot["m"], slot["ordinal"]) not in {(32, 1), (32, 2), (32, 3), (20, 1), (20, 2), (20, 3)}:
            raise ValueError("immutable confirmation slot")
        validate_confirmation_execution(
            request["environment"], model_manifest_sha256=request["model_manifest_sha256"],
            corpus_manifest_sha256=request["corpus_manifest_sha256"],
        )
        validate_confirmation_workflow_input(inputs, expected_head=request["environment"]["head_sha"])
        _identity(request["run_identity"])
    elif stage == "continuation":
        if request["mode"] not in MODES or not isinstance(request["config"], dict): raise ValueError("unsupported bounded continuation mode")
        _digest("prior_evidence_sha256", request["prior_evidence_sha256"]); _validate_continuation_config(request["mode"], request["config"])
    return request


def screening_plan(config: "CampaignConfig | None" = None) -> dict[str, Any]:
    family = [{"m": m, "metric": metric, "baseline_ef": 200, "candidate_ef": 300} for m in (16, 20, 32) for metric in ("recall_at_10", "recall_at_20")]
    validate_declared_family(family, family_name="d04_ef_300_vs_200", expected_size=6)
    config = config or CampaignConfig()
    return {"corpus": {"rows": config.rows, "dimensions": config.dimensions, "queries": config.probes, "truth": "seeded_vector_exact"}, "index": {"type": "hnsw_sq", "m": [16, 20, 32], "ef_construction": 300}, "query_ef": [100, 150, 200, 300], "per_build_max_seconds": 180, "authorization": "none"}


def select_stage1_nominees(builds: list[dict[str, Any]], statistics: dict[str, Any]) -> list[int]:
    """Return up to two D-04-qualified screening candidates in canonical rank order."""
    comparisons = statistics.get("comparisons") if isinstance(statistics, dict) else None
    if not isinstance(comparisons, list):
        raise ValueError("Stage 1 nomination statistics")
    effect_by_m: dict[int, dict[str, dict[str, Any]]] = {}
    for record in comparisons:
        comparison = record.get("comparison") if isinstance(record, dict) else None
        if not isinstance(comparison, dict) or comparison.get("m") not in (16, 20, 32) \
                or comparison.get("metric") not in {"recall_at_10", "recall_at_20"}:
            raise ValueError("Stage 1 nomination comparison")
        m, metric = comparison["m"], comparison["metric"]
        if metric in effect_by_m.setdefault(m, {}):
            raise ValueError("Stage 1 duplicate nomination comparison")
        mean, interval, holm = record.get("mean_effect"), record.get("basic_ci_95"), record.get("holm_adjusted_p")
        if isinstance(mean, bool) or not isinstance(mean, (int, float)) or not math.isfinite(mean) \
                or not isinstance(interval, list) or len(interval) != 2 \
                or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in interval) \
                or isinstance(holm, bool) or not isinstance(holm, (int, float)) or not math.isfinite(holm):
            raise ValueError("Stage 1 nomination statistic values")
        effect_by_m[m][metric] = record

    ranked: list[tuple[float, float, int, int]] = []
    for build in builds:
        card, groups = build.get("build"), build.get("queries")
        if not isinstance(card, dict) or not isinstance(groups, list) or card.get("m") not in effect_by_m:
            raise ValueError("Stage 1 nomination build")
        m = card["m"]
        metrics = effect_by_m[m]
        if set(metrics) != {"recall_at_10", "recall_at_20"}:
            raise ValueError("Stage 1 nomination metric completeness")
        qualified = any(
            record["mean_effect"] > 0 and record["basic_ci_95"][0] > 0 and record["holm_adjusted_p"] <= 0.05
            for record in metrics.values()
        ) and all(record["mean_effect"] >= 0 for record in metrics.values())
        if not qualified:
            continue
        at_300 = [group for group in groups if isinstance(group, dict) and group.get("query_ef") == 300]
        if len(at_300) != 1:
            raise ValueError("Stage 1 nominee ef=300 measurement")
        group = at_300[0]
        r10, r20, p95, index_bytes = (
            group.get("recall_at_10"), group.get("recall_at_20"),
            group.get("latency_p95_ms"), card.get("index_bytes"),
        )
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
               for value in (r10, r20, p95, index_bytes)) or p95 < 0 or index_bytes <= 0:
            raise ValueError("Stage 1 nominee measurement values")
        ranked.append((-(r10 + r20) / 2, p95, int(index_bytes), m))
    return [m for _, _, _, m in sorted(ranked)[:2]]


def _vector_digest(vectors: Any) -> str:
    """Bind the actual seeded vector values, rather than a representative manifest label."""
    return benchmark._matrix_digest(vectors)


def _stress_identity(*, config: "CampaignConfig", exact_ids: tuple[tuple[str, ...], ...]) -> dict[str, Any]:
    corpus = benchmark._vectors(config.rows, config.dimensions, benchmark.CORPUS_SEED)
    queries = benchmark._vectors(config.probes, config.dimensions, benchmark.QUERY_SEED)
    exact_digest = hashlib.sha256(json.dumps([list(ids) for ids in exact_ids], separators=(",", ":")).encode()).hexdigest()
    return {
        "schema_version": 1,
        "corpus_sha256": _vector_digest(corpus),
        "query_sha256": _vector_digest(queries),
        "exact_truth_sha256": exact_digest,
        "corpus_seed": benchmark.CORPUS_SEED,
        "query_seed": benchmark.QUERY_SEED,
        "shape": {"rows": config.rows, "dimensions": config.dimensions, "queries": config.probes},
        "algorithm": {
            "vectors": "benchmark_ann_build._vectors/v1",
            "exact_truth": "LanceDbIndexRepository.search_dense_exact_batch/cosine/limit20",
        },
    }


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
            nominees = select_stage1_nominees(builds, statistics)
            return {"plan": screening_plan(self.config), "stress_identity": _stress_identity(config=self.config, exact_ids=exact), "exact_truth_computed_once": True, "build_count": 3,
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

    @staticmethod
    def _confirmation_d20_member_statistics(baseline: dict[str, Any], primary: dict[str, Any]) -> dict[str, Any]:
        """Two paired rows from this packet; reconciliation joins both m values per ordinal."""
        baseline_group = next(group for group in baseline["queries"] if group["query_ef"] == 100)
        primary_group = next(group for group in primary["queries"] if group["query_ef"] == 300)
        comparisons = []
        for metric in ("recall_at_10", "recall_at_20"):
            comparison = {"family": "d20_current_baseline", "metric": metric,
                          "baseline_m": 16, "baseline_ef": 100,
                          "candidate_m": primary["build"]["m"], "candidate_ef": 300,
                          "baseline_build_id": baseline["build_id"], "candidate_build_id": primary["build_id"]}
            pairs = [[left[metric], right[metric]] for left, right in zip(baseline_group["queries"], primary_group["queries"], strict=True)]
            comparisons.append({"comparison": comparison, **paired_basic_effect(pairs, comparison=comparison),
                                "raw_permutation_p": paired_permutation_p(pairs, comparison=comparison), "paired_rows": pairs})
        return {"schema_version": 1, "family_name": "d20_current_baseline_member", "family_size": 2,
                "comparisons": comparisons, "authorization": "none"}

    def confirmation(self, request: dict[str, Any]) -> dict[str, Any]:
        slot = request["workflow_inputs"]["slot"]
        def operation(root, exact, exact_ms):
            builds = [self._build(root, m=m, ef_construction=300, query_ef=(100, 200, 300) if m == 16 else (200, 300), exact_ids=exact, exact_ms=exact_ms) for m in (16, 20, 32)]
            baseline, primary = next(build for build in builds if build["build"]["m"] == 16), next(build for build in builds if build["build"]["m"] == slot["m"])
            return {"slot": slot, "run_identity": request["run_identity"], "workflow_inputs_sha256": request["workflow_inputs"]["record_self_sha256"], "build_count": 3, "builds": builds, "primary_build_id": primary["build_id"], "d04_statistics": self._screening_statistics(builds), "d20_member_statistics": self._confirmation_d20_member_statistics(baseline, primary), "d20_baseline_build_id": baseline["build_id"], "d20_baseline_query_ef": 100, "locked_execution": request["environment"], "authorization": "none"}
        return self._with_truth(operation)

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


def confirmation_packet_from_result(*, result: dict[str, Any], workflow_inputs: dict[str, Any],
                                    run_id: int, run_attempt: int, job_id: int, job_key: str,
                                    job_allocation_nonce: str, raw_tree_sha256: str,
                                    replacement_for_run_id: int | None = None) -> dict[str, Any]:
    """Convert production campaign output to the strict, self-sealed packet shape."""
    if result.get("build_count") != 3 or result.get("workflow_inputs_sha256") != workflow_inputs.get("record_self_sha256"):
        raise ValueError("confirmation campaign result binding")
    validate_confirmation_execution(result.get("locked_execution"))
    expected_result_fields = {
        "slot", "run_identity", "workflow_inputs_sha256", "build_count", "builds", "primary_build_id",
        "d04_statistics", "d20_member_statistics", "d20_baseline_build_id", "d20_baseline_query_ef",
        "locked_execution", "authorization",
    }
    if set(result) != expected_result_fields or result.get("authorization") != "none" \
            or result.get("slot") != workflow_inputs.get("slot"):
        raise ValueError("strict confirmation result schema")
    d04, d20 = result.get("d04_statistics"), result.get("d20_member_statistics")
    if not isinstance(d04, dict) or not isinstance(d20, dict):
        raise ValueError("missing computed confirmation families")
    if replacement_for_run_id is not None and (
        not isinstance(replacement_for_run_id, int) or isinstance(replacement_for_run_id, bool)
        or replacement_for_run_id <= 0 or replacement_for_run_id == run_id
    ):
        raise ValueError("invalid confirmation replacement origin")
    packet = {"schema_version": 1, "campaign_stage": "confirmation", "workflow_inputs_sha256": workflow_inputs["record_self_sha256"],
              "slot": workflow_inputs["slot"], "run_id": run_id, "run_attempt": run_attempt, "job_id": job_id,
              "job_key": job_key, "job_allocation_nonce": job_allocation_nonce, "status": "numeric-success",
              "failure_class": None, "replacement_for_run_id": replacement_for_run_id,
              "builds": [{"build_id": build["build_id"], "m": build["build"]["m"], "ef_construction": build["build"]["ef_construction"], "query_ef": [group["query_ef"] for group in build["queries"]]} for build in result["builds"]],
              "d04": {"family_name": d04["family_name"], "family_size": d04["family_size"], "comparisons": d04["comparisons"], "raw_p_values": [row["raw_permutation_p"] for row in d04["comparisons"]], "holm_adjusted_p_values": [row["holm_adjusted_p"] for row in d04["comparisons"]], "basic_ci_95": [row["basic_ci_95"] for row in d04["comparisons"]]},
              "d20": {"family_name": d20["family_name"], "family_size": d20["family_size"], "baseline_build_id": result["d20_baseline_build_id"], "comparisons": d20["comparisons"], "raw_p_values": [row["raw_permutation_p"] for row in d20["comparisons"]], "basic_ci_95": [row["basic_ci_95"] for row in d20["comparisons"]]},
              "locked_execution": result["locked_execution"],
              "measurements": {"builds": result["builds"], "d04_statistics": d04, "d20_member_statistics": d20},
              "raw_tree_sha256": raw_tree_sha256, "retention_days": 90}
    packet["record_self_sha256"] = canonical_digest(packet)
    return packet


_CONFIRMATION_RAW_FILES = frozenset({
    "confirmation-request.json", "confirmation-ledger.json", "confirmation-result.json",
    "dispatch-bundle.json", "allocation.json",
})
_CONFIRMATION_ARTIFACT_FILES = _CONFIRMATION_RAW_FILES | {"confirmation-packet.json"}


def confirmation_raw_tree_sha256(root: Path, *, require_wrapper: bool = False) -> str:
    """Digest the finite confirmation evidence tree without the self-referential wrapper."""
    if not root.is_dir() or root.is_symlink():
        raise ValueError("confirmation artifact directory")
    expected_names = _CONFIRMATION_ARTIFACT_FILES if require_wrapper else _CONFIRMATION_RAW_FILES
    if {path.name for path in root.iterdir()} != expected_names:
        raise ValueError("strict confirmation artifact allowlist")
    names = set()
    digest = hashlib.sha256()
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if path.name == "confirmation-packet.json" and require_wrapper:
            if path.is_symlink() or not path.is_file():
                raise ValueError("strict confirmation wrapper")
            continue
        if path.is_symlink() or not path.is_file() or path.name not in _CONFIRMATION_RAW_FILES:
            raise ValueError("strict confirmation artifact allowlist")
        names.add(path.name)
        digest.update(path.name.encode("utf-8")); digest.update(b"\0")
        digest.update(path.read_bytes()); digest.update(b"\0")
    if names != _CONFIRMATION_RAW_FILES:
        raise ValueError("missing confirmation artifact evidence")
    return digest.hexdigest()


def confirmation_content_tree_sha256(root: Path) -> str:
    """Digest all six finite artifact files after the raw/wrapper shape check."""
    confirmation_raw_tree_sha256(root, require_wrapper=True)
    digest = hashlib.sha256()
    for name in sorted(_CONFIRMATION_ARTIFACT_FILES):
        path = root / name
        digest.update(name.encode("utf-8")); digest.update(b"\0")
        digest.update(path.read_bytes()); digest.update(b"\0")
    return digest.hexdigest()


def _sealed_confirmation_json(path: Path, *, expected: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("symlinked confirmation source")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {expected}") from exc
    if not isinstance(value, dict) or value.get("record_self_sha256") != canonical_digest(value):
        raise ValueError(f"unsealed {expected}")
    return value


def validate_confirmation_artifact_tree(
    root: Path, *, expected_head: str | None = None, expected_run_id: int | None = None,
    expected_run_attempt: int | None = None, expected_job_key: str | None = None,
) -> dict[str, Any]:
    """Validate the one uploadable confirmation tree at every trust boundary.

    This is intentionally the only validator for an exported confirmation packet.
    It checks the finite filesystem shape before opening JSON, proves the raw
    request/ledger/result chain, rebuilds the packet from raw output, and binds
    every allocation to the caller's hosted identity when supplied.
    """
    from eval.phase07_operator_gate import validate_confirmation_dispatch_bundle
    from eval.reconcile_ann_gate import validate_confirmation_packet

    if root.is_symlink() or not root.is_dir():
        raise ValueError("strict confirmation artifact root")
    if {path.name for path in root.iterdir()} != _CONFIRMATION_ARTIFACT_FILES:
        raise ValueError("strict confirmation artifact allowlist")
    for name in _CONFIRMATION_ARTIFACT_FILES:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("confirmation artifact members must be regular files")

    raw_tree_sha256 = confirmation_raw_tree_sha256(root, require_wrapper=True)
    request_record = _sealed_confirmation_json(root / "confirmation-request.json", expected="confirmation request")
    request = dict(request_record)
    request.pop("record_self_sha256", None)
    validate_request(request)
    if request_record != {**request, "record_self_sha256": canonical_digest(request)}:
        raise ValueError("confirmation request exact schema")

    ledger = _sealed_confirmation_json(root / "confirmation-ledger.json", expected="confirmation ledger")
    if set(ledger) != {"schema_version", "stage", "request_sha256", "result_sha256", "authorization", "record_self_sha256"} \
            or ledger.get("schema_version") != 1 or ledger.get("stage") != "confirmation" \
            or ledger.get("authorization") != "none" or ledger.get("request_sha256") != canonical_digest(request) \
            or ledger.get("result_sha256") != hashlib.sha256((root / "confirmation-result.json").read_bytes()).hexdigest():
        raise ValueError("confirmation raw ledger/request authorization binding")

    result_record = _sealed_confirmation_json(root / "confirmation-result.json", expected="confirmation result")
    if set(result_record) != {"schema_version", "stage", "request_sha256", "result", "authorization", "record_self_sha256"} \
            or result_record.get("schema_version") != 1 or result_record.get("stage") != "confirmation" \
            or result_record.get("authorization") != "none" or result_record.get("request_sha256") != canonical_digest(request) \
            or not isinstance(result_record.get("result"), dict):
        raise ValueError("confirmation raw result/request authorization binding")
    result = result_record["result"]
    if result.get("authorization") != "none" or result.get("locked_execution") != request["environment"]:
        raise ValueError("confirmation raw result locked-execution binding")

    # Dispatch bundles are deliberately not self-sealed: their two canonical
    # members are already self-digested and the validator regenerates both.
    try:
        bundle = json.loads((root / "dispatch-bundle.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid confirmation dispatch bundle") from exc
    if not isinstance(bundle, dict) or set(bundle) != {"confirmation_request", "workflow_input"}:
        raise ValueError("strict confirmation dispatch bundle")
    workflow_input = validate_confirmation_dispatch_bundle(
        bundle, expected_head=request["environment"]["head_sha"],
    )
    if bundle["workflow_input"] != request["workflow_inputs"]:
        raise ValueError("confirmation dispatch/request binding")

    allocation = _sealed_confirmation_json(root / "allocation.json", expected="confirmation allocation")
    if set(allocation) != {"schema_version", "campaign_stage", "workflow_inputs_sha256", "status", "allocation", "record_self_sha256"} \
            or allocation.get("schema_version") != 1 or allocation.get("campaign_stage") != "confirmation" \
            or allocation.get("status") != "success" \
            or allocation.get("workflow_inputs_sha256") != workflow_input["record_self_sha256"] \
            or not isinstance(allocation.get("allocation"), dict):
        raise ValueError("strict confirmation allocation ledger")
    identity = allocation["allocation"]
    if set(identity) != {"run_id", "run_attempt", "job_id", "job_key", "job_allocation_nonce"} \
            or not all(isinstance(identity[name], int) and identity[name] > 0 for name in ("run_id", "run_attempt", "job_id")) \
            or identity.get("job_key") != "phase07-confirmation" \
            or not isinstance(identity.get("job_allocation_nonce"), str) or len(identity["job_allocation_nonce"]) < 32:
        raise ValueError("confirmation allocation identity")
    expected_identity = {name: identity[name] for name in ("run_id", "run_attempt", "job_id", "job_allocation_nonce")}
    if result.get("workflow_inputs_sha256") != workflow_input["record_self_sha256"] \
            or result.get("slot") != workflow_input["slot"] or result.get("run_identity") != expected_identity:
        raise ValueError("confirmation result slot/allocation binding")

    wrapper = _sealed_confirmation_json(root / "confirmation-packet.json", expected="confirmation packet wrapper")
    if set(wrapper) != {"schema_version", "kind", "packet", "raw_tree_sha256", "files", "record_self_sha256"} \
            or wrapper.get("schema_version") != 1 or wrapper.get("kind") != "phase07-confirmation-packet/v1" \
            or wrapper.get("raw_tree_sha256") != raw_tree_sha256 or not isinstance(wrapper.get("files"), dict) \
            or set(wrapper["files"]) != _CONFIRMATION_RAW_FILES \
            or any(wrapper["files"][name] != hashlib.sha256((root / name).read_bytes()).hexdigest() for name in _CONFIRMATION_RAW_FILES):
        raise ValueError("confirmation packet wrapper/file digest binding")
    packet = wrapper.get("packet")
    expected_packet = confirmation_packet_from_result(
        result=result, workflow_inputs=workflow_input, run_id=identity["run_id"],
        run_attempt=identity["run_attempt"], job_id=identity["job_id"], job_key=identity["job_key"],
        job_allocation_nonce=identity["job_allocation_nonce"], raw_tree_sha256=raw_tree_sha256,
    )
    if packet != expected_packet:
        raise ValueError("confirmation packet must exactly reconstruct raw result")
    validate_confirmation_packet(packet, workflow_input)
    if expected_head is not None and request["environment"]["head_sha"] != expected_head:
        raise ValueError("confirmation finalizer head binding")
    if expected_run_id is not None and identity["run_id"] != expected_run_id:
        raise ValueError("confirmation finalizer run binding")
    if expected_run_attempt is not None and identity["run_attempt"] != expected_run_attempt:
        raise ValueError("confirmation finalizer attempt binding")
    if expected_job_key is not None and identity["job_key"] != expected_job_key:
        raise ValueError("confirmation finalizer job binding")
    return {
        "request": request_record, "ledger": ledger, "result_record": result_record, "result": result,
        "dispatch_bundle": bundle, "workflow_input": workflow_input, "allocation": allocation,
        "packet": packet, "wrapper": wrapper, "raw_tree_sha256": raw_tree_sha256,
        "content_tree_sha256": confirmation_content_tree_sha256(root),
        "raw_file_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in _CONFIRMATION_RAW_FILES},
    }


def export_confirmation_packet(*, campaign_output_dir: Path, dispatch_bundle: Path,
                               allocation_ledger: Path, artifact_dir: Path,
                               replacement_for_run_id: int | None = None) -> None:
    """Create the sole uploadable confirmation evidence tree from real campaign output."""
    from eval.phase07_operator_gate import validate_confirmation_dispatch_bundle

    if campaign_output_dir.is_symlink() or not campaign_output_dir.is_dir():
        raise ValueError("confirmation campaign output directory")
    source_names = {path.name for path in campaign_output_dir.iterdir()}
    required_source = {"confirmation-request.json", "confirmation-ledger.json", "confirmation-result.json"}
    if source_names != required_source:
        raise ValueError("strict confirmation campaign output allowlist")
    request = _sealed_confirmation_json(campaign_output_dir / "confirmation-request.json", expected="confirmation request")
    ledger = _sealed_confirmation_json(campaign_output_dir / "confirmation-ledger.json", expected="confirmation ledger")
    result_record = _sealed_confirmation_json(campaign_output_dir / "confirmation-result.json", expected="confirmation result")
    if request.get("stage") != "confirmation" or ledger.get("stage") != "confirmation" or result_record.get("stage") != "confirmation":
        raise ValueError("confirmation campaign stage")
    if result_record.get("request_sha256") != canonical_digest(request) or not isinstance(result_record.get("result"), dict):
        raise ValueError("confirmation campaign request/result binding")
    if dispatch_bundle.is_symlink() or allocation_ledger.is_symlink():
        raise ValueError("symlinked confirmation binding")
    try:
        bundle = json.loads(dispatch_bundle.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("invalid confirmation dispatch bundle") from exc
    if not isinstance(bundle, dict) or set(bundle) != {"confirmation_request", "workflow_input"}:
        raise ValueError("strict confirmation dispatch bundle")
    workflow_inputs = validate_confirmation_dispatch_bundle(bundle, expected_head=request.get("environment", {}).get("head_sha", ""))
    if bundle["workflow_input"] != request.get("workflow_inputs"):
        raise ValueError("confirmation dispatch/request mismatch")
    allocation = _sealed_confirmation_json(allocation_ledger, expected="confirmation allocation")
    if set(allocation) != {"schema_version", "campaign_stage", "workflow_inputs_sha256", "status", "allocation", "record_self_sha256"} \
            or allocation.get("schema_version") != 1 or allocation.get("campaign_stage") != "confirmation" \
            or allocation.get("status") != "success" or allocation.get("workflow_inputs_sha256") != workflow_inputs["record_self_sha256"] \
            or not isinstance(allocation.get("allocation"), dict):
        raise ValueError("strict confirmation allocation ledger")
    identity = allocation["allocation"]
    if set(identity) != {"run_id", "run_attempt", "job_id", "job_key", "job_allocation_nonce"} \
            or not all(isinstance(identity[key], int) and identity[key] > 0 for key in ("run_id", "run_attempt", "job_id")) \
            or identity.get("job_key") != "phase07-confirmation" \
            or not isinstance(identity.get("job_allocation_nonce"), str) or len(identity["job_allocation_nonce"]) < 32:
        raise ValueError("confirmation allocation identity")
    result = result_record["result"]
    expected_run_identity = {name: identity[name] for name in ("run_id", "run_attempt", "job_id", "job_allocation_nonce")}
    if result.get("workflow_inputs_sha256") != workflow_inputs["record_self_sha256"] \
            or result.get("slot") != workflow_inputs["slot"] or result.get("run_identity") != expected_run_identity:
        raise ValueError("confirmation result slot/allocation binding")
    if artifact_dir.exists():
        if artifact_dir.is_symlink() or any(artifact_dir.iterdir()):
            raise ValueError("confirmation artifact destination must be empty")
    else:
        artifact_dir.mkdir(parents=True)
    for name, source in (
        ("confirmation-request.json", campaign_output_dir / "confirmation-request.json"),
        ("confirmation-ledger.json", campaign_output_dir / "confirmation-ledger.json"),
        ("confirmation-result.json", campaign_output_dir / "confirmation-result.json"),
        ("dispatch-bundle.json", dispatch_bundle),
        ("allocation.json", allocation_ledger),
    ):
        (artifact_dir / name).write_bytes(source.read_bytes())
    raw_tree_sha256 = confirmation_raw_tree_sha256(artifact_dir)
    packet = confirmation_packet_from_result(
        result=result, workflow_inputs=workflow_inputs,
        run_id=identity["run_id"], run_attempt=identity["run_attempt"], job_id=identity["job_id"],
        job_key=identity["job_key"], job_allocation_nonce=identity["job_allocation_nonce"],
        raw_tree_sha256=raw_tree_sha256, replacement_for_run_id=replacement_for_run_id,
    )
    file_digests = {name: hashlib.sha256((artifact_dir / name).read_bytes()).hexdigest() for name in sorted(_CONFIRMATION_RAW_FILES)}
    wrapper = {"schema_version": 1, "kind": "phase07-confirmation-packet/v1", "packet": packet,
               "raw_tree_sha256": raw_tree_sha256, "files": file_digests}
    wrapper["record_self_sha256"] = canonical_digest(wrapper)
    (artifact_dir / "confirmation-packet.json").write_text(json.dumps(wrapper, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    if {path.name for path in artifact_dir.iterdir()} != _CONFIRMATION_ARTIFACT_FILES:
        raise ValueError("strict exported confirmation artifact allowlist")
    # Export cannot become a weaker trust boundary than later finalization or
    # download: re-open the finished tree through the shared strict validator.
    validate_confirmation_artifact_tree(artifact_dir)


def _trusted_test_config(value: str | None) -> CampaignConfig | None:
    """A pytest-only finite seam; ordinary CLI invocations retain immutable production scale."""
    if value is None:
        return None
    if not os.environ.get("PYTEST_CURRENT_TEST"):
        raise ValueError("trusted test config is pytest-only")
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError("trusted test config JSON") from exc
    if not isinstance(payload, dict) or set(payload) != {"rows", "dimensions", "probes", "work_dir"} \
            or not all(isinstance(payload[key], int) and not isinstance(payload[key], bool) for key in ("rows", "dimensions", "probes")) \
            or payload["dimensions"] != 384 or not 2 <= payload["probes"] < payload["rows"] <= 128 \
            or not isinstance(payload["work_dir"], str) or not payload["work_dir"]:
        raise ValueError("trusted test config")
    return CampaignConfig(rows=payload["rows"], dimensions=payload["dimensions"], probes=payload["probes"], work_dir=Path(payload["work_dir"]))


def execute(request: dict[str, Any], output_dir: Path, *, runner: Callable[[dict[str, Any]], dict[str, Any]] | None = None) -> dict[str, Any]:
    request = validate_request(request); output_dir.mkdir(parents=True, exist_ok=True); _write(output_dir / f"{request['stage']}-request.json", request)
    ledger_path = output_dir / f"{request['stage']}-ledger.json"
    ledger = {"schema_version": 1, "stage": request["stage"], "request_sha256": canonical_digest(request), "authorization": "none"}
    _write(ledger_path, ledger)
    try:
        result = (runner or Phase07AnnCampaignRunner().run)(request)
        if not isinstance(result, dict) or result.get("authorization") != "none": raise ValueError("campaign results are evidence only")
        record = {"schema_version": 1, "stage": request["stage"], "request_sha256": canonical_digest(request), "result": result, "authorization": "none"}
        result_path = output_dir / f"{request['stage']}-result.json"
        _write(result_path, record)
        if request["stage"] == "confirmation":
            ledger["result_sha256"] = hashlib.sha256(result_path.read_bytes()).hexdigest()
            _write(ledger_path, ledger)
        return record
    except Exception as exc:
        _write(output_dir / f"{request['stage']}-rejection.json", {"schema_version": 1, "stage": request["stage"], "status": "reject-evidence", "reason": f"{type(exc).__name__}: {exc}", "authorization": "none"}); raise


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "export-confirmation-packet":
        parser = argparse.ArgumentParser()
        parser.add_argument("command")
        parser.add_argument("--campaign-output-dir", type=Path, required=True)
        parser.add_argument("--dispatch-bundle", type=Path, required=True)
        parser.add_argument("--allocation-ledger", type=Path, required=True)
        parser.add_argument("--artifact-dir", type=Path, required=True)
        parser.add_argument("--replacement-for-run-id", type=int)
        args = parser.parse_args()
        try:
            export_confirmation_packet(campaign_output_dir=args.campaign_output_dir, dispatch_bundle=args.dispatch_bundle,
                                       allocation_ledger=args.allocation_ledger, artifact_dir=args.artifact_dir,
                                       replacement_for_run_id=args.replacement_for_run_id)
            return 0
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"[FAIL] confirmation packet export: {exc}", file=os.sys.stderr)
            return 1
    parser = argparse.ArgumentParser(); parser.add_argument("--request-file", type=Path, required=True); parser.add_argument("--output-dir", type=Path, required=True); parser.add_argument("--trusted-test-config"); args = parser.parse_args()
    try:
        config = _trusted_test_config(args.trusted_test_config)
        execute(json.loads(args.request_file.read_text(encoding="utf-8")), args.output_dir,
                runner=Phase07AnnCampaignRunner(config).run if config else None)
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc: print(f"[FAIL] Phase 7 campaign: {exc}", file=os.sys.stderr); return 1


if __name__ == "__main__": raise SystemExit(main())
