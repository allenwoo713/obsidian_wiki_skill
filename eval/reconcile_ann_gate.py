"""Fail-closed PR reconciliation for held-out ANN decision artifacts."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import sys
import re
from pathlib import Path
from typing import Any

import benchmark_ann_build as benchmark
from benchmark_ann_build import validate_evidence
from run_eval import validate_candidate_decision_records
from eval.phase07_ann_campaign import Phase07AnnCampaignRunner, canonical_digest as campaign_digest


PHASE07_PACKET_FIELDS = frozenset({
    "repository", "run_id", "run_attempt", "job_id", "job_allocation_nonce",
    "artifact_id", "artifact_name", "archive_sha256", "content_sha256",
    "record_self_sha256", "retention_days_requested", "retention_days_accepted",
    "head_sha", "runner", "lock_identity", "build_id", "retry_lineage",
})
PHASE07_INFRA_FAILURES = frozenset({
    "github_infrastructure", "hosted_runner_unavailable", "artifact_service",
})
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_SECRET_MARKERS = ("token", "secret", "password", "authorization", "private_key", "ghp_")
_STAGE1_FILES = frozenset({
    "archive.zip", "extracted",
})
_STAGE1_EXTRACTED_FILES = frozenset({
    "screening-request.json", "screening-ledger.json", "screening-result.json",
})
_STAGE1_REQUEST_FIELDS = frozenset({
    "schema_version", "mode", "authorization", "repository", "branch", "workflow_path", "head_sha", "run_id", "run_attempt",
    "job_id", "job_key", "job_allocation_nonce", "runtime", "model_manifest_sha256",
    "corpus_manifest_sha256", "workflow_name", "event", "status", "conclusion", "runner",
    "lock_identity", "retry_lineage", "artifact", "campaign_result_sha256",
    "campaign_request_sha256", "campaign_ledger_sha256", "record_self_sha256",
})


def _reject_secrets(value, location="packet") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if any(marker in str(key).lower() for marker in _SECRET_MARKERS):
                raise ValueError(f"secret-like packet field: {location}.{key}")
            _reject_secrets(item, f"{location}.{key}")
    elif isinstance(value, list):
        for item in value:
            _reject_secrets(item, location)
    elif isinstance(value, str) and any(marker in value.lower() for marker in ("ghp_", "github_pat_", "bearer ")):
        raise ValueError("secret-like packet value")


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


def _stage1_reject_secrets(value: Any, location: str = "request") -> None:
    """Allow the explicit no-authority marker, reject every other credential-shaped value."""
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered == "authorization" and item == "none":
                continue
            if any(marker in lowered for marker in _SECRET_MARKERS):
                raise ValueError(f"secret-like Stage 1 field: {location}.{key}")
            _stage1_reject_secrets(item, f"{location}.{key}")
    elif isinstance(value, list):
        for item in value:
            _stage1_reject_secrets(item, location)
    elif isinstance(value, str) and any(marker in value.lower() for marker in ("ghp_", "github_pat_", "bearer ")):
        raise ValueError("secret-like Stage 1 value")


def _stage1_file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _stage1_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("Stage 1 symlinked artifact content")
        if path.is_file():
            digest.update(path.relative_to(root).as_posix().encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        elif not path.is_dir():
            raise ValueError("unknown Stage 1 artifact entry")
    return digest.hexdigest()


def _stage1_read(path: Path) -> dict[str, Any]:
    payload = _read_json(path)
    if payload.get("record_self_sha256") != campaign_digest(payload):
        raise ValueError(f"Stage 1 artifact self-digest mismatch: {path.name}")
    return payload


def _stage1_stress_digest(exact_ids: list[list[str]]) -> str:
    return hashlib.sha256(json.dumps(exact_ids, separators=(",", ":")).encode()).hexdigest()


def _validate_stage1_request(request: dict[str, Any], artifact_dir: Path) -> None:
    _stage1_reject_secrets(request)
    if set(request) != _STAGE1_REQUEST_FIELDS or request["schema_version"] != 1 or request["mode"] != "screening" or request["authorization"] != "none":
        raise ValueError("strict Stage 1 post-download request schema")
    if request["record_self_sha256"] != _canonical_sha256({key: value for key, value in request.items() if key != "record_self_sha256"}):
        raise ValueError("Stage 1 request self-digest mismatch")
    if not isinstance(request["repository"], str) or not request["repository"] or not isinstance(request["branch"], str) or request["branch"] in {"", "main", "master"} or request["workflow_path"] != ".github/workflows/eval.yml" or not isinstance(request["head_sha"], str) or not re.fullmatch(r"[0-9a-f]{40}", request["head_sha"]):
        raise ValueError("Stage 1 repository/head identity")
    if request["workflow_name"] != "eval.yml" or request["event"] != "workflow_dispatch" or request["status"] != "completed" or request["conclusion"] != "success":
        raise ValueError("Stage 1 run must be completed workflow-dispatch success")
    if not all(isinstance(request[name], int) and request[name] > 0 for name in ("run_id", "run_attempt", "job_id")):
        raise ValueError("Stage 1 run/attempt/job identity")
    if not isinstance(request["job_key"], str) or not request["job_key"] or request["job_allocation_nonce"] != f"{request['run_id']}-{request['run_attempt']}-{request['job_key']}":
        raise ValueError("Stage 1 job allocation binding")
    if request["runtime"] != {"python": "3.13", "lancedb": "0.34.0", "numpy": "2.2.6", "pyarrow": "25.0.0", "omp_num_threads": 2}:
        raise ValueError("Stage 1 locked runtime/OMP identity")
    if not all(isinstance(request[name], str) and _HEX64.fullmatch(request[name]) for name in ("model_manifest_sha256", "corpus_manifest_sha256", "lock_identity", "campaign_result_sha256", "campaign_request_sha256", "campaign_ledger_sha256")):
        raise ValueError("Stage 1 digest identity")
    runner = request["runner"]
    if not isinstance(runner, dict) or set(runner) != {"name", "group", "labels", "os", "image", "architecture"} or not all(isinstance(runner[name], str) and runner[name] for name in ("name", "group", "os", "image", "architecture")) or not isinstance(runner["labels"], list) or not runner["labels"]:
        raise ValueError("Stage 1 runner identity")
    retry = request["retry_lineage"]
    if not isinstance(retry, dict) or set(retry) != {"failure_class", "original_run_id", "replacement_run_id"} or retry != {"failure_class": None, "original_run_id": None, "replacement_run_id": None}:
        raise ValueError("Stage 1 retry lineage")
    artifact = request["artifact"]
    required_artifact = {"artifact_id", "name", "job_id", "job_key", "retention_days_requested", "retention_days_accepted", "created_at", "expires_at", "expired", "api_archive_sha256", "local_archive_path", "local_archive_sha256", "content_tree_sha256"}
    if not isinstance(artifact, dict) or set(artifact) != required_artifact or not isinstance(artifact["artifact_id"], int) or artifact["artifact_id"] <= 0 or artifact["job_id"] != request["job_id"] or artifact["job_key"] != request["job_key"] or not isinstance(artifact["name"], str) or not artifact["name"] or artifact["retention_days_requested"] != 90 or artifact["retention_days_accepted"] != 90 or artifact["expired"] is not False:
        raise ValueError("Stage 1 artifact identity/retention")
    try:
        created = dt.datetime.fromisoformat(artifact["created_at"].replace("Z", "+00:00"))
        expires = dt.datetime.fromisoformat(artifact["expires_at"].replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ValueError("Stage 1 artifact timestamps") from exc
    if not dt.timedelta(days=89, hours=23, minutes=59, seconds=30) <= expires - created <= dt.timedelta(days=90):
        raise ValueError("Stage 1 artifact retention interval")
    if not isinstance(artifact["local_archive_path"], str) or Path(artifact["local_archive_path"]).name != artifact["local_archive_path"]:
        raise ValueError("Stage 1 local archive path")
    archive = artifact_dir / artifact["local_archive_path"]
    if archive.is_symlink() or not archive.is_file() or _stage1_file_sha256(archive) != artifact["local_archive_sha256"] or artifact["api_archive_sha256"] != artifact["local_archive_sha256"]:
        raise ValueError("Stage 1 archive digest binding")
    extracted = artifact_dir / "extracted"
    if extracted.is_symlink() or not extracted.is_dir() or not all(isinstance(artifact[name], str) and _HEX64.fullmatch(artifact[name]) for name in ("api_archive_sha256", "local_archive_sha256", "content_tree_sha256")) or _stage1_tree_sha256(extracted) != artifact["content_tree_sha256"]:
        raise ValueError("Stage 1 content-tree digest binding")


def _validate_stage1_result(result_record: dict[str, Any], request_record: dict[str, Any], *, expected_shape: tuple[int, int, int]) -> dict[str, Any]:
    if set(result_record) != {"schema_version", "stage", "request_sha256", "result", "authorization", "record_self_sha256"} or result_record["schema_version"] != 1 or result_record["stage"] != "screening" or result_record["authorization"] != "none" or result_record["request_sha256"] != campaign_digest(request_record):
        raise ValueError("Stage 1 result/request binding")
    result = result_record["result"]
    if not isinstance(result, dict) or result.get("authorization") != "none" or result.get("build_count") != 3 or result.get("exact_truth_computed_once") is not True:
        raise ValueError("Stage 1 screening result shape")
    plan = result.get("plan")
    if not isinstance(plan, dict) or plan.get("index") != {"type": "hnsw_sq", "m": [16, 20, 32], "ef_construction": 300} or plan.get("query_ef") != [100, 150, 200, 300] or plan.get("per_build_max_seconds") != 180:
        raise ValueError("Stage 1 SQ-first plan")
    stress = result.get("stress_identity")
    expected_stress_shape = {"rows": expected_shape[0], "dimensions": expected_shape[1], "queries": expected_shape[2]}
    expected_algorithm = {
        "vectors": "benchmark_ann_build._vectors/v1",
        "exact_truth": "LanceDbIndexRepository.search_dense_exact_batch/cosine/limit20",
    }
    expected_corpus = benchmark._matrix_digest(benchmark._vectors(expected_shape[0], expected_shape[1], benchmark.CORPUS_SEED))
    expected_queries = benchmark._matrix_digest(benchmark._vectors(expected_shape[2], expected_shape[1], benchmark.QUERY_SEED))
    if not isinstance(stress, dict) or set(stress) != {"schema_version", "corpus_sha256", "query_sha256", "exact_truth_sha256", "corpus_seed", "query_seed", "shape", "algorithm"} or stress.get("schema_version") != 1 or stress.get("corpus_seed") != benchmark.CORPUS_SEED or stress.get("query_seed") != benchmark.QUERY_SEED or stress.get("shape") != expected_stress_shape or stress.get("algorithm") != expected_algorithm or stress.get("corpus_sha256") != expected_corpus or stress.get("query_sha256") != expected_queries or not isinstance(stress.get("exact_truth_sha256"), str) or not _HEX64.fullmatch(stress["exact_truth_sha256"]):
        raise ValueError("Stage 1 stress identity")
    if plan.get("corpus") != {"rows": expected_shape[0], "dimensions": expected_shape[1], "queries": expected_shape[2], "truth": "seeded_vector_exact"}:
        raise ValueError("Stage 1 plan/stress shape binding")
    builds = result.get("builds")
    if not isinstance(builds, list) or len(builds) != 3:
        raise ValueError("Stage 1 requires exactly three builds")
    observed_m, build_ids, truth_identity = set(), set(), None
    for build in builds:
        if not isinstance(build, dict) or set(build) != {"build_id", "build", "queries"} or not isinstance(build["build_id"], str) or not _HEX64.fullmatch(build["build_id"]) or build["build_id"] in build_ids:
            raise ValueError("Stage 1 fresh build identity")
        build_ids.add(build["build_id"])
        card = build["build"]
        watchdog = card.get("watchdog", {}) if isinstance(card, dict) else {}
        if not isinstance(card, dict) or card.get("candidate") != "ivf-hnsw-sq" or card.get("m") not in {16, 20, 32} or card.get("ef_construction") != 300 or card.get("normal_ann_request_count") != 4 * stress["shape"]["queries"] or watchdog.get("cap_seconds") != 180 or watchdog.get("owner") != "parent" or watchdog.get("child_exitcode") != 0 or card.get("reopen_verified") is not True or card.get("unindexed_dense_rows") != 0 or not isinstance(card.get("index_build_ms"), (int, float)) or not math.isfinite(card["index_build_ms"]) or not 0 <= card["index_build_ms"] <= 180_000 or not isinstance(card.get("index_bytes"), (int, float)) or not math.isfinite(card["index_bytes"]) or card["index_bytes"] <= 0:
            raise ValueError("Stage 1 build card/watchdog")
        observed_m.add(card["m"])
        groups = build["queries"]
        if not isinstance(groups, list) or [group.get("query_ef") for group in groups if isinstance(group, dict)] != [100, 150, 200, 300]:
            raise ValueError("Stage 1 query EF matrix")
        for group in groups:
            samples = group.get("queries") if isinstance(group, dict) else None
            if not isinstance(samples, list) or len(samples) != stress["shape"]["queries"]:
                raise ValueError("Stage 1 query sample cardinality")
            exact_ids = []
            for ordinal, sample in enumerate(samples):
                if not isinstance(sample, dict) or sample.get("query_index") != ordinal:
                    raise ValueError("Stage 1 query sample identity")
                exact10, exact20 = sample.get("exact_top_10"), sample.get("exact_top_20")
                candidate10, candidate20 = sample.get("candidate_top_10"), sample.get("candidate_top_20")
                if not all(isinstance(values, list) for values in (exact10, exact20, candidate10, candidate20)) or len(exact10) != 10 or len(exact20) != 20 or len(candidate10) != 10 or len(candidate20) != 20:
                    raise ValueError("Stage 1 exact truth sample")
                expected10 = len(set(exact10) & set(candidate10[:10])) / 10
                expected20 = len(set(exact20) & set(candidate20[:20])) / 20
                if sample.get("recall_at_10") != expected10 or sample.get("recall_at_20") != expected20:
                    raise ValueError("Stage 1 per-query recall recomputation")
                exact_ids.append(exact20)
            expected_group10 = sum(sample["recall_at_10"] for sample in samples) / len(samples)
            expected_group20 = sum(sample["recall_at_20"] for sample in samples) / len(samples)
            if group.get("recall_at_10") != expected_group10 or group.get("recall_at_20") != expected_group20 or not all(isinstance(group.get(name), (int, float)) and math.isfinite(group[name]) and group[name] >= 0 for name in ("build_time_ms", "exact_time_ms", "total_bytes", "index_delta_bytes", "latency_p50_ms", "latency_p95_ms")):
                raise ValueError("Stage 1 group aggregate/measurement recomputation")
            digest = _stage1_stress_digest(exact_ids)
            if truth_identity is None:
                truth_identity = digest
            elif digest != truth_identity:
                raise ValueError("Stage 1 exact truth differs across build/query records")
    if observed_m != {16, 20, 32} or truth_identity != stress["exact_truth_sha256"]:
        raise ValueError("Stage 1 distinct m/truth identity")
    statistics = result.get("d04_statistics")
    if statistics != Phase07AnnCampaignRunner._screening_statistics(builds):
        raise ValueError("Stage 1 D-04 six-member statistics recomputation")
    expected_nominees = sorted({record["comparison"]["m"] for record in statistics["comparisons"] if record["mean_effect"] > 0 and record["basic_ci_95"][0] > 0 and record["holm_adjusted_p"] <= 0.05})[:2]
    if result.get("nominated_m") != expected_nominees or len(expected_nominees) > 2:
        raise ValueError("Stage 1 nomination boundary")
    return result


def reconcile_stage1_screening(*, stage1_request: Path, artifact_dir: Path, output: Path, mode: str, expected_shape: tuple[int, int, int] = (77_348, 384, 256)) -> dict[str, Any]:
    """Recompute and seal one complete, non-authorizing Stage 1 screening artifact."""
    if mode != "screening" or artifact_dir.is_symlink() or not artifact_dir.is_dir():
        raise ValueError("strict Stage 1 reconciliation mode/artifact directory")
    found = {path.name for path in artifact_dir.iterdir()}
    if found != _STAGE1_FILES or any(path.is_symlink() for path in artifact_dir.iterdir()):
        raise ValueError("Stage 1 artifact has extra, missing, or symlinked files")
    extracted = artifact_dir / "extracted"
    extracted_found = {path.name for path in extracted.iterdir()}
    if any(name.endswith("-rejection.json") for name in extracted_found):
        raise ValueError("Stage 1 rejection artifact is never successful evidence")
    if extracted_found != _STAGE1_EXTRACTED_FILES or any(path.is_symlink() or not path.is_file() for path in extracted.iterdir()):
        raise ValueError("Stage 1 extracted content has extra, missing, or symlinked files")
    request = _read_json(stage1_request)
    _validate_stage1_request(request, artifact_dir)
    request_record = _stage1_read(extracted / "screening-request.json")
    ledger_record = _stage1_read(extracted / "screening-ledger.json")
    result_record = _stage1_read(extracted / "screening-result.json")
    if _stage1_file_sha256(extracted / "screening-request.json") != request["campaign_request_sha256"] or _stage1_file_sha256(extracted / "screening-ledger.json") != request["campaign_ledger_sha256"] or _stage1_file_sha256(extracted / "screening-result.json") != request["campaign_result_sha256"]:
        raise ValueError("Stage 1 post-download file digest binding")
    environment = request_record.get("environment", {})
    if environment.get("branch") != request["branch"] or environment.get("workflow_path") != request["workflow_path"] or environment.get("head_sha") != request["head_sha"] or environment.get("run_id") != request["run_id"] or environment.get("run_attempt") != request["run_attempt"] or environment.get("job_key") != request["job_key"] or environment.get("job_allocation_nonce") != request["job_allocation_nonce"] or environment.get("runtime") != request["runtime"] or request_record.get("model_manifest_sha256") != request["model_manifest_sha256"] or request_record.get("corpus_manifest_sha256") != request["corpus_manifest_sha256"] or request_record.get("lock_identity") != request["lock_identity"]:
        raise ValueError("Stage 1 campaign/API provenance binding")
    if ledger_record.get("stage") != "screening" or ledger_record.get("authorization") != "none":
        raise ValueError("Stage 1 campaign ledger")
    result = _validate_stage1_result(result_record, request_record, expected_shape=expected_shape)
    ledger = {
        "schema_version": 1, "mode": "screening", "status": "success", "authorization": "none",
        "repository": request["repository"], "branch": request["branch"], "workflow_path": request["workflow_path"], "head_sha": request["head_sha"],
        "run": {name: request[name] for name in ("run_id", "run_attempt", "job_id", "job_key", "job_allocation_nonce")},
        "artifact": request["artifact"], "runtime": request["runtime"], "runner": request["runner"],
        "model_manifest_sha256": request["model_manifest_sha256"], "corpus_manifest_sha256": request["corpus_manifest_sha256"], "lock_identity": request["lock_identity"],
        "stress_identity": result["stress_identity"],
        "builds": result["builds"],
        "d04_statistics": result["d04_statistics"], "nominated_m": result["nominated_m"],
        "source_digests": {"post_download_request": request["record_self_sha256"], "campaign_result": request["campaign_result_sha256"], "campaign_request": request["campaign_request_sha256"], "campaign_ledger": request["campaign_ledger_sha256"]},
    }
    ledger["record_self_sha256"] = _canonical_sha256(ledger)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(ledger, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return ledger


def validate_feature_worktree_preflight(record: dict) -> dict:
    """Reject an operator ledger that does not seal a non-default feature checkout."""
    if not isinstance(record, dict):
        raise ValueError("feature worktree preflight must be an object")
    branch_state = record.get("branch_head_status")
    if not isinstance(branch_state, dict):
        raise ValueError("missing branch/head/status evidence")
    branch = branch_state.get("branch")
    head = branch_state.get("head_sha")
    root = branch_state.get("worktree_root")
    status = branch_state.get("status")
    if not isinstance(branch, str) or branch in {"master", "main"}:
        raise ValueError("unsafe feature branch evidence")
    if not isinstance(head, str) or len(head) != 40:
        raise ValueError("invalid feature head evidence")
    if not isinstance(root, str) or not root:
        raise ValueError("missing worktree root evidence")
    if not isinstance(status, list) or not all(isinstance(item, str) for item in status):
        raise ValueError("invalid worktree status evidence")
    return branch_state


def validate_phase07_evidence_packet(packet: dict) -> dict:
    """Validate remote artifact identity, retention, digest, and retry lineage."""
    if not isinstance(packet, dict) or set(packet) != PHASE07_PACKET_FIELDS:
        raise ValueError("incomplete or unknown Phase 07 artifact fields")
    _reject_secrets(packet)
    for key in ("run_id", "run_attempt", "job_id", "artifact_id", "retention_days_requested", "retention_days_accepted"):
        if not isinstance(packet[key], int) or packet[key] <= 0:
            raise ValueError(f"invalid {key}")
    if packet["retention_days_requested"] != 90 or packet["retention_days_accepted"] != 90:
        raise ValueError("requested and accepted retention must be exactly 90 days")
    if not all(isinstance(packet[key], str) and _HEX64.fullmatch(packet[key]) for key in ("archive_sha256", "content_sha256", "record_self_sha256", "build_id")):
        raise ValueError("invalid artifact digest")
    if not isinstance(packet["head_sha"], str) or len(packet["head_sha"]) != 40:
        raise ValueError("invalid artifact head")
    if not isinstance(packet["job_allocation_nonce"], str) or len(packet["job_allocation_nonce"]) < 16:
        raise ValueError("invalid job allocation nonce")
    runner = packet["runner"]
    if not isinstance(runner, dict) or not {"os", "image", "architecture"} <= set(runner):
        raise ValueError("incomplete runner variance metadata")
    retry = packet["retry_lineage"]
    if not isinstance(retry, dict) or set(retry) != {"failure_class", "original_run_id", "replacement_run_id"}:
        raise ValueError("invalid retry lineage")
    if retry["replacement_run_id"] is not None and retry["failure_class"] not in PHASE07_INFRA_FAILURES:
        raise ValueError("non-infrastructure failures cannot be retried")
    expected = dict(packet)
    expected.pop("record_self_sha256")
    if packet["record_self_sha256"] != _canonical_sha256(expected):
        raise ValueError("artifact self-digest mismatch")
    return packet


def validate_phase07_evidence_set(packets: list[dict], *, expected_repository: str, expected_head: str) -> list[dict]:
    """Cross-run reconciliation rejects stale identity, allocation reuse, and fresh-build reuse."""
    allocations, builds = set(), set()
    for packet in packets:
        validate_phase07_evidence_packet(packet)
        if packet["repository"] != expected_repository or packet["head_sha"] != expected_head:
            raise ValueError("repository/head identity")
        allocation = tuple(packet[key] for key in ("run_id", "run_attempt", "job_id", "job_allocation_nonce"))
        if allocation in allocations or packet["build_id"] in builds:
            raise ValueError("reused allocation or build identity")
        allocations.add(allocation); builds.add(packet["build_id"])
    return packets


def reconcile(
    *, scale_evidence: Path, model_records: Path, conclusions: dict[str, str],
    expected_head: str, expected_pr_head: str,
) -> dict:
    """Validate one same-head numeric-success evidence set before reporting it."""
    for name in ("test-and-eval", "scale", "model-backed"):
        if conclusions.get(name) != "success":
            raise ValueError(f"required job {name} != \"success\"")
    if not isinstance(expected_head, str) or len(expected_head) != 40:
        raise ValueError("expected head")
    if not isinstance(expected_pr_head, str) or len(expected_pr_head) != 40:
        raise ValueError("expected PR head")

    scale = validate_evidence(_read_json(scale_evidence))
    source = scale["source"]
    if source["head_sha"] != expected_head:
        raise ValueError("scale artifact head")
    model = validate_candidate_decision_records(_read_json(model_records), scale)
    if model.get("head_sha") != expected_head:
        raise ValueError("model artifact head")
    if model.get("actions_merge_checkout_sha") != expected_head:
        raise ValueError("model artifact Actions merge checkout")
    if model.get("pr_head_sha") != expected_pr_head:
        raise ValueError("model artifact PR head")

    return {
        "schema_version": 1,
        "all_required_jobs_numeric_success": True,
        "head_sha": expected_head,
        "actions_merge_checkout_sha": expected_head,
        "pr_head_sha": expected_pr_head,
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
    parser.add_argument("--scale-evidence", type=Path)
    parser.add_argument("--model-records", type=Path)
    parser.add_argument("--test-and-eval")
    parser.add_argument("--scale")
    parser.add_argument("--model-backed")
    parser.add_argument("--expected-head")
    parser.add_argument("--expected-pr-head")
    parser.add_argument("--stage1-request", type=Path)
    parser.add_argument("--artifact-dir", type=Path)
    parser.add_argument("--mode")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        if args.stage1_request is not None or args.artifact_dir is not None or args.mode is not None:
            if args.stage1_request is None or args.artifact_dir is None or args.output is None:
                raise ValueError("Stage 1 reconciliation requires request, artifact directory, and output")
            reconcile_stage1_screening(
                stage1_request=args.stage1_request, artifact_dir=args.artifact_dir,
                output=args.output, mode=args.mode or "",
            )
            print("[PASS] Stage 1 ANN reconciliation", file=sys.stderr)
            return 0
        if None in (args.scale_evidence, args.model_records, args.test_and_eval, args.scale, args.model_backed, args.expected_head, args.expected_pr_head):
            raise ValueError("PR reconciliation requires all legacy evidence inputs")
        report = reconcile(
            scale_evidence=args.scale_evidence,
            model_records=args.model_records,
            conclusions={
                "test-and-eval": args.test_and_eval,
                "scale": args.scale,
                "model-backed": args.model_backed,
            },
            expected_head=args.expected_head,
            expected_pr_head=args.expected_pr_head,
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
