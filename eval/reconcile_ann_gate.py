"""Fail-closed PR reconciliation for held-out ANN decision artifacts."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import statistics
import sys
import re
import stat
import zipfile
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    # Direct-file entry points start with ``eval/`` on sys.path, whereas
    # run_eval imports the repository ``eval`` package.
    sys.path.insert(0, str(REPOSITORY_ROOT))

from eval import benchmark_ann_build as benchmark
from eval.benchmark_ann_build import validate_evidence
from eval.run_eval import validate_candidate_decision_records
from eval.phase07_ann_campaign import (
    Phase07AnnCampaignRunner,
    canonical_digest as campaign_digest,
    select_stage1_nominees,
    validate_confirmation_execution,
)
from eval.ann_frontier_statistics import (
    holm_adjust,
    paired_basic_effect,
    paired_permutation_p,
    select_stage2,
)


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
    "campaign_request_sha256", "campaign_ledger_sha256", "run_created_at",
    "api_provenance", "workflow_retention_days", "record_self_sha256",
})
_STAGE1_API_PROVENANCE_FIELDS = frozenset({"workflow_run", "job", "artifact"})
_STAGE1_API_WORKFLOW_RUN_FIELDS = frozenset({
    "run_id", "run_attempt", "head_branch", "head_sha", "event", "status",
    "conclusion", "created_at",
})
_STAGE1_API_JOB_FIELDS = frozenset({
    "job_id", "run_id", "name", "status", "conclusion", "runner_name",
    "runner_group_name", "labels",
})
_STAGE1_API_ARTIFACT_FIELDS = frozenset({
    "artifact_id", "job_id", "job_key", "run_id", "name", "created_at",
    "expires_at", "expired",
})
# This is deliberately scoped to the derived, final reconciliation ledger.
# Raw hybrid artifacts and their provenance documents retain schema v1.
HYBRID_POSTDOWNLOAD_LEDGER_SCHEMA_VERSION = 2


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


def canonical_digest(payload: dict) -> str:
    """Public canonical digest for newly sealed confirmation packets."""
    value = dict(payload)
    value.pop("record_self_sha256", None)
    return _canonical_sha256(value)


def validate_confirmation_packet(packet: dict, workflow_inputs: dict) -> dict:
    """Validate one non-pooled physical confirmation packet before reconciliation."""
    required = {
        "schema_version", "campaign_stage", "workflow_inputs_sha256", "slot", "run_id", "run_attempt",
        "job_id", "job_key", "job_allocation_nonce", "status", "failure_class", "replacement_for_run_id",
        "builds", "d25", "raw_tree_sha256", "retention_days", "record_self_sha256",
    }
    extended = {"locked_execution", "measurements"}
    if not isinstance(packet, dict) or set(packet) not in (required, required | extended) \
            or packet.get("record_self_sha256") != canonical_digest(packet):
        raise ValueError("sealed confirmation packet")
    if packet["campaign_stage"] != "confirmation" or packet["workflow_inputs_sha256"] != workflow_inputs.get("record_self_sha256") or packet["slot"] != workflow_inputs.get("slot"):
        raise ValueError("confirmation input/request binding")
    if not all(isinstance(packet[key], int) and packet[key] > 0 for key in ("run_id", "run_attempt", "job_id")) \
            or packet["job_key"] != "phase07-confirmation" \
            or not isinstance(packet["job_allocation_nonce"], str) \
            or re.fullmatch(r"[0-9a-f]{32}", packet["job_allocation_nonce"]) is None:
        raise ValueError("confirmation allocation identity")
    if packet["retention_days"] != 90 or not isinstance(packet["raw_tree_sha256"], str) or not _HEX64.fullmatch(packet["raw_tree_sha256"]):
        raise ValueError("confirmation artifact retention/digest")
    failure = packet["failure_class"]
    if packet["status"] == "numeric-success":
        if failure is not None:
            raise ValueError("numeric success cannot carry a failure")
    elif failure not in PHASE07_INFRA_FAILURES | {"numeric", "recall", "hybrid", "watchdog", "malformed", "reconciliation"}:
        raise ValueError("typed confirmation failure class")
    if packet["status"] != "numeric-success" and packet["replacement_for_run_id"] is not None:
        raise ValueError("rejected origin cannot replace another run")
    builds = packet["builds"]
    if not isinstance(builds, list) or len(builds) != 3:
        raise ValueError("confirmation requires exactly three fresh builds")
    by_m = {}
    build_ids = set()
    for build in builds:
        if not isinstance(build, dict) or set(build) != {"build_id", "m", "ef_construction", "query_ef"} or build.get("m") not in {16, 20, 32} or build.get("ef_construction") != 300 or not isinstance(build.get("build_id"), str) or not _HEX64.fullmatch(build["build_id"]) or build["build_id"] in build_ids:
            raise ValueError("fresh HNSW-SQ build identity")
        if build["m"] in by_m or not isinstance(build["query_ef"], list):
            raise ValueError("duplicate build m")
        by_m[build["m"]] = build; build_ids.add(build["build_id"])
    if set(by_m) != {16, 20, 32} or by_m[16]["query_ef"] != [100] \
            or any(by_m[m]["query_ef"] != [300] for m in (20, 32)):
        raise ValueError("D-25 fixed build/query allocation")
    family = packet["d25"]
    family_fields = {
        "family_name", "family_size", "baseline_build_id", "candidate_build_ids",
        "raw_p_values", "holm_adjusted_p_values", "basic_ci_95", "comparisons",
    }
    if not isinstance(family, dict) or set(family) != family_fields:
        raise ValueError("missing numeric confirmation family members")
    if family.get("family_name") != "d25_candidate_vs_production_baseline" \
            or family.get("family_size") != 4:
        raise ValueError("D-25 family/build identity")
    if family.get("baseline_build_id") != by_m[16]["build_id"]:
        raise ValueError("D-25 family must reference this packet m=16 baseline")
    if family.get("candidate_build_ids") != {
        "20": by_m[20]["build_id"], "32": by_m[32]["build_id"],
    }:
        raise ValueError("D-25 candidate build identity")
    for name in ("raw_p_values", "holm_adjusted_p_values", "basic_ci_95", "comparisons"):
        if not isinstance(family.get(name), list) or len(family[name]) != 4:
            raise ValueError("missing numeric confirmation family members")
    for value in [
        *family["raw_p_values"], *family["holm_adjusted_p_values"],
        *(item for interval in family["basic_ci_95"] for item in interval),
    ]:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("non-finite confirmation statistic")
    if family["holm_adjusted_p_values"] != holm_adjust(family["raw_p_values"]):
        raise ValueError("declared confirmation Holm mismatch")
    canonical_order = [
        (m, metric) for m in (20, 32) for metric in ("recall_at_10", "recall_at_20")
    ]
    seen = []
    for index, comparison in enumerate(family["comparisons"]):
        rows = comparison.get("paired_rows") if isinstance(comparison, dict) else None
        declared = comparison.get("comparison") if isinstance(comparison, dict) else None
        if not isinstance(rows, list) or not rows:
            raise ValueError("missing paired confirmation samples")
        if not isinstance(declared, dict) or set(declared) != {
            "family", "metric", "baseline_m", "baseline_ef", "candidate_m", "candidate_ef",
            "baseline_build_id", "candidate_build_id",
        }:
            raise ValueError("missing confirmation comparison binding")
        key = (declared["candidate_m"], declared["metric"])
        if key != canonical_order[index] or declared["family"] != family["family_name"] \
                or declared["baseline_m"] != 16 or declared["baseline_ef"] != 100 \
                or declared["candidate_ef"] != 300 \
                or declared["baseline_build_id"] != by_m[16]["build_id"] \
                or declared["candidate_build_id"] != by_m[declared["candidate_m"]]["build_id"]:
            raise ValueError("noncanonical D-25 paired member")
        seen.append(key)
        for row in rows:
            if not isinstance(row, list) or len(row) != 2 or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) for value in row
            ):
                raise ValueError("invalid paired confirmation samples")
        if comparison.get("raw_permutation_p") != family["raw_p_values"][index] \
                or comparison.get("holm_adjusted_p") != family["holm_adjusted_p_values"][index] \
                or comparison.get("basic_ci_95") != family["basic_ci_95"][index]:
            raise ValueError("declared confirmation statistic mismatch")
        expected_effect = paired_basic_effect(rows, comparison=declared)
        expected_p = paired_permutation_p(rows, comparison=declared)
        if comparison.get("mean_effect") != expected_effect["mean_effect"] \
                or comparison.get("basic_ci_95") != expected_effect["basic_ci_95"] \
                or comparison.get("raw_permutation_p") != expected_p:
            raise ValueError("recomputed confirmation statistic mismatch")
    if seen != canonical_order:
        raise ValueError("duplicate canonical confirmation member")
    if extended <= set(packet):
        validate_confirmation_execution(packet["locked_execution"])
        measurements = packet["measurements"]
        raw_builds = measurements.get("builds") if isinstance(measurements, dict) else None
        expected_builds = [
            {"build_id": build.get("build_id"), "m": build.get("build", {}).get("m"),
             "ef_construction": build.get("build", {}).get("ef_construction"),
             "query_ef": [group.get("query_ef") for group in build.get("queries", [])]}
            for build in raw_builds
        ] if isinstance(raw_builds, list) else None
        expected_statistics = {
            "schema_version": 1, "family_name": family["family_name"],
            "family_size": 4, "comparisons": family["comparisons"], "authorization": "none",
        }
        expected_statistics["record_self_sha256"] = campaign_digest(expected_statistics)
        if not isinstance(measurements, dict) or set(measurements) != {"builds", "paired_statistics"} \
                or expected_builds != packet["builds"] \
                or measurements["paired_statistics"] != expected_statistics:
            raise ValueError("confirmation raw measurement binding")
        raw_by_m = {build.get("build", {}).get("m"): build for build in raw_builds}
        if set(raw_by_m) != {16, 20, 32}:
            raise ValueError("confirmation raw build cardinality")
        for m, raw in raw_by_m.items():
            card, groups = raw.get("build"), raw.get("queries")
            expected_ef = 100 if m == 16 else 300
            if not isinstance(card, dict) or card.get("candidate") != "ivf-hnsw-sq" \
                    or card.get("ef_construction") != 300 or card.get("unindexed_dense_rows") != 0 \
                    or card.get("reopen_verified") is not True or not isinstance(groups, list) \
                    or len(groups) != 1 or groups[0].get("query_ef") != expected_ef:
                raise ValueError("D-25 raw build/query evidence")
            watchdog = card.get("watchdog")
            if not isinstance(watchdog, dict) or watchdog.get("owner") != "parent" \
                    or watchdog.get("child_exitcode") != 0 \
                    or not isinstance(watchdog.get("cap_seconds"), (int, float)) \
                    or isinstance(watchdog.get("cap_seconds"), bool) \
                    or not 0 < watchdog["cap_seconds"] <= 180:
                raise ValueError("D-25 per-build watchdog evidence")
            for name in ("index_build_ms", "index_bytes"):
                value = card.get(name)
                if isinstance(value, bool) or not isinstance(value, (int, float)) \
                        or not math.isfinite(value) or value < 0:
                    raise ValueError("finite D-25 build cost evidence")
            group = groups[0]
            p95, samples = group.get("latency_p95_ms"), group.get("queries")
            if isinstance(p95, bool) or not isinstance(p95, (int, float)) \
                    or not math.isfinite(p95) or p95 < 0 or not isinstance(samples, list) or not samples:
                raise ValueError("finite D-25 query cost/samples")
            if [sample.get("query_index") for sample in samples] != list(range(len(samples))):
                raise ValueError("ordered D-25 query samples")
        baseline_samples = raw_by_m[16]["queries"][0]["queries"]
        for comparison in family["comparisons"]:
            declared = comparison["comparison"]
            candidate_samples = raw_by_m[declared["candidate_m"]]["queries"][0]["queries"]
            metric = declared["metric"]
            raw_pairs = [
                [left.get(metric), right.get(metric)]
                for left, right in zip(baseline_samples, candidate_samples, strict=True)
            ]
            if comparison["paired_rows"] != raw_pairs:
                raise ValueError("paired statistics/raw query mismatch")
    return packet


def reconcile_confirmation(plan: dict, packets: list[dict]) -> dict:
    """Seal exactly three ordinal numeric-success runs for D-25."""
    from eval.phase07_operator_gate import validate_confirmation_plan
    validate_confirmation_plan(plan)
    if not isinstance(packets, list) or not packets:
        raise ValueError("confirmation packets required")
    inputs = {record["record_self_sha256"]: record for record in plan["workflow_inputs"]}
    physical = []
    per_slot: dict[str, list[dict]] = {key: [] for key in inputs}
    seen_allocations, seen_run_ids, seen_job_ids, seen_nonces = set(), set(), set(), set()
    for packet in packets:
        key = packet.get("workflow_inputs_sha256") if isinstance(packet, dict) else None
        if key not in inputs:
            raise ValueError("hand-authored or unknown confirmation input")
        validate_confirmation_packet(packet, inputs[key])
        allocation = (packet["run_id"], packet["run_attempt"], packet["job_id"], packet["job_allocation_nonce"])
        if allocation in seen_allocations or packet["run_id"] in seen_run_ids \
                or packet["job_id"] in seen_job_ids \
                or packet["job_allocation_nonce"] in seen_nonces:
            raise ValueError("replayed confirmation physical run")
        seen_allocations.add(allocation); seen_run_ids.add(packet["run_id"])
        seen_job_ids.add(packet["job_id"]); seen_nonces.add(packet["job_allocation_nonce"])
        per_slot[key].append(packet)
    eligible = []
    for key, slot_packets in per_slot.items():
        success = [packet for packet in slot_packets if packet["status"] == "numeric-success"]
        rejected = [packet for packet in slot_packets if packet["status"] != "numeric-success"]
        if rejected or len(success) != 1 or success[0]["replacement_for_run_id"] is not None:
            raise ValueError("D-25 confirmation requires one unreplaced numeric result per ordinal")
        physical.append({**success[0], "eligible": True})
        eligible.append(success[0])
    if len(eligible) != 3 or len({record["workflow_inputs_sha256"] for record in eligible}) != 3:
        raise ValueError("exact three eligible confirmation ordinal records")
    eligible.sort(key=lambda record: record["slot"]["ordinal"])
    build_ids = [build["build_id"] for record in eligible for build in record["builds"]]
    if len(build_ids) != 9 or len(set(build_ids)) != 9:
        raise ValueError("three disjoint fresh build-ID sets required")
    ordinal_families = [
        {"ordinal": record["slot"]["ordinal"], **record["d25"]}
        for record in eligible
    ]
    if [family["ordinal"] for family in ordinal_families] != [1, 2, 3]:
        raise ValueError("exact ordinal family order")
    ledger = {"schema_version": 1, "campaign_stage": "confirmation", "confirmation_plan_sha256": plan["record_self_sha256"], "eligible_evidence_runs": eligible, "all_physical_workflow_runs": physical, "paired_ordinal_families": ordinal_families}
    ledger["record_self_sha256"] = canonical_digest(ledger)
    return ledger


def reconcile_confirmation_request(request: dict, source: dict) -> dict:
    """Consume downloaded packet wrappers and recompute the three-ordinal ledger."""
    from eval.phase07_operator_gate import canonical_digest as operator_digest, validate_confirmation_dispatch_bundle
    if not isinstance(request, dict) or request.get("record_self_sha256") != operator_digest(request):
        raise ValueError("sealed confirmation request")
    wrappers = source.get("packets") if isinstance(source, dict) else None
    if not isinstance(wrappers, list) or not wrappers:
        raise ValueError("downloaded confirmation packets required")
    inputs, packets = [], []
    for wrapper in wrappers:
        if not isinstance(wrapper, dict) or set(wrapper) != {"dispatch_bundle", "packet"}:
            raise ValueError("downloaded packet wrapper schema")
        bundle = wrapper["dispatch_bundle"]
        record = validate_confirmation_dispatch_bundle(bundle, expected_head=request.get("post_task0_head", ""))
        if bundle["confirmation_request"] != request:
            raise ValueError("cross-request confirmation replay")
        inputs.append(record); packets.append(wrapper["packet"])
    if len({record["record_self_sha256"] for record in inputs}) != 3:
        raise ValueError("exact three unique downloaded confirmation inputs")
    unique_inputs = {record["record_self_sha256"]: record for record in inputs}
    plan = {"schema_version": 1, "confirmation_request": request, "workflow_inputs": sorted(unique_inputs.values(), key=lambda row: row["slot"]["ordinal"]),
            "artifact_reported_nominated_m": request["artifact_reported_nominated_m"], "authoritative_nominated_m": request["authoritative_nominated_m"]}
    plan["record_self_sha256"] = operator_digest(plan)
    return reconcile_confirmation(plan, packets)


_CONFIRMATION_RAW_FILES = frozenset({
    "confirmation-request.json", "confirmation-ledger.json", "confirmation-result.json",
    "dispatch-bundle.json", "allocation.json",
})
_CONFIRMATION_ARTIFACT_FILES = _CONFIRMATION_RAW_FILES | {"confirmation-packet.json"}
_CONFIRMATION_PROVENANCE_FIELDS = frozenset({
    "run_id", "run_attempt", "job_id", "job_key", "job_name", "artifact_id", "artifact_name", "status", "conclusion",
    "head_branch", "head_sha", "event",
    "runner", "run_created_at", "artifact_expires_at", "api_archive_sha256", "local_archive_sha256",
    "archive", "extracted_dir",
})


def _confirmation_tree_sha256(root: Path) -> str:
    """Compatibility facade for test tooling; production validation is shared."""
    from eval.phase07_ann_campaign import confirmation_raw_tree_sha256
    return confirmation_raw_tree_sha256(root, require_wrapper=True)


def _confirmation_timestamp(value: Any, label: str) -> dt.datetime:
    parsed = _stage1_timestamp(value, label=f"confirmation {label}")
    return parsed


def _validate_confirmation_provenance(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _CONFIRMATION_PROVENANCE_FIELDS:
        raise ValueError("strict confirmation API provenance schema")
    if not all(isinstance(value[name], int) and value[name] > 0 for name in ("run_id", "run_attempt", "job_id", "artifact_id")) \
            or not isinstance(value["artifact_name"], str) or not value["artifact_name"] \
            or value["status"] != "completed" or value["conclusion"] != "success":
        raise ValueError("confirmation API run/job/artifact status")
    if value["job_key"] != "phase07-confirmation" \
            or value["job_name"] != "Phase 07 independent confirmation campaign" \
            or not isinstance(value["head_branch"], str) or value["head_branch"] in {"", "main", "master"} \
            or not isinstance(value["head_sha"], str) or not re.fullmatch(r"[0-9a-f]{40}", value["head_sha"]) \
            or value["event"] != "workflow_dispatch":
        raise ValueError("confirmation API head/event/job binding")
    if value["artifact_name"] != f"phase07-confirmation-{value['run_id']}-{value['run_attempt']}":
        raise ValueError("confirmation artifact name/run-attempt binding")
    runner = value["runner"]
    if not isinstance(runner, dict) or set(runner) != {"name", "group", "labels", "os", "image", "architecture"} \
            or not isinstance(runner["name"], str) or not runner["name"] \
            or not isinstance(runner["group"], str) or not runner["group"] \
            or not isinstance(runner["labels"], list) or not runner["labels"] \
            or runner["os"] != "Linux" or runner["architecture"] != "X64" \
            or not isinstance(runner["image"], str) or not runner["image"]:
        raise ValueError("confirmation runner identity")
    created, expires = _confirmation_timestamp(value["run_created_at"], "created"), _confirmation_timestamp(value["artifact_expires_at"], "expiry")
    retention = expires - created
    if not dt.timedelta(days=89, hours=23, minutes=59, seconds=30) <= retention <= dt.timedelta(days=90, seconds=30):
        raise ValueError("confirmation artifact retention interval")
    for name in ("api_archive_sha256", "local_archive_sha256"):
        if not isinstance(value[name], str) or not _HEX64.fullmatch(value[name]):
            raise ValueError("confirmation archive digest schema")
    archive, extracted = Path(value["archive"]), Path(value["extracted_dir"])
    if archive.is_symlink() or not archive.is_file() or extracted.is_symlink() or not extracted.is_dir():
        raise ValueError("confirmation downloaded archive/extraction unavailable")
    local = hashlib.sha256(archive.read_bytes()).hexdigest()
    if value["api_archive_sha256"] != local or value["local_archive_sha256"] != local:
        raise ValueError("confirmation API/local archive digest mismatch")
    try:
        with zipfile.ZipFile(archive) as compressed:
            members = compressed.infolist()
            names = [member.filename for member in members]
            if len(names) != len(_CONFIRMATION_ARTIFACT_FILES) \
                    or len(set(names)) != len(names) \
                    or set(names) != _CONFIRMATION_ARTIFACT_FILES \
                    or any(member.is_dir() or stat.S_ISLNK(member.external_attr >> 16) for member in members):
                raise ValueError("strict confirmation archive allowlist")
            for name in _CONFIRMATION_ARTIFACT_FILES:
                extracted_member = extracted / name
                if extracted_member.is_symlink() or not extracted_member.is_file() \
                        or compressed.read(name) != extracted_member.read_bytes():
                    raise ValueError("confirmation archive/extracted content mismatch")
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise ValueError("invalid confirmation downloaded archive") from exc
    return value


def _read_confirmation_artifact(root: Path, provenance: dict[str, Any], request: dict[str, Any]) -> tuple[dict, dict, dict]:
    """Use the shared artifact validator, then attach immutable API provenance."""
    from eval.phase07_ann_campaign import validate_confirmation_artifact_tree

    validated = validate_confirmation_artifact_tree(root, expected_head=request.get("post_task0_head", ""))
    generated, packet = validated["workflow_input"], validated["packet"]
    identity = validated["allocation"]["allocation"]
    if any(identity[name] != provenance[name] for name in ("run_id", "run_attempt", "job_id")) \
            or packet["run_id"] != provenance["run_id"] or packet["run_attempt"] != provenance["run_attempt"] \
            or packet["job_id"] != provenance["job_id"] or packet["job_allocation_nonce"] != identity["job_allocation_nonce"]:
        raise ValueError("confirmation packet/API allocation identity mismatch")
    host = packet["locked_execution"]["host"]
    if {name: provenance["runner"][name] for name in ("os", "image", "architecture")} != {
        name: host[name] for name in ("os", "image", "architecture")
    }:
        raise ValueError("confirmation runner/locked-host provenance mismatch")
    validated_provenance = {
        **provenance,
        "content_sha256": validated["content_tree_sha256"],
        "raw_tree_sha256": validated["raw_tree_sha256"],
        "packet_self_sha256": packet["record_self_sha256"],
        "wrapper_self_sha256": validated["wrapper"]["record_self_sha256"],
        "raw_result_sha256": validated["raw_file_sha256"]["confirmation-result.json"],
    }
    return generated, packet, {
        "validated_provenance": validated_provenance,
        "validated_measurements": validated["result"],
    }


def reconcile_confirmation_postdownload(request: dict, manifest: dict) -> dict:
    """Reconcile exactly three successful downloaded ordinal artifacts."""
    from eval.phase07_operator_gate import canonical_digest as operator_digest

    if not isinstance(request, dict) or request.get("record_self_sha256") != operator_digest(request):
        raise ValueError("sealed confirmation request")
    if not isinstance(manifest, dict) or set(manifest) != {"schema_version", "evidence", "record_self_sha256"} \
            or manifest.get("schema_version") != 1 or manifest.get("record_self_sha256") != canonical_digest(manifest) \
            or not isinstance(manifest.get("evidence"), list) or len(manifest["evidence"]) != 3:
        raise ValueError("strict confirmation provenance manifest cardinality")
    inputs, packets, provenance_by_allocation, seen_archives = [], [], {}, set()
    artifact_ids, job_ids, run_identities = set(), set(), set()
    for row in manifest["evidence"]:
        if not isinstance(row, dict) or set(row) != {"artifact_dir", "provenance"}:
            raise ValueError("strict confirmation evidence record")
        provenance_document = _read_json(Path(row["provenance"]))
        if set(provenance_document) != {"schema_version", "evidence", "record_self_sha256"} \
                or provenance_document.get("schema_version") != 1 \
                or provenance_document.get("record_self_sha256") != canonical_digest(provenance_document) \
                or not isinstance(provenance_document.get("evidence"), list) or len(provenance_document["evidence"]) != 1:
            raise ValueError("strict per-artifact API provenance document")
        provenance = _validate_confirmation_provenance(provenance_document["evidence"][0])
        if provenance["head_sha"] != request.get("post_task0_head"):
            raise ValueError("confirmation API/request head mismatch")
        if provenance["archive"] in seen_archives:
            raise ValueError("replayed confirmation archive")
        seen_archives.add(provenance["archive"])
        if Path(row["artifact_dir"]).resolve() != Path(provenance["extracted_dir"]).resolve():
            raise ValueError("manifest/extracted artifact mismatch")
        artifact_id, job_id = provenance["artifact_id"], provenance["job_id"]
        run_identity = (provenance["run_id"], provenance["run_attempt"])
        if artifact_id in artifact_ids or job_id in job_ids or run_identity in run_identities:
            raise ValueError("duplicate confirmation artifact/job/run provenance")
        artifact_ids.add(artifact_id); job_ids.add(job_id); run_identities.add(run_identity)
        generated, packet, metadata = _read_confirmation_artifact(Path(row["artifact_dir"]), provenance, request)
        allocation = tuple(packet[name] for name in ("run_id", "run_attempt", "job_id", "job_allocation_nonce"))
        inputs.append(generated); packets.append(packet); provenance_by_allocation[allocation] = metadata
    if len({item["record_self_sha256"] for item in inputs}) != 3:
        raise ValueError("duplicate confirmation input provenance")
    plan = {"schema_version": 1, "confirmation_request": request,
            "workflow_inputs": sorted({item["record_self_sha256"]: item for item in inputs}.values(), key=lambda row: row["slot"]["ordinal"]),
            "artifact_reported_nominated_m": request["artifact_reported_nominated_m"],
            "authoritative_nominated_m": request["authoritative_nominated_m"]}
    plan["record_self_sha256"] = operator_digest(plan)
    ledger = reconcile_confirmation(plan, packets)
    for record in [*ledger["eligible_evidence_runs"], *ledger["all_physical_workflow_runs"]]:
        allocation = tuple(record[name] for name in ("run_id", "run_attempt", "job_id", "job_allocation_nonce"))
        metadata = provenance_by_allocation.get(allocation)
        if metadata is None:
            raise ValueError("missing validated confirmation provenance")
        record.update(metadata)
    ledger["record_self_sha256"] = canonical_digest(ledger)
    return ledger


# Hybrid evidence is deliberately reconciled separately from confirmation.  A
# complete 30k hybrid packet has a different authority chain (one fixed
# candidate per hosted run) and neither a generic packet nor a confirmation
# packet can be relabelled into this path.
_HYBRID_PROVENANCE_FIELDS = frozenset({
    "run_id", "run_attempt", "job_id", "job_key", "job_name", "artifact_id", "artifact_name",
    "status", "conclusion", "head_branch", "head_sha", "event", "runner", "run_created_at",
    "artifact_created_at", "artifact_expires_at", "api_archive_sha256", "local_archive_sha256",
    "role", "config", "bundle_sha256",
    "archive", "extracted_dir",
})
_HYBRID_FROZEN_PREPARE_FIELD = "frozen_prepare"
_HYBRID_MANIFEST_EVIDENCE_FIELDS = frozenset({
    "run_id", "run_attempt", "job_id", "artifact_id", "role", "config", "bundle_sha256",
    "archive", "extracted_dir", "provenance",
})
_HYBRID_ARTIFACT_FILES = frozenset({
    "hybrid-request.json", "hybrid-ledger.json", "hybrid-result.json", "dispatch-bundle.json",
    "locked-execution.json", "allocation.json", "hybrid-packet.json",
})


def _reject_hybrid_secrets(value: Any, location: str = "hybrid") -> None:
    """Reject secret material while admitting the sealed `authorization: none` marker."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "authorization" and item == "none":
                continue
            if any(marker in str(key).lower() for marker in _SECRET_MARKERS):
                raise ValueError(f"secret-like hybrid field: {location}.{key}")
            _reject_hybrid_secrets(item, f"{location}.{key}")
    elif isinstance(value, list):
        for item in value:
            _reject_hybrid_secrets(item, location)
    elif isinstance(value, str) and any(marker in value.lower() for marker in ("ghp_", "github_pat_", "bearer ")):
        raise ValueError("secret-like hybrid value")


def _validate_hybrid_provenance(value: Any) -> dict[str, Any]:
    """Validate one API-bound hybrid download before reading its JSON tree."""
    _reject_hybrid_secrets(value, "hybrid provenance")
    fields = set(value) if isinstance(value, dict) else set()
    if fields not in (_HYBRID_PROVENANCE_FIELDS, _HYBRID_PROVENANCE_FIELDS | {_HYBRID_FROZEN_PREPARE_FIELD}):
        raise ValueError("strict hybrid API provenance schema")
    if not all(isinstance(value[name], int) and not isinstance(value[name], bool) and value[name] > 0
               for name in ("run_id", "run_attempt", "job_id", "artifact_id")) \
            or value["run_attempt"] != 1 \
            or value["job_key"] != "phase07-hybrid" \
            or value["job_name"] != "Phase 07 independent hybrid campaign" \
            or value["status"] != "completed" or value["conclusion"] != "success":
        raise ValueError("hybrid API run/job/artifact status")
    from eval.phase07_operator_gate import HYBRID_ROLE_CONFIGS, FROZEN_HYBRID_ROLE_CONFIGS
    if (value.get("role"), value.get("config")) not in HYBRID_ROLE_CONFIGS + FROZEN_HYBRID_ROLE_CONFIGS \
            or not isinstance(value["bundle_sha256"], str) \
            or not _HEX64.fullmatch(value["bundle_sha256"]):
        raise ValueError("hybrid provenance role/config/bundle identity")
    if value.get("role") in {"m20", "m32"} or _HYBRID_FROZEN_PREPARE_FIELD in value:
        from eval.phase07_frozen_base import validate_frozen_prepare_identity_shape
        value[_HYBRID_FROZEN_PREPARE_FIELD] = validate_frozen_prepare_identity_shape(
            value.get(_HYBRID_FROZEN_PREPARE_FIELD),
            expected_repository="allenwoo713/obsidian_wiki_skill", expected_head=value.get("head_sha", ""),
        )
    if not isinstance(value["head_branch"], str) or value["head_branch"] in {"", "main", "master"} \
            or not isinstance(value["head_sha"], str) or not re.fullmatch(r"[0-9a-f]{40}", value["head_sha"]) \
            or value["event"] != "workflow_dispatch" \
            or value["artifact_name"] != f"phase07-hybrid-{value['run_id']}-{value['run_attempt']}":
        raise ValueError("hybrid API head/event/artifact binding")
    runner = value["runner"]
    if not isinstance(runner, dict) or set(runner) != {"name", "group", "labels", "os", "image", "architecture"} \
            or not isinstance(runner["name"], str) or not runner["name"].startswith("GitHub Actions ") \
            or runner["group"] != "GitHub Actions" \
            or not isinstance(runner["labels"], list) \
            or not all(isinstance(label, str) and label for label in runner["labels"]) \
            or "ubuntu-latest" not in runner["labels"] or "ARM64" in runner["labels"] \
            or runner["os"] != "Linux" or runner["architecture"] != "X64" \
            or not isinstance(runner["image"], str) \
            or re.fullmatch(r"ubuntu[^ ]* [^ ]+", runner["image"], flags=re.IGNORECASE) is None:
        raise ValueError("hybrid runner identity")
    run_created = _stage1_timestamp(value["run_created_at"], label="hybrid workflow-run creation")
    artifact_created = _stage1_timestamp(value["artifact_created_at"], label="hybrid artifact creation")
    expires = _stage1_timestamp(value["artifact_expires_at"], label="hybrid artifact expiry")
    retention = expires - artifact_created
    if not dt.timedelta(days=89, hours=23, minutes=59, seconds=30) <= retention <= dt.timedelta(days=90, seconds=30):
        raise ValueError("hybrid artifact retention interval")
    if run_created > artifact_created or artifact_created >= expires \
            or expires <= dt.datetime.now(dt.timezone.utc):
        raise ValueError("expired hybrid artifact evidence")
    for field in ("api_archive_sha256", "local_archive_sha256"):
        if not isinstance(value[field], str) or not _HEX64.fullmatch(value[field]):
            raise ValueError("hybrid archive digest schema")
    archive, extracted = Path(value["archive"]), Path(value["extracted_dir"])
    if archive.is_symlink() or not archive.is_file() or extracted.is_symlink() or not extracted.is_dir():
        raise ValueError("hybrid downloaded archive/extraction unavailable")
    local_digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    if value["api_archive_sha256"] != local_digest or value["local_archive_sha256"] != local_digest:
        raise ValueError("hybrid API/local archive digest mismatch")
    try:
        with zipfile.ZipFile(archive) as compressed:
            members = compressed.infolist()
            names = [member.filename for member in members]
            if len(names) != len(_HYBRID_ARTIFACT_FILES) or len(set(names)) != len(names) \
                    or set(names) != _HYBRID_ARTIFACT_FILES \
                    or any(member.is_dir() or stat.S_ISLNK(member.external_attr >> 16) for member in members):
                raise ValueError("strict hybrid archive allowlist")
            for name in _HYBRID_ARTIFACT_FILES:
                destination = extracted / name
                if destination.is_symlink() or not destination.is_file() \
                        or compressed.read(name) != destination.read_bytes():
                    raise ValueError("hybrid archive/extracted content mismatch")
    except (OSError, zipfile.BadZipFile, KeyError) as exc:
        raise ValueError("invalid hybrid downloaded archive") from exc
    return value


def _read_hybrid_artifact(root: Path, provenance: dict[str, Any], request: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct an exact packet and bind it to the GitHub API record."""
    from eval.phase07_ann_campaign import validate_hybrid_artifact_tree

    validated = validate_hybrid_artifact_tree(root)
    member, result = validated["workflow_input"], validated["result"]
    allocation = result.get("allocation", {}).get("allocation") if isinstance(result.get("allocation"), dict) else None
    if not isinstance(allocation, dict):
        raise ValueError("hybrid artifact allocation")
    if result["hybrid_request_sha256"] != request["record_self_sha256"] \
            or member["hybrid_request_sha256"] != request["record_self_sha256"]:
        raise ValueError("hybrid artifact/request binding")
    if member.get("role") != provenance["role"] or member.get("config") != provenance["config"] \
            or member.get("record_self_sha256") != provenance["bundle_sha256"]:
        raise ValueError("hybrid packet/API role/config bundle mismatch")
    if any(allocation[name] != provenance[name] for name in ("run_id", "run_attempt", "job_id")):
        raise ValueError("hybrid packet/API allocation identity mismatch")
    host = result["locked_execution"].get("host")
    if not isinstance(host, dict) or {name: provenance["runner"][name] for name in ("os", "image", "architecture")} != {
        name: host.get(name) for name in ("os", "image", "architecture")
    }:
        raise ValueError("hybrid runner/locked-host provenance mismatch")
    if result.get("head_sha") != provenance["head_sha"]:
        raise ValueError("hybrid packet/API head binding")
    frozen_prepare = validated.get("frozen_prepare")
    if frozen_prepare is not None:
        if provenance.get(_HYBRID_FROZEN_PREPARE_FIELD) != frozen_prepare:
            raise ValueError("hybrid API/artifact frozen prepare mismatch")
        if result.get("source_before_sha256") != frozen_prepare["base_tree_sha256"] \
                or result.get("source_after_sha256") != frozen_prepare["base_tree_sha256"]:
            raise ValueError("hybrid frozen source mutation evidence")
    return {
        "role": member["role"], "config": member["config"],
        "original_observations": result["original_observations"],
        "expanded_observations": result["expanded_observations"],
        "packet_identity": {
            "raw_tree_sha256": validated["raw_tree_sha256"],
            "packet_self_sha256": validated["wrapper"]["record_self_sha256"],
            "raw_result_sha256": hashlib.sha256((root / "hybrid-result.json").read_bytes()).hexdigest(),
            "raw_ledger_sha256": hashlib.sha256((root / "hybrid-ledger.json").read_bytes()).hexdigest(),
            "bundle_sha256": member["record_self_sha256"],
            "allocation_nonce": allocation["job_allocation_nonce"],
            "expanded_content_tree_sha256": result["expanded_content_tree_sha256"],
            "expanded_member_count": result["expanded_member_count"],
            "queries_sha256": result["queries_sha256"],
            "baselines_sha256": result["baselines_sha256"],
        },
        "provenance": dict(provenance), "frozen_prepare": frozen_prepare,
    }


def _pair_hybrid_role_observations(
    *, baseline: list[dict[str, Any]], candidate: list[dict[str, Any]], label: str,
) -> list[dict[str, Any]]:
    """Join two independently hosted role streams by exact ordinal/query digest."""
    if not isinstance(baseline, list) or not isinstance(candidate, list) \
            or len(baseline) != 105 or len(candidate) != 105:
        raise ValueError(f"complete hybrid {label} role evidence required")
    paired: list[dict[str, Any]] = []
    for ordinal, (baseline_row, candidate_row) in enumerate(zip(baseline, candidate, strict=True)):
        fields = {"ordinal", "query_sha256", "observation"}
        if not isinstance(baseline_row, dict) or not isinstance(candidate_row, dict) \
                or set(baseline_row) != fields or set(candidate_row) != fields \
                or baseline_row["ordinal"] != ordinal or candidate_row["ordinal"] != ordinal \
                or baseline_row["query_sha256"] != candidate_row["query_sha256"]:
            raise ValueError(f"hybrid {label} ordinal/query join mismatch")
        paired.append({
            "ordinal": ordinal, "query_sha256": baseline_row["query_sha256"],
            "baseline": baseline_row["observation"], "candidate": candidate_row["observation"],
        })
    return paired


def reconcile_hybrid_postdownload(request: dict, manifest: dict) -> dict:
    """Recompute gates from one baseline plus exact m20/m32 role packets.

    Pairing happens only here, after all three independent first-attempt hosted
    artifacts pass byte, API, runner, source, query and allocation validation.
    """
    from eval.phase07_operator_gate import (
        HYBRID_BASELINE, HYBRID_CANDIDATES, HYBRID_ROLE_CONFIGS, FROZEN_HYBRID_ROLE_CONFIGS,
        _validate_hybrid_request, canonical_digest as operator_digest,
        recompute_hybrid_gate_verdicts,
    )
    from eval.run_eval import aggregate_hybrid_serialized_metrics, aggregate_hybrid_serialized_scale_diagnostics

    _reject_hybrid_secrets(request, "hybrid request")
    if not isinstance(request, dict) or request.get("record_self_sha256") != operator_digest(request):
        raise ValueError("sealed hybrid request")
    head = request.get("hybrid_implementation_head")
    if not isinstance(head, str):
        raise ValueError("hybrid implementation head")
    _validate_hybrid_request(request, expected_head=head)
    _reject_hybrid_secrets(manifest, "hybrid evidence manifest")
    if not isinstance(manifest, dict) or set(manifest) != {"schema_version", "evidence", "record_self_sha256"} \
            or manifest.get("schema_version") != 1 or manifest.get("record_self_sha256") != canonical_digest(manifest) \
            or not isinstance(manifest.get("evidence"), list) or len(manifest["evidence"]) != 3:
        raise ValueError("strict exactly-three hybrid provenance manifest")

    role_records: list[dict[str, Any]] = []
    seen_runs, seen_jobs, seen_artifacts, seen_archives, seen_bundles, seen_roles = (set() for _ in range(6))
    for row in manifest["evidence"]:
        row_fields = set(row) if isinstance(row, dict) else set()
        if not isinstance(row, dict) or row_fields not in (_HYBRID_MANIFEST_EVIDENCE_FIELDS,
                                                           _HYBRID_MANIFEST_EVIDENCE_FIELDS | {_HYBRID_FROZEN_PREPARE_FIELD}) \
                or any(not isinstance(row[name], int) or isinstance(row[name], bool) or row[name] <= 0
                       for name in ("run_id", "run_attempt", "job_id", "artifact_id")) \
                or row["run_attempt"] != 1 \
                or (row.get("role"), row.get("config")) not in HYBRID_ROLE_CONFIGS + FROZEN_HYBRID_ROLE_CONFIGS \
                or not isinstance(row["bundle_sha256"], str) or not _HEX64.fullmatch(row["bundle_sha256"]) \
                or not all(isinstance(row[name], str) and row[name]
                           for name in ("archive", "extracted_dir", "provenance")):
            raise ValueError("strict hybrid evidence record")
        provenance_path = Path(row["provenance"])
        if provenance_path.is_symlink() or not provenance_path.is_file():
            raise ValueError("hybrid per-artifact provenance document unavailable")
        provenance_document = _read_json(provenance_path)
        _reject_hybrid_secrets(provenance_document, "hybrid per-artifact provenance")
        if set(provenance_document) != {"schema_version", "evidence", "record_self_sha256"} \
                or provenance_document.get("schema_version") != 1 \
                or provenance_document.get("record_self_sha256") != canonical_digest(provenance_document) \
                or not isinstance(provenance_document.get("evidence"), list) \
                or len(provenance_document["evidence"]) != 1:
            raise ValueError("strict per-artifact hybrid API provenance document")
        provenance = _validate_hybrid_provenance(provenance_document["evidence"][0])
        if provenance["head_sha"] != head:
            raise ValueError("hybrid API/request head mismatch")
        cross_bound = ("run_id", "run_attempt", "job_id", "artifact_id", "role", "config", "bundle_sha256")
        if any(row[name] != provenance[name] for name in cross_bound) \
                or Path(row["archive"]).resolve() != Path(provenance["archive"]).resolve() \
                or Path(row["extracted_dir"]).resolve() != Path(provenance["extracted_dir"]).resolve():
            raise ValueError("hybrid manifest/API provenance identity mismatch")
        if (_HYBRID_FROZEN_PREPARE_FIELD in row or _HYBRID_FROZEN_PREPARE_FIELD in provenance) and (
                row.get(_HYBRID_FROZEN_PREPARE_FIELD) != provenance.get(_HYBRID_FROZEN_PREPARE_FIELD)):
            raise ValueError("hybrid collection frozen prepare identity mismatch")
        artifact_dir = Path(row["extracted_dir"])
        run_identity = (provenance["run_id"], provenance["run_attempt"])
        archive_identity = provenance["local_archive_sha256"]
        if run_identity in seen_runs or provenance["job_id"] in seen_jobs \
                or provenance["artifact_id"] in seen_artifacts or archive_identity in seen_archives:
            raise ValueError("duplicate hybrid run/job/artifact/archive provenance")
        seen_runs.add(run_identity); seen_jobs.add(provenance["job_id"])
        seen_artifacts.add(provenance["artifact_id"]); seen_archives.add(archive_identity)
        record = _read_hybrid_artifact(artifact_dir, provenance, request)
        role_identity = (record["role"], canonical_digest(record["config"]))
        if record["role"] != row["role"] or record["config"] != row["config"] \
                or record["packet_identity"]["bundle_sha256"] != row["bundle_sha256"] \
                or (record["role"], record["config"]) not in HYBRID_ROLE_CONFIGS + FROZEN_HYBRID_ROLE_CONFIGS \
                or role_identity in seen_roles:
            raise ValueError("duplicate or unapproved hybrid role/config")
        if record["packet_identity"]["bundle_sha256"] in seen_bundles:
            raise ValueError("duplicate hybrid dispatch bundle")
        seen_roles.add(role_identity); seen_bundles.add(record["packet_identity"]["bundle_sha256"])
        role_records.append(record)
    observed_roles = {(record["role"], canonical_digest(record["config"])) for record in role_records}
    generic_roles = {(role, canonical_digest(dict(config))) for role, config in HYBRID_ROLE_CONFIGS}
    frozen_roles = {(role, canonical_digest(dict(config))) for role, config in FROZEN_HYBRID_ROLE_CONFIGS}
    if observed_roles not in (generic_roles, frozen_roles):
        raise ValueError("exact baseline/m20/m32 hybrid role evidence required")
    frozen_mode = observed_roles == frozen_roles
    shared_prepare = None
    if frozen_mode:
        prepares = [record["frozen_prepare"] for record in role_records]
        if any(prepare is None for prepare in prepares) or len({canonical_digest(prepare) for prepare in prepares}) != 1:
            raise ValueError("mixed or missing frozen prepare evidence")
        shared_prepare = prepares[0]
    baseline_record = next(record for record in role_records
                           if record["role"] == "baseline" and record["config"] == HYBRID_BASELINE)
    queries_path = Path(__file__).resolve().parent / "queries.jsonl"
    queries = [json.loads(line) for line in queries_path.read_text(encoding="utf-8").splitlines() if line]
    candidate_records: list[dict[str, Any]] = []
    for candidate_record in sorted(
            (record for record in role_records if record["role"] != "baseline"),
            key=lambda record: record["config"]["m"]):
        original_paired = _pair_hybrid_role_observations(
            baseline=baseline_record["original_observations"],
            candidate=candidate_record["original_observations"], label="original")
        expanded_paired = _pair_hybrid_role_observations(
            baseline=baseline_record["expanded_observations"],
            candidate=candidate_record["expanded_observations"], label="expanded")
        aggregate_metrics = {"original_absolute": aggregate_hybrid_serialized_metrics(
            specifications=queries, observations=original_paired)}
        expanded_scale_diagnostics = aggregate_hybrid_serialized_scale_diagnostics(
            observations=expanded_paired)
        gates = recompute_hybrid_gate_verdicts(
            original_absolute=aggregate_metrics["original_absolute"],
            expanded_scale_diagnostics=expanded_scale_diagnostics, committed_baseline=None,
            baselines_sha256=candidate_record["packet_identity"]["baselines_sha256"],
        )
        candidate_records.append({
            "candidate": candidate_record["config"], "status": gates["candidate_verdict"],
            "original_absolute_gate": gates["original_absolute_gate"],
            "expanded_30k_scale_diagnostics": gates["expanded_30k_scale_diagnostics"],
            "aggregate_metrics": aggregate_metrics,
            "packet_identity": candidate_record["packet_identity"],
            "provenance": candidate_record["provenance"],
        })
    ledger = {
        "schema_version": HYBRID_POSTDOWNLOAD_LEDGER_SCHEMA_VERSION,
        "campaign_stage": "hybrid", "authorization": "none",
        "hybrid_request_sha256": request["record_self_sha256"],
        "dense_ledger_sha256": request["dense_ledger_sha256"], "dense_source_head": request["dense_source_head"],
        "hybrid_implementation_head": head, "baseline": request["baseline"],
        "candidates": request["candidates"], "scale": request["scale"], "query_count": request["query_count"],
        "evidence_manifest_sha256": manifest["record_self_sha256"],
        "baseline_record": {
            "config": baseline_record["config"],
            "packet_identity": baseline_record["packet_identity"],
            "provenance": baseline_record["provenance"],
        },
        "candidate_records": candidate_records,
    }
    if frozen_mode:
        ledger["frozen_prepare"] = shared_prepare
    ledger["record_self_sha256"] = canonical_digest(ledger)
    return ledger


def _recall_family_confirms(
    comparisons: list[dict[str, Any]], adjusted_p_values: list[float],
) -> bool:
    """Apply the shared direction/CI/Holm/non-regression rule to two recalls."""
    if len(comparisons) != 2 or len(adjusted_p_values) != 2:
        raise ValueError("confirmation selector requires both recall metrics")
    metrics = {
        comparison.get("comparison", {}).get("metric")
        for comparison in comparisons if isinstance(comparison, dict)
    }
    if metrics != {"recall_at_10", "recall_at_20"}:
        raise ValueError("confirmation selector recall family")
    positive = False
    for comparison, adjusted in zip(comparisons, adjusted_p_values, strict=True):
        effect, interval = comparison.get("mean_effect"), comparison.get("basic_ci_95")
        if isinstance(effect, bool) or not isinstance(effect, (int, float)) or not math.isfinite(effect) \
                or not isinstance(interval, list) or len(interval) != 2 \
                or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in interval) \
                or isinstance(adjusted, bool) or not isinstance(adjusted, (int, float)) or not math.isfinite(adjusted):
            raise ValueError("confirmation selector finite statistics")
        positive |= effect > 0 and interval[0] > 0 and adjusted <= 0.05
        if interval[1] < 0:
            return False
    return positive


def _confirmed_d04_m(ledger: dict[str, Any]) -> list[int]:
    records = ledger.get("eligible_evidence_runs")
    if not isinstance(records, list) or len(records) != 6:
        raise ValueError("confirmation selector requires six eligible records")
    candidates = sorted({record.get("slot", {}).get("m") for record in records})
    confirmed: list[int] = []
    for m in candidates:
        replicates = [record for record in records if record.get("slot", {}).get("m") == m]
        if len(replicates) != 3 or {record.get("slot", {}).get("ordinal") for record in replicates} != {1, 2, 3}:
            raise ValueError("confirmation selector requires three primary replicates per m")
        passes = []
        for record in replicates:
            family = record.get("d04")
            comparisons = family.get("comparisons") if isinstance(family, dict) else None
            adjusted = family.get("holm_adjusted_p_values") if isinstance(family, dict) else None
            if not isinstance(comparisons, list) or not isinstance(adjusted, list) or len(comparisons) != len(adjusted):
                raise ValueError("confirmation selector D-04 family")
            selected = [
                (comparison, p_value)
                for comparison, p_value in zip(comparisons, adjusted, strict=True)
                if comparison.get("comparison", {}).get("m") == m
            ]
            passes.append(_recall_family_confirms(
                [item[0] for item in selected], [item[1] for item in selected],
            ))
        if all(passes):
            confirmed.append(m)
    return confirmed


def _confirmed_d20_m(ledger: dict[str, Any]) -> list[int]:
    families = ledger.get("d20_ordinal_families")
    records = ledger.get("eligible_evidence_runs")
    if not isinstance(families, list) or len(families) != 3 or not isinstance(records, list):
        raise ValueError("confirmation selector D-20 ordinal families")
    candidates = sorted({record.get("slot", {}).get("m") for record in records})
    confirmed: list[int] = []
    for m in candidates:
        ordinal_passes = []
        for family in families:
            comparisons = family.get("comparisons") if isinstance(family, dict) else None
            adjusted = family.get("holm_adjusted_p_values") if isinstance(family, dict) else None
            if not isinstance(comparisons, list) or not isinstance(adjusted, list) or len(comparisons) != len(adjusted):
                raise ValueError("confirmation selector D-20 family")
            selected = [
                (comparison, p_value)
                for comparison, p_value in zip(comparisons, adjusted, strict=True)
                if comparison.get("comparison", {}).get("candidate_m") == m
            ]
            ordinal_passes.append(_recall_family_confirms(
                [item[0] for item in selected], [item[1] for item in selected],
            ))
        if all(ordinal_passes):
            confirmed.append(m)
    return confirmed


def _confirmation_candidate_measurements(ledger: dict[str, Any], m: int) -> dict[str, float | int]:
    """Reduce three primary ef=300 records to one deterministic D-21 input."""
    observations: list[dict[str, float]] = []
    for record in ledger["eligible_evidence_runs"]:
        if record.get("slot", {}).get("m") != m:
            continue
        result = record.get("validated_measurements")
        builds = result.get("builds") if isinstance(result, dict) else None
        if not isinstance(builds, list):
            raise ValueError("validated confirmation measurements required for continuation")
        matched = [build for build in builds if build.get("build", {}).get("m") == m]
        if len(matched) != 1:
            raise ValueError("one primary build measurement required")
        card = matched[0]["build"]
        groups = matched[0].get("queries")
        query = [group for group in groups if group.get("query_ef") == 300] if isinstance(groups, list) else []
        if len(query) != 1:
            raise ValueError("one primary ef=300 measurement required")
        values = {
            "recall_at_10": query[0].get("recall_at_10"),
            "recall_at_20": query[0].get("recall_at_20"),
            "p95_ms": query[0].get("latency_p95_ms"),
            "index_bytes": card.get("index_bytes"),
            "build_time_s": card.get("index_build_ms") / 1000 if isinstance(card.get("index_build_ms"), (int, float)) else None,
        }
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in values.values()):
            raise ValueError("finite primary continuation measurements required")
        observations.append({name: float(value) for name, value in values.items()})
    if len(observations) != 3:
        raise ValueError("three primary measurements required for continuation")
    return {
        "m": m,
        **{name: statistics.median(row[name] for row in observations) for name in observations[0]},
    }


def _write_json_atomic(path: Path, record: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError(f"refusing to overwrite continuation authority: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def emit_confirmation_continuation(
    request: dict[str, Any], ledger: dict[str, Any], *,
    continuation_request_output: Path, continuation_preflight_output: Path,
    no_continuation_output: Path,
) -> dict[str, Any]:
    """Materialize exactly one D-20..D-22 authority outcome and link it."""
    raise ValueError("D-25 retires all confirmation continuation authority")
    outputs = {continuation_request_output, continuation_preflight_output, no_continuation_output}
    if len(outputs) != 3 or any(path.exists() or path.is_symlink() for path in outputs):
        raise ValueError("continuation output paths must be distinct and unused")
    evidence_sha256 = ledger.get("record_self_sha256")
    if not isinstance(evidence_sha256, str) or not _HEX64.fullmatch(evidence_sha256):
        raise ValueError("sealed confirmation evidence required for continuation")
    confirmed_d04 = _confirmed_d04_m(ledger)
    confirmed_d20 = _confirmed_d20_m(ledger)
    artifacts: dict[str, Any] = {
        "result": "continuation" if confirmed_d04 or not confirmed_d20 else "no-continuation",
        "reconciled_evidence_sha256": evidence_sha256,
        "continuation_request_sha256": None,
        "continuation_preflight_sha256": None,
        "no_continuation_sha256": None,
    }
    if not confirmed_d04 and confirmed_d20:
        decision = {
            "schema_version": 1,
            "kind": "phase07-no-continuation/v1",
            "campaign_stage": "confirmation",
            "confirmation_request_sha256": request["record_self_sha256"],
            "confirmation_evidence_sha256": evidence_sha256,
            "result": "no-continuation",
            "reason": "D-04 blocks Stage 2 and D-20 makes the matched FLAT diagnostic unnecessary",
            "confirmed_d04_m": confirmed_d04,
            "confirmed_d20_m": confirmed_d20,
            "selected_branches": [],
            "skipped_branches": ["stage2_sq", "flat_diagnostic", "refinement"],
            "authorization": "none",
        }
        decision["record_self_sha256"] = canonical_digest(decision)
        _write_json_atomic(no_continuation_output, decision)
        artifacts["no_continuation_sha256"] = decision["record_self_sha256"]
    else:
        measurements = [
            _confirmation_candidate_measurements(ledger, m)
            for m in (confirmed_d04 or sorted({record["slot"]["m"] for record in ledger["eligible_evidence_runs"]}))
        ]
        if confirmed_d04:
            stage2 = select_stage2(measurements)
            selected_branch = "stage2_sq"
            bindings = [
                {
                    "mode": "stage2_sq",
                    "prior_evidence_sha256": evidence_sha256,
                    "config": {"approved_d04_sha256": evidence_sha256, "m": candidate["m"]},
                }
                for candidate in stage2["stage2_candidates"]
            ]
            skipped = ["flat_diagnostic"]
        else:
            selected = min(
                measurements,
                key=lambda item: (
                    -(float(item["recall_at_10"]) + float(item["recall_at_20"])),
                    float(item["p95_ms"]), float(item["index_bytes"]),
                    float(item["build_time_s"]), int(item["m"]),
                ),
            )
            selected_branch = "flat_diagnostic"
            bindings = [{
                "mode": "flat_diagnostic",
                "prior_evidence_sha256": evidence_sha256,
                "config": {"no_confirmed_sq_sha256": evidence_sha256, "m": selected["m"], "query_ef": 300},
            }]
            skipped = ["stage2_sq", "refinement"]
        continuation = {
            "schema_version": 1,
            "kind": "phase07-continuation-request/v1",
            "campaign_stage": "continuation",
            "confirmation_request_sha256": request["record_self_sha256"],
            "confirmation_evidence_sha256": evidence_sha256,
            "selection": {
                "selected_branch": selected_branch,
                "confirmed_d04_m": confirmed_d04,
                "confirmed_d20_m": confirmed_d20,
                "continuation_bindings": bindings,
                "skipped_branches": skipped,
            },
            "authorization": "none",
        }
        continuation["record_self_sha256"] = canonical_digest(continuation)
        authority_path = (REPOSITORY_ROOT / request["stage1_ledger_path"]).resolve()
        if not authority_path.is_relative_to(REPOSITORY_ROOT):
            raise ValueError("Stage 1 authority path containment")
        authority = _read_json(authority_path)
        if authority.get("original_ledger_sha256") != request.get("stage1_ledger_sha256"):
            raise ValueError("Stage 1 authority/request binding")
        preflight = {
            "repository": authority["repository"],
            "branch": authority["branch"],
            "worktree_root": str(REPOSITORY_ROOT),
            "head_sha": request["post_task0_head"],
            "allowed_dirty_paths": [],
            "workflow_name": "eval",
            "campaign_stage": "continuation",
            "continuation_binding": {
                "kind": continuation["kind"],
                "continuation_request_sha256": continuation["record_self_sha256"],
                "confirmation_evidence_sha256": evidence_sha256,
            },
            "require_upstream_head": True,
            "ledger_path": str(continuation_preflight_output.with_name(
                continuation_preflight_output.name.replace("-request", "-ledger")
            ).resolve()),
        }
        _write_json_atomic(continuation_request_output, continuation)
        _write_json_atomic(continuation_preflight_output, preflight)
        artifacts["continuation_request_sha256"] = continuation["record_self_sha256"]
        artifacts["continuation_preflight_sha256"] = _canonical_sha256(preflight)
    linked = dict(ledger)
    linked["continuation_artifacts"] = artifacts
    linked["record_self_sha256"] = canonical_digest(linked)
    return linked


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


def _stage1_timestamp(value: Any, *, label: str) -> dt.datetime:
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ValueError(f"Stage 1 {label} timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"Stage 1 {label} timestamp timezone")
    return parsed.astimezone(dt.timezone.utc)


def _validate_stage1_workflow_retention(request: dict[str, Any]) -> None:
    """Tie the claimed API retention to the exact screening workflow contract."""
    if request["workflow_retention_days"] != 90:
        raise ValueError("Stage 1 workflow retention must be exactly 90 days")
    workflow = (REPOSITORY_ROOT / request["workflow_path"]).resolve()
    if workflow.parent != (REPOSITORY_ROOT / ".github" / "workflows").resolve():
        raise ValueError("Stage 1 workflow path containment")
    try:
        section = workflow.read_text(encoding="utf-8").split("  phase07-screening:", 1)[1].split(
            "  phase07-confirmation:", 1
        )[0]
    except (OSError, IndexError) as exc:
        raise ValueError("Stage 1 screening workflow unavailable") from exc
    if "retention-days: 90" not in section:
        raise ValueError("Stage 1 screening workflow retention declaration")


def _validate_stage1_runner(runner: Any) -> dict[str, Any]:
    if not isinstance(runner, dict) or set(runner) != {"name", "group", "labels", "os", "image", "architecture"}:
        raise ValueError("Stage 1 runner identity schema")
    if not isinstance(runner["name"], str) or not runner["name"].startswith("GitHub Actions ") \
            or runner["group"] != "GitHub Actions" or runner["labels"] != ["ubuntu-latest"] \
            or runner["os"] != "Linux" or runner["architecture"] != "X64" \
            or not isinstance(runner["image"], str) or not runner["image"]:
        raise ValueError("Stage 1 must use hosted ubuntu-latest Linux/X64")
    return runner


def _validate_stage1_api_provenance(request: dict[str, Any], runner: dict[str, Any]) -> dt.datetime:
    provenance = request["api_provenance"]
    if not isinstance(provenance, dict) or set(provenance) != _STAGE1_API_PROVENANCE_FIELDS:
        raise ValueError("strict Stage 1 API provenance schema")
    workflow_run, job, artifact = (
        provenance["workflow_run"], provenance["job"], provenance["artifact"],
    )
    if not isinstance(workflow_run, dict) or set(workflow_run) != _STAGE1_API_WORKFLOW_RUN_FIELDS \
            or not isinstance(job, dict) or set(job) != _STAGE1_API_JOB_FIELDS \
            or not isinstance(artifact, dict) or set(artifact) != _STAGE1_API_ARTIFACT_FIELDS:
        raise ValueError("strict Stage 1 API provenance members")
    if workflow_run != {
        "run_id": request["run_id"], "run_attempt": request["run_attempt"],
        "head_branch": request["branch"], "head_sha": request["head_sha"],
        "event": request["event"], "status": request["status"],
        "conclusion": request["conclusion"], "created_at": request["run_created_at"],
    }:
        raise ValueError("Stage 1 workflow API provenance binding")
    if job != {
        "job_id": request["job_id"], "run_id": request["run_id"],
        "name": "Phase 07 bounded SQ screening campaign", "status": "completed",
        "conclusion": "success", "runner_name": runner["name"],
        "runner_group_name": runner["group"], "labels": runner["labels"],
    }:
        raise ValueError("Stage 1 job API provenance binding")
    request_artifact = request["artifact"]
    if artifact != {
        "artifact_id": request_artifact["artifact_id"], "job_id": request["job_id"],
        "job_key": request["job_key"], "run_id": request["run_id"],
        "name": request_artifact["name"], "created_at": request_artifact["created_at"],
        "expires_at": request_artifact["expires_at"], "expired": request_artifact["expired"],
    }:
        raise ValueError("Stage 1 artifact API provenance binding")
    return _stage1_timestamp(workflow_run["created_at"], label="workflow-run creation")


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
    runner = _validate_stage1_runner(request["runner"])
    run_created = _validate_stage1_api_provenance(request, runner)
    _validate_stage1_workflow_retention(request)
    if not all(isinstance(request[name], str) and _HEX64.fullmatch(request[name]) for name in ("model_manifest_sha256", "corpus_manifest_sha256", "lock_identity", "campaign_result_sha256", "campaign_request_sha256", "campaign_ledger_sha256")):
        raise ValueError("Stage 1 digest identity")
    retry = request["retry_lineage"]
    if not isinstance(retry, dict) or set(retry) != {"failure_class", "original_run_id", "replacement_run_id"} or retry != {"failure_class": None, "original_run_id": None, "replacement_run_id": None}:
        raise ValueError("Stage 1 retry lineage")
    artifact = request["artifact"]
    required_artifact = {"artifact_id", "name", "job_id", "job_key", "retention_days_requested", "retention_days_accepted", "created_at", "expires_at", "expired", "api_archive_sha256", "local_archive_path", "local_archive_sha256", "content_tree_sha256"}
    if not isinstance(artifact, dict) or set(artifact) != required_artifact or not isinstance(artifact["artifact_id"], int) or artifact["artifact_id"] <= 0 or artifact["job_id"] != request["job_id"] or artifact["job_key"] != request["job_key"] or not isinstance(artifact["name"], str) or not artifact["name"] or artifact["retention_days_requested"] != 90 or artifact["retention_days_accepted"] != 90 or artifact["expired"] is not False:
        raise ValueError("Stage 1 artifact identity/retention")
    created = _stage1_timestamp(artifact["created_at"], label="artifact creation")
    expires = _stage1_timestamp(artifact["expires_at"], label="artifact expiry")
    if created < run_created or not dt.timedelta(days=89, hours=23, minutes=59, seconds=30) <= expires - run_created <= dt.timedelta(days=90, seconds=30):
        raise ValueError("Stage 1 artifact retention interval")
    if not isinstance(artifact["local_archive_path"], str) or Path(artifact["local_archive_path"]).name != artifact["local_archive_path"]:
        raise ValueError("Stage 1 local archive path")
    archive = artifact_dir / artifact["local_archive_path"]
    if archive.is_symlink() or not archive.is_file() or _stage1_file_sha256(archive) != artifact["local_archive_sha256"] or artifact["api_archive_sha256"] != artifact["local_archive_sha256"]:
        raise ValueError("Stage 1 archive digest binding")
    extracted = artifact_dir / "extracted"
    if extracted.is_symlink() or not extracted.is_dir() or not all(isinstance(artifact[name], str) and _HEX64.fullmatch(artifact[name]) for name in ("api_archive_sha256", "local_archive_sha256", "content_tree_sha256")) or _stage1_tree_sha256(extracted) != artifact["content_tree_sha256"]:
        raise ValueError("Stage 1 content-tree digest binding")


def _validate_stage1_nominee_list(value: Any, *, label: str) -> list[int]:
    """Accept only the small, explicit Stage 1 nominee domain."""
    if not isinstance(value, list) or len(value) > 2 \
            or len(set(value)) != len(value) \
            or any(not isinstance(m, int) or isinstance(m, bool) or m not in {16, 20, 32} for m in value):
        raise ValueError(f"Stage 1 {label} nomination boundary")
    return list(value)


def _validate_stage1_result(result_record: dict[str, Any], request_record: dict[str, Any], *, expected_shape: tuple[int, int, int]) -> tuple[dict[str, Any], list[int]]:
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
    # The generator is pinned by source head, seed, shape, algorithm, locked
    # runtime, OMP setting, hosted runner and sealed artifact.  Re-generating
    # normalized float32 vectors on a different BLAS/OS is not a trustworthy
    # equality oracle: it can change bytes despite identical locked inputs.
    # The runner-produced corpus/query digests are therefore immutable artifact
    # identities; the exact IDs and recalls below are independently recomputed.
    if not isinstance(stress, dict) or set(stress) != {"schema_version", "corpus_sha256", "query_sha256", "exact_truth_sha256", "corpus_seed", "query_seed", "shape", "algorithm"} or stress.get("schema_version") != 1 or stress.get("corpus_seed") != benchmark.CORPUS_SEED or stress.get("query_seed") != benchmark.QUERY_SEED or stress.get("shape") != expected_stress_shape or stress.get("algorithm") != expected_algorithm or not all(isinstance(stress.get(name), str) and _HEX64.fullmatch(stress[name]) for name in ("corpus_sha256", "query_sha256", "exact_truth_sha256")) or stress["corpus_sha256"] == stress["query_sha256"]:
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
    reported_nominees = _validate_stage1_nominee_list(
        result.get("nominated_m"), label="artifact-reported",
    )
    # The screening artifact records the campaign's own provisional list, but
    # the reconciler is its authority: recompute the D-04-qualified ranking
    # from sealed numeric observations before publishing any nominee.
    reconciled = dict(result)
    reconciled["nominated_m"] = _validate_stage1_nominee_list(
        select_stage1_nominees(builds, statistics), label="reconciled",
    )
    return reconciled, reported_nominees


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
    if set(ledger_record) != {"schema_version", "stage", "authorization", "request_sha256", "record_self_sha256"} \
            or ledger_record.get("schema_version") != 1 or ledger_record.get("stage") != "screening" \
            or ledger_record.get("authorization") != "none" \
            or ledger_record.get("request_sha256") != campaign_digest(request_record):
        raise ValueError("Stage 1 campaign ledger")
    result, artifact_reported_nominees = _validate_stage1_result(
        result_record, request_record, expected_shape=expected_shape,
    )
    ledger = {
        "schema_version": 1, "mode": "screening", "status": "success", "authorization": "none",
        "repository": request["repository"], "branch": request["branch"], "workflow_path": request["workflow_path"], "head_sha": request["head_sha"],
        "run": {name: request[name] for name in ("run_id", "run_attempt", "job_id", "job_key", "job_allocation_nonce")},
        "artifact": request["artifact"], "runtime": request["runtime"], "runner": request["runner"],
        "model_manifest_sha256": request["model_manifest_sha256"], "corpus_manifest_sha256": request["corpus_manifest_sha256"], "lock_identity": request["lock_identity"],
        "stress_identity": result["stress_identity"],
        "builds": result["builds"],
        "d04_statistics": result["d04_statistics"],
        "artifact_reported_nominated_m": artifact_reported_nominees,
        "nominated_m": result["nominated_m"],
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


def main(argv: list[str] | None = None) -> int:
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
    parser.add_argument("--confirmation-request", type=Path)
    parser.add_argument("--confirmation-ledger", type=Path)
    parser.add_argument("--confirmation-evidence-manifest", type=Path)
    parser.add_argument("--hybrid-request", type=Path)
    parser.add_argument("--hybrid-evidence-manifest", type=Path)
    parser.add_argument("--continuation-request-output", type=Path)
    parser.add_argument("--continuation-preflight-output", type=Path)
    parser.add_argument("--no-continuation-output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.hybrid_evidence_manifest is not None or args.hybrid_request is not None:
            if args.hybrid_request is None or args.hybrid_evidence_manifest is None \
                    or args.output is None or args.mode != "hybrid-postdownload":
                raise ValueError("post-download hybrid reconciliation requires request, manifest, output, and mode")
            result = reconcile_hybrid_postdownload(
                _read_json(args.hybrid_request), _read_json(args.hybrid_evidence_manifest),
            )
            if args.output.exists() or args.output.is_symlink():
                raise ValueError("refusing to overwrite hybrid evidence ledger")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print("[PASS] post-download hybrid ANN reconciliation", file=sys.stderr)
            return 0
        if args.confirmation_evidence_manifest is not None:
            if args.confirmation_request is None or args.output is None or args.mode != "confirmation-postdownload":
                raise ValueError("post-download confirmation reconciliation requires request, manifest, output, and mode")
            result = reconcile_confirmation_postdownload(
                _read_json(args.confirmation_request), _read_json(args.confirmation_evidence_manifest),
            )
            continuation_outputs = (
                args.continuation_request_output,
                args.continuation_preflight_output,
                args.no_continuation_output,
            )
            if any(path is not None for path in continuation_outputs):
                raise ValueError("D-25 rejects retired continuation authority outputs")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print("[PASS] post-download confirmation ANN reconciliation", file=sys.stderr)
            return 0
        if args.confirmation_request is not None or args.confirmation_ledger is not None:
            if args.confirmation_request is None or args.confirmation_ledger is None or args.mode != "confirmation":
                raise ValueError("confirmation reconciliation requires request, downloaded ledger, and confirmation mode")
            result = reconcile_confirmation_request(_read_json(args.confirmation_request), _read_json(args.confirmation_ledger))
            args.confirmation_ledger.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            print("[PASS] confirmation ANN reconciliation", file=sys.stderr)
            return 0
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
