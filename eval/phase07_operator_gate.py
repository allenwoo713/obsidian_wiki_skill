"""Fail-closed local operator gates for the Phase 07 hosted evidence workflow.

The program deliberately accepts only fixed JSON request/ledger paths.  It never
contacts GitHub and it rejects unsealed or unsafe operator state before a caller
can push, dispatch, or rely on an artifact.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


# ``-m`` executes this source as ``__main__`` while the production runner
# imports its package name.  Alias the executing module before any capability
# issuer exists, or fail closed rather than minting a second issuer.
_CANONICAL_MODULE_NAME = "eval.phase07_operator_gate"
if __name__ == "__main__":
    _executing_module = sys.modules[__name__]
    _canonical_module = sys.modules.get(_CANONICAL_MODULE_NAME)
    if _canonical_module is not None and _canonical_module is not _executing_module:
        raise RuntimeError("duplicate phase07 operator module identity")
    sys.modules[_CANONICAL_MODULE_NAME] = _executing_module


# ``python eval/phase07_operator_gate.py`` puts only ``eval/`` on sys.path.
# Establish the checkout root before any delayed ``eval.*`` import so the
# direct-file command used by both hosted runner families is deterministic.
_REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))


SHA = re.compile(r"^[0-9a-f]{40}$")
STAGES = frozenset({"preflight", "screening", "confirmation", "continuation", "pr-acceptance", "hybrid"})
INFRA_FAILURES = frozenset({"github_infrastructure", "hosted_runner_unavailable", "artifact_service"})
SECRET_MARKERS = ("token", "secret", "password", "authorization", "private_key", "ghp_")
HEX64 = re.compile(r"^[0-9a-f]{64}$")
STAGE1_RUNTIME_FIELDS = frozenset({
    "branch", "workflow_path", "head_sha", "run_id", "run_attempt", "job_key", "job_allocation_nonce", "runtime",
})
STAGE1_RUNTIME_IDENTITY = {
    "python": "3.13", "lancedb": "0.34.0", "numpy": "2.2.6", "pyarrow": "25.0.0",
    "omp_num_threads": 2,
}
STAGE1_LEDGER_DIGEST = "6c424135ec4db8983136826575d9436b6ba88da5029384ada47fadb2d1918e33"
STAGE1_NUMERIC_HEAD = "c45934d7fb8e699faa389c1dc3e80bb9dcc774c8"
CONFIRMATION_IMPLEMENTATION_BASE = "105dab9764cb58fb10610b78345b85d7282ef4d6"
CANONICAL_STAGE1_AUTHORITY_PATH = _REPOSITORY_ROOT / "eval" / "phase07-stage1-authority.json"
CANONICAL_STAGE1_AUTHORITY_REFERENCE = "eval/phase07-stage1-authority.json"
STAGE1_AUTHORITY_DIGEST = "a983ac4be1bb992e237474fc237cd927529ef9dd60e9330de7b959e6159bb319"
CONFIRMATION_SLOTS = (1, 2, 3)
D25_BASELINE = {"index_type": "hnsw_sq", "m": 16, "ef_construction": 300, "query_ef": 100}
D25_CANDIDATES = (
    {"index_type": "hnsw_sq", "m": 20, "ef_construction": 300, "query_ef": 300},
    {"index_type": "hnsw_sq", "m": 32, "ef_construction": 300, "query_ef": 300},
)
CONFIRMATION_WORKFLOW_INPUT_FIELDS = frozenset({
    "schema_version", "campaign_stage", "confirmation_request_sha256",
    "slot", "post_task0_head", "d25_binding",
    "d25_binding_sha256", "dispatch_identity", "record_self_sha256",
})
JOB_DISPLAY_NAMES = {
    "phase07-confirmation": "Phase 07 independent confirmation campaign",
    "phase07-hybrid": "Phase 07 independent hybrid campaign",
    "phase07-hybrid-prepare": "Phase 07 candidate-neutral frozen corpus prepare",
}
CONFIRMATION_DOWNLOAD_FIELDS = frozenset({"run_id", "run_attempt", "archive", "extracted_dir"})
DENSE_LEDGER_DIGEST = "71335b6bfa03f24368414ae56a22fd8896d4479c6bfbe871c36a14b26e3b211b"
DENSE_SOURCE_HEAD = "2f15d6a4fef54dda9b0f4a258e78898e2ef6ea57"
HYBRID_BASELINE = {"index_type": "hnsw_sq", "m": 16, "ef_construction": 300, "query_ef": 100}
HYBRID_CANDIDATES = (
    {"index_type": "hnsw_sq", "m": 20, "ef_construction": 300, "query_ef": 300},
    {"index_type": "hnsw_sq", "m": 32, "ef_construction": 300, "query_ef": 300},
)
HYBRID_ROLE_CONFIGS = (
    ("baseline", HYBRID_BASELINE),
    ("candidate", HYBRID_CANDIDATES[0]),
    ("candidate", HYBRID_CANDIDATES[1]),
)
HYBRID_WORKFLOW_INPUT_FIELDS = frozenset({
    "schema_version", "campaign_stage", "hybrid_request_sha256", "dense_source_head",
    "hybrid_implementation_head", "role", "config", "scale", "query_count",
    "authorization", "retention_days", "replacement_for_run_id", "dispatch_identity",
    "record_self_sha256",
})
HYBRID_DENSE_ORDINAL_IDENTITIES = (
    {"ordinal": 1, "run_id": 32801985769, "run_attempt": 1, "job_id": 97664517767, "artifact_id": 9546915747,
     "build_ids": ["078cc8451c21e17dfac726d6a26aa7519375756d1ec71c38ce7ecf8bc5f256dc", "d550c98e3aff255c53d459b0ffe19b00d19d6afa702023e37558777fe9223e73", "bb0b4a1fd23cbdfab4845a4975a0119f4d9963fc65bf1d6afd825d4ba6d2b42a"]},
    {"ordinal": 2, "run_id": 32802007002, "run_attempt": 1, "job_id": 97664580321, "artifact_id": 9546916208,
     "build_ids": ["45d772249ce3790b85955ca68cbea16d5a003e8db34177609bb18f3a9536fd02", "42aaee989e396e3b9030bdd099b183a1903b80f8fcd1d0d07c9694828e2f8744", "2674f4dd7caba0744a0cf499fece5d8bf0b2c7841b22cc8b62113e3a976a87c4"]},
    {"ordinal": 3, "run_id": 32802027355, "run_attempt": 1, "job_id": 97664640212, "artifact_id": 9546924769,
     "build_ids": ["825c1a3f523e62affe8385443f628aa3d0638db7468f8c6871e76a9038ef44c0", "b6f66e475137e3e58c7554d503b0714586df6a0558b5ab9c22b2118cf28ea4e4", "95194274760a8b9f083c629250e4437d4199efe633241a0ae02bb161300b848f"]},
)

# The frozen prepare is intentionally a separate state machine from the
# legacy three-role hybrid request.  A local measurement can authorize *only*
# a prepare dispatch; an artifact identity does not exist until that run has
# completed, so requiring one at this point would be a circular future-ID gate.
FROZEN_ROLE_CONFIGS = (
    ("baseline", {"index_type": "hnsw_sq", "m": 16, "ef_construction": 300, "query_ef": 100}),
    ("m20", {"index_type": "hnsw_sq", "m": 20, "ef_construction": 300, "query_ef": 300}),
    ("m32", {"index_type": "hnsw_sq", "m": 32, "ef_construction": 300, "query_ef": 300}),
)
# Generic PR evidence retains its historic candidate labels.  D-25 frozen
# dispatches use the stricter immutable role names below.
FROZEN_HYBRID_ROLE_CONFIGS = FROZEN_ROLE_CONFIGS

def validate_frozen_prepare_identity(identity: object, *, expected_head: str,
                                     locked_execution: object) -> dict[str, Any]:
    """Bind A's one strict identity shape to measured locked execution facts."""
    if not isinstance(identity, dict) or not SHA.fullmatch(expected_head):
        raise ValueError("strict frozen prepare identity")
    from eval.phase07_frozen_base import validate_frozen_prepare_identity_shape
    identity = validate_frozen_prepare_identity_shape(
        identity, expected_repository=identity.get("repository"), expected_head=expected_head,
    )
    from eval.phase07_ann_campaign import validate_confirmation_execution
    execution = validate_confirmation_execution(
        locked_execution,
        model_manifest_sha256=identity["model_manifest_sha256"],
        corpus_manifest_sha256=identity["corpus_manifest_sha256"],
    )
    if execution["head_sha"] != expected_head or identity["runtime"] != execution["runtime"]:
        raise ValueError("frozen prepare locked execution binding")
    return identity


def build_frozen_prepare_bundle(local_preflight: dict[str, Any], *, expected_head: str) -> dict[str, Any]:
    """Mint the sole non-authorizing prepare bundle from measured local facts.

    No remotely allocated ID is accepted here: the API identity is a result of
    this prepare run and is deliberately required only by ``build_frozen_role_bundles``.
    """
    required = {
        "schema_version", "head_sha", "target_size", "wall_time_seconds", "cap_minutes",
        "uncompressed_bytes", "archive_bytes", "file_count", "largest_file", "descriptor_sha256",
        "tree_sha256", "repository_capability", "human_authorized",
    }
    if not SHA.fullmatch(expected_head) or not isinstance(local_preflight, dict) or set(local_preflight) != required \
            or local_preflight.get("schema_version") != 1 or local_preflight.get("head_sha") != expected_head \
            or local_preflight.get("target_size") != 30000 or local_preflight.get("cap_minutes") != 120 \
            or local_preflight.get("repository_capability") is not True or local_preflight.get("human_authorized") is not True \
            or not all(isinstance(local_preflight.get(name), int) and not isinstance(local_preflight[name], bool) and local_preflight[name] > 0
                       for name in ("wall_time_seconds", "uncompressed_bytes", "archive_bytes", "file_count", "largest_file")) \
            or local_preflight["wall_time_seconds"] > 120 * 60 \
            or not all(isinstance(local_preflight.get(name), str) and HEX64.fullmatch(local_preflight[name])
                       for name in ("descriptor_sha256", "tree_sha256")):
        raise ValueError("sealed frozen local preflight")
    return _sealed({
        "schema_version": 1, "campaign_stage": "hybrid-prepare", "head_sha": expected_head,
        "local_preflight": dict(local_preflight), "authorization": "none", "retention_days": 90,
    })


def validate_frozen_prepare_bundle(value: object, *, expected_head: str) -> dict[str, Any]:
    fields = {"schema_version", "campaign_stage", "head_sha", "local_preflight", "authorization", "retention_days", "record_self_sha256"}
    if not isinstance(value, dict) or set(value) != fields or value.get("record_self_sha256") != canonical_digest(value) \
            or value.get("campaign_stage") != "hybrid-prepare" or value.get("head_sha") != expected_head \
            or value.get("authorization") != "none" or value.get("retention_days") != 90:
        raise ValueError("sealed frozen prepare bundle")
    # Reuse the source validator rather than creating a weaker second parser.
    if build_frozen_prepare_bundle(value["local_preflight"], expected_head=expected_head) != value:
        raise ValueError("sealed frozen prepare bundle")
    return value


def build_frozen_role_bundles(prepare_provenance: object, *, expected_head: str,
                              locked_execution: object | None = None) -> list[dict[str, Any]]:
    """Return exactly three role bundles only after one successful API-bound prepare.

    Rejected, pending, retry, replacement, expired, or malformed prepare evidence
    has one safe output: zero role bundles.
    """
    if locked_execution is None:
        return []
    try:
        prepare = validate_frozen_prepare_identity(
            prepare_provenance, expected_head=expected_head, locked_execution=locked_execution,
        )
    except ValueError:
        return []
    base = {
        "schema_version": 1, "campaign_stage": "hybrid", "head_sha": expected_head,
        "prepare": dict(prepare), "authorization": "none", "retention_days": 90,
    }
    return [
        _sealed({**base, "role": role, "config": dict(config), "dispatch_identity": f"phase07-hybrid/{role}"})
        for role, config in FROZEN_ROLE_CONFIGS
    ]


def seal_frozen_size_preflight(*, request_file: Path, ledger_file: Path) -> int:
    """Seal local-only 30k sizing evidence; this command never materializes it.

    The intentionally small preflight is the only route which may mint a
    prepare input.  In particular, it has no artifact/run fields: requiring
    those before dispatch would be a future-ID authorization loop.
    """
    try:
        local = _read_object(request_file)
        head = local.get("head_sha") if isinstance(local, dict) else ""
        bundle = build_frozen_prepare_bundle(local, expected_head=head)
        _write_ledger(ledger_file, bundle)
        return 0
    except (OSError, ValueError, json.JSONDecodeError):
        return 1


def run_frozen_prepare_plan(*, preflight_file: Path, workflow_input_file: Path,
                            expected_head: str) -> int:
    """Write the sole sealed prepare dispatch member from local preflight."""
    bundle = validate_frozen_prepare_bundle(_read_object(preflight_file), expected_head=expected_head)
    if workflow_input_file.exists() or workflow_input_file.is_symlink():
        raise ValueError("frozen prepare workflow input must be new")
    workflow_input_file.parent.mkdir(parents=True, exist_ok=True)
    workflow_input_file.write_text(
        json.dumps(bundle, sort_keys=True, indent=2) + "\n", encoding="utf-8",
    )
    return 0


def _frozen_archive_sha256(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("frozen prepare archive is unavailable")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_frozen_prepare_provenance(*, prepare_bundle: dict[str, Any], repository: str,
                                      run_id: int, run_attempt: int, archive: Path,
                                      frozen_dir: Path, locked_execution: dict[str, Any],
                                      token: str, client: Any | None = None) -> dict[str, Any]:
    """Return one API- and byte-bound frozen prepare identity.

    The API is queried by the caller-provided *exact* run and attempt.  A
    familiar artifact name is never a selector: it is accepted only when it is
    the one artifact returned for that exact run and the local downloaded bytes
    match GitHub's digest.  The result is the only value role planning accepts.
    """
    if not repository or not SHA.fullmatch(prepare_bundle.get("head_sha", "")) \
            or not isinstance(run_id, int) or run_id <= 0 or run_attempt != 1 or not token:
        raise ValueError("frozen prepare collection identity")
    bundle = validate_frozen_prepare_bundle(prepare_bundle, expected_head=prepare_bundle["head_sha"])
    from eval.phase07_ann_campaign import validate_confirmation_execution
    execution = validate_confirmation_execution(locked_execution)
    if execution.get("head_sha") != bundle["head_sha"]:
        raise ValueError("frozen prepare execution head")
    if frozen_dir.is_symlink() or not frozen_dir.is_dir():
        raise ValueError("frozen prepare extracted root")
    try:
        from eval.phase07_frozen_base import validate_frozen_base
        base_tree_sha256 = validate_frozen_base(frozen_dir, expected_wiki_root=frozen_dir / "Wiki")
        descriptor = _read_object(frozen_dir / "frozen-base.json")
        descriptor_self_sha256 = descriptor.get("record_self_sha256")
        if not isinstance(descriptor_self_sha256, str) or not HEX64.fullmatch(descriptor_self_sha256):
            raise ValueError("frozen prepare descriptor identity")
        from eval.ann_corpus_manifest import public_distractor_recipe_sha256
        source = execution["source_digests"]
        expected_descriptor = {
            "model_manifest_sha256": source["model_manifest_sha256"],
            "corpus_manifest_sha256": source["corpus_manifest_sha256"],
            "generator_recipe_sha256": public_distractor_recipe_sha256(),
            "runtime": execution["runtime"],
        }
        if any(descriptor.get(name) != value for name, value in expected_descriptor.items()):
            raise ValueError("frozen prepare descriptor execution binding")
        archive_sha256 = _frozen_archive_sha256(archive)
        github = client or GitHubActionsClient()
        run_url = f"/repos/{repository}/actions/runs/{run_id}"
        run = github.get_json(run_url, token)
        jobs_payload = github.get_json(f"{run_url}/attempts/{run_attempt}/jobs", token)
        artifacts_payload = github.get_json(f"{run_url}/artifacts", token)
    except Exception as exc:
        raise ValueError("frozen prepare API/download validation failed") from exc
    if not isinstance(run, dict) or run.get("id") != run_id or run.get("run_attempt") != 1 \
            or run.get("event") != "workflow_dispatch" or run.get("head_sha") != bundle["head_sha"] \
            or run.get("status") != "completed" or run.get("conclusion") != "success" \
            or not isinstance(run.get("head_branch"), str) or not run["head_branch"]:
        raise ValueError("frozen prepare workflow run")
    jobs = jobs_payload.get("jobs") if isinstance(jobs_payload, dict) else None
    matches = [job for job in jobs if isinstance(job, dict) and job.get("name") == JOB_DISPLAY_NAMES["phase07-hybrid-prepare"]
               and job.get("run_id") == run_id and job.get("run_attempt") == 1
               and job.get("status") == "completed" and job.get("conclusion") == "success"] if isinstance(jobs, list) else []
    if len(matches) != 1 or not isinstance(matches[0].get("id"), int) or matches[0]["id"] <= 0:
        raise ValueError("frozen prepare exact job")
    artifacts = artifacts_payload.get("artifacts") if isinstance(artifacts_payload, dict) else None
    name = f"phase07-frozen-base-{run_id}-1"
    artifacts = [item for item in artifacts if isinstance(item, dict) and item.get("name") == name] if isinstance(artifacts, list) else []
    if len(artifacts) != 1:
        raise ValueError("frozen prepare exact artifact")
    artifact = artifacts[0]
    created = _parse_utc_timestamp(artifact.get("created_at"), label="prepare artifact creation")
    expires = _parse_utc_timestamp(artifact.get("expires_at"), label="prepare artifact expiry")
    now = dt.datetime.now(dt.timezone.utc)
    if not isinstance(artifact.get("id"), int) or artifact["id"] <= 0 or artifact.get("expired") is not False \
            or artifact.get("digest") != f"sha256:{archive_sha256}" \
            or artifact.get("size_in_bytes") != archive.stat().st_size \
            or not isinstance(artifact.get("workflow_run"), dict) \
            or artifact["workflow_run"].get("id") != run_id \
            or artifact["workflow_run"].get("head_sha") != bundle["head_sha"] \
            or not created <= now < expires \
            or not dt.timedelta(days=89, hours=23, minutes=59, seconds=30) <= expires - created <= dt.timedelta(days=90, seconds=30):
        raise ValueError("frozen prepare artifact digest/retention")
    identity = {
        "repository": repository, "head_sha": bundle["head_sha"],
        "run_id": run_id, "run_attempt": 1, "job_id": matches[0]["id"], "artifact_id": artifact["id"],
        "artifact_name": name, "archive_sha256": archive_sha256, "archive_size_bytes": archive.stat().st_size,
        "descriptor_self_sha256": descriptor_self_sha256, "base_tree_sha256": base_tree_sha256,
        "model_manifest_sha256": source["model_manifest_sha256"],
        "corpus_manifest_sha256": source["corpus_manifest_sha256"],
        "generator_recipe_sha256": public_distractor_recipe_sha256(), "runtime": execution["runtime"],
        "artifact_created_at": artifact["created_at"], "artifact_expires_at": artifact["expires_at"],
        "retention_days": 90, "replacement_for_run_id": None, "status": "success",
    }
    return validate_frozen_prepare_identity(identity, expected_head=bundle["head_sha"], locked_execution=execution)


def build_frozen_role_dispatch_bundles(*, hybrid_request: dict[str, Any],
                                       frozen_prepare: dict[str, Any],
                                       expected_head: str) -> list[dict[str, Any]]:
    """Mint the one baseline/m20/m32 dispatch set from collector-only evidence.

    This deliberately does not take an identity file path.  The caller must
    have just obtained ``frozen_prepare`` from ``collect_frozen_prepare_provenance``
    in the same command path; a hand-authored JSON identity therefore has no
    CLI route to authorize a role plan.
    """
    request = _validate_hybrid_request(hybrid_request, expected_head=expected_head)
    from eval.phase07_frozen_base import validate_frozen_prepare_identity_shape
    prepare = validate_frozen_prepare_identity_shape(
        frozen_prepare, expected_repository="allenwoo713/obsidian_wiki_skill", expected_head=expected_head,
    )
    if prepare["head_sha"] != request["hybrid_implementation_head"]:
        raise ValueError("frozen prepare/hybrid head binding")
    bundles = []
    for role, config in FROZEN_HYBRID_ROLE_CONFIGS:
        member = _sealed({
            "schema_version": 1, "campaign_stage": "hybrid",
            "hybrid_request_sha256": request["record_self_sha256"],
            "dense_source_head": DENSE_SOURCE_HEAD,
            "hybrid_implementation_head": expected_head,
            "role": role, "config": dict(config), "scale": 30000, "query_count": 105,
            "authorization": "none", "retention_days": 90, "replacement_for_run_id": None,
            "dispatch_identity": f"phase07-hybrid/{role}",
        })
        bundles.append(_sealed({
            "schema_version": 1, "hybrid_request": request, "workflow_input": member,
            "frozen_prepare": prepare, "replacement_for_run_id": None,
        }))
    return bundles


def run_frozen_role_plan(*, prepare_bundle: Path, hybrid_request: Path, workflow_inputs_dir: Path,
                         repository: str, run_id: int, run_attempt: int, archive: Path,
                         frozen_dir: Path, locked_execution: Path, token: str,
                         client: Any | None = None) -> int:
    """Collect live prepare evidence and immediately write the exact role set."""
    request = _read_object(hybrid_request)
    head = request.get("hybrid_implementation_head") if isinstance(request, dict) else ""
    prepare = collect_frozen_prepare_provenance(
        prepare_bundle=_read_object(prepare_bundle), repository=repository, run_id=run_id,
        run_attempt=run_attempt, archive=archive, frozen_dir=frozen_dir,
        locked_execution=_read_object(locked_execution), token=token, client=client,
    )
    bundles = build_frozen_role_dispatch_bundles(
        hybrid_request=request, frozen_prepare=prepare, expected_head=head,
    )
    if workflow_inputs_dir.exists() and (workflow_inputs_dir.is_symlink() or any(workflow_inputs_dir.iterdir())):
        raise ValueError("frozen role output must be new and empty")
    workflow_inputs_dir.mkdir(parents=True, exist_ok=True)
    expected_names = ("hybrid-baseline.json", "hybrid-m20.json", "hybrid-m32.json")
    for name, bundle in zip(expected_names, bundles, strict=True):
        (workflow_inputs_dir / name).write_text(json.dumps(bundle, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    validate_hybrid_workflow_inputs_dir(workflow_inputs_dir, expected_head=head)
    return 0


def canonical_digest(payload: dict[str, Any]) -> str:
    """Digest a record without its recursive self-digest field."""
    value = dict(payload)
    value.pop("record_self_sha256", None)
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _sealed(payload: dict[str, Any]) -> dict[str, Any]:
    record = dict(payload)
    record["record_self_sha256"] = canonical_digest(record)
    return record


def d25_confirmation_binding(ordinal: int) -> dict[str, Any]:
    """Return the complete immutable D-25 build/query allowlist for one run."""
    if isinstance(ordinal, bool) or ordinal not in CONFIRMATION_SLOTS:
        raise ValueError("D-25 confirmation ordinal must be 1, 2, or 3")
    return {
        "kind": "phase07-d25-paired-confirmation/v1",
        "prior_screening_sha256": STAGE1_LEDGER_DIGEST,
        "run_ordinal": ordinal,
        "baseline": dict(D25_BASELINE),
        "candidates": [dict(candidate) for candidate in D25_CANDIDATES],
    }


def confirmation_input_membership_digest(record: dict[str, Any]) -> str:
    """Stable member identity deliberately excludes the request/self digest cycle."""
    value = dict(record)
    value.pop("record_self_sha256", None)
    value.pop("confirmation_request_sha256", None)
    return canonical_digest(value)


def _build_confirmation_plan_from_immutable(*, stage1_ledger_path: str, post_task0_head: str) -> dict[str, Any]:
    if not SHA.fullmatch(post_task0_head):
        raise ValueError("confirmation requires an exact post-Task-0 head")
    seed_records = []
    for ordinal in CONFIRMATION_SLOTS:
        slot = {"ordinal": ordinal}
        binding = d25_confirmation_binding(ordinal)
        seed_records.append({"schema_version": 1, "campaign_stage": "confirmation", "slot": slot,
                             "post_task0_head": post_task0_head, "d25_binding": binding,
                             "d25_binding_sha256": canonical_digest(binding),
                             "dispatch_identity": f"phase07-confirmation/ordinal/{ordinal}"})
    request = _sealed({
        "schema_version": 1, "campaign_stage": "confirmation",
        "stage1_ledger_path": stage1_ledger_path, "stage1_ledger_sha256": STAGE1_LEDGER_DIGEST,
        "artifact_reported_nominated_m": [16, 20], "reconciled_nominated_m": [32, 20],
        "authoritative_nominated_m": [32, 20], "stage1_numeric_head": STAGE1_NUMERIC_HEAD,
        "confirmation_implementation_base": CONFIRMATION_IMPLEMENTATION_BASE,
        "post_task0_head": post_task0_head,
        "workflow_input_membership_sha256": [confirmation_input_membership_digest(record) for record in seed_records],
    })
    inputs = []
    for record in seed_records:
        inputs.append(_sealed({**record, "confirmation_request_sha256": request["record_self_sha256"]}))
    return _sealed({
        "schema_version": 1, "confirmation_request": request, "workflow_inputs": inputs,
        "artifact_reported_nominated_m": [16, 20], "authoritative_nominated_m": [32, 20],
    })


def build_confirmation_plan(stage1_ledger: Path, *, post_task0_head: str) -> dict[str, Any]:
    """Derive all confirmation dispatch authority from the immutable Stage 1 ledger."""
    if stage1_ledger.resolve() != CANONICAL_STAGE1_AUTHORITY_PATH.resolve():
        raise ValueError("confirmation requires canonical Stage 1 authority path")
    ledger = _read_object(stage1_ledger)
    required = {"schema_version", "kind", "original_ledger_sha256", "repository", "branch", "workflow_path", "head_sha", "run", "artifact", "runner", "runtime", "source_digests", "model_manifest_sha256", "corpus_manifest_sha256", "lock_identity", "stress_identity", "artifact_reported_nominated_m", "authoritative_nominated_m", "authorization", "record_self_sha256"}
    if set(ledger) != required or ledger.get("schema_version") != 1 or ledger.get("kind") != "phase07-sealed-stage1-authority/v1":
        raise ValueError("unrecognized compact Stage 1 authority schema")
    recomputed_digest = canonical_digest(ledger)
    if (
        ledger.get("record_self_sha256") != STAGE1_AUTHORITY_DIGEST
        or recomputed_digest != STAGE1_AUTHORITY_DIGEST
        or ledger.get("original_ledger_sha256") != STAGE1_LEDGER_DIGEST
    ):
        raise ValueError("unrecognized immutable Stage 1 authority")
    if ledger.get("artifact_reported_nominated_m") != [16, 20] or ledger.get("authoritative_nominated_m") != [32, 20] or ledger.get("head_sha") != STAGE1_NUMERIC_HEAD:
        raise ValueError("immutable Stage 1 nominee/head provenance")
    return _build_confirmation_plan_from_immutable(
        stage1_ledger_path=CANONICAL_STAGE1_AUTHORITY_REFERENCE,
        post_task0_head=post_task0_head,
    )


def validate_confirmation_plan(plan: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(plan, dict) or set(plan) != {"schema_version", "confirmation_request", "workflow_inputs", "artifact_reported_nominated_m", "authoritative_nominated_m", "record_self_sha256"} or plan.get("schema_version") != 1:
        raise ValueError("sealed confirmation plan schema")
    if plan.get("record_self_sha256") != canonical_digest(plan):
        raise ValueError("confirmation plan self-digest")
    request = plan["confirmation_request"]
    required_request = {"schema_version", "campaign_stage", "stage1_ledger_path", "stage1_ledger_sha256", "artifact_reported_nominated_m", "reconciled_nominated_m", "authoritative_nominated_m", "stage1_numeric_head", "confirmation_implementation_base", "post_task0_head", "workflow_input_membership_sha256", "record_self_sha256"}
    if not isinstance(request, dict) or set(request) != required_request or request.get("record_self_sha256") != canonical_digest(request):
        raise ValueError("sealed confirmation request")
    if request["campaign_stage"] != "confirmation" or request["stage1_ledger_sha256"] != STAGE1_LEDGER_DIGEST or request["stage1_numeric_head"] != STAGE1_NUMERIC_HEAD or request["confirmation_implementation_base"] != CONFIRMATION_IMPLEMENTATION_BASE or not SHA.fullmatch(request["post_task0_head"]):
        raise ValueError("confirmation request provenance")
    if plan["artifact_reported_nominated_m"] != [16, 20] or plan["authoritative_nominated_m"] != [32, 20] or request["artifact_reported_nominated_m"] != [16, 20] or request["reconciled_nominated_m"] != [32, 20] or request["authoritative_nominated_m"] != [32, 20]:
        raise ValueError("confirmation nominees")
    inputs = plan["workflow_inputs"]
    if not isinstance(inputs, list) or len(inputs) != 3:
        raise ValueError("exactly three generated confirmation ordinal inputs")
    slots = []
    for record in inputs:
        if not isinstance(record, dict) or set(record) != CONFIRMATION_WORKFLOW_INPUT_FIELDS or record.get("record_self_sha256") != canonical_digest(record):
            raise ValueError("sealed generated workflow input")
        slot = record.get("slot")
        if not isinstance(slot, dict) or set(slot) != {"ordinal"} or slot["ordinal"] not in CONFIRMATION_SLOTS:
            raise ValueError("immutable confirmation slot")
        binding = d25_confirmation_binding(slot["ordinal"])
        if record.get("campaign_stage") != "confirmation" or record.get("confirmation_request_sha256") != request["record_self_sha256"] or record.get("post_task0_head") != request["post_task0_head"] or record.get("d25_binding") != binding or record.get("d25_binding_sha256") != canonical_digest(binding) or record.get("dispatch_identity") != f"phase07-confirmation/ordinal/{slot['ordinal']}":
            raise ValueError("confirmation input binding")
        slots.append(slot["ordinal"])
    if slots != list(CONFIRMATION_SLOTS):
        raise ValueError("duplicate, missing, or replayed confirmation slot")
    if request["workflow_input_membership_sha256"] != [confirmation_input_membership_digest(record) for record in inputs] or len(set(request["workflow_input_membership_sha256"])) != 3:
        raise ValueError("confirmation request bundle membership")
    return plan


def validate_confirmation_dispatch_bundle(bundle: dict[str, Any], *, expected_head: str) -> dict[str, Any]:
    """Require an input to carry the sealed request that proves bundle membership."""
    allowed = {"confirmation_request", "workflow_input", "replacement_for_run_id"}
    if not isinstance(bundle, dict) or set(bundle) != allowed:
        raise ValueError("typed confirmation dispatch bundle")
    request, record = bundle["confirmation_request"], bundle["workflow_input"]
    if not SHA.fullmatch(expected_head) or _git("rev-parse", "HEAD") != expected_head:
        raise ValueError("confirmation feature head is not the exact generated head")
    expected = build_confirmation_plan(CANONICAL_STAGE1_AUTHORITY_PATH, post_task0_head=expected_head)
    # The request is not a self-declared authority: it must byte-for-byte match
    # the deterministic immutable lineage and all three generated members.
    if request != expected["confirmation_request"]:
        raise ValueError("noncanonical confirmation request authority")
    validate_confirmation_plan(expected)
    slot = record.get("slot") if isinstance(record, dict) else None
    expected_record = next((item for item in expected["workflow_inputs"] if item["slot"] == slot), None)
    if expected_record is None or record != expected_record:
        raise ValueError("confirmation input is not a canonical generated member")
    if bundle.get("replacement_for_run_id") is not None:
        raise ValueError("D-25 confirmation replacements are not authorized")
    return record


def validate_confirmation_workflow_input(record: dict[str, Any], *, expected_head: str) -> dict[str, Any]:
    """Fail closed before build when a dispatched input is not one canonical slot."""
    if not SHA.fullmatch(expected_head) or not isinstance(record, dict) or set(record) != CONFIRMATION_WORKFLOW_INPUT_FIELDS or record.get("record_self_sha256") != canonical_digest(record):
        raise ValueError("sealed generated workflow input")
    slot = record.get("slot")
    if not isinstance(slot, dict) or set(slot) != {"ordinal"} or slot["ordinal"] not in CONFIRMATION_SLOTS:
        raise ValueError("immutable confirmation slot")
    binding = d25_confirmation_binding(slot["ordinal"])
    if record.get("campaign_stage") != "confirmation" or record.get("post_task0_head") != expected_head or not isinstance(record.get("confirmation_request_sha256"), str) or not HEX64.fullmatch(record["confirmation_request_sha256"]) or record.get("d25_binding") != binding or record.get("d25_binding_sha256") != canonical_digest(binding) or record.get("dispatch_identity") != f"phase07-confirmation/ordinal/{slot['ordinal']}":
        raise ValueError("confirmation input/request/head/binding mismatch")
    return record


def _validate_dense_ledger(dense_ledger: Path) -> list[dict[str, Any]]:
    """Validate the three hosted D-25 ordinals before hybrid work can exist."""
    ledger = _read_object(dense_ledger)
    required = {
        "schema_version", "campaign_stage", "confirmation_plan_sha256",
        "eligible_evidence_runs", "all_physical_workflow_runs",
        "paired_ordinal_families", "record_self_sha256",
    }
    if set(ledger) != required or ledger.get("schema_version") != 1 \
            or ledger.get("campaign_stage") != "confirmation" \
            or ledger.get("record_self_sha256") != canonical_digest(ledger) \
            or ledger.get("record_self_sha256") != DENSE_LEDGER_DIGEST:
        raise ValueError("sealed exact 07-05 dense ledger")
    records = ledger.get("eligible_evidence_runs")
    physical = ledger.get("all_physical_workflow_runs")
    if not isinstance(records, list) or len(records) != 3 or not isinstance(physical, list) or len(physical) != 3:
        raise ValueError("exact three dense ordinal records")
    expected_ids = (
        (1, 32801985769, 1, 97664517767, 9546915747),
        (2, 32802007002, 1, 97664580321, 9546916208),
        (3, 32802027355, 1, 97664640212, 9546924769),
    )
    seen_builds: set[str] = set()
    for record, (ordinal, run_id, run_attempt, job_id, artifact_id) in zip(records, expected_ids, strict=True):
        # Reconciliation enriches the packet with API provenance, so this
        # retained packet digest is intentionally not a digest of the enriched
        # ledger record.  The enclosing ledger self-digest seals that join.
        if not isinstance(record, dict) or not isinstance(record.get("record_self_sha256"), str) \
                or not HEX64.fullmatch(record["record_self_sha256"]):
            raise ValueError("sealed dense ordinal record")
        if record.get("slot") != {"ordinal": ordinal} \
                or record.get("run_id") != run_id or record.get("run_attempt") != run_attempt \
                or record.get("job_id") != job_id or record.get("job_key") != "phase07-confirmation" \
                or record.get("status") != "numeric-success" or record.get("failure_class") is not None \
                or record.get("replacement_for_run_id") is not None or record.get("retention_days") != 90:
            raise ValueError("dense ordinal run identity")
        measurements, provenance = record.get("validated_measurements"), record.get("validated_provenance")
        expires_at = provenance.get("artifact_expires_at") if isinstance(provenance, dict) else None
        try:
            expiry = dt.datetime.fromisoformat(str(expires_at).replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("dense artifact expiry") from exc
        if expiry.tzinfo is None or expiry <= dt.datetime.now(dt.timezone.utc):
            raise ValueError("dense artifact expiry")
        if not isinstance(measurements, dict) or measurements.get("authorization") != "none" \
                or not isinstance(provenance, dict) or provenance.get("head_sha") != DENSE_SOURCE_HEAD \
                or provenance.get("run_id") != run_id or provenance.get("run_attempt") != run_attempt \
                or provenance.get("job_id") != job_id or provenance.get("artifact_id") != artifact_id \
                or provenance.get("status") != "completed" or provenance.get("conclusion") != "success" \
                or provenance.get("job_key") != "phase07-confirmation":
            raise ValueError("dense ordinal API provenance")
        builds = record.get("builds")
        if not isinstance(builds, list) or len(builds) != 3:
            raise ValueError("dense ordinal build cardinality")
        expected_builds = ((16, [100]), (20, [300]), (32, [300]))
        ids: dict[int, str] = {}
        for build, (m, query_ef) in zip(builds, expected_builds, strict=True):
            if not isinstance(build, dict) or build.get("m") != m \
                    or build.get("ef_construction") != 300 or build.get("query_ef") != query_ef \
                    or not isinstance(build.get("build_id"), str) or not HEX64.fullmatch(build["build_id"]):
                raise ValueError("dense ordinal D-25 build policy")
            ids[m] = build["build_id"]
            if build["build_id"] in seen_builds:
                raise ValueError("dense ordinal builds must be pairwise disjoint")
            seen_builds.add(build["build_id"])
        measured_builds = measurements.get("builds")
        if not isinstance(measured_builds, list) or len(measured_builds) != 3 \
                or measurements.get("build_count") != 3 \
                or measurements.get("baseline_build_id") != ids[16] \
                or measurements.get("candidate_build_ids") != {"20": ids[20], "32": ids[32]}:
            raise ValueError("dense ordinal measurement/build binding")
        for measured, compact in zip(measured_builds, builds, strict=True):
            card = measured.get("build", measured) if isinstance(measured, dict) else None
            query_ef = measured.get("query_ef") if isinstance(measured, dict) else None
            if query_ef is None and isinstance(measured, dict) and isinstance(measured.get("queries"), list):
                query_ef = [row.get("query_ef") for row in measured["queries"] if isinstance(row, dict)]
            if not isinstance(measured, dict) or measured.get("build_id") != compact["build_id"] \
                    or not isinstance(card, dict) or card.get("m") != compact["m"] \
                    or card.get("ef_construction") != 300 or query_ef != compact["query_ef"]:
                raise ValueError("dense measured build policy")
    physical_ids = [
        (row.get("slot", {}).get("ordinal"), row.get("run_id"), row.get("run_attempt"), row.get("job_id"))
        if isinstance(row, dict) else None
        for row in physical
    ]
    eligible_ids = [
        (row["slot"]["ordinal"], row["run_id"], row["run_attempt"], row["job_id"])
        for row in records
    ]
    if physical_ids != eligible_ids:
        raise ValueError("dense physical/eligible ordinal lineage")
    if _hybrid_ordinal_identities(records) != [dict(item) for item in HYBRID_DENSE_ORDINAL_IDENTITIES]:
        raise ValueError("dense immutable build identity")
    return records


def _hybrid_ordinal_identities(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "ordinal": record["slot"]["ordinal"], "run_id": record["run_id"],
            "run_attempt": record["run_attempt"], "job_id": record["job_id"],
            "artifact_id": record["validated_provenance"]["artifact_id"],
            "build_ids": [build["build_id"] for build in record["builds"]],
        }
        for record in records
    ]


def build_hybrid_plan(dense_ledger: Path, *, post_implementation_head: str) -> dict[str, Any]:
    """Derive the only D-25 hybrid inputs from the sealed dense evidence."""
    if not SHA.fullmatch(post_implementation_head):
        raise ValueError("hybrid requires exact implementation head")
    records = _validate_dense_ledger(dense_ledger)
    identities = _hybrid_ordinal_identities(records)
    request = _sealed({
        "schema_version": 1, "campaign_stage": "hybrid",
        "dense_ledger_sha256": DENSE_LEDGER_DIGEST, "dense_source_head": DENSE_SOURCE_HEAD,
        "hybrid_implementation_head": post_implementation_head,
        "dense_ordinal_identities": identities, "baseline": dict(HYBRID_BASELINE),
        "candidates": [dict(row) for row in HYBRID_CANDIDATES], "scale": 30000,
        "query_count": 105, "authorization": "none", "retention_days": 90,
        "replacement_for_run_id": None,
    })
    inputs = [
        _sealed({
            "schema_version": 1, "campaign_stage": "hybrid",
            "hybrid_request_sha256": request["record_self_sha256"],
            "dense_source_head": DENSE_SOURCE_HEAD,
            "hybrid_implementation_head": post_implementation_head,
            "role": role, "config": dict(config),
            "scale": 30000, "query_count": 105, "authorization": "none",
            "retention_days": 90, "replacement_for_run_id": None,
            "dispatch_identity": "phase07-hybrid/baseline" if role == "baseline" else f"phase07-hybrid/m{config['m']}",
        })
        for role, config in HYBRID_ROLE_CONFIGS
    ]
    return _sealed({"schema_version": 1, "hybrid_request": request, "workflow_inputs": inputs})


def validate_hybrid_plan(plan: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(plan, dict) or set(plan) != {"schema_version", "hybrid_request", "workflow_inputs", "record_self_sha256"} \
            or plan.get("schema_version") != 1 or plan.get("record_self_sha256") != canonical_digest(plan):
        raise ValueError("sealed hybrid plan")
    request, inputs = plan["hybrid_request"], plan["workflow_inputs"]
    request_fields = {
        "schema_version", "campaign_stage", "dense_ledger_sha256", "dense_source_head",
        "hybrid_implementation_head", "dense_ordinal_identities", "baseline", "candidates",
        "scale", "query_count", "authorization", "retention_days", "replacement_for_run_id",
        "record_self_sha256",
    }
    if not isinstance(request, dict) or set(request) != request_fields \
            or request.get("record_self_sha256") != canonical_digest(request) \
            or request.get("campaign_stage") != "hybrid" \
            or request.get("dense_ledger_sha256") != DENSE_LEDGER_DIGEST \
            or request.get("dense_source_head") != DENSE_SOURCE_HEAD \
            or not SHA.fullmatch(request.get("hybrid_implementation_head", "")) \
            or request.get("baseline") != HYBRID_BASELINE \
            or request.get("candidates") != list(HYBRID_CANDIDATES) \
            or request.get("scale") != 30000 or request.get("query_count") != 105 \
            or request.get("authorization") != "none" or request.get("retention_days") != 90 \
            or request.get("replacement_for_run_id") is not None:
        raise ValueError("hybrid request authority")
    identities = request.get("dense_ordinal_identities")
    if not isinstance(identities, list) or len(identities) != 3:
        raise ValueError("hybrid dense ordinal identity")
    expected = ((1, 32801985769, 1, 97664517767, 9546915747), (2, 32802007002, 1, 97664580321, 9546916208), (3, 32802027355, 1, 97664640212, 9546924769))
    if [tuple(row.get(name) for name in ("ordinal", "run_id", "run_attempt", "job_id", "artifact_id")) if isinstance(row, dict) else () for row in identities] != list(expected) \
            or any(not isinstance(row.get("build_ids"), list) or len(row["build_ids"]) != 3 for row in identities):
        raise ValueError("hybrid dense ordinal identity")
    if not isinstance(inputs, list) or len(inputs) != 3:
        raise ValueError("exactly one baseline and two hybrid candidate bundles")
    for record, (role, config) in zip(inputs, HYBRID_ROLE_CONFIGS, strict=True):
        if not isinstance(record, dict) or set(record) != HYBRID_WORKFLOW_INPUT_FIELDS \
                or record.get("record_self_sha256") != canonical_digest(record) \
                or record.get("campaign_stage") != "hybrid" \
                or record.get("hybrid_request_sha256") != request["record_self_sha256"] \
                or record.get("dense_source_head") != DENSE_SOURCE_HEAD \
                or record.get("hybrid_implementation_head") != request["hybrid_implementation_head"] \
                or record.get("role") != role or record.get("config") != config \
                or record.get("scale") != 30000 or record.get("query_count") != 105 \
                or record.get("authorization") != "none" or record.get("retention_days") != 90 \
                or record.get("replacement_for_run_id") is not None \
                or record.get("dispatch_identity") != ("phase07-hybrid/baseline" if role == "baseline" else f"phase07-hybrid/m{config['m']}"):
            raise ValueError("sealed hybrid workflow input")
    return plan


def validate_hybrid_workflow_input(record: dict[str, Any], *, expected_head: str) -> dict[str, Any]:
    if not SHA.fullmatch(expected_head) or not isinstance(record, dict) \
            or set(record) != HYBRID_WORKFLOW_INPUT_FIELDS \
            or record.get("record_self_sha256") != canonical_digest(record) \
            or record.get("hybrid_implementation_head") != expected_head:
        raise ValueError("hybrid generated workflow input")
    # The sealed fixed member still rejects every retired or user-selected
    # configuration before it can reach an expensive model/index path.
    role, config = record.get("role"), record.get("config")
    if (role, config) not in HYBRID_ROLE_CONFIGS + FROZEN_HYBRID_ROLE_CONFIGS \
            or record.get("dense_source_head") != DENSE_SOURCE_HEAD \
            or record.get("scale") != 30000 or record.get("query_count") != 105 \
            or record.get("authorization") != "none" or record.get("retention_days") != 90 \
            or record.get("replacement_for_run_id") is not None \
            or record.get("dispatch_identity") != ("phase07-hybrid/baseline" if role == "baseline" else f"phase07-hybrid/m{config['m']}") \
            or not isinstance(record.get("hybrid_request_sha256"), str) \
            or not HEX64.fullmatch(record["hybrid_request_sha256"]):
        raise ValueError("hybrid candidate authority")
    return record


def _validate_hybrid_request(request: object, *, expected_head: str) -> dict[str, Any]:
    # Direct-file and package execution can coexist in Python's module cache.
    # They are the same checked-in source, but tests intentionally patch the
    # package module's sealed fixture digest.  Resolve that one source value
    # rather than silently validating two divergent copies of this file.
    package_gate = sys.modules.get("eval.phase07_operator_gate")
    expected_dense_digest = getattr(package_gate, "DENSE_LEDGER_DIGEST", DENSE_LEDGER_DIGEST)
    fields = {
        "schema_version", "campaign_stage", "dense_ledger_sha256", "dense_source_head",
        "hybrid_implementation_head", "dense_ordinal_identities", "baseline", "candidates",
        "scale", "query_count", "authorization", "retention_days", "replacement_for_run_id",
        "record_self_sha256",
    }
    if not isinstance(request, dict) or set(request) != fields \
            or request.get("record_self_sha256") != canonical_digest(request) \
            or request.get("campaign_stage") != "hybrid" \
            or request.get("dense_ledger_sha256") != expected_dense_digest \
            or request.get("dense_source_head") != DENSE_SOURCE_HEAD \
            or request.get("hybrid_implementation_head") != expected_head \
            or request.get("dense_ordinal_identities") != [dict(row) for row in HYBRID_DENSE_ORDINAL_IDENTITIES] \
            or request.get("baseline") != HYBRID_BASELINE \
            or request.get("candidates") != list(HYBRID_CANDIDATES) \
            or request.get("scale") != 30000 or request.get("query_count") != 105 \
            or request.get("authorization") != "none" or request.get("retention_days") != 90 \
            or request.get("replacement_for_run_id") is not None:
        raise ValueError("exact sealed hybrid request authority")
    return request


def validate_hybrid_dispatch_bundle(bundle: dict[str, Any], *, expected_head: str) -> dict[str, Any]:
    """Fail closed before any 30k build unless this is one exact generated member."""
    frozen_fields = {"schema_version", "hybrid_request", "workflow_input", "frozen_prepare", "replacement_for_run_id", "record_self_sha256"}
    generic_fields = {"schema_version", "hybrid_request", "workflow_input", "replacement_for_run_id", "record_self_sha256"}
    fields = set(bundle) if isinstance(bundle, dict) else set()
    if not SHA.fullmatch(expected_head) or _git("rev-parse", "HEAD") != expected_head \
            or not isinstance(bundle, dict) or (fields != frozen_fields and fields != generic_fields) \
            or bundle.get("schema_version") != 1 or bundle.get("record_self_sha256") != canonical_digest(bundle) \
            or bundle.get("replacement_for_run_id") is not None:
        raise ValueError("typed sealed hybrid dispatch bundle")
    request = _validate_hybrid_request(bundle.get("hybrid_request"), expected_head=expected_head)
    member = bundle.get("workflow_input")
    validate_hybrid_workflow_input(member, expected_head=expected_head)
    if member.get("hybrid_request_sha256") != request["record_self_sha256"]:
        raise ValueError("hybrid dispatch membership")
    if fields == frozen_fields:
        from eval.phase07_frozen_base import validate_frozen_prepare_identity_shape
        frozen_prepare = validate_frozen_prepare_identity_shape(
            bundle.get("frozen_prepare"), expected_repository="allenwoo713/obsidian_wiki_skill",
            expected_head=expected_head,
        )
        if (member.get("role"), member.get("config")) not in FROZEN_HYBRID_ROLE_CONFIGS \
                or frozen_prepare["head_sha"] != member["hybrid_implementation_head"]:
            raise ValueError("frozen hybrid dispatch membership")
    elif (member.get("role"), member.get("config")) not in HYBRID_ROLE_CONFIGS:
        # Historic generic PR evidence is intentionally kept readable.  It
        # cannot select the frozen workflow because that path requires the
        # additional frozen_prepare field above.
        raise ValueError("generic hybrid dispatch membership")
    return member


def validate_hybrid_workflow_inputs_dir(path: Path, *, expected_head: str) -> list[dict[str, Any]]:
    """Generated workflow input directories are an exact three-member allowlist."""
    expected_names = {"hybrid-baseline.json", "hybrid-m20.json", "hybrid-m32.json"}
    if path.is_symlink() or not path.is_dir() or {item.name for item in path.iterdir()} != expected_names:
        raise ValueError("strict hybrid generated input allowlist")
    records = []
    expected_members = (("hybrid-baseline.json", "baseline", 16), ("hybrid-m20.json", "m20", 20), ("hybrid-m32.json", "m32", 32))
    for name, role, m in expected_members:
        item = path / name
        if item.is_symlink() or not item.is_file():
            raise ValueError("strict hybrid generated input member")
        try:
            bundle = json.loads(item.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError("invalid hybrid generated input") from exc
        member = validate_hybrid_dispatch_bundle(bundle, expected_head=expected_head)
        if member["role"] != role or member["config"]["m"] != m:
            raise ValueError("hybrid generated input membership")
        records.append(member)
    return records


def build_hybrid_preflight(bundle: dict[str, Any], *, expected_head: str) -> dict[str, Any]:
    """Produce the typed worktree request only after exact dispatch validation."""
    member = validate_hybrid_dispatch_bundle(bundle, expected_head=expected_head)
    root = Path(_git("rev-parse", "--show-toplevel")).resolve()
    return {
        "repository": "allenwoo713/obsidian_wiki_skill", "branch": _git("branch", "--show-current"),
        "worktree_root": str(root), "head_sha": expected_head, "allowed_dirty_paths": [],
        "workflow_name": "eval", "campaign_stage": "hybrid",
        "continuation_binding": bundle["record_self_sha256"], "require_upstream_head": True,
    }


def _validate_hybrid_allocation(value: object) -> dict[str, Any]:
    required = {"run_id", "run_attempt", "job_id", "job_key", "job_allocation_nonce"}
    if not isinstance(value, dict) or set(value) != required \
            or not all(not isinstance(value[name], bool) and isinstance(value[name], int) and value[name] > 0 for name in ("run_id", "run_attempt", "job_id")) \
            or value.get("run_attempt") != 1 \
            or value.get("job_key") != "phase07-hybrid" \
            or not isinstance(value.get("job_allocation_nonce"), str) or len(value["job_allocation_nonce"]) != 32 \
            or any(char not in "0123456789abcdef" for char in value["job_allocation_nonce"]):
        raise ValueError("hybrid allocation identity")
    return value


# Public on purpose: the hosted workflow and post-download collector must use
# the same narrow allocation schema.  Keeping a private implementation lets
# the capability boundary below remain explicit without creating a second,
# weaker parser for the CLI route.
def validate_hybrid_allocation(value: object) -> dict[str, Any]:
    return _validate_hybrid_allocation(value)


_HYBRID_CAPABILITY_ISSUER = object()


class _HybridExecutionCapability:
    """Opaque, process-local authority for one already-validated dispatch.

    It is intentionally neither JSON serializable nor a mapping.  The public
    boundary below is the only minting site; the evaluator consumes it through
    the companion private verifier instead of accepting a workflow member.
    """

    __slots__ = ("_issuer", "_bundle", "_locked_execution", "_allocation", "_dispatch_sha256")

    def __init__(self, *, issuer: object, bundle: dict[str, Any], locked_execution: dict[str, Any],
                 allocation: dict[str, Any], dispatch_sha256: str) -> None:
        self._issuer = issuer
        self._bundle = bundle
        self._locked_execution = locked_execution
        self._allocation = allocation
        self._dispatch_sha256 = dispatch_sha256


def _mint_hybrid_execution_capability(*, bundle: dict[str, Any], locked_execution: dict[str, Any],
                                      allocation: dict[str, Any], dispatch_sha256: str) -> _HybridExecutionCapability:
    # Detach the capability's facts from caller-owned mutable JSON before it
    # crosses the production runner boundary.
    copied = json.loads(json.dumps({"bundle": bundle, "execution": locked_execution, "allocation": allocation},
                                   sort_keys=True))
    return _HybridExecutionCapability(
        issuer=_HYBRID_CAPABILITY_ISSUER, bundle=copied["bundle"], locked_execution=copied["execution"],
        allocation=copied["allocation"], dispatch_sha256=dispatch_sha256,
    )


def _consume_hybrid_execution_capability(value: object) -> dict[str, Any]:
    """Return detached role authority only for a token minted by this module."""
    if type(value) is not _HybridExecutionCapability or value._issuer is not _HYBRID_CAPABILITY_ISSUER:
        raise ValueError("unminted hybrid execution capability")
    bundle, execution, allocation = value._bundle, value._locked_execution, value._allocation
    if not isinstance(bundle, dict) or not isinstance(execution, dict) or not isinstance(allocation, dict) \
            or not isinstance(value._dispatch_sha256, str) or not HEX64.fullmatch(value._dispatch_sha256):
        raise ValueError("invalid hybrid execution capability")
    # Revalidate the preserved facts: a private type alone must not become an
    # alternate authority path.
    head = _git("rev-parse", "HEAD")
    if bundle.get("record_self_sha256") != value._dispatch_sha256:
        raise ValueError("hybrid capability dispatch digest")
    member = validate_hybrid_dispatch_bundle(bundle, expected_head=head)
    from eval.phase07_ann_campaign import validate_confirmation_execution
    validate_confirmation_execution(execution)
    if execution.get("head_sha") != head:
        raise ValueError("hybrid capability locked execution head")
    _validate_hybrid_allocation(allocation)
    if "frozen_prepare" not in bundle:
        # Ordinary PR paths retain their established member-only capability.
        return json.loads(json.dumps(member, sort_keys=True))
    # Keep the collector-bound prepare fact with the opaque authority.  The
    # runner cannot silently lose it and replace the downloaded corpus with a
    # same-shaped local tree.
    return json.loads(json.dumps({
        "member": member, "frozen_prepare": bundle["frozen_prepare"],
    }, sort_keys=True))


def execute_hybrid_dispatch(*, bundle: dict[str, Any], locked_execution: dict[str, Any], allocation: dict[str, Any],
                            work_dir: Path, runner: Any | None = None,
                            frozen_dir: Path | None = None) -> dict[str, Any]:
    """The single build boundary; validate all authority before invoking the runner."""
    head = _git("rev-parse", "HEAD")
    member = validate_hybrid_dispatch_bundle(bundle, expected_head=head)
    from eval.phase07_ann_campaign import validate_confirmation_execution
    validate_confirmation_execution(locked_execution)
    if locked_execution.get("head_sha") != head:
        raise ValueError("hybrid locked execution head")
    _validate_hybrid_allocation(allocation)
    capability = _mint_hybrid_execution_capability(
        bundle=bundle, locked_execution=locked_execution, allocation=allocation,
        dispatch_sha256=bundle["record_self_sha256"],
    )
    production_runner = runner is None
    if production_runner:
        from eval.run_eval import _run_phase07_hybrid_campaign_with_capability
        runner = _run_phase07_hybrid_campaign_with_capability
    runner_kwargs = {"capability": capability, "work_dir": work_dir / "campaign" if production_runner else work_dir}
    if frozen_dir is not None:
        # Only the production runner accepts the trusted downloaded root.  A
        # custom test runner must not gain an unvalidated filesystem authority.
        if not production_runner:
            raise ValueError("frozen source is only valid for the production hybrid runner")
        runner_kwargs["frozen_dir"] = Path(frozen_dir)
    result = runner(**runner_kwargs)
    if not isinstance(result, dict) or result.get("authorization") != "none":
        raise ValueError("hybrid runner produced authorizing evidence")
    if production_runner:
        from eval.phase07_ann_campaign import export_hybrid_artifact_tree
        export_hybrid_artifact_tree(campaign_result=result, dispatch_bundle=bundle,
                                    locked_execution=locked_execution, allocation=allocation, output_dir=work_dir / "raw")
    return result


_HYBRID_GATE_QUALITY = (
    "functional_final_retrieval_ann_overlap_at_10", "page_recall_at_5", "evidence_recall_at_10",
    "exact_lookup_hit_at_3", "mrr_at_10",
)
_HYBRID_GATE_ZERO_TOLERANCE = (
    "citation_violation_count", "context_overflow_count", "budget_violation_count", "graph_unsupported_count",
)


def committed_hybrid_baseline() -> tuple[dict[str, float | int], str]:
    """Read the one committed hybrid-quality floor set, including its bytes hash."""
    path = _REPOSITORY_ROOT / "eval" / "baselines.json"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        quality = document["quality"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError("committed hybrid baseline unavailable") from exc
    floors = {
        "page_recall_at_5": quality.get("page_recall_at_5"),
        "evidence_recall_at_10": quality.get("evidence_recall_at_10"),
        "exact_lookup_hit_at_3": quality.get("exact_lookup_hit_at_3"),
        "mrr_at_10": quality.get("mrr_at_10"),
        **{key: 0 for key in _HYBRID_GATE_ZERO_TOLERANCE},
    }
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value))
           for value in floors.values()):
        raise ValueError("committed hybrid baseline invalid")
    return floors, hashlib.sha256(path.read_bytes()).hexdigest()


def recompute_hybrid_gate_verdicts(*, original_absolute: dict[str, Any], expanded_paired: dict[str, Any],
                                   committed_baseline: dict[str, float | int] | None = None,
                                   baselines_sha256: str | None = None) -> dict[str, Any]:
    """Recompute non-authorizing absolute and paired gates from complete metrics.

    The original campaign is an absolute check against the committed quality
    floors; the expanded campaign is a paired non-regression check.  Both
    sides are validated so a broken observed baseline cannot bless a candidate.
    """
    expected_floors, expected_digest = committed_hybrid_baseline()
    if committed_baseline is None:
        committed_baseline = expected_floors
    if baselines_sha256 is None:
        baselines_sha256 = expected_digest
    if baselines_sha256 != expected_digest or committed_baseline != expected_floors:
        raise ValueError("committed hybrid baseline identity")

    required = set(_HYBRID_GATE_QUALITY) | set(_HYBRID_GATE_ZERO_TOLERANCE)
    def valid_metric(value: object) -> bool:
        return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value))

    def gate(name: str, aggregate: object, *, absolute: bool) -> dict[str, Any]:
        if not isinstance(aggregate, dict) or set(aggregate) != {"baseline", "candidate"} \
                or not isinstance(aggregate["baseline"], dict) or not isinstance(aggregate["candidate"], dict):
            raise ValueError("complete hybrid aggregate metrics")
        baseline, candidate = aggregate["baseline"], aggregate["candidate"]
        if set(baseline) != required or set(candidate) != required:
            raise ValueError("complete hybrid aggregate metrics")
        if not all(valid_metric(value) for value in (*baseline.values(), *candidate.values())):
            raise ValueError("finite hybrid aggregate metrics")
        # Counters represent observed events, never rates; reject floats and
        # negative values before deciding whether a candidate is acceptable.
        if any(isinstance(baseline[key], bool) or isinstance(candidate[key], bool)
               or not isinstance(baseline[key], int) or not isinstance(candidate[key], int)
               or baseline[key] < 0 or candidate[key] < 0 for key in _HYBRID_GATE_ZERO_TOLERANCE):
            raise ValueError("hybrid zero-tolerance metrics")
        rejected = False
        if absolute:
            # Functional retrieval deliberately has no ANN-floor alias.  It is
            # required to be real and finite above, but is not compared to the
            # dense ANN benchmark.
            for key in ("page_recall_at_5", "evidence_recall_at_10", "exact_lookup_hit_at_3", "mrr_at_10"):
                rejected |= baseline[key] < committed_baseline[key] - 0.02
                rejected |= candidate[key] < committed_baseline[key] - 0.02
                rejected |= candidate[key] < baseline[key] - 0.02
            rejected |= any(baseline[key] != 0 or candidate[key] != 0 for key in _HYBRID_GATE_ZERO_TOLERANCE)
        else:
            rejected |= any(candidate[key] < baseline[key] - 0.02 for key in _HYBRID_GATE_QUALITY)
            # A non-zero observed baseline is an instrumentation failure, not
            # a tolerance that a candidate may inherit.
            rejected |= any(baseline[key] != 0 or candidate[key] != 0 for key in _HYBRID_GATE_ZERO_TOLERANCE)
        return {"stratum": name, "baseline_metrics": baseline, "candidate_metrics": candidate,
                "candidate_verdict": "rejected-candidate" if rejected else "numeric-success", "authorization": "none"}
    original_gate = gate("original_absolute", original_absolute, absolute=True)
    paired_gate = gate("paired_30k", expanded_paired, absolute=False)
    candidate_verdict = "rejected-candidate" if "rejected-candidate" in {original_gate["candidate_verdict"], paired_gate["candidate_verdict"]} else "numeric-success"
    return {"original_absolute_gate": original_gate, "paired_30k_non_regression_gate": paired_gate,
            "candidate_verdict": candidate_verdict, "authorization": "none", "write_graph_artifact": True}


def reject_retired_hybrid_authority(value: object) -> None:
    """Retired modes are values in authority fields, never arbitrary text substrings."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in {"mode", "campaign_mode", "authority_mode"} and item in {"representative_ann", "continuation", "stage2", "flat", "refinement"}:
                raise ValueError("retired hybrid authority")
            reject_retired_hybrid_authority(item)
    elif isinstance(value, list):
        for item in value:
            reject_retired_hybrid_authority(item)


def validate_hybrid_artifact_tree(root: Path) -> dict[str, Any]:
    """Operator-facing alias for the shared strict raw-tree validator."""
    from eval.phase07_ann_campaign import validate_hybrid_artifact_tree as validate_tree
    return validate_tree(root)


def allocate_confirmation_job(client: Any, *, repository: str, run_id: int, run_attempt: int,
                              job_key: str, token: str) -> dict[str, Any]:
    """Bind the hosted allocation before any build without retaining the token."""
    if not repository or not isinstance(run_id, int) or run_id <= 0 or not isinstance(run_attempt, int) or run_attempt <= 0 or not job_key or not token:
        raise ValueError("invalid attempt-scoped allocation identity")
    try:
        response = client.get_json(f"/repos/{repository}/actions/runs/{run_id}/attempts/{run_attempt}/jobs", token)
        jobs = response.get("jobs") if isinstance(response, dict) else None
        expected_name = JOB_DISPLAY_NAMES.get(job_key, job_key)
        matches = [job for job in jobs if isinstance(job, dict) and job.get("name") == expected_name and job.get("run_id") == run_id and job.get("run_attempt") == run_attempt] if isinstance(jobs, list) else []
        if len(matches) != 1 or not isinstance(matches[0].get("id"), int) or matches[0]["id"] <= 0:
            raise ValueError("attempt-scoped job must match exactly once")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError("attempt-scoped job lookup failed") from exc
    try:
        nonce = secrets.token_hex(16)
    except Exception as exc:
        raise ValueError("allocation nonce generation failed") from exc
    if not isinstance(nonce, str) or len(nonce) != 32:
        raise ValueError("allocation nonce generation failed")
    return {"run_id": run_id, "run_attempt": run_attempt, "job_id": matches[0]["id"], "job_key": job_key, "job_allocation_nonce": nonce}


_HYBRID_DOWNLOAD_FIELDS = frozenset({
    "run_id", "run_attempt", "job_id", "artifact_id", "role", "config", "bundle_sha256",
    "archive", "extracted_dir",
})
_HYBRID_COLLECTION_FIELDS = frozenset({
    "schema_version", "campaign_stage", "repository", "head_sha", "hybrid_request",
    "hybrid_request_sha256", "downloads", "record_self_sha256",
})


def seal_hybrid_allocation(*, workflow_inputs: dict[str, Any], output: Path, repository: str,
                           run_id: int, run_attempt: int, job_key: str, head_sha: str,
                           token: str, client: Any | None = None) -> int:
    """Seal the exact `phase07-hybrid` Jobs-API allocation before a 30k build."""
    record: dict[str, Any] = {"schema_version": 1, "campaign_stage": "hybrid"}
    try:
        if run_attempt != 1:
            raise ValueError("hybrid evidence permits only first-attempt runs")
        validate_hybrid_dispatch_bundle(workflow_inputs, expected_head=head_sha)
        allocation = _validate_hybrid_allocation(allocate_confirmation_job(
            client or GitHubActionsClient(), repository=repository, run_id=run_id,
            run_attempt=run_attempt, job_key=job_key, token=token,
        ))
        record.update(status="success", allocation=allocation)
        _write_ledger(output, record)
        return 0
    except Exception:
        # No exception text: API exceptions can include transport details and
        # must never become a token-adjacent hosted artifact.
        record.update(status="reject-evidence", failure_class="attempt_scoped_hybrid_job_or_nonce")
        _write_ledger(output, record)
        return 1


def _parse_utc_timestamp(value: object, *, label: str) -> dt.datetime:
    if not isinstance(value, str):
        raise ValueError(f"hybrid {label} timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"hybrid {label} timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"hybrid {label} timestamp timezone")
    return parsed.astimezone(dt.timezone.utc)


def build_hybrid_collection_request(*, hybrid_request: dict[str, Any], downloads: dict[str, Any]) -> dict[str, Any]:
    """Bind exactly one baseline plus m20/m32 archives to their sealed request.

    This stage deliberately does not trust a local raw packet as API evidence;
    it only derives the candidate identities needed to request the exact API
    records.  Tree/digest verification happens in `collect_hybrid_provenance`.
    """
    _reject_secrets(hybrid_request); _reject_secrets(downloads)
    head = hybrid_request.get("hybrid_implementation_head") if isinstance(hybrid_request, dict) else ""
    request = _validate_hybrid_request(hybrid_request, expected_head=head)
    if not isinstance(downloads, dict) or set(downloads) != {"schema_version", "downloads", "record_self_sha256"} \
            or downloads.get("schema_version") != 1 \
            or downloads.get("record_self_sha256") != canonical_digest(downloads) \
            or not isinstance(downloads.get("downloads"), list):
        raise ValueError("strict hybrid downloads manifest")
    if len(downloads["downloads"]) != 3:
        raise ValueError("hybrid collection requires exactly three downloads")

    selected: list[dict[str, Any]] = []
    seen_runs: set[tuple[int, int]] = set()
    seen_paths: set[tuple[Path, Path]] = set()
    seen_roles: set[tuple[str, int]] = set()
    for value in downloads["downloads"]:
        if not isinstance(value, dict) or set(value) != _HYBRID_DOWNLOAD_FIELDS:
            raise ValueError("strict hybrid download record")
        run_id, run_attempt = value["run_id"], value["run_attempt"]
        if isinstance(run_id, bool) or not isinstance(run_id, int) or run_id <= 0 \
                or isinstance(run_attempt, bool) or not isinstance(run_attempt, int) or run_attempt != 1:
            raise ValueError("hybrid download run identity")
        if any(isinstance(value[name], bool) or not isinstance(value[name], int) or value[name] <= 0
               for name in ("job_id", "artifact_id")) \
                or (value.get("role"), value.get("config")) not in HYBRID_ROLE_CONFIGS + FROZEN_HYBRID_ROLE_CONFIGS \
                or not isinstance(value.get("bundle_sha256"), str) or not HEX64.fullmatch(value["bundle_sha256"]):
            raise ValueError("hybrid download job/artifact/candidate identity")
        archive, extracted = Path(value["archive"]), Path(value["extracted_dir"])
        identity, paths = (run_id, run_attempt), (archive.resolve(), extracted.resolve())
        if identity in seen_runs or paths in seen_paths or archive.is_symlink() or not archive.is_file() \
                or extracted.is_symlink() or not extracted.is_dir():
            raise ValueError("duplicate or unavailable hybrid download")
        # Candidate identity may be read only from the sealed dispatch member;
        # later tree validation proves this leaf was not substituted.
        try:
            bundle = _read_object(extracted / "dispatch-bundle.json")
            member = bundle["workflow_input"]
            role, config, bundle_sha256 = member["role"], member["config"], member["record_self_sha256"]
        except (ValueError, KeyError, TypeError):
            raise ValueError("hybrid downloaded dispatch identity") from None
        role_identity = (role, config.get("m") if isinstance(config, dict) else -1)
        if role != value["role"] or config != value["config"] or bundle_sha256 != value["bundle_sha256"] \
                or (role, config) not in HYBRID_ROLE_CONFIGS + FROZEN_HYBRID_ROLE_CONFIGS or role_identity in seen_roles:
            raise ValueError("hybrid role cardinality")
        seen_runs.add(identity); seen_paths.add(paths); seen_roles.add(role_identity)
        selected.append({**value, "role": role, "config": dict(config)})
    generic_roles = {("baseline", 16), ("candidate", 20), ("candidate", 32)}
    frozen_roles = {("baseline", 16), ("m20", 20), ("m32", 32)}
    if seen_roles != generic_roles and seen_roles != frozen_roles:
        raise ValueError("hybrid role set")
    return _sealed({
        "schema_version": 1, "campaign_stage": "hybrid", "repository": "allenwoo713/obsidian_wiki_skill",
        "head_sha": head, "hybrid_request": request, "hybrid_request_sha256": request["record_self_sha256"],
        "downloads": sorted(selected, key=lambda row: (row["role"] != "baseline", row["config"]["m"])),
    })


def _validate_hybrid_collection_request(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _HYBRID_COLLECTION_FIELDS \
            or value.get("schema_version") != 1 or value.get("campaign_stage") != "hybrid" \
            or value.get("record_self_sha256") != canonical_digest(value) \
            or value.get("repository") != "allenwoo713/obsidian_wiki_skill":
        raise ValueError("sealed hybrid collection request")
    request = _validate_hybrid_request(value.get("hybrid_request"), expected_head=value.get("head_sha", ""))
    if value.get("hybrid_request_sha256") != request["record_self_sha256"] \
            or value.get("head_sha") != request["hybrid_implementation_head"] \
            or not isinstance(value.get("downloads"), list) or len(value["downloads"]) != 3:
        raise ValueError("hybrid collection authority")
    generic_expected = {("baseline", 16): HYBRID_BASELINE, ("candidate", 20): HYBRID_CANDIDATES[0], ("candidate", 32): HYBRID_CANDIDATES[1]}
    frozen_expected = {("baseline", 16): HYBRID_BASELINE, ("m20", 20): HYBRID_CANDIDATES[0], ("m32", 32): HYBRID_CANDIDATES[1]}
    expected = frozen_expected if any(row.get("role") in {"m20", "m32"} for row in value["downloads"] if isinstance(row, dict)) else generic_expected
    found: set[tuple[str, int]] = set()
    for row in value["downloads"]:
        if not isinstance(row, dict) or set(row) != _HYBRID_DOWNLOAD_FIELDS \
                or (row.get("role"), row.get("config")) not in HYBRID_ROLE_CONFIGS + FROZEN_HYBRID_ROLE_CONFIGS \
                or not isinstance(row.get("bundle_sha256"), str) or not HEX64.fullmatch(row["bundle_sha256"]):
            raise ValueError("hybrid collection download shape")
        if any(isinstance(row[key], bool) or not isinstance(row[key], int) or row[key] <= 0
               for key in ("run_id", "run_attempt", "job_id", "artifact_id")):
            raise ValueError("hybrid collection hosted identity")
        if row["run_attempt"] != 1:
            raise ValueError("hybrid collection requires independent first-attempt runs")
        identity = (row["role"], row["config"]["m"])
        if identity in found or row["config"] != expected[identity]:
            raise ValueError("hybrid collection role binding")
        found.add(identity)
    if found != set(expected):
        raise ValueError("hybrid collection role cardinality")
    return value


def collect_hybrid_provenance(*, request_file: Path, output: Path, provenance_dir: Path,
                              token: str, client: Any | None = None) -> int:
    """API-bind the exact three role packets; non-success invalidates the batch."""
    collection = _validate_hybrid_collection_request(_read_object(request_file))
    _reject_secrets(collection)
    if not token:
        raise ValueError("GitHub actions read token unavailable")
    if output.exists() or provenance_dir.is_symlink() or (provenance_dir.exists() and any(provenance_dir.iterdir())):
        raise ValueError("hybrid provenance output must be new and empty")
    github = client or GitHubActionsClient()
    evidence: list[dict[str, str]] = []
    seen_ids: set[tuple[int, int, int, int]] = set()
    for row in collection["downloads"]:
        run_id, run_attempt = row["run_id"], row["run_attempt"]
        if run_attempt != 1:
            raise ValueError("hybrid evidence requires three independent first-attempt runs")
        archive, extracted = Path(row["archive"]), Path(row["extracted_dir"])
        validated = validate_hybrid_artifact_tree(extracted)
        allocation = validated["result"].get("allocation", {}).get("allocation")
        if not isinstance(allocation, dict):
            raise ValueError("hybrid artifact allocation")
        allocation = _validate_hybrid_allocation(allocation)
        if allocation["run_id"] != run_id or allocation["run_attempt"] != run_attempt \
                or allocation["job_id"] != row["job_id"] \
                or validated["role"] != row["role"] or validated["config"] != row["config"] \
                or validated["workflow_input"].get("record_self_sha256") != row["bundle_sha256"]:
            raise ValueError("hybrid downloaded artifact binding")
        host = validated["result"].get("locked_execution", {}).get("host")
        if not isinstance(host, dict) or host.get("os") != "Linux" or host.get("architecture") != "X64" \
                or not isinstance(host.get("image"), str) or not host["image"]:
            raise ValueError("hybrid locked hosted runner identity")
        run_url = f"/repos/{collection['repository']}/actions/runs/{run_id}"
        try:
            run = github.get_json(run_url, token)
            jobs_payload = github.get_json(f"{run_url}/attempts/{run_attempt}/jobs", token)
            artifacts_payload = github.get_json(f"{run_url}/artifacts", token)
        except Exception as exc:
            raise ValueError("GitHub hybrid provenance lookup failed") from exc
        if not isinstance(run, dict) or run.get("id") != run_id or run.get("run_attempt") != run_attempt \
                or run.get("head_sha") != collection["head_sha"] or run.get("event") != "workflow_dispatch" \
                or run.get("status") != "completed" or run.get("conclusion") != "success" \
                or not isinstance(run.get("head_branch"), str) or run["head_branch"] in {"", "main", "master"}:
            raise ValueError("hybrid workflow-run API binding")
        jobs = jobs_payload.get("jobs") if isinstance(jobs_payload, dict) else None
        matches = [job for job in jobs if isinstance(job, dict) and job.get("id") == row["job_id"]
                   and job.get("run_id") == run_id and job.get("run_attempt") == run_attempt
                   and job.get("name") == JOB_DISPLAY_NAMES["phase07-hybrid"]
                   and job.get("status") == "completed" and job.get("conclusion") == "success"] if isinstance(jobs, list) else []
        if len(matches) != 1:
            raise ValueError("hybrid attempt-job API binding")
        job = matches[0]
        labels = job.get("labels")
        if not isinstance(job.get("runner_name"), str) or not job["runner_name"].startswith("GitHub Actions ") \
                or job.get("runner_group_name") != "GitHub Actions" \
                or not isinstance(labels, list) or "ubuntu-latest" not in labels \
                or ({"X64", "ARM64"} & set(labels) and "X64" not in labels):
            raise ValueError("hybrid runner API binding")
        if re.fullmatch(r"ubuntu[^ ]* [^ ]+", host["image"], flags=re.IGNORECASE) is None:
            raise ValueError("hybrid raw host image is not a measured GitHub Ubuntu image")
        artifacts = artifacts_payload.get("artifacts") if isinstance(artifacts_payload, dict) else None
        name = f"phase07-hybrid-{run_id}-{run_attempt}"
        matches = [artifact for artifact in artifacts if isinstance(artifact, dict) and artifact.get("name") == name] if isinstance(artifacts, list) else []
        if len(matches) != 1:
            raise ValueError("hybrid artifact API binding")
        artifact = matches[0]; local_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
        run_created = _parse_utc_timestamp(run.get("created_at"), label="run creation")
        artifact_created = _parse_utc_timestamp(artifact.get("created_at"), label="artifact creation")
        expires = _parse_utc_timestamp(artifact.get("expires_at"), label="artifact expiry")
        if artifact.get("id") != row["artifact_id"] or artifact.get("expired") is not False \
                or artifact.get("digest") != f"sha256:{local_sha}" or not isinstance(artifact.get("workflow_run"), dict) \
                or artifact["workflow_run"].get("id") != run_id or artifact["workflow_run"].get("head_sha") != collection["head_sha"] \
                or not run_created <= artifact_created < expires \
                or not dt.timedelta(days=89, hours=23, minutes=59, seconds=30) <= expires - artifact_created <= dt.timedelta(days=90, seconds=30) \
                or expires <= dt.datetime.now(dt.timezone.utc):
            raise ValueError("hybrid artifact API identity/digest/retention")
        identity = (run_id, run_attempt, row["job_id"], row["artifact_id"])
        if identity in seen_ids:
            raise ValueError("duplicate hybrid hosted identity")
        seen_ids.add(identity)
        provenance = {
            "run_id": run_id, "run_attempt": run_attempt, "job_id": row["job_id"],
            "job_key": allocation["job_key"], "job_name": JOB_DISPLAY_NAMES["phase07-hybrid"],
            "artifact_id": row["artifact_id"], "artifact_name": name, "status": run["status"],
            "conclusion": run["conclusion"], "head_branch": run["head_branch"],
            "head_sha": collection["head_sha"], "event": run["event"],
            "runner": {"name": job["runner_name"], "group": job["runner_group_name"], "labels": job["labels"],
                       "os": host["os"], "image": host["image"], "architecture": host["architecture"]},
            "run_created_at": run["created_at"], "artifact_created_at": artifact["created_at"],
            "artifact_expires_at": artifact["expires_at"],
            "api_archive_sha256": local_sha, "local_archive_sha256": local_sha,
            "role": row["role"], "config": row["config"], "bundle_sha256": row["bundle_sha256"],
            "archive": str(archive.resolve()), "extracted_dir": str(extracted.resolve()),
        }
        provenance_dir.mkdir(parents=True, exist_ok=True)
        provenance_path = provenance_dir / f"hybrid-{run_id}-{run_attempt}-provenance.json"
        _write_ledger(provenance_path, {"schema_version": 1, "evidence": [provenance]})
        evidence.append({
            "run_id": run_id, "run_attempt": run_attempt, "job_id": row["job_id"],
            "artifact_id": row["artifact_id"], "role": row["role"], "config": row["config"],
            "bundle_sha256": row["bundle_sha256"], "archive": str(archive.resolve()),
            "extracted_dir": str(extracted.resolve()), "provenance": str(provenance_path.resolve()),
        })
    _write_ledger(output, {"schema_version": 1, "evidence": evidence})
    return 0


class GitHubActionsClient:
    """Minimal fakeable stdlib client; its token is header-only and never serialized."""
    def get_json(self, url: str, token: str) -> dict[str, Any]:
        request = urllib.request.Request(
            f"https://api.github.com{url}", headers={"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                if response.status != 200:
                    raise ValueError("attempt-scoped job API status")
                value = json.loads(response.read().decode("utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("attempt-scoped job API unavailable") from exc
        if not isinstance(value, dict):
            raise ValueError("attempt-scoped job API JSON")
        return value


def seal_confirmation_allocation(*, workflow_inputs: dict[str, Any], output: Path, repository: str,
                                 run_id: int, run_attempt: int, job_key: str, head_sha: str, token: str,
                                 client: Any | None = None) -> int:
    """Seal lookup/entropy rejection before a workflow is permitted to build."""
    record: dict[str, Any] = {"schema_version": 1, "campaign_stage": "confirmation"}
    try:
        workflow_input = validate_confirmation_dispatch_bundle(workflow_inputs, expected_head=head_sha)
        record["workflow_inputs_sha256"] = workflow_input["record_self_sha256"]
        allocation = allocate_confirmation_job(client or GitHubActionsClient(), repository=repository, run_id=run_id,
                                                run_attempt=run_attempt, job_key=job_key, token=token)
        record.update(status="success", allocation=allocation)
        _write_ledger(output, record)
        return 0
    except Exception:
        record.update(status="reject-evidence", failure_class="attempt_scoped_job_or_nonce")
        _write_ledger(output, record)
        return 1


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON input: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("request and ledger must be JSON objects")
    return value


def _reject_secrets(value: Any, *, location: str = "payload") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered == "authorization" and item == "none":
                continue
            if any(marker in lowered for marker in SECRET_MARKERS):
                raise ValueError(f"secret-like field is forbidden: {location}.{key}")
            _reject_secrets(item, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_secrets(item, location=f"{location}[{index}]")
    elif isinstance(value, str) and any(marker in value.lower() for marker in ("ghp_", "github_pat_", "bearer ")):
        raise ValueError(f"secret-like value is forbidden: {location}")


def validate_stage1_screening_runtime(environment: object) -> dict[str, Any]:
    """Reject a hosted Stage 1 request that is not bound to its allocation/runtime."""
    if not isinstance(environment, dict) or set(environment) != STAGE1_RUNTIME_FIELDS:
        raise ValueError("strict Stage 1 hosted runtime binding")
    if not isinstance(environment["branch"], str) or environment["branch"] in {"", "main", "master"} or environment["workflow_path"] != ".github/workflows/eval.yml":
        raise ValueError("Stage 1 branch/workflow path")
    if not isinstance(environment["head_sha"], str) or not SHA.fullmatch(environment["head_sha"]):
        raise ValueError("Stage 1 immutable head")
    if not isinstance(environment["run_id"], int) or environment["run_id"] <= 0:
        raise ValueError("Stage 1 run identity")
    if not isinstance(environment["run_attempt"], int) or environment["run_attempt"] <= 0:
        raise ValueError("Stage 1 run attempt")
    if not isinstance(environment["job_key"], str) or not environment["job_key"]:
        raise ValueError("Stage 1 job key")
    if environment["job_allocation_nonce"] != f"{environment['run_id']}-{environment['run_attempt']}-{environment['job_key']}":
        raise ValueError("Stage 1 job allocation nonce")
    if environment["runtime"] != STAGE1_RUNTIME_IDENTITY:
        raise ValueError("Stage 1 locked runtime/OMP identity")
    return environment


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()


def _configured_live_upstream_head(branch: str) -> str:
    """Resolve the one configured tracking ref directly from its remote."""
    try:
        remote = _git("config", "--get", f"branch.{branch}.remote")
        merge_ref = _git("config", "--get", f"branch.{branch}.merge")
    except (OSError, subprocess.SubprocessError):
        raise ValueError("unresolved configured upstream") from None
    if len(remote.splitlines()) != 1 or len(merge_ref.splitlines()) != 1 \
            or not remote or not merge_ref or not merge_ref.startswith("refs/"):
        raise ValueError("invalid configured upstream")
    try:
        reply = _git("ls-remote", "--exit-code", "--refs", remote, merge_ref)
    except (OSError, subprocess.SubprocessError):
        raise ValueError("unresolved configured upstream") from None
    lines = reply.splitlines()
    if len(lines) != 1:
        raise ValueError("invalid configured upstream")
    fields = lines[0].split("\t")
    if len(fields) != 2 or fields[1] != merge_ref or not SHA.fullmatch(fields[0]):
        raise ValueError("invalid configured upstream")
    return fields[0]


def validate_feature_worktree_preflight(request: dict[str, Any], *, root: Path | None = None) -> dict[str, Any]:
    """Validate exact feature worktree identity and its explicit dirty policy."""
    allowed = {"repository", "branch", "worktree_root", "head_sha", "allowed_dirty_paths", "workflow_name", "campaign_stage", "continuation_binding", "require_upstream_head", "ledger_path", "evidence_packet"}
    unknown = set(request) - allowed
    if unknown:
        raise ValueError(f"unknown request fields: {sorted(unknown)}")
    required = {"repository", "branch", "worktree_root", "head_sha", "allowed_dirty_paths", "workflow_name", "campaign_stage"}
    if set(request) & required != required:
        raise ValueError("missing required preflight fields")
    if request["campaign_stage"] not in STAGES:
        raise ValueError("unknown campaign_stage")
    if not isinstance(request["branch"], str) or request["branch"] in {"master", "main"}:
        raise ValueError("unsafe feature branch")
    if not isinstance(request["head_sha"], str) or not SHA.fullmatch(request["head_sha"]):
        raise ValueError("invalid immutable head SHA")
    if not isinstance(request["allowed_dirty_paths"], list) or not all(isinstance(item, str) and item and not Path(item).is_absolute() and ".." not in Path(item).parts for item in request["allowed_dirty_paths"]):
        raise ValueError("invalid allowed_dirty_paths")
    actual_root = (root or Path(_git("rev-parse", "--show-toplevel"))).resolve()
    if actual_root != Path(request["worktree_root"]).resolve():
        raise ValueError("wrong worktree root")
    if _git("branch", "--show-current") != request["branch"]:
        raise ValueError("wrong feature branch")
    if _git("rev-parse", "HEAD") != request["head_sha"]:
        raise ValueError("wrong immutable head")
    if "require_upstream_head" in request and not isinstance(request["require_upstream_head"], bool):
        raise ValueError("invalid upstream head requirement")
    if request.get("require_upstream_head"):
        try:
            upstream_head = _git("rev-parse", "--verify", "@{upstream}")
        except (OSError, subprocess.SubprocessError) as exc:
            raise ValueError("unresolved upstream head") from exc
        if not SHA.fullmatch(upstream_head) or upstream_head != request["head_sha"]:
            raise ValueError("upstream head differs from immutable head")
        if _configured_live_upstream_head(request["branch"]) != request["head_sha"]:
            raise ValueError("configured upstream differs from immutable head")
    status = _git("status", "--porcelain=v1").splitlines()
    dirty = sorted(line[3:] for line in status if len(line) >= 4)
    if set(dirty) - set(request["allowed_dirty_paths"]):
        raise ValueError("disallowed dirty worktree state")
    return {"worktree_root": str(actual_root), "branch": request["branch"], "head_sha": request["head_sha"], "status": dirty}


def validate_phase07_evidence_packet(packet: dict[str, Any]) -> dict[str, Any]:
    """Validate untrusted hosted-artifact metadata at the local trust boundary."""
    required = {"repository", "run_id", "run_attempt", "job_id", "job_allocation_nonce", "artifact_id", "artifact_name", "archive_sha256", "content_sha256", "record_self_sha256", "retention_days_requested", "retention_days_accepted", "head_sha", "runner", "lock_identity", "build_id", "retry_lineage"}
    if set(packet) != required:
        raise ValueError("untrusted artifact packet fields")
    _reject_secrets(packet)
    if not isinstance(packet["repository"], str) or not packet["repository"]:
        raise ValueError("repository binding")
    for key in ("run_id", "run_attempt", "job_id", "artifact_id", "retention_days_requested", "retention_days_accepted"):
        if not isinstance(packet[key], int) or packet[key] <= 0:
            raise ValueError(f"invalid {key}")
    if not isinstance(packet["job_allocation_nonce"], str) or len(packet["job_allocation_nonce"]) < 16:
        raise ValueError("job allocation nonce")
    if not SHA.fullmatch(packet["head_sha"]) or not all(isinstance(packet[key], str) and HEX64.fullmatch(packet[key]) for key in ("archive_sha256", "content_sha256")):
        raise ValueError("immutable digest or head binding")
    if packet["retention_days_requested"] != 90 or packet["retention_days_accepted"] != 90:
        raise ValueError("exact 90-day requested and accepted retention is mandatory")
    if not isinstance(packet["artifact_name"], str) or not packet["artifact_name"]:
        raise ValueError("artifact name binding")
    if not isinstance(packet["runner"], dict) or not {"os", "image", "architecture"} <= set(packet["runner"]):
        raise ValueError("runner variance metadata")
    retry = packet["retry_lineage"]
    if not isinstance(retry, dict) or set(retry) != {"failure_class", "original_run_id", "replacement_run_id"}:
        raise ValueError("retry lineage")
    if retry["replacement_run_id"] is not None and retry["failure_class"] not in INFRA_FAILURES:
        raise ValueError("only classified GitHub infrastructure failures are retryable")
    if packet["record_self_sha256"] != canonical_digest(packet):
        raise ValueError("artifact self-digest")
    return packet


def validate_phase07_evidence_set(packets: list[dict[str, Any]], *, expected_repository: str, expected_head: str) -> list[dict[str, Any]]:
    """Require distinct hosted allocations and fresh builds; runner variance is data, not equality."""
    if not packets:
        raise ValueError("hosted evidence set is empty")
    tuples: set[tuple[int, int, int, str]] = set()
    builds: set[str] = set()
    for packet in packets:
        validate_phase07_evidence_packet(packet)
        if packet["repository"] != expected_repository or packet["head_sha"] != expected_head:
            raise ValueError("repository or head binding")
        allocation = (packet["run_id"], packet["run_attempt"], packet["job_id"], packet["job_allocation_nonce"])
        if allocation in tuples:
            raise ValueError("reused hosted allocation tuple")
        tuples.add(allocation)
        build_id = packet["build_id"]
        if not isinstance(build_id, str) or not HEX64.fullmatch(build_id) or build_id in builds:
            raise ValueError("fresh build identity")
        builds.add(build_id)
    return packets


def _write_ledger(path: Path, record: dict[str, Any]) -> None:
    record["record_self_sha256"] = canonical_digest(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def seal_hosted_preflight(*, output: Path, stage: str, continuation: str, repository: str,
                          run_id: int, run_attempt: int, head_sha: str, job_key: str,
                          runner_os: str, runner_architecture: str, root: Path = Path(".")) -> int:
    """Stdlib-only hosted preflight: always seal success or rejection before exit."""
    record: dict[str, Any] = {
        "schema_version": 2, "repository": repository, "run_id": run_id,
        "run_attempt": run_attempt, "head_sha": head_sha, "campaign_stage": "preflight",
        "job_key": job_key, "job_allocation_nonce": f"{run_id}-{run_attempt}-{job_key}",
        "runner": {"os": runner_os, "architecture": runner_architecture},
        "retention_days_requested": 90, "authorization": "none",
    }
    try:
        if stage != "preflight" or continuation:
            raise ValueError("preflight accepts only empty continuation input")
        if not SHA.fullmatch(head_sha) or not repository or not job_key:
            raise ValueError("invalid immutable hosted identity")
        lock = (root / "requirements.txt").read_text(encoding="utf-8")
        manifest = json.loads((root / "eval" / "model-manifest.json").read_text(encoding="utf-8"))
        if not all(value in lock for value in ("lancedb==0.34.0", "numpy==2.2.6", "pyarrow==25.0.0")):
            raise ValueError("lock syntax")
        from eval.ann_corpus_manifest import validate_model_manifest
        validate_model_manifest(manifest)
        record["status"] = "success"
        record["asvs_l1"] = {"input_validation": "pass", "secret_handling": "pass", "access_control": "pending-artifact-upload", "artifact_trust_boundary": "pending-artifact-upload"}
        code = 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        record["status"] = "reject-evidence"
        record["reason"] = f"{type(exc).__name__}: {exc}"
        code = 1
    _write_ledger(output, record)
    return code


def finalize_pipeline_artifact(*, output_dir: Path, stage: str, head_sha: str,
                               run_id: int, run_attempt: int, job_key: str,
                               job_status: str) -> int:
    """Preserve campaign output or seal a no-secret rejection for every Python-visible failure."""
    if stage == "hybrid":
        raw_dir = output_dir / "raw"
        if job_status == "success":
            try:
                validated = validate_hybrid_artifact_tree(raw_dir)
                allocation = validated["result"].get("allocation", {}).get("allocation")
                if validated["result"].get("head_sha") != head_sha \
                        or not isinstance(allocation, dict) or allocation.get("run_id") != run_id \
                        or allocation.get("run_attempt") != 1 or run_attempt != 1 \
                        or allocation.get("job_key") != job_key:
                    raise ValueError("hybrid finalizer allocation binding")
                return 0
            except (ImportError, OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
        _seal_hybrid_pipeline_rejection(
            output_dir=output_dir, head_sha=head_sha, run_id=run_id,
            run_attempt=run_attempt, job_key=job_key, job_status=job_status,
        )
        return 0
    if stage == "confirmation":
        if job_status == "success":
            try:
                from eval.phase07_ann_campaign import validate_confirmation_artifact_tree
                validate_confirmation_artifact_tree(
                    output_dir, expected_head=head_sha, expected_run_id=run_id,
                    expected_run_attempt=run_attempt, expected_job_key=job_key,
                )
                return 0
            except (ImportError, OSError, TypeError, ValueError, json.JSONDecodeError):
                pass
        _seal_confirmation_pipeline_rejection(
            output_dir=output_dir, stage=stage, head_sha=head_sha, run_id=run_id,
            run_attempt=run_attempt, job_key=job_key, job_status=job_status,
        )
        return 0
    # A caller cannot relabel a confirmation tree as another campaign stage to
    # bypass its strict finalizer.  This check is lexical and never traverses a
    # symlinked root.
    if output_dir.is_symlink() or (output_dir.is_dir() and (output_dir / "confirmation-packet.json").exists()):
        _seal_confirmation_pipeline_rejection(
            output_dir=output_dir, stage=stage, head_sha=head_sha, run_id=run_id,
            run_attempt=run_attempt, job_key=job_key, job_status=job_status,
        )
        return 0
    output_dir.mkdir(parents=True, exist_ok=True)
    if any(output_dir.glob("*-result.json")) or any(output_dir.glob("*-rejection.json")):
        return 0
    _write_ledger(output_dir / f"{stage}-pipeline-rejection.json", {
        "schema_version": 1, "stage": stage, "status": "reject-evidence",
        "head_sha": head_sha, "run_id": run_id, "run_attempt": run_attempt,
        "job_key": job_key, "job_status": job_status, "authorization": "none",
    })
    return 0


def _seal_hybrid_pipeline_rejection(*, output_dir: Path, head_sha: str, run_id: int,
                                    run_attempt: int, job_key: str, job_status: str) -> None:
    """Replace the task-owned hybrid root with one uploadable rejection leaf."""
    if output_dir.is_symlink() or (output_dir.exists() and not output_dir.is_dir()):
        output_dir.unlink()
    elif output_dir.is_dir():
        shutil.rmtree(output_dir)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=False)
    _write_ledger(raw_dir / "hybrid-pipeline-rejection.json", {
        "schema_version": 1, "stage": "hybrid", "status": "reject-evidence",
        "head_sha": head_sha, "run_id": run_id, "run_attempt": run_attempt,
        "job_key": job_key, "job_status": job_status, "authorization": "none",
    })


def _seal_confirmation_pipeline_rejection(*, output_dir: Path, stage: str, head_sha: str,
                                          run_id: int, run_attempt: int, job_key: str,
                                          job_status: str) -> None:
    """Replace only our output root; never traverse an external symlink target."""
    if output_dir.is_symlink():
        output_dir.unlink()
    elif output_dir.exists() and not output_dir.is_dir():
        output_dir.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in output_dir.iterdir():
        if path.is_symlink() or path.is_file():
            path.unlink()
        elif path.is_dir():
            # ``rmtree`` refuses a symlink argument; the branch is regular and
            # remains rooted inside this just-created/owned output directory.
            shutil.rmtree(path)
        else:
            path.unlink()
    _write_ledger(output_dir / "confirmation-pipeline-rejection.json", {
        "schema_version": 1, "stage": stage, "status": "reject-evidence",
        "head_sha": head_sha, "run_id": run_id, "run_attempt": run_attempt,
        "job_key": job_key, "job_status": job_status, "authorization": "none",
    })


def reconcile_hosted(binding_file: Path, output: Path) -> int:
    """Seal strict operator-supplied hosted evidence; no download or authorization occurs here."""
    try:
        binding = _read_object(binding_file); _reject_secrets(binding)
        if set(binding) != {"repository", "head_sha", "packets"} or not isinstance(binding["packets"], list):
            raise ValueError("strict reconciliation binding")
        packets = validate_phase07_evidence_set(binding["packets"], expected_repository=binding["repository"], expected_head=binding["head_sha"])
        output.mkdir(parents=True, exist_ok=True)
        _write_ledger(output / "reconciliation-result.json", {"schema_version": 1, "status": "success", "authorization": "none", "repository": binding["repository"], "head_sha": binding["head_sha"], "packets": packets})
        return 0
    except (ValueError, OSError, json.JSONDecodeError, KeyError) as exc:
        output.mkdir(parents=True, exist_ok=True)
        _write_ledger(output / "reconciliation-rejection.json", {"schema_version": 1, "status": "reject-evidence", "authorization": "none", "reason": f"{type(exc).__name__}: {exc}"})
        return 1


def collect_confirmation_provenance(*, request_file: Path, output: Path,
                                    provenance_dir: Path, token: str,
                                    client: Any | None = None) -> int:
    """Collect API-bound provenance for downloaded confirmation artifacts.

    The request supplies only immutable run/attempt identities and local download
    paths.  Every status, job, runner label, artifact identity, expiry, and API
    digest comes from GitHub; locked host details come from the already-sealed
    production packet and are cross-checked again by the post-download reconciler.
    """
    request = _read_object(request_file)
    _reject_secrets(request)
    if set(request) != {"schema_version", "repository", "head_sha", "downloads", "record_self_sha256"} \
            or request.get("schema_version") != 1 \
            or request.get("record_self_sha256") != canonical_digest(request):
        raise ValueError("sealed confirmation download collection request")
    repository, head_sha, downloads = request["repository"], request["head_sha"], request["downloads"]
    if not isinstance(repository, str) or not repository or not isinstance(head_sha, str) or not SHA.fullmatch(head_sha):
        raise ValueError("confirmation collection repository/head binding")
    if not isinstance(downloads, list) or len(downloads) != 3:
        raise ValueError("confirmation collection requires exactly three successful downloads")
    if not token:
        raise ValueError("GitHub actions read token unavailable")
    if output.exists() or provenance_dir.is_symlink() or (provenance_dir.exists() and any(provenance_dir.iterdir())):
        raise ValueError("confirmation provenance output must be new and empty")

    from eval.phase07_ann_campaign import validate_confirmation_artifact_tree
    from eval.reconcile_ann_gate import _validate_confirmation_provenance

    github = client or GitHubActionsClient()
    records: list[dict[str, Any]] = []
    seen_runs: set[tuple[int, int]] = set()
    seen_paths: set[tuple[Path, Path]] = set()
    for download in downloads:
        if not isinstance(download, dict) or set(download) != CONFIRMATION_DOWNLOAD_FIELDS:
            raise ValueError("strict confirmation download record")
        run_id, run_attempt = download["run_id"], download["run_attempt"]
        if not isinstance(run_id, int) or run_id <= 0 or not isinstance(run_attempt, int) or run_attempt <= 0:
            raise ValueError("confirmation download run identity")
        archive, extracted = Path(download["archive"]), Path(download["extracted_dir"])
        identity, paths = (run_id, run_attempt), (archive.resolve(), extracted.resolve())
        if identity in seen_runs or paths in seen_paths or archive.is_symlink() or not archive.is_file() \
                or extracted.is_symlink() or not extracted.is_dir():
            raise ValueError("duplicate or unavailable confirmation download")
        seen_runs.add(identity); seen_paths.add(paths)

        validated = validate_confirmation_artifact_tree(
            extracted, expected_head=head_sha, expected_run_id=run_id,
            expected_run_attempt=run_attempt, expected_job_key="phase07-confirmation",
        )
        packet, allocation = validated["packet"], validated["allocation"]["allocation"]
        run_url = f"/repos/{repository}/actions/runs/{run_id}"
        jobs_url = f"{run_url}/attempts/{run_attempt}/jobs"
        artifacts_url = f"{run_url}/artifacts"
        try:
            run = github.get_json(run_url, token)
            jobs_payload = github.get_json(jobs_url, token)
            artifacts_payload = github.get_json(artifacts_url, token)
        except Exception as exc:
            raise ValueError("GitHub confirmation provenance lookup failed") from exc

        if not isinstance(run, dict) or run.get("id") != run_id or run.get("run_attempt") != run_attempt \
                or run.get("head_sha") != head_sha or not isinstance(run.get("head_branch"), str) \
                or run["head_branch"] in {"", "main", "master"} or run.get("event") != "workflow_dispatch" \
                or run.get("status") != "completed" or run.get("conclusion") != "success" \
                or not isinstance(run.get("created_at"), str):
            raise ValueError("confirmation workflow-run API binding")
        jobs = jobs_payload.get("jobs") if isinstance(jobs_payload, dict) else None
        job_name = JOB_DISPLAY_NAMES["phase07-confirmation"]
        matches = [job for job in jobs if isinstance(job, dict) and job.get("id") == allocation["job_id"]
                   and job.get("run_id") == run_id and job.get("run_attempt") == run_attempt
                   and job.get("name") == job_name and job.get("status") == "completed"
                   and job.get("conclusion") == "success"] if isinstance(jobs, list) else []
        if len(matches) != 1:
            raise ValueError("confirmation attempt-job API binding")
        job = matches[0]
        if not isinstance(job.get("runner_name"), str) or not job["runner_name"] \
                or not isinstance(job.get("runner_group_name"), str) or not job["runner_group_name"] \
                or not isinstance(job.get("labels"), list) or not job["labels"] \
                or not all(isinstance(label, str) and label for label in job["labels"]):
            raise ValueError("confirmation runner API binding")
        artifacts = artifacts_payload.get("artifacts") if isinstance(artifacts_payload, dict) else None
        artifact_name = f"phase07-confirmation-{run_id}-{run_attempt}"
        artifact_matches = [artifact for artifact in artifacts if isinstance(artifact, dict)
                            and artifact.get("name") == artifact_name] if isinstance(artifacts, list) else []
        if len(artifact_matches) != 1:
            raise ValueError("confirmation artifact API binding")
        artifact = artifact_matches[0]
        workflow_run = artifact.get("workflow_run")
        api_digest = artifact.get("digest")
        local_archive_sha256 = hashlib.sha256(archive.read_bytes()).hexdigest()
        if not isinstance(artifact.get("id"), int) or artifact["id"] <= 0 or artifact.get("expired") is not False \
                or not isinstance(api_digest, str) or api_digest != f"sha256:{local_archive_sha256}" \
                or not isinstance(artifact.get("expires_at"), str) \
                or not isinstance(workflow_run, dict) or workflow_run.get("id") != run_id \
                or workflow_run.get("head_branch") != run["head_branch"] or workflow_run.get("head_sha") != head_sha:
            raise ValueError("confirmation artifact API identity/digest")
        host = packet.get("locked_execution", {}).get("host", {})
        if not isinstance(host, dict) or not all(isinstance(host.get(name), str) and host[name]
                                                for name in ("os", "image", "architecture")):
            raise ValueError("confirmation locked host provenance")
        provenance = {
            "run_id": run_id, "run_attempt": run_attempt, "job_id": job["id"],
            "job_key": allocation["job_key"], "job_name": job_name,
            "artifact_id": artifact["id"], "artifact_name": artifact_name,
            "status": run["status"], "conclusion": run["conclusion"],
            "head_branch": run["head_branch"], "head_sha": head_sha, "event": run["event"],
            "runner": {"name": job["runner_name"], "group": job["runner_group_name"],
                       "labels": job["labels"], "os": host["os"], "image": host["image"],
                       "architecture": host["architecture"]},
            "run_created_at": run["created_at"], "artifact_expires_at": artifact["expires_at"],
            "api_archive_sha256": local_archive_sha256, "local_archive_sha256": local_archive_sha256,
            "archive": str(archive.resolve()), "extracted_dir": str(extracted.resolve()),
        }
        _validate_confirmation_provenance(provenance)
        records.append(provenance)

    provenance_dir.mkdir(parents=True, exist_ok=True)
    evidence = []
    for provenance in records:
        document = {"schema_version": 1, "evidence": [provenance]}
        path = provenance_dir / f"confirmation-{provenance['run_id']}-{provenance['run_attempt']}-provenance.json"
        _write_ledger(path, document)
        evidence.append({"artifact_dir": provenance["extracted_dir"], "provenance": str(path.resolve())})
    _write_ledger(output, {"schema_version": 1, "evidence": evidence})
    return 0


def run_preflight(request_file: Path, ledger_file: Path) -> int:
    request = _read_object(request_file)
    _reject_secrets(request)
    if request.get("ledger_path") != str(ledger_file.resolve()):
        raise ValueError("request must bind the exact ledger output path")
    evidence = validate_feature_worktree_preflight(request)
    record = {"schema_version": 1, "mode": "preflight", "request": request, "branch_head_status": evidence, "asvs_l1": {"input_validation": "pass", "access_control": "pending-hosted-proof", "secret_handling": "pass", "artifact_trust_boundary": "pending-hosted-proof"}}
    _write_ledger(ledger_file, record)
    return 0


def run_campaign(request_file: Path, ledger_file: Path) -> int:
    request = _read_object(request_file)
    _reject_secrets(request)
    evidence = validate_feature_worktree_preflight(request)
    packet = request.get("evidence_packet")
    if not isinstance(packet, dict):
        raise ValueError("campaign requires one evidence_packet object")
    validate_phase07_evidence_packet(packet)
    _write_ledger(ledger_file, {"schema_version": 1, "mode": "campaign", "branch_head_status": evidence, "evidence_packet": packet})
    return 0


def run_decision(request_file: Path, ledger_file: Path) -> int:
    return run_campaign(request_file, ledger_file)


def run_pr_gates(request_file: Path, ledger_file: Path) -> int:
    return run_campaign(request_file, ledger_file)


def run_confirmation_plan(*, stage1_ledger: Path, request_file: Path, workflow_inputs_dir: Path,
                          preflight_request: Path) -> int:
    """Materialize the only three dispatchable ordinal records from the exact head."""
    root = Path(_git("rev-parse", "--show-toplevel"))
    head = _git("rev-parse", "HEAD")
    plan = build_confirmation_plan(stage1_ledger, post_task0_head=head)
    validate_confirmation_plan(plan)
    request_file.parent.mkdir(parents=True, exist_ok=True)
    request_file.write_text(json.dumps(plan["confirmation_request"], sort_keys=True, indent=2) + "\n", encoding="utf-8")
    workflow_inputs_dir.mkdir(parents=True, exist_ok=True)
    for record in plan["workflow_inputs"]:
        slot = record["slot"]
        (workflow_inputs_dir / f"confirmation-ordinal{slot['ordinal']}.json").write_text(
            json.dumps({"confirmation_request": plan["confirmation_request"], "workflow_input": record,
                        "replacement_for_run_id": None}, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    ledger_name = preflight_request.name.replace("-request.json", "-ledger.json")
    if ledger_name == preflight_request.name:
        ledger_name = f"{preflight_request.stem}-ledger{preflight_request.suffix}"
    preflight = {
        "repository": _read_object(stage1_ledger)["repository"], "branch": _git("branch", "--show-current"),
        "worktree_root": str(root), "head_sha": head, "allowed_dirty_paths": [],
        "workflow_name": "eval", "campaign_stage": "confirmation", "continuation_binding": "",
        "require_upstream_head": True, "ledger_path": str(preflight_request.with_name(ledger_name).resolve()),
    }
    preflight_request.parent.mkdir(parents=True, exist_ok=True)
    preflight_request.write_text(json.dumps(preflight, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 0


def run_hybrid_plan(*, dense_ledger: Path, request_file: Path, workflow_inputs_dir: Path,
                    preflight_request: Path) -> int:
    """Write one baseline and two candidate bundles after ledger validation."""
    root = Path(_git("rev-parse", "--show-toplevel"))
    head = _git("rev-parse", "HEAD")
    plan = build_hybrid_plan(dense_ledger, post_implementation_head=head)
    validate_hybrid_plan(plan)
    request_file.parent.mkdir(parents=True, exist_ok=True)
    request_file.write_text(json.dumps(plan["hybrid_request"], sort_keys=True, indent=2) + "\n", encoding="utf-8")
    workflow_inputs_dir.mkdir(parents=True, exist_ok=True)
    for input_record in plan["workflow_inputs"]:
        suffix = "baseline" if input_record["role"] == "baseline" else f"m{input_record['config']['m']}"
        bundle = _sealed({
            "schema_version": 1, "hybrid_request": plan["hybrid_request"], "workflow_input": input_record,
            "replacement_for_run_id": None,
        })
        (workflow_inputs_dir / f"hybrid-{suffix}.json").write_text(
            json.dumps(bundle, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )
    validate_hybrid_workflow_inputs_dir(workflow_inputs_dir, expected_head=head)
    ledger_name = preflight_request.name.replace("-request.json", "-ledger.json")
    preflight = {
        "repository": "allenwoo713/obsidian_wiki_skill", "branch": _git("branch", "--show-current"),
        "worktree_root": str(root), "head_sha": head, "allowed_dirty_paths": [],
        "workflow_name": "eval", "campaign_stage": "hybrid", "continuation_binding": "",
        "require_upstream_head": True,
        "ledger_path": str(preflight_request.with_name(ledger_name).resolve()),
    }
    preflight_request.parent.mkdir(parents=True, exist_ok=True)
    preflight_request.write_text(json.dumps(preflight, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 0


def main(argv: list[str] | None = None, *, github_client: Any | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "campaign", "decision", "pr-gates", "hosted-preflight", "finalize", "reconcile-hosted", "confirmation-allocation", "confirmation-plan", "confirmation-provenance", "hybrid-plan", "hybrid-allocation", "hybrid-dispatch", "hybrid-collection-request", "hybrid-provenance", "frozen-size-preflight", "prepare-plan", "frozen-prepare-provenance", "role-plan"))
    parser.add_argument("--request-file", type=Path)
    parser.add_argument("--ledger-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--stage")
    parser.add_argument("--continuation", default="")
    parser.add_argument("--repository")
    parser.add_argument("--run-id", type=int)
    parser.add_argument("--run-attempt", type=int)
    parser.add_argument("--head-sha")
    parser.add_argument("--job-key")
    parser.add_argument("--runner-os", default="")
    parser.add_argument("--runner-architecture", default="")
    parser.add_argument("--job-status", default="unknown")
    parser.add_argument("--stage1-ledger", type=Path)
    parser.add_argument("--dense-ledger", type=Path)
    parser.add_argument("--workflow-inputs-dir", type=Path)
    parser.add_argument("--preflight-request", type=Path)
    parser.add_argument("--provenance-dir", type=Path)
    parser.add_argument("--dispatch-bundle", type=Path)
    parser.add_argument("--locked-execution", type=Path)
    parser.add_argument("--allocation", type=Path)
    parser.add_argument("--downloads-file", type=Path)
    parser.add_argument("--frozen-dir", type=Path)
    parser.add_argument("--workflow-input-file", type=Path)
    parser.add_argument("--prepare-bundle", type=Path)
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--hybrid-request", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "frozen-size-preflight":
            if args.request_file is None or args.ledger_file is None:
                raise ValueError("frozen-size-preflight requires local preflight and ledger")
            return seal_frozen_size_preflight(request_file=args.request_file, ledger_file=args.ledger_file)
        if args.command == "prepare-plan":
            if args.request_file is None or args.workflow_input_file is None or not args.head_sha:
                raise ValueError("prepare-plan requires preflight, output, and exact head")
            return run_frozen_prepare_plan(
                preflight_file=args.request_file, workflow_input_file=args.workflow_input_file,
                expected_head=args.head_sha,
            )
        if args.command == "frozen-prepare-provenance":
            if None in (args.prepare_bundle, args.ledger_file, args.archive, args.frozen_dir, args.locked_execution):
                raise ValueError("frozen prepare provenance requires sealed input, archive, root, execution, and output")
            identity = collect_frozen_prepare_provenance(
                prepare_bundle=_read_object(args.prepare_bundle), repository=args.repository or "",
                run_id=args.run_id or 0, run_attempt=args.run_attempt or 0, archive=args.archive,
                frozen_dir=args.frozen_dir, locked_execution=_read_object(args.locked_execution),
                token=os.environ.get("GITHUB_TOKEN", ""), client=github_client,
            )
            # This is an audit copy of the collector's *exact* shared
            # identity shape.  Do not add a second self-digest/envelope: that
            # would create a competing schema which role code could mistake
            # for API provenance.
            args.ledger_file.parent.mkdir(parents=True, exist_ok=True)
            args.ledger_file.write_text(
                json.dumps(identity, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            return 0
        if args.command == "role-plan":
            if None in (args.prepare_bundle, args.hybrid_request, args.workflow_inputs_dir,
                        args.archive, args.frozen_dir, args.locked_execution):
                raise ValueError("role-plan requires API-bound prepare, request, root, execution, and output")
            return run_frozen_role_plan(
                prepare_bundle=args.prepare_bundle, hybrid_request=args.hybrid_request,
                workflow_inputs_dir=args.workflow_inputs_dir, repository=args.repository or "",
                run_id=args.run_id or 0, run_attempt=args.run_attempt or 0, archive=args.archive,
                frozen_dir=args.frozen_dir, locked_execution=args.locked_execution,
                token=os.environ.get("GITHUB_TOKEN", ""), client=github_client,
            )
        if args.command == "hosted-preflight":
            if args.ledger_file is None:
                raise ValueError("hosted preflight requires --ledger-file")
            return seal_hosted_preflight(output=args.ledger_file, stage=args.stage or "", continuation=args.continuation,
                repository=args.repository or "", run_id=args.run_id or 0, run_attempt=args.run_attempt or 0,
                head_sha=args.head_sha or "", job_key=args.job_key or "", runner_os=args.runner_os,
                runner_architecture=args.runner_architecture)
        if args.command == "finalize":
            return finalize_pipeline_artifact(output_dir=args.output_dir or Path("."), stage=args.stage or "",
                head_sha=args.head_sha or "", run_id=args.run_id or 0, run_attempt=args.run_attempt or 0,
                job_key=args.job_key or "", job_status=args.job_status)
        if args.command == "reconcile-hosted":
            if args.request_file is None or args.ledger_file is None:
                raise ValueError("reconcile-hosted requires binding and output")
            return reconcile_hosted(args.request_file, args.ledger_file)
        if args.command == "confirmation-allocation":
            if args.request_file is None or args.ledger_file is None:
                raise ValueError("confirmation allocation requires sealed input and output")
            token = os.environ.get("GITHUB_TOKEN", "")
            return seal_confirmation_allocation(workflow_inputs=_read_object(args.request_file), output=args.ledger_file,
                                                repository=args.repository or "", run_id=args.run_id or 0,
                                                run_attempt=args.run_attempt or 0, job_key=args.job_key or "", head_sha=args.head_sha or "", token=token)
        if args.command == "confirmation-plan":
            if None in (args.stage1_ledger, args.request_file, args.workflow_inputs_dir, args.preflight_request):
                raise ValueError("confirmation-plan requires immutable ledger and all generated output paths")
            return run_confirmation_plan(stage1_ledger=args.stage1_ledger, request_file=args.request_file,
                                         workflow_inputs_dir=args.workflow_inputs_dir, preflight_request=args.preflight_request)
        if args.command == "hybrid-plan":
            if None in (args.dense_ledger, args.request_file, args.workflow_inputs_dir, args.preflight_request):
                raise ValueError("hybrid-plan requires sealed dense ledger and all generated output paths")
            return run_hybrid_plan(dense_ledger=args.dense_ledger, request_file=args.request_file,
                                   workflow_inputs_dir=args.workflow_inputs_dir,
                                   preflight_request=args.preflight_request)
        if args.command == "hybrid-allocation":
            if args.request_file is None or args.ledger_file is None:
                raise ValueError("hybrid allocation requires sealed input and output")
            return seal_hybrid_allocation(
                workflow_inputs=_read_object(args.request_file), output=args.ledger_file,
                repository=args.repository or "", run_id=args.run_id or 0,
                run_attempt=args.run_attempt or 0, job_key=args.job_key or "", head_sha=args.head_sha or "",
                token=os.environ.get("GITHUB_TOKEN", ""), client=github_client,
            )
        if args.command == "hybrid-dispatch":
            if None in (args.dispatch_bundle, args.locked_execution, args.allocation, args.output_dir):
                raise ValueError("hybrid-dispatch requires sealed dispatch, locked execution, allocation, and output")
            allocation_document = _read_object(args.allocation)
            allocation = allocation_document.get("allocation") if set(allocation_document) >= {"allocation"} else allocation_document
            execute_hybrid_dispatch(
                bundle=_read_object(args.dispatch_bundle), locked_execution=_read_object(args.locked_execution),
                allocation=allocation, work_dir=args.output_dir, frozen_dir=args.frozen_dir,
            )
            return 0
        if args.command == "hybrid-collection-request":
            if args.request_file is None or args.downloads_file is None or args.ledger_file is None:
                raise ValueError("hybrid collection requires request, downloads, and output")
            record = build_hybrid_collection_request(
                hybrid_request=_read_object(args.request_file), downloads=_read_object(args.downloads_file),
            )
            _write_ledger(args.ledger_file, record)
            return 0
        if args.command == "hybrid-provenance":
            if args.request_file is None or args.ledger_file is None or args.provenance_dir is None:
                raise ValueError("hybrid provenance requires collection request, manifest, and provenance directory")
            return collect_hybrid_provenance(
                request_file=args.request_file, output=args.ledger_file, provenance_dir=args.provenance_dir,
                token=os.environ.get("GITHUB_TOKEN", ""), client=github_client,
            )
        if args.command == "confirmation-provenance":
            if None in (args.request_file, args.ledger_file, args.provenance_dir):
                raise ValueError("confirmation provenance requires request, manifest, and provenance directory")
            return collect_confirmation_provenance(
                request_file=args.request_file, output=args.ledger_file,
                provenance_dir=args.provenance_dir, token=os.environ.get("GITHUB_TOKEN", ""),
                client=github_client,
            )
        if args.request_file is None or args.ledger_file is None:
            raise ValueError("operator command requires request and ledger files")
        return {"preflight": run_preflight, "campaign": run_campaign, "decision": run_decision, "pr-gates": run_pr_gates}[args.command](args.request_file, args.ledger_file)
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"[FAIL] Phase 07 operator gate: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
