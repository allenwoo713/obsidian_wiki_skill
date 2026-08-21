"""Fail-closed, lightweight input manifests for Phase 7 ANN evaluation."""
from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath


_SHA256 = set("0123456789abcdef")
_MODEL_RUNTIME = {"python": "3.13", "scipy": "1.15.3", "lancedb": "0.34.0"}


def canonical_sha256(value: Mapping[str, object]) -> str:
    payload = dict(value)
    payload.pop("record_self_sha256", None)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _digest(name: str, value: object) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value) - _SHA256:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _positive(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_truth_strata(manifest: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != 1:
        raise ValueError("truth-strata schema_version")
    expected = {"stress", "hybrid", "personal_wiki_ann", "generator"}
    if set(manifest) != expected | {"schema_version"}:
        raise ValueError("three truth strata must be complete and separate")
    stress = manifest["stress"]
    hybrid = manifest["hybrid"]
    personal = manifest["personal_wiki_ann"]
    generator = manifest["generator"]
    if not isinstance(stress, Mapping) or stress.get("kind") != "seeded_vector_exact":
        raise ValueError("stress stratum")
    if _positive("stress.rows", stress.get("rows")) != 77_348 or _positive("stress.dimensions", stress.get("dimensions")) != 384:
        raise ValueError("stress stratum dimensions")
    for name in ("corpus_seed", "query_seed", "corpus_sha256", "query_sha256", "exact_truth_sha256"):
        value = stress.get(name)
        if name.endswith("sha256"):
            _digest(f"stress.{name}", value)
        elif not isinstance(value, str) or not value:
            raise ValueError(f"stress.{name}")
    if not isinstance(hybrid, Mapping) or hybrid.get("kind") != "labeled_natural_language_hybrid" or _positive("hybrid.query_count", hybrid.get("query_count")) != 105:
        raise ValueError("hybrid stratum")
    for name in ("labels_sha256", "query_sha256"):
        _digest(f"hybrid.{name}", hybrid.get(name))
    if not isinstance(personal, Mapping) or personal.get("kind") != "natural_language_ann_exact" or _positive("personal.query_count", personal.get("query_count")) != 256:
        raise ValueError("personal ANN stratum")
    for name in ("query_sha256", "exact_truth_sha256"):
        _digest(f"personal.{name}", personal.get(name))
    if personal.get("indexed_query_overlap_count") != 0:
        raise ValueError("natural ANN queries must never be indexed")
    if not isinstance(generator, Mapping) or generator.get("version") != "public-distractor-v1":
        raise ValueError("public distractor generator")
    if not isinstance(generator.get("seed"), str) or not generator["seed"]:
        raise ValueError("generator seed")
    for name in ("source_fixture_sha256", "rules_sha256"):
        _digest(f"generator.{name}", generator.get(name))
    return manifest


def validate_query_corpus_separation(manifest: Mapping[str, object], *, indexed_row_digests: Iterable[str], query_row_digests: Iterable[str]) -> None:
    validate_truth_strata(manifest)
    indexed, queries = set(indexed_row_digests), set(query_row_digests)
    if not indexed.isdisjoint(queries):
        raise ValueError("query/corpus overlap")


def canonical_content_tree_sha256(root: Path) -> str:
    """Digest a public corpus by canonical relative path and file bytes, never path root."""
    if not root.is_dir() or root.is_symlink():
        raise ValueError("unsafe corpus root")
    digest = hashlib.sha256()
    files = sorted((path for path in root.rglob("*") if path.is_file()), key=lambda path: path.relative_to(root).as_posix())
    if not files:
        raise ValueError("empty corpus")
    for path in files:
        if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
            raise ValueError("unsafe corpus entry")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(relative); digest.update(b"\0"); digest.update(path.read_bytes()); digest.update(b"\0")
    return digest.hexdigest()


def validate_indexed_query_digest_separation(*, indexed_row_digests: Iterable[str], query_row_digests: Iterable[str]) -> dict[str, object]:
    """Fail closed on actual indexed/query digest overlap and return sealed identities."""
    indexed, queries = set(indexed_row_digests), set(query_row_digests)
    if not indexed or not queries or any(not isinstance(value, str) or len(value) != 64 for value in indexed | queries):
        raise ValueError("indexed/query digest identities")
    overlap = indexed & queries
    if overlap:
        raise ValueError("query/corpus overlap")
    return {"indexed_digest_set_sha256": hashlib.sha256("\n".join(sorted(indexed)).encode()).hexdigest(),
            "query_digest_set_sha256": hashlib.sha256("\n".join(sorted(queries)).encode()).hexdigest(),
            "indexed_query_overlap_count": len(overlap)}


def validate_lightweight_repository_inputs(paths: Iterable[str]) -> None:
    forbidden_suffixes = (".npy", ".npz", ".parquet", ".lance", ".safetensors", ".bin", ".arrow")
    for raw in paths:
        path = PurePosixPath(raw)
        if path.is_absolute() or ".." in path.parts or raw.endswith(forbidden_suffixes) or "/lancedb/" in f"/{raw}/":
            raise ValueError("generated embeddings or indexes are not lightweight inputs")


def validate_model_tree(model_root: Path, lock: Mapping[str, object], *, allow_download: bool = False) -> dict[str, object]:
    """Validate an already-present immutable model tree; never hydrate it."""
    if allow_download:
        raise ValueError("model validation never downloads or hydrates")
    if not isinstance(lock, Mapping) or lock.get("schema_version") != 1:
        raise ValueError("model manifest schema")
    if lock.get("model_id") != "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2":
        raise ValueError("model identity")
    revision = lock.get("revision")
    if not isinstance(revision, str) or len(revision) != 40 or set(revision) - set("0123456789abcdef"):
        raise ValueError("model revision must be an immutable provider commit")
    if lock.get("runtime") != _MODEL_RUNTIME:
        raise ValueError("exact locked runtime")
    files = lock.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("model manifest files")
    if not model_root.is_dir() or model_root.is_symlink():
        raise ValueError("model root unavailable or unsafe")
    expected: dict[str, str] = {}
    for item in files:
        if not isinstance(item, Mapping):
            raise ValueError("model file record")
        raw = item.get("path")
        if not isinstance(raw, str) or not raw or PurePosixPath(raw).is_absolute() or ".." in PurePosixPath(raw).parts:
            raise ValueError("unsafe model file path")
        if raw in expected:
            raise ValueError("duplicate model file path")
        expected[raw] = _digest("model file sha256", item.get("sha256"))
    actual: dict[str, str] = {}
    for path in model_root.rglob("*"):
        relative = path.relative_to(model_root).as_posix()
        if path.is_symlink() or not path.is_file() or not stat.S_ISREG(path.stat().st_mode):
            raise ValueError("model tree has symlink or nonregular file")
        actual[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError("model tree missing, extra, or changed file")
    claimed = lock.get("record_self_sha256")
    if not isinstance(claimed, str) or claimed != canonical_sha256(lock):
        raise ValueError("model manifest self digest")
    return dict(lock)


def load_manifest(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"manifest unavailable or invalid: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("manifest must be an object")
    if value.get("record_self_sha256") != canonical_sha256(value):
        raise ValueError("manifest self digest")
    return value
