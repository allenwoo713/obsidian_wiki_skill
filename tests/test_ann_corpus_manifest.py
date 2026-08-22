"""Phase 07 red contracts for separate ANN truth strata and model manifests."""
from __future__ import annotations

import importlib
import hashlib
from copy import deepcopy
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent


def _manifests():
    """Load the Phase 07 fail-closed manifest boundary (D-06..D-10, D-18)."""
    return importlib.import_module("eval.ann_corpus_manifest")


def _three_strata() -> dict[str, object]:
    return {
        "schema_version": 1,
        "stress": {
            "kind": "seeded_vector_exact", "rows": 77348, "dimensions": 384,
            "corpus_seed": "phase07-stress-corpus", "query_seed": "phase07-stress-queries",
            "corpus_sha256": "a" * 64, "query_sha256": "b" * 64, "exact_truth_sha256": "c" * 64,
        },
        "hybrid": {
            "kind": "labeled_natural_language_hybrid", "query_count": 105,
            "labels_sha256": "d" * 64, "query_sha256": "e" * 64,
        },
        "personal_wiki_ann": {
            "kind": "natural_language_ann_exact", "query_count": 256,
            "query_sha256": "f" * 64, "exact_truth_sha256": "0" * 64,
            "indexed_query_overlap_count": 0,
        },
        "generator": {
            "version": "public-distractor-v1", "seed": "phase07-public-corpus",
            "source_fixture_sha256": "1" * 64, "rules_sha256": "2" * 64,
        },
    }


def test_d06_to_d09_three_truth_strata_are_versioned_and_never_merge() -> None:
    """D-06/D-07/D-08/D-09: each truth source has a distinct immutable identity."""
    manifests = _manifests()
    manifest = _three_strata()
    assert manifests.validate_truth_strata(manifest) is manifest
    for removed in ("stress", "hybrid", "personal_wiki_ann"):
        broken = deepcopy(manifest)
        broken.pop(removed)
        with pytest.raises(ValueError):
            manifests.validate_truth_strata(broken)


def test_d08_query_rows_cannot_leak_into_the_indexed_corpus() -> None:
    """D-08: natural-language ANN query content is never an indexed row."""
    manifests = _manifests()
    manifest = _three_strata()
    assert manifests.validate_query_corpus_separation(
        manifest, indexed_row_digests={"a" * 64}, query_row_digests={"b" * 64},
    ) is None
    with pytest.raises(ValueError, match="overlap"):
        manifests.validate_query_corpus_separation(
            manifest, indexed_row_digests={"a" * 64}, query_row_digests={"a" * 64},
        )


def test_d09_d10_public_distractor_recipe_accepts_only_lightweight_inputs() -> None:
    """D-09/D-10: repository manifests may not smuggle vectors or LanceDB indexes."""
    manifests = _manifests()
    tracked = {
        "eval/personal_wiki_corpus_manifest.json",
        "eval/public_distractor_recipe.json",
        "tests/fixtures/public_wiki_source.json",
    }
    assert manifests.validate_lightweight_repository_inputs(tracked) is None
    for prohibited in (
        "eval/generated/embeddings.npy",
        "eval/generated/lancedb/data.lance",
        "eval/generated/dense_vectors.parquet",
    ):
        with pytest.raises(ValueError):
            manifests.validate_lightweight_repository_inputs(tracked | {prohibited})


def test_d10_d18_model_tree_lock_and_cache_poisoning_fail_closed(tmp_path: Path) -> None:
    """D-10/D-18: local validation is read-only and rejects mutable/poisoned trees."""
    manifests = _manifests()
    model_root = tmp_path / "model"
    model_root.mkdir()
    (model_root / "config.json").write_text("{}", encoding="utf-8")
    lock = {
        "schema_version": 1,
        "model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "revision": "immutable-provider-commit",
        "runtime": {"python": "3.13", "scipy": "1.15.3", "lancedb": "0.34.0"},
        "files": [{"path": "config.json", "sha256": "0" * 64}],
        "record_self_sha256": "sealed-by-phase07-implementation",
    }
    with pytest.raises(ValueError):
        manifests.validate_model_tree(model_root, lock, allow_download=False)


def test_model_tree_allows_nested_regular_files_but_rejects_poisoning(tmp_path: Path) -> None:
    manifests = _manifests()
    root = tmp_path / "model"; nested = root / "1_Pooling"; nested.mkdir(parents=True)
    content = b"{}"; (nested / "config.json").write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    lock = {"schema_version": 2, "model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "provider": {"name":"huggingface","repository":"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2","revision":"a" * 40}, "runtime": {"python":"3.13","scipy":"1.15.3","lancedb":"0.34.0"}, "provider_files": [{"path":"1_Pooling/config.json","sha256":digest}], "local_compatible_metadata": {"path":"configuration.json", "sha256":"a" * 64, "provenance":{"provider":"modelscope","model_id":"sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2","kind":"legacy-local-bootstrap"}}}
    lock["record_self_sha256"] = manifests.canonical_sha256(lock)
    assert manifests.validate_model_tree(root, lock) == lock
    (root / "extra.json").write_text("x")
    with pytest.raises(ValueError): manifests.validate_model_tree(root, lock)


def test_model_tree_accepts_only_pinned_provider_files_and_one_compatible_metadata(tmp_path: Path) -> None:
    """The ModelScope-only metadata is optional, singular, and hash-locked."""
    manifests = _manifests()
    root = tmp_path / "model"; root.mkdir()
    provider = b"provider"; compatible = b'{"framework":"pytorch"}'
    (root / "config.json").write_bytes(provider)
    lock = {
        "schema_version": 2,
        "model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "provider": {"name": "huggingface", "repository": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "revision": "a" * 40},
        "runtime": {"python": "3.13", "scipy": "1.15.3", "lancedb": "0.34.0"},
        "provider_files": [{"path": "config.json", "sha256": hashlib.sha256(provider).hexdigest()}],
        "local_compatible_metadata": {
            "path": "configuration.json", "sha256": hashlib.sha256(compatible).hexdigest(),
            "provenance": {"provider": "modelscope", "model_id": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "kind": "legacy-local-bootstrap"},
        },
    }
    lock["record_self_sha256"] = manifests.canonical_sha256(lock)

    assert manifests.validate_model_tree(root, lock) == lock
    (root / "configuration.json").write_bytes(compatible)
    assert manifests.validate_model_tree(root, lock) == lock
    (root / "unexpected.json").write_text("no", encoding="utf-8")
    with pytest.raises(ValueError, match="missing, extra, or changed"):
        manifests.validate_model_tree(root, lock)
    (root / "unexpected.json").unlink()
    (root / "configuration.json").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="missing, extra, or changed"):
        manifests.validate_model_tree(root, lock)
    (root / "configuration.json").unlink()
    (root / "configuration.json").symlink_to(root / "config.json")
    with pytest.raises(ValueError, match="symlink"):
        manifests.validate_model_tree(root, lock)
