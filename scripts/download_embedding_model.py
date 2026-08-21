"""Download the skill's embedding model into the skill-local models directory.

This is intentionally an explicit bootstrap step.  Index builds must use an
already-local model so a production build never silently changes provider or
pollutes a user-wide Hugging Face/ModelScope cache.
"""
from __future__ import annotations

import shutil
import argparse
import json
import sys
from pathlib import Path, PurePosixPath


SKILL_ROOT = Path(__file__).resolve().parent.parent
if str(SKILL_ROOT) not in sys.path:
    sys.path.insert(0, str(SKILL_ROOT))
MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
MODEL_DIR = SKILL_ROOT / "models" / "paraphrase-multilingual-MiniLM-L12-v2"
CACHE_DIR = SKILL_ROOT / ".cache" / "huggingface"


def _model_is_complete(path: Path) -> bool:
    return (path / "model.safetensors").is_file()


def validate_manifest_only() -> dict[str, object]:
    """Validate the immutable manifest without reading or changing a model tree."""
    from eval.ann_corpus_manifest import load_manifest

    lock = load_manifest(SKILL_ROOT / "eval" / "model-manifest.json")
    if set(lock) != {"schema_version", "model_id", "revision", "runtime", "files", "record_self_sha256"}:
        raise ValueError("model manifest schema")
    if lock["schema_version"] != 1 or lock["model_id"] != MODEL_ID:
        raise ValueError("model manifest identity")
    revision = lock["revision"]
    if not isinstance(revision, str) or len(revision) != 40 or set(revision) - set("0123456789abcdef") or revision == "0" * 40:
        raise ValueError("model revision must be an immutable provider commit")
    if lock["runtime"] != {"python": "3.13", "scipy": "1.15.3", "lancedb": "0.34.0"}:
        raise ValueError("model manifest runtime")
    files = lock["files"]
    if not isinstance(files, list) or not files:
        raise ValueError("model manifest file allowlist")
    paths: set[str] = set()
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError("model manifest file record")
        path, digest = item["path"], item["sha256"]
        if (
            not isinstance(path, str) or not path or PurePosixPath(path).is_absolute()
            or ".." in PurePosixPath(path).parts or "\\" in path or ":" in path
            or path in paths or not isinstance(digest, str) or len(digest) != 64
            or set(digest) - set("0123456789abcdef")
        ):
            raise ValueError("model manifest file allowlist")
        paths.add(path)
    return lock


def validate_model_tree_only() -> None:
    """Validate the pinned tree without a network or filesystem hydration path."""
    from eval.ann_corpus_manifest import load_manifest, validate_model_tree
    lock = load_manifest(SKILL_ROOT / "eval" / "model-manifest.json")
    validate_model_tree(MODEL_DIR, lock, allow_download=False)


def hydrate_exact_manifest_model(*, snapshot_download=None) -> None:
    """Hydrate only the manifest's immutable HF revision, then verify every file."""
    from eval.ann_corpus_manifest import load_manifest, validate_model_tree
    lock = load_manifest(SKILL_ROOT / "eval" / "model-manifest.json")
    revision = lock["revision"]
    allow_patterns = [item["path"] for item in lock["files"]]
    if snapshot_download is None:
        from huggingface_hub import snapshot_download as download
    else:
        download = snapshot_download
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = Path(download(
        repo_id=MODEL_ID, revision=revision, allow_patterns=allow_patterns,
        cache_dir=str(CACHE_DIR), local_files_only=False,
    ))
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)
    MODEL_DIR.mkdir()
    for relative in allow_patterns:
        source, target = downloaded / relative, MODEL_DIR / relative
        if not source.is_file():
            raise RuntimeError(f"immutable provider snapshot omitted {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    validate_model_tree(MODEL_DIR, lock, allow_download=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-model-tree", action="store_true")
    parser.add_argument("--validate-manifest-only", action="store_true")
    args = parser.parse_args()
    if args.validate_manifest_only:
        lock = validate_manifest_only()
        print(json.dumps({"valid": True, "mode": "manifest-only", "revision": lock["revision"]}, sort_keys=True))
        return
    if args.validate_model_tree:
        validate_model_tree_only()
        print(json.dumps({"valid": True, "model_dir": str(MODEL_DIR)}, sort_keys=True))
        return
    if _model_is_complete(MODEL_DIR):
        print(f"embedding model already available: {MODEL_DIR}")
        return

    hydrate_exact_manifest_model()
    print(f"embedding model deployed: {MODEL_DIR}")


if __name__ == "__main__":
    main()
