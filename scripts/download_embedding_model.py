"""Download the skill's embedding model into the skill-local models directory.

This is intentionally an explicit bootstrap step.  Index builds must use an
already-local model so a production build never silently changes provider or
pollutes a user-wide Hugging Face/ModelScope cache.
"""
from __future__ import annotations

import os
import shutil
import stat
import tempfile
import uuid
import argparse
import json
import sys
from pathlib import Path
from pathlib import PurePosixPath


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
    from eval.ann_corpus_manifest import load_manifest, validate_model_manifest

    lock = load_manifest(SKILL_ROOT / "eval" / "model-manifest.json")
    validate_model_manifest(lock)
    return lock


def validate_model_tree_only(*, model_dir: Path = MODEL_DIR) -> None:
    """Validate the pinned tree without a network or filesystem hydration path."""
    from eval.ann_corpus_manifest import load_manifest, validate_model_tree
    lock = load_manifest(SKILL_ROOT / "eval" / "model-manifest.json")
    validate_model_tree(model_dir, lock, allow_download=False)


def _require_plain_provider_directory(path: Path) -> None:
    """Reject links and non-directories without resolving an untrusted path."""
    try:
        mode = os.lstat(path).st_mode
    except OSError as exc:
        raise RuntimeError(f"unsafe provider snapshot directory {path}") from exc
    if not stat.S_ISDIR(mode):
        raise RuntimeError(f"unsafe provider snapshot directory {path}")


def _require_plain_provider_file(root: Path, relative: str) -> Path:
    """Walk provider-owned components via lstat so no link is ever followed."""
    _require_plain_provider_directory(root)
    parts = PurePosixPath(relative).parts
    current = root
    for component in parts[:-1]:
        current = current / component
        _require_plain_provider_directory(current)
    source = current / parts[-1]
    try:
        mode = os.lstat(source).st_mode
    except OSError as exc:
        raise RuntimeError(f"immutable provider snapshot omitted {relative}") from exc
    if not stat.S_ISREG(mode):
        raise RuntimeError(f"unsafe provider snapshot file {relative}")
    return source


def hydrate_exact_manifest_model(*, snapshot_download=None) -> None:
    """Stage and seal only immutable HF provider files before replacing the target."""
    from eval.ann_corpus_manifest import load_manifest, sha256_file, validate_model_manifest, validate_model_tree
    lock = load_manifest(SKILL_ROOT / "eval" / "model-manifest.json")
    manifest = validate_model_manifest(lock)
    revision = lock["provider"]["revision"]
    allow_patterns = list(manifest["provider_files"])
    if snapshot_download is None:
        from huggingface_hub import snapshot_download as download
    else:
        download = snapshot_download
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)
    target_parent = MODEL_DIR.parent.resolve()
    provider_root = Path(tempfile.mkdtemp(prefix=f".{MODEL_DIR.name}.provider-", dir=target_parent))
    provider_local_dir = provider_root / "snapshot"
    provider_local_dir.mkdir()
    staging = Path(tempfile.mkdtemp(prefix=f".{MODEL_DIR.name}.staging-", dir=target_parent))
    backup: Path | None = None
    try:
        downloaded = Path(download(
            repo_id=MODEL_ID, revision=revision, allow_patterns=allow_patterns,
            cache_dir=str(CACHE_DIR), local_dir=str(provider_local_dir), local_files_only=False,
        ))
        if downloaded != provider_local_dir:
            raise RuntimeError("provider snapshot returned unexpected root")
        _require_plain_provider_directory(provider_local_dir)
        for relative, expected_digest in manifest["provider_files"].items():
            source, target = _require_plain_provider_file(provider_local_dir, relative), staging / relative
            if sha256_file(source) != expected_digest:
                raise ValueError(f"immutable provider snapshot changed file {relative}")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            if not stat.S_ISREG(os.lstat(target).st_mode):
                raise RuntimeError(f"unsafe staged provider file {relative}")
        validate_model_tree(staging, lock, allow_download=False)
        if MODEL_DIR.exists():
            backup = MODEL_DIR.parent / f".{MODEL_DIR.name}.previous-{uuid.uuid4().hex}"
            os.replace(MODEL_DIR, backup)
        try:
            os.replace(staging, MODEL_DIR)
        except BaseException:
            if backup is not None and backup.exists():
                os.replace(backup, MODEL_DIR)
                backup = None
            raise
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
    finally:
        if staging.exists():
            shutil.rmtree(staging)
        if provider_root.exists():
            shutil.rmtree(provider_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate-model-tree", action="store_true")
    parser.add_argument("--validate-manifest-only", action="store_true")
    parser.add_argument("--model-dir", type=Path)
    args = parser.parse_args()
    if args.model_dir is not None and not args.validate_model_tree:
        parser.error("--model-dir requires --validate-model-tree")
    if args.validate_manifest_only:
        lock = validate_manifest_only()
        print(json.dumps({"valid": True, "mode": "manifest-only", "revision": lock["provider"]["revision"]}, sort_keys=True))
        return
    if args.validate_model_tree:
        model_dir = args.model_dir if args.model_dir is not None else MODEL_DIR
        validate_model_tree_only(model_dir=model_dir)
        print(json.dumps({"valid": True, "model_dir": str(model_dir)}, sort_keys=True))
        return
    if MODEL_DIR.exists():
        validate_model_tree_only()
        print(f"embedding model already available: {MODEL_DIR}")
        return

    hydrate_exact_manifest_model()
    print(f"embedding model deployed: {MODEL_DIR}")


if __name__ == "__main__":
    main()
