"""Run #65 acceptance without installing dependencies or modifying baselines."""
from __future__ import annotations
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--with-model-eval", action="store_true")
    p.add_argument("--base-ref", help="PR base SHA for the baseline-change guard")
    args = p.parse_args()
    repo = Path(__file__).resolve().parents[1]
    baseline = repo / "eval" / "baselines.json"
    baseline_before = hashlib.sha256(baseline.read_bytes()).hexdigest()
    records = []
    output = repo / ".review-tmp" / "issue65-acceptance.json"
    output.parent.mkdir(exist_ok=True)

    def run(command, *, cwd=repo, timeout=3600):
        print("+", " ".join(map(str, command)), flush=True)
        started = time.perf_counter()
        result = subprocess.run(list(map(str, command)), cwd=cwd, timeout=timeout)
        records.append({"command": list(map(str, command)),
                        "exit_code": result.returncode,
                        "elapsed_seconds": round(time.perf_counter() - started, 3)})
        if result.returncode:
            raise RuntimeError(f"command failed with exit {result.returncode}")

    status = "fail"
    try:
        if args.base_ref:
            run(["git", "diff", "--exit-code", args.base_ref, "--", "eval/baselines.json"])
        run([sys.executable, "scripts/compile_requirements.py", "--check"])
        run([sys.executable, "-S", "tests/test_wiki_freshness_unit.py"])
        suites = [
            "tests/test_wiki_freshness_unit.py",
            "tests/test_index_freshness_integration.py",
            "tests/test_index_chunking_contract.py",
            "tests/test_index_safety_lock.py",
            "tests/test_index_durability.py",
            "tests/test_index_post_commit.py",
            "tests/test_lancedb_storage_contract.py",
            "tests/test_online_incremental.py",
            "tests/test_online_incremental_policy.py",
            "tests/test_online_incremental_cli.py",
            "tests/test_issue14_context_contract.py",
            "tests/test_community_reports.py",
            "tests/test_image_admission.py",
            "tests/test_query_result.py",
        ]
        run([sys.executable, "-m", "pytest", "-p", "no:cacheprovider",
             "--basetemp=.review-tmp/issue65-contract", *suites, "-q"])
        run(["lint-imports", "--config", "../.importlinter", "--no-cache"],
            cwd=repo / "scripts")
        run([sys.executable, "tests/test_architecture_foundation.py"])
        if args.with_model_eval:
            # Model must already be bootstrapped via the repository's existing step.
            run([sys.executable, "scripts/verify_issue65_e2e.py"], timeout=7200)
            run([sys.executable, "-m", "eval.compare_build_modes",
                 "--work-dir", ".review-tmp/issue65-modes",
                 "--output", ".review-tmp/issue65-modes/equivalence.json"], timeout=7200)
            run([sys.executable, "-m", "eval.run_eval",
                 "--work-dir", ".review-tmp/issue65-eval"], timeout=7200)
        if hashlib.sha256(baseline.read_bytes()).hexdigest() != baseline_before:
            raise RuntimeError("eval/baselines.json changed during acceptance")
        status = "pass"
    finally:
        evidence = {
            "status": status, "with_model_eval": args.with_model_eval,
            "baseline_before_sha256": baseline_before,
            "baseline_after_sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
            "commands": records,
        }
        output.write_text(json.dumps(evidence, ensure_ascii=False, indent=2),
                          encoding="utf-8")
        print(f"Acceptance evidence: {output}")


if __name__ == "__main__":
    main()
