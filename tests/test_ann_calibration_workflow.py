"""Two-stage ANN calibration workflow contracts."""

from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "eval.yml"


def test_manual_calibration_isolated_from_pull_request_acceptance() -> None:
    """Calibration is dispatch-only; PR acceptance needs committed static inputs."""
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert "github.event_name == 'workflow_dispatch'" in workflow
    assert "github.event_name == 'pull_request'" in workflow
    acceptance = workflow.split("issue41-scale-benchmark:", 1)[1].split("model-backed-ann-decision:", 1)[0]
    assert "--approved-static-cap" in acceptance
    assert "--approved-calibration-sha256" in acceptance
    assert "--calibrate" not in acceptance


def test_manual_calibration_persists_actual_observational_batch_artifact() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    calibration = workflow.split("issue41-manual-calibration:", 1)[1].split(
        "issue41-scale-benchmark:", 1
    )[0]
    assert "--calibration-batch-output" in calibration
    assert "issue41-ann-calibration-batch-spike" in calibration
    assert "eval/index-benchmark.json" not in calibration


def test_manual_calibration_has_ten_run_timeout_without_relaxing_pr_acceptance() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")
    calibration = workflow.split("issue41-manual-calibration:", 1)[1].split(
        "issue41-scale-benchmark:", 1
    )[0]
    acceptance = workflow.split("issue41-scale-benchmark:", 1)[1].split(
        "model-backed-ann-decision:", 1
    )[0]

    assert "timeout-minutes: 30" in calibration
    assert "timeout-minutes: 15" in acceptance
    assert "--max-seconds 60" in acceptance
