"""Two-stage ANN calibration workflow contracts."""

from pathlib import Path


WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "eval.yml"


def test_manual_calibration_isolated_from_pull_request_acceptance() -> None:
    """Calibration is dispatch-only; PR acceptance needs committed static inputs."""
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "workflow_dispatch:" in workflow
    assert "github.event_name == 'workflow_dispatch'" in workflow
    assert "github.event_name == 'pull_request'" in workflow
    assert "--approved-static-cap" in workflow
    assert "--approved-calibration-sha256" in workflow


def test_manual_calibration_persists_actual_observational_batch_artifact() -> None:
    workflow = WORKFLOW.read_text(encoding="utf-8")

    assert "--calibration-batch-output" in workflow
    assert "issue41-ann-calibration-batch-spike" in workflow
    assert "index-benchmark.json" not in workflow.split("workflow_dispatch", 1)[1].split(
        "pull_request", 1
    )[0]
