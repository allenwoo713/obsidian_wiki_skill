"""Public build-mode contract gates for online incremental indexing."""

from __future__ import annotations

import inspect

from build_index import WikiIndex


def test_public_facade_exposes_explicit_incremental_mode() -> None:
    """The public facade must expose online incremental as a distinct request."""
    parameters = inspect.signature(WikiIndex.build).parameters

    assert parameters["build_mode"].default == "snapshot"
    assert "build_mode_policy_path" in parameters
