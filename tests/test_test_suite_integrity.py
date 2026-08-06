"""Static integrity gates for tests whose duplicate names Python would silently overwrite."""
from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path


TESTS_DIR = Path(__file__).resolve().parent


def _duplicates(names: list[str]) -> list[str]:
    return sorted(name for name, count in Counter(names).items() if count > 1)


def test_no_duplicate_test_definitions_in_any_module():
    """#34/#37：顶层测试及 Test* 类方法不得重名，避免旧测试被静默覆盖。"""
    failures: list[str] = []
    for path in sorted(TESTS_DIR.glob("test_*.py")):
        module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        top_level = [
            node.name for node in module.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
        ]
        duplicates = _duplicates(top_level)
        if duplicates:
            failures.append(f"{path.name}: top-level {duplicates}")
        for node in module.body:
            if not isinstance(node, ast.ClassDef) or not node.name.startswith("Test"):
                continue
            methods = [
                child.name for child in node.body
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef))
                and child.name.startswith("test_")
            ]
            duplicates = _duplicates(methods)
            if duplicates:
                failures.append(f"{path.name}:{node.name} {duplicates}")
    assert not failures, "duplicate test definitions:\n" + "\n".join(failures)
