"""Small static guardrails for recurring cross-platform defects.

Behavior belongs in the feature tests. These checks only prevent new code from
bypassing the reviewed subprocess and durability boundaries.
"""
from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_ROOTS = (ROOT / "eval", ROOT / "scripts")
SUBPROCESS_CALLS = {"run", "Popen", "check_output", "check_call"}
APPROVED_FSYNC_BOUNDARIES = {
    "eval/phase07_operator_gate.py": {
        "_write_canonical_frozen_archive", "_fsync_directory", "_write_new_durable_ledger",
    },
    "scripts/obsidian_wiki/application/build_lock.py": {"acquire"},
    "scripts/obsidian_wiki/application/durable_filesystem.py": {
        "_fsync_dir", "atomic_write_bytes",
    },
    "scripts/obsidian_wiki/infrastructure/lancedb_index_repository.py": {
        "seal",
    },
}


def _production_python_files() -> list[Path]:
    return sorted(path for root in PRODUCTION_ROOTS for path in root.rglob("*.py"))


def _keyword(call: ast.Call, name: str) -> ast.expr | None:
    return next((item.value for item in call.keywords if item.arg == name), None)


def _is_literal(node: ast.expr | None, value: object) -> bool:
    return isinstance(node, ast.Constant) and node.value == value


class _Bindings(ast.NodeVisitor):
    """Collect names bound in one lexical scope, excluding nested scopes."""

    def __init__(self) -> None:
        self.names: set[str] = set()

    def _target(self, node: ast.AST) -> None:
        if isinstance(node, ast.Name):
            self.names.add(node.id)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for item in node.elts:
                self._target(item)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.names.add(node.name)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.add(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._target(target)
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._target(node.target)
        if node.value is not None:
            self.visit(node.value)

    def visit_For(self, node: ast.For) -> None:
        self._target(node.target)
        self.generic_visit(node)


def _local_bindings(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    bindings = _Bindings()
    for argument in [
        *getattr(node.args, "posonlyargs", []), *node.args.args, *node.args.kwonlyargs,
    ]:
        bindings.names.add(argument.arg)
    if node.args.vararg:
        bindings.names.add(node.args.vararg.arg)
    if node.args.kwarg:
        bindings.names.add(node.args.kwarg.arg)
    for statement in node.body:
        bindings.visit(statement)
    return bindings.names


class _SubprocessTextVisitor(ast.NodeVisitor):
    def __init__(self, tree: ast.Module) -> None:
        self.module_aliases: set[str] = set()
        self.call_aliases: set[str] = set()
        for statement in tree.body:
            if isinstance(statement, ast.Import):
                self.module_aliases.update(
                    alias.asname or alias.name for alias in statement.names
                    if alias.name == "subprocess"
                )
            elif isinstance(statement, ast.ImportFrom) and statement.module == "subprocess":
                self.call_aliases.update(
                    alias.asname or alias.name for alias in statement.names
                    if alias.name in SUBPROCESS_CALLS
                )
        self.shadowed: list[set[str]] = [set()]
        self.violations: list[int] = []

    def _available(self, name: str) -> bool:
        return not any(name in scope for scope in self.shadowed)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.shadowed.append(_local_bindings(node))
        for statement in node.body:
            self.visit(statement)
        self.shadowed.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        recognized = (
            isinstance(node.func, ast.Attribute)
            and node.func.attr in SUBPROCESS_CALLS
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in self.module_aliases
            and self._available(node.func.value.id)
        ) or (
            isinstance(node.func, ast.Name)
            and node.func.id in self.call_aliases
            and self._available(node.func.id)
        )
        if recognized:
            text_mode = (
                _is_literal(_keyword(node, "text"), True)
                or _is_literal(_keyword(node, "universal_newlines"), True)
                or _keyword(node, "encoding") is not None
            )
            if text_mode and not (
                _is_literal(_keyword(node, "encoding"), "utf-8")
                and _is_literal(_keyword(node, "errors"), "strict")
            ):
                self.violations.append(node.lineno)
        self.generic_visit(node)


def _subprocess_text_violations(source: str) -> list[int]:
    tree = ast.parse(source)
    visitor = _SubprocessTextVisitor(tree)
    visitor.visit(tree)
    return visitor.violations


def test_captured_subprocess_text_declares_utf8_strict_policy() -> None:
    violations: list[str] = []
    for path in _production_python_files():
        for line in _subprocess_text_violations(path.read_text(encoding="utf-8")):
            violations.append(f"{path.relative_to(ROOT)}:{line}")
    assert not violations, (
        "subprocess text must set encoding='utf-8', errors='strict': "
        + ", ".join(violations)
    )


def test_subprocess_gate_covers_alias_encoding_only_and_shadowing() -> None:
    source = """
import subprocess as process
from subprocess import run as execute
process.run(['bad'], encoding='utf-8')
execute(['bad'], text=True)
process.run(['safe'], encoding='utf-8', errors='strict')
def shadowed(process, execute):
    process.run(['not subprocess'], text=True)
    execute(['not subprocess'], text=True)
"""
    assert _subprocess_text_violations(source) == [4, 5]


class _FsyncBoundaryVisitor(ast.NodeVisitor):
    def __init__(self, approved: set[str]) -> None:
        self.approved = approved
        self.functions: list[str] = []
        self.violations: list[int] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.functions.append(node.name)
        self.generic_visit(node)
        self.functions.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Call(self, node: ast.Call) -> None:
        is_fsync = (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "os"
            and node.func.attr == "fsync"
        )
        if is_fsync and (not self.functions or self.functions[-1] not in self.approved):
            self.violations.append(node.lineno)
        self.generic_visit(node)


def _unapproved_fsync_calls(source: str, approved: set[str]) -> list[int]:
    visitor = _FsyncBoundaryVisitor(approved)
    visitor.visit(ast.parse(source))
    return visitor.violations


def test_fsync_calls_remain_inside_reviewed_durability_boundaries() -> None:
    violations: list[str] = []
    for path in _production_python_files():
        relative = path.relative_to(ROOT).as_posix()
        for line in _unapproved_fsync_calls(
            path.read_text(encoding="utf-8"), APPROVED_FSYNC_BOUNDARIES.get(relative, set()),
        ):
            violations.append(f"{relative}:{line}")
    assert not violations, "new fsync bypasses a reviewed durability boundary: " + ", ".join(violations)


def test_fsync_boundary_gate_rejects_new_direct_call() -> None:
    source = """
import os
def approved(fd):
    os.fsync(fd)
def new_unreviewed(fd):
    os.fsync(fd)
"""
    assert _unapproved_fsync_calls(source, {"approved"}) == [6]


def test_production_code_does_not_use_signal_zero_as_liveness_probe() -> None:
    violations: list[str] = []
    for path in _production_python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or len(node.args) < 2:
                continue
            is_os_kill = (
                isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "os"
                and node.func.attr == "kill"
            )
            if is_os_kill and _is_literal(node.args[1], 0):
                violations.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert not violations, (
        "os.kill(pid, 0) is not a cross-platform liveness probe: " + ", ".join(violations)
    )
