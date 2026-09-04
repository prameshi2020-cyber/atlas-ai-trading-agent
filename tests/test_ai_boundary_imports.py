"""Architectural guard: the AI layer must never reach the trading stack.

ATLAS core principle: deterministic systems are authoritative; AI is advisory.
This test turns that rule into an executable contract. Every module under the
guarded packages is parsed with ``ast`` and the test fails if any of them
imports a forbidden package -- via absolute import, relative import, or a
string literal passed to ``importlib.import_module`` / ``__import__``.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = REPO_ROOT / "app"

# Packages whose modules are inspected.
GUARDED_PACKAGES: tuple[str, ...] = ("app.agents", "app.decision")

# Packages the AI layer may never import (directly or transitively by name).
FORBIDDEN_PACKAGES: frozenset[str] = frozenset(
    {
        "app.trading",  # broker, execution, positions, orders, journal, runner
        "app.risk",  # risk manager, sizing, leverage
        "app.strategy",  # live StrategyEngine / stop policy
    }
)

# Cross-layer rules: the AI layer may not reach the deployment gate and the gate
# may not reach the AI layer. Deterministic code decides; AI only advises upstream.
CROSS_LAYER_FORBIDDEN: dict[str, frozenset[str]] = {
    "app.agents": frozenset({"app.decision"}),
    "app.decision": frozenset({"app.agents"}),
}

def _module_name(path: Path) -> str:
    parts = list(path.relative_to(REPO_ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _package_of(module_name: str, path: Path) -> str:
    return module_name if path.name == "__init__.py" else module_name.rpartition(".")[0]


def _resolve_relative(package: str, level: int, module: str | None) -> str:
    base_parts = package.split(".")
    if level > 1:
        base_parts = base_parts[: len(base_parts) - (level - 1)]
    base = ".".join(base_parts)
    if module:
        return f"{base}.{module}" if base else module
    return base


def _is_forbidden(name: str, extra: frozenset[str] = frozenset()) -> bool:
    return any(name == pkg or name.startswith(pkg + ".") for pkg in FORBIDDEN_PACKAGES | extra)


def _imported_names(tree: ast.AST, package: str) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_relative(package, node.level, node.module) if node.level else (node.module or "")
            found.append((node.lineno, base))
            for alias in node.names:
                found.append((node.lineno, f"{base}.{alias.name}" if base else alias.name))
        elif isinstance(node, ast.Call):
            func = node.func
            dynamic = (isinstance(func, ast.Name) and func.id == "__import__") or (
                isinstance(func, ast.Attribute) and func.attr == "import_module"
            )
            if dynamic and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                found.append((node.lineno, node.args[0].value))
    return found


def _guarded_files() -> list[Path]:
    files: list[Path] = []
    for pkg in GUARDED_PACKAGES:
        pkg_dir = APP_ROOT.joinpath(*pkg.split(".")[1:])
        assert pkg_dir.is_dir(), f"guarded package directory missing: {pkg_dir}"
        files.extend(sorted(p for p in pkg_dir.rglob("*.py") if "__pycache__" not in p.parts))
    assert files, "no python files under guarded packages; the guard would be vacuous"
    return files


@pytest.mark.parametrize("path", _guarded_files(), ids=lambda p: p.relative_to(REPO_ROOT).as_posix())
def test_ai_layer_does_not_import_trading_stack(path: Path) -> None:
    module = _module_name(path)
    guarded = next(pkg for pkg in GUARDED_PACKAGES if module == pkg or module.startswith(pkg + "."))
    extra = CROSS_LAYER_FORBIDDEN.get(guarded, frozenset())

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    package = _package_of(module, path)

    violations = [
        f"{path.relative_to(REPO_ROOT).as_posix()}:{lineno} imports {name!r}"
        for lineno, name in _imported_names(tree, package)
        if _is_forbidden(name, extra)
    ]
    assert not violations, "AI layer must never reach the trading stack:\n" + "\n".join(violations)


def test_forbidden_packages_actually_exist() -> None:
    # Guard the guard: if a forbidden package is renamed, this must be updated
    # rather than silently protecting nothing.
    for pkg in sorted(FORBIDDEN_PACKAGES):
        assert APP_ROOT.joinpath(*pkg.split(".")[1:]).is_dir(), f"forbidden package not found: {pkg}"


def test_detector_flags_every_forbidden_import_form() -> None:
    src = (
        "import app.trading.broker\n"
        "from app.risk import manager\n"
        "from ..trading import execution\n"
        "from app import strategy\n"
        "importlib.import_module('app.strategy.engine')\n"
        "__import__('app.trading')\n"
    )
    names = [name for _, name in _imported_names(ast.parse(src), package="app.agents")]
    flagged = {name for name in names if _is_forbidden(name)}

    assert "app.trading.broker" in flagged
    assert "app.risk.manager" in flagged
    assert "app.trading.execution" in flagged
    assert "app.strategy" in flagged
    assert "app.strategy.engine" in flagged
    assert "app.trading" in flagged


def test_detector_allows_deterministic_analysis_imports() -> None:
    src = (
        "from app.analysis.ranking import CandidateScore\n"
        "from app.agents.llm_client import LLMClient\n"
        "from app.config.settings import settings\n"
        "import app.analysis.research\n"
    )
    names = [name for _, name in _imported_names(ast.parse(src), package="app.agents")]
    assert not any(_is_forbidden(name) for name in names)