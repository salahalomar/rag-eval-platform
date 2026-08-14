"""Enforces ENGINEERING.md's hard rule: packages/rag must never import from apps/.

Written as a test rather than left as a convention because the rule is what makes
"the eval harness calls the same code path the API calls" checkable. Once the library
can reach into the web layer, that claim quietly stops being true.
"""

import ast
from pathlib import Path

import pytest

RAG_SRC = Path(__file__).resolve().parents[1] / "packages" / "rag" / "src" / "rag"

FORBIDDEN_ROOTS = {"api", "apps", "fastapi", "starlette", "uvicorn"}


def module_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def python_files() -> list[Path]:
    return sorted(RAG_SRC.rglob("*.py"))


def test_the_library_has_source_files() -> None:
    assert python_files(), f"no python files found under {RAG_SRC}"


@pytest.mark.parametrize("path", python_files(), ids=lambda p: str(p.name))
def test_library_does_not_import_the_web_layer(path: Path) -> None:
    offending = module_imports(path) & FORBIDDEN_ROOTS
    assert not offending, f"{path} imports {sorted(offending)}; packages/rag must stay standalone"
