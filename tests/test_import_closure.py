"""The public interpreter has no private service dependency."""

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PURE_DEPENDENCIES = {"pydantic", "jsonschema", "tiktoken"}


def test_interpreter_imports_are_closed():
    names = [
        line
        for line in (ROOT / "runtime_modules.txt").read_text().splitlines()
        if not line.startswith("#")
    ]
    allowed = set(names)
    for name in names:
        for node in ast.walk(ast.parse((ROOT / "seldon_runtime" / f"{name}.py").read_text())):
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    assert node.level == 1, (name, node.module)
                    assert node.module in allowed, (name, node.module)
                    continue
                targets = [node.module or ""]
            else:
                continue
            for target in targets:
                assert target.split(".")[0] in sys.stdlib_module_names | PURE_DEPENDENCIES, (
                    name,
                    target,
                )
