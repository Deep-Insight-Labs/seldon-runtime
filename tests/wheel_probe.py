"""Server-free probe for an installed ``seldon_runtime`` wheel (stdlib + the wheel only).

Imports every module in ``runtime_modules.txt`` plus the hand-written ``local`` and
``runner`` (which must import without torch/gliner2) and recomputes the plan / spec digests
against stored goldens, so dependency floors that change canonical JSON (e.g. pydantic
float serialization) fail loudly. Run from outside the source tree:

    python wheel_probe.py <runtime_modules.txt> <plan-fixtures-dir> <digest_goldens.json>
"""

import importlib
import json
import sys
from pathlib import Path


def main(modules_file: str, plans_dir: str, goldens_file: str) -> int:
    lines = Path(modules_file).read_text().splitlines()
    names = [n for n in (line.split("#")[0].strip() for line in lines) if n]
    mods = {n: importlib.import_module(f"seldon_runtime.{n}") for n in names}
    importlib.import_module("seldon_runtime.local")
    importlib.import_module("seldon_runtime.runner")
    contracts = mods["contracts"]
    plans = sorted(Path(plans_dir).glob("function-plan-v*.json"))
    specs = json.loads(Path(goldens_file).read_text())["specs"]
    if not plans or not specs:
        print(f"no fixtures found: plans={len(plans)} specs={len(specs)}")
        return 1

    bad = []
    for path in plans:
        stored = json.loads(path.read_text())
        if contracts.Plan.model_validate(stored["plan"]).digest != stored["digest"]:
            bad.append(path.name)
    for name, golden in specs.items():
        if contracts.FunctionSpec.model_validate(golden["spec"]).digest != golden["digest"]:
            bad.append(f"spec {name}")
    print(f"imported {len(names)} modules; checked {len(plans)} plans, {len(specs)} specs")
    if bad:
        print("DIGEST MISMATCH: " + ", ".join(bad))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
