import importlib
import shutil
import sys
from pathlib import Path


def test_staged_tree_normalizes_gliner_spans(tmp_path, monkeypatch):
    shutil.copytree(Path(__file__).resolve().parents[1] / "seldon_runtime", tmp_path / "staged_rt")
    (tmp_path / "staged_rt" / "__init__.py").touch()
    monkeypatch.syspath_prepend(str(tmp_path))
    try:
        gliner = importlib.import_module("staged_rt.gliner_normalize")
        payload = {
            "primitive": {
                "kind": "extract",
                "source": "Ada met Bob.",
                "config": {"mode": "spans", "labels": {"person": "a person"}},
            }
        }
        raw = {"result": {"entities": {"person": [{"text": "Bob", "start": 8, "end": 11}]}}}
        spans = gliner.normalize(payload, raw)
        assert [(s["label"], s["text"], s["start"], s["end"]) for s in spans] == [
            ("person", "Bob", 8, 11)
        ]
    finally:
        for name in [n for n in sys.modules if n.startswith("staged_rt")]:
            del sys.modules[name]
