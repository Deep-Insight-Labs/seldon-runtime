"""Local model provider: serve GLiNER ``extract`` steps from open weights on this machine.

``LocalModels(...).provider`` is the ``provider(model, payload)`` the executor takes.
Run it under ``execute(..., serving=True)``: only serving treats the pre-dispatch
refusal ``primitive_adapter_unavailable`` as fallback-eligible, as hosted serving does.

Requests are admitted and built exactly as hosted: the same context fit
(``output_budget.resolve``, so ``token_limit``/``context_unverified`` match) and the
same Fastino request body (``gliner_normalize.request``). gliner2's raw output is shaped
like Fastino's response and runs through the same ``normalize``, so grounding, label
checks, null policy and schema validation match. ``torch``, ``gliner2`` and
``huggingface_hub`` are imported lazily, at call time.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from .gliner_normalize import normalize, request
from .local_models import LOCAL_MODELS
from .output_budget import REQUEST, resolve

UNAVAILABLE = "primitive_adapter_unavailable"
IDENTITY_CHANGED = "model_identity_changed"
# ponytail: fixed back-off after a failed load; make it configurable if 60s bites.
RETRY_AFTER = 60.0

# Loaded extractors per process, keyed by (model id, hf_revision); shared by instances.
_LOADED: dict[tuple[str, str], object] = {}
# Failed loads: key -> (monotonic retry time, refusal code). Calls before then refuse.
_FAILED: dict[tuple[str, str], tuple[float, str]] = {}
# ponytail: one lock for every load; per-key locks if several local models ever load at once.
_LOCK = threading.Lock()


def load_gliner2(entry):
    """Pinned weights from the Hugging Face cache/hub, fp32 on CPU.

    Every file is checked against its pinned sha256 before loading, so a different
    revision or edited bytes under the pinned snapshot raise ``model_identity_changed``.
    """
    import torch
    from gliner2 import AutoExtractor
    from huggingface_hub import snapshot_download

    path = Path(
        snapshot_download(
            entry["hf_repo"],
            revision=entry["hf_revision"],
            allow_patterns=["*.json", "*.safetensors"],
        )
    )
    for name, digest in entry["files"].items():
        with open(path / name, "rb") as f:
            if hashlib.file_digest(f, "sha256").hexdigest() != digest:
                raise ValueError(IDENTITY_CHANGED)
    return AutoExtractor.from_pretrained(str(path)).to("cpu", torch.float32)


def _target(model, entry):
    return SimpleNamespace(
        upstream_model_name=model,
        context_length=entry["context_length"],
        max_output_tokens=entry["max_output_tokens"],
    )


def _wire_body(payload, target):
    return request(target, payload)[1]


def _no_estimator(_target):
    return None  # as hosted: no published GLiNER token counter is registered


class LocalModels:
    def __init__(self, models=LOCAL_MODELS, *, loader=load_gliner2):
        self._models = models
        self._loader = loader

    async def provider(self, model, payload):
        entry = self._models.get(model)
        primitive = payload.get("primitive")
        if (
            entry is None
            or entry.get("loader") != "gliner2"
            or not isinstance(primitive, dict)
            or primitive.get("kind") != "extract"
        ):
            raise ValueError(UNAVAILABLE)
        target = _target(model, entry)
        # Hosted transport order: context fit, then the request body (dispatch).
        resolve(payload, target, REQUEST, body=_wire_body, estimator_for=_no_estimator)
        body = _wire_body(payload, target)
        extractor = await asyncio.to_thread(self._extractor, model, entry)
        raw = await asyncio.to_thread(_infer, extractor, entry, payload, body)
        return normalize(payload, raw)

    def _extractor(self, model, entry):
        key = (model, entry["hf_revision"])
        with _LOCK:
            if key in _LOADED:
                return _LOADED[key]
            failed = _FAILED.get(key)
            if failed and time.monotonic() < failed[0]:
                raise ValueError(failed[1])
            try:
                _LOADED[key] = self._loader(entry)
            except Exception as exc:
                identity = isinstance(exc, ValueError) and str(exc) == IDENTITY_CHANGED
                code = IDENTITY_CHANGED if identity else UNAVAILABLE
                _FAILED[key] = (time.monotonic() + RETRY_AFTER, code)
                raise ValueError(code) from exc
            _FAILED.pop(key, None)
            return _LOADED[key]


def _infer(extractor, entry, payload, body):
    """Run gliner2 on the Fastino request ``body``; return Fastino-shaped raw output."""
    mode = payload["primitive"]["config"].get("mode")
    schema = body["schema"]
    if mode in {"spans", "records"}:
        text = body["messages"][0]["content"]
    else:
        # ponytail: Fastino does not document how it flattens several messages; parity
        # of this join is unmeasured (work log, Slice 2 review). Measure in Slice 5.
        text = "\n".join(message["content"] for message in body["messages"])
    if mode == "spans":
        schema = {e["name"]: e["description"] for e in schema["entities"]}
        run = extractor.extract_entities
    elif mode == "records":
        schema = schema["structures"]
        run = extractor.extract_json
    elif "tasks" in schema:  # constrained classification (label_descriptions)
        run = functools.partial(_classify_constrained, extractor)
    else:
        schema = {task.pop("task"): task for task in schema["classifications"]}
        run = extractor.classify_text
    try:
        raw = run(
            text, schema, threshold=entry["threshold"], include_spans=True, include_confidence=True
        )
    except Exception as exc:
        raise ValueError("provider_unavailable") from exc
    if mode == "spans":
        return raw
    if mode == "records":
        # gliner2 adds a record-level "confidence"; Fastino records carry only fields.
        return {
            name: [
                {k: v for k, v in record.items() if k in {f["name"] for f in fields}}
                for record in raw.get(name, [])
            ]
            for name, fields in schema.items()
        }
    return {"choices": [{"message": {"content": json.dumps(raw)}}]}


def _classify_constrained(extractor, text, schema, **_options):
    """Fastino constrained classification (``_meta.feasible`` result) from local gliner2.

    Hosted sends no span/threshold options here; gliner2's defaults include confidence.
    """
    from gliner2.classification import ClassificationSchema, Classifier

    return Classifier(extractor).classify(text, ClassificationSchema.from_dict(schema)).to_dict()
