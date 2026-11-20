"""AC-L2.x: ``LocalModels`` serves GLiNER ``extract`` steps from local weights.

Default tests replay recorded raw gliner2 outputs through a fake extractor (no torch).
The ``local_model`` test runs the pinned weights; with ``SELDON_RECORD_LOCAL=1`` it
rewrites ``fixtures/local_gliner_raw.json`` from the real model.
"""

import asyncio
import hashlib
import importlib
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

MODEL = "fastino/gliner2.5-multi-v1"
REVISION = "7b005274e82da37dcf1f5fd9d6a2ef413c6a004c"
FIXTURE = Path(__file__).parent / "fixtures" / "local_gliner_raw.json"
RECORDS_SCHEMA = {
    "type": "object",
    "properties": {
        "jobs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "full name of a person"},
                    "company": {"type": ["string", "null"], "description": "employer"},
                },
                "additionalProperties": False,
            },
        }
    },
    "additionalProperties": False,
}
SENTIMENT_SCHEMA = {
    "type": "object",
    "properties": {"sentiment": {"type": "string", "enum": ["positive", "negative"]}},
    "required": ["sentiment"],
    "additionalProperties": False,
}
DESCRIPTIONS = {
    "sentiment": {
        "positive": "the writer is happy with the product",
        "negative": "the writer is unhappy with the product",
    }
}
PAYLOADS = {
    "spans": {
        "messages": [{"role": "user", "content": "ignored for spans"}],
        "primitive": {
            "kind": "extract",
            "source": "Ada Lovelace met Bob Smith in London.",
            "config": {"mode": "spans", "labels": {"person": "a named person", "city": "a city"}},
        },
    },
    "records": {
        "messages": [{"role": "user", "content": "ignored for records"}],
        "primitive": {
            "kind": "extract",
            "source": "Ada Lovelace works at Acme Corp. Bob Smith works at Globex.",
            "config": {"mode": "records", "schema": RECORDS_SCHEMA},
        },
    },
    "classification": {
        "messages": [{"role": "user", "content": "I love this product, it works great!"}],
        "primitive": {"kind": "extract", "config": {"schema": SENTIMENT_SCHEMA}},
    },
    "constrained": {
        "messages": [{"role": "user", "content": "I love this product, it works great!"}],
        "primitive": {
            "kind": "extract",
            "config": {"schema": SENTIMENT_SCHEMA, "label_descriptions": DESCRIPTIONS},
        },
    },
}
EXPECTED = {
    "spans": [
        ("city", "London", 30, 36),
        ("person", "Ada Lovelace", 0, 12),
        ("person", "Bob Smith", 17, 26),
    ],
    "records": {
        "jobs": [
            {"name": "Ada Lovelace", "company": "Acme Corp"},
            {"name": "Bob Smith", "company": "Globex"},
        ]
    },
    "classification": {"sentiment": "positive"},
    "constrained": {"sentiment": "positive"},
}
CALLS = {
    "spans": "extract_entities",
    "records": "extract_json",
    "classification": "classify_text",
    "constrained": "classify_constrained",
}


def comparable(mode, output):
    if mode == "spans":
        return sorted((s["label"], s["text"], s["start"], s["end"]) for s in output)
    return output


@pytest.fixture(scope="module")
def local(tmp_path_factory):
    """Import an isolated copy so dependency doubles cannot leak between modules."""
    root = tmp_path_factory.mktemp("rt")
    pkg = root / "seldon_runtime"
    shutil.copytree(Path(__file__).resolve().parents[1] / "seldon_runtime", pkg)

    def ours():
        return [n for n in sys.modules if n == "seldon_runtime" or n.startswith("seldon_runtime.")]

    saved = {n: sys.modules.pop(n) for n in ours()}
    sys.path.insert(0, str(root))
    try:
        yield importlib.import_module("seldon_runtime.local")
    finally:
        sys.path.remove(str(root))
        for n in ours():
            del sys.modules[n]
        sys.modules.update(saved)


@pytest.fixture(autouse=True)
def fresh_cache(request):
    if "local" in request.fixturenames:
        module = request.getfixturevalue("local")
        module._LOADED.clear()
        module._FAILED.clear()


class FakeExtractor:
    """Replays recorded raw outputs; tokenizer counts whitespace words."""

    def __init__(self, raw):
        self.raw = raw
        self.calls = []
        self.processor = SimpleNamespace(
            tokenizer=SimpleNamespace(encode=lambda text, add_special_tokens: text.split())
        )

    def _replay(self, name, text, schema, **options):
        self.calls.append((name, text, schema, options))
        return json.loads(json.dumps(self.raw[name]))

    def extract_entities(self, text, schema, **options):
        return self._replay("extract_entities", text, schema, **options)

    def extract_json(self, text, schema, **options):
        return self._replay("extract_json", text, schema, **options)

    def classify_text(self, text, schema, **options):
        return self._replay("classify_text", text, schema, **options)


def recorded():
    return json.loads(FIXTURE.read_text())


def fake_loader(calls=None, delay=0.0):
    fixture = recorded()
    extractor = FakeExtractor({v["call"]: v["raw"] for v in fixture.values()})

    def load(entry):
        if calls is not None:
            calls.append(entry)
        time.sleep(delay)
        return extractor

    return load, extractor


def run(coro):
    return asyncio.run(coro)


def test_import_needs_no_torch_or_gliner2(local):
    entry = local.LOCAL_MODELS[MODEL]
    assert list(local.LOCAL_MODELS) == [MODEL]
    assert (entry["loader"], entry["hf_repo"], entry["hf_revision"]) == ("gliner2", MODEL, REVISION)
    # The hosted catalog limits for this model (GLINER_MAX_CONTEXT, card max_output_tokens).
    assert (entry["context_length"], entry["max_output_tokens"]) == (4096, 4096)
    assert set(entry["files"]) == {
        "config.json",
        "encoder_config/config.json",
        "model.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    for heavy in ("torch", "gliner2", "huggingface_hub"):
        assert heavy not in sys.modules


@pytest.mark.parametrize("mode", ["spans", "records", "classification"])
def test_recorded_raw_normalizes_like_hosted(local, mode):
    load, extractor = fake_loader()
    output = run(local.LocalModels(loader=load).provider(MODEL, PAYLOADS[mode]))
    assert comparable(mode, output) == EXPECTED[mode]
    (call, text, schema, options) = extractor.calls[-1]
    assert call == recorded()[mode]["call"]
    assert options == {"threshold": 0.5, "include_spans": True, "include_confidence": True}
    if mode == "spans":
        assert (text, schema) == (
            PAYLOADS[mode]["primitive"]["source"],
            PAYLOADS[mode]["primitive"]["config"]["labels"],
        )
    if mode == "records":
        assert schema == {
            "jobs": [
                {"name": "name", "dtype": "str", "description": "full name of a person"},
                {"name": "company", "dtype": "str", "description": "employer"},
            ]
        }
    if mode == "classification":
        assert text == "I love this product, it works great!"
        assert schema == {
            "sentiment": {"labels": ["positive", "negative"], "multi_label": False, "top_k": 1}
        }


def test_classification_joins_every_message_like_before(local):
    """Pins the unmeasured flattening: hosted sends the list to Fastino as is (work log)."""
    load, extractor = fake_loader()
    payload = json.loads(json.dumps(PAYLOADS["classification"]))
    payload["messages"].insert(0, {"role": "system", "content": "Classify sentiment."})
    run(local.LocalModels(loader=load).provider(MODEL, payload))
    assert extractor.calls[-1][1] == "Classify sentiment.\nI love this product, it works great!"


def test_constrained_classification_uses_gliner2_classifier(local, monkeypatch):
    load, extractor = fake_loader()
    seen = {}

    class Classifier:
        def __init__(self, model):
            seen["model"] = model

        def classify(self, text, schema):
            seen["call"] = (text, schema)
            return SimpleNamespace(to_dict=lambda: recorded()["constrained"]["raw"])

    classification = SimpleNamespace(
        Classifier=Classifier, ClassificationSchema=SimpleNamespace(from_dict=lambda d: d)
    )
    monkeypatch.setitem(sys.modules, "gliner2", SimpleNamespace(classification=classification))
    monkeypatch.setitem(sys.modules, "gliner2.classification", classification)
    output = run(local.LocalModels(loader=load).provider(MODEL, PAYLOADS["constrained"]))
    assert output == EXPECTED["constrained"]
    assert seen["model"] is extractor
    assert seen["call"] == (
        "I love this product, it works great!",
        {  # byte-for-byte the schema hosted request() sends
            "version": 3,
            "tasks": {
                "sentiment": {"labels": DESCRIPTIONS["sentiment"], "min_labels": 1, "max_labels": 1}
            },
            "constraints": [],
        },
    )
    infeasible = json.loads(json.dumps(recorded()["constrained"]["raw"]))
    infeasible["_meta"]["feasible"] = False
    Classifier.classify = lambda self, text, schema: SimpleNamespace(to_dict=lambda: infeasible)
    with pytest.raises(ValueError, match="^invalid_provider_output$"):
        run(local.LocalModels(loader=load).provider(MODEL, PAYLOADS["constrained"]))


@pytest.mark.parametrize("mode", ["spans", "records"])
@pytest.mark.parametrize(
    ("source", "code"),
    [
        (None, "invalid_primitive_source"),
        (42, "invalid_primitive_source"),
        # Hosted transport fits context before building the request, so this is token_limit.
        ("x" * 262145, "token_limit"),
    ],
    ids=["none", "int", "oversized"],
)
def test_invalid_source_is_refused_like_hosted(local, mode, source, code):
    load, extractor = fake_loader(calls := [])
    payload = json.loads(json.dumps(PAYLOADS[mode]))
    payload["primitive"]["source"] = source
    with pytest.raises(ValueError, match=f"^{code}$"):
        run(local.LocalModels(loader=load).provider(MODEL, payload))
    assert calls == [] and extractor.calls == []  # refused before loading weights


def test_model_call_failure_is_provider_unavailable(local):
    load, extractor = fake_loader()

    def boom(*args, **kwargs):
        raise RuntimeError("cuda? oom?")

    extractor.extract_entities = boom
    with pytest.raises(ValueError, match="^provider_unavailable$") as info:
        run(local.LocalModels(loader=load).provider(MODEL, PAYLOADS["spans"]))
    assert isinstance(info.value.__cause__, RuntimeError)


def test_records_without_matches_are_empty_collections(local):
    load, extractor = fake_loader()
    extractor.raw["extract_json"] = {}  # what gliner2 returns when nothing matches
    output = run(local.LocalModels(loader=load).provider(MODEL, PAYLOADS["records"]))
    assert output == {"jobs": []}


def test_grounding_and_label_checks_still_apply(local):
    load, extractor = fake_loader()
    extractor.raw["extract_entities"] = {
        "entities": {"person": [{"text": "Zed", "start": 0, "end": 3, "confidence": 0.9}]}
    }
    with pytest.raises(ValueError, match="invalid_grounded_span"):
        run(local.LocalModels(loader=load).provider(MODEL, PAYLOADS["spans"]))
    extractor.raw["extract_entities"] = {"entities": {"pet": []}}
    with pytest.raises(ValueError, match="invalid_provider_output"):
        run(local.LocalModels(loader=load).provider(MODEL, PAYLOADS["spans"]))


@pytest.mark.parametrize(
    ("model", "payload"),
    [
        ("fastino/unknown", PAYLOADS["spans"]),
        (MODEL, {"messages": []}),
        (MODEL, {**PAYLOADS["spans"], "primitive": {"kind": "classify", "config": {}}}),
    ],
    ids=["unknown-model", "no-primitive", "not-extract"],
)
def test_unservable_requests_are_adapter_unavailable(local, model, payload):
    load, _ = fake_loader(calls := [])
    with pytest.raises(ValueError, match="^primitive_adapter_unavailable$"):
        run(local.LocalModels(loader=load).provider(model, payload))
    assert calls == []  # refused before loading weights


def test_unknown_loader_is_adapter_unavailable(local):
    registry = {MODEL: {**local.LOCAL_MODELS[MODEL], "loader": "onnx"}}
    load, _ = fake_loader(calls := [])
    with pytest.raises(ValueError, match="^primitive_adapter_unavailable$"):
        run(local.LocalModels(registry, loader=load).provider(MODEL, PAYLOADS["spans"]))
    assert calls == []


@pytest.mark.parametrize("error", [ImportError("gliner2"), OSError("no weights"), RuntimeError()])
def test_load_failure_is_adapter_unavailable(local, error):
    def load(entry):
        raise error

    with pytest.raises(ValueError, match="^primitive_adapter_unavailable$"):
        run(local.LocalModels(loader=load).provider(MODEL, PAYLOADS["spans"]))


@pytest.mark.parametrize("code", ["primitive_adapter_unavailable", "model_identity_changed"])
def test_failed_load_backs_off_before_retrying(local, monkeypatch, code):
    now = [1000.0]
    monkeypatch.setattr(local.time, "monotonic", lambda: now[0])
    attempts = []
    ok, _ = fake_loader()

    def load(entry):
        attempts.append(entry)
        if len(attempts) == 1:
            raise ValueError(code) if code == "model_identity_changed" else OSError("offline")
        return ok(entry)

    models = local.LocalModels(loader=load)
    for _ in range(2):  # the second call refuses from memory, without a new load
        with pytest.raises(ValueError, match=f"^{code}$"):
            run(models.provider(MODEL, PAYLOADS["spans"]))
    assert len(attempts) == 1
    now[0] += local.RETRY_AFTER
    run(models.provider(MODEL, PAYLOADS["spans"]))
    assert len(attempts) == 2 and local._FAILED == {}


def _spans_payload(local, size, policy=None):
    """A spans payload whose hosted ``request``-policy input bound is ``size`` bytes."""
    canonical = sys.modules["seldon_runtime.contracts"].canonical
    payload = json.loads(json.dumps(PAYLOADS["spans"]))
    payload["primitive"]["source"] = ""
    if policy:
        payload["seldon_output_token_policy"] = policy

    def bound():
        return (
            len(canonical(payload["messages"]).encode())
            + 256 * len(payload["messages"])
            + len(canonical(payload["primitive"]).encode())
        )

    payload["primitive"]["source"] = "a" * (size - bound())
    assert bound() == size
    return payload


@pytest.mark.parametrize(
    ("policy", "size", "code"),
    [
        # request: bytes + 256/message + primitive bytes, plus 2048 default output <= 4096.
        (None, 2048, None),
        (None, 2049, "token_limit"),
        # bounded_8192 clamps output to what is left; refuses only when nothing is.
        ("bounded_8192", 4095, None),
        ("bounded_8192", 4096, "token_limit"),
        # v1 without a registered GLiNER counter refuses the byte overflow as unverified.
        ("context_measured_v1", 4096, "context_unverified"),
    ],
)
def test_context_fit_is_the_hosted_rule(local, policy, size, code):
    load, extractor = fake_loader(calls := [])
    extractor.raw["extract_entities"] = {"entities": {"person": [], "city": []}}
    payload = _spans_payload(local, size, policy)
    provider = local.LocalModels(loader=load).provider
    if code is None:
        run(provider(MODEL, payload))
        assert extractor.calls
        return
    with pytest.raises(ValueError, match=f"^{code}$"):
        run(provider(MODEL, payload))
    assert calls == [] and extractor.calls == []  # never handed to the model to truncate


@pytest.mark.parametrize("mode", ["spans", "records", "classification", "constrained"])
def test_v2_fit_counts_the_fastino_request_body(local, mode, monkeypatch):
    """context_measured_v2: o200k count of the exact Fastino body + 2048 output > 4096."""
    budget = sys.modules["seldon_runtime.output_budget"]
    gliner = sys.modules["seldon_runtime.gliner_normalize"]
    counted = []
    monkeypatch.setattr(budget, "assumed_token_count", lambda body: counted.append(body) or 2049)
    load, extractor = fake_loader()
    payload = {**PAYLOADS[mode], "seldon_output_token_policy": "context_measured_v2"}
    target = SimpleNamespace(upstream_model_name=MODEL)
    with pytest.raises(ValueError, match="^token_limit$"):
        run(local.LocalModels(loader=load).provider(MODEL, payload))
    assert counted == [gliner.request(target, payload)[1]]
    assert extractor.calls == []


def test_weights_load_once_across_concurrent_calls_and_instances(local):
    load, _ = fake_loader(calls := [], delay=0.05)

    async def many():
        first, second = local.LocalModels(loader=load), local.LocalModels(loader=load)
        await asyncio.gather(*(m.provider(MODEL, PAYLOADS["spans"]) for m in [first] * 3))
        await second.provider(MODEL, PAYLOADS["records"])

    run(many())
    assert calls == [local.LOCAL_MODELS[MODEL]]


def test_inference_runs_off_the_event_loop(local):
    load, extractor = fake_loader()
    loop_thread = []
    original = extractor.extract_entities

    def spy(*args, **kwargs):
        loop_thread.append(threading.current_thread() is threading.main_thread())
        return original(*args, **kwargs)

    extractor.extract_entities = spy
    run(local.LocalModels(loader=load).provider(MODEL, PAYLOADS["spans"]))
    assert loop_thread == [False]


@pytest.fixture
def fake_hub(local, monkeypatch, tmp_path):
    """Torch-free stand-ins for the default loader's lazy imports, over a real snapshot dir.

    ``seen["entry"]`` pins the files written under ``tmp_path`` by their sha256.
    """
    seen = {"snapshot": tmp_path / "snapshots" / REVISION}
    files = {}
    for name in local.LOCAL_MODELS[MODEL]["files"]:
        path = seen["snapshot"] / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        files[name] = hashlib.sha256(name.encode()).hexdigest()
    seen["entry"] = {**local.LOCAL_MODELS[MODEL], "files": files}

    def snapshot_download(repo, revision, allow_patterns):
        seen["download"] = (repo, revision, allow_patterns)
        return str(seen["snapshot"])

    class Model:
        def to(self, device, dtype):
            seen["to"] = (device, dtype)
            return self

    def from_pretrained(path):
        seen["path"] = path
        return Model()

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(float32="fp32"))
    monkeypatch.setitem(
        sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=snapshot_download)
    )
    monkeypatch.setitem(
        sys.modules,
        "gliner2",
        SimpleNamespace(AutoExtractor=SimpleNamespace(from_pretrained=from_pretrained)),
    )
    return seen


def test_default_loader_pins_revision_and_loads_fp32_cpu(local, fake_hub):
    local.load_gliner2(fake_hub["entry"])
    assert fake_hub["download"] == (MODEL, REVISION, ["*.json", "*.safetensors"])
    assert fake_hub["path"] == str(fake_hub["snapshot"])
    assert fake_hub["to"] == ("cpu", "fp32")


@pytest.mark.parametrize("name", ["model.safetensors", "encoder_config/config.json"])
def test_other_bytes_on_disk_are_refused(local, fake_hub, name):
    """A different revision, or edited files, under the pinned snapshot path."""
    (fake_hub["snapshot"] / name).write_bytes(b"another revision")
    registry = {MODEL: fake_hub["entry"]}
    with pytest.raises(ValueError, match="^model_identity_changed$"):
        run(local.LocalModels(registry).provider(MODEL, PAYLOADS["spans"]))
    assert "path" not in fake_hub  # never loaded


def test_missing_weight_file_is_adapter_unavailable(local, fake_hub):
    (fake_hub["snapshot"] / "model.safetensors").unlink()
    registry = {MODEL: fake_hub["entry"]}
    with pytest.raises(ValueError, match="^primitive_adapter_unavailable$"):
        run(local.LocalModels(registry).provider(MODEL, PAYLOADS["spans"]))
    assert "path" not in fake_hub


def test_missing_gliner2_is_adapter_unavailable(local, fake_hub, monkeypatch):
    monkeypatch.setitem(sys.modules, "gliner2", None)  # import gliner2 -> ImportError
    with pytest.raises(ValueError, match="^primitive_adapter_unavailable$"):
        run(local.LocalModels().provider(MODEL, PAYLOADS["spans"]))


def _serving_case(local):
    """A v3 plan: one GLiNER classification step, falling back to the original call."""
    contracts = importlib.import_module("seldon_runtime.contracts")
    spec = contracts.FunctionSpec.model_validate(
        {
            "calls": [
                {
                    "id": "classify",
                    "model": "openai/reference",
                    "messages": [{"role": "user", "content": "{{message}}"}],
                }
            ],
            "inputs": {"message": "I love this product, it works great!"},
            "bindings": [
                {
                    "call": "classify",
                    "message": 0,
                    "placeholder": "{{message}}",
                    "source": "input#/message",
                }
            ],
            "output_schema": SENTIMENT_SCHEMA,
        }
    )
    plan = contracts.Plan.model_validate(
        {
            "version": "function-plan-v3",
            "fallback": True,
            "steps": [
                {
                    "id": "a",
                    "op": "extract",
                    "model": MODEL,
                    "call": "classify",
                    "config": {"schema": SENTIMENT_SCHEMA},
                }
            ],
        }
    )
    return spec, plan


def _refusing(local, code):
    """``(registry, loader)`` under which the local provider raises ``code``."""
    load, extractor = fake_loader()
    registry = local.LOCAL_MODELS
    if code == "token_limit":
        registry = {MODEL: {**local.LOCAL_MODELS[MODEL], "context_length": 64}}
    elif code == "provider_unavailable":
        extractor.classify_text = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("oom"))
    elif code in {"primitive_adapter_unavailable", "model_identity_changed"}:

        def load(entry):
            raise ValueError(code) if code == "model_identity_changed" else OSError("offline")

    return registry, load


@pytest.mark.parametrize(
    "code",
    [
        "primitive_adapter_unavailable",
        "model_identity_changed",
        "token_limit",
        "provider_unavailable",
    ],
)
def test_serving_falls_back_on_every_local_refusal(local, code):
    """AC-L2.2: fallback eligibility is hosted serving's, provided the runner serves."""
    executor = importlib.import_module("seldon_runtime.executor")
    spec, plan = _serving_case(local)
    registry, load = _refusing(local, code)
    models = local.LocalModels(registry, loader=load)
    sent = []

    async def provider(model, payload):
        sent.append(model)
        if model in registry:
            return await models.provider(model, payload)
        return {"sentiment": "positive"}

    result = run(executor.execute(plan, spec, spec.inputs, provider, serving=True))
    assert result.fallback_used is True and result.fallback_reason == code
    assert result.output == {"sentiment": "positive"}
    assert sent == [MODEL, "openai/reference"]


def test_serving_case_is_served_locally_without_refusals(local):
    executor = importlib.import_module("seldon_runtime.executor")
    spec, plan = _serving_case(local)
    load, _ = fake_loader()
    provider = local.LocalModels(loader=load).provider
    result = run(executor.execute(plan, spec, spec.inputs, provider, serving=True))
    assert (result.fallback_used, result.output) == (False, {"sentiment": "positive"})


def test_evaluation_keeps_adapter_unavailable_typed(local):
    """Without serving=True the pre-dispatch refusal does not fall back (hosted rule)."""
    executor = importlib.import_module("seldon_runtime.executor")
    spec, plan = _serving_case(local)
    registry, load = _refusing(local, "primitive_adapter_unavailable")
    provider = local.LocalModels(registry, loader=load).provider
    with pytest.raises(ValueError, match="^primitive_adapter_unavailable$"):
        run(executor.execute(plan, spec, spec.inputs, provider))


@pytest.mark.local_model
@pytest.mark.filterwarnings("ignore:`torch.jit.script` is deprecated:FutureWarning")
def test_pinned_weights_serve_every_mode(local, monkeypatch):
    """Real model. ``SELDON_RECORD_LOCAL=1`` rewrites the recorded fixture."""
    raw = {}

    def recording_loader(entry):
        extractor = local.load_gliner2(entry)
        for name in ("extract_entities", "extract_json", "classify_text"):

            def wrap(fn=getattr(extractor, name), name=name):
                def call(*args, **kwargs):
                    raw[name] = fn(*args, **kwargs)
                    return raw[name]

                return call

            setattr(extractor, name, wrap())
        return extractor

    def classify_constrained(*args, **kwargs):
        raw["classify_constrained"] = constrained(*args, **kwargs)
        return raw["classify_constrained"]

    constrained = local._classify_constrained
    monkeypatch.setattr(local, "_classify_constrained", classify_constrained)
    models = local.LocalModels(loader=recording_loader)
    for mode, payload in PAYLOADS.items():
        output = run(models.provider(MODEL, payload))
        assert comparable(mode, output) == EXPECTED[mode]
    fixture = {mode: {"call": call, "raw": raw[call]} for mode, call in CALLS.items()}
    if os.environ.get("SELDON_RECORD_LOCAL") == "1":
        FIXTURE.write_text(json.dumps(fixture, indent=2) + "\n")
