"""GLiNER wire shape: the Fastino request body and output normalization (grounded
spans, null policy, label and schema checks).

Pure leaf module (stdlib + jsonschema + allowlisted siblings) so the staged
``seldon_runtime`` wheel builds the same request and runs the same ``normalize`` as
the hosted router.
"""

from __future__ import annotations

import json

from jsonschema import Draft202012Validator

from .native_schema import (
    INVALID_GROUNDED_SPAN,
    SCHEMA_MISMATCH,
    UNGROUNDED_QUOTATION,
    structured_schema,
)
from .semantic import normalize_spans

CONSTRAINED_SCHEMA_VERSION = 3
# Fastino constrained classification (docs.fastino.ai/inference/chat-completions).
# Models whose ``GET /v1/base-models`` ``encoder_features`` include
# ``constrained_classification`` (recorded 2026-11-02). Others get no descriptions.
CONSTRAINED_CLASSIFICATION_MODELS = frozenset({"fastino/gliner2.5-multi-v1"})


def request(target, payload, profile=None):
    """``(route, body)`` for a GLiNER extraction step."""
    primitive = payload["primitive"]
    config = primitive["config"]
    if primitive["kind"] == "extract" and config.get("mode") in {"spans", "records"}:
        source = primitive.get("source")
        if not isinstance(source, str) or len(source.encode()) > 262144:
            raise ValueError("invalid_primitive_source")
        schema = (
            {"entities": [{"name": k, "description": v} for k, v in config["labels"].items()]}
            if config["mode"] == "spans"
            else structured_schema(config["schema"])
        )
        return "chat/completions", {
            "model": target.upstream_model_name,
            "messages": [{"role": "user", "content": source}],
            "schema": schema,
            "threshold": 0.5,
            "include_spans": True,
            "include_confidence": True,
            "store": False,
        }
    descriptions = config.get("label_descriptions")
    if descriptions:
        if target.upstream_model_name not in CONSTRAINED_CLASSIFICATION_MODELS:
            raise ValueError("unsupported_native_schema")
        # Same keys as the recorded constrained request (no span options).
        return "chat/completions", {
            "model": target.upstream_model_name,
            "messages": payload["messages"],
            "schema": classification_schema(config["schema"], descriptions),
            "include_confidence": True,
            "store": False,
        }
    return "chat/completions", {
        "model": target.upstream_model_name,
        "messages": payload["messages"],
        "schema": classification_schema(config["schema"]),
        "include_spans": True,
        "include_confidence": True,
        "store": False,
    }


def classification_schema(schema, descriptions=None):
    """Translate a closed string-enum object into Fastino classification tasks."""
    if (
        schema.get("type") != "object"
        or schema.get("additionalProperties") is not False
        or not isinstance(schema.get("properties"), dict)
    ):
        raise ValueError("unsupported_native_schema")
    properties = schema["properties"]
    required = schema.get("required")
    if not properties or set(required or ()) != set(properties):
        raise ValueError("unsupported_native_schema")
    tasks = []
    for field, value in properties.items():
        labels = value.get("enum") if isinstance(value, dict) else None
        if (
            not isinstance(value, dict)
            or value.get("type") != "string"
            or not isinstance(labels, list)
            or not 2 <= len(labels) <= 128
            or any(not isinstance(label, str) or not label for label in labels)
            or len(set(labels)) != len(labels)
        ):
            raise ValueError("unsupported_native_schema")
        tasks.append(
            {
                "task": field,
                "labels": labels,
                "multi_label": False,
                "top_k": 1,
            }
        )
    if not descriptions:
        return {"classifications": tasks}
    # Classic entries accept only task/labels/multi_label/cls_threshold/top_k; a
    # ``label_descriptions`` key is a 400 (recorded 2026-11-02). Descriptions go
    # through constrained classification, where labels map to their descriptions.
    if set(descriptions) != set(properties) or any(
        set(descriptions[t["task"]]) != set(t["labels"]) for t in tasks
    ):
        raise ValueError("unsupported_native_schema")
    return {
        "version": CONSTRAINED_SCHEMA_VERSION,
        "tasks": {
            t["task"]: {
                "labels": {label: descriptions[t["task"]][label] for label in t["labels"]},
                "min_labels": 1,
                "max_labels": 1,
            }
            for t in tasks
        },
        "constraints": [],
    }


def grounded_spans(source, spans):
    try:
        return normalize_spans(source, spans)
    except ValueError as exc:
        if str(exc) in {"invalid_span", "conflicting_span"}:
            raise ValueError(INVALID_GROUNDED_SPAN) from None
        raise


def apply_null_policy(value, schema, policy):
    """Declared ``empty_string:v1`` maps null to "" only at string-typed fields."""
    if policy is None:
        return value
    if isinstance(value, dict) and schema.get("type") == "object":
        properties = schema.get("properties", {})
        return {k: apply_null_policy(v, properties.get(k, {}), policy) for k, v in value.items()}
    if isinstance(value, list) and schema.get("type") == "array":
        return [apply_null_policy(v, schema.get("items", {}), policy) for v in value]
    if value is None and schema.get("type") == "string":
        return ""
    return value


def normalize(payload, raw):
    primitive = payload["primitive"]
    config = primitive["config"]
    mode = config.get("mode")
    if primitive["kind"] == "extract" and mode in {"spans", "records"}:
        content = raw.get("result", raw)
        if "choices" in raw:
            content = json.loads(raw["choices"][0]["message"]["content"])
        if not isinstance(content, dict):
            raise ValueError("invalid_provider_output")
        if mode == "spans":
            entities = content.get("entities")
            if isinstance(entities, list) and len(entities) == 1 and isinstance(entities[0], dict):
                entities = entities[0]
            if not isinstance(entities, dict) or set(entities) - set(config["labels"]):
                raise ValueError("invalid_provider_output")
            spans = []
            for label, values in entities.items():
                if not isinstance(values, list):
                    raise ValueError("invalid_provider_output")
                for span in values:
                    if not isinstance(span, dict):
                        raise ValueError("invalid_provider_output")
                    spans.append({**span, "label": label})
            return grounded_spans(primitive["source"], spans)
        schema = config["schema"]
        if set(content) != set(schema.get("properties", {})):
            raise ValueError("invalid_provider_output")

        def unpack(value):
            if isinstance(value, dict):
                if "text" in value and set(value) <= {"text", "start", "end", "confidence"}:
                    if "start" in value or "end" in value:
                        grounded_spans(primitive["source"], [value])
                    return value["text"]
                return {k: unpack(v) for k, v in value.items()}
            if isinstance(value, list):
                if len(value) > 32:
                    raise ValueError("record_limit")
                return [unpack(v) for v in value]
            return value

        output = apply_null_policy(unpack(content), schema, config.get("null_policy"))
        if not Draft202012Validator(schema).is_valid(output):
            # e.g. null under an all-string record contract; never weaken the schema.
            raise ValueError(SCHEMA_MISMATCH)
        return output
    content = json.loads(raw["choices"][0]["message"]["content"])
    output_schema = primitive["config"]["schema"]
    fields = output_schema.get("properties", {})
    if config.get("label_descriptions"):
        return _constrained(content, output_schema)
    try:
        classification_schema(output_schema)
    except ValueError:
        # Read-only compatibility for already stored legacy evidence. New
        # requests reject this shape in request_payload before dispatch.
        if (
            isinstance(content, dict)
            and set(content) == set(fields)
            and all(isinstance(v, str) for v in content.values())
        ):
            return content
        entities = content.get("entities", {}) if isinstance(content, dict) else {}
        legacy = {}
        for field in fields:
            spans = entities.get(field, [])
            if (
                len(spans) != 1
                or not isinstance(spans[0], dict)
                or not isinstance(spans[0].get("text"), str)
            ):
                raise ValueError("invalid_provider_output")
            text = spans[0]["text"]
            if not any(text in message["content"] for message in payload["messages"]):
                raise ValueError(UNGROUNDED_QUOTATION)
            legacy[field] = text
        return legacy
    result = {}
    if not isinstance(content, dict) or set(content) != set(fields):
        raise ValueError("invalid_provider_output")
    for field, schema in fields.items():
        answer = content.get(field)
        if not isinstance(answer, dict) or set(answer) - {"label", "confidence"}:
            raise ValueError("invalid_provider_output")
        label = answer.get("label")
        if label not in schema.get("enum", ()):
            raise ValueError("invalid_provider_output")
        confidence = answer.get("confidence")
        if confidence is not None and (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not 0 <= confidence <= 1
        ):
            raise ValueError("invalid_provider_output")
        result[field] = label
    if not Draft202012Validator(output_schema).is_valid(result):
        raise ValueError(SCHEMA_MISMATCH)
    return result


def _constrained(content, output_schema):
    """One feasible single-label answer per field from a constrained result.

    Shape: ``{field: {"value", "confidence", "probabilities"}, "_meta": {...}}``.
    A best-effort (``feasible`` false) result is not an answer.
    """
    fields = output_schema.get("properties", {})
    if not isinstance(content, dict) or set(content) != set(fields) | {"_meta"}:
        raise ValueError("invalid_provider_output")
    meta = content["_meta"]
    if not isinstance(meta, dict) or meta.get("feasible") is not True:
        raise ValueError("invalid_provider_output")
    result = {}
    for field, schema in fields.items():
        answer = content[field]
        value = answer.get("value") if isinstance(answer, dict) else None
        if not isinstance(value, str) or value not in schema.get("enum", ()):
            raise ValueError("invalid_provider_output")
        result[field] = value
    if not Draft202012Validator(output_schema).is_valid(result):
        raise ValueError(SCHEMA_MISMATCH)
    return result
