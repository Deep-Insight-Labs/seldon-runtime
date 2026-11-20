"""Execution schema and native-step validation; planner registries stay server-side."""

from .semantic import OPERATORS


def reject_incompatible_schema(actual, required):
    """Reject provable mismatches in the supported structural schema fragment.

    Unknown/conditional schemas remain guarded by runtime validation. This is
    not a claim of arbitrary JSON Schema subsumption or semantic equivalence.
    """
    if actual is False:
        raise ValueError("incompatible_component_inputs")
    if not isinstance(actual, dict) or not isinstance(required, dict):
        return

    def types(schema):
        value = schema.get("type")
        result = set(value if isinstance(value, list) else [value]) if value else set()
        if "integer" in result:
            result.add("number")
        return result

    if types(actual) and types(required) and not types(actual) & types(required):
        raise ValueError("incompatible_component_inputs")
    if "enum" in actual and "enum" in required:
        if not any(v in required["enum"] for v in actual["enum"]):
            raise ValueError("incompatible_component_inputs")
    if required.get("type") == "array":
        reject_incompatible_schema(actual.get("items"), required.get("items"))
    if required.get("type") == "object":
        properties = actual.get("properties", {})
        for key in required.get("required", []):
            if key not in properties and actual.get("additionalProperties") is False:
                raise ValueError("incompatible_component_inputs")
        for key, target in required.get("properties", {}).items():
            if key in properties:
                reject_incompatible_schema(properties[key], target)


def collection(properties, maximum, required=None):
    return {
        "type": "array",
        "maxItems": maximum,
        "items": {
            "type": "object",
            "properties": properties,
            "required": required or list(properties),
        },
    }


STRING = {"type": "string"}

MENTIONS = collection({"id": STRING, "text": STRING}, 32)

CATALOG = collection(
    {"id": STRING, "name": STRING, "aliases": {"type": "array", "maxItems": 32, "items": STRING}},
    500,
    ["id", "name"],
)

PAIRS = collection({"id": STRING, "mention_id": STRING, "candidate_id": STRING}, 160)

DECISIONS = collection({"id": STRING, "match": {"type": "boolean"}}, 160)

RETRIEVAL = {
    "type": "object",
    "properties": {
        "pairs": PAIRS,
        "catalog_id": STRING,
        "omitted_pairs": {"type": "integer", "minimum": 0},
        "mention_ids": {"type": "array", "maxItems": 32, "items": STRING},
    },
    "required": ["pairs", "mention_ids"],
}

LINKS = collection(
    {
        "mention_id": STRING,
        "candidate_id": {"type": ["string", "null"]},
        "status": {"enum": ["matched", "no_match", "ambiguous"]},
    },
    32,
)

SEMANTIC_IO = {
    "align_spans": (
        [STRING, collection({"text": STRING, "label": STRING}, 32)],
        collection(
            {
                "id": STRING,
                "text": STRING,
                "label": STRING,
                "start": {"type": "integer"},
                "end": {"type": "integer"},
            },
            32,
        ),
    ),
    "project_records": (
        [{"type": "array", "maxItems": 32, "items": {"type": "object"}}],
        {"type": "array", "maxItems": 32, "items": {"type": "object"}},
    ),
    "merge_spans": (
        [STRING, {"type": "array", "maxItems": 32}],
        {"type": "array", "maxItems": 32},
    ),
    "merge_records": ([{"type": "array", "maxItems": 32}], {"type": "object"}),
    "merge_labels": ([{"type": "array", "maxItems": 32}], {"type": "object"}),
    "normalize_text": (
        [STRING],
        {"type": "object", "required": ["text", "source_map", "document_id"]},
    ),
    "chunk_text": (
        [STRING],
        collection(
            {
                "id": STRING,
                "start": {"type": "integer"},
                "end": {"type": "integer"},
                "text": STRING,
            },
            32,
        ),
    ),
    "retrieve_candidates": ([MENTIONS, CATALOG], RETRIEVAL),
    "exact_links": ([MENTIONS, CATALOG], LINKS),
    "resolve_links": ([RETRIEVAL, DECISIONS], LINKS),
    "assemble_records": (
        [MENTIONS, collection({"id": STRING}, 32)],
        {"type": "array", "maxItems": 32},
    ),
}

CONFIG_FIELDS = {
    "casefold": {"type": "boolean"},
    "unique": {"type": "boolean"},
    "size": {"type": "integer", "minimum": 1, "maximum": 16384},
    "overlap": {"type": "integer", "minimum": 0, "maximum": 16383},
    "top_k": {"enum": [1, 3, 5]},
    "unit": {"enum": ["chars", "tokens"]},
    "rule": {"enum": ["all_agree", "any_true", "max_ordinal", "union"]},
    "order": {"type": "array", "minItems": 2, "maxItems": 128, "items": {"type": "string"}},
    "sums": {"type": "object", "maxProperties": 32},
    "fields": {
        "type": "array",
        "minItems": 1,
        "maxItems": 32,
        "uniqueItems": True,
        "items": {"type": "string", "minLength": 1},
    },
}


def semantic_contract(op):
    arity, keys = OPERATORS[op]
    inputs, output = SEMANTIC_IO[op]
    return {
        "input_schema": {
            "type": "array",
            "prefixItems": inputs,
            "minItems": arity,
            "maxItems": arity,
        },
        "output_schema": output,
        "config_schema": {
            "type": "object",
            "properties": {
                k: (
                    {
                        "type": "object",
                        "minProperties": 1,
                        "maxProperties": 32,
                        "propertyNames": {"minLength": 1},
                        "additionalProperties": {"enum": ["array", "scalar", "ignore"]},
                    }
                    if op == "merge_records" and k == "fields"
                    else {
                        "type": "object",
                        "minProperties": 1,
                        "maxProperties": 32,
                        "propertyNames": {"minLength": 1},
                        "additionalProperties": {"type": "string", "pattern": "^/"},
                    }
                    if op == "project_records" and k == "fields"
                    else CONFIG_FIELDS[k]
                )
                for k in keys
            },
            "additionalProperties": False,
            "required": ["fields"]
            if op in {"assemble_records", "project_records", "merge_records"}
            else [],
        },
        "source_identity": "original_codepoint_offsets_and_explicit_ids",
        "failure_classes": [
            "invalid_kernel_configuration",
            "invalid_identity",
            "input_limit",
            "invalid_input_shape",
        ],
        "max_input_bytes": 262144,
        "max_output_bytes": 262144,
    }


NULL_POLICIES = frozenset({"empty_string:v1"})


def validate_label_descriptions(descriptions, schema):
    """One bounded description for exactly each enum label of each field."""
    properties = schema.get("properties") or {}
    if not isinstance(descriptions, dict) or set(descriptions) - set(properties):
        raise ValueError("invalid_primitive_configuration")
    for field, labels in descriptions.items():
        if (
            not isinstance(labels, dict)
            or set(labels) != set(properties[field].get("enum") or [])
            or any(not isinstance(v, str) or not v or len(v) > 1024 for v in labels.values())
        ):
            raise ValueError("invalid_primitive_configuration")


def validate_native(step, capability):
    if step.op != capability["kind"]:
        raise ValueError("unsupported_component")
    if step.op == "extract":
        mode = step.config.get("mode", "legacy")
        if mode != "legacy" and mode not in capability.get("modes", []):
            raise ValueError("primitive_mode_unqualified")
        if mode not in {"legacy", "spans", "records"}:
            raise ValueError("unsupported_component")
        if mode == "spans":
            labels = step.config.get("labels")
            if (
                not isinstance(labels, dict)
                or not 1 <= len(labels) <= 32
                or any(
                    not isinstance(k, str) or not k or not isinstance(v, str) or len(v) > 1024
                    for k, v in labels.items()
                )
                or set(step.config) - {"mode", "labels"}
            ):
                raise ValueError("invalid_primitive_configuration")
        else:
            from .contracts import bounded_schema

            bounded_schema(step.config.get("schema", {}))
            allowed = {"mode", "schema", "null_policy"}
            if mode == "legacy":
                allowed |= {"view", "label_descriptions"}
            if set(step.config) - allowed:
                raise ValueError("invalid_primitive_configuration")
            if "label_descriptions" in step.config:
                validate_label_descriptions(
                    step.config["label_descriptions"], step.config.get("schema", {})
                )
            # A declared, typed normalization; only where the task permits it (W6).
            if "null_policy" in step.config and (
                mode != "records" or step.config["null_policy"] not in NULL_POLICIES
            ):
                raise ValueError("invalid_primitive_configuration")
        if mode in {"spans", "records"} and (len(step.inputs) != 1 or step.prompt or step.messages):
            raise ValueError("unsupported_native_prompt")
        if mode == "records":
            from .native_schema import structured_schema

            structured_schema(step.config["schema"])
        if "view" in step.config and (
            step.config["view"] not in {"full", "bindings"} or mode in {"spans", "records"}
        ):
            raise ValueError("invalid_primitive_configuration")
    elif step.op == "choice":
        if step.config.get("mode") == "pairs":
            if "pairs" not in capability.get("modes", []):
                raise ValueError("primitive_mode_unqualified")
            if set(step.config) != {"mode"} or len(step.inputs) < 1:
                raise ValueError("invalid_primitive_configuration")
            return
        questions = step.config.get("questions")
        if (
            not isinstance(questions, dict)
            or not 1 <= len(questions) <= 128
            or set(step.config) - {"questions", "view"}
            or step.config.get("view", "full") not in {"full", "bindings"}
        ):
            raise ValueError("invalid_primitive_configuration")
        for field, question in questions.items():
            if not isinstance(field, str) or not isinstance(question, dict):
                raise ValueError("invalid_primitive_configuration")
            options = question.get("options", [])
            if question.get("type") in {"noul", "score"}:
                # Only capability runs pass card ``answers``; legacy primitives
                # (and Jev without a card) never accept these answer types.
                from .contracts import check_question

                if question["type"] not in capability.get("answers", ()):
                    raise ValueError("invalid_primitive_configuration")
                check_question(question)
                if question["type"] == "score" and len(question["levels"]) > capability.get(
                    "max_levels", 10
                ):
                    raise ValueError("invalid_primitive_configuration")
            elif question.get("type") == "boolean":
                if set(question) != {"type"}:
                    raise ValueError("invalid_primitive_configuration")
            elif (
                question.get("type") != "choice"
                or not isinstance(options, list)
                or not 1 <= len(options) <= 128
                or any(not isinstance(v, str) for v in options)
                or len(set(options)) != len(options)
                or set(question) != {"type", "options"}
            ):
                raise ValueError("invalid_primitive_configuration")
