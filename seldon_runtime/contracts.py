"""Data-only contracts: imported requests, bounded programs and frozen SLOs."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any, ClassVar, Literal
from uuid import UUID

from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from .conditional import Predicate


def canonical(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def bounded_schema(schema):
    """No remote resolution, recursive schemas or unbounded regex evaluation."""
    count = 0

    def visit(value, depth=0, refs=()):
        nonlocal count
        count += 1
        if depth > 24 or count > 2048:
            raise ValueError("schema_complexity_limit")
        if isinstance(value, dict):
            if any(
                k in value
                for k in ("pattern", "patternProperties", "$dynamicRef", "$recursiveRef", "$id")
            ):
                raise ValueError("unsupported_schema_keyword")
            ref = value.get("$ref")
            if ref:
                if not isinstance(ref, str) or not ref.startswith("#/") or ref in refs:
                    raise ValueError("unsupported_schema_reference")
                target = schema
                for key in ref[2:].split("/"):
                    target = target[key.replace("~1", "/").replace("~0", "~")]
                visit(target, depth + 1, (*refs, ref))
            for key, item in value.items():
                if key != "$ref":
                    visit(item, depth + 1, refs)
        elif isinstance(value, list):
            for item in value:
                visit(item, depth + 1, refs)

    from jsonschema.exceptions import SchemaError

    try:
        visit(schema)
        Draft202012Validator.check_schema(schema)
    except (KeyError, TypeError) as exc:
        raise ValueError("invalid_schema_reference") from exc
    except SchemaError as exc:
        # Not a ValueError: left unconverted it escaped Plan validation and aborted
        # proposal construction instead of rejecting one proposal.
        raise ValueError("invalid_schema") from exc


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    @property
    def digest(self) -> str:
        return digest(self.model_dump(mode="json"))


class Message(Contract):
    role: Literal["system", "developer", "user", "assistant"]
    content: str = Field(max_length=262144)


class RequestContract(Contract):
    id: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_]{0,63}$")
    model: str = Field(min_length=1, max_length=256)
    messages: list[Message] = Field(min_length=1, max_length=32)
    temperature: float | None = Field(default=0, ge=0, le=2)
    # The selected offering validates the upper bound; imports may declare its full limit.
    max_tokens: int | None = Field(default=2048, ge=1)
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = Field(default=None, strict=True)

    @model_serializer(mode="wrap")
    def compatible_dump(self, handler):
        value = handler(self)
        for key in ("tools", "tool_choice", "parallel_tool_calls"):
            if getattr(self, key) is None:
                value.pop(key, None)
        return value

    @model_validator(mode="after")
    def check_output_tool(self):
        from .tool_wire import output_tool

        tool = output_tool(self.model_dump(exclude_none=True))
        if tool:
            bounded_schema(tool["parameters"])
        return self


def reference(value: str) -> tuple[str, str]:
    root, sep, path = value.partition("#")
    if not sep or not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]{0,63}", root):
        raise ValueError("invalid_reference")
    if path and not path.startswith("/"):
        raise ValueError("invalid_pointer")
    if re.search(r"~(?![01])", path):
        raise ValueError("invalid_pointer")
    return root, path


class Binding(Contract):
    call: str
    message: int = Field(ge=0, le=31)
    placeholder: str = Field(min_length=1, max_length=256)
    source: str


class ObjectiveCheck(Contract):
    kind: Literal["collection", "quotes"]
    path: str = Field(pattern=r"^(?:/.*)?$", max_length=256)
    min_items: int = Field(default=0, ge=0, le=1000)
    max_items: int = Field(default=1000, ge=0, le=1000)
    source_field: str = Field(default="memo", min_length=1, max_length=64)
    quote_field: str = Field(default="quote", min_length=1, max_length=64)
    sources: dict[str, str] = Field(default_factory=dict, max_length=16)

    @model_validator(mode="after")
    def check(self):
        reference("output#" + self.path)
        if self.min_items > self.max_items or (self.kind == "quotes" and not self.sources):
            raise ValueError("invalid_objective_check")
        for path in self.sources.values():
            reference("input#" + path)
        return self


class Rubric(Contract):
    # rubric-v2 replaces the memo-demo ``source_field`` default with the neutral
    # "source" used by EvaluationCheck. Saved rubric-v1 contracts keep "memo".
    version: Literal["rubric-v1", "rubric-v2"] = "rubric-v1"
    judge_model: str = Field(min_length=1, max_length=256)
    criteria: list[str] = Field(min_length=1, max_length=6)
    checks: list[ObjectiveCheck] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def check(self):
        if any(not c.strip() or len(c) > 1024 for c in self.criteria):
            raise ValueError("invalid_rubric")
        if self.version == "rubric-v2" and any(
            "source_field" not in check.model_fields_set for check in self.checks
        ):
            checks = [
                check
                if "source_field" in check.model_fields_set
                else ObjectiveCheck.model_validate({**check.model_dump(), "source_field": "source"})
                for check in self.checks
            ]
            object.__setattr__(self, "checks", checks)
        return self


class EvaluationCheck(Contract):
    """Configuration of bounded built-in checks, never generated executable code."""

    kind: Literal["schema", "collection", "spans", "quotes", "enum", "exact", "records", "decimal"]
    path: str = Field(default="", max_length=256)
    source_path: str | None = Field(default=None, max_length=256)
    min_items: int = Field(default=0, ge=0, le=1000)
    max_items: int = Field(default=1000, ge=0, le=1000)
    text_field: str = Field(default="text", min_length=1, max_length=64)
    start_field: str = Field(default="start", min_length=1, max_length=64)
    end_field: str = Field(default="end", min_length=1, max_length=64)
    source_field: str = Field(default="source", min_length=1, max_length=64)
    quote_field: str = Field(default="quote", min_length=1, max_length=64)
    sources: dict[str, str] = Field(default_factory=dict, max_length=16)
    allowed: list[str | int | float | bool | None] = Field(default_factory=list, max_length=128)
    # evaluation-semantics-v2 only; omitted from serialization unless set.
    min_chars: int = Field(default=1, ge=1, le=10000)
    normalization: Literal["none", "typographic_whitespace"] = "none"
    offset_unit: Literal["codepoint", "utf16"] = "codepoint"

    V2_FIELDS: ClassVar[frozenset[str]] = frozenset({"min_chars", "normalization", "offset_unit"})

    @model_serializer(mode="wrap")
    def compatible_dump(self, handler):
        value = handler(self)
        for key in self.V2_FIELDS:
            if key not in self.model_fields_set:
                value.pop(key, None)
        return value

    @model_validator(mode="after")
    def check(self):
        reference("output#" + self.path)
        if self.source_path is not None:
            reference("input#" + self.source_path)
        for path in self.sources.values():
            reference("input#" + path)
        if self.min_items > self.max_items:
            raise ValueError("invalid_evaluation_check")
        if self.kind == "spans" and self.source_path is None:
            raise ValueError("invalid_evaluation_check")
        if self.kind == "quotes" and not self.sources and self.source_path is None:
            raise ValueError("invalid_evaluation_check")
        if self.kind == "enum" and not self.allowed:
            raise ValueError("invalid_evaluation_check")
        return self


class EvaluationExample(Contract):
    input_origin: Literal["synthetic", "uploaded", "recorded"] = Field(
        default="uploaded", json_schema_extra=lambda schema: schema.pop("default", None)
    )
    label_origin: Literal["human", "reference", "judge", "unknown"] = Field(
        default="unknown", json_schema_extra=lambda schema: schema.pop("default", None)
    )
    input: dict[str, Any]
    output: Any = None
    expected: Any = None
    assertions: list[str] = Field(default_factory=list, max_length=6)
    group: str | None = Field(default=None, min_length=1, max_length=128)
    weight: float | None = Field(default=None, gt=0)
    role: Literal["training", "development", "confirmation"] | None = None
    value: float | None = Field(default=None, ge=0, allow_inf_nan=False)

    @model_serializer(mode="wrap")
    def compatible_dump(self, handler):
        value = handler(self)
        if self.input_origin == "uploaded":
            value.pop("input_origin", None)
        if self.label_origin == "unknown":
            value.pop("label_origin", None)
        for key in ("output", "expected", "group", "role", "value"):
            if key not in self.model_fields_set:
                value.pop(key, None)
        if "weight" not in self.model_fields_set:
            value.pop("weight", None)
        return value

    @model_validator(mode="after")
    def check(self):
        if self.weight is not None and (not math.isfinite(self.weight) or self.weight <= 0):
            raise ValueError("invalid_example_weight")
        if any(not a.strip() or len(a) > 2048 for a in self.assertions):
            raise ValueError("invalid_evaluation_assertion")
        if len(canonical(self.model_dump()).encode()) > 262144:
            raise ValueError("evaluation_example_too_large")
        return self


class PredictionMapping(Contract):
    source: Any
    target: Any


class PredictionThreshold(Contract):
    """Maps a numeric score to the positive (true) or negative (false) class."""

    operator: Literal["ge", "gt", "le", "lt"]
    value: float = Field(allow_inf_nan=False)


TASK_METRICS = (
    "exact_match",
    "binary_f1",
    "value_weighted_recall",
    "within_tolerance",
    "mean_absolute_error",
    "macro_f1",
    "set_f1",
    "field_accuracy",
    "record_f1",
    "span_f1",
)


def _scalar(value):
    return type(value) in {str, int, float, bool} or value is None


def _label_type_ok(metric, label):
    """Type gate for evaluation-semantics-v2 labels; v1 metrics are unchanged."""
    if metric in {"within_tolerance", "mean_absolute_error"}:
        return type(label) in {int, float} and math.isfinite(label)
    if metric == "macro_f1":
        return type(label) in {str, int, bool}
    if metric == "set_f1":
        return isinstance(label, list) and all(_scalar(item) for item in label)
    if metric == "field_accuracy":
        return isinstance(label, dict)
    if metric in {"record_f1", "span_f1"}:
        return isinstance(label, list) and all(isinstance(item, dict) for item in label)
    return True


def _pointer_list(values, *, limit):
    if len(values) > limit or len(set(values)) != len(values):
        raise ValueError("invalid_metric_fields")
    for value in values:
        if not value.startswith("/") or re.search(r"~(?![01])", value):
            raise ValueError("invalid_metric_fields")


MAX_EVALUATION_EXAMPLES = 1000


class EvaluationFunction(Contract):
    recording_examples_ref: str | None = Field(default=None, max_length=1024)
    version: Literal["evaluation-function-v1"] = "evaluation-function-v1"
    instructions: str = Field(min_length=1, max_length=8192)
    criteria: list[str] = Field(default_factory=list, max_length=6)
    checks: list[EvaluationCheck] = Field(default_factory=list, max_length=16)
    examples: list[EvaluationExample] = Field(
        default_factory=list, max_length=MAX_EVALUATION_EXAMPLES
    )
    metric: (
        Literal[
            "exact_match",
            "binary_f1",
            "value_weighted_recall",
            "within_tolerance",
            "mean_absolute_error",
            "macro_f1",
            "set_f1",
            "field_accuracy",
            "record_f1",
            "span_f1",
        ]
        | None
    ) = None
    positive_label: Any = None
    prediction_path: str | None = Field(default=None, max_length=1024)
    prediction_map: list[PredictionMapping] = Field(default_factory=list, max_length=128)
    # evaluation-semantics-v2 parameters. Each is omitted from serialization unless
    # set, so v1 definitions, specs and digests stay byte-identical.
    negative_label: Any = None
    prediction_threshold: PredictionThreshold | None = None
    absolute_tolerance: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    relative_tolerance: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    fields: list[str] = Field(default_factory=list, max_length=32)
    record_key: list[str] = Field(default_factory=list, max_length=8)
    empty_values: Literal["distinct", "equivalent"] = "distinct"
    span_match: Literal["exact", "overlap"] = "exact"
    span_start_field: str = Field(default="start", min_length=1, max_length=64)
    span_end_field: str = Field(default="end", min_length=1, max_length=64)
    span_label_field: str = Field(default="label", min_length=1, max_length=64)
    span_source_path: str | None = Field(default=None, max_length=256)
    # Explicit authority for a generated span recipe to treat context as a hint.
    span_context_policy: Literal["strict", "hint"] = "strict"
    offset_unit: Literal["codepoint", "utf16"] = "codepoint"
    expected_role: Literal["authoritative", "reference_answer"] = "authoritative"
    judge_profile: Literal["standard", "verify_derived"] = "standard"

    V1_OPTIONAL: ClassVar[tuple[str, ...]] = (
        "metric",
        "positive_label",
        "prediction_path",
        "prediction_map",
    )
    V2_FIELDS: ClassVar[frozenset[str]] = frozenset(
        {
            "negative_label",
            "prediction_threshold",
            "absolute_tolerance",
            "relative_tolerance",
            "fields",
            "record_key",
            "empty_values",
            "span_match",
            "span_start_field",
            "span_end_field",
            "span_label_field",
            "span_source_path",
            "span_context_policy",
            "offset_unit",
            "expected_role",
            "judge_profile",
        }
    )

    @model_serializer(mode="wrap")
    def compatible_dump(self, handler):
        value = handler(self)
        if self.recording_examples_ref is None:
            value.pop("recording_examples_ref", None)
        else:
            # Recording payload lives in its expiring run manifest, never retained specs.
            value["examples"] = [
                e for e in value.get("examples", []) if e.get("input_origin") != "recorded"
            ]
        for key in (*self.V1_OPTIONAL, *self.V2_FIELDS):
            if key not in self.model_fields_set:
                value.pop(key, None)
        return value

    def metric_parameters(self) -> dict:
        """Every explicitly set metric parameter; part of the v2 definition digest."""
        return {
            key: value
            for key, value in self.model_dump(mode="json", exclude={"examples"}).items()
            if key in self.V2_FIELDS or key in self.V1_OPTIONAL
        }

    @model_validator(mode="after")
    def check(self):
        if not self.instructions.strip() or any(
            not c.strip() or len(c) > 1024 for c in self.criteria
        ):
            raise ValueError("invalid_evaluation")
        if (
            self.metric in {"binary_f1", "value_weighted_recall"}
            and "positive_label" not in self.model_fields_set
        ):
            raise ValueError("positive_label_required")
        if len(canonical(self.model_dump()).encode()) > 5 * 1024 * 1024:
            raise ValueError("evaluation_examples_too_large")
        if self.prediction_path is not None:
            if self.prediction_path and not self.prediction_path.startswith("/"):
                raise ValueError("invalid_prediction_path")
            if re.search(r"~(?![01])", self.prediction_path):
                raise ValueError("invalid_prediction_path")
        mapping_sources = [canonical(entry.source) for entry in self.prediction_map]
        if len(mapping_sources) != len(set(mapping_sources)):
            raise ValueError("duplicate_prediction_mapping")
        if self.prediction_threshold is not None:
            if self.prediction_map:
                raise ValueError("prediction_threshold_with_map")
            if self.metric not in {"binary_f1", "value_weighted_recall"}:
                raise ValueError("prediction_threshold_requires_binary_metric")
            if "negative_label" not in self.model_fields_set:
                raise ValueError("negative_label_required")
            if canonical(self.negative_label) == canonical(self.positive_label):
                raise ValueError("invalid_threshold_classes")
        _pointer_list(self.fields, limit=32)
        _pointer_list(self.record_key, limit=8)
        if self.span_source_path is not None:
            reference("input#" + self.span_source_path)
        seen = set()
        for example in self.examples:
            if self.metric and "expected" not in example.model_fields_set:
                raise ValueError("example_label_required")
            if self.metric == "value_weighted_recall" and example.value is None:
                raise ValueError("example_value_required")
            if self.prediction_threshold is not None and canonical(example.expected) not in {
                canonical(self.positive_label),
                canonical(self.negative_label),
            }:
                raise ValueError("label_outside_threshold_classes")
            if self.metric == "record_f1" and self.record_key:
                self._check_record_keys(example.expected)
            if self.metric and not _label_type_ok(self.metric, example.expected):
                raise ValueError("label_type_invalid")
            identity = digest(example.input)
            if identity in seen:
                raise ValueError("duplicate_evaluation_example")
            seen.add(identity)
        return self

    def _check_record_keys(self, records):
        from .json_pointer import pointer_value

        if not isinstance(records, list):
            return
        keys = set()
        for record in records:
            key = canonical(
                [pointer_value(record, pointer, default=None) for pointer in self.record_key]
            )
            if key in keys:
                raise ValueError("duplicate_record_key")
            keys.add(key)


class Preparation(Contract):
    version: Literal["preparation-v1"] = "preparation-v1"
    kind: Literal["description", "example", "request", "trace_call", "sdk"]
    status: Literal["unresolved", "resolved"]
    description: str | None = Field(default=None, min_length=1, max_length=16384)
    source_version: str | None = Field(default=None, max_length=128)
    source_id: str | None = Field(default=None, max_length=256)
    pending_settings: list[Literal["temperature", "max_tokens"]] = Field(
        default_factory=list, max_length=2
    )

    @model_serializer(mode="wrap")
    def compatible_dump(self, handler):
        value = handler(self)
        if not self.pending_settings:
            value.pop("pending_settings", None)
        return value

    @model_validator(mode="after")
    def check(self):
        if self.kind == "description" and not self.description:
            raise ValueError("invalid_description")
        return self


class ReferenceBinding(Contract):
    version: Literal["function-reference-binding-v1"] = "function-reference-binding-v1"
    connection_id: UUID
    logical_model: str
    mapping_revision: int = Field(ge=1)
    provider_slug: str
    endpoint_host: str
    upstream_model: str

    @model_serializer(mode="wrap")
    def safe_identity(self, handler):
        value = handler(self)
        value["connection_id"] = str(self.connection_id)
        return value


class FunctionSpec(Contract):
    version: Literal["function-spec-v1"] = "function-spec-v1"
    # Values are frozen identities; see output_limits.py before adding one.
    output_token_policy: Literal[
        "request",
        "endpoint_max",
        "bounded_8192",
        "context_measured_v1",
        "context_measured_v2",
    ] = "request"
    reference_binding: ReferenceBinding | None = None
    calls: list[RequestContract] = Field(min_length=1, max_length=3)
    inputs: dict[str, Any]
    input_schema: dict[str, Any] | None = None
    wire_mode: Literal["messages", "input_object", "template"] = "messages"
    template_syntax: Literal["python_format", "jinja_variables"] | None = None
    bindings: list[Binding] = Field(default_factory=list, max_length=64)
    output_schema: dict[str, Any]
    evaluator: Literal["exact", "records", "decimal", "rubric"] = "exact"
    evaluation_pending: bool = False
    rubric: Rubric | None = None
    evaluation: EvaluationFunction | None = None
    preparation: Preparation | None = None

    @model_serializer(mode="wrap")
    def compatible_dump(self, handler):
        value = handler(self)
        if self.reference_binding is None:
            value.pop("reference_binding", None)
        if self.output_token_policy == "request":
            value.pop("output_token_policy", None)
        if not self.evaluation_pending:
            value.pop("evaluation_pending", None)
        if self.rubric is None:
            value.pop("rubric", None)
        if self.evaluation is None:
            value.pop("evaluation", None)
        if self.input_schema is None:
            value.pop("input_schema", None)
        if self.preparation is None:
            value.pop("preparation", None)
        if self.template_syntax is None:
            value.pop("template_syntax", None)
        return value

    assumptions: list[str] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def check(self):
        if self.wire_mode == "template":
            from .templates import parse_template

            if len(self.calls) != 1 or self.template_syntax is None:
                raise ValueError("single_call_template_required")
            expected = set()
            for index, message in enumerate(self.calls[0].messages):
                parsed = parse_template(message.content, self.template_syntax)
                expected.update(
                    (index, token, "input#/" + name)
                    for token, name in zip(parsed.placeholders, parsed.names)
                )
            actual = {(b.message, b.placeholder, b.source) for b in self.bindings}
            if actual != expected:
                raise ValueError("template_binding_mismatch")
        elif self.template_syntax is not None:
            raise ValueError("template_mode_required")
        if self.reference_binding and any(
            call.model != self.reference_binding.logical_model for call in self.calls
        ):
            raise ValueError("reference_single_model_required")
        bounded_schema(self.output_schema)
        if self.input_schema is not None:
            bounded_schema(self.input_schema)
            if not Draft202012Validator(self.input_schema).is_valid(self.inputs):
                raise ValueError("invalid_example_inputs")
        if self.output_schema.get("type") not in {"object", "array"}:
            raise ValueError("unsupported_evaluator")
        request = self.model_dump()
        request.pop("evaluation", None)
        if len(canonical(request).encode()) > 262144:
            raise ValueError("request_too_large")
        for call in self.calls:
            from .tool_wire import request_schema

            schema = request_schema(call.model_dump(exclude_none=True))
            if schema:
                bounded_schema(schema)
        if any(call.tools is not None for call in self.calls):
            if len(self.calls) != 1 or self.wire_mode not in {"messages", "template"}:
                raise ValueError("single_call_output_tool_required")
            if request_schema(self.calls[0].model_dump(exclude_none=True)) != self.output_schema:
                raise ValueError("output_tool_schema_mismatch")
        ids = [c.id for c in self.calls]
        if len(set(ids)) != len(ids) or "input" in ids:
            raise ValueError("duplicate_call_id")
        occupied = set()
        for binding in self.bindings:
            if binding.call not in ids:
                raise ValueError("unknown_call")
            call = self.calls[ids.index(binding.call)]
            root, _ = reference(binding.source)
            if root != "input" and root not in ids[: ids.index(binding.call)]:
                raise ValueError("forward_reference")
            if binding.message >= len(call.messages):
                raise ValueError("missing_message")
            content = call.messages[binding.message].content
            if content.count(binding.placeholder) != 1 and self.wire_mode != "template":
                raise ValueError("ambiguous_placeholder")
            key = (binding.call, binding.message, binding.placeholder)
            if key in occupied:
                raise ValueError("duplicate_binding")
            occupied.add(key)
        for call in self.calls:
            for index, message in enumerate(call.messages):
                spans = sorted(
                    (message.content.index(b.placeholder), len(b.placeholder))
                    for b in self.bindings
                    if b.call == call.id and b.message == index
                )
                if any(a + length > b for (a, length), (b, _) in zip(spans, spans[1:])):
                    raise ValueError("overlapping_binding")
        return self


class Step(Contract):
    id: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_]{0,63}$")
    op: Literal[
        "branch",
        "merge",
        "return",
        "copy",
        "parse_json",
        "object",
        "lookup",
        "compare",
        "filter",
        "map",
        "decimal_sum",
        "decimal_add",
        "decimal_multiply",
        "generated_kernel",
        "llm",
        "choice",
        "extract",
        "normalize_text",
        "chunk_text",
        "retrieve_candidates",
        "exact_links",
        "resolve_links",
        "assemble_records",
        "align_spans",
        "project_records",
        "merge_spans",
        "merge_records",
        "merge_labels",
    ]
    inputs: list[str] = Field(default_factory=list, max_length=32)
    config: dict[str, Any] = Field(default_factory=dict)
    model: str | None = None
    call: str | None = None
    instruction_mode: Literal["original_call_ref", "append_instruction", "replacement_messages"] = (
        "original_call_ref"
    )
    prompt: str | None = Field(default=None, max_length=16384)
    messages: list[Message] | None = None
    output_schema: dict[str, Any] | None = None
    output_binding: Literal["call", "stage"] = "call"
    # Fan a paid step out over a chunk array. Omitted unless set, so older
    # plan digests stay byte-identical.
    over: str | None = None
    max_chunks: int | None = Field(default=None, ge=1, le=16)

    next: str | None = None
    predicate: Predicate | None = None
    true_next: str | None = None
    false_next: str | None = None
    incoming: dict[str, str] | None = None
    bind_call: str | None = None

    @model_serializer(mode="wrap")
    def compatible_dump(self, handler):
        value = handler(self)
        for key in ("next", "predicate", "true_next", "false_next", "incoming", "bind_call"):
            if getattr(self, key) is None:
                value.pop(key, None)
        if self.output_binding == "call":
            value.pop("output_binding", None)
        if self.over is None:
            value.pop("over", None)
        if self.max_chunks is None:
            value.pop("max_chunks", None)
        return value


QUESTION_TYPES = frozenset({"boolean", "choice", "noul", "score"})

SCORE_LEVELS = (2, 10)


def check_question(question) -> None:
    """Raise ``invalid_primitive_configuration`` for a question this image cannot serve.

    ``noul``: ``{"type", "threshold"}`` with ``0 < threshold < 1``; the answer is
    ``true`` iff the returned probability exceeds it. ``score``: ``{"type",
    "levels", "reduce"}`` with 2-10 unique string levels and ``reduce:
    nearest_level``; the answer is the level at ``floor(score + 0.5)``.
    """
    kind = question.get("type") if isinstance(question, dict) else None
    if kind not in QUESTION_TYPES:
        raise ValueError("invalid_primitive_configuration")
    if kind == "noul":
        threshold = question.get("threshold")
        if (
            set(question) != {"type", "threshold"}
            or isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not 0 < threshold < 1
        ):
            raise ValueError("invalid_primitive_configuration")
    if kind == "score":
        levels = question.get("levels")
        if (
            set(question) != {"type", "levels", "reduce"}
            or question.get("reduce") != "nearest_level"
            or not isinstance(levels, list)
            or not SCORE_LEVELS[0] <= len(levels) <= SCORE_LEVELS[1]
            or any(not isinstance(level, str) or not level for level in levels)
            or len(set(levels)) != len(levels)
        ):
            raise ValueError("invalid_primitive_configuration")


SEMANTIC_PLAN_VERSIONS = frozenset({"function-plan-v3", "function-plan-v4", "function-plan-v5"})


def semantic_plan(version: str) -> bool:
    return version in SEMANTIC_PLAN_VERSIONS


class Plan(Contract):
    version: Literal[
        "function-plan-v1",
        "function-plan-v2",
        "function-plan-v3",
        "function-plan-v4",
        "function-plan-v5",
    ] = "function-plan-v1"
    steps: list[Step] = Field(min_length=1, max_length=16)
    fallback: bool = False
    max_items: int = Field(default=1000, ge=1, le=10000)

    @model_validator(mode="after")
    def check(self):
        from .conditional import CONTROL_OPS, validate_graph

        conditional = self.version == "function-plan-v5"
        seen = {"input"}
        learned = 0
        for step in self.steps:
            from .semantic import OPERATORS, validate_config

            if not conditional and (
                step.op in CONTROL_OPS
                or any(
                    getattr(step, key) is not None
                    for key in (
                        "next",
                        "predicate",
                        "true_next",
                        "false_next",
                        "incoming",
                        "bind_call",
                    )
                )
            ):
                raise ValueError("control_flow_requires_v5")
            if step.op == "generated_kernel":
                from .research_kernels import validate_config

                validate_config(step)
            if step.op in OPERATORS:
                if not semantic_plan(self.version):
                    raise ValueError("operator_requires_v3")
                if step.op in {"merge_spans", "merge_records", "merge_labels"} and (
                    self.version not in {"function-plan-v4", "function-plan-v5"}
                ):
                    raise ValueError("operator_requires_v4")
                validate_config(step.op, step.inputs, step.config)
            if step.output_binding == "stage" and not semantic_plan(self.version):
                raise ValueError("stage_binding_requires_v3")
            if (
                step.op in {"choice", "extract"}
                and step.config.get("mode") in {"pairs", "spans", "records"}
                and not semantic_plan(self.version)
            ):
                raise ValueError("operator_requires_v3")
            if step.over or step.max_chunks is not None:
                if self.version != "function-plan-v4":
                    raise ValueError("fanout_requires_v4")
                if not step.over or reference(step.over)[0] not in seen:
                    raise ValueError("forward_reference")
            if step.config.get("view") not in {None, "full", "bindings"}:
                raise ValueError("invalid_primitive_configuration")
            if step.op == "choice" and isinstance(step.config.get("questions"), dict):
                for question in step.config["questions"].values():
                    check_question(question)
            if step.config.get("view") == "bindings" and not semantic_plan(self.version):
                raise ValueError("view_requires_v3")
            if step.id in seen:
                raise ValueError("duplicate_step")
            for ref in step.inputs:
                if reference(ref)[0] not in seen:
                    raise ValueError("forward_reference")
            if step.output_schema:
                bounded_schema(step.output_schema)
            if step.op in {"llm", "choice", "extract"}:
                learned += 1
                if not step.model or not step.call:
                    raise ValueError("model_binding_required")
                if step.instruction_mode == "append_instruction" and not step.prompt:
                    raise ValueError("prompt_required")
                if step.instruction_mode == "replacement_messages" and not step.messages:
                    raise ValueError("messages_required")
            elif step.model or step.prompt or step.messages:
                raise ValueError("deterministic_step_has_prompt")
            seen.add(step.id)
        if conditional:
            validate_graph(self)
        elif learned > 3:
            raise ValueError("model_call_limit")
        return self
