"""Typed forward-only control flow, shared by validation, execution and budgeting.

This module deliberately imports contracts lazily: Predicate is embedded in Step.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .contract_versions import CONDITIONAL_CONTRACT  # noqa: F401  (re-export)

CONTROL_OPS = frozenset({"branch", "merge", "return"})

LEARNED_OPS = frozenset({"llm", "choice", "extract"})

CONDITIONAL_VERSION = "function-plan-v5"


class Predicate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    operator: Literal["eq", "in", "lt", "le", "gt", "ge", "all", "any"]
    source: str | None = None
    value: Any = None
    clauses: list[Predicate] = Field(default_factory=list, max_length=4)

    @model_validator(mode="after")
    def bounded(self):
        from .contracts import canonical, reference

        def visit(predicate, depth=0):
            if predicate.operator in {"all", "any"}:
                if (
                    depth >= 2
                    or not predicate.clauses
                    or predicate.source
                    or predicate.value is not None
                ):
                    raise ValueError("invalid_predicate")
                return sum(visit(child, depth + 1) for child in predicate.clauses)
            if (
                predicate.clauses
                or predicate.source is None
                or "value" not in predicate.model_fields_set
            ):
                raise ValueError("invalid_predicate")
            reference(predicate.source)
            if predicate.operator == "in":
                if not isinstance(predicate.value, list) or not 1 <= len(predicate.value) <= 64:
                    raise ValueError("invalid_predicate")
                values = predicate.value
            else:
                values = [predicate.value]
            if any(type(value) not in {str, int, float, bool, type(None)} for value in values):
                raise ValueError("predicate_type")
            if any(isinstance(value, float) and not math.isfinite(value) for value in values):
                raise ValueError("predicate_type")
            if len({_kind(value) for value in values}) > 1:
                raise ValueError("predicate_type")
            if (
                predicate.operator in {"lt", "le", "gt", "ge"}
                and _kind(predicate.value) != "number"
            ):
                raise ValueError("predicate_type")
            return 1

        if visit(self) > 4 or len(canonical(self.model_dump()).encode()) > 8192:
            raise ValueError("predicate_complexity_limit")
        return self


def _kind(value):
    if type(value) in {int, float}:
        return "number"
    return {str: "string", bool: "boolean", type(None): "null"}.get(type(value), "invalid")


def predicate_refs(predicate):
    if predicate.source is not None:
        return [predicate.source]
    return [ref for child in predicate.clauses for ref in predicate_refs(child)]


def evaluate_predicate(predicate, values):
    from .executor import resolve

    if predicate.operator in {"all", "any"}:
        # Validate every operand; short circuiting must not hide missing data.
        results = [evaluate_predicate(child, values) for child in predicate.clauses]
        return all(results) if predicate.operator == "all" else any(results)
    actual = resolve(predicate.source, values)
    expected = predicate.value[0] if predicate.operator == "in" else predicate.value
    if _kind(actual) != _kind(expected) or _kind(actual) == "invalid":
        raise ValueError("predicate_type")
    if isinstance(actual, float) and not math.isfinite(actual):
        raise ValueError("predicate_type")
    if predicate.operator == "eq":
        return actual == predicate.value
    if predicate.operator == "in":
        return actual in predicate.value
    if predicate.operator == "lt":
        return actual < predicate.value
    if predicate.operator == "le":
        return actual <= predicate.value
    if predicate.operator == "gt":
        return actual > predicate.value
    return actual >= predicate.value


def successors(step):
    if step.op == "return":
        return ()
    if step.op == "branch":
        return tuple(dict.fromkeys((step.true_next, step.false_next)))
    return (step.next,)


def predecessors(plan):
    incoming = {step.id: [] for step in plan.steps}
    for step in plan.steps:
        for target in successors(step):
            incoming[target].append(step.id)
    return incoming


def max_path(plan, weight):
    """Maximum additive resource along a reachable path; preserves Decimal types."""
    totals = {}
    for step in reversed(plan.steps):
        totals[step.id] = weight(step) + max(
            (totals[target] for target in successors(step)), default=0
        )
    return totals[plan.steps[0].id]


def validate_graph(plan):
    from .contracts import reference

    positions = {step.id: index for index, step in enumerate(plan.steps)}
    returns = [step for step in plan.steps if step.op == "return"]
    if len(returns) != 1 or plan.steps[-1].op != "return":
        raise ValueError("single_return_required")
    for step in plan.steps:
        if step.op == "branch":
            if (
                not step.predicate
                or not step.true_next
                or not step.false_next
                or step.next
                or step.inputs
            ):
                raise ValueError("invalid_branch")
        elif step.predicate or step.true_next or step.false_next:
            raise ValueError("unexpected_control_fields")
        if step.op == "merge":
            if not step.incoming or not step.output_schema or step.inputs:
                raise ValueError("invalid_merge")
        elif step.incoming is not None or step.bind_call is not None:
            raise ValueError("unexpected_control_fields")
        if step.op in CONTROL_OPS and (step.config or step.call or step.model):
            raise ValueError("invalid_control_configuration")
        if step.op == "return" and (step.next or len(step.inputs) != 1):
            raise ValueError("invalid_return")
        for target in successors(step):
            if target not in positions:
                raise ValueError("missing_control_target")
            if positions[target] <= positions[step.id]:
                raise ValueError("forward_control_edge")
    incoming = predecessors(plan)
    available = {}
    for index, step in enumerate(plan.steps):
        parents = incoming[step.id]
        if index and not parents:
            raise ValueError("unreachable_step")
        before = (
            set.intersection(*(available[parent] for parent in parents)) if parents else {"input"}
        )
        refs = list(step.inputs)
        if step.op == "branch":
            refs.extend(predicate_refs(step.predicate))
        if any(reference(ref)[0] not in before for ref in refs):
            raise ValueError("non_dominating_reference")
        if step.op == "merge":
            if set(step.incoming) != set(parents):
                raise ValueError("merge_predecessors")
            for parent, ref in step.incoming.items():
                if reference(ref)[0] not in available[parent]:
                    raise ValueError("non_dominating_reference")
        available[step.id] = before | {step.id}
    if max_path(plan, lambda step: int(step.op in LEARNED_OPS)) > 3:
        raise ValueError("model_call_limit")
    from .attempt_bound import _step_attempts

    if max_path(plan, lambda step: _step_attempts(step, plan.version)) > 5:
        raise ValueError("call_limit")


def binding_states(plan, spec):
    """Only identical call bindings survive joins; merges explicitly rebind them."""
    from .contracts import reference

    incoming, after, before = predecessors(plan), {}, {}
    calls = {call.id for call in spec.calls}
    for step in plan.steps:
        parents = incoming[step.id]
        shared = dict(after[parents[0]]) if parents else {}
        for parent in parents[1:]:
            shared = {
                key: value for key, value in shared.items() if after[parent].get(key) == value
            }
        before[step.id] = dict(shared)
        if step.op in LEARNED_OPS:
            if step.call not in calls:
                raise ValueError("unknown_call")
            for binding in spec.bindings:
                if binding.call == step.call:
                    root, _ = reference(binding.source)
                    if root != "input" and root not in shared:
                        raise ValueError("missing_intermediate_call")
            if step.output_binding == "call":
                shared[step.call] = step.id
        if step.bind_call:
            if step.bind_call not in calls:
                raise ValueError("unknown_call")
            shared[step.bind_call] = step.id
        after[step.id] = shared
    return before


def validate_predicate_schema(predicate, schemas):
    if predicate.operator in {"all", "any"}:
        for child in predicate.clauses:
            validate_predicate_schema(child, schemas)
        return
    schema = schemas.get(predicate.source)
    if not schema or not schema.get("type"):
        return
    expected = predicate.value[0] if predicate.operator == "in" else predicate.value
    types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
    if not any(
        value == _kind(expected) or value == "integer" and _kind(expected) == "number"
        for value in types
    ):
        raise ValueError("predicate_schema_type")


def canonical_graph(plan):
    """Rename node identities and every typed edge/ref; never rewrite user text."""
    from .contracts import reference

    names = {step.id: f"s{index}" for index, step in enumerate(plan.steps)}

    def ref(value):
        root, path = reference(value)
        return names.get(root, root) + "#" + path

    def predicate(value):
        result = dict(value)
        if result.get("source"):
            result["source"] = ref(result["source"])
        result["clauses"] = [predicate(child) for child in result.get("clauses", [])]
        return result

    value = plan.model_dump(mode="json")
    for step in value["steps"]:
        step["id"] = names[step["id"]]
        step["inputs"] = [ref(item) for item in step["inputs"]]
        for key in ("next", "true_next", "false_next"):
            if key in step:
                step[key] = names[step[key]]
        if "incoming" in step:
            step["incoming"] = {names[key]: ref(item) for key, item in step["incoming"].items()}
        if "predicate" in step:
            step["predicate"] = predicate(step["predicate"])
    return value
