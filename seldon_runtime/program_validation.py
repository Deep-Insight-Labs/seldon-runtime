"""Plan graph, kernel configuration and schema-path validation before evaluation."""

from .contracts import reference
from .semantic import OPERATORS, validate_config
from .tool_wire import request_schema


def validate_program(plan, spec):
    """Validate graph, kernel configuration and known schema paths before evaluation."""
    schemas = {"input": spec.input_schema}
    calls = {c.id: c for c in spec.calls}
    completed = {}
    conditional = plan.version == "function-plan-v5"
    if conditional:
        from .conditional import (
            binding_states,
            predicate_refs,
            validate_graph,
            validate_predicate_schema,
        )

        validate_graph(plan)
        call_states = binding_states(plan, spec)
    arity = {
        "copy": 1,
        "parse_json": 1,
        "lookup": 1,
        "compare": 2,
        "filter": 1,
        "map": 1,
        "decimal_sum": 1,
        "decimal_add": 2,
        "decimal_multiply": 2,
    }
    for step in plan.steps:
        if step.op == "generated_kernel":
            from .research_kernels import require

            require().validate_step(step)
        if conditional:
            completed = call_states[step.id]
        if step.op in OPERATORS:
            validate_config(step.op, step.inputs, step.config)
            if (
                step.op == "align_spans"
                and step.config.get("context_policy") == "hint"
                and (spec.evaluation is None or spec.evaluation.span_context_policy != "hint")
            ):
                raise ValueError("span_context_hint_not_authorized")
        if step.op in {*arity, "object"}:
            from .executor import validate_kernel_config

            validate_kernel_config(step.op, step.config)
        if step.op in arity and len(step.inputs) != arity[step.op]:
            raise ValueError("invalid_kernel_inputs")
        if step.op == "object":
            fields = step.config.get("fields")
            if (
                not isinstance(fields, list)
                or len(fields) != len(step.inputs)
                or any(not isinstance(f, str) for f in fields)
                or len(set(fields)) != len(fields)
            ):
                raise ValueError("invalid_kernel_configuration")
        if step.op == "lookup" and not isinstance(step.config.get("key"), str):
            raise ValueError("invalid_kernel_configuration")
        if step.op == "compare" and step.config.get("operator", "eq") not in {"eq", "lt"}:
            raise ValueError("invalid_kernel_configuration")
        if step.op in {"map", "filter"}:
            reference("input#" + step.config.get("path", "invalid"))
            if step.op == "filter" and "equals" not in step.config:
                raise ValueError("invalid_kernel_configuration")
        refs = list(step.inputs)
        if conditional:
            refs.extend((step.incoming or {}).values())
            if step.predicate:
                refs.extend(predicate_refs(step.predicate))
        if step.op in {"llm", "choice", "extract"}:
            if step.call not in calls:
                raise ValueError("unknown_call")
            for b in spec.bindings:
                if b.call == step.call:
                    root, path = reference(b.source)
                    if root != "input" and root not in completed:
                        raise ValueError("missing_intermediate_call")
                    refs.append((completed.get(root, root)) + "#" + path)
            if step.output_binding == "call":
                completed[step.call] = step.id
        resolved_schemas = {}
        for ref in refs:
            root, path = reference(ref)
            schema = schemas.get(root)
            for raw in path.split("/")[1:] if path else []:
                if not schema:
                    break
                key = raw.replace("~1", "/").replace("~0", "~")
                if schema.get("type") == "array":
                    if not key.isdigit():
                        raise ValueError("invalid_array_index")
                    schema = schema.get("items")
                else:
                    properties = schema.get("properties", {})
                    if key not in properties and schema.get("additionalProperties") is False:
                        raise ValueError("missing_binding")
                    schema = properties.get(key)
            resolved_schemas[ref] = schema
        if step.op == "generated_kernel":
            require().validate_step(step, resolved_schemas.get(step.inputs[0]))
        if conditional and step.predicate:
            validate_predicate_schema(step.predicate, resolved_schemas)
        if conditional and step.op == "merge":
            from .components import reject_incompatible_schema

            for ref in step.incoming.values():
                reject_incompatible_schema(resolved_schemas.get(ref), step.output_schema)
        if step.op in OPERATORS:
            from .components import reject_incompatible_schema, semantic_contract

            required_inputs = semantic_contract(step.op)["input_schema"]["prefixItems"]
            for ref, required in zip(step.inputs, required_inputs, strict=True):
                reject_incompatible_schema(resolved_schemas.get(ref), required)
        produced = step.output_schema or (
            semantic_contract(step.op)["output_schema"]
            if step.op in OPERATORS
            else request_schema(calls[step.call].model_dump(exclude_none=True))
            if step.call in calls
            else None
        )
        if conditional and step.op == "branch":
            produced = {"type": "boolean"}
        if conditional and step.op in {"copy", "return"} and not produced:
            produced = resolved_schemas.get(step.inputs[0])
        # A fan-out step returns one row per chunk. Per-item schema stays on the step.
        if step.over:
            produced = {"type": "array", "maxItems": 32, "items": {"type": "object"}}
        schemas[step.id] = produced
    return plan
