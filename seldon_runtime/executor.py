"""Bounded interpreter used unchanged by evaluation, Try and serving.

Model transport is injected; this module has no worker or optimizer dependencies.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from jsonschema import Draft202012Validator

from .contracts import FunctionSpec, Plan, Step, canonical, digest, reference, semantic_plan
from .provider_error import ProviderError


def pointer(value: Any, path: str) -> Any:
    for raw in path.split("/")[1:] if path else []:
        key = raw.replace("~1", "/").replace("~0", "~")
        try:
            if isinstance(value, list):
                if not key.isdigit() or str(int(key)) != key:
                    raise ValueError("invalid_array_index")
                value = value[int(key)]
            else:
                value = value[key]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("missing_binding") from exc
    return value


def resolve(ref: str, values: dict[str, Any]) -> Any:
    root, path = reference(ref)
    if root not in values:
        raise ValueError("missing_binding")
    return pointer(values[root], path)


def validate_inputs(spec: FunctionSpec, inputs: dict, plan: Plan | None = None):
    """Check all external dependencies before the first paid step, including fallback."""
    if not isinstance(inputs, dict) or len(canonical(inputs).encode()) > 262144:
        raise ValueError("invalid_function_inputs")
    if spec.input_schema and not Draft202012Validator(spec.input_schema).is_valid(inputs):
        raise ValueError("invalid_function_inputs")
    refs = [b.source for b in spec.bindings]
    if plan:
        if semantic_plan(plan.version):
            from .attempt_bound import program_attempt_bound

            program_attempt_bound(plan, spec)
        if plan.version == "function-plan-v2" or semantic_plan(plan.version):
            from .program_validation import validate_program

            validate_program(plan, spec)
        # Conditional nodes validate their data when the selected path visits
        # them. Unvisited optional inputs must not become global requirements.
        if plan.version != "function-plan-v5":
            refs.extend(ref for step in plan.steps for ref in step.inputs)
    for ref in refs:
        if reference(ref)[0] == "input":
            resolve(ref, {"input": inputs})
    if (
        plan
        and plan.version != "function-plan-v5"
        and (plan.version == "function-plan-v2" or semantic_plan(plan.version))
    ):
        for step in plan.steps:
            for ref in step.inputs:
                if reference(ref)[0] != "input":
                    continue
                value = resolve(ref, {"input": inputs})
                if (
                    (step.op in {"map", "filter", "decimal_sum"} and not isinstance(value, list))
                    or (step.op == "parse_json" and not isinstance(value, str))
                    or (step.op == "lookup" and not isinstance(value, dict))
                ):
                    raise ValueError("invalid_function_inputs")
    # Render external-only calls now so expanded bodies fail before paid dispatch.
    for call in spec.calls:
        if all(reference(b.source)[0] == "input" for b in spec.bindings if b.call == call.id):
            render_request(spec, call.id, inputs, {})


def render_request(spec: FunctionSpec, call_id: str, inputs: dict, prior: dict) -> dict:
    call = next((c for c in spec.calls if c.id == call_id), None)
    if call is None:
        raise ValueError("unknown_call")
    result = call.model_dump(exclude={"id", "model"}, exclude_none=True)
    if spec.template_syntax is not None:
        from .templates import parse_template

        for index, message in enumerate(call.messages):
            template = parse_template(message.content, spec.template_syntax)
            values = {name: inputs[name] for name in template.names}
            if any(not isinstance(value, str) for value in values.values()):
                raise ValueError("invalid_function_inputs")
            result["messages"][index]["content"] = template.render(values)
        if len(canonical(result).encode()) > 262144:
            raise ValueError("expanded_request_too_large")
        return result
    # Resolve against original text once: inserted data cannot become template code.
    by_message = {}
    for binding in spec.bindings:
        if binding.call == call_id:
            value = resolve(binding.source, {**prior, "input": inputs})
            by_message.setdefault(binding.message, []).append(
                (binding.placeholder, value if isinstance(value, str) else canonical(value))
            )
    for index, replacements in by_message.items():
        original = result["messages"][index]["content"]
        spans = sorted((original.index(key), len(key), value) for key, value in replacements)
        pieces, end = [], 0
        for start, length, value in spans:
            if start < end:
                raise ValueError("overlapping_binding")
            pieces.extend([original[end:start], value])
            end = start + length
        pieces.append(original[end:])
        result["messages"][index]["content"] = "".join(pieces)
    if len(canonical(result).encode()) > 262144:
        raise ValueError("expanded_request_too_large")
    return result


def decimal(value: Any) -> Decimal:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("invalid_decimal") from exc
    if not result.is_finite():
        raise ValueError("invalid_decimal")
    return result


def validate_kernel_config(op, config):
    allowed = {
        "copy": set(),
        "parse_json": set(),
        "object": {"fields"},
        "lookup": {"key"},
        "compare": {"operator"},
        "filter": {"path", "equals"},
        "map": {"path"},
        "decimal_sum": set(),
        "decimal_add": set(),
        "decimal_multiply": set(),
    }
    if op not in allowed or set(config) - allowed[op]:
        raise ValueError("unsupported_kernel_configuration")


def kernel(op: str, args: list, config: dict, max_items: int, diagnostics=None):
    from .semantic import OPERATORS
    from .semantic import kernel as semantic_kernel

    if op in OPERATORS:
        return semantic_kernel(op, args, config, diagnostics)
    validate_kernel_config(op, config)
    if any(isinstance(a, list) and len(a) > max_items for a in args):
        raise ValueError("item_limit")
    if op == "copy" and len(args) == 1:
        return args[0]
    if op == "parse_json" and len(args) == 1:
        return json.loads(
            args[0], parse_constant=lambda _: (_ for _ in ()).throw(ValueError("invalid_json"))
        )
    if op == "object" and len(config.get("fields", [])) == len(args):
        fields = config["fields"]
        if len(set(fields)) != len(fields):
            raise ValueError("duplicate_field")
        return dict(zip(fields, args, strict=True))
    if op == "lookup" and len(args) == 1:
        return args[0].get(config["key"])
    if op == "compare" and len(args) == 2:
        if config.get("operator", "eq") == "eq":
            return args[0] == args[1]
        if config.get("operator") == "lt":
            return decimal(args[0]) < decimal(args[1])
    if op == "map" and len(args) == 1:
        return [pointer(a, config["path"]) for a in args[0]]
    if op == "filter" and len(args) == 1:
        return [a for a in args[0] if pointer(a, config["path"]) == config["equals"]]
    if op == "decimal_sum" and len(args) == 1:
        return str(sum((decimal(a) for a in args[0]), Decimal(0)))
    if op == "decimal_add" and len(args) == 2:
        return str(decimal(args[0]) + decimal(args[1]))
    if op == "decimal_multiply" and len(args) == 2:
        return str(decimal(args[0]) * decimal(args[1]))
    raise ValueError("invalid_kernel_inputs")


@dataclass(frozen=True)
class Execution:
    output: Any
    calls: int
    elapsed_ms: float
    fallback_used: bool = False
    fallback_reason: str | None = None
    chunked: bool = False
    path: tuple[str, ...] = ()
    escalation_used: bool = False
    timing_available: bool = True
    # Counts from drop-mode ``align_spans`` steps on the answering path
    # (``quoted_records``, ``dropped_quotations``, ``dropped_reasons``); ``None``
    # when no such step ran. Never contains quotation text.
    diagnostics: dict | None = None
    # Internal durable evidence: a successful retry does not settle earlier attempts.
    attempt_accounting: dict | None = None


def merge_diagnostics(total, step):
    """Sum per-step drop counts; ``total`` is updated in place."""
    total["quoted_records"] = total.get("quoted_records", 0) + step["quoted_records"]
    total["dropped_quotations"] = total.get("dropped_quotations", 0) + step["dropped_quotations"]
    if "repaired_contexts" in step:
        total["repaired_contexts"] = total.get("repaired_contexts", 0) + step["repaired_contexts"]
    reasons = total.setdefault("dropped_reasons", {})
    for reason, count in step["dropped_reasons"].items():
        reasons[reason] = reasons.get(reason, 0) + count
    return total


def failure_reason(exc):
    """Sanitized typed code for a failed execution step; never upstream text."""
    from .deadline import ServingCapacity

    if isinstance(exc, ServingCapacity):
        return "provider_capacity"
    if isinstance(exc, ProviderError):
        return exc.code if re.fullmatch(r"[a-z_]{1,80}", exc.code or "") else "provider_error"
    if isinstance(exc, json.JSONDecodeError):
        return "invalid_provider_output"
    if isinstance(exc, TimeoutError):
        return "execution_timeout"
    if isinstance(exc, ValueError) and re.fullmatch(r"[a-z_]{1,80}", str(exc)):
        return str(exc)
    return "execution_failed"


def evidence_value(value):
    data = canonical(value)
    return {
        "digest": digest(value),
        "bytes": len(data.encode()),
        "value": value if len(data.encode()) <= 8192 else None,
        "content_omitted": len(data.encode()) > 8192,
    }


def bounded_trace(trace):
    """Bound stored evidence without pretending omitted content was empty."""
    from copy import deepcopy

    result = deepcopy(trace)
    if len(canonical(result).encode()) > 65536:
        for step in result:
            for item in [*step.get("inputs", []), step.get("output")]:
                if isinstance(item, dict) and "value" in item:
                    item.update(value=None, content_omitted=True)
    return result


@contextmanager
def step_observation(trace, plan, step, phase, inputs, values, call_count):
    if trace is None:
        yield {}
        return
    start = time.monotonic()
    record = {
        "version": "function-step-v1",
        "id": digest([plan.digest, phase, step.id]),
        "step": step.id,
        "phase": phase,
        "operator": step.op,
        "model": step.model,
        "dependencies": step.inputs,
        "state": "completed",
        "started_unix_ns": str(time.time_ns()),
    }
    initial = call_count()
    if trace is not None:
        record["attempt_start"] = 1 + sum(t["attempt_count"] for t in trace)
        record["inputs"] = [evidence_value(resolve(r, values)) for r in step.inputs]
    try:
        yield record
    except BaseException as exc:
        record["state"] = "failed"
        record["error"] = (
            str(exc)
            if isinstance(exc, ValueError)
            and str(exc).replace("_", "").isalpha()
            and len(str(exc)) < 80
            else "step_failed"
        )
        raise
    finally:
        if trace is not None:
            record["elapsed_ms"] = (time.monotonic() - start) * 1000
            record["attempt_count"] = call_count() - initial
            trace.append(record)


def _bound_messages(spec, step, values):
    if step.inputs:
        bound = [resolve(ref, values) for ref in step.inputs]
    else:
        bound = [
            resolve(binding.source, values)
            for binding in spec.bindings
            if binding.call == step.call and binding.source.startswith("input#")
        ]
    return [{"role": "user", "content": canonical({"inputs": bound})}]


def _chunk_payload(step, payload, chunk):
    text = chunk.get("text") if isinstance(chunk, dict) else None
    if not isinstance(text, str):
        raise ValueError("invalid_text")
    body = {**payload, "messages": [dict(message) for message in payload["messages"]]}
    if step.op == "extract" and step.config.get("mode") in {"spans", "records"}:
        body["primitive"] = {**body["primitive"], "source": text}
    else:
        body["messages"] = [{"role": "user", "content": text}]
    return body


async def _fanout(step, payload, chunks, provider):
    sem = asyncio.Semaphore(4)

    async def one(chunk):
        async with sem:
            return await provider(step.model, _chunk_payload(step, payload, chunk))

    tasks = [asyncio.create_task(one(chunk)) for chunk in chunks]
    pending = asyncio.gather(*tasks)
    try:
        # Own cancellation here so a second caller cancellation cannot interrupt
        # a provider's settlement after the first failure/cancellation.
        rows = await asyncio.shield(pending)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        drained = asyncio.gather(*tasks, return_exceptions=True)
        while not drained.done():
            try:
                await asyncio.shield(drained)
            except asyncio.CancelledError:
                # Settlement must finish before callers snapshot accounting or
                # begin fallback. Re-raise the original typed failure below.
                continue
        if pending.done() and not pending.cancelled():
            pending.exception()
        raise
    return [
        {
            "chunk_id": chunk.get("id"),
            "start": chunk.get("start"),
            "end": chunk.get("end"),
            "value": row,
        }
        for chunk, row in zip(chunks, rows, strict=True)
    ]


def step_payload(spec, step, inputs, prior, values, *, instruction_envelope=None):
    """Build the same immutable provider request for execution and preflight pricing.

    ``instruction_envelope`` is the run's or revision's frozen rendering contract
    (``optimizer.instruction_envelope``). ``None`` keeps the legacy bytes.
    """
    if instruction_envelope is not None:
        from .contract_versions import INSTRUCTION_ENVELOPE_CONTRACT

        if instruction_envelope != INSTRUCTION_ENVELOPE_CONTRACT:
            raise ValueError("instruction_envelope_unsupported")
    payload = render_request(spec, step.call, inputs, prior)
    if "max_tokens" not in payload:
        # Explicit null freezes provider omission even when a run normally sizes
        # output itself; the request marker also reserves the endpoint ceiling.
        payload["seldon_output_token_policy"] = "request"
    elif spec.output_token_policy != "request":
        payload["seldon_output_token_policy"] = spec.output_token_policy
    if instruction_envelope is not None:
        from .instruction_envelope import render

        notice = render(step)
        if notice is not None:
            payload["messages"].append(notice)
        elif step.instruction_mode == "append_instruction":
            # Steps whose messages are replaced below keep their legacy bytes.
            payload["messages"].append({"role": "developer", "content": step.prompt})
    elif step.instruction_mode == "append_instruction":
        payload["messages"].append({"role": "developer", "content": step.prompt})
    if step.instruction_mode == "replacement_messages":
        # Replacement text is an immutable complete template. Bindings still
        # resolve through the reviewed original messages, supplied as context.
        payload["messages"] = [m.model_dump() for m in step.messages] + [
            {
                "role": "user",
                "content": canonical({"original_request": payload["messages"]}),
            }
        ]
    if step.output_schema:
        for key in ("tools", "tool_choice", "parallel_tool_calls"):
            payload.pop(key, None)
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "stage_output",
                "schema": step.output_schema,
                "strict": True,
            },
        }
    if step.inputs:
        payload["messages"].append(
            {
                "role": "user",
                "content": canonical({"stage_inputs": [resolve(r, values) for r in step.inputs]}),
            }
        )
    if step.op != "llm":
        for key in ("tools", "tool_choice", "parallel_tool_calls"):
            payload.pop(key, None)
        payload["primitive"] = {"kind": step.op, "config": step.config}
        if step.op == "extract" and step.config.get("mode") in {"spans", "records"}:
            if len(step.inputs) != 1:
                raise ValueError("invalid_primitive_source")
            payload["primitive"]["source"] = resolve(step.inputs[0], values)
    if step.config.get("view") == "bindings":
        payload["messages"] = _bound_messages(spec, step, values)
    return payload


async def _execute(
    plan: Plan,
    spec: FunctionSpec,
    inputs: dict,
    provider: Callable[[str, dict], Awaitable[Any]],
    *,
    timeout: float = 55,
    trace=None,
    phase="primary",
    probe=False,
    path_state=None,
    instruction_envelope=None,
) -> Execution:
    start = time.monotonic()
    values, prior, calls = {"input": inputs}, {}, 0
    chunked = False
    conditional = plan.version == "function-plan-v5"
    by_id = {step.id: step for step in plan.steps}
    index, predecessor = 0, None
    path_state = path_state if path_state is not None else {}
    path_state.update(path=[], escalation_used=False)
    learned_seen, after_learned_branch = False, False
    diagnostics = {}
    async with asyncio.timeout(timeout):
        while index < len(plan.steps):
            step = plan.steps[index]
            successor = step.next if conditional else None
            if conditional:
                path_state["path"].append(step.id)
                if step.op == "branch" and learned_seen:
                    after_learned_branch = True
                if step.op in {"llm", "choice", "extract"}:
                    path_state["escalation_used"] |= after_learned_branch
                    learned_seen = True
            with step_observation(
                trace, plan, step, phase, inputs, values, lambda: calls
            ) as record:
                if conditional and step.op == "branch":
                    from .conditional import evaluate_predicate

                    value = evaluate_predicate(step.predicate, values)
                    successor = step.true_next if value else step.false_next
                    record["successor"] = successor
                    record["predicate"] = step.predicate.model_dump(mode="json")
                elif conditional and step.op == "merge":
                    value = resolve(step.incoming[predecessor], values)
                    if step.bind_call:
                        prior[step.bind_call] = value
                    record["predecessor"] = predecessor
                elif conditional and step.op == "return":
                    value = resolve(step.inputs[0], values)
                elif step.op in {"llm", "choice", "extract"}:
                    payload = step_payload(
                        spec,
                        step,
                        inputs,
                        prior,
                        values,
                        instruction_envelope=instruction_envelope,
                    )
                    if step.over:
                        chunks = resolve(step.over, values)
                        cap = step.max_chunks or 16
                        if not isinstance(chunks, list) or len(chunks) > cap:
                            raise ValueError("chunk_limit")
                        if len(chunks) > 1:
                            chunked = True
                        value = await _fanout(step, payload, chunks, provider)
                        calls += len(chunks)
                    elif step.op == "choice" and step.config.get("mode") == "pairs":
                        from .semantic import bounded_rows

                        retrieval = resolve(step.inputs[0], values)
                        pairs = retrieval["pairs"]
                        bounded_rows(pairs, 160, "pair_limit")
                        value = []
                        for offset in range(0, len(pairs), 64):
                            batch = pairs[offset : offset + 64]
                            batch_payload = {
                                **payload,
                                "primitive": {
                                    "kind": "choice",
                                    "config": {
                                        "questions": {p["id"]: {"type": "boolean"} for p in batch},
                                        "mode": "pairs",
                                        "pairs": batch,
                                    },
                                },
                            }
                            calls += 1
                            decisions = await provider(step.model, batch_payload)
                            if (
                                not isinstance(decisions, dict)
                                or set(decisions) != {p["id"] for p in batch}
                                or any(type(v) is not bool for v in decisions.values())
                            ):
                                raise ValueError("invalid_provider_output")
                            value.extend(
                                {"id": p["id"], "match": decisions[p["id"]]} for p in batch
                            )
                    else:
                        calls += 1
                        value = await provider(step.model, payload)
                    call = next(c for c in spec.calls if c.id == step.call)
                    from .tool_wire import request_schema

                    schema = step.output_schema or request_schema(
                        call.model_dump(exclude_none=True)
                    )
                    if not probe and schema:
                        checked = [row["value"] for row in value] if step.over else [value]
                        if any(not Draft202012Validator(schema).is_valid(item) for item in checked):
                            raise ValueError("output_schema")
                    # V3 intermediates are stage data, not outputs of the original
                    # workflow. Only its final implementation can satisfy a call binding.
                    if step.output_binding == "call":
                        prior[step.call] = value
                elif step.op == "generated_kernel":
                    from .research_kernels import require

                    runtime = require()
                    runtime.validate_step(step)
                    value = await runtime.invoke(
                        step, resolve(step.inputs[0], values), provider=provider
                    )
                else:
                    step_diagnostics = {}
                    try:
                        value = kernel(
                            step.op,
                            [resolve(r, values) for r in step.inputs],
                            step.config,
                            plan.max_items,
                            step_diagnostics,
                        )
                        if step_diagnostics:
                            # Counts only: quotations can be PII (D-3).
                            record["diagnostics"] = step_diagnostics
                            merge_diagnostics(diagnostics, step_diagnostics)
                    except (ValueError, TypeError, AttributeError) as exc:
                        if plan.version != "function-plan-v2" and not semantic_plan(plan.version):
                            raise
                        if isinstance(exc, ValueError) and str(exc) in {
                            "merge_conflict",
                            "chunk_limit",
                            "span_limit",
                            "invalid_span",
                            "invalid_chunk_configuration",
                            "invalid_text",
                            "invalid_kernel_configuration",
                        }:
                            raise
                        raise ValueError("deterministic_step_failed") from exc
                if not probe and step.output_schema and not step.over:
                    if not Draft202012Validator(step.output_schema).is_valid(value):
                        raise ValueError("output_schema")
                if not probe and len(canonical(value).encode()) > 262144:
                    raise ValueError("intermediate_size_limit")
                values[step.id] = value
                if trace is not None:
                    record["output"] = evidence_value(value)
            predecessor = step.id
            if conditional:
                if step.op == "return":
                    break
                index = plan.steps.index(by_id[successor])
            else:
                index += 1
        output = values[plan.steps[-1].id]
        if not probe and not Draft202012Validator(spec.output_schema).is_valid(output):
            if trace:
                trace[-1].update(state="failed", error="output_schema")
            raise ValueError("output_schema")
    return Execution(
        output,
        calls,
        (time.monotonic() - start) * 1000,
        chunked=chunked,
        path=tuple(path_state["path"]),
        escalation_used=path_state["escalation_used"],
        diagnostics=diagnostics or None,
    )


# Refusals raised by transport before any reservation or HTTP request (P0-F).
# Serving may answer with the original call because nothing was sent. Compiler
# evaluation does not widen its set: a model disabled mid-run must fail the
# candidate, not produce measurements that are really fallback measurements.
SERVING_PRE_DISPATCH_FALLBACK = frozenset(
    {
        "primitive_adapter_unavailable",
        "primitive_mode_unqualified",
        "pin_version_unsupported",
        "model_capability_contract_unsupported",
        "tool_output_unsupported",
        "wire_profile_unsupported",
        "offering_circuit_open",
    }
)


def execution_call_limits(plan, spec):
    """Shared serving bounds for enforcement and the management presentation."""
    if plan.version == "function-plan-v4":
        return 40 + (len(spec.calls) if plan.fallback else 0), 40
    limit = 8 if plan.version in {"function-plan-v3", "function-plan-v5"} else 6
    return limit, min(5, limit - len(spec.calls)) if plan.fallback else 5


async def execute(
    plan,
    spec,
    inputs,
    provider,
    *,
    timeout=55,
    trace=None,
    contract=None,
    probe=False,
    conditional=None,
    serving=False,
    instruction_envelope=None,
):
    """Run ``plan``. ``serving=True`` is the customer request path (runtime.py);
    only it may fall back on :data:`SERVING_PRE_DISPATCH_FALLBACK` refusals.

    ``instruction_envelope`` is the frozen request rendering of the run or
    revision (``optimizer.instruction_envelope``); ``None`` is the legacy bytes.
    """
    from .contract_versions import (
        CONDITIONAL_CONTRACT,
        CONTEXT_CHUNKING_CONTRACT,
        EVALUATION_FIDELITY_CONTRACT,
        SPAN_GROUNDING_CONTRACT,
    )

    # v6 freezes the conditional capability separately; ``conditional=False``
    # (a v6 run created with conditional workflows off) rejects v5 plans.
    v5_allowed = contract == CONDITIONAL_CONTRACT or (
        contract in {EVALUATION_FIDELITY_CONTRACT, SPAN_GROUNDING_CONTRACT}
        and conditional is not False
    )
    if plan.version == "function-plan-v5" and not v5_allowed:
        raise ValueError("plan_version_unsupported")
    chunking = {
        CONTEXT_CHUNKING_CONTRACT,
        "optimizer-correctness-v4",
        CONDITIONAL_CONTRACT,
        EVALUATION_FIDELITY_CONTRACT,
        SPAN_GROUNDING_CONTRACT,
    }
    if plan.version == "function-plan-v4" and contract not in chunking:
        raise ValueError("plan_version_unsupported")
    if any(step.config.get("view") == "bindings" for step in plan.steps) and (
        contract not in chunking
    ):
        raise ValueError("view_requires_v3")
    from .deadline import admission_deadline

    with admission_deadline(time.monotonic() + timeout):
        return await _execute_with_fallback(
            plan,
            spec,
            inputs,
            provider,
            timeout=timeout,
            trace=trace,
            probe=probe,
            serving=serving,
            instruction_envelope=instruction_envelope,
        )


async def _execute_with_fallback(
    plan,
    spec,
    inputs,
    provider,
    *,
    timeout=55,
    trace=None,
    contract=None,
    probe=False,
    serving=False,
    instruction_envelope=None,
):
    """One deadline covers candidate and observable original-workflow fallback."""
    start = time.monotonic()
    calls = 0
    validate_inputs(spec, inputs, plan)
    fallback_phase = False
    primary_path = {}
    call_lock = asyncio.Lock()

    async def counted(model, payload):
        nonlocal calls
        async with call_lock:
            limit, primary_limit = execution_call_limits(plan, spec)
            if calls + 1 > limit or (
                semantic_plan(plan.version) and not fallback_phase and calls + 1 > primary_limit
            ):
                raise ValueError("call_limit")
            calls += 1
        return await provider(model, payload)

    # Preserve the isolated study's accounting sink through the call-limit wrapper.
    for attribute in ("research_receipts", "research_operation"):
        if hasattr(provider, attribute):
            setattr(counted, attribute, getattr(provider, attribute))

    async with asyncio.timeout(timeout):
        try:
            return await _execute(
                plan,
                spec,
                inputs,
                counted,
                timeout=timeout,
                trace=trace,
                probe=probe,
                path_state=primary_path,
                instruction_envelope=instruction_envelope,
            )
        except (ValueError, ProviderError) as exc:
            eligible = {
                "output_schema",
                "invalid_provider_output",
                # Typed native contract failures keep the fallback eligibility they
                # had when they were reported as invalid_provider_output.
                "native_output_schema_mismatch",
                "invalid_grounded_span",
                "ungrounded_quotation",
                "intermediate_size_limit",
                "provider_unavailable",
                "model_identity_changed",
                "token_limit",
                "context_unverified",
            }
            from .deadline import ServingCapacity

            if serving:
                eligible |= SERVING_PRE_DISPATCH_FALLBACK
            allowed = isinstance(exc, ValueError) and str(exc) in eligible
            # Serving admission refused before dispatch: nothing was sent, so the
            # original workflow may run. Compiler capacity (plain/AdmissionFailure
            # "provider_capacity") stays a retry signal and is never a fallback.
            allowed = allowed or isinstance(exc, ServingCapacity)
            if semantic_plan(plan.version) and isinstance(exc, ValueError):
                allowed = allowed or str(exc) in {
                    "span_limit",
                    "pair_limit",
                    "catalog_limit",
                    "mention_limit",
                    "chunk_limit",
                    "record_limit",
                    "invalid_span",
                    "invalid_primitive_source",
                    "merge_conflict",
                }
            if plan.version == "function-plan-v2" or semantic_plan(plan.version):
                allowed = (
                    allowed
                    or isinstance(exc, json.JSONDecodeError)
                    or str(exc) == "deterministic_step_failed"
                )
                if isinstance(exc, ProviderError) and exc.code == "upstream_context_refused":
                    allowed = exc.provider_dispatched is False
                elif isinstance(exc, ProviderError):
                    refused = (exc.upstream_code or exc.code) in {
                        "content_filter",
                        "provider_refusal",
                        "content_policy_violation",
                    }
                    # A network failure after dispatch may have completed paid work.
                    # Only known non-dispatch or explicit throttle rejection permits
                    # an automatic fallback; uncertainty retains the original hold.
                    unambiguous = exc.provider_dispatched is False or exc.upstream_status == 429
                    allowed = (
                        not refused
                        and unambiguous
                        and (
                            exc.category == "unavailable"
                            or exc.code in {"upstream_unavailable", "rate_limited"}
                        )
                    )
            if not plan.fallback or not allowed:
                raise
            original = Plan(
                steps=[Step(id=c.id, op="llm", model=c.model, call=c.id) for c in spec.calls]
            )
            fallback_phase = True
            remaining = timeout - (time.monotonic() - start)
            result = await _execute(
                original,
                spec,
                inputs,
                counted,
                timeout=max(0, remaining),
                trace=trace,
                phase="fallback",
                instruction_envelope=instruction_envelope,
            )
            return Execution(
                result.output,
                calls,
                (time.monotonic() - start) * 1000,
                True,
                failure_reason(exc),
                path=tuple(primary_path.get("path", [])),
                escalation_used=primary_path.get("escalation_used", False),
            )
