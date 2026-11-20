"""Resolve compiler output allowance before identity, reservation and dispatch.

Legacy request policy preserves the existing wire request and token estimate.
Full-request policies budget the complete request and use the selected offering's
remaining context:

* ``endpoint_max`` allows the endpoint's full ``max_output_tokens``.
* ``bounded_8192`` allows at most 8,192 output tokens (reasoning included where
  the endpoint counts it as output).
* ``context_measured_v1`` sizes like ``bounded_8192`` but decides context fit
  from a provider-published token count registered for the offering pin. A
  native primitive's declared allowance must then fit unclamped. Without a
  registered estimator it makes the byte decision and refuses
  ``context_unverified``. The byte bound still prices the hold and TPM estimate.

Native primitives retain their bounded request allowance, capped to their own
endpoint (and to the policy ceiling) when inherited from a full-request LLM call.

A policy value is part of frozen run/spec identity. Never change the ceiling of
an existing policy; add a new policy value instead.

Pure leaf (stdlib + tiktoken + ``contracts``) so the staged ``seldon_runtime``
wheel refuses exactly the requests the hosted transport refuses. Server lookups
(the wire body and a registered token counter) are passed in by ``output_limits``.
"""

from copy import deepcopy

from .contracts import canonical

REQUEST = "request"
ENDPOINT_MAX = "endpoint_max"
BOUNDED_8192 = "bounded_8192"
CONTEXT_MEASURED = "context_measured_v1"
CONTEXT_MEASURED_V2 = "context_measured_v2"
MARKER = "seldon_output_token_policy"
# Internal key carrying a capability run's frozen wire profile into fit and
# admission (the request depends on the host's state form). Never sent.
PROFILE_KEY = "seldon_wire_profile"
# Assumed tokenizer for models that have not published one. The 1.15 factor and
# the flat 32-token margin keep the estimate above an OpenAI encoding of the
# same text. It decides fit only; holds stay on the byte bound.
ASSUMED_TOKEN_ESTIMATE = "o200k_base_assumed"
ASSUMED_TOKEN_MARGIN = 1.15
ASSUMED_TOKEN_PAD = 32
# Policies that size the output allowance themselves instead of honoring the
# request's ``max_tokens``. Callers use this set rather than comparing to one value.
FULL_REQUEST_POLICIES = frozenset(
    {ENDPOINT_MAX, BOUNDED_8192, CONTEXT_MEASURED, CONTEXT_MEASURED_V2}
)
POLICIES = frozenset({REQUEST, *FULL_REQUEST_POLICIES})
# Policy ceiling below the endpoint maximum. ``endpoint_max`` has none.
_CEILINGS = {BOUNDED_8192: 8192, CONTEXT_MEASURED: 8192, CONTEXT_MEASURED_V2: 8192}
_OUTPUT_FIELDS = frozenset({"max_tokens", "max_completion_tokens", "max_output_tokens"})


def assumed_token_count(value) -> int:
    """Upper estimate of provider tokens for ``value`` using ``o200k_base``."""
    import math

    import tiktoken

    text = value if isinstance(value, str) else canonical(value)
    counted = len(tiktoken.get_encoding("o200k_base").encode(text))
    return math.ceil(counted * ASSUMED_TOKEN_MARGIN) + ASSUMED_TOKEN_PAD


def resolve(payload, target, policy, *, body, estimator_for):
    """``output_limits.resolve_payload`` with its server lookups passed in.

    ``body(payload, target)`` is the JSON the provider receives; ``estimator_for(target)``
    is the registered provider token counter, or ``None``.
    """
    selected = payload.get(MARKER, policy)
    if selected == CONTEXT_MEASURED_V2:
        # Fit uses the assumed encoding of the provider body. The byte bound
        # returned by ``_resolve`` still prices the hold.
        return _resolve(
            payload,
            target,
            CONTEXT_MEASURED_V2,
            lambda _ignored: assumed_token_count(body(payload, target)),
        )
    if selected != CONTEXT_MEASURED:
        return _resolve(payload, target, selected)
    estimator = estimator_for(target)
    if estimator is None:
        try:
            return _resolve(payload, target, BOUNDED_8192)
        except ContextOverflow:
            raise ValueError("context_unverified") from None
    return _resolve(payload, target, CONTEXT_MEASURED, estimator)


class ContextOverflow(ValueError):
    """The request does not fit the endpoint context; other limits are separate."""

    def __init__(self):
        super().__init__("token_limit")


def _resolve(payload, target, policy, estimator=None):
    omitted_limit = payload.get(MARKER) == REQUEST and "max_tokens" not in payload
    resolved = deepcopy(payload)
    resolved.pop(MARKER, None)
    resolved.pop(PROFILE_KEY, None)
    selected = policy
    maximum = getattr(target, "max_output_tokens", None)
    context = getattr(target, "context_length", None)
    if (
        selected not in POLICIES
        or type(maximum) is not int
        or maximum < 1
        or type(context) is not int
        or context < 1
    ):
        raise ValueError("token_limit")

    ceiling = min(maximum, _CEILINGS.get(selected, maximum))
    framing = 256 * len(resolved["messages"])
    if selected in FULL_REQUEST_POLICIES and "primitive" not in resolved:
        # Include schema/tool instructions and every other input-bearing field.
        complete_input = {k: v for k, v in resolved.items() if k not in _OUTPUT_FIELDS}
        input_bound = len(canonical(complete_input).encode()) + framing
        occupied = estimator(complete_input) if estimator else input_bound
        output_limit = min(ceiling, context - occupied)
        for alias in _OUTPUT_FIELDS - {"max_tokens"}:
            resolved.pop(alias, None)
        resolved["max_tokens"] = output_limit
    else:
        # Compiler request contracts use max_tokens only. Other aliases would
        # permit dispatch to exceed the amount accounted for by the old path.
        if any(alias in resolved for alias in _OUTPUT_FIELDS - {"max_tokens"}):
            raise ValueError("token_limit")
        input_bound = len(canonical(resolved["messages"]).encode()) + framing
        if resolved.get("tools") is not None:
            # Tool schemas are paid input; old content-only request bounds stay frozen.
            input_bound = (
                len(
                    canonical(
                        {k: v for k, v in resolved.items() if k not in _OUTPUT_FIELDS}
                    ).encode()
                )
                + framing
            )
        if "primitive" in resolved:
            input_bound += len(canonical(resolved["primitive"]).encode())
        occupied = estimator(resolved) if estimator else input_bound
        # Omission is a frozen request choice: reserve the endpoint's possible
        # output without adding a generation option the caller never supplied.
        output_limit = resolved.get(
            "max_tokens", min(maximum, context - occupied) if omitted_limit else 2048
        )
        if selected in FULL_REQUEST_POLICIES:
            if type(output_limit) is not int or output_limit < 1:
                raise ValueError("token_limit")
            output_limit = min(output_limit, ceiling)
            if not estimator:
                output_limit = min(output_limit, context - input_bound)
            resolved["max_tokens"] = output_limit

    if type(output_limit) is not int or output_limit > maximum:
        raise ValueError("token_limit")
    if output_limit < 1 or occupied + output_limit > context:
        raise ContextOverflow()
    return resolved, input_bound, output_limit
