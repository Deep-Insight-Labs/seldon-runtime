"""Explicit process-local research seam; normal compilation/serving cannot execute code.

Only the isolated study worker installs a context. Artifact identity remains in
serialized steps, while executable source stays in its run-scoped artifact store.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from decimal import Decimal
from typing import Any, Protocol

VERSION = "generated-kernel-v1"


class ResearchRuntime(Protocol):
    synthesis_enabled: bool

    def validate_step(self, step, input_schema=None) -> None: ...

    def compute_bound(self, plan, inputs: int) -> tuple[Decimal, float]: ...

    async def invoke(self, step, value: Any, *, provider) -> Any:
        """Await isolated execution, including admission and settled resource receipt."""
        ...

    async def run_agent(self, legacy, **kwargs): ...


_CURRENT: ContextVar[ResearchRuntime | None] = ContextVar("kernel_research_runtime", default=None)


def active():
    return _CURRENT.get()


@contextmanager
def installed(runtime):
    token = _CURRENT.set(runtime)
    try:
        yield runtime
    finally:
        _CURRENT.reset(token)


def generated(plan):
    return any(step.op == "generated_kernel" for step in plan.steps)


def require():
    runtime = active()
    if runtime is None or not runtime.synthesis_enabled:
        raise ValueError("generated_kernel_research_only")
    return runtime


def bound(plan, inputs=1):
    if not generated(plan):
        return Decimal(0), 0.0
    return require().compute_bound(plan, inputs)


def validate_config(step):
    if (
        step.config.get("version") != VERSION
        or set(step.config) != {"version", "revision_id"}
        or not isinstance(step.config.get("revision_id"), str)
        or len(step.config["revision_id"]) != 64
        or any(c not in "0123456789abcdef" for c in step.config["revision_id"])
        or len(step.inputs) != 1
        or step.model is not None
        or step.call is not None
        or step.over is not None
        or step.output_schema is None
    ):
        raise ValueError("invalid_generated_kernel_contract")
