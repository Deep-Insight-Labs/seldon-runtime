"""Per-step and whole-program provider-attempt bounds for a plan."""


def _step_attempts(step, version):
    paid = (
        3
        if step.op == "choice" and step.config.get("mode") == "pairs"
        else int(step.op in {"llm", "choice", "extract"})
    )
    if version == "function-plan-v4" and step.over:
        chunks = step.max_chunks or 16
        if not 1 <= chunks <= 16:
            raise ValueError("chunk_limit")
        return paid * chunks
    return paid


def program_attempt_bound(plan, spec):
    if plan.version == "function-plan-v5":
        from .conditional import max_path

        primary = max_path(plan, lambda step: _step_attempts(step, plan.version))
    else:
        primary = sum(_step_attempts(step, plan.version) for step in plan.steps)
    fallback = len(spec.calls) if plan.fallback else 0
    bound = primary + fallback
    if plan.version == "function-plan-v4":
        if primary > 40:
            raise ValueError("call_limit")
        return bound
    if bound > (8 if plan.version in {"function-plan-v3", "function-plan-v5"} else 6):
        raise ValueError("call_limit")
    if plan.version in {"function-plan-v3", "function-plan-v5"} and primary > 5:
        raise ValueError("call_limit")
    return bound
