"""Request-deadline context and the serving/compiler admission failure types."""

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar


class ServingCapacity(ValueError):
    """Serving admission refused before any provider dispatch: nothing was sent.

    Message stays ``provider_capacity`` for existing error surfaces. The distinct
    type lets the serving executor fall back; compiler code never raises it.
    """

    def __init__(self):
        super().__init__("provider_capacity")


# Context-local so forks, judges and nested candidate/fallback calls inherit the
# enclosing deadline without changing the public executor/provider interface.
_deadline = ContextVar("compiler_admission_deadline", default=None)


@contextmanager
def admission_deadline(deadline):
    enclosing = _deadline.get()
    value = min(deadline, enclosing) if enclosing is not None else deadline
    token = _deadline.set(value)
    try:
        yield value
    except TimeoutError as exc:
        # Admission proves nothing was dispatched. If the executor deadline
        # interrupts its asynchronous cleanup, asyncio wraps the cancellation
        # and otherwise hides that retryable outcome behind a generic timeout.
        # Restore only this direct chain; provider timeouts and caller-driven
        # cancellation retain their original meaning.
        cancellation = exc.__cause__
        capacity = getattr(cancellation, "__context__", None)
        if isinstance(cancellation, asyncio.CancelledError) and isinstance(
            capacity, AdmissionFailure
        ):
            raise type(capacity)(capacity.admission) from None
        raise
    finally:
        _deadline.reset(token)


class AdmissionFailure(ValueError):
    def __init__(self, diagnostics):
        super().__init__("provider_capacity")
        self.admission = diagnostics
