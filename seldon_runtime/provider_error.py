"""Client-safe upstream provider failure shared by the router and the interpreter."""

from typing import Any


class ProviderError(Exception):
    """An upstream failure mapped to a client-safe code/message.

    Never carries the raw upstream body, URL, headers, or keys (invariant I6).
    ``status_code`` is the HTTP status the route should return.
    ``category`` is an optional typed failure class used by the executor for
    throttle/fallback decisions (Bedrock Phase 2+): throttled | timeout |
    access | invalid_request | unavailable | internal.
    ``provider_dispatched`` distinguishes wrapper admission failures from an
    upstream call so terminal client denials and attempt evidence remain exact.
    """

    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
        *,
        category: str | None = None,
        retry_after_seconds: int | None = None,
        upstream_status: int | None = None,
        upstream_code: str | None = None,
        provider_dispatched: bool | None = None,
        transport_evidence: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.category = category
        self.retry_after_seconds = retry_after_seconds
        self.upstream_status = upstream_status
        self.upstream_code = upstream_code
        self.provider_dispatched = provider_dispatched
        # Sanitized physical-attempt evidence (stage, transmission state, status);
        # never upstream bodies, URLs, headers or credentials.
        self.transport_evidence = transport_evidence
