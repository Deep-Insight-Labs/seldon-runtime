"""Payload-free local reports, shared by the router and standalone runtime."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .local_models import LOCAL_MODELS

Count = Annotated[int, Field(strict=True, ge=0, le=1_000_000)]
Version = Annotated[
    str, Field(pattern=r"^\d+\.\d+\.\d+(?:[a-z]+\d*)?(?:\.dev\d+)?$", max_length=32)
]
Reason = Literal[
    "provider_unavailable",
    "provider_capacity",
    "provider_refusal",
    "provider_error",
    "primitive_adapter_unavailable",
    "invalid_provider_output",
    "invalid_output",
    "token_limit",
    "context_unverified",
    "execution_timeout",
    "execution_failed",
    "bundle_expired",
    "bundle_invalid",
    "runtime_upgrade_required",
    "conformance_failed",
    "conformance_unavailable",
    "model_identity_changed",
    "function_not_found",
    "unauthorized",
    "local_ineligible",
    "local_ineligible_not_activated",
    "local_ineligible_hosted_only_model",
    "local_ineligible_plan_version",
    "local_ineligible_reference_bound",
    "local_ineligible_disabled",
]
LATENCY_BOUNDS_MS = (10, 100, 500, 1000, 5000)


class LocalReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    revision_id: UUID
    runtime_version: Version
    model_versions: dict[str, str] = Field(max_length=16)

    @model_validator(mode="after")
    def models_known(self):
        if any(
            name not in LOCAL_MODELS or revision != LOCAL_MODELS[name]["hf_revision"]
            for name, revision in self.model_versions.items()
        ):
            raise ValueError("unknown_local_model_version")
        return self


class LocalUsageBatch(LocalReport):
    batch_id: UUID
    requests: Count
    failures: Count = 0
    fallbacks: dict[Reason, Count] = Field(default_factory=dict)
    # Disjoint bins: <=10, <=100, <=500, <=1000, <=5000, >5000 ms.
    latency_histogram: tuple[Count, Count, Count, Count, Count, Count]

    @model_validator(mode="after")
    def consistent_counts(self):
        if (
            self.requests == 0
            or sum(self.latency_histogram) != self.requests
            or self.failures > self.requests
            or sum(self.fallbacks.values()) > self.requests
        ):
            raise ValueError("invalid_local_counts")
        return self


class LocalVerification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    samples: Annotated[int, Field(strict=True, ge=1, le=200)]
    agreement: float = Field(ge=0, le=1, allow_inf_nan=False)
    p50_ms: Annotated[int, Field(strict=True, ge=0, le=55_000)]
    p95_ms: Annotated[int, Field(strict=True, ge=0, le=55_000)]
    passed: bool = Field(strict=True)

    @model_validator(mode="after")
    def consistent_result(self):
        if self.p50_ms > self.p95_ms or self.passed != (self.agreement >= 0.95):
            raise ValueError("invalid_local_verification")
        return self


class LocalStatusReport(LocalReport):
    # Random per Runner lifetime: separates reports without identifying a machine.
    reporting_id: UUID
    verification: LocalVerification | None = None
    state: Literal["serving", "stopped", "expired", "refresh_refused"]
    reason: Reason | None = None


def bounded_reason(reason):
    from typing import get_args

    return reason if reason in get_args(Reason) else "execution_failed"
