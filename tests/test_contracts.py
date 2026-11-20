"""Published execution contracts retain the frozen synthetic plan/spec digests."""

import json
from pathlib import Path

import pytest

from seldon_runtime import contracts

FIXTURES = Path(__file__).parent / "fixtures"
PLANS = {
    p.stem: json.loads(p.read_text()) for p in sorted((FIXTURES / "serving_compat").glob("*.json"))
}
SPECS = json.loads((FIXTURES / "digest_goldens.json").read_text())["specs"]


def test_corpus_covers_v3_v4_v5():
    assert {"function-plan-v3", "function-plan-v4", "function-plan-v5"} <= set(PLANS)


@pytest.mark.parametrize("name", sorted(PLANS))
def test_plan_digest(name):
    stored = PLANS[name]
    assert contracts.Plan.model_validate(stored["plan"]).digest == stored["digest"]


@pytest.mark.parametrize("name", sorted(SPECS))
def test_spec_digest(name):
    stored = SPECS[name]
    assert contracts.FunctionSpec.model_validate(stored["spec"]).digest == stored["digest"]


def test_compiler_policy_and_pricing_are_not_public():
    from seldon_runtime import components, conditional

    for name in ("MetricConstraint", "Policy", "SearchPreferences", "Limits", "ADMISSION_FIELDS"):
        assert not hasattr(contracts, name)
    for name in ("registry", "study_primitive_contract", "primitive_contract", "primitive_prices"):
        assert not hasattr(components, name)
    assert not hasattr(conditional, "program_cost_bound")
    assert not hasattr(conditional, "model_cost_bounds")
