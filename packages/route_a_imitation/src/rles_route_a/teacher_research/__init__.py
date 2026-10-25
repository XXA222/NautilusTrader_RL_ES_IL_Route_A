"""Causal, research-only teacher components.

These helpers produce candidate/provenance records; they do not authorize
orders, load market data, or claim economic validity.
"""

from .candidate_events import (
    OpportunityEvent,
    CausalOpportunityGenerator,
    audit_candidate_decision,
)
from .opportunity_models import (
    FrozenOpportunityModel,
    NetValueEstimate,
    OpportunityOutcome,
    fit_opportunity_model,
    feature_vector_sha256,
)
from .opportunity_labels import (
    CounterfactualOutcomeBatch,
    CounterfactualTradeReplay,
    label_counterfactual_candidates,
)

__all__ = [
    "OpportunityEvent",
    "CausalOpportunityGenerator",
    "audit_candidate_decision",
    "OpportunityOutcome",
    "NetValueEstimate",
    "FrozenOpportunityModel",
    "fit_opportunity_model",
    "feature_vector_sha256",
    "CounterfactualTradeReplay",
    "CounterfactualOutcomeBatch",
    "label_counterfactual_candidates",
]
