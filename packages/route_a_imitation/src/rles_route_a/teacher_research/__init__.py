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
from .portfolio_replay import (
    CandidateDecisionRecord,
    ResearchEquitySnapshot,
    ResearchFundingSettlement,
    ResearchMinuteBar,
    ResearchPortfolioRecipe,
    ResearchPortfolioResult,
    PortfolioTradeAccounting,
    SequentialAdmission,
    SequentialPortfolioResult,
    admit_single_position_candidates,
    replay_single_position_portfolio,
    simulate_causal_research_portfolio,
)
from .gate_funnel import FrozenGateFunnel, summarize_frozen_gate_funnel
from .opportunity_diagnostics import (
    CalibrationBin,
    CandidatePredictionDiagnostic,
    DiagnosticAblationSet,
    DiagnosticAblationOutcomeAudit,
    OpportunityPredictionReport,
    ScreenOutcomeAudit,
    ScoreDistribution,
    SupportFeatureDiagnostic,
    audit_screen_ablation_outcomes_after_prediction,
    audit_screen_outcomes_after_prediction,
    build_diagnostic_ablation_sets,
    build_opportunity_prediction_report,
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
    "ResearchPortfolioRecipe",
    "ResearchMinuteBar",
    "ResearchFundingSettlement",
    "ResearchEquitySnapshot",
    "CandidateDecisionRecord",
    "ResearchPortfolioResult",
    "simulate_causal_research_portfolio",
    "SequentialAdmission",
    "admit_single_position_candidates",
    "PortfolioTradeAccounting",
    "SequentialPortfolioResult",
    "replay_single_position_portfolio",
    "FrozenGateFunnel",
    "summarize_frozen_gate_funnel",
    "CandidatePredictionDiagnostic",
    "CalibrationBin",
    "DiagnosticAblationSet",
    "DiagnosticAblationOutcomeAudit",
    "ScoreDistribution",
    "SupportFeatureDiagnostic",
    "OpportunityPredictionReport",
    "ScreenOutcomeAudit",
    "build_opportunity_prediction_report",
    "build_diagnostic_ablation_sets",
    "audit_screen_ablation_outcomes_after_prediction",
    "audit_screen_outcomes_after_prediction",
]
