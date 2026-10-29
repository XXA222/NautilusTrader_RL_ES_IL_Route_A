"""Project only available 1m indicators while retaining exact causal timestamps."""
from __future__ import annotations

from collections.abc import Mapping

from rles_route_a.integration.l2_adapter import CAUSAL_1M_INDICATORS_V1_ROUTE_A_FEATURES


def bind_optional_1m_context(provider, visible_context):
    """Refresh the native provider's feature map at each single-env barrier.

    Indicator calculators omit unavailable values (for example, undefined
    z-score in a constant-price interval). Individual teachers then report
    INDICATOR_NOT_READY and the model rejects missing-feature candidates.
    """
    if not callable(visible_context):
        raise TypeError("native visible-context callback required")

    def visible(env):
        context = visible_context(env)
        market = context.get("market") if isinstance(context, Mapping) else None
        if not isinstance(market, Mapping):
            raise ValueError("native 1m context lacks market features")
        mapping = {name: host for name, host in CAUSAL_1M_INDICATORS_V1_ROUTE_A_FEATURES.items()
                   if host in market}
        if "bar_close_1m" not in mapping:
            raise ValueError("native context lacks the completed 1m close")
        provider.market_feature_map = mapping
        return context

    provider.visible_context = visible
