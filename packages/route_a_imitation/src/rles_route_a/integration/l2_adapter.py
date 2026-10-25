"""Adapter for ALREADY AUTHORIZED point-in-time L2 features + L3 account.

Never open files or create a second L2 scaler. The host must produce the
market mapping with available_at timestamps and native account state.
"""
from __future__ import annotations
from collections.abc import Mapping
from typing import Any
from rles_route_a.contracts import PointInTimeObservation


CAUSAL_MTF_V5_ROUTE_A_FEATURES = {
    "trend_1h": "closed_return_1h",
    "trend_1d": "closed_return_1d",
    "vol_1m": "volatility_1m",
    "zscore_1m": "zscore_1m",
    "efficiency_15m": "trend_efficiency_15m",
    "efficiency_1h": "trend_efficiency_60m",
}

# Versioned projection for the expanded causal v6 host FeatureSpec. v5 remains
# unchanged; the three flow fields are available only when the host explicitly
# selects causal_mtf_v6. v7 adds the independent rolling-close breakout.
CAUSAL_MTF_V6_ROUTE_A_FEATURES = {
    **CAUSAL_MTF_V5_ROUTE_A_FEATURES,
    "return_1m": "return_1m",
    "return_5m": "return_5m",
    "return_15m": "return_15m",
    "return_60m": "return_60m",
    "volatility_5m": "volatility_5m",
    "volatility_15m": "volatility_15m",
    "volatility_60m": "volatility_60m",
    "volume_5m": "volume_5m",
    "basis": "basis",
    "last_settled_funding": "last_settled_funding",
    "taker_buy_imbalance_5m": "taker_buy_imbalance_5m",
    "taker_buy_imbalance_15m": "taker_buy_imbalance_15m",
    "log_trade_count_5m": "log_trade_count_5m",
}

# v7 adds a causal close-breakout value against the preceding 60 completed
# one-minute bars. It is separate from v5/v6 so historical recipes are stable.
CAUSAL_MTF_V7_ROUTE_A_FEATURES = {
    **CAUSAL_MTF_V6_ROUTE_A_FEATURES,
    "breakout_1h": "breakout_1h",
}


def project_visible_observation(*, fold_id: str, episode_id: str,
                                decision_ts_ns: int,
                                feature_values: Mapping[str, float],
                                feature_available_at_ns: Mapping[str, int],
                                native_account_fields: Mapping[str, float],
                                pending_order_ids: tuple[str, ...] = (),
                                source_id: str = "", snapshot_id: str = "",
                                feature_spec_hash: str = "") -> PointInTimeObservation:
    """Validation is fail-closed if any 1h/1d feature was not yet available."""
    return PointInTimeObservation(
        episode_id=episode_id, decision_ts_ns=decision_ts_ns,
        market=dict(feature_values), feature_available_at_ns=dict(feature_available_at_ns),
        account=dict(native_account_fields), pending_order_ids=tuple(pending_order_ids),
        fold_id=fold_id, source_id=source_id, snapshot_id=snapshot_id,
        feature_spec_hash=feature_spec_hash,
    )


def project_host_visible_context(*, context: Mapping[str, Any], fold_id: str,
                                 episode_id: str,
                                 market_feature_map: Mapping[str, str] | None = None
                                 ) -> PointInTimeObservation:
    """Adapt the Gym bridge's reviewed pre-action L2/L3 context.

    The default names are bound to ``causal_mtf_v5``. Feature values and their
    ``available_at_ns`` come from the existing L2 feature rows; account and
    pending-order facts come from the current native L3 barrier.
    """
    if not isinstance(context, Mapping):
        raise TypeError("host visible context must be a mapping")
    decision_ts_ns = context.get("decision_ts_ns")
    raw_market = context.get("market")
    raw_account = context.get("account")
    if (type(decision_ts_ns) is not int or not isinstance(raw_market, Mapping)
            or not isinstance(raw_account, Mapping)):
        raise ValueError("host context lacks decision-time market/account facts")
    mapping = (CAUSAL_MTF_V5_ROUTE_A_FEATURES if market_feature_map is None
               else dict(market_feature_map))
    if (not isinstance(mapping, Mapping) or any(not isinstance(k, str) or not k.strip()
                           or not isinstance(v, str) or not v.strip()
                           for k, v in mapping.items())
            or len(set(mapping.values())) != len(mapping)):
        raise ValueError("market feature mapping must be explicit and one-to-one")
    values: dict[str, float] = {}
    availability: dict[str, int] = {}
    for route_name, host_name in mapping.items():
        if host_name not in raw_market:
            raise ValueError(f"host causal feature is missing: {host_name}")
        feature = raw_market[host_name]
        if not hasattr(feature, "available_at_ns") or not hasattr(feature, "value"):
            raise TypeError("host market feature must include value and availability time")
        values[route_name] = feature.value
        availability[route_name] = feature.available_at_ns

    account = dict(raw_account)
    # The v4/v5 field is native position PnL divided by native gross notional.
    # Keep the Route A name as a read-only alias with identical units.
    if "unrealized_pnl_pct" in account:
        account["unrealized_pnl_fraction"] = account["unrealized_pnl_pct"]
    for group in ("risk", "execution", "time"):
        fields = context.get(group, {})
        if not isinstance(fields, Mapping):
            raise ValueError(f"host {group} context must be a mapping")
        account.update({f"{group}.{name}": value for name, value in fields.items()})
    pending = context.get("pending_order_ids")
    if not isinstance(pending, tuple):
        raise ValueError("native pending order identities are missing from host context")
    return project_visible_observation(
        fold_id=fold_id, episode_id=episode_id, decision_ts_ns=decision_ts_ns,
        feature_values=values, feature_available_at_ns=availability,
        native_account_fields=account, pending_order_ids=pending,
        source_id=context.get("source_id", ""),
        snapshot_id=context.get("snapshot_id", ""),
        feature_spec_hash=context.get("feature_spec_hash", ""),
    )
