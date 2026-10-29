"""Bind modeled costs to the host execution recipe, separately from episode identity."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation


def validate_native_execution(factory, *, scenario, expected_recipe, fee_bps, slippage_bps):
    """Check the host's actual market-order profile and scenario economics."""
    if (not isinstance(expected_recipe, str) or not expected_recipe
            or getattr(factory, "route_a_execution_recipe_hash", None) != expected_recipe):
        raise ValueError("native execution recipe differs from the model label recipe")
    if getattr(factory, "mode", None) != scenario or scenario not in ("base", "stress"):
        raise ValueError("native factory cost scenario mismatch")
    execution = getattr(factory, "execution", None)
    if getattr(execution, "order_type", None) != "MARKET" or not callable(getattr(execution, "fill", None)):
        raise ValueError("next-open research requires a native market-order execution recipe")
    profile = execution.fill(scenario)
    if getattr(profile, "data_granularity", None) != "bar":
        raise ValueError("next-open model requires the preregistered bar fill profile")
    try:
        actual_slippage = Decimal(str(profile.slippage_bps))
        actual_fee = Decimal(str(factory.rule.payload["taker_fee"])) * Decimal("10000") * Decimal(str(profile.fee_multiplier))
        expected_fee, expected_slippage = Decimal(str(fee_bps)), Decimal(str(slippage_bps))
    except (AttributeError, KeyError, TypeError, InvalidOperation) as exc:
        raise ValueError("native fee and slippage evidence is unavailable") from exc
    if (not all(x.is_finite() and x >= 0 for x in
                (actual_slippage, actual_fee, expected_fee, expected_slippage))
            or actual_fee != expected_fee or actual_slippage != expected_slippage):
        raise ValueError("native per-leg fee or slippage differs from the model costs")
