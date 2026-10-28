from __future__ import annotations

from typing import Any

from qa.analyze_route_a_exit_loss_attribution import (
    classify_exit_timing,
    classify_loss,
    cashflow_sign_summary,
    loss_summary,
    side_economics_summary,
)


def _case(
    *,
    net: float,
    gross_bps: float,
    mfe_bps: float,
    mae_bps: float,
    cost_bps: float = 12.0,
    side: int = 1,
    block: str = "block-1",
    gross_cash_pnl: float = 0.0,
    fees: float = 0.0,
    funding: float = 0.0,
    slippage_cost: float = 0.0,
) -> dict[str, Any]:
    return {
        "status": "ORDINARY_COMPLETE",
        "realized_pnl": net,
        "gross_price_return_bps": gross_bps,
        "mfe_bps": mfe_bps,
        "mae_bps": mae_bps,
        "round_trip_cost_bps": cost_bps,
        "teacher_id": "T01_EMA_9_21",
        "exit_variant": "X1",
        "block": block,
        "scenario": "base",
        "side": side,
        "gross_cash_pnl_before_fees_and_funding": gross_cash_pnl,
        "fees": fees,
        "funding": funding,
        "slippage_cost": slippage_cost,
        "equity_currency": "USDT",
        "gross_price_profit_capture_ratio": None,
    }


def test_loss_categories_separate_cost_erasure_giveback_and_adverse_path():
    assert classify_loss(
        _case(net=-1.0, gross_bps=2.0, mfe_bps=5.0, mae_bps=8.0)
    ) == "positive_price_pnl_erased_by_fees_or_funding"
    assert classify_loss(
        _case(net=-1.0, gross_bps=-2.0, mfe_bps=15.0, mae_bps=8.0)
    ) == "large_mfe_giveback_or_potential_late_exit"
    assert classify_loss(
        _case(net=-1.0, gross_bps=-2.0, mfe_bps=5.0, mae_bps=8.0)
    ) == "adverse_excursion_dominant"
    assert classify_loss(
        _case(net=-1.0, gross_bps=-2.0, mfe_bps=5.0, mae_bps=3.0)
    ) == "negative_price_or_weak_edge"
    assert classify_loss(
        _case(net=1.0, gross_bps=2.0, mfe_bps=15.0, mae_bps=3.0)
    ) is None


def test_exit_timing_category_uses_actual_cash_delta_and_holding_delta():
    assert classify_exit_timing(
        x1_hold_minutes=5.0, x2_hold_minutes=10.0,
        x1_net_return_bps=-5.0, x2_net_return_bps=2.0,
    ) == "x1_potentially_early_x2_later_better"
    assert classify_exit_timing(
        x1_hold_minutes=10.0, x2_hold_minutes=5.0,
        x1_net_return_bps=-5.0, x2_net_return_bps=2.0,
    ) == "x1_potentially_late_x2_earlier_better"
    assert classify_exit_timing(
        x1_hold_minutes=5.0, x2_hold_minutes=5.0,
        x1_net_return_bps=1.0, x2_net_return_bps=1.0,
    ) == "net_return_tie"


def test_loss_summary_keeps_development_blocks_separate():
    summary = loss_summary([
        _case(
            net=-1.0, gross_bps=-2.0, mfe_bps=5.0, mae_bps=8.0,
            block="block-1",
        ),
        _case(
            net=1.0, gross_bps=2.0, mfe_bps=15.0, mae_bps=3.0,
            block="block-2",
        ),
    ])
    first = summary["T01_EMA_9_21/X1/block-1/base"]
    second = summary["T01_EMA_9_21/X1/block-2/base"]
    assert first["complete_trades"] == 1
    assert first["loss_trades"] == 1
    assert second["complete_trades"] == 1
    assert second["loss_trades"] == 0


def test_side_economics_keeps_long_short_and_block_separate():
    summary = side_economics_summary([
        _case(
            net=-1.0, gross_bps=2.0, mfe_bps=5.0, mae_bps=8.0,
            side=1, block="block-1", gross_cash_pnl=2.0,
            fees=2.5, funding=-0.5, slippage_cost=0.2,
        ),
        _case(
            net=-2.0, gross_bps=-2.0, mfe_bps=5.0, mae_bps=8.0,
            side=-1, block="block-1", gross_cash_pnl=-2.0,
            fees=0.0, funding=0.0,
        ),
        _case(
            net=1.0, gross_bps=2.0, mfe_bps=15.0, mae_bps=3.0,
            side=1, block="block-2", gross_cash_pnl=1.0,
        ),
    ])
    long = summary["T01_EMA_9_21/X1/block-1/base/LONG"]
    short = summary["T01_EMA_9_21/X1/block-1/base/SHORT"]
    other_block = summary["T01_EMA_9_21/X1/block-2/base/LONG"]
    assert long["currency"] == "USDT"
    assert long["net_pnl"] == -1.0
    assert long["gross_cash_pnl_before_fees_and_funding"] == 2.0
    assert long["fees"] == 2.5
    assert long["funding"] == -0.5
    assert long["slippage_attribution"] == 0.2
    assert short["net_pnl"] == -2.0
    assert other_block["net_pnl"] == 1.0

    scenario_summary = cashflow_sign_summary(summary)["base"]
    assert scenario_summary == {
        "populated_side_cells": 3,
        "positive_net_pnl_cells": 1,
        "negative_net_pnl_cells": 2,
        "zero_net_pnl_cells": 0,
        "gross_positive_net_negative_cells": 1,
        "gross_nonpositive_net_negative_cells": 1,
    }
