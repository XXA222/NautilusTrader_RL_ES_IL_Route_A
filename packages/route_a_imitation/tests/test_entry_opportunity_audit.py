import json

import pytest

from rles_route_a.teacher_research.entry_opportunity_audit import (
    MINUTE_NS, audit_entry_opportunities,
)


def _row(index, signal="NONE"):
    ts = (index + 1) * MINUTE_NS
    return {
        "economic_status": "SCENARIO_ONLY", "source_identity": "source",
        "fold_id": "fold", "decision_ts_ns": str(ts),
        "latest_1m_close_ts_ns": str(ts), "latest_1m_available_at_ns": str(ts),
        "indicator_values_json": json.dumps({"bar_close_1m": 100 + index}),
        "raw_signal": signal, "case_id": "",
    }


def test_candidates_horizons_cashflow_and_censoring():
    rows = [_row(i, "LONG_ENTRY" if i == 0 else "SHORT_ENTRY" if i == 120 else "NONE")
            for i in range(121)]
    rows[0]["case_id"] = "case"
    trade = {"trade_case_id": "case", "side": 1, "status": "ORDINARY_COMPLETE",
             "realized_pnl": 0.8, "gross_cash_pnl_before_fees_and_funding": 1.0,
             "fees": 0.3, "funding": 0.1, "slippage_cost": 0.2,
             "mfe_bps": 8.0, "mae_bps": 3.0}
    result = audit_entry_opportunities(rows, [trade])
    long = result["long"]
    assert long["candidate_count"] == long["completed_trades"] == 1
    assert long["net_cash"] == pytest.approx(0.8)
    assert long["gross_price_cash"] == pytest.approx(1.2)
    assert long["horizons"]["5"]["mean_directional_bps"] == pytest.approx(500)
    assert long["horizons"]["120"]["observed"] == 1
    assert result["short"]["horizons"]["5"]["censored"] == 1


def test_missing_bar_and_source_change_fail_closed():
    rows = [_row(0, "LONG_ENTRY"), _row(1), _row(3), _row(4), _row(5)]
    result = audit_entry_opportunities(rows, [])
    assert result["long"]["horizons"]["5"]["censored"] == 1
    rows[-1]["source_identity"] = "other"
    with pytest.raises(ValueError, match="source/fold"):
        audit_entry_opportunities(rows, [])


def test_unjoined_trade_is_rejected():
    rows = [_row(0)]
    trade = {"trade_case_id": "missing", "side": 1, "status": "ORDINARY_COMPLETE"}
    with pytest.raises(ValueError, match="unjoined"):
        audit_entry_opportunities(rows, [trade])
