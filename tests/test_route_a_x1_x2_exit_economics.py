from __future__ import annotations

import pytest

from qa.analyze_route_a_x1_x2_exit_economics import _same_entry


def _case(
    *,
    fill_ts: int,
    notional: float,
    status: str = "ORDINARY_COMPLETE",
    pnl: float = 0.0,
) -> dict[str, object]:
    return {
        "entry_fill_ts_ns": fill_ts,
        "side": 1,
        "entry_notional": notional,
        "status": status,
        "realized_pnl": pnl,
        "net_case_return_bps": pnl,
        "gross_price_return_bps": pnl,
        "holding_minutes": 5.0,
        "mfe_bps": 8.0,
        "max_mfe_to_close_giveback_bps": 3.0,
    }


def test_same_entry_requires_rounded_notional_and_complete_pair_for_realized_delta():
    result = _same_entry(
        [
            _case(fill_ts=10, notional=100.1234561, pnl=1.0),
            _case(fill_ts=20, notional=120.0, pnl=2.0),
            _case(fill_ts=30, notional=130.0, pnl=3.0),
        ],
        [
            _case(fill_ts=10, notional=100.1234564, pnl=2.0),
            _case(fill_ts=20, notional=121.0, pnl=4.0),
            _case(
                fill_ts=30, notional=130.0, status="TERMINAL_CENSORED", pnl=99.0,
            ),
        ],
    )

    assert result["same_fill_side_keys_before_notional_filter"] == 3
    assert result["entry_keys_rejected_for_rounded_notional_mismatch"] == 1
    assert result["same_rounded_notional_pairs_all_statuses"] == 2
    assert result["same_rounded_notional_pairs_excluded_from_realized_deltas"] == 1
    assert result["same_exact_entry_complete_pairs"] == 1
    assert result["matched_pair_net_case_pnl_delta_new_minus_old"] == 1.0


def test_same_entry_rejects_duplicate_fill_timestamp_and_side_keys():
    repeated = _case(fill_ts=10, notional=100.0)
    with pytest.raises(RuntimeError, match="duplicate entry-fill timestamp/side key"):
        _same_entry([repeated, dict(repeated)], [])
