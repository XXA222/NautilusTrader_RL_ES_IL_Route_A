from __future__ import annotations
import pytest
from rles_route_a.contracts import PointInTimeObservation


@pytest.fixture
def observed():
    def make(ts: int = 1_000_000_000, *, episode: str = "e1", fold: str = "f1",
             pos: float = 0., hourly: float = .01, daily: float = .02, vol: float = .01,
             dd: float = .01, pnl: float = .0, pending: tuple[str, ...] = ()):
        market = {"trend_1h": hourly, "trend_1d": daily, "vol_1m": vol,
                  "breakout_1h": 0., "zscore_1m": 0.}
        return PointInTimeObservation(
            episode_id=episode, fold_id=fold, decision_ts_ns=ts,
            market=market, feature_available_at_ns={key: ts - 1 for key in market},
            account={"net_exposure": pos, "drawdown": dd,
                     "unrealized_pnl_fraction": pnl}, pending_order_ids=pending)
    return make
