from __future__ import annotations
from hashlib import sha256
import pytest
from rles_route_a.contracts import PointInTimeObservation


@pytest.fixture
def observed():
    def make(ts: int = 1_000_000_000, *, episode: str = "e1", fold: str = "f1",
             pos: float = 0., hourly: float = .01, daily: float = .02, vol: float = .01,
             dd: float = .01, pnl: float = .0, pending: tuple[str, ...] = ()):
        market = {"trend_1h": hourly, "trend_1d": daily, "vol_1m": vol,
                  "breakout_1h": 0., "zscore_1m": 0., "efficiency_15m": .5}
        return PointInTimeObservation(
            episode_id=episode, fold_id=fold, decision_ts_ns=ts,
            market=market, feature_available_at_ns={key: ts - 1 for key in market},
            account={"net_exposure": pos, "drawdown": dd,
                     "unrealized_pnl_fraction": pnl,
                     "position_duration_ns": 0.0,
                     "position_avg_px_open": 0.0,
                     "native_last_fill_age_seconds": -1.0}, pending_order_ids=pending,
            source_id="fixture-source",
            snapshot_id=sha256(f"{episode}:{ts}".encode()).hexdigest(),
            feature_spec_hash="0" * 64,
        )
    return make
