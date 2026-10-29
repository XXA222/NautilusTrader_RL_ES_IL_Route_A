"""Explicit research-only limit-touch full-fill entry assumption."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True, slots=True)
class LimitBar:
    start_ns: int
    end_ns: int
    low: float
    high: float


@dataclass(frozen=True, slots=True)
class LimitEntryResult:
    status: str
    observed_until_ns: int
    fill_ts_ns: int | None = None
    fill_price: float | None = None


def replay_limit_entry(*, decision_ts_ns: int, side: int, limit_price: float,
                       bars: tuple[LimitBar, ...], block_end_ns: int,
                       cancel_at_ns: int | None = None) -> LimitEntryResult:
    """A touched limit fills fully at its limit, reported at bar end.

    Bars beginning before the decision are excluded. A cancellation within a bar
    makes that whole bar unusable because OHLC cannot establish touch ordering.
    """
    if (type(decision_ts_ns) is not int or type(block_end_ns) is not int
            or block_end_ns <= decision_ts_ns or type(side) is not int or side not in (-1, 1)
            or isinstance(limit_price, bool) or not isinstance(limit_price, (int, float))
            or not isfinite(limit_price) or limit_price <= 0
            or (cancel_at_ns is not None and (type(cancel_at_ns) is not int
                or not decision_ts_ns < cancel_at_ns <= block_end_ns))):
        raise ValueError("invalid limit entry parameters")
    previous_end = None
    for bar in bars:
        if (not isinstance(bar, LimitBar) or type(bar.start_ns) is not int
                or type(bar.end_ns) is not int or bar.start_ns >= bar.end_ns
                or not isfinite(bar.low) or not isfinite(bar.high)
                or bar.low <= 0 or bar.high < bar.low
                or (previous_end is not None and bar.start_ns < previous_end)):
            raise ValueError("invalid or overlapping limit bars")
        previous_end = bar.end_ns
        if bar.start_ns <= decision_ts_ns:
            continue
        if bar.end_ns >= block_end_ns:
            break
        if cancel_at_ns is not None and bar.end_ns >= cancel_at_ns:
            if bar.start_ns < cancel_at_ns:
                # The order may have touched before cancellation; OHLC cannot
                # determine their ordering, so do not claim a known non-fill.
                return LimitEntryResult("RIGHT_CENSORED", cancel_at_ns)
            break
        if (side == 1 and bar.low <= limit_price) or (side == -1 and bar.high >= limit_price):
            return LimitEntryResult("FILLED", bar.end_ns, bar.end_ns, float(limit_price))
    if cancel_at_ns is not None and cancel_at_ns < block_end_ns:
        return LimitEntryResult("CANCELLED", cancel_at_ns)
    return LimitEntryResult("RIGHT_CENSORED", block_end_ns)
