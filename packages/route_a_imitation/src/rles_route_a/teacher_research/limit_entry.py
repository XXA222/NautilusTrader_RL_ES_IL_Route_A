"""Explicit research-only limit-touch full-fill entry assumption."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Literal


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
                       cancel_at_ns: int | None = None,
                       cancel_status: Literal["UNFILLED", "CANCELLED"] = "CANCELLED") -> LimitEntryResult:
    """A touched limit fills fully at its limit, reported at bar end.

    The order activates at the first bar starting strictly after the decision;
    earlier bars are excluded. A cancellation within an active bar makes that
    bar unusable because OHLC cannot establish touch ordering.
    """
    if (type(decision_ts_ns) is not int or type(block_end_ns) is not int
            or block_end_ns <= decision_ts_ns or type(side) is not int or side not in (-1, 1)
            or isinstance(limit_price, bool) or not isinstance(limit_price, (int, float))
            or not isfinite(limit_price) or limit_price <= 0
            or (cancel_at_ns is not None and (type(cancel_at_ns) is not int
                or not decision_ts_ns < cancel_at_ns <= block_end_ns))
            or cancel_status not in ("UNFILLED", "CANCELLED")):
        raise ValueError("invalid limit entry parameters")
    previous_end = None
    last_observed = None
    for bar in bars:
        if (not isinstance(bar, LimitBar) or type(bar.start_ns) is not int
                or type(bar.end_ns) is not int or bar.start_ns >= bar.end_ns
                or not isfinite(bar.low) or not isfinite(bar.high)
                or bar.low <= 0 or bar.high < bar.low
                or (previous_end is not None and bar.start_ns < previous_end)):
            raise ValueError("invalid or overlapping limit bars")
        previous_end = bar.end_ns
        if bar.start_ns <= decision_ts_ns:
            # This bar cannot produce a fill, but still establishes observed
            # coverage through its end when it straddles the decision.
            if bar.end_ns > decision_ts_ns:
                last_observed = bar.end_ns
            continue
        if last_observed is not None and bar.start_ns > last_observed:
            return LimitEntryResult("RIGHT_CENSORED", last_observed)
        if cancel_at_ns is not None and cancel_at_ns <= bar.start_ns:
            if last_observed is not None and last_observed >= cancel_at_ns:
                return LimitEntryResult(cancel_status, cancel_at_ns)
            return LimitEntryResult("RIGHT_CENSORED", last_observed or decision_ts_ns + 1)
        if bar.start_ns >= block_end_ns:
            break
        if bar.end_ns > block_end_ns:
            return LimitEntryResult("RIGHT_CENSORED", bar.start_ns)
        cancel_inside_bar = cancel_at_ns is not None and bar.start_ns < cancel_at_ns < bar.end_ns
        cancel_at_bar_end = cancel_at_ns is not None and cancel_at_ns == bar.end_ns
        if cancel_inside_bar:
            # The order may have touched before cancellation; OHLC cannot
            # determine their ordering, so do not claim a known non-fill.
            return LimitEntryResult("RIGHT_CENSORED", cancel_at_ns)
        touched = ((side == 1 and bar.low <= limit_price)
                   or (side == -1 and bar.high >= limit_price))
        if touched and cancel_at_bar_end:
            # Touch and cancel share the bar boundary; OHLC cannot order them.
            return LimitEntryResult("RIGHT_CENSORED", cancel_at_ns)
        if touched and bar.end_ns < block_end_ns:
            return LimitEntryResult("FILLED", bar.end_ns, bar.end_ns, float(limit_price))
        last_observed = bar.end_ns
        if cancel_at_bar_end:
            return LimitEntryResult(cancel_status, cancel_at_ns)
    if last_observed is not None and last_observed >= block_end_ns:
        # Complete coverage with no eligible fill is a known missed entry,
        # not a zero-return trade or a censored trade.
        return LimitEntryResult("MISSED", block_end_ns)
    # Missing later bars cannot establish either no fill or cancellation.
    return LimitEntryResult("RIGHT_CENSORED",
                            last_observed if last_observed is not None
                            else min(decision_ts_ns + 1, block_end_ns))



def make_limit_entry_provider(*, bars: tuple[LimitBar, ...], price_for_group,
                              cancel_after_ns: int | None = None,
                              cancel_status: Literal["UNFILLED", "CANCELLED"] = "CANCELLED"):
    """Bind a causal limit price to the research-only touch assumption."""
    if not callable(price_for_group):
        raise TypeError("limit price must be computed from the causal candidate")
    if (cancel_after_ns is not None and (type(cancel_after_ns) is not int or cancel_after_ns <= 0)
            or cancel_status not in ("UNFILLED", "CANCELLED")):
        raise ValueError("cancel delay must be a positive number of nanoseconds")

    def entry(group, block_end_ns):
        decision = group.observation.decision_ts_ns
        cancel_at = decision + cancel_after_ns if cancel_after_ns is not None else None
        if cancel_at is not None and cancel_at >= block_end_ns:
            cancel_at = None
        return replay_limit_entry(decision_ts_ns=decision, side=group.side,
                                  limit_price=price_for_group(group), bars=bars,
                                  block_end_ns=block_end_ns, cancel_at_ns=cancel_at,
                                  cancel_status=cancel_status)
    return entry
