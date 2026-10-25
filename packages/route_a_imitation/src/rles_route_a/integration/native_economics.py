"""Optional account-economics consistency check for no-external-cashflow steps.

Native ledger L3/L6 remains authoritative. Funding, commission, slippage must
already be reflected in host equity. Do not deduct them twice.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import isclose, isfinite, log


@dataclass(frozen=True)
class NativeRewardReceipt:
    prior_equity: float
    next_equity: float
    recorded_r_econ: float
    has_external_cashflow: bool
    host_cashflow_adjustment_receipt: str | None = None

    def validate(self, *, tolerance: float = 1e-8) -> None:
        if (not all(isfinite(v) for v in (self.prior_equity, self.next_equity,
                                          self.recorded_r_econ))
                or self.prior_equity <= 0 or self.next_equity <= 0):
            raise ValueError("invalid native cash-equity facts")
        if self.has_external_cashflow:
            # Native host MUST have separately verified cashflow adjustment;
            # an equity ratio alone cannot certify return across deposits.
            if not self.host_cashflow_adjustment_receipt:
                raise ValueError("external cashflow needs L6 provenance and adjustment")
            return
        expected = log(self.next_equity) - log(self.prior_equity)
        if not isclose(expected, self.recorded_r_econ, rel_tol=1e-7, abs_tol=tolerance):
            raise ValueError("native log-reward/equity conservation failed")
