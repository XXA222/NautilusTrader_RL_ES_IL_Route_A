"""Real historical perpetual environment assembly, with no test imports."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from decimal import Decimal
from hashlib import sha256
from bisect import bisect_left, bisect_right
from itertools import islice
import json
from math import gcd
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from rles_contracts.dto import DataQualityGrade, ExecutionPlan
from rles_data.contracts import to_market_event
from rles_data.core import (
    Availability,
    CuratedStore,
    DataError,
    Event,
    FeatureField,
    FeatureSpec,
    KINDS,
    MINUTE_NS,
    Window,
    digest,
)
from rles_data.coverage import CoveragePolicy
from rles_data.features import Scaler, fit_train, make_feature_row
from rles_data.io import read_curated
from rles_execution import (
    InstrumentRules,
    PendingOrder,
    ResolvedExecutionConfig,
    TargetQuantityPlanner,
    to_contract_plan,
)
from rles_nautilus_adapter.perpetual import QUOTE_MAX_AGE_NS, make_perpetual_session
from rles_nautilus_adapter.session import VisibleBatch
from rles_risk import HardRiskConfig, HardRiskGuard, SoftRiskParams, SoftRiskTransform

from .episode import BarrierEpisode, BarrierFrame, EpisodeReplay, RiskActionReview
from .env import NautilusBarrierEnv
from .l2_bridge import L2FrozenScalerAdapter, visible_market_features
from .observation import ObservationBuilder, ObservationSchema
from .reward import RewardConfig


_FUNDING_TIME_HORIZON_NS = 24 * 60 * 60 * 1_000_000_000


class _AppendOnlyNativeHistory:
    """Own every detached native fact once; snapshots expose bounded read-only views."""

    def __init__(self) -> None:
        self.rows: list[dict] = []
        self.timestamps: list[int] = []
        self.rows_by_timestamp: dict[int, list[tuple[int, dict]]] = {}

    def append(self, row: dict, *, timestamp_ns: int | None = None) -> int:
        index = len(self.rows)
        self.rows.append(row)
        if timestamp_ns is not None:
            if type(timestamp_ns) is not int or timestamp_ns < 0:
                raise DataError("native event history has an invalid timestamp")
            if timestamp_ns not in self.rows_by_timestamp:
                self.rows_by_timestamp[timestamp_ns] = []
                position = bisect_left(self.timestamps, timestamp_ns)
                self.timestamps.insert(position, timestamp_ns)
            self.rows_by_timestamp[timestamp_ns].append((index, row))
        return len(self.rows)

    def view(self, *, start: int = 0, end: int | None = None) -> _AppendOnlyNativeHistoryView:
        return _AppendOnlyNativeHistoryView(self, start=start, end=len(self.rows) if end is None else end)


class _AppendOnlyNativeHistoryView:
    """Full cumulative native event history bounded to one exact snapshot revision."""

    __slots__ = ("_history", "_start", "_end")

    def __init__(self, history: _AppendOnlyNativeHistory, *, start: int, end: int) -> None:
        if type(start) is not int or type(end) is not int or not 0 <= start <= end <= len(history.rows):
            raise DataError("native event snapshot view bounds are invalid")
        self._history, self._start, self._end = history, start, end

    def __len__(self) -> int:
        return self._end - self._start

    def __iter__(self):
        return islice(self._history.rows, self._start, self._end)

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            return tuple(self._history.rows[self._start + offset]
                         for offset in range(start, stop, step))
        if type(index) is not int:
            raise TypeError("native history view indexes must be integers or slices")
        normalized = index if index >= 0 else len(self) + index
        if not 0 <= normalized < len(self):
            raise IndexError(index)
        return self._history.rows[self._start + normalized]

    def for_event_interval(self, start_ns: int, end_ns: int):
        if type(start_ns) is not int or type(end_ns) is not int or start_ns > end_ns:
            raise DataError("native event interval is invalid")
        left = bisect_right(self._history.timestamps, start_ns)
        right = bisect_right(self._history.timestamps, end_ns)
        for timestamp in self._history.timestamps[left:right]:
            for index, row in self._history.rows_by_timestamp[timestamp]:
                if self._start <= index < self._end:
                    yield row


class _VersionedNativeOrdersView:
    """Exact full native order state at one revision, structurally shared by barriers."""

    __slots__ = ("_versions", "_revision")

    def __init__(self, versions: dict[str, tuple[list[int], list[dict]]], revision: int) -> None:
        self._versions, self._revision = versions, revision

    def _row_at(self, identity: str) -> dict | None:
        versions = self._versions.get(identity)
        if versions is None:
            return None
        revisions, rows = versions
        position = bisect_right(revisions, self._revision) - 1
        return rows[position] if position >= 0 else None

    def for_ids(self, identities: set[str]) -> tuple[dict, ...]:
        rows = []
        for identity in sorted(identities):
            row = self._row_at(identity)
            if row is not None:
                rows.append(row)
        return tuple(rows)

    def __iter__(self):
        for identity in self._versions:
            row = self._row_at(identity)
            if row is not None:
                yield row

    def __len__(self) -> int:
        return sum(1 for _ in self)

    def __getitem__(self, index):
        if type(index) is not int:
            raise TypeError("native order view indexes must be integers")
        normalized = index if index >= 0 else len(self) + index
        if not 0 <= normalized < len(self):
            raise IndexError(index)
        for position, row in enumerate(self):
            if position == normalized:
                return row
        raise IndexError(index)


def scheduled_funding_time_features(
    decision_ts_ns: int, settlement_times_ns: tuple[int, ...]
) -> dict[str, float]:
    """Encode only the source-declared next funding time, never realized events."""
    if type(decision_ts_ns) is not int or decision_ts_ns <= 0:
        raise DataError("funding time feature requires a positive decision timestamp")
    if (
        not isinstance(settlement_times_ns, tuple)
        or any(type(ts) is not int or ts <= 0 for ts in settlement_times_ns)
        or settlement_times_ns != tuple(sorted(set(settlement_times_ns)))
    ):
        raise DataError("funding time feature requires an ordered immutable schedule")
    future = next((ts for ts in settlement_times_ns if ts > decision_ts_ns), None)
    if future is None:
        return {"funding_settlement_within_24h": 0.0, "funding_settlement_time_to_event_24h": 1.0}
    fraction = min((future - decision_ts_ns) / _FUNDING_TIME_HORIZON_NS, 1.0)
    return {
        "funding_settlement_within_24h": float(future - decision_ts_ns <= _FUNDING_TIME_HORIZON_NS),
        "funding_settlement_time_to_event_24h": fraction,
    }


def _money(value: Any) -> Decimal:
    # Native Money strings carry a currency suffix; account balance dicts may
    # instead carry bare decimal strings. No market-derived balance estimates.
    result = Decimal(str(value).replace(",", "").split()[0])
    if not result.is_finite():
        raise DataError("nonfinite native account amount")
    return result


def signed_quantity(snapshot) -> Decimal:
    return sum(
        (
            Decimal(str(row["quantity"])) * (-1 if row["side"] == "SHORT" else 1)
            for row in snapshot.positions
        ),
        Decimal(0),
    )


class NativeMarkExcursionTracker:
    """Track decision-visible mark MFE/MAE for one native position lifecycle.

    These are sampled only at Gym decision barriers, not intrabar extrema. A
    position identity/open-time/side/entry-price change starts a new sample
    window; missing native facts fail closed rather than being inferred.
    """

    def __init__(self) -> None:
        self._identity: tuple[str, int, str, Decimal] | None = None
        self._last_ts_ns: int | None = None
        self._mfe_bps = Decimal(0)
        self._mae_bps = Decimal(0)

    @classmethod
    def reconstruct_open_position(cls, snapshots, *, position_id: str,
                                  opened_ns: int, side: str):
        """Rebuild decision-sampled excursion state from native barrier history.

        Callers must provide the retained native snapshot interval beginning at
        entry and ending at the current open-position barrier. The method does
        not accept strategy-requested state or infer missing marks from bars.
        """
        if (not isinstance(position_id, str) or not position_id.strip()
                or type(opened_ns) is not int or opened_ns <= 0
                or side not in {"LONG", "SHORT"}):
            raise DataError("native position identity is required for reconstruction")
        tracker = cls()
        started = False
        previous_ts_ns = None
        last_snapshot = None
        for snapshot in snapshots:
            ts_ns = getattr(snapshot, "decision_ts_ns", None)
            if type(ts_ns) is not int or ts_ns <= 0 or (
                    previous_ts_ns is not None and ts_ns <= previous_ts_ns):
                raise DataError("native reconstruction snapshots must be strictly ordered")
            previous_ts_ns = ts_ns
            last_snapshot = snapshot
            positions = tuple(getattr(snapshot, "positions", ()))
            matching = [row for row in positions
                        if row.get("position_id") == position_id]
            if not started:
                if not matching:
                    continue
                row = matching[0]
                if (len(positions) != 1 or row.get("ts_opened") != opened_ns
                        or row.get("side") != side):
                    raise DataError("native position history identity mismatch")
                started = True
            else:
                if len(matching) != 1 or len(positions) != 1:
                    raise DataError("native position history has a gap or closed position")
                row = matching[0]
                if row.get("ts_opened") != opened_ns or row.get("side") != side:
                    raise DataError("native position history identity changed")
            tracker.update(snapshot)
        if not started or last_snapshot is None:
            raise DataError("native position history does not contain the requested position")
        final_positions = tuple(getattr(last_snapshot, "positions", ()))
        if (len(final_positions) != 1
                or final_positions[0].get("position_id") != position_id):
            raise DataError("native position history does not end at the requested open position")
        return tracker

    def update(self, snapshot) -> dict[str, float]:
        if len(snapshot.positions) > 1:
            raise DataError("native mark excursions require one netted position")
        if not snapshot.positions:
            self._identity = None
            self._last_ts_ns = None
            self._mfe_bps = self._mae_bps = Decimal(0)
            return {"position_mfe_bps": 0.0, "position_mae_bps": 0.0}
        position = snapshot.positions[0]
        position_id = position.get("position_id")
        opened = position.get("ts_opened")
        side = position.get("side")
        avg_px_open = position.get("avg_px_open")
        ts_ns = getattr(snapshot, "decision_ts_ns", None)
        if (not isinstance(position_id, str) or not position_id
                or type(opened) is not int or opened <= 0
                or side not in {"LONG", "SHORT"}
                or type(ts_ns) is not int or ts_ns <= 0
                or snapshot.mark is None or avg_px_open is None):
            raise DataError("native position identity, side, entry, mark and timestamp required")
        try:
            entry, mark = Decimal(str(avg_px_open)), Decimal(str(snapshot.mark))
        except Exception as exc:
            raise DataError("native mark excursion price invalid") from exc
        if (not entry.is_finite() or entry <= 0 or not mark.is_finite()
                or mark <= 0):
            raise DataError("native mark excursion price invalid")
        identity = (position_id, opened, side, entry)
        if identity != self._identity:
            self._identity = identity
            self._last_ts_ns = None
            self._mfe_bps = self._mae_bps = Decimal(0)
        if self._last_ts_ns is not None and ts_ns <= self._last_ts_ns:
            raise DataError("native mark excursion timestamps must increase")
        signed_move_bps = (mark - entry) / entry * Decimal(10_000)
        if side == "SHORT":
            signed_move_bps = -signed_move_bps
        self._mfe_bps = max(self._mfe_bps, signed_move_bps, Decimal(0))
        self._mae_bps = max(self._mae_bps, -signed_move_bps, Decimal(0))
        self._last_ts_ns = ts_ns
        return self.current()

    def current(self) -> dict[str, float]:
        """Return the last reconstructed/updated decision-barrier facts."""
        return {
            "position_mfe_bps": float(self._mfe_bps),
            "position_mae_bps": float(self._mae_bps),
        }


def native_balances(snapshot) -> tuple[Decimal, Decimal]:
    balances = snapshot.account_events[-1]["balances"]
    matches = [b for b in balances if b["currency"] == snapshot.currency]
    if len(matches) != 1:
        raise DataError("exactly one settlement-currency native balance required")
    return _money(matches[0]["total"]), _money(matches[0]["free"])


def load_rules(path: str | Path) -> tuple[Event, ...]:
    """Require source bytes and explicit historical availability/effective dates."""
    source = Path(path).resolve()
    if not source.is_file():
        raise DataError(f"historical instrument rules unavailable: {source}")
    document = json.loads(source.read_text(encoding="utf-8"))
    if set(document) != {"sources", "events"} or not document["sources"] or not document["events"]:
        raise DataError("rules manifest requires source documents and normalized events")
    verified = set()
    for evidence in document["sources"]:
        p = Path(evidence["path"])
        p = p if p.is_absolute() else source.parent / p
        actual = sha256(p.read_bytes()).hexdigest()
        if actual != evidence["sha256"] or not evidence.get("provenance"):
            raise DataError("historical rule source integrity/provenance missing")
        verified.add(actual)
    events = []
    for row in document["events"]:
        event = Event(**{**row, "availability_source": Availability(row["availability_source"])})
        if event.kind != "instrument_rule" or event.source_hash not in verified:
            raise DataError("rule event is not bound to a verified historical source")
        events.append(event)
    return tuple(events)


def historical_feature_spec(
    maximum_age_ns: dict[str, int], *, profile: str = "legacy"
) -> FeatureSpec:
    """Return a frozen causal feature recipe selected by the run configuration.

    ``legacy`` preserves the original observation identity for existing bundles.
    ``causal_mtf_v1`` adds closed-bar return and volatility context at longer
    horizons. ``causal_mtf_v2`` also adds bounded trend-efficiency features that
    distinguish persistent directional movement from choppy price paths. The
    ``causal_mtf_v3`` adds a source-scheduled next-settlement countdown to the
    observation schema. ``causal_mtf_v4`` exposes the latest completed 1m
    close to causal experts. ``causal_mtf_v5`` adds the closed 1h/1d returns,
    1m volatility and rolling 1m z-score required by Route A T1/T2.
    ``causal_mtf_v6`` adds causal kline taker-buy imbalance and trade activity.
    ``causal_mtf_v7`` adds a completed-close breakout against the preceding
    60 completed 1m bars, leaving earlier feature identities unchanged.
    Each observation schema and transition identity remains bound to its recipe.
    """
    if set(maximum_age_ns) != {"mark", "index", "funding_update"}:
        raise DataError("historical feature recipe requires complete freshness limits")
    if not isinstance(profile, str) or profile not in {
        "legacy",
        "causal_mtf_v1",
        "causal_mtf_v2",
        "causal_mtf_v3",
        "causal_mtf_v4",
        "causal_mtf_v5",
        "causal_mtf_v6",
        "causal_mtf_v7",
    }:
        raise DataError("unsupported historical feature profile")
    fields: tuple[FeatureField, ...] = (
        FeatureField("return_1m", "float64", "ratio", "ohlcv_1m", 1, 1, "return"),
        FeatureField("volatility_5m", "float64", "ratio", "ohlcv_1m", 5, 1, "volatility"),
        FeatureField("volume_5m", "float64", "ETH", "ohlcv_1m", 5, 1, "volume_mean"),
        FeatureField("basis", "float64", "ratio", "mark", 1, 1, "basis", maximum_age_ns["mark"]),
        FeatureField(
            "last_settled_funding",
            "float64",
            "ratio",
            "funding_update",
            1,
            1,
            "latest",
            maximum_age_ns["funding_update"],
        ),
    )
    if profile in {
        "causal_mtf_v1",
        "causal_mtf_v2",
        "causal_mtf_v3",
        "causal_mtf_v4",
        "causal_mtf_v5",
        "causal_mtf_v6",
        "causal_mtf_v7",
    }:
        fields += (
            FeatureField("return_5m", "float64", "ratio", "ohlcv_1m", 5, 1, "return"),
            FeatureField("return_15m", "float64", "ratio", "ohlcv_1m", 15, 1, "return"),
            FeatureField("return_60m", "float64", "ratio", "ohlcv_1m", 60, 1, "return"),
            FeatureField("volatility_15m", "float64", "ratio", "ohlcv_1m", 15, 1, "volatility"),
            FeatureField("volatility_60m", "float64", "ratio", "ohlcv_1m", 60, 1, "volatility"),
        )
    if profile in {"causal_mtf_v2", "causal_mtf_v3", "causal_mtf_v4", "causal_mtf_v5", "causal_mtf_v6", "causal_mtf_v7"}:
        fields += (
            FeatureField(
                "trend_efficiency_15m", "float64", "ratio", "ohlcv_1m", 15, 1, "trend_efficiency"
            ),
            FeatureField(
                "trend_efficiency_60m", "float64", "ratio", "ohlcv_1m", 60, 1, "trend_efficiency"
            ),
        )
    if profile in {"causal_mtf_v4", "causal_mtf_v5", "causal_mtf_v6", "causal_mtf_v7"}:
        fields += (
            FeatureField("close_1m", "float64", "quote_per_base", "ohlcv_1m", 1, 1, "close"),
        )
    if profile in {"causal_mtf_v5", "causal_mtf_v6", "causal_mtf_v7"}:
        fields += (
            FeatureField("closed_return_1h", "float64", "ratio", "ohlcv_1m", 1, 60, "return"),
            FeatureField("closed_return_1d", "float64", "ratio", "ohlcv_1m", 1, 1440, "return"),
            FeatureField("volatility_1m", "float64", "ratio", "ohlcv_1m", 60, 1, "volatility"),
            FeatureField("zscore_1m", "float64", "ratio", "ohlcv_1m", 60, 1, "zscore"),
        )
    if profile in {"causal_mtf_v6", "causal_mtf_v7"}:
        fields += (
            FeatureField(
                "taker_buy_imbalance_5m", "float64", "ratio", "ohlcv_1m", 5, 1,
                "taker_buy_imbalance",
            ),
            FeatureField(
                "taker_buy_imbalance_15m", "float64", "ratio", "ohlcv_1m", 15, 1,
                "taker_buy_imbalance",
            ),
            FeatureField(
                "log_trade_count_5m", "float64", "log_count", "ohlcv_1m", 5, 1,
                "log_trade_count",
            ),
        )
    if profile == "causal_mtf_v7":
        fields += (
            FeatureField("breakout_1h", "float64", "ratio", "ohlcv_1m", 60, 1, "breakout"),
        )
    return FeatureSpec("0.4.0", fields)


def _visible_batches_by_decision(
    batches: tuple[tuple[int, Any], ...], decision_times: tuple[int, ...]
) -> tuple[tuple[Any, ...], ...]:
    """Partition ordered market batches once instead of rescanning history per row."""
    result = []
    offset = 0
    for decision_ns in decision_times:
        first = offset
        while offset < len(batches) and batches[offset][0] <= decision_ns:
            offset += 1
        result.append(tuple(batch for _, batch in batches[first:offset]))
    return tuple(result)


def partition_decision_lanes(full_steps: int, lane_count: int) -> tuple[tuple[int, int], ...]:
    """Partition decision offsets into contiguous, rank-exclusive half-open lanes."""
    if (
        type(full_steps) is not int
        or type(lane_count) is not int
        or full_steps < 1
        or not 1 <= lane_count <= full_steps
    ):
        raise DataError("invalid historical decision lane partition")
    return tuple(
        (rank * full_steps // lane_count, (rank + 1) * full_steps // lane_count)
        for rank in range(lane_count)
    )


def _episode_lane(
    *, full_steps: int, episode_rank: int, episode_count: int, episode_steps: int
) -> tuple[int, int]:
    """Return a rank-exclusive decision lane for deterministic episode rotation."""
    if (
        type(full_steps) is not int
        or type(episode_rank) is not int
        or type(episode_count) is not int
        or type(episode_steps) is not int
        or full_steps < 1
        or not 0 <= episode_rank < episode_count
        or episode_steps < 1
    ):
        raise DataError("invalid rotating historical episode partition")
    start, end = partition_decision_lanes(full_steps, episode_count)[episode_rank]
    if end - start < episode_steps:
        raise DataError("historical episode must fit inside every rank-exclusive time lane")
    return start, end


def _episode_start(
    *, lane_start: int, lane_end: int, episode_steps: int, episode_index: int
) -> int:
    """Walk every legal lane start exactly once before the sequence repeats."""
    if type(episode_index) is not int or episode_index < 0:
        raise DataError("historical episode index must be a non-negative integer")
    positions = lane_end - lane_start - episode_steps + 1
    if positions <= 0:
        raise DataError("historical episode has no legal start position in its lane")
    if positions == 1:
        return lane_start
    stride = max(1, int(positions * 0.6180339887498948))
    while gcd(stride, positions) != 1:
        stride += 1
        if stride >= positions:
            stride = 1
    return lane_start + episode_index * stride % positions


class HistoricalEnvironmentFactory:
    """Freeze data, rules and training-only scaler; create a fresh Engine per Env."""

    def __init__(
        self,
        *,
        market_path: str | Path,
        rules_path: str | Path,
        instrument: str,
        start_ns: int,
        end_ns: int,
        coverage_policy: dict,
        execution: dict,
        hard_risk: dict,
        soft_risk: dict,
        starting_balance: str,
        risk_budget_notional: str,
        mode: str = "base",
        scaler: Scaler | None = None,
        fit_scaler: bool = False,
        split: str = "train_inner",
        compact_native_history: bool = True,
        compact_report_snapshots: bool = False,
        share_snapshot_histories: bool = False,
        feature_profile: str = "legacy",
        reward_scale: float = 1.0,
        reward_lambda_dd: float = 0.0,
        reward_lambda_guard: float = 0.0,
        quality_purpose: str = "rl_es",
        prepared_store: CuratedStore | None = None,
    ) -> None:
        # Set only by the application layer after prepare() computes its
        # frozen root-config recipe identity.
        self.application_training_recipe_hash: str | None = None
        if split not in {"train_inner", "validation", "test", "holdout"}:
            raise DataError("unknown historical research split")
        if fit_scaler and split != "train_inner":
            raise DataError("only Train-inner may fit feature scaling")
        if not fit_scaler and scaler is None:
            raise DataError("evaluation must reuse the frozen training feature scaler")
        if type(compact_native_history) is not bool:
            raise DataError("native history compaction flag must be Boolean")
        if type(compact_report_snapshots) is not bool:
            raise DataError("report snapshot compaction flag must be Boolean")
        if type(share_snapshot_histories) is not bool:
            raise DataError("snapshot history sharing flag must be Boolean")
        if share_snapshot_histories and compact_report_snapshots:
            raise DataError("shared full snapshot histories cannot be combined with report compaction")
        if not isinstance(feature_profile, str) or feature_profile not in {
            "legacy",
            "causal_mtf_v1",
            "causal_mtf_v2",
            "causal_mtf_v3",
            "causal_mtf_v4",
            "causal_mtf_v5",
            "causal_mtf_v6",
            "causal_mtf_v7",
        }:
            raise DataError("unsupported historical feature profile")
        self.compact_native_history = compact_native_history
        self.compact_report_snapshots = compact_report_snapshots
        self.share_snapshot_histories = share_snapshot_histories
        self.feature_profile = feature_profile
        self.reward_config = RewardConfig(
            lambda_dd=reward_lambda_dd,
            lambda_guard=reward_lambda_guard,
            reward_scale=reward_scale,
        )
        if prepared_store is not None and not isinstance(prepared_store, CuratedStore):
            raise DataError("prepared history must be a validated CuratedStore")
        self.store = (
            prepared_store
            if prepared_store is not None
            else CuratedStore((*read_curated(market_path).events, *load_rules(rules_path)))
        )
        self.instrument, self.split, self.mode = instrument, split, mode
        self.start_ns, self.end_ns = start_ns, end_ns
        if quality_purpose not in {"directional", "rl_es"}:
            raise DataError("historical environment supports only directional or RL-ES quality use")
        self.quality_purpose = quality_purpose
        self.coverage = CoveragePolicy(**coverage_policy)
        self.execution = ResolvedExecutionConfig.from_dict(execution)
        required_kinds = (
            tuple(sorted(KINDS)) if self.execution.base_fill.data_granularity == "quote" else None
        )
        self.quality = self.store.quality(
            start_ns=start_ns,
            end_ns=end_ns,
            expected_instruments=(instrument,),
            coverage_policy=self.coverage,
            **({"required": required_kinds} if required_kinds is not None else {}),
        )
        self.store.require_usable(self.quality, self.quality_purpose)
        hard_values: dict[str, Any] = {
            k: v if type(v) is bool else Decimal(v) for k, v in hard_risk.items()
        }
        self.hard = HardRiskConfig(**hard_values)
        self.soft = SoftRiskParams.from_search_dict(soft_risk)
        self.capital, self.budget = Decimal(starting_balance), Decimal(risk_budget_notional)
        if (
            not self.capital.is_finite()
            or self.capital <= 0
            or not self.budget.is_finite()
            or self.budget <= 0
        ):
            raise DataError("positive finite capital and risk budget required")
        if self.hard.max_leverage > 1 or self.hard.incomplete_data_max_leverage > 1:
            raise DataError("leveraged liquidation replay is not yet certified by this factory")
        self.rule = self.store.latest_rule(start_ns, instrument)
        if any(
            dict(self.store.latest_rule(t, instrument).payload) != dict(self.rule.payload)
            for t in (end_ns - 1,)
        ):
            raise DataError("split episode at historical instrument-rule changes")
        self.features = historical_feature_spec(
            dict(self.coverage.maximum_age_ns), profile=self.feature_profile
        )
        account = (
            "equity",
            "peak_equity",
            "drawdown",
            "cash",
            "free_margin",
            "initial_margin",
            "maintenance_margin",
            "unrealized_pnl",
            "net_exposure",
            "position_quantity",
        )
        execution_names = (
            "prev_executed_target",
            "prev_raw_target",
            "pending_order_count",
            "pending_signed_exposure",
            "pending_gross_exposure",
            "fill_rate",
        )
        names = tuple("market." + f.name for f in self.features.fields) + tuple(
            "account." + n for n in account
        )
        units = tuple(f.unit for f in self.features.fields) + (
            "USDT",
            "USDT",
            "ratio",
            "USDT",
            "USDT",
            "USDT",
            "USDT",
            "USDT",
            "ratio",
            "ETH",
        )
        time_names = (
            ("time.funding_settlement_within_24h", "time.funding_settlement_time_to_event_24h")
            if self.feature_profile in {"causal_mtf_v3", "causal_mtf_v4"}
            else ()
        )
        if (
            self.feature_profile in {"causal_mtf_v3", "causal_mtf_v4"}
            and self.coverage.known_schedule_times_ns is None
        ):
            raise DataError("causal_mtf_v3 requires a source-declared future funding calendar")
        self.schema = ObservationSchema(
            "0.4.0",
            names
            + time_names
            + ("risk.guard_intervention", "risk.headroom")
            + tuple("execution." + n for n in execution_names),
            units
            + ("ratio",) * len(time_names)
            + ("ratio", "ratio", "ratio", "ratio", "count", "ratio", "ratio", "ratio"),
        )
        # Fold membership uses economic event time. Availability still becomes
        # each row's decision clock, so delayed facts never become observable
        # early and cannot drift into a different score split merely due to
        # ingestion latency.
        bars = [
            e
            for e in self.store.events
            if e.kind == "ohlcv_1m"
            and e.payload["instrument"] == instrument
            and e.available_at_ns is not None
            and start_ns <= e.ts_event_ns < end_ns
        ]
        if len(bars) < 5:
            raise DataError("at least five scoring/drain bars plus feature warmup required")
        self.rows = tuple(
            make_feature_row(
                self.store, self.features, int(e.available_at_ns or 0), instrument=instrument
            )
            for e in bars
        )
        if self.execution.base_fill.data_granularity == "quote":
            quotes = tuple(
                e
                for e in self.store.events
                if e.kind == "best_quote"
                and e.payload["instrument"] == instrument
                and e.available_at_ns is not None
            )
            quote_cursor = 0
            latest_quote: Event | None = None
            for row in self.rows:
                while quote_cursor < len(quotes):
                    candidate = quotes[quote_cursor]
                    available_at_ns = candidate.available_at_ns
                    if available_at_ns is None or available_at_ns > row.decision_ts_ns:
                        break
                    latest_quote = candidate
                    quote_cursor += 1
                if (
                    latest_quote is None
                    or latest_quote.available_at_ns is None
                    or row.decision_ts_ns - latest_quote.available_at_ns > QUOTE_MAX_AGE_NS
                ):
                    raise DataError(
                        "quote execution requires fresh best-quote coverage at every decision"
                    )
        selected_scaler = (
            fit_train(
                self.features,
                Window(start_ns, start_ns, end_ns),
                ((r.decision_ts_ns, r.values) for r in self.rows[:-2]),
            )
            if fit_scaler
            else scaler
        )
        assert selected_scaler is not None
        self.scaler: Scaler = selected_scaler
        if any(
            dict(self.store.latest_rule(row.decision_ts_ns, instrument).payload)
            != dict(self.rule.payload)
            for row in self.rows
        ):
            raise DataError("split episode at historical instrument-rule changes")
        self._bars = tuple(bars)
        config_identity: dict[str, Any] = {
            "execution": execution,
            "quote_max_age_ns": (
                QUOTE_MAX_AGE_NS if self.execution.base_fill.data_granularity == "quote" else None
            ),
            "hard": hard_risk,
            "soft": soft_risk,
            "capital": starting_balance,
            "budget": risk_budget_notional,
        }
        if self.quality_purpose != "rl_es":
            config_identity["quality_purpose"] = self.quality_purpose
        # Preserve historical default run identities; bind every experimental
        # non-default optimizer reward scale to its new artifacts.
        if self.reward_config.reward_scale != 1.0:
            config_identity["reward_scale"] = self.reward_config.reward_scale
        if self.reward_config.lambda_dd != 0.0:
            config_identity["reward_lambda_dd"] = self.reward_config.lambda_dd
        if self.reward_config.lambda_guard != 0.0:
            config_identity["reward_lambda_guard"] = self.reward_config.lambda_guard
        self.base_config_hash = digest({**config_identity, "mode": "base"})
        self.config_hash = digest({**config_identity, "mode": mode})
        transition_identity: dict[str, Any] = {
            "terminal_value_semantics": "finite_settlement_no_bootstrap_v1",
            "data": self.store.snapshot_hash,
            "config": self.config_hash,
            "schema": self.schema.hash,
            "features": self.features.hash,
            "scaler": self.scaler.hash,
            "window": [start_ns, end_ns],
        }
        if self.reward_config.reward_scale != 1.0:
            transition_identity["reward_scale"] = self.reward_config.reward_scale
        if self.reward_config.lambda_dd != 0.0:
            transition_identity["reward_lambda_dd"] = self.reward_config.lambda_dd
        if self.reward_config.lambda_guard != 0.0:
            transition_identity["reward_lambda_guard"] = self.reward_config.lambda_guard
        if self.feature_profile in {"causal_mtf_v3", "causal_mtf_v4"}:
            known_schedule = self.coverage.known_schedule_times_ns
            if known_schedule is None:
                raise DataError("causal_mtf_v3 requires a source-declared future funding calendar")
            transition_identity["feature_profile"] = self.feature_profile
            transition_identity["known_schedule"] = digest(
                {symbol: tuple(times) for symbol, times in known_schedule.items()}
            )
        elif self.feature_profile in {"causal_mtf_v5", "causal_mtf_v6", "causal_mtf_v7"}:
            transition_identity["feature_profile"] = self.feature_profile
        self.transition_hash = digest(transition_identity)
        # Route A needs one frozen execution/feature recipe across temporally
        # disjoint Train windows. The episode transition hash above correctly
        # binds each run to its exact window; this second identity deliberately
        # omits only the window so fit, selection, and student replay can prove
        # recipe parity without pretending they are the same episode.
        self.route_a_execution_recipe_hash = digest(
            {
                "base_config_hash": self.base_config_hash,
                "instrument": self.instrument,
                "rule": dict(self.rule.payload),
                "schema_hash": self.schema.hash,
                "feature_hash": self.features.hash,
                "feature_profile": self.feature_profile,
                "scaler_hash": self.scaler.hash,
                "reward": {
                    "lambda_dd": self.reward_config.lambda_dd,
                    "lambda_guard": self.reward_config.lambda_guard,
                    "reward_scale": self.reward_config.reward_scale,
                },
            }
        )
        self.snapshots: list[Any] = []
        self.teacher_snapshots: list[Any] = []
        self.fill_references: dict[str, Decimal] = {}
        self._market_batch_index: tuple[tuple[int, VisibleBatch], ...] | None = None
        self._market_batch_times: tuple[int, ...] = ()
        self._market_batch_groups_by_row: tuple[tuple[VisibleBatch, ...], ...] = ()
        self._barrier_frames_by_row: tuple[BarrierFrame, ...] | None = None

    def _visible_market_batch_index(self) -> tuple[tuple[int, VisibleBatch], ...]:
        """Build one immutable availability-time index for all Episodes in this factory."""
        if self._market_batch_index is not None:
            return self._market_batch_index
        warmup_minutes = (
            max(
                (
                    field.lookback_bars * field.timeframe_minutes
                    for field in self.features.fields
                    if field.source_kind == "ohlcv_1m"
                ),
                default=1,
            )
            + 2
        )
        first_decision_ns = self.rows[0].decision_ts_ns
        last_decision_ns = self.rows[-1].decision_ts_ns
        feed_start_ns = max(0, first_decision_ns - warmup_minutes * MINUTE_NS)
        groups: dict[int, list] = defaultdict(list)
        for event in self.store.available_event_window(feed_start_ns, last_decision_ns):
            if event.kind in {"instrument_rule", "funding_update"}:
                continue
            if event.payload.get("instrument") != self.instrument:
                continue
            available_at_ns = event.available_at_ns
            if available_at_ns is None:
                continue
            groups[available_at_ns].append(
                to_market_event(event, quality_grade=DataQualityGrade(self.quality.grade.value))
            )
        batches = tuple(
            (timestamp, VisibleBatch(timestamp, tuple(events), True))
            for timestamp, events in sorted(groups.items())
        )
        self._market_batch_times = tuple(timestamp for timestamp, _batch in batches)
        self._market_batch_groups_by_row = _visible_batches_by_decision(
            batches,
            tuple(row.decision_ts_ns for row in self.rows),
        )
        self._market_batch_index = batches
        return batches

    def _episode_market_batch_groups(
        self, *, start_offset: int, episode_rows: tuple[Any, ...], feed_start_ns: int
    ) -> tuple[tuple[VisibleBatch, ...], ...]:
        """Reuse row deltas while restoring the episode's required warmup prefix."""
        batches = self._visible_market_batch_index()
        first_batch = bisect_left(self._market_batch_times, feed_start_ns)
        first_decision_end = bisect_right(
            self._market_batch_times,
            episode_rows[0].decision_ts_ns,
        )
        initial = tuple(batch for _timestamp, batch in batches[first_batch:first_decision_end])
        following = self._market_batch_groups_by_row[
            start_offset + 1 : start_offset + len(episode_rows)
        ]
        if len(following) != len(episode_rows) - 1:
            raise DataError("cached visible-batch rows do not cover the Episode window")
        return (initial, *following)

    def _cached_barrier_frames(self) -> tuple[BarrierFrame, ...]:
        """Build immutable observation/event frames once for this frozen factory."""
        if self._barrier_frames_by_row is not None:
            return self._barrier_frames_by_row
        self._visible_market_batch_index()
        frames = []
        for row, batches in zip(self.rows, self._market_batch_groups_by_row, strict=True):
            market_features = visible_market_features(
                row,
                self.features,
                self.schema,
                instrument=self.instrument,
                decision_ts_ns=row.decision_ts_ns,
            )
            if self.feature_profile in {"causal_mtf_v3", "causal_mtf_v4"}:
                known_schedule = self.coverage.known_schedule_times_ns
                if known_schedule is None:
                    raise DataError(
                        "causal_mtf_v3 requires a source-declared future funding calendar"
                    )
                time_features = scheduled_funding_time_features(
                    row.decision_ts_ns,
                    known_schedule[self.instrument],
                )
            else:
                time_features = {}
            frames.append(BarrierFrame(row.decision_ts_ns, batches, market_features, time_features))
        self._barrier_frames_by_row = tuple(frames)
        return self._barrier_frames_by_row

    def _episode_barrier_frames(
        self, *, start_offset: int, episode_rows: tuple[Any, ...], feed_start_ns: int
    ) -> tuple[BarrierFrame, ...]:
        """Reuse cached frames, rebuilding only a shifted Episode's warmup frame."""
        cached_frames = self._cached_barrier_frames()
        stop = start_offset + len(episode_rows)
        selected = cached_frames[start_offset:stop]
        if len(selected) != len(episode_rows):
            raise DataError("cached barrier frames do not cover the Episode window")
        if start_offset == 0:
            return selected
        first_batches = self._episode_market_batch_groups(
            start_offset=start_offset,
            episode_rows=episode_rows[:1],
            feed_start_ns=feed_start_ns,
        )[0]
        first = BarrierFrame(
            selected[0].timestamp_ns, first_batches, selected[0].market, selected[0].time_features
        )
        return (first, *selected[1:])

    def observation_state(self, snapshot) -> dict:
        cash, free = native_balances(snapshot)
        if snapshot.mark is None:
            raise DataError("native mark required for exposure observations")
        notional = signed_quantity(snapshot) * snapshot.mark
        initial = sum((_money(m["initial"]) for m in snapshot.margins), Decimal(0))
        maintenance = sum((_money(m["maintenance"]) for m in snapshot.margins), Decimal(0))
        submitted = Decimal(str(getattr(snapshot, "cumulative_submitted_qty", 0)))
        filled = Decimal(str(getattr(snapshot, "cumulative_filled_qty", 0)))
        if not hasattr(snapshot, "cumulative_submitted_qty"):
            submitted = sum((Decimal(o["quantity"]) for o in snapshot.all_orders), Decimal(0))
            filled = sum((Decimal(o["filled_qty"]) for o in snapshot.all_orders), Decimal(0))
        pending_signed = Decimal(0)
        pending_gross = Decimal(0)
        for order in snapshot.orders:
            side = order.get("side")
            leaves = Decimal(str(order.get("leaves_qty", "0")))
            if side not in {"BUY", "SELL"} or not leaves.is_finite() or leaves < 0:
                raise DataError("native pending order side/quantity invalid for observation")
            signed = leaves if side == "BUY" else -leaves
            pending_signed += signed * snapshot.mark / self.budget
            pending_gross += leaves * snapshot.mark / self.budget
        cap = min(self.hard.max_notional, snapshot.equity * self.hard.max_leverage)
        return {
            "account": {
                "cash": float(cash),
                "free_margin": float(free),
                "initial_margin": float(initial),
                "maintenance_margin": float(maintenance),
                "unrealized_pnl": float(snapshot.equity - cash),
                "net_exposure": float(notional / self.budget),
                "position_quantity": float(signed_quantity(snapshot)),
            },
            "risk": {"headroom": float(max(Decimal(0), cap - abs(notional)) / self.budget)},
            "execution": {
                "pending_order_count": float(len(snapshot.orders)),
                "pending_signed_exposure": float(pending_signed),
                "pending_gross_exposure": float(pending_gross),
                "fill_rate": float(filled / submitted) if submitted else 0.0,
            },
        }

    def teacher_state(self, snapshot) -> dict[str, float]:
        """Return native position facts for rule teachers, outside L4 input schema.

        Duration and average open price come from the native Position record,
        not from elapsed bars or inferred fills. This projection deliberately
        does not synthesize MFE/MAE; those require a separately audited
        decision-time mark accumulator.
        """
        cash, _free = native_balances(snapshot)
        if snapshot.mark is None:
            raise DataError("native mark required for teacher risk context")
        notional = signed_quantity(snapshot) * snapshot.mark
        unrealized = snapshot.equity - cash
        if len(snapshot.positions) > 1:
            raise DataError("teacher native position projection expects one netted position")
        if snapshot.positions:
            position = snapshot.positions[0]
            opened_ns = position.get("ts_opened")
            decision_ts_ns = getattr(snapshot, "decision_ts_ns", None)
            avg_px_open = position.get("avg_px_open")
            if (type(opened_ns) is not int or opened_ns <= 0
                    or type(decision_ts_ns) is not int
                    or decision_ts_ns < opened_ns
                    or avg_px_open is None):
                raise DataError("native position open time/decision clock/fill price unavailable")
            try:
                average_open_price = Decimal(str(avg_px_open))
            except Exception as exc:
                raise DataError("native position average open price invalid") from exc
            if not average_open_price.is_finite() or average_open_price <= 0:
                raise DataError("native position average open price invalid")
            # Position.duration_ns is not a live elapsed clock for every open
            # Nautilus Position implementation. Derive age from its native
            # ts_opened and the current native decision barrier instead.
            position_duration_ns = float(decision_ts_ns - opened_ns)
            position_avg_px_open = float(average_open_price)
        else:
            position_duration_ns = 0.0
            position_avg_px_open = 0.0
        return {
            "unrealized_pnl_pct": float(unrealized / abs(notional)) if notional else 0.0,
            "position_duration_ns": position_duration_ns,
            "position_avg_px_open": position_avg_px_open,
        }

    def make_env(
        self,
        transition_hash: str | None = None,
        *,
        replay: EpisodeReplay | None = None,
        soft_multipliers: dict | None = None,
        episode_rank: int = 0,
        episode_count: int = 1,
        episode_steps: int | None = None,
    ) -> NautilusBarrierEnv:
        transition = transition_hash or self.transition_hash
        pending_replay = replay
        full_steps = len(self.rows) - 3
        selected_steps = full_steps if episode_steps is None else episode_steps
        if (
            type(episode_rank) is not int
            or type(episode_count) is not int
            or type(selected_steps) is not int
            or not 0 <= episode_rank < episode_count
            or selected_steps < 1
            or selected_steps > full_steps
        ):
            raise DataError("invalid rotating historical episode partition")
        lane_start, lane_end = _episode_lane(
            full_steps=full_steps,
            episode_rank=episode_rank,
            episode_count=episode_count,
            episode_steps=selected_steps,
        )

        def episode_factory(*, seed, options):
            nonlocal pending_replay
            if not isinstance(options, dict):
                raise DataError("historical episode options must be a dictionary")
            requested_index = options.get("rles_episode_index", 0)
            current_replay = pending_replay
            episode_index = (
                current_replay.episode_index
                if current_replay is not None and current_replay.episode_index is not None
                else requested_index
            )
            if type(episode_index) is not int or episode_index < 0:
                raise DataError("historical episode index must be a non-negative integer")
            start_offset = _episode_start(
                lane_start=lane_start,
                lane_end=lane_end,
                episode_steps=selected_steps,
                episode_index=episode_index,
            )
            if (
                current_replay is not None
                and current_replay.episode_start_offset is not None
                and current_replay.episode_start_offset != start_offset
            ):
                raise DataError("checkpoint episode window differs from deterministic schedule")
            episode_rows = self.rows[start_offset : start_offset + selected_steps + 3]
            episode_fold_hash = (
                digest([self.start_ns, self.end_ns])
                if episode_steps is None and episode_count == 1
                else digest(
                    {
                        "window": [episode_rows[0].decision_ts_ns, episode_rows[-1].decision_ts_ns],
                        "partition": [episode_rank, episode_count, selected_steps],
                        "episode_index": episode_index,
                        "start_offset": start_offset,
                    }
                )
            )
            self.snapshots = []
            self.teacher_snapshots = []
            self.fill_references = {}
            seen_fills: dict[str, dict] = {}
            seen_funding: dict[str, dict] = {}
            seen_liquidations: dict[str, dict] = {}
            seen_orders: dict[str, dict] = {}
            order_versions: dict[str, tuple[list[int], list[dict]]] = {}
            fill_history = _AppendOnlyNativeHistory()
            funding_history = _AppendOnlyNativeHistory()
            snapshot_revision = 0
            order_filled: dict[str, Decimal] = {}
            cumulative_submitted = Decimal(0)
            cumulative_filled = Decimal(0)
            last_fill_ts_ns: int | None = None

            def unique_delta(records, seen, identity_field="event_id"):
                delta = []
                for record in records:
                    identity = str(record.get(identity_field, ""))
                    if not identity:
                        raise DataError("native history fact has no stable identity")
                    previous = seen.get(identity)
                    if previous is not None:
                        if previous != record:
                            raise DataError("native history identity has conflicting facts")
                        continue
                    copied = dict(record)
                    seen[identity] = copied
                    delta.append(copied)
                return tuple(delta)

            def record_snapshot(snapshot):
                nonlocal cumulative_submitted, cumulative_filled, last_fill_ts_ns, snapshot_revision
                for order in snapshot.all_orders:
                    identity = str(order.get("client_order_id", ""))
                    if not identity:
                        raise DataError("native order history has no client identity")
                    quantity = Decimal(str(order["quantity"]))
                    filled = Decimal(str(order["filled_qty"]))
                    previous = order_filled.get(identity)
                    if previous is None:
                        cumulative_submitted += quantity
                        previous = Decimal(0)
                    if filled < previous or filled > quantity:
                        raise DataError("native order fill total regressed or exceeded order size")
                    cumulative_filled += filled - previous
                    order_filled[identity] = filled
                    prior_order = seen_orders.get(identity)
                    if prior_order is not None and any(
                        prior_order.get(field) != order.get(field)
                        for field in ("instrument_id", "side", "quantity")
                    ):
                        raise DataError("native order identity has conflicting immutable facts")
                    if prior_order != order:
                        # Native Order.to_dict() is repeated at every market
                        # barrier. Reuse an unchanged detached row rather than
                        # retaining thousands of identical full dictionaries.
                        # Stored rows are never mutated by this factory; a
                        # changed row gets a fresh copy, keeping each historical
                        # M2.2 status exact.
                        detached_order = dict(order)
                        seen_orders[identity] = detached_order
                        if self.share_snapshot_histories:
                            revisions, rows = order_versions.setdefault(identity, ([], []))
                            revisions.append(snapshot_revision)
                            rows.append(detached_order)
                fills = unique_delta(snapshot.fills, seen_fills)
                fill_start = len(fill_history.rows)
                for fill in fills:
                    fill_ts_ns = fill.get("ts_event")
                    if (type(fill_ts_ns) is not int or fill_ts_ns <= 0
                            or fill_ts_ns > snapshot.decision_ts_ns):
                        raise DataError("native fill timestamp is missing or after its decision barrier")
                    last_fill_ts_ns = max(last_fill_ts_ns or 0, fill_ts_ns)
                    if self.share_snapshot_histories:
                        fill_history.append(fill, timestamp_ns=fill_ts_ns)
                funding = unique_delta(snapshot.funding, seen_funding)
                funding_start = len(funding_history.rows)
                if self.share_snapshot_histories:
                    for adjustment in funding:
                        funding_history.append(adjustment)
                liquidations = (
                    None
                    if snapshot.liquidation_orders is None
                    else unique_delta(
                        snapshot.liquidation_orders, seen_liquidations, "client_order_id"
                    )
                )
                final_boundary = snapshot.decision_ts_ns >= episode_rows[-2].decision_ts_ns
                if self.compact_report_snapshots:
                    # Retain only the native contract scalars needed to verify
                    # open linear equity. Full Position.to_dict histories can
                    # grow with every fill and are not required by the report.
                    position_fields = (
                        "position_id",
                        "side",
                        "quantity",
                        "multiplier",
                        "is_inverse",
                        "settlement_currency",
                    )
                    report_positions = tuple(
                        {key: row.get(key) for key in position_fields} for row in snapshot.positions
                    )
                else:
                    report_positions = snapshot.positions
                compact = SimpleNamespace(
                    equity=snapshot.equity,
                    currency=snapshot.currency,
                    positions=report_positions,
                    orders=(
                        snapshot.orders
                        if not self.compact_report_snapshots or final_boundary
                        else ()
                    ),
                    account=(snapshot.account if not self.compact_report_snapshots else {}),
                    account_events=(
                        snapshot.account_events[-1:] if not self.compact_report_snapshots else ()
                    ),
                    fills=(
                        fill_history.view()
                        if self.share_snapshot_histories
                        else tuple(seen_fills.values())
                        if not self.compact_report_snapshots or final_boundary
                        else fills
                    ),
                    new_fills=(
                        fill_history.view(start=fill_start)
                        if self.share_snapshot_histories else fills
                    ),
                    funding=(
                        funding_history.view()
                        if self.share_snapshot_histories
                        else tuple(seen_funding.values())
                        if not self.compact_report_snapshots or final_boundary
                        else funding
                    ),
                    new_funding=(
                        funding_history.view(start=funding_start)
                        if self.share_snapshot_histories else funding
                    ),
                    # The native facts reader may purge closed orders after it
                    # snapshots them. Keep the latest native row for each ID
                    # in every full M2.2 barrier so same-clock callbacks do
                    # not make completed orders disappear from the final view.
                    all_orders=(
                        _VersionedNativeOrdersView(order_versions, snapshot_revision)
                        if self.share_snapshot_histories
                        else tuple(seen_orders.values())
                        if not self.compact_report_snapshots or final_boundary
                        else ()
                    ),
                    liquidation_orders=(
                        None
                        if snapshot.liquidation_orders is None
                        else tuple(seen_liquidations.values())
                        if not self.compact_report_snapshots or final_boundary
                        else liquidations
                    ),
                    margins=(snapshot.margins if not self.compact_report_snapshots else ()),
                    mark=snapshot.mark,
                    liquidation_capability=snapshot.liquidation_capability,
                    decision_ts_ns=snapshot.decision_ts_ns,
                    cumulative_submitted_qty=cumulative_submitted,
                    cumulative_filled_qty=cumulative_filled,
                )
                self.snapshots.append(compact)
                snapshot_revision += 1
                teacher_snapshot = SimpleNamespace(
                    decision_ts_ns=snapshot.decision_ts_ns,
                    positions=tuple({
                        key: row.get(key)
                        for key in ("position_id", "ts_opened", "side", "avg_px_open")
                    } for row in snapshot.positions),
                    mark=snapshot.mark,
                )
                if teacher_snapshot.positions:
                    self.teacher_snapshots.append(teacher_snapshot)
                else:
                    # Retain only the current flat boundary before a new
                    # position; recovery needs history for the live lifecycle,
                    # not every earlier completed position in the episode.
                    self.teacher_snapshots[:] = [teacher_snapshot]

            def record_reference(client_id, reference):
                previous = self.fill_references.get(client_id)
                if previous is not None and previous != reference:
                    raise DataError(
                        "multiple native fill references per order require event-level allocation"
                    )
                self.fill_references[client_id] = reference

            session = make_perpetual_session(
                rule=self.rule.payload,
                execution=self.execution,
                mode=self.mode,
                starting_balance=self.capital,
                record_snapshot=record_snapshot,
                record_fill_reference=record_reference,
                compact_native_history=self.compact_native_history,
            )
            planner = TargetQuantityPlanner(self.execution)
            rules = InstrumentRules(
                self.instrument,
                Decimal(self.rule.payload["step_size"]),
                Decimal(self.rule.payload["min_qty"]),
                Decimal(self.rule.payload["min_notional"]),
                Decimal(1),
                self.rule.payload.get("reduce_only_min_notional_exempt", False),
            )
            soft = SoftRiskTransform(
                self.soft if not soft_multipliers else self.soft.with_multipliers(soft_multipliers)
            )
            hard = HardRiskGuard(self.hard)

            def plan_with_rounding(target, snapshot, *, terminal=False, force_reduce=False):
                decision = planner.plan(
                    decision_id=f"{'terminal' if terminal else 'policy'}-{snapshot.decision_ts_ns}",
                    instrument=rules,
                    target_exposure=target,
                    risk_budget_notional=self.budget,
                    equity=snapshot.equity,
                    mark_price=snapshot.mark,
                    current_qty=signed_quantity(snapshot),
                    pending=tuple(
                        PendingOrder(
                            o["client_order_id"],
                            Decimal(o["leaves_qty"]) * (-1 if o["side"] == "SELL" else 1),
                            o["status"],
                        )
                        for o in snapshot.orders
                    ),
                    terminal=terminal,
                    force_reduce=force_reduce,
                )
                contract_plan = to_contract_plan(
                    decision,
                    ExecutionPlan,
                    latency_model_id=self.execution.latency(self.mode).model_id,
                    quantity_unit="ETH",
                )
                rounded_exposure = (
                    decision.target_qty * snapshot.mark * rules.contract_multiplier / self.budget
                )
                return contract_plan, rounded_exposure

            def plan(target, snapshot, *, terminal=False, force_reduce=False):
                return plan_with_rounding(
                    target, snapshot, terminal=terminal, force_reduce=force_reduce
                )[0]

            def plan_action(action, snapshot, drawdown):
                current = signed_quantity(snapshot) * snapshot.mark / self.budget
                known = self.store.visible(snapshot.decision_ts_ns, "funding_update")
                rate = Decimal(str(known[-1].payload["rate"])) if known else None
                softened = soft.apply(
                    action, current_exposure=current, drawdown=drawdown, funding_abs=rate
                )
                adjusted = hard.apply(
                    softened.adjusted,
                    budget_notional=self.budget,
                    equity=snapshot.equity,
                    free_margin=native_balances(snapshot)[1],
                    current_notional=current * self.budget,
                    mark_available=True,
                    funding_available=rate is not None,
                    liquidation_capability=True,
                    data_quality_grade=self.quality.grade.value,
                )
                contract_plan, rounded_exposure = plan_with_rounding(
                    adjusted.adjusted,
                    snapshot,
                    force_reduce=adjusted.force_reduce,
                )
                return RiskActionReview(
                    contract_plan,
                    abs(action - adjusted.adjusted),
                    softened.intervened_fraction,
                    adjusted.intervened_fraction,
                    True,
                    softened.adjusted,
                    adjusted.adjusted,
                    rounded_exposure,
                )

            warmup_minutes = (
                max(
                    (
                        field.lookback_bars * field.timeframe_minutes
                        for field in self.features.fields
                        if field.source_kind == "ohlcv_1m"
                    ),
                    default=1,
                )
                + 2
            )
            feed_start_ns = max(0, episode_rows[0].decision_ts_ns - warmup_minutes * MINUTE_NS)
            frames = self._episode_barrier_frames(
                start_offset=start_offset,
                episode_rows=episode_rows,
                feed_start_ns=feed_start_ns,
            )
            mark_excursions = NativeMarkExcursionTracker()
            teacher_state_restored = current_replay is None

            def episode_teacher_state(snapshot) -> dict[str, float]:
                nonlocal teacher_state_restored, mark_excursions
                native_state = self.teacher_state(snapshot)
                if (last_fill_ts_ns is not None
                        and last_fill_ts_ns > snapshot.decision_ts_ns):
                    raise DataError("native last-fill clock is after the decision barrier")
                native_state["native_last_fill_age_seconds"] = (
                    -1.0 if last_fill_ts_ns is None
                    else (snapshot.decision_ts_ns - last_fill_ts_ns) / 1_000_000_000
                )
                if not teacher_state_restored:
                    if (not self.teacher_snapshots
                            or self.teacher_snapshots[-1].decision_ts_ns
                            != snapshot.decision_ts_ns):
                        raise DataError(
                            "native teacher recovery history does not reach the current barrier"
                        )
                    if snapshot.positions:
                        position = snapshot.positions[0]
                        mark_excursions = NativeMarkExcursionTracker.reconstruct_open_position(
                            self.teacher_snapshots,
                            position_id=position["position_id"],
                            opened_ns=position["ts_opened"],
                            side=position["side"],
                        )
                        mark_state = mark_excursions.current()
                    else:
                        mark_state = mark_excursions.update(snapshot)
                    teacher_state_restored = True
                    return {**native_state, **mark_state}
                return {**native_state, **mark_excursions.update(snapshot)}

            episode = BarrierEpisode(
                native_session=session,
                native_identity={
                    "dataset_hash": self.store.snapshot_hash,
                    "fold_hash": episode_fold_hash,
                    "config_hash": self.config_hash,
                },
                transition_hash=transition,
                frames=frames,
                scoring_frames=len(frames) - 2,
                plan_action=plan_action,
                plan_terminal=lambda s: plan(Decimal(0), s, terminal=True),
                executed_exposure=lambda s: signed_quantity(s) * s.mark / self.budget,
                reward_config=self.reward_config,
                replay=current_replay,
                episode_index=episode_index,
                episode_start_offset=start_offset,
                observation_state=self.observation_state,
                teacher_state=episode_teacher_state,
            )
            pending_replay = None
            return episode

        bridge = L2FrozenScalerAdapter(
            self.schema, self.features, self.scaler, expected_scaler_hash=self.scaler.hash
        )
        return NautilusBarrierEnv(
            episode_factory=episode_factory,
            observation_builder=ObservationBuilder(
                self.schema,
                scaler=bridge,
                expected_scaler_hash=bridge.fingerprint,
                split="train" if self.split == "train_inner" else self.split,
            ),
            reward_config=self.reward_config,
            max_steps=selected_steps,
            transition_hash=transition,
        )

    def metadata(self) -> dict:
        return {
            "quality": asdict(self.quality),
            "features": asdict(self.features),
            "feature_profile": self.feature_profile,
            "reward_config": asdict(self.reward_config),
            "known_schedule_times_ns": (
                None
                if self.coverage.known_schedule_times_ns is None
                else {
                    symbol: tuple(times)
                    for symbol, times in self.coverage.known_schedule_times_ns.items()
                }
            ),
            "schema": asdict(self.schema),
            "scaler": asdict(self.scaler),
            "transition_hash": self.transition_hash,
            "identities": {
                "dataset": self.store.snapshot_hash,
                "features": self.features.hash,
                "schema": self.schema.hash,
                "scaler": self.scaler.hash,
                "execution": self.execution.fingerprint(),
                "hard_risk": self.hard.signature(),
                "config": self.config_hash,
            },
        }


def make_historical_es_env(
    *, env_config: dict, candidate_context: dict, soft_risk_multipliers: dict
) -> NautilusBarrierEnv:
    """Installed production factory for candidate-isolated L6 spawn workers."""
    if candidate_context.get("split") != "train_inner":
        raise DataError("ES may evaluate only the frozen Train-inner window")
    raw = dict(env_config["scaler"])
    raw["mean"], raw["std"] = tuple(raw["mean"]), tuple(raw["std"])
    prepared_store = env_config.get("prepared_store")
    if prepared_store is not None and not isinstance(prepared_store, CuratedStore):
        raise DataError("ES worker prepared history must be a validated CuratedStore")
    factory = HistoricalEnvironmentFactory(
        **{
            **env_config["recipe"],
            "fit_scaler": False,
            "scaler": Scaler(**raw),
            "prepared_store": prepared_store,
        }
    )
    return factory.make_env(env_config["transition_hash"], soft_multipliers=soft_risk_multipliers)
