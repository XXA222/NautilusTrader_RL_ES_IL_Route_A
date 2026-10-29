"""Registered, bounded Route A 1m teacher source validation and native runner.

The source preflight is intentionally separate from decoding: it verifies the
frozen protocol and protected-month guard before the existing project reader is
allowed to reserve/read Train. Native execution is performed only by the
existing HistoricalEnvironmentFactory/Nautilus provider.
"""
# mypy: disable-error-code=import-untyped
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
from pathlib import Path
import re
import sys
from time import monotonic
from typing import Any, Literal, cast

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "qa"))

import run_route_a_train_student_screen_20261020 as harness  # noqa: E402
from rles_orchestrator import application  # noqa: E402
from rles_orchestrator.research_access import (  # noqa: E402
    ACTIVE_OWNER,
    market_key,
    reject_protected_monthly_archive_source,
)
from rles_data.features import causal_1m_indicator_inputs  # noqa: E402
from rles_gym_bridge.observation import VisibleFeature  # noqa: E402
from rles_research.native_metrics import native_report  # noqa: E402
from rles_route_a.evaluation.teacher_case_report_1m import (  # noqa: E402
    NativeTradeCase1m,
    write_teacher_research_artifacts_1m,
)
from rles_route_a.evaluation.teacher_funnel_1m import (  # noqa: E402
    TeacherDecisionRecord,
    TeacherExecutionRecord,
    TeacherPlannerRecord,
    TeacherPreDecision,
)
from rles_route_a.indicators_1m import CausalBar, IndicatorSnapshot, StreamingIndicators1m  # noqa: E402
from rles_route_a.integration.l2_adapter import (  # noqa: E402
    CAUSAL_1M_INDICATORS_V1_ROUTE_A_FEATURES,
)
from rles_route_a.teachers_1m.registry import (  # noqa: E402
    ROC_COST_HYSTERESIS_FADE_TEACHER_ID,
    ROC_COST_HYSTERESIS_TEACHER_ID,
    build_teacher_1m,
)

TEACHERS = ("T01_EMA_9_21", "T02_ROC_20_60", ROC_COST_HYSTERESIS_TEACHER_ID,
            ROC_COST_HYSTERESIS_FADE_TEACHER_ID, "T03_DONCHIAN_20",
            "T04_ZSCORE_60")
ROOT_DEFAULT_PREREG = ROOT / "configs/route_a_1m_teacher_native_prereg_20261026.json"
DEFAULT_OUTPUT = ROOT / ".artifacts/route-a-1m-teacher-native-s06"
SCENARIO_RULES = ROOT / ".artifacts/audit-development-20261016/data/scenario-rules.json"
EXECUTION_TEMPLATE = ROOT / "configs/ethusdt_20240101_real.json"
FEATURE_RECIPE = ROOT / "configs/route_a_1m_teacher_library_v1.json"
MINUTE_NS = 60_000_000_000
WINDOW_IDS = ("smoke", "inner-screen-1", "inner-screen-2", "inner-screen-3")
NativeExitVariant = Literal["X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"]
EmaReversalMode = Literal["EXIT_ONLY_V1", "REVERSE_AFTER_FLAT_V1"]


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _ns(value: str) -> int:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("pre-registered timestamps must include a timezone")
    delta = parsed.astimezone(timezone.utc) - datetime(1970, 1, 1, tzinfo=timezone.utc)
    return (delta.days * 86_400 + delta.seconds) * 1_000_000_000 + delta.microseconds * 1_000


def verify_preregistration(
    path: Path, *, feature_recipe_override: Path | None = None
) -> tuple[dict[str, Any], str]:
    """Verify immutable protocol/source receipts without reading market rows."""
    digest_file = path.with_suffix(path.suffix + ".sha256")
    if not path.is_file() or not digest_file.is_file():
        raise RuntimeError("frozen pre-registration or SHA-256 sidecar is missing")
    content = path.read_bytes()
    actual_sha = sha256(content).hexdigest()
    expected_sha = digest_file.read_text(encoding="ascii").strip()
    if actual_sha != expected_sha:
        raise RuntimeError("frozen pre-registration SHA-256 mismatch")
    doc = json.loads(content)
    if (doc.get("schema") != "rles.route-a.1m-teacher-native-prereg.v1"
            or doc.get("status") != "frozen_before_source_read"
            or doc.get("qualification") != "D1_DEVELOPMENT_DIAGNOSTIC_ONLY_NOT_G1_G2"):
        raise RuntimeError("pre-registration identity/status is not the S06 diagnostic protocol")
    source = doc["source"]
    market = ROOT / source["market_path"]
    marker = ROOT / source["marker_path"]
    import_manifest = ROOT / source["import_manifest_path"]
    for candidate in (market, marker, import_manifest):
        if not candidate.is_file():
            raise RuntimeError(f"pre-registered source file is missing: {candidate}")
    # These small metadata reads are allowed before the registered market-row read.
    marker_doc = json.loads(marker.read_text(encoding="utf-8"))
    imported = json.loads(import_manifest.read_text(encoding="utf-8"))
    if _sha256(marker) != source["marker_sha256"]:
        raise RuntimeError("pre-registered COMPLETED marker checksum changed")
    if _sha256(import_manifest) != source["import_manifest_sha256"]:
        raise RuntimeError("pre-registered source import manifest checksum changed")
    if (marker_doc.get("schema_version") != source["schema_version"]
            or marker_doc.get("snapshot_hash") != source["snapshot_hash"]
            or marker_doc.get("events") != source["events"]
            or marker_doc.get("file_sha256") != source["market_file_sha256_from_marker"]):
        raise RuntimeError("pre-registered market snapshot differs from the completion marker")
    if (imported.get("start_ns") != doc["roles"]["source_read_start_ns"]
            or imported.get("end_ns") != doc["roles"]["source_read_end_ns_exclusive"]
            or imported.get("event_count") != source["events"]):
        raise RuntimeError("pre-registered physical read bounds/event count differ from import metadata")
    monthly_sources = {
        match.group(1)
        for row in imported.get("sources", [])
        if isinstance(row, dict) and isinstance(row.get("url"), str)
        and "/monthly/" in row["url"]
        for match in [re.search(r"-(\d{4}-\d{2})\.zip$", row["url"])]
        if match is not None
    }
    if sorted(monthly_sources) != sorted(source["physical_source_months"]):
        raise RuntimeError("pre-registered monthly physical source months changed")
    for section, field, expected in (
        ("rules", "path", doc["rules"]["sha256"]),
        ("native_execution_template", "path", doc["native_execution_template"]["sha256"]),
        ("feature_recipe", "path", doc["feature_recipe"]["sha256"]),
    ):
        candidate = ROOT / doc[section][field]
        if section == "feature_recipe" and feature_recipe_override is not None:
            candidate = feature_recipe_override
        if _sha256(candidate) != expected:
            raise RuntimeError(f"pre-registered {section} checksum changed")
    return doc, actual_sha


def _window_bounds(doc: dict[str, Any], window_id: str) -> tuple[int, int, str]:
    """Resolve only the windows frozen in the pre-registration."""
    if window_id not in WINDOW_IDS:
        raise ValueError(f"window is not a registered S07 window: {window_id}")
    if window_id == "smoke":
        start_text, end_text = doc["roles"]["smoke_window"]
        block_id = "smoke-2024-01-02_2024-01-03"
    else:
        index = int(window_id.rsplit("-", 1)[1]) - 1
        blocks = doc["roles"].get("inner_screen_blocks")
        if not isinstance(blocks, list) or index >= len(blocks):
            raise ValueError("pre-registration omitted the requested inner-screen block")
        start_text, end_text = blocks[index]
        start_day = start_text[:10].replace("-", "")
        end_day = end_text[:10].replace("-", "")
        block_id = f"inner-screen-{start_day}_{end_day}"
    start_ns, end_ns = _ns(start_text), _ns(end_text)
    if (start_ns >= end_ns or start_ns % MINUTE_NS or end_ns % MINUTE_NS
            or start_ns < doc["roles"]["source_read_start_ns"]
            or end_ns > doc["roles"]["source_read_end_ns_exclusive"]):
        raise ValueError("pre-registered window is outside its registered Train read scope")
    return start_ns, end_ns, block_id


def _source_audit(doc: dict[str, Any], prereg_sha: str, output: Path,
                  *, window_id: str = "smoke") -> dict[str, Any]:
    source = doc["source"]
    market = ROOT / source["market_path"]
    marker = ROOT / source["marker_path"]
    imported = json.loads((ROOT / source["import_manifest_path"]).read_text(encoding="utf-8"))
    scope_start, scope_end = imported["start_ns"], imported["end_ns"]
    start_ns, end_ns, block_id = _window_bounds(doc, window_id)

    # Same repository guard used by the registered reader; it opens no market rows.
    reject_protected_monthly_archive_source(market, market_key(doc["market"]["instrument"]))

    harness.MARKET = market
    harness.MARKER = marker
    harness.IMPORT = ROOT / source["import_manifest_path"]
    harness.RULES = ROOT / doc["rules"]["path"]
    harness.CONFIG = ROOT / doc["native_execution_template"]["path"]
    harness.OUT_BASE = output
    harness.FEATURE_PROFILE = "causal_mtf_v5"
    harness.FOLD_ID = doc["protocol_id"]
    config, _factory = harness.make_config(
        label=f"train-route-a-1m-source-audit-{window_id}",
        marker_sha=source["marker_sha256"],
        scope_start=scope_start,
        scope_end=scope_end,
        start_ns=start_ns,
        end_ns=end_ns,
        mode="base",
        fit_scaler=True,
        prepare_factory=False,
        source_mode=source["read_mode"],
    )
    config["output"] = str(output / "registry-reservation")
    config["train_data_access"]["label"] = "train-route-a-1m-s06-source-audit"
    owner = ACTIVE_OWNER.set(doc["protocol_id"])
    try:
        report = application.audit_market(config)
    finally:
        ACTIVE_OWNER.reset(owner)
    train = report["splits"]["train"]
    return {
        "schema": "rles.route-a.1m-teacher-native-source-audit.v1",
        "protocol_id": doc["protocol_id"],
        "preregistration_sha256": prereg_sha,
        "source": {
            "snapshot_hash": source["snapshot_hash"],
            "marker_sha256": source["marker_sha256"],
            "events": source["events"],
            "registered_read_scope_ns": [scope_start, scope_end],
            "window_id": window_id,
            "block_id": block_id,
            "window_ns": [start_ns, end_ns],
            "physical_source_months": source["physical_source_months"],
        },
        "quality": train,
        "economic_status": "SCENARIO_ONLY",
        "qualification": "D1_DEVELOPMENT_DIAGNOSTIC_ONLY_NOT_G1_G2",
        "market_rows_decoded_after_registered_train_reservation": True,
        "protected_month_guard_passed_before_decode": True,
    }


def fit_development_scaler(doc: dict[str, Any], output: Path) -> tuple[Any, dict[str, Any]]:
    """Fit the host's required observation scaler only on registered development-fit."""
    source = doc["source"]
    market = ROOT / source["market_path"]
    marker = ROOT / source["marker_path"]
    imported = json.loads((ROOT / source["import_manifest_path"]).read_text(encoding="utf-8"))
    start_ns, end_ns = map(_ns, doc["roles"]["development_fit"])
    scope_start, scope_end = imported["start_ns"], imported["end_ns"]
    if not scope_start <= start_ns < end_ns <= scope_end:
        raise ValueError("development-fit window is outside its registered Train read scope")
    reject_protected_monthly_archive_source(market, market_key(doc["market"]["instrument"]))
    output.mkdir(parents=True, exist_ok=True)
    harness.MARKET = market
    harness.MARKER = marker
    harness.IMPORT = ROOT / source["import_manifest_path"]
    harness.RULES = ROOT / doc["rules"]["path"]
    harness.CONFIG = ROOT / doc["native_execution_template"]["path"]
    harness.OUT_BASE = output
    harness.FEATURE_PROFILE = "causal_mtf_v5"
    harness.FOLD_ID = doc["protocol_id"]
    label = "route-a-1m-s07-scaler-development-fit"
    owner = ACTIVE_OWNER.set(doc["protocol_id"])
    try:
        config, factory = harness.make_config(
            label=label,
            marker_sha=source["marker_sha256"],
            scope_start=scope_start,
            scope_end=scope_end,
            start_ns=start_ns,
            end_ns=end_ns,
            mode="base",
            fit_scaler=True,
            prepare_factory=True,
            source_mode=source["read_mode"],
        )
    finally:
        ACTIVE_OWNER.reset(owner)
    if factory is None or factory.scaler.training_only is not True:
        raise RuntimeError("development-fit did not produce a frozen Train-only scaler")
    raw_scaler = {
        "feature_hash": factory.scaler.feature_hash,
        "mean": list(factory.scaler.mean),
        "std": list(factory.scaler.std),
        "train_window_hash": factory.scaler.train_window_hash,
        "sample_count": factory.scaler.sample_count,
        "training_only": factory.scaler.training_only,
    }
    encoded = json.dumps(raw_scaler, sort_keys=True, separators=(",", ":"), allow_nan=False)
    frozen_sha = sha256(encoded.encode("utf-8")).hexdigest()
    receipt = {
        "schema": "rles.route-a.1m-teacher-development-scaler.v1",
        "protocol_id": doc["protocol_id"],
        "fit_window_ns": [start_ns, end_ns],
        "fit_window_role": "development_fit",
        "scaler_hash": factory.scaler.hash,
        "scaler_payload_sha256": frozen_sha,
        "feature_hash": factory.scaler.feature_hash,
        "train_window_hash": factory.scaler.train_window_hash,
        "sample_count": factory.scaler.sample_count,
        "training_only": factory.scaler.training_only,
        "quality_grade": factory.quality.grade,
        "quality_findings": list(factory.quality.findings),
        "inner_screen_used_for_fit": False,
        "calibration": doc["roles"]["calibration"],
        "economic_status": "SCENARIO_ONLY",
    }
    payload = {**receipt, "scaler": raw_scaler}
    (output / "development-scaler.json").write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return factory.scaler, receipt


def prepare_window_factory(doc: dict[str, Any], output: Path, *, window_id: str,
                           scenario: str, scaler: Any,
                           share_snapshot_histories: bool = False
                           ) -> tuple[dict[str, Any], Any]:
    """Create one scenario/block host factory that reuses the frozen fit scaler."""
    source = doc["source"]
    market = ROOT / source["market_path"]
    marker = ROOT / source["marker_path"]
    imported = json.loads((ROOT / source["import_manifest_path"]).read_text(encoding="utf-8"))
    start_ns, end_ns, _block_id = _window_bounds(doc, window_id)
    reject_protected_monthly_archive_source(market, market_key(doc["market"]["instrument"]))
    if scenario not in doc["cost_scenarios"]:
        raise ValueError("scenario is not frozen by the pre-registration")
    harness.MARKET = market
    harness.MARKER = marker
    harness.IMPORT = ROOT / source["import_manifest_path"]
    harness.RULES = ROOT / doc["rules"]["path"]
    harness.CONFIG = ROOT / doc["native_execution_template"]["path"]
    harness.OUT_BASE = output
    harness.FEATURE_PROFILE = "causal_mtf_v5"
    harness.FOLD_ID = doc["protocol_id"]
    label = f"route-a-1m-s07-{window_id}-{scenario}-frozen-scaler"
    owner = ACTIVE_OWNER.set(doc["protocol_id"])
    try:
        config, factory = harness.make_config(
            label=label,
            marker_sha=source["marker_sha256"],
            scope_start=imported["start_ns"],
            scope_end=imported["end_ns"],
            start_ns=start_ns,
            end_ns=end_ns,
            mode=scenario,
            fit_scaler=False,
            scaler=scaler,
            prepare_factory=True,
            source_mode=source["read_mode"],
            share_snapshot_histories=share_snapshot_histories,
        )
    finally:
        ACTIVE_OWNER.reset(owner)
    if factory is None or factory.scaler.hash != scaler.hash:
        raise RuntimeError("screen factory did not reuse the frozen development-fit scaler")
    return config, factory


def _money(value: Any, *, field: str) -> tuple[Decimal, str]:
    if not isinstance(value, str):
        raise ValueError(f"native {field} must include an explicit currency")
    parts = value.replace(",", "").split()
    if len(parts) != 2:
        raise ValueError(f"native {field} must contain amount and currency")
    try:
        amount = Decimal(parts[0])
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"native {field} amount is invalid") from exc
    if not amount.is_finite():
        raise ValueError(f"native {field} amount must be finite")
    return amount, parts[1]


def _fill_costs(fills: tuple[dict[str, Any], ...] | list[dict[str, Any]],
                references: dict[str, Decimal]) -> tuple[Decimal, Decimal, Decimal, tuple[dict[str, Any], ...]]:
    fees = Decimal(0)
    slippage = Decimal(0)
    notional = Decimal(0)
    normalized: list[dict[str, Any]] = []
    currency: str | None = None
    for fill in fills:
        commission, fee_currency = _money(fill.get("commission", fill.get("fee")), field="commission")
        client_id = fill.get("client_order_id")
        if not isinstance(client_id, str) or client_id not in references:
            raise ValueError("native fill has no retained fill-reference price")
        try:
            price = Decimal(str(fill["last_px"]))
            quantity = Decimal(str(fill["last_qty"]))
        except (KeyError, InvalidOperation, TypeError, ValueError) as exc:
            raise ValueError("native fill price/quantity is malformed") from exc
        side = fill.get("order_side")
        if (not price.is_finite() or not quantity.is_finite() or price <= 0 or quantity <= 0
                or side not in {"BUY", "SELL"}):
            raise ValueError("native fill facts are invalid")
        reference = references[client_id]
        cost = (price - reference) * quantity * (1 if side == "BUY" else -1)
        if cost < Decimal("-0.00000001"):
            raise ValueError("conservative native fill model produced price improvement")
        if currency is not None and fee_currency != currency:
            raise ValueError("native fills use multiple fee currencies")
        currency = fee_currency
        fees += commission
        slippage += max(Decimal(0), cost)
        notional += price * quantity
        normalized.append(dict(fill))
    return fees, slippage, notional, tuple(normalized)


class _StreamingVisibleContext:
    """Advance the S02 engine from authorized L2 bars at the host decision clock."""

    def __init__(self, factory: Any, *, start_ns: int, end_ns: int, instrument: str,
                 warmup_bars: int) -> None:
        first_close = max(0, start_ns - (warmup_bars + 2) * MINUTE_NS)
        inputs = causal_1m_indicator_inputs(
            factory.store, end_ns, instrument=instrument,
            start_event_ns=first_close, require_latest=False,
        )
        self._bars = tuple(CausalBar(
            open_time_ns=row.open_time_ns,
            close_time_ns=row.close_time_ns,
            available_at_ns=row.available_at_ns,
            open=row.open,
            high=row.high,
            low=row.low,
            close=row.close,
            volume=row.volume,
            timeframe_minutes=1,
        ) for row in inputs)
        self._engine = StreamingIndicators1m()
        self._cursor = 0
        self._snapshot: IndicatorSnapshot | None = None

    def __call__(self, env: Any) -> dict[str, Any]:
        context = env.rles_visible_context_inputs()
        decision_ns = context.get("decision_ts_ns")
        if type(decision_ns) is not int:
            raise ValueError("native Gym context omitted its exact decision clock")
        while (self._cursor < len(self._bars)
               and self._bars[self._cursor].available_at_ns <= decision_ns):
            bar = self._bars[self._cursor]
            self._snapshot = self._engine.update(bar, decision_ts_ns=decision_ns)
            self._cursor += 1
        if self._snapshot is None:
            raise ValueError("no completed 1m bar was available by the first decision")
        if self._snapshot.available_at_ns > decision_ns:
            raise ValueError("1m indicator state contains a future availability clock")
        market = dict(context["market"])
        for name, value in self._snapshot.values.items():
            market[name] = VisibleFeature(
                value=float(value), available_at_ns=self._snapshot.available_at_ns,
                unit="route_a_1m_indicator_v1",
            )
        context["market"] = market
        context["exact_market_clocks_ns"] = (
            ("latest_1m_close_ts_ns", self._snapshot.close_time_ns),
        )
        return context


class _TeacherResearchHook1m:
    """Capture S05 facts and native round trips; owns neither reader nor ledger."""

    def __init__(self, *, factory: Any, teacher: Any, output: Path, source_identity: str,
                 scenario: str, exit_variant: str, block_id: str) -> None:
        self.factory = factory
        self.teacher = teacher
        self.output = output
        self.source_identity = source_identity
        self.scenario = scenario
        self.exit_variant = exit_variant
        self.block_id = block_id
        self.records: tuple[TeacherDecisionRecord, ...] = ()
        self.cases: tuple[NativeTradeCase1m, ...] = ()
        self._active: dict[str, Any] | None = None
        self._cases: list[NativeTradeCase1m] = []
        self._pre_accounts: dict[int, dict[str, float]] = {}

    def capture_pre_decision(self, observation: Any, advice: Any,
                             visible_context: dict[str, Any], teacher: Any) -> TeacherPreDecision:
        signal = getattr(teacher, "last_signal", None)
        if signal is None:
            raise TypeError("1m teacher omitted its raw signal state")
        close_ns = observation.exact_market_clock("latest_1m_close_ts_ns")
        availability = observation.feature_available_at_ns
        latest_available = max(availability.values())
        self._pre_accounts[observation.decision_ts_ns] = dict(observation.account)
        required = tuple(getattr(teacher, "required_fields", ()))
        ready = all(name in observation.market for name in required)
        status: Literal["READY", "WARMUP"] = ("READY" if ready else "WARMUP")
        order_veto = (
            "PENDING_NATIVE_ORDER"
            if signal.raw_signal in ("LONG_ENTRY", "SHORT_ENTRY")
            and observation.pending_order_ids else None
        )
        return TeacherPreDecision(
            teacher_id=advice.expert_id,
            recipe_id=teacher.recipe.indicator_recipe_id,
            fold_id=observation.fold_id,
            source_identity=self.source_identity,
            decision_ts_ns=observation.decision_ts_ns,
            latest_1m_close_ts_ns=close_ns,
            latest_1m_available_at_ns=latest_available,
            bar_status=status,
            indicator_values=dict(observation.market),
            raw_signal=signal.raw_signal,
            raw_signal_reason=signal.reason_code,
            position_before=observation.account_value("net_exposure"),
            native_pending_orders=len(observation.pending_order_ids),
            risk_veto_reason=None,
            order_state_veto_reason=order_veto,
            exit_policy_id=self.exit_variant,
            scenario_id=self.scenario,
            economic_status="SCENARIO_ONLY",
            block_id=self.block_id,
        )

    def capture_post_decision(self, pre: TeacherPreDecision, advice: Any,
                              gym_info: dict[str, Any], barrier: Any,
                              order_facts: Any) -> TeacherDecisionRecord:
        trace = barrier.trace
        if bool(gym_info.get("native_liquidation_detected")):
            raise RuntimeError("native liquidation observed; S06 smoke is not scoreable")
        risk_veto = (
            "NATIVE_RISK_SUPPRESSED_ENTRY"
            if (pre.raw_signal in ("LONG_ENTRY", "SHORT_ENTRY")
                and trace.planner_action != "SUBMIT"
                and pre.native_pending_orders == 0
                and abs(trace.risk_adjusted_target - trace.requested_action) > 1e-6)
            else None
        )
        order_veto = (
            "PENDING_NATIVE_ORDER"
            if pre.raw_signal in ("LONG_ENTRY", "SHORT_ENTRY")
            and pre.native_pending_orders > 0 else None
        )
        pre = replace(
            pre,
            risk_veto_reason=risk_veto,
            order_state_veto_reason=order_veto,
        )
        fills = tuple(dict(row) for row in order_facts.native_policy_fills)
        fees, slippage, _notional, _fills = _fill_costs(fills, self.factory.fill_references)
        funding = Decimal(0)
        raw_funding = gym_info.get("native_funding")
        if not isinstance(raw_funding, (tuple, list)):
            raise ValueError("native funding facts are missing from the Gym step")
        for event in raw_funding:
            change, currency = _money(event.get("pnl_change"), field="funding")
            if currency != trace.fee_currency:
                raise ValueError("native funding currency differs from account equity")
            funding += change
        planner = TeacherPlannerRecord(
            advice_intent=advice.intent,
            requested_exposure=advice.requested_exposure,
            risk_adjusted_target=trace.risk_adjusted_target,
            planner_target=trace.planner_target,
            planner_status=trace.planner_action,
            planner_reason=trace.planner_reason,
            submitted_order_ids=trace.submitted_order_ids,
            related_existing_order_ids=trace.related_existing_order_ids,
        )
        post_exposure = trace.actual_exposure_after
        pre_exposure = pre.position_before
        account_state = self._pre_accounts.pop(pre.decision_ts_ns)
        age_minutes = max(0.0, account_state.get("position_duration_ns", 0.0) / MINUTE_NS)
        mfe = max(0.0, account_state.get("position_mfe_bps", 0.0))
        mae = max(0.0, account_state.get("position_mae_bps", 0.0))
        if self._active is not None:
            self._active["fees"] += float(fees)
            self._active["slippage"] += float(slippage)
            self._active["funding"] += float(funding)
            self._active["mfe"] = max(self._active["mfe"], mfe)
            self._active["mae"] = max(self._active["mae"], mae)
            self._active["last_decision_ns"] = pre.decision_ts_ns
            self._active["last_equity"] = trace.native_equity_after
        elif abs(pre_exposure) > 1e-8:
            raise ValueError("native position is open but no teacher entry case is active")

        case_status: Literal["NONE", "ORDINARY_COMPLETE", "OPEN_CENSORED", "TERMINAL_CENSORED", "LIQUIDATION"] = "NONE"
        case_id: str | None = None
        exit_reason: str | None = None
        terminal = bool(gym_info.get("terminal_settled"))
        if self._active is None and abs(pre_exposure) <= 1e-8 and abs(post_exposure) > 1e-8:
            if not fills:
                raise ValueError("native exposure opened without a policy-barrier fill event")
            entry_fill_times: list[int] = []
            for row in fills:
                value = row.get("ts_event", row.get("ts_event_ns"))
                if type(value) is not int:
                    raise ValueError("native entry fill timestamp is missing")
                entry_fill_times.append(value)
            entry_fill_ns = min(entry_fill_times)
            entry_fill = next(
                row for row in fills
                if row.get("ts_event", row.get("ts_event_ns")) == entry_fill_ns
            )
            if entry_fill_ns <= pre.latest_1m_available_at_ns:
                raise ValueError("native entry fill was not later than its available signal")
            _fee, _slip, entry_notional, _normalized = _fill_costs(
                fills, self.factory.fill_references,
            )
            side = 1 if post_exposure > 0 else -1
            entry_px_qty = [
                (Decimal(str(row["last_px"])), Decimal(str(row["last_qty"])))
                for row in fills
                if row.get("order_side") == ("BUY" if side > 0 else "SELL")
            ]
            total_qty = sum((qty for _px, qty in entry_px_qty), Decimal(0))
            if not entry_px_qty or total_qty <= 0:
                raise ValueError("native entry fill side/quantity does not match exposure")
            entry_px = sum((px * qty for px, qty in entry_px_qty), Decimal(0)) / total_qty
            event_id = entry_fill.get("event_id")
            if not isinstance(event_id, str) or not event_id:
                raise ValueError("native entry fill has no event identity")
            self._active = {
                "case_id": "native-case-" + event_id,
                "side": side,
                "entry_pre": pre,
                "entry_fill_ns": entry_fill_ns,
                "entry_fill_px": float(entry_px),
                "entry_notional": float(entry_notional),
                "equity_start": trace.native_equity_before,
                "fees": float(fees),
                "slippage": float(slippage),
                "funding": float(funding),
                "mfe": mfe,
                "mae": mae,
                "last_decision_ns": pre.decision_ts_ns,
                "last_equity": trace.native_equity_after,
                "fee_currency": trace.fee_currency,
                "entry_signal_price": pre.indicator_values["bar_close_1m"],
            }

        if self._active is not None:
            closed_by_policy = abs(post_exposure) <= 1e-8 and bool(fills)
            terminal_censor = terminal and not closed_by_policy
            if closed_by_policy or terminal_censor:
                active = self._active
                case_id = active["case_id"]
                case_status = "TERMINAL_CENSORED" if terminal_censor else "ORDINARY_COMPLETE"
                if case_status == "ORDINARY_COMPLETE":
                    exit_fills = [
                        row for row in fills
                        if row.get("order_side") == ("SELL" if active["side"] > 0 else "BUY")
                    ]
                    if not exit_fills:
                        raise ValueError("native flat transition has no opposing policy fill")
                    exit_fill_times: list[int] = []
                    for row in exit_fills:
                        value = row.get("ts_event", row.get("ts_event_ns"))
                        if type(value) is not int:
                            raise ValueError("native exit fill timestamp is missing")
                        exit_fill_times.append(value)
                    exit_fill_ns = min(exit_fill_times)
                    if exit_fill_ns <= pre.latest_1m_available_at_ns:
                        raise ValueError("native exit fill did not follow its available decision")
                    exit_qty = sum((Decimal(str(row["last_qty"])) for row in exit_fills), Decimal(0))
                    exit_px = sum((Decimal(str(row["last_px"])) * Decimal(str(row["last_qty"]))
                                   for row in exit_fills), Decimal(0)) / exit_qty
                    exit_signal_price = pre.indicator_values["bar_close_1m"]
                    exit_reason = (
                        pre.raw_signal_reason if pre.raw_signal in ("EXIT_LONG", "EXIT_SHORT")
                        else "NATIVE_POLICY_FLAT:" + trace.planner_reason
                    )
                    case = NativeTradeCase1m(
                        trade_case_id=case_id,
                        signal_id=f"{pre.teacher_id}:{active['entry_pre'].decision_ts_ns}:{case_id}",
                        teacher_id=pre.teacher_id,
                        exit_policy_id=pre.exit_policy_id,
                        scenario_id=pre.scenario_id,
                        block_id=pre.block_id,
                        side=active["side"],
                        status="ORDINARY_COMPLETE",
                        entry_decision_ts_ns=active["entry_pre"].decision_ts_ns,
                        entry_signal_price=float(active["entry_signal_price"]),
                        entry_fill_ts_ns=active["entry_fill_ns"],
                        entry_fill_price=active["entry_fill_px"],
                        entry_notional=active["entry_notional"],
                        exit_signal_price=float(exit_signal_price),
                        exit_fill_ts_ns=exit_fill_ns,
                        exit_fill_price=float(exit_px),
                        fees=active["fees"],
                        fee_currency=active["fee_currency"],
                        equity_currency=active["fee_currency"],
                        slippage_cost=active["slippage"],
                        funding=active["funding"],
                        realized_pnl=trace.native_equity_after - active["equity_start"],
                        equity_start=active["equity_start"],
                        equity_end=trace.native_equity_after,
                        mfe_bps=active["mfe"],
                        mae_bps=active["mae"],
                        holding_minutes=max(0.0, (exit_fill_ns - active["entry_fill_ns"]) / MINUTE_NS),
                        exit_reason=exit_reason,
                        economic_status="SCENARIO_ONLY",
                    )
                    self._cases.append(case)
                else:
                    settlement = gym_info.get("terminal_settlement")
                    if not isinstance(settlement, dict):
                        raise ValueError("terminal-censored native case requires L6 settlement facts")
                    terminal_fills = tuple(settlement.get("fills", ()))
                    terminal_fees, terminal_slippage, _terminal_notional, _ = _fill_costs(
                        terminal_fills, self.factory.fill_references,
                    )
                    terminal_funding = Decimal(0)
                    for event in settlement.get("funding", ()):
                        change, currency = _money(event.get("pnl_change"), field="terminal funding")
                        if currency != active["fee_currency"]:
                            raise ValueError("terminal funding currency differs from account equity")
                        terminal_funding += change
                    active["fees"] += float(terminal_fees)
                    active["slippage"] += float(terminal_slippage)
                    active["funding"] += float(terminal_funding)
                    end_equity = float(settlement["equity_after"])
                    case = NativeTradeCase1m(
                        trade_case_id=case_id,
                        signal_id=f"{pre.teacher_id}:{active['entry_pre'].decision_ts_ns}:{case_id}",
                        teacher_id=pre.teacher_id,
                        exit_policy_id=pre.exit_policy_id,
                        scenario_id=pre.scenario_id,
                        block_id=pre.block_id,
                        side=active["side"],
                        status="TERMINAL_CENSORED",
                        entry_decision_ts_ns=active["entry_pre"].decision_ts_ns,
                        entry_signal_price=float(active["entry_signal_price"]),
                        entry_fill_ts_ns=active["entry_fill_ns"],
                        entry_fill_price=active["entry_fill_px"],
                        entry_notional=active["entry_notional"],
                        exit_signal_price=None,
                        exit_fill_ts_ns=None,
                        exit_fill_price=None,
                        fees=active["fees"],
                        fee_currency=active["fee_currency"],
                        equity_currency=active["fee_currency"],
                        slippage_cost=active["slippage"],
                        funding=active["funding"],
                        realized_pnl=end_equity - active["equity_start"],
                        equity_start=active["equity_start"],
                        equity_end=end_equity,
                        mfe_bps=active["mfe"],
                        mae_bps=active["mae"],
                        holding_minutes=max(0.0, (int(settlement["settlement_barrier_ts_ns"])
                                                    - active["entry_fill_ns"]) / MINUTE_NS),
                        exit_reason=None,
                        economic_status="SCENARIO_ONLY",
                    )
                    self._cases.append(case)
                self._active = None

        execution = TeacherExecutionRecord(
            execution_ts_ns=trace.execution_barrier_ts_ns,
            native_filled_order_ids=tuple(sorted(order_facts.filled_order_ids)),
            native_exposure_after=post_exposure,
            fee=float(fees),
            fee_currency=trace.fee_currency,
            equity_currency=trace.fee_currency,
            slippage_cost=float(slippage),
            funding=float(funding),
            account_equity=trace.native_equity_after,
            mfe_bps=mfe,
            mae_bps=mae,
            position_age_minutes=age_minutes,
            case_status=case_status,
            exit_reason=exit_reason,
            case_id=case_id,
            pending_order_ids_after=trace.pending_order_ids_after,
            unfilled_order_reasons=tuple(sorted(
                (order_id, status)
                for order_id in planner.submitted_order_ids
                if order_id not in order_facts.filled_order_ids
                and order_id not in order_facts.pending_order_ids_after
                for status in [next((row["status"] for row in order_facts.native_order_statuses_after
                                    if row["client_order_id"] == order_id), "NATIVE_TERMINAL_STATE_UNKNOWN")]
            )),
        )
        return TeacherDecisionRecord(pre=pre, planner=planner, execution=execution)

    def build_native_cases(self, records: tuple[TeacherDecisionRecord, ...]) -> tuple[NativeTradeCase1m, ...]:
        if self._active is not None:
            raise ValueError("active native position reached report build without terminal censoring")
        self.records = records
        self.cases = tuple(self._cases)
        return self.cases

    def commit(self, records: tuple[TeacherDecisionRecord, ...],
               cases: tuple[NativeTradeCase1m, ...]) -> None:
        submitted = tuple(order_id for row in records if row.planner
                          for order_id in row.planner.submitted_order_ids)
        filled = tuple(order_id for row in records if row.execution
                       for order_id in row.execution.native_filled_order_ids)
        write_teacher_research_artifacts_1m(
            self.output / "teacher-research-1m", records, cases,
            expected_submitted_order_ids=submitted,
            expected_filled_order_ids=filled,
            expected_case_ids=tuple(case.trade_case_id for case in cases),
        )



def _run_native_window(doc: dict[str, Any], prereg_sha: str, output: Path,
                       *, teacher_id: str, exit_variant: NativeExitVariant,
                       scenario: str,
                       window_id: str = "smoke", prepared: tuple[dict[str, Any], Any] | None = None,
                       ema_reversal_mode: EmaReversalMode = "EXIT_ONLY_V1",
                       bounds_override: tuple[int, int, str] | None = None,
                       seed: int = 20261026,
                       teacher_override: Any | None = None,
                       run_identity: Any | None = None) -> dict[str, Any]:
    started = monotonic()
    source = doc["source"]
    market = ROOT / source["market_path"]
    marker = ROOT / source["marker_path"]
    imported = json.loads((ROOT / source["import_manifest_path"]).read_text(encoding="utf-8"))
    scope_start, scope_end = imported["start_ns"], imported["end_ns"]
    start_ns, end_ns, block_id = (
        bounds_override if bounds_override is not None
        else _window_bounds(doc, window_id)
    )
    reject_protected_monthly_archive_source(market, market_key(doc["market"]["instrument"]))

    harness.MARKET = market
    harness.MARKER = marker
    harness.IMPORT = ROOT / source["import_manifest_path"]
    harness.RULES = ROOT / doc["rules"]["path"]
    harness.CONFIG = ROOT / doc["native_execution_template"]["path"]
    harness.OUT_BASE = output
    harness.FEATURE_PROFILE = "causal_mtf_v5"
    harness.FOLD_ID = doc["protocol_id"]
    reversal_suffix = (
        f"-{ema_reversal_mode}" if teacher_id == "T01_EMA_9_21" else ""
    )
    label = f"native-{teacher_id}-{exit_variant}-{scenario}-{block_id}{reversal_suffix}"
    if prepared is None:
        if window_id == "smoke":
            owner = ACTIVE_OWNER.set(doc["protocol_id"])
            try:
                config, factory = harness.make_config(
                    label=label,
                    marker_sha=source["marker_sha256"],
                    scope_start=scope_start,
                    scope_end=scope_end,
                    start_ns=start_ns,
                    end_ns=end_ns,
                    mode=scenario,
                    fit_scaler=True,
                    prepare_factory=True,
                    source_mode=source["read_mode"],
                )
            finally:
                ACTIVE_OWNER.reset(owner)
        else:
            scaler_dir = output / "preparation"
            scaler, _scaler_receipt = fit_development_scaler(doc, scaler_dir)
            config, factory = prepare_window_factory(
                doc, scaler_dir, window_id=window_id, scenario=scenario, scaler=scaler,
            )
    else:
        config, factory = prepared
    if factory is None:
        raise RuntimeError("registered native factory preparation returned no factory")
    if getattr(factory, "quality", None) is None:
        raise RuntimeError("prepared native factory omitted its Train quality receipt")

    scenario_costs = doc["cost_scenarios"][scenario]
    round_trip_cost_bps = 2 * (
        scenario_costs["taker_fee_bps_per_leg"]
        + scenario_costs["slippage_bps_per_leg"]
    )
    exit_round_trip_cost_bps = (
        round_trip_cost_bps if exit_variant in ("X3", "X4", "X5", "X8") else None
    )
    if teacher_override is not None:
        teacher = teacher_override
        if getattr(teacher, "expert_id", None) != teacher_id:
            raise ValueError("teacher override identity differs from the requested teacher id")
    elif teacher_id in (
        ROC_COST_HYSTERESIS_TEACHER_ID,
        ROC_COST_HYSTERESIS_FADE_TEACHER_ID,
    ):
        teacher = build_teacher_1m(
            teacher_id,
            exit_variant=exit_variant,
            roc_round_trip_cost_bps=round_trip_cost_bps,
            exit_round_trip_cost_bps=exit_round_trip_cost_bps,
        )
    else:
        teacher = build_teacher_1m(
            teacher_id,
            exit_variant=exit_variant,
            reversal_mode=ema_reversal_mode if teacher_id == "T01_EMA_9_21"
            else "EXIT_ONLY_V1",
            exit_round_trip_cost_bps=exit_round_trip_cost_bps,
        )
    context = _StreamingVisibleContext(
        factory,
        start_ns=start_ns,
        end_ns=end_ns,
        instrument=doc["market"]["instrument"],
        warmup_bars=max(doc["roles"]["smoke_warmup_bars"], teacher.recipe.warmup_bars),
    )
    provider, grant = harness.make_provider(
        factory,
        teacher,
        source["marker_sha256"],
        f"{teacher_id}-{exit_variant}-{scenario}-{block_id}-seed-{seed}",
        market_feature_map=CAUSAL_1M_INDICATORS_V1_ROUTE_A_FEATURES,
    )
    provider.visible_context = context
    hook = _TeacherResearchHook1m(
        factory=factory,
        teacher=teacher,
        output=output,
        source_identity=source["marker_sha256"],
        scenario=scenario,
        exit_variant=exit_variant,
        block_id=block_id,
    )
    duration_minutes = (end_ns - start_ns) // MINUTE_NS
    episode_steps = duration_minutes - 3
    if not 1 <= episode_steps:
        raise ValueError("pre-registered window is too short for native episode drain")
    if window_id == "smoke" and episode_steps > doc["resource_limits"]["max_decision_bars_per_smoke"]:
        raise ValueError("pre-registered smoke window exceeds the frozen decision-bar limit")
    if window_id != "smoke" and episode_steps > duration_minutes:
        raise ValueError("native episode decision count exceeds its registered window")
    barriers, native = provider.rollout_with_m22_trace(
        factory=factory,
        host_rollout=application.rollout,
        grant=grant,
        seed=seed,
        decision_policy=None,
        rollout_id=label,
        episode_steps=episode_steps,
        reset_options={"rles_episode_index": seed},
        teacher_funnel_hook=hook,
    )
    if len(barriers) != episode_steps:
        raise RuntimeError("native smoke decision count differs from its frozen window")
    verification = native.get("route_a_native_order_verification")
    if (not isinstance(verification, dict)
            or verification.get("barriers") != episode_steps
            or verification.get("authority") != "retained_m22_session_snapshots"):
        raise RuntimeError("M2.2 did not verify every S06 decision barrier")
    if native.get("liquidations") != 0 or native.get("liquidations_observed") is not True:
        raise RuntimeError("native smoke has a liquidation or missing liquidation evidence")
    if native.get("route_a_teacher_1m_research", {}).get("decision_rows") != episode_steps:
        raise RuntimeError("S05 row count differs from verified native decisions")
    account_report = native_report(
        factory.snapshots,
        fill_references=factory.fill_references,
        sample_interval_ns=MINUTE_NS,
        allow_price_improvement=False,
        include_account_rows=False,
    )
    metrics = account_report.get("metrics")
    if not isinstance(metrics, dict):
        raise RuntimeError("native account metrics were not emitted")
    if not factory.snapshots:
        raise RuntimeError("native account snapshot history is empty")
    first_snapshot, last_snapshot = factory.snapshots[0], factory.snapshots[-1]
    result = {
        "schema": ("rles.route-a.1m-teacher-native-smoke.v1" if window_id == "smoke"
                   else "rles.route-a.1m-teacher-native-run.v1"),
        "status": ("ENGINEERING_SMOKE_PASS_SCENARIO_ONLY" if window_id == "smoke"
                   else "ENGINEERING_RESEARCH_BLOCK_PASS_SCENARIO_ONLY"),
        "protocol_id": doc["protocol_id"],
        "preregistration_sha256": prereg_sha,
        "verified_feature_recipe_path": str(FEATURE_RECIPE),
        "verified_feature_recipe_sha256": doc["feature_recipe"]["sha256"],
        "teacher": teacher_id,
        "teacher_recipe": {
            "indicator_recipe_id": teacher.recipe.indicator_recipe_id,
            "parameters": teacher.recipe.parameters,
            "entry_rule_id": teacher.recipe.entry_rule_id,
            "exit_variant": exit_variant,
            "exit_policy_id": teacher.recipe.indicator_exit_rule_id,
            "ema_reversal_mode": getattr(teacher, "reversal_mode", None),
            "higher_timeframe_mode": teacher.recipe.htf_mode,
            "target_exposure": teacher.recipe.target_exposure,
        },
        "execution": {
            "scenario": scenario,
            "seed": seed,
            "scenario_costs": doc["cost_scenarios"][scenario],
            "starting_balance": doc["native_execution_template"]["starting_balance"],
            "risk_budget_notional": doc["native_execution_template"]["risk_budget_notional"],
            "max_leverage": doc["native_execution_template"]["hard_risk_max_leverage"],
            "scaler_hash": factory.scaler.hash,
            "scaler_fit_window_hash": factory.scaler.train_window_hash,
            "native_snapshot_compaction": factory.compact_report_snapshots,
            "native_snapshot_history_sharing": factory.share_snapshot_histories,
        },
        "source": {
            "snapshot_hash": source["snapshot_hash"],
            "marker_sha256": source["marker_sha256"],
            "registered_read_scope_ns": [scope_start, scope_end],
            "window_id": window_id,
            "window_ns": [start_ns, end_ns],
            "events": source["events"],
            "quality_grade": factory.quality.grade,
            "quality_findings": list(factory.quality.findings),
        },
        "decisions": {
            "requested": episode_steps,
            "observed": len(barriers),
            "m22_verified": verification["barriers"],
            "policy_fill_events": verification["policy_fill_events"],
            "submitted_order_count": native["route_a_teacher_1m_research"]["submitted_order_ids"],
            "filled_order_count": native["route_a_teacher_1m_research"]["filled_order_ids"],
            "native_trade_cases": native["route_a_teacher_1m_research"]["native_trade_cases"],
        },
        "account": {
            "equity_currency": first_snapshot.currency,
            "equity_start": float(first_snapshot.equity),
            "equity_end": float(last_snapshot.equity),
            "net_return": metrics.get("net_return"),
            "max_drawdown": metrics.get("max_drawdown"),
            "fees": metrics.get("total_fee"),
            "slippage_cost": metrics.get("total_slippage_cost"),
            "funding": metrics.get("total_funding"),
            "turnover_ratio": metrics.get("turnover_ratio"),
            "liquidations": native["liquidations"],
        },
        "qualification": "D1_DEVELOPMENT_DIAGNOSTIC_ONLY_NOT_G1_G2",
        "economic_status": "SCENARIO_ONLY",
        "block_id": block_id,
        "runtime_seconds": monotonic() - started,
        "limitations": [
            "development Train diagnostic only; no G1/G2 qualification",
            "public market history and scenario instrument, fee, margin, and fill assumptions",
            "native Nautilus execution/account engine; no independent shadow ledger",
            "bounded single-teacher smoke is not an economic teacher selection",
        ],
    }
    if run_identity is not None:
        from route_a_run_identity import RunIdentityV1, require_identity

        if not isinstance(run_identity, RunIdentityV1):
            raise TypeError("RunIdentityV1 required for fresh teacher screen")
        result["run_identity"] = run_identity.document()
        require_identity(result, run_identity)
    destination = output / ("native-smoke.json" if window_id == "smoke" else "native-run.json")
    destination.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )
    # Native facts, cases and the account report are already persisted; release
    # per-bar snapshots before the next independent S07 account reuses the store.
    factory.snapshots.clear()
    factory.teacher_snapshots.clear()
    factory.fill_references.clear()
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher", choices=TEACHERS, default="T01_EMA_9_21")
    parser.add_argument(
        "--exit", choices=("X1", "X2", "X3", "X4", "X5", "X6", "X7", "X8"), default="X1",
    )
    parser.add_argument(
        "--ema-reversal-mode",
        choices=("EXIT_ONLY_V1", "REVERSE_AFTER_FLAT_V1"),
        default="EXIT_ONLY_V1",
    )
    parser.add_argument("--scenario", choices=("base", "stress"), default="base")
    parser.add_argument("--window", choices=WINDOW_IDS, default="smoke")
    parser.add_argument("--prereg", type=Path, default=ROOT_DEFAULT_PREREG)
    parser.add_argument(
        "--feature-recipe-override",
        type=Path,
        help="alternate immutable snapshot; its bytes must match the preregistered SHA-256",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-validate-source", action="store_true")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"immutable S06 output already exists: {args.output}")
    recipe_override = args.feature_recipe_override
    if recipe_override is not None and not recipe_override.is_absolute():
        recipe_override = ROOT / recipe_override
    if recipe_override is not None:
        recipe_override = recipe_override.resolve()
    doc, prereg_sha = verify_preregistration(
        args.prereg, feature_recipe_override=recipe_override,
    )
    global FEATURE_RECIPE
    FEATURE_RECIPE = recipe_override or (ROOT / doc["feature_recipe"]["path"])
    if args.teacher not in TEACHERS:
        parser.error("unsupported teacher id")
    if args.scenario not in doc["cost_scenarios"]:
        parser.error("scenario is not frozen by the pre-registration")
    if (args.ema_reversal_mode == "REVERSE_AFTER_FLAT_V1"
            and (args.teacher != "T01_EMA_9_21" or args.exit != "X1")):
        parser.error("reverse-after-flat mode only applies to EMA/X1")
    args.output.mkdir(parents=True)
    try:
        if args.dry_validate_source:
            report = _source_audit(doc, prereg_sha, args.output, window_id=args.window)
            destination = args.output / "source-audit.json"
            destination.write_text(
                json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
                encoding="utf-8",
            )
            print(json.dumps({
                "report": str(destination),
                "quality_grade": report["quality"].get("grade"),
                "findings": report["quality"].get("findings"),
                "economic_status": report["economic_status"],
                "qualification": report["qualification"],
            }, sort_keys=True))
        else:
            report = _run_native_window(
                doc, prereg_sha, args.output,
                teacher_id=args.teacher, exit_variant=cast(NativeExitVariant, args.exit),
                scenario=args.scenario,
                window_id=args.window,
                ema_reversal_mode=cast(EmaReversalMode, args.ema_reversal_mode),
            )
            print(json.dumps({
                "report": str(args.output / ("native-smoke.json" if args.window == "smoke"
                                              else "native-run.json")),
                "status": report["status"],
                "decisions": report["decisions"],
                "account": report["account"],
                "qualification": report["qualification"],
            }, sort_keys=True))
    except Exception as exc:
        failure = {
            "schema": "rles.route-a.1m-teacher-native-failure.v1",
            "stage": "source_audit" if args.dry_validate_source else "native_smoke",
            "teacher": args.teacher,
            "exit_variant": args.exit,
            "scenario": args.scenario,
            "exception_type": type(exc).__name__,
            "error": str(exc),
            "qualification": "BLOCKED_NOT_A_PASS",
            "economic_status": "SCENARIO_ONLY",
        }
        (args.output / "failure.json").write_text(
            json.dumps(failure, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(failure, sort_keys=True))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
