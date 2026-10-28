"""Read-only preflight for the frozen Route A teacher/student Train protocol."""
from __future__ import annotations

# mypy: disable-error-code=import-untyped
from datetime import date, datetime, timedelta, timezone
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROOT = Path("/mnt/d/Program Files/NautilusTrader_RL_ES_IL_Route_A")
CONFIG = ROOT / "configs/route_a_teacher_student_train_prereg_20261026.json"
ERRATA = ROOT / "configs/route_a_teacher_student_train_prereg_20261026_errata.json"
SOURCE_ERRATA = ROOT / "configs/route_a_teacher_student_train_prereg_20261026_source_errata.json"
REGISTRY = ROOT / ".artifacts/research-access.sqlite"


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def frozen(path: Path) -> tuple[dict, str]:
    sha = digest(path)
    if path.with_suffix(path.suffix + ".sha256").read_text(encoding="ascii").strip() != sha:
        raise ValueError(f"bad SHA sidecar: {path.name}")
    return json.loads(path.read_text(encoding="utf-8")), sha


def ns(value: str) -> int:
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if stamp.tzinfo is None:
        raise ValueError("timestamp timezone required")
    return int(stamp.astimezone(timezone.utc).timestamp() * 1_000_000_000)


def windows(rows: list[list[str]], label: str) -> list[tuple[int, int]]:
    result = [(ns(a), ns(b)) for a, b in rows]
    if not result or any(a >= b for a, b in result):
        raise ValueError(f"invalid {label} windows")
    if any(left[1] != right[0] for left, right in zip(result, result[1:])):
        raise ValueError(f"{label} windows must be contiguous and disjoint")
    return result


def main() -> None:
    if ROOT.resolve() != EXPECTED_ROOT or sys.version_info[:3] != (3, 14, 7):
        raise RuntimeError("canonical WSL root or Python 3.14.7 mismatch")
    if (ROOT / ".canonical-project-root").read_text(encoding="utf-8").strip() != r"D:\Program Files\NautilusTrader_RL_ES_IL_Route_A":
        raise RuntimeError("canonical root marker mismatch")
    prereg, psha = frozen(CONFIG)
    errata, esha = frozen(ERRATA)
    source_errata, sesha = frozen(SOURCE_ERRATA)
    if prereg.get("status") != "frozen_before_any_source_read":
        raise ValueError("parent protocol was not frozen before source read")
    if errata.get("supersedes_prereg_sha256") != psha or source_errata.get("supersedes_prereg_sha256") != psha or source_errata.get("supersedes_previous_errata_sha256") != esha:
        raise ValueError("pre-source errata chain mismatch")
    scope = prereg["scope"]
    start, scored, end = (scope["source_start_utc_inclusive"], scope["scored_start_utc_inclusive"], scope["end_utc_exclusive"])
    if (start, scored, end) != ("2026-10-23T00:00:00Z", "2026-10-25T00:00:00Z", "2026-11-30T00:00:00Z"):
        raise ValueError("frozen scope changed")
    train = windows(prereg["teacher_screen"]["windows"], "teacher")
    student = windows(prereg["student_screen"]["windows"], "student")
    if train[0][0] != ns(scored) or train[-1][1] != student[0][0] or student[-1][1] != ns(end):
        raise ValueError("teacher/student temporal split changed")
    if prereg["teacher_screen"]["seeds"] != [20261026, 20261027] or prereg["student_screen"]["seeds"] != [20261026, 20261027]:
        raise ValueError("frozen seed matrix changed")
    from rles_route_a.teachers_1m.registry import build_teacher_1m
    candidates = errata["corrected_candidate_matrix"]
    if len(candidates) != 9:
        raise ValueError("teacher candidate count changed")
    for row in candidates:
        build_teacher_1m(row["teacher_id"], exit_variant=row["exit_variant"], reversal_mode=row["reversal_mode"])
    code_overrides = source_errata.get("code_sha256_overrides", {})
    runtime_code = source_errata.get("runtime_code_sha256", {})
    if set(code_overrides) - set(prereg["code_sha256"]):
        raise ValueError("source erratum overrides a file outside the frozen code manifest")
    if set(runtime_code) & set(prereg["code_sha256"]):
        raise ValueError("runtime code binding duplicates the frozen code manifest")
    for rel, frozen_sha in prereg["code_sha256"].items():
        expected = code_overrides.get(rel, frozen_sha)
        if digest(ROOT / rel) != expected:
            raise ValueError(f"teacher/execution code changed: {rel}")
    for rel, expected in runtime_code.items():
        if digest(ROOT / rel) != expected:
            raise ValueError(f"teacher screen runner/runtime code changed: {rel}")
    execution = source_errata["native_execution_template"]
    execution_path = ROOT / execution["path"]
    if digest(execution_path) != execution["sha256"]:
        raise ValueError("native execution and cost template differs from the pre-read erratum")
    environment = json.loads(execution_path.read_text(encoding="utf-8"))["environment"]
    scenario_rules_path = ROOT / ".artifacts/audit-development-20261016/data/scenario-rules.json"
    scenario_rules = json.loads(scenario_rules_path.read_text(encoding="utf-8"))
    instrument_rules = next(
        row["payload"] for row in scenario_rules["events"]
        if row.get("kind") == "instrument_rule"
        and row.get("payload", {}).get("instrument") == "ETHUSDT-PERP.SIM"
    )
    base_fee_bps = float(instrument_rules["taker_fee"]) * 10_000
    base_slippage_bps = float(environment["execution"]["base_fill"]["slippage_bps"])
    stress_fee_bps = base_fee_bps * float(
        environment["execution"]["stress_fill"].get("fee_multiplier", 1)
    )
    stress_slippage_bps = float(
        environment["execution"]["stress_fill"]["slippage_bps"]
    )
    if source_errata["import_invocation"]["output"] != "data/routea_teacher_train_20261023_20261129":
        raise ValueError("fresh Train import output path differs from the frozen source erratum")
    fit = source_errata["warmup_scaler_fit_window_utc"]
    if source_errata.get("feature_profile") != "causal_mtf_v4":
        raise ValueError("teacher screen feature profile differs from the frozen source erratum")
    if source_errata.get("funding_calendar_mode") != "known_schedule_times_ns_equals_frozen_8h_settlement_times_ns":
        raise ValueError("v4 funding countdown must use the frozen settlement schedule")
    if fit != ["2026-10-23T02:02:00Z", scored]:
        raise ValueError("scaler fit must use only the registered pre-scored warmup interval")
    staged = [source_errata["warmup_scaler_import"],
              *source_errata["teacher_window_imports"]]
    if len(source_errata["teacher_window_imports"]) != len(train):
        raise ValueError("each teacher window must have exactly one frozen staged import")
    outputs = [row.get("output") for row in staged]
    if len(set(outputs)) != len(outputs):
        raise ValueError("staged import output paths must be unique")

    def month_keys(first: date, last: date) -> list[str]:
        keys = []
        cursor = date(first.year, first.month, 1)
        finish = date(last.year, last.month, 1)
        while cursor <= finish:
            keys.append(cursor.strftime("%Y-%m"))
            cursor = date(cursor.year + (cursor.month == 12), cursor.month % 12 + 1, 1)
        return keys

    def verify_stage(row: dict, first: date, last: date, label: str) -> None:
        if (row.get("start_day") != first.isoformat()
                or row.get("end_day_inclusive") != last.isoformat()
                or row.get("funding_api_months") != month_keys(first, last)
                or not isinstance(row.get("output"), str)
                or not row["output"].startswith("data/routea_teacher_")):
            raise ValueError(f"{label} staged import differs from its frozen date coverage")

    source_start_day = datetime.fromisoformat(start.replace("Z", "+00:00")).date()
    scored_start_day = datetime.fromisoformat(scored.replace("Z", "+00:00")).date()
    verify_stage(source_errata["warmup_scaler_import"], source_start_day,
                 scored_start_day - timedelta(days=1), "warmup scaler")
    for index, row in enumerate(source_errata["teacher_window_imports"]):
        if row.get("window_index") != index:
            raise ValueError("staged teacher imports must preserve frozen window order")
        window_first = datetime.fromtimestamp(train[index][0] / 1e9, tz=timezone.utc).date()
        window_after = datetime.fromtimestamp(train[index][1] / 1e9, tz=timezone.utc).date()
        verify_stage(row, window_first - timedelta(days=2),
                     window_after - timedelta(days=1), f"teacher window {index}")
    if source_errata["cost_scenarios"] != {
        "base": {
            "taker_fee_bps_per_leg": base_fee_bps,
            "slippage_bps_per_leg": base_slippage_bps,
        },
        "stress": {
            "taker_fee_bps_per_leg": stress_fee_bps,
            "slippage_bps_per_leg": stress_slippage_bps,
        },
    }:
        raise ValueError("Base/Stress costs differ from the frozen execution template")
    importer = ROOT / "packages/data/src/rles_data/binance_archive.py"
    if digest(importer) != source_errata["importer_sha256"]:
        raise ValueError("source importer SHA differs from frozen source erratum")
    invocation = source_errata["import_invocation"]
    if invocation["development_protocol_sha256"] != psha or invocation["funding_api_months"] != ["2026-10", "2026-11"]:
        raise ValueError("source import no longer matches the parent preregistration")
    source_start, source_end = ns(start), ns(end)
    key = invocation["development_access_key"]
    with sqlite3.connect(REGISTRY.resolve().as_uri() + "?mode=ro", uri=True) as db:
        access = db.execute("SELECT owner,market,family,start_ns,end_ns,purpose,identity,state FROM access WHERE access_key=?", (key,)).fetchone()
        test_holdout = []
        for table in ("access", "query_intent"):
            test_holdout.extend(db.execute(f"SELECT 1 FROM {table} WHERE market=? AND family=? AND purpose IN ('test','holdout') AND start_ns<? AND end_ns>?", ("ETHUSDT", "perpetual-market-v1", source_end, source_start)).fetchall())
    if access is None:
        raise PermissionError("pre-read Train access is missing")
    owner, market, family, a, b, purpose, identity, state = access
    if (owner != prereg["protocol_id"] or market != "ETHUSDT" or family != "perpetual-market-v1" or a != source_start or b != source_end or purpose != "development" or state != "recorded" or json.loads(identity).get("protocol_sha256") != psha):
        raise PermissionError("Train registry row differs from frozen protocol")
    with sqlite3.connect(REGISTRY.resolve().as_uri() + "?mode=ro", uri=True) as db:
        child_rows = db.execute(
            "SELECT owner,start_ns,end_ns,identity FROM access WHERE market=? AND family=? "
            "AND purpose='development' AND access_key<>? AND start_ns<? AND end_ns>?",
            ("ETHUSDT", "perpetual-market-v1", key, source_end, source_start),
        ).fetchall()
    other_dev = []
    for child_owner, child_start, child_end, child_identity in child_rows:
        try:
            child = json.loads(child_identity)
        except (TypeError, json.JSONDecodeError):
            child = None
        child_window = child.get("window") if isinstance(child, dict) else None
        if not (
            child_owner == owner and a <= child_start < child_end <= b
            and isinstance(child, dict)
            and isinstance(child.get("environment"), dict)
            and isinstance(child.get("label"), str)
            and child.get("label", "").startswith("train-")
            and child_window == {"start_ns": child_start, "end_ns": child_end}
        ):
            other_dev.append((child_owner, child_start, child_end))
    if test_holdout or other_dev:
        raise PermissionError("registered source range overlaps protected or consumed data")
    staged_paths = [ROOT / row["output"] for row in staged]
    present = sum(path.exists() for path in staged_paths)
    if present == 0:
        status = "PREFLIGHT_PASS_SOURCE_PENDING"
    elif present == len(staged_paths):
        status = "PREFLIGHT_PASS_STAGED_SOURCES_PRESENT_METADATA_NOT_READ"
    else:
        status = "PREFLIGHT_PASS_STAGED_SOURCES_PARTIAL_METADATA_NOT_READ"
    print(json.dumps({"status": status, "protocol_id": prereg["protocol_id"], "prereg_sha256": psha, "teacher_errata_sha256": esha, "source_errata_sha256": sesha, "registry_state": state, "teacher_windows": len(train), "student_screen_windows": len(student), "teacher_candidates": len(candidates), "staged_imports_present": present, "staged_imports_expected": len(staged_paths), "source_start_utc": start, "source_end_utc_exclusive": end, "market_payload_read": False}, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
