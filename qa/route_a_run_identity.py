"""A single frozen identity for native screen resume and aggregation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from hashlib import sha256
import json
import re
from typing import Any, Mapping


def canonical_sha(value: Any) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class RunIdentityV1:
    protocol_sha256: str
    errata_sha256: str
    source_errata_sha256: str
    code_manifest_sha256: str
    feature_recipe_sha256: str
    scaler_sha256: str
    warmup_marker_sha256: str
    market_marker_sha256: str
    market_snapshot_sha256: str
    import_manifest_sha256: str
    execution_template_sha256: str
    instrument_rules_sha256: str
    cost_recipe_sha256: str
    teacher_recipe_sha256: str
    candidate_id: str
    exit_variant: str
    scenario: str
    seed: int
    window_ns: tuple[int, int]

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if field.name.endswith("_sha256"):
                if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                    raise ValueError(f"invalid run identity {field.name}")
        for name in ("candidate_id", "exit_variant", "scenario"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"invalid run identity {name}")
        if (type(self.seed) is not int or self.seed < 0
                or not isinstance(self.window_ns, tuple) or len(self.window_ns) != 2
                or any(type(x) is not int for x in self.window_ns)
                or not 0 < self.window_ns[0] < self.window_ns[1]):
            raise ValueError("invalid run identity seed/window")

    def document(self) -> dict[str, Any]:
        result = asdict(self)
        result["window_ns"] = list(self.window_ns)
        return result


def require_identity(report: Mapping[str, Any], expected: RunIdentityV1) -> None:
    """Reject legacy, mixed, or internally inconsistent native reports."""
    stored = report.get("run_identity")
    if stored is None:
        raise RuntimeError("LEGACY_INCOMPATIBLE: native run has no frozen RunIdentityV1")
    if stored != expected.document():
        raise RuntimeError("native run identity differs from the frozen RunIdentityV1")
    source = report.get("source")
    execution = report.get("execution")
    recipe = report.get("teacher_recipe")
    if (not isinstance(source, Mapping) or not isinstance(execution, Mapping)
            or not isinstance(recipe, Mapping)
            or report.get("preregistration_sha256") != expected.protocol_sha256
            or report.get("verified_feature_recipe_sha256") != expected.feature_recipe_sha256
            or canonical_sha(recipe) != expected.teacher_recipe_sha256
            or recipe.get("exit_variant") != expected.exit_variant
            or source.get("marker_sha256") != expected.market_marker_sha256
            or source.get("snapshot_hash") != expected.market_snapshot_sha256
            or source.get("window_ns") != list(expected.window_ns)
            or execution.get("scaler_hash") != expected.scaler_sha256
            or execution.get("scenario") != expected.scenario
            or type(execution.get("seed")) is not int
            or execution.get("seed") != expected.seed
            or canonical_sha(execution.get("scenario_costs")) != expected.cost_recipe_sha256):
        raise RuntimeError("native run facts disagree with its frozen RunIdentityV1")
