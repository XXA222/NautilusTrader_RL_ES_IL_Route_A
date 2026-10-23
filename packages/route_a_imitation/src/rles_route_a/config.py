"""Strict config parsing: proposed Route-A schema, not an existing RLES CLI schema."""
from __future__ import annotations

from dataclasses import dataclass, fields
from importlib.resources import files
import json
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, AbstractSet


def _strict(cls: type, values: Mapping[str, Any]) -> Any:
    actual, allowed = set(values), {field.name for field in fields(cls)}
    if actual != allowed:
        raise ValueError(f"{cls.__name__} config keys mismatch missing={sorted(allowed - actual)} unknown={sorted(actual - allowed)}")
    return cls(**dict(values))


def _section(name: str, value: Any, *, required: set[str],
             optional: AbstractSet[str] = frozenset()) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} config must be an object")
    actual = set(value)
    missing, unknown = required - actual, actual - required - optional
    if missing or unknown:
        raise ValueError(
            f"{name} config keys mismatch missing={sorted(missing)} unknown={sorted(unknown)}"
        )
    return value


def _finite_number(section: str, key: str, value: Any, *, lower: float,
                   upper: float | None = None, lower_inclusive: bool = False,
                   upper_inclusive: bool = False) -> None:
    if (type(value) not in (int, float) or not isfinite(value)
            or (value < lower if lower_inclusive else value <= lower)
            or (upper is not None and
                (value > upper if upper_inclusive else value >= upper))):
        lower_bound = "inclusive" if lower_inclusive else "exclusive"
        upper_bound = "inclusive" if upper_inclusive else "exclusive"
        raise ValueError(
            f"{section}.{key} must be finite and within {lower_bound}/{upper_bound} bounds"
        )


@dataclass(frozen=True)
class BCConfig:
    feature_names: tuple[str, ...]
    context_bars: int = 120
    max_gap_ns: int | None = None
    hidden_size: int = 128
    layers: int = 2
    batch_size: int = 128
    learning_rate: float = 0.0003
    epochs: int = 50
    intent_weight: float = 0.25
    exposure_weight: float = 1.0
    grad_clip: float = 1.0
    seed: int = 31
    inference_mode: str = "continuous_exposure_v1"
    exposure_target_mode: str = "teacher_request_v2"
    stop_loss_fraction: float = 0.0
    stop_cooldown_ns: int = 0
    minimum_position_hold_ns: int = 0
    open_intent_min_confidence: float = 0.0
    intent_temperature: float = 1.0
    intent_class_weighting: str = "none"
    intent_sampling: str = "shuffle"

    def __post_init__(self) -> None:
        if self.inference_mode not in {"continuous_exposure_v1", "intent_aware_v1"}:
            raise ValueError("unsupported BC inference mode")
        if self.exposure_target_mode not in {"executed_only_v1", "teacher_request_v2"}:
            raise ValueError("unsupported BC exposure target mode")
        if (type(self.stop_loss_fraction) not in (int, float)
                or not isfinite(self.stop_loss_fraction)
                or not 0 <= self.stop_loss_fraction < 1
                or type(self.stop_cooldown_ns) is not int
                or self.stop_cooldown_ns < 0
                or type(self.minimum_position_hold_ns) is not int
                or self.minimum_position_hold_ns < 0
                or type(self.open_intent_min_confidence) not in (int, float)
                or not isfinite(self.open_intent_min_confidence)
                or not 0 <= self.open_intent_min_confidence <= 1
                or type(self.intent_temperature) not in (int, float)
                or not isfinite(self.intent_temperature)
                or self.intent_temperature <= 0
                or (self.stop_cooldown_ns > 0 and self.stop_loss_fraction == 0)):
            raise ValueError("stop cooldown requires a valid stop loss threshold")
        if self.intent_class_weighting not in {
            "none", "inverse_sqrt_v1", "inverse_frequency_v1"
        }:
            raise ValueError("unsupported BC intent class weighting")
        if self.intent_sampling not in {
            "shuffle", "balanced_intent_v1", "sqrt_balanced_intent_v1",
            "moderately_balanced_intent_v1"
        }:
            raise ValueError("unsupported BC intent sampling mode")
        if self.max_gap_ns is not None and (type(self.max_gap_ns) is not int or self.max_gap_ns <= 0):
            raise ValueError("max_gap_ns must be a positive integer nanosecond threshold")
        if not self.feature_names or len(set(self.feature_names)) != len(self.feature_names):
            raise ValueError("feature names missing or duplicated")
        if any(not isinstance(x, str) or not x for x in self.feature_names):
            raise ValueError("invalid feature name")
        if any(type(getattr(self, n)) is not int or getattr(self, n) <= 0 for n in ("context_bars", "hidden_size", "layers", "batch_size", "epochs")):
            raise ValueError("positive integer train settings required")
        if (not all(type(getattr(self, k)) in (int, float) and isfinite(getattr(self, k))
                    for k in ("learning_rate", "grad_clip", "intent_weight", "exposure_weight"))
                or not (0 < self.learning_rate < 1 and self.grad_clip > 0
                        and self.intent_weight >= 0 and self.exposure_weight > 0)) :
            raise ValueError("invalid loss/optimizer parameters")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be non-negative")

    @classmethod
    def parse(cls, payload: Mapping[str, Any]) -> BCConfig:
        d = dict(payload)
        d.setdefault("max_gap_ns", None)  # additive v0.2 / v0.1 manifest compatibility
        d.setdefault("inference_mode", "continuous_exposure_v1")
        # Older bundles masked requested-target regression whenever the native
        # action was not filled. Preserve that recipe identity on parse.
        d.setdefault("exposure_target_mode", "executed_only_v1")
        d.setdefault("minimum_position_hold_ns", 0)
        d.setdefault("open_intent_min_confidence", 0.0)
        d.setdefault("intent_temperature", 1.0)
        d.setdefault("intent_class_weighting", "none")
        d.setdefault("intent_sampling", "shuffle")
        if "feature_names" in d:
            d["feature_names"] = tuple(d["feature_names"])
        return _strict(cls, d)


@dataclass(frozen=True)
class DAggerConfig:
    iterations: int = 3
    expert_fraction: float = 0.7
    sample_split: str = "train"

    def __post_init__(self) -> None:
        if type(self.iterations) is not int or self.iterations < 1:
            raise ValueError("DAgger iterations must be a positive integer")
        if (type(self.expert_fraction) not in (int, float)
                or not isfinite(self.expert_fraction)
                or not 0 < self.expert_fraction < 1):
            raise ValueError("DAgger expert_fraction must be finite and between zero and one")
        if self.sample_split != "train":
            raise PermissionError("DAgger labels may only be collected from Train")

    @classmethod
    def parse(cls, payload: Mapping[str, Any]) -> DAggerConfig:
        return _strict(cls, payload)


@dataclass(frozen=True)
class RouteAConfig:
    schema: str
    action_schema: str
    market: dict[str, Any]
    teacher: dict[str, Any]
    bc: BCConfig
    dagger: DAggerConfig
    iql: dict[str, Any]
    evaluation: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> RouteAConfig:
        return cls._parse_payload(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def load_default(cls) -> RouteAConfig:
        """Load the validated config example shipped inside the wheel."""
        resource = files("rles_route_a").joinpath("configs", "route_a_v1.json")
        return cls._parse_payload(json.loads(resource.read_text(encoding="utf-8")))

    @classmethod
    def _parse_payload(cls, d: Any) -> RouteAConfig:
        if not isinstance(d, dict) or not isinstance(d.get("bc"), dict):
            raise ValueError("invalid route A config object")
        market = _section("market", d.get("market"), required={
            "base_bar", "informative_bars", "train_fold", "context_bars",
        })
        teacher = _section("teacher", d.get("teacher"), required={
            "strategies", "require_economic_gate",
        }, optional={"selector"})
        iql = _section("iql", d.get("iql"), required={
            "enabled_only_if", "freeze_encoder", "expectile", "gamma", "beta", "max_weight",
        })
        evaluation = _section("evaluation", d.get("evaluation"), required={
            "scenarios", "reuse_l6_research_ledger", "final_holdout_once",
        })

        if (market.get("base_bar") != "1m"
                or market.get("informative_bars") != ["1h", "1d"]):
            raise ValueError("only the causal 1m/1h/1d market schema is supported")
        if market.get("train_fold") not in {"60/15/15", "360/15/15"}:
            raise ValueError("only host-authorized WFO protocols are supported")
        if type(market.get("context_bars")) is not int or market["context_bars"] <= 0:
            raise ValueError("market.context_bars must be a positive integer")

        strategies = teacher.get("strategies")
        supported_strategies = {"trend", "breakout", "mean_reversion", "wait", "recovery"}
        if (not isinstance(strategies, list) or not strategies
                or any(not isinstance(item, str) or item not in supported_strategies
                       for item in strategies)
                or len(set(strategies)) != len(strategies)):
            raise ValueError("teacher.strategies must be unique supported Route A teachers")
        if teacher.get("require_economic_gate") is not True:
            raise PermissionError("Route A teacher selection requires the Train-inner economic gate")
        if "selector" in teacher and teacher["selector"] != "train_frozen_rule":
            raise ValueError("teacher.selector must use train_frozen_rule")

        if iql.get("enabled_only_if") != "RA3_PASS" or iql.get("freeze_encoder") is not True:
            raise PermissionError("IQL requires RA3_PASS and a frozen BC encoder")
        _finite_number("iql", "expectile", iql.get("expectile"), lower=0, upper=1)
        _finite_number("iql", "gamma", iql.get("gamma"), lower=0, upper=1,
                       lower_inclusive=True)
        _finite_number("iql", "beta", iql.get("beta"), lower=0)
        _finite_number("iql", "max_weight", iql.get("max_weight"), lower=0)

        if evaluation.get("scenarios") != ["zero", "base", "stress"]:
            raise ValueError("evaluation.scenarios must retain zero, base, and stress")
        if (evaluation.get("reuse_l6_research_ledger") is not True
                or evaluation.get("final_holdout_once") is not True):
            raise PermissionError("formal evaluation must reuse L6 and keep Holdout one-shot")
        d["bc"] = BCConfig.parse(d["bc"])
        if not isinstance(d.get("dagger"), dict):
            raise ValueError("invalid Route A DAgger config")
        d["dagger"] = DAggerConfig.parse(d["dagger"])
        obj = _strict(cls, d)
        if obj.schema != "route_a.v1" or obj.action_schema != "single_exposure_v1":
            raise ValueError("unsupported Route A or action schema")
        if obj.market["context_bars"] != obj.bc.context_bars:
            raise ValueError("market and BC context_bars must match")
        return obj
