"""Stateful episode-aware BC inference; same features as frozen training bundle."""
from __future__ import annotations
from collections import deque
import numpy as np
import torch
from rles_route_a.contracts import INTENTS, PointInTimeObservation
from rles_route_a.config import BCConfig
from rles_route_a.models.bc import BCGRUPolicy


def intent_aware_exposure(intent: str, raw_target: float, current: float) -> float:
    """Project the jointly trained intent/target heads onto a coherent target.

    HOLD/FLAT/EXIT use their discrete semantics directly. ADD/REDUCE are
    accepted only when the regression target agrees with the classified move;
    conflicting heads resolve to the reached exposure (no trade).
    """
    if intent not in INTENTS or not np.isfinite(raw_target) or not np.isfinite(current):
        raise ValueError("intent-aware BC projection requires finite supported values")
    if not -1 <= raw_target <= 1 or not -1 <= current <= 1:
        raise ValueError("BC projection target outside exposure contract")
    if intent in {"FLAT", "EXIT"}:
        return 0.0
    if intent == "HOLD":
        return float(current)
    if intent == "OPEN_LONG":
        if raw_target <= 0:
            return float(current)
        if current > 1e-6 and raw_target < current:
            return float(current)
        return float(raw_target)
    if intent == "OPEN_SHORT":
        if raw_target >= 0:
            return float(current)
        if current < -1e-6 and raw_target > current:
            return float(current)
        return float(raw_target)
    if intent == "ADD":
        if abs(current) <= 1e-6:
            return float(current)
        if current * raw_target <= 0 or abs(raw_target) <= abs(current):
            return float(current)
        return float(raw_target)
    if intent == "REDUCE":
        if abs(current) <= 1e-6:
            return 0.0
        if current * raw_target <= 0 or abs(raw_target) >= abs(current):
            return float(current)
        return float(raw_target)
    raise AssertionError("unreachable BC intent")


class BCInferencePolicy:
    def __init__(self, model: BCGRUPolicy, cfg: BCConfig, *, device: str = "cpu"):
        if tuple(model.encoder.feature_names) != tuple(cfg.feature_names):
            raise ValueError("model/schema mismatch")
        self.model = model.to(device).eval()
        self.cfg, self.device = cfg, device
        self._series: deque[np.ndarray] = deque(maxlen=cfg.context_bars)
        self._episode: str | None = None
        self._observation_identity: tuple[str, str, str] | None = None
        self._last_ts = -1
        self._stop_cooldown_until_ns = 0
        self._position_since_ns: int | None = None
        self._last_observed_exposure = 0.0

    def reset(self, episode_id: str) -> None:
        if not episode_id:
            raise ValueError("episode ID required")
        self._episode = episode_id
        self._observation_identity = None
        self._series.clear()
        self._last_ts = -1
        self._stop_cooldown_until_ns = 0
        self._position_since_ns = None
        self._last_observed_exposure = 0.0

    def predict_exposure(self, observation: PointInTimeObservation) -> float:
        if observation.episode_id != self._episode:
            raise ValueError("reset() required before new episode")
        if observation.decision_ts_ns <= self._last_ts:
            raise ValueError("observation clock regressed or repeated")
        identity = (observation.source_id, observation.fold_id, observation.feature_spec_hash)
        if self._observation_identity is not None and identity != self._observation_identity:
            raise ValueError("observation source/fold/feature identity changed; reset required")
        # Work on a candidate history: a missing feature, float32 overflow or
        # non-finite model output must not advance recurrent deployment state.
        if (self._last_ts >= 0 and self.cfg.max_gap_ns is not None
                and observation.decision_ts_ns - self._last_ts > self.cfg.max_gap_ns):
            candidate: deque[np.ndarray] = deque(maxlen=self.cfg.context_bars)
        else:
            candidate = deque(self._series, maxlen=self.cfg.context_bars)
        current_exposure = observation.account_value("net_exposure")
        position_since_ns = self._position_since_ns
        if abs(current_exposure) <= 1e-8:
            position_since_ns = None
        elif (position_since_ns is None
              or self._last_observed_exposure * current_exposure < 0):
            position_since_ns = observation.decision_ts_ns
        minimum_hold_active = (
            position_since_ns is not None
            and self.cfg.minimum_position_hold_ns > 0
            and observation.decision_ts_ns - position_since_ns
            < self.cfg.minimum_position_hold_ns
        )
        stop_cooldown_until_ns = self._stop_cooldown_until_ns
        stop_triggered = (
            self.cfg.stop_loss_fraction > 0
            and abs(current_exposure) > 1e-8
            and observation.account_value("unrealized_pnl_fraction")
            <= -self.cfg.stop_loss_fraction
        )
        if stop_triggered:
            stop_cooldown_until_ns = max(
                stop_cooldown_until_ns,
                observation.decision_ts_ns + self.cfg.stop_cooldown_ns,
            )
        vals = []
        for key in self.cfg.feature_names:
            if key.startswith("market."):
                vals.append(observation.market_value(key[7:]))
            elif key.startswith("account."):
                vals.append(observation.account_value(key[8:]))
            else:
                raise ValueError(f"unsupported inference feature namespace: {key}")
        with np.errstate(over="ignore", invalid="ignore"):
            row = np.asarray(vals, dtype=np.float32)
        if not np.isfinite(row).all():
            raise ValueError("nonfinite/overflowed inference feature")
        candidate.append(row)
        x = np.zeros((1, self.cfg.context_bars, len(vals)), np.float32)
        x[0, -len(candidate):] = np.stack(tuple(candidate))
        with torch.no_grad():
            logits, output = self.model(
                torch.from_numpy(x).to(self.device),
                torch.tensor([len(candidate)], device=self.device),
            )
        if not isinstance(output, torch.Tensor) or output.shape != (1,):
            raise ValueError("invalid actor output shape")
        result = float(output[0].item())
        if not np.isfinite(result) or not -1 <= result <= 1:
            raise ValueError("invalid actor output")
        if self.cfg.inference_mode == "intent_aware_v1":
            if (not isinstance(logits, torch.Tensor) or logits.shape != (1, len(INTENTS))
                    or not bool(torch.isfinite(logits).all().item())):
                raise ValueError("invalid or nonfinite intent logits")
            intent_index = int(logits[0].argmax().item())
            intent = INTENTS[intent_index]
            result = intent_aware_exposure(
                intent, result, current_exposure
            )
            if intent.startswith("OPEN_") and self.cfg.open_intent_min_confidence > 0:
                probabilities = torch.softmax(logits[0] / self.cfg.intent_temperature, dim=-1)
                if (not bool(torch.isfinite(probabilities).all().item())
                        or not bool(torch.isclose(
                            probabilities.sum(), probabilities.new_tensor(1.0),
                            rtol=1e-5, atol=1e-6,
                        ).item())):
                    raise ValueError("invalid intent probabilities")
                if float(probabilities[intent_index].item()) < self.cfg.open_intent_min_confidence:
                    result = float(current_exposure)
        if stop_triggered or (
            self.cfg.stop_cooldown_ns > 0
            and observation.decision_ts_ns < stop_cooldown_until_ns
        ):
            result = 0.0
        elif minimum_hold_active:
            result = float(current_exposure)
        self._series = candidate
        self._observation_identity = identity
        self._last_ts = observation.decision_ts_ns
        self._stop_cooldown_until_ns = stop_cooldown_until_ns
        self._position_since_ns = position_since_ns
        self._last_observed_exposure = current_exposure
        return result
