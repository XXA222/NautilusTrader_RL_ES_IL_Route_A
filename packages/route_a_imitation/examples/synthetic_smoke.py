"""Synthetic contract-only demo. NO NautilusTrader engine, PnL, or economic claim.

Run from package root:
  PYTHONPATH=src python examples/synthetic_smoke.py
"""
from __future__ import annotations
import json
from tempfile import TemporaryDirectory
from pathlib import Path

import torch
from rles_route_a.contracts import PointInTimeObservation
from rles_route_a.config import BCConfig
from rles_route_a.datasets.sequence_dataset import EpisodeSequenceDataset, LabeledStep
from rles_route_a.models.bc import BCGRUPolicy
from rles_route_a.teachers.trend import TrendTeacher
from rles_route_a.training.bc_trainer import train_bc
from rles_route_a.export.bundle import save_bc_bundle, load_bc_bundle


def main() -> None:
    torch.set_num_threads(1)
    features = ("market.trend_1h", "market.trend_1d", "market.vol_1m",
                "account.net_exposure", "account.drawdown", "account.unrealized_pnl_fraction")
    steps = []
    for ep in range(3):
        for t in range(20):
            clock = 1_700_000_000_000_000_000 + (ep*30+t)*60_000_000_000
            trend = .01 if (ep+t//8) % 2 == 0 else -.01
            o = PointInTimeObservation(
                fold_id="synthetic-train-only", episode_id=f"synthetic-{ep}",
                decision_ts_ns=clock,
                market={"trend_1h": trend, "trend_1d": trend, "vol_1m": .01},
                feature_available_at_ns={"trend_1h": clock-1, "trend_1d": clock-1,
                                         "vol_1m": clock-1},
                account={"net_exposure": 0., "drawdown": 0.,
                         "unrealized_pnl_fraction": 0.})
            a = TrendTeacher().advise(o)
            # This ONLY demonstrates sequence and training APIs. NOT a native
            # executable label: here a toy synthetic projector returns a value.
            steps.append(LabeledStep(o, a, a.requested_exposure))
    cfg = BCConfig(features, context_bars=12, hidden_size=24, layers=1, batch_size=16, epochs=3)
    dataset = EpisodeSequenceDataset(steps, feature_names=features, context_bars=12)
    model = BCGRUPolicy(features, 24, 1)
    history = train_bc(model, dataset, cfg)
    with TemporaryDirectory() as temp:
        save_bc_bundle(model, cfg, Path(temp) / "model")
        restored, _ = load_bc_bundle(Path(temp) / "model")
        item = dataset[len(dataset)-1]
        with torch.no_grad():
            _, action = restored(item["x"].unsqueeze(0), item["length"].reshape(1))
        print(json.dumps({"status": "SYNTHETIC_ENGINEERING_SMOKE_ONLY",
                          "train_rows": len(dataset), "epochs": len(history),
                          "last_train_loss": history[-1].train_loss,
                          "synthetic_action": action.item(),
                          "real_market_data_read": False,
                          "native_engine_run": False,
                          "economic_approval": False}, indent=2))


if __name__ == "__main__":
    main()
