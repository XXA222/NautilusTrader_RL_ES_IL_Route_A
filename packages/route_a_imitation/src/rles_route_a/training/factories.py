"""Deterministic learner construction, separate from continued DAgger fine-tuning.

Do not reseed/reset existing weights at the start of DAgger; that would erase
previous policy learning. Use these factories only for *new* policy replicas.
"""
from __future__ import annotations
import torch
from rles_route_a.config import BCConfig
from rles_route_a.models.bc import BCGRUPolicy
from rles_route_a.training.iql_trainer import IQLTrainer, IQLHyper
from rles_route_a.training.td3bc_trainer import TD3BCTrainer, TD3BCHyper


def new_bc_policy(config: BCConfig, *, device: str = "cpu") -> BCGRUPolicy:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.seed)
        model = BCGRUPolicy(config.feature_names, config.hidden_size, config.layers)
    return model.to(device)


def new_iql(state_dim: int, hyper: IQLHyper, *, seed: int, device: str = "cpu") -> IQLTrainer:
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed out of range")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = IQLTrainer(state_dim, hyper, device=device)
    return model


def new_td3bc(state_dim: int, hyper: TD3BCHyper, *, seed: int, device: str = "cpu") -> TD3BCTrainer:
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed out of range")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = TD3BCTrainer(state_dim, hyper, device=device, seed=seed)
    return model
