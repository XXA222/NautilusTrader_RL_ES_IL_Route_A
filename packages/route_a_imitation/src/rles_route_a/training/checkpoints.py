"""Restorable IQL/TD3+BC optimizer checkpoint without pickle.

This is a train-only checkpoint, NEVER a deployment bundle or permission to
access research data. Native risk/economic execution remains the host's job.

The JSON codec encodes nested optimizer structures with explicit type tags;
PyTorch tensors are saved in a bounded, SHA-verified NPZ. Model checkpoints are
bound to one Fold/Train source, execution recipe, frozen encoder and runtime.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory
import json
import os
import sys
from typing import Any, cast
import numpy as np
import torch
from rles_route_a import BASELINE_COMMIT
from rles_route_a.training.iql_trainer import IQLTrainer, IQLHyper
from rles_route_a.training.td3bc_trainer import TD3BCTrainer, TD3BCHyper
from rles_route_a.export.archive_safety import validate_weight_archive


def _pack(obj: Any, tensors: dict[str,np.ndarray]) -> Any:
    if isinstance(obj, torch.Tensor):
        key = f"t{len(tensors):08d}"
        arr = obj.detach().cpu().numpy()
        if arr.dtype.kind not in "fiub" or not np.isfinite(arr).all():
            raise ValueError("invalid optimizer/weight checkpoint tensor")
        tensors[key] = arr
        return {"t": key}
    if isinstance(obj, dict):
        return {"d": [[_pack(k,tensors), _pack(v,tensors)] for k,v in obj.items()]}
    if isinstance(obj, tuple):
        return {"tuple": [_pack(v,tensors) for v in obj]}
    if isinstance(obj, list):
        return {"list": [_pack(v,tensors) for v in obj]}
    if obj is None or isinstance(obj, (str,int,bool)):
        return {"scalar": obj}
    if isinstance(obj,float) and isfinite(obj):
        return {"scalar": obj}
    raise TypeError(f"checkpoint refuses arbitrary serializer type: {type(obj).__name__}")


def _unpack(node: Any, arrays: dict[str,torch.Tensor]) -> Any:
    if not isinstance(node, dict) or len(node) != 1:
        raise ValueError("invalid checkpoint tagged-value schema")
    tag,value = next(iter(node.items()))
    if tag == "t":
        if not isinstance(value, str) or value not in arrays:
            raise ValueError("missing checkpoint tensor")
        return arrays[value].clone()
    if tag == "d":
        if not isinstance(value,list):
            raise ValueError("invalid checkpoint dict")
        values = [(_unpack(k,arrays), _unpack(v,arrays)) for k,v in value]
        if len({k for k,v in values}) != len(values):
            raise ValueError("duplicate checkpoint dictionary key")
        return dict(values)
    if tag in ("list","tuple"):
        if not isinstance(value,list):
            raise ValueError("invalid checkpoint list")
        decoded = [_unpack(x,arrays) for x in value]
        return decoded if tag=="list" else tuple(decoded)
    if tag == "scalar":
        if value is None or isinstance(value,(str,int,bool)) or (
                type(value) is float and isfinite(value)):
            return value
    raise ValueError("unsupported checkpoint tagged value")


def _identify(learner: IQLTrainer | TD3BCTrainer) -> tuple[str,int,dict[str,Any]]:
    if isinstance(learner,IQLTrainer):
        return "iql",learner.state_dim,asdict(learner.cfg)
    if isinstance(learner,TD3BCTrainer):
        return "td3bc",learner.state_dim,asdict(learner.cfg)
    raise TypeError("only IQL/TD3+BC training checkpoints are supported")


def save_training_checkpoint(learner: IQLTrainer | TD3BCTrainer, path: str | Path, *,
                             registered_fold_id: str, execution_recipe: str,
                             dataset_identity: str, frozen_encoder_identity: str,
                             reward_scale: float, update_index: int,
                             sampler_state: torch.Tensor | None = None) -> Path:
    if not all(type(v) is str and v for v in
               (registered_fold_id,execution_recipe,dataset_identity,frozen_encoder_identity)):
        raise ValueError("checkpoint must bind registered native training inputs")
    if type(update_index) is not int or update_index < 0:
        raise ValueError("invalid update index")
    if not isinstance(reward_scale,(float,int)) or isinstance(reward_scale,bool) or not isfinite(reward_scale) or reward_scale<=0:
        raise ValueError("invalid checkpoint numerical reward scale")
    kind,state_dim,hyper=_identify(learner)
    if sampler_state is not None and (sampler_state.ndim != 1 or sampler_state.dtype != torch.uint8):
        raise ValueError("sampler state must be torch uint8 generator state")
    snapshot: dict[str, Any]
    if isinstance(learner, IQLTrainer):
        snapshot = {
            "q1": learner.q1.state_dict(), "q2":learner.q2.state_dict(),
            "value":learner.value.state_dict(), "actor":learner.actor.state_dict(),
            "q1_target":learner.q1_target.state_dict(),"q2_target":learner.q2_target.state_dict(),
            "q_opt":learner.q_opt.state_dict(),"v_opt":learner.v_opt.state_dict(),
            "actor_opt":learner.actor_opt.state_dict(),
        }
    elif isinstance(learner, TD3BCTrainer):
        snapshot = {
            "actor":learner.actor.state_dict(),"q1":learner.q1.state_dict(),
            "q2":learner.q2.state_dict(),"actor_target":learner.actor_target.state_dict(),
            "q1_target":learner.q1_target.state_dict(),"q2_target":learner.q2_target.state_dict(),
            "actor_opt":learner.actor_opt.state_dict(),"critic_opt":learner.critic_opt.state_dict(),
            "steps":learner.steps,"noise_seed":learner.noise_seed,
            "noise_generator_state":learner._noise_generator.get_state(),
        }
    else:
        raise TypeError("only IQL/TD3+BC training checkpoints are supported")
    if sampler_state is not None:
        snapshot["sampler_state"] = sampler_state
    tensors:dict[str,np.ndarray]={}
    encoded=_pack(snapshot,tensors)
    dest=Path(path)
    if dest.exists():
        raise FileExistsError(dest)
    dest.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix="ra-train-",dir=dest.parent) as d:
        tmp=Path(d)
        np.savez_compressed(tmp/"state.npz", **cast(Any, tensors))
        payload=(tmp/"state.npz").read_bytes()
        manifest={"schema":"route_a_training_checkpoint_v1","baseline":BASELINE_COMMIT,
                  "algorithm":kind,"state_dim":state_dim,"config":hyper,
                  "training_identity":{"fold":registered_fold_id,"execution":execution_recipe,
                                       "dataset":dataset_identity,"encoder":frozen_encoder_identity},
                  "reward_scale":float(reward_scale),"update_index":update_index,
                  "python_minor":list(sys.version_info[:2]),"torch_version":torch.__version__,
                  "tensor_sha256":sha256(payload).hexdigest(),"tensor_names":sorted(tensors),
                  "model_optimizer_state":encoded,"scope":"TRAIN_ONLY_NOT_A_PROFITABILITY_CLAIM"}
        (tmp/"manifest.json").write_text(json.dumps(manifest,sort_keys=True,allow_nan=False),encoding="utf-8")
        os.rename(tmp,dest)
    return dest


def load_training_checkpoint(path: str | Path, *, expected_fold_id: str,
                             expected_execution_recipe: str,
                             expected_dataset_identity: str,
                             expected_frozen_encoder_identity: str,
                             expected_reward_scale: float, device: str="cpu") -> tuple[IQLTrainer|TD3BCTrainer, int, torch.Tensor | None]:
    manifest_path=Path(path)/"manifest.json"
    if manifest_path.stat().st_size > 4*1024*1024:
        raise ValueError("checkpoint manifest too large")
    m=json.loads(manifest_path.read_text(encoding="utf-8"))
    if (m.get("schema")!="route_a_training_checkpoint_v1" or m.get("baseline")!=BASELINE_COMMIT
            or m.get("scope")!="TRAIN_ONLY_NOT_A_PROFITABILITY_CLAIM"):
        raise ValueError("incompatible training checkpoint baseline/scope")
    if m.get("training_identity") != {"fold":expected_fold_id,"execution":expected_execution_recipe,
                                        "dataset":expected_dataset_identity,"encoder":expected_frozen_encoder_identity}:
        raise PermissionError("training checkpoint fold, data, encoder or execution provenance mismatch")
    if m.get("reward_scale") != expected_reward_scale:
        raise ValueError("training reward scale mismatch")
    if m.get("python_minor") != list(sys.version_info[:2]) or m.get("torch_version") != torch.__version__:
        raise RuntimeError("optimizer checkpoint requires exact runtime version match")
    kind=m.get("algorithm")
    width=m.get("state_dim")
    if type(width) is not int or not 1<=width<=8192 or kind not in ("iql","td3bc"):
        raise ValueError("unsupported learner checkpoint architecture")
    if kind == "iql":
        iql_cfg = IQLHyper(**m["config"])
        learner: IQLTrainer | TD3BCTrainer = IQLTrainer(width, iql_cfg, device=device)
    else:
        td3bc_cfg = TD3BCHyper(**m["config"])
        learner = TD3BCTrainer(width, td3bc_cfg, device=device)
    weights=Path(path)/"state.npz"
    if sha256(weights.read_bytes()).hexdigest()!=m.get("tensor_sha256"):
        raise ValueError("checkpoint tensor checksum mismatch")
    names=m.get("tensor_names")
    if not isinstance(names,list) or len(names)!=len(set(names)) or not all(isinstance(x,str) for x in names):
        raise ValueError("invalid checkpoint tensor inventory")
    validate_weight_archive(weights,{k:() for k in names})  # validates members and byte budget
    with np.load(weights,allow_pickle=False) as z:
        if set(z.files)!=set(names):
            raise ValueError("checkpoint tensor mismatch")
        arrays={}
        for k in names:
            arr=z[k]
            if arr.dtype.kind not in "fiub" or not np.isfinite(arr).all():
                raise ValueError("invalid training checkpoint tensor")
            arrays[k]=torch.as_tensor(arr.copy())
    state=_unpack(m["model_optimizer_state"],arrays)
    if isinstance(learner, IQLTrainer):
        for role in ("q1","q2","value","actor","q1_target","q2_target"):
            getattr(learner,role).load_state_dict(state[role],strict=True)
        for role in ("q_opt","v_opt","actor_opt"):
            getattr(learner,role).load_state_dict(state[role])
    else:
        for role in ("q1","q2","actor","q1_target","q2_target","actor_target"):
            getattr(learner,role).load_state_dict(state[role],strict=True)
        for role in ("actor_opt","critic_opt"):
            getattr(learner,role).load_state_dict(state[role])
        learner.steps=state["steps"]
        learner.noise_seed=state["noise_seed"]
        learner._noise_generator.set_state(state["noise_generator_state"])
    idx=m.get("update_index")
    if type(idx) is not int or idx<0:
        raise ValueError("checkpoint update index invalid")
    return learner,idx,state.get("sampler_state")
