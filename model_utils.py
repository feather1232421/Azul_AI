from pathlib import Path

import torch

from azul_net import AzulNet
from azul_transformer import AzulTransformer


DEFAULT_ACTION_DIM = 180
DEFAULT_OBS_DIM = 567


def get_model_kwargs(model_type, action_dim=DEFAULT_ACTION_DIM, obs_dim=DEFAULT_OBS_DIM):
    if model_type == "mlp":
        return {
            "obs_dim": obs_dim,
            "action_dim": action_dim,
        }
    if model_type == "transformer":
        return {
            "action_dim": action_dim,
        }
    raise ValueError(f"Unsupported model_type: {model_type}")


def build_model(model_type="transformer", action_dim=DEFAULT_ACTION_DIM, obs_dim=DEFAULT_OBS_DIM):
    if model_type == "mlp":
        return AzulNet(**get_model_kwargs(model_type, action_dim=action_dim, obs_dim=obs_dim))
    if model_type == "transformer":
        return AzulTransformer(**get_model_kwargs(model_type, action_dim=action_dim, obs_dim=obs_dim))
    raise ValueError(f"Unsupported model_type: {model_type}")


def unwrap_checkpoint_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def infer_model_type_from_state_dict(state_dict):
    keys = set(state_dict.keys())
    if "factory_emb.weight" in keys or "transformer.layers.0.self_attn.in_proj_weight" in keys:
        return "transformer"
    if "trunk.0.weight" in keys or "policy_head.weight" in keys:
        return "mlp"
    raise ValueError("Could not infer model type from state_dict keys.")


def infer_model_type_from_checkpoint(checkpoint):
    if isinstance(checkpoint, dict) and checkpoint.get("model_type"):
        return checkpoint["model_type"]
    return infer_model_type_from_state_dict(unwrap_checkpoint_state_dict(checkpoint))


def load_checkpoint(model_path, device):
    return torch.load(model_path, map_location=device)


def load_model(model_path, device, model_type=None, action_dim=DEFAULT_ACTION_DIM, obs_dim=DEFAULT_OBS_DIM):
    checkpoint = load_checkpoint(model_path, device)
    state_dict = unwrap_checkpoint_state_dict(checkpoint)
    resolved_model_type = model_type or infer_model_type_from_checkpoint(checkpoint)
    model = build_model(
        model_type=resolved_model_type,
        action_dim=action_dim,
        obs_dim=obs_dim,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, checkpoint, resolved_model_type


def build_checkpoint_payload(model, optimizer=None, epoch=None, extra=None):
    payload = {
        "model": model.state_dict(),
        "model_type": infer_model_type_from_state_dict(model.state_dict()),
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if epoch is not None:
        payload["epoch"] = epoch
    if extra:
        payload.update(extra)
    return payload


def describe_checkpoint(model_path, device=None):
    if device is None:
        device = torch.device("cpu")
    checkpoint = load_checkpoint(Path(model_path), device)
    return {
        "model_type": infer_model_type_from_checkpoint(checkpoint),
        "has_optimizer": isinstance(checkpoint, dict) and "optimizer" in checkpoint,
        "has_wrapped_model": isinstance(checkpoint, dict) and "model" in checkpoint,
    }
