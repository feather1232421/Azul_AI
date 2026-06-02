from pathlib import Path

import torch

from azul_net import AzulNet
from azul_transformer import AzulTransformer
from config import ACTION_DIM, TRANSFORMER_OBS_DIM


DEFAULT_ACTION_DIM = ACTION_DIM
DEFAULT_OBS_DIM = TRANSFORMER_OBS_DIM


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


def load_state_dict_partial(model, state_dict):
    model_state = model.state_dict()
    matched_state = {}
    skipped = []

    for key, tensor in state_dict.items():
        if key not in model_state:
            skipped.append((key, "missing_key"))
            continue
        if model_state[key].shape != tensor.shape:
            skipped.append((key, f"shape_mismatch {tuple(tensor.shape)} -> {tuple(model_state[key].shape)}"))
            continue
        matched_state[key] = tensor

    missing = [key for key in model_state.keys() if key not in matched_state]
    model_state.update(matched_state)
    model.load_state_dict(model_state)
    return {
        "loaded_keys": sorted(matched_state.keys()),
        "skipped": skipped,
        "missing": missing,
    }


def load_model(
    model_path,
    device,
    model_type=None,
    action_dim=DEFAULT_ACTION_DIM,
    obs_dim=DEFAULT_OBS_DIM,
    allow_partial_load=False,
):
    checkpoint = load_checkpoint(model_path, device)
    state_dict = unwrap_checkpoint_state_dict(checkpoint)
    resolved_model_type = model_type or infer_model_type_from_checkpoint(checkpoint)
    model = build_model(
        model_type=resolved_model_type,
        action_dim=action_dim,
        obs_dim=obs_dim,
    )

    partial_load_info = None
    if allow_partial_load:
        partial_load_info = load_state_dict_partial(model, state_dict)
        print(
            "Partial load summary:",
            {
                "loaded": len(partial_load_info["loaded_keys"]),
                "skipped": len(partial_load_info["skipped"]),
                "missing": len(partial_load_info["missing"]),
            },
        )
    else:
        model.load_state_dict(state_dict)

    model.to(device)
    model.eval()
    return model, checkpoint, resolved_model_type, partial_load_info


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
