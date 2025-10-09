import torch
import torch.nn.functional as F

from .model import (DynamicsConfig, brownian_bridge_nll, brownian_inc_nll,
                    ou_inc_nll)


def masked_cross_entropy(logits: torch.Tensor, targets: torch.Tensor, mask_idx: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over masked positions only."""
    B, L, V = logits.shape
    logits = logits.reshape(-1, V)
    targets = targets.reshape(-1)
    mask_flat = mask_idx.reshape(-1)
    if mask_flat.sum() == 0:
        return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
    logits_m = logits[mask_flat]
    targets_m = targets[mask_flat]
    return F.cross_entropy(logits_m, targets_m)


def increment_nll(z_s, z_t, dt, dyn: DynamicsConfig):
    if dyn.process == "brownian":
        return brownian_inc_nll(z_s, z_t, dt, dyn.sigma).mean()
    elif dyn.process == "ou":
        return ou_inc_nll(z_s, z_t, dt, dyn.sigma, dyn.kappa).mean()
    else:
        raise ValueError(f"Unknown process: {dyn.process}")


def bridge_nll(z0, zu, zT, u, T, dyn: DynamicsConfig):
    if dyn.process == "brownian":
        return brownian_bridge_nll(z0, zu, zT, u, T, dyn.sigma).mean()
    else:
        raise NotImplementedError("OU bridge NLL not implemented in this minimal release.")

