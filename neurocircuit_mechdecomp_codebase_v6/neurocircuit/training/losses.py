from __future__ import annotations

import torch

from neurocircuit.models.model import NeurocircuitMechDecomp


def latent_dynamics_loss(
    model: NeurocircuitMechDecomp,
    x_hat: torch.Tensor,
    anat_mask: torch.Tensor,
    *,
    tr_seconds: float,
    max_lag_seconds: float | None = None,
    dynamics_mask: torch.Tensor | None = None,
    return_details: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict]:
    """
    M-step objective for fixed inferred latent states.

    Transformer drive is fed into the SSM, so both routing parameters and B
    receive gradients through the transition residual.
    """
    route = model.route_latent(
        x_hat,
        anat_mask,
        tr_seconds=tr_seconds,
        max_lag_seconds=max_lag_seconds,
        return_pi=False,
    )
    pred = model.ssm.transition_prediction(
        x_hat,
        u=route["drive"],
        tr_seconds=tr_seconds,
        dynamics_mask=anat_mask if dynamics_mask is None else dynamics_mask,
    )
    resid = x_hat[:, :, 1:] - pred
    loss = (resid ** 2).mean()
    if not return_details:
        return loss
    return loss, {"route": route, "prediction": pred, "residual": resid}


def routing_entropy_penalty(edge_mass: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Optional weak penalty discouraging diffuse incoming routing per target."""
    # edge_mass [B,R_src,R_tgt]
    p = edge_mass / edge_mass.sum(dim=1, keepdim=True).clamp_min(eps)
    ent = -(p.clamp_min(eps) * p.clamp_min(eps).log()).sum(dim=1)
    return ent.mean()
