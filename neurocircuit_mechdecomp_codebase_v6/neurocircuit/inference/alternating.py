from __future__ import annotations

import torch

from neurocircuit.inference.map_state_estimator import infer_latent_states_map
from neurocircuit.models.model import NeurocircuitMechDecomp


def infer_latent_and_drive_alternating(
    model: NeurocircuitMechDecomp,
    y_observed: torch.Tensor,
    anat_mask: torch.Tensor,
    *,
    tr_seconds: float,
    hrf_kernel: torch.Tensor,
    hrf_gain: torch.Tensor | None = None,
    systemic_waveform: torch.Tensor | None = None,
    vascular_delay_seconds: torch.Tensor | None = None,
    vascular_amplitude: torch.Tensor | None = None,
    initial_x: torch.Tensor | None = None,
    n_outer_steps: int = 3,
    map_steps_first: int = 250,
    map_steps_refine: int = 100,
    map_lr: float = 5e-2,
    observation_weight: float = 1.0,
    dynamics_weight: float = 0.02,
    l2_weight: float = 1e-5,
    max_lag_seconds: float | None = None,
    return_pi: bool = False,
    dynamics_mask: torch.Tensor | None = None,
) -> dict:
    """
    Alternating measurement/circuit inference for a fixed trained model.

    1. Infer latent neural state from BOLD using fixed rsHRF + RAPIDTIDE inputs.
    2. Route the latent state with the Transformer to estimate drive u_hat(t).
    3. Re-infer latent state while enforcing x(t+1) ~= A x(t) + B u_hat(t).
    4. Repeat a small number of times.

    This makes B operational in the state equation instead of leaving the SSM
    autonomous (u=None), while keeping the hemodynamic parameters fixed for
    identifiability.
    """
    if n_outer_steps < 1:
        raise ValueError("n_outer_steps must be >= 1")

    dyn_mask = anat_mask if dynamics_mask is None else dynamics_mask
    A_d, B_d = model.dynamics_matrices(
        tr_seconds=tr_seconds,
        dynamics_mask=dyn_mask,
    )

    x_init = initial_x
    drive = None
    history = []
    current = None

    for outer in range(n_outer_steps):
        n_steps = map_steps_first if outer == 0 else map_steps_refine
        current = infer_latent_states_map(
            y_observed,
            A_d=A_d,
            B_d=B_d if drive is not None else None,
            u=drive,
            observation_model=model.observation,
            hrf_kernel=hrf_kernel,
            tr_seconds=tr_seconds,
            hrf_gain=hrf_gain,
            systemic_waveform=systemic_waveform,
            vascular_delay_seconds=vascular_delay_seconds,
            vascular_amplitude=vascular_amplitude,
            initial_x=x_init,
            n_steps=n_steps,
            lr=map_lr,
            observation_weight=observation_weight,
            dynamics_weight=dynamics_weight,
            l2_weight=l2_weight,
        )
        x_hat = current["x_hat"]
        route = model.route_latent(
            x_hat,
            anat_mask,
            tr_seconds=tr_seconds,
            max_lag_seconds=max_lag_seconds,
            return_pi=return_pi and outer == n_outer_steps - 1,
        )
        drive = route["drive"].detach()
        x_init = x_hat
        history.append(
            {
                "outer_step": outer + 1,
                "map_history": current["history"],
            }
        )

    final_route = model.route_latent(
        current["x_hat"],
        anat_mask,
        tr_seconds=tr_seconds,
        max_lag_seconds=max_lag_seconds,
        return_pi=return_pi,
    )
    dyn_pred = model.ssm.transition_prediction(
        current["x_hat"],
        u=final_route["drive"],
        tr_seconds=tr_seconds,
        dynamics_mask=dyn_mask,
    )
    dyn_resid = current["x_hat"][:, :, 1:] - dyn_pred

    return {
        "x_hat": current["x_hat"],
        "transformer": final_route,
        "observation": current["observation"],
        "A_d": A_d.detach(),
        "B_d": B_d.detach(),
        "dynamics_residual": dyn_resid.detach(),
        "history": history,
    }
