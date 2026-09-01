from __future__ import annotations

import torch

from neurocircuit.models.observation_model import HemodynamicObservationModel


def _dynamics_prediction(
    x: torch.Tensor,
    A_d: torch.Tensor,
    B_d: torch.Tensor | None = None,
    u: torch.Tensor | None = None,
) -> torch.Tensor:
    pred = x[:, :, :-1].transpose(1, 2) @ A_d.T
    pred = pred.transpose(1, 2)
    if u is not None:
        if B_d is None:
            raise ValueError("u supplied without B_d")
        inp = u[:, :, :-1].transpose(1, 2) @ B_d.T
        pred = pred + inp.transpose(1, 2)
    return pred


def infer_latent_states_map(
    y_observed: torch.Tensor,
    *,
    A_d: torch.Tensor,
    observation_model: HemodynamicObservationModel,
    hrf_kernel: torch.Tensor,
    tr_seconds: float,
    hrf_gain: torch.Tensor | None = None,
    systemic_waveform: torch.Tensor | None = None,
    vascular_delay_seconds: torch.Tensor | None = None,
    vascular_amplitude: torch.Tensor | None = None,
    B_d: torch.Tensor | None = None,
    u: torch.Tensor | None = None,
    initial_x: torch.Tensor | None = None,
    n_steps: int = 300,
    lr: float = 5e-2,
    observation_weight: float = 1.0,
    dynamics_weight: float = 0.02,
    l2_weight: float = 1e-5,
) -> dict:
    """
    MAP-like latent-state inference with FIXED hemodynamic parameters.

    This is the first non-blind HRF deconvolution path for NMD. It estimates x(t)
    by minimizing both measurement error and disagreement with the SSM dynamics.
    The rsHRF and RAPIDTIDE inputs are held fixed, preventing neural lag from
    freely trading against HRF or vascular timing during this inference step.

    For model training, use this as an E-step / alternating-optimization building
    block: infer x with fixed model/hemodynamic parameters, then update dynamics/
    routing parameters in a separate step.
    """
    if y_observed.ndim != 3:
        raise ValueError("y_observed must be [B,R,T]")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    # A practical warm start is vascular-residual BOLD. A separate rsHRF Wiener
    # deconvolution may also be supplied as initial_x, but is not treated as truth.
    if initial_x is None:
        if observation_model.include_systemic:
            with torch.no_grad():
                zero = torch.zeros_like(y_observed)
                sys = observation_model(
                    zero,
                    hrf_kernel=hrf_kernel,
                    tr_seconds=tr_seconds,
                    hrf_gain=hrf_gain,
                    systemic_waveform=systemic_waveform,
                    vascular_delay_seconds=vascular_delay_seconds,
                    vascular_amplitude=vascular_amplitude,
                )["systemic_bold"]
            x0 = y_observed - sys
        else:
            x0 = y_observed.clone()
    else:
        if initial_x.shape != y_observed.shape:
            raise ValueError("initial_x must match y_observed")
        x0 = initial_x

    x = torch.nn.Parameter(x0.detach().clone())
    opt = torch.optim.Adam([x], lr=lr)
    history = []

    A_fixed = A_d.detach()
    B_fixed = None if B_d is None else B_d.detach()

    for step in range(n_steps):
        opt.zero_grad()
        obs = observation_model(
            x,
            hrf_kernel=hrf_kernel,
            tr_seconds=tr_seconds,
            hrf_gain=hrf_gain,
            systemic_waveform=systemic_waveform,
            vascular_delay_seconds=vascular_delay_seconds,
            vascular_amplitude=vascular_amplitude,
            y_observed=y_observed,
        )
        loss_obs = (obs["residual"] ** 2).mean()

        pred_next = _dynamics_prediction(x, A_fixed, B_fixed, u)
        dyn_resid = x[:, :, 1:] - pred_next
        loss_dyn = (dyn_resid ** 2).mean()
        loss_l2 = (x ** 2).mean()

        loss = (
            observation_weight * loss_obs
            + dynamics_weight * loss_dyn
            + l2_weight * loss_l2
        )
        loss.backward()
        opt.step()

        if step == 0 or step == n_steps - 1 or (step + 1) % 50 == 0:
            history.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach()),
                    "observation_loss": float(loss_obs.detach()),
                    "dynamics_loss": float(loss_dyn.detach()),
                }
            )

    with torch.no_grad():
        final_obs = observation_model(
            x,
            hrf_kernel=hrf_kernel,
            tr_seconds=tr_seconds,
            hrf_gain=hrf_gain,
            systemic_waveform=systemic_waveform,
            vascular_delay_seconds=vascular_delay_seconds,
            vascular_amplitude=vascular_amplitude,
            y_observed=y_observed,
        )

    return {
        "x_hat": x.detach(),
        "observation": {k: v.detach() for k, v in final_obs.items()},
        "history": history,
    }
