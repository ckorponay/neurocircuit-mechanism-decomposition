from __future__ import annotations

import torch
from torch import nn

from neurocircuit.models.ssm_head import LinearSSM
from neurocircuit.models.transformer_head import TransformerPropagationHead
from neurocircuit.models.observation_model import HemodynamicObservationModel


class NeurocircuitMechDecomp(nn.Module):
    """
    Neurocircuit mechanism decomposition model.

    Legacy mode reproduces the original scaffold. Recommended production mode
    uses physical-time continuous SSM dynamics, edge-conditioned sparse routing,
    corrected lag ordering, and fixed external hemodynamic measurement inputs.
    """

    def __init__(
        self,
        n_regions: int,
        d_model: int = 128,
        n_lags: int | None = 13,
        lag_embedding_mode: str = "index",
        max_lag_seconds: float = 12.0,
        attention_mode: str = "legacy_additive",
        lag_interaction_dim: int = 16,
        legacy_oldest_first: bool = True,
        ssm_parameterization: str = "discrete",
        ssm_initial_state: str = "first_observation",
        continuous_stability_mode: str = "projected",
        discretization_method: str = "solve",
        include_systemic_observation: bool = True,
    ):
        super().__init__()
        self.ssm = LinearSSM(
            n_regions=n_regions,
            parameterization=ssm_parameterization,
            initial_state=ssm_initial_state,
            continuous_stability_mode=continuous_stability_mode,
            discretization_method=discretization_method,
        )
        self.transformer = TransformerPropagationHead(
            d_model=d_model,
            n_lags=n_lags,
            lag_embedding_mode=lag_embedding_mode,
            max_lag_seconds=max_lag_seconds,
            attention_mode=attention_mode,
            interaction_dim=lag_interaction_dim,
            legacy_oldest_first=legacy_oldest_first,
            dropout=0.0 if attention_mode == "edge_conditioned_sparse" else 0.1,
        )
        self.observation = HemodynamicObservationModel(
            include_systemic=include_systemic_observation
        )

    def route_latent(
        self,
        x_neural: torch.Tensor,
        anat_mask: torch.Tensor,
        *,
        max_lag: int | None = None,
        tr_seconds: float | None = None,
        max_lag_seconds: float | None = None,
        return_pi: bool | None = None,
    ) -> dict:
        return self.transformer(
            x_neural,
            anat_mask=anat_mask,
            max_lag=max_lag,
            tr_seconds=tr_seconds,
            max_lag_seconds=max_lag_seconds,
            return_pi=return_pi,
        )

    def observe_latent(
        self,
        x_neural: torch.Tensor,
        *,
        hrf_kernel: torch.Tensor,
        tr_seconds: float,
        hrf_gain: torch.Tensor | None = None,
        systemic_waveform: torch.Tensor | None = None,
        vascular_delay_seconds: torch.Tensor | None = None,
        vascular_amplitude: torch.Tensor | None = None,
        y_observed: torch.Tensor | None = None,
    ) -> dict:
        return self.observation(
            x_neural,
            hrf_kernel=hrf_kernel,
            tr_seconds=tr_seconds,
            hrf_gain=hrf_gain,
            systemic_waveform=systemic_waveform,
            vascular_delay_seconds=vascular_delay_seconds,
            vascular_amplitude=vascular_amplitude,
            y_observed=y_observed,
        )

    def dynamics_matrices(
        self,
        *,
        tr_seconds: float,
        dynamics_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.ssm.step_matrices(
            tr_seconds,
            dynamics_mask=dynamics_mask,
        )

    def forward_from_latent(
        self,
        x_neural: torch.Tensor,
        anat_mask: torch.Tensor,
        *,
        tr_seconds: float,
        hrf_kernel: torch.Tensor | None = None,
        hrf_gain: torch.Tensor | None = None,
        systemic_waveform: torch.Tensor | None = None,
        vascular_delay_seconds: torch.Tensor | None = None,
        vascular_amplitude: torch.Tensor | None = None,
        y_observed: torch.Tensor | None = None,
        max_lag: int | None = None,
        max_lag_seconds: float | None = None,
        return_pi: bool | None = None,
        dynamics_mask: torch.Tensor | None = None,
    ) -> dict:
        tr_out = self.route_latent(
            x_neural,
            anat_mask,
            max_lag=max_lag,
            tr_seconds=tr_seconds,
            max_lag_seconds=max_lag_seconds,
            return_pi=return_pi,
        )
        A_d, B_d = self.dynamics_matrices(
            tr_seconds=tr_seconds,
            dynamics_mask=anat_mask if dynamics_mask is None else dynamics_mask,
        )
        dyn_pred = self.ssm.transition_prediction(
            x_neural,
            u=tr_out["drive"],
            tr_seconds=tr_seconds,
            dynamics_mask=anat_mask if dynamics_mask is None else dynamics_mask,
        )
        out = {
            "transformer": tr_out,
            "x_hat": x_neural,
            "A_d": A_d,
            "B_d": B_d,
            "dynamics_prediction": dyn_pred,
        }
        if hrf_kernel is not None:
            out["observation"] = self.observe_latent(
                x_neural,
                hrf_kernel=hrf_kernel,
                tr_seconds=tr_seconds,
                hrf_gain=hrf_gain,
                systemic_waveform=systemic_waveform,
                vascular_delay_seconds=vascular_delay_seconds,
                vascular_amplitude=vascular_amplitude,
                y_observed=y_observed,
            )
        return out

    def forward(
        self,
        y: torch.Tensor,
        anat_mask: torch.Tensor,
        max_lag: int | None = None,
        tr_seconds: float | None = None,
        max_lag_seconds: float | None = None,
    ) -> dict:
        """Backward-compatible scaffold forward for old smoke tests/checkpoints."""
        ssm_out = self.ssm(
            y,
            u=None,
            tr_seconds=tr_seconds,
            dynamics_mask=None,  # preserve old behavior in legacy forward
        )
        x_hat = ssm_out["x_hat"]
        tr_out = self.route_latent(
            x_hat,
            anat_mask,
            max_lag=max_lag,
            tr_seconds=tr_seconds,
            max_lag_seconds=max_lag_seconds,
            return_pi=True,
        )
        return {"ssm": ssm_out, "transformer": tr_out}
