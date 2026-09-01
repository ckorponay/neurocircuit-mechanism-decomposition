import torch

from neurocircuit.inference.map_state_estimator import infer_latent_states_map
from neurocircuit.models.observation_model import (
    HemodynamicObservationModel,
    causal_hrf_convolution,
    delayed_systemic_component,
)


def test_map_state_estimator_recovers_synthetic_neural_trajectory():
    torch.manual_seed(0)
    B, R, T = 1, 1, 100
    a = 0.85
    A_d = torch.tensor([[a]])

    x = torch.zeros(B, R, T)
    u_noise = 0.30 * torch.randn(B, R, T)
    x[:, :, 0] = 1.0
    for t in range(T - 1):
        x[:, :, t + 1] = a * x[:, :, t] + u_noise[:, :, t]

    h = torch.tensor([[0.0, 0.2, 0.5, 0.3]])
    s = torch.sin(torch.arange(T, dtype=torch.float32) * 0.1).unsqueeze(0)
    delay = torch.tensor([[2.0]])
    amp = torch.tensor([[0.4]])
    systemic = delayed_systemic_component(s, delay, amp, tr_seconds=1.0)
    y = causal_hrf_convolution(x, h) + systemic + 0.02 * torch.randn_like(x)

    obs = HemodynamicObservationModel(include_systemic=True)
    fit = infer_latent_states_map(
        y,
        A_d=A_d,
        observation_model=obs,
        hrf_kernel=h,
        tr_seconds=1.0,
        systemic_waveform=s,
        vascular_delay_seconds=delay,
        vascular_amplitude=amp,
        n_steps=600,
        lr=0.05,
        dynamics_weight=0.02,
    )
    xhat = fit["x_hat"].flatten()
    corr = torch.corrcoef(torch.stack([x.flatten(), xhat]))[0, 1]
    assert corr > 0.95
