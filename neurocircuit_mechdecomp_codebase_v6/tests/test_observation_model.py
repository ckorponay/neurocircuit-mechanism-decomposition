import torch

from neurocircuit.models.observation_model import (
    causal_hrf_convolution,
    delayed_systemic_component,
    HemodynamicObservationModel,
)


def test_causal_hrf_convolution_impulse():
    x = torch.zeros(1, 1, 6)
    x[0, 0, 1] = 1.0
    h = torch.tensor([[1.0, 2.0, 3.0]])
    y = causal_hrf_convolution(x, h)
    assert torch.allclose(y[0, 0], torch.tensor([0., 1., 2., 3., 0., 0.]))


def test_rapidtide_delay_is_fractional_systemic_shift():
    s = torch.tensor([[1., 0., 0., 0., 0.]])
    delay = torch.tensor([[1.0]])
    amp = torch.tensor([[2.0]])
    y = delayed_systemic_component(s, delay, amp, tr_seconds=1.0)
    assert torch.allclose(y[0, 0], torch.tensor([0., 2., 0., 0., 0.]))


def test_observation_decomposes_neural_and_systemic():
    x = torch.zeros(1, 1, 5)
    x[0, 0, 0] = 1.0
    h = torch.tensor([[1.0]])
    s = torch.tensor([[0., 1., 0., 0., 0.]])
    delay = torch.tensor([[0.]])
    amp = torch.tensor([[0.5]])
    m = HemodynamicObservationModel(include_systemic=True)
    out = m(
        x,
        hrf_kernel=h,
        tr_seconds=1.0,
        systemic_waveform=s,
        vascular_delay_seconds=delay,
        vascular_amplitude=amp,
    )
    assert torch.allclose(out["y_hat"], out["neural_bold"] + out["systemic_bold"])
