import torch

from neurocircuit.models.model import NeurocircuitMechDecomp
from neurocircuit.models.transformer_head import n_lags_for_seconds
from neurocircuit.models.ssm_head import exact_zoh_discretize


def portable_model(R=6):
    return NeurocircuitMechDecomp(
        n_regions=R,
        d_model=16,
        n_lags=None,
        lag_embedding_mode="seconds",
        max_lag_seconds=12.0,
        ssm_parameterization="continuous",
        ssm_initial_state="zero",
    )


def test_physical_lag_span_by_dataset_tr():
    assert n_lags_for_seconds(12.0, 0.720) == 17
    assert n_lags_for_seconds(12.0, 0.735) == 17
    assert n_lags_for_seconds(12.0, 0.800) == 16
    assert n_lags_for_seconds(12.0, 2.000) == 7


def test_portable_forward_uses_dataset_specific_number_of_samples():
    B, R, T = 1, 6, 30
    y = torch.randn(B, R, T)
    mask = torch.ones(R, R, dtype=torch.bool)
    model = portable_model(R).eval()

    expected = {0.720: 17, 0.735: 17, 0.800: 16, 2.000: 7}
    with torch.no_grad():
        for tr, L in expected.items():
            out = model(y, mask, tr_seconds=tr, max_lag_seconds=12.0)
            assert out["transformer"]["pi"].shape == (B, T, R, R, L)
            lag_s = out["transformer"]["lag_seconds"]
            assert lag_s[-1].item() <= 12.0 + 1e-6
            assert lag_s.shape[0] == L


def test_continuous_parameters_are_tr_invariant_but_discrete_steps_change():
    A_c = torch.tensor([[-0.2, 0.0], [0.0, -0.5]], dtype=torch.float64)
    B_c = torch.eye(2, dtype=torch.float64)
    A1, _ = exact_zoh_discretize(A_c, B_c, 0.72)
    A2, _ = exact_zoh_discretize(A_c, B_c, 2.0)
    assert not torch.allclose(A1, A2)

    # Continuous eigenvalues recovered from the discrete eigenvalues are the same.
    e1 = torch.log(torch.linalg.eigvals(A1).to(torch.complex128)) / 0.72
    e2 = torch.log(torch.linalg.eigvals(A2).to(torch.complex128)) / 2.0
    assert torch.allclose(torch.sort(e1.real).values, torch.sort(e2.real).values, atol=1e-8)


def test_continuous_ssm_exports_physical_stability():
    B, R, T = 1, 5, 20
    y = torch.randn(B, R, T)
    mask = torch.ones(R, R, dtype=torch.bool)
    model = portable_model(R).eval()
    with torch.no_grad():
        out = model(y, mask, tr_seconds=0.8)
    ssm = out["ssm"]
    assert "A_c" in ssm and "B_c" in ssm
    assert "stability_margin_per_second" in ssm
    assert ssm["stability_margin_per_second"].item() > 0
