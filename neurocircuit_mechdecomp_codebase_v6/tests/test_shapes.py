import torch
from neurocircuit.models.model import NeurocircuitMechDecomp

def test_forward_shapes():
    B, R, T = 2, 12, 60
    y = torch.randn(B, R, T)
    anat_mask = torch.ones(R, R, dtype=torch.bool)
    model = NeurocircuitMechDecomp(n_regions=R, d_model=32, n_lags=7)
    out = model(y, anat_mask=anat_mask, max_lag=7)
    assert out["transformer"]["pi"].shape == (B, T, R, R, 7)
    assert out["transformer"]["drive"].shape == (B, R, T)
