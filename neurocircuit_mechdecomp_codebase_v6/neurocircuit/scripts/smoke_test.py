import torch
from neurocircuit.models.model import NeurocircuitMechDecomp

def main():
    B, R, T = 2, 8, 100
    y = torch.randn(B, R, T)
    anat_mask = torch.ones(R, R, dtype=torch.bool)
    model = NeurocircuitMechDecomp(n_regions=R, d_model=64, n_lags=5)
    out = model(y, anat_mask=anat_mask, max_lag=5)
    assert out["transformer"]["pi"].shape == (B, T, R, R, 5)
    assert out["transformer"]["drive"].shape == (B, R, T)
    print("✅ smoke_test passed")

if __name__ == "__main__":
    main()
