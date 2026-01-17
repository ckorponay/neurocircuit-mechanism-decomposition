import torch
from torch import nn
from neurocircuit.models.ssm_head import LinearSSM
from neurocircuit.models.transformer_head import TransformerPropagationHead

class NeurocircuitMechDecomp(nn.Module):
    """Coupled Transformer–SSM scaffold (Strategy 1 default)."""
    def __init__(self, n_regions: int, d_model: int = 128, n_lags: int = 13):
        super().__init__()
        self.ssm = LinearSSM(n_regions=n_regions)
        self.transformer = TransformerPropagationHead(d_model=d_model, n_lags=n_lags)

    def forward(self, y: torch.Tensor, anat_mask: torch.Tensor, max_lag: int = 13) -> dict:
        ssm_out = self.ssm(y, u=None)
        x_hat = ssm_out["x_hat"]
        tr_out = self.transformer(x_hat, anat_mask=anat_mask, max_lag=max_lag)
        return {"ssm": ssm_out, "transformer": tr_out}
