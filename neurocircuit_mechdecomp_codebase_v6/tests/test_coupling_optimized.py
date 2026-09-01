import torch

from neurocircuit.models.model import NeurocircuitMechDecomp
from neurocircuit.training.losses import latent_dynamics_loss


def test_dynamics_loss_backprops_to_B_and_routing():
    torch.manual_seed(1)
    B, R, T = 1, 4, 12
    model = NeurocircuitMechDecomp(
        n_regions=R,
        d_model=8,
        n_lags=None,
        lag_embedding_mode="seconds",
        max_lag_seconds=2.0,
        attention_mode="edge_conditioned_sparse",
        lag_interaction_dim=4,
        ssm_parameterization="continuous",
        continuous_stability_mode="diagonal_dominant",
    )
    x = torch.randn(B, R, T)
    mask = torch.ones(R, R, dtype=torch.bool)
    loss = latent_dynamics_loss(
        model,
        x,
        mask,
        tr_seconds=1.0,
        max_lag_seconds=2.0,
    )
    loss.backward()
    assert model.ssm.U.grad is not None
    assert model.ssm.U.grad.abs().sum().item() > 0
    assert model.transformer.attention.edge_k.weight.grad is not None
    assert model.transformer.attention.edge_k.weight.grad.abs().sum().item() > 0
