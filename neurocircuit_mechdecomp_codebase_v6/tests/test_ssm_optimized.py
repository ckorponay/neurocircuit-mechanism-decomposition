import torch

from neurocircuit.models.ssm_head import LinearSSM, exact_zoh_discretize


def test_diagonal_dominant_continuous_A_is_stable_and_graph_masked():
    R = 5
    ssm = LinearSSM(
        R,
        parameterization="continuous",
        continuous_stability_mode="diagonal_dominant",
    )
    mask = torch.zeros(R, R, dtype=torch.bool)
    mask[0, 1] = True  # source 0 -> target 1
    mask[2, 3] = True
    A = ssm.effective_A_continuous(mask)
    eig = torch.linalg.eigvals(A)
    assert eig.real.max().item() < 0
    # A row target, column source.
    assert A[1, 0] != 0
    assert A[0, 1] == 0


def test_fast_and_augmented_discretization_agree():
    A = torch.tensor([[-0.5, 0.1], [0.0, -0.3]], dtype=torch.float64)
    B = torch.tensor([[1.0, 0.0], [0.2, 1.0]], dtype=torch.float64)
    a1, b1 = exact_zoh_discretize(A, B, 0.8, method="solve")
    a2, b2 = exact_zoh_discretize(A, B, 0.8, method="augmented")
    assert torch.allclose(a1, a2, atol=1e-10, rtol=1e-10)
    assert torch.allclose(b1, b2, atol=1e-10, rtol=1e-10)
