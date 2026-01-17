import torch
from torch import nn

class LinearSSM(nn.Module):
    """Simplified linear SSM scaffold.

    x_{t+1} = A x_t + B u_t + w_t
    Observation HRF convolution is left as a TODO in this scaffold.
    """
    def __init__(self, n_regions: int, rank_B: int = 3):
        super().__init__()
        self.n = n_regions
        self.A = nn.Parameter(0.01 * torch.randn(n_regions, n_regions))
        self.U = nn.Parameter(0.01 * torch.randn(n_regions, rank_B))
        self.V = nn.Parameter(0.01 * torch.randn(n_regions, rank_B))

    def stability_project_(self, rho_max: float = 0.98):
        with torch.no_grad():
            rho = torch.linalg.eigvals(self.A).abs().max().real
            if rho > rho_max:
                self.A.mul_(rho_max / (rho + 1e-8))

    def forward(self, y: torch.Tensor, u: torch.Tensor | None = None) -> dict:
        Bsz, R, T = y.shape
        self.stability_project_()
        Bmat = self.U @ self.V.T
        x = torch.zeros((Bsz, R, T), device=y.device, dtype=y.dtype)
        x[:, :, 0] = y[:, :, 0]
        for t in range(T - 1):
            inp = torch.zeros((Bsz, R), device=y.device, dtype=y.dtype) if u is None else u[:, :, t]
            x[:, :, t + 1] = (x[:, :, t] @ self.A.T) + (inp @ Bmat.T)
        return {"x_hat": x, "A": self.A, "B": Bmat}
