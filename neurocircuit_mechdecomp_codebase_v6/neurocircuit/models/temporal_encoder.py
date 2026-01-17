import torch
from torch import nn

class TemporalEncoder(nn.Module):
    """Lightweight temporal encoder (1D convs) over time per region.

    Input:  y (B, R, T)
    Output: tokens (B, R, T, D)
    """
    def __init__(self, d_model: int = 128, kernel_size: int = 5):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(1, d_model, kernel_size=kernel_size, padding=padding)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        b, r, t = y.shape
        x = y.reshape(b * r, 1, t)
        x = self.conv(x)              # (b*r, D, t)
        x = x.transpose(1, 2)         # (b*r, t, D)
        x = self.norm(x)
        return x.reshape(b, r, t, -1)
