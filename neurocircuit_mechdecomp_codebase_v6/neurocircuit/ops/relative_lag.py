import torch
from torch import nn

class RelativeLagEmbedding(nn.Module):
    """Learned relative lag embeddings for temporal attention."""
    def __init__(self, n_lags: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(n_lags, dim)

    def forward(self, lag_idx: torch.Tensor) -> torch.Tensor:
        return self.emb(lag_idx)
