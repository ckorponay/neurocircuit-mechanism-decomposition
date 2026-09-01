import math
import torch
from torch import nn


class RelativeLagEmbedding(nn.Module):
    """
    Learned relative-lag representation.

    Modes
    -----
    index:
        Backward-compatible lookup table indexed by sample lag.
    seconds:
        Continuous learned embedding of physical lag in seconds. This allows the
        same trained weights to be evaluated at different TRs (e.g. HCP/UKB/PDC/
        CAN-BIND) without pretending that lag index has the same meaning.
    """

    def __init__(
        self,
        n_lags: int | None,
        dim: int,
        mode: str = "index",
        max_lag_seconds: float = 12.0,
    ):
        super().__init__()
        if mode not in {"index", "seconds"}:
            raise ValueError("mode must be 'index' or 'seconds'")
        self.mode = mode
        self.max_lag_seconds = float(max_lag_seconds)

        if mode == "index":
            if n_lags is None or n_lags < 1:
                raise ValueError("n_lags must be provided for index mode")
            self.emb = nn.Embedding(n_lags, dim)
            self.net = None
        else:
            self.emb = None
            # Continuous learned relative-lag embedding. Fourier-like input
            # features preserve smooth physical-time structure while remaining
            # learnable.
            self.net = nn.Sequential(
                nn.Linear(5, dim),
                nn.Tanh(),
                nn.Linear(dim, dim),
            )

    def forward(self, lag: torch.Tensor) -> torch.Tensor:
        if self.mode == "index":
            return self.emb(lag.long())

        lag_s = lag.to(dtype=torch.float32)
        scale = max(self.max_lag_seconds, 1e-6)
        x = lag_s / scale
        feats = torch.stack(
            [
                x,
                torch.sin(math.pi * x),
                torch.cos(math.pi * x),
                torch.sin(2.0 * math.pi * x),
                torch.cos(2.0 * math.pi * x),
            ],
            dim=-1,
        )
        return self.net(feats)
