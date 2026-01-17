import torch
from torch import nn
from einops import rearrange
from neurocircuit.ops.graph_mask import apply_graph_mask
from neurocircuit.ops.relative_lag import RelativeLagEmbedding

class FactorizedSpatialTemporalAttention(nn.Module):
    """Factorized routing attention producing π_{i,ℓ→j}."""
    def __init__(self, d_model: int, n_lags: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_lags = n_lags
        self.spatial_q = nn.Linear(d_model, d_model, bias=False)
        self.spatial_k = nn.Linear(d_model, d_model, bias=False)
        self.temporal_q = nn.Linear(d_model, d_model, bias=False)
        self.rel_lag = RelativeLagEmbedding(n_lags=n_lags, dim=d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, anat_mask: torch.Tensor, lag_idx: torch.Tensor) -> torch.Tensor:
        B, R, T, D = tokens.shape

        # Spatial logits S_{ij} at each time t: (B,T,R_tgt,R_src)
        q = rearrange(self.spatial_q(tokens), "b r t d -> b t r d")
        k = rearrange(self.spatial_k(tokens), "b r t d -> b t r d")
        spatial = torch.einsum("b t j d, b t i d -> b t j i", q, k) / (D ** 0.5)
        spatial = apply_graph_mask(spatial, anat_mask.T)  # (tgt=j, src=i)

        # Temporal logits T_ℓ: (B,T,L)
        q_t = self.temporal_q(tokens.mean(dim=1))  # mean over regions -> (B,T,D)
        lag_emb = self.rel_lag(lag_idx).to(tokens.device)  # (L,D)
        temporal = torch.einsum("b t d, l d -> b t l", q_t, lag_emb) / (D ** 0.5)

        # Combine additively and softmax over (src,lag) per target
        comb = spatial.unsqueeze(-1) + temporal.unsqueeze(-2).unsqueeze(-2)  # (B,T,R,R,L)
        comb_flat = comb.reshape(B, T, R, R * self.n_lags)
        pi_flat = torch.softmax(comb_flat, dim=-1)
        pi_flat = self.drop(pi_flat)
        return pi_flat.reshape(B, T, R, R, self.n_lags)

class TransformerPropagationHead(nn.Module):
    def __init__(self, d_model: int = 128, n_lags: int = 13, dropout: float = 0.1):
        super().__init__()
        self.factor_attn = FactorizedSpatialTemporalAttention(d_model=d_model, n_lags=n_lags, dropout=dropout)
        self.proj = nn.Linear(1, d_model)

    def forward(self, x_hat: torch.Tensor, anat_mask: torch.Tensor, max_lag: int) -> dict:
        B, R, T = x_hat.shape
        tokens = self.proj(x_hat.unsqueeze(-1))  # (B,R,T,D)
        lag_idx = torch.arange(max_lag, device=x_hat.device, dtype=torch.long)
        pi = self.factor_attn(tokens, anat_mask=anat_mask, lag_idx=lag_idx)

        # Drive û^j(t) = Σ_{i,ℓ} π_{i,ℓ→j} x_i(t-ℓ)
        # Vectorized computation using time-unfolding (avoids explicit lag loop).
        # x_windows: (B, R_src, T, L) where last index is lag ℓ (0=current).
        import torch.nn.functional as F
        x_pad = F.pad(x_hat, (max_lag - 1, 0))  # left pad time dimension
        x_windows = x_pad.unfold(dimension=-1, size=max_lag, step=1)  # (B,R,T,L)
        # Align with π which is (B,T,R_tgt,R_src,L):
        drive_btj = torch.einsum("b t j i l, b i t l -> b t j", pi, x_windows)
        drive = drive_btj.permute(0, 2, 1).contiguous()  # (B,R,T)

        return {"pi": pi, "drive": drive}
