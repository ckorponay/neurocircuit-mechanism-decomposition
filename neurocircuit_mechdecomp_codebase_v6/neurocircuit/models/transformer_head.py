from __future__ import annotations

import math
import torch
from torch import nn
from einops import rearrange

from neurocircuit.ops.graph_mask import apply_graph_mask
from neurocircuit.ops.relative_lag import RelativeLagEmbedding


def n_lags_for_seconds(max_lag_seconds: float, tr_seconds: float) -> int:
    """Number of sampled lags from zero through the largest lag <= max_lag_seconds."""
    if tr_seconds <= 0:
        raise ValueError("tr_seconds must be > 0")
    if max_lag_seconds < 0:
        raise ValueError("max_lag_seconds must be >= 0")
    return int(math.floor(max_lag_seconds / tr_seconds + 1e-9)) + 1


def lagged_windows(x: torch.Tensor, n_lags: int, *, legacy_oldest_first: bool = False) -> torch.Tensor:
    """
    Return [B,R,T,L] history windows.

    In the corrected convention, lag index 0 is x(t), lag index 1 is x(t-1),
    etc. The original repository's unfold operation returned the opposite order;
    `legacy_oldest_first=True` preserves that behavior for old checkpoints/results.
    """
    import torch.nn.functional as F

    if n_lags < 1:
        raise ValueError("n_lags must be >= 1")
    x_pad = F.pad(x, (n_lags - 1, 0))
    windows = x_pad.unfold(dimension=-1, size=n_lags, step=1)
    return windows if legacy_oldest_first else windows.flip(-1)


class FactorizedSpatialTemporalAttention(nn.Module):
    """
    Original additive factorized attention retained only for legacy replication.

    Important: S(i,j,t) + T(l,t) followed by one softmax over (i,l) factorizes
    algebraically, so the conditional lag distribution is identical across edges.
    New analyses should use EdgeConditionedSparseAttention instead.
    """

    def __init__(
        self,
        d_model: int,
        n_lags: int | None,
        dropout: float = 0.1,
        lag_embedding_mode: str = "index",
        max_lag_seconds: float = 12.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_lags = n_lags
        self.lag_embedding_mode = lag_embedding_mode

        self.spatial_q = nn.Linear(d_model, d_model, bias=False)
        self.spatial_k = nn.Linear(d_model, d_model, bias=False)
        self.temporal_q = nn.Linear(d_model, d_model, bias=False)
        self.rel_lag = RelativeLagEmbedding(
            n_lags=n_lags,
            dim=d_model,
            mode=lag_embedding_mode,
            max_lag_seconds=max_lag_seconds,
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        anat_mask: torch.Tensor,
        lag_values: torch.Tensor,
    ) -> torch.Tensor:
        B, R, T, D = tokens.shape
        L = int(lag_values.numel())

        q = rearrange(self.spatial_q(tokens), "b r t d -> b t r d")
        k = rearrange(self.spatial_k(tokens), "b r t d -> b t r d")
        spatial = torch.einsum("b t j d, b t i d -> b t j i", q, k) / (D ** 0.5)
        spatial = apply_graph_mask(spatial, anat_mask.T)

        q_t = self.temporal_q(tokens.mean(dim=1))
        lag_emb = self.rel_lag(lag_values).to(tokens.device)
        temporal = torch.einsum("b t d, l d -> b t l", q_t, lag_emb) / (D ** 0.5)

        comb = spatial.unsqueeze(-1) + temporal.unsqueeze(-2).unsqueeze(-2)
        comb_flat = comb.reshape(B, T, R, R * L)
        pi_flat = torch.softmax(comb_flat, dim=-1)
        pi_flat = self.drop(pi_flat)
        return pi_flat.reshape(B, T, R, R, L)


class EdgeConditionedSparseAttention(nn.Module):
    """
    Anatomically sparse, pathway-specific lag attention.

    For each allowed source i -> target j, lag logits contain an interaction
    between the current source/target state embedding and the physical-lag
    embedding. This breaks the degeneracy of the original S_ij + T_l design and
    permits a different lag distribution for each pathway.

    The implementation loops over target nodes and only materializes incoming
    anatomical edges for that target, avoiding the dense [R,R,L] tensor during
    normal training/inference. Full dense pi is optional and intended only for
    small debugging/visualization runs.
    """

    def __init__(
        self,
        d_model: int,
        n_lags: int | None,
        dropout: float = 0.0,
        lag_embedding_mode: str = "seconds",
        max_lag_seconds: float = 12.0,
        interaction_dim: int = 16,
    ):
        super().__init__()
        self.d_model = int(d_model)
        self.interaction_dim = int(interaction_dim)
        self.n_lags = n_lags
        self.lag_embedding_mode = lag_embedding_mode

        self.spatial_q = nn.Linear(d_model, d_model, bias=False)
        self.spatial_k = nn.Linear(d_model, d_model, bias=False)
        self.edge_q = nn.Linear(d_model, interaction_dim, bias=False)
        self.edge_k = nn.Linear(d_model, interaction_dim, bias=False)
        self.global_lag_q = nn.Linear(d_model, interaction_dim, bias=False)
        self.rel_lag = RelativeLagEmbedding(
            n_lags=n_lags,
            dim=interaction_dim,
            mode=lag_embedding_mode,
            max_lag_seconds=max_lag_seconds,
        )
        self.interaction_gain = nn.Parameter(torch.tensor(1.0))
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        x_windows: torch.Tensor,
        anat_mask: torch.Tensor,
        lag_values: torch.Tensor,
        lag_seconds: torch.Tensor,
        *,
        return_pi: bool = False,
    ) -> dict:
        B, R, T, D = tokens.shape
        L = int(lag_values.numel())
        if anat_mask.shape != (R, R):
            raise ValueError("anat_mask must be [R_src,R_tgt]")
        if x_windows.shape != (B, R, T, L):
            raise ValueError("x_windows must be [B,R,T,L]")

        mask = anat_mask.to(dtype=torch.bool, device=tokens.device)
        lag_emb = self.rel_lag(lag_values).to(tokens.device)
        global_context = tokens.mean(dim=1)  # [B,T,D]
        global_lag = torch.einsum(
            "btd,ld->btl",
            self.global_lag_q(global_context),
            lag_emb,
        ) / math.sqrt(self.interaction_dim)

        drive = torch.zeros(B, R, T, device=tokens.device, dtype=tokens.dtype)
        edge_mass = torch.zeros(B, R, R, device=tokens.device, dtype=tokens.dtype)
        edge_lag_centroid = torch.full_like(edge_mass, float("nan"))
        edge_lag_peak = torch.full_like(edge_mass, float("nan"))
        edge_lag_concentration = torch.full_like(edge_mass, float("nan"))

        pi_dense = None
        if return_pi:
            pi_dense = torch.zeros(
                B, T, R, R, L,
                device=tokens.device,
                dtype=tokens.dtype,
            )

        lag_s = lag_seconds.to(tokens.dtype)
        log_L = math.log(max(L, 2))

        for j in range(R):
            src = torch.nonzero(mask[:, j], as_tuple=False).flatten()
            if src.numel() == 0:
                continue

            tgt_tok = tokens[:, j, :, :]  # [B,T,D]
            src_tok = tokens[:, src, :, :].permute(0, 2, 1, 3)  # [B,T,K,D]

            q_sp = self.spatial_q(tgt_tok)
            k_sp = self.spatial_k(src_tok)
            spatial = torch.einsum("btd,btkd->btk", q_sp, k_sp) / math.sqrt(D)

            q_edge = self.edge_q(tgt_tok).unsqueeze(2)  # [B,T,1,d]
            k_edge = self.edge_k(src_tok)               # [B,T,K,d]
            edge_ctx = torch.tanh(q_edge + k_edge)
            interaction = torch.einsum(
                "btkd,ld->btkl", edge_ctx, lag_emb
            ) / math.sqrt(self.interaction_dim)

            scores = (
                spatial.unsqueeze(-1)
                + global_lag.unsqueeze(2)
                + self.interaction_gain * interaction
            )
            K = int(src.numel())
            pi = torch.softmax(scores.reshape(B, T, K * L), dim=-1).reshape(B, T, K, L)
            pi = self.drop(pi)

            x_src = x_windows[:, src, :, :].permute(0, 2, 1, 3)  # [B,T,K,L]
            drive[:, j, :] = (pi * x_src).sum(dim=(2, 3))

            # Time-integrated pathway summaries, keeping source and target explicit.
            pi_time = pi.sum(dim=1)  # [B,K,L]
            mass = pi_time.sum(dim=-1)  # [B,K]
            cond = pi_time / mass.unsqueeze(-1).clamp_min(1e-12)
            centroid = (cond * lag_s.view(1, 1, L)).sum(dim=-1)
            peak_idx = cond.argmax(dim=-1)
            peak = lag_s[peak_idx]
            entropy = -(cond.clamp_min(1e-12) * cond.clamp_min(1e-12).log()).sum(dim=-1)
            concentration = 1.0 - entropy / log_L

            edge_mass[:, src, j] = mass / float(T)
            edge_lag_centroid[:, src, j] = centroid
            edge_lag_peak[:, src, j] = peak
            edge_lag_concentration[:, src, j] = concentration

            if pi_dense is not None:
                # pi_dense: target j, source i
                pi_dense[:, :, j, src, :] = pi

        out = {
            "drive": drive,
            "edge_mass": edge_mass,
            "edge_lag_centroid_seconds": edge_lag_centroid,
            "edge_lag_peak_seconds": edge_lag_peak,
            "edge_lag_concentration": edge_lag_concentration,
        }
        if pi_dense is not None:
            out["pi"] = pi_dense
        return out


class TransformerPropagationHead(nn.Module):
    def __init__(
        self,
        d_model: int = 128,
        n_lags: int | None = 13,
        dropout: float = 0.1,
        lag_embedding_mode: str = "index",
        max_lag_seconds: float = 12.0,
        attention_mode: str = "legacy_additive",
        interaction_dim: int = 16,
        legacy_oldest_first: bool = True,
    ):
        super().__init__()
        if attention_mode not in {"legacy_additive", "edge_conditioned_sparse"}:
            raise ValueError("unknown attention_mode")
        self.n_lags = n_lags
        self.lag_embedding_mode = lag_embedding_mode
        self.max_lag_seconds = float(max_lag_seconds)
        self.attention_mode = attention_mode
        self.legacy_oldest_first = bool(legacy_oldest_first)

        if attention_mode == "legacy_additive":
            self.attention = FactorizedSpatialTemporalAttention(
                d_model=d_model,
                n_lags=n_lags,
                dropout=dropout,
                lag_embedding_mode=lag_embedding_mode,
                max_lag_seconds=max_lag_seconds,
            )
        else:
            self.attention = EdgeConditionedSparseAttention(
                d_model=d_model,
                n_lags=n_lags,
                dropout=dropout,
                lag_embedding_mode=lag_embedding_mode,
                max_lag_seconds=max_lag_seconds,
                interaction_dim=interaction_dim,
            )
        self.proj = nn.Linear(1, d_model)

    def _lag_grid(
        self,
        x_hat: torch.Tensor,
        max_lag: int | None,
        tr_seconds: float | None,
        max_lag_seconds: float | None,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        if self.lag_embedding_mode == "index":
            L = int(max_lag if max_lag is not None else self.n_lags)
            lag_values = torch.arange(L, device=x_hat.device, dtype=torch.long)
            dt = 1.0 if tr_seconds is None else float(tr_seconds)
            lag_seconds = lag_values.to(x_hat.dtype) * dt
            return lag_values, lag_seconds, L

        if tr_seconds is None:
            raise ValueError("seconds-based lag embedding requires tr_seconds")
        span = self.max_lag_seconds if max_lag_seconds is None else float(max_lag_seconds)
        L = n_lags_for_seconds(span, float(tr_seconds))
        lag_seconds = torch.arange(L, device=x_hat.device, dtype=x_hat.dtype) * float(tr_seconds)
        return lag_seconds, lag_seconds, L

    def forward(
        self,
        x_hat: torch.Tensor,
        anat_mask: torch.Tensor,
        max_lag: int | None = None,
        tr_seconds: float | None = None,
        max_lag_seconds: float | None = None,
        *,
        return_pi: bool | None = None,
    ) -> dict:
        B, R, T = x_hat.shape
        tokens = self.proj(x_hat.unsqueeze(-1))
        lag_values, lag_seconds, L = self._lag_grid(
            x_hat, max_lag, tr_seconds, max_lag_seconds
        )

        if self.attention_mode == "legacy_additive":
            pi = self.attention(tokens, anat_mask=anat_mask, lag_values=lag_values)
            x_windows = lagged_windows(
                x_hat,
                L,
                legacy_oldest_first=self.legacy_oldest_first,
            )
            drive_btj = torch.einsum("b t j i l, b i t l -> b t j", pi, x_windows)
            drive = drive_btj.permute(0, 2, 1).contiguous()
            return {
                "pi": pi,
                "drive": drive,
                "lag_seconds": lag_seconds,
                "n_lags": L,
                "attention_mode": self.attention_mode,
            }

        # Corrected convention for new analyses: lag 0 = current sample.
        x_windows = lagged_windows(x_hat, L, legacy_oldest_first=False)
        if return_pi is None:
            return_pi = False
        out = self.attention(
            tokens,
            x_windows=x_windows,
            anat_mask=anat_mask,
            lag_values=lag_values,
            lag_seconds=lag_seconds,
            return_pi=bool(return_pi),
        )
        out.update(
            {
                "lag_seconds": lag_seconds,
                "n_lags": L,
                "attention_mode": self.attention_mode,
            }
        )
        return out
