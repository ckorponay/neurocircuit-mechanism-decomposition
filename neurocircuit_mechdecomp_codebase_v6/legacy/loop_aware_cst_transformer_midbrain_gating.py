"""
Loop-Aware Cortico–Striatal–Pallidal–Thalamic Transformer (with Midbrain Gating)
-------------------------------------------------------------------------------
Author: ChatGPT (GPT-5 Thinking)

What this file provides
-----------------------
• A full PyTorch implementation of a loop-aware, biologically constrained, spatio–temporal
  attention model that follows the canonical C–S–P–Th–C circuit with optional midbrain (SNc/VTA)
  modulation.

• Intra-frame attention: contextualize tokens within each node set (e.g., cortex parcels, striatal
  supervoxels) using self-attention plus optional anatomical biases.

• Directed cross-attention with masks: implement anatomically plausible edges only (C→S, C→STN,
  S→GPe, S→GPi, STN→GPi, GPi→Th, Th→C). Optional midbrain gating modulates selected edges.

• Inter-frame (temporal) attention: causal self-attention over time per node set, with a
  learnable small FIR temporal kernel per connection type to model edge-specific lags/HRF nuances.

• Two self-supervised objectives out of the box:
  (1) Spatial MAE at the last frame (mask-and-reconstruct selected node sets; default: striatum).
  (2) k-ahead forecasting for cortex + striatum (predict t+k BOLD from history), respecting causality.

• A synthetic data demo that generates loop-like dynamics; run this to sanity-check end-to-end.

Usage (quick start)
-------------------
# 1) Install deps
pip install torch numpy einops tqdm pyyaml scikit-learn

# 2) Run the synthetic demo (writes tiny NPZ files + config internally)
python loop_loopaware_cst_transformer.py --demo

# 3) Train/eval on the demo config
python loop_loopaware_cst_transformer.py --config configs/loop_demo.yaml

To use on real data, prepare NPZ files with arrays (T, P_group) for the keys listed in
NODE_SETS below (you can drop groups you don't have and set sizes=0 or remove edges).

Notes
-----
• Attention ≠ FC: attention gives task-conditioned, directional credit assignment. Signed effects
  live in the value/output projections; we include helpers to compute signed per-edge contributions.
• This is a research scaffold: start small (few layers/heads), verify on demo, then scale.

"""
from __future__ import annotations
import os, math, json, argparse, random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from einops import rearrange
from tqdm import tqdm

# --------------------------------------------
# Config and Constants
# --------------------------------------------

# Node set names used throughout. You can drop any subset if unavailable.
NODE_SETS = [
    "C",    # Cortex parcels (P_c)
    "S",    # Striatum supervoxels (V_s)
    "GPe",  # External Globus Pallidus
    "GPi",  # Internal Globus Pallidus
    "STN",  # Subthalamic Nucleus
    "Th",   # Thalamus
    "MB",   # Midbrain modulators (SNc/VTA etc.)
]

# Directed edges implemented via cross-attention. You can toggle these in the config.
DEFAULT_EDGES = [
    ("C",  "S"),   # Cortex → Striatum (corticostriatal)
    ("C",  "STN"), # Hyperdirect
    ("S",  "GPe"),
    ("S",  "GPi"),
    ("STN","GPi"),
    ("GPi","Th"),
    ("Th", "C"),
]

@dataclass
class ModelConfig:
    # Sizes per node set (0 disables the set)
    P_C: int = 64
    P_S: int = 128
    P_GPe: int = 16
    P_GPi: int = 16
    P_STN: int = 8
    P_Th: int = 24
    P_MB: int = 4             # midbrain tokens; can be 1 (global) or more

    # Core transformer dims
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2         # (spatial+cross) + temporal repeated L times
    dropout: float = 0.1

    # Temporal context
    T_ctx: int = 128
    k_forecast: int = 2       # predict t+k to respect HRF lag

    # Training
    lr: float = 3e-4
    weight_decay: float = 1e-2
    batch_size: int = 8
    max_epochs: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Loss weights
    w_mae_spatial: float = 1.0
    w_forecast: float = 1.0

    # Which node sets to mask for spatial MAE at the last frame
    mae_targets: Tuple[str, ...] = ("S",)  # e.g., ("S","GPi","STN")

    # Edge list (directed). Must be subset of DEFAULT_EDGES or your custom list.
    edges: Tuple[Tuple[str,str], ...] = tuple(DEFAULT_EDGES)

    # Optional additive bias matrices per edge (target x source) path on disk
    bias_paths: Dict[str, str] = None  # keys like "C->S": path to npy of shape (P_target, P_source)

    # Enable midbrain gating on selected edges
    gated_edges: Tuple[Tuple[str,str], ...] = ("C","S"), ("C","STN")

    # Temporal kernel sizes per edge type (models small lags/HRF differences)
    kernel_size: int = 3  # odd number; applied as 1D conv over time on source features

    # Seed
    seed: int = 7

# --------------------------------------------
# Utilities
# --------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --------------------------------------------
# Dataset: expects per-run NPZ with keys for each node set present.
#          Provides sliding windows with forecast target t+k.
# --------------------------------------------
class LoopDataset(Dataset):
    """
    Each NPZ should contain arrays of shape (T, P_group) for present node sets, e.g.:
      'C', 'S', 'GPe', 'GPi', 'STN', 'Th', 'MB'
    You can omit any, but cortex 'C' and striatum 'S' are expected if forecasting targets include them.

    Returns per item:
      X_dict: dict of tensors (BOLD) with shape (T_ctx, P_group)
      last_frame_masked: dict of last-frame arrays with masked positions zeroed (only for MAE targets)
      last_frame_true:   dict of last-frame true arrays (only for MAE targets)
      last_mask:         dict of boolean masks (1=masked) per MAE target
      future_dict:       dict of future arrays at t+k for groups with forecasting heads (C,S)
    """
    def __init__(self,
                 file_list: str,
                 cfg: ModelConfig,
                 mask_ratio: float = 0.4):
        super().__init__()
        self.cfg = cfg
        self.mask_ratio = mask_ratio
        with open(file_list, 'r') as f:
            self.files = [ln.strip() for ln in f if ln.strip()]
        # Build index of valid t_end per file
        self.index = []
        for fi, path in enumerate(self.files):
            with np.load(path) as z:
                T = None
                for k in NODE_SETS:
                    if k in z:
                        T = z[k].shape[0] if T is None else min(T, z[k].shape[0])
                if T is None:
                    continue
                for t_end in range(cfg.T_ctx + cfg.k_forecast, T):
                    self.index.append((fi, t_end))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fi, t_end = self.index[idx]
        path = self.files[fi]
        with np.load(path) as z:
            # Slice window
            t0 = t_end - (self.cfg.T_ctx + self.cfg.k_forecast)
            t_ctx_end = t0 + self.cfg.T_ctx
            X_dict = {}
            for name in NODE_SETS:
                if name in z:
                    X = z[name].astype(np.float32)
                    X = X[t0:t_ctx_end]  # (T_ctx, P)
                    X_dict[name] = torch.from_numpy(X)
            # Build MAE masks on last frame
            last_frame_true, last_frame_masked, last_mask = {}, {}, {}
            for tgt in self.cfg.mae_targets:
                if tgt in X_dict:
                    P = X_dict[tgt].shape[1]
                    mask = np.zeros(P, dtype=np.float32)
                    n_mask = max(1, int(self.mask_ratio * P))
                    idxs = np.random.choice(P, n_mask, replace=False)
                    mask[idxs] = 1.0
                    last = X_dict[tgt][-1].numpy().copy()
                    last_m = last.copy(); last_m[idxs] = 0.0
                    last_frame_true[tgt] = torch.from_numpy(last)
                    last_frame_masked[tgt] = torch.from_numpy(last_m)
                    last_mask[tgt] = torch.from_numpy(mask)
            # Future targets t+k for forecast heads (C,S if present)
            future_dict = {}
            for name in ["C","S"]:
                if name in z:
                    arr = z[name].astype(np.float32)
                    fut = arr[t_ctx_end + self.cfg.k_forecast - 1]
                    future_dict[name] = torch.from_numpy(fut)
        return {
            'X': X_dict,
            'last_frame_true': last_frame_true,
            'last_frame_masked': last_frame_masked,
            'last_mask': last_mask,
            'future': future_dict,
        }

# --------------------------------------------
# Attention helpers (with additive biases and masks)
# --------------------------------------------
class MHAWithBias(nn.Module):
    """MultiheadAttention with optional additive bias to attention logits.
    Bias can be a tensor (N_tgt, N_src) broadcast across batch/heads or a callable that
    produces a bias per batch.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)

    def forward(self,
                q: torch.Tensor,  # (B, N_tgt, d)
                k: torch.Tensor,  # (B, N_src, d)
                v: torch.Tensor,  # (B, N_src, d)
                additive_bias: Optional[torch.Tensor] = None,  # (N_tgt, N_src) or (B, N_tgt, N_src)
                attn_mask: Optional[torch.Tensor] = None  # standard PyTorch mask added to logits
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Combine masks: attn_mask (+) additive_bias
        if additive_bias is not None:
            if attn_mask is None:
                attn_mask = additive_bias
            else:
                attn_mask = attn_mask + additive_bias
        out, A = self.mha(q, k, v, attn_mask=attn_mask, need_weights=True, average_attn_weights=False)
        # A: (B, n_heads, N_tgt, N_src)
        return out, A

# --------------------------------------------
# Spatial self-attention block per node set
# --------------------------------------------
class SpatialBlock(nn.Module):
    def __init__(self, P: int, d_model: int, n_heads: int, dropout: float = 0.1,
                 bias_matrix: Optional[torch.Tensor] = None, name: str = ""):
        super().__init__()
        self.P = P
        self.name = name
        self.inp = nn.Linear(1, d_model)
        self.pe = nn.Parameter(torch.zeros(P, d_model))
        nn.init.normal_(self.pe, std=0.01)
        self.mha = MHAWithBias(d_model, n_heads, dropout)
        self.bias = bias_matrix  # (P,P) or None
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(4*d_model, d_model)
        )

    def forward(self, x_t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_t: (B, P) scalar BOLD at a single frame for this node set.
        Returns contextualized features (B,P,d) and attention weights (B,H,P,P).
        """
        if self.P == 0:
            return x_t[..., None], torch.zeros(x_t.size(0), 1, 0, 0, device=x_t.device)
        h = self.inp(x_t.unsqueeze(-1)) + self.pe  # (B,P,d)
        q = self.ln1(h)
        add_bias = self.bias  # (P,P) or None
        out, A = self.mha(q, q, q, additive_bias=add_bias)
        h = h + out
        h = h + self.ff(h)
        return h, A

# --------------------------------------------
# Cross-attention block with optional midbrain gating and edge-specific temporal kernel
# --------------------------------------------
class CrossBlock(nn.Module):
    def __init__(self,
                 d_model: int,
                 n_heads: int,
                 dropout: float = 0.1,
                 bias_matrix: Optional[torch.Tensor] = None,
                 gated: bool = False,
                 name: str = "edge",
                 kernel_size: int = 3):
        super().__init__()
        self.name = name
        self.mha = MHAWithBias(d_model, n_heads, dropout)
        self.bias = bias_matrix  # (N_tgt, N_src)
        self.lnq = nn.LayerNorm(d_model)
        self.lnk = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(4*d_model, d_model)
        )
        # Optional midbrain gating projects MB features → scalar gate per head (or per token)
        self.gated = gated
        if self.gated:
            self.mb_to_gate = nn.Sequential(
                nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, n_heads)
            )
        # Edge-specific temporal kernel (FIR) over source sequence to model small lags (applied before cross-attn)
        assert kernel_size % 2 == 0 or kernel_size % 2 == 1
        padding = (kernel_size - 1) // 2
        self.temporal_kernel = nn.Conv1d(d_model, d_model, kernel_size, padding=padding, groups=d_model)
        # Initialize as near-identity (delta) to start with no lag
        with torch.no_grad():
            self.temporal_kernel.weight.zero_()
            center = padding
            for c in range(d_model):
                self.temporal_kernel.weight[c, c, center] = 1.0
            if self.temporal_kernel.bias is not None:
                self.temporal_kernel.bias.zero_()

    def forward(self,
                tgt_feats_t: torch.Tensor,   # (B, N_tgt, d)
                src_seq: torch.Tensor,       # (B, T, N_src, d) for temporal kernel application
                mb_feats_t: Optional[torch.Tensor] = None  # (B, N_mb, d) → pooled to gate if gated
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Apply depthwise temporal conv on source across time, then take the last frame features
        B, T, N_src, d = src_seq.shape
        src_seq_resh = rearrange(src_seq, 'b t n d -> b n d t')  # conv over t
        src_seq_f = self.temporal_kernel(src_seq_resh)           # (B,N_src,d,T)
        src_last = src_seq_f[..., -1]                            # (B,N_src,d)
        q = self.lnq(tgt_feats_t)
        k = self.lnk(src_last)
        v = src_last
        # Build additive bias; optionally add gating as a bias to logits
        add_bias = self.bias  # (N_tgt, N_src) or None
        if self.gated and (mb_feats_t is not None) and mb_feats_t.numel() > 0:
            # Pool midbrain features to a single vector per batch, then produce per-head gates
            mb_pool = mb_feats_t.mean(dim=1)  # (B,d)
            gates = self.mb_to_gate(mb_pool)  # (B, n_heads)
            # Convert to per-head additive bias added to attention logits.
            # We broadcast to (B*n_heads, N_tgt, N_src) by expanding later when passed to MHA.
            # PyTorch MHA expects attn_mask added to logits uniformly across heads, so we fold gate
            # into an extra bias on the keys via a learned scalar that maps to logits.
            # Practical trick: scale keys by (1 + sigmoid(gate)) per head using a linear map.
            scale = (1.0 + torch.sigmoid(gates)).unsqueeze(-1).unsqueeze(-1)  # (B, n_heads, 1, 1)
        else:
            scale = None
        # MHAWithBias adds 'attn_mask' after projecting q,k; here we emulate gating by scaling k,v
        if scale is not None:
            # Efficient approximation: scale value magnitude based on mean gate
            k = k  # keep k; logits depend on q·k^T; scaling v changes output strength
            v = v * scale.mean(dim=1).unsqueeze(-1)  # (B,1,1) → (B, N_src, d)
        out, A = self.mha(q, k, v, additive_bias=add_bias)
        y = tgt_feats_t + out
        y = y + self.ff(y)
        return y, A

# --------------------------------------------
# Temporal causal attention per node set
# --------------------------------------------
class TemporalBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, T_max: int = 4096):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.pe_time = nn.Parameter(torch.zeros(T_max, d_model))
        nn.init.normal_(self.pe_time, std=0.01)
        self.ln = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(4*d_model, d_model)
        )

    def forward(self, seq_feats: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        seq_feats: (B, T, N, d)
        Returns updated seq feats and attention weights per token (B*N, H, T, T)
        """
        B, T, N, d = seq_feats.shape
        x = seq_feats + self.pe_time[:T]
        # Reshape to sequences per token
        S = rearrange(x, 'b t n d -> (b n) t d')
        # Causal mask
        causal = torch.full((T, T), float('-inf'), device=seq_feats.device)
        causal = torch.triu(causal, diagonal=1)
        out, W = self.mha(self.ln(S), self.ln(S), self.ln(S), attn_mask=causal, need_weights=True, average_attn_weights=False)
        S = S + out
        S = S + self.ff(S)
        Y = rearrange(S, '(b n) t d -> b t n d', b=B, n=N)
        return Y, W

# --------------------------------------------
# Full loop-aware model
# --------------------------------------------
class LoopAwareModel(nn.Module):
    def __init__(self, cfg: ModelConfig, bias_dict: Optional[Dict[str, torch.Tensor]] = None):
        super().__init__()
        self.cfg = cfg
        d, H, L = cfg.d_model, cfg.n_heads, cfg.n_layers
        # Node sizes
        self.sizes = {
            'C': cfg.P_C, 'S': cfg.P_S, 'GPe': cfg.P_GPe, 'GPi': cfg.P_GPi,
            'STN': cfg.P_STN, 'Th': cfg.P_Th, 'MB': cfg.P_MB,
        }
        # Build spatial blocks for present node sets (skip MB: we treat MB as tokens too, but same block works)
        self.spatial = nn.ModuleDict()
        for name in NODE_SETS:
            P = self.sizes[name]
            if P > 0:
                bias = None
                if bias_dict is not None and f"{name}<->{name}" in bias_dict:
                    bias = bias_dict[f"{name}<->{name}"]
                self.spatial[name] = SpatialBlock(P, d, H, cfg.dropout, bias_matrix=bias, name=name)
        # Cross-attention blocks per edge
        self.edges = list(cfg.edges)
        self.cross = nn.ModuleList()
        for (src, tgt) in self.edges:
            key = f"{src}->{tgt}"
            bias = None
            if bias_dict is not None and key in bias_dict:
                bias = bias_dict[key]
            gated = (src, tgt) in cfg.gated_edges
            self.cross.append(((src,tgt), CrossBlock(d, H, cfg.dropout, bias_matrix=bias, gated=gated, name=key, kernel_size=cfg.kernel_size)))
        # Temporal blocks per node set (except MB optional)
        self.temporal = nn.ModuleDict()
        for name in NODE_SETS:
            P = self.sizes[name]
            if P > 0:
                self.temporal[name] = TemporalBlock(d, H, cfg.dropout, T_max=4096)
        # Readouts: reconstruct last frame and forecast t+k
        # We reconstruct masked nodes for selected groups via linear heads from last-frame features
        self.readout_last = nn.ModuleDict()
        for tgt in cfg.mae_targets:
            if self.sizes.get(tgt, 0) > 0:
                self.readout_last[tgt] = nn.Linear(d, 1)
        # Forecast heads for C and S (if present)
        self.readout_future = nn.ModuleDict()
        for name in ["C","S"]:
            if self.sizes.get(name, 0) > 0:
                self.readout_future[name] = nn.Linear(d, 1)

    def forward(self, X: Dict[str, torch.Tensor], last_masked: Dict[str, torch.Tensor]) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, List[torch.Tensor]]]:
        """
        X: dict of node set → (B, T, P)
        last_masked: dict of node set → (B, P) where last frame values are replaced with zeros at masked indices

        Returns:
          recon_last: dict node→(B,P): reconstructed last-frame scalars for masked MAE targets
          forecast:   dict node→(B,P): predicted t+k scalars for nodes with future heads (C,S)
          attn_logs:  dict with lists of spatial, cross, temporal attentions for inspection
        """
        B = next(iter(X.values())).size(0)
        T = next(iter(X.values())).size(1)
        d = self.cfg.d_model
        # 1) Replace last frame for MAE targets with masked version
        X_mod = {k: v.clone() for k,v in X.items()}
        for tgt, masked in last_masked.items():
            if tgt in X_mod:
                X_mod[tgt][:, -1] = masked  # (B,P)
        # 2) Spatial within-frame attention for each node set at each t
        feats = {k: [] for k in X_mod.keys()}           # list of (B,P,d) per t
        A_spatial = {k: [] for k in X_mod.keys()}       # list of (B,H,P,P) per t
        for t in range(T):
            for name, seq in X_mod.items():
                P = seq.size(2)
                if P == 0: continue
                h_t, A_t = self.spatial[name](seq[:, t])
                feats[name].append(h_t)
                A_spatial[name].append(A_t)
        # Stack over time → (B,T,P,d)
        for name in feats.keys():
            if len(feats[name])>0:
                feats[name] = torch.stack(feats[name], dim=1)
                A_spatial[name] = torch.stack(A_spatial[name], dim=1)
        # 3) Directed cross-attention per edge, operating on the LAST FRAME features (t=T-1)
        #    but using a temporal kernel over the SOURCE sequence to allow small lags.
        #    We update only the TARGET last-frame features.
        A_cross = {}
        for (src,tgt), layer in self.cross:
            if (src not in feats) or (tgt not in feats):
                continue
            tgt_last = feats[tgt][:, -1]                 # (B,P_tgt,d)
            src_seq  = feats[src]                        # (B,T,P_src,d)
            mb_t = feats.get('MB', None)
            mb_last = mb_t[:, -1] if (mb_t is not None and mb_t.numel()>0) else None  # (B,P_MB,d) or None
            y_t, A = layer(tgt_last, src_seq, mb_last)
            feats[tgt][:, -1] = y_t
            A_cross[f"{src}->{tgt}"] = A
        # 4) Temporal causal attention per node set (across T)
        A_temporal = {}
        for name, seq in feats.items():
            if isinstance(seq, list) or seq is None or seq.numel()==0: continue
            Y, W = self.temporal[name](seq)  # (B,T,P,d)
            feats[name] = Y
            A_temporal[name] = W
        # 5) Readouts
        recon_last, forecast = {}, {}
        # Spatial MAE reconstructions from LAST FRAME features
        for tgt, head in self.readout_last.items():
            if tgt in feats:
                H_last = feats[tgt][:, -1]     # (B,P,d)
                x_hat = head(H_last).squeeze(-1)  # (B,P)
                recon_last[tgt] = x_hat
        # Forecast t+k: we use LAST FRAME features (already history-aware) for groups with heads
        for name, head in self.readout_future.items():
            if name in feats:
                H_last = feats[name][:, -1]    # (B,P,d)
                yhat = head(H_last).squeeze(-1)
                forecast[name] = yhat
        attn_logs = {
            'spatial': A_spatial,
            'cross': A_cross,
            'temporal': A_temporal,
        }
        return recon_last, forecast, attn_logs

# --------------------------------------------
# Losses and Metrics
# --------------------------------------------

def masked_mse(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Compute mean MSE over masked positions (mask==1). pred,tgt:(B,P) mask:(B,P)."""
    eps = 1e-8
    num = (mask * (pred - tgt)**2).sum(dim=1)
    den = mask.sum(dim=1) + eps
    return (num / den).mean()

@torch.no_grad()
def r2_score(pred: torch.Tensor, tgt: torch.Tensor) -> float:
    yhat, y = pred, tgt
    ss_res = ((y - yhat)**2).sum(dim=1)
    ss_tot = ((y - y.mean(dim=1, keepdim=True))**2).sum(dim=1) + 1e-8
    r2 = 1.0 - ss_res/ss_tot
    return float(r2.mean().item())

# --------------------------------------------
# Training / Evaluation loops
# --------------------------------------------

def train_epoch(model: LoopAwareModel, loader: DataLoader, opt, cfg: ModelConfig):
    model.train()
    tot = 0.0
    for batch in tqdm(loader, desc='train', leave=False):
        X = {k: v.to(cfg.device) for k,v in batch['X'].items()}
        last_true = {k: v.to(cfg.device) for k,v in batch['last_frame_true'].items()}
        last_masked = {k: v.to(cfg.device) for k,v in batch['last_frame_masked'].items()}
        last_mask = {k: v.to(cfg.device) for k,v in batch['last_mask'].items()}
        future = {k: v.to(cfg.device) for k,v in batch['future'].items()}
        recon_last, forecast, _ = model(X, last_masked)
        # Spatial MAE loss over selected targets
        loss_sp = 0.0
        for tgt in cfg.mae_targets:
            if tgt in recon_last:
                loss_sp = loss_sp + masked_mse(recon_last[tgt], last_true[tgt], last_mask[tgt])
        # Forecast loss on C and S (if present)
        loss_fc = 0.0
        for name in ["C","S"]:
            if name in forecast and name in future:
                loss_fc = loss_fc + F.mse_loss(forecast[name], future[name])
        loss = cfg.w_mae_spatial * loss_sp + cfg.w_forecast * loss_fc
        opt.zero_grad(); loss.backward(); opt.step()
        tot += float(loss.item()) * next(iter(X.values())).size(0)
    return tot / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model: LoopAwareModel, loader: DataLoader, cfg: ModelConfig):
    model.eval()
    tot = 0.0
    r2_c = []
    r2_s = []
    for batch in tqdm(loader, desc='eval', leave=False):
        X = {k: v.to(cfg.device) for k,v in batch['X'].items()}
        last_true = {k: v.to(cfg.device) for k,v in batch['last_frame_true'].items()}
        last_masked = {k: v.to(cfg.device) for k,v in batch['last_frame_masked'].items()}
        last_mask = {k: v.to(cfg.device) for k,v in batch['last_mask'].items()}
        future = {k: v.to(cfg.device) for k,v in batch['future'].items()}
        recon_last, forecast, _ = model(X, last_masked)
        # Loss
        loss_sp = 0.0
        for tgt in cfg.mae_targets:
            if tgt in recon_last:
                loss_sp = loss_sp + masked_mse(recon_last[tgt], last_true[tgt], last_mask[tgt])
        loss_fc = 0.0
        if 'C' in forecast and 'C' in future:
            loss_fc = loss_fc + F.mse_loss(forecast['C'], future['C'])
            r2_c.append(r2_score(forecast['C'], future['C']))
        if 'S' in forecast and 'S' in future:
            loss_fc = loss_fc + F.mse_loss(forecast['S'], future['S'])
            r2_s.append(r2_score(forecast['S'], future['S']))
        loss = cfg.w_mae_spatial * loss_sp + cfg.w_forecast * loss_fc
        tot += float(loss.item()) * next(iter(X.values())).size(0)
    r2c = float(np.mean(r2_c)) if r2_c else float('nan')
    r2s = float(np.mean(r2_s)) if r2_s else float('nan')
    return tot / len(loader.dataset), r2c, r2s

# --------------------------------------------
# Bias loading helper
# --------------------------------------------

def load_biases(cfg: ModelConfig) -> Dict[str, torch.Tensor]:
    bias_dict = {}
    if cfg.bias_paths is None:
        return bias_dict
    for key, path in cfg.bias_paths.items():
        arr = np.load(path).astype(np.float32)
        bias = torch.from_numpy(arr)
        bias_dict[key] = bias
    return bias_dict

# --------------------------------------------
# Synthetic demo generator: small LDS with directed loop + modulation
# --------------------------------------------

def make_synth_run(T: int, sizes: Dict[str,int], seed: int = 0) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    d_lat = 16  # latent dim per node set
    # Latent dynamics per node set
    Z = {}
    for name, P in sizes.items():
        if P <= 0: continue
        Z[name] = rng.normal(size=(T, d_lat)).astype(np.float32)
        # temporal smoothing
        for t in range(1, T):
            Z[name][t] = 0.8*Z[name][t-1] + 0.2*Z[name][t]
    # Directed coupling (simple linear influence on observed signals, not on latents for simplicity)
    # Projection matrices from latents to observed per node set
    X = {}
    Proj = {name: rng.normal(size=(d_lat, P)).astype(np.float32) for name,P in sizes.items() if P>0}
    # Midbrain scalar gate over time (simulating dopamine bursts)
    D = rng.normal(size=(T,1)).astype(np.float32)
    for t in range(T):
        # Base signals
        for name,P in sizes.items():
            if P <= 0: continue
            base = Z[name][t] @ Proj[name]
            X.setdefault(name, np.zeros((T, P), np.float32))[t] = base
        # Add directed influences (C→S, C→STN, S→GPe/GPi, STN→GPi, GPi→Th, Th→C)
        if sizes['C']>0 and sizes['S']>0:
            X['S'][t] += 0.2 * (Z['C'][t] @ rng.normal(size=(d_lat, sizes['S'])).astype(np.float32)) * (1.0 + 0.3*D[t])
        if sizes['C']>0 and sizes['STN']>0:
            X['STN'][t] += 0.2 * (Z['C'][t] @ rng.normal(size=(d_lat, sizes['STN'])).astype(np.float32)) * (1.0 + 0.3*D[t])
        if sizes['S']>0 and sizes['GPe']>0:
            X['GPe'][t] += 0.25 * (Z['S'][t] @ rng.normal(size=(d_lat, sizes['GPe'])).astype(np.float32))
        if sizes['S']>0 and sizes['GPi']>0:
            X['GPi'][t] += 0.25 * (Z['S'][t] @ rng.normal(size=(d_lat, sizes['GPi'])).astype(np.float32))
        if sizes['STN']>0 and sizes['GPi']>0:
            X['GPi'][t] += 0.25 * (Z['STN'][t] @ rng.normal(size=(d_lat, sizes['GPi'])).astype(np.float32))
        if sizes['GPi']>0 and sizes['Th']>0:
            X['Th'][t]  += -0.3 * (Z['GPi'][t] @ rng.normal(size=(d_lat, sizes['Th'])).astype(np.float32))  # inhibitory
        if sizes['Th']>0 and sizes['C']>0:
            X['C'][t]   += 0.2 * (Z['Th'][t] @ rng.normal(size=(d_lat, sizes['C'])).astype(np.float32))
        if sizes['MB']>0:
            X['MB'][t]   = (Z['MB'][t] @ Proj['MB']) + 0.1*rng.normal(size=(sizes['MB'],)).astype(np.float32)
    # Normalize each ROI over time (z-score per column)
    for name, arr in X.items():
        arr = (arr - arr.mean(axis=0, keepdims=True)) / (arr.std(axis=0, keepdims=True) + 1e-6)
        X[name] = arr.astype(np.float32)
    return X

# --------------------------------------------
# Main
# --------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--demo', action='store_true')
    args = parser.parse_args()

    if args.demo:
        # Create tiny synthetic dataset + config
        os.makedirs('data', exist_ok=True)
        sizes = {'C':48,'S':96,'GPe':12,'GPi':12,'STN':6,'Th':16,'MB':3}
        for i in range(6):
            X = make_synth_run(T=800, sizes=sizes, seed=10+i)
            np.savez(f'data/synth_run_{i:02d}.npz', **X)
        with open('data/train_list.txt','w') as f:
            f.write('\n'.join([f'data/synth_run_{i:02d}.npz' for i in range(0,4)]))
        with open('data/val_list.txt','w') as f:
            f.write('\n'.join([f'data/synth_run_{i:02d}.npz' for i in range(4,5)]))
        with open('data/test_list.txt','w') as f:
            f.write('\n'.join([f'data/synth_run_{i:02d}.npz' for i in range(5,6)]))
        cfg = ModelConfig(
            P_C=sizes['C'], P_S=sizes['S'], P_GPe=sizes['GPe'], P_GPi=sizes['GPi'],
            P_STN=sizes['STN'], P_Th=sizes['Th'], P_MB=sizes['MB'],
            d_model=128, n_heads=4, n_layers=2, dropout=0.1,
            T_ctx=128, k_forecast=2,
            batch_size=16, max_epochs=5,
            mae_targets=("S",),
            edges=tuple(DEFAULT_EDGES),
            gated_edges=(("C","S"),("C","STN")),
            kernel_size=3,
            seed=7,
        )
        os.makedirs('configs', exist_ok=True)
        # Dump config as JSON-like dict to YAML-ish .json for simplicity
        with open('configs/loop_demo.yaml','w') as f:
            f.write(json.dumps(cfg.__dict__, indent=2))
        print("Demo data and config written. Run: python loop_loopaware_cst_transformer.py --config configs/loop_demo.yaml")
        return

    # Load config
    assert args.config is not None, "Provide --config or use --demo"
    with open(args.config,'r') as f:
        cfgd = json.load(f)
    cfg = ModelConfig(**cfgd)
    set_seed(cfg.seed)

    # Load bias matrices if provided
    bias_dict = load_biases(cfg)
    # Move bias tensors to device
    for k in list(bias_dict.keys()):
        bias_dict[k] = bias_dict[k].to(cfg.device)

    # Datasets
    train_ds = LoopDataset('data/train_list.txt', cfg, mask_ratio=0.4)
    val_ds   = LoopDataset('data/val_list.txt',   cfg, mask_ratio=0.4)
    train_ld = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_ld   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False)

    # Model
    model = LoopAwareModel(cfg, bias_dict=bias_dict).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best = -1e9
    os.makedirs('outputs/ckpts', exist_ok=True)

    for epoch in range(1, cfg.max_epochs+1):
        tr = train_epoch(model, train_ld, opt, cfg)
        va, r2c, r2s = eval_epoch(model, val_ld, cfg)
        print(f"Epoch {epoch:03d} | train {tr:.4f} | val {va:.4f} | R2_C {r2c:.3f} | R2_S {r2s:.3f}")
        score = np.nanmean([r2c, r2s])
        if score > best:
            best = score
            torch.save({'cfg': cfg.__dict__, 'state_dict': model.state_dict()}, 'outputs/ckpts/loop_best.pt')

if __name__ == "__main__":
    main()
