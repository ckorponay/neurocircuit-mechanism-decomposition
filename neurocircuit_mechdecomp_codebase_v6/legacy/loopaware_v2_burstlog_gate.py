"""
Loop-Aware Cortico–Basal Ganglia–Thalamic Transformer — v2 (burst + gate logging)
---------------------------------------------------------------------------------
This variant logs, for each cross-edge at the last frame of the context window:
  • attention weights:      attn_logs['cross']["SRC->TGT"]            → (B,H,N_tgt,N_src)
  • value norms:            attn_logs['cross_vnorm']["SRC->TGT"]       → (B,N_src)
  • MB gate scalar:         attn_logs['cross_gate']["SRC->TGT"]        → (B,) or None
  • MB per-head gates:      attn_logs['cross_gate_heads']["SRC->TGT"]  → (B,H) or None

The MB (midbrain; e.g., VTA/SN) gate is a multiplicative **gain** applied to the value stream
of edges listed in cfg.gated_edges (e.g., C→S, Th→S, A→S, H→S). It models dopaminergic-like
moment-to-moment modulation of striatal responsiveness to excitatory inputs.

API is compatible with the existing training/eval scripts; just import this module instead
of the non-logging version to get gate logs in attn_logs.
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

NODE_SETS = [
    "C",   # Cortex parcels
    "S",   # Striatum supervoxels
    "A",   # Amygdala tokens
    "H",   # Hippocampus tokens
    "GPe", # External Globus Pallidus
    "GPi", # Internal Globus Pallidus
    "STN", # Subthalamic Nucleus
    "Th",  # Thalamus
    "MB",  # Midbrain (VTA/SN)
]

DEFAULT_EDGES = [
    ("C","S"), ("A","S"), ("H","S"), ("Th","S"),
    ("C","STN"),
    ("S","GPe"), ("S","GPi"), ("STN","GPi"),
    ("GPe","S"),
    ("GPi","Th"), ("Th","C"),
]

@dataclass
class ModelConfig:
    P_C: int = 64
    P_S: int = 128
    P_A: int = 0
    P_H: int = 0
    P_GPe: int = 16
    P_GPi: int = 16
    P_STN: int = 8
    P_Th: int = 24
    P_MB: int = 0

    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1

    T_ctx: int = 128
    k_forecast: int = 2

    lr: float = 3e-4
    weight_decay: float = 1e-2
    batch_size: int = 8
    max_epochs: int = 10
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    w_mae_spatial: float = 1.0
    w_forecast: float = 1.0

    mae_targets: Tuple[str, ...] = ("S",)

    edges: Tuple[Tuple[str,str], ...] = tuple(DEFAULT_EDGES)

    bias_paths: Dict[str, str] = None

    gated_edges: Tuple[Tuple[str,str], ...] = (("C","S"),("Th","S"),("A","S"),("H","S"),("C","STN"))

    inhibitory_edges: Tuple[Tuple[str,str], ...] = tuple()  # keep empty unless you want priors

    kernel_size: int = 3

    seed: int = 7

# --------------------------------------------
# Utils & Dataset
# --------------------------------------------

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class LoopDataset(Dataset):
    def __init__(self, file_list: str, cfg: ModelConfig, mask_ratio: float = 0.4):
        super().__init__(); self.cfg = cfg; self.mask_ratio = mask_ratio
        with open(file_list, 'r') as f:
            self.files = [ln.strip() for ln in f if ln.strip()]
        self.index = []
        for fi, path in enumerate(self.files):
            with np.load(path) as z:
                T = None
                for k in NODE_SETS:
                    if k in z:
                        T = z[k].shape[0] if T is None else min(T, z[k].shape[0])
                if T is None: continue
                for t_end in range(cfg.T_ctx + cfg.k_forecast, T):
                    self.index.append((fi, t_end))
    def __len__(self): return len(self.index)
    def __getitem__(self, idx):
        fi, t_end = self.index[idx]; path = self.files[fi]
        with np.load(path) as z:
            t0 = t_end - (self.cfg.T_ctx + self.cfg.k_forecast)
            t_ctx_end = t0 + self.cfg.T_ctx
            X_dict = {}
            for name in NODE_SETS:
                if name in z:
                    X = z[name].astype(np.float32)[t0:t_ctx_end]
                    X_dict[name] = torch.from_numpy(X)
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
            future_dict = {}
            for name in ["C","S"]:
                if name in z:
                    arr = z[name].astype(np.float32)
                    fut = arr[t_ctx_end + self.cfg.k_forecast - 1]
                    future_dict[name] = torch.from_numpy(fut)
        return {'X': X_dict,'last_frame_true': last_frame_true,'last_frame_masked': last_frame_masked,'last_mask': last_mask,'future': future_dict}

# --------------------------------------------
# Attention blocks
# --------------------------------------------
class MHAWithBias(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
    def forward(self, q, k, v, additive_bias: Optional[torch.Tensor] = None, attn_mask: Optional[torch.Tensor] = None):
        if additive_bias is not None:
            attn_mask = additive_bias if attn_mask is None else attn_mask + additive_bias
        out, A = self.mha(q, k, v, attn_mask=attn_mask, need_weights=True, average_attn_weights=False)
        return out, A

class SpatialBlock(nn.Module):
    def __init__(self, P: int, d_model: int, n_heads: int, dropout: float = 0.1,
                 bias_matrix: Optional[torch.Tensor] = None, name: str = ""):
        super().__init__(); self.P = P; self.name = name
        self.inp = nn.Linear(1, d_model)
        self.pe = nn.Parameter(torch.zeros(P, d_model)); nn.init.normal_(self.pe, std=0.01)
        self.mha = MHAWithBias(d_model, n_heads, dropout)
        self.bias = bias_matrix
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(4*d_model, d_model))
    def forward(self, x_t: torch.Tensor):
        if self.P == 0: return x_t[..., None], torch.zeros(x_t.size(0), 1, 0, 0, device=x_t.device)
        h = self.inp(x_t.unsqueeze(-1)) + self.pe
        q = self.ln1(h)
        out, A = self.mha(q, q, q, additive_bias=self.bias)
        h = h + out; h = h + self.ff(h)
        return h, A

class CrossBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 bias_matrix: Optional[torch.Tensor] = None,
                 gated: bool = False,
                 inhibitory_value: bool = False,
                 name: str = "edge",
                 kernel_size: int = 3):
        super().__init__(); self.name = name
        self.mha = MHAWithBias(d_model, n_heads, dropout)
        self.bias = bias_matrix
        self.lnq = nn.LayerNorm(d_model)
        self.lnk = nn.LayerNorm(d_model)
        self.value_proj = nn.Linear(d_model, d_model)
        self.inhibitory_value = inhibitory_value
        self.ff = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(4*d_model, d_model))
        self.gated = gated
        if self.gated:
            self.mb_to_gate = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, n_heads))
        padding = (kernel_size - 1) // 2
        self.temporal_kernel = nn.Conv1d(d_model, d_model, kernel_size, padding=padding, groups=d_model)
        with torch.no_grad():
            self.temporal_kernel.weight.zero_(); center = padding
            for c in range(d_model): self.temporal_kernel.weight[c, c, center] = 1.0
            if self.temporal_kernel.bias is not None: self.temporal_kernel.bias.zero_()
    def forward(self, tgt_feats_t: torch.Tensor, src_seq: torch.Tensor, mb_feats_t: Optional[torch.Tensor] = None):
        B, T, N_src, d = src_seq.shape
        # reset gate logs each call
        self.last_gate_heads = None
        self.last_gate_scalar = None
        src_seq_resh = rearrange(src_seq, 'b t n d -> b n d t')
        src_seq_f = self.temporal_kernel(src_seq_resh)
        src_last = src_seq_f[..., -1]
        q = self.lnq(tgt_feats_t); k = self.lnk(src_last)
        v = self.value_proj(src_last)
        if self.inhibitory_value: v = -F.softplus(v)
        add_bias = self.bias
        if self.gated and (mb_feats_t is not None) and mb_feats_t.numel() > 0:
            # Midbrain (MB) gating: compute per-head gate from MB features at this TR
            mb_pool = mb_feats_t.mean(dim=1)               # (B, d_model) pooled MB
            gates = self.mb_to_gate(mb_pool)               # (B, H) pre-sigmoid
            gate_heads = 1.0 + torch.sigmoid(gates)        # (B, H) > 1 means up‑gain
            gate_scalar = gate_heads.mean(dim=1, keepdim=True)  # (B,1) average gain across heads
            # Log for analysis
            self.last_gate_heads = gate_heads.detach()      # (B, H)
            self.last_gate_scalar = gate_scalar.detach().squeeze(1)  # (B,)
            # Apply scalar gain to value stream (broadcast over tokens and channels)
            v = v * gate_scalar.unsqueeze(1).unsqueeze(-1)  # (B, N_src, d) * (B,1,1)
        v_norm = torch.norm(v, dim=-1)  # (B, N_src)
        out, A = self.mha(q, k, v, additive_bias=add_bias)
        y = tgt_feats_t + out; y = y + self.ff(y)
        return y, A, v_norm

class TemporalBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, T_max: int = 4096):
        super().__init__()
        self.mha = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.pe_time = nn.Parameter(torch.zeros(T_max, d_model)); nn.init.normal_(self.pe_time, std=0.01)
        self.ln = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(4*d_model, d_model))
    def forward(self, seq_feats: torch.Tensor):
        B, T, N, d = seq_feats.shape
        x = seq_feats + self.pe_time[:T]
        S = rearrange(x, 'b t n d -> (b n) t d')
        causal = torch.full((T, T), float('-inf'), device=seq_feats.device)
        causal = torch.triu(causal, diagonal=1)
        out, W = self.mha(self.ln(S), self.ln(S), self.ln(S), attn_mask=causal, need_weights=True, average_attn_weights=False)
        S = S + out; S = S + self.ff(S)
        Y = rearrange(S, '(b n) t d -> b t n d', b=B, n=N)
        return Y, W

# --------------------------------------------
# Full model
# --------------------------------------------
class LoopAwareModel(nn.Module):
    def __init__(self, cfg: ModelConfig, bias_dict: Optional[Dict[str, torch.Tensor]] = None):
        super().__init__(); self.cfg = cfg
        d, H = cfg.d_model, cfg.n_heads
        self.sizes = {'C': cfg.P_C, 'S': cfg.P_S, 'A': cfg.P_A, 'H': cfg.P_H,'GPe': cfg.P_GPe, 'GPi': cfg.P_GPi, 'STN': cfg.P_STN, 'Th': cfg.P_Th, 'MB': cfg.P_MB}
        self.spatial = nn.ModuleDict()
        for name, P in self.sizes.items():
            if P > 0:
                bias = None
                if bias_dict is not None and f"{name}<->{name}" in bias_dict:
                    bias = bias_dict[f"{name}<->{name}"]
                self.spatial[name] = SpatialBlock(P, d, H, cfg.dropout, bias_matrix=bias, name=name)
        self.edges = list(cfg.edges)
        self.cross = nn.ModuleList()
        for (src, tgt) in self.edges:
            key = f"{src}->{tgt}"; bias = None
            if bias_dict is not None and key in bias_dict: bias = bias_dict[key]
            gated = (src, tgt) in cfg.gated_edges; inhibitory = (src, tgt) in cfg.inhibitory_edges
            self.cross.append(((src,tgt), CrossBlock(d, H, cfg.dropout, bias_matrix=bias, gated=gated, inhibitory_value=inhibitory, name=key, kernel_size=cfg.kernel_size)))
        self.temporal = nn.ModuleDict()
        for name, P in self.sizes.items():
            if P > 0: self.temporal[name] = TemporalBlock(d, H, cfg.dropout, T_max=4096)
        self.readout_last = nn.ModuleDict()
        for tgt in cfg.mae_targets:
            if self.sizes.get(tgt, 0) > 0: self.readout_last[tgt] = nn.Linear(d, 1)
        self.readout_future = nn.ModuleDict()
        for name in ["C","S"]:
            if self.sizes.get(name, 0) > 0: self.readout_future[name] = nn.Linear(d, 1)

    def forward(self, X: Dict[str, torch.Tensor], last_masked: Dict[str, torch.Tensor]):
        B = next(iter(X.values())).size(0); T = next(iter(X.values())).size(1)
        X_mod = {k: v.clone() for k,v in X.items()}
        for tgt, masked in last_masked.items():
            if tgt in X_mod: X_mod[tgt][:, -1] = masked
        feats = {k: [] for k in X_mod.keys()}; A_spatial = {k: [] for k in X_mod.keys()}
        for t in range(T):
            for name, seq in X_mod.items():
                P = seq.size(2)
                if P == 0: continue
                h_t, A_t = self.spatial[name](seq[:, t])
                feats[name].append(h_t); A_spatial[name].append(A_t)
        for name in feats.keys():
            if len(feats[name])>0:
                feats[name] = torch.stack(feats[name], dim=1)
                A_spatial[name] = torch.stack(A_spatial[name], dim=1)
        A_cross = {}; V_cross = {}; G_cross = {}; G_heads = {}
        for (src,tgt), layer in self.cross:
            if (src not in feats) or (tgt not in feats): continue
            tgt_last = feats[tgt][:, -1]; src_seq  = feats[src]
            mb_t = feats.get('MB', None); mb_last = mb_t[:, -1] if (mb_t is not None and mb_t.numel()>0) else None
            y_t, A, vnorm = layer(tgt_last, src_seq, mb_last)
            feats[tgt][:, -1] = y_t
            key = f"{src}->{tgt}"
            A_cross[key] = A
            V_cross[key] = vnorm
            G_cross[key] = getattr(layer, 'last_gate_scalar', None)
            G_heads[key] = getattr(layer, 'last_gate_heads', None)
        A_temporal = {}
        for name, seq in feats.items():
            if isinstance(seq, list) or seq is None or seq.numel()==0: continue
            Y, W = self.temporal[name](seq)
            feats[name] = Y; A_temporal[name] = W
        recon_last, forecast = {}, {}
        for tgt, head in self.readout_last.items():
            if tgt in feats:
                H_last = feats[tgt][:, -1]; x_hat = head(H_last).squeeze(-1); recon_last[tgt] = x_hat
        for name, head in self.readout_future.items():
            if name in feats:
                H_last = feats[name][:, -1]; yhat = head(H_last).squeeze(-1); forecast[name] = yhat
        attn_logs = {
            'spatial': A_spatial,
            'cross': A_cross,
            'temporal': A_temporal,
            'cross_vnorm': V_cross,
            'cross_gate': G_cross,
            'cross_gate_heads': G_heads,
        }
        return recon_last, forecast, attn_logs

# --------------------------------------------
# Losses & training helpers
# --------------------------------------------

def masked_mse(pred: torch.Tensor, tgt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    eps = 1e-8; num = (mask * (pred - tgt)**2).sum(dim=1); den = mask.sum(dim=1) + eps
    return (num / den).mean()

@torch.no_grad()
def r2_score(pred: torch.Tensor, tgt: torch.Tensor) -> float:
    yhat, y = pred, tgt
    ss_res = ((y - yhat)**2).sum(dim=1)
    ss_tot = ((y - y.mean(dim=1, keepdim=True))**2).sum(dim=1) + 1e-8
    r2 = 1.0 - ss_res/ss_tot
    return float(r2.mean().item())

def train_epoch(model: LoopAwareModel, loader: DataLoader, opt, cfg: ModelConfig):
    model.train(); tot=0.0
    for batch in tqdm(loader, desc='train', leave=False):
        X = {k: v.to(cfg.device) for k,v in batch['X'].items()}
        last_true = {k: v.to(cfg.device) for k,v in batch['last_frame_true'].items()}
        last_masked = {k: v.to(cfg.device) for k,v in batch['last_frame_masked'].items()}
        last_mask = {k: v.to(cfg.device) for k,v in batch['last_mask'].items()}
        future = {k: v.to(cfg.device) for k,v in batch['future'].items()}
        recon_last, forecast, _ = model(X, last_masked)
        loss_sp = 0.0
        for tgt in cfg.mae_targets:
            if tgt in recon_last: loss_sp = loss_sp + masked_mse(recon_last[tgt], last_true[tgt], last_mask[tgt])
        loss_fc = 0.0
        for name in ["C","S"]:
            if name in forecast and name in future: loss_fc = loss_fc + F.mse_loss(forecast[name], future[name])
        loss = cfg.w_mae_spatial*loss_sp + cfg.w_forecast*loss_fc
        opt.zero_grad(); loss.backward(); opt.step()
        tot += float(loss.item()) * next(iter(X.values())).size(0)
    return tot / len(loader.dataset)

@torch.no_grad()
def eval_epoch(model: LoopAwareModel, loader: DataLoader, cfg: ModelConfig):
    model.eval(); tot=0.0; r2_c=[]; r2_s=[]
    for batch in tqdm(loader, desc='eval', leave=False):
        X = {k: v.to(cfg.device) for k,v in batch['X'].items()}
        last_true = {k: v.to(cfg.device) for k,v in batch['last_frame_true'].items()}
        last_masked = {k: v.to(cfg.device) for k,v in batch['last_frame_masked'].items()}
        last_mask = {k: v.to(cfg.device) for k,v in batch['last_mask'].items()}
        future = {k: v.to(cfg.device) for k,v in batch['future'].items()}
        recon_last, forecast, _ = model(X, last_masked)
        loss_sp = 0.0
        for tgt in cfg.mae_targets:
            if tgt in recon_last: loss_sp = loss_sp + masked_mse(recon_last[tgt], last_true[tgt], last_mask[tgt])
        loss_fc = 0.0
        if 'C' in forecast and 'C' in future:
            loss_fc = loss_fc + F.mse_loss(forecast['C'], future['C']); r2_c.append(r2_score(forecast['C'], future['C']))
        if 'S' in forecast and 'S' in future:
            loss_fc = loss_fc + F.mse_loss(forecast['S'], future['S']); r2_s.append(r2_score(forecast['S'], future['S']))
        loss = cfg.w_mae_spatial*loss_sp + cfg.w_forecast*loss_fc
        tot += float(loss.item()) * next(iter(X.values())).size(0)
    r2c = float(np.mean(r2_c)) if r2_c else float('nan')
    r2s = float(np.mean(r2_s)) if r2_s else float('nan')
    return tot/len(loader.dataset), r2c, r2s

# --------------------------------------------
# Bias loader & minimal CLI
# --------------------------------------------

def load_biases(cfg: ModelConfig) -> Dict[str, torch.Tensor]:
    bias_dict = {}
    if cfg.bias_paths is None: return bias_dict
    for key, path in cfg.bias_paths.items():
        arr = np.load(path).astype(np.float32); bias_dict[key] = torch.from_numpy(arr)
    return bias_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    args = parser.parse_args()
    with open(args.config,'r') as f: cfgd = json.load(f)
    cfg = ModelConfig(**cfgd); random.seed(cfg.seed)
    bias_dict = load_biases(cfg)
    for k in list(bias_dict.keys()): bias_dict[k] = bias_dict[k].to(cfg.device)
    train_ds = LoopDataset('data/train_list.txt', cfg, mask_ratio=0.4)
    val_ds   = LoopDataset('data/val_list.txt',   cfg, mask_ratio=0.4)
    train_ld = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_ld   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False)
    model = LoopAwareModel(cfg, bias_dict=bias_dict).to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best = -1e9; os.makedirs('outputs/ckpts', exist_ok=True)
    for epoch in range(1, cfg.max_epochs+1):
        tr = train_epoch(model, train_ld, opt, cfg)
        va, r2c, r2s = eval_epoch(model, val_ld, cfg)
        print(f"Epoch {epoch:03d} | train {tr:.4f} | val {va:.4f} | R2_C {r2c:.3f} | R2_S {r2s:.3f}")
        score = np.nanmean([r2c, r2s])
        if score > best:
            best = score
            torch.save({'cfg': cfg.__dict__, 'state_dict': model.state_dict()}, 'outputs/ckpts/loop_v2_burstlog_gate_best.pt')
