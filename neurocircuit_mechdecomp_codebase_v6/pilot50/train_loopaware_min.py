#!/usr/bin/env python
"""Minimal pilot trainer to demonstrate feasibility on ~50 subjects.

This is intentionally lightweight: it trains a tiny loop-aware model for a few epochs
and saves a checkpoint + resolved config.

The full mechanistic Transformer-SSM model in the paper is in `neurocircuit/models/`.
This pilot trainer is meant to mirror the existing "Pilot50" workflow you outlined.
"""

import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


def infer_sizes_from_npzlist(list_path: str):
    with open(list_path, "r") as f:
        first = f.readline().strip()
    d = np.load(first)
    return {f"P_{k}": d[k].shape[1] for k in d.files}


@dataclass
class ModelConfig:
    # token sizes filled automatically
    P_C: int = 0
    P_S: int = 0
    P_Th: int = 0
    P_A: int = 0
    P_H: int = 0
    P_MB: int = 0

    d_model: int = 96
    dropout: float = 0.1

    T_ctx: int = 64
    k_forecast: int = 1

    batch_size: int = 8
    max_epochs: int = 5

    w_recon: float = 1.0
    mae_targets: tuple = ("S",)

    # symbolic edge list, used for interpretability (not enforced in this minimal stub)
    edges: tuple = ()
    gated_edges: tuple = ()
    inhibitory_edges: tuple = ()


class LoopDataset(Dataset):
    """Loads NPZ tokens and returns windowed sequences."""

    def __init__(self, npz_list_path: str, cfg: ModelConfig, mask_ratio: float = 0.4):
        self.cfg = cfg
        self.mask_ratio = mask_ratio
        with open(npz_list_path, "r") as f:
            self.paths = [ln.strip() for ln in f if ln.strip()]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        d = np.load(self.paths[idx])
        X = {}
        for k in d.files:
            arr = d[k]  # (T, P)
            if arr.shape[0] < self.cfg.T_ctx + 1:
                raise ValueError(f"Time series too short: {arr.shape[0]} < {self.cfg.T_ctx+1}")
            start = np.random.randint(0, arr.shape[0] - (self.cfg.T_ctx + 1))
            seg = arr[start : start + self.cfg.T_ctx + 1]
            X[k] = torch.from_numpy(seg.astype(np.float32))  # (T_ctx+1, P)

        # Last frame masked input (simple feature dropout)
        last_frame_masked = {}
        for k, v in X.items():
            x_last = v[-1].clone()
            mask = torch.rand_like(x_last) < self.mask_ratio
            x_last[mask] = 0.0
            last_frame_masked[k] = x_last

        return {"X": X, "last_frame_masked": last_frame_masked}


class LoopAwareTinyModel(nn.Module):
    """Tiny model: concatenate all tokens -> MLP -> reconstruct selected targets."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        # infer total input dimensionality from cfg
        sizes = {k: getattr(cfg, f"P_{k}", 0) for k in ["C", "S", "Th", "A", "H", "MB"]}
        self.keys = [k for k, v in sizes.items() if v and v > 0]
        in_dim = sum(sizes[k] for k in self.keys)

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
        )

        # Heads per target family
        self.heads = nn.ModuleDict()
        for k in cfg.mae_targets:
            Pk = getattr(cfg, f"P_{k}")
            self.heads[k] = nn.Linear(cfg.d_model, Pk)

    def forward(self, X_last_masked: dict):
        feats = torch.cat([X_last_masked[k] for k in self.keys], dim=-1)
        z = self.encoder(feats)
        out = {k: self.heads[k](z) for k in self.heads}
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True, help="JSON config template")
    ap.add_argument("--list", required=True, help="Text file: one NPZ path per line")
    ap.add_argument("--out", required=True, help="Output directory")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    with open(args.cfg, "r") as f:
        cfgd = json.load(f)

    sizes = infer_sizes_from_npzlist(args.list)
    cfgd.update(sizes)
    cfgd.setdefault("d_model", 96)
    cfgd.setdefault("dropout", 0.1)
    cfgd.setdefault("T_ctx", 64)
    cfgd.setdefault("k_forecast", 1)
    cfgd.setdefault("batch_size", args.batch)
    cfgd.setdefault("max_epochs", args.epochs)
    cfgd.setdefault("mae_targets", ["S"])

    with open(os.path.join(args.out, "cfg_resolved.json"), "w") as f:
        json.dump(cfgd, f, indent=2)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = ModelConfig(
        P_C=cfgd.get("P_C", 0),
        P_S=cfgd.get("P_S", 0),
        P_Th=cfgd.get("P_Th", 0),
        P_A=cfgd.get("P_A", 0),
        P_H=cfgd.get("P_H", 0),
        P_MB=cfgd.get("P_MB", 0),
        d_model=cfgd.get("d_model", 96),
        dropout=cfgd.get("dropout", 0.1),
        T_ctx=cfgd.get("T_ctx", 64),
        k_forecast=cfgd.get("k_forecast", 1),
        batch_size=cfgd.get("batch_size", args.batch),
        max_epochs=cfgd.get("max_epochs", args.epochs),
        mae_targets=tuple(cfgd.get("mae_targets", ["S"])),
    )

    train_ds = LoopDataset(args.list, cfg, mask_ratio=0.4)
    train_ld = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)

    model = LoopAwareTinyModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_loss = 1e9

    for epoch in range(cfg.max_epochs):
        model.train()
        ep_loss = 0.0
        for batch in tqdm(train_ld, desc=f"train ep{epoch+1}"):
            X = batch["X"]
            last_masked = batch["last_frame_masked"]

            # move to device
            X = {k: v.to(device) for k, v in X.items()}
            last_masked = {k: v.to(device) for k, v in last_masked.items()}

            pred = model(last_masked)

            loss = 0.0
            for k in cfg.mae_targets:
                loss = loss + torch.nn.functional.l1_loss(pred[k], X[k][:, -1, :])

            opt.zero_grad()
            loss.backward()
            opt.step()

            ep_loss += float(loss.item())

        ep_loss /= max(1, len(train_ld))
        print({"epoch": epoch + 1, "train_loss": ep_loss})

        if ep_loss < best_loss:
            best_loss = ep_loss
            torch.save({"state_dict": model.state_dict(), "cfg": cfgd, "train_loss": best_loss}, os.path.join(args.out, "loop_pilot_best.pt"))

    print("[OK] training done; best train_loss", best_loss)


if __name__ == "__main__":
    main()
