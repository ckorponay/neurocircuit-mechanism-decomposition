#!/usr/bin/env python
"""Run the minimal method pipeline on a single ABCD resting-state BOLD run.

This script is designed for *preliminary grant figures*:
- Converts a 4D BOLD NIfTI + ROI atlases/masks -> NPZ token time-series
- Builds a simple anatomical mask
- Runs a forward pass through the dual-head model to produce routing tensor π
- Writes a CSV of interpretable lag/routing summaries

NOTE: This is a lightweight *prototype runner* for early validation, not a finalized
training/evaluation pipeline.
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import torch

from pilot50.vol_to_npz import main as vol_to_npz_main
from neurocircuit.models.model import NeurocircuitMechDecomp
from tools.build_anatomy_mask import build_mask


def load_tokens(npz_path: str):
    dat = np.load(npz_path)
    groups = list(dat.keys())
    mats = [dat[g] for g in groups]  # each (T, K)
    token_sizes = {g: dat[g].shape[1] for g in groups}
    X = np.concatenate(mats, axis=1)  # (T, R)
    return groups, token_sizes, X


def summarize_pi(pi: torch.Tensor, groups: list[str], token_sizes: dict[str, int], out_csv: str):
    """Summarize routing tensor into edge weights + lag stats.

    pi: (B, T, R_tgt, R_src, L)
    We average over batch/time and output per edge:
      influence = sum_l pi
      peak_lag = argmax_l pi
      centroid = E[l]
      concentration = 1/Var(l)
    """
    # average over batch and time
    pim = pi.mean(dim=(0, 1))  # (R_tgt, R_src, L)

    R_tgt, R_src, L = pim.shape
    lags = torch.arange(L, device=pim.device, dtype=pim.dtype)

    # influence
    infl = pim.sum(dim=-1)  # (tgt, src)

    # peak lag
    peak = pim.argmax(dim=-1)  # (tgt, src)

    # centroid + variance
    centroid = (pim * lags).sum(dim=-1) / (pim.sum(dim=-1) + 1e-8)
    second = (pim * (lags ** 2)).sum(dim=-1) / (pim.sum(dim=-1) + 1e-8)
    var = torch.clamp(second - centroid**2, min=1e-8)
    conc = 1.0 / var

    # map indices back to group labels
    idx_to_label = []
    for g in groups:
        for i in range(token_sizes[g]):
            idx_to_label.append(f"{g}{i:03d}")

    rows = []
    for tgt in range(R_tgt):
        for src in range(R_src):
            rows.append([
                idx_to_label[src],
                idx_to_label[tgt],
                float(infl[tgt, src].cpu()),
                int(peak[tgt, src].cpu()),
                float(centroid[tgt, src].cpu()),
                float(conc[tgt, src].cpu()),
            ])

    header = "src,tgt,influence,peak_lag_tr,centroid_lag_tr,concentration\n"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w") as f:
        f.write(header)
        for r in rows:
            f.write(",".join(map(str, r)) + "\n")


def main():
    ap = argparse.ArgumentParser()

    # Inputs
    ap.add_argument("--func", required=True, help="ABCD 4D BOLD NIfTI")
    ap.add_argument("--atlas-cortex", required=True)
    ap.add_argument("--atlas-striatum", required=True)
    ap.add_argument("--atlas-thalamus", default=None)

    ap.add_argument("--mask-amygdala", nargs="*", default=None)
    ap.add_argument("--mask-hippocampus", nargs="*", default=None)
    ap.add_argument("--mask-gpe", nargs="*", default=None)
    ap.add_argument("--mask-gpi", nargs="*", default=None)
    ap.add_argument("--mask-stn", nargs="*", default=None)
    ap.add_argument("--mask-midbrain", nargs="*", default=None)

    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-lag", type=int, default=13)
    ap.add_argument("--d-model", type=int, default=128)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    token_npz = os.path.join(args.outdir, "tokens.npz")
    meta_dir = os.path.join(args.outdir, "meta")

    # 1) Convert to tokens (call the converter module)
    import sys
    sys.argv = [
        "vol_to_npz",
        "--func", args.func,
        "--atlas-cortex", args.atlas_cortex,
        "--atlas-striatum", args.atlas_striatum,
        "--out", token_npz,
        "--meta-dir", meta_dir,
    ]
    if args.atlas_thalamus:
        sys.argv += ["--atlas-thalamus", args.atlas_thalamus]

    def add_masks(flag, val):
        if val:
            sys.argv.append(flag)
            sys.argv += list(val)

    add_masks("--mask-amygdala", args.mask_amygdala)
    add_masks("--mask-hippocampus", args.mask_hippocampus)
    add_masks("--mask-gpe", args.mask_gpe)
    add_masks("--mask-gpi", args.mask_gpi)
    add_masks("--mask-stn", args.mask_stn)
    add_masks("--mask-midbrain", args.mask_midbrain)

    vol_to_npz_main()

    # 2) Load token matrix
    groups, token_sizes, X = load_tokens(token_npz)  # X is (T,R)

    # 3) Build anatomy mask
    anat = build_mask(token_sizes)  # (R,R) boolean

    # 4) Model forward
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeurocircuitMechDecomp(n_regions=X.shape[1], d_model=args.d_model, n_lags=args.max_lag).to(device)

    y = torch.from_numpy(X.T[None, ...]).float().to(device)  # (B=1,R,T)
    anat_mask = torch.from_numpy(anat).to(device)

    with torch.no_grad():
        out = model(y, anat_mask=anat_mask, max_lag=args.max_lag)

    pi = out["transformer"]["pi"].detach().cpu()

    # 5) Summarize
    summary_csv = os.path.join(args.outdir, "routing_lag_summary.csv")
    summarize_pi(pi, groups, token_sizes, summary_csv)

    print(f"[OK] wrote {summary_csv}")


if __name__ == "__main__":
    main()
