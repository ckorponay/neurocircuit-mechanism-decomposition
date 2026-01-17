#!/usr/bin/env python
"""Run a *single* ABCD resting-state run through the mechanism decomposition model.

This script is designed for **preliminary grant figures**: it converts an fMRI run
into tokens (ROI time series), runs a forward pass of the dual-head model, and
exports summary routing/lag metrics.

It does NOT perform full end-to-end training (which you should do for final results).
Instead, it provides a reproducible "smoke + feature extraction" pathway.

Example:
  python pilot50/run_single_subject_abcd.py \
    --func sub-XXXX_task-rest_bold.nii.gz \
    --atlas-cortex cortex_parcels.nii.gz \
    --atlas-striatum striatum_kparc.nii.gz \
    --outdir out_abcd_single \
    --max-lag 13
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import torch

from pilot50.vol_to_npz import extract_label_ts
import nibabel as nib

from neurocircuit.models.model import NeurocircuitMechDecomp
from tools.build_anatomy_mask import build_mask, token_order


def concat_npz_to_y(npz: dict[str, np.ndarray]) -> tuple[np.ndarray, dict[str, int], list[str]]:
    """Concatenate token matrices into y with shape (R,T) and return token sizes."""
    token_sizes = {k: npz[k].shape[1] for k in npz.keys()}
    order = token_order(token_sizes)
    mats = [npz[k] for k in order]  # each (T, n_k)
    X = np.concatenate(mats, axis=1)  # (T, R)
    y = X.T.astype(np.float32)        # (R, T)
    return y, token_sizes, order


def routing_metrics(pi: torch.Tensor) -> dict[str, np.ndarray]:
    """Compute influence and lag summary metrics from π.

    pi shape: (B,T,R_tgt,R_src,L)
    Returns arrays for each (src,tgt).
    """
    with torch.no_grad():
        # Average across time for stable summaries
        pi_mean = pi.mean(dim=1)  # (B,R_tgt,R_src,L)
        pi0 = pi_mean[0]          # (R_tgt,R_src,L)

        influence = pi0.sum(dim=-1)  # (R_tgt,R_src)

        lags = torch.arange(pi0.shape[-1], device=pi0.device, dtype=pi0.dtype)
        # centroid lag
        centroid = (pi0 * lags).sum(dim=-1) / (pi0.sum(dim=-1) + 1e-12)

        # peak lag
        peak = pi0.argmax(dim=-1).to(torch.float32)

        # concentration ~ inverse variance
        var = (pi0 * (lags - centroid.unsqueeze(-1)) ** 2).sum(dim=-1) / (pi0.sum(dim=-1) + 1e-12)
        concentration = 1.0 / (var + 1e-6)

        return {
            "influence": influence.cpu().numpy(),
            "peak_lag": peak.cpu().numpy(),
            "centroid_lag": centroid.cpu().numpy(),
            "concentration": concentration.cpu().numpy(),
        }


def save_edge_table(out_csv: str, metrics: dict[str, np.ndarray], region_labels: list[str]):
    R = len(region_labels)
    with open(out_csv, "w") as f:
        f.write("src\ttgt\tinfluence\tpeak_lag\tcentroid_lag\tconcentration\n")
        for tgt in range(R):
            for src in range(R):
                f.write(
                    f"{region_labels[src]}\t{region_labels[tgt]}\t"
                    f"{metrics['influence'][tgt,src]:.6g}\t"
                    f"{metrics['peak_lag'][tgt,src]:.3g}\t"
                    f"{metrics['centroid_lag'][tgt,src]:.6g}\t"
                    f"{metrics['concentration'][tgt,src]:.6g}\n"
                )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--func", required=True, help="4D resting-state BOLD NIfTI")
    ap.add_argument("--atlas-cortex", required=True, help="Integer cortical parcels NIfTI")
    ap.add_argument("--atlas-striatum", required=True, help="Integer striatum K-parcellation NIfTI")
    ap.add_argument("--atlas-thalamus", default=None)

    ap.add_argument("--outdir", required=True)
    ap.add_argument("--max-lag", type=int, default=13)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--rank-b", type=int, default=3)
    ap.add_argument("--allow-within", action="store_true")

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    func = nib.load(args.func)
    npz = {
        "C": extract_label_ts(func, nib.load(args.atlas_cortex)),
        "S": extract_label_ts(func, nib.load(args.atlas_striatum)),
    }
    if args.atlas_thalamus:
        npz["Th"] = extract_label_ts(func, nib.load(args.atlas_thalamus))

    y_np, token_sizes, order = concat_npz_to_y(npz)

    # Build region labels for CSV readability
    region_labels = []
    for k in order:
        for i in range(token_sizes[k]):
            region_labels.append(f"{k}{i:03d}")

    anat = build_mask(token_sizes, allow_within=args.allow_within)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y = torch.from_numpy(y_np).unsqueeze(0).to(device)  # (B=1,R,T)
    anat_mask = torch.from_numpy(anat).to(device)

    model = NeurocircuitMechDecomp(n_regions=y.shape[1], d_model=args.d_model, n_lags=args.max_lag).to(device)
    model.eval()

    out = model(y, anat_mask=anat_mask, max_lag=args.max_lag)

    pi = out["transformer"]["pi"]
    drive = out["transformer"]["drive"][0].detach().cpu().numpy()  # (R,T)

    mets = routing_metrics(pi)

    np.save(os.path.join(args.outdir, "drive.npy"), drive)
    save_edge_table(os.path.join(args.outdir, "routing_metrics.tsv"), mets, region_labels)

    print(f"[OK] wrote: {args.outdir}/routing_metrics.tsv and drive.npy")


if __name__ == "__main__":
    main()
