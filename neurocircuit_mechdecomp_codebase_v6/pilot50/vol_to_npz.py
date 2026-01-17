#!/usr/bin/env python
"""Convert volumetric 4D rest fMRI NIfTI -> NPZ tokens.

This script extracts mean timeseries for:
- cortical parcels (integer atlas)
- striatal K-parcellation (integer labels)
- optional integer thalamus atlas
- optional binary masks for region families (amygdala, hippocampus, GPe, GPi, STN, midbrain)

Based on the Pilot50 minimal working bundle.
"""

import argparse
import os
import numpy as np
import nibabel as nib


def extract_label_ts(func_img: nib.Nifti1Image, label_img: nib.Nifti1Image) -> np.ndarray:
    f = func_img.get_fdata()  # (X,Y,Z,T)
    if f.ndim != 4:
        raise ValueError("func must be 4D")

    lab = label_img.get_fdata().astype(np.int32)
    L = int(lab.max())

    if func_img.shape[:3] != label_img.shape[:3]:
        raise ValueError("func and label image grids differ; please resample beforehand")

    T = f.shape[3]
    out = np.zeros((T, L), dtype=np.float32)

    for ell in range(1, L + 1):
        mask = (lab == ell)
        if mask.sum() == 0:
            out[:, ell - 1] = 0.0
        else:
            vox = f[mask]  # (Nvox, T)
            out[:, ell - 1] = vox.mean(axis=0)

    return out


def extract_mask_ts(func_img: nib.Nifti1Image, mask_img: nib.Nifti1Image) -> np.ndarray:
    f = func_img.get_fdata()
    m = mask_img.get_fdata() > 0

    if func_img.shape[:3] != mask_img.shape[:3]:
        raise ValueError("func and mask image grids differ; please resample beforehand")

    vox = f[m]
    if vox.size == 0:
        return np.zeros((f.shape[3], 1), dtype=np.float32)

    return vox.mean(axis=0, keepdims=True).astype(np.float32).T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--func", required=True)
    ap.add_argument("--atlas-cortex", required=True, help="Integer label NIfTI for cortical parcels")
    ap.add_argument("--atlas-striatum", required=True, help="Integer label NIfTI for K-parcellated striatum")
    ap.add_argument("--atlas-thalamus", default=None)

    ap.add_argument("--mask-amygdala", nargs="*", default=None)
    ap.add_argument("--mask-hippocampus", nargs="*", default=None)
    ap.add_argument("--mask-gpe", nargs="*", default=None)
    ap.add_argument("--mask-gpi", nargs="*", default=None)
    ap.add_argument("--mask-stn", nargs="*", default=None)
    ap.add_argument("--mask-midbrain", nargs="*", default=None, help="VTA/SN masks")

    ap.add_argument("--out", required=True)
    ap.add_argument("--meta-dir", default=None, help="Directory to write *_index.tsv label maps")

    args = ap.parse_args()

    func = nib.load(args.func)

    C = extract_label_ts(func, nib.load(args.atlas_cortex))
    S = extract_label_ts(func, nib.load(args.atlas_striatum))

    X = {"C": C, "S": S}
    index_maps = {
        "C": [f"C_{i}" for i in range(C.shape[1])],
        "S": [f"S_{i}" for i in range(S.shape[1])],
    }

    if args.atlas_thalamus:
        Th = extract_label_ts(func, nib.load(args.atlas_thalamus))
        X["Th"] = Th
        index_maps["Th"] = [f"Th_{i}" for i in range(Th.shape[1])]

    for name, masks in [
        ("A", args.mask_amygdala),
        ("H", args.mask_hippocampus),
        ("GPe", args.mask_gpe),
        ("GPi", args.mask_gpi),
        ("STN", args.mask_stn),
        ("MB", args.mask_midbrain),
    ]:
        if masks:
            cols = []
            for mpath in masks:
                cols.append(extract_mask_ts(func, nib.load(mpath)))
            X[name] = np.concatenate(cols, axis=1)
            index_maps[name] = [f"{name}_{i}" for i in range(X[name].shape[1])]

    np.savez_compressed(args.out, **X)
    print(f"[OK] wrote {args.out} keys={list(X.keys())} shapes={ {k: v.shape for k, v in X.items()} }")

    if args.meta_dir:
        os.makedirs(args.meta_dir, exist_ok=True)
        for k, labels in index_maps.items():
            with open(os.path.join(args.meta_dir, f"{k}_index.tsv"), "w") as f:
                f.write("token\tlabel\n")
                for i, lab in enumerate(labels):
                    f.write(f"{i}\t{lab}\n")
        print(f"[OK] wrote index maps to {args.meta_dir}")


if __name__ == "__main__":
    main()
