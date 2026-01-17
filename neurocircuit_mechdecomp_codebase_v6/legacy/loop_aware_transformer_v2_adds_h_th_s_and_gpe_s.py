"""
HCP-YA Converter: CIFTI → Cortical Parcels + Striatum Supervoxels (+ Amygdala/Hippocampus) → NPZ
-----------------------------------------------------------------------------------------------
This utility prepares HCP(-YA) runs for the loop-aware attention models in your canvas.
It reads an fMRI **dtseries.nii** (CIFTI), parcellates the **cortex (C)** using a
provided **dlabel** atlas (e.g., Glasser360, Schaefer400), extracts **subcortical volume**,
clusters the **striatum** into **K supervoxels (S)**, and (optionally) clusters **amygdala (A)**
and **hippocampus (H)** into small supervoxel sets—then writes a per-run **.npz** with arrays
shaped `(T, P_group)` ready for training:

  • 'C'  : (T, P_C)  – cortex parcels (z-scored per parcel)
  • 'S'  : (T, P_S)  – striatum supervoxels (z-scored per supervoxel)
  • 'A'  : (T, P_A)  – amygdala supervoxels (optional)
  • 'H'  : (T, P_H)  – hippocampus supervoxels (optional)

Optionally, if you provide masks and Ks, it can also cluster:
  • 'GPi', 'GPe', 'STN', 'Th', 'MB'  → added to the NPZ if supplied (as before).

It also writes helper coordinate files when requested:
  • data/striatum_coords.npy : (P_S, 3) MNI centers of S supervoxels
  • data/roi_coords.npy      : (P_C, 3) (optional; if you provide a CSV of parcel centroids)

Dependencies (install once):
    pip install numpy nibabel scikit-learn tqdm pandas

Recommended: **Connectome Workbench** in PATH (wb_command). We use it to:
  - parcellate cortex (robust) and
  - extract subcortical volume from CIFTI.
If wb_command is unavailable, the script falls back to a pure-Python parcellation
that works when the dlabel matches the dtseries grayordinates exactly (may be slow
and can fail for some atlases). For subcortex, you can also pass a pre-extracted
NIfTI 4D volume with --subcortex-nii.

Examples
--------
# Process one HCP-YA resting run (Schaefer400; K=1000 S; add A/H)
python hcp_to_npz_converter.py make-npz \
  --dtseries /path/HCP/100307/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas_MSMAll.dtseries.nii \
  --dlabel   /path/atlases/Schaefer2018_400Parcels_7Networks_order.dlabel.nii \
  --subcortex-nii /path/HCP/100307/subcortex_4d.nii.gz \
  --striatum-mask /path/masks/striatum_mask_MNI.nii.gz \
  --K-S 1000 \
  --Amy-mask /path/masks/amygdala_mask_MNI.nii.gz --K-A 8 \
  --Hipp-mask /path/masks/hippocampus_mask_MNI.nii.gz --K-H 16 \
  --out data/sub-100307_REST1_LR.npz \
  --save-striatum-coords

# Batch a folder of runs into NPZ and write a manifest (adds A/H if masks provided)
python hcp_to_npz_converter.py batch \
  --runs-root /path/HCP \
  --pattern "rfMRI_REST*_Atlas_MSMAll.dtseries.nii" \
  --dlabel /path/atlases/Schaefer2018_400Parcels_7Networks_order.dlabel.nii \
  --striatum-mask /path/masks/striatum_mask_MNI.nii.gz \
  --K-S 1000 \
  --Amy-mask /path/masks/amygdala_mask_MNI.nii.gz --K-A 8 \
  --Hipp-mask /path/masks/hippocampus_mask_MNI.nii.gz --K-H 16 \
  --outdir data/npz \
  --manifest data/train_list.txt

Notes
-----
• **Z-scoring:** All outputs are column-wise z-scored (mean 0; std 1 within run).
• **Coordinates:** Cortical centroids can be provided via --roi-centroids-csv if you have them.
• **Spaces:** All masks (striatum/A/H/etc.) must be in the **same space** as the subcortex 4D volume
  extracted from the dtseries.
• **Family-aware splits:** Use the manifest to build splits that keep HCP families separate.
"""

from __future__ import annotations
import os, sys, json, glob, shutil, argparse, tempfile, subprocess
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np
import nibabel as nib
from sklearn.cluster import KMeans
from tqdm import tqdm

# -------------------------------
# Utilities
# -------------------------------

def have_wb() -> bool:
    return shutil.which('wb_command') is not None


def run(cmd: List[str]):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        print("
[wb_command output]
" + p.stdout)
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    return p.stdout


def zscore_columns(X: np.ndarray) -> np.ndarray:
    X = X.astype(np.float32)
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-6
    return (X - mu) / sd

# -------------------------------
# Cortex parcellation
# -------------------------------

def parcellate_cortex(dtseries_path: str, dlabel_path: str) -> np.ndarray:
    """Return C (T, P) parcel timeseries from dtseries and dlabel.
    Prefers wb_command; falls back to Python if needed.
    """
    if have_wb():
        with tempfile.TemporaryDirectory() as td:
            ptseries = os.path.join(td, 'C.ptseries.nii')
            txt = os.path.join(td, 'C.txt')
            run(['wb_command', '-cifti-parcellate', dtseries_path, dlabel_path, 'COLUMN', ptseries])
            run(['wb_command', '-cifti-convert', '-to-text', ptseries, txt])
            C = np.loadtxt(txt)
            C = C if C.shape[0] > C.shape[1] else C.T  # ensure (T,P)
            return zscore_columns(C)
    # Fallback: pure Python (may fail if grayordinate order mismatches)
    dt = nib.load(dtseries_path)  # Cifti2Image
    dl = nib.load(dlabel_path)
    data = dt.get_fdata().T  # (T, G)
    lab_data = dl.get_fdata()
    labels = np.argmax(lab_data, axis=0) if lab_data.ndim == 2 else lab_data.squeeze().astype(int)
    P = int(labels.max()+1)
    C = np.zeros((data.shape[0], P), np.float32)
    for p in range(P):
        idx = labels == p
        if idx.sum() == 0: continue
        C[:, p] = data[:, idx].mean(axis=1)
    return zscore_columns(C)

# -------------------------------
# Subcortex extraction
# -------------------------------

def extract_subcortex_4d(dtseries_path: str, out_nii_path: str):
    if not have_wb():
        raise RuntimeError("wb_command is required to extract subcortex from CIFTI. Alternatively pass --subcortex-nii.")
    run(['wb_command', '-cifti-separate', dtseries_path, 'COLUMN', '-volume-all', out_nii_path])

# -------------------------------
# Supervoxel builder
# -------------------------------

def build_supervoxels(subcortex_nii: str, mask_nii: str, K: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (S (T,K), centers (K,3), labels per voxel)."""
    vol = nib.load(subcortex_nii)
    X = vol.get_fdata()  # (X,Y,Z,T)
    T = X.shape[-1]
    mask = nib.load(mask_nii).get_fdata().astype(bool)
    if mask.shape != X.shape[:3]:
        raise ValueError("Mask shape does not match subcortex volume shape.")
    vox_ts = X[mask].reshape((-1, T)).T  # (T, V)
    vox_ts = zscore_columns(vox_ts)
    if vox_ts.shape[1] < K:
        raise ValueError(f"Requested K={K} > number of masked voxels V={vox_ts.shape[1]}.")
    km = KMeans(n_clusters=K, n_init=10, random_state=0)
    labels = km.fit_predict(vox_ts.T)
    S = np.stack([vox_ts[:, labels==k].mean(axis=1) for k in range(K)], axis=1)
    ijk = np.array(np.where(mask)).T
    xyz = nib.affines.apply_affine(vol.affine, ijk)
    centers = np.vstack([xyz[labels==k].mean(axis=0) for k in range(K)])
    return zscore_columns(S), centers.astype(np.float32), labels

# -------------------------------
# Optional: generic nucleus clustering (A/H/GPi/GPe/STN/Th/MB)
# -------------------------------

def cluster_region(subcortex_nii: str, region_mask_nii: str, K: int) -> Optional[np.ndarray]:
    vol = nib.load(subcortex_nii)
    X = vol.get_fdata()
    T = X.shape[-1]
    mask = nib.load(region_mask_nii).get_fdata().astype(bool)
    if mask.shape != X.shape[:3]:
        raise ValueError("Region mask shape does not match subcortex volume shape.")
    V = mask.sum()
    if V == 0: return None
    K = min(max(1, K), V)
    vox_ts = X[mask].reshape((-1, T)).T
    vox_ts = zscore_columns(vox_ts)
    if K == 1:
        R = vox_ts.mean(axis=1, keepdims=True)
    else:
        km = KMeans(n_clusters=K, n_init=10, random_state=0)
        labels = km.fit_predict(vox_ts.T)
        R = np.stack([vox_ts[:, labels==k].mean(axis=1) for k in range(K)], axis=1)
    return zscore_columns(R.astype(np.float32))

# -------------------------------
# NPZ writer
# -------------------------------

def write_npz(out_path: str, arrays: Dict[str, np.ndarray]):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez(out_path, **arrays)
    print(f"[OK] wrote {out_path} with keys: {list(arrays.keys())}")

# -------------------------------
# CLI commands
# -------------------------------

def cmd_make_npz(args):
    # 1) Cortex parcellation
    C = parcellate_cortex(args.dtseries, args.dlabel)

    # 2) Subcortex 4D volume
    if args.subcortex_nii is None:
        with tempfile.TemporaryDirectory() as td:
            subvol = os.path.join(td, 'subcortex_4d.nii.gz')
            extract_subcortex_4d(args.dtseries, subvol)
            subcortex_path = subvol
    else:
        subcortex_path = args.subcortex_nii

    # 3) Striatum supervoxels (required)
    S, S_centers, _ = build_supervoxels(subcortex_path, args.striatum_mask, args.K_S)
    arrays = {'C': C.astype(np.float32), 'S': S.astype(np.float32)}

    # 4) Optional limbic: Amygdala, Hippocampus
    if args.Amy_mask and args.K_A > 0:
        R = cluster_region(subcortex_path, args.Amy_mask, args.K_A)
        if R is not None: arrays['A'] = R.astype(np.float32)
    if args.Hipp_mask and args.K_H > 0:
        R = cluster_region(subcortex_path, args.Hipp_mask, args.K_H)
        if R is not None: arrays['H'] = R.astype(np.float32)

    # 5) Optional other nuclei (existing options)
    if args.GPi_mask and args.K_GPi>0:
        R = cluster_region(subcortex_path, args.GPi_mask, args.K_GPi)
        if R is not None: arrays['GPi'] = R.astype(np.float32)
    if args.GPe_mask and args.K_GPe>0:
        R = cluster_region(subcortex_path, args.GPe_mask, args.K_GPe)
        if R is not None: arrays['GPe'] = R.astype(np.float32)
    if args.STN_mask and args.K_STN>0:
        R = cluster_region(subcortex_path, args.STN_mask, args.K_STN)
        if R is not None: arrays['STN'] = R.astype(np.float32)
    if args.Th_mask and args.K_Th>0:
        R = cluster_region(subcortex_path, args.Th_mask, args.K_Th)
        if R is not None: arrays['Th'] = R.astype(np.float32)
    if args.MB_mask and args.K_MB>0:
        R = cluster_region(subcortex_path, args.MB_mask, args.K_MB)
        if R is not None: arrays['MB'] = R.astype(np.float32)

    # 6) Save NPZ
    write_npz(args.out, arrays)

    # 7) Save coords if requested
    if args.save_striatum_coords:
        outc = os.path.join(os.path.dirname(args.out), 'striatum_coords.npy')
        np.save(outc, S_centers.astype(np.float32))
        print(f"[OK] wrote {outc}")

    # 8) Optionally save user-provided ROI centroids CSV as roi_coords.npy (format: label,x,y,z)
    if args.roi_centroids_csv and os.path.exists(args.roi_centroids_csv):
        import pandas as pd
        df = pd.read_csv(args.roi_centroids_csv)
        cols = [c for c in df.columns if c.lower() in ('x','y','z') or c.lower().endswith(('_x','_y','_z'))]
        if len(cols) >= 3:
            coords = df[[cols[0], cols[1], cols[2]]].values.astype(np.float32)
            outc = os.path.join(os.path.dirname(args.out), 'roi_coords.npy')
            np.save(outc, coords)
            print(f"[OK] wrote {outc}")
        else:
            print("[WARN] roi_centroids_csv found but could not parse X/Y/Z columns; skipped roi_coords.npy")


def cmd_batch(args):
    os.makedirs(args.outdir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(args.runs_root, '**', args.pattern), recursive=True))
    if len(paths) == 0:
        print("[ERR] No dtseries files matched. Check --runs-root and --pattern.")
        sys.exit(1)
    manifest_lines = []
    for dt in tqdm(paths, desc='batch'):
        subj = None
        parts = dt.replace('\','/').split('/')
        for p in parts:
            if p.startswith('sub-'):
                subj = p
            elif p.isdigit() and len(p) in (6,7):
                subj = f"sub-{p}"
        runname = os.path.basename(dt).replace('_Atlas_MSMAll.dtseries.nii','')
        out = os.path.join(args.outdir, f"{subj}_{runname}.npz") if subj else os.path.join(args.outdir, f"{runname}.npz")
        with tempfile.TemporaryDirectory() as td:
            if args.subcortex_nii:
                subvol = args.subcortex_nii
            else:
                subvol = os.path.join(td, 'subcortex_4d.nii.gz')
                extract_subcortex_4d(dt, subvol)
            C = parcellate_cortex(dt, args.dlabel)
            S, S_centers, _ = build_supervoxels(subvol, args.striatum_mask, args.K_S)
            arrays = {'C': C.astype(np.float32), 'S': S.astype(np.float32)}
            # Optional A/H
            if args.Amy_mask and args.K_A>0:
                R = cluster_region(subvol, args.Amy_mask, args.K_A)
                if R is not None: arrays['A'] = R.astype(np.float32)
            if args.Hipp_mask and args.K_H>0:
                R = cluster_region(subvol, args.Hipp_mask, args.K_H)
                if R is not None: arrays['H'] = R.astype(np.float32)
            # Optional other nuclei
            if args.GPi_mask and args.K_GPi>0:
                R = cluster_region(subvol, args.GPi_mask, args.K_GPi)
                if R is not None: arrays['GPi'] = R.astype(np.float32)
            if args.GPe_mask and args.K_GPe>0:
                R = cluster_region(subvol, args.GPe_mask, args.K_GPe)
                if R is not None: arrays['GPe'] = R.astype(np.float32)
            if args.STN_mask and args.K_STN>0:
                R = cluster_region(subvol, args.STN_mask, args.K_STN)
                if R is not None: arrays['STN'] = R.astype(np.float32)
            if args.Th_mask and args.K_Th>0:
                R = cluster_region(subvol, args.Th_mask, args.K_Th)
                if R is not None: arrays['Th'] = R.astype(np.float32)
            if args.MB_mask and args.K_MB>0:
                R = cluster_region(subvol, args.MB_mask, args.K_MB)
                if R is not None: arrays['MB'] = R.astype(np.float32)

            write_npz(out, arrays)
            manifest_lines.append(out)
            if args.save_striatum_coords:
                np.save(os.path.join(args.outdir, f"{os.path.splitext(os.path.basename(out))[0]}_striatum_coords.npy"), S_centers.astype(np.float32))
    if args.manifest:
        with open(args.manifest,'w') as f:
            f.write('
'.join(manifest_lines))
        print(f"[OK] wrote manifest with {len(manifest_lines)} lines to {args.manifest}")

# -------------------------------
# Main
# -------------------------------

def main():
    ap = argparse.ArgumentParser(description='HCP-YA CIFTI→NPZ converter (cortex parcels + striatum + amygdala/hippocampus supervoxels)')
    sub = ap.add_subparsers(dest='cmd', required=True)

    p1 = sub.add_parser('make-npz', help='Convert a single dtseries run to NPZ')
    p1.add_argument('--dtseries', required=True, help='Path to *_Atlas_MSMAll.dtseries.nii')
    p1.add_argument('--dlabel',   required=True, help='Parcel atlas .dlabel.nii (Glasser, Schaefer, etc.)')
    p1.add_argument('--subcortex-nii', default=None, help='Optional: pre-extracted 4D subcortex NIfTI')
    p1.add_argument('--striatum-mask', required=True, help='Striatum mask NIfTI in same space as subcortex volume')
    p1.add_argument('--K-S', type=int, default=1000, help='Number of striatum supervoxels')
    # New limbic inputs
    p1.add_argument('--Amy-mask', dest='Amy_mask', default=None, help='Amygdala mask NIfTI in subcortex space')
    p1.add_argument('--K-A', dest='K_A', type=int, default=0, help='# of amygdala supervoxels')
    p1.add_argument('--Hipp-mask', dest='Hipp_mask', default=None, help='Hippocampus mask NIfTI in subcortex space')
    p1.add_argument('--K-H', dest='K_H', type=int, default=0, help='# of hippocampus supervoxels')
    # Optional nuclei (existing)
    p1.add_argument('--GPi-mask', default=None)
    p1.add_argument('--K-GPi', type=int, default=0)
    p1.add_argument('--GPe-mask', default=None)
    p1.add_argument('--K-GPe', type=int, default=0)
    p1.add_argument('--STN-mask', default=None)
    p1.add_argument('--K-STN', type=int, default=0)
    p1.add_argument('--Th-mask', default=None)
    p1.add_argument('--K-Th', type=int, default=0)
    p1.add_argument('--MB-mask', default=None)
    p1.add_argument('--K-MB', type=int, default=0)
    # Coords
    p1.add_argument('--save-striatum-coords', action='store_true', help='Write striatum_coords.npy alongside NPZ')
    p1.add_argument('--roi-centroids-csv', default=None, help='Optional CSV with cortical parcel centroids (x,y,z) to save as roi_coords.npy')
    p1.add_argument('--out', required=True, help='Output NPZ path')
    p1.set_defaults(func=cmd_make_npz)

    p2 = sub.add_parser('batch', help='Batch-convert many dtseries to NPZ and write a manifest list')
    p2.add_argument('--runs-root', required=True, help='Root folder to search (recursively)')
    p2.add_argument('--pattern', required=True, help='Glob pattern for CIFTI dtseries (e.g., "rfMRI_REST*_Atlas_MSMAll.dtseries.nii")')
    p2.add_argument('--dlabel', required=True)
    p2.add_argument('--subcortex-nii', default=None)
    p2.add_argument('--striatum-mask', required=True)
    p2.add_argument('--K-S', type=int, default=1000)
    # New limbic inputs
    p2.add_argument('--Amy-mask', dest='Amy_mask', default=None)
    p2.add_argument('--K-A', dest='K_A', type=int, default=0)
    p2.add_argument('--Hipp-mask', dest='Hipp_mask', default=None)
    p2.add_argument('--K-H', dest='K_H', type=int, default=0)
    # Optional nuclei (existing)
    p2.add_argument('--GPi-mask', default=None)
    p2.add_argument('--K-GPi', type=int, default=0)
    p2.add_argument('--GPe-mask', default=None)
    p2.add_argument('--K-GPe', type=int, default=0)
    p2.add_argument('--STN-mask', default=None)
    p2.add_argument('--K-STN', type=int, default=0)
    p2.add_argument('--Th-mask', default=None)
    p2.add_argument('--K-Th', type=int, default=0)
    p2.add_argument('--MB-mask', default=None)
    p2.add_argument('--K-MB', type=int, default=0)
    p2.add_argument('--outdir', required=True)
    p2.add_argument('--manifest', default=None, help='Write list of NPZ paths here')
    p2.add_argument('--save-striatum-coords', action='store_true')
    p2.set_defaults(func=cmd_batch)

    args = ap.parse_args()

    if not have_wb() and args.subcortex_nii is None:
        print('[WARN] wb_command not found. You must provide --subcortex-nii (4D NIfTI of subcortex).')
    args.func(args)

if __name__ == '__main__':
    main()
