#!/usr/bin/env python
"""Make a fixed striatum K-parcellation in MNI space for cross-subject consistency.

Based on the Pilot50 minimal working bundle.

Inputs
------
--striatal-mask : binary NIfTI mask of bilateral striatum (MNI grid)
--K             : number of parcels
--out           : output integer-label NIfTI (labels 1..K)

Notes
-----
Clustering uses voxel MNI coordinates in mm.
"""

import argparse
import nibabel as nib
import numpy as np
from sklearn.cluster import KMeans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--striatal-mask", required=True, help="Binary NIfTI of bilateral striatum in MNI grid")
    ap.add_argument("--K", type=int, default=200)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    img = nib.load(args.striatal_mask)
    data = img.get_fdata()
    vox = np.argwhere(data > 0)
    if vox.size == 0:
        raise SystemExit("Striatal mask has no voxels")

    aff = img.affine
    xyz = nib.affines.apply_affine(aff, vox)

    km = KMeans(n_clusters=args.K, n_init=10, random_state=args.seed).fit(xyz)

    labels = np.zeros(data.shape, dtype=np.int32)
    labels[vox[:, 0], vox[:, 1], vox[:, 2]] = km.labels_.astype(np.int32) + 1

    nib.save(nib.Nifti1Image(labels, img.affine, img.header), args.out)
    print(f"[OK] wrote {args.out} with {args.K} parcels")


if __name__ == "__main__":
    main()
