# ABCD resting-state quick run (single subject)

This is the minimal protocol to generate **preliminary mechanism features** from one ABCD resting-state run.

## Inputs

- A 4D BOLD NIfTI file (ABCD rs-fMRI), e.g. `sub-XXXX_ses-XX_task-rest_bold.nii.gz`
- ROI masks/parcellations in the same space as the BOLD (MNI or subject space)
  - striatum mask (required for K-parcellation)
  - optional thalamus / pallidum / midbrain masks

## Step 1 — Make striatum K-parcellation

```bash
python pilot50/make_striatum_kparc.py \
  --striatum-mask masks/striatum_mask.nii.gz \
  --k 10 \
  --out masks/striatum_k10_labels.nii.gz
```

## Step 2 — Convert volume to NPZ tokens

```bash
python pilot50/vol_to_npz.py \
  --bold /path/to/ABCD_rest_bold.nii.gz \
  --masks-dir masks \
  --striatum-kparc masks/striatum_k10_labels.nii.gz \
  --tr 0.8 \
  --out out_npz/sub-XXXX_rest.npz
```

## Step 3 — Run the model and export features

```bash
python tools/run_npz_and_export_features.py \
  --npz out_npz/sub-XXXX_rest.npz \
  --anat-mask masks/anatomy_mask.npy \
  --outdir out_features/sub-XXXX
```

### Outputs

- `routing_pi.npy` : routing tensor π(i,ℓ→j)
- `drive_u.npy` : drive signals û(t)
- `routing_metrics.csv` : edge strength + lag centroid/peak + concentration per edge

> For grant prelims, the key deliverable is `routing_metrics.csv` (summary of directed influences + timing).
