#!/usr/bin/env bash
set -euo pipefail

# ==== EDIT paths ====
SUBJ_TSV="lists/subjects_volrest.tsv"   # (sub\t/path) full list
PILOT_N=50                               # take first 50
OUTROOT="outputs/pilot50"
NPZDIR="data/npz_rest"
METADIR="data/npz_rest/meta"
CFG="configs/rest_pilot.json"

# Atlases / masks (same grid as func)
ATLAS_CORTEX="/path/to/Schaefer2018_400Parcels_2mm.nii.gz"
STR_MASK="/path/to/striatal_mask_MNI2mm.nii.gz"
THAL_ATLAS="/path/to/thalamus_parcels_2mm.nii.gz"  # optional
AMY_MASKS=(/path/to/amygdala_L_2mm.nii.gz /path/to/amygdala_R_2mm.nii.gz)
HIP_MASKS=(/path/to/hippocampus_L_2mm.nii.gz /path/to/hippocampus_R_2mm.nii.gz)
MB_MASKS=(/path/to/VTA_2mm.nii.gz /path/to/SN_2mm.nii.gz)

mkdir -p lists "$NPZDIR" "$METADIR" configs "$OUTROOT" masks

# 1) Make fixed striatum K-parcellation (once)
python pilot50/make_striatum_kparc.py \
  --striatal-mask "$STR_MASK" --K 200 \
  --out masks/striatum_K200_2mm.nii.gz

# 2) Take first N subjects
head -n $PILOT_N "$SUBJ_TSV" > lists/subjects_pilot50.tsv

# 3) Convert each subject to NPZ
rm -f lists/pilot50_rest.txt
while IFS=$'\t' read -r SID FP; do
  OUTNPZ="$NPZDIR/${SID}_rest.npz"
  python pilot50/vol_to_npz.py \
    --func "$FP" \
    --atlas-cortex "$ATLAS_CORTEX" \
    --atlas-striatum masks/striatum_K200_2mm.nii.gz \
    --atlas-thalamus "$THAL_ATLAS" \
    --mask-amygdala "${AMY_MASKS[@]}" \
    --mask-hippocampus "${HIP_MASKS[@]}" \
    --mask-midbrain "${MB_MASKS[@]}" \
    --out "$OUTNPZ" \
    --meta-dir "$METADIR"
  echo "$OUTNPZ" >> lists/pilot50_rest.txt
  echo "[OK] ${SID}"
done < lists/subjects_pilot50.tsv

# 4) Minimal training run (few epochs)
python - <<'PYIN'
import json, os
cfg = {
  "d_model": 96,
  "dropout": 0.1,
  "T_ctx": 64,
  "k_forecast": 1,
  "mae_targets": ["S"],
}
os.makedirs('configs', exist_ok=True)
with open('configs/rest_pilot.json', 'w') as f:
  json.dump(cfg, f, indent=2)
print('[OK] wrote configs/rest_pilot.json')
PYIN

python pilot50/train_loopaware_min.py \
  --cfg "$CFG" \
  --list lists/pilot50_rest.txt \
  --out "$OUTROOT" \
  --epochs 5 --batch 8 --lr 1e-3

echo "[DONE] Pilot50 conversion + tiny training finished."
