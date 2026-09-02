# Source atlases for cbgtc_v1

Do not use per-subject/per-run clustering. Build from fixed group atlases.

## Tian source-resolution policy

Use the **3T** Tian atlas for these 3T fMRI cohorts.

- **Striatum:** Tian S3. This is the finest 3T striatal subdivision (20 total striatal parcels); the striatal labels are unchanged in S4.
- **Thalamus:** Tian S4 (16 parcels).
- **Hippocampus:** Tian S4 (10 parcels).
- **Amygdala:** Tian S4 labels (4 parcels); the amygdala is not subdivided further from S3 to S4 at 3T.

Official labels live in the Tian repository under `Group-Parcellation/3T/Subcortex-Only`.

## HCP-space build (`MNI152NLin6Asym`)

1. **Tian 2020 Melbourne Subcortex Atlas, 3T Scale III**
   - `Tian_Subcortex_S3_3T.nii`
   - `Tian_Subcortex_S3_3T_label.txt`
   - used for striatum only.

2. **Tian 2020 Melbourne Subcortex Atlas, 3T Scale IV**
   - `Tian_Subcortex_S4_3T.nii`
   - `Tian_Subcortex_S4_3T_label.txt`
   - used for thalamus, amygdala and hippocampus; defines the target grid.

3. **Schaefer2018 100 parcels, 7 networks**
   - Prefer an explicit `MNI152NLin6Asym` representation and matching label table.

4. **CIT168 v1.1**
   - Use the MNI152NLin6Asym projection.
   - Source labels used: GPe, GPi, VeP, STN, SNc, SNr, VTA.

## fMRIPrep-space build (`MNI152NLin2009cAsym`)

Use corresponding Tian/Schaefer/CIT168 representations in MNI152NLin2009cAsym and the same `cbgtc_v1_roi_ids.txt`. Integer output labels and ROI ordering must remain identical across spaces.
