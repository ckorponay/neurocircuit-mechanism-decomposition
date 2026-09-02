# HCP-YA hemodynamic adapter: RAPIDTIDE + rsHRF -> NMD

The model expects all quantities in the **same fixed ROI identity/order** as the neural input. Production preprocessing is implemented in:

```text
neurocircuit/scripts/prepare_hemodynamics.py
```

## Recommended HCP order

RAPIDTIDE's current documentation specifically notes an HCP/FIX special case: estimate the systemic regressor and voxel delays from the **minimally processed** BOLD, then use `--denoisesourcefile` to apply the final sLFO regression to the **FIX-denoised** BOLD. This lets the systemic estimation see signal that FIX can attenuate while producing an rsHRF input on top of the conventional HCP FIX processing.

Conceptually:

```text
HCP minimally processed volumetric BOLD
            |
            +---- RAPIDTIDE: estimate s(t), delta(x)
            |
HCP FIX BOLD +---- final RAPIDTIDE regression
            |
            +---- cleaned BOLD -> fixed ROI reduction -> rsHRF

original FIX ROI BOLD + fixed RAPIDTIDE + fixed rsHRF
            |
            +---- NMD measurement-aware latent-state inference
```

The RAPIDTIDE-cleaned 4-D BOLD is **not** the primary NMD observed signal. It is an intermediate used to estimate rsHRF and an ablation/benchmark. It can be deleted after the derived HRF and compact ROI-level products have been validated.

## One-command RAPIDTIDE + reduction stage

Example for one HCP-YA run:

```tcsh
python -m neurocircuit.scripts.prepare_hemodynamics \
  --rapidtide-source /path/to/minimally_processed_bold.nii.gz \
  --denoise-source /path/to/FIX_bold.nii.gz \
  --rapidtide-prefix /path/to/derivatives/rapidtide/100307_REST1_LR \
  --atlas /path/to/fixed_nmd_label_atlas.nii.gz \
  --roi-map /path/to/nmd_roi_labels.tsv \
  --roi-schema /path/to/nmd_roi_ids.txt \
  --tr-seconds 0.72 \
  --out-dir /path/to/derivatives/nmd_hemodynamics/100307_REST1_LR \
  --run-rapidtide \
  --filterband lfo \
  --searchrange -7.5 15 \
  --ampthresh 0.15 \
  --spatialfilt 3 \
  --despecklepasses 4 \
  --passes 3 \
  --nprocs 2
```

The wrapper calls RAPIDTIDE with `--denoising`, discovers the BIDS-style outputs, and writes:

```text
systemic_waveform.npy                 [T]
vascular_delay_seconds.npy            [R]
vascular_amplitude.npy                [R]
rapidtide_r2.npy                       [R]  # if lfofilterR exists
rshrf_input_cleaned_roi_TxR.npy       [T,R]
rshrf_input_cleaned_roi_TxR.tsv       [T,R]
hemodynamics_provenance.json
```

It also records the exact RAPIDTIDE command and all source/output paths in the provenance JSON.

## RAPIDTIDE fields

Current RAPIDTIDE sources used are:

- systemic waveform: `*_desc-refinedmovingregressor_timeseries.tsv[.gz]`;
- vascular delay: `*_desc-maxtimerefined_map.nii[.gz]`, in seconds;
- vascular amplitude: `*_desc-lfofilterCoeff_map.nii[.gz]`;
- fit R²: `*_desc-lfofilterR2_map.nii[.gz]`;
- fit correlation (fallback/QC): `*_desc-lfofilterR_map.nii[.gz]`;
- temporary cleaned signal for rsHRF: `*_desc-lfofilterCleaned_bold.nii[.gz]`.

NMD stores `rapidtide_r2` from RAPIDTIDE's documented voxelwise `lfofilterR2` map. If that map is absent in an older run, the adapter falls back to squaring the voxelwise `lfofilterR` map before ROI reduction. Do **not** use `maxcorr` as the observation-model amplitude. `lfofilterCoeff` is the fitted regression coefficient and corresponds to the alpha term in the generative model.

Default ROI reductions are:

- delay: median;
- fit coefficient: median;
- fit R²: median of voxelwise `lfofilterR2`;
- cleaned BOLD: voxel mean per ROI.

A sensitivity option is available for R2-weighted mean delay:

```tcsh
--delay-summary r2_weighted_mean
```

but the plain median is the production default so the vascular-delay phenotype is not circularly dominated by fit strength.

## Fixed ROI definition

`--roi-map` must contain the integer atlas value for every NMD ROI, in exactly the same order as `--roi-schema`:

```text
roi_id  label
C_0     1
C_1     2
...
S_0     101
```

The wrapper refuses to silently reorder the mapping. It also checks spatial dimensions and NIfTI affines before reduction. Empty ROIs cause a hard failure.

The demo HCP converter in the original repository can refit KMeans supervoxels independently per run. Do **not** use that behavior for cross-subject mechanistic training. Use a fixed atlas or a once-fit/frozen reference parcellation.

## rsHRF bridge

rsHRF accepts image data or observation-by-voxel matrices. The wrapper therefore exports the RAPIDTIDE-cleaned fixed-ROI matrix as `[T,R]`. Run rsHRF on that matrix using the selected reproducible rsHRF workflow and export the resulting HRF kernels as `[R,K]` (or `[K,R]`; the wrapper detects/transposes the latter).

Then rerun `prepare_hemodynamics` **without** `--run-rapidtide`, adding:

```tcsh
--hrf-kernel /path/to/rshrf_hrf.npy
```

The final output is:

```text
hemodynamics.npz
```

with:

```text
hrf_kernel                 [R,K]  unit-L1 HRF shape
hrf_gain                   [R]    separated HRF magnitude
systemic_waveform          [T]
vascular_delay_seconds     [R]
vascular_amplitude         [R]
rapidtide_r2               [R]    optional
roi_ids                    [R]
roi_schema_hash            scalar
tr_seconds                 scalar
```

An optional `--rshrf-command-template` hook is provided for a site-specific/containerized rsHRF invocation, with placeholders `{input}`, `{out_dir}`, `{tr}`, and `{roi_schema}`. We intentionally do not hard-code an rsHRF package-internal API because the maintained rsHRF project currently exposes a BIDS-App workflow and its implementation interface may evolve.

## Storage policy

Keep permanently:

- the original/preprocessed source BOLD in the source dataset;
- RAPIDTIDE refined systemic regressor;
- delay map;
- coefficient map;
- fit-correlation map;
- run options/command/provenance;
- ROI-reduced hemodynamics products;
- rsHRF kernels.

Keep the full RAPIDTIDE cleaned 4-D BOLD only while estimating/validating rsHRF or while running the cleaned-BOLD ablation. It is not required by NMD training after the compact derived products are generated.
