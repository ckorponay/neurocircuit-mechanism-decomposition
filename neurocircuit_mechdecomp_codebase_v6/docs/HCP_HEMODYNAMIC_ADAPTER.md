# HCP-YA hemodynamic adapter: RAPIDTIDE + rsHRF -> NMD

The model expects all quantities in the **same fixed ROI identity/order** as the neural input.

## RAPIDTIDE fields

For each resting run, preferred RAPIDTIDE sources are:

- systemic waveform: final/refined moving regressor at the fMRI sampling grid;
- vascular delay: refined maximum-time/delay map, in seconds;
- vascular amplitude: sLFO regression fit coefficient map (`lfofilterCoeff`), not correlation strength;
- optional vascular QC/phenotype: `lfofilterR2` map.

Reduce the voxelwise delay/coefficient/R2 maps into the exact same fixed ROIs used by NMD. For delay, an R2-weighted robust ROI mean/median is a useful sensitivity analysis, but keep a plain unweighted summary too so the weighting choice is auditable.

Do **not** use `maxcorr` as alpha in the generative equation: it is a correlation strength, whereas the observation model's alpha is a regression amplitude. `lfofilterCoeff` corresponds more directly to that role.

## rsHRF fields

Run rsHRF on the ROI-level signal after removing/conditioning on the systemic component. Export one HRF kernel per fixed ROI in model ROI order.

The NMD loader automatically separates each raw HRF into:

- unit-L1 HRF shape,
- `hrf_gain`.

Use rsHRF's Wiener-deconvolved series only as an optional MAP warm start.

## Assemble one run

After ROI reduction, create arrays:

    hrf.npy                  [R,K]
    systemic.tsv             [T]
    vascular_delay.npy       [R]
    vascular_coeff.npy       [R]
    rapidtide_r2.npy         [R]  # optional

Then:

    python -m neurocircuit.scripts.build_hemodynamics_npz \
      --hrf-kernel hrf.npy \
      --systemic-waveform systemic.tsv \
      --vascular-delay vascular_delay.npy \
      --vascular-amplitude vascular_coeff.npy \
      --rapidtide-r2 rapidtide_r2.npy \
      --out sub-100307_REST1_LR_hemodynamics.npz

Validate:

    python -m neurocircuit.scripts.validate_hemodynamics_npz \
      sub-100307_REST1_LR_hemodynamics.npz \
      --n-regions R --n-timepoints T

## Critical ROI note

The demo HCP converter in the original repository can refit KMeans supervoxels independently per run. Do not use that behavior for cross-subject mechanistic training. Use fixed atlas ROIs or a once-fit/frozen reference parcellation so that every hemodynamic array and every neural array has identical node meaning.
