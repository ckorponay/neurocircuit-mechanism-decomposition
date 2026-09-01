# Production roadmap

## Phase 1: HCP-YA validation

- Freeze canonical ROI schema.
- Produce conventional BOLD ROI time series.
- Produce RAPIDTIDE waveform/delay/coefficient/R2 in same ROI schema.
- Estimate rsHRF kernels after systemic conditioning.
- Train optimized NMD from scratch.
- Compare four measurement variants:
  1. canonical HRF, no systemic model;
  2. RAPIDTIDE-cleaned BOLD, canonical HRF;
  3. RAPIDTIDE-cleaned + rsHRF Wiener deconvolution;
  4. explicit RAPIDTIDE branch + fixed rsHRF forward model (primary).
- Run synthetic recovery, HCP test-retest, task dissociation, and scan-length/TR stress tests.

## Phase 2: UK Biobank scale/finalization

- Run all image-heavy ROI/RAPIDTIDE/rsHRF preprocessing once on RAP.
- Store compact per-run neural + hemodynamic arrays.
- Train/sample-size learning curve: 1k -> 5k -> 20k -> full eligible.
- Do not materialize full pi during routine training.
- Freeze a preregistered model/checkpoint and feature definitions before clinical outcome testing.

## Phase 3: HCP-PDC perturbation

- Frozen run-level inference at each visit.
- Aggregate run estimates within visit with uncertainty.
- Analyze baseline mechanisms, treatment-induced delta mechanisms, and response associations.
- Keep neural circuit, local HRF, and systemic vascular changes as separate outcome families.

## Phase 4: CAN-BIND1 external generalization

- No temporal upsampling interpreted as new information.
- Restrict primary replication claims to latent quantities that survived the HCP TR=2 s degradation experiment.
- Frozen baseline/week-2/week-8 inference.
- Test whether early mechanistic change forecasts later response.
