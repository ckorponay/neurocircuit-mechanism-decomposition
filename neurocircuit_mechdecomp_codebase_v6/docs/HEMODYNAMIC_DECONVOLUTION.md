# HRF-aware state estimation with rsHRF + RAPIDTIDE

## Measurement model

The portable NMD observation equation is

    observed BOLD_i(t)
      = [HRF_i * neural_state_i](t)
      + alpha_i * systemic_waveform(t - vascular_delay_i)
      + residual_i(t)

The two timing terms have deliberately different meanings:

- `HRF_i`: local neurovascular impulse response to local neural activity.
- `vascular_delay_i`: RAPIDTIDE systemic blood-arrival timing.

**Never substitute RAPIDTIDE delay for HRF latency.**

## Version-1 workflow

1. Run RAPIDTIDE on the original/preprocessed BOLD.
2. Retain, per run:
   - systemic waveform;
   - ROI-level vascular arrival delay;
   - ROI-level systemic amplitude;
   - R2/QC when available.
3. For HRF estimation, remove the fitted systemic term and run rsHRF on the
   residual ROI time series.
4. Export the estimated ROI HRF kernels at the actual run TR.
5. Freeze all rsHRF/RAPIDTIDE measurement parameters during NMD latent-state
   inference.
6. Estimate the latent neural trajectory by MAP optimization against BOTH:
   - the BOLD measurement equation; and
   - the SSM dynamics prior.
7. Feed the resulting latent neural states to the lag-resolved Transformer.

rsHRF Wiener-deconvolved time series can be used as an `initial_x` warm start
and as an ablation comparator. They are not defined as neural ground truth.

## Hemodynamics NPZ contract

One per run:

- `hrf_kernel`              [R,K]  — rsHRF local HRF
- `systemic_waveform`       [T]    — RAPIDTIDE systemic regressor
- `vascular_delay_seconds`  [R]    — RAPIDTIDE arrival delay
- `vascular_amplitude`      [R]    — fitted systemic amplitude
- `rapidtide_r2`            [R]    — optional QC/vascular phenotype

Validate with:

    python -m neurocircuit.scripts.validate_hemodynamics_npz \
      --hemodynamics sub-XXX_run-YY_hemodynamics.npz \
      --n-regions 170 \
      --n-timepoints 1200

## Identifiability guardrails

Version 1 deliberately does NOT jointly learn HRF latency and neural propagation
lag from the same run. Doing so would allow a temporal difference to be
arbitrarily reassigned between measurement and circuit mechanisms.

Later versions may allow small HRF deviations around rsHRF estimates, but only
with hierarchical shrinkage / strong priors and after synthetic identifiability
validation.

## HCP validation matrix

Compare:

A. Original BOLD + canonical HRF
B. RAPIDTIDE residual BOLD + canonical HRF
C. RAPIDTIDE residual + rsHRF Wiener deconvolution
D. Original BOLD + explicit RAPIDTIDE branch + fixed rsHRF kernel + MAP state inference

Primary validity tests:
- recovery of simulated neural lag when HRF latency changes independently;
- recovery when systemic vascular delay changes independently;
- test-retest reliability of neural latent metrics;
- Motor/Gambling task dissociations;
- reduced association between inferred neural propagation lag and RAPIDTIDE
  vascular arrival delay without erasing biologically meaningful variance.
