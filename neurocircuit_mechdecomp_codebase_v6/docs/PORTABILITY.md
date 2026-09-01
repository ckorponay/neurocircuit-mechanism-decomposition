# Cross-dataset portability: HCP-YA -> UKB -> HCP-PDC -> CAN-BIND1

This patch adds a backward-compatible physical-time path without deleting the
original sample-indexed behavior.

## Why the change is needed

The original scaffold encoded:
- SSM stability in a discrete transition matrix `A`;
- lag timing as integer sample indices.

Those quantities are not directly comparable when TR changes. A discrete
eigenvalue at TR=0.72 s is not the same biological dynamical quantity as the
same eigenvalue at TR=2.0 s, and lag index 3 means 2.16 s in HCP-YA but 6 s in
CAN-BIND1.

## Portable mode

Use:
- `ssm_parameterization: continuous`
- `lag_embedding_mode: seconds`
- `max_lag_seconds: 12.0`
- actual `data.tr_seconds` for every run

The SSM learns `A_c` and `B_c` in physical time and exactly discretizes them for
the run's TR. Report cross-dataset stability from `A_c`, not `A_d`.

The transformer learns a continuous relative-lag representation from lag in
seconds. The number of sampled lag positions therefore changes with TR while
the physical support remains 0-12 s.

Approximate sampled lag counts:
- HCP-YA TR=.720: 17 positions (0 ... 11.52 s)
- UKB TR=.735: 17 positions (0 ... 11.76 s)
- HCP-PDC TR=.800: 16 positions (0 ... 12.0 s)
- CAN-BIND1 TR=2.0: 7 positions (0 ... 12.0 s)

Do not upsample CAN-BIND1 and treat interpolated samples as new temporal
information.

## Backward compatibility

Old checkpoints can continue to use:
- `ssm_parameterization: discrete`
- `lag_embedding_mode: index`
- `n_lags: <old value>`

This is intentional. Do not load an old index-lag/discrete-SSM checkpoint into
portable mode and interpret it as physically equivalent.

## Canonical NPZ input

`neurocircuit.data.load_grouped_npz()` consumes the repository's existing NPZ
group convention (`C`, `S`, `Th`, `A`, `H`, `GPe`, `GPi`, `STN`, `MB`), checks
time-length consistency, concatenates groups in a fixed order, and emits one
canonical `[R,T]` float32 record with explicit TR and ROI ordering.

All future dataset adapters should end at this same contract.

## Recommended scientific sequence

1. Preserve the legacy HCP-YA path as a regression reference.
2. Train/validate the physical-time version in HCP-YA.
3. Stress-test HCP-YA by anti-alias filtering and resampling to UKB/PDC/CAN-BIND
   acquisition regimes.
4. Scale/finalize the model in UKB.
5. Freeze model weights before treatment-outcome analyses.
6. Apply run-level inference to HCP-PDC, aggregate within visit, test treatment
   perturbation and response.
7. Apply frozen inference to CAN-BIND1 and restrict claims to latents that pass
   the temporal-resolution stress test.

## Important existing scaffold limitations not solved by this patch

The current repository training loss is still a placeholder and the HRF-aware
observation model described in the methods remains TODO in `ssm_head.py`.
Those should be implemented before treating the repo as the final analysis
pipeline. This portability patch deliberately does not pretend those missing
components already exist.


## Hemodynamic state estimation

See `docs/HEMODYNAMIC_DECONVOLUTION.md` for the explicit rsHRF + RAPIDTIDE measurement model and MAP latent-state inference path.
