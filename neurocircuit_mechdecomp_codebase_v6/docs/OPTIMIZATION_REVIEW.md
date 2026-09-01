# Architecture review and recommended production mode (September 2026)

This review starts from the public `neurocircuit_mechdecomp_codebase_v6` scaffold and asks a narrow question: what changes improve *identifiability, biological interpretation, portability, and UKB-scale compute* without adding complexity for its own sake?

## Bottom line

Do **not** make the network deeper yet. The highest-value changes are structural:

1. make time physical (seconds, continuous-time SSM),
2. make lag truly pathway-specific,
3. correct lag indexing,
4. actually couple Transformer drive into SSM responsiveness B,
5. separate neural, local-HRF, and systemic-vascular timing,
6. keep ROI identity fixed across every subject/run,
7. avoid materializing the full routing tensor unless needed.

The code now provides a legacy mode for reproducing the original scaffold and a recommended optimized mode for new analyses.

---

## 1. Original additive attention cannot identify pathway-specific lag

The original head computes

    score(i,j,l,t) = spatial(i,j,t) + temporal(l,t)

and applies one softmax over `(source i, lag l)` for each target `j`. Because the exponential of a sum factorizes,

    exp(S_ij + T_l) = exp(S_ij) exp(T_l),

the conditional lag distribution is the same for every source pathway at a given time/target. That conflicts with the scientific goal of estimating `pi(i,l->j)` as a pathway-specific propagation distribution.

### Fix

Recommended `attention_mode: edge_conditioned_sparse` adds an explicit source-target × lag interaction. Lag preferences can now differ between, for example, cortical source A→striatum and cortical source B→striatum while sharing a global temporal prior.

A legacy `legacy_additive` mode remains for old result reproduction.

---

## 2. Original history windows reverse the documented lag labels

PyTorch `unfold` on the left-padded time series returns each history window oldest→current. The original code treated array index 0 as lag 0/current, effectively reversing the lag axis used in drive calculation.

### Fix

New analyses use `lagged_windows(..., legacy_oldest_first=False)`, which flips the history axis so:

- index 0 = x(t)
- index 1 = x(t-TR)
- index 2 = x(t-2TR)
- ...

Legacy mode preserves the old ordering explicitly.

---

## 3. B was present in the SSM but absent from the canonical coupled inference

The original model calls the SSM with `u=None` and only estimates Transformer drive after the SSM pass. Therefore the low-rank B matrix does not affect the latent trajectory in the canonical forward pass, so it cannot function as the claimed input-responsiveness mechanism.

### Fix

Recommended inference/training is alternating:

1. infer latent neural state from observed BOLD under the fixed hemodynamic observation model;
2. Transformer routes that state to estimate drive `u_hat(t)`;
3. re-estimate/score latent state under `x(t+dt) ~= A x(t) + B u_hat(t)`;
4. repeat a small number of times;
5. M-step trains routing and B from the transition residual.

`latent_dynamics_loss()` has an explicit test showing gradient flow into both Transformer routing parameters and B.

---

## 4. Stability should be physical-time and preferably stable by construction

A discrete eigenvalue has different physical meaning at TR=.72 s and TR=2 s. The original `rho(A)<=.98` projection is therefore not cross-dataset portable. Repeated eigendecomposition/projection also adds unnecessary optimizer overhead.

### Fix

Recommended mode learns continuous-time dynamics and exactly discretizes at the actual run TR. `continuous_stability_mode: diagonal_dominant` constructs a graph-masked A_c whose Gershgorin discs lie strictly in the left half-plane, guaranteeing stability without an eigendecomposition/projection every optimizer step.

Cross-dataset reports use:

- stability margin in 1/s,
- slowest decay time in seconds,
- continuous A_c/B_c.

Discrete A_d is retained as a run-specific computational object, not the cross-dataset biological parameter.

---

## 5. rsHRF and RAPIDTIDE should constrain different parts of the measurement model

Recommended observation equation:

    BOLD_i(t) = [h_i * x_i](t) + alpha_i s(t-delta_i) + epsilon_i(t)

where:

- `x_i(t)` = latent neural state,
- `h_i` = rsHRF-informed local HRF shape,
- `s(t)` = RAPIDTIDE systemic waveform,
- `delta_i` = RAPIDTIDE vascular arrival delay,
- `alpha_i` = RAPIDTIDE sLFO regression coefficient.

RAPIDTIDE timing is not called HRF timing. rsHRF Wiener deconvolution can initialize x but is not treated as neural ground truth.

### HRF scale identifiability

HRF amplitude and latent-neural amplitude otherwise trade off directly. The loader therefore splits each rsHRF kernel into:

- unit-L1 HRF **shape**, fixed in the observation kernel,
- explicit `hrf_gain`, retained separately.

This makes local neurovascular gain visible rather than silently rescaling x.

---

## 6. ROI identity must be fixed; per-run clustering is not valid for production

The repository's HCP demo converter can fit KMeans supervoxels separately on each run. `S_0` from one run therefore need not represent the same tissue as `S_0` from another run. A shared neural network and graph mask require the opposite: identical node meaning and order everywhere.

### Production rule

Use one of:

- a fixed anatomical atlas/parcellation for all datasets, **preferred**;
- or fit any data-driven subparcellation ONCE on a training/reference dataset, freeze the voxel labels, and apply those exact labels to all subjects/runs.

The new canonical data schema includes ROI identity/order checks and a schema checksum. New cross-subject training should fail rather than silently combine independently refit parcellations.

---

## 7. Full pi is an analysis output, not a training representation

A dense routing tensor `[B,T,R_target,R_source,L]` is expensive at UKB scale. Most training and subject-level feature extraction only need:

- per-target drive,
- edge mass,
- pathway lag centroid,
- pathway peak lag,
- lag concentration.

### Fix

The sparse attention path loops only over anatomically allowed incoming edges and computes compact summaries directly. Full dense `pi` is materialized only with `return_pi=True` for selected validation/visualization runs.

---

## 8. Two possible future optimizations I would NOT add yet

### Constrain B further

The current low-rank full B can mix target-specific drive across regions. A later comparison should test:

- diagonal positive B (clean regional responsiveness),
- graph-masked B,
- diagonal + low-rank residual,
- current rank-3 full B.

Do this as an identifiability ablation after the core model works, not now.

### Gating g(t)

Dynamic gating is scientifically appealing but adds another multiplicative degree of freedom that can trade off with routing mass and B. Leave it off until the ungated neural/hemodynamic decomposition passes synthetic recovery, test-retest, and task validation.

---

## Recommended new-analysis configuration

    attention_mode: edge_conditioned_sparse
    lag_embedding_mode: seconds
    max_lag_seconds: 12.0
    legacy_oldest_first: false
    ssm_parameterization: continuous
    continuous_stability_mode: diagonal_dominant
    discretization_method: solve
    hrf_source: rshrf_external_fixed
    systemic_source: rapidtide_external_fixed
    state_inference: map_alternating

Keep the original HCP config untouched as the legacy regression reference.

## Validation sequence before scientific claims

1. Unit/synthetic recovery (already covered by package tests).
2. Reproduce legacy HCP behavior under explicit legacy mode.
3. Train optimized HCP-YA model from scratch; do **not** reinterpret legacy checkpoints as optimized model checkpoints.
4. Independently manipulate neural lag, HRF lag/dispersion, systemic delay, and systemic amplitude in simulation; verify separation.
5. HCP-YA scan-length/TR degradation tests for UKB/PDC/CAN-BIND regimes.
6. HCP-YA test-retest and Motor/Gambling validation.
7. Scale/freeze with UKB.
8. Frozen application to PDC and then CAN-BIND.
