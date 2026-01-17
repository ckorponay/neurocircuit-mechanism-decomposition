# End-to-end workflow (formal overview)

This repository implements an end-to-end pipeline for **neurocircuit mechanism decomposition from fMRI** using a dual-head architecture that separates:

1. **Directed, lag-resolved propagation** through an anatomically constrained circuit graph (Transformer head), and  
2. **Latent neural dynamics + hemodynamic observation** (state-space model head).

The primary intended use is to transform ROI-level fMRI time series into **mechanistic circuit features** that are interpretable in terms of *routing*, *timing*, *baseline sensitivity*, and *time-varying modulation*.

---

## Workflow stages

### Stage 0 — ROI specification and circuit graph
Define the circuit nodes (ROIs/parcels) and an anatomical adjacency mask specifying allowed directed edges (e.g., cortex→striatum permitted; direct striatum→cortex disallowed). The adjacency mask constrains model capacity and ensures interpretability of inferred pathways.

### Stage 1 — Data conversion to model tokens
Convert raw fMRI data into compact, model-ready inputs:
- Extract ROI time series (region × time),
- Optionally sub-parcellate selected ROIs (e.g., striatum K-parcellation),
- Save as `.npz` with ROI metadata and subject/run identifiers.

**Output:** standardized per-run arrays with shape `(regions, time)` and a manifest of ROI definitions.

### Stage 2 — Measurement-aware state estimation (SSM head)
Fit a measurement-aware state-space model that explicitly accounts for hemodynamic convolution (HRF) to produce deconvolved latent state estimates.

**Goal:** reduce confounding between true neural propagation delays and vascular/HRF timing effects.

**Output:** deconvolved latent state trajectories `x̂(t)` and dynamics parameters (e.g., A/B matrices, stability, timescales).

### Stage 3 — Directed propagation and lag inference (Transformer head)
Using deconvolved states as input, estimate **directed** and **lag-resolved** predictive dependencies under the anatomical graph constraint. The transformer yields a routing tensor π(i,ℓ→j) that specifies how strongly source `i` at lag `ℓ` predicts target `j`.

**Output:** pathway influence weights, lag distributions, and drive signals `û(t)`.

### Stage 4 — Mechanistic feature extraction
Summarize the routing tensor and SSM parameters into interpretable metrics, including:
- edge influence strength,
- peak/centroid lag and temporal concentration,
- per-target drive traces `û(t)`,
- baseline input sensitivity (B),
- intrinsic stability and timescale descriptors from A.

### Stage 5 — Optional modulatory gating and state transitions
Optionally estimate time-varying gating variables (e.g., `g(t)`) that multiplicatively modulate driver→target families, capturing context-dependent routing and loop-level state changes.

### Stage 6 — Downstream scientific analysis
Use mechanistic features for subject-level inference, hypothesis tests, subgroup discovery, and phenotype association (e.g., regression, clustering, hazard/burst transition analyses).

---

## Outputs (what you can cite / report)
For each subject/run, the pipeline yields a structured “mechanism report” including:
- **who drives whom** (directed pathways),
- **when** influences arrive (lag metrics),
- **how sensitive** targets are to inputs (baseline sensitivity),
- **how dynamics behave** (stability/timescales),
- **whether routing is dynamically modulated** (gating).

These features are designed to be more interpretable than standard functional connectivity because they make directionality, timing, and measurement confounds explicit.
