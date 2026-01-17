See `docs/WORKFLOW.md` for a formal end-to-end workflow overview.

# Summary Protocol (end-to-end)

This protocol summarizes **how the scripts in this repository fit together** and which components match the **current methods-only Transformer–SSM description** in the preprint.

The repo includes two parallel "tracks":

- **Track A (current / method-aligned):** clean, modular implementation of the *Transformer–SSM* architecture (recommended).
- **Track B (legacy / exploratory):** earlier loop-aware Transformer prototypes and downstream burst / hazard / visualization utilities.

If you only want a grant-citable, maintainable pipeline that matches the PDF method description, use **Track A**.

---

## Track A (current): Transformer–SSM pipeline

### Step A0 — Environment
```bash
conda env create -f environment.yml
conda activate neurocircuit-mechdecomp
```

### Step A1 — Build tokenized inputs (NPZ)
You need per-run `.npz` files containing region/group time series (recommended keys: `C`, `S`, optionally `Th`, `A`, `H`, `GPe`, `GPi`, `STN`, `MB`).

You have **two preprocessors** depending on your data format:

#### A1.1 Volumetric NIfTI inputs (Pilot50 style)
- **Script:** `pilot50/vol_to_npz.py`
- **Use when:** you already have 4D NIfTI data in MNI space.

Example:
```bash
python pilot50/vol_to_npz.py \
  --func /path/to/rest_bold.nii.gz \
  --atlas-cortex /path/to/cortex_labels.nii.gz \
  --atlas-striatum /path/to/striatum_kparc.nii.gz \
  --out /path/to/out_run.npz
```

#### A1.2 HCP-YA CIFTI inputs (dtseries)
- **Script:** `tools/hcp_to_npz_converter.py`
- **Use when:** inputs are HCP `*.dtseries.nii`.

Example:
```bash
python tools/hcp_to_npz_converter.py make-npz \
  --dtseries rfMRI_REST1_LR_Atlas_MSMAll.dtseries.nii \
  --dlabel   Schaefer2018_400Parcels_7Networks_order.dlabel.nii \
  --striatum-mask striatum_mask_MNI.nii.gz \
  --K-S 1000 \
  --out data/npz/sub-100307_REST1_LR.npz
```

Create a manifest listing your NPZ files (one per line):
```bash
ls data/npz/*.npz > data/npz_manifest.txt
```

### Step A2 — Train the modular model (skeleton)
- **Entrypoint:** `python -m neurocircuit.scripts.train --config ...`
- **Current status:** This training script is a **scaffold** (loss is placeholder). It instantiates the *Transformer head* + *SSM head* and runs end-to-end forward passes.

Example:
```bash
python -m neurocircuit.scripts.train --config configs/hcpya_example.yaml
```

Where the architecture lives:
- Shared encoder: `neurocircuit/models/temporal_encoder.py`
- Graph-masked factorized attention + drive outputs: `neurocircuit/models/transformer_head.py`
- Measurement-aware SSM (HRF-aware) head: `neurocircuit/models/ssm_head.py`
- Full coupled model wrapper: `neurocircuit/models/model.py`

### Step A3 — Extract interpretable routing / lag metrics
The key method-aligned output is the routing tensor:

- `pi` with shape `(B, T, R_tgt, R_src, n_lags)`

From this you can compute:
- **Edge influence weights** (sum over lags)
- **Peak lag / centroid lag / concentration** per pathway
- **Per-target drive signals** `u_hat(t)`

At the moment, **analysis utilities for Track A** are intentionally minimal. If you need these metrics in production:

1) Add a checkpoint saver to `neurocircuit/scripts/train.py`
2) Write a small script that loads NPZ → builds `y` → runs the model → summarizes `pi`

(If you want, I can generate that extraction script in the same style as `analysis/lag_summary_legacy.py`, but using `neurocircuit/models/transformer_head.py`.)

---

## Track B (legacy): loop-aware Transformer prototypes + downstream analyses

This track was used during early development of the idea. It is **useful for exploratory runs** (especially if you already trained a prototype and logged attention weights), but it does **not** exactly match the current methods-only description.

### B1 — Legacy training prototypes
Located in `legacy/`:

- `legacy/loopaware_v2_burstlog_gate.py`
  - A loop-aware model with burst/log gating ideas.
  - Produces attention logs that downstream scripts expect.

- `legacy/loop_aware_cst_transformer_midbrain_gating.py`
  - A larger loop-aware cortico-striatal-pallidal-thalamic transformer prototype with midbrain gating.

- `legacy/loop_aware_transformer_v2_adds_h_th_s_and_gpe_s.py`
  - A variant that explicitly adds additional pathways.

**Important:** these are not guaranteed to implement the exact factorized spatial-temporal attention + stop-gradient coupling described in the preprint.

### B2 — Routing lag summaries (legacy)
- **Script:** `analysis/lag_summary_legacy.py`
- **What it does:** Reads logged cross-attention arrays and reports **mean/median/peak lag** per edge.

### B3 — Burst/state and hazard transition analyses (legacy)
- `analysis/burst_state_analysis_gate.py`
  - Segments time into burst-like states (based on gating/log activity)
  - Summarizes state occupancy / transition structure

- `analysis/hazard_transition_analysis.py`
  - Computes simple hazard / transition statistics over detected states

### B4 — Visualization / reporting (legacy)
- `analysis/network_overlays.py`
  - Convenience plotting for circuit overlays / network assignment summaries

- `analysis/clinical_panel.py`
  - A lightweight clinical-style panel / report generator

---

## Which scripts are "up to date"?

**Up-to-date with the current method description (recommended):**
- `neurocircuit/models/*` (Transformer head, SSM head, coupling)
- `pilot50/*` (a runnable minimal demo for volumetric inputs)
- `tools/hcp_to_npz_converter.py` (input conversion; conceptually aligned)

**Legacy / exploratory (keep for reference, not the canonical method):**
- `legacy/*`
- `analysis/*` (lag/burst/hazard/panel scripts are coupled to legacy logging conventions)

---

## Practical recommended workflow

If your goal is **“I need a stable, citable pipeline that mirrors the PDF”**:

1) Convert inputs → NPZ using `pilot50/vol_to_npz.py` or `tools/hcp_to_npz_converter.py`
2) Run the modular model forward pass (or extend training)
3) Save `pi`, `drive`, and SSM outputs (`x_hat`, HRF params)
4) Summarize lags/edges for figures

If your goal is **“I want quick feasibility plots right now”**:

1) Use Pilot50 (`pilot50/pilot50_run.sh`) for a sanity run
2) If you already trained a legacy prototype, run `analysis/lag_summary_legacy.py`

