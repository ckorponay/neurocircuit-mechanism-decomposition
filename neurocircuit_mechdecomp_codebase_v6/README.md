# Neurocircuit Mechanism Decomposition (Transformer–SSM)

This repository is a **GitHub-friendly reference implementation** for the *methods-only* architecture described in the accompanying preprint:

**A Dual-Head Transformer–State-Space Architecture for Neurocircuit Mechanism Decomposition from fMRI**  
Cole Korponay (2026)

## Quick start


## Documentation

- **Protocol:** `docs/PROTOCOL.md` (step-by-step runbook)
- **Workflow overview:** `docs/WORKFLOW.md` (formal end-to-end description)

### 1) Create an environment
```bash
conda env create -f environment.yml
conda activate neurocircuit-mechdecomp
```

### 2) Run a smoke test
```bash
python -m neurocircuit.scripts.smoke_test
```

### 3) Train (skeleton)
```bash
python -m neurocircuit.scripts.train --config configs/hcpya_example.yaml
```

> **Note:** This is a clean, reproducible scaffold intended for grant-citable methods sharing.
> Replace placeholders (data loading, evaluation, logging) with your project specifics.

## Repo structure

```
neurocircuit-mechdecomp/
  neurocircuit/                # library code (installable package)
    data/                      # dataset abstractions + loaders
    models/                    # Transformer head, SSM head, coupling
    ops/                       # masking, lag embeddings, Kalman ops
    scripts/                   # entrypoints (train/eval/smoke_test)
    utils/                     # config, seeding, metrics
  configs/                     # YAML configs
  tests/                       # minimal shape/sanity tests
  paper/                       # paper build assets (optional)
```

## Citation

- Code: https://github.com/<your-username>/neurocircuit-mechanism-decomposition
- Preprint: *(bioRxiv link once posted)*

A `CITATION.cff` is included for GitHub citation export.


## Pilot50 volumetric rest demo (optional)

This repo includes a minimal, runnable **~50-subject volumetric resting-state pilot** that mirrors the `Pilot50` bundle:

- Make a fixed **striatal K-parcellation** in MNI space
- Convert 4D volumetric rest NIfTIs -> compressed NPZ token files (C, S, optional Th/A/H/MB)
- Run a tiny loop-aware training stub for quick feasibility outputs

### Inputs
Create `lists/subjects_volrest.tsv` with 2 columns (no header):

```
sub-100307	/path/to/sub-100307_rest_bold.nii.gz
sub-100408	/path/to/sub-100408_rest_bold.nii.gz
...
```

### Run
Edit atlas/mask paths in `pilot50/pilot50_run.sh`, then:

```bash
bash pilot50/pilot50_run.sh
```

Outputs will land in `outputs/pilot50/`.

> The full mechanistic Transformer–SSM architecture described in the paper lives in `neurocircuit/models/`.
> The Pilot50 scripts are a lightweight scaffold for producing quick, grant-ready feasibility artifacts.

## Protocol (how scripts fit together)

For an end-to-end overview (preprocessing → training → extracting routing/lag metrics), see:

- `docs/PROTOCOL.md`

## Legacy scripts

During development, a set of earlier "loop-aware" prototypes and analysis utilities were created.
They are included for reproducibility and reference:

- `legacy/` (prototype model scripts)
- `analysis/` (lag summaries, burst/hazard analyses, visualization)

These legacy scripts are **not the canonical implementation** of the factorized Transformer–SSM described in the preprint.
