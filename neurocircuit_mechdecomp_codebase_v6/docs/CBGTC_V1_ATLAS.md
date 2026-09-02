# Canonical CBGTC Atlas v1 (`cbgtc_v1`)

`cbgtc_v1` fixes node identity across HCP-YA, UK Biobank, HCP-PDC and CAN-BIND and replaces the pilot per-run KMeans/supervoxel striatal representation.

## Primary composition: 164 nodes

- **Cortex (100):** Schaefer2018 100 parcels, 7 networks.
- **Striatum (20):** Tian 3T **Scale III**, the finest Tian 3T striatal subdivision: bilateral PUT-VA/DA/VP/DP, CAU-VA/DA/body/tail, NAc shell/core. These striatal labels are unchanged at S4.
- **Thalamus (16):** Tian 3T **Scale IV**, bilateral VAip, VAia, VPm, VPl, VAs, DAm, DAl and DP.
- **Amygdala (4):** Tian 3T **Scale IV** labels, bilateral lateral and medial amygdala. Tian does not introduce an additional 3T amygdala split between S3 and S4.
- **Hippocampus (10):** Tian 3T **Scale IV**, bilateral head-medial-1, head-medial-2, head-lateral, body and tail.
- **Specialized BG/midbrain (14):** CIT168 bilateral GPe, GPi, ventral pallidum, STN, SNc, SNr and VTA.

Tian pallidal parcels are intentionally not used because GPe versus GPi (plus VeP) is more mechanistically informative for the CBGTC model. CIT168 supplies these specialized identities.

## Primary routing graph

The hard routing prior answers **whether a pathway class is anatomically plausible**, not which exact target parcel it must prefer. We therefore avoid baking expected corticostriatal/limbic topography into the target definition.

Primary striatal input classes all have access to all 20 Tian S3 striatal parcels:

- `C -> S`
- `A -> S`
- `H -> S`
- `Th -> S`
- `SNc -> S`
- `VTA -> S`

The primary recurrent cortical/limbic/thalamic macro-circuit includes:

- `C <-> A`
- `C <-> H`
- `C <-> Th`
- `A <-> H`

Canonical basal-ganglia paths include:

- `C -> STN`
- `S -> GPe`, `S -> GPi`, `S -> SNr`, `S -> VeP`
- `GPe -> S`
- `GPe <-> STN`
- `GPe -> GPi/SNr`
- `STN -> GPi/SNr`
- `GPi/SNr/VeP -> Th`

The `extended` routing graph adds full `C -> C` routing as an explicit architecture ablation plus additional plausible modulatory/re-entrant routes. Generic cortico-cortical routing is **not** part of the primary Transformer decomposition.

## Cortico-cortical dynamics without a 9,900-edge routing explosion

Cortical interactions are not assumed absent. They are represented in the continuous-time SSM rather than the primary Transformer routing head.

The primary SSM uses two complementary cortical terms:

1. **Sparse explicit A edges** within the same Schaefer/Yeo network, controlled by `cbgtc_v1_dynamics_core.npy`.
2. **A global directed low-rank cortical block**, default rank 8, that can capture distributed cross-network cortical recurrency without estimating every one of the 100 x 99 possible directed cortical pairs independently.

The low-rank cortical term is included inside the stable-by-construction diagonal-dominant continuous-time A matrix, so stability is preserved after adding it.

Portable configs set:

```yaml
model:
  n_regions: 164
  n_cortical_regions: 100
  cortical_low_rank_rank: 8
```

A full explicit `C -> C` Transformer model remains available through the extended routing mask for validation/ablation.

## Build the schema

```bash
python -m neurocircuit.scripts.write_cbgtc_v1_schema --out-dir atlases/cbgtc_v1
```

This writes ROI IDs, labels, metadata, edge lists and `[R_src,R_tgt]` routing/dynamics masks.

## Build a volumetric atlas

Install optional atlas dependencies:

```bash
python -m pip install -r requirements-atlas.txt
```

For HCP space:

```bash
python -m neurocircuit.scripts.build_cbgtc_v1 \
  --tian-s3-atlas /path/Tian_Subcortex_S3_3T.nii \
  --tian-s3-labels /path/Tian_Subcortex_S3_3T_label.txt \
  --tian-s4-atlas /path/Tian_Subcortex_S4_3T.nii \
  --tian-s4-labels /path/Tian_Subcortex_S4_3T_label.txt \
  --schaefer-atlas /path/Schaefer2018_100Parcels_7Networks_dseg.nii.gz \
  --schaefer-labels /path/Schaefer2018_100Parcels_7Networks_dseg.tsv \
  --cit168-atlas /path/CIT168_dseg.nii.gz \
  --cit168-labels /path/CIT168_labels.txt \
  --space MNI152NLin6Asym \
  --out-dir atlases/cbgtc_v1
```

Tian S4 defines the output grid. S3 supplies only striatum; S4 supplies thalamus, amygdala and hippocampus. Schaefer, S3 and CIT168 are nearest-neighbor resampled to the S4 grid. Specialized CIT168 nuclei take priority in overlap. The output is a single non-overlapping integer label image with the exact same order as `cbgtc_v1_roi_ids.txt`.

Build the same node identities in `MNI152NLin2009cAsym` for fMRIPrep cohorts. Node identity/order must not change across template spaces or datasets.

## Architecture ablations to retain

1. Primary: no generic C->C Transformer routing; rank-8 cortical SSM + sparse within-network A.
2. Extended routing: explicit full C->C Transformer routing.
3. Cortical-rank sensitivity: e.g. rank 0, 4, 8, 16.

Judge these by held-out reconstruction, synthetic recovery, HCP task dissociation, test-retest reliability, TR-degradation robustness and stability of the circuit quantities of interest—not by in-sample fit alone.
