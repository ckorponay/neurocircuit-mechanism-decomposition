from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
import csv
import json
from typing import Iterable

import numpy as np

CBGTC_V1_NAME = "cbgtc_v1"
CBGTC_V1_N_REGIONS = 164
CBGTC_V1_N_CORTEX = 100


@dataclass(frozen=True)
class ROI:
    roi_id: str
    group: str
    structure: str
    hemisphere: str
    source_atlas: str
    source_label: str
    role: str


def _cortical_rois() -> list[ROI]:
    rois: list[ROI] = []
    # Schaefer100 7-network ordering is 50 LH followed by 50 RH in TemplateFlow.
    for hemi, start in (("L", 1), ("R", 51)):
        for within in range(1, 51):
            source_index = start + within - 1
            rois.append(
                ROI(
                    roi_id=f"C_{hemi}_{within:03d}",
                    group="C",
                    structure="cortex",
                    hemisphere=hemi,
                    source_atlas="Schaefer2018_100Parcels_7Networks",
                    source_label=str(source_index),
                    role="cortical",
                )
            )
    return rois


def _tian_striatum_s3_rois() -> list[ROI]:
    """Tian 3T Scale III: finest striatal subdivision (unchanged at S4)."""
    rois: list[ROI] = []
    per_hemi = [
        ("PutVA", "PUT-VA"),
        ("PutDA", "PUT-DA"),
        ("PutVP", "PUT-VP"),
        ("PutDP", "PUT-DP"),
        ("CauVA", "CAU-VA"),
        ("CauDA", "CAU-DA"),
        ("CauBody", "CAU-body"),
        ("CauTail", "CAU-tail"),
        ("NAcShell", "NAc-shell"),
        ("NAcCore", "NAc-core"),
    ]
    for hemi, suffix in (("L", "lh"), ("R", "rh")):
        for short, source in per_hemi:
            rois.append(
                ROI(
                    roi_id=f"S_{hemi}_{short}",
                    group="S",
                    structure="striatum",
                    hemisphere=hemi,
                    source_atlas="Tian2020_3T_S3",
                    source_label=f"{source}-{suffix}",
                    role="striatal",
                )
            )
    return rois


def _tian_s4_rois() -> list[ROI]:
    """Finest 3T Tian subdivisions for thalamus, amygdala and hippocampus."""
    rois: list[ROI] = []
    per_hemi = [
        # Scale IV thalamus (8 per hemisphere).
        ("Th", "thalamus", "VAip", "THA-VAip", "thalamic"),
        ("Th", "thalamus", "VAia", "THA-VAia", "thalamic"),
        ("Th", "thalamus", "VPm", "THA-VPm", "thalamic"),
        ("Th", "thalamus", "VPl", "THA-VPl", "thalamic"),
        ("Th", "thalamus", "VAs", "THA-VAs", "thalamic"),
        ("Th", "thalamus", "DAm", "THA-DAm", "thalamic"),
        ("Th", "thalamus", "DAl", "THA-DAl", "thalamic"),
        ("Th", "thalamus", "DP", "THA-DP", "thalamic"),
        # Amygdala reaches its finest 3T split by S3 and is unchanged at S4.
        ("A", "amygdala", "Lat", "lAMY", "limbic_input"),
        ("A", "amygdala", "Med", "mAMY", "limbic_input"),
        # Scale IV hippocampus (5 per hemisphere).
        ("H", "hippocampus", "HeadM1", "HIP-head-m1", "limbic_input"),
        ("H", "hippocampus", "HeadM2", "HIP-head-m2", "limbic_input"),
        ("H", "hippocampus", "HeadL", "HIP-head-l", "limbic_input"),
        ("H", "hippocampus", "Body", "HIP-body", "limbic_input"),
        ("H", "hippocampus", "Tail", "HIP-tail", "limbic_input"),
    ]
    for hemi, suffix in (("L", "lh"), ("R", "rh")):
        for group, structure, short, source, role in per_hemi:
            rois.append(
                ROI(
                    roi_id=f"{group}_{hemi}_{short}",
                    group=group,
                    structure=structure,
                    hemisphere=hemi,
                    source_atlas="Tian2020_3T_S4",
                    source_label=f"{source}-{suffix}",
                    role=role,
                )
            )
    return rois


def _cit168_rois() -> list[ROI]:
    rois: list[ROI] = []
    structures = [
        ("GPe", "GPe", "pallidum_external", "bg_intermediate"),
        ("GPi", "GPi", "pallidum_internal", "bg_output"),
        ("VeP", "VeP", "ventral_pallidum", "bg_intermediate"),
        ("STN", "STN", "subthalamic_nucleus", "bg_intermediate"),
        ("SNc", "SNc", "substantia_nigra_compacta", "dopaminergic_input"),
        ("SNr", "SNr", "substantia_nigra_reticulata", "bg_output"),
        ("VTA", "VTA", "ventral_tegmental_area", "dopaminergic_input"),
    ]
    for hemi in ("L", "R"):
        for group, source, structure, role in structures:
            rois.append(
                ROI(
                    roi_id=f"{group}_{hemi}",
                    group=group,
                    structure=structure,
                    hemisphere=hemi,
                    source_atlas="CIT168_v1.1",
                    source_label=source,
                    role=role,
                )
            )
    return rois


def canonical_rois() -> list[ROI]:
    rois = _cortical_rois() + _tian_striatum_s3_rois() + _tian_s4_rois() + _cit168_rois()
    if len(rois) != CBGTC_V1_N_REGIONS:
        raise AssertionError(f"cbgtc_v1 ROI count changed: {len(rois)}")
    ids = [r.roi_id for r in rois]
    if len(ids) != len(set(ids)):
        raise AssertionError("duplicate cbgtc_v1 ROI ids")
    return rois


def canonical_roi_ids() -> list[str]:
    return [r.roi_id for r in canonical_rois()]


def _allow_group(
    mask: np.ndarray,
    rois: list[ROI],
    src_group: str,
    tgt_group: str,
    *,
    src_pred=None,
    tgt_pred=None,
) -> None:
    for i, s in enumerate(rois):
        if s.group != src_group or (src_pred is not None and not src_pred(s)):
            continue
        for j, t in enumerate(rois):
            if t.group != tgt_group or (tgt_pred is not None and not tgt_pred(t)):
                continue
            if i != j:
                mask[i, j] = True


def build_routing_mask(profile: str = "core") -> np.ndarray:
    """
    Directed [R_src,R_tgt] mask for interpretable Transformer circuit drive.

    The primary mask encodes pathway classes with strong anatomical support but
    deliberately avoids parcel-specific target topographies. For example,
    cortex, amygdala, hippocampus and thalamus may each route to every Tian S3
    striatal parcel; the model learns the topography rather than receiving it as
    a hard prior.
    """
    if profile not in {"core", "extended"}:
        raise ValueError("profile must be 'core' or 'extended'")
    rois = canonical_rois()
    mask = np.zeros((len(rois), len(rois)), dtype=bool)
    allow = lambda s, t, **kw: _allow_group(mask, rois, s, t, **kw)

    # Recurrent cortical/limbic/thalamic macro-circuit, excluding generic C->C
    # from the primary interpretable routing head.
    allow("C", "A")
    allow("A", "C")
    allow("C", "H")
    allow("H", "C")
    allow("C", "Th")
    allow("Th", "C")
    allow("A", "H")
    allow("H", "A")

    # Major inputs to striatum. No parcel-specific topography is hard-coded.
    allow("C", "S")
    allow("A", "S")
    allow("H", "S")
    allow("Th", "S")
    allow("SNc", "S")
    allow("VTA", "S")

    # Hyperdirect pathway.
    allow("C", "STN")

    # Direct/indirect/output basal-ganglia pathways.
    allow("S", "GPe")
    allow("S", "GPi")
    allow("S", "VeP")
    allow("S", "SNr")
    allow("GPe", "S")
    allow("GPe", "STN")
    allow("STN", "GPe")
    allow("GPe", "GPi")
    allow("GPe", "SNr")
    allow("STN", "GPi")
    allow("STN", "SNr")
    allow("GPi", "Th")
    allow("SNr", "Th")
    allow("VeP", "Th")
    allow("VeP", "VTA")

    if profile == "extended":
        # Explicit whole-cortex propagation is an ablation, not the primary
        # mechanistic decomposition.
        allow("C", "C")
        # Additional plausible modulatory/re-entrant routes retained for
        # sensitivity analyses rather than imposed in the primary graph.
        allow("S", "SNc")
        allow("S", "VTA")
        allow("VTA", "C")
        allow("VTA", "A")
        allow("VTA", "H")
        allow("VTA", "VeP")
        allow("SNc", "C")
        allow("Th", "A")
        allow("Th", "H")

    return mask


def _cortical_network(roi: ROI) -> str | None:
    if roi.group != "C":
        return None
    idx = int(roi.source_label)
    ranges = [
        (1, 9, "Vis"), (10, 15, "SomMot"), (16, 23, "DorsAttn"),
        (24, 30, "SalVentAttn"), (31, 33, "Limbic"), (34, 37, "Cont"),
        (38, 50, "Default"), (51, 58, "Vis"), (59, 66, "SomMot"),
        (67, 73, "DorsAttn"), (74, 78, "SalVentAttn"), (79, 80, "Limbic"),
        (81, 89, "Cont"), (90, 100, "Default"),
    ]
    for lo, hi, name in ranges:
        if lo <= idx <= hi:
            return name
    raise AssertionError(idx)


def build_dynamics_mask(profile: str = "core") -> np.ndarray:
    """
    Sparse explicit [R_src,R_tgt] mask for the linear SSM A matrix.

    In the primary model this permits within-network cortical couplings in the
    explicit sparse A term. A separate low-rank cortical component (configured
    in the SSM, default rank 8) captures distributed cross-network C->C
    background dynamics without adding 9,900 independently estimated cortical
    edges or turning them into Transformer routing mechanisms.
    """
    if profile not in {"core", "extended"}:
        raise ValueError("profile must be 'core' or 'extended'")
    rois = canonical_rois()
    mask = build_routing_mask(profile).copy()
    allow = lambda s, t, **kw: _allow_group(mask, rois, s, t, **kw)

    if profile == "core":
        for i, s in enumerate(rois):
            if s.group != "C":
                continue
            ns = _cortical_network(s)
            for j, t in enumerate(rois):
                if i != j and t.group == "C" and _cortical_network(t) == ns:
                    mask[i, j] = True
    else:
        allow("C", "C")

    return mask


def build_graph_mask(profile: str = "core") -> np.ndarray:
    """Backward-compatible alias: graph mask means Transformer routing mask."""
    return build_routing_mask(profile)


def graph_edges(mask: np.ndarray) -> Iterable[tuple[str, str]]:
    ids = canonical_roi_ids()
    ii, jj = np.nonzero(mask)
    for i, j in zip(ii.tolist(), jj.tolist()):
        yield ids[i], ids[j]


def write_schema_bundle(out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rois = canonical_rois()

    (out / "cbgtc_v1_roi_ids.txt").write_text("\n".join(r.roi_id for r in rois) + "\n")

    fields = ["label", "roi_id", "group", "structure", "hemisphere", "source_atlas", "source_label", "role"]
    with (out / "cbgtc_v1_labels.tsv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for label, roi in enumerate(rois, start=1):
            row = asdict(roi)
            row = {"label": label, **row}
            w.writerow(row)

    for profile in ("core", "extended"):
        routing = build_routing_mask(profile)
        dynamics = build_dynamics_mask(profile)
        for kind, m in (("routing", routing), ("dynamics", dynamics)):
            np.save(out / f"cbgtc_v1_{kind}_{profile}.npy", m)
            with (out / f"cbgtc_v1_{kind}_{profile}.tsv").open("w", newline="") as f:
                w = csv.writer(f, delimiter="\t")
                w.writerow(["source_roi", "target_roi"])
                w.writerows(graph_edges(m))
        np.save(out / f"cbgtc_v1_graph_{profile}.npy", routing)

    metadata = {
        "name": CBGTC_V1_NAME,
        "n_regions": len(rois),
        "n_cortical_regions": CBGTC_V1_N_CORTEX,
        "cortex": "Schaefer2018 100 parcels, 7 networks",
        "striatum": "Tian2020 Melbourne Subcortex Atlas, 3T Scale III (20 parcels; finest 3T striatal split)",
        "thalamus": "Tian2020 3T Scale IV (16 parcels; finest 3T split)",
        "amygdala": "Tian2020 3T Scale IV labels (4 parcels; no additional split beyond S3)",
        "hippocampus": "Tian2020 3T Scale IV (10 parcels; finest 3T split)",
        "cit168": "CIT168 v1.1; GPe/GPi/VeP/STN/SNc/SNr/VTA",
        "primary_routing_graph": "core",
        "primary_dynamics_graph": "core",
        "recommended_cortical_low_rank_rank": 8,
        "notes": [
            "Node identity/order is fixed across datasets.",
            "Tian pallidum is intentionally replaced by CIT168 mechanistic GPe/GPi/VeP identities.",
            "Cortex, amygdala, hippocampus and thalamus may route to all striatal parcels in the primary graph; target topography is learned rather than hard-coded.",
            "Cortex<->amygdala, cortex<->hippocampus, cortex<->thalamus and amygdala<->hippocampus are primary macro-circuit routes.",
            "Thalamus->striatum, striatum->GPe/GPi/SNr/VeP and GPe->striatum are included in the core circuitry.",
            "Generic cortex->cortex Transformer routing is excluded from the core graph and enabled only in the extended ablation.",
            "Core SSM dynamics use sparse within-network C->C edges plus a separate low-rank global cortical dynamics term.",
            "Self-routing is excluded from Transformer masks; SSM diagonal A models intrinsic persistence.",
        ],
    }
    (out / "cbgtc_v1_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return out
