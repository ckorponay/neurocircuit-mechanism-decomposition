from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from neurocircuit.atlas.cbgtc_v1 import canonical_rois, write_schema_bundle


def _lazy_import_nibabel():
    try:
        import nibabel as nib
        from nibabel.processing import resample_from_to
    except ImportError as exc:
        raise RuntimeError(
            "Building the volumetric atlas requires nibabel/scipy. Install "
            "requirements-atlas.txt in the project environment."
        ) from exc
    return nib, resample_from_to


def _read_simple_labels(path: str | Path) -> dict[int, str]:
    """Read one-label-per-line or a table with numeric first column + name."""
    lines = [
        x.strip()
        for x in Path(path).read_text().splitlines()
        if x.strip() and not x.lstrip().startswith("#")
    ]
    if not lines:
        raise ValueError(f"empty labels file: {path}")
    out: dict[int, str] = {}
    for line_no, line in enumerate(lines, start=1):
        parts = line.replace(",", "\t").split()
        try:
            idx = int(parts[0])
            name = parts[1]
        except (ValueError, IndexError):
            idx = line_no
            name = parts[0]
        out[idx] = name
    return out


def _read_schaefer_tsv(path: str | Path) -> dict[int, str]:
    p = Path(path)
    with p.open() as f:
        sample = f.readline()
        f.seek(0)
        delim = "\t" if "\t" in sample else None
        if delim:
            reader = csv.DictReader(f, delimiter="\t")
            rows = list(reader)
            if rows and "index" in rows[0] and "name" in rows[0]:
                return {int(r["index"]): r["name"] for r in rows}
    return _read_simple_labels(p)


def _resample(img, ref_img, resample_from_to):
    if img.shape == ref_img.shape and np.allclose(img.affine, ref_img.affine, atol=1e-4):
        return img
    return resample_from_to(img, (ref_img.shape, ref_img.affine), order=0)


def _world_x_grid(shape, affine) -> np.ndarray:
    i = np.arange(shape[0], dtype=np.float32)[:, None, None]
    j = np.arange(shape[1], dtype=np.float32)[None, :, None]
    k = np.arange(shape[2], dtype=np.float32)[None, None, :]
    return affine[0, 0] * i + affine[0, 1] * j + affine[0, 2] * k + affine[0, 3]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build fixed non-overlapping cbgtc_v1. Tian 3T S3 supplies the finest "
            "striatal parcellation; Tian 3T S4 supplies the finest thalamic, "
            "hippocampal and amygdala labels. S4 defines the target grid. "
            "Schaefer100, S3 and CIT168 are nearest-neighbor resampled to it; "
            "CIT168 specialized BG nuclei override other labels."
        )
    )
    ap.add_argument("--tian-s3-atlas", required=True, help="Tian_Subcortex_S3_3T NIfTI in desired MNI space")
    ap.add_argument("--tian-s3-labels", required=True, help="Official Tian S3 3T label text file")
    ap.add_argument("--tian-s4-atlas", required=True, help="Tian_Subcortex_S4_3T NIfTI in desired MNI space")
    ap.add_argument("--tian-s4-labels", required=True, help="Official Tian S4 3T label text file")
    ap.add_argument("--schaefer-atlas", required=True, help="Schaefer2018 100Parcels7Networks integer NIfTI in same template space")
    ap.add_argument("--schaefer-labels", required=True, help="TemplateFlow/CBIG Schaefer label TSV/TXT")
    ap.add_argument("--cit168-atlas", required=True, help="CIT168 integer dseg NIfTI; bilateral labels are OK")
    ap.add_argument("--cit168-labels", required=True, help="CIT168 labels.txt")
    ap.add_argument("--space", required=True, choices=["MNI152NLin6Asym", "MNI152NLin2009cAsym"])
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    nib, resample_from_to = _lazy_import_nibabel()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_schema_bundle(out)

    s4_img = nib.load(args.tian_s4_atlas)
    if len(s4_img.shape) != 3:
        raise ValueError("Tian S4 atlas must be a 3D integer label image")
    s3_img = _resample(nib.load(args.tian_s3_atlas), s4_img, resample_from_to)
    sch_img = _resample(nib.load(args.schaefer_atlas), s4_img, resample_from_to)
    cit_img = _resample(nib.load(args.cit168_atlas), s4_img, resample_from_to)

    s3 = np.rint(np.asarray(s3_img.dataobj)).astype(np.int32)
    s4 = np.rint(np.asarray(s4_img.dataobj)).astype(np.int32)
    sch = np.rint(np.asarray(sch_img.dataobj)).astype(np.int32)
    cit = np.rint(np.asarray(cit_img.dataobj)).astype(np.int32)

    s3_labels = _read_simple_labels(args.tian_s3_labels)
    s4_labels = _read_simple_labels(args.tian_s4_labels)
    sch_labels = _read_schaefer_tsv(args.schaefer_labels)
    cit_labels = _read_simple_labels(args.cit168_labels)
    s3_by_name = {v: k for k, v in s3_labels.items()}
    s4_by_name = {v: k for k, v in s4_labels.items()}
    cit_by_name = {v: k for k, v in cit_labels.items()}

    if len([x for x in np.unique(sch) if x != 0]) != 100:
        raise ValueError("Schaefer source must contain exactly 100 nonzero parcels")
    if set(range(1, 101)) - set(sch_labels):
        raise ValueError("Schaefer labels must define indices 1..100")

    rois = canonical_rois()
    out_data = np.zeros(s4.shape, dtype=np.int16)
    world_x = _world_x_grid(s4.shape, s4_img.affine)
    qc = {
        "space": args.space,
        "tian_s3": str(args.tian_s3_atlas),
        "tian_s4": str(args.tian_s4_atlas),
        "overwrites": {},
        "voxel_counts": {},
    }

    # Cortex.
    for out_label, roi in enumerate(rois, start=1):
        if roi.group != "C":
            continue
        src_idx = int(roi.source_label)
        m = sch == src_idx
        if not m.any():
            raise ValueError(f"empty Schaefer source label {src_idx}: {sch_labels.get(src_idx)}")
        out_data[m] = out_label
        qc["voxel_counts"][roi.roi_id] = int(m.sum())

    # Tian: S3 only for striatum; S4 for thalamus/amygdala/hippocampus.
    for out_label, roi in enumerate(rois, start=1):
        if roi.source_atlas == "Tian2020_3T_S3":
            src_idx = s3_by_name.get(roi.source_label)
            source = s3
        elif roi.source_atlas == "Tian2020_3T_S4":
            src_idx = s4_by_name.get(roi.source_label)
            source = s4
        else:
            continue
        if src_idx is None:
            raise KeyError(f"{roi.source_atlas} label not found: {roi.source_label}")
        m = source == src_idx
        if not m.any():
            raise ValueError(f"empty {roi.source_atlas} source label {roi.source_label}")
        overwritten = int(np.count_nonzero(m & (out_data != 0)))
        out_data[m] = out_label
        qc["overwrites"][roi.roi_id] = overwritten
        qc["voxel_counts"][roi.roi_id] = int(m.sum())

    # CIT168 specialized nuclei. Source labels may be bilateral, so split by
    # world x coordinate to preserve stable left/right node identities.
    for out_label, roi in enumerate(rois, start=1):
        if roi.source_atlas != "CIT168_v1.1":
            continue
        src_idx = cit_by_name.get(roi.source_label)
        if src_idx is None:
            raise KeyError(f"CIT168 label not found: {roi.source_label}")
        hemi = world_x < 0 if roi.hemisphere == "L" else world_x > 0
        m = (cit == src_idx) & hemi
        if not m.any():
            raise ValueError(f"empty CIT168 source label {roi.source_label} {roi.hemisphere}")
        overwritten = int(np.count_nonzero(m & (out_data != 0)))
        out_data[m] = out_label
        qc["overwrites"][roi.roi_id] = overwritten
        qc["voxel_counts"][roi.roi_id] = int(m.sum())

    present = set(int(x) for x in np.unique(out_data) if x != 0)
    expected = set(range(1, len(rois) + 1))
    missing = sorted(expected - present)
    if missing:
        raise ValueError(f"final atlas has empty canonical labels: {missing}")

    out_name = f"cbgtc_v1_{args.space}_dseg.nii.gz"
    header = s4_img.header.copy()
    header.set_data_dtype(np.int16)
    nib.save(nib.Nifti1Image(out_data, s4_img.affine, header), out / out_name)
    (out / "cbgtc_v1_build_qc.json").write_text(json.dumps(qc, indent=2) + "\n")
    print(f"wrote {out / out_name}")
    print(f"canonical ROIs: {len(rois)}")


if __name__ == "__main__":
    main()
