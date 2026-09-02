from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np

from neurocircuit.data.hemodynamics import normalize_hrf_kernels
from neurocircuit.data.schema import load_roi_ids, roi_schema_hash
from neurocircuit.preprocessing.rapidtide_adapter import (
    build_rapidtide_command,
    find_rapidtide_outputs,
    load_bids_timeseries_column,
    load_roi_label_map,
    reduce_label_map,
    reduce_label_timeseries,
    run_rapidtide,
)


def _load_nifti(path: str | Path):
    try:
        import nibabel as nib
    except ImportError as exc:
        raise RuntimeError(
            "NIfTI ROI reduction requires nibabel. Install the hemodynamics extras: "
            "python -m pip install -r requirements-hemodynamics.txt"
        ) from exc
    return nib.load(str(path))


def _same_grid(a, b, *, atol: float = 1e-4) -> None:
    if a.shape[:3] != b.shape[:3]:
        raise ValueError(f"Spatial shape mismatch: {a.shape[:3]} vs {b.shape[:3]}")
    if not np.allclose(a.affine, b.affine, atol=atol, rtol=0):
        raise ValueError("NIfTI affine mismatch; resample the atlas/maps to a common grid first")


def _load_hrf(path: str | Path) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npy":
        h = np.load(p)
    elif p.suffix == ".npz":
        z = np.load(p)
        if "hrf_kernel" in z:
            h = z["hrf_kernel"]
        elif len(z.files) == 1:
            h = z[z.files[0]]
        else:
            raise ValueError(f"{p}: NPZ has multiple arrays and no hrf_kernel key")
    else:
        h = np.loadtxt(p)
    return np.asarray(h, dtype=np.float32)


def _save_hemodynamics(
    out: Path,
    *,
    hrf_raw: np.ndarray,
    systemic: np.ndarray,
    delay: np.ndarray,
    amplitude: np.ndarray,
    r2: np.ndarray | None,
    roi_ids: list[str],
    tr_seconds: float,
) -> None:
    if hrf_raw.ndim != 2:
        raise ValueError(f"HRF must be [R,K], got {hrf_raw.shape}")
    R = len(roi_ids)
    if hrf_raw.shape[0] != R and hrf_raw.shape[1] == R:
        # rsHRF/matrix workflows commonly emit [K,R]; make this ergonomic but explicit.
        hrf_raw = hrf_raw.T
    if hrf_raw.shape[0] != R:
        raise ValueError(f"HRF has R={hrf_raw.shape[0]}, canonical ROI schema has R={R}")
    if delay.shape != (R,) or amplitude.shape != (R,):
        raise ValueError("delay/amplitude do not match canonical ROI count")
    if r2 is not None and r2.shape != (R,):
        raise ValueError("R2 does not match canonical ROI count")

    hrf_shape, hrf_gain = normalize_hrf_kernels(hrf_raw)
    payload = dict(
        hrf_kernel=hrf_shape.astype(np.float32),
        hrf_gain=hrf_gain.astype(np.float32),
        systemic_waveform=systemic.astype(np.float32),
        vascular_delay_seconds=delay.astype(np.float32),
        vascular_amplitude=amplitude.astype(np.float32),
        roi_ids=np.asarray(roi_ids),
        roi_schema_hash=np.asarray(roi_schema_hash(roi_ids)),
        tr_seconds=np.asarray(float(tr_seconds), dtype=np.float32),
    )
    if r2 is not None:
        payload["rapidtide_r2"] = r2.astype(np.float32)
    np.savez_compressed(out, **payload)


def _run_command_template(template: str, *, rshrf_input: Path, out_dir: Path, tr: float, roi_schema: Path) -> None:
    command = template.format(
        input=str(rshrf_input),
        out_dir=str(out_dir),
        tr=str(tr),
        roi_schema=str(roi_schema),
    )
    print(f"rsHRF command:\n{command}")
    subprocess.run(command, shell=True, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Prepare fixed per-run RAPIDTIDE/rsHRF inputs for NMD. Runs or reuses "
            "RAPIDTIDE, reduces vascular maps and cleaned BOLD into a fixed NIfTI "
            "label atlas, emits an rsHRF-ready [T,R] matrix, and optionally assembles "
            "the final hemodynamics.npz when an rsHRF HRF kernel is available."
        )
    )
    # Input/output and fixed ROI identity.
    ap.add_argument("--rapidtide-source", required=True, help="BOLD used to estimate sLFO/delay (HCP: preferably minimally processed NIfTI)")
    ap.add_argument("--denoise-source", default=None, help="Optional BOLD to receive final sLFO regression (HCP: FIX-denoised NIfTI)")
    ap.add_argument("--rapidtide-prefix", required=True, help="RAPIDTIDE output root/prefix")
    ap.add_argument("--atlas", required=True, help="Fixed integer-label NIfTI atlas in RAPIDTIDE output grid")
    ap.add_argument("--roi-map", required=True, help="TSV/CSV: roi_id, label")
    ap.add_argument("--roi-schema", required=True, help="Canonical one-column ROI id file")
    ap.add_argument("--tr-seconds", required=True, type=float)
    ap.add_argument("--out-dir", required=True)

    # RAPIDTIDE execution.
    ap.add_argument("--run-rapidtide", action="store_true", help="Actually invoke RAPIDTIDE before harvesting outputs")
    ap.add_argument("--dry-run", action="store_true", help="Print external command(s) without executing")
    ap.add_argument("--rapidtide-executable", default="rapidtide")
    ap.add_argument("--filterband", default="lfo")
    ap.add_argument("--searchrange", type=float, nargs=2, default=(-7.5, 15.0), metavar=("MIN", "MAX"))
    ap.add_argument("--ampthresh", type=float, default=0.15)
    ap.add_argument("--spatialfilt", type=float, default=3.0)
    ap.add_argument("--despecklepasses", type=int, default=4)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--nprocs", type=int, default=2)
    ap.add_argument("--outputlevel", choices=["min", "less", "normal", "more", "max"], default="normal")
    ap.add_argument("--rapidtide-extra-arg", action="append", default=[], help="Append one literal RAPIDTIDE CLI token; repeat as needed")

    # Reduction choices.
    ap.add_argument("--delay-summary", choices=["median", "mean", "r2_weighted_mean"], default="median")
    ap.add_argument("--map-summary", choices=["median", "mean"], default="median")
    ap.add_argument("--bold-summary", choices=["mean", "median"], default="mean")
    ap.add_argument("--systemic-column", default=None, help="Refined-regressor TSV column name/index; default prefers a filtered column")

    # rsHRF bridge. We deliberately do not hard-code an unstable package-internal API.
    ap.add_argument("--hrf-kernel", default=None, help="Existing rsHRF HRF matrix [R,K] or [K,R]; if supplied, final NPZ is written")
    ap.add_argument("--rshrf-command-template", default=None, help="Optional shell template using {input}, {out_dir}, {tr}, {roi_schema}")
    ap.add_argument("--rshrf-output-hrf", default=None, help="HRF matrix produced by --rshrf-command-template")
    args = ap.parse_args()

    if args.tr_seconds <= 0:
        ap.error("--tr-seconds must be > 0")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = Path(args.rapidtide_prefix)

    canonical_ids = load_roi_ids(args.roi_schema)
    map_ids, label_values = load_roi_label_map(args.roi_map)
    if map_ids != canonical_ids:
        raise ValueError(
            "ROI map identity/order differs from --roi-schema. Fix this before preprocessing; "
            "NMD will not silently reorder nodes."
        )

    command = build_rapidtide_command(
        args.rapidtide_source,
        prefix,
        denoise_source=args.denoise_source,
        rapidtide_executable=args.rapidtide_executable,
        filterband=args.filterband,
        searchrange=tuple(args.searchrange),
        ampthresh=args.ampthresh,
        spatialfilt=args.spatialfilt,
        despecklepasses=args.despecklepasses,
        passes=args.passes,
        nprocs=args.nprocs,
        outputlevel=args.outputlevel,
        extra_args=args.rapidtide_extra_arg,
    )
    if args.run_rapidtide:
        run_rapidtide(command, dry_run=args.dry_run)
        if args.dry_run:
            return

    outputs = find_rapidtide_outputs(prefix)
    systemic = load_bids_timeseries_column(outputs.refined_regressor, args.systemic_column)

    atlas_img = _load_nifti(args.atlas)
    delay_img = _load_nifti(outputs.delay_map)
    coeff_img = _load_nifti(outputs.coefficient_map)
    _same_grid(atlas_img, delay_img)
    _same_grid(atlas_img, coeff_img)
    atlas = np.asarray(atlas_img.dataobj)
    delay_data = np.asarray(delay_img.dataobj)
    coeff_data = np.asarray(coeff_img.dataobj)

    r2_data = None
    r2 = None
    r2_definition = None
    if outputs.r2_map is not None:
        r2_img = _load_nifti(outputs.r2_map)
        _same_grid(atlas_img, r2_img)
        r2_data = np.asarray(r2_img.dataobj)
        r2 = reduce_label_map(r2_data, atlas, label_values, statistic=args.map_summary)
        r2_definition = "ROI-reduced RAPIDTIDE lfofilterR2"
    elif outputs.correlation_map is not None:
        corr_img = _load_nifti(outputs.correlation_map)
        _same_grid(atlas_img, corr_img)
        corr_data = np.asarray(corr_img.dataobj)
        r2_data = np.square(corr_data)
        r2 = reduce_label_map(r2_data, atlas, label_values, statistic=args.map_summary)
        r2_definition = "ROI reduction of squared RAPIDTIDE lfofilterR (fallback)"

    if args.delay_summary == "r2_weighted_mean":
        if r2_data is None:
            raise ValueError("r2_weighted_mean requested but RAPIDTIDE lfofilterR2/R maps are absent")
        delay = reduce_label_map(delay_data, atlas, label_values, statistic="weighted_mean", weights=r2_data)
    else:
        delay = reduce_label_map(delay_data, atlas, label_values, statistic=args.delay_summary)
    amplitude = reduce_label_map(coeff_data, atlas, label_values, statistic=args.map_summary)

    # Fail loudly on empty ROIs rather than letting NaNs leak into state inference.
    for name, vec in [("vascular_delay_seconds", delay), ("vascular_amplitude", amplitude)]:
        bad = np.flatnonzero(~np.isfinite(vec))
        if bad.size:
            ids = [canonical_ids[i] for i in bad[:10]]
            raise ValueError(f"{name} contains empty/nonfinite ROIs: {ids}")

    np.save(out_dir / "vascular_delay_seconds.npy", delay)
    np.save(out_dir / "vascular_amplitude.npy", amplitude)
    np.save(out_dir / "systemic_waveform.npy", systemic)
    if r2 is not None:
        np.save(out_dir / "rapidtide_r2.npy", r2)

    rshrf_input = None
    if outputs.cleaned_bold is not None:
        cleaned_img = _load_nifti(outputs.cleaned_bold)
        _same_grid(atlas_img, cleaned_img)
        cleaned = np.asarray(cleaned_img.dataobj)
        if cleaned.ndim != 4:
            raise ValueError(f"Cleaned BOLD must be 4D NIfTI, got {cleaned.shape}")
        rshrf_matrix = reduce_label_timeseries(cleaned, atlas, label_values, statistic=args.bold_summary)
        if rshrf_matrix.shape[0] != systemic.shape[0]:
            raise ValueError(
                f"RAPIDTIDE refined regressor T={systemic.shape[0]} but cleaned BOLD T={rshrf_matrix.shape[0]}"
            )
        if not np.isfinite(rshrf_matrix).all():
            raise ValueError("rsHRF ROI matrix contains NaN/inf; check ROI coverage")
        rshrf_input = out_dir / "rshrf_input_cleaned_roi_TxR.npy"
        np.save(rshrf_input, rshrf_matrix.astype(np.float32))
        np.savetxt(out_dir / "rshrf_input_cleaned_roi_TxR.tsv", rshrf_matrix, delimiter="\t", fmt="%.8g")
    else:
        print(
            "WARNING: RAPIDTIDE cleaned BOLD is absent. Vascular maps were prepared, "
            "but the rsHRF-ready ROI matrix could not be generated. Use --denoising/" 
            "outputlevel >= less or provide an existing HRF kernel."
        )

    hrf_path = Path(args.hrf_kernel) if args.hrf_kernel else None
    if args.rshrf_command_template:
        if rshrf_input is None:
            raise ValueError("Cannot run rsHRF command: no cleaned ROI input was generated")
        _run_command_template(
            args.rshrf_command_template,
            rshrf_input=rshrf_input,
            out_dir=out_dir,
            tr=args.tr_seconds,
            roi_schema=Path(args.roi_schema),
        )
        if args.dry_run:
            return
        if not args.rshrf_output_hrf:
            raise ValueError("--rshrf-command-template requires --rshrf-output-hrf")
        hrf_path = Path(args.rshrf_output_hrf)

    hemo_path = None
    if hrf_path is not None:
        hrf = _load_hrf(hrf_path)
        hemo_path = out_dir / "hemodynamics.npz"
        _save_hemodynamics(
            hemo_path,
            hrf_raw=hrf,
            systemic=systemic,
            delay=delay,
            amplitude=amplitude,
            r2=r2,
            roi_ids=canonical_ids,
            tr_seconds=args.tr_seconds,
        )

    provenance = {
        "rapidtide_source": str(Path(args.rapidtide_source).resolve()),
        "denoise_source": str(Path(args.denoise_source).resolve()) if args.denoise_source else None,
        "rapidtide_prefix": str(prefix.resolve()),
        "rapidtide_command": command,
        "atlas": str(Path(args.atlas).resolve()),
        "roi_map": str(Path(args.roi_map).resolve()),
        "roi_schema": str(Path(args.roi_schema).resolve()),
        "roi_schema_hash": roi_schema_hash(canonical_ids),
        "n_regions": len(canonical_ids),
        "tr_seconds": args.tr_seconds,
        "systemic_column": args.systemic_column,
        "delay_summary": args.delay_summary,
        "map_summary": args.map_summary,
        "bold_summary": args.bold_summary,
        "rapidtide_outputs": {k: (str(v) if v is not None else None) for k, v in outputs.__dict__.items()},
        "r2_definition": r2_definition,
        "hrf_kernel_source": str(hrf_path.resolve()) if hrf_path is not None else None,
        "hemodynamics_npz": str(hemo_path.resolve()) if hemo_path is not None else None,
    }
    (out_dir / "hemodynamics_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")

    print(f"Prepared RAPIDTIDE physiology for R={len(canonical_ids)}, T={systemic.shape[0]}")
    print(f"ROI schema hash: {roi_schema_hash(canonical_ids)}")
    if rshrf_input is not None:
        print(f"rsHRF-ready cleaned ROI matrix: {rshrf_input}")
    if hemo_path is not None:
        print(f"Final NMD hemodynamics: {hemo_path}")
    else:
        print(
            "Final hemodynamics.npz not written yet. Run rsHRF on the cleaned [T,R] matrix "
            "and rerun this command with --hrf-kernel /path/to/hrf.npy (without --run-rapidtide)."
        )


if __name__ == "__main__":
    main()
