from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from neurocircuit.data.hemodynamics import normalize_hrf_kernels


def load_vector(path: str, name: str) -> np.ndarray:
    p = Path(path)
    if p.suffix == ".npy":
        a = np.load(p)
    elif p.suffix == ".npz":
        z = np.load(p)
        if name in z:
            a = z[name]
        elif len(z.files) == 1:
            a = z[z.files[0]]
        else:
            raise ValueError(f"{p}: specify an array whose NPZ key is {name}")
    else:
        # Text/TSV input; gzip is handled transparently by numpy only poorly, so
        # use loadtxt which supports filenames including .gz.
        a = np.loadtxt(p)
    return np.asarray(a, dtype=np.float32).squeeze()


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Assemble the canonical per-run hemodynamics NPZ after rsHRF and RAPIDTIDE "
            "have been reduced into the SAME canonical ROI order as the NMD neural input."
        )
    )
    ap.add_argument("--hrf-kernel", required=True, help=".npy/.npz [R,K] rsHRF kernels")
    ap.add_argument("--systemic-waveform", required=True, help="RAPIDTIDE final regressor [T]")
    ap.add_argument("--vascular-delay", required=True, help="ROI RAPIDTIDE delay in seconds [R]")
    ap.add_argument("--vascular-amplitude", required=True, help="ROI systemic fitted amplitude [R]")
    ap.add_argument("--rapidtide-r2", default=None, help="Optional ROI RAPIDTIDE R^2 [R]")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    hrf = np.asarray(np.load(args.hrf_kernel), dtype=np.float32)
    if hrf.ndim != 2:
        raise ValueError(f"hrf kernel must be [R,K], got {hrf.shape}")
    hrf_shape, hrf_gain = normalize_hrf_kernels(hrf)

    systemic = load_vector(args.systemic_waveform, "systemic_waveform")
    delay = load_vector(args.vascular_delay, "vascular_delay_seconds")
    amp = load_vector(args.vascular_amplitude, "vascular_amplitude")
    r2 = None if args.rapidtide_r2 is None else load_vector(args.rapidtide_r2, "rapidtide_r2")

    R = hrf_shape.shape[0]
    if delay.shape != (R,) or amp.shape != (R,):
        raise ValueError(f"delay/amplitude must each be [R={R}], got {delay.shape}/{amp.shape}")
    if systemic.ndim != 1:
        raise ValueError("systemic waveform must be one-dimensional")
    if r2 is not None and r2.shape != (R,):
        raise ValueError(f"rapidtide R2 must be [R={R}], got {r2.shape}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(
        hrf_kernel=hrf_shape,
        hrf_gain=hrf_gain,
        systemic_waveform=systemic,
        vascular_delay_seconds=delay,
        vascular_amplitude=amp,
    )
    if r2 is not None:
        payload["rapidtide_r2"] = r2
    np.savez_compressed(out, **payload)
    print(f"wrote {out}: R={R}, T={systemic.size}, HRF_K={hrf_shape.shape[1]}")
    print("HRF kernels stored as unit-L1 shapes plus explicit hrf_gain to avoid scale confounding.")


if __name__ == "__main__":
    main()
