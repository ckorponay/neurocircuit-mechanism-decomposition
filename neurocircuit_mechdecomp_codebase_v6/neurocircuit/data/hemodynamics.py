from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch


def normalize_hrf_kernels(
    hrf_kernel: np.ndarray,
    *,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Separate HRF shape from amplitude using discrete L1 normalization.

    Returns
    -------
    hrf_shape : [R,K], sum(abs(shape), axis=-1) ~= 1
    hrf_gain  : [R], original discrete L1 magnitude

    This removes a direct scale tradeoff between fixed rsHRF amplitude and the
    inferred latent neural state. hrf_gain can be retained as a local
    neurovascular phenotype. Cross-dataset comparisons of gain should still be
    calibrated because HRF sampling differs with TR.
    """
    h = np.asarray(hrf_kernel, dtype=np.float32)
    if h.ndim != 2:
        raise ValueError("hrf_kernel must be [R,K]")
    gain = np.sum(np.abs(h), axis=-1).astype(np.float32)
    safe = np.maximum(gain, eps)
    return (h / safe[:, None]).astype(np.float32), gain


@dataclass(frozen=True)
class HemodynamicInputs:
    """Fixed per-run measurement inputs for NMD state estimation."""

    hrf_kernel: np.ndarray                 # [R,K], normalized rsHRF shape
    systemic_waveform: np.ndarray          # [T], RAPIDTIDE final probe regressor
    vascular_delay_seconds: np.ndarray     # [R], RAPIDTIDE arrival delay
    vascular_amplitude: np.ndarray         # [R], fitted systemic amplitude
    hrf_gain: np.ndarray | None = None      # [R], separated local HRF magnitude
    rapidtide_r2: np.ndarray | None = None # [R], optional QC/phenotype

    def validate(self, n_regions: int, n_timepoints: int) -> None:
        h = np.asarray(self.hrf_kernel)
        s = np.asarray(self.systemic_waveform)
        d = np.asarray(self.vascular_delay_seconds)
        a = np.asarray(self.vascular_amplitude)
        if h.ndim != 2 or h.shape[0] != n_regions:
            raise ValueError("hrf_kernel must be [R,K]")
        if s.shape != (n_timepoints,):
            raise ValueError("systemic_waveform must be [T]")
        if d.shape != (n_regions,) or a.shape != (n_regions,):
            raise ValueError("vascular delay/amplitude must be [R]")
        if self.hrf_gain is not None and np.asarray(self.hrf_gain).shape != (n_regions,):
            raise ValueError("hrf_gain must be [R]")
        if self.rapidtide_r2 is not None and np.asarray(self.rapidtide_r2).shape != (n_regions,):
            raise ValueError("rapidtide_r2 must be [R]")

    def to_torch(self, device=None, dtype=torch.float32) -> dict:
        return {
            "hrf_kernel": torch.as_tensor(self.hrf_kernel, dtype=dtype, device=device),
            "hrf_gain": None if self.hrf_gain is None else torch.as_tensor(self.hrf_gain, dtype=dtype, device=device),
            "systemic_waveform": torch.as_tensor(self.systemic_waveform, dtype=dtype, device=device),
            "vascular_delay_seconds": torch.as_tensor(self.vascular_delay_seconds, dtype=dtype, device=device),
            "vascular_amplitude": torch.as_tensor(self.vascular_amplitude, dtype=dtype, device=device),
            "rapidtide_r2": None if self.rapidtide_r2 is None else torch.as_tensor(self.rapidtide_r2, dtype=dtype, device=device),
        }


def load_hemodynamics_npz(
    path: str | Path,
    *,
    normalize_hrf_if_gain_missing: bool = True,
) -> HemodynamicInputs:
    """
    Expected NPZ fields:
      hrf_kernel              [R,K]   from rsHRF (shape or raw kernel)
      hrf_gain                [R]     optional separated local HRF magnitude
      systemic_waveform       [T]     RAPIDTIDE final moving regressor
      vascular_delay_seconds  [R]     RAPIDTIDE arrival delay
      vascular_amplitude      [R]     fitted systemic contribution amplitude
      rapidtide_r2            [R]     optional

    Backward compatibility: if hrf_gain is absent and normalization is enabled,
    the loader splits the supplied raw HRF into unit-L1 shape + gain.
    """
    z = np.load(path)
    required = [
        "hrf_kernel",
        "systemic_waveform",
        "vascular_delay_seconds",
        "vascular_amplitude",
    ]
    missing = [k for k in required if k not in z]
    if missing:
        raise ValueError(f"Missing hemodynamic fields: {missing}")

    h = np.asarray(z["hrf_kernel"], dtype=np.float32)
    if "hrf_gain" in z:
        gain = np.asarray(z["hrf_gain"], dtype=np.float32)
    elif normalize_hrf_if_gain_missing:
        h, gain = normalize_hrf_kernels(h)
    else:
        gain = None

    return HemodynamicInputs(
        hrf_kernel=h,
        hrf_gain=gain,
        systemic_waveform=np.asarray(z["systemic_waveform"], dtype=np.float32),
        vascular_delay_seconds=np.asarray(z["vascular_delay_seconds"], dtype=np.float32),
        vascular_amplitude=np.asarray(z["vascular_amplitude"], dtype=np.float32),
        rapidtide_r2=(
            np.asarray(z["rapidtide_r2"], dtype=np.float32)
            if "rapidtide_r2" in z else None
        ),
    )
