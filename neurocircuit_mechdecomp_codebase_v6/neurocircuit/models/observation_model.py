from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def causal_hrf_convolution(x: torch.Tensor, hrf_kernel: torch.Tensor) -> torch.Tensor:
    """
    Causal ROI-wise HRF convolution.

    Parameters
    ----------
    x : [B, R, T]
        Latent neural state trajectory.
    hrf_kernel : [R, K] or [B, R, K]
        Fixed local HRF kernel sampled at the run TR. h[..., 0] is zero-lag.

    Returns
    -------
    [B, R, T] predicted neural BOLD component.
    """
    if x.ndim != 3:
        raise ValueError(f"x must be [B,R,T], got {tuple(x.shape)}")
    B, R, T = x.shape
    if hrf_kernel.ndim == 2:
        if hrf_kernel.shape[0] != R:
            raise ValueError("HRF region dimension must match x")
        h = hrf_kernel.unsqueeze(0).expand(B, -1, -1)
    elif hrf_kernel.ndim == 3:
        if hrf_kernel.shape[:2] != (B, R):
            raise ValueError("batched HRF must be [B,R,K]")
        h = hrf_kernel
    else:
        raise ValueError("hrf_kernel must be [R,K] or [B,R,K]")

    K = h.shape[-1]
    if K < 1:
        raise ValueError("HRF kernel must have at least one sample")

    # Windows are [x[t-K+1], ..., x[t]]. Reverse h so h[0] multiplies x[t].
    windows = F.pad(x, (K - 1, 0)).unfold(dimension=-1, size=K, step=1)
    return (windows * h.flip(-1).unsqueeze(2)).sum(dim=-1)


def delayed_systemic_component(
    systemic_waveform: torch.Tensor,
    vascular_delay_seconds: torch.Tensor,
    vascular_amplitude: torch.Tensor,
    tr_seconds: float,
) -> torch.Tensor:
    """
    Construct alpha_i * s(t - delta_i) using linear fractional-sample delay.

    `vascular_delay_seconds` is a RAPIDTIDE blood-arrival delay. It is NOT an
    HRF latency parameter.

    Inputs
    ------
    systemic_waveform : [B,T] or [T]
    vascular_delay_seconds : [B,R] or [R]
    vascular_amplitude : [B,R] or [R]
    """
    if tr_seconds <= 0:
        raise ValueError("tr_seconds must be > 0")

    s = systemic_waveform
    if s.ndim == 1:
        s = s.unsqueeze(0)
    if s.ndim != 2:
        raise ValueError("systemic_waveform must be [T] or [B,T]")

    d = vascular_delay_seconds
    a = vascular_amplitude
    if d.ndim == 1:
        d = d.unsqueeze(0)
    if a.ndim == 1:
        a = a.unsqueeze(0)
    if d.ndim != 2 or a.ndim != 2 or d.shape != a.shape:
        raise ValueError("vascular delay/amplitude must be matching [R] or [B,R]")

    B, T = s.shape
    Bd, R = d.shape
    if Bd not in (1, B):
        raise ValueError("delay batch must be 1 or match waveform batch")
    if a.shape[0] not in (1, B):
        raise ValueError("amplitude batch must be 1 or match waveform batch")
    d = d.expand(B, -1)
    a = a.expand(B, -1)

    t = torch.arange(T, device=s.device, dtype=s.dtype).view(1, 1, T)
    source_index = t - d.to(dtype=s.dtype).unsqueeze(-1) / float(tr_seconds)
    lo = torch.floor(source_index).long()
    hi = lo + 1
    frac = source_index - lo.to(source_index.dtype)

    valid_lo = (lo >= 0) & (lo < T)
    valid_hi = (hi >= 0) & (hi < T)
    lo_c = lo.clamp(0, T - 1)
    hi_c = hi.clamp(0, T - 1)

    s_exp = s.unsqueeze(1).expand(B, R, T)
    slo = torch.gather(s_exp, 2, lo_c) * valid_lo.to(s.dtype)
    shi = torch.gather(s_exp, 2, hi_c) * valid_hi.to(s.dtype)
    shifted = (1.0 - frac) * slo + frac * shi
    return a.to(dtype=s.dtype).unsqueeze(-1) * shifted


class HemodynamicObservationModel(nn.Module):
    """
    BOLD measurement model:

        y_i(t) = [h_i * x_i](t) + alpha_i s(t-delta_i) + epsilon_i(t)

    Identifiability rule:
    - h_i shape is estimated externally (e.g. rsHRF), normalized, and fixed.
    - optional hrf_gain_i is held separately so HRF amplitude does not silently
      rescale the latent neural trajectory.
    - s, delta_i, alpha_i are estimated externally (RAPIDTIDE) and passed fixed.
    - this module contains no trainable HRF or vascular timing parameters.
    """

    def __init__(self, include_systemic: bool = True):
        super().__init__()
        self.include_systemic = bool(include_systemic)

    def forward(
        self,
        x_neural: torch.Tensor,
        *,
        hrf_kernel: torch.Tensor,
        tr_seconds: float,
        hrf_gain: torch.Tensor | None = None,
        systemic_waveform: torch.Tensor | None = None,
        vascular_delay_seconds: torch.Tensor | None = None,
        vascular_amplitude: torch.Tensor | None = None,
        y_observed: torch.Tensor | None = None,
    ) -> dict:
        neural_bold = causal_hrf_convolution(x_neural, hrf_kernel)
        if hrf_gain is not None:
            g = hrf_gain
            if g.ndim == 1:
                g = g.unsqueeze(0)
            if g.ndim != 2 or g.shape[-1] != x_neural.shape[1]:
                raise ValueError("hrf_gain must be [R] or [B,R]")
            g = g.expand(x_neural.shape[0], -1).to(
                device=x_neural.device, dtype=x_neural.dtype
            )
            neural_bold = neural_bold * g.unsqueeze(-1)

        if self.include_systemic:
            required = (systemic_waveform, vascular_delay_seconds, vascular_amplitude)
            if any(v is None for v in required):
                raise ValueError(
                    "include_systemic=True requires systemic_waveform, "
                    "vascular_delay_seconds, and vascular_amplitude"
                )
            systemic_bold = delayed_systemic_component(
                systemic_waveform=systemic_waveform,
                vascular_delay_seconds=vascular_delay_seconds,
                vascular_amplitude=vascular_amplitude,
                tr_seconds=tr_seconds,
            )
        else:
            systemic_bold = torch.zeros_like(neural_bold)

        y_hat = neural_bold + systemic_bold
        out = {
            "neural_bold": neural_bold,
            "systemic_bold": systemic_bold,
            "y_hat": y_hat,
        }
        if y_observed is not None:
            if y_observed.shape != y_hat.shape:
                raise ValueError("y_observed must match predicted BOLD shape")
            out["residual"] = y_observed - y_hat
        return out
