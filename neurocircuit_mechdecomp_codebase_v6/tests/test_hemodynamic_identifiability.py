import numpy as np
import torch

from neurocircuit.data.hemodynamics import normalize_hrf_kernels
from neurocircuit.models.observation_model import HemodynamicObservationModel


def test_hrf_shape_gain_split_reconstructs_same_neural_bold():
    raw = np.array([[0.0, 1.0, 2.0, 1.0]], dtype=np.float32)
    shape, gain = normalize_hrf_kernels(raw)
    assert np.allclose(np.abs(shape).sum(axis=-1), 1.0)

    x = torch.zeros(1, 1, 8)
    x[0, 0, 1] = 1.0
    obs = HemodynamicObservationModel(include_systemic=False)
    y_raw = obs(x, hrf_kernel=torch.from_numpy(raw), tr_seconds=1.0)["neural_bold"]
    y_split = obs(
        x,
        hrf_kernel=torch.from_numpy(shape),
        hrf_gain=torch.from_numpy(gain),
        tr_seconds=1.0,
    )["neural_bold"]
    assert torch.allclose(y_raw, y_split, atol=1e-6)
