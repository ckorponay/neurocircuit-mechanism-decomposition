import numpy as np
from neurocircuit.data.schema import load_grouped_npz

def test_grouped_npz_to_canonical_record(tmp_path):
    p = tmp_path / "run.npz"
    np.savez_compressed(
        p,
        C=np.zeros((10, 3), dtype=np.float32),
        S=np.ones((10, 2), dtype=np.float32),
    )
    r = load_grouped_npz(
        p,
        subject_id="s1",
        visit_id="v1",
        run_id="r1",
        dataset="HCP-YA",
        tr_seconds=0.72,
    )
    assert r.timeseries.shape == (5, 10)
    assert list(r.roi_ids) == ["C_0", "C_1", "C_2", "S_0", "S_1"]
