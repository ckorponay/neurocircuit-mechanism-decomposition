from pathlib import Path
import gzip

import numpy as np

from neurocircuit.preprocessing.rapidtide_adapter import (
    build_rapidtide_command,
    find_rapidtide_outputs,
    load_bids_timeseries_column,
    load_roi_label_map,
    reduce_label_map,
    reduce_label_timeseries,
)


def test_build_hcp_rapidtide_command_uses_separate_denoise_source():
    cmd = build_rapidtide_command(
        "minimal.nii.gz",
        "out/sub-1_task-rest",
        denoise_source="fix.nii.gz",
        searchrange=(-7.5, 15.0),
        nprocs=2,
    )
    assert cmd[:3] == ["rapidtide", "minimal.nii.gz", "out/sub-1_task-rest"]
    assert "--denoising" in cmd
    assert cmd[cmd.index("--denoisesourcefile") + 1] == "fix.nii.gz"
    i = cmd.index("--searchrange")
    assert cmd[i + 1 : i + 3] == ["-7.5", "15.0"]


def test_load_filtered_regressor_column_from_gz(tmp_path: Path):
    p = tmp_path / "x.tsv.gz"
    with gzip.open(p, "wt") as f:
        f.write("raw\tfiltered\n1\t10\n2\t20\n")
    x = load_bids_timeseries_column(p)
    np.testing.assert_allclose(x, [10, 20])
    y = load_bids_timeseries_column(p, "raw")
    np.testing.assert_allclose(y, [1, 2])


def test_find_rapidtide_outputs_prefers_documented_R2_and_keeps_R(tmp_path: Path):
    prefix = tmp_path / "sub-1_task-rest"
    required = [
        "_desc-refinedmovingregressor_timeseries.tsv.gz",
        "_desc-maxtimerefined_map.nii.gz",
        "_desc-lfofilterCoeff_map.nii.gz",
        "_desc-lfofilterR2_map.nii.gz",
        "_desc-lfofilterR_map.nii.gz",
        "_desc-lfofilterCleaned_bold.nii.gz",
    ]
    for suffix in required:
        (Path(f"{prefix}{suffix}")).touch()
    out = find_rapidtide_outputs(prefix)
    assert out.r2_map.name.endswith("_desc-lfofilterR2_map.nii.gz")
    assert out.correlation_map.name.endswith("_desc-lfofilterR_map.nii.gz")
    assert out.cleaned_bold is not None


def test_fixed_label_reduction_and_timeseries_order():
    atlas = np.array([[1, 1], [2, 2]])
    scalar = np.array([[1.0, 3.0], [10.0, 14.0]])
    got = reduce_label_map(scalar, atlas, [2, 1], statistic="median")
    np.testing.assert_allclose(got, [12.0, 2.0])

    bold = np.stack([scalar, scalar + 1.0, scalar + 2.0], axis=-1)
    ts = reduce_label_timeseries(bold, atlas, [2, 1], statistic="mean")
    assert ts.shape == (3, 2)
    np.testing.assert_allclose(ts[:, 0], [12.0, 13.0, 14.0])
    np.testing.assert_allclose(ts[:, 1], [2.0, 3.0, 4.0])


def test_weighted_delay_and_roi_map_parser(tmp_path: Path):
    atlas = np.array([[1, 1], [2, 2]])
    delay = np.array([[0.0, 10.0], [2.0, 6.0]])
    weights = np.array([[1.0, 3.0], [1.0, 1.0]])
    got = reduce_label_map(delay, atlas, [1, 2], statistic="weighted_mean", weights=weights)
    np.testing.assert_allclose(got, [7.5, 4.0])

    p = tmp_path / "roi.tsv"
    p.write_text("roi_id\tlabel\nC_0\t1\nS_0\t2\n")
    ids, labels = load_roi_label_map(p)
    assert ids == ["C_0", "S_0"]
    np.testing.assert_array_equal(labels, [1, 2])
