from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import csv
import gzip
import shlex
import subprocess

import numpy as np


@dataclass(frozen=True)
class RapidtideOutputs:
    """Subset of RAPIDTIDE outputs consumed by NMD."""

    refined_regressor: Path
    delay_map: Path
    coefficient_map: Path
    r2_map: Path | None
    correlation_map: Path | None
    cleaned_bold: Path | None


def build_rapidtide_command(
    source_bold: str | Path,
    output_prefix: str | Path,
    *,
    denoise_source: str | Path | None = None,
    rapidtide_executable: str = "rapidtide",
    filterband: str = "lfo",
    searchrange: tuple[float, float] = (-7.5, 15.0),
    ampthresh: float = 0.15,
    spatialfilt: float = 3.0,
    despecklepasses: int = 4,
    passes: int = 3,
    nprocs: int = 2,
    outputlevel: str = "normal",
    extra_args: Sequence[str] = (),
) -> list[str]:
    """
    Construct the reproducible RAPIDTIDE command used by NMD.

    `source_bold` is the dataset used to estimate the systemic waveform/delays.
    For HCP, this should preferentially be minimally processed BOLD; pass the
    FIX-denoised file via `denoise_source` so RAPIDTIDE estimates physiology on
    the minimally processed data but applies the final regression to FIX data.
    """
    cmd = [
        rapidtide_executable,
        str(source_bold),
        str(output_prefix),
        "--denoising",
        "--filterband", str(filterband),
        "--searchrange", str(searchrange[0]), str(searchrange[1]),
        "--ampthresh", str(ampthresh),
        "--spatialfilt", str(spatialfilt),
        "--despecklepasses", str(despecklepasses),
        "--passes", str(passes),
        "--outputlevel", str(outputlevel),
        "--nprocs", str(nprocs),
    ]
    if denoise_source is not None:
        cmd.extend(["--denoisesourcefile", str(denoise_source)])
    cmd.extend(str(x) for x in extra_args)
    return cmd


def run_rapidtide(command: Sequence[str], *, dry_run: bool = False) -> None:
    printable = " ".join(shlex.quote(str(x)) for x in command)
    print(f"RAPIDTIDE command:\n{printable}")
    if dry_run:
        return
    subprocess.run(list(command), check=True)


def _require_one(prefix: Path, suffixes: Sequence[str]) -> Path:
    candidates = []
    for suffix in suffixes:
        p = Path(f"{prefix}{suffix}")
        if p.exists():
            candidates.append(p)
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected exactly one RAPIDTIDE output for {prefix} among {list(suffixes)}; "
            f"found {[str(p) for p in candidates]}"
        )
    return candidates[0]


def _optional_one(prefix: Path, suffixes: Sequence[str]) -> Path | None:
    candidates = []
    for suffix in suffixes:
        p = Path(f"{prefix}{suffix}")
        if p.exists():
            candidates.append(p)
    if len(candidates) > 1:
        raise RuntimeError(f"Ambiguous RAPIDTIDE outputs: {[str(p) for p in candidates]}")
    return candidates[0] if candidates else None


def find_rapidtide_outputs(output_prefix: str | Path) -> RapidtideOutputs:
    """
    Resolve the current BIDS-style RAPIDTIDE filenames required by NMD.

    RAPIDTIDE 3.2 documents both lfofilterR2 and lfofilterR. NMD prefers the
    voxelwise R2 map and falls back to squaring R only for compatibility.
    """
    prefix = Path(output_prefix)
    return RapidtideOutputs(
        refined_regressor=_require_one(
            prefix,
            ["_desc-refinedmovingregressor_timeseries.tsv.gz",
             "_desc-refinedmovingregressor_timeseries.tsv"],
        ),
        delay_map=_require_one(prefix, ["_desc-maxtimerefined_map.nii.gz", "_desc-maxtimerefined_map.nii"]),
        coefficient_map=_require_one(prefix, ["_desc-lfofilterCoeff_map.nii.gz", "_desc-lfofilterCoeff_map.nii"]),
        r2_map=_optional_one(prefix, ["_desc-lfofilterR2_map.nii.gz", "_desc-lfofilterR2_map.nii"]),
        correlation_map=_optional_one(prefix, ["_desc-lfofilterR_map.nii.gz", "_desc-lfofilterR_map.nii"]),
        cleaned_bold=_optional_one(prefix, ["_desc-lfofilterCleaned_bold.nii.gz", "_desc-lfofilterCleaned_bold.nii"]),
    )


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt")
    return path.open("rt")


def load_bids_timeseries_column(path: str | Path, column: str | None = None) -> np.ndarray:
    """
    Load one numeric column from a RAPIDTIDE TSV/TSV.GZ.

    If `column` is omitted and multiple numeric columns are present, prefer a
    header containing 'filtered'; otherwise use the last numeric column. This
    is intentionally deterministic and prints the selected column upstream.
    """
    p = Path(path)
    with _open_text(p) as f:
        reader = csv.reader(f, delimiter="\t")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty timeseries file: {p}")

    # Detect header by asking whether every cell in the first row is numeric.
    def numeric(s: str) -> bool:
        try:
            float(s)
            return True
        except ValueError:
            return False

    has_header = not all(numeric(x) for x in rows[0])
    if has_header:
        header = [x.strip() for x in rows[0]]
        data_rows = rows[1:]
    else:
        header = [f"col{i}" for i in range(len(rows[0]))]
        data_rows = rows

    if not data_rows:
        raise ValueError(f"No samples in timeseries file: {p}")
    arr = np.asarray([[float(x) for x in row] for row in data_rows], dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[:, None]

    if column is not None:
        if column in header:
            idx = header.index(column)
        else:
            try:
                idx = int(column)
            except ValueError as exc:
                raise ValueError(f"Column {column!r} not in {header}") from exc
    else:
        preferred = [i for i, name in enumerate(header) if "filtered" in name.lower()]
        idx = preferred[-1] if preferred else arr.shape[1] - 1
    if idx < 0 or idx >= arr.shape[1]:
        raise IndexError(f"Column index {idx} outside 0..{arr.shape[1]-1}")
    return arr[:, idx].astype(np.float32, copy=False)


def load_roi_label_map(path: str | Path) -> tuple[list[str], np.ndarray]:
    """
    Load TSV/CSV containing `roi_id` and integer `label` columns.

    A two-column headerless file is also accepted as: roi_id<TAB>label.
    """
    p = Path(path)
    text = p.read_text().strip().splitlines()
    if not text:
        raise ValueError(f"Empty ROI label map: {p}")
    delim = "\t" if "\t" in text[0] else ","
    first = [x.strip() for x in text[0].split(delim)]
    header = [x.lower() for x in first]
    has_header = "roi_id" in header and "label" in header
    if has_header:
        roi_idx = header.index("roi_id")
        lab_idx = header.index("label")
        lines = text[1:]
    else:
        roi_idx, lab_idx = 0, 1
        lines = text

    roi_ids: list[str] = []
    labels: list[int] = []
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = [x.strip() for x in line.split(delim)]
        if len(parts) < 2:
            raise ValueError(f"Malformed ROI map line: {line}")
        roi_ids.append(parts[roi_idx])
        labels.append(int(parts[lab_idx]))
    if len(set(roi_ids)) != len(roi_ids):
        raise ValueError("Duplicate roi_id in ROI label map")
    if len(set(labels)) != len(labels):
        raise ValueError("Duplicate integer label in ROI label map")
    return roi_ids, np.asarray(labels, dtype=np.int64)


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not finite.any():
        return float("nan")
    w = weights[finite].astype(np.float64)
    v = values[finite].astype(np.float64)
    return float(np.sum(v * w) / np.sum(w))


def reduce_label_map(
    data: np.ndarray,
    atlas_labels: np.ndarray,
    label_values: Sequence[int],
    *,
    statistic: str = "median",
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Reduce a scalar spatial map into a fixed ordered ROI vector."""
    d = np.asarray(data)
    lab = np.asarray(atlas_labels)
    if d.shape != lab.shape:
        raise ValueError(f"map/atlas shapes differ: {d.shape} vs {lab.shape}")
    if weights is not None and np.asarray(weights).shape != d.shape:
        raise ValueError("weights must match map shape")

    out = np.empty(len(label_values), dtype=np.float32)
    for i, value in enumerate(label_values):
        m = lab == int(value)
        vals = d[m]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            out[i] = np.nan
            continue
        if statistic == "median":
            out[i] = np.median(vals)
        elif statistic == "mean":
            out[i] = np.mean(vals)
        elif statistic == "weighted_mean":
            if weights is None:
                raise ValueError("weighted_mean requires weights")
            out[i] = _weighted_mean(d[m], np.asarray(weights)[m])
        else:
            raise ValueError(f"Unknown statistic: {statistic}")
    return out


def reduce_label_timeseries(
    data_4d: np.ndarray,
    atlas_labels: np.ndarray,
    label_values: Sequence[int],
    *,
    statistic: str = "mean",
) -> np.ndarray:
    """Reduce a [X,Y,Z,T] BOLD image to [T,R] in fixed ROI order."""
    x = np.asarray(data_4d)
    lab = np.asarray(atlas_labels)
    if x.ndim != lab.ndim + 1 or x.shape[:-1] != lab.shape:
        raise ValueError(f"BOLD/atlas shapes differ: {x.shape} vs {lab.shape}")
    T = x.shape[-1]
    out = np.empty((T, len(label_values)), dtype=np.float32)
    for i, value in enumerate(label_values):
        vox = x[lab == int(value), :]
        if vox.size == 0:
            out[:, i] = np.nan
            continue
        if statistic == "mean":
            out[:, i] = np.nanmean(vox, axis=0)
        elif statistic == "median":
            out[:, i] = np.nanmedian(vox, axis=0)
        else:
            raise ValueError(f"Unknown statistic: {statistic}")
    return out
