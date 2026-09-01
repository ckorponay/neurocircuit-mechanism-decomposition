from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence
import hashlib
import numpy as np


DEFAULT_GROUP_ORDER = ("C", "S", "Th", "A", "H", "GPe", "GPi", "STN", "MB")


def roi_schema_hash(roi_ids: Sequence[str]) -> str:
    """Stable short checksum of ROI identity AND order."""
    payload = "\n".join(str(x) for x in roi_ids).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def load_roi_ids(path: str | Path) -> list[str]:
    """
    Load a canonical ROI schema from a one-column text file/TSV or a TSV whose
    first column is the ROI id. Blank/comment lines are ignored.
    """
    ids = []
    with Path(path).open("r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            ids.append(s.split("\t")[0].split(",")[0])
    if not ids:
        raise ValueError(f"No ROI ids found in {path}")
    if len(ids) != len(set(ids)):
        raise ValueError("ROI schema contains duplicate identifiers")
    return ids


@dataclass(frozen=True)
class TimeseriesRecord:
    """Canonical per-run input. timeseries is [R,T] float32."""

    subject_id: str
    visit_id: str
    run_id: str
    dataset: str
    tr_seconds: float
    timeseries: np.ndarray
    roi_ids: Sequence[str]
    valid_mask: np.ndarray | None = None
    site: str | None = None
    scanner: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self, expected_roi_ids: Sequence[str] | None = None) -> None:
        x = np.asarray(self.timeseries)
        if x.ndim != 2:
            raise ValueError(f"timeseries must be [R,T], got {x.shape}")
        if x.dtype != np.float32:
            raise ValueError(f"timeseries must be float32, got {x.dtype}")
        if self.tr_seconds <= 0:
            raise ValueError("tr_seconds must be > 0")
        if len(self.roi_ids) != x.shape[0]:
            raise ValueError("roi_ids length must equal R")
        if len(set(self.roi_ids)) != len(self.roi_ids):
            raise ValueError("roi_ids must be unique")
        if self.valid_mask is not None:
            m = np.asarray(self.valid_mask)
            if m.shape != (x.shape[1],) or m.dtype != np.bool_:
                raise ValueError("valid_mask must be bool [T]")
        if expected_roi_ids is not None and list(self.roi_ids) != list(expected_roi_ids):
            raise ValueError(
                "ROI identity/order differs from canonical schema. Do not train across "
                "runs/subjects whose parcellations were independently refit."
            )

    @property
    def schema_hash(self) -> str:
        return roi_schema_hash(self.roi_ids)


def load_grouped_npz(
    path: str | Path,
    *,
    subject_id: str,
    visit_id: str,
    run_id: str,
    dataset: str,
    tr_seconds: float,
    group_order: Sequence[str] = DEFAULT_GROUP_ORDER,
    expected_roi_ids: Sequence[str] | None = None,
    site: str | None = None,
    scanner: str | None = None,
) -> TimeseriesRecord:
    """
    Load existing repo NPZ convention (each group stored [T,R_group]), concatenate
    groups deterministically, and emit canonical [R,T].

    IMPORTANT: the generated names (e.g. S_0) only encode array position. They do
    NOT make independently fitted KMeans/supervoxel solutions comparable. For
    multi-subject production analyses, use a fixed atlas/parcellation or fit a
    reference clustering ONCE and reuse the same voxel labels everywhere; then
    pass `expected_roi_ids` to fail loudly if order changes.
    """
    z = np.load(path)
    arrays = []
    roi_ids = []
    T = None

    for key in group_order:
        if key not in z:
            continue
        a = np.asarray(z[key], dtype=np.float32)
        if a.ndim == 1:
            a = a[:, None]
        if a.ndim != 2:
            raise ValueError(f"{key} must be [T,R_group], got {a.shape}")
        if T is None:
            T = a.shape[0]
        elif a.shape[0] != T:
            raise ValueError(f"{key} has T={a.shape[0]}, expected {T}")
        arrays.append(a)
        roi_ids.extend([f"{key}_{i}" for i in range(a.shape[1])])

    if not arrays:
        raise ValueError(f"No recognized ROI groups in {path}")

    tr = np.concatenate(arrays, axis=1).T.astype(np.float32, copy=False)
    rec = TimeseriesRecord(
        subject_id=subject_id,
        visit_id=visit_id,
        run_id=run_id,
        dataset=dataset,
        tr_seconds=float(tr_seconds),
        timeseries=tr,
        roi_ids=roi_ids,
        site=site,
        scanner=scanner,
        metadata={"source_npz": str(path), "roi_schema_hash": roi_schema_hash(roi_ids)},
    )
    rec.validate(expected_roi_ids)
    return rec
