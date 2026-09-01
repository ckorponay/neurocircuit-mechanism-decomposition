import numpy as np
import pytest
from neurocircuit.data.schema import TimeseriesRecord, roi_schema_hash


def test_roi_schema_hash_is_order_sensitive():
    assert roi_schema_hash(["a", "b"]) != roi_schema_hash(["b", "a"])


def test_record_rejects_reordered_schema():
    r = TimeseriesRecord(
        subject_id="s",
        visit_id="v",
        run_id="r",
        dataset="d",
        tr_seconds=.72,
        timeseries=np.zeros((2, 10), dtype=np.float32),
        roi_ids=["a", "b"],
    )
    with pytest.raises(ValueError):
        r.validate(["b", "a"])
