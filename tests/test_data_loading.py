"""Tests for src.data_loading streaming HDF5 writes."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from src import data_loading
from src.config import TrainingConfig
from src.data_loading import OUTPUT_DTYPE, _append_events, load_to_hdf5
from src.sample_info import SampleInfo


def _events(labels: list[int]) -> np.ndarray:
    events = np.zeros(len(labels), dtype=OUTPUT_DTYPE)
    events["is_signal"] = np.array(labels, dtype=np.int8)
    return events


def test_append_events_extends_resizable_dataset(tmp_path: Path) -> None:
    path = tmp_path / "events.h5"
    with h5py.File(path, "w") as f:
        dataset = f.create_dataset(
            "events",
            shape=(0,),
            maxshape=(None,),
            dtype=OUTPUT_DTYPE,
            chunks=True,
        )
        _append_events(dataset, _events([1, 0]))
        _append_events(dataset, _events([0]))

    with h5py.File(path, "r") as f:
        events = f["events"][:]

    assert len(events) == 3
    np.testing.assert_array_equal(events["is_signal"], [1, 0, 0])


def test_load_to_hdf5_writes_incrementally_and_replaces_final_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    output = tmp_path / "processed" / "events.h5"

    samples = {
        "sig": SampleInfo(dsid=1, is_signal=True, description="signal"),
        "bkg": SampleInfo(dsid=2, is_signal=False, description="background"),
    }
    for sample in samples.values():
        (raw_dir / f"mc_{sample.dsid}.root").touch()

    def fake_process_sample(
        path: Path,
        key: str,
        info: SampleInfo,
        lumi: float,
        cuts: dict[str, object],
        events: h5py.Dataset,
    ) -> tuple[int, int]:
        chunk = _events([1, 1] if info.is_signal else [0, 0, 0])
        _append_events(events, chunk)
        return len(chunk), int(chunk["is_signal"].sum())

    monkeypatch.setattr(data_loading, "SAMPLES", samples)
    monkeypatch.setattr(data_loading, "_process_sample", fake_process_sample)

    config = TrainingConfig(
        raw_dir=str(raw_dir),
        processed_path=str(output),
        lumi=36.1,
        cuts={},
    )
    load_to_hdf5(config)

    with h5py.File(output, "r") as f:
        assert len(f["events"]) == 5
        assert f["events"].dtype == OUTPUT_DTYPE
        assert f.attrs["n_signal"] == 2
        assert f.attrs["n_background"] == 3
        assert f.attrs["lumi_fb_inv"] == 36.1
    assert not output.with_name(f"{output.name}.tmp").exists()


def test_load_to_hdf5_raises_on_n_signal_mismatch(
    tmp_path: Path,
    monkeypatch,
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    output = tmp_path / "processed" / "events.h5"

    samples = {
        "sig": SampleInfo(dsid=1, is_signal=True, description="signal"),
        "bkg": SampleInfo(dsid=2, is_signal=False, description="background"),
    }
    for sample in samples.values():
        (raw_dir / f"mc_{sample.dsid}.root").touch()

    def fake_process_sample(
        path: Path,
        key: str,
        info: SampleInfo,
        lumi: float,
        cuts: dict[str, object],
        events: h5py.Dataset,
    ) -> tuple[int, int]:
        chunk = _events([1, 1] if info.is_signal else [0, 0, 0])
        _append_events(events, chunk)
        # Return a wrong n_signal count to trigger the validation check
        return len(chunk), 99

    monkeypatch.setattr(data_loading, "SAMPLES", samples)
    monkeypatch.setattr(data_loading, "_process_sample", fake_process_sample)

    config = TrainingConfig(
        raw_dir=str(raw_dir),
        processed_path=str(output),
        lumi=36.1,
        cuts={},
    )
    with pytest.raises(RuntimeError, match="n_signal mismatch"):
        load_to_hdf5(config)
    assert not output.with_name(f"{output.name}.tmp").exists()
