"""SeisBench-format and STEAD-layout reader on small fixtures that follow the real file layouts."""

import csv
import math

import h5py
import numpy as np
import pytest
import torch

from pyhazards.datasets import load_dataset
from pyhazards.datasets.earthquake import parse_trace_name


def _write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _seisbench_chunk(root, chunk="", n=5, length=600, offset=0.0, data_format=True):
    """SeisBench layout: trace blocks ``bucket<k>`` of shape (N, C, W) and a data_format group."""
    rng = np.random.default_rng(len(chunk))
    blocks = {"bucket0": rng.standard_normal((3, 3, length)), "bucket1": rng.standard_normal((n - 3, 3, length + 20))}
    rows = []
    splits = ["train", "dev", "test", "train", "test"]
    for k in range(n):
        block, i = ("bucket0", k) if k < 3 else ("bucket1", k - 3)
        rows.append(
            {
                "trace_start_time": "2020-01-01T00:00:00",
                "trace_name": f"{block}${i},:3,:{length}",
                "trace_sampling_rate_hz": "100.0",
                "trace_p_arrival_sample": "" if k == 4 else str(100 + k),
                "trace_s_arrival_sample": "" if k == 4 else str(300 + k),
                "trace_category": "noise" if k == 4 else "earthquake_local",
                "split": splits[k],
            }
        )
    _write_csv(root / f"metadata{chunk}.csv", rows)
    with h5py.File(root / f"waveforms{chunk}.hdf5", "w") as handle:
        for name, array in blocks.items():
            handle.create_dataset(f"data/{name}", data=(array + offset).astype("float32"))
        if data_format:
            group = handle.create_group("data_format")
            group["dimension_order"] = "CW"
            group["component_order"] = "ZNE"
            group["sampling_rate"] = 100
            group["measurement"] = "velocity"
            group["unit"] = "counts"
            group["instrument_response"] = "not restituted"
    return blocks, rows


def test_trace_block_names_are_parsed():
    assert parse_trace_name("bucket3$12,:3,:6000") == ("bucket3", (12, slice(None, 3), slice(None, 6000)))
    assert parse_trace_name("b$1,0:3:1") == ("b", (1, slice(0, 3, 1)))
    assert parse_trace_name("plain_trace") == ("plain_trace", None)
    with pytest.raises(ValueError):
        parse_trace_name("b$1,x")


def test_seisbench_format_with_trace_blocks(tmp_path):
    blocks, _ = _seisbench_chunk(tmp_path)
    bundle = load_dataset("seisbench_waveforms", path=str(tmp_path)).load()
    assert bundle.metadata["component_order"] == "ZNE"
    assert bundle.metadata["sampling_rate"] == 100.0
    assert bundle.metadata["synthetic"] is False
    train, val, test = (bundle.get_split(name) for name in ("train", "val", "test"))
    assert [len(train.inputs), len(val.inputs), len(test.inputs)] == [2, 1, 2]  # "dev" -> val
    torch.testing.assert_close(train.inputs[0], torch.from_numpy(blocks["bucket0"][0]).float())
    torch.testing.assert_close(train.inputs[1], torch.from_numpy(blocks["bucket1"][0, :, :600]).float())
    assert train.targets.tolist() == [[100.0, 300.0], [103.0, 303.0]]
    assert test.targets[1].isnan().all()  # noise trace
    assert test.metadata["trace_names"] == ["bucket0$2,:3,:600", "bucket1$1,:3,:600"]


def test_chunks_window_category_and_limits(tmp_path):
    _seisbench_chunk(tmp_path, "a")
    _seisbench_chunk(tmp_path, "b", offset=10.0)
    (tmp_path / "chunks").write_text("a\nb\n", encoding="utf-8")
    bundle = load_dataset("seisbench_waveforms", path=str(tmp_path), window_samples=250, window_start=200).load()
    test = bundle.get_split("test")
    assert test.inputs.shape == (4, 3, 250)
    assert test.targets[0, 0].isnan() and test.targets[0, 1].item() == 102.0  # P before the window
    one = load_dataset("seisbench_waveforms", path=str(tmp_path), trace_category="noise").load()
    assert len(one.get_split("test").inputs) == 2 and len(one.get_split("train").inputs) == 0
    micro = load_dataset("seisbench_waveforms", path=str(tmp_path), max_traces=1).load()
    assert [len(micro.get_split(s).inputs) for s in ("train", "val", "test")] == [1, 1, 1]
    padded = load_dataset("seisbench_waveforms", path=str(tmp_path), window_samples=700).load()
    assert padded.get_split("train").inputs[0, :, 600:].abs().sum() == 0


def test_unblocked_traces_without_data_format_need_component_order(tmp_path):
    rng = np.random.default_rng(1)
    rows = []
    with h5py.File(tmp_path / "waveforms.hdf5", "w") as handle:
        for k in range(3):
            name = f"IV.ABC..HH.{k}"
            handle.create_dataset(f"data/{name}", data=rng.standard_normal((3, 1200)).astype("float32"))
            rows.append({"trace_name": name, "trace_dt_s": "0.01", "trace_P_arrival_sample": "150",
                         "trace_S_arrival_sample": "450.0", "split": "test"})
    _write_csv(tmp_path / "metadata.csv", rows)
    with pytest.raises(ValueError, match="component_order"):
        load_dataset("seisbench_waveforms", path=str(tmp_path)).load()
    bundle = load_dataset("seisbench_waveforms", path=str(tmp_path), preset="instance").load()  # INSTANCE: E, N, Z
    assert bundle.metadata["component_order"] == "ENZ"
    assert bundle.get_split("test").targets.tolist() == [[150.0, 450.0]] * 3


def test_stead_layout_and_test_split(tmp_path):
    rng = np.random.default_rng(2)
    names = ["AB.STA1_20190101000000_EV", "AB.STA2_20190101000000_EV", "AB.STA3_201901010000_NO"]
    rows = []
    arrays = {}
    with h5py.File(tmp_path / "merged.hdf5", "w") as handle:
        for k, name in enumerate(names):
            arrays[name] = rng.standard_normal((6000, 3)).astype("float32")  # (samples, E/N/Z)
            dataset = handle.create_dataset(f"data/{name}", data=arrays[name])
            noise = name.endswith("_NO")
            dataset.attrs["p_arrival_sample"] = "None" if noise else 500.0 + k
            rows.append({"network_code": "AB", "p_arrival_sample": "" if noise else str(500.0 + k),
                         "s_arrival_sample": "" if noise else str(900.0 + k),
                         "trace_category": "noise" if noise else "earthquake_local", "trace_name": name})
    _write_csv(tmp_path / "merged.csv", rows)
    np.save(tmp_path / "test.npy", np.array(names[1:]))
    bundle = load_dataset("seisbench_waveforms", path=str(tmp_path), preset="stead", test_trace_names=str(tmp_path / "test.npy")).load()
    assert bundle.metadata["component_order"] == "ENZ" and bundle.metadata["layout"] == "stead"
    train, test = bundle.get_split("train"), bundle.get_split("test")
    torch.testing.assert_close(train.inputs[0], torch.from_numpy(arrays[names[0]].T))
    assert train.targets.tolist() == [[500.0, 900.0]]
    assert test.targets[0].tolist() == [501.0, 901.0] and test.targets[1].isnan().all()
    everything = load_dataset("seisbench_waveforms", path=str(tmp_path), layout="stead").load()
    assert len(everything.get_split("test").inputs) == 3


def test_reader_errors(tmp_path):
    with pytest.raises(ValueError, match="path="):
        load_dataset("seisbench_waveforms").load()
    with pytest.raises(FileNotFoundError):
        load_dataset("seisbench_waveforms", path=str(tmp_path / "missing")).load()
    with pytest.raises(FileNotFoundError):
        load_dataset("seisbench_waveforms", path=str(tmp_path)).load()
    with pytest.raises(ValueError, match="preset"):
        load_dataset("seisbench_waveforms", path=str(tmp_path), preset="nope").load()
    _seisbench_chunk(tmp_path)
    with pytest.raises(ValueError, match="window_samples"):
        rows_mixed = tmp_path / "metadata.csv"
        text = rows_mixed.read_text(encoding="utf-8").replace("bucket1$0,:3,:600", "bucket1$0,:3,:620")
        rows_mixed.write_text(text, encoding="utf-8")
        load_dataset("seisbench_waveforms", path=str(tmp_path)).load()


def test_synthetic_picking_windows_are_labelled_and_deterministic():
    first = load_dataset("earthquake_waveforms_synthetic", micro=True).load()
    second = load_dataset("earthquake_waveforms_synthetic", micro=True).load()
    torch.testing.assert_close(first.get_split("train").inputs, second.get_split("train").inputs)
    test = first.get_split("test")
    assert test.inputs.shape == (4, 3, 6000)
    assert first.metadata["synthetic"] is True and first.metadata["component_order"] == "ZNE"
    events = ~test.targets.isnan().any(dim=1)
    assert events.any() and (~events).any()
    assert bool((test.targets[events, 1] > test.targets[events, 0]).all())
    short = load_dataset("earthquake_waveforms_synthetic", samples=6, length=512).load()
    arrivals = short.get_split("train").targets
    finite = arrivals[~arrivals.isnan()]
    assert finite.max() < 512 and not math.isnan(float(finite.min()))
