"""The ``seisbench_waveforms`` reader checked against SeisBench's own writer and reader.

SeisBench 0.12.6 (GPL-3.0, installed only in the oracle job) writes a SeisBench-format dataset with
its ``WaveformDataWriter`` (trace blocks / buckets, ``data_format`` group, its CSV conventions) from the
100 real STEAD traces that EQTransformer ships (``ModelsAndSampleData/100samples.*``, STEAD layout,
CC BY 4.0), converted the way SeisBench converts STEAD (``seisbench/data/stead.py``: E, N, Z ->
Z, N, E, samples x channels -> channels x samples, renamed arrival columns). PyHazards must read the
same waveforms and arrivals as SeisBench's ``WaveformDataset``, and the same waveforms as its own
reading of the original STEAD-layout files.
"""

from __future__ import annotations

import csv

import h5py
import numpy as np
import pytest
import torch

from oracle_utils import oracle_package, oracle_repo
from pyhazards.datasets import load_dataset


@pytest.fixture(scope="module")
def stead_sample():
    return oracle_repo("EQTransformer") / "ModelsAndSampleData"


@pytest.fixture(scope="module")
def seisbench_copy(tmp_path_factory, stead_sample):
    oracle_package("seisbench", "0.12.6", "requirements-earthquake.txt")
    import seisbench.data as sbd

    root = tmp_path_factory.mktemp("stead_seisbench")
    with (stead_sample / "100samples.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    splits = ["train", "dev", "test", "test"]
    with h5py.File(stead_sample / "100samples.hdf5", "r") as source, sbd.WaveformDataWriter(
        root / "metadata.csv", root / "waveforms.hdf5"
    ) as writer:
        writer.data_format = {
            "dimension_order": "CW",
            "component_order": "ZNE",
            "sampling_rate": 100,
            "measurement": "velocity",
            "unit": "counts",
            "instrument_response": "not restituted",
        }
        writer.bucket_size = 16  # several trace blocks
        for k, row in enumerate(rows):
            waveform = source["data"][row["trace_name"]][()].T[[2, 1, 0]]  # (6000, ENZ) -> (ZNE, 6000)
            metadata = {
                "trace_name_original": row["trace_name"],
                "trace_p_arrival_sample": float(row["p_arrival_sample"]),
                "trace_s_arrival_sample": float(row["s_arrival_sample"]),
                "trace_category": row["trace_category"],
                "split": splits[k % 4],
            }
            writer.add_trace(metadata, waveform)
    return root


def test_reads_what_seisbench_reads(seisbench_copy):
    import seisbench.data as sbd

    reference = sbd.WaveformDataset(seisbench_copy, component_order="ZNE", dimension_order="NCW")
    bundle = load_dataset("seisbench_waveforms", path=str(seisbench_copy)).load()
    assert bundle.metadata["component_order"] == "ZNE" and bundle.metadata["sampling_rate"] == 100.0
    metadata = reference.metadata
    assert any("$" in name for name in metadata["trace_name"])  # trace blocks were written
    for split, name in (("train", "train"), ("dev", "val"), ("test", "test")):
        mask = (metadata["split"] == split).to_numpy()
        expected = reference.get_waveforms(np.flatnonzero(mask))
        data = bundle.get_split(name)
        torch.testing.assert_close(data.inputs, torch.from_numpy(expected).float(), rtol=0, atol=0)
        arrivals = metadata.loc[mask, ["trace_p_arrival_sample", "trace_s_arrival_sample"]].to_numpy()
        np.testing.assert_array_equal(data.targets.numpy(), arrivals.astype("float32"))
        assert data.metadata["trace_names"] == list(metadata.loc[mask, "trace_name"])


def test_seisbench_copy_equals_the_original_stead_files(seisbench_copy, stead_sample):
    converted = load_dataset("seisbench_waveforms", path=str(seisbench_copy)).load()
    original = load_dataset("seisbench_waveforms", path=str(stead_sample), preset="stead").load()
    assert original.metadata["component_order"] == "ENZ"
    stead = original.get_split("test")
    order = [2, 1, 0]  # ENZ -> ZNE
    for k, split in enumerate(("train", "val", "test")):
        indices = [i for i in range(100) if ["train", "val", "test", "test"][i % 4] == split]
        torch.testing.assert_close(converted.get_split(split).inputs, stead.inputs[indices][:, order], rtol=0, atol=0)
        torch.testing.assert_close(converted.get_split(split).targets, stead.targets[indices], rtol=0, atol=0)
