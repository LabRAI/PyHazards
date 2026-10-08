"""NeuralHydrology LSTM / EA-LSTM, hydrological metrics and streamflow readers against the official code.

References (pinned in repos.yaml): neuralhydrology @ ea94a40 (BSD-3-Clause) for ``CudaLSTM``, ``EALSTM``,
``evaluation/metrics.py``, the ``CamelsUS`` / ``Caravan`` datasets and the ``Tester``; the paper code of
Kratzert et al. (2019), kratzert/ealstm_regional_modeling @ d118158 (Apache-2.0), for the layout of the
official HydroShare checkpoints. Real data: the 4 CAMELS-US basins (2000-2002) in NeuralHydrology's
``test/test_data`` and the real Caravan netCDF basin files (ERA5-Land + streamflow, 1981-2020) in
google-research/flood-forecasting's ``test/test_data/multimet``.

The 939 MB HydroShare archive with the 48 official 2019 runs (``kratzert2019_hydroshare_runs``,
``large: true``) gates the checkpoint and paper-number tests.
"""

from __future__ import annotations

import io
import pickle
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from oracle_utils import import_from, oracle_large_asset, oracle_repo
from pyhazards.benchmarks.flood import evaluate_streamflow
from pyhazards.datasets.flood import (
    KRATZERT2019_DYNAMIC_INPUTS,
    KRATZERT2019_STATIC_ATTRIBUTES,
    CamelsUSStreamflowDataset,
    CaravanStreamflowDataset,
)
from pyhazards.metrics import hydrology
from pyhazards.models import build_model
from pyhazards.models.neuralhydrology_lstm import convert_kratzert2019_state_dict

pytest.importorskip("xarray")


@pytest.fixture(scope="module")
def nh():
    """The NeuralHydrology package imported from the pinned checkout (kept importable for the module)."""
    root = oracle_repo("neuralhydrology")
    pytest.importorskip("ruamel.yaml")
    pytest.importorskip("numba")
    sys.path.insert(0, str(root))
    try:
        import neuralhydrology  # noqa: F401
        from neuralhydrology.datasetzoo.camelsus import CamelsUS
        from neuralhydrology.datasetzoo.caravan import Caravan
        from neuralhydrology.evaluation import metrics
        from neuralhydrology.evaluation.tester import RegressionTester
        from neuralhydrology.modelzoo.cudalstm import CudaLSTM
        from neuralhydrology.modelzoo.ealstm import EALSTM
        from neuralhydrology.utils.config import Config

        yield {
            "root": root,
            "Config": Config,
            "CudaLSTM": CudaLSTM,
            "EALSTM": EALSTM,
            "CamelsUS": CamelsUS,
            "Caravan": Caravan,
            "metrics": metrics,
            "RegressionTester": RegressionTester,
        }
    finally:
        sys.path.remove(str(root))
        for name in [m for m in sys.modules if m.split(".")[0] == "neuralhydrology"]:
            del sys.modules[name]


def _n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _config(nh, tmp_path: Path, **overrides):
    """A NeuralHydrology run configuration at the Kratzert et al. (2019) model settings."""
    raw = {
        "experiment_name": "pyhazards_oracle",
        "run_dir": str(tmp_path / "run"),
        "model": "cudalstm",
        "head": "regression",
        "output_activation": "linear",
        "hidden_size": 256,
        "initial_forget_bias": 5,
        "output_dropout": 0.4,
        "optimizer": "Adam",
        "loss": "MSE",
        "learning_rate": 1e-3,
        "batch_size": 256,
        "epochs": 30,
        "seq_length": 270,
        "predict_last_n": 1,
        "dataset": "camels_us",
        "data_dir": str(tmp_path),
        "forcings": ["maurer_extended"],
        "dynamic_inputs": list(KRATZERT2019_DYNAMIC_INPUTS),
        "static_attributes": list(KRATZERT2019_STATIC_ATTRIBUTES),
        "target_variables": ["QObs(mm/d)"],
        "train_basin_file": str(tmp_path / "basins.txt"),
        "validation_basin_file": str(tmp_path / "basins.txt"),
        "test_basin_file": str(tmp_path / "basins.txt"),
        "train_start_date": "01/10/1999",
        "train_end_date": "30/09/2008",
        "validation_start_date": "01/10/1980",
        "validation_end_date": "30/09/1989",
        "test_start_date": "01/10/1989",
        "test_end_date": "30/09/1999",
        "device": "cpu",
        "metrics": ["NSE"],
        "verbose": 0,
    }
    raw.update(overrides)
    cfg = nh["Config"](raw)
    cfg.train_dir = tmp_path / "run" / "train_data"  # where training datasets write their scaler
    cfg.train_dir.mkdir(parents=True, exist_ok=True)
    return cfg


def _assert_same_state(reference: torch.nn.Module, port: torch.nn.Module) -> None:
    ref_state, port_state = reference.state_dict(), port.state_dict()
    assert list(ref_state) == list(port_state)
    for key, value in ref_state.items():
        assert value.shape == port_state[key].shape, key
        assert torch.equal(value, port_state[key]), key


def _nh_inputs(x_d: torch.Tensor, x_s: torch.Tensor | None, names):
    data = {"x_d": {name: x_d[..., i: i + 1] for i, name in enumerate(names)}}
    if x_s is not None:
        data["x_s"] = x_s
    return data


def _compare(ref_model, port, x_d, x_s, names, keys):
    ref_model.eval()
    port.eval()
    port_inputs = {"x_d": x_d} if x_s is None else {"x_d": x_d, "x_s": x_s}
    with torch.no_grad():
        expected = ref_model(_nh_inputs(x_d, x_s, names))
        actual = port(port_inputs)
    for key in keys:
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-5, atol=1e-6)
    ref_model.train()
    port.train()
    torch.manual_seed(7)
    expected = ref_model(_nh_inputs(x_d, x_s, names))
    torch.manual_seed(7)
    actual = port(port_inputs)
    for key in keys:
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "model_name, nh_model, statics, expected_params",
    [
        ("neuralhydrology_lstm", "cudalstm", True, 297_217),
        ("neuralhydrology_lstm", "cudalstm", False, 269_569),
        ("neuralhydrology_ealstm", "ealstm", True, 208_641),
    ],
)
def test_model_matches_neuralhydrology(nh, tmp_path, model_name, nh_model, statics, expected_params):
    static = list(KRATZERT2019_STATIC_ATTRIBUTES) if statics else []
    cfg = _config(nh, tmp_path, model=nh_model, static_attributes=static)
    reference_cls = nh["CudaLSTM"] if nh_model == "cudalstm" else nh["EALSTM"]
    for seed in (0, 3):
        torch.manual_seed(seed)
        reference = reference_cls(cfg)
        torch.manual_seed(seed)
        port = build_model(model_name, task="regression", n_static=len(static))
        assert _n_params(reference) == _n_params(port) == expected_params
        _assert_same_state(reference, port)  # names, shapes and seeded initial values

    # Trained-like weights: perturb the reference, copy them into the port with strict=True.
    with torch.no_grad():
        for parameter in reference.parameters():
            parameter.add_(0.05 * torch.randn_like(parameter))
    port.load_state_dict(reference.state_dict(), strict=True)
    torch.manual_seed(1)
    x_d = torch.randn(4, 270, 5)
    x_s = torch.randn(4, len(static)) if static else None
    keys = ["y_hat", "h_n", "c_n"] + (["lstm_output"] if nh_model == "cudalstm" else [])
    _compare(reference, port, x_d, x_s, KRATZERT2019_DYNAMIC_INPUTS, keys)


def test_metrics_match_neuralhydrology(nh):
    import xarray as xr

    metrics = nh["metrics"]
    rng = np.random.default_rng(0)
    n = 1500
    dates = pd.date_range("1990-01-01", periods=n, freq="1D")
    for trial in range(24):
        base = np.abs(np.cumsum(rng.normal(size=n))) + rng.gamma(1.0, 2.0, size=n) * (rng.random(n) < 0.1) * 10
        obs = base.copy()
        sim = base * rng.uniform(0.7, 1.3) + rng.normal(scale=0.5, size=n)
        if trial % 3 == 0:  # missing observations and simulations (gaps in the peak windows)
            obs[rng.choice(n, 100, replace=False)] = np.nan
            sim[rng.choice(n, 20, replace=False)] = np.nan
        if trial % 4 == 1:  # zero flows and negative simulations (FDC metrics)
            obs[rng.choice(n, 50, replace=False)] = 0.0
            sim[rng.choice(n, 50, replace=False)] = -0.3
        observed = xr.DataArray(obs, coords={"date": dates}, dims=["date"])
        simulated = xr.DataArray(sim, coords={"date": dates}, dims=["date"])
        expected = metrics.calculate_all_metrics(observed, simulated)
        actual = hydrology.calculate_metrics(obs, sim, dates=dates.values)
        assert set(hydrology.NEURALHYDROLOGY_METRIC_NAMES) == set(expected)
        for name, value in expected.items():
            np.testing.assert_allclose(actual[hydrology.NEURALHYDROLOGY_METRIC_NAMES[name]], float(value),
                                       rtol=1e-12, atol=1e-12, equal_nan=True, err_msg=name)
    # Short series and a fraction that rounds to zero elements (FLV keeps the whole curve, as the reference).
    obs, sim = np.array([1.0, 2.0, 0.5]), np.array([0.8, 2.5, 0.4])
    observed = xr.DataArray(obs, coords={"date": dates[:3]}, dims=["date"])
    simulated = xr.DataArray(sim, coords={"date": dates[:3]}, dims=["date"])
    np.testing.assert_allclose(hydrology.fdc_flv(obs, sim, l=0.1), metrics.fdc_flv(observed, simulated, l=0.1))
    np.testing.assert_allclose(hydrology.kge(obs[:1], sim[:1]), metrics.kge(observed[:1], simulated[:1]), equal_nan=True)


def _camels_test_data(nh, tmp_path: Path) -> Path:
    data_dir = nh["root"] / "test" / "test_data" / "camels_us"
    (tmp_path / "basins.txt").write_text((nh["root"] / "test" / "test_data" / "4_basins_test_set.txt").read_text())
    return data_dir


def _nh_dataset_samples(dataset, dynamic_inputs):
    """(basin, last date) -> (x_d, x_s, y) of every sample of a NeuralHydrology dataset."""
    samples = {}
    for i in range(len(dataset)):
        sample = dataset[i]
        basin, _ = dataset.lookup_table[i]
        x_d = torch.cat([sample["x_d"][name] for name in dynamic_inputs], dim=-1)
        date = pd.Timestamp(sample["date"][-1])
        samples[(basin, date)] = (x_d, sample.get("x_s"), sample["y"])
    return samples


def _assert_windows_match(windows, samples, static: bool):
    keys = {(windows.basins[b], pd.Timestamp(d)) for b, d in zip(windows.sample_basin, windows.sample_date)}
    assert keys == set(samples)
    for i in range(len(windows)):
        inputs, y = windows[i]
        key = (windows.basins[windows.sample_basin[i]], pd.Timestamp(windows.sample_date[i]))
        x_d, x_s, y_ref = samples[key]
        torch.testing.assert_close(inputs["x_d"], x_d, rtol=1e-5, atol=1e-6, equal_nan=True)
        torch.testing.assert_close(y, y_ref, rtol=1e-5, atol=1e-6, equal_nan=True)
        if static:
            torch.testing.assert_close(inputs["x_s"], x_s, rtol=1e-5, atol=1e-6)


PERIODS = {"train": ("2000-01-01", "2001-12-31"), "test": ("2001-06-01", "2002-12-31")}
NH_PERIODS = {
    "train_start_date": "01/01/2000",
    "train_end_date": "31/12/2001",
    "validation_start_date": "01/06/2001",
    "validation_end_date": "31/12/2002",
    "test_start_date": "01/06/2001",
    "test_end_date": "31/12/2002",
}


def test_camels_us_reader_matches_neuralhydrology(nh, tmp_path):
    data_dir = _camels_test_data(nh, tmp_path)
    cfg = _config(nh, tmp_path, data_dir=str(data_dir), seq_length=90, **NH_PERIODS)
    reference_train = nh["CamelsUS"](cfg, is_train=True, period="train")
    bundle = CamelsUSStreamflowDataset(
        data_dir=data_dir, basins=tmp_path / "basins.txt", periods=PERIODS, seq_length=90
    ).load()

    scaler = bundle.metadata["scaler"]
    center, scale = reference_train.scaler["xarray_feature_center"], reference_train.scaler["xarray_feature_scale"]
    np.testing.assert_allclose(scaler["dynamic_center"], [float(center[v]) for v in KRATZERT2019_DYNAMIC_INPUTS], rtol=1e-5)
    np.testing.assert_allclose(scaler["dynamic_scale"], [float(scale[v]) for v in KRATZERT2019_DYNAMIC_INPUTS], rtol=1e-5)
    np.testing.assert_allclose(scaler["target_center"], [float(center["QObs(mm/d)"])], rtol=1e-5)
    np.testing.assert_allclose(scaler["target_scale"], [float(scale["QObs(mm/d)"])], rtol=1e-5)
    static_names = bundle.metadata["static_attributes"]
    assert static_names == sorted(KRATZERT2019_STATIC_ATTRIBUTES)
    np.testing.assert_allclose(scaler["static_mean"], reference_train.scaler["attribute_means"][static_names].values, rtol=1e-6)
    np.testing.assert_allclose(scaler["static_std"], reference_train.scaler["attribute_stds"][static_names].values, rtol=1e-6)

    train = bundle.get_split("train").inputs
    _assert_windows_match(train, _nh_dataset_samples(reference_train, KRATZERT2019_DYNAMIC_INPUTS), static=True)

    test = bundle.get_split("test").inputs
    test_samples = {}
    for basin in test.basins:
        reference_test = nh["CamelsUS"](cfg, is_train=False, period="test", basin=basin, scaler=reference_train.scaler)
        test_samples.update(_nh_dataset_samples(reference_test, KRATZERT2019_DYNAMIC_INPUTS))
    _assert_windows_match(test, test_samples, static=True)


def test_streamflow_evaluation_matches_neuralhydrology_tester(nh, tmp_path):
    """Whole pipeline: CAMELS-US reader + LSTM + rescaling + clipping + per-basin metrics vs. the NH Tester."""
    data_dir = _camels_test_data(nh, tmp_path)
    cfg = _config(
        nh, tmp_path, data_dir=str(data_dir), seq_length=90, hidden_size=32,
        clip_targets_to_zero=["QObs(mm/d)"], metrics=["all"], **NH_PERIODS,
    )
    cfg.run_dir.mkdir(parents=True, exist_ok=True)
    nh["CamelsUS"](cfg, is_train=True, period="train")  # writes the training scaler into the run directory
    torch.manual_seed(0)
    reference = nh["CudaLSTM"](cfg)
    with torch.no_grad():
        reference.head.net[0].bias.fill_(0.3)  # an untrained model would predict ~0 everywhere
    tester = nh["RegressionTester"](cfg=cfg, run_dir=cfg.run_dir, period="test", init_model=False)
    results = tester.evaluate(model=reference, metrics=["all"], save_results=False)

    bundle = CamelsUSStreamflowDataset(
        data_dir=data_dir, basins=tmp_path / "basins.txt", periods=PERIODS, seq_length=90
    ).load()
    port = build_model("neuralhydrology_lstm", task="regression", hidden_size=32)
    port.load_state_dict(reference.state_dict(), strict=True)
    port.eval()
    summary, per_basin = evaluate_streamflow(port, bundle, "test", clip_negative=True)

    assert set(per_basin) == set(results)
    # The Tester computes the metrics on float32 DataArrays, PyHazards in float64 (the definitions agree to
    # 1e-12 in test_metrics_match_neuralhydrology); float32 rounding of logs and means needs rtol 2e-4.
    for basin, values in per_basin.items():
        expected = results[basin]["1D"]
        for nh_name, key in hydrology.NEURALHYDROLOGY_METRIC_NAMES.items():
            np.testing.assert_allclose(values[key], float(expected[nh_name]), rtol=2e-4, atol=1e-5,
                                       equal_nan=True, err_msg=f"{basin} {nh_name}")
    nse = [float(results[b]["1D"]["NSE"]) for b in results]
    assert summary["nse"] == pytest.approx(float(np.median(nse)), rel=2e-4)
    assert summary["n_basins"] == len(results)


CARAVAN_BASINS = ["camelsaus_102101A", "hysets_01075000", "lamah_1145"]
CARAVAN_DYNAMIC = ["total_precipitation_sum", "temperature_2m_mean", "potential_evaporation_sum"]


def test_caravan_reader_matches_neuralhydrology(nh, tmp_path):
    data_dir = oracle_repo("flood-forecasting") / "test" / "test_data" / "multimet"
    (tmp_path / "basins.txt").write_text("\n".join(CARAVAN_BASINS) + "\n")
    cfg = _config(
        nh, tmp_path, dataset="caravan", data_dir=str(data_dir), forcings=None, seq_length=120,
        dynamic_inputs=CARAVAN_DYNAMIC, static_attributes=["area", "gauge_lat", "gauge_lon"],
        target_variables=["streamflow"],
        train_start_date="01/01/2000", train_end_date="31/12/2001",
        validation_start_date="01/01/2002", validation_end_date="30/06/2002",
        test_start_date="01/01/2002", test_end_date="30/06/2002",
    )
    reference_train = nh["Caravan"](cfg, is_train=True, period="train")
    bundle = CaravanStreamflowDataset(
        data_dir=data_dir, basins=CARAVAN_BASINS, dynamic_inputs=CARAVAN_DYNAMIC,
        static_attributes=["area", "gauge_lat", "gauge_lon"], seq_length=120,
        periods={"train": ("2000-01-01", "2001-12-31"), "test": ("2002-01-01", "2002-06-30")},
    ).load()
    _assert_windows_match(bundle.get_split("train").inputs, _nh_dataset_samples(reference_train, CARAVAN_DYNAMIC), static=True)
    test = bundle.get_split("test").inputs
    samples = {}
    for basin in CARAVAN_BASINS:
        reference_test = nh["Caravan"](cfg, is_train=False, period="test", basin=basin, scaler=reference_train.scaler)
        samples.update(_nh_dataset_samples(reference_test, CARAVAN_DYNAMIC))
    _assert_windows_match(test, samples, static=True)

    csv_root = tmp_path / "caravan_csv"
    for basin in CARAVAN_BASINS:  # the same basins in the CSV layout
        source = basin.split("_")[0]
        frame = __import__("xarray").open_dataset(data_dir / "timeseries" / "netcdf" / source / f"{basin}.nc").to_dataframe()
        (csv_root / "timeseries" / "csv" / source).mkdir(parents=True, exist_ok=True)
        frame.to_csv(csv_root / "timeseries" / "csv" / source / f"{basin}.csv", index_label="date")
        (csv_root / "attributes").mkdir(exist_ok=True)
    import shutil

    shutil.copytree(data_dir / "attributes", csv_root / "attributes", dirs_exist_ok=True)
    from_csv = CaravanStreamflowDataset(
        data_dir=csv_root, basins=CARAVAN_BASINS, dynamic_inputs=CARAVAN_DYNAMIC, filetype="csv",
        static_attributes=["area", "gauge_lat", "gauge_lon"], seq_length=120,
        periods={"train": ("2000-01-01", "2001-12-31"), "test": ("2002-01-01", "2002-06-30")},
    ).load()
    first_nc, first_csv = bundle.get_split("test").inputs[5], from_csv.get_split("test").inputs[5]
    torch.testing.assert_close(first_csv[0]["x_d"], first_nc[0]["x_d"], rtol=1e-5, atol=1e-5)


# --------------------------------------------------------------------------------------------------
# Official Kratzert et al. (2019) runs (HydroShare, 939 MB): gated on the large asset.

RUNS = "Kratzert_et_al_2019_HESSD_LSTM_4_REGIONAL_MODELING/runs"
EALSTM_NSE_RUNS = ["run_1906_1004_seed111", "run_1906_1005_seed222", "run_1906_1005_seed333", "run_1906_1006_seed444",
                   "run_1906_1006_seed555", "run_1906_1006_seed666", "run_1906_1007_seed777", "run_1906_1007_seed888"]
LSTM_NSE_RUNS = ["run_1906_1009_seed111", "run_1906_1009_seed222", "run_1906_1009_seed333", "run_1906_1010_seed444",
                 "run_1906_1010_seed555", "run_1906_1010_seed666", "run_1906_1011_seed777", "run_1906_1011_seed888"]
NOSTATIC_NSE_RUN = "run_1606_0921_seed111"


@pytest.fixture(scope="module")
def runs_zip():
    path = oracle_large_asset("kratzert2019_hydroshare_runs") / "Kratzert_et_al_2019_HESSD_LSTM_4_REGIONAL_MODELING.zip"
    with zipfile.ZipFile(path) as archive:
        yield archive


@pytest.fixture(scope="module")
def paper_code():
    root = oracle_repo("ealstm_regional_modeling")
    lstm = import_from(root, "papercode.lstm")
    ealstm = import_from(root, "papercode.ealstm")
    source = (root / "main.py").read_text(encoding="utf-8")
    start = source.index("class Model(nn.Module)")
    end = source.index("###########################\n# Train or evaluate model")
    namespace = {"torch": torch, "nn": torch.nn, "Tuple": tuple, "LSTM": lstm.LSTM, "EALSTM": ealstm.EALSTM}
    exec(compile(source[start:end], str(root / "main.py"), "exec"), namespace)  # the Model wrapper, unchanged
    return namespace["Model"]


@pytest.mark.parametrize(
    "run, model_name, n_static, concat_static, no_static",
    [
        (EALSTM_NSE_RUNS[0], "neuralhydrology_ealstm", 27, False, False),
        # The 2019 "LSTM with static inputs" concatenates the 27 attributes to the 5 forcings (32 inputs).
        (LSTM_NSE_RUNS[0], "neuralhydrology_lstm", 27, True, False),
        (NOSTATIC_NSE_RUN, "neuralhydrology_lstm", 0, False, True),
    ],
)
def test_official_2019_checkpoints(runs_zip, paper_code, run, model_name, n_static, concat_static, no_static):
    state = torch.load(io.BytesIO(runs_zip.read(f"{RUNS}/{run}/model_epoch30.pt")), map_location="cpu", weights_only=True)
    reference = paper_code(input_size_dyn=32 if concat_static else 5, input_size_stat=0 if no_static else 27,
                           hidden_size=256, dropout=0.4, concat_static=concat_static, no_static=no_static)
    reference.load_state_dict(state, strict=True)
    reference.eval()
    port = build_model(model_name, task="regression", n_dynamic=5, n_static=n_static)
    port.load_state_dict(convert_kratzert2019_state_dict(state), strict=True)
    assert _n_params(port) == sum(v.numel() for v in state.values()) + (1024 if model_name == "neuralhydrology_lstm" else 0)
    port.eval()
    torch.manual_seed(0)
    x_d = torch.randn(8, 270, 5)
    x_s = torch.randn(8, 27)
    with torch.no_grad():
        if concat_static:
            expected = reference(torch.cat([x_d, x_s.unsqueeze(1).repeat(1, 270, 1)], dim=-1))[0]
            actual = port({"x_d": x_d, "x_s": x_s})["y_hat"][:, -1]
        elif no_static:
            expected = reference(x_d)[0]
            actual = port({"x_d": x_d})["y_hat"][:, -1]
        else:
            expected = reference(x_d, x_s)[0]
            actual = port({"x_d": x_d, "x_s": x_s})["y_hat"][:, -1]
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_official_2019_attribute_order(runs_zip, tmp_path):
    """KRATZERT2019_CHECKPOINT_STATIC_ORDER is the column order of the runs' attribute databases."""
    import sqlite3

    from pyhazards.datasets.flood import KRATZERT2019_CHECKPOINT_STATIC_ORDER

    database = tmp_path / "attributes.db"
    database.write_bytes(runs_zip.read(f"{RUNS}/{EALSTM_NSE_RUNS[0]}/attributes.db"))
    with sqlite3.connect(database) as connection:
        columns = [row[1] for row in connection.execute("PRAGMA table_info('basin_attributes')")]
    root = oracle_repo("ealstm_regional_modeling")
    invalid = import_from(root, "papercode.datautils").INVALID_ATTR
    kept = [c for c in columns if c not in invalid and c not in ("gauge_id", "gauge_lat", "gauge_lon")]
    assert kept == KRATZERT2019_CHECKPOINT_STATIC_ORDER


def _stored_simulations(runs_zip, run):
    names = [n for n in runs_zip.namelist() if n.startswith(f"{RUNS}/{run}/") and n.endswith(".p")]
    assert len(names) == 1
    return pickle.loads(runs_zip.read(names[0]))


def _nse_summary(observed, simulated):
    per_basin = {b: hydrology.calculate_metrics(observed[b], simulated[b], metrics=["nse"]) for b in observed}
    return hydrology.aggregate_basin_metrics(per_basin)


# Kratzert et al. (2019), Table 2: single model mean NSE, median NSE, basins with NSE <= 0 (averages over
# the 8 seeds), then the same for the ensemble mean of the 8 runs.
TABLE2 = {
    ("lstm_no_static", "MSE"): (0.24, 0.60, 44, 0.36, 0.65, 31),
    ("lstm_no_static", "NSE*"): (0.39, 0.59, 28, 0.49, 0.64, 20),
    ("lstm_static", "MSE"): (0.66, 0.73, 6, 0.71, 0.76, 3),
    ("lstm_static", "NSE*"): (0.69, 0.73, 2, 0.72, 0.76, 2),
    ("ealstm", "MSE"): (0.63, 0.71, 9, 0.68, 0.74, 6),
    ("ealstm", "NSE*"): (0.67, 0.71, 3, 0.70, 0.74, 2),
}


@pytest.fixture(scope="module")
def run_groups(runs_zip):
    import json

    groups = {}
    for name in runs_zip.namelist():
        if name.endswith("/cfg.json"):
            cfg = json.loads(runs_zip.read(name))
            kind = "lstm_no_static" if cfg["no_static"] else ("lstm_static" if cfg["concat_static"] else "ealstm")
            groups.setdefault((kind, "MSE" if cfg["use_mse"] else "NSE*"), []).append(name.split("/")[2])
    return {key: sorted(runs) for key, runs in groups.items()}


@pytest.mark.parametrize("group", sorted(TABLE2))
def test_official_2019_simulations_reproduce_table2(runs_zip, run_groups, nh, group):
    """PyHazards metrics and aggregation on the stored 1989-1999 test simulations give Table 2 of the paper."""
    import xarray as xr

    runs = run_groups[group]
    assert len(runs) == 8
    means, medians, below_zero = [], [], []
    observed, ensemble = None, None
    for run in runs:
        simulations = _stored_simulations(runs_zip, run)
        assert len(simulations) == 531
        if observed is None:
            observed = {b: df["qobs"].to_numpy(dtype=np.float64) for b, df in simulations.items()}
            ensemble = {b: np.zeros(len(df)) for b, df in simulations.items()}
        simulated = {b: df["qsim"].to_numpy(dtype=np.float64) for b, df in simulations.items()}
        summary = _nse_summary(observed, simulated)
        means.append(summary["nse_mean"])
        medians.append(summary["nse"])
        below_zero.append(summary["n_basins_nse_le_0"])
        for b in ensemble:
            ensemble[b] += simulated[b] / len(runs)
        basin, df = next(iter(simulations.items()))  # same value as NeuralHydrology's NSE
        obs = xr.DataArray(observed[basin], coords={"date": df.index}, dims=["date"])
        sim = xr.DataArray(simulated[basin], coords={"date": df.index}, dims=["date"])
        assert hydrology.nse(observed[basin], simulated[basin]) == pytest.approx(nh["metrics"].nse(obs, sim), rel=1e-12)
    ensemble_summary = _nse_summary(observed, ensemble)
    mean, median, n_le_0, ens_mean, ens_median, ens_le_0 = TABLE2[group]
    tolerance = 0.005 + 1e-9  # the table rounds to two decimals
    assert abs(np.mean(means) - mean) <= tolerance
    assert abs(np.mean(medians) - median) <= tolerance
    assert abs(np.mean(below_zero) - n_le_0) <= 0.5
    assert abs(ensemble_summary["nse_mean"] - ens_mean) <= tolerance
    assert abs(ensemble_summary["nse"] - ens_median) <= tolerance
    assert ensemble_summary["n_basins_nse_le_0"] == ens_le_0
