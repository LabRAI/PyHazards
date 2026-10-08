"""Hurricast checked against the official code (leobix/hurricast, pinned).

The repository has no licence: pyhazards/models/hurricast.py is written from the paper, and the official
modules are imported from the pinned checkout here only as a test oracle:

* ``src/models/experimental_models.py`` (``ExperimentalHurricast``, ``CNNEncoder``, ``ExpTRANSFORMER``,
  ``ExpLSTM``) with every encoder / decoder preset of ``scripts/config.py``: parameter counts, state-dict
  keys, seeded initial weights, outputs and embeddings in eval and train mode, gradients;
* ``src/utils/run.py`` ``compute_l2`` (the L2 term of the training loss) and a training step of
  ``src/run.py`` ``train_epoch``;
* the XGBoost stage: the feature-column rule and feature names of notebooks/Compute_results_*_Round2.ipynb,
  the default hyperparameters of ``train_xgb_track`` / ``train_xgb_intensity`` in scripts/run_embeddings.py,
  and predictions of ``xgboost.XGBRegressor`` fitted the official way;
* ``src/utils/data_processing.py`` (storm selection, wind category, displacements) on real IBTrACS rows.
"""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from oracle_utils import import_from, load_definitions, oracle_package, oracle_repo
from pyhazards.datasets.tc import read_ibtracs
from pyhazards.datasets.tc.hurricast import hurricast_storms
from pyhazards.models import build_model
from pyhazards.models.hurricast import (
    DECODER_CONFIGS,
    ENCODER_CONFIGS,
    HURRICAST_STAT_FEATURES,
    HURRICAST_XGBOOST_PARAMS,
    hurricast_l2_penalty,
    hurricast_network,
    hurricast_xgboost_columns,
)

IBTRACS_CSV = Path(__file__).resolve().parents[1] / "fixtures" / "ibtracs" / "ibtracs.fixture.list.v04r01.csv"
PRESETS = [
    (decoder, encoder)
    for decoder in DECODER_CONFIGS
    for encoder in (["full_encoder_config", "split_encoder_config"] if decoder != "transformer_config_noviz" else [None])
]


def _official():
    repo = oracle_repo("hurricast")
    models = import_from(repo, "src.models.experimental_models")
    spec = importlib.util.spec_from_file_location("hurricast_official_config", repo / "scripts" / "config.py")
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    return models, config, repo


def _official_network(models, config, decoder, encoder, n_pred):
    return models.ExperimentalHurricast(
        n_pred=n_pred,
        decoder_config=config.create_config(decoder),
        encoder_config=config.create_config(encoder) if encoder else None,
        decoder_name="ExpTRANSFORMER" if decoder.startswith("transformer") else "ExpLSTM",
        encoder_name="CNNEncoder" if encoder else None,
        split_cnns=encoder == "split_encoder_config",
    )


def _inputs(encoder, batch=4, seed=1):
    g = torch.Generator().manual_seed(seed)
    stat = torch.randn(batch, 8, 14 if encoder else 10, generator=g)
    maps = torch.randn(batch, 8, 9, 25, 25, generator=g) if encoder else None
    return stat, maps


def test_presets_are_the_official_configurations():
    _, config, _ = _official()
    for name, (_, kwargs) in DECODER_CONFIGS.items():
        assert config.create_config(name) == kwargs, name
    for name, kwargs in ENCODER_CONFIGS.items():
        official = config.create_config(name)
        assert {**official, "hidden_configuration": tuple(official["hidden_configuration"])} == kwargs, name


@pytest.mark.parametrize("decoder,encoder", PRESETS)
@pytest.mark.parametrize("target,n_pred", [("intensity", 1), ("displacement", 2)])
def test_network_matches_official(decoder, encoder, target, n_pred):
    models, config, _ = _official()
    torch.manual_seed(0)
    reference = _official_network(models, config, decoder, encoder, n_pred)
    torch.manual_seed(0)
    port = hurricast_network(target, decoder, encoder)
    assert sum(p.numel() for p in port.parameters()) == sum(p.numel() for p in reference.parameters())
    ref_state, port_state = reference.state_dict(), port.state_dict()
    assert list(ref_state) == list(port_state)
    for key, value in ref_state.items():
        assert torch.equal(value, port_state[key]), key

    stat, maps = _inputs(encoder)
    for mode in ("eval", "train"):
        getattr(reference, mode)()
        getattr(port, mode)()
        with torch.no_grad():
            torch.manual_seed(3)
            expected = reference(stat, maps)
            torch.manual_seed(3)
            actual = port(stat, maps)
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
            torch.manual_seed(3)
            expected = reference.get_embeddings(stat, maps, xgb=True)
            torch.manual_seed(3)
            actual = port.get_embeddings(stat, maps, xgb=True)
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
            torch.manual_seed(3)
            expected = reference.get_embeddings(stat, maps, xgb=False)
            torch.manual_seed(3)
            torch.testing.assert_close(port.get_embeddings(stat, maps), expected, rtol=1e-5, atol=1e-6)
    # BatchNorm running statistics were updated identically in train mode.
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(port.state_dict()[key], value, rtol=1e-6, atol=1e-7, msg=key)


def test_paper_configuration_counts_and_gradients():
    models, config, _ = _official()
    for target, n_pred, count in (("displacement", 2, 2_969_914), ("intensity", 1, 2_969_771)):
        torch.manual_seed(0)
        reference = _official_network(models, config, "transformer_config", "full_encoder_config", n_pred)
        port = build_model("hurricast", task="regression", target=target, predictor="network")
        port.network.load_state_dict(reference.state_dict(), strict=True)  # official keys under ``network.``
        assert sum(p.numel() for p in port.parameters()) == count
        stat, maps = _inputs("full_encoder_config", batch=6)
        reference.train()
        port.train()
        x_stat = torch.cat([stat, torch.randn(6, 8, 16)], dim=-1)  # Hurricast takes all 30 features
        expected = reference(stat, maps)
        actual = port({"x_stat": x_stat, "x_viz": maps})  # target scaling is 0 / 1 before fit()
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        expected.square().sum().backward()
        actual.square().sum().backward()
        for (name, ref_param), port_param in zip(reference.named_parameters(), port.network.parameters()):
            torch.testing.assert_close(port_param.grad, ref_param.grad, rtol=1e-4, atol=1e-6, msg=name)


def test_l2_penalty_is_the_official_compute_l2():
    models, config, repo = _official()
    compute_l2 = load_definitions(repo / "src" / "utils" / "run.py", ["compute_l2"])["compute_l2"]
    torch.manual_seed(0)
    reference = _official_network(models, config, "transformer_config", "full_encoder_config", 1)
    port = hurricast_network("intensity")
    port.load_state_dict(reference.state_dict())
    torch.testing.assert_close(hurricast_l2_penalty(port), compute_l2(reference))


def test_training_step_matches_official_train_epoch():
    """One epoch (one batch) of Hurricast.fit_network equals src/run.py train_epoch: MSE + 2 / B * l2 * sum(W^2), Adam.

    Displacement targets, for which the official loss compares tensors of the same shape (see the card for the
    intensity case)."""
    models, config, repo = _official()
    run = import_from(repo, "src.run")
    torch.manual_seed(0)
    reference = _official_network(models, config, "transformer_config", "full_encoder_config", 2)
    model = build_model("hurricast", task="regression", target="displacement", predictor="network")
    model.network.load_state_dict(reference.state_dict())
    stat, maps = _inputs("full_encoder_config", batch=8, seed=5)
    targets = torch.randn(8, 2, generator=torch.Generator().manual_seed(6))
    optimizer = torch.optim.Adam(reference.parameters(), lr=4e-4)
    # The same sample order as fit_network(seed=0): the conv biases in front of BatchNorm get gradients that
    # are zero up to rounding, whose sign (which Adam turns into a full step) depends on the summation order.
    order = torch.randperm(8, generator=torch.Generator().manual_seed(0))
    batch = ({"x_stat": stat[order], "x_viz": maps[order]}, {"trg_y": targets[order]})
    run.train_epoch(reference, [batch], optimizer, torch.nn.MSELoss(), 0.01, 0)
    x_stat = torch.cat([stat, torch.zeros(8, 8, 16)], dim=-1)
    model.fit_network({"x_stat": x_stat, "x_viz": maps}, targets, epochs=1, batch_size=8, learning_rate=4e-4, l2_reg=0.01)
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(model.network.state_dict()[key], value, rtol=1e-5, atol=1e-6, msg=key)


def _round2_notebook_source(repo: Path, name: str) -> str:
    notebook = json.loads((repo / "notebooks" / name).read_text(encoding="utf-8"))
    return "".join(notebook["cells"][0]["source"] if notebook["cells"][0]["cell_type"] == "code" else "")


def test_xgboost_columns_follow_the_official_notebooks():
    import pandas as pd

    _, _, repo = _official()
    for name in ("Compute_results_intensity_24h_Round2.ipynb", "Compute_results_track24_Round2.ipynb"):
        notebook = json.loads((repo / "notebooks" / name).read_text(encoding="utf-8"))
        source = next("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code" and "names = [" in "".join(cell["source"]))
        tree = ast.parse(source)
        names = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "names")
        assert tuple(names) == HURRICAST_STAT_FEATURES, name
        # The notebook's column rule, executed on a frame with the notebook's column names.
        window_size = 8
        names_all = names * window_size
        for i in range(len(names_all)):
            names_all[i] += "_" + str(i // 30)
        frame = pd.DataFrame(np.zeros((1, len(names_all))), columns=names_all)
        rule = next(ast.get_source_segment(source, node.value) for node in tree.body if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "cols")
        cols = eval(rule, {"X_train": frame})
        assert cols == hurricast_xgboost_columns(window_size), name
        assert len(cols) == 14 * 8 + 16


def test_xgboost_defaults_are_the_official_ones():
    _, _, repo = _official()
    tree = ast.parse((repo / "scripts" / "run_embeddings.py").read_text(encoding="utf-8"))
    for function in ("train_xgb_track", "train_xgb_intensity"):
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function)
        defaults = dict(zip([a.arg for a in node.args.args][-len(node.args.defaults) :], [ast.literal_eval(d) for d in node.args.defaults]))
        assert {key: defaults[key] for key in HURRICAST_XGBOOST_PARAMS} == HURRICAST_XGBOOST_PARAMS, function


@pytest.mark.parametrize("target", ["intensity", "displacement"])
def test_xgboost_stage_matches_official_regressors(target):
    xgboost = oracle_package("xgboost", "3.2.0", "requirements-tc.txt", distribution="xgboost-cpu")
    import pandas as pd

    models, config, _ = _official()
    n_pred = 1 if target == "intensity" else 2
    torch.manual_seed(0)
    reference = _official_network(models, config, "transformer_config", "full_encoder_config", n_pred).eval()
    model = build_model("hurricast", task="regression", target=target, n_jobs=1)
    model.network.load_state_dict(reference.state_dict())
    g = torch.Generator().manual_seed(4)
    n = 160
    x_stat = torch.randn(n, 8, 30, generator=g)
    x_viz = torch.randn(n, 8, 9, 25, 25, generator=g)
    position = torch.randn(n, 2, generator=g) * 10 + 20
    raw = torch.randn(n, n_pred, generator=g) * 3 + 1
    inputs = {"x_stat": x_stat, "x_viz": x_viz, "position": position}
    targets = raw[:, 0] if target == "intensity" else (position + raw).unsqueeze(1)
    model.fit(inputs, targets, train_network=False)

    # The official way: embeddings of the frozen network appended to the selected statistical columns,
    # standardised targets, one XGBRegressor per target with the run_embeddings.py defaults.
    with torch.no_grad():
        embeddings = reference.get_embeddings(x_stat[:, :, :14], x_viz, xgb=True)[0].numpy()
    names_all = [f"{name}_{i // 30}" for i, name in enumerate(list(HURRICAST_STAT_FEATURES) * 8)]
    frame = pd.DataFrame(x_stat.reshape(n, -1).numpy(), columns=names_all)
    cols = [c for c in frame.columns if c.lower()[-2:] == "_0" or c.lower()[:3] != "cat"]
    features = np.concatenate((frame[cols], embeddings), axis=1)
    mean, std = raw.mean(0), raw.std(0)
    expected = []
    for k in range(n_pred):
        regressor = xgboost.XGBRegressor(**HURRICAST_XGBOOST_PARAMS, n_jobs=1)
        regressor.fit(features, ((raw[:, k] - mean[k]) / std[k]).numpy())
        expected.append(np.array(regressor.predict(features)) * float(std[k]) + float(mean[k]))
    expected = np.stack(expected, axis=1)
    with torch.no_grad():
        actual = model(inputs).numpy()
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
    if target == "displacement":
        forecast = model.forecast(inputs)
        np.testing.assert_allclose(forecast["lat"].numpy()[:, 0], position[:, 0].numpy() + expected[:, 0], rtol=1e-5)


def test_storm_statistics_follow_the_official_data_processing():
    """Storm cut, interpolated WMO wind / pressure, wind category and displacements on real IBTrACS rows."""
    import pandas as pd

    _, _, repo = _official()
    processing = import_from(repo, "src.utils.data_processing")
    table = read_ibtracs(IBTRACS_CSV)
    on_grid = table[(table["ISO_TIME"].dt.hour % 3 == 0) & (table["ISO_TIME"].dt.minute == 0)]
    for sid in ("2023265N29284", "2026058S18168"):  # OPHELIA (NA), URMIL (SP, crosses the dateline)
        rows = on_grid[on_grid["SID"] == sid].reset_index(drop=True)
        min_wind, min_steps, max_steps = 34, 5, 120
        # prepare_tabular_data_vision without its tensor stacking, on this storm's 3-hourly rows.
        frame = processing.numeric_data(rows[["SID", "ISO_TIME", "LAT", "LON", "WMO_WIND", "WMO_PRES", "DIST2LAND", "STORM_SPEED", "STORM_DIR"]].copy())
        frame = processing.add_storm_category_val(frame)
        storms = processing.sort_storm(frame, min_wind, min_steps)
        padded = processing.add_displacement_lat_lon2(processing.pad_traj(storms, max_steps))
        official = padded[1]
        ours = hurricast_storms(rows, min_wind, min_steps, max_steps)
        assert len(ours) == 1
        features = pd.DataFrame(ours[0]["features"], columns=HURRICAST_STAT_FEATURES)
        official = official.iloc[: len(features)]
        assert (pd.to_datetime(official["ISO_TIME"]).to_numpy("datetime64[ns]") == ours[0]["times"]).all()
        pairs = {
            "LAT": "LAT", "LON": "LON", "WMO_WIND": "WMO_WIND", "WMO_PRES": "WMO_PRES", "DIST2LAND": "DIST2LAND",
            "STORM_SPEED": "STORM_SPEED", "cat_storm_category": "storm_category",
            "STORM_DISPLACEMENT_X": "DISPLACEMENT_LAT", "STORM_DISPLACEMENT_Y": "DISPLACEMENT_LON",
        }
        for ours_name, official_name in pairs.items():
            np.testing.assert_allclose(features[ours_name].to_numpy(), official[official_name].to_numpy(dtype=np.float64), rtol=1e-12, atol=1e-12, err_msg=f"{sid} {ours_name}")
        direction = np.deg2rad(official["STORM_DIR"].to_numpy(dtype=np.float64))
        np.testing.assert_allclose(features["COS_STORM_DIR"].to_numpy(), np.cos(direction), atol=1e-12)
