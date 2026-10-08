"""UrbanFloodCast DNO checked against the official code (HydroPML/UrbanFloodCast, pinned in repos.yaml).

The official repository has no licence: PyHazards' DNO is written from the paper and from U-NO (BSD-2) /
FNO (MIT) building blocks, and the official ``DNO/models/DNO.py`` (DNO-3, the default of ``DNO_main.py``),
the one-shot data pipeline ``flood_data`` and the metrics of ``DNO/utils25.py`` are imported here from the
pinned checkout only as test oracles. ``DNO_main.py`` cannot be imported as released (``models/FNO.py``
does ``from utils import grid`` and the repository has no ``utils.py``), so its ``get_eval_pred`` is
executed from the file's own source and its test-loop metric lines are reproduced below.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from oracle_utils import import_from, load_definitions, oracle_repo
from pyhazards.benchmarks.flood import evaluate_inundation
from pyhazards.datasets import load_dataset
from pyhazards.datasets.base import DataBundle, DataSplit, FeatureSpec, LabelSpec
from pyhazards.datasets.flood.urbanfloodcast import prepare_urbanfloodcast_event, synthetic_urbanfloodcast_event
from pyhazards.metrics.inundation import inundation_metrics
from pyhazards.models import build_model
from pyhazards.models.urbanfloodcast import UrbanFloodCast


def _dno_dir() -> Path:
    return oracle_repo("UrbanFloodCast") / "DNO"


def _official_dno():
    return import_from(_dno_dir(), "models.DNO").DNO


def _utils25():
    return import_from(_dno_dir(), "utils25")


def _assert_same_state(reference, port) -> None:
    ref_state, port_state = reference.state_dict(), port.state_dict()
    assert list(ref_state) == list(port_state)
    for key, value in ref_state.items():
        assert value.shape == port_state[key].shape and value.dtype == port_state[key].dtype, key
        assert torch.equal(value, port_state[key]), key


@pytest.mark.parametrize("seed", [0, 1])
def test_parameters_names_and_seeded_initialisation_match_official(seed):
    DNO = _official_dno()
    # DNO_main.py builds DNO(num_channels=5, width=10, initial_step=1, pad=args.time_pad (False), factor=1)
    # after torch.manual_seed(args.seed) (default 1); models/DNO.py seeds 0 at import time.
    torch.manual_seed(seed)
    reference = DNO(num_channels=5, width=10, initial_step=1, pad=False, factor=1)
    torch.manual_seed(seed)
    port = build_model("urbanfloodcast", task="regression")
    assert sum(p.numel() for p in reference.parameters()) == sum(p.numel() for p in port.parameters()) == 4_470_437
    assert sum(p.numel() * (1 + p.is_complex()) for p in port.parameters()) == 8_937_637
    _assert_same_state(reference, port)


@pytest.mark.parametrize("shape", [(2, 32, 32, 24, 1, 5), (1, 28, 36, 16, 1, 5)], ids=["32x32x24", "28x36x16"])
def test_outputs_and_gradients_match_official(shape):
    DNO = _official_dno()
    torch.manual_seed(1)
    reference = DNO(num_channels=5, width=10, initial_step=1, pad=False, factor=1)
    port = UrbanFloodCast()
    port.load_state_dict(reference.state_dict(), strict=True)
    x = torch.randn(*shape, generator=torch.Generator().manual_seed(2))
    for train in (False, True):  # InstanceNorm3d without running statistics: same in both modes
        reference.train(train)
        port.train(train)
        with torch.no_grad():
            expected = reference(x)
            torch.testing.assert_close(port(x), expected, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(port(x.reshape(*shape[:4], -1)), expected, rtol=1e-5, atol=1e-6)
    reference.zero_grad()
    port.zero_grad()
    reference(x).square().sum().backward()
    port(x).square().sum().backward()
    for (name, ref_param), port_param in zip(reference.named_parameters(), port.parameters()):
        torch.testing.assert_close(port_param.grad, ref_param.grad, rtol=1e-4, atol=1e-6, msg=name)


def _events(directory: Path, count: int = 3, seed: int = 0, missing_terrain: bool = False):
    generator = torch.Generator().manual_seed(seed)
    events = []
    for i in range(count):
        event = synthetic_urbanfloodcast_event(30, 34, 25, generator)
        if missing_terrain and i == 0:
            # A missing terrain cell: the reference's nan_to_num(nan=z.max() + 30) cannot fill it because
            # torch.max propagates NaN, so the whole terrain channel becomes NaN (reproduced).
            event[3, 5, 7:, 4] = float("nan")
        torch.save(event, directory / f"{i}.pt")
        events.append(event)
    return events


def test_one_shot_samples_match_flood_data(tmp_path):
    flood_data = _utils25().flood_data
    events = _events(tmp_path, missing_terrain=True)
    for train in (True, False):
        official = flood_data(path_root=str(tmp_path), T_in=1, T_out=24, train=train, strategy="oneshot")
        assert len(official) == len(events)
        for idx in range(len(official)):
            x, y, mask = official[idx]
            event = events[int(Path(official.data[idx]).stem)]
            before = event.clone()
            ours = prepare_urbanfloodcast_event(event, T_in=1, T_out=24)
            assert torch.equal(event.nan_to_num(-1.0), before.nan_to_num(-1.0))  # the event is not modified
            for actual, expected in zip(ours, (x, y, mask)):
                assert actual.shape == expected.shape and actual.dtype == expected.dtype
                assert torch.equal(actual.nan_to_num(-7.0), expected.nan_to_num(-7.0))


def test_inundation_metrics_match_dno_main_test_loop(tmp_path):
    """Relative L2, NSE, Pearson r and CSI as DNO_main.py computes them in its test loop (batch size 1)."""
    utils25 = _utils25()
    get_eval_pred = load_definitions(_dno_dir() / "DNO_main.py", ["get_eval_pred"], {"torch": torch})["get_eval_pred"]
    DNO = _official_dno()
    torch.manual_seed(1)
    model = DNO(num_channels=5, width=10, initial_step=1, pad=False, factor=1).eval()
    _events(tmp_path)
    test_data = utils25.flood_data(path_root=str(tmp_path), T_in=1, T_out=24, train=False, strategy="oneshot")
    loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    lploss = utils25.LpLoss(size_average=False)
    Sy, Sx, T, num_channels_y = 30, 34, 24, 3  # noqa: N806 (DNO_main names)
    totals = dict.fromkeys(["relative_l2", "nse", "pearson_r", "csi_1cm", "csi_10cm", "csi_50cm"], 0.0)
    xs, ys, masks = [], [], []
    with torch.no_grad():
        for xx, yy, mask in loader:
            xs.append(xx[0])
            ys.append(yy[0].clone())
            masks.append(mask[0])
            yy = yy * mask
            pred = get_eval_pred(model=model, x=xx, strategy="oneshot", T=T, times=[]).view(len(xx), Sy, Sx, T, num_channels_y)
            pred = pred * mask
            totals["relative_l2"] += lploss(pred.reshape(len(pred), -1, num_channels_y), yy.reshape(len(yy), -1, num_channels_y)).item()
            totals["nse"] += utils25.nse(pred.reshape(len(pred), -1, num_channels_y), yy.reshape(len(yy), -1, num_channels_y)).item()
            totals["pearson_r"] += utils25.corr(pred.reshape(len(pred), -1, num_channels_y), yy.reshape(len(yy), -1, num_channels_y)).item()
            for name, threshold in (("csi_1cm", 0.01), ("csi_10cm", 0.1), ("csi_50cm", 0.5)):
                totals[name] += utils25.critical_success_index(
                    pred[..., 0:1].reshape(len(pred), -1, 1), yy[..., 0:1].reshape(len(yy), -1, 1), threshold
                ).item()
    expected = {name: value / len(test_data) for name, value in totals.items()}

    port = UrbanFloodCast()
    port.load_state_dict(model.state_dict(), strict=True)
    port.eval()
    x, y, m = torch.stack(xs), torch.stack(ys), torch.stack(masks)
    bundle = DataBundle(
        splits={"test": DataSplit(x, y, metadata={"mask": m})},
        feature_spec=FeatureSpec(channels=5),
        label_spec=LabelSpec(num_targets=3),
        metadata={"depth_channel": 0, "inundation_layout": "channels_last"},
    )
    metrics, _ = evaluate_inundation(port, bundle, "test", batch_size=1)
    for name, value in expected.items():
        assert metrics[name] == pytest.approx(value, rel=1e-5, abs=1e-7), name
    with torch.no_grad():
        direct = inundation_metrics(port(x), y, mask=m, depth_index=0)
    for name, value in expected.items():
        assert direct[name] == pytest.approx(value, rel=1e-5, abs=1e-7), name


def test_metric_functions_match_utils25():
    utils25 = _utils25()
    from pyhazards.metrics import inundation

    generator = torch.Generator().manual_seed(0)
    pred, target = torch.rand(2, 500, 3, generator=generator), torch.rand(2, 500, 3, generator=generator)
    torch.testing.assert_close(inundation.relative_l2(pred, target).sum(), utils25.LpLoss(size_average=False)(pred, target))
    torch.testing.assert_close(inundation.nash_sutcliffe(pred, target), utils25.nse(pred, target))
    torch.testing.assert_close(inundation.pearson_correlation(pred, target), utils25.corr(pred, target))
    for threshold in (0.01, 0.1, 0.5):
        torch.testing.assert_close(
            inundation.critical_success_index(pred[..., :1], target[..., :1], threshold),
            utils25.critical_success_index(pred[..., :1], target[..., :1], threshold),
        )


def test_synthetic_dataset_uses_the_official_layout():
    bundle = load_dataset("urbanfloodcast_synthetic", micro=True).load()
    x = bundle.get_split("train").inputs
    assert tuple(x.shape[1:]) == (32, 32, 24, 1, 5)
    assert np.isfinite(x.numpy()).all()
