import pytest
import torch
import torch.nn.functional as F

from pyhazards.metrics.wavefield import wavefield_acc, wavefield_metrics, wavefield_rfne, wavefield_rmse
from pyhazards.models import build_model
from pyhazards.models.wavecastnet import (
    ConvLEMCell,
    WaveCastNet,
    WaveCastNetLoss,
    WaveCastNetSparse,
    WavefieldMetrics,
    load_wavecastnet_checkpoint,
)


def _count(model):
    return sum(p.numel() for p in model.parameters())


def _coords(stations=20, height=32, width=24, seed=0):
    """Distinct (row, column) grid points."""
    flat = torch.randperm(height * width, generator=torch.Generator().manual_seed(seed))[:stations]
    return torch.stack([flat // width, flat % width], dim=1)


def test_parameter_counts_and_official_names():
    dense = WaveCastNet()
    assert _count(dense) == 10_093_242
    counts = {name: _count(module) for name, module in dense.named_children()}
    assert counts == {
        "encoder": 209_844,
        "decoder": 331_920,
        "encoder_1_convlem": 2_387_808,
        "encoder_2_convlem": 2_387_808,
        "decoder_1_convlem": 2_387_808,
        "decoder_2_convlem": 2_387_808,
        "conv1": 246,
    }
    state = dense.state_dict()
    for key in (
        "encoder.model.encoder_layer1.layer.0.weight",
        "encoder.model.encoder_layer3.layer.1.running_var",
        "decoder.model.decoder_layer1.layer.0.weight",
        "encoder_1_convlem.convx.weight",
        "decoder_2_convlem.W_z4",
        "conv1.bias",
    ):
        assert key in state
    assert state["encoder_1_convlem.W_z1"].shape == (144, 43, 28)
    assert state["encoder_1_convlem.convx.weight"].shape == (720, 144, 3, 3)
    assert state["encoder_1_convlem.convy.weight"].shape == (576, 144, 3, 3)

    sparse = WaveCastNetSparse(station_coords=_coords(564, 344, 224))
    assert _count(sparse) == 16_535_430
    assert sparse.state_dict()["encoder.FC1.0.weight"].shape == (1204, 564)
    assert sparse.state_dict()["encoder.FC2.0.weight"].shape == (4816, 1204)
    assert "encoder.station_coords" not in sparse.state_dict()  # a non-persistent buffer


def test_cell_variants_and_initialisation():
    cell = ConvLEMCell(4, 6, (5, 3), reset_gate=False)
    assert not hasattr(cell, "W_z4") and cell.convx.out_channels == 24 and cell.convy.out_channels == 18
    assert all(float(p.detach().abs().sum()) > 0 for p in (cell.W_z1, cell.W_z2))  # xavier, not zero
    assert float(cell.convx.bias.detach().abs().sum()) == 0.0
    h, c = cell(torch.randn(2, 4, 5, 3), torch.zeros(2, 6, 5, 3), torch.zeros(2, 6, 5, 3))
    assert h.shape == c.shape == (2, 6, 5, 3)
    with pytest.raises(ValueError, match="shape"):
        cell(torch.randn(2, 4, 4, 3), torch.zeros(2, 6, 4, 3), torch.zeros(2, 6, 4, 3))
    with pytest.raises(ValueError, match="activation"):
        ConvLEMCell(4, 6, (5, 3), activation="gelu")


def test_forward_horizon_rollout_and_noise_hooks():
    torch.manual_seed(0)
    model = build_model("wavecastnet", task="forecasting", height=16, width=24, future_seq=2).eval()
    x = torch.randn(2, 3, 4, 16, 24)
    with torch.no_grad():
        assert model(x).shape == (2, 3, 2, 16, 24)
        assert model(x, 5).shape == (2, 3, 5, 16, 24)
        # The decoder starts from uniform noise even in evaluation mode.
        assert not torch.equal(model(x), model(x))
        same = [model(x, generator=torch.Generator().manual_seed(3)) for _ in range(2)]
        assert torch.equal(same[0], same[1])
        noise = torch.rand(2, 144, 2, 3, generator=torch.Generator().manual_seed(3))
        assert torch.equal(model(x, decoder_noise=noise), same[0])
        torch.manual_seed(5)
        first = model(x)
        torch.manual_seed(5)
        assert torch.equal(model(x), first)
        rolled = model.rollout(x, 5, generator=torch.Generator().manual_seed(1))
        assert rolled.shape == (2, 3, 5, 16, 24)
        g = torch.Generator().manual_seed(1)
        part1 = model(x, 2, generator=g)
        part2 = model(part1, 2, generator=g)
        part3 = model(part2, 1, generator=g)
        assert torch.equal(rolled, torch.cat([part1, part2, part3], dim=2))
        with pytest.raises(ValueError, match="shape"):
            model(x, decoder_noise=torch.rand(2, 144, 3, 3))


def test_input_validation_and_builder_arguments():
    model = WaveCastNet(height=16, width=16)
    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(1, 3, 2, 16, 24))
    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(1, 2, 2, 16, 16))
    with pytest.raises(ValueError, match="multiples of 8"):
        WaveCastNet(height=20, width=16)
    with pytest.raises(ValueError, match="48"):
        WaveCastNet(height=16, width=16, num_kernels=64)
    with pytest.raises(TypeError, match="no longer"):
        build_model("wavecastnet", task="forecasting", hidden_dim=32)
    with pytest.raises(ValueError, match="forecasting"):
        build_model("wavecastnet", task="classification")
    with pytest.raises(ValueError, match="variant"):
        build_model("wavecastnet", task="forecasting", variant="graph", height=16, width=16)
    assert isinstance(build_model("wavecastnet", task="regression", height=16, width=16), WaveCastNet)
    with pytest.raises(ValueError, match="dense checkpoint"):
        build_model("wavecastnet", task="forecasting", height=16, width=16, pretrained="dense")


def test_sparse_station_masking():
    coords = _coords(20)
    torch.manual_seed(0)
    model = build_model("wavecastnet", task="forecasting", variant="sparse", height=32, width=24, station_coords=coords).eval()
    assert isinstance(model, WaveCastNetSparse) and model.mask_mode
    x = torch.randn(2, 3, 3, 32, 24)
    noise = torch.rand(2, 144, 4, 3)
    with torch.no_grad():
        everything = model(x, station_mask=torch.ones(20, dtype=torch.bool), decoder_noise=noise)
        model.mask_mode = False
        assert torch.equal(model(x, decoder_noise=noise), everything)
        model.mask_mode = True
        # Masked stations are zeroed: changing the wavefield there changes nothing.
        keep = torch.arange(20) % 2 == 0
        changed = x.clone()
        for row, col in coords[~keep].tolist():
            changed[:, :, :, row, col] += 10.0
        a = model(x, station_mask=keep, decoder_noise=noise)
        assert torch.equal(model(changed, station_mask=keep, decoder_noise=noise), a)
        assert not torch.equal(model(changed, station_mask=torch.ones(20, dtype=torch.bool), decoder_noise=noise), everything)
        # In evaluation the official model still draws a random station mask.
        g1, g2 = torch.Generator().manual_seed(1), torch.Generator().manual_seed(2)
        assert not torch.equal(model(x, decoder_noise=noise, generator=g1), model(x, decoder_noise=noise, generator=g2))
        with pytest.raises(ValueError, match="shape"):
            model(x, station_mask=torch.ones(3, dtype=torch.bool))
    with pytest.raises(ValueError, match="station_coords"):
        WaveCastNetSparse(station_coords=torch.tensor([[40, 3]]), height=32, width=24)


def test_loss_variants():
    torch.manual_seed(0)
    pred, target = torch.randn(4, 3, 2, 8, 8), torch.randn(4, 3, 2, 8, 8)
    official = WaveCastNetLoss()(pred, target)
    expected = F.smooth_l1_loss(pred, target, beta=0.2) + 0.1  # Huber / delta + delta / 2
    torch.testing.assert_close(official, expected)
    torch.testing.assert_close(WaveCastNetLoss(delta=0.5, variant="paper")(pred, target), F.huber_loss(pred, target, delta=0.5))
    with pytest.raises(ValueError, match="variant"):
        WaveCastNetLoss(variant="l2")


def test_wavefield_metrics_are_per_sample_and_channel():
    target = torch.zeros(2, 3, 2, 2, 2)
    target[:, :, 0, 0, 0] = 1.0
    target[:, 2] *= 100.0  # a channel with a much larger scale
    pred = target.clone()
    pred[:, 0] *= -1.0  # channel x anti-correlated
    acc = wavefield_acc(pred, target)
    assert acc.shape == (2, 3)
    torch.testing.assert_close(acc, torch.tensor([[-1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]]))
    torch.testing.assert_close(wavefield_rfne(pred, target), torch.tensor([[2.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
    torch.testing.assert_close(wavefield_rmse(pred, target)[:, 0], torch.full((2,), (4.0 / 8) ** 0.5))
    metrics = wavefield_metrics(pred, target, ["x", "y", "z"])
    assert metrics["acc"] == pytest.approx(1.0 / 3.0) and metrics["acc_x"] == -1.0 and metrics["acc_z"] == 1.0
    assert metrics["rfne"] == pytest.approx(2.0 / 3.0) and metrics["rfne_y"] == 0.0
    assert WavefieldMetrics.compute_all(pred, target)["ACC"] == pytest.approx(1.0 / 3.0)
    with pytest.raises(ValueError, match="shape"):
        wavefield_acc(pred[0], target[0])


def test_checkpoint_loader_strips_data_parallel_prefix(tmp_path):
    source = WaveCastNet(height=16, width=16)
    path = tmp_path / "state.pt"
    torch.save({f"module.{k}": v for k, v in source.state_dict().items()}, path)
    target = load_wavecastnet_checkpoint(WaveCastNet(height=16, width=16), path)
    for key, value in source.state_dict().items():
        assert torch.equal(target.state_dict()[key], value)
