import numpy as np
import pytest
import torch

from pyhazards.models import build_model
from pyhazards.models.hurricast import (
    DECODER_CONFIGS,
    HURRICAST_STAT_FEATURES,
    Hurricast,
    hurricast_network,
    hurricast_training_loss,
    hurricast_xgboost_columns,
    hurricast_xgboost_features,
)


def _n_params(model):
    return sum(p.numel() for p in model.parameters())


def _inputs(n=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "x_stat": torch.randn(n, 8, 30, generator=g),
        "x_viz": torch.randn(n, 8, 9, 25, 25, generator=g),
        "position": torch.randn(n, 2, generator=g) * 10 + 20,
    }


def test_paper_configuration_parameter_counts():
    intensity = build_model("hurricast", task="regression", predictor="network")
    track = build_model("hurricast", task="regression", target="displacement", predictor="network")
    assert _n_params(intensity) == 2_969_771 and _n_params(track) == 2_969_914
    assert _n_params(intensity.network.encoder) == 2_732_800
    assert _n_params(intensity.network.decoder) == 236_828
    assert intensity.network.embedding_dim == 142 and intensity.network.n_stat == 14
    # Official key layout under ``network.``.
    keys = list(intensity.state_dict())
    assert keys[:3] == ["target_mean", "target_std", "network.encoder.layers.0.weight"] and "network.decoder.pe.pe" in keys
    assert _n_params(hurricast_network("displacement", "lstm_config_best_dis")) == 3_864_194


def test_network_shapes_and_validation():
    model = build_model("hurricast", task="regression", target="displacement", predictor="network").eval()
    x = _inputs()
    with torch.no_grad():
        out = model(x)
        forecast = model.forecast(x)
    assert out.shape == (6, 2)
    torch.testing.assert_close(forecast["lat"][:, 0], x["position"][:, 0] + out[:, 0])
    with pytest.raises(ValueError, match="x_viz must be shaped"):
        model({**x, "x_viz": torch.randn(6, 8, 3, 25, 25)})
    with pytest.raises(ValueError, match="x_stat"):
        model({"x_stat": torch.randn(6, 30), "x_viz": x["x_viz"]})
    with pytest.raises(ValueError, match="position"):
        model.forecast({"x_stat": x["x_stat"], "x_viz": x["x_viz"]})
    with pytest.raises(ValueError, match="regression"):
        build_model("hurricast", task="classification")
    with pytest.raises(ValueError, match="decoder_config"):
        build_model("hurricast", task="regression", decoder_config="gru_paper")
    # Statistics only (official transformer_config_noviz, no encoder): 10 features per step.
    noviz = hurricast_network("intensity", "transformer_config_noviz", None).eval()
    assert noviz(torch.randn(2, 8, 10)).shape == (2, 1)
    for name in DECODER_CONFIGS:
        if name != "transformer_config_noviz":
            net = hurricast_network("intensity", name, "split_encoder_config").eval()
            assert net.get_embeddings(torch.randn(2, 8, 14), torch.randn(2, 8, 9, 25, 25)).shape == (2, net.embedding_dim)


def test_xgboost_feature_layout():
    columns = hurricast_xgboost_columns(8)
    assert len(columns) == 14 * 8 + 16
    assert columns[:30] == [f"{name}_0" for name in HURRICAST_STAT_FEATURES]  # every feature of the oldest step
    assert all(not c.startswith("cat") for c in columns[30:])  # later steps: numerical features only
    x_stat = torch.arange(2 * 8 * 30, dtype=torch.float32).reshape(2, 8, 30)
    features = hurricast_xgboost_features(x_stat, torch.ones(2, 142))
    assert features.shape == (2, 128 + 142)
    np.testing.assert_array_equal(features[0, :30], np.arange(30))
    np.testing.assert_array_equal(features[0, 30:32], [30, 31])  # LAT_1, LON_1
    with pytest.raises(ValueError, match=r"\(batch, T, 30\)"):
        hurricast_xgboost_features(torch.zeros(2, 8, 14))


def test_training_loss_matches_the_paper_formula():
    net = hurricast_network("intensity")
    out, target = torch.randn(4, 1), torch.randn(4, 1)
    l2 = sum((p**2).sum() for n, p in net.named_parameters() if "weight" in n)
    torch.testing.assert_close(hurricast_training_loss(out, target, net, 0.01), ((out - target) ** 2).mean() + 2 / 4 * 0.01 * l2)
    with pytest.raises(ValueError, match="shape"):
        hurricast_training_loss(out, target[:, 0], net, 0.01)


def test_two_stage_fit_predict_save_load(tmp_path):
    from sklearn.linear_model import Ridge

    torch.manual_seed(0)
    model = build_model("hurricast", task="regression", target="displacement", estimator=Ridge(alpha=1.0))
    x = _inputs(n=12)
    displacement = torch.randn(12, 2)
    targets = (x["position"] + displacement).unsqueeze(1)  # tc.track_intensity layout: position 24 h ahead
    model.fit(x, targets, x, targets, epochs=2, batch_size=4)
    torch.testing.assert_close(model.target_mean, displacement.mean(0), rtol=1e-5, atol=1e-5)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (12, 2)
    features = model.xgboost_features(x)
    assert features.shape == (12, 128 + 142)
    expected = model.xgboost.estimator.predict(features) * model.target_std.numpy() + model.target_mean.numpy()
    np.testing.assert_allclose(out.numpy(), expected, rtol=1e-5, atol=1e-5)
    path = tmp_path / "hurricast.joblib"
    model.save(path)
    restored = Hurricast.load(path)
    with torch.no_grad():
        torch.testing.assert_close(restored(x), out)
    from pyhazards.engine import Trainer

    with pytest.raises(TypeError, match="two stages"):
        Trainer(model, device="cpu").fit(None)


def test_network_training_keeps_the_best_validation_epoch():
    torch.manual_seed(0)
    model = build_model("hurricast", task="regression", predictor="network")
    x = _inputs(n=8)
    before = {k: v.clone() for k, v in model.network.state_dict().items()}
    model.fit(x, torch.randn(8) * 10 + 50, x, torch.randn(8) * 10 + 50, epochs=2, batch_size=4, learning_rate=1e-3)
    assert any(not torch.equal(before[k], v) for k, v in model.network.state_dict().items())
    assert not model.network.training
    history = model.fit_network(x, torch.randn(8, 1), val=(x, torch.randn(8, 1)), epochs=3, batch_size=4)
    assert len(history) == 3
    with pytest.raises(ValueError, match="batch_size"):
        model.fit_network(x, torch.randn(8, 1), batch_size=64)


def test_xgboost_stage():
    pytest.importorskip("xgboost")
    x = _inputs(n=40)
    y = torch.randn(40) * 10 + 60
    model = build_model("hurricast", task="regression", n_jobs=1, n_estimators=5)
    model.fit(x, y, train_network=False)
    with torch.no_grad():
        assert model(x).shape == (40, 1)
    assert model.xgboost.estimator.get_params()["max_depth"] == 8
    stat_only = build_model("hurricast", task="regression", use_embeddings=False, n_jobs=1, n_estimators=5)
    assert stat_only.network is None
    stat_only.fit({"x_stat": x["x_stat"]}, y)
    assert stat_only({"x_stat": x["x_stat"]}).shape == (40, 1)
