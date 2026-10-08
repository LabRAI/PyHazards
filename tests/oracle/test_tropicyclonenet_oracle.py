"""TropiCycloneNet checked against the official generator, discriminator, checkpoint and evaluation code.

The official code (xiaochengfuhuo/TropiCycloneNet, pinned) is imported from the checkout and run on the
CPU: its ``.cuda()`` calls are redirected to CPU copies while it runs (``_cpu_cuda``). The released
checkpoint (Zenodo 10.5281/zenodo.15024028, CC BY 4.0) is a pinned asset.
"""

from __future__ import annotations

import contextlib
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

from oracle_utils import import_from, load_definitions, oracle_asset, oracle_repo
from pyhazards.benchmarks.tc import track_intensity_metrics
from pyhazards.models import build_model
from pyhazards.models import tropicyclonenet as port_module
from pyhazards.models.tropicyclonenet import ENV_FEATURES, TrajectoryDiscriminator, load_tropicyclonenet_checkpoint

# Generator / discriminator arguments stored in the released checkpoint (its ``args``).
OFFICIAL_G_ARGS = dict(
    obs_len=8, pred_len=4, embedding_dim=32, encoder_h_dim=64, decoder_h_dim=64, mlp_dim=128, num_layers=1,
    noise_dim=(16,), noise_type="gaussian", noise_mix_type="ped", pooling_type=None, pool_every_timestep=False,
    dropout=0, bottleneck_dim=16, neighborhood_size=2.0, grid_size=8, batch_norm=0,
)
OFFICIAL_D_ARGS = dict(obs_len=8, pred_len=4, embedding_dim=32, h_dim=128, mlp_dim=128, num_layers=1, dropout=0, batch_norm=0, d_type="local")


@contextlib.contextmanager
def _cpu_cuda():
    """Run official code written for CUDA on the CPU: ``tensor.cuda()`` returns a CPU copy."""
    original = torch.Tensor.cuda
    torch.Tensor.cuda = lambda self, *args, **kwargs: self.to("cpu", copy=True)
    try:
        yield
    finally:
        torch.Tensor.cuda = original


def _official(module: str = "TCNM.models_prior_unet"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # nn.TransformerEncoder nested-tensor notice
        return import_from(oracle_repo("TropiCycloneNet"), module)


def _build(factory, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return factory(**kwargs)


def _checkpoint():
    path = oracle_asset("tropicyclonenet_checkpoint") / "checkpoint_with_model_16000.pt"
    return path, torch.load(path, map_location="cpu", weights_only=False)


def _batch(batch: int = 4, seed: int = 3):
    g = torch.Generator().manual_seed(seed)
    obs = 0.3 * torch.randn(8, batch, 4, generator=g)
    rel = torch.zeros_like(obs)
    rel[1:] = obs[1:] - obs[:-1]
    image = torch.rand(batch, 1, 8, 64, 64, generator=g)
    env = {}
    for key, width in ENV_FEATURES:
        if width == 1:
            env[key] = torch.rand(batch, 8, 1, generator=g)
        else:
            env[key] = torch.nn.functional.one_hot(torch.randint(0, width, (batch, 8), generator=g), width).float()
    seq_start_end = torch.stack([torch.arange(batch), torch.arange(batch) + 1], dim=1)
    return obs, rel, seq_start_end, image, env


def _assert_same_state(reference, port):
    ref_state, port_state = reference.state_dict(), port.state_dict()
    assert list(ref_state) == list(port_state)
    for key, value in ref_state.items():
        assert torch.equal(value, port_state[key]), key


def test_parameter_counts_and_seeded_initialisation_match_official():
    official = _official()
    torch.manual_seed(0)
    reference = _build(official.TrajectoryGenerator, **OFFICIAL_G_ARGS)
    torch.manual_seed(0)
    port = build_model("tropicyclonenet", task="regression")
    assert sum(p.numel() for p in reference.parameters()) == sum(p.numel() for p in port.parameters()) == 4_767_195
    _assert_same_state(reference, port)

    torch.manual_seed(0)
    reference_d = official.TrajectoryDiscriminator(**OFFICIAL_D_ARGS)
    torch.manual_seed(0)
    port_d = TrajectoryDiscriminator()
    assert sum(p.numel() for p in reference_d.parameters()) == sum(p.numel() for p in port_d.parameters()) == 231_009
    _assert_same_state(reference_d, port_d)


def test_released_checkpoint_loads_and_all_generators_match_official():
    official = _official()
    path, checkpoint = _checkpoint()
    assert checkpoint["args"]["best_k"] == 6 and checkpoint["args"]["noise_dim"] == (16,)
    reference = _build(official.TrajectoryGenerator, **OFFICIAL_G_ARGS)
    reference.load_state_dict(checkpoint["g_state"], strict=True)
    port = load_tropicyclonenet_checkpoint(path)  # sha256-checked, strict
    reference.eval()
    port.eval()
    obs, rel, sse, image, env = _batch()
    with torch.no_grad(), _cpu_cuda():
        torch.manual_seed(11)
        expected = reference(obs, rel, sse, image, env, num_samples=6, all_g_out=True)
    with torch.no_grad():
        torch.manual_seed(11)
        actual = port(obs, rel, sse, image, env, num_samples=6, all_g_out=True)
    for name, a, e in zip(("relative steps", "GPH frames", "chooser logits"), actual[:3], expected[:3]):
        torch.testing.assert_close(a, e, rtol=1e-5, atol=1e-6, msg=name)
    assert torch.equal(actual[3], torch.as_tensor(expected[3]))  # same RNG stream -> same sampled decoders
    assert actual[0].shape == (4, 6, 4, 4)

    # The released discriminator, on real-shaped inputs.
    reference_d = official.TrajectoryDiscriminator(**OFFICIAL_D_ARGS)
    reference_d.load_state_dict(checkpoint["d_state"], strict=True)
    port_d = load_tropicyclonenet_checkpoint(path, state="d_state")
    traj_rel = torch.cat([rel, actual[0][:, 0]], dim=0)
    traj = torch.cat([obs, obs[-1:] + torch.cumsum(actual[0][:, 0], dim=0)], dim=0)
    with torch.no_grad(), _cpu_cuda():
        expected_d = reference_d(traj, traj_rel, sse, actual[1])
    with torch.no_grad():
        actual_d = port_d(traj, traj_rel, sse, actual[1])
    torch.testing.assert_close(actual_d[0], expected_d[0], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual_d[1], expected_d[1], rtol=1e-5, atol=1e-6)


def test_sampled_forward_matches_official_loop_exactly():
    """``official_sample_loop=True`` reproduces the released evaluation path bit for bit (same seed)."""
    official = _official()
    _, checkpoint = _checkpoint()
    reference = _build(official.TrajectoryGenerator, **OFFICIAL_G_ARGS)
    reference.load_state_dict(checkpoint["g_state"])
    port = build_model("tropicyclonenet", task="regression", official_sample_loop=True)
    port.load_state_dict(checkpoint["g_state"], strict=True)
    reference.eval()
    port.eval()
    obs, rel, sse, image, env = _batch(batch=7, seed=5)
    for seed in (0, 1, 2):
        with torch.no_grad(), _cpu_cuda():
            torch.manual_seed(seed)
            expected = reference(obs, rel, sse, image, env, num_samples=6, all_g_out=False)
        with torch.no_grad():
            torch.manual_seed(seed)
            actual = port(obs, rel, sse, image, env, num_samples=6, all_g_out=False)
        torch.testing.assert_close(actual[0], expected[0], rtol=1e-5, atol=1e-6)
        assert torch.equal(actual[3], torch.as_tensor(expected[3]))


def test_default_sample_loop_uses_every_sampled_decoder(monkeypatch):
    """With zero noise, each sample must equal the output of the decoder that was sampled for it.

    The official loop visits decoder indices ``0 .. n_distinct - 1`` instead of the distinct sampled
    indices, so storms whose sampled index is not visited keep the placeholder value 1; PyHazards'
    default loop does not.
    """
    official = _official()
    _, checkpoint = _checkpoint()
    zeros = lambda shape, noise_type, **kwargs: torch.zeros(*shape)  # noqa: E731
    monkeypatch.setattr(port_module, "get_noise", zeros)
    port = build_model("tropicyclonenet", task="regression")
    port.load_state_dict(checkpoint["g_state"], strict=True)
    port.eval()
    obs, rel, sse, image, env = _batch(batch=9, seed=7)
    with torch.no_grad():
        every = port(obs, rel, sse, image, env, all_g_out=True, user_noise=torch.zeros(9, 16))[0]  # (L, 6, B, 4)
        torch.manual_seed(4)
        sampled, _, _, index = port(obs, rel, sse, image, env, num_samples=6)
    for b in range(9):
        for s in range(6):
            torch.testing.assert_close(sampled[:, s, b], every[:, int(index[b, s]), b], rtol=1e-5, atol=1e-6)

    official_module = _official()
    reference = _build(official_module.TrajectoryGenerator, **OFFICIAL_G_ARGS)
    reference.load_state_dict(checkpoint["g_state"])
    reference.eval()
    monkeypatch.setattr(official_module, "get_noise", lambda shape, noise_type: torch.zeros(*shape))
    with torch.no_grad(), _cpu_cuda():
        torch.manual_seed(4)
        expected, _, _, official_index = reference(obs, rel, sse, image, env, num_samples=6)
    official_index = torch.as_tensor(official_index)
    assert torch.equal(official_index, index)
    for s in range(6):
        visited = set(range(len(torch.unique(official_index[:, s]))))
        for b in range(9):
            if int(official_index[b, s]) in visited:
                torch.testing.assert_close(expected[:, s, b], sampled[:, s, b], rtol=1e-5, atol=1e-6)
            else:
                assert torch.all(expected[:, s, b] == 1.0)


def test_train_mode_matches_official():
    official = _official()
    _, checkpoint = _checkpoint()
    reference = _build(official.TrajectoryGenerator, **OFFICIAL_G_ARGS)
    reference.load_state_dict(checkpoint["g_state"])
    port = build_model("tropicyclonenet", task="regression")
    port.load_state_dict(checkpoint["g_state"])
    reference.train()
    port.train()
    obs, rel, sse, image, env = _batch(batch=5, seed=9)
    with _cpu_cuda():
        torch.manual_seed(21)
        expected = reference(obs, rel, sse, image, env, num_samples=6, all_g_out=True)
    torch.manual_seed(21)
    actual = port(obs, rel, sse, image, env, num_samples=6, all_g_out=True)
    for a, e in zip(actual[:3], expected[:3]):
        torch.testing.assert_close(a, e, rtol=1e-5, atol=1e-6)
    _assert_same_state(reference, port)  # BatchNorm running statistics updated identically


def test_forecast_matches_official_postprocessing():
    """Absolute positions / intensities equal the official relative_to_abs + toNE conversion."""
    utils = _official("TCNM.utils")
    losses = _official("TCNM.losses")
    path, _ = _checkpoint()
    port = load_tropicyclonenet_checkpoint(path).eval()
    obs, rel, sse, image, env = _batch(batch=3, seed=13)
    batch = {"obs_traj": obs, "obs_traj_rel": rel, "image_obs": image, "env_data": env, "seq_start_end": sse}
    with torch.no_grad():
        torch.manual_seed(0)
        forecast = port.forecast(batch, num_samples=6)
        torch.manual_seed(0)
        rel_pred = port(batch, num_samples=6)[0]
    track = utils.relative_to_abs(rel_pred[..., :2], obs[-1, :, :2])
    intensity = utils.relative_to_abs(rel_pred[..., 2:], obs[-1, :, 2:])
    for sample in range(6):
        tenths, me = losses.toNE(track[:, sample].clone(), intensity[:, sample].clone())
        torch.testing.assert_close(forecast["lon"][:, sample], (tenths[..., 0] / 10).T, rtol=1e-5, atol=1e-4)
        torch.testing.assert_close(forecast["lat"][:, sample], (tenths[..., 1] / 10).T, rtol=1e-5, atol=1e-4)
        torch.testing.assert_close(forecast["pres"][:, sample], me[..., 0].T, rtol=1e-5, atol=1e-3)
        torch.testing.assert_close(forecast["wind"][:, sample], me[..., 1].T, rtol=1e-5, atol=1e-3)


def test_best_of_k_metrics_match_official_evaluation():
    """PyHazards' best-of-k errors with the TCN distance equal evaluate_model_Me.py's per-lead errors."""
    repo = oracle_repo("TropiCycloneNet")
    losses = _official("TCNM.losses")
    helpers = load_definitions(repo / "scripts" / "evaluate_model_Me.py", ["evaluate_helper", "ve_evaluate_helper"], {"torch": torch})
    g = torch.Generator().manual_seed(0)
    storms, samples, lead = 10, 6, 4
    # Physical values in the official units: track in tenths of a degree, pressure hPa, wind m/s.
    true_track = torch.stack([1300 + 50 * torch.rand(lead, storms, generator=g), 150 + 50 * torch.rand(lead, storms, generator=g)], -1)
    true_me = torch.stack([980 + 20 * torch.rand(lead, storms, generator=g), 30 + 10 * torch.rand(lead, storms, generator=g)], -1)
    fake_track = true_track.unsqueeze(1) + 10 * torch.randn(lead, samples, storms, 2, generator=g)
    fake_me = true_me.unsqueeze(1) + 3 * torch.randn(lead, samples, storms, 2, generator=g)
    tde = [losses.trajectory_displacement_error(fake_track[:, s], true_track, mode="raw") for s in range(samples)]
    ve = [losses.value_error(fake_me[:, s], true_me, mode="raw") for s in range(samples)]
    sse = torch.stack([torch.arange(storms), torch.arange(storms) + 1], dim=1)
    official_track = torch.stack([helpers["evaluate_helper"]([x[:, i] for x in tde], sse) for i in range(lead)]) / storms
    official_me = torch.tensor([[float(v) for v in helpers["ve_evaluate_helper"]([x[:, i] for x in ve], sse)] for i in range(lead)]) / storms

    forecast = {
        "lon": fake_track[..., 0].permute(2, 1, 0) / 10,
        "lat": fake_track[..., 1].permute(2, 1, 0) / 10,
        "pres": fake_me[..., 0].permute(2, 1, 0),
        "wind": fake_me[..., 1].permute(2, 1, 0),
    }
    target = {
        "lon": true_track[..., 0].T / 10,
        "lat": true_track[..., 1].T / 10,
        "pres": true_me[..., 0].T,
        "wind": true_me[..., 1].T,
    }
    metrics = track_intensity_metrics(
        {k: v.double() for k, v in forecast.items()}, {k: v.double() for k, v in target.items()}, [6, 12, 18, 24], "tcn_equirectangular"
    )
    for i, hours in enumerate((6, 12, 18, 24)):
        assert metrics[f"best_of_k_track_error_km_{hours}h"] == pytest.approx(float(official_track[i]), rel=1e-5)
        assert metrics[f"best_of_k_pressure_mae_{hours}h"] == pytest.approx(float(official_me[i, 0]), rel=1e-5)
        assert metrics[f"best_of_k_intensity_mae_{hours}h"] == pytest.approx(float(official_me[i, 1]), rel=1e-5)
    # Great-circle distance differs from the official equirectangular approximation by well under 1 %.
    great_circle = track_intensity_metrics(
        {k: v.double() for k, v in forecast.items()}, {k: v.double() for k, v in target.items()}, [6, 12, 18, 24]
    )
    assert great_circle["best_of_k_track_error_km"] == pytest.approx(metrics["best_of_k_track_error_km"], rel=1e-2)


def test_tcnd_reader_matches_official_loader(tmp_path):
    """``tropicyclonenet_dataset`` equals the official TrajectoryDataset + seq_collate on a real storm.

    Data1d and Env-Data are the real files of tests/fixtures/tcnd (EP 2018 Carlotta); the GPH crops
    are random values in the official file layout (100 x 100 float64), since the real ones are too
    large to vendor.
    """
    import shutil

    from oracle_utils import oracle_package
    from pyhazards.datasets import load_dataset
    from pyhazards.datasets.tc import read_tcnd_track

    oracle_package("cv2", "5.0.0", "requirements-tc.txt", distribution="opencv-python-headless")
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "tcnd"
    root = tmp_path / "tcnd"
    shutil.copytree(fixture / "BST_data", root / "BST_data")
    shutil.copytree(fixture / "Env_data", root / "Env_data")
    _, dates, _ = read_tcnd_track(fixture / "BST_data" / "EP" / "test" / "EP2018BSTCARLOTTA.txt")
    storm = root / "ERA5_gph500" / "EP" / "2018" / "CARLOTTA"
    storm.mkdir(parents=True)
    rng = np.random.default_rng(0)
    for date in dates:
        np.save(storm / f"{date}.npy", rng.uniform(55000.0, 59500.0, size=(100, 100)))

    loader = _official("TCNM.data.trajectoriesWithMe_unet")
    dataset = loader.TrajectoryDataset({"root": str(root), "type": "test"}, obs_len=8, pred_len=4, skip=1, delim="\t", areas=["EP"])
    batch = loader.seq_collate([dataset[i] for i in range(len(dataset))])
    obs_traj, pred_traj, obs_rel, _, _, _, _, obs_me, pred_me, obs_rel_me = batch[:10]
    image_obs, env = batch[13], batch[15]

    port = load_dataset("tropicyclonenet_dataset", root=str(root), areas=["EP"], splits=["test"]).load().splits["test"]
    assert len(dataset) == len(port.targets) == 7
    torch.testing.assert_close(port.inputs["obs_traj"], torch.cat([obs_traj, obs_me], dim=2))
    torch.testing.assert_close(port.inputs["obs_traj_rel"], torch.cat([obs_rel, obs_rel_me], dim=2))
    torch.testing.assert_close(port.inputs["image_obs"], image_obs, rtol=1e-5, atol=1e-6)
    for key, _ in ENV_FEATURES:
        torch.testing.assert_close(port.inputs["env_data"][key], env[key], msg=key)
    tenths, me = _official("TCNM.losses").toNE(pred_traj.clone(), pred_me.clone())
    expected = torch.stack([tenths[..., 1] / 10, tenths[..., 0] / 10, me[..., 0], me[..., 1]], dim=-1).permute(1, 0, 2)
    torch.testing.assert_close(port.targets, expected, rtol=1e-5, atol=1e-4)
