"""Earthformer checked against the official ``CuboidTransformerModel``.

Reference: amazon-science/earth-forecasting-transformer at 7732b03 (pinned in repos.yaml, Apache-2.0),
``src/earthformer/cuboid_transformer/cuboid_transformer.py``; it needs only torch and einops
(requirements-earthformer.txt). Configurations are the ``model`` sections of the official YAML files,
turned into constructor arguments the way the training scripts do (each pattern repeated once per
hierarchy; ``block_units`` is commented out in every file and defaults to None).

Checks:

- the presets equal the official YAML files, and parameter counts at the official configurations
  (SEVIR 8,659,677, Moving-MNIST 6,702,109, SEVIR-LR, ICAR-ENSO) and at the configurations that
  reproduce the paper's counts (15.1M / 13.1M SEVIR, 7.61M / 6.61M Moving-MNIST, 7.6M / 6.6M ENSO);
- the same seed gives identical parameter names, buffers and initial values;
- outputs in eval mode and in train mode (dropout 0.1, same seed) and input gradients agree, for the
  SEVIR, Moving-MNIST and ENSO configurations and for other patterns and options of the class;
- the official SEVIR and ICAR-ENSO checkpoints load with ``strict=True`` and give the same outputs
  at their full input sizes (13x384x384 and 12x24x48).

To keep CPU time low the SEVIR architecture is compared at 13x48x48 and 13x50x50 frames (the 12x
initial down-sampling then gives 4x4 and 5x5 tokens; 50 exercises the padding paths) and
Moving-MNIST at 10x32x32; ENSO and the checkpoints run at their reference sizes.
"""

from __future__ import annotations

import pytest
import torch
import yaml

from oracle_utils import import_from, oracle_asset, oracle_package, oracle_repo
from pyhazards.models import build_model
from pyhazards.models.earthformer import (
    EARTHFORMER_CHECKPOINTS,
    EARTHFORMER_CONFIGS,
    CuboidTransformerModel,
    Earthformer,
    EarthformerSegmenter,
    earthformer_checkpoint_path,
    earthformer_config,
)

CONFIG_FILES = {
    "sevir": "sevir/earthformer_sevir_v1.yaml",
    "sevir_cfg": "sevir/cfg_sevir.yaml",
    "sevir_lr": "sevir/cfg_sevirlr.yaml",
    "moving_mnist": "moving_mnist/cfg.yaml",
    "nbody": "nbody/cfg.yaml",
    "enso": "enso/earthformer_enso_v1.yaml",
    "enso_cfg": "enso/cfg.yaml",
}
PRESET_OF = {"sevir_cfg": "sevir", "enso_cfg": "enso"}

# Global vectors as implied by the paper's parameter counts: the global vectors share the local
# q/k/v projections, have no FFN of their own and the decoder self-attention does not use them.
PAPER_GLOBAL = dict(
    num_global_vectors=8,
    use_dec_self_global=False,
    use_dec_cross_global=False,
    use_global_vector_ffn=False,
    use_global_self_attn=False,
    separate_global_qkv=False,
)
NO_GLOBAL = dict(num_global_vectors=0)
PAPER_COUNTS = [
    # (official file, overrides, parameters, paper value)
    ("sevir", dict(enc_depth=[2, 2], dec_depth=[2, 2], **PAPER_GLOBAL), 15_082_069),  # Table 6: 15.1M
    ("sevir", dict(enc_depth=[2, 2], dec_depth=[2, 2], **NO_GLOBAL), 13_075_029),  # Table 6: 13.1M w/o global
    ("moving_mnist", dict(pos_embed_type="t+h+w", **PAPER_GLOBAL), 7_610_781),  # Tables 4/5: 7.61M
    ("moving_mnist", dict(pos_embed_type="t+h+w"), 6_611_997),  # Tables 4/5: 6.61M (Axial, w/o global)
    ("enso", dict(enc_depth=[4, 4], dec_depth=[4, 4], **PAPER_GLOBAL), 7_600_077),  # Table 7: 7.6M
    ("enso", dict(enc_depth=[4, 4], dec_depth=[4, 4]), 6_601_293),  # Table 7: 6.6M w/o global
]


def _official_module():
    oracle_package("einops", "0.8.2", "requirements-earthformer.txt")
    root = oracle_repo("earth-forecasting-transformer")
    return import_from(root / "src", "earthformer.cuboid_transformer.cuboid_transformer")


def _yaml_model_section(name: str) -> dict:
    root = oracle_repo("earth-forecasting-transformer")
    path = root / "scripts" / "cuboid_transformer" / CONFIG_FILES[name]
    section = yaml.safe_load(path.read_text(encoding="utf-8"))["model"]
    section.setdefault("block_units", None)  # commented out in the files; the scripts default it to None
    return section


def _official_kwargs(name: str, **overrides) -> dict:
    """Constructor arguments as built by the official training scripts (train_cuboid_*.py)."""
    cfg = dict(_yaml_model_section(name))
    cfg.update(overrides)
    num_blocks = len(cfg["enc_depth"])

    def expand(pattern):
        return [pattern] * num_blocks if isinstance(pattern, str) else list(pattern)

    kwargs = {key: value for key, value in cfg.items() if key not in ("self_pattern", "cross_self_pattern", "cross_pattern")}
    kwargs["enc_attn_patterns"] = expand(cfg["self_pattern"])
    kwargs["dec_self_attn_patterns"] = expand(cfg["cross_self_pattern"])
    kwargs["dec_cross_attn_patterns"] = expand(cfg["cross_pattern"])
    return kwargs


def _n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _assert_same_state(reference: torch.nn.Module, port: torch.nn.Module) -> None:
    ref_state, port_state = reference.state_dict(), port.state_dict()
    assert list(ref_state) == list(port_state)
    for key, value in ref_state.items():
        assert value.dtype == port_state[key].dtype, key
        assert torch.equal(value, port_state[key]), key


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def _randomise(reference: torch.nn.Module, port: torch.nn.Module, seed: int) -> None:
    """Perturb every floating-point parameter so that zero-initialised biases and vectors matter."""
    generator = torch.Generator().manual_seed(seed)
    state = reference.state_dict()
    for value in state.values():
        if value.is_floating_point():
            value.add_(0.05 * torch.randn(value.shape, generator=generator))
    reference.load_state_dict(state, strict=True)
    port.load_state_dict(state, strict=True)


def _build_pair(official, kwargs: dict, port_cls=CuboidTransformerModel):
    torch.manual_seed(0)
    reference = official.CuboidTransformerModel(**kwargs)
    torch.manual_seed(0)
    port = port_cls(**kwargs)
    return reference, port


def _compare(reference, port, batch: int = 2, seed: int = 1, train: bool = True) -> None:
    """Seeded weights, then eval outputs, train outputs (dropout) and input gradients."""
    _assert_same_state(reference, port)
    _randomise(reference, port, seed)
    torch.manual_seed(seed + 1)
    x = torch.randn((batch,) + tuple(port.input_shape))
    reference.eval()
    port.eval()
    with torch.no_grad():
        _assert_close(port(x), reference(x))
    if not train:
        return
    reference.train()
    port.train()
    x_ref = x.clone().requires_grad_(True)
    x_port = x.clone().requires_grad_(True)
    torch.manual_seed(seed + 2)
    out_ref = reference(x_ref)
    torch.manual_seed(seed + 2)
    out_port = port(x_port)
    _assert_close(out_port, out_ref)
    weights = torch.randn(out_ref.shape, generator=torch.Generator().manual_seed(seed + 3))
    (out_ref * weights).sum().backward()
    (out_port * weights).sum().backward()
    _assert_close(x_port.grad, x_ref.grad)
    for (name, p_ref), (_, p_port) in zip(reference.named_parameters(), port.named_parameters()):
        if p_ref.grad is None:
            assert p_port.grad is None, name
        else:
            _assert_close(p_port.grad, p_ref.grad)


@pytest.mark.parametrize("name", sorted(CONFIG_FILES))
def test_presets_equal_official_yaml(name):
    assert EARTHFORMER_CONFIGS[PRESET_OF.get(name, name)] == _yaml_model_section(name)


@pytest.mark.parametrize(
    "name, expected",
    [("sevir", 8_659_677), ("moving_mnist", 6_702_109), ("sevir_lr", 1_505_069), ("enso", 1_394_325)],
)
def test_parameter_counts_at_official_configs(name, expected):
    official = _official_module()
    with torch.device("meta"):
        reference = official.CuboidTransformerModel(**_official_kwargs(name))
        port = build_model("earthformer", task="forecasting", config=name)
    assert _n_params(reference) == _n_params(port) == expected
    assert [(k, v.shape) for k, v in reference.state_dict().items()] == [(k, v.shape) for k, v in port.state_dict().items()]


@pytest.mark.parametrize("name, overrides, expected", PAPER_COUNTS)
def test_parameter_counts_reproducing_the_paper(name, overrides, expected):
    official = _official_module()
    with torch.device("meta"):
        reference = official.CuboidTransformerModel(**_official_kwargs(name, **overrides))
        port = build_model("earthformer", task="forecasting", config=name, **overrides)
    assert _n_params(reference) == _n_params(port) == expected


def test_sevir_architecture_matches_reference():
    official = _official_module()
    for size in (48, 50):
        kwargs = _official_kwargs("sevir", input_shape=[13, size, size, 1], target_shape=[12, size, size, 1])
        assert kwargs == earthformer_config("sevir", input_shape=[13, size, size, 1], target_shape=[12, size, size, 1])
        reference, port = _build_pair(official, kwargs)
        _compare(reference, port, seed=size)


def test_moving_mnist_architecture_matches_reference():
    official = _official_module()
    kwargs = _official_kwargs("moving_mnist", input_shape=[10, 32, 32, 1], target_shape=[10, 32, 32, 1])
    reference, port = _build_pair(official, kwargs)
    _compare(reference, port)


def test_enso_architecture_matches_reference():
    official = _official_module()
    reference, port = _build_pair(official, _official_kwargs("enso"))
    _compare(reference, port)


SMALL = dict(input_shape=[5, 16, 16, 2], target_shape=[3, 16, 16, 3], base_units=32, num_heads=4)
VARIANTS = {
    # The training scripts' own defaults for SEVIR: shared global q/k/v, global FFN, decoder global
    # vectors, hierarchical decoder embeddings and gradient checkpointing of FFN and attention.
    "sevir_script_defaults": dict(
        num_global_vectors=8,
        use_dec_self_global=True,
        use_dec_cross_global=True,
        use_global_vector_ffn=True,
        use_global_self_attn=False,
        separate_global_qkv=False,
        dec_hierarchical_pos_embed=True,
        checkpoint_level=2,
        pos_embed_type="t+hw",
    ),
    # Separate global projections with wider global vectors, decoder global self-attention.
    "separate_global_ratio2": dict(
        num_global_vectors=4,
        use_dec_self_global=True,
        dec_self_update_global=False,
        use_dec_cross_global=True,
        use_global_vector_ffn=True,
        use_global_self_attn=True,
        separate_global_qkv=True,
        global_dim_ratio=2,
        checkpoint_level=1,
    ),
    # Shifted windows with masks, dilated cuboids, padding with 'ignore'.
    "video_swin_padding_ignore": dict(
        self_pattern="video_swin_2x4",
        cross_self_pattern="spatial_lg_v1",
        cross_pattern="cross_4x4_heter",
        padding_type="ignore",
        input_shape=[5, 20, 20, 2],
        target_shape=[3, 20, 20, 3],
        num_global_vectors=2,
    ),
    "axial_dilate_gated_nearest": dict(
        self_pattern="axial_space_dilate_2",
        cross_self_pattern="divided_st",
        cross_pattern="cross_2x2_lg",
        padding_type="nearest",
        gated_ffn=True,
        ffn_activation="leaky",
        z_init_method="nearest_interp",
        dec_use_first_self_attn=True,
        enc_use_inter_ffn=False,
        dec_use_inter_ffn=False,
        attn_linear_init_mode="1",
        ffn_linear_init_mode="1",
    ),
    "full_attention_rms_last": dict(
        self_pattern="full",
        cross_self_pattern="full",
        cross_pattern="cross_1x1",
        norm_layer="rms_norm",
        z_init_method="last",
        dec_cross_last_n_frames=2,
        use_relative_pos=False,
        self_attn_use_final_proj=False,
        initial_downsample_scale=[1, 1, 2],
    ),
    "three_hierarchies_mean": dict(
        enc_depth=[1, 1, 1],
        dec_depth=[1, 1, 1],
        input_shape=[4, 32, 32, 2],
        target_shape=[2, 32, 32, 3],
        z_init_method="mean",
        dec_cross_start=1,
        initial_downsample_conv_layers=1,
        final_upsample_conv_layers=2,
        block_units=[32, 48, 64],
    ),
}


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_other_patterns_and_options_match_reference(variant):
    official = _official_module()
    overrides = dict(SMALL)
    overrides.update(VARIANTS[variant])
    kwargs = _official_kwargs("moving_mnist", **overrides)
    reference, port = _build_pair(official, kwargs)
    _compare(reference, port, seed=7)


def test_forecasting_and_segmentation_wrappers_keep_the_core_parameters():
    official = _official_module()
    kwargs = _official_kwargs("sevir", input_shape=[4, 24, 24, 3], target_shape=[2, 24, 24, 2])
    reference, port = _build_pair(official, kwargs, port_cls=Earthformer)
    _assert_same_state(reference, port)
    _randomise(reference, port, seed=3)
    x = torch.randn(2, 4, 3, 24, 24)  # (batch, time, channels, H, W)
    reference.eval()
    port.eval()
    with torch.no_grad():
        _assert_close(port(x), reference(x.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3))

    # task="segmentation": the same network with target_shape (1, H, W, 1); logits (batch, 1, H, W).
    kwargs = _official_kwargs("sevir", input_shape=[4, 24, 24, 3], target_shape=[1, 24, 24, 1])
    reference, _ = _build_pair(official, kwargs)
    torch.manual_seed(0)
    port = build_model("earthformer", task="segmentation", history=4, in_channels=3, img_size=24)
    assert isinstance(port, EarthformerSegmenter)
    _assert_same_state(reference, port)
    reference.eval()
    port.eval()
    with torch.no_grad():
        _assert_close(port(x), reference(x.permute(0, 1, 3, 4, 2))[:, 0].permute(0, 3, 1, 2))


@pytest.mark.parametrize("name", sorted(EARTHFORMER_CHECKPOINTS))
def test_official_checkpoints(name, tmp_path, monkeypatch):
    official = _official_module()
    spec = EARTHFORMER_CHECKPOINTS[name]
    checkpoint = oracle_asset(f"earthformer_{name}") / f"earthformer_{name}.pt"
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)

    reference = official.CuboidTransformerModel(**_official_kwargs(spec["config"]))
    reference.load_state_dict(state, strict=True)
    # pretrained="<name>" finds the sha256-verified file in the torch hub cache without downloading.
    hub = tmp_path / "hub"
    (hub / "checkpoints").mkdir(parents=True)
    (hub / "checkpoints" / f"earthformer_{name}.pt").write_bytes(checkpoint.read_bytes())
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(hub))
    assert earthformer_checkpoint_path(name) == hub / "checkpoints" / f"earthformer_{name}.pt"
    port = build_model("earthformer", task="forecasting", config=spec["config"], pretrained=name)
    assert isinstance(port, Earthformer)
    _assert_same_state(reference, port)

    T, H, W, C = reference.input_shape
    x = torch.rand((1, T, H, W, C), generator=torch.Generator().manual_seed(0))
    reference.eval()
    port.eval()
    with torch.no_grad():
        expected = reference(x)
        actual = port(x.permute(0, 1, 4, 2, 3)).permute(0, 1, 3, 4, 2)
    _assert_close(actual, expected)
    assert expected.shape == (1,) + tuple(reference.target_shape)
