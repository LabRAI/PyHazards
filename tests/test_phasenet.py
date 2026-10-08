import math

import numpy as np
import pytest
import torch

from pyhazards.models import build_model
from pyhazards.models.phasenet import PhaseNet, _tf_same_padding, state_dict_from_tf_checkpoint


def _n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_released_configuration_has_268443_parameters():
    model = build_model("phasenet", task="picking")
    assert isinstance(model, PhaseNet)
    assert _n_params(model) == 268_443
    widths = [getattr(model, f"DownConv_{d}").__getattr__(f"down_conv1_{d + 1}").out_channels for d in range(5)]
    assert widths == [8, 16, 32, 64, 128]


@pytest.mark.parametrize("length", [3000, 3001, 400, 7])
def test_any_length_gives_per_sample_probabilities(length):
    model = PhaseNet().eval()
    with torch.no_grad():
        probabilities = model(torch.randn(2, 3, length))
        logits = model(torch.randn(2, 3, length), logits=True)
    assert probabilities.shape == logits.shape == (2, 3, length)
    torch.testing.assert_close(probabilities.sum(dim=1), torch.ones(2, length))


def test_tensorflow_same_padding_is_asymmetric_for_strided_convolutions():
    assert _tf_same_padding(3000, 7, 4) == (1, 2)
    assert _tf_same_padding(750, 7, 4) == (2, 3)
    assert _tf_same_padding(47, 7, 4) == (2, 2)
    assert _tf_same_padding(100, 7, 1) == (3, 3)


def test_input_validation_and_builder_arguments():
    model = PhaseNet()
    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(3, 3000))
    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(1, 2, 3000))
    with pytest.raises(ValueError, match="picking"):
        build_model("phasenet", task="regression")
    with pytest.raises(TypeError, match="hidden_dim"):
        build_model("phasenet", task="picking", hidden_dim=32)
    with pytest.raises(ValueError, match="default configuration"):
        build_model("phasenet", task="picking", filters_root=4, pretrained="original")


def test_tensorflow_variables_map_onto_the_state_dict():
    model = PhaseNet()
    variables = {}
    for name, tensor in model.state_dict().items():
        if name.endswith("num_batches_tracked"):
            continue
        scope, layer, kind = name.split(".")
        array = tensor.numpy()
        if kind == "weight" and "conv" in layer:
            array = array.transpose(2, 1, 0)[:, None]  # back to the TF kernel layout (k, 1, ., .)
            tf_kind = "kernel"
        else:
            tf_kind = {"weight": "gamma", "bias": "beta" if "_bn" in layer else "bias",
                       "running_mean": "moving_mean", "running_var": "moving_variance"}[kind]
        variables[f"{scope}/{layer}/{tf_kind}"] = array + 1.0
    variables["global_step"] = np.array(3)
    variables["Input/input_conv/kernel/Adam"] = np.zeros((7, 1, 3, 8), dtype=np.float32)
    converted = state_dict_from_tf_checkpoint(variables)
    restored = PhaseNet()
    restored.load_state_dict(converted, strict=True)
    for name, tensor in model.state_dict().items():
        if not name.endswith("num_batches_tracked"):
            torch.testing.assert_close(restored.state_dict()[name], tensor + 1.0)


def test_annotate_normalises_and_extract_picks_finds_peaks():
    model = PhaseNet().eval()
    raw = torch.randn(2, 3, 3000) * 1e4 + 5.0
    with torch.no_grad():
        scaled = model.annotate(raw)
        reference = model.annotate(raw * 3.0 - 2.0)  # per-channel standardisation removes scale/offset
    torch.testing.assert_close(scaled, reference, rtol=1e-4, atol=1e-5)

    annotations = torch.zeros(1, 3, 1000)
    annotations[0, 1, 300] = 0.9
    annotations[0, 1, 320] = 0.7  # within 0.5 s of a higher peak: removed
    annotations[0, 1, 600] = 0.4  # below 0.5
    annotations[0, 2, 800] = 0.6
    picks = model.extract_picks(annotations)
    assert [i for i, _ in picks[0]["P"]] == [300.0]
    assert [i for i, _ in picks[0]["S"]] == [800.0]
    assert math.isclose(picks[0]["P"][0][1], 0.9, rel_tol=1e-6)
    picks = model.extract_picks(annotations, p_threshold=0.3, min_distance=10)
    assert [i for i, _ in picks[0]["P"]] == [300.0, 320.0, 600.0]
