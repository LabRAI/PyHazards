"""EQTransformer port: shapes, parameter counts, building blocks and pick extraction (CPU, offline).

The comparison with the official Keras models lives in tests/oracle/test_eqtransformer_oracle.py.
"""

from __future__ import annotations

import h5py
import numpy as np
import pytest
import torch

from pyhazards.models import build_model
from pyhazards.models.eqtransformer import (
    EQTransformer,
    KerasBatchNorm1d,
    KerasLSTM,
    SeqSelfAttention,
    eqtransformer_builder,
    keras_weight_map,
    load_eqtransformer_keras_weights,
)


def _n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@pytest.mark.parametrize("variant, expected", [("original", 371_639), ("conservative", 376_423)])
def test_parameter_counts_match_the_released_keras_models(variant, expected):
    model = build_model("eqtransformer", task="picking", variant=variant)
    assert _n_params(model) == expected
    assert model.lstm_blocks == (2 if variant == "original" else 3)


def test_forward_returns_detection_p_and_s_probabilities():
    torch.manual_seed(0)
    model = build_model("eqtransformer", task="picking").eval()
    x = torch.randn(2, 3, 6000)
    with torch.no_grad():
        probabilities = model(x)
        logits = model(x, logits=True)
        annotated = model.annotate(x * 1e4 + 3.0)
    assert probabilities.shape == (2, 3, 6000)
    assert torch.all((probabilities >= 0) & (probabilities <= 1))
    torch.testing.assert_close(torch.sigmoid(logits), probabilities)
    # annotate() normalises each channel (demean, divide by its standard deviation) first.
    normalised = (x - x.mean(-1, keepdim=True)) / x.std(-1, unbiased=False, keepdim=True)
    with torch.no_grad():
        torch.testing.assert_close(annotated, model(normalised), rtol=1e-4, atol=1e-5)
    assert model.output_names == ("detection", "P", "S")
    assert model.component_order == "ENZ" and model.sampling_rate == 100.0


@pytest.mark.parametrize("shape", [(2, 3, 3000), (2, 2, 6000), (3, 6000), (1, 3, 6001)])
def test_invalid_input_shapes_raise(shape):
    model = build_model("eqtransformer", task="picking")
    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(*shape))


def test_builder_validation():
    with pytest.raises(ValueError, match="picking"):
        build_model("eqtransformer", task="regression")
    with pytest.raises(TypeError, match="hidden_dim"):
        build_model("eqtransformer", task="picking", hidden_dim=48)
    with pytest.raises(ValueError, match="variant"):
        build_model("eqtransformer", task="picking", variant="large")
    with pytest.raises(ValueError, match="variant"):
        eqtransformer_builder("picking", variant="original", pretrained="conservative")
    with pytest.raises(ValueError, match="in_channels=3"):
        eqtransformer_builder("picking", in_channels=4, pretrained=True)


def test_eval_is_deterministic_unless_mc_dropout():
    torch.manual_seed(0)
    model = build_model("eqtransformer", task="picking").eval()
    x = torch.randn(1, 3, 6000)
    with torch.no_grad():
        assert torch.equal(model(x), model(x))
        model.mc_dropout = True
        assert not torch.equal(model(x), model(x))
        model.mc_dropout = False
        averaged = model.annotate(x, mc_samples=3)
    assert averaged.shape == (1, 3, 6000) and model.mc_dropout is False


def test_keras_lstm_equals_torch_lstm_with_one_bias():
    torch.manual_seed(0)
    lstm = KerasLSTM(5, 4)
    reference = torch.nn.LSTM(5, 4, batch_first=True)
    with torch.no_grad():  # PyTorch's gate order (i, f, g, o) is Keras' (i, f, c, o)
        reference.weight_ih_l0.copy_(lstm.kernel.T)
        reference.weight_hh_l0.copy_(lstm.recurrent_kernel.T)
        reference.bias_ih_l0.copy_(lstm.bias)
        reference.bias_hh_l0.zero_()
    x = torch.randn(3, 7, 5)
    torch.testing.assert_close(lstm(x), reference(x)[0])
    torch.testing.assert_close(lstm(x, reverse=True), reference(x.flip(1))[0].flip(1))
    assert torch.equal(lstm.bias[4:8], torch.ones(4))  # unit forget bias
    hard = KerasLSTM(5, 4, recurrent_activation="hard_sigmoid")
    hard.load_state_dict(lstm.state_dict())
    assert not torch.allclose(hard(x), lstm(x))


def test_local_attention_only_sees_neighbours():
    torch.manual_seed(0)
    attention = SeqSelfAttention(4, 8, attention_width=3)
    x = torch.randn(1, 10, 4)
    moved = x.clone()
    moved[:, 7:] += 5.0
    out, weights = attention(x, return_attention=True)
    torch.testing.assert_close(attention(moved)[:, :6], out[:, :6])
    assert torch.all(weights[0].triu(2) == 0) and torch.all(weights[0].tril(-2) == 0)


@pytest.mark.parametrize("zero_debias", [False, True])
def test_keras_batch_norm_running_statistics(zero_debias):
    torch.manual_seed(0)
    bn = KerasBatchNorm1d(3, zero_debias=zero_debias)
    batches = [torch.randn(4, 3, 10) * (i + 1) + i for i in range(3)]
    mean, var = np.zeros(3), np.ones(3)
    biased_mean, biased_var = np.zeros(3), np.zeros(3)
    bn.train()
    for step, x in enumerate(batches, start=1):
        out = bn(x)
        values = x.double().numpy()
        n = values.shape[0] * values.shape[2]
        batch_mean = values.mean(axis=(0, 2))
        batch_var = values.var(axis=(0, 2)) * n / (n - (1 + 1e-3))
        if zero_debias:  # TensorFlow assign_moving_average(..., zero_debias=True)
            biased_mean = 0.99 * biased_mean + 0.01 * batch_mean
            biased_var = 0.99 * biased_var + 0.01 * batch_var
            mean, var = biased_mean / (1 - 0.99**step), biased_var / (1 - 0.99**step)
        else:  # Keras 2.3 K.moving_average_update
            mean, var = mean - (mean - batch_mean) * 0.01, var - (var - batch_var) * 0.01
        expected = (values - values.mean(axis=(0, 2), keepdims=True)) / np.sqrt(values.var(axis=(0, 2), keepdims=True) + 1e-3)
        np.testing.assert_allclose(out.detach().numpy(), expected, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(bn.running_mean.numpy(), mean, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(bn.running_var.numpy(), var, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("variant", ["original", "conservative"])
def test_keras_h5_round_trip(tmp_path, variant):
    """Write a model's tensors in the released files' Keras layout and load them back strictly."""
    torch.manual_seed(1)
    source = EQTransformer(variant=variant)
    with torch.no_grad():
        for buffer in source.buffers():
            if buffer.dtype.is_floating_point:
                buffer.uniform_(0.5, 1.5)
    state = source.state_dict()
    path = tmp_path / "model.h5"
    layers = {}
    for key, (layer, name) in keras_weight_map(source.lstm_blocks).items():
        value = state[key].numpy()
        if name == "kernel" and value.ndim == 3:
            value = value.transpose(2, 1, 0)
        layers.setdefault(layer, []).append((f"{layer}/{name}:0", value))
    with h5py.File(path, "w") as handle:
        group = handle.create_group("model_weights")
        group.attrs["layer_names"] = [name.encode() for name in layers]
        for layer, weights in layers.items():
            sub = group.create_group(layer)
            sub.attrs["weight_names"] = [name.encode() for name, _ in weights]
            for name, value in weights:
                sub.create_dataset(name, data=value)
    target = load_eqtransformer_keras_weights(EQTransformer(variant=variant), path)
    for key, value in target.state_dict().items():
        if not key.endswith("num_batches_tracked"):
            assert torch.equal(value, state[key]), key
    other = "conservative" if variant == "original" else "original"
    with pytest.raises((KeyError, ValueError)):
        load_eqtransformer_keras_weights(EQTransformer(variant=other), path)


def test_extract_picks_follows_the_official_picker():
    model = EQTransformer()
    annotations = torch.zeros(2, 3, 6000)
    # Trace 0: one event 1000-2000 with P peaks at 990 (0.4) and 1100 (0.8), S peaks at 1500 and 1700.
    annotations[0, 0, 1000:2001] = 0.9
    for sample, value in ((990, 0.4), (1100, 0.8), (1600, 0.95)):
        annotations[0, 1, sample] = value
    for sample, value in ((1500, 0.3), (1700, 0.9)):
        annotations[0, 2, sample] = value
    # Trace 1: a detection of only 5 samples (ignored) and a P peak without a detection.
    annotations[1, 0, 3000:3005] = 0.9
    annotations[1, 1, 3002] = 0.9
    picks = model.extract_picks(annotations)
    # The earliest S inside the window, then the most probable P between on - 100 and S - 10.
    assert picks[0] == {"P": [(1100.0, 0.8)], "S": [(1500.0, 0.3)]}
    assert picks[1] == {"P": [], "S": []}
    detections = model.extract_detections(annotations)
    assert detections[0] == [(1000, 2000, 0.9)] and detections[1] == []
    stricter = model.extract_picks(annotations, p_threshold=0.85, s_threshold=0.5)
    assert stricter[0] == {"P": [(1600.0, 0.95)], "S": [(1700.0, 0.9)]}
