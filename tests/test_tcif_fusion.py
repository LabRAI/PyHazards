import numpy as np
import pytest
import torch

from pyhazards.models import build_model
from pyhazards.models.tcif_fusion import (
    TCIFFusion,
    apply_model_knowledge,
    era5_all,
    load_keras_weights,
    model_knowledge_heatmaps,
)

SMALL = {"grid_size": 9, "ir_size": 32, "widths": (4, 8, 8), "vgg_filters": (4, 4, 8, 8, 8), "vgg_fc": 16}


def _n_params(model):
    return sum(p.numel() for p in model.parameters())


def _inputs(batch=2, grid=9, ir=32, all_channels=None, seed=0):
    g = torch.Generator().manual_seed(seed)
    x = {
        "u": torch.randn(batch, grid, grid, 5, 4, generator=g),
        "v": torch.randn(batch, grid, grid, 5, 4, generator=g),
        "w": torch.randn(batch, grid, grid, 5, 4, generator=g),
        "sst": torch.rand(batch, grid, grid, 5, 1, generator=g),
        "his": torch.rand(batch, 30, generator=g),
        "ir": torch.rand(batch, ir, ir, 5, generator=g),
    }
    if all_channels is not None:
        x["all"] = torch.randn(batch, grid, grid, all_channels, generator=g)
    return x


def test_parameter_counts_of_the_paper_and_notebook_graphs():
    with torch.device("meta"):
        paper = build_model("tcif_fusion", task="regression")
        notebook = build_model("tcif_fusion", task="regression", all_channels=85)
    assert _n_params(paper) == 299_588_223
    assert _n_params(notebook) == 299_611_263
    assert paper.era5_flat_dim == 212_992 and paper.all_channels == 65
    names = paper.keras_layer_names()
    assert len(names) == 42 + 25 + 14
    assert names["conv3d_1"] == "u_block1.conv1" and names["conv2d_10"] == "ir.conv1_1" and names["dense_14"] == "output"


def test_forward_shapes_inputs_and_validation():
    model = build_model("tcif_fusion", task="regression", **SMALL).eval()
    x = _inputs()
    with torch.no_grad():
        out = model(x)
        explicit = model({**x, "all": era5_all(x["u"], x["v"], x["w"], x["sst"])})
        positional = model(x["u"], x["v"], x["w"], x["sst"], None, x["his"], x["ir"])
    assert out.shape == (2, 1)
    torch.testing.assert_close(explicit, out)
    torch.testing.assert_close(positional, out)
    assert era5_all(x["u"], x["v"], x["w"], x["sst"]).shape == (2, 9, 9, 65)
    with pytest.raises(ValueError, match="missing"):
        model({k: v for k, v in x.items() if k != "ir"})
    with pytest.raises(ValueError, match="must be shaped"):
        model({**x, "u": x["u"].permute(0, 4, 1, 2, 3)})
    notebook = build_model("tcif_fusion", task="regression", all_channels=85, **SMALL)
    with pytest.raises(ValueError, match="pass inputs\\['all'\\]"):
        notebook(x)
    assert notebook(_inputs(all_channels=85)).shape == (2, 1)
    with pytest.raises(ValueError, match="regression"):
        build_model("tcif_fusion", task="classification")


def test_keras_weights_are_transposed_into_place():
    model = TCIFFusion(**SMALL)
    modules = dict(model.named_modules())
    rng = np.random.default_rng(0)
    weights = {}
    for keras_name, module_name in model.keras_layer_names().items():
        weight = modules[module_name].weight
        if weight.ndim == 5:
            shape = (*weight.shape[2:], weight.shape[1], weight.shape[0])
        elif weight.ndim == 4:
            shape = (*weight.shape[2:], weight.shape[1], weight.shape[0])
        else:
            shape = (weight.shape[1], weight.shape[0])
        weights[keras_name] = [rng.standard_normal(shape).astype("float32"), rng.standard_normal(weight.shape[0]).astype("float32")]
    load_keras_weights(model, weights)
    np.testing.assert_array_equal(model.u_block1.conv1.weight[3, 2].detach().numpy(), weights["conv3d_1"][0][:, :, :, 2, 3])
    np.testing.assert_array_equal(model.ir.conv1_1.weight[1, 0].detach().numpy(), weights["conv2d_10"][0][:, :, 0, 1])
    np.testing.assert_array_equal(model.output.weight.detach().numpy(), weights["dense_14"][0].T)
    with pytest.raises(ValueError, match="missing Keras layers"):
        load_keras_weights(model, {k: v for k, v in weights.items() if k != "dense_3"})


def test_model_knowledge_step():
    torch.manual_seed(0)
    model = build_model("tcif_fusion", task="regression", **SMALL)
    x = _inputs()
    heatmaps = model_knowledge_heatmaps(model, x)
    assert model.training  # restored
    assert heatmaps["u"].shape == (2, 9, 9, 5) and heatmaps["sst"].shape == (2, 9, 9, 5) and heatmaps["ir"].shape == (2, 32, 32)
    guided = apply_model_knowledge(x, heatmaps)
    torch.testing.assert_close(guided["w"], x["w"] * heatmaps["w"].abs().unsqueeze(-1))
    torch.testing.assert_close(guided["ir"], x["ir"] * heatmaps["ir"].abs().unsqueeze(-1))
    torch.testing.assert_close(guided["all"], era5_all(guided["u"], guided["v"], guided["w"], guided["sst"]))
    assert guided["his"] is x["his"]
    # The heatmap of an input is its channel mean weighted by the spatially averaged gradient.
    sst = x["sst"].clone().requires_grad_(True)
    model.eval()
    grad = torch.autograd.grad(model({**x, "sst": sst}).sum(), sst)[0]
    expected = (x["sst"] * grad.mean(dim=(1, 2), keepdim=True)).mean(-1)
    torch.testing.assert_close(heatmaps["sst"], expected)
