"""TCIF-fusion checked against the official notebooks (wangchong96/TCIF-fusion, pinned).

The notebooks have no licence and cannot run as published (Keras 2 / TensorFlow 1 session code, no data
loading); pyhazards/models/tcif_fusion.py is written from the paper and the notebook's layer list. Here:

* the model summary printed in ``obtain model knowledge.ipynb`` by the authors' own run (Keras 2,
  TensorFlow 1.14) gives every layer's parameter count and the total, 299,611,263;
* the notebook's ``F3d``, ``F2d``, ``VGG192d`` and ``TCIF_fusion`` functions are executed unchanged from the
  pinned ``TCIF-fusion.ipynb`` with Keras 3 on the PyTorch backend (only the ALL input's channel count is
  substituted for the paper's 65-channel variant); the Keras weights are copied into the port and the
  outputs compared;
* the training settings of the notebook's ``compile`` / ``fit`` call and the first step of its
  model-knowledge example (gradient of the forecast w.r.t. the SST input, averaged over the batch and the
  two horizontal axes).
"""

from __future__ import annotations

import ast
import json
import os
import re
from pathlib import Path

import numpy as np
import pytest
import torch

from oracle_utils import oracle_package, oracle_repo
from pyhazards.models import build_model
from pyhazards.models.tcif_fusion import TCIF_TRAINING, load_keras_weights, model_knowledge_heatmaps

os.environ.setdefault("KERAS_BACKEND", "torch")


def _cells(repo: Path, name: str):
    notebook = json.loads((repo / name).read_text(encoding="utf-8"))
    return [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]


def _keras():
    keras = oracle_package("keras", "3.12.4", "requirements-tc.txt")
    if keras.backend.backend() != "torch":
        pytest.skip("set KERAS_BACKEND=torch before keras is imported")
    return keras


def _official_graph(all_channels: int = 85):
    """The notebook's TCIF_fusion() built with Keras 3; returns the model and its weights by Keras-2 name."""
    keras = _keras()
    from keras.layers import Concatenate, Dense, Flatten, Input, add
    from keras.models import Model

    repo = oracle_repo("TCIF-fusion")
    source = "\n".join("".join(cell["source"]) for cell in _cells(repo, "TCIF-fusion.ipynb"))
    if all_channels != 85:
        assert source.count("shape = (25, 25, 85)") == 1
        source = source.replace("shape = (25, 25, 85)", f"shape = (25, 25, {all_channels})")
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {"F3d", "F2d", "VGG192d", "TCIF_fusion"}]
    assert len(nodes) == 4
    scope = {"keras": keras, "Concatenate": Concatenate, "Dense": Dense, "Flatten": Flatten, "Input": Input, "Model": Model, "add": add}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "TCIF-fusion.ipynb", "exec"), scope)
    keras.backend.clear_session()
    keras.utils.set_random_seed(0)
    model = scope["TCIF_fusion"]()
    # Keras names layers <kind>, <kind>_1, ... in creation order; the notebook's Keras 2 named them <kind>_1, _2, ...
    groups = {}
    for layer in model.layers:
        if layer.weights:
            kind, _, suffix = layer.name.partition("_")
            index = int(suffix) if suffix.isdigit() else 0
            groups.setdefault(kind, []).append((index, layer))
    weights = {}
    for kind, items in groups.items():
        for number, (_, layer) in enumerate(sorted(items, key=lambda item: item[0]), start=1):
            weights[f"{kind}_{number}"] = [np.asarray(w) for w in layer.get_weights()]
    return model, weights


def _inputs(all_channels: int, batch: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    arrays = [rng.standard_normal((batch, 25, 25, 5, 4)).astype("float32") for _ in range(3)]
    arrays += [
        rng.standard_normal((batch, 25, 25, 5, 1)).astype("float32"),
        rng.standard_normal((batch, 25, 25, all_channels)).astype("float32"),
        rng.standard_normal((batch, 30)).astype("float32"),
        rng.random((batch, 224, 224, 5)).astype("float32"),
    ]
    return arrays  # notebook input order: u, v, w, sst, h (ALL), left (history), sat (IR)


def test_parameters_match_the_authors_printed_summary():
    repo = oracle_repo("TCIF-fusion")
    cell = next(cell for cell in _cells(repo, "obtain model knowledge.ipynb") if "def TCIF_fusion" in "".join(cell["source"]))
    text = "".join("".join(output.get("text", [])) for output in cell["outputs"])
    lines = text.splitlines()
    header = next(line for line in lines if line.startswith("Layer (type)"))
    shape_col, param_col, link_col = header.index("Output Shape"), header.index("Param #"), header.index("Connected to")
    printed = {}
    for line in lines:
        match = re.match(r"^(\w+) \((\w+)\)", line)
        if match and line != header and len(line) > param_col:
            printed[match.group(1)] = (match.group(2), line[shape_col:param_col].strip(), int(line[param_col:link_col].strip()))
    total = int(re.search(r"Total params: ([\d,]+)", text).group(1).replace(",", ""))
    assert total == 299_611_263
    port = build_model("tcif_fusion", task="regression", all_channels=85)
    assert sum(p.numel() for p in port.parameters()) == total
    assert printed["input_5"][1] == "(None, 25, 25, 85)"  # the ALL input of the notebook's graph
    modules = dict(port.named_modules())
    weighted = {name: entry for name, entry in printed.items() if entry[2] > 0}
    assert set(weighted) == set(port.keras_layer_names())
    for keras_name, module_name in port.keras_layer_names().items():
        assert sum(p.numel() for p in modules[module_name].parameters()) == weighted[keras_name][2], keras_name
    # The paper's variant: 65 ALL channels (Figure 1), 23,040 fewer weights in the first ALL convolution.
    assert sum(p.numel() for p in build_model("tcif_fusion", task="regression").parameters()) == 299_588_223


@pytest.mark.parametrize("all_channels", [85, 65])
def test_outputs_match_the_notebook_graph(all_channels):
    reference, weights = _official_graph(all_channels)
    port = build_model("tcif_fusion", task="regression", all_channels=all_channels).eval()
    assert reference.count_params() == sum(p.numel() for p in port.parameters())
    load_keras_weights(port, weights)
    arrays = _inputs(all_channels)
    expected = np.asarray(reference.predict(arrays, verbose=0))
    tensors = [torch.as_tensor(a) for a in arrays]
    with torch.no_grad():
        positional = port(*tensors).numpy()
        named = port(dict(zip(("u", "v", "w", "sst", "all", "his", "ir"), tensors))).numpy()
    np.testing.assert_allclose(positional, expected, rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(named, positional)


def test_initialisation_follows_keras_defaults():
    """Glorot-uniform kernels and zero biases like the Keras layers (the random draws differ)."""
    torch.manual_seed(0)
    port = build_model("tcif_fusion", task="regression")
    modules = dict(port.named_modules())
    for module_name in ("u_block1.conv1", "fusion_block3.conv3", "all_block1.conv1", "ir.conv5_4", "era5_dense1", "res2_dense3"):
        layer = modules[module_name]
        receptive = layer.weight[0, 0].numel()
        limit = np.sqrt(6.0 / (layer.weight.shape[1] * receptive + layer.weight.shape[0] * receptive))
        weight = layer.weight.detach()
        assert float(weight.abs().max()) <= limit + 1e-7
        assert float(weight.std()) == pytest.approx(limit / np.sqrt(3.0), rel=0.05)
        assert not torch.any(layer.bias)


def test_training_settings_are_the_notebooks():
    repo = oracle_repo("TCIF-fusion")
    source = "".join(next(cell for cell in _cells(repo, "TCIF-fusion.ipynb") if ".fit(" in "".join(cell["source"]))["source"])
    calls = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call):
            name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)
            calls.setdefault(name, {}).update({kw.arg: kw.value for kw in node.keywords})
    literal = lambda name, key: ast.literal_eval(calls[name][key])  # noqa: E731
    assert literal("Adam", "lr") == TCIF_TRAINING["learning_rate"]
    assert literal("compile", "loss") == TCIF_TRAINING["loss"]
    assert literal("fit", "batch_size") == TCIF_TRAINING["batch_size"]
    assert literal("fit", "epochs") == TCIF_TRAINING["epochs"]
    plateau = TCIF_TRAINING["reduce_lr_on_plateau"]
    assert (literal("ReduceLROnPlateau", "factor"), literal("ReduceLROnPlateau", "patience"), literal("ReduceLROnPlateau", "min_lr")) == (plateau["factor"], plateau["patience"], plateau["min_lr"])


def test_model_knowledge_gradients_follow_the_notebook():
    """The MK example takes d(forecast)/d(SST input) of one sample and averages it over axes (0, 1, 2)."""
    reference, weights = _official_graph(85)
    port = build_model("tcif_fusion", task="regression", all_channels=85).eval()
    load_keras_weights(port, weights)
    arrays = _inputs(85, batch=1, seed=3)
    tensors = [torch.as_tensor(a) for a in arrays]
    sst = tensors[3].clone().requires_grad_(True)
    output = reference([*tensors[:3], sst, *tensors[4:]], training=False)
    grads = torch.autograd.grad(output[:, 0].sum(), sst)[0]
    pooled = grads.mean(dim=(0, 1, 2))  # K.mean(grads, axis=(0, 1, 2)) of the notebook: (time, channel)
    expected = (tensors[3][0] * pooled).mean(dim=-1)  # Grad-CAM: weighted channels, averaged
    heatmaps = model_knowledge_heatmaps(port, dict(zip(("u", "v", "w", "sst", "all", "his", "ir"), tensors)), names=("sst",))
    torch.testing.assert_close(heatmaps["sst"][0], expected.detach(), rtol=1e-4, atol=1e-9)
