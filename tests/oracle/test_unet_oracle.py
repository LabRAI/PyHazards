"""U-Net checked against the authors' released Caffe network (u-net-release-2015-10-02).

The release (no license; fetched at test time only, never vendored) contains the network
definition ``phseg_v5-train.prototxt`` and the trained ``phseg_v5.caffemodel`` of the ISBI 2015
cell-tracking submission. Caffe itself is not needed: the weights are read with a small protobuf
reader, and the reference forward pass runs in OpenCV's Caffe importer (Apache-2.0), an
independent implementation of Caffe's layers. The released network is the ``phseg_v5`` variant
of the PyHazards U-Net; the paper's Fig. 1 network differs from it only where the tests say so.
"""

from __future__ import annotations

import re
import tarfile
from functools import lru_cache

import numpy as np
import torch

from oracle_utils import oracle_asset, oracle_package
from pyhazards.models import build_model
from pyhazards.models.unet import unet_output_size

RELEASE = "u-net-release-2015-10-02.tar.gz"
PHSEG_V5_PARAMETERS = 31_100_354


def _n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


@lru_cache(maxsize=None)
def _release_member(name: str) -> bytes:
    with tarfile.open(oracle_asset("unet_release") / RELEASE) as archive:
        return archive.extractfile(f"u-net-release/{name}").read()


def _prototxt_layers() -> list[dict]:
    """Layers of ``phseg_v5-train.prototxt`` (one V1 layer per line) as dictionaries."""
    layers = []
    for line in _release_member("phseg_v5-train.prototxt").decode().splitlines():
        if not line.startswith("layers"):
            continue
        layer = {
            "name": re.search(r"name: '([^']+)'", line).group(1),
            "type": re.search(r"type: (\w+)", line).group(1),
            "bottom": re.findall(r"bottom: '([^']+)'", line),
            "top": re.findall(r"top: '([^']+)'", line),
            "train_only": "phase: TRAIN" in line,
        }
        for key in ("num_output", "kernel_size", "stride", "pad"):
            match = re.search(rf"{key}: (\d+)", line)
            layer[key] = int(match.group(1)) if match else None
        match = re.search(r"dropout_ratio: ([\d.]+)", line)
        layer["dropout_ratio"] = float(match.group(1)) if match else None
        match = re.search(r"weight_filler \{ type: '(\w+)'", line)
        layer["filler"] = match.group(1) if match else None
        layers.append(layer)
    return layers


# --- minimal protobuf reader for Caffe's V1 NetParameter (layers = 2; name = 4, blobs = 6;
# BlobProto num/channels/height/width = 1-4, packed float data = 5) ---------------------------
def _varint(buf: memoryview, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        byte = buf[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if byte < 0x80:
            return result, pos
        shift += 7


def _fields(buf: memoryview):
    pos = 0
    while pos < len(buf):
        key, pos = _varint(buf, pos)
        field, wire = key >> 3, key & 7
        if wire == 0:
            value, pos = _varint(buf, pos)
        elif wire == 2:
            size, pos = _varint(buf, pos)
            value, pos = buf[pos : pos + size], pos + size
        elif wire == 5:
            value, pos = buf[pos : pos + 4], pos + 4
        elif wire == 1:
            value, pos = buf[pos : pos + 8], pos + 8
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        yield field, wire, value


def _caffemodel_blobs(data: bytes) -> dict[str, list[np.ndarray]]:
    layers: dict[str, list[np.ndarray]] = {}
    for field, _, layer in _fields(memoryview(data)):
        if field != 2:
            continue
        name, blobs = None, []
        for layer_field, _, value in _fields(layer):
            if layer_field == 4:
                name = bytes(value).decode()
            elif layer_field == 6:
                dims, chunks = [0, 0, 0, 0], []
                for blob_field, wire, blob_value in _fields(value):
                    if blob_field in (1, 2, 3, 4):
                        dims[blob_field - 1] = blob_value
                    elif blob_field == 5:
                        chunks.append(np.frombuffer(blob_value, dtype="<f4"))
                blobs.append(np.concatenate(chunks).reshape(dims))
        if blobs:
            layers[name] = blobs
    return layers


def _release_state_dict() -> dict[str, torch.Tensor]:
    """The caffemodel as a state dict: Caffe layer ``conv_d0a-b`` -> module ``conv_d0a_b``."""
    state = {}
    for name, (weight, bias) in _caffemodel_blobs(_release_member("phseg_v5.caffemodel")).items():
        key = name.replace("-", "_")
        state[f"{key}.weight"] = torch.from_numpy(weight.copy())
        state[f"{key}.bias"] = torch.from_numpy(bias.reshape(-1).copy())
    return state


def _deploy_prototxt(height: int, width: int) -> str:
    """Rewrite the release prototxt for OpenCV (current layer syntax, test phase, fixed input size).

    The release's CROP layer aligns its two inputs through the layers' coordinate maps; OpenCV's
    Crop needs explicit offsets, so they are derived here the same way: each blob carries the
    affine map from its coordinates to input coordinates, and the offset is the shift that
    aligns the skip feature with the up-sampled map.
    """
    lines = ['name: "phseg_v5"', 'input: "data"'] + [f"input_dim: {d}" for d in (1, 1, height, width)]
    size = {"data": np.array([height, width])}
    coord = {"data": (1.0, 0.0)}  # input coordinate = scale * blob coordinate + shift
    for layer in _prototxt_layers():
        if layer["train_only"]:
            continue
        name, kind, bottom, top = layer["name"], layer["type"], layer["bottom"], layer["top"][0]
        scale, shift = coord[bottom[0]]
        if kind == "CONVOLUTION":
            k = layer["kernel_size"]
            lines.append(
                f'layer {{ name: "{name}" type: "Convolution" bottom: "{bottom[0]}" top: "{top}" '
                f"convolution_param {{ num_output: {layer['num_output']} pad: 0 kernel_size: {k} }} }}"
            )
            size[top], coord[top] = size[bottom[0]] - (k - 1), (scale, shift + scale * (k - 1) / 2)
        elif kind == "POOLING":
            lines.append(
                f'layer {{ name: "{name}" type: "Pooling" bottom: "{bottom[0]}" top: "{top}" '
                "pooling_param { pool: MAX kernel_size: 2 stride: 2 } }"
            )
            size[top], coord[top] = size[bottom[0]] // 2, (2 * scale, shift + 0.5 * scale)
        elif kind == "DECONVOLUTION":
            lines.append(
                f'layer {{ name: "{name}" type: "Deconvolution" bottom: "{bottom[0]}" top: "{top}" '
                f"convolution_param {{ num_output: {layer['num_output']} pad: 0 kernel_size: 2 stride: 2 }} }}"
            )
            size[top], coord[top] = size[bottom[0]] * 2, (scale / 2, shift - 0.25 * scale)
        elif kind == "RELU":
            lines.append(f'layer {{ name: "{name}" type: "ReLU" bottom: "{bottom[0]}" top: "{top}" }}')
        elif kind == "CROP":
            (skip_scale, skip_shift), (up_scale, up_shift) = coord[bottom[0]], coord[bottom[1]]
            assert skip_scale == up_scale
            offset = (up_shift - skip_shift) / skip_scale
            assert offset == int(offset) and offset >= 0
            assert np.all(size[bottom[0]] - size[bottom[1]] == 2 * offset)  # the crop is centred
            lines.append(
                f'layer {{ name: "{name}" type: "Crop" bottom: "{bottom[0]}" bottom: "{bottom[1]}" top: "{top}" '
                f"crop_param {{ axis: 2 offset: {int(offset)} offset: {int(offset)} }} }}"
            )
            size[top], coord[top] = size[bottom[1]], coord[bottom[1]]
        elif kind == "CONCAT":
            lines.append(
                f'layer {{ name: "{name}" type: "Concat" bottom: "{bottom[0]}" bottom: "{bottom[1]}" top: "{top}" }}'
            )
            size[top], coord[top] = size[bottom[0]], coord[bottom[0]]
        else:
            raise AssertionError(f"unexpected layer type {kind}")
    assert tuple(size["score"]) == (unet_output_size(height), unet_output_size(width))
    return "\n".join(lines) + "\n"


def _opencv_forward(x: np.ndarray) -> np.ndarray:
    cv2 = oracle_package("cv2", "4.10.0", requirements="requirements-unet.txt")
    net = cv2.dnn.readNetFromCaffe(
        np.frombuffer(_deploy_prototxt(*x.shape[-2:]).encode(), dtype=np.uint8),
        np.frombuffer(_release_member("phseg_v5.caffemodel"), dtype=np.uint8),
    )
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.enableWinograd(False)
    net.setInput(x)
    return net.forward()


def _assert_close_to_opencv(actual: torch.Tensor, expected: np.ndarray) -> None:
    # OpenCV and PyTorch use different float32 convolution kernels (summation order) through 23
    # layers; the outputs reach about 10 in magnitude and differ by about 1e-5, so the absolute
    # tolerance is 1e-4 instead of 1e-6.
    torch.testing.assert_close(actual, torch.from_numpy(expected), rtol=1e-5, atol=1e-4)


def test_phseg_v5_variant_has_the_released_layers():
    model = build_model("unet", task="segmentation", variant="phseg_v5")
    modules = dict(model.named_modules())
    layers = _prototxt_layers()
    convs = [layer for layer in layers if layer["type"] in ("CONVOLUTION", "DECONVOLUTION")]
    assert len(convs) == 23  # "In total the network has 23 convolutional layers" (paper Sec. 2)
    for layer in convs:
        module = modules[layer["name"].replace("-", "_")]
        transposed = layer["type"] == "DECONVOLUTION"
        assert isinstance(module, torch.nn.ConvTranspose2d if transposed else torch.nn.Conv2d), layer["name"]
        assert module.out_channels == layer["num_output"], layer["name"]
        assert module.kernel_size == (layer["kernel_size"],) * 2, layer["name"]
        assert module.stride == ((layer["stride"] or 1),) * 2 and module.padding == (layer["pad"],) * 2
        assert layer["filler"] == "xavier"  # Olaf's filler: N(0, 2 / fan_in), see the next test
    # Up-sampled map first in every concatenation, then the centre-cropped skip feature.
    concats = [layer for layer in layers if layer["type"] == "CONCAT"]
    assert [layer["bottom"] for layer in concats] == [["u3a", "d3cc"], ["u2a", "d2cc"], ["u1a", "d1cc"], ["u0a", "d0cc"]]
    # A ReLU follows every up-convolution in the release (not in the paper's Fig. 1).
    relus = {layer["top"][0] for layer in layers if layer["type"] == "RELU"}
    assert {"u3a", "u2a", "u1a", "u0a"} <= relus
    # Two in-place dropout layers (ratio 0.5, training only) on the 512- and 1024-channel levels;
    # d3c is dropped out before it feeds both the pooling and the crop for the skip connection.
    dropouts = [layer for layer in layers if layer["type"] == "DROPOUT"]
    assert [(d["name"], d["bottom"], d["top"], d["dropout_ratio"], d["train_only"]) for d in dropouts] == [
        ("dropout_d3c", ["d3c"], ["d3c"], 0.5, True),
        ("dropout_d4c", ["d4c"], ["d4c"], 0.5, True),
    ]
    assert model.dropout_d3c.p == model.dropout_d4c.p == 0.5
    names = [layer["name"] for layer in layers]
    assert names.index("dropout_d3c") < names.index("pool_d3c-4a") < names.index("crop_d3c-d3cc")

    paper = build_model("unet", task="segmentation")
    assert _n_params(paper) == 31_030_658
    assert _n_params(model) == PHSEG_V5_PARAMETERS
    # The paper's last up-convolution halves the channels to 64; the release keeps 128.
    assert paper.upconv_u1d_u0a.out_channels == 64 and model.upconv_u1d_u0a.out_channels == 128


def test_initialisation_follows_the_release_filler():
    # The release patches Caffe's "xavier" filler into a Gaussian with std sqrt(2 / fan_in), where
    # fan_in = count / num of the weight blob (the paper's "sqrt(2/N)", Sec. 3).
    filler = _release_member("caffe-unet-src/include/caffe/filler.hpp").decode()
    assert "Dtype scale = sqrt(Dtype(2) / fan_in);" in filler
    assert "int fan_in = blob->count() / blob->num();" in filler
    torch.manual_seed(0)
    model = build_model("unet", task="segmentation", variant="phseg_v5")
    for name, param in model.named_parameters():
        if name.endswith(".bias"):
            assert torch.count_nonzero(param) == 0, name
            continue
        # Caffe blobs: (out, in, k, k) for convolutions, (in, out, k, k) for deconvolutions, which are
        # also PyTorch's Conv2d / ConvTranspose2d layouts, so fan_in = numel / size(0) for both.
        expected = (2.0 / (param.numel() / param.size(0))) ** 0.5
        if param.numel() >= 100_000:
            assert abs(param.std().item() / expected - 1) < 0.02, name
            assert abs(param.mean().item()) < 0.02 * expected, name


def test_released_weights_load_strict_and_match_opencv_caffe():
    model = build_model("unet", task="segmentation", variant="phseg_v5").eval()
    state = _release_state_dict()
    assert list(state) == list(model.state_dict())
    model.load_state_dict(state, strict=True)
    assert _n_params(model) == PHSEG_V5_PARAMETERS

    rng = np.random.RandomState(0)
    x = rng.rand(1, 1, 572, 572).astype(np.float32)  # the paper's Fig. 1 tile
    with torch.no_grad():
        out = model(torch.from_numpy(x))
    assert out.shape == (1, 2, 388, 388)
    _assert_close_to_opencv(out, _opencv_forward(x))


def test_released_weights_on_sample_image_tile_match_opencv_caffe():
    cv2 = oracle_package("cv2", "4.10.0", requirements="requirements-unet.txt")
    image = cv2.imdecode(
        np.frombuffer(_release_member("PhC-C2DH-U373/01/t000.tif"), dtype=np.uint8), cv2.IMREAD_UNCHANGED
    )
    assert image.shape == (520, 696) and image.dtype == np.uint8
    # First of the two tiles that the release's segmentAndTrack script feeds the network for these
    # images (tmp-test.prototxt: 444 x 892): im2double, mirror padding of (input - output) / 2,
    # zero fill beyond the padded image.
    height, width = 444, 892
    border = (height - unet_output_size(height)) // 2
    padded = np.pad(image.astype(np.float32) / 255.0, border, mode="reflect")
    tile = np.zeros((1, 1, height, width), dtype=np.float32)
    rows, cols = min(height, padded.shape[0]), min(width, padded.shape[1])
    tile[0, 0, :rows, :cols] = padded[:rows, :cols]

    model = build_model("unet", task="segmentation", variant="phseg_v5").eval()
    model.load_state_dict(_release_state_dict(), strict=True)
    with torch.no_grad():
        out = model(torch.from_numpy(tile))
    assert out.shape == (1, 2, 260, 708)
    reference = _opencv_forward(tile)
    _assert_close_to_opencv(out, reference)
    # Same foreground/background decision for every pixel of the tile.
    assert torch.equal(out.argmax(1), torch.from_numpy(reference).argmax(1))
