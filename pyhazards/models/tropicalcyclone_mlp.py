"""24-hour tropical cyclone intensity-change MLP of Xu et al. (2021).

Paper: W. Xu, K. Balaguru, A. August, N. Lalo, N. Hodas, M. DeMaria and D. Judi, "Deep Learning
Experiments for Tropical Cyclone Intensity Forecasts", Weather and Forecasting 36(4), 1453-1470
(2021), doi:10.1175/WAF-D-20-0104.1.

Port of ``models.mlp`` and of the predictor list ``utils.hand_features`` from the official
repository wenweixu/tropicalcyclone_MLP (``models.py`` / ``utils.py`` at
73800596fc824c6d3f150fc41af0bdb7b9c3a7dc), BSD 2-Clause License, Copyright (c) 2021, Wenwei Xu.
The BSD notice is reproduced in ``LICENSE_NOTICE`` below as the license requires.

The official network is a Keras model: 121 SHIPS predictors -> Dense(2048, sigmoid) ->
Dense(2048, relu) -> Dense(1, linear), trained with the MAE loss to predict the 24-hour change of
the maximum sustained wind (``dvs24``, knots). Keras ``Dense`` computes ``x @ kernel + bias`` with
glorot-uniform kernels and zero biases; :class:`TropicalCycloneMLP` uses ``nn.Linear`` (whose
weight is the transposed kernel) with the same initialisation distribution, and
:func:`load_keras_weights` copies the weights of a Keras model (``model.get_weights()``) into it.
"""

from __future__ import annotations

from typing import Sequence, Union

import numpy as np
import torch
import torch.nn as nn

LICENSE_NOTICE = """\
BSD 2-Clause License

Copyright (c) 2021, Wenwei Xu
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

# The 121 predictors of the 24-hour model, in the official column order (utils.hand_features):
# initial intensity vs0, 18 GOES/PSLV, 21 MTPW and 20 IR00 predictors, 0-24 h time averages of
# the SHIPS environmental predictors (``*_t24``) and the 12-hour intensity change DELV-12.
SHIPS_PREDICTORS = (
    "vs0", "PSLV_v2", "PSLV_v3", "PSLV_v4", "PSLV_v5", "PSLV_v6", "PSLV_v7",
    "PSLV_v8", "PSLV_v9", "PSLV_v10", "PSLV_v11", "PSLV_v12", "PSLV_v13",
    "PSLV_v14", "PSLV_v15", "PSLV_v16", "PSLV_v17", "PSLV_v18", "PSLV_v19",
    "MTPW_v2", "MTPW_v3", "MTPW_v4", "MTPW_v5", "MTPW_v6", "MTPW_v7",
    "MTPW_v8", "MTPW_v9", "MTPW_v10", "MTPW_v11", "MTPW_v12", "MTPW_v13",
    "MTPW_v14", "MTPW_v15", "MTPW_v16", "MTPW_v17", "MTPW_v18", "MTPW_v19",
    "MTPW_v20", "MTPW_v21", "MTPW_v22", "IR00_v2", "IR00_v3", "IR00_v4",
    "IR00_v5", "IR00_v6", "IR00_v7", "IR00_v8", "IR00_v9", "IR00_v10",
    "IR00_v11", "IR00_v12", "IR00_v13", "IR00_v14", "IR00_v15", "IR00_v16",
    "IR00_v17", "IR00_v18", "IR00_v19", "IR00_v20", "IR00_v21", "CSST_t24",
    "CD20_t24", "CD26_t24", "COHC_t24", "DTL_t24", "RSST_t24", "U200_t24",
    "U20C_t24", "V20C_t24", "E000_t24", "EPOS_t24", "ENEG_t24", "EPSS_t24",
    "ENSS_t24", "RHLO_t24", "RHMD_t24", "RHHI_t24", "Z850_t24", "D200_t24",
    "REFC_t24", "PEFC_t24", "T000_t24", "R000_t24", "Z000_t24", "TLAT_t24",
    "TLON_t24", "TWAC_t24", "TWXC_t24", "G150_t24", "G200_t24", "G250_t24",
    "V000_t24", "V850_t24", "V500_t24", "V300_t24", "TGRD_t24", "TADV_t24",
    "PENC_t24", "SHDC_t24", "SDDC_t24", "SHGC_t24", "DIVC_t24", "T150_t24",
    "T200_t24", "T250_t24", "SHRD_t24", "SHTD_t24", "SHRS_t24", "SHTS_t24",
    "SHRG_t24", "PENV_t24", "VMPI_t24", "VVAV_t24", "VMFX_t24", "VVAC_t24",
    "HE07_t24", "HE05_t24", "O500_t24", "O700_t24", "CFLX_t24", "DELV-12",
)

_ACTIVATIONS = {
    "sigmoid": nn.Sigmoid,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "linear": nn.Identity,
}


def _layer_name(index: int) -> str:
    # Keras names consecutive Dense layers dense, dense_1, dense_2, ...
    return "dense" if index == 0 else f"dense_{index}"


class TropicalCycloneMLP(nn.Module):
    """Fully connected regressor of the 24-hour intensity change from SHIPS predictors.

    Input ``(batch, input_dim)`` standardised predictors (default: the 121 of
    :data:`SHIPS_PREDICTORS`, in that order); output ``(batch, 1)``, the predicted 24-hour change
    of the maximum sustained wind in the target's units (knots in the paper).
    """

    def __init__(
        self,
        input_dim: int = len(SHIPS_PREDICTORS),
        hidden_dims: Sequence[int] = (2048, 2048),
        activations: Union[str, Sequence[str]] = ("sigmoid", "relu"),
    ):
        super().__init__()
        hidden_dims = [int(width) for width in hidden_dims]
        if isinstance(activations, str):
            activations = [activations] * len(hidden_dims)
        activations = [name.lower() for name in activations]
        if len(activations) != len(hidden_dims):
            raise ValueError("activations needs one entry per hidden layer")
        unknown = sorted(set(activations) - set(_ACTIVATIONS))
        if unknown:
            raise ValueError(f"unknown activation(s) {unknown}; choose from {sorted(_ACTIVATIONS)}")
        if int(input_dim) < 1 or any(width < 1 for width in hidden_dims):
            raise ValueError("input_dim and hidden_dims must be positive")
        self.input_dim = int(input_dim)
        self.hidden_dims = tuple(hidden_dims)
        self.activation_names = tuple(activations)

        widths = [self.input_dim, *hidden_dims, 1]
        self.layer_names = []
        for index, (width_in, width_out) in enumerate(zip(widths[:-1], widths[1:])):
            linear = nn.Linear(width_in, width_out)
            # Keras Dense defaults: glorot_uniform kernel, zero bias.
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            name = _layer_name(index)
            self.add_module(name, linear)
            self.layer_names.append(name)
        self.activations = nn.ModuleList([_ACTIVATIONS[name]() for name in activations])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2 or x.size(1) != self.input_dim:
            raise ValueError(
                f"TropicalCycloneMLP expects predictors shaped (batch, {self.input_dim}), got {tuple(x.shape)}"
            )
        hidden = x
        for name, activation in zip(self.layer_names[:-1], self.activations):
            hidden = activation(getattr(self, name)(hidden))
        return getattr(self, self.layer_names[-1])(hidden)


def load_keras_weights(model: TropicalCycloneMLP, weights: Sequence[np.ndarray]) -> TropicalCycloneMLP:
    """Copy Keras ``Model.get_weights()`` (kernel, bias per Dense layer) into ``model`` in place."""
    weights = list(weights)
    if len(weights) != 2 * len(model.layer_names):
        raise ValueError(
            f"expected {2 * len(model.layer_names)} arrays (kernel and bias per Dense layer), got {len(weights)}"
        )
    with torch.no_grad():
        for index, name in enumerate(model.layer_names):
            linear = getattr(model, name)
            kernel = torch.as_tensor(np.asarray(weights[2 * index]), dtype=linear.weight.dtype)
            bias = torch.as_tensor(np.asarray(weights[2 * index + 1]), dtype=linear.bias.dtype)
            if kernel.shape != linear.weight.T.shape or bias.shape != linear.bias.shape:
                raise ValueError(
                    f"{name}: Keras kernel {tuple(kernel.shape)} / bias {tuple(bias.shape)} do not match "
                    f"in_features={linear.in_features}, out_features={linear.out_features}"
                )
            linear.weight.copy_(kernel.T)
            linear.bias.copy_(bias)
    return model


def tropicalcyclone_mlp_builder(
    task: str,
    input_dim: int = len(SHIPS_PREDICTORS),
    hidden_dims: Sequence[int] = (2048, 2048),
    activations: Union[str, Sequence[str]] = ("sigmoid", "relu"),
    **kwargs,
) -> nn.Module:
    _ = kwargs
    if task.lower() != "regression":
        raise ValueError("tropicalcyclone_mlp is a regression model (24-hour intensity change); use task='regression'.")
    return TropicalCycloneMLP(input_dim=input_dim, hidden_dims=hidden_dims, activations=activations)


__all__ = [
    "LICENSE_NOTICE",
    "SHIPS_PREDICTORS",
    "TropicalCycloneMLP",
    "load_keras_weights",
    "tropicalcyclone_mlp_builder",
]
