"""Optional runner: produce forecast fields with NVIDIA earth2studio, then track them with PyHazards.

earth2studio (https://github.com/NVIDIA/earth2studio, Apache-2.0) wraps the official FourCastNet,
SFNO (FourCastNet v2), Pangu-Weather and GraphCast checkpoints behind one prognostic-model interface.
PyHazards does not install it or any weights: install it yourself (Python >= 3.11; checked with
0.19.0) together with the extra each model needs (see the earth2studio documentation), and accept the
weights' licences. Weights are downloaded by earth2studio when a model is loaded:

=========================  ======================================  =============================================
``model``                  earth2studio class                      weights and licence (as stated by the providers)
=========================  ======================================  =============================================
``fourcastnet``            ``earth2studio.models.px.FCN``          26-variable AFNO retrain of FourCastNet (HF
                                                                   ``nvidia/fourcastnet1``, model card: Apache-2.0);
                                                                   not the 20-variable checkpoint of Pathak et al.
``fourcastnet_v2``         ``earth2studio.models.px.SFNO``         SFNO ``sfno_73ch_small`` (NGC; the model TCBench
                                                                   calls FourCastNet v2)
``pangu_6`` / ``pangu_24``  ``earth2studio.models.px.Pangu6/24``    official ONNX graphs, CC BY-NC-SA 4.0:
                                                                   non-commercial use only
``graphcast_operational``  ``earth2studio.models.px.GraphCastOperational``  DeepMind checkpoint (needs JAX); the
``graphcast_small``        ``earth2studio.models.px.GraphCastSmall``        repository README states CC BY 4.0 since
                                                                            2026-08-06, the checkpoint metadata
                                                                            still says CC BY-NC-SA 4.0
=========================  ======================================  =============================================

:func:`run_earth2studio_forecast` follows ``earth2studio.run.deterministic``: fetch the initial state
from a data source, iterate the prognostic model and keep the variables a tracker needs, renamed to
the PyHazards conventions (:mod:`pyhazards.forecasts.fields`). Global 0.25-degree models need a GPU
with about 40 GB of memory (earth2studio's badges).
"""

from __future__ import annotations

import importlib
from typing import Dict, Optional, Sequence

import numpy as np

__all__ = [
    "EARTH2STUDIO_MODELS",
    "EARTH2STUDIO_VARIABLES",
    "load_earth2studio_model",
    "run_earth2studio_forecast",
]

EARTH2STUDIO_MODELS: Dict[str, str] = {
    "fourcastnet": "FCN",
    "fourcastnet_v2": "SFNO",
    "pangu_3": "Pangu3",
    "pangu_6": "Pangu6",
    "pangu_24": "Pangu24",
    "graphcast_operational": "GraphCastOperational",
    "graphcast_small": "GraphCastSmall",
}

EARTH2STUDIO_VARIABLES: Dict[str, str] = {"u10": "u10m", "v10": "v10m", "t2m": "t2m", "msl": "msl"}
"""PyHazards name -> earth2studio name where they differ (pressure levels: ``u850`` in both)."""


def _earth2studio():
    try:
        return importlib.import_module("earth2studio")
    except ImportError as exc:
        raise ImportError(
            "run_earth2studio_forecast needs NVIDIA earth2studio, which PyHazards does not install. "
            "Install it yourself (pip install earth2studio, plus the model's extra; Python >= 3.11)."
        ) from exc


def load_earth2studio_model(model: str, device: Optional[str] = None):
    """Load an earth2studio prognostic model with its default (downloaded) checkpoint."""
    if model not in EARTH2STUDIO_MODELS:
        raise ValueError(f"unknown model {model!r}; choose from {sorted(EARTH2STUDIO_MODELS)}")
    _earth2studio()
    px = importlib.import_module("earth2studio.models.px")
    cls = getattr(px, EARTH2STUDIO_MODELS[model])
    prognostic = cls.load_model(cls.load_default_package())
    return prognostic.to(device) if device is not None else prognostic


def run_earth2studio_forecast(
    prognostic,
    data,
    init_time,
    lead_hours: int = 120,
    variables: Sequence[str] = ("msl", "u10", "v10", "z300", "z500"),
    device: Optional[str] = None,
):
    """Run ``prognostic`` from ``init_time`` and return the tracker fields as an ``xarray.Dataset``.

    ``prognostic`` is an earth2studio prognostic model (or a name from :data:`EARTH2STUDIO_MODELS`,
    loaded with :func:`load_earth2studio_model`); ``data`` an earth2studio data source (e.g.
    ``earth2studio.data.ARCO()`` for ERA5, ``GFS()``, ``WB2ERA5()``). The model's own time step sets
    the output times, from the initial state (lead 0) to ``lead_hours``.
    """
    import pandas as pd
    import torch
    import xarray as xr

    _earth2studio()
    from earth2studio.data import fetch_data

    if isinstance(prognostic, str):
        prognostic = load_earth2studio_model(prognostic, device)
    device = torch.device(device) if device is not None else torch.device("cpu")
    prognostic = prognostic.to(device)
    wanted = [EARTH2STUDIO_VARIABLES.get(name, name) for name in variables]
    input_coords = prognostic.input_coords()
    output_variables = list(prognostic.output_coords(input_coords)["variable"])
    missing = [name for name, e2s in zip(variables, wanted) if e2s not in output_variables]
    if missing:
        raise KeyError(f"the model does not predict {missing} (earth2studio names {output_variables})")
    init = pd.Timestamp(init_time)
    x, coords = fetch_data(
        source=data,
        time=np.array([init.to_datetime64()]),
        variable=input_coords["variable"],
        lead_time=input_coords["lead_time"],
        device=device,
    )
    frames, valid_times = [], []
    lat = lon = None
    with torch.inference_mode():
        for state, state_coords in prognostic.create_iterator(x, coords):
            names = list(state_coords["variable"])
            axis = list(state_coords).index("variable")
            index = torch.as_tensor([names.index(v) for v in wanted], device=state.device)
            selected = torch.index_select(state, axis, index).detach().cpu().numpy()
            keys = list(state_coords)
            # Reduce the batch / time / lead_time axes (size 1 for a deterministic single run).
            for name in [k for k in keys if k not in ("variable", "lat", "lon")]:
                position = keys.index(name)
                selected = np.take(selected, -1, axis=position)
                keys.pop(position)
            selected = np.moveaxis(selected, keys.index("variable"), 0)
            lead = np.asarray(state_coords["lead_time"]).ravel()[-1]
            hours = float(lead / np.timedelta64(1, "h"))
            if hours > lead_hours + 1e-9:
                break
            frames.append(selected.astype(np.float32))
            valid_times.append(init + pd.Timedelta(hours=hours))
            lat, lon = np.asarray(state_coords["lat"]), np.asarray(state_coords["lon"])
            if hours >= lead_hours - 1e-9:
                break
    if not frames:
        raise RuntimeError("the model produced no output")
    stacked = np.stack(frames, axis=1)  # (variable, time, lat, lon)
    ds = xr.Dataset(
        {name: (("time", "lat", "lon"), stacked[k]) for k, name in enumerate(variables)},
        coords={"time": valid_times, "lat": lat.astype(np.float64), "lon": lon.astype(np.float64)},
    )
    ds.attrs.update({"source": f"earth2studio:{type(prognostic).__name__}", "init_time": str(init)})
    return ds
