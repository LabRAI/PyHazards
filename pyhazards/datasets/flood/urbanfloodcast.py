"""UrbanFloodCast inputs: one-shot event samples for the deep neural operator, and a synthetic stand-in.

The UrbanFloodCast benchmark (Xu et al., J. Hydrology 2025; Zenodo 10.5281/zenodo.15700880, 7.3 GB,
CC BY 4.0) stores hydrodynamic simulations of design storms over two areas of Berlin as GeoTIFF rasters
(water depth ``H`` and unit discharges ``U`` / ``V`` every 300 s, rainfall tables, terrain). The official
code first stacks one event into a tensor ``(Sy, Sx, 25, 5)`` with the channels depth, x discharge,
y discharge, rainfall and terrain elevation (``TIFF2PT``), then builds the one-shot sample
(``flood_data`` in ``DNO/utils25.py``). :func:`prepare_urbanfloodcast_event` does the second step the
same way (written from the reference behaviour, which the oracle test compares):

- input: depth / discharges at the first ``T_in`` steps, discharges set to 0 where the depth is 0, NaN
  set to 0, repeated over the ``T_out`` output steps; rainfall of each output step as
  ``log(1 + p / 0.01) / 10``; terrain of each output step with NaN replaced by ``max + 30`` and
  L2-normalised along the x axis (``torch.nn.functional.normalize`` over dimension 1);
- target: depth / discharges of the ``T_out`` output steps, discharges 0 where the depth is 0, NaN set
  to 0, and a mask that is False where the simulation is NaN (buildings, outside the domain).

PyHazards does not read the Berlin GeoTIFFs yet (the official TIFF-to-tensor script is incomplete as
released); ``urbanfloodcast_synthetic`` generates random events in this layout for smoke tests.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from ..base import DataBundle, DataSplit, Dataset, FeatureSpec, LabelSpec

URBANFLOODCAST_EVENT_CHANNELS = ("water_depth", "discharge_x", "discharge_y", "rainfall", "elevation")
URBANFLOODCAST_INPUT_CHANNELS = ("water_depth", "discharge_x", "discharge_y", "rainfall_log", "elevation_normalized")
URBANFLOODCAST_TARGETS = ("water_depth", "discharge_x", "discharge_y")


def _log_transform(data: torch.Tensor, eps: float = 1e-2) -> torch.Tensor:
    return torch.log(1 + data / eps)


def prepare_urbanfloodcast_event(
    event: torch.Tensor, T_in: int = 1, T_out: int = 24  # noqa: N803 (reference names)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """One-shot sample of an event tensor ``(Sy, Sx, >= T_in + T_out, 5)``.

    Returns ``x`` ``(Sy, Sx, T_out, T_in, 5)``, ``y`` ``(Sy, Sx, T_out, 3)`` and ``mask`` (bool, like ``y``).
    The event tensor is not modified.
    """
    if event.ndim != 4 or event.shape[-1] != 5:
        raise ValueError(
            f"UrbanFloodCast events must have shape (Sy, Sx, time, 5) with channels {URBANFLOODCAST_EVENT_CHANNELS}; "
            f"got {tuple(event.shape)}."
        )
    if event.shape[2] < T_in + T_out:
        raise ValueError(f"The event has {event.shape[2]} steps; T_in + T_out = {T_in + T_out} are needed.")
    if T_in != 1:
        raise ValueError("The one-shot UrbanFloodCast sample is defined for T_in=1 (the reference concatenation needs it).")
    pde = event.clone()
    x = pde[..., :T_in, :3]
    mask = x[..., 0:1] == 0.0
    x[..., 1:2][mask] = 0.0
    x[..., 2:3][mask] = 0.0
    x[..., :3] = torch.nan_to_num(x[..., :3], nan=0.0)
    x1 = x.unsqueeze(-3).repeat([1, 1, T_out, 1, 1])
    rain = pde[..., T_in : T_in + T_out, 3:4]
    x2 = torch.unsqueeze(_log_transform(rain) / 10.0, dim=-1)
    z = pde[..., T_in : T_in + T_out, 4:5]
    z = torch.nan_to_num(z, nan=z.max() + 30.0)
    x3 = torch.unsqueeze(F.normalize(z), dim=-1)
    inputs = torch.cat((torch.cat((x1, x2), dim=-1), x3), dim=-1)
    y = pde[..., T_in : T_in + T_out, :3]
    mask_y = y[..., 0:1] == 0.0
    y[..., 1:2][mask_y] = 0.0
    y[..., 2:3][mask_y] = 0.0
    valid = ~torch.isnan(y)
    return inputs, torch.nan_to_num(y, nan=0.0), valid


def synthetic_urbanfloodcast_event(
    height: int = 32, width: int = 32, steps: int = 25, generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    """A random event tensor ``(height, width, steps, 5)``: rain on a tilted terrain with buildings (NaN)."""
    rows = torch.linspace(0, 1, height).view(height, 1)
    cols = torch.linspace(0, 1, width).view(1, width)
    terrain = 34.0 + 2.0 * rows + 1.0 * cols + 0.2 * torch.rand(height, width, generator=generator)
    rain_total = 10 + 40 * torch.rand(1, generator=generator).item()
    profile = torch.exp(-((torch.arange(steps) - steps / 3) ** 2) / (2 * (steps / 6) ** 2))
    rain = rain_total * profile / profile.sum()
    cumulative = torch.cumsum(rain, 0) / 1000.0
    low = (terrain.max() - terrain) / (terrain.max() - terrain.min())
    depth = cumulative.view(1, 1, steps) * (0.5 + 3 * low).unsqueeze(-1)
    depth = depth * (torch.rand(height, width, 1, generator=generator) > 0.1)
    qx = 0.1 * depth * torch.randn(height, width, 1, generator=generator)
    qy = 0.1 * depth * torch.randn(height, width, 1, generator=generator)
    event = torch.stack(
        [depth, qx, qy, rain.view(1, 1, steps).expand(height, width, steps), terrain.unsqueeze(-1).expand(height, width, steps)],
        dim=-1,
    )
    buildings = torch.rand(height, width, generator=generator) < 0.05
    event[buildings, :, :3] = float("nan")
    return event


class SyntheticUrbanFloodCastDataset(Dataset):
    """Synthetic urban flood events in the UrbanFloodCast one-shot layout (random numbers; smoke tests only).

    Every split holds ``inputs`` ``(n, Sy, Sx, T_out, 1, 5)`` and ``targets`` ``(n, Sy, Sx, T_out, 3)``
    built by :func:`prepare_urbanfloodcast_event`; the NaN mask is ``split.metadata["mask"]``.
    """

    name = "urbanfloodcast_synthetic"

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        events: int = 8,
        height: int = 32,
        width: int = 32,
        T_out: int = 24,  # noqa: N803
        seed: int = 0,
        micro: bool = False,
    ):
        super().__init__(cache_dir=cache_dir)
        self.events = 4 if micro else int(events)
        self.height = int(height)
        self.width = int(width)
        self.T_out = int(T_out)
        self.seed = int(seed)
        if self.events < 3:
            raise ValueError("urbanfloodcast_synthetic needs at least 3 events (train / val / test).")

    def _load(self) -> DataBundle:
        generator = torch.Generator().manual_seed(self.seed)
        samples = [
            prepare_urbanfloodcast_event(
                synthetic_urbanfloodcast_event(self.height, self.width, self.T_out + 1, generator), 1, self.T_out
            )
            for _ in range(self.events)
        ]
        x = torch.stack([s[0] for s in samples])
        y = torch.stack([s[1] for s in samples])
        mask = torch.stack([s[2] for s in samples])
        n_test = max(1, self.events // 4)
        n_val = max(1, self.events // 4)
        n_train = self.events - n_val - n_test
        bounds = {"train": (0, n_train), "val": (n_train, n_train + n_val), "test": (n_train + n_val, self.events)}
        splits = {
            name: DataSplit(x[a:b], y[a:b], metadata={"mask": mask[a:b]}) for name, (a, b) in bounds.items()
        }
        return DataBundle(
            splits=splits,
            feature_spec=FeatureSpec(
                channels=5,
                description="Synthetic UrbanFloodCast inputs: " + ", ".join(URBANFLOODCAST_INPUT_CHANNELS) + " per cell and output step.",
                extra={"height": self.height, "width": self.width, "T_out": self.T_out, "initial_step": 1},
            ),
            label_spec=LabelSpec(
                num_targets=3,
                task_type="regression",
                description="Synthetic water depth and x / y unit discharge at every output step.",
                extra={"variables": list(URBANFLOODCAST_TARGETS)},
            ),
            metadata={
                "dataset": self.name,
                "source_dataset": self.name,
                "hazard_task": "flood.inundation",
                "inundation_layout": "channels_last",
                "depth_channel": 0,
                "synthetic": True,
            },
        )


__all__ = [
    "SyntheticUrbanFloodCastDataset",
    "URBANFLOODCAST_EVENT_CHANNELS",
    "URBANFLOODCAST_INPUT_CHANNELS",
    "URBANFLOODCAST_TARGETS",
    "prepare_urbanfloodcast_event",
    "synthetic_urbanfloodcast_event",
]
