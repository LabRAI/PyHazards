"""Interface shared by the seismic phase pickers (PhaseNet, EQTransformer, GPD).

The ``earthquake.picking`` benchmark scores any model that follows this interface:

- ``sampling_rate`` (Hz) and ``component_order`` (e.g. ``"ENZ"``) describe the waveforms the released
  weights expect; the benchmark reorders dataset channels to ``component_order`` and refuses other
  sampling rates.
- ``annotate(waveforms)`` takes raw windows ``(batch, 3, samples)``, applies the official per-window
  normalisation and returns the model's probabilities (per-sample traces, or per-window values for
  sliding-window classifiers).
- ``extract_picks(annotations)`` turns those probabilities into picks with the official
  post-processing of each model: one ``{"P": [(sample, probability), ...], "S": [...]}`` dict per trace.
- Models with an event-detection output also provide ``extract_detections(annotations)``: one list of
  ``(on, off, probability)`` windows per trace.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

TracePicks = Dict[str, List[Tuple[float, float]]]


class WaveformPicker:
    """Mixin documenting the phase-picker interface (see the module docstring)."""

    sampling_rate: float = 100.0
    component_order: str = "ZNE"
    output_names: Tuple[str, ...] = ()

    def annotate(self, waveforms: torch.Tensor) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError

    def extract_picks(self, annotations: torch.Tensor, **kwargs) -> List[TracePicks]:  # pragma: no cover
        raise NotImplementedError


class KerasBatchNorm1d(nn.BatchNorm1d):
    """``BatchNorm1d`` with the training-time running statistics of standalone Keras 2.2 / 2.3.

    Shared by the EQTransformer and GPD ports (released weights trained with Keras 2.2.2 / 2.2.4 / 2.3.0).

    Normalisation (batch statistics in train mode, running statistics in eval mode) is that of
    ``BatchNorm1d`` with Keras' defaults (epsilon 1e-3, momentum 0.99). Only the running-statistics
    update differs: ``keras/layers/normalization.py`` scales the batch variance by
    ``n / (n - (1 + epsilon))`` (PyTorch: ``n / (n - 1)``), and ``K.moving_average_update`` is a plain
    exponential moving average in Keras 2.3.0 (``zero_debias=False``, the paper model) but TensorFlow's
    zero-debiased ``assign_moving_average`` in Keras 2.2.4 (``zero_debias=True``, the conservative
    model): a zero-initialised accumulator divided by ``1 - 0.99 ** t``, with ``t`` counted by
    ``num_batches_tracked`` (so it restarts after loading weights, as in Keras).
    """

    def __init__(self, num_features: int, zero_debias: bool = False, eps: float = 1e-3, keras_momentum: float = 0.99):
        super().__init__(num_features, eps=eps, momentum=1.0 - keras_momentum)
        self.zero_debias = bool(zero_debias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return super().forward(x)
        output = F.batch_norm(x, None, None, self.weight, self.bias, True, 0.0, self.eps)
        with torch.no_grad():
            dims = [0] if x.ndim == 2 else [0, 2]
            sample_size = float(x.numel() // x.shape[1])
            mean = x.mean(dim=dims)
            variance = x.var(dim=dims, unbiased=False) * (sample_size / (sample_size - (1.0 + self.eps)))
            decay = 1.0 - float(self.momentum)
            steps = int(self.num_batches_tracked)
            for buffer, value in ((self.running_mean, mean), (self.running_var, variance)):
                value = value.to(buffer.dtype)
                if self.zero_debias:
                    biased = buffer * (1.0 - decay**steps) if steps > 0 else torch.zeros_like(buffer)
                    biased = decay * biased + (1.0 - decay) * value
                    buffer.copy_(biased / (1.0 - decay ** (steps + 1)))
                else:
                    buffer.sub_((buffer - value) * (1.0 - decay))
            self.num_batches_tracked.add_(1)
        return output


def std_normalize(waveforms: torch.Tensor) -> torch.Tensor:
    """Remove the mean of every channel and divide by its standard deviation (1 where it is 0).

    The per-window normalisation of PhaseNet (``normalize`` in ``phasenet/data_reader.py``) and of
    EQTransformer (``normalize(data, mode="std")`` in ``EqT_utils.py``): population standard deviation
    over the samples of each channel, ``(batch, channels, samples)`` layout.
    """
    if waveforms.ndim != 3:
        raise ValueError(f"Expected waveforms shaped (batch, channels, samples), got shape {tuple(waveforms.shape)}.")
    centred = waveforms - waveforms.mean(dim=-1, keepdim=True)
    std = centred.std(dim=-1, unbiased=False, keepdim=True)
    return centred / torch.where(std == 0, torch.ones_like(std), std)


def reorder_components(waveforms: torch.Tensor, source: str, target: str) -> torch.Tensor:
    """Reorder the channel axis from component order ``source`` (e.g. ``"ZNE"``) to ``target``."""
    source, target = source.upper(), target.upper()
    if source == target:
        return waveforms
    if sorted(source) != sorted(target):
        raise ValueError(f"Cannot reorder components {source!r} to {target!r}.")
    index = [source.index(component) for component in target]
    return waveforms[:, index]


def check_waveforms(waveforms: torch.Tensor, channels: int, name: str) -> None:
    if waveforms.ndim != 3 or waveforms.shape[1] != channels:
        raise ValueError(
            f"{name} expects waveforms shaped (batch, {channels}, samples), got shape {tuple(waveforms.shape)}."
        )


def picks_from_peaks(
    annotations: torch.Tensor,
    phase_channels: Dict[str, int],
    thresholds: Dict[str, float],
    min_distance: int,
) -> List[TracePicks]:
    """Peak picks on per-sample probability traces (the PhaseNet rule; see ``metrics.picking``)."""
    from ..metrics.picking import peak_picks

    values = annotations.detach().cpu().double().numpy()
    picks: List[TracePicks] = []
    for trace in values:
        picks.append(
            {
                phase: peak_picks(trace[channel], thresholds[phase], min_distance)
                for phase, channel in phase_channels.items()
            }
        )
    return picks


def as_sequence(value, names: Sequence[str]) -> Dict[str, float]:
    """A per-phase dict from a scalar or a dict."""
    if isinstance(value, dict):
        return {name: float(value[name]) for name in names}
    return {name: float(value) for name in names}


__all__ = [
    "KerasBatchNorm1d",
    "TracePicks",
    "WaveformPicker",
    "as_sequence",
    "check_waveforms",
    "picks_from_peaks",
    "reorder_components",
    "std_normalize",
]
