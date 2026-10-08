"""EQNet: end-to-end earthquake detection with joint phase picking and association.

Zhu, Tai, Mousavi, Bailis & Beroza, "An End-to-End Earthquake Detection Method for Joint Phase Picking and
Association Using Deep Learning", J. Geophys. Res. Solid Earth 127(3), e2021JB023283 (2022),
doi:10.1029/2021JB023283 (arXiv 2109.09911).

Rebuilt from the paper (Section 2.1, Figure 3, Section 2.2 and the Discussion); no code was copied. The
official repository AI4EPS/EQNet is under an academic / non-commercial licence (and had no licence before
2023-10-26), it was published after the paper, and it does not contain the paper's shift-and-stack model.
It serves only as a test oracle (``tests/oracle/test_eqnet_oracle.py``): its ``ResNet(BasicBlock,
[2, 2, 2, 2])`` (``eqnet/models/resnet1d.py`` at af94a08) for the feature extractor, and the author's
first EQNet heads (``eqnet/models/eqnet.py`` at a9df49a, July 2022) at the paper's channel widths for
the picking and detection heads. Module names of the feature extractor follow torchvision's ResNet
(BSD-3-Clause), which the 1-D network is "modified from" (paper, Section 2.1).

Architecture (1,043,619 parameters):

- Feature extraction (Figure 3a, 996,960 parameters): a 1-D ResNet-18 applied to every station,
  ``Conv1d(3, 32, 7, stride 2) -> BN -> ReLU -> MaxPool(3, 2)``, four stages of two basic blocks with 32,
  64, 128 and 256 channels (stride 2 in stages 2-4), and ``Conv1d(256, 128, 1) -> BN``: waveforms
  ``[Nt, 3]`` become features ``[Nt/32, 128]``.
- Phase picking (Figure 3b, 2 x 7,825 parameters): one sub-network for P on feature channels 0-63 and one
  for S on channels 64-127 (paper: "each network takes half of the extracted features"), each
  ``Conv1d(64, 32, 3) -> BN -> ReLU -> x4 -> Conv1d(32, 16, 3) -> BN -> ReLU -> x4 -> Conv1d(16, 1, 3)``:
  an activation sequence ``[Nt/2]`` (output sample ``j`` is input sample ``2 j``).
- Shift-and-stack (Section 2.1): for every candidate hypocentre the P features of each station are read
  ``t_P`` seconds later and the S features ``t_S`` seconds later (theoretical travel times from the
  station to the candidate), so that both align on the origin time, and averaged over the stations:
  features ``[Nt/32, 128]`` per candidate.
- Event detection (Figure 3c, 31,009 parameters): ``Conv1d(128, 64, 3) -> BN -> ReLU -> Conv1d(64, 32, 3)
  -> BN -> ReLU -> Conv1d(32, 1, 3)``: an activation over origin time ``[Nt/32]`` for every candidate.

Choices the paper leaves open (each one is listed under ``reproduction.deviations`` on the model card):
reflect padding, bias-free convolutions before batch normalisation and a biased output convolution,
linear x4 interpolation (``align_corners=False``) and no ReLU after the extractor's last batch
normalisation follow the author's code; the station average, linear interpolation of fractional shifts,
zero features outside the window and per-channel standardisation of each station's window (as in the
official data reader) are PyHazards choices.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ._seismic import TracePicks, WaveformPicker, as_sequence, std_normalize

FEATURE_STRIDE = 32  # input samples per feature sample (Figure 3a: [Nt/32, 128])
PICK_STRIDE = 2  # input samples per picking-output sample (Figure 3b: [Nt/2, 1])


def _conv3(in_planes: int, out_planes: int, stride: int = 1, bias: bool = False) -> nn.Conv1d:
    return nn.Conv1d(in_planes, out_planes, 3, stride=stride, padding=1, bias=bias, padding_mode="reflect")


class BasicBlock1d(nn.Module):
    """ResNet basic block on ``(batch, channels, time)``: two 3-tap convolutions with BN, identity or
    1x1-projection shortcut, ReLU after the sum."""

    def __init__(self, inplanes: int, planes: int, stride: int = 1, downsample: Optional[nn.Module] = None):
        super().__init__()
        self.conv1 = _conv3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class EQNetBackbone(nn.Module):
    """Feature extraction network (Figure 3a): ``(n, 3, samples)`` -> ``(n, 128, samples / 32)``."""

    widths = (32, 64, 128, 256)

    def __init__(self, in_channels: int = 3, blocks: Sequence[int] = (2, 2, 2, 2), out_channels: int = 128):
        super().__init__()
        self.inplanes = self.widths[0]
        self.conv1 = nn.Conv1d(in_channels, self.inplanes, 7, stride=2, padding=3, bias=False, padding_mode="reflect")
        self.bn1 = nn.BatchNorm1d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(self.widths[0], blocks[0], stride=1)
        self.layer2 = self._make_layer(self.widths[1], blocks[1], stride=2)
        self.layer3 = self._make_layer(self.widths[2], blocks[2], stride=2)
        self.layer4 = self._make_layer(self.widths[3], blocks[3], stride=2)
        self.conv2 = nn.Conv1d(self.widths[3], out_channels, 1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes, 1, stride=stride, bias=False), nn.BatchNorm1d(planes)
            )
        layers = [BasicBlock1d(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes
        layers.extend(BasicBlock1d(planes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"EQNetBackbone expects (n, channels, samples), got shape {tuple(x.shape)}.")
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.bn2(self.conv2(x))


class PhasePickingHead(nn.Module):
    """Phase picking sub-network (Figure 3b): ``(n, 64, T)`` features -> ``(n, 16 T)`` logits."""

    def __init__(self, in_channels: int = 64, channels: Sequence[int] = (32, 16), scale: int = 4):
        super().__init__()
        self.scale = int(scale)
        self.conv1 = _conv3(in_channels, channels[0])
        self.bn1 = nn.BatchNorm1d(channels[0])
        self.conv2 = _conv3(channels[0], channels[1])
        self.bn2 = nn.BatchNorm1d(channels[1])
        self.conv_out = _conv3(channels[1], 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(F.relu(self.bn1(self.conv1(x))), scale_factor=self.scale, mode="linear", align_corners=False)
        x = F.interpolate(F.relu(self.bn2(self.conv2(x))), scale_factor=self.scale, mode="linear", align_corners=False)
        return self.conv_out(x).squeeze(1)


class EventDetectionHead(nn.Module):
    """Event detection sub-network (Figure 3c): ``(n, 128, T)`` stacked features -> ``(n, T)`` logits."""

    def __init__(self, in_channels: int = 128, channels: Sequence[int] = (64, 32)):
        super().__init__()
        self.conv1 = _conv3(in_channels, channels[0])
        self.bn1 = nn.BatchNorm1d(channels[0])
        self.conv2 = _conv3(channels[0], channels[1])
        self.bn2 = nn.BatchNorm1d(channels[1])
        self.conv_out = _conv3(channels[1], 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return self.conv_out(x).squeeze(1)


def _shift(features: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    """``out[b, s, k, c, t] = features[b, s, c, t + shifts[b, s, k]]``, linear in the fractional part,
    zero outside ``[0, T - 1]``. ``features`` ``(B, S, C, T)``, ``shifts`` ``(B, S, K)`` (feature samples)."""
    batch, stations, channels, length = features.shape
    position = torch.arange(length, device=features.device, dtype=features.dtype) + shifts.unsqueeze(-1)
    lower = torch.floor(position)
    weight = (position - lower).unsqueeze(3)  # (B, S, K, 1, T)
    lower = lower.long()
    out = 0
    for index, factor in ((lower, 1.0 - weight), (lower + 1, weight)):
        valid = ((index >= 0) & (index < length)).unsqueeze(3)
        gather_index = index.clamp(0, length - 1).unsqueeze(3).expand(-1, -1, -1, channels, -1)
        source = features.unsqueeze(2).expand(-1, -1, gather_index.shape[2], -1, -1)
        out = out + torch.gather(source, 4, gather_index) * factor * valid
    return out


def shift_and_stack(
    features: torch.Tensor,
    travel_times: torch.Tensor,
    feature_rate: float,
    station_mask: Optional[torch.Tensor] = None,
    p_channels: Optional[int] = None,
) -> torch.Tensor:
    """Shift-and-stack module: align station features on the origin time of each candidate hypocentre.

    ``features``: ``(batch, stations, channels, T)`` backbone features; the first ``p_channels`` (default
    half) are shifted by the P travel time, the rest by the S travel time. ``travel_times``:
    ``(batch, stations, candidates, 2)`` P and S travel times in seconds. ``feature_rate``: feature samples
    per second (``sampling_rate / 32``). ``station_mask``: ``(batch, stations)``, ``False`` for padded
    stations. Returns ``(batch, candidates, channels, T)``: at time ``t`` and candidate ``k``, the mean over
    stations of the features at ``t + travel_time``.
    """
    if features.ndim != 4 or travel_times.ndim != 4 or travel_times.shape[-1] != 2:
        raise ValueError(
            "shift_and_stack expects features shaped (batch, stations, channels, time) and travel_times "
            f"(batch, stations, candidates, 2); got shapes {tuple(features.shape)} and {tuple(travel_times.shape)}."
        )
    if travel_times.shape[:2] != features.shape[:2]:
        raise ValueError(
            f"travel_times shape {tuple(travel_times.shape)} does not match {features.shape[0]} batches of "
            f"{features.shape[1]} stations."
        )
    split = features.shape[2] // 2 if p_channels is None else int(p_channels)
    shifts = travel_times.to(features.dtype) * float(feature_rate)
    shifted = torch.cat(
        [_shift(features[:, :, :split], shifts[..., 0]), _shift(features[:, :, split:], shifts[..., 1])], dim=3
    )  # (B, S, K, C, T)
    if station_mask is None:
        return shifted.mean(dim=1)
    weights = station_mask.to(features.dtype).reshape(*station_mask.shape, 1, 1, 1)
    return (shifted * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def eqnet_travel_times(
    station_locations: torch.Tensor,
    candidate_locations: torch.Tensor,
    vp: float = 6.0,
    vs: float = 3.4,
) -> torch.Tensor:
    """Travel times ``(batch, stations, candidates, 2)`` in seconds in a uniform velocity model.

    Locations are Cartesian coordinates in km, ``(batch, stations, 3)`` (or ``(stations, 3)``) and
    ``(candidates, 3)`` (or ``(batch, candidates, 3)``); the distance is Euclidean. The defaults are the
    paper's Ridgecrest setting: 6 km/s for P and 3.4 km/s for S (Discussion).
    """
    stations = torch.as_tensor(station_locations)
    candidates = torch.as_tensor(candidate_locations, dtype=stations.dtype, device=stations.device)
    if stations.ndim == 2:
        stations = stations.unsqueeze(0)
    if candidates.ndim == 2:
        candidates = candidates.unsqueeze(0).expand(stations.shape[0], -1, -1)
    if stations.shape[-1] != candidates.shape[-1] or stations.shape[0] != candidates.shape[0]:
        raise ValueError(
            f"station_locations {tuple(stations.shape)} and candidate_locations {tuple(candidates.shape)} do not match."
        )
    distance = torch.cdist(stations, candidates)  # (B, S, K)
    return torch.stack([distance / float(vp), distance / float(vs)], dim=-1)


def eqnet_candidate_grid(
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    spacing: float = 4.0,
    depth: float = 0.0,
) -> torch.Tensor:
    """Candidate hypocentres ``(candidates, 3)`` on a horizontal grid (km), the paper's ~4 km search."""
    xs = torch.arange(float(x_range[0]), float(x_range[1]) + 1e-9, float(spacing))
    ys = torch.arange(float(y_range[0]), float(y_range[1]) + 1e-9, float(spacing))
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
    return torch.stack([grid_x.flatten(), grid_y.flatten(), torch.full_like(grid_x.flatten(), float(depth))], dim=-1)


def geometric_median(points: torch.Tensor, iterations: int = 200, eps: float = 1e-7) -> torch.Tensor:
    """Geometric median of ``(n, d)`` points (Weiszfeld's algorithm)."""
    points = points.double()
    estimate = points.mean(dim=0)
    for _ in range(int(iterations)):
        distance = torch.linalg.vector_norm(points - estimate, dim=1).clamp_min(eps)
        weights = 1.0 / distance
        update = (points * weights[:, None]).sum(dim=0) / weights.sum()
        if torch.linalg.vector_norm(update - estimate) < eps:
            estimate = update
            break
        estimate = update
    return estimate


class EQNet(nn.Module, WaveformPicker):
    """EQNet: multi-station phase picking and event detection.

    Input contract (``forward``):

    - ``waveforms``: ``(batch, stations, 3, samples)`` three-component windows (E, N, Z at 100 Hz),
      standardised per station and channel (:meth:`annotate` does it for raw windows). Pad missing
      stations with zeros and mark them in ``station_mask``.
    - ``travel_times`` (optional): ``(batch, stations, candidates, 2)`` P and S travel times in seconds
      from each station to each candidate hypocentre, e.g. :func:`eqnet_travel_times` of station and
      candidate coordinates (:func:`eqnet_candidate_grid`) or a table from any velocity model.
    - ``station_mask`` (optional): ``(batch, stations)`` booleans, ``False`` = padded station (left out of
      the stack).

    Output: a dict with ``"phase"``, ``(batch, stations, 2, 16 * T)`` P / S activations (``T`` feature
    samples, ``samples / 32`` when ``samples`` is a multiple of 32; output sample ``j`` is input sample
    ``2 j``), and, with ``travel_times``, ``"event"``, ``(batch, candidates, T)`` event activations over
    origin time (feature sample ``t`` is input sample ``32 t``). Probabilities by default, logits with
    ``logits=True`` (what :class:`EQNetLoss` takes).

    Single-station picking (the STEAD benchmark of the paper, ``earthquake.picking``): :meth:`annotate`
    takes ``(n, 3, samples)`` and returns ``(n, 2, samples // 2)`` P / S probabilities;
    :meth:`extract_picks` returns peaks above 0.5. :meth:`extract_events` turns event activations into
    origin times and locations.
    """

    sampling_rate = 100.0
    component_order = "ENZ"
    output_names = ("P", "S")
    phase_channels = {"P": 0, "S": 1}

    def __init__(self, in_channels: int = 3, sampling_rate: float = 100.0, candidate_chunk: int = 256):
        super().__init__()
        if int(in_channels) < 1 or float(sampling_rate) <= 0 or int(candidate_chunk) < 1:
            raise ValueError("EQNet needs positive in_channels, sampling_rate and candidate_chunk.")
        self.in_channels = int(in_channels)
        self.sampling_rate = float(sampling_rate)
        self.candidate_chunk = int(candidate_chunk)
        self.backbone = EQNetBackbone(self.in_channels)
        self.p_picker = PhasePickingHead(64)
        self.s_picker = PhasePickingHead(64)
        self.event_detector = EventDetectionHead(128)

    @property
    def feature_rate(self) -> float:
        return self.sampling_rate / FEATURE_STRIDE

    def features(self, waveforms: torch.Tensor) -> torch.Tensor:
        """Backbone features ``(batch, stations, 128, T)`` of ``(batch, stations, 3, samples)`` waveforms."""
        if waveforms.ndim != 4 or waveforms.shape[2] != self.in_channels:
            raise ValueError(
                f"EQNet expects waveforms shaped (batch, stations, {self.in_channels}, samples), "
                f"got shape {tuple(waveforms.shape)}."
            )
        batch, stations = waveforms.shape[:2]
        features = self.backbone(waveforms.reshape(batch * stations, *waveforms.shape[2:]))
        return features.reshape(batch, stations, *features.shape[1:])

    def detect(
        self,
        features: torch.Tensor,
        travel_times: torch.Tensor,
        station_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Event logits ``(batch, candidates, T)``: shift-and-stack, then the event detection network.

        In evaluation mode candidates are processed in chunks of ``candidate_chunk`` to bound memory; in
        training mode all candidates form one batch (batch-normalisation statistics then cover them all).
        """
        batch, stations = features.shape[:2]
        if travel_times.ndim != 4 or travel_times.shape[:2] != (batch, stations) or travel_times.shape[-1] != 2:
            raise ValueError(
                f"travel_times must be shaped ({batch}, {stations}, candidates, 2), got shape {tuple(travel_times.shape)}."
            )
        if station_mask is not None and tuple(station_mask.shape) != (batch, stations):
            raise ValueError(f"station_mask must be shaped ({batch}, {stations}), got shape {tuple(station_mask.shape)}.")
        candidates = travel_times.shape[2]
        chunk = candidates if self.training else self.candidate_chunk
        logits = []
        for start in range(0, candidates, chunk):
            stacked = shift_and_stack(features, travel_times[:, :, start : start + chunk], self.feature_rate, station_mask)
            count = stacked.shape[1]
            out = self.event_detector(stacked.reshape(batch * count, *stacked.shape[2:]))
            logits.append(out.reshape(batch, count, -1))
        return torch.cat(logits, dim=1)

    def forward(
        self,
        waveforms: torch.Tensor,
        travel_times: Optional[torch.Tensor] = None,
        station_mask: Optional[torch.Tensor] = None,
        logits: bool = False,
    ) -> Dict[str, torch.Tensor]:
        features = self.features(waveforms)
        batch, stations = features.shape[:2]
        flat = features.reshape(batch * stations, *features.shape[2:])
        phase = torch.stack([self.p_picker(flat[:, :64]), self.s_picker(flat[:, 64:])], dim=1)
        outputs = {"phase": phase.reshape(batch, stations, 2, -1)}
        if travel_times is not None:
            outputs["event"] = self.detect(features, travel_times, station_mask)
        if not logits:
            outputs = {key: torch.sigmoid(value) for key, value in outputs.items()}
        return outputs

    def annotate(self, waveforms: torch.Tensor) -> torch.Tensor:
        """P / S probabilities of raw windows: ``(n, 3, samples)`` -> ``(n, 2, samples // 2)`` (single
        stations) or ``(batch, stations, 3, samples)`` -> ``(batch, stations, 2, samples // 2)``. Each
        station's window is standardised per channel first (mean removed, divided by the standard
        deviation, as the official EQNet and PhaseNet readers do)."""
        single = waveforms.ndim == 3
        if single:
            waveforms = waveforms.unsqueeze(1)
        if waveforms.ndim != 4:
            raise ValueError(
                f"EQNet.annotate expects (n, 3, samples) or (batch, stations, 3, samples), got shape {tuple(waveforms.shape)}."
            )
        batch, stations, channels, samples = waveforms.shape
        normalised = std_normalize(waveforms.reshape(batch * stations, channels, samples))
        phase = self(normalised.reshape(batch, stations, channels, samples))["phase"][..., : samples // PICK_STRIDE]
        return phase[:, 0] if single else phase

    def extract_picks(
        self,
        annotations: torch.Tensor,
        p_threshold: float = 0.5,
        s_threshold: float = 0.5,
        min_distance_s: float = 0.5,
        **kwargs,
    ) -> List[TracePicks]:
        """Peaks above the threshold (0.5 in the paper) of ``(n, 2, length)`` P / S activations, at least
        ``min_distance_s`` apart (the PhaseNet rule; the paper gives no distance). Pick samples are in
        input samples (``2 j``)."""
        from ..metrics.picking import peak_picks

        if kwargs:
            raise TypeError(f"Unexpected EQNet pick parameters: {sorted(kwargs)}.")
        if annotations.ndim != 3 or annotations.shape[1] != 2:
            raise ValueError(f"extract_picks expects (n, 2, length) activations, got shape {tuple(annotations.shape)}.")
        thresholds = as_sequence({"P": p_threshold, "S": s_threshold}, ("P", "S"))
        distance = max(1, int(round(min_distance_s * self.sampling_rate / PICK_STRIDE)))
        values = annotations.detach().cpu().double().numpy()
        picks: List[TracePicks] = []
        for trace in values:
            picks.append(
                {
                    phase: [(PICK_STRIDE * index, probability) for index, probability in peak_picks(trace[channel], thresholds[phase], distance)]
                    for phase, channel in self.phase_channels.items()
                }
            )
        return picks

    def extract_events(
        self,
        event_activation: torch.Tensor,
        candidate_locations: torch.Tensor,
        threshold: float = 0.5,
        top_k: int = 20,
        min_distance_s: float = 2.0,
    ) -> List[List[Dict[str, object]]]:
        """Events from ``(batch, candidates, T)`` activations (paper, Section 2.2): origin times are the
        peaks above ``threshold`` of the maximum over candidates, at least ``min_distance_s`` apart (not
        given in the paper); each location is the geometric median of the ``top_k`` (20) candidates with
        the highest activation at that time. Returns per batch item a list of
        ``{"time": seconds, "sample": input sample, "probability": p, "location": [x, y, z]}``."""
        from ..metrics.picking import peak_picks

        if event_activation.ndim != 3:
            raise ValueError(f"extract_events expects (batch, candidates, T) activations, got shape {tuple(event_activation.shape)}.")
        candidates = torch.as_tensor(candidate_locations).double().cpu()
        if candidates.ndim != 2 or candidates.shape[0] != event_activation.shape[1]:
            raise ValueError(
                f"candidate_locations must be shaped ({event_activation.shape[1]}, dims), got shape {tuple(candidates.shape)}."
            )
        distance = max(1, int(round(min_distance_s * self.feature_rate)))
        activation = event_activation.detach().cpu().double()
        events: List[List[Dict[str, object]]] = []
        for item in activation:
            found = []
            for index, probability in peak_picks(item.max(dim=0).values.numpy(), threshold, distance):
                t = int(index)
                top = torch.argsort(item[:, t], descending=True)[: int(top_k)]
                found.append(
                    {
                        "time": t / self.feature_rate,
                        "sample": FEATURE_STRIDE * t,
                        "probability": probability,
                        "location": geometric_median(candidates[top]).tolist(),
                    }
                )
            events.append(found)
        return events


class EQNetLoss(nn.Module):
    """Equations (1)-(4): ``lambda_P L_P + lambda_S L_S + lambda_EQ L_EQ``, binary cross-entropies summed
    over time (mean over stations / candidates and batch), all weights 1 in the paper.

    ``outputs`` are EQNet logits (``forward(..., logits=True)``); ``phase_targets`` match
    ``outputs["phase"]`` ``(batch, stations, 2, L)`` and ``event_targets`` match ``outputs["event"]``
    ``(batch, candidates, T)``. The paper's targets are truncated Gaussians around the manual picks
    (width 1 s) and the origin time at the true location (width 2 s), with negative sampling of other
    candidate locations.
    """

    def __init__(self, weights: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        super().__init__()
        self.weights = tuple(float(w) for w in weights)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        phase_targets: torch.Tensor,
        event_targets: Optional[torch.Tensor] = None,
        station_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        phase = outputs["phase"]
        if phase.shape != phase_targets.shape:
            raise ValueError(f"phase_targets must be shaped {tuple(phase.shape)}, got shape {tuple(phase_targets.shape)}.")
        terms = F.binary_cross_entropy_with_logits(phase, phase_targets, reduction="none").sum(dim=-1)  # (B, S, 2)
        if station_mask is not None:
            mask = station_mask.to(terms.dtype).unsqueeze(-1)
            per_phase = (terms * mask).sum(dim=(0, 1)) / mask.sum(dim=(0, 1)).clamp_min(1.0)
        else:
            per_phase = terms.mean(dim=(0, 1))
        loss = self.weights[0] * per_phase[0] + self.weights[1] * per_phase[1]
        if event_targets is not None:
            event = outputs["event"]
            if event.shape != event_targets.shape:
                raise ValueError(f"event_targets must be shaped {tuple(event.shape)}, got shape {tuple(event_targets.shape)}.")
            event_loss = F.binary_cross_entropy_with_logits(event, event_targets, reduction="none").sum(dim=-1).mean()
            loss = loss + self.weights[2] * event_loss
        return loss


_OLD_ARGUMENTS = {"hidden_dim", "num_heads", "num_layers", "dropout"}


def eqnet_builder(
    task: str = "picking",
    in_channels: int = 3,
    sampling_rate: float = 100.0,
    candidate_chunk: int = 256,
    **kwargs,
) -> EQNet:
    """EQNet for ``task="picking"`` or ``task="detection"`` (the same multi-task network)."""
    kwargs.pop("name", None)
    old = sorted(set(kwargs) & _OLD_ARGUMENTS)
    if old:
        raise TypeError(
            f"EQNet no longer takes {old}: the former transformer stand-in was replaced by the paper's ResNet "
            "network with picking heads, shift-and-stack and event detection."
        )
    if kwargs:
        raise TypeError(f"Unexpected EQNet arguments: {sorted(kwargs)}.")
    if task.lower() not in {"picking", "detection"}:
        raise ValueError(f"EQNet supports task='picking' or 'detection', got {task!r}.")
    return EQNet(in_channels=in_channels, sampling_rate=sampling_rate, candidate_chunk=candidate_chunk)


__all__ = [
    "BasicBlock1d",
    "EQNet",
    "EQNetBackbone",
    "EQNetLoss",
    "EventDetectionHead",
    "PhasePickingHead",
    "eqnet_builder",
    "eqnet_candidate_grid",
    "eqnet_travel_times",
    "geometric_median",
    "shift_and_stack",
]
