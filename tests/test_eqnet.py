import math

import pytest
import torch
import torch.nn.functional as F

from pyhazards.models import build_model
from pyhazards.models.eqnet import (
    EQNet,
    EQNetLoss,
    eqnet_candidate_grid,
    eqnet_travel_times,
    geometric_median,
    shift_and_stack,
)

RATE = 100.0 / 32  # feature samples per second at 100 Hz


def _count(module):
    return sum(p.numel() for p in module.parameters())


def test_parameter_counts_follow_figure_3():
    model = EQNet()
    assert _count(model) == 1_043_619
    assert {name: _count(m) for name, m in model.named_children()} == {
        "backbone": 996_960,
        "p_picker": 7_825,
        "s_picker": 7_825,
        "event_detector": 31_009,
    }
    assert model.backbone.conv1.weight.shape == (32, 3, 7)
    assert model.backbone.conv2.weight.shape == (128, 256, 1)
    assert model.p_picker.conv1.weight.shape == (32, 64, 3) and model.p_picker.conv_out.bias is not None
    assert model.event_detector.conv1.weight.shape == (64, 128, 3)


def test_multi_station_contract_and_single_station_annotation():
    torch.manual_seed(0)
    model = build_model("eqnet", task="detection").eval()
    waveforms = torch.randn(2, 4, 3, 1024)
    travel_times = torch.rand(2, 4, 7, 2) * 5
    with torch.no_grad():
        out = model(waveforms, travel_times)
        assert out["phase"].shape == (2, 4, 2, 512) and out["event"].shape == (2, 7, 32)
        assert 0 <= float(out["phase"].min()) and float(out["phase"].max()) <= 1
        logits = model(waveforms, travel_times, logits=True)
        torch.testing.assert_close(torch.sigmoid(logits["event"]), out["event"])
        assert set(model(waveforms)) == {"phase"}
        # Candidates are chunked in evaluation mode; the result does not depend on the chunk size.
        model.candidate_chunk = 2
        torch.testing.assert_close(model(waveforms, travel_times)["event"], out["event"])
        # Single stations (the picking benchmark): (n, 3, samples) -> (n, 2, samples // 2).
        assert model.annotate(torch.randn(3, 3, 6000)).shape == (3, 2, 3000)
        assert model.annotate(torch.randn(1, 2, 3, 3000)).shape == (1, 2, 2, 1500)
    with pytest.raises(ValueError, match="shape"):
        model(torch.randn(2, 3, 1024))
    with pytest.raises(ValueError, match="shape"):
        model(waveforms, torch.rand(2, 3, 7, 2))
    with pytest.raises(ValueError, match="shape"):
        model(waveforms, travel_times, station_mask=torch.ones(2, 3, dtype=torch.bool))


def test_shift_and_stack_finds_the_true_hypocentre_and_origin_time():
    stations = torch.tensor([[0.0, 0.0, 0.0], [40.0, 0.0, 0.0], [0.0, 40.0, 0.0], [40.0, 40.0, 0.0], [20.0, -5.0, 0.0], [-5.0, 20.0, 0.0]])
    candidates = eqnet_candidate_grid((0.0, 40.0), (0.0, 40.0), spacing=4.0, depth=8.0)
    assert candidates.shape == (121, 3)
    true_index, origin = 47, 20.0  # feature samples
    travel_times = eqnet_travel_times(stations, candidates)  # (1, 6, 121, 2), 6 and 3.4 km/s
    assert travel_times.shape == (1, 6, 121, 2)
    distance = torch.linalg.vector_norm(stations[2] - candidates[5])
    torch.testing.assert_close(travel_times[0, 2, 5], torch.stack([distance / 6.0, distance / 3.4]))

    time = torch.arange(96.0)
    features = torch.zeros(1, 6, 128, 96)
    for s in range(6):
        p_arrival = origin + travel_times[0, s, true_index, 0] * RATE
        s_arrival = origin + travel_times[0, s, true_index, 1] * RATE
        features[0, s, :64] = torch.exp(-0.5 * ((time - p_arrival) / 0.7) ** 2)
        features[0, s, 64:] = torch.exp(-0.5 * ((time - s_arrival) / 0.7) ** 2)
    stacked = shift_and_stack(features, travel_times, RATE)
    assert stacked.shape == (1, 121, 128, 96)
    score = stacked.sum(dim=2)[0]
    best = int(score.argmax())
    assert divmod(best, 96) == (true_index, int(origin))

    # The detector path runs on the same geometry, chunked or not.
    model = EQNet().eval()
    with torch.no_grad():
        event = model.detect(features, travel_times)
    assert event.shape == (1, 121, 96)


def test_shift_and_stack_interpolates_masks_and_backpropagates():
    features = torch.randn(1, 2, 4, 10, requires_grad=True)
    half = torch.full((1, 2, 1, 2), 0.5 / RATE)  # half a feature sample for P and S
    shifted = shift_and_stack(features, half, RATE)
    expected = 0.5 * (features[..., :-1] + features[..., 1:]).mean(dim=1)
    torch.testing.assert_close(shifted[0, 0, :, :-1], expected[0])
    torch.testing.assert_close(shifted[0, 0, :, -1], 0.5 * features[0, :, :, -1].mean(dim=0))  # zero beyond the end
    shifted.sum().backward()
    assert features.grad is not None and float(features.grad.abs().sum()) > 0

    garbage = features.detach().clone()
    garbage[0, 1] = 1e6
    mask = torch.tensor([[True, False]])
    zero = torch.zeros(1, 2, 3, 2)
    torch.testing.assert_close(shift_and_stack(garbage, zero, RATE, station_mask=mask)[0, 0], garbage[0, 0])


def test_pick_and_event_extraction():
    model = EQNet()
    annotations = torch.zeros(1, 2, 200)
    annotations[0, 0, 30] = 0.9
    annotations[0, 1, 60] = 0.7
    annotations[0, 1, 62] = 0.6  # within 0.5 s of the stronger peak: dropped
    annotations[0, 1, 150] = 0.4  # below the threshold
    picks = model.extract_picks(annotations)
    assert picks == [{"P": [(60.0, pytest.approx(0.9))], "S": [(120.0, pytest.approx(0.7))]}]

    candidates = eqnet_candidate_grid((0.0, 40.0), (0.0, 40.0), spacing=4.0)
    centre = candidates[60]
    activation = torch.zeros(1, 121, 96)
    distance = torch.linalg.vector_norm(candidates - centre, dim=1)
    activation[0, :, 40] = 0.95 * torch.exp(-((distance / 6.0) ** 2))
    events = model.extract_events(activation, candidates, top_k=9)
    assert len(events[0]) == 1
    event = events[0][0]
    assert event["time"] == pytest.approx(40 / RATE) and event["sample"] == 40 * 32
    assert event["probability"] == pytest.approx(0.95)
    assert event["location"] == pytest.approx(centre.tolist(), abs=1e-6)

    square = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0], [2.0, 2.0], [1.0, 1.0]])
    assert geometric_median(square).tolist() == pytest.approx([1.0, 1.0], abs=1e-6)


def test_loss_is_the_sum_of_three_binary_cross_entropies():
    torch.manual_seed(1)
    outputs = {"phase": torch.randn(2, 3, 2, 16), "event": torch.randn(2, 5, 8)}
    phase_targets = torch.rand(2, 3, 2, 16)
    event_targets = torch.rand(2, 5, 8)
    expected = sum(
        F.binary_cross_entropy_with_logits(outputs["phase"][:, :, i], phase_targets[:, :, i], reduction="sum") / 6
        for i in range(2)
    ) + F.binary_cross_entropy_with_logits(outputs["event"], event_targets, reduction="sum") / 10
    torch.testing.assert_close(EQNetLoss()(outputs, phase_targets, event_targets), expected)
    weighted = EQNetLoss(weights=(1.0, 0.0, 0.0))(outputs, phase_targets, event_targets)
    torch.testing.assert_close(weighted, F.binary_cross_entropy_with_logits(outputs["phase"][:, :, 0], phase_targets[:, :, 0], reduction="sum") / 6)
    with pytest.raises(ValueError, match="shape"):
        EQNetLoss()(outputs, phase_targets[..., :8])


def test_builder_contract():
    assert isinstance(build_model("eqnet", task="picking"), EQNet)
    with pytest.raises(ValueError, match="picking"):
        build_model("eqnet", task="regression")
    with pytest.raises(TypeError, match="no longer"):
        build_model("eqnet", task="picking", hidden_dim=48)
    assert math.isclose(EQNet().feature_rate, RATE)
