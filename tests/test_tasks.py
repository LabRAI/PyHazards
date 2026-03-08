import pytest

from pyhazards.tasks import available_hazard_tasks, get_hazard_task, has_hazard_task


def test_available_hazard_tasks_contains_wave_one_targets():
    names = available_hazard_tasks()
    assert "earthquake.picking" in names
    assert "wildfire.spread" in names
    assert "flood.streamflow" in names
    assert "tc.track_intensity" in names


def test_get_hazard_task_returns_structured_record():
    task = get_hazard_task("wildfire.spread")
    assert task.hazard == "wildfire"
    assert task.target == "spread"
    assert "burned-area" in task.description


def test_unknown_hazard_task_raises():
    assert not has_hazard_task("unknown.task")
    with pytest.raises(KeyError):
        get_hazard_task("unknown.task")
