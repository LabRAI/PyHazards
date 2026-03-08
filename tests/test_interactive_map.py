from pyhazards import RAI_FIRE_URL, open_interactive_map
from pyhazards.__main__ import main


def test_open_interactive_map_returns_url_without_browser():
    assert open_interactive_map(open_browser=False) == RAI_FIRE_URL


def test_open_interactive_map_attempts_browser(monkeypatch):
    calls = []

    def fake_open(url, new=0):
        calls.append((url, new))
        return True

    monkeypatch.setattr("pyhazards.interactive_map._can_launch_browser", lambda: True)
    monkeypatch.setattr("pyhazards.interactive_map.webbrowser.open", fake_open)

    assert open_interactive_map() == RAI_FIRE_URL
    assert calls == [(RAI_FIRE_URL, 2)]


def test_open_interactive_map_swallows_browser_errors(monkeypatch):
    def fake_open(url, new=0):
        raise RuntimeError("browser unavailable")

    monkeypatch.setattr("pyhazards.interactive_map._can_launch_browser", lambda: True)
    monkeypatch.setattr("pyhazards.interactive_map.webbrowser.open", fake_open)

    assert open_interactive_map() == RAI_FIRE_URL


def test_open_interactive_map_skips_browser_when_headless(monkeypatch):
    calls = []

    def fake_open(url, new=0):
        calls.append((url, new))
        return True

    monkeypatch.setattr("pyhazards.interactive_map._can_launch_browser", lambda: False)
    monkeypatch.setattr("pyhazards.interactive_map.webbrowser.open", fake_open)

    assert open_interactive_map() == RAI_FIRE_URL
    assert calls == []


def test_cli_map_prints_url(monkeypatch, capsys):
    calls = []

    def fake_open_map(open_browser=True):
        calls.append(open_browser)
        return RAI_FIRE_URL

    monkeypatch.setattr("pyhazards.__main__.open_interactive_map", fake_open_map)

    assert main(["map"]) == 0
    out = capsys.readouterr().out

    assert "RAI Fire interactive map" in out
    assert RAI_FIRE_URL in out
    assert calls == [True]
