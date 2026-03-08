"""Helpers for the external RAI Fire interactive map."""

from __future__ import annotations

import os
import sys
import webbrowser


#: Canonical URL for the external RAI Fire interactive map.
RAI_FIRE_URL: str = "https://rai-fire.com/"


def _can_launch_browser() -> bool:
    if sys.platform.startswith("linux"):
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return True


def open_interactive_map(open_browser: bool = True) -> str:
    """Open the RAI Fire map in the user's browser when possible.

    Args:
        open_browser: Whether to attempt to open the default browser.

    Returns:
        The canonical RAI Fire URL.
    """

    if open_browser and _can_launch_browser():
        try:
            webbrowser.open(RAI_FIRE_URL, new=2)
        except Exception:
            # Headless and restricted environments should still get the URL.
            pass
    return RAI_FIRE_URL


__all__ = ["RAI_FIRE_URL", "open_interactive_map"]
