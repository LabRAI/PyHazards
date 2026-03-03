from __future__ import annotations

from pyhazards.datasets.fpa_fod import inspect_fpa_fod_weekly


def main(argv: list[str] | None = None) -> int:
    return inspect_fpa_fod_weekly(argv)


if __name__ == "__main__":
    raise SystemExit(main())
