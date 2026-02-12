from __future__ import annotations

from pyhazards.datasets.inspection import main as merra2_main


def main(argv: list[str] | None = None) -> int:
    return merra2_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

