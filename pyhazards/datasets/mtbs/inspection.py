from __future__ import annotations

from pyhazards.datasets._generic_inspection import run_generic_dataset_inspection


def main(argv: list[str] | None = None) -> int:
    return run_generic_dataset_inspection(
        dataset_name="mtbs",
        dataset_doc_url="https://burnseverity.cr.usgs.gov/",
        argv=argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())

