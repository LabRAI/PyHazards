from __future__ import annotations

from pyhazards.datasets._generic_inspection import run_generic_dataset_inspection


def main(argv: list[str] | None = None) -> int:
    return run_generic_dataset_inspection(
        dataset_name="wfigs",
        dataset_doc_url="https://data-nifc.opendata.arcgis.com/",
        argv=argv,
    )


if __name__ == "__main__":
    raise SystemExit(main())

