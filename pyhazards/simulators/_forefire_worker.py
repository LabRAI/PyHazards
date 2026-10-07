"""Worker process for :class:`pyhazards.simulators.ForeFireSimulator` with ``isolate=True``."""

import sys

from pyhazards.simulators.forefire import worker_main

if __name__ == "__main__":
    raise SystemExit(worker_main(sys.argv[1:]))
