Simulators
===================

Overview
--------

ForeFire and WRF-SFIRE are physics-based wildland-fire simulators, not neural networks. PyHazards
does not reimplement them and does not list them as models. Instead it offers two thin, honest
interfaces to the official software:

- :class:`~pyhazards.simulators.ForeFireSimulator` runs the official ForeFire engine, which you
  install yourself, on PyHazards rasters and returns arrival-time and burned-mask rasters.
- The ``wrf_sfire_spread`` dataset and :func:`~pyhazards.datasets.wrf_sfire.read_wrf_sfire_fire_grid`
  read the fire-grid outputs (``wrfout`` files) of WRF-SFIRE runs that you made with the official
  model. See :doc:`datasets/wrf_sfire`.

Both appear as ``External simulator`` entries on :doc:`appendix_a_coverage`.

ForeFire
--------

`ForeFire <https://github.com/forefireAPI/forefire>`_ (Filippi et al., `JOSS 2025
<https://doi.org/10.21105/joss.08680>`_; front-tracking method in Filippi et al., `SIMULATION 86(10)
<https://doi.org/10.1177/0037549709343117>`_) is a C++ discrete-event front-tracking simulator with
official Python bindings, ``pyforefire``.

**Installation.** ForeFire is GPL-3.0 and PyHazards is MIT, so ForeFire is not a PyHazards
dependency and no ForeFire code is copied into PyHazards. Install the official wheel yourself
(Linux x86_64/aarch64 and macOS; checked with 2.5.0):

.. code-block:: bash

    pip install forefire

``pyforefire`` is imported only when a simulation runs. Whoever redistributes PyHazards together
with ForeFire must follow the GPL.

**What the adapter does.** It makes the same calls as ForeFire's official Python examples: a
``FireDomain`` covering the fuel raster, the propagation model, the fuel index map
(``addIndexLayer``), optional altitude and wind layers (``addScalarLayer``), ignitions
(``startFire``) and wind changes (``trigger[wind;...]``), then ``step``. It reads back ForeFire's
burning map (``arrival_time``) and reduces it to the output grid by taking, for each output cell,
the earliest arrival time of the burning-map cells inside it. Each run happens in a fresh process by
default, because ForeFire keeps its parameters in a process-wide singleton.

**What it does not do.** It does not change ForeFire's propagation models, fuel tables or
parameters, calibrate anything, or learn anything. The physical realism of a run depends on the fuel
table, propagation model and parameters you choose.

.. code-block:: python

    import numpy as np
    from pyhazards.simulators import ForeFireSimulator

    table = open("fuels.csv").read()        # a ForeFire fuel table (';'-separated, 'Index' column)
    fuel = np.full((100, 100), 1, dtype=np.int32)   # codes from the table's Index column, north-up
    dem = np.zeros((100, 100))                      # terrain height in metres
    ignition = np.zeros((100, 100), dtype=bool)
    ignition[50, 50] = True

    sim = ForeFireSimulator(propagation_model="Rothermel", fuels_table=table)
    result = sim.run(fuel, cell_size=30.0, duration=3600.0, ignition_mask=ignition,
                     wind=(3.0, 0.0), altitude=dem)
    result.arrival_time                      # (100, 100) seconds since ignition, inf if unburned
    result.burned_masks([1200, 2400, 3600])  # (3, 100, 100) spread targets

Conventions: rasters are ``(rows, cols)`` with row 0 at the northern edge (``origin="upper"``) unless
``origin="lower"``; the domain is ``cols * dx`` by ``rows * dy`` metres with its south-west corner
at ``(0, 0)``; ``wind`` is ``(u, v)`` in m/s; times are seconds.

**Verification** (``tests/oracle/test_forefire_oracle.py``, ForeFire 2.5.0): given the rasters of
ForeFire's own regression case ``tests/runff`` (a 64 km Corsican landscape, Rothermel, three
ignitions and a wind change), the adapter reproduces the reference arrival-time map
``ForeFire.0.nc.ref`` within the tolerance of ForeFire's ``compare_nc.py``, and it reproduces the
official ``tests/python/idealizedwind.py`` run bit for bit.

WRF-SFIRE
---------

`WRF-SFIRE <https://github.com/openwfm/WRF-SFIRE>`_ (Mandel, Beezley and Kochanski, `GMD 2011
<https://doi.org/10.5194/gmd-4-591-2011>`_; Mandel et al., `NHESS 2014
<https://doi.org/10.5194/nhess-14-2829-2014>`_) couples the WRF atmosphere model with the SFIRE
level-set fire-spread model. It is a Fortran/MPI model built per machine and run on clusters, so a
PyHazards object cannot run it. PyHazards reads what it writes:

.. code-block:: python

    from pyhazards.datasets.wrf_sfire import read_wrf_sfire_fire_grid

    grid = read_wrf_sfire_fire_grid("run/wrfout_d01_*", variables=("TIGN_G", "LFN", "FIRE_AREA"))
    grid.arrival_time()   # TIGN_G where burned, inf elsewhere (s since simulation start)
    grid.burned_masks()   # (time, rows, cols), LFN < 0

The variable names, the fire-subgrid layout (refinement ``sr_x`` by ``sr_y`` with padding rows and
columns) and the meaning of ``TIGN_G`` and ``LFN`` follow the official ``Registry/registry.fire``
and SFIRE source of release ``W4.4-S0.1``, and were checked on a real history file of the official
ideal case ``test/em_fire/hill`` (its header is kept in
``tests/fixtures/wrf_sfire_hill_wrfout_header.json``). To produce ``wrfout`` files, build WRF-SFIRE from the
official repository (ideal cases in ``test/em_fire``) or use the official real-data workflow
`wrfxpy <https://github.com/openwfm/wrfxpy>`_.

API
---

See :doc:`api/pyhazards.simulators`.
