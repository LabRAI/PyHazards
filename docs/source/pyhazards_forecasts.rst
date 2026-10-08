Foundation Weather Models
=========================

Overview
--------

FourCastNet, GraphCast and Pangu-Weather are global medium-range weather models (36-75 million
parameters, 0.25-degree global fields, 6-hourly steps). They are not tropical cyclone models: their
cyclone results come from running a **tracker** on their forecast fields and scoring the tracks against
best tracks. PyHazards therefore does not list them as models and does not reimplement them. Earlier
releases registered ``fourcastnet_tc``, ``graphcast_tc`` and ``pangu_tc``, which were 15K-152K-parameter
networks with no forecast fields, weights or tracker; they were removed.

:mod:`pyhazards.forecasts` provides the pipeline instead:

#. **Forecast fields or tracks** from where they are officially published (TCBench, WeatherBench 2), or
   from the real models run through NVIDIA earth2studio, which you install yourself.
#. **Trackers**: the TempestExtremes rules TCBench uses (checked against the official binaries) and the
   ECMWF-style tracker described in the Pangu-Weather and GraphCast papers.
#. **Matching and scoring** against IBTrACS: great-circle track error in km, wind error in knots and
   pressure error in hPa per lead time (the ``tc`` benchmark's metrics), optionally computed exactly as
   TCBench's evaluation computes them.

These entries appear as ``Foundation model pipeline`` on :doc:`appendix_a_coverage`.

The models
----------

.. list-table::
   :widths: 14 30 28 28
   :header-rows: 1
   :class: dataset-list

   * - Model
     - Code and weights (licences)
     - Paper's cyclone evaluation
     - PyHazards routes
   * - FourCastNet
     - `NVlabs/FourCastNet <https://github.com/NVlabs/FourCastNet>`_ (BSD-3-Clause); 74,691,840-parameter
       AFNO; NERSC v0 weights (licence not stated). earth2studio's ``FCN`` is a 26-variable retrain
       (HF ``nvidia/fourcastnet1``, Apache-2.0 model card).
     - Qualitative only (Hurricane Michael 2018 ensemble, eye = MSLP minimum; Pathak et al. 2022, Sec. 3.1).
     - Run with earth2studio (``fourcastnet``); TCBench's 2023 tracks and fields of FourCastNet v2
       (``fourcastnet_v2``: SFNO-small, a different model). WeatherBench 2 has no FourCastNet forecasts.
   * - GraphCast
     - `google-deepmind/weathernext <https://github.com/google-deepmind/weathernext>`_ (Apache-2.0, JAX);
       36,348,131 parameters used (paper: 36.7M); weights CC BY 4.0 per the README since 2026-08-06
       (the checkpoint metadata still says CC BY-NC-SA 4.0).
     - Modified ECMWF tracker (never released) on 2018-2021 forecasts, homogeneous comparison with HRES,
       median and mean geodesic error to 5 days; numbers only in figures (Lam et al. 2023, supplement 8.1).
     - WeatherBench 2 forecasts (2018 from the paper's 1979-2017 model, and 2020); earth2studio
       (``graphcast_operational``, ``graphcast_small``); :data:`~pyhazards.forecasts.GRAPHCAST_TRACKER`.
   * - Pangu-Weather
     - `198808xc/Pangu-Weather <https://github.com/198808xc/Pangu-Weather>`_ (no code licence; ONNX
       inference only); about 64.2M learnable parameters per lead-time model; **weights CC BY-NC-SA 4.0,
       commercial use forbidden**.
     - ECMWF-style tracker on 88 named 2018 cyclones (TC2018): mean direct position error 120.29 km at
       3 days and 195.65 km at 5 days (HRES 162.28 / 272.10 km) (Bi et al., arXiv:2211.02556, Sec. 4.2.2).
     - TCBench's 2023 tracks and raw fields (``pangu``); WeatherBench 2 forecasts 2018-2022; earth2studio
       (``pangu_6``, ``pangu_24``); :data:`~pyhazards.forecasts.PANGU_TRACKER` or the TCBench rules.

The same information is available in code as :data:`pyhazards.forecasts.FOUNDATION_MODELS`. PyHazards
never downloads, bundles or redistributes model weights or forecasts; everything is read at run time
from the providers. Forecasts derived from the Pangu-Weather weights inherit their non-commercial terms.

Trackers
--------

**TempestExtremes rules (TCBench).** :func:`~pyhazards.forecasts.detect_nodes` and
:func:`~pyhazards.forecasts.stitch_nodes` implement the DetectNodes / StitchNodes options of TCBench's
``dev/TempestExtremes_example.sh`` (MSLP minima merged within 6 degrees, a 200 Pa closed MSLP contour
within 5.5 degrees, a 58.8 m2 s-2 warm-core closed contour of z300 - z500 within 6.5 degrees; nodes
linked within 8 degrees, tracks of at least 12 h with 10 m wind >= 10 m/s at two points and
\|lat\| <= 50 at one). The code is written from the published algorithm (Ullrich and Zarzycki 2017;
Ullrich et al. 2021); the official C++ tools are used only as the test oracle.

**ECMWF-style tracker (Pangu-Weather and GraphCast papers).** :func:`~pyhazards.forecasts.follow_cyclone`
follows one cyclone from its observed position: candidates are MSLP local minima within 445 km (of the
current position for Pangu-Weather, of a first guess mixing the last displacement and the 200-850 hPa
steering wind for ECMWF / GraphCast), checked for 850 hPa vorticity above 5e-5 s-1 within 278 km, a
thickness maximum when extratropical and 10 m wind above 8 m/s when over land. The GraphCast preset uses
the hyper-parameters the supplement marks in bold (222.5 km candidate radius, 208.5 km check radius,
50-50 first guess, no turn of 90 degrees or more). Neither paper released its tracker, so this is a
re-implementation of their text; what the text leaves open (the definition of "extratropical", the
land mask, grid-point positions) is a documented parameter.

Verification
------------

Run with ``tests/oracle/test_tempest_oracle.py``, ``tests/oracle/test_tcbench_evaluation_oracle.py``
and ``tests/oracle/test_forecasts_weather_oracle.py`` (oracle suites ``tc-tracking`` and ``weather``),
plus ``tests/test_forecasts_*.py``:

* **Tracker vs official TempestExtremes v2.4.2** (built from source): identical nodes, printed values
  and track files on synthetic fields with cyclones in both hemispheres, across the dateline and near
  the pole, for TCBench's options and for variants (diagonal connectivity, regional grids,
  ``--maxgap``, threshold counts).
* **Tracker vs TCBench's released tracks**: on TCBench's raw Pangu-Weather fields for the forecasts of
  2023-01-15 12 UTC, 2023-05-07 00 UTC, 2023-07-16 12 UTC and 2023-10-19 12 UTC, PyHazards writes
  TCBench's ``unmatched_tracks`` files byte for byte (so do the official binaries), and matching the
  tracks to TCBench's 2023 IBTrACS extract gives exactly TCBench's ``matched_tracks`` rows.
* **Scoring vs TCBench's own code**: on every row TCBench evaluates (10,983 of the 11,031 released
  Pangu-Weather rows, 8,335 of the 8,403 FourCastNet v2 rows), ``protocol="tcbench"`` equals TCBench's
  ``_DPE`` and ``_AE`` (``dev/metrics_test.py``) to 1e-9. This includes a TCBench quirk: it reads IBTrACS latitudes
  and longitudes as ``float16`` and evaluates the haversine formula with the half-precision latitude,
  which moves individual errors by up to several km (mean errors by about 0.1 km).
* **Following tracker**: synthetic vortices on known tracks in both hemispheres are followed within half
  a grid cell by all presets; lows without cyclonic vorticity, weak lows over land, cold-core lows
  poleward of the extratropical latitude and (GraphCast preset) reversals are rejected.
* **Sources**: TCBench field reader on files with TCBench's packed layout; the earth2studio runner with
  earth2studio's own ``Persistence`` model; WeatherBench 2 slices of the real Pangu-Weather and GraphCast
  stores locate Typhoon Kong-rey's centre within 100 km of IBTrACS.

Reproduced published numbers
----------------------------

TCBench's official notebook prints the errors of sample rows of its 2023 Pangu-Weather evaluation.
``scripts/reproduce_tcbench_tracks.py`` recomputes them end to end: raw Pangu-Weather fields from
TCBench's Hugging Face dataset, PyHazards' TempestExtremes-rule tracker, HuracanPy-rule matching,
``protocol="tcbench"`` scoring against NCEI IBTrACS v04r01.

.. list-table::
   :widths: 22 22 18 18 20
   :header-rows: 1
   :class: dataset-list

   * - Storm (SID)
     - Forecast
     - TCBench DPE (km)
     - PyHazards DPE (km)
     - Wind / pressure error
   * - Don (2023193N37305)
     - 2023-07-16 12 UTC + 84 h
     - 302.565885
     - 302.565885
     - 21.254614 kt / 9.488 hPa (as published)
   * - Norma (2023290N12256)
     - 2023-10-19 12 UTC + 12 h
     - 18.454374
     - 18.454374
     - 62.780767 kt / 39.9856 hPa (as published)
   * - Mocha (2023129N08091)
     - 2023-05-07 00 UTC + 120 h
     - 424.145817
     - 424.145817
     - 25.624799 kt / 9.5378 hPa (as published)
   * - Cheneso (2023013S08081)
     - 2023-01-15 12 UTC + 60 h
     - 216.013123
     - 206.263455 (216.013123 with TCBench's IBTrACS copy)
     - 4.770411 kt / 6.561 hPa (as published)

The Cheneso row differs only because NCEI revised the best track since TCBench's evaluation (longitude
at 2023-01-18 00 UTC 55.8 then, 55.9 now); with the position in TCBench's own IBTrACS extract the
published value is reproduced. Over the whole released 2023 sample, TCBench's protocol gives mean track
errors of 68.8 / 103.5 / 150.7 / 202.6 / 270.8 km at 1-5 days for Pangu-Weather (453-554 rows per lead,
73 storms) and 69.2 / 111.1 / 170.5 / 225.4 / 314.3 km for FourCastNet v2; TCBench itself reports these
only in figures, so they are PyHazards' computation from its released tracks, not published numbers.

**Not reproduced.** Pangu-Weather's TC2018 numbers (120.29 / 195.65 km) need its deterministic forecasts
for every initial time of 88 cyclones; the only public archive (WeatherBench 2) stores each lead time as a
full global chunk with all 13 pressure levels, about 1.8 GB per forecast for the fields the tracker
needs (41.9 MB per lead time for each of u and v), roughly a terabyte for TC2018, and the paper's
exact sample (cyclones present in both IBTrACS and HRES / TIGGE, initial times) is not published. GraphCast's numbers exist only in figures, its tracker
was never released, and its 2018-2021 forecasts at 06 / 18 UTC are not published (WeatherBench 2 has
00 / 12 UTC forecasts for 2018 and 2020). FourCastNet's paper has no aggregate cyclone numbers. As a
qualitative check on the paper case of Pangu-Weather's Fig. 1 (Typhoon Kong-rey from 2018-09-30 00 UTC),
the Pangu-Weather tracker on WeatherBench 2's Pangu-Weather forecast is 15, 39, 12, 12 and 26 km from
IBTrACS at 1-5 days (the paper describes its track as almost coinciding with the best track);
WeatherBench 2's GraphCast forecast of the same case, tracked with the GraphCast preset, is 15, 12, 33,
88 and 222 km off.

Usage
-----

Score TCBench's released tracks with TCBench's protocol:

.. code-block:: python

    from pyhazards.datasets.tc import download_ibtracs, read_ibtracs
    from pyhazards.forecasts import (download_tcbench_file, read_tcbench_matched_tracks,
                                     score_forecast_tracks, tcbench_ibtracs, tcbench_path)

    tracks = read_tcbench_matched_tracks(download_tcbench_file(tcbench_path("pangu", "matched"), "data/tcbench"))
    ibtracs = tcbench_ibtracs(read_ibtracs(download_ibtracs("data/ibtracs", subset="ALL")), year=2023)
    result = score_forecast_tracks(tracks, ibtracs, lead_hours=[24, 48, 72, 96, 120], protocol="tcbench")
    print(result.metrics["track_error_km_72h"], result.metadata["units"])

Track a forecast yourself (TCBench rules, or the papers' tracker):

.. code-block:: python

    from pyhazards.forecasts import (GRAPHCAST_TRACKER, follow_forecast_tracks, read_tcbench_fields,
                                     read_weatherbench2_forecast, required_variables,
                                     tempest_forecast_tracks)

    fields = read_tcbench_fields("pangu", "2023-07-16 12:00")          # ~200 MB of range requests
    table = tempest_forecast_tracks(fields, ibtracs, "2023-07-16 12:00")  # forecast-track table

    gc = read_weatherbench2_forecast("graphcast", "2018-09-30T00",     # needs pyhazards[weather]
                                     variables=required_variables(GRAPHCAST_TRACKER),
                                     lat_bounds=(0, 55), lon_bounds=(95, 185))
    table = follow_forecast_tracks(gc, ibtracs, "2018-09-30T00", GRAPHCAST_TRACKER)

Run a model through earth2studio (install it and the model's extra yourself; weights are downloaded by
earth2studio under their own licences; a 0.25-degree model needs a GPU with about 40 GB):

.. code-block:: python

    from earth2studio.data import ARCO
    from pyhazards.forecasts import load_earth2studio_model, run_earth2studio_forecast

    model = load_earth2studio_model("fourcastnet", device="cuda")
    fields = run_earth2studio_forecast(model, ARCO(), "2018-10-07T00", lead_hours=120, device="cuda")
    table = tempest_forecast_tracks(fields, ibtracs, "2018-10-07T00")

API
---

See :doc:`api/pyhazards.forecasts`.
