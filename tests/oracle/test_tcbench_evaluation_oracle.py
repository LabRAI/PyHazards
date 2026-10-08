"""TCBench-protocol scoring checked against TCBench's own evaluation code on its released tracks.

TCBench_Alpha (https://github.com/msgomez06/TCBench_Alpha, MIT; pinned in repos.yaml) computes the
direct position error and intensity errors of each forecast row in ``dev/metrics_test.py`` (``_DPE``,
``_AE``) against IBTrACS read by ``dev/utils/toolbox.read_hist_track_file`` with the column types of
``dev/utils/constants.py`` (``LAT`` / ``LON`` as float16). Those functions are executed from the pinned
files (their modules import plotting and deep-learning packages the functions do not use, so only the
function definitions are loaded) on TCBench's released ``matched_tracks`` files for Pangu-Weather and
FourCastNet v2 with TCBench's own 2023 IBTrACS extract, after the filtering of
``dev/evaluate_tracks.py`` (00/12 UTC initialisations, storms of the year, duplicates dropped) and the
unit conversion of ``dev/track_matcher.py``. :func:`pyhazards.forecasts.forecast_track_errors` with
``protocol="tcbench"`` must give the same value for every row.
"""

from __future__ import annotations

import os
import types

import numpy as np
import pandas as pd
import pytest

from oracle_utils import import_from, load_definitions, oracle_asset, oracle_repo

from pyhazards.forecasts import forecast_track_errors, read_tcbench_matched_tracks
from pyhazards.forecasts.scoring import TCBENCH_KT_PER_MS


@pytest.fixture(scope="module")
def tcbench():
    import joblib

    dev = oracle_repo("TCBench_Alpha") / "dev"
    constants = import_from(dev / "utils", "constants")
    toolbox_defs = load_definitions(dev / "utils" / "toolbox.py", ["haversine", "read_hist_track_file"], {"np": np, "pd": pd, "os": os, "constants": constants})
    toolbox = types.SimpleNamespace(**toolbox_defs)
    namespace = {"np": np, "pd": pd, "jl": joblib, "toolbox": toolbox}
    return load_definitions(dev / "metrics_test.py", ["_DPE", "_AE"], namespace), toolbox


@pytest.mark.parametrize("asset, filename", [("tcbench_matched_pangu", "2023_PANGU.csv"), ("tcbench_matched_fcnet", "2023_fcnet.csv")])
def test_tcbench_protocol_matches_official_evaluation(tcbench, tmp_path, asset, filename):
    functions, toolbox = tcbench
    ib_dir = tmp_path / "ibtracs"
    ib_dir.mkdir()
    (ib_dir / "2023_IBTrACS.csv").write_bytes((oracle_asset("tcbench_ibtracs_2023") / "2023_IBTrACS.csv").read_bytes())
    # TCBench's 2023 extract has no units row (the NCEI files do), hence skip_rows=None.
    ibtracs = toolbox.read_hist_track_file(tracks_path=str(ib_dir), skip_rows=None)
    ibtracs = ibtracs[pd.to_datetime(ibtracs["ISO_TIME"]).dt.year == 2023]
    assert ibtracs["LAT"].dtype == np.float16

    released = pd.read_csv(oracle_asset(asset) / filename)
    # dev/track_matcher.py converts to knots and hPa before evaluation.
    tracks = released.copy()
    tracks["wind max"] = tracks["wind max"] * TCBENCH_KT_PER_MS
    tracks["pressure min"] = tracks["pressure min"] / 100
    # dev/evaluate_tracks.py filtering.
    tracks["Initial Time"] = pd.to_datetime(tracks["Initial Time"])
    tracks = tracks[tracks["Initial Time"].dt.hour.isin({0, 12})]
    tracks = tracks[tracks["SID"].isin(ibtracs["SID"].unique())]
    tracks = tracks.drop_duplicates(subset=["Initial Time", "Valid Time", "SID"], keep="first").reset_index(drop=True)
    dpe = functions["_DPE"](reference=ibtracs, predictions=tracks)
    ae = np.asarray(functions["_AE"](reference=ibtracs, predictions=tracks))

    reference = pd.read_csv(oracle_asset("tcbench_ibtracs_2023") / "2023_IBTrACS.csv", keep_default_na=False, na_values=[""])
    reference = reference[pd.to_datetime(reference["ISO_TIME"]).dt.year == 2023]
    ours = forecast_track_errors(read_tcbench_matched_tracks(oracle_asset(asset) / filename), reference, protocol="tcbench")
    assert len(ours) == len(tracks)
    assert (ours["SID"].to_numpy() == tracks["SID"].to_numpy()).all()
    np.testing.assert_allclose(ours["track_error_km"].to_numpy(), np.asarray(dpe, dtype=float), rtol=0, atol=1e-9, equal_nan=True)
    np.testing.assert_allclose(np.abs(ours["wind_error_kt"].to_numpy()), ae[:, 0].astype(float), rtol=0, atol=1e-9, equal_nan=True)
    np.testing.assert_allclose(np.abs(ours["pres_error_hpa"].to_numpy()), ae[:, 1].astype(float), rtol=0, atol=1e-9, equal_nan=True)
    assert np.isfinite(ours["track_error_km"]).sum() > 1000
