"""Tests for the multi-model blend (backend/core/blend.py)."""
import random
from datetime import date, timedelta

import pytest

from backend.core.blend import HistoryRow, fit_calibration
from backend.data.temp_intervals import TempInterval

T0 = date(2026, 6, 1)


def _history(n=40, seed=1):
    """Model A: accurate but 2F cold. Model B: noisy. Model C: same family as A, worse."""
    rnd = random.Random(seed)
    rows = []
    for i in range(n):
        actual = 70 + 8 * rnd.random()
        rows.append(HistoryRow(
            T0 - timedelta(days=n - i),
            {"gfs_seamless": actual - 2 + rnd.gauss(0, 0.7),
             "icon_seamless": actual + rnd.gauss(0, 3.0),
             "ncep_nbm_conus": actual - 2 + rnd.gauss(0, 2.0)},
            actual,
        ))
    return rows


def test_weights_favour_accurate_model_and_dedupe_family():
    cal = fit_calibration(_history(), T0, ["gfs_seamless", "icon_seamless", "ncep_nbm_conus"])
    assert "ncep_nbm_conus" not in cal.weights          # same family as GFS, worse -> dropped
    assert cal.weights["gfs_seamless"] > cal.weights["icon_seamless"]
    assert sum(cal.weights.values()) == pytest.approx(1.0)


def test_bias_is_learned_and_sigma_reasonable():
    cal = fit_calibration(_history(), T0, ["gfs_seamless", "icon_seamless"])
    assert 1.4 < cal.model_bias["gfs_seamless"] < 2.3   # corrects most of GFS's 2F cold bias
    assert abs(cal.bias) < 0.5                          # little left over after per-model correction
    assert 1.0 <= cal.sigma < 2.6
    # prediction for a day when GFS says 70 and ICON says 72 should sit near 72
    assert 71.0 < cal.predict({"gfs_seamless": 70.0, "icon_seamless": 72.0}) < 72.6


def test_no_lookahead():
    rows = _history()
    future = HistoryRow(T0, {"gfs_seamless": 0.0}, 999.0)   # absurd row on target day
    a = fit_calibration(rows, T0, ["gfs_seamless", "icon_seamless"])
    b = fit_calibration(rows + [future], T0, ["gfs_seamless", "icon_seamless"])
    assert a.bias == b.bias and a.sigma == b.sigma and a.weights == b.weights and a.model_bias == b.model_bias


def test_cold_start_equal_weights_and_prior_sigma():
    cal = fit_calibration([], T0, ["gfs_seamless", "ecmwf_ifs025", "ecmwf_aifs025"])
    assert set(cal.weights) == {"gfs_seamless", "ecmwf_ifs025"}
    assert cal.bias == 0 and cal.sigma == pytest.approx(2.5)


def test_probability_over_bucket_set_sums_to_one():
    cal = fit_calibration(_history(), T0, ["gfs_seamless", "icon_seamless"])
    f = {"gfs_seamless": 72.0, "icon_seamless": 75.0}
    buckets = [TempInterval(None, 67)] + [TempInterval(t, t + 1) for t in range(68, 80, 2)] + [TempInterval(80, None)]
    assert sum(cal.probability(f, b) for b in buckets) == pytest.approx(1.0, abs=1e-9)
