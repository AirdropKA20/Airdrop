"""Multi-model forecast blend with walk-forward calibration.

A re-implementation of the approach described by PolyWeather's "DEB"
(Dynamic Error Balancing) docs -- no code copied:

1. Each model's own recent signed bias at the city is removed (shrunk toward
   zero when there are few samples, so one odd day can't swing it).
2. Each bias-corrected model is weighted by the inverse of its recent,
   recency-weighted mean absolute error. (Weighting on *debiased* error means
   an accurate model with a steady, correctable offset isn't penalised.)
   Only the best model of each family (e.g. ECMWF IFS vs AIFS) is kept.
   Any residual bias of the blend is removed the same way.
3. Uncertainty (sigma) is a robust (MAD-based) estimate of recent blend
   errors, shrunk toward a prior and floored.
4. Contract probability = Normal(mu, sigma) mass over the whole-degree
   interval (+/-0.5 for rounding).

Pure functions only: callers supply history rows, so the same code runs in
the live bot and in the backtest (where history must end before the target
date -- no lookahead).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, Iterable, List, Mapping, Optional

from backend.core.weather_model import interval_probability
from backend.data.temp_intervals import TempInterval

# Models in the same family share most of their physics/initial conditions;
# keep only the best-performing member of each family.
MODEL_FAMILY: Dict[str, str] = {
    "gfs_seamless": "ncep",
    "gfs025": "ncep",
    "ncep_nbm_conus": "ncep",
    "ecmwf_ifs025": "ecmwf",
    "ecmwf_aifs025": "ecmwf",
    "icon_seamless": "dwd",
    "gem_seamless": "cmc",
    "gem_global": "cmc",
    "jma_seamless": "jma",
    "ukmo_seamless": "ukmo",
    "ukmo_global_ensemble_20km": "ukmo",
    "bom_access_global_ensemble": "bom",
}


@dataclass
class HistoryRow:
    day: date
    forecasts: Dict[str, float]  # model -> forecast settlement-day high (F)
    actual: float                # official high (F)


@dataclass
class Calibration:
    weights: Dict[str, float]          # normalised, models not listed get 0
    bias: float                        # added to the weighted blend
    sigma: float = 2.5
    model_bias: Dict[str, float] = field(default_factory=dict)  # added per model
    n: int = 0                         # history rows used
    model_mae: Dict[str, float] = field(default_factory=dict)

    def predict(self, forecasts: Mapping[str, float]) -> Optional[float]:
        """Blended, bias-corrected high. Weights are renormalised over the
        models actually present; falls back to an equal-weight mean."""
        usable = {m: f + self.model_bias.get(m, 0.0) for m, f in forecasts.items()
                  if f is not None and self.weights.get(m, 0) > 0}
        if usable:
            wsum = sum(self.weights[m] for m in usable)
            mu = sum(self.weights[m] * f for m, f in usable.items()) / wsum
        else:
            vals = [f for f in forecasts.values() if f is not None]
            if not vals:
                return None
            mu = sum(vals) / len(vals)
        return mu + self.bias

    def probability(self, forecasts: Mapping[str, float], interval: TempInterval) -> Optional[float]:
        mu = self.predict(forecasts)
        if mu is None:
            return None
        return interval_probability([mu], interval, self.sigma)


def _recency_weights(rows: List[HistoryRow], target: date, half_life_days: float) -> List[float]:
    return [0.5 ** (max((target - r.day).days, 0) / half_life_days) for r in rows]


def _weighted_median(values: List[float], weights: List[float]) -> float:
    pairs = sorted(zip(values, weights))
    total = sum(weights)
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= total / 2:
            return v
    return pairs[-1][0]


def fit_calibration(
    history: Iterable[HistoryRow],
    target: date,
    models: Iterable[str],
    *,
    lookback_days: int = 60,
    half_life_days: float = 14.0,
    min_model_samples: int = 7,
    bias_shrink_k: float = 7.0,
    sigma_prior: float = 2.5,
    sigma_prior_n: float = 10.0,
    sigma_floor: float = 1.0,
) -> Calibration:
    """Fit weights / bias / sigma from rows strictly before ``target``."""
    models = list(models)
    rows = sorted(
        (r for r in history if r.day < target and (target - r.day).days <= lookback_days),
        key=lambda r: r.day,
    )
    rw = _recency_weights(rows, target, half_life_days)

    def shrunk_mean(vals: List[float], ws: List[float]) -> float:
        n_eff = sum(ws) ** 2 / sum(w * w for w in ws)
        return (sum(v * w for v, w in zip(vals, ws)) / sum(ws)) * n_eff / (n_eff + bias_shrink_k)

    # 1) per-model shrunk bias, then recency-weighted MAE of the debiased model
    mae: Dict[str, float] = {}
    model_bias: Dict[str, float] = {}
    for m in models:
        resid, ws = [], []
        for r, w in zip(rows, rw):
            f = r.forecasts.get(m)
            if f is not None:
                resid.append(r.actual - f)
                ws.append(w)
        if len(resid) >= min_model_samples and sum(ws) > 0:
            b = shrunk_mean(resid, ws)
            model_bias[m] = b
            mae[m] = sum(abs(e - b) * w for e, w in zip(resid, ws)) / sum(ws)

    # 2) best model per family, inverse-MAE weights
    best_by_family: Dict[str, str] = {}
    for m, e in mae.items():
        fam = MODEL_FAMILY.get(m, m)
        if fam not in best_by_family or e < mae[best_by_family[fam]]:
            best_by_family[fam] = m
    chosen = list(best_by_family.values())
    if chosen:
        raw = {m: 1.0 / max(mae[m], 0.3) for m in chosen}
    else:  # cold start: equal weights over one model per family
        seen = {}
        for m in models:
            seen.setdefault(MODEL_FAMILY.get(m, m), m)
        raw = {m: 1.0 for m in seen.values()}
    total = sum(raw.values())
    weights = {m: w / total for m, w in raw.items()}

    # 3) blend residuals -> shrunk bias, robust sigma
    model_bias = {m: b for m, b in model_bias.items() if m in weights}
    unbiased = Calibration(weights=weights, bias=0.0, sigma=sigma_prior, model_bias=model_bias)
    resid, rws = [], []
    for r, w in zip(rows, rw):
        mu = unbiased.predict(r.forecasts)
        if mu is not None:
            resid.append(r.actual - mu)
            rws.append(w)

    n = len(resid)
    if n == 0:
        return Calibration(weights=weights, bias=0.0, sigma=sigma_prior, model_bias=model_bias, model_mae=mae)

    n_eff = sum(rws) ** 2 / sum(w * w for w in rws)
    bias = shrunk_mean(resid, rws)

    centred = [e - bias for e in resid]
    med = _weighted_median(centred, rws)
    mad = _weighted_median([abs(e - med) for e in centred], rws)
    s_robust = 1.4826 * mad
    sigma = math.sqrt((n_eff * s_robust ** 2 + sigma_prior_n * sigma_prior ** 2) / (n_eff + sigma_prior_n))
    sigma = max(sigma, sigma_floor)

    return Calibration(weights=weights, bias=bias, sigma=sigma, n=n,
                       model_bias=model_bias, model_mae=mae)
