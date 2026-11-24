"""History for live blend calibration: past model forecasts + official highs.

* Model forecasts: Open-Meteo historical-forecast archive (the same source the
  backtest uses; it also serves the latest forecast for upcoming days).
* Kalshi actuals: ``expiration_value`` of recently settled markets (exact
  official high, public API).
* Polymarket actuals: winning bucket of past daily events (midpoint of the
  2F range; Polymarket settles on different stations from Kalshi).
"""
from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import httpx

from backend.core.blend import Calibration, HistoryRow, fit_calibration
from backend.data.temp_intervals import parse_temp_interval
from backend.data.weather import CITY_CONFIG, STATIONS, extract_member_extremes, settlement_window, station_for

logger = logging.getLogger("trading_bot")

HIST_FORECAST = "https://historical-forecast-api.open-meteo.com/v1/forecast"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
BLEND_MODELS = ["gfs_seamless", "ecmwf_ifs025", "icon_seamless", "gem_seamless", "jma_seamless", "ukmo_seamless"]
LOOKBACK_DAYS = 60

_cache: Dict[str, tuple] = {}


def _cached(key: str, ttl: float):
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < ttl:
        return hit[1]
    return None


async def _fetch_hourly(city_key: str, platform: str, start: date, end: date, model: str) -> dict:
    key = f"hr:{city_key}:{platform}:{start}:{end}:{model}"
    hit = _cached(key, 1800)
    if hit is not None:
        return hit
    st = STATIONS[station_for(city_key, platform)]
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.get(HIST_FORECAST, params={
            "latitude": st["lat"], "longitude": st["lon"], "hourly": "temperature_2m",
            "temperature_unit": "fahrenheit", "timezone": "GMT", "models": model,
            "start_date": (start - timedelta(days=1)).isoformat(),
            "end_date": (end + timedelta(days=1)).isoformat(),
        })
        r.raise_for_status()
        hourly = r.json().get("hourly", {})
    _cache[key] = (time.time(), hourly)
    return hourly


async def fetch_model_highs(
    city_key: str, start: date, end: date, platform: str = "kalshi",
    models: Optional[List[str]] = None, after_utc: Optional[datetime] = None,
) -> Dict[str, Dict[str, float]]:
    """{model: {iso_date: settlement-day high F}} at the platform's station.

    With ``after_utc``, each day's value is the highest forecast temperature
    in the part of the settlement window *after* that time (used intraday:
    "how much warmer can it still get?"). Days with no hours left are omitted.
    """
    models = models or BLEND_MODELS
    out: Dict[str, Dict[str, float]] = {}
    for model in models:
        try:
            hourly = await _fetch_hourly(city_key, platform, start, end, model)
        except Exception as e:
            logger.warning(f"Model history fetch failed ({city_key} {model}): {e!r}")
            continue
        per = {}
        d = start
        while d <= end:
            s, e = settlement_window(city_key, d, platform)
            if after_utc is not None:
                # next full hour after now, so the current partial hour isn't double counted
                s = max(s, after_utc.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1))
            if s < e:
                highs, _ = extract_member_extremes(hourly, s, e, min_hours=1 if after_utc else None)
                if highs:
                    per[d.isoformat()] = highs[0]
            d += timedelta(days=1)
        out[model] = per
    return out


async def kalshi_actuals(city_key: str) -> Dict[str, float]:
    from backend.data.kalshi_markets import CITY_SERIES, _parse_kalshi_date

    key = f"ka:{city_key}"
    hit = _cached(key, 6 * 3600)
    if hit is not None:
        return hit
    out: Dict[str, float] = {}
    async with httpx.AsyncClient(timeout=30) as client:
        r = await client.get(f"{KALSHI}/markets", params={
            "series_ticker": CITY_SERIES[city_key], "status": "settled", "limit": 1000})
        r.raise_for_status()
        for m in r.json().get("markets", []):
            d = _parse_kalshi_date(m.get("ticker", ""))
            try:
                v = float(m.get("expiration_value"))
            except (TypeError, ValueError):
                continue
            if d:
                out[d.isoformat()] = v
    _cache[key] = (time.time(), out)
    return out


async def polymarket_actuals(city_key: str, days: int = LOOKBACK_DAYS) -> Dict[str, float]:
    from backend.data.weather_markets import _polymarket_event_slug

    key = f"pa:{city_key}"
    hit = _cached(key, 6 * 3600)
    if hit is not None:
        return hit
    out: Dict[str, float] = {}
    today = date.today()
    async with httpx.AsyncClient(timeout=20) as client:
        for i in range(1, days + 1):
            d = today - timedelta(days=i)
            try:
                r = await client.get("https://gamma-api.polymarket.com/events",
                                     params={"slug": _polymarket_event_slug(city_key, d)})
                events = r.json()
            except Exception:
                continue
            for ev in events or []:
                for m in ev.get("markets", []):
                    prices = m.get("outcomePrices")
                    if isinstance(prices, str):
                        try:
                            prices = json.loads(prices)
                        except ValueError:
                            continue
                    if not prices or str(prices[0]) not in ("1", "1.0"):
                        continue
                    iv = parse_temp_interval(m.get("question", ""))
                    if iv is None:
                        continue
                    if iv.lo is not None and iv.hi is not None:
                        out[d.isoformat()] = (iv.lo + iv.hi) / 2
                    else:  # open-ended tail: best available point estimate
                        out[d.isoformat()] = float(iv.lo if iv.lo is not None else iv.hi)
    _cache[key] = (time.time(), out)
    return out


async def get_calibration(city_key: str, target: date, platform: str = "kalshi") -> Optional[Calibration]:
    key = f"cal:{city_key}:{platform}:{target}"
    hit = _cached(key, 6 * 3600)
    if hit is not None:
        return hit
    try:
        actuals = await (kalshi_actuals(city_key) if platform == "kalshi" else polymarket_actuals(city_key))
        start = target - timedelta(days=LOOKBACK_DAYS)
        fc = await fetch_model_highs(city_key, start, target - timedelta(days=1), platform)
    except Exception as e:
        logger.warning(f"Calibration history unavailable for {city_key}/{platform}: {e}")
        return None

    rows = []
    for ds, actual in actuals.items():
        d = date.fromisoformat(ds)
        if d >= target:
            continue
        f = {m: v[ds] for m, v in fc.items() if ds in v}
        if f:
            rows.append(HistoryRow(d, f, actual))
    cal = fit_calibration(rows, target, BLEND_MODELS)
    logger.info(f"Blend calibration {city_key}/{platform}: n={cal.n} bias={cal.bias:+.2f}F "
                f"sigma={cal.sigma:.2f}F weights={ {m: round(w, 2) for m, w in cal.weights.items()} }")
    _cache[key] = (time.time(), cal)
    return cal
