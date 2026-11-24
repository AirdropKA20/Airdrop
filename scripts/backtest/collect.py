"""Collect data for the Kalshi daily-high backtest.

For each city and settlement date in range it caches (JSON, resumable):
  * every market in the event: strike interval, result, exact official high
    (expiration_value)
  * hourly yes_bid / yes_ask candles over the market's life
  * archived Open-Meteo forecasts from several models, reduced to each
    model's settlement-day (local standard time) high at the station

Usage:
    python -m scripts.backtest.collect --start 2025-12-01 --end 2026-11-22
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx

from backend.data.kalshi_markets import CITY_SERIES, _parse_kalshi_date
from backend.data.temp_intervals import interval_from_kalshi
from backend.data.weather import CITY_CONFIG, STATIONS, extract_member_extremes, settlement_window

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"
HIST_FORECAST = "https://historical-forecast-api.open-meteo.com/v1/forecast"
MODELS = [
    "gfs_seamless", "ecmwf_ifs025", "ecmwf_aifs025", "icon_seamless",
    "gem_seamless", "jma_seamless", "ukmo_seamless",
]
CACHE = Path(__file__).resolve().parents[2] / "data" / "backtest_cache"


class Kalshi:
    def __init__(self, min_interval: float = 0.15):
        self.c = httpx.Client(timeout=30)
        self.min_interval = min_interval
        self._last = 0.0

    def get(self, path: str, params: dict | None = None) -> dict:
        for attempt in range(10):
            wait = self.min_interval - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.time()
            try:
                r = self.c.get(KALSHI + path, params=params)
            except httpx.HTTPError:
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(min(2 ** attempt, 30))
                continue
            if r.status_code == 404:
                return {}
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"Kalshi GET failed repeatedly: {path}")


def _ts(s: str) -> int:
    return int(datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp())


def _price(block: dict | None) -> float | None:
    if not block:
        return None
    v = block.get("close_dollars", block.get("close"))
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def settled_events(k: Kalshi, series: str, start: date, end: date) -> list[str]:
    out, cursor = [], None
    while True:
        p = {"series_ticker": series, "status": "settled", "limit": 200}
        if cursor:
            p["cursor"] = cursor
        d = k.get("/events", p)
        for e in d.get("events", []):
            t = e["event_ticker"]
            dt = _parse_kalshi_date(t + "-X")
            if t.startswith("KX") and dt and start <= dt <= end:
                out.append(t)
        cursor = d.get("cursor")
        if not cursor or not d.get("events"):
            return sorted(out)


def collect_event(k: Kalshi, series: str, event: str, cutoff_ts: int) -> dict:
    d = k.get("/markets", {"event_ticker": event})
    markets, historical = d.get("markets", []), False
    if not markets:
        markets, historical = k.get("/historical/markets", {"event_ticker": event}).get("markets", []), True

    out = []
    for m in markets:
        iv = interval_from_kalshi(m)
        if iv is None or m.get("result") not in ("yes", "no"):
            continue
        start, end = _ts(m["open_time"]), _ts(m["close_time"])
        path = (f"/historical/markets/{m['ticker']}/candlesticks" if historical
                else f"/series/{series}/markets/{m['ticker']}/candlesticks")
        cd = k.get(path, {"start_ts": start - 3600, "end_ts": end, "period_interval": 60})
        candles = [
            [c["end_period_ts"], _price(c.get("yes_bid")), _price(c.get("yes_ask"))]
            for c in cd.get("candlesticks", [])
        ]
        out.append({
            "ticker": m["ticker"], "lo": iv.lo, "hi": iv.hi, "result": m["result"],
            "expiration_value": m.get("expiration_value"),
            "open_ts": start, "close_ts": end, "candles": candles,
        })
    return {"event": event, "markets": out}


def collect_forecasts(city: str, start: date, end: date) -> dict:
    st = STATIONS[CITY_CONFIG[city]["station"]]
    out: dict[str, dict[str, float]] = {}
    with httpx.Client(timeout=90) as c:
        for model in MODELS:
            r = c.get(HIST_FORECAST, params={
                "latitude": st["lat"], "longitude": st["lon"], "hourly": "temperature_2m",
                "temperature_unit": "fahrenheit", "timezone": "GMT", "models": model,
                "start_date": start.isoformat(), "end_date": (end + timedelta(days=1)).isoformat(),
            })
            if r.status_code != 200:
                print(f"  {city} {model}: HTTP {r.status_code}")
                continue
            hourly = r.json().get("hourly", {})
            per = {}
            d = start
            while d <= end:
                s, e = settlement_window(city, d, "kalshi")
                highs, _ = extract_member_extremes(hourly, s, e)
                if highs:
                    per[d.isoformat()] = highs[0]
                d += timedelta(days=1)
            out[model] = per
            time.sleep(0.5)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=date.fromisoformat, required=True)
    ap.add_argument("--end", type=date.fromisoformat, required=True)
    ap.add_argument("--cities", default=",".join(CITY_SERIES))
    args = ap.parse_args()

    k = Kalshi()
    cutoff = k.get("/historical/cutoff").get("market_settled_ts", "1970-01-01T00:00:00Z")
    cutoff_ts = _ts(cutoff)

    for city in args.cities.split(","):
        series = CITY_SERIES[city]
        cdir = CACHE / city
        cdir.mkdir(parents=True, exist_ok=True)

        fpath = cdir / "forecasts.json"
        if not fpath.exists():
            print(f"{city}: forecasts")
            fpath.write_text(json.dumps(collect_forecasts(city, args.start, args.end)))

        events = settled_events(k, series, args.start, args.end)
        todo = [e for e in events if not (cdir / f"{e}.json").exists()]
        print(f"{city}: {len(events)} events, {len(todo)} to fetch", flush=True)
        for i, ev in enumerate(todo, 1):
            try:
                data = collect_event(k, series, ev, cutoff_ts)
            except Exception as e:  # skip this day; a rerun will retry it
                print(f"  {city} {ev}: skipped ({e})", flush=True)
                time.sleep(30)
                continue
            (cdir / f"{ev}.json").write_text(json.dumps(data))
            if i % 25 == 0:
                print(f"  {city} {i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
