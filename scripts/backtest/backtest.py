"""Walk-forward backtest of weather models against Kalshi daily-high markets.

For every settled event (city, day D):
  * calibrate the blend on days strictly before D (no outcome lookahead)
  * price every bucket at the entry time using the real hourly yes_bid/yes_ask
  * buy YES at the ask or NO at (1 - bid) when model edge after fees clears
    the threshold; $10 per trade, settled on the real result

IMPORTANT CAVEAT: forecasts come from Open-Meteo's historical-forecast archive,
which stitches together the first hours of successive model runs. For a
given day it therefore contains information from runs issued *during* that
day, i.e. somewhat more than a trader would have had at the entry time. The
results are an optimistic upper bound on what the model could have done.

Usage:
    python -m scripts.backtest.backtest [--entry-hour 8] [--entry-day-offset 0]
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from statistics import mean

from backend.core.blend import HistoryRow, fit_calibration
from backend.core.weather_model import interval_probability
from backend.data.kalshi_markets import _parse_kalshi_date
from backend.data.temp_intervals import TempInterval
from backend.data.weather import CITY_CONFIG

CACHE = Path(__file__).resolve().parents[2] / "data" / "backtest_cache"
STAKE = 10.0
WARMUP_DAYS = 21


def kalshi_fee(contracts: float, price: float, rate: float = 0.07) -> float:
    """Kalshi taker fee for an order, rounded up to the cent."""
    return math.ceil(100 * rate * contracts * price * (1 - price) - 1e-9) / 100


@dataclass
class Bucket:
    ticker: str
    interval: TempInterval
    won: bool
    bid: float | None
    ask: float | None


def quote_at(candles: list, ts: int, max_age_s: int = 6 * 3600):
    """Latest hourly close at or before ts (bid, ask)."""
    best = None
    for end_ts, bid, ask in candles:
        if end_ts <= ts and ts - end_ts <= max_age_s:
            best = (bid, ask)
    return best or (None, None)


NOISE_SD = 0.0  # extra forecast error (F) to undo the archive's lookahead; see --noise-sd


def _perturb(city: str, day: date, forecasts: dict) -> dict:
    """Add a shared daily error plus smaller per-model errors (deterministic seed)."""
    if NOISE_SD <= 0:
        return forecasts
    import random
    rnd = random.Random(f"{city}:{day.isoformat()}")
    common = rnd.gauss(0, NOISE_SD)
    return {m: f + common + rnd.gauss(0, NOISE_SD / 2) for m, f in sorted(forecasts.items())}


def load_city(city: str):
    cdir = CACHE / city
    fc = json.loads((cdir / "forecasts.json").read_text())
    models = [m for m, v in fc.items() if v]
    events = []
    for p in sorted(cdir.glob("KX*.json")):
        ev = json.loads(p.read_text())
        ms = ev["markets"]
        if not ms:
            continue
        day = _parse_kalshi_date(ev["event"] + "-X")
        try:
            actual = float(ms[0]["expiration_value"])
        except (TypeError, ValueError):
            continue
        forecasts = _perturb(city, day, {m: fc[m][day.isoformat()] for m in models if day.isoformat() in fc[m]})
        events.append((day, actual, forecasts, ms))
    return models, events


def evaluate(entry_hour: int, entry_day_offset: int, thresholds: list[float]):
    results = {name: {"ll": [], "trades": defaultdict(list), "city_trades": defaultdict(list)}
               for name in ("market", "gfs_only", "blend")}
    mae = defaultdict(list)
    n_events = 0

    for city in sorted(p.name for p in CACHE.iterdir() if p.is_dir()):
        models, events = load_city(city)
        history = [HistoryRow(d, f, a) for d, a, f, _ in events]
        if not events:
            continue
        first_day = events[0][0]
        lst = timezone(timedelta(hours=CITY_CONFIG[city]["lst_utc_offset"]))

        for day, actual, forecasts, ms in events:
            if (day - first_day).days < WARMUP_DAYS or "gfs_seamless" not in forecasts:
                continue
            entry_dt = datetime.combine(day + timedelta(days=entry_day_offset), datetime.min.time(), tzinfo=lst) \
                + timedelta(hours=entry_hour)
            ets = int(entry_dt.timestamp())

            buckets = []
            for m in ms:
                if not (m["open_ts"] <= ets < m["close_ts"]):
                    continue
                bid, ask = quote_at(m["candles"], ets)
                buckets.append(Bucket(m["ticker"], TempInterval(m["lo"], m["hi"]), m["result"] == "yes", bid, ask))
            if len(buckets) < 2 or not any(b.won for b in buckets):
                continue
            n_events += 1

            cal = fit_calibration(history, day, models)
            mu = cal.predict(forecasts)
            mae["gfs_only"].append(abs(forecasts["gfs_seamless"] - actual))
            mae["blend"].append(abs(mu - actual))

            probs = {
                "gfs_only": [interval_probability([forecasts["gfs_seamless"]], b.interval, 2.0) for b in buckets],
                "blend": [interval_probability([mu], b.interval, cal.sigma) for b in buckets],
            }
            mids = [((b.bid or 0) + (b.ask or 0)) / 2 if b.ask else 0.0 for b in buckets]
            s = sum(mids)
            probs["market"] = [x / s for x in mids] if s > 0 else [1 / len(buckets)] * len(buckets)

            for name, ps in probs.items():
                pw = next(p for p, b in zip(ps, buckets) if b.won)
                results[name]["ll"].append(-math.log(min(max(pw, 1e-3), 1.0)))
                if name == "market":
                    continue
                for b, p in zip(buckets, ps):
                    cands = []
                    if b.ask and 0.02 <= b.ask <= 0.98:
                        c = STAKE / b.ask
                        cost = STAKE + kalshi_fee(c, b.ask)
                        cands.append((p - cost / c, c, cost, b.won, b.ask))
                    if b.bid and 0.02 <= b.bid <= 0.98:
                        no_px = 1 - b.bid
                        c = STAKE / no_px
                        cost = STAKE + kalshi_fee(c, no_px)
                        cands.append(((1 - p) - cost / c, c, cost, not b.won, no_px))
                    if not cands:
                        continue
                    edge, c, cost, win, px = max(cands)
                    pnl = (c if win else 0.0) - cost
                    for t in thresholds:
                        if edge >= t:
                            results[name]["trades"][t].append((edge, pnl, win, cost, px))
                            results[name]["city_trades"][(t, city)].append(pnl)
    return results, mae, n_events


def summarize(trades):
    if not trades:
        return "   0 trades"
    pnl = sum(t[1] for t in trades)
    cost = sum(t[3] for t in trades)
    wins = sum(1 for t in trades if t[2])
    return (f"{len(trades):5d} trades  win {100 * wins / len(trades):5.1f}%  "
            f"P&L ${pnl:+9.2f}  ROI {100 * pnl / cost:+6.1f}%  avg entry {mean(t[4] for t in trades):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry-hour", type=int, default=8, help="local standard time hour")
    ap.add_argument("--entry-day-offset", type=int, default=0, help="0 = settlement day, -1 = day before")
    ap.add_argument("--noise-sd", type=float, default=0.0,
                    help="extra forecast error in F (shared daily + half per model) to offset the "
                         "archive's lookahead; ~1.5 makes errors realistic for next-day forecasts")
    args = ap.parse_args()
    global NOISE_SD
    NOISE_SD = args.noise_sd
    thresholds = [0.03, 0.05, 0.08, 0.12, 0.20]

    results, mae, n = evaluate(args.entry_hour, args.entry_day_offset, thresholds)
    day = "settlement day" if args.entry_day_offset == 0 else f"day {args.entry_day_offset:+d}"
    print(f"Entry: {args.entry_hour:02d}:00 local standard time, {day}.  Extra forecast noise: "
          f"{args.noise_sd:.1f}F.  Events: {n}")
    print("\nForecast error (MAE of daily high, F):")
    for k, v in mae.items():
        print(f"  {k:9s} {mean(v):.2f}")
    print("\nLog-loss on the winning bucket (lower is better):")
    for k in ("market", "gfs_only", "blend"):
        print(f"  {k:9s} {mean(results[k]['ll']):.3f}")
    for k in ("gfs_only", "blend"):
        print(f"\nTrading with {k} ($10/trade, real ask + fees):")
        for t in thresholds:
            print(f"  edge >= {t:.2f}: {summarize(results[k]['trades'][t])}")
    print("\nBlend P&L by city at edge >= 0.08:")
    cities = sorted({c for (t, c) in results["blend"]["city_trades"] if t == 0.08})
    for c in cities:
        v = results["blend"]["city_trades"][(0.08, c)]
        print(f"  {c:12s} {len(v):4d} trades  P&L ${sum(v):+8.2f}")


if __name__ == "__main__":
    main()
