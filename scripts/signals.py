"""Print current weather-market signals (paper only: places no orders).

    python -m scripts.signals                  # all cities, Kalshi + Polymarket
    python -m scripts.signals --cities nyc,chicago --platform kalshi
    python -m scripts.signals --all            # include non-actionable rows
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

from backend.config import settings
from backend.core.weather_signals import generate_weather_signal
from backend.data.kalshi_markets import fetch_kalshi_weather_markets
from backend.data.weather_markets import fetch_polymarket_weather_markets


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cities", default=settings.WEATHER_CITIES)
    ap.add_argument("--platform", choices=["kalshi", "polymarket", "both"], default="both")
    ap.add_argument("--all", action="store_true", help="show every market, not just actionable ones")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--no-log", action="store_true", help="don't append to data/paper_log.jsonl")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    cities = [c.strip() for c in args.cities.split(",") if c.strip()]
    markets = []
    if args.platform in ("kalshi", "both"):
        markets += await fetch_kalshi_weather_markets(cities)
    if args.platform in ("polymarket", "both"):
        markets += await fetch_polymarket_weather_markets(cities)
    print(f"{len(markets)} markets. Model: {settings.WEATHER_MODEL}. "
          f"Min edge: {settings.WEATHER_MIN_EDGE_THRESHOLD:.0%}\n")

    signals = []
    for m in markets:
        try:
            s = await generate_weather_signal(m)
        except Exception as e:  # keep going if one market fails
            logging.warning(f"{m.market_id}: {e!r}")
            continue
        if s:
            signals.append(s)

    if not args.no_log:
        _log(signals)
    signals.sort(key=lambda s: s.edge, reverse=True)
    shown = [s for s in signals if args.all or s.passes_threshold]
    if not shown:
        print("No actionable signals right now.")
    for s in shown:
        m = s.market
        flag = "TRADE" if s.passes_threshold else "     "
        price = m.yes_price if s.direction == "yes" else m.no_price
        print(f"{flag} {m.platform:10s} {m.city_name[:13]:13s} {m.target_date} {m.interval.label():>7}  "
              f"model {s.model_probability:4.0%}  buy {s.direction.upper():3s} @ {price:.2f}  "
              f"edge {s.edge:+5.1%}  size ${s.suggested_size:3.0f}  [{s.sources[0]}]")
    print(f"\n{sum(s.passes_threshold for s in signals)} actionable of {len(signals)} signals. "
          "Paper only: nothing was traded.")


LOG = Path(__file__).resolve().parents[1] / "data" / "paper_log.jsonl"


def _log(signals) -> None:
    """Append every signal to the paper-trading log (scored by scripts.paper_report)."""
    LOG.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with LOG.open("a") as f:
        for s in signals:
            m = s.market
            f.write(json.dumps({
                "ts": now, "platform": m.platform, "market_id": m.market_id, "slug": m.slug,
                "city": m.city_key, "date": m.target_date.isoformat(),
                "lo": m.interval.lo, "hi": m.interval.hi,
                "p": round(s.model_probability, 4), "mu": round(s.ensemble_mean, 2),
                "sigma": round(s.ensemble_std, 2), "yes_ask": m.yes_price, "no_ask": m.no_price,
                "direction": s.direction, "edge": round(s.edge, 4),
                "actionable": s.passes_threshold, "source": s.sources[0],
            }) + "\n")


if __name__ == "__main__":
    asyncio.run(main())
