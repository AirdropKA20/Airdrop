"""Score the paper-trading log against settled Kalshi results (leak-free test).

    python -m scripts.paper_report            # actionable signals, first sighting per market
    python -m scripts.paper_report --min-edge 0.12

For each Kalshi market the FIRST logged actionable signal is treated as a $10
paper trade at the logged ask (plus fee). Also reports the model's forecast
error using Kalshi's exact settled high. Polymarket rows are skipped (no exact
settlement value via the public API).
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict
from pathlib import Path

import httpx

LOG = Path(__file__).resolve().parents[1] / "data" / "paper_log.jsonl"
KALSHI = "https://api.elections.kalshi.com/trade-api/v2"


def fee(contracts: float, price: float) -> float:
    return math.ceil(100 * 0.07 * contracts * price * (1 - price) - 1e-9) / 100


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-edge", type=float, default=None, help="default: the logged 'actionable' flag")
    args = ap.parse_args()
    if not LOG.exists():
        print("No paper log yet. Run `python -m scripts.signals` a few times a day first.")
        return

    rows = [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["platform"] == "kalshi"]
    first: dict[str, dict] = {}
    mus: dict[tuple, float] = {}
    for r in sorted(rows, key=lambda r: r["ts"]):
        mus.setdefault((r["city"], r["date"]), r["mu"])
        take = r["edge"] >= args.min_edge if args.min_edge is not None else r["actionable"]
        if take and r["market_id"] not in first:
            first[r["market_id"]] = r

    tickers = sorted({r["market_id"] for r in rows})
    results: dict[str, tuple] = {}
    with httpx.Client(timeout=20) as c:
        for t in tickers:
            for attempt in range(5):
                resp = c.get(f"{KALSHI}/markets/{t}")
                if resp.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                break
            m = resp.json().get("market", {}) if resp.status_code == 200 else {}
            if m.get("result") in ("yes", "no"):
                try:
                    ev = float(m.get("expiration_value"))
                except (TypeError, ValueError):
                    ev = None
                results[t] = (m["result"] == "yes", ev)
            time.sleep(0.15)

    errs = []
    actual_by_day = {}
    for r in rows:
        res = results.get(r["market_id"])
        if res and res[1] is not None:
            actual_by_day[(r["city"], r["date"])] = res[1]
    for k, mu in mus.items():
        if k in actual_by_day:
            errs.append(abs(mu - actual_by_day[k]))

    trades = []
    by_city = defaultdict(list)
    for t, r in first.items():
        if t not in results:
            continue
        won_yes = results[t][0]
        px = r["yes_ask"] if r["direction"] == "yes" else r["no_ask"]
        if not (0 < px < 1):
            continue
        n = 10 / px
        cost = 10 + fee(n, px)
        win = won_yes if r["direction"] == "yes" else not won_yes
        pnl = (n if win else 0) - cost
        trades.append((win, pnl, cost))
        by_city[r["city"]].append(pnl)

    print(f"Logged Kalshi signals: {len(rows)}  markets: {len(tickers)}  settled: {len(results)}")
    if errs:
        print(f"Model forecast error (first logged forecast per day): MAE {sum(errs) / len(errs):.2f}F "
              f"over {len(errs)} city-days")
    if not trades:
        print("No settled paper trades yet.")
        return
    pnl = sum(t[1] for t in trades)
    cost = sum(t[2] for t in trades)
    wins = sum(t[0] for t in trades)
    print(f"Paper trades: {len(trades)}  win {100 * wins / len(trades):.1f}%  "
          f"P&L ${pnl:+.2f}  ROI {100 * pnl / cost:+.1f}%")
    for city, v in sorted(by_city.items()):
        print(f"  {city:12s} {len(v):4d} trades  P&L ${sum(v):+.2f}")
    if len(trades) < 100:
        print("Fewer than 100 trades: too few to tell skill from luck yet.")


if __name__ == "__main__":
    main()
