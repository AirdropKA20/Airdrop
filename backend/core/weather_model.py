"""Pure helpers for the weather model: settlement-day windows and interval
probabilities from an ensemble forecast. No network or heavy dependencies."""
from __future__ import annotations

import math
from datetime import date, datetime, time, timedelta, timezone
from typing import Iterable, Optional, Tuple
from zoneinfo import ZoneInfo

from backend.data.temp_intervals import TempInterval


def settlement_window_utc(
    target_date: date,
    lst_utc_offset_hours: int,
    iana_tz: Optional[str] = None,
    platform: str = "kalshi",
) -> Tuple[datetime, datetime]:
    """UTC [start, end) of the day a market settles on.

    * Kalshi settles on the NWS Daily Climate Report, which always covers
      midnight-to-midnight *local standard time*, even during daylight saving
      (so in summer the "day" runs 1am-1am clock time).
    * Polymarket resolves on the Weather Underground daily history, which uses
      the local civil (DST-aware) calendar day.
    """
    if platform == "polymarket" and iana_tz:
        tz = ZoneInfo(iana_tz)
        start = datetime.combine(target_date, time.min, tzinfo=tz)
        end = datetime.combine(target_date + timedelta(days=1), time.min, tzinfo=tz)
        return start.astimezone(timezone.utc), end.astimezone(timezone.utc)

    lst = timezone(timedelta(hours=lst_utc_offset_hours))
    start = datetime.combine(target_date, time.min, tzinfo=lst)
    return start.astimezone(timezone.utc), (start + timedelta(days=1)).astimezone(timezone.utc)


def _phi(x: float) -> float:
    if x == math.inf:
        return 1.0
    if x == -math.inf:
        return 0.0
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _round_half_up(x: float) -> int:
    return math.floor(x + 0.5)


def interval_probability(
    members: Iterable[float],
    interval: TempInterval,
    sigma_f: float,
) -> Optional[float]:
    """Probability that the officially reported (whole-degree) temperature
    lands in ``interval``.

    Each ensemble member is "dressed" with a Gaussian of width ``sigma_f``.
    This turns 31 discrete members into a smooth distribution (so 2-degree
    buckets don't jump in 1/31 steps) and partly corrects for raw GFS ensembles
    being under-dispersed. The +/-0.5 edges account for rounding to whole
    degrees. sigma_f <= 0 falls back to plain member counting.
    """
    members = [float(m) for m in members]
    if not members:
        return None

    upper = math.inf if interval.hi is None else interval.hi + 0.5
    lower = -math.inf if interval.lo is None else interval.lo - 0.5

    if sigma_f <= 0:
        hits = sum(1 for m in members if interval.contains(_round_half_up(m)))
        return hits / len(members)

    total = 0.0
    for m in members:
        total += _phi((upper - m) / sigma_f) - _phi((lower - m) / sigma_f)
    return total / len(members)


def kalshi_taker_fee(price: float, rate: float = 0.07) -> float:
    """Approximate Kalshi taker fee per $1 contract: rate * P * (1 - P).
    (Kalshi rounds the total fee up to the cent per order; ignored here.)"""
    return rate * price * (1.0 - price)
