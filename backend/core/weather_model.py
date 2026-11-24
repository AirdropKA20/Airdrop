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


def floored_interval_probability(
    members: Iterable[float],
    interval: TempInterval,
    sigma_f: float,
    observed_floor: Optional[int],
) -> Optional[float]:
    """Like interval_probability, but conditioned on the day's high being at
    least ``observed_floor`` (whole degrees F already observed on the
    settlement day: the high can't end up lower than that)."""
    members = [float(m) for m in members]
    if observed_floor is None:
        return interval_probability(members, interval, sigma_f)
    floor_iv = TempInterval(observed_floor, None)
    denom = interval_probability(members, floor_iv, sigma_f)
    if interval.hi is not None and interval.hi < observed_floor:
        return 0.0
    lo = observed_floor if interval.lo is None else max(interval.lo, observed_floor)
    num = interval_probability(members, TempInterval(lo, interval.hi), sigma_f)
    if denom is None or num is None:
        return None
    if denom < 1e-9:
        # Forecast says the high should already be below what was observed:
        # trust the observation; the high is most likely exactly the floor.
        return 1.0 if interval.contains(observed_floor) else 0.0
    return min(1.0, num / denom)


def _max_with_integer_observed(remaining_mu, sigma_f, interval: TempInterval, observed: int) -> float:
    if interval.hi is not None and interval.hi < observed:
        return 0.0
    if remaining_mu is None:
        return 1.0 if interval.contains(observed) else 0.0
    p = 0.0
    if interval.contains(observed):
        # R rounds to <= observed -> final high stays at the observed value
        p += _phi((observed + 0.5 - remaining_mu) / sigma_f)
    lo = observed + 1 if interval.lo is None else max(interval.lo, observed + 1)
    if interval.hi is None or lo <= interval.hi:
        p += interval_probability([remaining_mu], TempInterval(lo, interval.hi), sigma_f)
    return min(1.0, p)


def max_with_observed_probability(
    remaining_mu: Optional[float],
    sigma_f: float,
    interval: TempInterval,
    observed: float,
    observed_sd: float = 0.0,
) -> float:
    """P(final high in interval) where final = max(O, R).

    * O is the true high so far. We only see ``observed`` (max of the
      available readings, e.g. hourly); the true value can be higher by a
      half-normal amount with scale ``observed_sd`` (never lower).
    * R ~ Normal(remaining_mu, sigma) is the highest temperature still to
      come; remaining_mu=None means no hours remain.
    Everything is rounded to whole degrees, as the official reading is.
    """
    if observed_sd <= 0:
        return _max_with_integer_observed(remaining_mu, sigma_f, interval, _round_half_up(observed))
    # distribution of the rounded true high so far
    total, p = 0.0, 0.0
    k = _round_half_up(observed)
    while True:
        lo_edge = max(k - 0.5, observed)
        hi_edge = k + 0.5
        if hi_edge > lo_edge:
            # half-normal CDF: P(|Z|*sd <= x) = 2*Phi(x/sd) - 1
            w = (2 * _phi((hi_edge - observed) / observed_sd) - 1) - (2 * _phi((lo_edge - observed) / observed_sd) - 1)
            if w > 0:
                total += w
                p += w * _max_with_integer_observed(remaining_mu, sigma_f, interval, k)
        if hi_edge - observed > 5 * observed_sd:
            break
        k += 1
    return min(1.0, p / total) if total > 0 else 0.0


def kalshi_taker_fee(price: float, rate: float = 0.07) -> float:
    """Approximate Kalshi taker fee per $1 contract: rate * P * (1 - P).
    (Kalshi rounds the total fee up to the cent per order; ignored here.)"""
    return rate * price * (1.0 - price)
