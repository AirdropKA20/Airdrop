"""Settlement intervals for daily temperature markets.

Official daily highs/lows (NWS climate reports, Weather Underground history)
are whole degrees Fahrenheit, so every temperature contract can be expressed as
an inclusive integer interval [lo, hi] (either side may be unbounded):

    "45° to 46°"          -> [45, 46]
    "52° or above"        -> [52, +inf)
    "exceed 75°F"         -> [76, +inf)   (strictly greater than 75)
    "39° or below"        -> (-inf, 39]
    "be 45°F"             -> [45, 45]

This module is deliberately free of network / heavy dependencies so it can be
unit-tested in isolation.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TempInterval:
    lo: Optional[int]  # inclusive lower bound, None = unbounded
    hi: Optional[int]  # inclusive upper bound, None = unbounded

    def __post_init__(self):
        if self.lo is None and self.hi is None:
            raise ValueError("interval must be bounded on at least one side")
        if self.lo is not None and self.hi is not None and self.lo > self.hi:
            raise ValueError(f"empty interval [{self.lo}, {self.hi}]")

    def contains(self, temp_f: int) -> bool:
        return (self.lo is None or temp_f >= self.lo) and (self.hi is None or temp_f <= self.hi)

    @property
    def direction(self) -> str:
        """Legacy label used by the API/UI: above, below, between or exact."""
        if self.hi is None:
            return "above"
        if self.lo is None:
            return "below"
        return "exact" if self.lo == self.hi else "between"

    @property
    def threshold(self) -> float:
        """A single representative temperature for legacy display fields."""
        if self.hi is None:
            return float(self.lo)
        if self.lo is None:
            return float(self.hi)
        return (self.lo + self.hi) / 2.0

    def label(self) -> str:
        if self.hi is None:
            return f">={self.lo}F"
        if self.lo is None:
            return f"<={self.hi}F"
        if self.lo == self.hi:
            return f"={self.lo}F"
        return f"{self.lo}-{self.hi}F"


# ---------------------------------------------------------------------------
# Kalshi
# ---------------------------------------------------------------------------

def _num(value) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def interval_from_kalshi(market: dict) -> Optional[TempInterval]:
    """Build the settlement interval for a Kalshi market object.

    Prefers the structured ``strike_type`` / ``floor_strike`` / ``cap_strike``
    fields from the Kalshi API. Falls back to the ticker suffix plus the
    market's title text when those are missing. Returns None when the
    contract can't be determined unambiguously (better to skip than guess).
    """
    strike_type = (market.get("strike_type") or "").lower()
    floor_s = _num(market.get("floor_strike"))
    cap_s = _num(market.get("cap_strike"))

    try:
        if strike_type == "between" and floor_s is not None and cap_s is not None:
            return TempInterval(math.ceil(floor_s), math.floor(cap_s))
        if strike_type == "greater" and floor_s is not None:
            return TempInterval(math.floor(floor_s) + 1, None)
        if strike_type == "greater_or_equal" and floor_s is not None:
            return TempInterval(math.ceil(floor_s), None)
        if strike_type == "less" and cap_s is not None:
            return TempInterval(None, math.ceil(cap_s) - 1)
        if strike_type == "less_or_equal" and cap_s is not None:
            return TempInterval(None, math.floor(cap_s))
    except ValueError:
        return None

    return _interval_from_kalshi_ticker(market)


_KALSHI_SUFFIX = re.compile(r"-([BT])(-?\d+(?:\.\d+)?)$")


def _interval_from_kalshi_ticker(market: dict) -> Optional[TempInterval]:
    ticker = market.get("ticker", "") or ""
    m = _KALSHI_SUFFIX.search(ticker)
    if not m:
        return None
    kind, value = m.group(1), float(m.group(2))

    if kind == "B":
        # Bracket ticker: B45.5 is the 2-degree bucket 45-46F.
        lo = math.floor(value)
        return TempInterval(lo, lo + 1)

    # "T" = tail. Its direction (above vs below) is not encoded in the ticker,
    # so it must come from the market text.
    text = " ".join(
        str(market.get(k) or "") for k in ("yes_sub_title", "subtitle", "title")
    )
    parsed = parse_temp_interval(text)
    if parsed and parsed.direction in ("above", "below"):
        return parsed
    return None


# ---------------------------------------------------------------------------
# Free text (Polymarket questions, Kalshi subtitles)
# ---------------------------------------------------------------------------

_DEG = r"\s*(?:°\s*f?\b|°|degrees?(?:\s+f(?:ahrenheit)?)?\b|f\b)"
_NUM = r"(-?\d{1,3})"

_RANGE_RE = re.compile(
    rf"(?:between\s+)?{_NUM}\s*°?\s*f?\s*(?:-|–|—|to|and)\s*{_NUM}{_DEG}"
    rf"|between\s+{_NUM}\s*°?\s*f?\s*(?:-|–|—|to|and)\s*{_NUM}\b"
)
_OR_ABOVE_RE = re.compile(rf"{_NUM}{_DEG}\s*(?:or\s+(?:higher|above|more|warmer|greater)|\+)")
_OR_BELOW_RE = re.compile(rf"{_NUM}{_DEG}\s*or\s+(?:lower|below|less|colder|under)")
_AT_LEAST_RE = re.compile(rf"(?:at\s+least|no\s+less\s+than|>=|≥)\s*{_NUM}{_DEG}")
_AT_MOST_RE = re.compile(rf"(?:at\s+most|no\s+more\s+than|<=|≤)\s*{_NUM}{_DEG}")
_ABOVE_RE = re.compile(
    rf"(?:above|exceed(?:s|ing)?|over|higher\s+than|greater\s+than|more\s+than|warmer\s+than|>)\s*{_NUM}{_DEG}"
)
_BELOW_RE = re.compile(
    rf"(?:below|under|less\s+than|lower\s+than|colder\s+than|<)\s*{_NUM}{_DEG}"
)
_EXACT_RE = re.compile(rf"{_NUM}{_DEG}")


def parse_temp_interval(text: str) -> Optional[TempInterval]:
    """Parse the settlement interval out of a market question / subtitle."""
    t = (text or "").lower()

    m = _RANGE_RE.search(t)
    if m:
        a, b = [int(g) for g in m.groups() if g is not None][:2]
        lo, hi = min(a, b), max(a, b)
        return TempInterval(lo, hi)

    for regex, build in (
        (_OR_ABOVE_RE, lambda v: TempInterval(v, None)),
        (_OR_BELOW_RE, lambda v: TempInterval(None, v)),
        (_AT_LEAST_RE, lambda v: TempInterval(v, None)),
        (_AT_MOST_RE, lambda v: TempInterval(None, v)),
        (_ABOVE_RE, lambda v: TempInterval(v + 1, None)),
        (_BELOW_RE, lambda v: TempInterval(None, v - 1)),
        (_EXACT_RE, lambda v: TempInterval(v, v)),
    ):
        m = regex.search(t)
        if m:
            return build(int(m.group(1)))
    return None


_HIGH_RE = re.compile(r"\b(?:high|highest|maximum|max)\b")
_LOW_RE = re.compile(r"\b(?:low|lowest|minimum|min)\b")


def parse_metric(text: str) -> str:
    """Return "high" or "low". Word boundaries avoid matching be-LOW / LOW-er."""
    t = (text or "").lower()
    if _HIGH_RE.search(t):
        return "high"
    if _LOW_RE.search(t):
        return "low"
    return "high"
