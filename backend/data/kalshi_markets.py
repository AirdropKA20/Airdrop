"""Kalshi weather temperature market fetcher."""
import logging
import re
from datetime import date, datetime
from typing import Dict, List, Optional

from backend.data.kalshi_client import KalshiClient, kalshi_credentials_present
from backend.data.temp_intervals import TempInterval, interval_from_kalshi
from backend.data.weather_markets import WeatherMarket

logger = logging.getLogger("trading_bot")

# Kalshi series tickers for high-temperature markets by city
CITY_SERIES: Dict[str, str] = {
    "nyc": "KXHIGHNY",
    "chicago": "KXHIGHCHI",
    "miami": "KXHIGHMIA",
    "los_angeles": "KXHIGHLAX",
    "denver": "KXHIGHDEN",
}

CITY_NAMES: Dict[str, str] = {
    "nyc": "New York",
    "chicago": "Chicago",
    "miami": "Miami",
    "los_angeles": "Los Angeles",
    "denver": "Denver",
}

# Month abbreviation mapping for ticker parsing
MONTH_ABBR = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


_DATE_RE = re.compile(r"^[A-Z]+-(\d{2})([A-Z]{3})(\d{2})-")


def _parse_kalshi_date(ticker: str) -> Optional[date]:
    """KXHIGHNY-26MAR01-B45.5 -> 2026-03-01."""
    match = _DATE_RE.match(ticker)
    if not match:
        return None
    month = MONTH_ABBR.get(match.group(2))
    if not month:
        return None
    try:
        return date(2000 + int(match.group(1)), month, int(match.group(3)))
    except ValueError:
        return None


def _kalshi_price(m: dict, field: str) -> Optional[float]:
    """Read a Kalshi quote as a 0-1 probability.

    Supports both the ``*_dollars`` string fields (e.g. "0.4200") and the
    older integer-cent fields (e.g. 42). Returns None when absent / zero.
    """
    dollars = m.get(f"{field}_dollars")
    if dollars not in (None, ""):
        try:
            v = float(dollars)
            return v if 0 < v < 1 else None
        except (TypeError, ValueError):
            pass
    cents = m.get(field)
    if cents in (None, ""):
        return None
    try:
        v = float(cents) / 100.0
    except (TypeError, ValueError):
        return None
    return v if 0 < v < 1 else None


def kalshi_entry_prices(m: dict) -> Optional[tuple]:
    """Executable (yes_cost, no_cost) for buying each side now.

    Buying NO at the NO ask is equivalent to selling YES at the YES bid, so
    no_cost falls back to 1 - yes_bid (and vice versa). Never uses last trade
    price: that isn't a price you can actually get.
    """
    yes_ask = _kalshi_price(m, "yes_ask")
    no_ask = _kalshi_price(m, "no_ask")
    yes_bid = _kalshi_price(m, "yes_bid")
    no_bid = _kalshi_price(m, "no_bid")

    yes_cost = yes_ask if yes_ask is not None else (1.0 - no_bid if no_bid is not None else None)
    no_cost = no_ask if no_ask is not None else (1.0 - yes_bid if yes_bid is not None else None)
    if yes_cost is None or no_cost is None:
        return None
    return yes_cost, no_cost


async def fetch_kalshi_weather_markets(
    city_keys: Optional[List[str]] = None,
) -> List[WeatherMarket]:
    """
    Fetch open weather temperature markets from Kalshi.

    Queries the KXHIGH{city} series for each configured city,
    handles cursor-based pagination, and returns WeatherMarket objects.
    """
    client = KalshiClient()  # market data is public; keys are optional
    markets: List[WeatherMarket] = []
    today = date.today()

    cities = city_keys or list(CITY_SERIES.keys())

    for city_key in cities:
        series = CITY_SERIES.get(city_key)
        if not series:
            continue

        city_name = CITY_NAMES.get(city_key, city_key)
        cursor = None

        try:
            while True:
                params = {
                    "series_ticker": series,
                    "status": "open",
                    "limit": 200,
                }
                if cursor:
                    params["cursor"] = cursor

                data = await client.get_markets(params)
                raw_markets = data.get("markets", [])

                for m in raw_markets:
                    ticker = m.get("ticker", "")
                    target_date = _parse_kalshi_date(ticker)
                    if not target_date or target_date < today:
                        continue

                    interval = interval_from_kalshi(m)
                    if interval is None:
                        logger.debug(f"Skipping Kalshi market with unknown strike: {ticker}")
                        continue

                    prices = kalshi_entry_prices(m)
                    if prices is None:
                        continue  # one-sided / empty book
                    yes_price, no_price = prices

                    # Skip effectively resolved markets
                    if yes_price > 0.98 or yes_price < 0.02:
                        continue

                    volume = float(m.get("volume_fp") or m.get("volume") or 0)

                    markets.append(WeatherMarket(
                        slug=ticker,
                        market_id=ticker,
                        platform="kalshi",
                        title=m.get("title", ticker),
                        city_key=city_key,
                        city_name=city_name,
                        target_date=target_date,
                        threshold_f=interval.threshold,
                        metric="high",  # KXHIGH* series are daily highs
                        direction=interval.direction,
                        yes_price=yes_price,
                        no_price=no_price,
                        volume=volume,
                        interval=interval,
                    ))

                # Handle pagination
                cursor = data.get("cursor")
                if not cursor or not raw_markets:
                    break

        except Exception as e:
            logger.warning(f"Failed to fetch Kalshi markets for {city_key} ({series}): {e}")

    logger.info(f"Found {len(markets)} Kalshi weather markets")
    return markets
