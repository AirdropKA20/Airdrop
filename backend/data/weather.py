"""Weather data fetcher using Open-Meteo Ensemble API and NWS observations.

Forecasts are taken at the *settlement station* (not the city centre) and the
daily high/low is computed over the *settlement day* for the platform
(see backend.core.weather_model.settlement_window_utc).
"""
import httpx
import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple
import statistics
import time

from backend.core.weather_model import interval_probability, settlement_window_utc
from backend.data.temp_intervals import TempInterval

logger = logging.getLogger("trading_bot")

# Observation stations that markets settle on.
STATIONS: Dict[str, dict] = {
    "KNYC": {"name": "Central Park, NY", "lat": 40.7789, "lon": -73.9692},
    "KLGA": {"name": "LaGuardia Airport, NY", "lat": 40.7769, "lon": -73.8740},
    "KMDW": {"name": "Chicago Midway", "lat": 41.7861, "lon": -87.7522},
    "KORD": {"name": "Chicago O'Hare", "lat": 41.9786, "lon": -87.9048},
    "KMIA": {"name": "Miami International", "lat": 25.7906, "lon": -80.3164},
    "KLAX": {"name": "Los Angeles International", "lat": 33.9382, "lon": -118.3866},
    "KDEN": {"name": "Denver International", "lat": 39.8467, "lon": -104.6562},
}

# City configuration.
#   station            - station Kalshi's KXHIGH* contracts settle on (NWS CLI report)
#   polymarket_station - station Polymarket resolves on, when different
#   lst_utc_offset     - local *standard* time offset (NWS climate day)
#   tz                 - IANA zone (Polymarket / Weather Underground civil day)
# Verified against live contract rules on 2026-11-24 (Kalshi rules_primary
# CLINYC/CLIMDW/CLIMIA/CLILAX/CLIDEN; Polymarket event descriptions).
# Settlement sources change: re-check before trading.
CITY_CONFIG: Dict[str, dict] = {
    "nyc": {
        "name": "New York City",
        "station": "KNYC",
        "polymarket_station": "KLGA",
        "lst_utc_offset": -5,
        "tz": "America/New_York",
        "polymarket_slug": "nyc",
    },
    "chicago": {
        "name": "Chicago",
        "station": "KMDW",
        "polymarket_station": "KORD",
        "lst_utc_offset": -6,
        "tz": "America/Chicago",
    },
    "miami": {
        "name": "Miami",
        "station": "KMIA",
        "lst_utc_offset": -5,
        "tz": "America/New_York",
    },
    "los_angeles": {
        "name": "Los Angeles",
        "station": "KLAX",
        "lst_utc_offset": -8,
        "tz": "America/Los_Angeles",
    },
    "denver": {
        "name": "Denver",
        "station": "KDEN",
        "lst_utc_offset": -7,
        "tz": "America/Denver",
    },
}

# Backwards-compatible lat/lon/nws_station on each city (used by the API/UI).
for _cfg in CITY_CONFIG.values():
    _st = STATIONS[_cfg["station"]]
    _cfg.setdefault("lat", _st["lat"])
    _cfg.setdefault("lon", _st["lon"])
    _cfg.setdefault("nws_station", _cfg["station"])


def station_for(city_key: str, platform: str = "kalshi") -> str:
    city = CITY_CONFIG[city_key]
    if platform == "polymarket":
        return city.get("polymarket_station", city["station"])
    return city["station"]


def settlement_window(city_key: str, target_date: date, platform: str = "kalshi") -> Tuple[datetime, datetime]:
    city = CITY_CONFIG[city_key]
    return settlement_window_utc(target_date, city["lst_utc_offset"], city["tz"], platform)


@dataclass
class EnsembleForecast:
    """Ensemble weather forecast with per-member data."""
    city_key: str
    city_name: str
    target_date: date
    member_highs: List[float]  # Settlement-day max temps (F) per ensemble member
    member_lows: List[float]   # Settlement-day min temps (F) per ensemble member
    station: str = ""
    platform: str = "kalshi"
    mean_high: float = 0.0
    std_high: float = 0.0
    mean_low: float = 0.0
    std_low: float = 0.0
    num_members: int = 0
    fetched_at: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self):
        if self.member_highs:
            self.mean_high = statistics.mean(self.member_highs)
            self.std_high = statistics.stdev(self.member_highs) if len(self.member_highs) > 1 else 0.0
            self.num_members = len(self.member_highs)
        if self.member_lows:
            self.mean_low = statistics.mean(self.member_lows)
            self.std_low = statistics.stdev(self.member_lows) if len(self.member_lows) > 1 else 0.0

    @property
    def ensemble_agreement(self) -> float:
        """How tightly members cluster around the median high: share of
        members within 2F of it (dashboard display only)."""
        if not self.member_highs:
            return 0.0
        median = statistics.median(self.member_highs)
        return sum(1 for h in self.member_highs if abs(h - median) <= 2.0) / len(self.member_highs)

    def members(self, metric: str) -> List[float]:
        return self.member_highs if metric == "high" else self.member_lows

    def probability_in(
        self, interval: TempInterval, metric: str, sigma_f: float, shift_f: float = 0.0
    ) -> Optional[float]:
        """Probability the official whole-degree high/low falls in ``interval``.
        ``shift_f`` is a bias correction added to every member."""
        return interval_probability([m + shift_f for m in self.members(metric)], interval, sigma_f)


# Simple cache: (station, target_date, platform) -> (timestamp, EnsembleForecast)
_forecast_cache: Dict[str, tuple] = {}
_CACHE_TTL = 900  # 15 minutes
_MIN_HOURS = 20   # require most of the 24 settlement-day hours


def _celsius_to_fahrenheit(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def _parse_utc_hour(s: str) -> datetime:
    # Open-Meteo returns "YYYY-MM-DDTHH:MM" in the requested timezone (GMT here).
    from datetime import timezone
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def extract_member_extremes(
    hourly: dict, start_utc: datetime, end_utc: datetime
) -> Tuple[List[float], List[float]]:
    """From Open-Meteo hourly ensemble output, compute each member's max and
    min over [start_utc, end_utc). Keys look like ``temperature_2m`` (control)
    and ``temperature_2m_member01`` ... ``_member30``."""
    times = [_parse_utc_hour(t) for t in hourly.get("time", [])]
    idx = [i for i, t in enumerate(times) if start_utc <= t < end_utc]

    highs: List[float] = []
    lows: List[float] = []
    for key, values in hourly.items():
        if not key.startswith("temperature_2m") or not isinstance(values, list):
            continue
        window = [values[i] for i in idx if i < len(values) and values[i] is not None]
        if len(window) < _MIN_HOURS:
            continue
        highs.append(float(max(window)))
        lows.append(float(min(window)))
    return highs, lows


async def fetch_ensemble_forecast(
    city_key: str,
    target_date: Optional[date] = None,
    platform: str = "kalshi",
) -> Optional[EnsembleForecast]:
    """
    Fetch the GFS ensemble (31 members) from Open-Meteo at the settlement
    station and compute each member's high/low over the settlement day.
    """
    if city_key not in CITY_CONFIG:
        logger.warning(f"Unknown city key: {city_key}")
        return None

    if target_date is None:
        target_date = date.today()

    city = CITY_CONFIG[city_key]
    station = station_for(city_key, platform)
    st = STATIONS[station]
    start_utc, end_utc = settlement_window(city_key, target_date, platform)

    cache_key = f"{station}_{target_date.isoformat()}_{platform}"
    now = time.time()
    if cache_key in _forecast_cache:
        cached_time, cached_forecast = _forecast_cache[cache_key]
        if now - cached_time < _CACHE_TTL:
            return cached_forecast

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            params = {
                "latitude": st["lat"],
                "longitude": st["lon"],
                "hourly": "temperature_2m",
                "temperature_unit": "fahrenheit",
                "timezone": "GMT",
                "start_date": start_utc.date().isoformat(),
                "end_date": (end_utc - timedelta(seconds=1)).date().isoformat(),
                "models": "gfs_seamless",
            }

            response = await client.get(
                "https://ensemble-api.open-meteo.com/v1/ensemble",
                params=params,
            )
            response.raise_for_status()
            data = response.json()

            member_highs, member_lows = extract_member_extremes(
                data.get("hourly", {}), start_utc, end_utc
            )

            if not member_highs:
                logger.warning(f"No ensemble data for {city_key} ({station}) on {target_date}")
                return None

            forecast = EnsembleForecast(
                city_key=city_key,
                city_name=city["name"],
                target_date=target_date,
                member_highs=member_highs,
                member_lows=member_lows,
                station=station,
                platform=platform,
            )

            _forecast_cache[cache_key] = (now, forecast)
            logger.info(f"Ensemble forecast for {city['name']} ({station}) on {target_date}: "
                        f"High {forecast.mean_high:.1f}F +/- {forecast.std_high:.1f}F "
                        f"({forecast.num_members} members)")

            return forecast

    except Exception as e:
        logger.warning(f"Failed to fetch ensemble forecast for {city_key}: {e}")
        return None


async def fetch_nws_observed_temperature(
    city_key: str,
    target_date: Optional[date] = None,
    platform: str = "kalshi",
) -> Optional[Dict[str, float]]:
    """
    Fetch observed temperatures from the NWS API over the settlement day.
    Returns dict with 'high' and 'low' in Fahrenheit, or None if not available.

    Note: these are hourly/sub-hourly observations, so the max can be slightly
    below the official climate-report high. Use for monitoring, not settlement.
    """
    if city_key not in CITY_CONFIG:
        return None

    if target_date is None:
        target_date = date.today()

    station = station_for(city_key, platform)
    start_utc, end_utc = settlement_window(city_key, target_date, platform)

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            url = f"https://api.weather.gov/stations/{station}/observations"
            headers = {"User-Agent": "(weather-trading-bot)"}
            params = {
                "start": start_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "end": end_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
            }

            response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
            data = response.json()

            temps = []
            for obs in data.get("features", []):
                temp_c = obs.get("properties", {}).get("temperature", {}).get("value")
                if temp_c is not None:
                    temps.append(_celsius_to_fahrenheit(temp_c))

            if not temps:
                return None

            return {"high": max(temps), "low": min(temps)}

    except Exception as e:
        logger.warning(f"Failed to fetch NWS observations for {city_key}: {e}")
        return None
