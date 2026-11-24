"""Offline tests for the weather strategy fixes (no network access needed)."""
import asyncio
from datetime import date, datetime, timedelta, timezone

import pytest

from backend.core.weather_model import interval_probability, kalshi_taker_fee, settlement_window_utc
from backend.data.temp_intervals import TempInterval, interval_from_kalshi, parse_metric, parse_temp_interval


# --- Kalshi strike parsing -------------------------------------------------

@pytest.mark.parametrize("market,expected", [
    ({"strike_type": "between", "floor_strike": 45, "cap_strike": 46}, TempInterval(45, 46)),
    ({"strike_type": "greater", "floor_strike": 51}, TempInterval(52, None)),
    ({"strike_type": "greater_or_equal", "floor_strike": 52}, TempInterval(52, None)),
    ({"strike_type": "less", "cap_strike": 40}, TempInterval(None, 39)),
    ({"strike_type": "less_or_equal", "cap_strike": 39}, TempInterval(None, 39)),
    # Fallback: bracket ticker is a 2-degree range, NOT "above 45.5"
    ({"ticker": "KXHIGHNY-26MAR01-B45.5"}, TempInterval(45, 46)),
    # Fallback: tail ticker direction comes from the text
    ({"ticker": "KXHIGHNY-26MAR01-T52", "yes_sub_title": "52° or above"}, TempInterval(52, None)),
    ({"ticker": "KXHIGHNY-26MAR01-T39", "yes_sub_title": "39° or below"}, TempInterval(None, 39)),
])
def test_interval_from_kalshi(market, expected):
    assert interval_from_kalshi(market) == expected


def test_kalshi_tail_without_text_is_skipped():
    assert interval_from_kalshi({"ticker": "KXHIGHNY-26MAR01-T52"}) is None


# --- Free-text parsing (Polymarket) ---------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("Will the highest temperature in NYC be between 44-45°F on November 25?", TempInterval(44, 45)),
    ("Highest temperature in Chicago 30° to 31°F on Dec 1", TempInterval(30, 31)),
    ("Will the highest temperature in Miami be 85°F or higher on May 3?", TempInterval(85, None)),
    ("Will the highest temperature in Denver be 20°F or below on Jan 9?", TempInterval(None, 20)),
    ("Will the high temperature in New York exceed 75°F on March 5?", TempInterval(76, None)),
    ("Will Miami's low be below 65°F on March 7?", TempInterval(None, 64)),
    ("Will the highest temperature in NYC be 45°F on November 25?", TempInterval(45, 45)),
    ("Will it rain in NYC on March 5?", None),
])
def test_parse_temp_interval(text, expected):
    assert parse_temp_interval(text) == expected


@pytest.mark.parametrize("text,metric", [
    ("NYC high temperature below 40°F on March 5", "high"),  # "below" must not mean low
    ("Will Miami's low be above 65°F", "low"),
    ("Lowest temperature in Denver on Jan 9", "low"),
    ("Temperature in Denver above 70°F", "high"),
])
def test_parse_metric(text, metric):
    assert parse_metric(text) == metric


def test_polymarket_city_aliases_use_word_boundaries():
    from backend.data.weather_markets import _parse_weather_market_title
    assert _parse_weather_market_title("Highest temperature in Atlanta 70-71°F on May 3?") is None
    assert _parse_weather_market_title("Highest temperature in Dallas 70-71°F on May 3?") is None
    parsed = _parse_weather_market_title("Highest temperature in Los Angeles 70-71°F on May 3?")
    assert parsed["city_key"] == "los_angeles"
    assert parsed["interval"] == TempInterval(70, 71)


def test_polymarket_prefers_order_book_prices():
    from backend.data.weather_markets import _parse_polymarket_weather
    future = date.today() + timedelta(days=2)
    m = _parse_polymarket_weather({
        "id": "1",
        "question": f"Highest temperature in NYC 44-45°F on {future.strftime('%B')} {future.day}, {future.year}?",
        "outcomePrices": '["0.30", "0.70"]',
        "bestAsk": "0.32", "bestBid": "0.28",
    }, "slug")
    assert m.yes_price == pytest.approx(0.32)
    assert m.no_price == pytest.approx(0.72)


# --- Settlement windows ---------------------------------------------------

def test_kalshi_window_is_local_standard_time_even_in_summer():
    start, end = settlement_window_utc(date(2026, 7, 4), -5, "America/New_York", "kalshi")
    assert start == datetime(2026, 7, 4, 5, tzinfo=timezone.utc)
    assert end == datetime(2026, 7, 5, 5, tzinfo=timezone.utc)


def test_polymarket_window_is_civil_day_with_dst():
    start, end = settlement_window_utc(date(2026, 7, 4), -5, "America/New_York", "polymarket")
    assert start == datetime(2026, 7, 4, 4, tzinfo=timezone.utc)
    assert end == datetime(2026, 7, 5, 4, tzinfo=timezone.utc)


def test_extract_member_extremes_uses_window_only():
    from backend.data.weather import extract_member_extremes
    base = datetime(2026, 1, 10, 0)
    times = [(base + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M") for h in range(48)]
    # LA standard-time day for Jan 10 = 08:00Z Jan 10 .. 08:00Z Jan 11.
    # Put a spike before the window that must be ignored.
    temps = [60.0] * 48
    temps[3] = 99.0     # 03Z Jan 10 = 7pm Jan 9 local -> previous day
    temps[30] = 72.0    # 06Z Jan 11 = 10pm Jan 10 local -> inside
    hourly = {"time": times, "temperature_2m": temps, "temperature_2m_member01": temps}
    start, end = settlement_window_utc(date(2026, 1, 10), -8)
    highs, lows = extract_member_extremes(hourly, start, end)
    assert highs == [72.0, 72.0]
    assert lows == [60.0, 60.0]


# --- Probabilities --------------------------------------------------------

def test_interval_probability_counting_rounds_to_whole_degrees():
    members = [44.4, 44.6, 45.4, 45.6, 46.6]
    # rounded: 44, 45, 45, 46, 47 -> 3 of 5 in [45, 46]
    assert interval_probability(members, TempInterval(45, 46), 0) == pytest.approx(0.6)


def test_interval_probability_dressed_is_smooth_and_sums_to_one():
    members = [70.0 + 0.3 * i for i in range(31)]
    buckets = [TempInterval(None, 69)] + [TempInterval(t, t + 1) for t in range(70, 80, 2)] + [TempInterval(80, None)]
    total = sum(interval_probability(members, b, 2.0) for b in buckets)
    assert total == pytest.approx(1.0, abs=1e-9)
    p = interval_probability(members, TempInterval(74, 75), 2.0)
    assert 0.1 < p < 0.5


def test_kalshi_fee():
    assert kalshi_taker_fee(0.5) == pytest.approx(0.0175)
    assert kalshi_taker_fee(0.9) == pytest.approx(0.0063)


# --- Kalshi prices --------------------------------------------------------

def test_kalshi_entry_prices():
    from backend.data.kalshi_markets import kalshi_entry_prices
    assert kalshi_entry_prices({"yes_ask": 40, "no_ask": 63}) == pytest.approx((0.40, 0.63))
    # NO cost falls back to 1 - YES bid, not 1 - YES ask
    assert kalshi_entry_prices({"yes_ask": 40, "yes_bid": 35}) == pytest.approx((0.40, 0.65))
    assert kalshi_entry_prices({"yes_ask_dollars": "0.4100", "no_ask_dollars": "0.6200"}) == pytest.approx((0.41, 0.62))
    assert kalshi_entry_prices({"yes_ask": 40, "last_price": 50}) is None


# --- End-to-end signal with a mocked forecast ------------------------------

def test_generate_weather_signal_end_to_end(monkeypatch):
    from backend.core import weather_signals
    from backend.data.weather import EnsembleForecast
    from backend.data.weather_markets import WeatherMarket

    target = date.today() + timedelta(days=1)
    forecast = EnsembleForecast(
        city_key="nyc", city_name="New York City", target_date=target,
        member_highs=[50.0] * 31, member_lows=[40.0] * 31, station="KNYC",
    )

    async def fake_fetch(city_key, target_date, platform="kalshi"):
        assert platform == "kalshi"
        return forecast

    monkeypatch.setattr(weather_signals, "fetch_ensemble_forecast", fake_fetch)

    market = WeatherMarket(
        slug="KXHIGHNY-X-B49.5", market_id="KXHIGHNY-X-B49.5", platform="kalshi",
        title="", city_key="nyc", city_name="New York City", target_date=target,
        threshold_f=49.5, metric="high", direction="between",
        yes_price=0.20, no_price=0.82, interval=TempInterval(49, 50),
    )
    sig = asyncio.run(weather_signals.generate_weather_signal(market))
    # Every member at 50F: dressed probability of 49-50F is ~0.38, well above 0.20 + fee
    assert sig.direction == "yes"
    assert 0.3 < sig.model_probability < 0.45
    assert sig.edge == pytest.approx(sig.model_probability - 0.20 - kalshi_taker_fee(0.20), abs=1e-9)
    assert sig.suggested_size > 0
