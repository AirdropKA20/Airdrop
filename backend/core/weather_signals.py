"""Signal generator for weather temperature markets using ensemble forecasts."""
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from backend.config import settings
from backend.core.signals import calculate_kelly_size
from backend.core.weather_model import kalshi_taker_fee
from backend.data.weather import fetch_ensemble_forecast, EnsembleForecast, CITY_CONFIG
from backend.data.weather_markets import WeatherMarket, fetch_polymarket_weather_markets
from backend.models.database import SessionLocal, Signal

logger = logging.getLogger("trading_bot")


@dataclass
class WeatherTradingSignal:
    """A trading signal for a weather temperature market."""
    market: WeatherMarket

    # Core signal data
    model_probability: float = 0.5   # Ensemble probability of YES outcome
    market_probability: float = 0.5  # Market's implied YES probability
    edge: float = 0.0
    direction: str = "yes"           # "yes" or "no"

    # Confidence and sizing
    confidence: float = 0.5
    kelly_fraction: float = 0.0
    suggested_size: float = 0.0

    # Metadata
    sources: List[str] = field(default_factory=list)
    reasoning: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)

    # Forecast context
    ensemble_mean: float = 0.0
    ensemble_std: float = 0.0
    ensemble_members: int = 0

    @property
    def passes_threshold(self) -> bool:
        """Check if signal passes minimum edge threshold."""
        return self.edge >= settings.WEATHER_MIN_EDGE_THRESHOLD


async def generate_weather_signal(market: WeatherMarket) -> Optional[WeatherTradingSignal]:
    """
    Generate a trading signal for a weather temperature market.

    - Forecast is taken at the market's settlement station, over its
      settlement day (see backend.data.weather).
    - Model probability = chance the official whole-degree high/low lands in
      the contract's interval (dressed ensemble, see weather_model).
    - Edge is computed separately for buying YES and buying NO at their actual
      ask prices, net of Kalshi taker fees.
    """
    if market.interval is None:
        return None

    forecast = await fetch_ensemble_forecast(market.city_key, market.target_date, market.platform)
    if not forecast or not forecast.members(market.metric):
        return None

    raw_prob = forecast.probability_in(market.interval, market.metric, settings.WEATHER_MEMBER_SIGMA_F)
    if raw_prob is None:
        return None

    # Clip extreme probabilities (ensemble can be unanimous but don't bet 100%)
    model_yes_prob = max(0.02, min(0.98, raw_prob))

    fee_rate = settings.KALSHI_FEE_RATE if market.platform == "kalshi" else 0.0
    yes_cost = market.yes_price + kalshi_taker_fee(market.yes_price, fee_rate)
    no_cost = market.no_price + kalshi_taker_fee(market.no_price, fee_rate)

    yes_edge = model_yes_prob - yes_cost
    no_edge = (1.0 - model_yes_prob) - no_cost
    if yes_edge >= no_edge:
        direction, edge, entry_price = "yes", yes_edge, market.yes_price
    else:
        direction, edge, entry_price = "no", no_edge, market.no_price

    # Entry price filter
    if entry_price > settings.WEATHER_MAX_ENTRY_PRICE:
        edge = 0.0  # Zero out but still return for UI visibility

    # Confidence = how one-sided the model is
    confidence = min(0.9, max(model_yes_prob, 1.0 - model_yes_prob))

    # Kelly sizing. calculate_kelly_size prices the "down" side as
    # 1 - market_price, so pass the price matching the chosen side.
    bankroll = settings.INITIAL_BANKROLL
    if direction == "yes":
        suggested_size = calculate_kelly_size(
            edge=max(edge, 0.0), probability=model_yes_prob,
            market_price=yes_cost, direction="up", bankroll=bankroll,
        )
    else:
        suggested_size = calculate_kelly_size(
            edge=max(edge, 0.0), probability=model_yes_prob,
            market_price=1.0 - no_cost, direction="down", bankroll=bankroll,
        )
    if edge <= 0:
        suggested_size = 0.0
    suggested_size = min(suggested_size, settings.WEATHER_MAX_TRADE_SIZE)

    # Ensemble stats for display
    mean_val = forecast.mean_high if market.metric == "high" else forecast.mean_low
    std_val = forecast.std_high if market.metric == "high" else forecast.std_low

    filter_status = "ACTIONABLE" if edge >= settings.WEATHER_MIN_EDGE_THRESHOLD else "FILTERED"
    filter_note = (
        f" [entry {entry_price:.0%} > {settings.WEATHER_MAX_ENTRY_PRICE:.0%}]"
        if entry_price > settings.WEATHER_MAX_ENTRY_PRICE else ""
    )

    reasoning = (
        f"[{filter_status}]{filter_note} "
        f"{market.city_name} ({forecast.station}) {market.metric} {market.interval.label()} on {market.target_date} | "
        f"Ensemble: {mean_val:.1f}F +/- {std_val:.1f}F ({forecast.num_members} members) | "
        f"Model YES: {model_yes_prob:.0%} vs YES ask {market.yes_price:.0%} / NO ask {market.no_price:.0%} | "
        f"Edge{' (after fees)' if fee_rate else ''}: {edge:+.1%} -> {direction.upper()} @ {entry_price:.0%}"
    )

    return WeatherTradingSignal(
        market=market,
        model_probability=model_yes_prob,
        market_probability=market.yes_price,
        edge=edge,
        direction=direction,
        confidence=confidence,
        kelly_fraction=suggested_size / bankroll if bankroll > 0 else 0,
        suggested_size=suggested_size,
        sources=[f"open_meteo_gfs_ensemble_{forecast.num_members}m@{forecast.station}"],
        reasoning=reasoning,
        ensemble_mean=mean_val,
        ensemble_std=std_val,
        ensemble_members=forecast.num_members,
    )


async def scan_for_weather_signals() -> List[WeatherTradingSignal]:
    """
    Scan weather markets and generate ensemble-based signals.
    """
    signals = []

    city_keys = [c.strip() for c in settings.WEATHER_CITIES.split(",") if c.strip()]

    logger.info("=" * 50)
    logger.info("WEATHER SCAN: Fetching temperature markets...")

    markets = []

    # Polymarket
    try:
        poly_markets = await fetch_polymarket_weather_markets(city_keys)
        markets.extend(poly_markets)
        logger.info(f"Polymarket: {len(poly_markets)} weather markets")
    except Exception as e:
        logger.error(f"Failed to fetch Polymarket weather markets: {e}")

    # Kalshi
    if settings.KALSHI_ENABLED:
        try:
            from backend.data.kalshi_client import kalshi_credentials_present
            from backend.data.kalshi_markets import fetch_kalshi_weather_markets
            if kalshi_credentials_present():
                kalshi_markets = await fetch_kalshi_weather_markets(city_keys)
                markets.extend(kalshi_markets)
                logger.info(f"Kalshi: {len(kalshi_markets)} weather markets")
        except Exception as e:
            logger.error(f"Failed to fetch Kalshi weather markets: {e}")

    logger.info(f"Found {len(markets)} total weather temperature markets")

    for market in markets:
        try:
            signal = await generate_weather_signal(market)
            if signal:
                signals.append(signal)
        except Exception as e:
            logger.debug(f"Weather signal generation failed for {market.title}: {e}")

    # Sort by absolute edge
    signals.sort(key=lambda s: s.edge, reverse=True)

    actionable = [s for s in signals if s.passes_threshold]
    logger.info(f"WEATHER SCAN COMPLETE: {len(signals)} signals, {len(actionable)} actionable")

    for signal in actionable[:5]:
        logger.info(f"  {signal.market.city_name}: {signal.market.metric} {signal.market.interval.label()} "
                     f"| {signal.direction.upper()} edge: {signal.edge:+.1%}")

    # Persist signals to DB
    _persist_weather_signals(signals)

    return signals


def _persist_weather_signals(signals: list):
    """Save weather signals to DB for calibration tracking."""
    to_save = [s for s in signals if abs(s.edge) > 0]
    if not to_save:
        return

    db = SessionLocal()
    try:
        for signal in to_save:
            # Dedup: skip if already logged for this market
            existing = db.query(Signal).filter(
                Signal.market_ticker == signal.market.market_id,
                Signal.timestamp >= signal.timestamp.replace(second=0, microsecond=0),
            ).first()
            if existing:
                continue

            db_signal = Signal(
                market_ticker=signal.market.market_id,
                platform=signal.market.platform,
                market_type="weather",
                timestamp=signal.timestamp,
                direction=signal.direction,
                model_probability=signal.model_probability,
                market_price=signal.market_probability,
                edge=signal.edge,
                confidence=signal.confidence,
                kelly_fraction=signal.kelly_fraction,
                suggested_size=signal.suggested_size,
                sources=signal.sources,
                reasoning=signal.reasoning,
                executed=False,
            )
            db.add(db_signal)

        db.commit()
    except Exception as e:
        logger.warning(f"Failed to persist weather signals: {e}")
        db.rollback()
    finally:
        db.close()
