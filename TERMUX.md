# Running on Android (Termux)

Paper-only tools: they print signals and backtest results and **never place orders**.

## 1. Install (once)

Install Termux from F-Droid (the Play Store build is outdated), then:

```bash
pkg update && pkg upgrade -y
pkg install -y python git python-cryptography
```

Get the code. The repo is probably private, so log in to GitHub first (`gh` is easiest):

```bash
pkg install -y gh
gh auth login            # choose GitHub.com, HTTPS, log in with a browser
gh repo clone AirdropKA20/Airdrop -- -b claude/polymarket-kalshi-weather-bot-9b38n7
cd Airdrop
```

Install the Python packages:

```bash
pip install -r requirements-termux.txt
```

If that fails while building `pydantic-core` or `cryptography`, install the Rust
toolchain and retry (the build takes 10–20 minutes on a phone):

```bash
pkg install -y rust binutils
pip install -r requirements-termux.txt
```

Check that it works:

```bash
python -m pytest -q tests      # optional: pkg install python-pytest, or pip install pytest
```

## 2. Live signals

```bash
python -m scripts.signals                      # actionable signals, all cities, both platforms
python -m scripts.signals --all                # every market, including no-trade rows
python -m scripts.signals --cities nyc,chicago --platform kalshi
python -m scripts.signals --verbose            # also print each city's calibration
```

Each row shows the contract, the model probability, which side to buy, the
real price you would pay, the edge after fees and a suggested size. `TRADE`
marks rows above `WEATHER_MIN_EDGE_THRESHOLD`.

Re-scan every 30 minutes (keep the phone awake with `termux-wake-lock`):

```bash
termux-wake-lock
while true; do date; python -m scripts.signals; sleep 1800; done
```

## 3. Paper-trade test (the honest test)

Every `scripts.signals` run appends what the model said and the real prices to
`data/paper_log.jsonl`. Once markets settle, score it:

```bash
python -m scripts.paper_report               # trades = first actionable signal per market, $10 each
python -m scripts.paper_report --min-edge 0.12
```

It reports the win %, P&L and the model's real forecast error (MAE). Unlike the
backtest, nothing here can see the future. Run the scanner a few times a day
for 3–4 weeks; under ~100 settled trades the result is mostly luck.

## 4. Backtest

Step 1 downloads a year of Kalshi markets, hourly prices and archived
forecasts. Kalshi rate-limits it, so it takes **3–4 hours** for 5 cities. It
can be stopped and restarted: finished days are cached in `data/backtest_cache/`.

```bash
termux-wake-lock
python -m scripts.backtest.collect --start 2025-12-01 --end 2026-11-22
# quicker trial: one city, two months
python -m scripts.backtest.collect --start 2026-09-01 --end 2026-11-22 --cities nyc
```

Step 2 runs the walk-forward backtest on whatever has been downloaded:

```bash
python -m scripts.backtest.backtest                         # enter 08:00 local time on the day
python -m scripts.backtest.backtest --entry-hour 20 --entry-day-offset -1   # evening before
python -m scripts.backtest.backtest --noise-sd 2.0          # add realistic next-day forecast error
```

It compares the market, a GFS-only model and the multi-model blend on the
same days. It reports forecast error, log-loss (lower is better than the
market only if the model's number is below the market's), and simulated
trades at the real ask with Kalshi fees at several edge thresholds.

**Read the results as an upper bound.** The archived forecasts are stitched
from short-range runs and know a few hours more than you would at entry time.
If the blend can't beat the market's log-loss here, it won't in real trading.

## Settings

Put overrides in a `.env` file in the repo folder, e.g.:

```env
WEATHER_MIN_EDGE_THRESHOLD=0.08
WEATHER_CITIES=nyc,chicago,miami,los_angeles,denver
WEATHER_MODEL=blend          # or "ensemble" for the GFS-ensemble model
WEATHER_USE_OBSERVED_FLOOR=true
```

Kalshi API keys are **not** needed: market data is public.
