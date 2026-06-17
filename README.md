# vitaliy-kronos

ICT session backtester with Kronos AI directional filter for BTC/USDT.

## Strategy

1. Mark **Asia** (00:00–08:00 UTC) and **London** (08:00–13:00 UTC) session highs/lows each day
2. Wait for a **liquidity sweep** of those levels in the 13:00–16:00 UTC window (London–NY overlap)
3. Drop to **1-minute** chart, find a **Fair Value Gap** in the reaction
4. Enter at FVG midpoint, SL at sweep extreme, TP at next **draw on liquidity**
5. **Kronos AI filter**: only take trades where the Kronos foundation model's directional forecast agrees with the setup

## Quickstart

```bash
pip install -r requirements.txt

# Synthetic data (no API needed)
python src/run.py

# Real Binance data (60 days BTC/USDT 1m)
python src/run.py --live

# Real Binance + real Kronos model
git clone https://github.com/shiyu-coder/Kronos
PYTHONPATH=./Kronos python src/run.py --live --real-kronos
```

Opens `report.html` — dark-mode dashboard comparing ICT-only vs ICT+Kronos equity curves, win rate, drawdown, and full trade log.

## Project structure

```
src/
  backtester.py     — session ranges, sweep detection, FVG, trade simulation
  kronos.py         — KronosSignalGenerator (real) + MockKronos (stub)
  synthetic_data.py — realistic synthetic BTC OHLCV for offline testing
  run.py            — main entry point, generates HTML report
data/
  sample_report.html — example output (synthetic data)
```

## Parameters (src/run.py)

| Param | Default | Notes |
|---|---|---|
| Leverage | 5x | Applied to position sizing |
| Risk/trade | 1% account | Per trade, pre-leverage |
| Session window | 13:00–16:00 UTC | London–NY overlap |
| Trade cooldown | 120 min | One trade per 2-hour block |
| Fallback RR | 2.0 | Used when no clear DOL target |

## Switching to real Kronos

1. Clone [shiyu-coder/Kronos](https://github.com/shiyu-coder/Kronos)
2. Install its requirements: `pip install -r Kronos/requirements.txt`
3. Run: `PYTHONPATH=./Kronos python src/run.py --live --real-kronos`

Kronos-mini (4.1M params) runs on CPU but is slow. GPU recommended for daily signal generation.

---

*Not financial advice. For research and backtesting only.*
