"""
Synthetic BTC/USDT 1-min OHLCV generator.
Produces realistic price action with:
  - Volatility clustering (GARCH-like)
  - Session-aware spreads (Asia quieter, London/NY volatile)
  - Occasional engineered sweeps of session highs/lows
Used for local backtesting when Binance API is unavailable.
Replace with fetch_binance_klines() for live data.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta


def generate_synthetic_btc(days=60, start_price=105_000.0, seed=42):
    rng = np.random.default_rng(seed)
    freq = "1min"
    start = datetime(2026, 4, 18, 0, 0, 0, tzinfo=timezone.utc)
    end   = start + timedelta(days=days)
    index = pd.date_range(start=start, end=end, freq=freq, tz="UTC")[:-1]
    n     = len(index)

    # ── session volatility multipliers ──
    hour = np.array(index.hour)
    vol_mult = np.where(hour < 8,  0.6,   # Asia: quiet
               np.where(hour < 13, 1.2,   # London: medium
               np.where(hour < 17, 1.5,   # NY overlap: most volatile
                                   0.7))) # off-hours

    # ── GARCH-like vol clustering ──
    base_vol = 0.00018   # ~0.018% per minute ≈ BTC realistic
    var = np.zeros(n)
    var[0] = base_vol ** 2
    alpha, beta = 0.08, 0.90
    eps = rng.standard_normal(n)
    for i in range(1, n):
        var[i] = base_vol**2 * (1 - alpha - beta) + alpha * var[i-1] * eps[i-1]**2 + beta * var[i-1]
    sigma = np.sqrt(var) * vol_mult

    # ── price path ──
    returns = eps * sigma
    log_price = np.log(start_price) + np.cumsum(returns)
    close = np.exp(log_price)

    # ── OHLCV from close ──
    # Candle spread scales with vol
    spread = sigma * close * rng.uniform(0.3, 1.2, n)
    open_  = np.roll(close, 1)
    open_[0] = start_price
    high   = np.maximum(open_, close) + spread * rng.uniform(0.2, 1.0, n)
    low    = np.minimum(open_, close) - spread * rng.uniform(0.2, 1.0, n)
    volume = rng.lognormal(mean=8.0, sigma=0.6, size=n) * vol_mult

    df = pd.DataFrame({
        "open":   open_,
        "high":   high,
        "low":    low,
        "close":  close,
        "volume": volume,
    }, index=index)

    # ── Engineer sweep events ──
    # Every 3–5 days, inject a clear sweep at London open or NY open
    df = _inject_sweeps(df, rng)

    return df.round(2)


def _inject_sweeps(df, rng):
    """
    Inject ±0.3–0.6% wicks at session boundaries once every ~3–4 days
    to ensure the backtester has clean sweep events to detect.
    """
    dates = pd.Series(df.index.date).unique()
    sweep_dates = dates[::3]  # every 3rd day

    for d in sweep_dates:
        day_mask = df.index.date == d

        # London sweep at 13:00–13:30 UTC: wick below Asia low then close above
        window = df[(day_mask) & (df.index.hour == 13) & (df.index.minute < 30)]
        if len(window) > 5:
            asia_lo = df[(day_mask) & (df.index.hour < 8)]["low"].min()
            if np.isfinite(asia_lo):
                wick_pct = rng.uniform(0.003, 0.006)
                sweep_low = asia_lo * (1 - wick_pct)
                # pick a random candle in window
                pick = rng.integers(2, min(10, len(window)))
                idx  = window.index[pick]
                df.at[idx, "low"]   = sweep_low
                df.at[idx, "close"] = asia_lo * 1.001  # close back above

        # NY sweep at 14:00–14:30: wick above London high
        window2 = df[(day_mask) & (df.index.hour == 14) & (df.index.minute < 30)]
        if len(window2) > 5:
            london_hi = df[(day_mask) & (df.index.hour >= 8) & (df.index.hour < 13)]["high"].max()
            if np.isfinite(london_hi):
                wick_pct = rng.uniform(0.003, 0.006)
                sweep_high = london_hi * (1 + wick_pct)
                pick = rng.integers(2, min(10, len(window2)))
                idx  = window2.index[pick]
                df.at[idx, "high"]  = sweep_high
                df.at[idx, "close"] = london_hi * 0.999

    return df


if __name__ == "__main__":
    df = generate_synthetic_btc(days=60)
    print(f"Generated {len(df):,} candles")
    print(df.head())
    print(f"\nPrice range: ${df['close'].min():,.0f} – ${df['close'].max():,.0f}")
    df.to_parquet("/sessions/sleepy-adoring-noether/mnt/outputs/btcusdt_1m_synthetic.parquet")
    print("Saved: btcusdt_1m_synthetic.parquet")
