"""
Vitaliy Kronos Project -- Synthetic data generators
  generate_synthetic_btc()    -- crypto OHLCV (UTC, 24/7)
  generate_synthetic_equity() -- stock OHLCV (ET, market hours + premarket)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta


def generate_synthetic_btc(days=60, start_price=105_000.0, seed=42):
    """GARCH-like BTC 1-min candles, UTC, 24/7."""
    rng = np.random.default_rng(seed)
    n   = days * 24 * 60
    start = pd.Timestamp("2025-01-01", tz="UTC")
    idx = pd.date_range(start, periods=n, freq="1min", tz="UTC")

    vol = 0.0003
    prices = [start_price]
    vols   = [vol]
    for _ in range(n - 1):
        v = np.clip(0.94 * vols[-1] + 0.06 * abs(rng.normal(0, 0.0003)), 0.0001, 0.003)
        r = rng.normal(0, v)
        prices.append(max(prices[-1] * (1 + r), 1000.0))
        vols.append(v)

    closes = np.array(prices)
    df = _to_ohlcv(closes, idx, rng)
    df = _inject_sweeps(df, rng)
    return df


def generate_synthetic_equity(symbol="QQQ", days=60, start_price=490.0, seed=42):
    """
    QQQ/SPY-like 1-min candles in America/New_York timezone.
    Includes premarket (04:00-09:29) + regular session (09:30-15:59).
    Weekends excluded.
    """
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
    rng = np.random.default_rng(seed)

    # Build minute-by-minute index for market days only (Mon-Fri)
    base = pd.Timestamp("2025-01-02", tz="America/New_York")  # Thursday
    rows = []
    price = start_price
    vol   = 0.0002  # much lower than crypto

    for day_offset in range(days + 20):  # overshoot, trim to `days` trading days
        day = base + timedelta(days=day_offset)
        if day.weekday() >= 5:  # Sat/Sun
            continue
        # Premarket: 04:00 - 09:29
        for h in range(4, 10):
            for m in range(60):
                if h == 9 and m >= 30: break
                ts = day.replace(hour=h, minute=m, second=0, microsecond=0)
                vol = np.clip(0.95 * vol + 0.05 * abs(rng.normal(0, 0.00015)), 0.00005, 0.001)
                r   = rng.normal(0, vol * 0.5)  # lower vol in premarket
                price = max(price * (1 + r), 1.0)
                rows.append((ts, price))
        # Regular session: 09:30 - 15:59
        for h in range(9, 16):
            for m in range(60):
                if h == 9 and m < 30: continue
                ts = day.replace(hour=h, minute=m, second=0, microsecond=0)
                # Higher vol at open/close
                if (h == 9 and m < 45) or (h == 15 and m >= 45):
                    base_vol = 0.0004
                else:
                    base_vol = 0.00018
                vol = np.clip(0.94 * vol + 0.06 * abs(rng.normal(0, base_vol)), 0.00005, 0.002)
                r   = rng.normal(0, vol)
                price = max(price * (1 + r), 1.0)
                rows.append((ts, price))
        if len({r[0].date() for r in rows}) >= days:
            break

    idx    = pd.DatetimeIndex([r[0] for r in rows])
    closes = np.array([r[1] for r in rows])
    df = _to_ohlcv(closes, idx, rng, spread_factor=0.3)
    df = _inject_equity_sweeps(df, rng)
    return df


# ── helpers ──

def _to_ohlcv(closes, idx, rng, spread_factor=1.0):
    n = len(closes)
    spread = closes * 0.0008 * spread_factor
    highs  = closes + rng.uniform(0, spread)
    lows   = closes - rng.uniform(0, spread)
    opens  = np.roll(closes, 1); opens[0] = closes[0]
    opens += rng.normal(0, spread * 0.3)
    opens  = np.clip(opens, lows, highs)
    vol    = rng.uniform(100, 1000, n) * closes / closes.mean()
    return pd.DataFrame({"open": opens, "high": highs, "low": lows,
                         "close": closes, "volume": vol}, index=idx)


def _inject_sweeps(df, rng, every_n_days=3):
    """Force engineered sweep events every Nth day for testability (crypto)."""
    dates = pd.Series(df.index.date).unique()
    for i, d in enumerate(dates):
        if i % every_n_days != 0: continue
        day_mask = df.index.date == d
        hour = df.index.hour[day_mask]
        # Asia range
        asia = df[day_mask][hour < 8]
        if len(asia) < 10: continue
        asia_lo = asia["low"].min()
        # Inject bullish sweep candle at 13:30 UTC
        trade_mask = day_mask & (df.index.hour == 13) & (df.index.minute == 30)
        if trade_mask.sum() == 0: continue
        idx = df.index[trade_mask][0]
        df.loc[idx, "low"]   = asia_lo * 0.998
        df.loc[idx, "close"] = asia_lo * 1.001
    return df


def _inject_equity_sweeps(df, rng, every_n_days=3):
    """Force engineered sweep events every Nth trading day (equity)."""
    dates = pd.Series(df.index.date).unique()
    for i, d in enumerate(dates):
        if i % every_n_days != 0: continue
        day_mask = df.index.date == d
        hour = df.index.hour[day_mask]
        # Premarket range (4-9:30)
        pm = df[day_mask][(hour >= 4) & (hour < 10)]
        pm = pm[~((pm.index.hour == 9) & (pm.index.minute >= 30))]
        if len(pm) < 5: continue
        pm_lo = pm["low"].min()
        # Inject bullish sweep at 09:35 ET
        trade_mask = (day_mask & (df.index.hour == 9) & (df.index.minute == 35))
        if trade_mask.sum() == 0: continue
        idx = df.index[trade_mask][0]
        df.loc[idx, "low"]   = pm_lo * 0.998
        df.loc[idx, "close"] = pm_lo * 1.001
    return df
