"""
Vitaliy Kronos Project — ICT Session Backtester (BTC/USDT)
Strategy:
  1. Mark Asia (00:00–08:00 UTC) and London (08:00–13:00 UTC) H/L each day
  2. Detect sweep of those levels in 13:00–16:00 UTC window
     (wick beyond + close back inside = liquidity grab)
  3. Find Fair Value Gap on the 1-min chart in the reaction
  4. Enter at FVG midpoint on pullback
  5. SL = sweep extreme; TP = draw on liquidity (or 2R fallback)
  6. One trade per 2-hour cooldown, 5x leverage, 1% account risk
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime, timezone, timedelta


# ──────────────────────────────────────────────
# DATA FETCHING (live — requires network access)
# ──────────────────────────────────────────────

def fetch_binance_klines(symbol="BTCUSDT", interval="1m", start_dt=None, end_dt=None):
    import requests
    url = "https://api.binance.us/api/v3/klines"
    all_klines, start_ms, end_ms = [], int(start_dt.timestamp()*1000), int(end_dt.timestamp()*1000)
    while start_ms < end_ms:
        r = requests.get(url, params={"symbol":symbol,"interval":interval,
                                       "startTime":start_ms,"endTime":end_ms,"limit":1000}, timeout=15)
        r.raise_for_status()
        data = r.json()
        if not data: break
        all_klines.extend(data)
        start_ms = data[-1][0] + 60_000
    cols = ["open_time","open","high","low","close","volume","close_time",
            "quote_vol","trades","tbbase","tbquote","ignore"]
    df = pd.DataFrame(all_klines, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open","high","low","close","volume"]: df[c] = df[c].astype(float)
    df.set_index("open_time", inplace=True)
    return df[["open","high","low","close","volume"]]


# ──────────────────────────────────────────────
# CORE ENGINE (numpy-vectorised for speed)
# ──────────────────────────────────────────────

def _arrays(df):
    """Extract numpy arrays + time-index lookup dict once."""
    return {
        "open":  df["open"].values,
        "high":  df["high"].values,
        "low":   df["low"].values,
        "close": df["close"].values,
        "ts":    df.index,                       # DatetimeIndex
        "hour":  df.index.hour,
        "date":  np.array(df.index.date),
        "loc":   {t: i for i, t in enumerate(df.index)},  # O(1) lookup
    }


def session_ranges(a):
    """Per-date Asia/London H/L using vectorised numpy grouping."""
    dates = np.unique(a["date"])
    out = {}
    for d in dates:
        mask = a["date"] == d
        h = a["hour"][mask]
        hi = a["high"][mask]
        lo = a["low"][mask]

        am = h < 8
        lm = (h >= 8) & (h < 13)

        out[d] = {
            "asia_high":   hi[am].max()  if am.sum() >= 10 else None,
            "asia_low":    lo[am].min()  if am.sum() >= 10 else None,
            "london_high": hi[lm].max()  if lm.sum() >= 10 else None,
            "london_low":  lo[lm].min()  if lm.sum() >= 10 else None,
        }
    return out


def detect_sweeps(a, ranges, h_start=13, h_end=16):
    """Vectorised sweep detection — wick beyond level + close back inside."""
    sweeps = []
    dates = np.unique(a["date"])
    for d in dates:
        r = ranges.get(d, {})
        mask = (a["date"] == d) & (a["hour"] >= h_start) & (a["hour"] < h_end)
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            continue
        for sess in ("asia", "london"):
            hi = r.get(f"{sess}_high")
            lo = r.get(f"{sess}_low")
            if hi is None or lo is None:
                continue
            for i in idxs:
                # Bullish sweep: wick below low, close back above
                if a["low"][i] < lo and a["close"][i] > lo:
                    sweeps.append({"pos": i, "date": d, "session": sess,
                                   "direction": "bullish", "level": lo,
                                   "extreme": a["low"][i], "time": a["ts"][i]})
                # Bearish sweep: wick above high, close back below
                elif a["high"][i] > hi and a["close"][i] < hi:
                    sweeps.append({"pos": i, "date": d, "session": sess,
                                   "direction": "bearish", "level": hi,
                                   "extreme": a["high"][i], "time": a["ts"][i]})
    return sweeps


def find_fvg(a, sweep_pos, direction, lookforward=30):
    """Scan next `lookforward` candles for an FVG in trade direction."""
    end = min(sweep_pos + lookforward, len(a["high"]) - 2)
    for i in range(sweep_pos + 1, end):
        ph, nl = a["high"][i-1], a["low"][i+1]
        pl, nh = a["low"][i-1],  a["high"][i+1]
        if direction == "bullish" and ph < nl:
            mid = (nl + ph) / 2
            return {"pos": i, "top": nl, "bot": ph, "mid": mid,
                    "direction": "bullish", "entry": mid}
        if direction == "bearish" and pl > nh:
            mid = (pl + nh) / 2
            return {"pos": i, "top": pl, "bot": nh, "mid": mid,
                    "direction": "bearish", "entry": mid}
    return None


def draw_on_liquidity(a, sweep_pos, direction, lookback=200):
    """Find nearest swing H/L above/below price = next liquidity target."""
    start = max(0, sweep_pos - lookback)
    hi = a["high"][start:sweep_pos]
    lo = a["low"][start:sweep_pos]
    cur = a["close"][sweep_pos]

    # 5-candle local swing
    swing_highs, swing_lows = [], []
    for j in range(2, len(hi) - 2):
        if hi[j] > hi[j-1] and hi[j] > hi[j-2] and hi[j] > hi[j+1] and hi[j] > hi[j+2]:
            swing_highs.append(hi[j])
        if lo[j] < lo[j-1] and lo[j] < lo[j-2] and lo[j] < lo[j+1] and lo[j] < lo[j+2]:
            swing_lows.append(lo[j])

    if direction == "bullish":
        above = [h for h in swing_highs if h > cur]
        return min(above) if above else None
    else:
        below = [l for l in swing_lows if l < cur]
        return max(below) if below else None


def simulate_trade(a, sweep, fvg, dol, risk_pct=0.01, fallback_rr=2.0, max_hold=120):
    direction = sweep["direction"]
    buf = fvg["entry"] * 0.0005

    sl = (sweep["extreme"] - buf) if direction == "bullish" else (sweep["extreme"] + buf)
    risk = abs(fvg["entry"] - sl)
    if risk < fvg["entry"] * 0.0001:   # < 0.01% risk = skip
        return None

    # TP
    if dol is not None and abs(dol - fvg["entry"]) / risk >= 1.0:
        tp  = dol
        rr  = abs(tp - fvg["entry"]) / risk
    else:
        rr  = fallback_rr
        tp  = fvg["entry"] + rr * risk * (1 if direction == "bullish" else -1)

    # Find fill: first candle that touches FVG zone after the FVG candle
    fill_pos = None
    entry_price = fvg["mid"]
    for i in range(fvg["pos"], min(fvg["pos"] + 30, len(a["high"]))):
        if direction == "bullish" and a["low"][i] <= fvg["top"] and a["low"][i] >= fvg["bot"]:
            fill_pos = i; break
        if direction == "bearish" and a["high"][i] >= fvg["bot"] and a["high"][i] <= fvg["top"]:
            fill_pos = i; break

    if fill_pos is None:
        return None

    entry_price = fvg["mid"]
    risk = abs(entry_price - sl)
    if risk == 0: return None
    tp = entry_price + rr * risk * (1 if direction == "bullish" else -1)

    # Walk forward
    entry_time = a["ts"][fill_pos]
    for i in range(fill_pos + 1, min(fill_pos + max_hold, len(a["high"]))):
        if direction == "bullish":
            if a["low"][i]  <= sl: return _result(entry_time, a["ts"][i], direction, entry_price, sl, tp, sl,  -1.0, risk_pct, "loss")
            if a["high"][i] >= tp: return _result(entry_time, a["ts"][i], direction, entry_price, sl, tp, tp,   rr,  risk_pct, "win")
        else:
            if a["high"][i] >= sl: return _result(entry_time, a["ts"][i], direction, entry_price, sl, tp, sl,  -1.0, risk_pct, "loss")
            if a["low"][i]  <= tp: return _result(entry_time, a["ts"][i], direction, entry_price, sl, tp, tp,   rr,  risk_pct, "win")

    # Timeout
    ep = a["close"][min(fill_pos + max_hold, len(a["high"]) - 1)]
    pnl_r = ((ep - entry_price) if direction == "bullish" else (entry_price - ep)) / risk
    return _result(entry_time, a["ts"][min(fill_pos + max_hold, len(a["high"])-1)],
                   direction, entry_price, sl, tp, ep, pnl_r, risk_pct, "timeout")


def _result(entry_time, exit_time, direction, entry, sl, tp, exit_px, rr, risk_pct, outcome):
    return {
        "entry_time":  entry_time,
        "exit_time":   exit_time,
        "direction":   direction,
        "entry":       round(entry, 2),
        "sl":          round(sl, 2),
        "tp":          round(tp, 2),
        "exit_price":  round(exit_px, 2),
        "rr_achieved": round(rr, 3),
        "outcome":     outcome,
        "pnl_pct":     round(risk_pct * rr * 100, 4),  # % of account per trade
    }


# ──────────────────────────────────────────────
# FULL PIPELINE
# ──────────────────────────────────────────────

def run_backtest(df, risk_pct=0.01, label="ICT", verbose=True, kronos_signals=None):
    if verbose:
        print(f"\n{'='*55}\n Running: {label}\n{'='*55}")

    a = _arrays(df)

    if verbose: print("Session ranges...")
    ranges = session_ranges(a)

    if verbose: print("Sweep detection (13:00–16:00 UTC)...")
    sweeps = detect_sweeps(a, ranges)
    if verbose: print(f"  -> {len(sweeps)} raw sweeps")

    trades, last_exit_pos = [], -1
    COOLDOWN = 120   # candles (= 120 min)

    for sw in sweeps:
        pos = sw["pos"]
        if pos - last_exit_pos < COOLDOWN:
            continue

        # Kronos filter: skip if forecast contradicts sweep direction
        if kronos_signals is not None:
            sig = kronos_signals.get(sw["time"].date())
            if sig is not None and sig != sw["direction"]:
                continue

        fvg = find_fvg(a, pos, sw["direction"])
        if fvg is None:
            continue

        dol = draw_on_liquidity(a, pos, sw["direction"])
        t   = simulate_trade(a, sw, fvg, dol, risk_pct=risk_pct)
        if t is None:
            continue

        t["session"] = sw["session"]
        t["label"]   = label
        trades.append(t)
        last_exit_pos = pos + COOLDOWN  # next trade after cooldown

    if verbose: print(f"  -> {len(trades)} trades")
    return trades


def compute_stats(trades, label=""):
    if not trades:
        return {"label": label, "trades": 0, "win_rate": 0,
                "avg_rr": 0, "total_pnl_pct": 0, "max_drawdown_pct": 0}

    df_t = pd.DataFrame(trades)
    wins = (df_t["outcome"] == "win").sum()

    eq = [100.0]
    for _, r in df_t.iterrows():
        eq.append(eq[-1] * (1 + r["pnl_pct"] / 100))
    eq = pd.Series(eq)
    dd = ((eq - eq.cummax()) / eq.cummax() * 100).min()

    return {
        "label":           label,
        "trades":          len(df_t),
        "win_rate":        round(wins / len(df_t) * 100, 1),
        "avg_rr":          round(df_t["rr_achieved"].mean(), 2),
        "total_pnl_pct":   round(df_t["pnl_pct"].sum(), 2),
        "max_drawdown_pct":round(dd, 2),
        "equity":          eq.tolist(),
        "trades_df":       df_t,
    }


# ──────────────────────────────────────────────
# MAIN — swap USE_LIVE=True when on real network
# ──────────────────────────────────────────────

if __name__ == "__main__":
    USE_LIVE = False   # set True to pull real Binance data

    if USE_LIVE:
        end_dt   = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        start_dt = end_dt - timedelta(days=60)
        print(f"Fetching live BTC/USDT 1m: {start_dt.date()} -> {end_dt.date()}")
        df = fetch_binance_klines("BTCUSDT", "1m", start_dt, end_dt)
    else:
        from synthetic_data import generate_synthetic_btc
        print("Generating synthetic BTC/USDT 1m data (60 days)...")
        df = generate_synthetic_btc(days=60)

    print(f"  {len(df):,} candles — ${df['close'].min():,.0f}–${df['close'].max():,.0f}")

    trades = run_backtest(df, risk_pct=0.01, label="ICT_only")
    stats  = compute_stats(trades, label="ICT only")

    print("\n── Results ──")
    for k, v in stats.items():
        if k not in ("equity", "trades_df"):
            print(f"  {k}: {v}")

    pd.DataFrame(trades).to_csv("ict_trades.csv", index=False)
    json.dump({k: v for k, v in stats.items() if k != "trades_df"},
              open("ict_stats.json", "w"), indent=2)
    print("\nSaved: ict_trades.csv, ict_stats.json")
