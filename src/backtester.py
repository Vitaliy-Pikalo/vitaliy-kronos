"""
Vitaliy Kronos Project -- ICT Session Backtester
Supports BTC/USDT (Binance.US, UTC sessions) and QQQ/SPY (yfinance, ET sessions).

Strategy:
  1. Mark overnight/premarket range (or Asia/London for crypto)
  2. Detect sweep of those levels in trade window
  3. Find Fair Value Gap on 1-min chart
  4. Enter at FVG midpoint, SL at sweep extreme, TP at draw on liquidity
  5. Optional Kronos directional filter
"""

import numpy as np
import pandas as pd
import json
from datetime import datetime, timezone, timedelta

# ── Market configs ──
# session_a_hours / session_b_hours: integer hour range in the index timezone
# trade_windows: list of (start_min, end_min) tuples in minutes-since-midnight
MARKET_CONFIGS = {
    "btc": {
        "label":           "BTC/USDT",
        "tz":              "UTC",
        "session_a_hours": (0, 8),               # Asia: 00:00-08:00 UTC
        "session_b_hours": (8, 13),              # London: 08:00-13:00 UTC
        "trade_windows":   [(13*60, 16*60)],     # London-NY overlap: 13:00-16:00 UTC
        "min_sess_candles": 10,
    },
    "equity": {
        "label":           "QQQ/SPY",
        "tz":              "America/New_York",
        "session_a_hours": (4, 10),              # Premarket: 04:00-10:00 ET
        "session_b_hours": (4, 8),               # Early premarket: 04:00-08:00 ET
        "trade_windows":   [(9*60+30, 11*60+30), # NY open: 09:30-11:30 ET
                            (14*60,   16*60)],   # Power hour: 14:00-16:00 ET
        "min_sess_candles": 5,
    },
}


# ── Data fetching ──

def fetch_binance_klines(symbol="BTCUSDT", interval="1m", start_dt=None, end_dt=None):
    import requests
    url = "https://api.binance.us/api/v3/klines"
    all_klines, start_ms, end_ms = [], int(start_dt.timestamp()*1000), int(end_dt.timestamp()*1000)
    while start_ms < end_ms:
        r = requests.get(url, params={"symbol": symbol, "interval": interval,
                                      "startTime": start_ms, "endTime": end_ms, "limit": 1000},
                         timeout=15)
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


def fetch_yfinance_klines(symbol="QQQ", days=7, interval="1m"):
    """Pull equity 1-min data via yfinance (max 7 days for 1m, 60 days for 5m)."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("Run: pip install yfinance")
    from zoneinfo import ZoneInfo
    ET = ZoneInfo("America/New_York")
    period = f"{min(days, 7)}d" if interval == "1m" else f"{min(days, 60)}d"
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=period, interval=interval, prepost=True)
    df.columns = [c.lower() for c in df.columns]
    df = df[["open","high","low","close","volume"]]
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("America/New_York")
    else:
        df.index = df.index.tz_convert("America/New_York")
    return df


# ── Core engine (numpy-vectorised) ──

def _arrays(df):
    return {
        "open":  df["open"].values,
        "high":  df["high"].values,
        "low":   df["low"].values,
        "close": df["close"].values,
        "ts":    df.index,
        "hour":  df.index.hour,
        "mins":  df.index.hour * 60 + df.index.minute,  # minutes since midnight
        "date":  np.array(df.index.date),
        "loc":   {t: i for i, t in enumerate(df.index)},
    }


def session_ranges(a, market_cfg=None):
    """Per-date range for session A (Asia/premarket) and session B (London/early-premarket)."""
    if market_cfg is None:
        market_cfg = MARKET_CONFIGS["btc"]
    sa = market_cfg["session_a_hours"]
    sb = market_cfg["session_b_hours"]
    mc = market_cfg["min_sess_candles"]

    dates = np.unique(a["date"])
    out = {}
    for d in dates:
        mask = a["date"] == d
        h  = a["hour"][mask]
        hi = a["high"][mask]
        lo = a["low"][mask]

        am = (h >= sa[0]) & (h < sa[1])
        lm = (h >= sb[0]) & (h < sb[1])

        out[d] = {
            "asia_high":   hi[am].max() if am.sum() >= mc else None,
            "asia_low":    lo[am].min() if am.sum() >= mc else None,
            "london_high": hi[lm].max() if lm.sum() >= mc else None,
            "london_low":  lo[lm].min() if lm.sum() >= mc else None,
        }
    return out


def detect_sweeps(a, ranges, trade_windows=None):
    """
    Sweep detection across all trade windows.
    trade_windows: list of (start_min, end_min) tuples in minutes-since-midnight.
    """
    if trade_windows is None:
        trade_windows = [(13*60, 16*60)]  # default: BTC 13:00-16:00 UTC

    sweeps = []
    dates = np.unique(a["date"])
    for d in dates:
        r = ranges.get(d, {})
        day_mask = a["date"] == d
        # Build combined window mask
        win_mask = np.zeros(len(a["mins"]), dtype=bool)
        for (ws, we) in trade_windows:
            win_mask |= (day_mask & (a["mins"] >= ws) & (a["mins"] < we))
        idxs = np.where(win_mask)[0]
        if len(idxs) == 0:
            continue
        for sess in ("asia", "london"):
            hi = r.get(f"{sess}_high")
            lo = r.get(f"{sess}_low")
            if hi is None or lo is None:
                continue
            for i in idxs:
                if a["low"][i] < lo and a["close"][i] > lo:
                    sweeps.append({"pos": i, "date": d, "session": sess,
                                   "direction": "bullish", "level": lo,
                                   "extreme": a["low"][i], "time": a["ts"][i]})
                elif a["high"][i] > hi and a["close"][i] < hi:
                    sweeps.append({"pos": i, "date": d, "session": sess,
                                   "direction": "bearish", "level": hi,
                                   "extreme": a["high"][i], "time": a["ts"][i]})
    return sweeps


def find_fvg(a, sweep_pos, direction, lookforward=30, require_displacement=False):
    if require_displacement:
        body_start = max(0, sweep_pos - 20)
        bodies = np.abs(a["close"][body_start:sweep_pos] - a["open"][body_start:sweep_pos])
        avg_body = bodies.mean() if len(bodies) > 5 else 0
    else:
        avg_body = 0
    end = min(sweep_pos + lookforward, len(a["high"]) - 2)
    for i in range(sweep_pos + 1, end):
        # Displacement filter: require a strong impulse candle (1.5x avg body)
        if require_displacement and avg_body > 0:
            body = abs(a["close"][i] - a["open"][i])
            if body < 1.5 * avg_body:
                continue
        ph, nl = a["high"][i-1], a["low"][i+1]
        pl, nh = a["low"][i-1],  a["high"][i+1]
        if direction == "bullish" and ph < nl:
            mid = (nl + ph) / 2
            return {"pos": i, "top": nl, "bot": ph, "mid": mid, "direction": "bullish", "entry": mid}
        if direction == "bearish" and pl > nh:
            mid = (pl + nh) / 2
            return {"pos": i, "top": pl, "bot": nh, "mid": mid, "direction": "bearish", "entry": mid}
    return None


def draw_on_liquidity(a, sweep_pos, direction, lookback=200):
    start = max(0, sweep_pos - lookback)
    hi = a["high"][start:sweep_pos]
    lo = a["low"][start:sweep_pos]
    cur = a["close"][sweep_pos]
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


def simulate_trade(a, sweep, fvg, dol, risk_pct=0.01, fallback_rr=2.0, max_hold=120,
                   use_ote=False, tight_sl=False):
    direction = sweep["direction"]
    fvg_range = fvg["top"] - fvg["bot"]

    # Entry: OTE (62% deep into FVG from entry side) or midpoint
    if use_ote and fvg_range > 0:
        entry_price = (fvg["bot"] + 0.38 * fvg_range) if direction == "bullish" \
                      else (fvg["top"] - 0.38 * fvg_range)
    else:
        entry_price = fvg["mid"]

    # SL: tight (just beyond FVG edge) or wide (sweep extreme)
    buf = entry_price * 0.0003
    if tight_sl:
        sl = (fvg["bot"] - buf) if direction == "bullish" else (fvg["top"] + buf)
    else:
        buf = entry_price * 0.0005
        sl = (sweep["extreme"] - buf) if direction == "bullish" else (sweep["extreme"] + buf)

    risk = abs(entry_price - sl)
    if risk < entry_price * 0.0001:
        return None
    if dol is not None and abs(dol - entry_price) / risk >= 1.0:
        tp = dol
        rr = abs(tp - entry_price) / risk
    else:
        rr = fallback_rr
        tp = entry_price + rr * risk * (1 if direction == "bullish" else -1)
    fill_pos = None
    for i in range(fvg["pos"], min(fvg["pos"] + 30, len(a["high"]))):
        if direction == "bullish" and a["low"][i] <= fvg["top"] and a["high"][i] >= fvg["bot"]:
            fill_pos = i; break
        if direction == "bearish" and a["high"][i] >= fvg["bot"] and a["low"][i] <= fvg["top"]:
            fill_pos = i; break
    if fill_pos is None:
        return None
    risk = abs(entry_price - sl)
    if risk == 0: return None
    tp = entry_price + rr * risk * (1 if direction == "bullish" else -1)
    entry_time = a["ts"][fill_pos]
    for i in range(fill_pos + 1, min(fill_pos + max_hold, len(a["high"]))):
        if direction == "bullish":
            if a["low"][i]  <= sl: return _result(entry_time, a["ts"][i], direction, entry_price, sl, tp, sl,   -1.0, risk_pct, "loss")
            if a["high"][i] >= tp: return _result(entry_time, a["ts"][i], direction, entry_price, sl, tp, tp,    rr,  risk_pct, "win")
        else:
            if a["high"][i] >= sl: return _result(entry_time, a["ts"][i], direction, entry_price, sl, tp, sl,   -1.0, risk_pct, "loss")
            if a["low"][i]  <= tp: return _result(entry_time, a["ts"][i], direction, entry_price, sl, tp, tp,    rr,  risk_pct, "win")
    ep = a["close"][min(fill_pos + max_hold, len(a["high"]) - 1)]
    pnl_r = ((ep - entry_price) if direction == "bullish" else (entry_price - ep)) / risk
    return _result(entry_time, a["ts"][min(fill_pos + max_hold, len(a["high"])-1)],
                   direction, entry_price, sl, tp, ep, pnl_r, risk_pct, "timeout")


def _result(entry_time, exit_time, direction, entry, sl, tp, exit_px, rr, risk_pct, outcome):
    return {
        "entry_time":  entry_time, "exit_time": exit_time,
        "direction":   direction,  "entry":     round(entry, 2),
        "sl":          round(sl, 2), "tp":       round(tp, 2),
        "exit_price":  round(exit_px, 2), "rr_achieved": round(rr, 3),
        "outcome":     outcome,
        "pnl_pct":     round(risk_pct * rr * 100, 4),
    }


# ── Full pipeline ──

def run_backtest(df, risk_pct=0.01, label="ICT", verbose=True,
                 kronos_signals=None, market="btc",
                 use_ote=False, tight_sl=False, require_displacement=False,
                 max_hold=120, fallback_rr=2.0):
    market_cfg = MARKET_CONFIGS.get(market, MARKET_CONFIGS["btc"])
    if verbose:
        flags = []
        if use_ote: flags.append("OTE")
        if tight_sl: flags.append("TightSL")
        if require_displacement: flags.append("Displacement")
        tag = f" [{', '.join(flags)}]" if flags else ""
        print(f"\n{'='*55}\n Running: {label}{tag} [{market_cfg['label']}]\n{'='*55}")

    a = _arrays(df)
    if verbose: print("Session ranges...")
    ranges = session_ranges(a, market_cfg)

    trade_windows = market_cfg["trade_windows"]
    window_str = ", ".join(f"{ws//60:02d}:{ws%60:02d}-{we//60:02d}:{we%60:02d}" for ws, we in trade_windows)
    if verbose: print(f"Sweep detection ({window_str})...")
    sweeps = detect_sweeps(a, ranges, trade_windows)
    if verbose: print(f"  -> {len(sweeps)} raw sweeps")

    trades, last_exit_pos = [], -1
    COOLDOWN = 120

    for sw in sweeps:
        pos = sw["pos"]
        if pos - last_exit_pos < COOLDOWN:
            continue
        if kronos_signals is not None:
            sig = kronos_signals.get(sw["time"].date())
            if sig is not None and sig != sw["direction"]:
                continue
        fvg = find_fvg(a, pos, sw["direction"], require_displacement=require_displacement)
        if fvg is None:
            continue
        dol = draw_on_liquidity(a, pos, sw["direction"])
        t   = simulate_trade(a, sw, fvg, dol, risk_pct=risk_pct, max_hold=max_hold,
                              use_ote=use_ote, tight_sl=tight_sl, fallback_rr=fallback_rr)
        if t is None:
            continue
        t["session"] = sw["session"]
        t["label"]   = label
        trades.append(t)
        last_exit_pos = pos + COOLDOWN

    if verbose: print(f"  -> {len(trades)} trades")
    return trades


def compute_stats(trades, label=""):
    if not trades:
        return {"label": label, "trades": 0, "win_rate": 0,
                "avg_rr": 0, "total_pnl_pct": 0, "max_drawdown_pct": 0, "equity": [100.0]}
    df_t = pd.DataFrame(trades)
    wins = (df_t["outcome"] == "win").sum()
    eq = [100.0]
    for _, r in df_t.iterrows():
        eq.append(eq[-1] * (1 + r["pnl_pct"] / 100))
    eq = pd.Series(eq)
    dd = ((eq - eq.cummax()) / eq.cummax() * 100).min()
    return {
        "label":            label,
        "trades":           len(df_t),
        "win_rate":         round(wins / len(df_t) * 100, 1),
        "avg_rr":           round(df_t["rr_achieved"].mean(), 2),
        "total_pnl_pct":    round(df_t["pnl_pct"].sum(), 2),
        "max_drawdown_pct": round(dd, 2),
        "equity":           eq.tolist(),
        "trades_df":        df_t,
    }
