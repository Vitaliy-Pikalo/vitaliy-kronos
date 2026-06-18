"""
Vitaliy Kronos -- ICT Signal Bot
Supports BTC/USDT (Binance.US) and equity (QQQ/SPY via yfinance).
Set SYMBOL in config.env to switch markets.

Setup:
  1. cp config.example.env config.env  -- fill in credentials + set SYMBOL
  2. pip install requests pandas yfinance
  3. PYTHONUTF8=1 python src/signal_bot.py
"""

import sys, time, smtplib, requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from zoneinfo import ZoneInfo

CONFIG_PATH = Path(__file__).parent.parent / "config.env"

ET  = ZoneInfo("America/New_York")
UTC = timezone.utc


# ── config ──

def load_config():
    cfg = {}
    if not CONFIG_PATH.exists():
        print(f"[ERROR] config.env not found. Copy config.example.env and fill in credentials.")
        sys.exit(1)
    for line in CONFIG_PATH.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            cfg[k.strip()] = v.strip()
    return cfg


# ── data fetching ──

def fetch_btc_candles(hours=48):
    url = "https://api.binance.us/api/v3/klines"
    end_ms   = int(datetime.now(UTC).timestamp() * 1000)
    start_ms = end_ms - hours * 3600 * 1000
    all_klines = []
    while start_ms < end_ms:
        r = requests.get(url, params={"symbol": "BTCUSDT", "interval": "1m",
                                      "startTime": start_ms, "endTime": end_ms, "limit": 1000}, timeout=15)
        r.raise_for_status()
        data = r.json()
        if not data: break
        all_klines.extend(data)
        start_ms = data[-1][0] + 60_000
    cols = ["open_time","open","high","low","close","volume","close_time","qv","trades","tbbase","tbquote","ignore"]
    df = pd.DataFrame(all_klines, columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for c in ["open","high","low","close","volume"]: df[c] = df[c].astype(float)
    df.set_index("open_time", inplace=True)
    return df[["open","high","low","close","volume"]]


def fetch_equity_candles(symbol="QQQ", days=5):
    import yfinance as yf
    ticker = yf.Ticker(symbol)
    df = ticker.history(period=f"{days}d", interval="1m", prepost=True)
    df.columns = [c.lower() for c in df.columns]
    df = df[["open","high","low","close","volume"]]
    if df.index.tzinfo is None:
        df.index = df.index.tz_localize("America/New_York")
    else:
        df.index = df.index.tz_convert("America/New_York")
    return df


# ── session ranges ──

def get_ranges_btc(df):
    today = datetime.now(UTC).date()
    day = df[df.index.date == today]
    h = day.index.hour
    asia   = day[h < 8]
    london = day[(h >= 8) & (h < 13)]
    return {
        "asia_high":   asia["high"].max()   if len(asia)   >= 10 else None,
        "asia_low":    asia["low"].min()    if len(asia)   >= 10 else None,
        "london_high": london["high"].max() if len(london) >= 10 else None,
        "london_low":  london["low"].min()  if len(london) >= 10 else None,
    }


def get_ranges_equity(df):
    today = datetime.now(ET).date()
    day = df[df.index.date == today]
    h = day.index.hour
    m = day.index.minute
    # Premarket: 04:00-09:30 ET
    pm = day[(h >= 4) & ~((h == 9) & (m >= 30)) & (h < 10)]
    # Early premarket: 04:00-08:00 ET
    early = day[(h >= 4) & (h < 8)]
    return {
        "asia_high":   pm["high"].max()    if len(pm)    >= 5 else None,
        "asia_low":    pm["low"].min()     if len(pm)    >= 5 else None,
        "london_high": early["high"].max() if len(early) >= 5 else None,
        "london_low":  early["low"].min()  if len(early) >= 5 else None,
    }


# ── sweep + FVG ──

def check_for_setup(df, ranges, lookback=5):
    recent = df.iloc[-lookback:]
    for sess in ("asia", "london"):
        hi = ranges.get(f"{sess}_high")
        lo = ranges.get(f"{sess}_low")
        if hi is None or lo is None: continue
        for i in range(len(recent)):
            c = recent.iloc[i]
            if c["low"] < lo and c["close"] > lo:
                fvg = _find_fvg(df, len(df) - lookback + i, "bullish")
                if fvg: return _build_signal("bullish", sess, lo, c["low"], fvg)
            elif c["high"] > hi and c["close"] < hi:
                fvg = _find_fvg(df, len(df) - lookback + i, "bearish")
                if fvg: return _build_signal("bearish", sess, hi, c["high"], fvg)
    return None


def _find_fvg(df, sweep_pos, direction, lookforward=20):
    end = min(sweep_pos + lookforward, len(df) - 2)
    for i in range(sweep_pos + 1, end):
        ph, nl = df.iloc[i-1]["high"], df.iloc[i+1]["low"]
        pl, nh = df.iloc[i-1]["low"],  df.iloc[i+1]["high"]
        if direction == "bullish" and ph < nl: return {"top": nl, "bot": ph, "mid": (nl+ph)/2}
        if direction == "bearish" and pl > nh: return {"top": pl, "bot": nh, "mid": (pl+nh)/2}
    return None


def _build_signal(direction, session, level, extreme, fvg, symbol="BTC"):
    entry = fvg["mid"]
    buf   = entry * 0.0005
    sl    = (extreme - buf) if direction == "bullish" else (extreme + buf)
    risk  = abs(entry - sl)
    tp    = entry + 2.0 * risk * (1 if direction == "bullish" else -1)
    rr    = abs(tp - entry) / risk if risk > 0 else 0
    tv_sym = "BINANCE:BTCUSDT" if symbol == "BTC" else f"NASDAQ:{symbol}"
    return {
        "time":      datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "symbol":    symbol,
        "direction": direction,
        "session":   session,
        "entry":     round(entry, 2),
        "sl":        round(sl, 2),
        "tp":        round(tp, 2),
        "rr":        round(rr, 2),
        "chart_url": f"https://www.tradingview.com/chart/?symbol={tv_sym}&interval=1",
    }


# ── trade windows ──

def in_trade_window_btc():
    now = datetime.now(UTC)
    return 13 <= now.hour < 16


def in_trade_window_equity():
    now = datetime.now(ET)
    if now.weekday() >= 5: return False  # weekend
    mins = now.hour * 60 + now.minute
    return (9*60+30 <= mins <= 11*60+30) or (14*60 <= mins <= 16*60)


def window_closes_at_btc():
    now = datetime.now(UTC)
    if now.hour >= 16: return None  # closed for today
    return 16 * 60  # 16:00 UTC in minutes


def window_closes_at_equity():
    now = datetime.now(ET)
    mins = now.hour * 60 + now.minute
    if mins < 9*60+30: return 9*60+30
    if mins <= 11*60+30: return 11*60+30
    if mins < 14*60: return 14*60
    if mins <= 16*60: return 16*60
    return None


# ── alerts ──

def send_discord(signal, webhook_url):
    direction_label = "LONG" if signal["direction"] == "bullish" else "SHORT"
    color = 0x22c55e if signal["direction"] == "bullish" else 0xef4444
    payload = {"embeds": [{"title": f"ICT SIGNAL -- {direction_label} {signal['symbol']}",
        "color": color,
        "fields": [
            {"name": "Session",    "value": signal["session"].upper(), "inline": True},
            {"name": "Direction",  "value": direction_label,           "inline": True},
            {"name": "Entry",      "value": f"${signal['entry']:,.2f}", "inline": True},
            {"name": "Stop Loss",  "value": f"${signal['sl']:,.2f}",   "inline": True},
            {"name": "Take Profit","value": f"${signal['tp']:,.2f}",   "inline": True},
            {"name": "R:R",        "value": f"1:{signal['rr']}",       "inline": True},
            {"name": "Chart",      "value": f"[TradingView]({signal['chart_url']})", "inline": False},
        ],
        "footer": {"text": f"Vitaliy Kronos Signal Bot | {signal['time']}"},
    }]}
    r = requests.post(webhook_url, json=payload, timeout=10)
    print(f"[Discord] {'OK' if r.status_code in (200,204) else f'Failed: {r.status_code}'}")


def send_email(signal, gmail_user, gmail_pass, recipient):
    direction_label = "LONG" if signal["direction"] == "bullish" else "SHORT"
    subject = f"ICT Signal: {direction_label} {signal['symbol']} @ ${signal['entry']:,.2f}"
    body = f"""ICT Session Signal Bot
======================
Time:      {signal['time']}
Symbol:    {signal['symbol']}
Direction: {direction_label}
Session:   {signal['session'].upper()} sweep

Entry:       ${signal['entry']:,.2f}
Stop Loss:   ${signal['sl']:,.2f}
Take Profit: ${signal['tp']:,.2f}
R:R:         1:{signal['rr']}

Chart: {signal['chart_url']}
-- Vitaliy Kronos Signal Bot"""
    msg = MIMEMultipart()
    msg["From"] = gmail_user; msg["To"] = recipient; msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=15) as s:
            s.login(gmail_user, gmail_pass); s.send_message(msg)
        print(f"[Email] Sent to {recipient}")
    except Exception as e:
        print(f"[Email] Failed: {e}")


def send_alerts(signal, cfg):
    if cfg.get("DISCORD_WEBHOOK"):
        send_discord(signal, cfg["DISCORD_WEBHOOK"])
    if cfg.get("GMAIL_USER") and cfg.get("GMAIL_APP_PASSWORD"):
        send_email(signal, cfg["GMAIL_USER"], cfg["GMAIL_APP_PASSWORD"],
                   cfg.get("ALERT_EMAIL", cfg["GMAIL_USER"]))


# ── main loop ──

def run_bot():
    cfg = load_config()

    # Determine market from config (default: BTC)
    symbol = cfg.get("SYMBOL", "BTC").upper().strip()
    is_equity = symbol not in ("BTC", "BTCUSDT")
    market_label = symbol if is_equity else "BTC/USDT"

    print(f"[Bot] Vitaliy Kronos Signal Bot | Market: {market_label}")
    print(f"[Bot] Started at {datetime.now(UTC).strftime('%H:%M UTC')} / {datetime.now(ET).strftime('%H:%M ET')}")
    if is_equity:
        print("[Bot] Trade windows: 09:30-11:30 ET (NY open) + 14:00-16:00 ET (power hour)")
    else:
        print("[Bot] Trade window: 13:00-16:00 UTC")

    in_window  = in_trade_window_equity if is_equity else in_trade_window_btc
    get_ranges = get_ranges_equity      if is_equity else get_ranges_btc
    fetch_data = (lambda: fetch_equity_candles(symbol, days=5)) if is_equity else (lambda: fetch_btc_candles(48))

    alerted = set()
    last_fetch = None
    df = None

    while True:
        now_et  = datetime.now(ET)
        now_utc = datetime.now(UTC)

        if not in_window():
            # Check if done for the day
            if is_equity and now_et.hour >= 16:
                print("\n[Bot] Market closed. Done for today.")
                break
            elif not is_equity and now_utc.hour >= 16:
                print("\n[Bot] Trade window closed. Done for today.")
                break
            print(f"[Bot] Waiting for trade window... {now_utc.strftime('%H:%M UTC')} / {now_et.strftime('%H:%M ET')}", end="\r")
            time.sleep(60)
            continue

        # Refresh data every 2 min
        if last_fetch is None or (now_utc - last_fetch).seconds > 120:
            try:
                print(f"\n[Bot] Fetching {market_label} data...")
                df = fetch_data()
                last_fetch = now_utc
                print(f"[Bot] {len(df):,} candles | latest close: ${df['close'].iloc[-1]:,.2f}")
            except Exception as e:
                print(f"\n[Bot] Fetch error: {e}")
                time.sleep(30)
                continue

        ranges = get_ranges(df)
        signal = check_for_setup(df, ranges, lookback=3)

        if signal:
            signal["symbol"] = market_label
            sig_key = f"{signal['direction']}_{signal['entry']}"
            if sig_key not in alerted:
                print(f"\n[Bot] *** SETUP: {signal['direction'].upper()} {market_label} @ ${signal['entry']:,.2f} ***")
                send_alerts(signal, cfg)
                alerted.add(sig_key)
        else:
            ts = now_et.strftime("%H:%M ET") if is_equity else now_utc.strftime("%H:%M UTC")
            print(f"[Bot] {ts} -- watching {market_label}... no setup", end="\r")

        time.sleep(60)

    print("\n[Bot] Done.")


if __name__ == "__main__":
    run_bot()
