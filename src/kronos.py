"""
Vitaliy Kronos Project — Kronos Directional Signal Generator
Uses Kronos-mini (4.1M params, HuggingFace) to produce a daily bullish/bearish
bias that filters ICT sweep trades in backtester.py.

How it works:
  1. For each day in the dataset, feed the prior 400 1-min candles into Kronos
  2. Predict the next 120 candles (2 hours ahead)
  3. Compare predicted close at t+120 vs current close
     → predicted_close > current:  "bullish"  (only take bullish ICT setups)
     → predicted_close < current:  "bearish"  (only take bearish ICT setups)
  4. Signal is generated at 12:30 UTC (30 min before trade window opens)

Requires: pip install transformers torch  (CPU works, just slow)
Falls back to MockKronos if torch/model unavailable — produces random signals
to allow dry-run testing of the pipeline.
"""

import sys
import numpy as np
import pandas as pd
from datetime import timezone

# ── Kronos repo path (cloned to C:/kronos) ──
KRONOS_REPO = "C:/kronos"

CONTEXT_LEN  = 400    # candles fed to Kronos
FORECAST_LEN = 120    # candles predicted (~2 hr on 1-min)
SIGNAL_HOUR  = 12     # UTC hour at which signal is computed
SIGNAL_MIN   = 30


# ──────────────────────────────────────────────
# REAL KRONOS (requires torch + HuggingFace)
# ──────────────────────────────────────────────

class KronosSignalGenerator:
    """
    Loads Kronos-mini from HuggingFace and generates directional forecasts.
    Call `generate_signals(df)` → dict {date: "bullish"|"bearish"}
    """

    def __init__(self, model_name="NeoQuasar/Kronos-mini",
                 tokenizer_name="NeoQuasar/Kronos-Tokenizer-2k"):
        print(f"[Kronos] Loading model: {model_name} ...")
        try:
            import sys, os
            if KRONOS_REPO not in sys.path:
                sys.path.insert(0, KRONOS_REPO)
            # Kronos repo must be on the path — clone it alongside this project:
            #   git clone https://github.com/shiyu-coder/Kronos
            #   export PYTHONPATH=./Kronos
            from model import Kronos, KronosTokenizer, KronosPredictor  # noqa

            self.tokenizer  = KronosTokenizer.from_pretrained(tokenizer_name)
            self.model      = Kronos.from_pretrained(model_name)
            self.predictor  = KronosPredictor(self.model, self.tokenizer,
                                              max_context=CONTEXT_LEN)
            self.available  = True
            print("[Kronos] Model loaded ✓")
        except Exception as e:
            print(f"[Kronos] Model unavailable ({e}). Using MockKronos.")
            self.available = False

    def _forecast_direction(self, context_df):
        """
        Feed `context_df` (OHLCV, len=CONTEXT_LEN) → 'bullish' or 'bearish'.
        """
        if not self.available:
            raise RuntimeError("Model not loaded")

        import pandas as pd
        x_ts = context_df.index
        # Build future timestamps (next FORECAST_LEN minutes)
        last_ts  = x_ts[-1]
        freq     = pd.tseries.frequencies.to_offset("1min")
        y_ts     = pd.date_range(start=last_ts + freq,
                                 periods=FORECAST_LEN, freq=freq, tz="UTC")

        pred = self.predictor.predict(
            df=context_df[["open","high","low","close","volume"]],
            x_timestamp=x_ts,
            y_timestamp=y_ts,
            pred_len=FORECAST_LEN,
            T=0.8, top_p=0.9, sample_count=1,   # 1 sample for CPU speed
        )

        current_close   = context_df["close"].iloc[-1]
        predicted_close = pred["close"].iloc[-1]
        return "bullish" if predicted_close > current_close else "bearish"

    def generate_signals(self, df):
        """
        Returns {date: 'bullish'|'bearish'} for every trading day in df.
        Signal is computed from data available at SIGNAL_HOUR:SIGNAL_MIN UTC.
        """
        signals = {}
        dates = pd.Series(df.index.date).unique()

        print(f"[Kronos] Generating {len(dates)} daily signals (CPU: ~3-5s each)...")
        for i, d in enumerate(dates):
            # Get the exact signal candle: 12:30 UTC on day d
            signal_ts = pd.Timestamp(year=d.year, month=d.month, day=d.day,
                                     hour=SIGNAL_HOUR, minute=SIGNAL_MIN,
                                     tz="UTC")
            # Find position of signal candle in df
            idx_arr = df.index.searchsorted(signal_ts)
            if idx_arr < CONTEXT_LEN or idx_arr >= len(df):
                continue

            context = df.iloc[idx_arr - CONTEXT_LEN : idx_arr]
            try:
                direction = self._forecast_direction(context)
                signals[d] = direction
                print(f"  [{i+1}/{len(dates)}] {d}: {direction}", end="\r")
            except Exception as e:
                print(f"[Kronos] Forecast failed for {d}: {e}")
        print()  # newline after progress

        bull = sum(1 for v in signals.values() if v == "bullish")
        print(f"[Kronos] {len(signals)} signals generated — "
              f"bullish: {bull}, bearish: {len(signals)-bull}")
        return signals


# ──────────────────────────────────────────────
# MOCK KRONOS (random — for pipeline testing)
# ──────────────────────────────────────────────

class MockKronos:
    """
    Generates pseudo-random daily directional signals seeded by price action.
    Not predictive — used only to verify the filter pipeline works end-to-end.
    Replace with KronosSignalGenerator() on a machine with GPU + Kronos cloned.
    """

    def __init__(self, seed=99):
        self.seed = seed

    def generate_signals(self, df):
        rng    = np.random.default_rng(self.seed)
        dates  = pd.Series(df.index.date).unique()
        dirs   = rng.choice(["bullish", "bearish"], size=len(dates))
        signals = {d: dirs[i] for i, d in enumerate(dates)}
        bull = sum(1 for v in signals.values() if v == "bullish")
        print(f"[MockKronos] {len(signals)} signals — "
              f"bullish: {bull}, bearish: {len(signals)-bull}")
        return signals


# ──────────────────────────────────────────────
# HELPER: auto-select real vs mock
# ──────────────────────────────────────────────

def get_kronos(use_real=True):
    """
    Returns KronosSignalGenerator if use_real=True and model loads OK,
    otherwise MockKronos.
    """
    if use_real:
        gen = KronosSignalGenerator()
        if gen.available:
            return gen
        print("[Kronos] Falling back to mock.")
    return MockKronos()


# ──────────────────────────────────────────────
# STANDALONE TEST
# ──────────────────────────────────────────────

if __name__ == "__main__":
    from synthetic_data import generate_synthetic_btc
    df = generate_synthetic_btc(days=30)
    print(f"Data: {len(df):,} candles")

    gen     = get_kronos(use_real=False)
    signals = gen.generate_signals(df)

    sample = dict(list(signals.items())[:5])
    print("\nSample signals:")
    for d, s in sample.items():
        print(f"  {d}: {s}")
