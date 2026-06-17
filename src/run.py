"""
Vitaliy Kronos Project — Main runner
Runs both strategies side-by-side and produces an HTML report:
  1. ICT only  (session sweep + FVG, no AI filter)
  2. ICT + Kronos  (same setups, filtered by Kronos directional bias)

Usage:
  python run.py                      # synthetic data, mock Kronos
  python run.py --live               # real Binance data
  python run.py --live --real-kronos # real Binance + real Kronos model
"""

import argparse
import json
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path

from backtester import run_backtest, compute_stats
from kronos import get_kronos

OUT = Path(".")


def get_data(live=False):
    if live:
        from backtester import fetch_binance_klines
        end_dt   = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        start_dt = end_dt - timedelta(days=60)
        print(f"Fetching live BTC/USDT 1m: {start_dt.date()} → {end_dt.date()}")
        return fetch_binance_klines("BTCUSDT", "1m", start_dt, end_dt)
    else:
        from synthetic_data import generate_synthetic_btc
        print("Generating synthetic BTC/USDT 1m (60 days)...")
        return generate_synthetic_btc(days=60)


def run(live=False, real_kronos=False):
    df = get_data(live)
    print(f"  {len(df):,} candles — ${df['close'].min():,.0f}–${df['close'].max():,.0f}\n")

    # ── Strategy 1: ICT only ──
    ict_trades = run_backtest(df, risk_pct=0.01, label="ICT only")
    ict_stats  = compute_stats(ict_trades, label="ICT only")

    # ── Kronos signals ──
    kronos_gen  = get_kronos(use_real=real_kronos)
    signals     = kronos_gen.generate_signals(df)

    # ── Strategy 2: ICT + Kronos filter ──
    ick_trades = run_backtest(df, risk_pct=0.01, label="ICT + Kronos",
                              kronos_signals=signals)
    ick_stats  = compute_stats(ick_trades, label="ICT + Kronos")

    # ── Print summary ──
    for stats in (ict_stats, ick_stats):
        print(f"\n── {stats['label']} ──")
        for k, v in stats.items():
            if k not in ("equity", "trades_df"):
                print(f"  {k}: {v}")

    # ── Save CSVs ──
    pd.DataFrame(ict_trades).to_csv(OUT / "ict_trades.csv", index=False)
    pd.DataFrame(ick_trades).to_csv(OUT / "ict_kronos_trades.csv", index=False)

    # ── Build HTML report ──
    html = build_report(ict_stats, ick_stats, df)
    report_path = OUT / "report.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"\nReport saved: {report_path.resolve()}")

    return ict_stats, ick_stats


# ──────────────────────────────────────────────
# HTML REPORT
# ──────────────────────────────────────────────

def build_report(s1, s2, df):
    # Equity curves as JSON for Chart.js
    eq1 = s1.get("equity", [100])
    eq2 = s2.get("equity", [100])
    labels1 = list(range(len(eq1)))
    labels2 = list(range(len(eq2)))

    # Trade log tables
    def trade_rows(stats):
        if stats["trades"] == 0:
            return "<tr><td colspan='7'>No trades</td></tr>"
        rows = []
        for _, t in stats["trades_df"].iterrows():
            c = "#22c55e" if t["outcome"] == "win" else ("#ef4444" if t["outcome"] == "loss" else "#f59e0b")
            rows.append(
                f"<tr>"
                f"<td>{str(t['entry_time'])[:16]}</td>"
                f"<td>{t['direction'].upper()}</td>"
                f"<td>{t['session'].upper()}</td>"
                f"<td style='color:{c};font-weight:600'>{t['outcome'].upper()}</td>"
                f"<td>{t['rr_achieved']:.2f}R</td>"
                f"<td style='color:{c}'>{t['pnl_pct']:+.2f}%</td>"
                f"<td>{t['entry']:,.0f}</td>"
                f"</tr>"
            )
        return "\n".join(rows)

    def stat_card(label, value, sub="", positive=None):
        if positive is True:   col = "#22c55e"
        elif positive is False: col = "#ef4444"
        else:                   col = "#e2e8f0"
        return f"""
        <div class="card">
          <div class="card-label">{label}</div>
          <div class="card-value" style="color:{col}">{value}</div>
          <div class="card-sub">{sub}</div>
        </div>"""

    def stats_row(s):
        wr   = s.get("win_rate", 0)
        pnl  = s.get("total_pnl_pct", 0)
        dd   = s.get("max_drawdown_pct", 0)
        ar   = s.get("avg_rr", 0)
        n    = s.get("trades", 0)
        return (
            stat_card("Win Rate",      f"{wr}%",       f"{n} trades",             positive=(wr >= 40)) +
            stat_card("Avg R:R",       f"{ar:.2f}R",   "avg per trade",           positive=(ar >= 1.5)) +
            stat_card("Total P&L",     f"{pnl:+.1f}%", "account % (no leverage)", positive=(pnl >= 0)) +
            stat_card("Max Drawdown",  f"{dd:.1f}%",   "peak-to-trough",          positive=(dd > -10))
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Vitaliy Kronos — ICT vs ICT+Kronos</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0f172a; color: #e2e8f0; font-family: 'Segoe UI', sans-serif; padding: 24px; }}
  h1 {{ font-size: 1.6rem; font-weight: 700; margin-bottom: 4px; color: #f1f5f9; }}
  .subtitle {{ color: #94a3b8; font-size: 0.85rem; margin-bottom: 28px; }}
  .section {{ margin-bottom: 36px; }}
  h2 {{ font-size: 1.05rem; font-weight: 600; color: #cbd5e1; margin-bottom: 14px;
        border-left: 3px solid #6366f1; padding-left: 10px; }}
  .cards {{ display: flex; gap: 14px; flex-wrap: wrap; }}
  .card {{ background: #1e293b; border-radius: 10px; padding: 18px 22px; min-width: 160px; flex: 1; }}
  .card-label {{ font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: .06em; }}
  .card-value {{ font-size: 1.6rem; font-weight: 700; margin: 6px 0 3px; }}
  .card-sub {{ font-size: 0.72rem; color: #64748b; }}
  .chart-wrap {{ background: #1e293b; border-radius: 10px; padding: 20px; }}
  .two-col {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.8rem; }}
  th {{ background: #1e293b; padding: 8px 12px; text-align: left; color: #94a3b8;
        font-weight: 600; text-transform: uppercase; font-size: 0.68rem; letter-spacing: .05em; }}
  td {{ padding: 7px 12px; border-bottom: 1px solid #1e293b; }}
  tr:hover td {{ background: #1e293b44; }}
  .table-wrap {{ background: #0f172a; border-radius: 10px; border: 1px solid #1e293b;
                 max-height: 340px; overflow-y: auto; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 99px; font-size: 0.7rem;
            font-weight: 600; }}
  .tag-ict {{ background:#6366f122; color:#818cf8; padding: 3px 10px; border-radius: 6px; font-size:.8rem; }}
  .tag-kr  {{ background:#10b98122; color:#34d399; padding: 3px 10px; border-radius: 6px; font-size:.8rem; }}
  .verdict {{ background: #1e293b; border-radius: 10px; padding: 20px 24px; line-height: 1.7;
              color: #cbd5e1; font-size: 0.9rem; }}
  .verdict strong {{ color: #f1f5f9; }}
</style>
</head>
<body>

<h1>🧠 Vitaliy Kronos — Strategy Comparison</h1>
<p class="subtitle">BTC/USDT 1-min · 60 days · Asia/London sweep + FVG · 5× leverage · 1% risk/trade</p>

<div class="section">
  <h2><span class="tag-ict">ICT Only</span> — Session sweep + FVG, no AI filter</h2>
  <div class="cards">{stats_row(s1)}</div>
</div>

<div class="section">
  <h2><span class="tag-kr">ICT + Kronos</span> — Same setups, Kronos directional filter applied</h2>
  <div class="cards">{stats_row(s2)}</div>
</div>

<div class="section">
  <h2>Equity Curves</h2>
  <div class="chart-wrap">
    <canvas id="eqChart" height="90"></canvas>
  </div>
</div>

<div class="section two-col">
  <div>
    <h2><span class="tag-ict">ICT Only</span> — Trade Log</h2>
    <div class="table-wrap">
      <table>
        <thead><tr>
          <th>Time (UTC)</th><th>Dir</th><th>Session</th>
          <th>Outcome</th><th>R:R</th><th>P&L %</th><th>Entry</th>
        </tr></thead>
        <tbody>{trade_rows(s1)}</tbody>
      </table>
    </div>
  </div>
  <div>
    <h2><span class="tag-kr">ICT + Kronos</span> — Trade Log</h2>
    <div class="table-wrap">
      <table>
        <thead><tr>
          <th>Time (UTC)</th><th>Dir</th><th>Session</th>
          <th>Outcome</th><th>R:R</th><th>P&L %</th><th>Entry</th>
        </tr></thead>
        <tbody>{trade_rows(s2)}</tbody>
      </table>
    </div>
  </div>
</div>

<div class="section">
  <h2>Verdict</h2>
  <div class="verdict">
    <strong>ICT only:</strong> {s1['trades']} trades · {s1['win_rate']}% WR · {s1['total_pnl_pct']:+.1f}% total P&L · {s1['max_drawdown_pct']:.1f}% max DD<br>
    <strong>ICT + Kronos:</strong> {s2['trades']} trades · {s2['win_rate']}% WR · {s2['total_pnl_pct']:+.1f}% total P&L · {s2['max_drawdown_pct']:.1f}% max DD<br><br>
    Trade reduction from Kronos filter: <strong>{s1['trades'] - s2['trades']} trades skipped
    ({round((s1['trades'] - s2['trades']) / max(s1['trades'],1) * 100)}%)</strong>.
    On synthetic (random walk) data the filter adds noise rather than signal — results become meaningful
    on real BTC data where Kronos was trained. Run with <code>--live --real-kronos</code> to compare apples-to-apples.
  </div>
</div>

<script>
const eq1 = {json.dumps(eq1)};
const eq2 = {json.dumps(eq2)};
const labels = Array.from({{length: Math.max(eq1.length, eq2.length)}}, (_, i) => i);

new Chart(document.getElementById('eqChart'), {{
  type: 'line',
  data: {{
    labels,
    datasets: [
      {{
        label: 'ICT only',
        data: eq1,
        borderColor: '#818cf8',
        backgroundColor: '#818cf810',
        borderWidth: 2,
        pointRadius: 0,
        fill: true,
        tension: 0.3,
      }},
      {{
        label: 'ICT + Kronos',
        data: eq2,
        borderColor: '#34d399',
        backgroundColor: '#34d39910',
        borderWidth: 2,
        pointRadius: 0,
        fill: true,
        tension: 0.3,
      }}
    ]
  }},
  options: {{
    responsive: true,
    plugins: {{
      legend: {{ labels: {{ color: '#94a3b8', font: {{ size: 12 }} }} }},
      tooltip: {{
        callbacks: {{
          label: ctx => ` ${{ctx.dataset.label}}: ${{ctx.parsed.y.toFixed(2)}}%`
        }}
      }}
    }},
    scales: {{
      x: {{ ticks: {{ color: '#475569', maxTicksLimit: 10 }}, grid: {{ color: '#1e293b' }} }},
      y: {{
        ticks: {{ color: '#475569', callback: v => v.toFixed(1) + '%' }},
        grid: {{ color: '#1e293b' }}
      }}
    }}
  }}
}});
</script>
</body>
</html>"""
    return html


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live",         action="store_true", help="Pull real Binance data")
    parser.add_argument("--real-kronos",  action="store_true", help="Use real Kronos model")
    args = parser.parse_args()
    run(live=args.live, real_kronos=args.real_kronos)
