"""
Vitaliy Kronos Project -- Main runner
Compares ICT strategy across markets and AI filter.

Usage:
  python src/run.py                          # synthetic BTC + QQQ, mock Kronos
  python src/run.py --market btc             # BTC only
  python src/run.py --market equity          # QQQ only
  python src/run.py --live                   # real data (Binance.US + yfinance)
  python src/run.py --live --real-kronos     # real data + real Kronos model
  python src/run.py --symbol SPY             # use SPY instead of QQQ
"""

import argparse
import json
import sys
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path

from backtester import run_backtest, compute_stats, fetch_binance_klines, fetch_yfinance_klines
from kronos import get_kronos

OUT = Path(__file__).parent.parent  # C:\vk\


def get_btc_data(live=False):
    if live:
        end_dt   = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        start_dt = end_dt - timedelta(days=60)
        print(f"Fetching live BTC/USDT 1m: {start_dt.date()} -> {end_dt.date()}")
        return fetch_binance_klines("BTCUSDT", "1m", start_dt, end_dt)
    else:
        from synthetic_data import generate_synthetic_btc
        print("Generating synthetic BTC/USDT 1m (60 days)...")
        return generate_synthetic_btc(days=60)


def get_equity_data(symbol="QQQ", live=False):
    if live:
        print(f"Fetching live {symbol} 1m (7 days via yfinance)...")
        return fetch_yfinance_klines(symbol, days=7, interval="1m")
    else:
        from synthetic_data import generate_synthetic_equity
        print(f"Generating synthetic {symbol} 1m (60 days)...")
        return generate_synthetic_equity(symbol, days=60)


def run_market(df, market_key, market_label, real_kronos=False):
    """Run ICT-only and ICT+Kronos for one market. Returns (ict_stats, kronos_stats)."""
    ict_trades   = run_backtest(df, risk_pct=0.01, label=f"ICT {market_label}",
                                market=market_key)
    ict_stats    = compute_stats(ict_trades, label=f"ICT {market_label}")

    kronos_gen   = get_kronos(use_real=real_kronos)
    signals      = kronos_gen.generate_signals(df)
    ick_trades   = run_backtest(df, risk_pct=0.01, label=f"ICT+Kronos {market_label}",
                                market=market_key, kronos_signals=signals)
    ick_stats    = compute_stats(ick_trades, label=f"ICT+Kronos {market_label}")

    for s in (ict_stats, ick_stats):
        print(f"\n-- {s['label']} --")
        for k, v in s.items():
            if k not in ("equity", "trades_df"):
                print(f"  {k}: {v}")

    return ict_stats, ick_stats


def run(live=False, real_kronos=False, market="all", symbol="QQQ"):
    results = {}

    if market in ("all", "btc"):
        df_btc = get_btc_data(live)
        print(f"  {len(df_btc):,} BTC candles -- ${df_btc['close'].min():,.0f}-${df_btc['close'].max():,.0f}")
        results["btc_ict"], results["btc_kronos"] = run_market(df_btc, "btc", "BTC", real_kronos)
        pd.DataFrame(results["btc_ict"].get("trades_df", [])).to_csv(OUT / "btc_trades.csv", index=False)

    if market in ("all", "equity"):
        df_eq = get_equity_data(symbol, live)
        print(f"  {len(df_eq):,} {symbol} candles -- ${df_eq['close'].min():,.2f}-${df_eq['close'].max():,.2f}")
        results["eq_ict"], results["eq_kronos"] = run_market(df_eq, "equity", symbol, real_kronos)
        pd.DataFrame(results["eq_ict"].get("trades_df", [])).to_csv(OUT / f"{symbol.lower()}_trades.csv", index=False)

    if len(results) > 0:
        html = build_report(results, symbol=symbol, live=live)
        report_path = OUT / "report.html"
        report_path.write_text(html, encoding="utf-8")
        print(f"\nReport saved: {report_path.resolve()}")

    return results


# ── HTML report ──

def build_report(results, symbol="QQQ", live=False):
    import json as _json

    data_note = "live Binance.US + yfinance" if live else "synthetic data"

    def stat_card(label, value, sub="", positive=None):
        col = "#22c55e" if positive is True else ("#ef4444" if positive is False else "#e2e8f0")
        return f'<div class="card"><div class="card-label">{label}</div><div class="card-value" style="color:{col}">{value}</div><div class="card-sub">{sub}</div></div>'

    def stats_row(s):
        wr, pnl, dd, ar, n = s.get("win_rate",0), s.get("total_pnl_pct",0), s.get("max_drawdown_pct",0), s.get("avg_rr",0), s.get("trades",0)
        return (stat_card("Win Rate", f"{wr}%", f"{n} trades", positive=(wr>=40)) +
                stat_card("Avg R:R", f"{ar:.2f}R", "avg per trade", positive=(ar>=1.5)) +
                stat_card("Total P&L", f"{pnl:+.1f}%", "account % (1% risk/trade)", positive=(pnl>=0)) +
                stat_card("Max Drawdown", f"{dd:.1f}%", "peak-to-trough", positive=(dd>-10)))

    def trade_rows(s):
        if s.get("trades", 0) == 0 or "trades_df" not in s:
            return "<tr><td colspan='7'>No trades</td></tr>"
        rows = []
        for _, t in s["trades_df"].iterrows():
            c = "#22c55e" if t["outcome"]=="win" else ("#ef4444" if t["outcome"]=="loss" else "#f59e0b")
            rows.append(f"<tr><td>{str(t['entry_time'])[:16]}</td><td>{t['direction'].upper()}</td>"
                        f"<td>{t['session'].upper()}</td><td style='color:{c};font-weight:600'>{t['outcome'].upper()}</td>"
                        f"<td>{t['rr_achieved']:.2f}R</td><td style='color:{c}'>{t['pnl_pct']:+.2f}%</td>"
                        f"<td>{t['entry']:,.2f}</td></tr>")
        return "\n".join(rows)

    # Build dataset list for Chart.js
    colors = {"btc_ict": "#818cf8", "btc_kronos": "#6366f1",
              "eq_ict": "#34d399",  "eq_kronos": "#059669"}
    names  = {"btc_ict": "BTC ICT", "btc_kronos": "BTC+Kronos",
              "eq_ict":  f"{symbol} ICT", "eq_kronos": f"{symbol}+Kronos"}

    datasets_js = []
    for key, s in results.items():
        eq = s.get("equity", [100])
        datasets_js.append(f"""{{
            label: '{names.get(key, key)}',
            data: {_json.dumps(eq)},
            borderColor: '{colors.get(key, "#fff")}',
            backgroundColor: '{colors.get(key, "#fff")}18',
            borderWidth: 2, pointRadius: 0, fill: true, tension: 0.3
        }}""")

    # Sections
    sections = ""
    if "btc_ict" in results:
        sections += f"""
        <div class="section">
          <h2><span class="tag-btc">BTC/USDT</span> ICT Only</h2>
          <div class="cards">{stats_row(results['btc_ict'])}</div>
        </div>
        <div class="section">
          <h2><span class="tag-btc">BTC/USDT</span> ICT + Kronos</h2>
          <div class="cards">{stats_row(results['btc_kronos'])}</div>
        </div>"""
    if "eq_ict" in results:
        sections += f"""
        <div class="section">
          <h2><span class="tag-eq">{symbol}</span> ICT Only</h2>
          <div class="cards">{stats_row(results['eq_ict'])}</div>
        </div>
        <div class="section">
          <h2><span class="tag-eq">{symbol}</span> ICT + Kronos</h2>
          <div class="cards">{stats_row(results['eq_kronos'])}</div>
        </div>"""

    # Trade log tabs
    trade_logs = ""
    for key, label in names.items():
        if key not in results: continue
        trade_logs += f"""
        <div>
          <h2>{label} -- Trade Log</h2>
          <div class="table-wrap"><table>
            <thead><tr><th>Time</th><th>Dir</th><th>Session</th><th>Outcome</th><th>R:R</th><th>P&L%</th><th>Entry</th></tr></thead>
            <tbody>{trade_rows(results[key])}</tbody>
          </table></div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<title>Vitaliy Kronos -- Market Comparison</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0f172a;color:#e2e8f0;font-family:'Segoe UI',sans-serif;padding:24px}}
  h1{{font-size:1.6rem;font-weight:700;margin-bottom:4px;color:#f1f5f9}}
  .subtitle{{color:#94a3b8;font-size:.85rem;margin-bottom:28px}}
  .section{{margin-bottom:30px}}
  h2{{font-size:1.05rem;font-weight:600;color:#cbd5e1;margin-bottom:14px;border-left:3px solid #6366f1;padding-left:10px}}
  .cards{{display:flex;gap:14px;flex-wrap:wrap}}
  .card{{background:#1e293b;border-radius:10px;padding:18px 22px;min-width:150px;flex:1}}
  .card-label{{font-size:.72rem;color:#64748b;text-transform:uppercase;letter-spacing:.06em}}
  .card-value{{font-size:1.5rem;font-weight:700;margin:6px 0 3px}}
  .card-sub{{font-size:.72rem;color:#64748b}}
  .chart-wrap{{background:#1e293b;border-radius:10px;padding:20px}}
  .grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}
  table{{width:100%;border-collapse:collapse;font-size:.8rem}}
  th{{background:#1e293b;padding:8px 12px;text-align:left;color:#94a3b8;font-weight:600;text-transform:uppercase;font-size:.68rem;letter-spacing:.05em}}
  td{{padding:7px 12px;border-bottom:1px solid #1e293b}}
  tr:hover td{{background:#1e293b44}}
  .table-wrap{{background:#0f172a;border-radius:10px;border:1px solid #1e293b;max-height:300px;overflow-y:auto}}
  .tag-btc{{background:#6366f122;color:#818cf8;padding:3px 10px;border-radius:6px;font-size:.8rem}}
  .tag-eq{{background:#10b98122;color:#34d399;padding:3px 10px;border-radius:6px;font-size:.8rem}}
</style>
</head><body>
<h1>Vitaliy Kronos -- BTC vs {symbol} Comparison</h1>
<p class="subtitle">ICT session strategy · 1-min candles · 1% risk/trade · 120-min cooldown · {data_note}</p>

{sections}

<div class="section">
  <h2>Equity Curves (all strategies)</h2>
  <div class="chart-wrap"><canvas id="eqChart" height="80"></canvas></div>
</div>

<div class="section grid">{trade_logs}</div>

<script>
const maxLen = Math.max({",".join(str(len(s.get("equity",[100]))) for s in results.values())});
const labels = Array.from({{length: maxLen}}, (_,i) => i);
new Chart(document.getElementById('eqChart'), {{
  type: 'line',
  data: {{ labels, datasets: [{",".join(datasets_js)}] }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ labels: {{ color: '#94a3b8', font: {{ size: 12 }} }} }},
               tooltip: {{ callbacks: {{ label: ctx => ` ${{ctx.dataset.label}}: ${{ctx.parsed.y.toFixed(2)}}%` }} }} }},
    scales: {{
      x: {{ ticks: {{ color: '#475569', maxTicksLimit: 10 }}, grid: {{ color: '#1e293b' }} }},
      y: {{ ticks: {{ color: '#475569', callback: v => v.toFixed(1)+'%' }}, grid: {{ color: '#1e293b' }} }}
    }}
  }}
}});
</script>
</body></html>"""
    return html


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--live",        action="store_true", help="Real Binance + yfinance data")
    parser.add_argument("--real-kronos", action="store_true", help="Use real Kronos model")
    parser.add_argument("--market",      default="all", choices=["all","btc","equity"])
    parser.add_argument("--symbol",      default="QQQ", help="Equity symbol: QQQ or SPY")
    args = parser.parse_args()
    run(live=args.live, real_kronos=args.real_kronos, market=args.market, symbol=args.symbol)
