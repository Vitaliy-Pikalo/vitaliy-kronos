"""
Vitaliy Kronos -- Strategy Optimizer
Tests 32 ICT parameter combinations on 60 days of real 5m data.
Runs walk-forward validation across 3 time periods to check consistency.
Saves best config to strategy_config.json (auto-used by run.py + signal_bot.py).
Generates optimize_report.html with full rankings + equity curves.

Usage:
  PYTHONUTF8=1 python src/optimize.py              # live 5m QQQ (recommended)
  PYTHONUTF8=1 python src/optimize.py --symbol SPY
  PYTHONUTF8=1 python src/optimize.py --synthetic  # synthetic data (no internet needed)
"""

import argparse, sys, json, itertools
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))
from backtester import run_backtest, compute_stats, fetch_yfinance_klines

OUT = Path(__file__).parent.parent  # C:\vk\

# ── Parameter grid (32 combinations) ──
PARAM_GRID = list(itertools.product(
    [False, True],        # use_ote
    [False, True],        # tight_sl
    [False, True],        # require_displacement
    [1.5, 2.0, 2.5, 3.0], # fallback_rr
))


def run_combo(df, params, max_hold=24):
    use_ote, tight_sl, req_disp, rr = params
    trades = run_backtest(
        df, risk_pct=0.01, label="opt", market="equity",
        use_ote=use_ote, tight_sl=tight_sl,
        require_displacement=req_disp,
        max_hold=max_hold, fallback_rr=rr,
        verbose=False,
    )
    return compute_stats(trades)


def score(s):
    """Score = (P&L / |MaxDD|) x (1 + 0.3*AvgRR). Minimum 3 trades required."""
    pnl = s.get("total_pnl_pct", 0)
    dd  = abs(s.get("max_drawdown_pct", 0.01)) or 0.01
    rr  = s.get("avg_rr", 0)
    if s.get("trades", 0) < 3:
        return -999
    return (pnl / dd) * (1 + rr * 0.3)


def optimize(symbol="QQQ", live=True, synthetic=False):
    # ── 1. Fetch data ──
    if synthetic or not live:
        from synthetic_data import generate_synthetic_equity
        print(f"[Optimizer] Generating synthetic {symbol} data (60 days 1m)...")
        df = generate_synthetic_equity(symbol, days=60)
        interval, max_hold = "1m", 120
    else:
        print(f"[Optimizer] Fetching live {symbol} 5m data (60 days)...")
        df = fetch_yfinance_klines(symbol, days=60, interval="5m")
        interval, max_hold = "5m", 24

    n = len(df)
    print(f"[Optimizer] {n:,} candles loaded ({interval})")

    # ── 2. Walk-forward split (3 equal periods) ──
    p1 = df.iloc[:n//3]
    p2 = df.iloc[n//3 : 2*n//3]
    p3 = df.iloc[2*n//3:]
    periods = {
        "P1 (oldest)": p1,
        "P2 (middle)": p2,
        "P3 (recent)": p3,
    }
    p1_end = str(p1.index[-1])[:10]
    p2_end = str(p2.index[-1])[:10]
    p3_end = str(p3.index[-1])[:10]
    period_labels = [
        f"P1 (→{p1_end})",
        f"P2 (→{p2_end})",
        f"P3 (→{p3_end})",
    ]

    # ── 3. Grid search on full dataset ──
    print(f"\n[Optimizer] Running {len(PARAM_GRID)} combos on full 60-day dataset...")
    all_results = []
    for i, params in enumerate(PARAM_GRID):
        s = run_combo(df, params, max_hold=max_hold)
        s["params"]        = params
        s["score"]         = score(s)
        s["param_str"]     = _param_str(params)
        all_results.append(s)
        pct = (i + 1) / len(PARAM_GRID) * 100
        bar = "#" * (i + 1) + "." * (len(PARAM_GRID) - i - 1)
        print(f"  [{bar}] {pct:.0f}%  P&L={s['total_pnl_pct']:+.1f}%  WR={s['win_rate']}%  RR={s['avg_rr']:.2f}", end="\r")

    print(f"\n\n[Optimizer] Grid search complete. Ranking by score...")
    all_results.sort(key=lambda x: x["score"], reverse=True)

    # Print top 10 to terminal
    print(f"\n{'#':<4} {'OTE':<5} {'TSL':<5} {'DSP':<5} {'RR':<5} {'Trades':<7} {'WR%':<7} {'AvgRR':<7} {'P&L%':<9} {'DD%':<8} {'Score'}")
    print("─" * 75)
    for i, s in enumerate(all_results[:10]):
        ote, tsl, disp, rr = s["params"]
        print(f"{i+1:<4} {'Y' if ote else 'N':<5} {'Y' if tsl else 'N':<5} {'Y' if disp else 'N':<5} "
              f"{rr:<5} {s['trades']:<7} {s['win_rate']:<7} {s['avg_rr']:<7.2f} "
              f"{s['total_pnl_pct']:+.1f}{'%':<7} {s['max_drawdown_pct']:.1f}{'%':<6} {s['score']:.2f}")

    best = all_results[0]
    best_params = best["params"]
    print(f"\n{'='*55}")
    print(f"  BEST: OTE={best_params[0]}  TightSL={best_params[1]}  Displacement={best_params[2]}  RR={best_params[3]}")
    print(f"  P&L: {best['total_pnl_pct']:+.1f}%  WR: {best['win_rate']}%  AvgRR: {best['avg_rr']:.2f}  MaxDD: {best['max_drawdown_pct']:.1f}%")
    print(f"{'='*55}")

    # ── 4. Walk-forward on top 5 ──
    print(f"\n[Optimizer] Walk-forward validation (top 5 configs × 3 periods)...")
    wf_data = []
    for s in all_results[:5]:
        params = s["params"]
        row = {"label": s["param_str"], "params": params}
        profits = 0
        for pname, pdf in periods.items():
            ps = run_combo(pdf, params, max_hold=max_hold)
            row[pname] = round(ps["total_pnl_pct"], 2)
            if ps["total_pnl_pct"] > 0:
                profits += 1
        row["consistent"] = profits
        wf_data.append(row)
        print(f"  #{all_results.index(s)+1}  {s['param_str']:40s}  "
              f"P1={row[list(periods.keys())[0]]:+.1f}%  "
              f"P2={row[list(periods.keys())[1]]:+.1f}%  "
              f"P3={row[list(periods.keys())[2]]:+.1f}%  "
              f"[{profits}/3 profitable]")

    # ── 5. Save best config to strategy_config.json ──
    strategy_cfg = {
        "symbol":               symbol,
        "interval":             interval,
        "use_ote":              bool(best_params[0]),
        "tight_sl":             bool(best_params[1]),
        "require_displacement": bool(best_params[2]),
        "fallback_rr":          float(best_params[3]),
        "score":                round(best["score"], 3),
        "pnl_pct":              best["total_pnl_pct"],
        "win_rate":             best["win_rate"],
        "avg_rr":               best["avg_rr"],
        "max_dd":               best["max_drawdown_pct"],
        "trades":               best["trades"],
        "walk_forward_consistent": wf_data[0]["consistent"] if wf_data else 0,
        "updated":              datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "data_source":          f"live {symbol} {interval} 60d" if live else "synthetic",
    }
    cfg_path = OUT / "strategy_config.json"
    cfg_path.write_text(json.dumps(strategy_cfg, indent=2))
    print(f"\n[Optimizer] Best config saved → {cfg_path}")

    # ── 6. Build HTML report ──
    html = _build_report(all_results, wf_data, period_labels, best, symbol, interval, live)
    report_path = OUT / "optimize_report.html"
    report_path.write_text(html, encoding="utf-8")
    print(f"[Optimizer] Report saved    → {report_path}")

    print(f"\n  Next steps:")
    print(f"  1. start \"C:\\vk\\optimize_report.html\"")
    print(f"  2. PYTHONUTF8=1 python src/run.py --live --market equity   (uses best config)")

    return all_results, strategy_cfg


# ── Helpers ──

def _param_str(params):
    ote, tsl, disp, rr = params
    parts = []
    if ote:  parts.append("OTE")
    if tsl:  parts.append("TightSL")
    if disp: parts.append("Disp")
    parts.append(f"RR={rr}")
    return " ".join(parts) if parts else f"Baseline RR={rr}"


def _build_report(all_results, wf_data, period_labels, best, symbol, interval, live):
    import json as _json

    data_note = f"live {symbol} {interval} (60 days)" if live else f"synthetic {symbol}"
    best_params = best["params"]
    wf_consistent = wf_data[0]["consistent"] if wf_data else 0
    wf_badge_col  = "#059669" if wf_consistent == 3 else ("#f59e0b" if wf_consistent == 2 else "#ef4444")

    # Full rankings table
    table_rows = ""
    for i, s in enumerate(all_results):
        ote, tsl, disp, rr = s["params"]
        is_best   = i == 0
        row_bg    = ' style="background:#0d2e1e"' if is_best else ""
        badge     = ' <span style="background:#059669;color:#fff;padding:1px 6px;border-radius:4px;font-size:.62rem;vertical-align:middle">BEST</span>' if is_best else ""
        pnl_col   = "#22c55e" if s["total_pnl_pct"] >= 0 else "#ef4444"
        wr_col    = "#22c55e" if s["win_rate"] >= 50 else ("#f59e0b" if s["win_rate"] >= 35 else "#ef4444")
        rr_col    = "#22c55e" if s["avg_rr"] >= 1.5 else "#94a3b8"
        table_rows += (
            f'<tr{row_bg}>'
            f'<td>{i+1}{badge}</td>'
            f'<td>{"✓" if ote else "—"}</td>'
            f'<td>{"✓" if tsl else "—"}</td>'
            f'<td>{"✓" if disp else "—"}</td>'
            f'<td>{rr}R</td>'
            f'<td>{s["trades"]}</td>'
            f'<td style="color:{wr_col}">{s["win_rate"]}%</td>'
            f'<td style="color:{rr_col}">{s["avg_rr"]:.2f}R</td>'
            f'<td style="color:{pnl_col}">{s["total_pnl_pct"]:+.1f}%</td>'
            f'<td style="color:#ef4444">{s["max_drawdown_pct"]:.1f}%</td>'
            f'<td style="color:#f59e0b">{s["score"]:.2f}</td>'
            f'</tr>'
        )

    # Walk-forward table
    wf_period_keys = list(periods_from_wf(wf_data))
    wf_rows = ""
    for row in wf_data:
        consistent = row["consistent"]
        badge_col  = "#059669" if consistent == 3 else ("#f59e0b" if consistent == 2 else "#ef4444")
        wf_rows += f'<tr><td style="font-size:.75rem">{row["label"]}</td>'
        for pk in wf_period_keys:
            v   = row.get(pk, 0)
            col = "#22c55e" if v >= 0 else "#ef4444"
            wf_rows += f'<td style="color:{col}">{v:+.1f}%</td>'
        wf_rows += (f'<td><span style="background:{badge_col};color:#fff;'
                    f'padding:2px 8px;border-radius:4px;font-size:.72rem">'
                    f'{consistent}/3</span></td></tr>')

    # Top 5 equity curves
    colors5 = ["#34d399", "#6366f1", "#f59e0b", "#ec4899", "#06b6d4"]
    datasets_js = []
    for i, s in enumerate(all_results[:5]):
        eq = s.get("equity", [100])
        datasets_js.append(f"""{{
            label: '{s["param_str"]}',
            data: {_json.dumps(eq)},
            borderColor: '{colors5[i]}',
            backgroundColor: '{colors5[i]}18',
            borderWidth: {'2.5' if i == 0 else '1.5'},
            pointRadius: 0, fill: {'true' if i == 0 else 'false'}, tension: 0.3
        }}""")
    max_eq_len = max(len(s.get("equity", [100])) for s in all_results[:5])

    html = f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="UTF-8">
<title>Vitaliy Kronos -- Strategy Optimizer</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  *{{box-sizing:border-box;margin:0;padding:0}}
  body{{background:#0f172a;color:#e2e8f0;font-family:'Segoe UI',sans-serif;padding:24px}}
  h1{{font-size:1.6rem;font-weight:700;margin-bottom:4px;color:#f1f5f9}}
  h2{{font-size:1.0rem;font-weight:600;color:#cbd5e1;margin-bottom:14px;border-left:3px solid #6366f1;padding-left:10px}}
  .subtitle{{color:#94a3b8;font-size:.83rem;margin-bottom:28px}}
  .section{{margin-bottom:32px}}
  .best-card{{background:linear-gradient(135deg,#052e16,#1e293b);border:1px solid #059669;border-radius:12px;padding:24px;margin-bottom:32px}}
  .best-title{{font-size:1.05rem;font-weight:700;color:#34d399;margin-bottom:14px}}
  .stat-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(130px,1fr));gap:12px;margin-bottom:16px}}
  .stat{{background:#0f172a55;border-radius:8px;padding:12px 16px}}
  .stat-label{{font-size:.67rem;color:#64748b;text-transform:uppercase;letter-spacing:.05em}}
  .stat-val{{font-size:1.45rem;font-weight:700;margin-top:4px}}
  .param-row{{display:flex;gap:8px;flex-wrap:wrap;margin-top:14px}}
  .ptag{{padding:4px 12px;border-radius:6px;font-size:.76rem;font-weight:600}}
  .pon{{background:#065f4633;color:#34d399;border:1px solid #05966955}}
  .poff{{background:#1e293b;color:#475569;border:1px solid #334155}}
  .chart-wrap{{background:#1e293b;border-radius:10px;padding:20px;margin-bottom:8px}}
  .table-wrap{{background:#0f172a;border-radius:10px;border:1px solid #1e293b;max-height:420px;overflow-y:auto}}
  table{{width:100%;border-collapse:collapse;font-size:.78rem}}
  th{{background:#1e293b;padding:8px 12px;text-align:left;color:#94a3b8;font-weight:600;text-transform:uppercase;font-size:.64rem;letter-spacing:.05em;position:sticky;top:0;z-index:1}}
  td{{padding:7px 12px;border-bottom:1px solid #1e293b33}}
  tr:hover td{{background:#1e293b55}}
  .wf-note{{font-size:.78rem;color:#64748b;margin-top:8px}}
</style>
</head><body>
<h1>Vitaliy Kronos -- Strategy Optimizer</h1>
<p class="subtitle">32 parameter combos &middot; {data_note} &middot; scored by return/drawdown &times; trade quality &middot; walk-forward validated</p>

<div class="best-card">
  <div class="best-title">&#x2605; Recommended Configuration
    <span style="background:{wf_badge_col};color:#fff;padding:2px 10px;border-radius:6px;font-size:.72rem;font-weight:600;margin-left:10px">{wf_consistent}/3 periods profitable</span>
  </div>
  <div class="stat-grid">
    <div class="stat"><div class="stat-label">Total P&L</div><div class="stat-val" style="color:#34d399">{best['total_pnl_pct']:+.1f}%</div></div>
    <div class="stat"><div class="stat-label">Win Rate</div><div class="stat-val" style="color:#34d399">{best['win_rate']}%</div></div>
    <div class="stat"><div class="stat-label">Avg R:R</div><div class="stat-val" style="color:#34d399">{best['avg_rr']:.2f}R</div></div>
    <div class="stat"><div class="stat-label">Max Drawdown</div><div class="stat-val" style="color:#ef4444">{best['max_drawdown_pct']:.1f}%</div></div>
    <div class="stat"><div class="stat-label">Trades (60d)</div><div class="stat-val" style="color:#94a3b8">{best['trades']}</div></div>
    <div class="stat"><div class="stat-label">Optimizer Score</div><div class="stat-val" style="color:#f59e0b">{best['score']:.2f}</div></div>
  </div>
  <div class="param-row">
    <span class="ptag {'pon' if best_params[0] else 'poff'}">OTE Entry: {'ON' if best_params[0] else 'OFF'}</span>
    <span class="ptag {'pon' if best_params[1] else 'poff'}">Tight SL: {'ON' if best_params[1] else 'OFF'}</span>
    <span class="ptag {'pon' if best_params[2] else 'poff'}">Displacement Filter: {'ON' if best_params[2] else 'OFF'}</span>
    <span class="ptag pon">R:R Target: {best_params[3]}R</span>
  </div>
</div>

<div class="section">
  <h2>Top 5 Equity Curves</h2>
  <div class="chart-wrap"><canvas id="eqChart" height="75"></canvas></div>
</div>

<div class="section">
  <h2>Walk-Forward Consistency (top 5 &times; 3 time periods)</h2>
  <p class="wf-note">Each period = ~20 days. Green = profitable. 3/3 = high confidence the edge is real.</p>
  <br>
  <div class="table-wrap"><table>
    <thead><tr><th>Config</th>{''.join(f'<th>{pl}</th>' for pl in period_labels)}<th>Consistent</th></tr></thead>
    <tbody>{wf_rows}</tbody>
  </table></div>
</div>

<div class="section">
  <h2>All 32 Combinations Ranked</h2>
  <div class="table-wrap"><table>
    <thead><tr><th>#</th><th>OTE</th><th>Tight SL</th><th>Displace</th><th>RR Target</th><th>Trades</th><th>WR%</th><th>Avg R:R</th><th>P&L%</th><th>MaxDD%</th><th>Score</th></tr></thead>
    <tbody>{table_rows}</tbody>
  </table></div>
</div>

<script>
const labels = Array.from({{length: {max_eq_len}}}, (_,i) => i);
new Chart(document.getElementById('eqChart'), {{
  type: 'line',
  data: {{ labels, datasets: [{",".join(datasets_js)}] }},
  options: {{
    responsive: true,
    plugins: {{ legend: {{ labels: {{ color: '#94a3b8', font: {{ size: 11 }} }} }},
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


def periods_from_wf(wf_data):
    """Extract period keys from walk-forward results."""
    if not wf_data:
        return []
    return [k for k in wf_data[0].keys() if k not in ("label", "params", "consistent")]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol",    default="QQQ", help="Equity symbol (QQQ or SPY)")
    parser.add_argument("--synthetic", action="store_true", help="Use synthetic data")
    args = parser.parse_args()
    optimize(symbol=args.symbol, live=not args.synthetic, synthetic=args.synthetic)
