"""build_dashboard.py — FORTRESS cloud dashboard builder.

Runs in GitHub Actions once each weekday (~2h after the US close):
  1. loads paper_state.json (core holdings, NAV, benchmark levels as of last_mark),
  2. fetches adjusted daily closes via yfinance for every held name + GLD + benchmarks,
  3. computes the core sleeve's daily return (buy-and-hold intra-sleeve drift, the same
     marking math as the Deen Capital cloud tracker), appends it to core_returns.csv,
  4. recomputes the FROZEN clamp overlay on the full core-return series
     (sigma*=0.20, D*=0.12 — the parameters locked in the audited backtest; nothing is
     ever refit here), derives the executed equity fraction with the 5pp drift band and
     the t+2 execution delay, and marks the portfolio NAV:
         r_p = f * r_core + 0.5*(1-f) * r_GLD          (cash earns 0%, halal)
  5. regenerates a fully self-contained index.html for GitHub Pages.

--offline : skip the price fetch, just rebuild index.html from existing data.

The paper NAV is SIMULATED (closing prices, not broker fills), gross of costs, and the
strategy's 2010-2025 figures are hypothetical backtest results. See the disclosures
section of the page.
"""
import csv
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
STATE = os.path.join(DATA, "paper_state.json")
TRACK = os.path.join(DATA, "paper_track.csv")
CORE_RET = os.path.join(DATA, "core_returns.csv")
BOOK = os.path.join(DATA, "target_book.json")
STATS = os.path.join(DATA, "tear_sheet_stats.json")
OUT = os.path.join(HERE, "index.html")

BENCH = ["SPY", "QQQ", "SPUS", "HLAL"]
TRACK_COLS = ["date", "nav", "SPY", "QQQ", "SPUS", "HLAL",
              "f_applied", "w_eq", "w_gld", "w_cash"]

# ---- FROZEN strategy parameters (F1C-RISK2, params.json of the audited backtest).
# Changing ANY value here is a strategy change and voids the live record's comparability.
P = dict(sigma_star=0.20, d_star=0.12,
         vol_fast=20, vol_slow=60, dd_lookback=252, dd_minp=60,
         f_min=0.25, f_max=1.00, brake_floor_mult=2.0,
         gld_share=0.50, drift_band=0.05, exec_shift=2)

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def long_date(iso):
    y, m, d = iso.split("-")
    return f"{MONTHS[int(m)-1]} {int(d)}, {y}"


# ---------------- the clamp (faithful port of F1C-RISK2 clamp_signal) ----------------
def clamp_path(ret):
    """ret: pandas Series of core daily simple returns (DatetimeIndex, ascending).
    Returns DataFrame with f_raw, f_held, f_applied — identical math to the audited
    strategy: dual-window vol target, 252d drawdown brake, min() composition,
    5pp drift-band execution, t+2 application."""
    import numpy as np
    import pandas as pd
    rt = math.sqrt(252.0)
    s_fast = ret.rolling(P["vol_fast"]).std() * rt
    s_slow = ret.rolling(P["vol_slow"]).std() * rt
    sigma = pd.concat([s_fast, s_slow], axis=1).max(axis=1)
    f_vol = np.minimum(1.0, P["sigma_star"] / sigma)

    eq = (1.0 + ret).cumprod()
    peak = eq.rolling(P["dd_lookback"], min_periods=P["dd_minp"]).max()
    dd = eq / peak - 1.0
    depth = (-dd).clip(lower=0.0)
    ramp = 1.0 - (depth - P["d_star"]) / (P["d_star"] * (P["brake_floor_mult"] - 1.0))
    f_dd = np.where(dd >= -P["d_star"], 1.0, np.clip(ramp, P["f_min"], 1.0))
    f_dd = pd.Series(f_dd, index=ret.index)

    f_raw = pd.concat([f_vol, f_dd], axis=1).min(axis=1).clip(P["f_min"], P["f_max"])

    held = np.full(len(f_raw), np.nan)
    cur = np.nan
    for i, v in enumerate(f_raw.values):
        if np.isnan(v):
            held[i] = np.nan
            continue
        if np.isnan(cur) or abs(v - cur) > P["drift_band"]:
            cur = v
        held[i] = cur
    out = pd.DataFrame({"f_raw": f_raw, "f_held": held}, index=ret.index)
    out["f_applied"] = out["f_held"].shift(P["exec_shift"])
    return out


# ---------------- marking ----------------
def _atomic(write_fn, path):
    tmp = path + ".tmp"
    write_fn(tmp)
    os.replace(tmp, path)


def fetch_and_mark():
    import numpy as np
    import pandas as pd
    import yfinance as yf

    state = json.load(open(STATE))
    core_w = state["core_weights"]
    tickers = sorted(set(core_w) | set(BENCH) | {"GLD"})

    start = min(state["last_mark"], _core_last_date())
    print(f"[build] fetching {len(tickers)} tickers from {start} ...")
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False,
                      threads=True, group_by="column")
    if raw is None or len(raw) == 0:
        print("[build] WARNING: empty price frame — skipping mark")
        return 0
    close = None
    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw.xs("Close", axis=1, level=0)
        elif "Close" in raw.columns.get_level_values(-1):
            close = raw.xs("Close", axis=1, level=-1)
    if close is None:
        close = raw
    if isinstance(close, pd.Series):
        close = close.to_frame()
    close = close.reindex(columns=tickers)
    close.index = pd.to_datetime(close.index)
    close = close.sort_index()
    rets = close.pct_change()

    missing = [t for t in tickers if t not in close.columns or close[t].notna().sum() == 0]
    if missing:
        print(f"[build] note: no data for {missing} (treated flat)")

    # ---- 1) extend the core return series (buy-and-hold intra-sleeve drift) ----
    core = pd.read_csv(CORE_RET)
    core["date"] = pd.to_datetime(core["date"])
    core_last = core["date"].max()
    new_core_days = [d for d in rets.index if d > core_last]
    w = dict(core_w)
    added = []
    for d in new_core_days:
        r = rets.loc[d]
        sret, nw, tot = 0.0, {}, 0.0
        for t, wt in w.items():
            ri = r.get(t, np.nan) if t in rets.columns else np.nan
            ri = 0.0 if not np.isfinite(ri) else float(ri)
            sret += wt * ri
            nv = wt * (1.0 + ri)
            nw[t] = nv
            tot += nv
        w = {t: v / tot for t, v in nw.items()} if tot > 0 else w
        added.append({"date": d, "ret": sret, "source": "live"})
    state["core_weights"] = w
    if added:
        core = pd.concat([core, pd.DataFrame(added)], ignore_index=True)
        core = core.drop_duplicates("date", keep="first").sort_values("date")
        _atomic(lambda p: core.assign(date=core["date"].dt.date).to_csv(p, index=False),
                CORE_RET)

    # ---- 2) recompute the frozen clamp on the full series ----
    rser = core.set_index("date")["ret"].astype(float)
    fpath = clamp_path(rser)

    # ---- 3) mark NAV forward ----
    last = pd.Timestamp(state["last_mark"])
    days = [d for d in rets.index if d > last and d in fpath.index]
    rows = []
    for d in days:
        f = fpath.loc[d, "f_applied"]
        if not np.isfinite(f):
            print(f"[build] skip {d.date()} — f not defined yet")
            continue
        r_core = float(rser.loc[d])
        rg = rets.loc[d].get("GLD", np.nan)
        r_gld = 0.0 if not np.isfinite(rg) else float(rg)
        w_eq = float(f)
        w_gld = P["gld_share"] * (1.0 - w_eq)
        w_cash = 1.0 - w_eq - w_gld
        r_p = w_eq * r_core + w_gld * r_gld
        state["nav"] *= (1.0 + r_p)
        for b in BENCH:
            rb = rets.loc[d].get(b, np.nan)
            state["bench"][b] *= (1.0 + (0.0 if not np.isfinite(rb) else float(rb)))
        rows.append([str(d.date()), state["nav"]] + [state["bench"][b] for b in BENCH]
                    + [round(w_eq, 4), round(w_eq, 4), round(w_gld, 4), round(w_cash, 4)])
    if days:
        state["last_mark"] = str(days[-1].date())

    if rows:
        tr = pd.read_csv(TRACK)
        merged = (pd.concat([tr, pd.DataFrame(rows, columns=TRACK_COLS)], ignore_index=True)
                  .drop_duplicates("date", keep="last").sort_values("date"))
        _atomic(lambda p: merged.to_csv(p, index=False), TRACK)
    _atomic(lambda p: json.dump(state, open(p, "w"), indent=1), STATE)

    # refresh the displayed target book (f may have moved even without a NAV day)
    fh = fpath["f_held"].dropna()
    if len(fh):
        book = json.load(open(BOOK))
        f_now = float(fh.iloc[-1])
        book["f_held"] = round(f_now, 4)
        book["w_eq"] = round(f_now, 4)
        book["w_gld"] = round(P["gld_share"] * (1 - f_now), 4)
        book["w_cash"] = round((1 - P["gld_share"]) * (1 - f_now), 4)
        book["f_asof"] = str(fh.index[-1].date())
        _atomic(lambda p: json.dump(book, open(p, "w"), indent=1), BOOK)

    print(f"[build] appended {len(rows)} daily mark(s); last_mark={state['last_mark']}")
    return len(rows)


def _core_last_date():
    with open(CORE_RET) as f:
        last = None
        for row in csv.DictReader(f):
            last = row["date"]
    return last


# ---------------- page data ----------------
def compute_data():
    rows = list(csv.DictReader(open(TRACK)))
    dates = [r["date"] for r in rows]
    keys = ["nav"] + BENCH
    series = {k: [float(r[k]) for r in rows] for k in keys}
    f_hist = [float(r["f_applied"]) if r["f_applied"] else None for r in rows]

    def tot(v):
        return (v[-1] / v[0] - 1.0) * 100.0

    def mdd(v):
        peak, m = v[0], 0.0
        for x in v:
            peak = max(peak, x)
            m = min(m, x / peak - 1.0)
        return m * 100.0

    idx = {k: [100.0 * x / series[k][0] for x in series[k]] for k in keys}
    state = json.load(open(STATE))
    book = json.load(open(BOOK))
    stats = json.load(open(STATS))

    standings = []
    for k in ["nav"] + BENCH:
        standings.append({
            "k": "FORTRESS" if k == "nav" else k,
            "ret": round(tot(series[k]), 2),
            "dd": round(mdd(series[k]), 2),
            "level": round(series[k][-1], 0),
        })
    return {
        "dates": dates, "idx": idx, "f_hist": f_hist,
        "asof": dates[-1], "inception": state["inception"],
        "nav": series["nav"][-1],
        "ret_pct": round(tot(series["nav"]), 2),
        "spread_spy": round(tot(series["nav"]) - tot(series["SPY"]), 2),
        "standings": standings,
        "book": book, "stats": stats,
        "n_marks": len(rows) - 1,
    }


# ---------------- page ----------------
def render(d):
    payload = json.dumps({
        "dates": d["dates"],
        "series": [
            {"key": "FORTRESS", "vals": d["idx"]["nav"]},
            {"key": "SPY", "vals": d["idx"]["SPY"]},
            {"key": "QQQ", "vals": d["idx"]["QQQ"]},
            {"key": "SPUS", "vals": d["idx"]["SPUS"]},
            {"key": "HLAL", "vals": d["idx"]["HLAL"]},
        ],
        "f": d["f_hist"],
    })
    b = d["book"]
    st = d["stats"]
    core_rows = "".join(
        f"<tr><td>{t}</td><td class='num'>{w*100:.2f}%</td></tr>"
        for t, w in sorted(b["core_weights"].items(), key=lambda kv: -kv[1])[:15])
    stand_rows = "".join(
        f"<tr class='{'hero' if s['k']=='FORTRESS' else ''}'><td>{s['k']}</td>"
        f"<td class='num'>{s['ret']:+.2f}%</td><td class='num'>{s['dd']:.2f}%</td>"
        f"<td class='num'>${s['level']:,.0f}</td></tr>" for s in d["standings"])
    tear_rows = "".join(
        f"<tr class='{'hero' if r['k']=='FORTRESS' else ''}'><td>{r['k']}</td>"
        f"<td class='num'>{r['cagr']}%</td><td class='num'>{r['sharpe']}</td>"
        f"<td class='num'>{r['mdd']}%</td><td class='num'>{r['worst']}%</td></tr>"
        for r in st["rows"])
    target_rows = "".join(
        f"<tr><td>{t['name']}</td><td class='num'>{t['value']}</td><td class='num'>{t['target']}</td>"
        f"<td class='{'ok' if t['pass'] else 'miss'}'>{'PASS' if t['pass'] else 'MISS'}</td></tr>"
        for t in st["targets"])
    ann = st["annual"]
    ann_cells = "".join(f"<div class='yr'><span>{y}</span><b class='{ 'neg' if v<0 else ''}'>"
                        f"{v:+.1f}%</b></div>" for y, v in ann.items())

    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>FORTRESS — live record</title>
<style>
:root {{
  color-scheme: light;
  --page:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --ring:rgba(11,11,11,.10); --good:#006300; --bad:#d03b3b;
  --s1:#2a78d6; --s2:#eb6834; --s3:#1baf7a; --s4:#eda100; --s5:#e87ba4;
  --f-fill:#9ec5f4;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    color-scheme: dark;
    --page:#0d0d0d; --surface:#1a1a19; --ink:#ffffff; --ink2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,.10); --good:#0ca30c; --bad:#e66767;
    --s1:#3987e5; --s2:#d95926; --s3:#199e70; --s4:#c98500; --s5:#d55181;
    --f-fill:#184f95;
  }}
}}
* {{ box-sizing:border-box; margin:0; }}
body {{ background:var(--page); color:var(--ink);
  font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; padding:24px 16px 64px; }}
.wrap {{ max-width:980px; margin:0 auto; }}
h1 {{ font-size:26px; letter-spacing:.5px; }}
h1 small {{ color:var(--ink2); font-size:14px; font-weight:400; margin-left:10px; }}
h2 {{ font-size:15px; color:var(--ink2); font-weight:600; margin:34px 0 10px; }}
.card {{ background:var(--surface); border:1px solid var(--ring); border-radius:12px;
  padding:18px; }}
.tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  gap:10px; margin-top:14px; }}
.tile {{ background:var(--surface); border:1px solid var(--ring); border-radius:12px;
  padding:12px 14px; }}
.tile .lbl {{ font-size:12px; color:var(--muted); }}
.tile .val {{ font-size:22px; font-weight:650; margin-top:2px; }}
.tile .sub {{ font-size:12px; color:var(--ink2); }}
.pos {{ color:var(--good); }} .neg {{ color:var(--bad); }}
table {{ width:100%; border-collapse:collapse; font-size:14px; }}
th {{ text-align:left; color:var(--muted); font-weight:500; font-size:12px;
  border-bottom:1px solid var(--grid); padding:4px 8px; }}
td {{ padding:6px 8px; border-bottom:1px solid var(--grid); }}
tr:last-child td {{ border-bottom:none; }}
td.num, th.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
tr.hero td {{ font-weight:650; }}
td.ok {{ color:var(--good); font-weight:600; }} td.miss {{ color:var(--bad); font-weight:600; }}
.legend {{ display:flex; flex-wrap:wrap; gap:14px; font-size:12.5px; color:var(--ink2);
  margin:8px 2px 2px; }}
.legend span {{ display:inline-flex; align-items:center; gap:6px; }}
.sw {{ width:14px; height:3px; border-radius:2px; display:inline-block; }}
.shield-bar {{ display:flex; height:26px; border-radius:8px; overflow:hidden; margin:10px 0 6px;
  background:var(--grid); }}
.shield-bar div {{ height:100%; }}
.slabel {{ display:flex; gap:16px; font-size:12.5px; color:var(--ink2); }}
.years {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(74px,1fr)); gap:6px; }}
.yr {{ background:var(--surface); border:1px solid var(--ring); border-radius:8px;
  padding:6px 8px; font-size:12px; }}
.yr span {{ color:var(--muted); display:block; }}
.yr b {{ font-variant-numeric:tabular-nums; }}
#chart, #fchart {{ width:100%; display:block; }}
#tip {{ position:fixed; pointer-events:none; background:var(--surface);
  border:1px solid var(--ring); border-radius:8px; padding:8px 10px; font-size:12.5px;
  box-shadow:0 4px 14px rgba(0,0,0,.12); display:none; z-index:5; min-width:150px; }}
#tip .d {{ color:var(--muted); margin-bottom:4px; }}
#tip .row {{ display:flex; justify-content:space-between; gap:14px; }}
.note {{ font-size:12.5px; color:var(--muted); margin-top:8px; }}
.disc {{ font-size:12px; color:var(--muted); line-height:1.6; }}
a {{ color:var(--s1); }}
</style></head><body><div class="wrap">

<h1>FORTRESS <small>halal S&amp;P-500 strategy — live paper record</small></h1>
<div class="note">Beats the S&amp;P with a shield, not a rocket. Long-only, unlevered,
AAOIFI-screened. Marked {long_date(d["asof"])} · record since {long_date(d["inception"])}
· {d["n_marks"]} trading day(s).</div>

<div class="tiles">
  <div class="tile"><div class="lbl">Paper NAV</div>
    <div class="val">${d["nav"]:,.0f}</div>
    <div class="sub {'pos' if d['ret_pct']>=0 else 'neg'}">{d["ret_pct"]:+.2f}% since inception</div></div>
  <div class="tile"><div class="lbl">vs SPY</div>
    <div class="val {'pos' if d['spread_spy']>=0 else 'neg'}">{d["spread_spy"]:+.2f} pp</div>
    <div class="sub">spread since inception</div></div>
  <div class="tile"><div class="lbl">Shield — equity fraction f</div>
    <div class="val">{b["f_held"]*100:.0f}%</div>
    <div class="sub">as of {b.get("f_asof", d["asof"])}</div></div>
  <div class="tile"><div class="lbl">De-risked sleeve</div>
    <div class="val">{(b["w_gld"]+b["w_cash"])*100:.0f}%</div>
    <div class="sub">{b["w_gld"]*100:.0f}% gold · {b["w_cash"]*100:.0f}% cash (0%)</div></div>
</div>

<h2>Every $100 since inception</h2>
<div class="card">
  <svg id="chart" height="320" role="img" aria-label="Growth of $100: FORTRESS vs SPY, QQQ, SPUS, HLAL"></svg>
  <div class="legend">
    <span><i class="sw" style="background:var(--s1)"></i>FORTRESS</span>
    <span><i class="sw" style="background:var(--s2)"></i>SPY</span>
    <span><i class="sw" style="background:var(--s3)"></i>QQQ</span>
    <span><i class="sw" style="background:var(--s4)"></i>SPUS</span>
    <span><i class="sw" style="background:var(--s5)"></i>HLAL</span>
  </div>
</div>

<h2>The shield in action — executed equity fraction</h2>
<div class="card">
  <svg id="fchart" height="120" role="img" aria-label="Executed equity fraction over time"></svg>
  <div class="note">f rides 100% in calm tape; the vol target and the drawdown brake pull it
  as low as 25% under stress. De-risked capital: half gold, half 0% cash. Frozen rule —
  never refit.</div>
</div>

<h2>Standing vs the field (live record)</h2>
<div class="card"><table>
  <tr><th>Series</th><th class="num">Return</th><th class="num">Max DD</th><th class="num">$250k grew to</th></tr>
  {stand_rows}
</table></div>

<h2>Audited backtest — 2010–2025, net of costs, delisting-safe data</h2>
<div class="card"><table>
  <tr><th></th><th class="num">CAGR</th><th class="num">Sharpe</th><th class="num">Max DD</th><th class="num">Worst yr</th></tr>
  {tear_rows}
</table>
<div class="note">{st["note"]}</div></div>

<h2>Product targets (full period, net)</h2>
<div class="card"><table>
  <tr><th>Target</th><th class="num">Value</th><th class="num">Bar</th><th>Verdict</th></tr>
  {target_rows}
</table></div>

<h2>Backtest annual returns (gross)</h2>
<div class="years">{ann_cells}</div>

<h2>Core book — top 15 of {len(b["core_weights"])} holdings ({b["w_eq"]*100:.0f}% of NAV)</h2>
<div class="card"><table>
  <tr><th>Ticker</th><th class="num">Weight in core</th></tr>
  {core_rows}
</table>
<div class="note">Top-100 halal Nasdaq book (deployed top-60 form), cap-weighted, 20% single-name
cap, AAOIFI screen + 130-name exclusion list. Reconstituted monthly.</div></div>

<h2>Disclosures</h2>
<div class="disc">Paper NAV is simulated — modeled from adjusted daily closing prices, not
broker fills — and gross of trading costs. The 2010–2025 figures are hypothetical backtested
results on point-in-time, delisting-safe CRSP/Compustat data; the 2020–2025 window is
semi-out-of-sample (the strategy's core was finalized after that window had been observed
once). The live record on this page is the strategy's first true forward test. The clamp
parameters (σ*=0.20, D*=0.12) and every convention are frozen; nothing on this page is ever
refit. Past performance does not predict future results. For the operator's use — not an
offer or solicitation. Cash earns 0% (no interest, halal). Purification of impermissible
dividend income is handled at the administrator level.</div>

</div>
<div id="tip"></div>
<script>
const DATA = {payload};
const CSS = getComputedStyle(document.documentElement);
const COL = ["--s1","--s2","--s3","--s4","--s5"].map(v => `var(${{v}})`);

function draw() {{
  drawGrowth(); drawF();
}}
function sizes(svg, mL, mR, mT, mB) {{
  const W = svg.clientWidth, H = parseInt(svg.getAttribute("height"));
  svg.setAttribute("viewBox", `0 0 ${{W}} ${{H}}`);
  return {{W, H, mL, mR, mT, mB, iw: W-mL-mR, ih: H-mT-mB}};
}}
function drawGrowth() {{
  const svg = document.getElementById("chart");
  const g = sizes(svg, 44, 70, 10, 22);
  const n = DATA.dates.length;
  let lo = Infinity, hi = -Infinity;
  DATA.series.forEach(s => s.vals.forEach(v => {{ lo = Math.min(lo,v); hi = Math.max(hi,v); }}));
  const pad = (hi-lo)*0.06 || 2; lo -= pad; hi += pad;
  const x = i => g.mL + (n<=1 ? 0 : i/(n-1)*g.iw);
  const y = v => g.mT + (1-(v-lo)/(hi-lo))*g.ih;
  let el = "";
  const ticks = 4;
  for (let t=0;t<=ticks;t++) {{
    const v = lo + (hi-lo)*t/ticks, yy = y(v);
    el += `<line x1="${{g.mL}}" x2="${{g.mL+g.iw}}" y1="${{yy}}" y2="${{yy}}" stroke="var(--grid)"/>`;
    el += `<text x="${{g.mL-6}}" y="${{yy+4}}" text-anchor="end" font-size="11" fill="var(--muted)">$${{v.toFixed(0)}}</text>`;
  }}
  const step = Math.max(1, Math.floor(n/6));
  for (let i=0;i<n;i+=step) {{
    el += `<text x="${{x(i)}}" y="${{g.H-6}}" text-anchor="middle" font-size="11" fill="var(--muted)">${{DATA.dates[i].slice(5)}}</text>`;
  }}
  DATA.series.forEach((s,si) => {{
    const pts = s.vals.map((v,i)=>`${{x(i).toFixed(1)}},${{y(v).toFixed(1)}}`).join(" ");
    el += `<polyline points="${{pts}}" fill="none" stroke="${{COL[si]}}" stroke-width="${{si===0?2.5:2}}" ${{si>0?'opacity="0.85"':''}}/>`;
    const vy = y(s.vals[n-1]);
    if (si < 2) el += `<text x="${{g.mL+g.iw+6}}" y="${{vy+4}}" font-size="11.5" font-weight="600" fill="var(--ink2)">${{s.key}} $${{s.vals[n-1].toFixed(1)}}</text>`;
  }});
  el += `<line id="xh" y1="${{g.mT}}" y2="${{g.mT+g.ih}}" stroke="var(--axis)" stroke-dasharray="3,3" visibility="hidden"/>`;
  svg.innerHTML = el;
  hover(svg, g, n, i => {{
    let h = `<div class="d">${{DATA.dates[i]}}</div>`;
    DATA.series.forEach((s,si)=>{{ h += `<div class="row"><span><i class="sw" style="background:${{COL[si]}};margin-right:5px"></i>${{s.key}}</span><b>$${{s.vals[i].toFixed(1)}}</b></div>`; }});
    return h;
  }}, x);
}}
function drawF() {{
  const svg = document.getElementById("fchart");
  const g = sizes(svg, 44, 70, 8, 20);
  const f = DATA.f, n = f.length;
  const x = i => g.mL + (n<=1 ? 0 : i/(n-1)*g.iw);
  const y = v => g.mT + (1-(v-0)/(1.05))*g.ih;
  let el = "";
  [0.25, 0.5, 1.0].forEach(v => {{
    el += `<line x1="${{g.mL}}" x2="${{g.mL+g.iw}}" y1="${{y(v)}}" y2="${{y(v)}}" stroke="var(--grid)"/>`;
    el += `<text x="${{g.mL-6}}" y="${{y(v)+4}}" text-anchor="end" font-size="11" fill="var(--muted)">${{(v*100)}}%</text>`;
  }});
  let area = `M ${{x(0)}} ${{y(0)}} `;
  f.forEach((v,i)=>{{ area += `L ${{x(i).toFixed(1)}} ${{y(v==null?0:v).toFixed(1)}} `; }});
  area += `L ${{x(n-1)}} ${{y(0)}} Z`;
  el += `<path d="${{area}}" fill="var(--f-fill)" opacity="0.55"/>`;
  const pts = f.map((v,i)=>`${{x(i).toFixed(1)}},${{y(v==null?0:v).toFixed(1)}}`).join(" ");
  el += `<polyline points="${{pts}}" fill="none" stroke="var(--s1)" stroke-width="2"/>`;
  el += `<line id="xh" y1="${{g.mT}}" y2="${{g.mT+g.ih}}" stroke="var(--axis)" stroke-dasharray="3,3" visibility="hidden"/>`;
  svg.innerHTML = el;
  hover(svg, g, n, i => `<div class="d">${{DATA.dates[i]}}</div><div class="row"><span>equity f</span><b>${{(f[i]*100).toFixed(0)}}%</b></div>`, x);
}}
function hover(svg, g, n, html, x) {{
  const tip = document.getElementById("tip");
  const xh = svg.querySelector("#xh");
  svg.addEventListener("mousemove", e => {{
    const r = svg.getBoundingClientRect();
    const px = e.clientX - r.left;
    if (px < g.mL || px > g.mL+g.iw) {{ tip.style.display="none"; xh.setAttribute("visibility","hidden"); return; }}
    const i = Math.round((px-g.mL)/g.iw*(n-1));
    xh.setAttribute("x1", x(i)); xh.setAttribute("x2", x(i));
    xh.setAttribute("visibility","visible");
    tip.innerHTML = html(i);
    tip.style.display = "block";
    tip.style.left = Math.min(e.clientX+14, window.innerWidth-190) + "px";
    tip.style.top = (e.clientY+14) + "px";
  }});
  svg.addEventListener("mouseleave", () => {{
    tip.style.display="none"; xh.setAttribute("visibility","hidden");
  }});
}}
draw();
addEventListener("resize", draw);
</script>
</body></html>"""


def main():
    offline = "--offline" in sys.argv
    if not offline:
        try:
            fetch_and_mark()
        except Exception as e:
            print(f"[build] fetch/mark FAILED ({e!r}) — rebuilding page from last good data")
    d = compute_data()
    html = render(d)
    _atomic(lambda p: open(p, "w", encoding="utf-8").write(html), OUT)
    print(f"[build] wrote index.html (marked through {d['asof']})")


if __name__ == "__main__":
    main()
