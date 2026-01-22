# app.py
import streamlit as st
import pandas as pd
import numpy as np

from data_engine import get_realtime_synced_data, format_kst_time, get_market_macro
from scoring import classify_mode
from strategies import default_params
from backtest import backtest_single

# ============================================================
# Page config
# ============================================================
st.set_page_config(page_title="AI Stock Sniper", layout="wide")

# ============================================================
# Global UI CSS (clean, consistent scale, spacing)
# ============================================================
st.markdown("""
<style>
/* ----- Global Typography & Layout ----- */
:root{
  --bg: #0b0f17;
  --panel: #0f1624;
  --panel2:#0c1320;
  --border: rgba(255,255,255,0.08);
  --text: rgba(255,255,255,0.88);
  --muted: rgba(255,255,255,0.58);
  --muted2: rgba(255,255,255,0.42);
  --accent: #6ea8fe;
  --good: #34d399;
  --warn: #fbbf24;
  --bad:  #fb7185;
  --chip: rgba(255,255,255,0.06);
  --shadow: 0 10px 30px rgba(0,0,0,0.35);
}

.stApp{
  background: radial-gradient(1200px 600px at 20% 0%, rgba(110,168,254,0.16), transparent 55%),
              radial-gradient(900px 600px at 80% 10%, rgba(52,211,153,0.10), transparent 60%),
              var(--bg);
  color: var(--text);
}

/* Remove extra top padding */
.block-container{ padding-top: 1.4rem; padding-bottom: 2rem; }

/* Sidebar */
section[data-testid="stSidebar"]{
  background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] .stMarkdown{ color: var(--text); }

/* Inputs */
.stTextInput input, .stNumberInput input{
  background: rgba(255,255,255,0.05) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: 10px !important;
}
div[data-baseweb="select"] > div{
  background: rgba(255,255,255,0.05) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
}
button[kind="primary"]{
  border-radius: 12px !important;
  border: 1px solid rgba(110,168,254,0.30) !important;
  background: linear-gradient(180deg, rgba(110,168,254,0.30), rgba(110,168,254,0.12)) !important;
  box-shadow: 0 10px 22px rgba(0,0,0,0.22);
}
button[kind="secondary"]{
  border-radius: 12px !important;
}

/* Dataframe */
[data-testid="stDataFrame"]{
  border: 1px solid var(--border);
  border-radius: 14px;
  overflow: hidden;
}

/* Tabs */
button[data-baseweb="tab"]{
  font-size: 0.95rem !important;
  color: var(--muted) !important;
}
button[data-baseweb="tab"][aria-selected="true"]{
  color: var(--text) !important;
}

/* Headings scale */
h1{ font-size: 2.0rem !important; letter-spacing: -0.02em; margin-bottom: 0.4rem; }
h2{ font-size: 1.35rem !important; letter-spacing: -0.01em; margin-top: 0.2rem; }
h3{ font-size: 1.05rem !important; margin-top: 0.2rem; }
small, .muted { color: var(--muted); }

/* Cards */
.card{
  background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.03));
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 14px 16px;
  box-shadow: var(--shadow);
}
.card-title{ font-size: 0.85rem; color: var(--muted); margin-bottom: 6px; }
.card-value{ font-size: 1.55rem; font-weight: 800; letter-spacing: -0.02em; }
.card-sub{ font-size: 0.85rem; color: var(--muted); margin-top: 6px; }

/* Section */
.section{
  margin-top: 14px;
  margin-bottom: 10px;
}
.section-header{
  display:flex; align-items:center; justify-content:space-between;
  margin: 8px 2px 10px 2px;
}
.section-title{
  font-size: 1.05rem; font-weight: 800; color: var(--text);
}
.section-sub{
  font-size: 0.85rem; color: var(--muted);
}

/* Chips */
.chip{
  display:inline-flex;
  align-items:center;
  gap:8px;
  padding: 6px 10px;
  border-radius: 999px;
  background: var(--chip);
  border: 1px solid var(--border);
  color: var(--muted);
  font-size: 0.82rem;
  white-space: nowrap;
}
.dot{
  width: 8px; height: 8px; border-radius: 999px;
  display:inline-block;
}
.hr{
  height:1px; background: var(--border); margin: 10px 0 8px 0;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# UI helpers
# ============================================================
def color_for_metric(name: str, value) -> str:
    """Very light heuristic only for display."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "var(--muted)"
    if name in ("winrate", "cagr", "expectancy", "final_equity", "profit_factor"):
        # positive good
        if name == "profit_factor":
            if value >= 1.5: return "var(--good)"
            if value < 1.0: return "var(--bad)"
            return "var(--warn)"
        if value >= 0.0: return "var(--good)"
        return "var(--bad)"
    if name in ("mdd",):
        # more negative worse
        if value <= -0.35: return "var(--bad)"
        if value <= -0.20: return "var(--warn)"
        return "var(--good)"
    return "var(--text)"

def kpi_card(title: str, value: str, subtitle: str = "", tone: str = "var(--text)"):
    st.markdown(f"""
    <div class="card">
      <div class="card-title">{title}</div>
      <div class="card-value" style="color:{tone};">{value}</div>
      <div class="card-sub">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

def chip(label: str, tone: str = "var(--muted)"):
    st.markdown(f"""
    <span class="chip" style="color:{tone};">
      <span class="dot" style="background:{tone};"></span>
      {label}
    </span>
    """, unsafe_allow_html=True)

def section_header(title: str, subtitle: str = ""):
    st.markdown(f"""
    <div class="section-header">
      <div>
        <div class="section-title">{title}</div>
        <div class="section-sub">{subtitle}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

def fmt_pct(x, digits=1):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"{x*100:.{digits}f}%"

def fmt_float(x, digits=2):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"{x:.{digits}f}"

# ============================================================
# Sidebar (clean inputs)
# ============================================================
with st.sidebar:
    st.markdown("## AI Stock Sniper")
    st.markdown('<span class="muted">Quantile scoring • Mode-specific • Backtest-ready</span>', unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    ticker = st.text_input("Ticker", value="NVDA").upper().strip()
    include_extended = st.toggle("Include extended hours (pre/post)", value=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown("### Backtest settings")
    fee_bps = st.number_input("Fee (bps, round trip per side included in engine)", value=20.0, step=5.0)

    # Optional: mode overrides (UI only). Engine 변경 없이도 가능.
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    override = st.toggle("Advanced: override default params", value=False)

    overrides = {}
    if override:
        mode_pick = st.selectbox("Mode to override", ["MOMO", "SQUEEZE", "QUALITY"])
        base = default_params(mode_pick)
        overrides["mode"] = mode_pick
        overrides["entry_score"] = st.slider("entry_score", 0.50, 0.90, float(base["entry_score"]), 0.01)
        overrides["max_hold"] = st.slider("max_hold (days)", 5, 120, int(base["max_hold"]), 1)
        overrides["atr_stop"] = st.slider("atr_stop", 1.0, 4.0, float(base["atr_stop"]), 0.1)
        overrides["atr_target"] = st.slider("atr_target", 2.0, 10.0, float(base["atr_target"]), 0.1)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    run = st.button("Run analysis", type="primary", use_container_width=True)

if not run:
    st.stop()

# ============================================================
# Data load
# ============================================================
df, data_time = get_realtime_synced_data(ticker, include_extended=include_extended)
if df is None or df.empty:
    st.error("데이터를 불러올 수 없습니다.")
    st.stop()

macro = get_market_macro()
auto_mode = classify_mode(df)

# ============================================================
# Header
# ============================================================
left, right = st.columns([3, 2], vertical_alignment="center")
with left:
    st.markdown(f"# {ticker}")
    st.markdown(
        f'<span class="muted">Last update: {format_kst_time(data_time)} • Auto mode: <b>{auto_mode}</b></span>',
        unsafe_allow_html=True
    )
with right:
    # macro chips
    status = macro.get("status", "Unknown")
    if status == "FEAR":
        chip(f"Macro: FEAR (VIX high)", "var(--bad)")
    elif status == "CALM":
        chip(f"Macro: CALM (VIX low)", "var(--good)")
    else:
        chip(f"Macro: Normal", "var(--muted)")

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

# ============================================================
# Tabs
# ============================================================
tab_overview, tab_backtests, tab_data = st.tabs(["Overview", "Backtests", "Data preview"])

# ============================================================
# Overview
# ============================================================
with tab_overview:
    section_header("Snapshot", "Key context and current state (display only)")

    last = df.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Price (Close)", f"${fmt_float(float(last['Close']), 2)}", "Latest close")
    with c2:
        kpi_card("ATR (14)", f"{fmt_float(float(last['ATR']), 2)}", "Volatility unit")
    with c3:
        kpi_card("MA20", f"${fmt_float(float(last['MA20']), 2)}", "Short trend")
    with c4:
        kpi_card("MA120", f"${fmt_float(float(last['MA120']), 2)}", "Long trend")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    section_header("What this UI should NOT do", "Avoid confusing UI polish with decision quality")
    st.info(
        "이 화면은 '보기 좋게' 만든 것이지, 신호의 유효성을 보장하지 않습니다. "
        "신호는 Backtests 탭의 성과/분산/실패구간을 먼저 통과해야 의미가 생깁니다."
    )

# ============================================================
# Backtests
# ============================================================
with tab_backtests:
    section_header("Mode comparison", "MOMO / SQUEEZE / QUALITY (single-position long, next-open entry)")

    results = []
    curves = {}
    trades_map = {}

    modes = ["MOMO", "SQUEEZE", "QUALITY"]

    # If override, run one overridden mode + others default (so you can compare)
    override_mode = overrides.get("mode") if override else None

    for mode in modes:
        p = default_params(mode)
        if override and mode == override_mode:
            p = dict(p)
            p["entry_score"] = overrides["entry_score"]
            p["max_hold"] = overrides["max_hold"]
            p["atr_stop"] = overrides["atr_stop"]
            p["atr_target"] = overrides["atr_target"]

        curve, trades, summary = backtest_single(
            df=df,
            mode=mode,
            entry_score=p["entry_score"],
            max_hold=p["max_hold"],
            atr_stop=p["atr_stop"],
            atr_target=p["atr_target"],
            fee_bps=fee_bps
        )

        curves[mode] = curve
        trades_map[mode] = trades
        results.append({
            "mode": mode,
            "trades": summary.get("trades"),
            "CAGR": summary.get("cagr"),
            "MDD": summary.get("mdd"),
            "Winrate": summary.get("winrate"),
            "Expectancy": summary.get("expectancy"),
            "ProfitFactor": summary.get("profit_factor"),
            "FinalEquity": summary.get("final_equity"),
            "entry_score": p["entry_score"],
            "max_hold": p["max_hold"],
            "atr_stop": p["atr_stop"],
            "atr_target": p["atr_target"],
        })

    res_df = pd.DataFrame(results)

    # KPI row (best/worst)
    best = res_df.sort_values("FinalEquity", ascending=False).iloc[0]
    worst_mdd = res_df.sort_values("MDD", ascending=True).iloc[0]  # most negative

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("Best (Final Equity)", f"{best['mode']} • {fmt_float(float(best['FinalEquity']), 2)}",
                 "Higher is better", "var(--good)")
    with k2:
        kpi_card("Worst (MDD)", f"{worst_mdd['mode']} • {fmt_float(float(worst_mdd['MDD']), 2)}",
                 "More negative is worse", "var(--bad)")
    with k3:
        # macro display
        vix = macro.get("vix")
        tnx = macro.get("tnx")
        kpi_card("VIX / TNX", f"{fmt_float(vix,1)} / {fmt_float(tnx,2)}", "Context only", "var(--muted)")
    with k4:
        kpi_card("Fee", f"{fee_bps:.0f} bps", "Applied in backtest", "var(--muted)")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # Clean table formatting
    table = res_df.copy()
    # format numeric
    def _f(x):
        if x is None or (isinstance(x, float) and np.isnan(x)): return None
        return float(x)

    for c in ["CAGR", "MDD", "Winrate", "Expectancy"]:
        table[c] = table[c].apply(_f)
    table["CAGR"] = table["CAGR"].apply(lambda x: None if x is None else round(x, 3))
    table["MDD"] = table["MDD"].apply(lambda x: None if x is None else round(x, 3))
    table["Winrate"] = table["Winrate"].apply(lambda x: None if x is None else round(x, 3))
    table["Expectancy"] = table["Expectancy"].apply(lambda x: None if x is None else round(x, 4))
    table["ProfitFactor"] = table["ProfitFactor"].apply(lambda x: None if x is None else (np.inf if x == np.inf else round(float(x), 2)))
    table["FinalEquity"] = table["FinalEquity"].apply(lambda x: None if x is None else round(float(x), 2))

    st.dataframe(table, use_container_width=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    # Mode detail viewer
    section_header("Details", "Equity curve + recent trades")
    pick = st.radio("Select mode", modes, horizontal=True)

    colA, colB = st.columns([2, 1])
    with colA:
        st.line_chart(curves[pick].set_index(curves[pick].index)[["equity"]])
    with colB:
        row = res_df[res_df["mode"] == pick].iloc[0]
        tone_cagr = color_for_metric("cagr", float(row["CAGR"]) if row["CAGR"] is not None else None)
        tone_mdd = color_for_metric("mdd", float(row["MDD"]) if row["MDD"] is not None else None)
        tone_win = color_for_metric("winrate", float(row["Winrate"]) if row["Winrate"] is not None else None)

        kpi_card("CAGR", f"{fmt_pct(row['CAGR'],1) if row['CAGR'] is not None else 'N/A'}",
                 "Annualized (approx)", tone_cagr)
        kpi_card("MDD", f"{fmt_float(float(row['MDD']),2) if row['MDD'] is not None else 'N/A'}",
                 "Max drawdown", tone_mdd)
        kpi_card("Winrate", f"{fmt_pct(row['Winrate'],1) if row['Winrate'] is not None else 'N/A'}",
                 "Hit ratio", tone_win)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    trades = trades_map[pick]
    if trades is not None and not trades.empty:
        st.markdown("##### Recent trades (last 25)")
        st.dataframe(trades.tail(25), use_container_width=True)
    else:
        st.info("No trades for this mode under current settings.")

# ============================================================
# Data preview
# ============================================================
with tab_data:
    section_header("Raw data preview", "This is mostly for debugging / sanity checks")
    st.dataframe(df.tail(30), use_container_width=True)
