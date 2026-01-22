# app.py
import streamlit as st
import pandas as pd
import numpy as np

from ui_api import run_app_payload
from strategies import default_params

st.set_page_config(page_title="AI Stock Sniper", layout="wide")

# -------------------- Clean CSS (consistent type/spacing) --------------------
st.markdown("""
<style>
:root{
  --bg:#0b0f17; --panel:rgba(255,255,255,0.04); --border:rgba(255,255,255,0.08);
  --text:rgba(255,255,255,0.90); --muted:rgba(255,255,255,0.60);
  --good:#34d399; --warn:#fbbf24; --bad:#fb7185; --accent:#6ea8fe;
  --shadow:0 10px 30px rgba(0,0,0,0.35);
}
.stApp{ background: radial-gradient(1200px 600px at 15% 0%, rgba(110,168,254,0.18), transparent 55%),
                 radial-gradient(900px 600px at 85% 10%, rgba(52,211,153,0.10), transparent 60%),
                 var(--bg); color:var(--text); }
.block-container{ padding-top: 1.2rem; padding-bottom: 2.0rem; max-width: 1400px; }
section[data-testid="stSidebar"]{ border-right:1px solid var(--border); background: rgba(255,255,255,0.02); }
h1{ font-size:2.0rem !important; letter-spacing:-0.02em; margin-bottom:.2rem; }
h2{ font-size:1.25rem !important; letter-spacing:-0.01em; margin-top:0.2rem; }
small{ color:var(--muted); }
.hr{ height:1px; background:var(--border); margin: 10px 0 10px 0; }
.card{
  background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
  border:1px solid var(--border); border-radius:16px; padding:14px 16px; box-shadow:var(--shadow);
}
.card-title{ font-size:.82rem; color:var(--muted); margin-bottom:6px;}
.card-value{ font-size:1.55rem; font-weight:850; letter-spacing:-0.02em;}
.card-sub{ font-size:.85rem; color:var(--muted); margin-top:6px;}
.chip{
  display:inline-flex; align-items:center; gap:8px; padding:6px 10px; border-radius:999px;
  background: rgba(255,255,255,0.05); border:1px solid var(--border); color:var(--muted);
  font-size:.82rem; white-space:nowrap;
}
.dot{ width:8px; height:8px; border-radius:999px; display:inline-block;}
[data-testid="stDataFrame"]{ border:1px solid var(--border); border-radius:14px; overflow:hidden; }
button[kind="primary"]{
  border-radius:12px !important;
  border: 1px solid rgba(110,168,254,0.30) !important;
  background: linear-gradient(180deg, rgba(110,168,254,0.30), rgba(110,168,254,0.12)) !important;
}
</style>
""", unsafe_allow_html=True)

def kpi_card(title, value, sub="", tone="var(--text)"):
    st.markdown(f"""
    <div class="card">
      <div class="card-title">{title}</div>
      <div class="card-value" style="color:{tone};">{value}</div>
      <div class="card-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

def chip(label, tone="var(--muted)"):
    st.markdown(f"""
    <span class="chip" style="color:{tone};">
      <span class="dot" style="background:{tone};"></span>
      {label}
    </span>
    """, unsafe_allow_html=True)

def tone_score(x: float) -> str:
    if x >= 0.75: return "var(--good)"
    if x >= 0.65: return "var(--accent)"
    if x >= 0.55: return "var(--warn)"
    return "var(--bad)"

# -------------------- Sidebar --------------------
with st.sidebar:
    st.markdown("## AI Stock Sniper")
    st.markdown("<small>Quantile signals • Mode-specific scoring • Backtest-ready</small>", unsafe_allow_html=True)
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    ticker = st.text_input("Ticker", value="NVDA").upper().strip()
    include_extended = st.toggle("Include extended hours (pre/post)", value=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.markdown("### Backtest")
    fee_bps = st.number_input("Fee (bps)", value=20.0, step=5.0)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    override = st.toggle("Advanced override (one mode)", value=False)
    overrides = {}
    if override:
        mode_pick = st.selectbox("Override mode", ["MOMO", "SQUEEZE", "QUALITY"])
        base = default_params(mode_pick)
        overrides["mode"] = mode_pick
        overrides["entry_score"] = st.slider("entry_score", 0.50, 0.90, float(base["entry_score"]), 0.01)
        overrides["max_hold"] = st.slider("max_hold", 5, 120, int(base["max_hold"]), 1)
        overrides["atr_stop"] = st.slider("atr_stop", 1.0, 4.0, float(base["atr_stop"]), 0.1)
        overrides["atr_target"] = st.slider("atr_target", 2.0, 10.0, float(base["atr_target"]), 0.1)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    run = st.button("Run analysis", type="primary", use_container_width=True)

if not run:
    st.stop()

# -------------------- Run --------------------
with st.spinner("Loading & analyzing..."):
    payload = run_app_payload(
        ticker=ticker,
        include_extended=include_extended,
        fee_bps=fee_bps,
        overrides=overrides if override else None
    )

if not payload.get("ok"):
    st.error(payload.get("error", "Unknown error"))
    st.stop()

snap = payload["snapshot"]
auto_mode = payload["auto_mode"]
short_pack = payload["short_pack"]
bt = payload["backtests"]
intel = payload["intel"]
current_scores = payload["current_scores"]

# -------------------- Header --------------------
left, right = st.columns([3, 2], vertical_alignment="center")
with left:
    st.markdown(f"# {payload['ticker']}")
    st.markdown(f"<small>{snap['name']} • Updated: {snap['updated_at']} • Auto mode: <b>{auto_mode}</b></small>", unsafe_allow_html=True)

with right:
    # Short-data quality (informational risk)
    q = short_pack.get("quality", "MISSING/STALE")
    if "OK" in q:
        chip("Short data: OK", "var(--accent)")
    else:
        chip("Short data: Missing/Unreliable", "var(--warn)")

st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

tabs = st.tabs(["Overview", "Backtests", "News", "Insider", "Filings", "Data"])

# -------------------- Overview --------------------
with tabs[0]:
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Price (Close)", f"${snap['close']:.2f}", "Latest close")
    with c2:
        kpi_card("ATR(14)", f"{snap['atr']:.2f}", "Volatility unit")
    with c3:
        kpi_card("MA20", f"${snap['ma20']:.2f}", "Short trend")
    with c4:
        kpi_card("MA120", f"${snap['ma120']:.2f}", "Long trend")

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)

    c5, c6, c7, c8 = st.columns(4)
    with c5:
        kpi_card("RSI(14)", f"{snap['rsi']:.1f}", "Context (not a decision alone)")
    with c6:
        kpi_card("Vol spike", f"{snap['vol_spike']:.2f}x", "Volume / 20D avg")
    with c7:
        kpi_card("Squeeze", "ON" if snap["squeeze_on"] else "OFF",
                 f"streak {snap['squeeze_streak']} days",
                 "var(--accent)" if snap["squeeze_on"] else "var(--muted)")
    with c8:
        # current best mode by score today (not a trade)
        best_mode = max(current_scores.keys(), key=lambda m: current_scores[m]["score"])
        best_score = current_scores[best_mode]["score"]
        kpi_card("Today score (best mode)", f"{best_mode} • {best_score:.2f}", "Quantile-based composite",
                 tone_score(best_score))

    st.info(
        "중요: 이 화면은 ‘보기 좋게 정리’된 것이지, 의사결정 품질을 보장하지 않습니다. "
        "Backtests 탭에서 성과/드로다운/거래 수가 충분히 설득력 있는지 먼저 보세요."
    )

# -------------------- Backtests --------------------
with tabs[1]:
    rows = []
    for mode, pack in bt.items():
        s = pack["summary"]
        p = pack["params"]
        rows.append({
            "mode": mode,
            "trades": s.get("trades"),
            "CAGR": s.get("cagr"),
            "MDD": s.get("mdd"),
            "winrate": s.get("winrate"),
            "expectancy": s.get("expectancy"),
            "profit_factor": s.get("profit_factor"),
            "final_equity": s.get("final_equity"),
            "entry_score": p["entry_score"],
            "max_hold": p["max_hold"],
            "atr_stop": p["atr_stop"],
            "atr_target": p["atr_target"],
        })
    df_res = pd.DataFrame(rows)

    # Executive critique hint (risk-first)
    # If trades too few, all metrics are unreliable.
    warn_msgs = []
    for _, r in df_res.iterrows():
        if (r["trades"] is None) or (int(r["trades"]) < 20):
            warn_msgs.append(f"{r['mode']}: 거래 수가 적음(<20) → 성과지표 신뢰 낮음")
        if (r["MDD"] is not None) and (float(r["MDD"]) < -0.35):
            warn_msgs.append(f"{r['mode']}: MDD가 깊음({float(r['MDD']):.2f}) → 리스크 구조 불리")
    if warn_msgs:
        st.warning(" | ".join(warn_msgs))

    st.dataframe(df_res, use_container_width=True)

    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    pick = st.radio("Select mode", ["MOMO", "SQUEEZE", "QUALITY"], horizontal=True)

    colA, colB = st.columns([2, 1])
    with colA:
        curve = bt[pick]["curve"]
        st.line_chart(curve[["equity"]])
    with colB:
        s = bt[pick]["summary"]
        score_now = current_scores[pick]["score"]
        kpi_card("Today score", f"{score_now:.2f}", "Composite (quantile-based)", tone_score(score_now))
        kpi_card("Trades", f"{s.get('trades', 0)}", "Count (too low = unreliable)")
        mdd = s.get("mdd", None)
        kpi_card("MDD", "N/A" if mdd is None else f"{float(mdd):.2f}", "Max drawdown",
                 "var(--bad)" if (mdd is not None and float(mdd) < -0.30) else "var(--muted)")

    trades = bt[pick]["trades"]
    if trades is not None and not trades.empty:
        st.markdown("#### Recent trades (last 30)")
        st.dataframe(trades.tail(30), use_container_width=True)
    else:
        st.info("No trades for this mode under current settings.")

# -------------------- News --------------------
with tabs[2]:
    items = intel.get("news", []) or []
    if not items:
        st.warning("뉴스를 불러오지 못했습니다(네트워크/차단/일시 오류 가능).")
    else:
        for it in items:
            st.markdown(f"- [{it['title']}]({it['url']})")

# -------------------- Insider --------------------
with tabs[3]:
    ins = intel.get("insider", None)
    if ins is None or (hasattr(ins, "empty") and ins.empty):
        st.info("내부자 거래 데이터가 없거나 소스에서 제공되지 않습니다(= ‘없다’가 아니라 ‘잡히지 않았다’일 수 있음).")
    else:
        st.dataframe(ins, use_container_width=True)

# -------------------- Filings --------------------
with tabs[4]:
    filings = intel.get("filings", []) or []
    if not filings:
        st.warning("공시를 불러오지 못했습니다(SEC 차단/UA 설정/네트워크 이슈 가능).")
        st.info("해결: 환경변수 SEC_USER_AGENT를 실제 연락처 포함 형태로 설정 권장.")
    else:
        st.dataframe(pd.DataFrame(filings), use_container_width=True)

# -------------------- Data preview --------------------
with tabs[5]:
    st.dataframe(payload["df_tail"], use_container_width=True)
