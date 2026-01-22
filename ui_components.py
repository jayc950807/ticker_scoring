import streamlit as st

def inject_style():
    st.markdown("""
    <style>
      .stApp { background-color:#0b1020; color:#e7eaf3; }
      .block-container { padding-top: 1.1rem; padding-bottom: 2.2rem; max-width: 1200px; }
      h1,h2,h3 { letter-spacing: -0.02em; }
      .muted { color:#aab2c5; }
      .card {
        background:#111a33; border:1px solid #223058; border-radius:16px;
        padding:14px 16px; margin: 10px 0;
      }
      .pill {
        display:inline-block; padding:4px 10px; border-radius:999px;
        border:1px solid #2a3a6a; background:#0e1630; color:#dde3f5; font-size:0.85rem;
      }
      .good { color:#20c997; }
      .bad { color:#ff6b6b; }
      .warn { color:#ffd43b; }
      .info { color:#4dabf7; }
      [data-testid="stMetricValue"] { font-size:1.55rem; }
      [data-testid="stMetricLabel"] { color:#aab2c5; }
      a { color:#79c0ff !important; text-decoration:none; }
      a:hover { text-decoration:underline; }
      .grid2 { display:grid; grid-template-columns: 1fr 1fr; gap:10px; }
      .kv { display:flex; justify-content:space-between; gap:12px; padding:8px 0; border-bottom:1px solid #1e2a4f; }
      .kv:last-child { border-bottom:0; }
    </style>
    """, unsafe_allow_html=True)

def header(ticker: str, name: str):
    st.markdown(f"<h1>{ticker} <span class='muted' style='font-size:0.55em'> {name}</span></h1>", unsafe_allow_html=True)

def banner(reco: str, reason: str, score: int, winprob: float):
    cls = "good" if reco == "매수" else ("bad" if reco.startswith("매도") else "warn")
    st.markdown(
        f"""
        <div class='card'>
          <div style='display:flex; justify-content:space-between; align-items:center; gap:12px;'>
            <div>
              <span class='pill'>규칙 기반 추천</span>
              <span class='{cls}' style='font-size:1.45rem; font-weight:900; margin-left:10px;'>{reco}</span>
              <div class='muted' style='margin-top:8px'>{reason}</div>
            </div>
            <div style='text-align:right; min-width:260px;'>
              <div class='muted'>신뢰 점수(0~100)</div>
              <div style='font-size:2rem; font-weight:900;'>{score}</div>
              <div class='muted' style='margin-top:6px'>워크포워드 승률(가중)</div>
              <div style='font-size:1.25rem; font-weight:800;'>{winprob:.1f}%</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def summary_card(title: str, bullets: list[str]):
    st.markdown(
        "<div class='card'>"
        f"<div style='font-weight:900; margin-bottom:8px;'>{title}</div>"
        + "".join([f"<div class='muted'>• {b}</div>" for b in bullets])
        + "</div>",
        unsafe_allow_html=True
    )

def kv_card(title: str, items: list[tuple[str,str]]):
    rows = "".join([f"<div class='kv'><div class='muted'>{k}</div><div>{v}</div></div>" for k,v in items])
    st.markdown(f"<div class='card'><div style='font-weight:900; margin-bottom:8px;'>{title}</div>{rows}</div>", unsafe_allow_html=True)

def score_from_oos(oos: dict) -> tuple[int, float]:
    """
    점수(0~100)와 '승률'을 같이 노출.
    점수는 과신 방지를 위해 '최악 구간' 페널티 포함.
    """
    if not oos or not oos.get("ok", False):
        return 0, 0.0

    win = float(oos.get("가중승률(%)", 0.0))
    pf = float(oos.get("PF(중앙값)", 0.0))
    worst_pf = float(oos.get("최악PF", 0.0))
    trades = int(oos.get("총트레이드수", 0))
    zero = int(oos.get("무거래구간수", 0))

    score = 50
    score += min(20, max(-20, (win - 50) * 0.8))      # win 50% 기준
    score += min(20, max(-20, (pf - 1.2) * 25))       # pf 1.2 기준
    score -= min(20, max(0, (1.0 - worst_pf) * 40))   # 최악 구간 붕괴 페널티
    score += min(10, trades / 10)                     # 표본 보너스(최대 10)
    score -= min(15, zero * 4)                        # 무거래 구간 많으면 감점

    score = int(max(0, min(100, round(score))))
    return score, win
