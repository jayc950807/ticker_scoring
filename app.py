# app.py
from __future__ import annotations
import streamlit as st
import pandas as pd
import pytz
from datetime import datetime

from core.config import AppConfig
from core.ui import apply_theme, card
from core.data_engine import load_price_data, load_info
from core.indicators import add_indicators
from core.regime import detect_regime
from core.strategy import compute_entry_signal, squeeze_state
from core.optimizer import walkforward_optimize
from core.backtest import run_backtest
from core.scoring import score_all
from core.explain import decide_and_explain
from core.news import google_news_rss, try_translate_titles
from core.sec import get_recent_filings

CFG = AppConfig()

def kst_time_str(ts) -> str:
    try:
        tz = pytz.timezone(CFG.TZ)
        if getattr(ts, "tzinfo", None) is None:
            ts = pytz.utc.localize(ts)
        return ts.astimezone(tz).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(ts)

def short_metrics(info: dict) -> dict:
    spof = info.get("shortPercentOfFloat", None)
    dtc = info.get("shortRatio", None)
    float_shares = info.get("floatShares", None)

    q = "MISSING"
    if spof is not None or dtc is not None:
        q = "OK" if (spof is not None and dtc is not None) else "STALE/UNSURE"

    return {
        "short_pct": (None if spof is None else float(spof) * 100),
        "dtc": (None if dtc is None else float(dtc)),
        "float_shares": float_shares,
        "quality": q
    }

def fundamentals(info: dict) -> dict:
    return {
        "이름": info.get("longName") or info.get("shortName") or "",
        "시총": info.get("marketCap"),
        "PER": info.get("trailingPE"),
        "PBR": info.get("priceToBook"),
        "ROE": info.get("returnOnEquity"),
        "내부자보유율": info.get("heldPercentInsiders"),
    }

@st.cache_data(ttl=60*5, show_spinner=False)
def cached_news(ticker: str):
    items = google_news_rss(ticker, limit=8)
    return try_translate_titles(items)

@st.cache_data(ttl=60*10, show_spinner=False)
def cached_filings(ticker: str):
    return get_recent_filings(ticker, limit=20)

def gate_check(oos_stats: dict, worst_pf: float) -> dict:
    pf = float(oos_stats.get("pf", 0.0))
    win = float(oos_stats.get("win_rate", 0.0))
    mdd = float(oos_stats.get("mdd", 0.0))
    trades = int(oos_stats.get("trades", 0))

    fails = []
    if pf < CFG.GATE_MIN_PF_OOS: fails.append(f"PF<{CFG.GATE_MIN_PF_OOS}")
    if worst_pf < CFG.GATE_WORST_SEGMENT_PF: fails.append(f"최악구간 PF<{CFG.GATE_WORST_SEGMENT_PF}")
    if win < CFG.GATE_MIN_WIN_OOS: fails.append(f"승률<{CFG.GATE_MIN_WIN_OOS}%")
    if trades < CFG.GATE_MIN_TRADES_OOS: fails.append(f"거래수<{CFG.GATE_MIN_TRADES_OOS}")
    if mdd < -CFG.GATE_MAX_MDD_OOS: fails.append(f"MDD<-{CFG.GATE_MAX_MDD_OOS}%")

    return {"pass": len(fails) == 0, "fail_reason": ", ".join(fails)}

def main():
    apply_theme()

    st.markdown("## 🦄 AI Stock Sniper (Auto)")

    with st.sidebar:
        st.markdown("### 입력")
        ticker = st.text_input("티커", value="NVDA").upper().strip()
        include_extended = st.toggle("프리/애프터 포함(당일 보정)", value=True)
        st.markdown("---")
        st.markdown("### 실행")
        run = st.button("자동 분석 실행", use_container_width=True)
        st.caption("※ 설정을 건드리게 만들면 과최적화가 시작됩니다. 기본은 완전 자동입니다.")

    if not run:
        st.info("좌측에서 티커 입력 후 **자동 분석 실행**을 눌러.")
        return

    with st.spinner(f"{ticker} 자동 분석 중… (데이터/지표/워크포워드/결론 생성)"):
        info = load_info(ticker)
        df_raw, data_time = load_price_data(ticker, include_extended, CFG)
        if df_raw is None or df_raw.empty:
            st.error("가격 데이터를 불러오지 못했습니다. 티커/네트워크/yfinance 상태를 확인해.")
            return

        df = add_indicators(df_raw)
        df = df.dropna(subset=["MA200", "MA20", "ATR14", "RSI14"]).copy()
        if len(df) < 400:
            st.error("지표 계산 후 유효 데이터가 부족합니다.")
            return

        # 레짐/신호
        regime = detect_regime(df)
        entry_sig = compute_entry_signal(df)
        latest_signal = bool(entry_sig.iloc[-1])
        sqz = squeeze_state(df, min_on_days=5)

        # 유동성(20일 평균 거래대금)
        dollar20 = float(df["DollarVol20"].iloc[-1]) if "DollarVol20" in df else 0.0
        liquidity_ok = dollar20 >= CFG.MIN_DOLLAR_VOL_20D

        # 워크포워드 최적화(파라미터 선택)
        opt = walkforward_optimize(df, entry_sig, CFG)
        if not opt.get("ok", False):
            st.error(f"최적화 실패: {opt.get('reason','')}")
            return

        best = opt["best"]
        stop_atr = float(best["stop_atr"])
        take_atr = float(best["take_atr"])
        max_hold = int(best["max_hold"])
        worst_pf = float(best["pf_worst"])

        # OOS 성능을 대표로 삼기 위해: 워크포워드 전체 구간과 동일 철학으로 최근 test 길이로 별도 측정
        # (실전에서는 “최근 성과”에 과집착하면 망가져서, 여기서는 참고용 요약만 제공)
        # 전체 구간에 대해 chosen params로 백테스트(투명 공개)
        bt = run_backtest(df, entry_sig, stop_atr, take_atr, max_hold, CFG.COST_BPS)
        trades = bt["trades"]
        equity = bt["equity"]
        total_stats = bt["stats"]

        # OOS 대체 지표: 워크포워드 표의 중앙값 성능(더 보수적)
        oos_stats = {
            "pf": float(best["pf_med"]),
            "win_rate": float(best["win_med"]),
            "mdd": float(best["mdd_worst"]),    # 보수적으로 worst로
            "trades": int(best["trades_sum"])
        }

        gate = gate_check(oos_stats, worst_pf)
        sm = short_metrics(info)
        sc = score_all(oos_stats, regime, liquidity_ok, sm["quality"])
        decision = decide_and_explain(
            score=sc["score"],
            gate=gate,
            regime=regime,
            latest_signal=latest_signal,
            oos_stats=oos_stats,
            liquidity_ok=liquidity_ok
        )

        # 뉴스/공시
        news = cached_news(ticker)
        filings = cached_filings(ticker)

        # 펀더멘털
        fund = fundamentals(info)

    # =========================
    # 상단 요약
    # =========================
    name = fund.get("이름") or ticker
    st.markdown(f"### {ticker}  <span class='muted' style='font-size:0.95rem;'>({name})</span>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card("결론(BUY/WAIT/SELL)", decision["action"], decision["headline"])
    with c2:
        card("신뢰 점수(0~100)", str(decision["score"]), "검증(OOS)+레짐+실행가능성 기반")
    with c3:
        card("OOS PF(중앙값)", f"{oos_stats['pf']:.2f}", f"최악구간 PF {worst_pf:.2f}")
    with c4:
        card("OOS 승률(중앙값)", f"{oos_stats['win_rate']:.1f}%", f"OOS 거래수 합 {oos_stats['trades']}회")

    st.markdown(
        f"<div class='pill'>데이터 기준 시각: {kst_time_str(data_time)}</div> "
        f"<div class='pill'>레짐: {regime['레짐']}</div> "
        f"<div class='pill'>최적화: STOP {stop_atr} ATR / TAKE {take_atr} ATR / 최대보유 {max_hold}일</div> "
        f"<div class='pill'>유동성(20일 평균 거래대금): ${dollar20:,.0f}</div>",
        unsafe_allow_html=True
    )

    # =========================
    # 탭(요청사항: “각 탭별 현황 써머리” + “투명성”)
    # =========================
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🧾 요약 리포트",
        "🔍 투명 데이터(지표/원천)",
        "📈 백테스트/트레이드 로그",
        "📰 뉴스(번역) & 공시",
        "🧷 공매도/내부자/기본정보"
    ])

    with tab1:
        st.markdown("#### 현황 요약")
        st.write({
            "결론": decision["action"],
            "점수": decision["score"],
            "게이트 통과": gate["pass"],
            "게이트 실패 사유": gate["fail_reason"] if not gate["pass"] else "-",
            "오늘 진입 신호": latest_signal,
            "스퀴즈": sqz,
            "레짐": regime,
            "유동성 OK": liquidity_ok,
        })

        st.markdown("#### AI 근거(감사 가능한 형태)")
        for r in decision["reasons"]:
            st.markdown(f"- {r}")

        st.markdown("---")
        st.caption("⚠️ 이 결론은 ‘규칙 기반 자동판단’이며, 시장이 레짐 전환하면 성능은 급격히 붕괴할 수 있습니다. (그래서 BUY 차단/게이트를 강하게 걸었습니다.)")

    with tab2:
        st.markdown("#### 현황 요약")
        st.write({
            "표시 목적": "모든 산출 데이터 투명 공개",
            "포인트": "결론을 믿으려면, 아래 원천/지표가 어떻게 계산되는지 직접 감사 가능해야 함"
        })

        st.markdown("#### 최근 120일 원천 OHLCV + 주요 지표")
        cols = ["Open","High","Low","Close","Volume","MA20","MA60","MA200","RSI14","ATR14","MACD","MACD_Signal","BB_Upper","BB_Lower","KC_Upper","KC_Lower","SQUEEZE_RAW_ON","OBV","AD_Line","MFI14","DollarVol20","RangePct"]
        show = df[cols].tail(120).copy()
        st.dataframe(show, use_container_width=True)

        st.markdown("#### 진입 신호(최근 120일)")
        sig_df = pd.DataFrame({
            "Close": df["Close"].tail(120),
            "진입신호": entry_sig.tail(120).astype(int)
        })
        st.dataframe(sig_df, use_container_width=True)

    with tab3:
        st.markdown("#### 현황 요약")
        st.write({
            "전체 백테스트(투명 공개)": "선택된 파라미터로 전체 구간 트레이드 로그 제공",
            "주의": "전체 성과는 레짐이 섞여 과대평가/과소평가 가능 → 판단은 OOS 게이트 중심"
        })

        st.markdown("#### 전체 구간 성과(참고)")
        st.write(total_stats)

        st.markdown("#### 트레이드 로그")
        if trades is None or trades.empty:
            st.warning("트레이드가 거의 없거나 신호가 발생하지 않았습니다. (이 경우 실전 적용은 매우 위험)")
        else:
            st.dataframe(trades.tail(200), use_container_width=True)

        st.markdown("#### 워크포워드 후보 성능표(상위 20개)")
        st.dataframe(opt["wf_table"].head(20), use_container_width=True)

    with tab4:
        st.markdown("#### 현황 요약")
        st.write({
            "뉴스": "Google News RSS 기반 + 한국어 번역",
            "공시": "SEC data.sec.gov JSON 기반(직접 수집) — yfinance 누락 문제 회피"
        })

        st.markdown("#### 뉴스(한국어 번역)")
        if news:
            for it in news:
                st.markdown(f"- [{it['title_ko']}]({it['url']})")
                st.caption(it["title"])
        else:
            st.warning("뉴스를 불러오지 못했습니다(네트워크/차단/소스 상태).")

        st.markdown("---")
        st.markdown("#### SEC 공시")
        if filings is not None and not filings.empty:
            st.dataframe(filings, use_container_width=True)
        else:
            st.warning("SEC 공시를 불러오지 못했습니다(티커- CI K 매핑 실패/SEC 차단/네트워크).")

    with tab5:
        st.markdown("#### 현황 요약")
        st.write({
            "공매도(short)": "yfinance 소스는 누락/지연이 잦음 → 품질 플래그를 같이 표시",
            "내부자 보유율": "heldPercentInsiders 기반(없을 수 있음)",
        })

        sm = short_metrics(info)
        insider_pct = fund.get("내부자보유율", None)

        c1, c2, c3 = st.columns(3)
        with c1:
            card("공매도 비중(유통주 대비)", f"{sm['short_pct']:.2f}%" if sm["short_pct"] is not None else "N/A", f"데이터 품질: {sm['quality']}")
        with c2:
            card("DTC(Days To Cover)", f"{sm['dtc']:.2f}" if sm["dtc"] is not None else "N/A", "shortRatio 기반")
        with c3:
            card("내부자 보유율", f"{(insider_pct*100):.2f}%" if insider_pct is not None else "N/A", "heldPercentInsiders")

        st.markdown("#### 기본 정보(원천)")
        st.write({
            "시총": fund.get("시총"),
            "PER": fund.get("PER"),
            "PBR": fund.get("PBR"),
            "ROE": fund.get("ROE"),
            "유통주식수(float)": sm.get("float_shares"),
        })

if __name__ == "__main__":
    main()
