import streamlit as st
import pandas as pd

from data_engine import get_daily_ohlcv, get_info, liquidity_ok
from indicators import add_indicators, add_quantile_thresholds
from regime import add_market_regime, regime_summary
from strategy_swing import swing_signals, rule_reco
from backtest import backtest, walk_forward_oos
from validation import trust_gate
from news_engine import get_google_news_rss, translate_ko
from ui_components import inject_style, header, banner, summary_card, kv_card, score_from_oos

st.set_page_config(page_title="AI Stock Sniper (Swing Final)", page_icon="📈", layout="wide")

def main():
    inject_style()

    with st.sidebar:
        st.subheader("스윙 분석(최종)")
        ticker = st.text_input("티커", value="NVDA").upper().strip()
        period = st.selectbox("데이터 기간", ["5y", "3y"], index=0)

        st.markdown("---")
        st.caption("거래 비용(보수적으로 잡는 게 안전)")
        fee_bps = st.slider("수수료(왕복) bps", 0, 30, 5, 1)
        slip_bps = st.slider("슬리피지 bps", 0, 120, 10, 1)

        st.markdown("---")
        st.caption("백테스트 리스크 파라미터(스윙)")
        stop_atr = st.slider("손절(ATR 배수)", 1.0, 4.0, 2.0, 0.5)
        take_atr = st.slider("익절(ATR 배수)", 2.0, 8.0, 4.0, 0.5)
        max_hold = st.slider("최대 보유일(타임스탑)", 10, 80, 30, 5)

        st.markdown("---")
        st.caption("워크포워드(OOS) 설정")
        train_years = st.selectbox("학습 구간", ["3y", "2y"], index=0)
        test_months = st.selectbox("테스트 구간", ["6m", "3m"], index=0)

        st.markdown("---")
        st.caption("신뢰도 게이트(OOS 기준)")
        min_trades = st.slider("최소 OOS 트레이드 수", 20, 120, 40, 5)
        min_pf = st.slider("최소 PF(중앙값)", 1.0, 2.0, 1.2, 0.1)
        min_win = st.slider("최소 승률(가중, %)", 35, 65, 45, 1)

        run = st.button("분석 실행")

    if not run:
        st.info("왼쪽에서 티커 설정 후 ‘분석 실행’을 누르세요.")
        return

    with st.spinner("데이터/레짐/지표/백테스트/워크포워드 계산 중..."):
        info = get_info(ticker)

        df = get_daily_ohlcv(ticker, period=period)
        if df is None or df.empty or len(df) < 260:
            st.error("데이터가 부족하거나 불러오지 못했습니다.")
            return

        # 시장 레짐 데이터
        spy = get_daily_ohlcv("SPY", period=period)
        vix = get_daily_ohlcv("^VIX", period=period)

        # 지표
        df = add_indicators(df)
        df = add_quantile_thresholds(df, lookback=504 if period == "5y" else 252)

        # 레짐 결합 + 시그널 생성
        df = add_market_regime(df, spy, vix)
        df = swing_signals(df)

        # 유동성
        liq_pass = liquidity_ok(df, min_dollar_vol_20d=2_000_000)

        # 백테스트(전체)
        bt_df, trades, stats = backtest(
            df,
            fee_bps=fee_bps,
            slippage_bps=slip_bps,
            stop_atr=stop_atr,
            take_atr=take_atr,
            max_hold_days=max_hold
        )

        # 워크포워드(OOS)
        train_days = 252*3 if train_years == "3y" else 252*2
        test_days = 252//2 if test_months == "6m" else 63
        wf_df, oos = walk_forward_oos(
            df,
            train_days=train_days,
            test_days=test_days,
            step_days=63,
            fee_bps=fee_bps,
            slippage_bps=slip_bps,
            stop_atr=stop_atr,
            take_atr=take_atr,
            max_hold_days=max_hold
        )

        # 신뢰도 게이트(핵심)
        gate = trust_gate(
            liquidity_pass=liq_pass,
            oos=oos,
            min_total_trades=min_trades,
            min_pf_median=min_pf,
            min_winrate=float(min_win),
            max_zero_windows=2
        )

        latest = df.iloc[-1]
        reco, reason = rule_reco(latest, gate)

        score, winprob = score_from_oos(oos)

        # 레짐 요약
        reg_state, reg_msg = regime_summary(latest)

        # 보조 지표(공매도/내부자)
        insider = info.get("heldPercentInsiders", None)
        short_float = info.get("shortPercentOfFloat", None)
        dtc = info.get("shortRatio", None)

        name = info.get("longName", ticker)

    # 헤더/배너
    header(ticker, name)
    banner(reco, reason, score=score, winprob=winprob)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📌 요약", "🧪 검증(OOS/성과)", "🧾 트레이드/실패분석", "📰 뉴스(번역)", "ℹ️ 지표/데이터"
    ])

    with tab1:
        summary_card("현재 결론(핵심만)", [
            f"추천: {reco} (규칙 기반, OOS 게이트 적용)",
            f"레짐: {reg_state} - {reg_msg}",
            f"유동성: {'통과' if liq_pass else '미달(스윙 위험)'}",
            f"OOS 승률(가중): {winprob:.1f}% / PF(중앙값): {oos.get('PF(중앙값)', '-')}",
            f"오늘 ENTRY 조건: {'충족' if bool(latest.get('ENTRY', False)) else '미충족'}",
        ])

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("현재가", f"${float(latest['Close']):.2f}")
        c2.metric("RSI", f"{float(latest['RSI']):.1f}")
        c3.metric("MA200", f"{float(latest['MA200']):.2f}")
        c4.metric("VIX", "-" if pd.isna(latest.get("VIX_Close")) else f"{float(latest['VIX_Close']):.2f}")

        kv_card("공매도/내부자(가능할 때만)", [
            ("내부자 보유율", "-" if insider is None else f"{float(insider)*100:.2f}%"),
            ("공매도 비중(Float)", "-" if short_float is None else f"{float(short_float)*100:.2f}%"),
            ("DTC(일)", "-" if dtc is None else f"{float(dtc):.2f}"),
        ])

        st.caption("주의: 공매도/내부자 데이터는 소스 지연/누락이 잦습니다. ‘-’는 ‘없다’가 아니라 ‘못 가져왔다’일 수 있습니다.")

    with tab2:
        summary_card("이 탭을 봐야 ‘확실한 척’ 안 하게 됨", [
            "OOS(워크포워드)에서 구간별 성과가 흔들리면, 매수 신호는 신뢰하면 안 됩니다.",
            "특히 ‘최악 구간 PF’가 낮으면 장세가 바뀔 때 전략이 무너진다는 뜻입니다.",
        ])

        st.markdown("### 전체 백테스트(참고)")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("트레이드 수", stats.get("트레이드수", 0))
        c2.metric("승률", f"{stats.get('승률', 0):.1f}%")
        c3.metric("PF", f"{stats.get('PF', 0):.2f}")
        c4.metric("CAGR(근사)", f"{stats.get('CAGR(근사)', 0):.1%}")
        c5.metric("MDD", f"{stats.get('MDD', 0):.1%}")

        st.markdown("### 워크포워드(OOS) 결과(핵심)")
        if oos and oos.get("ok", False):
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("총 트레이드", oos.get("총트레이드수", 0))
            c2.metric("가중 승률", f"{oos.get('가중승률(%)', 0):.1f}%")
            c3.metric("PF(중앙값)", oos.get("PF(중앙값)", 0))
            c4.metric("최악 PF", oos.get("최악PF", 0))
            c5.metric("무거래 구간", oos.get("무거래구간수", 0))

            st.markdown("#### 구간별 OOS 성과표")
            st.dataframe(wf_df, use_container_width=True)
        else:
            st.warning(f"OOS 결과를 만들지 못했습니다: {oos.get('reason','') if isinstance(oos, dict) else ''}")

        st.markdown("#### 에퀴티 커브(참고)")
        if "EQUITY" in bt_df.columns:
            st.line_chart(bt_df["EQUITY"].dropna())

    with tab3:
        summary_card("실패 분석이 없으면 ‘전략 개선’이 아니라 ‘희망회로’가 됨", [
            "손절/추세훼손/타임스탑 중 어떤 이유가 많은지 보고, 룰을 바꿀지 말지 판단해야 합니다.",
            "손실 상위 트레이드가 특정 레짐(RISK_OFF)에서 몰리면, 레짐 필터를 더 강하게 해야 합니다.",
        ])

        if trades:
            tdf = pd.DataFrame(trades).sort_values("진입일", ascending=False)
            st.dataframe(tdf, use_container_width=True)

            st.markdown("#### 청산 사유 분포")
            reason_counts = tdf["사유"].value_counts()
            st.bar_chart(reason_counts)

            st.markdown("#### 손실 상위 10개")
            worst = tdf.sort_values("수익률").head(10)
            st.dataframe(worst, use_container_width=True)
        else:
            st.warning("트레이드가 없습니다(신호가 거의 없거나 데이터/게이트 조건이 너무 빡빡할 수 있음).")

    with tab4:
        summary_card("뉴스는 ‘근거’가 아니라 ‘상황 파악’", [
            "번역은 뉘앙스/법률 용어 오역이 있을 수 있습니다.",
            "매수/매도는 전략 룰과 OOS 게이트 기준으로만 결정하세요(뉴스로 흔들리면 손익이 무너짐).",
        ])
        items = translate_ko(get_google_news_rss(ticker, limit=10))
        if not items:
            st.warning("뉴스를 불러오지 못했습니다.")
        else:
            for it in items:
                st.markdown(f"- [{it['title']}]({it['url']})")

    with tab5:
        reg_state, reg_msg = regime_summary(latest)
        kv_card("레짐/지표 상태(오늘 기준)", [
            ("레짐", f"{reg_state} - {reg_msg}"),
            ("RISK_OFF", "True" if bool(latest.get("RISK_OFF", True)) else "False"),
            ("ENTRY 조건", "True" if bool(latest.get("ENTRY", False)) else "False"),
            ("추세(Close>MA200)", "True" if float(latest["Close"]) > float(latest["MA200"]) else "False"),
            ("눌림(RSI<=Q40)", "True" if float(latest["RSI"]) <= float(latest.get("RSI_Q40", 0)) else "False"),
            ("거래량(VOL_RATIO>=Q80)", "True" if float(latest["VOL_RATIO"]) >= float(latest.get("VOL_Q80", 999)) else "False"),
        ])

        st.caption("‘확실’은 UI가 아니라 OOS(워크포워드)와 레짐 내성에서 나옵니다. 이 앱은 그걸 강제로 보이게 만드는 구조입니다.")

if __name__ == "__main__":
    main()
