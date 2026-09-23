#!/usr/bin/env python3
"""
MFI + 슈퍼트렌드 스크리너 - Streamlit 대시보드

설치:
    pip install streamlit pykrx yfinance pandas numpy lxml

실행 (반드시 screener.py와 같은 폴더에서):
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import screener as sc

st.set_page_config(page_title="매매 신호 스크리너", page_icon="📈", layout="wide")

st.markdown("""
<style>
.stApp { background-color: #0e1116; }
.block-container { padding-top: 2rem; max-width: 1100px; }
h1, h2, h3 { color: #e7e5df; }
.card {
    background: #161a21; border: 1px solid #262b34; border-radius: 14px;
    padding: 18px 20px; margin-bottom: 14px;
}
.card-title { font-size: 16px; font-weight: 600; color: #e7e5df; margin-bottom: 10px; }
.price-row { display: flex; gap: 24px; margin-bottom: 10px; }
.price-box { flex: 1; }
.price-label { font-size: 11px; color: #8b8f98; margin-bottom: 2px; }
.price-value { font-size: 20px; font-weight: 600; font-variant-numeric: tabular-nums; }
.buy { color: #4fbf8f; }
.sell { color: #e0685f; }
.reason { font-size: 13px; color: #a8acb3; margin-bottom: 10px; line-height: 1.5; }
.winrate-row { display: flex; align-items: center; gap: 8px; font-size: 13px; }
.badge {
    font-size: 11px; padding: 2px 8px; border-radius: 6px; font-weight: 500;
}
.badge-good { background: #1c3a2c; color: #4fbf8f; }
.badge-warn { background: #3a2e1c; color: #e0a84f; }
.badge-low  { background: #241c1c; color: #6b7078; }
</style>
""", unsafe_allow_html=True)

st.title("📈 매매 신호 스크리너")
st.caption("MFI + 슈퍼트렌드 + 거래량 조합 신호 · 종목별 과거 실제 승률 기반")

with st.sidebar:
    st.header("설정")
    market = st.selectbox("시장", ["전체", "국내만", "해외(S&P100)만"])
    top_n = st.slider("스캔할 종목 수 (시총 상위)", 10, 100, 30, step=10)
    st.caption("숫자가 클수록 정확하지만 시간이 오래 걸려요.")
    run = st.button("🔍 스캔 실행", type="primary", use_container_width=True)

    st.divider()
    st.caption(
        "⚠️ 표시되는 승률은 그 종목에 이 정확한 신호 조건이 과거 2년간 "
        "실제로 몇 번(n) 나왔는지를 계산한 것입니다. n이 작으면 신뢰도가 "
        "낮습니다. 투자 조언이 아닙니다."
    )

market_map = {"전체": "all", "국내만": "kr", "해외(S&P100)만": "us"}


@st.cache_data(ttl=3600, show_spinner=False)
def run_scan(market_key, top_n):
    results = []
    progress_log = []
    if market_key in ("kr", "all"):
        results += sc.scan_market("kr", top_n)
    if market_key in ("us", "all"):
        results += sc.scan_market("us", top_n)
    return results


def winrate_badge(n, win_rate):
    if n < 5:
        return '<span class="badge badge-low">표본 매우 적음 (n=%d)</span>' % n
    elif n < 10:
        return '<span class="badge badge-warn">참고용 (n=%d)</span>' % n
    else:
        return '<span class="badge badge-good">n=%d</span>' % n


if run:
    with st.spinner(f"{top_n}개 종목 스캔 중... (몇 분 걸릴 수 있어요)"):
        try:
            results = run_scan(market_map[market], top_n)
            st.session_state["results"] = results
        except Exception as e:
            st.error(f"데이터를 가져오는 중 문제가 발생했습니다: {e}")
            st.info(
                "국내 데이터(KRX)는 클라우드 서버 IP에서 접속이 간헐적으로 막힐 수 있습니다. "
                "'해외(S&P100)만'으로 먼저 테스트해보시거나, 잠시 후 다시 시도해보세요."
            )
            st.stop()

if "results" not in st.session_state:
    st.info("왼쪽에서 조건을 정하고 '스캔 실행'을 눌러주세요.")
    st.stop()

results = st.session_state["results"]
active = [r for r in results if r["active_signal"]]
active.sort(key=lambda r: (r["win_rate"] or 0), reverse=True)

col1, col2, col3 = st.columns(3)
col1.metric("스캔한 종목", len(results))
col2.metric("매수 신호 활성", len(active))
col3.metric("신호 없음", len(results) - len(active))

st.divider()

if not active:
    st.warning("현재 조건에 맞는 매수 신호가 없습니다. 조건을 바꾸거나 나중에 다시 시도해보세요.")
else:
    for r in active:
        cur = r["currency"]
        n = r["n_trades"]
        wr_txt = f"{r['win_rate']:.0f}%" if r["win_rate"] is not None else "N/A"
        avg_txt = f"{r['avg_ret']:+.1f}%" if r["avg_ret"] is not None else "N/A"

        st.markdown(f"""
        <div class="card">
            <div class="card-title">{r['name']} <span style="color:#5c616b; font-weight:400;">({r['ticker']})</span></div>
            <div class="price-row">
                <div class="price-box">
                    <div class="price-label">매수 타점</div>
                    <div class="price-value buy">{r['entry_price']:,.0f}{cur}</div>
                </div>
                <div class="price-box">
                    <div class="price-label">매도 타점 (추정)</div>
                    <div class="price-value sell">{r['target_price']:,.0f}{cur}</div>
                </div>
            </div>
            <div class="reason">
                근거: MFI {r['mfi_now']:.1f} · 슈퍼트렌드 상승 전환 · 거래량 평균 대비 {r['vol_ratio_now']:.1f}배
            </div>
            <div class="winrate-row">
                과거 승률 <b>{wr_txt}</b> (평균 수익률 {avg_txt}) {winrate_badge(n, r['win_rate'])}
            </div>
        </div>
        """, unsafe_allow_html=True)

with st.expander("전체 종목 결과 (신호 없는 것 포함)"):
    df = pd.DataFrame(results)
    if not df.empty:
        display_cols = ["name", "ticker", "active_signal", "n_trades", "win_rate", "avg_ret"]
        df_show = df[display_cols].rename(columns={
            "name": "종목명", "ticker": "티커", "active_signal": "신호중",
            "n_trades": "과거신호횟수", "win_rate": "승률(%)", "avg_ret": "평균수익률(%)"
        })
        st.dataframe(df_show, use_container_width=True, hide_index=True)
