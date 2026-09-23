#!/usr/bin/env python3
"""
멀티 전략 스크리너 - Streamlit 대시보드
실행: streamlit run app.py  (screener.py와 같은 폴더에서)
"""

import streamlit as st
import pandas as pd
import time as _time
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
.strategy-badge {
    display: inline-block; font-size: 11px; padding: 3px 9px; border-radius: 6px;
    background: #1c2432; color: #6ea8f0; font-weight: 500; margin-bottom: 8px;
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
.badge { font-size: 11px; padding: 2px 8px; border-radius: 6px; font-weight: 500; }
.badge-good { background: #1c3a2c; color: #4fbf8f; }
.badge-warn { background: #3a2e1c; color: #e0a84f; }
.badge-low  { background: #241c1c; color: #6b7078; }
.multi-badge {
    display: inline-block; font-size: 12px; font-weight: 600; padding: 4px 10px;
    border-radius: 6px; background: #2a3a1c; color: #a8d94f; margin-bottom: 6px;
}
.strategy-block {
    border-top: 1px solid #21252d; margin-top: 12px; padding-top: 12px;
}
.strategy-block:first-of-type { border-top: none; margin-top: 4px; padding-top: 4px; }
.strategy-block-label { font-size: 12px; color: #6ea8f0; font-weight: 500; margin-bottom: 6px; }
.wr-high { color: #4fbf8f; }
.wr-mid  { color: #e0a84f; }
.wr-low  { color: #e0685f; }
</style>
""", unsafe_allow_html=True)

st.title("📈 매매 신호 스크리너")
st.caption("MFI 단순 · 콤보 · 골든크로스 · 거래량돌파 4개 전략 동시 스캔 · 종목별 과거 실제 승률 기반")

STRATEGY_OPTIONS = {k: v["label"] for k, v in sc.STRATEGIES.items()}

with st.sidebar:
    st.header("설정")
    market = st.selectbox("시장", ["전체", "국내만", "해외(S&P100)만"])
    top_n = st.slider("스캔할 종목 수 (시총 상위)", 10, 100, 30, step=10)
    st.caption("숫자가 클수록 정확하지만 시간이 오래 걸려요.")

    selected_strats = st.multiselect(
        "확인할 전략",
        options=list(STRATEGY_OPTIONS.keys()),
        default=list(STRATEGY_OPTIONS.keys()),
        format_func=lambda k: STRATEGY_OPTIONS[k],
    )

    run = st.button("🔍 스캔 실행", type="primary", use_container_width=True)
    force_refresh = st.checkbox("최신 데이터로 새로고침 (캐시 무시)", value=False)

    st.divider()
    st.caption(
        "⚠️ 표시되는 승률은 그 종목·그 전략 조건이 과거 2년간 실제로 몇 번(n) "
        "나왔는지를 계산한 것입니다. n이 작으면 신뢰도가 낮습니다. 투자 조언이 아닙니다."
    )

market_map = {"전체": "all", "국내만": "kr", "해외(S&P100)만": "us"}


def run_scan_with_progress(market_key, top_n, strat_keys):
    if "_df_cache" not in st.session_state:
        st.session_state["_df_cache"] = {}
    df_cache = st.session_state["_df_cache"]

    progress_bar = st.progress(0, text="시작 중...")
    markets = [m for m in (["kr"] if market_key in ("kr", "all") else []) +
               (["us"] if market_key in ("us", "all") else [])]
    total_markets = len(markets)
    results = []

    for mi, m in enumerate(markets):
        label = "국내" if m == "kr" else "해외(S&P100)"

        def cb(done, total, m=m, mi=mi):
            frac = (mi + done / max(total, 1)) / total_markets
            progress_bar.progress(min(frac, 1.0), text=f"{label} 스캔 중... ({done}/{total})")

        results += sc.scan_market(m, top_n, strat_keys, progress_callback=cb, df_cache=df_cache)

    progress_bar.progress(1.0, text="완료")
    progress_bar.empty()
    return results


def winrate_badge(n):
    if n < 5:
        return f'<span class="badge badge-low">과거 신호 {n}번뿐 — 통계적 의미 없음</span>'
    elif n < 10:
        return f'<span class="badge badge-warn">과거 신호 {n}번 — 표본 부족, 신뢰하기 이름</span>'
    return f'<span class="badge badge-good">과거 신호 {n}번 확인됨</span>'


def winrate_color_class(win_rate):
    if win_rate is None:
        return "wr-mid"
    if win_rate >= 80:
        return "wr-high"
    elif win_rate >= 50:
        return "wr-mid"
    return "wr-low"


if run:
    if not selected_strats:
        st.warning("최소 1개 이상의 전략을 선택하세요.")
        st.stop()
    if force_refresh:
        st.session_state["_df_cache"] = {}
        st.session_state["_scan_cache"] = {}
    with st.spinner(f"{top_n}개 종목 × {len(selected_strats)}개 전략 스캔 중... (몇 분 걸릴 수 있어요)"):
        try:
            cache_key = (market_map[market], top_n, tuple(selected_strats))
            if not force_refresh and cache_key in st.session_state.get("_scan_cache", {}):
                results = st.session_state["_scan_cache"][cache_key]
            else:
                results = run_scan_with_progress(*cache_key)
                st.session_state.setdefault("_scan_cache", {})[cache_key] = results
            st.session_state["results"] = results
            st.session_state["scan_time"] = _time.strftime("%Y-%m-%d %H:%M:%S")
        except Exception as e:
            st.error(f"데이터를 가져오는 중 문제가 발생했습니다: {e}")
            st.info("국내 데이터가 계속 실패하면 '해외(S&P100)만'으로 먼저 테스트해보세요.")
            st.stop()

if "results" not in st.session_state:
    st.info("왼쪽에서 조건을 정하고 '스캔 실행'을 눌러주세요.")
    st.stop()

results = st.session_state["results"]

if "scan_time" in st.session_state:
    st.caption(
        f"📅 데이터 기준 시각: {st.session_state['scan_time']} "
        "(이 시각의 종가 기준이며, 실시간 시세가 아닙니다. 같은 세션에서 전략만 바꾸면 "
        "이 시각 데이터를 재사용합니다 — 최신으로 갱신하려면 '최신 데이터로 새로고침' 체크)"
    )

# 같은 종목에 여러 전략이 동시에 신호를 낸 경우 하나로 묶기
active = [r for r in results if r["active_signal"]]
grouped = {}
for r in active:
    key = (r["ticker"], r["currency"])
    grouped.setdefault(key, []).append(r)

groups = list(grouped.values())

col1, col2, col3, col4 = st.columns(4)
col1.metric("스캔한 종목×전략", len(results))
col2.metric("신호 활성 종목", len(groups))
multi_count = sum(1 for g in groups if len(g) > 1)
col3.metric("2개 이상 전략 동시신호", multi_count)
col4.metric("신호 없음", len(results) - len(active))

st.divider()

if not groups:
    st.warning("현재 조건에 맞는 신호가 없습니다. 전략을 더 선택하거나 종목 수를 늘려보세요.")
else:
    sort_option = st.selectbox(
        "정렬 기준",
        ["동시신호 많은순 → 승률순", "승률 높은순", "평균 수익률순", "종목명"],
        index=0,
    )

    def group_best_winrate(g):
        rates = [x["win_rate"] for x in g if x["win_rate"] is not None]
        return max(rates) if rates else 0

    def group_best_avgret(g):
        rets = [x["avg_ret"] for x in g if x["avg_ret"] is not None]
        return max(rets) if rets else -999

    if sort_option == "동시신호 많은순 → 승률순":
        groups.sort(key=lambda g: (len(g), group_best_winrate(g)), reverse=True)
    elif sort_option == "승률 높은순":
        groups.sort(key=group_best_winrate, reverse=True)
    elif sort_option == "평균 수익률순":
        groups.sort(key=group_best_avgret, reverse=True)
    else:
        groups.sort(key=lambda g: g[0]["name"])

    for g in groups:
        r0 = g[0]
        cur = r0["currency"]
        multi = len(g) > 1

        badges_html = "".join(
            f'<span class="strategy-badge">{x["strategy_label"]}</span> ' for x in g
        )
        header = (
            f'<span class="multi-badge">⚡ {len(g)}개 전략 동시 신호</span><br>' if multi else ""
        )

        st.markdown(f"""
        <div class="card">
            {header}
            {badges_html}
            <div class="card-title" style="margin-top:8px;">{r0['name']} <span style="color:#5c616b; font-weight:400;">({r0['ticker']})</span></div>
        """, unsafe_allow_html=True)

        for x in g:
            n = x["n_trades"]
            wr_txt = f"{x['win_rate']:.0f}%" if x["win_rate"] is not None else "N/A"
            avg_txt = f"{x['avg_ret']:+.1f}%" if x["avg_ret"] is not None else "N/A"
            wr_class = winrate_color_class(x["win_rate"])

            st.markdown(f"""
            <div class="strategy-block">
                <div class="strategy-block-label">{x['strategy_label']}</div>
                <div class="price-row">
                    <div class="price-box">
                        <div class="price-label">매수 타점</div>
                        <div class="price-value buy">{x['entry_price']:,.0f}{cur}</div>
                    </div>
                    <div class="price-box">
                        <div class="price-label">매도 타점 (추정)</div>
                        <div class="price-value sell">{x['target_price']:,.0f}{cur}</div>
                    </div>
                </div>
                <div class="reason">근거: {x['reason']}</div>
                <div class="winrate-row">
                    과거 승률 <b class="{wr_class}">{wr_txt}</b> (평균 수익률 {avg_txt}) {winrate_badge(n)}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

with st.expander("전체 결과 (신호 없는 것 포함)"):
    df = pd.DataFrame(results)
    if not df.empty:
        display_cols = ["name", "ticker", "strategy_label", "active_signal", "n_trades", "win_rate", "avg_ret"]
        df_show = df[display_cols].rename(columns={
            "name": "종목명", "ticker": "티커", "strategy_label": "전략",
            "active_signal": "신호중", "n_trades": "과거신호횟수",
            "win_rate": "승률(%)", "avg_ret": "평균수익률(%)"
        })
        st.dataframe(df_show, use_container_width=True, hide_index=True)
