#!/usr/bin/env python3
"""
멀티 전략 스크리너 - MFI단순 / MFI+슈퍼트렌드+거래량 콤보 / 골든크로스 / 거래량돌파
각 종목에 대해 4개 전략을 모두 계산하고, 현재 신호가 걸린 것만 추려서 보여줍니다.
각 전략의 "승률"은 그 종목·그 전략 조건으로 과거 실제 몇 번(n) 발생했는지 계산한 값입니다.

설치:
    pip install pykrx yfinance pandas numpy requests lxml
"""

import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

MFI_PERIOD = 14
ST_PERIOD = 10
ST_MULT = 3.0
VOL_LOOKBACK = 20
MA_SHORT = 50
MA_LONG = 200
BREAKOUT_VOL_MULT = 2.5
BREAKOUT_HOLD_DAYS = 7


# ---------- 지표 계산 ----------

def compute_mfi(df, period=MFI_PERIOD):
    tp = (df["high"] + df["low"] + df["close"]) / 3
    raw_flow = tp * df["volume"]
    delta = tp.diff()
    pos_flow = raw_flow.where(delta > 0, 0.0)
    neg_flow = raw_flow.where(delta < 0, 0.0)
    pos_sum = pos_flow.rolling(period).sum()
    neg_sum = neg_flow.rolling(period).sum()
    mfi = 100 - (100 / (1 + pos_sum / neg_sum.replace(0, np.nan)))
    return mfi.fillna(100)


def compute_supertrend(df, period=ST_PERIOD, mult=ST_MULT):
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    n = len(df)

    hl2 = (high + low) / 2
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr = pd.Series(tr).rolling(period).mean().to_numpy()

    upper = hl2 + mult * atr
    lower = hl2 - mult * atr

    trend = np.ones(n, dtype=np.int64)
    final_upper = upper.copy()
    final_lower = lower.copy()

    for i in range(1, n):
        if close[i - 1] > final_upper[i - 1]:
            final_upper[i] = min(upper[i], final_upper[i - 1]) if trend[i - 1] == -1 else upper[i]
        else:
            final_upper[i] = upper[i]

        if close[i] > final_upper[i]:
            trend[i] = 1
        elif close[i] < final_lower[i]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]

        if trend[i] == 1:
            final_lower[i] = max(lower[i], final_lower[i - 1]) if trend[i - 1] == 1 else lower[i]
        else:
            final_lower[i] = lower[i]

    return pd.Series(trend, index=df.index)


def build_indicators(df):
    df = df.copy()
    df["mfi"] = compute_mfi(df)
    df["st_trend"] = compute_supertrend(df)
    df["vol_ma"] = df["volume"].rolling(VOL_LOOKBACK).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma"]
    df["ma_short"] = df["close"].rolling(MA_SHORT).mean()
    df["ma_long"] = df["close"].rolling(MA_LONG).mean()
    return df


# ---------- 전략 정의 ----------
# 각 전략: entry(df)->bool Series, exit(df)->bool Series 또는 hold_days(고정 보유일)

def _mfi_simple_entry(df):
    return (df["mfi"] < 30) & (df["mfi"].shift(1) >= 30)

def _mfi_simple_exit(df):
    return df["mfi"] > 70


def _combo_entry(df):
    st_flip_up = (df["st_trend"] == 1) & (df["st_trend"].shift(1) == -1)
    return st_flip_up & (df["mfi"] < 45) & (df["vol_ratio"] > 1.5)

def _combo_exit(df):
    st_flip_down = (df["st_trend"] == -1) & (df["st_trend"].shift(1) == 1)
    return st_flip_down | (df["mfi"] > 70)


def _golden_cross_entry(df):
    return (df["ma_short"] > df["ma_long"]) & (df["ma_short"].shift(1) <= df["ma_long"].shift(1))

def _golden_cross_exit(df):
    return (df["ma_short"] < df["ma_long"]) & (df["ma_short"].shift(1) >= df["ma_long"].shift(1))


def _volume_breakout_entry(df):
    return (df["vol_ratio"] > BREAKOUT_VOL_MULT) & (df["close"] > df["close"].shift(1))


STRATEGIES = {
    "mfi_simple": {
        "label": "MFI 단순 과매도",
        "entry": _mfi_simple_entry,
        "exit": _mfi_simple_exit,
        "hold_days": None,
        "reason": lambda row: f"MFI {row['mfi']:.1f} (과매도 진입)",
    },
    "combo": {
        "label": "MFI+슈퍼트렌드+거래량 콤보",
        "entry": _combo_entry,
        "exit": _combo_exit,
        "hold_days": None,
        "reason": lambda row: f"MFI {row['mfi']:.1f} · 슈퍼트렌드 상승전환 · 거래량 {row['vol_ratio']:.1f}배",
    },
    "golden_cross": {
        "label": "이동평균 골든크로스",
        "entry": _golden_cross_entry,
        "exit": _golden_cross_exit,
        "hold_days": None,
        "reason": lambda row: f"{MA_SHORT}일선이 {MA_LONG}일선 상향 돌파",
    },
    "volume_breakout": {
        "label": "거래량 돌파",
        "entry": _volume_breakout_entry,
        "exit": None,
        "hold_days": BREAKOUT_HOLD_DAYS,
        "reason": lambda row: f"거래량 평균 대비 {row['vol_ratio']:.1f}배 급증 + 상승",
    },
}


# ---------- 백테스트 (공통 엔진) ----------

def backtest_strategy(df, strat):
    entry_sig = strat["entry"](df)
    hold_days = strat["hold_days"]
    exit_fn = strat["exit"]

    trades = []
    in_position = False
    entry_price = None
    entry_idx = None

    for i in range(len(df)):
        if not in_position and bool(entry_sig.iloc[i]):
            in_position = True
            entry_price = df["close"].iloc[i]
            entry_idx = i
            continue

        if in_position:
            should_exit = False
            if hold_days is not None:
                should_exit = (i - entry_idx) >= hold_days
            elif exit_fn is not None:
                should_exit = bool(exit_fn(df).iloc[i])

            if should_exit:
                exit_price = df["close"].iloc[i]
                ret = (exit_price - entry_price) / entry_price * 100
                trades.append(ret)
                in_position = False

    open_trade = None
    if in_position:
        open_trade = {"entry_price": entry_price, "entry_idx": entry_idx}

    return trades, open_trade


def summarize_strategy(name, ticker, df, strat_key, currency):
    strat = STRATEGIES[strat_key]
    trades, open_trade = backtest_strategy(df, strat)
    n = len(trades)
    win_rate = (sum(1 for r in trades if r > 0) / n * 100) if n else None
    avg_ret = (sum(trades) / n) if n else None

    result = {
        "name": name, "ticker": ticker, "currency": currency,
        "strategy": strat_key, "strategy_label": strat["label"],
        "n_trades": n, "win_rate": win_rate, "avg_ret": avg_ret,
        "active_signal": open_trade is not None,
    }
    if open_trade:
        entry = open_trade["entry_price"]
        result["entry_price"] = entry
        if avg_ret and avg_ret > 0:
            result["target_price"] = entry * (1 + avg_ret / 100)
        else:
            result["target_price"] = df["high"].tail(60).max()
        last_row = df.iloc[-1]
        result["reason"] = strat["reason"](last_row)
    return result


# ---------- 종목 유니버스 ----------

_SP100_FALLBACK = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "AMZN": "Amazon",
    "GOOGL": "Alphabet A", "GOOG": "Alphabet C", "META": "Meta Platforms",
    "BRK-B": "Berkshire Hathaway", "AVGO": "Broadcom", "TSLA": "Tesla",
    "LLY": "Eli Lilly", "JPM": "JPMorgan Chase", "V": "Visa", "XOM": "Exxon Mobil",
    "UNH": "UnitedHealth", "MA": "Mastercard", "PG": "Procter & Gamble",
    "COST": "Costco", "HD": "Home Depot", "JNJ": "Johnson & Johnson",
    "ABBV": "AbbVie", "NFLX": "Netflix", "CRM": "Salesforce", "BAC": "Bank of America",
    "KO": "Coca-Cola", "MRK": "Merck", "AMD": "AMD", "PEP": "PepsiCo",
    "ADBE": "Adobe", "WMT": "Walmart", "CSCO": "Cisco", "TMO": "Thermo Fisher",
    "MCD": "McDonald's", "ABT": "Abbott", "ACN": "Accenture", "LIN": "Linde",
    "PM": "Philip Morris", "DHR": "Danaher", "WFC": "Wells Fargo", "GE": "GE Aerospace",
    "TXN": "Texas Instruments", "QCOM": "Qualcomm", "IBM": "IBM", "VZ": "Verizon",
    "CAT": "Caterpillar", "AMGN": "Amgen", "INTU": "Intuit", "NOW": "ServiceNow",
    "SPGI": "S&P Global", "AXP": "American Express", "PFE": "Pfizer", "NKE": "Nike",
    "UNP": "Union Pacific", "RTX": "RTX", "LOW": "Lowe's", "GS": "Goldman Sachs",
    "T": "AT&T", "HON": "Honeywell", "BKNG": "Booking Holdings", "COP": "ConocoPhillips",
    "ISRG": "Intuitive Surgical", "MS": "Morgan Stanley", "BLK": "BlackRock",
    "SYK": "Stryker", "ELV": "Elevance Health", "SCHW": "Charles Schwab",
    "MDT": "Medtronic", "C": "Citigroup", "MU": "Micron", "PLD": "Prologis",
    "ADP": "ADP", "CVX": "Chevron", "GILD": "Gilead", "CI": "Cigna",
    "CB": "Chubb", "TJX": "TJX", "MMC": "Marsh McLennan", "SBUX": "Starbucks",
    "AMT": "American Tower", "PGR": "Progressive", "ETN": "Eaton",
    "BSX": "Boston Scientific", "ADI": "Analog Devices", "DE": "Deere",
    "LMT": "Lockheed Martin", "BA": "Boeing", "SO": "Southern Company",
    "MDLZ": "Mondelez", "REGN": "Regeneron", "VRTX": "Vertex Pharma",
    "PANW": "Palo Alto Networks", "ORCL": "Oracle", "DUK": "Duke Energy",
    "TMUS": "T-Mobile", "CMCSA": "Comcast", "FI": "Fiserv", "EOG": "EOG Resources",
    "SHW": "Sherwin-Williams", "ZTS": "Zoetis", "CME": "CME Group",
    "MO": "Altria", "NEE": "NextEra Energy", "APD": "Air Products",
    "PYPL": "PayPal", "TGT": "Target", "USB": "US Bancorp", "KLAC": "KLA Corp",
}

_KOSPI_TOP_FALLBACK = {
    "005930": "삼성전자", "000660": "SK하이닉스", "373220": "LG에너지솔루션",
    "207940": "삼성바이오로직스", "005380": "현대차", "000270": "기아",
    "068270": "셀트리온", "035420": "NAVER", "051910": "LG화학",
    "006400": "삼성SDI", "035720": "카카오", "105560": "KB금융",
    "055550": "신한지주", "012330": "현대모비스", "028260": "삼성물산",
    "066570": "LG전자", "015760": "한국전력", "034730": "SK",
    "018260": "삼성에스디에스", "032830": "삼성생명", "086790": "하나금융지주",
    "010130": "고려아연", "009150": "삼성전기", "259960": "크래프톤",
    "003550": "LG", "017670": "SK텔레콤", "316140": "우리금융지주",
    "030200": "KT", "024110": "기업은행", "090430": "아모레퍼시픽",
    "011070": "LG이노텍", "010950": "S-Oil", "005490": "POSCO홀딩스",
    "000810": "삼성화재", "042700": "한미반도체", "267250": "HD현대중공업",
    "010140": "삼성중공업", "097950": "CJ제일제당", "051900": "LG생활건강",
    "000720": "현대건설", "004020": "현대제철", "006800": "미래에셋증권",
    "023530": "롯데쇼핑", "096770": "SK이노베이션", "302440": "SK바이오사이언스",
    "011200": "HMM", "009540": "HD한국조선해양", "003670": "포스코퓨처엠",
    "047050": "포스코인터내셔널", "128940": "한미약품", "180640": "한진칼",
    "009830": "DL", "010620": "HD현대미포", "251270": "넷마블",
    "352820": "하이브", "091990": "셀트리온헬스케어", "058470": "리노공업",
    "393890": "더존비즈온", "064350": "현대로템", "323410": "카카오뱅크",
    "018880": "한온시스템", "138040": "메리츠금융지주", "004990": "롯데지주",
    "071050": "한국금융지주", "016360": "삼성증권", "000100": "유한양행",
    "139480": "이마트", "078930": "GS", "021240": "코웨이",
    "029780": "삼성카드", "088350": "한화생명", "047810": "한국항공우주",
    "241560": "두산밥캣", "034020": "두산에너빌리티", "336260": "두산퓨얼셀",
    "112610": "씨에스윈드", "298050": "효성첨단소재", "011780": "금호석유",
    "004370": "농심", "001040": "CJ", "271560": "오리온", "008930": "한미사이언스",
}


def get_kr_universe(top_n=100):
    from pykrx import stock
    today = pd.Timestamp.today()
    cap, last_error = None, None
    for back in range(10):
        date_str = (today - pd.Timedelta(days=back)).strftime("%Y%m%d")
        try:
            result = stock.get_market_cap_by_ticker(date_str, market="ALL")
            if result is not None and len(result) > 0:
                cap = result
                break
        except Exception as e:
            last_error = e

    if cap is not None and len(cap) > 0:
        cap = cap.sort_values("시가총액", ascending=False).head(top_n)
        tickers = cap.index.tolist()
        names = {t: stock.get_market_ticker_name(t) for t in tickers}
        return tickers, names

    print(f"  [경고] KRX 실시간 조회 실패, 내장 목록으로 대체: {last_error}")
    tickers = list(_KOSPI_TOP_FALLBACK.keys())[:top_n]
    return tickers, _KOSPI_TOP_FALLBACK


def get_kr_ohlcv(ticker, years=2):
    from pykrx import stock
    end = pd.Timestamp.today().strftime("%Y%m%d")
    start = (pd.Timestamp.today() - pd.Timedelta(days=365 * years)).strftime("%Y%m%d")
    df = stock.get_market_ohlcv(start, end, ticker)
    df = df.rename(columns={"시가": "open", "고가": "high", "저가": "low",
                             "종가": "close", "거래량": "volume"})
    return df[["open", "high", "low", "close", "volume"]]


def get_us_universe():
    import requests
    from io import StringIO
    headers = {"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                               "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")}
    try:
        resp = requests.get("https://en.wikipedia.org/wiki/S%26P_100", headers=headers, timeout=10)
        resp.raise_for_status()
        tables = pd.read_html(StringIO(resp.text))
        for t in tables:
            if "Symbol" in t.columns:
                tickers = t["Symbol"].str.replace(".", "-", regex=False).tolist()
                name_col = "Name" if "Name" in t.columns else "Company"
                names = dict(zip(tickers, t[name_col])) if name_col in t.columns else {tk: tk for tk in tickers}
                return tickers, names
    except Exception as e:
        print(f"  [경고] 위키피디아 조회 실패, 내장 목록으로 대체: {e}")
    return list(_SP100_FALLBACK.keys()), _SP100_FALLBACK


def get_us_ohlcv(ticker, years=2):
    import yfinance as yf
    import requests
    session = requests.Session()
    session.headers.update({"User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                                            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")})
    try:
        df = yf.download(ticker, period=f"{years}y", interval="1d", progress=False, session=session)
    except TypeError:
        df = yf.download(ticker, period=f"{years}y", interval="1d", progress=False)
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                             "Close": "close", "Volume": "volume"})
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    return df[["open", "high", "low", "close", "volume"]]


# ---------- 실행 ----------

def _process_one(ticker, name, fetch, strategy_keys, currency):
    df = fetch(ticker)
    if len(df) < max(MA_LONG, 80):
        return []
    df = build_indicators(df)
    return [summarize_strategy(name, ticker, df, sk, currency) for sk in strategy_keys]


def scan_market(market, top_n, strategy_keys=None, progress_callback=None, max_workers=8):
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if strategy_keys is None:
        strategy_keys = list(STRATEGIES.keys())

    if market == "kr":
        tickers, names = get_kr_universe(top_n)
        fetch = get_kr_ohlcv
        currency = "원"
    else:
        tickers, names = get_us_universe()
        tickers = tickers[:top_n]
        fetch = get_us_ohlcv
        currency = "$"

    results = []
    fetch_fail = 0
    total = len(tickers)
    done = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        future_map = {
            ex.submit(_process_one, t, names.get(t, t), fetch, strategy_keys, currency): t
            for t in tickers
        }
        for fut in as_completed(future_map):
            t = future_map[fut]
            done += 1
            try:
                results.extend(fut.result())
            except Exception as e:
                fetch_fail += 1
                print(f"  [스킵] {t}: {e}")
            if progress_callback:
                progress_callback(done, total)

    if fetch_fail == len(tickers) and len(tickers) > 0:
        raise RuntimeError(
            f"{market.upper()} 시세를 {len(tickers)}개 종목 모두 가져오지 못했습니다. "
            "데이터 제공처가 서버 IP를 일시적으로 막고 있을 수 있습니다."
        )
    return results


def print_report(results):
    active = [r for r in results if r["active_signal"]]
    active.sort(key=lambda r: (r["win_rate"] or 0), reverse=True)

    print(f"\n현재 신호 활성: {len(active)}건 (종목×전략 조합 기준)")
    for r in active:
        cur = r["currency"]
        wr = f"{r['win_rate']:.0f}%" if r["win_rate"] is not None else "N/A"
        print(f"\n[{r['strategy_label']}] {r['name']} ({r['ticker']})")
        print(f"  매수: {r['entry_price']:,.0f}{cur} / 매도: {r['target_price']:,.0f}{cur}")
        print(f"  근거: {r['reason']}")
        print(f"  과거 승률: {wr} (n={r['n_trades']})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=["kr", "us", "all"], default="all")
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    all_results = []
    if args.market in ("kr", "all"):
        all_results += scan_market("kr", args.top)
    if args.market in ("us", "all"):
        all_results += scan_market("us", args.top)
    print_report(all_results)
