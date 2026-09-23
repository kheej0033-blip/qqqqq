#!/usr/bin/env python3
"""
MFI + 슈퍼트렌드 + 거래량 조합 스크리너 & 종목별 과거 승률 백테스트

설치:
    pip install pykrx yfinance pandas numpy requests lxml

실행:
    python screener.py                 # 국내 상위100 + S&P100 전체 스캔
    python screener.py --market kr     # 국내만
    python screener.py --market us     # 해외만
    python screener.py --top 30        # 시총 상위 30개만 (테스트용, 빠름)

중요:
  - 이 스크립트가 보여주는 "승률"은 그 종목에 그 정확한 신호 조건이 과거에
    몇 번 나왔고 결과가 어땠는지를 계산한 것입니다. 표본(n)이 작으면
    (특히 n<10) 통계적으로 거의 의미가 없습니다. 반드시 n을 같이 확인하세요.
  - 과거 승률이 좋았다고 미래도 같다는 보장은 없습니다. 투자 조언이 아닙니다.
  - 상위 100 종목 리스트는 "현재 시점 기준"이라 과거로 갈수록 생존편향
    (지금 잘나가는 종목만 보게 되는 편향)이 있을 수 있습니다.
"""

import argparse
import time
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

MFI_PERIOD = 14
MFI_OVERSOLD = 30
MFI_OVERBOUGHT = 70
ST_PERIOD = 10
ST_MULT = 3.0
VOL_LOOKBACK = 20
VOL_RATIO_MIN = 1.5   # 평균 거래량의 1.5배 이상


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
    hl2 = (df["high"] + df["low"]) / 2
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - df["close"].shift()).abs(),
        (df["low"] - df["close"].shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    upper = hl2 + mult * atr
    lower = hl2 - mult * atr

    trend = pd.Series(index=df.index, dtype="int64")
    final_upper = upper.copy()
    final_lower = lower.copy()
    trend.iloc[0] = 1

    for i in range(1, len(df)):
        if df["close"].iloc[i - 1] > final_upper.iloc[i - 1]:
            final_upper.iloc[i] = min(upper.iloc[i], final_upper.iloc[i - 1]) if trend.iloc[i - 1] == -1 else upper.iloc[i]
        else:
            final_upper.iloc[i] = upper.iloc[i]

        if df["close"].iloc[i] > final_upper.iloc[i]:
            trend.iloc[i] = 1
        elif df["close"].iloc[i] < final_lower.iloc[i]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i - 1]

        if trend.iloc[i] == 1:
            final_lower.iloc[i] = max(lower.iloc[i], final_lower.iloc[i - 1]) if trend.iloc[i - 1] == 1 else lower.iloc[i]
        else:
            final_lower.iloc[i] = lower.iloc[i]

    supertrend_line = np.where(trend == 1, final_lower, final_upper)
    return pd.Series(supertrend_line, index=df.index), trend


def build_signals(df):
    df = df.copy()
    df["mfi"] = compute_mfi(df)
    st_line, st_trend = compute_supertrend(df)
    df["st_trend"] = st_trend
    df["vol_ma"] = df["volume"].rolling(VOL_LOOKBACK).mean()
    df["vol_ratio"] = df["volume"] / df["vol_ma"]

    st_flip_up = (df["st_trend"] == 1) & (df["st_trend"].shift(1) == -1)
    st_flip_down = (df["st_trend"] == 1).shift(1).fillna(False) & (df["st_trend"] == -1)

    df["buy_signal"] = st_flip_up & (df["mfi"] < MFI_OVERSOLD + 15) & (df["vol_ratio"] > VOL_RATIO_MIN)
    df["sell_signal"] = st_flip_down | (df["mfi"] > MFI_OVERBOUGHT)
    return df


def backtest_ticker(df):
    """buy_signal 이후 다음 sell_signal까지의 수익률을 전부 모아 승률/평균수익률 계산."""
    trades = []
    in_position = False
    entry_price = None
    entry_date = None

    for i in range(len(df)):
        row = df.iloc[i]
        if not in_position and row["buy_signal"]:
            in_position = True
            entry_price = row["close"]
            entry_date = df.index[i]
        elif in_position and row["sell_signal"]:
            exit_price = row["close"]
            ret = (exit_price - entry_price) / entry_price * 100
            trades.append({"entry_date": entry_date, "exit_date": df.index[i],
                            "entry": entry_price, "exit": exit_price, "ret": ret})
            in_position = False

    open_trade = None
    if in_position:
        open_trade = {"entry_date": entry_date, "entry": entry_price}

    return trades, open_trade


def summarize(name, ticker, df, currency="원"):
    trades, open_trade = backtest_ticker(df)
    n = len(trades)
    if n == 0:
        win_rate, avg_ret = None, None
    else:
        wins = sum(1 for t in trades if t["ret"] > 0)
        win_rate = wins / n * 100
        avg_ret = sum(t["ret"] for t in trades) / n

    result = {
        "name": name, "ticker": ticker, "n_trades": n,
        "win_rate": win_rate, "avg_ret": avg_ret,
        "active_signal": open_trade is not None,
    }

    if open_trade:
        entry = open_trade["entry"]
        result["entry_price"] = entry
        if avg_ret and avg_ret > 0:
            result["target_price"] = entry * (1 + avg_ret / 100)
        else:
            recent_high = df["high"].tail(60).max()
            result["target_price"] = recent_high
        result["mfi_now"] = df["mfi"].iloc[-1]
        result["vol_ratio_now"] = df["vol_ratio"].iloc[-1]

    return result


# ---------- 데이터 소스 ----------

def get_kr_universe(top_n=100):
    from pykrx import stock
    today = pd.Timestamp.today()
    cap = None
    last_error = None
    for back in range(10):  # 최근 영업일 찾기
        date_str = (today - pd.Timedelta(days=back)).strftime("%Y%m%d")
        try:
            result = stock.get_market_cap_by_ticker(date_str, market="ALL")
            if result is not None and len(result) > 0:
                cap = result
                break
        except Exception as e:
            last_error = e
            continue

    if cap is None or len(cap) == 0:
        raise RuntimeError(
            "KRX 시가총액 데이터를 가져오지 못했습니다. "
            "(Streamlit Cloud 서버에서 KRX 접속이 막혔거나 일시적으로 응답이 없을 수 있습니다. "
            f"마지막 에러: {last_error})"
        )

    cap = cap.sort_values("시가총액", ascending=False).head(top_n)
    tickers = cap.index.tolist()
    names = {t: stock.get_market_ticker_name(t) for t in tickers}
    return tickers, names


def get_kr_ohlcv(ticker, years=2):
    from pykrx import stock
    end = pd.Timestamp.today().strftime("%Y%m%d")
    start = (pd.Timestamp.today() - pd.Timedelta(days=365 * years)).strftime("%Y%m%d")
    df = stock.get_market_ohlcv(start, end, ticker)
    df = df.rename(columns={"시가": "open", "고가": "high", "저가": "low",
                             "종가": "close", "거래량": "volume"})
    return df[["open", "high", "low", "close", "volume"]]


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


def get_us_universe():
    import requests
    from io import StringIO
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")
    }
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
        print(f"  [경고] 위키피디아에서 S&P100 목록을 못 가져와 내장 목록으로 대체합니다: {e}")

    return list(_SP100_FALLBACK.keys()), _SP100_FALLBACK


def get_us_ohlcv(ticker, years=2):
    import yfinance as yf
    import requests
    session = requests.Session()
    session.headers.update({
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36")
    })
    try:
        df = yf.download(ticker, period=f"{years}y", interval="1d", progress=False, session=session)
    except TypeError:
        # 일부 yfinance 버전은 session 인자를 지원하지 않음
        df = yf.download(ticker, period=f"{years}y", interval="1d", progress=False)
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low",
                             "Close": "close", "Volume": "volume"})
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    return df[["open", "high", "low", "close", "volume"]]


# ---------- 실행 ----------

def scan_market(market, top_n):
    results = []
    if market == "kr":
        tickers, names = get_kr_universe(top_n)
        fetch = get_kr_ohlcv
        currency = "원"
    else:
        tickers, names = get_us_universe()
        tickers = tickers[:top_n]
        fetch = get_us_ohlcv
        currency = "$"

    fetch_fail = 0
    for i, t in enumerate(tickers):
        try:
            df = fetch(t)
            if len(df) < 80:
                continue
            df = build_signals(df)
            res = summarize(names.get(t, t), t, df, currency)
            res["currency"] = currency
            results.append(res)
        except Exception as e:
            fetch_fail += 1
            print(f"  [스킵] {t}: {e}")
        time.sleep(0.05)
        if (i + 1) % 20 == 0:
            print(f"  ...{i+1}/{len(tickers)} 처리 중")

    if fetch_fail == len(tickers) and len(tickers) > 0:
        raise RuntimeError(
            f"{market.upper()} 시세를 {len(tickers)}개 종목 모두 가져오지 못했습니다. "
            "데이터 제공처(Yahoo Finance/KRX)가 서버 IP를 일시적으로 막고 있을 수 있습니다."
        )

    return results


def print_report(results):
    active = [r for r in results if r["active_signal"]]
    active.sort(key=lambda r: (r["win_rate"] or 0), reverse=True)

    print("\n" + "=" * 70)
    print(f"현재 매수 신호 활성 종목: {len(active)}개")
    print("=" * 70)

    for r in active:
        cur = r["currency"]
        n = r["n_trades"]
        wr = f"{r['win_rate']:.0f}%" if r["win_rate"] is not None else "N/A"
        avg = f"{r['avg_ret']:+.1f}%" if r["avg_ret"] is not None else "N/A"
        reliability = "⚠️ 표본 부족" if n < 10 else ""

        print(f"\n종목: {r['name']} ({r['ticker']})")
        print(f"  매수 타점: {r['entry_price']:,.0f}{cur}")
        print(f"  매도 타점: {r['target_price']:,.0f}{cur}  (과거 평균수익률 기반 추정)")
        print(f"  근거: MFI {r['mfi_now']:.1f} (슈퍼트렌드 상승전환 + 거래량 {r['vol_ratio_now']:.1f}배)")
        print(f"  과거 승률: {wr} (n={n}건, 평균수익률 {avg}) {reliability}")

    skipped_no_signal = len(results) - len(active)
    print(f"\n(신호 없는 종목 {skipped_no_signal}개는 생략)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--market", choices=["kr", "us", "all"], default="all")
    parser.add_argument("--top", type=int, default=100)
    args = parser.parse_args()

    all_results = []
    if args.market in ("kr", "all"):
        print("국내 상위 종목 스캔 중...")
        all_results += scan_market("kr", args.top)
    if args.market in ("us", "all"):
        print("S&P100 스캔 중...")
        all_results += scan_market("us", args.top)

    print_report(all_results)


if __name__ == "__main__":
    main()
