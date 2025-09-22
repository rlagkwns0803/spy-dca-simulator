# qqq_best_dates.py

import argparse
import calendar
import math
from typing import List, Tuple
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

# 숫자 출력 형식 지정 (쉼표 + 소수점 2자리)
pd.set_option("display.float_format", "{:,.2f}".format)

# ----------------------------
# 데이터 가져오기 (무조건 Series 반환)
# ----------------------------
def fetch_prices(ticker: str, start: str, end: str) -> pd.Series:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"No data for {ticker} between {start} and {end}")
    if "Adj Close" in df.columns:
        ser = df["Adj Close"].copy()
    elif "Close" in df.columns:
        ser = df["Close"].copy()
    else:
        ser = df.iloc[:, 0].copy()  # 안전장치
    ser.name = ticker
    return ser.sort_index()

# ----------------------------
# 윤년 판정
# ----------------------------
def is_leap_year(y: int) -> bool:
    return calendar.isleap(y)

# ----------------------------
# 다음 거래일 찾기
# ----------------------------
def get_next_trading_day_index(index: pd.DatetimeIndex, dt: pd.Timestamp):
    pos = index.searchsorted(dt)
    if pos >= len(index):
        return None
    return pos

# ----------------------------
# 특정 날짜(월/일) 투자 시뮬레이션
# ----------------------------
def simulate_fixed_dates(prices: pd.Series, start_year: int, end_year: int,
                         buy_dates: List[Tuple[int, int]], annual_investment: float) -> dict:
    idx = prices.index
    total_shares = 0.0
    total_invested = 0.0
    purchases = []  # 연도별 매수 기록

    for year in range(start_year, end_year + 1):
        per_buy_amount = annual_investment / len(buy_dates)
        for (m, d) in buy_dates:
            try:
                candidate = pd.Timestamp(year=year, month=m, day=d)
            except ValueError:
                continue
            pos = get_next_trading_day_index(idx, candidate)
            if pos is None:
                continue

            price_val = prices.iloc[pos]
            if isinstance(price_val, pd.Series):
                price_val = price_val.iloc[0]
            px = float(price_val)

            if px > 0:
                shares = per_buy_amount / px
                total_shares += shares
                total_invested += per_buy_amount
                purchases.append({
                    "year": year,
                    "date": prices.index[pos].strftime("%Y-%m-%d"),
                    "price": px,
                    "shares_bought": shares,
                    "amount_invested": per_buy_amount
                })

    # 최종 매도일: 2025-01-01 이후 첫 거래일
    final_target = pd.Timestamp("2025-01-01")
    pos_final = get_next_trading_day_index(prices.index, final_target)
    if pos_final is None:
        raise RuntimeError("Final trading day not found in price series.")
    final_price = float(prices.iloc[pos_final])
    final_date = prices.index[pos_final]

    final_value = total_shares * final_price
    profit = final_value - total_invested
    roi = (profit / total_invested * 100.0) if total_invested > 0 else 0.0

    return {
        "buy_dates": buy_dates,
        "total_invested": total_invested,
        "final_value": final_value,
        "profit": profit,
        "roi_pct": roi,
        "final_date": final_date.strftime("%Y-%m-%d"),
        "purchases": purchases,  # 추가
    }

# ----------------------------
# 최적 날짜 찾기
# ----------------------------
def find_top_dates(ticker: str, start_year: int, end_year: int,
                   annual_investment: float, top_n: int = 10, export_csv: str = None):
    prices = fetch_prices(ticker, start=f"{start_year}-01-01", end="2025-01-10")
    results = []

    # 1년에 1번
    print("Testing 365 single-date candidates (1 buy/year)...")
    for m in range(1, 13):
        for d in range(1, 32):
            try:
                pd.Timestamp(year=2000, month=m, day=d)  # 유효한 날짜 확인
            except ValueError:
                continue
            res = simulate_fixed_dates(prices, start_year, end_year, [(m, d)], annual_investment)
            results.append(res)

    df1 = pd.DataFrame(results).sort_values(by="final_value", ascending=False).reset_index(drop=True)
    print("\nTop 10 single-date strategies:")
    print(df1.head(top_n)[["buy_dates", "total_invested", "final_value", "profit", "roi_pct"]].to_string(index=False))

    # 1등 전략 상세 출력
    print("\nTop 1 single-date strategy details:")
    top1 = df1.iloc[0]
    for p in top1["purchases"]:
        print(f"{p['year']}: {p['date']} - Price: {p['price']:.2f}, Shares: {p['shares_bought']:.4f}, Invested: {p['amount_invested']:.2f}")

    # 1년에 2번 (약 6개월 간격)
    print("\nTesting ~180-day spaced pairs (2 buys/year)...")
    results2 = []
    for m1 in range(1, 13):
        for d1 in [1, 15]:
            try:
                pd.Timestamp(year=2000, month=m1, day=d1)
            except ValueError:
                continue
            for m2 in range(1, 13):
                for d2 in [1, 15]:
                    try:
                        pd.Timestamp(year=2000, month=m2, day=d2)
                    except ValueError:
                        continue
                    # 최소 150일 이상 차이나야 "반년 간격"으로 인정
                    dt1 = datetime(2000, m1, d1)
                    dt2 = datetime(2000, m2, d2)
                    delta = abs((dt2 - dt1).days)
                    if 150 <= delta <= 210:
                        res = simulate_fixed_dates(prices, start_year, end_year, [(m1, d1), (m2, d2)], annual_investment)
                        results2.append(res)

    df2 = pd.DataFrame(results2).sort_values(by="final_value", ascending=False).reset_index(drop=True)
    print("\nTop 10 double-date strategies (~6 months apart):")
    print(df2.head(top_n)[["buy_dates", "total_invested", "final_value", "profit", "roi_pct"]].to_string(index=False))

    # 1등 전략 상세 출력
    print("\nTop 1 double-date strategy details:")
    top1_double = df2.iloc[0]
    for p in top1_double["purchases"]:
        print(f"{p['year']}: {p['date']} - Price: {p['price']:.2f}, Shares: {p['shares_bought']:.4f}, Invested: {p['amount_invested']:.2f}")

    if export_csv:
        with pd.ExcelWriter(export_csv) as writer:
            df1.to_excel(writer, sheet_name="1_per_year", index=False)
            df2.to_excel(writer, sheet_name="2_per_year", index=False)
        print(f"\nAll results saved to: {export_csv}")

    return df1, df2

# ----------------------------
# 실행 부분
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find best QQQ buy dates (1 or 2 times per year)")
    parser.add_argument("--annual-investment", type=float, default=36500.0)
    parser.add_argument("--start-year", type=int, default=2004)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--ticker", type=str, default="QQQ")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--export-csv", type=str, default=None)
    args = parser.parse_args()

    find_top_dates(
        ticker=args.ticker,
        start_year=args.start_year,
        end_year=args.end_year,
        annual_investment=args.annual_investment,
        top_n=args.top_n,
        export_csv=args.export_csv,
    )
