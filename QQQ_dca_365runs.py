# qqq_dca_365runs.py

import argparse
import math
import calendar
from typing import List
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def fetch_prices(ticker: str, start: str, end: str) -> pd.Series:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise RuntimeError(f"No data for {ticker} between {start} and {end}")
    if 'Adj Close' in df.columns:
        ser = df['Adj Close'].copy()
    else:
        ser = df['Close'].copy()
    ser.name = ticker
    return ser.sort_index()

def is_leap_year(y: int) -> bool:
    return calendar.isleap(y)

def month_based_dates(year: int, n: int) -> List[pd.Timestamp]:
    months = []
    for i in range(n):
        m = int(math.floor(i * 12.0 / n)) + 1
        months.append(m)
    months = sorted(set(months))
    dates = [pd.Timestamp(year=year, month=m, day=1) for m in months]
    return dates

def day_based_dates(year: int, n: int) -> List[pd.Timestamp]:
    days_in_year = 366 if is_leap_year(year) else 365
    positions = np.linspace(0, days_in_year - 1, n)
    offsets = np.rint(positions).astype(int)
    offsets = np.unique(offsets)
    dates = [pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=int(d)) for d in offsets]
    return dates

def generate_buy_dates_for_year(year: int, n: int) -> List[pd.Timestamp]:
    if n <= 0:
        return []
    if n <= 12:
        return month_based_dates(year, n)
    else:
        return day_based_dates(year, n)

def get_next_trading_day_index(index: pd.DatetimeIndex, dt: pd.Timestamp):
    pos = index.searchsorted(dt)
    if pos >= len(index):
        return None
    return pos

def simulate_for_buys_per_year(prices: pd.Series, start_year: int, end_year: int,
                               buys_per_year: int, annual_investment: float) -> dict:
    idx = prices.index
    total_shares = 0.0
    total_invested = 0.0
    total_trades = 0

    for year in range(start_year, end_year + 1):
        raw_dates = generate_buy_dates_for_year(year, buys_per_year)
        raw_dates = [d for d in raw_dates if d.year == year]
        if len(raw_dates) == 0:
            continue
        per_buy_amount = annual_investment / len(raw_dates)
        for buy_dt in raw_dates:
            pos = get_next_trading_day_index(idx, buy_dt)
            if pos is None:
                continue
            trade_dt = idx[pos]
            price = float(prices.iloc[pos])
            shares = per_buy_amount / price if price > 0 else 0.0
            total_shares += shares
            total_invested += per_buy_amount
            total_trades += 1

    final_target = pd.Timestamp("2025-01-01")
    pos_final = get_next_trading_day_index(prices.index, final_target)
    if pos_final is None:
        future = yf.download(prices.name, start="2025-01-01", end="2025-01-15", auto_adjust=True, progress=False)
        if future is None or future.empty:
            raise RuntimeError("Cannot determine final trading day in early 2025 for final valuation.")
        future_ser = future['Close'] if 'Close' in future.columns else future.iloc[:,0]
        future_ser.name = prices.name
        combined = pd.concat([prices, future_ser])
        combined = combined[~combined.index.duplicated(keep='first')].sort_index()
        pos_final = get_next_trading_day_index(combined.index, final_target)
        final_price = float(combined.iloc[pos_final])
        final_date = combined.index[pos_final]
    else:
        final_price = float(prices.iloc[pos_final])
        final_date = prices.index[pos_final]

    final_value = total_shares * final_price
    profit = final_value - total_invested
    roi = (profit / total_invested * 100.0) if total_invested > 0 else 0.0

    return {
        'buys_per_year': buys_per_year,
        'total_invested': total_invested,
        'final_value': final_value,
        'profit': profit,
        'roi_pct': roi,
        'final_date': final_date.strftime("%Y-%m-%d"),
        'total_trades': total_trades
    }

def run_all(annual_investment: float, start_year: int = 2004, end_year: int = 2024,
            min_buys: int = 1, max_buys: int = 365, ticker: str = "QQQ", export_csv: str = None):
    prices = fetch_prices(ticker, start=f"{start_year}-01-01", end="2025-01-15")
    results = []
    print(f"Running simulations for buys_per_year = {min_buys}..{max_buys} (annual_investment=${annual_investment:,.2f}, ticker={ticker})")
    for n in range(min_buys, max_buys + 1):
        res = simulate_for_buys_per_year(prices, start_year, end_year, n, annual_investment)
        results.append(res)
    df = pd.DataFrame(results)
    df_sorted = df.sort_values(by='final_value', ascending=False).reset_index(drop=True)
    pd.set_option('display.float_format', lambda x: f'{x:,.2f}')
    print("\nTop 20 strategies by final value (2025-01-01 기준):")
    print(df_sorted.head(20)[['buys_per_year','total_invested','final_value','profit','roi_pct','total_trades','final_date']].to_string(index=False))
    if export_csv:
        df_sorted.to_csv(export_csv, index=False)
        print(f"\nAll results saved to: {export_csv}")
    return df_sorted

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QQQ DCA sims for buys_per_year = 1..365 and rank by final value")
    parser.add_argument("--annual-investment", type=float, default=36500.0, help="Total money invested per year (USD). Default 36500 -> $100/day)")
    parser.add_argument("--start-year", type=int, default=2004, help="Start year inclusive")
    parser.add_argument("--end-year", type=int, default=2024, help="End year inclusive")
    parser.add_argument("--min-buys", type=int, default=1, help="Minimum buys per year to simulate (default 1)")
    parser.add_argument("--max-buys", type=int, default=365, help="Maximum buys per year to simulate (default 365)")
    parser.add_argument("--ticker", type=str, default="QQQ", help="Ticker to simulate")
    parser.add_argument("--export-csv", type=str, default=None, help="Path to export full sorted results CSV")
    args = parser.parse_args()

    run_all(args.annual_investment, args.start_year, args.end_year, args.min_buys, args.max_buys, args.ticker, args.export_csv)
