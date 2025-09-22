# qqq_dca_365runs_fixed.py
"""
Fixed version of spy_dca_365runs.py (QQQ version)
- Ensures `prices` is always a pandas Series (adjusted close).
- Robust handling when fetching small future window for final valuation.
- Uses .iat safely to extract scalar price values.
- Plots ROI vs buys_per_year as before.
"""

import argparse
import math
import calendar
from typing import List
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# ----------------------------
# Fetch prices (return guaranteed Series)
# ----------------------------
def fetch_prices(ticker: str, start: str, end: str) -> pd.Series:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"No data for {ticker} between {start} and {end}")
    # Prefer 'Adj Close' then 'Close'
    if isinstance(df, pd.DataFrame):
        if "Adj Close" in df.columns:
            ser = df["Adj Close"].copy()
        elif "Close" in df.columns:
            ser = df["Close"].copy()
        elif df.shape[1] == 1:
            ser = df.iloc[:, 0].copy()
        else:
            ser = df.squeeze()
    else:
        ser = pd.Series(df).copy()
    ser = ser.sort_index()
    if isinstance(ser, pd.DataFrame):
        ser = ser.iloc[:, 0].copy()
    ser.name = ticker
    return ser

# ----------------------------
# Helpers
# ----------------------------
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

# ----------------------------
# Simulation
# ----------------------------
def simulate_for_buys_per_year(prices_raw, start_year: int, end_year: int,
                               buys_per_year: int, annual_investment: float) -> dict:
    if isinstance(prices_raw, pd.DataFrame):
        if "Adj Close" in prices_raw.columns:
            prices = prices_raw["Adj Close"].copy().sort_index()
        elif "Close" in prices_raw.columns:
            prices = prices_raw["Close"].copy().sort_index()
        else:
            prices = prices_raw.iloc[:, 0].copy().sort_index()
    else:
        prices = prices_raw.copy().sort_index()

    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:, 0].copy().sort_index()

    idx = prices.index
    total_shares, total_invested, total_trades = 0.0, 0.0, 0

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
            price_val = prices.iat[pos]
            if isinstance(price_val, (pd.Series, pd.DataFrame)):
                price_val = price_val.iloc[0]
            px = float(price_val)
            if px <= 0:
                continue
            shares = per_buy_amount / px
            total_shares += shares
            total_invested += per_buy_amount
            total_trades += 1

    final_target = pd.Timestamp("2025-01-01")
    pos_final = get_next_trading_day_index(prices.index, final_target)

    if pos_final is None:
        fut_df = yf.download(prices.name, start="2025-01-01", end="2025-01-15", auto_adjust=True, progress=False)
        if fut_df is None or fut_df.empty:
            raise RuntimeError("Cannot determine final trading day in early 2025 for final valuation.")
        if isinstance(fut_df, pd.DataFrame):
            if "Adj Close" in fut_df.columns:
                fut_ser = fut_df["Adj Close"].copy()
            elif "Close" in fut_df.columns:
                fut_ser = fut_df["Close"].copy()
            else:
                fut_ser = fut_df.iloc[:, 0].copy()
        else:
            fut_ser = pd.Series(fut_df).copy()
        fut_ser.name = prices.name
        combined = pd.concat([prices, fut_ser]).sort_index()
        combined = combined[~combined.index.duplicated(keep='first')]
        pos_final = get_next_trading_day_index(combined.index, final_target)
        if pos_final is None:
            raise RuntimeError("Cannot find a trading day on/after 2025-01-01 in combined series.")
        final_price = float(combined.iat[pos_final])
        final_date = combined.index[pos_final]
    else:
        final_price = float(prices.iat[pos_final])
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

# ----------------------------
# Run all sims
# ----------------------------
def run_all(annual_investment: float, start_year: int = 2004, end_year: int = 2024,
            min_buys: int = 1, max_buys: int = 365, ticker: str = "QQQ", export_csv: str = None):
    prices = fetch_prices(ticker, start=f"{start_year}-01-01", end="2025-01-15")
    results = []
    print(f"Running simulations for buys_per_year = {min_buys}..{max_buys} "
          f"(annual_investment=${annual_investment:,.2f}, ticker={ticker})")
    for n in range(min_buys, max_buys + 1):
        try:
            res = simulate_for_buys_per_year(prices, start_year, end_year, n, annual_investment)
        except Exception as e:
            print(f"Warning: simulation for n={n} raised an error: {e}. Skipping.")
            continue
        results.append(res)

    df_all = pd.DataFrame(results)
    df_top = df_all.sort_values(by='final_value', ascending=False).reset_index(drop=True)
    pd.set_option('display.float_format', lambda x: f'{x:,.2f}')
    print("\nTop 20 strategies by final value (2025-01-01 기준):")
    if not df_top.empty:
        print(df_top.head(20)[['buys_per_year','total_invested','final_value',
                               'profit','roi_pct','total_trades','final_date']].to_string(index=False))
    else:
        print("No results (all simulations failed).")

    if export_csv and not df_all.empty:
        df_all.to_csv(export_csv, index=False)
        print(f"\nAll results saved to: {export_csv}")

    return df_all

# ----------------------------
# CLI + plotting
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run QQQ DCA sims (fixed) and plot ROI")
    parser.add_argument("--annual-investment", type=float, default=36500.0, help="Total money invested per year (USD)")
    parser.add_argument("--start-year", type=int, default=2004, help="Start year inclusive")
    parser.add_argument("--end-year", type=int, default=2024, help="End year inclusive")
    parser.add_argument("--min-buys", type=int, default=1, help="Minimum buys per year (default 1)")
    parser.add_argument("--max-buys", type=int, default=365, help="Maximum buys per year (default 365)")
    parser.add_argument("--ticker", type=str, default="QQQ", help="Ticker to simulate (default QQQ)")
    parser.add_argument("--export-csv", type=str, default=None, help="Path to save full results CSV")
    args = parser.parse_args()

    df_all = run_all(args.annual_investment, args.start_year, args.end_year,
                     args.min_buys, args.max_buys, args.ticker, args.export_csv)

    if df_all.empty:
        raise SystemExit("No simulation results available to plot.")

    df_plot = df_all.sort_values(by='buys_per_year').reset_index(drop=True)
    x = df_plot['buys_per_year']
    y = df_plot['roi_pct']

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    axes[0].plot(x, y, linestyle='-', linewidth=1)
    axes[0].set_title(f"ROI (%) vs Buys per Year — Line (ticker={args.ticker})", fontsize=14)
    axes[0].set_ylabel("ROI (%)", fontsize=12)
    axes[0].grid(True)

    axes[1].scatter(x, y, s=12)
    axes[1].set_title(f"ROI (%) vs Buys per Year — Points (ticker={args.ticker})", fontsize=14)
    axes[1].set_xlabel("Buys per Year", fontsize=12)
    axes[1].set_ylabel("ROI (%)", fontsize=12)
    axes[1].grid(True)

    max_x = int(df_plot['buys_per_year'].max())
    step = max(1, max_x // 20)
    axes[1].set_xticks(range(1, max_x + 1, step))

    plt.tight_layout()
    plt.show()
