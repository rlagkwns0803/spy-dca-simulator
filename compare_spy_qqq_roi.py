#!/usr/bin/env python3
# compare_spy_qqq_roi.py
"""
Compare SPY vs QQQ: ROI (%) vs Buys per Year (1..N) with twin y-axis
- Left y-axis: SPY ROI (%)
- Right y-axis: QQQ ROI (%)
- Usage example:
    python compare_spy_qqq_roi.py --annual-investment 36500 --start-year 2004 --end-year 2024 --max-buys 365 --save-png compare.png
"""

import argparse
import calendar
import math
from typing import List
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt

# ---- helper: fetch adjusted series ----
def fetch_prices_series(ticker: str, start: str, end: str) -> pd.Series:
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"No data for {ticker} between {start} and {end}")
    if isinstance(df, pd.DataFrame):
        if "Adj Close" in df.columns:
            ser = df["Adj Close"].copy()
        elif "Close" in df.columns:
            ser = df["Close"].copy()
        else:
            ser = df.iloc[:, 0].copy()
    else:
        ser = pd.Series(df).copy()
    ser = ser.sort_index()
    if isinstance(ser, pd.DataFrame):
        ser = ser.iloc[:, 0].copy()
    ser.name = ticker
    return ser

# ---- date generation helpers ----
def is_leap_year(y: int) -> bool:
    return calendar.isleap(y)

def month_based_dates(year:int, n:int) -> List[pd.Timestamp]:
    months = []
    for i in range(n):
        m = int(math.floor(i * 12.0 / n)) + 1
        months.append(m)
    months = sorted(set(months))
    return [pd.Timestamp(year=year, month=m, day=1) for m in months]

def day_based_dates(year:int, n:int) -> List[pd.Timestamp]:
    days = 366 if is_leap_year(year) else 365
    positions = np.linspace(0, days-1, n)
    offsets = np.rint(positions).astype(int)
    offsets = np.unique(offsets)
    return [pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=int(d)) for d in offsets]

def generate_buy_dates_for_year(year:int, n:int) -> List[pd.Timestamp]:
    if n <= 0:
        return []
    if n <= 12:
        return month_based_dates(year, n)
    else:
        return day_based_dates(year, n)

def get_next_trading_index(index:pd.DatetimeIndex, dt:pd.Timestamp):
    pos = index.searchsorted(dt)
    if pos >= len(index):
        return None
    return pos

# ---- simulation for one ticker ----
def simulate_for_buys_per_year(prices_raw, start_year:int, end_year:int, buys_per_year:int, annual_investment:float):
    # ensure series
    if isinstance(prices_raw, pd.DataFrame):
        if "Adj Close" in prices_raw.columns:
            prices = prices_raw["Adj Close"].copy().sort_index()
        elif "Close" in prices_raw.columns:
            prices = prices_raw["Close"].copy().sort_index()
        else:
            prices = prices_raw.iloc[:,0].copy().sort_index()
    else:
        prices = prices_raw.copy().sort_index()
    if isinstance(prices, pd.DataFrame):
        prices = prices.iloc[:,0].copy().sort_index()

    idx = prices.index
    total_shares = 0.0
    total_invested = 0.0
    total_trades = 0

    for year in range(start_year, end_year+1):
        buy_dates = generate_buy_dates_for_year(year, buys_per_year)
        buy_dates = [d for d in buy_dates if d.year == year]
        if not buy_dates:
            continue
        per_buy = annual_investment / len(buy_dates)
        for dt in buy_dates:
            pos = get_next_trading_index(idx, dt)
            if pos is None:
                continue
            price_val = prices.iat[pos]
            if isinstance(price_val, (pd.Series, pd.DataFrame)):
                price_val = price_val.iloc[0]
            px = float(price_val)
            if px <= 0:
                continue
            shares = per_buy / px
            total_shares += shares
            total_invested += per_buy
            total_trades += 1

    # final valuation on 2025-01-01 or next trading day
    final_target = pd.Timestamp("2025-01-01")
    pos_final = get_next_trading_index(prices.index, final_target)
    if pos_final is None:
        fut = yf.download(prices.name, start="2025-01-01", end="2025-01-15", auto_adjust=True, progress=False)
        if fut is None or fut.empty:
            raise RuntimeError("Cannot determine final trading day in early 2025.")
        if isinstance(fut, pd.DataFrame):
            if "Adj Close" in fut.columns:
                fut_ser = fut["Adj Close"].copy()
            elif "Close" in fut.columns:
                fut_ser = fut["Close"].copy()
            else:
                fut_ser = fut.iloc[:,0].copy()
        else:
            fut_ser = pd.Series(fut).copy()
        fut_ser.name = prices.name
        combined = pd.concat([prices, fut_ser]).sort_index()
        combined = combined[~combined.index.duplicated(keep='first')]
        pos_final = get_next_trading_index(combined.index, final_target)
        if pos_final is None:
            raise RuntimeError("Cannot find trading day on/after 2025-01-01.")
        final_px = float(combined.iat[pos_final])
    else:
        final_px = float(prices.iat[pos_final])

    final_value = total_shares * final_px
    profit = final_value - total_invested
    roi_pct = (profit / total_invested * 100.0) if total_invested > 0 else 0.0

    return {
        "buys_per_year": buys_per_year,
        "total_invested": total_invested,
        "final_value": final_value,
        "profit": profit,
        "roi_pct": roi_pct,
        "total_trades": total_trades
    }

# ---- run for a ticker across range ----
def run_ticker(ticker:str, start_year:int, end_year:int, min_buys:int, max_buys:int, annual_investment:float):
    print(f"Fetching prices for {ticker} ...")
    prices = fetch_prices_series(ticker, start=f"{start_year}-01-01", end="2025-01-15")
    results = []
    total = max_buys - min_buys + 1
    for i, n in enumerate(range(min_buys, max_buys+1), start=1):
        if i % 50 == 0 or i==1 or i==total:
            print(f"  {ticker}: sim {i}/{total} (buys_per_year={n})")
        try:
            r = simulate_for_buys_per_year(prices, start_year, end_year, n, annual_investment)
            results.append(r)
        except Exception as e:
            print(f"  Warning: {ticker} n={n} error: {e}  (skipping)")
            continue
    df = pd.DataFrame(results)
    # ensure full coverage of buys (in case of skips)
    df = df.sort_values("buys_per_year").reset_index(drop=True)
    return df

# ---- main: run SPY & QQQ and plot combined with twin y-axis ----
def main():
    parser = argparse.ArgumentParser(description="Compare SPY and QQQ ROI vs Buys per Year (twin y-axis)")
    parser.add_argument("--annual-investment", type=float, default=36500.0)
    parser.add_argument("--start-year", type=int, default=2004)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--min-buys", type=int, default=1)
    parser.add_argument("--max-buys", type=int, default=365)
    parser.add_argument("--save-csv", type=str, default=None, help="Export combined results CSV")
    parser.add_argument("--save-png", type=str, default=None, help="Save plot PNG")
    args = parser.parse_args()

    tickers = ["SPY", "QQQ"]
    dfs = {}
    for t in tickers:
        dfs[t] = run_ticker(t, args.start_year, args.end_year, args.min_buys, args.max_buys, args.annual_investment)

    # x-axis full range
    x_full = list(range(args.min_buys, args.max_buys+1))

    # Prepare plotting series for each ticker aligned to x_full (may contain NaN for missing)
    df_plot_map = {}
    for t in tickers:
        df = dfs[t][["buys_per_year","roi_pct"]].copy()
        df_plot = pd.DataFrame({"buys_per_year": x_full}).merge(df, on="buys_per_year", how="left")
        df_plot_map[t] = df_plot

    # Create twin-axis plot
    fig, ax_left = plt.subplots(figsize=(14,7))
    ax_right = ax_left.twinx()

    colors = {"SPY": "tab:blue", "QQQ": "tab:orange"}

    # SPY on left axis
    spy_dfp = df_plot_map["SPY"]
    ax_left.plot(spy_dfp["buys_per_year"], spy_dfp["roi_pct"], color=colors["SPY"], linestyle='-', linewidth=1.5, label="SPY")
    ax_left.set_ylabel("ROI (%) - SPY", color=colors["SPY"], fontsize=12)
    ax_left.tick_params(axis="y", labelcolor=colors["SPY"])
    ax_left.grid(True, linestyle='--', alpha=0.6)

    # QQQ on right axis
    qqq_dfp = df_plot_map["QQQ"]
    ax_right.plot(qqq_dfp["buys_per_year"], qqq_dfp["roi_pct"], color=colors["QQQ"], linestyle='-', linewidth=1.5, label="QQQ")
    ax_right.set_ylabel("ROI (%) - QQQ", color=colors["QQQ"], fontsize=12)
    ax_right.tick_params(axis="y", labelcolor=colors["QQQ"])

    # Title and x-label
    ax_left.set_xlabel("Buys per Year", fontsize=12)
    ax_left.set_title(f"ROI (%) vs Buys per Year — SPY vs QQQ (start {args.start_year}, end {args.end_year})", fontsize=14)

    # Combined legend (collect handles from both axes)
    lines_left, labels_left = ax_left.get_legend_handles_labels()
    lines_right, labels_right = ax_right.get_legend_handles_labels()
    ax_left.legend(lines_left + lines_right, labels_left + labels_right, loc="upper left")

    # x ticks
    max_x = args.max_buys
    step = max(1, max_x // 20)
    ax_left.set_xticks(range(args.min_buys, max_x+1, step))

    plt.tight_layout()

    # Save PNG if requested
    if args.save_png:
        plt.savefig(args.save_png, dpi=300)
        print(f"Saved plot to {args.save_png}")

    plt.show()

    # optional CSV save: combine both into one CSV
    if args.save_csv:
        combined = pd.DataFrame({"buys_per_year": x_full})
        for t in tickers:
            df = dfs[t][["buys_per_year","roi_pct"]].rename(columns={"roi_pct":f"{t}_roi_pct"})
            combined = combined.merge(df, on="buys_per_year", how="left")
        combined.to_csv(args.save_csv, index=False)
        print(f"Saved combined CSV to {args.save_csv}")

if __name__ == "__main__":
    main()
