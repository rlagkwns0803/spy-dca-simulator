#!/usr/bin/env python3
"""
topk_mcap_dca_march7.py (fixed with readable output + constituents column)

Changes:
 - pd.set_option to avoid scientific notation in results
 - simulate_topk_dca now includes "constituents" column (tickers used)
 - top20 printouts include constituents
"""
import argparse
import math
import time
import sys
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from io import StringIO

# ---------- Pandas display setup ----------
pd.set_option("display.float_format", "{:,.2f}".format)

# ---------- Utility: fetch Wikipedia tables robustly ----------
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"}

def fetch_wikipedia_table(url: str, candidate_columns: List[str], timeout: float = 15.0) -> pd.DataFrame:
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    html = r.text
    try:
        tables = pd.read_html(StringIO(html))
        for tbl in tables:
            cols = [str(c).lower() for c in tbl.columns.astype(str)]
            for key in candidate_columns:
                if any(key.lower() in c for c in cols):
                    return tbl
    except Exception:
        pass
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table", {"class": "wikitable"})
    for t in tables:
        rows = []
        for tr in t.find_all("tr"):
            cols = [td.get_text(strip=True) for td in tr.find_all(["th","td"])]
            if cols:
                rows.append(cols)
        if not rows:
            continue
        header = rows[0]
        body = rows[1:]
        try:
            df = pd.DataFrame(body, columns=header)
        except Exception:
            df = pd.DataFrame(body)
            df.columns = [f"col_{i}" for i in range(df.shape[1])]
        cols_lower = [c.lower() for c in df.columns.astype(str)]
        for key in candidate_columns:
            if any(key.lower() in c for c in cols_lower):
                return df
    raise RuntimeError(f"No suitable table found at {url}")

def get_sp500_constituents() -> List[str]:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    df = fetch_wikipedia_table(url, ["Symbol", "Security", "GICS Sector"])
    sym_col = None
    for c in df.columns:
        if 'symbol' in str(c).lower():
            sym_col = c
            break
    if sym_col is None:
        sym_col = df.columns[0]
    tickers = df[sym_col].astype(str).str.strip().tolist()
    return [t.replace('.', '-') for t in tickers]

def get_nasdaq100_constituents() -> List[str]:
    url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    df = fetch_wikipedia_table(url, ["Ticker", "Ticker symbol", "Symbol", "Company"])
    sym_col = None
    for c in df.columns:
        if any(k in str(c).lower() for k in ['ticker', 'symbol']):
            sym_col = c
            break
    if sym_col is None:
        sym_col = df.columns[0]
    tickers = df[sym_col].astype(str).str.strip().tolist()
    return [t.replace('.', '-') for t in tickers]

# ---------- Price download helpers ----------
def safe_download_series(ticker: str, start: str, end: str, retry: int = 2) -> pd.Series:
    for attempt in range(retry + 1):
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            if df is None or df.empty:
                return pd.Series(dtype=float)
            if isinstance(df, pd.DataFrame):
                if 'Adj Close' in df.columns:
                    ser = df['Adj Close'].copy()
                elif 'Close' in df.columns:
                    ser = df['Close'].copy()
                else:
                    ser = df.iloc[:, 0].copy()
            else:
                ser = pd.Series(df).copy()
            if isinstance(ser, pd.DataFrame):
                if 'Adj Close' in ser.columns:
                    ser = ser['Adj Close'].copy()
                else:
                    ser = ser.iloc[:, 0].copy()
            ser.index = pd.to_datetime(ser.index)
            ser.name = ticker
            ser = ser.sort_index()
            return ser
        except Exception as e:
            if attempt < retry:
                time.sleep(1 + attempt * 0.5)
                continue
            else:
                print(f"Warning: failed to download {ticker}: {e}", file=sys.stderr)
                return pd.Series(dtype=float)

def batch_download(tickers: List[str], start: str, end: str, pause: float = 0.02) -> Dict[str, pd.Series]:
    result = {}
    for i, t in enumerate(tickers, start=1):
        if i % 50 == 0 or i == 1 or i == len(tickers):
            print(f"  downloading {i}/{len(tickers)}: {t}")
        ser = safe_download_series(t, start, end, retry=2)
        result[t] = ser
        time.sleep(pause)
    return result

# ---------- Market cap retrieval ----------
def get_market_caps(tickers: List[str], method: str = 'current', start_price_date: str = None) -> Dict[str, float]:
    caps = {}
    for i, t in enumerate(tickers, start=1):
        if i % 50 == 0 or i==1 or i==len(tickers):
            print(f"  fetching marketCap {i}/{len(tickers)}: {t}")
        try:
            tk = yf.Ticker(t)
            info = tk.info or {}
        except Exception:
            info = {}
        cap = float('nan')
        if method == 'current':
            cap = info.get('marketCap', None)
            if cap is None:
                shares = info.get('sharesOutstanding', None)
                try:
                    hist = tk.history(period="1d", auto_adjust=True)
                    lastp = float(hist['Close'].iloc[-1]) if hist is not None and not hist.empty else None
                except Exception:
                    lastp = None
                if shares and lastp:
                    cap = shares * lastp
        else:
            shares = info.get('sharesOutstanding', None)
            cap = None
            if shares and start_price_date:
                try:
                    hist = tk.history(start=start_price_date, end=pd.to_datetime(start_price_date)+pd.Timedelta(days=7), auto_adjust=True)
                    if hist is not None and not hist.empty:
                        p0 = float(hist['Close'].iloc[0])
                        cap = shares * p0
                except Exception:
                    cap = None
        caps[t] = float(cap) if (cap is not None and not (isinstance(cap, float) and math.isnan(cap))) else float('nan')
        time.sleep(0.03)
    return caps

# ---------- Helper: next trading date price ----------
def next_trading_price(series: pd.Series, target_dt: pd.Timestamp) -> Tuple[pd.Timestamp, float]:
    if series is None or series.empty:
        return (None, None)
    if isinstance(series, pd.DataFrame):
        if 'Adj Close' in series.columns:
            series = series['Adj Close']
        elif 'Close' in series.columns:
            series = series['Close']
        else:
            series = series.iloc[:, 0]
    pos = series.index.searchsorted(pd.to_datetime(target_dt))
    if pos >= len(series):
        return (None, None)
    val = series.iat[pos]
    try:
        return (series.index[pos], float(val))
    except Exception:
        return (series.index[pos], None)

# ---------- Core simulation ----------
def simulate_topk_dca(
    constituents_sorted: List[str],
    caps_map: Dict[str,float],
    price_map: Dict[str,pd.Series],
    start_year: int,
    end_year: int,
    buy_month: int,
    buy_day: int,
    annual_investment: float,
    final_valuation_date: str = "2025-01-02"
) -> pd.DataFrame:
    years = list(range(start_year, end_year + 1))
    final_dt = pd.to_datetime(final_valuation_date)
    results = []
    N = len(constituents_sorted)
    final_price_map = {}
    for t in constituents_sorted:
        ser = price_map.get(t, pd.Series(dtype=float))
        d, px = next_trading_price(ser, final_dt)
        final_price_map[t] = px
    for k in range(1, N+1):
        sel = constituents_sorted[:k]
        caps = np.array([ (caps_map.get(t, float('nan')) if not (math.isnan(caps_map.get(t, float('nan')))) else 0.0) for t in sel ], dtype=float)
        if np.nansum(caps) <= 0 or np.nanmax(caps) <= 0:
            base_weights = np.ones(len(sel)) / len(sel)
        else:
            caps[caps < 0] = 0.0
            caps = np.nan_to_num(caps, nan=0.0)
            base_weights = caps / caps.sum() if caps.sum() > 0 else np.ones(len(sel))/len(sel)
        shares = {t: 0.0 for t in sel}
        total_invested = 0.0
        for y in years:
            buy_dt = pd.Timestamp(year=y, month=buy_month, day=buy_day)
            available_idx = []
            for idx, t in enumerate(sel):
                ser = price_map.get(t, pd.Series(dtype=float))
                d, px = next_trading_price(ser, buy_dt)
                if d is not None and px is not None:
                    available_idx.append(idx)
            if len(available_idx) == 0:
                continue
            avail_weights = base_weights[available_idx]
            if avail_weights.sum() <= 0:
                avail_weights = np.ones_like(avail_weights) / len(avail_weights)
            else:
                avail_weights = avail_weights / avail_weights.sum()
            for idx_in_list, idx_sel in enumerate(available_idx):
                t = sel[idx_sel]
                amt = annual_investment * float(avail_weights[idx_in_list])
                ser = price_map.get(t, pd.Series(dtype=float))
                d, px = next_trading_price(ser, buy_dt)
                if d is None or px is None or px <= 0 or math.isnan(px):
                    continue
                qty = amt / px
                shares[t] += qty
                total_invested += amt
        final_value = 0.0
        for t in sel:
            px = final_price_map.get(t, None)
            if px is None or (isinstance(px, float) and math.isnan(px)):
                ser = price_map.get(t, pd.Series(dtype=float))
                if ser is None or ser.empty:
                    continue
                try:
                    px = float(ser.iloc[-1])
                except Exception:
                    continue
            final_value += shares[t] * px
        profit = final_value - total_invested
        roi_pct = (profit / total_invested * 100.0) if total_invested > 0 else float('nan')
        results.append({
            'k': k,
            'num_stocks': k,
            'total_invested': total_invested,
            'final_value': final_value,
            'profit': profit,
            'roi_pct': roi_pct,
            'constituents': ", ".join(sel)
        })
    return pd.DataFrame(results)

# ---------- Main CLI ----------
def main():
    p = argparse.ArgumentParser(description="Top-k market-cap DCA sim (annual single buy on date)")
    p.add_argument("--annual-investment", type=float, default=36500.0)
    p.add_argument("--start-year", type=int, default=2004)
    p.add_argument("--end-year", type=int, default=2024)
    p.add_argument("--buy-month", type=int, default=3)
    p.add_argument("--buy-day", type=int, default=7)
    p.add_argument("--final-date", type=str, default="2025-01-02")
    p.add_argument("--cap-method", choices=['current','start'], default='current')
    p.add_argument("--max-sp500", type=int, default=500)
    p.add_argument("--max-nasdaq", type=int, default=100)
    p.add_argument("--save-csv-prefix", type=str, default=None)
    args = p.parse_args()
    print("1) Fetching constituents...")
    sp500 = get_sp500_constituents()[:args.max_sp500]
    nas100 = get_nasdaq100_constituents()[:args.max_nasdaq]
    print(f"  S&P500: {len(sp500)}, NASDAQ100: {len(nas100)}")
    all_tickers = sorted(set(sp500 + nas100))
    print(f"\n2) Downloading prices for {len(all_tickers)} tickers...")
    price_map = batch_download(all_tickers, start=f"{args.start_year}-01-01", end=args.final_date, pause=0.01)
    print(f"\n3) Retrieving market caps ({args.cap_method})...")
    if args.cap_method == 'start':
        caps_map = get_market_caps(all_tickers, method='start', start_price_date=f"{args.start_year}-01-01")
    else:
        caps_map = get_market_caps(all_tickers, method='current')
    def sort_by_caps(list_t):
        def keyfn(x):
            cap = caps_map.get(x, float('nan'))
            cap_val = 0.0 if (isinstance(cap, float) and math.isnan(cap)) else float(cap)
            return (-cap_val, x)
        return sorted(list_t, key=keyfn)
    sp500_sorted = sort_by_caps(sp500)[:args.max_sp500]
    nas100_sorted = sort_by_caps(nas100)[:args.max_nasdaq]
    print("\n4) Running simulations...")
    df_sp500 = simulate_topk_dca(sp500_sorted, caps_map, price_map,
                                 args.start_year, args.end_year,
                                 args.buy_month, args.buy_day, args.annual_investment,
                                 final_valuation_date=args.final_date)
    df_nas100 = simulate_topk_dca(nas100_sorted, caps_map, price_map,
                                  args.start_year, args.end_year,
                                  args.buy_month, args.buy_day, args.annual_investment,
                                  final_valuation_date=args.final_date)
    top20_sp500_by_profit = df_sp500.sort_values('profit', ascending=False).head(20)
    top20_sp500_by_roi = df_sp500.sort_values('roi_pct', ascending=False).head(20)
    top20_nas_by_profit = df_nas100.sort_values('profit', ascending=False).head(20)
    top20_nas_by_roi = df_nas100.sort_values('roi_pct', ascending=False).head(20)
    print("\n=== S&P500: Top 20 by PROFIT ===")
    print(top20_sp500_by_profit.to_string(index=False))
    print("\n=== S&P500: Top 20 by ROI% ===")
    print(top20_sp500_by_roi.to_string(index=False))
    print("\n=== NASDAQ100: Top 20 by PROFIT ===")
    print(top20_nas_by_profit.to_string(index=False))
    print("\n=== NASDAQ100: Top 20 by ROI% ===")
    print(top20_nas_by_roi.to_string(index=False))
    if args.save_csv_prefix:
        sp_fname = f"{args.save_csv_prefix}_SP500_k_results.csv"
        nas_fname = f"{args.save_csv_prefix}_NAS100_k_results.csv"
        df_sp500.to_csv(sp_fname, index=False)
        df_nas100.to_csv(nas_fname, index=False)
        print(f"\nSaved detailed results to: {sp_fname}, {nas_fname}")

if __name__ == "__main__":
    main()
