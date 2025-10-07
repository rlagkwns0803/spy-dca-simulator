#!/usr/bin/env python3
"""
topk_mcap_dca_monthly.py

Fixed + pretty-print version of your monthly top-k market-cap DCA simulator.

- 2004..2024, monthly buys on earliest tradable day >= 1st of each month
- For each buy: rank constituents by market-cap (at that buy date) and invest annual_investment/12
  across top-k by market-cap proportionally (we run k=1..N).
- Returns top-k table with final_value, profit, roi_pct and the list of constituents actually used.

This version:
 - uses Wikipedia scraping to obtain constituents (no yf.Ticker(...).constituents),
 - fixes various pandas/dataframe extraction edge-cases,
 - prints the top-20 results as clearly separated blocks (one block per k).
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
from collections import defaultdict

# --- Display settings: avoid scientific notation for nicer printing
pd.set_option("display.float_format", "{:,.2f}".format)
pd.set_option("display.max_colwidth", 400)

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                         "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"}


# ---------- Wikipedia table helpers ----------
def fetch_wikipedia_table(url: str, candidate_columns: List[str], timeout: float = 15.0) -> pd.DataFrame:
    """Fetch page with requests (UA header) and try pd.read_html(StringIO(html)).
       Falls back to BeautifulSoup simple parsing of wikitable if needed."""
    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    html = r.text
    # try pandas read_html on StringIO to avoid futurewarning
    try:
        tables = pd.read_html(StringIO(html))
        for tbl in tables:
            cols = [str(c).lower() for c in tbl.columns.astype(str)]
            for key in candidate_columns:
                if any(key.lower() in c for c in cols):
                    return tbl
    except Exception:
        pass
    # fallback: simple soup parse
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table", {"class": "wikitable"})
    for t in tables:
        rows = []
        for tr in t.find_all("tr"):
            cols = [td.get_text(strip=True) for td in tr.find_all(["th", "td"])]
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
    sym_col = next((c for c in df.columns if 'symbol' in str(c).lower()), df.columns[0])
    tickers = df[sym_col].astype(str).str.strip().tolist()
    return [t.replace('.', '-') for t in tickers]


def get_nasdaq100_constituents() -> List[str]:
    url = "https://en.wikipedia.org/wiki/NASDAQ-100"
    df = fetch_wikipedia_table(url, ["Ticker", "Ticker symbol", "Symbol", "Company"])
    sym_col = next((c for c in df.columns if any(k in str(c).lower() for k in ['ticker', 'symbol'])), df.columns[0])
    tickers = df[sym_col].astype(str).str.strip().tolist()
    return [t.replace('.', '-') for t in tickers]


# ---------- Price download helpers ----------
def safe_download_series(ticker: str, start: str, end: str, retry: int = 2) -> pd.Series:
    """Download adjusted close (or close) and ensure a pandas Series is returned."""
    for attempt in range(retry + 1):
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            if df is None or df.empty:
                return pd.Series(dtype=float)
            # prefer Adj Close then Close else first column
            if isinstance(df, pd.DataFrame):
                if 'Adj Close' in df.columns:
                    ser = df['Adj Close'].copy()
                elif 'Adj_Close' in df.columns:
                    ser = df['Adj_Close'].copy()
                elif 'Close' in df.columns:
                    ser = df['Close'].copy()
                else:
                    ser = df.iloc[:, 0].copy()
            else:
                ser = pd.Series(df).copy()
            # if it's still a DataFrame for some reason, reduce to first usable column
            if isinstance(ser, pd.DataFrame):
                if 'Adj Close' in ser.columns:
                    ser = ser['Adj Close'].copy()
                else:
                    ser = ser.iloc[:, 0].copy()
            ser.index = pd.to_datetime(ser.index)
            ser.name = ticker
            return ser.sort_index()
        except Exception as e:
            if attempt < retry:
                time.sleep(1 + attempt * 0.5)
                continue
            else:
                print(f"Warning: failed to download {ticker}: {e}", file=sys.stderr)
                return pd.Series(dtype=float)


def batch_download(tickers: List[str], start: str, end: str, pause: float = 0.02) -> Dict[str, pd.Series]:
    result: Dict[str, pd.Series] = {}
    for i, t in enumerate(tickers, start=1):
        if i % 50 == 0 or i == 1 or i == len(tickers):
            print(f"  downloading {i}/{len(tickers)}: {t}")
        result[t] = safe_download_series(t, start, end, retry=2)
        time.sleep(pause)
    return result


# ---------- sharesOutstanding (approx for market cap) ----------
def get_shares_outstanding(tickers: List[str], price_map: Dict[str, pd.Series] = None) -> Dict[str, float]:
    """
    Fetch sharesOutstanding via yfinance once per ticker.
    If sharesOutstanding is missing but 'marketCap' is present in info and a price_map entry exists,
    estimate sharesOutstanding = marketCap / last_price (as a fallback).
    Returns a dict ticker -> sharesOutstanding (float or NaN).
    """
    res: Dict[str, float] = {}
    for i, t in enumerate(tickers, start=1):
        if i % 50 == 0 or i == 1 or i == len(tickers):
            print(f"  fetching sharesOutstanding {i}/{len(tickers)}: {t}")
        try:
            info = yf.Ticker(t).info or {}
            val = info.get('sharesOutstanding', None)
            if val is None:
                # fallback: try to estimate from marketCap / last price if possible
                market_cap = info.get('marketCap', None)
                est = float('nan')
                if market_cap is not None and price_map is not None:
                    ser = price_map.get(t)
                    if ser is not None and not ser.empty:
                        try:
                            last_price = float(ser.iloc[-1])
                            if last_price > 0:
                                est = float(market_cap) / last_price
                        except Exception:
                            est = float('nan')
                if est == est and not math.isnan(est):
                    res[t] = est
                else:
                    res[t] = float('nan')
            else:
                res[t] = float(val)
        except Exception:
            res[t] = float('nan')
        time.sleep(0.03)
    return res


# ---------- robust next_trading_price ----------
def _ensure_series(obj) -> pd.Series:
    """Normalize DataFrame/Series to a 1-d Series of prices."""
    if obj is None:
        return pd.Series(dtype=float)
    if isinstance(obj, pd.DataFrame):
        # prefer Adj Close -> Close -> first column
        for col in ['Adj Close', 'Adj_Close', 'Close', 'close']:
            if col in obj.columns:
                s = obj[col]
                s.index = pd.to_datetime(s.index)
                return s.sort_index()
        # fallback to first column
        s = obj.iloc[:, 0]
        s.index = pd.to_datetime(s.index)
        return s.sort_index()
    if isinstance(obj, pd.Series):
        s = obj.copy()
        s.index = pd.to_datetime(s.index)
        return s.sort_index()
    # try to coerce
    try:
        s = pd.Series(obj)
        s.index = pd.to_datetime(s.index)
        return s.sort_index()
    except Exception:
        return pd.Series(dtype=float)


def next_trading_price(series_obj, target_dt: pd.Timestamp) -> Tuple[pd.Timestamp, float]:
    """
    Return (date, price) for first trading day >= target_dt, else (None,None).
    Accepts Series or DataFrame; normalizes internally.
    """
    s = _ensure_series(series_obj)
    if s.empty:
        return (None, None)
    pos = s.index.searchsorted(pd.to_datetime(target_dt))
    if pos >= len(s):
        return (None, None)
    # extract scalar safely
    try:
        val = s.iat[pos]
    except Exception:
        # fallback for odd structures
        try:
            val = s.iloc[pos]
        except Exception:
            return (None, None)
    try:
        return (s.index[pos], float(val))
    except Exception:
        return (s.index[pos], None)


# ---------- Core simulation: monthly DCA, dynamic ranking by mcap at each buy ----------
def simulate_topk_dca_monthly(
    constituents: List[str],
    shares_map: Dict[str, float],
    price_map: Dict[str, pd.Series],
    start_year: int,
    end_year: int,
    annual_investment: float,
    final_valuation_date: str = "2025-01-02"
) -> pd.DataFrame:
    """
    For each month (1..12) each year, compute market-cap on that month-1 (or the earliest trading day >= 1st)
    and allocate that month's amount (= annual_investment/12) across top-k by mcap (k=1..N).
    """
    years = list(range(start_year, end_year + 1))
    final_dt = pd.to_datetime(final_valuation_date)
    N = len(constituents)

    # state per k
    shares_owned = [defaultdict(float) for _ in range(N + 1)]
    total_invested = [0.0] * (N + 1)
    constituents_used = [set() for _ in range(N + 1)]

    # precompute final prices once
    final_price_map: Dict[str, float] = {}
    for t in constituents:
        d, px = next_trading_price(price_map.get(t, pd.Series(dtype=float)), final_dt)
        final_price_map[t] = px

    monthly_amt = annual_investment / 12.0

    # for each year/month: compute mcap for all constituents at that date, rank, then allocate
    for y in years:
        for m in range(1, 13):
            buy_dt = pd.Timestamp(year=y, month=m, day=1)
            # compute market caps at this buy_dt
            mcap_map: Dict[str, float] = {}
            price_at_date: Dict[str, float] = {}
            for t in constituents:
                ser = price_map.get(t, pd.Series(dtype=float))
                d, px = next_trading_price(ser, buy_dt)
                if d is None or px is None or px != px:
                    # no price at/after this date for this ticker -> skip
                    continue
                shares_out = shares_map.get(t, float('nan'))
                if shares_out is None or (isinstance(shares_out, float) and math.isnan(shares_out)):
                    # missing sharesOutstanding -> cannot compute mcap here; skip
                    continue
                mcap = shares_out * px
                if mcap <= 0 or mcap != mcap:
                    continue
                mcap_map[t] = mcap
                price_at_date[t] = px
            if not mcap_map:
                # nothing tradable this month
                continue
            # rank tickers by mcap (desc)
            ranked = sorted(mcap_map.items(), key=lambda kv: kv[1], reverse=True)
            ranked_tickers = [kv[0] for kv in ranked]
            L = len(ranked_tickers)
            # For each k, allocate monthly_amt across top min(k,L) tickers by mcap weights
            for k in range(1, N + 1):
                take = min(k, L)
                sel = ranked_tickers[:take]
                caps = np.array([mcap_map[t] for t in sel], dtype=float)
                if caps.sum() <= 0:
                    weights = np.ones_like(caps) / len(caps)
                else:
                    weights = caps / caps.sum()
                for i, t in enumerate(sel):
                    amt = monthly_amt * float(weights[i])
                    px = price_at_date[t]
                    if px is None or px != px or px <= 0:
                        continue
                    qty = amt / px
                    shares_owned[k][t] += qty
                    total_invested[k] += amt
                    constituents_used[k].add(t)

    # build results
    rows = []
    for k in range(1, N + 1):
        final_value = 0.0
        for t, qty in shares_owned[k].items():
            px = final_price_map.get(t, None)
            if px is None or (isinstance(px, float) and math.isnan(px)):
                # fallback to last available price
                ser = price_map.get(t, pd.Series(dtype=float))
                if ser is None or ser.empty:
                    continue
                try:
                    px = float(ser.iloc[-1])
                except Exception:
                    continue
            final_value += qty * px
        invested = total_invested[k]
        profit = final_value - invested
        roi_pct = (profit / invested * 100.0) if invested > 0 else float('nan')
        rows.append({
            'k': k,
            'num_stocks': k,
            'total_invested': invested,
            'final_value': final_value,
            'profit': profit,
            'roi_pct': roi_pct,
            'constituents_used': ", ".join(sorted(constituents_used[k]))
        })
    df = pd.DataFrame(rows)
    return df


# ---------- Main ----------
def main():
    p = argparse.ArgumentParser(description="Top-k market-cap monthly DCA simulation (monthly buys on each 1st)")
    p.add_argument("--annual-investment", type=float, default=36500.0)
    p.add_argument("--start-year", type=int, default=2004)
    p.add_argument("--end-year", type=int, default=2024)
    p.add_argument("--final-date", type=str, default="2025-01-02")
    p.add_argument("--max-sp500", type=int, default=500)
    p.add_argument("--max-nasdaq", type=int, default=100)
    args = p.parse_args()

    print("1) Fetching constituents...")
    sp500 = get_sp500_constituents()[:args.max_sp500]
    nas100 = get_nasdaq100_constituents()[:args.max_nasdaq]
    print(f"  S&P500: {len(sp500)}, NASDAQ100: {len(nas100)}")

    all_tickers = sorted(set(sp500 + nas100))

    print("\n2) Downloading historical prices (this can take several minutes)...")
    price_map = batch_download(all_tickers, start=f"{args.start_year}-01-01", end=args.final_date, pause=0.01)

    print("\n3) Retrieving sharesOutstanding (used to compute market-cap at each buy date)...")
    # <-- Pass price_map so we can estimate sharesOutstanding from marketCap / last_price when missing
    shares_map = get_shares_outstanding(all_tickers, price_map)

    print("\n4) Simulating S&P500 monthly DCA (k=1..N)...")
    df_sp = simulate_topk_dca_monthly(sp500, shares_map, price_map,
                                      args.start_year, args.end_year,
                                      args.annual_investment, final_valuation_date=args.final_date)

    # Pretty print top 20 with separated blocks (one block per result):
    top_sp = df_sp.sort_values('profit', ascending=False).head(20).reset_index(drop=True)

    print("\n" + "=" * 90)
    print("📊 S&P500: Top 20 by Profit (monthly top-k mcap DCA)")
    print("=" * 90)
    for idx, row in top_sp.iterrows():
        k = int(row['k'])
        invested = float(row['total_invested'])
        final_value = float(row['final_value'])
        profit = float(row['profit'])
        roi_pct = float(row['roi_pct']) if not pd.isna(row['roi_pct']) else float('nan')
        constituents = row.get('constituents_used', "")
        # safe count of constituents
        if isinstance(constituents, str) and constituents.strip():
            cons_list = [c.strip() for c in constituents.split(',') if c.strip()]
            cons_count = len(cons_list)
        else:
            cons_list = []
            cons_count = 0

        print(f"\n--- Rank {idx+1} (k={k}) ---")
        print(f"Total invested: {invested:,.2f}   Final value: {final_value:,.2f}   Profit: {profit:,.2f}   ROI%: {roi_pct:.2f}")
        print(f"Constituents used ({cons_count}):")
        # print constituents in a wrapped form for readability
        if cons_count == 0:
            print("  (none)")
        else:
            # print the list, wrapping lines at ~100 chars
            line = "  "
            for i, c in enumerate(cons_list):
                item = (c + (", " if i < len(cons_list)-1 else ""))
                if len(line) + len(item) > 100:
                    print(line.rstrip())
                    line = "  " + item
                else:
                    line += item
            if line.strip():
                print(line.rstrip())
    print("\n" + "=" * 90 + "\n\n")

    print("5) Simulating NASDAQ100 monthly DCA (k=1..N)...")
    df_nq = simulate_topk_dca_monthly(nas100, shares_map, price_map,
                                      args.start_year, args.end_year,
                                      args.annual_investment, final_valuation_date=args.final_date)

    top_nq = df_nq.sort_values('profit', ascending=False).head(20).reset_index(drop=True)

    print("\n" + "=" * 90)
    print("📈 NASDAQ100: Top 20 by Profit (monthly top-k mcap DCA)")
    print("=" * 90)
    for idx, row in top_nq.iterrows():
        k = int(row['k'])
        invested = float(row['total_invested'])
        final_value = float(row['final_value'])
        profit = float(row['profit'])
        roi_pct = float(row['roi_pct']) if not pd.isna(row['roi_pct']) else float('nan')
        constituents = row.get('constituents_used', "")
        if isinstance(constituents, str) and constituents.strip():
            cons_list = [c.strip() for c in constituents.split(',') if c.strip()]
            cons_count = len(cons_list)
        else:
            cons_list = []
            cons_count = 0

        print(f"\n--- Rank {idx+1} (k={k}) ---")
        print(f"Total invested: {invested:,.2f}   Final value: {final_value:,.2f}   Profit: {profit:,.2f}   ROI%: {roi_pct:.2f}")
        print(f"Constituents used ({cons_count}):")
        if cons_count == 0:
            print("  (none)")
        else:
            line = "  "
            for i, c in enumerate(cons_list):
                item = (c + (", " if i < len(cons_list)-1 else ""))
                if len(line) + len(item) > 100:
                    print(line.rstrip())
                    line = "  " + item
                else:
                    line += item
            if line.strip():
                print(line.rstrip())
    print("\n" + "=" * 90 + "\n")

    # optionally save csvs
    if False:
        df_sp.to_csv("SP500_monthly_results.csv", index=False)
        df_nq.to_csv("NAS100_monthly_results.csv", index=False)


if __name__ == "__main__":
    main()
