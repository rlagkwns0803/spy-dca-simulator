
# spy_dca_sim.py
# -------------------------------------------------------------
# SPY DCA backtester (2004–2024)
# - Input: buys per year (frequency), annual investment amount, etc.
# - Output: total profit over the period + useful performance metrics.
#
# Requirements:
#   pip install yfinance pandas numpy python-dateutil
#
# Notes:
# - Uses Adjusted Close (auto_adjust=True) from Yahoo via yfinance.
# - Adjusted Close approximates total-return pricing (dividends reinvested).
# - Commissions/taxes/slippage are ignored unless you set --commission.
# - IRR is computed with day-based compounding (XIRR); a robust bisection solver is included.
# -------------------------------------------------------------

import argparse
import datetime as dt
from dateutil.relativedelta import relativedelta
import numpy as np
import pandas as pd
import yfinance as yf

# ---------- Helpers ----------

def fetch_prices(ticker: str, start: str, end: str) -> pd.Series:
    """
    Download price data (Adjusted Close) for [start, end] inclusive trading days.
    """
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(f"No data returned for {ticker} in range {start} to {end}.")
    # With auto_adjust=True, 'Close' is adjusted. Prefer 'Close' if present.
    if 'Adj Close' in df.columns:
        # Some yfinance versions still include 'Adj Close'; keep consistent
        ser = df['Adj Close'].copy()
    else:
        ser = df['Close'].copy()
    ser.name = ticker
    return ser

def evenly_spaced_trading_days(prices: pd.Series, year: int, n: int) -> list:
    """
    Pick n nearly evenly spaced trading days from the given year.
    If n > number of trading days, cap at available days.
    """
    year_slice = prices.loc[str(year)]
    if year_slice.empty:
        return []
    n = int(n)
    if n <= 0:
        return []
    idxs = np.linspace(0, len(year_slice) - 1, min(n, len(year_slice)), dtype=int)
    return list(year_slice.index[idxs])

def monthly_last_trading_days(prices: pd.Series, year: int) -> list:
    """
    Last trading day of each month within the given year.
    """
    year_slice = prices.loc[str(year)]
    if year_slice.empty:
        return []
    # Resample by month end, grab last valid index per month
    months = year_slice.resample('M').last()
    dates = []
    for d in months.index:
        # 'd' is month-end calendar date; map to the last index <= d in year_slice
        sub = year_slice.loc[:d]
        if not sub.empty:
            dates.append(sub.index[-1])
    return dates

def pick_buy_dates(prices: pd.Series, year: int, buys_per_year: int, scheme: str) -> list:
    """
    Select buy dates for a given year according to scheme.
    """
    scheme = scheme.lower()
    if scheme == "evenly":
        return evenly_spaced_trading_days(prices, year, buys_per_year)
    elif scheme == "monthly":
        # Ignore buys_per_year; monthly implies up to 12 per year (existing months with data)
        return monthly_last_trading_days(prices, year)
    else:
        raise ValueError(f"Unknown scheme: {scheme}. Use 'evenly' or 'monthly'.")

def xnpv(rate: float, cashflows: list) -> float:
    """
    Compute NPV for irregular (date, amount) cashflows using day-count convention.
    rate is annual rate; cashflows is list of (date, amount).
    """
    if abs(rate + 1.0) < 1e-12:
        # Avoid division by zero in (1+rate)
        rate += 1e-8
    cashflows = sorted(cashflows, key=lambda x: x[0])
    t0 = cashflows[0][0]
    total = 0.0
    for (d, amt) in cashflows:
        days = (d - t0).days
        total += amt / (1.0 + rate) ** (days / 365.2425)
    return total

def xirr(cashflows: list, low: float = -0.9999, high: float = 10.0, tol: float = 1e-7, max_iter: int = 200) -> float:
    """
    Solve for IRR (annualized) via bisection on irregular cashflows.
    Returns a float (e.g., 0.08 for 8%). May raise ValueError if not bracketed.
    """
    f_low = xnpv(low, cashflows)
    f_high = xnpv(high, cashflows)
    if np.isnan(f_low) or np.isnan(f_high):
        raise ValueError("NPV calculation failed (NaN).")
    if f_low * f_high > 0:
        # Try expanding bounds once
        for factor in [2, 5, 10]:
            f_high = xnpv(high * factor, cashflows)
            if f_low * f_high <= 0:
                high *= factor
                break
        else:
            raise ValueError("Cannot bracket IRR; cashflows may not have a sign change.")
    a, b = low, high
    fa, fb = f_low, f_high
    for _ in range(max_iter):
        m = (a + b) / 2.0
        fm = xnpv(m, cashflows)
        if abs(fm) < tol:
            return m
        if fa * fm < 0:
            b, fb = m, fm
        else:
            a, fa = m, fm
    return (a + b) / 2.0

# ---------- Core Simulation ----------

def simulate_dca(
    ticker: str = "SPY",
    start_year: int = 2004,
    end_year: int = 2024,
    buys_per_year: int = 12,
    annual_investment: float = 12000.0,
    scheme: str = "evenly",
    commission: float = 0.0,
    seed: int | None = None,
) -> dict:
    """
    Simulate DCA for [start_year, end_year] inclusive using chosen buy schedule.
    Returns dict with summary and detailed transactions DataFrame.
    """
    if start_year > end_year:
        raise ValueError("start_year must be <= end_year.")
    if buys_per_year <= 0 and scheme == "evenly":
        raise ValueError("buys_per_year must be positive for 'evenly' scheme.")
    if annual_investment < 0:
        raise ValueError("annual_investment must be non-negative.")
    rng = np.random.default_rng(seed)

    start = f"{start_year}-01-01"
    end = f"{end_year}-12-31"
    prices = fetch_prices(ticker, start, end)

    all_buys = []
    cashflows = []  # (date, amount), negative for investments, positive for final liquidation
    total_shares = 0.0
    total_invested = 0.0

    for y in range(start_year, end_year + 1):
        # Determine buy dates for the year
        if scheme == "monthly":
            dates = pick_buy_dates(prices, y, buys_per_year, scheme="monthly")
            # If user asked buys_per_year < 12 for 'monthly', randomly select that many months (optional tweak)
            if buys_per_year and buys_per_year < len(dates):
                dates = list(rng.choice(dates, size=buys_per_year, replace=False))
                dates.sort()
        else:
            dates = pick_buy_dates(prices, y, buys_per_year, scheme="evenly")

        if not dates:
            continue

        per_buy = annual_investment / len(dates) if len(dates) > 0 else 0.0

        for d in dates:
            px = float(prices.loc[d])
            gross_amt = per_buy
            cost = gross_amt - commission
            if cost < 0:
                raise ValueError("Commission larger than per-buy amount.")
            shares = cost / px if px > 0 else 0.0

            total_shares += shares
            total_invested += gross_amt

            all_buys.append({
                "date": d,
                "price": px,
                "gross_invest": gross_amt,
                "commission": commission,
                "net_invest": cost,
                "shares_bought": shares,
                "cum_shares": total_shares,
                "cum_gross_invest": total_invested,
            })
            # Cashflow: investment is negative (outflow)
            cashflows.append((pd.Timestamp(d).to_pydatetime(), -gross_amt))

    # Final liquidation on the last trading day in end_year
    end_slice = prices.loc[str(end_year)]
    if end_slice.empty:
        raise ValueError(f"No trading days in {end_year} for {ticker}.")
    final_date = end_slice.index[-1]
    final_price = float(end_slice.iloc[-1])
    final_value = total_shares * final_price
    profit = final_value - total_invested

    # Positive cashflow on liquidation
    cashflows.append((pd.Timestamp(final_date).to_pydatetime(), final_value))

    # Try IRR; may fail if cashflows do not change sign
    irr = None
    try:
        irr = xirr(cashflows)
    except Exception as e:
        irr = None

    # Simple CAGR proxy (money-weighted CAGR is IRR; this is geometric from first to last cash date on gross invested)
    # When using ongoing contributions, IRR is the correct money-weighted measure.
    first_cf_date = cashflows[0][0] if cashflows else pd.Timestamp(f"{start_year}-01-01").to_pydatetime()
    last_cf_date = cashflows[-1][0] if cashflows else pd.Timestamp(f"{end_year}-12-31").to_pydatetime()
    years = (last_cf_date - first_cf_date).days / 365.2425
    cagr_proxy = None
    if total_invested > 0 and years > 0:
        cagr_proxy = (final_value / total_invested) ** (1 / years) - 1

    tx = pd.DataFrame(all_buys)
    summary = {
        "ticker": ticker,
        "start_year": start_year,
        "end_year": end_year,
        "buys_per_year": buys_per_year,
        "scheme": scheme,
        "annual_investment": annual_investment,
        "commission": commission,
        "final_date": pd.Timestamp(final_date).date().isoformat(),
        "final_price": final_price,
        "total_invested": total_invested,
        "final_value": final_value,
        "profit": profit,
        "irr_xirr": irr,
        "cagr_proxy": cagr_proxy,
        "years": years,
        "num_trades": len(tx),
    }
    return {"summary": summary, "transactions": tx}

# ---------- CLI ----------

def main():
    p = argparse.ArgumentParser(description="SPY DCA backtester (2004–2024)")
    p.add_argument("--ticker", type=str, default="SPY", help="Ticker (default: SPY)")
    p.add_argument("--start-year", type=int, default=2004, help="Start year (inclusive)")
    p.add_argument("--end-year", type=int, default=2024, help="End year (inclusive)")
    p.add_argument("--buys-per-year", type=int, default=12, help="Number of buys per year (evenly spaced)")
    p.add_argument("--annual-investment", type=float, default=12000.0, help="Total amount invested per year")
    p.add_argument("--scheme", type=str, default="evenly", choices=["evenly", "monthly"],
                   help="Buy schedule: 'evenly' (N evenly spaced trades per year) or 'monthly' (last trading day each month; optionally subsample to N)")
    p.add_argument("--commission", type=float, default=0.0, help="Commission per trade")
    p.add_argument("--seed", type=int, default=None, help="Random seed (for monthly subsampling)")
    p.add_argument("--export-csv", type=str, default=None, help="Path to save transactions CSV")
    args = p.parse_args()

    result = simulate_dca(
        ticker=args.ticker,
        start_year=args.start_year,
        end_year=args.end_year,
        buys_per_year=args.buys_per_year,
        annual_investment=args.annual_investment,
        scheme=args.scheme,
        commission=args.commission,
        seed=args.seed,
    )
    summary = result["summary"]
    tx = result["transactions"]

    print("=== Summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k:>18}: {v:,.6f}")
        else:
            print(f"{k:>18}: {v}")
    if args.export_csv:
        tx.to_csv(args.export_csv, index=False)
        print(f"\nTransactions saved to: {args.export_csv}")
    else:
        print("\n(Use --export-csv path/to/file.csv to save detailed transactions.)")

if __name__ == "__main__":
    main()
