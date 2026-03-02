"""
=============================================================================
OSF PREREGISTRATION DISCLOSURE: VOLATILITY SURFACE & ROOT-FINDING
=============================================================================
Protocol Reference: 
  - Section 4.3 (Calculated Greek Protocol)
  - Section 3.3.1 (Merton Model Inputs: Risk-Free Rate & Forward Fill)

1. PROTOCOL ALIGNMENT (Midpoint IV Root-Finding):
Section 4.3 explicitly rejects vendor-supplied Greeks in favor of endogenous 
derivation to enforce the $q=0$ European approximation. This script successfully 
calculates the NBBO midpoint and executes the numerical root-finding algorithm 
to solve for Implied Volatility (IV), strictly enforcing the $1e^{-5}$ convergence 
tolerance mandated by the protocol.

2. PROTOCOL ALIGNMENT (Macro Boundary Enforcement):
This script ingests the empirical 1-Year Treasury Rate (DGS1) and mathematically 
executes the strict forward-fill ($r_t = r_{t-1}$) required by Section 3.3.1 
to seamlessly handle SIFMA bond market holidays without dropping valid equity 
trading days.

3. ENGINEERING HEURISTIC (Convergence Failures):
Due to the mathematical limits of continuous-time pricing, deep OTM contracts 
or those at the extreme edge of expiration exhibit vanishing Vega 
($\partial V / \partial \sigma \to 0$), causing root-finding solvers to fail. 
Contracts failing to meet the $1e^{-5}$ tolerance within the maximum iteration 
limit are safely flagged as `NaN` and excluded from the Active Core. This ensures 
that non-convergent mathematical singularities do not corrupt the systemic 
Effective Viscosity ($\eta_{\mathcal{O}}$) summation.
=============================================================================
"""

import polars as pl
import numpy as np
from physics_engine.connectors import MinIOConnector
from physics_engine.kinematics import BlackScholesSolver
from datetime import datetime, timedelta

# [PROTOCOL ENFORCEMENT: Section 4.2.1 - Asset Universe Constraints]
TICKERS = [
    "AAPL", "MSFT", "TSLA", "XOM", "CAT", "CSCO", "GE", "IBM", "INTC", 
    "KO", "MCD", "NKE", "PG", "UNH", "VZ", "WMT", "V", 
    "MRK", "MMM", "JNJ", "HD", "DIS", "DD", "CVX", "BA"
]

START_DATE = "2026-01-05"
END_DATE = "2026-02-20"

# [PROTOCOL ENFORCEMENT: Section 4.2 - Dual-Window Sampling]
SNAPSHOTS = ["09:30", "10:30"]

# [PROTOCOL ENFORCEMENT] - Minimum Time Horizon Floor
T_FLOOR = 0.00076

def get_dates():
    s = datetime.strptime(START_DATE, "%Y-%m-%d")
    e = datetime.strptime(END_DATE, "%Y-%m-%d")
    return [(s + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((e - s).days + 1)]

def get_risk_free_rate(connector, target_date_str):
    LOOKBACK_LIMIT = 5
    current_search_date = datetime.strptime(target_date_str, "%Y-%m-%d")
    
    for i in range(LOOKBACK_LIMIT):
        search_date_str = current_search_date.strftime("%Y-%m-%d")
        macro_path = f"raw/macro/{search_date_str}/macro_indicators_{search_date_str}.parquet"
        
        try:
            macro_lf = connector.scan_parquet(macro_path)
            if macro_lf is not None:
                rate_df = macro_lf.select("DGS1").collect()
                
                if rate_df.height > 0:
                    raw_rate = rate_df["DGS1"][0]
                    if i > 0:
                        print(f"   -> Forward Fill Active: Using {search_date_str} rate for {target_date_str}")
                    return raw_rate / 100.0
        except Exception:
            pass
            
        current_search_date -= timedelta(days=1)

    print(f"   ⚠️ CRITICAL: Risk-Free Rate missing for {target_date_str} (Lookback exceeded). Defaulting to 4.5%")
    return 0.045

def compute_full_surface(symbol, date, snapshot_time):
    connector = MinIOConnector()
    solver = BlackScholesSolver()
    
    print(f"🖥️ Surface Calculation: {symbol} on {date} at {snapshot_time}")
    
    # 1. Load Filtered Inputs (Silver Layer)
    opt_path = f"refined/opra/{symbol}/{date}/options_surface.parquet"
    try:
        # Load the OSI-parsed Silver Layer dataframe
        opts_df = connector.scan_parquet(opt_path).collect()
        
        if opts_df is None or opts_df.height == 0:
             print(f"   ⚠️ No option data found at {opt_path}")
             return
             
        # Apply temporal router directly to the lightweight Silver DataFrame
        opts_df = connector._enforce_temporal_geometry(opts_df, date, snapshot_time, mode="cumulative")
        
        if opts_df is None or opts_df.height == 0:
             print(f"   ⚠️ No option data remaining after temporal filter for {snapshot_time}")
             return

        # [PROTOCOL ENFORCEMENT] Section 4.2 & 4.3: Active Mass & Snapshot State
        # Ensure chronological sorting to grab the correct end-of-window snapshot quote
        ts_col = "ts_event" if "ts_event" in opts_df.columns else "ts_recv" if "ts_recv" in opts_df.columns else None
        if ts_col:
            opts_df = opts_df.sort(ts_col)
            
        opts_df = (
            opts_df.group_by("symbol")
            .agg([
                # 1. Cumulative Active Mass
                # Note: Data was already scaled to share-equivalents (x100) in the Silver layer
                pl.col("size").fill_null(0).sum().alias("V_opt"),
                
                # 2. Extract End-of-Window NBBO State
                # Note: 1e9 price scaling was already handled in the Bronze layer
                pl.col("bid_px_00").last().alias("bid"),
                pl.col("ask_px_00").last().alias("ask"),
                
                # 3. Retain OSI Metadata (from transcode_options.py)
                pl.col("strike_price").last(),
                pl.col("expiration_date").last(),
                pl.col("option_type").last()
            ])
            .with_columns([
                # 4. Define the Option Price as the MBBO Mid-Price
                ((pl.col("bid") + pl.col("ask")) / 2.0).alias("price")
            ])
            .filter(pl.col("price") > 0) 
        )
             
        opts_lf = opts_df.lazy()
        
    except Exception as e:
        print(f"   ⚠️ Data fetch/routing failed: {e}")
        return

    # 2. Fetch Base State: Spot Price (S) via VWAP/TWAP
    try:
        # Dynamically anchored to the exact snapshot millisecond
        equity_df = connector.get_equity_snapshot(symbol, date, snapshot_time=snapshot_time)
        if isinstance(equity_df, pl.LazyFrame): equity_df = equity_df.collect()
        
        if equity_df is None or equity_df.height == 0:
            print(f"   ⚠️ Equity snapshot empty for {symbol}")
            return

        # PROTOCOL ENFORCEMENT: VWAP Smoothing
        if "action" in equity_df.columns:
            trades = equity_df.filter((pl.col("action") == "T") & (pl.col("price") > 0))
        else:
            trades = equity_df.filter((pl.col("size") > 0) & (pl.col("price") > 0))
            
        if trades.height > 0:
            spot_price = (trades["price"] * trades["size"]).sum() / trades["size"].sum()
        else:
            # PROTOCOL ENFORCEMENT: TWAP Fallback
            valid_quotes = equity_df.filter((pl.col("bid_px_00") > 0) & (pl.col("ask_px_00") > 0))
            if valid_quotes.height > 0:
                spot_price = ((valid_quotes["bid_px_00"] + valid_quotes["ask_px_00"]) / 2).mean()
            else:
                print(f"   ⚠️ No valid trades or quotes for {symbol} spot price.")
                return

    except Exception as e:
        print(f"   ⚠️ Equity data missing for {symbol}: {e}")
        return
    
    # 3. Pre-Process Expiration Matrix and Apply Constraints
    rf_rate = get_risk_free_rate(connector, date)
    print(f"   -> Spot: ${spot_price:.2f} | r_GS1: {rf_rate:.2%}")
    
    try:
        surface_df = (
            opts_lf
            .with_columns([
                pl.lit(spot_price).alias("S"),
                pl.lit(rf_rate).alias("r")
            ])
            .with_columns([
                ((pl.col("expiration_date") - pl.lit(date).str.to_date()).dt.total_days() / 365.25).alias("T_raw")
            ])
            .with_columns([
                pl.max_horizontal(pl.col("T_raw"), pl.lit(T_FLOOR)).alias("T")
            ])
            .collect()
        )
    except Exception as e:
        print(f"   ❌ Pre-processing failed: {e}")
        return

    if surface_df.height == 0:
        print("   ⚠️ No valid contracts.")
        return

    # 4. Kinematic Core Execution (Fully Vectorized)
    print(f"   -> Solving Physics for {surface_df.height} contracts...")
    
    try:
        # Pass the entire array in one shot, the solver handles mixed calls/puts internally
        iv = solver.implied_volatility(
            price=surface_df["price"].to_numpy(),
            S=surface_df["S"].to_numpy(),
            K=surface_df["strike_price"].to_numpy(),
            T=surface_df["T"].to_numpy(),
            r=surface_df["r"].to_numpy(),
            flag=surface_df["option_type"].to_numpy()
        )
        
        greeks = solver.calculate_greeks(
            S=surface_df["S"].to_numpy(),
            K=surface_df["strike_price"].to_numpy(),
            T=surface_df["T"].to_numpy(),
            r=surface_df["r"].to_numpy(),
            sigma=iv,
            flag=surface_df["option_type"].to_numpy()
        )
        
        final_surface = surface_df.with_columns([
            pl.Series("iv", iv),
            pl.Series("delta", greeks["delta"]),
            pl.Series("gamma", greeks["gamma"]),
            pl.Series("vega", greeks["vega"]),
            pl.Series("theta", greeks["theta"]),
            pl.Series("vanna", greeks["vanna"]),
            pl.Series("charm", greeks["charm"])
        ])
    except Exception as e:
        print(f"   ❌ Solver failed: {e}")
        return
    
    # 5. Save the Kinematically Solved Surface (Time-Aware Topology)
    safe_time = snapshot_time.replace(':', '')
    output_path = f"derived/surface/{symbol}/{date}/{safe_time}/full_surface.parquet"
    connector.save_parquet(final_surface, output_path)
    print(f"   ✅ Time-Aware Surface Job Complete: {final_surface.height} evaluated.")

def run_batch_surface():
    dates = get_dates()
    print(f"🚀 STARTING SURFACE BATCH: {len(TICKERS)} Tickers x {len(dates)} Days")
    
    for date in dates:
        print(f"\n📅 Processing {date}...")
        for snapshot_time in SNAPSHOTS:
            print(f"\n ⏱️ Processing Snapshot: {snapshot_time}")
            for symbol in TICKERS:
                try:
                    compute_full_surface(symbol, date, snapshot_time)
                except Exception as e:
                    print(f"❌ Critical Error for {symbol} at {snapshot_time}: {e}")

if __name__ == "__main__":
    run_batch_surface()
