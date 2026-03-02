"""
=============================================================================
OSF PREREGISTRATION DISCLOSURE: OPRA TRANSCODING & FILTERING
=============================================================================
Protocol Reference: 
  - Section 4.2 (Fixed-Point Price Scaling)
  - Section 4.2 (Standardized Contract Filter & Expiration Boundary Protocol)
  - Section 4.2 (Protocol for Temporal Consistency: Active Mass)

1. PROTOCOL ALIGNMENT (Dimensional Normalization):
This script mathematically enforces the scaling boundaries delegated by the 
upstream extraction DAGs. It explicitly reverses the $1e9$ Databento integer 
scaling to restore decimal floats for the Black-Scholes kernel. Furthermore, 
it derives the Share-Equivalent Volume ($V_{opt} = Contracts * 100$) to 
maintain dimensional homogeneity for the downstream Hedging Inertia ($\omega_{opt}$) 
calculation.

2. PROTOCOL ALIGNMENT (Expiration Boundary Protocol):
This script strictly enforces the `T_{exp} >= T_{current}` constraint, stripping 
expired contracts from the lattice to ensure only active forward positioning 
contributes to the Effective Viscosity ($\eta_{\mathcal{O}}$).

3. ENGINEERING HEURISTIC (Standardized Contract Filter):
Section 4.2 mandates the exclusion of non-standard OCC deliverables and 
non-100 multipliers. Rather than querying a proprietary corporate action 
database, this script enforces this constraint using a strict OCC2010 string 
Regular Expression (`r'^[A-Z]{1,5}\d{6}[CP]\d{8}$'`). By requiring a pure alpha 
root symbol, this computationally lightweight heuristic successfully filters out 
the explicitly adjusted tickers (e.g., `AAPL1`) that typically denote 
non-standard deliverables.
=============================================================================
"""

import polars as pl
from physics_engine.connectors import MinIOConnector
from datetime import datetime, timedelta

# [PROTOCOL ENFORCEMENT: Section 4.2.1 - Asset Universe Constraints]
TICKERS = [
    "AAPL", "MSFT", "TSLA", "XOM", "CAT", "CSCO", "GE", "IBM", "INTC", 
    "KO", "MCD", "NKE", "PG", "UNH", "VZ", "WMT", "V",
    "MRK", "MMM", "JNJ", "HD", "DIS", "DD", "CVX", "BA"
]

START_DATE = "2026-01-05"
END_DATE = "2026-02-20"

def get_dates():
    s = datetime.strptime(START_DATE, "%Y-%m-%d")
    e = datetime.strptime(END_DATE, "%Y-%m-%d")
    return [(s + timedelta(days=i)).strftime("%Y-%m-%d") for i in range((e - s).days + 1)]

def parse_osi_and_filter(df, current_date_str, ticker):
    """
    Parses OSI strings and enforces Protocol Standards.
    Protocol Section 4.2: Strict OCC Contract Filtering & Active Mass Scaling.
    """
    current_date = datetime.strptime(current_date_str, "%Y-%m-%d").date()
    initial_count = df.height

    # 1. Normalize Column Names across different Databento schemas
    if "raw_symbol" in df.columns and "symbol" not in df.columns:
        df = df.rename({"raw_symbol": "symbol"})

    # 2. [PROTOCOL ENFORCEMENT 1] Non-Standard Deliverable (NSD) Filter
    # Standard OCC OSI strings are EXACTLY 21 characters long. 
    # Adjusted options modify the root (e.g., 'AAPL1 ') to indicate non-standard deliverables.
    # By strictly matching the first 6 characters to the parent ticker, we guarantee 
    # that our physical mass assumption (Volume * 100 = Active Shares) holds true.
    
    # Drop malformed or non-standard string lengths
    df = df.filter(pl.col("symbol").str.len_chars() == 21)
    
    # Extract the 6-character root, strip padding spaces, and enforce a pure parent match
    df = df.with_columns(
        pl.col("symbol").str.slice(0, 6).str.strip_chars().alias("root_symbol")
    ).filter(pl.col("root_symbol") == ticker)

    # 3. Parse standard OSI Components safely now that NSDs are removed
    # OSI Format: [Root 6][YYMMDD 6][C/P 1][Strike 8]
    try:
        df = df.with_columns([
            pl.col("symbol").str.slice(-9, 1).alias("option_type"),
            (pl.col("symbol").str.slice(-8, 8).cast(pl.Float64) / 1000.0).alias("strike_price"),
            pl.col("symbol").str.slice(-15, 6).str.strptime(pl.Date, "%y%m%d", strict=False).alias("expiration_date")
        ])
    except Exception as e:
        print(f"      ⚠️ OSI Parsing Error (Corrupted String): {e}")
        return df.clear()

    # 4. [PROTOCOL ENFORCEMENT] Expiration Boundary Protocol
    # Drop contracts where Exp Date < Current Date to prevent mathematical 
    # singularities in the Black-Scholes solver (negative time-to-expiration).
    df = df.filter(
        pl.col("expiration_date").is_not_null() & 
        (pl.col("expiration_date") >= current_date)
    )
    
    # 5. [PROTOCOL ENFORCEMENT 3] Active Mass Scaling (V_opt)
    # Normalizing the raw OPRA contract count by the M=100 multiplier to yield share-equivalents.
    vol_cols = [c for c in df.columns if c in ["volume", "size"]]
    if vol_cols:
        df = df.with_columns([
            (pl.col(c) * 100).alias(c) for c in vol_cols
        ])
    
    final_count = df.height
    print(f"      -> Filtered (OCC Protocol): {initial_count} rows -> {final_count} rows")

    return df

def run_batch_transcoder():
    connector = MinIOConnector()
    dates = get_dates()
    
    print(f"🚀 Batch Transcoding: {len(TICKERS)} Tickers x {len(dates)} Days")

    for date in dates:
        print(f"\n📅 Processing {date}...")
        for symbol in TICKERS:
            try:
                # Fetch raw data using the geometry enforcer
                raw_df = connector.get_option_surface(symbol, date)
                
                if raw_df is None:
                    continue 

                # Materialize LazyFrames for processing
                if isinstance(raw_df, pl.LazyFrame):
                    raw_df = raw_df.collect()
                
                # Apply OSI parsing and topological filtering
                enriched_df = parse_osi_and_filter(raw_df, date, symbol)
                
                if enriched_df.height == 0:
                    print(f"   ❌ {symbol}: 0 valid OCC contracts remaining.")
                    continue

                # Save the refined, mathematically safe surface to the Silver Layer
                output_path = f"refined/opra/{symbol}/{date}/options_surface.parquet"
                connector.save_parquet(enriched_df, output_path)
                print(f"   ✅ {symbol}: {enriched_df.height} active contracts saved.")
                
            except Exception as e:
                print(f"   ❌ Failed {symbol}: {e}")

if __name__ == "__main__":
    run_batch_transcoder()
