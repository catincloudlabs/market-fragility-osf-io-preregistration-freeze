"""
=============================================================================
OSF PREREGISTRATION DISCLOSURE: ENGINEERING PoC ARTIFACT
=============================================================================
Protocol Reference: Section 4.1 (Disclosure of Engineering Proof-of-Concept)

WARNING: This script is an early-stage engineering artifact used EXCLUSIVELY 
to generate the static `universe_2026.csv` for the computational out-of-sample 
Proof-of-Concept (PoC). 

KNOWN PROTOCOL DEVIATIONS:
This specific script iteration does NOT dynamically enforce the rigorous 
boundary conditions defined in Section 4.2 of the protocol, specifically:
  1. It does not programmatically exclude Financials/Real Estate (GICS 40/60).
  2. It does not force the inclusion of the Top 10 Nasdaq constituents.
  3. It does not execute the stratified random sampling (N=40) using the 
     preregistered seed (12345).

RATIONALE FOR INCLUSION:
In accordance with OSF transparency standards, this script is frozen "as-is" 
to maintain a mathematically honest audit trail of the exact PoC pipeline 
executed prior to preregistration. The static output (`universe_2026.csv`) 
is also included in this repository. 

The final production run for the 2019-2025 Validation Cohort will utilize 
an updated script that strictly enforces the Section 4.2 topology.
=============================================================================
"""

import norgatedata
import pandas as pd
import os
import sys

print(f"🔌 Connecting to Norgate Data Updater... (Lib: {norgatedata.__version__})")

# 1. Fetch Symbols
target_index = 'S&P 500'
try:
    symbols = norgatedata.watchlist_symbols(target_index)
    print(f"✅ Found {len(symbols)} constituents in {target_index}.")
except Exception as e:
    print(f"❌ Critical Error finding watchlist: {e}")
    sys.exit(1)

universe_data = []
print(f"\n⏳ Hydrating metadata for {len(symbols)} symbols...")

# 2. Main Loop (Corrected API Calls)
for sym in symbols:
    try:
        # Fetch Name
        name = norgatedata.security_name(sym)
        
        # --- Fetch GICS Sector ---
        # We ask for: Scheme='GICS', Format='Name', Level=1 (Sector)
        # If the asset is delisted or new, this might return None, so we handle it.
        try:
            sector = norgatedata.classification_at_level(sym, 'GICS', 'Name', 1)
            if not sector:
                sector = 'Unknown'
        except AttributeError:
            # Fallback if the method signature is different on this specific version
            sector = 'Error_MethodMissing'
        except Exception:
            sector = 'Unknown'

        # --- Fetch Exchange ---
        try:
            exchange = norgatedata.exchange_name(sym)
        except:
            exchange = 'Unknown'

        universe_data.append({
            'symbol': sym,
            'name': name,
            'sector': sector,
            'exchange': exchange
        })
        
    except Exception as e:
        print(f"⚠️ Skipping {sym}: {e}")

# 3. Export Logic
df = pd.DataFrame(universe_data)

# Validation Stats
unknown_sectors = len(df[df['sector'].isin(['Unknown', 'Error_MethodMissing'])])
nasdaq_count = len(df[df['exchange'] == 'Nasdaq'])

print(f"\n📊 Extraction Complete.")
print(f"   Total Assets: {len(df)}")
print(f"   Nasdaq Listed: {nasdaq_count}")
print(f"   Missing Sectors: {unknown_sectors}")

output_file = 'universe_2026.csv'
df.to_csv(output_file, index=False)
print(f"🚀 Success! Saved to {os.path.abspath(output_file)}")
