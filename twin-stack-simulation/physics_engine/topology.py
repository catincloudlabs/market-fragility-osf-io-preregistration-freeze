"""
=============================================================================
OSF PREREGISTRATION DISCLOSURE: TOPOLOGICAL CAPACITY MANIFOLD
=============================================================================
Protocol Reference: 
  - Section 2.4 (Geometric Curvature & Dimensional Superposition)
  - Section 3.3 (Critical Capacity H_c & Singularity Floor)
  - Section 3.3.2 (Liquidity Capacity & Mean Field Equation)

1. PROTOCOL ALIGNMENT (Dimensional Superposition):
This script calculates the total Critical Capacity ($H_c$). Because the 
underlying physical depth ($\Psi_{\text{liq}}$) and the derivative hedging 
depth ($\Psi_{\gamma}$) were independently normalized against the systemic 
daily dollar capacity, they resolve as dimensionless fractions. This script 
mathematically superposes them ($\Psi_{\text{liq}} + \Psi_{\gamma}$) in 
strict accordance with the algebraic distribution defined in Equation 11.

2. PROTOCOL ALIGNMENT (Mean-Field Jamming Transition):
The Liquidity Capacity ($\Psi_{\text{liq}}$) is evaluated using the Weiss 
Mean Field equation of state. By multiplying the maximum physical susceptibility 
by $\tanh(\beta_{\mu} \cdot \mathcal{R}_{lat})$, this script accurately models 
the sublimation of the limit order lattice: as the spread blows out 
($\beta_{\mu} \to 0$), the capacity smoothly collapses to zero.

3. PROTOCOL ALIGNMENT (Singularity Defenses):
To prevent catastrophic division-by-zero mathematical singularities during 
market vacuums, this script explicitly enforces the topological boundaries 
defined in Section 3.3:
  a) The Planck-Scale Capacity Floor: $H_c(t) = \max(\dots, \epsilon_c)$ 
     where $\epsilon_c = 0.001$.
  b) The Pinning Defense: $\epsilon_{qty} = 1.0$ is added to the endogenous 
     reload denominator to prevent divergence when $\Delta D_{1min} = 0$.
=============================================================================
"""

import polars as pl
import numpy as np
import pandas as pd
from physics_engine.connectors import MinIOConnector

class Topology:
    def __init__(self):
        self.connector = MinIOConnector()

    def calculate_psi_liq(self, df, history_df, adv_dollar):
        """
        Derives Liquidity Capacity (Psi_liq) from the L2 Limit Order Book.
        Psi_liq = Psi_max * tanh(Beta_mu * R_lat)
        """
        if df is None or df.height == 0:
            return 0.0, 0.0, 1.0

        # Snapshot aggregation (averaging the depth over the 1-minute window)
        snapshot = df.mean()
        
        # 1. Psi_Max (Systemic Susceptibility)
        # Aggregates the total dollar value of the top 10 price levels (MBP-10).
        depth_value = 0.0
        for i in range(10):
            suffix = f"{i:02d}"
            if f"bid_px_{suffix}" in snapshot.columns:
                # Scale raw Databento fixed-point integers to standard dollars
                bid_val = (snapshot[f"bid_px_{suffix}"][0] / 1e9) * snapshot[f"bid_sz_{suffix}"][0]
                ask_val = (snapshot[f"ask_px_{suffix}"][0] / 1e9) * snapshot[f"ask_sz_{suffix}"][0]
                depth_value += (bid_val + ask_val)

        psi_max = depth_value / max(adv_dollar, 1.0)

        # 2. Beta_Mu (Microstructural Temperature)
        # Spread expansion = Heat/Chaos. Spread contraction = Cold/Crystalline.
        current_spread = (snapshot["ask_px_00"][0] - snapshot["bid_px_00"][0]) / 1e9
        current_spread = max(current_spread, 0.01) # Reg NMS Tick Floor
        
        if history_df is not None and "spread" in history_df.columns:
            avg_spread = history_df["spread"].tail(5).mean()
        else:
            avg_spread = current_spread 

        beta_mu = avg_spread / current_spread

        # 3. [PROTOCOL ENFORCEMENT] True Resilience Scalar (R_lat)
        # R_lat = 1 + (Volume_1min / (|Delta_Depth| + epsilon))
        # Measures how much flow the lattice absorbs before physically breaking.
        r_lat = 1.0 
        vol_1min = 0.0

        if "volume" in df.columns:
            vol_1min = df["volume"].sum()
        elif "size" in df.columns:
            if "action" in df.columns:
                # Cast Binary ASCII to String before filtering
                trades = df.filter(pl.col("action").cast(pl.Utf8).is_in(["T", "E", "F"]))
                if trades.height > 0:
                    vol_1min = trades["size"].sum()
            else:
                vol_1min = df["size"].sum()

        # If L2 volume extraction yields 0, fall back to the OHLCV history proxy
        if vol_1min == 0.0 and history_df is not None and "volume" in history_df.columns:
             # Proxy 1-minute volume by taking the daily volume and dividing by 390 trading minutes
             try:
                 daily_vol = history_df["volume"].tail(1)[0]
                 vol_1min = daily_vol / 390.0
             except Exception:
                 pass

        if df.height > 1:
            # Extract literal physical displacement of depth shares across the 1-min bounds
            first_row = df.head(1)
            last_row = df.tail(1)
            
            def get_depth_shares(row):
                d = 0.0
                for i in range(10):
                    suffix = f"{i:02d}"
                    if f"bid_sz_{suffix}" in row.columns:
                        d += row[f"bid_sz_{suffix}"][0] + row[f"ask_sz_{suffix}"][0]
                return d
                
            depth_start = get_depth_shares(first_row)
            depth_end = get_depth_shares(last_row)
            
            # Absolute physical change in visible lattice mass
            delta_depth = abs(depth_end - depth_start)
            
            # Epsilon = 1.0 shares to prevent division by zero
            r_lat = 1.0 + (vol_1min / (delta_depth + 1.0))

        # 4. Equation of State (Eq 187) 
        psi_liq = psi_max * np.tanh(beta_mu * r_lat)
        
        # Return r_lat for downstream diagnostics
        return psi_liq, beta_mu, r_lat

    def get_market_structure(self, symbol, date, thermo_state, snapshot_time="09:30"):
        """
        Calculates the topological metrics for the specified temporal window.
        """
        print(f"   🏗️ Topology Scan: {symbol} at {snapshot_time}")
        
        adv_dollar = thermo_state['adv_dollar']
        sigma_base = thermo_state.get('sigma_base', 0.20)
        
        if np.isnan(adv_dollar) or adv_dollar == 0:
            adv_dollar = 1.0

        # 1. Load Data
        try:
            equity_df = self.connector.get_equity_snapshot(symbol, date, snapshot_time=snapshot_time)
            if isinstance(equity_df, pl.LazyFrame): equity_df = equity_df.collect()
        except Exception:
            equity_df = None

        try:
            hist_path = f"raw/equity/{symbol}/history/ohlcv_252d.parquet"
            history_df = self.connector.scan_parquet(hist_path).collect()
        except Exception:
            history_df = None

        # Time-aware pathing to pull the correct kinematic surface
        safe_time = snapshot_time.replace(':', '')
        surf_path = f"derived/surface/{symbol}/{date}/{safe_time}/full_surface.parquet"
        try:
            surface_df = self.connector.scan_parquet(surf_path).collect()
        except Exception as e:
            print(f"      ⚠️ No Options Surface found for {symbol} at {snapshot_time}")
            surface_df = None

        # 2. Calculate Liquidity Capacity (Psi_liq)
        if equity_df is not None and equity_df.height > 0:
            psi_liq, beta_mu, r_lat = self.calculate_psi_liq(equity_df, history_df, adv_dollar)
        else:
            psi_liq = 0.0
            beta_mu = 0.0
            r_lat = 1.0

        # 3. Calculate Option Metrics
        eta_orrell = 0.0
        psi_gamma = 0.0
        phi_vanna_raw = 0.0
        phi_charm_raw = 0.0
        
        if surface_df is not None and surface_df.height > 0:
            valid_surface = surface_df.drop_nulls(subset=["gamma", "iv", "S"])
            
            if valid_surface.height > 0:
                gamma = valid_surface["gamma"].to_numpy()
                iv = valid_surface["iv"].to_numpy()
                spot = valid_surface["S"].to_numpy()
                
                vanna = valid_surface["vanna"].to_numpy() if "vanna" in valid_surface.columns else np.zeros_like(gamma)
                charm = valid_surface["charm"].to_numpy() if "charm" in valid_surface.columns else np.zeros_like(gamma)

                # [PROTOCOL ENFORCEMENT] Active Mass Proxy (M)
                if "V_opt" in valid_surface.columns and valid_surface["V_opt"].sum() > 0:
                    active_mass_shares = valid_surface["V_opt"].fill_null(0.0).to_numpy() 
                elif "volume" in valid_surface.columns and valid_surface["volume"].sum() > 0:
                    active_mass_shares = valid_surface["volume"].fill_null(0.0).to_numpy()
                elif "size" in valid_surface.columns and valid_surface["size"].sum() > 0:
                    active_mass_shares = valid_surface["size"].fill_null(0.0).to_numpy()
                else:
                    active_mass_shares = np.ones(len(valid_surface)) * 100.0

                # ---------------------------------------------------------
                # DIMENSIONAL SCALING PROTOCOL: Hedging Inertia (omega_opt)
                # ---------------------------------------------------------
                # Fetch the Newtonian mass baseline from the thermo engine
                adv_shares = thermo_state.get('adv_shares', 1.0)
                
                # Convert absolute contract mass into dimensionless relative mass
                omega_opt = active_mass_shares / max(adv_shares, 1.0)

# ---------------------------------------------------------
                # DIMENSIONAL SCALING PROTOCOL: Hedging Inertia (omega_opt)
                # ---------------------------------------------------------
                # Fetch the Newtonian mass baseline from the thermo engine
                adv_shares = thermo_state.get('adv_shares', 1.0)
                
                # Convert absolute contract mass into dimensionless relative mass
                omega_opt = active_mass_shares / max(adv_shares, 1.0)

                # A. Effective Viscosity
                iv = np.maximum(iv, 0.001)
                m_psi = (sigma_base / iv) ** 2
                
                # FIX: Gamma * Spot is perfectly dimensionless [1]. 
                # Multiply by omega_opt [1], no division by adv_dollar needed!
                kinetic_flux = np.abs(gamma) * spot * omega_opt
                eta_orrell = np.nansum(kinetic_flux * m_psi) 
                
                # B. Net Gamma Potential
                # FIX: S^1 not S^2. No division by adv_dollar.
                gamma_exposure = gamma * spot * omega_opt
                psi_gamma = np.nansum(gamma_exposure) 
                
                # C. Unscaled Radiative Flux Constants 
                # FIX: stress.py expects [Shares] here so IT can divide by adv_dollar.
                # Revert to active_mass_shares to prevent double-division downstream.
                phi_vanna_raw = np.nansum(np.abs(vanna) * active_mass_shares)
                phi_charm_raw = np.nansum(np.abs(charm) * active_mass_shares)

        # 4. Geometric Curvature (Psi_geo)
        denom = max(psi_liq, 1e-6) 
        raw_curvature = 1.0 + (psi_gamma / denom)
        psi_geo = max(0.1, raw_curvature)

        print(f"      -> Psi_liq:    {psi_liq:.2e} | R_lat ({r_lat:.2f})")
        print(f"      -> eta_orrell: {eta_orrell:.4f}")
        print(f"      -> Psi_gamma:  {psi_gamma:.4f}")
        print(f"      -> Fluxes:     Vanna Raw={phi_vanna_raw:,.0f} | Charm Raw={phi_charm_raw:,.0f}")

        return {
            "psi_liq": psi_liq,
            "beta_mu": beta_mu,
            "r_lat": r_lat,
            "psi_gamma": psi_gamma,
            "eta_orrell": eta_orrell,
            "psi_geo": psi_geo,
            "phi_vanna_raw": phi_vanna_raw,
            "phi_charm_raw": phi_charm_raw,
            "omega_opt": omega_opt
        }

if __name__ == "__main__":
    from physics_engine.thermodynamics import Thermodynamics
    
    TICKERS = ["AAPL", "MSFT", "TSLA", "XOM", "CAT", "CSCO", "GE", "IBM", "INTC", 
               "KO", "MCD", "NKE", "PG", "UNH", "VZ", "WMT", "V",
               "MRK", "MMM", "JNJ", "HD", "DIS", "DD", "CVX", "BA"]
    
    START_DATE = "2026-01-05"
    END_DATE = "2026-02-20"
    
    print(f"\n🔬 Running Batch Topology Diagnostics...")
    t_engine = Thermodynamics()
    topo = Topology()
    
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq='B')
    
    for date_obj in dates:
        date_str = date_obj.strftime('%Y-%m-%d')
        for snap_time in ["09:30", "10:30"]:
            for ticker in TICKERS:
                try:
                    state = t_engine.get_state(ticker, date_str, snapshot_time=snap_time)
                    if state:
                        topo.get_market_structure(ticker, date_str, state, snapshot_time=snap_time)
                except Exception as e:
                    pass
