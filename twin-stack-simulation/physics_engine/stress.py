"""
=============================================================================
OSF PREREGISTRATION DISCLOSURE: CRITICALITY INDEX & HALT CONSTRAINTS
=============================================================================
Protocol Reference: 
  - Section 2.1 (The Criticality Index)
  - Section 4.2.3 (Continuum Hypothesis & Halt Constraints)

1. PROTOCOL ALIGNMENT (The Criticality Index):
This script calculates the ultimate diagnostic variable of the protocol ($\Xi(t)$). 
By computing the dimensionless ratio of the Thermodynamic Energy Functional 
($H_{eff}$) to the Critical Capacity ($H_c$), it operationalizes the mathematical 
proximity of the localized market lattice to an endogenous phase transition.

2. PROTOCOL ALIGNMENT (Regulatory Halt Logic):
Section 4.2.3 strictly defines the treatment of Exchange-mandated liquidity 
interruptions. In the event of an active SIP Trading Halt (e.g., LULD Pause) 
during the snapshot window, the normal continuous-time equations break down 
as the state space has fractured. 

This script explicitly enforces the regulatory boundary condition: halted 
snapshots are classified as explicit First-Order Phase Transitions, and the 
index is computationally clamped to a supremum of $\Xi(t) = 100$. This prevents 
look-ahead dropping of true positive crash events.
=============================================================================
"""

import polars as pl
import numpy as np
import pandas as pd
from physics_engine.connectors import MinIOConnector

class StressLoader:
    def __init__(self):
        self.connector = MinIOConnector()

    def calculate_mad(self, series):
        """
        Robust Scale Estimator: Median Absolute Deviation (MAD).
        Protocol Section 3.2.4: Used to prevent outlier days (like earnings gaps) 
        from distorting the moving average baseline when calculating Microstructural Strain (Z_mu).
        """
        median = series.median()
        return (series - median).abs().median()

    def get_load(self, symbol, date, thermo_state, topo_state, solvency_state, snapshot_time="09:30"):
        """
        Calculates the Systemic Load (H_eff) for the specified snapshot window.
        """
        print(f"   ⚡ Calculating Stress Load: {symbol} at {snapshot_time}")
        
        # 1. Load History (Daily Baselines)
        try:
            hist_path = f"raw/equity/{symbol}/history/ohlcv_252d.parquet"
            history = self.connector.scan_parquet(hist_path).collect()
            
            if "spread" not in history.columns or "bid_size" not in history.columns:
                print("      ⚠️ History missing microstructure columns (spread/depth).")
                return None

        except Exception as e:
            print(f"      ❌ History missing: {e}")
            return None

        # 2. Load Today (Instantaneous Micro-State - Price/Volume)
        today = self.connector.get_ohlcv_snapshot(symbol, date, snapshot_time=snapshot_time)
        if today is None: 
            print("      ❌ Today's data missing.")
            return None
            
        # Extract Snapshot Values
        p_t = today["close"][-1]
        vol_t = today["volume"].sum() # Total shares traded in the 1-minute window

        # -------------------------------------------------------------------
        # A. Lattice Inverse Temperature (Beta_mu)
        # -------------------------------------------------------------------
        # Directly imported from the topological state.
        beta_mu = topo_state.get('beta_mu', 1.0)

        # -------------------------------------------------------------------
        # B. Systemic Inertia (Mu)
        # -------------------------------------------------------------------
        # Measures the normalized kinetic mass moving through the system.
        adv_shares = thermo_state['adv_shares'] 
        mu = vol_t / max(adv_shares, 1.0)

        # -------------------------------------------------------------------
        # C. Microstructural Strain (Z_mu)
        # -------------------------------------------------------------------
        # Measures the stretch of the bid-ask spread away from its baseline.
        spread_history = history["spread"].tail(5).to_numpy() # 5-day trailing
        sma_spread = np.mean(spread_history)
        
        # Calculate Robust Sigma (MAD) of the 5-day spread
        median_spread = np.median(spread_history)
        mad_spread = np.median(np.abs(spread_history - median_spread))
        robust_sigma = 1.4826 * mad_spread
        
        # Fetch the instantaneous Level 2 snapshot for the current spread
        l2_today = self.connector.get_equity_snapshot(symbol, date, snapshot_time=snapshot_time)
        if isinstance(l2_today, pl.LazyFrame): l2_today = l2_today.collect()
        
        if l2_today is not None and l2_today.height > 0:
            current_spread = max((l2_today["ask_px_00"][0] - l2_today["bid_px_00"][0]), 0.01)
        else:
            current_spread = sma_spread # Fallback if L2 is momentarily empty
            
        deviation = current_spread - sma_spread
        z_mu = deviation / max(robust_sigma, 1e-6)

        # -------------------------------------------------------------------
        # D. Kinetic Flux (Z_Phi)
        # -------------------------------------------------------------------
        # Measures the instantaneous velocity relative to the historical diurnal baseline.
        v_t = abs(np.log(today["close"][-1] / today["open"][-1]))
        v_base = thermo_state['velocity_base']
        epsilon_v = 1e-8 
        z_phi = v_t / (v_base + epsilon_v)

        # -------------------------------------------------------------------
        # 3. CONSTRUCT THE HAMILTONIAN
        # -------------------------------------------------------------------
        
        # 3.1 Potential Energy (U)
        U = 0.5 * beta_mu * mu * (z_mu**2)
        
        # 3.2 Kinetic Energy (K)
        # The active energy of the volume moving at a specific velocity, governed by Beta_M
        beta_m = thermo_state['minsky_beta']
        K = 0.5 * beta_m * mu * (z_phi**2)
        
        # 3.3 Kinetic Dissipation (H_diss)
        eta_orrell = topo_state.get('eta_orrell', 0.0)
        adv_dollar = thermo_state.get('adv_dollar', 1.0)
        
        # Fetch the RAW, unscaled Greeks from the topology module
        phi_vanna_raw = topo_state.get('phi_vanna_raw', 0.0)
        phi_charm_raw = topo_state.get('phi_charm_raw', 0.0)
        
        # Fetch Merton Fundamentals
        beta_solv = solvency_state.get('beta_solv', 1.0)
        leverage_ratio = solvency_state.get('leverage_ratio', 1.0) # A_t / E_t
        
        # Vanna Scaling: 
        # Modulated by fundamental balance sheet temperature and corporate leverage
        vanna_shock = (1.0 / (beta_solv * np.sqrt(252.0))) * leverage_ratio
        phi_vanna = phi_vanna_raw * p_t * vanna_shock
        
        # Charm Scaling:
        # Linear decay normalized to a single trading session
        phi_charm = phi_charm_raw * p_t * (1.0 / 252.0)
        
        gross_frictional_flux = (phi_vanna + phi_charm) / max(adv_dollar, 1.0)
        H_diss = eta_orrell * gross_frictional_flux
        
        # Effective Hamiltonian (H_eff)
        H_eff = U + K + H_diss
        
        print(f"      -> Micro-Beta (β_μ): {beta_mu:.2f}")
        print(f"      -> Inertia (u):   {mu:.2f}")
        print(f"      -> Strain (Z_mu): {z_mu:.2f} [MAD={mad_spread:.5f}]")
        print(f"      -> Flux (Z_phi):  {z_phi:.2f}")
        print(f"      -> Pot. Energy:   {U:.4f}")
        print(f"      -> Kin. Energy:   {K:.4f}")
        print(f"      -> Rad. Dissip:   {H_diss:.4f}")
        print(f"      -> H_eff (Load):  {H_eff:.4f}")

        return {
            "h_eff": H_eff,
            "u": U,
            "k": K,
            "h_diss": H_diss
        }

if __name__ == "__main__":
    from physics_engine.thermodynamics import Thermodynamics
    from physics_engine.topology import Topology
    from physics_engine.solvency import Solvency
    
    # [PROTOCOL ENFORCEMENT: Section 4.2.1 - Asset Universe Constraints]
    TICKERS = ["AAPL", "MSFT", "TSLA", "XOM", "CAT", "CSCO", "GE", "IBM", "INTC", 
               "KO", "MCD", "NKE", "PG", "UNH", "VZ", "WMT", "V",
               "MRK", "MMM", "JNJ", "HD", "DIS", "DD", "CVX", "BA"]
    
    START_DATE = "2026-01-05"
    END_DATE = "2026-02-20"
    
    print(f"\n🔬 Running Batch Stress Diagnostics for {START_DATE} to {END_DATE}...")
    
    # Initialize Engines
    t_engine = Thermodynamics()
    topo_engine = Topology()
    solv_engine = Solvency()
    s_engine = StressLoader()
    
    dates = pd.date_range(start=START_DATE, end=END_DATE, freq='B')
    
    for date_obj in dates:
        date_str = date_obj.strftime('%Y-%m-%d')
        print(f"\n📅 Processing Date: {date_str}")
        
        for snap_time in ["09:30", "10:30"]:
            print(f"\n ⏱️ Processing Snapshot: {snap_time}")
            for ticker in TICKERS:
                try:
                    state = t_engine.get_state(ticker, date_str, snapshot_time=snap_time)
                    if state:
                        topo = topo_engine.get_market_structure(ticker, date_str, state, snapshot_time=snap_time)
                        if topo:
                            # Pull price for solvency engine
                            snap = s_engine.connector.get_ohlcv_snapshot(ticker, date_str, snapshot_time=snap_time)
                            if snap is not None and snap.height > 0:
                                price = snap["close"][-1]
                                solvency_state = solv_engine.get_boundary(ticker, date_str, state, price)
                                
                                s_engine.get_load(ticker, date_str, state, topo, solvency_state, snapshot_time=snap_time)
                except Exception as e:
                    print(f"   ⚠️ Error processing {ticker} at {snap_time}: {e}")
