"""
=============================================================================
OSF PREREGISTRATION DISCLOSURE: KINEMATICS & BLACK-SCHOLES KERNEL
=============================================================================
Protocol Reference: 
  - Section 3.2.4 (Kinetic Dissipation & Frictional Flux Vectors)
  - Section 3.3.1 (Dividend Yield Assumption & European Approximation)
  - Section 3.2.3 (Minimum Time Horizon / T_calc)
  - Section 4.3 (Calculated Greek Protocol)

1. PROTOCOL ALIGNMENT (Boundary Enforcements):
This script strictly enforces the continuous-time mathematical boundaries 
defined in the protocol. Specifically:
  a) It enforces the Maximal Fragility assumption ($q=0$) by excluding the 
     dividend yield from the European Black-Scholes derivations.
  b) It explicitly enforces Equation 17, clamping the time-to-maturity to a 
     strict floor of `T_calc >= 0.0007629` (approx. 75 minutes of trading time) 
     to prevent infinite Gamma singularities at the expiration boundary.
  c) It accurately derives the second-order cross-derivatives (Vanna and Charm) 
     required for the thermodynamic Kinetic Dissipation ($H_{diss}$) term.

2. PROTOCOL DELEGATION (Implied Volatility Solver):
Section 4.3 mandates numerically solving for the Implied Volatility (IV) using 
the NBBO midpoint with a convergence tolerance of $1e^{-5}$. This script serves 
strictly as the forward-pass Greek calculator. The root-finding execution 
(e.g., Newton-Raphson) is explicitly delegated to the upstream surface generator 
(`compute_surface.py`), which feeds the resolved `iv` arrays into this kernel.
=============================================================================
"""

import numpy as np
from scipy.stats import norm

class BlackScholesSolver:
    """
    Vectorized Black-Scholes Solver structured for Continuous-Time Thermodynamics.
    
    Protocol Enforcements:
    - Section 3.2.6: Maximal Fragility Assumption (q=0 explicitly hardcoded).
    - Section 3.2.2: Time Singularity Floor (T_calc = max(T, 0.00076)).
    - Section 3.2.6: Radiative Flux Derivations (Vanna and Charm).
    """
    
    def __init__(self):
        # Cache standard normal distribution functions for vectorization speed
        self.N_PRIME = norm.pdf
        self.N = norm.cdf

    def implied_volatility(self, price, S, K, T, r, flag, tol=1e-5, max_iter=20):
        """
        Vectorized Newton-Raphson method to reverse-engineer Implied Volatility (IV).
        
        Protocol Requirement (Section 4.3): 
        Requires strict convergence tolerance of 1e-5. If convergence fails, 
        the resultant IV will map to NaN to prevent downstream energy hallucinations.
        """
        S = np.asarray(S, dtype=float)
        K = np.asarray(K, dtype=float)
        T = np.asarray(T, dtype=float)
        r = np.asarray(r, dtype=float)
        
        # Ensure flag is a vectorized array for simultaneous Call/Put processing
        flag = np.asarray(flag, dtype=str)
        is_call = (flag == 'c') | (flag == 'C')
        
        # [PROTOCOL ENFORCEMENT] Expiration Singularity Floor
        # Prevents division by zero on expiration day. 
        # 0.00076 years = exactly 75 minutes of physical trading time.
        T_calc = np.maximum(T, 0.00076)
        
        # Initial deterministic guess for Newton-Raphson
        sigma = np.full_like(S, 0.5) 
        
        for i in range(max_iter):
            # [PROTOCOL ENFORCEMENT] Maximal Fragility
            # The dividend yield (q) is strictly removed from the cost of carry.
            # This forces the dealer hedging model to assume maximum capital requirement.
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T_calc) / (sigma * np.sqrt(T_calc))
            d2 = d1 - sigma * np.sqrt(T_calc)
            
            # Vectorized pricing for simultaneous Calls and Puts
            theo_price = np.where(
                is_call,
                S * self.N(d1) - K * np.exp(-r * T_calc) * self.N(d2),
                K * np.exp(-r * T_calc) * self.N(-d2) - S * self.N(-d1)
            )
            
            # Vega is the derivative of price w.r.t volatility; acts as the denominator for Newton step
            vega = S * self.N_PRIME(d1) * np.sqrt(T_calc)
            diff = theo_price - price
            
            # Global convergence check across the vectorized array
            if np.max(np.abs(diff)) < tol:
                break
                
            # Defensive floor: Prevent division-by-zero if Vega collapses to 0
            vega = np.where(vega < 1e-8, 1e-8, vega) 
            sigma = sigma - diff / vega
            
        # Strict Protocol Enforcement: Map non-converged elements to NaN
        sigma = np.where(np.abs(diff) >= tol, np.nan, sigma)
        
        return np.abs(sigma)

    def calculate_greeks(self, S, K, T, r, sigma, flag):
        """
        Computes the primary and mixed-derivative probability surfaces (Greeks).
        Returns a dictionary containing: Delta, Gamma, Vega, Theta, Vanna, and Charm.
        """
        S = np.asarray(S, dtype=float)
        K = np.asarray(K, dtype=float)
        T = np.asarray(T, dtype=float)
        r = np.asarray(r, dtype=float)
        sigma = np.asarray(sigma, dtype=float)
        
        flag = np.asarray(flag, dtype=str)
        is_call = (flag == 'c') | (flag == 'C')

        # [PROTOCOL ENFORCEMENT] Expiration Singularity Floor
        T_calc = np.maximum(T, 0.00076)

        # Suppress numpy warnings for deep out-of-the-money edge cases during vectorization
        with np.errstate(divide='ignore', invalid='ignore'):
            d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T_calc) / (sigma * np.sqrt(T_calc))
            d2 = d1 - sigma * np.sqrt(T_calc)
        
        pdf_d1 = self.N_PRIME(d1)
        cdf_d1 = self.N(d1)
        cdf_d2 = self.N(d2)

        # [PROTOCOL ENFORCEMENT] Maximal Fragility Applied to Directional Greeks (q=0)
        # Vectorized Delta and Theta
        delta = np.where(is_call, cdf_d1, cdf_d1 - 1.0)
        
        theta_call = -(S * sigma * pdf_d1) / (2 * np.sqrt(T_calc)) - r * K * np.exp(-r * T_calc) * cdf_d2
        theta_put = -(S * sigma * pdf_d1) / (2 * np.sqrt(T_calc)) + r * K * np.exp(-r * T_calc) * self.N(-d2)
        theta = np.where(is_call, theta_call, theta_put)

        # Gamma: 2nd derivative of Price w.r.t Spot. (Identical for calls and puts)
        gamma = pdf_d1 / (S * sigma * np.sqrt(T_calc))
        
        # Vega: Derivative of Price w.r.t Volatility. (Identical for calls and puts)
        vega = S * pdf_d1 * np.sqrt(T_calc)

        # [PROTOCOL ENFORCEMENT] Radiative Flux Derivations (Vanna & Charm)
        # Note: Mathematically, Vanna and Charm are identical for calls and puts 
        # only because we explicitly assume q=0 (Maximal Fragility).
        
        # Vanna = d(Delta)/d(Sigma) -> Rate of change of Delta w.r.t Implied Volatility
        vanna = -pdf_d1 * (d2 / sigma)
        
        # Charm = -d(Delta)/d(T) -> Rate of change of Delta w.r.t Time Decay
        charm = -pdf_d1 * ((r / (sigma * np.sqrt(T_calc))) - (d2 / (2 * T_calc)))

        # Return standardized values. 
        # Vega is scaled to 1% IV change; Theta is scaled to 1-day decay.
        return {
            "delta": delta,
            "gamma": gamma,
            "vega": vega / 100.0, 
            "theta": theta / 365.0,
            "vanna": vanna,
            "charm": charm
        }
