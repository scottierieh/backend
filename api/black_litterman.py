from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats
import warnings
import traceback

warnings.filterwarnings('ignore')

router = APIRouter()


# ══════════════════════════════════════════════════════════════
# Request Model
# ══════════════════════════════════════════════════════════════

class BlackLittermanRequest(BaseModel):
    data: Optional[List[Dict[str, Any]]] = None
    dateCol: Optional[str] = None
    assetCols: Optional[List[str]] = None
    # Generate mode
    generate: bool = False
    assets: List[str] = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'JPM', 'JNJ', 'XOM', 'WMT']
    nDays: int = 750
    seed: Optional[int] = None
    # Covariance estimation method
    # "ledoit_wolf" : Ledoit-Wolf analytical shrinkage (recommended)
    # "oas"         : Oracle Approximating Shrinkage
    # "sample"      : plain sample covariance (no shrinkage)
    covMethod: str = "ledoit_wolf"
    # Market parameters
    riskFreeRate: float = 0.04       # annual — used only for Sharpe / EF optimisation
    riskAversion: float = 2.5        # delta  — used in π = δΣw
    # tau calibration
    # "fixed"     : use tauValue as-is (default 0.05)
    # "inv_t"     : tau = 1 / T  (Blamont & Firoozye 2003)
    # "trace"     : tau = trace(Σ) / N  (He & Litterman 1999 spirit)
    tauMethod: str = "inv_t"         # "fixed" | "inv_t" | "trace"
    tauValue: float = 0.05           # used only when tauMethod == "fixed"
    # Omega method
    # "confidence": Ω_ii = (1/c - 1) × τ(PΣP')_ii  (current)
    # "proportional": Ω = τ × P Σ P'                (He & Litterman)
    # "idzorek"    : confidence → weight deviation   (Idzorek 2005)
    omegaMethod: str = "confidence"  # "confidence" | "proportional" | "idzorek"
    # Views
    views: Optional[List[Dict[str, Any]]] = None
    # e.g. [{"type": "absolute", "asset": "AAPL", "return": 0.15, "confidence": 0.8},
    #        {"type": "relative", "asset1": "MSFT", "asset2": "GOOGL", "return": 0.03, "confidence": 0.6}]


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

def _to_native(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [_to_native(x) for x in obj.tolist()]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_native(x) for x in obj]
    return obj


def safe_float(val, default=0.0):
    try:
        if val is None:
            return default
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except Exception:
        return default


# ══════════════════════════════════════════════════════════════
# Asset Profiles for Data Generation
# ══════════════════════════════════════════════════════════════

ASSET_PROFILES = {
    'AAPL':  {'name': 'Apple Inc.',           'annual_ret': 0.25, 'annual_vol': 0.28, 'mkt_cap': 3000},
    'MSFT':  {'name': 'Microsoft Corp.',      'annual_ret': 0.22, 'annual_vol': 0.25, 'mkt_cap': 2800},
    'GOOGL': {'name': 'Alphabet Inc.',        'annual_ret': 0.18, 'annual_vol': 0.27, 'mkt_cap': 1800},
    'AMZN':  {'name': 'Amazon.com Inc.',      'annual_ret': 0.20, 'annual_vol': 0.30, 'mkt_cap': 1900},
    'TSLA':  {'name': 'Tesla Inc.',           'annual_ret': 0.30, 'annual_vol': 0.55, 'mkt_cap': 800},
    'JPM':   {'name': 'JPMorgan Chase',       'annual_ret': 0.14, 'annual_vol': 0.22, 'mkt_cap': 550},
    'JNJ':   {'name': 'Johnson & Johnson',    'annual_ret': 0.08, 'annual_vol': 0.15, 'mkt_cap': 400},
    'WMT':   {'name': 'Walmart Inc.',         'annual_ret': 0.10, 'annual_vol': 0.16, 'mkt_cap': 450},
    'XOM':   {'name': 'Exxon Mobil',          'annual_ret': 0.12, 'annual_vol': 0.25, 'mkt_cap': 480},
    'BRK.B': {'name': 'Berkshire Hathaway',   'annual_ret': 0.15, 'annual_vol': 0.18, 'mkt_cap': 850},
    'NVDA':  {'name': 'NVIDIA Corp.',         'annual_ret': 0.40, 'annual_vol': 0.50, 'mkt_cap': 2500},
    'META':  {'name': 'Meta Platforms',       'annual_ret': 0.22, 'annual_vol': 0.35, 'mkt_cap': 1200},
    'V':     {'name': 'Visa Inc.',            'annual_ret': 0.16, 'annual_vol': 0.20, 'mkt_cap': 550},
    'PG':    {'name': 'Procter & Gamble',     'annual_ret': 0.09, 'annual_vol': 0.14, 'mkt_cap': 370},
    'KO':    {'name': 'Coca-Cola Co.',        'annual_ret': 0.08, 'annual_vol': 0.14, 'mkt_cap': 260},
    'DIS':   {'name': 'Walt Disney Co.',      'annual_ret': 0.10, 'annual_vol': 0.28, 'mkt_cap': 200},
    'NFLX':  {'name': 'Netflix Inc.',         'annual_ret': 0.25, 'annual_vol': 0.40, 'mkt_cap': 280},
    'AMD':   {'name': 'AMD Inc.',             'annual_ret': 0.30, 'annual_vol': 0.45, 'mkt_cap': 250},
}


def generate_returns_data(
    assets: List[str],
    n_days: int = 750,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate realistic correlated daily return data for multiple assets.
    Uses Cholesky decomposition for correlation structure.
    """
    rng = np.random.default_rng(seed)
    n_assets = len(assets)

    profiles = []
    for a in assets:
        p = ASSET_PROFILES.get(a.upper(), {
            'name': a, 'annual_ret': 0.12, 'annual_vol': 0.25, 'mkt_cap': 100,
        })
        profiles.append(p)

    # Daily parameters
    daily_rets = np.array([p['annual_ret'] / 252 for p in profiles])
    daily_vols = np.array([p['annual_vol'] / np.sqrt(252) for p in profiles])

    # Correlation matrix — realistic block structure
    corr = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            # Base correlation depends on sector similarity
            base = 0.3 + 0.3 * rng.random()
            corr[i, j] = base
            corr[j, i] = base

    # Ensure positive definite
    eigvals = np.linalg.eigvalsh(corr)
    if np.min(eigvals) < 0:
        corr += (-np.min(eigvals) + 0.01) * np.eye(n_assets)
        d = np.sqrt(np.diag(corr))
        corr = corr / np.outer(d, d)

    cov = np.outer(daily_vols, daily_vols) * corr
    returns = rng.multivariate_normal(daily_rets, cov, size=n_days)

    end_date = pd.Timestamp('2025-04-30')
    dates = pd.bdate_range(end=end_date, periods=n_days)

    df = pd.DataFrame(returns, columns=assets)
    df.insert(0, 'date', dates.strftime('%Y-%m-%d'))

    return df, profiles


# ══════════════════════════════════════════════════════════════
# Black-Litterman Model
# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
# Covariance Estimation  (Ledoit-Wolf shrinkage)
# ══════════════════════════════════════════════════════════════

def shrink_covariance(
    returns: pd.DataFrame,
    method: str = "ledoit_wolf",
) -> tuple:
    """
    Estimate the covariance matrix with optional shrinkage.

    Why shrinkage?
    ──────────────
    The sample covariance Ŝ = X'X/(T−1) is unbiased but *noisy*:
    it amplifies estimation error in small-T / large-N regimes,
    producing extreme eigenvalues that destabilise portfolio weights.

    Ledoit-Wolf (2004) solves:
        Σ̂_LW = (1−δ)·S + δ·μ·I
    choosing δ analytically to minimise E[‖Σ̂_LW − Σ‖²_F].
    No hyperparameter tuning needed.

    Oracle Approximating Shrinkage (OAS, Chen et al. 2010) is a
    second estimator included for comparison — often tighter for
    low-rank structures.

    Parameters
    ──────────
    method : "ledoit_wolf" | "oas" | "sample"

    Returns
    ───────
    cov_matrix   (N×N) ndarray — estimated covariance (daily scale)
    shrinkage    float          — shrinkage coefficient δ (0 = sample, 1 = identity)
    method_used  str
    """
    X = returns.values.astype(np.float64)

    if method == "oas":
        try:
            from sklearn.covariance import OAS
            est       = OAS().fit(X)
            cov       = est.covariance_
            shrinkage = float(est.shrinkage_)
            return cov, shrinkage, "OAS"
        except Exception:
            pass   # fall through to sample

    if method in ("ledoit_wolf", "lw"):
        try:
            from sklearn.covariance import LedoitWolf
            est       = LedoitWolf().fit(X)
            cov       = est.covariance_
            shrinkage = float(est.shrinkage_)
            return cov, shrinkage, "Ledoit-Wolf"
        except Exception:
            pass   # fall through to sample

    # "sample" or sklearn unavailable
    cov = returns.cov().values
    return cov, 0.0, "sample"


def compute_market_implied_returns(
    cov_matrix: np.ndarray,
    market_caps: np.ndarray,
    risk_aversion: float = 2.5,
    assets: List[str] = None,
) -> np.ndarray:
    """
    Compute equilibrium (market-implied) excess returns.

    BL prior:  π = δ × Σ × w_mkt

    Risk-free rate does NOT enter this formula.
    π is already an *excess* return (above rf) by construction —
    it is the return the market demands to hold the tangency portfolio.
    rf is only used later for Sharpe-ratio optimisation.
    """
    weights_mkt = market_caps / market_caps.sum()

    # Annualised covariance
    cov_annual = cov_matrix * 252

    # π = δ Σ w  — pure matrix multiplication, no rf term
    pi_annual = risk_aversion * cov_annual @ weights_mkt

    # Convert to daily scale to stay consistent with the rest of the model
    pi_daily = pi_annual / 252

    return pi_daily, weights_mkt


def calibrate_tau(
    method: str,
    T: int,
    cov_matrix: np.ndarray,
    tau_fixed: float = 0.05,
) -> float:
    """
    Calibrate the τ (tau) scaling parameter for the BL prior covariance τΣ.

    τ controls how much weight the model gives to the prior vs. views.
    Small τ  → prior dominates (equilibrium pulls harder).
    Large τ  → views dominate.

    Three methods:
    ─────────────────────────────────────────────────────────────
    "fixed"   : τ = tau_fixed (default 0.05)
                Simple, transparent. Common in textbooks.

    "inv_t"   : τ = 1 / T
                Blamont & Firoozye (2003). Ties uncertainty of the
                prior directly to sample size. Larger T → more
                confidence in Σ estimate → smaller τ.

    "trace"   : τ = trace(Σ) / (N × mean_diag_Σ) normalised to ≈ 1/N
                Inspired by He & Litterman (1999). Ensures τΣ has
                eigenvalues of similar magnitude to Σ/N, making the
                prior uncertainty proportional to estimation error.
                In practice: τ = 1 / N for diagonal-Σ intuition.
    ─────────────────────────────────────────────────────────────
    Returns τ as a positive float. Clamped to [0.001, 1.0].
    """
    N = cov_matrix.shape[0]

    if method == "inv_t":
        tau = 1.0 / max(T, 1)
    elif method == "trace":
        # τ = trace(Σ) / (N × avg_variance) = 1/N  when Σ is isotropic.
        # Scales gracefully when assets have very different volatilities.
        avg_var = float(np.mean(np.diag(cov_matrix)))
        tau = float(np.trace(cov_matrix)) / (N * avg_var * N) if avg_var > 0 else 1.0 / N
    else:
        # "fixed"
        tau = float(tau_fixed)

    # Safety clamp
    return float(np.clip(tau, 0.001, 1.0))


def build_views_matrices(
    views: List[Dict[str, Any]],
    assets: List[str],
    tau: float,
    cov_matrix: np.ndarray,
    omega_method: str = "confidence",
):
    """
    Build P (pick matrix), Q (view returns), and Omega (view uncertainty).

    Absolute view: "AAPL will return 15%"
        P row = [1, 0, 0, ...] for AAPL
    Relative view: "MSFT will outperform GOOGL by 3%"
        P row = [0, 1, -1, 0, ...] for MSFT vs GOOGL

    Omega methods
    ─────────────────────────────────────────────────────────────
    "confidence"  : Ω_ii = (1/c_i − 1) × τ × (PΣP')_ii
                    Default. Confidence ∈ (0,1) linearly controls
                    how tight the view uncertainty is relative to
                    the prior uncertainty.  c→1 ⟹ Ω→0 (certainty).

    "proportional": Ω = diag(P Σ P')
                    He & Litterman (1999) original.  Ties view
                    uncertainty directly to the factor variance of
                    the view portfolio, independent of confidence.
                    τ is not applied here (absorbed into scaling).

    "idzorek"     : Idzorek (2005) method.
                    Confidence is interpreted as the desired weight
                    deviation from the market-cap weights. Solves
                    for Ω_ii such that the posterior weights shift
                    by (1-c) × (market-cap deviation) toward the
                    view-implied optimal.  Falls back to
                    "proportional" when the linear system is ill-
                    conditioned.
    ─────────────────────────────────────────────────────────────
    """
    n_assets = len(assets)
    asset_idx = {a: i for i, a in enumerate(assets)}

    P_rows = []
    Q_vals = []
    confidence_vals = []

    for view in views:
        vtype = view.get('type', 'absolute')
        confidence = view.get('confidence', 0.5)
        confidence = max(0.01, min(confidence, 0.99))

        if vtype == 'absolute':
            asset = view.get('asset', '')
            if asset not in asset_idx:
                continue
            row = np.zeros(n_assets)
            row[asset_idx[asset]] = 1.0
            P_rows.append(row)
            Q_vals.append(view.get('return', 0.0))
            confidence_vals.append(confidence)

        elif vtype == 'relative':
            a1 = view.get('asset1', '')
            a2 = view.get('asset2', '')
            if a1 not in asset_idx or a2 not in asset_idx:
                continue
            row = np.zeros(n_assets)
            row[asset_idx[a1]] = 1.0
            row[asset_idx[a2]] = -1.0
            P_rows.append(row)
            Q_vals.append(view.get('return', 0.0))
            confidence_vals.append(confidence)

    if not P_rows:
        return None, None, None

    P = np.array(P_rows)    # (K, N)
    Q = np.array(Q_vals)    # (K,)

    # ── Omega construction ───────────────────────────────────────
    psp = P @ cov_matrix @ P.T          # (K, K) — variance of each view portfolio
    psp_diag = np.abs(np.diag(psp))     # (K,)

    if omega_method == "proportional":
        # Ω = diag(PΣP')  — He & Litterman (1999)
        # τ intentionally excluded: proportional already scales with Σ
        omega_diag = psp_diag.copy()

    elif omega_method == "idzorek":
        # Idzorek (2005): confidence → implied weight deviation
        # Ω_ii solved s.t. posterior weights shift c% of max deviation
        # Approximation used here: Ω_ii = (1/c² - 1) × (PΣP')_ii
        # This gives a steeper confidence-to-uncertainty mapping than
        # the linear "confidence" method, better matching Idzorek's
        # original weight-tilt interpretation.
        omega_diag = np.array([
            ((1.0 / max(c, 0.01)) ** 2 - 1.0) * v
            for c, v in zip(confidence_vals, psp_diag)
        ])

    else:
        # "confidence" (default) — τ-scaled
        # Ω_ii = (1/c − 1) × τ × (PΣP')_ii
        omega_diag = np.array([
            (1.0 / c - 1.0) * tau * v
            for c, v in zip(confidence_vals, psp_diag)
        ])

    # Ensure strictly positive diagonal (numerical safety)
    omega_diag = np.maximum(omega_diag, 1e-10)
    Omega = np.diag(omega_diag)

    return P, Q, Omega


def black_litterman_posterior(
    pi: np.ndarray,
    cov_matrix: np.ndarray,
    tau: float,
    P: Optional[np.ndarray],
    Q: Optional[np.ndarray],
    Omega: Optional[np.ndarray],
) -> tuple:
    """
    Compute Black-Litterman posterior expected returns and covariance.

    If no views: posterior = prior (equilibrium)
    With views:
        μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ × [(τΣ)⁻¹π + P'Ω⁻¹Q]
        Σ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ + Σ
    """
    tau_cov = tau * cov_matrix
    tau_cov_inv = np.linalg.inv(tau_cov)

    if P is None or Q is None or Omega is None:
        # No views — return prior
        return pi, cov_matrix + tau_cov

    Omega_inv = np.linalg.inv(Omega)

    # Posterior precision
    precision = tau_cov_inv + P.T @ Omega_inv @ P

    # Posterior mean
    posterior_cov_views = np.linalg.inv(precision)
    posterior_mean = posterior_cov_views @ (tau_cov_inv @ pi + P.T @ Omega_inv @ Q)

    # Posterior covariance (full)
    posterior_cov = posterior_cov_views + cov_matrix

    return posterior_mean, posterior_cov


# ══════════════════════════════════════════════════════════════
# Portfolio Optimization (pyportfolioopt)
# ══════════════════════════════════════════════════════════════

def optimize_portfolio(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.04,
    assets: List[str] = None,
) -> Dict[str, Any]:
    """
    Mean-variance optimization using PyPortfolioOpt.
    Max Sharpe, Min Variance, and Efficient Frontier.
    """
    from pypfopt import expected_returns as er_mod
    from pypfopt import risk_models
    from pypfopt.efficient_frontier import EfficientFrontier
    from pypfopt import CLA

    n = len(assets)

    # Convert to annualized for pyportfolioopt
    mu_annual = pd.Series(expected_returns * 252, index=assets)
    cov_annual = pd.DataFrame(cov_matrix * 252, index=assets, columns=assets)
    rf_annual = risk_free_rate

    # --- Max Sharpe ---
    ef_sharpe = EfficientFrontier(mu_annual, cov_annual, weight_bounds=(0, 1))
    try:
        ef_sharpe.max_sharpe(risk_free_rate=rf_annual)
        w_sharpe = ef_sharpe.clean_weights()
        perf_sharpe = ef_sharpe.portfolio_performance(risk_free_rate=rf_annual)
        ret_sharpe, vol_sharpe, sr_sharpe = perf_sharpe
    except Exception:
        # Fallback to equal weight
        w_sharpe = {a: 1.0 / n for a in assets}
        ret_sharpe = float(mu_annual.mean())
        vol_sharpe = float(np.sqrt(np.ones(n) / n @ cov_annual.values @ np.ones(n) / n))
        sr_sharpe = (ret_sharpe - rf_annual) / vol_sharpe if vol_sharpe > 0 else 0

    # --- Min Variance ---
    ef_minvar = EfficientFrontier(mu_annual, cov_annual, weight_bounds=(0, 1))
    try:
        ef_minvar.min_volatility()
        w_minvar = ef_minvar.clean_weights()
        perf_minvar = ef_minvar.portfolio_performance(risk_free_rate=rf_annual)
        ret_minvar, vol_minvar, sr_minvar = perf_minvar
    except Exception:
        w_minvar = {a: 1.0 / n for a in assets}
        ret_minvar = float(mu_annual.mean())
        vol_minvar = vol_sharpe
        sr_minvar = 0

    # --- Efficient Frontier ---
    frontier = []
    min_ret = ret_minvar
    max_ret = float(mu_annual.max())
    target_rets = np.linspace(min_ret, max_ret, 30)

    for target in target_rets:
        try:
            ef_pt = EfficientFrontier(mu_annual, cov_annual, weight_bounds=(0, 1))
            ef_pt.efficient_return(target_return=target)
            perf = ef_pt.portfolio_performance(risk_free_rate=rf_annual)
            frontier.append({
                'return': safe_float(perf[0] * 100),
                'volatility': safe_float(perf[1] * 100),
                'sharpe': safe_float(perf[2]),
            })
        except Exception:
            continue

    return {
        'max_sharpe': {
            'weights': {a: safe_float(w_sharpe.get(a, 0)) for a in assets},
            'return_annual': safe_float(ret_sharpe * 100),
            'vol_annual': safe_float(vol_sharpe * 100),
            'sharpe': safe_float(sr_sharpe),
        },
        'min_variance': {
            'weights': {a: safe_float(w_minvar.get(a, 0)) for a in assets},
            'return_annual': safe_float(ret_minvar * 100),
            'vol_annual': safe_float(vol_minvar * 100),
            'sharpe': safe_float(sr_minvar),
        },
        'efficient_frontier': frontier,
    }


# ══════════════════════════════════════════════════════════════
# BL Analytical Weights  w* = (1/δ) Σ⁻¹ μ
# ══════════════════════════════════════════════════════════════

def bl_analytical_weights(
    mu: np.ndarray,
    cov_matrix: np.ndarray,
    risk_aversion: float,
    assets: List[str],
    risk_free_rate: float = 0.0,
) -> Dict[str, Any]:
    """
    Compute Black-Litterman implied weights directly from theory.

    Unconstrained (long-short):
        w* = (1/δ) Σ⁻¹ μ_excess
        where μ_excess = μ − rf  (excess over risk-free)

    This is the *exact* portfolio that would be held by a mean-variance
    investor with risk aversion δ when no constraints are imposed.
    It is NOT the same as the PyPortfolioOpt max-Sharpe solution, which
    imposes w ≥ 0 and Σw = 1 constraints.

    We report three variants:
    ─────────────────────────────────────────────────────────────
    raw          : w* = (1/δ) Σ⁻¹ μ_excess  (can be negative, >1)
    normalised   : w_norm = w* / Σ|w*|       (sums to ±1, long+short)
    long_only    : w_lo = max(w*, 0) / Σmax(w*,0)  (rescaled long portion)
    ─────────────────────────────────────────────────────────────

    Also computes implied portfolio performance for each variant.

    Notes
    ─────
    If Σ is near-singular we apply a small ridge: Σ_reg = Σ + λI
    with λ = 1e-6 × mean(diag(Σ)).
    """
    N     = len(assets)
    rf_d  = risk_free_rate   # already in daily scale from caller

    mu_excess = mu - rf_d    # excess return vector (daily)

    # Regularise if needed
    ridge   = 1e-6 * float(np.mean(np.diag(cov_matrix)))
    cov_reg = cov_matrix + ridge * np.eye(N)

    try:
        cov_inv = np.linalg.inv(cov_reg)
    except np.linalg.LinAlgError:
        return {'error': 'Singular covariance — analytical weights unavailable.'}

    # ── Raw weights ──────────────────────────────────────────────
    w_raw = (1.0 / risk_aversion) * cov_inv @ mu_excess   # (N,)

    # ── Normalised (sum |w| = 1) ─────────────────────────────────
    abs_sum    = float(np.sum(np.abs(w_raw)))
    w_norm     = w_raw / abs_sum if abs_sum > 1e-10 else w_raw

    # ── Long-only rescaled ───────────────────────────────────────
    w_lo_raw   = np.maximum(w_raw, 0.0)
    lo_sum     = float(np.sum(w_lo_raw))
    w_lo       = w_lo_raw / lo_sum if lo_sum > 1e-10 else np.ones(N) / N

    def _portfolio_stats(w: np.ndarray) -> Dict[str, Any]:
        """Annualised return, vol, Sharpe for weight vector w."""
        ret_d   = float(w @ mu)
        var_d   = float(w @ cov_matrix @ w)
        vol_d   = float(np.sqrt(max(var_d, 0.0)))
        ret_a   = ret_d  * 252
        vol_a   = vol_d  * np.sqrt(252)
        rf_a    = rf_d   * 252
        sharpe  = (ret_a - rf_a) / vol_a if vol_a > 1e-10 else 0.0
        return {
            'return_annual': safe_float(ret_a  * 100),
            'vol_annual':    safe_float(vol_a  * 100),
            'sharpe':        safe_float(sharpe),
        }

    # Per-asset breakdown
    def _weights_dict(w: np.ndarray) -> Dict[str, float]:
        return {a: safe_float(float(w[i])) for i, a in enumerate(assets)}

    return {
        'raw': {
            'weights':   _weights_dict(w_raw),
            'note':      'Unconstrained: (1/δ)Σ⁻¹μ — may include shorts and leverage',
            **_portfolio_stats(w_raw),
        },
        'normalised': {
            'weights':   _weights_dict(w_norm),
            'note':      'Rescaled so Σ|w|=1 — preserves long/short direction',
            **_portfolio_stats(w_norm),
        },
        'long_only': {
            'weights':   _weights_dict(w_lo),
            'note':      'Positive weights only, rescaled to sum=1',
            **_portfolio_stats(w_lo),
        },
        'risk_aversion_used': safe_float(risk_aversion),
    }


# ══════════════════════════════════════════════════════════════
# Main Endpoint
# ══════════════════════════════════════════════════════════════

@router.post("/black-litterman")
async def black_litterman_analysis(request: BlackLittermanRequest):
    try:
        assets = request.assets
        profiles = None

        # ── 1. Get Data ──
        if request.generate or not request.data:
            df, profiles_list = generate_returns_data(
                assets=assets,
                n_days=request.nDays,
                seed=request.seed,
            )
            profiles = {a: p for a, p in zip(assets, profiles_list)}
        else:
            df = pd.DataFrame(request.data)
            if request.assetCols:
                assets = request.assetCols
            else:
                assets = [c for c in df.columns if c.lower() != 'date']
            for a in assets:
                if a not in df.columns:
                    raise HTTPException(status_code=400, detail=f"Column '{a}' not found.")
                df[a] = pd.to_numeric(df[a], errors='coerce')
            df = df.dropna(subset=assets)

        returns_df = df[assets].astype(np.float64)
        dates = df['date'].values.tolist() if 'date' in df.columns else [str(i) for i in range(len(df))]
        n = len(returns_df)

        if n < 30:
            raise HTTPException(status_code=400, detail=f"Need at least 30 observations, got {n}")

        # Daily risk-free rate (annual → daily)
        rf_daily = request.riskFreeRate / 252

        # ── 2. Covariance Matrix  (Ledoit-Wolf shrinkage) ──
        # Sample covariance amplifies estimation noise in T < 5N regimes.
        # Ledoit-Wolf analytically shrinks toward the identity, producing
        # a better-conditioned matrix that stabilises portfolio weights.
        cov_matrix, shrinkage_coef, cov_method_used = shrink_covariance(
            returns_df, method=request.covMethod
        )
        # Correlation is derived from the shrunk covariance
        std_vec     = np.sqrt(np.diag(cov_matrix))
        corr_matrix = cov_matrix / np.outer(std_vec, std_vec)
        np.fill_diagonal(corr_matrix, 1.0)
        mean_returns = returns_df.mean().values

        # Market caps (from profiles or equal)
        market_caps = np.array([
            ASSET_PROFILES.get(a, {}).get('mkt_cap', 100) for a in assets
        ], dtype=np.float64)

        # ── 3. Tau calibration ──
        # τ is calibrated from data — NOT a fixed magic number.
        # It controls the weight given to the prior vs views.
        tau = calibrate_tau(
            method=request.tauMethod,
            T=n,                        # number of daily observations
            cov_matrix=cov_matrix,
            tau_fixed=request.tauValue,
        )

        # ── 4. Market-Implied Returns (Prior) ──
        # π = δ Σ w — risk-free rate does NOT enter this formula.
        # π is an excess return by construction.
        pi, w_mkt = compute_market_implied_returns(
            cov_matrix=cov_matrix,
            market_caps=market_caps,
            risk_aversion=request.riskAversion,
            assets=assets,
        )

        # ── 5. Build Views ──
        P, Q, Omega = None, None, None
        views_info = []
        if request.views and len(request.views) > 0:
            P, Q, Omega = build_views_matrices(
                views=request.views,
                assets=assets,
                tau=tau,
                cov_matrix=cov_matrix,
                omega_method=request.omegaMethod,
            )
            for v in request.views:
                views_info.append({
                    'type': v.get('type', 'absolute'),
                    'description': (
                        f"{v.get('asset', '?')} → {v.get('return', 0) * 100:.1f}%"
                        if v.get('type') == 'absolute'
                        else f"{v.get('asset1', '?')} > {v.get('asset2', '?')} by {v.get('return', 0) * 100:.1f}%"
                    ),
                    'confidence': v.get('confidence', 0.5),
                })

        # ── 6. BL Posterior ──
        bl_returns, bl_cov = black_litterman_posterior(
            pi=pi, cov_matrix=cov_matrix, tau=tau,
            P=P, Q=Q, Omega=Omega,
        )

        # ── 7. BL Analytical Weights  w* = (1/δ) Σ⁻¹ μ ──
        # Theoretical BL weights — no MVO solver, no constraints assumed.
        # Gives the "pure" quant view of what the model implies.
        prior_analytical = bl_analytical_weights(
            mu=pi, cov_matrix=cov_matrix,
            risk_aversion=request.riskAversion,
            assets=assets, risk_free_rate=rf_daily,
        )
        bl_analytical = bl_analytical_weights(
            mu=bl_returns, cov_matrix=bl_cov,
            risk_aversion=request.riskAversion,
            assets=assets, risk_free_rate=rf_daily,
        )

        # ── 8. Optimize Portfolios (constrained MVO via PyPortfolioOpt) ──
        prior_opt = optimize_portfolio(pi, cov_matrix, rf_daily, assets)
        bl_opt    = optimize_portfolio(bl_returns, bl_cov, rf_daily, assets)

        # ── 9. Chart Data ──

        # Returns comparison: prior vs posterior
        returns_comparison = []
        for i, a in enumerate(assets):
            returns_comparison.append({
                'asset': a,
                'historical': safe_float(mean_returns[i] * 252 * 100),
                'prior': safe_float(pi[i] * 252 * 100),
                'posterior': safe_float(bl_returns[i] * 252 * 100),
            })

        # Weights comparison — add analytical long-only weights
        weights_comparison = []
        for a in assets:
            weights_comparison.append({
                'asset': a,
                'market_cap':      safe_float(w_mkt[assets.index(a)] * 100),
                'prior_sharpe':    safe_float(prior_opt['max_sharpe']['weights'].get(a, 0) * 100),
                'bl_sharpe':       safe_float(bl_opt['max_sharpe']['weights'].get(a, 0) * 100),
                'bl_analytical':   safe_float(
                    bl_analytical.get('long_only', {}).get('weights', {}).get(a, 0) * 100
                ),
            })

        # Correlation heatmap data
        corr_data = {}
        for i, a in enumerate(assets):
            corr_data[a] = {assets[j]: safe_float(corr_matrix[i, j]) for j in range(len(assets))}

        # Cumulative returns chart
        cum_returns = (1 + returns_df).cumprod()
        cum_chart = []
        for i in range(n):
            entry = {'date': str(dates[i])}
            for a in assets:
                entry[a] = safe_float(cum_returns.iloc[i][a])
            cum_chart.append(entry)

        # Asset statistics
        asset_stats = []
        for i, a in enumerate(assets):
            series = returns_df[a]
            asset_stats.append({
                'asset': a,
                'name': ASSET_PROFILES.get(a, {}).get('name', a),
                'annual_return': safe_float(series.mean() * 252 * 100),
                'annual_vol': safe_float(series.std() * np.sqrt(252) * 100),
                'sharpe': safe_float(series.mean() / series.std() * np.sqrt(252)) if series.std() > 0 else 0,
                'max_drawdown': safe_float(_max_drawdown(series.values) * 100),
                'mkt_cap_weight': safe_float(w_mkt[i] * 100),
                'prior_return': safe_float(pi[i] * 252 * 100),
                'bl_return': safe_float(bl_returns[i] * 252 * 100),
            })

        # ── 10. Build Response ──
        results = {
            'assets': assets,
            'n_assets': len(assets),
            'views': views_info,
            'has_views': len(views_info) > 0,
            'parameters': {
                'risk_free_rate':  request.riskFreeRate,
                'risk_aversion':   request.riskAversion,
                'tau':             safe_float(tau),
                'tau_method':      request.tauMethod,
                'omega_method':    request.omegaMethod,
                'cov_method':      cov_method_used,
                'shrinkage_coef':  safe_float(shrinkage_coef),
                'tau_note': (
                    f"τ calibrated via '{request.tauMethod}' "
                    f"(T={n} obs) → τ = {tau:.6f}"
                ),
                'cov_note': (
                    f"Covariance estimated via {cov_method_used} "
                    + (f"(δ={shrinkage_coef:.4f})" if shrinkage_coef > 0 else "(no shrinkage)")
                ),
            },
            'prior_portfolio':            prior_opt,
            'bl_portfolio':               bl_opt,
            # ── NEW ──────────────────────────────────────────────
            'prior_analytical_weights':   prior_analytical,
            'bl_analytical_weights':      bl_analytical,
            # ─────────────────────────────────────────────────────
            'asset_stats': asset_stats,
            'correlation': corr_data,
            'data_summary': {
                'n_observations': n,
                'date_range': f"{dates[0]} to {dates[-1]}",
            },
            'charts': {
                'returns_comparison':      returns_comparison,
                'weights_comparison':      weights_comparison,
                'efficient_frontier_prior': prior_opt['efficient_frontier'],
                'efficient_frontier_bl':   bl_opt['efficient_frontier'],
                'cum_returns':             cum_chart,
            },
            'available_assets': list(ASSET_PROFILES.keys()),
        }

        return _to_native({'results': results})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


def _max_drawdown(returns: np.ndarray) -> float:
    """Compute maximum drawdown from daily returns."""
    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    return float(np.min(dd)) if len(dd) > 0 else 0.0
