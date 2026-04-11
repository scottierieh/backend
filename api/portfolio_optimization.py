from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy import stats
import warnings
import traceback

warnings.filterwarnings('ignore')

router = APIRouter()


# ══════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════

RISK_FREE_RATE_ANNUAL = 0.05   # 5% annual default

STOCK_PROFILES: Dict[str, Dict] = {
    'AAPL':  {'name': 'Apple Inc.',         'mu': 0.0120, 'sigma': 0.065, 'sector': 'Tech'},
    'MSFT':  {'name': 'Microsoft Corp.',    'mu': 0.0110, 'sigma': 0.058, 'sector': 'Tech'},
    'GOOGL': {'name': 'Alphabet Inc.',      'mu': 0.0100, 'sigma': 0.062, 'sector': 'Tech'},
    'AMZN':  {'name': 'Amazon.com Inc.',    'mu': 0.0115, 'sigma': 0.072, 'sector': 'Tech'},
    'TSLA':  {'name': 'Tesla Inc.',         'mu': 0.0090, 'sigma': 0.130, 'sector': 'EV'},
    'NVDA':  {'name': 'NVIDIA Corp.',       'mu': 0.0160, 'sigma': 0.095, 'sector': 'Semis'},
    'JPM':   {'name': 'JPMorgan Chase',     'mu': 0.0075, 'sigma': 0.055, 'sector': 'Finance'},
    'JNJ':   {'name': 'J&J',               'mu': 0.0055, 'sigma': 0.035, 'sector': 'Health'},
    'XOM':   {'name': 'Exxon Mobil',        'mu': 0.0070, 'sigma': 0.052, 'sector': 'Energy'},
    'WMT':   {'name': 'Walmart Inc.',       'mu': 0.0060, 'sigma': 0.040, 'sector': 'Retail'},
    'PG':    {'name': 'Procter & Gamble',   'mu': 0.0052, 'sigma': 0.032, 'sector': 'Consumer'},
    'KO':    {'name': 'Coca-Cola Co.',      'mu': 0.0048, 'sigma': 0.030, 'sector': 'Consumer'},
    'BRK.B': {'name': 'Berkshire Hathaway','mu': 0.0078, 'sigma': 0.042, 'sector': 'Finance'},
    'META':  {'name': 'Meta Platforms',     'mu': 0.0105, 'sigma': 0.075, 'sector': 'Tech'},
    'V':     {'name': 'Visa Inc.',          'mu': 0.0082, 'sigma': 0.045, 'sector': 'Finance'},
    'AMD':   {'name': 'AMD Inc.',           'mu': 0.0140, 'sigma': 0.110, 'sector': 'Semis'},
    'NFLX':  {'name': 'Netflix Inc.',       'mu': 0.0095, 'sigma': 0.085, 'sector': 'Media'},
    'DIS':   {'name': 'Walt Disney Co.',    'mu': 0.0058, 'sigma': 0.055, 'sector': 'Media'},
    'GLD':   {'name': 'Gold ETF',           'mu': 0.0040, 'sigma': 0.028, 'sector': 'Commodity'},
    'TLT':   {'name': 'US 20Y Bond ETF',    'mu': 0.0025, 'sigma': 0.020, 'sector': 'Bond'},
}

# Realistic cross-asset correlation clusters
SECTOR_CORR = {
    ('Tech',   'Tech'):     0.72,
    ('Tech',   'Semis'):    0.65,
    ('Tech',   'Finance'):  0.35,
    ('Tech',   'Health'):   0.20,
    ('Tech',   'Energy'):   0.15,
    ('Tech',   'Consumer'): 0.25,
    ('Tech',   'Retail'):   0.30,
    ('Tech',   'EV'):       0.55,
    ('Tech',   'Media'):    0.50,
    ('Tech',   'Commodity'):0.05,
    ('Tech',   'Bond'):    -0.10,
    ('Semis',  'Semis'):    0.78,
    ('Semis',  'Finance'):  0.30,
    ('Semis',  'Health'):   0.15,
    ('Semis',  'Energy'):   0.12,
    ('Semis',  'Consumer'): 0.18,
    ('Semis',  'Retail'):   0.22,
    ('Semis',  'EV'):       0.60,
    ('Semis',  'Media'):    0.45,
    ('Semis',  'Commodity'):0.08,
    ('Semis',  'Bond'):    -0.12,
    ('Finance','Finance'):  0.68,
    ('Finance','Health'):   0.28,
    ('Finance','Energy'):   0.40,
    ('Finance','Consumer'): 0.32,
    ('Finance','Retail'):   0.38,
    ('Finance','EV'):       0.25,
    ('Finance','Media'):    0.30,
    ('Finance','Commodity'):0.15,
    ('Finance','Bond'):     0.05,
    ('Health', 'Health'):   0.55,
    ('Health', 'Energy'):   0.18,
    ('Health', 'Consumer'): 0.35,
    ('Health', 'Retail'):   0.28,
    ('Health', 'EV'):       0.10,
    ('Health', 'Media'):    0.20,
    ('Health', 'Commodity'):0.10,
    ('Health', 'Bond'):     0.15,
    ('Energy', 'Energy'):   0.70,
    ('Energy', 'Consumer'): 0.22,
    ('Energy', 'Retail'):   0.25,
    ('Energy', 'EV'):       0.20,
    ('Energy', 'Media'):    0.18,
    ('Energy', 'Commodity'):0.55,
    ('Energy', 'Bond'):    -0.05,
    ('Consumer','Consumer'):0.60,
    ('Consumer','Retail'):  0.48,
    ('Consumer','EV'):      0.15,
    ('Consumer','Media'):   0.32,
    ('Consumer','Commodity'):0.12,
    ('Consumer','Bond'):    0.10,
    ('Retail', 'Retail'):   0.65,
    ('Retail', 'EV'):       0.18,
    ('Retail', 'Media'):    0.28,
    ('Retail', 'Commodity'):0.10,
    ('Retail', 'Bond'):     0.08,
    ('EV',     'EV'):       1.00,
    ('EV',     'Media'):    0.30,
    ('EV',     'Commodity'):0.12,
    ('EV',     'Bond'):    -0.08,
    ('Media',  'Media'):    0.62,
    ('Media',  'Commodity'):0.08,
    ('Media',  'Bond'):    -0.05,
    ('Commodity','Commodity'):0.80,
    ('Commodity','Bond'):   0.20,
    ('Bond',   'Bond'):     0.85,
}


def _sector_corr(s1: str, s2: str) -> float:
    if s1 == s2:
        return SECTOR_CORR.get((s1, s1), 0.60)
    key = (s1, s2) if (s1, s2) in SECTOR_CORR else (s2, s1)
    return SECTOR_CORR.get(key, 0.25)


# ══════════════════════════════════════════════════════════════
# Request Model
# ══════════════════════════════════════════════════════════════

class PortfolioRequest(BaseModel):
    # Data mode
    tickers: Optional[List[str]] = None          # for generate mode
    data: Optional[List[Dict[str, Any]]] = None  # user-supplied returns
    returnCols: Optional[List[str]] = None        # which columns are asset returns
    dateCol: Optional[str] = 'date'
    # Generate settings
    nMonths: int = 120
    seed: Optional[int] = None
    # Optimization settings
    riskFreeRate: float = RISK_FREE_RATE_ANNUAL   # annual
    nFrontierPoints: int = 60
    nMonteCarloPortfolios: int = 2000
    allowShortSelling: bool = False
    # Feature flags
    includeBlackLitterman: bool = True
    includeRiskParity: bool = True
    includeCVaR: bool = True
    includeRolling: bool = True
    rollingWindow: int = 36
    # Black-Litterman views: list of {asset, view_return (annual %), confidence}
    blViews: Optional[List[Dict[str, Any]]] = None
    # Constraints
    maxWeight: float = 1.0       # max weight per asset
    minWeight: float = 0.0       # min weight (0 = long-only)


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

def _to_native(obj):
    if isinstance(obj, (np.integer,)):  return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    if isinstance(obj, np.ndarray):     return [_to_native(x) for x in obj.tolist()]
    if isinstance(obj, np.bool_):       return bool(obj)
    if isinstance(obj, dict):           return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):           return [_to_native(x) for x in obj]
    return obj


def safe_float(val, default=0.0):
    try:
        if val is None: return default
        f = float(val)
        return default if (np.isnan(f) or np.isinf(f)) else f
    except Exception:
        return default


# ══════════════════════════════════════════════════════════════
# Data Generator
# ══════════════════════════════════════════════════════════════

def generate_portfolio_data(
    tickers: List[str],
    n_months: int = 120,
    seed: Optional[int] = None,
) -> tuple:
    """
    Generate correlated monthly return series for a list of tickers.
    Returns (DataFrame with date + per-ticker returns, asset_names dict).
    """
    rng = np.random.default_rng(seed)
    n = len(tickers)

    profiles = {t: STOCK_PROFILES.get(t, {'name': t, 'mu': 0.008, 'sigma': 0.06, 'sector': 'Tech'})
                for t in tickers}

    mu_monthly   = np.array([profiles[t]['mu']    for t in tickers])
    sigma_monthly = np.array([profiles[t]['sigma'] for t in tickers])
    sectors       = [profiles[t]['sector']         for t in tickers]

    # Build correlation matrix
    corr = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            c = _sector_corr(sectors[i], sectors[j])
            corr[i, j] = c
            corr[j, i] = c

    # Ensure positive semi-definite via nearest PSD
    eigvals = np.linalg.eigvalsh(corr)
    if eigvals.min() < 0:
        corr = corr + (-eigvals.min() + 1e-8) * np.eye(n)
        d = np.diag(corr) ** 0.5
        corr = corr / np.outer(d, d)

    cov = np.outer(sigma_monthly, sigma_monthly) * corr

    returns = rng.multivariate_normal(mu_monthly, cov, size=n_months)  # (T, N)

    end_date   = pd.Timestamp('2025-04-30')
    dates      = pd.date_range(end=end_date, periods=n_months, freq='ME')
    date_strs  = dates.strftime('%Y-%m').tolist()

    df = pd.DataFrame(
        np.round(returns * 100, 4),
        columns=tickers,
    )
    df.insert(0, 'date', date_strs)

    asset_names = {t: profiles[t]['name'] for t in tickers}
    return df, asset_names, cov * 12  # annualised cov for reference


# ══════════════════════════════════════════════════════════════
# Core Portfolio Math
# ══════════════════════════════════════════════════════════════

def portfolio_stats(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray,
                    rf: float = 0.0, periods_per_year: int = 12) -> Dict:
    """Annualised return, volatility, Sharpe for a given weight vector."""
    w = np.array(weights)
    ret_ann  = float(w @ mu) * periods_per_year
    var_ann  = float(w @ cov @ w) * periods_per_year
    vol_ann  = float(np.sqrt(max(var_ann, 0)))
    sharpe   = (ret_ann - rf) / vol_ann if vol_ann > 0 else 0.0
    return {'return': ret_ann, 'volatility': vol_ann, 'sharpe': sharpe}


def neg_sharpe(weights, mu, cov, rf, ppy):
    s = portfolio_stats(weights, mu, cov, rf, ppy)
    return -s['sharpe']


def portfolio_variance(weights, cov):
    w = np.array(weights)
    return float(w @ cov @ w)


def _optimize(
    mu: np.ndarray,
    cov: np.ndarray,
    rf: float,
    ppy: int,
    objective: str,          # 'sharpe' | 'variance' | 'target_return'
    target_return: float = 0.0,
    allow_short: bool = False,
    max_w: float = 1.0,
    min_w: float = 0.0,
) -> Optional[np.ndarray]:
    n = len(mu)
    bounds = [(min_w, max_w)] * n
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    if objective == 'target_return':
        constraints.append({
            'type': 'eq',
            'fun': lambda w: np.dot(w, mu) * ppy - target_return,
        })
    w0 = np.ones(n) / n

    if objective == 'sharpe':
        result = minimize(neg_sharpe, w0, args=(mu, cov, rf, ppy),
                          method='SLSQP', bounds=bounds, constraints=constraints,
                          options={'ftol': 1e-12, 'maxiter': 500})
    elif objective in ('variance', 'target_return'):
        result = minimize(portfolio_variance, w0, args=(cov,),
                          method='SLSQP', bounds=bounds, constraints=constraints,
                          options={'ftol': 1e-12, 'maxiter': 500})
    else:
        return None

    return result.x if result.success else None


# ══════════════════════════════════════════════════════════════
# Efficient Frontier
# ══════════════════════════════════════════════════════════════

def compute_efficient_frontier(
    mu: np.ndarray, cov: np.ndarray, rf: float, ppy: int,
    tickers: List[str],
    n_points: int = 60, allow_short: bool = False,
    max_w: float = 1.0, min_w: float = 0.0,
) -> List[Dict]:
    n = len(mu)
    ret_min = float(mu.min()) * ppy
    ret_max = float(mu.max()) * ppy
    # Pad slightly above max to allow extrapolation
    targets = np.linspace(ret_min * 0.95, ret_max * 1.05, n_points)

    frontier = []
    for target in targets:
        w = _optimize(mu, cov, rf, ppy, 'target_return',
                      target_return=target,
                      allow_short=allow_short,
                      max_w=max_w, min_w=min_w)
        if w is None:
            continue
        s = portfolio_stats(w, mu, cov, rf, ppy)
        if s['volatility'] < 1e-8:
            continue
        frontier.append({
            'volatility': round(s['volatility'] * 100, 4),
            'return':     round(s['return']     * 100, 4),
            'sharpe':     round(s['sharpe'], 4),
            'weights':    {t: round(float(ww), 4) for t, ww in zip(tickers, w)},
        })

    return frontier


# ══════════════════════════════════════════════════════════════
# Monte Carlo Simulation
# ══════════════════════════════════════════════════════════════

def monte_carlo_portfolios(
    mu: np.ndarray, cov: np.ndarray, rf: float, ppy: int,
    tickers: List[str], n_sim: int = 2000,
    allow_short: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> List[Dict]:
    if rng is None:
        rng = np.random.default_rng(42)
    n = len(tickers)
    portfolios = []
    for _ in range(n_sim):
        if allow_short:
            w = rng.standard_normal(n)
        else:
            w = rng.dirichlet(np.ones(n))
        w = w / w.sum()
        s = portfolio_stats(w, mu, cov, rf, ppy)
        portfolios.append({
            'volatility': round(s['volatility'] * 100, 4),
            'return':     round(s['return']     * 100, 4),
            'sharpe':     round(s['sharpe'], 4),
        })
    return portfolios


# ══════════════════════════════════════════════════════════════
# Special Portfolios
# ══════════════════════════════════════════════════════════════

def min_variance_portfolio(mu, cov, rf, ppy, tickers,
                            allow_short, max_w, min_w) -> Dict:
    w = _optimize(mu, cov, rf, ppy, 'variance',
                  allow_short=allow_short, max_w=max_w, min_w=min_w)
    if w is None:
        w = np.ones(len(mu)) / len(mu)
    s = portfolio_stats(w, mu, cov, rf, ppy)
    return {
        'weights':    {t: round(float(ww), 4) for t, ww in zip(tickers, w)},
        'return':     round(s['return']     * 100, 4),
        'volatility': round(s['volatility'] * 100, 4),
        'sharpe':     round(s['sharpe'], 4),
    }


def max_sharpe_portfolio(mu, cov, rf, ppy, tickers,
                          allow_short, max_w, min_w) -> Dict:
    w = _optimize(mu, cov, rf, ppy, 'sharpe',
                  allow_short=allow_short, max_w=max_w, min_w=min_w)
    if w is None:
        w = np.ones(len(mu)) / len(mu)
    s = portfolio_stats(w, mu, cov, rf, ppy)
    return {
        'weights':    {t: round(float(ww), 4) for t, ww in zip(tickers, w)},
        'return':     round(s['return']     * 100, 4),
        'volatility': round(s['volatility'] * 100, 4),
        'sharpe':     round(s['sharpe'], 4),
    }


def equal_weight_portfolio(mu, cov, rf, ppy, tickers) -> Dict:
    n = len(mu)
    w = np.ones(n) / n
    s = portfolio_stats(w, mu, cov, rf, ppy)
    return {
        'weights':    {t: round(1.0 / n, 4) for t in tickers},
        'return':     round(s['return']     * 100, 4),
        'volatility': round(s['volatility'] * 100, 4),
        'sharpe':     round(s['sharpe'], 4),
    }


# ══════════════════════════════════════════════════════════════
# Risk Parity
# ══════════════════════════════════════════════════════════════

def risk_parity_portfolio(mu, cov, rf, ppy, tickers) -> Dict:
    """
    Equal Risk Contribution (ERC) portfolio.
    Minimise sum of squared differences of marginal risk contributions.
    """
    n = len(mu)

    def risk_contributions(w):
        w = np.array(w)
        port_vol = np.sqrt(w @ cov @ w)
        if port_vol < 1e-10:
            return np.ones(n) / n
        mrc = (cov @ w) / port_vol   # marginal risk contributions
        return w * mrc               # risk contributions

    def objective(w):
        rc = risk_contributions(w)
        target = np.sum(rc) / n
        return float(np.sum((rc - target) ** 2))

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(1e-4, 1.0)] * n
    w0 = np.ones(n) / n

    result = minimize(objective, w0, method='SLSQP',
                      bounds=bounds, constraints=constraints,
                      options={'ftol': 1e-12, 'maxiter': 1000})
    w = result.x if result.success else w0

    rc = risk_contributions(w)
    s  = portfolio_stats(w, mu, cov, rf, ppy)
    port_vol = np.sqrt(w @ cov @ w) * np.sqrt(ppy)

    return {
        'weights':             {t: round(float(ww), 4) for t, ww in zip(tickers, w)},
        'return':              round(s['return']     * 100, 4),
        'volatility':          round(s['volatility'] * 100, 4),
        'sharpe':              round(s['sharpe'], 4),
        'risk_contributions':  {t: round(float(rc[i] / max(port_vol, 1e-10)) * 100, 4)
                                for i, t in enumerate(tickers)},
        'converged':           result.success,
    }


# ══════════════════════════════════════════════════════════════
# CVaR (Conditional Value at Risk)
# ══════════════════════════════════════════════════════════════

def cvar_portfolio(
    returns_df: pd.DataFrame,
    tickers: List[str],
    rf: float,
    ppy: int,
    confidence: float = 0.95,
) -> Dict:
    """
    Minimise CVaR (Expected Shortfall) at given confidence level.
    Uses historical simulation on the sample return matrix.
    """
    R = returns_df[tickers].values / 100.0   # (T, N)
    T, n = R.shape
    rf_period = rf / ppy

    def portfolio_cvar(w):
        w = np.array(w)
        port_ret = R @ w
        var_threshold = np.percentile(port_ret, (1 - confidence) * 100)
        tail = port_ret[port_ret <= var_threshold]
        if len(tail) == 0:
            return 0.0
        return float(-np.mean(tail))

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    bounds = [(0.0, 1.0)] * n
    w0 = np.ones(n) / n

    result = minimize(portfolio_cvar, w0, method='SLSQP',
                      bounds=bounds, constraints=constraints,
                      options={'ftol': 1e-10, 'maxiter': 500})
    w = result.x if result.success else w0

    port_ret = (R @ w) * 100
    var_val  = float(np.percentile(port_ret, (1 - confidence) * 100))
    cvar_val = float(-np.mean(port_ret[port_ret <= var_val]))

    mu_sample = R.mean(axis=0)
    cov_sample = np.cov(R.T, ddof=1)
    s = portfolio_stats(w, mu_sample, cov_sample, rf / ppy, ppy)

    return {
        'weights':    {t: round(float(ww), 4) for t, ww in zip(tickers, w)},
        'return':     round(s['return']     * 100, 4),
        'volatility': round(s['volatility'] * 100, 4),
        'sharpe':     round(s['sharpe'], 4),
        'var_95':     round(var_val, 4),
        'cvar_95':    round(cvar_val, 4),
        'confidence': confidence,
        'converged':  result.success,
    }


# ══════════════════════════════════════════════════════════════
# Black-Litterman
# ══════════════════════════════════════════════════════════════

def black_litterman(
    mu_market: np.ndarray,     # prior equilibrium returns (annual, decimal)
    cov: np.ndarray,           # annual covariance
    views: List[Dict],         # [{asset_idx, view_return, confidence}]
    tickers: List[str],
    rf: float,
    ppy: int,
    tau: float = 0.05,
    allow_short: bool = False,
    max_w: float = 1.0,
    min_w: float = 0.0,
) -> Dict:
    """
    Black-Litterman model:
      Posterior μ = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹ [(τΣ)⁻¹ μ_eq + P'Ω⁻¹Q]
    """
    n = len(tickers)
    if not views:
        # No views → return market-cap equal-weight
        w = np.ones(n) / n
        s = portfolio_stats(w, mu_market, cov, rf, ppy)
        return {
            'weights':         {t: round(1.0 / n, 4) for t in tickers},
            'return':          round(s['return']     * 100, 4),
            'volatility':      round(s['volatility'] * 100, 4),
            'sharpe':          round(s['sharpe'], 4),
            'posterior_mu':    {t: round(float(mu_market[i]) * 100, 4)
                                for i, t in enumerate(tickers)},
            'views_applied':   0,
        }

    valid_views = [v for v in views
                   if v.get('asset') in tickers
                   and v.get('view_return') is not None]

    if not valid_views:
        w = _optimize(mu_market, cov, rf, ppy, 'sharpe',
                      allow_short=allow_short, max_w=max_w, min_w=min_w) or np.ones(n) / n
        s = portfolio_stats(w, mu_market, cov, rf, ppy)
        return {
            'weights':         {t: round(float(ww), 4) for t, ww in zip(tickers, w)},
            'return':          round(s['return'] * 100, 4),
            'volatility':      round(s['volatility'] * 100, 4),
            'sharpe':          round(s['sharpe'], 4),
            'posterior_mu':    {t: round(float(mu_market[i]) * 100, 4)
                                for i, t in enumerate(tickers)},
            'views_applied':   0,
        }

    k = len(valid_views)
    P = np.zeros((k, n))
    Q = np.zeros(k)
    omega_diag = np.zeros(k)

    for i, v in enumerate(valid_views):
        asset_idx = tickers.index(v['asset'])
        P[i, asset_idx] = 1.0
        Q[i] = v['view_return'] / 100.0               # convert % to decimal
        conf = max(min(float(v.get('confidence', 0.5)), 1.0), 0.01)
        omega_diag[i] = tau * float(P[i] @ cov @ P[i]) / conf

    Omega = np.diag(omega_diag)
    tau_cov = tau * cov

    try:
        tau_cov_inv = np.linalg.inv(tau_cov + 1e-10 * np.eye(n))
        Omega_inv   = np.linalg.inv(Omega + 1e-10 * np.eye(k))

        M_inv  = tau_cov_inv + P.T @ Omega_inv @ P
        M      = np.linalg.inv(M_inv + 1e-10 * np.eye(n))
        mu_bl  = M @ (tau_cov_inv @ mu_market + P.T @ Omega_inv @ Q)
    except np.linalg.LinAlgError:
        mu_bl = mu_market.copy()

    w = _optimize(mu_bl, cov, rf, ppy, 'sharpe',
                  allow_short=allow_short, max_w=max_w, min_w=min_w)
    if w is None:
        w = np.ones(n) / n
    s = portfolio_stats(w, mu_bl, cov, rf, ppy)

    return {
        'weights':       {t: round(float(ww), 4) for t, ww in zip(tickers, w)},
        'return':        round(s['return']     * 100, 4),
        'volatility':    round(s['volatility'] * 100, 4),
        'sharpe':        round(s['sharpe'], 4),
        'posterior_mu':  {t: round(float(mu_bl[i]) * 100, 4) for i, t in enumerate(tickers)},
        'prior_mu':      {t: round(float(mu_market[i]) * 100, 4) for i, t in enumerate(tickers)},
        'views_applied': k,
        'views_detail':  [
            {'asset': v['asset'],
             'view_return': v['view_return'],
             'confidence':  v.get('confidence', 0.5)}
            for v in valid_views
        ],
    }


# ══════════════════════════════════════════════════════════════
# Rolling Optimization
# ══════════════════════════════════════════════════════════════

def rolling_optimization(
    returns_df: pd.DataFrame,
    tickers: List[str],
    rf: float,
    ppy: int,
    window: int = 36,
    allow_short: bool = False,
    max_w: float = 1.0,
    min_w: float = 0.0,
) -> List[Dict]:
    """
    Walk-forward rolling portfolio optimization.
    Returns per-window stats for MaxSharpe, MinVar, EqualWeight.
    """
    n_obs = len(returns_df)
    if n_obs < window:
        return []

    results = []
    dates = returns_df['date'].values if 'date' in returns_df.columns else [str(i) for i in range(n_obs)]
    R_all = returns_df[tickers].values / 100.0

    for i in range(window, n_obs + 1):
        R_slice = returns_df.iloc[i - window:i][tickers].copy()
        R_slice = R_slice.dropna()
        if len(R_slice) < max(len(tickers) + 2, window // 2):
            continue
        R = R_slice.values / 100.0
        mu_w   = R.mean(axis=0)
        cov_w  = np.cov(R.T, ddof=1)
        if cov_w.ndim == 0:
            cov_w = np.array([[float(cov_w)]])

        w_ms = _optimize(mu_w, cov_w, rf / ppy, ppy, 'sharpe',
                         allow_short=allow_short, max_w=max_w, min_w=min_w)
        w_mv = _optimize(mu_w, cov_w, rf / ppy, ppy, 'variance',
                         allow_short=allow_short, max_w=max_w, min_w=min_w)
        w_eq = np.ones(len(tickers)) / len(tickers)

        row = {'date': str(dates[i - 1])}
        for label, w in [('max_sharpe', w_ms), ('min_var', w_mv), ('equal_weight', w_eq)]:
            if w is None:
                w = w_eq
            s = portfolio_stats(w, mu_w, cov_w, rf / ppy, ppy)
            row[f'{label}_return']     = round(s['return']     * 100, 4)
            row[f'{label}_volatility'] = round(s['volatility'] * 100, 4)
            row[f'{label}_sharpe']     = round(s['sharpe'], 4)
            for t, ww in zip(tickers, w):
                row[f'{label}_w_{t}'] = round(float(ww), 4)

        results.append(row)

    return results


# ══════════════════════════════════════════════════════════════
# Performance Statistics
# ══════════════════════════════════════════════════════════════

def full_performance_stats(
    weights_dict: Dict[str, float],
    returns_df: pd.DataFrame,
    tickers: List[str],
    rf: float,
    ppy: int,
) -> Dict:
    """
    Compute comprehensive performance metrics from historical returns.
    """
    w = np.array([weights_dict.get(t, 0.0) for t in tickers])
    R = returns_df[tickers].values / 100.0   # (T, N)
    port_ret = R @ w                          # (T,) monthly

    ann_ret  = float(np.mean(port_ret)) * ppy
    ann_vol  = float(np.std(port_ret, ddof=1)) * np.sqrt(ppy)
    sharpe   = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0.0

    # Drawdown
    cum  = np.cumprod(1 + port_ret)
    roll_max = np.maximum.accumulate(cum)
    dd   = (cum - roll_max) / roll_max
    max_dd = float(dd.min())

    # Calmar
    calmar = ann_ret / abs(max_dd) if max_dd < 0 else np.nan

    # Sortino
    downside = port_ret[port_ret < 0]
    sortino_vol = float(np.std(downside, ddof=1)) * np.sqrt(ppy) if len(downside) > 1 else ann_vol
    sortino = (ann_ret - rf) / sortino_vol if sortino_vol > 0 else 0.0

    # CVaR 95%
    pct_returns = port_ret * 100
    var_95  = float(np.percentile(pct_returns, 5))
    cvar_95 = float(-np.mean(pct_returns[pct_returns <= var_95]))

    # Skewness / kurtosis
    skew = float(stats.skew(port_ret))
    kurt = float(stats.kurtosis(port_ret))

    return {
        'annualised_return':   round(ann_ret  * 100, 4),
        'annualised_vol':      round(ann_vol  * 100, 4),
        'sharpe':              round(sharpe, 4),
        'sortino':             round(sortino, 4),
        'calmar':              round(calmar, 4) if not np.isnan(calmar) else None,
        'max_drawdown':        round(max_dd  * 100, 4),
        'var_95':              round(var_95, 4),
        'cvar_95':             round(cvar_95, 4),
        'skewness':            round(skew, 4),
        'kurtosis':            round(kurt, 4),
        'n_observations':      int(len(port_ret)),
    }


# ══════════════════════════════════════════════════════════════
# Correlation & Covariance Helpers
# ══════════════════════════════════════════════════════════════

def correlation_matrix(returns_df: pd.DataFrame, tickers: List[str]) -> Dict:
    corr = returns_df[tickers].corr()
    return {col: {c: safe_float(corr.loc[col, c]) for c in corr.columns}
            for col in corr.columns}


def asset_stats(returns_df: pd.DataFrame, tickers: List[str], ppy: int) -> Dict:
    result = {}
    for t in tickers:
        s = returns_df[t] / 100.0
        ann_ret = float(s.mean()) * ppy
        ann_vol = float(s.std(ddof=1)) * np.sqrt(ppy)
        result[t] = {
            'mean_monthly':   round(float(s.mean()) * 100, 4),
            'return_annual':  round(ann_ret * 100, 4),
            'vol_annual':     round(ann_vol * 100, 4),
            'sharpe_annual':  round(ann_ret / ann_vol, 4) if ann_vol > 0 else 0.0,
            'skewness':       round(float(stats.skew(s)), 4),
            'kurtosis':       round(float(stats.kurtosis(s)), 4),
            'min_monthly':    round(float(s.min()) * 100, 4),
            'max_monthly':    round(float(s.max()) * 100, 4),
        }
    return result


# ══════════════════════════════════════════════════════════════
# Cumulative Returns Chart Data
# ══════════════════════════════════════════════════════════════

def cumulative_returns(
    portfolios: Dict[str, Dict[str, float]],   # label → weights dict
    returns_df: pd.DataFrame,
    tickers: List[str],
    ppy: int,
) -> List[Dict]:
    """
    Compute cumulative return index (starting at 100) for each portfolio.
    """
    R = returns_df[tickers].values / 100.0
    dates = returns_df['date'].values if 'date' in returns_df.columns else [str(i) for i in range(len(R))]

    cum_rets: Dict[str, np.ndarray] = {}
    for label, wd in portfolios.items():
        w = np.array([wd.get(t, 0.0) for t in tickers])
        pr = R @ w
        cum_rets[label] = np.cumprod(1 + pr) * 100

    rows = []
    for i, date in enumerate(dates):
        row = {'date': str(date)}
        for label, cr in cum_rets.items():
            row[label] = round(float(cr[i]), 4)
        rows.append(row)
    return rows




# ══════════════════════════════════════════════════════════════
# Diversification Effect Analysis
# ══════════════════════════════════════════════════════════════

def diversification_analysis(
    portfolios_dict: Dict[str, Dict],
    mu: np.ndarray,
    cov: np.ndarray,
    tickers: List[str],
    ppy: int,
    rf: float,
) -> Dict:
    """
    Quantify diversification benefit of each portfolio vs equal-weight baseline.

    Metrics:
      diversification_ratio = weighted avg asset vol / portfolio vol
        > 1 means diversification reduces vol below weighted average
      vol_reduction_pct     = % vol reduction vs equal-weight
      effective_n           = 1 / sum(w_i^2)  — Herfindahl-based concentration
      concentration_pct     = top-3 weight concentration
    """
    n = len(tickers)
    asset_vols = np.sqrt(np.diag(cov)) * np.sqrt(ppy)  # annualised per-asset vol

    result = {}
    for label, port in portfolios_dict.items():
        w = np.array([port['weights'].get(t, 0.0) for t in tickers])
        port_vol = float(np.sqrt(w @ cov @ w) * np.sqrt(ppy))

        weighted_avg_vol = float(w @ asset_vols)
        div_ratio = weighted_avg_vol / port_vol if port_vol > 1e-8 else 1.0

        # Equal-weight baseline
        w_eq = np.ones(n) / n
        eq_vol = float(np.sqrt(w_eq @ cov @ w_eq) * np.sqrt(ppy))
        vol_reduction = (eq_vol - port_vol) / eq_vol * 100 if eq_vol > 0 else 0.0

        # Effective N (inverse Herfindahl)
        herfindahl = float(np.sum(w ** 2))
        effective_n = 1.0 / herfindahl if herfindahl > 0 else n

        # Top-3 concentration
        top3_weight = float(np.sort(w)[::-1][:3].sum()) * 100

        # Narrative
        if div_ratio >= 1.3:
            narrative = (
                f"Strong diversification — portfolio vol ({port_vol*100:.1f}%) is "
                f"{(div_ratio-1)*100:.0f}% below the weighted-average single-asset vol. "
                f"Effective N = {effective_n:.1f} of {n} assets are meaningfully contributing."
            )
        elif div_ratio >= 1.1:
            narrative = (
                f"Moderate diversification (ratio={div_ratio:.2f}). "
                f"Portfolio vol is reduced vs the weighted-average asset vol, "
                f"but concentrated positions (top-3 = {top3_weight:.0f}%) limit the benefit."
            )
        else:
            narrative = (
                f"Limited diversification (ratio={div_ratio:.2f}). "
                f"High correlation or concentration (effective N={effective_n:.1f}) "
                f"means diversification adds little vol reduction."
            )

        result[label] = {
            'port_vol_ann':       safe_float(port_vol * 100),
            'weighted_avg_vol':   safe_float(weighted_avg_vol * 100),
            'diversification_ratio': safe_float(div_ratio),
            'vol_reduction_vs_ew':   safe_float(vol_reduction),
            'effective_n':           safe_float(effective_n),
            'top3_concentration':    safe_float(top3_weight),
            'narrative':             narrative,
        }

    return result


# ══════════════════════════════════════════════════════════════
# Marginal Risk Contribution (MRC) per Asset
# ══════════════════════════════════════════════════════════════

def marginal_risk_contributions(
    portfolios_dict: Dict[str, Dict],
    cov: np.ndarray,
    tickers: List[str],
    ppy: int,
) -> Dict:
    """
    For each portfolio, compute each asset's:
      MRC (marginal risk contribution) = ∂σ_p/∂w_i = (Σw)_i / σ_p
      RC  (risk contribution)          = w_i × MRC_i
      RC% (% of total portfolio vol)   = RC_i / σ_p × 100
    """
    result = {}
    for label, port in portfolios_dict.items():
        w = np.array([port['weights'].get(t, 0.0) for t in tickers])
        port_var = float(w @ cov @ w)
        port_vol = np.sqrt(max(port_var, 1e-12)) * np.sqrt(ppy)

        sigma_w  = cov @ w                          # (N,) covariance × weights
        mrc      = sigma_w / (port_vol / np.sqrt(ppy))   # marginal risk contrib (monthly)
        rc       = w * mrc                           # risk contribution
        rc_ann   = rc * np.sqrt(ppy)                 # annualised
        rc_pct   = rc_ann / port_vol * 100 if port_vol > 0 else np.ones(len(tickers)) / len(tickers) * 100

        # Dominant risk contributor
        max_idx = int(np.argmax(rc_pct))

        asset_detail = []
        for i, t in enumerate(tickers):
            asset_detail.append({
                'ticker':  t,
                'weight':  safe_float(float(w[i]) * 100),
                'mrc':     safe_float(float(mrc[i]) * 100),
                'rc_pct':  safe_float(float(rc_pct[i])),
            })

        result[label] = {
            'assets':       asset_detail,
            'dominant':     tickers[max_idx],
            'dominant_pct': safe_float(float(rc_pct[max_idx])),
            'hhi':          safe_float(float(np.sum((rc_pct / 100) ** 2))),  # risk concentration
        }

    return result


# ══════════════════════════════════════════════════════════════
# Portfolio Comparison Insights
# ══════════════════════════════════════════════════════════════

def portfolio_comparison_insights(
    portfolios_dict: Dict[str, Dict],
    performance: Dict[str, Dict],
    tickers: List[str],
    rf: float,
) -> Dict:
    """
    Auto-generate narrative comparisons between portfolio strategies.

    Returns:
      best_sharpe    : which portfolio has highest Sharpe
      best_return    : which has highest return
      lowest_vol     : which has lowest vol
      lowest_drawdown: which has smallest max drawdown
      summary        : narrative text
      comparison_table: structured rows for display
    """
    if not performance:
        return {}

    metrics = {
        label: {
            'sharpe':   p.get('sharpe',             0.0),
            'return':   p.get('annualised_return',  0.0),
            'vol':      p.get('annualised_vol',      0.0),
            'mdd':      p.get('max_drawdown',        0.0),
            'sortino':  p.get('sortino',             0.0),
            'cvar':     p.get('cvar_95',             0.0),
        }
        for label, p in performance.items()
    }

    best_sharpe  = max(metrics, key=lambda k: metrics[k]['sharpe'])
    best_return  = max(metrics, key=lambda k: metrics[k]['return'])
    lowest_vol   = min(metrics, key=lambda k: metrics[k]['vol'])
    lowest_mdd   = min(metrics, key=lambda k: metrics[k]['mdd'])  # mdd is negative

    # Equal-weight as baseline
    eq = metrics.get('equal_weight', {})
    ms = metrics.get('max_sharpe', {})
    mv = metrics.get('min_var', {})

    insights = []

    # MaxSharpe vs EW
    if ms and eq:
        sharpe_diff = ms['sharpe'] - eq['sharpe']
        ret_diff    = ms['return'] - eq['return']
        if abs(sharpe_diff) > 0.05:
            direction = 'higher' if sharpe_diff > 0 else 'lower'
            insights.append(
                f"Max-Sharpe has {abs(sharpe_diff):.2f} {direction} Sharpe than Equal-Weight "
                f"({'outperforms' if sharpe_diff > 0 else 'underperforms'} on risk-adjusted basis)."
            )

    # MinVar safety
    if mv and eq:
        vol_saving = eq['vol'] - mv['vol']
        if abs(vol_saving) > 0.5:
            insights.append(
                f"Min-Variance reduces annualised vol by {abs(vol_saving):.1f}pp vs Equal-Weight "
                f"({mv['vol']:.1f}% vs {eq['vol']:.1f}%), at the cost of "
                f"{'lower' if mv['return'] < eq['return'] else 'comparable'} expected return."
            )

    # Drawdown
    if lowest_mdd in metrics:
        insights.append(
            f"{lowest_mdd.replace('_',' ').title()} has the smallest max drawdown "
            f"({metrics[lowest_mdd]['mdd']:.1f}%), offering the best downside protection."
        )

    summary = ' '.join(insights) if insights else (
        f"Best risk-adjusted portfolio: {best_sharpe.replace('_',' ').title()} "
        f"(Sharpe={metrics[best_sharpe]['sharpe']:.2f})."
    )

    comparison_table = [
        {
            'label':   label.replace('_', ' ').title(),
            'key':     label,
            'sharpe':  safe_float(m['sharpe']),
            'return':  safe_float(m['return']),
            'vol':     safe_float(m['vol']),
            'mdd':     safe_float(m['mdd']),
            'sortino': safe_float(m['sortino']),
            'cvar':    safe_float(m['cvar']),
        }
        for label, m in metrics.items()
    ]

    return {
        'best_sharpe':       best_sharpe,
        'best_return':       best_return,
        'lowest_vol':        lowest_vol,
        'lowest_drawdown':   lowest_mdd,
        'summary':           summary,
        'insights':          insights,
        'comparison_table':  comparison_table,
    }


# ══════════════════════════════════════════════════════════════
# Endpoint
# ══════════════════════════════════════════════════════════════

@router.post("/portfolio-optimization")
async def portfolio_optimization(request: PortfolioRequest):
    try:
        ppy = 12   # monthly data → 12 periods per year
        rf_period = request.riskFreeRate / ppy

        # ── 1. Prepare data ──────────────────────────────────────────
        tickers = request.tickers or ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'JNJ', 'XOM', 'GLD', 'TLT']
        asset_names: Dict[str, str] = {}

        if request.data:
            returns_df = pd.DataFrame(request.data)
            cols = request.returnCols or [c for c in returns_df.columns if c != (request.dateCol or 'date')]
            tickers = [c for c in cols if c in returns_df.columns]
            for col in tickers:
                returns_df[col] = pd.to_numeric(returns_df[col], errors='coerce')
            returns_df = returns_df.dropna(subset=tickers)
            asset_names = {t: t for t in tickers}
        else:
            returns_df, asset_names, _ = generate_portfolio_data(
                tickers=tickers,
                n_months=request.nMonths,
                seed=request.seed,
            )

        n_assets = len(tickers)
        n_obs    = len(returns_df)
        if n_obs < 12:
            raise HTTPException(status_code=400, detail=f"Need at least 12 observations, got {n_obs}.")
        if n_assets < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 assets.")

        R = returns_df[tickers].values / 100.0   # (T, N)
        mu_sample   = R.mean(axis=0)             # monthly
        cov_sample  = np.cov(R.T, ddof=1)        # monthly
        if cov_sample.ndim == 0:
            cov_sample = np.array([[float(cov_sample)]])

        # ── Ledoit-Wolf shrinkage for ill-conditioned covariance ──
        # Applies when n_obs < 3 × n_assets (small-sample regime)
        if n_obs < 3 * n_assets and n_assets > 1:
            trace_val = float(np.trace(cov_sample))
            shrink    = min(0.1 + (n_assets / n_obs) * 0.5, 0.5)
            cov_sample = (1 - shrink) * cov_sample + shrink * (trace_val / n_assets) * np.eye(n_assets)

        # ── 2. Special Portfolios ────────────────────────────────────
        mv_port  = min_variance_portfolio(mu_sample, cov_sample, rf_period, ppy, tickers,
                                          request.allowShortSelling, request.maxWeight, request.minWeight)
        ms_port  = max_sharpe_portfolio(mu_sample, cov_sample, rf_period, ppy, tickers,
                                        request.allowShortSelling, request.maxWeight, request.minWeight)
        eq_port  = equal_weight_portfolio(mu_sample, cov_sample, rf_period, ppy, tickers)

        # ── 3. Efficient Frontier (analytical) ──────────────────────
        frontier = compute_efficient_frontier(
            mu_sample, cov_sample, rf_period, ppy,
            tickers,
            n_points=request.nFrontierPoints,
            allow_short=request.allowShortSelling,
            max_w=request.maxWeight,
            min_w=request.minWeight,
        )

        # ── 4. Monte Carlo ───────────────────────────────────────────
        rng = np.random.default_rng(request.seed or 42)
        mc_ports = monte_carlo_portfolios(
            mu_sample, cov_sample, rf_period, ppy, tickers,
            n_sim=request.nMonteCarloPortfolios,
            allow_short=request.allowShortSelling,
            rng=rng,
        )

        # ── 5. Risk Parity ───────────────────────────────────────────
        rp_port = None
        if request.includeRiskParity:
            rp_port = risk_parity_portfolio(mu_sample, cov_sample, rf_period, ppy, tickers)

        # ── 6. CVaR ──────────────────────────────────────────────────
        cvar_port = None
        if request.includeCVaR:
            cvar_port = cvar_portfolio(returns_df, tickers, request.riskFreeRate, ppy)

        # ── 7. Black-Litterman ───────────────────────────────────────
        bl_port = None
        if request.includeBlackLitterman:
            mu_annual = mu_sample * ppy
            cov_annual = cov_sample * ppy
            views = request.blViews or []
            bl_port = black_litterman(
                mu_annual, cov_annual, views, tickers,
                request.riskFreeRate, ppy,
                allow_short=request.allowShortSelling,
                max_w=request.maxWeight,
                min_w=request.minWeight,
            )

        # ── 8. Rolling Optimization ──────────────────────────────────
        rolling = None
        if request.includeRolling:
            rolling = rolling_optimization(
                returns_df, tickers,
                request.riskFreeRate, ppy,
                window=request.rollingWindow,
                allow_short=request.allowShortSelling,
                max_w=request.maxWeight,
                min_w=request.minWeight,
            )

        # ── 9. Full performance stats ─────────────────────────────────
        perf_portfolios = {'max_sharpe': ms_port, 'min_var': mv_port, 'equal_weight': eq_port}
        if rp_port:  perf_portfolios['risk_parity'] = rp_port
        if cvar_port: perf_portfolios['cvar']       = cvar_port

        performance: Dict[str, Dict] = {}
        for label, port in perf_portfolios.items():
            performance[label] = full_performance_stats(
                port['weights'], returns_df, tickers,
                request.riskFreeRate, ppy,
            )

        # ── 10. Cumulative Returns Chart ──────────────────────────────
        weights_for_cum = {k: v['weights'] for k, v in perf_portfolios.items()}
        cum_ret_chart = cumulative_returns(weights_for_cum, returns_df, tickers, ppy)

        # ── 11. Asset-level stats ──────────────────────────────────────
        a_stats = asset_stats(returns_df, tickers, ppy)
        corr    = correlation_matrix(returns_df, tickers)

        # ── 12. Individual asset frontier points (for scatter) ────────
        individual_assets = []
        for t in tickers:
            s = a_stats[t]
            individual_assets.append({
                'ticker':     t,
                'name':       asset_names.get(t, t),
                'return':     s['return_annual'],
                'volatility': s['vol_annual'],
                'sharpe':     s['sharpe_annual'],
            })

        # ── 13a. Diversification analysis ────────────────────────────
        div_analysis = diversification_analysis(
            perf_portfolios, mu_sample, cov_sample, tickers, ppy, request.riskFreeRate
        )

        # ── 13b. Marginal risk contributions ──────────────────────────
        mrc_analysis = marginal_risk_contributions(
            perf_portfolios, cov_sample, tickers, ppy
        )

        # ── 13c. Comparison insights ──────────────────────────────────
        comparison_insights = portfolio_comparison_insights(
            perf_portfolios, performance, tickers, request.riskFreeRate
        )

        # ── 13. Date range ────────────────────────────────────────────
        date_col = request.dateCol or 'date'
        dates_list = returns_df[date_col].values if date_col in returns_df.columns else []
        date_range = f"{dates_list[0]} to {dates_list[-1]}" if len(dates_list) > 0 else ''

        # ── 14. Assemble response ──────────────────────────────────────
        result = {
            'tickers':          tickers,
            'asset_names':      asset_names,
            'n_assets':         n_assets,
            'n_observations':   n_obs,
            'date_range':       date_range,
            'risk_free_rate':   request.riskFreeRate,
            # Key portfolios
            'portfolios': {
                'max_sharpe':   ms_port,
                'min_var':      mv_port,
                'equal_weight': eq_port,
                'risk_parity':  rp_port,
                'cvar':         cvar_port,
                'black_litterman': bl_port,
            },
            # Frontier & simulation
            'efficient_frontier':   frontier,
            'monte_carlo':          mc_ports,
            # Rolling
            'rolling':              rolling,
            # Stats
            'performance':          performance,
            'asset_stats':          a_stats,
            'correlation':          corr,
            # Charts
            'charts': {
                'cumulative_returns': cum_ret_chart,
                'individual_assets':  individual_assets,
            },
            # Analytics
            'diversification':        div_analysis,
            'risk_contributions':     mrc_analysis,
            'comparison_insights':    comparison_insights,
            # Settings echo
            'settings': {
                'allow_short_selling':      request.allowShortSelling,
                'rolling_window':           request.rollingWindow,
                'n_monte_carlo':            request.nMonteCarloPortfolios,
                'n_frontier_points':        request.nFrontierPoints,
            },
        }

        return _to_native({'results': result})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")
