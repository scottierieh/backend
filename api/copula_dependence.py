from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats, optimize
import warnings
import traceback

warnings.filterwarnings('ignore')

router = APIRouter()


class CopulaRequest(BaseModel):
    data: Optional[List[Dict[str, Any]]] = None
    generate: bool = False
    assets: List[str] = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'XOM']
    nDays: int = 1000
    seed: Optional[int] = None
    tailQuantile: float = 0.05
    rollingWindow: int = 120


def _to_native(obj):
    if isinstance(obj, (np.integer,)): return int(obj)
    elif isinstance(obj, (np.floating,)):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    elif isinstance(obj, np.ndarray): return [_to_native(x) for x in obj.tolist()]
    elif isinstance(obj, np.bool_): return bool(obj)
    elif isinstance(obj, dict): return {k: _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [_to_native(x) for x in obj]
    return obj

def safe_float(val, default=0.0):
    try:
        if val is None: return default
        f = float(val)
        return default if (np.isnan(f) or np.isinf(f)) else f
    except Exception: return default


# ══════════════════════════════════════════════════════════════
# Asset Profiles & Data Generation
# ══════════════════════════════════════════════════════════════

PROFILES = {
    'AAPL':  {'name': 'Apple',     'mu': 0.0008, 'sig': 0.018},
    'MSFT':  {'name': 'Microsoft', 'mu': 0.0007, 'sig': 0.016},
    'GOOGL': {'name': 'Alphabet',  'mu': 0.0006, 'sig': 0.019},
    'AMZN':  {'name': 'Amazon',    'mu': 0.0007, 'sig': 0.020},
    'TSLA':  {'name': 'Tesla',     'mu': 0.001,  'sig': 0.035},
    'JPM':   {'name': 'JPMorgan',  'mu': 0.0005, 'sig': 0.015},
    'XOM':   {'name': 'Exxon',     'mu': 0.0004, 'sig': 0.018},
    'JNJ':   {'name': 'J&J',       'mu': 0.0003, 'sig': 0.010},
    'NVDA':  {'name': 'NVIDIA',    'mu': 0.0015, 'sig': 0.030},
    'SPY':   {'name': 'S&P 500',   'mu': 0.0004, 'sig': 0.011},
    'GLD':   {'name': 'Gold ETF',  'mu': 0.0002, 'sig': 0.009},
    'TLT':   {'name': 'Treasury',  'mu': 0.0001, 'sig': 0.012},
}

AVAILABLE_ASSETS = list(PROFILES.keys())


def generate_returns(assets: List[str], n_days: int, seed=None) -> pd.DataFrame:
    """
    Generate correlated returns with regime-switching:
    Normal regime (low correlation) + Crisis regime (high correlation, fat tails).
    This embeds real tail dependence for copula analysis to detect.
    """
    rng = np.random.default_rng(seed)
    k = len(assets)

    # Base correlation matrix
    base_corr = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            # Sector-based correlation
            r = rng.uniform(0.2, 0.5)
            base_corr[i, j] = base_corr[j, i] = r

    # Crisis correlation (much higher)
    crisis_corr = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            r = rng.uniform(0.6, 0.9)
            crisis_corr[i, j] = crisis_corr[j, i] = r

    vols = np.array([PROFILES.get(a, PROFILES['AAPL'])['sig'] for a in assets])
    means = np.array([PROFILES.get(a, PROFILES['AAPL'])['mu'] for a in assets])

    returns = np.zeros((n_days, k))

    for t in range(n_days):
        # Regime: ~8% crisis days
        is_crisis = rng.random() < 0.08

        if is_crisis:
            corr = crisis_corr
            vol_mult = rng.uniform(1.5, 3.0)  # vol spikes
            # Use t-distribution for fat tails
            df_t = rng.uniform(3, 6)
            L = np.linalg.cholesky(corr)
            z = rng.standard_t(df_t, size=k)
            eps = L @ z
            returns[t] = means - vol_mult * vols * 0.5 + vols * vol_mult * eps
        else:
            corr = base_corr
            L = np.linalg.cholesky(corr)
            z = rng.standard_normal(k)
            eps = L @ z
            returns[t] = means + vols * eps

    dates = pd.bdate_range(end='2025-04-30', periods=n_days)
    df = pd.DataFrame(returns, columns=assets)
    df.insert(0, 'date', dates.strftime('%Y-%m-%d'))
    return df


# ══════════════════════════════════════════════════════════════
# Copula Fitting
# ══════════════════════════════════════════════════════════════

def to_pseudo_obs(x: np.ndarray) -> np.ndarray:
    """Convert to pseudo-observations (empirical CDF, rank-based)."""
    n = len(x)
    ranks = stats.rankdata(x)
    return ranks / (n + 1)  # Avoids 0 and 1


def fit_gaussian_copula(u1: np.ndarray, u2: np.ndarray) -> Dict[str, Any]:
    """Gaussian copula: no tail dependence (λ_L = λ_U = 0)."""
    z1 = stats.norm.ppf(u1)
    z2 = stats.norm.ppf(u2)
    rho = np.corrcoef(z1, z2)[0, 1]
    rho = np.clip(rho, -0.999, 0.999)

    # Log-likelihood
    n = len(u1)
    R = np.array([[1, rho], [rho, 1]])
    R_inv = np.linalg.inv(R)
    det_R = np.linalg.det(R)
    ll = 0
    for i in range(n):
        z = np.array([z1[i], z2[i]])
        ll += -0.5 * np.log(det_R) - 0.5 * z @ (R_inv - np.eye(2)) @ z
    aic = -2 * ll + 2 * 1
    bic = -2 * ll + np.log(n) * 1

    return {
        'type': 'Gaussian', 'params': {'rho': safe_float(rho)},
        'll': safe_float(ll), 'aic': safe_float(aic), 'bic': safe_float(bic),
        'tail_lower': 0.0, 'tail_upper': 0.0,
    }


def fit_student_t_copula(u1: np.ndarray, u2: np.ndarray) -> Dict[str, Any]:
    """Student-t copula: symmetric tail dependence λ_L = λ_U > 0."""
    z1 = stats.norm.ppf(u1)
    z2 = stats.norm.ppf(u2)
    rho = np.corrcoef(z1, z2)[0, 1]
    rho = np.clip(rho, -0.999, 0.999)

    # Estimate df by profile likelihood
    def neg_ll_t(log_nu):
        nu = np.exp(log_nu) + 2.01  # nu > 2
        t1 = stats.t.ppf(u1, df=nu)
        t2 = stats.t.ppf(u2, df=nu)
        rho_t = np.corrcoef(t1, t2)[0, 1]
        rho_t = np.clip(rho_t, -0.999, 0.999)
        n = len(u1)
        ll = 0
        for i in range(n):
            x = np.array([t1[i], t2[i]])
            R = np.array([[1, rho_t], [rho_t, 1]])
            try:
                ll += stats.multivariate_t.logpdf(x, loc=[0, 0], shape=R, df=nu)
                ll -= stats.t.logpdf(t1[i], df=nu) + stats.t.logpdf(t2[i], df=nu)
            except Exception:
                pass
        return -ll

    try:
        res = optimize.minimize_scalar(neg_ll_t, bounds=(0, 4), method='bounded')
        nu = np.exp(res.x) + 2.01
        ll = -res.fun
    except Exception:
        nu = 5.0
        ll = 0.0

    n = len(u1)
    aic = -2 * ll + 2 * 2
    bic = -2 * ll + np.log(n) * 2

    # Tail dependence: λ = 2·t_{ν+1}(-√((ν+1)(1-ρ)/(1+ρ)))
    if nu > 0 and abs(rho) < 1:
        arg = -np.sqrt((nu + 1) * (1 - rho) / (1 + rho))
        tail = 2 * stats.t.cdf(arg, df=nu + 1)
    else:
        tail = 0.0

    return {
        'type': 'Student-t', 'params': {'rho': safe_float(rho), 'nu': safe_float(nu)},
        'll': safe_float(ll), 'aic': safe_float(aic), 'bic': safe_float(bic),
        'tail_lower': safe_float(tail), 'tail_upper': safe_float(tail),
    }


def fit_clayton_copula(u1: np.ndarray, u2: np.ndarray) -> Dict[str, Any]:
    """Clayton copula: lower tail dependence λ_L = 2^{-1/θ}, λ_U = 0."""
    def neg_ll(log_theta):
        theta = np.exp(log_theta) + 0.01
        n = len(u1)
        ll = 0
        for i in range(n):
            v1, v2 = u1[i], u2[i]
            if v1 <= 0 or v2 <= 0: continue
            c = (1 + theta) * (v1 * v2) ** (-theta - 1) * (v1 ** (-theta) + v2 ** (-theta) - 1) ** (-1 / theta - 2)
            if c > 0:
                ll += np.log(c)
        return -ll

    try:
        res = optimize.minimize_scalar(neg_ll, bounds=(-2, 5), method='bounded')
        theta = np.exp(res.x) + 0.01
        ll = -res.fun
    except Exception:
        theta = 1.0
        ll = 0.0

    n = len(u1)
    aic = -2 * ll + 2 * 1
    bic = -2 * ll + np.log(n) * 1
    tail_l = 2 ** (-1 / theta) if theta > 0 else 0

    return {
        'type': 'Clayton', 'params': {'theta': safe_float(theta)},
        'll': safe_float(ll), 'aic': safe_float(aic), 'bic': safe_float(bic),
        'tail_lower': safe_float(tail_l), 'tail_upper': 0.0,
    }


def fit_gumbel_copula(u1: np.ndarray, u2: np.ndarray) -> Dict[str, Any]:
    """Gumbel copula: upper tail dependence λ_U = 2 - 2^{1/θ}, λ_L = 0."""
    def neg_ll(log_theta):
        theta = np.exp(log_theta) + 1.001  # θ >= 1
        n = len(u1)
        ll = 0
        for i in range(n):
            v1, v2 = u1[i], u2[i]
            if v1 <= 0 or v2 <= 0 or v1 >= 1 or v2 >= 1: continue
            t1 = (-np.log(v1)) ** theta
            t2 = (-np.log(v2)) ** theta
            A = (t1 + t2) ** (1 / theta)
            C = np.exp(-A)
            # Density
            dC = C * (t1 + t2) ** (1 / theta - 2) * ((-np.log(v1)) ** (theta - 1)) * ((-np.log(v2)) ** (theta - 1)) / (v1 * v2) * (A + theta - 1)
            if dC > 0:
                ll += np.log(dC)
        return -ll

    try:
        res = optimize.minimize_scalar(neg_ll, bounds=(-1, 4), method='bounded')
        theta = np.exp(res.x) + 1.001
        ll = -res.fun
    except Exception:
        theta = 1.5
        ll = 0.0

    n = len(u1)
    aic = -2 * ll + 2 * 1
    bic = -2 * ll + np.log(n) * 1
    tail_u = 2 - 2 ** (1 / theta) if theta > 1 else 0

    return {
        'type': 'Gumbel', 'params': {'theta': safe_float(theta)},
        'll': safe_float(ll), 'aic': safe_float(aic), 'bic': safe_float(bic),
        'tail_lower': 0.0, 'tail_upper': safe_float(tail_u),
    }


# ══════════════════════════════════════════════════════════════
# Empirical Tail Dependence
# ══════════════════════════════════════════════════════════════

def empirical_tail_dep(u1: np.ndarray, u2: np.ndarray, q: float = 0.05) -> Dict[str, float]:
    """
    Empirical tail dependence:
    λ_L(q) = P(U2 < q | U1 < q)
    λ_U(q) = P(U2 > 1-q | U1 > 1-q)
    """
    n = len(u1)
    lower_mask = (u1 < q) & (u2 < q)
    cond_lower = u1 < q
    lambda_l = lower_mask.sum() / max(cond_lower.sum(), 1)

    upper_mask = (u1 > 1 - q) & (u2 > 1 - q)
    cond_upper = u1 > 1 - q
    lambda_u = upper_mask.sum() / max(cond_upper.sum(), 1)

    return {'lower': safe_float(lambda_l), 'upper': safe_float(lambda_u)}


# ══════════════════════════════════════════════════════════════
# Main Endpoint
# ══════════════════════════════════════════════════════════════

@router.post("/copula-dependence")
async def copula_dependence(request: CopulaRequest):
    try:
        # ── 1. Data ──
        if request.generate or not request.data:
            assets = [a for a in request.assets if a in PROFILES][:8]
            if len(assets) < 2:
                assets = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'XOM']
            df = generate_returns(assets, request.nDays, request.seed)
        else:
            df = pd.DataFrame(request.data)
            date_col = None
            for c in ['date', 'Date', 'timestamp']:
                if c in df.columns:
                    date_col = c
                    break
            assets = [c for c in df.columns if c != date_col and not c.startswith('_')]
            for a in assets:
                df[a] = pd.to_numeric(df[a], errors='coerce')
            df = df.dropna(subset=assets)

        n = len(df)
        k = len(assets)
        if n < 50 or k < 2:
            raise HTTPException(status_code=400, detail=f"Need >=50 rows and >=2 assets. Got {n} rows, {k} assets.")

        returns = df[assets].values

        # ── 2. Pseudo-observations ──
        U = np.column_stack([to_pseudo_obs(returns[:, j]) for j in range(k)])

        # ── 3. Pairwise Copula Analysis ──
        pairs = []
        for i in range(k):
            for j in range(i + 1, k):
                u1, u2 = U[:, i], U[:, j]

                # Fit all 4 copulas
                gauss = fit_gaussian_copula(u1, u2)
                t_cop = fit_student_t_copula(u1, u2)
                clay = fit_clayton_copula(u1, u2)
                gumb = fit_gumbel_copula(u1, u2)

                fits = [gauss, t_cop, clay, gumb]
                best = min(fits, key=lambda x: x['bic'])

                # Empirical tail dep
                emp = empirical_tail_dep(u1, u2, request.tailQuantile)

                # Kendall tau
                tau, tau_p = stats.kendalltau(returns[:, i], returns[:, j])

                # Pearson
                pearson = np.corrcoef(returns[:, i], returns[:, j])[0, 1]

                pairs.append({
                    'asset1': assets[i],
                    'asset2': assets[j],
                    'pearson': safe_float(pearson),
                    'kendall_tau': safe_float(tau),
                    'empirical_tail_lower': emp['lower'],
                    'empirical_tail_upper': emp['upper'],
                    'best_copula': best['type'],
                    'best_bic': best['bic'],
                    'copula_fits': fits,
                })

        # ── 4. Tail Dependence Matrices ──
        tail_lower_mat = np.zeros((k, k))
        tail_upper_mat = np.zeros((k, k))
        pearson_mat = np.zeros((k, k))

        for p in pairs:
            i = assets.index(p['asset1'])
            j = assets.index(p['asset2'])
            best_fit = min(p['copula_fits'], key=lambda x: x['bic'])
            tail_lower_mat[i, j] = tail_lower_mat[j, i] = best_fit['tail_lower']
            tail_upper_mat[i, j] = tail_upper_mat[j, i] = best_fit['tail_upper']
            pearson_mat[i, j] = pearson_mat[j, i] = p['pearson']

        np.fill_diagonal(tail_lower_mat, 1.0)
        np.fill_diagonal(tail_upper_mat, 1.0)
        np.fill_diagonal(pearson_mat, 1.0)

        # ── 5. Rolling Tail Dependence ──
        w = request.rollingWindow
        rolling_chart = []
        if n > w + 20:
            step = max(1, (n - w) // 80)
            for t in range(w, n, step):
                window_ret = returns[t - w:t]
                u1_w = to_pseudo_obs(window_ret[:, 0])
                u2_w = to_pseudo_obs(window_ret[:, 1])
                emp_w = empirical_tail_dep(u1_w, u2_w, request.tailQuantile)
                tau_w, _ = stats.kendalltau(window_ret[:, 0], window_ret[:, 1])
                pearson_w = np.corrcoef(window_ret[:, 0], window_ret[:, 1])[0, 1]

                date_str = str(df.iloc[t].get('date', t))
                rolling_chart.append({
                    'date': date_str,
                    'pearson': safe_float(pearson_w),
                    'kendall_tau': safe_float(tau_w),
                    'tail_lower': emp_w['lower'],
                    'tail_upper': emp_w['upper'],
                })

        # ── 6. Charts ──

        # Copula scatter for first pair
        copula_scatter = []
        if k >= 2:
            step_s = max(1, n // 400)
            for t in range(0, n, step_s):
                copula_scatter.append({
                    'u1': safe_float(U[t, 0]),
                    'u2': safe_float(U[t, 1]),
                    'asset1': assets[0],
                    'asset2': assets[1],
                    'is_lower_tail': bool(U[t, 0] < request.tailQuantile and U[t, 1] < request.tailQuantile),
                    'is_upper_tail': bool(U[t, 0] > 1 - request.tailQuantile and U[t, 1] > 1 - request.tailQuantile),
                })

        # Model comparison chart (all pairs)
        model_comparison = []
        for p in pairs:
            for fit in p['copula_fits']:
                model_comparison.append({
                    'pair': f"{p['asset1']}-{p['asset2']}",
                    'copula': fit['type'],
                    'bic': fit['bic'],
                    'aic': fit['aic'],
                    'tail_lower': fit['tail_lower'],
                    'tail_upper': fit['tail_upper'],
                })

        # Tail dep heatmap data
        heatmap_lower = []
        heatmap_upper = []
        for i in range(k):
            for j in range(k):
                heatmap_lower.append({'row': assets[i], 'col': assets[j], 'value': safe_float(tail_lower_mat[i, j])})
                heatmap_upper.append({'row': assets[i], 'col': assets[j], 'value': safe_float(tail_upper_mat[i, j])})

        # Pair summary chart
        pair_chart = []
        for p in pairs:
            best = min(p['copula_fits'], key=lambda x: x['bic'])
            pair_chart.append({
                'pair': f"{p['asset1']}-{p['asset2']}",
                'pearson': p['pearson'],
                'kendall_tau': p['kendall_tau'],
                'tail_lower': best['tail_lower'],
                'tail_upper': best['tail_upper'],
                'empirical_lower': p['empirical_tail_lower'],
                'empirical_upper': p['empirical_tail_upper'],
                'best_copula': best['type'],
            })

        # ── 7. Summary ──
        avg_tail_l = safe_float(np.mean([p['empirical_tail_lower'] for p in pairs]))
        avg_tail_u = safe_float(np.mean([p['empirical_tail_upper'] for p in pairs]))
        max_tail_l = max(p['empirical_tail_lower'] for p in pairs)
        max_pair_l = [p for p in pairs if p['empirical_tail_lower'] == max_tail_l][0]

        results = {
            'n_observations': n,
            'n_assets': k,
            'assets': assets,
            'tail_quantile': request.tailQuantile,
            'rolling_window': w,
            'summary': {
                'avg_lower_tail': avg_tail_l,
                'avg_upper_tail': avg_tail_u,
                'max_lower_tail': safe_float(max_tail_l),
                'max_lower_pair': f"{max_pair_l['asset1']}-{max_pair_l['asset2']}",
                'avg_pearson': safe_float(np.mean([p['pearson'] for p in pairs])),
                'avg_kendall': safe_float(np.mean([p['kendall_tau'] for p in pairs])),
                'n_pairs': len(pairs),
            },
            'pairs': pair_chart,
            'charts': {
                'copula_scatter': copula_scatter,
                'rolling': rolling_chart,
                'model_comparison': model_comparison,
                'heatmap_lower': heatmap_lower,
                'heatmap_upper': heatmap_upper,
            },
        }

        return _to_native({'results': results})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")
