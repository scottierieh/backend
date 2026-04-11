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


class StressTestRequest(BaseModel):
    data: Optional[List[Dict[str, Any]]] = None
    generate: bool = False
    assets: List[str] = ['AAPL', 'MSFT', 'GOOGL', 'JPM', 'XOM']
    nDays: int = 750
    seed: Optional[int] = None
    # Portfolio weights (equal if empty)
    weights: Optional[List[float]] = None
    # Scenario config
    nSimPaths: int = 1000
    stressHorizon: int = 60       # days per stress path
    scenarios: List[str] = ['gfc_2008', 'covid_2020', 'rate_shock', 'custom']
    # Custom scenario params
    customVolMult: float = 3.0
    customCorrOverride: float = 0.80
    customTailDf: float = 3.0
    customDriftShock: float = -0.003


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
# Asset Profiles
# ══════════════════════════════════════════════════════════════

PROFILES = {
    'AAPL':  {'name': 'Apple',     'mu': 0.0008, 'sig': 0.018},
    'MSFT':  {'name': 'Microsoft', 'mu': 0.0007, 'sig': 0.016},
    'GOOGL': {'name': 'Alphabet',  'mu': 0.0006, 'sig': 0.019},
    'AMZN':  {'name': 'Amazon',    'mu': 0.0007, 'sig': 0.020},
    'TSLA':  {'name': 'Tesla',     'mu': 0.001,  'sig': 0.035},
    'JPM':   {'name': 'JPMorgan',  'mu': 0.0005, 'sig': 0.015},
    'XOM':   {'name': 'Exxon',     'mu': 0.0004, 'sig': 0.018},
    'NVDA':  {'name': 'NVIDIA',    'mu': 0.0015, 'sig': 0.030},
    'SPY':   {'name': 'S&P 500',   'mu': 0.0004, 'sig': 0.011},
    'GLD':   {'name': 'Gold',      'mu': 0.0002, 'sig': 0.009},
    'TLT':   {'name': 'Treasury',  'mu': 0.0001, 'sig': 0.012},
    'META':  {'name': 'Meta',      'mu': 0.0008, 'sig': 0.022},
}


def generate_returns(assets, n_days, seed=None):
    rng = np.random.default_rng(seed)
    k = len(assets)
    corr = np.eye(k)
    for i in range(k):
        for j in range(i + 1, k):
            corr[i, j] = corr[j, i] = rng.uniform(0.2, 0.5)
    vols = np.array([PROFILES.get(a, PROFILES['AAPL'])['sig'] for a in assets])
    means = np.array([PROFILES.get(a, PROFILES['AAPL'])['mu'] for a in assets])
    L = np.linalg.cholesky(corr)
    z = rng.standard_normal((n_days, k))
    returns = means + (z @ L.T) * vols
    dates = pd.bdate_range(end='2025-04-30', periods=n_days)
    df = pd.DataFrame(returns, columns=assets)
    df.insert(0, 'date', dates.strftime('%Y-%m-%d'))
    return df


# ══════════════════════════════════════════════════════════════
# Scenario Generators — The "Synthetic Data" Core
# ══════════════════════════════════════════════════════════════

SCENARIO_DEFS = {
    'gfc_2008': {
        'label': 'GFC 2008',
        'desc': 'Global Financial Crisis — extreme correlation, 3.5× vol, fat tails (df=3)',
        'vol_mult': 3.5,
        'corr_override': 0.85,
        'tail_df': 3.0,
        'drift_shock': -0.004,
        'recovery_start': 0.7,  # recovery begins at 70% of horizon
    },
    'covid_2020': {
        'label': 'COVID-19 Crash',
        'desc': 'Sharp V-shaped crash — 4× vol spike then rapid mean-reversion',
        'vol_mult': 4.0,
        'corr_override': 0.80,
        'tail_df': 4.0,
        'drift_shock': -0.006,
        'recovery_start': 0.35,
    },
    'rate_shock': {
        'label': 'Rate Shock',
        'desc': 'Interest rate spike — bonds and equities fall together, moderate vol',
        'vol_mult': 2.0,
        'corr_override': 0.70,
        'tail_df': 5.0,
        'drift_shock': -0.002,
        'recovery_start': 0.8,
    },
    'custom': {
        'label': 'Custom Scenario',
        'desc': 'User-defined stress parameters',
        'vol_mult': 3.0,
        'corr_override': 0.80,
        'tail_df': 3.0,
        'drift_shock': -0.003,
        'recovery_start': 0.6,
    },
}


def generate_stress_paths(
    means: np.ndarray,
    vols: np.ndarray,
    base_corr: np.ndarray,
    scenario: Dict[str, Any],
    n_paths: int,
    horizon: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Generate synthetic crisis return paths.
    Returns: (n_paths, horizon, n_assets) array
    """
    k = len(means)
    vol_mult = scenario['vol_mult']
    corr_over = scenario['corr_override']
    tail_df = scenario['tail_df']
    drift = scenario['drift_shock']
    recovery_pct = scenario.get('recovery_start', 0.7)
    recovery_day = int(horizon * recovery_pct)

    # Build stress correlation matrix
    stress_corr = np.full((k, k), corr_over)
    np.fill_diagonal(stress_corr, 1.0)

    # Ensure positive definite
    eigvals = np.linalg.eigvalsh(stress_corr)
    if eigvals.min() < 1e-6:
        stress_corr += np.eye(k) * (1e-6 - eigvals.min())

    L = np.linalg.cholesky(stress_corr)
    stress_vols = vols * vol_mult

    paths = np.zeros((n_paths, horizon, k))

    for p in range(n_paths):
        for t in range(horizon):
            # Time-varying vol: peaks mid-crisis, eases in recovery
            if t < recovery_day:
                progress = t / max(recovery_day, 1)
                vol_scale = 0.5 + 0.5 * np.sin(progress * np.pi)  # ramp up then peak
                current_drift = drift * (1 + progress)
            else:
                decay = (t - recovery_day) / max(horizon - recovery_day, 1)
                vol_scale = 1.0 - 0.6 * decay  # vol decays
                current_drift = drift * (1 - decay * 0.8)  # drift recovers

            # Fat-tailed innovations
            z = rng.standard_t(tail_df, size=k)
            eps = L @ z

            paths[p, t] = current_drift + stress_vols * vol_scale * eps

    return paths


# ══════════════════════════════════════════════════════════════
# Risk Metrics
# ══════════════════════════════════════════════════════════════

def compute_portfolio_metrics(
    paths: np.ndarray,
    weights: np.ndarray,
    confidence: float = 0.05,
) -> Dict[str, Any]:
    """
    paths: (n_paths, horizon, n_assets)
    Returns portfolio-level risk metrics.
    """
    n_paths, horizon, k = paths.shape

    # Portfolio returns per path: (n_paths, horizon)
    port_returns = np.sum(paths * weights, axis=2)

    # Cumulative returns per path
    cum_returns = np.cumprod(1 + port_returns, axis=1)
    terminal = cum_returns[:, -1] - 1  # total return per path

    # Daily portfolio returns (flatten for VaR)
    daily_flat = port_returns.flatten()

    # VaR
    var_daily = -np.percentile(daily_flat, confidence * 100)
    var_period = -np.percentile(terminal, confidence * 100)

    # CVaR (Expected Shortfall)
    tail_mask = daily_flat <= -var_daily
    cvar_daily = -daily_flat[tail_mask].mean() if tail_mask.sum() > 0 else var_daily

    tail_mask_t = terminal <= -var_period
    cvar_period = -terminal[tail_mask_t].mean() if tail_mask_t.sum() > 0 else var_period

    # Max drawdown per path
    max_drawdowns = []
    for p in range(n_paths):
        cummax = np.maximum.accumulate(cum_returns[p])
        dd = (cum_returns[p] - cummax) / cummax
        max_drawdowns.append(dd.min())
    max_drawdowns = np.array(max_drawdowns)

    # Recovery time (days to get back to starting value)
    recovery_days = []
    for p in range(n_paths):
        below = cum_returns[p] < 1.0
        if below.any():
            last_below = np.where(below)[0][-1]
            recovery_days.append(last_below)
        else:
            recovery_days.append(0)
    recovery_days = np.array(recovery_days)

    # Worst path
    worst_idx = np.argmin(terminal)

    return {
        'var_daily_pct': safe_float(var_daily * 100),
        'var_period_pct': safe_float(var_period * 100),
        'cvar_daily_pct': safe_float(cvar_daily * 100),
        'cvar_period_pct': safe_float(cvar_period * 100),
        'mean_return_pct': safe_float(terminal.mean() * 100),
        'median_return_pct': safe_float(np.median(terminal) * 100),
        'worst_return_pct': safe_float(terminal.min() * 100),
        'best_return_pct': safe_float(terminal.max() * 100),
        'mean_max_dd_pct': safe_float(max_drawdowns.mean() * 100),
        'worst_max_dd_pct': safe_float(max_drawdowns.min() * 100),
        'p95_max_dd_pct': safe_float(np.percentile(max_drawdowns, 5) * 100),
        'mean_recovery_days': safe_float(recovery_days.mean()),
        'p95_recovery_days': safe_float(np.percentile(recovery_days, 95)),
        'prob_loss_gt_10': safe_float((terminal < -0.10).mean() * 100),
        'prob_loss_gt_20': safe_float((terminal < -0.20).mean() * 100),
        'prob_loss_gt_30': safe_float((terminal < -0.30).mean() * 100),
    }


# ══════════════════════════════════════════════════════════════
# Main Endpoint
# ══════════════════════════════════════════════════════════════

@router.post("/stress-test")
async def stress_test(request: StressTestRequest):
    try:
        rng = np.random.default_rng(request.seed)

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
                if c in df.columns: date_col = c; break
            assets = [c for c in df.columns if c != date_col and not c.startswith('_')]
            for a in assets:
                df[a] = pd.to_numeric(df[a], errors='coerce')
            df = df.dropna(subset=assets)

        k = len(assets)
        n = len(df)
        if n < 30 or k < 2:
            raise HTTPException(status_code=400, detail=f"Need >=30 rows, >=2 assets. Got {n}, {k}.")

        returns = df[assets].values

        # Weights
        if request.weights and len(request.weights) == k:
            weights = np.array(request.weights)
            weights = weights / weights.sum()
        else:
            weights = np.ones(k) / k

        # ── 2. Learn Normal Market Profile ──
        means = returns.mean(axis=0)
        vols = returns.std(axis=0)
        corr = np.corrcoef(returns.T)
        skews = [safe_float(stats.skew(returns[:, j])) for j in range(k)]
        kurts = [safe_float(stats.kurtosis(returns[:, j])) for j in range(k)]

        normal_port = returns @ weights
        normal_metrics = {
            'annual_return_pct': safe_float(normal_port.mean() * 252 * 100),
            'annual_vol_pct': safe_float(normal_port.std() * np.sqrt(252) * 100),
            'sharpe': safe_float(normal_port.mean() / normal_port.std() * np.sqrt(252)) if normal_port.std() > 0 else 0,
            'var_95_daily_pct': safe_float(-np.percentile(normal_port, 5) * 100),
            'max_dd_pct': safe_float(((np.maximum.accumulate(np.cumprod(1 + normal_port)) - np.cumprod(1 + normal_port)) / np.maximum.accumulate(np.cumprod(1 + normal_port))).max() * 100),
        }

        # ── 3. Generate Stress Scenarios ──
        horizon = request.stressHorizon
        n_paths = min(request.nSimPaths, 2000)

        scenario_results = []
        all_fan_data = {}

        for sc_key in request.scenarios:
            sc_def = SCENARIO_DEFS.get(sc_key, SCENARIO_DEFS['custom']).copy()

            # Override custom params
            if sc_key == 'custom':
                sc_def['vol_mult'] = request.customVolMult
                sc_def['corr_override'] = request.customCorrOverride
                sc_def['tail_df'] = request.customTailDf
                sc_def['drift_shock'] = request.customDriftShock

            paths = generate_stress_paths(means, vols, corr, sc_def, n_paths, horizon, rng)
            metrics = compute_portfolio_metrics(paths, weights)

            # Fan chart data (percentile bands)
            port_paths = np.sum(paths * weights, axis=2)  # (n_paths, horizon)
            cum_paths = np.cumprod(1 + port_paths, axis=1)

            fan = []
            for t in range(horizon):
                vals = cum_paths[:, t]
                fan.append({
                    'day': t + 1,
                    'p5': safe_float((np.percentile(vals, 5) - 1) * 100),
                    'p25': safe_float((np.percentile(vals, 25) - 1) * 100),
                    'median': safe_float((np.median(vals) - 1) * 100),
                    'p75': safe_float((np.percentile(vals, 75) - 1) * 100),
                    'p95': safe_float((np.percentile(vals, 95) - 1) * 100),
                    'worst': safe_float((vals.min() - 1) * 100),
                })

            # Terminal return distribution
            terminal = (cum_paths[:, -1] - 1) * 100
            bins = np.linspace(terminal.min(), terminal.max(), 40)
            dist = []
            for j in range(len(bins) - 1):
                count = int(((terminal >= bins[j]) & (terminal < bins[j + 1])).sum())
                dist.append({'range': f'{(bins[j] + bins[j + 1]) / 2:.1f}', 'count': count})

            # Per-asset stress stats
            asset_stress = []
            for ai in range(k):
                asset_paths = paths[:, :, ai]  # (n_paths, horizon)
                cum_a = np.cumprod(1 + asset_paths, axis=1)
                term_a = cum_a[:, -1] - 1
                asset_stress.append({
                    'asset': assets[ai],
                    'mean_return_pct': safe_float(term_a.mean() * 100),
                    'var_95_pct': safe_float(-np.percentile(term_a, 5) * 100),
                    'worst_pct': safe_float(term_a.min() * 100),
                })

            scenario_results.append({
                'key': sc_key,
                'label': sc_def['label'],
                'desc': sc_def['desc'],
                'params': {
                    'vol_mult': sc_def['vol_mult'],
                    'corr_override': sc_def['corr_override'],
                    'tail_df': sc_def['tail_df'],
                    'drift_shock': sc_def['drift_shock'],
                },
                'metrics': metrics,
                'asset_stress': asset_stress,
                'fan_chart': fan,
                'terminal_dist': dist,
            })

        # ── 4. Comparison Chart ──
        comparison = []
        for sc in scenario_results:
            m = sc['metrics']
            comparison.append({
                'scenario': sc['label'],
                'var_95': m['var_period_pct'],
                'cvar_95': m['cvar_period_pct'],
                'worst_dd': m['worst_max_dd_pct'],
                'mean_return': m['mean_return_pct'],
                'prob_loss_20': m['prob_loss_gt_20'],
            })

        # Normal correlation heatmap
        corr_heatmap = []
        for i in range(k):
            for j in range(k):
                corr_heatmap.append({'row': assets[i], 'col': assets[j], 'normal': safe_float(corr[i, j])})

        # ── 5. Response ──
        results = {
            'n_observations': n,
            'n_assets': k,
            'assets': assets,
            'weights': [safe_float(w) for w in weights],
            'n_paths': n_paths,
            'stress_horizon': horizon,
            'normal_profile': {
                'means': [safe_float(m * 252 * 100) for m in means],
                'vols': [safe_float(v * np.sqrt(252) * 100) for v in vols],
                'skewness': skews,
                'kurtosis': kurts,
                'metrics': normal_metrics,
            },
            'scenarios': scenario_results,
            'charts': {
                'comparison': comparison,
                'correlation_heatmap': corr_heatmap,
            },
        }

        return _to_native({'results': results})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")
