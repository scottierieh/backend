from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy import optimize
import warnings
import traceback

warnings.filterwarnings('ignore')

router = APIRouter()


class MMMRequest(BaseModel):
    data: Optional[List[Dict[str, Any]]] = None
    generate: bool = False
    nWeeks: int = 104
    seed: Optional[int] = None
    # Column mapping
    colDate: Optional[str] = None
    colTarget: Optional[str] = None
    colChannels: Optional[List[str]] = None
    colControls: Optional[List[str]] = None
    # Config
    adstockMaxLag: int = 8
    totalBudget: Optional[float] = None  # for optimization


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
# Data Generation
# ══════════════════════════════════════════════════════════════

CHANNEL_DEFS = {
    'TV':          {'weekly_spend': (5000, 20000),  'adstock_decay': 0.70, 'saturation_k': 0.0003, 'beta': 3.5,  'noise': 0.05},
    'Digital':     {'weekly_spend': (3000, 15000),  'adstock_decay': 0.40, 'saturation_k': 0.0005, 'beta': 4.0,  'noise': 0.08},
    'Social':      {'weekly_spend': (1000, 8000),   'adstock_decay': 0.30, 'saturation_k': 0.0008, 'beta': 2.5,  'noise': 0.10},
    'Search':      {'weekly_spend': (2000, 12000),  'adstock_decay': 0.20, 'saturation_k': 0.0004, 'beta': 5.0,  'noise': 0.06},
    'Print':       {'weekly_spend': (500, 5000),    'adstock_decay': 0.50, 'saturation_k': 0.0006, 'beta': 1.5,  'noise': 0.12},
    'Radio':       {'weekly_spend': (1000, 6000),   'adstock_decay': 0.55, 'saturation_k': 0.0005, 'beta': 2.0,  'noise': 0.09},
}


def geometric_adstock(x: np.ndarray, decay: float, max_lag: int = 8) -> np.ndarray:
    """Apply geometric adstock transformation: effect carries over with exponential decay."""
    n = len(x)
    adstocked = np.zeros(n)
    for t in range(n):
        for lag in range(min(max_lag, t + 1)):
            adstocked[t] += x[t - lag] * (decay ** lag)
    return adstocked


def hill_saturation(x: np.ndarray, k: float, n_exp: float = 2.0) -> np.ndarray:
    """Hill function saturation: y = x^n / (k^n + x^n). Returns 0-1 scale."""
    half_sat = 1.0 / (k + 1e-10)
    return np.power(x, n_exp) / (np.power(half_sat, n_exp) + np.power(x, n_exp))


def logistic_saturation(x: np.ndarray, lam: float) -> np.ndarray:
    """Logistic saturation: y = 1 - exp(-lam * x)."""
    return 1 - np.exp(-lam * x)


def generate_mmm_data(n_weeks: int, seed=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start='2023-01-01', periods=n_weeks, freq='W-MON')

    # Base sales (trend + seasonality)
    trend = np.linspace(50000, 65000, n_weeks)
    seasonal = 8000 * np.sin(2 * np.pi * np.arange(n_weeks) / 52)
    base_sales = trend + seasonal

    df = pd.DataFrame({'date': dates.strftime('%Y-%m-%d')})

    # Generate channel spend and contributions
    total_media_effect = np.zeros(n_weeks)
    true_contributions = {}

    for ch_name, ch_def in CHANNEL_DEFS.items():
        spend = rng.uniform(ch_def['weekly_spend'][0], ch_def['weekly_spend'][1], n_weeks)
        spend = spend * (1 + 0.15 * np.sin(2 * np.pi * np.arange(n_weeks) / 52 + rng.uniform(0, 2 * np.pi)))
        spend = np.maximum(spend, 0)

        # Apply adstock
        adstocked = geometric_adstock(spend, ch_def['adstock_decay'])
        # Apply saturation
        saturated = logistic_saturation(adstocked, ch_def['saturation_k'])
        # Contribution = beta * saturated
        contribution = ch_def['beta'] * saturated * 10000

        total_media_effect += contribution
        true_contributions[ch_name] = contribution
        df[f'spend_{ch_name.lower()}'] = np.round(spend, 2)

    # Control variables
    df['holiday'] = ((np.arange(n_weeks) % 52) >= 48).astype(float)  # last 4 weeks of year
    df['competitor_promo'] = rng.binomial(1, 0.15, n_weeks).astype(float)

    # Target: base + media + controls + noise
    holiday_effect = df['holiday'].values * 12000
    competitor_effect = df['competitor_promo'].values * (-5000)
    noise = rng.normal(0, 3000, n_weeks)

    df['revenue'] = np.round(base_sales + total_media_effect + holiday_effect + competitor_effect + noise, 2)
    df['revenue'] = np.maximum(df['revenue'], 0)

    return df


# ══════════════════════════════════════════════════════════════
# MMM Engine
# ══════════════════════════════════════════════════════════════

def fit_adstock_params(spend: np.ndarray, target_residual: np.ndarray, max_lag: int = 8):
    """Find optimal adstock decay for a single channel via grid search."""
    best_corr = -1
    best_decay = 0.3

    for decay in np.arange(0.05, 0.95, 0.05):
        adstocked = geometric_adstock(spend, decay, max_lag)
        if adstocked.std() > 0:
            corr = abs(np.corrcoef(adstocked, target_residual)[0, 1])
            if corr > best_corr:
                best_corr = corr
                best_decay = decay

    return best_decay, best_corr


def fit_saturation_params(x: np.ndarray, y: np.ndarray):
    """Find optimal saturation lambda via grid search."""
    best_r2 = -np.inf
    best_lam = 0.0005

    for lam in np.logspace(-5, -2, 30):
        saturated = logistic_saturation(x, lam)
        if saturated.std() > 0:
            # Simple linear fit: y ≈ beta * saturated
            beta = np.dot(saturated, y) / (np.dot(saturated, saturated) + 1e-10)
            pred = beta * saturated
            ss_res = np.sum((y - pred) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2) + 1e-10
            r2 = 1 - ss_res / ss_tot
            if r2 > best_r2:
                best_r2 = r2
                best_lam = lam

    return best_lam, best_r2


def budget_optimization(
    channel_params: Dict[str, Dict],
    total_budget: float,
    current_spends: Dict[str, float],
) -> Dict[str, Any]:
    """Optimize budget allocation across channels to maximize total contribution."""
    channels = list(channel_params.keys())
    n_ch = len(channels)
    current_total = sum(current_spends.values())

    def neg_total_contribution(budget_fracs):
        total_contrib = 0
        for i, ch in enumerate(channels):
            spend = budget_fracs[i] * total_budget
            p = channel_params[ch]
            adstocked = spend * (1 / (1 - p['decay'] + 1e-10))  # steady-state adstock
            saturated = logistic_saturation(np.array([adstocked]), p['lambda'])[0]
            total_contrib += p['beta'] * saturated
        return -total_contrib

    # Constraints: fractions sum to 1
    constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0}
    bounds = [(0.01, 0.60) for _ in channels]
    x0 = np.ones(n_ch) / n_ch

    result = optimize.minimize(neg_total_contribution, x0, method='SLSQP',
                               bounds=bounds, constraints=constraints)

    optimized = {}
    for i, ch in enumerate(channels):
        opt_spend = result.x[i] * total_budget
        cur_spend = current_spends.get(ch, total_budget / n_ch)
        optimized[ch] = {
            'current_spend': safe_float(cur_spend),
            'current_pct': safe_float(cur_spend / current_total * 100) if current_total > 0 else 0,
            'optimal_spend': safe_float(opt_spend),
            'optimal_pct': safe_float(result.x[i] * 100),
            'change_pct': safe_float((opt_spend - cur_spend) / (cur_spend + 1e-10) * 100),
        }

    return {
        'optimized_allocation': optimized,
        'total_budget': safe_float(total_budget),
        'improvement_pct': safe_float((-result.fun - (-neg_total_contribution(
            np.array([current_spends.get(ch, total_budget / n_ch) / total_budget for ch in channels])
        ))) / abs(neg_total_contribution(
            np.array([current_spends.get(ch, total_budget / n_ch) / total_budget for ch in channels])
        ) + 1e-10) * 100),
    }


# ══════════════════════════════════════════════════════════════
# Main Endpoint
# ══════════════════════════════════════════════════════════════

@router.post("/mmm")
async def marketing_mix_model(request: MMMRequest):
    try:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error

        # ── 1. Data ──
        if request.generate or not request.data:
            df = generate_mmm_data(request.nWeeks, request.seed)
            col_date = 'date'
            col_target = 'revenue'
            spend_cols = [c for c in df.columns if c.startswith('spend_')]
            control_cols = ['holiday', 'competitor_promo']
            channel_names = [c.replace('spend_', '').title() for c in spend_cols]
        else:
            df = pd.DataFrame(request.data)
            col_date = request.colDate or next((c for c in df.columns if 'date' in c.lower() or 'week' in c.lower()), None)
            col_target = request.colTarget or next((c for c in df.columns if 'revenue' in c.lower() or 'sales' in c.lower() or 'target' in c.lower()), None)

            if not col_target:
                raise HTTPException(status_code=400, detail="Cannot find target (revenue/sales) column.")

            if request.colChannels:
                spend_cols = request.colChannels
            else:
                spend_cols = [c for c in df.columns if 'spend' in c.lower() or 'cost' in c.lower() or 'budget' in c.lower()]
                if not spend_cols:
                    spend_cols = [c for c in df.columns if c not in [col_date, col_target] and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]

            control_cols = request.colControls or [c for c in df.columns if c not in [col_date, col_target] + spend_cols and df[c].dtype in ['float64', 'int64', 'float32', 'int32']]
            channel_names = [c.replace('spend_', '').replace('_', ' ').title() for c in spend_cols]

        for c in spend_cols + [col_target] + control_cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=[col_target])

        n = len(df)
        if n < 20:
            raise HTTPException(status_code=400, detail=f"Need >=20 rows. Got {n}.")

        y = df[col_target].values
        n_channels = len(spend_cols)
        max_lag = request.adstockMaxLag

        # ── 2. Fit Adstock & Saturation per Channel ──
        channel_params = {}
        transformed_features = {}

        for i, (sc, ch_name) in enumerate(zip(spend_cols, channel_names)):
            spend = df[sc].values

            # Find optimal adstock decay
            decay, decay_corr = fit_adstock_params(spend, y, max_lag)

            # Apply adstock
            adstocked = geometric_adstock(spend, decay, max_lag)

            # Find optimal saturation
            sat_lam, sat_r2 = fit_saturation_params(adstocked, y)

            # Apply saturation
            saturated = logistic_saturation(adstocked, sat_lam)

            channel_params[ch_name] = {
                'spend_col': sc,
                'decay': safe_float(decay),
                'lambda': safe_float(sat_lam),
                'avg_spend': safe_float(spend.mean()),
                'total_spend': safe_float(spend.sum()),
            }
            transformed_features[ch_name] = saturated

        # ── 3. Build Feature Matrix & Fit Ridge ──
        X_cols = []
        X_data = []
        for ch_name in channel_names:
            X_data.append(transformed_features[ch_name])
            X_cols.append(ch_name)

        for cc in control_cols:
            if cc in df.columns:
                X_data.append(df[cc].values)
                X_cols.append(cc)

        X = np.column_stack(X_data)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = Ridge(alpha=1.0)
        model.fit(X_scaled, y)
        y_pred = model.predict(X_scaled)

        r2 = safe_float(r2_score(y, y_pred))
        mae = safe_float(mean_absolute_error(y, y_pred))
        mape = safe_float(mean_absolute_percentage_error(y, y_pred) * 100)

        # ── 4. Contribution Decomposition ──
        # ---------------------------------------------------------------
        # FIX: Convert scaled coefficients back to original feature scale.
        #
        # StandardScaler: X_scaled_j = (X_j - mean_j) / std_j
        # Ridge:  y = intercept_s + sum(coef_s_j * X_scaled_j)
        #
        # Substituting back:
        #   y = [intercept_s - sum(coef_s_j * mean_j / std_j)]
        #       + sum((coef_s_j / std_j) * X_j)
        #
        #   coef_original_j    = coef_s_j / std_j
        #   intercept_original = intercept_s - sum(coef_s_j * mean_j / std_j)
        # ---------------------------------------------------------------
        coefs_original = model.coef_ / scaler.scale_
        intercept_original = model.intercept_ - np.sum(model.coef_ * scaler.mean_ / scaler.scale_)

        # Decompose using ORIGINAL (unscaled) features
        contributions = {}
        total_media_contrib = 0
        for j, col_name in enumerate(X_cols):
            raw_contrib = coefs_original[j] * X[:, j]
            contributions[col_name] = raw_contrib

        # Aggregate
        base_contribution = float(intercept_original)
        channel_contributions = {}
        control_contributions = {}

        for col_name, contrib in contributions.items():
            total_c = float(contrib.sum())
            if col_name in channel_names:
                channel_contributions[col_name] = {
                    'total': safe_float(total_c),
                    'mean_weekly': safe_float(contrib.mean()),
                    'pct_of_revenue': safe_float(total_c / y.sum() * 100) if y.sum() != 0 else 0,
                    'coef': safe_float(coefs_original[X_cols.index(col_name)]),
                    'timeline': [safe_float(v) for v in contrib.tolist()],
                }
                total_media_contrib += total_c
            else:
                control_contributions[col_name] = {
                    'total': safe_float(total_c),
                    'pct_of_revenue': safe_float(total_c / y.sum() * 100) if y.sum() != 0 else 0,
                }

        # ROAS per channel
        for ch_name in channel_names:
            total_spend = channel_params[ch_name]['total_spend']
            total_contrib = channel_contributions[ch_name]['total']
            channel_contributions[ch_name]['roas'] = safe_float(total_contrib / total_spend) if total_spend > 0 else 0
            channel_contributions[ch_name]['mroas'] = safe_float(total_contrib / total_spend * 1.1) if total_spend > 0 else 0  # marginal approximation

            # Store params
            channel_contributions[ch_name]['decay'] = channel_params[ch_name]['decay']
            channel_contributions[ch_name]['saturation_lambda'] = channel_params[ch_name]['lambda']
            channel_contributions[ch_name]['avg_spend'] = channel_params[ch_name]['avg_spend']
            channel_contributions[ch_name]['total_spend'] = total_spend

        # ── 5. Charts ──

        # Waterfall
        waterfall = [{'component': 'Base', 'contribution': safe_float(base_contribution * n), 'pct': safe_float(base_contribution * n / y.sum() * 100)}]
        for ch in channel_names:
            waterfall.append({
                'component': ch,
                'contribution': channel_contributions[ch]['total'],
                'pct': channel_contributions[ch]['pct_of_revenue'],
            })
        for cc, cv in control_contributions.items():
            waterfall.append({'component': cc.replace('_', ' ').title(), 'contribution': cv['total'], 'pct': cv['pct_of_revenue']})

        # Actual vs Predicted
        fit_chart = []
        dates_list = df[col_date].tolist() if col_date and col_date in df.columns else [str(i) for i in range(n)]
        step = max(1, n // 200)
        for i in range(0, n, step):
            fit_chart.append({'date': str(dates_list[i]), 'actual': safe_float(y[i]), 'predicted': safe_float(y_pred[i])})

        # Adstock curves
        adstock_curves = []
        for ch_name in channel_names:
            decay = channel_params[ch_name]['decay']
            for lag in range(max_lag + 1):
                adstock_curves.append({'channel': ch_name, 'lag': lag, 'weight': safe_float(decay ** lag)})

        # Saturation curves
        saturation_curves = []
        for ch_name in channel_names:
            lam = channel_params[ch_name]['lambda']
            avg_spend = channel_params[ch_name]['avg_spend']
            max_spend = avg_spend * 3
            for pct in range(0, 101, 5):
                spend_val = max_spend * pct / 100
                sat_val = 1 - np.exp(-lam * spend_val)
                saturation_curves.append({
                    'channel': ch_name,
                    'spend': safe_float(spend_val),
                    'saturation': safe_float(sat_val),
                    'is_current': abs(spend_val - avg_spend) < avg_spend * 0.1,
                })

        # ROAS comparison
        roas_chart = []
        for ch in channel_names:
            cc = channel_contributions[ch]
            roas_chart.append({
                'channel': ch,
                'roas': cc['roas'],
                'total_spend': cc['total_spend'],
                'contribution': cc['total'],
                'pct': cc['pct_of_revenue'],
            })
        roas_chart.sort(key=lambda x: x['roas'], reverse=True)

        # Contribution timeline (stacked)
        contrib_timeline = []
        for i in range(0, n, step):
            entry = {'date': str(dates_list[i])}
            for ch in channel_names:
                entry[ch] = safe_float(channel_contributions[ch]['timeline'][i])
            contrib_timeline.append(entry)

        # ── 6. Budget Optimization ──
        optimization = None
        current_spends = {ch: channel_params[ch]['total_spend'] / n for ch in channel_names}
        total_budget = request.totalBudget or sum(current_spends.values())

        opt_params = {ch: {'decay': channel_params[ch]['decay'],
                           'lambda': channel_params[ch]['lambda'],
                           'beta': abs(channel_contributions[ch]['coef'])}
                      for ch in channel_names}
        optimization = budget_optimization(opt_params, total_budget, current_spends)

        opt_chart = []
        if optimization:
            for ch, alloc in optimization['optimized_allocation'].items():
                opt_chart.append({
                    'channel': ch,
                    'current_pct': alloc['current_pct'],
                    'optimal_pct': alloc['optimal_pct'],
                    'change_pct': alloc['change_pct'],
                })

        # ── Response ──
        results = {
            'n_weeks': n,
            'n_channels': n_channels,
            'channel_names': channel_names,
            'total_revenue': safe_float(y.sum()),
            'total_spend': safe_float(sum(channel_params[ch]['total_spend'] for ch in channel_names)),
            'model_fit': {'r2': r2, 'mae': mae, 'mape': mape},
            'base_pct': safe_float(base_contribution * n / y.sum() * 100),
            'media_pct': safe_float(total_media_contrib / y.sum() * 100),
            'channel_contributions': channel_contributions,
            'control_contributions': control_contributions,
            'optimization': optimization,
            'charts': {
                'waterfall': waterfall,
                'fit': fit_chart,
                'adstock_curves': adstock_curves,
                'saturation_curves': saturation_curves,
                'roas': roas_chart,
                'contribution_timeline': contrib_timeline,
                'budget_optimization': opt_chart,
            },
        }

        return _to_native({'results': results})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")
