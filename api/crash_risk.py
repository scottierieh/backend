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

class CrashRiskRequest(BaseModel):
    data: Optional[List[Dict[str, Any]]] = None
    generate: bool = False
    nFirms: int = 80
    nPeriods: int = 12       # quarterly panel
    seed: Optional[int] = None
    crashThreshold: float = -3.09   # z-score for crash week
    varConfidence: float = 0.05


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
        if val is None: return default
        f = float(val)
        return default if (np.isnan(f) or np.isinf(f)) else f
    except Exception:
        return default


# ══════════════════════════════════════════════════════════════
# Crash Risk Measures (Chen, Hong & Stein 2001)
# ══════════════════════════════════════════════════════════════

def compute_ncskew(returns: np.ndarray) -> float:
    """
    Negative Coefficient of Skewness.
    NCSKEW = -[n(n-1)^{3/2} Σr³] / [(n-1)(n-2)(Σr²)^{3/2}]
    Higher → more crash-prone (fatter left tail).
    """
    n = len(returns)
    if n < 5: return 0.0
    d = returns - returns.mean()
    s2 = np.sum(d ** 2)
    s3 = np.sum(d ** 3)
    if s2 == 0: return 0.0
    return float(-(n * (n - 1) ** 1.5 * s3) / ((n - 1) * (n - 2) * s2 ** 1.5))


def compute_duvol(returns: np.ndarray) -> float:
    """
    Down-to-Up Volatility ratio.
    DUVOL = log[(n_u-1)Σ(down²) / (n_d-1)Σ(up²)]
    Higher → more crash risk.
    """
    m = returns.mean()
    up = returns[returns >= m]
    down = returns[returns < m]
    if len(up) < 2 or len(down) < 2: return 0.0
    up_var = np.sum((up - m) ** 2) / (len(up) - 1)
    down_var = np.sum((down - m) ** 2) / (len(down) - 1)
    if up_var == 0: return 0.0
    return float(np.log(down_var / up_var))


def count_crash_weeks(returns: np.ndarray, threshold_z: float = -3.09) -> int:
    if len(returns) < 5: return 0
    z = (returns - returns.mean()) / (returns.std() + 1e-10)
    return int((z < threshold_z).sum())


# ══════════════════════════════════════════════════════════════
# Panel Data Generator
# ══════════════════════════════════════════════════════════════

SECTORS = ['Technology', 'Healthcare', 'Financial', 'Consumer', 'Energy', 'Industrial', 'Utilities', 'Real Estate']

GOVERNANCE_VARS = [
    'board_independence',   # % independent directors (0-1)
    'ceo_duality',          # CEO = chairman? (0/1)
    'insider_ownership',    # % insider shares (0-1)
    'inst_ownership',       # % institutional (0-1)
    'audit_quality',        # Big4 auditor? (0/1)
]

FINANCIAL_VARS = [
    'opacity',          # earnings management proxy
    'leverage',         # debt/assets
    'roa',              # return on assets
    'turnover',         # share turnover (detrended)
    'size',             # log market cap
    'mb_ratio',         # market-to-book
    'ret_volatility',   # prior return vol
    'avg_return',       # prior period return
]


def generate_panel_data(n_firms: int = 80, n_periods: int = 12, seed=None) -> pd.DataFrame:
    """
    Generate firm-quarter panel data with governance + financial features
    and simulated crash outcomes correlated with risk factors.
    """
    rng = np.random.default_rng(seed)
    rows = []

    consonants = 'BCDFGHJKLMNPQRSTVWXYZ'
    vowels = 'AEIOU'
    used_tickers = set()

    # Pre-generate firm-level fixed effects
    firm_fe = rng.normal(0, 0.3, size=n_firms)

    for i in range(n_firms):
        # Unique ticker
        while True:
            length = rng.choice([3, 4])
            ticker = ''.join(rng.choice(list(consonants)) if j % 2 == 0 else rng.choice(list(vowels)) for j in range(length))
            if ticker not in used_tickers:
                used_tickers.add(ticker)
                break

        sector = SECTORS[i % len(SECTORS)]

        # Firm-level baseline characteristics (persistent)
        base_board_indep = rng.uniform(0.4, 0.9)
        base_ceo_dual = rng.choice([0, 1], p=[0.65, 0.35])
        base_insider = rng.uniform(0.02, 0.25)
        base_inst = rng.uniform(0.3, 0.85)
        base_audit = rng.choice([0, 1], p=[0.3, 0.7])
        base_leverage = rng.uniform(0.1, 0.7)
        base_size = rng.uniform(6, 11)  # log mcap

        for t in range(n_periods):
            quarter = f"Q{(t % 4) + 1}-{2022 + t // 4}"

            # Governance (slowly varying)
            board_indep = np.clip(base_board_indep + rng.normal(0, 0.02), 0.2, 1.0)
            ceo_dual = base_ceo_dual
            insider_own = np.clip(base_insider + rng.normal(0, 0.01), 0.01, 0.5)
            inst_own = np.clip(base_inst + rng.normal(0, 0.03), 0.1, 0.95)
            audit = base_audit

            # Financial (more volatile)
            opacity = np.clip(rng.normal(0.05, 0.03) + (1 - board_indep) * 0.05, 0, 0.3)
            leverage = np.clip(base_leverage + rng.normal(0, 0.03), 0.05, 0.9)
            roa = rng.normal(0.03, 0.04) - leverage * 0.02
            turnover = rng.lognormal(-1, 0.5)
            size = base_size + rng.normal(0, 0.1)
            mb = rng.lognormal(0.5, 0.4)
            ret_vol = rng.uniform(0.15, 0.60)
            avg_ret = rng.normal(0.02, 0.05)

            # Weekly returns for this firm-quarter (13 weeks)
            weekly_sigma = ret_vol / np.sqrt(52)
            weekly_returns = rng.normal(avg_ret / 52, weekly_sigma, size=13)

            # Inject crash probability based on risk factors
            # Logistic model: P(crash) depends on governance weakness + financial risk
            crash_score = (
                firm_fe[i]
                - 1.5 * board_indep      # weak governance → crash
                + 0.8 * ceo_dual
                + 1.2 * opacity
                + 0.9 * leverage
                - 0.6 * roa * 10
                + 0.4 * ret_vol
                + 0.5 * insider_own * 3   # high insider → info asymmetry
                - 0.3 * inst_own
                - 0.4 * audit
                + rng.normal(0, 0.5)
            )
            crash_prob = 1 / (1 + np.exp(-crash_score))

            # Decide if crash occurs this quarter
            had_crash = rng.random() < crash_prob
            if had_crash:
                # Inject a crash week
                crash_week = rng.integers(2, 12)
                weekly_returns[crash_week] = -abs(rng.normal(0.08, 0.03))

            ncskew = compute_ncskew(weekly_returns)
            duvol = compute_duvol(weekly_returns)
            n_crash_weeks = count_crash_weeks(weekly_returns, -3.09)
            crash_binary = 1 if n_crash_weeks > 0 else 0

            rows.append({
                'ticker': ticker,
                'sector': sector,
                'quarter': quarter,
                'period': t,
                # Governance
                'board_independence': round(board_indep, 3),
                'ceo_duality': int(ceo_dual),
                'insider_ownership': round(insider_own, 4),
                'inst_ownership': round(inst_own, 3),
                'audit_quality': int(audit),
                # Financial
                'opacity': round(opacity, 4),
                'leverage': round(leverage, 3),
                'roa': round(roa, 4),
                'turnover': round(turnover, 4),
                'size': round(size, 3),
                'mb_ratio': round(mb, 3),
                'ret_volatility': round(ret_vol, 4),
                'avg_return': round(avg_ret, 4),
                # Crash measures
                'ncskew': round(ncskew, 4),
                'duvol': round(duvol, 4),
                'crash_weeks': n_crash_weeks,
                'crash': crash_binary,
            })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# Logistic Panel Regression
# ══════════════════════════════════════════════════════════════

def run_logistic_panel(
    df: pd.DataFrame,
    y_col: str,
    x_cols: List[str],
) -> Dict[str, Any]:
    """
    Pooled logistic regression with cluster-robust SE (firm-level).
    Approximates panel logistic — true FE logistic drops non-varying firms.
    """
    import statsmodels.api as sm
    from sklearn.preprocessing import StandardScaler

    y = df[y_col].values.astype(float)
    X_raw = df[x_cols].values.astype(float)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X = sm.add_constant(X_scaled)

    # Fit logistic
    model = sm.Logit(y, X)
    # Use cluster-robust SE if ticker available
    if 'ticker' in df.columns:
        result = model.fit(disp=0, cov_type='cluster', cov_kwds={'groups': df['ticker'].values})
    else:
        result = model.fit(disp=0)

    # Extract results
    coef_names = ['const'] + x_cols
    coefficients = []
    for i, name in enumerate(coef_names):
        coefficients.append({
            'variable': name,
            'coef': safe_float(result.params[i]),
            'std_err': safe_float(result.bse[i]),
            'z_stat': safe_float(result.tvalues[i]),
            'p_value': safe_float(result.pvalues[i]),
            'odds_ratio': safe_float(np.exp(result.params[i])),
            'significant': bool(result.pvalues[i] < 0.05),
        })

    # Predicted probabilities
    pred_probs = result.predict(X)

    # Marginal effects at mean
    marginal = result.get_margeff(at='mean')
    marg_effects = []
    for i, name in enumerate(x_cols):
        marg_effects.append({
            'variable': name,
            'dy_dx': safe_float(marginal.margeff[i]),
            'std_err': safe_float(marginal.margeff_se[i]),
            'p_value': safe_float(marginal.pvalues[i]),
        })

    return {
        'coefficients': coefficients,
        'marginal_effects': marg_effects,
        'pred_probs': pred_probs,
        'pseudo_r2': safe_float(result.prsquared),
        'log_likelihood': safe_float(result.llf),
        'aic': safe_float(result.aic),
        'bic': safe_float(result.bic),
        'n_obs': int(result.nobs),
    }


# ══════════════════════════════════════════════════════════════
# Main Endpoint
# ══════════════════════════════════════════════════════════════

@router.post("/crash-risk")
async def crash_risk(request: CrashRiskRequest):
    try:
        from sklearn.metrics import (
            roc_auc_score, roc_curve, precision_recall_curve,
            confusion_matrix, classification_report,
        )

        # ── 1. Data ──
        if request.generate or not request.data:
            df = generate_panel_data(request.nFirms, request.nPeriods, request.seed)
        else:
            df = pd.DataFrame(request.data)

        n = len(df)
        has_crash = 'crash' in df.columns

        if not has_crash:
            raise HTTPException(status_code=400, detail="Need 'crash' column (0/1 binary).")

        # Identify available feature columns
        gov_cols = [c for c in GOVERNANCE_VARS if c in df.columns]
        fin_cols = [c for c in FINANCIAL_VARS if c in df.columns]
        all_x = gov_cols + fin_cols

        if len(all_x) < 3:
            raise HTTPException(status_code=400, detail=f"Need >=3 feature columns, found {len(all_x)}")

        for c in all_x + ['crash']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        df = df.dropna(subset=all_x + ['crash'])
        n = len(df)

        crash_rate = df['crash'].mean()

        # ── 2. Logistic Panel ──
        logit_results = run_logistic_panel(df, 'crash', all_x)
        pred_probs = logit_results['pred_probs']

        # ── 3. Classification Metrics ──
        y_true = df['crash'].values
        threshold = 0.5
        y_pred = (pred_probs >= threshold).astype(int)

        auc = safe_float(roc_auc_score(y_true, pred_probs))
        cm = confusion_matrix(y_true, y_pred).tolist()

        fpr, tpr, _ = roc_curve(y_true, pred_probs)
        step = max(1, len(fpr) // 100)
        roc_data = [{'fpr': safe_float(fpr[i]), 'tpr': safe_float(tpr[i])} for i in range(0, len(fpr), step)]

        prec, rec, _ = precision_recall_curve(y_true, pred_probs)
        step2 = max(1, len(prec) // 100)
        pr_data = [{'precision': safe_float(prec[i]), 'recall': safe_float(rec[i])} for i in range(0, len(prec), step2)]

        # ── 4. Firm-Level Risk Scores ──
        df['_crash_prob'] = pred_probs

        # Aggregate to firm level (latest or average)
        firm_risk = []
        tickers = df['ticker'].unique() if 'ticker' in df.columns else df.index.unique()

        for ticker in tickers:
            mask = df['ticker'] == ticker if 'ticker' in df.columns else df.index == ticker
            subset = df[mask]
            latest = subset.iloc[-1]
            avg_prob = subset['_crash_prob'].mean()
            max_prob = subset['_crash_prob'].max()
            n_crashes = int(subset['crash'].sum())

            entry = {
                'ticker': str(ticker),
                'sector': str(latest.get('sector', '')),
                'avg_crash_prob': safe_float(avg_prob),
                'max_crash_prob': safe_float(max_prob),
                'latest_crash_prob': safe_float(latest['_crash_prob']),
                'total_crashes': n_crashes,
                'n_periods': int(len(subset)),
            }

            if 'ncskew' in df.columns:
                entry['latest_ncskew'] = safe_float(latest.get('ncskew', 0))
            if 'duvol' in df.columns:
                entry['latest_duvol'] = safe_float(latest.get('duvol', 0))
            for c in gov_cols[:3]:
                entry[c] = safe_float(latest.get(c, 0))

            # Risk tier
            if max_prob > 0.7:
                entry['risk_tier'] = 'Critical'
            elif max_prob > 0.5:
                entry['risk_tier'] = 'High'
            elif max_prob > 0.3:
                entry['risk_tier'] = 'Elevated'
            else:
                entry['risk_tier'] = 'Low'

            firm_risk.append(entry)

        firm_risk.sort(key=lambda x: x['max_crash_prob'], reverse=True)

        # Risk tier distribution
        tier_counts = {}
        for fr in firm_risk:
            t = fr['risk_tier']
            tier_counts[t] = tier_counts.get(t, 0) + 1

        # ── 5. Chart Data ──

        # Coefficient plot
        coef_chart = []
        for c in logit_results['coefficients']:
            if c['variable'] == 'const':
                continue
            coef_chart.append({
                'variable': c['variable'],
                'odds_ratio': c['odds_ratio'],
                'coef': c['coef'],
                'significant': c['significant'],
                'ci_low': safe_float(c['coef'] - 1.96 * c['std_err']),
                'ci_high': safe_float(c['coef'] + 1.96 * c['std_err']),
            })

        # Marginal effects
        marg_chart = logit_results['marginal_effects']

        # Crash probability distribution
        prob_bins = np.linspace(0, 1, 25)
        prob_hist = []
        for j in range(len(prob_bins) - 1):
            lo, hi = prob_bins[j], prob_bins[j + 1]
            mask = (pred_probs >= lo) & (pred_probs < hi)
            prob_hist.append({
                'range': f'{(lo + hi) / 2:.2f}',
                'count': int(mask.sum()),
                'crash': int((mask & (y_true == 1)).sum()),
                'no_crash': int((mask & (y_true == 0)).sum()),
            })

        # Crash rate by sector
        sector_chart = []
        if 'sector' in df.columns:
            for sector in df['sector'].unique():
                mask = df['sector'] == sector
                sector_chart.append({
                    'sector': str(sector),
                    'crash_rate': safe_float(df.loc[mask, 'crash'].mean() * 100),
                    'avg_prob': safe_float(pred_probs[mask].mean() * 100),
                    'count': int(mask.sum()),
                })
            sector_chart.sort(key=lambda x: x['crash_rate'], reverse=True)

        # Crash prob over time
        time_chart = []
        if 'period' in df.columns:
            for p in sorted(df['period'].unique()):
                mask = df['period'] == p
                time_chart.append({
                    'period': int(p),
                    'quarter': str(df.loc[mask, 'quarter'].iloc[0]) if 'quarter' in df.columns else str(p),
                    'crash_rate': safe_float(df.loc[mask, 'crash'].mean() * 100),
                    'avg_prob': safe_float(pred_probs[mask].mean() * 100),
                    'n_firms': int(mask.sum()),
                })

        # NCSKEW vs DUVOL scatter (if available)
        skew_scatter = []
        if 'ncskew' in df.columns and 'duvol' in df.columns:
            step_s = max(1, n // 400)
            for i in range(0, n, step_s):
                row = df.iloc[i]
                skew_scatter.append({
                    'ncskew': safe_float(row['ncskew']),
                    'duvol': safe_float(row['duvol']),
                    'crash': int(row['crash']),
                    'ticker': str(row.get('ticker', '')),
                })

        # ── 6. Summary ──
        n_firms = df['ticker'].nunique() if 'ticker' in df.columns else n
        n_critical = tier_counts.get('Critical', 0)
        n_high = tier_counts.get('High', 0)

        results = {
            'summary': {
                'n_observations': n,
                'n_firms': n_firms,
                'n_periods': int(df['period'].nunique()) if 'period' in df.columns else 1,
                'crash_rate_pct': safe_float(crash_rate * 100),
                'auc': auc,
                'pseudo_r2': logit_results['pseudo_r2'],
                'aic': logit_results['aic'],
                'bic': logit_results['bic'],
                'n_features': len(all_x),
                'gov_features': gov_cols,
                'fin_features': fin_cols,
            },
            'risk_tiers': tier_counts,
            'n_critical': n_critical,
            'n_high': n_high,
            'confusion_matrix': cm,
            'firm_risk': firm_risk[:30],  # top 30
            'charts': {
                'coefficients': coef_chart,
                'marginal_effects': marg_chart,
                'roc': roc_data,
                'pr_curve': pr_data,
                'prob_distribution': prob_hist,
                'sector_crash': sector_chart,
                'time_trend': time_chart,
                'ncskew_duvol': skew_scatter,
            },
        }

        return _to_native({'results': results})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")
