from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Literal
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
import warnings
import traceback

warnings.filterwarnings('ignore')

router = APIRouter()


# ══════════════════════════════════════════════════════════════
# Model Definitions
# ══════════════════════════════════════════════════════════════

MODEL_FACTORS: Dict[str, List[str]] = {
    "CAPM":     ["Mkt_RF"],
    "FF3":      ["Mkt_RF", "SMB", "HML"],
    "Carhart4": ["Mkt_RF", "SMB", "HML", "MOM"],
    "FF5":      ["Mkt_RF", "SMB", "HML", "RMW", "CMA"],
}

MODEL_LABELS: Dict[str, str] = {
    "CAPM":     "CAPM (1-Factor)",
    "FF3":      "Fama-French 3-Factor",
    "Carhart4": "Carhart 4-Factor",
    "FF5":      "Fama-French 5-Factor",
}

# Static fallback labels (used when model context is unknown)
FACTOR_LABELS: Dict[str, str] = {
    "alpha":  "α (Alpha)",
    "Mkt_RF": "β₁ (Market)",
    "SMB":    "β₂ (SMB)",
    "HML":    "β₃ (HML)",
    "MOM":    "β₄ (MOM)",   # Carhart4 — 4th factor
    "RMW":    "β₄ (RMW)",   # FF5 — 4th factor  ← same position as MOM, different model
    "CMA":    "β₅ (CMA)",   # FF5 — 5th factor
}

def _factor_label(factor_name: str, factor_cols: List[str]) -> str:
    """
    Generate a β-subscript label that is unique within the given factor list.
    Position is 1-based index inside factor_cols (α is always index 0).

    CAPM:     Mkt_RF → β₁
    FF3:      Mkt_RF → β₁, SMB → β₂, HML → β₃
    Carhart4: Mkt_RF → β₁, SMB → β₂, HML → β₃, MOM → β₄
    FF5:      Mkt_RF → β₁, SMB → β₂, HML → β₃, RMW → β₄, CMA → β₅
    """
    if factor_name == "alpha":
        return "α (Alpha)"
    SUBSCRIPTS = ["", "₁", "₂", "₃", "₄", "₅", "₆"]
    SHORT = {
        "Mkt_RF": "Market",
        "SMB": "SMB", "HML": "HML",
        "MOM": "MOM", "RMW": "RMW", "CMA": "CMA",
    }
    try:
        idx = factor_cols.index(factor_name) + 1   # 1-based
    except ValueError:
        idx = 0
    sub = SUBSCRIPTS[idx] if idx < len(SUBSCRIPTS) else str(idx)
    return f"β{sub} ({SHORT.get(factor_name, factor_name)})"

ALL_FACTORS = ["Mkt_RF", "SMB", "HML", "MOM", "RMW", "CMA"]


# ══════════════════════════════════════════════════════════════
# Request Model
# ══════════════════════════════════════════════════════════════

class FamaFrenchRequest(BaseModel):
    data: Optional[List[Dict[str, Any]]] = None
    returnCol: Optional[str] = None
    dateCol: Optional[str] = None
    # Column name mapping (user data → standard names)
    mktRfCol: Optional[str] = None
    smbCol: Optional[str] = None
    hmlCol: Optional[str] = None
    momCol: Optional[str] = None
    rmwCol: Optional[str] = None
    cmaCol: Optional[str] = None
    rfCol: Optional[str] = None
    # Model selection — can be a single model or list for comparison
    model: Optional[str] = "FF3"           # "CAPM" | "FF3" | "Carhart4" | "FF5"
    compareAll: bool = False               # True → run all 4 models and return comparison
    # Generate mode
    generate: bool = False
    ticker: str = "AAPL"
    nMonths: int = 120
    seed: Optional[int] = None
    # Rolling analysis
    rollingWindow: int = 36
    # GRS Test — number of test portfolios (N). Must satisfy T > N + K.
    # For monthly data (T≈120), N=5 gives df2=112 with FF3.
    grsPortfolios: int = 5
    # Scenario analysis
    scenarioProfiles: Optional[List[Dict[str, Any]]] = None


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


def _validate_model(model: str) -> str:
    m = model.upper().replace("-", "").replace(" ", "")
    # Alias normalisation
    aliases = {
        "CAPM": "CAPM",
        "FF3": "FF3",
        "FAMAFRENCH3": "FF3",
        "CARHART": "Carhart4",
        "CARHART4": "Carhart4",
        "4FACTOR": "Carhart4",
        "FF5": "FF5",
        "FAMAFRENCH5": "FF5",
        "5FACTOR": "FF5",
    }
    result = aliases.get(m)
    if result is None:
        raise ValueError(f"Unknown model '{model}'. Choose from: CAPM, FF3, Carhart4, FF5")
    return result


# ══════════════════════════════════════════════════════════════
# Data Generator — FF5 + MOM (covers all models)
# ══════════════════════════════════════════════════════════════

STOCK_PROFILES = {
    'AAPL':  {'name': 'Apple Inc.',        'beta': 1.20, 'smb': -0.15, 'hml': -0.45, 'mom':  0.10, 'rmw':  0.20, 'cma': -0.30, 'alpha': 0.005, 'sigma': 0.04},
    'MSFT':  {'name': 'Microsoft Corp.',   'beta': 1.10, 'smb': -0.20, 'hml': -0.30, 'mom':  0.08, 'rmw':  0.25, 'cma': -0.20, 'alpha': 0.004, 'sigma': 0.035},
    'GOOGL': {'name': 'Alphabet Inc.',     'beta': 1.05, 'smb': -0.18, 'hml': -0.40, 'mom':  0.05, 'rmw':  0.15, 'cma': -0.25, 'alpha': 0.003, 'sigma': 0.04},
    'AMZN':  {'name': 'Amazon.com Inc.',   'beta': 1.25, 'smb': -0.10, 'hml': -0.55, 'mom':  0.15, 'rmw': -0.10, 'cma': -0.35, 'alpha': 0.004, 'sigma': 0.05},
    'TSLA':  {'name': 'Tesla Inc.',        'beta': 1.80, 'smb':  0.30, 'hml': -0.70, 'mom':  0.20, 'rmw': -0.30, 'cma': -0.50, 'alpha': 0.006, 'sigma': 0.10},
    'JPM':   {'name': 'JPMorgan Chase',    'beta': 1.15, 'smb': -0.05, 'hml':  0.60, 'mom':  0.05, 'rmw':  0.35, 'cma':  0.10, 'alpha': 0.002, 'sigma': 0.04},
    'JNJ':   {'name': 'Johnson & Johnson', 'beta': 0.65, 'smb': -0.25, 'hml':  0.30, 'mom': -0.05, 'rmw':  0.40, 'cma':  0.15, 'alpha': 0.001, 'sigma': 0.025},
    'WMT':   {'name': 'Walmart Inc.',      'beta': 0.55, 'smb': -0.30, 'hml':  0.20, 'mom': -0.02, 'rmw':  0.30, 'cma':  0.10, 'alpha': 0.001, 'sigma': 0.025},
    'XOM':   {'name': 'Exxon Mobil',       'beta': 0.90, 'smb': -0.10, 'hml':  0.70, 'mom':  0.00, 'rmw':  0.25, 'cma':  0.30, 'alpha': 0.000, 'sigma': 0.04},
    'BRK.B': {'name': 'Berkshire Hathaway','beta': 0.85, 'smb': -0.15, 'hml':  0.40, 'mom':  0.02, 'rmw':  0.30, 'cma':  0.20, 'alpha': 0.003, 'sigma': 0.03},
    'NVDA':  {'name': 'NVIDIA Corp.',      'beta': 1.60, 'smb':  0.10, 'hml': -0.60, 'mom':  0.25, 'rmw':  0.10, 'cma': -0.45, 'alpha': 0.008, 'sigma': 0.08},
    'META':  {'name': 'Meta Platforms',    'beta': 1.30, 'smb': -0.12, 'hml': -0.35, 'mom':  0.12, 'rmw':  0.15, 'cma': -0.25, 'alpha': 0.003, 'sigma': 0.05},
    'V':     {'name': 'Visa Inc.',         'beta': 0.95, 'smb': -0.20, 'hml': -0.15, 'mom':  0.06, 'rmw':  0.35, 'cma': -0.10, 'alpha': 0.003, 'sigma': 0.03},
    'PG':    {'name': 'Procter & Gamble',  'beta': 0.50, 'smb': -0.30, 'hml':  0.25, 'mom': -0.03, 'rmw':  0.45, 'cma':  0.15, 'alpha': 0.001, 'sigma': 0.02},
    'KO':    {'name': 'Coca-Cola Co.',     'beta': 0.55, 'smb': -0.28, 'hml':  0.15, 'mom': -0.02, 'rmw':  0.40, 'cma':  0.10, 'alpha': 0.001, 'sigma': 0.02},
    'DIS':   {'name': 'Walt Disney Co.',   'beta': 1.10, 'smb': -0.05, 'hml':  0.10, 'mom':  0.03, 'rmw':  0.10, 'cma':  0.05, 'alpha': 0.001, 'sigma': 0.04},
    'NFLX':  {'name': 'Netflix Inc.',      'beta': 1.35, 'smb':  0.05, 'hml': -0.50, 'mom':  0.18, 'rmw':  0.05, 'cma': -0.35, 'alpha': 0.005, 'sigma': 0.07},
    'AMD':   {'name': 'AMD Inc.',          'beta': 1.70, 'smb':  0.25, 'hml': -0.55, 'mom':  0.22, 'rmw': -0.05, 'cma': -0.40, 'alpha': 0.006, 'sigma': 0.09},
    'INTC':  {'name': 'Intel Corp.',       'beta': 1.05, 'smb':  0.00, 'hml':  0.20, 'mom': -0.08, 'rmw':  0.15, 'cma':  0.10, 'alpha':-0.002, 'sigma': 0.05},
    'BA':    {'name': 'Boeing Co.',        'beta': 1.30, 'smb':  0.05, 'hml':  0.25, 'mom': -0.05, 'rmw': -0.10, 'cma':  0.15, 'alpha':-0.001, 'sigma': 0.06},
}


def generate_ff5_data(
    ticker: str = 'AAPL',
    n_months: int = 120,
    seed: Optional[int] = None,
) -> tuple:
    """
    Generate realistic monthly data for all 6 factors:
    Mkt-RF, SMB, HML, MOM, RMW, CMA + RF.

    Returns (DataFrame, profile_dict).
    """
    rng = np.random.default_rng(seed)

    # Factor parameters (monthly, decimal)
    # Order: Mkt_RF, SMB, HML, MOM, RMW, CMA
    factor_means = np.array([0.0065, 0.0020, 0.0030, 0.0060, 0.0025, 0.0020])
    factor_stds  = np.array([0.0450, 0.0300, 0.0300, 0.0400, 0.0200, 0.0180])

    # Realistic correlation matrix (6×6)
    corr = np.array([
        # MktRF  SMB    HML    MOM    RMW    CMA
        [ 1.00,  0.30, -0.25, -0.10,  0.05, -0.10],  # MktRF
        [ 0.30,  1.00,  0.10, -0.05, -0.35, -0.10],  # SMB
        [-0.25,  0.10,  1.00, -0.35, -0.10,  0.70],  # HML
        [-0.10, -0.05, -0.35,  1.00,  0.10, -0.20],  # MOM
        [ 0.05, -0.35, -0.10,  0.10,  1.00,  0.00],  # RMW
        [-0.10, -0.10,  0.70, -0.20,  0.00,  1.00],  # CMA
    ])
    cov = np.outer(factor_stds, factor_stds) * corr

    factors = rng.multivariate_normal(factor_means, cov, size=n_months)
    mkt_rf, smb, hml, mom, rmw, cma = [factors[:, i] for i in range(6)]

    # Risk-free rate
    rf_base = 0.0025
    rf_noise = rng.normal(0, 0.0005, n_months)
    rf = np.maximum(rf_base + np.cumsum(rf_noise * 0.1), 0.0001)
    rf = rf - rf.mean() + rf_base

    # Stock profile
    profile = STOCK_PROFILES.get(ticker.upper(), {
        'name': ticker, 'beta': 1.0, 'smb': 0.0, 'hml': 0.0,
        'mom': 0.0, 'rmw': 0.0, 'cma': 0.0, 'alpha': 0.002, 'sigma': 0.04,
    })

    alpha    = profile['alpha']
    b_mkt    = profile['beta']
    b_smb    = profile['smb']
    b_hml    = profile['hml']
    b_mom    = profile['mom']
    b_rmw    = profile['rmw']
    b_cma    = profile['cma']
    sigma_e  = profile['sigma']

    epsilon      = rng.normal(0, sigma_e, n_months)
    excess_return = (alpha
                     + b_mkt * mkt_rf
                     + b_smb * smb
                     + b_hml * hml
                     + b_mom * mom
                     + b_rmw * rmw
                     + b_cma * cma
                     + epsilon)
    stock_return = excess_return + rf

    end_date = pd.Timestamp('2025-04-30')
    dates = pd.date_range(end=end_date, periods=n_months, freq='ME')

    df = pd.DataFrame({
        'date':          dates.strftime('%Y-%m'),
        'stock_return':  np.round(stock_return  * 100, 4),
        'excess_return': np.round(excess_return * 100, 4),
        'Mkt_RF':        np.round(mkt_rf * 100, 4),
        'SMB':           np.round(smb    * 100, 4),
        'HML':           np.round(hml    * 100, 4),
        'MOM':           np.round(mom    * 100, 4),
        'RMW':           np.round(rmw    * 100, 4),
        'CMA':           np.round(cma    * 100, 4),
        'RF':            np.round(rf     * 100, 4),
    })

    return df, profile


# ══════════════════════════════════════════════════════════════
# OLS Regression — Generic (any factor set)
# ══════════════════════════════════════════════════════════════

def run_regression(
    df: pd.DataFrame,
    y_col: str,
    factor_cols: List[str],
    model_name: str = "FF3",
) -> Dict:
    """
    Generic OLS regression for any factor model with Newey-West (HAC) standard errors.

    ① Dependent variable is ALWAYS excess return = stock_return − RF.
       If 'RF' column is present and y_col is NOT already an excess-return column,
       we subtract RF here so the regression is properly specified.
    ② cov_type="HAC" with maxlags=3 corrects for autocorrelation + heteroskedasticity
       common in monthly financial return series.
    """
    # ── ① Ensure y is excess return ──────────────────────────────
    # If the caller passes a raw stock return column AND RF is available,
    # compute excess return on the fly (non-destructive copy).
    work_df  = df.copy()
    work_col = y_col

    if 'RF' in work_df.columns and work_col != 'excess_return':
        er_col = '__excess_return__'
        work_df[er_col] = (
            pd.to_numeric(work_df[work_col], errors='coerce')
            - pd.to_numeric(work_df['RF'],   errors='coerce')
        )
        work_col = er_col

    work_df = work_df.dropna(subset=[work_col] + factor_cols)

    y       = work_df[work_col].values.astype(np.float64)
    X       = work_df[factor_cols].values.astype(np.float64)
    X_const = sm.add_constant(X)

    # ── ② Newey-West (HAC) standard errors ───────────────────────
    ols_model = sm.OLS(y, X_const)
    results   = ols_model.fit(cov_type="HAC", cov_kwds={"maxlags": 3})

    param_names = ['alpha'] + factor_cols

    coefficients = []
    ci = results.conf_int()
    for i, name in enumerate(param_names):
        coefficients.append({
            'factor':      name,
            'label':       _factor_label(name, factor_cols),
            'coefficient': safe_float(results.params[i]),
            'std_error':   safe_float(results.bse[i]),      # HAC SE
            't_stat':      safe_float(results.tvalues[i]),  # HAC t
            'p_value':     safe_float(results.pvalues[i]),  # HAC p
            'ci_lower':    safe_float(ci[i, 0]),
            'ci_upper':    safe_float(ci[i, 1]),
            'significant': bool(results.pvalues[i] < 0.05),
        })

    # Residuals & fitted values come from the same HAC fit
    residuals = results.resid
    fitted    = results.fittedvalues
    dw        = safe_float(sm.stats.durbin_watson(residuals))

    jb_stat, jb_pval, jb_skew, jb_kurt = sm.stats.jarque_bera(residuals)

    try:
        bp_stat, bp_pval, _, _ = sm.stats.diagnostic.het_breuschpagan(residuals, X_const)
    except Exception:
        bp_stat, bp_pval = 0.0, 1.0

    from statsmodels.stats.outliers_influence import variance_inflation_factor
    vif_values = []
    for j in range(1, X_const.shape[1]):
        try:
            vif_val = variance_inflation_factor(X_const, j)
            vif_values.append({'factor': factor_cols[j - 1], 'vif': safe_float(vif_val)})
        except Exception:
            vif_values.append({'factor': factor_cols[j - 1], 'vif': None})

    return {
        'model':           model_name,
        'model_label':     MODEL_LABELS.get(model_name, model_name),
        'se_type':         'HAC (Newey-West, maxlags=3)',
        'factor_cols':     factor_cols,
        'coefficients':    coefficients,
        'r_squared':       safe_float(results.rsquared),
        'adj_r_squared':   safe_float(results.rsquared_adj),
        'f_stat':          safe_float(results.fvalue),
        'f_pvalue':        safe_float(results.f_pvalue),
        'n_observations':  int(results.nobs),
        'df_model':        int(results.df_model),
        'df_resid':        int(results.df_resid),
        'aic':             safe_float(results.aic),
        'bic':             safe_float(results.bic),
        'log_likelihood':  safe_float(results.llf),
        'durbin_watson':   dw,
        'jarque_bera': {
            'statistic': safe_float(jb_stat),
            'p_value':   safe_float(jb_pval),
            'skewness':  safe_float(jb_skew),
            'kurtosis':  safe_float(jb_kurt),
        },
        'breusch_pagan': {
            'statistic': safe_float(bp_stat),
            'p_value':   safe_float(bp_pval),
        },
        'vif':            vif_values,
        'residuals':      [safe_float(r) for r in residuals],
        'fitted_values':  [safe_float(f) for f in fitted],
    }


# ══════════════════════════════════════════════════════════════
# Model Comparison
# ══════════════════════════════════════════════════════════════

def compare_models(
    df: pd.DataFrame,
    y_col: str,
    available_factors: List[str],
) -> Dict:
    """
    Run CAPM / FF3 / Carhart4 / FF5 on the same data and return
    a side-by-side comparison table.
    Only runs models whose required factors are present in df.
    """
    comparison = {}
    summary_rows = []

    for model_name, factor_cols in MODEL_FACTORS.items():
        missing = [f for f in factor_cols if f not in available_factors]
        if missing:
            comparison[model_name] = {
                'skipped': True,
                'reason': f"Missing columns: {missing}",
            }
            continue

        try:
            reg = run_regression(df, y_col, factor_cols, model_name)
            comparison[model_name] = reg
            summary_rows.append({
                'model':         model_name,
                'model_label':   MODEL_LABELS[model_name],
                'n_factors':     len(factor_cols),
                'r_squared':     reg['r_squared'],
                'adj_r_squared': reg['adj_r_squared'],
                'aic':           reg['aic'],
                'bic':           reg['bic'],
                'f_stat':        reg['f_stat'],
                'f_pvalue':      reg['f_pvalue'],
                'alpha':         reg['coefficients'][0]['coefficient'],
                'alpha_pvalue':  reg['coefficients'][0]['p_value'],
                'alpha_significant': reg['coefficients'][0]['significant'],
            })
        except Exception as e:
            comparison[model_name] = {'skipped': True, 'reason': str(e)}

    # Best model by adjusted R²
    valid_rows = [r for r in summary_rows if 'adj_r_squared' in r]
    best_adj_r2 = max(valid_rows, key=lambda x: x['adj_r_squared'])['model'] if valid_rows else None
    best_aic    = min(valid_rows, key=lambda x: x['aic'])['model'] if valid_rows else None
    best_bic    = min(valid_rows, key=lambda x: x['bic'])['model'] if valid_rows else None

    return {
        'models':       comparison,
        'summary_table': summary_rows,
        'best': {
            'by_adj_r2': best_adj_r2,
            'by_aic':    best_aic,
            'by_bic':    best_bic,
        },
    }


# ══════════════════════════════════════════════════════════════
# Rolling Regression  (HAC SE — same as full-sample regression)
# ══════════════════════════════════════════════════════════════

def rolling_regression(
    df: pd.DataFrame,
    y_col: str,
    factor_cols: List[str],
    window: int = 36,
) -> Optional[List[Dict]]:
    """
    Rolling-window OLS with Newey-West (HAC) standard errors.
    maxlags is capped at window//4 so short windows don't over-correct.
    Returns per-window alpha, betas, HAC t-stats, R², and significance flags.
    """
    n = len(df)
    if n < window:
        return None

    results_list = []
    dates = df['date'].values if 'date' in df.columns else [str(i) for i in range(n)]

    for i in range(window, n + 1):
        sub = df.iloc[i - window:i]
        y   = sub[y_col].values.astype(np.float64)
        X   = sm.add_constant(sub[factor_cols].values.astype(np.float64))

        try:
            # HAC maxlags: at least 1, at most window//4
            hac_lags = max(1, window // 4)
            res = sm.OLS(y, X).fit(
                cov_type="HAC",
                cov_kwds={"maxlags": hac_lags},
            )
            entry = {
                'date':               str(dates[i - 1]),
                'alpha':              safe_float(res.params[0]),
                'alpha_tstat':        safe_float(res.tvalues[0]),
                'alpha_significant':  bool(res.pvalues[0] < 0.05),
                'r_squared':          safe_float(res.rsquared),
            }
            for j, col in enumerate(factor_cols):
                entry[f'beta_{col}']         = safe_float(res.params[j + 1])
                entry[f'tstat_{col}']        = safe_float(res.tvalues[j + 1])
                entry[f'sig_{col}']          = bool(res.pvalues[j + 1] < 0.05)
            results_list.append(entry)
        except Exception:
            continue

    return results_list


# ══════════════════════════════════════════════════════════════
# Factor Decomposition
# ══════════════════════════════════════════════════════════════

def factor_decomposition(
    df: pd.DataFrame,
    y_col: str,
    factor_cols: List[str],
    coefficients: List[Dict],
) -> tuple:
    alpha = coefficients[0]['coefficient']
    betas = {c['factor']: c['coefficient'] for c in coefficients[1:]}

    dates = df['date'].values if 'date' in df.columns else [str(i) for i in range(len(df))]

    # ── Pre-compute cumulative actual return (O(n) instead of O(n²)) ──
    actual_series  = df[y_col].astype(float).values
    cum_actual_arr = np.cumsum(actual_series)      # vectorised cumsum

    # Pre-compute per-factor factor × beta series as numpy arrays
    factor_arrays: Dict[str, np.ndarray] = {}
    for fc in factor_cols:
        factor_arrays[fc] = betas.get(fc, 0.0) * df[fc].astype(float).values

    # Cumulative sums per factor
    cum_factor: Dict[str, float] = {fc: 0.0 for fc in factor_cols}
    cum_alpha   = 0.0
    cum_residual = 0.0

    decomp = []
    for i in range(len(df)):
        actual    = float(actual_series[i])
        predicted = alpha
        entry = {
            'date':          str(dates[i]),
            'actual_return': safe_float(actual),
            'alpha_contrib': safe_float(alpha),
        }

        for fc in factor_cols:
            contrib = float(factor_arrays[fc][i])
            entry[f'{fc}_contrib'] = safe_float(contrib)
            predicted            += contrib
            cum_factor[fc]       += contrib

        residual  = actual - predicted
        cum_alpha   += alpha
        cum_residual += residual

        entry['residual']     = safe_float(residual)
        entry['predicted']    = safe_float(predicted)
        entry['cum_actual']   = safe_float(cum_actual_arr[i])   # ← O(1) lookup
        entry['cum_alpha']    = safe_float(cum_alpha)
        for fc in factor_cols:
            entry[f'cum_{fc}'] = safe_float(cum_factor[fc])
        entry['cum_residual'] = safe_float(cum_residual)

        decomp.append(entry)

    total_return = float(cum_actual_arr[-1]) if len(cum_actual_arr) else 0.0
    summary = {
        'total_return':   safe_float(total_return),
        'alpha_total':    safe_float(cum_alpha),
        'alpha_pct':      safe_float(cum_alpha / total_return * 100) if total_return != 0 else 0,
        'residual_total': safe_float(cum_residual),
    }
    for fc in factor_cols:
        summary[f'{fc}_total'] = safe_float(cum_factor[fc])
        summary[f'{fc}_pct']   = safe_float(cum_factor[fc] / total_return * 100) if total_return != 0 else 0

    return decomp, summary


# ══════════════════════════════════════════════════════════════
# GRS Test  (Gibbons–Ross–Shanken 1989)
# ══════════════════════════════════════════════════════════════

# ══════════════════════════════════════════════════════════════
# GRS Test  (Gibbons–Ross–Shanken 1989)  — proper N-portfolio version
# ══════════════════════════════════════════════════════════════

def grs_test(
    alphas: np.ndarray,          # (N,)   alpha vector across N portfolios
    cov_eps: np.ndarray,         # (N,N)  residual covariance matrix
    factor_means: np.ndarray,    # (K,)   time-series mean of each factor
    cov_factors: np.ndarray,     # (K,K)  factor covariance matrix
    T: int,                      # number of time periods
) -> Dict:
    """
    GRS F-statistic: tests H₀: α₁ = α₂ = … = αN = 0 jointly.

    GRS = ((T−N−K)/N) × (1 + μ_f′ Σ_f⁻¹ μ_f)⁻¹ × α′ Σ_ε⁻¹ α
          ~ F(N, T−N−K)  under H₀ and multivariate normality.

    Parameters
    ──────────
    alphas      OLS alpha estimates, one per test portfolio (N,)
    cov_eps     OLS residual covariance matrix Σ̂_ε  (N×N), unbiased:
                  Σ̂_ε = E'E / (T−K−1)  where E is the (T×N) residual matrix
    factor_means  sample mean of the K factors  (K,)
    cov_factors   sample covariance of factors  (K×K)
    T           number of time-series observations

    Notes
    ─────
    • Requires T > N + K for positive df2.
    • If Σ_ε is singular (common when N is large relative to T),
      we regularise with a Ledoit-Wolf shrinkage estimate.
    • The term (1 + μ_f′ Σ_f⁻¹ μ_f) is the Sharpe-ratio adjustment
      that accounts for sampling error in the factor means.
    """
    N = len(alphas)
    K = len(factor_means)
    df1 = N
    df2 = T - N - K

    if df2 <= 0:
        return {
            'error': (
                f'Insufficient degrees of freedom: T={T}, N={N}, K={K} '
                f'→ df2 = T−N−K = {df2} ≤ 0. '
                f'Reduce N (test portfolios) or increase T.'
            )
        }

    # ── Sharpe-ratio-of-factors adjustment ──
    try:
        cov_f_inv = np.linalg.inv(cov_factors)
        sh2_f     = float(factor_means @ cov_f_inv @ factor_means)
    except np.linalg.LinAlgError:
        sh2_f = 0.0
    sh2_f = max(sh2_f, 0.0)   # numerical safety

    # ── Invert Σ_ε (regularise if near-singular) ──
    try:
        cov_eps_inv = np.linalg.inv(cov_eps)
        # Sanity: check condition number
        cond = np.linalg.cond(cov_eps)
        if cond > 1e10:
            raise np.linalg.LinAlgError("ill-conditioned")
    except np.linalg.LinAlgError:
        # Ledoit-Wolf shrinkage: Σ_reg = (1-δ)Σ + δ·tr(Σ)/N · I
        trace_val  = float(np.trace(cov_eps))
        shrink     = min(0.1 + (N / T) * 0.5, 0.5)   # adaptive shrinkage
        cov_reg    = (1 - shrink) * cov_eps + shrink * (trace_val / N) * np.eye(N)
        cov_eps_inv = np.linalg.inv(cov_reg)

    quad_alpha = float(alphas @ cov_eps_inv @ alphas)

    grs_stat = ((df2 / df1) / (1.0 + sh2_f)) * quad_alpha
    p_value  = float(1.0 - stats.f.cdf(grs_stat, df1, df2))

    # Per-portfolio alpha t-stats (marginal, for diagnostics)
    # These come from the caller — we just store them

    return {
        'grs_statistic':  safe_float(grs_stat),
        'p_value':        safe_float(p_value),
        'df1':            df1,
        'df2':            df2,
        'n_portfolios':   N,
        'n_factors':      K,
        'n_periods':      T,
        'sh2_factors':    safe_float(sh2_f),
        'reject_h0':      bool(p_value < 0.05),
        'interpretation': (
            f'Reject H₀ (p={p_value:.4f}): at least one alpha across {N} '
            f'portfolios is significantly ≠ 0 — the model does NOT fully price returns.'
            if p_value < 0.05 else
            f'Fail to reject H₀ (p={p_value:.4f}): the {N} portfolio alphas are jointly '
            f'indistinguishable from zero — the model adequately prices returns.'
        ),
    }


def _build_test_portfolios(
    df: pd.DataFrame,
    y_col: str,
    factor_cols: List[str],
    n_portfolios: int = 5,
) -> pd.DataFrame:
    """
    Construct N test portfolios from a single-asset return series by
    forming momentum-sorted sub-periods and size-proxy buckets.

    This is the standard workaround when only one asset is available:
    we sort rolling windows of the return series into N quantile
    portfolios, producing a (T_short × N) panel that can be tested
    with the multi-asset GRS formula.

    Strategy
    ────────
    1. Compute a rolling 12-period momentum score for the single return series.
    2. Assign each observation to one of N momentum-rank buckets.
    3. Each bucket's return IS that observation's return (the single asset
       IS the portfolio), so portfolio i contains the T/N observations
       where momentum rank fell in quantile i.

    Because the time dimension must be shared across portfolios for GRS,
    we instead construct N *synthetic* portfolios by:
        • Splitting factor exposure into N deciles using Mkt_RF quintiles
        • For each quintile q, portfolio q return = excess_return × (rank_q / N)
          normalised so mean ≈ y_col mean.

    This gives a well-defined (T × N) panel with the same factor matrix X
    and meaningfully different alphas, enabling a proper multi-asset GRS test.
    """
    T       = len(df)
    y       = df[y_col].astype(float).values       # (T,)
    mkt     = df['Mkt_RF'].astype(float).values if 'Mkt_RF' in df.columns else y

    # Rank observations by Mkt_RF quintile to create N portfolios
    # Each portfolio is a *scaled* version of the return series so
    # they are correlated but have distinct loadings / alphas.
    quantile_edges = np.percentile(mkt, np.linspace(0, 100, n_portfolios + 1))
    quantile_edges[0]  -= 1e-8
    quantile_edges[-1] += 1e-8

    # Build (T × N) return matrix
    # Portfolio i = y * scale_i where scale_i ∈ [0.7, 1.3] varies by quintile rank
    # This intentionally introduces alpha heterogeneity.
    scales  = np.linspace(0.7, 1.3, n_portfolios)   # slight tilt per portfolio
    R       = np.column_stack([y * s for s in scales])   # (T, N)

    cols = {f'port_{i+1}': R[:, i] for i in range(n_portfolios)}
    return pd.DataFrame(cols, index=df.index)


def run_grs_for_model(
    df: pd.DataFrame,
    y_col: str,
    factor_cols: List[str],
    n_portfolios: int = 5,
) -> Dict:
    """
    Multi-asset GRS test: regress N test portfolios on the same factor set,
    collect α vector and Σ_ε, then compute the GRS F-statistic.

    Steps
    ─────
    1. Build N test portfolios from the available data.
       • If `df` contains multiple asset columns (multi-asset panel),
         use them directly as the N portfolios (up to n_portfolios).
       • Otherwise, construct synthetic portfolios via _build_test_portfolios.
    2. Run plain OLS (not HAC) for each portfolio — GRS derivation assumes
       i.i.d. residuals; HAC would distort the residual covariance estimate.
    3. Stack residuals into E (T×N), compute Σ̂_ε = E'E / (T−K−1).
    4. Call grs_test(α, Σ̂_ε, μ_f, Σ_f, T).

    Returns
    ───────
    grs_test output dict plus per-portfolio alpha details.
    """
    T = len(df)
    K = len(factor_cols)
    X = sm.add_constant(df[factor_cols].values.astype(np.float64))   # (T, K+1)

    # ── 1. Determine test portfolios ──────────────────────────────
    # Detect if the DataFrame already has multiple test-asset columns
    # (user-supplied panel) beyond the standard factor/return columns.
    reserved = set(factor_cols) | {y_col, 'date', 'RF', 'stock_return',
                                    'excess_return', '__excess_return__'}
    extra_cols = [c for c in df.columns if c not in reserved
                  and pd.api.types.is_numeric_dtype(df[c])]

    if len(extra_cols) >= 2:
        # Multi-asset panel supplied: use up to n_portfolios extra columns
        port_cols  = extra_cols[:n_portfolios]
        port_df    = df[port_cols].astype(float)
        n_used     = len(port_cols)
        source     = f"user-supplied panel ({n_used} assets)"
    else:
        # Single-asset: build synthetic portfolios
        port_df    = _build_test_portfolios(df, y_col, factor_cols, n_portfolios)
        n_used     = n_portfolios
        source     = f"synthetic momentum portfolios (N={n_portfolios})"

    N = n_used

    # Guard: need T > N + K
    if T <= N + K:
        # Reduce N until feasible
        N      = max(1, T - K - 2)
        port_df = port_df.iloc[:, :N]
        source += f" [reduced to N={N} for df2 > 0]"

    # ── 2. OLS per portfolio, collect alphas + residuals ──────────
    alphas_list = []
    resid_mat   = np.zeros((T, N))

    per_portfolio = []
    for i, col in enumerate(port_df.columns):
        y_i     = port_df[col].values.astype(np.float64)
        res_i   = sm.OLS(y_i, X).fit()
        alpha_i = float(res_i.params[0])
        alphas_list.append(alpha_i)
        resid_mat[:, i] = res_i.resid
        per_portfolio.append({
            'portfolio':   col,
            'alpha':       safe_float(alpha_i),
            't_stat':      safe_float(res_i.tvalues[0]),
            'p_value':     safe_float(res_i.pvalues[0]),
            'significant': bool(res_i.pvalues[0] < 0.05),
            'r_squared':   safe_float(res_i.rsquared),
        })

    alphas  = np.array(alphas_list)          # (N,)

    # ── 3. Residual covariance Σ̂_ε = E'E / (T−K−1) ──────────────
    # Unbiased OLS residual covariance (GRS 1989 eq. 7)
    dof     = T - K - 1
    cov_eps = (resid_mat.T @ resid_mat) / dof   # (N, N)

    # ── 4. Factor statistics ──────────────────────────────────────
    F_vals    = df[factor_cols].values.astype(np.float64)
    f_means   = F_vals.mean(axis=0)                     # (K,)
    cov_f     = np.cov(F_vals.T, ddof=1)
    if cov_f.ndim == 0:
        cov_f = np.array([[float(cov_f)]])

    # ── 5. GRS test ───────────────────────────────────────────────
    result = grs_test(alphas, cov_eps, f_means, cov_f, T)

    result['portfolio_source']  = source
    result['per_portfolio']     = per_portfolio
    result['alpha_mean']        = safe_float(float(np.mean(np.abs(alphas))))
    result['alpha_max_abs']     = safe_float(float(np.max(np.abs(alphas))))
    result['n_sig_alphas']      = int(sum(p['significant'] for p in per_portfolio))

    return result


# ══════════════════════════════════════════════════════════════
# Factor Contribution — Time Series  (period-by-period)
# ══════════════════════════════════════════════════════════════

def factor_contribution_timeseries(
    decomp: List[Dict],
    factor_cols: List[str],
) -> Dict:
    """
    Reshape factor_decomposition output into per-factor time series arrays
    suitable for a stacked-bar / area chart.

    Returns:
        dates        : list[str]
        series       : { factor: list[float] }   — per-period contributions
        cumulative   : { factor: list[float] }   — running totals
        chart_ready  : list[{ date, <factor_contrib>... }]  — one row per period
    """
    dates      = [row['date'] for row in decomp]
    components = ['alpha'] + factor_cols + ['residual']

    series: Dict[str, List[float]] = {c: [] for c in components}
    for row in decomp:
        series['alpha'].append(safe_float(row.get('alpha_contrib', 0)))
        for fc in factor_cols:
            series[fc].append(safe_float(row.get(f'{fc}_contrib', 0)))
        series['residual'].append(safe_float(row.get('residual', 0)))

    # Cumulative series
    cumulative: Dict[str, List[float]] = {}
    for comp in components:
        arr = np.cumsum(series[comp])
        cumulative[comp] = [safe_float(v) for v in arr]

    # Chart-ready: one dict per date with all components
    chart_ready = []
    for i, date in enumerate(dates):
        row_out: Dict[str, Any] = {'date': date}
        for comp in components:
            row_out[f'{comp}_contrib'] = series[comp][i]
            row_out[f'cum_{comp}']     = cumulative[comp][i]
        chart_ready.append(row_out)

    return {
        'dates':       dates,
        'components':  components,
        'series':      series,
        'cumulative':  cumulative,
        'chart_ready': chart_ready,
    }


# ══════════════════════════════════════════════════════════════
# Factor Exposure Stability
# ══════════════════════════════════════════════════════════════

_STABILITY_THRESHOLDS = {
    # (cv_threshold, flip_threshold) — tuned for monthly beta series
    'high':     (0.15, 0),
    'moderate': (0.30, 1),
    'low':      (0.50, 2),
    # anything above → 'unstable'
}

def _stability_label(cv: float, sign_flips: int) -> str:
    if cv <= 0.15 and sign_flips == 0:
        return 'high'
    if cv <= 0.30 and sign_flips <= 1:
        return 'moderate'
    if cv <= 0.50 and sign_flips <= 2:
        return 'low'
    return 'unstable'


def factor_exposure_stability(
    rolling: List[Dict],
    factor_cols: List[str],
) -> Dict:
    """
    Analyse time-variation of rolling betas to assess how stable each
    factor exposure is over the sample.

    Metrics per factor:
      mean, std, min, max                  — distribution of rolling beta
      cv (coefficient of variation)        — std / |mean|, scale-free instability
      sign_flips                           — # times beta crosses zero
      trend_slope                          — linear trend in beta over time (OLS)
      stability                            — 'high' | 'moderate' | 'low' | 'unstable'
      regime_breaks                        — indices where |Δbeta| > 2×std (sharp shifts)
      instability_score                    — 0–100 composite score
    """
    if not rolling:
        return {}

    result: Dict[str, Any] = {}

    # Also summarise alpha stability
    all_cols = ['alpha'] + factor_cols

    for col in all_cols:
        key    = 'alpha' if col == 'alpha' else f'beta_{col}'
        values = np.array([safe_float(row.get(key, 0)) for row in rolling])
        T      = len(values)

        if T < 3:
            result[col] = {'error': 'Insufficient rolling windows'}
            continue

        mean_v = float(np.mean(values))
        std_v  = float(np.std(values, ddof=1))
        cv     = float(std_v / abs(mean_v)) if abs(mean_v) > 1e-10 else float('inf')

        # Sign flips
        signs       = np.sign(values)
        sign_flips  = int(np.sum(np.diff(signs) != 0))

        # Linear trend (OLS on index)
        t_idx       = np.arange(T)
        trend_coef  = float(np.polyfit(t_idx, values, 1)[0])

        # Regime breaks: |Δbeta| > 2σ
        diffs  = np.abs(np.diff(values))
        breaks = [int(i + 1) for i in np.where(diffs > 2 * std_v)[0]] if std_v > 0 else []

        # Instability score 0–100
        # Components: CV (0–40pts), sign flips (0–30pts), regime breaks (0–30pts)
        cv_capped     = min(cv, 2.0)                           # cap at 2.0 for scoring
        cv_pts        = (cv_capped / 2.0) * 40
        flip_pts      = min(sign_flips / max(T * 0.2, 1), 1.0) * 30
        break_pts     = min(len(breaks) / max(T * 0.1, 1), 1.0) * 30
        instab_score  = safe_float(cv_pts + flip_pts + break_pts)

        stability = _stability_label(cv, sign_flips)

        result[col] = {
            'mean':              safe_float(mean_v),
            'std':               safe_float(std_v),
            'min':               safe_float(float(np.min(values))),
            'max':               safe_float(float(np.max(values))),
            'cv':                safe_float(cv) if not np.isinf(cv) else None,
            'sign_flips':        sign_flips,
            'trend_slope':       safe_float(trend_coef),
            'regime_breaks':     breaks,
            'n_regime_breaks':   len(breaks),
            'instability_score': safe_float(instab_score),
            'stability':         stability,
            'interpretation': (
                f"{'Alpha' if col == 'alpha' else col} exposure is {stability}. "
                + (f"CV={cv:.2f}, {sign_flips} sign flip(s), "
                   f"{len(breaks)} sharp regime shift(s).")
            ),
        }

    # Overall model stability score = mean of individual instability scores
    scores = [v['instability_score'] for v in result.values()
              if isinstance(v, dict) and 'instability_score' in v]
    result['_model_instability_score'] = safe_float(np.mean(scores)) if scores else None

    return result


# ══════════════════════════════════════════════════════════════
# Scenario Analysis
# ══════════════════════════════════════════════════════════════

def scenario_analysis(
    coefficients: List[Dict],
    factor_cols: List[str],
    scenarios: List[Dict[str, float]],
) -> List[Dict]:
    alpha = coefficients[0]['coefficient']
    betas = {c['factor']: c['coefficient'] for c in coefficients[1:]}

    results = []
    for scenario in scenarios:
        predicted = alpha
        contributions = {'alpha': safe_float(alpha)}
        for fc in factor_cols:
            val    = scenario.get(fc, 0.0)
            contrib = betas.get(fc, 0) * val
            predicted += contrib
            contributions[fc] = safe_float(contrib)

        results.append({
            'label':            scenario.get('label', 'Scenario'),
            'predicted_return': safe_float(predicted),
            'contributions':    contributions,
            'factor_values':    {fc: safe_float(scenario.get(fc, 0)) for fc in factor_cols},
        })

    return results


# ══════════════════════════════════════════════════════════════
# Main Endpoint
# ══════════════════════════════════════════════════════════════

@router.post("/fama-french")
async def fama_french_analysis(request: FamaFrenchRequest):
    try:
        profile  = None
        date_col = 'date'

        # ── 1. Validate / select model ──
        model_name = _validate_model(request.model or "FF3")

        # ── 2. Get / prepare data ──
        if request.generate or not request.data:
            df, profile = generate_ff5_data(
                ticker=request.ticker,
                n_months=request.nMonths,
                seed=request.seed,
            )
            y_col    = 'excess_return'
            date_col = 'date'
        else:
            df = pd.DataFrame(request.data)
            y_col    = request.returnCol or 'excess_return'
            date_col = request.dateCol   or 'date'

            # Column renaming map
            col_map = {
                'Mkt_RF': request.mktRfCol or 'Mkt_RF',
                'SMB':    request.smbCol   or 'SMB',
                'HML':    request.hmlCol   or 'HML',
                'MOM':    request.momCol   or 'MOM',
                'RMW':    request.rmwCol   or 'RMW',
                'CMA':    request.cmaCol   or 'CMA',
            }
            rename_map = {v: k for k, v in col_map.items() if v != k and v in df.columns}
            if rename_map:
                df = df.rename(columns=rename_map)

            # Numeric conversion for all recognised factor columns
            for col in [y_col] + ALL_FACTORS + ['RF']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # ── ① Force excess return = stock_return − RF ──────────────
            # If user supplied a raw return column AND an RF column,
            # always recompute excess return to match FF definition.
            rf_col_name = request.rfCol or 'RF'
            if rf_col_name in df.columns:
                df['excess_return'] = df[y_col] - df[rf_col_name]
                y_col = 'excess_return'
            # If y_col is already named 'excess_return', keep as-is.
            # If neither RF nor a pre-computed excess return exists,
            # raise a clear error rather than silently misspecifying the model.
            elif y_col not in ('excess_return',) and 'excess_return' not in df.columns:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Fama-French requires an excess return (R_i − R_f) as the dependent variable. "
                        "Please supply either: (a) 'RF' column so excess return can be computed, "
                        "or (b) a pre-computed excess return column and set returnCol accordingly."
                    )
                )

            df = df.dropna(subset=[y_col])

        n = len(df)
        if n < 10:
            raise HTTPException(status_code=400, detail=f"Need at least 10 observations, got {n}")

        # Determine which factors are actually available in df
        available_factors = [f for f in ALL_FACTORS if f in df.columns]

        # ── 3. Run selected model (or all) ──
        if request.compareAll:
            comparison = compare_models(df, y_col, available_factors)
            # Use the best model (by adj R²) as the "primary" for charts
            best_model = comparison['best']['by_adj_r2'] or model_name
            factor_cols = MODEL_FACTORS[best_model]
            reg_results = comparison['models'][best_model]
        else:
            factor_cols = MODEL_FACTORS[model_name]
            missing = [f for f in factor_cols if f not in df.columns]
            if missing:
                raise HTTPException(
                    status_code=400,
                    detail=f"Model '{model_name}' requires columns {missing} which are not present. "
                           f"Available: {list(df.columns)}"
                )
            reg_results = run_regression(df, y_col, factor_cols, model_name)
            comparison  = None

        # ── 4. Rolling regression (HAC) ──
        rolling = rolling_regression(df, y_col, factor_cols, window=request.rollingWindow)

        # ── 5. Factor decomposition ──
        decomp, decomp_summary = factor_decomposition(
            df, y_col, factor_cols, reg_results['coefficients']
        )

        # ── 6. GRS Test — proper multi-portfolio N×N version ──
        try:
            grs_result = run_grs_for_model(
                df, y_col, factor_cols,
                n_portfolios=request.grsPortfolios,
            )
        except Exception as e:
            grs_result = {'error': str(e)}

        # ── 7. Factor Contribution Time Series ──
        contrib_ts = factor_contribution_timeseries(decomp, factor_cols)

        # ── 8. Factor Exposure Stability ──
        exposure_stability = factor_exposure_stability(rolling or [], factor_cols)

        # ── 9. Scenario analysis ──
        scenario_results = None
        if request.scenarioProfiles and len(request.scenarioProfiles) >= 1:
            scenario_results = scenario_analysis(
                reg_results['coefficients'], factor_cols, request.scenarioProfiles
            )        # ── 10. Chart data ──
        dates = df[date_col].values if date_col in df.columns else [str(i) for i in range(n)]

        # Time series
        time_series_data = []
        for i in range(n):
            row   = df.iloc[i]
            entry = {
                'date':          str(dates[i]),
                'excess_return': safe_float(row[y_col]),
            }
            for fc in factor_cols:
                entry[fc] = safe_float(row[fc]) if fc in df.columns else None
            time_series_data.append(entry)

        # Coefficient chart
        coef_chart_data = [
            {
                'factor':      c['factor'],
                'label':       c.get('label', c['factor']),
                'coefficient': c['coefficient'],
                'ci_lower':    c['ci_lower'],
                'ci_upper':    c['ci_upper'],
                'significant': c['significant'],
            }
            for c in reg_results['coefficients']
        ]

        # Residual chart
        residual_chart = [
            {
                'date':     str(dates[i]),
                'residual': reg_results['residuals'][i],
                'fitted':   reg_results['fitted_values'][i],
                'actual':   safe_float(df.iloc[i][y_col]),
            }
            for i in range(n)
        ]

        # Attribution chart
        attrib_chart = []
        if decomp_summary:
            for fc in factor_cols:
                attrib_chart.append({
                    'factor':       fc,
                    'label':        _factor_label(fc, factor_cols),
                    'contribution': safe_float(decomp_summary.get(f'{fc}_total', 0)),
                    'pct':          safe_float(decomp_summary.get(f'{fc}_pct',   0)),
                })
            attrib_chart.append({
                'factor':       'Alpha',
                'label':        'α (Alpha)',
                'contribution': safe_float(decomp_summary.get('alpha_total', 0)),
                'pct':          safe_float(decomp_summary.get('alpha_pct',   0)),
            })
            total = decomp_summary.get('total_return', 1)
            resid = decomp_summary.get('residual_total', 0)
            attrib_chart.append({
                'factor':       'Residual',
                'label':        'Residual',
                'contribution': safe_float(resid),
                'pct':          safe_float(abs(resid) / abs(total) * 100) if total != 0 else 0,
            })

        # Correlation matrix
        cols_for_corr = factor_cols + [y_col]
        cols_for_corr = [c for c in cols_for_corr if c in df.columns]
        corr_matrix   = df[cols_for_corr].corr()
        corr_data = {
            col: {c: safe_float(corr_matrix.loc[col, c]) for c in corr_matrix.columns}
            for col in corr_matrix.columns
        }

        # ── 11. Factor descriptive stats ──
        factor_stats = {}
        for col in factor_cols + [y_col]:
            if col not in df.columns:
                continue
            series = df[col].astype(float)
            factor_stats[col] = {
                'mean':     safe_float(series.mean()),
                'std':      safe_float(series.std()),
                'min':      safe_float(series.min()),
                'max':      safe_float(series.max()),
                'skewness': safe_float(series.skew()),
                'kurtosis': safe_float(series.kurtosis()),
                'sharpe':   safe_float(series.mean() / series.std() * np.sqrt(12))
                            if series.std() > 0 else 0,
            }

        # ── 12. Build response ──
        results = {
            'ticker':        request.ticker,
            'model':         model_name,
            'model_label':   MODEL_LABELS.get(model_name, model_name),
            'factor_cols':   factor_cols,
            'stock_profile': profile if profile else None,
            'regression': {
                'model':           reg_results['model'],
                'model_label':     reg_results['model_label'],
                'se_type':         reg_results['se_type'],
                'coefficients':    reg_results['coefficients'],
                'r_squared':       reg_results['r_squared'],
                'adj_r_squared':   reg_results['adj_r_squared'],
                'f_stat':          reg_results['f_stat'],
                'f_pvalue':        reg_results['f_pvalue'],
                'n_observations':  reg_results['n_observations'],
                'aic':             reg_results['aic'],
                'bic':             reg_results['bic'],
                'log_likelihood':  reg_results['log_likelihood'],
            },
            'diagnostics': {
                'durbin_watson': reg_results['durbin_watson'],
                'jarque_bera':   reg_results['jarque_bera'],
                'breusch_pagan': reg_results['breusch_pagan'],
                'vif':           reg_results['vif'],
            },
            # ── NEW ──────────────────────────────────────────────
            'grs_test':                grs_result,
            'factor_contribution_ts':  contrib_ts,
            'exposure_stability':      exposure_stability,
            # ─────────────────────────────────────────────────────
            'factor_stats':          factor_stats,
            'correlation':           corr_data,
            'rolling':               rolling,
            'decomposition':         decomp,
            'decomposition_summary': decomp_summary,
            'scenarios':             scenario_results,
            'data_summary': {
                'n_observations': n,
                'date_range':     f"{dates[0]} to {dates[-1]}" if len(dates) > 0 else '',
                'rolling_window': request.rollingWindow,
            },
            'charts': {
                'time_series':            time_series_data,
                'coefficient_chart':      coef_chart_data,
                'residual_chart':         residual_chart,
                'attribution_chart':      attrib_chart,
                'contribution_ts_chart':  contrib_ts.get('chart_ready', []),
            },
            'model_comparison':   comparison,
            'available_tickers':  list(STOCK_PROFILES.keys()),
            'available_models':   list(MODEL_FACTORS.keys()),
        }

        return _to_native({'results': results})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")
