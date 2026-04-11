from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats
from scipy.stats import spearmanr, pearsonr
import warnings
import traceback

warnings.filterwarnings('ignore')

router = APIRouter()


# ══════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════

ALL_FACTORS = ["Mkt_RF", "SMB", "HML", "MOM", "RMW", "CMA"]

FACTOR_LABELS = {
    "Mkt_RF": "Market (Mkt-RF)",
    "SMB":    "Size (SMB)",
    "HML":    "Value (HML)",
    "MOM":    "Momentum (MOM)",
    "RMW":    "Profitability (RMW)",
    "CMA":    "Investment (CMA)",
}

# Realistic stock profiles for data generation
STOCK_PROFILES = {
    'AAPL':  {'name': 'Apple Inc.',         'beta': 1.20, 'smb': -0.15, 'hml': -0.45, 'mom':  0.10, 'rmw':  0.20, 'cma': -0.30, 'alpha': 0.005, 'sigma': 0.04, 'sector': 'Tech',     'size': 'Large', 'bm': 0.15},
    'MSFT':  {'name': 'Microsoft Corp.',    'beta': 1.10, 'smb': -0.20, 'hml': -0.30, 'mom':  0.08, 'rmw':  0.25, 'cma': -0.20, 'alpha': 0.004, 'sigma': 0.035,'sector': 'Tech',     'size': 'Large', 'bm': 0.12},
    'GOOGL': {'name': 'Alphabet Inc.',      'beta': 1.05, 'smb': -0.18, 'hml': -0.40, 'mom':  0.05, 'rmw':  0.15, 'cma': -0.25, 'alpha': 0.003, 'sigma': 0.04, 'sector': 'Tech',     'size': 'Large', 'bm': 0.18},
    'AMZN':  {'name': 'Amazon.com Inc.',    'beta': 1.25, 'smb': -0.10, 'hml': -0.55, 'mom':  0.15, 'rmw': -0.10, 'cma': -0.35, 'alpha': 0.004, 'sigma': 0.05, 'sector': 'Tech',     'size': 'Large', 'bm': 0.10},
    'TSLA':  {'name': 'Tesla Inc.',         'beta': 1.80, 'smb':  0.30, 'hml': -0.70, 'mom':  0.20, 'rmw': -0.30, 'cma': -0.50, 'alpha': 0.006, 'sigma': 0.10, 'sector': 'EV',       'size': 'Large', 'bm': 0.08},
    'NVDA':  {'name': 'NVIDIA Corp.',       'beta': 1.60, 'smb':  0.10, 'hml': -0.60, 'mom':  0.25, 'rmw':  0.10, 'cma': -0.45, 'alpha': 0.008, 'sigma': 0.08, 'sector': 'Semis',   'size': 'Large', 'bm': 0.09},
    'JPM':   {'name': 'JPMorgan Chase',     'beta': 1.15, 'smb': -0.05, 'hml':  0.60, 'mom':  0.05, 'rmw':  0.35, 'cma':  0.10, 'alpha': 0.002, 'sigma': 0.04, 'sector': 'Finance', 'size': 'Large', 'bm': 0.65},
    'JNJ':   {'name': 'Johnson & Johnson',  'beta': 0.65, 'smb': -0.25, 'hml':  0.30, 'mom': -0.05, 'rmw':  0.40, 'cma':  0.15, 'alpha': 0.001, 'sigma': 0.025,'sector': 'Health',  'size': 'Large', 'bm': 0.45},
    'XOM':   {'name': 'Exxon Mobil',        'beta': 0.90, 'smb': -0.10, 'hml':  0.70, 'mom':  0.00, 'rmw':  0.25, 'cma':  0.30, 'alpha': 0.000, 'sigma': 0.04, 'sector': 'Energy',  'size': 'Large', 'bm': 0.80},
    'WMT':   {'name': 'Walmart Inc.',       'beta': 0.55, 'smb': -0.30, 'hml':  0.20, 'mom': -0.02, 'rmw':  0.30, 'cma':  0.10, 'alpha': 0.001, 'sigma': 0.025,'sector': 'Retail',  'size': 'Large', 'bm': 0.40},
    'PG':    {'name': 'Procter & Gamble',   'beta': 0.50, 'smb': -0.30, 'hml':  0.25, 'mom': -0.03, 'rmw':  0.45, 'cma':  0.15, 'alpha': 0.001, 'sigma': 0.02, 'sector': 'Consumer','size': 'Large', 'bm': 0.38},
    'KO':    {'name': 'Coca-Cola Co.',      'beta': 0.55, 'smb': -0.28, 'hml':  0.15, 'mom': -0.02, 'rmw':  0.40, 'cma':  0.10, 'alpha': 0.001, 'sigma': 0.02, 'sector': 'Consumer','size': 'Large', 'bm': 0.35},
    'META':  {'name': 'Meta Platforms',     'beta': 1.30, 'smb': -0.12, 'hml': -0.35, 'mom':  0.12, 'rmw':  0.15, 'cma': -0.25, 'alpha': 0.003, 'sigma': 0.05, 'sector': 'Tech',    'size': 'Large', 'bm': 0.20},
    'AMD':   {'name': 'AMD Inc.',           'beta': 1.70, 'smb':  0.25, 'hml': -0.55, 'mom':  0.22, 'rmw': -0.05, 'cma': -0.40, 'alpha': 0.006, 'sigma': 0.09, 'sector': 'Semis',   'size': 'Mid',   'bm': 0.11},
    'INTC':  {'name': 'Intel Corp.',        'beta': 1.05, 'smb':  0.00, 'hml':  0.20, 'mom': -0.08, 'rmw':  0.15, 'cma':  0.10, 'alpha':-0.002, 'sigma': 0.05, 'sector': 'Semis',   'size': 'Large', 'bm': 0.55},
    'NFLX':  {'name': 'Netflix Inc.',       'beta': 1.35, 'smb':  0.05, 'hml': -0.50, 'mom':  0.18, 'rmw':  0.05, 'cma': -0.35, 'alpha': 0.005, 'sigma': 0.07, 'sector': 'Media',   'size': 'Large', 'bm': 0.14},
    'BA':    {'name': 'Boeing Co.',         'beta': 1.30, 'smb':  0.05, 'hml':  0.25, 'mom': -0.05, 'rmw': -0.10, 'cma':  0.15, 'alpha':-0.001, 'sigma': 0.06, 'sector': 'Industry','size': 'Large', 'bm': 0.60},
    'GS':    {'name': 'Goldman Sachs',      'beta': 1.40, 'smb': -0.05, 'hml':  0.50, 'mom':  0.10, 'rmw':  0.30, 'cma':  0.05, 'alpha': 0.002, 'sigma': 0.055,'sector': 'Finance', 'size': 'Large', 'bm': 0.70},
    'V':     {'name': 'Visa Inc.',          'beta': 0.95, 'smb': -0.20, 'hml': -0.15, 'mom':  0.06, 'rmw':  0.35, 'cma': -0.10, 'alpha': 0.003, 'sigma': 0.03, 'sector': 'Finance', 'size': 'Large', 'bm': 0.22},
    'BRK.B': {'name': 'Berkshire Hathaway','beta': 0.85, 'smb': -0.15, 'hml':  0.40, 'mom':  0.02, 'rmw':  0.30, 'cma':  0.20, 'alpha': 0.003, 'sigma': 0.03, 'sector': 'Finance', 'size': 'Large', 'bm': 0.75},
}

DEFAULT_TICKERS = list(STOCK_PROFILES.keys())


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

def _to_native(obj):
    if isinstance(obj, (np.integer,)):   return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    if isinstance(obj, np.ndarray):      return [_to_native(x) for x in obj.tolist()]
    if isinstance(obj, np.bool_):        return bool(obj)
    if isinstance(obj, dict):            return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):            return [_to_native(x) for x in obj]
    return obj


def safe_float(val, default=0.0):
    try:
        if val is None: return default
        f = float(val)
        return default if (np.isnan(f) or np.isinf(f)) else f
    except Exception:
        return default


# ══════════════════════════════════════════════════════════════
# Data Generator — cross-sectional panel (N stocks × T months)
# ══════════════════════════════════════════════════════════════

def generate_panel_data(
    tickers: List[str],
    n_months: int = 60,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate a cross-sectional panel:
      rows   = stock × month  (N × T rows total)
      cols   = date, ticker, return, Mkt_RF, SMB, HML, MOM, RMW, CMA, RF,
               size_score, value_score, momentum_score, quality_score,
               market_cap, book_to_market, past_12m_return
    Factor returns are shared across stocks (same market environment per month).
    Stock returns are generated from the factor model + idiosyncratic noise.
    """
    rng = np.random.default_rng(seed)

    # ── Common factor returns (T × 6) ──
    factor_means  = np.array([0.0065, 0.0020, 0.0030, 0.0060, 0.0025, 0.0020])
    factor_stds   = np.array([0.0450, 0.0300, 0.0300, 0.0400, 0.0200, 0.0180])
    corr = np.array([
        [ 1.00,  0.30, -0.25, -0.10,  0.05, -0.10],
        [ 0.30,  1.00,  0.10, -0.05, -0.35, -0.10],
        [-0.25,  0.10,  1.00, -0.35, -0.10,  0.70],
        [-0.10, -0.05, -0.35,  1.00,  0.10, -0.20],
        [ 0.05, -0.35, -0.10,  0.10,  1.00,  0.00],
        [-0.10, -0.10,  0.70, -0.20,  0.00,  1.00],
    ])
    cov = np.outer(factor_stds, factor_stds) * corr
    F = rng.multivariate_normal(factor_means, cov, size=n_months)  # (T, 6)
    mkt_rf, smb, hml, mom, rmw, cma = [F[:, i] for i in range(6)]

    rf_base = 0.0025
    rf = np.maximum(rf_base + np.cumsum(rng.normal(0, 0.0005, n_months) * 0.1),
                    0.0001)
    rf = rf - rf.mean() + rf_base

    end_date = pd.Timestamp('2025-04-30')
    dates = pd.date_range(end=end_date, periods=n_months, freq='ME')
    date_strs = dates.strftime('%Y-%m').tolist()

    rows = []
    for ticker in tickers:
        p = STOCK_PROFILES.get(ticker, {
            'name': ticker, 'beta': 1.0, 'smb': 0.0, 'hml': 0.0,
            'mom': 0.0, 'rmw': 0.0, 'cma': 0.0, 'alpha': 0.002,
            'sigma': 0.04, 'sector': 'Unknown', 'size': 'Mid', 'bm': 0.5,
        })
        eps = rng.normal(0, p['sigma'], n_months)
        excess_ret = (
            p['alpha']
            + p['beta'] * mkt_rf
            + p['smb']  * smb
            + p['hml']  * hml
            + p['mom']  * mom
            + p['rmw']  * rmw
            + p['cma']  * cma
            + eps
        )

        # Rolling 12m momentum signal (lagged return)
        cum_ret = np.cumprod(1 + excess_ret)

        for t in range(n_months):
            # Characteristic signals (cross-sectionally meaningful)
            # Add time-varying noise so they vary across periods
            size_noise     = rng.normal(0, 0.05)
            value_noise    = rng.normal(0, 0.05)
            quality_noise  = rng.normal(0, 0.04)

            # Past-12m momentum: cumulative return over prior 12 months
            if t >= 12:
                mom_signal = float(cum_ret[t - 1] / cum_ret[t - 12] - 1)
            elif t >= 1:
                mom_signal = float(cum_ret[t - 1] - 1)
            else:
                mom_signal = 0.0

            rows.append({
                'date':             date_strs[t],
                'ticker':           ticker,
                'name':             p['name'],
                'sector':           p['sector'],
                # Returns (%)
                'return':           round(float(excess_ret[t] + rf[t]) * 100, 4),
                'excess_return':    round(float(excess_ret[t]) * 100, 4),
                # Common factors (%)
                'Mkt_RF':           round(float(mkt_rf[t]) * 100, 4),
                'SMB':              round(float(smb[t])    * 100, 4),
                'HML':              round(float(hml[t])    * 100, 4),
                'MOM':              round(float(mom[t])    * 100, 4),
                'RMW':              round(float(rmw[t])    * 100, 4),
                'CMA':              round(float(cma[t])    * 100, 4),
                'RF':               round(float(rf[t])     * 100, 4),
                # Characteristics / signals
                'size_score':       round(float(-p['smb'] + size_noise), 4),      # large = high
                'value_score':      round(float( p['hml'] + value_noise), 4),     # value = high
                'momentum_score':   round(float(mom_signal * 100), 4),            # %
                'quality_score':    round(float( p['rmw'] + quality_noise), 4),   # profitable = high
                'inv_score':        round(float(-p['cma'] + rng.normal(0, 0.03)), 4),
                # Fundamental proxies
                'market_cap':       round(float(np.exp(10 - p['smb'] * 2 + rng.normal(0, 0.1))), 2),
                'book_to_market':   round(float(max(p['bm'] + rng.normal(0, 0.05), 0.01)), 4),
                'past_12m_return':  round(float(mom_signal * 100), 4),
                # True loadings (for exposure comparison)
                'true_beta_Mkt_RF': p['beta'],
                'true_beta_SMB':    p['smb'],
                'true_beta_HML':    p['hml'],
                'true_beta_MOM':    p['mom'],
                'true_beta_RMW':    p['rmw'],
                'true_beta_CMA':    p['cma'],
            })

    df = pd.DataFrame(rows)
    df['date'] = pd.Categorical(df['date'], categories=date_strs, ordered=True)
    df = df.sort_values(['date', 'ticker']).reset_index(drop=True)
    df['date'] = df['date'].astype(str)
    return df


# ══════════════════════════════════════════════════════════════
# Request Models
# ══════════════════════════════════════════════════════════════

class PanelRequest(BaseModel):
    """Shared base for all panel-data endpoints."""
    # Data
    data:           Optional[List[Dict[str, Any]]] = None
    dateCol:        str = 'date'
    tickerCol:      str = 'ticker'
    returnCol:      str = 'excess_return'
    factorCols:     Optional[List[str]] = None      # defaults to ALL_FACTORS present in data
    # Signal/characteristic columns (for IC, cross-sectional)
    signalCols:     Optional[List[str]] = None
    # Generate mode
    generate:       bool = True
    tickers:        Optional[List[str]] = None
    nMonths:        int = 60
    seed:           Optional[int] = None
    # Common params
    rollingWindow:  int = 24


# ── 1. Factor Exposure ──────────────────────────────────────

class FactorExposureRequest(PanelRequest):
    model:          str = 'FF5'            # CAPM | FF3 | Carhart4 | FF5
    rollingWindow:  int = 36
    minObs:         int = 24               # min obs per stock for reliable beta
    heatmapPeriods: int = 12               # periods for trailing heatmap


# ── 2. Factor IC ────────────────────────────────────────────

class FactorICRequest(PanelRequest):
    icMethod:       str = 'rank'           # 'rank' (Spearman) | 'pearson'
    forwardPeriods: int = 1                # forward return lag
    rollingWindow:  int = 24
    signalCols:     Optional[List[str]] = None


# ── 3. Factor Decay ─────────────────────────────────────────

class FactorDecayRequest(PanelRequest):
    maxLag:         int = 12              # max forward horizon to test
    icMethod:       str = 'rank'
    signalCols:     Optional[List[str]] = None


# ── 4. Factor Rotation ──────────────────────────────────────

class FactorRotationRequest(PanelRequest):
    rollingWindow:  int = 24
    topN:           int = 2               # top N factors per period for rotation map


# ── 5. Long-Short Portfolio ─────────────────────────────────

class LongShortRequest(PanelRequest):
    signalCol:      str = 'momentum_score'
    nQuantiles:     int = 5               # number of quantile buckets (5 = quintiles)
    rebalance:      str = 'monthly'       # monthly | quarterly
    weightScheme:   str = 'equal'         # equal | value_weighted
    marketCapCol:   Optional[str] = 'market_cap'
    transactionCost: float = 0.001        # one-way cost (0.1%)


# ── 6. Cross-Sectional Regression (Fama-MacBeth) ────────────

class CrossSectionalRequest(PanelRequest):
    signalCols:     Optional[List[str]] = None
    controls:       Optional[List[str]] = None   # control variables
    neweyWestLags:  int = 4
    minStocksPerPeriod: int = 5


# ══════════════════════════════════════════════════════════════
# Shared Panel Prep
# ══════════════════════════════════════════════════════════════

MODEL_FACTORS = {
    'CAPM':     ['Mkt_RF'],
    'FF3':      ['Mkt_RF', 'SMB', 'HML'],
    'Carhart4': ['Mkt_RF', 'SMB', 'HML', 'MOM'],
    'FF5':      ['Mkt_RF', 'SMB', 'HML', 'RMW', 'CMA'],
}

DEFAULT_SIGNALS = ['size_score', 'value_score', 'momentum_score', 'quality_score', 'inv_score']


def _prepare_panel(req: PanelRequest) -> pd.DataFrame:
    if req.generate or not req.data:
        tickers = req.tickers or DEFAULT_TICKERS[:15]
        df = generate_panel_data(tickers, n_months=req.nMonths, seed=req.seed)
    else:
        df = pd.DataFrame(req.data)
        for col in df.select_dtypes(include=['object']).columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='ignore')
            except Exception:
                pass
    return df


def _factor_cols_for_model(model: str, df: pd.DataFrame) -> List[str]:
    wanted = MODEL_FACTORS.get(model.upper().replace('-','').replace(' ',''),
                               MODEL_FACTORS['FF5'])
    return [f for f in wanted if f in df.columns]


# ══════════════════════════════════════════════════════════════
# 1. FACTOR EXPOSURE
# ══════════════════════════════════════════════════════════════

def _run_ols_stock(df_stock: pd.DataFrame, ret_col: str,
                   factor_cols: List[str]) -> Optional[Dict]:
    """OLS with HAC SE for a single stock time series."""
    sub = df_stock[[ret_col] + factor_cols].dropna()
    if len(sub) < len(factor_cols) + 3:
        return None
    y = sub[ret_col].values.astype(np.float64)
    X = sm.add_constant(sub[factor_cols].values.astype(np.float64))
    try:
        res = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': 3})
        coefs = {}
        for i, name in enumerate(['alpha'] + factor_cols):
            coefs[name] = {
                'coef':  safe_float(res.params[i]),
                'tstat': safe_float(res.tvalues[i]),
                'pval':  safe_float(res.pvalues[i]),
                'sig':   bool(res.pvalues[i] < 0.05),
            }
        return {
            'coefs':        coefs,
            'r2':           safe_float(res.rsquared),
            'adj_r2':       safe_float(res.rsquared_adj),
            'n':            int(res.nobs),
        }
    except Exception:
        return None


def compute_factor_exposure(df: pd.DataFrame, req: FactorExposureRequest) -> Dict:
    factor_cols = _factor_cols_for_model(req.model, df)
    ret_col     = req.returnCol
    tickers     = df[req.tickerCol].unique().tolist()

    # ── Full-sample betas ──────────────────────────────────
    full_sample: List[Dict] = []
    for ticker in tickers:
        sub = df[df[req.tickerCol] == ticker].sort_values(req.dateCol)
        res = _run_ols_stock(sub, ret_col, factor_cols)
        if res is None:
            continue
        p = STOCK_PROFILES.get(ticker, {})
        row = {
            'ticker':  ticker,
            'name':    p.get('name', ticker),
            'sector':  p.get('sector', ''),
            'n_obs':   res['n'],
            'r2':      res['r2'],
            'adj_r2':  res['adj_r2'],
            'alpha':   res['coefs']['alpha']['coef'],
            'alpha_sig': res['coefs']['alpha']['sig'],
        }
        for f in factor_cols:
            c = res['coefs'].get(f, {})
            row[f'beta_{f}']  = c.get('coef',  None)
            row[f'tstat_{f}'] = c.get('tstat', None)
            row[f'sig_{f}']   = c.get('sig',   False)
        full_sample.append(row)

    # ── Rolling betas (averaged across tickers, per factor) ──
    dates_sorted = sorted(df[req.dateCol].unique())
    T = len(dates_sorted)
    w = req.rollingWindow
    rolling_avg: List[Dict] = []

    date_to_idx = {d: i for i, d in enumerate(dates_sorted)}
    df_indexed  = df.copy()
    df_indexed['_t'] = df_indexed[req.dateCol].map(date_to_idx)

    for end_t in range(w - 1, T):
        start_t = end_t - w + 1
        window_df = df_indexed[(df_indexed['_t'] >= start_t) &
                                (df_indexed['_t'] <= end_t)]
        entry = {'date': dates_sorted[end_t]}
        for f in factor_cols:
            betas = []
            for ticker in tickers:
                sub = window_df[window_df[req.tickerCol] == ticker]
                if len(sub) < max(len(factor_cols) + 2, w // 2):
                    continue
                res = _run_ols_stock(sub, ret_col, factor_cols)
                if res and f in res['coefs']:
                    betas.append(res['coefs'][f]['coef'])
            entry[f'avg_beta_{f}'] = safe_float(np.mean(betas)) if betas else None
            entry[f'std_beta_{f}'] = safe_float(np.std(betas))  if len(betas) > 1 else None
        entry['n_stocks'] = len(tickers)
        rolling_avg.append(entry)

    # ── Sector-level average exposures ────────────────────
    sector_exposure: Dict[str, Dict] = {}
    if full_sample:
        fs_df = pd.DataFrame(full_sample)
        if 'sector' in fs_df.columns:
            for sector, grp in fs_df.groupby('sector'):
                row = {'sector': sector, 'n': len(grp)}
                for f in factor_cols:
                    col = f'beta_{f}'
                    if col in grp.columns:
                        row[f'avg_beta_{f}'] = safe_float(grp[col].mean())
                sector_exposure[str(sector)] = row

    # ── Exposure heatmap (recent N periods, per-stock beta_Mkt_RF) ──
    recent_dates = dates_sorted[-req.heatmapPeriods:]
    heatmap: List[Dict] = []
    for ticker in tickers[:20]:  # cap at 20 for readability
        sub = df[df[req.tickerCol] == ticker].set_index(req.dateCol)
        row = {'ticker': ticker}
        for d in recent_dates:
            if d in sub.index:
                row[d] = safe_float(sub.loc[d, ret_col])
            else:
                row[d] = None
        heatmap.append(row)

    # ── Cross-stock dispersion in beta ────────────────────
    dispersion: List[Dict] = []
    if full_sample:
        fs_df = pd.DataFrame(full_sample)
        for f in factor_cols:
            col = f'beta_{f}'
            if col not in fs_df.columns:
                continue
            vals = fs_df[col].dropna()
            dispersion.append({
                'factor': f,
                'label':  FACTOR_LABELS.get(f, f),
                'mean':   safe_float(vals.mean()),
                'std':    safe_float(vals.std()),
                'min':    safe_float(vals.min()),
                'max':    safe_float(vals.max()),
                'q25':    safe_float(vals.quantile(0.25)),
                'q75':    safe_float(vals.quantile(0.75)),
                'n_sig':  int((fs_df.get(f'sig_{f}', pd.Series(dtype=bool))).sum()),
            })

    return {
        'model':           req.model,
        'factor_cols':     factor_cols,
        'n_stocks':        len(tickers),
        'n_periods':       T,
        'full_sample':     full_sample,
        'rolling_avg':     rolling_avg,
        'sector_exposure': sector_exposure,
        'dispersion':      dispersion,
        'heatmap_dates':   recent_dates,
        'heatmap':         heatmap,
    }


@router.post("/factor-exposure")
async def factor_exposure_endpoint(request: FactorExposureRequest):
    try:
        df = _prepare_panel(request)
        result = compute_factor_exposure(df, request)
        return _to_native({'results': result})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════
# 2. FACTOR IC  (Information Coefficient)
# ══════════════════════════════════════════════════════════════

def compute_ic(
    df: pd.DataFrame,
    signal_col: str,
    fwd_ret_col: str,
    method: str = 'rank',
) -> Optional[float]:
    """IC for one period: correlation between signal and forward return."""
    sub = df[[signal_col, fwd_ret_col]].dropna()
    if len(sub) < 5:
        return None
    x, y = sub[signal_col].values, sub[fwd_ret_col].values
    try:
        if method == 'rank':
            ic, _ = spearmanr(x, y)
        else:
            ic, _ = pearsonr(x, y)
        return safe_float(ic)
    except Exception:
        return None


def compute_factor_ic(df: pd.DataFrame, req: FactorICRequest) -> Dict:
    signal_cols = req.signalCols or [c for c in DEFAULT_SIGNALS if c in df.columns]
    dates_sorted = sorted(df[req.dateCol].unique())
    T = len(dates_sorted)

    # Build forward return: for each stock, return at t+lag
    df = df.sort_values([req.tickerCol, req.dateCol]).copy()
    df['_fwd_ret'] = df.groupby(req.tickerCol)[req.returnCol].shift(-req.forwardPeriods)

    # ── Per-period IC ──────────────────────────────────────
    ic_series: Dict[str, List[Dict]] = {s: [] for s in signal_cols}

    for date in dates_sorted[:-req.forwardPeriods]:
        cross = df[df[req.dateCol] == date]
        for sig in signal_cols:
            ic = compute_ic(cross, sig, '_fwd_ret', req.icMethod)
            ic_series[sig].append({'date': date, 'ic': ic})

    # ── Summary stats per signal ───────────────────────────
    ic_summary: List[Dict] = []
    for sig in signal_cols:
        vals = [r['ic'] for r in ic_series[sig] if r['ic'] is not None]
        if not vals:
            continue
        arr   = np.array(vals)
        ic_mean = float(np.mean(arr))
        ic_std  = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        icir    = ic_mean / ic_std if ic_std > 0 else 0.0
        t_stat  = ic_mean / (ic_std / np.sqrt(len(arr))) if ic_std > 0 else 0.0
        p_val   = float(2 * (1 - stats.t.cdf(abs(t_stat), df=len(arr) - 1)))
        pct_pos = float(np.mean(arr > 0) * 100)

        ic_summary.append({
            'signal':    sig,
            'label':     sig.replace('_score', '').replace('_', ' ').title(),
            'ic_mean':   round(ic_mean, 4),
            'ic_std':    round(ic_std, 4),
            'icir':      round(icir, 4),
            't_stat':    round(t_stat, 4),
            'p_value':   round(p_val, 4),
            'significant': bool(p_val < 0.05),
            'pct_positive': round(pct_pos, 2),
            'n_periods': len(vals),
        })

    # ── Rolling IC (window) ────────────────────────────────
    rolling_ic: Dict[str, List[Dict]] = {}
    w = req.rollingWindow
    for sig in signal_cols:
        series = ic_series[sig]
        rolled = []
        for i in range(w - 1, len(series)):
            window = [r['ic'] for r in series[i - w + 1:i + 1] if r['ic'] is not None]
            if len(window) < w // 2:
                continue
            arr = np.array(window)
            rolled.append({
                'date':    series[i]['date'],
                'ic_mean': round(float(np.mean(arr)), 4),
                'icir':    round(float(np.mean(arr) / np.std(arr, ddof=1)), 4)
                           if np.std(arr, ddof=1) > 0 else 0.0,
            })
        rolling_ic[sig] = rolled

    # ── IC histogram bins ──────────────────────────────────
    ic_histograms: Dict[str, Dict] = {}
    for sig in signal_cols:
        vals = [r['ic'] for r in ic_series[sig] if r['ic'] is not None]
        if not vals:
            continue
        counts, bin_edges = np.histogram(vals, bins=20, range=(-1, 1))
        ic_histograms[sig] = {
            'counts':    [int(c) for c in counts],
            'bin_edges': [round(float(e), 3) for e in bin_edges],
        }

    # ── Flat IC series (for chart) ─────────────────────────
    ic_chart: Dict[str, List[Dict]] = {s: ic_series[s] for s in signal_cols}

    ic_quality = ic_quality_interpretation(ic_summary)

    return {
        'signal_cols':   signal_cols,
        'ic_method':     req.icMethod,
        'forward_periods': req.forwardPeriods,
        'n_periods':     T,
        'ic_summary':    ic_summary,
        'ic_quality':    ic_quality,
        'ic_series':     ic_chart,
        'rolling_ic':    rolling_ic,
        'ic_histograms': ic_histograms,
    }


@router.post("/factor-ic")
async def factor_ic_endpoint(request: FactorICRequest):
    try:
        df = _prepare_panel(request)
        result = compute_factor_ic(df, request)
        return _to_native({'results': result})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════
# 3. FACTOR DECAY
# ══════════════════════════════════════════════════════════════

def compute_factor_decay(df: pd.DataFrame, req: FactorDecayRequest) -> Dict:
    """
    IC Decay: compute IC(signal → return at lag h) for h = 1 … maxLag.
    Shows how predictive power decays over the holding horizon.
    """
    signal_cols = req.signalCols or [c for c in DEFAULT_SIGNALS if c in df.columns]
    df = df.sort_values([req.tickerCol, req.dateCol]).copy()

    decay_curves: Dict[str, List[Dict]] = {}

    for sig in signal_cols:
        curve = []
        for lag in range(1, req.maxLag + 1):
            fwd_col = f'_fwd_{lag}'
            df[fwd_col] = df.groupby(req.tickerCol)[req.returnCol].shift(-lag)
            dates = sorted(df[req.dateCol].unique())[:-lag]
            ics = []
            for date in dates:
                cross = df[df[req.dateCol] == date]
                ic = compute_ic(cross, sig, fwd_col, req.icMethod)
                if ic is not None:
                    ics.append(ic)
            if ics:
                arr = np.array(ics)
                ic_mean = float(np.mean(arr))
                ic_std  = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
                t_stat  = ic_mean / (ic_std / np.sqrt(len(arr))) if ic_std > 0 else 0.0
                curve.append({
                    'lag':        lag,
                    'ic_mean':    round(ic_mean, 4),
                    'ic_std':     round(ic_std, 4),
                    'ic_ci_upper': round(ic_mean + 1.96 * ic_std / np.sqrt(max(len(arr), 1)), 4),
                    'ic_ci_lower': round(ic_mean - 1.96 * ic_std / np.sqrt(max(len(arr), 1)), 4),
                    't_stat':     round(t_stat, 4),
                    'significant': bool(abs(t_stat) > 1.96),
                    'n':          len(ics),
                })
        decay_curves[sig] = curve
        # 누적 컬럼 제거 — 메모리 절약
        drop_cols = [f'_fwd_{l}' for l in range(1, req.maxLag + 1) if f'_fwd_{l}' in df.columns]
        if drop_cols:
            df.drop(columns=drop_cols, inplace=True)

    # ── Half-life estimation ───────────────────────────────
    # Find the lag where |IC| drops below 50% of lag-1 IC
    half_lives: List[Dict] = []
    for sig, curve in decay_curves.items():
        if not curve:
            continue
        ic0 = abs(curve[0]['ic_mean']) if curve[0]['ic_mean'] != 0 else 0
        half_life = None
        for pt in curve[1:]:
            if ic0 > 0 and abs(pt['ic_mean']) <= ic0 * 0.5:
                half_life = pt['lag']
                break
        # Fit exponential decay: IC(h) ≈ IC(1) * exp(-λh)
        ic_vals = np.array([abs(p['ic_mean']) for p in curve])
        lags    = np.arange(1, len(ic_vals) + 1)
        decay_rate = None
        try:
            if ic_vals[0] > 0:
                log_ic = np.log(ic_vals / ic_vals[0] + 1e-10)
                slope, _ = np.polyfit(lags, log_ic, 1)
                decay_rate = round(float(-slope), 4)
                if half_life is None and decay_rate > 0:
                    half_life = round(float(np.log(2) / decay_rate), 1)
        except Exception:
            pass
        half_lives.append({
            'signal':     sig,
            'label':      sig.replace('_score', '').replace('_', ' ').title(),
            'half_life':  half_life,
            'decay_rate': decay_rate,
            'ic_lag1':    round(curve[0]['ic_mean'], 4),
            'ic_lag3':    round(curve[2]['ic_mean'], 4) if len(curve) > 2 else None,
            'ic_lag6':    round(curve[5]['ic_mean'], 4) if len(curve) > 5 else None,
            'ic_lag12':   round(curve[11]['ic_mean'], 4) if len(curve) > 11 else None,
        })

    # ── Combined decay chart (all signals at each lag) ────
    all_lags = list(range(1, req.maxLag + 1))
    combined_chart = []
    for lag in all_lags:
        row: Dict = {'lag': lag}
        for sig, curve in decay_curves.items():
            pt = next((p for p in curve if p['lag'] == lag), None)
            row[sig] = pt['ic_mean'] if pt else None
        combined_chart.append(row)

    return {
        'signal_cols':    signal_cols,
        'max_lag':        req.maxLag,
        'ic_method':      req.icMethod,
        'decay_curves':   decay_curves,
        'half_lives':     half_lives,
        'combined_chart': combined_chart,
    }


@router.post("/factor-decay")
async def factor_decay_endpoint(request: FactorDecayRequest):
    try:
        df = _prepare_panel(request)
        result = compute_factor_decay(df, request)
        return _to_native({'results': result})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════
# 4. FACTOR ROTATION
# ══════════════════════════════════════════════════════════════

def compute_factor_rotation(df: pd.DataFrame, req: FactorRotationRequest) -> Dict:
    """
    Rolling factor performance + dominance:
    - Per-period factor return (proxy: correlation with Mkt_RF excess return)
    - Rolling Sharpe per factor
    - Factor regime classification (risk-on / risk-off)
    - Rotation signal: which factor is currently "winning"
    """
    factor_cols = req.factorCols or [f for f in ALL_FACTORS if f in df.columns]
    dates_sorted = sorted(df[req.dateCol].unique())
    T = len(dates_sorted)
    w = req.rollingWindow

    # Average factor return per period (mean across stocks, but factors
    # are common so just take one stock's factor series)
    # Factor returns are the same across stocks — pick first stock
    first_ticker = df[req.tickerCol].iloc[0]
    factor_ts = df[df[req.tickerCol] == first_ticker].sort_values(req.dateCol)
    factor_ts = factor_ts.set_index(req.dateCol)[factor_cols]

    # ── Period-by-period factor returns ───────────────────
    period_returns: List[Dict] = []
    for date in dates_sorted:
        row: Dict = {'date': date}
        if date in factor_ts.index:
            for f in factor_cols:
                row[f] = safe_float(factor_ts.loc[date, f])
        period_returns.append(row)

    # ── Rolling Sharpe per factor ──────────────────────────
    rolling_sharpe: List[Dict] = []
    factor_arr = factor_ts.reindex(dates_sorted).values  # (T, K)

    for end_t in range(w - 1, T):
        window = factor_arr[end_t - w + 1:end_t + 1]  # (w, K)
        row = {'date': dates_sorted[end_t]}
        for j, f in enumerate(factor_cols):
            col = window[:, j].astype(float)
            col = col[~np.isnan(col)]
            if len(col) < 3:
                row[f'sharpe_{f}'] = None
                continue
            mu  = float(np.mean(col))
            sig = float(np.std(col, ddof=1))
            row[f'sharpe_{f}'] = round(mu / sig * np.sqrt(12), 4) if sig > 0 else 0.0
        rolling_sharpe.append(row)

    # ── Dominant factor per period ─────────────────────────
    # Factor with highest rolling Sharpe = "leading" factor
    dominance: List[Dict] = []
    for row in rolling_sharpe:
        sharpes = {f: row.get(f'sharpe_{f}') for f in factor_cols
                   if row.get(f'sharpe_{f}') is not None}
        if not sharpes:
            continue
        top = sorted(sharpes.items(), key=lambda x: x[1], reverse=True)[:req.topN]
        bottom = sorted(sharpes.items(), key=lambda x: x[1])[:req.topN]
        dominance.append({
            'date':         row['date'],
            'top_factors':  [t[0] for t in top],
            'top_sharpes':  [round(t[1], 3) for t in top],
            'bottom_factors': [t[0] for t in bottom],
        })

    # ── Cumulative factor returns ──────────────────────────
    cum_factor: Dict[str, List[Dict]] = {}
    for f in factor_cols:
        ret_series = factor_ts[f].reindex(dates_sorted).fillna(0).values / 100.0
        cum = np.cumprod(1 + ret_series) * 100
        cum_factor[f] = [{'date': d, 'cum': round(float(c), 4)}
                         for d, c in zip(dates_sorted, cum)]

    # ── Factor correlation over full period ───────────────
    corr_df   = factor_ts.reindex(dates_sorted).dropna()
    corr_matrix = {}
    if len(corr_df) > 3:
        corr = corr_df.corr()
        corr_matrix = {
            col: {c: safe_float(corr.loc[col, c]) for c in corr.columns}
            for col in corr.columns
        }

    # ── Factor regime (risk-on / risk-off by Mkt_RF) ──────
    regime_series: List[Dict] = []
    if 'Mkt_RF' in factor_cols:
        mkt = factor_ts['Mkt_RF'].reindex(dates_sorted)
        trailing_mean = mkt.rolling(window=6, min_periods=3).mean()
        for d in dates_sorted:
            val = safe_float(trailing_mean.get(d, np.nan))
            regime_series.append({
                'date':   d,
                'regime': 'risk_on' if val > 0 else 'risk_off',
                'mkt_trailing': round(val, 4),
            })

    # ── Summary statistics per factor ─────────────────────
    factor_summary: List[Dict] = []
    for f in factor_cols:
        vals = factor_ts[f].dropna().values / 100.0
        if len(vals) < 3:
            continue
        ann_ret = float(np.mean(vals)) * 12
        ann_vol = float(np.std(vals, ddof=1)) * np.sqrt(12)
        sharpe  = ann_ret / ann_vol if ann_vol > 0 else 0.0
        cum_ret = float(np.prod(1 + vals) - 1)
        factor_summary.append({
            'factor':       f,
            'label':        FACTOR_LABELS.get(f, f),
            'ann_return':   round(ann_ret * 100, 4),
            'ann_vol':      round(ann_vol * 100, 4),
            'sharpe':       round(sharpe, 4),
            'cum_return':   round(cum_ret * 100, 4),
            'max_monthly':  round(float(vals.max()) * 100, 4),
            'min_monthly':  round(float(vals.min()) * 100, 4),
        })

    return {
        'factor_cols':     factor_cols,
        'n_periods':       T,
        'rolling_window':  w,
        'period_returns':  period_returns,
        'rolling_sharpe':  rolling_sharpe,
        'dominance':       dominance,
        'cum_factor':      cum_factor,
        'corr_matrix':     corr_matrix,
        'regime_series':   regime_series,
        'factor_summary':  factor_summary,
    }


@router.post("/factor-rotation")
async def factor_rotation_endpoint(request: FactorRotationRequest):
    try:
        df = _prepare_panel(request)
        result = compute_factor_rotation(df, request)
        return _to_native({'results': result})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════
# 5. LONG-SHORT PORTFOLIO
# ══════════════════════════════════════════════════════════════

def compute_long_short(df: pd.DataFrame, req: LongShortRequest) -> Dict:
    """
    Quantile-based long-short portfolio:
    - Sort stocks into N quantiles by signal each period
    - Long top quantile, Short bottom quantile
    - Compute portfolio returns, Sharpe, max drawdown, turnover
    """
    dates_sorted = sorted(df[req.dateCol].unique())
    T = len(dates_sorted)

    sig_col  = req.signalCol
    ret_col  = req.returnCol
    ticker_col = req.tickerCol
    Q = req.nQuantiles
    tc = req.transactionCost

    if sig_col not in df.columns:
        raise HTTPException(400, f"Signal column '{sig_col}' not found. Available: {list(df.columns)}")

    # Auto-adjust Q if not enough stocks per period
    n_stocks = df[ticker_col].nunique()
    Q = min(Q, max(2, n_stocks // 2))

    # ── Build forward returns (t+1) for each stock ────────
    df = df.sort_values([ticker_col, req.dateCol]).copy()
    df['_fwd'] = df.groupby(ticker_col)[ret_col].shift(-1)

    # ── Per-period quantile sort & portfolio return ────────
    quantile_rets: List[Dict] = []
    long_weights_prev:  Dict[str, float] = {}
    short_weights_prev: Dict[str, float] = {}

    for date in dates_sorted[:-1]:
        cross = df[df[req.dateCol] == date][[ticker_col, sig_col, '_fwd', req.marketCapCol or 'market_cap']].dropna()
        if len(cross) < Q * 2:
            continue

        cross = cross.copy()
        cross['_q'] = pd.qcut(cross[sig_col], Q, labels=False, duplicates='drop')

        long_stocks  = cross[cross['_q'] == cross['_q'].max()]
        short_stocks = cross[cross['_q'] == cross['_q'].min()]

        if req.weightScheme == 'value_weighted' and req.marketCapCol in cross.columns:
            def vw_ret(grp):
                w = grp[req.marketCapCol] / grp[req.marketCapCol].sum()
                return float((w * grp['_fwd']).sum())
            long_ret  = vw_ret(long_stocks)
            short_ret = vw_ret(short_stocks)
        else:
            long_ret  = float(long_stocks['_fwd'].mean())
            short_ret = float(short_stocks['_fwd'].mean())

        # Turnover cost
        long_now  = set(long_stocks[ticker_col].tolist())
        short_now = set(short_stocks[ticker_col].tolist())
        long_turnover  = len(long_now  - set(long_weights_prev.keys()))  / max(len(long_now),  1)
        short_turnover = len(short_now - set(short_weights_prev.keys())) / max(len(short_now), 1)
        cost = (long_turnover + short_turnover) * tc * 100

        ls_ret = long_ret - short_ret - cost

        row: Dict = {
            'date':          date,
            'long_return':   safe_float(long_ret),
            'short_return':  safe_float(short_ret),
            'ls_return':     safe_float(ls_ret),
            'spread':        safe_float(long_ret - short_ret),
            'n_long':        len(long_stocks),
            'n_short':       len(short_stocks),
            'turnover_cost': safe_float(cost),
        }
        # All quantile returns
        for q in range(Q):
            grp = cross[cross['_q'] == q]
            row[f'q{q+1}_return'] = safe_float(float(grp['_fwd'].mean())) if len(grp) > 0 else None

        quantile_rets.append(row)
        long_weights_prev  = {t: 1 / len(long_now)  for t in long_now}
        short_weights_prev = {t: 1 / len(short_now) for t in short_now}

    if not quantile_rets:
        raise HTTPException(400, "Insufficient data for long-short portfolio construction.")

    # ── Performance metrics ────────────────────────────────
    def perf(ret_series: List[float]) -> Dict:
        arr = np.array(ret_series) / 100.0
        mu  = float(np.mean(arr))
        sig = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
        sharpe  = mu / sig * np.sqrt(12) if sig > 0 else 0.0
        cum     = np.cumprod(1 + arr)
        roll_max = np.maximum.accumulate(cum)
        dd      = (cum - roll_max) / roll_max
        max_dd  = float(dd.min())
        ann_ret = mu * 12
        ann_vol = sig * np.sqrt(12)
        # safe_float 적용 — NaN/Inf 모두 0.0으로 치환 (FastAPI JSON 직렬화 실패 방지)
        return {
            'ann_return':  safe_float(ann_ret * 100),
            'ann_vol':     safe_float(ann_vol * 100),
            'sharpe':      safe_float(sharpe),
            'max_drawdown':safe_float(max_dd * 100),
            'hit_rate':    safe_float(float(np.mean(arr > 0) * 100)),
            'avg_monthly': safe_float(mu * 100),
            'n_periods':   len(arr),
        }

    ls_rets   = [r['ls_return']    for r in quantile_rets]
    long_rets = [r['long_return']  for r in quantile_rets]
    short_rets= [r['short_return'] for r in quantile_rets]

    # avg_turnover: 기간별 turnover_cost에서 역산 (tc로 나눔)
    avg_turnover_val = 0.0
    if tc > 0:
        tc_vals = [r.get('turnover_cost', 0) for r in quantile_rets if r.get('turnover_cost') is not None]
        if tc_vals:
            avg_tc_cost = float(sum(tc_vals) / len(tc_vals))
            avg_turnover_val = avg_tc_cost / (tc * 100 * 2) if tc > 0 else 0.0

    ls_perf   = perf(ls_rets)
    long_perf = perf(long_rets)
    short_perf= perf(short_rets)
    # feasibility에서 쓸 수 있도록 avg_turnover 추가
    ls_perf['avg_turnover'] = safe_float(avg_turnover_val)

    performance = {
        'long_short': ls_perf,
        'long_only':  long_perf,
        'short_only': short_perf,
    }

    # ── Cumulative return chart ────────────────────────────
    cum_chart: List[Dict] = []
    cum_ls = cum_long = cum_short = 100.0
    for r in quantile_rets:
        cum_ls    *= (1 + r['ls_return']    / 100)
        cum_long  *= (1 + r['long_return']  / 100)
        cum_short *= (1 + r['short_return'] / 100)
        cum_chart.append({
            'date':       r['date'],
            'long_short': safe_float(cum_ls),
            'long_only':  safe_float(cum_long),
            'short_only': safe_float(cum_short),
        })

    # ── Quantile return bar (average per quantile) ────────
    quantile_avg: List[Dict] = []
    for q in range(Q):
        col = f'q{q+1}_return'
        vals = [r[col] for r in quantile_rets if r.get(col) is not None]
        quantile_avg.append({
            'quantile': q + 1,
            'label':    f'Q{q+1}' + (' (Short)' if q == 0 else ' (Long)' if q == Q-1 else ''),
            'avg_return': safe_float(float(np.mean(vals))) if vals else 0.0,
        })

    # ── Monthly return distribution ────────────────────────
    ls_rets_clean = [r for r in ls_rets if r is not None and not (r != r)]
    if ls_rets_clean and len(set(ls_rets_clean)) > 1:
        counts, edges = np.histogram(ls_rets_clean, bins=20)
        return_dist = {
            'counts':    [int(c) for c in counts],
            'bin_edges': [round(float(e), 3) for e in edges],
        }
    else:
        return_dist = {'counts': [], 'bin_edges': []}

    # ── Drawdown series ────────────────────────────────────
    cum_arr = np.array([r['long_short'] for r in cum_chart])
    roll_max = np.maximum.accumulate(cum_arr)
    dd_arr   = (cum_arr - roll_max) / roll_max * 100
    drawdown_series = [
        {'date': r['date'], 'drawdown': safe_float(float(dd_arr[i]))}
        for i, r in enumerate(cum_chart)
    ]

    feasibility = long_short_feasibility(
        performance['long_short'], quantile_avg, tc, Q, sig_col
    )

    return {
        'signal_col':     sig_col,
        'n_quantiles':    Q,
        'weight_scheme':  req.weightScheme,
        'n_periods':      len(quantile_rets),
        'performance':    performance,
        'feasibility':    feasibility,
        'quantile_rets':  quantile_rets,
        'quantile_avg':   quantile_avg,
        'cum_chart':      cum_chart,
        'drawdown_series':drawdown_series,
        'return_dist':    return_dist,
    }


@router.post("/long-short-portfolio")
async def long_short_endpoint(request: LongShortRequest):
    try:
        df = _prepare_panel(request)
        result = compute_long_short(df, request)
        return _to_native({'results': result})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")




# ══════════════════════════════════════════════════════════════
# Factor IC Quality Interpretation
# ══════════════════════════════════════════════════════════════

def ic_quality_interpretation(ic_summary: List[Dict]) -> List[Dict]:
    """
    Auto-generate signal quality narrative for each factor IC result.

    Grades:
      Excellent : |ICIR| >= 0.5  and  |IC mean| >= 0.05  and  pct_pos >= 55%
      Good      : |ICIR| >= 0.3  and  |IC mean| >= 0.03
      Moderate  : |ICIR| >= 0.15 and significant
      Weak      : significant but low ICIR
      Noise     : not significant
    """
    results = []
    for s in ic_summary:
        ic_mean   = s.get('ic_mean', 0.0)
        icir      = s.get('icir', 0.0)
        sig       = s.get('significant', False)
        pct_pos   = s.get('pct_positive', 50.0)
        t_stat    = s.get('t_stat', 0.0)
        n_periods = s.get('n_periods', 0)

        abs_ic   = abs(ic_mean)
        abs_icir = abs(icir)

        if abs_icir >= 0.5 and abs_ic >= 0.05 and pct_pos >= 55:
            grade = 'Excellent'
            color = 'emerald'
        elif abs_icir >= 0.3 and abs_ic >= 0.03 and sig:
            grade = 'Good'
            color = 'blue'
        elif abs_icir >= 0.15 and sig:
            grade = 'Moderate'
            color = 'amber'
        elif sig:
            grade = 'Weak'
            color = 'orange'
        else:
            grade = 'Noise'
            color = 'slate'

        direction = 'positive' if ic_mean >= 0 else 'negative (contrarian)'

        # Consistency narrative
        if pct_pos >= 65:
            consistency = f"highly consistent ({pct_pos:.0f}% positive periods)"
        elif pct_pos >= 55:
            consistency = f"mostly consistent ({pct_pos:.0f}% positive)"
        elif pct_pos >= 45:
            consistency = f"mixed direction ({pct_pos:.0f}% positive)"
        else:
            consistency = f"mostly negative ({pct_pos:.0f}% positive — contrarian signal)"

        # Build narrative
        if grade == 'Noise':
            narrative = (
                f"IC mean = {ic_mean:.4f}, ICIR = {icir:.3f} — not statistically significant "
                f"(t={t_stat:.2f}, n={n_periods}). This signal shows no reliable predictive "
                f"power over the sample period. Consider dropping or combining with other signals."
            )
        else:
            narrative = (
                f"IC mean = {ic_mean:.4f} ({direction}), ICIR = {icir:.3f}, "
                f"t-stat = {t_stat:.2f} over {n_periods} periods — {consistency}. "
            )
            if grade == 'Excellent':
                narrative += (
                    f"ICIR above 0.5 is institutional-grade predictive power. "
                    f"This signal is a strong candidate for portfolio construction."
                )
            elif grade == 'Good':
                narrative += (
                    f"Solid predictive power. Suitable as a primary factor with appropriate "
                    f"position sizing. Combine with decay analysis to determine optimal rebalance frequency."
                )
            elif grade == 'Moderate':
                narrative += (
                    f"Moderate but reliable signal. Best used in combination with higher-ICIR "
                    f"factors. Monitor for regime-dependent performance."
                )
            else:
                narrative += (
                    f"Statistical significance is marginal. Use with caution — the signal may "
                    f"be regime-specific or require refinement."
                )

        results.append({
            'signal':    s['signal'],
            'label':     s.get('label', s['signal']),
            'grade':     grade,
            'color':     color,
            'ic_mean':   ic_mean,
            'icir':      icir,
            'pct_pos':   pct_pos,
            'narrative': narrative,
        })

    # Rank by |ICIR|
    results.sort(key=lambda x: abs(x['icir']), reverse=True)
    return results


# ══════════════════════════════════════════════════════════════
# Long-Short Execution Feasibility
# ══════════════════════════════════════════════════════════════

def long_short_feasibility(
    performance: Dict,
    quantile_avg: List[Dict],
    tc: float,
    n_quantiles: int,
    signal_col: str,
) -> Dict:
    """
    Assess whether the long-short strategy is executable after costs.

    Metrics:
      gross_sharpe     : Sharpe before transaction costs
      net_sharpe       : Sharpe after costs (from performance dict)
      cost_drag_pct    : % of gross return consumed by costs
      spread_pct       : Q5–Q1 avg monthly return spread
      breakeven_tc     : transaction cost at which net Sharpe = 0
      recommendation   : Executable / Marginal / Not Executable
    """
    # factor_analysis perf() 키: ann_return, ann_vol, sharpe, avg_monthly, n_periods
    # portfolio_optimization 키: annualised_return, annualised_vol, sharpe
    # 둘 다 지원하도록 다중 fallback
    gross_ret  = safe_float(performance.get('ann_return',
                  performance.get('gross_return_ann',
                  performance.get('annualised_return',
                  performance.get('return_ann', 0)))))
    net_ret    = safe_float(performance.get('ann_return',
                  performance.get('net_return_ann',
                  performance.get('annualised_return',
                  performance.get('return_ann', 0)))))
    net_sharpe = safe_float(performance.get('sharpe',
                  performance.get('net_sharpe', 0)))
    ann_vol    = safe_float(performance.get('ann_vol',
                  performance.get('annualised_vol',
                  performance.get('vol_ann', 1))))
    turnover   = safe_float(performance.get('avg_turnover', 0))

    # Q spread
    top_q    = next((r for r in quantile_avg if r.get('quantile') == n_quantiles), None)
    bot_q    = next((r for r in quantile_avg if r.get('quantile') == 1), None)
    spread   = None
    if top_q and bot_q:
        spread = safe_float(top_q.get('avg_return', 0)) - safe_float(bot_q.get('avg_return', 0))

    # Cost drag
    annual_cost = turnover * tc * 12 * 2  # round-trip × 12 months × 2 (long + short)
    cost_drag   = (annual_cost / gross_ret * 100) if gross_ret > 0 else None

    # Breakeven tc: net_ret = gross_ret - turnover * tc_be * 24 = 0
    breakeven_tc = gross_ret / (turnover * 24) if turnover > 0 else None

    # Recommendation
    if net_sharpe >= 0.5:
        rec   = 'Executable'
        color = 'emerald'
        rec_text = (
            f"Net Sharpe of {net_sharpe:.2f} after {tc*100:.2f}% one-way costs is "
            f"strong. The strategy is viable with current transaction cost assumptions."
        )
    elif net_sharpe >= 0.2:
        rec   = 'Marginal'
        color = 'amber'
        rec_text = (
            f"Net Sharpe of {net_sharpe:.2f} is marginal. Consider reducing rebalance "
            f"frequency or using a larger signal threshold to cut turnover."
        )
    else:
        rec   = 'Not Executable'
        color = 'red'
        rec_text = (
            f"Net Sharpe of {net_sharpe:.2f} falls below the minimum threshold after costs. "
            f"{'Breakeven tc = ' + f'{breakeven_tc*100:.3f}%' if breakeven_tc else 'High turnover'} "
            f"suggests the signal needs stronger predictive power or lower costs to be viable."
        )

    return {
        'signal_col':     signal_col,
        'gross_return':   safe_float(gross_ret),
        'net_return':     safe_float(net_ret),
        'net_sharpe':     safe_float(net_sharpe),
        'ann_vol':        safe_float(ann_vol),
        'annual_cost_pct':safe_float(annual_cost * 100),
        'cost_drag_pct':  safe_float(cost_drag) if cost_drag else None,
        'avg_turnover':   safe_float(turnover),
        'tc_one_way':     tc,
        'q_spread_monthly': safe_float(spread) if spread else None,
        'breakeven_tc':   safe_float(breakeven_tc) if breakeven_tc else None,
        'recommendation': rec,
        'color':          color,
        'narrative':      rec_text,
    }


# ══════════════════════════════════════════════════════════════
# Fama-MacBeth Risk Premium Interpretation
# ══════════════════════════════════════════════════════════════

def fm_risk_premium_interpretation(fm_results: List[Dict], avg_r2: float) -> Dict:
    """
    Interpret Fama-MacBeth lambda estimates as risk premia.

    For each signal:
      - Is lambda significantly positive/negative?
      - Economic magnitude: λ × 1σ signal → return implication
      - Rank signals by |t-stat|
      - Overall model adequacy from avg R²
    """
    if not fm_results:
        return {}

    interpretations = []
    for r in fm_results:
        lam    = r.get('lambda_mean', 0.0)
        t_stat = r.get('t_stat', 0.0)
        p_val  = r.get('p_value', 1.0)
        sig    = r.get('significant', False)
        pct_pos= r.get('pct_positive', 50.0)
        label  = r.get('label', r.get('signal', ''))

        # Direction
        if sig and lam > 0:
            direction = 'positive premium'
            color = 'emerald'
        elif sig and lam < 0:
            direction = 'negative premium (contrarian)'
            color = 'red'
        else:
            direction = 'no significant premium'
            color = 'slate'

        # Magnitude interpretation
        mag = abs(lam)
        if mag >= 0.5:
            mag_label = 'large'
        elif mag >= 0.2:
            mag_label = 'moderate'
        elif mag >= 0.05:
            mag_label = 'small'
        else:
            mag_label = 'negligible'

        if sig:
            narrative = (
                f"{label} — {direction}. λ = {lam:.4f}%/period "
                f"(t={t_stat:.2f}, p={p_val:.3f}), positive in {pct_pos:.0f}% of periods. "
                f"A 1σ increase in {label} is associated with a {mag_label} shift in expected "
                f"returns. This is a {'reliable' if abs(t_stat) > 2.5 else 'borderline'} "
                f"cross-sectional risk factor."
            )
        else:
            narrative = (
                f"{label} — no significant cross-sectional premium (λ={lam:.4f}, "
                f"t={t_stat:.2f}, p={p_val:.3f}). "
                f"The signal does not reliably command a return premium in this sample."
            )

        interpretations.append({
            'signal':    r.get('signal'),
            'label':     label,
            'direction': direction,
            'color':     color,
            'lambda':    safe_float(lam),
            't_stat':    safe_float(t_stat),
            'significant': sig,
            'narrative': narrative,
        })

    # Sort by |t-stat|
    interpretations.sort(key=lambda x: abs(x['t_stat']), reverse=True)

    # Model adequacy
    n_sig = sum(1 for r in fm_results if r.get('significant'))
    n_total = len(fm_results)
    if avg_r2 >= 0.10:
        model_quality = 'Good'
        model_narrative = (
            f"Average cross-sectional R² = {avg_r2:.3f} — the signal set explains a meaningful "
            f"portion of cross-sectional return variation. "
        )
    elif avg_r2 >= 0.03:
        model_quality = 'Moderate'
        model_narrative = (
            f"Average cross-sectional R² = {avg_r2:.3f} — typical for monthly return regressions. "
        )
    else:
        model_quality = 'Low'
        model_narrative = (
            f"Average cross-sectional R² = {avg_r2:.3f} — low explanatory power. "
            f"Consider adding more/stronger signals or controlling for sector effects. "
        )
    model_narrative += (
        f"{n_sig} of {n_total} signals are statistically significant at 5%."
    )

    return {
        'interpretations':  interpretations,
        'model_quality':    model_quality,
        'model_narrative':  model_narrative,
        'n_significant':    n_sig,
        'n_total':          n_total,
        'avg_r2':           safe_float(avg_r2),
    }


# ══════════════════════════════════════════════════════════════
# 6. CROSS-SECTIONAL REGRESSION  (Fama-MacBeth)
# ══════════════════════════════════════════════════════════════

def compute_fama_macbeth(df: pd.DataFrame, req: CrossSectionalRequest) -> Dict:
    """
    Fama-MacBeth (1973) two-pass procedure:
    Step 1: Cross-sectional OLS each period → λ_t (price of risk)
    Step 2: Time-series mean/t-stat of λ_t with Newey-West SE
    """
    signal_cols = req.signalCols or [c for c in DEFAULT_SIGNALS if c in df.columns]
    controls    = req.controls   or []
    all_x_cols  = signal_cols + [c for c in controls if c not in signal_cols]

    dates_sorted = sorted(df[req.dateCol].unique())

    # Build forward returns
    df = df.sort_values([req.tickerCol, req.dateCol]).copy()
    df['_fwd'] = df.groupby(req.tickerCol)[req.returnCol].shift(-1)

    # ── Step 1: Cross-sectional regression per period ──────
    lambda_series: Dict[str, List[float]] = {col: [] for col in all_x_cols}
    lambda_dates:  List[str] = []
    r2_series:     List[float] = []
    n_series:      List[int] = []

    per_period: List[Dict] = []

    for date in dates_sorted[:-1]:
        cross = df[df[req.dateCol] == date][[req.tickerCol, '_fwd'] + all_x_cols].dropna()
        if len(cross) < req.minStocksPerPeriod:
            continue

        y = cross['_fwd'].values.astype(np.float64)
        X = cross[all_x_cols].values.astype(np.float64)
        # Standardise X cross-sectionally
        X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)
        X_const = sm.add_constant(X_std)

        try:
            res = sm.OLS(y, X_const).fit()
            for i, col in enumerate(all_x_cols):
                lambda_series[col].append(float(res.params[i + 1]))
            lambda_dates.append(date)
            r2_series.append(safe_float(res.rsquared))
            n_series.append(int(len(cross)))

            row: Dict = {'date': date, 'n_stocks': len(cross), 'r2': safe_float(res.rsquared)}
            for i, col in enumerate(all_x_cols):
                row[f'lambda_{col}'] = safe_float(res.params[i + 1])
                row[f'tstat_{col}']  = safe_float(res.tvalues[i + 1])
            per_period.append(row)
        except Exception:
            continue

    if not lambda_dates:
        raise HTTPException(400, "Insufficient cross-sectional data for Fama-MacBeth.")

    # ── Step 2: Time-series inference with Newey-West ──────
    fm_results: List[Dict] = []
    nlags = req.neweyWestLags

    for col in all_x_cols:
        lam = np.array(lambda_series[col])
        T   = len(lam)
        if T < nlags + 2:
            continue
        lam_mean = float(np.mean(lam))
        lam_std  = float(np.std(lam, ddof=1))

        # Newey-West variance of mean
        gamma0 = float(np.var(lam, ddof=1))
        nw_var = gamma0
        for lag in range(1, nlags + 1):
            gamma_l = float(np.cov(lam[lag:], lam[:-lag])[0, 1])
            nw_var += 2 * (1 - lag / (nlags + 1)) * gamma_l
        nw_se = float(np.sqrt(max(nw_var, 0) / T))
        t_stat = lam_mean / nw_se if nw_se > 0 else 0.0
        p_val  = float(2 * (1 - stats.t.cdf(abs(t_stat), df=T - 1)))

        fm_results.append({
            'signal':       col,
            'label':        col.replace('_score', '').replace('_', ' ').title(),
            'lambda_mean':  round(lam_mean, 6),
            'lambda_std':   round(lam_std, 6),
            'nw_se':        round(nw_se, 6),
            't_stat':       round(t_stat, 4),
            'p_value':      round(p_val, 4),
            'significant':  bool(p_val < 0.05),
            'n_periods':    T,
            'pct_positive': round(float(np.mean(lam > 0) * 100), 2),
        })

    # ── Lambda time series chart ───────────────────────────
    lambda_chart: List[Dict] = []
    for i, date in enumerate(lambda_dates):
        row: Dict = {'date': date}
        for col in all_x_cols:
            vals = lambda_series[col]
            row[f'lambda_{col}'] = round(vals[i], 6) if i < len(vals) else None
        row['r2']       = r2_series[i] if i < len(r2_series) else None
        row['n_stocks'] = n_series[i]  if i < len(n_series)  else None
        lambda_chart.append(row)

    # ── Average R² and N per period ────────────────────────
    avg_r2 = float(np.mean(r2_series)) if r2_series else 0.0
    avg_n  = float(np.mean(n_series))  if n_series  else 0.0

    # ── Lambda distribution per signal ────────────────────
    lambda_dists: Dict[str, Dict] = {}
    for col in all_x_cols:
        vals = lambda_series[col]
        if not vals:
            continue
        counts, edges = np.histogram(vals, bins=20)
        lambda_dists[col] = {
            'counts':    [int(c) for c in counts],
            'bin_edges': [round(float(e), 6) for e in edges],
        }

    # ── Significance summary ───────────────────────────────
    sig_count = sum(1 for r in fm_results if r['significant'])

    fm_interpretation = fm_risk_premium_interpretation(fm_results, avg_r2)

    return {
        'signal_cols':    signal_cols,
        'controls':       controls,
        'n_periods':      len(lambda_dates),
        'avg_r2':         round(avg_r2, 4),
        'avg_n_stocks':   round(avg_n, 1),
        'newey_west_lags': nlags,
        'fm_results':     fm_results,
        'fm_interpretation': fm_interpretation,
        'lambda_chart':   lambda_chart,
        'lambda_dists':   lambda_dists,
        'per_period':     per_period,
        'n_significant':  sig_count,
        'n_signals':      len(fm_results),
    }


@router.post("/cross-sectional")
async def cross_sectional_endpoint(request: CrossSectionalRequest):
    try:
        df = _prepare_panel(request)
        result = compute_fama_macbeth(df, request)
        return _to_native({'results': result})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")
