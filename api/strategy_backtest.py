from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
import warnings
import traceback

warnings.filterwarnings('ignore')

router = APIRouter()


# ══════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════

STOCK_PROFILES = {
    'AAPL':  {'name': 'Apple Inc.',         'beta': 1.20, 'smb': -0.15, 'hml': -0.45, 'mom':  0.10, 'rmw':  0.20, 'cma': -0.30, 'alpha': 0.005, 'sigma': 0.04},
    'MSFT':  {'name': 'Microsoft Corp.',    'beta': 1.10, 'smb': -0.20, 'hml': -0.30, 'mom':  0.08, 'rmw':  0.25, 'cma': -0.20, 'alpha': 0.004, 'sigma': 0.035},
    'GOOGL': {'name': 'Alphabet Inc.',      'beta': 1.05, 'smb': -0.18, 'hml': -0.40, 'mom':  0.05, 'rmw':  0.15, 'cma': -0.25, 'alpha': 0.003, 'sigma': 0.04},
    'AMZN':  {'name': 'Amazon.com Inc.',    'beta': 1.25, 'smb': -0.10, 'hml': -0.55, 'mom':  0.15, 'rmw': -0.10, 'cma': -0.35, 'alpha': 0.004, 'sigma': 0.05},
    'TSLA':  {'name': 'Tesla Inc.',         'beta': 1.80, 'smb':  0.30, 'hml': -0.70, 'mom':  0.20, 'rmw': -0.30, 'cma': -0.50, 'alpha': 0.006, 'sigma': 0.10},
    'NVDA':  {'name': 'NVIDIA Corp.',       'beta': 1.60, 'smb':  0.10, 'hml': -0.60, 'mom':  0.25, 'rmw':  0.10, 'cma': -0.45, 'alpha': 0.008, 'sigma': 0.08},
    'JPM':   {'name': 'JPMorgan Chase',     'beta': 1.15, 'smb': -0.05, 'hml':  0.60, 'mom':  0.05, 'rmw':  0.35, 'cma':  0.10, 'alpha': 0.002, 'sigma': 0.04},
    'JNJ':   {'name': 'J&J',               'beta': 0.65, 'smb': -0.25, 'hml':  0.30, 'mom': -0.05, 'rmw':  0.40, 'cma':  0.15, 'alpha': 0.001, 'sigma': 0.025},
    'XOM':   {'name': 'Exxon Mobil',        'beta': 0.90, 'smb': -0.10, 'hml':  0.70, 'mom':  0.00, 'rmw':  0.25, 'cma':  0.30, 'alpha': 0.000, 'sigma': 0.04},
    'WMT':   {'name': 'Walmart Inc.',       'beta': 0.55, 'smb': -0.30, 'hml':  0.20, 'mom': -0.02, 'rmw':  0.30, 'cma':  0.10, 'alpha': 0.001, 'sigma': 0.025},
    'META':  {'name': 'Meta Platforms',     'beta': 1.30, 'smb': -0.12, 'hml': -0.35, 'mom':  0.12, 'rmw':  0.15, 'cma': -0.25, 'alpha': 0.003, 'sigma': 0.05},
    'V':     {'name': 'Visa Inc.',          'beta': 0.95, 'smb': -0.20, 'hml': -0.15, 'mom':  0.06, 'rmw':  0.35, 'cma': -0.10, 'alpha': 0.003, 'sigma': 0.03},
    'PG':    {'name': 'Procter & Gamble',   'beta': 0.50, 'smb': -0.30, 'hml':  0.25, 'mom': -0.03, 'rmw':  0.45, 'cma':  0.15, 'alpha': 0.001, 'sigma': 0.02},
    'KO':    {'name': 'Coca-Cola Co.',      'beta': 0.55, 'smb': -0.28, 'hml':  0.15, 'mom': -0.02, 'rmw':  0.40, 'cma':  0.10, 'alpha': 0.001, 'sigma': 0.02},
    'AMD':   {'name': 'AMD Inc.',           'beta': 1.70, 'smb':  0.25, 'hml': -0.55, 'mom':  0.22, 'rmw': -0.05, 'cma': -0.40, 'alpha': 0.006, 'sigma': 0.09},
    'NFLX':  {'name': 'Netflix Inc.',       'beta': 1.35, 'smb':  0.05, 'hml': -0.50, 'mom':  0.18, 'rmw':  0.05, 'cma': -0.35, 'alpha': 0.005, 'sigma': 0.07},
    'GS':    {'name': 'Goldman Sachs',      'beta': 1.40, 'smb': -0.05, 'hml':  0.50, 'mom':  0.10, 'rmw':  0.30, 'cma':  0.05, 'alpha': 0.002, 'sigma': 0.055},
    'BA':    {'name': 'Boeing Co.',         'beta': 1.30, 'smb':  0.05, 'hml':  0.25, 'mom': -0.05, 'rmw': -0.10, 'cma':  0.15, 'alpha':-0.001, 'sigma': 0.06},
    'BRK.B': {'name': 'Berkshire Hathaway','beta': 0.85, 'smb': -0.15, 'hml':  0.40, 'mom':  0.02, 'rmw':  0.30, 'cma':  0.20, 'alpha': 0.003, 'sigma': 0.03},
    'INTC':  {'name': 'Intel Corp.',        'beta': 1.05, 'smb':  0.00, 'hml':  0.20, 'mom': -0.08, 'rmw':  0.15, 'cma':  0.10, 'alpha':-0.002, 'sigma': 0.05},
}

SIGNAL_CONFIGS = {
    'momentum':  {'lookback': 12, 'skip': 1,  'description': '12-1 month momentum'},
    'value':     {'lookback':  1, 'skip': 0,  'description': 'Book-to-market value'},
    'quality':   {'lookback':  1, 'skip': 0,  'description': 'Operating profitability'},
    'mean_rev':  {'lookback':  1, 'skip': 0,  'description': '1-month mean reversion'},
    'vol_timing':{'lookback':  3, 'skip': 0,  'description': 'Volatility-timed entry'},
    'ma_cross':  {'lookback': 12, 'skip': 0,  'description': '1m / 12m MA crossover'},
}


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


def _perf_stats(returns: np.ndarray, rf_annual: float = 0.05,
                ppy: int = 12) -> Dict:
    """Comprehensive performance statistics from return series."""
    arr = np.array(returns, dtype=float)
    arr = arr[~np.isnan(arr)]
    if len(arr) < 3:
        return {}

    mu     = float(np.mean(arr))
    sigma  = float(np.std(arr, ddof=1))
    rf_per = rf_annual / ppy

    ann_ret = mu * ppy
    ann_vol = sigma * np.sqrt(ppy)
    sharpe  = (ann_ret - rf_annual) / ann_vol if ann_vol > 0 else 0.0

    # Sortino
    down    = arr[arr < rf_per]
    sor_vol = float(np.std(down, ddof=1)) * np.sqrt(ppy) if len(down) > 1 else ann_vol
    sortino = (ann_ret - rf_annual) / sor_vol if sor_vol > 0 else 0.0

    # Drawdown
    cum      = np.cumprod(1 + arr)
    peak     = np.maximum.accumulate(cum)
    dd       = (cum - peak) / peak
    max_dd   = float(dd.min())
    calmar   = ann_ret / abs(max_dd) if max_dd < -1e-10 else np.nan

    # VaR / CVaR
    var_95  = float(np.percentile(arr * 100, 5))
    cvar_95 = float(-np.mean(arr[arr <= np.percentile(arr, 5)] * 100))

    # Hit rate
    hit_rate = float(np.mean(arr > 0) * 100)

    # Skewness / kurtosis
    skew = float(stats.skew(arr))
    kurt = float(stats.kurtosis(arr))

    # Omega ratio
    threshold = rf_per
    gains  = arr[arr > threshold] - threshold
    losses = threshold - arr[arr <= threshold]
    omega  = float(gains.sum() / losses.sum()) if losses.sum() > 0 else np.nan

    return {
        'ann_return':   round(ann_ret * 100, 4),
        'ann_vol':      round(ann_vol * 100, 4),
        'sharpe':       round(sharpe, 4),
        'sortino':      round(sortino, 4),
        'calmar':       round(calmar, 4) if not np.isnan(calmar) else None,
        'max_drawdown': round(max_dd * 100, 4),
        'var_95':       round(var_95, 4),
        'cvar_95':      round(cvar_95, 4),
        'hit_rate':     round(hit_rate, 2),
        'skewness':     round(skew, 4),
        'kurtosis':     round(kurt, 4),
        'omega_ratio':  round(omega, 4) if not np.isnan(omega) else None,
        'n_periods':    int(len(arr)),
        'avg_monthly':  round(mu * 100, 4),
    }


def _equity_curve(returns: np.ndarray, dates: List[str],
                  initial: float = 100.0) -> List[Dict]:
    arr = np.array(returns, dtype=float)
    cum = np.cumprod(1 + arr / 100) * initial
    return [{'date': d, 'equity': round(float(c), 4),
             'return': round(float(arr[i]), 4)}
            for i, (d, c) in enumerate(zip(dates, cum))]


def _drawdown_series(returns: np.ndarray, dates: List[str]) -> List[Dict]:
    arr  = np.array(returns, dtype=float) / 100
    cum  = np.cumprod(1 + arr)
    peak = np.maximum.accumulate(cum)
    dd   = (cum - peak) / peak * 100
    return [{'date': d, 'drawdown': round(float(dd[i]), 4)}
            for i, d in enumerate(dates)]


def _monthly_heatmap(returns: np.ndarray, dates: List[str]) -> List[Dict]:
    """Returns list of {year, month, return} for calendar heatmap."""
    rows = []
    for ret, date in zip(returns, dates):
        try:
            parts = str(date).split('-')
            year, month = int(parts[0]), int(parts[1])
            rows.append({'year': year, 'month': month,
                         'return': round(float(ret), 4)})
        except Exception:
            pass
    return rows


# ══════════════════════════════════════════════════════════════
# Data Generator
# ══════════════════════════════════════════════════════════════

def _generate_strategy_data(
    ticker: str = 'AAPL',
    benchmark: str = 'SPY',
    n_months: int = 120,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate monthly price + factor data for backtest.
    Returns DataFrame with: date, return, benchmark_return,
    Mkt_RF, SMB, HML, MOM, RMW, CMA, RF,
    momentum_signal, value_signal, quality_signal, vol_signal, ma_signal.
    """
    rng = np.random.default_rng(seed)

    factor_means = np.array([0.0065, 0.0020, 0.0030, 0.0060, 0.0025, 0.0020])
    factor_stds  = np.array([0.0450, 0.0300, 0.0300, 0.0400, 0.0200, 0.0180])
    corr = np.array([
        [ 1.00,  0.30, -0.25, -0.10,  0.05, -0.10],
        [ 0.30,  1.00,  0.10, -0.05, -0.35, -0.10],
        [-0.25,  0.10,  1.00, -0.35, -0.10,  0.70],
        [-0.10, -0.05, -0.35,  1.00,  0.10, -0.20],
        [ 0.05, -0.35, -0.10,  0.10,  1.00,  0.00],
        [-0.10, -0.10,  0.70, -0.20,  0.00,  1.00],
    ])
    cov = np.outer(factor_stds, factor_stds) * corr
    F   = rng.multivariate_normal(factor_means, cov, size=n_months)
    mkt_rf, smb, hml, mom, rmw, cma = [F[:, i] for i in range(6)]

    rf_base = 0.0025
    rf = np.maximum(rf_base + np.cumsum(rng.normal(0, 0.0005, n_months) * 0.1), 0.0001)
    rf = rf - rf.mean() + rf_base

    # Stock returns
    p = STOCK_PROFILES.get(ticker.upper(), {
        'beta': 1.0, 'smb': 0.0, 'hml': 0.0, 'mom': 0.0,
        'rmw': 0.0, 'cma': 0.0, 'alpha': 0.002, 'sigma': 0.04,
    })
    eps      = rng.normal(0, p['sigma'], n_months)
    stock_er = (p['alpha'] + p['beta']*mkt_rf + p['smb']*smb + p['hml']*hml
                + p['mom']*mom + p['rmw']*rmw + p['cma']*cma + eps)

    # Benchmark (pure market)
    bm_eps   = rng.normal(0, 0.02, n_months)
    bm_ret   = mkt_rf + bm_eps

    end_date = pd.Timestamp('2025-04-30')
    dates    = pd.date_range(end=end_date, periods=n_months, freq='ME')
    date_strs= dates.strftime('%Y-%m').tolist()

    # Cumulative return for signal construction
    cum_stock = np.cumprod(1 + stock_er)

    # Signals (standardised cross-sectionally; here just time-series)
    # Momentum: 12-1 return
    mom_signal = np.zeros(n_months)
    for t in range(12, n_months):
        mom_signal[t] = cum_stock[t-1] / cum_stock[max(t-12, 0)] - 1

    # Mean reversion: negative 1-month return
    mr_signal = np.zeros(n_months)
    mr_signal[1:] = -stock_er[:-1]

    # Volatility signal: inverse realized vol (3m)
    vol_signal = np.zeros(n_months)
    for t in range(3, n_months):
        v = float(np.std(stock_er[t-3:t]))
        vol_signal[t] = 1 / v if v > 0 else 0

    # MA crossover: price / 12m SMA > 1 → long
    ma_signal = np.zeros(n_months)
    for t in range(12, n_months):
        sma12 = cum_stock[t-12:t].mean()
        ma_signal[t] = 1.0 if cum_stock[t] > sma12 else -1.0

    df = pd.DataFrame({
        'date':             date_strs,
        'return':           np.round(stock_er * 100, 4),
        'excess_return':    np.round(stock_er * 100, 4),
        'benchmark_return': np.round(bm_ret * 100, 4),
        'RF':               np.round(rf * 100, 4),
        'Mkt_RF':           np.round(mkt_rf * 100, 4),
        'SMB':              np.round(smb * 100, 4),
        'HML':              np.round(hml * 100, 4),
        'MOM':              np.round(mom * 100, 4),
        'RMW':              np.round(rmw * 100, 4),
        'CMA':              np.round(cma * 100, 4),
        'momentum_signal':  np.round(mom_signal * 100, 4),
        'mean_rev_signal':  np.round(mr_signal * 100, 4),
        'vol_signal':       np.round(vol_signal, 6),
        'ma_signal':        ma_signal,
    })
    return df


# ══════════════════════════════════════════════════════════════
# Request Models
# ══════════════════════════════════════════════════════════════

class BacktestBase(BaseModel):
    data:           Optional[List[Dict[str, Any]]] = None
    dateCol:        str = 'date'
    returnCol:      str = 'return'
    benchmarkCol:   Optional[str] = 'benchmark_return'
    # Generate mode
    generate:       bool = True
    ticker:         str = 'AAPL'
    benchmark:      str = 'SPY'
    nMonths:        int = 120
    seed:           Optional[int] = None
    # Common
    riskFreeRate:   float = 0.05
    transactionCost:float = 0.001   # one-way
    initialCapital: float = 100.0


class StrategyBacktestRequest(BacktestBase):
    """
    Full strategy backtest:
    - Signal-based position sizing
    - Long / Long-Short / Market-Timed
    - Rebalancing frequency
    """
    signalCol:      Optional[str] = 'momentum_signal'
    positionType:   str = 'long_only'    # 'long_only' | 'long_short' | 'market_time'
    sizingMethod:   str = 'equal'        # 'equal' | 'vol_target' | 'signal_weighted'
    volTarget:      float = 0.10         # annualised vol target (for vol_target sizing)
    rebalance:      str = 'monthly'      # 'monthly' | 'quarterly'
    entryThreshold: float = 0.0          # signal threshold for entry
    stopLoss:       Optional[float] = None   # % stop-loss (e.g. -0.05)
    takeProfit:     Optional[float] = None   # % take-profit
    rollingWindow:  int = 12             # for rolling perf stats


class SignalBacktestRequest(BacktestBase):
    """
    Signal IC + quantile backtest across multiple signals.
    Tests multiple signals simultaneously.
    """
    signalCols:     Optional[List[str]] = None   # defaults to all available
    nQuantiles:     int = 5
    forwardPeriods: int = 1
    icMethod:       str = 'rank'   # 'rank' | 'pearson'


class WalkForwardRequest(BacktestBase):
    """
    Walk-forward optimisation + out-of-sample test.
    """
    signalCol:      Optional[str] = 'momentum_signal'
    trainPeriods:   int = 36     # in-sample window (months)
    testPeriods:    int = 12     # out-of-sample window (months)
    paramGrid:      Optional[Dict[str, List[Any]]] = None
    # Default param grid: entry thresholds
    positionType:   str = 'long_only'
    sizingMethod:   str = 'equal'
    rollingWindow:  int = 12


# ══════════════════════════════════════════════════════════════
# Core: Position & Return Engine
# ══════════════════════════════════════════════════════════════

def _apply_strategy(
    returns:   np.ndarray,    # raw asset returns (%)
    signal:    np.ndarray,    # signal values
    position_type: str,
    sizing:    str,
    entry_threshold: float,
    tc:        float,
    vol_target: float,
    stop_loss: Optional[float],
    take_profit: Optional[float],
    ppy: int = 12,
) -> np.ndarray:
    """
    Apply strategy to return series.
    Returns strategy returns array (%).
    """
    T   = len(returns)
    pos = np.zeros(T)      # position at each period (-1, 0, 1 or fractional)
    ret = np.zeros(T)
    prev_pos = 0.0
    cum_ret_from_entry = 0.0

    # Normalise signal to [-1, 1] for sizing
    sig_std = float(np.std(signal)) if np.std(signal) > 0 else 1.0

    for t in range(1, T):
        sig = float(signal[t - 1])   # use lagged signal (no look-ahead)
        r   = float(returns[t]) / 100.0

        # ── Determine raw position ──
        if position_type == 'long_only':
            raw_pos = 1.0 if sig > entry_threshold else 0.0
        elif position_type == 'long_short':
            raw_pos = np.sign(sig - entry_threshold) if abs(sig - entry_threshold) > 0 else 0.0
        else:  # market_time
            raw_pos = 1.0 if sig > entry_threshold else 0.0

        # ── Sizing ──
        if sizing == 'vol_target' and t >= 3:
            realised_vol = float(np.std(returns[max(0, t-3):t])) / 100 * np.sqrt(ppy)
            scale = vol_target / realised_vol if realised_vol > 0 else 1.0
            scale = min(scale, 2.0)   # cap leverage at 2×
            raw_pos *= scale
        elif sizing == 'signal_weighted':
            raw_pos = np.clip(sig / (2 * sig_std), -1.0, 1.0)
            if position_type == 'long_only':
                raw_pos = max(raw_pos, 0.0)

        new_pos = np.clip(raw_pos, -2.0, 2.0)

        # ── Stop-Loss / Take-Profit ──
        if prev_pos != 0:
            cum_ret_from_entry += r * prev_pos
            if stop_loss is not None and cum_ret_from_entry < stop_loss:
                new_pos = 0.0
                cum_ret_from_entry = 0.0
            elif take_profit is not None and cum_ret_from_entry > take_profit:
                new_pos = 0.0
                cum_ret_from_entry = 0.0
        else:
            cum_ret_from_entry = 0.0

        # ── Transaction cost ──
        turnover = abs(new_pos - prev_pos)
        cost     = turnover * tc

        # ── Strategy return ──
        ret[t] = new_pos * r * 100 - cost * 100
        pos[t] = new_pos
        prev_pos = new_pos

    return ret, pos


# ══════════════════════════════════════════════════════════════
# 1. STRATEGY BACKTEST
# ══════════════════════════════════════════════════════════════

@router.post("/strategy-backtest")
async def strategy_backtest(request: StrategyBacktestRequest):
    try:
        ppy = 12

        # ── Data ──
        if request.generate or not request.data:
            df = _generate_strategy_data(
                ticker=request.ticker,
                n_months=request.nMonths,
                seed=request.seed,
            )
        else:
            df = pd.DataFrame(request.data)
            for col in df.select_dtypes(include='object').columns:
                if col != request.dateCol:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=[request.returnCol])

        dates   = df[request.dateCol].astype(str).tolist()
        returns = df[request.returnCol].values.astype(float)

        # Signal
        sig_col = request.signalCol or 'momentum_signal'
        if sig_col in df.columns:
            signal = df[sig_col].values.astype(float)
        else:
            signal = np.zeros(len(returns))

        # Benchmark
        bm_col = request.benchmarkCol or 'benchmark_return'
        benchmark = df[bm_col].values.astype(float) if bm_col in df.columns else returns.copy()

        # ── Apply strategy ──
        strat_rets, positions = _apply_strategy(
            returns=returns,
            signal=signal,
            position_type=request.positionType,
            sizing=request.sizingMethod,
            entry_threshold=request.entryThreshold,
            tc=request.transactionCost,
            vol_target=request.volTarget,
            stop_loss=request.stopLoss,
            take_profit=request.takeProfit,
            ppy=ppy,
        )

        # ── Buy & Hold baseline ──
        bh_rets = returns.copy()

        # ── Performance stats ──
        strat_perf = _perf_stats(strat_rets / 100, request.riskFreeRate, ppy)
        bm_perf    = _perf_stats(benchmark / 100,  request.riskFreeRate, ppy)
        bh_perf    = _perf_stats(bh_rets / 100,    request.riskFreeRate, ppy)

        # ── Equity curves ──
        equity_strat = _equity_curve(strat_rets, dates, request.initialCapital)
        equity_bh    = _equity_curve(bh_rets,    dates, request.initialCapital)
        equity_bm    = _equity_curve(benchmark,  dates, request.initialCapital)

        # ── Drawdown series ──
        dd_strat = _drawdown_series(strat_rets / 100, dates)

        # ── Monthly heatmap ──
        heatmap = _monthly_heatmap(strat_rets, dates)

        # ── Rolling stats ──
        w = request.rollingWindow
        rolling = []
        for i in range(w, len(strat_rets) + 1):
            window = strat_rets[i-w:i] / 100
            mu   = float(np.mean(window)) * ppy
            vol  = float(np.std(window, ddof=1)) * np.sqrt(ppy)
            sr   = (mu - request.riskFreeRate) / vol if vol > 0 else 0.0
            rolling.append({
                'date':   dates[i-1],
                'sharpe': round(sr, 4),
                'return': round(mu * 100, 4),
                'vol':    round(vol * 100, 4),
            })

        # ── Benchmark relative ──
        active_rets = strat_rets - benchmark
        te = float(np.std(active_rets / 100, ddof=1)) * np.sqrt(ppy) * 100
        alpha_ann = (float(np.mean(active_rets / 100)) * ppy * 100)
        info_ratio = alpha_ann / te if te > 0 else 0.0

        # ── Combined chart ──
        combined = []
        for i, d in enumerate(dates):
            combined.append({
                'date':      d,
                'strategy':  equity_strat[i]['equity'],
                'buy_hold':  equity_bh[i]['equity'],
                'benchmark': equity_bm[i]['equity'],
                'position':  round(float(positions[i]), 3),
                'signal':    round(float(signal[i]), 4),
            })

        result = {
            'ticker':        request.ticker,
            'n_periods':     len(dates),
            'date_range':    f"{dates[0]} to {dates[-1]}",
            'signal_col':    sig_col,
            'position_type': request.positionType,
            'sizing_method': request.sizingMethod,
            'performance': {
                'strategy':    strat_perf,
                'buy_hold':    bh_perf,
                'benchmark':   bm_perf,
            },
            'relative': {
                'alpha_ann':   round(alpha_ann, 4),
                'tracking_error': round(te, 4),
                'info_ratio':  round(info_ratio, 4),
            },
            'charts': {
                'combined':    combined,
                'drawdown':    dd_strat,
                'rolling':     rolling,
                'heatmap':     heatmap,
            },
            'settings': {
                'transaction_cost': request.transactionCost,
                'vol_target':       request.volTarget,
                'entry_threshold':  request.entryThreshold,
                'stop_loss':        request.stopLoss,
                'take_profit':      request.takeProfit,
            },
        }

        return _to_native({'results': result})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════
# 2. SIGNAL BACKTEST
# ══════════════════════════════════════════════════════════════

@router.post("/signal-backtest")
async def signal_backtest(request: SignalBacktestRequest):
    try:
        ppy = 12

        # ── Data ──
        if request.generate or not request.data:
            df = _generate_strategy_data(
                ticker=request.ticker,
                n_months=request.nMonths,
                seed=request.seed,
            )
        else:
            df = pd.DataFrame(request.data)
            for col in df.select_dtypes(include='object').columns:
                if col != request.dateCol:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

        dates   = df[request.dateCol].astype(str).tolist()
        returns = df[request.returnCol].values.astype(float)

        # Available signal columns
        default_sigs = ['momentum_signal', 'mean_rev_signal', 'vol_signal', 'ma_signal']
        sig_cols = request.signalCols or [c for c in default_sigs if c in df.columns]

        # Build forward returns
        fwd_ret = np.roll(returns, -request.forwardPeriods)
        fwd_ret[-request.forwardPeriods:] = np.nan

        results_per_signal: List[Dict] = []
        ic_series_all: Dict[str, List] = {}

        for sig_col in sig_cols:
            if sig_col not in df.columns:
                continue
            signal = df[sig_col].values.astype(float)
            label  = sig_col.replace('_signal', '').replace('_', ' ').title()

            # ── IC per period ──
            ic_list = []
            for t in range(len(dates) - request.forwardPeriods):
                s_t = float(signal[t])
                r_t = float(fwd_ret[t])
                if np.isnan(s_t) or np.isnan(r_t):
                    ic_list.append({'date': dates[t], 'ic': None})
                    continue
                # single-obs IC = sign(signal) * sign(fwd_ret) — simplified
                ic_list.append({'date': dates[t], 'ic': round(float(s_t * r_t), 6)})

            ic_vals = np.array([x['ic'] for x in ic_list if x['ic'] is not None])
            ic_mean = float(np.mean(ic_vals)) if len(ic_vals) > 0 else 0.0
            ic_std  = float(np.std(ic_vals, ddof=1)) if len(ic_vals) > 1 else 0.0
            icir    = ic_mean / ic_std if ic_std > 0 else 0.0
            t_stat  = ic_mean / (ic_std / np.sqrt(max(len(ic_vals), 1))) if ic_std > 0 else 0.0
            p_val   = float(2 * (1 - stats.t.cdf(abs(t_stat), df=max(len(ic_vals)-1, 1))))
            pct_pos = float(np.mean(ic_vals > 0) * 100) if len(ic_vals) > 0 else 50.0

            ic_series_all[sig_col] = ic_list

            # ── Quantile backtest ──
            Q = min(request.nQuantiles, 5)
            quantile_rets: Dict[int, List[float]] = {q: [] for q in range(Q)}

            for t in range(len(dates) - request.forwardPeriods):
                s_t = float(signal[t])
                r_t = float(fwd_ret[t])
                if np.isnan(s_t) or np.isnan(r_t):
                    continue
                # Assign to quantile based on signal rank vs rolling history
                hist = signal[max(0, t-23):t+1]
                hist_clean = hist[~np.isnan(hist)]
                if len(hist_clean) < Q:
                    continue
                pct = float(np.mean(hist_clean <= s_t))
                q   = min(int(pct * Q), Q - 1)
                quantile_rets[q].append(r_t)

            quantile_avg = []
            for q in range(Q):
                vals = quantile_rets[q]
                avg  = float(np.mean(vals)) if vals else 0.0
                quantile_avg.append({
                    'quantile': q + 1,
                    'label':    f'Q{q+1}' + (' (Short)' if q == 0 else ' (Long)' if q == Q-1 else ''),
                    'avg_return': round(avg, 4),
                    'n': len(vals),
                })

            # ── Long-Short portfolio from this signal ──
            ls_rets = []
            for t in range(1, len(dates)):
                s_lag = float(signal[t-1])
                r_t   = float(returns[t]) / 100
                if np.isnan(s_lag): continue
                pos = 1.0 if s_lag > 0 else -1.0
                ls_rets.append(pos * r_t)

            ls_perf = _perf_stats(np.array(ls_rets), request.riskFreeRate, ppy) if ls_rets else {}

            results_per_signal.append({
                'signal':     sig_col,
                'label':      label,
                'ic_mean':    round(ic_mean, 6),
                'ic_std':     round(ic_std, 6),
                'icir':       round(icir, 4),
                't_stat':     round(t_stat, 4),
                'p_value':    round(p_val, 4),
                'significant':bool(p_val < 0.05),
                'pct_positive':round(pct_pos, 2),
                'n_periods':  len(ic_vals),
                'quantile_avg': quantile_avg,
                'ls_performance': ls_perf,
            })

        # ── IC chart (combined) ──
        ic_chart = []
        all_dates_set = sorted({r['date'] for s in ic_series_all.values() for r in s})
        for d in all_dates_set:
            row: Dict = {'date': d}
            for sig_col in sig_cols:
                if sig_col not in ic_series_all:
                    continue
                pt = next((x for x in ic_series_all[sig_col] if x['date'] == d), None)
                row[sig_col] = pt['ic'] if pt else None
            ic_chart.append(row)

        # ── Equity curve per signal (LS) ──
        equity_per_signal: Dict[str, List[Dict]] = {}
        for sig_col in sig_cols:
            if sig_col not in df.columns:
                continue
            signal = df[sig_col].values.astype(float)
            ls_r   = []
            ls_d   = []
            for t in range(1, len(dates)):
                s_lag = float(signal[t-1])
                r_t   = float(returns[t])
                if not np.isnan(s_lag):
                    pos = 1.0 if s_lag > 0 else -1.0
                    ls_r.append(pos * r_t)
                    ls_d.append(dates[t])
            equity_per_signal[sig_col] = _equity_curve(
                np.array(ls_r), ls_d, request.initialCapital
            )

        result = {
            'ticker':       request.ticker,
            'n_periods':    len(dates),
            'date_range':   f"{dates[0]} to {dates[-1]}",
            'signal_cols':  sig_cols,
            'forward_periods': request.forwardPeriods,
            'signal_results': results_per_signal,
            'charts': {
                'ic_series':       ic_chart,
                'equity_per_signal': equity_per_signal,
            },
        }

        return _to_native({'results': result})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════
# 3. WALK-FORWARD TEST
# ══════════════════════════════════════════════════════════════

@router.post("/walk-forward")
async def walk_forward_test(request: WalkForwardRequest):
    try:
        ppy = 12

        # ── Data ──
        if request.generate or not request.data:
            df = _generate_strategy_data(
                ticker=request.ticker,
                n_months=request.nMonths,
                seed=request.seed,
            )
        else:
            df = pd.DataFrame(request.data)
            for col in df.select_dtypes(include='object').columns:
                if col != request.dateCol:
                    df[col] = pd.to_numeric(df[col], errors='coerce')

        dates   = df[request.dateCol].astype(str).tolist()
        returns = df[request.returnCol].values.astype(float)

        sig_col = request.signalCol or 'momentum_signal'
        signal  = df[sig_col].values.astype(float) if sig_col in df.columns else np.zeros(len(returns))

        bm_col    = request.benchmarkCol or 'benchmark_return'
        benchmark = df[bm_col].values.astype(float) if bm_col in df.columns else returns.copy()

        T      = len(returns)
        train  = request.trainPeriods
        test   = request.testPeriods

        if T < train + test:
            raise HTTPException(400, f"Need at least {train + test} observations.")

        # Parameter grid to optimise in-sample
        default_grid = {'entry_threshold': [-0.5, 0.0, 0.5, 1.0, 2.0]}
        param_grid   = request.paramGrid or default_grid

        # ══ Walk-forward loop ══
        wf_windows:  List[Dict] = []
        oos_rets:    List[float] = []
        oos_dates:   List[str]   = []
        is_rets_all: List[float] = []
        best_params_history: List[Dict] = []

        # Simple grid: just entry_threshold
        thresholds = param_grid.get('entry_threshold', [0.0])

        start = 0
        window_id = 0
        while start + train + test <= T:
            # In-sample
            is_idx   = slice(start, start + train)
            is_ret   = returns[is_idx]
            is_sig   = signal[is_idx]
            is_dates = dates[start: start + train]

            # Out-of-sample
            oos_idx   = slice(start + train, start + train + test)
            oos_ret   = returns[oos_idx]
            oos_sig   = signal[oos_idx]
            oos_dt    = dates[start + train: start + train + test]

            # ── Optimise on IS ──
            best_sharpe = -np.inf
            best_thresh = 0.0
            is_perf_best = {}
            for thresh in thresholds:
                sr, _ = _apply_strategy(
                    returns=is_ret, signal=is_sig,
                    position_type=request.positionType,
                    sizing=request.sizingMethod,
                    entry_threshold=thresh,
                    tc=request.transactionCost,
                    vol_target=0.10,
                    stop_loss=None, take_profit=None,
                )
                p = _perf_stats(sr / 100, request.riskFreeRate, ppy)
                sh = p.get('sharpe', -999)
                if sh is not None and sh > best_sharpe:
                    best_sharpe = sh
                    best_thresh = thresh
                    is_perf_best = p

            # ── Apply best params OOS ──
            oos_sr, _ = _apply_strategy(
                returns=oos_ret, signal=oos_sig,
                position_type=request.positionType,
                sizing=request.sizingMethod,
                entry_threshold=best_thresh,
                tc=request.transactionCost,
                vol_target=0.10,
                stop_loss=None, take_profit=None,
            )
            oos_perf = _perf_stats(oos_sr / 100, request.riskFreeRate, ppy)

            # Collect OOS returns
            oos_rets.extend(oos_sr.tolist())
            oos_dates.extend(oos_dt)
            is_rets_all.extend(is_ret.tolist())
            best_params_history.append({'window': window_id, 'entry_threshold': best_thresh})

            wf_windows.append({
                'window':       window_id,
                'is_start':     is_dates[0],
                'is_end':       is_dates[-1],
                'oos_start':    oos_dt[0],
                'oos_end':      oos_dt[-1],
                'best_threshold': best_thresh,
                'is_sharpe':    safe_float(is_perf_best.get('sharpe')),
                'oos_sharpe':   safe_float(oos_perf.get('sharpe')),
                'is_return':    safe_float(is_perf_best.get('ann_return')),
                'oos_return':   safe_float(oos_perf.get('ann_return')),
                'is_maxdd':     safe_float(is_perf_best.get('max_drawdown')),
                'oos_maxdd':    safe_float(oos_perf.get('max_drawdown')),
            })

            start += test   # roll forward by test period
            window_id += 1

        if not oos_rets:
            raise HTTPException(400, "No walk-forward windows generated.")

        # ── Aggregate OOS performance ──
        oos_arr  = np.array(oos_rets) / 100
        oos_perf = _perf_stats(oos_arr, request.riskFreeRate, ppy)

        # IS performance (full sample, using median best threshold)
        med_thresh = float(np.median([w['best_threshold'] for w in wf_windows]))
        is_full, _ = _apply_strategy(
            returns=returns, signal=signal,
            position_type=request.positionType,
            sizing=request.sizingMethod,
            entry_threshold=med_thresh,
            tc=request.transactionCost,
            vol_target=0.10,
            stop_loss=None, take_profit=None,
        )
        is_perf = _perf_stats(is_full / 100, request.riskFreeRate, ppy)

        # ── OOS equity curve ──
        oos_equity = _equity_curve(np.array(oos_rets), oos_dates, request.initialCapital)
        oos_dd     = _drawdown_series(oos_arr, oos_dates)
        bm_oos_ret = benchmark[len(returns)-len(oos_rets):]
        bm_equity  = _equity_curve(bm_oos_ret, oos_dates, request.initialCapital)

        # ── IS vs OOS Sharpe scatter (for overfitting diagnosis) ──
        is_oos_scatter = [
            {'window': w['window'], 'is_sharpe': w['is_sharpe'], 'oos_sharpe': w['oos_sharpe']}
            for w in wf_windows
        ]
        # Degradation ratio: OOS Sharpe / IS Sharpe
        valid = [(w['is_sharpe'], w['oos_sharpe']) for w in wf_windows
                 if w['is_sharpe'] and w['oos_sharpe'] and w['is_sharpe'] != 0]
        degradation = float(np.mean([o / i for i, o in valid])) if valid else None

        # ── Heatmap ──
        heatmap = _monthly_heatmap(np.array(oos_rets), oos_dates)

        result = {
            'ticker':        request.ticker,
            'n_windows':     len(wf_windows),
            'train_periods': train,
            'test_periods':  test,
            'date_range':    f"{dates[0]} to {dates[-1]}",
            'signal_col':    sig_col,
            'performance': {
                'oos':       oos_perf,
                'is_full':   is_perf,
            },
            'degradation_ratio': round(degradation, 4) if degradation else None,
            'median_threshold':  round(med_thresh, 4),
            'wf_windows':        wf_windows,
            'best_params_history': best_params_history,
            'charts': {
                'oos_equity':      oos_equity,
                'bm_equity':       bm_equity,
                'oos_drawdown':    oos_dd,
                'is_oos_scatter':  is_oos_scatter,
                'heatmap':         heatmap,
            },
        }

        return _to_native({'results': result})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")
