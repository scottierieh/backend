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

class SpreadRequest(BaseModel):
    data: Optional[List[Dict[str, Any]]] = None
    generate: bool = False
    ticker: str = 'AAPL'
    nDays: int = 750
    seed: Optional[int] = None
    rollingWindow: int = 21
    # Column mapping
    dateCol: Optional[str] = None
    highCol: Optional[str] = None
    lowCol: Optional[str] = None
    closeCol: Optional[str] = None
    volumeCol: Optional[str] = None


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
# Asset Profiles
# ══════════════════════════════════════════════════════════════

ASSET_PROFILES = {
    'AAPL':  {'name': 'Apple Inc.',         'base': 185, 'drift': 0.0008, 'vol': 0.018, 'spread_bp': 2.0,  'avg_vol': 65_000_000},
    'MSFT':  {'name': 'Microsoft Corp.',    'base': 420, 'drift': 0.0007, 'vol': 0.016, 'spread_bp': 1.8,  'avg_vol': 25_000_000},
    'GOOGL': {'name': 'Alphabet Inc.',      'base': 170, 'drift': 0.0006, 'vol': 0.019, 'spread_bp': 2.5,  'avg_vol': 22_000_000},
    'AMZN':  {'name': 'Amazon.com Inc.',    'base': 185, 'drift': 0.0007, 'vol': 0.020, 'spread_bp': 2.2,  'avg_vol': 45_000_000},
    'TSLA':  {'name': 'Tesla Inc.',         'base': 245, 'drift': 0.001,  'vol': 0.035, 'spread_bp': 5.0,  'avg_vol': 95_000_000},
    'JPM':   {'name': 'JPMorgan Chase',     'base': 195, 'drift': 0.0005, 'vol': 0.015, 'spread_bp': 2.0,  'avg_vol': 10_000_000},
    'JNJ':   {'name': 'Johnson & Johnson',  'base': 155, 'drift': 0.0003, 'vol': 0.010, 'spread_bp': 1.5,  'avg_vol': 7_000_000},
    'NVDA':  {'name': 'NVIDIA Corp.',       'base': 880, 'drift': 0.0015, 'vol': 0.030, 'spread_bp': 3.5,  'avg_vol': 40_000_000},
    'META':  {'name': 'Meta Platforms',     'base': 510, 'drift': 0.0008, 'vol': 0.022, 'spread_bp': 2.8,  'avg_vol': 18_000_000},
    'XOM':   {'name': 'Exxon Mobil',        'base': 115, 'drift': 0.0004, 'vol': 0.018, 'spread_bp': 2.0,  'avg_vol': 15_000_000},
    'WMT':   {'name': 'Walmart Inc.',       'base': 170, 'drift': 0.0003, 'vol': 0.010, 'spread_bp': 1.2,  'avg_vol': 8_000_000},
    'SPY':   {'name': 'S&P 500 ETF',       'base': 530, 'drift': 0.0004, 'vol': 0.011, 'spread_bp': 0.5,  'avg_vol': 80_000_000},
    'IWM':   {'name': 'Russell 2000 ETF',  'base': 210, 'drift': 0.0003, 'vol': 0.015, 'spread_bp': 1.5,  'avg_vol': 25_000_000},
    'QQQ':   {'name': 'Nasdaq-100 ETF',    'base': 480, 'drift': 0.0005, 'vol': 0.014, 'spread_bp': 0.8,  'avg_vol': 50_000_000},
    'GME':   {'name': 'GameStop Corp.',     'base': 25,  'drift': 0.0002, 'vol': 0.055, 'spread_bp': 15.0, 'avg_vol': 5_000_000},
}

AVAILABLE_TICKERS = list(ASSET_PROFILES.keys())


# ══════════════════════════════════════════════════════════════
# Data Generator
# ══════════════════════════════════════════════════════════════

def generate_ohlcv_data(ticker: str, n_days: int = 750, seed=None) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p = ASSET_PROFILES.get(ticker.upper(), ASSET_PROFILES['AAPL'])

    close = np.zeros(n_days)
    close[0] = p['base']
    sigma = np.full(n_days, p['vol'])

    for t in range(1, n_days):
        sigma[t] = p['vol'] * 0.3 + 0.65 * sigma[t - 1] + 0.05 * abs(rng.standard_normal())
        close[t] = close[t - 1] * np.exp(p['drift'] + sigma[t] * rng.standard_normal())

    spread_bp = p['spread_bp'] / 10000
    high = np.zeros(n_days)
    low = np.zeros(n_days)
    opens = np.zeros(n_days)
    volumes = np.zeros(n_days, dtype=int)

    for t in range(n_days):
        true_range = close[t] * sigma[t] * rng.uniform(0.6, 1.4)
        spread_comp = close[t] * spread_bp * rng.uniform(0.5, 1.5)
        total = true_range + spread_comp
        mid = close[t] + rng.uniform(-0.3, 0.3) * true_range
        high[t] = mid + total * rng.uniform(0.4, 0.6)
        low[t] = mid - total * rng.uniform(0.4, 0.6)
        low[t] = max(low[t], close[t] * 0.93)
        high[t] = max(high[t], close[t] * 1.001)
        low[t] = min(low[t], close[t] * 0.999)
        opens[t] = close[t] * (1 + rng.uniform(-0.005, 0.005))
        volumes[t] = int(p['avg_vol'] * (sigma[t] / p['vol']) * rng.lognormal(0, 0.3))

    dates = pd.bdate_range(end='2025-04-30', periods=n_days)
    return pd.DataFrame({
        'date': dates.strftime('%Y-%m-%d'),
        'open': np.round(opens, 2), 'high': np.round(high, 2),
        'low': np.round(low, 2), 'close': np.round(close, 2),
        'volume': volumes,
    })


# ══════════════════════════════════════════════════════════════
# Corwin-Schultz Spread Estimator
# ══════════════════════════════════════════════════════════════

def corwin_schultz_spread(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """
    Corwin & Schultz (2012, Journal of Finance):
    Estimate bid-ask spread from daily high-low prices.

    Insight: The high-low ratio captures both volatility and spread.
    By comparing single-day vs two-day high-low ratios, the spread
    component is separated from the volatility component.

    S = 2(e^α - 1) / (1 + e^α)
    α = (√(2β) - √β) / (3 - 2√2) - √(γ / (3 - 2√2))
    β = E[Σ ln(Ht/Lt)²]  (single day)
    γ = [ln(H2d / L2d)]²  (two-day)
    """
    n = len(high)
    spreads = np.full(n, np.nan)
    sqrt2 = np.sqrt(2)
    denom = 3 - 2 * sqrt2

    for t in range(1, n):
        ln_hl_t = np.log(high[t] / low[t])
        ln_hl_tm1 = np.log(high[t - 1] / low[t - 1])
        beta = ln_hl_t ** 2 + ln_hl_tm1 ** 2

        h2 = max(high[t], high[t - 1])
        l2 = min(low[t], low[t - 1])
        gamma = np.log(h2 / l2) ** 2

        alpha = (sqrt2 * np.sqrt(beta) - np.sqrt(beta)) / denom - np.sqrt(max(gamma / denom, 0))

        if alpha > 0:
            ea = np.exp(alpha)
            spreads[t] = 2 * (ea - 1) / (1 + ea)
        else:
            spreads[t] = 0.0

    return spreads


# ══════════════════════════════════════════════════════════════
# Additional Spread / Liquidity Measures
# ══════════════════════════════════════════════════════════════

def roll_spread(close: np.ndarray) -> np.ndarray:
    """
    Roll (1984) implied spread estimator.
    S = 2 * √(-Cov(Δp_t, Δp_{t-1})) if covariance is negative, else 0.
    """
    n = len(close)
    spreads = np.full(n, np.nan)
    delta = np.diff(close)

    window = 21
    for t in range(window + 1, n):
        d1 = delta[t - window:t]
        d0 = delta[t - window - 1:t - 1]
        cov = np.cov(d1, d0)[0, 1]
        if cov < 0:
            spreads[t] = 2 * np.sqrt(-cov) / close[t] * 100  # percent
        else:
            spreads[t] = 0.0

    return spreads


def amihud_illiquidity(ret: np.ndarray, volume: np.ndarray, window: int = 21) -> np.ndarray:
    """
    Amihud (2002) illiquidity ratio.
    ILLIQ = |r_t| / Volume_t  (averaged over window)
    Higher = less liquid.
    """
    n = len(ret)
    illiq = np.full(n, np.nan)
    for t in range(window, n):
        r_slice = np.abs(ret[t - window:t])
        v_slice = volume[t - window:t].astype(float)
        v_slice[v_slice == 0] = np.nan
        ratio = r_slice / v_slice
        illiq[t] = np.nanmean(ratio) * 1e6  # scale
    return illiq


def realized_volatility(ret: np.ndarray, window: int = 21) -> np.ndarray:
    """Rolling realized volatility (annualized)."""
    n = len(ret)
    rv = np.full(n, np.nan)
    for t in range(window, n):
        rv[t] = np.std(ret[t - window:t]) * np.sqrt(252) * 100  # annualized %
    return rv


# ══════════════════════════════════════════════════════════════
# Main Endpoint
# ══════════════════════════════════════════════════════════════

@router.post("/corwin-schultz")
async def corwin_schultz_endpoint(request: SpreadRequest):
    try:
        # ── 1. Data ──
        if request.generate or not request.data:
            df = generate_ohlcv_data(request.ticker, request.nDays, request.seed)
            high_col, low_col, close_col, volume_col, date_col = 'high', 'low', 'close', 'volume', 'date'
        else:
            df = pd.DataFrame(request.data)

            def find_col(candidates, override):
                if override and override in df.columns:
                    return override
                for c in candidates:
                    if c in df.columns:
                        return c
                    if c.lower() in [x.lower() for x in df.columns]:
                        return [x for x in df.columns if x.lower() == c.lower()][0]
                return None

            date_col = find_col(['date', 'Date', 'timestamp'], request.dateCol)
            high_col = find_col(['high', 'High'], request.highCol)
            low_col = find_col(['low', 'Low'], request.lowCol)
            close_col = find_col(['close', 'Close', 'price', 'Price'], request.closeCol)
            volume_col = find_col(['volume', 'Volume', 'vol'], request.volumeCol)

            if not high_col or not low_col or not close_col:
                raise HTTPException(status_code=400, detail="Need high, low, close columns. Map via highCol/lowCol/closeCol.")

            for c in [high_col, low_col, close_col]:
                df[c] = pd.to_numeric(df[c], errors='coerce')
            if volume_col:
                df[volume_col] = pd.to_numeric(df[volume_col], errors='coerce')
            df = df.dropna(subset=[high_col, low_col, close_col])

        n = len(df)
        if n < 30:
            raise HTTPException(status_code=400, detail=f"Need >=30 rows, got {n}")

        high = df[high_col].values
        low = df[low_col].values
        close = df[close_col].values
        volume = df[volume_col].values if volume_col else None
        dates = df[date_col].values if date_col else [str(i) for i in range(n)]
        returns = np.diff(np.log(close)) * 100  # log returns %
        returns = np.concatenate([[0], returns])

        # ── 2. Corwin-Schultz ──
        cs_spread = corwin_schultz_spread(high, low)
        cs_spread_bp = cs_spread * 10000  # to basis points

        # Rolling spread
        w = request.rollingWindow
        rolling_cs = pd.Series(cs_spread_bp).rolling(w, min_periods=w // 2).mean().values

        # ── 3. Other Measures ──
        roll_sp = roll_spread(close)
        amihud = amihud_illiquidity(returns, volume, window=w) if volume is not None else None
        rv = realized_volatility(returns, window=w)

        # ── 4. Summary Stats ──
        valid_cs = cs_spread_bp[~np.isnan(cs_spread_bp)]
        cs_mean = safe_float(np.mean(valid_cs))
        cs_median = safe_float(np.median(valid_cs))
        cs_std = safe_float(np.std(valid_cs))
        cs_p25 = safe_float(np.percentile(valid_cs, 25))
        cs_p75 = safe_float(np.percentile(valid_cs, 75))
        cs_p95 = safe_float(np.percentile(valid_cs, 95))
        cs_min = safe_float(np.min(valid_cs))
        cs_max = safe_float(np.max(valid_cs))

        # Dollar spread (spread × price)
        dollar_spread = cs_spread * close
        avg_dollar_spread = safe_float(np.nanmean(dollar_spread))

        # Effective annual cost
        # If you trade round-trip once a day for 252 days
        turnover_cost_annual = safe_float(cs_mean / 10000 * 252 * 100)  # %

        # Volume-weighted spread
        if volume is not None:
            vw_spread = safe_float(np.nansum(cs_spread_bp * volume) / np.nansum(volume[~np.isnan(cs_spread_bp)]))
        else:
            vw_spread = cs_mean

        # Spread-return correlation
        valid_mask = ~np.isnan(cs_spread_bp) & ~np.isnan(returns)
        spread_ret_corr = safe_float(np.corrcoef(cs_spread_bp[valid_mask], np.abs(returns[valid_mask]))[0, 1])

        # Spread-volume correlation
        spread_vol_corr = None
        if volume is not None:
            valid_mask2 = ~np.isnan(cs_spread_bp) & (volume > 0)
            spread_vol_corr = safe_float(np.corrcoef(cs_spread_bp[valid_mask2], np.log(volume[valid_mask2]))[0, 1])

        # ── 5. Liquidity regimes ──
        regime_labels = np.full(n, 'normal', dtype=object)
        for t in range(n):
            if not np.isnan(cs_spread_bp[t]):
                if cs_spread_bp[t] > cs_p95:
                    regime_labels[t] = 'illiquid'
                elif cs_spread_bp[t] > cs_p75:
                    regime_labels[t] = 'wide'
                elif cs_spread_bp[t] < cs_p25:
                    regime_labels[t] = 'tight'

        n_illiquid = int((regime_labels == 'illiquid').sum())
        n_wide = int((regime_labels == 'wide').sum())
        n_tight = int((regime_labels == 'tight').sum())

        # ── 6. Chart Data ──

        # Time series (sampled for large datasets)
        step = max(1, n // 600)
        ts_chart = []
        for i in range(0, n, step):
            entry = {
                'date': str(dates[i]),
                'close': safe_float(close[i]),
                'cs_spread_bp': safe_float(cs_spread_bp[i]),
                'rolling_cs': safe_float(rolling_cs[i]),
                'return': safe_float(returns[i]),
                'realized_vol': safe_float(rv[i]),
                'regime': str(regime_labels[i]),
            }
            if volume is not None:
                entry['volume'] = int(volume[i])
            if amihud is not None:
                entry['amihud'] = safe_float(amihud[i])
            entry['roll_spread'] = safe_float(roll_sp[i])
            ts_chart.append(entry)

        # Spread distribution histogram
        bins = np.linspace(0, min(np.nanpercentile(valid_cs, 99), cs_max), 40)
        spread_hist = []
        for j in range(len(bins) - 1):
            lo, hi = bins[j], bins[j + 1]
            count = int(((valid_cs >= lo) & (valid_cs < hi)).sum())
            spread_hist.append({'range': f'{(lo + hi) / 2:.1f}', 'count': count})

        # Monthly aggregation
        monthly_chart = []
        if date_col:
            df['_month'] = pd.to_datetime(df[date_col]).dt.to_period('M')
            df['_cs'] = cs_spread_bp
            df['_rv'] = rv
            if volume_col:
                df['_vol'] = volume

            monthly = df.groupby('_month').agg(
                cs_mean=('_cs', 'mean'),
                cs_std=('_cs', 'std'),
                rv_mean=('_rv', 'mean'),
                count=('_cs', 'count'),
            ).dropna()

            if volume_col:
                vol_monthly = df.groupby('_month')['_vol'].mean()
                monthly = monthly.join(vol_monthly.rename('vol_mean'))

            for idx, row in monthly.iterrows():
                entry = {
                    'month': str(idx),
                    'cs_mean': safe_float(row['cs_mean']),
                    'cs_std': safe_float(row['cs_std']),
                    'rv_mean': safe_float(row['rv_mean']),
                    'count': int(row['count']),
                }
                if 'vol_mean' in row:
                    entry['vol_mean'] = safe_float(row['vol_mean'])
                monthly_chart.append(entry)

            df.drop(columns=['_month', '_cs', '_rv'] + (['_vol'] if volume_col else []), inplace=True, errors='ignore')

        # Spread vs Volume scatter
        scatter_data = []
        if volume is not None:
            step_sc = max(1, n // 300)
            for i in range(0, n, step_sc):
                if not np.isnan(cs_spread_bp[i]) and volume[i] > 0:
                    scatter_data.append({
                        'log_volume': safe_float(np.log10(volume[i])),
                        'cs_spread_bp': safe_float(cs_spread_bp[i]),
                        'regime': str(regime_labels[i]),
                    })

        # Measure comparison (daily averages by month)
        measure_comparison = []
        valid_roll = roll_sp[~np.isnan(roll_sp)]
        if len(valid_roll) > 0:
            measure_comparison.append({'measure': 'CS Spread', 'mean_bp': cs_mean, 'median_bp': cs_median, 'std_bp': cs_std})
            measure_comparison.append({
                'measure': 'Roll Spread',
                'mean_bp': safe_float(np.mean(valid_roll) * 100),  # approx bps
                'median_bp': safe_float(np.median(valid_roll) * 100),
                'std_bp': safe_float(np.std(valid_roll) * 100),
            })

        # ── 7. Build Response ──
        profile = ASSET_PROFILES.get(request.ticker.upper(), {})

        results = {
            'ticker': request.ticker.upper(),
            'asset_name': profile.get('name', request.ticker),
            'n_observations': n,
            'date_range': f'{dates[0]} to {dates[-1]}' if len(dates) > 1 else '',
            'rolling_window': w,
            'summary': {
                'cs_mean_bp': cs_mean,
                'cs_median_bp': cs_median,
                'cs_std_bp': cs_std,
                'cs_p25_bp': cs_p25,
                'cs_p75_bp': cs_p75,
                'cs_p95_bp': cs_p95,
                'cs_min_bp': cs_min,
                'cs_max_bp': cs_max,
                'avg_dollar_spread': avg_dollar_spread,
                'vw_spread_bp': vw_spread,
                'annual_roundtrip_cost_pct': turnover_cost_annual,
                'spread_return_corr': spread_ret_corr,
                'spread_volume_corr': spread_vol_corr,
            },
            'regimes': {
                'illiquid': n_illiquid,
                'wide': n_wide,
                'normal': int((regime_labels == 'normal').sum()),
                'tight': n_tight,
            },
            'return_stats': {
                'mean': safe_float(returns.mean()),
                'std': safe_float(returns.std()),
                'annual_vol': safe_float(returns.std() * np.sqrt(252)),
            },
            'charts': {
                'time_series': ts_chart,
                'spread_distribution': spread_hist,
                'monthly': monthly_chart,
                'spread_vs_volume': scatter_data,
                'measure_comparison': measure_comparison,
            },
        }

        return _to_native({'results': results})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")
