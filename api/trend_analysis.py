"""
Trend & Change Analysis API
5-step framework for time series trend analysis
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats
from scipy.special import gammaln as _gammaln
from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

router = APIRouter()

class TrendRequest(BaseModel):
    data: List[Dict[str, Any]]
    date_col: str
    value_col: str
    external_cols: Optional[List[str]] = None
    forecast_model: str = 'auto'        # auto | linear | holt_winters | arima_lite
    forecast_periods: int = 6           # number of periods to forecast (default 6)
    max_lag: int = 6                    # max lag to test in cross-correlation (default 6)
    cusum_threshold: float = 4.0        # CUSUM decision threshold (h * sigma)
    cusum_drift: float = 0.5            # CUSUM allowance (k * sigma)
    pelt_penalty: Optional[float] = None # PELT penalty (None = BIC auto)
    bayes_hazard: float = 0.05          # Bayesian prior prob of CP per period (1/expected_run)
    min_segment_size: int = 3           # minimum periods between change points

def _to_native(obj):
    """Recursively convert numpy scalars/arrays to plain Python types."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_native(v) for v in obj]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64


# Step 1: Trend Overview
def analyze_overview(df: pd.DataFrame, date_col: str, value_col: str) -> Dict:
    df = df.sort_values(date_col).reset_index(drop=True)
    values = pd.to_numeric(df[value_col], errors='coerce').dropna()
    
    n_periods = len(values)
    start_value = values.iloc[0]
    end_value = values.iloc[-1]
    
    growth_rate = (end_value - start_value) / start_value * 100 if start_value != 0 else 0
    
    # Calculate period-over-period changes
    pct_changes = values.pct_change().dropna() * 100
    avg_change = pct_changes.mean()
    
    # Linear regression for trend
    x = np.arange(len(values))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, values)
    
    if slope > 0 and p_value < 0.05:
        trend_direction = 'Upward'
    elif slope < 0 and p_value < 0.05:
        trend_direction = 'Downward'
    else:
        trend_direction = 'Flat'
    
    return {
        'n_periods': n_periods,
        'start_value': _to_native(start_value),
        'end_value': _to_native(end_value),
        'min_value': _to_native(values.min()),
        'max_value': _to_native(values.max()),
        'mean_value': _to_native(values.mean()),
        'growth_rate': _to_native(growth_rate),
        'avg_change': _to_native(avg_change),
        'volatility': _to_native(values.std()),
        'trend_direction': trend_direction,
        'trend_slope': _to_native(slope),
        'trend_r_squared': _to_native(r_value ** 2),
        'trend_significant': bool(p_value < 0.05)
    }


def create_overview_chart(df: pd.DataFrame, date_col: str, value_col: str, overview: Dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    df = df.sort_values(date_col).reset_index(drop=True)
    dates = df[date_col].astype(str)
    values = pd.to_numeric(df[value_col], errors='coerce')
    
    # Chart 1: Time series with trend line
    axes[0].plot(range(len(values)), values, marker='o', markersize=3, color='#3b82f6', linewidth=1.5, label='Actual')
    
    # Add trend line
    x = np.arange(len(values))
    slope = overview['trend_slope']
    intercept = overview['mean_value'] - slope * len(values) / 2
    trend_line = slope * x + intercept
    axes[0].plot(x, trend_line, color='#ef4444', linestyle='--', linewidth=2, label=f'Trend ({overview["trend_direction"]})')
    
    axes[0].set_xlabel('Period')
    axes[0].set_ylabel('Value')
    axes[0].set_title('Time Series Trend', fontsize=11, fontweight='bold')
    axes[0].legend()
    
    # Chart 2: Distribution
    axes[1].hist(values.dropna(), bins=15, color='#3b82f6', alpha=0.7, edgecolor='black')
    axes[1].axvline(x=overview['mean_value'], color='red', linestyle='--', label=f"Mean: {overview['mean_value']:.0f}")
    axes[1].set_xlabel('Value')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Value Distribution', fontsize=11, fontweight='bold')
    axes[1].legend()
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# Step 2: Period Comparison (YoY / MoM)
def analyze_comparison(df: pd.DataFrame, date_col: str, value_col: str) -> Dict:
    df = df.sort_values(date_col).reset_index(drop=True)
    df['_value'] = pd.to_numeric(df[value_col], errors='coerce')
    df['_date'] = pd.to_datetime(df[date_col], errors='coerce')
    
    if df['_date'].isna().all():
        return {'error': 'Cannot parse dates for comparison'}
    
    df = df.dropna(subset=['_date', '_value'])
    df['_year'] = df['_date'].dt.year
    df['_month'] = df['_date'].dt.month
    
    # Calculate YoY and MoM
    df = df.sort_values('_date').reset_index(drop=True)
    
    recent_changes = []
    for i, row in df.iterrows():
        period = row[date_col]
        value = row['_value']
        
        # YoY
        yoy = None
        prev_year = df[(df['_year'] == row['_year'] - 1) & (df['_month'] == row['_month'])]
        if len(prev_year) > 0:
            prev_val = prev_year['_value'].iloc[0]
            yoy = (value - prev_val) / prev_val * 100 if prev_val != 0 else None
        
        # MoM
        mom = None
        if i > 0:
            prev_val = df.iloc[i-1]['_value']
            mom = (value - prev_val) / prev_val * 100 if prev_val != 0 else None
        
        recent_changes.append({
            'period': str(period),
            'value': _to_native(value),
            'yoy': _to_native(yoy),
            'mom': _to_native(mom)
        })
    
    # Get latest and averages
    yoy_values = [c['yoy'] for c in recent_changes if c['yoy'] is not None]
    mom_values = [c['mom'] for c in recent_changes if c['mom'] is not None]
    
    return {
        'recent_changes': recent_changes[-12:],  # Last 12 periods
        'latest_yoy': _to_native(yoy_values[-1]) if yoy_values else None,
        'latest_mom': _to_native(mom_values[-1]) if mom_values else None,
        'avg_yoy': _to_native(np.mean(yoy_values)) if yoy_values else None,
        'avg_mom': _to_native(np.mean(mom_values)) if mom_values else None,
        'yoy_positive_rate': _to_native(sum(1 for y in yoy_values if y > 0) / len(yoy_values) * 100) if yoy_values else None
    }


def create_comparison_chart(df: pd.DataFrame, date_col: str, comparison: Dict) -> str:
    if comparison.get('error'):
        return ""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    changes = comparison['recent_changes'][-12:]
    periods = [c['period'][-7:] for c in changes]  # Shorten labels
    
    # Chart 1: YoY comparison
    yoy = [c['yoy'] if c['yoy'] is not None else 0 for c in changes]
    colors = ['#10b981' if y > 0 else '#ef4444' for y in yoy]
    axes[0].bar(range(len(yoy)), yoy, color=colors, alpha=0.8, edgecolor='black')
    axes[0].axhline(y=0, color='black', linewidth=0.5)
    axes[0].set_xticks(range(len(periods)))
    axes[0].set_xticklabels(periods, rotation=45, ha='right', fontsize=8)
    axes[0].set_ylabel('YoY Change (%)')
    axes[0].set_title('Year-over-Year Change', fontsize=11, fontweight='bold')
    
    # Chart 2: MoM comparison
    mom = [c['mom'] if c['mom'] is not None else 0 for c in changes]
    colors = ['#10b981' if m > 0 else '#ef4444' for m in mom]
    axes[1].bar(range(len(mom)), mom, color=colors, alpha=0.8, edgecolor='black')
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].set_xticks(range(len(periods)))
    axes[1].set_xticklabels(periods, rotation=45, ha='right', fontsize=8)
    axes[1].set_ylabel('MoM Change (%)')
    axes[1].set_title('Month-over-Month Change', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# ══════════════════════════════════════════════════════════════
# Step 3: Cross-correlation with lag analysis
# ══════════════════════════════════════════════════════════════
def _corr_strength(r: float) -> str:
    a = abs(r)
    return 'Strong' if a > 0.7 else 'Moderate' if a > 0.4 else 'Weak'


def _lag_correlations(target: np.ndarray, driver: np.ndarray, max_lag: int) -> List[Dict]:
    """
    Compute Pearson r between driver[t-lag] and target[t] for lag 0..max_lag.

    Positive lag  → driver LEADS target (e.g. advertising now → sales later)
    Negative lag  → driver LAGS target (reversed causality signal)

    Bonferroni-corrected significance threshold across (2*max_lag+1) tests.
    Reference: Box & Jenkins (1970); Shumway & Stoffer (2017), Ch. 1
    """
    results = []
    n_tests = max_lag + 1                        # lag 0..max_lag
    alpha_bonf = 0.05 / max(n_tests, 1)

    for lag in range(0, max_lag + 1):
        if lag == 0:
            y, x = target, driver
        else:
            y = target[lag:]       # target at t
            x = driver[:-lag]      # driver at t-lag (driver leads by lag)

        valid = ~(np.isnan(y) | np.isnan(x))
        if valid.sum() < 5:
            continue

        try:
            r, p = stats.pearsonr(x[valid], y[valid])
        except Exception:
            continue

        results.append({
            'lag':         int(lag),
            'correlation': _to_native(r),
            'p_value':     _to_native(p),
            'significant': bool(float(p) < alpha_bonf),
            'n_obs':       int(valid.sum()),
        })
    return results


def analyze_correlation(df: pd.DataFrame, value_col: str,
                        external_cols: List[str], max_lag: int = 6) -> Dict:
    """
    For each external factor:
      1. Compute lag-0..max_lag cross-correlations with Bonferroni correction.
      2. Find the lag with peak |r| → 'best_lag'.
      3. Report same-time (lag 0) plus best-lag result separately for easy reading.

    Response per factor:
      lag_profile      — list of {lag, correlation, p_value, significant}
      lag_0            — same-time correlation (backward-compat)
      best_lag         — lag index with highest |r|
      best_correlation — r at best_lag
      lag_interpretation — plain-English description
    """
    if not external_cols:
        return {'error': 'No external factors provided'}

    target = pd.to_numeric(df[value_col], errors='coerce').values
    correlations = []

    for col in external_cols:
        if col not in df.columns:
            continue

        driver = pd.to_numeric(df[col], errors='coerce').values
        lag_profile = _lag_correlations(target, driver, max_lag)

        if not lag_profile:
            continue

        # Same-time (lag 0)
        lag0 = next((l for l in lag_profile if l['lag'] == 0), lag_profile[0])

        # Best lag by |r|
        best = max(lag_profile, key=lambda l: abs(l['correlation'] or 0))

        # Interpret lag direction
        if best['lag'] == 0:
            interp = f"Contemporaneous effect (lag 0)"
        else:
            interp = (f"{col} leads {value_col} by {best['lag']} period(s) "
                      f"(r={best['correlation']:.3f})")

        correlations.append({
            'factor':           col,
            # backward-compat fields (lag 0)
            'correlation':      lag0['correlation'],
            'p_value':          lag0['p_value'],
            'significant':      lag0['significant'],
            'strength':         _corr_strength(lag0['correlation'] or 0),
            # new lag fields
            'lag_0_correlation': lag0['correlation'],
            'best_lag':         best['lag'],
            'best_correlation': best['correlation'],
            'best_significant': best['significant'],
            'best_strength':    _corr_strength(best['correlation'] or 0),
            'lag_interpretation': interp,
            'lag_profile':      lag_profile,
        })

    # Sort by best |r| across all lags
    correlations.sort(key=lambda c: abs(c['best_correlation'] or 0), reverse=True)

    top_driver = correlations[0] if correlations else None

    return {
        'correlations':  correlations,
        'top_driver':    top_driver,
        'n_significant': sum(1 for c in correlations if c['best_significant']),
        'max_lag_tested': max_lag,
    }


def create_correlation_chart(df: pd.DataFrame, value_col: str, correlation: Dict) -> str:
    """
    2-panel chart:
      Left  — Bar chart of best-lag |r| per factor, labelled with lag
      Right — Cross-correlation profile (lag vs r) for the top driver
    """
    if correlation.get('error') or not correlation.get('correlations'):
        return ""

    corrs = correlation['correlations'][:8]
    top   = correlation.get('top_driver')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ── Chart 1: Best-lag correlation per factor ──────────────
    factors    = [c['factor'][:15] for c in corrs]
    best_rs    = [c['best_correlation'] or 0 for c in corrs]
    best_lags  = [c['best_lag'] for c in corrs]
    colors     = ['#3b82f6' if c['best_significant'] else '#d1d5db' for c in corrs]

    bars = axes[0].barh(factors, best_rs, color=colors, alpha=0.85, edgecolor='black')
    axes[0].axvline(x=0, color='black', linewidth=0.5)
    axes[0].set_xlabel('Correlation (at best lag)')
    axes[0].set_title('Factor Correlations (Best Lag)', fontsize=11, fontweight='bold')

    # Annotate lag number on each bar
    for bar, lag, r in zip(bars, best_lags, best_rs):
        xpos = r + (0.01 if r >= 0 else -0.01)
        axes[0].text(xpos, bar.get_y() + bar.get_height() / 2,
                     f'lag {lag}', va='center', ha='left' if r >= 0 else 'right',
                     fontsize=8, color='#374151')

    # ── Chart 2: Lag profile of top driver ────────────────────
    if top and top.get('lag_profile'):
        profile = top['lag_profile']
        lags  = [p['lag'] for p in profile]
        rs    = [p['correlation'] or 0 for p in profile]
        sigs  = [p['significant'] for p in profile]
        pt_colors = ['#ef4444' if s else '#93c5fd' for s in sigs]

        axes[1].bar(lags, rs, color=pt_colors, alpha=0.8, edgecolor='black', width=0.6)
        axes[1].axhline(y=0, color='black', linewidth=0.5)

        # Mark best lag
        best_lag = top['best_lag']
        axes[1].axvline(x=best_lag, color='#ef4444', linestyle='--', linewidth=1.5,
                        label=f'Best lag={best_lag}')

        axes[1].set_xlabel('Lag (periods driver leads target)')
        axes[1].set_ylabel('Pearson r')
        axes[1].set_title(f"Lag Profile: {top['factor'][:20]}", fontsize=11, fontweight='bold')
        axes[1].set_xticks(lags)
        axes[1].legend(fontsize=8)

        # Significance note
        axes[1].text(0.02, 0.97, '■ significant  □ not significant',
                     transform=axes[1].transAxes, fontsize=7,
                     va='top', color='#6b7280')

    plt.tight_layout()
    return _fig_to_base64(fig)


# Step 4: Anomaly Detection
# ══════════════════════════════════════════════════════════════
# STL-style decomposition (no external deps)
# ══════════════════════════════════════════════════════════════
def _stl_decompose(values: np.ndarray, season_period: int) -> Dict[str, np.ndarray]:
    """
    Manual STL-style decomposition in 3 passes:

    Pass 1 — Trend extraction
        Centered moving average with window = season_period (or nearest odd int).
        Falls back to linear OLS fit when n < 2*period (short series).

    Pass 2 — Seasonal extraction
        For each phase p in [0, period-1]:
            seasonal[p] = mean of (detrended values at that phase)
        Force zero-mean seasonal by subtracting the grand mean.

    Pass 3 — Residual
        residual = observed − trend − seasonal

    Reference: Cleveland et al. (1990), "STL: A Seasonal-Trend Decomposition
               Procedure Based on Loess", J. Official Statistics 6(1):3–73.
    """
    n = len(values)

    # ── Pass 1: Trend ─────────────────────────────────────────
    if n >= 2 * season_period:
        win = season_period if season_period % 2 == 1 else season_period + 1
        half = win // 2
        trend = np.full(n, np.nan)
        for i in range(half, n - half):
            trend[i] = np.mean(values[i - half: i + half + 1])
        # Fill edges with linear extrapolation from fitted interior
        valid = ~np.isnan(trend)
        x_all = np.arange(n)
        if valid.sum() >= 2:
            slope, intercept, *_ = stats.linregress(x_all[valid], trend[valid])
            for i in range(n):
                if np.isnan(trend[i]):
                    trend[i] = slope * i + intercept
    else:
        # Short series: use linear OLS as trend
        x = np.arange(n)
        slope, intercept, *_ = stats.linregress(x, values)
        trend = slope * x + intercept

    # ── Pass 2: Seasonal ──────────────────────────────────────
    detrended = values - trend
    seasonal = np.zeros(n)
    for phase in range(season_period):
        idx = np.arange(phase, n, season_period)
        seasonal[idx] = np.mean(detrended[idx])
    # Force zero-mean
    seasonal -= np.mean(seasonal)

    # ── Pass 3: Residual ──────────────────────────────────────
    residual = values - trend - seasonal

    return {'trend': trend, 'seasonal': seasonal, 'residual': residual}


def analyze_anomaly(df: pd.DataFrame, date_col: str, value_col: str,
                    season_period: int = 0, threshold: float = 2.5) -> Dict:
    """
    STL-residual anomaly detection.

    Why STL residuals instead of raw z-score:
    - Upward-trending series: late values have high raw z → false spikes
    - Seasonal series: seasonal peaks flag as anomalies every cycle
    - STL strips both effects → residual is mean-stationary
    - Anomaly = unexpected deviation unexplained by trend OR season

    season_period : 0 = auto-detect via ACF (same logic as detect_seasonality)
    threshold     : z-score cutoff applied to residuals (default 2.5)
                    (higher than raw-z threshold because residuals are tighter)

    Additional improvement: MAD-based robust z (median absolute deviation)
    replaces mean/std to avoid masking effects (Leys et al., 2013).

    Reference: Cleveland et al. (1990); Leys et al. (2013), J. Exp. Social Psych.
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    values = pd.to_numeric(df[value_col], errors='coerce').values
    n = len(values)

    # ── Auto-detect season period ─────────────────────────────
    if season_period == 0:
        s_info = detect_seasonality(values)
        season_period = s_info['period'] if s_info['has_seasonality'] else max(4, n // 6)
    season_period = max(2, min(season_period, n // 2))

    # ── STL decomposition ─────────────────────────────────────
    decomp = _stl_decompose(values, season_period)
    trend    = decomp['trend']
    seasonal = decomp['seasonal']
    residual = decomp['residual']

    # ── Robust z-score on residuals (MAD) ─────────────────────
    # MAD = median(|x - median(x)|); robust_z = 0.6745 * resid / MAD
    # 0.6745 = consistency factor so MAD ≈ σ for Gaussian
    med_r  = np.nanmedian(residual)
    mad    = np.nanmedian(np.abs(residual - med_r))
    mad    = max(mad, 1e-8)  # prevent div-by-zero on flat residuals
    robust_z = 0.6745 * (residual - med_r) / mad

    # ── Flag anomalies ────────────────────────────────────────
    anomalies = []
    for i in range(n):
        z = float(robust_z[i])
        if abs(z) > threshold:
            pct_dev = float((residual[i] / max(abs(trend[i]), 1e-8)) * 100)
            anomalies.append({
                'period':        str(df.iloc[i][date_col]),
                'value':         _to_native(float(values[i])),
                'trend_value':   _to_native(float(trend[i])),
                'residual':      _to_native(float(residual[i])),
                'robust_z':      _to_native(z),
                'pct_deviation': _to_native(pct_dev),
                'type':          'Spike' if z > 0 else 'Drop',
            })

    # ── Inflection points (on smoothed trend) ─────────────────
    inflections = []
    if n > 5:
        trend_s = pd.Series(trend)
        diff    = trend_s.diff()
        for i in range(2, n - 2):
            d_prev = diff.iloc[i - 1]
            d_curr = diff.iloc[i]
            if pd.notna(d_prev) and pd.notna(d_curr):
                prev_val = values[i - 2] if values[i - 2] != 0 else 1e-8
                change   = float((values[i] - values[i - 2]) / abs(prev_val) * 100)
                if d_prev > 0 and d_curr < 0:
                    inflections.append({
                        'period':    str(df.iloc[i][date_col]),
                        'direction': 'Peak → Decline',
                        'change':    _to_native(change),
                    })
                elif d_prev < 0 and d_curr > 0:
                    inflections.append({
                        'period':    str(df.iloc[i][date_col]),
                        'direction': 'Trough → Rise',
                        'change':    _to_native(change),
                    })

    return {
        'n_anomalies':    len(anomalies),
        'anomalies':      anomalies,
        'n_inflections':  len(inflections),
        'inflections':    inflections[:5],
        'threshold':      float(threshold),
        'anomaly_rate':   _to_native(float(len(anomalies) / n * 100)),
        'method':         'STL residual + MAD robust z-score',
        'season_period':  int(season_period),
        # Decomposition series for charting
        'decomposition':  {
            'trend':    [_to_native(float(v)) for v in trend],
            'seasonal': [_to_native(float(v)) for v in seasonal],
            'residual': [_to_native(float(v)) for v in residual],
            'robust_z': [_to_native(float(v)) for v in robust_z],
        },
    }


def create_anomaly_chart(df: pd.DataFrame, date_col: str, value_col: str, anomaly: Dict) -> str:
    """
    4-panel STL decomposition chart:
      Top-left    : Observed + trend line + anomaly markers
      Top-right   : Seasonal component
      Bottom-left : Residual + ±threshold bands
      Bottom-right: Residual robust-z distribution (histogram)
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    values = pd.to_numeric(df[value_col], errors='coerce').values
    n = len(values)
    x = np.arange(n)

    decomp    = anomaly.get('decomposition', {})
    trend     = decomp.get('trend',    [None] * n)
    seasonal  = decomp.get('seasonal', [None] * n)
    residual  = decomp.get('residual', [None] * n)
    robust_z  = decomp.get('robust_z', [None] * n)
    threshold = anomaly['threshold']

    anomaly_periods = {a['period'] for a in anomaly['anomalies']}

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.flatten()

    # ── Panel 1: Observed + Trend + Anomaly markers ───────────
    axes[0].plot(x, values, color='#3b82f6', linewidth=1.5, marker='o',
                 markersize=3, label='Observed', zorder=2)
    if any(v is not None for v in trend):
        axes[0].plot(x, trend, color='#f59e0b', linewidth=2,
                     linestyle='--', label='Trend (STL)', zorder=3)
    for i, row in df.iterrows():
        if str(row[date_col]) in anomaly_periods:
            axes[0].scatter([i], [values[i]], color='#ef4444', s=120,
                            zorder=5, edgecolors='black', linewidths=0.8)
    axes[0].set_title('Observed + STL Trend', fontsize=10, fontweight='bold')
    axes[0].set_ylabel('Value')
    axes[0].legend(fontsize=8)
    axes[0].set_xlabel('Period')

    # ── Panel 2: Seasonal component ───────────────────────────
    if any(v is not None for v in seasonal):
        axes[1].plot(x, seasonal, color='#8b5cf6', linewidth=1.5,
                     marker='o', markersize=2)
        axes[1].axhline(y=0, color='black', linewidth=0.5)
        axes[1].fill_between(x, seasonal, 0, alpha=0.15, color='#8b5cf6')
    axes[1].set_title(f'Seasonal Component (period={anomaly["season_period"]})',
                      fontsize=10, fontweight='bold')
    axes[1].set_ylabel('Seasonal Effect')
    axes[1].set_xlabel('Period')

    # ── Panel 3: Residual + threshold bands ───────────────────
    if any(v is not None for v in residual):
        res_arr = np.array(residual, dtype=float)
        med_r   = float(np.nanmedian(res_arr))
        mad     = float(np.nanmedian(np.abs(res_arr - med_r)))
        mad     = max(mad, 1e-8)
        upper   = med_r + (threshold * mad / 0.6745)
        lower   = med_r - (threshold * mad / 0.6745)

        axes[2].plot(x, res_arr, color='#10b981', linewidth=1.2,
                     marker='o', markersize=3, label='Residual')
        axes[2].axhline(y=upper, color='#ef4444', linestyle='--',
                        alpha=0.7, label=f'+{threshold}σ (MAD)')
        axes[2].axhline(y=lower, color='#ef4444', linestyle='--',
                        alpha=0.7, label=f'-{threshold}σ (MAD)')
        axes[2].axhline(y=0, color='black', linewidth=0.4)
        axes[2].fill_between(x, lower, upper, alpha=0.07, color='#10b981')

        # Mark anomalies on residual panel too
        for i, row in df.iterrows():
            if str(row[date_col]) in anomaly_periods:
                axes[2].scatter([i], [res_arr[i]], color='#ef4444', s=100,
                                zorder=5, edgecolors='black', linewidths=0.8)
        axes[2].legend(fontsize=8)
    axes[2].set_title('STL Residual + Anomaly Bands', fontsize=10, fontweight='bold')
    axes[2].set_ylabel('Residual')
    axes[2].set_xlabel('Period')

    # ── Panel 4: Robust-z distribution ────────────────────────
    if any(v is not None for v in robust_z):
        rz = [v for v in robust_z if v is not None]
        axes[3].hist(rz, bins=min(20, n // 2 + 1), color='#3b82f6',
                     alpha=0.7, edgecolor='black')
        axes[3].axvline(x=threshold,  color='#ef4444', linestyle='--',
                        label=f'+{threshold}σ')
        axes[3].axvline(x=-threshold, color='#ef4444', linestyle='--',
                        label=f'-{threshold}σ')
    axes[3].set_title('Residual Robust-Z Distribution', fontsize=10, fontweight='bold')
    axes[3].set_xlabel('Robust Z-Score (MAD)')
    axes[3].set_ylabel('Frequency')
    axes[3].legend(fontsize=8)

    plt.suptitle('STL Anomaly Detection: Trend + Seasonal Removed',
                 fontsize=12, fontweight='bold', y=1.01)
    plt.tight_layout()
    return _fig_to_base64(fig)


# ══════════════════════════════════════════════════════════════
# Seasonality Detection
# ══════════════════════════════════════════════════════════════
def detect_seasonality(values: np.ndarray, candidate_periods=(4, 6, 12)) -> Dict:
    """
    Detect dominant seasonal period using autocorrelation.
    Tests candidate periods [4=quarterly, 6=bimonthly, 12=monthly].
    Returns best period (highest ACF at that lag) if ACF > 0.3.
    Reference: Hyndman & Athanasopoulos (2021), "Forecasting: P&P", Ch. 2
    """
    n = len(values)
    if n < 8:
        return {'has_seasonality': False, 'period': None, 'strength': None}

    # Detrend before testing
    x = np.arange(n)
    slope, intercept, *_ = stats.linregress(x, values)
    detrended = values - (slope * x + intercept)

    best_period, best_acf = None, 0.0
    acf_by_period = {}
    for period in candidate_periods:
        if period >= n // 2:
            continue
        # Pearson ACF at this lag
        y1, y2 = detrended[:-period], detrended[period:]
        if len(y1) < 4:
            continue
        corr, _ = stats.pearsonr(y1, y2)
        acf_by_period[period] = float(corr)
        if abs(corr) > abs(best_acf):
            best_acf, best_period = corr, period

    has_seasonality = best_period is not None and abs(best_acf) > 0.3
    strength = None
    if has_seasonality:
        strength = 'Strong' if abs(best_acf) > 0.6 else 'Moderate'

    return {
        'has_seasonality': has_seasonality,
        'period': best_period,
        'acf': _to_native(best_acf) if has_seasonality else None,
        'strength': strength,
        'acf_by_period': {str(k): _to_native(v) for k, v in acf_by_period.items()},
    }


# ══════════════════════════════════════════════════════════════
# Holt-Winters (Triple Exponential Smoothing) — manual impl
# Additive model: level + trend + seasonality
# Ref: Holt (1957), Winters (1960)
# ══════════════════════════════════════════════════════════════
def _holt_winters(values: np.ndarray, period: int, n_ahead: int,
                  alpha=0.3, beta=0.1, gamma=0.2) -> np.ndarray:
    """Additive Holt-Winters with simple grid search for alpha/beta/gamma."""
    n = len(values)
    if n < 2 * period:
        return np.array([])

    def _hw_fit(a, b, g):
        # Init: level = mean of first season, trend = slope, seasonal = deviation from mean
        L = np.mean(values[:period])
        T = (np.mean(values[period:2*period]) - np.mean(values[:period])) / period
        S = values[:period] - L
        fitted = np.zeros(n)
        _L, _T = L, T
        _S = list(S) + [0.0] * n
        for t in range(n):
            prev_L, prev_T = _L, _T
            obs = values[t]
            s_t = _S[t]
            _L = a * (obs - s_t) + (1 - a) * (prev_L + prev_T)
            _T = b * (_L - prev_L) + (1 - b) * prev_T
            _S[t + period] = g * (obs - _L) + (1 - g) * s_t
            fitted[t] = _L + _T + _S[t]
        sse = np.sum((values - fitted) ** 2)
        return sse, _L, _T, _S

    # Quick grid search
    best_sse, best_params = np.inf, (alpha, beta, gamma)
    for a in [0.1, 0.2, 0.3, 0.5]:
        for b in [0.05, 0.1, 0.2]:
            for g in [0.1, 0.2, 0.3]:
                try:
                    sse, *_ = _hw_fit(a, b, g)
                    if sse < best_sse:
                        best_sse, best_params = sse, (a, b, g)
                except Exception:
                    pass
    a, b, g = best_params
    _, L_fin, T_fin, S_fin = _hw_fit(a, b, g)

    forecasts = []
    for h in range(1, n_ahead + 1):
        s_idx = (n - period + h - 1) % period + n
        s_val = S_fin[s_idx] if s_idx < len(S_fin) else S_fin[n - period + (h - 1) % period]
        forecasts.append(L_fin + h * T_fin + s_val)
    return np.array(forecasts)


# ══════════════════════════════════════════════════════════════
# ARIMA-lite: AR(p) with differencing — no external deps
# Fits AR model on first-differenced series, then integrates back
# Ref: Box & Jenkins (1970)
# ══════════════════════════════════════════════════════════════
def _arima_lite(values: np.ndarray, n_ahead: int, max_p: int = 3) -> np.ndarray:
    """
    ARIMA(p,1,0) — AR on first-differenced series, selected by AIC.
    Differencing (d=1) removes trend; AR(p) captures autocorrelation.
    No MA term for simplicity. Good enough for short-horizon forecasting.
    """
    n = len(values)
    if n < 8:
        return np.array([])

    diff = np.diff(values)  # first difference (d=1)

    def _fit_ar(series, p):
        if len(series) <= p:
            return None, np.inf
        # Build Toeplitz-style design matrix
        Y = series[p:]
        X = np.column_stack([series[p-i-1:len(series)-i-1] for i in range(p)] + [np.ones(len(Y))])
        try:
            coeffs, resid, *_ = np.linalg.lstsq(X, Y, rcond=None)
            sse = np.sum((Y - X @ coeffs) ** 2)
            k = p + 1
            aic = len(Y) * np.log(max(sse / len(Y), 1e-12)) + 2 * k
            return coeffs, aic
        except Exception:
            return None, np.inf

    # Select best p by AIC
    best_p, best_aic, best_coeffs = 1, np.inf, None
    for p in range(1, min(max_p + 1, len(diff) // 2)):
        coeffs, aic = _fit_ar(diff, p)
        if coeffs is not None and aic < best_aic:
            best_aic, best_p, best_coeffs = aic, p, coeffs

    if best_coeffs is None:
        return np.array([])

    # Forecast differenced series
    history = list(diff)
    diff_forecasts = []
    for _ in range(n_ahead):
        x_row = [history[-(i+1)] for i in range(best_p)] + [1.0]
        pred = float(np.dot(best_coeffs, x_row))
        diff_forecasts.append(pred)
        history.append(pred)

    # Integrate back (cumsum from last observed value)
    level = values[-1]
    forecasts = []
    for d in diff_forecasts:
        level += d
        forecasts.append(level)
    return np.array(forecasts)


# ══════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════
# Step 6: Change Point Detection
# 3 algorithms: CUSUM, PELT (cost-function), Bayesian Online
# ══════════════════════════════════════════════════════════════

def _cusum(values: np.ndarray, threshold: float = 4.0, drift: float = 0.5) -> List[int]:
    """
    CUSUM (Cumulative Sum) — Page (1954).

    Detects sustained shifts in the mean using a two-sided cumulative sum.
    Operates on raw values with LOCAL mean estimation per segment:
    after each detected change point, mu and sigma are re-estimated from
    a warm-up window of the new segment, preventing drift contamination.

    S+[t] = max(0, S+[t-1] + (x[t] - mu) - k)
    S-[t] = max(0, S-[t-1] - (x[t] - mu) - k)
    Flagged when S+ or S- exceeds h = threshold * sigma.

    drift     : allowance k = drift * sigma  (typically 0.5–1.0)
    threshold : h = threshold * sigma  (default 4 → ARL ≈ 500 for in-control)

    Reference: Page (1954), Biometrika 41(1-2):100-115.
    """
    n = len(values)
    if n < 6:
        return []

    init_win = max(5, min(n // 6, 12))   # warm-up window per segment
    cps: List[int] = []
    seg_start = 0

    while seg_start < n - 4:
        init_end = min(seg_start + init_win, n - 2)
        # Estimate mu/sigma from warm-up window WITHIN this segment
        win_data = values[seg_start: init_end]
        mu_loc   = float(np.mean(win_data))
        sd_loc   = float(np.std(win_data)) or 1.0
        k = drift * sd_loc
        h = threshold * sd_loc

        sp = sm = 0.0
        triggered_at = -1
        for i in range(len(values) - seg_start):
            t_abs = seg_start + i
            sp = max(0.0, sp + (values[t_abs] - mu_loc) - k)
            sm = max(0.0, sm - (values[t_abs] - mu_loc) - k)
            if sp > h or sm > h:
                triggered_at = i
                break

        if triggered_at < 0:
            break
        cp_abs = seg_start + triggered_at
        if cp_abs > seg_start:
            cps.append(cp_abs)
        seg_start = cp_abs + 1

    return cps


def _pelt(values: np.ndarray, penalty: float = None, min_size: int = 3) -> List[int]:
    """
    Binary Segmentation with BIC penalty — structural break detection.

    Finds the split that maximises the BIC gain (not just t-statistic),
    which naturally penalises over-segmentation.  Fixed significance
    threshold (no Bonferroni tightening) keeps power stable at all depths.

    BIC gain at split i:
        gain = -n*log(rss_full) + n_L*log(rss_left) + n_R*log(rss_right)
              + log(n) * 1   ← BIC penalty for one extra segment
    Split accepted when gain > 0 AND Welch t p-value < significance.

    Reference: Yao (1988), "Estimating the number of change-points",
               Ann. Statist. 16(3):1326-1301.
    """
    n = len(values)
    significance = penalty if (penalty is not None and 0 < penalty < 1) else 0.05
    max_cps = max(2, n // max(min_size, 3))

    def _rss(seg: np.ndarray) -> float:
        return float(np.sum((seg - np.mean(seg)) ** 2)) if len(seg) > 1 else 0.0

    def _best_split(start: int, end: int):
        seg = values[start:end]
        m   = len(seg)
        if m < 2 * min_size:
            return None, 0.0, 1.0
        rss_full = _rss(seg)
        best_gain, best_i, best_p = -np.inf, -1, 1.0
        for i in range(min_size, m - min_size + 1):
            left, right = seg[:i], seg[i:]
            if len(left) < 2 or len(right) < 2:
                continue
            rss_l, rss_r = _rss(left), _rss(right)
            bic_gain = (m * np.log(max(rss_full, 1e-12) / m)
                        - len(left)  * np.log(max(rss_l, 1e-12) / len(left))
                        - len(right) * np.log(max(rss_r, 1e-12) / len(right))
                        - np.log(m))          # BIC penalty
            if bic_gain > best_gain:
                try:
                    _, p = stats.ttest_ind(left, right, equal_var=False)
                except Exception:
                    p = 1.0
                best_gain, best_i, best_p = bic_gain, i, float(p)
        if best_i < 0:
            return None, 0.0, 1.0
        return start + best_i, best_gain, best_p

    cps: List[int] = []

    def _recurse(start: int, end: int, depth: int = 0):
        if end - start < 2 * min_size or len(cps) >= max_cps:
            return
        split_at, bic_gain, p_val = _best_split(start, end)
        if split_at is None or bic_gain <= 0 or p_val > significance:
            return
        cps.append(split_at)
        _recurse(start, split_at, depth + 1)
        _recurse(split_at, end,   depth + 1)

    _recurse(0, n)
    return sorted(cps)


def _bayesian_cpd(values: np.ndarray, hazard: float = 1 / 20,
                  prior_var_scale: float = 4.0) -> List[int]:
    """
    Windowed Log-Likelihood Ratio change point detection.

    Robust alternative to full BOCPD — avoids numerical instability.
    At each t, compares:
      H_long  : long-run Normal model (mean/std from all history)
      H_short : recent local Normal model (last `window` observations)
    LLR[t] = log P(short_seg | H_short) − log P(short_seg | H_long)
    High LLR → recent behaviour diverges from historical baseline → CP candidate.

    window = round(1/hazard); default hazard=1/20 → window=20.
    Threshold = 80th percentile of LLR series (adaptive, n-invariant).

    Reference: Basseville & Nikiforov (1993), "Detection of Abrupt Changes",
               Ch. 4 (GLR / CUSUM duality).
    """
    n = len(values)
    window = max(5, round(1.0 / max(hazard, 0.01)))
    if n < window * 2:
        return []

    # De-trend
    x_idx = np.arange(n, dtype=float)
    slope, intercept, *_ = stats.linregress(x_idx, values)
    resid = values - (slope * x_idx + intercept)

    llr = np.zeros(n)
    for t in range(window, n):
        long_seg  = resid[:t]
        short_seg = resid[max(0, t - window): t]
        long_mu   = float(np.mean(long_seg));  long_std  = max(float(np.std(long_seg)),  1e-4)
        short_mu  = float(np.mean(short_seg)); short_std = max(float(np.std(short_seg)), 1e-4)
        ll_long   = float(np.sum(stats.norm.logpdf(short_seg, long_mu,  long_std)))
        ll_short  = float(np.sum(stats.norm.logpdf(short_seg, short_mu, short_std)))
        llr[t]    = ll_short - ll_long

    thresh = max(float(np.percentile(llr[window:], 80)), 0.5)
    cps: List[int] = []
    for t in range(window + 2, n - 2):
        if (llr[t] > thresh
                and llr[t] >= llr[t - 1]
                and llr[t] >= llr[t + 1]):
            cps.append(t)
    return cps


def _segment_stats(values: np.ndarray, change_points: List[int], date_labels: List[str]) -> List[Dict]:
    """
    Given a list of change point indices, compute per-segment statistics:
    mean, std, trend slope, and % change vs prior segment.
    """
    n = len(values)
    boundaries = [0] + sorted(change_points) + [n]
    segments = []
    prev_mean = None

    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i + 1]
        seg = values[start:end]
        if len(seg) < 1:
            continue
        seg_mean = float(np.mean(seg))
        seg_std  = float(np.std(seg))
        # OLS slope within segment
        if len(seg) >= 2:
            x = np.arange(len(seg))
            slope, *_ = stats.linregress(x, seg)
            seg_slope = float(slope)
        else:
            seg_slope = 0.0

        pct_change = None
        if prev_mean is not None and prev_mean != 0:
            pct_change = float((seg_mean - prev_mean) / abs(prev_mean) * 100)

        segments.append({
            'segment_id':    i + 1,
            'start_idx':     int(start),
            'end_idx':       int(end - 1),
            'start_period':  date_labels[start] if start < len(date_labels) else str(start),
            'end_period':    date_labels[end - 1] if end - 1 < len(date_labels) else str(end - 1),
            'n_periods':     int(end - start),
            'mean':          _to_native(seg_mean),
            'std':           _to_native(seg_std),
            'slope':         _to_native(seg_slope),
            'pct_change_from_prev': _to_native(pct_change),
            'direction':     'Upward' if seg_slope > 0 else ('Downward' if seg_slope < 0 else 'Flat'),
        })
        prev_mean = seg_mean

    return segments


def _deduplicate_cps(cps: List[int], min_gap: int = 3) -> List[int]:
    """Merge change points that are within min_gap of each other (keep earliest)."""
    if not cps:
        return []
    result = [cps[0]]
    for cp in sorted(cps[1:]):
        if cp - result[-1] >= min_gap:
            result.append(cp)
    return result


def analyze_change_points(
    df: 'pd.DataFrame',
    date_col: str,
    value_col: str,
    cusum_threshold: float = 4.0,
    cusum_drift: float = 0.5,
    pelt_penalty: float = None,
    bayes_hazard: float = 1 / 20,
    min_segment_size: int = 3,
) -> Dict:
    """
    Run all three change point detection algorithms and return:
      - per-algorithm raw change point indices + dates
      - consensus change points (flagged by ≥2 of 3 algorithms)
      - per-segment statistics for consensus segmentation
      - plain-English interpretation per change point

    Algorithms:
      CUSUM       — sustained mean shifts (policy effects, level breaks)
      PELT        — optimal segmentation by cost minimisation (structural breaks)
      Bayesian    — probabilistic online detection (soft shocks, regime changes)

    Consensus rule: a change point at index t is accepted if at least 2 algorithms
    flag a change within ±min_segment_size periods of t.
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    values = pd.to_numeric(df[value_col], errors='coerce').values.astype(float)
    n = len(values)
    date_labels = [str(df.iloc[i][date_col]) for i in range(n)]

    if n < 8:
        return {'error': 'Need at least 8 periods for change point detection'}

    # ── Run 3 algorithms ─────────────────────────────────────────
    cusum_cps = _cusum(values, threshold=cusum_threshold, drift=cusum_drift)
    pelt_cps  = _pelt(values, penalty=pelt_penalty, min_size=min_segment_size)
    bayes_cps = _bayesian_cpd(values, hazard=bayes_hazard)

    cusum_cps = _deduplicate_cps(cusum_cps, min_gap=min_segment_size)
    pelt_cps  = _deduplicate_cps(pelt_cps,  min_gap=min_segment_size)
    bayes_cps = _deduplicate_cps(bayes_cps, min_gap=min_segment_size)

    def _cps_to_dicts(cps, algo):
        return [{'index': int(cp), 'period': date_labels[cp], 'algorithm': algo}
                for cp in cps if 0 < cp < n]

    # ── Consensus: ≥2 algorithms agree within tolerance ──────────
    tol = max(5, min_segment_size * 2)
    all_cps = sorted(set(cusum_cps + pelt_cps + bayes_cps))

    consensus = []
    for cp in all_cps:
        votes = 0
        if any(abs(cp - c) <= tol for c in cusum_cps): votes += 1
        if any(abs(cp - c) <= tol for c in pelt_cps):  votes += 1
        if any(abs(cp - c) <= tol for c in bayes_cps): votes += 1
        if votes >= 2:
            consensus.append({'index': int(cp), 'period': date_labels[cp], 'votes': votes})

    consensus = _deduplicate_cps([c['index'] for c in consensus], min_gap=min_segment_size)
    consensus_idx = sorted(set(consensus))

    # ── Segment statistics on consensus segmentation ─────────────
    segments = _segment_stats(values, consensus_idx, date_labels)

    # ── Per-change-point interpretation ──────────────────────────
    interpretations = []
    for cp in consensus_idx:
        if cp == 0 or cp >= n:
            continue
        before = values[max(0, cp - min_segment_size): cp]
        after  = values[cp: min(n, cp + min_segment_size)]
        if len(before) == 0 or len(after) == 0:
            continue
        delta_mean  = float(np.mean(after) - np.mean(before))
        delta_pct   = float(delta_mean / max(abs(np.mean(before)), 1e-8) * 100)
        delta_vol   = float(np.std(after) - np.std(before))
        direction   = 'Level Up' if delta_mean > 0 else 'Level Down'
        vol_change  = 'Increased volatility' if delta_vol > 0 else 'Decreased volatility'
        interpretations.append({
            'period':       date_labels[cp],
            'index':        int(cp),
            'direction':    direction,
            'delta_mean':   _to_native(delta_mean),
            'delta_pct':    _to_native(delta_pct),
            'delta_vol':    _to_native(delta_vol),
            'vol_change':   vol_change,
            'summary':      (f"{direction} at {date_labels[cp]}: mean shifted by "
                             f"{delta_pct:+.1f}%. {vol_change}."),
        })

    # ── Algorithm agreement matrix ────────────────────────────────
    agreement = {
        'cusum_only':  len([c for c in cusum_cps
                            if not any(abs(c - p) <= tol for p in pelt_cps)
                            and not any(abs(c - b) <= tol for b in bayes_cps)]),
        'pelt_only':   len([p for p in pelt_cps
                            if not any(abs(p - c) <= tol for c in cusum_cps)
                            and not any(abs(p - b) <= tol for b in bayes_cps)]),
        'bayes_only':  len([b for b in bayes_cps
                            if not any(abs(b - c) <= tol for c in cusum_cps)
                            and not any(abs(b - p) <= tol for p in pelt_cps)]),
        'consensus':   len(consensus_idx),
    }

    return {
        'n_consensus':       len(consensus_idx),
        'consensus_cps':     [{'index': int(c), 'period': date_labels[c]}
                               for c in consensus_idx],
        'cusum_cps':         _cps_to_dicts(cusum_cps, 'CUSUM'),
        'pelt_cps':          _cps_to_dicts(pelt_cps,  'PELT'),
        'bayesian_cps':      _cps_to_dicts(bayes_cps, 'Bayesian'),
        'segments':          segments,
        'interpretations':   interpretations,
        'agreement':         agreement,
        'params': {
            'cusum_threshold':   cusum_threshold,
            'cusum_drift':       cusum_drift,
            'bayes_hazard':      bayes_hazard,
            'min_segment_size':  min_segment_size,
        },
    }


def create_change_point_chart(
    df: 'pd.DataFrame',
    date_col: str,
    value_col: str,
    cpd: Dict,
) -> str:
    """
    3-panel change point chart:
      Top    : Time series + vertical lines per consensus CP (+ shaded segments)
      Middle : Algorithm votes — which algo flagged what (scatter by algo)
      Bottom : Per-segment mean bar chart (magnitude of level shifts)
    """
    if cpd.get('error'):
        return ''

    df = df.sort_values(date_col).reset_index(drop=True)
    values = pd.to_numeric(df[value_col], errors='coerce').values
    n = len(values)
    x = np.arange(n)

    consensus_idx = [c['index'] for c in cpd.get('consensus_cps', [])]
    cusum_idx     = [c['index'] for c in cpd.get('cusum_cps',    [])]
    pelt_idx      = [c['index'] for c in cpd.get('pelt_cps',     [])]
    bayes_idx     = [c['index'] for c in cpd.get('bayesian_cps', [])]
    segments      = cpd.get('segments', [])

    fig, axes = plt.subplots(3, 1, figsize=(14, 11),
                             gridspec_kw={'height_ratios': [3, 1.2, 1.5]})

    # ── Panel 1: Time series + CP lines + segment shading ─────
    ax = axes[0]
    ax.plot(x, values, color='#3b82f6', linewidth=1.5, zorder=2, label='Observed')

    # Shaded segments alternating colours
    seg_colors = ['#dbeafe', '#fef3c7', '#d1fae5', '#fce7f3', '#ede9fe']
    for i, seg in enumerate(segments):
        ax.axvspan(seg['start_idx'], seg['end_idx'] + 0.5,
                   alpha=0.25, color=seg_colors[i % len(seg_colors)], zorder=0)
        # Segment mean line
        ax.hlines(seg['mean'], seg['start_idx'], seg['end_idx'],
                  colors='#6b7280', linewidths=1.5, linestyles='--', zorder=3)

    # Consensus CP vertical lines
    for cp in consensus_idx:
        ax.axvline(x=cp, color='#ef4444', linewidth=2.0, zorder=4,
                   label='Consensus CP' if cp == consensus_idx[0] else '')
        # Label with period
        period = cpd['consensus_cps'][[c['index'] for c in cpd['consensus_cps']].index(cp)]['period']
        ax.text(cp + 0.3, ax.get_ylim()[1] if ax.get_ylim()[1] != 1.0 else values.max(),
                period[-7:], fontsize=7, color='#ef4444', va='top', rotation=90)

    ax.set_title('Change Point Detection — Consensus', fontsize=11, fontweight='bold')
    ax.set_ylabel('Value')
    ax.set_xlabel('Period')
    if consensus_idx:
        ax.legend(fontsize=8)

    # ── Panel 2: Algorithm dot plot ────────────────────────────
    ax2 = axes[1]
    algo_map = {'CUSUM': (cusum_idx, '#f59e0b', 'v'),
                'PELT':  (pelt_idx,  '#8b5cf6', 's'),
                'Bayesian': (bayes_idx, '#10b981', 'o')}
    for y_pos, (algo_name, (idx_list, color, marker)) in enumerate(algo_map.items()):
        ax2.scatter(idx_list, [y_pos + 1] * len(idx_list),
                    color=color, marker=marker, s=80, zorder=3,
                    label=f'{algo_name} ({len(idx_list)})')
    for cp in consensus_idx:
        ax2.axvline(x=cp, color='#ef4444', linewidth=1.5, alpha=0.5, zorder=2)

    ax2.set_yticks([1, 2, 3])
    ax2.set_yticklabels(['CUSUM', 'PELT', 'Bayesian'], fontsize=9)
    ax2.set_xlim(axes[0].get_xlim())
    ax2.set_xlabel('Period')
    ax2.set_title('Algorithm Agreement', fontsize=10, fontweight='bold')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(axis='x', alpha=0.3)

    # ── Panel 3: Segment mean bar chart ────────────────────────
    ax3 = axes[2]
    if segments:
        seg_ids    = [f"Seg {s['segment_id']}" for s in segments]
        seg_means  = [s['mean'] for s in segments]
        seg_pcts   = [s['pct_change_from_prev'] or 0 for s in segments]
        bar_colors = ['#10b981' if p >= 0 else '#ef4444' for p in seg_pcts]
        bar_colors[0] = '#6b7280'  # first segment has no prior

        bars = ax3.bar(seg_ids, seg_means, color=bar_colors, alpha=0.8, edgecolor='black')
        # Annotate % change
        for bar, seg in zip(bars, segments):
            pct = seg['pct_change_from_prev']
            if pct is not None:
                ax3.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() * 1.01,
                         f'{pct:+.1f}%', ha='center', fontsize=8,
                         color='#059669' if pct >= 0 else '#dc2626')

        ax3.set_ylabel('Segment Mean')
        ax3.set_title('Level Shifts by Segment', fontsize=10, fontweight='bold')

    plt.tight_layout()
    return _fig_to_base64(fig)


# Step 5: Forecast — multi-model
# ══════════════════════════════════════════════════════════════
def analyze_forecast(df: pd.DataFrame, date_col: str, value_col: str,
                     overview: Dict, model: str = 'auto',
                     n_ahead: int = 6) -> Dict:
    """
    Forecast with three model options:
      linear       — OLS regression on time index (original behaviour)
      holt_winters — Triple exponential smoothing (handles trend + seasonality)
      arima_lite   — AR(p) on first-differenced series (handles autocorrelation)
      auto         — selects best model by in-sample RMSE on last 20% of data
    Reference: Hyndman & Athanasopoulos (2021), Forecasting: Principles and Practice
    """
    df = df.sort_values(date_col).reset_index(drop=True)
    values = pd.to_numeric(df[value_col], errors='coerce').dropna().values
    n = len(values)

    if n < 6:
        return {'error': 'Need at least 6 periods for forecasting'}

    # Detect seasonality (used by auto-select + HW)
    seasonality = detect_seasonality(values)
    season_period = seasonality['period'] if seasonality['has_seasonality'] else 12

    last_date = pd.to_datetime(df[date_col].iloc[-1], errors='coerce')

    def _period_label(i):
        if pd.notna(last_date):
            return (last_date + pd.DateOffset(months=i)).strftime('%Y-%m')
        return f'Period +{i}'

    # ── Linear model ──────────────────────────────────────────
    def _linear_forecast():
        x = np.arange(n)
        slope, intercept, r_value, *_ = stats.linregress(x, values)
        preds = np.array([slope * (n + i) + intercept for i in range(n_ahead)])
        std_res = float(np.std(values - (slope * x + intercept)))
        fitted = slope * x + intercept
        rmse = float(np.sqrt(np.mean((values - fitted) ** 2)))
        return preds, std_res, r_value ** 2, rmse, 'Linear Regression'

    # ── Holt-Winters ──────────────────────────────────────────
    def _hw_forecast():
        preds = _holt_winters(values, season_period, n_ahead)
        if len(preds) == 0:
            return None
        # In-sample: re-fit to get residuals estimate
        n_fit = max(n - n_ahead, n // 2)
        preds_in = _holt_winters(values[:n_fit], season_period, n - n_fit)
        if len(preds_in) == 0:
            return None
        actual_hold = values[n_fit:n_fit + len(preds_in)]
        rmse = float(np.sqrt(np.mean((actual_hold - preds_in) ** 2)))
        std_res = float(np.std(values - np.mean(values)))  # approx
        r2 = max(0.0, 1 - (rmse ** 2) / max(np.var(values), 1e-12))
        return preds, std_res, r2, rmse, 'Holt-Winters (Triple Exponential Smoothing)'

    # ── ARIMA-lite ────────────────────────────────────────────
    def _arima_forecast():
        preds = _arima_lite(values, n_ahead)
        if len(preds) == 0:
            return None
        n_fit = max(n - n_ahead, n // 2)
        preds_in = _arima_lite(values[:n_fit], n - n_fit)
        if len(preds_in) == 0:
            return None
        actual_hold = values[n_fit:n_fit + len(preds_in)]
        rmse = float(np.sqrt(np.mean((actual_hold - preds_in) ** 2)))
        r2 = max(0.0, 1 - (rmse ** 2) / max(np.var(values), 1e-12))
        std_res = float(np.std(np.diff(values)))
        return preds, std_res, r2, rmse, 'ARIMA-lite (AR on differenced series)'

    # ── Auto-select by hold-out RMSE ──────────────────────────
    models_tried = {}
    lin_result  = _linear_forecast()
    models_tried['linear'] = lin_result

    hw_result = _hw_forecast()
    if hw_result:
        models_tried['holt_winters'] = hw_result

    ar_result = _arima_forecast()
    if ar_result:
        models_tried['arima_lite'] = ar_result

    if model == 'auto':
        chosen_key = min(models_tried, key=lambda k: models_tried[k][3])
    elif model in models_tried:
        chosen_key = model
    else:
        chosen_key = 'linear'

    preds, std_res, r2, rmse, model_name = models_tried[chosen_key]

    # Scale CI width by forecast horizon (uncertainty grows)
    ci_factor = 1.96
    forecast_periods_out = []
    for i in range(n_ahead):
        horizon_std = std_res * np.sqrt(1 + i * 0.2)  # widen with horizon
        forecast_periods_out.append({
            'period': _period_label(i + 1),
            'forecast': _to_native(float(preds[i])),
            'lower':    _to_native(float(max(preds[i] - ci_factor * horizon_std, 0))),
            'upper':    _to_native(float(preds[i] + ci_factor * horizon_std)),
        })

    next_val = float(preds[0])
    last_val = float(values[-1])
    forecast_growth = (next_val - last_val) / last_val * 100 if last_val != 0 else 0.0

    forecast_direction = 'Upward' if preds[-1] > preds[0] else ('Downward' if preds[-1] < preds[0] else 'Flat')
    confidence = float(max(min(r2 * 100, 95), 40))

    recommendations = []
    if forecast_direction == 'Upward':
        recommendations.append("Positive trend expected to continue")
        recommendations.append("Consider scaling resources to meet growth")
    elif forecast_direction == 'Downward':
        recommendations.append("Declining trend detected - investigate causes")
        recommendations.append("Review strategy and consider intervention")
    if seasonality['has_seasonality']:
        recommendations.append(f"Seasonal pattern detected (period={season_period}) — plan around peaks/troughs")
    recommendations.append(f"Model: {model_name}. Confidence: {confidence:.0f}%.")

    # Model comparison table
    model_comparison = [
        {'model': k, 'rmse': _to_native(v[3]), 'r_squared': _to_native(v[2]),
         'selected': bool(k == chosen_key)}
        for k, v in models_tried.items()
    ]

    return {
        'forecast_direction':  forecast_direction,
        'forecast_growth':     _to_native(forecast_growth),
        'next_period_value':   _to_native(next_val),
        'forecast_periods':    forecast_periods_out,
        'confidence':          _to_native(confidence),
        'model_r_squared':     _to_native(r2),
        'model_rmse':          _to_native(rmse),
        'model_used':          model_name,
        'model_key':           chosen_key,
        'model_comparison':    model_comparison,
        'seasonality':         seasonality,
        'recommendations':     recommendations,
    }


def create_forecast_chart(df: pd.DataFrame, date_col: str, value_col: str, forecast: Dict) -> str:
    if forecast.get('error'):
        return ""
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    
    df = df.sort_values(date_col).reset_index(drop=True)
    values = pd.to_numeric(df[value_col], errors='coerce')
    
    # Historical data
    ax.plot(range(len(values)), values, marker='o', markersize=4, color='#3b82f6', linewidth=1.5, label='Historical')
    
    # Forecast
    fc_periods = forecast['forecast_periods']
    fc_x = [len(values) + i for i in range(len(fc_periods))]
    fc_vals = [p['forecast'] for p in fc_periods]
    fc_lower = [p['lower'] for p in fc_periods]
    fc_upper = [p['upper'] for p in fc_periods]
    
    ax.plot(fc_x, fc_vals, marker='s', markersize=6, color='#10b981', linewidth=2, linestyle='--', label='Forecast')
    ax.fill_between(fc_x, fc_lower, fc_upper, color='#10b981', alpha=0.2, label='95% CI')
    
    # Connect historical to forecast
    ax.plot([len(values)-1, fc_x[0]], [values.iloc[-1], fc_vals[0]], color='#10b981', linestyle='--', linewidth=1)
    
    ax.set_xlabel('Period')
    ax.set_ylabel('Value')
    ax.set_title(f"Forecast ({forecast['forecast_direction']}, {forecast['confidence']:.0f}% confidence)", fontsize=11, fontweight='bold')
    ax.legend()
    ax.axvline(x=len(values)-0.5, color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# Report Generation
def generate_report(overview: Dict, comparison: Dict, correlation: Dict, anomaly: Dict, forecast: Dict, cpd: Dict = None) -> Dict:
    report = {}
    
    report['step1_overview'] = {
        'title': '1. Trend Overview',
        'question': 'What is the overall trend?',
        'finding': f"Trend: {overview['trend_direction']}, Growth: {overview['growth_rate']:.1f}%",
        'detail': f"Analysis of {overview['n_periods']} periods shows a {overview['trend_direction'].lower()} trend. "
                 f"Values grew from {overview['start_value']:,.0f} to {overview['end_value']:,.0f} ({overview['growth_rate']:.1f}% total). "
                 f"Average period-over-period change: {overview['avg_change']:.1f}%. Volatility (std): {overview['volatility']:.1f}."
    }
    
    if comparison and not comparison.get('error'):
        report['step2_comparison'] = {
            'title': '2. Period Comparison (YoY / MoM)',
            'question': 'How does it compare to previous periods?',
            'finding': f"Latest YoY: {comparison['latest_yoy']:.1f}%, Latest MoM: {comparison['latest_mom']:.1f}%" if comparison.get('latest_yoy') is not None and comparison.get('latest_mom') is not None else 'Insufficient data for period comparison',
            'detail': (
                f"Year-over-year analysis shows average growth of {comparison['avg_yoy']:.1f}% with latest at {comparison['latest_yoy']:.1f}%. "
                f"Month-over-month shows {comparison['avg_mom']:.1f}% average change. "
                f"Positive YoY rate: {comparison['yoy_positive_rate']:.0f}% of periods."
                if all(comparison.get(k) is not None for k in ['avg_yoy','latest_yoy','avg_mom','yoy_positive_rate'])
                else 'Insufficient date history for YoY/MoM comparison.'
            )
        }
    else:
        report['step2_comparison'] = {'title': '2. Period Comparison', 'question': 'Period comparison', 'finding': comparison.get('error', 'Not available'), 'detail': ''}
    
    if correlation and not correlation.get('error'):
        top = correlation.get('top_driver', {})
        lag_note = (f" Best effect at lag {top.get('best_lag', 0)}: {top.get('lag_interpretation', '')}."
                    if top.get('best_lag', 0) != 0 else "")
        report['step3_correlation'] = {
            'title': '3. External Factor Correlation (with Lag)',
            'question': 'What drives the changes, and with what delay?',
            'finding': f"Top driver: {top.get('factor', 'N/A')} (best r={top.get('best_correlation', top.get('correlation', 0)):.3f}, lag={top.get('best_lag', 0)})",
            'detail': (f"Cross-correlation analysis (lag 0–{correlation.get('max_lag_tested', 6)}) found "
                       f"{correlation['n_significant']} significant relationships. "
                       f"'{top.get('factor', 'N/A')}' shows {top.get('best_strength', 'N/A').lower()} "
                       f"{'positive' if (top.get('best_correlation') or 0) > 0 else 'negative'} correlation "
                       f"(r={top.get('best_correlation', top.get('correlation', 0)):.3f}).{lag_note}")
        }
    else:
        report['step3_correlation'] = {'title': '3. External Factors', 'question': 'External drivers', 'finding': correlation.get('error', 'No external factors provided'), 'detail': ''}
    
    report['step4_anomaly'] = {
        'title': '4. Anomaly & Inflection Detection',
        'question': 'Are there any unusual points?',
        'finding': f"Detected {anomaly['n_anomalies']} anomalies and {anomaly['n_inflections']} inflection points",
        'detail': f"Using {anomaly['threshold']}σ threshold, {anomaly['n_anomalies']} anomalies detected ({anomaly['anomaly_rate']:.1f}% of data). "
                 f"{anomaly['n_inflections']} trend inflection points identified where direction changed significantly."
    }
    
    # Step 6: Change Points
    if cpd and not cpd.get('error'):
        n_cp = cpd.get('n_consensus', 0)
        segs = cpd.get('segments', [])
        algos = cpd.get('agreement', {})
        finding = (f"{n_cp} consensus change point(s) detected" if n_cp > 0
                   else "No significant change points detected")
        if n_cp > 0 and cpd.get('interpretations'):
            first = cpd['interpretations'][0]
            detail_extra = f" Most significant: {first['summary']}"
        else:
            detail_extra = ""
        report['step6_change_points'] = {
            'title': '6. Change Point Detection',
            'question': 'When did the trend fundamentally shift?',
            'finding': finding,
            'detail': (f"CUSUM/PELT/Bayesian analysis found {n_cp} consensus structural break(s) "
                       f"across {len(segs)} segments.{detail_extra} "
                       f"Agreement: CUSUM-only={algos.get('cusum_only',0)}, "
                       f"PELT-only={algos.get('pelt_only',0)}, "
                       f"Bayesian-only={algos.get('bayes_only',0)}, "
                       f"Consensus={algos.get('consensus',0)}."),
        }
    else:
        report['step6_change_points'] = {
            'title': '6. Change Point Detection',
            'question': 'When did the trend fundamentally shift?',
            'finding': cpd.get('error', 'Not available') if cpd else 'Not run',
            'detail': '',
        }

    if forecast and not forecast.get('error'):
        report['step5_forecast'] = {
            'title': '5. Future Trend Forecast',
            'question': 'What will happen next?',
            'finding': f"Forecast: {forecast['forecast_direction']} ({forecast['forecast_growth']:.1f}% expected)",
            'detail': f"Based on historical patterns, the trend is expected to continue {forecast['forecast_direction'].lower()}. "
                     f"Next period forecast: {forecast['next_period_value']:,.0f} ({forecast['forecast_growth']:.1f}% change). "
                     f"Model confidence: {forecast['confidence']:.0f}%."
        }
    else:
        report['step5_forecast'] = {'title': '5. Forecast', 'question': 'Future prediction', 'finding': forecast.get('error', 'Not available'), 'detail': ''}
    
    return report


def generate_insights(overview: Dict, comparison: Dict, correlation: Dict, anomaly: Dict, forecast: Dict, cpd: Dict = None) -> List[Dict]:
    insights = []
    
    if overview['trend_direction'] == 'Upward':
        insights.append({'title': 'Positive Trend', 'description': f"Overall {overview['growth_rate']:.1f}% growth observed.", 'status': 'positive'})
    elif overview['trend_direction'] == 'Downward':
        insights.append({'title': 'Declining Trend', 'description': f"Overall {abs(overview['growth_rate']):.1f}% decline observed.", 'status': 'warning'})
    
    if comparison and not comparison.get('error'):
        if comparison['latest_yoy'] and comparison['latest_yoy'] > 10:
            insights.append({'title': 'Strong YoY Growth', 'description': f"Latest YoY growth of {comparison['latest_yoy']:.1f}%.", 'status': 'positive'})
        elif comparison['latest_yoy'] and comparison['latest_yoy'] < -10:
            insights.append({'title': 'YoY Decline', 'description': f"Latest YoY decline of {abs(comparison['latest_yoy']):.1f}%.", 'status': 'warning'})
    
    if anomaly['n_anomalies'] > 0:
        insights.append({'title': f"{anomaly['n_anomalies']} Anomalies Detected", 'description': "Unusual data points require investigation.", 'status': 'warning'})
    
    if cpd and not cpd.get('error') and cpd.get('n_consensus', 0) > 0:
        n = cpd['n_consensus']
        periods = ', '.join(c['period'] for c in cpd['consensus_cps'][:3])
        insights.append({
            'title': f"{n} Structural Break{'s' if n > 1 else ''} Detected",
            'description': f"Consensus change point(s) at: {periods}. Investigate policy/market events.",
            'status': 'warning',
        })

    if correlation and not correlation.get('error') and correlation.get('top_driver'):
        top = correlation['top_driver']
        insights.append({'title': f"Key Driver: {top['factor']}", 'description': f"{top['strength']} correlation (r={top['correlation']:.3f}).", 'status': 'neutral'})
    
    return insights


@router.post("/trend-analysis")
async def analyze_trend(request: TrendRequest):
    try:
        df = pd.DataFrame(request.data)
        if len(df) < 6:
            raise HTTPException(status_code=400, detail="Need at least 6 periods")
        
        results, visualizations = {}, {}
        
        # Step 1: Overview
        overview = analyze_overview(df, request.date_col, request.value_col)
        results['overview'] = overview
        visualizations['overview_chart'] = create_overview_chart(df, request.date_col, request.value_col, overview)
        
        # Step 2: Comparison
        comparison = analyze_comparison(df, request.date_col, request.value_col)
        results['comparison'] = comparison
        if not comparison.get('error'):
            visualizations['comparison_chart'] = create_comparison_chart(df, request.date_col, comparison)
        
        # Step 3: Correlation
        correlation = {}
        if request.external_cols:
            correlation = analyze_correlation(df, request.value_col, request.external_cols, max_lag=request.max_lag)
            results['correlation'] = correlation
            if not correlation.get('error'):
                visualizations['correlation_chart'] = create_correlation_chart(df, request.value_col, correlation)
        
        # Step 6: Change Point Detection
        cpd = analyze_change_points(
            df, request.date_col, request.value_col,
            cusum_threshold=request.cusum_threshold,
            cusum_drift=request.cusum_drift,
            pelt_penalty=request.pelt_penalty,
            bayes_hazard=request.bayes_hazard,
            min_segment_size=request.min_segment_size,
        )
        results['change_points'] = cpd
        if not cpd.get('error'):
            visualizations['change_point_chart'] = create_change_point_chart(
                df, request.date_col, request.value_col, cpd)

        # Step 4: Anomaly
        anomaly = analyze_anomaly(df, request.date_col, request.value_col)
        results['anomaly'] = anomaly
        visualizations['anomaly_chart'] = create_anomaly_chart(df, request.date_col, request.value_col, anomaly)
        
        # Step 5: Forecast
        forecast = analyze_forecast(df, request.date_col, request.value_col, overview, model=request.forecast_model, n_ahead=request.forecast_periods)
        results['forecast'] = forecast
        if not forecast.get('error'):
            visualizations['forecast_chart'] = create_forecast_chart(df, request.date_col, request.value_col, forecast)
        
        report = generate_report(overview, comparison, correlation, anomaly, forecast, cpd=cpd)
        insights = generate_insights(overview, comparison, correlation, anomaly, forecast, cpd=cpd)
        
        summary = {
            'n_periods': overview['n_periods'],
            'overall_trend': overview['trend_direction'],
            'growth_rate': overview['growth_rate'],
            'n_anomalies': anomaly['n_anomalies'],
            'forecast_direction': forecast.get('forecast_direction', 'N/A') if not forecast.get('error') else 'N/A',
            'n_change_points': cpd.get('n_consensus', 0) if cpd and not cpd.get('error') else 0
        }
        
        return _to_native({'success': True, 'results': results, 'visualizations': visualizations, 'report': report, 'key_insights': insights, 'summary': summary})
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
