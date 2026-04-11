from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, List, Optional, Union
import numpy as np
import pandas as pd
import io, base64, warnings
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

from scipy.signal import periodogram, find_peaks
from scipy.fft import fft, fftfreq
from scipy.stats import shapiro, normaltest, skew, kurtosis, norm as sp_norm, probplot
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.stattools import acf as sm_acf

warnings.filterwarnings('ignore')

router = APIRouter()


# ══════════════════════════════════════════════════════════════════
# Request
# ══════════════════════════════════════════════════════════════════

class SeasonalAnalysisRequest(BaseModel):
    data:               List[dict]             = Field(...)
    variable:           Union[str, List[str]]  = Field(...)
    period:             Optional[int]          = 12
    test_periods:       Optional[List[int]]    = []
    auto_detect:        Optional[bool]         = True
    decomposition_type: Optional[str]          = 'auto'   # additive|multiplicative|auto


# ══════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════

def _to_native(obj):
    if isinstance(obj, np.integer):   return int(obj)
    elif isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj): return None
        return float(obj)
    elif isinstance(obj, np.ndarray): return [_to_native(x) for x in obj.tolist()]
    elif isinstance(obj, np.bool_):   return bool(obj)
    elif isinstance(obj, pd.Timestamp): return obj.isoformat()
    elif isinstance(obj, dict):       return {str(k): _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)): return [_to_native(x) for x in obj]
    try:
        if pd.isna(obj): return None
    except Exception:
        pass
    return obj


def safe_float(val, default=0.0):
    try:
        if val is None or pd.isna(val) or np.isinf(val): return default
        return float(val)
    except:
        return default


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"


# ══════════════════════════════════════════════════════════════════
# ① True STL decomposition (LOESS-based)
# ══════════════════════════════════════════════════════════════════

def _stl_core(series: pd.Series, period: int, robust: bool = True):
    """
    statsmodels STL — true LOESS-based decomposition.
    Falls back to classical MA decomposition for very short series.
    Ref: Cleveland et al. (1990), J. Official Statistics 6(1):3-73.
    """
    n = len(series)
    if n < 2 * period:
        res = seasonal_decompose(series, model='additive', period=period,
                                 extrapolate_trend='freq')
        return (pd.Series(res.trend,    index=series.index),
                pd.Series(res.seasonal, index=series.index),
                pd.Series(res.resid,    index=series.index))
    stl = STL(series, period=period, robust=robust)
    fit = stl.fit()
    return (pd.Series(fit.trend,    index=series.index),
            pd.Series(fit.seasonal, index=series.index),
            pd.Series(fit.resid,    index=series.index))


def decompose_series(series: pd.Series, period: int, decomp_type: str):
    """
    Wrapper: additive STL or multiplicative (log-space STL then exponentiate).
    """
    if decomp_type == 'multiplicative' and (series > 0).all():
        log_s = np.log(series)
        t_log, s_log, r_log = _stl_core(log_s, period)
        trend    = np.exp(t_log)
        seasonal = np.exp(s_log)
        residual = series / (trend * seasonal + 1e-12)
        return (pd.Series(trend.values,    index=series.index),
                pd.Series(seasonal.values, index=series.index),
                pd.Series(residual.values, index=series.index))
    return _stl_core(series, period)


# ══════════════════════════════════════════════════════════════════
# ② Auto period detection — ACF + periodogram + Fourier ensemble
# ══════════════════════════════════════════════════════════════════

def detect_periods_ensemble(series: pd.Series, max_period: int = None, top_k: int = 3):
    """
    Ensemble period detection: ACF peaks, periodogram peaks, FFT amplitude peaks.
    A period earns a vote from each method that nominates it (±2 tolerance).
    Consensus (votes>=2) → strong candidate.

    Returns
    -------
    dict: dominant, candidates, acf_periods, periodogram_periods, fourier_periods,
          freqs (periodogram), power (periodogram)
    """
    n = len(series)
    if max_period is None:
        max_period = min(n // 2, 200)

    # De-trend via OLS
    x = np.arange(n, dtype=float)
    coeffs = np.polyfit(x, series.values, 1)
    resid  = series.values - np.polyval(coeffs, x)

    # Method 1: ACF peaks
    max_lags = min(max_period, n // 2 - 1)
    acf_vals = sm_acf(resid, nlags=max_lags, fft=True)
    ci = 1.96 / np.sqrt(n)
    acf_peak_idx, _ = find_peaks(acf_vals[1:], height=ci, distance=2)
    acf_peak_idx += 1
    acf_periods = sorted(acf_peak_idx.tolist(),
                         key=lambda p: acf_vals[p], reverse=True)[:top_k]

    # Method 2: Periodogram
    freqs, power = periodogram(resid, scaling='spectrum')
    valid = (freqs > 0) & (freqs < 0.5)
    f_v   = freqs[valid]; p_v = power[valid]
    pg_peaks, _ = find_peaks(p_v, height=np.percentile(p_v, 80))
    pg_periods, seen = [], set()
    for i in sorted(pg_peaks, key=lambda i: -p_v[i]):
        per = int(round(1 / f_v[i])) if f_v[i] > 0 else None
        if per and 2 <= per <= max_period and per not in seen:
            pg_periods.append(per); seen.add(per)
        if len(pg_periods) >= top_k: break

    # Method 3: FFT amplitudes
    yf   = np.abs(fft(resid)); xf = fftfreq(n)
    pos  = xf > 0
    xf_p = xf[pos]; yf_p = yf[pos]
    ft_peaks, _ = find_peaks(yf_p, height=np.percentile(yf_p, 80))
    ft_periods, seen2 = [], set()
    for i in sorted(ft_peaks, key=lambda i: -yf_p[i]):
        per = int(round(1 / xf_p[i])) if xf_p[i] > 0 else None
        if per and 2 <= per <= max_period and per not in seen2:
            ft_periods.append(per); seen2.add(per)
        if len(ft_periods) >= top_k: break

    # Ensemble voting (tolerance ±2)
    TOL = 2
    all_cands = set(acf_periods) | set(pg_periods) | set(ft_periods)
    scored, counted = [], set()
    for p in sorted(all_cands):
        if any(abs(p - c) <= TOL for c in counted): continue
        counted.add(p)
        methods, votes = [], 0
        if any(abs(p - q) <= TOL for q in acf_periods):
            methods.append('ACF');         votes += 1
        if any(abs(p - q) <= TOL for q in pg_periods):
            methods.append('Periodogram'); votes += 1
        if any(abs(p - q) <= TOL for q in ft_periods):
            methods.append('Fourier');     votes += 1
        acf_score = float(acf_vals[min(p, len(acf_vals) - 1)])
        scored.append({'period': p, 'score': round(votes + max(acf_score, 0), 4),
                       'votes': votes, 'methods': methods})

    scored.sort(key=lambda c: (-c['votes'], -c['score']))
    dominant = scored[0]['period'] if scored else None

    return {'dominant': dominant, 'candidates': scored[:5],
            'acf_periods': acf_periods,
            'periodogram_periods': pg_periods,
            'fourier_periods': ft_periods,
            'freqs': freqs, 'power': power}


# ══════════════════════════════════════════════════════════════════
# ③ Multi-seasonality — iterative STL per period
# ══════════════════════════════════════════════════════════════════

def multi_seasonal_decompose(series: pd.Series, periods: List[int]):
    """
    Iterative STL for multiple periods (largest first).
    After extracting each seasonal component, remove it before the next pass.
    Ref: Taylor & Letham (2018), Forecasting at Scale (Prophet).
    """
    periods_sorted = sorted(set(periods), reverse=True)
    current   = series.copy()
    components = []
    base_trend = None

    for i, s in enumerate(periods_sorted):
        if len(current) < 2 * s or s < 2: continue
        try:
            trend, seasonal, residual = _stl_core(current, period=s)
            var_r  = float(np.nanvar(residual))
            var_dt = float(np.nanvar(current - trend))
            strength = max(0.0, 1 - var_r / max(var_dt, 1e-12))
            components.append({'period': s, 'seasonal': seasonal,
                                'trend': trend if i == 0 else None,
                                'strength': round(strength, 4)})
            current = current - seasonal
            if i == 0: base_trend = trend
        except Exception:
            continue

    final_residual = current - (base_trend if base_trend is not None
                                else current.rolling(3, center=True, min_periods=1).mean())
    return components, final_residual, base_trend


# ══════════════════════════════════════════════════════════════════
# ⑤ Additive vs multiplicative selection
# ══════════════════════════════════════════════════════════════════

def choose_decomposition_type(series: pd.Series, period: int) -> str:
    """
    Compare residual CV between additive and log-additive (=multiplicative) STL.
    Lower CV wins. Ref: Hyndman & Athanasopoulos (2021) FPP Ch.3.
    """
    if (series <= 0).any(): return 'additive'

    def _cv(s):
        try:
            _, _, res = _stl_core(s, period)
            res_c = res.dropna()
            return float(np.std(res_c)) / (abs(float(np.mean(s))) + 1e-8)
        except:
            return np.inf

    return 'multiplicative' if _cv(np.log(series)) < _cv(series) * 0.95 else 'additive'


# ══════════════════════════════════════════════════════════════════
# Strength indices
# ══════════════════════════════════════════════════════════════════

def seasonal_strength(trend, seasonal, residual) -> float:
    var_r  = float(np.nanvar(residual))
    var_sa = float(np.nanvar(residual + seasonal))
    return max(0.0, 1 - var_r / max(var_sa, 1e-12))


def trend_strength(trend, seasonal, residual) -> float:
    var_r  = float(np.nanvar(residual))
    var_dt = float(np.nanvar(residual + trend))
    return max(0.0, 1 - var_r / max(var_dt, 1e-12))


# ══════════════════════════════════════════════════════════════════
# Seasonal indices
# ══════════════════════════════════════════════════════════════════

def calculate_seasonal_indices(series: pd.Series, period: int, decomp_type: str):
    n = len(series)
    ma = series.rolling(window=period, center=True, min_periods=period // 2).mean()
    ratios = (series / ma) if decomp_type == 'multiplicative' and (series > 0).all()              else (series - ma)
    indices = []
    for i in range(period):
        vals = [float(ratios.iloc[j]) for j in range(i, n, period)
                if not np.isnan(ratios.iloc[j])]
        indices.append({'position': i + 1,
                        'index': safe_float(np.mean(vals) if vals else (1.0 if decomp_type == 'multiplicative' else 0.0)),
                        'std':   safe_float(np.std(vals)  if len(vals) > 1 else 0.0),
                        'n':     len(vals)})
    mean_idx = np.mean([s['index'] for s in indices])
    if abs(mean_idx) > 1e-8:
        for s in indices: s['index'] /= mean_idx
    return indices


# ══════════════════════════════════════════════════════════════════
# ④ Residual diagnostics
# ══════════════════════════════════════════════════════════════════

def residual_diagnostics(residual: pd.Series, period: int) -> dict:
    """
    Shapiro-Wilk, D'Agostino-Pearson normality tests + residual ACF.
    """
    r = residual.dropna().values
    n = len(r)
    out = {'mean': safe_float(np.mean(r)),
           'std':      safe_float(np.std(r)),
           'skewness': safe_float(float(skew(r))),
           'kurtosis': safe_float(float(kurtosis(r)))}

    if n >= 8:
        try:
            sw, sw_p = shapiro(r[:5000])
            out['shapiro_wilk'] = {
                'statistic': safe_float(sw), 'p_value': safe_float(sw_p),
                'normal':    bool(sw_p > 0.05),
                'conclusion': ('Normal (p={:.4f})'.format(sw_p) if sw_p > 0.05
                               else 'Non-normal (p={:.4f})'.format(sw_p))}
        except Exception as e:
            out['shapiro_wilk'] = {'error': str(e)}
        try:
            dp, dp_p = normaltest(r)
            out['dagostino_pearson'] = {
                'statistic': safe_float(dp), 'p_value': safe_float(dp_p),
                'normal': bool(dp_p > 0.05)}
        except Exception as e:
            out['dagostino_pearson'] = {'error': str(e)}
    else:
        out['normality_note'] = 'n<8, normality test skipped'

    max_lags_res = min(period * 2, n // 2 - 1, 40)
    if max_lags_res >= 1:
        try:
            res_acf = sm_acf(r, nlags=max_lags_res, fft=True)
            ci_r = 1.96 / np.sqrt(n)
            sig = [int(i) for i, v in enumerate(res_acf) if i > 0 and abs(v) > ci_r]
            out['acf'] = {
                'values': [safe_float(v) for v in res_acf],
                'confidence_interval': safe_float(ci_r),
                'significant_lags': sig,
                'has_autocorrelation': len(sig) > 0,
                'conclusion': (f'Sig. autocorrelation at lags {sig[:5]} — decomposition may be incomplete'
                               if sig else 'No sig. autocorrelation — residuals are white noise')}
        except Exception as e:
            out['acf'] = {'error': str(e)}

    issues = []
    if abs(out.get('skewness', 0)) > 1: issues.append('high skewness')
    if out.get('kurtosis', 0) > 3:      issues.append('heavy tails')
    if not out.get('shapiro_wilk', {}).get('normal', True): issues.append('non-normal')
    if out.get('acf', {}).get('has_autocorrelation', False): issues.append('autocorrelated residuals')
    out['verdict'] = 'Clean residuals' if not issues else 'Issues: ' + ', '.join(issues)
    return out


# ══════════════════════════════════════════════════════════════════
# Fourier components
# ══════════════════════════════════════════════════════════════════

def calculate_fourier_components(series: pd.Series, n_components: int = 5):
    n = len(series)
    x = np.arange(n, dtype=float)
    resid = series.values - np.polyval(np.polyfit(x, series.values, 1), x)
    yf = np.abs(fft(resid)); xf = fftfreq(n)
    pos = xf > 0; xf_p = xf[pos]; yf_p = yf[pos]
    top = np.argsort(yf_p)[-n_components:][::-1]
    comps = []
    for idx in top:
        freq = float(xf_p[idx])
        if freq > 0:
            p = 1 / freq
            if p < n / 2:
                comps.append({'period': safe_float(p), 'frequency': safe_float(freq),
                              'amplitude': safe_float(float(yf_p[idx])),
                              'power':     safe_float(float(yf_p[idx] ** 2))})
    return comps


# ══════════════════════════════════════════════════════════════════
# Analyze multiple candidate periods
# ══════════════════════════════════════════════════════════════════

def analyze_multiple_periods(series: pd.Series, periods: List[int], decomp_type: str):
    results = []
    for p in periods:
        if p >= len(series) // 2 or p < 2: continue
        try:
            tr, sea, res = decompose_series(series, p, decomp_type)
            ssi = seasonal_strength(tr, sea, res)
            tsi = trend_strength(tr, sea, res)
            results.append({'period': p,
                            'seasonal_strength': safe_float(ssi),
                            'trend_strength':    safe_float(tsi),
                            'combined_strength': safe_float((ssi + tsi) / 2)})
        except Exception:
            continue
    return results


# ══════════════════════════════════════════════════════════════════
# Insights
# ══════════════════════════════════════════════════════════════════

def generate_insights(ssi, tsi, period_detect, seasonal_indices,
                      period_results, decomp_type, residual_diag, multi_components):
    insights, recommendations = [], []

    level = 'Strong' if ssi > 0.7 else 'Moderate' if ssi > 0.4 else 'Weak'
    insights.append({'type': 'success' if ssi > 0.7 else 'info' if ssi > 0.4 else 'warning',
                     'title': f'{level} Seasonality (SSI={ssi:.3f})',
                     'description': (f'STL (LOESS) decomposition: SSI={ssi:.3f} → {level.lower()} seasonal pattern.')})

    if tsi is not None:
        tl = 'Strong' if tsi > 0.7 else 'Moderate' if tsi > 0.4 else 'Weak'
        insights.append({'type': 'success' if tsi > 0.7 else 'info',
                         'title': f'{tl} Trend (TSI={tsi:.3f})',
                         'description': f'Trend Strength Index {tsi:.3f}.'})

    if period_detect and period_detect.get('dominant'):
        dom = period_detect['dominant']
        cand = next((c for c in period_detect.get('candidates', []) if c['period'] == dom), None)
        v_str = f" ({cand['votes']}/3 methods: {', '.join(cand['methods'])})" if cand else ''
        insights.append({'type': 'info', 'title': f'Dominant Period: {dom}',
                         'description': f'ACF+Periodogram+Fourier ensemble{v_str}.'})

    insights.append({'type': 'info', 'title': f'Decomposition: {decomp_type.capitalize()}',
                     'description': ('Log-space STL (lower residual CV).' if decomp_type == 'multiplicative'
                                     else 'Additive STL (LOESS).')})

    if multi_components and len(multi_components) > 1:
        pstr = ', '.join(str(c['period']) for c in multi_components)
        insights.append({'type': 'info', 'title': f'Multiple Periods: [{pstr}]',
                         'description': f'{len(multi_components)} seasonal cycles via iterative STL.'})

    if residual_diag:
        verdict = residual_diag.get('verdict', '')
        insights.append({'type': 'success' if 'Clean' in verdict else 'warning',
                         'title': 'Residual Diagnostics', 'description': verdict})

    if seasonal_indices:
        mx = max(seasonal_indices, key=lambda x: x['index'])
        mn = min(seasonal_indices, key=lambda x: x['index'])
        insights.append({'type': 'info', 'title': 'Seasonal Range',
                         'description': (f"Peak pos {mx['position']} ({mx['index']:.2f}), "
                                         f"trough pos {mn['position']} ({mn['index']:.2f}).")})

    if ssi > 0.4:
        recommendations.append('Include seasonal components in forecasting (SARIMA, Prophet, ETS).')
        recommendations.append('Apply seasonal differencing for stationarity.')
    else:
        recommendations.append('Seasonality may not significantly improve forecasts.')

    if decomp_type == 'multiplicative':
        recommendations.append('Log-transform before ARIMA (multiplicative pattern).')
    if residual_diag and residual_diag.get('acf', {}).get('has_autocorrelation'):
        recommendations.append('Residual autocorrelation detected — add AR/MA terms.')
    if residual_diag and not residual_diag.get('shapiro_wilk', {}).get('normal', True):
        recommendations.append('Non-normal residuals — consider robust methods or transformations.')
    if multi_components and len(multi_components) > 1:
        recommendations.append('Multiple seasons → use TBATS or Prophet for multi-seasonal forecasting.')
    recommendations += ['Validate detected period with domain knowledge.',
                        'Consider Fourier terms for complex seasonality.']
    return insights, recommendations


# ══════════════════════════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════════════════════════

def create_plots(series, trend, seasonal, residual, period, variable,
                 ssi, tsi, seasonal_indices, period_detect, period_results,
                 decomp_type, residual_diag, multi_components):
    plots = {}
    x = np.arange(len(series))

    # 1. Decomposition
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
    fig.suptitle(f'STL Decomposition: {variable} — {decomp_type.capitalize()}, period={period}',
                 fontsize=13, fontweight='bold')
    axes[0].plot(x, series.values,  '#4C72B0', linewidth=1); axes[0].set_ylabel('Original')
    axes[1].plot(x, trend.values,   '#55A868', linewidth=2); axes[1].set_ylabel('Trend (LOESS)')
    axes[2].plot(x, seasonal.values,'#DD8452', linewidth=1); axes[2].set_ylabel('Seasonal')
    baseline = 1 if decomp_type == 'multiplicative' else 0
    axes[2].axhline(baseline, color='gray', linestyle='--', linewidth=0.5)
    axes[3].plot(x, residual.values,'#C44E52', linewidth=1, alpha=0.8); axes[3].set_ylabel('Residual')
    axes[3].axhline(baseline, color='gray', linestyle='--', linewidth=0.5); axes[3].set_xlabel('Index')
    for ax in axes: ax.grid(True, alpha=0.3)
    plt.tight_layout(); plots['decomposition'] = _fig_to_b64(fig)

    # 2. Seasonal pattern
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    positions = [s['position'] for s in seasonal_indices]
    idx_vals  = [s['index']    for s in seasonal_indices]
    stds      = [s['std']      for s in seasonal_indices]
    bl = 1.0 if decomp_type == 'multiplicative' else 0.0
    colors = ['#55A868' if v >= bl else '#C44E52' for v in idx_vals]
    axes[0].bar(positions, idx_vals, color=colors, alpha=0.7, edgecolor='white')
    axes[0].axhline(bl, color='gray', linestyle='--', linewidth=1.5)
    axes[0].errorbar(positions, idx_vals, yerr=stds, fmt='none', color='black', capsize=3)
    axes[0].set_title(f'Seasonal Indices ({decomp_type.capitalize()}, period={period})',
                      fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Position in Cycle'); axes[0].set_ylabel('Index')
    axes[0].set_xticks(positions); axes[0].grid(True, alpha=0.3, axis='y')
    theta = np.append(np.linspace(0, 2*np.pi, period, endpoint=False), 0)
    i_circ = idx_vals + [idx_vals[0]]
    ax_p = plt.subplot(122, projection='polar')
    ax_p.plot(theta, i_circ, 'b-', linewidth=2); ax_p.fill(theta, i_circ, alpha=0.25)
    ax_p.set_title('Seasonal Pattern (Polar)', fontsize=12, fontweight='bold')
    plt.tight_layout(); plots['seasonal_pattern'] = _fig_to_b64(fig)

    # 3. Strength gauge
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, val, title, color in zip(axes, [ssi, tsi], ['Seasonal Strength','Trend Strength'],
                                     ['#4C72B0','#55A868']):
        th = np.linspace(0, np.pi, 100)
        ax.plot(np.cos(th), np.sin(th), 'lightgray', linewidth=20, solid_capstyle='round')
        if val is not None:
            tv = np.linspace(0, np.pi * float(val), 100)
            ax.plot(np.cos(tv), np.sin(tv), color, linewidth=20, solid_capstyle='round')
        ax.text(0,  0.3, f'{val:.2f}' if val is not None else 'N/A',
                ha='center', va='center', fontsize=24, fontweight='bold')
        ax.text(0, -0.1, title, ha='center', va='center', fontsize=12)
        if val is not None:
            ax.text(0, -0.3, 'Strong' if val > 0.7 else 'Moderate' if val > 0.4 else 'Weak',
                    ha='center', va='center', fontsize=11, color='gray')
        ax.set_xlim(-1.5,1.5); ax.set_ylim(-0.5,1.3); ax.set_aspect('equal'); ax.axis('off')
    plt.tight_layout(); plots['gauge'] = _fig_to_b64(fig)

    # 4. Ensemble period detection (3-panel)
    if period_detect:
        freqs_raw = period_detect.get('freqs')
        power_raw = period_detect.get('power')
        if freqs_raw is not None and power_raw is not None:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            fig.suptitle('Period Detection: ACF · Periodogram · Fourier Ensemble',
                         fontsize=13, fontweight='bold')
            mask = freqs_raw > 0; f_v = freqs_raw[mask]; p_v = power_raw[mask]
            axes[0].semilogy(f_v, p_v, 'b-', linewidth=1)
            dom = period_detect.get('dominant')
            if dom:
                axes[0].axvline(1/dom, color='red', linestyle='--', linewidth=2, label=f'T={dom}')
            for c in period_detect.get('candidates', [])[:3]:
                if c['period'] != dom and c['period'] > 0:
                    axes[0].axvline(1/c['period'], color='orange', linestyle=':', linewidth=1.2,
                                    alpha=0.7, label=f"T={c['period']}")
            axes[0].set_xlabel('Frequency'); axes[0].set_ylabel('Power (log)')
            axes[0].set_title('Periodogram'); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)

            periods_arr = 1 / f_v; vlim = periods_arr < len(power_raw) / 2
            axes[1].plot(periods_arr[vlim], p_v[vlim], 'g-', linewidth=1)
            if dom: axes[1].axvline(dom, color='red', linestyle='--', linewidth=2, label=f'T={dom}')
            axes[1].set_xlabel('Period'); axes[1].set_title('Power by Period')
            axes[1].set_xlim(0, min(100, len(power_raw)//2)); axes[1].legend(fontsize=8)
            axes[1].grid(True, alpha=0.3)

            cands = period_detect.get('candidates', [])[:6]
            if cands:
                c_per = [str(c['period']) for c in cands]; c_sc = [c['score'] for c in cands]
                c_vo  = [c['votes']  for c in cands]
                bar_c = ['#2ecc71' if v >= 2 else '#3498db' if v == 1 else '#e74c3c' for v in c_vo]
                axes[2].barh(c_per[::-1], c_sc[::-1], color=bar_c[::-1], alpha=0.8, edgecolor='white')
                axes[2].set_xlabel('Ensemble Score')
                axes[2].set_title('Candidates (green=≥2 methods)')
                axes[2].grid(True, alpha=0.3, axis='x')
                from matplotlib.patches import Patch
                axes[2].legend(handles=[Patch(color='#2ecc71', label='≥2 methods'),
                                        Patch(color='#3498db', label='1 method')], fontsize=8)
            plt.tight_layout(); plots['period_detection'] = _fig_to_b64(fig)

    # 5. Multi-seasonality
    if multi_components and len(multi_components) > 1:
        n_c = len(multi_components)
        fig, axes = plt.subplots(n_c + 1, 1, figsize=(14, 4*(n_c+1)), sharex=True)
        fig.suptitle(f'Multi-Seasonal Decomposition ({variable})', fontsize=13, fontweight='bold')
        axes[0].plot(x, series.values, '#4C72B0', linewidth=1)
        axes[0].set_title('Original'); axes[0].grid(True, alpha=0.3)
        for i, comp in enumerate(multi_components, 1):
            axes[i].plot(x, comp['seasonal'].values, linewidth=1.2)
            axes[i].axhline(0, color='gray', linestyle='--', linewidth=0.5)
            axes[i].set_title(f"Seasonal period={comp['period']}, strength={comp['strength']:.2f}")
            axes[i].grid(True, alpha=0.3)
        plt.tight_layout(); plots['multi_seasonal'] = _fig_to_b64(fig)

    # 6. Residual diagnostics (2×2)
    if residual_diag:
        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        fig.suptitle('Residual Diagnostics', fontsize=13, fontweight='bold')
        r_vals = residual.dropna().values

        # Time plot + ±2σ
        axes[0,0].plot(np.arange(len(r_vals)), r_vals, '#C44E52', linewidth=1, alpha=0.8)
        axes[0,0].axhline(0, color='gray', linestyle='--', linewidth=0.8)
        sig2 = 2 * float(np.std(r_vals))
        axes[0,0].axhline(sig2,  color='orange', linestyle=':', alpha=0.7, label='±2σ')
        axes[0,0].axhline(-sig2, color='orange', linestyle=':', alpha=0.7)
        axes[0,0].set_title('Residuals over Time'); axes[0,0].legend(fontsize=8)
        axes[0,0].grid(True, alpha=0.3)

        # Histogram + normal fit
        axes[0,1].hist(r_vals, bins=min(30, len(r_vals)//5), color='#4C72B0',
                       alpha=0.7, density=True, edgecolor='white')
        mu, sd = float(np.mean(r_vals)), float(np.std(r_vals))
        xl = np.linspace(mu - 4*sd, mu + 4*sd, 200)
        axes[0,1].plot(xl, sp_norm.pdf(xl, mu, sd), 'r-', linewidth=2, label='Normal fit')
        sw_p = residual_diag.get('shapiro_wilk', {}).get('p_value')
        if sw_p is not None:
            axes[0,1].text(0.97, 0.95, f'S-W p={sw_p:.4f}',
                           transform=axes[0,1].transAxes, ha='right', va='top',
                           fontsize=9, color='green' if sw_p > 0.05 else 'red')
        axes[0,1].set_title('Residual Distribution'); axes[0,1].legend(fontsize=8)
        axes[0,1].grid(True, alpha=0.3)

        # Q-Q plot
        probplot(r_vals, dist='norm', plot=axes[1,0])
        axes[1,0].set_title('Q-Q Plot'); axes[1,0].grid(True, alpha=0.3)

        # Residual ACF
        acf_d = residual_diag.get('acf', {})
        acf_v = acf_d.get('values', [])
        if acf_v:
            ci_r = acf_d.get('confidence_interval', 0.15)
            lags_r = np.arange(len(acf_v))
            axes[1,1].bar(lags_r, acf_v, color='#4C72B0', alpha=0.7)
            axes[1,1].axhline( ci_r, color='red', linestyle='--', linewidth=1, alpha=0.7)
            axes[1,1].axhline(-ci_r, color='red', linestyle='--', linewidth=1, alpha=0.7)
            sig_l = acf_d.get('significant_lags', [])
            if sig_l:
                axes[1,1].text(0.97, 0.95, f'Sig lags: {sig_l[:5]}',
                               transform=axes[1,1].transAxes, ha='right', va='top',
                               fontsize=8, color='red')
            axes[1,1].set_title('Residual ACF'); axes[1,1].set_xlabel('Lag')
            axes[1,1].grid(True, alpha=0.3)

        plt.tight_layout(); plots['residual_diagnostics'] = _fig_to_b64(fig)

    # 7. Strength comparison
    if period_results and len(period_results) >= 2:
        fig, ax = plt.subplots(figsize=(12, 5))
        pl = [r['period'] for r in period_results]
        ss = [r['seasonal_strength'] for r in period_results]
        ts = [r.get('trend_strength') for r in period_results]
        xp = np.arange(len(pl)); w = 0.35
        ax.bar(xp - w/2, ss, w, label='Seasonal Strength', color='#4C72B0', alpha=0.7)
        if all(v is not None for v in ts):
            ax.bar(xp + w/2, ts, w, label='Trend Strength', color='#55A868', alpha=0.7)
        ax.set_xticks(xp); ax.set_xticklabels(pl)
        ax.set_xlabel('Period'); ax.set_ylabel('Strength Index')
        ax.set_title('Seasonal vs Trend Strength by Period', fontsize=13, fontweight='bold')
        ax.set_ylim(0, 1); ax.legend(); ax.grid(True, alpha=0.3, axis='y')
        best = int(np.argmax(ss))
        ax.annotate('Best', xy=(xp[best], ss[best]),
                    xytext=(xp[best], min(ss[best]+0.12, 0.98)), ha='center',
                    fontsize=10, color='red', arrowprops=dict(arrowstyle='->', color='red'))
        plt.tight_layout(); plots['strength_comparison'] = _fig_to_b64(fig)

    return plots


# ══════════════════════════════════════════════════════════════════
# Endpoint
# ══════════════════════════════════════════════════════════════════

@router.post("/seasonal-analysis")
def seasonal_analysis(req: SeasonalAnalysisRequest):
    try:
        df       = pd.DataFrame(req.data)
        variable = req.variable
        period   = req.period or 12
        test_periods = req.test_periods or []
        auto_detect  = req.auto_detect if req.auto_detect is not None else True
        decomp_req   = (req.decomposition_type or 'auto').lower()

        if isinstance(variable, list): variable = variable[0]
        if variable not in df.columns:
            raise ValueError(f"Variable '{variable}' not found")

        series = pd.to_numeric(df[variable], errors='coerce').dropna().reset_index(drop=True)
        n = len(series)
        if n < 20:
            raise ValueError(f"Need at least 20 observations, got {n}.")

        # ② Ensemble auto-detect
        period_detect = None
        if auto_detect:
            period_detect = detect_periods_ensemble(series)
            if period_detect['dominant'] and period == 12:
                period = period_detect['dominant']
        period = min(max(period, 2), n // 3)

        # ⑤ Decomposition type
        if decomp_req == 'auto':
            decomp_type = choose_decomposition_type(series, period)
        else:
            decomp_type = decomp_req

        # ① True STL
        trend, seasonal, residual = decompose_series(series, period, decomp_type)
        ssi = seasonal_strength(trend, seasonal, residual)
        tsi = trend_strength(trend, seasonal, residual)
        seasonal_indices = calculate_seasonal_indices(series, period, decomp_type)

        # ③ Multi-seasonality
        if not test_periods:
            test_periods = [4, 6, 7, 12, 24, 52] if n > 100 else [4, 6, 7, 12]
        test_periods = [p for p in test_periods if 2 <= p < n // 3]

        multi_periods = ([c['period'] for c in period_detect.get('candidates', [])
                          if c['votes'] >= 2 and 2 <= c['period'] < n // 3]
                         if period_detect else [period])
        if not multi_periods: multi_periods = [period]
        multi_components, _, _ = multi_seasonal_decompose(series, multi_periods)

        period_results     = analyze_multiple_periods(series, test_periods, decomp_type)
        fourier_components = calculate_fourier_components(series)

        # ④ Residual diagnostics
        residual_diag = residual_diagnostics(residual, period)

        # Insights + plots
        insights, recommendations = generate_insights(
            ssi, tsi, period_detect, seasonal_indices,
            period_results, decomp_type, residual_diag, multi_components)

        plots = create_plots(
            series, trend, seasonal, residual, period, variable,
            ssi, tsi, seasonal_indices, period_detect, period_results,
            decomp_type, residual_diag, multi_components)

        seasonality_level = 'Strong' if ssi > 0.7 else 'Moderate' if ssi > 0.4 else 'Weak'
        trend_level       = 'Strong' if tsi > 0.7 else 'Moderate' if tsi > 0.4 else 'Weak'

        parts = [
            '**Overall Analysis**',
            f'→ STL ({decomp_type}) on **{variable}**, n={n}.',
            f'→ Primary period: **{period}**, decomposition: {decomp_type}.',
        ]
        if period_detect and period_detect.get('dominant'):
            cand = next((c for c in period_detect['candidates']
                         if c['period'] == period_detect['dominant']), None)
            m_str = (', '.join(cand['methods']) if cand else '')
            parts.append(f"→ Auto-detected: **{period_detect['dominant']}** via {m_str}.")
        if multi_components and len(multi_components) > 1:
            parts.append('→ Multiple seasons: ' + ', '.join(str(c['period']) for c in multi_components))
        parts += ['', '**Key Metrics**',
                  f'→ SSI: {ssi:.3f} ({seasonality_level})',
                  f'→ TSI: {tsi:.3f} ({trend_level})',
                  f'→ Residuals: {residual_diag.get("verdict","N/A")}',
                  '', '**Recommendations**'] + [f'→ {r}' for r in recommendations[:5]]

        response = {
            'variable': variable, 'n_observations': n, 'period': period,
            'decomposition_type': decomp_type,
            'seasonal_strength_index': safe_float(ssi),
            'trend_strength_index':    safe_float(tsi),
            'period_detection': {k: v for k, v in (period_detect or {}).items()
                                 if k not in ('freqs', 'power')},
            'multi_seasonal_periods': [{'period': c['period'], 'strength': c['strength']}
                                       for c in multi_components],
            'seasonal_indices':    seasonal_indices,
            'period_comparison':   period_results,
            'fourier_components':  fourier_components[:5],
            'residual_diagnostics': residual_diag,
            'interpretation_summary': {'seasonality': seasonality_level,
                                       'trend': trend_level,
                                       'decomp_type': decomp_type},
            'insights':         insights,
            'recommendations':  recommendations,
            'interpretation':   '\n'.join(parts),
            'plots':            plots,
        }
        return _to_native(response)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))