from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

router = APIRouter()

sns.set_theme(style="whitegrid")
sns.set_context("notebook", font_scale=1.1)


class RollingStatisticsRequest(BaseModel):
    data: List[Dict[str, Any]]
    variable: str
    timeCol: Optional[str] = None
    window: Optional[int] = None        # None → auto-selected
    ewm_span: int = 10
    anomaly_threshold: float = 2.0
    compare_windows: Optional[List[int]] = None


def _to_native(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
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


def fig_to_base64(fig):
    """Convert figure to base64 WITHOUT prefix"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ══════════════════════════════════════════════════════════════════
# ① Window auto-recommendation
# ══════════════════════════════════════════════════════════════════

def _recommend_window(series: pd.Series,
                      candidate_windows: list = None) -> dict:
    """
    Recommend an optimal rolling window via three independent heuristics,
    then rank all candidates by smoothing quality.

    Heuristics
    ----------
    sqrt_n        : √n rounded — classic bias-variance rule of thumb
                    (Tukey 1977)
    acf_decay     : first ACF lag where |ρ| drops below 1/e ≈ 0.368,
                    captures the series memory length (Box & Jenkins 1970)
    variance_elbow: scan windows 2..min(n/2,40), pick elbow where
                    d(rolling-mean variance)/d(window) flattens to
                    < 10 % of its maximum drop

    Consensus recommendation = median of the three heuristic values
    (robust to any single outlier heuristic).

    Per-candidate metrics
    ---------------------
    rolling_mean_variance : var of the smoothed series  (lower = smoother)
    rolling_std_mean      : mean local volatility captured
    noise_reduction_pct   : % variance reduction vs raw series
    """
    n = len(series)
    if candidate_windows is None:
        candidate_windows = [5, 10, 20]
    candidate_windows = sorted(set(candidate_windows))
    candidate_windows = [w for w in candidate_windows if 2 <= w <= n // 2]
    if not candidate_windows:
        candidate_windows = [max(2, n // 10)]

    # Heuristic 1: sqrt(n)
    sqrt_n = max(2, int(round(n ** 0.5)))

    # Heuristic 2: ACF decay lag
    acf_win = sqrt_n
    try:
        from statsmodels.tsa.stattools import acf as sm_acf
        acf_vals = sm_acf(series.values, nlags=min(n // 2, 60), fft=True)
        for lag in range(1, len(acf_vals)):
            if abs(acf_vals[lag]) < 1 / np.e:
                acf_win = max(2, lag)
                break
    except Exception:
        pass

    # Heuristic 3: variance elbow
    elbow_win = sqrt_n
    scan = list(range(2, min(n // 2, 41)))
    if len(scan) >= 3:
        variances = [float(series.rolling(w).mean().var()) for w in scan]
        diffs = np.diff(variances)
        max_drop = abs(min(diffs)) if len(diffs) else 1e-9
        for i, d in enumerate(diffs):
            if max_drop > 0 and abs(d) < 0.1 * max_drop:
                elbow_win = scan[i]
                break

    # Consensus = median of three heuristics
    consensus = int(np.median([sqrt_n, acf_win, elbow_win]))
    consensus = max(2, min(consensus, n // 2))

    # Add consensus to evaluation pool
    all_candidates = sorted(set(candidate_windows + [consensus]))

    # Per-candidate metrics
    raw_var = float(series.var()) if float(series.var()) > 0 else 1.0
    candidate_stats = []
    for w in all_candidates:
        rm = series.rolling(w).mean().dropna()
        rs = series.rolling(w).std().dropna()
        rv = float(rm.var()) if len(rm) > 1 else raw_var
        noise_red = max(0.0, (1 - rv / raw_var) * 100)
        candidate_stats.append({
            'window':                w,
            'rolling_mean_variance': round(rv, 6),
            'rolling_std_mean':      round(float(rs.mean()) if len(rs) > 0 else 0, 4),
            'noise_reduction_pct':   round(noise_red, 1),
            'is_recommended':        w == consensus,
        })

    return {
        'recommended':       consensus,
        'heuristics': {
            'sqrt_n':          sqrt_n,
            'acf_decay':       acf_win,
            'variance_elbow':  elbow_win,
        },
        'consensus_method':  'median(sqrt_n, acf_decay, variance_elbow)',
        'all_candidates':    candidate_stats,
    }


# ══════════════════════════════════════════════════════════════════
# ② Rolling IQR anomaly detection
# ══════════════════════════════════════════════════════════════════

def _rolling_iqr_anomaly(series: pd.Series,
                          window: int,
                          multiplier: float = 1.5) -> dict:
    """
    Tukey fence anomaly detection on a rolling window.

    For each t:
      Q1(t), Q3(t) = rolling 25th/75th percentile over [t-w+1 .. t]
      IQR(t)       = Q3(t) - Q1(t)
      lower(t)     = Q1(t) - multiplier * IQR(t)
      upper(t)     = Q3(t) + multiplier * IQR(t)
      anomaly      if y(t) < lower(t) or y(t) > upper(t)

    Makes no Gaussian assumption — robust to heavy tails and skewed
    distributions.  Ref: Tukey (1977) Exploratory Data Analysis.
    """
    q1 = series.rolling(window, min_periods=window).quantile(0.25)
    q3 = series.rolling(window, min_periods=window).quantile(0.75)
    iqr = q3 - q1
    lo  = q1 - multiplier * iqr
    hi  = q3 + multiplier * iqr

    fence_ok    = hi.notna()
    is_high     = (series > hi)  & fence_ok
    is_low      = (series < lo)  & fence_ok
    is_anomaly  = is_high | is_low

    details = []
    for idx in np.where(is_anomaly)[0]:
        details.append({
            'index':       int(idx),
            'value':       float(series.iloc[idx]),
            'q1':          float(q1.iloc[idx])  if pd.notna(q1.iloc[idx])  else None,
            'q3':          float(q3.iloc[idx])  if pd.notna(q3.iloc[idx])  else None,
            'iqr':         float(iqr.iloc[idx]) if pd.notna(iqr.iloc[idx]) else None,
            'lower_fence': float(lo.iloc[idx])  if pd.notna(lo.iloc[idx])  else None,
            'upper_fence': float(hi.iloc[idx])  if pd.notna(hi.iloc[idx])  else None,
            'type':        'high' if is_high.iloc[idx] else 'low',
        })

    n = len(series)
    return {
        'summary': {
            'count':      int(is_anomaly.sum()),
            'rate':       round(float(is_anomaly.sum() / n * 100), 2),
            'high_count': int(is_high.sum()),
            'low_count':  int(is_low.sum()),
            'multiplier': multiplier,
        },
        'details':      details,
        'lower_fence':  lo.tolist(),
        'upper_fence':  hi.tolist(),
        'q1':           q1.tolist(),
        'q3':           q3.tolist(),
    }


@router.post("/rolling-statistics")
async def rolling_statistics_analysis(request: RollingStatisticsRequest):
    try:
        df = pd.DataFrame(request.data)
        variable        = request.variable
        time_col        = request.timeCol
        ewm_span        = request.ewm_span
        anomaly_threshold = request.anomaly_threshold
        compare_windows = request.compare_windows or [5, 10, 20]

        if variable not in df.columns:
            raise HTTPException(status_code=400, detail=f"Variable '{variable}' not found")

        # Prepare data (keep date alignment)
        value_raw  = pd.to_numeric(df[variable], errors='coerce')
        valid_mask = value_raw.notna()
        series     = value_raw[valid_mask].reset_index(drop=True)
        n          = len(series)

        dates = None
        if time_col and time_col in df.columns:
            dates = pd.to_datetime(df[time_col], errors='coerce')[valid_mask].reset_index(drop=True)

        # ① Window recommendation
        window_rec         = _recommend_window(series, candidate_windows=compare_windows)
        recommended_window = window_rec['recommended']
        window = request.window if request.window is not None else recommended_window
        window = max(2, min(int(window), n // 2))

        if n < window:
            raise HTTPException(status_code=400,
                detail=f"Not enough data. Need at least {window} observations, got {n}")

        # Rolling statistics
        rolling_mean = series.rolling(window=window).mean()
        rolling_std  = series.rolling(window=window).std()
        rolling_min  = series.rolling(window=window).min()
        rolling_max  = series.rolling(window=window).max()

        # EWM
        ewm_mean = series.ewm(span=ewm_span).mean()

        # Z-score anomaly
        zscore = (series - rolling_mean) / (rolling_std + 1e-10)

        # ② IQR anomaly
        iqr_result = _rolling_iqr_anomaly(series, window=window)
        # Sanitize nan/inf in fence lists so JSON serialisation never fails
        def _clean_list(lst):
            return [None if (v is None or (isinstance(v, float) and not np.isfinite(v))) else v
                    for v in lst]
        iqr_result['lower_fence'] = _clean_list(iqr_result['lower_fence'])
        iqr_result['upper_fence'] = _clean_list(iqr_result['upper_fence'])
        iqr_result['q1']          = _clean_list(iqr_result['q1'])
        iqr_result['q3']          = _clean_list(iqr_result['q3'])

        # Overall statistics
        overall = {
            'mean': float(series.mean()),
            'std': float(series.std()),
            'min': float(series.min()),
            'max': float(series.max()),
            'median': float(series.median()),
            'skew': float(series.skew()),
            'kurt': float(series.kurtosis())
        }

        # Rolling latest (last valid values)
        rolling_latest = {
            'mean': float(rolling_mean.iloc[-1]) if pd.notna(rolling_mean.iloc[-1]) else None,
            'std': float(rolling_std.iloc[-1]) if pd.notna(rolling_std.iloc[-1]) else None,
            'min': float(rolling_min.iloc[-1]) if pd.notna(rolling_min.iloc[-1]) else None,
            'max': float(rolling_max.iloc[-1]) if pd.notna(rolling_max.iloc[-1]) else None,
            'zscore': float(zscore.iloc[-1]) if pd.notna(zscore.iloc[-1]) else None
        }

        # Detect anomalies
        anomaly_mask = np.abs(zscore) > anomaly_threshold
        anomaly_indices = np.where(anomaly_mask)[0]
        
        high_anomalies = zscore > anomaly_threshold
        low_anomalies = zscore < -anomaly_threshold
        
        anomalies_summary = {
            'count': int(anomaly_mask.sum()),
            'rate': float(anomaly_mask.sum() / n * 100),
            'high_count': int(high_anomalies.sum()),
            'low_count': int(low_anomalies.sum())
        }

        # Build IQR anomaly index set for cross-reference
        iqr_set = {d['index'] for d in iqr_result['details']}

        # Z-score anomaly details (enriched with date + IQR agreement)
        anomaly_details = []
        for idx in anomaly_indices:
            entry = {
                'index':            int(idx),
                'value':            float(series.iloc[idx]),
                'zscore':           float(zscore.iloc[idx]) if pd.notna(zscore.iloc[idx]) else 0,
                'rolling_mean':     float(rolling_mean.iloc[idx]) if pd.notna(rolling_mean.iloc[idx]) else None,
                'rolling_std':      float(rolling_std.iloc[idx]) if pd.notna(rolling_std.iloc[idx]) else None,
                'type':             'high' if zscore.iloc[idx] > 0 else 'low',
                'confirmed_by_iqr': int(idx) in iqr_set,
            }
            if dates is not None and idx < len(dates):
                entry['date'] = str(dates.iloc[idx])[:10]
            anomaly_details.append(entry)

        # Enrich IQR details with dates
        for d in iqr_result['details']:
            if dates is not None and d['index'] < len(dates):
                d['date'] = str(dates.iloc[d['index']])[:10]

        # Combined: flagged by BOTH methods
        zscore_set = {int(i) for i in anomaly_indices}
        both_set   = zscore_set & iqr_set
        combined_anomaly = {
            'zscore_only':  sorted(zscore_set - iqr_set),
            'iqr_only':     sorted(iqr_set    - zscore_set),
            'both_methods': sorted(both_set),
            'n_confirmed':  len(both_set),
        }

        # Rolling data for export (includes IQR fences + optional date)
        lo_fence = iqr_result['lower_fence']
        hi_fence = iqr_result['upper_fence']

        def _safe_fence(lst, i):
            v = lst[i]
            return None if (v is None or (isinstance(v, float) and np.isnan(v))) else float(v)

        rolling_data = []
        for i in range(n):
            entry = {
                'index':        int(i),
                'value':        float(series.iloc[i]),
                'rolling_mean': float(rolling_mean.iloc[i]) if pd.notna(rolling_mean.iloc[i]) else None,
                'rolling_std':  float(rolling_std.iloc[i])  if pd.notna(rolling_std.iloc[i])  else None,
                'zscore':       float(zscore.iloc[i])        if pd.notna(zscore.iloc[i])        else None,
                'ewm_mean':     float(ewm_mean.iloc[i])      if pd.notna(ewm_mean.iloc[i])      else None,
                'iqr_lower':    _safe_fence(lo_fence, i),
                'iqr_upper':    _safe_fence(hi_fence, i),
            }
            if dates is not None and i < len(dates):
                entry['date'] = str(dates.iloc[i])[:10]
            rolling_data.append(entry)

        # Summary
        n_z     = anomalies_summary['count']
        n_iqr   = iqr_result['summary']['count']
        n_both  = combined_anomaly['n_confirmed']

        summary = {
            'n_observations':        n,
            'window':                window,
            'window_auto_selected':  request.window is None,
            'window_recommendation': window_rec,
            'overall':               overall,
            'rolling_latest':        rolling_latest,
            # Legacy key (backward compat)
            'anomalies':             anomalies_summary,
            # New structured keys
            'anomalies_zscore':      anomalies_summary,
            'anomalies_iqr':         iqr_result['summary'],
            'anomalies_combined':    combined_anomaly,
        }

        # Generate insights
        insights = []
        recommendations = []

        # Window recommendation insight
        h = window_rec['heuristics']
        auto_tag = ' (auto-selected)' if request.window is None else ''
        insights.append({
            'type': 'info',
            'title': f'Window: {window}{auto_tag}  |  Recommended: {recommended_window}',
            'description': (
                f'Three heuristics — √n={h["sqrt_n"]}, ACF decay={h["acf_decay"]}, '
                f'variance elbow={h["variance_elbow"]}. '
                f'Consensus (median): {recommended_window}. '
                + ('Override with the `window` parameter.' if request.window is None else '')
            )
        })

        # Dual anomaly insight
        if n_z == 0 and n_iqr == 0:
            insights.append({
                'type': 'info',
                'title': 'No Anomalies Detected ✓',
                'description': (
                    f'Both Z-score (±{anomaly_threshold}σ) and IQR (×1.5) methods agree: '
                    f'no anomalies in {n} observations.'
                )
            })
        else:
            insights.append({
                'type': 'warning',
                'title': f'Anomalies — Z-score: {n_z}  |  IQR: {n_iqr}  |  Both: {n_both}',
                'description': (
                    f'Z-score (Gaussian): {n_z} ({anomalies_summary["high_count"]} high, '
                    f'{anomalies_summary["low_count"]} low). '
                    f'IQR (distribution-free): {n_iqr} '
                    f'({iqr_result["summary"]["high_count"]} high, '
                    f'{iqr_result["summary"]["low_count"]} low). '
                    f'{n_both} confirmed by both methods (highest confidence).'
                )
            })
            if n_both > 0:
                insights.append({
                    'type': 'warning',
                    'title': f'{n_both} High-Confidence Anomaly Points',
                    'description': (
                        f'Indices {combined_anomaly["both_methods"][:10]} flagged by '
                        f'both Z-score and IQR — strongly suspect regardless of distribution shape.'
                    )
                })

        # Volatility insight
        cv = overall['std'] / abs(overall['mean']) * 100 if overall['mean'] != 0 else 0
        if cv > 50:
            insights.append({
                'type': 'warning',
                'title': 'High Volatility',
                'description': f'Coefficient of variation is {cv:.1f}%, indicating highly variable data.'
            })
        elif cv < 10:
            insights.append({
                'type': 'info',
                'title': 'Low Volatility ✓',
                'description': f'Coefficient of variation is {cv:.1f}%, indicating stable data.'
            })

        # Recommendations
        if request.window is None:
            recommendations.append(
                f'Window auto-selected as {window} (consensus of √n={h["sqrt_n"]}, '
                f'ACF={h["acf_decay"]}, elbow={h["variance_elbow"]}). '
                f'Pass `window` to override.'
            )
        else:
            recommendations.append(
                f'Window {window} used (recommended: {recommended_window}). '
                f'See `summary.window_recommendation` for full candidate comparison.'
            )
        if n_both > 0:
            recommendations.append(
                f'Prioritise {n_both} anomaly point(s) confirmed by both methods — '
                f'robust to distribution assumptions.'
            )
        elif n_z > 0 or n_iqr > 0:
            recommendations.append(
                'Z-score and IQR disagree — series may be heavy-tailed or skewed. '
                'Prefer IQR results in that case.'
            )
        if cv > 30:
            recommendations.append('Consider EWM for recent-trend emphasis due to high volatility.')
        recommendations.append('Compare candidate windows in `summary.window_recommendation.all_candidates`.')

        # ============ PLOTS ============
        
        # Plot 1: Rolling Mean with Std Bands
        fig1, ax1 = plt.subplots(figsize=(12, 5))
        ax1.plot(series.index, series.values, 'b-', alpha=0.5, linewidth=0.8, label='Actual')
        ax1.plot(rolling_mean.index, rolling_mean.values, 'r-', linewidth=2, label=f'Rolling Mean (w={window})')
        ax1.fill_between(rolling_mean.index, 
                        rolling_mean - 2*rolling_std, 
                        rolling_mean + 2*rolling_std, 
                        alpha=0.2, color='red', label='±2σ Band')
        ax1.set_title(f'Rolling Mean & Volatility Bands: {variable}', fontweight='bold')
        ax1.set_xlabel('Index')
        ax1.set_ylabel(variable)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        plot_rolling_mean_std = fig_to_base64(fig1)

        # Plot 2: Statistics Panel (4 subplots)
        fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Rolling Mean
        axes[0,0].plot(rolling_mean, 'b-', linewidth=1.5)
        axes[0,0].set_title('Rolling Mean', fontweight='bold')
        axes[0,0].grid(True, alpha=0.3)
        
        # Rolling Std
        axes[0,1].plot(rolling_std, 'purple', linewidth=1.5)
        axes[0,1].set_title('Rolling Std (Volatility)', fontweight='bold')
        axes[0,1].grid(True, alpha=0.3)
        
        # Rolling Min/Max
        axes[1,0].fill_between(series.index, rolling_min, rolling_max, alpha=0.3, color='green')
        axes[1,0].plot(series, 'k-', alpha=0.5, linewidth=0.5)
        axes[1,0].set_title('Rolling Range (Min-Max)', fontweight='bold')
        axes[1,0].grid(True, alpha=0.3)
        
        # Z-Score
        axes[1,1].plot(zscore, 'orange', linewidth=1)
        axes[1,1].axhline(y=anomaly_threshold, color='r', linestyle='--', alpha=0.7)
        axes[1,1].axhline(y=-anomaly_threshold, color='r', linestyle='--', alpha=0.7)
        axes[1,1].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        axes[1,1].set_title('Rolling Z-Score', fontweight='bold')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_statistics_panel = fig_to_base64(fig2)

        # Plot 3: Dual anomaly — 2 panels (Z-score top, IQR bottom)
        fig3, (ax3a, ax3b) = plt.subplots(2, 1, figsize=(13, 10), sharex=True)

        # ── Panel A: Z-score anomaly ─────────────────────────────────────────
        ax3a.plot(series.index, series.values, 'b-', alpha=0.6, linewidth=1, label='Values')
        ax3a.plot(rolling_mean.index, rolling_mean.values, 'gray', linewidth=1.5, alpha=0.7, label='Rolling Mean')
        zscore_lo = rolling_mean - anomaly_threshold * rolling_std
        zscore_hi = rolling_mean + anomaly_threshold * rolling_std
        ax3a.fill_between(series.index, zscore_lo, zscore_hi,
                          alpha=0.12, color='steelblue', label=f'±{anomaly_threshold}σ band')

        high_idx = np.where(high_anomalies)[0]
        low_idx  = np.where(low_anomalies)[0]
        # Highlight confirmed-by-both in different colour
        both_idx = np.array(combined_anomaly['both_methods'])

        if len(high_idx) > 0:
            ax3a.scatter(high_idx, series.iloc[high_idx],
                         c='red', s=55, zorder=5, marker='^',
                         label=f'Z-score high ({len(high_idx)})')
        if len(low_idx) > 0:
            ax3a.scatter(low_idx, series.iloc[low_idx],
                         c='orange', s=55, zorder=5, marker='v',
                         label=f'Z-score low ({len(low_idx)})')
        if len(both_idx) > 0:
            ax3a.scatter(both_idx, series.iloc[both_idx],
                         edgecolors='black', facecolors='none',
                         s=110, zorder=6, linewidths=1.8,
                         label=f'Confirmed by both ({len(both_idx)})')

        ax3a.set_title(f'Z-score Anomaly (±{anomaly_threshold}σ) — {len(high_idx)+len(low_idx)} flags',
                       fontweight='bold')
        ax3a.set_ylabel(variable)
        ax3a.legend(loc='upper right', fontsize=8)
        ax3a.grid(True, alpha=0.3)

        # ── Panel B: IQR anomaly ─────────────────────────────────────────────
        lo_series = pd.Series(iqr_result['lower_fence'])
        hi_series = pd.Series(iqr_result['upper_fence'])

        ax3b.plot(series.index, series.values, 'b-', alpha=0.6, linewidth=1, label='Values')
        ax3b.fill_between(series.index, lo_series, hi_series,
                          alpha=0.15, color='green', label='IQR fence (×1.5)')
        ax3b.plot(lo_series.index, lo_series.values, 'green', linewidth=1, alpha=0.6, linestyle='--')
        ax3b.plot(hi_series.index, hi_series.values, 'green', linewidth=1, alpha=0.6, linestyle='--')

        iqr_high = [d for d in iqr_result['details'] if d['type'] == 'high']
        iqr_low  = [d for d in iqr_result['details'] if d['type'] == 'low']
        if iqr_high:
            idx_h = [d['index'] for d in iqr_high]
            ax3b.scatter(idx_h, series.iloc[idx_h], c='red', s=55, zorder=5, marker='^',
                         label=f'IQR high ({len(iqr_high)})')
        if iqr_low:
            idx_l = [d['index'] for d in iqr_low]
            ax3b.scatter(idx_l, series.iloc[idx_l], c='orange', s=55, zorder=5, marker='v',
                         label=f'IQR low ({len(iqr_low)})')
        if len(both_idx) > 0:
            ax3b.scatter(both_idx, series.iloc[both_idx],
                         edgecolors='black', facecolors='none',
                         s=110, zorder=6, linewidths=1.8,
                         label=f'Confirmed by both ({len(both_idx)})')

        n_iqr_total = iqr_result['summary']['count']
        ax3b.set_title(f'IQR Anomaly (Q1/Q3 ± 1.5×IQR, rolling w={window}) — {n_iqr_total} flags',
                       fontweight='bold')
        ax3b.set_xlabel('Index')
        ax3b.set_ylabel(variable)
        ax3b.legend(loc='upper right', fontsize=8)
        ax3b.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_anomalies = fig_to_base64(fig3)

        # Plot 4: SMA vs EWM Comparison
        fig4, ax4 = plt.subplots(figsize=(12, 5))
        ax4.plot(series.index, series.values, 'lightgray', linewidth=0.8, label='Actual')
        ax4.plot(rolling_mean.index, rolling_mean.values, 'b-', linewidth=2, label=f'SMA (w={window})')
        ax4.plot(ewm_mean.index, ewm_mean.values, 'r-', linewidth=2, label=f'EWM (span={ewm_span})')
        ax4.set_title('Simple Moving Average vs Exponentially Weighted Mean', fontweight='bold')
        ax4.set_xlabel('Index')
        ax4.set_ylabel(variable)
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
        plot_ewm_comparison = fig_to_base64(fig4)

        # Plot 5: Trend Analysis
        fig5, axes5 = plt.subplots(3, 1, figsize=(12, 10))
        
        # MA Crossover
        short_ma = series.rolling(window=min(window//2, 5)).mean()
        long_ma = rolling_mean
        axes5[0].plot(series.index, series.values, 'lightgray', linewidth=0.5)
        axes5[0].plot(short_ma, 'blue', linewidth=1.5, label=f'Short MA ({min(window//2, 5)})')
        axes5[0].plot(long_ma, 'red', linewidth=1.5, label=f'Long MA ({window})')
        axes5[0].set_title('Moving Average Crossover', fontweight='bold')
        axes5[0].legend()
        axes5[0].grid(True, alpha=0.3)
        
        # Momentum
        momentum = series.diff(window)
        axes5[1].plot(momentum, 'green', linewidth=1)
        axes5[1].axhline(y=0, color='gray', linestyle='--')
        axes5[1].fill_between(momentum.index, 0, momentum, where=momentum > 0, alpha=0.3, color='green')
        axes5[1].fill_between(momentum.index, 0, momentum, where=momentum < 0, alpha=0.3, color='red')
        axes5[1].set_title(f'Momentum ({window}-period change)', fontweight='bold')
        axes5[1].grid(True, alpha=0.3)
        
        # Rate of Change
        roc = series.pct_change(window) * 100
        axes5[2].plot(roc, 'purple', linewidth=1)
        axes5[2].axhline(y=0, color='gray', linestyle='--')
        axes5[2].set_title(f'Rate of Change (%, {window}-period)', fontweight='bold')
        axes5[2].set_ylabel('%')
        axes5[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_trend = fig_to_base64(fig5)

        # Plot 6: Window comparison
        cands = window_rec['all_candidates']
        fig6, axes6 = plt.subplots(2, 1, figsize=(13, 9))

        # Top: overlaid smoothed series for each candidate window
        axes6[0].plot(series.index, series.values, 'lightgray', linewidth=0.8,
                      alpha=0.7, label='Raw')
        cmap6 = plt.cm.tab10(np.linspace(0, 0.9, len(cands)))
        for c, col in zip(cands, cmap6):
            w_c  = c['window']
            rm_c = series.rolling(w_c).mean()
            lw   = 2.5 if c['is_recommended'] else 1.2
            ls   = '-' if c['is_recommended'] else '--'
            lbl  = f'w={w_c} (✓ recommended)' if c['is_recommended'] else f'w={w_c}'
            axes6[0].plot(rm_c.index, rm_c.values, color=col,
                          linewidth=lw, linestyle=ls, label=lbl)
        axes6[0].set_title('Smoothed Series — Window Comparison', fontweight='bold')
        axes6[0].set_ylabel(variable)
        axes6[0].legend(loc='upper right', fontsize=8)
        axes6[0].grid(True, alpha=0.3)

        # Bottom: noise reduction % bar chart
        labels = [f'w={c["window"]}' + (' ✓' if c['is_recommended'] else '')
                  for c in cands]
        values_nr = [c['noise_reduction_pct'] for c in cands]
        bar_colors = ['#2ecc71' if c['is_recommended'] else '#4C72B0' for c in cands]
        bars = axes6[1].bar(labels, values_nr, color=bar_colors, edgecolor='white', alpha=0.85)
        for bar, val in zip(bars, values_nr):
            axes6[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                          f'{val:.1f}%', ha='center', va='bottom', fontsize=9)
        axes6[1].set_title('Noise Reduction % by Window (green = recommended)', fontweight='bold')
        axes6[1].set_ylabel('Noise Reduction (%)')
        axes6[1].set_ylim(0, max(values_nr) * 1.15 + 2 if values_nr else 100)
        axes6[1].grid(True, alpha=0.3, axis='y')

        from matplotlib.patches import Patch
        axes6[1].legend(handles=[
            Patch(color='#2ecc71', label=f'Recommended w={recommended_window}'),
            Patch(color='#4C72B0', label='Other candidates'),
        ], fontsize=8)

        plt.tight_layout()
        plot_window_comparison = fig_to_base64(fig6)

        # Build response
        result = {
            'variable':       variable,
            'summary':        summary,
            'anomaly_details':   anomaly_details,          # Z-score
            'iqr_anomaly':       iqr_result,               # IQR
            'combined_anomaly':  combined_anomaly,         # intersection
            'rolling_data':      rolling_data,
            'insights':          insights,
            'recommendations':   recommendations,
            'plots': {
                'rolling_mean_std':   plot_rolling_mean_std,
                'statistics_panel':   plot_statistics_panel,
                'anomalies':          plot_anomalies,       # dual-panel
                'ewm_comparison':     plot_ewm_comparison,
                'trend':              plot_trend,
                'window_comparison':  plot_window_comparison,
            }
        }

        return _to_native(result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
