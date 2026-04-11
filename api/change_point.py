from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

router = APIRouter()

sns.set_theme(style="darkgrid")
sns.set_context("notebook", font_scale=1.1)


class ChangePointRequest(BaseModel):
    data: List[Dict[str, Any]]
    variable: Any
    timeCol: Optional[str] = None          # optional date column → enables date labels
    max_breaks: int = Field(default=5, ge=1, le=20)
    min_segment_pct: float = Field(default=10, ge=5, le=30)
    cusum_threshold: float = Field(default=1.36, ge=0.5, le=3.0)


def _to_native_type(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [_to_native_type(x) for x in obj.tolist()]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def cusum_test(series, threshold=None):
    """
    CUSUM (Cumulative Sum) test for detecting structural breaks
    """
    n = len(series)
    mean = series.mean()
    std = series.std()
    
    if std == 0:
        return {'statistic': 0, 'breaks': [], 'cusum': [0] * n}
    
    # Calculate CUSUM
    cusum = np.zeros(n)
    cusum[0] = (series.iloc[0] - mean) / std
    for i in range(1, n):
        cusum[i] = cusum[i-1] + (series.iloc[i] - mean) / std
    
    # Normalize CUSUM
    cusum_normalized = cusum / np.sqrt(n)
    
    # Find threshold (using critical value approximation)
    if threshold is None:
        threshold = 1.36  # 5% significance level approximation
    
    # Detect breaks where CUSUM exceeds threshold
    breaks = []
    
    # Find peak deviations
    pos_peaks, _ = find_peaks(cusum_normalized, height=threshold)
    neg_peaks, _ = find_peaks(-cusum_normalized, height=threshold)
    
    all_peaks = sorted(list(pos_peaks) + list(neg_peaks))
    
    for peak in all_peaks:
        breaks.append({
            'index': int(peak),
            'cusum_value': float(cusum_normalized[peak]),
            'direction': 'positive' if cusum_normalized[peak] > 0 else 'negative'
        })
    
    # Calculate test statistic (max absolute CUSUM)
    statistic = np.max(np.abs(cusum_normalized))
    
    return {
        'statistic': float(statistic),
        'threshold': float(threshold),
        'significant': statistic > threshold,
        'breaks': breaks,
        'cusum': cusum_normalized.tolist()
    }


def chow_test(y, x_idx, break_point):
    """
    Chow test for structural break at a specific point
    """
    n = len(y)
    
    if break_point <= 2 or break_point >= n - 2:
        return {'f_statistic': 0, 'p_value': 1.0, 'significant': False}
    
    # Split data
    y1 = y[:break_point]
    y2 = y[break_point:]
    x1 = x_idx[:break_point]
    x2 = x_idx[break_point:]
    
    # Fit pooled regression
    X_pooled = np.column_stack([np.ones(n), x_idx])
    beta_pooled = np.linalg.lstsq(X_pooled, y, rcond=None)[0]
    residuals_pooled = y - X_pooled @ beta_pooled
    rss_pooled = np.sum(residuals_pooled ** 2)
    
    # Fit separate regressions
    X1 = np.column_stack([np.ones(len(x1)), x1])
    X2 = np.column_stack([np.ones(len(x2)), x2])
    
    try:
        beta1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
        beta2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
        
        residuals1 = y1 - X1 @ beta1
        residuals2 = y2 - X2 @ beta2
        
        rss1 = np.sum(residuals1 ** 2)
        rss2 = np.sum(residuals2 ** 2)
        rss_unrestricted = rss1 + rss2
    except:
        return {'f_statistic': 0, 'p_value': 1.0, 'significant': False}
    
    # Calculate F-statistic
    k = 2  # Number of parameters (intercept + slope)
    
    if rss_unrestricted == 0:
        return {'f_statistic': 0, 'p_value': 1.0, 'significant': False}
    
    f_stat = ((rss_pooled - rss_unrestricted) / k) / (rss_unrestricted / (n - 2 * k))
    
    if f_stat < 0:
        f_stat = 0
    
    # Calculate p-value
    df1 = k
    df2 = n - 2 * k
    
    if df2 <= 0:
        return {'f_statistic': float(f_stat), 'p_value': 1.0, 'significant': False}
    
    p_value = 1 - stats.f.cdf(f_stat, df1, df2)
    
    return {
        'f_statistic': float(f_stat),
        'p_value': float(p_value),
        'df1': int(df1),
        'df2': int(df2),
        'significant': p_value < 0.05,
        'rss_pooled': float(rss_pooled),
        'rss_unrestricted': float(rss_unrestricted)
    }


def bai_perron_optimal(series, max_breaks=5, min_segment=None, significance=0.05):
    """
    Global optimal multiple structural break detection via dynamic programming.

    Implements the Bai-Perron (1998, 2003) methodology:
      - Pure structural change model: y_t = μ_j + ε_t for segment j
      - Global minimiser of total within-segment RSS over all m-break partitions
        using Bellman recursion (O(n²) per break, O(m·n²) total)
      - Break significance evaluated with approximate F-test (Chow) at each
        candidate, then BIC used to select optimal m ≤ max_breaks
      - Supconstraint: each segment ≥ min_segment observations (trim condition)

    Refs:
      Bai, J. & Perron, P. (1998). "Estimating and Testing Linear Models
        with Multiple Structural Changes." Econometrica 66(1): 47–78.
      Bai, J. & Perron, P. (2003). "Computation and Analysis of Multiple
        Structural Change Models." J. Applied Econometrics 18(1): 1–22.
    """
    n = len(series)
    if min_segment is None:
        min_segment = max(int(n * 0.1), 10)
    y = series.values.astype(float)

    # ── Pre-compute segment RSS table (O(n²)) ────────────────────────────────
    # rss[i, j] = RSS of fitting a constant mean to y[i:j]
    rss_table = np.full((n, n), np.inf)
    for i in range(n):
        s = 0.0; s2 = 0.0
        for j in range(i, n):
            v = y[j]; s += v; s2 += v * v
            length = j - i + 1
            if length >= min_segment:
                rss_table[i, j] = s2 - s * s / length

    # ── DP: find optimal break positions for each m = 0 .. max_breaks ────────
    # dp[m][t]  = min total RSS using m breaks with last segment ending at t
    # bp[m][t]  = last break point that achieves dp[m][t]
    INF = np.inf
    dp  = {0: {t: rss_table[0, t] for t in range(n)}}
    bp  = {0: {t: 0               for t in range(n)}}

    for m in range(1, max_breaks + 1):
        dp[m] = {}; bp[m] = {}
        for t in range(m * min_segment - 1, n):
            best_rss = INF; best_k = -1
            for k in range((m - 1) * min_segment - 1,
                           t - min_segment + 1):
                if k < 0: continue
                prev = dp[m - 1].get(k, INF)
                seg  = rss_table[k + 1, t] if k + 1 <= t else INF
                if prev + seg < best_rss:
                    best_rss = prev + seg; best_k = k
            dp[m][n - 1] = best_rss  # will be overwritten correctly below
            dp[m][t] = best_rss; bp[m][t] = best_k

    # ── BIC-based model selection ─────────────────────────────────────────────
    # BIC = n·ln(RSS/n) + (m+1)·ln(n)   (m+1 segments, each fit with 1 param)
    def _bic(m):
        total_rss = dp[m].get(n - 1, INF)
        if total_rss <= 0 or total_rss == INF: return INF
        return n * np.log(total_rss / n) + (m + 1) * np.log(n)

    best_m = min(range(0, max_breaks + 1), key=_bic)

    # ── Back-track break positions for best_m ────────────────────────────────
    break_indices = []
    t = n - 1
    for m in range(best_m, 0, -1):
        k = bp[m].get(t, -1)
        if k < 0: break
        break_indices.append(k + 1)   # break is at start of new segment
        t = k
    break_indices = sorted(break_indices)

    # ── Significance filter: Chow test at each candidate break ───────────────
    x_idx   = np.arange(n)
    breaks  = []
    segments_out = [(0, n)]

    for bi in break_indices:
        result = chow_test(y, x_idx, bi)
        if result.get('p_value', 1.0) < significance:
            breaks.append({
                'index':       int(bi),
                'f_statistic': float(result['f_statistic']),
                'p_value':     float(result['p_value']),
                'bic_selected': True,
            })

    # Build segment list
    pts = sorted([b['index'] for b in breaks])
    boundaries = [0] + pts + [n]
    segments_out = [(boundaries[i], boundaries[i + 1])
                    for i in range(len(boundaries) - 1)]

    breaks.sort(key=lambda x: x['index'])
    return breaks, segments_out


def pettitt_test(series):
    """
    Pettitt test for detecting a single change point
    """
    n = len(series)
    y = series.values
    
    # Calculate U statistics
    U = np.zeros(n)
    
    for t in range(n):
        for i in range(t + 1):
            for j in range(t + 1, n):
                U[t] += np.sign(y[j] - y[i])
    
    # Find the point with maximum |U|
    K = np.max(np.abs(U))
    change_point = np.argmax(np.abs(U))
    
    # Approximate p-value
    p_value = 2 * np.exp(-6 * K ** 2 / (n ** 3 + n ** 2))
    p_value = min(1.0, p_value)
    
    return {
        'statistic': float(K),
        'change_point': int(change_point),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'U_values': U.tolist()
    }


def variance_change_test(series, window=None):
    """
    Detect changes in variance using rolling variance ratio
    """
    n = len(series)
    
    if window is None:
        window = max(int(n * 0.1), 10)
    
    # Calculate rolling variance
    rolling_var = series.rolling(window=window, center=True).var()
    
    # Calculate variance ratio
    overall_var = series.var()
    if overall_var == 0:
        return {'breaks': [], 'variance_ratio': [1.0] * n}
    
    variance_ratio = rolling_var / overall_var
    
    # Detect significant changes (using threshold)
    threshold = 2.0  # Variance ratio > 2 or < 0.5 indicates change
    
    breaks = []
    
    # Find peaks in variance ratio
    ratio_values = variance_ratio.dropna().values
    ratio_indices = variance_ratio.dropna().index.tolist()
    
    high_var_peaks, _ = find_peaks(ratio_values, height=threshold)
    low_var_peaks, _ = find_peaks(-ratio_values + 2, height=1.5)  # Detect low variance
    
    for peak in high_var_peaks:
        if peak < len(ratio_indices):
            breaks.append({
                'index': int(ratio_indices[peak]),
                'variance_ratio': float(ratio_values[peak]),
                'type': 'increase'
            })
    
    for peak in low_var_peaks:
        if peak < len(ratio_indices) and ratio_values[peak] < 0.5:
            breaks.append({
                'index': int(ratio_indices[peak]),
                'variance_ratio': float(ratio_values[peak]),
                'type': 'decrease'
            })
    
    breaks.sort(key=lambda x: x['index'])
    
    return {
        'breaks': breaks,
        'variance_ratio': variance_ratio.tolist(),
        'threshold': threshold
    }


def mean_shift_detection(series, window=None):
    """
    Detect mean shifts using moving average comparison
    """
    n = len(series)
    
    if window is None:
        window = max(int(n * 0.1), 10)
    
    # Calculate forward and backward rolling means
    forward_ma = series.rolling(window=window).mean()
    backward_ma = series.iloc[::-1].rolling(window=window).mean().iloc[::-1]
    
    # Calculate mean difference
    mean_diff = forward_ma - backward_ma
    
    # Normalize by rolling std
    rolling_std = series.rolling(window=window).std()
    normalized_diff = mean_diff / (rolling_std + 1e-10)
    
    # Find significant shifts
    threshold = 2.0
    breaks = []
    
    diff_values = normalized_diff.dropna().values
    diff_indices = list(normalized_diff.dropna().index)
    
    # Find peaks
    pos_peaks, _ = find_peaks(diff_values, height=threshold, distance=window)
    neg_peaks, _ = find_peaks(-diff_values, height=threshold, distance=window)
    
    for peak in pos_peaks:
        if peak < len(diff_indices):
            breaks.append({
                'index': int(diff_indices[peak]),
                'normalized_diff': float(diff_values[peak]),
                'direction': 'increase'
            })
    
    for peak in neg_peaks:
        if peak < len(diff_indices):
            breaks.append({
                'index': int(diff_indices[peak]),
                'normalized_diff': float(-diff_values[peak]),
                'direction': 'decrease'
            })
    
    breaks.sort(key=lambda x: x['index'])
    
    return {
        'breaks': breaks,
        'mean_diff': normalized_diff.tolist(),
        'threshold': threshold
    }


def calculate_segment_statistics(series, break_points):
    """
    Calculate statistics for each segment
    """
    n = len(series)
    all_breaks = [0] + sorted(break_points) + [n]
    
    segments = []
    
    for i in range(len(all_breaks) - 1):
        start = all_breaks[i]
        end = all_breaks[i + 1]
        segment_data = series.iloc[start:end]
        
        if len(segment_data) > 0:
            segments.append({
                'segment': i + 1,
                'start': int(start),
                'end': int(end),
                'length': int(end - start),
                'mean': float(segment_data.mean()),
                'std': float(segment_data.std()),
                'min': float(segment_data.min()),
                'max': float(segment_data.max()),
                'median': float(segment_data.median())
            })
    
    return segments


def _apply_date_xticks(ax, dates, n, max_ticks=10):
    """Replace numeric x-ticks with date labels when dates are available."""
    if dates is None:
        return
    step = max(1, n // max_ticks)
    tick_pos = list(range(0, n, step))
    tick_lbl = [str(dates[i])[:10] for i in tick_pos if i < len(dates)]
    ax.set_xticks(tick_pos[:len(tick_lbl)])
    ax.set_xticklabels(tick_lbl, rotation=30, ha='right', fontsize=8)


def _break_annotation(ax, idx, label, dates, y_pos, offset_x=3):
    """Annotate a break line with date label if available, else index."""
    date_str = (str(dates[idx])[:10] if dates is not None and idx < len(dates)
                else f'idx={idx}')
    ax.annotate(f'{label}\n{date_str}',
                xy=(idx, y_pos), xytext=(idx + offset_x, y_pos),
                fontsize=8, color='red',
                arrowprops=dict(arrowstyle='->', color='red', lw=0.8))


def create_cusum_plot(series, cusum_result, variable_name, dates=None):
    """Create CUSUM plot with optional date x-axis."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    n = len(series)
    x = np.arange(n)

    axes[0].plot(x, series.values, 'b-', linewidth=1, alpha=0.7)
    for brk in cusum_result['breaks']:
        axes[0].axvline(x=brk['index'], color='red', linestyle='--', linewidth=2, alpha=0.7)
    axes[0].set_ylabel(variable_name)
    axes[0].set_title(f'Time Series with Detected Breaks: {variable_name}',
                      fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    cusum = cusum_result['cusum']
    threshold = cusum_result['threshold']
    axes[1].plot(x, cusum, 'b-', linewidth=1.5, label='CUSUM')
    axes[1].axhline(y=threshold,  color='red', linestyle='--', linewidth=1.5,
                    label=f'Threshold (±{threshold:.2f})')
    axes[1].axhline(y=-threshold, color='red', linestyle='--', linewidth=1.5)
    axes[1].axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    axes[1].fill_between(x, -threshold, threshold, alpha=0.1, color='green')
    for brk in cusum_result['breaks']:
        axes[1].scatter([brk['index']], [brk['cusum_value']],
                        color='red', s=100, zorder=5, marker='o')

    axes[1].set_xlabel('Date' if dates is not None else 'Index')
    axes[1].set_ylabel('CUSUM')
    axes[1].set_title('CUSUM Test', fontsize=12, fontweight='bold')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    _apply_date_xticks(axes[1], dates, n)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def create_breaks_plot(series, all_breaks, segments, variable_name, dates=None):
    """Structural breaks + segment means, with optional date x-axis."""
    fig, ax = plt.subplots(figsize=(14, 6))
    n = len(series)
    x = np.arange(n)

    ax.plot(x, series.values, 'b-', linewidth=1, alpha=0.5, label='Original')
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(segments), 1)))

    for seg, color in zip(segments, colors):
        seg_x = np.arange(seg['start'], seg['end'])
        seg_y = series.iloc[seg['start']:seg['end']].values
        ax.fill_between(seg_x, seg_y.min(), seg_y.max(), alpha=0.2, color=color)
        ax.hlines(y=seg['mean'], xmin=seg['start'], xmax=seg['end'],
                  color=color, linewidth=3,
                  label=f"Seg {seg['segment']} μ={seg['mean']:.2f}")

    for i, brk in enumerate(sorted(all_breaks)):
        ax.axvline(x=brk, color='red', linestyle='--', linewidth=2, alpha=0.8)
        date_str = (str(dates[brk])[:10] if dates is not None and brk < len(dates)
                    else f'idx={brk}')
        ax.annotate('B'+str(i+1)+'\n'+date_str,
                    xy=(brk, series.max()),
                    xytext=(brk + max(1, n // 80), series.max() * 0.97),
                    fontsize=8, color='red',
                    arrowprops=dict(arrowstyle='->', color='red', lw=0.8))

    ax.set_xlabel('Date' if dates is not None else 'Index', fontsize=12)
    ax.set_ylabel(variable_name, fontsize=12)
    ax.set_title(f'Structural Breaks and Segments: {variable_name}',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    _apply_date_xticks(ax, dates, n)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def create_pettitt_plot(series, pettitt_result, variable_name, dates=None):
    """Create Pettitt test visualization with optional date x-axis."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    n = len(series)
    x = np.arange(n)
    cp = pettitt_result['change_point']
    cp_label = (str(dates[cp])[:10] if dates is not None and cp < len(dates)
                else f'index={cp}')

    axes[0].plot(x, series.values, 'b-', linewidth=1, alpha=0.7)
    axes[0].axvline(x=cp, color='red', linestyle='--', linewidth=2,
                    label=f'Change Point ({cp_label})')
    mean_before = series.iloc[:cp].mean()
    mean_after  = series.iloc[cp:].mean()
    axes[0].hlines(mean_before, 0,  cp,        color='green',  linewidth=2,
                   label=f'Before mean={mean_before:.2f}')
    axes[0].hlines(mean_after,  cp, len(series), color='orange', linewidth=2,
                   label=f'After mean={mean_after:.2f}')
    axes[0].set_ylabel(variable_name)
    axes[0].set_title(f'Pettitt Test: {variable_name}', fontsize=12, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)

    U = pettitt_result['U_values']
    axes[1].plot(x, U, 'purple', linewidth=1.5)
    axes[1].axvline(x=cp, color='red', linestyle='--', linewidth=2)
    axes[1].scatter([cp], [U[cp]], color='red', s=150, zorder=5, marker='*',
                    label=f'Max |U| at {cp_label}')
    axes[1].set_xlabel('Date' if dates is not None else 'Index')
    axes[1].set_ylabel('U Statistic')
    axes[1].set_title(f'Pettitt U Statistic (p={pettitt_result["p_value"]:.4f})',
                      fontsize=12, fontweight='bold')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    _apply_date_xticks(axes[1], dates, n)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def create_variance_plot(series, variance_result, variable_name, dates=None):
    """Variance change detection with optional date x-axis."""
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    n = len(series)
    x = np.arange(n)

    axes[0].plot(x, series.values, 'b-', linewidth=1, alpha=0.7)
    for brk in variance_result['breaks']:
        color = 'red' if brk['type'] == 'increase' else 'green'
        axes[0].axvline(x=brk['index'], color=color, linestyle='--', linewidth=2, alpha=0.7)
    axes[0].set_ylabel(variable_name)
    axes[0].set_title(f'Variance Change Detection: {variable_name}',
                      fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    var_ratio = variance_result['variance_ratio']
    threshold = variance_result['threshold']
    axes[1].plot(x, var_ratio, 'purple', linewidth=1.5)
    axes[1].axhline(y=threshold,   color='red',   linestyle='--',
                    label=f'High ({threshold})')
    axes[1].axhline(y=1/threshold, color='green', linestyle='--',
                    label=f'Low ({1/threshold:.2f})')
    axes[1].axhline(y=1, color='gray', linestyle='-', linewidth=0.5)
    for brk in variance_result['breaks']:
        color = 'red' if brk['type'] == 'increase' else 'green'
        axes[1].scatter([brk['index']], [brk['variance_ratio']],
                        color=color, s=100, zorder=5)
    axes[1].set_xlabel('Date' if dates is not None else 'Index')
    axes[1].set_ylabel('Variance Ratio')
    axes[1].set_title('Rolling Variance Ratio', fontsize=12, fontweight='bold')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    _apply_date_xticks(axes[1], dates, n)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def create_summary_plot(series, all_breaks, variable_name,
                        dates=None, consensus_info=None):
    """
    Comprehensive summary plot showing consensus vs non-consensus breaks.
    Consensus breaks (≥2 algorithms) shown in red solid lines;
    single-algorithm breaks shown in orange dashed.
    """
    fig, ax = plt.subplots(figsize=(14, 6))
    n = len(series)
    x = np.arange(n)

    ax.plot(x, series.values, 'b-', linewidth=1.5, alpha=0.8, label='Time Series')

    window = max(n // 20, 5)
    rolling_mean = series.rolling(window=window).mean()
    ax.plot(x, rolling_mean.values, 'orange', linewidth=2, alpha=0.8,
            label=f'Rolling Mean (w={window})')

    # Build vote lookup for styling
    vote_map = {}
    if consensus_info:
        for b in consensus_info.get('all_breaks_raw', []):
            vote_map[b['index']] = b.get('votes', 1)

    unique_breaks = sorted(set(all_breaks))
    legend_consensus_done = False
    legend_single_done    = False

    for i, brk in enumerate(unique_breaks):
        votes     = vote_map.get(brk, 1)
        is_cons   = votes >= consensus_info.get('min_votes', 2) if consensus_info else True
        color     = 'red'    if is_cons else 'darkorange'
        lstyle    = '-'      if is_cons else '--'
        lw        = 2.5      if is_cons else 1.5
        lbl = None
        if is_cons and not legend_consensus_done:
            lbl = f'Consensus Break (≥{consensus_info.get("min_votes",2)} algos)'
            legend_consensus_done = True
        elif not is_cons and not legend_single_done:
            lbl = 'Single-algo Break'
            legend_single_done = True
        ax.axvline(x=brk, color=color, linestyle=lstyle, linewidth=lw,
                   alpha=0.85, label=lbl)
        date_str = (str(dates[brk])[:10] if dates is not None and brk < len(dates)
                    else f'idx={brk}')
        vote_str = f' ({votes}v)' if consensus_info else ''
        ax.annotate(f'B{i+1}{vote_str}\n{date_str}',
                    xy=(brk, series.max()),
                    xytext=(brk + max(1, n // 80), series.max() * 0.97),
                    fontsize=8, color=color,
                    arrowprops=dict(arrowstyle='->', color=color, lw=0.8))

    ax.set_xlabel('Date' if dates is not None else 'Index', fontsize=12)
    ax.set_ylabel(variable_name, fontsize=12)
    ax.set_title(
        f'Structural Break Summary: {variable_name}  '
        f'({len(unique_breaks)} break{"s" if len(unique_breaks) != 1 else ""} detected)',
        fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    _apply_date_xticks(ax, dates, n)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ══════════════════════════════════════════════════════════════════
# ② Break merging + consensus voting
# ══════════════════════════════════════════════════════════════════

def _merge_nearby(indices, tolerance):
    """
    Merge break indices that are within `tolerance` of each other.
    Uses single-linkage clustering: greedily absorb neighbours.
    Returns list of (representative_index, cluster_members).
    """
    if not indices:
        return []
    sorted_idx = sorted(indices)
    clusters = [[sorted_idx[0]]]
    for idx in sorted_idx[1:]:
        if idx - clusters[-1][-1] <= tolerance:
            clusters[-1].append(idx)
        else:
            clusters.append([idx])
    # Representative = median of cluster
    return [(int(np.median(c)), c) for c in clusters]


def merge_breaks_consensus(cusum_breaks, pettitt_break, bp_breaks,
                            variance_breaks, mean_shift_breaks,
                            n, tolerance=None, min_votes=2):
    """
    Merge all algorithm break nominations with tolerance-based clustering,
    then keep only breaks endorsed by >= min_votes distinct algorithms.

    Parameters
    ----------
    tolerance   : int  — indices within this window are considered the same
                  break. Default = max(3, n // 50).
    min_votes   : int  — minimum number of algorithms that must agree
                  (default 2 out of 5).

    Algorithm vote mapping
    ----------------------
    'cusum'      : CUSUM peak indices
    'pettitt'    : single Pettitt change point (if significant)
    'bai_perron' : global DP optimal breaks
    'variance'   : rolling variance break indices
    'mean_shift' : mean-shift detection indices

    Returns
    -------
    dict with:
      consensus_breaks : list of dicts  — only breaks with votes >= min_votes
      all_breaks_raw   : list of dicts  — all breaks before vote filter
      tolerance        : int used
      min_votes        : int used
    """
    if tolerance is None:
        tolerance = max(3, n // 50)

    # Collect nominations per algorithm
    nominations = {
        'cusum':       [b['index'] for b in cusum_breaks],
        'pettitt':     ([pettitt_break] if pettitt_break is not None else []),
        'bai_perron':  [b['index'] for b in bp_breaks],
        'variance':    [b['index'] for b in variance_breaks],
        'mean_shift':  [b['index'] for b in mean_shift_breaks],
    }

    # Pool all indices, track source
    index_to_algos = {}  # index → set of algo names
    for algo, idxs in nominations.items():
        for idx in idxs:
            index_to_algos.setdefault(idx, set()).add(algo)

    if not index_to_algos:
        return {'consensus_breaks': [], 'all_breaks_raw': [],
                'tolerance': tolerance, 'min_votes': min_votes}

    # Merge nearby indices
    all_indices = sorted(index_to_algos.keys())
    clusters = _merge_nearby(all_indices, tolerance)

    all_breaks_raw = []
    consensus_breaks = []

    for rep, members in clusters:
        # Collect all algorithms that nominated any member of this cluster
        algos_voting = set()
        for m in members:
            algos_voting |= index_to_algos.get(m, set())
        votes = len(algos_voting)

        entry = {
            'index':         rep,
            'votes':         votes,
            'algorithms':    sorted(algos_voting),
            'cluster_range': [int(min(members)), int(max(members))],
            'is_consensus':  votes >= min_votes,
        }
        all_breaks_raw.append(entry)
        if votes >= min_votes:
            consensus_breaks.append(entry)

    # Sort by index
    consensus_breaks.sort(key=lambda x: x['index'])
    all_breaks_raw.sort(key=lambda x: x['index'])

    return {
        'consensus_breaks': consensus_breaks,
        'all_breaks_raw':   all_breaks_raw,
        'tolerance':        tolerance,
        'min_votes':        min_votes,
    }


def _enrich_with_dates(breaks_list, dates):
    """
    Add 'date' field to each break dict if a dates array is provided.
    dates : array-like of date labels aligned with series index.
    """
    if dates is None:
        return breaks_list
    for b in breaks_list:
        idx = b.get('index', -1)
        if 0 <= idx < len(dates):
            d = dates[idx]
            b['date'] = d.isoformat() if hasattr(d, 'isoformat') else str(d)
    return breaks_list


def generate_insights(cusum_result, pettitt_result, bai_perron_breaks, variance_result, mean_shift_result, segments, n, consensus_info=None):
    """Generate insights from structural break analysis"""
    insights = []
    recommendations = []
    
    # Use consensus breaks when available, fall back to raw union
    if consensus_info and consensus_info.get('consensus_breaks'):
        consensus_list = consensus_info['consensus_breaks']
        all_break_indices = {b['index'] for b in consensus_list}
    else:
        all_break_indices = set()
        for brk in cusum_result.get('breaks', []):
            all_break_indices.add(brk['index'])
        if pettitt_result.get('significant', False):
            all_break_indices.add(pettitt_result['change_point'])
        for brk in bai_perron_breaks:
            all_break_indices.add(brk['index'])

    n_breaks = len(all_break_indices)
    
    # Overall assessment
    if n_breaks == 0:
        insights.append({
            "type": "info",
            "title": "No Structural Breaks Detected ✓",
            "description": "The series appears stable with no significant structural changes detected by any test."
        })
    elif n_breaks == 1:
        insights.append({
            "type": "warning",
            "title": "Single Break Point Detected",
            "description": f"One structural break detected. This suggests a regime change at some point in the series."
        })
    else:
        insights.append({
            "type": "warning",
            "title": f"Multiple Breaks Detected ({n_breaks})",
            "description": f"Multiple structural breaks suggest the series has undergone several regime changes."
        })
    
    # CUSUM results
    if cusum_result['significant']:
        insights.append({
            "type": "warning",
            "title": "CUSUM Test: Significant",
            "description": f"CUSUM statistic ({cusum_result['statistic']:.4f}) exceeds threshold ({cusum_result['threshold']:.4f}), indicating cumulative deviation from mean."
        })
    else:
        insights.append({
            "type": "info",
            "title": "CUSUM Test: Not Significant ✓",
            "description": f"CUSUM statistic ({cusum_result['statistic']:.4f}) within threshold bounds."
        })
    
    # Pettitt results
    if pettitt_result['significant']:
        insights.append({
            "type": "warning",
            "title": "Pettitt Test: Significant Change Point",
            "description": f"Change point detected at index {pettitt_result['change_point']} (p-value = {pettitt_result['p_value']:.4f})."
        })
    else:
        insights.append({
            "type": "info",
            "title": "Pettitt Test: No Significant Change ✓",
            "description": f"No single dominant change point detected (p-value = {pettitt_result['p_value']:.4f})."
        })
    
    # Consensus summary
    if consensus_info:
        raw_total  = len(consensus_info.get('all_breaks_raw', []))
        consensus_n = len(consensus_info.get('consensus_breaks', []))
        tol = consensus_info.get('tolerance', '?')
        mv  = consensus_info.get('min_votes', 2)
        if raw_total > 0:
            insights.append({
                "type": "info",
                "title": f"Consensus Filter: {consensus_n}/{raw_total} breaks confirmed",
                "description": (
                    f"{raw_total} candidate breaks found across all algorithms. "
                    f"{consensus_n} passed consensus (≥{mv} algorithms agree within ±{tol} obs). "
                    + ("Duplicate/nearby detections merged." if raw_total > consensus_n else "")
                )
            })

    # Variance changes
    var_breaks = variance_result.get('breaks', [])
    if len(var_breaks) > 0:
        increase_count = len([b for b in var_breaks if b['type'] == 'increase'])
        decrease_count = len([b for b in var_breaks if b['type'] == 'decrease'])
        insights.append({
            "type": "warning",
            "title": "Variance Changes Detected",
            "description": f"Found {increase_count} variance increase(s) and {decrease_count} decrease(s), indicating heteroscedasticity."
        })
    
    # Segment analysis
    if len(segments) > 1:
        means = [seg['mean'] for seg in segments]
        mean_range = max(means) - min(means)
        
        insights.append({
            "type": "info",
            "title": f"Segment Analysis ({len(segments)} segments)",
            "description": f"Segment means range from {min(means):.4f} to {max(means):.4f} (range = {mean_range:.4f})."
        })
    
    # Recommendations
    recommendations.extend([
        "Investigate the cause of detected breaks (policy changes, external events, data issues).",
        "Consider modeling each regime separately for better fit.",
        "Use regime-switching models if breaks are recurrent.",
        "Verify breaks aren't due to data collection changes or outliers."
    ])
    
    if n_breaks > 3:
        recommendations.append("Many breaks may indicate high noise - consider smoothing or longer analysis windows.")
    
    recommendations = list(dict.fromkeys(recommendations))
    
    return insights, recommendations, list(all_break_indices)


@router.post("/change-point")
async def analyze_change_point(request: ChangePointRequest):
    try:
        data            = request.data
        variable        = request.variable
        time_col        = request.timeCol
        max_breaks      = request.max_breaks
        min_segment_pct = request.min_segment_pct
        cusum_threshold = request.cusum_threshold

        df = pd.DataFrame(data)
        if isinstance(variable, list):
            variable = variable[0]
        if variable not in df.columns:
            raise HTTPException(status_code=400,
                                detail=f"Variable '{variable}' not found")

        # ③ Aligned date array
        value_raw  = pd.to_numeric(df[variable], errors='coerce')
        valid_mask = value_raw.notna()
        series     = value_raw[valid_mask].reset_index(drop=True)
        n          = len(series)

        dates = None
        if time_col and time_col in df.columns:
            dates = pd.to_datetime(df[time_col], errors='coerce')[valid_mask].reset_index(drop=True)

        if n < 20:
            raise HTTPException(status_code=400,
                                detail=f"Need at least 20 observations, got {n}.")

        min_segment = max(int(n * min_segment_pct / 100), 5)

        # Run all tests
        cusum_result      = cusum_test(series, threshold=cusum_threshold)
        pettitt_result    = pettitt_test(series)
        # ① Global DP Bai-Perron
        bai_perron_breaks, bp_segments = bai_perron_optimal(
            series, max_breaks=max_breaks, min_segment=min_segment)
        variance_result   = variance_change_test(series)
        mean_shift_result = mean_shift_detection(series)

        # ② Consensus merging with tolerance
        pettitt_cp = (pettitt_result['change_point']
                      if pettitt_result.get('significant') else None)
        consensus_info = merge_breaks_consensus(
            cusum_breaks      = cusum_result.get('breaks', []),
            pettitt_break     = pettitt_cp,
            bp_breaks         = bai_perron_breaks,
            variance_breaks   = variance_result.get('breaks', []),
            mean_shift_breaks = mean_shift_result.get('breaks', []),
            n                 = n,
        )

        # ③ Enrich break dicts with date labels
        if dates is not None:
            _enrich_with_dates(cusum_result.get('breaks', []), dates)
            _enrich_with_dates(bai_perron_breaks, dates)
            _enrich_with_dates(variance_result.get('breaks', []), dates)
            _enrich_with_dates(mean_shift_result.get('breaks', []), dates)
            _enrich_with_dates(consensus_info.get('consensus_breaks', []), dates)
            _enrich_with_dates(consensus_info.get('all_breaks_raw', []), dates)
            if pettitt_cp is not None and pettitt_cp < len(dates):
                pettitt_result['date'] = str(dates[pettitt_cp])[:10]

        # Authoritative break list = consensus; fall back to raw union
        consensus_list = consensus_info.get('consensus_breaks', [])
        all_breaks = sorted([b['index'] for b in consensus_list])
        if not all_breaks:
            raw_set = set()
            for b in cusum_result.get('breaks', []): raw_set.add(b['index'])
            if pettitt_cp is not None: raw_set.add(pettitt_cp)
            for b in bai_perron_breaks: raw_set.add(b['index'])
            all_breaks = sorted(raw_set)

        segments = calculate_segment_statistics(series, all_breaks)
        if dates is not None:
            for seg in segments:
                si = seg['start']; ei = min(seg['end'], len(dates)-1)
                seg['start_date'] = str(dates[si])[:10] if si < len(dates) else None
                seg['end_date']   = str(dates[ei])[:10] if ei < len(dates) else None

        insights, recommendations = generate_insights(
            cusum_result, pettitt_result, bai_perron_breaks,
            variance_result, mean_shift_result, segments, n,
            consensus_info=consensus_info
        )[:2]

        d = dates
        plots = {
            'cusum':    create_cusum_plot(series, cusum_result, variable, dates=d),
            'pettitt':  create_pettitt_plot(series, pettitt_result, variable, dates=d),
            'breaks':   create_breaks_plot(series, all_breaks, segments, variable, dates=d),
            'variance': create_variance_plot(series, variance_result, variable, dates=d),
            'summary':  create_summary_plot(series, all_breaks, variable,
                                            dates=d, consensus_info=consensus_info),
        }

        output = {
            'variable':       variable,
            'n_observations': n,
            'has_dates':      dates is not None,
            'tests': {
                'cusum': {
                    'statistic':   _to_native_type(cusum_result['statistic']),
                    'threshold':   _to_native_type(cusum_result['threshold']),
                    'significant': cusum_result['significant'],
                    'n_breaks':    len(cusum_result['breaks']),
                    'breaks':      cusum_result['breaks'],
                },
                'pettitt': {
                    'statistic':    _to_native_type(pettitt_result['statistic']),
                    'change_point': _to_native_type(pettitt_result['change_point']),
                    'p_value':      _to_native_type(pettitt_result['p_value']),
                    'significant':  pettitt_result['significant'],
                    'date':         pettitt_result.get('date'),
                },
                'bai_perron': {
                    'method':   'global_dp_optimal',
                    'n_breaks': len(bai_perron_breaks),
                    'breaks':   bai_perron_breaks,
                },
                'variance': {
                    'n_breaks': len(variance_result['breaks']),
                    'breaks':   variance_result['breaks'],
                },
                'mean_shift': {
                    'n_breaks': len(mean_shift_result['breaks']),
                    'breaks':   mean_shift_result['breaks'],
                },
            },
            'consensus':       consensus_info,
            'all_breaks':      all_breaks,
            'segments':        segments,
            'insights':        insights,
            'recommendations': recommendations,
            'plots':           plots,
        }

        def _deep_native(obj):
            if isinstance(obj, dict):
                return {k: _deep_native(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_deep_native(v) for v in obj]
            elif isinstance(obj, pd.Timestamp):
                return obj.isoformat()
            return _to_native_type(obj)

        return _deep_native(output)

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        raise HTTPException(status_code=500,
                            detail=f"{str(e)}\n{traceback.format_exc()}")
