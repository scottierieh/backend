from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, List, Optional, Union
import numpy as np
import pandas as pd
import io, base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

from scipy import stats
from scipy.signal import find_peaks

router = APIRouter()

class StructuralBreakRequest(BaseModel):
    data: List[dict] = Field(...)
    variable: Union[str, List[str]] = Field(...)
    max_breaks: Optional[int] = 5
    min_segment_pct: Optional[int] = 10
    cusum_threshold: Optional[float] = 1.36

def _to_native(obj):
    if isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj): return None
        return float(obj)
    elif isinstance(obj, np.ndarray): return [_to_native(x) for x in obj.tolist()]
    elif isinstance(obj, np.bool_): return bool(obj)
    elif isinstance(obj, pd.Timestamp): return obj.isoformat()
    elif isinstance(obj, dict): return {str(k): _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)): return [_to_native(x) for x in obj]
    elif pd.isna(obj): return None
    return obj

def safe_float(val, default=0.0):
    try:
        if val is None or pd.isna(val) or np.isinf(val): return default
        return float(val)
    except: return default

def cusum_test(series, threshold=None):
    """CUSUM test for detecting structural breaks"""
    n = len(series)
    mean = series.mean()
    std = series.std()
    
    if std == 0:
        return {'statistic': 0, 'breaks': [], 'cusum': [0] * n, 'threshold': threshold or 1.36, 'significant': False}
    
    cusum = np.zeros(n)
    cusum[0] = (series.iloc[0] - mean) / std
    for i in range(1, n):
        cusum[i] = cusum[i-1] + (series.iloc[i] - mean) / std
    
    cusum_normalized = cusum / np.sqrt(n)
    
    if threshold is None:
        threshold = 1.36
    
    breaks = []
    pos_peaks, _ = find_peaks(cusum_normalized, height=threshold)
    neg_peaks, _ = find_peaks(-cusum_normalized, height=threshold)
    
    all_peaks = sorted(list(pos_peaks) + list(neg_peaks))
    
    for peak in all_peaks:
        breaks.append({
            'index': int(peak),
            'cusum_value': safe_float(cusum_normalized[peak]),
            'direction': 'positive' if cusum_normalized[peak] > 0 else 'negative'
        })
    
    statistic = np.max(np.abs(cusum_normalized))
    
    return {
        'statistic': safe_float(statistic),
        'threshold': safe_float(threshold),
        'significant': bool(statistic > threshold),
        'breaks': breaks,
        'cusum': [safe_float(c) for c in cusum_normalized]
    }

def chow_test(y, x_idx, break_point):
    """Chow test for structural break at a specific point"""
    n = len(y)
    
    if break_point <= 2 or break_point >= n - 2:
        return {'f_statistic': 0, 'p_value': 1.0, 'significant': False}
    
    y1, y2 = y[:break_point], y[break_point:]
    x1, x2 = x_idx[:break_point], x_idx[break_point:]
    
    X_pooled = np.column_stack([np.ones(n), x_idx])
    beta_pooled = np.linalg.lstsq(X_pooled, y, rcond=None)[0]
    residuals_pooled = y - X_pooled @ beta_pooled
    rss_pooled = np.sum(residuals_pooled ** 2)
    
    X1 = np.column_stack([np.ones(len(x1)), x1])
    X2 = np.column_stack([np.ones(len(x2)), x2])
    
    try:
        beta1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
        beta2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
        
        residuals1 = y1 - X1 @ beta1
        residuals2 = y2 - X2 @ beta2
        
        rss_unrestricted = np.sum(residuals1 ** 2) + np.sum(residuals2 ** 2)
    except:
        return {'f_statistic': 0, 'p_value': 1.0, 'significant': False}
    
    k = 2
    if rss_unrestricted == 0:
        return {'f_statistic': 0, 'p_value': 1.0, 'significant': False}
    
    f_stat = ((rss_pooled - rss_unrestricted) / k) / (rss_unrestricted / (n - 2 * k))
    f_stat = max(0, f_stat)
    
    df1, df2 = k, n - 2 * k
    if df2 <= 0:
        return {'f_statistic': safe_float(f_stat), 'p_value': 1.0, 'significant': False}
    
    p_value = 1 - stats.f.cdf(f_stat, df1, df2)
    
    return {
        'f_statistic': safe_float(f_stat),
        'p_value': safe_float(p_value),
        'df1': int(df1),
        'df2': int(df2),
        'significant': bool(p_value < 0.05)
    }

def bai_perron_sequential(series, max_breaks=5, min_segment=None, significance=0.05):
    """Sequential detection of multiple structural breaks"""
    n = len(series)
    
    if min_segment is None:
        min_segment = max(int(n * 0.1), 10)
    
    y = series.values
    x_idx = np.arange(n)
    
    breaks = []
    segments = [(0, n)]
    
    for _ in range(max_breaks):
        best_break = None
        best_f_stat = 0
        best_p_value = 1.0
        best_segment_idx = None
        
        for seg_idx, (start, end) in enumerate(segments):
            segment_length = end - start
            
            if segment_length < 2 * min_segment:
                continue
            
            y_seg = y[start:end]
            x_seg = x_idx[start:end] - start
            
            for bp in range(min_segment, segment_length - min_segment):
                result = chow_test(y_seg, x_seg, bp)
                
                if result['f_statistic'] > best_f_stat and result['p_value'] < significance:
                    best_f_stat = result['f_statistic']
                    best_p_value = result['p_value']
                    best_break = start + bp
                    best_segment_idx = seg_idx
        
        if best_break is None:
            break
        
        breaks.append({
            'index': int(best_break),
            'f_statistic': safe_float(best_f_stat),
            'p_value': safe_float(best_p_value)
        })
        
        old_start, old_end = segments[best_segment_idx]
        segments.pop(best_segment_idx)
        segments.insert(best_segment_idx, (old_start, best_break))
        segments.insert(best_segment_idx + 1, (best_break, old_end))
    
    breaks.sort(key=lambda x: x['index'])
    return breaks, segments

def pettitt_test(series):
    """Pettitt test for detecting a single change point"""
    n = len(series)
    y = series.values
    
    U = np.zeros(n)
    for t in range(n):
        for i in range(t + 1):
            for j in range(t + 1, n):
                U[t] += np.sign(y[j] - y[i])
    
    K = np.max(np.abs(U))
    change_point = np.argmax(np.abs(U))
    
    p_value = 2 * np.exp(-6 * K ** 2 / (n ** 3 + n ** 2))
    p_value = min(1.0, p_value)
    
    return {
        'statistic': safe_float(K),
        'change_point': int(change_point),
        'p_value': safe_float(p_value),
        'significant': bool(p_value < 0.05),
        'U_values': [safe_float(u) for u in U]
    }

def variance_change_test(series, window=None):
    """Detect changes in variance using rolling variance ratio"""
    n = len(series)
    
    if window is None:
        window = max(int(n * 0.1), 10)
    
    rolling_var = series.rolling(window=window, center=True).var()
    overall_var = series.var()
    
    if overall_var == 0:
        return {'breaks': [], 'variance_ratio': [1.0] * n, 'threshold': 2.0}
    
    variance_ratio = rolling_var / overall_var
    threshold = 2.0
    
    breaks = []
    ratio_values = variance_ratio.dropna().values
    ratio_indices = list(variance_ratio.dropna().index)
    
    high_var_peaks, _ = find_peaks(ratio_values, height=threshold)
    
    for peak in high_var_peaks:
        if peak < len(ratio_indices):
            breaks.append({
                'index': int(ratio_indices[peak]),
                'variance_ratio': safe_float(ratio_values[peak]),
                'type': 'increase'
            })
    
    # Detect low variance
    for i, val in enumerate(ratio_values):
        if val < 0.5 and i < len(ratio_indices):
            if i == 0 or ratio_values[i-1] >= 0.5:
                breaks.append({
                    'index': int(ratio_indices[i]),
                    'variance_ratio': safe_float(val),
                    'type': 'decrease'
                })
    
    breaks.sort(key=lambda x: x['index'])
    
    return {
        'breaks': breaks,
        'variance_ratio': [safe_float(v) if not pd.isna(v) else None for v in variance_ratio],
        'threshold': threshold
    }

def mean_shift_detection(series, window=None):
    """Detect mean shifts using moving average comparison"""
    n = len(series)
    
    if window is None:
        window = max(int(n * 0.1), 10)
    
    forward_ma = series.rolling(window=window).mean()
    backward_ma = series.iloc[::-1].rolling(window=window).mean().iloc[::-1]
    
    mean_diff = forward_ma - backward_ma
    rolling_std = series.rolling(window=window).std()
    normalized_diff = mean_diff / (rolling_std + 1e-10)
    
    threshold = 2.0
    breaks = []
    
    diff_values = normalized_diff.dropna().values
    diff_indices = list(normalized_diff.dropna().index)
    
    pos_peaks, _ = find_peaks(diff_values, height=threshold, distance=window)
    neg_peaks, _ = find_peaks(-diff_values, height=threshold, distance=window)
    
    for peak in pos_peaks:
        if peak < len(diff_indices):
            breaks.append({
                'index': int(diff_indices[peak]),
                'normalized_diff': safe_float(diff_values[peak]),
                'direction': 'increase'
            })
    
    for peak in neg_peaks:
        if peak < len(diff_indices):
            breaks.append({
                'index': int(diff_indices[peak]),
                'normalized_diff': safe_float(-diff_values[peak]),
                'direction': 'decrease'
            })
    
    breaks.sort(key=lambda x: x['index'])
    
    return {
        'breaks': breaks,
        'mean_diff': [safe_float(v) if not pd.isna(v) else None for v in normalized_diff],
        'threshold': threshold
    }

def calculate_segment_statistics(series, break_points):
    """Calculate statistics for each segment"""
    n = len(series)
    all_breaks = [0] + sorted(break_points) + [n]
    
    segments = []
    for i in range(len(all_breaks) - 1):
        start, end = all_breaks[i], all_breaks[i + 1]
        segment_data = series.iloc[start:end]
        
        if len(segment_data) > 0:
            segments.append({
                'segment': i + 1,
                'start': int(start),
                'end': int(end),
                'length': int(end - start),
                'mean': safe_float(segment_data.mean()),
                'std': safe_float(segment_data.std()),
                'min': safe_float(segment_data.min()),
                'max': safe_float(segment_data.max()),
                'median': safe_float(segment_data.median())
            })
    
    return segments

def generate_insights(cusum_result, pettitt_result, bai_perron_breaks, variance_result, mean_shift_result, n):
    """Generate insights from structural break analysis"""
    insights = []
    recommendations = []
    
    all_break_indices = set()
    
    for brk in cusum_result.get('breaks', []):
        all_break_indices.add(brk['index'])
    
    if pettitt_result.get('significant', False):
        all_break_indices.add(pettitt_result['change_point'])
    
    for brk in bai_perron_breaks:
        all_break_indices.add(brk['index'])
    
    n_breaks = len(all_break_indices)
    
    if n_breaks == 0:
        insights.append({
            "type": "success",
            "title": "No Structural Breaks Detected",
            "description": "The series appears stable with no significant structural changes detected by any test."
        })
    elif n_breaks == 1:
        insights.append({
            "type": "warning",
            "title": "Single Break Point Detected",
            "description": "One structural break detected, suggesting a regime change at some point in the series."
        })
    else:
        insights.append({
            "type": "warning",
            "title": f"Multiple Breaks Detected ({n_breaks})",
            "description": "Multiple structural breaks suggest the series has undergone several regime changes."
        })
    
    if cusum_result['significant']:
        insights.append({
            "type": "warning",
            "title": "CUSUM Test: Significant",
            "description": f"CUSUM statistic ({cusum_result['statistic']:.4f}) exceeds threshold ({cusum_result['threshold']:.4f})."
        })
    else:
        insights.append({
            "type": "success",
            "title": "CUSUM Test: Not Significant",
            "description": f"CUSUM statistic ({cusum_result['statistic']:.4f}) within threshold bounds."
        })
    
    if pettitt_result['significant']:
        insights.append({
            "type": "warning",
            "title": "Pettitt Test: Significant Change Point",
            "description": f"Change point at index {pettitt_result['change_point']} (p = {pettitt_result['p_value']:.4f})."
        })
    else:
        insights.append({
            "type": "success",
            "title": "Pettitt Test: No Significant Change",
            "description": f"No single dominant change point detected (p = {pettitt_result['p_value']:.4f})."
        })
    
    var_breaks = variance_result.get('breaks', [])
    if len(var_breaks) > 0:
        insights.append({
            "type": "warning",
            "title": "Variance Changes Detected",
            "description": f"Found {len(var_breaks)} variance change(s), indicating heteroscedasticity."
        })
    
    recommendations.extend([
        "Investigate the cause of detected breaks (policy changes, external events, data issues).",
        "Consider modeling each regime separately for better fit.",
        "Use regime-switching models if breaks are recurrent.",
        "Verify breaks aren't due to data collection changes or outliers."
    ])
    
    if n_breaks > 3:
        recommendations.append("Many breaks may indicate high noise - consider smoothing or longer analysis windows.")
    
    return insights, recommendations, list(all_break_indices)

def create_plots(series, cusum_result, pettitt_result, variance_result, all_breaks, segments, variable):
    """Create all visualization plots"""
    plots = {}
    n = len(series)
    x = np.arange(n)
    
    # 1. CUSUM Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    
    axes[0].plot(x, series.values, 'b-', linewidth=1, alpha=0.7)
    for brk in cusum_result['breaks']:
        axes[0].axvline(x=brk['index'], color='red', linestyle='--', linewidth=2, alpha=0.7)
    axes[0].set_ylabel(variable)
    axes[0].set_title(f'Time Series with Detected Breaks: {variable}', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    cusum = cusum_result['cusum']
    threshold = cusum_result['threshold']
    axes[1].plot(x, cusum, 'b-', linewidth=1.5, label='CUSUM')
    axes[1].axhline(y=threshold, color='red', linestyle='--', linewidth=1.5, label=f'±{threshold:.2f}')
    axes[1].axhline(y=-threshold, color='red', linestyle='--', linewidth=1.5)
    axes[1].axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    axes[1].fill_between(x, -threshold, threshold, alpha=0.1, color='green')
    for brk in cusum_result['breaks']:
        axes[1].scatter([brk['index']], [brk['cusum_value']], color='red', s=100, zorder=5)
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('CUSUM')
    axes[1].set_title('CUSUM Test', fontsize=12, fontweight='bold')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    plots['cusum'] = f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"
    
    # 2. Pettitt Plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    cp = pettitt_result['change_point']
    
    axes[0].plot(x, series.values, 'b-', linewidth=1, alpha=0.7)
    axes[0].axvline(x=cp, color='red', linestyle='--', linewidth=2, label=f'Change Point (idx={cp})')
    
    mean_before = series.iloc[:cp].mean() if cp > 0 else series.mean()
    mean_after = series.iloc[cp:].mean() if cp < n else series.mean()
    axes[0].hlines(y=mean_before, xmin=0, xmax=cp, color='green', linewidth=2, label=f'Before: {mean_before:.2f}')
    axes[0].hlines(y=mean_after, xmin=cp, xmax=n, color='orange', linewidth=2, label=f'After: {mean_after:.2f}')
    axes[0].set_ylabel(variable)
    axes[0].set_title(f'Pettitt Test: {variable}', fontsize=12, fontweight='bold')
    axes[0].legend(loc='best')
    axes[0].grid(True, alpha=0.3)
    
    U = pettitt_result['U_values']
    axes[1].plot(x, U, 'purple', linewidth=1.5)
    axes[1].axvline(x=cp, color='red', linestyle='--', linewidth=2)
    axes[1].scatter([cp], [U[cp]], color='red', s=150, zorder=5, marker='*')
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('U Statistic')
    axes[1].set_title(f'Pettitt U Statistic (p = {pettitt_result["p_value"]:.4f})', fontsize=12, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    plots['pettitt'] = f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"
    
    # 3. Breaks Plot with Segments
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(x, series.values, 'b-', linewidth=1, alpha=0.5, label='Original')
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(segments)))
    for seg, color in zip(segments, colors):
        seg_x = np.arange(seg['start'], seg['end'])
        seg_y = series.iloc[seg['start']:seg['end']].values
        ax.fill_between(seg_x, seg_y.min(), seg_y.max(), alpha=0.2, color=color)
        ax.hlines(y=seg['mean'], xmin=seg['start'], xmax=seg['end'], color=color, linewidth=3, 
                  label=f"Seg {seg['segment']} μ={seg['mean']:.2f}")
    
    for brk in all_breaks:
        ax.axvline(x=brk, color='red', linestyle='--', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Index', fontsize=12)
    ax.set_ylabel(variable, fontsize=12)
    ax.set_title(f'Structural Breaks and Segments: {variable}', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    plots['breaks'] = f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"
    
    # 4. Summary Plot
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(x, series.values, 'b-', linewidth=1.5, alpha=0.8, label='Time Series')
    
    window = max(n // 20, 5)
    rolling_mean = series.rolling(window=window).mean()
    ax.plot(x, rolling_mean.values, 'orange', linewidth=2, alpha=0.8, label=f'Rolling Mean (w={window})')
    
    unique_breaks = sorted(set(all_breaks))
    for i, brk in enumerate(unique_breaks):
        ax.axvline(x=brk, color='red', linestyle='--', linewidth=2, alpha=0.8,
                   label='Break Points' if i == 0 else '')
    
    ax.set_xlabel('Index', fontsize=12)
    ax.set_ylabel(variable, fontsize=12)
    ax.set_title(f'Structural Break Summary: {variable} ({len(unique_breaks)} breaks)', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    plots['summary'] = f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"
    
    return plots

@router.post("/structural-break")
def structural_break_analysis(req: StructuralBreakRequest):
    try:
        df = pd.DataFrame(req.data)
        variable = req.variable
        max_breaks = req.max_breaks or 5
        min_segment_pct = req.min_segment_pct or 10
        cusum_threshold = req.cusum_threshold or 1.36
        
        if isinstance(variable, list):
            variable = variable[0]
        
        if variable not in df.columns:
            raise ValueError(f"Variable '{variable}' not found")
        
        series = pd.to_numeric(df[variable], errors='coerce').dropna().reset_index(drop=True)
        n = len(series)
        
        if n < 20:
            raise ValueError(f"Need at least 20 observations, got {n}.")
        
        min_segment = max(int(n * min_segment_pct / 100), 5)
        
        # Run all tests
        cusum_result = cusum_test(series, threshold=cusum_threshold)
        pettitt_result = pettitt_test(series)
        bai_perron_breaks, bp_segments = bai_perron_sequential(series, max_breaks=max_breaks, min_segment=min_segment)
        variance_result = variance_change_test(series)
        mean_shift_result = mean_shift_detection(series)
        
        # Generate insights
        insights, recommendations, all_breaks = generate_insights(
            cusum_result, pettitt_result, bai_perron_breaks, variance_result, mean_shift_result, n
        )
        
        # Calculate segment statistics
        segments = calculate_segment_statistics(series, all_breaks)
        
        # Generate plots
        plots = create_plots(series, cusum_result, pettitt_result, variance_result, all_breaks, segments, variable)
        
        # Interpretation text
        interpretation_parts = []
        interpretation_parts.append("**Overall Analysis**")
        interpretation_parts.append(f"→ Structural break analysis performed on **{variable}** with **{n}** observations.")
        interpretation_parts.append(f"→ Total unique breaks detected: **{len(all_breaks)}**")
        
        interpretation_parts.append("")
        interpretation_parts.append("**Key Insights**")
        
        if cusum_result['significant']:
            interpretation_parts.append(f"→ **CUSUM Test**: Significant (statistic = {cusum_result['statistic']:.4f})")
        else:
            interpretation_parts.append(f"→ **CUSUM Test**: Not significant (statistic = {cusum_result['statistic']:.4f})")
        
        interpretation_parts.append(f"→ **Pettitt Test**: Change point at index {pettitt_result['change_point']} (p = {pettitt_result['p_value']:.4f})")
        interpretation_parts.append(f"→ **Bai-Perron**: {len(bai_perron_breaks)} break(s) detected")
        
        if len(segments) > 1:
            means = [seg['mean'] for seg in segments]
            interpretation_parts.append(f"→ **Segments**: {len(segments)} segments with means ranging from {min(means):.2f} to {max(means):.2f}")
        
        interpretation_parts.append("")
        interpretation_parts.append("**Recommendations**")
        for rec in recommendations[:4]:
            interpretation_parts.append(f"→ {rec}")
        
        interpretation = "\n".join(interpretation_parts)
        
        response = {
            'variable': variable,
            'n_observations': n,
            'tests': {
                'cusum': {
                    'statistic': cusum_result['statistic'],
                    'threshold': cusum_result['threshold'],
                    'significant': cusum_result['significant'],
                    'n_breaks': len(cusum_result['breaks']),
                    'breaks': cusum_result['breaks']
                },
                'pettitt': {
                    'statistic': pettitt_result['statistic'],
                    'change_point': pettitt_result['change_point'],
                    'p_value': pettitt_result['p_value'],
                    'significant': pettitt_result['significant']
                },
                'bai_perron': {
                    'n_breaks': len(bai_perron_breaks),
                    'breaks': bai_perron_breaks
                },
                'variance': {
                    'n_breaks': len(variance_result['breaks']),
                    'breaks': variance_result['breaks']
                },
                'mean_shift': {
                    'n_breaks': len(mean_shift_result['breaks']),
                    'breaks': mean_shift_result['breaks']
                }
            },
            'all_breaks': sorted(all_breaks),
            'segments': segments,
            'insights': insights,
            'recommendations': recommendations,
            'interpretation': interpretation,
            'plots': plots
        }
        
        return _to_native(response)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
