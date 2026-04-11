"""
Zero Crossing Rate Analysis Backend (FastAPI)
- Comprehensive ZCR profiling of audio signals
- Frame-wise ZCR with temporal dynamics and statistics
- ZCR-based signal characterization (voiced/unvoiced/noise/tonal)
- ZCR distribution analysis and percentile profiling
- ZCR vs RMS energy correlation for signal typing
- Band-filtered ZCR analysis (low/mid/high frequency content)
- Segment-wise ZCR trends, stability, and change-point detection
- ZCR histogram, autocorrelation, and periodicity analysis
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional, List
import numpy as np
import pandas as pd
import io, base64, tempfile, os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

try:
    import librosa
    import librosa.display
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

from scipy.signal import butter, sosfilt

sns.set_theme(style="darkgrid")

router = APIRouter()


# ──────────────────────────────────────────────
# Utility
# ──────────────────────────────────────────────

def _to_native(obj):
    if isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj): return None
        return float(obj)
    elif isinstance(obj, np.ndarray): return [_to_native(x) for x in obj.tolist()]
    elif isinstance(obj, np.bool_): return bool(obj)
    elif isinstance(obj, dict): return {str(k): _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)): return [_to_native(x) for x in obj]
    return obj


def safe_float(val, default=0.0):
    try:
        if val is None or pd.isna(val) or np.isinf(val): return default
        return float(val)
    except: return default


# ──────────────────────────────────────────────
# ZCR Analysis Functions
# ──────────────────────────────────────────────

def compute_zcr_profile(y: np.ndarray, sr: int, frame_length: int = 2048, hop_length: int = 512):
    """Compute frame-wise zero crossing rate."""
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(zcr)), sr=sr, hop_length=hop_length)
    return zcr, times


def compute_zcr_statistics(zcr: np.ndarray, sr: int, frame_length: int):
    """Compute comprehensive ZCR statistics."""
    pcts = np.percentile(zcr, [1, 5, 10, 25, 50, 75, 90, 95, 99])

    # Approximate dominant frequency from mean ZCR
    # ZCR ≈ 2 * f0 / sr  →  f0 ≈ ZCR * sr / 2
    approx_freq = np.mean(zcr) * sr / 2

    # ZCR variability
    zcr_diff = np.diff(zcr)
    volatility = np.std(zcr_diff)

    # Coefficient of variation
    cv = np.std(zcr) / (np.mean(zcr) + 1e-10)

    return {
        'mean_zcr': safe_float(np.mean(zcr)),
        'median_zcr': safe_float(np.median(zcr)),
        'std_zcr': safe_float(np.std(zcr)),
        'min_zcr': safe_float(np.min(zcr)),
        'max_zcr': safe_float(np.max(zcr)),
        'range_zcr': safe_float(np.max(zcr) - np.min(zcr)),
        'cv_zcr': safe_float(cv),
        'volatility': safe_float(volatility),
        'approx_dominant_freq_hz': safe_float(approx_freq),
        'skewness': safe_float(float(pd.Series(zcr).skew())),
        'kurtosis': safe_float(float(pd.Series(zcr).kurtosis())),
        'percentiles': {
            'p1': safe_float(pcts[0]), 'p5': safe_float(pcts[1]), 'p10': safe_float(pcts[2]),
            'p25': safe_float(pcts[3]), 'p50': safe_float(pcts[4]), 'p75': safe_float(pcts[5]),
            'p90': safe_float(pcts[6]), 'p95': safe_float(pcts[7]), 'p99': safe_float(pcts[8]),
        },
        'iqr': safe_float(pcts[5] - pcts[3]),
    }


def classify_zcr_frames(zcr: np.ndarray, rms_db: np.ndarray,
                         noise_zcr_threshold: float = 0.3,
                         voiced_zcr_threshold: float = 0.1,
                         silence_rms_threshold: float = -40):
    """Classify frames as voiced, unvoiced, noise, or silence using ZCR + RMS."""
    n = len(zcr)
    rms_trimmed = rms_db[:n] if len(rms_db) >= n else np.pad(rms_db, (0, n - len(rms_db)), constant_values=-100)

    classes = []
    counts = {'silence': 0, 'voiced': 0, 'unvoiced': 0, 'noise': 0}

    for i in range(n):
        if rms_trimmed[i] < silence_rms_threshold:
            classes.append('silence')
            counts['silence'] += 1
        elif zcr[i] >= noise_zcr_threshold:
            classes.append('noise')
            counts['noise'] += 1
        elif zcr[i] <= voiced_zcr_threshold:
            classes.append('voiced')
            counts['voiced'] += 1
        else:
            classes.append('unvoiced')
            counts['unvoiced'] += 1

    total = max(n, 1)
    distribution = {
        'silence_percent': safe_float(counts['silence'] / total * 100),
        'voiced_percent': safe_float(counts['voiced'] / total * 100),
        'unvoiced_percent': safe_float(counts['unvoiced'] / total * 100),
        'noise_percent': safe_float(counts['noise'] / total * 100),
        'silence_frames': counts['silence'],
        'voiced_frames': counts['voiced'],
        'unvoiced_frames': counts['unvoiced'],
        'noise_frames': counts['noise'],
    }
    return classes, distribution


def compute_zcr_rms_correlation(zcr: np.ndarray, y: np.ndarray, sr: int,
                                 frame_length: int, hop_length: int):
    """Compute correlation between ZCR and RMS energy."""
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    n = min(len(zcr), len(rms))
    zcr_t = zcr[:n]
    rms_t = rms[:n]
    rms_db = 20 * np.log10(rms_t + 1e-10)

    # Pearson correlation
    if np.std(zcr_t) > 0 and np.std(rms_t) > 0:
        corr_linear = safe_float(np.corrcoef(zcr_t, rms_t)[0, 1])
        corr_db = safe_float(np.corrcoef(zcr_t, rms_db)[0, 1])
    else:
        corr_linear = 0.0
        corr_db = 0.0

    return {
        'correlation_linear': corr_linear,
        'correlation_db': corr_db,
        'rms': rms_t,
        'rms_db': rms_db,
    }


def compute_band_zcr(y: np.ndarray, sr: int, frame_length: int, hop_length: int):
    """Compute ZCR after bandpass filtering into low/mid/high bands."""
    nyq = sr / 2
    bands = []

    band_defs = [
        ('Low (0-500 Hz)', 20, min(500, nyq - 1)),
        ('Mid (500-4k Hz)', 500, min(4000, nyq - 1)),
        ('High (4k+ Hz)', 4000, min(nyq - 1, nyq - 1)),
    ]

    for name, f_low, f_high in band_defs:
        if f_low >= nyq or f_high <= f_low:
            continue
        try:
            sos = butter(4, [f_low / nyq, f_high / nyq], btype='band', output='sos')
            y_filt = sosfilt(sos, y)
            zcr_band = librosa.feature.zero_crossing_rate(y=y_filt, frame_length=frame_length, hop_length=hop_length)[0]
            bands.append({
                'band': name,
                'mean_zcr': safe_float(np.mean(zcr_band)),
                'std_zcr': safe_float(np.std(zcr_band)),
                'max_zcr': safe_float(np.max(zcr_band)),
                'median_zcr': safe_float(np.median(zcr_band)),
            })
        except:
            continue

    return bands


def compute_zcr_segments(zcr: np.ndarray, times: np.ndarray, n_segments: int = 10):
    """Per-segment ZCR statistics."""
    n = len(zcr)
    seg_len = max(1, n // n_segments)
    segments = []

    for i in range(n_segments):
        start = i * seg_len
        end = min(start + seg_len, n)
        if start >= n:
            break
        seg = zcr[start:end]
        segments.append({
            'segment': i + 1,
            'start_sec': safe_float(times[start]),
            'end_sec': safe_float(times[min(end - 1, n - 1)]),
            'mean_zcr': safe_float(np.mean(seg)),
            'std_zcr': safe_float(np.std(seg)),
            'max_zcr': safe_float(np.max(seg)),
            'min_zcr': safe_float(np.min(seg)),
            'range_zcr': safe_float(np.max(seg) - np.min(seg)),
            'approx_freq_hz': safe_float(np.mean(seg) * len(seg) * (times[1] - times[0]) if len(times) > 1 else 0),
        })

    # Trend: is ZCR increasing or decreasing over time?
    seg_means = [s['mean_zcr'] for s in segments]
    if len(seg_means) > 2:
        x = np.arange(len(seg_means))
        slope = np.polyfit(x, seg_means, 1)[0]
    else:
        slope = 0.0

    return segments, safe_float(slope)


def compute_zcr_autocorrelation(zcr: np.ndarray, max_lag: int = 50):
    """Compute autocorrelation of ZCR to detect periodicity."""
    n = len(zcr)
    zcr_centered = zcr - np.mean(zcr)
    var = np.var(zcr)
    if var < 1e-10:
        return {'lags': list(range(min(max_lag, n))), 'values': [0.0] * min(max_lag, n), 'periodic': False, 'period_frames': 0}

    max_lag = min(max_lag, n - 1)
    acf = []
    for lag in range(max_lag):
        c = np.mean(zcr_centered[:n - lag] * zcr_centered[lag:])
        acf.append(safe_float(c / var))

    # Find first significant peak after lag 0
    periodic = False
    period = 0
    for i in range(2, len(acf) - 1):
        if acf[i] > acf[i - 1] and acf[i] > acf[i + 1] and acf[i] > 0.3:
            periodic = True
            period = i
            break

    return {
        'lags': list(range(max_lag)),
        'values': acf,
        'periodic': periodic,
        'period_frames': period,
    }


def generate_interpretation(stats, distribution, zcr_rms_corr, band_zcr, segments, trend_slope, autocorr):
    """Generate human-readable interpretation."""
    lines = []

    # Overall ZCR characterization
    mean_zcr = stats['mean_zcr']
    if mean_zcr > 0.3:
        lines.append(f"High mean ZCR ({mean_zcr:.4f}) — signal is dominated by noise-like or high-frequency content. Approximate dominant frequency: {stats['approx_dominant_freq_hz']:.0f} Hz.")
    elif mean_zcr > 0.1:
        lines.append(f"Moderate mean ZCR ({mean_zcr:.4f}) — mixed voiced/unvoiced content. Approximate dominant frequency: {stats['approx_dominant_freq_hz']:.0f} Hz.")
    else:
        lines.append(f"Low mean ZCR ({mean_zcr:.4f}) — signal is predominantly low-frequency or tonal. Approximate dominant frequency: {stats['approx_dominant_freq_hz']:.0f} Hz.")

    # Variability
    cv = stats['cv_zcr']
    if cv > 1.0:
        lines.append(f"Very high ZCR variability (CV: {cv:.2f}) — signal character changes dramatically over time.")
    elif cv > 0.5:
        lines.append(f"Moderate ZCR variability (CV: {cv:.2f}) — some variation in signal character.")
    else:
        lines.append(f"Low ZCR variability (CV: {cv:.2f}) — consistent signal character throughout.")

    # Classification
    v = distribution['voiced_percent']
    uv = distribution['unvoiced_percent']
    ns = distribution['noise_percent']
    sl = distribution['silence_percent']
    if v > 50:
        lines.append(f"Signal is {v:.0f}% voiced — predominantly harmonic/tonal content (e.g. speech vowels, musical tones).")
    elif ns > 50:
        lines.append(f"Signal is {ns:.0f}% noise-like — broadband or high-frequency dominated content.")
    elif uv > 40:
        lines.append(f"Signal is {uv:.0f}% unvoiced — fricative-like or mixed-frequency content.")
    if sl > 30:
        lines.append(f"Signal contains {sl:.0f}% silence.")

    # ZCR-RMS correlation
    corr = zcr_rms_corr['correlation_db']
    if corr > 0.5:
        lines.append(f"Positive ZCR–RMS correlation ({corr:.2f}) — louder sections tend to have higher ZCR (higher frequency content).")
    elif corr < -0.5:
        lines.append(f"Negative ZCR–RMS correlation ({corr:.2f}) — louder sections tend to have lower ZCR (lower frequency / tonal content).")
    else:
        lines.append(f"Weak ZCR–RMS correlation ({corr:.2f}) — energy and zero-crossing rate are largely independent.")

    # Trend
    if abs(trend_slope) > 0.001:
        direction = 'increasing' if trend_slope > 0 else 'decreasing'
        lines.append(f"ZCR trend is {direction} over time (slope: {trend_slope:.5f}/segment).")

    # Periodicity
    if autocorr['periodic']:
        lines.append(f"Periodic ZCR pattern detected with period ≈ {autocorr['period_frames']} frames.")

    # Band analysis
    if band_zcr:
        dom_band = max(band_zcr, key=lambda b: b['mean_zcr'])
        lines.append(f"Highest ZCR in {dom_band['band']} band ({dom_band['mean_zcr']:.4f}).")

    return {
        'summary': ' '.join(lines[:2]),
        'details': lines,
    }


def build_zcr_table(zcr, times, rms_db, n_points=400):
    """Build sampled ZCR + RMS data for frontend."""
    n = min(len(zcr), len(times), len(rms_db))
    step = max(1, n // n_points)
    table = []
    for i in range(0, n, step):
        table.append({
            'time_sec': safe_float(times[i]),
            'zcr': safe_float(zcr[i]),
            'rms_db': safe_float(rms_db[i]),
        })
    return table


# ──────────────────────────────────────────────
# Plot Generation
# ──────────────────────────────────────────────

def generate_zcr_plots(
    y, sr, zcr, times, stats, distribution, frame_classes,
    zcr_rms_corr, band_zcr, segments, autocorr, hop_length,
) -> str:
    """Generate comprehensive ZCR visualization."""

    fig = plt.figure(figsize=(18, 30))
    fig.suptitle('Zero Crossing Rate Analysis', fontsize=18, fontweight='bold')
    gs = gridspec.GridSpec(7, 2, figure=fig, hspace=0.55, wspace=0.3)

    # ── 1. Waveform ──
    ax1 = fig.add_subplot(gs[0, :])
    t_wav = np.arange(len(y)) / sr
    ax1.plot(t_wav, y, color='#2196F3', linewidth=0.3, alpha=0.6)
    ax1.set_title('Time Domain — Waveform', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, linestyle='--', alpha=0.4)

    # ── 2. ZCR Profile ──
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(times, zcr, color='#9C27B0', linewidth=0.7)
    ax2.fill_between(times, zcr, alpha=0.15, color='#9C27B0')
    ax2.axhline(y=stats['mean_zcr'], color='#F44336', linestyle='--', linewidth=0.8, label=f'Mean: {stats["mean_zcr"]:.4f}')
    ax2.axhline(y=stats['percentiles']['p95'], color='#4CAF50', linestyle=':', linewidth=0.8, alpha=0.6, label=f'P95: {stats["percentiles"]["p95"]:.4f}')
    ax2.axhline(y=stats['percentiles']['p5'], color='#2196F3', linestyle=':', linewidth=0.8, alpha=0.6, label=f'P5: {stats["percentiles"]["p5"]:.4f}')
    ax2.set_title('Zero Crossing Rate over Time', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('ZCR')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, linestyle='--', alpha=0.4)

    # ── 3. Signal Classification ──
    ax3 = fig.add_subplot(gs[2, 0])
    cls_colors = {'silence': '#BDBDBD', 'voiced': '#4CAF50', 'unvoiced': '#FF9800', 'noise': '#F44336'}
    n_cls = min(len(times), len(frame_classes))
    for cls in ['silence', 'voiced', 'unvoiced', 'noise']:
        mask = np.array([1.0 if frame_classes[i] == cls else 0.0 for i in range(n_cls)])
        if np.any(mask > 0):
            ax3.fill_between(times[:n_cls], 0, mask, alpha=0.5, color=cls_colors[cls], label=cls.capitalize())
    ax3.set_title('Frame Classification (ZCR + RMS)', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Active')
    ax3.legend(loc='upper right', fontsize=7)
    ax3.grid(True, linestyle='--', alpha=0.4)

    # ── 4. Classification Pie ──
    ax4 = fig.add_subplot(gs[2, 1])
    pie_labels = ['Silence', 'Voiced', 'Unvoiced', 'Noise']
    pie_vals = [distribution['silence_percent'], distribution['voiced_percent'],
                distribution['unvoiced_percent'], distribution['noise_percent']]
    pie_colors = ['#BDBDBD', '#4CAF50', '#FF9800', '#F44336']
    non_zero = [(l, v, c) for l, v, c in zip(pie_labels, pie_vals, pie_colors) if v > 0.5]
    if non_zero:
        ls, vs, cs = zip(*non_zero)
        ax4.pie(vs, labels=ls, autopct='%1.1f%%', colors=cs, startangle=90, textprops={'fontsize': 10})
    ax4.set_title('Signal Type Distribution', fontsize=13, fontweight='bold')

    # ── 5. ZCR Histogram ──
    ax5 = fig.add_subplot(gs[3, 0])
    ax5.hist(zcr, bins=60, color='#9C27B0', edgecolor='white', linewidth=0.3, alpha=0.85)
    ax5.axvline(x=stats['mean_zcr'], color='#F44336', linestyle='--', linewidth=1, label=f'Mean: {stats["mean_zcr"]:.4f}')
    ax5.axvline(x=stats['median_zcr'], color='#4CAF50', linestyle='--', linewidth=1, label=f'Median: {stats["median_zcr"]:.4f}')
    ax5.set_title('ZCR Distribution', fontsize=13, fontweight='bold')
    ax5.set_xlabel('ZCR')
    ax5.set_ylabel('Frame Count')
    ax5.legend(fontsize=8)
    ax5.grid(True, linestyle='--', alpha=0.4, axis='y')

    # ── 6. ZCR vs RMS Scatter ──
    ax6 = fig.add_subplot(gs[3, 1])
    rms_db_corr = zcr_rms_corr['rms_db']
    n_plot = min(len(zcr), len(rms_db_corr), 2000)
    step = max(1, min(len(zcr), len(rms_db_corr)) // n_plot)
    ax6.scatter(zcr[::step], rms_db_corr[::step], alpha=0.3, s=8, color='#2196F3', edgecolors='none')
    ax6.set_title(f'ZCR vs RMS Energy (r={zcr_rms_corr["correlation_db"]:.3f})', fontsize=13, fontweight='bold')
    ax6.set_xlabel('ZCR')
    ax6.set_ylabel('RMS (dB)')
    ax6.grid(True, linestyle='--', alpha=0.4)

    # ── 7. Segment-wise ZCR ──
    ax7 = fig.add_subplot(gs[4, 0])
    seg_labels = [f'S{s["segment"]}' for s in segments]
    seg_means = [s['mean_zcr'] for s in segments]
    seg_stds = [s['std_zcr'] for s in segments]
    x_seg = np.arange(len(segments))
    colors_seg = ['#F44336' if v == max(seg_means) else '#4CAF50' if v == min(seg_means) else '#9C27B0' for v in seg_means]
    ax7.bar(x_seg, seg_means, yerr=seg_stds, color=colors_seg, edgecolor='white', linewidth=0.8, capsize=3)
    ax7.set_title('Segment-wise Mean ZCR', fontsize=13, fontweight='bold')
    ax7.set_xlabel('Segment')
    ax7.set_ylabel('Mean ZCR')
    ax7.set_xticks(x_seg)
    ax7.set_xticklabels(seg_labels, fontsize=8)
    ax7.grid(True, linestyle='--', alpha=0.4, axis='y')

    # ── 8. Band ZCR ──
    ax8 = fig.add_subplot(gs[4, 1])
    if band_zcr:
        b_names = [b['band'].split('(')[0].strip() for b in band_zcr]
        b_means = [b['mean_zcr'] for b in band_zcr]
        b_stds = [b['std_zcr'] for b in band_zcr]
        b_colors = ['#3F51B5', '#4CAF50', '#FF5722'][:len(b_names)]
        ax8.bar(b_names, b_means, yerr=b_stds, color=b_colors, edgecolor='white', linewidth=0.8, capsize=3)
    ax8.set_title('Band-Filtered ZCR', fontsize=13, fontweight='bold')
    ax8.set_ylabel('Mean ZCR')
    ax8.grid(True, linestyle='--', alpha=0.4, axis='y')

    # ── 9. Autocorrelation ──
    ax9 = fig.add_subplot(gs[5, 0])
    if autocorr['values']:
        ax9.bar(autocorr['lags'], autocorr['values'], color='#00BCD4', edgecolor='white', linewidth=0.3, alpha=0.85)
        ax9.axhline(y=0, color='gray', linewidth=0.5)
        ax9.axhline(y=0.3, color='#F44336', linestyle='--', linewidth=0.8, alpha=0.5, label='Significance (0.3)')
        if autocorr['periodic']:
            ax9.axvline(x=autocorr['period_frames'], color='#4CAF50', linestyle='--', linewidth=1,
                        label=f'Period: {autocorr["period_frames"]} frames')
        ax9.legend(fontsize=8)
    ax9.set_title('ZCR Autocorrelation', fontsize=13, fontweight='bold')
    ax9.set_xlabel('Lag (frames)')
    ax9.set_ylabel('Autocorrelation')
    ax9.grid(True, linestyle='--', alpha=0.4)

    # ── 10. Percentile Profile ──
    ax10 = fig.add_subplot(gs[5, 1])
    pct_labels = ['P1', 'P5', 'P10', 'P25', 'P50', 'P75', 'P90', 'P95', 'P99']
    pct_vals = [stats['percentiles'][k] for k in ['p1', 'p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99']]
    ax10.barh(pct_labels, pct_vals, color='#9C27B0', edgecolor='white', linewidth=0.8)
    ax10.set_title('ZCR Percentile Profile', fontsize=13, fontweight='bold')
    ax10.set_xlabel('ZCR')
    ax10.grid(True, linestyle='--', alpha=0.4, axis='x')

    # ── 11. ZCR Delta (frame-to-frame change) ──
    ax11 = fig.add_subplot(gs[6, 0])
    zcr_delta = np.diff(zcr)
    delta_times = times[:len(zcr_delta)]
    ax11.plot(delta_times, zcr_delta, color='#FF5722', linewidth=0.5, alpha=0.7)
    ax11.fill_between(delta_times, zcr_delta, alpha=0.1, color='#FF5722')
    ax11.axhline(y=0, color='gray', linewidth=0.5)
    ax11.set_title('ZCR Delta (Frame-to-Frame Change)', fontsize=13, fontweight='bold')
    ax11.set_xlabel('Time (s)')
    ax11.set_ylabel('ΔZCR')
    ax11.grid(True, linestyle='--', alpha=0.4)

    # ── 12. Key Metrics ──
    ax12 = fig.add_subplot(gs[6, 1])
    met_labels = ['Mean\nZCR', 'Std\nZCR', 'CV', 'Volatility', 'Approx\nFreq (Hz)']
    met_vals = [stats['mean_zcr'], stats['std_zcr'], stats['cv_zcr'], stats['volatility'],
                stats['approx_dominant_freq_hz'] / 1000]  # show in kHz scale
    met_colors = ['#9C27B0', '#FF5722', '#4CAF50', '#2196F3', '#FF9800']
    bars12 = ax12.bar(met_labels, met_vals, color=met_colors, edgecolor='white', linewidth=0.8)
    ax12.set_title('Key ZCR Metrics', fontsize=13, fontweight='bold')
    ax12.set_ylabel('Value')
    ax12.grid(True, linestyle='--', alpha=0.4, axis='y')
    for bar, val, label in zip(bars12, met_vals, met_labels):
        display = f'{val:.4f}' if 'Freq' not in label else f'{val * 1000:.0f} Hz'
        ax12.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                 display, ha='center', va='bottom', fontsize=8, fontweight='bold')

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"


# ──────────────────────────────────────────────
# Main Endpoint
# ──────────────────────────────────────────────

@router.post("/zcr-analysis")
async def zcr_analysis(
    file: UploadFile = File(...),
    frame_length: Optional[int] = Form(2048),
    hop_length: Optional[int] = Form(512),
    n_segments: Optional[int] = Form(10),
    noise_zcr_threshold: Optional[float] = Form(0.3),
    voiced_zcr_threshold: Optional[float] = Form(0.1),
    silence_rms_threshold: Optional[float] = Form(-40),
    target_sr: Optional[int] = Form(None),
):
    """
    Zero Crossing Rate Analysis.

    Comprehensive ZCR profiling including frame-wise statistics, signal
    classification (voiced/unvoiced/noise/silence), ZCR-RMS correlation,
    band-filtered ZCR, segment analysis, and autocorrelation.

    Parameters:
    - file: Audio file (WAV, MP3, FLAC, etc.)
    - frame_length: Frame length for ZCR computation (default: 2048)
    - hop_length: Hop length (default: 512)
    - n_segments: Number of temporal segments (default: 10)
    - noise_zcr_threshold: ZCR above this = noise (default: 0.3)
    - voiced_zcr_threshold: ZCR below this = voiced (default: 0.1)
    - silence_rms_threshold: RMS below this (dB) = silence (default: -40)
    - target_sr: Resample to this rate (default: keep original)
    """
    if not LIBROSA_AVAILABLE:
        raise HTTPException(status_code=500, detail="librosa not installed. pip install librosa")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    allowed_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(status_code=400, detail=f"Unsupported format: {file_ext}")

    try:
        content = await file.read()
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            sr_param = target_sr if target_sr else None
            y, sr = librosa.load(tmp_path, sr=sr_param, mono=True)

            max_samples = sr * 60
            if len(y) > max_samples:
                y = y[:max_samples]

            fl = frame_length or 2048
            hl = hop_length or 512
            n_seg = n_segments or 10
            noise_th = noise_zcr_threshold if noise_zcr_threshold is not None else 0.3
            voiced_th = voiced_zcr_threshold if voiced_zcr_threshold is not None else 0.1
            sil_th = silence_rms_threshold if silence_rms_threshold is not None else -40

            # ZCR profile
            zcr, times = compute_zcr_profile(y, sr, frame_length=fl, hop_length=hl)

            # Statistics
            stats = compute_zcr_statistics(zcr, sr, fl)

            # RMS for classification & correlation
            rms = librosa.feature.rms(y=y, frame_length=fl, hop_length=hl)[0]
            rms_db = 20 * np.log10(rms + 1e-10)

            # Classification
            frame_classes, distribution = classify_zcr_frames(zcr, rms_db, noise_th, voiced_th, sil_th)

            # ZCR-RMS correlation
            zcr_rms_corr = compute_zcr_rms_correlation(zcr, y, sr, fl, hl)

            # Band ZCR
            band_zcr = compute_band_zcr(y, sr, fl, hl)

            # Segments
            segments, trend_slope = compute_zcr_segments(zcr, times, n_segments=n_seg)

            # Autocorrelation
            autocorr = compute_zcr_autocorrelation(zcr, max_lag=50)

            # Interpretation
            interpretation = generate_interpretation(stats, distribution, zcr_rms_corr, band_zcr, segments, trend_slope, autocorr)

            # Table
            zcr_table = build_zcr_table(zcr, times, rms_db, n_points=400)

            # Plot
            plot = generate_zcr_plots(
                y, sr, zcr, times, stats, distribution, frame_classes,
                zcr_rms_corr, band_zcr, segments, autocorr, hl
            )

            file_info = {
                'filename': file.filename,
                'file_size_mb': round(len(content) / (1024 * 1024), 2),
                'sample_rate': sr,
                'duration_sec': round(len(y) / sr, 2),
                'n_samples': len(y),
                'format': file_ext.replace('.', ''),
                'frame_length': fl,
                'hop_length': hl,
                'n_segments': n_seg,
                'n_frames': len(zcr),
                'time_resolution_ms': safe_float(hl / sr * 1000),
                'noise_zcr_threshold': safe_float(noise_th),
                'voiced_zcr_threshold': safe_float(voiced_th),
                'silence_rms_threshold': safe_float(sil_th),
            }

            return _to_native({
                'results': {
                    'file_info': file_info,
                    'zcr_statistics': stats,
                    'signal_classification': distribution,
                    'zcr_rms_correlation': {
                        'correlation_linear': zcr_rms_corr['correlation_linear'],
                        'correlation_db': zcr_rms_corr['correlation_db'],
                    },
                    'band_zcr': band_zcr,
                    'segments': segments,
                    'trend_slope': safe_float(trend_slope),
                    'autocorrelation': {
                        'periodic': autocorr['periodic'],
                        'period_frames': autocorr['period_frames'],
                    },
                    'zcr_table': zcr_table,
                    'interpretation': interpretation,
                },
                'plot': plot
            })

        finally:
            os.unlink(tmp_path)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ──────────────────────────────────────────────
# Router registration:
# from zcr_analysis import router as zcr_router
# app.include_router(zcr_router, prefix="/api/analysis", tags=["Audio Analysis"])
# → POST /api/analysis/zcr-analysis
# ──────────────────────────────────────────────
