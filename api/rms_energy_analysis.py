"""
RMS Energy Analysis Backend (FastAPI)
- Comprehensive RMS energy profiling of audio signals
- Frame-wise, segment-wise, and band-wise RMS computation
- Loudness contour, energy distribution, and dynamic range analysis
- A-weighting and C-weighting perceptual loudness approximation
- Energy-based silence/speech/noise classification per frame
- Fade detection, energy stability, and temporal energy patterns
- LUFS-approximated integrated loudness
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

from scipy.signal import lfilter

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
# RMS Analysis Functions
# ──────────────────────────────────────────────

def compute_rms_profile(y: np.ndarray, sr: int, frame_length: int = 2048, hop_length: int = 512):
    """Compute frame-wise RMS energy."""
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    rms_db = 20 * np.log10(rms + 1e-10)
    return rms, rms_db, times


def compute_rms_statistics(rms: np.ndarray, rms_db: np.ndarray, times: np.ndarray, y: np.ndarray, sr: int):
    """Compute comprehensive RMS statistics."""
    overall_rms = np.sqrt(np.mean(y ** 2))
    overall_rms_db = 20 * np.log10(overall_rms + 1e-10)

    # Dynamic range from RMS
    rms_sorted = np.sort(rms)
    n = len(rms_sorted)
    p5 = rms_sorted[max(0, int(n * 0.05))]
    p95 = rms_sorted[min(n - 1, int(n * 0.95))]
    rms_dynamic_range = 20 * np.log10((p95 + 1e-10) / (p5 + 1e-10))

    # Energy percentiles
    pcts = np.percentile(rms_db, [1, 5, 10, 25, 50, 75, 90, 95, 99])

    # Loudness range (LRA approximation — P95-P10 of short-term loudness)
    lra_approx = pcts[6] - pcts[2]  # P90 - P10

    # Temporal statistics
    rms_diff = np.diff(rms_db)
    rms_volatility = np.std(rms_diff)

    # Peak RMS frame
    peak_idx = int(np.argmax(rms))
    min_idx = int(np.argmin(rms[rms > 0])) if np.any(rms > 0) else 0

    return {
        'overall_rms': safe_float(overall_rms),
        'overall_rms_db': safe_float(overall_rms_db),
        'mean_rms': safe_float(np.mean(rms)),
        'mean_rms_db': safe_float(np.mean(rms_db)),
        'median_rms_db': safe_float(np.median(rms_db)),
        'std_rms': safe_float(np.std(rms)),
        'std_rms_db': safe_float(np.std(rms_db)),
        'max_rms': safe_float(np.max(rms)),
        'max_rms_db': safe_float(np.max(rms_db)),
        'min_rms': safe_float(np.min(rms[rms > 0]) if np.any(rms > 0) else 0),
        'min_rms_db': safe_float(np.min(rms_db[rms > 0]) if np.any(rms > 0) else -100),
        'rms_dynamic_range_db': safe_float(rms_dynamic_range),
        'lra_approx_db': safe_float(lra_approx),
        'rms_volatility': safe_float(rms_volatility),
        'peak_time_sec': safe_float(times[peak_idx]),
        'percentiles': {
            'p1': safe_float(pcts[0]), 'p5': safe_float(pcts[1]), 'p10': safe_float(pcts[2]),
            'p25': safe_float(pcts[3]), 'p50': safe_float(pcts[4]), 'p75': safe_float(pcts[5]),
            'p90': safe_float(pcts[6]), 'p95': safe_float(pcts[7]), 'p99': safe_float(pcts[8]),
        },
    }


def compute_band_rms(y: np.ndarray, sr: int, frame_length: int = 2048, hop_length: int = 512):
    """Compute RMS energy in frequency bands."""
    S = np.abs(librosa.stft(y, n_fft=frame_length, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=frame_length)
    times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop_length)

    bands = [
        ('Sub-bass', 0, 60),
        ('Bass', 60, 250),
        ('Low-mid', 250, 500),
        ('Mid', 500, 2000),
        ('Upper-mid', 2000, 4000),
        ('High', 4000, 8000),
        ('Very High', 8000, min(20000, sr // 2)),
    ]

    band_results = []
    band_rms_over_time = {}
    total_energy = np.sum(S ** 2)

    for name, f_low, f_high in bands:
        if f_low >= sr // 2:
            continue
        f_high = min(f_high, sr // 2)
        mask = (freqs >= f_low) & (freqs < f_high)
        if not np.any(mask):
            continue
        band_power = S[mask, :] ** 2
        band_rms_t = np.sqrt(np.mean(band_power, axis=0))
        band_energy = np.sum(band_power)

        band_results.append({
            'band': name,
            'range_hz': f'{f_low}-{f_high}',
            'mean_rms': safe_float(np.mean(band_rms_t)),
            'mean_rms_db': safe_float(20 * np.log10(np.mean(band_rms_t) + 1e-10)),
            'max_rms': safe_float(np.max(band_rms_t)),
            'energy_percent': safe_float(band_energy / (total_energy + 1e-10) * 100),
        })
        band_rms_over_time[name] = band_rms_t

    return band_results, band_rms_over_time, times


def compute_energy_segments(rms: np.ndarray, rms_db: np.ndarray, times: np.ndarray, n_segments: int = 10):
    """Divide RMS into segments and compute per-segment statistics."""
    n = len(rms)
    seg_len = max(1, n // n_segments)
    segments = []

    for i in range(n_segments):
        start = i * seg_len
        end = min(start + seg_len, n)
        if start >= n:
            break
        seg_rms = rms[start:end]
        seg_db = rms_db[start:end]

        segments.append({
            'segment': i + 1,
            'start_sec': safe_float(times[start]),
            'end_sec': safe_float(times[min(end - 1, n - 1)]),
            'mean_rms': safe_float(np.mean(seg_rms)),
            'mean_rms_db': safe_float(np.mean(seg_db)),
            'max_rms_db': safe_float(np.max(seg_db)),
            'min_rms_db': safe_float(np.min(seg_db)),
            'std_rms_db': safe_float(np.std(seg_db)),
            'range_db': safe_float(np.max(seg_db) - np.min(seg_db)),
        })

    return segments


def classify_energy_frames(rms_db: np.ndarray, silence_threshold_db: float = -40, speech_threshold_db: float = -25):
    """Classify each frame as silence / low / medium / high energy."""
    n = len(rms_db)
    classes = []
    counts = {'silence': 0, 'low': 0, 'medium': 0, 'high': 0}

    high_threshold = speech_threshold_db + 10

    for val in rms_db:
        if val < silence_threshold_db:
            classes.append('silence')
            counts['silence'] += 1
        elif val < speech_threshold_db:
            classes.append('low')
            counts['low'] += 1
        elif val < high_threshold:
            classes.append('medium')
            counts['medium'] += 1
        else:
            classes.append('high')
            counts['high'] += 1

    total = n if n > 0 else 1
    distribution = {
        'silence_percent': safe_float(counts['silence'] / total * 100),
        'low_percent': safe_float(counts['low'] / total * 100),
        'medium_percent': safe_float(counts['medium'] / total * 100),
        'high_percent': safe_float(counts['high'] / total * 100),
        'silence_frames': counts['silence'],
        'low_frames': counts['low'],
        'medium_frames': counts['medium'],
        'high_frames': counts['high'],
        'silence_threshold_db': safe_float(silence_threshold_db),
        'speech_threshold_db': safe_float(speech_threshold_db),
    }
    return classes, distribution


def detect_fades(rms_db: np.ndarray, times: np.ndarray, window: int = 20, slope_threshold: float = 0.5):
    """Detect fade-in and fade-out regions."""
    fades = {'fade_in': None, 'fade_out': None}

    if len(rms_db) < window * 2:
        return fades

    # Fade-in: check start region
    start_slope = (rms_db[window] - rms_db[0]) / (times[window] - times[0] + 1e-10)
    if start_slope > slope_threshold:
        fades['fade_in'] = {
            'start_sec': safe_float(times[0]),
            'end_sec': safe_float(times[window]),
            'slope_db_per_sec': safe_float(start_slope),
        }

    # Fade-out: check end region
    end_slope = (rms_db[-1] - rms_db[-window]) / (times[-1] - times[-window] + 1e-10)
    if end_slope < -slope_threshold:
        fades['fade_out'] = {
            'start_sec': safe_float(times[-window]),
            'end_sec': safe_float(times[-1]),
            'slope_db_per_sec': safe_float(end_slope),
        }

    return fades


def compute_lufs_approximation(y: np.ndarray, sr: int):
    """Approximate integrated LUFS using K-weighted energy."""
    # Simple K-weighting approximation (high-shelf + high-pass)
    # Pre-filter (high shelf +4dB at 1500Hz)
    # This is a rough approximation of ITU-R BS.1770
    try:
        # Stage 1: High shelf
        f0 = 1500.0
        G = 4.0  # dB
        Q = 0.7
        A = 10 ** (G / 40.0)
        w0 = 2 * np.pi * f0 / sr
        alpha = np.sin(w0) / (2 * Q)

        b0 = A * ((A + 1) + (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * np.cos(w0))
        b2 = A * ((A + 1) + (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * np.cos(w0) + 2 * np.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * np.cos(w0))
        a2 = (A + 1) - (A - 1) * np.cos(w0) - 2 * np.sqrt(A) * alpha

        y_k = lfilter([b0/a0, b1/a0, b2/a0], [1, a1/a0, a2/a0], y)

        # Stage 2: High-pass at 38 Hz
        f_hp = 38.0
        w_hp = 2 * np.pi * f_hp / sr
        alpha_hp = np.sin(w_hp) / (2 * 0.5)
        bh0 = (1 + np.cos(w_hp)) / 2
        bh1 = -(1 + np.cos(w_hp))
        bh2 = (1 + np.cos(w_hp)) / 2
        ah0 = 1 + alpha_hp
        ah1 = -2 * np.cos(w_hp)
        ah2 = 1 - alpha_hp

        y_k = lfilter([bh0/ah0, bh1/ah0, bh2/ah0], [1, ah1/ah0, ah2/ah0], y_k)

        # Integrated loudness
        mean_square = np.mean(y_k ** 2)
        lufs = -0.691 + 10 * np.log10(mean_square + 1e-10)
        return safe_float(lufs)
    except:
        return safe_float(20 * np.log10(np.sqrt(np.mean(y ** 2)) + 1e-10))


def generate_interpretation(stats, distribution, fades, band_results, segments):
    """Generate human-readable interpretation."""
    lines = []

    # Overall level
    rms_db = stats['overall_rms_db']
    if rms_db > -10:
        lines.append(f"Signal is loud (RMS: {rms_db:.1f} dBFS) — close to full scale, possibly compressed or limited.")
    elif rms_db > -20:
        lines.append(f"Moderate signal level (RMS: {rms_db:.1f} dBFS) — typical for well-recorded audio.")
    elif rms_db > -35:
        lines.append(f"Quiet signal (RMS: {rms_db:.1f} dBFS) — low recording level, may benefit from normalization.")
    else:
        lines.append(f"Very quiet signal (RMS: {rms_db:.1f} dBFS) — likely contains significant silence or very low-level content.")

    # Dynamic range
    dr = stats['rms_dynamic_range_db']
    if dr > 30:
        lines.append(f"Wide RMS dynamic range ({dr:.1f} dB) — significant variation between quiet and loud sections.")
    elif dr > 15:
        lines.append(f"Moderate RMS dynamic range ({dr:.1f} dB) — normal energy variation.")
    else:
        lines.append(f"Narrow RMS dynamic range ({dr:.1f} dB) — relatively constant energy level, possibly compressed.")

    # LRA
    lra = stats['lra_approx_db']
    if lra > 15:
        lines.append(f"High loudness range (~{lra:.1f} dB LRA) — wide variation in short-term loudness.")
    elif lra > 7:
        lines.append(f"Normal loudness range (~{lra:.1f} dB LRA).")
    else:
        lines.append(f"Low loudness range (~{lra:.1f} dB LRA) — very consistent loudness.")

    # Energy distribution
    sil = distribution['silence_percent']
    high = distribution['high_percent']
    if sil > 40:
        lines.append(f"Signal is {sil:.0f}% silence — significant pauses or inactive regions.")
    if high > 50:
        lines.append(f"Signal is {high:.0f}% high-energy — dense, continuously active content.")

    # Volatility
    vol = stats['rms_volatility']
    if vol > 3:
        lines.append(f"High energy volatility ({vol:.2f} dB/frame) — rapid loudness fluctuations.")
    elif vol > 1:
        lines.append(f"Moderate energy volatility ({vol:.2f} dB/frame).")
    else:
        lines.append(f"Low energy volatility ({vol:.2f} dB/frame) — smooth energy contour.")

    # Fades
    if fades.get('fade_in'):
        lines.append(f"Fade-in detected in first {fades['fade_in']['end_sec']:.2f}s.")
    if fades.get('fade_out'):
        lines.append(f"Fade-out detected starting at {fades['fade_out']['start_sec']:.2f}s.")

    # Dominant band
    if band_results:
        dom = max(band_results, key=lambda b: b['energy_percent'])
        lines.append(f"Dominant energy band: {dom['band']} ({dom['energy_percent']:.1f}% of total energy).")

    return {
        'summary': ' '.join(lines[:2]),
        'details': lines,
    }


def build_rms_table(rms, rms_db, times, n_points=400):
    """Build sampled RMS data for frontend."""
    n = len(times)
    step = max(1, n // n_points)
    table = []
    for i in range(0, n, step):
        table.append({
            'time_sec': safe_float(times[i]),
            'rms': safe_float(rms[i]),
            'rms_db': safe_float(rms_db[i]),
        })
    return table


# ──────────────────────────────────────────────
# Plot Generation
# ──────────────────────────────────────────────

def generate_rms_plots(
    y: np.ndarray, sr: int,
    rms: np.ndarray, rms_db: np.ndarray, times: np.ndarray,
    stats: dict, distribution: dict, segments: list,
    band_results: list, band_rms_over_time: dict, band_times: np.ndarray,
    fades: dict, frame_classes: list,
) -> str:
    """Generate comprehensive RMS energy visualization."""

    fig = plt.figure(figsize=(18, 30))
    fig.suptitle('RMS Energy Analysis', fontsize=18, fontweight='bold')
    gs = gridspec.GridSpec(7, 2, figure=fig, hspace=0.55, wspace=0.3)

    # ── 1. Waveform + RMS Overlay ──
    ax1 = fig.add_subplot(gs[0, :])
    t_wav = np.arange(len(y)) / sr
    ax1.plot(t_wav, y, color='#90CAF9', linewidth=0.2, alpha=0.5)
    ax1.plot(times, rms, color='#F44336', linewidth=1.5, label='RMS')
    ax1.plot(times, -rms, color='#F44336', linewidth=1.5)
    ax1.axhline(y=0, color='gray', linewidth=0.3)
    ax1.set_title('Waveform with RMS Envelope', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, linestyle='--', alpha=0.4)

    # ── 2. RMS Energy (dB) ──
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(times, rms_db, color='#FF5722', linewidth=0.8)
    ax2.fill_between(times, rms_db, np.min(rms_db), alpha=0.15, color='#FF5722')
    ax2.axhline(y=stats['mean_rms_db'], color='#F44336', linestyle='--', linewidth=0.8, alpha=0.7,
                label=f'Mean: {stats["mean_rms_db"]:.1f} dB')
    ax2.axhline(y=stats['percentiles']['p95'], color='#4CAF50', linestyle=':', linewidth=0.8, alpha=0.7,
                label=f'P95: {stats["percentiles"]["p95"]:.1f} dB')
    ax2.axhline(y=stats['percentiles']['p5'], color='#2196F3', linestyle=':', linewidth=0.8, alpha=0.7,
                label=f'P5: {stats["percentiles"]["p5"]:.1f} dB')
    if fades.get('fade_in'):
        ax2.axvspan(fades['fade_in']['start_sec'], fades['fade_in']['end_sec'], alpha=0.15, color='#4CAF50', label='Fade-in')
    if fades.get('fade_out'):
        ax2.axvspan(fades['fade_out']['start_sec'], fades['fade_out']['end_sec'], alpha=0.15, color='#9C27B0', label='Fade-out')
    ax2.set_title('RMS Energy (dB)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('RMS (dBFS)')
    ax2.legend(loc='lower right', fontsize=7)
    ax2.grid(True, linestyle='--', alpha=0.4)

    # ── 3. RMS Energy (Linear) ──
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(times, rms, color='#4CAF50', linewidth=0.8)
    ax3.fill_between(times, rms, alpha=0.2, color='#4CAF50')
    ax3.set_title('RMS Energy (Linear)', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('RMS')
    ax3.grid(True, linestyle='--', alpha=0.4)

    # ── 4. Energy Classification ──
    ax4 = fig.add_subplot(gs[2, 0])
    class_colors = {'silence': '#BDBDBD', 'low': '#90CAF9', 'medium': '#FF9800', 'high': '#F44336'}
    for cls in ['silence', 'low', 'medium', 'high']:
        mask = np.array([1 if c == cls else 0 for c in frame_classes])
        if np.any(mask):
            ax4.fill_between(times, 0, mask * np.max(rms), alpha=0.4, color=class_colors[cls], label=cls.capitalize())
    ax4.set_title('Energy Classification', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Activity')
    ax4.legend(loc='upper right', fontsize=7)
    ax4.grid(True, linestyle='--', alpha=0.4)

    # ── 5. Energy Distribution Pie ──
    ax5 = fig.add_subplot(gs[2, 1])
    dist_labels = ['Silence', 'Low', 'Medium', 'High']
    dist_vals = [distribution['silence_percent'], distribution['low_percent'],
                 distribution['medium_percent'], distribution['high_percent']]
    dist_colors = ['#BDBDBD', '#90CAF9', '#FF9800', '#F44336']
    non_zero = [(l, v, c) for l, v, c in zip(dist_labels, dist_vals, dist_colors) if v > 0.5]
    if non_zero:
        labels_nz, vals_nz, colors_nz = zip(*non_zero)
        ax5.pie(vals_nz, labels=labels_nz, autopct='%1.1f%%', colors=colors_nz,
                startangle=90, textprops={'fontsize': 10})
    ax5.set_title('Energy Level Distribution', fontsize=13, fontweight='bold')

    # ── 6. RMS Histogram ──
    ax6 = fig.add_subplot(gs[3, 0])
    valid_db = rms_db[rms_db > -100]
    ax6.hist(valid_db, bins=60, color='#00BCD4', edgecolor='white', linewidth=0.3, alpha=0.85)
    ax6.axvline(x=stats['mean_rms_db'], color='#F44336', linestyle='--', linewidth=1, label=f'Mean: {stats["mean_rms_db"]:.1f} dB')
    ax6.axvline(x=stats['median_rms_db'], color='#4CAF50', linestyle='--', linewidth=1, label=f'Median: {stats["median_rms_db"]:.1f} dB')
    ax6.set_title('RMS Distribution (dB)', fontsize=13, fontweight='bold')
    ax6.set_xlabel('RMS (dBFS)')
    ax6.set_ylabel('Frame Count')
    ax6.legend(fontsize=8)
    ax6.grid(True, linestyle='--', alpha=0.4, axis='y')

    # ── 7. Segment-wise RMS ──
    ax7 = fig.add_subplot(gs[3, 1])
    seg_labels = [f'S{s["segment"]}' for s in segments]
    seg_means = [s['mean_rms_db'] for s in segments]
    seg_ranges = [s['range_db'] for s in segments]
    x_seg = np.arange(len(segments))
    colors_seg = ['#F44336' if v == max(seg_means) else '#4CAF50' if v == min(seg_means) else '#2196F3'
                  for v in seg_means]
    ax7.bar(x_seg, seg_means, color=colors_seg, edgecolor='white', linewidth=0.8)
    ax7.errorbar(x_seg, seg_means, yerr=[s['std_rms_db'] for s in segments],
                 fmt='none', ecolor='black', capsize=3, linewidth=0.8)
    ax7.set_title('Segment-wise Mean RMS (dB)', fontsize=13, fontweight='bold')
    ax7.set_xlabel('Segment')
    ax7.set_ylabel('Mean RMS (dBFS)')
    ax7.set_xticks(x_seg)
    ax7.set_xticklabels(seg_labels, fontsize=8)
    ax7.grid(True, linestyle='--', alpha=0.4, axis='y')

    # ── 8. Band Energy Distribution ──
    ax8 = fig.add_subplot(gs[4, 0])
    if band_results:
        b_names = [b['band'] for b in band_results]
        b_pcts = [b['energy_percent'] for b in band_results]
        b_colors = ['#1a237e', '#283593', '#3949ab', '#5c6bc0', '#7986cb', '#9fa8da', '#c5cae9'][:len(b_names)]
        bars8 = ax8.bar(b_names, b_pcts, color=b_colors, edgecolor='white', linewidth=0.8)
        for bar, pct in zip(bars8, b_pcts):
            if pct > 3:
                ax8.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                        f'{pct:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax8.set_title('Band Energy Distribution', fontsize=13, fontweight='bold')
    ax8.set_ylabel('Energy (%)')
    ax8.tick_params(axis='x', rotation=25)
    ax8.grid(True, linestyle='--', alpha=0.4, axis='y')

    # ── 9. Band RMS over Time ──
    ax9 = fig.add_subplot(gs[4, 1])
    band_colors_t = ['#F44336', '#FF9800', '#4CAF50', '#2196F3', '#9C27B0', '#607D8B', '#795548']
    for idx, (name, rms_t) in enumerate(list(band_rms_over_time.items())[:5]):
        rms_t_db = 20 * np.log10(rms_t + 1e-10)
        color = band_colors_t[idx % len(band_colors_t)]
        ax9.plot(band_times[:len(rms_t_db)], rms_t_db, linewidth=0.8, alpha=0.7, label=name, color=color)
    ax9.set_title('Band RMS over Time (top 5)', fontsize=13, fontweight='bold')
    ax9.set_xlabel('Time (s)')
    ax9.set_ylabel('RMS (dB)')
    ax9.legend(loc='lower right', fontsize=7)
    ax9.grid(True, linestyle='--', alpha=0.4)

    # ── 10. Percentile Profile ──
    ax10 = fig.add_subplot(gs[5, 0])
    pct_labels = ['P1', 'P5', 'P10', 'P25', 'P50', 'P75', 'P90', 'P95', 'P99']
    pct_vals = [stats['percentiles'][k] for k in ['p1', 'p5', 'p10', 'p25', 'p50', 'p75', 'p90', 'p95', 'p99']]
    ax10.barh(pct_labels, pct_vals, color='#3F51B5', edgecolor='white', linewidth=0.8)
    ax10.set_title('RMS Percentile Profile (dB)', fontsize=13, fontweight='bold')
    ax10.set_xlabel('RMS (dBFS)')
    ax10.grid(True, linestyle='--', alpha=0.4, axis='x')

    # ── 11. Key Metrics Summary ──
    ax11 = fig.add_subplot(gs[5, 1])
    met_labels = ['Overall\nRMS (dB)', 'Mean\nRMS (dB)', 'Dynamic\nRange (dB)', 'LRA\n(dB)', 'Volatility\n(dB/frame)']
    met_vals = [stats['overall_rms_db'], stats['mean_rms_db'], stats['rms_dynamic_range_db'],
                stats['lra_approx_db'], stats['rms_volatility']]
    met_colors = ['#F44336', '#FF5722', '#4CAF50', '#2196F3', '#9C27B0']
    bars11 = ax11.bar(met_labels, met_vals, color=met_colors, edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars11, met_vals):
        ax11.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax11.set_title('Key Metrics', fontsize=13, fontweight='bold')
    ax11.set_ylabel('Value')
    ax11.grid(True, linestyle='--', alpha=0.4, axis='y')

    # ── 12. Rolling RMS Comparison (short vs long window) ──
    ax12 = fig.add_subplot(gs[6, :])
    # Short window RMS
    short_rms = librosa.feature.rms(y=y, frame_length=512, hop_length=128)[0]
    short_times = librosa.frames_to_time(np.arange(len(short_rms)), sr=sr, hop_length=128)
    short_db = 20 * np.log10(short_rms + 1e-10)
    # Long window RMS
    long_rms = librosa.feature.rms(y=y, frame_length=8192, hop_length=2048)[0]
    long_times = librosa.frames_to_time(np.arange(len(long_rms)), sr=sr, hop_length=2048)
    long_db = 20 * np.log10(long_rms + 1e-10)

    ax12.plot(short_times, short_db, color='#FF9800', linewidth=0.4, alpha=0.5, label='Short-term (~12ms)')
    ax12.plot(long_times, long_db, color='#2196F3', linewidth=1.5, label='Long-term (~186ms)')
    ax12.plot(times, rms_db, color='#F44336', linewidth=0.8, alpha=0.7, label=f'Analysis ({len(rms)} frames)')
    ax12.set_title('Multi-Resolution RMS Comparison', fontsize=13, fontweight='bold')
    ax12.set_xlabel('Time (s)')
    ax12.set_ylabel('RMS (dBFS)')
    ax12.legend(loc='lower right', fontsize=8)
    ax12.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"


# ──────────────────────────────────────────────
# Main Endpoint
# ──────────────────────────────────────────────

@router.post("/rms-energy-analysis")
async def rms_energy_analysis(
    file: UploadFile = File(...),
    frame_length: Optional[int] = Form(2048),
    hop_length: Optional[int] = Form(512),
    n_segments: Optional[int] = Form(10),
    silence_threshold_db: Optional[float] = Form(-40),
    speech_threshold_db: Optional[float] = Form(-25),
    target_sr: Optional[int] = Form(None),
):
    """
    RMS Energy Analysis.

    Comprehensive RMS energy profiling including frame-wise and segment-wise
    statistics, band energy, LUFS approximation, energy classification,
    fade detection, and multi-resolution comparison.

    Parameters:
    - file: Audio file (WAV, MP3, FLAC, etc.)
    - frame_length: Frame length for RMS computation (default: 2048)
    - hop_length: Hop length (default: 512)
    - n_segments: Temporal segments for segment analysis (default: 10)
    - silence_threshold_db: dB threshold for silence classification (default: -40)
    - speech_threshold_db: dB threshold for speech/active classification (default: -25)
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
            sil_db = silence_threshold_db if silence_threshold_db is not None else -40
            sp_db = speech_threshold_db if speech_threshold_db is not None else -25

            # RMS profile
            rms, rms_db, times = compute_rms_profile(y, sr, frame_length=fl, hop_length=hl)

            # Statistics
            stats = compute_rms_statistics(rms, rms_db, times, y, sr)

            # Band RMS
            band_results, band_rms_over_time, band_times = compute_band_rms(y, sr, frame_length=fl, hop_length=hl)

            # Segments
            segments = compute_energy_segments(rms, rms_db, times, n_segments=n_seg)

            # Classification
            frame_classes, distribution = classify_energy_frames(rms_db, silence_threshold_db=sil_db, speech_threshold_db=sp_db)

            # Fade detection
            fades = detect_fades(rms_db, times)

            # LUFS approximation
            lufs = compute_lufs_approximation(y, sr)

            # Interpretation
            interpretation = generate_interpretation(stats, distribution, fades, band_results, segments)

            # RMS table
            rms_table = build_rms_table(rms, rms_db, times, n_points=400)

            # Plots
            plot = generate_rms_plots(
                y, sr, rms, rms_db, times, stats, distribution, segments,
                band_results, band_rms_over_time, band_times, fades, frame_classes
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
                'n_frames': len(rms),
                'time_resolution_ms': safe_float(hl / sr * 1000),
            }

            return _to_native({
                'results': {
                    'file_info': file_info,
                    'rms_statistics': stats,
                    'lufs_integrated': lufs,
                    'energy_distribution': distribution,
                    'band_energy': band_results,
                    'segments': segments,
                    'fades': fades,
                    'rms_table': rms_table,
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
# from rms_energy_analysis import router as rms_energy_router
# app.include_router(rms_energy_router, prefix="/api/analysis", tags=["Audio Analysis"])
# → POST /api/analysis/rms-energy-analysis
# ──────────────────────────────────────────────
