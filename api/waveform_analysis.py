"""
Waveform Analysis Backend (FastAPI)
- Comprehensive time-domain analysis of audio signals
- Amplitude statistics, envelope extraction, crest factor
- Zero-crossing rate, RMS energy profile, dynamic range
- Silence/activity detection, clipping detection
- Temporal segmentation and transient analysis
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
# Waveform Analysis Functions
# ──────────────────────────────────────────────

def compute_amplitude_stats(y: np.ndarray, sr: int):
    """Compute comprehensive amplitude statistics."""
    abs_y = np.abs(y)
    rms_total = np.sqrt(np.mean(y ** 2))
    peak = np.max(abs_y)

    # Crest factor: ratio of peak to RMS (in dB)
    crest_factor_linear = peak / (rms_total + 1e-10)
    crest_factor_db = 20 * np.log10(crest_factor_linear + 1e-10)

    # Dynamic range
    # Signal above noise floor
    sorted_abs = np.sort(abs_y)
    noise_floor = sorted_abs[int(len(sorted_abs) * 0.05)]  # 5th percentile
    dynamic_range_db = 20 * np.log10((peak + 1e-10) / (noise_floor + 1e-10))

    # Peak-to-average ratio
    par = peak / (np.mean(abs_y) + 1e-10)

    # Clipping detection
    clip_threshold = 0.99
    n_clipped = int(np.sum(abs_y >= clip_threshold))
    clip_percent = n_clipped / len(y) * 100

    return {
        'peak_amplitude': safe_float(peak),
        'peak_db': safe_float(20 * np.log10(peak + 1e-10)),
        'rms_amplitude': safe_float(rms_total),
        'rms_db': safe_float(20 * np.log10(rms_total + 1e-10)),
        'mean_amplitude': safe_float(np.mean(abs_y)),
        'std_amplitude': safe_float(np.std(y)),
        'min_amplitude': safe_float(np.min(y)),
        'max_amplitude': safe_float(np.max(y)),
        'crest_factor_linear': safe_float(crest_factor_linear),
        'crest_factor_db': safe_float(crest_factor_db),
        'dynamic_range_db': safe_float(dynamic_range_db),
        'peak_to_average_ratio': safe_float(par),
        'noise_floor_amplitude': safe_float(noise_floor),
        'noise_floor_db': safe_float(20 * np.log10(noise_floor + 1e-10)),
        'n_clipped_samples': n_clipped,
        'clip_percent': safe_float(clip_percent),
        'is_clipping': n_clipped > 10,
        'dc_offset': safe_float(np.mean(y)),
    }


def compute_envelope(y: np.ndarray, sr: int, frame_length: int = 2048, hop_length: int = 512):
    """Compute signal envelope using RMS and peak tracking."""
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    # Peak envelope via max-pooling
    n_frames = len(rms)
    peak_env = np.zeros(n_frames)
    for i in range(n_frames):
        start = i * hop_length
        end = min(start + frame_length, len(y))
        peak_env[i] = np.max(np.abs(y[start:end]))

    # RMS in dB
    rms_db = 20 * np.log10(rms + 1e-10)
    peak_db = 20 * np.log10(peak_env + 1e-10)

    return {
        'times': times,
        'rms': rms,
        'rms_db': rms_db,
        'peak_envelope': peak_env,
        'peak_db': peak_db,
    }


def compute_zcr_profile(y: np.ndarray, sr: int, frame_length: int = 2048, hop_length: int = 512):
    """Compute zero-crossing rate over time."""
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(zcr)), sr=sr, hop_length=hop_length)

    return {
        'times': times,
        'zcr': zcr,
        'mean_zcr': safe_float(np.mean(zcr)),
        'std_zcr': safe_float(np.std(zcr)),
        'max_zcr': safe_float(np.max(zcr)),
    }


def detect_silence_activity(y: np.ndarray, sr: int, top_db: int = 30, frame_length: int = 2048, hop_length: int = 512):
    """Detect silent and active regions in the signal."""
    # Use librosa's split to find non-silent intervals
    intervals = librosa.effects.split(y, top_db=top_db, frame_length=frame_length, hop_length=hop_length)

    active_regions = []
    total_active_samples = 0
    for start, end in intervals:
        duration = (end - start) / sr
        active_regions.append({
            'start_sec': safe_float(start / sr),
            'end_sec': safe_float(end / sr),
            'duration_sec': safe_float(duration),
            'rms': safe_float(np.sqrt(np.mean(y[start:end] ** 2))),
        })
        total_active_samples += (end - start)

    total_duration = len(y) / sr
    active_duration = total_active_samples / sr
    silence_duration = total_duration - active_duration

    return {
        'active_regions': active_regions[:50],  # limit for frontend
        'n_active_regions': len(intervals),
        'active_duration_sec': safe_float(active_duration),
        'silence_duration_sec': safe_float(silence_duration),
        'active_percent': safe_float(active_duration / (total_duration + 1e-10) * 100),
        'silence_percent': safe_float(silence_duration / (total_duration + 1e-10) * 100),
        'silence_threshold_db': top_db,
    }


def compute_temporal_statistics(y: np.ndarray, sr: int, n_segments: int = 10):
    """Divide signal into segments and compute per-segment stats."""
    segment_length = len(y) // n_segments
    if segment_length < 1:
        segment_length = len(y)
        n_segments = 1

    segments = []
    for i in range(n_segments):
        start = i * segment_length
        end = min(start + segment_length, len(y))
        seg = y[start:end]

        segments.append({
            'segment': i + 1,
            'start_sec': safe_float(start / sr),
            'end_sec': safe_float(end / sr),
            'rms': safe_float(np.sqrt(np.mean(seg ** 2))),
            'rms_db': safe_float(20 * np.log10(np.sqrt(np.mean(seg ** 2)) + 1e-10)),
            'peak': safe_float(np.max(np.abs(seg))),
            'peak_db': safe_float(20 * np.log10(np.max(np.abs(seg)) + 1e-10)),
            'zcr': safe_float(np.mean(np.abs(np.diff(np.sign(seg))) > 0)),
            'crest_factor': safe_float(np.max(np.abs(seg)) / (np.sqrt(np.mean(seg ** 2)) + 1e-10)),
        })

    return segments


def compute_histogram(y: np.ndarray, n_bins: int = 100):
    """Compute amplitude histogram."""
    counts, bin_edges = np.histogram(y, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return {
        'bin_centers': [safe_float(x) for x in bin_centers],
        'counts': [int(x) for x in counts],
        'n_bins': n_bins,
    }


def compute_statistical_moments(y: np.ndarray):
    """Compute higher-order statistical moments of the waveform."""
    from scipy import stats as sp_stats

    skewness = sp_stats.skew(y)
    kurtosis = sp_stats.kurtosis(y)  # excess kurtosis
    # Percentiles
    percentiles = np.percentile(y, [1, 5, 25, 50, 75, 95, 99])

    return {
        'skewness': safe_float(skewness),
        'kurtosis': safe_float(kurtosis),
        'p1': safe_float(percentiles[0]),
        'p5': safe_float(percentiles[1]),
        'p25': safe_float(percentiles[2]),
        'median': safe_float(percentiles[3]),
        'p75': safe_float(percentiles[4]),
        'p95': safe_float(percentiles[5]),
        'p99': safe_float(percentiles[6]),
        'iqr': safe_float(percentiles[4] - percentiles[2]),
    }


def generate_interpretation(amp_stats, zcr_profile, silence_activity, moments):
    """Generate human-readable interpretation of waveform analysis."""
    lines = []

    # Peak / RMS
    rms_db = amp_stats['rms_db']
    peak_db = amp_stats['peak_db']
    if rms_db > -6:
        lines.append(f"Signal is loud (RMS: {rms_db:.1f} dB, Peak: {peak_db:.1f} dB) — potential limiting or compression applied.")
    elif rms_db > -20:
        lines.append(f"Moderate signal level (RMS: {rms_db:.1f} dB, Peak: {peak_db:.1f} dB) — typical for recorded audio.")
    else:
        lines.append(f"Low signal level (RMS: {rms_db:.1f} dB, Peak: {peak_db:.1f} dB) — may need amplification.")

    # Crest factor
    cf = amp_stats['crest_factor_db']
    if cf < 6:
        lines.append(f"Low crest factor ({cf:.1f} dB) — heavily compressed or clipped signal.")
    elif cf < 15:
        lines.append(f"Normal crest factor ({cf:.1f} dB) — typical dynamic signal.")
    else:
        lines.append(f"High crest factor ({cf:.1f} dB) — signal has sharp transients or impulsive peaks.")

    # Clipping
    if amp_stats['is_clipping']:
        lines.append(f"⚠ Clipping detected: {amp_stats['n_clipped_samples']} samples ({amp_stats['clip_percent']:.2f}%) at or above ±0.99.")
    else:
        lines.append("No clipping detected — signal amplitude is within safe range.")

    # DC offset
    dc = amp_stats['dc_offset']
    if abs(dc) > 0.01:
        lines.append(f"DC offset detected: {dc:.4f}. Consider applying a high-pass filter to remove.")
    else:
        lines.append("Negligible DC offset — signal is well-centered around zero.")

    # ZCR
    mean_zcr = zcr_profile['mean_zcr']
    if mean_zcr > 0.3:
        lines.append(f"High zero-crossing rate ({mean_zcr:.3f}) — indicates noisy or high-frequency content.")
    elif mean_zcr > 0.1:
        lines.append(f"Moderate zero-crossing rate ({mean_zcr:.3f}) — typical for mixed content.")
    else:
        lines.append(f"Low zero-crossing rate ({mean_zcr:.3f}) — dominated by low-frequency content.")

    # Silence
    sp = silence_activity['silence_percent']
    if sp > 50:
        lines.append(f"Signal is {sp:.1f}% silent — contains significant gaps or pauses.")
    elif sp > 10:
        lines.append(f"Signal has {sp:.1f}% silence — normal for speech or segmented recordings.")
    else:
        lines.append(f"Signal is {100 - sp:.1f}% active — continuous or dense recording.")

    # Distribution shape
    sk = moments['skewness']
    ku = moments['kurtosis']
    if abs(sk) > 1:
        lines.append(f"Amplitude distribution is skewed ({sk:.2f}) — asymmetric waveform.")
    if ku > 3:
        lines.append(f"High kurtosis ({ku:.2f}) — heavy-tailed distribution with outlier peaks.")
    elif ku < -1:
        lines.append(f"Low kurtosis ({ku:.2f}) — flat, uniform-like amplitude distribution.")

    return {
        'summary': ' '.join(lines[:2]),
        'details': lines,
    }


def build_envelope_table(envelope, n_points=300):
    """Build sampled envelope data for frontend."""
    times = envelope['times']
    n = len(times)
    step = max(1, n // n_points)
    table = []
    for i in range(0, n, step):
        table.append({
            'time_sec': safe_float(times[i]),
            'rms': safe_float(envelope['rms'][i]),
            'rms_db': safe_float(envelope['rms_db'][i]),
            'peak': safe_float(envelope['peak_envelope'][i]),
            'peak_db': safe_float(envelope['peak_db'][i]),
        })
    return table


# ──────────────────────────────────────────────
# Plot Generation
# ──────────────────────────────────────────────

def generate_waveform_plots(
    y: np.ndarray, sr: int,
    envelope: dict, zcr_profile: dict,
    silence_activity: dict, segments: list,
    amp_stats: dict, histogram: dict, moments: dict,
) -> str:
    """Generate comprehensive waveform visualization."""

    fig = plt.figure(figsize=(18, 28))
    fig.suptitle('Waveform Analysis', fontsize=18, fontweight='bold')
    gs = gridspec.GridSpec(6, 2, figure=fig, hspace=0.55, wspace=0.3)

    time = np.arange(len(y)) / sr

    # ── 1. Full Waveform ──
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time, y, color='#2196F3', linewidth=0.3, alpha=0.7)
    # Mark active/silent regions
    for region in silence_activity['active_regions'][:20]:
        ax1.axvspan(region['start_sec'], region['end_sec'], alpha=0.08, color='#4CAF50')
    if amp_stats['is_clipping']:
        clip_mask = np.abs(y) >= 0.99
        clip_times = time[clip_mask]
        if len(clip_times) > 0:
            ax1.scatter(clip_times[:500], y[clip_mask][:500], color='#F44336', s=2, alpha=0.5, label='Clipping')
            ax1.legend(loc='upper right', fontsize=8)
    ax1.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
    ax1.set_title('Full Waveform (green = active regions)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, linestyle='--', alpha=0.4)

    # ── 2. RMS Envelope ──
    ax2 = fig.add_subplot(gs[1, 0])
    env_t = envelope['times']
    ax2.plot(env_t, envelope['rms_db'], color='#FF5722', linewidth=1.0, label='RMS (dB)')
    ax2.plot(env_t, envelope['peak_db'], color='#FF9800', linewidth=0.7, alpha=0.6, label='Peak (dB)')
    ax2.axhline(y=amp_stats['rms_db'], color='#F44336', linestyle='--', linewidth=0.8, alpha=0.5, label=f'Avg RMS: {amp_stats["rms_db"]:.1f} dB')
    ax2.set_title('Envelope (dB)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Level (dB)')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, linestyle='--', alpha=0.4)

    # ── 3. RMS Envelope (Linear) ──
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(env_t, envelope['rms'], color='#4CAF50', linewidth=1.0, label='RMS')
    ax3.fill_between(env_t, envelope['rms'], alpha=0.15, color='#4CAF50')
    ax3.plot(env_t, envelope['peak_envelope'], color='#FF9800', linewidth=0.7, alpha=0.6, label='Peak')
    ax3.set_title('Envelope (Linear)', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Amplitude')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.grid(True, linestyle='--', alpha=0.4)

    # ── 4. Zero Crossing Rate ──
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(zcr_profile['times'], zcr_profile['zcr'], color='#9C27B0', linewidth=0.8)
    ax4.fill_between(zcr_profile['times'], zcr_profile['zcr'], alpha=0.15, color='#9C27B0')
    ax4.axhline(y=zcr_profile['mean_zcr'], color='#F44336', linestyle='--', linewidth=0.8, alpha=0.5,
                label=f'Mean: {zcr_profile["mean_zcr"]:.4f}')
    ax4.set_title('Zero Crossing Rate', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('ZCR')
    ax4.legend(loc='upper right', fontsize=8)
    ax4.grid(True, linestyle='--', alpha=0.4)

    # ── 5. Amplitude Histogram ──
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.bar(histogram['bin_centers'], histogram['counts'], width=(histogram['bin_centers'][1] - histogram['bin_centers'][0]) * 0.9,
            color='#00BCD4', edgecolor='white', linewidth=0.3, alpha=0.85)
    ax5.axvline(x=0, color='gray', linewidth=0.8, alpha=0.5)
    ax5.set_title('Amplitude Distribution', fontsize=13, fontweight='bold')
    ax5.set_xlabel('Amplitude')
    ax5.set_ylabel('Count')
    ax5.grid(True, linestyle='--', alpha=0.4, axis='y')
    # Add stats annotation
    ax5.text(0.98, 0.95, f"μ={amp_stats['dc_offset']:.4f}\nσ={amp_stats['std_amplitude']:.4f}\nSkew={moments['skewness']:.2f}\nKurt={moments['kurtosis']:.2f}",
             transform=ax5.transAxes, fontsize=8, verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # ── 6. Segment-wise RMS ──
    ax6 = fig.add_subplot(gs[3, 0])
    seg_labels = [f"{s['start_sec']:.1f}s" for s in segments]
    seg_rms_db = [s['rms_db'] for s in segments]
    seg_peak_db = [s['peak_db'] for s in segments]
    x_seg = np.arange(len(segments))
    width = 0.35
    ax6.bar(x_seg - width/2, seg_rms_db, width, color='#3F51B5', label='RMS (dB)', edgecolor='white')
    ax6.bar(x_seg + width/2, seg_peak_db, width, color='#FF9800', label='Peak (dB)', edgecolor='white')
    ax6.set_title('Segment-wise Levels', fontsize=13, fontweight='bold')
    ax6.set_xlabel('Segment Start')
    ax6.set_ylabel('Level (dB)')
    ax6.set_xticks(x_seg)
    ax6.set_xticklabels(seg_labels, rotation=30, fontsize=8)
    ax6.legend(fontsize=8)
    ax6.grid(True, linestyle='--', alpha=0.4, axis='y')

    # ── 7. Segment-wise Crest Factor ──
    ax7 = fig.add_subplot(gs[3, 1])
    seg_crest = [s['crest_factor'] for s in segments]
    colors_crest = ['#F44336' if c < 2 else '#FF9800' if c < 5 else '#4CAF50' for c in seg_crest]
    ax7.bar(x_seg, seg_crest, color=colors_crest, edgecolor='white', linewidth=0.8)
    ax7.axhline(y=amp_stats['crest_factor_linear'], color='#F44336', linestyle='--', linewidth=0.8, alpha=0.5,
                label=f'Overall: {amp_stats["crest_factor_linear"]:.1f}')
    ax7.set_title('Segment-wise Crest Factor', fontsize=13, fontweight='bold')
    ax7.set_xlabel('Segment Start')
    ax7.set_ylabel('Crest Factor')
    ax7.set_xticks(x_seg)
    ax7.set_xticklabels(seg_labels, rotation=30, fontsize=8)
    ax7.legend(fontsize=8)
    ax7.grid(True, linestyle='--', alpha=0.4, axis='y')

    # ── 8. Zoomed Waveform (first 50ms) ──
    ax8 = fig.add_subplot(gs[4, 0])
    zoom_samples = min(int(sr * 0.05), len(y))
    t_zoom = np.arange(zoom_samples) / sr * 1000  # in ms
    ax8.plot(t_zoom, y[:zoom_samples], color='#2196F3', linewidth=0.8)
    ax8.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
    ax8.set_title('Zoomed Waveform (first 50ms)', fontsize=13, fontweight='bold')
    ax8.set_xlabel('Time (ms)')
    ax8.set_ylabel('Amplitude')
    ax8.grid(True, linestyle='--', alpha=0.4)

    # ── 9. Zoomed Waveform (center 50ms) ──
    ax9 = fig.add_subplot(gs[4, 1])
    center = len(y) // 2
    half_zoom = zoom_samples // 2
    start_z = max(0, center - half_zoom)
    end_z = min(len(y), center + half_zoom)
    t_zoom_c = (np.arange(end_z - start_z) + start_z) / sr * 1000
    ax9.plot(t_zoom_c, y[start_z:end_z], color='#FF5722', linewidth=0.8)
    ax9.axhline(y=0, color='gray', linewidth=0.5, alpha=0.5)
    ax9.set_title('Zoomed Waveform (center 50ms)', fontsize=13, fontweight='bold')
    ax9.set_xlabel('Time (ms)')
    ax9.set_ylabel('Amplitude')
    ax9.grid(True, linestyle='--', alpha=0.4)

    # ── 10. Activity / Silence Pie ──
    ax10 = fig.add_subplot(gs[5, 0])
    act_pct = silence_activity['active_percent']
    sil_pct = silence_activity['silence_percent']
    ax10.pie([act_pct, sil_pct], labels=['Active', 'Silent'],
             autopct='%1.1f%%', colors=['#4CAF50', '#BDBDBD'],
             startangle=90, textprops={'fontsize': 11})
    ax10.set_title('Activity vs Silence', fontsize=13, fontweight='bold')

    # ── 11. Summary Stats Bar ──
    ax11 = fig.add_subplot(gs[5, 1])
    stat_labels = ['Peak\n(dB)', 'RMS\n(dB)', 'Noise Floor\n(dB)', 'Dyn Range\n(dB)', 'Crest Factor\n(dB)']
    stat_vals = [amp_stats['peak_db'], amp_stats['rms_db'], amp_stats['noise_floor_db'],
                 amp_stats['dynamic_range_db'], amp_stats['crest_factor_db']]
    colors_stats = ['#F44336', '#FF5722', '#607D8B', '#4CAF50', '#2196F3']
    bars11 = ax11.bar(stat_labels, stat_vals, color=colors_stats, edgecolor='white', linewidth=0.8)
    ax11.set_title('Key Metrics Summary', fontsize=13, fontweight='bold')
    ax11.set_ylabel('Value (dB)')
    ax11.grid(True, linestyle='--', alpha=0.4, axis='y')
    for bar, val in zip(bars11, stat_vals):
        ax11.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"


# ──────────────────────────────────────────────
# Main Endpoint
# ──────────────────────────────────────────────

@router.post("/waveform-analysis")
async def waveform_analysis(
    file: UploadFile = File(...),
    frame_length: Optional[int] = Form(2048),
    hop_length: Optional[int] = Form(512),
    n_segments: Optional[int] = Form(10),
    silence_threshold_db: Optional[int] = Form(30),
    target_sr: Optional[int] = Form(None),
):
    """
    Waveform Analysis.

    Comprehensive time-domain analysis of audio signals including
    amplitude statistics, envelope extraction, clipping detection,
    silence/activity detection, and temporal segmentation.

    Parameters:
    - file: Audio file (WAV, MP3, FLAC, etc.)
    - frame_length: Frame length for envelope computation (default: 2048)
    - hop_length: Hop length for envelope computation (default: 512)
    - n_segments: Number of temporal segments for segment analysis (default: 10)
    - silence_threshold_db: Threshold in dB for silence detection (default: 30)
    - target_sr: Resample to this rate (default: keep original)

    Response:
    {
        "results": {
            "file_info": {...},
            "amplitude_stats": {...},
            "zcr_profile": {...},
            "silence_activity": {...},
            "statistical_moments": {...},
            "segments": [...],
            "envelope_table": [...],
            "interpretation": {...}
        },
        "plot": "data:image/png;base64,..."
    }
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
            # Load audio
            sr_param = target_sr if target_sr else None
            y, sr = librosa.load(tmp_path, sr=sr_param, mono=True)

            # Limit to 60s
            max_samples = sr * 60
            if len(y) > max_samples:
                y = y[:max_samples]

            frame_len = frame_length if frame_length else 2048
            hop_val = hop_length if hop_length else 512
            n_seg = n_segments if n_segments else 10
            sil_db = silence_threshold_db if silence_threshold_db else 30

            # Amplitude statistics
            amp_stats = compute_amplitude_stats(y, sr)

            # Envelope
            envelope = compute_envelope(y, sr, frame_length=frame_len, hop_length=hop_val)

            # ZCR profile
            zcr_profile = compute_zcr_profile(y, sr, frame_length=frame_len, hop_length=hop_val)

            # Silence / Activity detection
            silence_activity = detect_silence_activity(y, sr, top_db=sil_db, frame_length=frame_len, hop_length=hop_val)

            # Temporal segments
            segments = compute_temporal_statistics(y, sr, n_segments=n_seg)

            # Histogram
            histogram = compute_histogram(y, n_bins=100)

            # Statistical moments
            moments = compute_statistical_moments(y)

            # Interpretation
            interpretation = generate_interpretation(amp_stats, zcr_profile, silence_activity, moments)

            # Envelope table (sampled)
            envelope_table = build_envelope_table(envelope, n_points=300)

            # Generate plots
            plot = generate_waveform_plots(
                y, sr, envelope, zcr_profile, silence_activity,
                segments, amp_stats, histogram, moments
            )

            # File info
            file_info = {
                'filename': file.filename,
                'file_size_mb': round(len(content) / (1024 * 1024), 2),
                'sample_rate': sr,
                'duration_sec': round(len(y) / sr, 2),
                'n_samples': len(y),
                'format': file_ext.replace('.', ''),
                'frame_length': frame_len,
                'hop_length': hop_val,
                'n_segments': n_seg,
                'silence_threshold_db': sil_db,
                'bit_depth_estimate': '16-bit' if np.max(np.abs(y)) <= 1.0 else '32-bit float',
            }

            return _to_native({
                'results': {
                    'file_info': file_info,
                    'amplitude_stats': amp_stats,
                    'zcr_profile': {
                        'mean_zcr': zcr_profile['mean_zcr'],
                        'std_zcr': zcr_profile['std_zcr'],
                        'max_zcr': zcr_profile['max_zcr'],
                    },
                    'silence_activity': silence_activity,
                    'statistical_moments': moments,
                    'segments': segments,
                    'envelope_table': envelope_table,
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
# from waveform_analysis import router as waveform_router
# app.include_router(waveform_router, prefix="/api/analysis", tags=["Audio Analysis"])
# → POST /api/analysis/waveform-analysis
# ──────────────────────────────────────────────
