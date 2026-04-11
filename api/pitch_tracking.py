"""
Pitch Tracking Analysis Backend (FastAPI)
- Comprehensive fundamental frequency (F0) tracking
- Multiple pitch estimation: pYIN, YIN autocorrelation
- Voiced/unvoiced segmentation with voicing probability
- Pitch statistics: mean, median, range, std, jitter, shimmer-approx
- Pitch contour smoothing and interpolation
- Musical note mapping (Hz → MIDI → note name + cents deviation)
- Pitch histogram and distribution analysis
- Segment-wise pitch analysis and modulation detection
- Vibrato detection (rate and extent estimation)
- Pitch stability and confidence metrics
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional
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

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


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


def hz_to_midi(hz):
    if hz <= 0: return 0
    return 69 + 12 * np.log2(hz / 440.0)

def midi_to_note(midi_num):
    if midi_num <= 0: return 'N/A'
    note_idx = int(round(midi_num)) % 12
    octave = int(round(midi_num)) // 12 - 1
    return f'{NOTE_NAMES[note_idx]}{octave}'

def hz_to_cents_deviation(hz):
    """Cents deviation from nearest MIDI note."""
    if hz <= 0: return 0
    midi = hz_to_midi(hz)
    nearest = round(midi)
    return (midi - nearest) * 100


# ──────────────────────────────────────────────
# Pitch Tracking Functions
# ──────────────────────────────────────────────

def track_pitch_pyin(y, sr, fmin=65, fmax=2093, hop_length=512):
    """Track pitch using pYIN."""
    f0, voiced_flag, voiced_prob = librosa.pyin(
        y, fmin=fmin, fmax=fmax, sr=sr, hop_length=hop_length
    )
    times = librosa.frames_to_time(np.arange(len(f0)), sr=sr, hop_length=hop_length)
    return f0, voiced_flag, voiced_prob, times


def compute_pitch_statistics(f0, voiced_flag):
    """Comprehensive pitch statistics on voiced frames."""
    voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
    if len(voiced_f0) == 0:
        return {k: 0 for k in ['mean_hz','median_hz','std_hz','min_hz','max_hz','range_hz',
                                'mean_midi','mean_note','range_semitones','voiced_percent',
                                'n_voiced','n_total','cv','skewness','kurtosis',
                                'percentiles']}

    midi_vals = [hz_to_midi(h) for h in voiced_f0]
    mean_midi = np.mean(midi_vals)
    pcts = np.percentile(voiced_f0, [5, 10, 25, 50, 75, 90, 95])

    return {
        'mean_hz': safe_float(np.mean(voiced_f0)),
        'median_hz': safe_float(np.median(voiced_f0)),
        'std_hz': safe_float(np.std(voiced_f0)),
        'min_hz': safe_float(np.min(voiced_f0)),
        'max_hz': safe_float(np.max(voiced_f0)),
        'range_hz': safe_float(np.max(voiced_f0) - np.min(voiced_f0)),
        'mean_midi': safe_float(mean_midi),
        'mean_note': midi_to_note(mean_midi),
        'range_semitones': safe_float(hz_to_midi(np.max(voiced_f0)) - hz_to_midi(np.min(voiced_f0))),
        'voiced_percent': safe_float(len(voiced_f0) / max(len(f0), 1) * 100),
        'n_voiced': int(len(voiced_f0)),
        'n_total': int(len(f0)),
        'cv': safe_float(np.std(voiced_f0) / (np.mean(voiced_f0) + 1e-10)),
        'skewness': safe_float(float(pd.Series(voiced_f0).skew())),
        'kurtosis': safe_float(float(pd.Series(voiced_f0).kurtosis())),
        'percentiles': {
            'p5': safe_float(pcts[0]), 'p10': safe_float(pcts[1]), 'p25': safe_float(pcts[2]),
            'p50': safe_float(pcts[3]), 'p75': safe_float(pcts[4]), 'p90': safe_float(pcts[5]),
            'p95': safe_float(pcts[6]),
        },
    }


def compute_jitter_shimmer(f0, voiced_flag, y, sr, hop_length):
    """Approximate jitter (pitch perturbation) and shimmer (amplitude perturbation)."""
    voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
    if len(voiced_f0) < 3:
        return {'jitter_percent': 0, 'jitter_abs_hz': 0, 'shimmer_approx_db': 0}

    # Jitter: mean absolute difference between consecutive periods
    period_diffs = np.abs(np.diff(voiced_f0))
    jitter_abs = np.mean(period_diffs)
    jitter_pct = jitter_abs / (np.mean(voiced_f0) + 1e-10) * 100

    # Shimmer approximation: amplitude variation at voiced frames
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
    voiced_rms = rms[:len(voiced_flag)][voiced_flag[:len(rms)]]
    if len(voiced_rms) > 2:
        rms_db = 20 * np.log10(voiced_rms + 1e-10)
        shimmer_db = np.mean(np.abs(np.diff(rms_db)))
    else:
        shimmer_db = 0

    return {
        'jitter_percent': safe_float(jitter_pct),
        'jitter_abs_hz': safe_float(jitter_abs),
        'shimmer_approx_db': safe_float(shimmer_db),
    }


def detect_vibrato(f0, voiced_flag, sr, hop_length, min_rate=4, max_rate=8):
    """Detect vibrato from pitch contour."""
    voiced_f0 = f0.copy()
    voiced_f0[~voiced_flag | np.isnan(voiced_f0)] = 0

    # Get longest voiced segment
    segments = []
    start = None
    for i in range(len(voiced_f0)):
        if voiced_f0[i] > 0:
            if start is None: start = i
        else:
            if start is not None:
                segments.append((start, i))
                start = None
    if start is not None:
        segments.append((start, len(voiced_f0)))

    if not segments:
        return {'detected': False, 'rate_hz': 0, 'extent_cents': 0, 'extent_hz': 0}

    # Use longest segment
    longest = max(segments, key=lambda s: s[1] - s[0])
    seg = voiced_f0[longest[0]:longest[1]]
    if len(seg) < 20:
        return {'detected': False, 'rate_hz': 0, 'extent_cents': 0, 'extent_hz': 0}

    # Detrend
    seg_detrended = seg - np.polyval(np.polyfit(np.arange(len(seg)), seg, 1), np.arange(len(seg)))

    # FFT of pitch contour
    frame_rate = sr / hop_length
    fft_vals = np.abs(np.fft.rfft(seg_detrended))
    freqs_fft = np.fft.rfftfreq(len(seg_detrended), d=1.0 / frame_rate)

    # Look for peak in vibrato range
    mask = (freqs_fft >= min_rate) & (freqs_fft <= max_rate)
    if not np.any(mask):
        return {'detected': False, 'rate_hz': 0, 'extent_cents': 0, 'extent_hz': 0}

    fft_masked = fft_vals[mask]
    freqs_masked = freqs_fft[mask]
    peak_idx = np.argmax(fft_masked)
    peak_power = fft_masked[peak_idx]
    total_power = np.sum(fft_vals[1:]) + 1e-10

    vibrato_rate = freqs_masked[peak_idx]
    vibrato_ratio = peak_power / total_power

    # Extent: std of detrended pitch in the segment
    extent_hz = np.std(seg_detrended) * 2  # approximate peak-to-peak / 2
    mean_hz = np.mean(seg)
    extent_cents = 1200 * np.log2((mean_hz + extent_hz) / (mean_hz + 1e-10)) if mean_hz > 0 else 0

    detected = vibrato_ratio > 0.15 and extent_hz > 1.0

    return {
        'detected': bool(detected),
        'rate_hz': safe_float(vibrato_rate),
        'extent_hz': safe_float(extent_hz),
        'extent_cents': safe_float(abs(extent_cents)),
        'strength': safe_float(vibrato_ratio),
    }


def compute_note_histogram(f0, voiced_flag):
    """Map voiced F0 to nearest notes and build histogram."""
    voiced_f0 = f0[voiced_flag & ~np.isnan(f0)]
    if len(voiced_f0) == 0:
        return []

    note_counts = {}
    for hz in voiced_f0:
        midi = hz_to_midi(hz)
        nearest = int(round(midi))
        name = midi_to_note(nearest)
        note_counts[name] = note_counts.get(name, 0) + 1

    total = sum(note_counts.values())
    histogram = sorted(
        [{'note': k, 'count': v, 'percent': safe_float(v / total * 100)}
         for k, v in note_counts.items()],
        key=lambda x: -x['count']
    )
    return histogram[:20]  # top 20 notes


def compute_pitch_segments(f0, voiced_flag, times, n_segments=10):
    """Per-segment pitch statistics."""
    n = len(f0)
    seg_len = max(1, n // n_segments)
    segments = []

    for i in range(n_segments):
        start = i * seg_len
        end = min(start + seg_len, n)
        if start >= n: break
        seg_f0 = f0[start:end]
        seg_voiced = voiced_flag[start:end]
        seg_v = seg_f0[seg_voiced & ~np.isnan(seg_f0)]

        segments.append({
            'segment': i + 1,
            'start_sec': safe_float(times[start]),
            'end_sec': safe_float(times[min(end - 1, n - 1)]),
            'mean_hz': safe_float(np.mean(seg_v)) if len(seg_v) > 0 else None,
            'std_hz': safe_float(np.std(seg_v)) if len(seg_v) > 0 else None,
            'voiced_percent': safe_float(len(seg_v) / max(end - start, 1) * 100),
            'mean_note': midi_to_note(np.mean([hz_to_midi(h) for h in seg_v])) if len(seg_v) > 0 else 'N/A',
        })

    return segments


def compute_pitch_confidence(voiced_prob):
    """Confidence metrics from voicing probability."""
    return {
        'mean_confidence': safe_float(np.mean(voiced_prob)),
        'median_confidence': safe_float(np.median(voiced_prob)),
        'high_confidence_percent': safe_float(np.mean(voiced_prob > 0.8) * 100),
        'low_confidence_percent': safe_float(np.mean(voiced_prob < 0.3) * 100),
    }


def generate_interpretation(stats, jitter_shimmer, vibrato, confidence, segments, note_hist):
    """Generate interpretation."""
    lines = []

    if stats['n_voiced'] == 0:
        return {'summary': 'No voiced content detected — signal may be noise or non-pitched.', 'details': ['No pitched frames found in the signal.']}

    # Pitch range
    lines.append(f"Mean pitch: {stats['mean_hz']:.1f} Hz ({stats['mean_note']}) with range {stats['range_hz']:.1f} Hz ({stats['range_semitones']:.1f} semitones).")

    # Voicing
    vp = stats['voiced_percent']
    if vp > 80:
        lines.append(f"Signal is {vp:.0f}% voiced — predominantly pitched content (speech vowels, singing, tonal instruments).")
    elif vp > 40:
        lines.append(f"Signal is {vp:.0f}% voiced — mix of pitched and unpitched content.")
    else:
        lines.append(f"Signal is only {vp:.0f}% voiced — mostly unpitched content (noise, consonants, percussion).")

    # Stability
    cv = stats['cv']
    if cv < 0.05:
        lines.append(f"Very stable pitch (CV: {cv:.4f}) — sustained tone or monotone.")
    elif cv < 0.15:
        lines.append(f"Relatively stable pitch (CV: {cv:.4f}) — normal speaking or singing range.")
    else:
        lines.append(f"Wide pitch variation (CV: {cv:.4f}) — expressive speech, singing melody, or multiple sources.")

    # Jitter
    jit = jitter_shimmer['jitter_percent']
    if jit < 1.0:
        lines.append(f"Low jitter ({jit:.2f}%) — smooth pitch transitions.")
    elif jit < 3.0:
        lines.append(f"Moderate jitter ({jit:.2f}%) — normal for speech.")
    else:
        lines.append(f"High jitter ({jit:.2f}%) — rough or breathy voice quality.")

    # Vibrato
    if vibrato['detected']:
        lines.append(f"Vibrato detected: rate ≈ {vibrato['rate_hz']:.1f} Hz, extent ≈ {vibrato['extent_cents']:.0f} cents ({vibrato['extent_hz']:.1f} Hz).")
    else:
        lines.append("No significant vibrato detected.")

    # Top notes
    if note_hist:
        top3 = ', '.join([f"{n['note']} ({n['percent']:.1f}%)" for n in note_hist[:3]])
        lines.append(f"Most frequent notes: {top3}.")

    # Confidence
    hc = confidence['high_confidence_percent']
    lines.append(f"Pitch tracking confidence: {hc:.0f}% high-confidence frames (>{0.8} voicing probability).")

    return {'summary': ' '.join(lines[:2]), 'details': lines}


def build_pitch_table(f0, voiced_flag, voiced_prob, times, n_points=400):
    """Sampled pitch data for frontend."""
    n = len(times)
    step = max(1, n // n_points)
    table = []
    for i in range(0, n, step):
        hz = safe_float(f0[i]) if not np.isnan(f0[i]) else None
        table.append({
            'time_sec': safe_float(times[i]),
            'f0_hz': hz,
            'midi': safe_float(hz_to_midi(hz)) if hz and hz > 0 else None,
            'note': midi_to_note(hz_to_midi(hz)) if hz and hz > 0 else None,
            'cents_dev': safe_float(hz_to_cents_deviation(hz)) if hz and hz > 0 else None,
            'voiced': bool(voiced_flag[i]),
            'confidence': safe_float(voiced_prob[i]),
        })
    return table


# ──────────────────────────────────────────────
# Plot Generation
# ──────────────────────────────────────────────

def generate_pitch_plots(
    y, sr, f0, voiced_flag, voiced_prob, times,
    stats, jitter_shimmer, vibrato, note_hist, segments,
    hop_length,
) -> str:

    fig = plt.figure(figsize=(18, 32))
    fig.suptitle('Pitch Tracking Analysis', fontsize=18, fontweight='bold')
    gs = gridspec.GridSpec(8, 2, figure=fig, hspace=0.55, wspace=0.3)

    voiced_f0 = f0.copy()
    voiced_f0[~voiced_flag | np.isnan(voiced_f0)] = np.nan

    # 1. Waveform
    ax1 = fig.add_subplot(gs[0, :])
    t = np.arange(len(y)) / sr
    ax1.plot(t, y, color='#90CAF9', linewidth=0.3, alpha=0.5)
    ax1.set_title('Time Domain — Waveform', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Amplitude')
    ax1.grid(True, linestyle='--', alpha=0.4)

    # 2. Pitch Contour (Hz)
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(times, voiced_f0, color='#F44336', linewidth=1, marker='.', markersize=1)
    if stats['mean_hz'] > 0:
        ax2.axhline(y=stats['mean_hz'], color='#2196F3', linestyle='--', linewidth=0.8,
                     label=f'Mean: {stats["mean_hz"]:.1f} Hz ({stats["mean_note"]})')
    ax2.set_title('Pitch Contour (F0)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Frequency (Hz)')
    ax2.legend(fontsize=8); ax2.grid(True, linestyle='--', alpha=0.4)

    # 3. Pitch Contour (MIDI)
    ax3 = fig.add_subplot(gs[2, 0])
    midi_vals = np.array([hz_to_midi(h) if not np.isnan(h) and h > 0 else np.nan for h in voiced_f0])
    ax3.plot(times, midi_vals, color='#9C27B0', linewidth=0.8, marker='.', markersize=1)
    if stats['mean_midi'] > 0:
        ax3.axhline(y=stats['mean_midi'], color='#F44336', linestyle='--', linewidth=0.8)
    ax3.set_title('Pitch Contour (MIDI Number)', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Time (s)'); ax3.set_ylabel('MIDI')
    ax3.grid(True, linestyle='--', alpha=0.4)

    # 4. Voicing Probability
    ax4 = fig.add_subplot(gs[2, 1])
    ax4.plot(times, voiced_prob, color='#4CAF50', linewidth=0.6)
    ax4.fill_between(times, voiced_prob, alpha=0.15, color='#4CAF50')
    ax4.axhline(y=0.5, color='gray', linestyle=':', linewidth=0.5)
    ax4.set_title('Voicing Probability', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Time (s)'); ax4.set_ylabel('Probability')
    ax4.set_ylim(0, 1); ax4.grid(True, linestyle='--', alpha=0.4)

    # 5. Pitch Histogram (Hz)
    ax5 = fig.add_subplot(gs[3, 0])
    valid_f0 = f0[voiced_flag & ~np.isnan(f0)]
    if len(valid_f0) > 0:
        ax5.hist(valid_f0, bins=60, color='#FF5722', edgecolor='white', linewidth=0.3, alpha=0.85)
        ax5.axvline(x=stats['mean_hz'], color='#F44336', linestyle='--', linewidth=1,
                     label=f'Mean: {stats["mean_hz"]:.1f} Hz')
        ax5.axvline(x=stats['median_hz'], color='#4CAF50', linestyle='--', linewidth=1,
                     label=f'Median: {stats["median_hz"]:.1f} Hz')
        ax5.legend(fontsize=8)
    ax5.set_title('Pitch Distribution (Hz)', fontsize=13, fontweight='bold')
    ax5.set_xlabel('Frequency (Hz)'); ax5.set_ylabel('Count')
    ax5.grid(True, linestyle='--', alpha=0.4, axis='y')

    # 6. Note Histogram
    ax6 = fig.add_subplot(gs[3, 1])
    if note_hist:
        names = [n['note'] for n in note_hist[:15]]
        pcts = [n['percent'] for n in note_hist[:15]]
        max_pct = max(pcts) if pcts else 1
        colors_n = ['#F44336' if p == max_pct else '#FF9800' if p > np.mean(pcts) else '#4CAF50' for p in pcts]
        ax6.barh(names[::-1], pcts[::-1], color=colors_n[::-1], edgecolor='white', linewidth=0.8)
    ax6.set_title('Note Distribution (top 15)', fontsize=13, fontweight='bold')
    ax6.set_xlabel('Percent (%)'); ax6.grid(True, linestyle='--', alpha=0.4, axis='x')

    # 7. Cents Deviation
    ax7 = fig.add_subplot(gs[4, 0])
    if len(valid_f0) > 0:
        cents = [hz_to_cents_deviation(h) for h in valid_f0]
        ax7.hist(cents, bins=50, color='#00BCD4', edgecolor='white', linewidth=0.3, alpha=0.85)
        ax7.axvline(x=0, color='gray', linewidth=0.5)
        ax7.axvline(x=np.mean(cents), color='#F44336', linestyle='--', linewidth=1,
                     label=f'Mean: {np.mean(cents):.1f} cents')
        ax7.legend(fontsize=8)
    ax7.set_title('Cents Deviation from Nearest Note', fontsize=13, fontweight='bold')
    ax7.set_xlabel('Cents'); ax7.set_ylabel('Count')
    ax7.grid(True, linestyle='--', alpha=0.4, axis='y')

    # 8. Voiced/Unvoiced Timeline
    ax8 = fig.add_subplot(gs[4, 1])
    voiced_int = voiced_flag.astype(float)
    ax8.fill_between(times, voiced_int, alpha=0.4, color='#4CAF50', label='Voiced')
    ax8.fill_between(times, 1 - voiced_int, alpha=0.2, color='#F44336', label='Unvoiced')
    ax8.set_title(f'Voiced/Unvoiced ({stats["voiced_percent"]:.0f}% voiced)', fontsize=13, fontweight='bold')
    ax8.set_xlabel('Time (s)'); ax8.set_ylabel('Active')
    ax8.legend(fontsize=8); ax8.grid(True, linestyle='--', alpha=0.4)

    # 9. Segment-wise Pitch
    ax9 = fig.add_subplot(gs[5, 0])
    seg_labels = [f'S{s["segment"]}' for s in segments]
    seg_means = [s['mean_hz'] if s['mean_hz'] is not None else 0 for s in segments]
    seg_stds = [s['std_hz'] if s['std_hz'] is not None else 0 for s in segments]
    colors_s = ['#F44336' if v == max(seg_means) else '#4CAF50' if v == min([x for x in seg_means if x > 0] or [0]) else '#2196F3' for v in seg_means]
    ax9.bar(range(len(segments)), seg_means, yerr=seg_stds, color=colors_s,
            edgecolor='white', linewidth=0.8, capsize=3)
    ax9.set_xticks(range(len(segments))); ax9.set_xticklabels(seg_labels, fontsize=7)
    ax9.set_title('Segment-wise Mean Pitch', fontsize=13, fontweight='bold')
    ax9.set_ylabel('F0 (Hz)'); ax9.grid(True, linestyle='--', alpha=0.4, axis='y')

    # 10. Confidence Histogram
    ax10 = fig.add_subplot(gs[5, 1])
    ax10.hist(voiced_prob, bins=50, color='#3F51B5', edgecolor='white', linewidth=0.3, alpha=0.85)
    ax10.axvline(x=0.5, color='gray', linestyle=':', linewidth=0.8, label='Threshold (0.5)')
    ax10.set_title('Voicing Confidence Distribution', fontsize=13, fontweight='bold')
    ax10.set_xlabel('Probability'); ax10.set_ylabel('Count')
    ax10.legend(fontsize=8); ax10.grid(True, linestyle='--', alpha=0.4, axis='y')

    # 11. Jitter/Shimmer/Vibrato Summary
    ax11 = fig.add_subplot(gs[6, 0])
    q_labels = ['Jitter\n(%)', 'Shimmer\n(dB)', 'Vibrato\nRate (Hz)', 'Vibrato\nExtent (¢)']
    q_vals = [jitter_shimmer['jitter_percent'], jitter_shimmer['shimmer_approx_db'],
              vibrato.get('rate_hz', 0), vibrato.get('extent_cents', 0)]
    q_colors = ['#F44336', '#FF9800', '#9C27B0', '#4CAF50']
    bars11 = ax11.bar(q_labels, q_vals, color=q_colors, edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars11, q_vals):
        ax11.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f'{val:.2f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax11.set_title('Voice Quality & Vibrato', fontsize=13, fontweight='bold')
    ax11.grid(True, linestyle='--', alpha=0.4, axis='y')

    # 12. Key Metrics
    ax12 = fig.add_subplot(gs[6, 1])
    k_labels = ['Mean\nF0 (Hz)', 'Range\n(st)', 'Voiced\n(%)', 'CV', 'Confidence\n(%)']
    k_vals_raw = [stats['mean_hz'], stats['range_semitones'], stats['voiced_percent'],
                  stats['cv'] * 100, safe_float(np.mean(voiced_prob) * 100)]
    k_colors = ['#F44336', '#2196F3', '#4CAF50', '#FF9800', '#9C27B0']
    bars12 = ax12.bar(k_labels, k_vals_raw, color=k_colors, edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars12, k_vals_raw):
        ax12.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax12.set_title('Key Metrics', fontsize=13, fontweight='bold')
    ax12.grid(True, linestyle='--', alpha=0.4, axis='y')

    # 13. Zoomed pitch contour
    ax13 = fig.add_subplot(gs[7, :])
    zoom_end = min(5.0, times[-1] if len(times) > 0 else 0)
    mask = times <= zoom_end
    ax13.plot(times[mask], voiced_f0[mask], color='#F44336', linewidth=1.5, marker='.', markersize=2)
    ax13.set_title(f'Zoomed Pitch Contour (0–{zoom_end:.1f}s)', fontsize=13, fontweight='bold')
    ax13.set_xlabel('Time (s)'); ax13.set_ylabel('F0 (Hz)')
    ax13.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"


# ──────────────────────────────────────────────
# Main Endpoint
# ──────────────────────────────────────────────

@router.post("/pitch-tracking")
async def pitch_tracking(
    file: UploadFile = File(...),
    hop_length: Optional[int] = Form(512),
    fmin: Optional[float] = Form(65.0),
    fmax: Optional[float] = Form(2093.0),
    n_segments: Optional[int] = Form(10),
    target_sr: Optional[int] = Form(None),
):
    """
    Pitch Tracking Analysis.

    Comprehensive F0 estimation using pYIN with voicing probability,
    pitch statistics, jitter/shimmer, vibrato detection, note mapping,
    and segment analysis.
    """
    if not LIBROSA_AVAILABLE:
        raise HTTPException(status_code=500, detail="librosa not installed")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    allowed = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed:
        raise HTTPException(status_code=400, detail=f"Unsupported: {ext}")

    try:
        content = await file.read()
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp.write(content); tmp_path = tmp.name

        try:
            y, sr = librosa.load(tmp_path, sr=target_sr if target_sr else None, mono=True)
            y_t, _ = librosa.effects.trim(y, top_db=30)
            if len(y_t) > sr * 60: y_t = y_t[:sr * 60]

            hl = hop_length or 512
            fmin_v = fmin or 65.0
            fmax_v = fmax or 2093.0
            n_seg = n_segments or 10

            f0, voiced_flag, voiced_prob, times = track_pitch_pyin(y_t, sr, fmin_v, fmax_v, hl)
            stats = compute_pitch_statistics(f0, voiced_flag)
            jitter_shimmer = compute_jitter_shimmer(f0, voiced_flag, y_t, sr, hl)
            vibrato = detect_vibrato(f0, voiced_flag, sr, hl)
            note_hist = compute_note_histogram(f0, voiced_flag)
            segments = compute_pitch_segments(f0, voiced_flag, times, n_seg)
            confidence = compute_pitch_confidence(voiced_prob)
            interpretation = generate_interpretation(stats, jitter_shimmer, vibrato, confidence, segments, note_hist)
            pitch_table = build_pitch_table(f0, voiced_flag, voiced_prob, times, 400)

            plot = generate_pitch_plots(
                y_t, sr, f0, voiced_flag, voiced_prob, times,
                stats, jitter_shimmer, vibrato, note_hist, segments, hl
            )

            file_info = {
                'filename': file.filename, 'file_size_mb': round(len(content)/(1024*1024), 2),
                'sample_rate': sr, 'duration_sec': round(len(y_t)/sr, 2),
                'n_samples': len(y_t), 'format': ext.replace('.',''),
                'hop_length': hl, 'fmin': fmin_v, 'fmax': fmax_v, 'n_segments': n_seg,
                'n_frames': int(len(f0)),
                'time_resolution_ms': safe_float(hl / sr * 1000),
            }

            return _to_native({
                'results': {
                    'file_info': file_info,
                    'pitch_stats': stats,
                    'jitter_shimmer': jitter_shimmer,
                    'vibrato': vibrato,
                    'confidence': confidence,
                    'note_histogram': note_hist,
                    'segments': segments,
                    'pitch_table': pitch_table,
                    'interpretation': interpretation,
                },
                'plot': plot
            })
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ──────────────────────────────────────────────
# from pitch_tracking import router as pitch_router
# app.include_router(pitch_router, prefix="/api/analysis", tags=["Audio Analysis"])
# → POST /api/analysis/pitch-tracking
# ──────────────────────────────────────────────
