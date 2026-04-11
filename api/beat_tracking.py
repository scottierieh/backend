"""
Beat Tracking Analysis Backend (FastAPI)
- Comprehensive beat and tempo analysis of audio signals
- Beat detection via librosa beat tracker
- Tempo estimation (global and local / dynamic)
- Beat strength / confidence per detected beat
- Inter-beat interval (IBI) analysis and regularity
- Onset strength envelope and onset detection
- Downbeat estimation and time signature hints
- Rhythm pattern analysis and beat grid alignment
- Tempo stability, groove consistency, and swing ratio
- Bar/measure segmentation from beat positions
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
# Beat Tracking Functions
# ──────────────────────────────────────────────

def compute_beats(y, sr, hop_length=512):
    """Detect beats and estimate tempo."""
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop_length)

    # Handle tempo as scalar or array
    if isinstance(tempo, np.ndarray):
        tempo_val = float(tempo[0]) if len(tempo) > 0 else 0.0
    else:
        tempo_val = float(tempo)

    return tempo_val, beat_frames, beat_times


def compute_onset_envelope(y, sr, hop_length=512):
    """Compute onset strength envelope."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=hop_length)
    return onset_env, onset_times


def compute_onset_peaks(y, sr, hop_length=512):
    """Detect onset peaks."""
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    return onset_frames, onset_times


def compute_ibi_analysis(beat_times):
    """Inter-beat interval analysis."""
    if len(beat_times) < 2:
        return {
            'mean_ibi': 0, 'std_ibi': 0, 'cv_ibi': 0, 'min_ibi': 0, 'max_ibi': 0,
            'median_ibi': 0, 'ibi_range': 0, 'regularity': 0,
            'ibis': [], 'ibi_times': [],
        }

    ibis = np.diff(beat_times)
    mean_ibi = np.mean(ibis)
    std_ibi = np.std(ibis)
    cv = std_ibi / (mean_ibi + 1e-10)

    # Regularity: 1 - CV (higher = more regular)
    regularity = max(0, 1.0 - cv)

    return {
        'mean_ibi': safe_float(mean_ibi),
        'std_ibi': safe_float(std_ibi),
        'cv_ibi': safe_float(cv),
        'min_ibi': safe_float(np.min(ibis)),
        'max_ibi': safe_float(np.max(ibis)),
        'median_ibi': safe_float(np.median(ibis)),
        'ibi_range': safe_float(np.max(ibis) - np.min(ibis)),
        'regularity': safe_float(regularity),
        'ibis': [safe_float(x) for x in ibis],
        'ibi_times': [safe_float(x) for x in beat_times[1:]],
    }


def compute_beat_strength(onset_env, beat_frames, onset_times):
    """Compute onset strength at each beat position."""
    strengths = []
    for bf in beat_frames:
        if bf < len(onset_env):
            strengths.append(safe_float(onset_env[bf]))
        else:
            strengths.append(0.0)
    mean_strength = np.mean(strengths) if strengths else 0.0
    return {
        'strengths': strengths,
        'mean_strength': safe_float(mean_strength),
        'max_strength': safe_float(max(strengths) if strengths else 0),
        'min_strength': safe_float(min(strengths) if strengths else 0),
    }


def compute_dynamic_tempo(onset_env, sr, hop_length=512, n_windows=8):
    """Estimate local tempo over time using windowed analysis."""
    n_frames = len(onset_env)
    win_size = max(1, n_frames // n_windows)
    local_tempos = []

    for i in range(n_windows):
        start = i * win_size
        end = min(start + win_size, n_frames)
        if end - start < 20:
            continue
        seg = onset_env[start:end]
        # Tempo from autocorrelation
        try:
            ac = librosa.autocorrelate(seg)
            # Find first peak after a minimum lag
            min_lag = int(sr / hop_length * 60 / 240)  # 240 BPM max
            max_lag = int(sr / hop_length * 60 / 30)   # 30 BPM min
            max_lag = min(max_lag, len(ac) - 1)
            if min_lag >= max_lag or min_lag >= len(ac):
                local_tempos.append({'window': i + 1, 'start_sec': safe_float(start * hop_length / sr),
                                      'tempo': 0, 'confidence': 0})
                continue
            ac_seg = ac[min_lag:max_lag]
            if len(ac_seg) == 0:
                local_tempos.append({'window': i + 1, 'start_sec': safe_float(start * hop_length / sr),
                                      'tempo': 0, 'confidence': 0})
                continue
            peak_lag = np.argmax(ac_seg) + min_lag
            local_bpm = 60 * sr / (peak_lag * hop_length) if peak_lag > 0 else 0
            confidence = ac_seg[peak_lag - min_lag] / (ac[0] + 1e-10) if ac[0] > 0 else 0
            local_tempos.append({
                'window': i + 1,
                'start_sec': safe_float(start * hop_length / sr),
                'tempo': safe_float(local_bpm),
                'confidence': safe_float(confidence),
            })
        except:
            local_tempos.append({'window': i + 1, 'start_sec': safe_float(start * hop_length / sr),
                                  'tempo': 0, 'confidence': 0})

    return local_tempos


def compute_tempogram(y, sr, hop_length=512):
    """Compute tempogram for visualization."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    tempogram = librosa.feature.tempogram(onset_envelope=onset_env, sr=sr, hop_length=hop_length)
    return tempogram


def estimate_time_signature(beat_times, onset_env, beat_frames):
    """Simple time signature estimation based on beat strength patterns."""
    if len(beat_frames) < 8:
        return {'estimated': '4/4', 'confidence': 0, 'pattern_length': 4}

    strengths = [onset_env[min(bf, len(onset_env)-1)] for bf in beat_frames]
    if not strengths:
        return {'estimated': '4/4', 'confidence': 0, 'pattern_length': 4}

    # Test periodicity at 2, 3, 4, 6
    best_score = -1
    best_meter = 4

    for meter in [2, 3, 4, 6]:
        if len(strengths) < meter * 2:
            continue
        # Average strength at each beat position within the meter
        pattern = np.zeros(meter)
        counts = np.zeros(meter)
        for i, s in enumerate(strengths):
            pos = i % meter
            pattern[pos] += s
            counts[pos] += 1
        pattern = pattern / (counts + 1e-10)

        # Score: how much does beat 0 (downbeat) stand out?
        if len(pattern) > 1:
            downbeat_ratio = pattern[0] / (np.mean(pattern[1:]) + 1e-10)
            score = downbeat_ratio * (np.max(pattern) - np.min(pattern))
        else:
            score = 0
        if score > best_score:
            best_score = score
            best_meter = meter

    sig_map = {2: '2/4', 3: '3/4', 4: '4/4', 6: '6/8'}
    return {
        'estimated': sig_map.get(best_meter, '4/4'),
        'confidence': safe_float(min(1, best_score / 3)),
        'pattern_length': best_meter,
    }


def compute_swing_ratio(beat_times):
    """Estimate swing ratio from beat subdivisions."""
    if len(beat_times) < 4:
        return {'swing_ratio': 1.0, 'is_swung': False}

    ibis = np.diff(beat_times)
    # Look at alternating beat pairs
    even_ibis = ibis[0::2]
    odd_ibis = ibis[1::2]
    n = min(len(even_ibis), len(odd_ibis))
    if n < 2:
        return {'swing_ratio': 1.0, 'is_swung': False}

    ratio = np.mean(even_ibis[:n]) / (np.mean(odd_ibis[:n]) + 1e-10)
    is_swung = abs(ratio - 1.0) > 0.15  # More than 15% deviation
    return {
        'swing_ratio': safe_float(ratio),
        'is_swung': bool(is_swung),
    }


def generate_interpretation(tempo, beat_times, ibi, beat_strength, dynamic_tempo, time_sig, swing):
    """Generate human-readable interpretation."""
    lines = []

    # Tempo
    if tempo > 0:
        if tempo > 160:
            genre_hint = "fast (allegro/vivace — dance, EDM, punk)"
        elif tempo > 120:
            genre_hint = "moderate-fast (allegro — pop, rock)"
        elif tempo > 90:
            genre_hint = "moderate (andante/moderato — many genres)"
        elif tempo > 60:
            genre_hint = "slow (adagio — ballads, ambient)"
        else:
            genre_hint = "very slow (largo — ambient, drone)"
        lines.append(f"Estimated tempo: {tempo:.1f} BPM — {genre_hint}.")
    else:
        lines.append("No clear tempo detected — signal may be non-rhythmic.")

    lines.append(f"Detected {len(beat_times)} beats over {beat_times[-1]:.1f}s." if len(beat_times) > 0 else "No beats detected.")

    # Regularity
    reg = ibi['regularity']
    if reg > 0.9:
        lines.append(f"Very regular rhythm (regularity: {reg:.3f}) — machine-like or metronomic.")
    elif reg > 0.7:
        lines.append(f"Regular rhythm (regularity: {reg:.3f}) — typical for recorded music.")
    elif reg > 0.4:
        lines.append(f"Somewhat irregular rhythm (regularity: {reg:.3f}) — rubato, live performance, or mixed tempos.")
    else:
        lines.append(f"Irregular rhythm (regularity: {reg:.3f}) — free tempo, non-metric, or complex rhythm.")

    # Time signature
    ts = time_sig['estimated']
    lines.append(f"Estimated time signature: {ts} (confidence: {time_sig['confidence']:.2f}).")

    # Dynamic tempo
    if dynamic_tempo:
        tempos = [dt['tempo'] for dt in dynamic_tempo if dt['tempo'] > 0]
        if tempos:
            t_range = max(tempos) - min(tempos)
            if t_range > 20:
                lines.append(f"Significant tempo variation ({min(tempos):.0f}–{max(tempos):.0f} BPM) — tempo changes or rubato detected.")
            elif t_range > 5:
                lines.append(f"Moderate tempo variation ({min(tempos):.0f}–{max(tempos):.0f} BPM).")
            else:
                lines.append(f"Stable tempo throughout ({min(tempos):.0f}–{max(tempos):.0f} BPM).")

    # Swing
    if swing['is_swung']:
        lines.append(f"Swing detected (ratio: {swing['swing_ratio']:.3f}) — alternating beat intervals suggest swing feel.")
    else:
        lines.append(f"Straight rhythm (swing ratio: {swing['swing_ratio']:.3f}).")

    # Beat strength
    lines.append(f"Mean beat strength: {beat_strength['mean_strength']:.3f}.")

    return {
        'summary': ' '.join(lines[:2]),
        'details': lines,
    }


def build_beat_table(beat_times, beat_strength_data, ibi_data):
    """Build beat data table for frontend."""
    table = []
    strengths = beat_strength_data['strengths']
    ibis = ibi_data['ibis']

    for i, bt in enumerate(beat_times):
        row = {
            'beat': i + 1,
            'time_sec': safe_float(bt),
            'strength': strengths[i] if i < len(strengths) else 0,
            'ibi': ibis[i - 1] if i > 0 and (i - 1) < len(ibis) else None,
            'local_bpm': safe_float(60.0 / ibis[i - 1]) if i > 0 and (i - 1) < len(ibis) and ibis[i - 1] > 0 else None,
        }
        table.append(row)
    return table


# ──────────────────────────────────────────────
# Plot Generation
# ──────────────────────────────────────────────

def generate_beat_plots(
    y, sr, beat_times, beat_frames, onset_env, onset_times,
    onset_peak_times, ibi, beat_strength_data, dynamic_tempo,
    tempogram, hop_length, tempo,
) -> str:

    fig = plt.figure(figsize=(18, 32))
    fig.suptitle('Beat Tracking Analysis', fontsize=18, fontweight='bold')
    gs = gridspec.GridSpec(8, 2, figure=fig, hspace=0.55, wspace=0.3)

    # 1. Waveform + Beats
    ax1 = fig.add_subplot(gs[0, :])
    t = np.arange(len(y)) / sr
    ax1.plot(t, y, color='#90CAF9', linewidth=0.3, alpha=0.5)
    for bt in beat_times:
        ax1.axvline(x=bt, color='#F44336', linewidth=0.5, alpha=0.6)
    ax1.set_title(f'Waveform with Beat Markers ({len(beat_times)} beats · {tempo:.1f} BPM)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Amplitude')
    ax1.grid(True, linestyle='--', alpha=0.4)

    # 2. Onset Strength + Beats
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(onset_times, onset_env, color='#FF5722', linewidth=0.7, alpha=0.7)
    ax2.fill_between(onset_times, onset_env, alpha=0.1, color='#FF5722')
    for bt in beat_times:
        ax2.axvline(x=bt, color='#2196F3', linewidth=0.5, alpha=0.5)
    for ot in onset_peak_times:
        ax2.axvline(x=ot, color='#4CAF50', linewidth=0.3, alpha=0.3)
    ax2.set_title('Onset Strength Envelope + Beats (blue) + Onsets (green)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Onset Strength')
    ax2.grid(True, linestyle='--', alpha=0.4)

    # 3. Tempogram
    ax3 = fig.add_subplot(gs[2, :])
    img3 = librosa.display.specshow(tempogram, x_axis='time', y_axis='tempo',
                                     sr=sr, hop_length=hop_length, ax=ax3, cmap='magma')
    ax3.axhline(y=tempo, color='white', linestyle='--', linewidth=1, alpha=0.8, label=f'{tempo:.1f} BPM')
    ax3.set_title('Tempogram', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=8, loc='upper right')
    fig.colorbar(img3, ax=ax3, label='Autocorrelation', pad=0.02)

    # 4. IBI over Time
    ax4 = fig.add_subplot(gs[3, 0])
    if ibi['ibis']:
        ax4.plot(ibi['ibi_times'], ibi['ibis'], 'o-', color='#9C27B0', markersize=3, linewidth=0.8)
        ax4.axhline(y=ibi['mean_ibi'], color='#F44336', linestyle='--', linewidth=0.8,
                     label=f'Mean: {ibi["mean_ibi"]:.3f}s')
        ax4.legend(fontsize=8)
    ax4.set_title('Inter-Beat Interval (IBI)', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Time (s)'); ax4.set_ylabel('IBI (s)')
    ax4.grid(True, linestyle='--', alpha=0.4)

    # 5. IBI Histogram
    ax5 = fig.add_subplot(gs[3, 1])
    if ibi['ibis']:
        ax5.hist(ibi['ibis'], bins=30, color='#9C27B0', edgecolor='white', linewidth=0.3, alpha=0.85)
        ax5.axvline(x=ibi['mean_ibi'], color='#F44336', linestyle='--', linewidth=1,
                     label=f'Mean: {ibi["mean_ibi"]:.3f}s')
        ax5.axvline(x=ibi['median_ibi'], color='#4CAF50', linestyle='--', linewidth=1,
                     label=f'Median: {ibi["median_ibi"]:.3f}s')
        ax5.legend(fontsize=8)
    ax5.set_title('IBI Distribution', fontsize=13, fontweight='bold')
    ax5.set_xlabel('IBI (s)'); ax5.set_ylabel('Count')
    ax5.grid(True, linestyle='--', alpha=0.4, axis='y')

    # 6. Beat Strength
    ax6 = fig.add_subplot(gs[4, 0])
    if beat_strength_data['strengths']:
        ax6.bar(range(len(beat_strength_data['strengths'])), beat_strength_data['strengths'],
                color='#FF9800', edgecolor='white', linewidth=0.2, alpha=0.85)
        ax6.axhline(y=beat_strength_data['mean_strength'], color='#F44336', linestyle='--', linewidth=0.8,
                     label=f'Mean: {beat_strength_data["mean_strength"]:.3f}')
        ax6.legend(fontsize=8)
    ax6.set_title('Beat Onset Strength', fontsize=13, fontweight='bold')
    ax6.set_xlabel('Beat #'); ax6.set_ylabel('Strength')
    ax6.grid(True, linestyle='--', alpha=0.4, axis='y')

    # 7. Local BPM (from IBI)
    ax7 = fig.add_subplot(gs[4, 1])
    if ibi['ibis']:
        local_bpms = [60.0 / x if x > 0 else 0 for x in ibi['ibis']]
        ax7.plot(ibi['ibi_times'], local_bpms, 'o-', color='#2196F3', markersize=3, linewidth=0.8)
        ax7.axhline(y=tempo, color='#F44336', linestyle='--', linewidth=0.8,
                     label=f'Global: {tempo:.1f} BPM')
        ax7.legend(fontsize=8)
    ax7.set_title('Local BPM (from IBI)', fontsize=13, fontweight='bold')
    ax7.set_xlabel('Time (s)'); ax7.set_ylabel('BPM')
    ax7.grid(True, linestyle='--', alpha=0.4)

    # 8. Dynamic Tempo Windows
    ax8 = fig.add_subplot(gs[5, 0])
    if dynamic_tempo:
        dt_x = [dt['start_sec'] for dt in dynamic_tempo]
        dt_y = [dt['tempo'] for dt in dynamic_tempo]
        dt_c = [dt['confidence'] for dt in dynamic_tempo]
        ax8.bar(range(len(dt_x)), dt_y, color=plt.cm.viridis([c for c in dt_c]),
                edgecolor='white', linewidth=0.8)
        ax8.axhline(y=tempo, color='#F44336', linestyle='--', linewidth=0.8, label=f'Global: {tempo:.1f}')
        ax8.set_xticks(range(len(dt_x)))
        ax8.set_xticklabels([f'{x:.1f}s' for x in dt_x], fontsize=7, rotation=30)
        ax8.legend(fontsize=8)
    ax8.set_title('Windowed Tempo Estimation', fontsize=13, fontweight='bold')
    ax8.set_xlabel('Window Start'); ax8.set_ylabel('Tempo (BPM)')
    ax8.grid(True, linestyle='--', alpha=0.4, axis='y')

    # 9. Beat Grid Alignment (deviation from grid)
    ax9 = fig.add_subplot(gs[5, 1])
    if len(beat_times) > 2:
        ideal_ibi = ibi['mean_ibi']
        if ideal_ibi > 0:
            ideal_times = np.arange(beat_times[0], beat_times[-1], ideal_ibi)
            deviations = []
            for bt in beat_times:
                closest = ideal_times[np.argmin(np.abs(ideal_times - bt))]
                deviations.append((bt - closest) * 1000)  # ms
            ax9.bar(range(len(deviations)), deviations, color='#00BCD4', edgecolor='white', linewidth=0.2)
            ax9.axhline(y=0, color='gray', linewidth=0.5)
            ax9.set_xlabel('Beat #'); ax9.set_ylabel('Deviation (ms)')
    ax9.set_title('Beat Grid Deviation', fontsize=13, fontweight='bold')
    ax9.grid(True, linestyle='--', alpha=0.4, axis='y')

    # 10. Onset Density
    ax10 = fig.add_subplot(gs[6, 0])
    # Bin onsets into 1-second windows
    if len(onset_peak_times) > 0:
        max_t = max(onset_peak_times)
        bins = np.arange(0, max_t + 1, 1)
        density, _ = np.histogram(onset_peak_times, bins=bins)
        ax10.bar(bins[:-1], density, width=0.9, color='#4CAF50', edgecolor='white', linewidth=0.3, alpha=0.85)
    ax10.set_title('Onset Density (per second)', fontsize=13, fontweight='bold')
    ax10.set_xlabel('Time (s)'); ax10.set_ylabel('Onsets')
    ax10.grid(True, linestyle='--', alpha=0.4, axis='y')

    # 11. Key Metrics
    ax11 = fig.add_subplot(gs[6, 1])
    met_labels = ['Tempo\n(BPM)', 'Beats', 'Regularity', 'Mean IBI\n(s)', 'Mean\nStrength']
    met_vals = [tempo / 10, len(beat_times) / 10, ibi['regularity'],
                ibi['mean_ibi'], beat_strength_data['mean_strength']]
    met_colors = ['#F44336', '#2196F3', '#4CAF50', '#9C27B0', '#FF9800']
    bars11 = ax11.bar(met_labels, met_vals, color=met_colors, edgecolor='white', linewidth=0.8)
    display_vals = [f'{tempo:.1f}', str(len(beat_times)), f'{ibi["regularity"]:.3f}',
                    f'{ibi["mean_ibi"]:.3f}', f'{beat_strength_data["mean_strength"]:.3f}']
    for bar, dv in zip(bars11, display_vals):
        ax11.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.02,
                 dv, ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax11.set_title('Key Metrics', fontsize=13, fontweight='bold')
    ax11.set_ylabel('Scaled Value'); ax11.grid(True, linestyle='--', alpha=0.4, axis='y')

    # 12. Click track (zoomed 5s)
    ax12 = fig.add_subplot(gs[7, :])
    zoom_end = min(5.0, len(y) / sr)
    zoom_mask = t <= zoom_end
    ax12.plot(t[zoom_mask], y[zoom_mask], color='#90CAF9', linewidth=0.5, alpha=0.7)
    for bt in beat_times:
        if bt <= zoom_end:
            ax12.axvline(x=bt, color='#F44336', linewidth=1.5, alpha=0.8)
    for ot in onset_peak_times:
        if ot <= zoom_end:
            ax12.axvline(x=ot, color='#4CAF50', linewidth=0.5, alpha=0.4)
    ax12.set_title(f'Zoomed View (0–{zoom_end:.1f}s) — Beats (red) + Onsets (green)', fontsize=13, fontweight='bold')
    ax12.set_xlabel('Time (s)'); ax12.set_ylabel('Amplitude')
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

@router.post("/beat-tracking")
async def beat_tracking(
    file: UploadFile = File(...),
    hop_length: Optional[int] = Form(512),
    n_tempo_windows: Optional[int] = Form(8),
    target_sr: Optional[int] = Form(None),
):
    """
    Beat Tracking Analysis.

    Comprehensive beat and tempo analysis including beat detection, tempo
    estimation (global + local), IBI analysis, onset detection, tempogram,
    time signature estimation, swing detection, and beat grid alignment.
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
            if len(y) > sr * 120: y = y[:sr * 120]  # 2 min max for beat tracking

            hl = hop_length or 512
            n_tw = n_tempo_windows or 8

            # Beat detection
            tempo, beat_frames, beat_times = compute_beats(y, sr, hl)

            # Onset envelope
            onset_env, onset_times = compute_onset_envelope(y, sr, hl)
            onset_peak_frames, onset_peak_times = compute_onset_peaks(y, sr, hl)

            # IBI
            ibi = compute_ibi_analysis(beat_times)

            # Beat strength
            beat_strength = compute_beat_strength(onset_env, beat_frames, onset_times)

            # Dynamic tempo
            dynamic_tempo = compute_dynamic_tempo(onset_env, sr, hl, n_tw)

            # Tempogram
            tempogram = compute_tempogram(y, sr, hl)

            # Time signature
            time_sig = estimate_time_signature(beat_times, onset_env, beat_frames)

            # Swing
            swing = compute_swing_ratio(beat_times)

            # Interpretation
            interpretation = generate_interpretation(
                tempo, beat_times, ibi, beat_strength, dynamic_tempo, time_sig, swing
            )

            # Beat table
            beat_table = build_beat_table(beat_times, beat_strength, ibi)

            # Plot
            plot = generate_beat_plots(
                y, sr, beat_times, beat_frames, onset_env, onset_times,
                onset_peak_times, ibi, beat_strength, dynamic_tempo,
                tempogram, hl, tempo
            )

            file_info = {
                'filename': file.filename, 'file_size_mb': round(len(content)/(1024*1024), 2),
                'sample_rate': sr, 'duration_sec': round(len(y)/sr, 2),
                'n_samples': len(y), 'format': ext.replace('.',''),
                'hop_length': hl, 'n_tempo_windows': n_tw,
            }

            return _to_native({
                'results': {
                    'file_info': file_info,
                    'tempo': safe_float(tempo),
                    'n_beats': len(beat_times),
                    'n_onsets': len(onset_peak_times),
                    'ibi_analysis': {k: v for k, v in ibi.items() if k not in ('ibis', 'ibi_times')},
                    'beat_strength': {k: v for k, v in beat_strength.items() if k != 'strengths'},
                    'dynamic_tempo': dynamic_tempo,
                    'time_signature': time_sig,
                    'swing': swing,
                    'beat_table': beat_table[:200],
                    'interpretation': interpretation,
                },
                'plot': plot
            })
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ──────────────────────────────────────────────
# from beat_tracking import router as beat_router
# app.include_router(beat_router, prefix="/api/analysis", tags=["Audio Analysis"])
# → POST /api/analysis/beat-tracking
# ──────────────────────────────────────────────
