"""
Onset Detection Analysis Backend (FastAPI)
- Comprehensive onset/transient detection and characterization
- Multiple onset detection functions (energy, spectral flux, HFC, complex)
- Onset strength envelope with configurable parameters
- Per-onset characterization: strength, rise time, inter-onset interval
- Onset density profiling and temporal distribution
- Onset clustering and regularity analysis
- Backtracking for precise onset timing
- Onset-based segmentation and event rate analysis
- Comparison of detection methods
- Attack/transient analysis per detected onset
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
# Onset Detection Functions
# ──────────────────────────────────────────────

def compute_onset_envelope(y, sr, hop_length=512, method='default'):
    """Compute onset strength envelope."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    times = librosa.frames_to_time(np.arange(len(onset_env)), sr=sr, hop_length=hop_length)
    return onset_env, times


def detect_onsets(y, sr, hop_length=512, backtrack=True, delta=0.07, wait=1):
    """Detect onsets with configurable parameters."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, hop_length=hop_length,
        backtrack=backtrack, delta=delta, wait=wait,
        onset_envelope=onset_env,
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    return onset_frames, onset_times, onset_env


def compute_multi_method_onsets(y, sr, hop_length=512):
    """Detect onsets using multiple spectral methods for comparison."""
    methods = {}

    # Default (mel-scaled spectrogram flux)
    env_default = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    frames_default = librosa.onset.onset_detect(onset_envelope=env_default, sr=sr, hop_length=hop_length)

    # RMS energy difference
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=hop_length)[0]
    rms_diff = np.diff(rms, prepend=0)
    rms_diff = np.maximum(0, rms_diff)  # half-wave rectify
    env_energy = rms_diff / (np.max(rms_diff) + 1e-10) * np.max(env_default)

    # Spectral flux (magnitude difference)
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=hop_length))
    flux = np.sqrt(np.mean(np.maximum(0, np.diff(S, axis=1)) ** 2, axis=0))
    flux = np.concatenate([[0], flux])
    env_flux = flux / (np.max(flux) + 1e-10) * np.max(env_default)

    # High-frequency content
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    hfc_weights = freqs ** 2
    hfc = np.sum(S ** 2 * hfc_weights[:, None], axis=0)
    hfc_diff = np.diff(hfc, prepend=0)
    hfc_diff = np.maximum(0, hfc_diff)
    env_hfc = hfc_diff / (np.max(hfc_diff) + 1e-10) * np.max(env_default)

    n = min(len(env_default), len(env_energy), len(env_flux), len(env_hfc))

    methods = {
        'default': {'envelope': env_default[:n], 'n_onsets': len(frames_default)},
        'energy': {'envelope': env_energy[:n], 'n_onsets': 0},
        'spectral_flux': {'envelope': env_flux[:n], 'n_onsets': 0},
        'hfc': {'envelope': env_hfc[:n], 'n_onsets': 0},
    }

    # Detect onsets for each method using peak picking on their envelopes
    for key in ['energy', 'spectral_flux', 'hfc']:
        try:
            frames = librosa.onset.onset_detect(onset_envelope=methods[key]['envelope'],
                                                 sr=sr, hop_length=hop_length)
            methods[key]['n_onsets'] = len(frames)
        except:
            methods[key]['n_onsets'] = 0

    return methods, n


def compute_onset_characterization(onset_frames, onset_times, onset_env, y, sr, hop_length):
    """Characterize each detected onset."""
    onsets = []
    n_env = len(onset_env)

    for i, (fr, t) in enumerate(zip(onset_frames, onset_times)):
        strength = safe_float(onset_env[min(fr, n_env - 1)])

        # Rise time: frames from previous local minimum to this peak
        rise_frames = 0
        if fr > 0:
            j = fr - 1
            while j > 0 and onset_env[j] < onset_env[j + 1]:
                rise_frames += 1
                j -= 1
        rise_time_ms = rise_frames * hop_length / sr * 1000

        # IOI (inter-onset interval)
        ioi = safe_float(onset_times[i] - onset_times[i - 1]) if i > 0 else None

        # Local RMS at onset
        sample_idx = int(t * sr)
        win = min(2048, len(y) - sample_idx)
        if win > 0:
            local_rms = np.sqrt(np.mean(y[sample_idx:sample_idx + win] ** 2))
        else:
            local_rms = 0
        local_rms_db = 20 * np.log10(local_rms + 1e-10)

        onsets.append({
            'index': i + 1,
            'time_sec': safe_float(t),
            'frame': int(fr),
            'strength': strength,
            'rise_time_ms': safe_float(rise_time_ms),
            'ioi_sec': ioi,
            'local_rms_db': safe_float(local_rms_db),
        })

    return onsets


def compute_ioi_analysis(onset_times):
    """Inter-onset interval statistics."""
    if len(onset_times) < 2:
        return {'mean': 0, 'std': 0, 'cv': 0, 'min': 0, 'max': 0, 'median': 0,
                'regularity': 0, 'iois': [], 'n_intervals': 0}

    iois = np.diff(onset_times)
    mean = np.mean(iois)
    std = np.std(iois)
    cv = std / (mean + 1e-10)

    return {
        'mean': safe_float(mean),
        'std': safe_float(std),
        'cv': safe_float(cv),
        'min': safe_float(np.min(iois)),
        'max': safe_float(np.max(iois)),
        'median': safe_float(np.median(iois)),
        'regularity': safe_float(max(0, 1 - cv)),
        'iois': [safe_float(x) for x in iois],
        'n_intervals': len(iois),
    }


def compute_onset_density(onset_times, duration, bin_size=1.0):
    """Onset density over time."""
    if len(onset_times) == 0:
        return {'bins': [], 'density': [], 'mean_density': 0, 'max_density': 0, 'global_rate': 0}

    bins = np.arange(0, duration + bin_size, bin_size)
    density, _ = np.histogram(onset_times, bins=bins)

    return {
        'bins': [safe_float(b) for b in bins[:-1]],
        'density': [int(d) for d in density],
        'mean_density': safe_float(np.mean(density)),
        'max_density': int(np.max(density)),
        'min_density': int(np.min(density)),
        'std_density': safe_float(np.std(density)),
        'global_rate': safe_float(len(onset_times) / duration),
    }


def compute_onset_segments(onset_times, duration, n_segments=10):
    """Per-segment onset statistics."""
    seg_dur = duration / n_segments
    segments = []

    for i in range(n_segments):
        start = i * seg_dur
        end = (i + 1) * seg_dur
        seg_onsets = [t for t in onset_times if start <= t < end]
        n = len(seg_onsets)
        iois = np.diff(seg_onsets) if len(seg_onsets) > 1 else []

        segments.append({
            'segment': i + 1,
            'start_sec': safe_float(start),
            'end_sec': safe_float(end),
            'n_onsets': n,
            'rate': safe_float(n / seg_dur),
            'mean_ioi': safe_float(np.mean(iois)) if len(iois) > 0 else None,
        })

    return segments


def compute_strength_statistics(onset_env, onset_frames):
    """Statistics of onset strength envelope."""
    all_strengths = onset_env
    onset_strengths = onset_env[onset_frames[onset_frames < len(onset_env)]] if len(onset_frames) > 0 else np.array([0])

    return {
        'env_mean': safe_float(np.mean(all_strengths)),
        'env_std': safe_float(np.std(all_strengths)),
        'env_max': safe_float(np.max(all_strengths)),
        'onset_mean_strength': safe_float(np.mean(onset_strengths)),
        'onset_std_strength': safe_float(np.std(onset_strengths)),
        'onset_max_strength': safe_float(np.max(onset_strengths)),
        'onset_min_strength': safe_float(np.min(onset_strengths)),
        'strength_ratio': safe_float(np.mean(onset_strengths) / (np.mean(all_strengths) + 1e-10)),
    }


def generate_interpretation(n_onsets, duration, ioi, density, strength_stats,
                             methods, segments, onset_data):
    """Generate interpretation."""
    lines = []

    rate = n_onsets / (duration + 1e-10)
    if rate > 10:
        lines.append(f"Very high onset rate ({rate:.1f}/sec, {n_onsets} total) — dense, busy signal with frequent transients.")
    elif rate > 4:
        lines.append(f"High onset rate ({rate:.1f}/sec, {n_onsets} total) — active signal with many events.")
    elif rate > 1:
        lines.append(f"Moderate onset rate ({rate:.1f}/sec, {n_onsets} total).")
    else:
        lines.append(f"Low onset rate ({rate:.1f}/sec, {n_onsets} total) — sparse events or sustained content.")

    # Regularity
    reg = ioi['regularity']
    if reg > 0.8:
        lines.append(f"Very regular onset timing (regularity: {reg:.3f}) — rhythmic or periodic events.")
    elif reg > 0.5:
        lines.append(f"Moderately regular timing (regularity: {reg:.3f}).")
    else:
        lines.append(f"Irregular onset timing (regularity: {reg:.3f}) — varied event spacing.")

    # IOI
    if ioi['mean'] > 0:
        lines.append(f"Mean inter-onset interval: {ioi['mean']*1000:.0f} ms (range: {ioi['min']*1000:.0f}–{ioi['max']*1000:.0f} ms).")

    # Strength
    sr = strength_stats['strength_ratio']
    lines.append(f"Onset strength ratio: {sr:.2f}× — onsets are {sr:.1f}× stronger than average envelope.")

    # Density variation
    if density['std_density'] > density['mean_density'] * 0.5:
        lines.append(f"High density variation (std: {density['std_density']:.1f}) — onset activity varies significantly across time.")
    else:
        lines.append(f"Consistent onset density across time.")

    # Method comparison
    method_counts = {k: v['n_onsets'] for k, v in methods.items()}
    most = max(method_counts, key=method_counts.get)
    least = min(method_counts, key=method_counts.get)
    lines.append(f"Detection method comparison: {most} found most onsets ({method_counts[most]}), {least} found fewest ({method_counts[least]}).")

    # Rise time
    if onset_data:
        rise_times = [o['rise_time_ms'] for o in onset_data if o['rise_time_ms'] > 0]
        if rise_times:
            mean_rise = np.mean(rise_times)
            if mean_rise < 10:
                lines.append(f"Short mean rise time ({mean_rise:.1f} ms) — sharp, percussive attacks.")
            elif mean_rise < 50:
                lines.append(f"Moderate mean rise time ({mean_rise:.1f} ms).")
            else:
                lines.append(f"Long mean rise time ({mean_rise:.1f} ms) — gradual, soft attacks.")

    return {
        'summary': ' '.join(lines[:2]),
        'details': lines,
    }


# ──────────────────────────────────────────────
# Plot Generation
# ──────────────────────────────────────────────

def generate_onset_plots(
    y, sr, onset_times, onset_frames, onset_env, onset_env_times,
    methods, n_method_frames, ioi, density, onset_data, strength_stats,
    hop_length,
) -> str:

    fig = plt.figure(figsize=(18, 32))
    fig.suptitle('Onset Detection Analysis', fontsize=18, fontweight='bold')
    gs = gridspec.GridSpec(8, 2, figure=fig, hspace=0.55, wspace=0.3)

    # 1. Waveform + Onsets
    ax1 = fig.add_subplot(gs[0, :])
    t = np.arange(len(y)) / sr
    ax1.plot(t, y, color='#90CAF9', linewidth=0.3, alpha=0.5)
    for ot in onset_times:
        ax1.axvline(x=ot, color='#F44336', linewidth=0.5, alpha=0.6)
    ax1.set_title(f'Waveform with Onset Markers ({len(onset_times)} onsets)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Amplitude')
    ax1.grid(True, linestyle='--', alpha=0.4)

    # 2. Onset Strength Envelope + Detected Peaks
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(onset_env_times, onset_env, color='#FF5722', linewidth=0.7)
    ax2.fill_between(onset_env_times, onset_env, alpha=0.1, color='#FF5722')
    valid_frames = onset_frames[onset_frames < len(onset_env)]
    if len(valid_frames) > 0:
        ax2.scatter(onset_env_times[valid_frames], onset_env[valid_frames],
                    color='#F44336', s=15, zorder=5, edgecolors='white', linewidth=0.3)
    ax2.set_title('Onset Strength Envelope with Detected Peaks', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Onset Strength')
    ax2.grid(True, linestyle='--', alpha=0.4)

    # 3. Multi-method Comparison
    ax3 = fig.add_subplot(gs[2, :])
    m_times = onset_env_times[:n_method_frames]
    colors_m = {'default': '#2196F3', 'energy': '#4CAF50', 'spectral_flux': '#FF9800', 'hfc': '#9C27B0'}
    for key, data in methods.items():
        env = data['envelope'][:n_method_frames]
        ax3.plot(m_times[:len(env)], env, color=colors_m.get(key, '#607D8B'),
                 linewidth=0.8, alpha=0.7, label=f'{key} ({data["n_onsets"]})')
    ax3.set_title('Onset Detection Methods Comparison', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Time (s)'); ax3.set_ylabel('Strength (normalized)')
    ax3.legend(fontsize=8); ax3.grid(True, linestyle='--', alpha=0.4)

    # 4. IOI over Time
    ax4 = fig.add_subplot(gs[3, 0])
    if ioi['iois']:
        ioi_times = [onset_times[i + 1] for i in range(len(ioi['iois']))]
        ax4.plot(ioi_times, [x * 1000 for x in ioi['iois']], 'o-', color='#9C27B0',
                 markersize=3, linewidth=0.8)
        ax4.axhline(y=ioi['mean'] * 1000, color='#F44336', linestyle='--', linewidth=0.8,
                     label=f'Mean: {ioi["mean"]*1000:.0f} ms')
        ax4.legend(fontsize=8)
    ax4.set_title('Inter-Onset Interval (IOI)', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Time (s)'); ax4.set_ylabel('IOI (ms)')
    ax4.grid(True, linestyle='--', alpha=0.4)

    # 5. IOI Histogram
    ax5 = fig.add_subplot(gs[3, 1])
    if ioi['iois']:
        ax5.hist([x * 1000 for x in ioi['iois']], bins=40, color='#9C27B0',
                 edgecolor='white', linewidth=0.3, alpha=0.85)
        ax5.axvline(x=ioi['mean'] * 1000, color='#F44336', linestyle='--', linewidth=1,
                     label=f'Mean: {ioi["mean"]*1000:.0f} ms')
        ax5.axvline(x=ioi['median'] * 1000, color='#4CAF50', linestyle='--', linewidth=1,
                     label=f'Median: {ioi["median"]*1000:.0f} ms')
        ax5.legend(fontsize=8)
    ax5.set_title('IOI Distribution', fontsize=13, fontweight='bold')
    ax5.set_xlabel('IOI (ms)'); ax5.set_ylabel('Count')
    ax5.grid(True, linestyle='--', alpha=0.4, axis='y')

    # 6. Onset Density
    ax6 = fig.add_subplot(gs[4, 0])
    if density['bins']:
        ax6.bar(density['bins'], density['density'], width=0.9, color='#4CAF50',
                edgecolor='white', linewidth=0.3, alpha=0.85)
        ax6.axhline(y=density['mean_density'], color='#F44336', linestyle='--', linewidth=0.8,
                     label=f'Mean: {density["mean_density"]:.1f}/sec')
        ax6.legend(fontsize=8)
    ax6.set_title('Onset Density (per second)', fontsize=13, fontweight='bold')
    ax6.set_xlabel('Time (s)'); ax6.set_ylabel('Onsets')
    ax6.grid(True, linestyle='--', alpha=0.4, axis='y')

    # 7. Onset Strength Distribution
    ax7 = fig.add_subplot(gs[4, 1])
    if onset_data:
        strengths = [o['strength'] for o in onset_data]
        ax7.hist(strengths, bins=40, color='#FF9800', edgecolor='white', linewidth=0.3, alpha=0.85)
        ax7.axvline(x=strength_stats['onset_mean_strength'], color='#F44336', linestyle='--', linewidth=1,
                     label=f'Mean: {strength_stats["onset_mean_strength"]:.3f}')
        ax7.legend(fontsize=8)
    ax7.set_title('Onset Strength Distribution', fontsize=13, fontweight='bold')
    ax7.set_xlabel('Strength'); ax7.set_ylabel('Count')
    ax7.grid(True, linestyle='--', alpha=0.4, axis='y')

    # 8. Rise Time Distribution
    ax8 = fig.add_subplot(gs[5, 0])
    if onset_data:
        rise_times = [o['rise_time_ms'] for o in onset_data if o['rise_time_ms'] > 0]
        if rise_times:
            ax8.hist(rise_times, bins=30, color='#00BCD4', edgecolor='white', linewidth=0.3, alpha=0.85)
            ax8.axvline(x=np.mean(rise_times), color='#F44336', linestyle='--', linewidth=1,
                         label=f'Mean: {np.mean(rise_times):.1f} ms')
            ax8.legend(fontsize=8)
    ax8.set_title('Onset Rise Time Distribution', fontsize=13, fontweight='bold')
    ax8.set_xlabel('Rise Time (ms)'); ax8.set_ylabel('Count')
    ax8.grid(True, linestyle='--', alpha=0.4, axis='y')

    # 9. Local RMS at Onsets
    ax9 = fig.add_subplot(gs[5, 1])
    if onset_data:
        rms_vals = [o['local_rms_db'] for o in onset_data]
        ax9.bar(range(min(100, len(rms_vals))), rms_vals[:100], color='#2196F3',
                edgecolor='white', linewidth=0.2, alpha=0.85)
    ax9.set_title('Local RMS at Onsets (dB)', fontsize=13, fontweight='bold')
    ax9.set_xlabel('Onset #'); ax9.set_ylabel('RMS (dB)')
    ax9.grid(True, linestyle='--', alpha=0.4, axis='y')

    # 10. Method Onset Counts
    ax10 = fig.add_subplot(gs[6, 0])
    m_names = list(methods.keys())
    m_counts = [methods[k]['n_onsets'] for k in m_names]
    m_colors = [colors_m.get(k, '#607D8B') for k in m_names]
    bars10 = ax10.bar(m_names, m_counts, color=m_colors, edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars10, m_counts):
        ax10.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 str(val), ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax10.set_title('Detection Method Comparison', fontsize=13, fontweight='bold')
    ax10.set_ylabel('Onsets Detected')
    ax10.grid(True, linestyle='--', alpha=0.4, axis='y')

    # 11. Key Metrics
    ax11 = fig.add_subplot(gs[6, 1])
    kl = ['Onsets', 'Rate\n(/sec)', 'Mean IOI\n(ms)', 'Regularity', 'Mean\nStrength']
    kv = [len(onset_times) / 10, density['global_rate'],
          ioi['mean'] * 100, ioi['regularity'], strength_stats['onset_mean_strength']]
    kc = ['#F44336', '#2196F3', '#9C27B0', '#4CAF50', '#FF9800']
    disp = [str(len(onset_times)), f'{density["global_rate"]:.1f}',
            f'{ioi["mean"]*1000:.0f}', f'{ioi["regularity"]:.3f}', f'{strength_stats["onset_mean_strength"]:.3f}']
    bars11 = ax11.bar(kl, kv, color=kc, edgecolor='white', linewidth=0.8)
    for bar, d in zip(bars11, disp):
        ax11.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 d, ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax11.set_title('Key Metrics', fontsize=13, fontweight='bold')
    ax11.grid(True, linestyle='--', alpha=0.4, axis='y')

    # 12. Zoomed view
    ax12 = fig.add_subplot(gs[7, :])
    zoom_end = min(3.0, len(y) / sr)
    mask = t <= zoom_end
    ax12.plot(t[mask], y[mask], color='#90CAF9', linewidth=0.5, alpha=0.7)
    for ot in onset_times:
        if ot <= zoom_end:
            ax12.axvline(x=ot, color='#F44336', linewidth=1.5, alpha=0.8)
    ax12.set_title(f'Zoomed View (0–{zoom_end:.1f}s)', fontsize=13, fontweight='bold')
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

@router.post("/onset-detection")
async def onset_detection(
    file: UploadFile = File(...),
    hop_length: Optional[int] = Form(512),
    delta: Optional[float] = Form(0.07),
    wait: Optional[int] = Form(1),
    backtrack: Optional[bool] = Form(True),
    n_segments: Optional[int] = Form(10),
    target_sr: Optional[int] = Form(None),
):
    """
    Onset Detection Analysis.

    Comprehensive onset/transient detection including multiple detection
    methods, per-onset characterization, IOI analysis, density profiling,
    rise time, and detection method comparison.
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
            if len(y) > sr * 120: y = y[:sr * 120]

            hl = hop_length or 512
            d = delta if delta is not None else 0.07
            w = wait if wait is not None else 1
            bt = backtrack if backtrack is not None else True
            n_seg = n_segments or 10

            duration = len(y) / sr

            # Onset detection
            onset_frames, onset_times, onset_env = detect_onsets(y, sr, hl, bt, d, w)
            onset_env_full, onset_env_times = compute_onset_envelope(y, sr, hl)

            # Multi-method comparison
            methods, n_method_frames = compute_multi_method_onsets(y, sr, hl)

            # Characterize onsets
            onset_data = compute_onset_characterization(onset_frames, onset_times, onset_env, y, sr, hl)

            # IOI
            ioi = compute_ioi_analysis(onset_times)

            # Density
            density = compute_onset_density(onset_times, duration)

            # Segments
            segments = compute_onset_segments(onset_times, duration, n_seg)

            # Strength statistics
            strength_stats = compute_strength_statistics(onset_env, onset_frames)

            # Interpretation
            interpretation = generate_interpretation(
                len(onset_times), duration, ioi, density, strength_stats,
                methods, segments, onset_data
            )

            # Plot
            plot = generate_onset_plots(
                y, sr, onset_times, onset_frames, onset_env, onset_env_times,
                methods, n_method_frames, ioi, density, onset_data, strength_stats, hl
            )

            file_info = {
                'filename': file.filename, 'file_size_mb': round(len(content)/(1024*1024), 2),
                'sample_rate': sr, 'duration_sec': round(duration, 2),
                'n_samples': len(y), 'format': ext.replace('.',''),
                'hop_length': hl, 'delta': d, 'wait': w, 'backtrack': bt,
                'n_segments': n_seg,
            }

            # Methods summary for JSON
            methods_summary = {k: {'n_onsets': v['n_onsets']} for k, v in methods.items()}

            return _to_native({
                'results': {
                    'file_info': file_info,
                    'n_onsets': len(onset_times),
                    'onset_rate': safe_float(len(onset_times) / duration),
                    'ioi_analysis': {k: v for k, v in ioi.items() if k != 'iois'},
                    'density': {k: v for k, v in density.items() if k != 'bins' and k != 'density'},
                    'strength_stats': strength_stats,
                    'methods_comparison': methods_summary,
                    'segments': segments,
                    'onset_table': onset_data[:200],
                    'interpretation': interpretation,
                },
                'plot': plot
            })
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ──────────────────────────────────────────────
# from onset_detection import router as onset_router
# app.include_router(onset_router, prefix="/api/analysis", tags=["Audio Analysis"])
# → POST /api/analysis/onset-detection
# ──────────────────────────────────────────────
