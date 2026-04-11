"""
STFT (Short-Time Fourier Transform) Analysis Backend (FastAPI)
- Time-frequency analysis of audio signals
- Spectrogram generation with multiple scales (linear, log, mel)
- Temporal feature extraction: spectral centroid, bandwidth, rolloff over time
- Onset detection, temporal energy envelope, frequency tracking
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
# STFT Analysis Functions
# ──────────────────────────────────────────────

def compute_stft(y: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512, window: str = 'hann'):
    """Compute STFT and return magnitude, phase, and time/frequency axes."""
    stft_complex = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window=window)
    magnitude = np.abs(stft_complex)
    phase = np.angle(stft_complex)
    power = magnitude ** 2
    db = librosa.amplitude_to_db(magnitude, ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    times = librosa.frames_to_time(np.arange(magnitude.shape[1]), sr=sr, hop_length=hop_length)
    return stft_complex, magnitude, phase, power, db, freqs, times


def compute_mel_spectrogram(y: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512, n_mels: int = 128):
    """Compute mel-scaled spectrogram."""
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec, mel_db


def compute_temporal_features(y: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512):
    """Compute time-varying spectral features."""
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, roll_percent=0.85)[0]
    spectral_flatness = librosa.feature.spectral_flatness(y=y, n_fft=n_fft, hop_length=hop_length)[0]
    rms = librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop_length)[0]
    zcr = librosa.feature.zero_crossing_rate(y=y, frame_length=n_fft, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(spectral_centroid)), sr=sr, hop_length=hop_length)

    return {
        'times': times,
        'spectral_centroid': spectral_centroid,
        'spectral_bandwidth': spectral_bandwidth,
        'spectral_rolloff': spectral_rolloff,
        'spectral_flatness': spectral_flatness,
        'rms': rms,
        'zcr': zcr,
    }


def detect_onsets(y: np.ndarray, sr: int, hop_length: int = 512):
    """Detect onset events in the signal."""
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop_length, onset_envelope=onset_env)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    return {
        'onset_envelope': onset_env,
        'onset_frames': onset_frames,
        'onset_times': onset_times,
        'n_onsets': len(onset_frames),
    }


def compute_time_frequency_stats(magnitude: np.ndarray, freqs: np.ndarray, times: np.ndarray):
    """Compute summary statistics across time-frequency representation."""
    # Average spectrum (mean across time)
    mean_spectrum = np.mean(magnitude, axis=1)
    std_spectrum = np.std(magnitude, axis=1)

    # Energy envelope (sum across frequency at each time frame)
    energy_envelope = np.sum(magnitude ** 2, axis=0)
    energy_envelope_db = 10 * np.log10(energy_envelope + 1e-10)

    # Frequency with max energy at each time frame
    peak_freq_over_time = freqs[np.argmax(magnitude, axis=0)]

    # Spectral flux (frame-to-frame spectral change)
    spectral_flux = np.sqrt(np.sum(np.diff(magnitude, axis=1) ** 2, axis=0))
    spectral_flux = np.concatenate([[0], spectral_flux])

    return {
        'mean_spectrum': mean_spectrum,
        'std_spectrum': std_spectrum,
        'energy_envelope': energy_envelope,
        'energy_envelope_db': energy_envelope_db,
        'peak_freq_over_time': peak_freq_over_time,
        'spectral_flux': spectral_flux,
    }


def compute_stft_summary(magnitude, freqs, times, temporal_features, onsets, sr, n_fft, hop_length):
    """Compute summary statistics for the STFT analysis."""
    energy_envelope = np.sum(magnitude ** 2, axis=0)
    mean_spectrum = np.mean(magnitude, axis=1)

    # Time-averaged spectral stats
    total_mag = np.sum(mean_spectrum) + 1e-10
    prob = mean_spectrum / total_mag
    avg_centroid = np.sum(freqs * prob)
    avg_spread = np.sqrt(np.sum(((freqs - avg_centroid) ** 2) * prob))
    avg_flatness = np.exp(np.mean(np.log(mean_spectrum + 1e-10))) / (np.mean(mean_spectrum) + 1e-10)

    # Temporal variability
    centroid_std = float(np.std(temporal_features['spectral_centroid']))
    rms_std = float(np.std(temporal_features['rms']))
    energy_range_db = float(10 * np.log10((np.max(energy_envelope) + 1e-10) / (np.min(energy_envelope) + 1e-10)))

    # Stationarity measure (mean spectral flux / mean energy)
    spectral_flux = np.sqrt(np.sum(np.diff(magnitude, axis=1) ** 2, axis=0))
    stationarity = 1.0 - min(1.0, float(np.mean(spectral_flux) / (np.mean(energy_envelope) + 1e-10)))

    return {
        'avg_spectral_centroid_hz': safe_float(avg_centroid),
        'avg_spectral_spread_hz': safe_float(avg_spread),
        'avg_spectral_flatness': safe_float(avg_flatness),
        'centroid_variability_hz': safe_float(centroid_std),
        'rms_variability': safe_float(rms_std),
        'energy_range_db': safe_float(energy_range_db),
        'stationarity_index': safe_float(stationarity),
        'n_onsets': onsets['n_onsets'],
        'onset_rate_per_sec': safe_float(onsets['n_onsets'] / (times[-1] + 1e-10)) if len(times) > 0 else 0,
        'n_time_frames': int(magnitude.shape[1]),
        'n_frequency_bins': int(magnitude.shape[0]),
        'time_resolution_ms': safe_float(hop_length / sr * 1000),
        'frequency_resolution_hz': safe_float(sr / n_fft),
    }


def generate_interpretation(summary, temporal_features, onsets):
    """Generate human-readable interpretation of STFT results."""
    lines = []

    # Stationarity
    si = summary['stationarity_index']
    if si > 0.8:
        lines.append("Signal is highly stationary — frequency content is consistent over time.")
    elif si > 0.5:
        lines.append("Signal shows moderate temporal variation in its spectral content.")
    else:
        lines.append("Signal is non-stationary — frequency content changes significantly over time.")

    # Centroid variability
    cv = summary['centroid_variability_hz']
    if cv > 500:
        lines.append(f"High spectral centroid variability ({cv:.0f} Hz std) suggests frequency sweeps or transient events.")
    elif cv > 100:
        lines.append(f"Moderate centroid variability ({cv:.0f} Hz std) indicates some spectral dynamics.")
    else:
        lines.append(f"Low centroid variability ({cv:.0f} Hz std) — tonal character is stable.")

    # Energy dynamics
    er = summary['energy_range_db']
    if er > 30:
        lines.append(f"Wide dynamic range ({er:.1f} dB) — signal has significant amplitude variations.")
    elif er > 15:
        lines.append(f"Moderate dynamic range ({er:.1f} dB) — some amplitude fluctuation present.")
    else:
        lines.append(f"Narrow dynamic range ({er:.1f} dB) — relatively constant energy level.")

    # Onsets
    n_onsets = onsets['n_onsets']
    if n_onsets > 20:
        lines.append(f"Detected {n_onsets} onsets — signal is highly transient/percussive.")
    elif n_onsets > 5:
        lines.append(f"Detected {n_onsets} onset events — intermittent transient activity.")
    elif n_onsets > 0:
        lines.append(f"Detected {n_onsets} onset(s) — relatively few transient events.")
    else:
        lines.append("No onsets detected — signal appears continuous and smooth.")

    # Flatness
    fl = summary['avg_spectral_flatness']
    if fl > 0.5:
        lines.append("Average spectrum is noise-like (high flatness).")
    elif fl > 0.1:
        lines.append("Average spectrum has mixed tonal and noise components.")
    else:
        lines.append("Average spectrum is highly tonal — clear dominant frequencies.")

    return {
        'summary': ' '.join(lines[:2]),
        'details': lines,
    }


def build_temporal_table(temporal_features, n_points=200):
    """Build a sampled table of temporal features for frontend display."""
    times = temporal_features['times']
    n = len(times)
    step = max(1, n // n_points)
    table = []
    for i in range(0, n, step):
        table.append({
            'time_sec': safe_float(times[i]),
            'spectral_centroid_hz': safe_float(temporal_features['spectral_centroid'][i]),
            'spectral_bandwidth_hz': safe_float(temporal_features['spectral_bandwidth'][i]),
            'spectral_rolloff_hz': safe_float(temporal_features['spectral_rolloff'][i]),
            'spectral_flatness': safe_float(temporal_features['spectral_flatness'][i]),
            'rms': safe_float(temporal_features['rms'][i]),
            'zcr': safe_float(temporal_features['zcr'][i]),
        })
    return table


# ──────────────────────────────────────────────
# Plot Generation
# ──────────────────────────────────────────────

def generate_stft_plots(
    y: np.ndarray, sr: int,
    stft_db: np.ndarray, mel_db: np.ndarray,
    freqs: np.ndarray, times: np.ndarray,
    temporal_features: dict, tf_stats: dict,
    onsets: dict, max_display_freq: float,
    n_fft: int, hop_length: int,
) -> str:
    """Generate comprehensive STFT visualization."""

    fig = plt.figure(figsize=(18, 24))
    fig.suptitle('STFT Time-Frequency Analysis', fontsize=18, fontweight='bold')
    gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.5, wspace=0.3)

    freq_mask = freqs <= max_display_freq
    freq_idx = np.sum(freq_mask)

    # ── 1. Waveform with onsets ──
    ax1 = fig.add_subplot(gs[0, :])
    t = np.arange(len(y)) / sr
    ax1.plot(t, y, color='#2196F3', linewidth=0.4, alpha=0.8)
    for ot in onsets['onset_times']:
        ax1.axvline(x=ot, color='#F44336', linestyle='--', alpha=0.5, linewidth=0.8)
    ax1.set_title(f'Waveform with Onset Detection ({onsets["n_onsets"]} onsets)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, linestyle='--', alpha=0.4)

    # ── 2. STFT Spectrogram (linear frequency) ──
    ax2 = fig.add_subplot(gs[1, 0])
    img2 = ax2.pcolormesh(times, freqs[:freq_idx], stft_db[:freq_idx, :],
                          shading='gouraud', cmap='magma')
    ax2.set_title('STFT Spectrogram (Linear Freq)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Frequency (Hz)')
    fig.colorbar(img2, ax=ax2, label='dB', pad=0.02)

    # ── 3. STFT Spectrogram (log frequency) ──
    ax3 = fig.add_subplot(gs[1, 1])
    log_freqs = freqs[:freq_idx]
    log_mask = log_freqs > 0
    if np.any(log_mask):
        img3 = ax3.pcolormesh(times, log_freqs[log_mask], stft_db[:freq_idx, :][log_mask, :],
                              shading='gouraud', cmap='magma')
        ax3.set_yscale('symlog', linthresh=100)
        fig.colorbar(img3, ax=ax3, label='dB', pad=0.02)
    ax3.set_title('STFT Spectrogram (Log Freq)', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Frequency (Hz)')

    # ── 4. Mel Spectrogram ──
    ax4 = fig.add_subplot(gs[2, 0])
    img4 = librosa.display.specshow(mel_db, x_axis='time', y_axis='mel', sr=sr,
                                     hop_length=hop_length, ax=ax4, cmap='magma')
    ax4.set_title('Mel Spectrogram', fontsize=13, fontweight='bold')
    fig.colorbar(img4, ax=ax4, label='dB', pad=0.02)

    # ── 5. Spectral Centroid + Bandwidth over Time ──
    ax5 = fig.add_subplot(gs[2, 1])
    tf = temporal_features
    tf_times = tf['times']
    ax5.plot(tf_times, tf['spectral_centroid'], color='#FF5722', linewidth=1.0, label='Centroid')
    ax5.fill_between(tf_times,
                     tf['spectral_centroid'] - tf['spectral_bandwidth'] / 2,
                     tf['spectral_centroid'] + tf['spectral_bandwidth'] / 2,
                     alpha=0.15, color='#FF5722', label='Bandwidth')
    ax5.plot(tf_times, tf['spectral_rolloff'], color='#4CAF50', linewidth=0.8, alpha=0.7, label='Rolloff (85%)')
    ax5.set_title('Spectral Centroid & Bandwidth', fontsize=13, fontweight='bold')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('Frequency (Hz)')
    ax5.legend(loc='upper right', fontsize=8)
    ax5.grid(True, linestyle='--', alpha=0.4)

    # ── 6. Energy Envelope + Spectral Flux ──
    ax6 = fig.add_subplot(gs[3, 0])
    ax6.plot(times, tf_stats['energy_envelope_db'], color='#00BCD4', linewidth=1.0, label='Energy (dB)')
    ax6.set_title('Energy Envelope', fontsize=13, fontweight='bold')
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('Energy (dB)')
    ax6.grid(True, linestyle='--', alpha=0.4)
    ax6_twin = ax6.twinx()
    ax6_twin.plot(times, tf_stats['spectral_flux'], color='#FF9800', linewidth=0.8, alpha=0.7, label='Spectral Flux')
    ax6_twin.set_ylabel('Spectral Flux', color='#FF9800')
    ax6_twin.tick_params(axis='y', labelcolor='#FF9800')
    lines1, labels1 = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6_twin.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)

    # ── 7. Peak Frequency over Time ──
    ax7 = fig.add_subplot(gs[3, 1])
    ax7.scatter(times, tf_stats['peak_freq_over_time'], s=2, c=tf_stats['energy_envelope_db'],
                cmap='viridis', alpha=0.6)
    ax7.set_title('Peak Frequency Tracking', fontsize=13, fontweight='bold')
    ax7.set_xlabel('Time (s)')
    ax7.set_ylabel('Peak Frequency (Hz)')
    ax7.set_ylim(0, max_display_freq)
    ax7.grid(True, linestyle='--', alpha=0.4)
    cb7 = fig.colorbar(ax7.collections[0], ax=ax7, label='Energy (dB)', pad=0.02)

    # ── 8. RMS Energy + Zero Crossing Rate ──
    ax8 = fig.add_subplot(gs[4, 0])
    ax8.plot(tf_times, tf['rms'], color='#9C27B0', linewidth=1.0, label='RMS')
    ax8.set_title('RMS Energy', fontsize=13, fontweight='bold')
    ax8.set_xlabel('Time (s)')
    ax8.set_ylabel('RMS')
    ax8.grid(True, linestyle='--', alpha=0.4)

    # ── 9. Spectral Flatness over Time ──
    ax9 = fig.add_subplot(gs[4, 1])
    ax9.plot(tf_times, tf['spectral_flatness'], color='#607D8B', linewidth=1.0)
    ax9.fill_between(tf_times, tf['spectral_flatness'], alpha=0.15, color='#607D8B')
    ax9.set_title('Spectral Flatness over Time', fontsize=13, fontweight='bold')
    ax9.set_xlabel('Time (s)')
    ax9.set_ylabel('Flatness')
    ax9.set_ylim(0, 1)
    ax9.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"


# ──────────────────────────────────────────────
# Main Endpoint
# ──────────────────────────────────────────────

@router.post("/stft-analysis")
async def stft_analysis(
    file: UploadFile = File(...),
    max_frequency: Optional[float] = Form(None),
    n_fft: Optional[int] = Form(2048),
    hop_length: Optional[int] = Form(512),
    window: Optional[str] = Form('hann'),
    n_mels: Optional[int] = Form(128),
    target_sr: Optional[int] = Form(None),
):
    """
    STFT Time-Frequency Analysis.

    Performs short-time Fourier transform analysis on an audio file
    to reveal how frequency content evolves over time.

    Parameters:
    - file: Audio file (WAV, MP3, FLAC, etc.)
    - max_frequency: Max frequency to display (default: sr/2 or 10kHz)
    - n_fft: FFT window size (default: 2048)
    - hop_length: Hop length in samples (default: 512)
    - window: Window function — hann, hamming, blackman (default: hann)
    - n_mels: Number of mel bands (default: 128)
    - target_sr: Resample to this rate (default: keep original)

    Response:
    {
        "results": {
            "file_info": {...},
            "summary": {...},
            "temporal_features_table": [...],
            "onsets": {...},
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

            # Trim silence
            y_trimmed, _ = librosa.effects.trim(y, top_db=30)

            # Limit to 60s
            max_samples = sr * 60
            if len(y_trimmed) > max_samples:
                y_trimmed = y_trimmed[:max_samples]

            n_fft_val = n_fft if n_fft else 2048
            hop_val = hop_length if hop_length else 512
            win_fn = window if window else 'hann'
            n_mels_val = n_mels if n_mels else 128

            # Max display frequency
            max_display_freq = max_frequency if max_frequency else min(sr / 2, 10000)

            # Compute STFT
            stft_complex, magnitude, phase, power, stft_db, freqs, times = compute_stft(
                y_trimmed, sr, n_fft=n_fft_val, hop_length=hop_val, window=win_fn
            )

            # Mel spectrogram
            mel_spec, mel_db = compute_mel_spectrogram(
                y_trimmed, sr, n_fft=n_fft_val, hop_length=hop_val, n_mels=n_mels_val
            )

            # Temporal features
            temporal_features = compute_temporal_features(
                y_trimmed, sr, n_fft=n_fft_val, hop_length=hop_val
            )

            # Onset detection
            onsets = detect_onsets(y_trimmed, sr, hop_length=hop_val)

            # Time-frequency statistics
            tf_stats = compute_time_frequency_stats(magnitude, freqs, times)

            # Summary statistics
            summary = compute_stft_summary(
                magnitude, freqs, times, temporal_features, onsets, sr, n_fft_val, hop_val
            )

            # Interpretation
            interpretation = generate_interpretation(summary, temporal_features, onsets)

            # Temporal features table (sampled)
            temporal_table = build_temporal_table(temporal_features, n_points=200)

            # Onset times list
            onset_data = {
                'n_onsets': int(onsets['n_onsets']),
                'onset_times': [safe_float(t) for t in onsets['onset_times'][:50]],
            }

            # Generate plots
            plot = generate_stft_plots(
                y_trimmed, sr, stft_db, mel_db, freqs, times,
                temporal_features, tf_stats, onsets, max_display_freq,
                n_fft_val, hop_val
            )

            # File info
            file_info = {
                'filename': file.filename,
                'file_size_mb': round(len(content) / (1024 * 1024), 2),
                'sample_rate': sr,
                'duration_sec': round(len(y_trimmed) / sr, 2),
                'n_samples': len(y_trimmed),
                'format': file_ext.replace('.', ''),
                'n_fft': n_fft_val,
                'hop_length': hop_val,
                'window_function': win_fn,
                'n_mels': n_mels_val,
                'time_resolution_ms': safe_float(hop_val / sr * 1000),
                'frequency_resolution_hz': safe_float(sr / n_fft_val),
                'max_display_frequency': safe_float(max_display_freq),
                'n_time_frames': int(magnitude.shape[1]),
                'n_frequency_bins': int(magnitude.shape[0]),
            }

            return _to_native({
                'results': {
                    'file_info': file_info,
                    'summary': summary,
                    'temporal_features_table': temporal_table,
                    'onsets': onset_data,
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
# from stft_analysis import router as stft_router
# app.include_router(stft_router, prefix="/api/analysis", tags=["Audio Analysis"])
# → POST /api/analysis/stft-analysis
# ──────────────────────────────────────────────
