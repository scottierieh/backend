"""
FFT Spectrum Analysis Backend (FastAPI)
- Deep frequency-domain analysis of audio signals
- Multiple FFT views: Linear, Log, Band Energy, Harmonic, Waterfall
- Peak detection, harmonic analysis, band energy breakdown
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
# FFT Analysis Functions
# ──────────────────────────────────────────────

def compute_fft(y: np.ndarray, sr: int):
    """Compute FFT magnitude and frequency bins."""
    n = len(y)
    fft_complex = np.fft.rfft(y)
    fft_magnitude = np.abs(fft_complex) * 2 / n  # normalized
    fft_power = fft_magnitude ** 2
    fft_db = 20 * np.log10(fft_magnitude + 1e-10)  # dB scale
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    return freqs, fft_magnitude, fft_power, fft_db


def find_peaks(freqs: np.ndarray, magnitude: np.ndarray, n_peaks: int = 10, min_distance_hz: float = 20.0):
    """Find top N frequency peaks with minimum distance between them."""
    peaks = []
    mag_copy = magnitude.copy()
    freq_resolution = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    min_distance_bins = max(1, int(min_distance_hz / freq_resolution))

    for _ in range(n_peaks):
        if np.max(mag_copy) <= 0:
            break
        idx = np.argmax(mag_copy)
        peaks.append({
            'frequency_hz': safe_float(freqs[idx]),
            'magnitude': safe_float(magnitude[idx]),
            'power': safe_float(magnitude[idx] ** 2),
            'db': safe_float(20 * np.log10(magnitude[idx] + 1e-10)),
            'rank': len(peaks) + 1,
        })
        # Zero out neighborhood
        start = max(0, idx - min_distance_bins)
        end = min(len(mag_copy), idx + min_distance_bins + 1)
        mag_copy[start:end] = 0

    return peaks


def analyze_harmonics(peaks: list, tolerance_ratio: float = 0.05):
    """Detect harmonic relationships between peaks."""
    if len(peaks) < 2:
        return {'fundamental_hz': peaks[0]['frequency_hz'] if peaks else 0, 'harmonics': [], 'is_harmonic': False}

    fundamental = peaks[0]['frequency_hz']
    if fundamental <= 0:
        return {'fundamental_hz': 0, 'harmonics': [], 'is_harmonic': False}

    harmonics = []
    for peak in peaks[1:]:
        ratio = peak['frequency_hz'] / fundamental
        nearest_int = round(ratio)
        if nearest_int >= 2 and abs(ratio - nearest_int) / nearest_int < tolerance_ratio:
            harmonics.append({
                'frequency_hz': peak['frequency_hz'],
                'harmonic_number': int(nearest_int),
                'magnitude': peak['magnitude'],
                'deviation_percent': safe_float(abs(ratio - nearest_int) / nearest_int * 100),
            })

    return {
        'fundamental_hz': safe_float(fundamental),
        'harmonics': harmonics,
        'n_harmonics': len(harmonics),
        'is_harmonic': len(harmonics) >= 2,
        'total_harmonic_distortion': safe_float(
            np.sqrt(sum(h['magnitude'] ** 2 for h in harmonics)) / (peaks[0]['magnitude'] + 1e-10) * 100
        ) if harmonics else 0,
    }


def compute_band_energy(freqs: np.ndarray, power: np.ndarray, sr: int):
    """Compute energy in standard frequency bands."""
    bands = [
        ('Sub-bass', 0, 60),
        ('Bass', 60, 250),
        ('Low-mid', 250, 500),
        ('Mid', 500, 2000),
        ('Upper-mid', 2000, 4000),
        ('High', 4000, 8000),
        ('Very High', 8000, min(20000, sr // 2)),
    ]

    total_energy = np.sum(power)
    band_results = []

    for name, f_low, f_high in bands:
        if f_low >= sr // 2:
            continue
        f_high = min(f_high, sr // 2)
        mask = (freqs >= f_low) & (freqs < f_high)
        band_energy = np.sum(power[mask])
        band_results.append({
            'band': name,
            'range_hz': f'{f_low}-{f_high}',
            'energy': safe_float(band_energy),
            'energy_percent': safe_float(band_energy / (total_energy + 1e-10) * 100),
            'peak_freq_hz': safe_float(freqs[mask][np.argmax(power[mask])] if np.any(mask) and np.sum(power[mask]) > 0 else 0),
            'mean_magnitude': safe_float(np.mean(np.sqrt(power[mask])) if np.any(mask) else 0),
        })

    dominant = max(band_results, key=lambda x: x['energy']) if band_results else None

    return {
        'bands': band_results,
        'total_energy': safe_float(total_energy),
        'dominant_band': dominant['band'] if dominant else 'N/A',
        'dominant_band_percent': safe_float(dominant['energy_percent']) if dominant else 0,
    }


def compute_spectral_stats(freqs: np.ndarray, magnitude: np.ndarray):
    """Compute statistical measures of the spectrum."""
    total = np.sum(magnitude) + 1e-10
    prob = magnitude / total

    centroid = np.sum(freqs * prob)
    spread = np.sqrt(np.sum(((freqs - centroid) ** 2) * prob))
    skewness = np.sum(((freqs - centroid) ** 3) * prob) / (spread ** 3 + 1e-10)
    kurtosis = np.sum(((freqs - centroid) ** 4) * prob) / (spread ** 4 + 1e-10)
    flatness = np.exp(np.mean(np.log(magnitude + 1e-10))) / (np.mean(magnitude) + 1e-10)
    entropy = -np.sum(prob * np.log2(prob + 1e-15))

    # Bandwidth (where 90% energy is concentrated)
    cumulative = np.cumsum(magnitude ** 2)
    total_power = cumulative[-1]
    low_idx = np.searchsorted(cumulative, total_power * 0.05)
    high_idx = np.searchsorted(cumulative, total_power * 0.95)
    bandwidth_90 = freqs[min(high_idx, len(freqs) - 1)] - freqs[min(low_idx, len(freqs) - 1)]

    return {
        'spectral_centroid_hz': safe_float(centroid),
        'spectral_spread_hz': safe_float(spread),
        'spectral_skewness': safe_float(skewness),
        'spectral_kurtosis': safe_float(kurtosis),
        'spectral_flatness': safe_float(flatness),
        'spectral_entropy': safe_float(entropy),
        'bandwidth_90_hz': safe_float(bandwidth_90),
    }


def generate_interpretation(peaks, harmonics, band_energy, spectral_stats):
    """Generate human-readable interpretation of FFT results."""
    lines = []

    # Dominant frequency
    if peaks:
        f = peaks[0]['frequency_hz']
        db = peaks[0]['db']
        if f < 100:
            desc = "very low frequency — likely a motor hum or structural vibration"
        elif f < 500:
            desc = "low-mid frequency — typical of rotating machinery"
        elif f < 2000:
            desc = "mid frequency — mechanical operation or resonance"
        elif f < 5000:
            desc = "high frequency — possible friction, bearing wear, or air leak"
        else:
            desc = "very high frequency — ultrasonic range, possible electrical discharge or leak"
        lines.append(f"Dominant frequency: {f:.1f} Hz ({db:.1f} dB) — {desc}.")

    # Harmonics
    if harmonics['is_harmonic']:
        lines.append(
            f"Harmonic structure detected: fundamental at {harmonics['fundamental_hz']:.1f} Hz "
            f"with {harmonics['n_harmonics']} harmonics. THD: {harmonics['total_harmonic_distortion']:.1f}%."
        )
    else:
        lines.append("No strong harmonic pattern detected — signal may be broadband or noisy.")

    # Band energy
    dom = band_energy['dominant_band']
    pct = band_energy['dominant_band_percent']
    lines.append(f"Energy concentrated in {dom} band ({pct:.1f}% of total).")

    # Spectral shape
    flatness = spectral_stats['spectral_flatness']
    if flatness > 0.5:
        lines.append("Spectrum is flat (noise-like). Broadband energy distribution.")
    elif flatness > 0.1:
        lines.append("Spectrum has mixed tonal and noise components.")
    else:
        lines.append("Spectrum is highly tonal — clear dominant frequencies.")

    return {
        'summary': ' '.join(lines[:2]),
        'details': lines,
    }


# ──────────────────────────────────────────────
# Plot Generation
# ──────────────────────────────────────────────

def generate_fft_plots(
    y: np.ndarray, sr: int,
    freqs: np.ndarray, magnitude: np.ndarray, power: np.ndarray, fft_db: np.ndarray,
    peaks: list, band_energy: dict, max_display_freq: float
) -> str:
    """Generate comprehensive FFT visualization."""

    fig = plt.figure(figsize=(18, 20))
    fig.suptitle('FFT Spectrum Analysis', fontsize=18, fontweight='bold')
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.3)

    freq_mask = freqs <= max_display_freq

    # ── 1. Waveform ──
    ax1 = fig.add_subplot(gs[0, :])
    time = np.arange(len(y)) / sr
    ax1.plot(time, y, color='#2196F3', linewidth=0.4, alpha=0.8)
    ax1.set_title('Time Domain — Waveform', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, linestyle='--', alpha=0.4)

    # ── 2. FFT Linear Scale ──
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(freqs[freq_mask], magnitude[freq_mask], color='#4CAF50', linewidth=0.8)
    ax2.fill_between(freqs[freq_mask], magnitude[freq_mask], alpha=0.2, color='#4CAF50')
    # Mark peaks
    for i, peak in enumerate(peaks[:5]):
        if peak['frequency_hz'] <= max_display_freq:
            ax2.axvline(x=peak['frequency_hz'], color='#F44336', linestyle='--', alpha=0.5, linewidth=0.8)
            ax2.annotate(f"{peak['frequency_hz']:.0f} Hz",
                        xy=(peak['frequency_hz'], peak['magnitude']),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color='#F44336', fontweight='bold')
    ax2.set_title('FFT Spectrum (Linear)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Magnitude')
    ax2.grid(True, linestyle='--', alpha=0.4)

    # ── 3. FFT dB Scale ──
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(freqs[freq_mask], fft_db[freq_mask], color='#9C27B0', linewidth=0.8)
    ax3.fill_between(freqs[freq_mask], fft_db[freq_mask], np.min(fft_db[freq_mask]), alpha=0.15, color='#9C27B0')
    for peak in peaks[:5]:
        if peak['frequency_hz'] <= max_display_freq:
            ax3.axvline(x=peak['frequency_hz'], color='#F44336', linestyle='--', alpha=0.5, linewidth=0.8)
    ax3.set_title('FFT Spectrum (dB Scale)', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Magnitude (dB)')
    ax3.grid(True, linestyle='--', alpha=0.4)

    # ── 4. FFT Log-Frequency Scale ──
    ax4 = fig.add_subplot(gs[2, 0])
    log_mask = (freqs > 1) & (freqs <= max_display_freq)
    ax4.semilogx(freqs[log_mask], fft_db[log_mask], color='#FF9800', linewidth=0.8)
    ax4.fill_between(freqs[log_mask], fft_db[log_mask], np.min(fft_db[log_mask]), alpha=0.15, color='#FF9800')
    ax4.set_title('FFT Spectrum (Log Frequency)', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Frequency (Hz) — Log Scale')
    ax4.set_ylabel('Magnitude (dB)')
    ax4.grid(True, linestyle='--', alpha=0.4, which='both')

    # ── 5. Band Energy Bar Chart ──
    ax5 = fig.add_subplot(gs[2, 1])
    bands = band_energy['bands']
    band_names = [b['band'] for b in bands]
    band_pcts = [b['energy_percent'] for b in bands]
    colors = ['#1a237e', '#283593', '#3949ab', '#5c6bc0', '#7986cb', '#9fa8da', '#c5cae9']
    colors = colors[:len(band_names)]
    bars = ax5.bar(band_names, band_pcts, color=colors, edgecolor='white', linewidth=0.8)
    ax5.set_title('Energy by Frequency Band', fontsize=13, fontweight='bold')
    ax5.set_ylabel('Energy (%)')
    ax5.tick_params(axis='x', rotation=35)
    ax5.grid(True, linestyle='--', alpha=0.4, axis='y')
    for bar, pct in zip(bars, band_pcts):
        if pct > 3:
            ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # ── 6. Power Spectral Density ──
    ax6 = fig.add_subplot(gs[3, 0])
    ax6.plot(freqs[freq_mask], 10 * np.log10(power[freq_mask] + 1e-20), color='#00BCD4', linewidth=0.8)
    ax6.fill_between(freqs[freq_mask], 10 * np.log10(power[freq_mask] + 1e-20),
                     np.min(10 * np.log10(power[freq_mask] + 1e-20)), alpha=0.15, color='#00BCD4')
    ax6.set_title('Power Spectral Density', fontsize=13, fontweight='bold')
    ax6.set_xlabel('Frequency (Hz)')
    ax6.set_ylabel('Power (dB)')
    ax6.grid(True, linestyle='--', alpha=0.4)

    # ── 7. Peak Frequency Chart ──
    ax7 = fig.add_subplot(gs[3, 1])
    if peaks:
        peak_freqs = [p['frequency_hz'] for p in peaks[:10]]
        peak_mags = [p['db'] for p in peaks[:10]]
        peak_labels = [f"{p['frequency_hz']:.0f} Hz" for p in peaks[:10]]
        colors_peak = ['#F44336' if i == 0 else '#FF9800' if i < 3 else '#4CAF50' for i in range(len(peaks[:10]))]
        bars = ax7.barh(peak_labels[::-1], peak_mags[::-1], color=colors_peak[::-1], edgecolor='white')
        ax7.set_title('Top Frequency Peaks (dB)', fontsize=13, fontweight='bold')
        ax7.set_xlabel('Magnitude (dB)')
        ax7.grid(True, linestyle='--', alpha=0.4, axis='x')

    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"


# ──────────────────────────────────────────────
# Main Endpoint
# ──────────────────────────────────────────────

@router.post("/fft-spectrum")
async def fft_spectrum_analysis(
    file: UploadFile = File(...),
    max_frequency: Optional[float] = Form(None),
    n_peaks: Optional[int] = Form(10),
    window: Optional[str] = Form('hann'),
    target_sr: Optional[int] = Form(None),
):
    """
    FFT Spectrum Analysis.

    Performs deep frequency-domain analysis on an audio file.

    Parameters:
    - file: Audio file (WAV, MP3, FLAC, etc.)
    - max_frequency: Max frequency to display (default: sr/2 or 10kHz)
    - n_peaks: Number of frequency peaks to detect (default: 10)
    - window: FFT window function — hann, hamming, blackman (default: hann)
    - target_sr: Resample to this rate (default: keep original)

    Response:
    {
        "results": {
            "file_info": {...},
            "peaks": [...],
            "harmonics": {...},
            "band_energy": {...},
            "spectral_stats": {...},
            "interpretation": {...},
            "fft_table": [...]
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

            # Apply window
            if window == 'hamming':
                win = np.hamming(len(y_trimmed))
            elif window == 'blackman':
                win = np.blackman(len(y_trimmed))
            else:  # hann (default)
                win = np.hanning(len(y_trimmed))
            y_windowed = y_trimmed * win

            # Compute FFT
            freqs, magnitude, power, fft_db = compute_fft(y_windowed, sr)

            # Max display frequency
            max_display_freq = max_frequency if max_frequency else min(sr / 2, 10000)
            n_peaks_val = n_peaks if n_peaks else 10

            # Find peaks
            display_mask = freqs <= max_display_freq
            peaks = find_peaks(freqs[display_mask], magnitude[display_mask], n_peaks=n_peaks_val)

            # Harmonic analysis
            harmonics = analyze_harmonics(peaks)

            # Band energy
            band_energy = compute_band_energy(freqs, power, sr)

            # Spectral statistics
            spectral_stats = compute_spectral_stats(freqs[display_mask], magnitude[display_mask])

            # Interpretation
            interpretation = generate_interpretation(peaks, harmonics, band_energy, spectral_stats)

            # Generate plots
            plot = generate_fft_plots(
                y_trimmed, sr, freqs, magnitude, power, fft_db,
                peaks, band_energy, max_display_freq
            )

            # Build FFT table (sampled — full FFT is too large)
            # Downsample to ~500 points for frontend table
            n_points = min(500, len(freqs[display_mask]))
            step = max(1, len(freqs[display_mask]) // n_points)
            fft_table = []
            for i in range(0, len(freqs[display_mask]), step):
                fft_table.append({
                    'frequency_hz': safe_float(freqs[i]),
                    'magnitude': safe_float(magnitude[i]),
                    'power': safe_float(power[i]),
                    'db': safe_float(fft_db[i]),
                })

            # File info
            file_info = {
                'filename': file.filename,
                'file_size_mb': round(len(content) / (1024 * 1024), 2),
                'sample_rate': sr,
                'duration_sec': round(len(y_trimmed) / sr, 2),
                'n_samples': len(y_trimmed),
                'format': file_ext.replace('.', ''),
                'window_function': window,
                'frequency_resolution_hz': safe_float(sr / len(y_trimmed)),
                'max_display_frequency': safe_float(max_display_freq),
            }

            return _to_native({
                'results': {
                    'file_info': file_info,
                    'peaks': peaks,
                    'harmonics': harmonics,
                    'band_energy': band_energy,
                    'spectral_stats': spectral_stats,
                    'interpretation': interpretation,
                    'fft_table': fft_table,
                },
                'plot': plot
            })

        finally:
            os.unlink(tmp_path)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ──────────────────────────────────────────────
# Router registration:
# from fft_spectrum_analysis import router as fft_router
# app.include_router(fft_router, prefix="/api/analysis", tags=["Audio Analysis"])
# → POST /api/analysis/fft-spectrum
# ──────────────────────────────────────────────
