"""
Mel Spectrogram Analysis Backend (FastAPI)
- Perceptual frequency analysis using mel-scaled spectrograms
- MFCC extraction and analysis
- Mel band energy distribution and temporal dynamics
- Chroma features, spectral contrast, and tonnetz
- Perceptual loudness and timbral analysis
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
# Mel Analysis Functions
# ──────────────────────────────────────────────

def compute_mel_spectrogram(y: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512,
                            n_mels: int = 128, fmin: float = 0.0, fmax: Optional[float] = None):
    """Compute mel-scaled spectrogram."""
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmin=fmin, fmax=fmax
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    times = librosa.frames_to_time(np.arange(mel_spec.shape[1]), sr=sr, hop_length=hop_length)
    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmin=fmin, fmax=fmax if fmax else sr / 2)
    return mel_spec, mel_db, times, mel_freqs


def compute_mfccs(y: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512,
                  n_mels: int = 128, n_mfcc: int = 13, fmin: float = 0.0, fmax: Optional[float] = None):
    """Compute Mel-Frequency Cepstral Coefficients."""
    mfccs = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmin=fmin, fmax=fmax
    )
    # Delta and delta-delta
    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
    times = librosa.frames_to_time(np.arange(mfccs.shape[1]), sr=sr, hop_length=hop_length)

    # Summary statistics per coefficient
    mfcc_stats = []
    for i in range(n_mfcc):
        mfcc_stats.append({
            'coefficient': i,
            'mean': safe_float(np.mean(mfccs[i])),
            'std': safe_float(np.std(mfccs[i])),
            'min': safe_float(np.min(mfccs[i])),
            'max': safe_float(np.max(mfccs[i])),
            'delta_mean': safe_float(np.mean(mfcc_delta[i])),
            'delta_std': safe_float(np.std(mfcc_delta[i])),
        })

    return mfccs, mfcc_delta, mfcc_delta2, times, mfcc_stats


def compute_chroma(y: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512):
    """Compute chroma features (pitch class distribution)."""
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length)

    pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    chroma_stats = []
    for i, pc in enumerate(pitch_classes):
        chroma_stats.append({
            'pitch_class': pc,
            'mean_energy': safe_float(np.mean(chroma[i])),
            'std_energy': safe_float(np.std(chroma[i])),
            'max_energy': safe_float(np.max(chroma[i])),
        })

    dominant_idx = np.argmax([s['mean_energy'] for s in chroma_stats])
    return chroma, times, chroma_stats, pitch_classes[dominant_idx]


def compute_spectral_contrast(y: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512):
    """Compute spectral contrast across frequency bands."""
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    times = librosa.frames_to_time(np.arange(contrast.shape[1]), sr=sr, hop_length=hop_length)

    band_labels = ['Band 1\n(0-100)', 'Band 2\n(100-200)', 'Band 3\n(200-400)',
                   'Band 4\n(400-800)', 'Band 5\n(800-1.6k)', 'Band 6\n(1.6k-3.2k)', 'Valley']
    contrast_stats = []
    for i in range(contrast.shape[0]):
        label = band_labels[i] if i < len(band_labels) else f'Band {i+1}'
        contrast_stats.append({
            'band': label.replace('\n', ' '),
            'mean_contrast': safe_float(np.mean(contrast[i])),
            'std_contrast': safe_float(np.std(contrast[i])),
        })

    return contrast, times, contrast_stats


def compute_mel_band_energy(mel_spec: np.ndarray, mel_freqs: np.ndarray, n_mels: int):
    """Compute energy distribution across mel bands (grouped)."""
    # Group mel bands into perceptual regions
    regions = [
        ('Low (0-300 Hz)', 0, 300),
        ('Low-Mid (300-1k Hz)', 300, 1000),
        ('Mid (1k-3k Hz)', 1000, 3000),
        ('High-Mid (3k-6k Hz)', 3000, 6000),
        ('High (6k-12k Hz)', 6000, 12000),
        ('Very High (12k+ Hz)', 12000, 100000),
    ]

    total_energy = np.sum(mel_spec)
    band_results = []

    for name, f_low, f_high in regions:
        mask = (mel_freqs >= f_low) & (mel_freqs < f_high)
        if not np.any(mask):
            continue
        band_energy = np.sum(mel_spec[mask, :])
        mean_db = safe_float(np.mean(librosa.power_to_db(mel_spec[mask, :] + 1e-10, ref=np.max(mel_spec))))
        band_results.append({
            'region': name,
            'energy': safe_float(band_energy),
            'energy_percent': safe_float(band_energy / (total_energy + 1e-10) * 100),
            'mean_db': mean_db,
        })

    dominant = max(band_results, key=lambda x: x['energy']) if band_results else None
    return {
        'bands': band_results,
        'total_energy': safe_float(total_energy),
        'dominant_region': dominant['region'] if dominant else 'N/A',
        'dominant_region_percent': safe_float(dominant['energy_percent']) if dominant else 0,
    }


def compute_perceptual_stats(mel_spec: np.ndarray, mel_db: np.ndarray, mel_freqs: np.ndarray,
                             mfccs: np.ndarray, chroma: np.ndarray):
    """Compute perceptual and timbral summary statistics."""
    # Mel-domain spectral centroid (perceptual brightness)
    mel_mag = np.mean(mel_spec, axis=1)
    total = np.sum(mel_mag) + 1e-10
    prob = mel_mag / total
    mel_centroid = np.sum(mel_freqs * prob)
    mel_spread = np.sqrt(np.sum(((mel_freqs - mel_centroid) ** 2) * prob))

    # Mel-domain flatness
    mel_flatness = np.exp(np.mean(np.log(mel_mag + 1e-10))) / (np.mean(mel_mag) + 1e-10)

    # Dynamic range in mel domain
    mel_max = np.max(mel_db)
    mel_min = np.min(mel_db)
    mel_dynamic_range = mel_max - mel_min

    # MFCC-based timbral descriptor
    # MFCC[0] ≈ overall energy, MFCC[1] ≈ spectral slope, MFCC[2-4] ≈ spectral shape
    timbral_brightness = safe_float(np.mean(mfccs[1])) if mfccs.shape[0] > 1 else 0
    timbral_shape = safe_float(np.mean(np.abs(mfccs[2:5]))) if mfccs.shape[0] > 4 else 0

    # Chroma strength (how tonal the signal is based on pitch classes)
    chroma_strength = safe_float(np.max(np.mean(chroma, axis=1)) - np.min(np.mean(chroma, axis=1)))

    return {
        'mel_centroid_hz': safe_float(mel_centroid),
        'mel_spread_hz': safe_float(mel_spread),
        'mel_flatness': safe_float(mel_flatness),
        'mel_dynamic_range_db': safe_float(mel_dynamic_range),
        'mel_peak_db': safe_float(mel_max),
        'mel_floor_db': safe_float(mel_min),
        'timbral_brightness': timbral_brightness,
        'timbral_shape_complexity': timbral_shape,
        'chroma_strength': chroma_strength,
    }


def generate_interpretation(perceptual_stats, mel_band_energy, mfcc_stats, chroma_dominant, n_mfcc):
    """Generate human-readable interpretation of mel spectrogram results."""
    lines = []

    # Perceptual centroid
    centroid = perceptual_stats['mel_centroid_hz']
    if centroid < 500:
        lines.append(f"Perceptual centroid at {centroid:.0f} Hz — signal sounds dark/warm with dominant low frequencies.")
    elif centroid < 2000:
        lines.append(f"Perceptual centroid at {centroid:.0f} Hz — balanced tonal character in the mid-frequency range.")
    else:
        lines.append(f"Perceptual centroid at {centroid:.0f} Hz — signal sounds bright with strong high-frequency content.")

    # Dominant mel band
    dom = mel_band_energy['dominant_region']
    pct = mel_band_energy['dominant_region_percent']
    lines.append(f"Energy concentrated in {dom} ({pct:.1f}% of total mel-scaled energy).")

    # Flatness
    fl = perceptual_stats['mel_flatness']
    if fl > 0.5:
        lines.append("Mel spectrum is noise-like (high flatness) — broadband energy distribution.")
    elif fl > 0.1:
        lines.append("Mel spectrum has mixed tonal and noise components.")
    else:
        lines.append("Mel spectrum is highly tonal — clear perceptual pitch structure.")

    # Dynamic range
    dr = perceptual_stats['mel_dynamic_range_db']
    if dr > 60:
        lines.append(f"Wide mel dynamic range ({dr:.1f} dB) — signal has large intensity variations across frequencies.")
    elif dr > 30:
        lines.append(f"Moderate mel dynamic range ({dr:.1f} dB).")
    else:
        lines.append(f"Narrow mel dynamic range ({dr:.1f} dB) — relatively uniform energy distribution.")

    # Chroma
    cs = perceptual_stats['chroma_strength']
    if cs > 0.3:
        lines.append(f"Strong pitch class structure detected — dominant pitch class: {chroma_dominant}. Indicates tonal/harmonic content.")
    elif cs > 0.1:
        lines.append(f"Moderate pitch class variation — some tonal content around {chroma_dominant}.")
    else:
        lines.append("Weak pitch class structure — signal is largely atonal or noise-dominated.")

    # MFCC timbral insight
    brightness = perceptual_stats['timbral_brightness']
    if brightness > 0:
        lines.append("MFCC analysis indicates spectral tilt toward higher frequencies (bright timbre).")
    else:
        lines.append("MFCC analysis indicates spectral tilt toward lower frequencies (dark timbre).")

    return {
        'summary': ' '.join(lines[:2]),
        'details': lines,
    }


def build_mfcc_temporal_table(mfccs, times, n_points=200):
    """Build sampled MFCC values over time for frontend display."""
    n = len(times)
    step = max(1, n // n_points)
    table = []
    for i in range(0, n, step):
        row = {'time_sec': safe_float(times[i])}
        for j in range(mfccs.shape[0]):
            row[f'mfcc_{j}'] = safe_float(mfccs[j, i])
        table.append(row)
    return table


# ──────────────────────────────────────────────
# Plot Generation
# ──────────────────────────────────────────────

def generate_mel_plots(
    y: np.ndarray, sr: int,
    mel_db: np.ndarray, mfccs: np.ndarray, mfcc_delta: np.ndarray,
    chroma: np.ndarray, contrast: np.ndarray,
    mel_band_energy: dict, perceptual_stats: dict,
    chroma_stats: list,
    n_fft: int, hop_length: int, n_mels: int,
    fmin: float, fmax: Optional[float],
) -> str:
    """Generate comprehensive Mel Spectrogram visualization."""

    fig = plt.figure(figsize=(18, 28))
    fig.suptitle('Mel Spectrogram Analysis', fontsize=18, fontweight='bold')
    gs = gridspec.GridSpec(6, 2, figure=fig, hspace=0.55, wspace=0.3)

    # ── 1. Waveform ──
    ax1 = fig.add_subplot(gs[0, :])
    t = np.arange(len(y)) / sr
    ax1.plot(t, y, color='#2196F3', linewidth=0.4, alpha=0.8)
    ax1.set_title('Time Domain — Waveform', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, linestyle='--', alpha=0.4)

    # ── 2. Mel Spectrogram ──
    ax2 = fig.add_subplot(gs[1, :])
    img2 = librosa.display.specshow(mel_db, x_axis='time', y_axis='mel', sr=sr,
                                     hop_length=hop_length, ax=ax2, cmap='magma',
                                     fmin=fmin, fmax=fmax)
    ax2.set_title(f'Mel Spectrogram ({n_mels} bands)', fontsize=13, fontweight='bold')
    fig.colorbar(img2, ax=ax2, label='dB', pad=0.02)

    # ── 3. MFCCs ──
    ax3 = fig.add_subplot(gs[2, 0])
    img3 = librosa.display.specshow(mfccs, x_axis='time', sr=sr, hop_length=hop_length,
                                     ax=ax3, cmap='coolwarm')
    ax3.set_title('MFCCs', fontsize=13, fontweight='bold')
    ax3.set_ylabel('MFCC Coefficient')
    fig.colorbar(img3, ax=ax3, label='Value', pad=0.02)

    # ── 4. MFCC Deltas ──
    ax4 = fig.add_subplot(gs[2, 1])
    img4 = librosa.display.specshow(mfcc_delta, x_axis='time', sr=sr, hop_length=hop_length,
                                     ax=ax4, cmap='coolwarm')
    ax4.set_title('MFCC Δ (Velocity)', fontsize=13, fontweight='bold')
    ax4.set_ylabel('MFCC Coefficient')
    fig.colorbar(img4, ax=ax4, label='Δ Value', pad=0.02)

    # ── 5. Chromagram ──
    ax5 = fig.add_subplot(gs[3, 0])
    img5 = librosa.display.specshow(chroma, x_axis='time', y_axis='chroma', sr=sr,
                                     hop_length=hop_length, ax=ax5, cmap='YlOrRd')
    ax5.set_title('Chromagram (Pitch Class)', fontsize=13, fontweight='bold')
    fig.colorbar(img5, ax=ax5, label='Energy', pad=0.02)

    # ── 6. Chroma Distribution ──
    ax6 = fig.add_subplot(gs[3, 1])
    pitch_classes = [s['pitch_class'] for s in chroma_stats]
    mean_energies = [s['mean_energy'] for s in chroma_stats]
    colors_chroma = ['#FF5722' if e == max(mean_energies) else '#FF9800' if e > np.mean(mean_energies) else '#4CAF50'
                     for e in mean_energies]
    bars = ax6.bar(pitch_classes, mean_energies, color=colors_chroma, edgecolor='white', linewidth=0.8)
    ax6.set_title('Pitch Class Distribution', fontsize=13, fontweight='bold')
    ax6.set_ylabel('Mean Energy')
    ax6.grid(True, linestyle='--', alpha=0.4, axis='y')
    for bar, val in zip(bars, mean_energies):
        if val > np.mean(mean_energies):
            ax6.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    # ── 7. Spectral Contrast ──
    ax7 = fig.add_subplot(gs[4, 0])
    img7 = librosa.display.specshow(contrast, x_axis='time', sr=sr, hop_length=hop_length,
                                     ax=ax7, cmap='viridis')
    ax7.set_title('Spectral Contrast', fontsize=13, fontweight='bold')
    ax7.set_ylabel('Frequency Band')
    fig.colorbar(img7, ax=ax7, label='Contrast (dB)', pad=0.02)

    # ── 8. Mel Band Energy Distribution ──
    ax8 = fig.add_subplot(gs[4, 1])
    bands = mel_band_energy['bands']
    region_names = [b['region'].split('(')[0].strip() for b in bands]
    region_pcts = [b['energy_percent'] for b in bands]
    colors_band = ['#1a237e', '#283593', '#3949ab', '#5c6bc0', '#7986cb', '#9fa8da']
    colors_band = colors_band[:len(region_names)]
    bars8 = ax8.bar(region_names, region_pcts, color=colors_band, edgecolor='white', linewidth=0.8)
    ax8.set_title('Mel Band Energy Distribution', fontsize=13, fontweight='bold')
    ax8.set_ylabel('Energy (%)')
    ax8.tick_params(axis='x', rotation=25)
    ax8.grid(True, linestyle='--', alpha=0.4, axis='y')
    for bar, pct in zip(bars8, region_pcts):
        if pct > 3:
            ax8.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{pct:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # ── 9. MFCC Coefficient Statistics ──
    ax9 = fig.add_subplot(gs[5, 0])
    from matplotlib.patches import Patch
    mfcc_means = [safe_float(np.mean(mfccs[i])) for i in range(mfccs.shape[0])]
    mfcc_stds = [safe_float(np.std(mfccs[i])) for i in range(mfccs.shape[0])]
    x_mfcc = np.arange(len(mfcc_means))
    ax9.bar(x_mfcc, mfcc_means, yerr=mfcc_stds, color='#9C27B0', edgecolor='white',
            linewidth=0.8, capsize=3, alpha=0.85)
    ax9.set_title('MFCC Statistics (Mean ± Std)', fontsize=13, fontweight='bold')
    ax9.set_xlabel('MFCC Coefficient')
    ax9.set_ylabel('Value')
    ax9.set_xticks(x_mfcc)
    ax9.grid(True, linestyle='--', alpha=0.4, axis='y')

    # ── 10. Perceptual Summary Radar-style (horizontal bar) ──
    ax10 = fig.add_subplot(gs[5, 1])
    ps = perceptual_stats
    labels = ['Mel Centroid', 'Mel Spread', 'Flatness', 'Brightness', 'Chroma\nStrength']
    # Normalize to 0-1 range for visualization
    raw_vals = [
        ps['mel_centroid_hz'] / 10000,
        ps['mel_spread_hz'] / 5000,
        ps['mel_flatness'],
        (ps['timbral_brightness'] + 100) / 200,  # rough normalize
        ps['chroma_strength'],
    ]
    vals = [min(1.0, max(0.0, v)) for v in raw_vals]
    colors_radar = ['#FF5722', '#FF9800', '#4CAF50', '#2196F3', '#9C27B0']
    bars10 = ax10.barh(labels, vals, color=colors_radar, edgecolor='white', linewidth=0.8)
    ax10.set_title('Perceptual Feature Summary', fontsize=13, fontweight='bold')
    ax10.set_xlim(0, 1)
    ax10.set_xlabel('Normalized Value')
    ax10.grid(True, linestyle='--', alpha=0.4, axis='x')

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"


# ──────────────────────────────────────────────
# Main Endpoint
# ──────────────────────────────────────────────

@router.post("/mel-spectrogram")
async def mel_spectrogram_analysis(
    file: UploadFile = File(...),
    n_fft: Optional[int] = Form(2048),
    hop_length: Optional[int] = Form(512),
    n_mels: Optional[int] = Form(128),
    n_mfcc: Optional[int] = Form(13),
    fmin: Optional[float] = Form(0.0),
    fmax: Optional[float] = Form(None),
    window: Optional[str] = Form('hann'),
    target_sr: Optional[int] = Form(None),
):
    """
    Mel Spectrogram Analysis.

    Perceptual frequency analysis using mel-scaled spectrograms,
    MFCCs, chroma features, and spectral contrast.

    Parameters:
    - file: Audio file (WAV, MP3, FLAC, etc.)
    - n_fft: FFT window size (default: 2048)
    - hop_length: Hop length in samples (default: 512)
    - n_mels: Number of mel bands (default: 128)
    - n_mfcc: Number of MFCCs to compute (default: 13)
    - fmin: Minimum frequency for mel filterbank (default: 0)
    - fmax: Maximum frequency for mel filterbank (default: sr/2)
    - window: Window function (default: hann)
    - target_sr: Resample to this rate (default: keep original)

    Response:
    {
        "results": {
            "file_info": {...},
            "perceptual_stats": {...},
            "mel_band_energy": {...},
            "mfcc_stats": [...],
            "chroma_stats": [...],
            "contrast_stats": [...],
            "mfcc_temporal_table": [...],
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
            n_mels_val = n_mels if n_mels else 128
            n_mfcc_val = n_mfcc if n_mfcc else 13
            fmin_val = fmin if fmin else 0.0
            fmax_val = fmax if fmax else None
            win_fn = window if window else 'hann'

            # Mel spectrogram
            mel_spec, mel_db, mel_times, mel_freqs = compute_mel_spectrogram(
                y_trimmed, sr, n_fft=n_fft_val, hop_length=hop_val,
                n_mels=n_mels_val, fmin=fmin_val, fmax=fmax_val
            )

            # MFCCs
            mfccs, mfcc_delta, mfcc_delta2, mfcc_times, mfcc_stats = compute_mfccs(
                y_trimmed, sr, n_fft=n_fft_val, hop_length=hop_val,
                n_mels=n_mels_val, n_mfcc=n_mfcc_val, fmin=fmin_val, fmax=fmax_val
            )

            # Chroma
            chroma, chroma_times, chroma_stats, chroma_dominant = compute_chroma(
                y_trimmed, sr, n_fft=n_fft_val, hop_length=hop_val
            )

            # Spectral contrast
            contrast, contrast_times, contrast_stats = compute_spectral_contrast(
                y_trimmed, sr, n_fft=n_fft_val, hop_length=hop_val
            )

            # Mel band energy
            mel_band_energy = compute_mel_band_energy(mel_spec, mel_freqs, n_mels_val)

            # Perceptual stats
            perceptual_stats = compute_perceptual_stats(mel_spec, mel_db, mel_freqs, mfccs, chroma)

            # Interpretation
            interpretation = generate_interpretation(
                perceptual_stats, mel_band_energy, mfcc_stats, chroma_dominant, n_mfcc_val
            )

            # MFCC temporal table
            mfcc_temporal_table = build_mfcc_temporal_table(mfccs, mfcc_times, n_points=200)

            # Generate plots
            plot = generate_mel_plots(
                y_trimmed, sr, mel_db, mfccs, mfcc_delta,
                chroma, contrast, mel_band_energy, perceptual_stats,
                chroma_stats, n_fft_val, hop_val, n_mels_val, fmin_val, fmax_val
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
                'n_mfcc': n_mfcc_val,
                'fmin': fmin_val,
                'fmax': fmax_val if fmax_val else sr / 2,
                'time_resolution_ms': safe_float(hop_val / sr * 1000),
                'frequency_resolution_hz': safe_float(sr / n_fft_val),
                'n_time_frames': int(mel_spec.shape[1]),
                'chroma_dominant_pitch': chroma_dominant,
            }

            return _to_native({
                'results': {
                    'file_info': file_info,
                    'perceptual_stats': perceptual_stats,
                    'mel_band_energy': mel_band_energy,
                    'mfcc_stats': mfcc_stats,
                    'chroma_stats': chroma_stats,
                    'contrast_stats': contrast_stats,
                    'mfcc_temporal_table': mfcc_temporal_table,
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
# from mel_spectrogram_analysis import router as mel_router
# app.include_router(mel_router, prefix="/api/analysis", tags=["Audio Analysis"])
# → POST /api/analysis/mel-spectrogram
# ──────────────────────────────────────────────
