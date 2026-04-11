"""
Audio Feature Analysis Backend (FastAPI)
- Equipment Sound Analysis: Feature Extraction
- Pattern: Same as hdbscan.py (router + single endpoint + base64 plot response)
- Input: WAV file upload
- Output: Features table + Waveform/Spectrogram/FFT plots (base64)
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
# Utility functions (same pattern as hdbscan.py)
# ──────────────────────────────────────────────

def _to_native(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [_to_native(x) for x in obj.tolist()]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {str(k): _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_native(x) for x in obj]
    return obj


def safe_float(val, default=0.0):
    """Safely convert value to float."""
    try:
        if val is None or pd.isna(val) or np.isinf(val):
            return default
        return float(val)
    except:
        return default


# ──────────────────────────────────────────────
# Feature Extraction Functions
# ──────────────────────────────────────────────

def extract_features(y: np.ndarray, sr: int) -> dict:
    """
    Extract audio features from a signal.
    Returns a flat dictionary of feature name -> value.
    """
    features = {}

    # 1. Time-domain features
    features['duration_sec'] = safe_float(len(y) / sr)
    features['sample_rate'] = int(sr)
    features['rms_energy'] = safe_float(np.sqrt(np.mean(y ** 2)))
    features['peak_amplitude'] = safe_float(np.max(np.abs(y)))
    features['zero_crossing_rate'] = safe_float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
    features['crest_factor'] = safe_float(
        np.max(np.abs(y)) / (np.sqrt(np.mean(y ** 2)) + 1e-10)
    )

    # 2. Frequency-domain features (FFT)
    fft = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), d=1.0 / sr)

    features['peak_frequency_hz'] = safe_float(freqs[np.argmax(fft)])
    features['mean_spectral_energy'] = safe_float(np.mean(fft ** 2))
    features['spectral_bandwidth'] = safe_float(
        np.sqrt(np.sum(((freqs - np.sum(freqs * fft) / (np.sum(fft) + 1e-10)) ** 2) * fft) / (np.sum(fft) + 1e-10))
    )

    # 3. Spectral features (librosa)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features['spectral_centroid'] = safe_float(np.mean(spectral_centroid))

    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features['spectral_rolloff'] = safe_float(np.mean(spectral_rolloff))

    spectral_flatness = librosa.feature.spectral_flatness(y=y)
    features['spectral_flatness'] = safe_float(np.mean(spectral_flatness))

    # 4. MFCC (13 coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc_{i + 1}'] = safe_float(np.mean(mfcc[i]))

    # 5. Statistical features (useful for equipment analysis)
    features['kurtosis'] = safe_float(
        np.mean((y - np.mean(y)) ** 4) / (np.std(y) ** 4 + 1e-10)
    )
    features['skewness'] = safe_float(
        np.mean((y - np.mean(y)) ** 3) / (np.std(y) ** 3 + 1e-10)
    )

    return features


def generate_feature_summary(features: dict) -> dict:
    """Generate human-readable summary of extracted features."""

    rms = features.get('rms_energy', 0)
    peak_freq = features.get('peak_frequency_hz', 0)
    zcr = features.get('zero_crossing_rate', 0)
    kurtosis = features.get('kurtosis', 0)
    crest = features.get('crest_factor', 0)

    # Signal strength assessment
    if rms > 0.3:
        strength = "Strong signal"
    elif rms > 0.1:
        strength = "Moderate signal"
    elif rms > 0.01:
        strength = "Weak signal"
    else:
        strength = "Very weak / near silence"

    # Frequency profile
    if peak_freq < 500:
        freq_profile = "Low-frequency dominant (motor hum, vibration)"
    elif peak_freq < 2000:
        freq_profile = "Mid-frequency dominant (mechanical operation)"
    elif peak_freq < 8000:
        freq_profile = "High-frequency dominant (friction, bearing wear)"
    else:
        freq_profile = "Very high frequency (leaks, electrical discharge)"

    # Impulsiveness (kurtosis)
    if kurtosis > 6:
        impulsive = "Highly impulsive — possible impact or fault"
    elif kurtosis > 4:
        impulsive = "Moderately impulsive"
    else:
        impulsive = "Smooth, stationary signal"

    # Key features summary
    key_features = [
        f"RMS Energy: {rms:.4f} — {strength}",
        f"Peak Frequency: {peak_freq:.1f} Hz — {freq_profile}",
        f"Kurtosis: {kurtosis:.2f} — {impulsive}",
        f"Crest Factor: {crest:.2f}",
        f"Zero Crossing Rate: {zcr:.4f}",
    ]

    return {
        'signal_strength': strength,
        'frequency_profile': freq_profile,
        'impulsiveness': impulsive,
        'key_features': key_features,
        'overview': f"{strength}. {freq_profile}. {impulsive}."
    }


# ──────────────────────────────────────────────
# Plot Generation
# ──────────────────────────────────────────────

def generate_plots(y: np.ndarray, sr: int, features: dict) -> str:
    """
    Generate a combined plot with:
    - Waveform
    - Spectrogram (Mel)
    - FFT Spectrum
    - Feature Bar Chart
    Returns base64 encoded PNG.
    """
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle('Audio Feature Analysis', fontsize=16, fontweight='bold')

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3)

    # ── 1. Waveform ──
    ax1 = fig.add_subplot(gs[0, :])
    time = np.arange(len(y)) / sr
    ax1.plot(time, y, color='#2196F3', linewidth=0.5, alpha=0.8)
    ax1.set_title('Waveform', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # RMS envelope overlay
    frame_length = 2048
    hop_length = 512
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_time = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    ax1.plot(rms_time, rms, color='#F44336', linewidth=2, label='RMS Envelope')
    ax1.plot(rms_time, -rms, color='#F44336', linewidth=2)
    ax1.legend(loc='upper right')

    # ── 2. Mel Spectrogram ──
    ax2 = fig.add_subplot(gs[1, 0])
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    img = librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax2, cmap='magma')
    ax2.set_title('Mel Spectrogram', fontsize=12, fontweight='bold')
    fig.colorbar(img, ax=ax2, format='%+2.0f dB', shrink=0.8)

    # ── 3. FFT Spectrum ──
    ax3 = fig.add_subplot(gs[1, 1])
    fft = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), d=1.0 / sr)

    # Limit to meaningful range (up to sr/2 or 10kHz)
    max_freq = min(sr / 2, 10000)
    freq_mask = freqs <= max_freq

    ax3.plot(freqs[freq_mask], fft[freq_mask], color='#4CAF50', linewidth=0.8)
    ax3.fill_between(freqs[freq_mask], fft[freq_mask], alpha=0.3, color='#4CAF50')
    ax3.set_title('FFT Spectrum', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('Magnitude')
    ax3.grid(True, linestyle='--', alpha=0.5)

    # Mark peak frequency
    peak_freq = features.get('peak_frequency_hz', 0)
    if peak_freq > 0 and peak_freq <= max_freq:
        peak_idx = np.argmin(np.abs(freqs - peak_freq))
        ax3.axvline(x=peak_freq, color='#F44336', linestyle='--', alpha=0.7)
        ax3.annotate(f'Peak: {peak_freq:.0f} Hz',
                     xy=(peak_freq, fft[peak_idx]),
                     xytext=(peak_freq + max_freq * 0.05, fft[peak_idx] * 0.9),
                     fontsize=9, color='#F44336', fontweight='bold')

    # ── 4. Key Features Bar Chart ──
    ax4 = fig.add_subplot(gs[2, 0])
    display_features = {
        'RMS Energy': features.get('rms_energy', 0),
        'ZCR': features.get('zero_crossing_rate', 0),
        'Spectral Centroid': features.get('spectral_centroid', 0) / 10000,  # normalize for display
        'Spectral Flatness': features.get('spectral_flatness', 0),
        'Crest Factor': features.get('crest_factor', 0) / 10,  # normalize
    }
    bars = ax4.barh(list(display_features.keys()), list(display_features.values()),
                    color=['#2196F3', '#4CAF50', '#FF9800', '#9C27B0', '#F44336'],
                    edgecolor='black', linewidth=0.5)
    ax4.set_title('Key Features (Normalized)', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Value')
    ax4.grid(True, linestyle='--', alpha=0.5, axis='x')

    # ── 5. MFCC Heatmap ──
    ax5 = fig.add_subplot(gs[2, 1])
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    img2 = librosa.display.specshow(mfcc, sr=sr, x_axis='time', ax=ax5, cmap='coolwarm')
    ax5.set_title('MFCC Coefficients', fontsize=12, fontweight='bold')
    ax5.set_ylabel('MFCC Index')
    fig.colorbar(img2, ax=ax5, shrink=0.8)

    # Save to base64
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    plot_base64 = f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"

    return plot_base64


# ──────────────────────────────────────────────
# Main Endpoint
# ──────────────────────────────────────────────

@router.post("/audio-features")
async def audio_feature_analysis(
    file: UploadFile = File(...),
    target_sr: Optional[int] = Form(None),    # Resample target (None = keep original)
    max_duration: Optional[float] = Form(None) # Limit duration in seconds (None = full)
):
    """
    Audio Feature Analysis Endpoint.

    - Upload a WAV file
    - Extracts time-domain, frequency-domain, spectral, MFCC features
    - Returns features table + combined visualization

    Response structure matches hdbscan pattern:
    {
        "results": {
            "file_info": { ... },
            "features": { ... },
            "feature_table": [ { ... } ],   # for frontend table rendering
            "summary": { ... },
        },
        "plot": "data:image/png;base64,..."
    }
    """

    if not LIBROSA_AVAILABLE:
        raise HTTPException(status_code=500, detail="librosa library not installed. pip install librosa")

    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    allowed_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format: {file_ext}. Supported: {', '.join(allowed_extensions)}"
        )

    try:
        # Save uploaded file temporarily
        content = await file.read()
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # Load audio with librosa
            sr_param = target_sr if target_sr else None
            y, sr = librosa.load(tmp_path, sr=sr_param, mono=True)

            # Trim silence (optional but helps with equipment recordings)
            y_trimmed, trim_idx = librosa.effects.trim(y, top_db=30)

            # Apply max_duration limit if specified
            if max_duration and len(y_trimmed) / sr > max_duration:
                y_trimmed = y_trimmed[:int(max_duration * sr)]

            # Extract features
            features = extract_features(y_trimmed, sr)

            # Generate summary
            summary = generate_feature_summary(features)

            # Generate plots
            plot = generate_plots(y_trimmed, sr, features)

            # Build feature table (for frontend table rendering)
            # Group features into categories for better display
            feature_table = []

            # Time-domain
            time_features = ['duration_sec', 'rms_energy', 'peak_amplitude',
                            'zero_crossing_rate', 'crest_factor']
            for key in time_features:
                if key in features:
                    feature_table.append({
                        'category': 'Time Domain',
                        'feature': key,
                        'value': features[key],
                        'description': _get_feature_description(key)
                    })

            # Frequency-domain
            freq_features = ['peak_frequency_hz', 'mean_spectral_energy', 'spectral_bandwidth',
                            'spectral_centroid', 'spectral_rolloff', 'spectral_flatness']
            for key in freq_features:
                if key in features:
                    feature_table.append({
                        'category': 'Frequency Domain',
                        'feature': key,
                        'value': features[key],
                        'description': _get_feature_description(key)
                    })

            # MFCC
            for i in range(1, 14):
                key = f'mfcc_{i}'
                if key in features:
                    feature_table.append({
                        'category': 'MFCC',
                        'feature': key,
                        'value': features[key],
                        'description': f'Mel-frequency cepstral coefficient {i}'
                    })

            # Statistical
            stat_features = ['kurtosis', 'skewness']
            for key in stat_features:
                if key in features:
                    feature_table.append({
                        'category': 'Statistical',
                        'feature': key,
                        'value': features[key],
                        'description': _get_feature_description(key)
                    })

            # File info
            file_info = {
                'filename': file.filename,
                'file_size_bytes': len(content),
                'file_size_mb': round(len(content) / (1024 * 1024), 2),
                'sample_rate': sr,
                'duration_sec': round(len(y_trimmed) / sr, 2),
                'original_duration_sec': round(len(y) / sr, 2),
                'n_samples': len(y_trimmed),
                'trimmed': len(y_trimmed) < len(y),
                'format': file_ext.replace('.', ''),
            }

            return _to_native({
                'results': {
                    'file_info': file_info,
                    'features': features,
                    'feature_table': feature_table,
                    'summary': summary,
                },
                'plot': plot
            })

        finally:
            # Clean up temp file
            os.unlink(tmp_path)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


def _get_feature_description(key: str) -> str:
    """Return human-readable description for each feature."""
    descriptions = {
        'duration_sec': 'Signal duration in seconds',
        'rms_energy': 'Root mean square energy — overall loudness',
        'peak_amplitude': 'Maximum absolute amplitude',
        'zero_crossing_rate': 'Rate of sign changes — indicates noise level',
        'crest_factor': 'Peak-to-RMS ratio — indicates impulsiveness',
        'peak_frequency_hz': 'Dominant frequency in the signal',
        'mean_spectral_energy': 'Average energy across all frequencies',
        'spectral_bandwidth': 'Spread of frequencies around centroid',
        'spectral_centroid': 'Center of mass of the spectrum — brightness',
        'spectral_rolloff': 'Frequency below which 85% of energy is concentrated',
        'spectral_flatness': 'How tone-like vs noise-like the signal is',
        'kurtosis': 'Peakedness of amplitude distribution — detects impacts',
        'skewness': 'Asymmetry of amplitude distribution',
    }
    return descriptions.get(key, '')


# ──────────────────────────────────────────────
# Router registration example (for main.py)
# ──────────────────────────────────────────────
#
# In your main FastAPI app:
#
# from audio_feature_analysis import router as audio_router
# app.include_router(audio_router, prefix="/api/analysis", tags=["Audio Analysis"])
#
# Then call: POST /api/analysis/audio-features
# with multipart/form-data: file=<WAV file>
