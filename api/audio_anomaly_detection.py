"""
Audio Anomaly Detection Backend (FastAPI)
- Equipment Sound Anomaly Detection
- Requires baseline (normal) profile to compare against
- Pattern: Same as hdbscan.py
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import Any, List, Optional, Dict
import numpy as np
import pandas as pd
import io, base64, tempfile, os, json

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

try:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

sns.set_theme(style="darkgrid")

router = APIRouter()


# ──────────────────────────────────────────────
# Utility functions
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
# Feature extraction (shared with audio_feature_analysis.py)
# In production, import from a shared module
# ──────────────────────────────────────────────

def extract_features(y: np.ndarray, sr: int) -> dict:
    """Extract audio features from signal. Same as audio_feature_analysis.py."""
    features = {}

    features['rms_energy'] = safe_float(np.sqrt(np.mean(y ** 2)))
    features['peak_amplitude'] = safe_float(np.max(np.abs(y)))
    features['zero_crossing_rate'] = safe_float(np.mean(librosa.feature.zero_crossing_rate(y=y)))
    features['crest_factor'] = safe_float(np.max(np.abs(y)) / (np.sqrt(np.mean(y ** 2)) + 1e-10))

    fft = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), d=1.0 / sr)
    features['peak_frequency_hz'] = safe_float(freqs[np.argmax(fft)])
    features['mean_spectral_energy'] = safe_float(np.mean(fft ** 2))

    features['spectral_centroid'] = safe_float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    features['spectral_rolloff'] = safe_float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    features['spectral_flatness'] = safe_float(np.mean(librosa.feature.spectral_flatness(y=y)))

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc_{i + 1}'] = safe_float(np.mean(mfcc[i]))

    features['kurtosis'] = safe_float(np.mean((y - np.mean(y)) ** 4) / (np.std(y) ** 4 + 1e-10))
    features['skewness'] = safe_float(np.mean((y - np.mean(y)) ** 3) / (np.std(y) ** 3 + 1e-10))

    return features


def features_to_vector(features: dict, feature_keys: List[str]) -> np.ndarray:
    """Convert feature dict to ordered numpy array."""
    return np.array([features.get(k, 0.0) for k in feature_keys])


# ──────────────────────────────────────────────
# Baseline (Normal Profile) Models
# ──────────────────────────────────────────────

# Features used for anomaly detection (subset of all features)
ANOMALY_FEATURE_KEYS = [
    'rms_energy', 'peak_amplitude', 'zero_crossing_rate', 'crest_factor',
    'peak_frequency_hz', 'mean_spectral_energy',
    'spectral_centroid', 'spectral_rolloff', 'spectral_flatness',
    'mfcc_1', 'mfcc_2', 'mfcc_3', 'mfcc_4', 'mfcc_5',
    'kurtosis', 'skewness'
]


class BaselineProfile:
    """Represents a normal sound profile for an equipment."""

    def __init__(self, feature_vectors: np.ndarray):
        self.n_samples = len(feature_vectors)
        self.mean = np.mean(feature_vectors, axis=0)
        self.std = np.std(feature_vectors, axis=0) + 1e-10  # avoid division by zero
        self.min = np.min(feature_vectors, axis=0)
        self.max = np.max(feature_vectors, axis=0)

        # Fit Isolation Forest on normal data
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(feature_vectors)

        self.isolation_forest = IsolationForest(
            contamination=0.05,  # expect ~5% borderline in normal data
            random_state=42,
            n_estimators=100
        )
        self.isolation_forest.fit(X_scaled)

    def score(self, feature_vector: np.ndarray) -> dict:
        """
        Score a new sample against the baseline.
        Returns judgment, anomaly_score, and per-feature deviations.
        """
        # Z-score per feature
        z_scores = np.abs((feature_vector - self.mean) / self.std)
        max_z = float(np.max(z_scores))
        mean_z = float(np.mean(z_scores))

        # Isolation Forest score (-1 = anomaly, 1 = normal)
        X_scaled = self.scaler.transform(feature_vector.reshape(1, -1))
        if_score = self.isolation_forest.decision_function(X_scaled)[0]
        if_prediction = self.isolation_forest.predict(X_scaled)[0]

        # Combined anomaly score (0-100, higher = more anomalous)
        # Normalize IF score: typically ranges from -0.5 to 0.5
        if_normalized = max(0, min(100, (0.5 - if_score) * 100))

        # Z-score component (mean z-score normalized)
        z_normalized = max(0, min(100, mean_z * 20))

        # Weighted combination
        anomaly_score = 0.6 * if_normalized + 0.4 * z_normalized
        anomaly_score = max(0, min(100, anomaly_score))

        # Judgment
        if anomaly_score < 30:
            judgment = "normal"
            judgment_label = "Normal"
            judgment_description = "Signal is within expected range."
        elif anomaly_score < 60:
            judgment = "caution"
            judgment_label = "Caution"
            judgment_description = "Some features deviate from normal. Monitor closely."
        else:
            judgment = "anomaly"
            judgment_label = "Anomaly Detected"
            judgment_description = "Significant deviation from normal pattern."

        # Per-feature deviation details
        deviations = []
        for i, key in enumerate(ANOMALY_FEATURE_KEYS):
            dev = {
                'feature': key,
                'value': safe_float(feature_vector[i]),
                'normal_mean': safe_float(self.mean[i]),
                'normal_std': safe_float(self.std[i]),
                'z_score': safe_float(z_scores[i]),
                'status': 'normal' if z_scores[i] < 2 else ('caution' if z_scores[i] < 3 else 'anomaly')
            }
            deviations.append(dev)

        # Sort by z_score descending (most abnormal first)
        deviations.sort(key=lambda x: x['z_score'], reverse=True)

        # Top deviating features for summary
        top_deviations = [d for d in deviations if d['z_score'] >= 2]

        return {
            'judgment': judgment,
            'judgment_label': judgment_label,
            'judgment_description': judgment_description,
            'anomaly_score': round(anomaly_score, 1),
            'isolation_forest_score': safe_float(if_score),
            'mean_z_score': round(mean_z, 2),
            'max_z_score': round(max_z, 2),
            'deviations': deviations,
            'top_deviations': top_deviations,
            'baseline_samples': self.n_samples,
        }


# ──────────────────────────────────────────────
# Plot Generation for Anomaly Detection
# ──────────────────────────────────────────────

def generate_anomaly_plots(
    y: np.ndarray, sr: int,
    features: dict,
    score_result: dict,
    baseline_mean: np.ndarray,
    baseline_std: np.ndarray
) -> str:
    """Generate anomaly detection visualization."""

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle('Equipment Sound Anomaly Detection', fontsize=16, fontweight='bold')

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

    judgment = score_result['judgment']
    color_map = {'normal': '#4CAF50', 'caution': '#FF9800', 'anomaly': '#F44336'}
    main_color = color_map.get(judgment, '#999')

    # ── 1. Judgment Banner + Score Gauge ──
    ax1 = fig.add_subplot(gs[0, 0])
    score = score_result['anomaly_score']

    # Draw gauge
    theta = np.linspace(np.pi, 0, 100)
    ax1.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=3)

    # Color zones
    for i, (start, end, color) in enumerate([
        (0, 30, '#4CAF50'), (30, 60, '#FF9800'), (60, 100, '#F44336')
    ]):
        t_start = np.pi * (1 - start / 100)
        t_end = np.pi * (1 - end / 100)
        t = np.linspace(t_start, t_end, 50)
        ax1.fill_between(np.cos(t) * 0.95, np.sin(t) * 0.95, np.sin(t) * 0.7,
                        color=color, alpha=0.3)

    # Needle
    needle_angle = np.pi * (1 - score / 100)
    ax1.plot([0, np.cos(needle_angle) * 0.8], [0, np.sin(needle_angle) * 0.8],
            color=main_color, linewidth=4)
    ax1.plot(0, 0, 'ko', markersize=8)

    ax1.set_xlim(-1.3, 1.3)
    ax1.set_ylim(-0.2, 1.2)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title(f'{score_result["judgment_label"]}  (Score: {score:.0f}/100)',
                 fontsize=14, fontweight='bold', color=main_color)

    # ── 2. Spectrogram ──
    ax2 = fig.add_subplot(gs[0, 1])
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel', ax=ax2, cmap='magma')
    ax2.set_title('Mel Spectrogram', fontsize=12, fontweight='bold')

    # ── 3. Feature Deviation Chart (Normal vs Current) ──
    ax3 = fig.add_subplot(gs[1, :])

    # Show top features with their z-scores
    top_n = min(10, len(score_result['deviations']))
    top_devs = score_result['deviations'][:top_n]

    feature_names = [d['feature'] for d in top_devs]
    z_scores = [d['z_score'] for d in top_devs]
    colors = [color_map.get(d['status'], '#999') for d in top_devs]

    bars = ax3.barh(feature_names, z_scores, color=colors, edgecolor='black', linewidth=0.5)
    ax3.axvline(x=2, color='#FF9800', linestyle='--', alpha=0.7, label='Caution threshold (z=2)')
    ax3.axvline(x=3, color='#F44336', linestyle='--', alpha=0.7, label='Anomaly threshold (z=3)')
    ax3.set_title('Feature Deviation from Normal (Z-Score)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Z-Score (standard deviations from normal)')
    ax3.legend(loc='lower right', fontsize=9)
    ax3.grid(True, linestyle='--', alpha=0.5, axis='x')
    ax3.invert_yaxis()

    # ── 4. Radar Chart (Current vs Normal) ──
    ax4 = fig.add_subplot(gs[2, 0], projection='polar')

    # Select key features for radar
    radar_keys = ['rms_energy', 'peak_frequency_hz', 'spectral_centroid',
                  'zero_crossing_rate', 'kurtosis', 'crest_factor']
    radar_labels = ['RMS', 'Peak Freq', 'Centroid', 'ZCR', 'Kurtosis', 'Crest']

    # Normalize both to 0-1 range using baseline stats
    current_vals = []
    normal_vals = []
    for i, key in enumerate(radar_keys):
        idx = ANOMALY_FEATURE_KEYS.index(key) if key in ANOMALY_FEATURE_KEYS else None
        if idx is not None:
            mean = baseline_mean[idx]
            std = baseline_std[idx]
            cur_val = features.get(key, 0)
            # Normalize: how many std from mean, capped at 3
            current_vals.append(min(3, abs((cur_val - mean) / (std + 1e-10))))
            normal_vals.append(0.5)  # Normal is centered
        else:
            current_vals.append(0)
            normal_vals.append(0.5)

    # Close the radar
    angles = np.linspace(0, 2 * np.pi, len(radar_keys), endpoint=False).tolist()
    current_vals += current_vals[:1]
    normal_vals += normal_vals[:1]
    angles += angles[:1]

    ax4.plot(angles, normal_vals, 'o-', color='#4CAF50', linewidth=2, label='Normal Range')
    ax4.fill(angles, normal_vals, alpha=0.15, color='#4CAF50')
    ax4.plot(angles, current_vals, 'o-', color=main_color, linewidth=2, label='Current')
    ax4.fill(angles, current_vals, alpha=0.15, color=main_color)
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels(radar_labels, fontsize=9)
    ax4.set_title('Normal vs Current', fontsize=12, fontweight='bold', pad=20)
    ax4.legend(loc='upper right', fontsize=8, bbox_to_anchor=(1.3, 1.1))

    # ── 5. FFT Comparison ──
    ax5 = fig.add_subplot(gs[2, 1])
    fft = np.abs(np.fft.rfft(y))
    freqs_arr = np.fft.rfftfreq(len(y), d=1.0 / sr)
    max_freq = min(sr / 2, 10000)
    freq_mask = freqs_arr <= max_freq

    ax5.plot(freqs_arr[freq_mask], fft[freq_mask], color=main_color, linewidth=0.8, label='Current')
    ax5.fill_between(freqs_arr[freq_mask], fft[freq_mask], alpha=0.2, color=main_color)
    ax5.set_title('FFT Spectrum', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Frequency (Hz)')
    ax5.set_ylabel('Magnitude')
    ax5.legend(fontsize=9)
    ax5.grid(True, linestyle='--', alpha=0.5)

    # Save
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"


# ──────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────

class BaselineData(BaseModel):
    """Baseline profile data (pre-computed features from normal recordings)."""
    features_list: List[Dict[str, float]] = Field(
        ...,
        description="List of feature dicts from normal recordings (output of /audio-features)"
    )


@router.post("/audio-baseline")
async def create_baseline(req: BaselineData):
    """
    Create a baseline (normal) profile from multiple normal recordings.

    Input: List of feature dicts (from /audio-features endpoint results.features)
    Output: Baseline statistics (mean, std, min, max per feature)

    Store this on the frontend or DB — send it back when detecting anomalies.
    """
    if not SKLEARN_AVAILABLE:
        raise HTTPException(status_code=500, detail="scikit-learn not installed")

    try:
        if len(req.features_list) < 3:
            raise ValueError("Need at least 3 normal recordings to build a reliable baseline.")

        # Convert to feature vectors
        vectors = []
        for feat_dict in req.features_list:
            vec = features_to_vector(feat_dict, ANOMALY_FEATURE_KEYS)
            vectors.append(vec)

        vectors = np.array(vectors)

        # Build profile stats
        baseline = {
            'n_samples': len(vectors),
            'feature_keys': ANOMALY_FEATURE_KEYS,
            'mean': _to_native(np.mean(vectors, axis=0).tolist()),
            'std': _to_native((np.std(vectors, axis=0) + 1e-10).tolist()),
            'min': _to_native(np.min(vectors, axis=0).tolist()),
            'max': _to_native(np.max(vectors, axis=0).tolist()),
            'feature_vectors': _to_native(vectors.tolist()),  # needed for IF training
        }

        return _to_native({
            'baseline': baseline,
            'message': f'Baseline created from {len(vectors)} normal recordings.'
        })

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


class AnomalyRequest(BaseModel):
    """Request for anomaly detection."""
    features: Dict[str, float] = Field(
        ..., description="Features of the new recording (from /audio-features)"
    )
    baseline: Dict[str, Any] = Field(
        ..., description="Baseline profile (from /audio-baseline)"
    )


@router.post("/audio-anomaly")
async def detect_anomaly(
    file: UploadFile = File(...),
    baseline_json: str = Form(...),
):
    """
    Detect anomaly in a new recording against a baseline profile.

    Input:
    - file: New WAV recording to analyze
    - baseline_json: JSON string of baseline profile (from /audio-baseline)

    Output:
    {
        "results": {
            "judgment": "normal" | "caution" | "anomaly",
            "anomaly_score": 0-100,
            "summary": "...",
            "features": { ... },
            "deviations": [ ... ],
            "file_info": { ... }
        },
        "plot": "data:image/png;base64,..."
    }
    """
    if not LIBROSA_AVAILABLE:
        raise HTTPException(status_code=500, detail="librosa not installed")
    if not SKLEARN_AVAILABLE:
        raise HTTPException(status_code=500, detail="scikit-learn not installed")

    try:
        # Parse baseline
        baseline_data = json.loads(baseline_json)

        # Load audio
        content = await file.read()
        file_ext = os.path.splitext(file.filename or '.wav')[1].lower()

        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        try:
            y, sr = librosa.load(tmp_path, sr=None, mono=True)
            y_trimmed, _ = librosa.effects.trim(y, top_db=30)

            # Extract features from new recording
            features = extract_features(y_trimmed, sr)
            feature_vector = features_to_vector(features, ANOMALY_FEATURE_KEYS)

            # Rebuild baseline profile
            baseline_vectors = np.array(baseline_data['feature_vectors'])
            profile = BaselineProfile(baseline_vectors)

            # Score
            score_result = profile.score(feature_vector)

            # Generate plots
            plot = generate_anomaly_plots(
                y_trimmed, sr, features, score_result,
                profile.mean, profile.std
            )

            # Build summary text
            top_devs = score_result.get('top_deviations', [])
            if top_devs:
                dev_text = ', '.join([f"{d['feature']} (z={d['z_score']:.1f})" for d in top_devs[:3]])
                summary = f"{score_result['judgment_label']}: Score {score_result['anomaly_score']}/100. Top deviations: {dev_text}."
            else:
                summary = f"{score_result['judgment_label']}: Score {score_result['anomaly_score']}/100. All features within normal range."

            # File info
            file_info = {
                'filename': file.filename,
                'file_size_mb': round(len(content) / (1024 * 1024), 2),
                'sample_rate': sr,
                'duration_sec': round(len(y_trimmed) / sr, 2),
                'format': file_ext.replace('.', ''),
            }

            return _to_native({
                'results': {
                    'judgment': score_result['judgment'],
                    'judgment_label': score_result['judgment_label'],
                    'judgment_description': score_result['judgment_description'],
                    'anomaly_score': score_result['anomaly_score'],
                    'summary': summary,
                    'features': features,
                    'deviations': score_result['deviations'],
                    'top_deviations': score_result['top_deviations'],
                    'baseline_samples': score_result['baseline_samples'],
                    'mean_z_score': score_result['mean_z_score'],
                    'max_z_score': score_result['max_z_score'],
                    'file_info': file_info,
                },
                'plot': plot
            })

        finally:
            os.unlink(tmp_path)

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid baseline JSON")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ──────────────────────────────────────────────
# Router registration example
# ──────────────────────────────────────────────
#
# from audio_anomaly_detection import router as anomaly_router
# app.include_router(anomaly_router, prefix="/api/analysis", tags=["Audio Anomaly"])
#
# Endpoints:
#   POST /api/analysis/audio-baseline    — Create baseline from normal recordings
#   POST /api/analysis/audio-anomaly     — Detect anomaly in new recording
#
# Workflow:
#   1. Record normal sounds → POST /api/analysis/audio-features (multiple times)
#   2. Collect features → POST /api/analysis/audio-baseline → save baseline
#   3. New recording → POST /api/analysis/audio-anomaly (with baseline) → judgment
