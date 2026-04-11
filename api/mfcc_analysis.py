"""
MFCC Analysis Backend (FastAPI)
- Deep Mel-Frequency Cepstral Coefficient analysis
- MFCC, Delta, Delta-Delta extraction with full statistics
- Cepstral distance metrics (Euclidean, Mahalanobis, cosine)
- MFCC-based temporal segmentation and clustering
- Cepstral liftering and feature normalization (CMS/CMVN)
- Per-coefficient temporal dynamics and stability analysis
- Correlation structure between coefficients
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
# MFCC Analysis Functions
# ──────────────────────────────────────────────

def extract_mfccs(y: np.ndarray, sr: int, n_fft: int = 2048, hop_length: int = 512,
                  n_mels: int = 128, n_mfcc: int = 13, fmin: float = 0.0,
                  fmax: Optional[float] = None, lifter: int = 0):
    """Extract MFCCs with optional cepstral liftering."""
    mfccs = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmin=fmin, fmax=fmax, lifter=lifter
    )
    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
    times = librosa.frames_to_time(np.arange(mfccs.shape[1]), sr=sr, hop_length=hop_length)
    return mfccs, mfcc_delta, mfcc_delta2, times


def apply_normalization(mfccs: np.ndarray, method: str = 'cms'):
    """Apply cepstral mean subtraction (CMS) or CMVN."""
    if method == 'cmvn':
        mean = np.mean(mfccs, axis=1, keepdims=True)
        std = np.std(mfccs, axis=1, keepdims=True) + 1e-10
        return (mfccs - mean) / std
    elif method == 'cms':
        mean = np.mean(mfccs, axis=1, keepdims=True)
        return mfccs - mean
    return mfccs


def compute_coefficient_stats(mfccs: np.ndarray, mfcc_delta: np.ndarray, mfcc_delta2: np.ndarray):
    """Compute detailed per-coefficient statistics."""
    n_mfcc = mfccs.shape[0]
    stats = []
    for i in range(n_mfcc):
        coeff = mfccs[i]
        delta = mfcc_delta[i]
        delta2 = mfcc_delta2[i]
        stats.append({
            'coefficient': i,
            'mean': safe_float(np.mean(coeff)),
            'std': safe_float(np.std(coeff)),
            'min': safe_float(np.min(coeff)),
            'max': safe_float(np.max(coeff)),
            'range': safe_float(np.max(coeff) - np.min(coeff)),
            'median': safe_float(np.median(coeff)),
            'skewness': safe_float(float(pd.Series(coeff).skew())),
            'kurtosis': safe_float(float(pd.Series(coeff).kurtosis())),
            'stability': safe_float(1.0 / (np.std(coeff) / (np.abs(np.mean(coeff)) + 1e-10) + 1e-10)),
            'delta_mean': safe_float(np.mean(delta)),
            'delta_std': safe_float(np.std(delta)),
            'delta2_mean': safe_float(np.mean(delta2)),
            'delta2_std': safe_float(np.std(delta2)),
            'delta_energy': safe_float(np.sqrt(np.mean(delta ** 2))),
            'delta2_energy': safe_float(np.sqrt(np.mean(delta2 ** 2))),
        })
    return stats


def compute_correlation_matrix(mfccs: np.ndarray):
    """Compute correlation between MFCC coefficients."""
    corr = np.corrcoef(mfccs)
    # Replace NaN with 0
    corr = np.nan_to_num(corr, nan=0.0)
    return corr


def compute_cepstral_distances(mfccs: np.ndarray, times: np.ndarray, n_segments: int = 5):
    """Compute cepstral distances between temporal segments."""
    n_frames = mfccs.shape[1]
    seg_len = n_frames // n_segments
    if seg_len < 2:
        return {'segments': [], 'distance_matrix': [], 'method': 'euclidean'}

    segment_means = []
    segment_info = []
    for i in range(n_segments):
        start = i * seg_len
        end = min(start + seg_len, n_frames)
        seg_mfcc = mfccs[:, start:end]
        seg_mean = np.mean(seg_mfcc, axis=1)
        segment_means.append(seg_mean)
        segment_info.append({
            'segment': i + 1,
            'start_sec': safe_float(times[start]),
            'end_sec': safe_float(times[min(end - 1, n_frames - 1)]),
            'mfcc_mean': [safe_float(x) for x in seg_mean],
        })

    # Euclidean distance matrix
    n = len(segment_means)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = np.linalg.norm(segment_means[i] - segment_means[j])

    # Cosine similarity matrix
    cosine_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            norm_i = np.linalg.norm(segment_means[i])
            norm_j = np.linalg.norm(segment_means[j])
            if norm_i > 0 and norm_j > 0:
                cosine_matrix[i, j] = np.dot(segment_means[i], segment_means[j]) / (norm_i * norm_j)
            else:
                cosine_matrix[i, j] = 0.0

    # Self-similarity (mean distance from each segment to all others)
    self_sim = []
    for i in range(n):
        others = [dist_matrix[i, j] for j in range(n) if j != i]
        self_sim.append(safe_float(np.mean(others)))

    return {
        'segments': segment_info,
        'distance_matrix': [[safe_float(dist_matrix[i, j]) for j in range(n)] for i in range(n)],
        'cosine_matrix': [[safe_float(cosine_matrix[i, j]) for j in range(n)] for i in range(n)],
        'self_similarity': self_sim,
        'mean_distance': safe_float(np.mean(dist_matrix[np.triu_indices(n, k=1)])),
        'max_distance': safe_float(np.max(dist_matrix)),
        'homogeneity_score': safe_float(1.0 / (1.0 + np.mean(dist_matrix[np.triu_indices(n, k=1)]))),
    }


def compute_global_summary(mfccs, mfcc_delta, mfcc_delta2, coeff_stats, cepstral_distances):
    """Compute global summary metrics for the MFCC analysis."""
    # Total MFCC energy
    total_energy = safe_float(np.sqrt(np.mean(mfccs ** 2)))
    delta_energy = safe_float(np.sqrt(np.mean(mfcc_delta ** 2)))
    delta2_energy = safe_float(np.sqrt(np.mean(mfcc_delta2 ** 2)))

    # Coefficient importance (by variance)
    variances = np.var(mfccs, axis=1)
    total_var = np.sum(variances) + 1e-10
    importance = variances / total_var

    # Most variable coefficient (excluding C0)
    most_variable = int(np.argmax(variances[1:]) + 1) if mfccs.shape[0] > 1 else 0
    most_stable = int(np.argmin(variances[1:]) + 1) if mfccs.shape[0] > 1 else 0

    # Temporal dynamics ratio (delta energy / static energy)
    dynamics_ratio = delta_energy / (total_energy + 1e-10)

    # Average coefficient stability
    stabilities = [s['stability'] for s in coeff_stats[1:]] if len(coeff_stats) > 1 else [0]
    avg_stability = safe_float(np.mean(stabilities))

    return {
        'total_mfcc_energy': total_energy,
        'delta_energy': delta_energy,
        'delta2_energy': delta2_energy,
        'dynamics_ratio': safe_float(dynamics_ratio),
        'n_coefficients': int(mfccs.shape[0]),
        'n_frames': int(mfccs.shape[1]),
        'most_variable_coeff': most_variable,
        'most_stable_coeff': most_stable,
        'avg_stability': avg_stability,
        'coefficient_importance': [safe_float(x) for x in importance],
        'homogeneity_score': cepstral_distances.get('homogeneity_score', 0),
        'mean_cepstral_distance': cepstral_distances.get('mean_distance', 0),
    }


def generate_interpretation(summary, coeff_stats, cepstral_distances, normalization):
    """Generate human-readable interpretation of MFCC analysis."""
    lines = []

    # Energy and dynamics
    dr = summary['dynamics_ratio']
    if dr > 0.5:
        lines.append(f"High temporal dynamics ratio ({dr:.2f}) — MFCCs change rapidly over time, indicating a highly non-stationary signal.")
    elif dr > 0.2:
        lines.append(f"Moderate dynamics ratio ({dr:.2f}) — signal has noticeable temporal variation in its cepstral features.")
    else:
        lines.append(f"Low dynamics ratio ({dr:.2f}) — MFCCs are relatively stable over time, suggesting a stationary signal.")

    # Homogeneity
    hs = summary['homogeneity_score']
    if hs > 0.7:
        lines.append(f"High cepstral homogeneity ({hs:.2f}) — signal maintains consistent spectral character throughout.")
    elif hs > 0.4:
        lines.append(f"Moderate cepstral homogeneity ({hs:.2f}) — some spectral variation across segments.")
    else:
        lines.append(f"Low cepstral homogeneity ({hs:.2f}) — significant spectral changes across time segments.")

    # Most variable coefficient
    mvc = summary['most_variable_coeff']
    lines.append(f"Most variable coefficient: MFCC {mvc} — carries the most discriminative information.")

    # C0 (energy)
    c0_mean = coeff_stats[0]['mean'] if coeff_stats else 0
    if c0_mean > 0:
        lines.append(f"MFCC[0] (energy proxy) mean: {c0_mean:.2f} — positive indicates moderate to high signal energy.")
    else:
        lines.append(f"MFCC[0] (energy proxy) mean: {c0_mean:.2f} — negative indicates low signal energy.")

    # Stability
    avg_stab = summary['avg_stability']
    if avg_stab > 5:
        lines.append(f"Average coefficient stability is high ({avg_stab:.1f}) — coefficients are consistent relative to their means.")
    elif avg_stab > 1:
        lines.append(f"Average coefficient stability is moderate ({avg_stab:.1f}).")
    else:
        lines.append(f"Average coefficient stability is low ({avg_stab:.1f}) — high relative variability in coefficients.")

    # Normalization note
    if normalization != 'none':
        lines.append(f"Normalization applied: {normalization.upper()}. This removes channel/environment effects for better comparability.")

    return {
        'summary': ' '.join(lines[:2]),
        'details': lines,
    }


def build_temporal_table(mfccs, mfcc_delta, mfcc_delta2, times, n_points=200):
    """Build sampled MFCC temporal table for frontend."""
    n = len(times)
    step = max(1, n // n_points)
    table = []
    for i in range(0, n, step):
        row = {'time_sec': safe_float(times[i])}
        for j in range(mfccs.shape[0]):
            row[f'mfcc_{j}'] = safe_float(mfccs[j, i])
        for j in range(min(mfcc_delta.shape[0], 5)):
            row[f'delta_{j}'] = safe_float(mfcc_delta[j, i])
        table.append(row)
    return table


# ──────────────────────────────────────────────
# Plot Generation
# ──────────────────────────────────────────────

def generate_mfcc_plots(
    y: np.ndarray, sr: int,
    mfccs: np.ndarray, mfcc_delta: np.ndarray, mfcc_delta2: np.ndarray,
    coeff_stats: list, corr_matrix: np.ndarray,
    cepstral_distances: dict, summary: dict,
    hop_length: int, n_mfcc: int,
) -> str:
    """Generate comprehensive MFCC visualization."""

    fig = plt.figure(figsize=(18, 30))
    fig.suptitle('MFCC Analysis', fontsize=18, fontweight='bold')
    gs = gridspec.GridSpec(7, 2, figure=fig, hspace=0.55, wspace=0.3)

    # ── 1. Waveform ──
    ax1 = fig.add_subplot(gs[0, :])
    t = np.arange(len(y)) / sr
    ax1.plot(t, y, color='#2196F3', linewidth=0.3, alpha=0.7)
    ax1.set_title('Time Domain — Waveform', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.grid(True, linestyle='--', alpha=0.4)

    # ── 2. MFCC Spectrogram ──
    ax2 = fig.add_subplot(gs[1, :])
    img2 = librosa.display.specshow(mfccs, x_axis='time', sr=sr, hop_length=hop_length,
                                     ax=ax2, cmap='coolwarm')
    ax2.set_title(f'MFCCs ({n_mfcc} coefficients)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('MFCC Coefficient')
    fig.colorbar(img2, ax=ax2, label='Value', pad=0.02)

    # ── 3. MFCC Delta ──
    ax3 = fig.add_subplot(gs[2, 0])
    img3 = librosa.display.specshow(mfcc_delta, x_axis='time', sr=sr, hop_length=hop_length,
                                     ax=ax3, cmap='coolwarm')
    ax3.set_title('MFCC Δ (Velocity)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Coefficient')
    fig.colorbar(img3, ax=ax3, label='Δ', pad=0.02)

    # ── 4. MFCC Delta-Delta ──
    ax4 = fig.add_subplot(gs[2, 1])
    img4 = librosa.display.specshow(mfcc_delta2, x_axis='time', sr=sr, hop_length=hop_length,
                                     ax=ax4, cmap='coolwarm')
    ax4.set_title('MFCC ΔΔ (Acceleration)', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Coefficient')
    fig.colorbar(img4, ax=ax4, label='ΔΔ', pad=0.02)

    # ── 5. Coefficient Mean ± Std ──
    ax5 = fig.add_subplot(gs[3, 0])
    means = [s['mean'] for s in coeff_stats]
    stds = [s['std'] for s in coeff_stats]
    x_c = np.arange(len(means))
    ax5.bar(x_c, means, yerr=stds, color='#9C27B0', edgecolor='white', linewidth=0.8,
            capsize=3, alpha=0.85)
    ax5.set_title('MFCC Statistics (Mean ± Std)', fontsize=13, fontweight='bold')
    ax5.set_xlabel('MFCC Coefficient')
    ax5.set_ylabel('Value')
    ax5.set_xticks(x_c)
    ax5.grid(True, linestyle='--', alpha=0.4, axis='y')

    # ── 6. Coefficient Importance (Variance) ──
    ax6 = fig.add_subplot(gs[3, 1])
    importance = summary['coefficient_importance']
    colors_imp = ['#F44336' if i == summary['most_variable_coeff'] else '#4CAF50' if v > np.mean(importance) else '#90CAF9'
                  for i, v in enumerate(importance)]
    ax6.bar(x_c, importance, color=colors_imp, edgecolor='white', linewidth=0.8)
    ax6.set_title('Coefficient Importance (Variance Ratio)', fontsize=13, fontweight='bold')
    ax6.set_xlabel('MFCC Coefficient')
    ax6.set_ylabel('Fraction of Total Variance')
    ax6.set_xticks(x_c)
    ax6.grid(True, linestyle='--', alpha=0.4, axis='y')

    # ── 7. Correlation Matrix ──
    ax7 = fig.add_subplot(gs[4, 0])
    im7 = ax7.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax7.set_title('MFCC Correlation Matrix', fontsize=13, fontweight='bold')
    ax7.set_xlabel('Coefficient')
    ax7.set_ylabel('Coefficient')
    ax7.set_xticks(x_c)
    ax7.set_yticks(x_c)
    fig.colorbar(im7, ax=ax7, label='Correlation', pad=0.02)

    # ── 8. Delta Energy per Coefficient ──
    ax8 = fig.add_subplot(gs[4, 1])
    delta_e = [s['delta_energy'] for s in coeff_stats]
    delta2_e = [s['delta2_energy'] for s in coeff_stats]
    width = 0.35
    ax8.bar(x_c - width / 2, delta_e, width, color='#FF5722', label='Δ Energy', edgecolor='white')
    ax8.bar(x_c + width / 2, delta2_e, width, color='#FF9800', label='ΔΔ Energy', edgecolor='white')
    ax8.set_title('Delta Energy per Coefficient', fontsize=13, fontweight='bold')
    ax8.set_xlabel('MFCC Coefficient')
    ax8.set_ylabel('RMS Energy')
    ax8.set_xticks(x_c)
    ax8.legend(fontsize=8)
    ax8.grid(True, linestyle='--', alpha=0.4, axis='y')

    # ── 9. Cepstral Distance Heatmap ──
    ax9 = fig.add_subplot(gs[5, 0])
    dist_mat = cepstral_distances.get('distance_matrix', [])
    if dist_mat:
        im9 = ax9.imshow(dist_mat, cmap='YlOrRd', aspect='auto')
        n_seg = len(dist_mat)
        ax9.set_xticks(range(n_seg))
        ax9.set_yticks(range(n_seg))
        ax9.set_xticklabels([f'S{i+1}' for i in range(n_seg)])
        ax9.set_yticklabels([f'S{i+1}' for i in range(n_seg)])
        fig.colorbar(im9, ax=ax9, label='Euclidean Distance', pad=0.02)
    ax9.set_title('Segment Cepstral Distance', fontsize=13, fontweight='bold')

    # ── 10. Cosine Similarity Heatmap ──
    ax10 = fig.add_subplot(gs[5, 1])
    cos_mat = cepstral_distances.get('cosine_matrix', [])
    if cos_mat:
        im10 = ax10.imshow(cos_mat, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
        n_seg = len(cos_mat)
        ax10.set_xticks(range(n_seg))
        ax10.set_yticks(range(n_seg))
        ax10.set_xticklabels([f'S{i+1}' for i in range(n_seg)])
        ax10.set_yticklabels([f'S{i+1}' for i in range(n_seg)])
        fig.colorbar(im10, ax=ax10, label='Cosine Similarity', pad=0.02)
    ax10.set_title('Segment Cosine Similarity', fontsize=13, fontweight='bold')

    # ── 11. Coefficient Stability ──
    ax11 = fig.add_subplot(gs[6, 0])
    stabilities = [min(s['stability'], 20) for s in coeff_stats]  # cap for display
    colors_stab = ['#4CAF50' if s > 5 else '#FF9800' if s > 1 else '#F44336' for s in stabilities]
    ax11.bar(x_c, stabilities, color=colors_stab, edgecolor='white', linewidth=0.8)
    ax11.set_title('Coefficient Stability (mean/CV)', fontsize=13, fontweight='bold')
    ax11.set_xlabel('MFCC Coefficient')
    ax11.set_ylabel('Stability Index')
    ax11.set_xticks(x_c)
    ax11.grid(True, linestyle='--', alpha=0.4, axis='y')

    # ── 12. Self-Similarity ──
    ax12 = fig.add_subplot(gs[6, 1])
    self_sim = cepstral_distances.get('self_similarity', [])
    if self_sim:
        seg_labels = [f'S{i+1}' for i in range(len(self_sim))]
        colors_ss = ['#F44336' if s == max(self_sim) else '#4CAF50' if s == min(self_sim) else '#2196F3'
                     for s in self_sim]
        ax12.bar(seg_labels, self_sim, color=colors_ss, edgecolor='white', linewidth=0.8)
        ax12.axhline(y=np.mean(self_sim), color='#F44336', linestyle='--', linewidth=0.8, alpha=0.5,
                     label=f'Mean: {np.mean(self_sim):.2f}')
        ax12.legend(fontsize=8)
    ax12.set_title('Segment Self-Dissimilarity', fontsize=13, fontweight='bold')
    ax12.set_xlabel('Segment')
    ax12.set_ylabel('Mean Cepstral Distance')
    ax12.grid(True, linestyle='--', alpha=0.4, axis='y')

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"


# ──────────────────────────────────────────────
# Main Endpoint
# ──────────────────────────────────────────────

@router.post("/mfcc-analysis")
async def mfcc_analysis(
    file: UploadFile = File(...),
    n_fft: Optional[int] = Form(2048),
    hop_length: Optional[int] = Form(512),
    n_mels: Optional[int] = Form(128),
    n_mfcc: Optional[int] = Form(13),
    fmin: Optional[float] = Form(0.0),
    fmax: Optional[float] = Form(None),
    lifter: Optional[int] = Form(0),
    normalization: Optional[str] = Form('none'),
    n_segments: Optional[int] = Form(5),
    target_sr: Optional[int] = Form(None),
):
    """
    MFCC Analysis.

    Deep cepstral domain analysis of audio signals including full MFCC/delta/delta-delta
    statistics, coefficient importance, correlation structure, cepstral distance metrics,
    and temporal segmentation analysis.

    Parameters:
    - file: Audio file (WAV, MP3, FLAC, etc.)
    - n_fft: FFT window size (default: 2048)
    - hop_length: Hop length in samples (default: 512)
    - n_mels: Number of mel bands (default: 128)
    - n_mfcc: Number of MFCCs (default: 13)
    - fmin: Min frequency for mel filterbank (default: 0)
    - fmax: Max frequency for mel filterbank (default: sr/2)
    - lifter: Cepstral liftering coefficient (default: 0 = no liftering)
    - normalization: 'none', 'cms' (mean subtraction), or 'cmvn' (mean+variance)
    - n_segments: Number of segments for cepstral distance (default: 5)
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

            # Trim & limit
            y_trimmed, _ = librosa.effects.trim(y, top_db=30)
            max_samples = sr * 60
            if len(y_trimmed) > max_samples:
                y_trimmed = y_trimmed[:max_samples]

            n_fft_val = n_fft or 2048
            hop_val = hop_length or 512
            n_mels_val = n_mels or 128
            n_mfcc_val = n_mfcc or 13
            fmin_val = fmin or 0.0
            fmax_val = fmax or None
            lifter_val = lifter or 0
            norm_method = normalization or 'none'
            n_seg = n_segments or 5

            # Extract MFCCs
            mfccs, mfcc_delta, mfcc_delta2, times = extract_mfccs(
                y_trimmed, sr, n_fft=n_fft_val, hop_length=hop_val,
                n_mels=n_mels_val, n_mfcc=n_mfcc_val, fmin=fmin_val,
                fmax=fmax_val, lifter=lifter_val
            )

            # Normalization
            mfccs_norm = apply_normalization(mfccs, method=norm_method)

            # Coefficient statistics
            coeff_stats = compute_coefficient_stats(mfccs_norm, mfcc_delta, mfcc_delta2)

            # Correlation matrix
            corr_matrix = compute_correlation_matrix(mfccs_norm)

            # Cepstral distances
            cepstral_distances = compute_cepstral_distances(mfccs_norm, times, n_segments=n_seg)

            # Global summary
            summary = compute_global_summary(mfccs_norm, mfcc_delta, mfcc_delta2, coeff_stats, cepstral_distances)

            # Interpretation
            interpretation = generate_interpretation(summary, coeff_stats, cepstral_distances, norm_method)

            # Temporal table
            temporal_table = build_temporal_table(mfccs_norm, mfcc_delta, mfcc_delta2, times, n_points=200)

            # Correlation as list
            corr_list = [[safe_float(corr_matrix[i, j]) for j in range(corr_matrix.shape[1])]
                         for i in range(corr_matrix.shape[0])]

            # Generate plots
            plot = generate_mfcc_plots(
                y_trimmed, sr, mfccs_norm, mfcc_delta, mfcc_delta2,
                coeff_stats, corr_matrix, cepstral_distances, summary,
                hop_val, n_mfcc_val
            )

            file_info = {
                'filename': file.filename,
                'file_size_mb': round(len(content) / (1024 * 1024), 2),
                'sample_rate': sr,
                'duration_sec': round(len(y_trimmed) / sr, 2),
                'n_samples': len(y_trimmed),
                'format': file_ext.replace('.', ''),
                'n_fft': n_fft_val,
                'hop_length': hop_val,
                'n_mels': n_mels_val,
                'n_mfcc': n_mfcc_val,
                'fmin': fmin_val,
                'fmax': fmax_val if fmax_val else sr / 2,
                'lifter': lifter_val,
                'normalization': norm_method,
                'n_segments': n_seg,
                'n_frames': int(mfccs.shape[1]),
                'time_resolution_ms': safe_float(hop_val / sr * 1000),
            }

            return _to_native({
                'results': {
                    'file_info': file_info,
                    'summary': summary,
                    'coefficient_stats': coeff_stats,
                    'correlation_matrix': corr_list,
                    'cepstral_distances': cepstral_distances,
                    'temporal_table': temporal_table,
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
# from mfcc_analysis import router as mfcc_router
# app.include_router(mfcc_router, prefix="/api/analysis", tags=["Audio Analysis"])
# → POST /api/analysis/mfcc-analysis
# ──────────────────────────────────────────────
