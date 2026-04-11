"""
Tonnetz Analysis Backend (FastAPI)
- Comprehensive tonal centroid (tonnetz) analysis
- 6-dimensional tonnetz features: fifths (x1,y1), minor thirds (x2,y2), major thirds (x3,y3)
- Tonnetz trajectory and tonal space movement
- Harmonic interval strength and tonal tension metrics
- Tonnetz-based key region detection and modulation tracking
- Segment-wise tonnetz statistics and stability
- Tonnetz distance/similarity between temporal segments
- Tonal complexity, dissonance estimation, and harmonic flow
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

TONNETZ_LABELS = [
    'Fifth (x₁)', 'Fifth (y₁)',
    'Minor 3rd (x₂)', 'Minor 3rd (y₂)',
    'Major 3rd (x₃)', 'Major 3rd (y₃)',
]
TONNETZ_SHORT = ['5th_x', '5th_y', 'm3_x', 'm3_y', 'M3_x', 'M3_y']


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
# Tonnetz Analysis Functions
# ──────────────────────────────────────────────

def compute_tonnetz(y, sr, hop_length=512):
    """Compute 6-dim tonnetz features."""
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    times = librosa.frames_to_time(np.arange(tonnetz.shape[1]), sr=sr, hop_length=hop_length)
    return tonnetz, chroma, times


def compute_tonnetz_statistics(tonnetz):
    """Per-dimension statistics."""
    stats = []
    for i in range(6):
        row = tonnetz[i]
        stats.append({
            'dimension': TONNETZ_LABELS[i],
            'short': TONNETZ_SHORT[i],
            'mean': safe_float(np.mean(row)),
            'std': safe_float(np.std(row)),
            'min': safe_float(np.min(row)),
            'max': safe_float(np.max(row)),
            'range': safe_float(np.max(row) - np.min(row)),
            'median': safe_float(np.median(row)),
            'skewness': safe_float(float(pd.Series(row).skew())),
            'energy': safe_float(np.sqrt(np.mean(row ** 2))),
        })
    return stats


def compute_interval_strength(tonnetz):
    """Compute interval strengths from tonnetz pairs."""
    # Each pair (x,y) represents a harmonic interval as a 2D vector
    # Magnitude = strength of that interval
    fifth_strength = np.sqrt(tonnetz[0] ** 2 + tonnetz[1] ** 2)
    minor3_strength = np.sqrt(tonnetz[2] ** 2 + tonnetz[3] ** 2)
    major3_strength = np.sqrt(tonnetz[4] ** 2 + tonnetz[5] ** 2)

    return {
        'fifth': {
            'mean': safe_float(np.mean(fifth_strength)),
            'std': safe_float(np.std(fifth_strength)),
            'max': safe_float(np.max(fifth_strength)),
            'values': fifth_strength,
        },
        'minor_third': {
            'mean': safe_float(np.mean(minor3_strength)),
            'std': safe_float(np.std(minor3_strength)),
            'max': safe_float(np.max(minor3_strength)),
            'values': minor3_strength,
        },
        'major_third': {
            'mean': safe_float(np.mean(major3_strength)),
            'std': safe_float(np.std(major3_strength)),
            'max': safe_float(np.max(major3_strength)),
            'values': major3_strength,
        },
    }


def compute_tonal_tension(tonnetz):
    """Estimate tonal tension as the total movement in tonnetz space."""
    # Tension = magnitude of tonnetz vector per frame
    magnitude = np.sqrt(np.sum(tonnetz ** 2, axis=0))
    # Tension flux = rate of change in tonnetz space
    tonnetz_diff = np.diff(tonnetz, axis=1)
    flux = np.sqrt(np.sum(tonnetz_diff ** 2, axis=0))

    return {
        'mean_magnitude': safe_float(np.mean(magnitude)),
        'std_magnitude': safe_float(np.std(magnitude)),
        'max_magnitude': safe_float(np.max(magnitude)),
        'mean_flux': safe_float(np.mean(flux)),
        'std_flux': safe_float(np.std(flux)),
        'max_flux': safe_float(np.max(flux)),
        'magnitude': magnitude,
        'flux': flux,
    }


def compute_tonnetz_correlation(tonnetz):
    """Correlation matrix between tonnetz dimensions."""
    corr = np.corrcoef(tonnetz)
    corr = np.nan_to_num(corr, nan=0.0)
    return corr


def compute_tonnetz_segments(tonnetz, times, n_segments=10):
    """Per-segment tonnetz statistics."""
    n = tonnetz.shape[1]
    seg_len = max(1, n // n_segments)
    segments = []

    for i in range(n_segments):
        start = i * seg_len
        end = min(start + seg_len, n)
        if start >= n: break
        seg = tonnetz[:, start:end]

        seg_mean = np.mean(seg, axis=1)
        seg_mag = np.sqrt(np.sum(seg_mean ** 2))

        # Dominant interval
        fifth_s = np.sqrt(seg_mean[0]**2 + seg_mean[1]**2)
        minor3_s = np.sqrt(seg_mean[2]**2 + seg_mean[3]**2)
        major3_s = np.sqrt(seg_mean[4]**2 + seg_mean[5]**2)
        strengths = {'Fifth': fifth_s, 'Minor 3rd': minor3_s, 'Major 3rd': major3_s}
        dominant = max(strengths, key=strengths.get)

        segments.append({
            'segment': i + 1,
            'start_sec': safe_float(times[start]),
            'end_sec': safe_float(times[min(end - 1, n - 1)]),
            'magnitude': safe_float(seg_mag),
            'fifth_strength': safe_float(fifth_s),
            'minor3_strength': safe_float(minor3_s),
            'major3_strength': safe_float(major3_s),
            'dominant_interval': dominant,
            'mean_values': [safe_float(x) for x in seg_mean],
        })
    return segments


def compute_tonnetz_distance_matrix(tonnetz, times, n_segments=10):
    """Cosine distance between segment means in tonnetz space."""
    n = tonnetz.shape[1]
    seg_len = max(1, n // n_segments)
    means = []
    seg_times = []

    for i in range(n_segments):
        start = i * seg_len
        end = min(start + seg_len, n)
        if start >= n: break
        means.append(np.mean(tonnetz[:, start:end], axis=1))
        seg_times.append(safe_float(times[start]))

    ns = len(means)
    dist = np.zeros((ns, ns))
    for i in range(ns):
        for j in range(ns):
            ni = np.linalg.norm(means[i])
            nj = np.linalg.norm(means[j])
            if ni > 0 and nj > 0:
                dist[i, j] = 1.0 - np.dot(means[i], means[j]) / (ni * nj)
            else:
                dist[i, j] = 1.0

    return dist, seg_times


def compute_tonal_complexity(tonnetz, tension):
    """Overall tonal complexity metrics."""
    # Dimensionality: how many tonnetz dimensions are active
    dim_energies = np.array([np.sqrt(np.mean(tonnetz[i] ** 2)) for i in range(6)])
    total_e = np.sum(dim_energies) + 1e-10
    dim_ratios = dim_energies / total_e

    # Entropy of dimension distribution
    dim_ratios_safe = dim_ratios + 1e-10
    entropy = -np.sum(dim_ratios_safe * np.log2(dim_ratios_safe))
    max_entropy = np.log2(6)
    norm_entropy = entropy / max_entropy

    # Tonal complexity = flux * entropy
    complexity = tension['mean_flux'] * norm_entropy

    return {
        'dimension_energies': [safe_float(x) for x in dim_energies],
        'dimension_ratios': [safe_float(x) for x in dim_ratios],
        'entropy': safe_float(entropy),
        'normalized_entropy': safe_float(norm_entropy),
        'tonal_complexity': safe_float(complexity),
        'mean_tension': tension['mean_magnitude'],
        'mean_flux': tension['mean_flux'],
    }


def generate_interpretation(stats, intervals, tension, complexity, segments):
    """Generate human-readable interpretation."""
    lines = []

    # Dominant interval
    ivs = {
        'Fifth': intervals['fifth']['mean'],
        'Minor 3rd': intervals['minor_third']['mean'],
        'Major 3rd': intervals['major_third']['mean'],
    }
    dom = max(ivs, key=ivs.get)
    lines.append(f"Dominant harmonic interval: {dom} (strength: {ivs[dom]:.4f}). "
                 f"This suggests {'strong tonal foundation' if dom == 'Fifth' else 'minor-key tendency' if dom == 'Minor 3rd' else 'major-key tendency'}.")

    # Tension
    mt = tension['mean_magnitude']
    if mt > 0.3:
        lines.append(f"High tonal tension (magnitude: {mt:.4f}) — rich harmonic content with strong interval presence.")
    elif mt > 0.1:
        lines.append(f"Moderate tonal tension (magnitude: {mt:.4f}) — typical for tonal music.")
    else:
        lines.append(f"Low tonal tension (magnitude: {mt:.4f}) — weak harmonic structure or noise-like content.")

    # Flux
    mf = tension['mean_flux']
    if mf > 0.1:
        lines.append(f"High tonnetz flux ({mf:.4f}) — rapid movement through tonal space, frequent modulations or chord changes.")
    elif mf > 0.03:
        lines.append(f"Moderate tonnetz flux ({mf:.4f}) — normal harmonic progression.")
    else:
        lines.append(f"Low tonnetz flux ({mf:.4f}) — harmonically static or sustained content.")

    # Complexity
    tc = complexity['tonal_complexity']
    ne = complexity['normalized_entropy']
    if ne > 0.85:
        lines.append(f"High tonal dimension entropy ({ne:.3f}) — all harmonic intervals contribute roughly equally.")
    elif ne > 0.6:
        lines.append(f"Moderate entropy ({ne:.3f}) — some intervals dominate but diversity is present.")
    else:
        lines.append(f"Low entropy ({ne:.3f}) — harmonic content is concentrated in few interval types.")

    lines.append(f"Overall tonal complexity: {tc:.4f}.")

    # Segment variation
    dom_intervals = [s['dominant_interval'] for s in segments]
    unique = len(set(dom_intervals))
    if unique == 1:
        lines.append(f"Consistently {dom_intervals[0]}-dominated throughout all segments.")
    else:
        lines.append(f"Dominant interval changes across segments ({unique} different): {', '.join(set(dom_intervals))}.")

    return {
        'summary': ' '.join(lines[:2]),
        'details': lines,
    }


def build_tonnetz_table(tonnetz, times, n_points=300):
    """Sampled tonnetz data for frontend."""
    n = min(tonnetz.shape[1], len(times))
    step = max(1, n // n_points)
    table = []
    for i in range(0, n, step):
        row = {'time_sec': safe_float(times[i])}
        for j in range(6):
            row[TONNETZ_SHORT[j]] = safe_float(tonnetz[j, i])
        table.append(row)
    return table


# ──────────────────────────────────────────────
# Plot Generation
# ──────────────────────────────────────────────

def generate_tonnetz_plots(
    y, sr, tonnetz, chroma, times,
    stats, intervals, tension, complexity,
    segments, dist_matrix, dist_times,
    corr_matrix, hop_length,
) -> str:

    fig = plt.figure(figsize=(18, 32))
    fig.suptitle('Tonnetz Analysis', fontsize=18, fontweight='bold')
    gs = gridspec.GridSpec(8, 2, figure=fig, hspace=0.55, wspace=0.3)

    # 1. Waveform
    ax1 = fig.add_subplot(gs[0, :])
    t = np.arange(len(y)) / sr
    ax1.plot(t, y, color='#2196F3', linewidth=0.3, alpha=0.6)
    ax1.set_title('Time Domain — Waveform', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Amplitude')
    ax1.grid(True, linestyle='--', alpha=0.4)

    # 2. Tonnetz Spectrogram
    ax2 = fig.add_subplot(gs[1, :])
    img2 = librosa.display.specshow(tonnetz, x_axis='time', sr=sr, hop_length=hop_length,
                                     ax=ax2, cmap='coolwarm')
    ax2.set_yticks(range(6))
    ax2.set_yticklabels(TONNETZ_SHORT, fontsize=9)
    ax2.set_title('Tonnetz Features over Time', fontsize=13, fontweight='bold')
    fig.colorbar(img2, ax=ax2, label='Value', pad=0.02)

    # 3. Interval Strengths over Time
    ax3 = fig.add_subplot(gs[2, :])
    ft = times[:len(intervals['fifth']['values'])]
    ax3.plot(ft, intervals['fifth']['values'], color='#2196F3', linewidth=0.8, alpha=0.7, label='Fifth')
    ax3.plot(ft, intervals['minor_third']['values'], color='#F44336', linewidth=0.8, alpha=0.7, label='Minor 3rd')
    ax3.plot(ft, intervals['major_third']['values'], color='#4CAF50', linewidth=0.8, alpha=0.7, label='Major 3rd')
    ax3.set_title('Harmonic Interval Strength over Time', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Time (s)'); ax3.set_ylabel('Strength')
    ax3.legend(fontsize=8); ax3.grid(True, linestyle='--', alpha=0.4)

    # 4. Dimension Mean ± Std
    ax4 = fig.add_subplot(gs[3, 0])
    means = [s['mean'] for s in stats]
    stds = [s['std'] for s in stats]
    colors_d = ['#2196F3', '#2196F3', '#F44336', '#F44336', '#4CAF50', '#4CAF50']
    ax4.bar(range(6), means, yerr=stds, color=colors_d, edgecolor='white', linewidth=0.8, capsize=3, alpha=0.85)
    ax4.set_xticks(range(6)); ax4.set_xticklabels(TONNETZ_SHORT, fontsize=8)
    ax4.set_title('Tonnetz Mean ± Std', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Value'); ax4.grid(True, linestyle='--', alpha=0.4, axis='y')

    # 5. Dimension Energy
    ax5 = fig.add_subplot(gs[3, 1])
    dim_e = complexity['dimension_energies']
    ax5.bar(range(6), dim_e, color=colors_d, edgecolor='white', linewidth=0.8, alpha=0.85)
    ax5.set_xticks(range(6)); ax5.set_xticklabels(TONNETZ_SHORT, fontsize=8)
    ax5.set_title('Dimension RMS Energy', fontsize=13, fontweight='bold')
    ax5.set_ylabel('Energy'); ax5.grid(True, linestyle='--', alpha=0.4, axis='y')

    # 6. Interval Strength Bar
    ax6 = fig.add_subplot(gs[4, 0])
    iv_names = ['Fifth', 'Minor 3rd', 'Major 3rd']
    iv_means = [intervals['fifth']['mean'], intervals['minor_third']['mean'], intervals['major_third']['mean']]
    iv_stds = [intervals['fifth']['std'], intervals['minor_third']['std'], intervals['major_third']['std']]
    iv_colors = ['#2196F3', '#F44336', '#4CAF50']
    ax6.bar(iv_names, iv_means, yerr=iv_stds, color=iv_colors, edgecolor='white', linewidth=0.8, capsize=4)
    ax6.set_title('Mean Interval Strength', fontsize=13, fontweight='bold')
    ax6.set_ylabel('Strength'); ax6.grid(True, linestyle='--', alpha=0.4, axis='y')

    # 7. Tonal Tension & Flux
    ax7 = fig.add_subplot(gs[4, 1])
    mag_t = times[:len(tension['magnitude'])]
    ax7.plot(mag_t, tension['magnitude'], color='#FF5722', linewidth=0.7, alpha=0.7, label='Magnitude')
    flux_t = times[:len(tension['flux'])]
    ax7.plot(flux_t, tension['flux'], color='#9C27B0', linewidth=0.7, alpha=0.7, label='Flux')
    ax7.set_title('Tonal Tension & Flux', fontsize=13, fontweight='bold')
    ax7.set_xlabel('Time (s)'); ax7.set_ylabel('Value')
    ax7.legend(fontsize=8); ax7.grid(True, linestyle='--', alpha=0.4)

    # 8. Correlation Matrix
    ax8 = fig.add_subplot(gs[5, 0])
    im8 = ax8.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax8.set_xticks(range(6)); ax8.set_yticks(range(6))
    ax8.set_xticklabels(TONNETZ_SHORT, fontsize=8); ax8.set_yticklabels(TONNETZ_SHORT, fontsize=8)
    ax8.set_title('Tonnetz Correlation Matrix', fontsize=13, fontweight='bold')
    fig.colorbar(im8, ax=ax8, label='Correlation', pad=0.02)

    # 9. Distance Matrix
    ax9 = fig.add_subplot(gs[5, 1])
    im9 = ax9.imshow(dist_matrix, cmap='YlOrRd', aspect='auto')
    ns = dist_matrix.shape[0]
    ax9.set_xticks(range(ns)); ax9.set_yticks(range(ns))
    ax9.set_xticklabels([f'S{i+1}' for i in range(ns)], fontsize=7)
    ax9.set_yticklabels([f'S{i+1}' for i in range(ns)], fontsize=7)
    ax9.set_title('Segment Tonnetz Distance', fontsize=13, fontweight='bold')
    fig.colorbar(im9, ax=ax9, label='Cosine Distance', pad=0.02)

    # 10. Tonnetz Trajectory (5th_x vs 5th_y)
    ax10 = fig.add_subplot(gs[6, 0])
    step = max(1, tonnetz.shape[1] // 2000)
    ax10.scatter(tonnetz[0, ::step], tonnetz[1, ::step], c=np.arange(0, tonnetz.shape[1], step),
                 cmap='viridis', s=3, alpha=0.5)
    ax10.set_title('Tonnetz Trajectory (Fifth plane)', fontsize=13, fontweight='bold')
    ax10.set_xlabel('Fifth x₁'); ax10.set_ylabel('Fifth y₁')
    ax10.grid(True, linestyle='--', alpha=0.4)

    # 11. Tonnetz Trajectory (M3 vs m3)
    ax11 = fig.add_subplot(gs[6, 1])
    m3_s = np.sqrt(tonnetz[2, ::step]**2 + tonnetz[3, ::step]**2)
    M3_s = np.sqrt(tonnetz[4, ::step]**2 + tonnetz[5, ::step]**2)
    ax11.scatter(m3_s, M3_s, c=np.arange(len(m3_s)), cmap='coolwarm', s=3, alpha=0.5)
    ax11.set_title('Minor 3rd vs Major 3rd Strength', fontsize=13, fontweight='bold')
    ax11.set_xlabel('Minor 3rd Strength'); ax11.set_ylabel('Major 3rd Strength')
    ax11.plot([0, max(np.max(m3_s), np.max(M3_s))], [0, max(np.max(m3_s), np.max(M3_s))],
             'k--', linewidth=0.5, alpha=0.3, label='Equal line')
    ax11.legend(fontsize=8); ax11.grid(True, linestyle='--', alpha=0.4)

    # 12. Segment Dominant Intervals
    ax12 = fig.add_subplot(gs[7, 0])
    seg_labels = [f'S{s["segment"]}' for s in segments]
    seg_5 = [s['fifth_strength'] for s in segments]
    seg_m3 = [s['minor3_strength'] for s in segments]
    seg_M3 = [s['major3_strength'] for s in segments]
    x_s = np.arange(len(segments))
    w = 0.25
    ax12.bar(x_s - w, seg_5, w, color='#2196F3', label='Fifth', edgecolor='white')
    ax12.bar(x_s, seg_m3, w, color='#F44336', label='Minor 3rd', edgecolor='white')
    ax12.bar(x_s + w, seg_M3, w, color='#4CAF50', label='Major 3rd', edgecolor='white')
    ax12.set_xticks(x_s); ax12.set_xticklabels(seg_labels, fontsize=7)
    ax12.set_title('Segment Interval Strengths', fontsize=13, fontweight='bold')
    ax12.set_ylabel('Strength'); ax12.legend(fontsize=7)
    ax12.grid(True, linestyle='--', alpha=0.4, axis='y')

    # 13. Complexity Summary
    ax13 = fig.add_subplot(gs[7, 1])
    c_labels = ['Tension', 'Flux', 'Entropy', 'Complexity']
    c_vals = [complexity['mean_tension'], complexity['mean_flux'],
              complexity['normalized_entropy'], complexity['tonal_complexity']]
    c_colors = ['#FF5722', '#9C27B0', '#FF9800', '#2196F3']
    bars13 = ax13.bar(c_labels, c_vals, color=c_colors, edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars13, c_vals):
        ax13.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.003,
                 f'{val:.4f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax13.set_title('Tonal Complexity Metrics', fontsize=13, fontweight='bold')
    ax13.set_ylabel('Value'); ax13.grid(True, linestyle='--', alpha=0.4, axis='y')

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"


# ──────────────────────────────────────────────
# Main Endpoint
# ──────────────────────────────────────────────

@router.post("/tonnetz-analysis")
async def tonnetz_analysis(
    file: UploadFile = File(...),
    hop_length: Optional[int] = Form(512),
    n_segments: Optional[int] = Form(10),
    target_sr: Optional[int] = Form(None),
):
    """
    Tonnetz Analysis.

    Comprehensive tonal centroid analysis including 6-dimensional tonnetz
    features, harmonic interval strengths, tonal tension/flux, dimension
    correlations, segment analysis, and tonal complexity metrics.
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
            n_seg = n_segments or 10

            tonnetz, chroma, times = compute_tonnetz(y_t, sr, hl)
            stats = compute_tonnetz_statistics(tonnetz)
            intervals = compute_interval_strength(tonnetz)
            tension = compute_tonal_tension(tonnetz)
            corr_matrix = compute_tonnetz_correlation(tonnetz)
            segments = compute_tonnetz_segments(tonnetz, times, n_seg)
            dist_matrix, dist_times = compute_tonnetz_distance_matrix(tonnetz, times, n_seg)
            complexity = compute_tonal_complexity(tonnetz, tension)
            interpretation = generate_interpretation(stats, intervals, tension, complexity, segments)
            tonnetz_table = build_tonnetz_table(tonnetz, times, 300)

            # Strip large arrays for JSON
            intervals_json = {k: {kk: vv for kk, vv in v.items() if kk != 'values'} for k, v in intervals.items()}
            tension_json = {k: v for k, v in tension.items() if k not in ('magnitude', 'flux')}

            plot = generate_tonnetz_plots(
                y_t, sr, tonnetz, chroma, times, stats, intervals, tension,
                complexity, segments, dist_matrix, dist_times, corr_matrix, hl
            )

            file_info = {
                'filename': file.filename, 'file_size_mb': round(len(content)/(1024*1024), 2),
                'sample_rate': sr, 'duration_sec': round(len(y_t)/sr, 2),
                'n_samples': len(y_t), 'format': ext.replace('.',''),
                'hop_length': hl, 'n_segments': n_seg,
                'n_frames': int(tonnetz.shape[1]),
                'time_resolution_ms': safe_float(hl / sr * 1000),
            }

            return _to_native({
                'results': {
                    'file_info': file_info,
                    'tonnetz_stats': stats,
                    'interval_strength': intervals_json,
                    'tonal_tension': tension_json,
                    'complexity': complexity,
                    'correlation_matrix': [[safe_float(corr_matrix[i,j]) for j in range(6)] for i in range(6)],
                    'segments': segments,
                    'distance_matrix': [[safe_float(dist_matrix[i,j]) for j in range(dist_matrix.shape[1])] for i in range(dist_matrix.shape[0])],
                    'tonnetz_table': tonnetz_table,
                    'interpretation': interpretation,
                },
                'plot': plot
            })
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ──────────────────────────────────────────────
# from tonnetz_analysis import router as tonnetz_router
# app.include_router(tonnetz_router, prefix="/api/analysis", tags=["Audio Analysis"])
# → POST /api/analysis/tonnetz-analysis
# ──────────────────────────────────────────────
