"""
Audio Segmentation Analysis Backend (FastAPI)
- Structural audio segmentation based on spectral feature changes
- Self-similarity matrix (SSM) from chroma, MFCC, and spectral features
- Novelty curve detection for segment boundaries
- Recurrence/repetition matrix for structural analysis
- Checkerboard kernel-based boundary detection
- Segment labeling via feature clustering (k-means on segment features)
- Per-segment statistics: duration, mean energy, spectral centroid, chroma
- Segment similarity matrix (cosine between segment mean features)
- Temporal structure visualization (ABA, ABAB patterns)
- Configurable segmentation: by novelty peaks, fixed count, or min duration
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional
import numpy as np
import pandas as pd
import io, base64, tempfile, os
from scipy.ndimage import uniform_filter
from scipy.spatial.distance import cdist

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

try:
    import librosa
    import librosa.display
    from sklearn.cluster import KMeans
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

sns.set_theme(style="darkgrid")

router = APIRouter()


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
# Feature Extraction
# ──────────────────────────────────────────────

def extract_features(y, sr, n_fft=2048, hop_length=512):
    """Extract multiple feature sets for segmentation."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop_length)
    spectral = np.vstack([
        librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length),
        librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length),
        librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length),
    ])
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(chroma.shape[1]), sr=sr, hop_length=hop_length)
    return chroma, mfcc, spectral, rms, times


def compute_ssm(features, metric='cosine'):
    """Compute self-similarity matrix."""
    F = features.T  # (n_frames, n_features)
    # Normalize
    norms = np.linalg.norm(F, axis=1, keepdims=True) + 1e-10
    F_norm = F / norms
    ssm = 1.0 - cdist(F_norm, F_norm, metric='cosine')
    ssm = np.nan_to_num(ssm, nan=0.0)
    return ssm


def compute_novelty_curve(ssm, kernel_size=64):
    """Checkerboard kernel novelty detection on SSM."""
    n = ssm.shape[0]
    half = kernel_size // 2
    if half < 2:
        half = 2
    # Build checkerboard kernel
    kernel = np.ones((kernel_size, kernel_size))
    kernel[:half, :half] = -1
    kernel[half:, half:] = -1

    novelty = np.zeros(n)
    for i in range(half, n - half):
        s = max(0, i - half)
        e = min(n, i + half)
        patch = ssm[s:e, s:e]
        ks = patch.shape[0]
        if ks < 4:
            continue
        kh = ks // 2
        k = np.ones((ks, ks))
        k[:kh, :kh] = -1
        k[kh:, kh:] = -1
        novelty[i] = np.sum(patch * k)

    # Normalize
    novelty = np.maximum(0, novelty)
    mx = np.max(novelty)
    if mx > 0:
        novelty = novelty / mx
    return novelty


def detect_boundaries(novelty, times, n_boundaries=None, min_duration=3.0, peak_thresh=0.3):
    """Detect segment boundaries from novelty curve peaks."""
    from scipy.signal import find_peaks

    # Find peaks
    peaks, props = find_peaks(novelty, height=peak_thresh, distance=int(min_duration / (times[1] - times[0] + 1e-10)))

    if n_boundaries is not None and len(peaks) > n_boundaries:
        # Keep top-N by height
        heights = novelty[peaks]
        top_idx = np.argsort(heights)[-n_boundaries:]
        peaks = np.sort(peaks[top_idx])

    boundary_times = times[peaks] if len(peaks) > 0 else np.array([])
    boundary_strengths = novelty[peaks] if len(peaks) > 0 else np.array([])

    return peaks, boundary_times, boundary_strengths


def build_segments(boundary_times, duration, times):
    """Build segment list from boundaries."""
    all_bounds = np.concatenate([[0], boundary_times, [duration]])
    all_bounds = np.unique(np.sort(all_bounds))
    segments = []
    for i in range(len(all_bounds) - 1):
        segments.append({
            'segment': i + 1,
            'start_sec': safe_float(all_bounds[i]),
            'end_sec': safe_float(all_bounds[i + 1]),
            'duration_sec': safe_float(all_bounds[i + 1] - all_bounds[i]),
        })
    return segments


def label_segments(segments, chroma, mfcc, rms, times, sr, hop_length, n_labels=None):
    """Cluster segment features for labeling."""
    seg_features = []
    for s in segments:
        mask = (times >= s['start_sec']) & (times < s['end_sec'])
        if np.sum(mask) == 0:
            seg_features.append(np.zeros(13 + 12 + 1))
            continue
        m_mfcc = np.mean(mfcc[:, mask], axis=1)
        m_chroma = np.mean(chroma[:, mask], axis=1)
        m_rms = np.mean(rms[mask[:len(rms)]]) if np.sum(mask[:len(rms)]) > 0 else 0
        seg_features.append(np.concatenate([m_mfcc, m_chroma, [m_rms]]))

    seg_features = np.array(seg_features)

    # Determine n_clusters
    n_segs = len(segments)
    if n_labels is not None:
        k = min(n_labels, n_segs)
    else:
        k = min(max(2, n_segs // 2), 8, n_segs)

    if k < 2 or n_segs < 2:
        for i, s in enumerate(segments):
            s['label'] = 'A'
        return segments, seg_features

    try:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        cluster_ids = km.fit_predict(seg_features)
        label_map = {}
        counter = 0
        for cid in cluster_ids:
            if cid not in label_map:
                label_map[cid] = chr(65 + counter % 26)  # A, B, C...
                counter += 1
        for i, s in enumerate(segments):
            s['label'] = label_map[cluster_ids[i]]
    except:
        for i, s in enumerate(segments):
            s['label'] = chr(65 + i % 26)

    return segments, seg_features


def enrich_segments(segments, chroma, mfcc, rms, spectral, times, sr):
    """Add per-segment statistics."""
    for s in segments:
        mask = (times >= s['start_sec']) & (times < s['end_sec'])
        n = np.sum(mask)
        if n == 0:
            s.update({'mean_rms_db': -80, 'mean_centroid': 0, 'dominant_chroma': 'C', 'n_frames': 0})
            continue
        seg_rms = rms[mask[:len(rms)]]
        mean_rms = np.mean(seg_rms) if len(seg_rms) > 0 else 0
        mean_rms_db = 20 * np.log10(mean_rms + 1e-10)

        seg_cent = spectral[0, mask[:spectral.shape[1]]]
        mean_cent = np.mean(seg_cent) if len(seg_cent) > 0 else 0

        seg_chroma = chroma[:, mask[:chroma.shape[1]]]
        chroma_means = np.mean(seg_chroma, axis=1) if seg_chroma.shape[1] > 0 else np.zeros(12)
        NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        dominant_chroma = NOTE_NAMES[np.argmax(chroma_means)]

        s.update({
            'mean_rms_db': safe_float(mean_rms_db),
            'mean_centroid': safe_float(mean_cent),
            'dominant_chroma': dominant_chroma,
            'n_frames': int(n),
        })
    return segments


def compute_segment_similarity(seg_features):
    """Cosine similarity between segment feature vectors."""
    norms = np.linalg.norm(seg_features, axis=1, keepdims=True) + 1e-10
    normed = seg_features / norms
    sim = normed @ normed.T
    sim = np.nan_to_num(sim, nan=0.0)
    return sim


def compute_structure_string(segments):
    """Create a structure summary like AABA."""
    return ''.join(s.get('label', '?') for s in segments)


def generate_interpretation(segments, boundary_times, novelty, structure_str, duration):
    lines = []
    n_seg = len(segments)
    n_bound = len(boundary_times)

    lines.append(f"Detected {n_bound} boundaries dividing the signal into {n_seg} segments over {duration:.1f}s.")

    unique_labels = len(set(s.get('label', '?') for s in segments))
    lines.append(f"Structure pattern: {structure_str} — {unique_labels} distinct section types identified.")

    # Duration stats
    durs = [s['duration_sec'] for s in segments]
    if durs:
        lines.append(f"Segment durations: {min(durs):.1f}s – {max(durs):.1f}s (mean: {np.mean(durs):.1f}s).")

    # Repetition
    label_counts = {}
    for s in segments:
        l = s.get('label', '?')
        label_counts[l] = label_counts.get(l, 0) + 1
    repeated = {k: v for k, v in label_counts.items() if v > 1}
    if repeated:
        rep_str = ', '.join([f'{k}×{v}' for k, v in sorted(repeated.items())])
        lines.append(f"Repeating sections: {rep_str} — indicates structural repetition (verse/chorus-like).")
    else:
        lines.append("No repeating sections detected — through-composed or continuous structure.")

    # Novelty
    mean_nov = np.mean(novelty[novelty > 0]) if np.any(novelty > 0) else 0
    lines.append(f"Mean novelty at boundaries: {mean_nov:.3f}.")

    return {'summary': ' '.join(lines[:2]), 'details': lines}


# ──────────────────────────────────────────────
# Plot Generation
# ──────────────────────────────────────────────

def generate_plots(
    y, sr, ssm_chroma, ssm_mfcc, novelty, novelty_times,
    boundary_peaks, boundary_times, segments, seg_sim,
    chroma, rms, times, hop_length, structure_str,
) -> str:

    fig = plt.figure(figsize=(18, 32))
    fig.suptitle('Audio Segmentation Analysis', fontsize=18, fontweight='bold')
    gs = gridspec.GridSpec(8, 2, figure=fig, hspace=0.55, wspace=0.3)

    # Segment colors
    seg_cmap = plt.cm.Set3(np.linspace(0, 1, max(len(segments), 1)))

    # 1. Waveform + boundaries
    ax1 = fig.add_subplot(gs[0, :])
    t = np.arange(len(y)) / sr
    ax1.plot(t, y, color='#90CAF9', linewidth=0.3, alpha=0.4)
    for i, s in enumerate(segments):
        ax1.axvspan(s['start_sec'], s['end_sec'], alpha=0.15, color=seg_cmap[i % len(seg_cmap)])
    for bt in boundary_times:
        ax1.axvline(x=bt, color='#F44336', linewidth=1, alpha=0.8)
    ax1.set_title(f'Waveform + Segment Boundaries ({len(segments)} segments: {structure_str})', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Amplitude')
    ax1.grid(True, linestyle='--', alpha=0.4)

    # 2. Novelty Curve
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(novelty_times[:len(novelty)], novelty, color='#FF5722', linewidth=0.8)
    ax2.fill_between(novelty_times[:len(novelty)], novelty, alpha=0.15, color='#FF5722')
    if len(boundary_peaks) > 0:
        bp_valid = boundary_peaks[boundary_peaks < len(novelty)]
        ax2.scatter(novelty_times[bp_valid], novelty[bp_valid],
                    color='#F44336', s=40, zorder=5, edgecolors='white', linewidth=0.5)
    ax2.set_title('Novelty Curve (segment boundary candidates)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('Novelty')
    ax2.grid(True, linestyle='--', alpha=0.4)

    # 3. SSM (Chroma)
    ax3 = fig.add_subplot(gs[2, 0])
    im3 = ax3.imshow(ssm_chroma, aspect='auto', origin='lower', cmap='magma',
                      extent=[0, times[-1], 0, times[-1]])
    for bt in boundary_times:
        ax3.axhline(y=bt, color='white', linewidth=0.3, alpha=0.5)
        ax3.axvline(x=bt, color='white', linewidth=0.3, alpha=0.5)
    ax3.set_title('Self-Similarity (Chroma)', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Time (s)'); ax3.set_ylabel('Time (s)')
    fig.colorbar(im3, ax=ax3, label='Similarity', pad=0.02)

    # 4. SSM (MFCC)
    ax4 = fig.add_subplot(gs[2, 1])
    im4 = ax4.imshow(ssm_mfcc, aspect='auto', origin='lower', cmap='magma',
                      extent=[0, times[-1], 0, times[-1]])
    for bt in boundary_times:
        ax4.axhline(y=bt, color='white', linewidth=0.3, alpha=0.5)
        ax4.axvline(x=bt, color='white', linewidth=0.3, alpha=0.5)
    ax4.set_title('Self-Similarity (MFCC)', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Time (s)'); ax4.set_ylabel('Time (s)')
    fig.colorbar(im4, ax=ax4, label='Similarity', pad=0.02)

    # 5. Chromagram + boundaries
    ax5 = fig.add_subplot(gs[3, :])
    librosa.display.specshow(chroma, x_axis='time', y_axis='chroma', sr=sr,
                              hop_length=hop_length, ax=ax5, cmap='coolwarm')
    for bt in boundary_times:
        ax5.axvline(x=bt, color='white', linewidth=1, alpha=0.7)
    ax5.set_title('Chromagram + Boundaries', fontsize=13, fontweight='bold')

    # 6. Segment Duration Bar
    ax6 = fig.add_subplot(gs[4, 0])
    seg_labels = [f'{s.get("label","?")}{s["segment"]}' for s in segments]
    seg_durs = [s['duration_sec'] for s in segments]
    seg_colors = [seg_cmap[i % len(seg_cmap)] for i in range(len(segments))]
    ax6.bar(range(len(segments)), seg_durs, color=seg_colors, edgecolor='white', linewidth=0.8)
    ax6.set_xticks(range(len(segments)))
    ax6.set_xticklabels(seg_labels, fontsize=8)
    ax6.set_title('Segment Durations', fontsize=13, fontweight='bold')
    ax6.set_ylabel('Duration (s)'); ax6.grid(True, linestyle='--', alpha=0.4, axis='y')

    # 7. Segment Energy
    ax7 = fig.add_subplot(gs[4, 1])
    seg_rms = [s.get('mean_rms_db', -80) for s in segments]
    ax7.bar(range(len(segments)), seg_rms, color=seg_colors, edgecolor='white', linewidth=0.8)
    ax7.set_xticks(range(len(segments)))
    ax7.set_xticklabels(seg_labels, fontsize=8)
    ax7.set_title('Segment Mean Energy (dB)', fontsize=13, fontweight='bold')
    ax7.set_ylabel('RMS (dB)'); ax7.grid(True, linestyle='--', alpha=0.4, axis='y')

    # 8. Segment Similarity Matrix
    ax8 = fig.add_subplot(gs[5, 0])
    if seg_sim is not None and seg_sim.shape[0] > 1:
        im8 = ax8.imshow(seg_sim, aspect='auto', cmap='YlGnBu', vmin=0, vmax=1)
        ax8.set_xticks(range(len(segments)))
        ax8.set_yticks(range(len(segments)))
        ax8.set_xticklabels(seg_labels, fontsize=7)
        ax8.set_yticklabels(seg_labels, fontsize=7)
        fig.colorbar(im8, ax=ax8, label='Cosine Similarity', pad=0.02)
    ax8.set_title('Segment Similarity Matrix', fontsize=13, fontweight='bold')

    # 9. Structure Timeline
    ax9 = fig.add_subplot(gs[5, 1])
    label_colors = {}
    c_idx = 0
    palette = plt.cm.Set2(np.linspace(0, 1, 8))
    for s in segments:
        l = s.get('label', '?')
        if l not in label_colors:
            label_colors[l] = palette[c_idx % len(palette)]
            c_idx += 1
        ax9.barh(0, s['duration_sec'], left=s['start_sec'],
                 color=label_colors[l], edgecolor='white', linewidth=0.5, height=0.6)
        mid = s['start_sec'] + s['duration_sec'] / 2
        ax9.text(mid, 0, l, ha='center', va='center', fontsize=10, fontweight='bold')
    ax9.set_title(f'Structure: {structure_str}', fontsize=13, fontweight='bold')
    ax9.set_xlabel('Time (s)'); ax9.set_yticks([])
    ax9.set_ylim(-0.5, 0.5)

    # 10. Segment Centroid
    ax10 = fig.add_subplot(gs[6, 0])
    seg_cent = [s.get('mean_centroid', 0) for s in segments]
    ax10.bar(range(len(segments)), seg_cent, color=seg_colors, edgecolor='white', linewidth=0.8)
    ax10.set_xticks(range(len(segments)))
    ax10.set_xticklabels(seg_labels, fontsize=8)
    ax10.set_title('Segment Spectral Centroid (Hz)', fontsize=13, fontweight='bold')
    ax10.set_ylabel('Hz'); ax10.grid(True, linestyle='--', alpha=0.4, axis='y')

    # 11. Label Distribution
    ax11 = fig.add_subplot(gs[6, 1])
    label_counts = {}
    for s in segments:
        l = s.get('label', '?')
        label_counts[l] = label_counts.get(l, 0) + s['duration_sec']
    l_names = list(label_counts.keys())
    l_vals = [label_counts[k] for k in l_names]
    l_cols = [label_colors.get(k, '#607D8B') for k in l_names]
    ax11.bar(l_names, l_vals, color=l_cols, edgecolor='white', linewidth=0.8)
    for bar, val in zip(ax11.patches, l_vals):
        ax11.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.1,
                 f'{val:.1f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax11.set_title('Section Duration by Label', fontsize=13, fontweight='bold')
    ax11.set_ylabel('Total Duration (s)'); ax11.grid(True, linestyle='--', alpha=0.4, axis='y')

    # 12. Zoomed
    ax12 = fig.add_subplot(gs[7, :])
    zoom = min(10.0, len(y) / sr)
    mask_z = t <= zoom
    ax12.plot(t[mask_z], y[mask_z], color='#90CAF9', linewidth=0.5, alpha=0.5)
    for s in segments:
        if s['start_sec'] < zoom:
            ax12.axvspan(s['start_sec'], min(s['end_sec'], zoom), alpha=0.15,
                         color=label_colors.get(s.get('label', '?'), '#607D8B'))
    for bt in boundary_times:
        if bt < zoom:
            ax12.axvline(x=bt, color='#F44336', linewidth=1.5, alpha=0.8)
    ax12.set_title(f'Zoomed (0–{zoom:.0f}s)', fontsize=13, fontweight='bold')
    ax12.set_xlabel('Time (s)'); ax12.set_ylabel('Amplitude')
    ax12.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"


# ──────────────────────────────────────────────
# Endpoint
# ──────────────────────────────────────────────

@router.post("/audio-segmentation")
async def audio_segmentation(
    file: UploadFile = File(...),
    hop_length: Optional[int] = Form(512),
    kernel_size: Optional[int] = Form(64),
    peak_thresh: Optional[float] = Form(0.3),
    min_duration: Optional[float] = Form(3.0),
    n_boundaries: Optional[int] = Form(None),
    n_labels: Optional[int] = Form(None),
    target_sr: Optional[int] = Form(None),
):
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
            ks = kernel_size or 64
            pt = peak_thresh if peak_thresh is not None else 0.3
            md = min_duration if min_duration is not None else 3.0
            nb = n_boundaries
            nl = n_labels
            duration = len(y) / sr

            chroma, mfcc, spectral, rms, times = extract_features(y, sr, 2048, hl)

            # Downsample features for SSM if too large
            max_ssm = 500
            if chroma.shape[1] > max_ssm:
                step = chroma.shape[1] // max_ssm
                chroma_ds = chroma[:, ::step]
                mfcc_ds = mfcc[:, ::step]
            else:
                chroma_ds = chroma
                mfcc_ds = mfcc

            ssm_chroma = compute_ssm(chroma_ds)
            ssm_mfcc = compute_ssm(mfcc_ds)

            # Novelty from combined features
            combined = np.vstack([chroma, mfcc])
            if combined.shape[1] > max_ssm:
                combined_ds = combined[:, ::step]
            else:
                combined_ds = combined
            ssm_combined = compute_ssm(combined_ds)
            novelty = compute_novelty_curve(ssm_combined, ks)

            # Map novelty back to full timeline
            nov_times = np.linspace(0, duration, len(novelty))

            boundary_peaks, boundary_times, boundary_strengths = detect_boundaries(
                novelty, nov_times, nb, md, pt
            )

            segments = build_segments(boundary_times, duration, times)
            segments, seg_features = label_segments(segments, chroma, mfcc, rms, times, sr, hl, nl)
            segments = enrich_segments(segments, chroma, mfcc, rms, spectral, times, sr)
            seg_sim = compute_segment_similarity(seg_features)
            structure_str = compute_structure_string(segments)
            interpretation = generate_interpretation(segments, boundary_times, novelty, structure_str, duration)

            plot = generate_plots(
                y, sr, ssm_chroma, ssm_mfcc, novelty, nov_times,
                boundary_peaks, boundary_times, segments, seg_sim,
                chroma, rms, times, hl, structure_str,
            )

            file_info = {
                'filename': file.filename, 'file_size_mb': round(len(content)/(1024*1024), 2),
                'sample_rate': sr, 'duration_sec': round(duration, 2),
                'n_samples': len(y), 'format': ext.replace('.',''),
                'hop_length': hl, 'kernel_size': ks, 'peak_thresh': pt,
                'min_duration': md, 'n_boundaries': nb, 'n_labels': nl,
            }

            seg_sim_list = [[safe_float(seg_sim[i, j]) for j in range(seg_sim.shape[1])] for i in range(seg_sim.shape[0])]

            return _to_native({
                'results': {
                    'file_info': file_info,
                    'n_segments': len(segments),
                    'n_boundaries': len(boundary_times),
                    'structure': structure_str,
                    'boundary_times': [safe_float(b) for b in boundary_times],
                    'boundary_strengths': [safe_float(b) for b in boundary_strengths],
                    'segments': segments,
                    'segment_similarity': seg_sim_list,
                    'interpretation': interpretation,
                },
                'plot': plot
            })
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# from audio_segmentation import router as seg_router
# app.include_router(seg_router, prefix="/api/analysis", tags=["Audio Analysis"])
# → POST /api/analysis/audio-segmentation
