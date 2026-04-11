"""
Chromagram Analysis Backend (FastAPI)
- Comprehensive pitch class / chroma feature analysis
- STFT-based, CQT-based, and CENS chromagrams
- Key detection and mode estimation (major/minor)
- Chord progression analysis via template matching
- Pitch class distribution, strength, and dominance
- Harmonic summary and tonal stability (tonnetz)
- Temporal chroma dynamics, segment-wise analysis
- Chroma self-similarity matrix for structure analysis
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

PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Major / minor profiles (Krumhansl-Kessler)
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])

# Simple chord templates (root-relative semitone patterns)
CHORD_TEMPLATES = {
    'maj': [0, 4, 7], 'min': [0, 3, 7], 'dim': [0, 3, 6],
    'aug': [0, 4, 8], 'sus2': [0, 2, 7], 'sus4': [0, 5, 7],
    '7': [0, 4, 7, 10], 'maj7': [0, 4, 7, 11], 'min7': [0, 3, 7, 10],
}


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
# Chromagram Functions
# ──────────────────────────────────────────────

def compute_chromagrams(y, sr, n_fft=2048, hop_length=512):
    """Compute STFT, CQT, and CENS chromagrams."""
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)
    chroma_cqt = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length)
    chroma_cens = librosa.feature.chroma_cens(y=y, sr=sr, hop_length=hop_length)
    times = librosa.frames_to_time(np.arange(chroma_stft.shape[1]), sr=sr, hop_length=hop_length)
    return chroma_stft, chroma_cqt, chroma_cens, times


def compute_pitch_class_stats(chroma):
    """Per-pitch-class statistics."""
    stats = []
    mean_energies = np.mean(chroma, axis=1)
    total = np.sum(mean_energies) + 1e-10
    for i in range(12):
        stats.append({
            'pitch_class': PITCH_CLASSES[i],
            'mean_energy': safe_float(mean_energies[i]),
            'energy_percent': safe_float(mean_energies[i] / total * 100),
            'std_energy': safe_float(np.std(chroma[i])),
            'max_energy': safe_float(np.max(chroma[i])),
            'median_energy': safe_float(np.median(chroma[i])),
        })
    return stats


def detect_key(chroma):
    """Detect musical key using Krumhansl-Kessler profiles."""
    mean_chroma = np.mean(chroma, axis=1)
    best_corr = -2
    best_key = 'C'
    best_mode = 'major'

    for shift in range(12):
        rotated = np.roll(mean_chroma, -shift)
        corr_maj = np.corrcoef(rotated, MAJOR_PROFILE)[0, 1]
        corr_min = np.corrcoef(rotated, MINOR_PROFILE)[0, 1]

        if corr_maj > best_corr:
            best_corr = corr_maj
            best_key = PITCH_CLASSES[shift]
            best_mode = 'major'
        if corr_min > best_corr:
            best_corr = corr_min
            best_key = PITCH_CLASSES[shift]
            best_mode = 'minor'

    return {
        'key': best_key,
        'mode': best_mode,
        'key_label': f'{best_key} {best_mode}',
        'confidence': safe_float(best_corr),
    }


def detect_chords_per_frame(chroma, n_frames_per_chord=10):
    """Simple chord detection by template matching every N frames."""
    n_frames = chroma.shape[1]
    step = max(1, n_frames_per_chord)
    chords = []

    for start in range(0, n_frames, step):
        end = min(start + step, n_frames)
        segment = np.mean(chroma[:, start:end], axis=1)
        segment = segment / (np.max(segment) + 1e-10)

        best_score = -1
        best_chord = 'N'

        for root in range(12):
            for ctype, intervals in CHORD_TEMPLATES.items():
                template = np.zeros(12)
                for iv in intervals:
                    template[(root + iv) % 12] = 1.0
                score = np.dot(segment, template) / (np.linalg.norm(segment) * np.linalg.norm(template) + 1e-10)
                if score > best_score:
                    best_score = score
                    best_chord = f'{PITCH_CLASSES[root]}{ctype}'

        chords.append({
            'frame_start': int(start),
            'frame_end': int(end),
            'chord': best_chord,
            'confidence': safe_float(best_score),
        })

    return chords


def compute_tonnetz(y, sr):
    """Compute tonal centroid features (tonnetz)."""
    tonnetz = librosa.feature.tonnetz(y=y, sr=sr)
    labels = ['Fifth (x1)', 'Fifth (y1)', 'Minor 3rd (x2)', 'Minor 3rd (y2)', 'Major 3rd (x3)', 'Major 3rd (y3)']
    stats = []
    for i in range(min(6, tonnetz.shape[0])):
        stats.append({
            'dimension': labels[i] if i < len(labels) else f'Dim {i}',
            'mean': safe_float(np.mean(tonnetz[i])),
            'std': safe_float(np.std(tonnetz[i])),
        })
    return tonnetz, stats


def compute_chroma_self_similarity(chroma, times, n_segments=20):
    """Compute chroma-based self-similarity matrix."""
    n_frames = chroma.shape[1]
    step = max(1, n_frames // n_segments)
    seg_means = []
    seg_times = []

    for i in range(n_segments):
        start = i * step
        end = min(start + step, n_frames)
        if start >= n_frames:
            break
        seg_means.append(np.mean(chroma[:, start:end], axis=1))
        seg_times.append(safe_float(times[start]))

    n = len(seg_means)
    sim_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ni = np.linalg.norm(seg_means[i])
            nj = np.linalg.norm(seg_means[j])
            if ni > 0 and nj > 0:
                sim_matrix[i, j] = np.dot(seg_means[i], seg_means[j]) / (ni * nj)

    return sim_matrix, seg_times


def compute_tonal_stability(chroma):
    """Measure how stable the tonal center is over time."""
    # Frame-wise dominant pitch class
    dominant_per_frame = np.argmax(chroma, axis=0)
    # Mode (most common dominant)
    from collections import Counter
    counts = Counter(dominant_per_frame.tolist())
    most_common_pc, most_common_count = counts.most_common(1)[0]
    stability = most_common_count / (len(dominant_per_frame) + 1e-10)

    # Chroma flux
    chroma_diff = np.diff(chroma, axis=1)
    chroma_flux = np.sqrt(np.mean(chroma_diff ** 2, axis=0))
    mean_flux = safe_float(np.mean(chroma_flux))

    return {
        'tonal_stability': safe_float(stability),
        'dominant_pitch_class': PITCH_CLASSES[most_common_pc],
        'dominant_percent': safe_float(stability * 100),
        'chroma_flux_mean': mean_flux,
        'chroma_flux_std': safe_float(np.std(chroma_flux)),
    }


def compute_chroma_segments(chroma, times, n_segments=10):
    """Per-segment chroma statistics."""
    n = chroma.shape[1]
    seg_len = max(1, n // n_segments)
    segments = []

    for i in range(n_segments):
        start = i * seg_len
        end = min(start + seg_len, n)
        if start >= n:
            break
        seg = chroma[:, start:end]
        mean_energy = np.mean(seg, axis=1)
        dom_idx = int(np.argmax(mean_energy))
        segments.append({
            'segment': i + 1,
            'start_sec': safe_float(times[start]),
            'end_sec': safe_float(times[min(end - 1, n - 1)]),
            'dominant_pc': PITCH_CLASSES[dom_idx],
            'dominant_energy': safe_float(mean_energy[dom_idx]),
            'chroma_strength': safe_float(np.max(mean_energy) - np.min(mean_energy)),
        })
    return segments


def generate_interpretation(key_info, pc_stats, tonal, chords, segments):
    """Generate human-readable interpretation."""
    lines = []

    # Key
    lines.append(f"Detected key: {key_info['key_label']} (confidence: {key_info['confidence']:.3f}).")

    # Tonal stability
    ts = tonal['tonal_stability']
    if ts > 0.4:
        lines.append(f"High tonal stability ({ts:.1%}) — pitch class {tonal['dominant_pitch_class']} dominates {tonal['dominant_percent']:.1f}% of frames.")
    elif ts > 0.2:
        lines.append(f"Moderate tonal stability ({ts:.1%}) — some modulation or pitch variation.")
    else:
        lines.append(f"Low tonal stability ({ts:.1%}) — highly atonal or noisy signal with no clear tonal center.")

    # Chroma flux
    flux = tonal['chroma_flux_mean']
    if flux > 0.3:
        lines.append(f"High chroma flux ({flux:.3f}) — rapid harmonic changes, possibly fast chord progressions.")
    elif flux > 0.1:
        lines.append(f"Moderate chroma flux ({flux:.3f}) — normal harmonic movement.")
    else:
        lines.append(f"Low chroma flux ({flux:.3f}) — harmonically stable or sustained tones.")

    # Pitch class distribution
    sorted_pcs = sorted(pc_stats, key=lambda x: x['energy_percent'], reverse=True)
    top3 = ', '.join([f"{p['pitch_class']} ({p['energy_percent']:.1f}%)" for p in sorted_pcs[:3]])
    lines.append(f"Strongest pitch classes: {top3}.")

    # Chord summary
    if chords:
        chord_names = [c['chord'] for c in chords]
        from collections import Counter
        common = Counter(chord_names).most_common(3)
        chord_str = ', '.join([f"{name} ({cnt}×)" for name, cnt in common])
        lines.append(f"Most frequent chords: {chord_str}.")

    # Segment variation
    seg_pcs = [s['dominant_pc'] for s in segments]
    unique_pcs = len(set(seg_pcs))
    if unique_pcs <= 2:
        lines.append(f"Signal stays within {unique_pcs} pitch class(es) across segments — very focused tonality.")
    elif unique_pcs <= 5:
        lines.append(f"Signal uses {unique_pcs} different dominant pitch classes across segments.")
    else:
        lines.append(f"Signal uses {unique_pcs} different pitch classes — wide tonal range or atonal content.")

    return {
        'summary': ' '.join(lines[:2]),
        'details': lines,
    }


# ──────────────────────────────────────────────
# Plot Generation
# ──────────────────────────────────────────────

def generate_chroma_plots(
    y, sr, chroma_stft, chroma_cqt, chroma_cens, times,
    pc_stats, key_info, tonnetz, sim_matrix, sim_times,
    tonal, chords, hop_length,
) -> str:

    fig = plt.figure(figsize=(18, 30))
    fig.suptitle('Chromagram Analysis', fontsize=18, fontweight='bold')
    gs = gridspec.GridSpec(7, 2, figure=fig, hspace=0.55, wspace=0.3)

    # 1. Waveform
    ax1 = fig.add_subplot(gs[0, :])
    t = np.arange(len(y)) / sr
    ax1.plot(t, y, color='#2196F3', linewidth=0.3, alpha=0.6)
    ax1.set_title('Time Domain — Waveform', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Amplitude')
    ax1.grid(True, linestyle='--', alpha=0.4)

    # 2. STFT Chromagram
    ax2 = fig.add_subplot(gs[1, :])
    img2 = librosa.display.specshow(chroma_stft, x_axis='time', y_axis='chroma', sr=sr,
                                     hop_length=hop_length, ax=ax2, cmap='YlOrRd')
    ax2.set_title(f'STFT Chromagram — Detected Key: {key_info["key_label"]} (conf: {key_info["confidence"]:.3f})',
                  fontsize=13, fontweight='bold')
    fig.colorbar(img2, ax=ax2, label='Energy', pad=0.02)

    # 3. CQT Chromagram
    ax3 = fig.add_subplot(gs[2, 0])
    img3 = librosa.display.specshow(chroma_cqt, x_axis='time', y_axis='chroma', sr=sr,
                                     hop_length=hop_length, ax=ax3, cmap='BuPu')
    ax3.set_title('CQT Chromagram', fontsize=13, fontweight='bold')
    fig.colorbar(img3, ax=ax3, label='Energy', pad=0.02)

    # 4. CENS Chromagram
    ax4 = fig.add_subplot(gs[2, 1])
    img4 = librosa.display.specshow(chroma_cens, x_axis='time', y_axis='chroma', sr=sr,
                                     hop_length=hop_length, ax=ax4, cmap='Greens')
    ax4.set_title('CENS Chromagram (normalized)', fontsize=13, fontweight='bold')
    fig.colorbar(img4, ax=ax4, label='Energy', pad=0.02)

    # 5. Pitch Class Distribution
    ax5 = fig.add_subplot(gs[3, 0])
    pcs = [p['pitch_class'] for p in pc_stats]
    energies = [p['energy_percent'] for p in pc_stats]
    max_e = max(energies)
    colors_pc = ['#F44336' if e == max_e else '#FF9800' if e > np.mean(energies) else '#4CAF50' for e in energies]
    bars5 = ax5.bar(pcs, energies, color=colors_pc, edgecolor='white', linewidth=0.8)
    ax5.set_title('Pitch Class Distribution (%)', fontsize=13, fontweight='bold')
    ax5.set_ylabel('Energy %')
    ax5.grid(True, linestyle='--', alpha=0.4, axis='y')
    for bar, val in zip(bars5, energies):
        if val > np.mean(energies):
            ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # 6. Tonnetz
    ax6 = fig.add_subplot(gs[3, 1])
    img6 = librosa.display.specshow(tonnetz, x_axis='time', sr=sr, hop_length=hop_length,
                                     ax=ax6, cmap='coolwarm')
    ax6.set_title('Tonnetz (Tonal Centroids)', fontsize=13, fontweight='bold')
    ax6.set_ylabel('Dimension')
    fig.colorbar(img6, ax=ax6, label='Value', pad=0.02)

    # 7. Self-Similarity Matrix
    ax7 = fig.add_subplot(gs[4, 0])
    im7 = ax7.imshow(sim_matrix, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto', origin='lower')
    n_sim = len(sim_times)
    tick_step = max(1, n_sim // 6)
    ax7.set_xticks(range(0, n_sim, tick_step))
    ax7.set_yticks(range(0, n_sim, tick_step))
    ax7.set_xticklabels([f'{sim_times[i]:.1f}s' for i in range(0, n_sim, tick_step)], fontsize=7)
    ax7.set_yticklabels([f'{sim_times[i]:.1f}s' for i in range(0, n_sim, tick_step)], fontsize=7)
    ax7.set_title('Chroma Self-Similarity', fontsize=13, fontweight='bold')
    fig.colorbar(im7, ax=ax7, label='Cosine Similarity', pad=0.02)

    # 8. Chroma Flux
    ax8 = fig.add_subplot(gs[4, 1])
    chroma_diff = np.diff(chroma_stft, axis=1)
    chroma_flux = np.sqrt(np.mean(chroma_diff ** 2, axis=0))
    flux_times = times[:len(chroma_flux)]
    ax8.plot(flux_times, chroma_flux, color='#FF5722', linewidth=0.7)
    ax8.fill_between(flux_times, chroma_flux, alpha=0.15, color='#FF5722')
    ax8.axhline(y=tonal['chroma_flux_mean'], color='#F44336', linestyle='--', linewidth=0.8,
                label=f'Mean: {tonal["chroma_flux_mean"]:.4f}')
    ax8.set_title('Chroma Flux (Harmonic Change)', fontsize=13, fontweight='bold')
    ax8.set_xlabel('Time (s)'); ax8.set_ylabel('Flux')
    ax8.legend(fontsize=8); ax8.grid(True, linestyle='--', alpha=0.4)

    # 9. Chord Timeline
    ax9 = fig.add_subplot(gs[5, :])
    if chords:
        chord_labels = [c['chord'] for c in chords]
        unique_chords = list(dict.fromkeys(chord_labels))
        chord_to_idx = {c: i for i, c in enumerate(unique_chords)}
        chord_y = [chord_to_idx[c] for c in chord_labels]
        chord_x = np.arange(len(chord_labels))
        colors_chord = plt.cm.Set3(np.linspace(0, 1, len(unique_chords)))
        for i, (x, cy) in enumerate(zip(chord_x, chord_y)):
            ax9.barh(cy, 1, left=x, color=colors_chord[cy], edgecolor='white', linewidth=0.3)
        ax9.set_yticks(range(len(unique_chords)))
        ax9.set_yticklabels(unique_chords, fontsize=7)
        ax9.set_xlabel('Chord Index')
    ax9.set_title('Chord Progression Timeline', fontsize=13, fontweight='bold')

    # 10. Tonal Stability Bar
    ax10 = fig.add_subplot(gs[6, 0])
    labels_t = ['Tonal\nStability', 'Key\nConfidence', 'Chroma\nFlux', 'Chroma\nStrength']
    chroma_strength = safe_float(np.max(np.mean(chroma_stft, axis=1)) - np.min(np.mean(chroma_stft, axis=1)))
    vals_t = [tonal['tonal_stability'], key_info['confidence'],
              min(1, tonal['chroma_flux_mean'] * 3), min(1, chroma_strength * 2)]
    colors_t = ['#4CAF50', '#2196F3', '#FF5722', '#9C27B0']
    ax10.barh(labels_t, vals_t, color=colors_t, edgecolor='white', linewidth=0.8)
    ax10.set_xlim(0, 1)
    ax10.set_title('Tonal Summary', fontsize=13, fontweight='bold')
    ax10.set_xlabel('Score (0–1)')
    ax10.grid(True, linestyle='--', alpha=0.4, axis='x')

    # 11. Dominant Pitch per Segment
    ax11 = fig.add_subplot(gs[6, 1])
    dom_per_frame = np.argmax(chroma_stft, axis=0)
    ax11.scatter(times[:len(dom_per_frame)], dom_per_frame, s=1, alpha=0.3, color='#9C27B0')
    ax11.set_yticks(range(12))
    ax11.set_yticklabels(PITCH_CLASSES, fontsize=8)
    ax11.set_title('Dominant Pitch Class over Time', fontsize=13, fontweight='bold')
    ax11.set_xlabel('Time (s)'); ax11.set_ylabel('Pitch Class')
    ax11.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"


# ──────────────────────────────────────────────
# Main Endpoint
# ──────────────────────────────────────────────

@router.post("/chromagram-analysis")
async def chromagram_analysis(
    file: UploadFile = File(...),
    n_fft: Optional[int] = Form(2048),
    hop_length: Optional[int] = Form(512),
    n_segments: Optional[int] = Form(10),
    n_sim_segments: Optional[int] = Form(20),
    frames_per_chord: Optional[int] = Form(10),
    target_sr: Optional[int] = Form(None),
):
    """
    Chromagram Analysis.

    Comprehensive pitch class and tonal analysis including STFT/CQT/CENS
    chromagrams, key detection, chord estimation, tonnetz, self-similarity,
    and tonal stability metrics.
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

            nf = n_fft or 2048; hl = hop_length or 512
            n_seg = n_segments or 10; n_sim = n_sim_segments or 20
            fpc = frames_per_chord or 10

            chroma_stft, chroma_cqt, chroma_cens, times = compute_chromagrams(y_t, sr, nf, hl)
            pc_stats = compute_pitch_class_stats(chroma_stft)
            key_info = detect_key(chroma_stft)
            chords = detect_chords_per_frame(chroma_stft, fpc)
            tonnetz_feat, tonnetz_stats = compute_tonnetz(y_t, sr)
            sim_matrix, sim_times = compute_chroma_self_similarity(chroma_stft, times, n_sim)
            tonal = compute_tonal_stability(chroma_stft)
            segments = compute_chroma_segments(chroma_stft, times, n_seg)
            interpretation = generate_interpretation(key_info, pc_stats, tonal, chords, segments)

            plot = generate_chroma_plots(
                y_t, sr, chroma_stft, chroma_cqt, chroma_cens, times,
                pc_stats, key_info, tonnetz_feat, sim_matrix, sim_times,
                tonal, chords, hl
            )

            # Chord summary (top 20 for frontend)
            chord_table = chords[:50] if len(chords) > 50 else chords

            file_info = {
                'filename': file.filename, 'file_size_mb': round(len(content)/(1024*1024), 2),
                'sample_rate': sr, 'duration_sec': round(len(y_t)/sr, 2),
                'n_samples': len(y_t), 'format': ext.replace('.',''),
                'n_fft': nf, 'hop_length': hl, 'n_segments': n_seg,
                'n_frames': int(chroma_stft.shape[1]),
                'time_resolution_ms': safe_float(hl/sr*1000),
                'frames_per_chord': fpc,
            }

            return _to_native({
                'results': {
                    'file_info': file_info,
                    'key_detection': key_info,
                    'pitch_class_stats': pc_stats,
                    'tonal_stability': tonal,
                    'tonnetz_stats': tonnetz_stats,
                    'chord_progression': chord_table,
                    'segments': segments,
                    'self_similarity': {
                        'matrix': [[safe_float(sim_matrix[i,j]) for j in range(sim_matrix.shape[1])] for i in range(sim_matrix.shape[0])],
                        'times': sim_times,
                    },
                    'interpretation': interpretation,
                },
                'plot': plot
            })
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ──────────────────────────────────────────────
# from chromagram_analysis import router as chromagram_router
# app.include_router(chromagram_router, prefix="/api/analysis", tags=["Audio Analysis"])
# → POST /api/analysis/chromagram-analysis
# ──────────────────────────────────────────────
