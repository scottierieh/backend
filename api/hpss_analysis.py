"""
Harmonic/Percussive Separation (HPSS) Analysis Backend (FastAPI)
- Median-filtering based harmonic/percussive source separation
- Full spectral analysis of harmonic and percussive components
- Harmonic-to-Percussive Ratio (HPR) global and over time
- Per-component RMS energy, spectral centroid, bandwidth
- Temporal dynamics: energy envelopes, onset density comparison
- Frequency-band energy breakdown for H and P components
- Residual analysis (original minus H+P reconstruction)
- Soft/hard mask comparison
- Segment-wise H/P balance and dominance tracking
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
# HPSS Functions
# ──────────────────────────────────────────────

def perform_hpss(y, sr, n_fft=2048, hop_length=512, margin=1.0):
    """Perform harmonic/percussive source separation."""
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    H, P = librosa.decompose.hpss(D, margin=margin)
    y_h = librosa.istft(H, hop_length=hop_length, length=len(y))
    y_p = librosa.istft(P, hop_length=hop_length, length=len(y))
    y_r = y - y_h - y_p  # residual
    return D, H, P, y_h, y_p, y_r


def compute_component_rms(y_h, y_p, y_r, y, sr, hop_length=512):
    """RMS energy for each component."""
    rms_o = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    rms_h = librosa.feature.rms(y=y_h, hop_length=hop_length)[0]
    rms_p = librosa.feature.rms(y=y_p, hop_length=hop_length)[0]
    rms_r = librosa.feature.rms(y=y_r, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms_o)), sr=sr, hop_length=hop_length)
    return rms_o, rms_h, rms_p, rms_r, times


def compute_hpr(rms_h, rms_p):
    """Harmonic-to-Percussive Ratio over time and global."""
    n = min(len(rms_h), len(rms_p))
    rms_h_t = rms_h[:n]
    rms_p_t = rms_p[:n]

    # Frame-wise HPR (dB)
    hpr_db = 10 * np.log10((rms_h_t ** 2 + 1e-20) / (rms_p_t ** 2 + 1e-20))

    # Global
    global_h_energy = np.sum(rms_h_t ** 2)
    global_p_energy = np.sum(rms_p_t ** 2)
    global_hpr_db = 10 * np.log10((global_h_energy + 1e-20) / (global_p_energy + 1e-20))

    h_pct = global_h_energy / (global_h_energy + global_p_energy + 1e-20) * 100
    p_pct = 100 - h_pct

    return {
        'hpr_db_over_time': hpr_db,
        'global_hpr_db': safe_float(global_hpr_db),
        'harmonic_percent': safe_float(h_pct),
        'percussive_percent': safe_float(p_pct),
        'global_h_rms': safe_float(np.sqrt(np.mean(rms_h_t ** 2))),
        'global_p_rms': safe_float(np.sqrt(np.mean(rms_p_t ** 2))),
    }


def compute_spectral_features(y_h, y_p, y, sr, n_fft=2048, hop_length=512):
    """Spectral centroid and bandwidth for each component."""
    def _feats(sig):
        cent = librosa.feature.spectral_centroid(y=sig, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        bw = librosa.feature.spectral_bandwidth(y=sig, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        return {'centroid_mean': safe_float(np.mean(cent)), 'centroid_std': safe_float(np.std(cent)),
                'bandwidth_mean': safe_float(np.mean(bw)), 'bandwidth_std': safe_float(np.std(bw)),
                'centroid': cent, 'bandwidth': bw}
    f_o = _feats(y)
    f_h = _feats(y_h)
    f_p = _feats(y_p)
    return f_o, f_h, f_p


def compute_band_energy(D_mag, freqs, sr):
    """Energy in frequency bands for a magnitude spectrogram."""
    bands = [
        ('Sub-bass (0-60)', 0, 60), ('Bass (60-250)', 60, 250),
        ('Low-mid (250-500)', 250, 500), ('Mid (500-2k)', 500, 2000),
        ('Upper-mid (2k-4k)', 2000, 4000), ('High (4k-8k)', 4000, 8000),
        ('Very high (8k+)', 8000, sr / 2),
    ]
    results = []
    total = np.sum(D_mag ** 2) + 1e-20
    for name, lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        energy = np.sum(D_mag[mask] ** 2)
        results.append({'band': name, 'energy_percent': safe_float(energy / total * 100)})
    return results


def compute_onset_comparison(y_h, y_p, sr, hop_length=512):
    """Compare onset densities of H and P components."""
    on_h = librosa.onset.onset_detect(y=y_h, sr=sr, hop_length=hop_length)
    on_p = librosa.onset.onset_detect(y=y_p, sr=sr, hop_length=hop_length)
    dur = len(y_h) / sr
    return {
        'harmonic_onsets': len(on_h),
        'percussive_onsets': len(on_p),
        'harmonic_onset_rate': safe_float(len(on_h) / dur),
        'percussive_onset_rate': safe_float(len(on_p) / dur),
    }


def compute_residual_stats(y_r, y, sr):
    """Residual analysis."""
    rms_r = np.sqrt(np.mean(y_r ** 2))
    rms_o = np.sqrt(np.mean(y ** 2))
    snr = 20 * np.log10((rms_o + 1e-10) / (rms_r + 1e-10))
    return {
        'residual_rms': safe_float(rms_r),
        'residual_rms_db': safe_float(20 * np.log10(rms_r + 1e-10)),
        'reconstruction_snr_db': safe_float(snr),
        'residual_percent': safe_float(np.sum(y_r ** 2) / (np.sum(y ** 2) + 1e-20) * 100),
    }


def compute_segments(rms_h, rms_p, times, n_segments=10):
    """Per-segment H/P balance."""
    n = min(len(rms_h), len(rms_p), len(times))
    seg_len = max(1, n // n_segments)
    segments = []
    for i in range(n_segments):
        s = i * seg_len
        e = min(s + seg_len, n)
        if s >= n: break
        h_e = np.mean(rms_h[s:e] ** 2)
        p_e = np.mean(rms_p[s:e] ** 2)
        total = h_e + p_e + 1e-20
        segments.append({
            'segment': i + 1,
            'start_sec': safe_float(times[s]),
            'harmonic_percent': safe_float(h_e / total * 100),
            'percussive_percent': safe_float(p_e / total * 100),
            'dominant': 'Harmonic' if h_e > p_e else 'Percussive',
            'hpr_db': safe_float(10 * np.log10((h_e + 1e-20) / (p_e + 1e-20))),
        })
    return segments


def generate_interpretation(hpr, spectral_h, spectral_p, onsets, residual, segments):
    lines = []
    hp = hpr['harmonic_percent']
    pp = hpr['percussive_percent']
    if hp > 70:
        lines.append(f"Signal is heavily harmonic ({hp:.0f}% / {pp:.0f}%) — tonal content dominates (sustained notes, singing, strings).")
    elif hp > 55:
        lines.append(f"Signal is mostly harmonic ({hp:.0f}% / {pp:.0f}%) — tonal content with some percussive elements.")
    elif pp > 70:
        lines.append(f"Signal is heavily percussive ({pp:.0f}% / {hp:.0f}%) — transient-rich content (drums, clicks, attacks).")
    elif pp > 55:
        lines.append(f"Signal is mostly percussive ({pp:.0f}% / {hp:.0f}%).")
    else:
        lines.append(f"Balanced harmonic/percussive mix ({hp:.0f}% / {pp:.0f}%).")

    lines.append(f"Global HPR: {hpr['global_hpr_db']:.1f} dB (positive = harmonic dominant, negative = percussive dominant).")

    # Spectral
    lines.append(f"Harmonic centroid: {spectral_h['centroid_mean']:.0f} Hz · Percussive centroid: {spectral_p['centroid_mean']:.0f} Hz.")

    # Onsets
    lines.append(f"Percussive component has {onsets['percussive_onsets']} onsets ({onsets['percussive_onset_rate']:.1f}/sec) vs {onsets['harmonic_onsets']} harmonic onsets ({onsets['harmonic_onset_rate']:.1f}/sec).")

    # Residual
    lines.append(f"Reconstruction SNR: {residual['reconstruction_snr_db']:.1f} dB — residual is {residual['residual_percent']:.1f}% of original energy.")

    # Segment variation
    dom_counts = {'Harmonic': 0, 'Percussive': 0}
    for s in segments:
        dom_counts[s['dominant']] += 1
    if dom_counts['Harmonic'] > dom_counts['Percussive'] * 2:
        lines.append("Harmonic dominance is consistent across all segments.")
    elif dom_counts['Percussive'] > dom_counts['Harmonic'] * 2:
        lines.append("Percussive dominance is consistent across all segments.")
    else:
        lines.append(f"H/P balance shifts across segments: {dom_counts['Harmonic']} harmonic-dominant, {dom_counts['Percussive']} percussive-dominant.")

    return {'summary': ' '.join(lines[:2]), 'details': lines}


# ──────────────────────────────────────────────
# Plot Generation
# ──────────────────────────────────────────────

def generate_hpss_plots(
    y, sr, y_h, y_p, y_r, D, H, P,
    rms_o, rms_h, rms_p, rms_r, rms_times,
    hpr, spectral_o, spectral_h, spectral_p,
    band_h, band_p, segments, hop_length
) -> str:

    fig = plt.figure(figsize=(18, 34))
    fig.suptitle('Harmonic/Percussive Separation Analysis', fontsize=18, fontweight='bold')
    gs = gridspec.GridSpec(9, 2, figure=fig, hspace=0.55, wspace=0.3)

    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    H_db = librosa.amplitude_to_db(np.abs(H), ref=np.max)
    P_db = librosa.amplitude_to_db(np.abs(P), ref=np.max)

    # 1. Original Spectrogram
    ax1 = fig.add_subplot(gs[0, :])
    img1 = librosa.display.specshow(S_db, x_axis='time', y_axis='log', sr=sr,
                                     hop_length=hop_length, ax=ax1, cmap='magma')
    ax1.set_title('Original Spectrogram', fontsize=13, fontweight='bold')
    fig.colorbar(img1, ax=ax1, label='dB', pad=0.02)

    # 2. Harmonic Spectrogram
    ax2 = fig.add_subplot(gs[1, 0])
    img2 = librosa.display.specshow(H_db, x_axis='time', y_axis='log', sr=sr,
                                     hop_length=hop_length, ax=ax2, cmap='Greens')
    ax2.set_title(f'Harmonic Component ({hpr["harmonic_percent"]:.0f}%)', fontsize=13, fontweight='bold')
    fig.colorbar(img2, ax=ax2, label='dB', pad=0.02)

    # 3. Percussive Spectrogram
    ax3 = fig.add_subplot(gs[1, 1])
    img3 = librosa.display.specshow(P_db, x_axis='time', y_axis='log', sr=sr,
                                     hop_length=hop_length, ax=ax3, cmap='Oranges')
    ax3.set_title(f'Percussive Component ({hpr["percussive_percent"]:.0f}%)', fontsize=13, fontweight='bold')
    fig.colorbar(img3, ax=ax3, label='dB', pad=0.02)

    # 4. Waveforms
    ax4 = fig.add_subplot(gs[2, :])
    t = np.arange(len(y)) / sr
    ax4.plot(t, y, color='#90CAF9', linewidth=0.3, alpha=0.3, label='Original')
    ax4.plot(t, y_h, color='#4CAF50', linewidth=0.3, alpha=0.5, label='Harmonic')
    ax4.plot(t, y_p, color='#FF5722', linewidth=0.3, alpha=0.5, label='Percussive')
    ax4.set_title('Waveforms (Original / Harmonic / Percussive)', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Time (s)'); ax4.set_ylabel('Amplitude')
    ax4.legend(fontsize=8); ax4.grid(True, linestyle='--', alpha=0.4)

    # 5. RMS Energy Over Time
    ax5 = fig.add_subplot(gs[3, :])
    rms_o_db = 20 * np.log10(rms_o + 1e-10)
    rms_h_db = 20 * np.log10(rms_h + 1e-10)
    rms_p_db = 20 * np.log10(rms_p + 1e-10)
    ax5.plot(rms_times, rms_o_db, color='#2196F3', linewidth=0.8, alpha=0.4, label='Original')
    ax5.plot(rms_times, rms_h_db, color='#4CAF50', linewidth=0.8, label='Harmonic')
    ax5.plot(rms_times, rms_p_db, color='#FF5722', linewidth=0.8, label='Percussive')
    ax5.set_title('RMS Energy (dB) — H vs P vs Original', fontsize=13, fontweight='bold')
    ax5.set_xlabel('Time (s)'); ax5.set_ylabel('RMS (dB)')
    ax5.legend(fontsize=8); ax5.grid(True, linestyle='--', alpha=0.4)

    # 6. HPR Over Time
    ax6 = fig.add_subplot(gs[4, 0])
    hpr_t = rms_times[:len(hpr['hpr_db_over_time'])]
    ax6.plot(hpr_t, hpr['hpr_db_over_time'], color='#9C27B0', linewidth=0.7)
    ax6.fill_between(hpr_t, hpr['hpr_db_over_time'], alpha=0.1, color='#9C27B0')
    ax6.axhline(y=0, color='gray', linewidth=0.5)
    ax6.axhline(y=hpr['global_hpr_db'], color='#F44336', linestyle='--', linewidth=0.8,
                label=f'Global: {hpr["global_hpr_db"]:.1f} dB')
    ax6.set_title('HPR Over Time (>0=Harmonic, <0=Percussive)', fontsize=13, fontweight='bold')
    ax6.set_xlabel('Time (s)'); ax6.set_ylabel('HPR (dB)')
    ax6.legend(fontsize=8); ax6.grid(True, linestyle='--', alpha=0.4)

    # 7. H/P Energy Pie
    ax7 = fig.add_subplot(gs[4, 1])
    pie_vals = [hpr['harmonic_percent'], hpr['percussive_percent']]
    pie_labels = [f'Harmonic\n{hpr["harmonic_percent"]:.1f}%', f'Percussive\n{hpr["percussive_percent"]:.1f}%']
    ax7.pie(pie_vals, labels=pie_labels, colors=['#4CAF50', '#FF5722'], startangle=90,
            autopct='', textprops={'fontsize': 11})
    ax7.set_title('Energy Distribution', fontsize=13, fontweight='bold')

    # 8. Spectral Centroid Comparison
    ax8 = fig.add_subplot(gs[5, 0])
    n_cent = min(len(spectral_h['centroid']), len(spectral_p['centroid']), len(rms_times))
    ax8.plot(rms_times[:n_cent], spectral_h['centroid'][:n_cent], color='#4CAF50', linewidth=0.6, alpha=0.7, label='Harmonic')
    ax8.plot(rms_times[:n_cent], spectral_p['centroid'][:n_cent], color='#FF5722', linewidth=0.6, alpha=0.7, label='Percussive')
    ax8.set_title('Spectral Centroid (H vs P)', fontsize=13, fontweight='bold')
    ax8.set_xlabel('Time (s)'); ax8.set_ylabel('Hz')
    ax8.legend(fontsize=8); ax8.grid(True, linestyle='--', alpha=0.4)

    # 9. Band Energy Comparison
    ax9 = fig.add_subplot(gs[5, 1])
    b_names = [b['band'].split('(')[0].strip() for b in band_h]
    b_h_vals = [b['energy_percent'] for b in band_h]
    b_p_vals = [b['energy_percent'] for b in band_p]
    x_b = np.arange(len(b_names))
    w = 0.35
    ax9.bar(x_b - w/2, b_h_vals, w, color='#4CAF50', label='Harmonic', edgecolor='white')
    ax9.bar(x_b + w/2, b_p_vals, w, color='#FF5722', label='Percussive', edgecolor='white')
    ax9.set_xticks(x_b); ax9.set_xticklabels(b_names, fontsize=7, rotation=30)
    ax9.set_title('Band Energy (H vs P)', fontsize=13, fontweight='bold')
    ax9.set_ylabel('Energy %'); ax9.legend(fontsize=8)
    ax9.grid(True, linestyle='--', alpha=0.4, axis='y')

    # 10. Segment H/P Balance
    ax10 = fig.add_subplot(gs[6, :])
    seg_labels = [f'S{s["segment"]}' for s in segments]
    seg_h = [s['harmonic_percent'] for s in segments]
    seg_p = [s['percussive_percent'] for s in segments]
    x_s = np.arange(len(segments))
    ax10.bar(x_s, seg_h, color='#4CAF50', label='Harmonic', edgecolor='white')
    ax10.bar(x_s, seg_p, bottom=seg_h, color='#FF5722', label='Percussive', edgecolor='white')
    ax10.set_xticks(x_s); ax10.set_xticklabels(seg_labels, fontsize=8)
    ax10.set_title('Segment-wise H/P Balance', fontsize=13, fontweight='bold')
    ax10.set_ylabel('Percent'); ax10.legend(fontsize=8)
    ax10.grid(True, linestyle='--', alpha=0.4, axis='y')

    # 11. Residual Waveform
    ax11 = fig.add_subplot(gs[7, 0])
    ax11.plot(t, y_r, color='#607D8B', linewidth=0.3, alpha=0.6)
    ax11.set_title('Residual (Original − H − P)', fontsize=13, fontweight='bold')
    ax11.set_xlabel('Time (s)'); ax11.set_ylabel('Amplitude')
    ax11.grid(True, linestyle='--', alpha=0.4)

    # 12. Summary Metrics
    ax12 = fig.add_subplot(gs[7, 1])
    m_labels = ['H Energy\n(%)', 'P Energy\n(%)', 'HPR\n(dB)', 'H Centroid\n(Hz/100)', 'P Centroid\n(Hz/100)']
    m_vals = [hpr['harmonic_percent'], hpr['percussive_percent'], hpr['global_hpr_db'],
              spectral_h['centroid_mean'] / 100, spectral_p['centroid_mean'] / 100]
    m_colors = ['#4CAF50', '#FF5722', '#9C27B0', '#4CAF50', '#FF5722']
    m_disp = [f'{hpr["harmonic_percent"]:.1f}%', f'{hpr["percussive_percent"]:.1f}%',
              f'{hpr["global_hpr_db"]:.1f} dB', f'{spectral_h["centroid_mean"]:.0f}',
              f'{spectral_p["centroid_mean"]:.0f}']
    bars12 = ax12.bar(m_labels, m_vals, color=m_colors, edgecolor='white', linewidth=0.8)
    for bar, d in zip(bars12, m_disp):
        ax12.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.3,
                 d, ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax12.set_title('Key Metrics', fontsize=13, fontweight='bold')
    ax12.grid(True, linestyle='--', alpha=0.4, axis='y')

    # 13. Zoomed waveforms
    ax13 = fig.add_subplot(gs[8, :])
    zoom = min(3.0, len(y)/sr)
    mask = t <= zoom
    ax13.plot(t[mask], y_h[mask], color='#4CAF50', linewidth=0.5, alpha=0.7, label='Harmonic')
    ax13.plot(t[mask], y_p[mask], color='#FF5722', linewidth=0.5, alpha=0.7, label='Percussive')
    ax13.set_title(f'Zoomed H/P Waveforms (0–{zoom:.1f}s)', fontsize=13, fontweight='bold')
    ax13.set_xlabel('Time (s)'); ax13.set_ylabel('Amplitude')
    ax13.legend(fontsize=8); ax13.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout(rect=[0, 0.02, 1, 0.96])
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"


# ──────────────────────────────────────────────
# Main Endpoint
# ──────────────────────────────────────────────

@router.post("/hpss-analysis")
async def hpss_analysis(
    file: UploadFile = File(...),
    n_fft: Optional[int] = Form(2048),
    hop_length: Optional[int] = Form(512),
    margin: Optional[float] = Form(1.0),
    n_segments: Optional[int] = Form(10),
    target_sr: Optional[int] = Form(None),
):
    """
    Harmonic/Percussive Source Separation Analysis.

    Separates audio into harmonic and percussive components via HPSS,
    then analyzes energy balance, spectral features, temporal dynamics,
    band energy, onset density, residual, and segment-wise H/P ratio.
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
            if len(y) > sr * 60: y = y[:sr * 60]

            nf = n_fft or 2048; hl = hop_length or 512
            mg = margin if margin is not None else 1.0
            n_seg = n_segments or 10

            D, H, P, y_h, y_p, y_r = perform_hpss(y, sr, nf, hl, mg)
            rms_o, rms_h, rms_p, rms_r, rms_times = compute_component_rms(y_h, y_p, y_r, y, sr, hl)
            hpr = compute_hpr(rms_h, rms_p)
            spectral_o, spectral_h, spectral_p = compute_spectral_features(y_h, y_p, y, sr, nf, hl)
            freqs = librosa.fft_frequencies(sr=sr, n_fft=nf)
            band_h = compute_band_energy(np.abs(H), freqs, sr)
            band_p = compute_band_energy(np.abs(P), freqs, sr)
            onsets = compute_onset_comparison(y_h, y_p, sr, hl)
            residual = compute_residual_stats(y_r, y, sr)
            segments = compute_segments(rms_h, rms_p, rms_times, n_seg)

            # Strip arrays for JSON
            spectral_h_json = {k: v for k, v in spectral_h.items() if k not in ('centroid', 'bandwidth')}
            spectral_p_json = {k: v for k, v in spectral_p.items() if k not in ('centroid', 'bandwidth')}
            spectral_o_json = {k: v for k, v in spectral_o.items() if k not in ('centroid', 'bandwidth')}

            interpretation = generate_interpretation(hpr, spectral_h, spectral_p, onsets, residual, segments)

            plot = generate_hpss_plots(
                y, sr, y_h, y_p, y_r, D, H, P,
                rms_o, rms_h, rms_p, rms_r, rms_times,
                hpr, spectral_o, spectral_h, spectral_p,
                band_h, band_p, segments, hl
            )

            hpr_json = {k: v for k, v in hpr.items() if k != 'hpr_db_over_time'}

            file_info = {
                'filename': file.filename, 'file_size_mb': round(len(content)/(1024*1024), 2),
                'sample_rate': sr, 'duration_sec': round(len(y)/sr, 2),
                'n_samples': len(y), 'format': ext.replace('.',''),
                'n_fft': nf, 'hop_length': hl, 'margin': mg, 'n_segments': n_seg,
            }

            return _to_native({
                'results': {
                    'file_info': file_info,
                    'hpr': hpr_json,
                    'spectral': {'original': spectral_o_json, 'harmonic': spectral_h_json, 'percussive': spectral_p_json},
                    'band_energy': {'harmonic': band_h, 'percussive': band_p},
                    'onsets': onsets,
                    'residual': residual,
                    'segments': segments,
                    'interpretation': interpretation,
                },
                'plot': plot
            })
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ──────────────────────────────────────────────
# from hpss_analysis import router as hpss_router
# app.include_router(hpss_router, prefix="/api/analysis", tags=["Audio Analysis"])
# → POST /api/analysis/hpss-analysis
# ──────────────────────────────────────────────
