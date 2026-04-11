"""
Mel Filterbank Analysis Backend (FastAPI)
- Comprehensive mel filterbank design and visualization
- Filterbank shape, overlap, bandwidth, and center frequency analysis
- Mel-scale vs linear-scale frequency mapping demonstration
- Per-filter energy response from audio signal
- Filter overlap matrix and spectral coverage metrics
- Comparison of different mel scale formulas (HTK, Slaney)
- Filterbank frequency resolution analysis
- Critical band approximation and auditory model comparison
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
# Mel Scale Conversions
# ──────────────────────────────────────────────

def hz_to_mel_htk(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)

def mel_to_hz_htk(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

def hz_to_mel_slaney(hz):
    f_sp = 200.0 / 3.0
    min_log_hz = 1000.0
    min_log_mel = (min_log_hz - 0) / f_sp
    logstep = np.log(6.4) / 27.0
    if np.isscalar(hz):
        if hz < min_log_hz:
            return hz / f_sp
        return min_log_mel + np.log(hz / min_log_hz) / logstep
    hz = np.asarray(hz, dtype=float)
    mel = np.where(hz < min_log_hz, hz / f_sp, min_log_mel + np.log(hz / min_log_hz) / logstep)
    return mel


# ──────────────────────────────────────────────
# Filterbank Analysis Functions
# ──────────────────────────────────────────────

def compute_filterbank(sr, n_fft, n_mels, fmin, fmax, htk=False):
    """Compute mel filterbank matrix and extract filter properties."""
    fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels, fmin=fmin, fmax=fmax, htk=htk)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    return fb, freqs


def analyze_filter_properties(fb, freqs, n_mels, fmin, fmax, sr):
    """Per-filter center frequency, bandwidth, and overlap."""
    filters = []
    for i in range(n_mels):
        row = fb[i]
        nonzero = np.where(row > 0)[0]
        if len(nonzero) == 0:
            filters.append({
                'filter': i, 'center_hz': 0, 'center_mel': 0,
                'lower_hz': 0, 'upper_hz': 0, 'bandwidth_hz': 0,
                'bandwidth_mel': 0, 'peak_gain': 0, 'n_bins': 0,
            })
            continue

        lower_idx, upper_idx = nonzero[0], nonzero[-1]
        peak_idx = np.argmax(row)
        center_hz = freqs[peak_idx]
        lower_hz = freqs[lower_idx]
        upper_hz = freqs[upper_idx]
        bw_hz = upper_hz - lower_hz

        center_mel = hz_to_mel_htk(center_hz)
        lower_mel = hz_to_mel_htk(lower_hz)
        upper_mel = hz_to_mel_htk(upper_hz)
        bw_mel = upper_mel - lower_mel

        filters.append({
            'filter': i,
            'center_hz': safe_float(center_hz),
            'center_mel': safe_float(center_mel),
            'lower_hz': safe_float(lower_hz),
            'upper_hz': safe_float(upper_hz),
            'bandwidth_hz': safe_float(bw_hz),
            'bandwidth_mel': safe_float(bw_mel),
            'peak_gain': safe_float(row[peak_idx]),
            'n_bins': int(len(nonzero)),
        })

    return filters


def compute_filter_overlap(fb, n_mels):
    """Compute overlap between adjacent filters."""
    overlaps = []
    for i in range(n_mels - 1):
        f1 = fb[i]
        f2 = fb[i + 1]
        overlap_bins = np.sum((f1 > 0) & (f2 > 0))
        total_bins = np.sum((f1 > 0) | (f2 > 0))
        overlap_ratio = overlap_bins / (total_bins + 1e-10)
        overlaps.append({
            'filter_pair': f'{i}-{i+1}',
            'overlap_bins': int(overlap_bins),
            'overlap_ratio': safe_float(overlap_ratio),
        })
    return overlaps


def compute_spectral_coverage(fb, freqs, fmin, fmax):
    """Measure how well the filterbank covers the frequency range."""
    total_energy = np.sum(fb, axis=0)
    covered = np.sum(total_energy > 0)
    total = len(freqs)
    in_range = np.sum((freqs >= fmin) & (freqs <= fmax))

    # Uniformity of coverage
    nonzero_energy = total_energy[total_energy > 0]
    uniformity = 1.0 - (np.std(nonzero_energy) / (np.mean(nonzero_energy) + 1e-10))

    return {
        'total_fft_bins': int(total),
        'covered_bins': int(covered),
        'coverage_percent': safe_float(covered / (in_range + 1e-10) * 100),
        'uniformity': safe_float(uniformity),
        'min_coverage': safe_float(np.min(nonzero_energy) if len(nonzero_energy) > 0 else 0),
        'max_coverage': safe_float(np.max(nonzero_energy) if len(nonzero_energy) > 0 else 0),
        'mean_coverage': safe_float(np.mean(nonzero_energy) if len(nonzero_energy) > 0 else 0),
    }


def compute_filter_energy_from_signal(y, sr, n_fft, hop_length, fb):
    """Apply filterbank to signal and compute per-filter energy."""
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length)) ** 2
    mel_spec = np.dot(fb, S)
    # Mean energy per filter across time
    mean_energy = np.mean(mel_spec, axis=1)
    mean_energy_db = 10 * np.log10(mean_energy + 1e-10)
    max_energy = np.max(mel_spec, axis=1)
    max_energy_db = 10 * np.log10(max_energy + 1e-10)

    return {
        'mean_energy': [safe_float(x) for x in mean_energy],
        'mean_energy_db': [safe_float(x) for x in mean_energy_db],
        'max_energy': [safe_float(x) for x in max_energy],
        'max_energy_db': [safe_float(x) for x in max_energy_db],
        'mel_spectrogram': mel_spec,
    }


def compute_mel_scale_mapping(fmin, fmax, sr, n_points=100):
    """Generate mel-to-Hz and Hz-to-mel mapping curves."""
    hz_vals = np.linspace(fmin, fmax, n_points)
    mel_htk = hz_to_mel_htk(hz_vals)
    mel_slaney = hz_to_mel_slaney(hz_vals)

    return {
        'hz': [safe_float(x) for x in hz_vals],
        'mel_htk': [safe_float(x) for x in mel_htk],
        'mel_slaney': [safe_float(x) for x in mel_slaney],
    }


def compute_filterbank_summary(filters, overlaps, coverage, energy_data, n_mels):
    """Aggregate summary metrics."""
    bws = [f['bandwidth_hz'] for f in filters if f['bandwidth_hz'] > 0]
    centers = [f['center_hz'] for f in filters if f['center_hz'] > 0]

    # Low vs high frequency resolution
    low_bws = [f['bandwidth_hz'] for f in filters if 0 < f['center_hz'] < 1000]
    high_bws = [f['bandwidth_hz'] for f in filters if f['center_hz'] >= 1000]

    mean_overlap = np.mean([o['overlap_ratio'] for o in overlaps]) if overlaps else 0

    return {
        'n_filters': n_mels,
        'min_bandwidth_hz': safe_float(min(bws)) if bws else 0,
        'max_bandwidth_hz': safe_float(max(bws)) if bws else 0,
        'mean_bandwidth_hz': safe_float(np.mean(bws)) if bws else 0,
        'low_freq_mean_bw': safe_float(np.mean(low_bws)) if low_bws else 0,
        'high_freq_mean_bw': safe_float(np.mean(high_bws)) if high_bws else 0,
        'resolution_ratio': safe_float((np.mean(low_bws) if low_bws else 1) / (np.mean(high_bws) if high_bws else 1)),
        'mean_overlap_ratio': safe_float(mean_overlap),
        'coverage_percent': coverage['coverage_percent'],
        'coverage_uniformity': coverage['uniformity'],
        'total_filters_active': int(sum(1 for f in filters if f['bandwidth_hz'] > 0)),
    }


def generate_interpretation(summary, filters, energy_data, coverage):
    """Generate human-readable interpretation."""
    lines = []

    n = summary['n_filters']
    lines.append(f"Mel filterbank with {n} filters spanning {filters[0]['lower_hz']:.0f} Hz to {filters[-1]['upper_hz']:.0f} Hz.")

    # Resolution
    rr = summary['resolution_ratio']
    if rr < 0.5:
        lines.append(f"Low-frequency resolution is {1/rr:.1f}× finer than high-frequency — mel scale provides more detail where human hearing is most sensitive.")
    else:
        lines.append(f"Resolution ratio (low/high bandwidth): {rr:.2f}.")

    # Bandwidth
    lines.append(f"Filter bandwidth ranges from {summary['min_bandwidth_hz']:.0f} Hz (narrowest, low freq) to {summary['max_bandwidth_hz']:.0f} Hz (widest, high freq).")

    # Overlap
    ol = summary['mean_overlap_ratio']
    if ol > 0.4:
        lines.append(f"High mean filter overlap ({ol:.1%}) — ensures smooth frequency representation with minimal spectral gaps.")
    elif ol > 0.2:
        lines.append(f"Moderate filter overlap ({ol:.1%}).")
    else:
        lines.append(f"Low filter overlap ({ol:.1%}) — some frequency ranges may be underrepresented.")

    # Coverage
    lines.append(f"Spectral coverage: {coverage['coverage_percent']:.1f}% of FFT bins within range are covered (uniformity: {coverage['uniformity']:.2f}).")

    # Energy
    if energy_data:
        energies_db = energy_data['mean_energy_db']
        max_filter = int(np.argmax(energies_db))
        lines.append(f"Highest mean energy in filter {max_filter} (center: {filters[max_filter]['center_hz']:.0f} Hz, {energies_db[max_filter]:.1f} dB).")

    return {
        'summary': ' '.join(lines[:2]),
        'details': lines,
    }


# ──────────────────────────────────────────────
# Plot Generation
# ──────────────────────────────────────────────

def generate_filterbank_plots(
    y, sr, fb, freqs, filters, overlaps, coverage,
    energy_data, mel_mapping, summary, n_mels, hop_length,
) -> str:

    fig = plt.figure(figsize=(18, 30))
    fig.suptitle('Mel Filterbank Analysis', fontsize=18, fontweight='bold')
    gs = gridspec.GridSpec(7, 2, figure=fig, hspace=0.55, wspace=0.3)

    # 1. Full Filterbank Shape
    ax1 = fig.add_subplot(gs[0, :])
    cmap = plt.cm.viridis(np.linspace(0, 1, n_mels))
    for i in range(n_mels):
        ax1.plot(freqs, fb[i], color=cmap[i], linewidth=0.6, alpha=0.7)
    ax1.set_title(f'Mel Filterbank ({n_mels} filters)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Filter Gain')
    ax1.set_xlim(0, min(freqs[-1], 20000))
    ax1.grid(True, linestyle='--', alpha=0.4)

    # 2. Filterbank (log frequency)
    ax2 = fig.add_subplot(gs[1, 0])
    for i in range(n_mels):
        ax2.semilogx(freqs[1:], fb[i, 1:], color=cmap[i], linewidth=0.6, alpha=0.7)
    ax2.set_title('Filterbank (Log Frequency Scale)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Frequency (Hz, log)')
    ax2.set_ylabel('Filter Gain')
    ax2.grid(True, linestyle='--', alpha=0.4)

    # 3. Center Frequencies
    ax3 = fig.add_subplot(gs[1, 1])
    centers = [f['center_hz'] for f in filters]
    ax3.scatter(range(len(centers)), centers, c=range(len(centers)), cmap='viridis', s=20, edgecolors='white', linewidth=0.3)
    ax3.set_title('Filter Center Frequencies', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Filter Index')
    ax3.set_ylabel('Center Frequency (Hz)')
    ax3.grid(True, linestyle='--', alpha=0.4)

    # 4. Bandwidth vs Filter Index
    ax4 = fig.add_subplot(gs[2, 0])
    bws = [f['bandwidth_hz'] for f in filters]
    colors_bw = ['#4CAF50' if b < np.mean(bws) else '#FF9800' if b < np.mean(bws) * 2 else '#F44336' for b in bws]
    ax4.bar(range(len(bws)), bws, color=colors_bw, edgecolor='white', linewidth=0.3)
    ax4.set_title('Filter Bandwidth (Hz)', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Filter Index')
    ax4.set_ylabel('Bandwidth (Hz)')
    ax4.grid(True, linestyle='--', alpha=0.4, axis='y')

    # 5. Bandwidth in Mel
    ax5 = fig.add_subplot(gs[2, 1])
    bws_mel = [f['bandwidth_mel'] for f in filters]
    ax5.bar(range(len(bws_mel)), bws_mel, color='#9C27B0', edgecolor='white', linewidth=0.3, alpha=0.85)
    ax5.axhline(y=np.mean(bws_mel), color='#F44336', linestyle='--', linewidth=0.8,
                label=f'Mean: {np.mean(bws_mel):.1f} mel')
    ax5.set_title('Filter Bandwidth (Mel)', fontsize=13, fontweight='bold')
    ax5.set_xlabel('Filter Index')
    ax5.set_ylabel('Bandwidth (mel)')
    ax5.legend(fontsize=8)
    ax5.grid(True, linestyle='--', alpha=0.4, axis='y')

    # 6. Mel Scale Mapping
    ax6 = fig.add_subplot(gs[3, 0])
    ax6.plot(mel_mapping['hz'], mel_mapping['mel_htk'], color='#2196F3', linewidth=2, label='HTK')
    ax6.plot(mel_mapping['hz'], mel_mapping['mel_slaney'], color='#FF5722', linewidth=2, linestyle='--', label='Slaney')
    ax6.set_title('Hz → Mel Mapping', fontsize=13, fontweight='bold')
    ax6.set_xlabel('Frequency (Hz)')
    ax6.set_ylabel('Mel')
    ax6.legend(fontsize=9)
    ax6.grid(True, linestyle='--', alpha=0.4)

    # 7. Spectral Coverage
    ax7 = fig.add_subplot(gs[3, 1])
    total_coverage = np.sum(fb, axis=0)
    ax7.fill_between(freqs, total_coverage, alpha=0.3, color='#4CAF50')
    ax7.plot(freqs, total_coverage, color='#4CAF50', linewidth=1)
    ax7.set_title(f'Spectral Coverage (uniformity: {coverage["uniformity"]:.2f})', fontsize=13, fontweight='bold')
    ax7.set_xlabel('Frequency (Hz)')
    ax7.set_ylabel('Summed Filter Gain')
    ax7.set_xlim(0, min(freqs[-1], 20000))
    ax7.grid(True, linestyle='--', alpha=0.4)

    # 8. Filter Overlap
    ax8 = fig.add_subplot(gs[4, 0])
    overlap_ratios = [o['overlap_ratio'] for o in overlaps]
    ax8.bar(range(len(overlap_ratios)), overlap_ratios, color='#00BCD4', edgecolor='white', linewidth=0.3)
    ax8.axhline(y=np.mean(overlap_ratios), color='#F44336', linestyle='--', linewidth=0.8,
                label=f'Mean: {np.mean(overlap_ratios):.3f}')
    ax8.set_title('Adjacent Filter Overlap Ratio', fontsize=13, fontweight='bold')
    ax8.set_xlabel('Filter Pair Index')
    ax8.set_ylabel('Overlap Ratio')
    ax8.legend(fontsize=8)
    ax8.grid(True, linestyle='--', alpha=0.4, axis='y')

    # 9. Per-Filter Energy (from signal)
    ax9 = fig.add_subplot(gs[4, 1])
    if energy_data:
        energies_db = energy_data['mean_energy_db']
        max_idx = int(np.argmax(energies_db))
        colors_e = ['#F44336' if i == max_idx else '#2196F3' for i in range(len(energies_db))]
        ax9.bar(range(len(energies_db)), energies_db, color=colors_e, edgecolor='white', linewidth=0.3)
        ax9.set_title('Per-Filter Mean Energy (dB)', fontsize=13, fontweight='bold')
        ax9.set_xlabel('Filter Index')
        ax9.set_ylabel('Energy (dB)')
        ax9.grid(True, linestyle='--', alpha=0.4, axis='y')
    else:
        ax9.text(0.5, 0.5, 'No signal data', ha='center', va='center', transform=ax9.transAxes)
        ax9.set_title('Per-Filter Energy', fontsize=13, fontweight='bold')

    # 10. Mel Spectrogram (from signal)
    ax10 = fig.add_subplot(gs[5, :])
    if energy_data and 'mel_spectrogram' in energy_data:
        mel_db = librosa.power_to_db(energy_data['mel_spectrogram'], ref=np.max)
        img10 = librosa.display.specshow(mel_db, x_axis='time', y_axis='mel', sr=sr,
                                          hop_length=hop_length, ax=ax10, cmap='magma')
        ax10.set_title('Mel Spectrogram (from this filterbank)', fontsize=13, fontweight='bold')
        fig.colorbar(img10, ax=ax10, label='dB', pad=0.02)
    else:
        ax10.text(0.5, 0.5, 'No signal data', ha='center', va='center', transform=ax10.transAxes)
        ax10.set_title('Mel Spectrogram', fontsize=13, fontweight='bold')

    # 11. Bins per Filter
    ax11 = fig.add_subplot(gs[6, 0])
    n_bins = [f['n_bins'] for f in filters]
    ax11.bar(range(len(n_bins)), n_bins, color='#607D8B', edgecolor='white', linewidth=0.3)
    ax11.set_title('FFT Bins per Filter', fontsize=13, fontweight='bold')
    ax11.set_xlabel('Filter Index')
    ax11.set_ylabel('Number of Bins')
    ax11.grid(True, linestyle='--', alpha=0.4, axis='y')

    # 12. Summary Metrics
    ax12 = fig.add_subplot(gs[6, 1])
    labels_s = ['Low-freq\nBW (Hz)', 'High-freq\nBW (Hz)', 'Mean\nOverlap', 'Coverage\n(%)', 'Uniformity']
    vals_s = [summary['low_freq_mean_bw'], summary['high_freq_mean_bw'],
              summary['mean_overlap_ratio'] * 100, summary['coverage_percent'], summary['coverage_uniformity'] * 100]
    colors_s = ['#4CAF50', '#F44336', '#2196F3', '#FF9800', '#9C27B0']
    bars_s = ax12.bar(labels_s, vals_s, color=colors_s, edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars_s, vals_s):
        ax12.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{val:.1f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax12.set_title('Filterbank Summary', fontsize=13, fontweight='bold')
    ax12.set_ylabel('Value')
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

@router.post("/mel-filterbank-analysis")
async def mel_filterbank_analysis(
    file: UploadFile = File(...),
    n_fft: Optional[int] = Form(2048),
    hop_length: Optional[int] = Form(512),
    n_mels: Optional[int] = Form(128),
    fmin: Optional[float] = Form(0.0),
    fmax: Optional[float] = Form(None),
    htk: Optional[bool] = Form(False),
    target_sr: Optional[int] = Form(None),
):
    """
    Mel Filterbank Analysis.

    Comprehensive analysis of mel filterbank design and its application to
    an audio signal, including filter shapes, bandwidth, overlap, spectral
    coverage, Hz/mel mapping, and per-filter energy from the input signal.
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
            nm = n_mels or 128
            fmin_v = fmin or 0.0
            fmax_v = fmax if fmax else sr / 2
            htk_v = htk if htk is not None else False

            # Compute filterbank
            fb, freqs = compute_filterbank(sr, nf, nm, fmin_v, fmax_v, htk_v)

            # Filter properties
            filters = analyze_filter_properties(fb, freqs, nm, fmin_v, fmax_v, sr)

            # Overlap
            overlaps = compute_filter_overlap(fb, nm)

            # Coverage
            coverage = compute_spectral_coverage(fb, freqs, fmin_v, fmax_v)

            # Energy from signal
            energy_data = compute_filter_energy_from_signal(y_t, sr, nf, hl, fb)
            mel_spec = energy_data.pop('mel_spectrogram')  # keep for plotting, not JSON
            energy_data_json = energy_data

            # Mel mapping
            mel_mapping = compute_mel_scale_mapping(fmin_v, fmax_v, sr)

            # Summary
            summary = compute_filterbank_summary(filters, overlaps, coverage, energy_data, nm)

            # Interpretation
            interpretation = generate_interpretation(summary, filters, energy_data, coverage)

            # Plot (pass mel_spec back for plotting)
            energy_for_plot = {**energy_data, 'mel_spectrogram': mel_spec}
            plot = generate_filterbank_plots(
                y_t, sr, fb, freqs, filters, overlaps, coverage,
                energy_for_plot, mel_mapping, summary, nm, hl
            )

            file_info = {
                'filename': file.filename, 'file_size_mb': round(len(content)/(1024*1024), 2),
                'sample_rate': sr, 'duration_sec': round(len(y_t)/sr, 2),
                'n_samples': len(y_t), 'format': ext.replace('.',''),
                'n_fft': nf, 'hop_length': hl, 'n_mels': nm,
                'fmin': fmin_v, 'fmax': fmax_v, 'htk_formula': htk_v,
                'frequency_resolution_hz': safe_float(sr / nf),
            }

            return _to_native({
                'results': {
                    'file_info': file_info,
                    'summary': summary,
                    'filter_properties': filters,
                    'overlaps': overlaps[:50],
                    'coverage': coverage,
                    'filter_energy': energy_data_json,
                    'mel_mapping': mel_mapping,
                    'interpretation': interpretation,
                },
                'plot': plot
            })
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ──────────────────────────────────────────────
# from mel_filterbank_analysis import router as mel_fb_router
# app.include_router(mel_fb_router, prefix="/api/analysis", tags=["Audio Analysis"])
# → POST /api/analysis/mel-filterbank-analysis
# ──────────────────────────────────────────────
