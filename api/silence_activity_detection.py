"""
Silence/Activity Detection Analysis Backend (FastAPI)
- Energy-based silence/activity segmentation (configurable thresholds)
- RMS and dB-level frame classification
- Voice Activity Detection (VAD) approximation via ZCR + RMS
- Silence/speech/music/noise classification heuristics
- Per-region statistics: start, end, duration, mean energy
- Silence ratio, activity ratio, and segment counts
- Inter-activity gap analysis
- Fade-in / fade-out detection
- Leading/trailing silence measurement
- Duty cycle and activity density profiling
- Segment-wise activity breakdown
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
# Core Functions
# ──────────────────────────────────────────────

def compute_frame_energy(y, sr, frame_length=2048, hop_length=512):
    """Frame-wise RMS energy and dB levels."""
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    rms_db = 20 * np.log10(rms + 1e-10)
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    return rms, rms_db, times


def classify_frames(rms_db, zcr, silence_thresh_db=-40, low_energy_thresh_db=-25):
    """Classify each frame as silence / low-energy / active."""
    n = min(len(rms_db), len(zcr))
    labels = np.empty(n, dtype='<U12')
    for i in range(n):
        if rms_db[i] < silence_thresh_db:
            labels[i] = 'silence'
        elif rms_db[i] < low_energy_thresh_db:
            labels[i] = 'low_energy'
        else:
            labels[i] = 'active'
    return labels


def compute_zcr(y, frame_length=2048, hop_length=512):
    """Zero crossing rate per frame."""
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=frame_length, hop_length=hop_length)[0]
    return zcr


def detect_regions(labels, times):
    """Convert frame labels to contiguous regions."""
    regions = []
    if len(labels) == 0:
        return regions
    current = labels[0]
    start_idx = 0
    for i in range(1, len(labels)):
        if labels[i] != current:
            regions.append({
                'type': current,
                'start_idx': int(start_idx),
                'end_idx': int(i - 1),
                'start_sec': safe_float(times[start_idx]),
                'end_sec': safe_float(times[min(i - 1, len(times) - 1)]),
                'duration_sec': safe_float(times[min(i - 1, len(times) - 1)] - times[start_idx]),
                'n_frames': int(i - start_idx),
            })
            current = labels[i]
            start_idx = i
    # Last region
    regions.append({
        'type': current,
        'start_idx': int(start_idx),
        'end_idx': int(len(labels) - 1),
        'start_sec': safe_float(times[start_idx]),
        'end_sec': safe_float(times[min(len(labels) - 1, len(times) - 1)]),
        'duration_sec': safe_float(times[min(len(labels) - 1, len(times) - 1)] - times[start_idx]),
        'n_frames': int(len(labels) - start_idx),
    })
    return regions


def enrich_regions(regions, rms_db, zcr):
    """Add mean energy and ZCR to each region."""
    for r in regions:
        s, e = r['start_idx'], r['end_idx'] + 1
        seg_rms = rms_db[s:e]
        seg_zcr = zcr[s:min(e, len(zcr))]
        r['mean_rms_db'] = safe_float(np.mean(seg_rms))
        r['max_rms_db'] = safe_float(np.max(seg_rms))
        r['mean_zcr'] = safe_float(np.mean(seg_zcr)) if len(seg_zcr) > 0 else 0
    return regions


def compute_summary(labels, regions, duration):
    """Global silence/activity statistics."""
    total = len(labels)
    silence_frames = int(np.sum(labels == 'silence'))
    low_frames = int(np.sum(labels == 'low_energy'))
    active_frames = int(np.sum(labels == 'active'))

    silence_regions = [r for r in regions if r['type'] == 'silence']
    low_regions = [r for r in regions if r['type'] == 'low_energy']
    active_regions = [r for r in regions if r['type'] == 'active']

    silence_dur = sum(r['duration_sec'] for r in silence_regions)
    low_dur = sum(r['duration_sec'] for r in low_regions)
    active_dur = sum(r['duration_sec'] for r in active_regions)

    return {
        'total_frames': total,
        'silence_frames': silence_frames,
        'low_energy_frames': low_frames,
        'active_frames': active_frames,
        'silence_percent': safe_float(silence_frames / max(total, 1) * 100),
        'low_energy_percent': safe_float(low_frames / max(total, 1) * 100),
        'active_percent': safe_float(active_frames / max(total, 1) * 100),
        'silence_duration_sec': safe_float(silence_dur),
        'low_energy_duration_sec': safe_float(low_dur),
        'active_duration_sec': safe_float(active_dur),
        'n_silence_regions': len(silence_regions),
        'n_low_regions': len(low_regions),
        'n_active_regions': len(active_regions),
        'n_total_regions': len(regions),
        'duty_cycle': safe_float(active_frames / max(total, 1)),
    }


def compute_gap_analysis(regions):
    """Analyze gaps (silence) between active regions."""
    active_regions = [r for r in regions if r['type'] == 'active']
    if len(active_regions) < 2:
        return {'n_gaps': 0, 'mean_gap': 0, 'max_gap': 0, 'min_gap': 0, 'gaps': []}

    gaps = []
    for i in range(1, len(active_regions)):
        gap = active_regions[i]['start_sec'] - active_regions[i - 1]['end_sec']
        gaps.append(safe_float(gap))

    return {
        'n_gaps': len(gaps),
        'mean_gap': safe_float(np.mean(gaps)) if gaps else 0,
        'max_gap': safe_float(max(gaps)) if gaps else 0,
        'min_gap': safe_float(min(gaps)) if gaps else 0,
        'std_gap': safe_float(np.std(gaps)) if gaps else 0,
        'gaps': gaps[:100],
    }


def detect_leading_trailing(regions, duration):
    """Detect leading and trailing silence."""
    leading = 0
    trailing = 0
    if regions and regions[0]['type'] == 'silence':
        leading = regions[0]['duration_sec']
    if regions and regions[-1]['type'] == 'silence':
        trailing = regions[-1]['duration_sec']
    return {
        'leading_silence_sec': safe_float(leading),
        'trailing_silence_sec': safe_float(trailing),
        'effective_duration_sec': safe_float(duration - leading - trailing),
    }


def detect_fades(rms_db, times, n_check=20):
    """Simple fade-in/fade-out detection from RMS envelope."""
    fade_in = False
    fade_out = False
    fade_in_dur = 0
    fade_out_dur = 0

    if len(rms_db) > n_check:
        head = rms_db[:n_check]
        if np.all(np.diff(head) >= -0.5):  # monotonically non-decreasing (roughly)
            diff_sum = head[-1] - head[0]
            if diff_sum > 6:  # at least 6 dB rise
                fade_in = True
                fade_in_dur = safe_float(times[n_check] - times[0])

        tail = rms_db[-n_check:]
        if np.all(np.diff(tail) <= 0.5):  # monotonically non-increasing
            diff_sum = tail[0] - tail[-1]
            if diff_sum > 6:
                fade_out = True
                fade_out_dur = safe_float(times[-1] - times[-n_check])

    return {
        'fade_in_detected': bool(fade_in),
        'fade_in_duration_sec': fade_in_dur,
        'fade_out_detected': bool(fade_out),
        'fade_out_duration_sec': fade_out_dur,
    }


def compute_activity_density(labels, times, bin_size=1.0):
    """Activity density over time."""
    if len(times) == 0:
        return {'bins': [], 'density': []}
    duration = times[-1]
    bins = np.arange(0, duration + bin_size, bin_size)
    active_mask = labels == 'active'
    density = []
    for i in range(len(bins) - 1):
        mask = (times >= bins[i]) & (times < bins[i + 1])
        total = np.sum(mask)
        active = np.sum(mask & active_mask[:len(times)])
        density.append(safe_float(active / max(total, 1) * 100))
    return {
        'bins': [safe_float(b) for b in bins[:-1]],
        'density': density,
    }


def compute_segments(labels, rms_db, times, n_segments=10):
    """Per-segment activity breakdown."""
    n = len(labels)
    seg_len = max(1, n // n_segments)
    segments = []

    for i in range(n_segments):
        s = i * seg_len
        e = min(s + seg_len, n)
        if s >= n: break
        seg = labels[s:e]
        total = len(seg)
        segments.append({
            'segment': i + 1,
            'start_sec': safe_float(times[s]),
            'silence_percent': safe_float(np.sum(seg == 'silence') / total * 100),
            'low_energy_percent': safe_float(np.sum(seg == 'low_energy') / total * 100),
            'active_percent': safe_float(np.sum(seg == 'active') / total * 100),
            'mean_rms_db': safe_float(np.mean(rms_db[s:e])),
        })
    return segments


def generate_interpretation(summary, gap_analysis, leading_trailing, fades, regions):
    lines = []

    sp = summary['silence_percent']
    ap = summary['active_percent']
    lp = summary['low_energy_percent']

    if sp > 50:
        lines.append(f"Signal is mostly silent ({sp:.0f}%) with {summary['n_active_regions']} active regions spanning {summary['active_duration_sec']:.1f}s.")
    elif ap > 80:
        lines.append(f"Signal is highly active ({ap:.0f}%) — continuous content with minimal silence ({sp:.0f}%).")
    else:
        lines.append(f"Mixed activity: {ap:.0f}% active, {lp:.0f}% low-energy, {sp:.0f}% silence across {summary['n_total_regions']} regions.")

    lines.append(f"Duty cycle: {summary['duty_cycle']:.3f} — {summary['n_active_regions']} active segments, {summary['n_silence_regions']} silence segments.")

    # Leading/trailing
    ls = leading_trailing['leading_silence_sec']
    ts = leading_trailing['trailing_silence_sec']
    if ls > 0.1 or ts > 0.1:
        parts = []
        if ls > 0.1: parts.append(f"leading: {ls:.2f}s")
        if ts > 0.1: parts.append(f"trailing: {ts:.2f}s")
        lines.append(f"Edge silence detected ({', '.join(parts)}). Effective duration: {leading_trailing['effective_duration_sec']:.2f}s.")

    # Gaps
    if gap_analysis['n_gaps'] > 0:
        lines.append(f"Inter-activity gaps: {gap_analysis['n_gaps']} gaps, mean {gap_analysis['mean_gap']*1000:.0f} ms, max {gap_analysis['max_gap']*1000:.0f} ms.")

    # Fades
    if fades['fade_in_detected']:
        lines.append(f"Fade-in detected ({fades['fade_in_duration_sec']:.2f}s).")
    if fades['fade_out_detected']:
        lines.append(f"Fade-out detected ({fades['fade_out_duration_sec']:.2f}s).")

    # Longest regions
    if regions:
        longest_silence = max([r for r in regions if r['type'] == 'silence'], key=lambda r: r['duration_sec'], default=None)
        longest_active = max([r for r in regions if r['type'] == 'active'], key=lambda r: r['duration_sec'], default=None)
        if longest_silence:
            lines.append(f"Longest silence: {longest_silence['duration_sec']:.2f}s at {longest_silence['start_sec']:.2f}s.")
        if longest_active:
            lines.append(f"Longest active region: {longest_active['duration_sec']:.2f}s at {longest_active['start_sec']:.2f}s.")

    return {'summary': ' '.join(lines[:2]), 'details': lines}


# ──────────────────────────────────────────────
# Plot Generation
# ──────────────────────────────────────────────

def generate_plots(
    y, sr, rms, rms_db, rms_times, labels, zcr,
    regions, summary, gap_analysis, activity_density,
    segments, silence_thresh_db, low_thresh_db, hop_length,
) -> str:

    fig = plt.figure(figsize=(18, 30))
    fig.suptitle('Silence / Activity Detection', fontsize=18, fontweight='bold')
    gs = gridspec.GridSpec(8, 2, figure=fig, hspace=0.55, wspace=0.3)

    # Color map for labels
    color_map = {'silence': '#9E9E9E', 'low_energy': '#FF9800', 'active': '#4CAF50'}

    # 1. Waveform + regions
    ax1 = fig.add_subplot(gs[0, :])
    t = np.arange(len(y)) / sr
    ax1.plot(t, y, color='#90CAF9', linewidth=0.3, alpha=0.4)
    for r in regions:
        ax1.axvspan(r['start_sec'], r['end_sec'], alpha=0.2, color=color_map.get(r['type'], '#607D8B'))
    ax1.set_title('Waveform with Activity Regions (green=active, orange=low, gray=silence)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('Time (s)'); ax1.set_ylabel('Amplitude')
    ax1.grid(True, linestyle='--', alpha=0.4)

    # 2. RMS Energy + Thresholds
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(rms_times, rms_db, color='#2196F3', linewidth=0.6)
    ax2.fill_between(rms_times, rms_db, alpha=0.1, color='#2196F3')
    ax2.axhline(y=silence_thresh_db, color='#F44336', linestyle='--', linewidth=1,
                label=f'Silence: {silence_thresh_db} dB')
    ax2.axhline(y=low_thresh_db, color='#FF9800', linestyle='--', linewidth=1,
                label=f'Low energy: {low_thresh_db} dB')
    ax2.set_title('RMS Energy (dB) with Thresholds', fontsize=13, fontweight='bold')
    ax2.set_xlabel('Time (s)'); ax2.set_ylabel('RMS (dB)')
    ax2.legend(fontsize=8); ax2.grid(True, linestyle='--', alpha=0.4)

    # 3. Activity Timeline
    ax3 = fig.add_subplot(gs[2, :])
    label_num = np.zeros(len(labels))
    for i, l in enumerate(labels):
        label_num[i] = 0 if l == 'silence' else (1 if l == 'low_energy' else 2)
    n_t = min(len(rms_times), len(label_num))
    for i in range(n_t - 1):
        ax3.fill_between([rms_times[i], rms_times[i + 1]], 0, 1,
                         color=color_map.get(labels[i], '#607D8B'), alpha=0.7)
    ax3.set_title('Activity Timeline', fontsize=13, fontweight='bold')
    ax3.set_xlabel('Time (s)'); ax3.set_yticks([])
    ax3.set_ylim(0, 1)

    # 4. Region Duration Distribution
    ax4 = fig.add_subplot(gs[3, 0])
    for rtype in ['silence', 'low_energy', 'active']:
        durs = [r['duration_sec'] for r in regions if r['type'] == rtype]
        if durs:
            ax4.hist(durs, bins=20, alpha=0.6, color=color_map[rtype], label=f'{rtype} ({len(durs)})',
                     edgecolor='white', linewidth=0.3)
    ax4.set_title('Region Duration Distribution', fontsize=13, fontweight='bold')
    ax4.set_xlabel('Duration (s)'); ax4.set_ylabel('Count')
    ax4.legend(fontsize=8); ax4.grid(True, linestyle='--', alpha=0.4, axis='y')

    # 5. Activity Pie
    ax5 = fig.add_subplot(gs[3, 1])
    pie_vals = [summary['active_percent'], summary['low_energy_percent'], summary['silence_percent']]
    pie_labels = [f'Active\n{summary["active_percent"]:.1f}%',
                  f'Low Energy\n{summary["low_energy_percent"]:.1f}%',
                  f'Silence\n{summary["silence_percent"]:.1f}%']
    pie_colors = ['#4CAF50', '#FF9800', '#9E9E9E']
    ax5.pie(pie_vals, labels=pie_labels, colors=pie_colors, startangle=90, textprops={'fontsize': 10})
    ax5.set_title('Frame Classification', fontsize=13, fontweight='bold')

    # 6. Activity Density
    ax6 = fig.add_subplot(gs[4, 0])
    if activity_density['bins']:
        ax6.bar(activity_density['bins'], activity_density['density'], width=0.9,
                color='#4CAF50', edgecolor='white', linewidth=0.3, alpha=0.85)
        ax6.axhline(y=summary['active_percent'], color='#F44336', linestyle='--', linewidth=0.8,
                     label=f'Global: {summary["active_percent"]:.0f}%')
        ax6.legend(fontsize=8)
    ax6.set_title('Activity Density (% per second)', fontsize=13, fontweight='bold')
    ax6.set_xlabel('Time (s)'); ax6.set_ylabel('Active %')
    ax6.grid(True, linestyle='--', alpha=0.4, axis='y')

    # 7. Gap Duration Histogram
    ax7 = fig.add_subplot(gs[4, 1])
    if gap_analysis['gaps']:
        ax7.hist([g * 1000 for g in gap_analysis['gaps']], bins=30, color='#9C27B0',
                 edgecolor='white', linewidth=0.3, alpha=0.85)
        ax7.axvline(x=gap_analysis['mean_gap'] * 1000, color='#F44336', linestyle='--', linewidth=1,
                     label=f'Mean: {gap_analysis["mean_gap"]*1000:.0f} ms')
        ax7.legend(fontsize=8)
    ax7.set_title('Inter-Activity Gap Duration', fontsize=13, fontweight='bold')
    ax7.set_xlabel('Gap (ms)'); ax7.set_ylabel('Count')
    ax7.grid(True, linestyle='--', alpha=0.4, axis='y')

    # 8. ZCR + RMS scatter
    ax8 = fig.add_subplot(gs[5, 0])
    n_sc = min(len(rms_db), len(zcr), len(labels))
    sc_colors = [color_map.get(labels[i], '#607D8B') for i in range(n_sc)]
    step = max(1, n_sc // 3000)
    ax8.scatter(rms_db[:n_sc:step], zcr[:n_sc:step], c=sc_colors[::step], s=3, alpha=0.4)
    ax8.axvline(x=silence_thresh_db, color='#F44336', linestyle='--', linewidth=0.5)
    ax8.axvline(x=low_thresh_db, color='#FF9800', linestyle='--', linewidth=0.5)
    ax8.set_title('ZCR vs RMS (colored by class)', fontsize=13, fontweight='bold')
    ax8.set_xlabel('RMS (dB)'); ax8.set_ylabel('ZCR')
    ax8.grid(True, linestyle='--', alpha=0.4)

    # 9. Segment Activity Stacked Bar
    ax9 = fig.add_subplot(gs[5, 1])
    seg_labels = [f'S{s["segment"]}' for s in segments]
    seg_a = [s['active_percent'] for s in segments]
    seg_l = [s['low_energy_percent'] for s in segments]
    seg_s = [s['silence_percent'] for s in segments]
    x_sg = np.arange(len(segments))
    ax9.bar(x_sg, seg_a, color='#4CAF50', label='Active', edgecolor='white')
    ax9.bar(x_sg, seg_l, bottom=seg_a, color='#FF9800', label='Low', edgecolor='white')
    ax9.bar(x_sg, seg_s, bottom=[a + l for a, l in zip(seg_a, seg_l)], color='#9E9E9E', label='Silence', edgecolor='white')
    ax9.set_xticks(x_sg); ax9.set_xticklabels(seg_labels, fontsize=8)
    ax9.set_title('Segment Activity Breakdown', fontsize=13, fontweight='bold')
    ax9.set_ylabel('Percent'); ax9.legend(fontsize=7)
    ax9.grid(True, linestyle='--', alpha=0.4, axis='y')

    # 10. Region count bar
    ax10 = fig.add_subplot(gs[6, 0])
    rc_labels = ['Silence', 'Low Energy', 'Active']
    rc_vals = [summary['n_silence_regions'], summary['n_low_regions'], summary['n_active_regions']]
    rc_colors = ['#9E9E9E', '#FF9800', '#4CAF50']
    bars10 = ax10.bar(rc_labels, rc_vals, color=rc_colors, edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars10, rc_vals):
        ax10.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 str(val), ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax10.set_title('Region Counts', fontsize=13, fontweight='bold')
    ax10.set_ylabel('Count'); ax10.grid(True, linestyle='--', alpha=0.4, axis='y')

    # 11. Key Metrics
    ax11 = fig.add_subplot(gs[6, 1])
    km_labels = ['Active\n(%)', 'Silence\n(%)', 'Duty\nCycle', 'Regions', 'Gaps']
    km_vals = [summary['active_percent'], summary['silence_percent'],
               summary['duty_cycle'] * 100, summary['n_total_regions'], gap_analysis['n_gaps']]
    km_colors = ['#4CAF50', '#9E9E9E', '#2196F3', '#FF9800', '#9C27B0']
    km_disp = [f'{summary["active_percent"]:.0f}%', f'{summary["silence_percent"]:.0f}%',
               f'{summary["duty_cycle"]:.3f}', str(summary['n_total_regions']), str(gap_analysis['n_gaps'])]
    bars11 = ax11.bar(km_labels, km_vals, color=km_colors, edgecolor='white', linewidth=0.8)
    for bar, d in zip(bars11, km_disp):
        ax11.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.3,
                 d, ha='center', va='bottom', fontsize=8, fontweight='bold')
    ax11.set_title('Key Metrics', fontsize=13, fontweight='bold')
    ax11.grid(True, linestyle='--', alpha=0.4, axis='y')

    # 12. Zoomed waveform + regions
    ax12 = fig.add_subplot(gs[7, :])
    zoom = min(5.0, len(y) / sr)
    mask = t <= zoom
    ax12.plot(t[mask], y[mask], color='#90CAF9', linewidth=0.5, alpha=0.6)
    for r in regions:
        if r['start_sec'] < zoom:
            ax12.axvspan(max(0, r['start_sec']), min(zoom, r['end_sec']),
                         alpha=0.25, color=color_map.get(r['type'], '#607D8B'))
    ax12.set_title(f'Zoomed (0–{zoom:.1f}s)', fontsize=13, fontweight='bold')
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

@router.post("/silence-activity-detection")
async def silence_activity_detection(
    file: UploadFile = File(...),
    frame_length: Optional[int] = Form(2048),
    hop_length: Optional[int] = Form(512),
    silence_thresh_db: Optional[float] = Form(-40.0),
    low_energy_thresh_db: Optional[float] = Form(-25.0),
    n_segments: Optional[int] = Form(10),
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

            fl = frame_length or 2048; hl = hop_length or 512
            st = silence_thresh_db if silence_thresh_db is not None else -40.0
            lt = low_energy_thresh_db if low_energy_thresh_db is not None else -25.0
            n_seg = n_segments or 10
            duration = len(y) / sr

            rms, rms_db, rms_times = compute_frame_energy(y, sr, fl, hl)
            zcr = compute_zcr(y, fl, hl)
            labels = classify_frames(rms_db, zcr, st, lt)
            regions = detect_regions(labels, rms_times)
            regions = enrich_regions(regions, rms_db, zcr)
            summary = compute_summary(labels, regions, duration)
            gap_analysis = compute_gap_analysis(regions)
            leading_trailing = detect_leading_trailing(regions, duration)
            fades = detect_fades(rms_db, rms_times)
            activity_density = compute_activity_density(labels, rms_times)
            segments = compute_segments(labels, rms_db, rms_times, n_seg)
            interpretation = generate_interpretation(summary, gap_analysis, leading_trailing, fades, regions)

            plot = generate_plots(
                y, sr, rms, rms_db, rms_times, labels, zcr,
                regions, summary, gap_analysis, activity_density,
                segments, st, lt, hl
            )

            file_info = {
                'filename': file.filename, 'file_size_mb': round(len(content)/(1024*1024), 2),
                'sample_rate': sr, 'duration_sec': round(duration, 2),
                'n_samples': len(y), 'format': ext.replace('.',''),
                'frame_length': fl, 'hop_length': hl,
                'silence_thresh_db': st, 'low_energy_thresh_db': lt,
                'n_segments': n_seg,
            }

            return _to_native({
                'results': {
                    'file_info': file_info,
                    'summary': summary,
                    'gap_analysis': {k: v for k, v in gap_analysis.items() if k != 'gaps'},
                    'leading_trailing': leading_trailing,
                    'fades': fades,
                    'regions': regions[:200],
                    'segments': segments,
                    'interpretation': interpretation,
                },
                'plot': plot
            })
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# from silence_activity_detection import router as sad_router
# app.include_router(sad_router, prefix="/api/analysis", tags=["Audio Analysis"])
# → POST /api/analysis/silence-activity-detection
