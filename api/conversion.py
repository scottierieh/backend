"""
Conversion Rate Analysis Router for FastAPI
Funnel Analysis, A/B Testing, Conversion Optimization
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Literal
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from scipy import stats
from scipy.stats import chi2_contingency, norm
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class ConversionRequest(BaseModel):
    data: List[Dict[str, Any]]
    visitor_col: Optional[str] = None  # Unique visitor ID
    stage_col: Optional[str] = None  # Funnel stage
    converted_col: Optional[str] = None  # Binary conversion flag
    variant_col: Optional[str] = None  # A/B test variant
    segment_col: Optional[str] = None  # Segment for breakdown
    date_col: Optional[str] = None  # Date for trend analysis
    value_col: Optional[str] = None  # Revenue/value per conversion
    funnel_stages: Optional[List[str]] = None  # Ordered funnel stages
    confidence_level: float = 0.95


def _to_native_type(obj):
    """Convert numpy/pandas types to JSON-serializable Python types"""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    return obj


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string"""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return image_base64


def calculate_conversion_rate(conversions: int, visitors: int) -> Dict[str, Any]:
    """Calculate conversion rate with confidence interval"""
    if visitors == 0:
        return {'rate': 0, 'lower': 0, 'upper': 0}
    
    rate = conversions / visitors
    
    # Wilson score interval
    z = 1.96  # 95% confidence
    n = visitors
    p = rate
    
    denominator = 1 + z**2 / n
    center = (p + z**2 / (2*n)) / denominator
    margin = z * np.sqrt((p * (1-p) + z**2 / (4*n)) / n) / denominator
    
    return {
        'rate': _to_native_type(rate),
        'lower': _to_native_type(max(0, center - margin)),
        'upper': _to_native_type(min(1, center + margin)),
        'conversions': conversions,
        'visitors': visitors
    }


def analyze_funnel(df: pd.DataFrame, stage_col: str, visitor_col: str,
                   funnel_stages: Optional[List[str]] = None) -> Dict[str, Any]:
    """Analyze conversion funnel"""
    
    if funnel_stages is None:
        # Auto-detect stages from data
        funnel_stages = df[stage_col].unique().tolist()
    
    # Count unique visitors at each stage
    stage_counts = []
    for stage in funnel_stages:
        stage_df = df[df[stage_col] == stage]
        count = stage_df[visitor_col].nunique() if visitor_col else len(stage_df)
        stage_counts.append({
            'stage': stage,
            'visitors': count
        })
    
    # Calculate conversion rates between stages
    funnel_metrics = []
    total_visitors = stage_counts[0]['visitors'] if stage_counts else 0
    
    for i, stage_data in enumerate(stage_counts):
        visitors = stage_data['visitors']
        
        # Stage-to-stage conversion
        if i > 0:
            prev_visitors = stage_counts[i-1]['visitors']
            stage_conversion = visitors / prev_visitors if prev_visitors > 0 else 0
            drop_off = prev_visitors - visitors
            drop_off_rate = drop_off / prev_visitors if prev_visitors > 0 else 0
        else:
            stage_conversion = 1.0
            drop_off = 0
            drop_off_rate = 0
        
        # Overall conversion from top of funnel
        overall_conversion = visitors / total_visitors if total_visitors > 0 else 0
        
        funnel_metrics.append({
            'stage': stage_data['stage'],
            'visitors': visitors,
            'stage_conversion_rate': _to_native_type(stage_conversion),
            'overall_conversion_rate': _to_native_type(overall_conversion),
            'drop_off': drop_off,
            'drop_off_rate': _to_native_type(drop_off_rate)
        })
    
    # Calculate overall funnel conversion
    if len(stage_counts) >= 2:
        overall_rate = stage_counts[-1]['visitors'] / stage_counts[0]['visitors'] if stage_counts[0]['visitors'] > 0 else 0
    else:
        overall_rate = 0
    
    # Find biggest drop-off
    biggest_dropoff = max(funnel_metrics[1:], key=lambda x: x['drop_off_rate']) if len(funnel_metrics) > 1 else None
    
    return {
        'stages': funnel_metrics,
        'total_visitors': total_visitors,
        'total_conversions': stage_counts[-1]['visitors'] if stage_counts else 0,
        'overall_conversion_rate': _to_native_type(overall_rate),
        'biggest_dropoff_stage': biggest_dropoff['stage'] if biggest_dropoff else None,
        'biggest_dropoff_rate': biggest_dropoff['drop_off_rate'] if biggest_dropoff else None
    }


def analyze_ab_test(df: pd.DataFrame, variant_col: str, converted_col: str,
                    confidence_level: float = 0.95) -> Dict[str, Any]:
    """Analyze A/B test results"""
    
    variants = df[variant_col].unique()
    
    if len(variants) < 2:
        return {'error': 'Need at least 2 variants for A/B test'}
    
    # Calculate metrics for each variant
    variant_results = []
    for variant in variants:
        variant_df = df[df[variant_col] == variant]
        visitors = len(variant_df)
        conversions = variant_df[converted_col].sum()
        
        cr = calculate_conversion_rate(int(conversions), visitors)
        cr['variant'] = str(variant)
        variant_results.append(cr)
    
    # Sort by conversion rate (control first if named)
    control_names = ['control', 'Control', 'A', 'Original', 'original']
    variant_results.sort(key=lambda x: (x['variant'] not in control_names, -x['rate']))
    
    # Statistical significance test (Chi-square)
    contingency = []
    for vr in variant_results:
        contingency.append([vr['conversions'], vr['visitors'] - vr['conversions']])
    
    if len(contingency) >= 2:
        chi2, p_value, dof, expected = chi2_contingency(contingency)
        is_significant = p_value < (1 - confidence_level)
    else:
        chi2, p_value, is_significant = 0, 1, False
    
    # Calculate lift (treatment vs control)
    control = variant_results[0]
    lifts = []
    for vr in variant_results[1:]:
        if control['rate'] > 0:
            lift = (vr['rate'] - control['rate']) / control['rate'] * 100
        else:
            lift = 0
        lifts.append({
            'variant': vr['variant'],
            'lift': _to_native_type(lift),
            'absolute_diff': _to_native_type(vr['rate'] - control['rate'])
        })
    
    # Calculate required sample size for 80% power
    if len(variant_results) >= 2:
        p1 = control['rate']
        p2 = variant_results[1]['rate'] if len(variant_results) > 1 else p1
        if p1 > 0 and p2 > 0 and p1 != p2:
            effect_size = abs(p2 - p1) / np.sqrt(p1 * (1-p1))
            required_n = int(2 * ((1.96 + 0.84) / effect_size) ** 2)
        else:
            required_n = None
    else:
        required_n = None
    
    # Winner determination
    if is_significant:
        winner = max(variant_results, key=lambda x: x['rate'])
        winner_name = winner['variant']
    else:
        winner_name = None
    
    return {
        'variants': variant_results,
        'chi_square': _to_native_type(chi2),
        'p_value': _to_native_type(p_value),
        'is_significant': is_significant,
        'confidence_level': confidence_level,
        'lifts': lifts,
        'winner': winner_name,
        'required_sample_size': required_n,
        'control': control['variant']
    }


def analyze_segments(df: pd.DataFrame, segment_col: str, converted_col: str) -> Dict[str, Any]:
    """Analyze conversion rates by segment"""
    
    segments = df[segment_col].unique()
    
    segment_results = []
    for segment in segments:
        segment_df = df[df[segment_col] == segment]
        visitors = len(segment_df)
        conversions = int(segment_df[converted_col].sum())
        
        cr = calculate_conversion_rate(conversions, visitors)
        cr['segment'] = str(segment)
        segment_results.append(cr)
    
    # Sort by conversion rate
    segment_results.sort(key=lambda x: x['rate'], reverse=True)
    
    # Calculate overall average
    total_visitors = len(df)
    total_conversions = int(df[converted_col].sum())
    overall_rate = total_conversions / total_visitors if total_visitors > 0 else 0
    
    # Find best and worst segments
    best_segment = segment_results[0] if segment_results else None
    worst_segment = segment_results[-1] if segment_results else None
    
    # Calculate segment lift vs average
    for seg in segment_results:
        if overall_rate > 0:
            seg['lift_vs_avg'] = _to_native_type((seg['rate'] - overall_rate) / overall_rate * 100)
        else:
            seg['lift_vs_avg'] = 0
    
    return {
        'segments': segment_results,
        'overall_rate': _to_native_type(overall_rate),
        'best_segment': best_segment['segment'] if best_segment else None,
        'best_rate': best_segment['rate'] if best_segment else None,
        'worst_segment': worst_segment['segment'] if worst_segment else None,
        'worst_rate': worst_segment['rate'] if worst_segment else None
    }


def analyze_trend(df: pd.DataFrame, date_col: str, converted_col: str,
                  visitor_col: Optional[str] = None) -> Dict[str, Any]:
    """Analyze conversion rate trend over time"""
    
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Group by date
    if visitor_col:
        daily = df.groupby(df[date_col].dt.date).agg({
            visitor_col: 'nunique',
            converted_col: 'sum'
        }).reset_index()
        daily.columns = ['date', 'visitors', 'conversions']
    else:
        daily = df.groupby(df[date_col].dt.date).agg({
            converted_col: ['count', 'sum']
        }).reset_index()
        daily.columns = ['date', 'visitors', 'conversions']
    
    daily['conversion_rate'] = daily['conversions'] / daily['visitors']
    daily['conversion_rate'] = daily['conversion_rate'].fillna(0)
    
    # Calculate trend
    if len(daily) >= 2:
        x = np.arange(len(daily))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, daily['conversion_rate'])
        trend_direction = 'increasing' if slope > 0.0001 else ('decreasing' if slope < -0.0001 else 'stable')
    else:
        slope, r_value, trend_direction = 0, 0, 'insufficient_data'
    
    # Calculate moving average
    if len(daily) >= 7:
        daily['ma_7'] = daily['conversion_rate'].rolling(7).mean()
    else:
        daily['ma_7'] = daily['conversion_rate']
    
    # Period comparison
    n = len(daily)
    if n >= 14:
        recent = daily.iloc[-7:]['conversion_rate'].mean()
        previous = daily.iloc[-14:-7]['conversion_rate'].mean()
        period_change = (recent - previous) / previous * 100 if previous > 0 else 0
    else:
        recent = daily['conversion_rate'].mean()
        previous = recent
        period_change = 0
    
    return {
        'daily_data': [
            {
                'date': str(row['date']),
                'visitors': int(row['visitors']),
                'conversions': int(row['conversions']),
                'conversion_rate': _to_native_type(row['conversion_rate']),
                'ma_7': _to_native_type(row['ma_7']) if pd.notna(row['ma_7']) else None
            }
            for _, row in daily.iterrows()
        ],
        'trend_slope': _to_native_type(slope),
        'trend_direction': trend_direction,
        'r_squared': _to_native_type(r_value ** 2),
        'current_rate': _to_native_type(recent),
        'previous_rate': _to_native_type(previous),
        'period_change_pct': _to_native_type(period_change)
    }


def calculate_revenue_metrics(df: pd.DataFrame, converted_col: str, 
                              value_col: str) -> Dict[str, Any]:
    """Calculate revenue-related conversion metrics"""
    
    total_visitors = len(df)
    converters = df[df[converted_col] == 1]
    
    total_revenue = converters[value_col].sum()
    avg_order_value = converters[value_col].mean() if len(converters) > 0 else 0
    revenue_per_visitor = total_revenue / total_visitors if total_visitors > 0 else 0
    
    conversion_rate = len(converters) / total_visitors if total_visitors > 0 else 0
    
    # Revenue distribution
    if len(converters) > 0:
        revenue_percentiles = {
            'p25': _to_native_type(converters[value_col].quantile(0.25)),
            'p50': _to_native_type(converters[value_col].quantile(0.50)),
            'p75': _to_native_type(converters[value_col].quantile(0.75)),
            'p90': _to_native_type(converters[value_col].quantile(0.90))
        }
    else:
        revenue_percentiles = {'p25': 0, 'p50': 0, 'p75': 0, 'p90': 0}
    
    return {
        'total_revenue': _to_native_type(total_revenue),
        'total_conversions': len(converters),
        'avg_order_value': _to_native_type(avg_order_value),
        'revenue_per_visitor': _to_native_type(revenue_per_visitor),
        'conversion_rate': _to_native_type(conversion_rate),
        'revenue_percentiles': revenue_percentiles
    }


def create_funnel_chart(funnel_data: Dict) -> str:
    """Create funnel visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    stages = funnel_data['stages']
    stage_names = [s['stage'] for s in stages]
    visitors = [s['visitors'] for s in stages]
    drop_rates = [s['drop_off_rate'] * 100 for s in stages]
    
    # Funnel chart
    colors = plt.cm.Blues(np.linspace(0.8, 0.3, len(stages)))
    
    y_pos = np.arange(len(stages))
    max_visitors = max(visitors)
    
    for i, (stage, v, color) in enumerate(zip(stage_names, visitors, colors)):
        width = v / max_visitors
        left = (1 - width) / 2
        ax1.barh(i, width, left=left, height=0.6, color=color, edgecolor='white', linewidth=2)
        ax1.text(0.5, i, f'{stage}\n{v:,} ({v/max_visitors*100:.1f}%)', 
                ha='center', va='center', fontsize=10, fontweight='bold')
    
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.5, len(stages) - 0.5)
    ax1.invert_yaxis()
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_title('Conversion Funnel', fontsize=14, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    
    # Drop-off chart
    if len(stages) > 1:
        drop_stages = stage_names[1:]
        drop_values = drop_rates[1:]
        colors2 = ['#ef4444' if d > 50 else '#f59e0b' if d > 30 else '#22c55e' for d in drop_values]
        
        bars = ax2.barh(drop_stages, drop_values, color=colors2, edgecolor='white', linewidth=2)
        ax2.set_xlabel('Drop-off Rate (%)')
        ax2.set_title('Stage Drop-off Rates', fontsize=14, fontweight='bold')
        ax2.set_xlim(0, 100)
        ax2.invert_yaxis()
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        for bar, val in zip(bars, drop_values):
            ax2.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                    f'{val:.1f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_ab_test_chart(ab_data: Dict) -> str:
    """Create A/B test visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    variants = ab_data['variants']
    names = [v['variant'] for v in variants]
    rates = [v['rate'] * 100 for v in variants]
    lowers = [v['lower'] * 100 for v in variants]
    uppers = [v['upper'] * 100 for v in variants]
    
    # Conversion rate comparison
    colors = ['#3b82f6' if i == 0 else '#22c55e' if v['rate'] == max(vv['rate'] for vv in variants) else '#6b7280' 
              for i, v in enumerate(variants)]
    
    bars = ax1.bar(names, rates, color=colors, edgecolor='white', linewidth=2)
    
    # Error bars
    for i, (bar, lower, upper, rate) in enumerate(zip(bars, lowers, uppers, rates)):
        ax1.plot([bar.get_x() + bar.get_width()/2] * 2, [lower, upper], 'k-', linewidth=2)
        ax1.plot([bar.get_x() + bar.get_width()/2 - 0.1, bar.get_x() + bar.get_width()/2 + 0.1], 
                [lower, lower], 'k-', linewidth=2)
        ax1.plot([bar.get_x() + bar.get_width()/2 - 0.1, bar.get_x() + bar.get_width()/2 + 0.1], 
                [upper, upper], 'k-', linewidth=2)
        ax1.text(bar.get_x() + bar.get_width()/2, rate + (upper - rate) + 1,
                f'{rate:.2f}%', ha='center', fontsize=10, fontweight='bold')
    
    ax1.set_ylabel('Conversion Rate (%)')
    ax1.set_title('Conversion Rate by Variant', fontsize=14, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Statistical significance indicator
    sig_text = "✓ Statistically Significant" if ab_data['is_significant'] else "✗ Not Significant"
    sig_color = 'green' if ab_data['is_significant'] else 'gray'
    ax1.text(0.5, -0.12, f"{sig_text} (p={ab_data['p_value']:.4f})", 
             transform=ax1.transAxes, ha='center', fontsize=10, color=sig_color)
    
    # Lift chart
    if ab_data['lifts']:
        lift_names = [l['variant'] for l in ab_data['lifts']]
        lift_values = [l['lift'] for l in ab_data['lifts']]
        lift_colors = ['#22c55e' if l > 0 else '#ef4444' for l in lift_values]
        
        bars2 = ax2.barh(lift_names, lift_values, color=lift_colors, edgecolor='white', linewidth=2)
        ax2.axvline(x=0, color='black', linewidth=1)
        ax2.set_xlabel('Lift vs Control (%)')
        ax2.set_title('Relative Lift', fontsize=14, fontweight='bold')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        for bar, val in zip(bars2, lift_values):
            ax2.text(bar.get_width() + (2 if val >= 0 else -8), bar.get_y() + bar.get_height()/2,
                    f'{val:+.1f}%', va='center', fontsize=10)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_segment_chart(segment_data: Dict) -> str:
    """Create segment analysis visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    segments = segment_data['segments']
    names = [s['segment'] for s in segments]
    rates = [s['rate'] * 100 for s in segments]
    lifts = [s['lift_vs_avg'] for s in segments]
    
    # Conversion rate by segment
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(segments)))
    colors = [colors[i] for i in np.argsort(np.argsort(rates))]
    
    bars = ax1.barh(names, rates, color=colors, edgecolor='white', linewidth=2)
    ax1.axvline(x=segment_data['overall_rate'] * 100, color='red', linestyle='--', 
                linewidth=2, label=f"Average: {segment_data['overall_rate']*100:.2f}%")
    ax1.set_xlabel('Conversion Rate (%)')
    ax1.set_title('Conversion Rate by Segment', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    for bar, rate in zip(bars, rates):
        ax1.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{rate:.2f}%', va='center', fontsize=9)
    
    # Lift vs average
    lift_colors = ['#22c55e' if l > 0 else '#ef4444' for l in lifts]
    bars2 = ax2.barh(names, lifts, color=lift_colors, edgecolor='white', linewidth=2)
    ax2.axvline(x=0, color='black', linewidth=1)
    ax2.set_xlabel('Lift vs Average (%)')
    ax2.set_title('Segment Performance vs Average', fontsize=14, fontweight='bold')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    for bar, val in zip(bars2, lifts):
        ax2.text(bar.get_width() + (1 if val >= 0 else -6), bar.get_y() + bar.get_height()/2,
                f'{val:+.1f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_trend_chart(trend_data: Dict) -> str:
    """Create trend visualization"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    
    daily = trend_data['daily_data']
    dates = [d['date'] for d in daily]
    rates = [d['conversion_rate'] * 100 for d in daily]
    ma7 = [d['ma_7'] * 100 if d['ma_7'] else None for d in daily]
    visitors = [d['visitors'] for d in daily]
    conversions = [d['conversions'] for d in daily]
    
    # Conversion rate trend
    ax1.plot(dates, rates, 'b-', alpha=0.4, linewidth=1, label='Daily')
    ax1.plot(dates, ma7, 'b-', linewidth=2, label='7-day MA')
    
    # Trend line
    x = np.arange(len(dates))
    z = np.polyfit(x, rates, 1)
    p = np.poly1d(z)
    ax1.plot(dates, p(x), 'r--', linewidth=2, alpha=0.7, label='Trend')
    
    ax1.set_ylabel('Conversion Rate (%)')
    ax1.set_title('Conversion Rate Trend', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Rotate x-axis labels
    if len(dates) > 20:
        ax1.set_xticks(ax1.get_xticks()[::len(dates)//10])
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Volume chart
    ax2.bar(dates, visitors, alpha=0.5, color='blue', label='Visitors')
    ax2.bar(dates, conversions, alpha=0.8, color='green', label='Conversions')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Count')
    ax2.set_title('Traffic and Conversions', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    if len(dates) > 20:
        ax2.set_xticks(ax2.get_xticks()[::len(dates)//10])
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_overview_chart(overall_metrics: Dict) -> str:
    """Create overall conversion metrics visualization"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Conversion rate gauge
    ax = axes[0]
    rate = overall_metrics['conversion_rate'] * 100
    
    # Create gauge
    theta = np.linspace(0, np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'lightgray', linewidth=20)
    
    # Fill based on rate (assuming 10% is max typical)
    fill_pct = min(rate / 10, 1)
    theta_fill = np.linspace(0, np.pi * fill_pct, 50)
    color = '#22c55e' if rate >= 3 else '#f59e0b' if rate >= 1 else '#ef4444'
    ax.plot(np.cos(theta_fill), np.sin(theta_fill), color, linewidth=20)
    
    ax.text(0, 0.2, f'{rate:.2f}%', ha='center', va='center', fontsize=24, fontweight='bold')
    ax.text(0, -0.2, 'Conversion Rate', ha='center', va='center', fontsize=12)
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-0.5, 1.2)
    ax.axis('off')
    
    # Visitors vs Conversions
    ax = axes[1]
    labels = ['Visitors', 'Conversions']
    values = [overall_metrics['visitors'], overall_metrics['conversions']]
    colors = ['#3b82f6', '#22c55e']
    
    bars = ax.bar(labels, values, color=colors, edgecolor='white', linewidth=2)
    ax.set_title('Traffic Overview', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values) * 0.02,
               f'{val:,}', ha='center', fontsize=10)
    
    # Revenue metrics (if available)
    ax = axes[2]
    if overall_metrics.get('revenue_per_visitor'):
        metrics = [
            ('AOV', overall_metrics.get('avg_order_value', 0)),
            ('RPV', overall_metrics.get('revenue_per_visitor', 0))
        ]
        labels = [m[0] for m in metrics]
        values = [m[1] for m in metrics]
        
        bars = ax.bar(labels, values, color=['#8b5cf6', '#ec4899'], edgecolor='white', linewidth=2)
        ax.set_title('Revenue Metrics', fontsize=12, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values) * 0.02,
                   f'${val:,.2f}', ha='center', fontsize=10)
    else:
        ax.text(0.5, 0.5, 'No revenue data', ha='center', va='center', transform=ax.transAxes)
        ax.axis('off')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_insights(results: Dict) -> List[Dict[str, Any]]:
    """Generate key insights from conversion analysis"""
    insights = []
    
    # Overall conversion rate insight
    overall = results.get('overall_metrics', {})
    rate = overall.get('conversion_rate', 0)
    
    if rate >= 0.05:
        insights.append({
            'title': f'Strong Conversion Rate: {rate*100:.2f}%',
            'description': 'Above industry average (2-5%). Focus on scaling traffic.',
            'status': 'positive'
        })
    elif rate >= 0.02:
        insights.append({
            'title': f'Average Conversion Rate: {rate*100:.2f}%',
            'description': 'Within typical range. Optimization opportunities exist.',
            'status': 'neutral'
        })
    else:
        insights.append({
            'title': f'Low Conversion Rate: {rate*100:.2f}%',
            'description': 'Below average. Focus on UX and value proposition.',
            'status': 'warning'
        })
    
    # Funnel insights
    funnel = results.get('funnel_analysis')
    if funnel and funnel.get('biggest_dropoff_stage'):
        insights.append({
            'title': f'Biggest Drop-off: {funnel["biggest_dropoff_stage"]}',
            'description': f'{funnel["biggest_dropoff_rate"]*100:.1f}% drop-off. Priority optimization area.',
            'status': 'warning'
        })
    
    # A/B test insights
    ab = results.get('ab_test')
    if ab and ab.get('is_significant'):
        winner = ab.get('winner')
        insights.append({
            'title': f'A/B Test Winner: {winner}',
            'description': f'Statistically significant result (p={ab["p_value"]:.4f}). Implement winner.',
            'status': 'positive'
        })
    elif ab and not ab.get('is_significant'):
        insights.append({
            'title': 'A/B Test Inconclusive',
            'description': 'No significant difference. Need more data or larger effect size.',
            'status': 'neutral'
        })
    
    # Segment insights
    segments = results.get('segment_analysis')
    if segments:
        best = segments.get('best_segment')
        worst = segments.get('worst_segment')
        if best and worst and best != worst:
            insights.append({
                'title': f'Best Segment: {best}',
                'description': f'Conversion: {segments["best_rate"]*100:.2f}%. Focus acquisition here.',
                'status': 'positive'
            })
    
    # Trend insights
    trend = results.get('trend_analysis')
    if trend:
        direction = trend.get('trend_direction')
        change = trend.get('period_change_pct', 0)
        if direction == 'increasing':
            insights.append({
                'title': f'Positive Trend: +{change:.1f}% WoW',
                'description': 'Conversion rate is improving. Continue current strategies.',
                'status': 'positive'
            })
        elif direction == 'decreasing':
            insights.append({
                'title': f'Negative Trend: {change:.1f}% WoW',
                'description': 'Conversion rate declining. Investigate recent changes.',
                'status': 'warning'
            })
    
    return insights


@router.post("/conversion")
async def run_conversion_analysis(request: ConversionRequest) -> Dict[str, Any]:
    """
    Perform Conversion Rate Analysis.
    """
    try:
        df = pd.DataFrame(request.data)
        
        # Validate minimum requirements
        if len(df) == 0:
            raise HTTPException(status_code=400, detail="No data provided")
        
        results = {}
        visualizations = {}
        
        # Overall metrics
        if request.converted_col and request.converted_col in df.columns:
            visitors = len(df)
            conversions = int(df[request.converted_col].sum())
            overall_cr = calculate_conversion_rate(conversions, visitors)
            
            overall_metrics = {
                'visitors': visitors,
                'conversions': conversions,
                'conversion_rate': overall_cr['rate'],
                'ci_lower': overall_cr['lower'],
                'ci_upper': overall_cr['upper']
            }
            
            # Revenue metrics
            if request.value_col and request.value_col in df.columns:
                revenue = calculate_revenue_metrics(df, request.converted_col, request.value_col)
                overall_metrics.update(revenue)
            
            results['overall_metrics'] = overall_metrics
            visualizations['overview_chart'] = create_overview_chart(overall_metrics)
        
        # Funnel analysis
        if request.stage_col and request.stage_col in df.columns:
            funnel = analyze_funnel(df, request.stage_col, request.visitor_col or 'index',
                                   request.funnel_stages)
            results['funnel_analysis'] = funnel
            visualizations['funnel_chart'] = create_funnel_chart(funnel)
        
        # A/B test analysis
        if request.variant_col and request.variant_col in df.columns and request.converted_col:
            ab = analyze_ab_test(df, request.variant_col, request.converted_col,
                               request.confidence_level)
            results['ab_test'] = ab
            if 'error' not in ab:
                visualizations['ab_test_chart'] = create_ab_test_chart(ab)
        
        # Segment analysis
        if request.segment_col and request.segment_col in df.columns and request.converted_col:
            segments = analyze_segments(df, request.segment_col, request.converted_col)
            results['segment_analysis'] = segments
            visualizations['segment_chart'] = create_segment_chart(segments)
        
        # Trend analysis
        if request.date_col and request.date_col in df.columns and request.converted_col:
            trend = analyze_trend(df, request.date_col, request.converted_col, request.visitor_col)
            results['trend_analysis'] = trend
            visualizations['trend_chart'] = create_trend_chart(trend)
        
        # Generate insights
        insights = generate_insights(results)
        
        # Summary
        summary = {
            'total_visitors': results.get('overall_metrics', {}).get('visitors', len(df)),
            'total_conversions': results.get('overall_metrics', {}).get('conversions', 0),
            'conversion_rate': results.get('overall_metrics', {}).get('conversion_rate', 0),
            'has_funnel': 'funnel_analysis' in results,
            'has_ab_test': 'ab_test' in results,
            'has_segments': 'segment_analysis' in results,
            'has_trend': 'trend_analysis' in results
        }
        
        return {
            'success': True,
            'results': results,
            'visualizations': visualizations,
            'key_insights': insights,
            'summary': summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversion analysis failed: {str(e)}")
