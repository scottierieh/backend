"""
Conversion Rate Analysis Router for FastAPI
Analyzes conversion rates by segments with statistical testing
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import io
import base64
import time
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class ConversionRequest(BaseModel):
    data: List[Dict[str, Any]]
    visitor_id_col: str
    converted_col: str
    revenue_col: Optional[str] = None
    date_col: Optional[str] = None
    segment_cols: List[str]
    funnel_cols: Optional[List[str]] = None


def _to_native_type(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return 0.0
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _fig_to_base64(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return image_base64


def calculate_segment_metrics(df: pd.DataFrame, segment_col: str, overall_rate: float) -> List[Dict]:
    """Calculate conversion metrics for each segment value"""
    results = []
    
    for value in df[segment_col].dropna().unique():
        seg_df = df[df[segment_col] == value]
        visitors = len(seg_df)
        conversions = seg_df['_converted'].sum()
        revenue = seg_df['_revenue'].sum() if '_revenue' in seg_df.columns else 0
        
        rate = (conversions / visitors * 100) if visitors > 0 else 0
        aov = (revenue / conversions) if conversions > 0 else 0
        rate_vs_avg = ((rate - overall_rate) / overall_rate * 100) if overall_rate > 0 else 0
        
        results.append({
            'segment': segment_col,
            'segment_value': str(value),
            'visitors': int(visitors),
            'conversions': int(conversions),
            'conversion_rate': float(rate),
            'revenue': float(revenue),
            'avg_order_value': float(aov),
            'rate_vs_avg': float(rate_vs_avg),
        })
    
    return sorted(results, key=lambda x: x['conversion_rate'], reverse=True)


def calculate_statistical_tests(df: pd.DataFrame, segment_col: str, overall_rate: float, overall_n: int) -> List[Dict]:
    """Perform z-tests for each segment value vs overall"""
    results = []
    p_overall = overall_rate / 100  # Convert to proportion
    
    for value in df[segment_col].dropna().unique():
        seg_df = df[df[segment_col] == value]
        n = len(seg_df)
        conversions = seg_df['_converted'].sum()
        p_seg = conversions / n if n > 0 else 0
        
        # Z-test for proportion
        if n > 0 and p_overall > 0 and p_overall < 1:
            # Pooled standard error
            se = np.sqrt(p_overall * (1 - p_overall) * (1/n + 1/overall_n))
            z_score = (p_seg - p_overall) / se if se > 0 else 0
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))  # Two-tailed
            
            # 95% Confidence interval for segment
            se_seg = np.sqrt(p_seg * (1 - p_seg) / n) if n > 0 else 0
            ci_lower = max(0, (p_seg - 1.96 * se_seg) * 100)
            ci_upper = min(100, (p_seg + 1.96 * se_seg) * 100)
        else:
            z_score = 0
            p_value = 1
            ci_lower = 0
            ci_upper = 0
        
        results.append({
            'segment': segment_col,
            'segment_value': str(value),
            'z_score': float(z_score),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
        })
    
    return results


def calculate_funnel(df: pd.DataFrame, funnel_cols: List[str]) -> List[Dict]:
    """Calculate funnel drop-off rates"""
    results = []
    total = len(df)
    
    for i, col in enumerate(funnel_cols):
        if col not in df.columns:
            continue
        
        visitors = df[col].sum()
        drop_rate = 0
        
        if i > 0:
            prev_visitors = results[i-1]['visitors'] if i > 0 and results else total
            drop_rate = ((prev_visitors - visitors) / prev_visitors * 100) if prev_visitors > 0 else 0
        
        cumulative_rate = (visitors / total * 100) if total > 0 else 0
        
        results.append({
            'step': col.replace('_', ' ').title(),
            'visitors': int(visitors),
            'drop_rate': float(drop_rate),
            'cumulative_rate': float(cumulative_rate),
        })
    
    return results


def calculate_time_trends(df: pd.DataFrame, date_col: str) -> List[Dict]:
    """Calculate conversion rate over time"""
    if date_col not in df.columns:
        return []
    
    df['_date'] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=['_date'])
    
    if len(df) == 0:
        return []
    
    # Group by week
    df['_week'] = df['_date'].dt.to_period('W')
    
    results = []
    for period, group in df.groupby('_week'):
        visitors = len(group)
        conversions = group['_converted'].sum()
        rate = (conversions / visitors * 100) if visitors > 0 else 0
        
        results.append({
            'period': str(period),
            'visitors': int(visitors),
            'conversions': int(conversions),
            'conversion_rate': float(rate),
        })
    
    return sorted(results, key=lambda x: x['period'])


# ============ VISUALIZATION ============
def create_segment_comparison_chart(segment_analysis: List[Dict], overall_rate: float) -> str:
    """Create bar chart comparing segment conversion rates"""
    # Group by segment and take top values
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Take top 15 by rate
    top_segments = sorted(segment_analysis, key=lambda x: x['conversion_rate'], reverse=True)[:15]
    
    labels = [f"{s['segment']}: {s['segment_value']}" for s in top_segments]
    rates = [s['conversion_rate'] for s in top_segments]
    
    colors = ['#22c55e' if r >= overall_rate * 1.2 else '#3b82f6' if r >= overall_rate else '#f59e0b' if r >= overall_rate * 0.7 else '#ef4444' for r in rates]
    
    bars = ax.barh(labels, rates, color=colors, edgecolor='white')
    ax.axvline(x=overall_rate, color='#64748b', linestyle='--', linewidth=2, label=f'Overall: {overall_rate:.2f}%')
    
    ax.set_xlabel('Conversion Rate (%)', fontsize=11)
    ax.set_title('Conversion Rate by Segment', fontsize=14, fontweight='bold')
    ax.legend()
    ax.invert_yaxis()
    
    for bar, rate in zip(bars, rates):
        ax.text(rate + 0.1, bar.get_y() + bar.get_height()/2, f'{rate:.2f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_funnel_chart(funnel_analysis: List[Dict]) -> str:
    """Create funnel visualization"""
    if not funnel_analysis:
        return ""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    steps = [f['step'] for f in funnel_analysis]
    visitors = [f['visitors'] for f in funnel_analysis]
    max_visitors = max(visitors) if visitors else 1
    
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(steps)))
    
    for i, (step, count, color) in enumerate(zip(steps, visitors, colors)):
        width = count / max_visitors
        left = (1 - width) / 2
        
        ax.barh(i, width, left=left, height=0.6, color=color, edgecolor='white')
        ax.text(0.5, i, f'{step}\n{count:,}', ha='center', va='center', fontsize=10, fontweight='bold')
        
        if i > 0:
            drop = funnel_analysis[i]['drop_rate']
            ax.text(0.95, i - 0.3, f'-{drop:.1f}%', ha='right', va='center', fontsize=9, color='#ef4444')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(steps) - 0.5)
    ax.axis('off')
    ax.set_title('Conversion Funnel', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_rate_distribution_chart(segment_analysis: List[Dict]) -> str:
    """Create histogram of conversion rates"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    rates = [s['conversion_rate'] for s in segment_analysis]
    
    ax.hist(rates, bins=15, color='#3b82f6', edgecolor='white', alpha=0.8)
    ax.axvline(x=np.mean(rates), color='#ef4444', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rates):.2f}%')
    
    ax.set_xlabel('Conversion Rate (%)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Distribution of Segment Conversion Rates', fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_time_trend_chart(time_trends: List[Dict]) -> str:
    """Create time series of conversion rate"""
    if not time_trends:
        return ""
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    periods = [t['period'] for t in time_trends]
    rates = [t['conversion_rate'] for t in time_trends]
    
    ax.plot(range(len(periods)), rates, marker='o', color='#3b82f6', linewidth=2, markersize=6)
    ax.fill_between(range(len(periods)), rates, alpha=0.2, color='#3b82f6')
    
    ax.axhline(y=np.mean(rates), color='#64748b', linestyle='--', alpha=0.7, label=f'Avg: {np.mean(rates):.2f}%')
    
    ax.set_xticks(range(0, len(periods), max(1, len(periods)//10)))
    ax.set_xticklabels([periods[i] for i in range(0, len(periods), max(1, len(periods)//10))], rotation=45, ha='right')
    
    ax.set_xlabel('Period', fontsize=11)
    ax.set_ylabel('Conversion Rate (%)', fontsize=11)
    ax.set_title('Conversion Rate Over Time', fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_recommendations(summary: Dict, segment_analysis: List[Dict], funnel_analysis: List[Dict]) -> List[Dict]:
    """Generate optimization recommendations"""
    recommendations = []
    overall_rate = summary['overall_rate']
    
    # Low performers
    low_performers = [s for s in segment_analysis if s['rate_vs_avg'] < -20]
    if low_performers:
        worst = min(low_performers, key=lambda x: x['conversion_rate'])
        recommendations.append({
            'priority': 'High', 'category': 'Segment Optimization',
            'recommendation': f"Investigate low conversion in {worst['segment']}: {worst['segment_value']} ({worst['conversion_rate']:.2f}%)",
            'impact': f"Potential {abs(worst['rate_vs_avg']):.0f}% improvement opportunity",
        })
    
    # High performers to scale
    high_performers = [s for s in segment_analysis if s['rate_vs_avg'] > 30 and s['visitors'] > 100]
    if high_performers:
        best = max(high_performers, key=lambda x: x['conversion_rate'])
        recommendations.append({
            'priority': 'Medium', 'category': 'Growth Opportunity',
            'recommendation': f"Scale successful segment {best['segment']}: {best['segment_value']} ({best['conversion_rate']:.2f}%)",
            'impact': "Increase traffic to high-converting segments",
        })
    
    # Funnel drop-offs
    if funnel_analysis:
        worst_drop = max(funnel_analysis[1:], key=lambda x: x['drop_rate']) if len(funnel_analysis) > 1 else None
        if worst_drop and worst_drop['drop_rate'] > 30:
            recommendations.append({
                'priority': 'High', 'category': 'Funnel Optimization',
                'recommendation': f"Fix {worst_drop['drop_rate']:.0f}% drop-off at '{worst_drop['step']}' step",
                'impact': "Significant conversion improvement potential",
            })
    
    # Overall rate check
    if overall_rate < 2:
        recommendations.append({
            'priority': 'High', 'category': 'Overall Performance',
            'recommendation': "Conversion rate below industry average. Review UX and value proposition.",
            'impact': "Target 2-3% benchmark rate",
        })
    
    return recommendations


def generate_key_insights(summary: Dict, segment_analysis: List[Dict], statistical_tests: List[Dict]) -> List[Dict]:
    """Generate key insights"""
    insights = []
    
    # Overall rate insight
    rate = summary['overall_rate']
    if rate >= 4:
        insights.append({'title': f"Strong Conversion Rate ({rate:.2f}%)", 'description': "Above industry benchmarks.", 'status': 'positive'})
    elif rate >= 2:
        insights.append({'title': f"Solid Conversion Rate ({rate:.2f}%)", 'description': "Within industry average range.", 'status': 'neutral'})
    else:
        insights.append({'title': f"Below Average Rate ({rate:.2f}%)", 'description': "Optimization opportunities exist.", 'status': 'warning'})
    
    # Best segment
    if segment_analysis:
        best = max(segment_analysis, key=lambda x: x['conversion_rate'])
        insights.append({
            'title': f"Best: {best['segment']}: {best['segment_value']}",
            'description': f"Conversion rate of {best['conversion_rate']:.2f}% ({best['rate_vs_avg']:+.0f}% vs avg)",
            'status': 'positive'
        })
    
    # Significant segments
    sig_count = len([t for t in statistical_tests if t['significant']])
    if sig_count > 0:
        insights.append({
            'title': f"{sig_count} Statistically Significant Segments",
            'description': "These segments differ significantly from the overall rate.",
            'status': 'neutral'
        })
    
    # Revenue insight
    if summary['total_revenue'] > 0:
        insights.append({
            'title': f"Revenue: ${summary['total_revenue']:,.0f}",
            'description': f"Average order value: ${summary['avg_order_value']:.2f}",
            'status': 'positive'
        })
    
    return insights


@router.post("/conversion-rate")
async def run_conversion_analysis(request: ConversionRequest) -> Dict[str, Any]:
    try:
        start_time = time.time()
        df = pd.DataFrame(request.data)
        
        # Validate required columns
        for col in [request.visitor_id_col, request.converted_col]:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{col}' not found")
        
        # Prepare data
        df['_converted'] = pd.to_numeric(df[request.converted_col], errors='coerce').fillna(0).astype(int)
        df['_revenue'] = pd.to_numeric(df[request.revenue_col], errors='coerce').fillna(0) if request.revenue_col and request.revenue_col in df.columns else 0
        
        # Overall metrics
        total_visitors = len(df)
        total_conversions = df['_converted'].sum()
        overall_rate = (total_conversions / total_visitors * 100) if total_visitors > 0 else 0
        total_revenue = df['_revenue'].sum()
        avg_order_value = (total_revenue / total_conversions) if total_conversions > 0 else 0
        
        # Segment analysis
        segment_analysis = []
        statistical_tests = []
        
        for seg_col in request.segment_cols:
            if seg_col in df.columns:
                seg_metrics = calculate_segment_metrics(df, seg_col, overall_rate)
                segment_analysis.extend(seg_metrics)
                
                seg_tests = calculate_statistical_tests(df, seg_col, overall_rate, total_visitors)
                statistical_tests.extend(seg_tests)
        
        # Sort by conversion rate
        segment_analysis.sort(key=lambda x: x['conversion_rate'], reverse=True)
        
        # Find best/worst
        best_segment = f"{segment_analysis[0]['segment']}: {segment_analysis[0]['segment_value']}" if segment_analysis else 'N/A'
        worst_segment = f"{segment_analysis[-1]['segment']}: {segment_analysis[-1]['segment_value']}" if segment_analysis else 'N/A'
        
        # Funnel analysis
        funnel_analysis = []
        if request.funnel_cols:
            funnel_analysis = calculate_funnel(df, request.funnel_cols)
        
        # Time trends
        time_trends = []
        if request.date_col and request.date_col in df.columns:
            time_trends = calculate_time_trends(df, request.date_col)
        
        summary_data = {
            'total_visitors': int(total_visitors),
            'total_conversions': int(total_conversions),
            'overall_rate': float(overall_rate),
            'total_revenue': float(total_revenue),
            'avg_order_value': float(avg_order_value),
            'best_segment': best_segment,
            'worst_segment': worst_segment,
        }
        
        solve_time_ms = int((time.time() - start_time) * 1000)
        
        # Visualizations
        visualizations = {
            'segment_comparison': create_segment_comparison_chart(segment_analysis, overall_rate),
            'funnel_chart': create_funnel_chart(funnel_analysis) if funnel_analysis else None,
            'rate_distribution': create_rate_distribution_chart(segment_analysis),
            'time_trend': create_time_trend_chart(time_trends) if time_trends else None,
        }
        
        # Recommendations and insights
        recommendations = generate_recommendations(summary_data, segment_analysis, funnel_analysis)
        key_insights = generate_key_insights(summary_data, segment_analysis, statistical_tests)
        
        return {
            'success': True,
            'results': {
                'summary': {k: _to_native_type(v) for k, v in summary_data.items()},
                'segment_analysis': segment_analysis,
                'funnel_analysis': funnel_analysis,
                'time_trends': time_trends,
                'statistical_tests': statistical_tests,
                'recommendations': recommendations,
            },
            'visualizations': visualizations,
            'key_insights': key_insights,
            'summary': {
                'analysis_date': datetime.now().strftime('%Y-%m-%d'),
                'segments_analyzed': len(segment_analysis),
                'solve_time_ms': solve_time_ms,
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Conversion analysis failed: {str(e)}")
