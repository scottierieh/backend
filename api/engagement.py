"""
Employee Engagement Survey Analysis Router for FastAPI
Engagement Drivers, eNPS, Sentiment Analysis, Department Comparison
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class EngagementRequest(BaseModel):
    data: List[Dict[str, Any]]
    employee_id_col: Optional[str] = None
    department_col: Optional[str] = None
    tenure_col: Optional[str] = None
    job_level_col: Optional[str] = None
    overall_engagement_col: Optional[str] = None  # Overall engagement score
    enps_col: Optional[str] = None  # eNPS question (0-10)
    dimension_cols: Optional[List[str]] = None  # Multiple engagement dimensions
    # Common dimensions: Leadership, Growth, Recognition, Work-Life Balance, etc.


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
    return obj


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string"""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return image_base64


def calculate_overall_engagement(df: pd.DataFrame, engagement_col: str) -> Dict[str, Any]:
    """Calculate overall engagement statistics"""
    
    scores = df[engagement_col].dropna()
    
    # Determine scale
    max_score = scores.max()
    if max_score <= 5:
        scale = 5
    elif max_score <= 7:
        scale = 7
    else:
        scale = 10
    
    # Normalize to 100
    normalized = scores / scale * 100
    
    # Categorize
    high_threshold = scale * 0.8
    low_threshold = scale * 0.6
    
    highly_engaged = (scores >= high_threshold).sum()
    moderately_engaged = ((scores >= low_threshold) & (scores < high_threshold)).sum()
    disengaged = (scores < low_threshold).sum()
    
    return {
        'mean_score': _to_native_type(scores.mean()),
        'median_score': _to_native_type(scores.median()),
        'std_score': _to_native_type(scores.std()),
        'min_score': _to_native_type(scores.min()),
        'max_score': _to_native_type(scores.max()),
        'scale': scale,
        'normalized_score': _to_native_type(normalized.mean()),
        'highly_engaged_count': int(highly_engaged),
        'highly_engaged_pct': _to_native_type(highly_engaged / len(scores) * 100),
        'moderately_engaged_count': int(moderately_engaged),
        'moderately_engaged_pct': _to_native_type(moderately_engaged / len(scores) * 100),
        'disengaged_count': int(disengaged),
        'disengaged_pct': _to_native_type(disengaged / len(scores) * 100),
        'response_count': len(scores)
    }


def calculate_enps(df: pd.DataFrame, enps_col: str) -> Dict[str, Any]:
    """Calculate Employee Net Promoter Score"""
    
    scores = df[enps_col].dropna()
    
    # eNPS scale is 0-10
    promoters = (scores >= 9).sum()
    passives = ((scores >= 7) & (scores < 9)).sum()
    detractors = (scores < 7).sum()
    
    total = len(scores)
    
    promoter_pct = promoters / total * 100
    detractor_pct = detractors / total * 100
    
    enps = promoter_pct - detractor_pct
    
    return {
        'enps_score': _to_native_type(enps),
        'promoters_count': int(promoters),
        'promoters_pct': _to_native_type(promoter_pct),
        'passives_count': int(passives),
        'passives_pct': _to_native_type(passives / total * 100),
        'detractors_count': int(detractors),
        'detractors_pct': _to_native_type(detractor_pct),
        'mean_score': _to_native_type(scores.mean()),
        'total_responses': total,
        'interpretation': 'Excellent' if enps >= 50 else ('Good' if enps >= 20 else ('Needs Improvement' if enps >= 0 else 'Critical'))
    }


def analyze_dimensions(df: pd.DataFrame, dimension_cols: List[str],
                       engagement_col: Optional[str] = None) -> Dict[str, Any]:
    """Analyze engagement by dimensions"""
    
    dimension_stats = []
    
    for col in dimension_cols:
        if col not in df.columns:
            continue
            
        scores = df[col].dropna()
        if len(scores) < 5:
            continue
        
        # Determine scale
        max_score = scores.max()
        scale = 5 if max_score <= 5 else (7 if max_score <= 7 else 10)
        normalized = scores.mean() / scale * 100
        
        stat = {
            'dimension': col,
            'mean_score': _to_native_type(scores.mean()),
            'median_score': _to_native_type(scores.median()),
            'std_score': _to_native_type(scores.std()),
            'scale': scale,
            'normalized_score': _to_native_type(normalized),
            'response_count': len(scores)
        }
        
        # Calculate correlation with overall engagement if available
        if engagement_col and engagement_col in df.columns:
            valid = df[[col, engagement_col]].dropna()
            if len(valid) >= 10:
                corr, p_val = stats.pearsonr(valid[col], valid[engagement_col])
                stat['correlation_with_engagement'] = _to_native_type(corr)
                stat['correlation_p_value'] = _to_native_type(p_val)
                stat['is_significant_driver'] = p_val < 0.05 and corr > 0.3
        
        dimension_stats.append(stat)
    
    # Sort by normalized score
    dimension_stats.sort(key=lambda x: x['normalized_score'], reverse=True)
    
    # Identify strengths and weaknesses
    if dimension_stats:
        avg_normalized = np.mean([d['normalized_score'] for d in dimension_stats])
        strengths = [d['dimension'] for d in dimension_stats if d['normalized_score'] > avg_normalized + 5]
        weaknesses = [d['dimension'] for d in dimension_stats if d['normalized_score'] < avg_normalized - 5]
    else:
        strengths, weaknesses = [], []
    
    return {
        'dimensions': dimension_stats,
        'strengths': strengths[:3],
        'weaknesses': weaknesses[:3],
        'avg_normalized_score': _to_native_type(avg_normalized) if dimension_stats else None
    }


def identify_engagement_drivers(df: pd.DataFrame, dimension_cols: List[str],
                                 engagement_col: str) -> List[Dict[str, Any]]:
    """Identify key drivers of engagement using regression"""
    
    # Prepare data
    valid_cols = [c for c in dimension_cols if c in df.columns and c != engagement_col]
    if not valid_cols:
        return []
    
    valid_df = df[valid_cols + [engagement_col]].dropna()
    if len(valid_df) < 20:
        return []
    
    X = valid_df[valid_cols].values
    y = valid_df[engagement_col].values
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit regression
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    # Get standardized coefficients (importance)
    drivers = []
    for i, col in enumerate(valid_cols):
        drivers.append({
            'dimension': col,
            'coefficient': _to_native_type(model.coef_[i]),
            'importance': _to_native_type(abs(model.coef_[i])),
            'direction': 'positive' if model.coef_[i] > 0 else 'negative'
        })
    
    # Sort by importance
    drivers.sort(key=lambda x: x['importance'], reverse=True)
    
    # Calculate R-squared
    r_squared = model.score(X_scaled, y)
    
    return {
        'drivers': drivers,
        'r_squared': _to_native_type(r_squared),
        'model_explanation': f"Model explains {r_squared*100:.1f}% of engagement variance"
    }


def analyze_by_segment(df: pd.DataFrame, segment_col: str,
                       engagement_col: str) -> List[Dict[str, Any]]:
    """Analyze engagement by segment (department, tenure, etc.)"""
    
    segments = df[segment_col].unique()
    overall_mean = df[engagement_col].mean()
    
    segment_stats = []
    
    for segment in segments:
        seg_df = df[df[segment_col] == segment]
        scores = seg_df[engagement_col].dropna()
        
        if len(scores) < 3:
            continue
        
        segment_stats.append({
            'segment': str(segment),
            'mean_score': _to_native_type(scores.mean()),
            'median_score': _to_native_type(scores.median()),
            'std_score': _to_native_type(scores.std()),
            'response_count': len(scores),
            'vs_overall': _to_native_type(scores.mean() - overall_mean),
            'vs_overall_pct': _to_native_type((scores.mean() - overall_mean) / overall_mean * 100) if overall_mean > 0 else 0
        })
    
    segment_stats.sort(key=lambda x: x['mean_score'], reverse=True)
    
    # ANOVA test
    groups = [df[df[segment_col] == s][engagement_col].dropna().values for s in segments if len(df[df[segment_col] == s]) >= 3]
    if len(groups) >= 2:
        f_stat, p_value = stats.f_oneway(*groups)
        significant_difference = p_value < 0.05
    else:
        f_stat, p_value, significant_difference = None, None, None
    
    return {
        'segments': segment_stats,
        'overall_mean': _to_native_type(overall_mean),
        'f_statistic': _to_native_type(f_stat),
        'p_value': _to_native_type(p_value),
        'significant_difference': significant_difference,
        'highest_segment': segment_stats[0]['segment'] if segment_stats else None,
        'lowest_segment': segment_stats[-1]['segment'] if segment_stats else None
    }


def analyze_by_tenure(df: pd.DataFrame, tenure_col: str,
                      engagement_col: str) -> Dict[str, Any]:
    """Analyze engagement by tenure"""
    
    valid_df = df[[tenure_col, engagement_col]].dropna()
    
    if len(valid_df) < 10:
        return {'error': 'Insufficient data'}
    
    # Correlation
    corr, p_value = stats.pearsonr(valid_df[tenure_col], valid_df[engagement_col])
    
    # Create tenure buckets
    tenure_buckets = pd.cut(valid_df[tenure_col], 
                            bins=[0, 1, 3, 5, 10, float('inf')],
                            labels=['<1 year', '1-3 years', '3-5 years', '5-10 years', '10+ years'])
    
    bucket_stats = []
    for bucket in tenure_buckets.unique():
        if pd.isna(bucket):
            continue
        bucket_df = valid_df[tenure_buckets == bucket]
        if len(bucket_df) >= 3:
            bucket_stats.append({
                'tenure_bucket': str(bucket),
                'mean_score': _to_native_type(bucket_df[engagement_col].mean()),
                'response_count': len(bucket_df)
            })
    
    return {
        'correlation': _to_native_type(corr),
        'p_value': _to_native_type(p_value),
        'is_significant': p_value < 0.05,
        'trend': 'positive' if corr > 0.1 else ('negative' if corr < -0.1 else 'neutral'),
        'bucket_analysis': bucket_stats
    }


def create_engagement_overview_chart(overall_data: Dict, enps_data: Optional[Dict]) -> str:
    """Create engagement overview visualization"""
    fig, axes = plt.subplots(1, 2 if enps_data else 1, figsize=(14 if enps_data else 8, 5))
    
    if enps_data:
        ax1, ax2 = axes
    else:
        ax1 = axes
    
    # Engagement distribution pie chart
    labels = ['Highly Engaged', 'Moderately Engaged', 'Disengaged']
    sizes = [overall_data['highly_engaged_count'], overall_data['moderately_engaged_count'], 
             overall_data['disengaged_count']]
    colors = ['#22c55e', '#fbbf24', '#ef4444']
    explode = (0.05, 0, 0)
    
    wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
                                        autopct='%1.1f%%', startangle=90)
    ax1.set_title(f'Engagement Distribution (Avg: {overall_data["normalized_score"]:.0f}%)', 
                 fontsize=12, fontweight='bold')
    
    # eNPS chart
    if enps_data:
        categories = ['Promoters\n(9-10)', 'Passives\n(7-8)', 'Detractors\n(0-6)']
        counts = [enps_data['promoters_count'], enps_data['passives_count'], enps_data['detractors_count']]
        colors2 = ['#22c55e', '#94a3b8', '#ef4444']
        
        bars = ax2.bar(categories, counts, color=colors2, edgecolor='white', linewidth=2)
        ax2.set_ylabel('Number of Responses')
        
        enps = enps_data['enps_score']
        enps_color = '#22c55e' if enps >= 20 else ('#fbbf24' if enps >= 0 else '#ef4444')
        ax2.set_title(f'eNPS: {enps:.0f} ({enps_data["interpretation"]})', 
                     fontsize=12, fontweight='bold', color=enps_color)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        for bar, count, pct in zip(bars, counts, [enps_data['promoters_pct'], enps_data['passives_pct'], enps_data['detractors_pct']]):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts) * 0.02,
                    f'{count}\n({pct:.0f}%)', ha='center', fontsize=9)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_dimensions_chart(dimension_data: Dict) -> str:
    """Create dimensions visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    dimensions = dimension_data['dimensions'][:10]
    names = [d['dimension'][:20] for d in dimensions]
    scores = [d['normalized_score'] for d in dimensions]
    avg_score = dimension_data['avg_normalized_score'] or np.mean(scores)
    
    # Horizontal bar chart
    colors = ['#22c55e' if s >= avg_score else '#ef4444' for s in scores]
    bars = ax1.barh(names[::-1], scores[::-1], color=colors[::-1], edgecolor='white', linewidth=2)
    ax1.axvline(x=avg_score, color='blue', linestyle='--', linewidth=2, label=f'Average: {avg_score:.0f}%')
    ax1.set_xlabel('Score (%)')
    ax1.set_xlim(0, 100)
    ax1.set_title('Engagement by Dimension', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    for bar, score in zip(bars, scores[::-1]):
        ax1.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{score:.0f}%', va='center', fontsize=9)
    
    # Radar chart (top 6 dimensions)
    top_dims = dimensions[:6]
    if len(top_dims) >= 3:
        categories = [d['dimension'][:15] for d in top_dims]
        values = [d['normalized_score'] for d in top_dims]
        
        # Complete the loop
        values += values[:1]
        angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
        angles += angles[:1]
        
        # Create polar subplot FIRST, then set properties
        ax2 = plt.subplot(122, polar=True)
        ax2.set_theta_offset(np.pi / 2)
        ax2.set_theta_direction(-1)
        
        ax2.plot(angles, values, 'o-', linewidth=2, color='#3b82f6')
        ax2.fill(angles, values, alpha=0.25, color='#3b82f6')
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(categories, size=9)
        ax2.set_ylim(0, 100)
        ax2.set_title('Dimension Profile', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_segment_chart(segment_data: Dict, segment_name: str) -> str:
    """Create segment comparison visualization"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    segments = segment_data['segments'][:10]
    names = [s['segment'][:15] for s in segments]
    scores = [s['mean_score'] for s in segments]
    overall = segment_data['overall_mean']
    
    colors = ['#22c55e' if s > overall else '#ef4444' for s in scores]
    bars = ax.barh(names[::-1], scores[::-1], color=colors[::-1], edgecolor='white', linewidth=2)
    ax.axvline(x=overall, color='blue', linestyle='--', linewidth=2, label=f'Overall: {overall:.2f}')
    
    ax.set_xlabel('Engagement Score')
    ax.set_title(f'Engagement by {segment_name}', fontsize=12, fontweight='bold')
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Significance annotation
    if segment_data['significant_difference']:
        ax.text(0.98, 0.02, f"ANOVA p={segment_data['p_value']:.4f} (Significant)", 
               transform=ax.transAxes, ha='right', fontsize=9, color='red')
    
    for bar, score in zip(bars, scores[::-1]):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
               f'{score:.2f}', va='center', fontsize=9)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_drivers_chart(drivers_data: Dict) -> str:
    """Create engagement drivers visualization"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    drivers = drivers_data['drivers'][:10]
    names = [d['dimension'][:20] for d in drivers]
    importance = [d['importance'] for d in drivers]
    directions = [d['direction'] for d in drivers]
    
    colors = ['#22c55e' if d == 'positive' else '#ef4444' for d in directions]
    bars = ax.barh(names[::-1], importance[::-1], color=colors[::-1], edgecolor='white', linewidth=2)
    
    ax.set_xlabel('Importance (Standardized Coefficient)')
    ax.set_title(f'Engagement Drivers (R² = {drivers_data["r_squared"]*100:.1f}%)', fontsize=12, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#22c55e', label='Positive Impact'),
                      Patch(facecolor='#ef4444', label='Negative Impact')]
    ax.legend(handles=legend_elements, loc='lower right')
    
    for bar, imp in zip(bars, importance[::-1]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
               f'{imp:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_insights(overall: Dict, enps: Optional[Dict], dimensions: Optional[Dict],
                      drivers: Optional[Dict], dept_data: Optional[Dict]) -> List[Dict[str, Any]]:
    """Generate key insights"""
    insights = []
    
    # Overall engagement
    norm_score = overall['normalized_score']
    if norm_score >= 75:
        insights.append({
            'title': f'Strong Engagement: {norm_score:.0f}%',
            'description': f'{overall["highly_engaged_pct"]:.0f}% highly engaged. Keep up the momentum!',
            'status': 'positive'
        })
    elif norm_score >= 60:
        insights.append({
            'title': f'Moderate Engagement: {norm_score:.0f}%',
            'description': f'{overall["disengaged_pct"]:.0f}% disengaged. Focus on improvement areas.',
            'status': 'neutral'
        })
    else:
        insights.append({
            'title': f'Low Engagement: {norm_score:.0f}%',
            'description': f'{overall["disengaged_pct"]:.0f}% disengaged. Urgent action needed.',
            'status': 'warning'
        })
    
    # eNPS
    if enps:
        enps_score = enps['enps_score']
        if enps_score >= 30:
            insights.append({
                'title': f'Excellent eNPS: {enps_score:.0f}',
                'description': f'{enps["promoters_pct"]:.0f}% promoters. Strong advocacy.',
                'status': 'positive'
            })
        elif enps_score < 0:
            insights.append({
                'title': f'Negative eNPS: {enps_score:.0f}',
                'description': f'{enps["detractors_pct"]:.0f}% detractors exceed promoters.',
                'status': 'warning'
            })
    
    # Dimensions
    if dimensions and dimensions.get('weaknesses'):
        insights.append({
            'title': f'Focus Area: {dimensions["weaknesses"][0]}',
            'description': f'Lowest scoring dimension. Prioritize improvement.',
            'status': 'warning'
        })
    
    if dimensions and dimensions.get('strengths'):
        insights.append({
            'title': f'Strength: {dimensions["strengths"][0]}',
            'description': f'Highest scoring dimension. Leverage this.',
            'status': 'positive'
        })
    
    # Top driver
    if drivers and drivers.get('drivers'):
        top_driver = drivers['drivers'][0]
        insights.append({
            'title': f'Key Driver: {top_driver["dimension"]}',
            'description': f'Most impactful factor ({top_driver["direction"]} effect).',
            'status': 'neutral'
        })
    
    # Department variance
    if dept_data and dept_data.get('significant_difference'):
        insights.append({
            'title': 'Significant Department Variance',
            'description': f'{dept_data["highest_segment"]} highest, {dept_data["lowest_segment"]} needs attention.',
            'status': 'warning'
        })
    
    return insights


@router.post("/engagement-survey")
async def run_engagement_analysis(request: EngagementRequest) -> Dict[str, Any]:
    """
    Perform Employee Engagement Survey Analysis.
    """
    try:
        df = pd.DataFrame(request.data)
        
        if len(df) < 10:
            raise HTTPException(status_code=400, detail="Need at least 10 responses")
        
        results = {}
        visualizations = {}
        
        # Overall engagement
        overall_data = None
        if request.overall_engagement_col and request.overall_engagement_col in df.columns:
            overall_data = calculate_overall_engagement(df, request.overall_engagement_col)
            results['overall_engagement'] = overall_data
        
        # eNPS
        enps_data = None
        if request.enps_col and request.enps_col in df.columns:
            enps_data = calculate_enps(df, request.enps_col)
            results['enps'] = enps_data
        
        # Create overview chart
        if overall_data or enps_data:
            if overall_data is None:
                # Calculate from eNPS if no overall
                overall_data = {
                    'normalized_score': enps_data['mean_score'] * 10,
                    'highly_engaged_count': enps_data['promoters_count'],
                    'highly_engaged_pct': enps_data['promoters_pct'],
                    'moderately_engaged_count': enps_data['passives_count'],
                    'moderately_engaged_pct': enps_data['passives_pct'],
                    'disengaged_count': enps_data['detractors_count'],
                    'disengaged_pct': enps_data['detractors_pct']
                }
                results['overall_engagement'] = overall_data
            visualizations['overview_chart'] = create_engagement_overview_chart(overall_data, enps_data)
        
        # Dimension analysis
        dimension_data = None
        if request.dimension_cols:
            dimension_data = analyze_dimensions(df, request.dimension_cols, request.overall_engagement_col)
            if dimension_data.get('dimensions'):
                results['dimensions'] = dimension_data
                visualizations['dimensions_chart'] = create_dimensions_chart(dimension_data)
        
        # Engagement drivers
        drivers_data = None
        if request.dimension_cols and request.overall_engagement_col:
            drivers_data = identify_engagement_drivers(df, request.dimension_cols, request.overall_engagement_col)
            if drivers_data and drivers_data.get('drivers'):
                results['drivers'] = drivers_data
                visualizations['drivers_chart'] = create_drivers_chart(drivers_data)
        
        # Department analysis
        dept_data = None
        if request.department_col and request.department_col in df.columns and request.overall_engagement_col:
            dept_data = analyze_by_segment(df, request.department_col, request.overall_engagement_col)
            results['by_department'] = dept_data
            visualizations['department_chart'] = create_segment_chart(dept_data, 'Department')
        
        # Job level analysis
        if request.job_level_col and request.job_level_col in df.columns and request.overall_engagement_col:
            level_data = analyze_by_segment(df, request.job_level_col, request.overall_engagement_col)
            results['by_job_level'] = level_data
        
        # Tenure analysis
        if request.tenure_col and request.tenure_col in df.columns and request.overall_engagement_col:
            tenure_data = analyze_by_tenure(df, request.tenure_col, request.overall_engagement_col)
            if 'error' not in tenure_data:
                results['by_tenure'] = tenure_data
        
        # Generate insights
        insights = generate_insights(overall_data or {}, enps_data, dimension_data, drivers_data, dept_data)
        
        # Summary
        summary = {
            'total_responses': len(df),
            'overall_score': overall_data['normalized_score'] if overall_data else None,
            'enps': enps_data['enps_score'] if enps_data else None,
            'highly_engaged_pct': overall_data['highly_engaged_pct'] if overall_data else None,
            'disengaged_pct': overall_data['disengaged_pct'] if overall_data else None,
            'top_strength': dimension_data['strengths'][0] if dimension_data and dimension_data.get('strengths') else None,
            'top_weakness': dimension_data['weaknesses'][0] if dimension_data and dimension_data.get('weaknesses') else None
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
        raise HTTPException(status_code=500, detail=f"Engagement analysis failed: {str(e)}")
