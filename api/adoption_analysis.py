"""
Feature Adoption Analysis Router for FastAPI
Statistical analysis for feature adoption metrics
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Literal
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import time
import warnings
from scipy import stats
from datetime import datetime

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class AdoptionRequest(BaseModel):
    data: List[Dict[str, Any]]
    user_col: Optional[str] = None
    feature_col: str
    adopted_col: str
    date_col: Optional[str] = None
    segment_col: Optional[str] = None
    analysis_type: Literal["adoption_rate", "time_to_adopt", "funnel"] = "adoption_rate"
    target_rate: float = 0.5
    time_period: Optional[str] = None


def _to_native_type(obj):
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
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return image_base64


def calculate_adoption_metrics(df: pd.DataFrame, feature_col: str, adopted_col: str) -> Dict:
    """Calculate overall adoption metrics"""
    features = df[feature_col].unique()
    metrics = {}
    
    for feature in features:
        feature_df = df[df[feature_col] == feature]
        total_users = len(feature_df)
        adopted_users = feature_df[adopted_col].sum()
        adoption_rate = adopted_users / total_users if total_users > 0 else 0
        
        # Wilson score interval for confidence
        if total_users > 0:
            z = 1.96  # 95% confidence
            p = adoption_rate
            n = total_users
            denominator = 1 + z**2 / n
            center = (p + z**2 / (2*n)) / denominator
            margin = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denominator
            ci_lower = max(0, center - margin)
            ci_upper = min(1, center + margin)
        else:
            ci_lower, ci_upper = 0, 0
        
        metrics[str(feature)] = {
            'total_users': int(total_users),
            'adopted_users': int(adopted_users),
            'adoption_rate': float(adoption_rate),
            'adoption_pct': float(adoption_rate * 100),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper)
        }
    
    return metrics


def calculate_segment_adoption(df: pd.DataFrame, feature_col: str, adopted_col: str, 
                                segment_col: str) -> List[Dict]:
    """Calculate adoption by segment"""
    results = []
    
    for segment in df[segment_col].unique():
        segment_df = df[df[segment_col] == segment]
        
        for feature in df[feature_col].unique():
            feature_df = segment_df[segment_df[feature_col] == feature]
            total = len(feature_df)
            adopted = feature_df[adopted_col].sum() if total > 0 else 0
            rate = adopted / total if total > 0 else 0
            
            results.append({
                'segment': str(segment),
                'feature': str(feature),
                'total_users': int(total),
                'adopted_users': int(adopted),
                'adoption_rate': float(rate),
                'adoption_pct': float(rate * 100)
            })
    
    return sorted(results, key=lambda x: x['adoption_rate'], reverse=True)


def perform_adoption_tests(df: pd.DataFrame, feature_col: str, adopted_col: str, 
                           target_rate: float) -> Dict:
    """Perform statistical tests for adoption"""
    results = {}
    features = df[feature_col].unique()
    
    for feature in features:
        feature_df = df[df[feature_col] == feature]
        n = len(feature_df)
        x = feature_df[adopted_col].sum()
        p_observed = x / n if n > 0 else 0
        
        # One-sample proportion z-test against target
        if n > 0:
            se = np.sqrt(target_rate * (1 - target_rate) / n)
            z_stat = (p_observed - target_rate) / se if se > 0 else 0
            p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
            
            results[str(feature)] = {
                'observed_rate': float(p_observed),
                'target_rate': float(target_rate),
                'difference': float(p_observed - target_rate),
                'z_statistic': float(z_stat),
                'p_value': float(p_value),
                'meets_target': p_observed >= target_rate,
                'significant': p_value < 0.05
            }
    
    return results


def compare_features(df: pd.DataFrame, feature_col: str, adopted_col: str) -> List[Dict]:
    """Compare adoption rates between features using chi-square test"""
    features = df[feature_col].unique()
    comparisons = []
    
    for i, f1 in enumerate(features):
        for f2 in features[i+1:]:
            df1 = df[df[feature_col] == f1]
            df2 = df[df[feature_col] == f2]
            
            n1, x1 = len(df1), df1[adopted_col].sum()
            n2, x2 = len(df2), df2[adopted_col].sum()
            
            if n1 > 0 and n2 > 0:
                # Create contingency table
                table = [[x1, n1-x1], [x2, n2-x2]]
                chi2, p_value, dof, expected = stats.chi2_contingency(table)
                
                rate1, rate2 = x1/n1, x2/n2
                
                comparisons.append({
                    'feature_1': str(f1),
                    'feature_2': str(f2),
                    'rate_1': float(rate1),
                    'rate_2': float(rate2),
                    'difference': float(rate1 - rate2),
                    'chi2_statistic': float(chi2),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05
                })
    
    return sorted(comparisons, key=lambda x: abs(x['difference']), reverse=True)


def calculate_adoption_funnel(df: pd.DataFrame, feature_col: str, stages: List[str]) -> Dict:
    """Calculate adoption funnel metrics"""
    funnel = {}
    
    for feature in df[feature_col].unique():
        feature_df = df[df[feature_col] == feature]
        total = len(feature_df)
        
        stage_data = []
        prev_count = total
        
        for stage in stages:
            if stage in feature_df.columns:
                count = feature_df[stage].sum()
                rate = count / total if total > 0 else 0
                conversion = count / prev_count if prev_count > 0 else 0
                
                stage_data.append({
                    'stage': stage,
                    'count': int(count),
                    'rate': float(rate),
                    'conversion': float(conversion),
                    'dropoff': float(1 - conversion)
                })
                prev_count = count
        
        funnel[str(feature)] = stage_data
    
    return funnel


def create_adoption_rate_chart(metrics: Dict) -> str:
    """Create adoption rate comparison chart"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    features = list(metrics.keys())
    rates = [metrics[f]['adoption_pct'] for f in features]
    ci_lower = [metrics[f]['ci_lower'] * 100 for f in features]
    ci_upper = [metrics[f]['ci_upper'] * 100 for f in features]
    
    colors = ['#3b82f6' if r >= 50 else '#94a3b8' for r in rates]
    
    y_pos = range(len(features))
    bars = ax.barh(y_pos, rates, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
    
    # Error bars for confidence intervals
    for i, (r, ci_l, ci_u) in enumerate(zip(rates, ci_lower, ci_upper)):
        ax.plot([ci_l, ci_u], [i, i], color='#1f2937', linewidth=2, marker='|', markersize=8)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.set_xlabel('Adoption Rate (%)', fontsize=11)
    ax.set_title('Feature Adoption Rates', fontsize=14, fontweight='bold')
    ax.axvline(50, color='#ef4444', linestyle='--', linewidth=2, alpha=0.7, label='50% Target')
    
    # Add value labels
    for bar, rate in zip(bars, rates):
        ax.annotate(f'{rate:.1f}%',
                    xy=(rate, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0),
                    textcoords="offset points",
                    ha='left', va='center', fontsize=10, fontweight='bold')
    
    ax.legend(loc='lower right')
    ax.set_xlim(0, max(rates) * 1.2)
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_segment_heatmap(segment_data: List[Dict]) -> str:
    """Create segment vs feature heatmap"""
    if not segment_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No segment data available', ha='center', va='center', fontsize=14)
        ax.axis('off')
        return _fig_to_base64(fig)
    
    # Pivot data for heatmap
    df = pd.DataFrame(segment_data)
    pivot = df.pivot(index='segment', columns='feature', values='adoption_pct')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    im = ax.imshow(pivot.values, cmap='Blues', aspect='auto')
    
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha='right')
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    
    # Add text annotations
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            color = 'white' if val > 50 else 'black'
            ax.text(j, i, f'{val:.1f}%', ha='center', va='center', color=color, fontsize=9)
    
    ax.set_title('Adoption Rate by Segment and Feature', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Adoption Rate (%)')
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_comparison_chart(comparisons: List[Dict]) -> str:
    """Create feature comparison chart"""
    if not comparisons:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No comparison data available', ha='center', va='center', fontsize=14)
        ax.axis('off')
        return _fig_to_base64(fig)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    labels = [f"{c['feature_1']} vs {c['feature_2']}" for c in comparisons[:10]]
    differences = [c['difference'] * 100 for c in comparisons[:10]]
    significances = [c['significant'] for c in comparisons[:10]]
    
    colors = ['#3b82f6' if sig else '#94a3b8' for sig in significances]
    
    y_pos = range(len(labels))
    bars = ax.barh(y_pos, differences, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel('Adoption Rate Difference (%)', fontsize=11)
    ax.set_title('Feature Adoption Comparison', fontsize=14, fontweight='bold')
    ax.axvline(0, color='#1f2937', linewidth=1)
    
    for bar, diff, sig in zip(bars, differences, significances):
        label = f'{diff:+.1f}%' + (' *' if sig else '')
        ax.annotate(label,
                    xy=(diff, bar.get_y() + bar.get_height() / 2),
                    xytext=(5 if diff >= 0 else -5, 0),
                    textcoords="offset points",
                    ha='left' if diff >= 0 else 'right',
                    va='center', fontsize=9)
    
    ax.legend([plt.Rectangle((0,0),1,1, fc='#3b82f6', alpha=0.8),
               plt.Rectangle((0,0),1,1, fc='#94a3b8', alpha=0.8)],
              ['Significant (p<0.05)', 'Not Significant'], loc='lower right')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_adoption_distribution(df: pd.DataFrame, feature_col: str, adopted_col: str) -> str:
    """Create adoption distribution chart"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    features = df[feature_col].unique()
    
    # Pie chart for overall adoption
    ax1 = axes[0]
    total_adopted = df[adopted_col].sum()
    total_not_adopted = len(df) - total_adopted
    
    ax1.pie([total_adopted, total_not_adopted], 
            labels=['Adopted', 'Not Adopted'],
            colors=['#3b82f6', '#e5e7eb'],
            autopct='%1.1f%%',
            startangle=90,
            explode=(0.05, 0))
    ax1.set_title('Overall Adoption', fontsize=12, fontweight='bold')
    
    # Stacked bar for by feature
    ax2 = axes[1]
    adopted_counts = []
    not_adopted_counts = []
    
    for feature in features:
        feature_df = df[df[feature_col] == feature]
        adopted_counts.append(feature_df[adopted_col].sum())
        not_adopted_counts.append(len(feature_df) - feature_df[adopted_col].sum())
    
    x = range(len(features))
    ax2.bar(x, adopted_counts, label='Adopted', color='#3b82f6', alpha=0.8)
    ax2.bar(x, not_adopted_counts, bottom=adopted_counts, label='Not Adopted', color='#e5e7eb', alpha=0.8)
    
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(f)[:15] for f in features], rotation=45, ha='right')
    ax2.set_ylabel('Users', fontsize=11)
    ax2.set_title('Adoption by Feature', fontsize=12, fontweight='bold')
    ax2.legend()
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_target_comparison_chart(test_results: Dict, target_rate: float) -> str:
    """Create chart comparing features against target"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    features = list(test_results.keys())
    observed = [test_results[f]['observed_rate'] * 100 for f in features]
    meets_target = [test_results[f]['meets_target'] for f in features]
    
    colors = ['#22c55e' if m else '#ef4444' for m in meets_target]
    
    x = range(len(features))
    bars = ax.bar(x, observed, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
    
    ax.axhline(target_rate * 100, color='#3b82f6', linestyle='--', linewidth=2, 
               label=f'Target: {target_rate*100:.0f}%')
    
    ax.set_xticks(x)
    ax.set_xticklabels([str(f)[:15] for f in features], rotation=45, ha='right')
    ax.set_ylabel('Adoption Rate (%)', fontsize=11)
    ax.set_title('Feature Performance vs Target', fontsize=14, fontweight='bold')
    
    for bar, rate, meets in zip(bars, observed, meets_target):
        symbol = '✓' if meets else '✗'
        ax.annotate(f'{rate:.1f}% {symbol}',
                    xy=(bar.get_x() + bar.get_width() / 2, rate),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.legend(loc='upper right')
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_key_insights(metrics: Dict, test_results: Dict, comparisons: List[Dict],
                          segment_data: List[Dict], target_rate: float) -> List[Dict]:
    """Generate key insights from adoption analysis"""
    insights = []
    
    # Overall adoption insight
    total_users = sum(m['total_users'] for m in metrics.values())
    total_adopted = sum(m['adopted_users'] for m in metrics.values())
    overall_rate = total_adopted / total_users if total_users > 0 else 0
    
    if overall_rate >= target_rate:
        insights.append({
            'title': f'Overall Adoption Exceeds Target ({overall_rate*100:.1f}%)',
            'description': f'Overall adoption rate of {overall_rate*100:.1f}% exceeds the {target_rate*100:.0f}% target.',
            'status': 'positive'
        })
    else:
        gap = (target_rate - overall_rate) * 100
        insights.append({
            'title': f'Adoption Below Target ({overall_rate*100:.1f}%)',
            'description': f'Overall adoption is {gap:.1f}pp below the {target_rate*100:.0f}% target.',
            'status': 'warning'
        })
    
    # Top performing feature
    if metrics:
        top_feature = max(metrics.items(), key=lambda x: x[1]['adoption_rate'])
        insights.append({
            'title': f'Top Feature: {top_feature[0]}',
            'description': f'Highest adoption at {top_feature[1]["adoption_pct"]:.1f}% with {top_feature[1]["adopted_users"]} users.',
            'status': 'positive'
        })
    
    # Underperforming feature
    if metrics:
        bottom_feature = min(metrics.items(), key=lambda x: x[1]['adoption_rate'])
        if bottom_feature[1]['adoption_rate'] < target_rate:
            insights.append({
                'title': f'Needs Attention: {bottom_feature[0]}',
                'description': f'Lowest adoption at {bottom_feature[1]["adoption_pct"]:.1f}%. Consider UX improvements or promotion.',
                'status': 'warning'
            })
    
    # Significant differences
    sig_comparisons = [c for c in comparisons if c['significant']]
    if sig_comparisons:
        top_diff = sig_comparisons[0]
        insights.append({
            'title': f'Significant Gap: {top_diff["feature_1"]} vs {top_diff["feature_2"]}',
            'description': f'{abs(top_diff["difference"]*100):.1f}pp difference is statistically significant (p={top_diff["p_value"]:.4f}).',
            'status': 'neutral'
        })
    
    # Segment insight
    if segment_data:
        top_segment = max(segment_data, key=lambda x: x['adoption_rate'])
        insights.append({
            'title': f'Top Segment: {top_segment["segment"]} - {top_segment["feature"]}',
            'description': f'{top_segment["adoption_pct"]:.1f}% adoption rate in this segment-feature combination.',
            'status': 'positive'
        })
    
    return insights


@router.post("/adoption")
async def run_adoption_analysis(request: AdoptionRequest) -> Dict[str, Any]:
    """Run feature adoption analysis"""
    try:
        start_time = time.time()
        df = pd.DataFrame(request.data)
        
        # Validate columns
        if request.feature_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Feature column '{request.feature_col}' not found")
        if request.adopted_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Adopted column '{request.adopted_col}' not found")
        
        # Convert adopted column to numeric
        df[request.adopted_col] = pd.to_numeric(df[request.adopted_col], errors='coerce').fillna(0).astype(int)
        
        # Calculate metrics
        metrics = calculate_adoption_metrics(df, request.feature_col, request.adopted_col)
        
        # Perform statistical tests
        test_results = perform_adoption_tests(df, request.feature_col, request.adopted_col, request.target_rate)
        
        # Compare features
        comparisons = compare_features(df, request.feature_col, request.adopted_col)
        
        # Segment analysis
        segment_data = []
        if request.segment_col and request.segment_col in df.columns:
            segment_data = calculate_segment_adoption(df, request.feature_col, request.adopted_col, request.segment_col)
        
        # Create visualizations
        visualizations = {
            'adoption_rates': create_adoption_rate_chart(metrics),
            'adoption_distribution': create_adoption_distribution(df, request.feature_col, request.adopted_col),
            'target_comparison': create_target_comparison_chart(test_results, request.target_rate),
            'feature_comparison': create_comparison_chart(comparisons)
        }
        
        if segment_data:
            visualizations['segment_heatmap'] = create_segment_heatmap(segment_data)
        
        # Generate insights
        key_insights = generate_key_insights(metrics, test_results, comparisons, segment_data, request.target_rate)
        
        analyze_time_ms = int((time.time() - start_time) * 1000)
        
        # Calculate totals
        total_users = sum(m['total_users'] for m in metrics.values())
        total_adopted = sum(m['adopted_users'] for m in metrics.values())
        overall_rate = total_adopted / total_users if total_users > 0 else 0
        
        # Prepare results
        results = {
            'metrics': {k: {kk: _to_native_type(vv) for kk, vv in v.items()} for k, v in metrics.items()},
            'test_results': {k: {kk: _to_native_type(vv) for kk, vv in v.items()} for k, v in test_results.items()},
            'comparisons': [{k: _to_native_type(v) for k, v in c.items()} for c in comparisons],
            'segment_data': [{k: _to_native_type(v) for k, v in s.items()} for s in segment_data],
            'totals': {
                'total_users': total_users,
                'total_adopted': total_adopted,
                'overall_rate': overall_rate,
                'overall_pct': overall_rate * 100,
                'num_features': len(metrics)
            }
        }
        
        summary = {
            'analysis_type': request.analysis_type,
            'feature_column': request.feature_col,
            'num_features': len(df[request.feature_col].unique()),
            'total_users': total_users,
            'overall_adoption_rate': overall_rate * 100,
            'target_rate': request.target_rate * 100,
            'meets_target': overall_rate >= request.target_rate,
            'analyze_time_ms': analyze_time_ms
        }
        
        return {
            'success': True,
            'results': results,
            'visualizations': visualizations,
            'key_insights': key_insights,
            'summary': summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Adoption analysis failed: {str(e)}")
