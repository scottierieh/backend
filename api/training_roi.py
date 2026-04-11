"""
Training ROI Analysis Router for FastAPI
Training Effectiveness, Cost-Benefit, Performance Impact, Skill Development
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
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class TrainingROIRequest(BaseModel):
    data: List[Dict[str, Any]]
    employee_id_col: Optional[str] = None
    training_name_col: Optional[str] = None
    training_type_col: Optional[str] = None
    training_cost_col: Optional[str] = None
    pre_score_col: Optional[str] = None  # Pre-training assessment
    post_score_col: Optional[str] = None  # Post-training assessment
    pre_performance_col: Optional[str] = None  # Performance before training
    post_performance_col: Optional[str] = None  # Performance after training
    completion_status_col: Optional[str] = None
    department_col: Optional[str] = None
    training_hours_col: Optional[str] = None
    satisfaction_col: Optional[str] = None  # Training satisfaction rating


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


def calculate_learning_effectiveness(df: pd.DataFrame, pre_col: str, post_col: str) -> Dict[str, Any]:
    """Calculate learning effectiveness from pre/post scores"""
    
    valid_df = df[[pre_col, post_col]].dropna()
    
    if len(valid_df) < 5:
        return {'error': 'Insufficient data'}
    
    pre_scores = valid_df[pre_col]
    post_scores = valid_df[post_col]
    
    # Score improvement
    improvement = post_scores - pre_scores
    improvement_pct = (improvement / pre_scores * 100).replace([np.inf, -np.inf], np.nan)
    
    # Statistical test (paired t-test)
    t_stat, p_value = stats.ttest_rel(pre_scores, post_scores)
    
    # Effect size (Cohen's d)
    diff = post_scores.mean() - pre_scores.mean()
    pooled_std = np.sqrt((pre_scores.std()**2 + post_scores.std()**2) / 2)
    cohens_d = diff / pooled_std if pooled_std > 0 else 0
    
    # Pass rate (assuming passing = score >= 70 or improvement > 0)
    improved_count = (improvement > 0).sum()
    pass_rate = improved_count / len(valid_df) * 100
    
    return {
        'participants': len(valid_df),
        'pre_score_mean': _to_native_type(pre_scores.mean()),
        'pre_score_std': _to_native_type(pre_scores.std()),
        'post_score_mean': _to_native_type(post_scores.mean()),
        'post_score_std': _to_native_type(post_scores.std()),
        'avg_improvement': _to_native_type(improvement.mean()),
        'avg_improvement_pct': _to_native_type(improvement_pct.mean()),
        'median_improvement': _to_native_type(improvement.median()),
        'improved_count': int(improved_count),
        'improved_pct': _to_native_type(pass_rate),
        't_statistic': _to_native_type(t_stat),
        'p_value': _to_native_type(p_value),
        'is_significant': p_value < 0.05,
        'cohens_d': _to_native_type(cohens_d),
        'effect_size': 'Large' if abs(cohens_d) >= 0.8 else ('Medium' if abs(cohens_d) >= 0.5 else 'Small')
    }


def calculate_performance_impact(df: pd.DataFrame, pre_col: str, post_col: str) -> Dict[str, Any]:
    """Calculate performance impact from pre/post performance ratings"""
    
    valid_df = df[[pre_col, post_col]].dropna()
    
    if len(valid_df) < 5:
        return {'error': 'Insufficient data'}
    
    pre_perf = valid_df[pre_col]
    post_perf = valid_df[post_col]
    
    improvement = post_perf - pre_perf
    
    # Statistical test
    t_stat, p_value = stats.ttest_rel(pre_perf, post_perf)
    
    # Categorize changes
    improved = (improvement > 0).sum()
    unchanged = (improvement == 0).sum()
    declined = (improvement < 0).sum()
    
    return {
        'participants': len(valid_df),
        'pre_performance_mean': _to_native_type(pre_perf.mean()),
        'post_performance_mean': _to_native_type(post_perf.mean()),
        'avg_improvement': _to_native_type(improvement.mean()),
        'improved_count': int(improved),
        'improved_pct': _to_native_type(improved / len(valid_df) * 100),
        'unchanged_count': int(unchanged),
        'declined_count': int(declined),
        't_statistic': _to_native_type(t_stat),
        'p_value': _to_native_type(p_value),
        'is_significant': p_value < 0.05
    }


def calculate_roi_metrics(df: pd.DataFrame, cost_col: str,
                          performance_improvement: Optional[float] = None,
                          avg_salary: float = 60000) -> Dict[str, Any]:
    """Calculate training ROI and cost metrics"""
    
    total_cost = df[cost_col].sum()
    avg_cost_per_person = df[cost_col].mean()
    participant_count = len(df)
    
    # Estimate benefit (simplified model)
    # Assume 1% performance improvement = 0.5% salary equivalent value
    if performance_improvement is not None and performance_improvement > 0:
        estimated_benefit_per_person = avg_salary * (performance_improvement / 100) * 0.5
        total_benefit = estimated_benefit_per_person * participant_count
        roi = ((total_benefit - total_cost) / total_cost * 100) if total_cost > 0 else 0
        benefit_cost_ratio = total_benefit / total_cost if total_cost > 0 else 0
    else:
        estimated_benefit_per_person = None
        total_benefit = None
        roi = None
        benefit_cost_ratio = None
    
    return {
        'total_cost': _to_native_type(total_cost),
        'avg_cost_per_person': _to_native_type(avg_cost_per_person),
        'participant_count': participant_count,
        'total_benefit': _to_native_type(total_benefit),
        'estimated_benefit_per_person': _to_native_type(estimated_benefit_per_person),
        'roi_pct': _to_native_type(roi),
        'benefit_cost_ratio': _to_native_type(benefit_cost_ratio),
        'cost_per_hour': _to_native_type(total_cost / df.get('training_hours', pd.Series([1])).sum()) if 'training_hours' in df.columns else None
    }


def analyze_by_training_type(df: pd.DataFrame, type_col: str,
                              pre_col: Optional[str] = None,
                              post_col: Optional[str] = None,
                              cost_col: Optional[str] = None) -> List[Dict[str, Any]]:
    """Analyze effectiveness by training type"""
    
    results = []
    
    for training_type in df[type_col].unique():
        type_df = df[df[type_col] == training_type]
        
        result = {
            'training_type': str(training_type),
            'participant_count': len(type_df)
        }
        
        # Learning improvement
        if pre_col and post_col and pre_col in df.columns and post_col in df.columns:
            valid = type_df[[pre_col, post_col]].dropna()
            if len(valid) >= 3:
                improvement = (valid[post_col] - valid[pre_col]).mean()
                result['avg_improvement'] = _to_native_type(improvement)
                result['pre_score'] = _to_native_type(valid[pre_col].mean())
                result['post_score'] = _to_native_type(valid[post_col].mean())
        
        # Cost
        if cost_col and cost_col in df.columns:
            result['total_cost'] = _to_native_type(type_df[cost_col].sum())
            result['avg_cost'] = _to_native_type(type_df[cost_col].mean())
        
        results.append(result)
    
    # Sort by improvement if available
    if results and 'avg_improvement' in results[0]:
        results.sort(key=lambda x: x.get('avg_improvement', 0) or 0, reverse=True)
    
    return results


def analyze_by_department(df: pd.DataFrame, dept_col: str,
                          pre_col: Optional[str] = None,
                          post_col: Optional[str] = None) -> List[Dict[str, Any]]:
    """Analyze training effectiveness by department"""
    
    results = []
    
    for dept in df[dept_col].unique():
        dept_df = df[df[dept_col] == dept]
        
        result = {
            'department': str(dept),
            'participant_count': len(dept_df)
        }
        
        if pre_col and post_col and pre_col in df.columns and post_col in df.columns:
            valid = dept_df[[pre_col, post_col]].dropna()
            if len(valid) >= 3:
                improvement = (valid[post_col] - valid[pre_col]).mean()
                result['avg_improvement'] = _to_native_type(improvement)
                result['improvement_pct'] = _to_native_type(improvement / valid[pre_col].mean() * 100) if valid[pre_col].mean() > 0 else 0
        
        results.append(result)
    
    results.sort(key=lambda x: x.get('avg_improvement', 0) or 0, reverse=True)
    
    return results


def analyze_satisfaction(df: pd.DataFrame, satisfaction_col: str) -> Dict[str, Any]:
    """Analyze training satisfaction scores"""
    
    scores = df[satisfaction_col].dropna()
    
    if len(scores) < 5:
        return {'error': 'Insufficient data'}
    
    # Assume 1-5 or 1-10 scale
    max_score = scores.max()
    if max_score <= 5:
        scale = 5
    else:
        scale = 10
    
    # Categorize
    satisfied = (scores >= scale * 0.7).sum()  # 70%+ is satisfied
    neutral = ((scores >= scale * 0.4) & (scores < scale * 0.7)).sum()
    dissatisfied = (scores < scale * 0.4).sum()
    
    return {
        'mean_score': _to_native_type(scores.mean()),
        'median_score': _to_native_type(scores.median()),
        'std_score': _to_native_type(scores.std()),
        'min_score': _to_native_type(scores.min()),
        'max_score': _to_native_type(scores.max()),
        'scale': scale,
        'normalized_score': _to_native_type(scores.mean() / scale * 100),
        'satisfied_count': int(satisfied),
        'satisfied_pct': _to_native_type(satisfied / len(scores) * 100),
        'neutral_count': int(neutral),
        'dissatisfied_count': int(dissatisfied),
        'nps_estimate': _to_native_type((satisfied - dissatisfied) / len(scores) * 100)
    }


def analyze_completion(df: pd.DataFrame, completion_col: str) -> Dict[str, Any]:
    """Analyze training completion rates"""
    
    # Normalize completion status
    df['_completed'] = df[completion_col].astype(str).str.lower().isin(
        ['completed', 'complete', 'yes', '1', 'true', 'passed', 'pass']
    )
    
    completed = df['_completed'].sum()
    total = len(df)
    
    return {
        'total_enrolled': total,
        'completed_count': int(completed),
        'completion_rate': _to_native_type(completed / total * 100),
        'incomplete_count': int(total - completed),
        'dropout_rate': _to_native_type((total - completed) / total * 100)
    }


def create_learning_chart(learning_data: Dict) -> str:
    """Create learning effectiveness visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Pre vs Post scores
    categories = ['Pre-Training', 'Post-Training']
    scores = [learning_data['pre_score_mean'], learning_data['post_score_mean']]
    errors = [learning_data['pre_score_std'], learning_data['post_score_std']]
    
    colors = ['#ef4444', '#22c55e']
    bars = ax1.bar(categories, scores, yerr=errors, color=colors, edgecolor='white', 
                   linewidth=2, capsize=5)
    
    ax1.set_ylabel('Score')
    ax1.set_title('Pre vs Post Training Scores', fontsize=12, fontweight='bold')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    for bar, score in zip(bars, scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(errors) + 1,
                f'{score:.1f}', ha='center', fontsize=11, fontweight='bold')
    
    # Improvement annotation
    improvement = learning_data['avg_improvement']
    ax1.annotate(f'+{improvement:.1f} ({learning_data["avg_improvement_pct"]:.1f}%)',
                xy=(1, scores[1]), xytext=(1.3, scores[1] - 5),
                fontsize=12, fontweight='bold', color='green',
                arrowprops=dict(arrowstyle='->', color='green'))
    
    # Effect size gauge
    effect_sizes = ['Small\n(<0.5)', 'Medium\n(0.5-0.8)', 'Large\n(>0.8)']
    cohens_d = abs(learning_data['cohens_d'])
    
    colors2 = ['#fbbf24', '#3b82f6', '#22c55e']
    current_idx = 2 if cohens_d >= 0.8 else (1 if cohens_d >= 0.5 else 0)
    bar_colors = ['#e5e7eb'] * 3
    bar_colors[current_idx] = colors2[current_idx]
    
    ax2.barh(effect_sizes, [0.5, 0.8, 1.0], color='#e5e7eb', edgecolor='white', linewidth=2)
    ax2.barh(effect_sizes[current_idx], [0.5, 0.3, 0.2][current_idx], 
             color=colors2[current_idx], edgecolor='white', linewidth=2)
    ax2.axvline(x=cohens_d, color='red', linestyle='--', linewidth=2, label=f"Cohen's d: {cohens_d:.2f}")
    
    ax2.set_xlabel("Effect Size")
    ax2.set_title(f'Effect Size: {learning_data["effect_size"]}', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Significance annotation
    sig_text = "✓ Statistically Significant" if learning_data['is_significant'] else "✗ Not Significant"
    sig_color = 'green' if learning_data['is_significant'] else 'red'
    ax2.text(0.5, -0.15, f"p-value: {learning_data['p_value']:.4f} ({sig_text})",
            transform=ax2.transAxes, ha='center', fontsize=10, color=sig_color)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_roi_chart(roi_data: Dict, by_type_data: Optional[List] = None) -> str:
    """Create ROI visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Cost vs Benefit
    if roi_data['total_benefit'] is not None:
        categories = ['Total Cost', 'Total Benefit']
        values = [roi_data['total_cost'], roi_data['total_benefit']]
        colors = ['#ef4444', '#22c55e']
        
        bars = ax1.bar(categories, values, color=colors, edgecolor='white', linewidth=2)
        ax1.set_ylabel('Amount ($)')
        ax1.set_title(f'Cost vs Benefit (ROI: {roi_data["roi_pct"]:.1f}%)', fontsize=12, fontweight='bold')
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values) * 0.02,
                    f'${val:,.0f}', ha='center', fontsize=10, fontweight='bold')
    else:
        ax1.text(0.5, 0.5, 'ROI calculation requires\nperformance improvement data',
                ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        ax1.set_title('Cost vs Benefit', fontsize=12, fontweight='bold')
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # By training type
    if by_type_data and len(by_type_data) > 0:
        types = [t['training_type'][:12] for t in by_type_data[:6]]
        improvements = [t.get('avg_improvement', 0) or 0 for t in by_type_data[:6]]
        
        colors2 = ['#22c55e' if i > 0 else '#ef4444' for i in improvements]
        bars2 = ax2.barh(types[::-1], improvements[::-1], color=colors2[::-1], 
                        edgecolor='white', linewidth=2)
        ax2.axvline(x=0, color='black', linewidth=1)
        ax2.set_xlabel('Score Improvement')
        ax2.set_title('Improvement by Training Type', fontsize=12, fontweight='bold')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        for bar, imp in zip(bars2, improvements[::-1]):
            ax2.text(bar.get_width() + 0.5 if imp >= 0 else bar.get_width() - 0.5,
                    bar.get_y() + bar.get_height()/2,
                    f'{imp:.1f}', va='center', fontsize=9,
                    ha='left' if imp >= 0 else 'right')
    else:
        ax2.text(0.5, 0.5, 'No training type data', ha='center', va='center', transform=ax2.transAxes)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_satisfaction_chart(satisfaction_data: Dict) -> str:
    """Create satisfaction visualization"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Satisfaction distribution
    labels = ['Satisfied', 'Neutral', 'Dissatisfied']
    sizes = [satisfaction_data['satisfied_count'], satisfaction_data['neutral_count'], 
             satisfaction_data['dissatisfied_count']]
    colors = ['#22c55e', '#fbbf24', '#ef4444']
    explode = (0.05, 0, 0)
    
    wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
                                        autopct='%1.1f%%', startangle=90)
    ax1.set_title(f'Satisfaction Distribution (Avg: {satisfaction_data["mean_score"]:.1f}/{satisfaction_data["scale"]})',
                 fontsize=12, fontweight='bold')
    
    # Normalized score gauge
    norm_score = satisfaction_data['normalized_score']
    
    # Create gauge-like visualization
    theta = np.linspace(0, np.pi, 100)
    r = 1
    
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-0.2, 1.2)
    ax2.set_aspect('equal')
    
    # Background arc
    for i, (start, end, color) in enumerate([(0, 40, '#ef4444'), (40, 70, '#fbbf24'), (70, 100, '#22c55e')]):
        theta_start = np.pi * (1 - end/100)
        theta_end = np.pi * (1 - start/100)
        theta_segment = np.linspace(theta_start, theta_end, 50)
        ax2.fill_between(np.cos(theta_segment), np.sin(theta_segment), 0, alpha=0.3, color=color)
    
    # Needle
    needle_angle = np.pi * (1 - norm_score/100)
    ax2.arrow(0, 0, 0.8*np.cos(needle_angle), 0.8*np.sin(needle_angle),
             head_width=0.08, head_length=0.05, fc='black', ec='black')
    ax2.add_patch(plt.Circle((0, 0), 0.08, color='black'))
    
    ax2.text(0, -0.1, f'{norm_score:.0f}%', ha='center', fontsize=16, fontweight='bold')
    ax2.text(-1, 0, '0', fontsize=10)
    ax2.text(1, 0, '100', fontsize=10)
    ax2.set_title('Satisfaction Score', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_insights(learning: Optional[Dict], performance: Optional[Dict],
                      roi: Optional[Dict], satisfaction: Optional[Dict],
                      completion: Optional[Dict]) -> List[Dict[str, Any]]:
    """Generate key insights"""
    insights = []
    
    # Learning effectiveness
    if learning and 'error' not in learning:
        if learning['is_significant']:
            insights.append({
                'title': f'Significant Learning Gain: +{learning["avg_improvement"]:.1f} points',
                'description': f'{learning["effect_size"]} effect size (Cohen\'s d: {learning["cohens_d"]:.2f}). {learning["improved_pct"]:.0f}% improved.',
                'status': 'positive'
            })
        else:
            insights.append({
                'title': f'Learning Improvement: +{learning["avg_improvement"]:.1f} points',
                'description': 'Improvement not statistically significant. Consider larger sample or different approach.',
                'status': 'warning'
            })
    
    # ROI
    if roi and roi.get('roi_pct') is not None:
        if roi['roi_pct'] > 100:
            insights.append({
                'title': f'Strong ROI: {roi["roi_pct"]:.0f}%',
                'description': f'Benefit-cost ratio of {roi["benefit_cost_ratio"]:.1f}x. Training investment is paying off.',
                'status': 'positive'
            })
        elif roi['roi_pct'] > 0:
            insights.append({
                'title': f'Positive ROI: {roi["roi_pct"]:.0f}%',
                'description': 'Training generates positive returns but could be optimized.',
                'status': 'neutral'
            })
        else:
            insights.append({
                'title': f'Negative ROI: {roi["roi_pct"]:.0f}%',
                'description': 'Training cost exceeds estimated benefits. Review program effectiveness.',
                'status': 'warning'
            })
    
    # Satisfaction
    if satisfaction and 'error' not in satisfaction:
        norm_score = satisfaction['normalized_score']
        if norm_score >= 80:
            insights.append({
                'title': f'High Satisfaction: {norm_score:.0f}%',
                'description': f'{satisfaction["satisfied_pct"]:.0f}% of participants satisfied.',
                'status': 'positive'
            })
        elif norm_score < 60:
            insights.append({
                'title': f'Low Satisfaction: {norm_score:.0f}%',
                'description': 'Consider improving training content or delivery.',
                'status': 'warning'
            })
    
    # Completion
    if completion:
        if completion['completion_rate'] >= 90:
            insights.append({
                'title': f'Excellent Completion: {completion["completion_rate"]:.0f}%',
                'description': f'{completion["completed_count"]} of {completion["total_enrolled"]} completed.',
                'status': 'positive'
            })
        elif completion['completion_rate'] < 70:
            insights.append({
                'title': f'Low Completion: {completion["completion_rate"]:.0f}%',
                'description': f'{completion["incomplete_count"]} dropouts. Investigate barriers.',
                'status': 'warning'
            })
    
    return insights


@router.post("/training-roi")
async def run_training_roi_analysis(request: TrainingROIRequest) -> Dict[str, Any]:
    """
    Perform Training ROI Analysis.
    """
    try:
        df = pd.DataFrame(request.data)
        
        if len(df) < 5:
            raise HTTPException(status_code=400, detail="Need at least 5 records")
        
        results = {}
        visualizations = {}
        
        # Learning effectiveness
        learning_data = None
        if request.pre_score_col and request.post_score_col:
            if request.pre_score_col in df.columns and request.post_score_col in df.columns:
                learning_data = calculate_learning_effectiveness(
                    df, request.pre_score_col, request.post_score_col
                )
                if 'error' not in learning_data:
                    results['learning_effectiveness'] = learning_data
                    visualizations['learning_chart'] = create_learning_chart(learning_data)
        
        # Performance impact
        if request.pre_performance_col and request.post_performance_col:
            if request.pre_performance_col in df.columns and request.post_performance_col in df.columns:
                performance_data = calculate_performance_impact(
                    df, request.pre_performance_col, request.post_performance_col
                )
                if 'error' not in performance_data:
                    results['performance_impact'] = performance_data
        
        # ROI metrics
        roi_data = None
        if request.training_cost_col and request.training_cost_col in df.columns:
            perf_improvement = None
            if 'performance_impact' in results:
                perf_improvement = results['performance_impact'].get('avg_improvement', 0) * 10  # Scale to percentage
            elif learning_data and 'error' not in learning_data:
                perf_improvement = learning_data.get('avg_improvement_pct', 0)
            
            roi_data = calculate_roi_metrics(df, request.training_cost_col, perf_improvement)
            results['roi_metrics'] = roi_data
        
        # By training type
        by_type_data = None
        if request.training_type_col and request.training_type_col in df.columns:
            by_type_data = analyze_by_training_type(
                df, request.training_type_col,
                request.pre_score_col, request.post_score_col,
                request.training_cost_col
            )
            results['by_training_type'] = by_type_data
        
        # Create ROI chart
        if roi_data or by_type_data:
            visualizations['roi_chart'] = create_roi_chart(roi_data or {}, by_type_data)
        
        # By department
        if request.department_col and request.department_col in df.columns:
            dept_data = analyze_by_department(
                df, request.department_col,
                request.pre_score_col, request.post_score_col
            )
            results['by_department'] = dept_data
        
        # Satisfaction
        satisfaction_data = None
        if request.satisfaction_col and request.satisfaction_col in df.columns:
            satisfaction_data = analyze_satisfaction(df, request.satisfaction_col)
            if 'error' not in satisfaction_data:
                results['satisfaction'] = satisfaction_data
                visualizations['satisfaction_chart'] = create_satisfaction_chart(satisfaction_data)
        
        # Completion
        completion_data = None
        if request.completion_status_col and request.completion_status_col in df.columns:
            completion_data = analyze_completion(df, request.completion_status_col)
            results['completion'] = completion_data
        
        # Generate insights
        insights = generate_insights(
            learning_data, results.get('performance_impact'),
            roi_data, satisfaction_data, completion_data
        )
        
        # Summary
        summary = {
            'total_participants': len(df),
            'avg_improvement': learning_data['avg_improvement'] if learning_data and 'error' not in learning_data else None,
            'improvement_pct': learning_data['avg_improvement_pct'] if learning_data and 'error' not in learning_data else None,
            'roi_pct': roi_data['roi_pct'] if roi_data else None,
            'satisfaction_score': satisfaction_data['normalized_score'] if satisfaction_data and 'error' not in satisfaction_data else None,
            'completion_rate': completion_data['completion_rate'] if completion_data else None
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
        raise HTTPException(status_code=500, detail=f"Training ROI analysis failed: {str(e)}")
