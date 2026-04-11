"""
Employee Engagement Survey Analysis Router for FastAPI
Statistical analysis for employee survey data
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

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class SurveyRequest(BaseModel):
    data: List[Dict[str, Any]]
    respondent_col: Optional[str] = None
    question_cols: List[str]
    department_col: Optional[str] = None
    tenure_col: Optional[str] = None
    manager_col: Optional[str] = None
    analysis_type: Literal["overall", "trend", "benchmark"] = "overall"
    benchmark_score: float = 3.5
    scale_max: int = 5


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


def calculate_question_statistics(df: pd.DataFrame, question_cols: List[str], 
                                   scale_max: int) -> List[Dict]:
    """Calculate statistics for each survey question"""
    results = []
    
    for col in question_cols:
        values = pd.to_numeric(df[col], errors='coerce').dropna()
        
        if len(values) >= 3:
            favorable = (values >= (scale_max * 0.8)).sum()
            neutral = ((values >= (scale_max * 0.4)) & (values < (scale_max * 0.8))).sum()
            unfavorable = (values < (scale_max * 0.4)).sum()
            
            results.append({
                'question': col,
                'mean': float(np.mean(values)),
                'median': float(np.median(values)),
                'std': float(np.std(values, ddof=1)) if len(values) > 1 else 0,
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'count': int(len(values)),
                'favorable': int(favorable),
                'neutral': int(neutral),
                'unfavorable': int(unfavorable),
                'favorable_pct': float(favorable / len(values) * 100),
                'score_pct': float(np.mean(values) / scale_max * 100)
            })
    
    return sorted(results, key=lambda x: x['mean'], reverse=True)


def calculate_engagement_index(df: pd.DataFrame, question_cols: List[str], 
                                scale_max: int) -> Dict:
    """Calculate overall engagement index"""
    all_scores = []
    
    for col in question_cols:
        values = pd.to_numeric(df[col], errors='coerce').dropna()
        all_scores.extend(values.tolist())
    
    if not all_scores:
        return {'index': 0, 'category': 'Unknown'}
    
    avg_score = np.mean(all_scores)
    index_pct = (avg_score / scale_max) * 100
    
    if index_pct >= 80:
        category = "Highly Engaged"
    elif index_pct >= 60:
        category = "Engaged"
    elif index_pct >= 40:
        category = "Neutral"
    elif index_pct >= 20:
        category = "Disengaged"
    else:
        category = "Highly Disengaged"
    
    return {
        'index': float(index_pct),
        'avg_score': float(avg_score),
        'category': category,
        'total_responses': len(all_scores),
        'questions_analyzed': len(question_cols)
    }


def analyze_by_department(df: pd.DataFrame, question_cols: List[str], 
                          department_col: str, scale_max: int) -> List[Dict]:
    """Analyze engagement by department"""
    results = []
    
    for dept in df[department_col].unique():
        dept_df = df[df[department_col] == dept]
        all_scores = []
        
        for col in question_cols:
            values = pd.to_numeric(dept_df[col], errors='coerce').dropna()
            all_scores.extend(values.tolist())
        
        if all_scores:
            avg = np.mean(all_scores)
            results.append({
                'department': str(dept),
                'avg_score': float(avg),
                'index_pct': float((avg / scale_max) * 100),
                'respondents': len(dept_df),
                'responses': len(all_scores)
            })
    
    return sorted(results, key=lambda x: x['avg_score'], reverse=True)


def analyze_by_tenure(df: pd.DataFrame, question_cols: List[str], 
                      tenure_col: str, scale_max: int) -> List[Dict]:
    """Analyze engagement by tenure"""
    results = []
    
    # Create tenure buckets
    df['tenure_bucket'] = pd.cut(
        pd.to_numeric(df[tenure_col], errors='coerce'),
        bins=[0, 1, 3, 5, 10, 100],
        labels=['<1 year', '1-3 years', '3-5 years', '5-10 years', '10+ years']
    )
    
    for bucket in df['tenure_bucket'].dropna().unique():
        bucket_df = df[df['tenure_bucket'] == bucket]
        all_scores = []
        
        for col in question_cols:
            values = pd.to_numeric(bucket_df[col], errors='coerce').dropna()
            all_scores.extend(values.tolist())
        
        if all_scores:
            avg = np.mean(all_scores)
            results.append({
                'tenure': str(bucket),
                'avg_score': float(avg),
                'index_pct': float((avg / scale_max) * 100),
                'respondents': len(bucket_df),
                'responses': len(all_scores)
            })
    
    return results


def perform_statistical_tests(df: pd.DataFrame, question_cols: List[str],
                              benchmark: float, scale_max: int) -> Dict:
    """Perform statistical tests against benchmark"""
    results = {}
    
    for col in question_cols:
        values = pd.to_numeric(df[col], errors='coerce').dropna().values
        
        if len(values) >= 3:
            # One-sample t-test against benchmark
            t_stat, p_value = stats.ttest_1samp(values, benchmark)
            
            # Effect size (Cohen's d)
            std = np.std(values, ddof=1)
            cohens_d = (np.mean(values) - benchmark) / std if std > 0 else 0
            
            results[col] = {
                'mean': float(np.mean(values)),
                'benchmark': float(benchmark),
                'difference': float(np.mean(values) - benchmark),
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'cohens_d': float(cohens_d),
                'significant': p_value < 0.05,
                'direction': 'above' if np.mean(values) > benchmark else 'below'
            }
    
    return results


def identify_strengths_weaknesses(question_stats: List[Dict], 
                                   benchmark: float) -> Dict:
    """Identify top strengths and areas for improvement"""
    sorted_by_score = sorted(question_stats, key=lambda x: x['mean'], reverse=True)
    
    strengths = [q for q in sorted_by_score if q['mean'] >= benchmark][:5]
    weaknesses = [q for q in sorted_by_score if q['mean'] < benchmark][-5:]
    weaknesses.reverse()
    
    return {
        'strengths': strengths,
        'weaknesses': weaknesses
    }


def create_engagement_overview_chart(question_stats: List[Dict], 
                                      benchmark: float, scale_max: int) -> str:
    """Create engagement overview chart"""
    fig, ax = plt.subplots(figsize=(12, max(6, len(question_stats) * 0.4)))
    
    questions = [q['question'][:40] for q in question_stats]
    scores = [q['mean'] for q in question_stats]
    
    colors = ['#22c55e' if s >= benchmark else '#f59e0b' if s >= benchmark * 0.8 else '#ef4444' 
              for s in scores]
    
    y_pos = range(len(questions))
    bars = ax.barh(y_pos, scores, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(questions)
    ax.set_xlabel('Average Score', fontsize=11)
    ax.set_title('Survey Question Scores', fontsize=14, fontweight='bold')
    ax.axvline(benchmark, color='#3b82f6', linestyle='--', linewidth=2, 
               label=f'Benchmark ({benchmark:.1f})')
    ax.set_xlim(0, scale_max)
    
    for bar, score in zip(bars, scores):
        ax.annotate(f'{score:.2f}',
                    xy=(score, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=9, fontweight='bold')
    
    ax.legend(loc='lower right')
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_department_comparison_chart(dept_data: List[Dict], benchmark: float) -> str:
    """Create department comparison chart"""
    if not dept_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No department data available', ha='center', va='center', fontsize=14)
        ax.axis('off')
        return _fig_to_base64(fig)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    depts = [d['department'][:20] for d in dept_data]
    scores = [d['index_pct'] for d in dept_data]
    
    colors = ['#22c55e' if s >= 70 else '#f59e0b' if s >= 50 else '#ef4444' for s in scores]
    
    x = range(len(depts))
    bars = ax.bar(x, scores, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(depts, rotation=45, ha='right')
    ax.set_ylabel('Engagement Index (%)', fontsize=11)
    ax.set_title('Engagement by Department', fontsize=14, fontweight='bold')
    ax.axhline(70, color='#22c55e', linestyle='--', linewidth=1.5, alpha=0.7, label='High (70%)')
    ax.axhline(50, color='#f59e0b', linestyle='--', linewidth=1.5, alpha=0.7, label='Moderate (50%)')
    ax.set_ylim(0, 100)
    
    for bar, score in zip(bars, scores):
        ax.annotate(f'{score:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, score),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.legend(loc='upper right')
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_favorable_distribution_chart(question_stats: List[Dict]) -> str:
    """Create favorable/neutral/unfavorable distribution chart"""
    if not question_stats:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=14)
        ax.axis('off')
        return _fig_to_base64(fig)
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(question_stats) * 0.4)))
    
    questions = [q['question'][:35] for q in question_stats]
    favorable = [q['favorable'] for q in question_stats]
    neutral = [q['neutral'] for q in question_stats]
    unfavorable = [q['unfavorable'] for q in question_stats]
    
    y = range(len(questions))
    
    ax.barh(y, favorable, label='Favorable', color='#22c55e', alpha=0.8)
    ax.barh(y, neutral, left=favorable, label='Neutral', color='#f59e0b', alpha=0.8)
    ax.barh(y, unfavorable, left=[f+n for f, n in zip(favorable, neutral)], 
            label='Unfavorable', color='#ef4444', alpha=0.8)
    
    ax.set_yticks(y)
    ax.set_yticklabels(questions)
    ax.set_xlabel('Number of Responses', fontsize=11)
    ax.set_title('Response Distribution by Question', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_engagement_gauge_chart(engagement_index: Dict) -> str:
    """Create engagement index gauge chart"""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    index = engagement_index['index']
    category = engagement_index['category']
    
    # Create gauge
    theta = np.linspace(0, np.pi, 100)
    r = 1
    
    # Background arcs
    colors_bg = ['#ef4444', '#f59e0b', '#eab308', '#84cc16', '#22c55e']
    for i, color in enumerate(colors_bg):
        start = i * np.pi / 5
        end = (i + 1) * np.pi / 5
        theta_seg = np.linspace(start, end, 20)
        ax.fill_between(theta_seg, 0.6, 1, color=color, alpha=0.3)
    
    # Needle
    needle_angle = np.pi * (1 - index / 100)
    ax.annotate('', xy=(needle_angle, 0.9), xytext=(np.pi/2, 0),
                arrowprops=dict(arrowstyle='->', color='#1f2937', lw=3))
    
    # Center text
    ax.text(np.pi/2, 0.3, f'{index:.1f}%', ha='center', va='center', 
            fontsize=28, fontweight='bold', color='#1f2937')
    ax.text(np.pi/2, 0.1, category, ha='center', va='center', 
            fontsize=14, color='#6b7280')
    
    ax.set_xlim(0, np.pi)
    ax.set_ylim(0, 1.2)
    ax.axis('off')
    ax.set_title('Engagement Index', fontsize=14, fontweight='bold', y=1.05)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_benchmark_comparison_chart(test_results: Dict, benchmark: float) -> str:
    """Create benchmark comparison chart"""
    if not test_results:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No test results available', ha='center', va='center', fontsize=14)
        ax.axis('off')
        return _fig_to_base64(fig)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    questions = list(test_results.keys())[:10]
    diffs = [test_results[q]['difference'] for q in questions]
    significant = [test_results[q]['significant'] for q in questions]
    
    colors = ['#22c55e' if d > 0 else '#ef4444' for d in diffs]
    edge_colors = ['#1f2937' if sig else 'white' for sig in significant]
    
    x = range(len(questions))
    bars = ax.bar(x, diffs, color=colors, alpha=0.8, edgecolor=edge_colors, linewidth=2)
    
    ax.set_xticks(x)
    ax.set_xticklabels([q[:20] for q in questions], rotation=45, ha='right')
    ax.set_ylabel('Difference from Benchmark', fontsize=11)
    ax.set_title(f'Score vs Benchmark ({benchmark:.1f})', fontsize=14, fontweight='bold')
    ax.axhline(0, color='#1f2937', linewidth=1)
    
    ax.legend([plt.Rectangle((0,0),1,1, fc='#22c55e', ec='#1f2937', lw=2),
               plt.Rectangle((0,0),1,1, fc='#ef4444', ec='white', lw=2)],
              ['Above (Significant)', 'Below'], loc='upper right')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_key_insights(engagement_index: Dict, question_stats: List[Dict],
                          dept_data: List[Dict], test_results: Dict,
                          benchmark: float) -> List[Dict]:
    """Generate key insights from survey analysis"""
    insights = []
    
    # Overall engagement insight
    if engagement_index['index'] >= 70:
        insights.append({
            'title': f"Strong Engagement ({engagement_index['index']:.1f}%)",
            'description': f"Overall engagement is {engagement_index['category']}. Employees show positive sentiment.",
            'status': 'positive'
        })
    elif engagement_index['index'] >= 50:
        insights.append({
            'title': f"Moderate Engagement ({engagement_index['index']:.1f}%)",
            'description': f"Engagement is moderate. There's room for improvement in key areas.",
            'status': 'neutral'
        })
    else:
        insights.append({
            'title': f"Low Engagement ({engagement_index['index']:.1f}%)",
            'description': f"Engagement is below optimal. Immediate attention needed.",
            'status': 'warning'
        })
    
    # Top strength
    if question_stats:
        top = question_stats[0]
        insights.append({
            'title': f"Top Strength: {top['question'][:30]}",
            'description': f"Score: {top['mean']:.2f} ({top['favorable_pct']:.1f}% favorable)",
            'status': 'positive'
        })
    
    # Area for improvement
    if question_stats:
        bottom = question_stats[-1]
        if bottom['mean'] < benchmark:
            insights.append({
                'title': f"Needs Attention: {bottom['question'][:30]}",
                'description': f"Score: {bottom['mean']:.2f} ({bottom['unfavorable']:.0f} unfavorable responses)",
                'status': 'warning'
            })
    
    # Department insight
    if dept_data:
        top_dept = dept_data[0]
        bottom_dept = dept_data[-1]
        gap = top_dept['index_pct'] - bottom_dept['index_pct']
        
        if gap > 20:
            insights.append({
                'title': f"Department Gap: {gap:.1f}pp",
                'description': f"{top_dept['department']} ({top_dept['index_pct']:.1f}%) vs {bottom_dept['department']} ({bottom_dept['index_pct']:.1f}%)",
                'status': 'warning' if gap > 30 else 'neutral'
            })
    
    # Significant differences from benchmark
    sig_above = sum(1 for t in test_results.values() if t.get('significant') and t.get('direction') == 'above')
    sig_below = sum(1 for t in test_results.values() if t.get('significant') and t.get('direction') == 'below')
    
    if sig_above > sig_below:
        insights.append({
            'title': f"{sig_above} Questions Above Benchmark",
            'description': f"Statistically significant positive performance in {sig_above} areas.",
            'status': 'positive'
        })
    elif sig_below > 0:
        insights.append({
            'title': f"{sig_below} Questions Below Benchmark",
            'description': f"{sig_below} questions show statistically significant underperformance.",
            'status': 'warning'
        })
    
    return insights


@router.post("/survey")
async def run_survey_analysis(request: SurveyRequest) -> Dict[str, Any]:
    """Run employee engagement survey analysis"""
    try:
        start_time = time.time()
        df = pd.DataFrame(request.data)
        
        # Validate columns
        missing_cols = [col for col in request.question_cols if col not in df.columns]
        if missing_cols:
            raise HTTPException(status_code=400, detail=f"Question columns not found: {missing_cols}")
        
        if len(df) < 5:
            raise HTTPException(status_code=400, detail="Need at least 5 survey responses")
        
        # Calculate question statistics
        question_stats = calculate_question_statistics(df, request.question_cols, request.scale_max)
        
        # Calculate engagement index
        engagement_index = calculate_engagement_index(df, request.question_cols, request.scale_max)
        
        # Department analysis
        dept_data = []
        if request.department_col and request.department_col in df.columns:
            dept_data = analyze_by_department(df, request.question_cols, 
                                              request.department_col, request.scale_max)
        
        # Tenure analysis
        tenure_data = []
        if request.tenure_col and request.tenure_col in df.columns:
            tenure_data = analyze_by_tenure(df, request.question_cols,
                                            request.tenure_col, request.scale_max)
        
        # Statistical tests
        test_results = perform_statistical_tests(df, request.question_cols,
                                                  request.benchmark_score, request.scale_max)
        
        # Strengths and weaknesses
        sw_analysis = identify_strengths_weaknesses(question_stats, request.benchmark_score)
        
        # Create visualizations
        visualizations = {
            'engagement_overview': create_engagement_overview_chart(question_stats, 
                                                                     request.benchmark_score,
                                                                     request.scale_max),
            'favorable_distribution': create_favorable_distribution_chart(question_stats),
            'engagement_gauge': create_engagement_gauge_chart(engagement_index),
            'benchmark_comparison': create_benchmark_comparison_chart(test_results,
                                                                       request.benchmark_score)
        }
        
        if dept_data:
            visualizations['department_comparison'] = create_department_comparison_chart(
                dept_data, request.benchmark_score)
        
        # Generate insights
        key_insights = generate_key_insights(engagement_index, question_stats,
                                              dept_data, test_results, request.benchmark_score)
        
        analyze_time_ms = int((time.time() - start_time) * 1000)
        
        # Prepare results
        results = {
            'engagement_index': {k: _to_native_type(v) for k, v in engagement_index.items()},
            'question_stats': [{k: _to_native_type(v) for k, v in q.items()} for q in question_stats],
            'department_data': [{k: _to_native_type(v) for k, v in d.items()} for d in dept_data],
            'tenure_data': [{k: _to_native_type(v) for k, v in t.items()} for t in tenure_data],
            'test_results': {k: {kk: _to_native_type(vv) for kk, vv in v.items()} 
                           for k, v in test_results.items()},
            'strengths_weaknesses': sw_analysis,
            'response_count': len(df)
        }
        
        summary = {
            'analysis_type': request.analysis_type,
            'response_count': len(df),
            'questions_analyzed': len(request.question_cols),
            'engagement_index': engagement_index['index'],
            'engagement_category': engagement_index['category'],
            'benchmark_score': request.benchmark_score,
            'scale_max': request.scale_max,
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
        raise HTTPException(status_code=500, detail=f"Survey analysis failed: {str(e)}")
