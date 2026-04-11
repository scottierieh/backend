"""
Diversity & Inclusion Analysis Router for FastAPI
Statistical analysis for workforce diversity metrics
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


class DiversityRequest(BaseModel):
    data: List[Dict[str, Any]]
    employee_col: Optional[str] = None
    gender_col: Optional[str] = None
    ethnicity_col: Optional[str] = None
    age_col: Optional[str] = None
    department_col: Optional[str] = None
    level_col: Optional[str] = None
    tenure_col: Optional[str] = None
    hire_date_col: Optional[str] = None
    analysis_type: Literal["representation", "parity", "trend"] = "representation"
    benchmark_data: Optional[Dict[str, float]] = None


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


def calculate_diversity_index(counts: Dict[str, int]) -> float:
    """Calculate Simpson's Diversity Index"""
    total = sum(counts.values())
    if total <= 1:
        return 0
    
    sum_n = sum(n * (n - 1) for n in counts.values())
    index = 1 - (sum_n / (total * (total - 1)))
    return float(index)


def calculate_representation(df: pd.DataFrame, col: str) -> Dict:
    """Calculate representation metrics for a demographic column"""
    if col not in df.columns:
        return {}
    
    counts = df[col].value_counts().to_dict()
    total = len(df)
    
    representation = {}
    for group, count in counts.items():
        representation[str(group)] = {
            'count': int(count),
            'percentage': float(count / total * 100),
            'proportion': float(count / total)
        }
    
    diversity_index = calculate_diversity_index(counts)
    
    return {
        'groups': representation,
        'total': total,
        'num_groups': len(counts),
        'diversity_index': diversity_index,
        'diversity_rating': 'High' if diversity_index >= 0.7 else 'Moderate' if diversity_index >= 0.4 else 'Low'
    }


def analyze_by_department(df: pd.DataFrame, demo_col: str, dept_col: str) -> List[Dict]:
    """Analyze demographic distribution by department"""
    results = []
    
    for dept in df[dept_col].unique():
        dept_df = df[df[dept_col] == dept]
        rep = calculate_representation(dept_df, demo_col)
        
        if rep:
            results.append({
                'department': str(dept),
                'total': rep['total'],
                'diversity_index': rep['diversity_index'],
                'groups': rep['groups']
            })
    
    return sorted(results, key=lambda x: x['diversity_index'], reverse=True)


def analyze_by_level(df: pd.DataFrame, demo_col: str, level_col: str) -> List[Dict]:
    """Analyze demographic distribution by job level"""
    results = []
    
    for level in df[level_col].unique():
        level_df = df[df[level_col] == level]
        rep = calculate_representation(level_df, demo_col)
        
        if rep:
            results.append({
                'level': str(level),
                'total': rep['total'],
                'diversity_index': rep['diversity_index'],
                'groups': rep['groups']
            })
    
    return results


def calculate_parity_gaps(df: pd.DataFrame, demo_col: str, level_col: str) -> Dict:
    """Calculate representation parity across job levels"""
    overall_rep = calculate_representation(df, demo_col)
    if not overall_rep:
        return {}
    
    level_analysis = analyze_by_level(df, demo_col, level_col)
    
    parity_gaps = {}
    for group in overall_rep['groups'].keys():
        overall_pct = overall_rep['groups'][group]['percentage']
        
        level_gaps = []
        for level_data in level_analysis:
            level_pct = level_data['groups'].get(group, {}).get('percentage', 0)
            gap = level_pct - overall_pct
            
            level_gaps.append({
                'level': level_data['level'],
                'percentage': level_pct,
                'gap': gap,
                'gap_direction': 'over' if gap > 0 else 'under' if gap < 0 else 'parity'
            })
        
        parity_gaps[group] = {
            'overall_percentage': overall_pct,
            'level_breakdown': level_gaps,
            'max_gap': max(abs(l['gap']) for l in level_gaps) if level_gaps else 0
        }
    
    return parity_gaps


def perform_chi_square_test(df: pd.DataFrame, demo_col: str, 
                            group_col: str) -> Dict:
    """Perform chi-square test for independence"""
    contingency = pd.crosstab(df[demo_col], df[group_col])
    
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return {'error': 'Insufficient categories for chi-square test'}
    
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    
    # Cramér's V for effect size
    n = contingency.sum().sum()
    min_dim = min(contingency.shape[0] - 1, contingency.shape[1] - 1)
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
    
    return {
        'chi2_statistic': float(chi2),
        'p_value': float(p_value),
        'degrees_of_freedom': int(dof),
        'cramers_v': float(cramers_v),
        'significant': p_value < 0.05,
        'effect_size': 'Large' if cramers_v >= 0.5 else 'Medium' if cramers_v >= 0.3 else 'Small'
    }


def analyze_age_distribution(df: pd.DataFrame, age_col: str) -> Dict:
    """Analyze age/generation distribution"""
    ages = pd.to_numeric(df[age_col], errors='coerce').dropna()
    
    if len(ages) == 0:
        return {}
    
    # Generation buckets
    def get_generation(age):
        if age >= 59:
            return 'Baby Boomers (59+)'
        elif age >= 43:
            return 'Gen X (43-58)'
        elif age >= 27:
            return 'Millennials (27-42)'
        else:
            return 'Gen Z (<27)'
    
    generations = ages.apply(get_generation)
    gen_counts = generations.value_counts().to_dict()
    total = len(ages)
    
    return {
        'statistics': {
            'mean': float(np.mean(ages)),
            'median': float(np.median(ages)),
            'std': float(np.std(ages, ddof=1)),
            'min': float(np.min(ages)),
            'max': float(np.max(ages))
        },
        'generations': {
            gen: {
                'count': int(count),
                'percentage': float(count / total * 100)
            }
            for gen, count in gen_counts.items()
        },
        'diversity_index': calculate_diversity_index(gen_counts)
    }


def create_representation_chart(representation: Dict, title: str) -> str:
    """Create representation pie/bar chart"""
    if not representation.get('groups'):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', fontsize=14)
        ax.axis('off')
        return _fig_to_base64(fig)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    groups = list(representation['groups'].keys())
    counts = [representation['groups'][g]['count'] for g in groups]
    percentages = [representation['groups'][g]['percentage'] for g in groups]
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(groups)))
    
    # Pie chart
    ax1 = axes[0]
    wedges, texts, autotexts = ax1.pie(counts, labels=groups, colors=colors,
                                        autopct='%1.1f%%', startangle=90)
    ax1.set_title(f'{title} Distribution', fontsize=12, fontweight='bold')
    
    # Bar chart
    ax2 = axes[1]
    bars = ax2.barh(groups, percentages, color=colors, alpha=0.8, edgecolor='white')
    ax2.set_xlabel('Percentage (%)', fontsize=11)
    ax2.set_title(f'{title} Breakdown', fontsize=12, fontweight='bold')
    
    for bar, pct in zip(bars, percentages):
        ax2.annotate(f'{pct:.1f}%',
                    xy=(pct, bar.get_y() + bar.get_height() / 2),
                    xytext=(5, 0), textcoords="offset points",
                    ha='left', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_department_diversity_chart(dept_data: List[Dict]) -> str:
    """Create department diversity comparison chart"""
    if not dept_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No department data available', ha='center', va='center', fontsize=14)
        ax.axis('off')
        return _fig_to_base64(fig)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    depts = [d['department'][:20] for d in dept_data]
    indices = [d['diversity_index'] * 100 for d in dept_data]
    
    colors = ['#22c55e' if i >= 70 else '#f59e0b' if i >= 40 else '#ef4444' for i in indices]
    
    x = range(len(depts))
    bars = ax.bar(x, indices, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(depts, rotation=45, ha='right')
    ax.set_ylabel('Diversity Index (%)', fontsize=11)
    ax.set_title('Diversity Index by Department', fontsize=14, fontweight='bold')
    ax.axhline(70, color='#22c55e', linestyle='--', linewidth=1.5, alpha=0.7, label='High (70%)')
    ax.axhline(40, color='#f59e0b', linestyle='--', linewidth=1.5, alpha=0.7, label='Moderate (40%)')
    ax.set_ylim(0, 100)
    
    for bar, idx in zip(bars, indices):
        ax.annotate(f'{idx:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, idx),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.legend(loc='upper right')
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_parity_heatmap(parity_gaps: Dict, levels: List[str]) -> str:
    """Create parity gap heatmap"""
    if not parity_gaps:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No parity data available', ha='center', va='center', fontsize=14)
        ax.axis('off')
        return _fig_to_base64(fig)
    
    groups = list(parity_gaps.keys())
    
    # Build matrix
    matrix = []
    for group in groups:
        row = []
        for level_data in parity_gaps[group]['level_breakdown']:
            row.append(level_data['gap'])
        matrix.append(row)
    
    if not matrix or not matrix[0]:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Insufficient data for heatmap', ha='center', va='center', fontsize=14)
        ax.axis('off')
        return _fig_to_base64(fig)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    matrix = np.array(matrix)
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=-20, vmax=20)
    
    level_labels = [parity_gaps[groups[0]]['level_breakdown'][i]['level'] 
                    for i in range(len(parity_gaps[groups[0]]['level_breakdown']))]
    
    ax.set_xticks(range(len(level_labels)))
    ax.set_xticklabels(level_labels, rotation=45, ha='right')
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups)
    
    # Add text annotations
    for i in range(len(groups)):
        for j in range(len(level_labels)):
            val = matrix[i, j]
            color = 'white' if abs(val) > 10 else 'black'
            ax.text(j, i, f'{val:+.1f}', ha='center', va='center', color=color, fontsize=9)
    
    ax.set_title('Representation Gap by Level (pp vs Overall)', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Gap (percentage points)')
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_level_pipeline_chart(level_data: List[Dict], demo_col: str) -> str:
    """Create pipeline/funnel chart by level"""
    if not level_data:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No level data available', ha='center', va='center', fontsize=14)
        ax.axis('off')
        return _fig_to_base64(fig)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    levels = [d['level'] for d in level_data]
    
    # Get all groups
    all_groups = set()
    for d in level_data:
        all_groups.update(d['groups'].keys())
    all_groups = sorted(list(all_groups))
    
    x = np.arange(len(levels))
    width = 0.8 / len(all_groups)
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(all_groups)))
    
    for i, group in enumerate(all_groups):
        percentages = [d['groups'].get(group, {}).get('percentage', 0) for d in level_data]
        offset = (i - len(all_groups)/2 + 0.5) * width
        ax.bar(x + offset, percentages, width, label=group, color=colors[i], alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_ylabel('Percentage (%)', fontsize=11)
    ax.set_title('Representation Across Job Levels', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_age_distribution_chart(age_data: Dict) -> str:
    """Create age/generation distribution chart"""
    if not age_data or not age_data.get('generations'):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No age data available', ha='center', va='center', fontsize=14)
        ax.axis('off')
        return _fig_to_base64(fig)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    generations = age_data['generations']
    gen_names = list(generations.keys())
    gen_counts = [generations[g]['count'] for g in gen_names]
    gen_pcts = [generations[g]['percentage'] for g in gen_names]
    
    colors = ['#8b5cf6', '#3b82f6', '#22c55e', '#f59e0b'][:len(gen_names)]
    
    # Pie chart
    ax1 = axes[0]
    ax1.pie(gen_counts, labels=gen_names, colors=colors, autopct='%1.1f%%', startangle=90)
    ax1.set_title('Generation Distribution', fontsize=12, fontweight='bold')
    
    # Bar chart with stats
    ax2 = axes[1]
    bars = ax2.bar(gen_names, gen_pcts, color=colors, alpha=0.8)
    ax2.set_ylabel('Percentage (%)', fontsize=11)
    ax2.set_title(f"Age: Mean={age_data['statistics']['mean']:.1f}, Median={age_data['statistics']['median']:.1f}", 
                  fontsize=12, fontweight='bold')
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, pct in zip(bars, gen_pcts):
        ax2.annotate(f'{pct:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, pct),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_key_insights(overall_rep: Dict, dept_data: List[Dict],
                          parity_gaps: Dict, chi_square: Dict,
                          demo_type: str) -> List[Dict]:
    """Generate key insights from D&I analysis"""
    insights = []
    
    # Overall diversity insight
    if overall_rep:
        di = overall_rep.get('diversity_index', 0)
        rating = overall_rep.get('diversity_rating', 'Unknown')
        
        if di >= 0.7:
            insights.append({
                'title': f'Strong {demo_type} Diversity ({di*100:.1f}%)',
                'description': f'Diversity index of {di*100:.1f}% indicates {rating.lower()} representation across groups.',
                'status': 'positive'
            })
        elif di >= 0.4:
            insights.append({
                'title': f'Moderate {demo_type} Diversity ({di*100:.1f}%)',
                'description': f'Room for improvement in {demo_type.lower()} diversity.',
                'status': 'neutral'
            })
        else:
            insights.append({
                'title': f'Low {demo_type} Diversity ({di*100:.1f}%)',
                'description': f'Significant opportunity to improve {demo_type.lower()} representation.',
                'status': 'warning'
            })
    
    # Department gap insight
    if dept_data and len(dept_data) >= 2:
        top = dept_data[0]
        bottom = dept_data[-1]
        gap = top['diversity_index'] - bottom['diversity_index']
        
        if gap > 0.3:
            insights.append({
                'title': f"Department Gap: {gap*100:.1f}pp",
                'description': f"{top['department']} ({top['diversity_index']*100:.1f}%) vs {bottom['department']} ({bottom['diversity_index']*100:.1f}%)",
                'status': 'warning'
            })
    
    # Parity gap insight
    if parity_gaps:
        max_gap_group = max(parity_gaps.items(), key=lambda x: x[1]['max_gap'])
        if max_gap_group[1]['max_gap'] > 10:
            insights.append({
                'title': f"Parity Gap: {max_gap_group[0]}",
                'description': f"Up to {max_gap_group[1]['max_gap']:.1f}pp variation across levels.",
                'status': 'warning'
            })
    
    # Statistical significance
    if chi_square and chi_square.get('significant'):
        insights.append({
            'title': 'Significant Distribution Difference',
            'description': f"Chi-square test shows significant relationship (p={chi_square['p_value']:.4f}, Cramér's V={chi_square['cramers_v']:.3f}).",
            'status': 'neutral'
        })
    
    return insights


@router.post("/diversity")
async def run_diversity_analysis(request: DiversityRequest) -> Dict[str, Any]:
    """Run diversity and inclusion analysis"""
    try:
        start_time = time.time()
        df = pd.DataFrame(request.data)
        
        if len(df) < 10:
            raise HTTPException(status_code=400, detail="Need at least 10 employee records")
        
        results = {}
        visualizations = {}
        
        # Gender analysis
        if request.gender_col and request.gender_col in df.columns:
            gender_rep = calculate_representation(df, request.gender_col)
            results['gender'] = gender_rep
            visualizations['gender_distribution'] = create_representation_chart(gender_rep, 'Gender')
            
            if request.department_col and request.department_col in df.columns:
                results['gender_by_department'] = analyze_by_department(df, request.gender_col, request.department_col)
            
            if request.level_col and request.level_col in df.columns:
                results['gender_by_level'] = analyze_by_level(df, request.gender_col, request.level_col)
                results['gender_parity'] = calculate_parity_gaps(df, request.gender_col, request.level_col)
                visualizations['gender_pipeline'] = create_level_pipeline_chart(results['gender_by_level'], 'Gender')
        
        # Ethnicity analysis
        if request.ethnicity_col and request.ethnicity_col in df.columns:
            ethnicity_rep = calculate_representation(df, request.ethnicity_col)
            results['ethnicity'] = ethnicity_rep
            visualizations['ethnicity_distribution'] = create_representation_chart(ethnicity_rep, 'Ethnicity')
            
            if request.department_col and request.department_col in df.columns:
                results['ethnicity_by_department'] = analyze_by_department(df, request.ethnicity_col, request.department_col)
                visualizations['department_diversity'] = create_department_diversity_chart(results['ethnicity_by_department'])
            
            if request.level_col and request.level_col in df.columns:
                results['ethnicity_by_level'] = analyze_by_level(df, request.ethnicity_col, request.level_col)
                results['ethnicity_parity'] = calculate_parity_gaps(df, request.ethnicity_col, request.level_col)
                
                if results['ethnicity_parity']:
                    visualizations['parity_heatmap'] = create_parity_heatmap(
                        results['ethnicity_parity'],
                        [d['level'] for d in results['ethnicity_by_level']]
                    )
        
        # Age analysis
        if request.age_col and request.age_col in df.columns:
            age_data = analyze_age_distribution(df, request.age_col)
            results['age'] = age_data
            visualizations['age_distribution'] = create_age_distribution_chart(age_data)
        
        # Chi-square tests
        chi_square_results = {}
        primary_demo = request.gender_col or request.ethnicity_col
        
        if primary_demo and request.department_col:
            chi_square_results['demo_x_department'] = perform_chi_square_test(
                df, primary_demo, request.department_col)
        
        if primary_demo and request.level_col:
            chi_square_results['demo_x_level'] = perform_chi_square_test(
                df, primary_demo, request.level_col)
        
        results['statistical_tests'] = chi_square_results
        
        # Generate insights
        primary_rep = results.get('gender') or results.get('ethnicity') or {}
        dept_diversity = results.get('gender_by_department') or results.get('ethnicity_by_department') or []
        parity = results.get('gender_parity') or results.get('ethnicity_parity') or {}
        chi_test = chi_square_results.get('demo_x_level') or chi_square_results.get('demo_x_department') or {}
        
        key_insights = generate_key_insights(
            primary_rep, dept_diversity, parity, chi_test,
            'Gender' if request.gender_col else 'Ethnicity'
        )
        
        analyze_time_ms = int((time.time() - start_time) * 1000)
        
        # Summary
        summary = {
            'analysis_type': request.analysis_type,
            'employee_count': len(df),
            'gender_analyzed': request.gender_col is not None,
            'ethnicity_analyzed': request.ethnicity_col is not None,
            'age_analyzed': request.age_col is not None,
            'overall_diversity_index': primary_rep.get('diversity_index', 0),
            'diversity_rating': primary_rep.get('diversity_rating', 'Unknown'),
            'analyze_time_ms': analyze_time_ms
        }
        
        return {
            'success': True,
            'results': {k: _to_native_type(v) if not isinstance(v, (dict, list)) else v 
                       for k, v in results.items()},
            'visualizations': visualizations,
            'key_insights': key_insights,
            'summary': summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diversity analysis failed: {str(e)}")
