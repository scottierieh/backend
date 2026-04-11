"""
Yield & Defect Analysis Router for FastAPI
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Literal
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

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class YieldDefectRequest(BaseModel):
    data: List[Dict[str, Any]]
    unit_col: Optional[str] = None
    defective_col: Optional[str] = None
    defect_type_col: Optional[str] = None
    defect_count_col: Optional[str] = None
    process_step_col: Optional[str] = None
    opportunities_col: Optional[str] = None
    opportunities_per_unit: int = 10
    analysis_type: Literal["basic", "pareto", "process", "dpmo"] = "pareto"


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


COLORS = {
    'defect': '#ef4444',
    'yield': '#22c55e',
    'cumulative': '#f59e0b',
    'vital': '#dc2626',
    'trivial': '#94a3b8',
}


def dpmo_to_sigma(dpmo: float) -> float:
    """Convert DPMO to sigma level"""
    if dpmo <= 0:
        return 6.0
    if dpmo >= 1000000:
        return 0.0
    yield_rate = 1 - (dpmo / 1000000)
    z = stats.norm.ppf(yield_rate)
    sigma = z + 1.5
    return max(0, min(6, sigma))


def calculate_overall_metrics(df: pd.DataFrame, defective_col: Optional[str],
                               defect_count_col: Optional[str],
                               opportunities_col: Optional[str],
                               opportunities_per_unit: int) -> Dict:
    n = len(df)
    
    if defective_col and defective_col in df.columns:
        defective_units = df[defective_col].sum()
    elif defect_count_col and defect_count_col in df.columns:
        defective_units = (df[defect_count_col] > 0).sum()
    else:
        defective_units = 0
    
    if defect_count_col and defect_count_col in df.columns:
        total_defects = df[defect_count_col].sum()
    else:
        total_defects = defective_units
    
    if opportunities_col and opportunities_col in df.columns:
        total_opportunities = df[opportunities_col].sum()
    else:
        total_opportunities = n * opportunities_per_unit
    
    fty = ((n - defective_units) / n * 100) if n > 0 else 0
    dpu = total_defects / n if n > 0 else 0
    dpmo = (total_defects / total_opportunities * 1000000) if total_opportunities > 0 else 0
    sigma_level = dpmo_to_sigma(dpmo)
    yield_percent = ((n - defective_units) / n * 100) if n > 0 else 0
    defect_rate = (defective_units / n * 100) if n > 0 else 0
    rty = np.exp(-dpu) * 100
    
    return {
        'total_units': n,
        'total_defects': int(total_defects),
        'total_defective_units': int(defective_units),
        'fty': fty,
        'rty': rty,
        'dpu': dpu,
        'dpmo': dpmo,
        'sigma_level': sigma_level,
        'yield_percent': yield_percent,
        'defect_rate': defect_rate,
    }


def calculate_defect_categories(df: pd.DataFrame, defect_type_col: str,
                                 total_defects: int, total_opportunities: int) -> List[Dict]:
    if not defect_type_col or defect_type_col not in df.columns:
        return []
    
    defect_df = df[df[defect_type_col].notna() & (df[defect_type_col] != '')]
    if len(defect_df) == 0:
        return []
    
    counts = defect_df[defect_type_col].value_counts()
    total = counts.sum()
    if total == 0:
        return []
    
    categories = []
    cumulative = 0
    
    for category, count in counts.items():
        percent = (count / total * 100)
        cumulative += percent
        dpmo = (count / total_opportunities * 1000000) if total_opportunities > 0 else 0
        
        categories.append({
            'category': str(category),
            'count': int(count),
            'percent': percent,
            'cumulative_percent': cumulative,
            'dpmo': dpmo,
        })
    
    return categories


def calculate_pareto(categories: List[Dict]) -> Dict:
    if not categories:
        return {'vital_few': [], 'trivial_many': [], 'vital_few_percent': 0}
    
    vital_few = []
    trivial_many = []
    vital_few_percent = 0
    
    for cat in categories:
        if cat['cumulative_percent'] <= 80:
            vital_few.append(cat['category'])
            vital_few_percent = cat['cumulative_percent']
        else:
            trivial_many.append(cat['category'])
    
    if not vital_few and categories:
        vital_few = [categories[0]['category']]
        vital_few_percent = categories[0]['percent']
    
    return {
        'vital_few': vital_few,
        'trivial_many': trivial_many,
        'vital_few_percent': vital_few_percent,
    }


def calculate_process_steps(df: pd.DataFrame, process_step_col: str,
                            defective_col: Optional[str],
                            defect_count_col: Optional[str]) -> List[Dict]:
    if not process_step_col or process_step_col not in df.columns:
        return []
    
    steps = []
    for step in df[process_step_col].unique():
        step_df = df[df[process_step_col] == step]
        n = len(step_df)
        
        if defect_count_col and defect_count_col in step_df.columns:
            defects = step_df[defect_count_col].sum()
            defective = (step_df[defect_count_col] > 0).sum()
        elif defective_col and defective_col in step_df.columns:
            defects = step_df[defective_col].sum()
            defective = defects
        else:
            defects = 0
            defective = 0
        
        yield_pct = ((n - defective) / n * 100) if n > 0 else 100
        dpu = defects / n if n > 0 else 0
        fty = np.exp(-dpu) * 100
        
        steps.append({
            'step': str(step),
            'input': n,
            'output': n - int(defective),
            'defects': int(defects),
            'yield_percent': yield_pct,
            'fty': fty,
            'dpu': dpu,
        })
    
    return steps


def create_pareto_chart(categories: List[Dict], vital_few: List[str]) -> str:
    if not categories:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No defect data', ha='center', va='center', transform=ax.transAxes)
        return _fig_to_base64(fig)
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    labels = [c['category'] for c in categories]
    counts = [c['count'] for c in categories]
    cumulative = [c['cumulative_percent'] for c in categories]
    colors = [COLORS['vital'] if c['category'] in vital_few else COLORS['trivial'] for c in categories]
    
    bars = ax1.bar(range(len(labels)), counts, color=colors, edgecolor='white', linewidth=1)
    ax1.set_ylabel('Defect Count', fontsize=11, color=COLORS['defect'])
    ax1.tick_params(axis='y', labelcolor=COLORS['defect'])
    
    ax2 = ax1.twinx()
    ax2.plot(range(len(labels)), cumulative, 'o-', color=COLORS['cumulative'], linewidth=2, markersize=6)
    ax2.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='80% threshold')
    ax2.set_ylabel('Cumulative %', fontsize=11, color=COLORS['cumulative'])
    ax2.tick_params(axis='y', labelcolor=COLORS['cumulative'])
    ax2.set_ylim(0, 105)
    
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=45, ha='right')
    ax1.set_title('Defect Pareto Analysis', fontsize=14, fontweight='bold')
    
    for bar, count in zip(bars, counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(counts) * 0.02,
                str(count), ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_defect_distribution_chart(categories: List[Dict]) -> str:
    if not categories:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No defect data', ha='center', va='center', transform=ax.transAxes)
        return _fig_to_base64(fig)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    labels = [c['category'] for c in categories[:8]]
    sizes = [c['percent'] for c in categories[:8]]
    
    if len(categories) > 8:
        other_pct = sum(c['percent'] for c in categories[8:])
        labels.append('Other')
        sizes.append(other_pct)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90,
           explode=[0.05 if i == 0 else 0 for i in range(len(labels))])
    ax.set_title('Defect Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_process_yield_chart(process_steps: List[Dict]) -> str:
    if not process_steps:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No process step data', ha='center', va='center', transform=ax.transAxes)
        return _fig_to_base64(fig)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    steps = [s['step'] for s in process_steps]
    yields = [s['yield_percent'] for s in process_steps]
    colors = [COLORS['yield'] if y >= 95 else COLORS['cumulative'] if y >= 90 else COLORS['defect'] for y in yields]
    
    bars = ax.bar(steps, yields, color=colors, edgecolor='white', linewidth=2)
    
    for bar, y in zip(bars, yields):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{y:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.axhline(y=95, color='green', linestyle='--', alpha=0.5, label='95% target')
    ax.axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='90% minimum')
    ax.set_ylabel('Yield (%)', fontsize=11)
    ax.set_title('Process Step Yield', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 105)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_dpmo_chart(overall: Dict, categories: List[Dict]) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    sigma = overall['sigma_level']
    theta = np.linspace(0, np.pi, 100)
    r = 1
    
    ax1.plot(r * np.cos(theta), r * np.sin(theta), 'lightgray', linewidth=20)
    
    sections = [(0, 2, '#ef4444'), (2, 3, '#f97316'), (3, 4, '#f59e0b'), 
                (4, 5, '#84cc16'), (5, 6, '#22c55e')]
    
    for start, end, color in sections:
        mask = (theta >= np.pi * (1 - end/6)) & (theta <= np.pi * (1 - start/6))
        ax1.plot(r * np.cos(theta[mask]), r * np.sin(theta[mask]), color, linewidth=20)
    
    needle_angle = np.pi * (1 - sigma / 6)
    ax1.annotate('', xy=(0.8 * np.cos(needle_angle), 0.8 * np.sin(needle_angle)),
                xytext=(0, 0), arrowprops=dict(arrowstyle='->', color='black', lw=3))
    
    ax1.text(0, -0.3, f'{sigma:.2f}σ', ha='center', va='top', fontsize=24, fontweight='bold')
    ax1.text(0, -0.5, f'DPMO: {overall["dpmo"]:,.0f}', ha='center', va='top', fontsize=12)
    ax1.set_xlim(-1.3, 1.3)
    ax1.set_ylim(-0.6, 1.3)
    ax1.set_aspect('equal')
    ax1.axis('off')
    ax1.set_title('Sigma Level', fontsize=14, fontweight='bold')
    
    if categories:
        labels = [c['category'] for c in categories[:6]]
        dpmos = [c['dpmo'] for c in categories[:6]]
        
        bars = ax2.barh(labels, dpmos, color=COLORS['defect'], alpha=0.7, edgecolor='white')
        for bar, dpmo in zip(bars, dpmos):
            ax2.text(bar.get_width() + max(dpmos) * 0.02, bar.get_y() + bar.get_height()/2,
                    f'{dpmo:,.0f}', va='center', fontsize=9)
        ax2.set_xlabel('DPMO', fontsize=11)
        ax2.set_title('DPMO by Defect Category', fontsize=14, fontweight='bold')
        ax2.invert_yaxis()
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_key_insights(overall: Dict, pareto: Dict, categories: List[Dict]) -> List[Dict]:
    insights = []
    
    if overall['fty'] >= 99:
        insights.append({
            'title': f'Excellent Yield: {overall["fty"]:.2f}%',
            'description': 'First Time Yield at or above 99%. World-class performance.',
            'status': 'positive'
        })
    elif overall['fty'] >= 95:
        insights.append({
            'title': f'Good Yield: {overall["fty"]:.2f}%',
            'description': 'Yield above 95%. Continue monitoring and improving.',
            'status': 'neutral'
        })
    else:
        insights.append({
            'title': f'Low Yield: {overall["fty"]:.2f}%',
            'description': 'Yield below 95% requires immediate attention.',
            'status': 'warning'
        })
    
    if overall['sigma_level'] >= 4:
        insights.append({
            'title': f'High Sigma: {overall["sigma_level"]:.2f}σ',
            'description': f'DPMO of {overall["dpmo"]:,.0f} indicates strong process capability.',
            'status': 'positive'
        })
    elif overall['sigma_level'] < 3:
        insights.append({
            'title': f'Low Sigma: {overall["sigma_level"]:.2f}σ',
            'description': 'Below 3 sigma indicates significant quality issues.',
            'status': 'warning'
        })
    
    if pareto['vital_few']:
        insights.append({
            'title': f'Vital Few: {", ".join(pareto["vital_few"][:3])}',
            'description': f'These {len(pareto["vital_few"])} categories account for {pareto["vital_few_percent"]:.1f}% of defects.',
            'status': 'neutral'
        })
    
    return insights


@router.post("/yield-defect")
async def run_yield_defect_analysis(request: YieldDefectRequest) -> Dict[str, Any]:
    try:
        start_time = time.time()
        df = pd.DataFrame(request.data)
        
        if len(df) == 0:
            raise HTTPException(status_code=400, detail="No data provided")
        
        if request.defective_col and request.defective_col in df.columns:
            df[request.defective_col] = pd.to_numeric(df[request.defective_col], errors='coerce').fillna(0)
        
        if request.defect_count_col and request.defect_count_col in df.columns:
            df[request.defect_count_col] = pd.to_numeric(df[request.defect_count_col], errors='coerce').fillna(0)
        
        if request.opportunities_col and request.opportunities_col in df.columns:
            df[request.opportunities_col] = pd.to_numeric(df[request.opportunities_col], errors='coerce').fillna(request.opportunities_per_unit)
        
        overall = calculate_overall_metrics(
            df, request.defective_col, request.defect_count_col,
            request.opportunities_col, request.opportunities_per_unit
        )
        
        if request.opportunities_col and request.opportunities_col in df.columns:
            total_opportunities = df[request.opportunities_col].sum()
        else:
            total_opportunities = len(df) * request.opportunities_per_unit
        
        categories = calculate_defect_categories(
            df, request.defect_type_col, overall['total_defects'], total_opportunities
        )
        
        pareto = calculate_pareto(categories)
        
        process_steps = calculate_process_steps(
            df, request.process_step_col, request.defective_col, request.defect_count_col
        )
        
        if process_steps:
            rty = 100
            for step in process_steps:
                rty *= (step['yield_percent'] / 100)
            overall['rty'] = rty
        
        solve_time_ms = int((time.time() - start_time) * 1000)
        
        visualizations = {
            'pareto_chart': create_pareto_chart(categories, pareto['vital_few']),
            'defect_distribution': create_defect_distribution_chart(categories),
            'dpmo_chart': create_dpmo_chart(overall, categories),
        }
        
        if process_steps:
            visualizations['process_yield'] = create_process_yield_chart(process_steps)
        
        key_insights = generate_key_insights(overall, pareto, categories)
        top_defect = categories[0]['category'] if categories else 'N/A'
        
        results = {
            'overall': {k: _to_native_type(v) for k, v in overall.items()},
            'defect_categories': [{k: _to_native_type(v) for k, v in c.items()} for c in categories],
            'process_steps': [{k: _to_native_type(v) for k, v in s.items()} for s in process_steps] if process_steps else None,
            'pareto': pareto,
        }
        
        summary = {
            'fty': overall['fty'],
            'rty': overall['rty'],
            'dpmo': overall['dpmo'],
            'sigma_level': overall['sigma_level'],
            'top_defect': top_defect,
            'solve_time_ms': solve_time_ms,
        }
        
        return {
            'success': True,
            'results': results,
            'visualizations': visualizations,
            'key_insights': key_insights,
            'summary': summary,
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Yield defect analysis failed: {str(e)}")
