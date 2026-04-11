"""
Cost-Effectiveness Analysis (CEA) Router for FastAPI
Compare alternatives based on cost per unit of effectiveness/outcome
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class CostEffectivenessRequest(BaseModel):
    data: List[Dict[str, Any]]
    alternative_col: str  # Alternative/intervention name column
    cost_col: str  # Cost column
    effectiveness_col: str  # Effectiveness/outcome column
    baseline_alternative: Optional[str] = None  # Baseline for ICER calculation
    budget_constraint: Optional[float] = None  # Budget limit
    effectiveness_unit: str = "units"  # Unit of effectiveness


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


def calculate_cer(cost: float, effectiveness: float) -> float:
    """Calculate Cost-Effectiveness Ratio"""
    if effectiveness == 0:
        return float('inf')
    return cost / effectiveness


def calculate_icer(cost1: float, cost2: float, eff1: float, eff2: float) -> Optional[float]:
    """Calculate Incremental Cost-Effectiveness Ratio"""
    delta_cost = cost1 - cost2
    delta_eff = eff1 - eff2
    
    if delta_eff == 0:
        return None  # Undefined
    return delta_cost / delta_eff


def identify_dominated_alternatives(df: pd.DataFrame, cost_col: str, eff_col: str) -> List[str]:
    """Identify dominated alternatives (higher cost, lower or equal effectiveness)"""
    dominated = []
    alternatives = df.index.tolist()
    
    for i, alt1 in enumerate(alternatives):
        cost1 = df.loc[alt1, cost_col]
        eff1 = df.loc[alt1, eff_col]
        
        for alt2 in alternatives:
            if alt1 == alt2:
                continue
            cost2 = df.loc[alt2, cost_col]
            eff2 = df.loc[alt2, eff_col]
            
            # alt1 is dominated if alt2 has lower cost AND higher effectiveness
            if cost2 < cost1 and eff2 >= eff1:
                dominated.append(alt1)
                break
            # alt1 is weakly dominated if alt2 has lower cost AND same effectiveness
            elif cost2 <= cost1 and eff2 > eff1:
                dominated.append(alt1)
                break
    
    return list(set(dominated))


def calculate_efficiency_frontier(df: pd.DataFrame, cost_col: str, eff_col: str) -> List[str]:
    """Identify alternatives on the efficiency frontier"""
    # Sort by effectiveness
    df_sorted = df.sort_values(eff_col)
    
    frontier = []
    min_cost = float('inf')
    
    for idx in df_sorted.index:
        cost = df_sorted.loc[idx, cost_col]
        eff = df_sorted.loc[idx, eff_col]
        
        # Check if this alternative is on the frontier
        is_frontier = True
        for other_idx in df_sorted.index:
            if idx == other_idx:
                continue
            other_cost = df_sorted.loc[other_idx, cost_col]
            other_eff = df_sorted.loc[other_idx, eff_col]
            
            # Dominated if another has same or better effectiveness at lower cost
            if other_eff >= eff and other_cost < cost:
                is_frontier = False
                break
        
        if is_frontier:
            frontier.append(idx)
    
    return frontier


def generate_ce_plane(df: pd.DataFrame, cost_col: str, eff_col: str, 
                       alt_col: str, dominated: List[str], frontier: List[str]) -> str:
    """Generate cost-effectiveness plane"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    colors = []
    for alt in df[alt_col]:
        if alt in dominated:
            colors.append('#ef4444')  # Red for dominated
        elif alt in frontier:
            colors.append('#22c55e')  # Green for frontier
        else:
            colors.append('#3b82f6')  # Blue for others
    
    scatter = ax.scatter(df[eff_col], df[cost_col], c=colors, s=200, 
                         alpha=0.7, edgecolors='white', linewidth=2, zorder=5)
    
    # Add labels
    for i, row in df.iterrows():
        ax.annotate(row[alt_col], (row[eff_col], row[cost_col]),
                   xytext=(8, 8), textcoords='offset points',
                   fontsize=10, fontweight='bold')
    
    # Draw efficiency frontier line
    frontier_df = df[df[alt_col].isin(frontier)].sort_values(eff_col)
    if len(frontier_df) > 1:
        ax.plot(frontier_df[eff_col], frontier_df[cost_col], 'g--', 
                linewidth=2, alpha=0.7, label='Efficiency Frontier')
    
    ax.set_xlabel(f'Effectiveness ({eff_col})', fontsize=12)
    ax.set_ylabel(f'Cost ({cost_col})', fontsize=12)
    ax.set_title('Cost-Effectiveness Plane', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Legend
    legend_elements = [
        plt.scatter([], [], c='#22c55e', s=100, label='Efficient (Frontier)'),
        plt.scatter([], [], c='#3b82f6', s=100, label='Non-dominated'),
        plt.scatter([], [], c='#ef4444', s=100, label='Dominated')
    ]
    ax.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_cer_bar_chart(df: pd.DataFrame, alt_col: str, cer_col: str, 
                            dominated: List[str], eff_unit: str) -> str:
    """Generate CER bar chart"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    df_sorted = df.sort_values(cer_col)
    colors = ['#ef4444' if alt in dominated else '#22c55e' if i == 0 else '#3b82f6' 
              for i, alt in enumerate(df_sorted[alt_col])]
    
    bars = ax.barh(df_sorted[alt_col], df_sorted[cer_col], color=colors, 
                   edgecolor='white', linewidth=2)
    
    # Add value labels
    for bar, cer in zip(bars, df_sorted[cer_col]):
        if cer and not np.isinf(cer):
            ax.text(bar.get_width() + max(df_sorted[cer_col]) * 0.02, 
                   bar.get_y() + bar.get_height()/2,
                   f'${cer:,.2f}', va='center', fontsize=10)
    
    ax.set_xlabel(f'Cost per {eff_unit}', fontsize=12)
    ax.set_ylabel('Alternative', fontsize=12)
    ax.set_title('Cost-Effectiveness Ratio Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3, axis='x')
    
    # Mark best (lowest CER)
    best_alt = df_sorted[alt_col].iloc[0]
    ax.annotate('★ Most Cost-Effective', 
                xy=(df_sorted[cer_col].iloc[0], 0),
                xytext=(df_sorted[cer_col].iloc[0] + max(df_sorted[cer_col]) * 0.1, 0.5),
                fontsize=10, color='green', fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_icer_tornado(icer_results: List[Dict], baseline: str) -> str:
    """Generate ICER tornado chart"""
    if not icer_results:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Filter valid ICERs
    valid_icers = [r for r in icer_results if r['icer'] is not None and not np.isinf(r['icer'])]
    
    if not valid_icers:
        plt.close(fig)
        return None
    
    # Sort by ICER
    valid_icers = sorted(valid_icers, key=lambda x: x['icer'])
    
    alternatives = [r['alternative'] for r in valid_icers]
    icers = [r['icer'] for r in valid_icers]
    colors = ['#22c55e' if icer < 0 else '#ef4444' if icer > np.median(icers) else '#3b82f6' 
              for icer in icers]
    
    bars = ax.barh(alternatives, icers, color=colors, edgecolor='white', linewidth=2)
    
    ax.axvline(x=0, color='gray', linestyle='-', linewidth=2)
    
    # Add value labels
    for bar, icer in zip(bars, icers):
        offset = max(abs(min(icers)), abs(max(icers))) * 0.05
        x_pos = bar.get_width() + offset if bar.get_width() >= 0 else bar.get_width() - offset
        ha = 'left' if bar.get_width() >= 0 else 'right'
        ax.text(x_pos, bar.get_y() + bar.get_height()/2,
               f'${icer:,.2f}', va='center', ha=ha, fontsize=10)
    
    ax.set_xlabel(f'ICER vs {baseline}', fontsize=12)
    ax.set_ylabel('Alternative', fontsize=12)
    ax.set_title(f'Incremental Cost-Effectiveness Ratio (vs {baseline})', 
                fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3, axis='x')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_budget_impact(df: pd.DataFrame, alt_col: str, cost_col: str, 
                            eff_col: str, budget: Optional[float]) -> str:
    """Generate budget impact analysis chart"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    df_sorted = df.sort_values(cost_col)
    
    # Cumulative analysis
    x = np.arange(len(df_sorted))
    
    ax.bar(x, df_sorted[cost_col], color='#ef4444', alpha=0.7, label='Cost', width=0.4)
    ax.bar(x + 0.4, df_sorted[eff_col] * (df_sorted[cost_col].max() / df_sorted[eff_col].max()), 
           color='#22c55e', alpha=0.7, label='Effectiveness (scaled)', width=0.4)
    
    if budget:
        ax.axhline(y=budget, color='orange', linestyle='--', linewidth=2, label=f'Budget: ${budget:,.0f}')
    
    ax.set_xticks(x + 0.2)
    ax.set_xticklabels(df_sorted[alt_col], rotation=45, ha='right')
    ax.set_ylabel('Amount', fontsize=12)
    ax.set_title('Budget Impact Analysis', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.3, axis='y')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_efficiency_radar(df: pd.DataFrame, alt_col: str, cost_col: str, 
                               eff_col: str, cer_col: str) -> str:
    """Generate efficiency radar/spider chart"""
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Normalize metrics (0-1 scale, inverted for cost/CER so higher is better)
    df_norm = df.copy()
    df_norm['cost_norm'] = 1 - (df[cost_col] - df[cost_col].min()) / (df[cost_col].max() - df[cost_col].min() + 1e-10)
    df_norm['eff_norm'] = (df[eff_col] - df[eff_col].min()) / (df[eff_col].max() - df[eff_col].min() + 1e-10)
    df_norm['cer_norm'] = 1 - (df[cer_col] - df[cer_col].min()) / (df[cer_col].max() - df[cer_col].min() + 1e-10)
    
    # Composite efficiency score
    df_norm['efficiency_score'] = (df_norm['cost_norm'] + df_norm['eff_norm'] + df_norm['cer_norm']) / 3
    
    categories = ['Low Cost', 'High Effectiveness', 'Good CER', 'Overall Efficiency']
    n_cats = len(categories)
    angles = np.linspace(0, 2 * np.pi, n_cats, endpoint=False).tolist()
    angles += angles[:1]
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(df_norm)))
    
    for i, (idx, row) in enumerate(df_norm.iterrows()):
        values = [row['cost_norm'], row['eff_norm'], row['cer_norm'], row['efficiency_score']]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=2, label=row[alt_col], color=colors[i])
        ax.fill(angles, values, alpha=0.1, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_title('Efficiency Comparison (Normalized)', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_interpretation(df: pd.DataFrame, alt_col: str, cost_col: str, 
                            eff_col: str, cer_col: str, dominated: List[str],
                            frontier: List[str], eff_unit: str) -> Dict[str, Any]:
    """Generate comprehensive interpretation"""
    key_insights = []
    
    # Best CER
    best_cer_idx = df[cer_col].idxmin()
    best_alt = df.loc[best_cer_idx, alt_col]
    best_cer = df.loc[best_cer_idx, cer_col]
    
    key_insights.append({
        'title': 'Most Cost-Effective',
        'description': f"'{best_alt}' has the lowest CER at ${best_cer:,.2f} per {eff_unit}.",
        'status': 'positive'
    })
    
    # Dominated alternatives
    if dominated:
        key_insights.append({
            'title': 'Dominated Alternatives',
            'description': f"{len(dominated)} alternative(s) are dominated: {', '.join(dominated)}. These should be avoided.",
            'status': 'negative'
        })
    
    # Efficiency frontier
    if frontier:
        key_insights.append({
            'title': 'Efficiency Frontier',
            'description': f"{len(frontier)} alternative(s) are on the efficiency frontier: {', '.join(frontier)}.",
            'status': 'positive'
        })
    
    # Cost range
    cost_range = df[cost_col].max() - df[cost_col].min()
    eff_range = df[eff_col].max() - df[eff_col].min()
    
    key_insights.append({
        'title': 'Value Variation',
        'description': f"Costs range from ${df[cost_col].min():,.0f} to ${df[cost_col].max():,.0f}. " +
                       f"Effectiveness ranges from {df[eff_col].min():.1f} to {df[eff_col].max():.1f} {eff_unit}.",
        'status': 'neutral'
    })
    
    # Recommendations
    recommendations = []
    recommendations.append(f"Consider '{best_alt}' for best cost-effectiveness.")
    if dominated:
        recommendations.append(f"Eliminate dominated alternatives: {', '.join(dominated)}.")
    if len(frontier) > 1:
        recommendations.append("Choose from frontier alternatives based on budget and effectiveness goals.")
    
    return {
        'key_insights': key_insights,
        'recommendations': recommendations,
        'best_alternative': best_alt,
        'best_cer': _to_native_type(best_cer)
    }


@router.post("/cost-effectiveness")
async def run_cost_effectiveness_analysis(request: CostEffectivenessRequest) -> Dict[str, Any]:
    """
    Perform Cost-Effectiveness Analysis (CEA).
    
    Compares alternatives based on cost per unit of effectiveness,
    identifies dominated alternatives and efficiency frontier.
    """
    try:
        data = request.data
        alternative_col = request.alternative_col
        cost_col = request.cost_col
        effectiveness_col = request.effectiveness_col
        baseline_alternative = request.baseline_alternative
        budget_constraint = request.budget_constraint
        effectiveness_unit = request.effectiveness_unit
        
        if not data:
            raise HTTPException(status_code=400, detail="Data not provided.")
        
        df = pd.DataFrame(data)
        
        # Validate columns
        for col in [alternative_col, cost_col, effectiveness_col]:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{col}' not found.")
        
        # Convert to numeric
        df[cost_col] = pd.to_numeric(df[cost_col], errors='coerce')
        df[effectiveness_col] = pd.to_numeric(df[effectiveness_col], errors='coerce')
        
        # Drop rows with missing values
        df_clean = df.dropna(subset=[cost_col, effectiveness_col])
        
        if len(df_clean) < 2:
            raise HTTPException(status_code=400, detail="At least 2 alternatives required.")
        
        # Calculate CER for each alternative
        df_clean['cer'] = df_clean.apply(
            lambda row: calculate_cer(row[cost_col], row[effectiveness_col]), axis=1)
        
        # Identify dominated alternatives
        df_indexed = df_clean.set_index(alternative_col)
        dominated = identify_dominated_alternatives(df_indexed, cost_col, effectiveness_col)
        
        # Identify efficiency frontier
        frontier = calculate_efficiency_frontier(df_indexed, cost_col, effectiveness_col)
        
        # Calculate ICER if baseline provided
        icer_results = []
        if baseline_alternative and baseline_alternative in df_clean[alternative_col].values:
            baseline_row = df_clean[df_clean[alternative_col] == baseline_alternative].iloc[0]
            baseline_cost = baseline_row[cost_col]
            baseline_eff = baseline_row[effectiveness_col]
            
            for _, row in df_clean.iterrows():
                if row[alternative_col] != baseline_alternative:
                    icer = calculate_icer(row[cost_col], baseline_cost, 
                                         row[effectiveness_col], baseline_eff)
                    icer_results.append({
                        'alternative': row[alternative_col],
                        'icer': _to_native_type(icer),
                        'delta_cost': _to_native_type(row[cost_col] - baseline_cost),
                        'delta_effectiveness': _to_native_type(row[effectiveness_col] - baseline_eff)
                    })
        
        # Generate visualizations
        ce_plane = generate_ce_plane(df_clean, cost_col, effectiveness_col, 
                                      alternative_col, dominated, frontier)
        cer_chart = generate_cer_bar_chart(df_clean, alternative_col, 'cer', 
                                           dominated, effectiveness_unit)
        icer_chart = generate_icer_tornado(icer_results, baseline_alternative) if icer_results else None
        budget_chart = generate_budget_impact(df_clean, alternative_col, cost_col, 
                                               effectiveness_col, budget_constraint)
        radar_chart = generate_efficiency_radar(df_clean, alternative_col, cost_col,
                                                 effectiveness_col, 'cer')
        
        # Prepare alternatives summary
        alternatives_summary = []
        for _, row in df_clean.iterrows():
            alt_name = row[alternative_col]
            alternatives_summary.append({
                'alternative': alt_name,
                'cost': _to_native_type(row[cost_col]),
                'effectiveness': _to_native_type(row[effectiveness_col]),
                'cer': _to_native_type(row['cer']),
                'is_dominated': alt_name in dominated,
                'is_frontier': alt_name in frontier,
                'rank': None  # Will be filled
            })
        
        # Rank by CER
        alternatives_summary = sorted(alternatives_summary, key=lambda x: x['cer'] if x['cer'] else float('inf'))
        for i, alt in enumerate(alternatives_summary):
            alt['rank'] = i + 1
        
        # Generate interpretation
        interpretation = generate_interpretation(df_clean, alternative_col, cost_col,
                                                  effectiveness_col, 'cer', dominated,
                                                  frontier, effectiveness_unit)
        
        # Statistics
        statistics = {
            'n_alternatives': len(df_clean),
            'n_dominated': len(dominated),
            'n_frontier': len(frontier),
            'cost_mean': _to_native_type(df_clean[cost_col].mean()),
            'cost_std': _to_native_type(df_clean[cost_col].std()),
            'effectiveness_mean': _to_native_type(df_clean[effectiveness_col].mean()),
            'effectiveness_std': _to_native_type(df_clean[effectiveness_col].std()),
            'cer_mean': _to_native_type(df_clean['cer'].mean()),
            'cer_min': _to_native_type(df_clean['cer'].min()),
            'cer_max': _to_native_type(df_clean['cer'].max())
        }
        
        return {
            'alternatives_summary': alternatives_summary,
            'dominated_alternatives': dominated,
            'efficiency_frontier': frontier,
            'icer_results': icer_results,
            'statistics': statistics,
            'ce_plane': ce_plane,
            'cer_chart': cer_chart,
            'icer_chart': icer_chart,
            'budget_chart': budget_chart,
            'radar_chart': radar_chart,
            'interpretation': interpretation,
            'effectiveness_unit': effectiveness_unit,
            'baseline_alternative': baseline_alternative,
            'budget_constraint': budget_constraint
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cost-Effectiveness analysis failed: {str(e)}")
