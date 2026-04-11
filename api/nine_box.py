"""
9-Box Grid Analysis Router for FastAPI
Talent management matrix for performance vs potential evaluation
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import io
import base64
from scipy import stats
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class NineBoxRequest(BaseModel):
    data: List[Dict[str, Any]]
    performance_col: str  # Performance score column
    potential_col: str  # Potential score column
    name_col: Optional[str] = None  # Employee name column
    department_col: Optional[str] = None  # Department column (optional)
    performance_thresholds: Optional[List[float]] = None  # [low, high] thresholds
    potential_thresholds: Optional[List[float]] = None  # [low, high] thresholds


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


# 9-Box Grid definitions
BOX_DEFINITIONS = {
    (3, 3): {'name': 'Star', 'box': 9, 'color': '#22c55e', 'action': 'Promote & Develop', 'priority': 1},
    (2, 3): {'name': 'High Potential', 'box': 8, 'color': '#84cc16', 'action': 'Invest & Challenge', 'priority': 2},
    (1, 3): {'name': 'Potential Gem', 'box': 7, 'color': '#eab308', 'action': 'Coach Performance', 'priority': 3},
    (3, 2): {'name': 'High Performer', 'box': 6, 'color': '#84cc16', 'action': 'Retain & Reward', 'priority': 2},
    (2, 2): {'name': 'Core Player', 'box': 5, 'color': '#3b82f6', 'action': 'Develop & Monitor', 'priority': 4},
    (1, 2): {'name': 'Inconsistent', 'box': 4, 'color': '#f97316', 'action': 'Coach & Monitor', 'priority': 5},
    (3, 1): {'name': 'Solid Performer', 'box': 3, 'color': '#eab308', 'action': 'Recognize & Retain', 'priority': 3},
    (2, 1): {'name': 'Average', 'box': 2, 'color': '#f97316', 'action': 'Develop Skills', 'priority': 5},
    (1, 1): {'name': 'Underperformer', 'box': 1, 'color': '#ef4444', 'action': 'Improve or Exit', 'priority': 6},
}


def calculate_thresholds(values: np.ndarray, method: str = 'tercile') -> List[float]:
    """Calculate thresholds for categorization"""
    if method == 'tercile':
        return [np.percentile(values, 33.33), np.percentile(values, 66.67)]
    elif method == 'equal':
        min_val, max_val = values.min(), values.max()
        range_val = max_val - min_val
        return [min_val + range_val/3, min_val + 2*range_val/3]
    else:
        return [np.percentile(values, 33.33), np.percentile(values, 66.67)]


def categorize_score(value: float, thresholds: List[float]) -> int:
    """Categorize score into Low(1), Medium(2), High(3)"""
    if value <= thresholds[0]:
        return 1
    elif value <= thresholds[1]:
        return 2
    else:
        return 3


def assign_box(perf_category: int, pot_category: int) -> Dict[str, Any]:
    """Assign employee to 9-box grid position"""
    return BOX_DEFINITIONS.get((perf_category, pot_category), BOX_DEFINITIONS[(2, 2)])


def generate_nine_box_grid(df: pd.DataFrame, perf_col: str, pot_col: str, 
                            name_col: str, perf_thresh: List[float], 
                            pot_thresh: List[float]) -> str:
    """Generate 9-Box Grid visualization"""
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Create grid
    for i in range(4):
        ax.axhline(y=i, color='gray', linewidth=2)
        ax.axvline(x=i, color='gray', linewidth=2)
    
    # Fill boxes with colors
    box_positions = {
        (1, 1): (0, 0), (2, 1): (1, 0), (3, 1): (2, 0),
        (1, 2): (0, 1), (2, 2): (1, 1), (3, 2): (2, 1),
        (1, 3): (0, 2), (2, 3): (1, 2), (3, 3): (2, 2),
    }
    
    for (perf, pot), (x, y) in box_positions.items():
        box_info = BOX_DEFINITIONS[(perf, pot)]
        rect = mpatches.FancyBboxPatch((x + 0.02, y + 0.02), 0.96, 0.96,
                                        boxstyle="round,pad=0.02",
                                        facecolor=box_info['color'],
                                        alpha=0.3, edgecolor=box_info['color'],
                                        linewidth=2)
        ax.add_patch(rect)
        
        # Box label
        ax.text(x + 0.5, y + 0.85, box_info['name'], ha='center', va='center',
                fontsize=11, fontweight='bold', color=box_info['color'])
        ax.text(x + 0.5, y + 0.7, f"Box {box_info['box']}", ha='center', va='center',
                fontsize=9, color='gray')
    
    # Plot employees
    for _, row in df.iterrows():
        perf_cat = row['performance_category']
        pot_cat = row['potential_category']
        
        # Calculate position within box (add jitter for overlapping)
        base_x = perf_cat - 1 + 0.5
        base_y = pot_cat - 1 + 0.35
        
        # Add small random jitter
        jitter_x = np.random.uniform(-0.25, 0.25)
        jitter_y = np.random.uniform(-0.15, 0.15)
        
        x_pos = base_x + jitter_x
        y_pos = base_y + jitter_y
        
        box_info = BOX_DEFINITIONS[(perf_cat, pot_cat)]
        ax.scatter(x_pos, y_pos, s=150, c=box_info['color'], edgecolors='white',
                   linewidth=2, zorder=5, alpha=0.8)
        
        # Add name label if available
        if name_col and pd.notna(row.get(name_col)):
            name = str(row[name_col])[:10]
            ax.annotate(name, (x_pos, y_pos), xytext=(3, 3), textcoords='offset points',
                       fontsize=7, alpha=0.7)
    
    # Count employees per box
    for (perf, pot), (x, y) in box_positions.items():
        count = len(df[(df['performance_category'] == perf) & (df['potential_category'] == pot)])
        if count > 0:
            ax.text(x + 0.5, y + 0.15, f"n = {count}", ha='center', va='center',
                    fontsize=10, fontweight='bold', 
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Axis labels
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_xticks([0.5, 1.5, 2.5])
    ax.set_xticklabels(['Low', 'Medium', 'High'], fontsize=12)
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels(['Low', 'Medium', 'High'], fontsize=12)
    ax.set_xlabel('Performance →', fontsize=14, fontweight='bold')
    ax.set_ylabel('Potential →', fontsize=14, fontweight='bold')
    ax.set_title('9-Box Talent Grid', fontsize=16, fontweight='bold', pad=20)
    
    # Add threshold info
    thresh_text = f"Performance: Low ≤{perf_thresh[0]:.1f}, High >{perf_thresh[1]:.1f}\n"
    thresh_text += f"Potential: Low ≤{pot_thresh[0]:.1f}, High >{pot_thresh[1]:.1f}"
    ax.text(0.02, 0.98, thresh_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_distribution_chart(df: pd.DataFrame) -> str:
    """Generate box distribution bar chart"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Box distribution
    box_counts = df['box_number'].value_counts().sort_index()
    colors = [BOX_DEFINITIONS[(p, t)]['color'] for p in [1,2,3] for t in [1,2,3]]
    
    bars = axes[0].bar(range(1, 10), [box_counts.get(i, 0) for i in range(1, 10)],
                       color=colors, edgecolor='white', linewidth=2)
    
    # Add labels
    box_names = [BOX_DEFINITIONS[(p, t)]['name'] for p in [1,2,3] for t in [1,2,3]]
    axes[0].set_xticks(range(1, 10))
    axes[0].set_xticklabels([f"Box {i}\n{box_names[i-1]}" for i in range(1, 10)], 
                            fontsize=8, rotation=45, ha='right')
    axes[0].set_ylabel('Number of Employees', fontsize=12)
    axes[0].set_title('Distribution by Box', fontsize=14, fontweight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.3, axis='y')
    
    # Add count labels
    for bar, count in zip(bars, [box_counts.get(i, 0) for i in range(1, 10)]):
        if count > 0:
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                        str(count), ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Priority distribution (pie chart)
    priority_groups = {
        'Top Talent (Box 6,8,9)': len(df[df['box_number'].isin([6, 8, 9])]),
        'Core Talent (Box 3,5,7)': len(df[df['box_number'].isin([3, 5, 7])]),
        'Needs Development (Box 2,4)': len(df[df['box_number'].isin([2, 4])]),
        'Action Required (Box 1)': len(df[df['box_number'] == 1])
    }
    
    values = [v for v in priority_groups.values() if v > 0]
    labels = [k for k, v in priority_groups.items() if v > 0]
    colors = ['#22c55e', '#3b82f6', '#f97316', '#ef4444'][:len(values)]
    
    if sum(values) > 0:
        wedges, texts, autotexts = axes[1].pie(values, labels=labels, colors=colors,
                                                autopct='%1.1f%%', startangle=90,
                                                explode=[0.02] * len(values))
        axes[1].set_title('Talent Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_scatter_plot(df: pd.DataFrame, perf_col: str, pot_col: str,
                          name_col: str, perf_thresh: List[float],
                          pot_thresh: List[float]) -> str:
    """Generate performance vs potential scatter plot"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Color by box
    colors = df.apply(lambda row: BOX_DEFINITIONS[(row['performance_category'], 
                                                    row['potential_category'])]['color'], axis=1)
    
    scatter = ax.scatter(df[perf_col], df[pot_col], c=colors, s=100, 
                         alpha=0.7, edgecolors='white', linewidth=1.5)
    
    # Add threshold lines
    ax.axvline(x=perf_thresh[0], color='gray', linestyle='--', alpha=0.5, label='Performance Threshold')
    ax.axvline(x=perf_thresh[1], color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=pot_thresh[0], color='gray', linestyle=':', alpha=0.5, label='Potential Threshold')
    ax.axhline(y=pot_thresh[1], color='gray', linestyle=':', alpha=0.5)
    
    # Add names if available
    if name_col:
        for _, row in df.iterrows():
            if pd.notna(row.get(name_col)):
                ax.annotate(str(row[name_col])[:8], (row[perf_col], row[pot_col]),
                           xytext=(5, 5), textcoords='offset points', fontsize=7, alpha=0.7)
    
    ax.set_xlabel(f'Performance ({perf_col})', fontsize=12)
    ax.set_ylabel(f'Potential ({pot_col})', fontsize=12)
    ax.set_title('Performance vs Potential Scatter', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    # Legend
    legend_elements = [mpatches.Patch(facecolor=info['color'], label=info['name'])
                       for info in list(BOX_DEFINITIONS.values())[::-1]]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=8, ncol=3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_department_chart(df: pd.DataFrame, dept_col: str) -> str:
    """Generate department breakdown chart"""
    if not dept_col or dept_col not in df.columns:
        return None
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Cross-tabulation
    dept_box = pd.crosstab(df[dept_col], df['box_name'])
    
    # Reorder columns
    box_order = ['Star', 'High Potential', 'High Performer', 'Potential Gem', 
                 'Core Player', 'Solid Performer', 'Inconsistent', 'Average', 'Underperformer']
    existing_cols = [col for col in box_order if col in dept_box.columns]
    dept_box = dept_box[existing_cols]
    
    # Stacked bar
    colors = [BOX_DEFINITIONS[(p, t)]['color'] for name in existing_cols 
              for (p, t), info in BOX_DEFINITIONS.items() if info['name'] == name]
    
    dept_box.plot(kind='barh', stacked=True, ax=ax, color=colors[:len(existing_cols)],
                  edgecolor='white', linewidth=0.5)
    
    ax.set_xlabel('Number of Employees', fontsize=12)
    ax.set_ylabel('Department', fontsize=12)
    ax.set_title('9-Box Distribution by Department', fontsize=14, fontweight='bold')
    ax.legend(title='Box', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.3, axis='x')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_interpretation(df: pd.DataFrame, box_summary: List[Dict]) -> Dict[str, Any]:
    """Generate comprehensive interpretation"""
    total = len(df)
    
    key_insights = []
    
    # Top talent insight
    top_talent = len(df[df['box_number'].isin([6, 8, 9])])
    top_pct = (top_talent / total * 100) if total > 0 else 0
    key_insights.append({
        'title': 'Top Talent Pool',
        'description': f"{top_talent} employees ({top_pct:.1f}%) are in the top talent boxes (6, 8, 9). " +
                       ("This is a healthy proportion." if top_pct >= 15 else "Consider talent acquisition strategies."),
        'status': 'positive' if top_pct >= 15 else 'warning'
    })
    
    # Stars
    stars = len(df[df['box_number'] == 9])
    if stars > 0:
        key_insights.append({
            'title': 'Stars (Box 9)',
            'description': f"{stars} employees are stars - high performers with high potential. Prioritize retention and succession planning.",
            'status': 'positive'
        })
    
    # Underperformers
    underperformers = len(df[df['box_number'] == 1])
    if underperformers > 0:
        under_pct = underperformers / total * 100
        key_insights.append({
            'title': 'Action Required',
            'description': f"{underperformers} employees ({under_pct:.1f}%) are underperformers. Implement performance improvement plans.",
            'status': 'negative' if under_pct > 10 else 'warning'
        })
    
    # Core players
    core = len(df[df['box_number'] == 5])
    core_pct = (core / total * 100) if total > 0 else 0
    key_insights.append({
        'title': 'Core Players',
        'description': f"{core} employees ({core_pct:.1f}%) are core players - the backbone of the organization.",
        'status': 'neutral'
    })
    
    # Recommendations
    recommendations = []
    if top_pct < 15:
        recommendations.append("Increase investment in high-potential employee development")
    if underperformers / total > 0.1 if total > 0 else False:
        recommendations.append("Review hiring processes and performance management systems")
    if stars < 3:
        recommendations.append("Identify and develop potential successors for key roles")
    
    return {
        'key_insights': key_insights,
        'recommendations': recommendations,
        'summary': {
            'total_employees': total,
            'top_talent_count': top_talent,
            'top_talent_pct': _to_native_type(top_pct),
            'stars_count': stars,
            'core_count': core,
            'underperformer_count': underperformers
        }
    }


@router.post("/nine-box")
async def run_nine_box_analysis(request: NineBoxRequest) -> Dict[str, Any]:
    """
    Perform 9-Box Grid Talent Analysis.
    
    Categorizes employees based on performance and potential into a 3x3 matrix
    for talent management and succession planning.
    """
    try:
        data = request.data
        performance_col = request.performance_col
        potential_col = request.potential_col
        name_col = request.name_col
        department_col = request.department_col
        perf_thresholds = request.performance_thresholds
        pot_thresholds = request.potential_thresholds
        
        if not data:
            raise HTTPException(status_code=400, detail="Data not provided.")
        
        df = pd.DataFrame(data)
        
        # Validate columns
        if performance_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Performance column '{performance_col}' not found.")
        if potential_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Potential column '{potential_col}' not found.")
        
        # Convert to numeric
        df[performance_col] = pd.to_numeric(df[performance_col], errors='coerce')
        df[potential_col] = pd.to_numeric(df[potential_col], errors='coerce')
        
        # Drop rows with missing values
        df_clean = df.dropna(subset=[performance_col, potential_col])
        
        if len(df_clean) < 3:
            raise HTTPException(status_code=400, detail="At least 3 employees required for analysis.")
        
        # Calculate thresholds if not provided
        if perf_thresholds is None or len(perf_thresholds) != 2:
            perf_thresholds = calculate_thresholds(df_clean[performance_col].values)
        if pot_thresholds is None or len(pot_thresholds) != 2:
            pot_thresholds = calculate_thresholds(df_clean[potential_col].values)
        
        # Ensure thresholds are sorted
        perf_thresholds = sorted(perf_thresholds)
        pot_thresholds = sorted(pot_thresholds)
        
        # Categorize employees
        df_clean['performance_category'] = df_clean[performance_col].apply(
            lambda x: categorize_score(x, perf_thresholds))
        df_clean['potential_category'] = df_clean[potential_col].apply(
            lambda x: categorize_score(x, pot_thresholds))
        
        # Assign boxes
        df_clean['box_info'] = df_clean.apply(
            lambda row: assign_box(row['performance_category'], row['potential_category']), axis=1)
        df_clean['box_name'] = df_clean['box_info'].apply(lambda x: x['name'])
        df_clean['box_number'] = df_clean['box_info'].apply(lambda x: x['box'])
        df_clean['action'] = df_clean['box_info'].apply(lambda x: x['action'])
        df_clean['priority'] = df_clean['box_info'].apply(lambda x: x['priority'])
        
        # Generate visualizations
        grid_plot = generate_nine_box_grid(df_clean, performance_col, potential_col,
                                            name_col, perf_thresholds, pot_thresholds)
        distribution_chart = generate_distribution_chart(df_clean)
        scatter_plot = generate_scatter_plot(df_clean, performance_col, potential_col,
                                              name_col, perf_thresholds, pot_thresholds)
        department_chart = generate_department_chart(df_clean, department_col) if department_col else None
        
        # Box summary
        box_summary = []
        for box_num in range(1, 10):
            box_df = df_clean[df_clean['box_number'] == box_num]
            if len(box_df) > 0:
                box_info = list(BOX_DEFINITIONS.values())[box_num - 1]
                box_summary.append({
                    'box': box_num,
                    'name': box_info['name'],
                    'count': len(box_df),
                    'percentage': _to_native_type(len(box_df) / len(df_clean) * 100),
                    'action': box_info['action'],
                    'priority': box_info['priority']
                })
        
        # Employee details
        employee_details = []
        for _, row in df_clean.iterrows():
            detail = {
                'name': row.get(name_col, f"Employee_{_}") if name_col else f"Employee_{_}",
                'performance': _to_native_type(row[performance_col]),
                'potential': _to_native_type(row[potential_col]),
                'box': row['box_number'],
                'box_name': row['box_name'],
                'action': row['action'],
                'priority': row['priority']
            }
            if department_col and department_col in df_clean.columns:
                detail['department'] = row.get(department_col)
            employee_details.append(detail)
        
        # Sort by priority then by performance
        employee_details = sorted(employee_details, key=lambda x: (x['priority'], -x['performance']))
        
        # Generate interpretation
        interpretation = generate_interpretation(df_clean, box_summary)
        
        return {
            'box_summary': box_summary,
            'employee_details': employee_details,
            'thresholds': {
                'performance': [_to_native_type(t) for t in perf_thresholds],
                'potential': [_to_native_type(t) for t in pot_thresholds]
            },
            'statistics': {
                'total_employees': len(df_clean),
                'performance_mean': _to_native_type(df_clean[performance_col].mean()),
                'performance_std': _to_native_type(df_clean[performance_col].std()),
                'potential_mean': _to_native_type(df_clean[potential_col].mean()),
                'potential_std': _to_native_type(df_clean[potential_col].std()),
                'correlation': _to_native_type(df_clean[performance_col].corr(df_clean[potential_col]))
            },
            'grid_plot': grid_plot,
            'distribution_chart': distribution_chart,
            'scatter_plot': scatter_plot,
            'department_chart': department_chart,
            'interpretation': interpretation
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"9-Box analysis failed: {str(e)}")
