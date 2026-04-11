"""
Bin Packing Optimization Router for FastAPI
Using Google OR-Tools and heuristic algorithms
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Literal
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io
import base64
import time
import warnings

from ortools.linear_solver import pywraplp

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class BinPackingRequest(BaseModel):
    data: List[Dict[str, Any]]
    item_id_col: str
    size_col: str
    weight_col: Optional[str] = None
    category_col: Optional[str] = None
    algorithm: Literal["first_fit_decreasing", "best_fit_decreasing", "optimal"] = "first_fit_decreasing"
    bin_capacity: int = 100
    weight_capacity: Optional[float] = None
    max_bins: Optional[int] = None


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


BIN_COLORS = [
    '#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6',
    '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1'
]


def first_fit_decreasing(items: List[Dict], bin_capacity: int) -> List[Dict]:
    """First Fit Decreasing algorithm"""
    # Sort items by size descending
    sorted_items = sorted(items, key=lambda x: x['size'], reverse=True)
    
    bins = []  # List of {'items': [], 'total_size': 0}
    
    for item in sorted_items:
        placed = False
        
        # Try to fit in existing bins
        for bin_data in bins:
            if bin_data['total_size'] + item['size'] <= bin_capacity:
                bin_data['items'].append(item)
                bin_data['total_size'] += item['size']
                placed = True
                break
        
        # Create new bin if needed
        if not placed:
            bins.append({
                'items': [item],
                'total_size': item['size']
            })
    
    return bins


def best_fit_decreasing(items: List[Dict], bin_capacity: int) -> List[Dict]:
    """Best Fit Decreasing algorithm"""
    # Sort items by size descending
    sorted_items = sorted(items, key=lambda x: x['size'], reverse=True)
    
    bins = []
    
    for item in sorted_items:
        best_bin_idx = -1
        min_remaining = bin_capacity + 1
        
        # Find bin with minimum remaining space that can fit item
        for idx, bin_data in enumerate(bins):
            remaining = bin_capacity - bin_data['total_size']
            if item['size'] <= remaining < min_remaining:
                best_bin_idx = idx
                min_remaining = remaining
        
        if best_bin_idx >= 0:
            bins[best_bin_idx]['items'].append(item)
            bins[best_bin_idx]['total_size'] += item['size']
        else:
            bins.append({
                'items': [item],
                'total_size': item['size']
            })
    
    return bins


def optimal_bin_packing(items: List[Dict], bin_capacity: int, max_bins: Optional[int] = None) -> List[Dict]:
    """Optimal bin packing using OR-Tools MIP solver"""
    n = len(items)
    sizes = [item['size'] for item in items]
    
    # Upper bound on number of bins
    num_bins = max_bins if max_bins else n
    
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        # Fallback to FFD if solver not available
        return first_fit_decreasing(items, bin_capacity)
    
    # Variables
    # x[i][j] = 1 if item i is placed in bin j
    x = {}
    for i in range(n):
        for j in range(num_bins):
            x[i, j] = solver.IntVar(0, 1, f'x_{i}_{j}')
    
    # y[j] = 1 if bin j is used
    y = {}
    for j in range(num_bins):
        y[j] = solver.IntVar(0, 1, f'y_{j}')
    
    # Constraints
    # Each item must be in exactly one bin
    for i in range(n):
        solver.Add(sum(x[i, j] for j in range(num_bins)) == 1)
    
    # Bin capacity constraint
    for j in range(num_bins):
        solver.Add(
            sum(sizes[i] * x[i, j] for i in range(n)) <= bin_capacity * y[j]
        )
    
    # Objective: minimize number of bins used
    solver.Minimize(sum(y[j] for j in range(num_bins)))
    
    # Solve with time limit
    solver.SetTimeLimit(30000)  # 30 seconds
    status = solver.Solve()
    
    if status not in [pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE]:
        # Fallback to FFD
        return first_fit_decreasing(items, bin_capacity)
    
    # Extract solution
    bins = []
    for j in range(num_bins):
        if y[j].solution_value() > 0.5:
            bin_items = []
            total_size = 0
            for i in range(n):
                if x[i, j].solution_value() > 0.5:
                    bin_items.append(items[i])
                    total_size += sizes[i]
            if bin_items:
                bins.append({
                    'items': bin_items,
                    'total_size': total_size
                })
    
    return bins


def create_bin_visualization(bins: List[Dict], bin_capacity: int) -> str:
    """Create visual representation of bins"""
    num_bins = len(bins)
    if num_bins == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No bins', ha='center', va='center', transform=ax.transAxes)
        return _fig_to_base64(fig)
    
    # Limit display
    display_bins = bins[:20]
    num_display = len(display_bins)
    
    cols = min(10, num_display)
    rows = (num_display + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(min(15, cols * 1.5), rows * 3))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, bin_data in enumerate(display_bins):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        utilization = bin_data['total_size'] / bin_capacity * 100
        color = BIN_COLORS[idx % len(BIN_COLORS)]
        
        # Draw bin outline
        ax.add_patch(plt.Rectangle((0.1, 0.1), 0.8, 0.8, fill=False, 
                                   edgecolor='black', linewidth=2))
        
        # Draw fill level
        fill_height = 0.8 * (bin_data['total_size'] / bin_capacity)
        ax.add_patch(plt.Rectangle((0.1, 0.1), 0.8, fill_height, 
                                   facecolor=color, alpha=0.7))
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(f'Bin {idx + 1}\n{utilization:.0f}%', fontsize=9)
    
    # Hide unused subplots
    for idx in range(num_display, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.suptitle('Bin Fill Levels', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_utilization_chart(bins: List[Dict], bin_capacity: int) -> str:
    """Create utilization bar chart"""
    if not bins:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return _fig_to_base64(fig)
    
    # Limit to 30 bins
    display_bins = bins[:30]
    
    bin_ids = [f'B{i+1}' for i in range(len(display_bins))]
    utilizations = [b['total_size'] / bin_capacity * 100 for b in display_bins]
    
    fig, ax = plt.subplots(figsize=(max(10, len(display_bins) * 0.4), 6))
    
    colors = ['#22c55e' if u >= 80 else '#f59e0b' if u >= 60 else '#ef4444' for u in utilizations]
    bars = ax.bar(bin_ids, utilizations, color=colors, edgecolor='white', linewidth=1)
    
    avg_util = np.mean(utilizations)
    ax.axhline(y=avg_util, color='blue', linestyle='--', alpha=0.7, label=f'Avg: {avg_util:.1f}%')
    ax.axhline(y=80, color='green', linestyle=':', alpha=0.5, label='Target: 80%')
    
    ax.set_ylabel('Utilization (%)', fontsize=11)
    ax.set_xlabel('Bin', fontsize=11)
    ax.set_title('Bin Utilization', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(loc='lower right')
    
    plt.xticks(rotation=45 if len(display_bins) > 15 else 0, ha='right' if len(display_bins) > 15 else 'center')
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_size_distribution(items: List[Dict], bin_capacity: int) -> str:
    """Create item size distribution chart"""
    if not items:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return _fig_to_base64(fig)
    
    sizes = [item['size'] for item in items]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(sizes, bins=20, color='#3b82f6', edgecolor='white', alpha=0.7)
    
    ax.axvline(x=np.mean(sizes), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(sizes):.1f}')
    ax.axvline(x=np.median(sizes), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(sizes):.1f}')
    ax.axvline(x=bin_capacity, color='black', linestyle='-', linewidth=2, label=f'Bin Capacity: {bin_capacity}')
    
    ax.set_xlabel('Item Size', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Item Size Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_bin_comparison(bins: List[Dict], bin_capacity: int) -> str:
    """Create bin comparison chart"""
    if not bins:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return _fig_to_base64(fig)
    
    display_bins = bins[:20]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bin_ids = range(len(display_bins))
    used = [b['total_size'] for b in display_bins]
    remaining = [bin_capacity - b['total_size'] for b in display_bins]
    
    ax.bar(bin_ids, used, label='Used', color='#3b82f6', edgecolor='white')
    ax.bar(bin_ids, remaining, bottom=used, label='Remaining', color='#e5e7eb', edgecolor='white')
    
    ax.set_xlabel('Bin', fontsize=11)
    ax.set_ylabel('Capacity', fontsize=11)
    ax.set_title('Bin Capacity Usage', fontsize=14, fontweight='bold')
    ax.set_xticks(bin_ids)
    ax.set_xticklabels([f'B{i+1}' for i in bin_ids])
    ax.legend()
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_key_insights(bins: List[Dict], total_items: int, total_size: int,
                          bin_capacity: int, wasted_space: int) -> List[Dict]:
    """Generate key insights"""
    insights = []
    
    num_bins = len(bins)
    theoretical_min = (total_size + bin_capacity - 1) // bin_capacity  # Ceiling division
    avg_utilization = (total_size / (num_bins * bin_capacity) * 100) if num_bins > 0 else 0
    
    # Efficiency insight
    if num_bins == theoretical_min:
        insights.append({
            'title': 'Optimal Solution Achieved',
            'description': f'Using exactly {num_bins} bins, which is the theoretical minimum.',
            'status': 'positive'
        })
    elif num_bins <= theoretical_min * 1.1:
        insights.append({
            'title': f'Near-Optimal: {num_bins} bins used',
            'description': f'Only {num_bins - theoretical_min} more than theoretical minimum ({theoretical_min}).',
            'status': 'positive'
        })
    else:
        insights.append({
            'title': f'{num_bins} Bins Used',
            'description': f'Theoretical minimum is {theoretical_min}. Consider different algorithm.',
            'status': 'neutral'
        })
    
    # Utilization insight
    if avg_utilization >= 80:
        insights.append({
            'title': f'High Utilization: {avg_utilization:.1f}%',
            'description': 'Excellent space efficiency across all bins.',
            'status': 'positive'
        })
    elif avg_utilization >= 65:
        insights.append({
            'title': f'Moderate Utilization: {avg_utilization:.1f}%',
            'description': 'Good efficiency with some room for improvement.',
            'status': 'neutral'
        })
    else:
        insights.append({
            'title': f'Low Utilization: {avg_utilization:.1f}%',
            'description': 'Consider adjusting bin size or using different algorithm.',
            'status': 'warning'
        })
    
    # Waste insight
    waste_percent = (wasted_space / (num_bins * bin_capacity) * 100) if num_bins > 0 else 0
    if waste_percent > 30:
        insights.append({
            'title': f'High Waste: {wasted_space} units ({waste_percent:.1f}%)',
            'description': 'Significant unused space. Review bin capacity or item grouping.',
            'status': 'warning'
        })
    
    return insights


@router.post("/bin-packing")
async def run_bin_packing(request: BinPackingRequest) -> Dict[str, Any]:
    """Run bin packing optimization"""
    try:
        start_time = time.time()
        df = pd.DataFrame(request.data)
        
        # Validate columns
        if request.item_id_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Item ID column '{request.item_id_col}' not found")
        if request.size_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Size column '{request.size_col}' not found")
        
        # Prepare items
        items = []
        for _, row in df.iterrows():
            item_id = str(row[request.item_id_col])
            size = int(row[request.size_col])
            
            if size > request.bin_capacity:
                continue  # Skip items larger than bin capacity
            
            items.append({
                'id': item_id,
                'size': size
            })
        
        if not items:
            raise HTTPException(status_code=400, detail="No valid items to pack")
        
        total_items = len(items)
        total_size = sum(item['size'] for item in items)
        
        # Run algorithm
        if request.algorithm == 'optimal':
            bins_result = optimal_bin_packing(items, request.bin_capacity, request.max_bins)
        elif request.algorithm == 'best_fit_decreasing':
            bins_result = best_fit_decreasing(items, request.bin_capacity)
        else:
            bins_result = first_fit_decreasing(items, request.bin_capacity)
        
        solve_time_ms = int((time.time() - start_time) * 1000)
        
        # Format results
        bins = []
        for idx, bin_data in enumerate(bins_result):
            bin_items = [item['id'] for item in bin_data['items']]
            bin_sizes = [item['size'] for item in bin_data['items']]
            utilization = (bin_data['total_size'] / request.bin_capacity * 100)
            
            bins.append({
                'bin_id': idx + 1,
                'items': bin_items,
                'item_sizes': bin_sizes,
                'total_size': bin_data['total_size'],
                'remaining_capacity': request.bin_capacity - bin_data['total_size'],
                'utilization': utilization
            })
        
        num_bins = len(bins)
        utilizations = [b['utilization'] for b in bins]
        wasted_space = sum(b['remaining_capacity'] for b in bins)
        
        # Find unassigned items (if any were too large)
        assigned_ids = set()
        for bin_data in bins:
            assigned_ids.update(bin_data['items'])
        unassigned = [str(row[request.item_id_col]) for _, row in df.iterrows() 
                     if str(row[request.item_id_col]) not in assigned_ids]
        
        # Create visualizations
        visualizations = {
            'bin_visualization': create_bin_visualization(bins_result, request.bin_capacity),
            'utilization_chart': create_utilization_chart(bins_result, request.bin_capacity),
            'size_distribution': create_size_distribution(items, request.bin_capacity),
            'bin_comparison': create_bin_comparison(bins_result, request.bin_capacity)
        }
        
        # Generate insights
        key_insights = generate_key_insights(bins, total_items, total_size, 
                                            request.bin_capacity, wasted_space)
        
        # Prepare results
        results = {
            'bins': [{k: _to_native_type(v) for k, v in b.items()} for b in bins],
            'num_bins_used': num_bins,
            'total_items': total_items,
            'total_size': total_size,
            'bin_capacity': request.bin_capacity,
            'avg_utilization': _to_native_type(np.mean(utilizations)) if utilizations else 0,
            'min_utilization': _to_native_type(min(utilizations)) if utilizations else 0,
            'max_utilization': _to_native_type(max(utilizations)) if utilizations else 0,
            'wasted_space': wasted_space,
            'unassigned_items': unassigned
        }
        
        summary = {
            'algorithm': request.algorithm,
            'num_items': total_items,
            'num_bins': num_bins,
            'bin_capacity': request.bin_capacity,
            'avg_utilization': _to_native_type(np.mean(utilizations)) if utilizations else 0,
            'solve_time_ms': solve_time_ms
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
        raise HTTPException(status_code=500, detail=f"Bin packing failed: {str(e)}")
