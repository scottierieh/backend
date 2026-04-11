"""
Assignment Problem Router for FastAPI
Using Google OR-Tools Linear Sum Assignment solver
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
import seaborn as sns
import io
import base64
import time
import warnings

from ortools.graph.python import linear_sum_assignment

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class AssignmentRequest(BaseModel):
    data: List[Dict[str, Any]]
    worker_col: str
    task_col: str
    cost_col: str
    problem_type: Literal["min_cost", "max_profit", "balanced"] = "min_cost"
    allow_partial: bool = False


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


ASSIGNMENT_COLORS = [
    '#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6',
    '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1'
]


def solve_assignment(cost_matrix: np.ndarray, workers: List[str], tasks: List[str],
                     problem_type: str) -> Dict:
    """Solve assignment problem using OR-Tools"""
    num_workers = len(workers)
    num_tasks = len(tasks)
    
    # Handle rectangular matrices by padding
    size = max(num_workers, num_tasks)
    padded_matrix = np.full((size, size), 0 if problem_type == 'max_profit' else 999999)
    padded_matrix[:num_workers, :num_tasks] = cost_matrix
    
    # For max profit, negate costs
    if problem_type == 'max_profit':
        solve_matrix = -padded_matrix
    else:
        solve_matrix = padded_matrix
    
    # Create assignment solver
    assignment = linear_sum_assignment.SimpleLinearSumAssignment()
    
    for i in range(size):
        for j in range(size):
            assignment.add_arc_with_cost(i, j, int(solve_matrix[i][j]))
    
    status = assignment.solve()
    
    if status != assignment.OPTIMAL:
        return {'status': 'NO_SOLUTION', 'assignments': []}
    
    # Extract assignments
    assignments = []
    for i in range(size):
        j = assignment.right_mate(i)
        if i < num_workers and j < num_tasks:
            assignments.append({
                'worker': workers[i],
                'task': tasks[j],
                'cost': int(cost_matrix[i][j])
            })
    
    total_cost = sum(a['cost'] for a in assignments)
    
    return {
        'status': 'SUCCESS',
        'assignments': assignments,
        'total_cost': total_cost
    }


def create_assignment_matrix(cost_matrix: np.ndarray, workers: List[str], 
                             tasks: List[str], assignments: List[Dict]) -> str:
    """Create cost matrix heatmap with assignment highlights"""
    if len(workers) == 0 or len(tasks) == 0:
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return _fig_to_base64(fig)
    
    fig, ax = plt.subplots(figsize=(max(10, len(tasks) * 0.8), max(8, len(workers) * 0.6)))
    
    # Create heatmap
    sns.heatmap(cost_matrix, annot=True, fmt='d', cmap='YlOrRd',
                xticklabels=tasks, yticklabels=workers, ax=ax,
                cbar_kws={'label': 'Cost'})
    
    # Highlight assignments
    assigned_pairs = {(a['worker'], a['task']) for a in assignments}
    for i, worker in enumerate(workers):
        for j, task in enumerate(tasks):
            if (worker, task) in assigned_pairs:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                          edgecolor='green', linewidth=3))
    
    ax.set_xlabel('Tasks', fontsize=11)
    ax.set_ylabel('Workers', fontsize=11)
    ax.set_title('Cost Matrix with Optimal Assignments (Green Border)', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_cost_distribution(assignments: List[Dict]) -> str:
    """Create cost distribution chart"""
    if not assignments:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return _fig_to_base64(fig)
    
    labels = [f"{a['worker']}→{a['task']}" for a in assignments]
    costs = [a['cost'] for a in assignments]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = [ASSIGNMENT_COLORS[i % len(ASSIGNMENT_COLORS)] for i in range(len(assignments))]
    bars = ax.bar(labels, costs, color=colors, edgecolor='white', linewidth=1)
    
    for bar, cost in zip(bars, costs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'${cost}', ha='center', va='bottom', fontsize=9)
    
    avg_cost = np.mean(costs)
    ax.axhline(y=avg_cost, color='gray', linestyle='--', alpha=0.7, label=f'Avg: ${avg_cost:.1f}')
    
    ax.set_ylabel('Cost ($)', fontsize=11)
    ax.set_xlabel('Assignment', fontsize=11)
    ax.set_title('Assignment Costs', fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_worker_assignments(assignments: List[Dict], workers: List[str]) -> str:
    """Create worker assignment visualization"""
    if not assignments:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return _fig_to_base64(fig)
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(assignments) * 0.5)))
    
    assigned_workers = {a['worker'] for a in assignments}
    y_pos = 0
    
    for i, assignment in enumerate(assignments):
        color = ASSIGNMENT_COLORS[i % len(ASSIGNMENT_COLORS)]
        
        # Worker node
        ax.scatter(0, y_pos, s=200, c=color, zorder=3)
        ax.annotate(assignment['worker'], (0, y_pos), xytext=(-0.3, 0),
                   textcoords='offset points', ha='right', va='center', fontsize=10)
        
        # Task node
        ax.scatter(1, y_pos, s=200, c=color, marker='s', zorder=3)
        ax.annotate(assignment['task'], (1, y_pos), xytext=(0.3, 0),
                   textcoords='offset points', ha='left', va='center', fontsize=10)
        
        # Connection line
        ax.plot([0, 1], [y_pos, y_pos], color=color, linewidth=2, zorder=2)
        
        # Cost label
        ax.annotate(f"${assignment['cost']}", (0.5, y_pos), xytext=(0, 5),
                   textcoords='offset points', ha='center', va='bottom', fontsize=9)
        
        y_pos += 1
    
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, y_pos)
    ax.axis('off')
    ax.set_title('Worker-Task Assignments', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_efficiency_chart(cost_matrix: np.ndarray, assignments: List[Dict],
                            workers: List[str], tasks: List[str]) -> str:
    """Create efficiency comparison chart"""
    if not assignments or len(cost_matrix) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return _fig_to_base64(fig)
    
    # Calculate metrics
    optimal_cost = sum(a['cost'] for a in assignments)
    
    # Random assignment average (sample)
    random_costs = []
    for _ in range(100):
        perm = np.random.permutation(min(len(workers), len(tasks)))
        cost = sum(cost_matrix[i][perm[i]] for i in range(len(perm)))
        random_costs.append(cost)
    random_avg = np.mean(random_costs)
    
    # Greedy assignment
    greedy_cost = 0
    used_tasks = set()
    for i in range(len(workers)):
        min_cost = float('inf')
        min_j = -1
        for j in range(len(tasks)):
            if j not in used_tasks and cost_matrix[i][j] < min_cost:
                min_cost = cost_matrix[i][j]
                min_j = j
        if min_j >= 0:
            greedy_cost += min_cost
            used_tasks.add(min_j)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    methods = ['Optimal\n(Hungarian)', 'Greedy', 'Random\n(Avg)']
    costs = [optimal_cost, greedy_cost, random_avg]
    colors = ['#22c55e', '#f59e0b', '#ef4444']
    
    bars = ax.bar(methods, costs, color=colors, edgecolor='white', linewidth=2)
    
    for bar, cost in zip(bars, costs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'${cost:.0f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add savings annotation
    savings = ((random_avg - optimal_cost) / random_avg) * 100
    ax.annotate(f'{savings:.0f}% savings\nvs random', xy=(0, optimal_cost),
               xytext=(0.5, optimal_cost + (random_avg - optimal_cost) / 2),
               ha='center', fontsize=10, color='green')
    
    ax.set_ylabel('Total Cost ($)', fontsize=11)
    ax.set_title('Optimization Method Comparison', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_key_insights(assignments: List[Dict], metrics: Dict,
                          num_workers: int, num_tasks: int) -> List[Dict]:
    """Generate key insights"""
    insights = []
    
    # Efficiency insight
    if metrics['efficiency_score'] >= 80:
        insights.append({
            'title': f"High Efficiency: {metrics['efficiency_score']:.1f}%",
            'description': 'Optimal assignments achieve excellent cost efficiency.',
            'status': 'positive'
        })
    elif metrics['efficiency_score'] >= 60:
        insights.append({
            'title': f"Good Efficiency: {metrics['efficiency_score']:.1f}%",
            'description': 'Reasonable optimization achieved.',
            'status': 'neutral'
        })
    else:
        insights.append({
            'title': f"Low Efficiency: {metrics['efficiency_score']:.1f}%",
            'description': 'Consider reviewing cost structure or constraints.',
            'status': 'warning'
        })
    
    # Balance insight
    if metrics['cost_variance'] < 50:
        insights.append({
            'title': 'Well-Balanced Assignments',
            'description': f"Cost variance is low (${metrics['cost_variance']:.1f}). Workload is evenly distributed.",
            'status': 'positive'
        })
    else:
        insights.append({
            'title': 'Uneven Cost Distribution',
            'description': f"High cost variance (${metrics['cost_variance']:.1f}). Some assignments are significantly more expensive.",
            'status': 'neutral'
        })
    
    # Coverage insight
    num_assigned = len(assignments)
    if num_assigned == min(num_workers, num_tasks):
        insights.append({
            'title': 'Full Assignment Coverage',
            'description': f'All {num_assigned} possible assignments have been made.',
            'status': 'positive'
        })
    else:
        insights.append({
            'title': f'Partial Assignment: {num_assigned}/{min(num_workers, num_tasks)}',
            'description': 'Some workers or tasks remain unassigned.',
            'status': 'warning'
        })
    
    return insights


@router.post("/assignment")
async def run_assignment(request: AssignmentRequest) -> Dict[str, Any]:
    """Run assignment optimization"""
    try:
        start_time = time.time()
        df = pd.DataFrame(request.data)
        
        # Validate columns
        if request.worker_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Worker column '{request.worker_col}' not found")
        if request.task_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Task column '{request.task_col}' not found")
        if request.cost_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Cost column '{request.cost_col}' not found")
        
        # Get unique workers and tasks
        workers = sorted(df[request.worker_col].unique().astype(str).tolist())
        tasks = sorted(df[request.task_col].unique().astype(str).tolist())
        
        if len(workers) < 2 or len(tasks) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 workers and 2 tasks")
        
        # Build cost matrix
        worker_idx = {w: i for i, w in enumerate(workers)}
        task_idx = {t: i for i, t in enumerate(tasks)}
        
        cost_matrix = np.full((len(workers), len(tasks)), 999999)
        
        for _, row in df.iterrows():
            w = str(row[request.worker_col])
            t = str(row[request.task_col])
            cost = float(row[request.cost_col])
            
            if w in worker_idx and t in task_idx:
                cost_matrix[worker_idx[w]][task_idx[t]] = int(cost)
        
        # Solve
        solution = solve_assignment(cost_matrix, workers, tasks, request.problem_type)
        
        solve_time_ms = int((time.time() - start_time) * 1000)
        
        if solution['status'] != 'SUCCESS' or not solution['assignments']:
            raise HTTPException(status_code=400, detail="No feasible assignment found")
        
        assignments = solution['assignments']
        total_cost = solution['total_cost']
        
        # Calculate metrics
        costs = [a['cost'] for a in assignments]
        
        # Calculate efficiency (compared to worst possible)
        max_possible_cost = np.max(cost_matrix[cost_matrix < 999999]) * len(assignments) if len(assignments) > 0 else 1
        min_possible_cost = np.min(cost_matrix) * len(assignments) if len(assignments) > 0 else 0
        
        if max_possible_cost > min_possible_cost:
            efficiency = (1 - (total_cost - min_possible_cost) / (max_possible_cost - min_possible_cost)) * 100
        else:
            efficiency = 100
        
        metrics = {
            'avg_cost': np.mean(costs) if costs else 0,
            'min_cost': min(costs) if costs else 0,
            'max_cost': max(costs) if costs else 0,
            'cost_variance': np.var(costs) if costs else 0,
            'efficiency_score': efficiency
        }
        
        # Find unassigned
        assigned_workers = {a['worker'] for a in assignments}
        assigned_tasks = {a['task'] for a in assignments}
        unassigned_workers = [w for w in workers if w not in assigned_workers]
        unassigned_tasks = [t for t in tasks if t not in assigned_tasks]
        
        # Create visualizations
        visualizations = {
            'assignment_matrix': create_assignment_matrix(cost_matrix, workers, tasks, assignments),
            'cost_distribution': create_cost_distribution(assignments),
            'worker_assignments': create_worker_assignments(assignments, workers),
            'efficiency_chart': create_efficiency_chart(cost_matrix, assignments, workers, tasks)
        }
        
        # Generate insights
        key_insights = generate_key_insights(assignments, metrics, len(workers), len(tasks))
        
        # Prepare results
        results = {
            'assignments': [{k: _to_native_type(v) for k, v in a.items()} for a in assignments],
            'total_cost': total_cost,
            'num_assigned': len(assignments),
            'num_workers': len(workers),
            'num_tasks': len(tasks),
            'unassigned_workers': unassigned_workers,
            'unassigned_tasks': unassigned_tasks,
            'metrics': {k: _to_native_type(v) for k, v in metrics.items()}
        }
        
        summary = {
            'problem_type': request.problem_type,
            'num_workers': len(workers),
            'num_tasks': len(tasks),
            'total_cost': total_cost,
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
        raise HTTPException(status_code=500, detail=f"Assignment optimization failed: {str(e)}")
