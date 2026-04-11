"""
Job Shop Scheduling Router for FastAPI
Using Google OR-Tools CP-SAT solver
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
from collections import defaultdict

from ortools.sat.python import cp_model

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class SchedulingRequest(BaseModel):
    data: List[Dict[str, Any]]
    job_col: str
    machine_col: str
    processing_time_col: str
    task_order_col: Optional[str] = None
    priority_col: Optional[str] = None
    due_date_col: Optional[str] = None
    problem_type: Literal["job_shop", "flow_shop", "flexible_job_shop"] = "job_shop"
    objective: Literal["makespan", "flow_time", "tardiness"] = "makespan"
    max_time_seconds: int = 30


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


MACHINE_COLORS = [
    '#3b82f6', '#ef4444', '#22c55e', '#f59e0b', '#8b5cf6',
    '#ec4899', '#06b6d4', '#84cc16', '#f97316', '#6366f1'
]


def prepare_job_data(df: pd.DataFrame, job_col: str, machine_col: str, 
                     processing_time_col: str, task_order_col: Optional[str]) -> Dict:
    """Prepare job shop data structure"""
    jobs_data = defaultdict(list)
    
    for _, row in df.iterrows():
        job_id = str(row[job_col])
        machine_id = str(row[machine_col])
        processing_time = int(row[processing_time_col])
        task_order = int(row[task_order_col]) if task_order_col and pd.notna(row[task_order_col]) else len(jobs_data[job_id]) + 1
        
        jobs_data[job_id].append({
            'machine': machine_id,
            'duration': processing_time,
            'order': task_order
        })
    
    # Sort tasks by order within each job
    for job_id in jobs_data:
        jobs_data[job_id] = sorted(jobs_data[job_id], key=lambda x: x['order'])
    
    return dict(jobs_data)


def solve_job_shop(jobs_data: Dict, machines: List[str], objective: str, 
                   max_time_seconds: int) -> Dict:
    """Solve job shop scheduling using CP-SAT"""
    model = cp_model.CpModel()
    
    # Calculate horizon
    horizon = sum(
        task['duration'] 
        for job_tasks in jobs_data.values() 
        for task in job_tasks
    )
    
    # Create variables
    all_tasks = {}  # (job_id, task_idx) -> {start, end, interval}
    machine_to_intervals = defaultdict(list)
    
    job_ids = list(jobs_data.keys())
    machine_to_idx = {m: i for i, m in enumerate(machines)}
    
    for job_id, tasks in jobs_data.items():
        for task_idx, task in enumerate(tasks):
            suffix = f'_{job_id}_{task_idx}'
            start_var = model.NewIntVar(0, horizon, f'start{suffix}')
            end_var = model.NewIntVar(0, horizon, f'end{suffix}')
            interval_var = model.NewIntervalVar(
                start_var, task['duration'], end_var, f'interval{suffix}'
            )
            
            all_tasks[(job_id, task_idx)] = {
                'start': start_var,
                'end': end_var,
                'interval': interval_var,
                'duration': task['duration'],
                'machine': task['machine']
            }
            
            machine_to_intervals[task['machine']].append(interval_var)
    
    # No overlap on machines
    for machine in machines:
        if machine in machine_to_intervals:
            model.AddNoOverlap(machine_to_intervals[machine])
    
    # Precedence within jobs
    for job_id, tasks in jobs_data.items():
        for task_idx in range(len(tasks) - 1):
            model.Add(
                all_tasks[(job_id, task_idx + 1)]['start'] >= 
                all_tasks[(job_id, task_idx)]['end']
            )
    
    # Objective
    if objective == 'makespan':
        makespan = model.NewIntVar(0, horizon, 'makespan')
        model.AddMaxEquality(makespan, [
            all_tasks[(job_id, len(tasks) - 1)]['end']
            for job_id, tasks in jobs_data.items()
        ])
        model.Minimize(makespan)
    elif objective == 'flow_time':
        flow_times = []
        for job_id, tasks in jobs_data.items():
            flow_times.append(all_tasks[(job_id, len(tasks) - 1)]['end'])
        model.Minimize(sum(flow_times))
    
    # Solve
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time_seconds
    status = solver.Solve(model)
    
    if status not in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        return {'status': 'NO_SOLUTION', 'task_assignments': []}
    
    # Extract solution
    task_assignments = []
    for (job_id, task_idx), task_info in all_tasks.items():
        task_assignments.append({
            'job_id': job_id,
            'task_id': task_idx + 1,
            'machine_id': task_info['machine'],
            'start_time': solver.Value(task_info['start']),
            'end_time': solver.Value(task_info['end']),
            'duration': task_info['duration']
        })
    
    # Sort by start time
    task_assignments.sort(key=lambda x: (x['start_time'], x['job_id']))
    
    makespan = max(t['end_time'] for t in task_assignments)
    
    return {
        'status': 'SUCCESS',
        'task_assignments': task_assignments,
        'makespan': makespan,
        'solve_time': solver.WallTime()
    }


def calculate_machine_schedules(task_assignments: List[Dict], machines: List[str], 
                                makespan: int) -> List[Dict]:
    """Calculate schedule metrics per machine"""
    machine_schedules = []
    
    for machine in machines:
        machine_tasks = [t for t in task_assignments if t['machine_id'] == machine]
        machine_tasks.sort(key=lambda x: x['start_time'])
        
        total_working = sum(t['duration'] for t in machine_tasks)
        idle_time = makespan - total_working
        utilization = (total_working / makespan * 100) if makespan > 0 else 0
        
        machine_schedules.append({
            'machine_id': machine,
            'tasks': machine_tasks,
            'total_working_time': total_working,
            'idle_time': idle_time,
            'utilization': utilization
        })
    
    return machine_schedules


def calculate_job_schedules(task_assignments: List[Dict], jobs: List[str]) -> List[Dict]:
    """Calculate schedule metrics per job"""
    job_schedules = []
    
    for job in jobs:
        job_tasks = [t for t in task_assignments if t['job_id'] == job]
        job_tasks.sort(key=lambda x: x['task_id'])
        
        if job_tasks:
            start_time = min(t['start_time'] for t in job_tasks)
            end_time = max(t['end_time'] for t in job_tasks)
            flow_time = end_time - start_time
            
            job_schedules.append({
                'job_id': job,
                'tasks': job_tasks,
                'start_time': start_time,
                'end_time': end_time,
                'flow_time': flow_time
            })
    
    return job_schedules


def create_gantt_chart(machine_schedules: List[Dict], makespan: int) -> str:
    """Create Gantt chart visualization"""
    if not machine_schedules:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return _fig_to_base64(fig)
    
    fig, ax = plt.subplots(figsize=(14, max(6, len(machine_schedules) * 0.8)))
    
    # Get unique jobs for coloring
    all_jobs = set()
    for machine in machine_schedules:
        for task in machine['tasks']:
            all_jobs.add(task['job_id'])
    jobs = sorted(list(all_jobs))
    job_colors = {job: MACHINE_COLORS[i % len(MACHINE_COLORS)] for i, job in enumerate(jobs)}
    
    for idx, machine in enumerate(machine_schedules):
        y = len(machine_schedules) - idx - 1
        
        for task in machine['tasks']:
            color = job_colors.get(task['job_id'], '#888888')
            ax.barh(y, task['duration'], left=task['start_time'], height=0.6,
                   color=color, edgecolor='white', linewidth=1)
            
            # Add job label if bar is wide enough
            if task['duration'] > makespan * 0.05:
                ax.text(task['start_time'] + task['duration']/2, y,
                       task['job_id'], ha='center', va='center',
                       fontsize=8, color='white', fontweight='bold')
    
    ax.set_yticks(range(len(machine_schedules)))
    ax.set_yticklabels([m['machine_id'] for m in reversed(machine_schedules)])
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Machine', fontsize=11)
    ax.set_title('Job Shop Schedule (Gantt Chart)', fontsize=14, fontweight='bold')
    ax.set_xlim(0, makespan * 1.05)
    ax.grid(axis='x', alpha=0.3)
    
    # Legend
    legend_patches = [mpatches.Patch(color=job_colors[job], label=job) for job in jobs[:10]]
    if len(jobs) > 10:
        legend_patches.append(mpatches.Patch(color='gray', label=f'+{len(jobs)-10} more'))
    ax.legend(handles=legend_patches, loc='upper right', fontsize=8, ncol=2)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_machine_utilization_chart(machine_schedules: List[Dict]) -> str:
    """Create machine utilization bar chart"""
    if not machine_schedules:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return _fig_to_base64(fig)
    
    machines = [m['machine_id'] for m in machine_schedules]
    utilizations = [m['utilization'] for m in machine_schedules]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#22c55e' if u >= 80 else '#f59e0b' if u >= 60 else '#ef4444' for u in utilizations]
    bars = ax.bar(machines, utilizations, color=colors, edgecolor='white', linewidth=1)
    
    for bar, util in zip(bars, utilizations):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{util:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    avg_util = np.mean(utilizations)
    ax.axhline(y=avg_util, color='blue', linestyle='--', alpha=0.7, label=f'Avg: {avg_util:.1f}%')
    ax.axhline(y=80, color='green', linestyle=':', alpha=0.5, label='Target: 80%')
    
    ax.set_ylabel('Utilization (%)', fontsize=11)
    ax.set_xlabel('Machine', fontsize=11)
    ax.set_title('Machine Utilization', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(loc='lower right')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_job_flow_times_chart(job_schedules: List[Dict]) -> str:
    """Create job flow times chart"""
    if not job_schedules:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return _fig_to_base64(fig)
    
    jobs = [j['job_id'] for j in job_schedules]
    flow_times = [j['flow_time'] for j in job_schedules]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = [MACHINE_COLORS[i % len(MACHINE_COLORS)] for i in range(len(jobs))]
    bars = ax.bar(jobs, flow_times, color=colors, edgecolor='white', linewidth=1)
    
    for bar, ft in zip(bars, flow_times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{ft}', ha='center', va='bottom', fontsize=9)
    
    avg_flow = np.mean(flow_times)
    ax.axhline(y=avg_flow, color='gray', linestyle='--', alpha=0.7, label=f'Avg: {avg_flow:.1f}')
    
    ax.set_ylabel('Flow Time', fontsize=11)
    ax.set_xlabel('Job', fontsize=11)
    ax.set_title('Job Flow Times', fontsize=14, fontweight='bold')
    ax.legend()
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_timeline_chart(machine_schedules: List[Dict], makespan: int) -> str:
    """Create timeline showing machine activity"""
    if not machine_schedules:
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
        return _fig_to_base64(fig)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for idx, machine in enumerate(machine_schedules):
        color = MACHINE_COLORS[idx % len(MACHINE_COLORS)]
        
        # Working periods
        for task in machine['tasks']:
            ax.plot([task['start_time'], task['end_time']], [idx, idx],
                   color=color, linewidth=8, solid_capstyle='butt')
        
        # Idle indicator
        ax.scatter([0], [idx], color=color, s=100, marker='s', zorder=5)
    
    ax.set_yticks(range(len(machine_schedules)))
    ax.set_yticklabels([m['machine_id'] for m in machine_schedules])
    ax.set_xlabel('Time', fontsize=11)
    ax.set_title('Machine Timeline', fontsize=14, fontweight='bold')
    ax.set_xlim(-5, makespan + 5)
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_key_insights(machine_schedules: List[Dict], job_schedules: List[Dict],
                          makespan: int, metrics: Dict) -> List[Dict]:
    """Generate key insights"""
    insights = []
    
    # Utilization insight
    avg_util = metrics['avg_utilization']
    if avg_util >= 80:
        insights.append({
            'title': f'High Utilization: {avg_util:.1f}%',
            'description': 'Machines are being used efficiently with minimal idle time.',
            'status': 'positive'
        })
    elif avg_util >= 60:
        insights.append({
            'title': f'Moderate Utilization: {avg_util:.1f}%',
            'description': 'Some room for improvement. Consider adding jobs or reducing machines.',
            'status': 'neutral'
        })
    else:
        insights.append({
            'title': f'Low Utilization: {avg_util:.1f}%',
            'description': 'Significant idle time detected. Review capacity planning.',
            'status': 'warning'
        })
    
    # Bottleneck insight
    if machine_schedules:
        max_util_machine = max(machine_schedules, key=lambda x: x['utilization'])
        min_util_machine = min(machine_schedules, key=lambda x: x['utilization'])
        
        if max_util_machine['utilization'] - min_util_machine['utilization'] > 20:
            insights.append({
                'title': f'Bottleneck: {max_util_machine["machine_id"]}',
                'description': f'{max_util_machine["utilization"]:.1f}% utilization vs {min_util_machine["utilization"]:.1f}% on {min_util_machine["machine_id"]}.',
                'status': 'warning'
            })
    
    # Makespan insight
    if job_schedules:
        total_processing = sum(j['flow_time'] for j in job_schedules)
        efficiency = (total_processing / (makespan * len(machine_schedules))) * 100 if makespan > 0 else 0
        insights.append({
            'title': f'Makespan: {makespan} minutes',
            'description': f'All jobs complete in {makespan} minutes with {efficiency:.1f}% schedule efficiency.',
            'status': 'neutral'
        })
    
    return insights


@router.post("/scheduling")
async def run_scheduling_optimization(request: SchedulingRequest) -> Dict[str, Any]:
    """Run job shop scheduling optimization"""
    try:
        start_time = time.time()
        df = pd.DataFrame(request.data)
        
        # Validate columns
        if request.job_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Job column '{request.job_col}' not found")
        if request.machine_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Machine column '{request.machine_col}' not found")
        if request.processing_time_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Processing time column '{request.processing_time_col}' not found")
        
        # Get unique machines and jobs
        machines = sorted(df[request.machine_col].unique().astype(str).tolist())
        jobs = sorted(df[request.job_col].unique().astype(str).tolist())
        
        if len(jobs) < 2 or len(machines) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 jobs and 2 machines")
        
        # Prepare job data
        jobs_data = prepare_job_data(df, request.job_col, request.machine_col,
                                     request.processing_time_col, request.task_order_col)
        
        # Solve
        solution = solve_job_shop(jobs_data, machines, request.objective, request.max_time_seconds)
        
        solve_time_ms = int((time.time() - start_time) * 1000)
        
        if solution['status'] != 'SUCCESS' or not solution['task_assignments']:
            raise HTTPException(status_code=400, detail="No feasible schedule found")
        
        task_assignments = solution['task_assignments']
        makespan = solution['makespan']
        
        # Calculate schedules
        machine_schedules = calculate_machine_schedules(task_assignments, machines, makespan)
        job_schedules = calculate_job_schedules(task_assignments, jobs)
        
        # Calculate metrics
        utilizations = [m['utilization'] for m in machine_schedules]
        flow_times = [j['flow_time'] for j in job_schedules]
        
        metrics = {
            'total_jobs': len(jobs),
            'total_machines': len(machines),
            'total_tasks': len(task_assignments),
            'avg_utilization': np.mean(utilizations) if utilizations else 0,
            'max_utilization': max(utilizations) if utilizations else 0,
            'min_utilization': min(utilizations) if utilizations else 0,
            'idle_time_total': sum(m['idle_time'] for m in machine_schedules)
        }
        
        total_flow_time = sum(flow_times)
        avg_flow_time = np.mean(flow_times) if flow_times else 0
        
        # Create visualizations
        visualizations = {
            'gantt_chart': create_gantt_chart(machine_schedules, makespan),
            'machine_utilization': create_machine_utilization_chart(machine_schedules),
            'job_flow_times': create_job_flow_times_chart(job_schedules),
            'machine_timeline': create_timeline_chart(machine_schedules, makespan)
        }
        
        # Generate insights
        key_insights = generate_key_insights(machine_schedules, job_schedules, makespan, metrics)
        
        # Prepare results
        results = {
            'makespan': makespan,
            'total_flow_time': total_flow_time,
            'avg_flow_time': _to_native_type(avg_flow_time),
            'machine_schedules': [{
                'machine_id': m['machine_id'],
                'tasks': [{k: _to_native_type(v) for k, v in t.items()} for t in m['tasks']],
                'total_working_time': m['total_working_time'],
                'idle_time': m['idle_time'],
                'utilization': _to_native_type(m['utilization'])
            } for m in machine_schedules],
            'job_schedules': [{
                'job_id': j['job_id'],
                'tasks': [{k: _to_native_type(v) for k, v in t.items()} for t in j['tasks']],
                'start_time': j['start_time'],
                'end_time': j['end_time'],
                'flow_time': j['flow_time']
            } for j in job_schedules],
            'task_assignments': [{k: _to_native_type(v) for k, v in t.items()} for t in task_assignments],
            'metrics': {k: _to_native_type(v) for k, v in metrics.items()}
        }
        
        summary = {
            'problem_type': request.problem_type,
            'num_jobs': len(jobs),
            'num_machines': len(machines),
            'makespan': makespan,
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
        raise HTTPException(status_code=500, detail=f"Scheduling optimization failed: {str(e)}")
