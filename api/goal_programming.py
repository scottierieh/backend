"""
Goal Programming Solver Router for FastAPI
Solve and visualize multi-objective optimization with prioritized goals
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import io
import base64
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class Goal(BaseModel):
    coeffs: List[float] = Field(description="Coefficients for each variable")
    type: str = Field(default="==", description="Goal type: '<=', '>=', or '=='")
    target: float = Field(description="Target value for the goal")
    priority: int = Field(default=1, description="Priority level (1 = highest)")


class Constraint(BaseModel):
    coeffs: List[float] = Field(description="Coefficients for each variable")
    type: str = Field(default="<=", description="Constraint type: '<=', '>=', or '=='")
    rhs: float = Field(description="Right-hand side value")


class GoalProgrammingRequest(BaseModel):
    """Goal Programming solver request parameters"""
    goals: List[Goal] = Field(
        default=[
            Goal(coeffs=[1, 0], type="==", target=80, priority=1),
            Goal(coeffs=[0, 1], type=">=", target=60, priority=2)
        ],
        description="List of goals with priorities"
    )
    constraints: List[Constraint] = Field(
        default=[Constraint(coeffs=[1, 1], type="<=", rhs=100)],
        description="Hard constraints that must be satisfied"
    )


def _to_native_type(obj):
    """Convert numpy types to JSON-serializable Python types"""
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


def solve_goal_programming(
    goals: List[Goal],
    constraints: List[Constraint]
) -> Dict[str, Any]:
    """
    Solve Goal Programming using preemptive (lexicographic) method.
    Solves goals in order of priority, fixing higher priority achievements.
    """
    
    n_vars = len(goals[0].coeffs)
    n_goals = len(goals)
    
    # Sort goals by priority
    sorted_goals = sorted(enumerate(goals), key=lambda x: x[1].priority)
    priority_groups = {}
    for idx, goal in sorted_goals:
        p = goal.priority
        if p not in priority_groups:
            priority_groups[p] = []
        priority_groups[p].append((idx, goal))
    
    # Variables: x1...xn, d1+, d1-, d2+, d2-, ...
    # Total vars: n_vars + 2 * n_goals (positive and negative deviations)
    total_vars = n_vars + 2 * n_goals
    
    # Build constraint matrix for hard constraints
    A_hard_ub = []
    b_hard_ub = []
    A_hard_eq = []
    b_hard_eq = []
    
    for c in constraints:
        row = c.coeffs + [0] * (2 * n_goals)  # No deviation variables in hard constraints
        if c.type == "<=":
            A_hard_ub.append(row)
            b_hard_ub.append(c.rhs)
        elif c.type == ">=":
            A_hard_ub.append([-x for x in row])
            b_hard_ub.append(-c.rhs)
        else:  # ==
            A_hard_eq.append(row)
            b_hard_eq.append(c.rhs)
    
    # Build goal constraint rows
    # For goal i: sum(coeffs * x) + d_i- - d_i+ = target
    A_goals = []
    b_goals = []
    
    for i, goal in enumerate(goals):
        row = [0] * total_vars
        for j, coef in enumerate(goal.coeffs):
            row[j] = coef
        # d_i- at position n_vars + 2*i
        # d_i+ at position n_vars + 2*i + 1
        row[n_vars + 2*i] = 1      # d_i- (negative deviation, underachievement)
        row[n_vars + 2*i + 1] = -1  # d_i+ (positive deviation, overachievement)
        A_goals.append(row)
        b_goals.append(goal.target)
    
    # Combine all equality constraints
    A_eq = A_hard_eq + A_goals
    b_eq = b_hard_eq + b_goals
    
    A_ub = A_hard_ub if A_hard_ub else None
    b_ub = b_hard_ub if b_hard_ub else None
    
    # Bounds: x >= 0, d >= 0
    bounds = [(0, None) for _ in range(total_vars)]
    
    # Solve iteratively by priority
    fixed_constraints_A = []
    fixed_constraints_b = []
    
    achieved_values = {}
    final_solution = None
    
    for priority in sorted(priority_groups.keys()):
        goal_indices = [idx for idx, _ in priority_groups[priority]]
        
        # Objective: minimize sum of unwanted deviations for this priority
        c_obj = [0] * total_vars
        for idx, goal in priority_groups[priority]:
            if goal.type == ">=":
                # Want to achieve at least target, minimize d- (underachievement)
                c_obj[n_vars + 2*idx] = 1
            elif goal.type == "<=":
                # Want to achieve at most target, minimize d+ (overachievement)
                c_obj[n_vars + 2*idx + 1] = 1
            else:  # ==
                # Want exact, minimize both
                c_obj[n_vars + 2*idx] = 1
                c_obj[n_vars + 2*idx + 1] = 1
        
        # Combine with fixed constraints from higher priorities
        A_ub_combined = (A_ub or []) + fixed_constraints_A if (A_ub or fixed_constraints_A) else None
        b_ub_combined = (b_ub or []) + fixed_constraints_b if (b_ub or fixed_constraints_b) else None
        
        if A_ub_combined:
            A_ub_combined = np.array(A_ub_combined)
            b_ub_combined = np.array(b_ub_combined)
        
        result = linprog(
            c_obj,
            A_ub=A_ub_combined,
            b_ub=b_ub_combined,
            A_eq=np.array(A_eq) if A_eq else None,
            b_eq=np.array(b_eq) if b_eq else None,
            bounds=bounds,
            method='highs'
        )
        
        if not result.success:
            return {
                "success": False,
                "solution": None,
                "deviations": {},
                "goal_achievements": {},
                "message": f"Failed at priority {priority}: {result.message}"
            }
        
        final_solution = result.x
        
        # Fix the achieved deviation for this priority level
        achieved_deviation = result.fun
        
        # Add constraint to maintain this achievement for next priority
        if achieved_deviation < 1e-6:
            # Achieved perfectly, fix deviations to 0
            for idx, goal in priority_groups[priority]:
                if goal.type == ">=":
                    row = [0] * total_vars
                    row[n_vars + 2*idx] = 1
                    fixed_constraints_A.append(row)
                    fixed_constraints_b.append(1e-6)
                elif goal.type == "<=":
                    row = [0] * total_vars
                    row[n_vars + 2*idx + 1] = 1
                    fixed_constraints_A.append(row)
                    fixed_constraints_b.append(1e-6)
                else:
                    row1 = [0] * total_vars
                    row1[n_vars + 2*idx] = 1
                    fixed_constraints_A.append(row1)
                    fixed_constraints_b.append(1e-6)
                    row2 = [0] * total_vars
                    row2[n_vars + 2*idx + 1] = 1
                    fixed_constraints_A.append(row2)
                    fixed_constraints_b.append(1e-6)
        
        achieved_values[priority] = achieved_deviation
    
    # Extract results
    solution = [_to_native_type(final_solution[i]) for i in range(n_vars)]
    
    deviations = {}
    goal_achievements = {}
    for i, goal in enumerate(goals):
        d_minus = _to_native_type(final_solution[n_vars + 2*i])
        d_plus = _to_native_type(final_solution[n_vars + 2*i + 1])
        deviations[f"Goal {i+1} (d-)"] = d_minus
        deviations[f"Goal {i+1} (d+)"] = d_plus
        
        achieved = sum(goal.coeffs[j] * solution[j] for j in range(n_vars))
        goal_achievements[f"Goal {i+1}"] = {
            "target": goal.target,
            "achieved": _to_native_type(achieved),
            "type": goal.type,
            "priority": goal.priority,
            "deviation_minus": d_minus,
            "deviation_plus": d_plus,
            "satisfied": (
                (goal.type == ">=" and achieved >= goal.target - 1e-6) or
                (goal.type == "<=" and achieved <= goal.target + 1e-6) or
                (goal.type == "==" and abs(achieved - goal.target) < 1e-6)
            )
        }
    
    return {
        "success": True,
        "solution": solution,
        "deviations": deviations,
        "goal_achievements": goal_achievements,
        "priority_achievements": {str(k): _to_native_type(v) for k, v in achieved_values.items()},
        "message": "Optimal solution found"
    }


def generate_goal_achievement_plot(goal_achievements: Dict) -> str:
    """Generate bar chart showing goal achievements vs targets"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    goals = list(goal_achievements.keys())
    targets = [g["target"] for g in goal_achievements.values()]
    achieved = [g["achieved"] for g in goal_achievements.values()]
    satisfied = [g["satisfied"] for g in goal_achievements.values()]
    priorities = [g["priority"] for g in goal_achievements.values()]
    
    x = np.arange(len(goals))
    width = 0.35
    
    colors_target = ['#BBDEFB'] * len(goals)
    colors_achieved = ['#4CAF50' if s else '#F44336' for s in satisfied]
    
    bars1 = ax.bar(x - width/2, targets, width, label='Target', color=colors_target, edgecolor='black')
    bars2 = ax.bar(x + width/2, achieved, width, label='Achieved', color=colors_achieved, edgecolor='black')
    
    # Add priority labels
    for i, (bar, p) in enumerate(zip(bars2, priorities)):
        ax.annotate(f'P{p}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    
    ax.set_xlabel('Goals')
    ax.set_ylabel('Value')
    ax.set_title('Goal Achievement Analysis')
    ax.set_xticks(x)
    ax.set_xticklabels(goals)
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_deviation_plot(deviations: Dict) -> str:
    """Generate bar chart showing deviations"""
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Group by goal
    n_goals = len(deviations) // 2
    goals = [f'Goal {i+1}' for i in range(n_goals)]
    d_minus = [deviations.get(f"Goal {i+1} (d-)", 0) for i in range(n_goals)]
    d_plus = [deviations.get(f"Goal {i+1} (d+)", 0) for i in range(n_goals)]
    
    x = np.arange(len(goals))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, d_minus, width, label='d⁻ (Under)', color='#F44336', edgecolor='black')
    bars2 = ax.bar(x + width/2, d_plus, width, label='d⁺ (Over)', color='#4CAF50', edgecolor='black')
    
    ax.set_xlabel('Goals')
    ax.set_ylabel('Deviation')
    ax.set_title('Goal Deviations')
    ax.set_xticks(x)
    ax.set_xticklabels(goals)
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_feasible_region_plot(
    goals: List[Goal],
    constraints: List[Constraint],
    solution: List[float]
) -> Optional[str]:
    """Generate 2D feasible region plot for 2-variable problems"""
    
    n_vars = len(goals[0].coeffs)
    if n_vars != 2:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Determine plot range
    max_val = max(
        max(c.rhs for c in constraints) if constraints else 10,
        max(g.target for g in goals)
    ) * 1.3
    
    x = np.linspace(0, max_val, 400)
    
    # Plot hard constraints
    colors_constraint = ['#4299e1', '#38b2ac', '#9f7aea']
    for i, c in enumerate(constraints):
        a1, a2 = c.coeffs
        if abs(a2) > 1e-10:
            y_line = (c.rhs - a1 * x) / a2
            ax.plot(x, y_line, '-', color=colors_constraint[i % len(colors_constraint)], 
                   linewidth=2, label=f'Constraint {i+1}')
        elif abs(a1) > 1e-10:
            ax.axvline(x=c.rhs/a1, color=colors_constraint[i % len(colors_constraint)], 
                      linewidth=2, label=f'Constraint {i+1}')
    
    # Plot goal lines
    colors_goal = ['#ed8936', '#e53e3e', '#48bb78', '#667eea']
    for i, g in enumerate(goals):
        a1, a2 = g.coeffs
        if abs(a2) > 1e-10:
            y_line = (g.target - a1 * x) / a2
            ax.plot(x, y_line, '--', color=colors_goal[i % len(colors_goal)], 
                   linewidth=2, alpha=0.7, label=f'Goal {i+1} (P{g.priority})')
        elif abs(a1) > 1e-10:
            ax.axvline(x=g.target/a1, color=colors_goal[i % len(colors_goal)], 
                      linewidth=2, linestyle='--', alpha=0.7, label=f'Goal {i+1} (P{g.priority})')
    
    # Plot solution
    ax.scatter([solution[0]], [solution[1]], color='red', s=200, marker='*',
              zorder=5, edgecolors='black', linewidths=1.5,
              label=f'Solution ({solution[0]:.2f}, {solution[1]:.2f})')
    
    ax.axhline(y=0, color='black', linewidth=1)
    ax.axvline(x=0, color='black', linewidth=1)
    
    ax.set_xlim(-0.5, max_val)
    ax.set_ylim(-0.5, max_val)
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title('Goal Programming - Solution Space', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_interpretation(result: Dict, params: Dict) -> Dict[str, Any]:
    """Generate interpretation of goal programming results"""
    key_insights = []
    
    if result['success']:
        goal_achievements = result['goal_achievements']
        n_satisfied = sum(1 for g in goal_achievements.values() if g['satisfied'])
        n_total = len(goal_achievements)
        
        key_insights.append({
            'title': 'Solution Found',
            'description': f"Achieved {n_satisfied} out of {n_total} goals.",
            'status': 'positive' if n_satisfied == n_total else 'neutral'
        })
        
        # Priority analysis
        priorities = sorted(set(g['priority'] for g in goal_achievements.values()))
        for p in priorities:
            p_goals = [name for name, g in goal_achievements.items() if g['priority'] == p]
            p_satisfied = sum(1 for name in p_goals if goal_achievements[name]['satisfied'])
            key_insights.append({
                'title': f'Priority {p} Goals',
                'description': f"{p_satisfied}/{len(p_goals)} satisfied: {', '.join(p_goals)}",
                'status': 'positive' if p_satisfied == len(p_goals) else 'warning'
            })
        
        # Solution summary
        sol_str = ', '.join([f'x{i+1}={v:.4f}' for i, v in enumerate(result['solution'])])
        key_insights.append({
            'title': 'Optimal Solution',
            'description': f"Decision variables: {sol_str}",
            'status': 'neutral'
        })
    else:
        key_insights.append({
            'title': 'No Solution Found',
            'description': result['message'],
            'status': 'warning'
        })
    
    # Recommendations
    recommendations = []
    if result['success']:
        unsatisfied = [name for name, g in result['goal_achievements'].items() if not g['satisfied']]
        if unsatisfied:
            recommendations.append(f"Goals not fully achieved: {', '.join(unsatisfied)}. Consider relaxing targets or adjusting priorities.")
        recommendations.append("Higher priority goals are satisfied first; lower priority goals may be compromised.")
    else:
        recommendations.append("Check if hard constraints are feasible.")
        recommendations.append("Try relaxing some goal targets.")
    
    return {
        'key_insights': key_insights,
        'recommendations': recommendations
    }


@router.post("/goal-programming")
async def solve_goal_programming_problem(request: GoalProgrammingRequest) -> Dict[str, Any]:
    """
    Solve a Goal Programming problem using preemptive (lexicographic) method.
    
    Goal Programming optimizes multiple objectives by minimizing deviations
    from target values, respecting priority ordering.
    
    For each goal: sum(coeffs * x) + d⁻ - d⁺ = target
    where d⁻ is underachievement and d⁺ is overachievement.
    """
    try:
        # Validate inputs
        if not request.goals:
            raise HTTPException(status_code=400, detail="At least one goal is required")
        
        n_vars = len(request.goals[0].coeffs)
        
        for i, goal in enumerate(request.goals):
            if len(goal.coeffs) != n_vars:
                raise HTTPException(status_code=400, detail=f"Goal {i+1} has inconsistent number of coefficients")
            if goal.type not in ["<=", ">=", "=="]:
                raise HTTPException(status_code=400, detail=f"Invalid goal type: {goal.type}")
        
        for i, c in enumerate(request.constraints):
            if len(c.coeffs) != n_vars:
                raise HTTPException(status_code=400, detail=f"Constraint {i+1} has inconsistent number of coefficients")
            if c.type not in ["<=", ">=", "=="]:
                raise HTTPException(status_code=400, detail=f"Invalid constraint type: {c.type}")
        
        # Solve
        result = solve_goal_programming(request.goals, request.constraints)
        
        # Generate plots
        plots = {}
        
        if result['success'] and result['solution']:
            plots['achievement'] = generate_goal_achievement_plot(result['goal_achievements'])
            plots['deviations'] = generate_deviation_plot(result['deviations'])
            
            if n_vars == 2:
                feasible_plot = generate_feasible_region_plot(
                    request.goals, request.constraints, result['solution']
                )
                if feasible_plot:
                    plots['feasible_region'] = feasible_plot
        
        # Generate interpretation
        params = {
            'goals': [g.dict() for g in request.goals],
            'constraints': [c.dict() for c in request.constraints]
        }
        interpretation = generate_interpretation(result, params)
        
        return {
            'success': result['success'],
            'message': result['message'],
            'solution': result['solution'],
            'deviations': result['deviations'],
            'goal_achievements': result['goal_achievements'],
            'priority_achievements': result.get('priority_achievements', {}),
            'problem': {
                'n_variables': n_vars,
                'n_goals': len(request.goals),
                'n_constraints': len(request.constraints),
                'priorities': sorted(set(g.priority for g in request.goals))
            },
            'plots': plots,
            'interpretation': interpretation
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Goal programming solver failed: {str(e)}")
