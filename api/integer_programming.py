"""
Integer Programming Solver Router for FastAPI
Solve and visualize mixed-integer linear programming (MILP) problems
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import milp, LinearConstraint, Bounds
import io
import base64
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class IntegerProgrammingRequest(BaseModel):
    """Integer Programming solver request parameters"""
    c: List[float] = Field(
        default=[-1, -2],
        description="Objective function coefficients"
    )
    A: List[List[float]] = Field(
        default=[[2, 1], [1, 2]],
        description="Constraint matrix"
    )
    b: List[float] = Field(
        default=[20, 20],
        description="Constraint bounds"
    )
    constraint_types: List[str] = Field(
        default=["<=", "<="],
        description="Constraint types: '<=', '>=', or '=='"
    )
    objective: str = Field(
        default="maximize",
        description="Optimization direction: 'maximize' or 'minimize'"
    )
    variable_types: List[str] = Field(
        default=["continuous", "continuous"],
        description="Variable types: 'continuous', 'integer', or 'binary'"
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


def solve_milp(
    c: List[float],
    A: List[List[float]],
    b: List[float],
    constraint_types: List[str],
    objective: str,
    variable_types: List[str]
) -> Dict[str, Any]:
    """Solve mixed-integer linear programming problem using scipy.optimize.milp"""
    
    c_arr = np.array(c, dtype=float)
    A_arr = np.array(A, dtype=float)
    b_arr = np.array(b, dtype=float)
    n_vars = len(c)
    
    # Convert to minimization (scipy only does minimization)
    if objective == "maximize":
        c_solve = -c_arr
    else:
        c_solve = c_arr.copy()
    
    # Build constraint bounds
    lb_constraints = []
    ub_constraints = []
    
    for i, ctype in enumerate(constraint_types):
        if ctype == "<=":
            lb_constraints.append(-np.inf)
            ub_constraints.append(b_arr[i])
        elif ctype == ">=":
            lb_constraints.append(b_arr[i])
            ub_constraints.append(np.inf)
        else:  # ==
            lb_constraints.append(b_arr[i])
            ub_constraints.append(b_arr[i])
    
    constraints = LinearConstraint(A_arr, lb_constraints, ub_constraints)
    
    # Variable bounds (non-negativity, binary bounds)
    lb_vars = np.zeros(n_vars)
    ub_vars = np.full(n_vars, np.inf)
    
    for i, vtype in enumerate(variable_types):
        if vtype == "binary":
            ub_vars[i] = 1.0
    
    bounds = Bounds(lb_vars, ub_vars)
    
    # Integrality constraints
    # 0 = continuous, 1 = integer
    integrality = []
    for vtype in variable_types:
        if vtype in ["integer", "binary"]:
            integrality.append(1)
        else:
            integrality.append(0)
    
    # Solve
    result = milp(
        c=c_solve,
        constraints=constraints,
        bounds=bounds,
        integrality=integrality
    )
    
    if result.success:
        optimal_value = -result.fun if objective == "maximize" else result.fun
        solution = result.x
        
        # Round integer/binary variables
        for i, vtype in enumerate(variable_types):
            if vtype in ["integer", "binary"]:
                solution[i] = round(solution[i])
        
        # Calculate slack/surplus
        slack = []
        for i, ctype in enumerate(constraint_types):
            lhs = np.dot(A_arr[i], solution)
            if ctype == "<=":
                slack.append(_to_native_type(b_arr[i] - lhs))
            elif ctype == ">=":
                slack.append(_to_native_type(lhs - b_arr[i]))
            else:
                slack.append(0.0)
        
        binding = [abs(s) < 1e-6 for s in slack]
        
        return {
            "success": True,
            "optimal_value": _to_native_type(optimal_value),
            "solution": [_to_native_type(x) for x in solution],
            "slack": slack,
            "binding_constraints": binding,
            "status": "Optimal solution found"
        }
    else:
        return {
            "success": False,
            "optimal_value": None,
            "solution": None,
            "slack": None,
            "binding_constraints": None,
            "status": result.message if hasattr(result, 'message') else "No solution found"
        }


def generate_feasible_region_plot(
    c: List[float],
    A: List[List[float]],
    b: List[float],
    constraint_types: List[str],
    solution: List[float],
    optimal_value: float,
    variable_types: List[str]
) -> Optional[str]:
    """Generate 2D feasible region plot with integer grid points"""
    
    if len(c) != 2:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    max_b = max(max(b), 1) * 1.3
    x_max = min(max_b, 50)
    y_max = min(max_b, 50)
    
    x = np.linspace(0, x_max, 400)
    
    # Plot constraints
    colors = plt.cm.Set2(np.linspace(0, 1, len(A)))
    
    for i, (row, bi, ctype) in enumerate(zip(A, b, constraint_types)):
        a1, a2 = row
        if abs(a2) > 1e-10:
            y_line = (bi - a1 * x) / a2
            ax.plot(x, y_line, '-', color=colors[i], linewidth=2,
                   label=f'{a1}x₁ + {a2}x₂ {ctype} {bi}')
        elif abs(a1) > 1e-10:
            x_val = bi / a1
            ax.axvline(x=x_val, color=colors[i], linewidth=2,
                      label=f'{a1}x₁ {ctype} {bi}')
    
    # Find and fill feasible region
    try:
        vertices = [(0, 0)]
        
        for i, (row, bi, ctype) in enumerate(zip(A, b, constraint_types)):
            a1, a2 = row
            if ctype in ["<=", "=="]:
                if abs(a1) > 1e-10 and bi/a1 >= 0:
                    vertices.append((bi/a1, 0))
                if abs(a2) > 1e-10 and bi/a2 >= 0:
                    vertices.append((0, bi/a2))
        
        for i in range(len(A)):
            for j in range(i+1, len(A)):
                a = np.array([A[i], A[j]])
                b_vec = np.array([b[i], b[j]])
                try:
                    if abs(np.linalg.det(a)) > 1e-10:
                        point = np.linalg.solve(a, b_vec)
                        if point[0] >= -1e-6 and point[1] >= -1e-6:
                            vertices.append((point[0], point[1]))
                except:
                    pass
        
        feasible_vertices = []
        for v in vertices:
            if v[0] < -1e-6 or v[1] < -1e-6:
                continue
            feasible = True
            for row, bi, ctype in zip(A, b, constraint_types):
                lhs = row[0] * v[0] + row[1] * v[1]
                if ctype == "<=" and lhs > bi + 1e-6:
                    feasible = False
                elif ctype == ">=" and lhs < bi - 1e-6:
                    feasible = False
                elif ctype == "==" and abs(lhs - bi) > 1e-6:
                    feasible = False
            if feasible:
                feasible_vertices.append(v)
        
        if len(feasible_vertices) >= 3:
            centroid = np.mean(feasible_vertices, axis=0)
            def angle(v):
                return np.arctan2(v[1] - centroid[1], v[0] - centroid[0])
            feasible_vertices.sort(key=angle)
            
            polygon = plt.Polygon(feasible_vertices, alpha=0.3, color='#4CAF50',
                                 label='Feasible Region')
            ax.add_patch(polygon)
    except:
        pass
    
    # Plot integer grid points if any variable is integer/binary
    has_integer = any(vt in ["integer", "binary"] for vt in variable_types)
    if has_integer:
        x_int_max = int(min(x_max, 30))
        y_int_max = int(min(y_max, 30))
        
        feasible_integers = []
        for xi in range(x_int_max + 1):
            for yi in range(y_int_max + 1):
                # Check if this integer point is feasible
                feasible = True
                for row, bi, ctype in zip(A, b, constraint_types):
                    lhs = row[0] * xi + row[1] * yi
                    if ctype == "<=" and lhs > bi + 1e-6:
                        feasible = False
                    elif ctype == ">=" and lhs < bi - 1e-6:
                        feasible = False
                    elif ctype == "==" and abs(lhs - bi) > 1e-6:
                        feasible = False
                if feasible:
                    feasible_integers.append((xi, yi))
        
        if feasible_integers:
            int_x = [p[0] for p in feasible_integers]
            int_y = [p[1] for p in feasible_integers]
            ax.scatter(int_x, int_y, color='blue', s=30, alpha=0.5,
                      zorder=3, label='Feasible Integer Points')
    
    # Plot optimal solution
    if solution:
        ax.scatter([solution[0]], [solution[1]], color='red', s=200,
                  zorder=5, marker='*', edgecolors='black', linewidths=1.5,
                  label=f'Optimal: ({solution[0]:.2f}, {solution[1]:.2f})')
    
    ax.axhline(y=0, color='black', linewidth=1)
    ax.axvline(x=0, color='black', linewidth=1)
    
    ax.set_xlim(-0.5, x_max)
    ax.set_ylim(-0.5, y_max)
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title('Integer Programming - Feasible Region', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_sensitivity_plot(
    c: List[float],
    solution: List[float],
    A: List[List[float]],
    b: List[float],
    variable_types: List[str]
) -> str:
    """Generate sensitivity analysis bar chart"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Variable contribution
    ax1 = axes[0]
    var_names = [f'x{i+1}' for i in range(len(c))]
    contributions = [c[i] * solution[i] for i in range(len(c))]
    
    # Color by variable type
    colors = []
    for i, vt in enumerate(variable_types):
        if vt == "binary":
            colors.append('#9f7aea')
        elif vt == "integer":
            colors.append('#4299e1')
        else:
            colors.append('#48bb78')
    
    bars = ax1.bar(var_names, contributions, color=colors, edgecolor='black', linewidth=1)
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.set_title('Variable Contributions to Z', fontsize=12)
    ax1.set_ylabel('Contribution')
    
    # Add value labels
    for bar, val in zip(bars, contributions):
        height = bar.get_height()
        ax1.annotate(f'{val:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -15),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    # Legend for variable types
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#48bb78', label='Continuous'),
        Patch(facecolor='#4299e1', label='Integer'),
        Patch(facecolor='#9f7aea', label='Binary')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # Constraint usage
    ax2 = axes[1]
    constraint_names = [f'C{i+1}' for i in range(len(b))]
    
    usage = []
    for row in A:
        lhs = sum(row[j] * solution[j] for j in range(len(solution)))
        usage.append(lhs)
    
    x_pos = np.arange(len(b))
    width = 0.35
    
    bars1 = ax2.bar(x_pos - width/2, usage, width, label='Used', color='#2196F3', edgecolor='black')
    bars2 = ax2.bar(x_pos + width/2, b, width, label='Available', color='#BBDEFB', edgecolor='black')
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(constraint_names)
    ax2.set_title('Constraint Usage vs. Available', fontsize=12)
    ax2.set_ylabel('Value')
    ax2.legend()
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_interpretation(result: Dict, params: Dict) -> Dict[str, Any]:
    """Generate interpretation of MILP results"""
    key_insights = []
    
    variable_types = params.get('variable_types', [])
    n_integer = sum(1 for vt in variable_types if vt == 'integer')
    n_binary = sum(1 for vt in variable_types if vt == 'binary')
    n_continuous = sum(1 for vt in variable_types if vt == 'continuous')
    
    if result['success']:
        key_insights.append({
            'title': 'Optimal Solution Found',
            'description': f"The {params['objective']} value is {result['optimal_value']:.4f}.",
            'status': 'positive'
        })
        
        # Variable type summary
        type_parts = []
        if n_continuous > 0:
            type_parts.append(f"{n_continuous} continuous")
        if n_integer > 0:
            type_parts.append(f"{n_integer} integer")
        if n_binary > 0:
            type_parts.append(f"{n_binary} binary")
        
        key_insights.append({
            'title': 'Problem Type',
            'description': f"Mixed-Integer LP with {', '.join(type_parts)} variable(s).",
            'status': 'neutral'
        })
        
        # Solution analysis
        solution = result['solution']
        active_vars = [(i, solution[i], variable_types[i]) for i in range(len(solution)) if abs(solution[i]) > 1e-6]
        if active_vars:
            var_str = ', '.join([f'x{i+1}={val:.2f} ({vt})' for i, val, vt in active_vars])
            key_insights.append({
                'title': 'Active Variables',
                'description': f"Non-zero variables: {var_str}",
                'status': 'neutral'
            })
        
        # Binding constraints
        if result['binding_constraints']:
            binding_idx = [i+1 for i, b in enumerate(result['binding_constraints']) if b]
            if binding_idx:
                key_insights.append({
                    'title': 'Binding Constraints',
                    'description': f"Constraints {binding_idx} are tight at the optimal solution.",
                    'status': 'neutral'
                })
    else:
        key_insights.append({
            'title': 'No Optimal Solution',
            'description': f"Status: {result['status']}. The problem may be infeasible.",
            'status': 'warning'
        })
    
    # Recommendations
    recommendations = []
    if result['success']:
        if n_integer + n_binary > 0:
            recommendations.append("Integer constraints may cause the solution to differ from the LP relaxation.")
        recommendations.append("Consider sensitivity analysis to understand how changes affect the optimal solution.")
    else:
        recommendations.append("Check if constraints are too restrictive for integer solutions.")
        recommendations.append("Try relaxing some integer constraints to continuous.")
    
    return {
        'key_insights': key_insights,
        'recommendations': recommendations
    }


@router.post("/integer-programming")
async def solve_integer_programming(request: IntegerProgrammingRequest) -> Dict[str, Any]:
    """
    Solve a Mixed-Integer Linear Programming (MILP) problem.
    
    MILP extends LP by allowing some variables to be restricted
    to integer or binary (0/1) values.
    
    Standard form:
        Maximize/Minimize: Z = c^T * x
        Subject to: Ax <= b (or >=, =)
                   x >= 0
                   some x_i ∈ Z (integers)
    """
    try:
        # Validate inputs
        n_vars = len(request.c)
        n_constraints = len(request.b)
        
        if len(request.A) != n_constraints:
            raise HTTPException(status_code=400, detail="Number of constraint rows must match length of b")
        
        for i, row in enumerate(request.A):
            if len(row) != n_vars:
                raise HTTPException(status_code=400, detail=f"Constraint {i+1} has wrong number of coefficients")
        
        if len(request.constraint_types) != n_constraints:
            raise HTTPException(status_code=400, detail="Number of constraint types must match number of constraints")
        
        if len(request.variable_types) != n_vars:
            raise HTTPException(status_code=400, detail="Number of variable types must match number of variables")
        
        for ct in request.constraint_types:
            if ct not in ["<=", ">=", "=="]:
                raise HTTPException(status_code=400, detail=f"Invalid constraint type: {ct}")
        
        for vt in request.variable_types:
            if vt not in ["continuous", "integer", "binary"]:
                raise HTTPException(status_code=400, detail=f"Invalid variable type: {vt}")
        
        if request.objective not in ["maximize", "minimize"]:
            raise HTTPException(status_code=400, detail="Objective must be 'maximize' or 'minimize'")
        
        # Solve MILP
        result = solve_milp(
            c=request.c,
            A=request.A,
            b=request.b,
            constraint_types=request.constraint_types,
            objective=request.objective,
            variable_types=request.variable_types
        )
        
        # Generate plots
        plots = {}
        
        if result['success'] and result['solution']:
            # 2D feasible region with integer points
            if n_vars == 2:
                feasible_plot = generate_feasible_region_plot(
                    request.c, request.A, request.b,
                    request.constraint_types, result['solution'],
                    result['optimal_value'], request.variable_types
                )
                if feasible_plot:
                    plots['feasible_region'] = feasible_plot
            
            # Sensitivity analysis
            plots['sensitivity'] = generate_sensitivity_plot(
                request.c, result['solution'], request.A, request.b,
                request.variable_types
            )
        
        # Generate interpretation
        params = {
            'c': request.c,
            'A': request.A,
            'b': request.b,
            'constraint_types': request.constraint_types,
            'objective': request.objective,
            'variable_types': request.variable_types
        }
        interpretation = generate_interpretation(result, params)
        
        # Build problem string
        obj_str = " + ".join([f"{c}x{i+1}" for i, c in enumerate(request.c)])
        constraint_strs = []
        for i, (row, bi, ct) in enumerate(zip(request.A, request.b, request.constraint_types)):
            lhs = " + ".join([f"{a}x{j+1}" for j, a in enumerate(row)])
            constraint_strs.append(f"{lhs} {ct} {bi}")
        
        return {
            'success': result['success'],
            'status': result['status'],
            'optimal_value': result['optimal_value'],
            'solution': result['solution'],
            'slack': result['slack'],
            'binding_constraints': result['binding_constraints'],
            'problem': {
                'objective': request.objective,
                'objective_function': f"Z = {obj_str}",
                'constraints': constraint_strs,
                'n_variables': n_vars,
                'n_constraints': n_constraints,
                'variable_types': request.variable_types
            },
            'plots': plots,
            'interpretation': interpretation
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Integer programming solver failed: {str(e)}")
