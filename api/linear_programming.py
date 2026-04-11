"""
Linear Programming Solver Router for FastAPI
Solve and visualize linear programming optimization problems
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from scipy.spatial import HalfspaceIntersection, ConvexHull
import io
import base64
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class LinearProgrammingRequest(BaseModel):
    """Linear Programming solver request parameters"""
    c: List[float] = Field(
        default=[-1, -2],
        description="Objective function coefficients (for minimization)"
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


def compute_feasible_vertices(
    A: List[List[float]],
    b: List[float],
    constraint_types: List[str]
) -> List[List[float]]:
    """Compute vertices of the feasible region for 2D problems"""
    if len(A[0]) != 2:
        return []
    
    A_arr = np.array(A)
    b_arr = np.array(b)
    
    vertices = []
    
    # Add origin
    vertices.append((0.0, 0.0))
    
    # Find intersections with axes
    for i, (row, bi, ctype) in enumerate(zip(A_arr, b_arr, constraint_types)):
        a1, a2 = row
        if ctype in ["<=", "=="]:
            if abs(a1) > 1e-10:
                vertices.append((bi/a1, 0.0))
            if abs(a2) > 1e-10:
                vertices.append((0.0, bi/a2))
    
    # Find intersections between constraints
    n = len(A_arr)
    for i in range(n):
        for j in range(i+1, n):
            a = np.array([A_arr[i], A_arr[j]])
            b_vec = np.array([b_arr[i], b_arr[j]])
            try:
                if abs(np.linalg.det(a)) > 1e-10:
                    point = np.linalg.solve(a, b_vec)
                    if point[0] >= -1e-6 and point[1] >= -1e-6:
                        vertices.append((float(point[0]), float(point[1])))
            except:
                pass
    
    # Filter vertices that satisfy all constraints
    feasible_vertices = []
    for v in vertices:
        if v[0] < -1e-6 or v[1] < -1e-6:
            continue
        feasible = True
        for row, bi, ctype in zip(A_arr, b_arr, constraint_types):
            lhs = row[0] * v[0] + row[1] * v[1]
            if ctype == "<=" and lhs > bi + 1e-6:
                feasible = False
            elif ctype == ">=" and lhs < bi - 1e-6:
                feasible = False
            elif ctype == "==" and abs(lhs - bi) > 1e-6:
                feasible = False
        if feasible:
            feasible_vertices.append(v)
    
    # Remove duplicates
    unique_vertices = []
    for v in feasible_vertices:
        is_dup = False
        for uv in unique_vertices:
            if abs(v[0] - uv[0]) < 1e-6 and abs(v[1] - uv[1]) < 1e-6:
                is_dup = True
                break
        if not is_dup:
            unique_vertices.append(v)
    
    # Sort by angle from centroid
    if len(unique_vertices) >= 3:
        centroid = np.mean(unique_vertices, axis=0)
        def angle(v):
            return np.arctan2(v[1] - centroid[1], v[0] - centroid[0])
        unique_vertices.sort(key=angle)
    
    return [[v[0], v[1]] for v in unique_vertices]


def solve_lp(
    c: List[float],
    A: List[List[float]],
    b: List[float],
    constraint_types: List[str],
    objective: str
) -> Dict[str, Any]:
    """Solve linear programming problem using scipy"""
    
    c_arr = np.array(c)
    A_arr = np.array(A)
    b_arr = np.array(b)
    
    # Convert to minimization (scipy only does minimization)
    if objective == "maximize":
        c_solve = -c_arr
    else:
        c_solve = c_arr
    
    # Separate constraints by type
    A_ub = []
    b_ub = []
    A_eq = []
    b_eq = []
    
    for i, ctype in enumerate(constraint_types):
        if ctype == "<=":
            A_ub.append(A_arr[i])
            b_ub.append(b_arr[i])
        elif ctype == ">=":
            A_ub.append(-A_arr[i])
            b_ub.append(-b_arr[i])
        else:  # ==
            A_eq.append(A_arr[i])
            b_eq.append(b_arr[i])
    
    A_ub = np.array(A_ub) if A_ub else None
    b_ub = np.array(b_ub) if b_ub else None
    A_eq = np.array(A_eq) if A_eq else None
    b_eq = np.array(b_eq) if b_eq else None
    
    # Non-negativity bounds
    bounds = [(0, None) for _ in range(len(c))]
    
    # Solve
    result = linprog(
        c_solve,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds,
        method='highs'
    )
    
    # Compute feasible vertices for 2D problems
    feasible_vertices = compute_feasible_vertices(A, b, constraint_types) if len(c) == 2 else []
    
    if result.success:
        optimal_value = -result.fun if objective == "maximize" else result.fun
        solution = result.x
        
        # Calculate slack/surplus for each constraint
        slack = []
        for i, ctype in enumerate(constraint_types):
            lhs = np.dot(A_arr[i], solution)
            if ctype == "<=":
                slack.append(_to_native_type(b_arr[i] - lhs))
            elif ctype == ">=":
                slack.append(_to_native_type(lhs - b_arr[i]))
            else:
                slack.append(0.0)
        
        # Identify binding constraints
        binding = [abs(s) < 1e-6 for s in slack]
        
        return {
            "success": True,
            "optimal_value": _to_native_type(optimal_value),
            "solution": [_to_native_type(x) for x in solution],
            "slack": slack,
            "binding_constraints": binding,
            "iterations": result.nit if hasattr(result, 'nit') else None,
            "status": "Optimal solution found",
            "feasible_vertices": feasible_vertices
        }
    else:
        return {
            "success": False,
            "optimal_value": None,
            "solution": None,
            "slack": None,
            "binding_constraints": None,
            "iterations": None,
            "status": result.message,
            "feasible_vertices": feasible_vertices
        }


def generate_feasible_region_plot(
    c: List[float],
    A: List[List[float]],
    b: List[float],
    constraint_types: List[str],
    solution: List[float],
    optimal_value: float,
    objective: str
) -> Optional[str]:
    """Generate 2D feasible region plot (only for 2-variable problems)"""
    
    if len(c) != 2:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Determine plot bounds
    max_b = max(max(b), 1) * 1.5
    x_max = max_b
    y_max = max_b
    
    x = np.linspace(0, x_max, 400)
    
    # Plot each constraint
    colors = plt.cm.Set2(np.linspace(0, 1, len(A)))
    
    for i, (row, bi, ctype) in enumerate(zip(A, b, constraint_types)):
        a1, a2 = row
        
        if abs(a2) > 1e-10:
            y_line = (bi - a1 * x) / a2
            ax.plot(x, y_line, '-', color=colors[i], linewidth=2, 
                   label=f'{a1}x₁ + {a2}x₂ {ctype} {bi}')
        else:
            x_val = bi / a1 if abs(a1) > 1e-10 else 0
            ax.axvline(x=x_val, color=colors[i], linewidth=2,
                      label=f'{a1}x₁ {ctype} {bi}')
    
    # Find and fill feasible region
    try:
        # Create polygon vertices by checking all intersections
        vertices = []
        
        # Add origin and axis intersections
        vertices.append((0, 0))
        
        # Find intersections with axes
        for i, (row, bi, ctype) in enumerate(zip(A, b, constraint_types)):
            a1, a2 = row
            if ctype in ["<=", "=="]:
                if abs(a1) > 1e-10:
                    vertices.append((bi/a1, 0))
                if abs(a2) > 1e-10:
                    vertices.append((0, bi/a2))
        
        # Find intersections between constraints
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
        
        # Filter vertices that satisfy all constraints
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
            # Sort vertices by angle from centroid
            centroid = np.mean(feasible_vertices, axis=0)
            def angle(v):
                return np.arctan2(v[1] - centroid[1], v[0] - centroid[0])
            feasible_vertices.sort(key=angle)
            
            polygon = plt.Polygon(feasible_vertices, alpha=0.3, color='#4CAF50', 
                                 label='Feasible Region')
            ax.add_patch(polygon)
    except Exception as e:
        pass
    
    # Plot objective function contours
    if solution:
        c1, c2 = c
        if abs(c2) > 1e-10:
            # Draw iso-profit/cost lines
            for level in np.linspace(0, optimal_value * 1.5, 5):
                y_obj = (level - c1 * x) / c2
                alpha = 0.3 if level != optimal_value else 0.8
                ax.plot(x, y_obj, '--', color='purple', alpha=alpha, linewidth=1)
            
            # Optimal contour
            y_opt = (optimal_value - c1 * x) / c2
            ax.plot(x, y_opt, '--', color='purple', linewidth=2, 
                   label=f'Optimal Z = {optimal_value:.2f}')
        
        # Plot optimal solution
        ax.scatter([solution[0]], [solution[1]], color='red', s=200, 
                  zorder=5, marker='*', edgecolors='black', linewidths=1.5,
                  label=f'Optimal: ({solution[0]:.2f}, {solution[1]:.2f})')
    
    # Non-negativity
    ax.axhline(y=0, color='black', linewidth=1)
    ax.axvline(x=0, color='black', linewidth=1)
    
    ax.set_xlim(-0.5, x_max)
    ax.set_ylim(-0.5, y_max)
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title('Linear Programming - Feasible Region', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_sensitivity_plot(
    c: List[float],
    A: List[List[float]],
    b: List[float],
    constraint_types: List[str],
    objective: str,
    solution: List[float],
    optimal_value: float
) -> str:
    """Generate sensitivity analysis bar chart"""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Variable contribution
    ax1 = axes[0]
    var_names = [f'x₁' if i == 0 else f'x₂' if i == 1 else f'x{i+1}' for i in range(len(c))]
    contributions = [c[i] * solution[i] for i in range(len(c))]
    
    colors = ['#4CAF50' if v >= 0 else '#F44336' for v in contributions]
    bars = ax1.bar(var_names, contributions, color=colors, edgecolor='black', linewidth=1)
    ax1.axhline(y=0, color='black', linewidth=0.5)
    ax1.set_title('Variable Contributions to Z', fontsize=12)
    ax1.set_ylabel('Contribution')
    
    for bar, val in zip(bars, contributions):
        height = bar.get_height()
        ax1.annotate(f'{val:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3 if height >= 0 else -15),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    
    # Constraint slack/surplus
    ax2 = axes[1]
    constraint_names = [f'C{i+1}' for i in range(len(b))]
    
    # Calculate usage
    usage = []
    for i, row in enumerate(A):
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


def generate_solution_space_3d(
    c: List[float],
    A: List[List[float]],
    b: List[float],
    solution: List[float],
    optimal_value: float
) -> Optional[str]:
    """Generate 3D visualization showing objective surface over feasible region"""
    
    if len(c) != 2:
        return None
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create mesh
    max_b = max(max(b), 1) * 1.2
    x1 = np.linspace(0, max_b, 50)
    x2 = np.linspace(0, max_b, 50)
    X1, X2 = np.meshgrid(x1, x2)
    
    # Objective function surface
    Z = c[0] * X1 + c[1] * X2
    
    # Mask infeasible region
    mask = np.ones_like(Z, dtype=bool)
    for row, bi in zip(A, b):
        constraint_values = row[0] * X1 + row[1] * X2
        mask &= (constraint_values <= bi + 0.01)
    mask &= (X1 >= 0) & (X2 >= 0)
    
    Z_masked = np.where(mask, Z, np.nan)
    
    # Plot surface
    surf = ax.plot_surface(X1, X2, Z_masked, cmap='viridis', alpha=0.7,
                          linewidth=0, antialiased=True)
    
    # Plot optimal point
    if solution:
        ax.scatter([solution[0]], [solution[1]], [optimal_value],
                  color='red', s=200, marker='*', edgecolors='black',
                  linewidths=1.5, zorder=5, label='Optimal')
    
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_zlabel('Z (Objective)')
    ax.set_title('Objective Function over Feasible Region')
    
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Z value')
    plt.tight_layout()
    
    return _fig_to_base64(fig)


def generate_interpretation(result: Dict, params: Dict) -> Dict[str, Any]:
    """Generate interpretation of LP results"""
    key_insights = []
    
    if result['success']:
        # Optimal solution found
        key_insights.append({
            'title': 'Optimal Solution Found',
            'description': f"The {params['objective']} value is {result['optimal_value']:.4f}.",
            'status': 'positive'
        })
        
        # Variable analysis
        solution = result['solution']
        active_vars = [i for i, x in enumerate(solution) if x > 1e-6]
        if active_vars:
            var_str = ', '.join([f'x{i+1}={solution[i]:.4f}' for i in active_vars])
            key_insights.append({
                'title': 'Active Variables',
                'description': f"Variables with positive values: {var_str}",
                'status': 'neutral'
            })
        
        # Binding constraints
        if result['binding_constraints']:
            binding_idx = [i+1 for i, b in enumerate(result['binding_constraints']) if b]
            if binding_idx:
                key_insights.append({
                    'title': 'Binding Constraints',
                    'description': f"Constraints {binding_idx} are tight (no slack). These limit further improvement.",
                    'status': 'neutral'
                })
        
        # Slack analysis
        if result['slack']:
            non_binding = [(i+1, s) for i, s in enumerate(result['slack']) if s > 1e-6]
            if non_binding:
                slack_str = ', '.join([f'C{i}: {s:.2f}' for i, s in non_binding])
                key_insights.append({
                    'title': 'Slack Resources',
                    'description': f"Unused capacity in constraints: {slack_str}",
                    'status': 'neutral'
                })
    else:
        key_insights.append({
            'title': 'No Optimal Solution',
            'description': f"Status: {result['status']}. The problem may be infeasible or unbounded.",
            'status': 'warning'
        })
    
    # Recommendations
    recommendations = []
    if result['success']:
        if result['binding_constraints'] and all(result['binding_constraints']):
            recommendations.append("All constraints are binding. Consider relaxing constraints for potential improvement.")
        if any(s > max(params['b']) * 0.5 for s in (result['slack'] or [])):
            recommendations.append("Some constraints have significant slack. They may be redundant for this solution.")
        recommendations.append("For sensitivity analysis, examine how changes in coefficients affect the optimal solution.")
    else:
        recommendations.append("Check constraint directions and values for feasibility.")
        recommendations.append("Verify that the problem is properly bounded.")
    
    return {
        'key_insights': key_insights,
        'recommendations': recommendations
    }


@router.post("/linear-programming")
async def solve_linear_programming(request: LinearProgrammingRequest) -> Dict[str, Any]:
    """
    Solve a Linear Programming problem.
    
    Linear Programming optimizes a linear objective function
    subject to linear equality and inequality constraints.
    
    Standard form:
        Maximize/Minimize: Z = c^T * x
        Subject to: Ax <= b (or >=, =)
                   x >= 0
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
        
        for ct in request.constraint_types:
            if ct not in ["<=", ">=", "=="]:
                raise HTTPException(status_code=400, detail=f"Invalid constraint type: {ct}")
        
        if request.objective not in ["maximize", "minimize"]:
            raise HTTPException(status_code=400, detail="Objective must be 'maximize' or 'minimize'")
        
        # Solve LP
        result = solve_lp(
            c=request.c,
            A=request.A,
            b=request.b,
            constraint_types=request.constraint_types,
            objective=request.objective
        )
        
        # Generate plots
        plots = {}
        
        if result['success'] and result['solution']:
            # 2D feasible region
            feasible_plot = generate_feasible_region_plot(
                request.c, request.A, request.b,
                request.constraint_types, result['solution'],
                result['optimal_value'], request.objective
            )
            if feasible_plot:
                plots['feasible_region'] = feasible_plot
            
            # Sensitivity analysis
            plots['sensitivity'] = generate_sensitivity_plot(
                request.c, request.A, request.b,
                request.constraint_types, request.objective,
                result['solution'], result['optimal_value']
            )
            
            # 3D surface
            surface_3d = generate_solution_space_3d(
                request.c, request.A, request.b,
                result['solution'], result['optimal_value']
            )
            if surface_3d:
                plots['surface_3d'] = surface_3d
        
        # Generate interpretation
        params = {
            'c': request.c,
            'A': request.A,
            'b': request.b,
            'constraint_types': request.constraint_types,
            'objective': request.objective
        }
        interpretation = generate_interpretation(result, params)
        
        # Build problem string for display
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
            'iterations': result['iterations'],
            'feasible_vertices': result.get('feasible_vertices', []),
            'problem': {
                'objective': request.objective,
                'objective_function': f"Z = {obj_str}",
                'constraints': constraint_strs,
                'n_variables': n_vars,
                'n_constraints': n_constraints
            },
            'plots': plots,
            'interpretation': interpretation
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Linear programming solver failed: {str(e)}")
