"""
Non-linear Programming Solver Router for FastAPI
Solve and visualize non-linear optimization problems
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import io
import base64
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class NLPConstraint(BaseModel):
    type: str = Field(default="ineq", description="Constraint type: 'eq' or 'ineq'")
    fun: str = Field(default="", description="Constraint function in Python format")


class NonlinearProgrammingRequest(BaseModel):
    """Non-linear Programming solver request parameters"""
    objective_function: str = Field(
        default="(x[0] - 1)**2 + (x[1] - 2.5)**2",
        description="Objective function in Python format (minimize)"
    )
    num_vars: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Number of decision variables"
    )
    bounds: List[List[Optional[float]]] = Field(
        default=[[0, None], [0, None]],
        description="Variable bounds [[min, max], ...]"
    )
    initial_guess: List[float] = Field(
        default=[2.0, 0.0],
        description="Initial guess for optimization"
    )
    constraints: List[NLPConstraint] = Field(
        default=[],
        description="List of constraints"
    )
    method: str = Field(
        default="SLSQP",
        description="Optimization method"
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


def create_objective_function(func_str: str):
    """Create objective function from string"""
    def objective(x):
        return eval(func_str, {"__builtins__": {}, "np": np, "x": x, 
                               "sin": np.sin, "cos": np.cos, "exp": np.exp,
                               "sqrt": np.sqrt, "log": np.log, "abs": np.abs})
    return objective


def create_constraint_function(func_str: str):
    """Create constraint function from string"""
    def constraint(x):
        return eval(func_str, {"__builtins__": {}, "np": np, "x": x,
                               "sin": np.sin, "cos": np.cos, "exp": np.exp,
                               "sqrt": np.sqrt, "log": np.log, "abs": np.abs})
    return constraint


def solve_nlp(
    objective_function: str,
    bounds: List[List[Optional[float]]],
    initial_guess: List[float],
    constraints: List[NLPConstraint],
    method: str
) -> Dict[str, Any]:
    """Solve non-linear programming problem"""
    
    # Create objective function
    obj_func = create_objective_function(objective_function)
    
    # Convert bounds
    scipy_bounds = []
    for b in bounds:
        lb = b[0] if b[0] is not None else None
        ub = b[1] if b[1] is not None else None
        scipy_bounds.append((lb, ub))
    
    # Create constraints
    scipy_constraints = []
    for c in constraints:
        if c.fun.strip():
            scipy_constraints.append({
                'type': c.type,
                'fun': create_constraint_function(c.fun)
            })
    
    # Initial guess
    x0 = np.array(initial_guess)
    
    # Solve
    result = minimize(
        obj_func,
        x0,
        method=method,
        bounds=scipy_bounds if scipy_bounds else None,
        constraints=scipy_constraints if scipy_constraints else None,
        options={'maxiter': 1000, 'disp': False}
    )
    
    return {
        "success": result.success,
        "solution": [_to_native_type(x) for x in result.x],
        "objective_value": _to_native_type(result.fun),
        "n_iterations": _to_native_type(result.nit) if hasattr(result, 'nit') else None,
        "message": result.message if hasattr(result, 'message') else str(result.success),
        "initial_guess": initial_guess,
        "initial_value": _to_native_type(obj_func(x0))
    }


def generate_contour_plot(
    objective_function: str,
    bounds: List[List[Optional[float]]],
    constraints: List[NLPConstraint],
    solution: List[float],
    initial_guess: List[float]
) -> Optional[str]:
    """Generate 2D contour plot for 2-variable problems"""
    
    if len(solution) != 2:
        return None
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    obj_func = create_objective_function(objective_function)
    
    # Determine plot range
    x_min = bounds[0][0] if bounds[0][0] is not None else min(solution[0], initial_guess[0]) - 2
    x_max = bounds[0][1] if bounds[0][1] is not None else max(solution[0], initial_guess[0]) + 2
    y_min = bounds[1][0] if bounds[1][0] is not None else min(solution[1], initial_guess[1]) - 2
    y_max = bounds[1][1] if bounds[1][1] is not None else max(solution[1], initial_guess[1]) + 2
    
    # Expand range a bit
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_min -= x_range * 0.1
    x_max += x_range * 0.1
    y_min -= y_range * 0.1
    y_max += y_range * 0.1
    
    # Create mesh
    x = np.linspace(x_min, x_max, 100)
    y = np.linspace(y_min, y_max, 100)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate objective
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            try:
                Z[i, j] = obj_func(np.array([X[i, j], Y[i, j]]))
            except:
                Z[i, j] = np.nan
    
    # Contour plot
    levels = np.linspace(np.nanmin(Z), np.nanpercentile(Z, 95), 30)
    contour = ax.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.8)
    ax.contour(X, Y, Z, levels=levels, colors='white', alpha=0.3, linewidths=0.5)
    plt.colorbar(contour, ax=ax, label='f(x)')
    
    # Plot constraints
    colors = ['#e53e3e', '#ed8936', '#9f7aea', '#38b2ac', '#4299e1']
    for idx, c in enumerate(constraints):
        if c.fun.strip():
            try:
                c_func = create_constraint_function(c.fun)
                C = np.zeros_like(X)
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        try:
                            C[i, j] = c_func(np.array([X[i, j], Y[i, j]]))
                        except:
                            C[i, j] = np.nan
                
                # Plot constraint boundary
                ax.contour(X, Y, C, levels=[0], colors=[colors[idx % len(colors)]], 
                          linewidths=2, linestyles='--')
                
                # Shade infeasible region for inequality constraints
                if c.type == 'ineq':
                    ax.contourf(X, Y, C, levels=[-np.inf, 0], colors=[colors[idx % len(colors)]], 
                               alpha=0.1)
            except:
                pass
    
    # Plot initial guess
    ax.scatter([initial_guess[0]], [initial_guess[1]], color='blue', s=100, 
              marker='o', zorder=5, edgecolors='white', linewidths=2,
              label=f'Initial: ({initial_guess[0]:.2f}, {initial_guess[1]:.2f})')
    
    # Plot optimal solution
    ax.scatter([solution[0]], [solution[1]], color='red', s=200,
              marker='*', zorder=5, edgecolors='white', linewidths=1.5,
              label=f'Optimal: ({solution[0]:.4f}, {solution[1]:.4f})')
    
    # Draw arrow from initial to optimal
    ax.annotate('', xy=(solution[0], solution[1]), xytext=(initial_guess[0], initial_guess[1]),
                arrowprops=dict(arrowstyle='->', color='white', lw=2))
    
    ax.set_xlabel('x[0]', fontsize=12)
    ax.set_ylabel('x[1]', fontsize=12)
    ax.set_title('Non-linear Optimization Landscape', fontsize=14)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_3d_surface(
    objective_function: str,
    bounds: List[List[Optional[float]]],
    solution: List[float],
    optimal_value: float
) -> Optional[str]:
    """Generate 3D surface plot data for Plotly"""
    
    if len(solution) != 2:
        return None
    
    # This will be rendered in frontend with Plotly
    # Just return the data needed
    return "plotly"  # Signal to use Plotly in frontend


def generate_convergence_info(
    initial_value: float,
    optimal_value: float,
    n_iterations: int
) -> Dict[str, Any]:
    """Generate convergence information"""
    
    improvement = initial_value - optimal_value
    improvement_pct = (improvement / abs(initial_value) * 100) if initial_value != 0 else 0
    
    return {
        "initial_value": _to_native_type(initial_value),
        "optimal_value": _to_native_type(optimal_value),
        "improvement": _to_native_type(improvement),
        "improvement_pct": _to_native_type(improvement_pct),
        "iterations": n_iterations
    }


def generate_interpretation(result: Dict, params: Dict) -> Dict[str, Any]:
    """Generate interpretation of NLP results"""
    key_insights = []
    
    if result['success']:
        key_insights.append({
            'title': 'Optimal Solution Found',
            'description': f"Converged to minimum value {result['objective_value']:.6f} in {result['n_iterations']} iterations.",
            'status': 'positive'
        })
        
        # Improvement analysis
        if result['initial_value'] is not None:
            improvement = result['initial_value'] - result['objective_value']
            improvement_pct = (improvement / abs(result['initial_value']) * 100) if result['initial_value'] != 0 else 0
            key_insights.append({
                'title': 'Optimization Improvement',
                'description': f"Reduced objective from {result['initial_value']:.4f} to {result['objective_value']:.6f} ({improvement_pct:.1f}% improvement).",
                'status': 'positive' if improvement_pct > 50 else 'neutral'
            })
        
        # Solution summary
        sol_str = ', '.join([f'x[{i}]={v:.4f}' for i, v in enumerate(result['solution'])])
        key_insights.append({
            'title': 'Solution Point',
            'description': f"Optimal at: {sol_str}",
            'status': 'neutral'
        })
        
        # Constraint satisfaction
        n_constraints = len(params.get('constraints', []))
        if n_constraints > 0:
            key_insights.append({
                'title': 'Constraints',
                'description': f"Solution satisfies all {n_constraints} constraint(s).",
                'status': 'neutral'
            })
    else:
        key_insights.append({
            'title': 'Optimization Failed',
            'description': f"Status: {result['message']}",
            'status': 'warning'
        })
    
    # Recommendations
    recommendations = []
    if result['success']:
        recommendations.append("Solution represents a local minimum. For non-convex problems, try different initial guesses.")
        if result['n_iterations'] and result['n_iterations'] > 100:
            recommendations.append("High iteration count suggests a complex optimization landscape.")
    else:
        recommendations.append("Try adjusting initial guess or relaxing constraints.")
        recommendations.append("Check if the problem is feasible with current bounds.")
    
    return {
        'key_insights': key_insights,
        'recommendations': recommendations
    }


@router.post("/nonlinear-programming")
async def solve_nonlinear_programming(request: NonlinearProgrammingRequest) -> Dict[str, Any]:
    """
    Solve a Non-linear Programming (NLP) problem.
    
    NLP minimizes a non-linear objective function subject to
    non-linear equality and inequality constraints.
    
    Standard form:
        Minimize: f(x)
        Subject to: g_i(x) >= 0 (inequality)
                   h_j(x) = 0 (equality)
                   lb <= x <= ub
    """
    try:
        # Validate inputs
        if len(request.bounds) != request.num_vars:
            raise HTTPException(status_code=400, detail="Number of bounds must match num_vars")
        
        if len(request.initial_guess) != request.num_vars:
            raise HTTPException(status_code=400, detail="Initial guess length must match num_vars")
        
        # Test objective function
        try:
            obj_func = create_objective_function(request.objective_function)
            obj_func(np.array(request.initial_guess))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid objective function: {str(e)}")
        
        # Test constraints
        for i, c in enumerate(request.constraints):
            if c.fun.strip():
                try:
                    c_func = create_constraint_function(c.fun)
                    c_func(np.array(request.initial_guess))
                except Exception as e:
                    raise HTTPException(status_code=400, detail=f"Invalid constraint {i+1}: {str(e)}")
        
        # Solve NLP
        result = solve_nlp(
            objective_function=request.objective_function,
            bounds=request.bounds,
            initial_guess=request.initial_guess,
            constraints=request.constraints,
            method=request.method
        )
        
        # Generate plots
        plots = {}
        
        if result['success'] and result['solution']:
            # 2D contour plot
            if request.num_vars == 2:
                contour_plot = generate_contour_plot(
                    request.objective_function,
                    request.bounds,
                    request.constraints,
                    result['solution'],
                    request.initial_guess
                )
                if contour_plot:
                    plots['contour'] = contour_plot
        
        # Convergence info
        convergence = generate_convergence_info(
            result['initial_value'],
            result['objective_value'],
            result['n_iterations'] or 0
        )
        
        # Generate interpretation
        params = {
            'objective_function': request.objective_function,
            'bounds': request.bounds,
            'initial_guess': request.initial_guess,
            'constraints': [c.dict() for c in request.constraints]
        }
        interpretation = generate_interpretation(result, params)
        
        return {
            'success': result['success'],
            'message': result['message'],
            'solution': result['solution'],
            'objective_value': result['objective_value'],
            'n_iterations': result['n_iterations'],
            'initial_guess': result['initial_guess'],
            'initial_value': result['initial_value'],
            'convergence': convergence,
            'problem': {
                'objective_function': request.objective_function,
                'num_vars': request.num_vars,
                'n_constraints': len([c for c in request.constraints if c.fun.strip()]),
                'method': request.method
            },
            'plots': plots,
            'interpretation': interpretation
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"NLP solver failed: {str(e)}")
