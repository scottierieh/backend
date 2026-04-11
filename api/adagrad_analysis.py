"""
Adagrad Optimizer Router for FastAPI
Adaptive Gradient Algorithm for function optimization
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import io
import base64
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class AdagradRequest(BaseModel):
    """Adagrad optimization request parameters"""
    objective_function: str = Field(
        ...,
        description="Python expression for objective function using 'x' as variable array (e.g., 'np.sum(x**2)')"
    )
    bounds: List[List[float]] = Field(
        ...,
        description="Variable bounds as [[min1, max1], [min2, max2], ...]"
    )
    learning_rate: float = Field(
        default=0.01,
        ge=1e-6,
        le=10.0,
        description="Initial learning rate (step size)"
    )
    epsilon: float = Field(
        default=1e-8,
        ge=1e-12,
        le=1e-4,
        description="Small constant for numerical stability"
    )
    max_iter: int = Field(
        default=1000,
        ge=10,
        le=100000,
        description="Maximum number of iterations"
    )
    tolerance: float = Field(
        default=1e-8,
        ge=1e-12,
        le=1e-3,
        description="Convergence tolerance"
    )
    random_state: Optional[int] = Field(
        default=42,
        description="Random seed for reproducibility"
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


def _safe_eval_function(func_str: str, x: np.ndarray) -> float:
    """Safely evaluate objective function string."""
    safe_dict = {
        'np': np,
        'numpy': np,
        'x': x,
        'sum': np.sum,
        'abs': np.abs,
        'sqrt': np.sqrt,
        'exp': np.exp,
        'log': np.log,
        'log10': np.log10,
        'sin': np.sin,
        'cos': np.cos,
        'tan': np.tan,
        'power': np.power,
        'square': np.square,
        'pi': np.pi,
        'e': np.e
    }
    
    try:
        result = eval(func_str, {"__builtins__": {}}, safe_dict)
        return float(result)
    except Exception as e:
        raise ValueError(f"Failed to evaluate function: {str(e)}")


def numerical_gradient(func_str: str, x: np.ndarray, h: float = 1e-7) -> np.ndarray:
    """Compute numerical gradient using central difference method."""
    grad = np.zeros_like(x)
    
    for i in range(len(x)):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += h
        x_minus[i] -= h
        
        f_plus = _safe_eval_function(func_str, x_plus)
        f_minus = _safe_eval_function(func_str, x_minus)
        
        grad[i] = (f_plus - f_minus) / (2 * h)
    
    return grad


def adagrad_optimizer(
    func_str: str,
    bounds: List[List[float]],
    learning_rate: float = 0.01,
    epsilon: float = 1e-8,
    max_iter: int = 1000,
    tolerance: float = 1e-8,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Adagrad (Adaptive Gradient Algorithm) optimizer.
    
    Adagrad adapts the learning rate to the parameters, performing larger 
    updates for infrequent features and smaller updates for frequent features.
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_vars = len(bounds)
    bounds_arr = np.array(bounds)
    
    # Initialize: random starting point within bounds
    x = np.random.uniform(bounds_arr[:, 0], bounds_arr[:, 1])
    initial_x = x.copy()
    
    # Initialize accumulated squared gradients
    G = np.zeros(n_vars)
    
    # History tracking
    convergence = []
    solution_history = [x.copy()]
    gradient_norms = []
    effective_lr_history = []
    
    best_x = x.copy()
    best_fitness = _safe_eval_function(func_str, x)
    convergence.append(best_fitness)
    
    iterations_used = 0
    converged = False
    convergence_reason = "Maximum iterations reached"
    
    for iteration in range(max_iter):
        iterations_used = iteration + 1
        
        # Compute gradient
        grad = numerical_gradient(func_str, x)
        grad_norm = np.linalg.norm(grad)
        gradient_norms.append(_to_native_type(grad_norm))
        
        # Check gradient convergence
        if grad_norm < tolerance:
            converged = True
            convergence_reason = f"Gradient norm below tolerance ({grad_norm:.2e} < {tolerance:.2e})"
            break
        
        # Accumulate squared gradients
        G += grad ** 2
        
        # Compute adaptive learning rate
        adjusted_lr = learning_rate / (np.sqrt(G) + epsilon)
        effective_lr_history.append(_to_native_type(np.mean(adjusted_lr)))
        
        # Update parameters
        x_new = x - adjusted_lr * grad
        
        # Project onto bounds
        x_new = np.clip(x_new, bounds_arr[:, 0], bounds_arr[:, 1])
        
        # Check solution convergence
        if np.linalg.norm(x_new - x) < tolerance:
            converged = True
            convergence_reason = f"Solution change below tolerance"
            x = x_new
            break
        
        x = x_new
        solution_history.append(x.copy())
        
        # Evaluate fitness
        fitness = _safe_eval_function(func_str, x)
        convergence.append(_to_native_type(fitness))
        
        # Update best
        if fitness < best_fitness:
            best_fitness = fitness
            best_x = x.copy()
    
    # Final gradient info
    final_grad = numerical_gradient(func_str, best_x)
    
    return {
        'best_solution': [_to_native_type(v) for v in best_x],
        'best_fitness': _to_native_type(best_fitness),
        'initial_solution': [_to_native_type(v) for v in initial_x],
        'initial_fitness': _to_native_type(_safe_eval_function(func_str, initial_x)),
        'convergence': convergence,
        'solution_history': [[_to_native_type(v) for v in s] for s in solution_history],
        'gradient_norms': gradient_norms,
        'effective_lr_history': effective_lr_history,
        'final_gradient': [_to_native_type(v) for v in final_grad],
        'final_gradient_norm': _to_native_type(np.linalg.norm(final_grad)),
        'accumulated_gradients': [_to_native_type(v) for v in G],
        'iterations_used': iterations_used,
        'converged': converged,
        'convergence_reason': convergence_reason,
        'n_vars': n_vars
    }


def generate_convergence_plot(convergence: List[float], title: str = "Convergence") -> str:
    """Generate convergence plot"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    iterations = list(range(1, len(convergence) + 1))
    
    ax.plot(iterations, convergence, 'b-', linewidth=2, label='Objective Value')
    ax.scatter([1], [convergence[0]], color='red', s=100, zorder=5, label=f'Start: {convergence[0]:.4f}')
    ax.scatter([len(convergence)], [convergence[-1]], color='green', s=100, zorder=5, label=f'End: {convergence[-1]:.6f}')
    
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Objective Value (Fitness)', fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    if len(convergence) > 10 and convergence[0] > 0 and convergence[-1] > 0:
        if convergence[0] / convergence[-1] > 100:
            ax.set_yscale('log')
            ax.set_ylabel('Objective Value (log scale)', fontsize=11)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_gradient_norm_plot(gradient_norms: List[float]) -> str:
    """Generate gradient norm history plot"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    iterations = list(range(1, len(gradient_norms) + 1))
    
    # Filter out zero or negative values for log scale
    valid_norms = [max(n, 1e-15) for n in gradient_norms]
    
    ax.semilogy(iterations, valid_norms, 'r-', linewidth=2, label='Gradient Norm')
    ax.axhline(y=1e-8, color='gray', linestyle='--', label='Typical Tolerance (1e-8)')
    
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Gradient Norm (log scale)', fontsize=11)
    ax.set_title('Gradient Norm History', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_learning_rate_plot(effective_lr: List[float], initial_lr: float) -> str:
    """Generate effective learning rate plot"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    iterations = list(range(1, len(effective_lr) + 1))
    
    # Filter for log scale
    valid_lr = [max(lr, 1e-15) for lr in effective_lr]
    
    ax.semilogy(iterations, valid_lr, 'g-', linewidth=2, label='Effective LR (mean)')
    ax.axhline(y=initial_lr, color='blue', linestyle='--', alpha=0.7, label=f'Initial LR: {initial_lr}')
    
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Learning Rate (log scale)', fontsize=11)
    ax.set_title('Adaptive Learning Rate History', fontsize=13, fontweight='bold')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_2d_path_plot(
    func_str: str,
    solution_history: List[List[float]],
    bounds: List[List[float]]
) -> str:
    """Generate 2D optimization path plot"""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create meshgrid for contour
    x_range = np.linspace(bounds[0][0], bounds[0][1], 100)
    y_range = np.linspace(bounds[1][0], bounds[1][1], 100)
    X, Y = np.meshgrid(x_range, y_range)
    
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            try:
                Z[i, j] = _safe_eval_function(func_str, np.array([X[i, j], Y[i, j]]))
            except:
                Z[i, j] = np.nan
    
    # Contour plot
    levels = np.linspace(np.nanmin(Z), np.nanpercentile(Z, 95), 30)
    contour = ax.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.8)
    ax.contour(X, Y, Z, levels=levels, colors='white', alpha=0.3, linewidths=0.5)
    plt.colorbar(contour, ax=ax, label='Objective Value')
    
    # Plot optimization path
    path = np.array(solution_history)
    ax.plot(path[:, 0], path[:, 1], 'r.-', linewidth=1.5, markersize=3, alpha=0.7, label='Optimization Path')
    ax.scatter(path[0, 0], path[0, 1], color='blue', s=150, marker='o', edgecolors='white', linewidths=2, zorder=5, label='Start')
    ax.scatter(path[-1, 0], path[-1, 1], color='red', s=200, marker='*', edgecolors='white', linewidths=2, zorder=5, label='Optimum')
    
    ax.set_xlabel('x[0]', fontsize=11)
    ax.set_ylabel('x[1]', fontsize=11)
    ax.set_title('Optimization Path (2D Contour)', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_3d_surface_plot(
    func_str: str,
    solution_history: List[List[float]],
    bounds: List[List[float]]
) -> str:
    """Generate 3D surface plot with optimization path"""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create meshgrid
    x_range = np.linspace(bounds[0][0], bounds[0][1], 80)
    y_range = np.linspace(bounds[1][0], bounds[1][1], 80)
    X, Y = np.meshgrid(x_range, y_range)
    
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            try:
                Z[i, j] = _safe_eval_function(func_str, np.array([X[i, j], Y[i, j]]))
            except:
                Z[i, j] = np.nan
    
    # Surface plot
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.7, linewidth=0, antialiased=True)
    
    # Plot optimization path
    path = np.array(solution_history)
    z_path = []
    for point in path:
        try:
            z_path.append(_safe_eval_function(func_str, np.array(point[:2])))
        except:
            z_path.append(np.nan)
    z_path = np.array(z_path)
    
    ax.plot(path[:, 0], path[:, 1], z_path, 'r-', linewidth=2, label='Path')
    ax.scatter(path[0, 0], path[0, 1], z_path[0], color='blue', s=100, marker='o', label='Start')
    ax.scatter(path[-1, 0], path[-1, 1], z_path[-1], color='red', s=200, marker='*', label='Optimum')
    
    ax.set_xlabel('x[0]', fontsize=10)
    ax.set_ylabel('x[1]', fontsize=10)
    ax.set_zlabel('f(x)', fontsize=10)
    ax.set_title('3D Surface with Optimization Path', fontsize=13, fontweight='bold')
    ax.legend(loc='upper left')
    
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, label='Objective Value')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_3d_wireframe_plot(
    func_str: str,
    best_solution: List[float],
    bounds: List[List[float]]
) -> str:
    """Generate 3D wireframe plot"""
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    x_range = np.linspace(bounds[0][0], bounds[0][1], 50)
    y_range = np.linspace(bounds[1][0], bounds[1][1], 50)
    X, Y = np.meshgrid(x_range, y_range)
    
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            try:
                Z[i, j] = _safe_eval_function(func_str, np.array([X[i, j], Y[i, j]]))
            except:
                Z[i, j] = np.nan
    
    ax.plot_wireframe(X, Y, Z, color='steelblue', alpha=0.6, linewidth=0.5)
    
    # Mark optimum
    z_opt = _safe_eval_function(func_str, np.array(best_solution[:2]))
    ax.scatter([best_solution[0]], [best_solution[1]], [z_opt], 
               color='red', s=200, marker='*', label=f'Optimum: {z_opt:.6f}')
    
    ax.set_xlabel('x[0]', fontsize=10)
    ax.set_ylabel('x[1]', fontsize=10)
    ax.set_zlabel('f(x)', fontsize=10)
    ax.set_title('3D Wireframe Plot', fontsize=13, fontweight='bold')
    ax.legend()
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_variable_history_plot(solution_history: List[List[float]], n_vars: int) -> str:
    """Generate variable values over iterations plot"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    path = np.array(solution_history)
    iterations = list(range(1, len(path) + 1))
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_vars))
    
    for i in range(min(n_vars, 10)):  # Plot max 10 variables
        ax.plot(iterations, path[:, i], '-', color=colors[i], linewidth=2, label=f'x[{i}]')
    
    ax.set_xlabel('Iteration', fontsize=11)
    ax.set_ylabel('Variable Value', fontsize=11)
    ax.set_title('Variable Values Over Iterations', fontsize=13, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_interpretation(result: Dict, params: Dict) -> Dict[str, Any]:
    """Generate detailed interpretation of Adagrad optimization results"""
    key_insights = []
    
    # 1. Convergence Quality Analysis
    initial_fitness = result['initial_fitness']
    best_fitness = result['best_fitness']
    improvement = initial_fitness - best_fitness
    improvement_pct = (improvement / abs(initial_fitness)) * 100 if initial_fitness != 0 else 0
    
    if result['converged']:
        conv_status = 'positive'
        conv_title = 'Successful Convergence'
        conv_desc = f"Optimization converged after {result['iterations_used']} iterations. {result['convergence_reason']}."
    else:
        conv_status = 'warning'
        conv_title = 'Maximum Iterations Reached'
        conv_desc = f"Optimization stopped at {result['iterations_used']} iterations without full convergence. Consider increasing max_iter or adjusting learning rate."
    
    key_insights.append({
        'title': conv_title,
        'description': conv_desc,
        'status': conv_status
    })
    
    # 2. Objective Function Improvement
    if improvement_pct > 90:
        imp_status = 'positive'
        imp_quality = 'Excellent improvement'
    elif improvement_pct > 50:
        imp_status = 'positive'
        imp_quality = 'Good improvement'
    elif improvement_pct > 10:
        imp_status = 'neutral'
        imp_quality = 'Moderate improvement'
    else:
        imp_status = 'warning'
        imp_quality = 'Limited improvement'
    
    key_insights.append({
        'title': 'Objective Function Improvement',
        'description': f'{imp_quality}. Initial value: {initial_fitness:.6f} → Final value: {best_fitness:.6f}. Total reduction: {improvement:.6f} ({improvement_pct:.2f}%)',
        'status': imp_status
    })
    
    # 3. Gradient Analysis
    final_grad_norm = result['final_gradient_norm']
    if final_grad_norm < 1e-6:
        grad_status = 'positive'
        grad_desc = f'Gradient norm is very small ({final_grad_norm:.2e}), indicating the solution is at or very near a local minimum.'
    elif final_grad_norm < 1e-4:
        grad_status = 'positive'
        grad_desc = f'Gradient norm is small ({final_grad_norm:.2e}), suggesting good convergence to a stationary point.'
    elif final_grad_norm < 1e-2:
        grad_status = 'neutral'
        grad_desc = f'Gradient norm is moderate ({final_grad_norm:.2e}). The solution may not be exactly at a minimum.'
    else:
        grad_status = 'warning'
        grad_desc = f'Gradient norm is relatively large ({final_grad_norm:.2e}). Consider running more iterations or adjusting parameters.'
    
    key_insights.append({
        'title': 'Gradient Analysis',
        'description': grad_desc,
        'status': grad_status
    })
    
    # 4. Adaptive Learning Rate Behavior
    if result['effective_lr_history']:
        initial_eff_lr = result['effective_lr_history'][0]
        final_eff_lr = result['effective_lr_history'][-1]
        lr_reduction = (1 - final_eff_lr / initial_eff_lr) * 100 if initial_eff_lr > 0 else 0
        
        key_insights.append({
            'title': 'Adaptive Learning Rate',
            'description': f'Effective learning rate decreased from {initial_eff_lr:.6f} to {final_eff_lr:.2e} ({lr_reduction:.1f}% reduction). This adaptive behavior helps prevent oscillation near the optimum.',
            'status': 'neutral'
        })
    
    # 5. Solution Quality
    best_solution = result['best_solution']
    sol_str = ', '.join([f'x[{i}]={v:.6f}' for i, v in enumerate(best_solution)])
    
    # Check if solution is near bounds
    bounds_arr = np.array(params['bounds'])
    near_bound = False
    for i, (val, (lb, ub)) in enumerate(zip(best_solution, bounds_arr)):
        if abs(val - lb) < 0.01 * (ub - lb) or abs(val - ub) < 0.01 * (ub - lb):
            near_bound = True
            break
    
    if near_bound:
        sol_status = 'warning'
        sol_desc = f'Optimal solution found: {sol_str}. Note: Solution is near variable bounds. Consider expanding the search space.'
    else:
        sol_status = 'positive'
        sol_desc = f'Optimal solution found: {sol_str}. Solution is well within the specified bounds.'
    
    key_insights.append({
        'title': 'Optimal Solution',
        'description': sol_desc,
        'status': sol_status
    })
    
    # 6. Accumulated Gradient Analysis (Adagrad-specific)
    acc_grads = result['accumulated_gradients']
    max_acc = max(acc_grads) if acc_grads else 0
    min_acc = min(acc_grads) if acc_grads else 0
    
    if max_acc > 0 and min_acc > 0:
        ratio = max_acc / min_acc
        if ratio > 100:
            acc_desc = f'Large variation in accumulated gradients (ratio: {ratio:.1f}). Some dimensions had much more gradient activity than others, which Adagrad handles well.'
        else:
            acc_desc = f'Relatively uniform accumulated gradients across dimensions (ratio: {ratio:.1f}). Gradient activity was similar across all variables.'
    else:
        acc_desc = 'Accumulated gradient analysis not available.'
    
    key_insights.append({
        'title': 'Adagrad Adaptation',
        'description': acc_desc,
        'status': 'neutral'
    })
    
    # Generate recommendations
    recommendations = []
    
    if not result['converged']:
        recommendations.append("Increase max_iter to allow more iterations for convergence.")
        recommendations.append("Try a larger initial learning_rate for faster initial progress.")
    
    if final_grad_norm > 1e-4:
        recommendations.append("Decrease tolerance if higher precision is needed.")
        recommendations.append("Verify the objective function is smooth and differentiable.")
    
    if near_bound:
        recommendations.append("Expand variable bounds to allow exploration of a wider search space.")
    
    if improvement_pct < 50 and not near_bound:
        recommendations.append("Try different initial random states to explore other starting points.")
        recommendations.append("Consider if the objective function has multiple local minima.")
    
    if not recommendations:
        recommendations.append("Optimization completed successfully with good results.")
        recommendations.append("Current parameters appear well-suited for this problem.")
    
    return {
        'key_insights': key_insights,
        'recommendations': recommendations,
        'summary': {
            'converged': result['converged'],
            'iterations': result['iterations_used'],
            'initial_fitness': _to_native_type(initial_fitness),
            'final_fitness': _to_native_type(best_fitness),
            'improvement_pct': _to_native_type(improvement_pct),
            'final_gradient_norm': _to_native_type(final_grad_norm)
        }
    }


@router.post("/adagrad")
async def run_adagrad_optimization(request: AdagradRequest) -> Dict[str, Any]:
    """
    Run Adagrad (Adaptive Gradient Algorithm) optimization.
    
    Adagrad is a gradient-based optimization algorithm that adapts the learning
    rate to each parameter, performing larger updates for infrequent features
    and smaller updates for frequent features. This makes it well-suited for
    dealing with sparse data and problems where different parameters have
    different scales.
    
    Algorithm Details:
    -----------------
    1. Initialize accumulated squared gradient G = 0
    2. For each iteration:
       - Compute gradient g = ∇f(x)
       - Accumulate squared gradient: G = G + g²
       - Update parameters: x = x - (η / √(G + ε)) * g
    
    Key Features:
    - Automatic learning rate adaptation per parameter
    - No manual learning rate tuning needed
    - Good for sparse data and non-stationary objectives
    
    Parameters
    ----------
    objective_function : str
        Python expression for the objective function to minimize.
        Use 'x' to reference the variable array.
        Examples: 'np.sum(x**2)', '(x[0]-1)**2 + (x[1]-2)**2'
    
    bounds : List[List[float]]
        Variable bounds as [[min1, max1], [min2, max2], ...]
        
    learning_rate : float
        Initial learning rate (default: 0.01)
        
    epsilon : float
        Small constant for numerical stability (default: 1e-8)
        
    max_iter : int
        Maximum number of iterations (default: 1000)
        
    tolerance : float
        Convergence tolerance (default: 1e-8)
        
    Returns
    -------
    Dict containing:
        - best_solution: Optimal variable values
        - best_fitness: Minimum objective function value
        - convergence: Fitness history over iterations
        - plots: Visualization plots (convergence, 3D surface, path, etc.)
        - interpretation: Detailed analysis of results
    """
    try:
        # Validate inputs
        if not request.bounds or len(request.bounds) == 0:
            raise HTTPException(status_code=400, detail="Bounds must be provided.")
        
        for i, bound in enumerate(request.bounds):
            if len(bound) != 2:
                raise HTTPException(status_code=400, detail=f"Bound {i} must have exactly 2 values [min, max].")
            if bound[0] >= bound[1]:
                raise HTTPException(status_code=400, detail=f"Bound {i}: min ({bound[0]}) must be less than max ({bound[1]}).")
        
        # Test objective function
        n_vars = len(request.bounds)
        test_x = np.zeros(n_vars)
        try:
            _safe_eval_function(request.objective_function, test_x)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid objective function: {str(e)}")
        
        # Run optimization
        result = adagrad_optimizer(
            func_str=request.objective_function,
            bounds=request.bounds,
            learning_rate=request.learning_rate,
            epsilon=request.epsilon,
            max_iter=request.max_iter,
            tolerance=request.tolerance,
            random_state=request.random_state
        )
        
        # Generate plots
        plots = {}
        
        # Convergence plot
        plots['convergence_plot'] = generate_convergence_plot(
            result['convergence'],
            title='Adagrad Optimization Convergence'
        )
        
        # Gradient norm plot
        if result['gradient_norms']:
            plots['gradient_norm_plot'] = generate_gradient_norm_plot(result['gradient_norms'])
        
        # Learning rate plot
        if result['effective_lr_history']:
            plots['learning_rate_plot'] = generate_learning_rate_plot(
                result['effective_lr_history'],
                request.learning_rate
            )
        
        # Variable history plot
        if len(result['solution_history']) > 1:
            plots['variable_history_plot'] = generate_variable_history_plot(
                result['solution_history'],
                result['n_vars']
            )
        
        # 2D and 3D plots (only for 2D problems)
        if n_vars == 2 and len(result['solution_history']) > 1:
            try:
                plots['path_2d_plot'] = generate_2d_path_plot(
                    request.objective_function,
                    result['solution_history'],
                    request.bounds
                )
                plots['surface_3d_plot'] = generate_3d_surface_plot(
                    request.objective_function,
                    result['solution_history'],
                    request.bounds
                )
                plots['wireframe_3d_plot'] = generate_3d_wireframe_plot(
                    request.objective_function,
                    result['best_solution'],
                    request.bounds
                )
            except Exception as e:
                # Continue without 3D plots if they fail
                pass
        
        # Generate interpretation
        params = {
            'bounds': request.bounds,
            'learning_rate': request.learning_rate,
            'epsilon': request.epsilon,
            'max_iter': request.max_iter,
            'tolerance': request.tolerance
        }
        interpretation = generate_interpretation(result, params)
        
        # Build response
        response = {
            'best_solution': result['best_solution'],
            'best_fitness': result['best_fitness'],
            'initial_solution': result['initial_solution'],
            'initial_fitness': result['initial_fitness'],
            'convergence': result['convergence'],
            'iterations_used': result['iterations_used'],
            'converged': result['converged'],
            'convergence_reason': result['convergence_reason'],
            'n_vars': result['n_vars'],
            'final_gradient': result['final_gradient'],
            'final_gradient_norm': result['final_gradient_norm'],
            'accumulated_gradients': result['accumulated_gradients'],
            'parameters': {
                'objective_function': request.objective_function,
                'bounds': request.bounds,
                'learning_rate': request.learning_rate,
                'epsilon': request.epsilon,
                'max_iter': request.max_iter,
                'tolerance': request.tolerance,
                'random_state': request.random_state
            },
            'plots': plots,
            'interpretation': interpretation
        }
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Adagrad optimization failed: {str(e)}")
