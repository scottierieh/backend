"""
Gradient Descent Simulation Router for FastAPI
Visualize gradient descent optimization on functions
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


class GradientDescentRequest(BaseModel):
    """Gradient Descent simulation request parameters"""
    objective_function: str = Field(
        default="x1^2 + x2^2",
        description="Objective function in math notation"
    )
    learning_rate: float = Field(
        default=0.1,
        ge=0.001,
        le=2.0,
        description="Learning rate (step size)"
    )
    start_x: float = Field(
        default=4.0,
        description="Starting x coordinate"
    )
    start_y: float = Field(
        default=4.0,
        description="Starting y coordinate"
    )
    num_steps: int = Field(
        default=50,
        ge=1,
        le=500,
        description="Number of gradient descent steps"
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


def math_to_python(expr: str) -> str:
    """Convert math notation to Python expression"""
    return expr \
        .replace('x1', 'x') \
        .replace('x2', 'y') \
        .replace('^', '**') \
        .replace('pi', 'np.pi') \
        .replace('sin(', 'np.sin(') \
        .replace('cos(', 'np.cos(') \
        .replace('exp(', 'np.exp(') \
        .replace('sqrt(', 'np.sqrt(') \
        .replace('log(', 'np.log(')


def create_function_and_gradient(func_str: str):
    """Create function and its numerical gradient"""
    python_expr = math_to_python(func_str)
    
    def f(x, y):
        return eval(python_expr, {"__builtins__": {}, "np": np, "x": x, "y": y})
    
    def grad_f(x, y, h=1e-7):
        df_dx = (f(x + h, y) - f(x - h, y)) / (2 * h)
        df_dy = (f(x, y + h) - f(x, y - h)) / (2 * h)
        return np.array([df_dx, df_dy])
    
    return f, grad_f


def run_gradient_descent(
    func_str: str,
    learning_rate: float,
    start_x: float,
    start_y: float,
    num_steps: int
) -> Dict[str, Any]:
    """Run gradient descent simulation"""
    
    f, grad_f = create_function_and_gradient(func_str)
    
    # Initialize
    path = []
    gradients = []
    current_pos = np.array([start_x, start_y])
    
    initial_value = f(current_pos[0], current_pos[1])
    path.append([
        _to_native_type(current_pos[0]),
        _to_native_type(current_pos[1]),
        _to_native_type(initial_value)
    ])
    
    # Run Gradient Descent
    for step in range(num_steps):
        gradient = grad_f(current_pos[0], current_pos[1])
        gradients.append([_to_native_type(gradient[0]), _to_native_type(gradient[1])])
        
        current_pos = current_pos - learning_rate * gradient
        value = f(current_pos[0], current_pos[1])
        
        path.append([
            _to_native_type(current_pos[0]),
            _to_native_type(current_pos[1]),
            _to_native_type(value)
        ])
        
        # Check for convergence
        if np.linalg.norm(gradient) < 1e-10:
            break
    
    final_pos = path[-1]
    initial_pos = path[0]
    
    return {
        "path": path,
        "gradients": gradients,
        "initial_position": initial_pos,
        "final_position": final_pos,
        "improvement": _to_native_type(initial_pos[2] - final_pos[2]),
        "improvement_pct": _to_native_type((initial_pos[2] - final_pos[2]) / abs(initial_pos[2]) * 100) if initial_pos[2] != 0 else 0,
        "steps_taken": len(path) - 1,
        "converged": len(path) - 1 < num_steps
    }


def generate_3d_surface_plot(func_str: str, path: List[List[float]]) -> str:
    """Generate 3D surface plot with optimization path"""
    f, _ = create_function_and_gradient(func_str)
    
    # Determine plot range based on path
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    
    x_min, x_max = min(xs) - 1, max(xs) + 1
    y_min, y_max = min(ys) - 1, max(ys) + 1
    
    # Ensure symmetric range for better visualization
    max_range = max(abs(x_min), abs(x_max), abs(y_min), abs(y_max))
    x_min, x_max = -max_range, max_range
    y_min, y_max = -max_range, max_range
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create surface
    x_range = np.linspace(x_min, x_max, 50)
    y_range = np.linspace(y_min, y_max, 50)
    X, Y = np.meshgrid(x_range, y_range)
    
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            try:
                Z[i, j] = f(X[i, j], Y[i, j])
            except:
                Z[i, j] = np.nan
    
    # Plot surface
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.6, linewidth=0)
    
    # Plot path
    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]
    path_z = [p[2] for p in path]
    
    ax.plot(path_x, path_y, path_z, 'r-', linewidth=2, label='Gradient Descent Path')
    ax.scatter(path_x[0], path_y[0], path_z[0], color='blue', s=100, marker='o', label='Start')
    ax.scatter(path_x[-1], path_y[-1], path_z[-1], color='red', s=150, marker='*', label='End')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('f(x, y)')
    ax.set_title('Gradient Descent on Surface')
    ax.legend()
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_surface_data(func_str: str, path: List[List[float]]) -> Dict[str, Any]:
    """Generate surface mesh data for Plotly 3D interactive visualization"""
    f, _ = create_function_and_gradient(func_str)
    
    # Determine plot range based on path
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    
    max_range = max(abs(min(xs)) + 1, abs(max(xs)) + 1, abs(min(ys)) + 1, abs(max(ys)) + 1)
    
    # Create mesh grid
    resolution = 40
    x_range = np.linspace(-max_range, max_range, resolution)
    y_range = np.linspace(-max_range, max_range, resolution)
    
    z_data = []
    for y_val in y_range:
        row = []
        for x_val in x_range:
            try:
                z_val = f(x_val, y_val)
                row.append(_to_native_type(z_val))
            except:
                row.append(None)
        z_data.append(row)
    
    return {
        'x': [_to_native_type(v) for v in x_range],
        'y': [_to_native_type(v) for v in y_range],
        'z': z_data
    }


def generate_2d_contour_plot(func_str: str, path: List[List[float]]) -> str:
    """Generate 2D contour plot with optimization path"""
    f, _ = create_function_and_gradient(func_str)
    
    # Determine plot range
    xs = [p[0] for p in path]
    ys = [p[1] for p in path]
    
    max_range = max(abs(min(xs)) + 1, abs(max(xs)) + 1, abs(min(ys)) + 1, abs(max(ys)) + 1)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    x_range = np.linspace(-max_range, max_range, 100)
    y_range = np.linspace(-max_range, max_range, 100)
    X, Y = np.meshgrid(x_range, y_range)
    
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            try:
                Z[i, j] = f(X[i, j], Y[i, j])
            except:
                Z[i, j] = np.nan
    
    # Contour plot
    levels = np.linspace(np.nanmin(Z), np.nanpercentile(Z, 90), 25)
    contour = ax.contourf(X, Y, Z, levels=levels, cmap='viridis', alpha=0.8)
    ax.contour(X, Y, Z, levels=levels, colors='white', alpha=0.3, linewidths=0.5)
    plt.colorbar(contour, ax=ax, label='f(x, y)')
    
    # Plot path
    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]
    
    ax.plot(path_x, path_y, 'r.-', linewidth=1.5, markersize=4, label='Path')
    ax.scatter(path_x[0], path_y[0], color='blue', s=100, marker='o', zorder=5, label='Start')
    ax.scatter(path_x[-1], path_y[-1], color='red', s=150, marker='*', zorder=5, label='End')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Gradient Descent Path (Contour View)')
    ax.legend()
    ax.set_aspect('equal')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_convergence_plot(path: List[List[float]]) -> str:
    """Generate convergence plot"""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    values = [p[2] for p in path]
    steps = list(range(len(values)))
    
    ax.plot(steps, values, 'b-', linewidth=2)
    ax.scatter([0], [values[0]], color='blue', s=100, zorder=5, label=f'Start: {values[0]:.4f}')
    ax.scatter([len(values)-1], [values[-1]], color='red', s=100, zorder=5, label=f'End: {values[-1]:.6f}')
    
    ax.set_xlabel('Step')
    ax.set_ylabel('f(x, y)')
    ax.set_title('Convergence History')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    
    if values[0] > 0 and values[-1] > 0 and values[0] / values[-1] > 10:
        ax.set_yscale('log')
        ax.set_ylabel('f(x, y) (log scale)')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_interpretation(result: Dict, params: Dict) -> Dict[str, Any]:
    """Generate interpretation of gradient descent results"""
    key_insights = []
    
    # Convergence analysis
    improvement_pct = result['improvement_pct']
    if improvement_pct > 99:
        key_insights.append({
            'title': 'Excellent Convergence',
            'description': f"Gradient descent achieved {improvement_pct:.1f}% reduction in objective value, reaching near-optimal solution.",
            'status': 'positive'
        })
    elif improvement_pct > 80:
        key_insights.append({
            'title': 'Good Convergence',
            'description': f"Achieved {improvement_pct:.1f}% reduction. The algorithm found a good solution.",
            'status': 'positive'
        })
    else:
        key_insights.append({
            'title': 'Partial Convergence',
            'description': f"Only {improvement_pct:.1f}% reduction achieved. Consider more steps or adjusting learning rate.",
            'status': 'warning'
        })
    
    # Learning rate analysis
    lr = params['learning_rate']
    if lr > 0.5:
        key_insights.append({
            'title': 'High Learning Rate',
            'description': f"Learning rate {lr} is relatively high. May converge fast but could overshoot the minimum.",
            'status': 'neutral'
        })
    elif lr < 0.01:
        key_insights.append({
            'title': 'Low Learning Rate',
            'description': f"Learning rate {lr} is small. Stable but slow convergence. Consider increasing if not converging.",
            'status': 'neutral'
        })
    
    # Final position
    final = result['final_position']
    key_insights.append({
        'title': 'Final Solution',
        'description': f"Converged to ({final[0]:.6f}, {final[1]:.6f}) with f(x,y) = {final[2]:.6f}",
        'status': 'neutral'
    })
    
    # Recommendations
    recommendations = []
    if improvement_pct < 95:
        recommendations.append("Increase number of steps for better convergence.")
    if result['converged']:
        recommendations.append("Algorithm converged early. Solution is likely optimal.")
    if lr > 0.3 and improvement_pct < 80:
        recommendations.append("Try reducing learning rate for more stable convergence.")
    if not recommendations:
        recommendations.append("Current parameters work well for this function.")
    
    return {
        'key_insights': key_insights,
        'recommendations': recommendations
    }


@router.post("/gradient-descent")
async def run_gradient_descent_simulation(request: GradientDescentRequest) -> Dict[str, Any]:
    """
    Run Gradient Descent simulation.
    
    Gradient Descent is a first-order iterative optimization algorithm
    for finding a local minimum of a differentiable function.
    
    Update rule: x_{n+1} = x_n - η * ∇f(x_n)
    
    where η is the learning rate and ∇f is the gradient.
    """
    try:
        # Validate function
        try:
            f, _ = create_function_and_gradient(request.objective_function)
            f(0, 0)  # Test evaluation
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid objective function: {str(e)}")
        
        # Run gradient descent
        result = run_gradient_descent(
            func_str=request.objective_function,
            learning_rate=request.learning_rate,
            start_x=request.start_x,
            start_y=request.start_y,
            num_steps=request.num_steps
        )
        
        # Generate plots
        plots = {
            'surface_3d': generate_3d_surface_plot(request.objective_function, result['path']),
            'contour_2d': generate_2d_contour_plot(request.objective_function, result['path']),
            'convergence': generate_convergence_plot(result['path'])
        }
        
        # Generate surface data for Plotly interactive 3D
        surface_data = generate_surface_data(request.objective_function, result['path'])
        
        # Generate interpretation
        params = {
            'learning_rate': request.learning_rate,
            'start_x': request.start_x,
            'start_y': request.start_y,
            'num_steps': request.num_steps
        }
        interpretation = generate_interpretation(result, params)
        
        return {
            'path': result['path'],
            'initial_position': result['initial_position'],
            'final_position': result['final_position'],
            'improvement': result['improvement'],
            'improvement_pct': result['improvement_pct'],
            'steps_taken': result['steps_taken'],
            'converged': result['converged'],
            'function_expression': request.objective_function,
            'parameters': params,
            'surface_data': surface_data,
            'plots': plots,
            'interpretation': interpretation
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gradient descent simulation failed: {str(e)}")
