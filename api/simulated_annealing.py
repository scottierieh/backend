"""
Simulated Annealing FastAPI Endpoint
Metaheuristic optimization inspired by metallurgical annealing
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

router = APIRouter()


class SARequest(BaseModel):
    """Request model for Simulated Annealing"""
    objective_function: str = Field(..., description="Python expression to minimize (e.g., 'np.sum(x**2)')")
    bounds: List[List[float]] = Field(..., description="Variable bounds [[min1, max1], [min2, max2], ...]")
    initial_temp: float = Field(default=1000.0, ge=1.0, le=10000.0)
    cooling_rate: float = Field(default=0.99, ge=0.8, le=0.9999)
    max_iter: int = Field(default=1000, ge=100, le=10000)


class SAResponse(BaseModel):
    """Response model for Simulated Annealing"""
    success: bool
    best_solution: List[float]
    best_fitness: float
    convergence_history: List[float]
    temperature_history: List[float]
    acceptance_rate: float
    iterations: int
    problem_info: Dict[str, Any]
    plots: Dict[str, str]
    interpretation: Dict[str, Any]


def evaluate_objective(func_str: str, x: np.ndarray) -> float:
    """Safely evaluate objective function"""
    try:
        # Create safe namespace
        namespace = {
            'np': np,
            'x': x,
            'abs': abs,
            'sum': sum,
            'max': max,
            'min': min,
            'sqrt': np.sqrt,
            'exp': np.exp,
            'log': np.log,
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan
        }
        result = eval(func_str, {"__builtins__": {}}, namespace)
        return float(result)
    except Exception as e:
        raise ValueError(f"Error evaluating function: {str(e)}")


def simulated_annealing(
    objective_func: str,
    bounds: np.ndarray,
    initial_temp: float,
    cooling_rate: float,
    max_iter: int
):
    """
    Simulated Annealing optimization
    
    Mimics the metallurgical process of annealing:
    - High temperature: Accept many worse solutions (exploration)
    - Low temperature: Accept fewer worse solutions (exploitation)
    """
    n_vars = len(bounds)
    
    # Initialize random solution
    current_solution = np.array([
        np.random.uniform(bounds[i][0], bounds[i][1])
        for i in range(n_vars)
    ])
    current_fitness = evaluate_objective(objective_func, current_solution)
    
    # Best solution tracking
    best_solution = current_solution.copy()
    best_fitness = current_fitness
    
    # History tracking
    convergence_history = [best_fitness]
    temperature_history = [initial_temp]
    
    # Statistics
    temperature = initial_temp
    accepted_moves = 0
    total_moves = 0
    
    for iteration in range(max_iter):
        # Generate neighbor solution (perturbation)
        neighbor = current_solution.copy()
        
        # Perturb random dimension
        dim = np.random.randint(0, n_vars)
        step_size = (bounds[dim][1] - bounds[dim][0]) * 0.1 * (temperature / initial_temp)
        neighbor[dim] += np.random.uniform(-step_size, step_size)
        
        # Enforce bounds
        neighbor[dim] = np.clip(neighbor[dim], bounds[dim][0], bounds[dim][1])
        
        # Evaluate neighbor
        neighbor_fitness = evaluate_objective(objective_func, neighbor)
        
        # Calculate acceptance probability
        delta = neighbor_fitness - current_fitness
        
        if delta < 0:
            # Better solution - always accept
            accept = True
        else:
            # Worse solution - accept with probability
            acceptance_prob = np.exp(-delta / temperature)
            accept = np.random.random() < acceptance_prob
        
        # Update solution
        total_moves += 1
        if accept:
            current_solution = neighbor
            current_fitness = neighbor_fitness
            accepted_moves += 1
            
            # Update best
            if current_fitness < best_fitness:
                best_solution = current_solution.copy()
                best_fitness = current_fitness
        
        # Cool down temperature
        temperature *= cooling_rate
        
        # Record history
        convergence_history.append(best_fitness)
        temperature_history.append(temperature)
    
    acceptance_rate = accepted_moves / total_moves if total_moves > 0 else 0
    
    return (
        best_solution,
        best_fitness,
        convergence_history,
        temperature_history,
        acceptance_rate
    )


def create_convergence_plot(convergence: List[float]) -> str:
    """Create convergence plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = range(len(convergence))
    ax.plot(iterations, convergence, linewidth=2, color='steelblue', label='Best Fitness')
    
    ax.set_xlabel('Iteration', fontsize=12, weight='bold')
    ax.set_ylabel('Fitness (Lower is Better)', fontsize=12, weight='bold')
    ax.set_title('Simulated Annealing - Convergence', fontsize=14, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Highlight final value
    ax.axhline(convergence[-1], color='red', linestyle='--', 
               linewidth=2, alpha=0.5, label=f'Final: {convergence[-1]:.4f}')
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return img_str


def create_temperature_plot(temperature: List[float]) -> str:
    """Create temperature schedule plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = range(len(temperature))
    ax.plot(iterations, temperature, linewidth=2, color='orangered', label='Temperature')
    
    ax.set_xlabel('Iteration', fontsize=12, weight='bold')
    ax.set_ylabel('Temperature', fontsize=12, weight='bold')
    ax.set_title('Temperature Schedule (Cooling)', fontsize=14, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return img_str


def create_3d_surface_plot(func_str: str, bounds: np.ndarray, best_solution: np.ndarray) -> Optional[str]:
    """Create 3D surface plot for 2D problems"""
    if len(bounds) != 2:
        return None
    
    try:
        # Create mesh
        x_range = np.linspace(bounds[0][0], bounds[0][1], 50)
        y_range = np.linspace(bounds[1][0], bounds[1][1], 50)
        X, Y = np.meshgrid(x_range, y_range)
        
        # Evaluate function
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = evaluate_objective(func_str, np.array([X[i, j], Y[i, j]]))
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, edgecolor='none')
        
        # Mark best solution
        best_z = evaluate_objective(func_str, best_solution)
        ax.scatter([best_solution[0]], [best_solution[1]], [best_z], 
                  color='red', s=200, marker='*', label='Best Solution', zorder=10)
        
        ax.set_xlabel('x[0]', fontsize=11, weight='bold')
        ax.set_ylabel('x[1]', fontsize=11, weight='bold')
        ax.set_zlabel('f(x)', fontsize=11, weight='bold')
        ax.set_title('Objective Function Landscape', fontsize=13, weight='bold')
        ax.legend()
        
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return img_str
    except:
        return None


def generate_interpretation(
    best_fitness: float,
    convergence: List[float],
    acceptance_rate: float,
    initial_temp: float,
    cooling_rate: float
) -> Dict[str, Any]:
    """Generate insights and recommendations"""
    
    key_insights = []
    recommendations = []
    
    # Convergence analysis
    improvement = convergence[0] - convergence[-1]
    if improvement > 0:
        pct_improvement = (improvement / abs(convergence[0])) * 100
        key_insights.append({
            "title": "Optimization Successful",
            "description": f"Fitness improved by {improvement:.4f} ({pct_improvement:.1f}%). Final value: {best_fitness:.4f}",
            "status": "positive"
        })
    else:
        key_insights.append({
            "title": "No Improvement",
            "description": "Algorithm did not find better solutions. Consider adjusting parameters.",
            "status": "warning"
        })
    
    # Acceptance rate analysis
    if acceptance_rate > 0.5:
        key_insights.append({
            "title": "High Acceptance Rate",
            "description": f"Accepted {acceptance_rate*100:.1f}% of moves. Good exploration of search space.",
            "status": "positive"
        })
        recommendations.append("High acceptance indicates good temperature schedule")
    elif acceptance_rate < 0.1:
        key_insights.append({
            "title": "Low Acceptance Rate",
            "description": f"Only {acceptance_rate*100:.1f}% of moves accepted. May be cooling too fast.",
            "status": "warning"
        })
        recommendations.append("Consider slower cooling rate (closer to 1.0)")
    else:
        key_insights.append({
            "title": "Moderate Acceptance Rate",
            "description": f"Accepted {acceptance_rate*100:.1f}% of moves. Balanced exploration/exploitation.",
            "status": "neutral"
        })
    
    # Cooling analysis
    if cooling_rate < 0.95:
        recommendations.append("Fast cooling - good for quick optimization")
    else:
        recommendations.append("Slow cooling - thorough search but slower")
    
    # Temperature analysis
    if initial_temp < 100:
        recommendations.append("Low initial temperature - may get stuck in local optima")
    elif initial_temp > 5000:
        recommendations.append("High initial temperature - extensive exploration")
    
    return {
        "key_insights": key_insights,
        "recommendations": recommendations
    }


@router.post("/simulated-annealing")
async def solve_simulated_annealing(request: SARequest):
    """
    Solve optimization problem using Simulated Annealing
    
    Mimics metallurgical annealing: heating and controlled cooling
    """
    try:
        # Validate bounds
        if len(request.bounds) == 0:
            raise HTTPException(400, "At least one variable required")
        
        bounds_array = np.array(request.bounds)
        
        # Run optimization
        best_solution, best_fitness, convergence, temperature, acceptance_rate = simulated_annealing(
            request.objective_function,
            bounds_array,
            request.initial_temp,
            request.cooling_rate,
            request.max_iter
        )
        
        # Generate plots
        plots = {
            "convergence": create_convergence_plot(convergence),
            "temperature": create_temperature_plot(temperature)
        }
        
        # Add 3D surface plot for 2D problems
        surface_plot = create_3d_surface_plot(request.objective_function, bounds_array, best_solution)
        if surface_plot:
            plots["surface"] = surface_plot
        
        # Generate interpretation
        interpretation = generate_interpretation(
            best_fitness,
            convergence,
            acceptance_rate,
            request.initial_temp,
            request.cooling_rate
        )
        
        return SAResponse(
            success=True,
            best_solution=best_solution.tolist(),
            best_fitness=float(best_fitness),
            convergence_history=convergence,
            temperature_history=temperature,
            acceptance_rate=float(acceptance_rate),
            iterations=request.max_iter,
            problem_info={
                "n_variables": len(request.bounds),
                "bounds": request.bounds,
                "objective_function": request.objective_function,
                "initial_fitness": float(convergence[0]),
                "final_fitness": float(convergence[-1]),
                "improvement": float(convergence[0] - convergence[-1])
            },
            plots=plots,
            interpretation=interpretation
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(e)}")
