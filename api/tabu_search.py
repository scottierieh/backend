"""
Tabu Search FastAPI Endpoint
Metaheuristic optimization using memory-based search strategy
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


class VariableInput(BaseModel):
    """Variable configuration"""
    name: str
    min_value: float
    max_value: float


class TabuRequest(BaseModel):
    """Request model for Tabu Search"""
    objective_function: str
    variables: List[VariableInput]
    max_iter: int = Field(default=1000, ge=100, le=5000)
    tabu_tenure: int = Field(default=10, ge=5, le=100)
    n_neighbors: int = Field(default=50, ge=10, le=200)


class VariableDetail(BaseModel):
    """Variable detail information"""
    name: str
    min_value: float
    max_value: float
    optimal_value: float
    range: float
    selected: bool


class TabuResponse(BaseModel):
    """Response model for Tabu Search"""
    success: bool
    best_fitness: float
    convergence_rate: float
    tabu_effectiveness: float
    efficiency: float
    best_solution: List[float]
    selected_variables: List[str]
    variable_details: List[VariableDetail]
    variable_details_by_range: List[VariableDetail]
    problem: Dict[str, Any]
    plots: Dict[str, Optional[str]]
    interpretation: Dict[str, Any]


def evaluate_objective(func_str: str, x: np.ndarray) -> float:
    """Safely evaluate objective function"""
    try:
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


def tabu_search(
    objective_func: str,
    bounds: np.ndarray,
    max_iter: int,
    tabu_tenure: int,
    n_neighbors: int
):
    """
    Tabu Search optimization
    
    Uses memory structure (tabu list) to avoid revisiting recent solutions
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
    
    # Tabu list (stores recent solutions)
    tabu_list = []
    
    # History tracking
    convergence_history = [best_fitness]
    tabu_size_history = []
    
    # Statistics
    tabu_hits = 0
    total_neighbors = 0
    
    for iteration in range(max_iter):
        # Generate neighbors
        neighbors = []
        fitnesses = []
        
        for _ in range(n_neighbors):
            neighbor = current_solution.copy()
            
            # Perturb random dimensions
            n_perturb = np.random.randint(1, max(2, n_vars // 2))
            dims = np.random.choice(n_vars, n_perturb, replace=False)
            
            for dim in dims:
                step_size = (bounds[dim][1] - bounds[dim][0]) * 0.1
                neighbor[dim] += np.random.uniform(-step_size, step_size)
                neighbor[dim] = np.clip(neighbor[dim], bounds[dim][0], bounds[dim][1])
            
            # Check if neighbor is in tabu list
            is_tabu = False
            for tabu_sol in tabu_list:
                if np.allclose(neighbor, tabu_sol, rtol=1e-3):
                    is_tabu = True
                    tabu_hits += 1
                    break
            
            total_neighbors += 1
            
            # Skip tabu solutions unless they're better (aspiration criterion)
            if not is_tabu:
                fitness = evaluate_objective(objective_func, neighbor)
                neighbors.append(neighbor)
                fitnesses.append(fitness)
        
        # Select best non-tabu neighbor
        if len(neighbors) > 0:
            best_neighbor_idx = np.argmin(fitnesses)
            current_solution = neighbors[best_neighbor_idx]
            current_fitness = fitnesses[best_neighbor_idx]
            
            # Update best solution
            if current_fitness < best_fitness:
                best_solution = current_solution.copy()
                best_fitness = current_fitness
            
            # Add current solution to tabu list
            tabu_list.append(current_solution.copy())
            
            # Maintain tabu tenure
            if len(tabu_list) > tabu_tenure:
                tabu_list.pop(0)
        
        # Record history
        convergence_history.append(best_fitness)
        tabu_size_history.append(len(tabu_list))
    
    # Calculate convergence rate
    if len(convergence_history) > 1:
        initial_fitness = convergence_history[0]
        final_fitness = convergence_history[-1]
        if initial_fitness != 0:
            convergence_rate = abs((initial_fitness - final_fitness) / initial_fitness) * 100
        else:
            convergence_rate = 100.0
    else:
        convergence_rate = 0.0
    
    # Calculate tabu effectiveness
    tabu_effectiveness = (tabu_hits / total_neighbors * 100) if total_neighbors > 0 else 0
    
    return (
        best_solution,
        best_fitness,
        convergence_history,
        tabu_size_history,
        convergence_rate,
        tabu_effectiveness
    )


def create_convergence_plot(convergence: List[float]) -> str:
    """Create convergence plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = range(len(convergence))
    ax.plot(iterations, convergence, linewidth=2, color='steelblue', label='Best Fitness')
    
    ax.set_xlabel('Iteration', fontsize=12, weight='bold')
    ax.set_ylabel('Fitness (Lower is Better)', fontsize=12, weight='bold')
    ax.set_title('Tabu Search - Convergence', fontsize=14, weight='bold')
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


def create_tabu_size_plot(tabu_size: List[int]) -> str:
    """Create tabu list size plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = range(len(tabu_size))
    ax.plot(iterations, tabu_size, linewidth=2, color='orange', label='Tabu List Size')
    
    ax.set_xlabel('Iteration', fontsize=12, weight='bold')
    ax.set_ylabel('Tabu List Size', fontsize=12, weight='bold')
    ax.set_title('Tabu Memory Evolution', fontsize=14, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return img_str


def create_solution_space_plot(func_str: str, bounds: np.ndarray, best_solution: np.ndarray) -> Optional[str]:
    """Create solution space visualization for 2D problems"""
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
        
        # Create contour plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        contour = ax.contourf(X, Y, Z, levels=20, cmap='viridis', alpha=0.8)
        ax.contour(X, Y, Z, levels=10, colors='black', alpha=0.3, linewidths=0.5)
        
        # Mark best solution
        ax.scatter([best_solution[0]], [best_solution[1]], 
                  color='red', s=200, marker='*', 
                  edgecolors='white', linewidths=2,
                  label='Best Solution', zorder=10)
        
        ax.set_xlabel(f'Variable 1', fontsize=11, weight='bold')
        ax.set_ylabel(f'Variable 2', fontsize=11, weight='bold')
        ax.set_title('Solution Space Landscape', fontsize=13, weight='bold')
        ax.legend()
        
        plt.colorbar(contour, ax=ax, label='Fitness')
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
    convergence_history: List[float],
    convergence_rate: float,
    tabu_effectiveness: float,
    tabu_tenure: int
) -> Dict[str, Any]:
    """Generate insights and recommendations"""
    
    key_insights = []
    recommendations = []
    
    # Convergence analysis
    improvement = convergence_history[0] - convergence_history[-1]
    if improvement > 0:
        key_insights.append({
            "title": "Successful Optimization",
            "description": f"Found improved solution. Final fitness: {best_fitness:.6f}. Convergence rate: {convergence_rate:.2f}%",
            "status": "positive"
        })
    else:
        key_insights.append({
            "title": "Limited Improvement",
            "description": "Search did not find better solutions. Consider adjusting parameters.",
            "status": "warning"
        })
    
    # Tabu effectiveness analysis
    if tabu_effectiveness > 30:
        key_insights.append({
            "title": "High Tabu Hit Rate",
            "description": f"Tabu list blocked {tabu_effectiveness:.1f}% of neighbors. Good diversification.",
            "status": "positive"
        })
        recommendations.append("High tabu effectiveness - memory is working well")
    elif tabu_effectiveness < 10:
        key_insights.append({
            "title": "Low Tabu Hit Rate",
            "description": f"Only {tabu_effectiveness:.1f}% of neighbors were tabu. May need longer tenure.",
            "status": "neutral"
        })
        recommendations.append("Consider increasing tabu tenure for better memory")
    
    # Efficiency analysis
    efficiency = min(100, convergence_rate)
    if efficiency >= 80:
        recommendations.append(f"High efficiency ({efficiency:.1f}%) - excellent performance")
    elif efficiency >= 50:
        recommendations.append(f"Moderate efficiency ({efficiency:.1f}%) - reasonable performance")
    else:
        recommendations.append(f"Low efficiency ({efficiency:.1f}%) - consider more iterations")
    
    # Tenure analysis
    if tabu_tenure < 10:
        recommendations.append("Short tabu tenure - may revisit solutions too quickly")
    elif tabu_tenure > 50:
        recommendations.append("Long tabu tenure - thorough diversification but may slow convergence")
    
    return {
        "key_insights": key_insights,
        "recommendations": recommendations
    }


@router.post("/tabu-search")
async def optimize_tabu_search(request: TabuRequest):
    """
    Solve optimization using Tabu Search
    
    Uses memory-based search to avoid local optima
    """
    try:
        # Validate variables
        if len(request.variables) == 0:
            raise HTTPException(400, "At least one variable required")
        
        # Prepare bounds
        bounds = np.array([[var.min_value, var.max_value] for var in request.variables])
        
        # Run optimization
        best_solution, best_fitness, convergence, tabu_size, convergence_rate, tabu_effectiveness = tabu_search(
            request.objective_function,
            bounds,
            request.max_iter,
            request.tabu_tenure,
            request.n_neighbors
        )
        
        # Generate plots
        plots = {
            "convergence": create_convergence_plot(convergence),
            "tabu_memory": create_tabu_size_plot(tabu_size)
        }
        
        # Add solution space plot for 2D
        solution_plot = create_solution_space_plot(request.objective_function, bounds, best_solution)
        if solution_plot:
            plots["solution_space"] = solution_plot
        
        # Variable details
        variable_details = []
        for i, var in enumerate(request.variables):
            variable_details.append(VariableDetail(
                name=var.name,
                min_value=var.min_value,
                max_value=var.max_value,
                optimal_value=float(best_solution[i]),
                range=var.max_value - var.min_value,
                selected=True
            ))
        
        # Sort by range
        variable_details_by_range = sorted(variable_details, key=lambda x: x.range, reverse=True)
        
        # Calculate efficiency
        efficiency = min(100, convergence_rate)
        
        # Generate interpretation
        interpretation = generate_interpretation(
            best_fitness,
            convergence,
            convergence_rate,
            tabu_effectiveness,
            request.tabu_tenure
        )
        
        return TabuResponse(
            success=True,
            best_fitness=float(best_fitness),
            convergence_rate=float(convergence_rate),
            tabu_effectiveness=float(tabu_effectiveness),
            efficiency=float(efficiency),
            best_solution=best_solution.tolist(),
            selected_variables=[var.name for var in request.variables],
            variable_details=variable_details,
            variable_details_by_range=variable_details_by_range,
            problem={
                "n_variables": len(request.variables),
                "max_iter": request.max_iter,
                "tabu_tenure": request.tabu_tenure,
                "n_neighbors": request.n_neighbors,
                "n_selected": len(request.variables)
            },
            plots=plots,
            interpretation=interpretation
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(e)}")
