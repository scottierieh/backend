"""
Ant Colony Optimization FastAPI Endpoint
Bio-inspired optimization mimicking ant foraging behavior
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


class ACORequest(BaseModel):
    """Request model for ACO"""
    objective_function: str
    variables: List[VariableInput]
    n_ants: int = Field(default=50, ge=10, le=200)
    n_iterations: int = Field(default=100, ge=10, le=500)
    evaporation_rate: float = Field(default=0.1, ge=0.01, le=0.5)
    alpha: float = Field(default=1.0, ge=0.0, le=5.0)
    beta: float = Field(default=2.0, ge=0.0, le=5.0)


class VariableDetail(BaseModel):
    """Variable detail information"""
    name: str
    min_value: float
    max_value: float
    optimal_value: float
    range: float
    selected: bool


class ACOResponse(BaseModel):
    """Response model for ACO"""
    success: bool
    best_fitness: float
    convergence_rate: float
    pheromone_strength: float
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


def ant_colony_optimization(
    objective_func: str,
    bounds: np.ndarray,
    n_ants: int,
    n_iterations: int,
    evaporation_rate: float,
    alpha: float,
    beta: float
):
    """
    Ant Colony Optimization
    
    Mimics ant foraging behavior using pheromone trails
    """
    n_vars = len(bounds)
    
    # Initialize pheromone matrix
    pheromone = np.ones((n_ants, n_vars))
    
    # Best solution tracking
    best_solution = None
    best_fitness = float('inf')
    convergence_history = []
    pheromone_history = []
    
    for iteration in range(n_iterations):
        solutions = []
        fitnesses = []
        
        # Each ant constructs a solution
        for ant in range(n_ants):
            solution = np.zeros(n_vars)
            
            for var in range(n_vars):
                # Probability based on pheromone
                prob = pheromone[ant, var] ** alpha
                
                # Heuristic information (random exploration)
                heuristic = np.random.random() ** beta
                
                # Combine pheromone and heuristic
                value = prob * heuristic
                
                # Generate solution within bounds
                solution[var] = bounds[var][0] + value * (bounds[var][1] - bounds[var][0])
                solution[var] = np.clip(solution[var], bounds[var][0], bounds[var][1])
            
            # Evaluate solution
            fitness = evaluate_objective(objective_func, solution)
            solutions.append(solution)
            fitnesses.append(fitness)
            
            # Update best solution
            if fitness < best_fitness:
                best_fitness = fitness
                best_solution = solution.copy()
        
        # Pheromone evaporation
        pheromone *= (1 - evaporation_rate)
        
        # Pheromone update based on solution quality
        for ant in range(n_ants):
            # Deposit pheromone inversely proportional to fitness
            pheromone_deposit = 1.0 / (1.0 + fitnesses[ant])
            pheromone[ant, :] += pheromone_deposit
        
        # Normalize pheromone
        pheromone = np.clip(pheromone, 0.1, 10.0)
        
        # Track convergence
        convergence_history.append(best_fitness)
        pheromone_history.append(np.mean(pheromone))
    
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
    
    # Calculate pheromone strength
    pheromone_strength = np.mean(pheromone)
    
    return (
        best_solution,
        best_fitness,
        convergence_history,
        pheromone_history,
        convergence_rate,
        pheromone_strength
    )


def create_convergence_plot(convergence: List[float]) -> str:
    """Create convergence plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = range(len(convergence))
    ax.plot(iterations, convergence, linewidth=2, color='steelblue', label='Best Fitness')
    
    ax.set_xlabel('Iteration', fontsize=12, weight='bold')
    ax.set_ylabel('Fitness (Lower is Better)', fontsize=12, weight='bold')
    ax.set_title('ACO - Convergence', fontsize=14, weight='bold')
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


def create_pheromone_plot(pheromone_history: List[float]) -> str:
    """Create pheromone strength plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = range(len(pheromone_history))
    ax.plot(iterations, pheromone_history, linewidth=2, color='orange', label='Avg Pheromone')
    
    ax.set_xlabel('Iteration', fontsize=12, weight='bold')
    ax.set_ylabel('Pheromone Strength', fontsize=12, weight='bold')
    ax.set_title('Pheromone Trail Evolution', fontsize=14, weight='bold')
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
    pheromone_strength: float,
    n_ants: int,
    n_iterations: int
) -> Dict[str, Any]:
    """Generate insights and recommendations"""
    
    key_insights = []
    recommendations = []
    
    # Convergence analysis
    improvement = convergence_history[0] - convergence_history[-1]
    if improvement > 0:
        key_insights.append({
            "title": "Successful Optimization",
            "description": f"Ants found improved solution. Final fitness: {best_fitness:.6f}. Convergence rate: {convergence_rate:.2f}%",
            "status": "positive"
        })
    else:
        key_insights.append({
            "title": "Limited Improvement",
            "description": "Colony did not find better solutions. Consider adjusting parameters.",
            "status": "warning"
        })
    
    # Pheromone analysis
    if pheromone_strength > 5.0:
        key_insights.append({
            "title": "Strong Pheromone Trails",
            "description": f"Average pheromone: {pheromone_strength:.2f}. Colony converged on promising regions.",
            "status": "positive"
        })
        recommendations.append("Strong pheromone indicates good exploitation")
    elif pheromone_strength < 1.0:
        key_insights.append({
            "title": "Weak Pheromone Trails",
            "description": f"Average pheromone: {pheromone_strength:.2f}. Colony still exploring.",
            "status": "neutral"
        })
        recommendations.append("Consider reducing evaporation rate for stronger trails")
    
    # Efficiency analysis
    efficiency = min(100, convergence_rate)
    if efficiency >= 80:
        recommendations.append(f"High efficiency ({efficiency:.1f}%) - excellent performance")
    elif efficiency >= 50:
        recommendations.append(f"Moderate efficiency ({efficiency:.1f}%) - reasonable performance")
    else:
        recommendations.append(f"Low efficiency ({efficiency:.1f}%) - consider more iterations")
    
    # Colony size analysis
    if n_ants < 30:
        recommendations.append("Small colony - may miss optimal regions")
    elif n_ants > 100:
        recommendations.append("Large colony - thorough exploration but slower")
    
    return {
        "key_insights": key_insights,
        "recommendations": recommendations
    }


@router.post("/ant-colony")
async def optimize_ant_colony(request: ACORequest):
    """
    Solve optimization using Ant Colony Optimization
    
    Mimics ant foraging behavior with pheromone trails
    """
    try:
        # Validate variables
        if len(request.variables) == 0:
            raise HTTPException(400, "At least one variable required")
        
        # Prepare bounds
        bounds = np.array([[var.min_value, var.max_value] for var in request.variables])
        
        # Run optimization
        best_solution, best_fitness, convergence, pheromone_hist, convergence_rate, pheromone_strength = ant_colony_optimization(
            request.objective_function,
            bounds,
            request.n_ants,
            request.n_iterations,
            request.evaporation_rate,
            request.alpha,
            request.beta
        )
        
        # Generate plots
        plots = {
            "convergence": create_convergence_plot(convergence),
            "pheromone": create_pheromone_plot(pheromone_hist)
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
            pheromone_strength,
            request.n_ants,
            request.n_iterations
        )
        
        return ACOResponse(
            success=True,
            best_fitness=float(best_fitness),
            convergence_rate=float(convergence_rate),
            pheromone_strength=float(pheromone_strength),
            efficiency=float(efficiency),
            best_solution=best_solution.tolist(),
            selected_variables=[var.name for var in request.variables],
            variable_details=variable_details,
            variable_details_by_range=variable_details_by_range,
            problem={
                "n_variables": len(request.variables),
                "n_ants": request.n_ants,
                "iterations": request.n_iterations,
                "evaporation_rate": request.evaporation_rate,
                "alpha": request.alpha,
                "beta": request.beta,
                "n_selected": len(request.variables)
            },
            plots=plots,
            interpretation=interpretation
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(e)}")
