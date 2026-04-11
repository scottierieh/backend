"""
Pareto Optimization FastAPI Endpoint
Multi-objective optimization to find Pareto-optimal solutions
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


class ObjectiveInput(BaseModel):
    """Objective function configuration"""
    name: str
    function: str
    weight: float = Field(default=1.0, ge=0.0, le=10.0)


class ParetoRequest(BaseModel):
    """Request model for Pareto Optimization"""
    objectives: List[ObjectiveInput]
    variables: List[VariableInput]
    n_solutions: int = Field(default=100, ge=10, le=500)
    n_iterations: int = Field(default=100, ge=10, le=500)


class VariableDetail(BaseModel):
    """Variable detail information"""
    name: str
    min_value: float
    max_value: float
    optimal_value: float
    range: float
    selected: bool


class ParetoResponse(BaseModel):
    """Response model for Pareto Optimization"""
    success: bool
    n_pareto_solutions: int
    best_solution: List[float]
    best_objectives: List[float]
    hypervolume: float
    convergence_rate: float
    efficiency: float
    pareto_front: List[List[float]]
    selected_variables: List[str]
    variable_details: List[VariableDetail]
    variable_details_by_range: List[VariableDetail]
    problem: Dict[str, Any]
    plots: Dict[str, Optional[str]]
    interpretation: Dict[str, Any]


def evaluate_objectives(objectives: List[ObjectiveInput], x: np.ndarray) -> np.ndarray:
    """Evaluate all objective functions"""
    results = []
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
    
    for obj in objectives:
        try:
            result = eval(obj.function, {"__builtins__": {}}, namespace)
            results.append(float(result) * obj.weight)
        except Exception as e:
            raise ValueError(f"Error evaluating {obj.name}: {str(e)}")
    
    return np.array(results)


def is_dominated(solution_a: np.ndarray, solution_b: np.ndarray) -> bool:
    """Check if solution_a is dominated by solution_b (minimization)"""
    return np.all(solution_b <= solution_a) and np.any(solution_b < solution_a)


def find_pareto_front(solutions: np.ndarray, objectives: np.ndarray) -> np.ndarray:
    """Find Pareto-optimal solutions"""
    n_solutions = len(solutions)
    is_pareto = np.ones(n_solutions, dtype=bool)
    
    for i in range(n_solutions):
        for j in range(n_solutions):
            if i != j and is_dominated(objectives[i], objectives[j]):
                is_pareto[i] = False
                break
    
    return is_pareto


def pareto_optimization(
    objectives_def: List[ObjectiveInput],
    bounds: np.ndarray,
    n_solutions: int,
    n_iterations: int
):
    """
    Multi-objective Pareto Optimization
    
    Finds Pareto-optimal solutions for multiple objectives
    """
    n_vars = len(bounds)
    n_objectives = len(objectives_def)
    
    # Initialize population
    population = np.random.uniform(
        bounds[:, 0],
        bounds[:, 1],
        size=(n_solutions, n_vars)
    )
    
    # Evaluate initial population
    objectives = np.array([
        evaluate_objectives(objectives_def, sol)
        for sol in population
    ])
    
    # Track best solutions
    best_population = population.copy()
    best_objectives = objectives.copy()
    
    convergence_history = []
    hypervolume_history = []
    
    for iteration in range(n_iterations):
        # Find current Pareto front
        pareto_mask = find_pareto_front(population, objectives)
        pareto_solutions = population[pareto_mask]
        pareto_objectives = objectives[pareto_mask]
        
        # Update best solutions
        combined_population = np.vstack([best_population, population])
        combined_objectives = np.vstack([best_objectives, objectives])
        combined_pareto = find_pareto_front(combined_population, combined_objectives)
        
        best_population = combined_population[combined_pareto]
        best_objectives = combined_objectives[combined_pareto]
        
        # Generate new solutions
        new_population = []
        for _ in range(n_solutions):
            if len(pareto_solutions) > 1:
                # Crossover
                parents = pareto_solutions[np.random.choice(len(pareto_solutions), 2, replace=False)]
                child = (parents[0] + parents[1]) / 2
                
                # Mutation
                mutation = np.random.normal(0, 0.1, n_vars)
                child += mutation * (bounds[:, 1] - bounds[:, 0])
                child = np.clip(child, bounds[:, 0], bounds[:, 1])
            else:
                # Random exploration
                child = np.random.uniform(bounds[:, 0], bounds[:, 1], n_vars)
            
            new_population.append(child)
        
        population = np.array(new_population)
        objectives = np.array([
            evaluate_objectives(objectives_def, sol)
            for sol in population
        ])
        
        # Calculate hypervolume (approximation)
        if len(pareto_objectives) > 0:
            reference_point = np.max(pareto_objectives, axis=0) + 1
            hv = np.sum([
                np.prod(reference_point - obj)
                for obj in pareto_objectives
            ])
            hypervolume_history.append(hv)
        else:
            hypervolume_history.append(0)
        
        # Track convergence (average objective values)
        avg_objectives = np.mean(pareto_objectives, axis=0) if len(pareto_objectives) > 0 else np.zeros(n_objectives)
        convergence_history.append(float(np.mean(avg_objectives)))
    
    # Final Pareto front
    final_pareto = find_pareto_front(best_population, best_objectives)
    pareto_solutions = best_population[final_pareto]
    pareto_objectives = best_objectives[final_pareto]
    
    # Select best compromise solution (closest to ideal point)
    if len(pareto_solutions) > 0:
        ideal_point = np.min(pareto_objectives, axis=0)
        distances = np.linalg.norm(pareto_objectives - ideal_point, axis=1)
        best_idx = np.argmin(distances)
        best_solution = pareto_solutions[best_idx]
        best_objectives_vals = pareto_objectives[best_idx]
    else:
        best_solution = best_population[0]
        best_objectives_vals = best_objectives[0]
    
    # Calculate convergence rate
    if len(convergence_history) > 1:
        initial = convergence_history[0]
        final = convergence_history[-1]
        if initial != 0:
            convergence_rate = abs((initial - final) / initial) * 100
        else:
            convergence_rate = 100.0
    else:
        convergence_rate = 0.0
    
    return (
        best_solution,
        best_objectives_vals,
        pareto_solutions,
        pareto_objectives,
        convergence_history,
        hypervolume_history,
        convergence_rate
    )


def create_convergence_plot(convergence: List[float]) -> str:
    """Create convergence plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = range(len(convergence))
    ax.plot(iterations, convergence, linewidth=2, color='steelblue', label='Avg Objectives')
    
    ax.set_xlabel('Iteration', fontsize=12, weight='bold')
    ax.set_ylabel('Average Objective Value', fontsize=12, weight='bold')
    ax.set_title('Multi-Objective Convergence', fontsize=14, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return img_str


def create_hypervolume_plot(hypervolume: List[float]) -> str:
    """Create hypervolume plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    iterations = range(len(hypervolume))
    ax.plot(iterations, hypervolume, linewidth=2, color='green', label='Hypervolume')
    
    ax.set_xlabel('Iteration', fontsize=12, weight='bold')
    ax.set_ylabel('Hypervolume', fontsize=12, weight='bold')
    ax.set_title('Hypervolume Evolution', fontsize=14, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return img_str


def create_pareto_front_plot(pareto_objectives: np.ndarray, objectives_names: List[str]) -> Optional[str]:
    """Create Pareto front visualization"""
    if len(pareto_objectives) == 0:
        return None
    
    n_objectives = pareto_objectives.shape[1]
    
    if n_objectives == 2:
        # 2D Pareto front
        fig, ax = plt.subplots(figsize=(10, 8))
        
        ax.scatter(pareto_objectives[:, 0], pareto_objectives[:, 1], 
                  s=100, c='red', alpha=0.6, edgecolors='black', linewidths=1.5,
                  label='Pareto Front')
        
        ax.set_xlabel(objectives_names[0] if len(objectives_names) > 0 else 'Objective 1', 
                     fontsize=12, weight='bold')
        ax.set_ylabel(objectives_names[1] if len(objectives_names) > 1 else 'Objective 2',
                     fontsize=12, weight='bold')
        ax.set_title('Pareto Front', fontsize=14, weight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return img_str
    
    elif n_objectives == 3:
        # 3D Pareto front
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(pareto_objectives[:, 0], pareto_objectives[:, 1], pareto_objectives[:, 2],
                  s=100, c='red', alpha=0.6, edgecolors='black', linewidths=1.5,
                  label='Pareto Front')
        
        ax.set_xlabel(objectives_names[0] if len(objectives_names) > 0 else 'Obj 1',
                     fontsize=11, weight='bold')
        ax.set_ylabel(objectives_names[1] if len(objectives_names) > 1 else 'Obj 2',
                     fontsize=11, weight='bold')
        ax.set_zlabel(objectives_names[2] if len(objectives_names) > 2 else 'Obj 3',
                     fontsize=11, weight='bold')
        ax.set_title('3D Pareto Front', fontsize=13, weight='bold')
        ax.legend()
        
        plt.tight_layout()
        
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        img_str = base64.b64encode(buffer.read()).decode()
        plt.close()
        
        return img_str
    
    return None


def generate_interpretation(
    n_pareto: int,
    hypervolume: float,
    convergence_rate: float,
    n_objectives: int
) -> Dict[str, Any]:
    """Generate insights and recommendations"""
    
    key_insights = []
    recommendations = []
    
    # Pareto front analysis
    if n_pareto > 20:
        key_insights.append({
            "title": "Large Pareto Front",
            "description": f"Found {n_pareto} Pareto-optimal solutions. Wide range of trade-offs available.",
            "status": "positive"
        })
        recommendations.append("Many trade-off solutions available - review based on preferences")
    elif n_pareto > 5:
        key_insights.append({
            "title": "Moderate Pareto Front",
            "description": f"Found {n_pareto} Pareto-optimal solutions. Good balance of trade-offs.",
            "status": "positive"
        })
    else:
        key_insights.append({
            "title": "Small Pareto Front",
            "description": f"Only {n_pareto} Pareto-optimal solutions found. Limited trade-offs.",
            "status": "warning"
        })
        recommendations.append("Consider increasing population size or iterations")
    
    # Hypervolume analysis
    if hypervolume > 1000:
        key_insights.append({
            "title": "Large Solution Space Covered",
            "description": f"Hypervolume: {hypervolume:.2f}. Excellent coverage of objective space.",
            "status": "positive"
        })
    
    # Convergence analysis
    if convergence_rate > 50:
        recommendations.append(f"Good convergence ({convergence_rate:.1f}%) - objectives improved significantly")
    else:
        recommendations.append(f"Limited convergence ({convergence_rate:.1f}%) - may need more iterations")
    
    # Multi-objective guidance
    if n_objectives == 2:
        recommendations.append("2 objectives - visualize Pareto front for trade-off analysis")
    elif n_objectives == 3:
        recommendations.append("3 objectives - use 3D visualization for trade-off analysis")
    else:
        recommendations.append(f"{n_objectives} objectives - complex trade-offs, consider priority weights")
    
    return {
        "key_insights": key_insights,
        "recommendations": recommendations
    }


@router.post("/pareto-optimization")
async def optimize_pareto(request: ParetoRequest):
    """
    Multi-objective Pareto Optimization
    
    Finds Pareto-optimal solutions balancing multiple objectives
    """
    try:
        # Validate
        if len(request.objectives) < 2:
            raise HTTPException(400, "At least 2 objectives required")
        if len(request.variables) == 0:
            raise HTTPException(400, "At least one variable required")
        
        # Prepare bounds
        bounds = np.array([[var.min_value, var.max_value] for var in request.variables])
        
        # Run optimization
        best_solution, best_objectives, pareto_sols, pareto_objs, convergence, hypervolume, conv_rate = pareto_optimization(
            request.objectives,
            bounds,
            request.n_solutions,
            request.n_iterations
        )
        
        # Generate plots
        plots = {
            "convergence": create_convergence_plot(convergence),
            "hypervolume": create_hypervolume_plot(hypervolume)
        }
        
        # Add Pareto front plot
        obj_names = [obj.name for obj in request.objectives]
        pareto_plot = create_pareto_front_plot(pareto_objs, obj_names)
        if pareto_plot:
            plots["pareto_front"] = pareto_plot
        
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
        efficiency = min(100, conv_rate)
        
        # Generate interpretation
        interpretation = generate_interpretation(
            len(pareto_sols),
            hypervolume[-1] if len(hypervolume) > 0 else 0,
            conv_rate,
            len(request.objectives)
        )
        
        return ParetoResponse(
            success=True,
            n_pareto_solutions=len(pareto_sols),
            best_solution=best_solution.tolist(),
            best_objectives=best_objectives.tolist(),
            hypervolume=float(hypervolume[-1]) if len(hypervolume) > 0 else 0,
            convergence_rate=float(conv_rate),
            efficiency=float(efficiency),
            pareto_front=pareto_objs.tolist(),
            selected_variables=[var.name for var in request.variables],
            variable_details=variable_details,
            variable_details_by_range=variable_details_by_range,
            problem={
                "n_variables": len(request.variables),
                "n_objectives": len(request.objectives),
                "n_solutions": request.n_solutions,
                "iterations": request.n_iterations,
                "n_selected": len(request.variables)
            },
            plots=plots,
            interpretation=interpretation
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(e)}")
