"""
Genetic Algorithm: Global Optimization Router for FastAPI
Evolutionary computation for global optimization problems
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from enum import Enum
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import warnings
import re
import math

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
sns.set_palette("husl")

router = APIRouter()


class Variable(BaseModel):
    """Variable definition for optimization"""
    name: str = Field(description="Variable name")
    min_value: float = Field(description="Minimum value")
    max_value: float = Field(description="Maximum value")


class GeneticAlgorithmRequest(BaseModel):
    """Genetic algorithm optimization request parameters"""
    
    # Problem definition
    objective_function: str = Field(
        ...,
        description="Objective function to minimize (Python expression)"
    )
    variables: List[Variable] = Field(
        ...,
        min_items=1,
        max_items=20,
        description="List of variables with bounds"
    )
    
    # GA parameters
    population_size: int = Field(
        default=50,
        ge=10,
        le=500,
        description="Population size"
    )
    generations: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Number of generations"
    )
    mutation_rate: float = Field(
        default=0.01,
        ge=0.0,
        le=1.0,
        description="Mutation rate"
    )
    crossover_rate: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Crossover rate"
    )
    elite_size: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Number of elite individuals to preserve"
    )
    
    # Analysis options
    track_diversity: bool = Field(
        default=True,
        description="Track population diversity over generations"
    )
    early_stopping: bool = Field(
        default=True,
        description="Enable early stopping when convergence is detected"
    )
    convergence_tolerance: float = Field(
        default=1e-6,
        ge=1e-10,
        le=1e-2,
        description="Convergence tolerance for early stopping"
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


def validate_objective_function(func_str: str, n_variables: int) -> bool:
    """Validate that the objective function is safe and well-formed"""
    # Check for dangerous imports or functions
    dangerous_patterns = [
        r'\b(import|exec|eval|open|file|input|raw_input)\b',
        r'\b(__.*__)\b',
        r'\b(os|sys|subprocess|shutil)\b'
    ]
    
    for pattern in dangerous_patterns:
        if re.search(pattern, func_str, re.IGNORECASE):
            return False
    
    # Test compilation and basic evaluation
    try:
        # Create test variables
        x = np.ones(n_variables)
        
        # Replace common patterns for evaluation
        test_func = func_str
        
        # Allow numpy functions
        allowed_globals = {
            'x': x,
            'np': np,
            'sum': np.sum,
            'abs': np.abs,
            'sqrt': np.sqrt,
            'exp': np.exp,
            'log': np.log,
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan,
            'pi': np.pi,
            'e': np.e,
            'len': len,
            'min': min,
            'max': max,
            'pow': pow
        }
        
        # Test evaluation
        result = eval(test_func, allowed_globals)
        return isinstance(result, (int, float, np.number)) and not (np.isnan(result) or np.isinf(result))
        
    except:
        return False


def evaluate_objective_function(func_str: str, x: np.ndarray) -> float:
    """Safely evaluate the objective function"""
    try:
        allowed_globals = {
            'x': x,
            'np': np,
            'sum': np.sum,
            'abs': np.abs,
            'sqrt': np.sqrt,
            'exp': np.exp,
            'log': np.log,
            'sin': np.sin,
            'cos': np.cos,
            'tan': np.tan,
            'pi': np.pi,
            'e': np.e,
            'len': len,
            'min': min,
            'max': max,
            'pow': pow
        }
        
        result = eval(func_str, allowed_globals)
        
        # Handle invalid results
        if np.isnan(result) or np.isinf(result):
            return 1e10  # Large penalty for invalid results
            
        return float(result)
        
    except:
        return 1e10  # Large penalty for evaluation errors


# =============================================================================
# Genetic Algorithm Implementation
# =============================================================================

class GeneticAlgorithm:
    """Genetic Algorithm for global optimization"""
    
    def __init__(self, objective_func: str, bounds: List[tuple], 
                 population_size: int = 50, mutation_rate: float = 0.01,
                 crossover_rate: float = 0.8, elite_size: int = 2):
        self.objective_func = objective_func
        self.bounds = bounds
        self.n_variables = len(bounds)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        
        # Storage for tracking
        self.fitness_history = []
        self.best_fitness_history = []
        self.diversity_history = []
        self.population_history = []
        
    def initialize_population(self) -> np.ndarray:
        """Initialize random population within bounds"""
        population = np.zeros((self.population_size, self.n_variables))
        
        for i in range(self.n_variables):
            min_val, max_val = self.bounds[i]
            population[:, i] = np.random.uniform(min_val, max_val, self.population_size)
            
        return population
    
    def evaluate_fitness(self, population: np.ndarray) -> np.ndarray:
        """Evaluate fitness for all individuals in population"""
        fitness = np.zeros(len(population))
        
        for i, individual in enumerate(population):
            fitness[i] = evaluate_objective_function(self.objective_func, individual)
            
        return fitness
    
    def selection(self, population: np.ndarray, fitness: np.ndarray, k: int = 3) -> np.ndarray:
        """Tournament selection"""
        selected = np.zeros((self.population_size, self.n_variables))
        
        for i in range(self.population_size):
            # Tournament selection
            tournament_indices = np.random.choice(len(population), k, replace=False)
            tournament_fitness = fitness[tournament_indices]
            winner_idx = tournament_indices[np.argmin(tournament_fitness)]
            selected[i] = population[winner_idx]
            
        return selected
    
    def crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> tuple:
        """Simulated Binary Crossover (SBX)"""
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        eta = 2.0  # Distribution index
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent2)
        
        for i in range(self.n_variables):
            if np.random.random() <= 0.5:
                if abs(parent1[i] - parent2[i]) > 1e-14:
                    y1, y2 = min(parent1[i], parent2[i]), max(parent1[i], parent2[i])
                    
                    # Generate offspring
                    rand = np.random.random()
                    if rand <= 0.5:
                        beta = (2 * rand) ** (1.0 / (eta + 1))
                    else:
                        beta = (1.0 / (2 * (1 - rand))) ** (1.0 / (eta + 1))
                    
                    child1[i] = 0.5 * ((y1 + y2) - beta * (y2 - y1))
                    child2[i] = 0.5 * ((y1 + y2) + beta * (y2 - y1))
                else:
                    child1[i] = parent1[i]
                    child2[i] = parent2[i]
            else:
                child1[i] = parent1[i]
                child2[i] = parent2[i]
        
        # Ensure bounds
        for i in range(self.n_variables):
            min_val, max_val = self.bounds[i]
            child1[i] = np.clip(child1[i], min_val, max_val)
            child2[i] = np.clip(child2[i], min_val, max_val)
            
        return child1, child2
    
    def mutation(self, individual: np.ndarray) -> np.ndarray:
        """Polynomial mutation"""
        eta = 20.0  # Distribution index
        mutated = individual.copy()
        
        for i in range(self.n_variables):
            if np.random.random() <= self.mutation_rate:
                min_val, max_val = self.bounds[i]
                
                delta1 = (mutated[i] - min_val) / (max_val - min_val)
                delta2 = (max_val - mutated[i]) / (max_val - min_val)
                
                rand = np.random.random()
                mut_pow = 1.0 / (eta + 1.0)
                
                if rand <= 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (eta + 1.0))
                    deltaq = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (eta + 1.0))
                    deltaq = 1.0 - val ** mut_pow
                
                mutated[i] = mutated[i] + deltaq * (max_val - min_val)
                mutated[i] = np.clip(mutated[i], min_val, max_val)
                
        return mutated
    
    def calculate_diversity(self, population: np.ndarray) -> float:
        """Calculate population diversity"""
        if len(population) < 2:
            return 0.0
        
        # Calculate pairwise distances
        distances = []
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                dist = np.linalg.norm(population[i] - population[j])
                distances.append(dist)
        
        return np.mean(distances) if distances else 0.0
    
    def optimize(self, max_generations: int, track_diversity: bool = True,
                 early_stopping: bool = True, convergence_tolerance: float = 1e-6) -> Dict:
        """Run genetic algorithm optimization"""
        
        # Initialize
        population = self.initialize_population()
        
        best_solution = None
        best_fitness = float('inf')
        convergence_count = 0
        
        for generation in range(max_generations):
            # Evaluate fitness
            fitness = self.evaluate_fitness(population)
            
            # Track best solution
            gen_best_idx = np.argmin(fitness)
            gen_best_fitness = fitness[gen_best_idx]
            
            if gen_best_fitness < best_fitness:
                best_fitness = gen_best_fitness
                best_solution = population[gen_best_idx].copy()
                convergence_count = 0
            else:
                convergence_count += 1
            
            # Store history
            self.fitness_history.append(fitness.copy())
            self.best_fitness_history.append(best_fitness)
            
            if track_diversity:
                diversity = self.calculate_diversity(population)
                self.diversity_history.append(diversity)
            
            # Store population sample
            if generation % max(1, max_generations // 20) == 0:
                self.population_history.append(population.copy())
            
            # Early stopping check
            if early_stopping and convergence_count > max_generations * 0.1:
                if len(self.best_fitness_history) > 10:
                    recent_improvement = (self.best_fitness_history[-10] - 
                                        self.best_fitness_history[-1])
                    if recent_improvement < convergence_tolerance:
                        break
            
            # Create next generation
            if generation < max_generations - 1:
                # Elite preservation
                elite_indices = np.argsort(fitness)[:self.elite_size]
                new_population = population[elite_indices].copy()
                
                # Selection
                selected = self.selection(population, fitness)
                
                # Crossover and mutation
                for i in range(self.elite_size, self.population_size, 2):
                    parent1 = selected[np.random.randint(len(selected))]
                    parent2 = selected[np.random.randint(len(selected))]
                    
                    child1, child2 = self.crossover(parent1, parent2)
                    child1 = self.mutation(child1)
                    child2 = self.mutation(child2)
                    
                    if i < self.population_size:
                        new_population = np.vstack([new_population, child1.reshape(1, -1)])
                    if i + 1 < self.population_size:
                        new_population = np.vstack([new_population, child2.reshape(1, -1)])
                
                population = new_population[:self.population_size]
        
        # Calculate final metrics
        final_diversity = self.calculate_diversity(population) if track_diversity else 0.0
        convergence_rate = max(0, (self.best_fitness_history[0] - best_fitness) / 
                              max(abs(self.best_fitness_history[0]), 1e-10)) * 100
        
        efficiency = max(0, min(100, 100 - generation / max_generations * 100))
        
        return {
            'best_solution': best_solution,
            'best_fitness': best_fitness,
            'convergence_rate': convergence_rate,
            'final_diversity': final_diversity,
            'efficiency': efficiency,
            'generations_completed': generation + 1
        }


# =============================================================================
# Plotting Functions
# =============================================================================

def generate_convergence_plot(best_fitness_history: List[float]) -> str:
    """Generate convergence plot"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    generations = range(len(best_fitness_history))
    ax.plot(generations, best_fitness_history, 'b-', linewidth=2, alpha=0.8)
    ax.fill_between(generations, best_fitness_history, alpha=0.3)
    
    # Mark key points
    min_fitness = min(best_fitness_history)
    min_gen = best_fitness_history.index(min_fitness)
    ax.plot(min_gen, min_fitness, 'ro', markersize=8, label=f'Best: {min_fitness:.6f}')
    
    ax.set_xlabel('Generation', fontsize=12, fontweight='600')
    ax.set_ylabel('Best Fitness', fontsize=12, fontweight='600')
    ax.set_title('Convergence Progress', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # Add convergence rate annotation
    if len(best_fitness_history) > 1:
        improvement = (best_fitness_history[0] - best_fitness_history[-1]) / max(abs(best_fitness_history[0]), 1e-10)
        ax.text(0.05, 0.95, f'Improvement: {improvement*100:.1f}%', 
                transform=ax.transAxes, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_fitness_distribution_plot(fitness_history: List[np.ndarray]) -> str:
    """Generate fitness distribution plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left: Box plot of fitness distribution over generations
    sample_generations = min(len(fitness_history), 20)
    sample_indices = np.linspace(0, len(fitness_history) - 1, sample_generations, dtype=int)
    
    box_data = [fitness_history[i] for i in sample_indices]
    box_labels = [f'Gen {i}' for i in sample_indices]
    
    bp = ax1.boxplot(box_data, labels=box_labels, patch_artist=True)
    
    # Color boxes with gradient
    colors = plt.cm.viridis(np.linspace(0, 1, len(bp['boxes'])))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax1.set_xlabel('Generation', fontsize=11, fontweight='600')
    ax1.set_ylabel('Fitness', fontsize=11, fontweight='600')
    ax1.set_title('Fitness Distribution Evolution', fontsize=13, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Right: Final generation fitness histogram
    if fitness_history:
        final_fitness = fitness_history[-1]
        ax2.hist(final_fitness, bins=20, alpha=0.7, color='skyblue', edgecolor='navy')
        ax2.axvline(np.min(final_fitness), color='red', linestyle='--', 
                   label=f'Best: {np.min(final_fitness):.4f}')
        ax2.axvline(np.mean(final_fitness), color='orange', linestyle='--', 
                   label=f'Mean: {np.mean(final_fitness):.4f}')
        
        ax2.set_xlabel('Fitness Value', fontsize=11, fontweight='600')
        ax2.set_ylabel('Frequency', fontsize=11, fontweight='600')
        ax2.set_title('Final Population Distribution', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_solution_space_plot(best_solution: np.ndarray, bounds: List[tuple],
                                variable_names: List[str]) -> str:
    """Generate solution space visualization"""
    n_vars = len(best_solution)
    
    if n_vars == 2:
        # 2D contour plot for 2 variables
        fig, ax = plt.subplots(figsize=(10, 8))
        
        x1_range = np.linspace(bounds[0][0], bounds[0][1], 50)
        x2_range = np.linspace(bounds[1][0], bounds[1][1], 50)
        X1, X2 = np.meshgrid(x1_range, x2_range)
        
        # Plot optimal point
        ax.plot(best_solution[0], best_solution[1], 'r*', markersize=20, 
               label=f'Optimal: ({best_solution[0]:.3f}, {best_solution[1]:.3f})')
        
        ax.set_xlabel(f'{variable_names[0]}', fontsize=12, fontweight='600')
        ax.set_ylabel(f'{variable_names[1]}', fontsize=12, fontweight='600')
        ax.set_title('Solution Space (2D)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
    else:
        # Parallel coordinates plot for >2 variables
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Normalize solution to [0, 1] for plotting
        normalized_solution = []
        for i, (val, (min_val, max_val)) in enumerate(zip(best_solution, bounds)):
            norm_val = (val - min_val) / (max_val - min_val)
            normalized_solution.append(norm_val)
        
        x_pos = range(n_vars)
        ax.plot(x_pos, normalized_solution, 'o-', linewidth=3, markersize=8, 
               color='red', alpha=0.8)
        
        # Add value labels
        for i, (pos, val, norm_val) in enumerate(zip(x_pos, best_solution, normalized_solution)):
            ax.annotate(f'{val:.3f}', (pos, norm_val), 
                       textcoords="offset points", xytext=(0,10), ha='center',
                       fontsize=10, fontweight='600')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(variable_names, rotation=45, ha='right')
        ax.set_ylabel('Normalized Value', fontsize=12, fontweight='600')
        ax.set_title('Optimal Solution (Parallel Coordinates)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_population_evolution_plot(population_history: List[np.ndarray],
                                     diversity_history: List[float]) -> str:
    """Generate population evolution and diversity plot"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Top: Population diversity over time
    if diversity_history:
        generations = range(len(diversity_history))
        ax1.plot(generations, diversity_history, 'g-', linewidth=2, alpha=0.8)
        ax1.fill_between(generations, diversity_history, alpha=0.3, color='green')
        
        ax1.set_xlabel('Generation', fontsize=11, fontweight='600')
        ax1.set_ylabel('Population Diversity', fontsize=11, fontweight='600')
        ax1.set_title('Population Diversity Evolution', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
    
    # Bottom: Population spread visualization (for first 2 variables)
    if population_history and len(population_history[0][0]) >= 2:
        colors = plt.cm.viridis(np.linspace(0, 1, len(population_history)))
        
        for i, (pop, color) in enumerate(zip(population_history, colors)):
            ax2.scatter(pop[:, 0], pop[:, 1], c=[color], alpha=0.6, s=30,
                       label=f'Gen {i * len(diversity_history) // len(population_history)}')
        
        ax2.set_xlabel('Variable 1', fontsize=11, fontweight='600')
        ax2.set_ylabel('Variable 2', fontsize=11, fontweight='600')
        ax2.set_title('Population Evolution in Search Space', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Add legend with fewer entries
        handles, labels = ax2.get_legend_handles_labels()
        ax2.legend(handles[::max(1, len(handles)//5)], labels[::max(1, len(labels)//5)], 
                  fontsize=9)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Analysis and Interpretation
# =============================================================================

def generate_interpretation(result: Dict, request: GeneticAlgorithmRequest) -> Dict:
    """Generate interpretation of GA results"""
    key_insights = []
    
    best_fitness = result['best_fitness']
    convergence_rate = result['convergence_rate']
    efficiency = result['efficiency']
    generations_completed = result['generations_completed']
    
    # Convergence analysis
    if convergence_rate > 80:
        key_insights.append({
            'title': 'Excellent Convergence',
            'description': f'Algorithm achieved {convergence_rate:.1f}% improvement from initial population.',
            'status': 'positive'
        })
    elif convergence_rate > 40:
        key_insights.append({
            'title': 'Good Convergence',
            'description': f'Algorithm achieved {convergence_rate:.1f}% improvement. Consider more generations.',
            'status': 'neutral'
        })
    else:
        key_insights.append({
            'title': 'Limited Convergence',
            'description': f'Only {convergence_rate:.1f}% improvement achieved. May need parameter tuning.',
            'status': 'warning'
        })
    
    # Efficiency analysis
    if efficiency > 80:
        key_insights.append({
            'title': 'Efficient Optimization',
            'description': f'Solution found with {efficiency:.1f}% efficiency in {generations_completed} generations.',
            'status': 'positive'
        })
    elif efficiency < 40:
        key_insights.append({
            'title': 'Slow Convergence',
            'description': f'Low efficiency ({efficiency:.1f}%). Consider adjusting parameters.',
            'status': 'warning'
        })
    
    # Solution quality analysis
    if abs(best_fitness) < 1e-6:
        key_insights.append({
            'title': 'Near-Optimal Solution',
            'description': f'Found solution with fitness {best_fitness:.2e}, very close to theoretical optimum.',
            'status': 'positive'
        })
    elif abs(best_fitness) < 1e-2:
        key_insights.append({
            'title': 'Good Solution Quality',
            'description': f'Found good solution with fitness {best_fitness:.4f}.',
            'status': 'neutral'
        })
    
    # Problem characteristics
    n_vars = len(request.variables)
    if n_vars > 5:
        key_insights.append({
            'title': 'High-Dimensional Problem',
            'description': f'Optimizing {n_vars} variables. Consider increasing population size or generations.',
            'status': 'neutral'
        })
    
    # Generate recommendations
    recommendations = []
    
    if convergence_rate < 30:
        recommendations.append("Consider increasing population size or generations for better convergence.")
    
    if efficiency < 50:
        recommendations.append("Try adjusting mutation rate or crossover rate to improve search efficiency.")
    
    if generations_completed == request.generations:
        recommendations.append("Algorithm used all generations. Consider running longer for potential improvement.")
    
    if request.population_size < 10 * n_vars:
        recommendations.append(f"For {n_vars} variables, consider population size of at least {10 * n_vars}.")
    
    if not recommendations:
        recommendations.append("Optimization completed successfully with good parameters.")
    
    return {
        'key_insights': key_insights,
        'recommendations': recommendations
    }


# =============================================================================
# API Endpoint
# =============================================================================

@router.post("/genetic-algorithm")
async def optimize_genetic_algorithm(request: GeneticAlgorithmRequest) -> Dict:
    """
    Global optimization using Genetic Algorithm.
    
    Implements evolutionary computation for finding global optima:
    1. Population-based search
    2. Selection, crossover, and mutation operators
    3. Elitism and diversity preservation
    4. Convergence monitoring
    
    Returns optimal solution, convergence history, and analysis.
    """
    try:
        # Validate objective function
        n_variables = len(request.variables)
        if not validate_objective_function(request.objective_function, n_variables):
            raise ValueError("Invalid or unsafe objective function")
        
        # Prepare bounds
        bounds = [(var.min_value, var.max_value) for var in request.variables]
        variable_names = [var.name for var in request.variables]
        
        # Validate bounds
        for i, (min_val, max_val) in enumerate(bounds):
            if min_val >= max_val:
                raise ValueError(f"Invalid bounds for variable {variable_names[i]}: min >= max")
        
        # Initialize and run GA
        ga = GeneticAlgorithm(
            objective_func=request.objective_function,
            bounds=bounds,
            population_size=request.population_size,
            mutation_rate=request.mutation_rate,
            crossover_rate=request.crossover_rate,
            elite_size=request.elite_size
        )
        
        result = ga.optimize(
            max_generations=request.generations,
            track_diversity=request.track_diversity,
            early_stopping=request.early_stopping,
            convergence_tolerance=request.convergence_tolerance
        )
        
        best_solution = result['best_solution']
        
        # Calculate variable details
        variable_details = []
        variable_details_by_range = []
        selected_variables = []
        
        for i, (var, value) in enumerate(zip(request.variables, best_solution)):
            var_range = var.max_value - var.min_value
            selected = True  # All variables are always "selected" in GA
            
            detail = {
                'name': var.name,
                'min_value': var.min_value,
                'max_value': var.max_value,
                'optimal_value': _to_native_type(value),
                'range': _to_native_type(var_range),
                'selected': selected
            }
            variable_details.append(detail)
            variable_details_by_range.append(detail)
            selected_variables.append(var.name)
        
        # Sort by range
        variable_details_by_range.sort(key=lambda x: x['range'], reverse=True)
        
        # Generate plots
        plots = {}
        plots['convergence'] = generate_convergence_plot(ga.best_fitness_history)
        plots['fitness_distribution'] = generate_fitness_distribution_plot(ga.fitness_history)
        plots['solution_space'] = generate_solution_space_plot(best_solution, bounds, variable_names)
        
        if request.track_diversity and ga.diversity_history:
            plots['population_evolution'] = generate_population_evolution_plot(
                ga.population_history, ga.diversity_history
            )
        
        # Generate interpretation
        interpretation = generate_interpretation(result, request)
        
        return {
            'success': True,
            'best_fitness': _to_native_type(result['best_fitness']),
            'convergence_rate': _to_native_type(result['convergence_rate']),
            'population_diversity': _to_native_type(result['final_diversity']),
            'efficiency': _to_native_type(result['efficiency']),
            'best_solution': [_to_native_type(x) for x in best_solution],
            'selected_variables': selected_variables,
            'variable_details': variable_details,
            'variable_details_by_range': variable_details_by_range,
            'problem': {
                'n_variables': n_variables,
                'generations': result['generations_completed'],
                'n_selected': len(selected_variables)
            },
            'plots': plots,
            'interpretation': interpretation,
            'parameters': {
                'objective_function': request.objective_function,
                'population_size': request.population_size,
                'generations': request.generations,
                'mutation_rate': request.mutation_rate,
                'crossover_rate': request.crossover_rate,
                'elite_size': request.elite_size
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Genetic algorithm optimization failed: {str(e)}")
