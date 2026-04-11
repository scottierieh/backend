"""
Particle Swarm Optimization: Swarm Intelligence Router for FastAPI
Bio-inspired optimization using collective intelligence of particle swarms
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


class ParticleSwarmRequest(BaseModel):
    """Particle swarm optimization request parameters"""
    
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
    
    # PSO parameters
    n_particles: int = Field(
        default=30,
        ge=5,
        le=200,
        description="Number of particles in swarm"
    )
    n_iterations: int = Field(
        default=100,
        ge=10,
        le=1000,
        description="Number of iterations"
    )
    inertia_weight: float = Field(
        default=0.5,
        ge=0.0,
        le=2.0,
        description="Inertia weight (w)"
    )
    cognitive_coeff: float = Field(
        default=1.5,
        ge=0.0,
        le=4.0,
        description="Cognitive coefficient (c1)"
    )
    social_coeff: float = Field(
        default=1.5,
        ge=0.0,
        le=4.0,
        description="Social coefficient (c2)"
    )
    
    # Advanced options
    inertia_decay: bool = Field(
        default=True,
        description="Enable inertia weight decay"
    )
    velocity_clamping: bool = Field(
        default=True,
        description="Enable velocity clamping"
    )
    topology: str = Field(
        default="global",
        description="Swarm topology (global, ring, star)"
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
        result = eval(func_str, allowed_globals)
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
# Particle Swarm Optimization Implementation
# =============================================================================

class Particle:
    """Individual particle in the swarm"""
    
    def __init__(self, dimensions: int, bounds: List[tuple]):
        self.dimensions = dimensions
        self.bounds = bounds
        
        # Initialize position randomly within bounds
        self.position = np.array([
            np.random.uniform(bounds[i][0], bounds[i][1]) 
            for i in range(dimensions)
        ])
        
        # Initialize velocity
        velocity_range = [(bounds[i][1] - bounds[i][0]) * 0.1 for i in range(dimensions)]
        self.velocity = np.array([
            np.random.uniform(-velocity_range[i], velocity_range[i]) 
            for i in range(dimensions)
        ])
        
        # Personal best
        self.best_position = self.position.copy()
        self.best_fitness = float('inf')
        self.fitness = float('inf')
        
        # History for visualization
        self.position_history = [self.position.copy()]
        self.fitness_history = []


class ParticleSwarmOptimizer:
    """Particle Swarm Optimization algorithm"""
    
    def __init__(self, objective_func: str, bounds: List[tuple], 
                 n_particles: int = 30, n_iterations: int = 100,
                 inertia_weight: float = 0.5, cognitive_coeff: float = 1.5,
                 social_coeff: float = 1.5, inertia_decay: bool = True,
                 velocity_clamping: bool = True, topology: str = "global"):
        
        self.objective_func = objective_func
        self.bounds = bounds
        self.n_dimensions = len(bounds)
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = inertia_weight  # Inertia weight
        self.c1 = cognitive_coeff  # Cognitive coefficient
        self.c2 = social_coeff  # Social coefficient
        self.inertia_decay = inertia_decay
        self.velocity_clamping = velocity_clamping
        self.topology = topology
        
        # Initialize particles
        self.particles = [Particle(self.n_dimensions, bounds) for _ in range(n_particles)]
        
        # Global best
        self.global_best_position = None
        self.global_best_fitness = float('inf')
        
        # History tracking
        self.global_fitness_history = []
        self.diversity_history = []
        self.swarm_positions_history = []
        
        # Velocity limits
        self.v_max = np.array([(bounds[i][1] - bounds[i][0]) * 0.2 for i in range(self.n_dimensions)])
    
    def evaluate_fitness(self, position: np.ndarray) -> float:
        """Evaluate fitness for a given position"""
        return evaluate_objective_function(self.objective_func, position)
    
    def update_global_best(self):
        """Update global best position"""
        for particle in self.particles:
            if particle.fitness < self.global_best_fitness:
                self.global_best_fitness = particle.fitness
                self.global_best_position = particle.position.copy()
    
    def calculate_diversity(self) -> float:
        """Calculate swarm diversity (average distance from centroid)"""
        positions = np.array([particle.position for particle in self.particles])
        centroid = np.mean(positions, axis=0)
        distances = [np.linalg.norm(pos - centroid) for pos in positions]
        return np.mean(distances)
    
    def update_particle(self, particle: Particle, iteration: int):
        """Update particle velocity and position"""
        # Current inertia weight (with optional decay)
        if self.inertia_decay:
            w_current = self.w * (1 - iteration / self.n_iterations)
        else:
            w_current = self.w
        
        # Random factors
        r1 = np.random.random(self.n_dimensions)
        r2 = np.random.random(self.n_dimensions)
        
        # Cognitive component (personal best)
        cognitive = self.c1 * r1 * (particle.best_position - particle.position)
        
        # Social component (global best)
        social = self.c2 * r2 * (self.global_best_position - particle.position)
        
        # Update velocity
        particle.velocity = (w_current * particle.velocity + cognitive + social)
        
        # Velocity clamping
        if self.velocity_clamping:
            particle.velocity = np.clip(particle.velocity, -self.v_max, self.v_max)
        
        # Update position
        particle.position += particle.velocity
        
        # Boundary handling (reflect)
        for i in range(self.n_dimensions):
            if particle.position[i] < self.bounds[i][0]:
                particle.position[i] = self.bounds[i][0]
                particle.velocity[i] *= -0.5  # Reflect and dampen
            elif particle.position[i] > self.bounds[i][1]:
                particle.position[i] = self.bounds[i][1]
                particle.velocity[i] *= -0.5  # Reflect and dampen
    
    def optimize(self) -> Dict:
        """Run PSO optimization"""
        
        # Initialize particle fitnesses
        for particle in self.particles:
            particle.fitness = self.evaluate_fitness(particle.position)
            particle.best_fitness = particle.fitness
            particle.fitness_history.append(particle.fitness)
        
        # Initialize global best
        self.update_global_best()
        self.global_fitness_history.append(self.global_best_fitness)
        
        # Store initial positions
        initial_positions = np.array([p.position.copy() for p in self.particles])
        self.swarm_positions_history.append(initial_positions)
        
        # Main optimization loop
        for iteration in range(self.n_iterations):
            # Update each particle
            for particle in self.particles:
                self.update_particle(particle, iteration)
                
                # Evaluate new fitness
                particle.fitness = self.evaluate_fitness(particle.position)
                particle.fitness_history.append(particle.fitness)
                
                # Update personal best
                if particle.fitness < particle.best_fitness:
                    particle.best_fitness = particle.fitness
                    particle.best_position = particle.position.copy()
                
                # Store position history
                particle.position_history.append(particle.position.copy())
            
            # Update global best
            self.update_global_best()
            self.global_fitness_history.append(self.global_best_fitness)
            
            # Calculate and store diversity
            diversity = self.calculate_diversity()
            self.diversity_history.append(diversity)
            
            # Store swarm positions (sample every few iterations for memory efficiency)
            if iteration % max(1, self.n_iterations // 20) == 0:
                current_positions = np.array([p.position.copy() for p in self.particles])
                self.swarm_positions_history.append(current_positions)
        
        # Calculate final metrics
        initial_fitness = self.global_fitness_history[0]
        final_fitness = self.global_best_fitness
        
        if initial_fitness > 0:
            convergence_rate = max(0, (initial_fitness - final_fitness) / initial_fitness * 100)
        else:
            convergence_rate = 100 if final_fitness < abs(initial_fitness) * 0.1 else 0
        
        final_diversity = self.diversity_history[-1] if self.diversity_history else 0
        
        # Calculate efficiency (how quickly it converged)
        efficiency = 100
        if len(self.global_fitness_history) > 10:
            # Find when 90% of improvement was achieved
            target_improvement = 0.9 * (initial_fitness - final_fitness)
            target_fitness = initial_fitness - target_improvement
            
            for i, fitness in enumerate(self.global_fitness_history):
                if fitness <= target_fitness:
                    efficiency = max(10, 100 - (i / len(self.global_fitness_history)) * 100)
                    break
        
        return {
            'best_solution': self.global_best_position,
            'best_fitness': self.global_best_fitness,
            'convergence_rate': convergence_rate,
            'final_diversity': final_diversity,
            'efficiency': efficiency,
            'iterations_completed': len(self.global_fitness_history) - 1
        }


# =============================================================================
# Plotting Functions
# =============================================================================

def generate_convergence_plot(fitness_history: List[float], diversity_history: List[float]) -> str:
    """Generate convergence plot with fitness and diversity"""
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    iterations = range(len(fitness_history))
    
    # Top: Fitness convergence
    ax1.plot(iterations, fitness_history, 'b-', linewidth=2, alpha=0.8, label='Global Best Fitness')
    ax1.fill_between(iterations, fitness_history, alpha=0.3)
    
    # Mark key points
    min_fitness = min(fitness_history)
    min_iter = fitness_history.index(min_fitness)
    ax1.plot(min_iter, min_fitness, 'ro', markersize=8, label=f'Best: {min_fitness:.6f}')
    
    ax1.set_xlabel('Iteration', fontsize=11, fontweight='600')
    ax1.set_ylabel('Best Fitness', fontsize=11, fontweight='600')
    ax1.set_title('PSO Convergence Progress', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log' if max(fitness_history) / min(fitness_history) > 100 else 'linear')
    
    # Bottom: Swarm diversity
    if diversity_history:
        div_iterations = range(len(diversity_history))
        ax2.plot(div_iterations, diversity_history, 'g-', linewidth=2, alpha=0.8, label='Swarm Diversity')
        ax2.fill_between(div_iterations, diversity_history, alpha=0.3, color='green')
        
        ax2.set_xlabel('Iteration', fontsize=11, fontweight='600')
        ax2.set_ylabel('Diversity', fontsize=11, fontweight='600')
        ax2.set_title('Swarm Diversity Evolution', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_swarm_behavior_plot(swarm_history: List[np.ndarray], 
                                global_best_position: np.ndarray) -> str:
    """Generate swarm behavior visualization"""
    
    if len(swarm_history) == 0 or len(swarm_history[0][0]) < 2:
        # Cannot visualize if no data or less than 2 dimensions
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'Swarm behavior visualization\nrequires 2+ dimensions', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('Swarm Behavior Analysis', fontsize=14, fontweight='bold')
        return _fig_to_base64(fig)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left: Swarm evolution (first 2 dimensions)
    colors = plt.cm.viridis(np.linspace(0, 1, len(swarm_history)))
    
    for i, (positions, color) in enumerate(zip(swarm_history, colors)):
        ax1.scatter(positions[:, 0], positions[:, 1], c=[color], alpha=0.6, s=30,
                   label=f'Iter {i * len(swarm_history) // len(swarm_history)}')
    
    # Mark global best
    ax1.scatter(global_best_position[0], global_best_position[1], 
               c='red', s=200, marker='*', edgecolor='darkred', linewidth=2,
               label='Global Best', zorder=10)
    
    ax1.set_xlabel('X1', fontsize=11, fontweight='600')
    ax1.set_ylabel('X2', fontsize=11, fontweight='600')
    ax1.set_title('Swarm Evolution in Search Space', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add legend with fewer entries
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[::max(1, len(handles)//6)], labels[::max(1, len(labels)//6)], 
              fontsize=9, loc='upper right')
    
    # Right: Particle density heatmap (final positions)
    final_positions = swarm_history[-1]
    
    try:
        # Create 2D histogram
        H, xedges, yedges = np.histogram2d(final_positions[:, 0], final_positions[:, 1], bins=20)
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        
        im = ax2.imshow(H.T, origin='lower', extent=extent, cmap='Blues', alpha=0.7)
        ax2.scatter(final_positions[:, 0], final_positions[:, 1], c='navy', alpha=0.8, s=20)
        ax2.scatter(global_best_position[0], global_best_position[1], 
                   c='red', s=200, marker='*', edgecolor='darkred', linewidth=2)
        
        plt.colorbar(im, ax=ax2, label='Particle Density')
        
    except:
        # Fallback to simple scatter
        ax2.scatter(final_positions[:, 0], final_positions[:, 1], alpha=0.7, s=30)
        ax2.scatter(global_best_position[0], global_best_position[1], 
                   c='red', s=200, marker='*', edgecolor='darkred', linewidth=2)
    
    ax2.set_xlabel('X1', fontsize=11, fontweight='600')
    ax2.set_ylabel('X2', fontsize=11, fontweight='600')
    ax2.set_title('Final Swarm Distribution', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_solution_space_plot(best_solution: np.ndarray, bounds: List[tuple],
                                variable_names: List[str]) -> str:
    """Generate solution space visualization"""
    
    n_vars = len(best_solution)
    
    if n_vars == 2:
        # 2D solution space
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot optimal point
        ax.plot(best_solution[0], best_solution[1], 'r*', markersize=20, 
               label=f'Optimal: ({best_solution[0]:.3f}, {best_solution[1]:.3f})')
        
        # Add bounds rectangle
        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]
        
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                           linewidth=2, edgecolor='blue', facecolor='none', alpha=0.7,
                           label='Search Space')
        ax.add_patch(rect)
        
        ax.set_xlabel(f'{variable_names[0]}', fontsize=12, fontweight='600')
        ax.set_ylabel(f'{variable_names[1]}', fontsize=12, fontweight='600')
        ax.set_title('Solution Space (2D)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Set appropriate limits
        ax.set_xlim(x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min))
        ax.set_ylim(y_min - 0.1 * (y_max - y_min), y_max + 0.1 * (y_max - y_min))
        
    else:
        # Parallel coordinates plot for >2 variables
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Normalize solution to [0, 1] for plotting
        normalized_solution = []
        for i, (val, (min_val, max_val)) in enumerate(zip(best_solution, bounds)):
            norm_val = (val - min_val) / (max_val - min_val) if max_val != min_val else 0.5
            normalized_solution.append(norm_val)
        
        x_pos = range(n_vars)
        ax.plot(x_pos, normalized_solution, 'o-', linewidth=3, markersize=8, 
               color='red', alpha=0.8, label='Optimal Solution')
        
        # Add value labels
        for i, (pos, val, norm_val) in enumerate(zip(x_pos, best_solution, normalized_solution)):
            ax.annotate(f'{val:.3f}', (pos, norm_val), 
                       textcoords="offset points", xytext=(0,10), ha='center',
                       fontsize=10, fontweight='600')
        
        # Add bound indicators
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='Lower Bounds')
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Upper Bounds')
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(variable_names, rotation=45, ha='right')
        ax.set_ylabel('Normalized Value', fontsize=12, fontweight='600')
        ax.set_title('Optimal Solution (Parallel Coordinates)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.1, 1.1)
        ax.legend(fontsize=10)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_particle_trajectories_plot(particles: List, bounds: List[tuple]) -> str:
    """Generate particle trajectory visualization"""
    
    if len(particles) == 0 or len(particles[0].position_history) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No trajectory data available', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title('Particle Trajectories', fontsize=14, fontweight='bold')
        return _fig_to_base64(fig)
    
    n_dims = len(particles[0].position_history[0])
    
    if n_dims >= 2:
        # 2D trajectory plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot sample trajectories (avoid overcrowding)
        n_sample = min(10, len(particles))
        sample_particles = particles[:n_sample]
        
        colors = plt.cm.tab10(np.linspace(0, 1, n_sample))
        
        for i, (particle, color) in enumerate(zip(sample_particles, colors)):
            trajectory = np.array(particle.position_history)
            ax.plot(trajectory[:, 0], trajectory[:, 1], 'o-', alpha=0.7, 
                   linewidth=2, markersize=4, color=color, label=f'Particle {i+1}')
            
            # Mark start and end
            ax.plot(trajectory[0, 0], trajectory[0, 1], 's', markersize=8, color=color)
            ax.plot(trajectory[-1, 0], trajectory[-1, 1], '^', markersize=8, color=color)
        
        # Add search space bounds
        x_min, x_max = bounds[0]
        y_min, y_max = bounds[1]
        rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                           linewidth=2, edgecolor='black', facecolor='none', alpha=0.3)
        ax.add_patch(rect)
        
        ax.set_xlabel('X1', fontsize=12, fontweight='600')
        ax.set_ylabel('X2', fontsize=12, fontweight='600')
        ax.set_title('Particle Trajectories in Search Space', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9, bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
    else:
        # 1D trajectory plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        n_sample = min(5, len(particles))
        sample_particles = particles[:n_sample]
        
        for i, particle in enumerate(sample_particles):
            trajectory = [pos[0] for pos in particle.position_history]
            iterations = range(len(trajectory))
            ax.plot(iterations, trajectory, 'o-', alpha=0.7, linewidth=2, 
                   label=f'Particle {i+1}')
        
        # Add bounds
        ax.axhline(y=bounds[0][0], color='red', linestyle='--', alpha=0.7, label='Lower Bound')
        ax.axhline(y=bounds[0][1], color='red', linestyle='--', alpha=0.7, label='Upper Bound')
        
        ax.set_xlabel('Iteration', fontsize=12, fontweight='600')
        ax.set_ylabel('Position', fontsize=12, fontweight='600')
        ax.set_title('Particle Position Evolution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Analysis and Interpretation
# =============================================================================

def generate_interpretation(result: Dict, request: ParticleSwarmRequest) -> Dict:
    """Generate interpretation of PSO results"""
    
    key_insights = []
    
    best_fitness = result['best_fitness']
    convergence_rate = result['convergence_rate']
    efficiency = result['efficiency']
    final_diversity = result['final_diversity']
    iterations_completed = result['iterations_completed']
    
    # Convergence analysis
    if convergence_rate > 80:
        key_insights.append({
            'title': 'Excellent Convergence',
            'description': f'Swarm achieved {convergence_rate:.1f}% improvement from initial solutions.',
            'status': 'positive'
        })
    elif convergence_rate > 40:
        key_insights.append({
            'title': 'Good Convergence',
            'description': f'Swarm achieved {convergence_rate:.1f}% improvement. Consider more iterations.',
            'status': 'neutral'
        })
    else:
        key_insights.append({
            'title': 'Limited Convergence',
            'description': f'Only {convergence_rate:.1f}% improvement achieved. May need parameter tuning.',
            'status': 'warning'
        })
    
    # Diversity analysis
    if final_diversity > 1.0:
        key_insights.append({
            'title': 'Good Exploration',
            'description': f'Final diversity of {final_diversity:.2f} indicates good exploration of search space.',
            'status': 'positive'
        })
    elif final_diversity < 0.1:
        key_insights.append({
            'title': 'Premature Convergence Risk',
            'description': f'Low final diversity ({final_diversity:.2f}) may indicate premature convergence.',
            'status': 'warning'
        })
    
    # Efficiency analysis
    if efficiency > 80:
        key_insights.append({
            'title': 'Efficient Search',
            'description': f'Algorithm converged efficiently with {efficiency:.1f}% efficiency.',
            'status': 'positive'
        })
    elif efficiency < 40:
        key_insights.append({
            'title': 'Slow Convergence',
            'description': f'Low efficiency ({efficiency:.1f}%). Consider adjusting PSO parameters.',
            'status': 'warning'
        })
    
    # Solution quality analysis
    if abs(best_fitness) < 1e-6:
        key_insights.append({
            'title': 'Near-Optimal Solution',
            'description': f'Found solution with fitness {best_fitness:.2e}, very close to global optimum.',
            'status': 'positive'
        })
    elif abs(best_fitness) < 1e-2:
        key_insights.append({
            'title': 'Good Solution Quality',
            'description': f'Found good solution with fitness {best_fitness:.4f}.',
            'status': 'neutral'
        })
    
    # Parameter analysis
    n_vars = len(request.variables)
    if request.n_particles < 5 * n_vars:
        key_insights.append({
            'title': 'Small Swarm Size',
            'description': f'Consider increasing swarm size for {n_vars} variables (recommended: {5 * n_vars}+ particles).',
            'status': 'neutral'
        })
    
    # Generate recommendations
    recommendations = []
    
    if convergence_rate < 30:
        recommendations.append("Consider increasing iterations or adjusting inertia weight for better convergence.")
    
    if final_diversity < 0.1:
        recommendations.append("Low diversity suggests premature convergence. Try reducing inertia or increasing swarm size.")
    
    if efficiency < 50:
        recommendations.append("Adjust cognitive/social coefficients. Try c1=c2=2.0 for balanced exploration/exploitation.")
    
    if iterations_completed == request.n_iterations:
        recommendations.append("Algorithm used all iterations. Consider running longer for potential improvement.")
    
    if request.n_particles < 20 and n_vars > 3:
        recommendations.append(f"For {n_vars} variables, consider using at least {max(20, 5 * n_vars)} particles.")
    
    if not recommendations:
        recommendations.append("PSO completed successfully with good parameters for this problem.")
    
    return {
        'key_insights': key_insights,
        'recommendations': recommendations
    }


# =============================================================================
# API Endpoint
# =============================================================================

@router.post("/particle-swarm")
async def optimize_particle_swarm(request: ParticleSwarmRequest) -> Dict:
    """
    Global optimization using Particle Swarm Optimization.
    
    Implements swarm intelligence for finding global optima:
    1. Population-based metaheuristic
    2. Social and cognitive learning
    3. Dynamic inertia weight control
    4. Velocity clamping for stability
    
    Returns optimal solution, swarm behavior analysis, and convergence history.
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
        
        # Initialize and run PSO
        pso = ParticleSwarmOptimizer(
            objective_func=request.objective_function,
            bounds=bounds,
            n_particles=request.n_particles,
            n_iterations=request.n_iterations,
            inertia_weight=request.inertia_weight,
            cognitive_coeff=request.cognitive_coeff,
            social_coeff=request.social_coeff,
            inertia_decay=request.inertia_decay,
            velocity_clamping=request.velocity_clamping,
            topology=request.topology
        )
        
        result = pso.optimize()
        best_solution = result['best_solution']
        
        # Calculate variable details
        variable_details = []
        variable_details_by_range = []
        selected_variables = []
        
        for i, (var, value) in enumerate(zip(request.variables, best_solution)):
            var_range = var.max_value - var.min_value
            selected = True  # All variables are always "selected" in PSO
            
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
        plots['convergence'] = generate_convergence_plot(
            pso.global_fitness_history, pso.diversity_history
        )
        plots['swarm_behavior'] = generate_swarm_behavior_plot(
            pso.swarm_positions_history, best_solution
        )
        plots['solution_space'] = generate_solution_space_plot(
            best_solution, bounds, variable_names
        )
        plots['particle_trajectories'] = generate_particle_trajectories_plot(
            pso.particles, bounds
        )
        
        # Generate interpretation
        interpretation = generate_interpretation(result, request)
        
        return {
            'success': True,
            'best_fitness': _to_native_type(result['best_fitness']),
            'convergence_rate': _to_native_type(result['convergence_rate']),
            'swarm_diversity': _to_native_type(result['final_diversity']),
            'efficiency': _to_native_type(result['efficiency']),
            'best_solution': [_to_native_type(x) for x in best_solution],
            'selected_variables': selected_variables,
            'variable_details': variable_details,
            'variable_details_by_range': variable_details_by_range,
            'problem': {
                'n_variables': n_variables,
                'iterations': result['iterations_completed'],
                'n_selected': len(selected_variables)
            },
            'plots': plots,
            'interpretation': interpretation,
            'parameters': {
                'objective_function': request.objective_function,
                'n_particles': request.n_particles,
                'n_iterations': request.n_iterations,
                'inertia_weight': request.inertia_weight,
                'cognitive_coeff': request.cognitive_coeff,
                'social_coeff': request.social_coeff,
                'inertia_decay': request.inertia_decay,
                'velocity_clamping': request.velocity_clamping,
                'topology': request.topology
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Particle swarm optimization failed: {str(e)}")
