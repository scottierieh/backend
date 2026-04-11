"""
Stochastic Programming using Pyomo
Two-stage stochastic programming with scenarios
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
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

router = APIRouter()


class VariableInput(BaseModel):
    """Variable configuration"""
    name: str
    min_value: float
    max_value: float


class ScenarioInput(BaseModel):
    """Scenario configuration"""
    name: str
    probability: float = Field(ge=0.0, le=1.0)
    demand: float  # Simplified: just demand parameter


class StochasticRequest(BaseModel):
    """Request model for Stochastic Programming"""
    variables: List[VariableInput]
    scenarios: List[ScenarioInput]
    fixed_cost: float = Field(default=100.0)
    variable_cost: float = Field(default=2.0)
    shortage_penalty: float = Field(default=10.0)
    excess_penalty: float = Field(default=1.0)


class VariableDetail(BaseModel):
    """Variable detail information"""
    name: str
    min_value: float
    max_value: float
    optimal_value: float
    range: float
    selected: bool


class StochasticResponse(BaseModel):
    """Response model for Stochastic Programming"""
    success: bool
    expected_cost: float
    first_stage_decision: float
    scenario_costs: Dict[str, float]
    second_stage_decisions: Dict[str, float]
    selected_variables: List[str]
    variable_details: List[VariableDetail]
    problem: Dict[str, Any]
    plots: Dict[str, Optional[str]]
    interpretation: Dict[str, Any]


def solve_two_stage_stochastic(
    scenarios: List[ScenarioInput],
    x_min: float,
    x_max: float,
    fixed_cost: float,
    variable_cost: float,
    shortage_penalty: float,
    excess_penalty: float
):
    """
    Two-stage stochastic programming using Pyomo
    
    Stage 1: Decide production quantity (x)
    Stage 2: Handle shortage/excess for each scenario
    """
    
    # Create concrete model
    model = pyo.ConcreteModel()
    
    # Sets
    model.scenarios = pyo.Set(initialize=[s.name for s in scenarios])
    
    # Parameters
    model.probability = pyo.Param(
        model.scenarios,
        initialize={s.name: s.probability for s in scenarios}
    )
    model.demand = pyo.Param(
        model.scenarios,
        initialize={s.name: s.demand for s in scenarios}
    )
    
    # First-stage decision variable (production quantity)
    model.x = pyo.Var(domain=pyo.NonNegativeReals, bounds=(x_min, x_max))
    
    # Second-stage decision variables (shortage and excess per scenario)
    model.shortage = pyo.Var(model.scenarios, domain=pyo.NonNegativeReals)
    model.excess = pyo.Var(model.scenarios, domain=pyo.NonNegativeReals)
    
    # First-stage cost
    def first_stage_cost_rule(m):
        return fixed_cost + variable_cost * m.x
    model.first_stage_cost = pyo.Expression(rule=first_stage_cost_rule)
    
    # Second-stage cost (expected value over scenarios)
    def second_stage_cost_rule(m):
        return sum(
            m.probability[s] * (
                shortage_penalty * m.shortage[s] + 
                excess_penalty * m.excess[s]
            )
            for s in m.scenarios
        )
    model.second_stage_cost = pyo.Expression(rule=second_stage_cost_rule)
    
    # Objective: minimize expected total cost
    def objective_rule(m):
        return m.first_stage_cost + m.second_stage_cost
    model.objective = pyo.Objective(rule=objective_rule, sense=pyo.minimize)
    
    # Constraints: balance production, shortage, and excess
    def balance_rule(m, s):
        return m.x + m.shortage[s] - m.excess[s] == m.demand[s]
    model.balance = pyo.Constraint(model.scenarios, rule=balance_rule)
    
    # Solve using GLPK (open-source solver)
    solver = SolverFactory('glpk')
    results = solver.solve(model, tee=False)
    
    if results.solver.termination_condition != pyo.TerminationCondition.optimal:
        raise ValueError("Solver did not find optimal solution")
    
    # Extract results
    first_stage_decision = pyo.value(model.x)
    expected_cost = pyo.value(model.objective)
    
    scenario_costs = {}
    second_stage_decisions = {}
    
    for s in scenarios:
        shortage = pyo.value(model.shortage[s.name])
        excess = pyo.value(model.excess[s.name])
        cost = (fixed_cost + variable_cost * first_stage_decision + 
                shortage_penalty * shortage + excess_penalty * excess)
        
        scenario_costs[s.name] = cost
        second_stage_decisions[s.name] = {
            'shortage': shortage,
            'excess': excess,
            'actual_supply': first_stage_decision + shortage - excess
        }
    
    return (
        first_stage_decision,
        expected_cost,
        scenario_costs,
        second_stage_decisions
    )


def create_scenario_plot(scenarios: List[ScenarioInput], scenario_costs: Dict[str, float]) -> str:
    """Create scenario comparison plot"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Demands and Probabilities
    names = [s.name for s in scenarios]
    demands = [s.demand for s in scenarios]
    probs = [s.probability for s in scenarios]
    
    x_pos = np.arange(len(names))
    
    ax1.bar(x_pos, demands, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.set_xlabel('Scenario', fontsize=12, weight='bold')
    ax1.set_ylabel('Demand', fontsize=12, weight='bold')
    ax1.set_title('Demand by Scenario', fontsize=14, weight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add probability labels
    for i, (d, p) in enumerate(zip(demands, probs)):
        ax1.text(i, d, f'p={p:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Costs by Scenario
    costs = [scenario_costs[s.name] for s in scenarios]
    colors = plt.cm.RdYlGn_r(np.array(probs))
    
    ax2.bar(x_pos, costs, alpha=0.7, color=colors, edgecolor='black')
    ax2.set_xlabel('Scenario', fontsize=12, weight='bold')
    ax2.set_ylabel('Total Cost', fontsize=12, weight='bold')
    ax2.set_title('Cost by Scenario', fontsize=14, weight='bold')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return img_str


def create_decision_plot(first_stage: float, scenarios: List[ScenarioInput], second_stage: Dict) -> str:
    """Create decision visualization"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = [s.name for s in scenarios]
    demands = [s.demand for s in scenarios]
    actual_supplies = [second_stage[s.name]['actual_supply'] for s in scenarios]
    
    x_pos = np.arange(len(names))
    width = 0.35
    
    ax.bar(x_pos - width/2, demands, width, label='Demand', alpha=0.7, color='orange', edgecolor='black')
    ax.bar(x_pos + width/2, actual_supplies, width, label='Actual Supply', alpha=0.7, color='green', edgecolor='black')
    ax.axhline(first_stage, color='red', linestyle='--', linewidth=2, label=f'First-Stage Decision: {first_stage:.2f}')
    
    ax.set_xlabel('Scenario', fontsize=12, weight='bold')
    ax.set_ylabel('Quantity', fontsize=12, weight='bold')
    ax.set_title('First-Stage vs. Second-Stage Decisions', fontsize=14, weight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    img_str = base64.b64encode(buffer.read()).decode()
    plt.close()
    
    return img_str


def generate_interpretation(
    first_stage: float,
    expected_cost: float,
    scenarios: List[ScenarioInput],
    scenario_costs: Dict[str, float],
    second_stage: Dict
) -> Dict[str, Any]:
    """Generate insights"""
    
    key_insights = []
    recommendations = []
    
    # Expected cost analysis
    key_insights.append({
        "title": "Optimal Production Quantity",
        "description": f"First-stage decision: {first_stage:.2f} units. Expected total cost: ${expected_cost:.2f}",
        "status": "positive"
    })
    
    # Scenario analysis
    avg_demand = sum(s.demand * s.probability for s in scenarios)
    if first_stage < avg_demand:
        key_insights.append({
            "title": "Conservative Strategy",
            "description": f"Production ({first_stage:.2f}) below expected demand ({avg_demand:.2f}). Hedging against excess inventory.",
            "status": "neutral"
        })
        recommendations.append("Consider shortage costs - may be cheaper than excess inventory")
    elif first_stage > avg_demand:
        key_insights.append({
            "title": "Aggressive Strategy",
            "description": f"Production ({first_stage:.2f}) above expected demand ({avg_demand:.2f}). Preparing for high demand scenarios.",
            "status": "neutral"
        })
        recommendations.append("Excess inventory risk - ensure storage costs are manageable")
    
    # Shortage/excess analysis
    total_shortage = sum(second_stage[s.name]['shortage'] * s.probability for s in scenarios)
    total_excess = sum(second_stage[s.name]['excess'] * s.probability for s in scenarios)
    
    if total_shortage > 0:
        recommendations.append(f"Expected shortage: {total_shortage:.2f} units - consider increasing production")
    if total_excess > 0:
        recommendations.append(f"Expected excess: {total_excess:.2f} units - may reduce production")
    
    # Cost variability
    costs = list(scenario_costs.values())
    cost_range = max(costs) - min(costs)
    if cost_range > expected_cost * 0.5:
        key_insights.append({
            "title": "High Cost Variability",
            "description": f"Cost range: ${cost_range:.2f}. Significant uncertainty in outcomes.",
            "status": "warning"
        })
        recommendations.append("High variability - consider risk mitigation strategies")
    
    return {
        "key_insights": key_insights,
        "recommendations": recommendations
    }


@router.post("/stochastic-programming-pyomo")
async def optimize_stochastic_pyomo(request: StochasticRequest):
    """
    Two-stage stochastic programming using Pyomo
    """
    try:
        if len(request.scenarios) < 2:
            raise HTTPException(400, "At least 2 scenarios required")
        if len(request.variables) == 0:
            raise HTTPException(400, "At least one variable required")
        
        # For now, support single variable (production quantity)
        var = request.variables[0]
        
        # Solve
        first_stage, expected_cost, scenario_costs, second_stage = solve_two_stage_stochastic(
            request.scenarios,
            var.min_value,
            var.max_value,
            request.fixed_cost,
            request.variable_cost,
            request.shortage_penalty,
            request.excess_penalty
        )
        
        # Generate plots
        plots = {
            "scenarios": create_scenario_plot(request.scenarios, scenario_costs),
            "decisions": create_decision_plot(first_stage, request.scenarios, second_stage)
        }
        
        # Variable details
        variable_details = [VariableDetail(
            name=var.name,
            min_value=var.min_value,
            max_value=var.max_value,
            optimal_value=first_stage,
            range=var.max_value - var.min_value,
            selected=True
        )]
        
        # Interpretation
        interpretation = generate_interpretation(
            first_stage,
            expected_cost,
            request.scenarios,
            scenario_costs,
            second_stage
        )
        
        return StochasticResponse(
            success=True,
            expected_cost=float(expected_cost),
            first_stage_decision=float(first_stage),
            scenario_costs=scenario_costs,
            second_stage_decisions={
                s: {
                    'shortage': float(second_stage[s]['shortage']),
                    'excess': float(second_stage[s]['excess']),
                    'actual_supply': float(second_stage[s]['actual_supply'])
                }
                for s in second_stage
            },
            selected_variables=[var.name],
            variable_details=variable_details,
            problem={
                "n_variables": 1,
                "n_scenarios": len(request.scenarios),
                "fixed_cost": request.fixed_cost,
                "variable_cost": request.variable_cost,
                "shortage_penalty": request.shortage_penalty,
                "excess_penalty": request.excess_penalty
            },
            plots=plots,
            interpretation=interpretation
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Optimization error: {str(e)}")
