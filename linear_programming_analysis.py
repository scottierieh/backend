
import sys
import json
import numpy as np

try:
    from scipy.optimize import linprog, milp
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

def _to_native_type(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def solve_lp_and_dual(c, A_ub, b_ub, A_eq, b_eq, bounds, objective='maximize'):
    """
    Solves the primal LP.
    """
    c_solver = -np.array(c) if objective == 'maximize' else np.array(c)
    
    res = linprog(c_solver, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if not res.success:
        return {"error": f"Primal problem could not be solved: {res.message}", "success": False}

    primal_solution = res.x
    primal_optimal_value = -res.fun if objective == 'maximize' else res.fun

    return {
        "success": True,
        "primal_solution": primal_solution.tolist(),
        "primal_optimal_value": primal_optimal_value,
    }
    
def solve_integer_lp(c, A_ub, b_ub, A_eq, b_eq, bounds, integrality, objective='maximize'):
    """
    Solves a Mixed-Integer Linear Program (MILP).
    """
    c_solver = -np.array(c) if objective == 'maximize' else np.array(c)
    
    constraints = []
    if A_ub is not None and A_ub.shape[0] > 0:
        constraints.append(milp(A_ub, ub=b_ub))
    if A_eq is not None and A_eq.shape[0] > 0:
        constraints.append(milp(A_eq, lb=b_eq, ub=b_eq))
        
    res = milp(c=c_solver, constraints=constraints, integrality=integrality, bounds=bounds)

    if not res.success:
        return {"error": f"Integer problem could not be solved: {res.message}", "success": False}

    solution = res.x
    optimal_value = -res.fun if objective == 'maximize' else res.fun
    
    return {
        "success": True,
        "solution": solution.tolist(),
        "optimal_value": optimal_value
    }


def main():
    if not SCIPY_AVAILABLE:
        print(json.dumps({"error": "SciPy library is not installed. Please install it to use this feature."}), file=sys.stderr)
        sys.exit(1)

    try:
        payload = json.load(sys.stdin)
        c = payload.get('c')
        A = np.array(payload.get('A'))
        b = np.array(payload.get('b'))
        constraint_types = payload.get('constraint_types')
        objective = payload.get('objective', 'maximize')
        problem_type = payload.get('problem_type', 'lp')
        variable_types = payload.get('variable_types')

        if not all([c is not None, A is not None, b is not None, constraint_types is not None]):
            raise ValueError("Missing required parameters: c, A, b, or constraint_types")
        
        num_vars = len(c)
        bounds = (0, None) # Non-negativity constraint

        if problem_type == 'integer' or problem_type == 'milp':
             integrality = np.array([1 if v_type == 'integer' else 0 for v_type in variable_types])
             A_ub, b_ub = [], []
             A_eq, b_eq = [], []
             for i, c_type in enumerate(constraint_types):
                if c_type == '<=':
                    A_ub.append(A[i])
                    b_ub.append(b[i])
                elif c_type == '>=':
                    A_ub.append(-A[i])
                    b_ub.append(-b[i])
                elif c_type == '==':
                    A_eq.append(A[i])
                    b_eq.append(b[i])
             
             result = solve_integer_lp(c, 
                                       np.array(A_ub) if A_ub else None, 
                                       np.array(b_ub) if b_ub else None, 
                                       np.array(A_eq) if A_eq else None,
                                       np.array(b_eq) if b_eq else None,
                                       bounds, integrality, objective)
        else: # Standard LP
            A_ub = [A[i] for i, t in enumerate(constraint_types) if t == '<=']
            b_ub = [b[i] for i, t in enumerate(constraint_types) if t == '<=']
            
            for i, t in enumerate(constraint_types):
                if t == '>=':
                    A_ub.append(-A[i])
                    b_ub.append(-b[i])

            A_eq = [A[i] for i, t in enumerate(constraint_types) if t == '==']
            b_eq = [b[i] for i, t in enumerate(constraint_types) if t == '==']
            
            result = solve_lp_and_dual(c, 
                                   np.array(A_ub) if A_ub else None, 
                                   np.array(b_ub) if b_ub else None, 
                                   np.array(A_eq) if A_eq else None,
                                   np.array(b_eq) if b_eq else None,
                                   bounds, objective)

        if not result.get("success"):
            raise ValueError(result.get("error", "An unknown error occurred in the solver."))

        print(json.dumps(result, default=_to_native_type))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
