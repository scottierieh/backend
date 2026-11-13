#!/usr/bin/env python3
"""Constrained Optimization — min/max an objective subject to equality and
inequality constraints. Shares the SLSQP core with the NLP backend.

Input: objective(str), maximize(bool), var_names[], start[],
       con_expr[], con_types[] (ineq|eq), con_ops[], con_rhs[], con_names[]
Output: results{..., n_binding (binding INEQUALITIES only), n_inequality, n_equality,
                success, solver_status_code, solver_message, ...}, plot
"""
import sys, json
import numpy as np
from nonlinear_programming_analysis import solve_nlp, build_result, make_plot, not_converged_text


def main():
    try:
        payload = json.load(sys.stdin)
        res, fobj, maximize, var_names, ce, co, cr, gf = solve_nlp(payload)
        out, x = build_result(res, fobj, maximize, var_names, ce, co, cr, gf)
        if out.get("unsolved"):
            print(json.dumps({"results": out, "plot": None})); return
        # The page reports this as "binding inequalities / total inequalities",
        # so equalities — always binding by definition — are counted separately
        # there and must not be included here.
        # The interpretation used to say "n_binding of len(gf)", mixing this
        # inequality-only count with a total that includes the equalities.
        # Both counts are now reported (and phrased) separately.
        ineq = [c for c in out["constraints"] if c["type"] != "eq"]
        n_eq = len(out["constraints"]) - len(ineq)
        n_binding = sum(1 for c in ineq if c["binding"])
        out["n_binding"] = n_binding
        out["n_inequality"] = len(ineq)
        out["n_equality"] = n_eq
        obj_val = out["objective_value"]
        if ineq and n_eq:
            con_text = (f"{n_binding} of {len(ineq)} inequality constraint(s) are binding (active at the optimum), "
                        f"and the {n_eq} equality constraint(s) hold exactly by definition. ")
        elif ineq:
            con_text = f"{n_binding} of {len(ineq)} inequality constraint(s) are binding (active at the optimum). "
        elif n_eq:
            con_text = f"All {n_eq} constraint(s) are equalities, which hold exactly by definition. "
        else:
            con_text = "There are no constraints. "
        out["interpretation"] = (
            (f"The {'maximum' if maximize else 'minimum'} is {obj_val:.4f} at " if out["status"] == "optimal"
             else not_converged_text(out) + f"The objective there is {obj_val:.4f}, at ")
            + ", ".join(f"{var_names[i]}={x[i]:.4f}" for i in range(len(var_names))) + ". "
            + con_text
            + ("Binding constraints are the limits actually shaping the solution; relaxing one would improve "
               "the objective, while non-binding inequalities have room to spare." if out["constraints"] else "")
        )
        print(json.dumps({"results": out, "plot": make_plot(fobj, var_names, x)}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
