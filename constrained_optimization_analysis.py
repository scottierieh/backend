#!/usr/bin/env python3
"""Constrained Optimization — min/max an objective subject to equality and
inequality constraints. Shares the SLSQP core with the NLP backend.

Input: objective(str), maximize(bool), var_names[], start[],
       con_expr[], con_types[] (ineq|eq), con_ops[], con_rhs[], con_names[]
Output: results{..., n_binding, ...}, plot
"""
import sys, json
import numpy as np
from nonlinear_programming_analysis import solve_nlp, build_result, make_plot


def main():
    try:
        payload = json.load(sys.stdin)
        res, fobj, maximize, var_names, ce, co, cr, gf = solve_nlp(payload)
        out, x = build_result(res, fobj, maximize, var_names, ce, co, cr, gf)
        if out.get("unsolved"):
            print(json.dumps({"results": out, "plot": None})); return
        n_binding = sum(1 for c in out["constraints"]
                        if c["slack"] is not None and abs(c["slack"]) < 1e-4)
        out["n_binding"] = n_binding
        obj_val = out["objective_value"]
        out["interpretation"] = (
            f"The {'maximum' if maximize else 'minimum'} is {obj_val:.4f} at "
            + ", ".join(f"{var_names[i]}={x[i]:.4f}" for i in range(len(var_names))) + ". "
            f"{n_binding} of {len(gf)} constraint(s) are binding (active at the optimum) — these are the "
            "limits actually shaping the solution; relaxing a binding constraint would improve the objective, "
            "while non-binding ones have room to spare."
        )
        print(json.dumps({"results": out, "plot": make_plot(fobj, var_names, x)}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
