#!/usr/bin/env python3
"""Convex Optimization — minimise a (convex) objective under constraints.
Shares the SLSQP core with the NLP backend; adds a numeric convexity check.

Input: objective(str), var_names[], start[], con_expr[], con_ops[], con_rhs[], con_names[]
Output: results{..., is_convex_verified, convexity_note, ...}, plot
"""
import sys, json
import numpy as np
from nonlinear_programming_analysis import solve_nlp, build_result, make_plot
from _optexpr import is_convex_numeric


def main():
    try:
        payload = json.load(sys.stdin)
        payload["maximize"] = False
        res, fobj, maximize, var_names, ce, co, cr, gf = solve_nlp(payload, force_min=True)
        convex, (ok, tot) = is_convex_numeric(fobj, res.x)
        out, x = build_result(res, fobj, maximize, var_names, ce, co, cr, gf,
                              extra={"is_convex_verified": bool(convex),
                                     "convexity_note": (
                                         "The objective passed a local convexity check at the solution, so this "
                                         "minimum is the global one." if convex else
                                         f"Local convexity check was inconclusive ({ok}/{tot} directions) — treat "
                                         "the result as a local minimum and verify from other starting points.")})
        if out.get("unsolved"):
            print(json.dumps({"results": out, "plot": None})); return
        obj_val = out["objective_value"]
        out["interpretation"] = (
            f"The minimum objective value is {obj_val:.4f} at "
            + ", ".join(f"{var_names[i]}={x[i]:.4f}" for i in range(len(var_names))) + ". "
            + out["convexity_note"] + " Convex problems are the well-behaved case in optimization: any local "
            "minimum is global and solvers converge reliably."
        )
        print(json.dumps({"results": out, "plot": make_plot(fobj, var_names, x)}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
