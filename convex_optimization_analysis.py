#!/usr/bin/env python3
"""Convex Optimization — minimise a (convex) objective under constraints.
Shares the SLSQP core with the NLP backend; adds a numeric convexity check of
the objective AND of the constraints.

Input: objective(str), var_names[], start[], con_expr[], con_ops[], con_rhs[], con_names[]
Output: results{..., is_convex_verified (objective: no curvature violation found),
                constraints_convex_verified (true / false / null = not testable),
                model_convex_verified, convexity_check{...}, constraint_convexity[{...}],
                convexity_note, ...}, plot

The convexity check is a numerical SAMPLING check (see _optexpr.curvature_check):
"no violation found at N points" — not a proof. Nothing here claims a global
optimum unconditionally.
"""
import sys, json
import numpy as np
from nonlinear_programming_analysis import solve_nlp, build_result, make_plot, not_converged_text
from _optexpr import CURV_TOL, curvature_check, curvature_sample_points


def _is_eq(op, ctype):
    return op in ("=", "==") or ctype == "eq"


def _ineq_region(gfuncs):
    """Points that satisfy every inequality (equalities are left out: random
    points never lie exactly on them)."""
    ineq = [(g, op, rhs) for (g, op, rhs, _n, ctype, _e) in gfuncs if not _is_eq(op, ctype)]

    def ok(p):
        for g, op, rhs in ineq:
            v = g(p)
            if not np.isfinite(v):
                return False
            tol = 1e-6 * max(1.0, abs(rhs))
            if op == ">=":
                if v < rhs - tol:
                    return False
            elif v > rhs + tol:
                return False
        return True
    return ok


def _brief(rep):
    return {k: rep[k] for k in ("ok", "require", "n_points", "n_directions", "n_tests", "n_violations", "worst")}


def model_convexity(fobj, x, gfuncs):
    """Objective must be convex; g(x) <= b needs g convex, g(x) >= b needs g
    concave, g(x) = b needs g affine. Returns the fields added to results."""
    pts = curvature_sample_points(x, region=_ineq_region(gfuncs))
    obj = curvature_check(fobj, pts, require="convex")
    cons = []
    for (g, op, rhs, name, ctype, expr) in gfuncs:
        need = "affine" if _is_eq(op, ctype) else ("concave" if op == ">=" else "convex")
        rep = curvature_check(g, pts, require=need)
        cons.append({"name": name, "expr": expr, "op": op, "type": "eq" if _is_eq(op, ctype) else "ineq",
                     "required": need, **{k: v for k, v in _brief(rep).items() if k != "require"}})
    bad = [c["name"] for c in cons if c["ok"] is False]
    untested = [c["name"] for c in cons if c["ok"] is None]
    cons_ok = False if bad else (None if untested else True)
    obj_ok = obj["ok"] is True
    model_ok = obj_ok and cons_ok is True

    n_pts = obj["n_points"]
    if obj["ok"] is None:
        obj_txt = ("The objective could not be evaluated around the solution, so its convexity was not checked. ")
    elif not obj_ok:
        w = obj["worst"]
        obj_txt = (f"The objective is NOT convex: its second derivative along direction {w['direction']} at "
                   f"{w['point']} is {w['second_derivative']:.4g} < 0 ({obj['n_violations']} of {obj['n_tests']} "
                   "sampled tests failed). ")
    else:
        obj_txt = (f"No convexity violation was found for the objective ({obj['n_tests']} second-derivative tests "
                   f"at {n_pts} sampled point(s) around the solution). ")
    if bad:
        con_txt = (f"Constraint(s) {', '.join(bad)} do not have the curvature a convex model needs "
                   "(≤ needs a convex left-hand side, ≥ a concave one, = a linear one), so the feasible region "
                   "may not be convex. ")
    elif untested:
        con_txt = f"Constraint(s) {', '.join(untested)} could not be evaluated around the solution and were not checked. "
    elif cons:
        con_txt = "Every constraint passed the same check (≤: convex, ≥: concave, =: linear). "
    else:
        con_txt = ""
    if model_ok:
        tail = ("This is a numerical sampling check, not a proof: if the model is indeed convex, this local "
                "minimum is also the global minimum.")
    else:
        tail = ("The model is not (verifiably) convex, so treat the result as a local minimum and check it "
                "from other starting points.")
    return {
        "is_convex_verified": bool(obj_ok),
        "constraints_convex_verified": cons_ok,
        "model_convex_verified": bool(model_ok),
        "convexity_check": {
            "method": "numeric_sampling", "proof": False,
            "tolerance": f"second directional derivative >= -{CURV_TOL:g} * max(1, |f(p)|)",
            "region": "the solution plus seeded random points around it that satisfy the inequality constraints",
            **_brief(obj),
        },
        "constraint_convexity": cons,
        "convexity_note": obj_txt + con_txt + tail,
    }


def main():
    try:
        payload = json.load(sys.stdin)
        payload["maximize"] = False
        res, fobj, maximize, var_names, ce, co, cr, gf = solve_nlp(payload, force_min=True)
        extra = model_convexity(fobj, res.x, gf) if np.all(np.isfinite(res.x)) else None
        out, x = build_result(res, fobj, maximize, var_names, ce, co, cr, gf, extra=extra)
        if out.get("unsolved"):
            print(json.dumps({"results": out, "plot": None})); return
        obj_val = out["objective_value"]
        out["interpretation"] = (
            ("The minimum objective value is " if out["status"] == "optimal"
             else not_converged_text(out) + "The objective there is ")
            + f"{obj_val:.4f} at "
            + ", ".join(f"{var_names[i]}={x[i]:.4f}" for i in range(len(var_names))) + ". "
            + out["convexity_note"]
            + (" Convex problems are the well-behaved case in optimization: any local minimum is global and "
               "solvers converge reliably." if out["model_convex_verified"] else "")
        )
        print(json.dumps({"results": out, "plot": make_plot(fobj, var_names, x)}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
