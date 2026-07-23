#!/usr/bin/env python3
"""Nonlinear Programming (NLP) — min/max a nonlinear objective under nonlinear
constraints. scipy.optimize.minimize (SLSQP).

Input: objective(str), maximize(bool), var_names[], start[],
       con_expr[], con_ops[] (<=|>=|=), con_rhs[], con_names[]
Output: results{status, unsolved, message, maximize, n_vars, n_constraints,
                objective_value, iterations, variables:[{name,value}],
                constraints:[{name,op,lhs,rhs,satisfied,slack}], interpretation}, plot
"""
import sys, json, io, base64
import numpy as np
from scipy.optimize import minimize
from _optexpr import make_func

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _fin(x, nd=6):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def solve_nlp(payload, force_min=False):
    obj_s = payload.get("objective")
    if not obj_s:
        raise ValueError("Provide an objective expression.")
    var_names = [str(v) for v in (payload.get("var_names") or [])]
    n = len(var_names)
    if n < 1:
        raise ValueError("Provide at least one variable.")
    x0 = [float(v) for v in (payload.get("start") or [1.0]*n)]
    if len(x0) != n:
        x0 = [1.0]*n
    maximize = (not force_min) and bool(payload.get("maximize", False))
    fobj = make_func(obj_s, var_names)
    f = (lambda x: -fobj(x)) if maximize else fobj

    con_expr = [str(c) for c in (payload.get("con_expr") or [])]
    con_ops = [str(o) for o in (payload.get("con_ops") or [])]
    con_rhs = [float(r) for r in (payload.get("con_rhs") or [])]
    con_types = payload.get("con_types") or None
    m = len(con_expr)
    cons = []
    gfuncs = []
    for i in range(m):
        g = make_func(con_expr[i], var_names); rhs = con_rhs[i]; op = con_ops[i] if i < len(con_ops) else "<="
        gfuncs.append((g, op, rhs))
        ctype = (con_types[i] if con_types and i < len(con_types) else None)
        if op in ("=", "==") or ctype == "eq":
            cons.append({"type": "eq", "fun": (lambda x, g=g, r=rhs: g(x) - r)})
        elif op == ">=":
            cons.append({"type": "ineq", "fun": (lambda x, g=g, r=rhs: g(x) - r)})
        else:  # <=
            cons.append({"type": "ineq", "fun": (lambda x, g=g, r=rhs: r - g(x))})

    res = minimize(f, np.array(x0, float), method="SLSQP", constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-9})
    return res, fobj, maximize, var_names, con_expr, con_ops, con_rhs, gfuncs


def build_result(res, fobj, maximize, var_names, con_expr, con_ops, con_rhs, gfuncs, extra=None):
    if not res.success and not np.all(np.isfinite(res.x)):
        return {"status": "failed", "unsolved": True,
                "message": f"Solver did not converge: {res.message}"}, None
    x = res.x
    obj_val = float(fobj(x))
    n = len(var_names)
    variables = [{"name": var_names[i], "value": _fin(x[i], 6)} for i in range(n)]
    constraints = []
    for i, (g, op, rhs) in enumerate(gfuncs):
        lhs = float(g(x))
        if op in ("=", "=="):
            sat = abs(lhs - rhs) <= 1e-4; slack = 0.0
        elif op == ">=":
            sat = lhs >= rhs - 1e-4; slack = lhs - rhs
        else:
            sat = lhs <= rhs + 1e-4; slack = rhs - lhs
        constraints.append({"name": f"C{i+1}", "op": op, "lhs": _fin(lhs, 6),
                            "rhs": _fin(rhs, 6), "satisfied": bool(sat), "slack": _fin(slack, 6)})
    out = {"status": "optimal" if res.success else "approximate", "unsolved": False,
           "maximize": maximize, "n_vars": n, "n_constraints": len(gfuncs),
           "objective_value": _fin(obj_val, 6),
           "iterations": int(getattr(res, "nit", 0) or 0),
           "variables": variables, "constraints": constraints}
    if extra:
        out.update(extra)
    return out, x


def make_plot(fobj, var_names, x, con=None):
    if len(var_names) != 2:
        return None
    try:
        fig, ax = plt.subplots(figsize=(7, 5.5), dpi=120)
        r = max(abs(x[0]), abs(x[1]), 1) * 2 + 2
        xs = np.linspace(min(0, x[0]-r), x[0]+r, 120)
        ys = np.linspace(min(0, x[1]-r), x[1]+r, 120)
        GX, GY = np.meshgrid(xs, ys)
        Z = np.vectorize(lambda a, b: fobj([a, b]))(GX, GY)
        cs = ax.contour(GX, GY, Z, levels=18, cmap="viridis", alpha=0.7)
        ax.clabel(cs, inline=True, fontsize=6)
        ax.plot(x[0], x[1], "*", color="#dc2626", markersize=18)
        ax.set_xlabel(var_names[0]); ax.set_ylabel(var_names[1])
        ax.set_title("Objective contours & optimum (★)")
        fig.tight_layout()
        buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        plt.close("all"); return None


def main():
    try:
        payload = json.load(sys.stdin)
        res, fobj, maximize, var_names, ce, co, cr, gf = solve_nlp(payload)
        out, x = build_result(res, fobj, maximize, var_names, ce, co, cr, gf)
        if out.get("unsolved"):
            print(json.dumps({"results": out, "plot": None})); return
        obj_val = out["objective_value"]
        out["interpretation"] = (
            f"The {'maximum' if maximize else 'minimum'} of the objective is {obj_val:.4f}, reached at "
            + ", ".join(f"{var_names[i]}={x[i]:.4f}" for i in range(len(var_names))) + ". "
            "SLSQP follows the objective's gradient while staying feasible; because the problem is nonlinear "
            "this is a local optimum — for a non-convex objective, try different starting points to check it is global."
        )
        plot = make_plot(fobj, var_names, x)
        print(json.dumps({"results": out, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
