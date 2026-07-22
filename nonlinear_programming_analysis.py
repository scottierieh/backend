#!/usr/bin/env python3
"""Nonlinear Programming — minimise/maximise a nonlinear objective under
nonlinear constraints. scipy.optimize.minimize (SLSQP).

Objective and constraints are given as math expressions in the variable names
(supporting + - * / ** ^ and exp, log, sqrt, sin, cos, tan, abs).

Input (from nonlinear-programming-page.tsx):
    objective  : str
    maximize   : bool (default False)
    var_names  : string[]
    start      : number[]
    con_expr   : string[]
    con_ops    : string[]    "<=", ">=", "="
    con_rhs    : number[]
    con_names  : string[]
Output: { results: {variables[], constraints[], objective_value, iterations, ...}, plot }
"""
import sys, json, io, base64
import numpy as np
from scipy.optimize import minimize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ENV = {k: getattr(np, k) for k in
        ("sin", "cos", "tan", "exp", "log", "log10", "sqrt", "abs", "maximum", "minimum", "pi", "e")}


def _fin(x, nd=6):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def _norm_op(op):
    o = str(op).strip()
    if o in ("<=", "≤", "<"):
        return "<="
    if o in (">=", "≥", ">"):
        return ">="
    return "=="


def make_func(expr, var_names):
    code = compile(str(expr).replace("^", "**"), "<expr>", "eval")
    def f(x):
        env = dict(_ENV)
        for i, nm in enumerate(var_names):
            env[nm] = x[i]
        return float(eval(code, {"__builtins__": {}}, env))
    return f


def main():
    try:
        p = json.load(sys.stdin)
        objective = p.get("objective")
        maximize = bool(p.get("maximize", False))
        var_names = p.get("var_names") or []
        start = [float(x) for x in (p.get("start") or [])]
        con_expr = p.get("con_expr") or []
        con_ops = [_norm_op(o) for o in (p.get("con_ops") or [])]
        con_rhs = [float(x) for x in (p.get("con_rhs") or [])]
        con_names = p.get("con_names") or [f"C{i+1}" for i in range(len(con_expr))]
        if not objective or not var_names:
            raise ValueError("Provide an objective expression and variable names.")
        n = len(var_names)
        if not start or len(start) != n:
            start = [1.0] * n

        sign = -1.0 if maximize else 1.0
        fobj = make_func(objective, var_names)
        def obj(x):
            return sign * fobj(x)

        cfuncs = [make_func(e, var_names) for e in con_expr]
        cons = []
        for i, g in enumerate(cfuncs):
            rhs = con_rhs[i]
            if con_ops[i] == "<=":
                cons.append({"type": "ineq", "fun": (lambda x, g=g, rhs=rhs: rhs - g(x))})
            elif con_ops[i] == ">=":
                cons.append({"type": "ineq", "fun": (lambda x, g=g, rhs=rhs: g(x) - rhs)})
            else:
                cons.append({"type": "eq", "fun": (lambda x, g=g, rhs=rhs: g(x) - rhs)})

        res = minimize(obj, np.array(start, dtype=float), constraints=cons, method="SLSQP",
                       options={"maxiter": 500, "ftol": 1e-9})
        if res.x is None:
            results = {"status": "unsolved", "unsolved": True, "message": res.message,
                       "maximize": maximize, "n_vars": n, "n_constraints": len(con_expr),
                       "interpretation": f"The solver failed: {res.message}"}
            print(json.dumps({"results": results, "plot": None})); return

        x = np.array(res.x, dtype=float)
        objective_value = float(fobj(x))
        variables = [{"name": var_names[i], "value": _fin(x[i], 6)} for i in range(n)]
        constraints = []
        for i, g in enumerate(cfuncs):
            val = float(g(x)); rhs = con_rhs[i]
            if con_ops[i] == "<=":
                ok = val <= rhs + 1e-5
            elif con_ops[i] == ">=":
                ok = val >= rhs - 1e-5
            else:
                ok = abs(val - rhs) < 1e-4
            constraints.append({"name": con_names[i], "expr": f"{con_expr[i]} {con_ops[i]} {rhs:g}",
                                "value": _fin(val, 6), "satisfied": bool(ok)})

        plot = None
        try:
            if n == 2:
                fig, ax = plt.subplots(figsize=(7.2, 6), dpi=120)
                span = max(abs(x[0]), abs(x[1]), 1) * 2.0
                gx, gy = np.meshgrid(np.linspace(x[0] - span, x[0] + span, 180),
                                     np.linspace(x[1] - span, x[1] + span, 180))
                Z = np.vectorize(lambda a, b: fobj([a, b]))(gx, gy)
                cs = ax.contour(gx, gy, Z, levels=20, cmap="viridis", alpha=0.7)
                ax.clabel(cs, inline=True, fontsize=6)
                ax.scatter([x[0]], [x[1]], color="#dc2626", s=80, zorder=6, label="Optimum")
                ax.set_xlabel(var_names[0]); ax.set_ylabel(var_names[1])
                ax.set_title("Objective contours & optimum"); ax.legend(fontsize=8, frameon=False)
            else:
                fig, ax = plt.subplots(figsize=(8.5, 5), dpi=120)
                ax.bar([v["name"] for v in variables], [v["value"] or 0 for v in variables], color="#2563eb")
                ax.set_ylabel("Optimal value"); ax.set_title("Optimal variables"); ax.tick_params(axis="x", rotation=25)
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        n_ok = sum(1 for cc in constraints if cc["satisfied"])
        interpretation = (
            f"The solver converged in {int(res.nit)} iterations to an objective value of {objective_value:,.6g} at "
            + ", ".join(f"{v['name']} = {v['value']:.4g}" for v in variables) + ". "
            f"{n_ok} of {len(constraints)} constraints are satisfied. Nonlinear programs can have multiple local optima, "
            "so this is the best point found from your starting guess — trying different start values is a good way to "
            "check whether it is the global optimum."
        )

        results = {
            "status": "ok", "unsolved": False, "maximize": maximize,
            "n_vars": n, "n_constraints": len(con_expr), "objective_value": _fin(objective_value, 6),
            "iterations": int(res.nit),
            "variables": variables, "constraints": constraints,
            "interpretation": interpretation,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
