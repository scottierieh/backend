#!/usr/bin/env python3
"""Multi-Objective Optimization — weighted-sum scalarisation with a Pareto front.
scipy SLSQP over expression objectives with linear constraints.

Input (from multi-objective-optimization-page.tsx):
    var_names      : string[]
    obj_expr       : string[]     objective expressions
    obj_names      : string[]
    obj_maximize   : bool[]
    obj_weights    : number[]     normalised weights for the chosen solution
    con_matrix     : number[][]   linear constraint coefficients
    con_ops        : string[]     "<=", ">=", "="
    con_rhs        : number[]
    con_names      : string[]
    n_pareto_points: int (default 15)
Output: { results: {chosen_objective_values, variables[], pareto_front[], ...}, plot }
"""
import sys, json, io, base64
import numpy as np
from scipy.optimize import minimize, LinearConstraint

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
        var_names = p.get("var_names") or []
        obj_expr = p.get("obj_expr") or []
        obj_names = p.get("obj_names") or [f"f{i+1}" for i in range(len(obj_expr))]
        obj_max = p.get("obj_maximize") or [False] * len(obj_expr)
        weights = [float(w) for w in (p.get("obj_weights") or [])]
        A = p.get("con_matrix") or []
        ops = [_norm_op(o) for o in (p.get("con_ops") or [])]
        b = [float(x) for x in (p.get("con_rhs") or [])]
        con_names = p.get("con_names") or [f"C{i+1}" for i in range(len(A))]
        n_pareto = int(p.get("n_pareto_points") or 15)
        n = len(var_names)
        K = len(obj_expr)
        if n == 0 or K < 1:
            raise ValueError("Provide variables and at least one objective.")
        if len(weights) != K:
            weights = [1.0 / K] * K

        fobjs = [make_func(e, var_names) for e in obj_expr]
        signs = [-1.0 if obj_max[k] else 1.0 for k in range(K)]  # solver minimises sign*f

        cA = np.array(A, dtype=float) if A else np.zeros((0, n))
        lin_cons = []
        for i in range(len(ops)):
            if ops[i] == "<=":
                lin_cons.append(LinearConstraint(cA[i], -np.inf, b[i]))
            elif ops[i] == ">=":
                lin_cons.append(LinearConstraint(cA[i], b[i], np.inf))
            else:
                lin_cons.append(LinearConstraint(cA[i], b[i], b[i]))
        bounds = [(0, None)] * n

        def solve_weighted(w):
            def scal(x):
                return sum(w[k] * signs[k] * fobjs[k](x) for k in range(K))
            res = minimize(scal, np.ones(n), bounds=bounds, constraints=lin_cons, method="SLSQP",
                           options={"maxiter": 400, "ftol": 1e-9})
            if res.x is None:
                return None
            x = np.array(res.x, dtype=float)
            return x, [float(fobjs[k](x)) for k in range(K)]

        chosen = solve_weighted(weights)
        if chosen is None:
            results = {"status": "unsolved", "unsolved": True, "message": "infeasible",
                       "n_vars": n, "n_objectives": K, "n_constraints": len(ops),
                       "interpretation": "No feasible solution exists for the given constraints."}
            print(json.dumps({"results": results, "plot": None})); return
        cx, cvals = chosen
        variables = [{"name": var_names[i], "value": _fin(cx[i], 6)} for i in range(n)]

        # Pareto front: sweep weights
        pareto = []
        if K == 2:
            for t in np.linspace(0, 1, max(2, n_pareto)):
                sol = solve_weighted([t, 1 - t])
                if sol is None:
                    continue
                sx, sv = sol
                pareto.append({"weights": [_fin(t, 4), _fin(1 - t, 4)],
                               "objective_values": [_fin(v, 6) for v in sv],
                               "variables": {var_names[i]: _fin(sx[i], 6) for i in range(n)}})
        else:
            rng = np.random.default_rng(0)
            for _ in range(max(2, n_pareto)):
                w = rng.random(K); w = w / w.sum()
                sol = solve_weighted(w.tolist())
                if sol is None:
                    continue
                sx, sv = sol
                pareto.append({"weights": [_fin(v, 4) for v in w],
                               "objective_values": [_fin(v, 6) for v in sv],
                               "variables": {var_names[i]: _fin(sx[i], 6) for i in range(n)}})

        plot = None
        try:
            if K == 2 and pareto:
                fig, ax = plt.subplots(figsize=(7.5, 5.6), dpi=120)
                f1 = [pt["objective_values"][0] for pt in pareto]
                f2 = [pt["objective_values"][1] for pt in pareto]
                order = np.argsort(f1)
                ax.plot(np.array(f1)[order], np.array(f2)[order], "-o", color="#2563eb", ms=5, label="Pareto front")
                ax.scatter([cvals[0]], [cvals[1]], color="#dc2626", s=90, zorder=6, marker="*", label="Chosen (weights)")
                ax.set_xlabel(obj_names[0] + (" (max)" if obj_max[0] else " (min)"))
                ax.set_ylabel(obj_names[1] + (" (max)" if obj_max[1] else " (min)"))
                ax.set_title("Pareto front — trade-off between objectives")
                ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.2)
            else:
                fig, ax = plt.subplots(figsize=(8.5, 5), dpi=120)
                ax.bar(obj_names, cvals, color="#2563eb")
                for i, v in enumerate(cvals):
                    ax.text(i, v, f"{v:.3g}", ha="center", va="bottom", fontsize=8)
                ax.set_ylabel("Objective value"); ax.set_title("Chosen solution — objective values")
                ax.tick_params(axis="x", rotation=20)
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        obj_desc = ", ".join(f"{obj_names[k]} = {cvals[k]:.4g} ({'max' if obj_max[k] else 'min'})" for k in range(K))
        interpretation = (
            f"With the chosen weights, the compromise solution achieves {obj_desc}, at "
            + ", ".join(f"{v['name']} = {v['value']:.4g}" for v in variables) + ". "
            f"The Pareto front traces {len(pareto)} non-dominated trade-off points: moving along it improves one "
            "objective only by sacrificing another, so no point on it is strictly better than the chosen one — the "
            "'best' depends entirely on the weights you assign. Points off the front are dominated and never worth choosing."
        )

        results = {
            "status": "ok", "unsolved": False,
            "n_vars": n, "n_objectives": K, "n_constraints": len(ops),
            "chosen_objective_values": [_fin(v, 6) for v in cvals],
            "variables": variables, "pareto_front": pareto,
            "objective_names": obj_names,
            "interpretation": interpretation,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
