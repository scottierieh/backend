#!/usr/bin/env python3
"""Multi-Objective Optimization — trade off several linear objectives under
linear constraints via the weighted-sum method, and trace the Pareto front.

Input: var_names[], obj_expr[] (linear), obj_names[], obj_maximize[bool],
       obj_weights[] (sum to 1), con_matrix[m][n], con_ops[], con_rhs[],
       con_names[], n_pareto_points(int)
Output: results{status, unsolved, message, n_vars, n_objectives, n_constraints,
                chosen_objective_values:[{name,value}], variables:[{name,value}],
                pareto_front:[{...obj values}], interpretation}, plot
"""
import sys, json, io, base64
import numpy as np
from scipy.optimize import linprog
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


def lin_coeffs(expr, var_names):
    f = make_func(expr, var_names)
    n = len(var_names)
    f0 = f([0.0]*n)
    coef = []
    for i in range(n):
        e = [0.0]*n; e[i] = 1.0
        coef.append(f(e) - f0)
    return np.array(coef, float), float(f0)


def build_lp(A, ops, rhs, n):
    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    for i in range(len(A)):
        ai = np.array(A[i], float); r = rhs[i]; op = ops[i] if i < len(ops) else "<="
        if op in ("=", "=="):
            A_eq.append(ai); b_eq.append(r)
        elif op == ">=":
            A_ub.append(-ai); b_ub.append(-r)
        else:
            A_ub.append(ai); b_ub.append(r)
    return (np.array(A_ub) if A_ub else None, np.array(b_ub) if b_ub else None,
            np.array(A_eq) if A_eq else None, np.array(b_eq) if b_eq else None)


def main():
    try:
        p = json.load(sys.stdin)
        var_names = [str(v) for v in (p.get("var_names") or [])]
        n = len(var_names)
        exprs = [str(e) for e in (p.get("obj_expr") or [])]
        onames = p.get("obj_names") or [f"Objective {i+1}" for i in range(len(exprs))]
        omax = [bool(b) for b in (p.get("obj_maximize") or [True]*len(exprs))]
        weights = [float(w) for w in (p.get("obj_weights") or [])]
        K = len(exprs)
        if n < 1 or K < 2:
            raise ValueError("Need at least one variable and two objectives.")
        if len(weights) != K:
            weights = [1.0/K]*K
        A = [[float(v) for v in row] for row in (p.get("con_matrix") or [])]
        ops = [str(o) for o in (p.get("con_ops") or [])]
        rhs = [float(r) for r in (p.get("con_rhs") or [])]
        npar = int(p.get("n_pareto_points") or 12)

        coefs = []; consts = []
        for e in exprs:
            cf, c0 = lin_coeffs(e, var_names); coefs.append(cf); consts.append(c0)
        A_ub, b_ub, A_eq, b_eq = build_lp(A, ops, rhs, n)
        bounds = [(0, None)]*n

        def solve_weighted(ws):
            # maximise sum_k ws_k * sign_k * (coef_k . x); linprog minimises
            c = np.zeros(n)
            for k in range(K):
                sign = 1.0 if omax[k] else -1.0
                c += ws[k] * sign * coefs[k]
            res = linprog(-c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
            return res

        res = solve_weighted(weights)
        if not res.success:
            print(json.dumps({"results": {"status": "infeasible", "unsolved": True,
                              "message": f"No feasible solution: {res.message}"}, "plot": None}))
            return
        x = res.x
        def obj_vals(xx):
            return [float(coefs[k] @ xx + consts[k]) for k in range(K)]
        chosen = obj_vals(x)
        variables = [{"name": var_names[i], "value": _fin(x[i], 6)} for i in range(n)]
        chosen_objective_values = [{"name": onames[k], "value": _fin(chosen[k], 6)} for k in range(K)]

        # Pareto front: vary weights (2-obj: sweep; >2: random samples)
        pareto = []
        seen = set()
        if K == 2:
            for t in np.linspace(0, 1, max(npar, 2)):
                r = solve_weighted([t, 1-t])
                if r.success:
                    v = obj_vals(r.x); key = tuple(round(z, 4) for z in v)
                    if key not in seen:
                        seen.add(key); pareto.append({onames[0]: _fin(v[0], 4), onames[1]: _fin(v[1], 4)})
        else:
            rng = np.random.default_rng(0)
            for _ in range(max(npar, 4)):
                w = rng.random(K); w /= w.sum()
                r = solve_weighted(w)
                if r.success:
                    v = obj_vals(r.x); key = tuple(round(z, 4) for z in v)
                    if key not in seen:
                        seen.add(key); pareto.append({onames[k]: _fin(v[k], 4) for k in range(K)})

        plot = None
        if K == 2 and pareto:
            try:
                fig, ax = plt.subplots(figsize=(7, 5.5), dpi=120)
                pv = sorted([(pt[onames[0]], pt[onames[1]]) for pt in pareto])
                xs = [a for a, _ in pv]; ys = [b for _, b in pv]
                ax.plot(xs, ys, "-o", color="#2563eb", label="Pareto front")
                ax.plot(chosen[0], chosen[1], "*", color="#dc2626", markersize=18, label="chosen (weights)")
                ax.set_xlabel(onames[0]); ax.set_ylabel(onames[1])
                ax.set_title("Pareto front — no point can improve one objective without hurting the other")
                ax.legend(fontsize=8, frameon=False); fig.tight_layout()
                buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
                plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            except Exception:
                plt.close("all"); plot = None

        interpretation = (
            f"With the given weights, the best compromise sets "
            + ", ".join(f"{var_names[i]}={x[i]:.3f}" for i in range(n)) + ", giving "
            + ", ".join(f"{onames[k]}={chosen[k]:.3f}" for k in range(K)) + ". "
            f"The Pareto front ({len(pareto)} point(s)) shows the full set of non-dominated trade-offs: "
            "moving along it, improving one objective necessarily worsens another. The chosen point is where "
            "your weights land on that front — change the weights to slide toward whichever objective matters more."
        )
        results = {"status": "optimal", "unsolved": False, "n_vars": n, "n_objectives": K,
                   "n_constraints": len(A), "chosen_objective_values": chosen_objective_values,
                   "variables": variables, "pareto_front": pareto, "interpretation": interpretation}
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
