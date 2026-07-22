#!/usr/bin/env python3
"""Quadratic Programming — minimise/maximise 0.5 xᵀQx + cᵀx under linear constraints.
scipy.optimize.minimize (SLSQP).

Input (from quadratic-programming-page.tsx):
    q_matrix   : number[][]   symmetric Q
    c_vector   : number[]     linear term c
    maximize   : bool (default False)
    var_names  : string[]
    con_matrix : number[][]
    con_ops    : string[]    "<=", ">=", "="
    con_rhs    : number[]
    con_names  : string[]
Output: { results: {variables[], constraints[], objective_value, is_convex, ...}, plot }
"""
import sys, json, io, base64
import numpy as np
from scipy.optimize import minimize, LinearConstraint

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def main():
    try:
        p = json.load(sys.stdin)
        Q = np.array(p.get("q_matrix"), dtype=float)
        c = np.array(p.get("c_vector"), dtype=float)
        maximize = bool(p.get("maximize", False))
        var_names = p.get("var_names") or [f"x{i+1}" for i in range(len(c))]
        A = p.get("con_matrix") or []
        ops = [_norm_op(o) for o in (p.get("con_ops") or [])]
        b = [float(x) for x in (p.get("con_rhs") or [])]
        con_names = p.get("con_names") or [f"C{i+1}" for i in range(len(A))]
        n = len(c)
        if Q.shape != (n, n):
            raise ValueError("Q matrix must be square with size equal to the number of variables.")

        Qs = 0.5 * (Q + Q.T)  # symmetrise
        eig = np.linalg.eigvalsh(Qs)
        is_convex = bool(np.all(eig >= -1e-8))   # PSD => convex (for minimisation)
        sign = -1.0 if maximize else 1.0

        def obj(x):
            return sign * (0.5 * x @ Qs @ x + c @ x)
        def grad(x):
            return sign * (Qs @ x + c)

        cons = []
        cA = np.array(A, dtype=float) if A else np.zeros((0, n))
        for i in range(len(ops)):
            if ops[i] == "<=":
                cons.append(LinearConstraint(cA[i], -np.inf, b[i]))
            elif ops[i] == ">=":
                cons.append(LinearConstraint(cA[i], b[i], np.inf))
            else:
                cons.append(LinearConstraint(cA[i], b[i], b[i]))
        bounds = [(0, None)] * n
        x0 = np.ones(n)
        res = minimize(obj, x0, jac=grad, bounds=bounds, constraints=cons, method="SLSQP",
                       options={"maxiter": 500, "ftol": 1e-9})

        if not res.success and res.x is None:
            results = {"status": "unsolved", "unsolved": True, "message": res.message,
                       "maximize": maximize, "n_vars": n, "n_constraints": len(ops),
                       "interpretation": f"The QP solver failed: {res.message}"}
            print(json.dumps({"results": results, "plot": None})); return

        x = np.array(res.x, dtype=float)
        objective_value = float(0.5 * x @ Qs @ x + c @ x)
        variables = [{"name": var_names[i], "value": _fin(x[i], 6)} for i in range(n)]
        constraints = []
        for i in range(len(ops)):
            lhs = float(cA[i] @ x)
            slack = b[i] - lhs if ops[i] == "<=" else (lhs - b[i] if ops[i] == ">=" else lhs - b[i])
            constraints.append({"name": con_names[i], "lhs": _fin(lhs, 6), "op": ops[i], "rhs": _fin(b[i], 6),
                                "slack": _fin(slack, 6), "binding": bool(abs(lhs - b[i]) < 1e-5)})
        n_binding = sum(1 for cc in constraints if cc["binding"])

        plot = None
        try:
            if n == 2:
                fig, ax = plt.subplots(figsize=(7.2, 6), dpi=120)
                span = max(abs(x[0]), abs(x[1]), 1) * 2.2
                gx, gy = np.meshgrid(np.linspace(0, span, 200), np.linspace(0, span, 200))
                Z = 0.5 * (Qs[0, 0] * gx**2 + 2 * Qs[0, 1] * gx * gy + Qs[1, 1] * gy**2) + c[0] * gx + c[1] * gy
                cs = ax.contour(gx, gy, Z, levels=18, cmap="viridis", alpha=0.7)
                ax.clabel(cs, inline=True, fontsize=6)
                for i in range(len(ops)):
                    a1, a2 = cA[i]
                    xs = np.linspace(0, span, 100)
                    if abs(a2) > 1e-9:
                        ax.plot(xs, (b[i] - a1 * xs) / a2, lw=1.3, ls="--", alpha=0.8, label=con_names[i])
                    elif abs(a1) > 1e-9:
                        ax.axvline(b[i] / a1, lw=1.3, ls="--", alpha=0.8, label=con_names[i])
                ax.scatter([x[0]], [x[1]], color="#dc2626", s=80, zorder=6, label="Optimum")
                ax.set_xlim(0, span); ax.set_ylim(0, span)
                ax.set_xlabel(var_names[0]); ax.set_ylabel(var_names[1])
                ax.set_title("Objective contours & optimum"); ax.legend(fontsize=7, frameon=False)
            else:
                fig, ax = plt.subplots(figsize=(8.5, 5), dpi=120)
                ax.bar([v["name"] for v in variables], [v["value"] or 0 for v in variables], color="#2563eb")
                ax.set_ylabel("Optimal value"); ax.set_title("Optimal decision variables"); ax.tick_params(axis="x", rotation=25)
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        interpretation = (
            f"The quadratic program reached an objective value of {objective_value:,.6g} at "
            + ", ".join(f"{v['name']} = {v['value']:.4g}" for v in variables) + ". "
            + (f"The Q matrix is positive semidefinite (all eigenvalues ≥ 0), so the problem is convex and this is the "
               "unique global optimum." if is_convex else
               "The Q matrix is not positive semidefinite, so the problem is non-convex — this is a local optimum from the "
               "chosen start, and other starts could give different results.")
            + f" {n_binding} of {len(ops)} constraints are binding at the solution."
        )

        results = {
            "status": "ok", "unsolved": False, "maximize": maximize,
            "n_vars": n, "n_constraints": len(ops), "objective_value": _fin(objective_value, 6),
            "n_binding": n_binding, "is_convex": is_convex,
            "variables": variables, "constraints": constraints,
            "interpretation": interpretation,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
