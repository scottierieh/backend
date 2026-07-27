#!/usr/bin/env python3
"""Quadratic Programming — optimise 0.5 x'Q x + c'x under linear constraints.
scipy.optimize.minimize (SLSQP); x >= 0.

Input: q_matrix[n][n], c_vector[n], maximize(bool), var_names[],
       con_matrix[m][n], con_ops[], con_rhs[], con_names[]
Output: results{status, unsolved, message, maximize, n_vars, n_constraints,
                objective_value, n_binding, is_convex,
                variables:[{name,value}], constraints:[{name,op,lhs,rhs,slack,binding}],
                interpretation}, plot
"""
import sys, json, io, base64
import numpy as np
from scipy.optimize import minimize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _fin(x, nd=6):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def main():
    try:
        p = json.load(sys.stdin)
        Q = np.array([[float(v) for v in row] for row in (p.get("q_matrix") or [])], float)
        c = np.array([float(v) for v in (p.get("c_vector") or [])], float)
        var_names = [str(v) for v in (p.get("var_names") or [])]
        n = len(var_names)
        if n < 1 or Q.shape != (n, n) or len(c) != n:
            raise ValueError("q_matrix must be n x n and c_vector length n, matching var_names.")
        maximize = bool(p.get("maximize", False))
        A = [[float(v) for v in row] for row in (p.get("con_matrix") or [])]
        ops = [str(o) for o in (p.get("con_ops") or [])]
        rhs = [float(r) for r in (p.get("con_rhs") or [])]
        cnames = p.get("con_names") or [f"C{i+1}" for i in range(len(A))]
        m = len(A)

        Qs = 0.5 * (Q + Q.T)   # symmetrise

        def obj(x):
            val = 0.5 * x @ Qs @ x + c @ x
            return -val if maximize else val

        cons = []
        for i in range(m):
            ai = np.array(A[i], float); r = rhs[i]; op = ops[i] if i < len(ops) else "<="
            if op in ("=", "=="):
                cons.append({"type": "eq", "fun": (lambda x, a=ai, r=r: a @ x - r)})
            elif op == ">=":
                cons.append({"type": "ineq", "fun": (lambda x, a=ai, r=r: a @ x - r)})
            else:
                cons.append({"type": "ineq", "fun": (lambda x, a=ai, r=r: r - a @ x)})

        x0 = np.ones(n)
        res = minimize(obj, x0, method="SLSQP", bounds=[(0, None)]*n,
                       constraints=cons, options={"maxiter": 500, "ftol": 1e-9})
        if not res.success and not np.all(np.isfinite(res.x)):
            print(json.dumps({"results": {"status": "failed", "unsolved": True,
                              "message": f"Solver did not converge: {res.message}"}, "plot": None}))
            return
        x = res.x
        obj_val = float(0.5 * x @ Qs @ x + c @ x)
        eig = np.linalg.eigvalsh(Qs)
        is_convex = bool(np.min(eig) >= -1e-8)   # PSD => convex (for minimisation)

        variables = [{"name": var_names[i], "value": _fin(x[i], 6)} for i in range(n)]
        constraints = []
        n_binding = 0
        for i in range(m):
            lhs = float(np.array(A[i]) @ x); op = ops[i] if i < len(ops) else "<="
            if op in ("=", "=="):
                slack = 0.0; binding = True
            elif op == ">=":
                slack = lhs - rhs[i]; binding = abs(slack) < 1e-5
            else:
                slack = rhs[i] - lhs; binding = abs(slack) < 1e-5
            if binding:
                n_binding += 1
            constraints.append({"name": cnames[i] if i < len(cnames) else f"C{i+1}", "op": op,
                                "lhs": _fin(lhs, 6), "rhs": _fin(rhs[i], 6),
                                "slack": _fin(slack, 6), "binding": bool(binding)})

        plot = None
        if n == 2:
            try:
                fig, ax = plt.subplots(figsize=(7, 5.5), dpi=120)
                r = max(abs(x[0]), abs(x[1]), 1)*2 + 2
                xs = np.linspace(0, x[0]+r, 120); ys = np.linspace(0, x[1]+r, 120)
                GX, GY = np.meshgrid(xs, ys)
                Z = 0.5*(Qs[0,0]*GX**2 + (Qs[0,1]+Qs[1,0])*GX*GY + Qs[1,1]*GY**2) + c[0]*GX + c[1]*GY
                csf = ax.contour(GX, GY, Z, levels=18, cmap="viridis", alpha=0.7)
                ax.clabel(csf, inline=True, fontsize=6)
                ax.plot(x[0], x[1], "*", color="#dc2626", markersize=18)
                ax.set_xlabel(var_names[0]); ax.set_ylabel(var_names[1])
                ax.set_title("Quadratic objective contours & optimum (★)")
                fig.tight_layout()
                buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
                plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            except Exception:
                plt.close("all"); plot = None

        interpretation = (
            f"The {'maximum' if maximize else 'minimum'} of the quadratic objective is {obj_val:.4f} at "
            + ", ".join(f"{var_names[i]}={x[i]:.4f}" for i in range(n)) + ". "
            + ("Q is positive semidefinite, so the objective is convex and this optimum is global. "
               if is_convex else "Q is not positive semidefinite, so the objective is non-convex — this is a local optimum. ")
            + f"{n_binding} of {m} constraint(s) are binding."
        )
        results = {"status": "optimal", "unsolved": False, "maximize": maximize,
                   "n_vars": n, "n_constraints": m, "objective_value": _fin(obj_val, 6),
                   "n_binding": n_binding, "is_convex": is_convex,
                   "variables": variables, "constraints": constraints, "interpretation": interpretation}
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
