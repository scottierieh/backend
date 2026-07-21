#!/usr/bin/env python3
"""Linear Programming (and Mixed-Integer LP) solver — PuLP + CBC.

CLI contract (same as every other backend script): read ONE JSON object from
stdin, print ONE JSON object to stdout on success, or print {"error": ...} to
stderr and exit(1) on failure.

Solves
    max / min   c . x
    subject to  A x  (<= | >= | =)  b ,   x >= 0
and returns the optimum plus the economics: which constraints are BINDING,
their SHADOW PRICES (dual values), the REDUCED COST of each variable, and a
feasible-region / solution plot.

Input JSON (from src/components/pages/statistica/linear-programming-page.tsx):
    objective    : number[n]            objective coefficients
    maximize     : bool                 True = maximise, False = minimise
    var_names    : string[n]  (optional)
    con_matrix   : number[m][n]         constraint coefficients
    con_ops      : string[m]            each "<=", ">=", or "="
    con_rhs      : number[m]
    con_names    : string[m]  (optional)
    problem_type : "lp" | "integer"     (optional; default "lp")
    var_types    : string[n]            (optional; "integer"/"continuous")

Output JSON: { "results": {...}, "plot": "data:image/png;base64,..." | None }
"""
import sys
import json
import io
import base64

try:
    import pulp
    PULP_OK = True
except Exception:
    PULP_OK = False

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _fin(x, ndigits=6):
    """Round to a JSON-safe float, or None if not finite."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(v):
        return None
    return round(v, ndigits)


def _make_plot(n, var_names, A, ops, rhs, obj, xstar, zstar, con_names):
    """2-var: feasible region + constraint lines + objective + optimum.
    Otherwise: horizontal bar chart of the optimal variable values."""
    try:
        fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=120)
        if n == 2:
            xmax = max(xstar[0] * 1.4, 1.0)
            ymax = max(xstar[1] * 1.4, 1.0)
            for i in range(len(A)):
                if A[i][0] > 1e-9:
                    xmax = max(xmax, (rhs[i] / A[i][0]) * 1.1)
                if A[i][1] > 1e-9:
                    ymax = max(ymax, (rhs[i] / A[i][1]) * 1.1)
            xmax = min(xmax, 1e6)
            ymax = min(ymax, 1e6)
            gx = np.linspace(0, xmax, 220)
            gy = np.linspace(0, ymax, 220)
            GX, GY = np.meshgrid(gx, gy)
            feas = np.ones_like(GX, dtype=bool)
            for i in range(len(A)):
                val = A[i][0] * GX + A[i][1] * GY
                if ops[i] == "<=":
                    feas &= val <= rhs[i] + 1e-6
                elif ops[i] == ">=":
                    feas &= val >= rhs[i] - 1e-6
                else:
                    feas &= np.abs(val - rhs[i]) <= (abs(rhs[i]) * 1e-3 + 1e-6)
            ax.contourf(GX, GY, feas.astype(int), levels=[0.5, 1.5],
                        colors=["#dbeafe"], alpha=0.8)
            colors = plt.cm.tab10(np.linspace(0, 1, max(len(A), 3)))
            xs = np.linspace(0, xmax, 100)
            for i in range(len(A)):
                if abs(A[i][1]) > 1e-9:
                    ys = (rhs[i] - A[i][0] * xs) / A[i][1]
                    ax.plot(xs, ys, color=colors[i], lw=2, label=con_names[i])
                elif abs(A[i][0]) > 1e-9:
                    ax.axvline(rhs[i] / A[i][0], color=colors[i], lw=2, label=con_names[i])
            if abs(obj[1]) > 1e-9:
                ys = (zstar - obj[0] * xs) / obj[1]
                ax.plot(xs, ys, color="#111827", lw=2, ls="--", label="Objective")
            ax.plot(xstar[0], xstar[1], "o", color="#dc2626", markersize=11)
            ax.annotate(f"  ({xstar[0]:.2f}, {xstar[1]:.2f})", (xstar[0], xstar[1]),
                        color="#dc2626", fontsize=10)
            ax.set_xlim(0, xmax)
            ax.set_ylim(0, ymax)
            ax.set_xlabel(var_names[0])
            ax.set_ylabel(var_names[1])
            ax.set_title("Feasible region & optimum")
            ax.legend(loc="upper right", fontsize=8, frameon=False)
        else:
            order = np.argsort(xstar)
            ax.barh([var_names[i] for i in order], [xstar[i] for i in order],
                    color="#2563eb")
            ax.set_xlabel("Optimal value")
            ax.set_title("Optimal decision variables")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        plt.close(fig)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        plt.close("all")
        return None


def main():
    if not PULP_OK:
        print(json.dumps({"error": "The 'pulp' package is not installed on the server."}),
              file=sys.stderr)
        sys.exit(1)
    try:
        p = json.load(sys.stdin)
        obj = [float(v) for v in (p.get("objective") or [])]
        n = len(obj)
        if n < 1:
            raise ValueError("Provide a numeric objective with at least one coefficient.")
        maximize = bool(p.get("maximize", True))
        var_names = p.get("var_names") or [f"x{i+1}" for i in range(n)]
        if len(var_names) != n:
            var_names = [f"x{i+1}" for i in range(n)]
        A = [[float(v) for v in row] for row in (p.get("con_matrix") or [])]
        m = len(A)
        if m < 1:
            raise ValueError("Provide at least one constraint.")
        if any(len(row) != n for row in A):
            raise ValueError(f"Each constraint needs exactly {n} coefficients.")
        raw_ops = p.get("con_ops") or []
        opmap = {"<=": "<=", "<": "<=", ">=": ">=", ">": ">=", "=": "=", "==": "="}
        ops = [opmap.get(o) for o in raw_ops]
        rhs = [float(v) for v in (p.get("con_rhs") or [])]
        if len(ops) != m or len(rhs) != m or any(o is None for o in ops):
            raise ValueError("Constraint operators / right-hand sides do not match the constraints.")
        con_names = p.get("con_names") or [f"C{i+1}" for i in range(m)]
        if len(con_names) != m:
            con_names = [f"C{i+1}" for i in range(m)]
        problem_type = (p.get("problem_type") or "lp").lower()
        is_integer = problem_type in ("integer", "milp")
        var_types = p.get("var_types") or (["integer"] * n if is_integer else ["continuous"] * n)

        # ---- build model ----
        prob = pulp.LpProblem("LP", pulp.LpMaximize if maximize else pulp.LpMinimize)
        xvars = []
        for j in range(n):
            cat = "Integer" if (is_integer and str(var_types[j]).lower().startswith("int")) else "Continuous"
            xvars.append(pulp.LpVariable(var_names[j], lowBound=0, cat=cat))
        prob += pulp.lpSum(obj[j] * xvars[j] for j in range(n)), "objective"
        constraints = []
        for i in range(m):
            lhs = pulp.lpSum(A[i][j] * xvars[j] for j in range(n))
            if ops[i] == "<=":
                named = pulp.LpConstraint(lhs - rhs[i], sense=pulp.LpConstraintLE, name=f"c{i}", rhs=0)
            elif ops[i] == ">=":
                named = pulp.LpConstraint(lhs - rhs[i], sense=pulp.LpConstraintGE, name=f"c{i}", rhs=0)
            else:
                named = pulp.LpConstraint(lhs - rhs[i], sense=pulp.LpConstraintEQ, name=f"c{i}", rhs=0)
            prob += named
            constraints.append(named)

        status_code = prob.solve(pulp.PULP_CBC_CMD(msg=0))
        status = pulp.LpStatus[status_code]

        if status in ("Infeasible", "Unbounded", "Undefined"):
            msg = {
                "Infeasible": "The problem is INFEASIBLE — no point satisfies all constraints at once.",
                "Unbounded": "The problem is UNBOUNDED — the objective can improve without limit; add a binding constraint.",
                "Undefined": "The solver could not find a solution (status: undefined).",
            }[status]
            print(json.dumps({"results": {"status": status.lower(), "unsolved": True, "message": msg},
                              "plot": None}))
            return

        xstar = [float(v.value() or 0.0) for v in xvars]
        zstar = float(pulp.value(prob.objective))

        duals_ok = not is_integer  # duals/reduced costs undefined for MILP
        tol = 1e-7
        variables = []
        for j in range(n):
            dj = _fin(xvars[j].dj) if duals_ok else None
            variables.append({
                "name": var_names[j], "value": _fin(xstar[j]), "coef": _fin(obj[j]),
                "reduced_cost": dj,
                "obj_from": None, "obj_till": None,  # CBC gives no objective ranging
                "basic": abs(xstar[j]) > tol,
            })

        cons_out = []
        n_binding = 0
        for i in range(m):
            lhs_val = sum(A[i][j] * xstar[j] for j in range(n))
            slack = (rhs[i] - lhs_val) if ops[i] == "<=" else (lhs_val - rhs[i]) if ops[i] == ">=" else 0.0
            pi = _fin(constraints[i].pi) if duals_ok else None
            binding = abs(lhs_val - rhs[i]) <= 1e-6 or (pi is not None and abs(pi) > tol)
            if binding:
                n_binding += 1
            cons_out.append({
                "name": con_names[i], "op": ops[i], "lhs": _fin(lhs_val), "rhs": _fin(rhs[i]),
                "slack": _fin(slack), "binding": bool(binding), "shadow_price": pi,
            })

        plot = _make_plot(n, var_names, A, ops, rhs, obj, xstar, zstar, con_names)

        active = [var_names[j] for j in range(n) if abs(xstar[j]) > tol]
        zero = [v for v in var_names if v not in active]
        interpretation = (
            f"The {'MILP' if is_integer else 'LP'} is optimal. "
            f"{'Maximising' if maximize else 'Minimising'} the objective gives "
            f"{'the maximum' if maximize else 'the minimum'} = {zstar:.4f} at "
            + ", ".join(f"{var_names[j]}={xstar[j]:.3f}" for j in range(n)) + ". "
            f"{n_binding} of {m} constraints are binding"
            + ("." if is_integer else
               " — their shadow prices say how much the objective would change per one-unit "
               "increase in the right-hand side. Variables at zero ("
               + (", ".join(zero) if zero else "none")
               + ") have a reduced cost equal to how much their objective coefficient must "
               "improve before they enter the solution.")
        )

        results = {
            "status": "optimal", "unsolved": False, "maximize": maximize,
            "problem_type": "integer" if is_integer else "lp",
            "n_vars": n, "n_constraints": m,
            "objective_value": _fin(zstar), "n_binding": n_binding,
            "variables": variables, "constraints": cons_out,
            "interpretation": interpretation,
        }
        print(json.dumps({"results": results, "plot": plot}))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
