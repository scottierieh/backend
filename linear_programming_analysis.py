#!/usr/bin/env python3
"""Linear / Integer Programming — solve an LP or MILP with full sensitivity.
scipy.optimize (linprog HiGHS for LP duals, milp for integer/binary).

Handles the shared contract used by the linear, integer, mixed-integer,
binary-integer and goal-programming pages.

Input (from *-programming-page.tsx):
    objective   : number[]        objective coefficients
    maximize    : bool            (default True)
    var_names   : string[]
    con_matrix  : number[][]      constraint coefficients (one row per constraint)
    con_ops     : string[]        each "<=", ">=", "=" / "=="
    con_rhs     : number[]
    con_names   : string[]
    var_types   : string[]        optional; "continuous"|"integer"|"binary"
Output: { results: {variables[], constraints[], objective_value, ...}, plot }
"""
import sys, json, io, base64
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import linprog, milp, LinearConstraint, Bounds


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
        c = p.get("objective")
        var_names = p.get("var_names") or []
        A = p.get("con_matrix")
        ops = p.get("con_ops")
        b = p.get("con_rhs")
        con_names = p.get("con_names") or []
        maximize = bool(p.get("maximize", True))
        var_types = p.get("var_types")

        if c is None or A is None or b is None or ops is None:
            raise ValueError("Missing required parameters: objective, con_matrix, con_ops or con_rhs.")
        c = [float(x) for x in c]
        A = [[float(x) for x in row] for row in A]
        b = [float(x) for x in b]
        ops = [_norm_op(o) for o in ops]
        n = len(c)
        m = len(A)
        if not var_names:
            var_names = [f"x{i+1}" for i in range(n)]
        if not con_names:
            con_names = [f"C{i+1}" for i in range(m)]
        if any(len(row) != n for row in A):
            raise ValueError("Each constraint row must have one coefficient per variable.")
        if len(b) != m or len(ops) != m:
            raise ValueError("Constraint matrix, operators and RHS must have the same length.")

        cA = np.array(A, dtype=float)
        cb = np.array(b, dtype=float)
        cc = np.array(c, dtype=float)

        is_integer = bool(var_types) and any(t in ("integer", "binary") for t in var_types)

        sign = -1.0 if maximize else 1.0   # solver always minimises
        variables, constraints = [], []
        objective_value = None
        unsolved = False
        message = ""

        if is_integer:
            # ---- MILP via scipy.optimize.milp ----
            integrality = np.array([1 if (var_types[i] in ("integer", "binary")) else 0 for i in range(n)])
            lb = np.array([0.0] * n)
            ub = np.array([1.0 if var_types[i] == "binary" else np.inf for i in range(n)])
            cons_list = []
            for i in range(m):
                row = cA[i]
                if ops[i] == "<=":
                    cons_list.append(LinearConstraint(row, -np.inf, cb[i]))
                elif ops[i] == ">=":
                    cons_list.append(LinearConstraint(row, cb[i], np.inf))
                else:
                    cons_list.append(LinearConstraint(row, cb[i], cb[i]))
            res = milp(c=sign * cc, constraints=cons_list, integrality=integrality,
                       bounds=Bounds(lb, ub))
            if not res.success or res.x is None:
                unsolved = True
                message = res.message or "The integer program has no feasible/optimal solution."
            else:
                x = np.array(res.x, dtype=float)
                objective_value = float(cc @ x)
                for i in range(n):
                    variables.append({"name": var_names[i], "value": _fin(x[i], 6), "coef": _fin(c[i], 6),
                                      "reduced_cost": None, "obj_from": None, "obj_till": None,
                                      "basic": bool(abs(x[i]) > 1e-7)})
                for i in range(m):
                    lhs = float(cA[i] @ x)
                    slack = cb[i] - lhs if ops[i] == "<=" else (lhs - cb[i] if ops[i] == ">=" else lhs - cb[i])
                    binding = bool(abs(lhs - cb[i]) < 1e-6)
                    constraints.append({"name": con_names[i], "op": ops[i], "lhs": _fin(lhs, 6),
                                        "rhs": _fin(cb[i], 6), "slack": _fin(slack, 6),
                                        "binding": binding, "shadow_price": None})
        else:
            # ---- continuous LP via linprog HiGHS (gives duals) ----
            A_ub, b_ub, A_eq, b_eq = [], [], [], []
            # map original constraint index -> ('ub'|'eq', position, sign_for_shadow)
            cmap = []
            for i in range(m):
                if ops[i] == "<=":
                    cmap.append(("ub", len(A_ub), 1.0)); A_ub.append(cA[i]); b_ub.append(cb[i])
                elif ops[i] == ">=":
                    cmap.append(("ub", len(A_ub), -1.0)); A_ub.append(-cA[i]); b_ub.append(-cb[i])
                else:
                    cmap.append(("eq", len(A_eq), 1.0)); A_eq.append(cA[i]); b_eq.append(cb[i])
            res = linprog(sign * cc,
                          A_ub=np.array(A_ub) if A_ub else None, b_ub=np.array(b_ub) if b_ub else None,
                          A_eq=np.array(A_eq) if A_eq else None, b_eq=np.array(b_eq) if b_eq else None,
                          bounds=[(0, None)] * n, method="highs")
            if not res.success or res.x is None:
                unsolved = True
                message = res.message or "The linear program is infeasible or unbounded."
            else:
                x = np.array(res.x, dtype=float)
                objective_value = float(cc @ x)
                # reduced costs from variable lower-bound marginals (solver sense -> original)
                try:
                    rc = np.array(res.lower.marginals, dtype=float)
                except Exception:
                    rc = np.full(n, np.nan)
                ineq_marg = np.array(getattr(res, "ineqlin").marginals, dtype=float) if A_ub else np.array([])
                eq_marg = np.array(getattr(res, "eqlin").marginals, dtype=float) if A_eq else np.array([])
                for i in range(n):
                    rci = rc[i] if i < len(rc) else np.nan
                    # solver minimises sign*c; convert reduced cost back to original objective sense
                    rc_orig = (-rci if maximize else rci) if np.isfinite(rci) else None
                    variables.append({"name": var_names[i], "value": _fin(x[i], 6), "coef": _fin(c[i], 6),
                                      "reduced_cost": _fin(rc_orig, 6),
                                      "obj_from": None, "obj_till": None,
                                      "basic": bool(abs(x[i]) > 1e-7)})
                for i in range(m):
                    kind, pos, sgn = cmap[i]
                    marg = (ineq_marg[pos] if kind == "ub" else eq_marg[pos]) if (
                        (kind == "ub" and pos < len(ineq_marg)) or (kind == "eq" and pos < len(eq_marg))) else np.nan
                    # d(objective)/d(rhs): solver marginal is d(min)/d(b_solver); undo sign flips
                    shadow = None
                    if np.isfinite(marg):
                        s = marg * sgn                    # undo >= negation of the RHS
                        shadow = (-s if maximize else s)  # undo objective negation for maximize
                    lhs = float(cA[i] @ x)
                    slack = cb[i] - lhs if ops[i] == "<=" else (lhs - cb[i] if ops[i] == ">=" else lhs - cb[i])
                    binding = bool(abs(lhs - cb[i]) < 1e-6)
                    constraints.append({"name": con_names[i], "op": ops[i], "lhs": _fin(lhs, 6),
                                        "rhs": _fin(cb[i], 6), "slack": _fin(slack, 6),
                                        "binding": binding,
                                        "shadow_price": _fin(shadow, 6)})

        n_binding = sum(1 for con in constraints if con["binding"])

        # ---- plot: 2-var feasible region, else variable-value bars ----
        plot = None
        try:
            if n == 2 and not unsolved:
                fig, ax = plt.subplots(figsize=(7.5, 6), dpi=120)
                xmax = max([cb[i] / cA[i][0] for i in range(m) if abs(cA[i][0]) > 1e-9] + [x[0] * 1.5 if not unsolved else 10, 10])
                ymax = max([cb[i] / cA[i][1] for i in range(m) if abs(cA[i][1]) > 1e-9] + [x[1] * 1.5 if not unsolved else 10, 10])
                xmax = float(min(xmax * 1.15, 1e4)); ymax = float(min(ymax * 1.15, 1e4))
                xs = np.linspace(0, xmax, 400)
                for i in range(m):
                    a1, a2 = cA[i]
                    if abs(a2) > 1e-9:
                        ax.plot(xs, (cb[i] - a1 * xs) / a2, lw=1.2, alpha=0.8, label=con_names[i])
                    elif abs(a1) > 1e-9:
                        ax.axvline(cb[i] / a1, lw=1.2, alpha=0.8, label=con_names[i])
                # shade feasible region on a grid
                gx, gy = np.meshgrid(np.linspace(0, xmax, 220), np.linspace(0, ymax, 220))
                feas = np.ones_like(gx, dtype=bool)
                for i in range(m):
                    lhs = cA[i][0] * gx + cA[i][1] * gy
                    if ops[i] == "<=":
                        feas &= lhs <= cb[i] + 1e-9
                    elif ops[i] == ">=":
                        feas &= lhs >= cb[i] - 1e-9
                    else:
                        feas &= np.abs(lhs - cb[i]) < (0.01 * (abs(cb[i]) + 1))
                ax.imshow(feas, extent=[0, xmax, 0, ymax], origin="lower", cmap="Blues", alpha=0.25, aspect="auto")
                ax.scatter([x[0]], [x[1]], color="#dc2626", s=80, zorder=6, label=f"Optimum ({x[0]:.2f}, {x[1]:.2f})")
                ax.set_xlim(0, xmax); ax.set_ylim(0, ymax)
                ax.set_xlabel(var_names[0]); ax.set_ylabel(var_names[1])
                ax.set_title("Feasible region & optimum"); ax.legend(fontsize=7, frameon=False); ax.grid(alpha=0.2)
            else:
                fig, ax = plt.subplots(figsize=(8.5, 5), dpi=120)
                if not unsolved:
                    vals = [v["value"] or 0 for v in variables]
                    ax.bar([v["name"] for v in variables], vals, color="#2563eb")
                    for i, v in enumerate(vals):
                        ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
                    ax.set_ylabel("Optimal value"); ax.set_title("Optimal decision variables")
                    ax.tick_params(axis="x", rotation=25)
                else:
                    ax.text(0.5, 0.5, "No feasible/optimal solution", ha="center", va="center")
                    ax.axis("off")
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        if unsolved:
            results = {"status": "unsolved", "unsolved": True, "message": message,
                       "maximize": maximize, "n_vars": n, "n_constraints": m,
                       "interpretation": f"The solver could not find an optimal solution: {message}"}
        else:
            model_kind = "integer program" if is_integer else "linear program"
            sig_cons = [con for con in constraints if con["binding"]]
            shadow_note = ""
            if not is_integer and sig_cons:
                top = max(sig_cons, key=lambda cc_: abs(cc_.get("shadow_price") or 0))
                if top.get("shadow_price"):
                    shadow_note = (f" The most valuable binding constraint is '{top['name']}', with a shadow price of "
                                   f"{top['shadow_price']:.4g} — the objective would change by that much per unit of extra "
                                   "right-hand side.")
            interpretation = (
                f"The {model_kind} was solved to optimality with an objective value of {objective_value:,.4f}. "
                f"{n_binding} of {m} constraints are binding (fully used at the optimum); the rest have slack. "
                + (f"The optimal plan is: " + ", ".join(f"{v['name']} = {v['value']:.4g}" for v in variables) + "."
                   if n <= 8 else "")
                + shadow_note
                + ("" if not is_integer else " Integer/binary variables were required to take whole (or 0/1) values, "
                   "which is why no shadow prices are reported — duality does not apply to integer programs.")
            )
            results = {
                "status": "ok", "unsolved": False, "maximize": maximize,
                "n_vars": n, "n_constraints": m, "objective_value": _fin(objective_value, 6),
                "n_binding": n_binding, "is_integer": is_integer,
                "variables": variables, "constraints": constraints,
                "interpretation": interpretation,
            }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
