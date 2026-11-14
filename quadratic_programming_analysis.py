#!/usr/bin/env python3
"""Quadratic Programming — optimise 0.5 x'Q x + c'x under linear constraints.
scipy.optimize.minimize (SLSQP); x >= 0.

Input: q_matrix[n][n], c_vector[n], maximize(bool), var_names[],
       con_matrix[m][n], con_ops[], con_rhs[], con_names[]
Output: results{status (optimal|infeasible|unbounded|not_converged|failed), unsolved,
                success, solver_status_code, message, solver_message, nit,
                maximize, n_vars, n_constraints, objective_value, n_binding, n_violated,
                is_convex (Q positive semidefinite — kept for old callers),
                q_psd, q_nsd, is_convex_for_sense (min needs Q PSD, max needs Q NSD),
                q_min_eigenvalue, q_max_eigenvalue, convexity_note,
                feasibility_check, variables:[{name,value}],
                constraints:[{name,op,lhs,rhs,slack,binding,satisfied,multiplier}],
                multiplier_source, interpretation}, plot

The status used to be "optimal" whatever SLSQP said, so a model with
contradicting constraints came back "optimal" with a violated constraint. Now
the solver's own verdict is passed through, every constraint carries a
`satisfied` flag, and when the returned point violates a constraint the
linear feasibility of the constraints (with x >= 0) is settled exactly by an
LP (HiGHS): no feasible point -> "infeasible", otherwise "not_converged".
"""
import sys, json, io, base64
import numpy as np
from scipy.optimize import minimize, linprog

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

FEAS_TOL = 1e-5      # constraint violation tolerance, x max(1, |rhs|)
BIND_TOL = 1e-5      # |slack| below this -> binding (unchanged)
EIG_TOL = 1e-8
BIG_X = 1e7          # a non-converged run that wandered this far is read as unbounded


def _fin(x, nd=6):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def _lp_feasible(A, ops, rhs, n):
    """Exact feasibility of {A x (op) rhs, x >= 0} by an LP with a zero objective.
    Returns (feasible: bool|None, highs status code)."""
    A_ub, b_ub, A_eq, b_eq = [], [], [], []
    for i, row in enumerate(A):
        a = np.array(row, float); op = ops[i] if i < len(ops) else "<="
        if op in ("=", "=="):
            A_eq.append(a); b_eq.append(rhs[i])
        elif op == ">=":
            A_ub.append(-a); b_ub.append(-rhs[i])
        else:
            A_ub.append(a); b_ub.append(rhs[i])
    try:
        r = linprog(np.zeros(n), A_ub=np.array(A_ub) if A_ub else None, b_ub=np.array(b_ub) if b_ub else None,
                    A_eq=np.array(A_eq) if A_eq else None, b_eq=np.array(b_eq) if b_eq else None,
                    bounds=[(0, None)] * n, method="highs")
    except Exception:
        return None, None
    if r.status == 0:
        return True, 0
    if r.status == 2:
        return False, 2
    return None, int(r.status)


def _improving_recession_direction(Qs, c, A, ops, n, maximize):
    """Convex QP only: is there a direction d >= 0 that keeps every constraint
    (A_i d <=/>=/= 0), leaves the quadratic part flat (Q d = 0) and strictly
    improves the linear part? Then the objective is unbounded along x + t d.
    Settled by an LP over 0 <= d <= 1. Returns True / False / None (LP failed)."""
    A_ub, b_ub = [], []
    A_eq, b_eq = [list(row) for row in Qs], [0.0] * n
    for i, row in enumerate(A):
        a = np.array(row, float); op = ops[i] if i < len(ops) else "<="
        if op in ("=", "=="):
            A_eq.append(a); b_eq.append(0.0)
        elif op == ">=":
            A_ub.append(-a); b_ub.append(0.0)
        else:
            A_ub.append(a); b_ub.append(0.0)
    try:
        r = linprog((-c if maximize else c), A_ub=np.array(A_ub) if A_ub else None, b_ub=np.array(b_ub) if b_ub else None,
                    A_eq=np.array(A_eq), b_eq=np.array(b_eq), bounds=[(0, 1)] * n, method="highs")
    except Exception:
        return None
    if r.status != 0:
        return None
    return bool(r.fun < -1e-9 * max(1.0, float(np.linalg.norm(c))))


def _kkt_multipliers(grad, A, active_rows, active_bounds):
    """d(objective)/d(rhs) of each active constraint from KKT stationarity:
    grad f = sum_i nu_i a_i + sum_j mu_j e_j over the active constraints and
    active bounds x_j = 0 (least squares). Inactive constraints get 0."""
    cols = [np.array(A[i], float) for i in active_rows]
    n = len(grad)
    cols += [np.eye(n)[j] for j in active_bounds]
    if not cols:
        return {}
    G = np.array(cols).T
    nu, *_ = np.linalg.lstsq(G, grad, rcond=None)
    return {i: float(nu[k]) for k, i in enumerate(active_rows)}


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
        try:
            code = int(res.status)
        except (TypeError, ValueError):
            code = None
        solver = {"success": bool(res.success), "solver_status_code": code,
                  "message": str(res.message), "solver_message": str(res.message),
                  "nit": int(getattr(res, "nit", 0) or 0)}
        if not np.all(np.isfinite(res.x)):
            print(json.dumps({"results": {"status": "failed", "unsolved": True, **solver,
                              "message": f"Solver did not converge: {res.message}"}, "plot": None}))
            return
        x = res.x
        obj_val = float(0.5 * x @ Qs @ x + c @ x)

        # Convexity depends on the sense: minimisation needs Q PSD, maximisation Q NSD.
        eig = np.linalg.eigvalsh(Qs)
        q_psd = bool(np.min(eig) >= -EIG_TOL)
        q_nsd = bool(np.max(eig) <= EIG_TOL)
        convex_for_sense = q_nsd if maximize else q_psd
        need = "negative semidefinite (maximization)" if maximize else "positive semidefinite (minimization)"
        convexity_note = (f"Q is {need.split(' (')[0]}, so this {'maximization' if maximize else 'minimization'} "
                          "is a convex QP: a local optimum is the global optimum."
                          if convex_for_sense else
                          f"A {'maximization' if maximize else 'minimization'} is a convex QP only when Q is {need.split(' (')[0]}; "
                          f"this Q is not (eigenvalues from {eig.min():.4g} to {eig.max():.4g}), so the problem is "
                          "non-convex: the point found is only a local (KKT) solution and a better one may exist.")

        constraints = []
        n_binding = 0
        violated = []
        for i in range(m):
            lhs = float(np.array(A[i]) @ x); op = ops[i] if i < len(ops) else "<="
            tol = FEAS_TOL * max(1.0, abs(rhs[i]))
            if op in ("=", "=="):
                slack = 0.0; binding = True; sat = abs(lhs - rhs[i]) <= tol
            elif op == ">=":
                slack = lhs - rhs[i]; binding = abs(slack) < BIND_TOL; sat = slack >= -tol
            else:
                slack = rhs[i] - lhs; binding = abs(slack) < BIND_TOL; sat = slack >= -tol
            name = cnames[i] if i < len(cnames) else f"C{i+1}"
            if binding and sat:
                n_binding += 1
            if not sat:
                violated.append(name)
            constraints.append({"name": name, "op": op,
                                "lhs": _fin(lhs, 6), "rhs": _fin(rhs[i], 6),
                                "slack": _fin(slack, 6), "binding": bool(binding),
                                "satisfied": bool(sat), "multiplier": None})
        bounds_ok = bool(np.all(x >= -FEAS_TOL))
        feasible_pt = not violated and bounds_ok

        feas_check = None
        unbounded_check = None
        if feasible_pt:
            # SLSQP can "succeed" far out on an unbounded model (a linear
            # objective with no upper limit stops at ~1e8 with success=True).
            # For a convex QP unboundedness is settled exactly by an LP over
            # the recession directions; for a non-convex one a run that
            # wandered past BIG_X is read as unbounded.
            if convex_for_sense:
                unb = _improving_recession_direction(Qs, c, A, ops, n, maximize)
                unbounded_check = {"method": "recession_direction_lp", "unbounded": unb}
            else:
                unb = bool(np.max(np.abs(x)) > BIG_X)
                unbounded_check = {"method": "variables_beyond_1e7", "unbounded": unb}
            if unb:
                status = "unbounded"
            elif res.success:
                status = "optimal"
            else:
                status = "not_converged"
        else:
            lp_ok, lp_code = _lp_feasible(A, ops, rhs, n)
            feas_check = {"method": "linprog_highs", "feasible": lp_ok, "highs_status_code": lp_code}
            status = "infeasible" if lp_ok is False else "not_converged"

        mult_src = None
        if status == "optimal" and m:
            try:
                grad = Qs @ x + c                                   # original objective
                active_rows = [i for i, cc in enumerate(constraints) if cc["binding"]]
                active_bounds = [j for j in range(n) if x[j] <= 1e-7]
                mu = _kkt_multipliers(grad, A, active_rows, active_bounds)
                for i, cc in enumerate(constraints):
                    cc["multiplier"] = _fin(mu.get(i, 0.0), 6)
                mult_src = "kkt_estimate"
            except Exception:
                mult_src = None

        variables = [{"name": var_names[i], "value": _fin(x[i], 6)} for i in range(n)]

        plot = None
        if n == 2 and status != "unbounded":   # a point far out on an unbounded ray draws nothing useful
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
                ax.set_title("Quadratic objective contours & optimum (★)" if status == "optimal"
                             else "Quadratic objective contours & point where the solver stopped (★)")
                fig.tight_layout()
                buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
                plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            except Exception:
                plt.close("all"); plot = None

        at = ", ".join(f"{var_names[i]}={x[i]:.4f}" for i in range(n))
        why = f"(solver status {code}: {res.message})"
        if status == "optimal":
            interpretation = (
                f"The {'maximum' if maximize else 'minimum'} of the quadratic objective is {obj_val:.4f} at {at}. "
                + convexity_note + " "
                + f"{n_binding} of {m} constraint(s) are binding."
            )
        elif status == "infeasible":
            interpretation = (
                "No feasible solution: the linear constraints cannot all hold together with x ≥ 0 "
                f"(confirmed by a linear feasibility check). At the point where SLSQP stopped {why}, "
                f"{', '.join(violated) if violated else 'the bounds x ≥ 0'} "
                f"{'is' if len(violated) == 1 else 'are'} violated. Check the right-hand sides for constraints "
                "that contradict each other."
            )
        elif status == "unbounded":
            exact = unbounded_check and unbounded_check["method"] == "recession_direction_lp"
            interpretation = (
                f"The objective is unbounded: it can be {'increased' if maximize else 'decreased'} without limit "
                "while every constraint still holds"
                + (" (confirmed by a linear check of the directions the constraints leave open)" if exact
                   else f" — SLSQP {'stopped' if not res.success else 'ended'} {why} with the variables at "
                   + ", ".join(f"{var_names[i]}={x[i]:.4g}" for i in range(n)))
                + ". A constraint that limits the variables is probably missing."
            )
        else:
            interpretation = (
                f"SLSQP stopped without converging {why}, so the point {at} "
                + ("violates " + ", ".join(violated) + " and " if violated else "")
                + "is not proven optimal (objective there: "
                + f"{obj_val:.4f}). Try rescaling the model or solving it with Nonlinear Programming from another starting point."
            )
        results = {"status": status, "unsolved": False, **solver, "maximize": maximize,
                   "n_vars": n, "n_constraints": m, "objective_value": _fin(obj_val, 6),
                   "n_binding": n_binding, "n_violated": len(violated),
                   "is_convex": q_psd, "q_psd": q_psd, "q_nsd": q_nsd,
                   "is_convex_for_sense": bool(convex_for_sense),
                   "q_min_eigenvalue": _fin(eig.min(), 8), "q_max_eigenvalue": _fin(eig.max(), 8),
                   "convexity_note": convexity_note, "feasibility_check": feas_check,
                   "unbounded_check": unbounded_check,
                   "multiplier_source": mult_src,
                   "variables": variables, "constraints": constraints, "interpretation": interpretation}
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
