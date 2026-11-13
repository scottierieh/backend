#!/usr/bin/env python3
"""Multi-Objective Optimization — trade off several linear objectives under
linear constraints via the weighted-sum method, and trace the Pareto front.

Input: var_names[], obj_expr[] (linear), obj_names[], obj_maximize[bool],
       obj_weights[] (optional — one non-negative weight per objective; leave it
       out or empty to get the Pareto points without picking one),
       con_matrix[m][n], con_ops[], con_rhs[], con_names[], n_pareto_points(int)
Output: results{status (optimal|infeasible|unbounded|iteration_limit|numerical_error),
                unsolved, message, solver_status_code, solver_message,
                n_vars, n_objectives, n_constraints, weighted(bool), weights_used,
                chosen_objective_values:[number] | null (objective order; null when
                no weights were given), variables:[{name,value}] ([] without weights),
                pareto_front:[{weights,objective_values,variables,dominated:false}],
                dominated_points:[{weights,objective_values,variables,dominated:true,
                                   dominated_by,reason}], n_dominated,
                failed_weightings:[{weights,status,solver_status_code}],
                interpretation}, plot

Objectives are read as linear functions (coefficients from f(0) and f(e_i));
every objective is first tested for linearity at seeded random points
(f(x+y) = f(x) + f(y) - f(0) and f(t x) - f(0) = t (f(x) - f(0))) and a
nonlinear one is refused with a clear error instead of being silently misread.

A weighting that gives some objective weight 0 can return a point that is
only weakly Pareto-optimal (another point is as good in every objective and
better in that one). Such weightings get a second, lexicographic LP that keeps
the weighted optimum and improves the zero-weight objectives; the weak point
is reported in dominated_points (dominated: true), not on the front.
"""
import sys, json, io, base64
import numpy as np
from scipy.optimize import linprog
from _optexpr import make_func

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

# scipy.optimize.linprog (HiGHS) status codes
LP_STATUS = {0: "optimal", 1: "iteration_limit", 2: "infeasible", 3: "unbounded", 4: "numerical_error"}


def _fin(x, nd=6):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def check_linear(f, n, label, seed=0):
    """Raise ValueError when f is not affine. Seeded random points of both
    signs and two scales; tolerance relative to the values involved."""
    rng = np.random.default_rng(seed)

    def ev(z):
        with np.errstate(all="ignore"):
            try:
                v = float(f(z))
            except Exception:
                return np.nan
        return v

    f0 = ev(np.zeros(n))
    bad = not np.isfinite(f0)
    for trial in range(8):
        if bad:
            break
        s = 1.0 if trial < 4 else 10.0
        x = rng.standard_normal(n) * s
        y = rng.standard_normal(n) * s
        t = float(rng.uniform(-3.0, 3.0))
        fx, fy, fxy, ftx = ev(x), ev(y), ev(x + y), ev(t * x)
        if not all(np.isfinite(v) for v in (fx, fy, fxy, ftx)):
            bad = True; break
        scale = max(1.0, abs(fx) + abs(fy) + abs(fxy) + abs(ftx) + abs(f0))
        if abs(fxy - fx - fy + f0) > 1e-7 * scale or abs((ftx - f0) - t * (fx - f0)) > 1e-7 * scale:
            bad = True
    if bad:
        raise ValueError(
            f"Objective '{label}' is not linear in the decision variables (it fails f(x+y) = f(x) + f(y) - f(0) "
            "at sampled points, or is undefined somewhere). Multi-objective optimization here reads each objective "
            "as a linear function, so a nonlinear one would be silently misread. Rewrite it as a linear expression, "
            "or use Nonlinear Programming / Constrained Optimization for a nonlinear objective.")
    return f0


def lin_coeffs(expr, var_names, label=None):
    f = make_func(expr, var_names)
    n = len(var_names)
    check_linear(f, n, label or expr)
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
        K = len(exprs)
        if n < 1 or K < 2:
            raise ValueError("Need at least one variable and two objectives.")
        if len(omax) < K:
            omax = omax + [True] * (K - len(omax))
        raw_w = p.get("obj_weights")
        weighted = isinstance(raw_w, list) and len(raw_w) > 0
        weights = None
        if weighted:
            weights = [float(w) for w in raw_w]
            if len(weights) != K:
                raise ValueError(f"obj_weights must give one weight per objective ({K}); got {len(weights)}.")
            if not all(np.isfinite(w) and w >= 0 for w in weights) or sum(weights) <= 0:
                raise ValueError("obj_weights must be non-negative numbers with a positive sum.")
        A = [[float(v) for v in row] for row in (p.get("con_matrix") or [])]
        ops = [str(o) for o in (p.get("con_ops") or [])]
        rhs = [float(r) for r in (p.get("con_rhs") or [])]
        npar = max(1, min(int(p.get("n_pareto_points") or 12), 200))

        def oname(k):
            return str(onames[k]) if k < len(onames) else f"Objective {k+1}"

        coefs = []; consts = []
        for k, e in enumerate(exprs):
            cf, c0 = lin_coeffs(e, var_names, label=f"{oname(k)}: {e}")
            coefs.append(cf); consts.append(c0)
        A_ub, b_ub, A_eq, b_eq = build_lp(A, ops, rhs, n)
        bounds = [(0, None)]*n
        sgn = [1.0 if omax[k] else -1.0 for k in range(K)]

        def lp(c_max, extra_ub=None, extra_b=None):
            """maximise c_max . x over the shared region (+ optional extra row)."""
            Au, bu = A_ub, b_ub
            if extra_ub is not None:
                Au = np.vstack([A_ub, extra_ub]) if A_ub is not None else np.array([extra_ub])
                bu = np.concatenate([b_ub, [extra_b]]) if b_ub is not None else np.array([extra_b])
            return linprog(-c_max, A_ub=Au, b_ub=bu, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

        def obj_vals(xx):
            return [float(coefs[k] @ xx + consts[k]) for k in range(K)]

        coef_l1 = [float(np.sum(np.abs(coefs[k]))) for k in range(K)]

        def better_eq(u, v, xscale=1.0, tol=1e-6):
            """sense-adjusted: u at least as good as v everywhere, strictly better somewhere.
            The per-objective tolerance covers the LP's own accuracy: values of
            size |f| and a coefficient vector times x of size |coef|_1 * |x|."""
            du = [sgn[k] * (u[k] - v[k]) for k in range(K)]
            sc = [tol * max(1.0, abs(u[k]), abs(v[k]), coef_l1[k] * xscale) for k in range(K)]
            return all(du[k] >= -sc[k] for k in range(K)) and any(du[k] > sc[k] for k in range(K))

        def xscale_of(*xs):
            return max([1.0] + [float(np.max(np.abs(np.asarray(z, float)))) for z in xs if len(z)])

        def solve_point(ws):
            """Weighted-sum LP, then (when some weight is 0) a lexicographic LP that
            keeps the weighted optimum and improves the zero-weight objectives.
            Returns (x_final | None, raw_res, weak_info | None)."""
            c = np.zeros(n)
            for k in range(K):
                c += ws[k] * sgn[k] * coefs[k]
            r = lp(c)
            if not r.success:
                return None, r, None
            x = r.x
            zero = [k for k in range(K) if ws[k] <= 0]
            if not zero:
                return x, r, None
            opt = float(c @ x)
            c2 = np.zeros(n)
            for k in zero:
                c2 += sgn[k] * coefs[k]
            r2 = lp(c2, extra_ub=-c, extra_b=-(opt - 1e-9 * max(1.0, abs(opt))))
            if r2.status == 3:
                # The zero-weight objective(s) improve without limit on the optimal
                # face: the raw point is dominated and this weighting has no
                # Pareto-optimal point.
                return None, r, {"x": x, "dominated_by": None,
                                 "reason": "a zero-weight objective can still be improved without limit "
                                           "while keeping the weighted optimum"}
            if r2.success and better_eq(obj_vals(r2.x), obj_vals(x), xscale_of(x, r2.x)):
                return r2.x, r, {"x": x, "dominated_by": obj_vals(r2.x),
                                 "reason": "found with a zero weight: another point with the same weighted "
                                           "optimum is at least as good in every objective and better in one"}
            return x, r, None

        def _point(w, xx, dominated=False):
            v = obj_vals(xx)
            return {
                "weights": [_fin(float(z), 4) for z in w],
                "objective_values": [_fin(z, 4) for z in v],
                "variables": {var_names[i]: _fin(xx[i], 6) for i in range(n)},
                "dominated": dominated,
            }, tuple(round(z, 4) for z in v)

        def fail_out(code, message):
            # The status used to be "infeasible" for every failure, even an
            # unbounded model; it now follows the HiGHS status code.
            st = LP_STATUS.get(code, "failed")
            words = {"infeasible": "No feasible solution", "unbounded": "Unbounded objective",
                     "iteration_limit": "Iteration limit reached", "numerical_error": "Numerical difficulties"}
            print(json.dumps({"results": {"status": st, "unsolved": True, "weighted": weighted,
                                          "solver_status_code": code, "solver_message": str(message),
                                          "message": f"{words.get(st, 'No solution')}: {message}"}, "plot": None}))

        # 1. Feasibility of the shared region on its own (zero objective) —
        #    separates "infeasible" from "unbounded" before any weighting.
        feas = lp(np.zeros(n))
        if not feas.success:
            fail_out(int(feas.status), feas.message); return

        # 2. The point the user's weights select (only when weights were given).
        chosen = None
        x = None
        chosen_weak = None
        chosen_res = feas
        if weighted:
            x, chosen_res, chosen_weak = solve_point(weights)
            if x is None:
                if chosen_weak is not None:
                    fail_out(3, chosen_weak["reason"]); return
                fail_out(int(chosen_res.status), chosen_res.message); return
            chosen = obj_vals(x)
        variables = [{"name": var_names[i], "value": _fin(x[i], 6)} for i in range(n)] if weighted else []
        chosen_objective_values = [_fin(chosen[k], 6) for k in range(K)] if weighted else None

        # 3. Pareto front: vary weights (2-obj: sweep; >2: random samples).
        if K == 2:
            wlist = [[float(t), float(1 - t)] for t in np.linspace(0, 1, max(npar, 2))]
        else:
            rng = np.random.default_rng(0)
            wlist = []
            for _ in range(max(npar, 4)):
                w = rng.random(K); w /= w.sum(); wlist.append(list(map(float, w)))
        pareto, seen = [], set()
        dominated, dseen = [], set()
        failed = []
        last_fail = None
        for w in wlist:
            xx, r, weak = solve_point(w)
            if weak is not None:
                dp, dkey = _point(w, weak["x"], dominated=True)
                if dkey not in dseen:
                    dseen.add(dkey)
                    dp["dominated_by"] = ([_fin(z, 4) for z in weak["dominated_by"]]
                                          if weak["dominated_by"] is not None else None)
                    dp["reason"] = weak["reason"]
                    dominated.append(dp)
            if xx is None:
                if weak is None:
                    failed.append({"weights": [_fin(z, 4) for z in w], "status": LP_STATUS.get(int(r.status), "failed"),
                                   "solver_status_code": int(r.status)})
                    last_fail = r
                continue
            pt, key = _point(w, xx)
            if key not in seen:
                seen.add(key); pareto.append(pt)
        # Safety net: drop any point another found point dominates.
        keep = []
        for i, pt in enumerate(pareto):
            px = [v for v in pt["variables"].values() if v is not None]
            dom = next((q for j, q in enumerate(pareto)
                        if j != i and better_eq(q["objective_values"], pt["objective_values"],
                                                xscale_of(px, [v for v in q["variables"].values() if v is not None]),
                                                tol=1e-5)), None)
            if dom is not None:
                pt = dict(pt); pt["dominated"] = True
                pt["dominated_by"] = dom["objective_values"]
                pt["reason"] = "dominated by another point found in the weight sweep"
                dominated.append(pt)
            else:
                keep.append(pt)
        pareto = keep
        if not weighted and not pareto:
            if last_fail is not None:
                fail_out(int(last_fail.status), last_fail.message); return
            fail_out(3, "no weighting has a Pareto-optimal point (a zero-weight objective is unbounded)"); return

        plot = None
        if K == 2 and pareto:
            try:
                fig, ax = plt.subplots(figsize=(7, 5.5), dpi=120)
                pv = sorted([(pt["objective_values"][0], pt["objective_values"][1]) for pt in pareto])
                xs = [a for a, _ in pv]; ys = [b for _, b in pv]
                ax.plot(xs, ys, "-o", color="#2563eb", label="Pareto front")
                if dominated:
                    ax.plot([d["objective_values"][0] for d in dominated], [d["objective_values"][1] for d in dominated],
                            "x", color="#9ca3af", markersize=9, linestyle="none", label="dominated (weight 0)")
                if weighted:
                    ax.plot(chosen[0], chosen[1], "*", color="#dc2626", markersize=18, label="chosen (weights)")
                ax.set_xlabel(oname(0)); ax.set_ylabel(oname(1))
                ax.set_title("Pareto front — no point can improve one objective without hurting the other")
                ax.legend(fontsize=8, frameon=False); fig.tight_layout()
                buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
                plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
            except Exception:
                plt.close("all"); plot = None

        front_txt = (f"The Pareto front ({len(pareto)} point(s)) shows the set of non-dominated trade-offs found: "
                     "moving along it, improving one objective necessarily worsens another. ")
        if weighted:
            interpretation = (
                f"With the given weights, the best compromise sets "
                + ", ".join(f"{var_names[i]}={x[i]:.3f}" for i in range(n)) + ", giving "
                + ", ".join(f"{oname(k)}={chosen[k]:.3f}" for k in range(K)) + ". "
                + front_txt
                + "The chosen point is where your weights land on that front — change the weights to slide toward "
                  "whichever objective matters more."
            )
        else:
            interpretation = (
                "No weights were given, so no single point is chosen. " + front_txt
                + "Which point to pick depends on how you rank the objectives; enter weights to select one."
            )
        if dominated:
            interpretation += (
                f" {len(dominated)} point(s) found with a zero weight were only weakly Pareto-optimal (another point "
                "is at least as good in every objective and better in one); they are left off the front and listed "
                "as dominated.")
        if failed:
            interpretation += (f" {len(failed)} weighting(s) had no optimal solution "
                               f"({', '.join(sorted(set(f['status'] for f in failed)))}) and are left out.")
        results = {"status": "optimal", "unsolved": False, "n_vars": n, "n_objectives": K,
                   "n_constraints": len(A), "weighted": weighted, "weights_used": weights,
                   "solver_status_code": int(chosen_res.status), "solver_message": str(chosen_res.message),
                   "chosen_objective_values": chosen_objective_values,
                   "variables": variables, "pareto_front": pareto,
                   "dominated_points": dominated, "n_dominated": len(dominated),
                   "failed_weightings": failed,
                   "interpretation": interpretation}
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
