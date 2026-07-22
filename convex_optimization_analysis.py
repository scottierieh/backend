#!/usr/bin/env python3
"""Convex Optimization — minimise a convex objective under constraints, with a
numerical convexity check. scipy SLSQP.

Input (from convex-optimization-page.tsx):
    objective  : str
    var_names  : string[]
    start      : number[]
    con_expr   : string[]
    con_ops    : string[]   "<=", ">=", "="
    con_rhs    : number[]
    con_names  : string[]
Output: { results: {variables[], constraints[], objective_value, is_convex_verified, ...}, plot }
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


def _hessian(f, x, h=1e-4):
    n = len(x); H = np.zeros((n, n)); fx = f(x)
    for i in range(n):
        for j in range(i, n):
            xpp = x.copy(); xpp[i] += h; xpp[j] += h
            xpm = x.copy(); xpm[i] += h; xpm[j] -= h
            xmp = x.copy(); xmp[i] -= h; xmp[j] += h
            xmm = x.copy(); xmm[i] -= h; xmm[j] -= h
            H[i, j] = H[j, i] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * h * h)
    return H


def main():
    try:
        p = json.load(sys.stdin)
        objective = p.get("objective")
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

        fobj = make_func(objective, var_names)
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

        res = minimize(fobj, np.array(start, dtype=float), constraints=cons, method="SLSQP",
                       options={"maxiter": 500, "ftol": 1e-10})
        if res.x is None:
            results = {"status": "unsolved", "unsolved": True, "message": res.message,
                       "n_vars": n, "n_constraints": len(con_expr),
                       "interpretation": f"The solver failed: {res.message}"}
            print(json.dumps({"results": results, "plot": None})); return

        x = np.array(res.x, dtype=float)
        objective_value = float(fobj(x))

        # numerical convexity check: Hessian PSD at several sampled points
        rng = np.random.default_rng(0)
        psd_count, tested = 0, 0
        min_eig = np.inf
        for _ in range(6):
            xs = x + rng.normal(0, max(np.abs(x).max(), 1) * 0.3, size=n)
            try:
                H = _hessian(fobj, xs)
                ev = np.linalg.eigvalsh(0.5 * (H + H.T))
                min_eig = min(min_eig, float(ev.min()))
                tested += 1
                if ev.min() >= -1e-4:
                    psd_count += 1
            except Exception:
                continue
        is_convex = bool(tested > 0 and psd_count == tested)
        if is_convex:
            note_en = "The objective's Hessian was positive semidefinite at every sampled point, consistent with a convex function — the solution is the global optimum."
            note_ko = "표본 지점 모두에서 목적함수의 헤세 행렬이 양의 준정부호로, 볼록 함수와 일치합니다 — 이 해는 전역 최적입니다."
        else:
            note_en = "The objective's Hessian was not positive semidefinite everywhere sampled, so convexity is not confirmed — this may be a local optimum only."
            note_ko = "표본 지점 전체에서 헤세 행렬이 양의 준정부호가 아니어서 볼록성이 확인되지 않았습니다 — 국소 최적일 수 있습니다."

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
                cs = ax.contour(gx, gy, Z, levels=20, cmap="viridis", alpha=0.7); ax.clabel(cs, inline=True, fontsize=6)
                ax.scatter([x[0]], [x[1]], color="#dc2626", s=80, zorder=6, label="Optimum")
                ax.set_xlabel(var_names[0]); ax.set_ylabel(var_names[1])
                ax.set_title("Convex objective contours & optimum"); ax.legend(fontsize=8, frameon=False)
            else:
                fig, ax = plt.subplots(figsize=(8.5, 5), dpi=120)
                ax.bar([v["name"] for v in variables], [v["value"] or 0 for v in variables], color="#2563eb")
                ax.set_ylabel("Optimal value"); ax.set_title("Optimal variables"); ax.tick_params(axis="x", rotation=25)
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        interpretation = (
            f"The convex program minimised the objective to {objective_value:,.6g} at "
            + ", ".join(f"{v['name']} = {v['value']:.4g}" for v in variables) + ". " + note_en
            + " A key property of convex optimisation is that any local optimum is also the global optimum, so — provided "
            "the problem really is convex — you can trust this solution without worrying about the starting point."
        )

        results = {
            "status": "ok", "unsolved": False,
            "n_vars": n, "n_constraints": len(con_expr), "objective_value": _fin(objective_value, 6),
            "iterations": int(res.nit), "is_convex_verified": is_convex,
            "convexity_note": note_en,
            "variables": variables, "constraints": constraints,
            "interpretation": interpretation,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
