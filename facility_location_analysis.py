#!/usr/bin/env python3
"""Facility Location — which facilities to open + who they serve, at min total cost.
Capacitated MILP via scipy.optimize.milp (binary open decisions + continuous flow).

Input: facility_names[F], fixed_cost[F], capacity[F],
       customer_names[C], demand[C], ship_cost[F][C]
Output: results{status, unsolved, message, n_facilities, n_customers, n_opened,
                total_cost, fixed_cost_total, shipping_cost_total,
                facilities:[{name,opened,used,capacity}],
                assignments:[{customer,facility,amount}], interpretation}, plot
"""
import sys, json, io, base64
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _fin(x, nd=4):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def main():
    try:
        p = json.load(sys.stdin)
        fac = [str(x) for x in (p.get("facility_names") or [])]
        fixed = [float(x) for x in (p.get("fixed_cost") or [])]
        cap = [float(x) for x in (p.get("capacity") or [])]
        cust = [str(x) for x in (p.get("customer_names") or [])]
        dem = [float(x) for x in (p.get("demand") or [])]
        S = [[float(v) for v in row] for row in (p.get("ship_cost") or [])]  # F x C
        F, C = len(fac), len(cust)
        if F < 1 or C < 1:
            raise ValueError("Need at least one facility and one customer.")
        if len(fixed) != F or len(cap) != F:
            raise ValueError("fixed_cost/capacity must match facility count.")
        if len(dem) != C:
            raise ValueError("demand must match customer count.")
        if len(S) != F or any(len(r) != C for r in S):
            raise ValueError(f"ship_cost must be {F} x {C}.")
        if sum(cap) + 1e-9 < sum(dem):
            results = {"status": "infeasible", "unsolved": True,
                       "message": f"Total capacity ({sum(cap):g}) is below total demand ({sum(dem):g})."}
            print(json.dumps({"results": results, "plot": None}))
            return

        # vars: x_fc (F*C continuous >=0) then y_f (F binary)
        nx = F * C
        nv = nx + F
        c_obj = np.zeros(nv)
        for f in range(F):
            for cc in range(C):
                c_obj[f * C + cc] = S[f][cc]
        for f in range(F):
            c_obj[nx + f] = fixed[f]

        cons = []
        # demand: sum_f x_fc = demand_c
        for cc in range(C):
            row = np.zeros(nv)
            for f in range(F):
                row[f * C + cc] = 1.0
            cons.append(LinearConstraint(row, dem[cc], dem[cc]))
        # capacity linking: sum_c x_fc - cap_f y_f <= 0
        for f in range(F):
            row = np.zeros(nv)
            for cc in range(C):
                row[f * C + cc] = 1.0
            row[nx + f] = -cap[f]
            cons.append(LinearConstraint(row, -np.inf, 0.0))

        integrality = np.zeros(nv); integrality[nx:] = 1   # y_f binary
        lb = np.zeros(nv); ub = np.full(nv, np.inf); ub[nx:] = 1.0
        res = milp(c=c_obj, constraints=cons, integrality=integrality, bounds=Bounds(lb, ub))
        if not res.success:
            results = {"status": "infeasible", "unsolved": True,
                       "message": f"No feasible facility plan: {res.message}"}
            print(json.dumps({"results": results, "plot": None}))
            return
        v = res.x
        X = v[:nx].reshape(F, C)
        Y = v[nx:]
        tol = 1e-6
        fixed_total = float(sum(fixed[f] for f in range(F) if Y[f] > 0.5))
        ship_total = float(np.sum(X * np.array(S)))
        total = fixed_total + ship_total
        n_opened = int(sum(1 for f in range(F) if Y[f] > 0.5))
        facilities = [{"name": fac[f], "opened": bool(Y[f] > 0.5),
                       "used": _fin(float(np.sum(X[f])), 4), "capacity": _fin(cap[f], 4)} for f in range(F)]
        assignments = []
        for cc in range(C):
            for f in range(F):
                if X[f, cc] > tol:
                    assignments.append({"customer": cust[cc], "facility": fac[f], "amount": _fin(X[f, cc], 4)})

        plot = None
        try:
            fig, ax = plt.subplots(figsize=(max(6, C * 1.0), max(4, F * 0.8)), dpi=120)
            im = ax.imshow(X, cmap="Blues", aspect="auto")
            for f in range(F):
                for cc in range(C):
                    if X[f, cc] > tol:
                        ax.text(cc, f, f"{X[f,cc]:g}", ha="center", va="center", fontsize=8,
                                color="white" if X[f, cc] > X.max()*0.6 else "#1e3a8a")
            ax.set_xticks(range(C)); ax.set_xticklabels(cust, rotation=40, ha="right", fontsize=8)
            ax.set_yticks(range(F)); ax.set_yticklabels([f"{fac[f]}{' ✓' if Y[f]>0.5 else ' ✗'}" for f in range(F)], fontsize=8)
            ax.set_title(f"{n_opened} facilities opened — total cost {total:,.0f}")
            fig.colorbar(im, ax=ax, shrink=0.8, label="units shipped"); fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        interpretation = (
            f"Opening {n_opened} of {F} facilities and serving every customer from them minimises total cost "
            f"at {total:,.2f} — {fixed_total:,.2f} in fixed opening costs plus {ship_total:,.2f} in shipping. "
            "The model weighs each facility's fixed cost against the shipping it saves; a facility opens only "
            "when the customers it can serve cheaply justify its fixed cost."
        )
        results = {"status": "optimal", "unsolved": False, "n_facilities": F, "n_customers": C,
                   "n_opened": n_opened, "total_cost": _fin(total, 2),
                   "fixed_cost_total": _fin(fixed_total, 2), "shipping_cost_total": _fin(ship_total, 2),
                   "facilities": facilities, "assignments": assignments, "interpretation": interpretation}
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
