#!/usr/bin/env python3
"""Resource Allocation — choose activity levels to maximise value within limits.
LP via scipy.optimize.linprog.

Input: resource_names[R], resource_available[R], activity_names[A],
       activity_value[A], usage_matrix[A][R]  (units of resource r per unit activity a)
Output: results{status, unsolved, message, n_activities, n_resources, total_value,
                activities:[{name,level,value}], resources:[{name,used,available,slack,binding}],
                interpretation}, plot
"""
import sys, json, io, base64
import numpy as np
from scipy.optimize import linprog

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
        rnames = [str(x) for x in (p.get("resource_names") or [])]
        avail = [float(x) for x in (p.get("resource_available") or [])]
        anames = [str(x) for x in (p.get("activity_names") or [])]
        value = [float(x) for x in (p.get("activity_value") or [])]
        U = [[float(v) for v in row] for row in (p.get("usage_matrix") or [])]  # A x R
        A, R = len(anames), len(rnames)
        if A < 1 or R < 1:
            raise ValueError("Need at least one activity and one resource.")
        if len(value) != A or len(avail) != R:
            raise ValueError("value/available lengths must match activity/resource counts.")
        if len(U) != A or any(len(r) != R for r in U):
            raise ValueError(f"usage_matrix must be {A} x {R}.")

        # maximize value.x  ->  minimize -value.x ; s.t. U^T-ish: for resource r: sum_a U[a][r] x_a <= avail_r
        c = -np.array(value)
        A_ub = np.array(U).T   # R x A
        res = linprog(c, A_ub=A_ub, b_ub=np.array(avail), bounds=[(0, None)] * A, method="highs")
        if not res.success:
            results = {"status": "infeasible", "unsolved": True, "message": res.message}
            print(json.dumps({"results": results, "plot": None}))
            return
        x = res.x
        total = float(np.dot(value, x))
        activities = [{"name": anames[a], "level": _fin(x[a], 4), "value": _fin(x[a] * value[a], 4)} for a in range(A)]
        resources = []
        n_binding = 0
        for r in range(R):
            used = float(sum(U[a][r] * x[a] for a in range(A)))
            slack = avail[r] - used
            binding = slack < 1e-6
            if binding:
                n_binding += 1
            resources.append({"name": rnames[r], "used": _fin(used, 4), "available": _fin(avail[r], 4),
                              "slack": _fin(slack, 4), "binding": bool(binding)})

        plot = None
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5), dpi=120)
            ax1.barh(anames, x, color="#2563eb"); ax1.set_title("Activity levels"); ax1.invert_yaxis()
            ax2.barh(rnames, [r["used"] for r in resources], color="#93c5fd", label="used")
            ax2.barh(rnames, avail, color="none", edgecolor="#dc2626", label="available")
            ax2.set_title("Resource usage"); ax2.invert_yaxis(); ax2.legend(fontsize=8, frameon=False)
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        active = [a["name"] for a in activities if (a["level"] or 0) > 1e-6]
        interpretation = (
            f"The best allocation delivers a total value of {total:,.2f}, running {len(active)} of {A} "
            f"activities. {n_binding} resource(s) are fully used (binding) — these are the constraints "
            "holding value back; adding more of a binding resource is where extra value comes from, while "
            "resources with slack are not limiting."
        )
        results = {"status": "optimal", "unsolved": False, "n_activities": A, "n_resources": R,
                   "total_value": _fin(total, 4), "activities": activities, "resources": resources,
                   "interpretation": interpretation}
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
