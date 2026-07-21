#!/usr/bin/env python3
"""Workforce Scheduling — staff each shift to meet demand at minimum cost.

Input: shift_names[], demand[] (workers required per shift), cost_per_worker[]
Output: results{status, unsolved, message, n_shifts, total_workers, total_cost,
                shifts:[{name,demand,workers,cost}], interpretation}, plot

Each shift must be staffed to at least its required demand; with per-shift
independent staffing the least-cost solution assigns exactly the demand (workers
are whole people, so demand is rounded up).
"""
import sys, json, io, base64
import math
import numpy as np

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
        names = [str(x) for x in (p.get("shift_names") or [])]
        demand = [float(x) for x in (p.get("demand") or [])]
        cost = [float(x) for x in (p.get("cost_per_worker") or [])]
        S = len(names)
        if S < 1:
            raise ValueError("Need at least one shift.")
        if not (len(demand) == len(cost) == S):
            raise ValueError("demand / cost_per_worker must match shift count.")
        if any(d < 0 for d in demand) or any(c < 0 for c in cost):
            raise ValueError("Demand and cost must be non-negative.")

        workers = [int(math.ceil(d - 1e-9)) for d in demand]  # whole workers, meet demand
        shifts = [{"name": names[s], "demand": _fin(demand[s], 2), "workers": workers[s],
                   "cost": _fin(workers[s] * cost[s], 2)} for s in range(S)]
        total_workers = int(sum(workers))
        total_cost = float(sum(workers[s] * cost[s] for s in range(S)))

        plot = None
        try:
            fig, ax = plt.subplots(figsize=(max(7, S * 0.9), 4.4), dpi=120)
            x = np.arange(S)
            ax.bar(x, workers, color="#2563eb")
            ax.plot(x, demand, "o--", color="#dc2626", label="demand")
            ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel("workers"); ax.set_title(f"Staffing per shift — total cost {total_cost:,.2f}")
            ax.legend(fontsize=8, frameon=False); fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        peak = names[int(np.argmax(workers))] if S else "—"
        interpretation = (
            f"Meeting every shift's demand needs {total_workers} worker-assignments at a total cost of "
            f"{total_cost:,.2f}. The heaviest shift is {peak}. Worker counts are rounded up to whole "
            "people, so each shift is covered with the fewest staff that still satisfies its requirement."
        )
        results = {"status": "optimal", "unsolved": False, "n_shifts": S,
                   "total_workers": total_workers, "total_cost": _fin(total_cost, 2),
                   "shifts": shifts, "interpretation": interpretation}
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
