#!/usr/bin/env python3
"""Assignment Problem — optimal one-to-one matching. scipy Hungarian algorithm.

Input (from assignment-problem-page.tsx):
    worker_names : string[]
    task_names   : string[]
    cost_matrix  : number[][]   rows = workers, cols = tasks
    minimize     : bool (default True)
Output: { results: {pairs[], total_cost, ...}, plot }
"""
import sys, json, io, base64
import numpy as np
from scipy.optimize import linear_sum_assignment

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
        workers = p.get("worker_names") or []
        tasks = p.get("task_names") or []
        cost = p.get("cost_matrix")
        minimize = bool(p.get("minimize", True))
        if cost is None or not workers or not tasks:
            raise ValueError("Provide worker names, task names and a cost matrix.")
        C = np.array(cost, dtype=float)
        if C.shape != (len(workers), len(tasks)):
            raise ValueError("Cost matrix must be workers (rows) by tasks (columns).")

        M = C if minimize else -C
        row_ind, col_ind = linear_sum_assignment(M)
        total_cost = float(C[row_ind, col_ind].sum())

        pairs = []
        assigned_cols = set(col_ind.tolist())
        for wi, ti in zip(row_ind, col_ind):
            pairs.append({"worker": workers[wi], "task": tasks[ti],
                          "cost": _fin(C[wi, ti], 4), "assigned": True})

        plot = None
        try:
            fig, ax = plt.subplots(figsize=(max(6, len(tasks) * 0.9 + 2), max(4.5, len(workers) * 0.7 + 1.5)), dpi=120)
            im = ax.imshow(C, cmap="Blues", aspect="auto")
            ax.set_xticks(range(len(tasks))); ax.set_xticklabels(tasks, rotation=30, ha="right", fontsize=8)
            ax.set_yticks(range(len(workers))); ax.set_yticklabels(workers, fontsize=8)
            for i in range(len(workers)):
                for j in range(len(tasks)):
                    ax.text(j, i, f"{C[i,j]:g}", ha="center", va="center", fontsize=7,
                            color="white" if C[i, j] > C.mean() else "#111827")
            for wi, ti in zip(row_ind, col_ind):
                ax.add_patch(plt.Rectangle((ti - 0.5, wi - 0.5), 1, 1, fill=False, edgecolor="#dc2626", lw=2.5))
            ax.set_title(f"Optimal assignment ({'min' if minimize else 'max'} cost = {total_cost:g})")
            fig.colorbar(im, ax=ax, label="Cost")
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        pair_txt = ", ".join(f"{pp['worker']}→{pp['task']}" for pp in pairs[:8])
        interpretation = (
            f"The Hungarian algorithm found the {'lowest' if minimize else 'highest'}-cost one-to-one matching, "
            f"assigning each worker to exactly one task for a total of {total_cost:,.4g}. The optimal pairing is "
            f"{pair_txt}. This is a globally optimal matching, not a greedy one: swapping any two assignments would "
            f"{'raise' if minimize else 'lower'} the total, so no rearrangement can do better. Each worker's individual "
            "cost is not necessarily their personal minimum — the algorithm trades off across the whole set to optimise the total."
        )

        results = {
            "status": "ok", "minimize": minimize,
            "n_workers": len(workers), "n_tasks": len(tasks),
            "total_cost": _fin(total_cost, 4), "pairs": pairs,
            "interpretation": interpretation,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
