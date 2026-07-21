#!/usr/bin/env python3
"""Assignment Problem — one worker per task at min (or max) total cost.
Hungarian algorithm via scipy.optimize.linear_sum_assignment.

Input: worker_names[], task_names[], cost_matrix[w][t], minimize(bool, default True)
Output: results{status, minimize, n_workers, n_tasks, total_cost,
                pairs:[{worker,task,cost}], interpretation}, plot
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
        workers = [str(x) for x in (p.get("worker_names") or [])]
        tasks = [str(x) for x in (p.get("task_names") or [])]
        C = np.array([[float(v) for v in row] for row in (p.get("cost_matrix") or [])], dtype=float)
        minimize = bool(p.get("minimize", True))
        w, t = len(workers), len(tasks)
        if w < 1 or t < 1:
            raise ValueError("Need at least one worker and one task.")
        if C.shape != (w, t):
            raise ValueError(f"cost_matrix must be {w} x {t}.")

        cost = C if minimize else -C
        rows, cols = linear_sum_assignment(cost)
        pairs = [{"worker": workers[r], "task": tasks[c], "cost": _fin(C[r, c], 4)}
                 for r, c in zip(rows, cols)]
        total = float(sum(C[r, c] for r, c in zip(rows, cols)))
        n_assigned = len(pairs)

        plot = None
        try:
            fig, ax = plt.subplots(figsize=(max(6, t * 1.1), max(4, w * 0.8)), dpi=120)
            mask = np.zeros_like(C)
            for r, c in zip(rows, cols):
                mask[r, c] = 1
            im = ax.imshow(C, cmap="Blues" if minimize else "Greens", aspect="auto")
            for r, c in zip(rows, cols):
                ax.add_patch(plt.Rectangle((c-0.5, r-0.5), 1, 1, fill=False, edgecolor="#dc2626", lw=2.5))
            ax.set_xticks(range(t)); ax.set_xticklabels(tasks, rotation=40, ha="right", fontsize=8)
            ax.set_yticks(range(w)); ax.set_yticklabels(workers, fontsize=8)
            ax.set_title(f"Optimal assignment ({'min' if minimize else 'max'} cost) — red = chosen")
            fig.colorbar(im, ax=ax, shrink=0.8)
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        interpretation = (
            f"The optimal one-to-one assignment pairs {n_assigned} worker(s) with task(s) for a "
            f"{'minimum' if minimize else 'maximum'} total cost of {total:,.2f}. "
            "Each worker gets exactly one task and vice versa; the Hungarian algorithm guarantees "
            "this is the best possible matching, not just a good one."
            + ("" if w == t else f" With {w} workers and {t} tasks, the {abs(w-t)} extra {'workers' if w>t else 'tasks'} are left unassigned.")
        )
        results = {"status": "optimal", "unsolved": False, "minimize": minimize,
                   "n_workers": w, "n_tasks": t, "total_cost": _fin(total, 4),
                   "pairs": pairs, "interpretation": interpretation}
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
