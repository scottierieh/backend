#!/usr/bin/env python3
"""Workforce Scheduling — cover every shift's demand at minimum cost.

Set-covering (Dantzig) model solved as an integer program via scipy.optimize.milp.

A worker is hired on a SHIFT PATTERN — a run of `shift_length` consecutive
shifts — not on a single shift. That is what makes this an optimisation problem
at all: one hire fills several shifts at once, so the shifts stop being
independent and the plan is no longer "copy the demand column".

With shift_length = 1 every pattern covers exactly one shift, the constraints
separate again, and the model reduces to the old per-shift staffing answer
(workers = ceil(demand)). That is the backward-compatible default.

Input: shift_names[S], demand[S] (workers required per shift),
       cost_per_worker[S] (wage for working ONE shift),
       shift_length (int >= 1, default 1), cyclic (bool, default True)
Output: results{status, unsolved, message, n_shifts, shift_length, cyclic,
                total_cost, total_workers, n_hires, total_assignments,
                total_surplus, separable,
                shifts:[{shift,demand,workers_assigned,cost,surplus}],
                patterns:[{pattern,shifts,hires,cost_each,cost_total}],
                interpretation}, plot
"""
import sys, json, io, base64
import math
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")


def _fin(x, nd=4):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def _build_patterns(S, length, cyclic):
    """Which shifts each hire covers.

    Cyclic: pattern p starts at shift p and wraps past the end — the night
    shift runs into the next morning, so the schedule is a circle, not a line.
    Non-cyclic: patterns that would run off the end are dropped, which is right
    for a week that genuinely stops on Friday.
    """
    if cyclic:
        starts = range(S)
    else:
        starts = range(max(S - length + 1, 1))
    pats = []
    seen = set()
    for p in starts:
        # Traversal order is what a roster reads like ("Night + Morning"), while the
        # sorted set is what decides whether two patterns are the same block.
        order = []
        for k in range(length):
            s = (p + k) % S
            if s not in order:
                order.append(s)
        key = tuple(sorted(order))
        if key in seen:
            continue
        seen.add(key)
        pats.append((sorted(order), order))
    return pats


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

        raw_len = p.get("shift_length", 1)
        try:
            length = int(raw_len)
        except (TypeError, ValueError):
            raise ValueError("shift_length must be a whole number.")
        if length < 1:
            raise ValueError("shift_length must be at least 1.")
        # A pattern longer than the schedule would cover every shift and make
        # every pattern identical; cap it rather than emit a degenerate model.
        length = min(length, S)
        cyclic = bool(p.get("cyclic", True))

        pats = _build_patterns(S, length, cyclic)
        P = len(pats)
        if P < 1:
            raise ValueError("No usable shift pattern — check shift_length against the shift count.")

        # A hire is paid the per-shift wage for every shift the pattern covers.
        pat_cost = np.array([sum(cost[s] for s in cov) for cov, _ in pats], float)

        # cover[s, p] = 1 when pattern p includes shift s.
        cover = np.zeros((S, P))
        for j, (cov, _) in enumerate(pats):
            for s in cov:
                cover[s, j] = 1.0

        # Demands are whole people, so round the floor up before covering it —
        # otherwise a demand of 7.2 is "met" by 7 workers at the solver's tolerance.
        need = np.array([math.ceil(d - 1e-9) for d in demand], float)

        res = milp(
            c=pat_cost,
            constraints=LinearConstraint(cover, need, np.full(S, np.inf)),
            bounds=Bounds(np.zeros(P), np.full(P, np.inf)),
            integrality=np.ones(P),                 # whole people, not fractions
        )
        if not res.success or res.x is None:
            results = {"status": "infeasible", "unsolved": True,
                       "message": str(res.message or "The covering problem could not be solved.")}
            print(json.dumps({"results": results, "plot": None}))
            return

        y = np.round(res.x).astype(int)
        covered = cover @ y
        total_cost = float(pat_cost @ y)
        n_hires = int(y.sum())
        total_assignments = int(covered.sum())
        # Every hire is paid per shift worked, so the bill splits cleanly by shift.
        per_shift_cost = [float(covered[s] * cost[s]) for s in range(S)]
        surplus = [float(covered[s] - need[s]) for s in range(S)]

        shifts = [{"shift": names[s], "demand": _fin(demand[s], 2),
                   "workers_assigned": int(covered[s]), "cost": _fin(per_shift_cost[s], 2),
                   "surplus": _fin(surplus[s], 2)} for s in range(S)]
        patterns = [{"pattern": " + ".join(names[s] for s in pats[j][1]),
                     "shifts": [names[s] for s in pats[j][1]],
                     "hires": int(y[j]), "cost_each": _fin(pat_cost[j], 2),
                     "cost_total": _fin(float(y[j] * pat_cost[j]), 2)}
                    for j in range(P)]

        # With length 1 the constraints separate and there is nothing to trade off;
        # the page says so rather than claiming an optimisation that did not happen.
        separable = length == 1
        total_surplus = float(sum(surplus))

        plot = None
        try:
            fig, axes = plt.subplots(1, 2 if not separable else 1,
                                     figsize=(max(7, S * 0.9) * (1.7 if not separable else 1.0), 4.4),
                                     dpi=120, squeeze=False)
            ax = axes[0][0]
            x = np.arange(S)
            ax.bar(x, covered, color="#2563eb", label="staffed")
            ax.plot(x, need, "o--", color="#dc2626", label="required")
            ax.set_xticks(x); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel("workers")
            ax.set_title(f"Coverage per shift — total cost {total_cost:,.2f}")
            ax.legend(fontsize=8, frameon=False)
            if not separable:
                ax2 = axes[0][1]
                used = [j for j in range(P) if y[j] > 0]
                if used:
                    ax2.barh(np.arange(len(used)), [y[j] for j in used], color="#0d9488")
                    ax2.set_yticks(np.arange(len(used)))
                    ax2.set_yticklabels([patterns[j]["pattern"] for j in used], fontsize=8)
                    ax2.invert_yaxis()
                ax2.set_xlabel("hires")
                ax2.set_title(f"{n_hires} people on {len(used)} pattern(s)")
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        peak = names[int(np.argmax(covered))] if S else "—"
        if separable:
            interpretation = (
                f"Each worker covers a single shift, so the shifts are independent and the least-cost "
                f"plan staffs every shift at exactly its requirement: {n_hires} hires at a total cost of "
                f"{total_cost:,.2f}. The heaviest shift is {peak}. Set the shift length above 1 if one "
                "worker actually spans consecutive shifts — that is when the plan stops being a copy of "
                "the requirement column."
            )
        else:
            interpretation = (
                f"Each worker covers {length} consecutive shifts, so one hire fills several requirements at "
                f"once and the shifts can no longer be staffed independently. Covering every requirement "
                f"takes {n_hires} people filling {total_assignments} shift slots, at a total cost of "
                f"{total_cost:,.2f}. The heaviest shift is {peak}. "
                + (f"{total_surplus:g} slot(s) are staffed above requirement — that surplus is the price of "
                   "patterns that cannot be split."
                   if total_surplus > 1e-9 else
                   "No shift is overstaffed, so the patterns happen to fit the requirements exactly.")
            )

        results = {"status": "optimal", "unsolved": False, "n_shifts": S,
                   "shift_length": length, "cyclic": cyclic, "separable": separable,
                   "total_cost": _fin(total_cost, 2), "total_workers": n_hires,
                   "n_hires": n_hires, "total_assignments": total_assignments,
                   "total_surplus": _fin(total_surplus, 2),
                   "shifts": shifts, "patterns": patterns,
                   "interpretation": interpretation}
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
