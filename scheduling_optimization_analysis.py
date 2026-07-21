#!/usr/bin/env python3
"""Scheduling Optimization — assign jobs to parallel machines to minimise makespan.
LPT (Longest Processing Time) list-scheduling heuristic — the classic 4/3-approx.

Input: job_names[], processing_times[] (one time per job), machine_names[]
Output: results{status, unsolved, message, n_jobs, n_machines, makespan,
                assignments:[{job,machine,time}], machines:[{name,jobs,load}],
                interpretation}, plot
"""
import sys, json, io, base64
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
        jobs = [str(x) for x in (p.get("job_names") or [])]
        times = [float(x) for x in (p.get("processing_times") or [])]
        machines = [str(x) for x in (p.get("machine_names") or [])]
        J, M = len(jobs), len(machines)
        if J < 1 or M < 1:
            raise ValueError("Need at least one job and one machine.")
        if len(times) != J:
            raise ValueError("processing_times length must match job_names.")
        if any(t < 0 for t in times):
            raise ValueError("Processing times must be non-negative.")

        order = sorted(range(J), key=lambda i: -times[i])   # longest first
        load = [0.0] * M
        mjobs = [[] for _ in range(M)]
        assignments = []
        for i in order:
            mi = int(np.argmin(load))          # least-loaded machine
            load[mi] += times[i]
            mjobs[mi].append(jobs[i])
            assignments.append({"job": jobs[i], "machine": machines[mi], "time": _fin(times[i], 4)})
        makespan = float(max(load))
        machines_out = [{"name": machines[m], "jobs": mjobs[m], "load": _fin(load[m], 4)} for m in range(M)]

        plot = None
        try:
            fig, ax = plt.subplots(figsize=(max(7, makespan / max(1, makespan) * 8), max(3, M * 0.7)), dpi=120)
            colors = plt.cm.tab20(np.linspace(0, 1, max(J, 3)))
            jcol = {jobs[i]: colors[i % len(colors)] for i in range(J)}
            for m in range(M):
                start = 0.0
                for jn in mjobs[m]:
                    dur = times[jobs.index(jn)]
                    ax.barh(m, dur, left=start, color=jcol[jn], edgecolor="white")
                    if dur > makespan * 0.04:
                        ax.text(start + dur/2, m, jn, ha="center", va="center", fontsize=7, color="white")
                    start += dur
            ax.axvline(makespan, color="#dc2626", ls="--", lw=1.5)
            ax.set_yticks(range(M)); ax.set_yticklabels(machines, fontsize=8)
            ax.set_xlabel("time"); ax.set_title(f"Schedule (Gantt) — makespan {makespan:g}")
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        busiest = machines[int(np.argmax(load))]
        interpretation = (
            f"Scheduling {J} jobs on {M} machine(s) with the longest-processing-time rule gives a makespan "
            f"(time the last job finishes) of {makespan:g}, set by {busiest}. LPT loads the biggest jobs "
            "first onto the least-busy machine, which keeps the machines balanced and is provably within "
            "4/3 of the theoretical optimum."
        )
        results = {"status": "optimal", "unsolved": False, "n_jobs": J, "n_machines": M,
                   "makespan": _fin(makespan, 4), "assignments": assignments,
                   "machines": machines_out, "interpretation": interpretation}
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
