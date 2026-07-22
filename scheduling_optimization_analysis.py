#!/usr/bin/env python3
"""Scheduling Optimization — minimise makespan on parallel machines (LPT).

Assigns jobs to identical parallel machines to minimise the makespan (the
finish time of the last machine). Uses the Longest-Processing-Time-first
heuristic, which is provably within 4/3 of optimal, and compares against the
theoretical lower bound.

Input (from scheduling-optimization-page.tsx):
    job_names        : string[]
    processing_times : number[]
    machine_names    : string[]
Output: { results: {assignments[], machines[], makespan, ...}, plot }
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
        jobs = p.get("job_names") or []
        times = [float(x) for x in (p.get("processing_times") or [])]
        machines = p.get("machine_names") or []
        if not jobs or not machines:
            raise ValueError("Provide job names, processing times and machine names.")
        if len(times) != len(jobs):
            raise ValueError("processing_times must have one entry per job.")
        if any(t < 0 for t in times):
            raise ValueError("Processing times must be non-negative.")

        M = len(machines)
        # LPT: sort jobs by descending time, assign each to the least-loaded machine
        order = sorted(range(len(jobs)), key=lambda i: -times[i])
        loads = [0.0] * M
        counts = [0] * M
        assignments = []
        for i in order:
            k = int(np.argmin(loads))
            loads[k] += times[i]; counts[k] += 1
            assignments.append({"job": jobs[i], "machine": machines[k], "processing_time": _fin(times[i], 4)})
        # keep assignments in original job order for readability
        assignments.sort(key=lambda a: jobs.index(a["job"]))

        makespan = float(max(loads)) if loads else 0.0
        total = sum(times)
        lower_bound = max(total / M, max(times) if times else 0.0)
        machines_out = [{"machine": machines[k], "load": _fin(loads[k], 4), "n_jobs": counts[k]} for k in range(M)]

        plot = None
        try:
            fig, ax = plt.subplots(figsize=(max(7, M * 1.1 + 2), 5), dpi=120)
            # stacked Gantt-style bars per machine
            colors = plt.cm.tab20(np.linspace(0, 1, max(len(jobs), 1)))
            job_color = {jobs[i]: colors[i % len(colors)] for i in range(len(jobs))}
            base = [0.0] * M
            for a in sorted(assignments, key=lambda a: -a["processing_time"]):
                k = machines.index(a["machine"])
                ax.barh(k, a["processing_time"], left=base[k], color=job_color[a["job"]], edgecolor="white")
                if a["processing_time"] > makespan * 0.06:
                    ax.text(base[k] + a["processing_time"] / 2, k, a["job"], ha="center", va="center", fontsize=7)
                base[k] += a["processing_time"]
            ax.axvline(makespan, color="#dc2626", ls="--", lw=1.4, label=f"Makespan = {makespan:g}")
            ax.axvline(lower_bound, color="#16a34a", ls=":", lw=1.4, label=f"Lower bound = {lower_bound:.2f}")
            ax.set_yticks(range(M)); ax.set_yticklabels(machines, fontsize=8)
            ax.set_xlabel("Time"); ax.set_title("Machine schedule (LPT)")
            ax.legend(fontsize=8, frameon=False); ax.grid(axis="x", alpha=0.2)
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        gap = (makespan - lower_bound) / lower_bound * 100 if lower_bound > 0 else 0.0
        interpretation = (
            f"Scheduling {len(jobs)} jobs across {M} machine(s) with the longest-processing-time-first rule gives a "
            f"makespan of {makespan:g} — the time the last machine finishes. The theoretical lower bound is "
            f"{lower_bound:.2f} (the larger of total work / machines and the single longest job), so this schedule is "
            f"within {gap:.1f}% of the best possible. Machine loads are {', '.join(f'{machines[k]}={loads[k]:g}' for k in range(M))}; "
            "the flatter these are, the better the balance. LPT works by placing the biggest jobs first, leaving small "
            "jobs to even out the load."
        )

        results = {
            "status": "ok", "unsolved": False,
            "n_jobs": len(jobs), "n_machines": M, "makespan": _fin(makespan, 4),
            "lower_bound": _fin(lower_bound, 4),
            "assignments": assignments, "machines": machines_out,
            "interpretation": interpretation,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
