#!/usr/bin/env python3
"""Funnel Analysis — step-by-step conversion and drop-off. pandas.

Takes an ordered list of funnel stages and the count of users reaching each
stage, then computes step conversion rates, cumulative (overall) conversion
from the top of the funnel, and the drop-off at every step so the biggest
leak is obvious.

Two input shapes are accepted:
  1. Aggregated: `steps` = [{"name": str, "count": number}, ...]  (already counted)
  2. Event data: `data` rows + `id_col` + `stage_col` + `stage_order` list;
     the script counts distinct ids observed at each stage.

Output: { results: {steps[], overall_conversion, biggest_drop}, plot }
"""
import sys, json, io, base64
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _fin(x, nd=4):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def _counts_from_events(p):
    rows = p.get("data") or []
    if not rows:
        raise ValueError("No data provided.")
    df = pd.DataFrame(rows)
    id_col = p.get("id_col"); stage_col = p.get("stage_col")
    order = p.get("stage_order") or []
    if not id_col or id_col not in df.columns:
        raise ValueError("Select the user id column.")
    if not stage_col or stage_col not in df.columns:
        raise ValueError("Select the stage column.")
    if not order:
        # infer order from first appearance
        order = list(pd.unique(df[stage_col].astype(str)))
    names, counts = [], []
    for st in order:
        n = int(df[df[stage_col].astype(str) == str(st)][id_col].nunique())
        names.append(str(st)); counts.append(n)
    return names, counts


def main():
    try:
        p = json.load(sys.stdin)
        if p.get("steps"):
            steps_in = p["steps"]
            names = [str(s.get("name", f"Step {i+1}")) for i, s in enumerate(steps_in)]
            counts = [float(s.get("count", 0)) for s in steps_in]
        else:
            names, counts = _counts_from_events(p)

        if len(names) < 2:
            raise ValueError("A funnel needs at least 2 stages.")
        if counts[0] <= 0:
            raise ValueError("The first stage must have a positive count.")

        top = float(counts[0])
        steps = []
        biggest_drop = {"from": None, "to": None, "drop_rate": 0.0}
        for i, (nm, c) in enumerate(zip(names, counts)):
            c = float(c)
            prev = float(counts[i - 1]) if i > 0 else c
            step_conv = (c / prev) if prev > 0 else None
            overall = (c / top) if top > 0 else None
            drop_rate = (1 - step_conv) if (i > 0 and step_conv is not None) else 0.0
            dropped = int(round(prev - c)) if i > 0 else 0
            steps.append({
                "stage": nm, "count": int(round(c)),
                "step_conversion": _fin(step_conv, 4),
                "overall_conversion": _fin(overall, 4),
                "drop_rate": _fin(drop_rate, 4),
                "dropped": dropped,
            })
            if i > 0 and drop_rate is not None and drop_rate > biggest_drop["drop_rate"]:
                biggest_drop = {"from": names[i - 1], "to": nm, "drop_rate": _fin(drop_rate, 4)}

        overall_conversion = _fin(counts[-1] / top, 4)

        plot = None
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=115,
                                           gridspec_kw={"width_ratios": [1.2, 1]})
            y = np.arange(len(names))[::-1]
            widths = [s["count"] for s in steps]
            maxw = max(widths) or 1
            for yi, s in zip(y, steps):
                w = s["count"]
                left = (maxw - w) / 2
                ax1.barh(yi, w, left=left, color="#3b82f6", height=0.62)
                ax1.text(maxw / 2, yi, f"{s['stage']}\n{s['count']:,} ({(s['overall_conversion'] or 0)*100:.1f}%)",
                         ha="center", va="center", fontsize=8, color="white", fontweight="bold")
            ax1.set_xlim(0, maxw); ax1.axis("off"); ax1.set_title("Conversion funnel")

            xs = range(len(names))
            sc = [(s["step_conversion"] or 0) * 100 for s in steps]
            ax2.bar(xs, sc, color="#10b981")
            ax2.set_xticks(list(xs)); ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=7)
            ax2.set_ylabel("Step conversion (%)"); ax2.set_ylim(0, 105)
            ax2.set_title("Step-to-step conversion"); ax2.grid(axis="y", alpha=0.2)
            for xi, v in zip(xs, sc):
                if xi > 0:
                    ax2.text(xi, v + 2, f"{v:.0f}%", ha="center", fontsize=7)
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        bd = biggest_drop
        interpretation = (
            f"Of {int(round(top)):,} users entering the funnel, {int(round(counts[-1])):,} reach "
            f"'{names[-1]}' — an overall conversion of {(overall_conversion or 0):.1%}. "
            + (f"The largest single leak is between '{bd['from']}' and '{bd['to']}', where "
               f"{(bd['drop_rate'] or 0):.1%} of users drop off; that step is the highest-leverage place to "
               "focus optimisation. " if bd["from"] else "")
            + "Step conversion isolates each transition so a healthy overall rate can't hide one broken stage, "
            "while the cumulative rate shows how much total volume survives to the goal."
        )

        results = {
            "status": "ok",
            "n_stages": len(names),
            "top_of_funnel": int(round(top)),
            "final_count": int(round(counts[-1])),
            "overall_conversion": overall_conversion,
            "biggest_drop": biggest_drop,
            "steps": steps,
            "interpretation": interpretation,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
