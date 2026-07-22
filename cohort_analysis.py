#!/usr/bin/env python3
"""Cohort Analysis — acquisition-cohort retention matrix. pandas.

Groups customers by the period of their first activity (their cohort), then
tracks what fraction remain active in each subsequent period, producing the
classic retention triangle and the average retention curve.

Input (from cohort-analysis-page.tsx):
    data       : list[dict]
    id_col     : str   customer id
    date_col   : str   activity/order date
    freq       : "M"|"W"   cohort granularity (default M = monthly)
    max_periods: int   (default 12) periods to track
Output: { results: {cohorts[], retention_curve, matrix}, plot }
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


def main():
    try:
        p = json.load(sys.stdin)
        rows = p.get("data") or []
        if not rows:
            raise ValueError("No data provided.")
        df = pd.DataFrame(rows)
        id_col = p.get("id_col"); date_col = p.get("date_col")
        freq = (p.get("freq") or "M").upper()
        max_periods = int(p.get("max_periods") or 12)
        if not id_col or id_col not in df.columns:
            raise ValueError("Select the customer id column.")
        if not date_col or date_col not in df.columns:
            raise ValueError("Select the date column.")
        freq = "W" if freq == "W" else "M"
        max_periods = max(2, min(max_periods, 24))

        df = df[[id_col, date_col]].copy()
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])
        if len(df) < 10:
            raise ValueError("Need at least 10 dated events.")

        period = df[date_col].dt.to_period(freq)
        df["_period"] = period
        # cohort = first period per customer
        first = df.groupby(id_col)["_period"].transform("min")
        df["_cohort"] = first
        # period index offset from cohort
        if freq == "M":
            df["_idx"] = (df["_period"].dt.year - df["_cohort"].dt.year) * 12 + (df["_period"].dt.month - df["_cohort"].dt.month)
        else:
            df["_idx"] = (df["_period"] - df["_cohort"]).apply(lambda x: x.n)
        df = df[(df["_idx"] >= 0) & (df["_idx"] <= max_periods)]

        # counts of distinct customers per (cohort, idx)
        counts = df.groupby(["_cohort", "_idx"])[id_col].nunique().unstack(fill_value=0)
        cohort_sizes = counts[0] if 0 in counts.columns else counts.iloc[:, 0]
        retention = counts.divide(cohort_sizes, axis=0)

        cohort_labels = [str(c) for c in retention.index]
        idxs = sorted([int(c) for c in retention.columns])
        matrix = []
        for ci, coh in enumerate(retention.index):
            row = {"cohort": str(coh), "size": int(cohort_sizes.iloc[ci]),
                   "retention": [_fin(float(retention.loc[coh, j]) if j in retention.columns else np.nan, 4) for j in idxs]}
            matrix.append(row)

        # average retention curve across cohorts (weighted by size, ignoring NaN)
        avg_curve = []
        for j in idxs:
            if j in retention.columns:
                vals = retention[j]
                w = cohort_sizes[vals.index]
                mask = vals.notna() & (counts[j] if j in counts.columns else vals).notna()
                v = float(np.average(vals[mask], weights=w[mask])) if mask.any() else np.nan
            else:
                v = np.nan
            avg_curve.append({"period": int(j), "retention": _fin(v, 4)})

        total_customers = int(df[id_col].nunique())
        n_cohorts = len(retention.index)
        # headline retention at period 1 and 3
        ret1 = avg_curve[1]["retention"] if len(avg_curve) > 1 else None
        ret3 = avg_curve[3]["retention"] if len(avg_curve) > 3 else None

        plot = None
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.2), dpi=115,
                                           gridspec_kw={"width_ratios": [1.5, 1]})
            M = retention.reindex(columns=idxs).values.astype(float)
            im = ax1.imshow(M * 100, cmap="Blues", aspect="auto", vmin=0, vmax=100)
            ax1.set_xticks(range(len(idxs))); ax1.set_xticklabels(idxs, fontsize=7)
            ax1.set_yticks(range(len(cohort_labels))); ax1.set_yticklabels(cohort_labels, fontsize=7)
            ax1.set_xlabel(f"Periods since acquisition ({'month' if freq=='M' else 'week'})")
            ax1.set_ylabel("Cohort"); ax1.set_title("Retention by cohort (%)")
            for i in range(M.shape[0]):
                for j in range(M.shape[1]):
                    if np.isfinite(M[i, j]):
                        ax1.text(j, i, f"{M[i,j]*100:.0f}", ha="center", va="center", fontsize=6,
                                 color="white" if M[i, j] > 0.5 else "#111827")
            fig.colorbar(im, ax=ax1, label="Retention %")
            cx = [c["period"] for c in avg_curve]; cy = [(c["retention"] or 0) * 100 for c in avg_curve]
            ax2.plot(cx, cy, "o-", color="#2563eb", lw=2)
            ax2.set_xlabel("Periods since acquisition"); ax2.set_ylabel("Avg retention (%)")
            ax2.set_title("Average retention curve"); ax2.grid(alpha=0.2); ax2.set_ylim(0, 105)
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        unit = "month" if freq == "M" else "week"
        interpretation = (
            f"Grouping {total_customers} customers into {n_cohorts} acquisition cohorts by {unit}, "
            + (f"an average of {ret1:.0%} return in the first {unit} after acquisition"
               + (f" and {ret3:.0%} are still active by {unit} 3" if ret3 is not None else "")
               + ". " if ret1 is not None else "")
            + "The retention triangle shows how each cohort decays over time; a curve that flattens indicates a loyal "
            "core, while one that keeps falling signals a leaky funnel. Comparing rows reveals whether newer cohorts "
            "retain better or worse than older ones — the clearest signal of whether product and onboarding changes "
            "are working."
        )

        results = {
            "status": "ok", "freq": freq, "n_customers": total_customers, "n_cohorts": n_cohorts,
            "max_periods": int(max(idxs) if idxs else 0),
            "avg_retention_1": _fin(ret1, 4), "avg_retention_3": _fin(ret3, 4),
            "periods": idxs, "cohorts": matrix, "retention_curve": avg_curve,
            "interpretation": interpretation,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
