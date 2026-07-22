#!/usr/bin/env python3
"""Comparable Company Analysis — relative valuation from peer multiples.
numpy / pandas.

For each selected valuation multiple, summarises the peer distribution (median,
mean, quartiles) and, given the target's matching fundamental, derives the
implied enterprise / equity / per-share value.

Input (from comparable-company-page.tsx):
    data       : list[dict]     peer companies
    multiples  : [ { col, kind: "equity"|"enterprise", target_fundamental } ]
    net_debt   : float          (optional) to bridge EV->equity for enterprise multiples
    shares     : float          (optional) for per-share value
Output: { results: {multiples[], implied}, plot }
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
            raise ValueError("No peer data provided.")
        df = pd.DataFrame(rows)
        specs = p.get("multiples") or []
        specs = [m for m in specs if m.get("col") in df.columns]
        if not specs:
            raise ValueError("Select at least one valuation multiple column.")
        net_debt = float(p.get("net_debt") or 0.0)
        shares = float(p.get("shares") or 0.0)

        out_multiples = []
        for m in specs:
            col = m["col"]
            kind = (m.get("kind") or "enterprise").lower()
            fund = m.get("target_fundamental")
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            s = s[np.isfinite(s)]
            if len(s) < 2:
                continue
            stats = {
                "median": _fin(float(s.median())), "mean": _fin(float(s.mean())),
                "q1": _fin(float(s.quantile(0.25))), "q3": _fin(float(s.quantile(0.75))),
                "min": _fin(float(s.min())), "max": _fin(float(s.max())),
                "n": int(len(s)),
            }
            implied = None
            if fund is not None and str(fund) != "":
                fv = float(fund)
                med = float(s.median())
                base = med * fv
                if kind == "enterprise":
                    ev = base; equity = ev - net_debt
                else:
                    equity = base; ev = equity + net_debt
                per_share = (equity / shares) if shares > 0 else None
                # range from Q1..Q3
                lo = float(s.quantile(0.25)) * fv
                hi = float(s.quantile(0.75)) * fv
                if kind == "equity":
                    eq_lo, eq_hi = lo, hi
                else:
                    eq_lo, eq_hi = lo - net_debt, hi - net_debt
                implied = {
                    "kind": kind, "target_fundamental": _fin(fv, 4),
                    "enterprise_value": _fin(ev, 2), "equity_value": _fin(equity, 2),
                    "per_share": _fin(per_share, 4) if per_share is not None else None,
                    "equity_low": _fin(eq_lo, 2), "equity_high": _fin(eq_hi, 2),
                    "per_share_low": _fin(eq_lo / shares, 4) if shares > 0 else None,
                    "per_share_high": _fin(eq_hi / shares, 4) if shares > 0 else None,
                }
            out_multiples.append({"multiple": col, "kind": kind, "stats": stats, "implied": implied})

        if not out_multiples:
            raise ValueError("No valid multiples with enough peer observations.")

        # blended implied equity (median across multiples that produced one)
        eqs = [m["implied"]["equity_value"] for m in out_multiples if m["implied"]]
        pss = [m["implied"]["per_share"] for m in out_multiples if m["implied"] and m["implied"]["per_share"] is not None]
        blended = {
            "equity_value": _fin(float(np.median(eqs)), 2) if eqs else None,
            "per_share": _fin(float(np.median(pss)), 4) if pss else None,
            "n_multiples": len(eqs),
        }

        # plot: peer multiple distributions (box) + implied equity range
        plot = None
        try:
            has_impl = any(m["implied"] for m in out_multiples)
            fig, axes = plt.subplots(1, 2 if has_impl else 1, figsize=(12.5 if has_impl else 7, 5), dpi=120)
            ax1 = axes[0] if has_impl else axes
            box_data = []
            labels = []
            for m in out_multiples:
                col = m["multiple"]
                s = pd.to_numeric(df[col], errors="coerce").dropna()
                box_data.append(s.values); labels.append(col)
            ax1.boxplot(box_data, tick_labels=labels, showmeans=True)
            ax1.set_ylabel("Multiple"); ax1.set_title("Peer multiple distributions")
            ax1.tick_params(axis="x", rotation=20)
            if has_impl:
                ax2 = axes[1]
                names = [m["multiple"] for m in out_multiples if m["implied"]]
                meds = [m["implied"]["equity_value"] for m in out_multiples if m["implied"]]
                los = [m["implied"]["equity_low"] for m in out_multiples if m["implied"]]
                his = [m["implied"]["equity_high"] for m in out_multiples if m["implied"]]
                yy = np.arange(len(names))
                for i in range(len(names)):
                    ax2.plot([los[i], his[i]], [i, i], color="#94a3b8", lw=6, solid_capstyle="round")
                    ax2.scatter([meds[i]], [i], color="#2563eb", zorder=5, s=50)
                if blended["equity_value"] is not None:
                    ax2.axvline(blended["equity_value"], color="#dc2626", ls="--", lw=1.2, label="Blended median")
                    ax2.legend(fontsize=8, frameon=False)
                ax2.set_yticks(yy); ax2.set_yticklabels(names)
                ax2.set_xlabel("Implied equity value"); ax2.set_title("Implied equity value by multiple (Q1–Q3)")
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        interpretation = (
            f"Across {len(out_multiples)} valuation multiple(s) benchmarked on the peer group, "
            + (f"the blended implied equity value is {blended['equity_value']:,.0f}"
               + (f" ({blended['per_share']:,.2f} per share)" if blended["per_share"] is not None else "")
               + ". " if blended["equity_value"] is not None else "the peer multiples are summarised above. ")
            + "Relative valuation prices the target off what the market is paying for similar companies today, so it "
            "reflects current sentiment — a useful cross-check on an intrinsic DCF, but one that inherits any "
            "mispricing of the peer group. The spread across multiples is itself informative: wide disagreement "
            "means the choice of metric and peers matters a lot."
        )

        results = {
            "status": "ok", "n_peers": int(len(df)), "net_debt": _fin(net_debt, 2),
            "shares": _fin(shares, 4), "multiples": out_multiples, "blended": blended,
            "interpretation": interpretation,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
