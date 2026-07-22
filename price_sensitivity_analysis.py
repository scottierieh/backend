#!/usr/bin/env python3
"""Price Sensitivity — Van Westendorp Price Sensitivity Meter. numpy / pandas.

From four price questions per respondent (too cheap / cheap / expensive / too
expensive), builds the cumulative price curves and finds the four classic
intersection points: OPP, IPP, PMC and PME, plus the range of acceptable prices.

Input (from price-sensitivity-page.tsx):
    data              : list[dict]
    too_cheap_col     : str
    cheap_col         : str
    expensive_col     : str
    too_expensive_col : str
Output: { results: {opp, ipp, pmc, pme, curves}, plot }
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


def _intersect(grid, a, b):
    """First price where curves a and b cross (linear interp)."""
    d = np.asarray(a) - np.asarray(b)
    for i in range(1, len(grid)):
        if d[i - 1] == 0:
            return float(grid[i - 1]), float(a[i - 1])
        if d[i - 1] * d[i] < 0:
            t = d[i - 1] / (d[i - 1] - d[i])
            price = grid[i - 1] + t * (grid[i] - grid[i - 1])
            val = a[i - 1] + t * (a[i] - a[i - 1])
            return float(price), float(val)
    return None, None


def main():
    try:
        p = json.load(sys.stdin)
        rows = p.get("data") or []
        if not rows:
            raise ValueError("No data provided.")
        df = pd.DataFrame(rows)
        tc = p.get("too_cheap_col"); ch = p.get("cheap_col")
        ex = p.get("expensive_col"); te = p.get("too_expensive_col")
        for name, c in [("too cheap", tc), ("cheap", ch), ("expensive", ex), ("too expensive", te)]:
            if not c or c not in df.columns:
                raise ValueError(f"Select the '{name}' price column.")

        d = df[[tc, ch, ex, te]].apply(pd.to_numeric, errors="coerce").dropna().reset_index(drop=True)
        n = len(d)
        if n < 10:
            raise ValueError("Need at least 10 complete respondents.")
        tcv, chv, exv, tev = d[tc].values, d[ch].values, d[ex].values, d[te].values

        lo = float(np.min([tcv.min(), chv.min(), exv.min(), tev.min()]))
        hi = float(np.max([tcv.max(), chv.max(), exv.max(), tev.max()]))
        grid = np.linspace(lo, hi, 200)

        # cumulative curves
        f_too_cheap = np.array([np.mean(tcv >= g) for g in grid])   # descending
        f_cheap     = np.array([np.mean(chv >= g) for g in grid])   # descending
        f_expensive = np.array([np.mean(exv <= g) for g in grid])   # ascending
        f_too_exp   = np.array([np.mean(tev <= g) for g in grid])   # ascending

        opp_p, opp_v = _intersect(grid, f_too_cheap, f_too_exp)     # optimal price point
        ipp_p, ipp_v = _intersect(grid, f_cheap, f_expensive)       # indifference price point
        pmc_p, pmc_v = _intersect(grid, f_too_cheap, f_expensive)   # point of marginal cheapness
        pme_p, pme_v = _intersect(grid, f_cheap, f_too_exp)         # point of marginal expensiveness

        acc_low = pmc_p; acc_high = pme_p
        median_ex = float(np.median(exv))

        curves = [{"price": _fin(grid[i], 2),
                   "too_cheap": _fin(float(f_too_cheap[i]), 4),
                   "cheap": _fin(float(f_cheap[i]), 4),
                   "expensive": _fin(float(f_expensive[i]), 4),
                   "too_expensive": _fin(float(f_too_exp[i]), 4)} for i in range(0, len(grid), 4)]

        plot = None
        try:
            fig, ax = plt.subplots(figsize=(10.5, 5.6), dpi=120)
            ax.plot(grid, f_too_cheap * 100, color="#16a34a", lw=1.8, label="Too cheap")
            ax.plot(grid, f_cheap * 100, color="#84cc16", lw=1.4, ls="--", label="Cheap / bargain")
            ax.plot(grid, f_expensive * 100, color="#f59e0b", lw=1.4, ls="--", label="Expensive")
            ax.plot(grid, f_too_exp * 100, color="#dc2626", lw=1.8, label="Too expensive")
            for pt, name, col in [(opp_p, "OPP", "#2563eb"), (ipp_p, "IPP", "#7c3aed"),
                                  (pmc_p, "PMC", "#0891b2"), (pme_p, "PME", "#be185d")]:
                if pt is not None:
                    ax.axvline(pt, color=col, ls=":", lw=1)
                    ax.text(pt, 102, f"{name}\n{pt:,.0f}", color=col, fontsize=7, ha="center")
            if acc_low is not None and acc_high is not None:
                ax.axvspan(acc_low, acc_high, color="#2563eb", alpha=0.06)
            ax.set_xlabel("Price"); ax.set_ylabel("Cumulative % of respondents")
            ax.set_title("Van Westendorp Price Sensitivity Meter")
            ax.legend(fontsize=8, frameon=False, loc="center right"); ax.grid(alpha=0.2); ax.set_ylim(0, 108)
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        interpretation = (
            f"From {n} respondents, the Van Westendorp analysis identifies an Optimal Price Point (OPP) of "
            f"{opp_p:,.0f} — where the proportions rating the product too cheap and too expensive are equal, the price "
            f"that minimises buyer resistance. "
            + (f"The range of acceptable prices runs from {acc_low:,.0f} (Point of Marginal Cheapness) to "
               f"{acc_high:,.0f} (Point of Marginal Expensiveness); pricing inside this band keeps most of the market "
               "comfortable. " if acc_low is not None and acc_high is not None else "")
            + f"The Indifference Price Point (IPP) of {ipp_p:,.0f} reflects the price the typical respondent considers "
            "normal. Below the lower bound, buyers suspect poor quality; above the upper bound, they balk at the cost."
        )

        results = {
            "status": "ok", "n_respondents": n,
            "opp": _fin(opp_p, 2), "ipp": _fin(ipp_p, 2), "pmc": _fin(pmc_p, 2), "pme": _fin(pme_p, 2),
            "acceptable_low": _fin(acc_low, 2), "acceptable_high": _fin(acc_high, 2),
            "median_expensive": _fin(median_ex, 2),
            "curves": curves,
            "interpretation": interpretation,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
