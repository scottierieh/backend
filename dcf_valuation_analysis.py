#!/usr/bin/env python3
"""DCF Valuation — discounted cash flow intrinsic value (Gordon terminal value).
numpy only.

Input (from dcf-valuation-page.tsx):
    cash_flows        : number[]  projected free cash flows, year 1..N
    discount_rate     : float     WACC (annual, e.g. 0.10)
    terminal_growth   : float     perpetual growth g (< discount_rate)
    net_debt          : float     (optional) debt - cash, to bridge EV -> equity
    shares_outstanding: float     (optional) for per-share value
    year_labels       : string[]  (optional)
Output: { results: {...}, plot } (PV waterfall + sensitivity).
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


def _valuation(cfs, r, g):
    n = len(cfs)
    disc = [(1 + r) ** -(i + 1) for i in range(n)]
    pv_cfs = [cfs[i] * disc[i] for i in range(n)]
    tv = cfs[-1] * (1 + g) / (r - g)            # Gordon growth terminal value
    pv_tv = tv * disc[-1]
    ev = sum(pv_cfs) + pv_tv
    return pv_cfs, tv, pv_tv, ev


def main():
    try:
        p = json.load(sys.stdin)
        cfs = [float(x) for x in (p.get("cash_flows") or [])]
        n = len(cfs)
        if n < 1:
            raise ValueError("Provide at least one projected cash flow.")
        r = float(p.get("discount_rate"))
        g = float(p.get("terminal_growth") or 0.0)
        if r <= 0:
            raise ValueError("Discount rate must be positive.")
        if g >= r:
            raise ValueError(f"Terminal growth ({g:.1%}) must be below the discount rate ({r:.1%}) — "
                             "otherwise the terminal value is infinite.")
        net_debt = float(p.get("net_debt") or 0.0)
        shares = float(p.get("shares_outstanding") or 0.0)
        labels = p.get("year_labels") or [f"Year {i+1}" for i in range(n)]

        pv_cfs, tv, pv_tv, ev = _valuation(cfs, r, g)
        equity_value = ev - net_debt
        per_share = (equity_value / shares) if shares > 0 else None
        tv_share = 100 * pv_tv / ev if ev != 0 else 0.0

        # sensitivity: EV across discount rate x terminal growth
        rs = [round(r + d, 4) for d in (-0.02, -0.01, 0.0, 0.01, 0.02)]
        gs = [round(g + d, 4) for d in (-0.01, -0.005, 0.0, 0.005, 0.01)]
        sens = []
        for rr in rs:
            row = []
            for gg in gs:
                if gg < rr:
                    _, _, _, ev2 = _valuation(cfs, rr, gg)
                    row.append(_fin(ev2, 2))
                else:
                    row.append(None)
            sens.append(row)

        # plot: PV contribution waterfall-ish + sensitivity heatmap
        plot = None
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5), dpi=120,
                                           gridspec_kw={"width_ratios": [1.2, 1]})
            xs = list(range(n)) + [n]
            heights = pv_cfs + [pv_tv]
            names = [str(l) for l in labels[:n]] + ["Terminal"]
            colors = ["#2563eb"] * n + ["#f59e0b"]
            ax1.bar(xs, heights, color=colors)
            ax1.set_xticks(xs); ax1.set_xticklabels(names, rotation=40, ha="right", fontsize=8)
            ax1.set_ylabel("Present value"); ax1.set_title(f"PV of cash flows + terminal (EV = {ev:,.0f})")
            # sensitivity heatmap
            M = np.array([[np.nan if v is None else v for v in row] for row in sens], float)
            im = ax2.imshow(M, cmap="RdYlGn", aspect="auto")
            ax2.set_xticks(range(len(gs))); ax2.set_xticklabels([f"{x:.1%}" for x in gs], fontsize=7, rotation=30)
            ax2.set_yticks(range(len(rs))); ax2.set_yticklabels([f"{x:.1%}" for x in rs], fontsize=7)
            ax2.set_xlabel("Terminal growth g"); ax2.set_ylabel("Discount rate r")
            ax2.set_title("Enterprise value sensitivity")
            for i in range(len(rs)):
                for j in range(len(gs)):
                    if not np.isnan(M[i, j]):
                        ax2.text(j, i, f"{M[i,j]:,.0f}", ha="center", va="center", fontsize=6)
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        interpretation = (
            f"Discounting {n} year(s) of projected cash flows plus a Gordon terminal value at a {r:.1%} "
            f"discount rate gives an enterprise value of {ev:,.2f}. "
            + (f"After netting {net_debt:,.0f} of net debt, equity is worth {equity_value:,.2f}" if net_debt else f"Equity value is {equity_value:,.2f}")
            + (f", or {per_share:,.2f} per share. " if per_share is not None else ". ")
            + f"The terminal value accounts for {tv_share:.0f}% of the total — a large share means the valuation "
            "leans heavily on assumptions about the distant future (the growth rate and discount rate), which is "
            "why the sensitivity grid matters as much as the point estimate."
        )

        results = {
            "status": "ok", "n_years": n, "discount_rate": _fin(r, 6), "terminal_growth": _fin(g, 6),
            "net_debt": _fin(net_debt, 2), "shares_outstanding": _fin(shares, 4),
            "pv_cash_flows": [{"year": str(labels[i]), "cash_flow": _fin(cfs[i], 2), "pv": _fin(pv_cfs[i], 2)} for i in range(n)],
            "terminal_value": _fin(tv, 2), "pv_terminal": _fin(pv_tv, 2), "terminal_pct": _fin(tv_share, 1),
            "enterprise_value": _fin(ev, 2), "equity_value": _fin(equity_value, 2),
            "per_share_value": _fin(per_share, 4) if per_share is not None else None,
            "sensitivity": {"discount_rates": rs, "growth_rates": gs, "ev": sens},
            "interpretation": interpretation,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
