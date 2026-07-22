#!/usr/bin/env python3
"""Marketing Mix Modeling (MMM) — decompose sales into channel contributions,
with adstock (carry-over), saturation (diminishing returns), ROI/ROAS, response
curves and budget optimization. numpy + scipy (NNLS + SLSQP).

Model:  sales_t = base + Σ_i β_i · sat_i( adstock_i(spend_i) )   (β_i ≥ 0, NNLS)
    adstock_i(x)_t = x_t + decay_i · adstock_i(x)_{t-1}   (geometric carry-over)
    sat_i(a)       = a / (a + K_i)                          (Hill, half-sat K_i)

Input (from mmm-page.tsx):
    data       : list[dict]
    target     : str            sales / revenue column
    channels   : string[]       media spend columns
    (optional) decay_grid, adstock per channel — else auto-selected
Output: { results: {channels[], response_curves, budget_opt, ...}, plot }
"""
import sys, json, io, base64
import numpy as np
import pandas as pd
from scipy.optimize import nnls, minimize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _fin(x, nd=6):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def _adstock(x, decay):
    out = np.zeros_like(x, dtype=float)
    acc = 0.0
    for t in range(len(x)):
        acc = x[t] + decay * acc
        out[t] = acc
    return out


def main():
    try:
        p = json.load(sys.stdin)
        rows = p.get("data") or []
        target = p.get("target") or p.get("targetVar")
        channels = p.get("channels") or p.get("features") or []
        if not rows or not target or len(channels) < 1:
            raise ValueError("Provide data, a target (sales) column and at least one channel.")
        df = pd.DataFrame(rows)
        if target not in df.columns:
            raise ValueError(f"Target column '{target}' not found.")
        channels = [c for c in channels if c in df.columns]
        if len(channels) < 1:
            raise ValueError("None of the channel columns were found in the data.")

        y = pd.to_numeric(df[target], errors="coerce").to_numpy(dtype=float)
        X_raw = np.column_stack([pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy(dtype=float) for c in channels])
        mask = np.isfinite(y)
        y = y[mask]; X_raw = X_raw[mask]
        T, n = X_raw.shape
        if T < len(channels) + 3:
            raise ValueError(f"Need at least {len(channels)+3} periods for {len(channels)} channels.")

        decay_grid = [0.0, 0.2, 0.4, 0.6, 0.8]
        # half-saturation per channel = mean of positive spend (robust scale)
        Ks = np.array([max(np.mean(X_raw[:, i][X_raw[:, i] > 0]) if np.any(X_raw[:, i] > 0) else 1.0, 1e-6)
                       for i in range(n)])

        def transform(decays):
            cols = []
            for i in range(n):
                a = _adstock(X_raw[:, i], decays[i])
                cols.append(a / (a + Ks[i]))
            return np.column_stack(cols)

        # choose per-channel decay by a light coordinate search maximising R²
        decays = [0.4] * n
        best = None
        for _ in range(2):
            for i in range(n):
                best_d, best_r2 = decays[i], -np.inf
                for d in decay_grid:
                    trial = decays.copy(); trial[i] = d
                    Xs = transform(trial)
                    A = np.column_stack([np.ones(T), Xs])
                    beta, _ = nnls(A, y)
                    pred = A @ beta
                    ss_res = np.sum((y - pred) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
                    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
                    if r2 > best_r2:
                        best_r2, best_d = r2, d
                decays[i] = best_d
        Xs = transform(decays)
        A = np.column_stack([np.ones(T), Xs])
        beta, _ = nnls(A, y)
        base, coefs = float(beta[0]), beta[1:]
        pred = A @ beta
        ss_res = np.sum((y - pred) ** 2); ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        # contribution decomposition (per period, summed)
        contrib = Xs * coefs                      # (T, n)
        total_contrib = contrib.sum(axis=0)       # per channel
        base_contrib = base * T
        total_sales = float(y.sum())
        total_spend_ch = X_raw.sum(axis=0)
        total_spend = float(total_spend_ch.sum())

        ch_rows = []
        for i, c in enumerate(channels):
            spend_i = float(total_spend_ch[i])
            inc_rev = float(total_contrib[i])
            roi = (inc_rev - spend_i) / spend_i if spend_i > 0 else None   # profit per $ ((rev-spend)/spend)
            roas = inc_rev / spend_i if spend_i > 0 else None              # revenue per $
            ch_rows.append({
                "channel": c, "spend": _fin(spend_i, 2),
                "contribution": _fin(inc_rev, 2),
                "contribution_pct": _fin(inc_rev / total_sales, 4) if total_sales else None,
                "incremental_revenue": _fin(inc_rev, 2),
                "roi": _fin(roi, 4), "roas": _fin(roas, 4),
                "decay": _fin(decays[i], 3), "beta": _fin(coefs[i], 4),
                "half_saturation": _fin(Ks[i], 2),
            })

        incremental_total = float(total_contrib.sum())
        overall_roas = incremental_total / total_spend if total_spend > 0 else None
        overall_roi = (incremental_total - total_spend) / total_spend if total_spend > 0 else None

        # ---- response curves (per channel, steady-state weekly spend) ----
        # steady-state adstock of constant spend s = s/(1-decay); then saturate
        response_curves = []
        smax_all = X_raw.max(axis=0)
        for i, c in enumerate(channels):
            smax = max(smax_all[i] * 1.8, Ks[i] * 3)
            grid = np.linspace(0, smax, 40)
            ss = grid / (1 - decays[i]) if decays[i] < 1 else grid
            resp = coefs[i] * (ss / (ss + Ks[i]))
            response_curves.append({"channel": c,
                                    "spend": [_fin(float(g), 2) for g in grid],
                                    "response": [_fin(float(rv), 4) for rv in resp]})

        # ---- budget optimization: allocate current total media budget to
        #      maximise predicted steady-state contribution ----
        budget = float(total_spend)
        avg_spend = total_spend_ch / T          # current avg weekly spend per channel
        def neg_resp(b):
            tot = 0.0
            for i in range(n):
                ss = b[i] / (1 - decays[i]) if decays[i] < 1 else b[i]
                tot += coefs[i] * (ss / (ss + Ks[i]))
            return -tot
        cur_weekly = avg_spend.copy()
        weekly_budget = float(cur_weekly.sum())
        cons = [{"type": "eq", "fun": lambda b: b.sum() - weekly_budget}]
        bounds = [(0, weekly_budget)] * n
        x0 = np.full(n, weekly_budget / n)
        opt = minimize(neg_resp, x0, bounds=bounds, constraints=cons, method="SLSQP",
                       options={"maxiter": 300, "ftol": 1e-9})
        opt_weekly = opt.x if opt.success else cur_weekly
        cur_resp = -neg_resp(cur_weekly)
        opt_resp = -neg_resp(opt_weekly)
        budget_gain = (opt_resp / cur_resp - 1) if cur_resp > 0 else 0.0
        for i in range(n):
            ch_rows[i]["current_weekly"] = _fin(float(cur_weekly[i]), 2)
            ch_rows[i]["optimal_weekly"] = _fin(float(opt_weekly[i]), 2)
            ch_rows[i]["optimal_budget"] = _fin(float(opt_weekly[i] * T), 2)
            ch_rows[i]["budget_change_pct"] = _fin((opt_weekly[i] / cur_weekly[i] - 1) if cur_weekly[i] > 0 else None, 4)

        # ---- plot: 2x2 (actual vs predicted, contribution, response curves, budget) ----
        plot = None
        try:
            fig, axes = plt.subplots(2, 2, figsize=(14.5, 10), dpi=110)
            # actual vs predicted over time
            ax = axes[0, 0]
            tt = np.arange(T)
            ax.plot(tt, y, color="#111827", lw=1.4, label="Actual")
            ax.plot(tt, pred, color="#2563eb", lw=1.4, ls="--", label="Predicted")
            ax.fill_between(tt, base, base + contrib.sum(axis=1), color="#93c5fd", alpha=0.3, label="Media-driven")
            ax.axhline(base, color="#16a34a", ls=":", lw=1, label="Base (non-media)")
            ax.set_title(f"1. Actual vs predicted sales (R²={r2:.2f})"); ax.set_xlabel("Period")
            ax.legend(fontsize=7, frameon=False)
            # channel contribution (stacked share)
            ax = axes[0, 1]
            labels = ["Base"] + channels
            vals = [base_contrib] + list(total_contrib)
            colors = ["#9ca3af"] + list(plt.cm.tab10(np.linspace(0, 1, n)))
            ax.bar(labels, vals, color=colors)
            for i, v in enumerate(vals):
                ax.text(i, v, f"{v/total_sales*100:.0f}%" if total_sales else "", ha="center", va="bottom", fontsize=7)
            ax.set_title("2. Contribution to sales"); ax.set_ylabel("Contribution"); ax.tick_params(axis="x", rotation=25)
            # response curves
            ax = axes[1, 0]
            for i, rc in enumerate(response_curves):
                ax.plot(rc["spend"], rc["response"], lw=2, color=plt.cm.tab10(i / max(n, 1)), label=rc["channel"])
                cur = cur_weekly[i]
                ax.scatter([cur], [coefs[i] * ((cur / (1 - decays[i])) / (cur / (1 - decays[i]) + Ks[i]))],
                           color=plt.cm.tab10(i / max(n, 1)), s=35, zorder=5)
            ax.set_title("3. Response curves (dots = current spend)"); ax.set_xlabel("Weekly spend")
            ax.set_ylabel("Incremental response"); ax.legend(fontsize=7, frameon=False); ax.grid(alpha=0.2)
            # budget optimization: current vs optimal allocation
            ax = axes[1, 1]
            xpos = np.arange(n); w = 0.38
            ax.bar(xpos - w / 2, cur_weekly, w, color="#94a3b8", label="Current")
            ax.bar(xpos + w / 2, opt_weekly, w, color="#2563eb", label="Optimal")
            ax.set_xticks(xpos); ax.set_xticklabels(channels, rotation=25, fontsize=8)
            ax.set_ylabel("Weekly spend")
            ax.set_title(f"4. Budget reallocation (+{budget_gain*100:.1f}% response, same spend)")
            ax.legend(fontsize=8, frameon=False); ax.grid(axis="y", alpha=0.2)
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        ch_sorted = sorted(ch_rows, key=lambda r_: -(r_["roas"] or 0))
        top = ch_sorted[0] if ch_sorted else None
        movers = sorted(ch_rows, key=lambda r_: -((r_["budget_change_pct"] or 0)))
        interpretation = (
            f"The model explains {r2:.0%} of sales variation. Of total sales {total_sales:,.0f}, about "
            f"{base_contrib/total_sales*100:.0f}% is base (non-media) demand and {incremental_total/total_sales*100:.0f}% "
            f"is driven by the {n} media channels, on {total_spend:,.0f} of spend (overall ROAS {overall_roas:.2f}). "
            + (f"{top['channel']} is the most efficient channel at a ROAS of {top['roas']:.2f}. " if top else "")
            + f"Because each channel saturates (diminishing returns), reallocating the same budget toward under-saturated "
              f"channels is predicted to lift media-driven sales by {budget_gain*100:.1f}% — "
            + (", ".join(f"{m['channel']} {'+' if (m['budget_change_pct'] or 0)>=0 else ''}{(m['budget_change_pct'] or 0)*100:.0f}%"
                         for m in movers[:3]))
            + ". The response curves show where extra spend still pays off and where a channel is already saturated."
        )

        results = {
            "status": "ok", "target": target, "n_periods": T, "n_channels": n,
            "r_squared": _fin(r2, 4),
            "total_sales": _fin(total_sales, 2), "total_spend": _fin(total_spend, 2),
            "base_contribution": _fin(base_contrib, 2),
            "base_contribution_pct": _fin(base_contrib / total_sales, 4) if total_sales else None,
            "incremental_revenue": _fin(incremental_total, 2),
            "overall_roas": _fin(overall_roas, 4), "overall_roi": _fin(overall_roi, 4),
            "budget_gain_pct": _fin(budget_gain, 4),
            "channels": ch_rows, "response_curves": response_curves,
            "actual": [_fin(float(v), 2) for v in y], "predicted": [_fin(float(v), 2) for v in pred],
            "interpretation": interpretation,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
