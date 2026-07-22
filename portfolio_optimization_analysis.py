#!/usr/bin/env python3
"""Portfolio Optimization — mean-variance efficient frontier & optimal weights.
PyPortfolioOpt (pypfopt).

Input (from portfolio-optimization-page.tsx):
    data              : list[dict]
    asset_cols        : string[]   price or return columns (>= 2 assets)
    is_returns        : bool
    return_type       : "simple"|"log"
    periods_per_year  : int   (default 252)
    rf_annual         : float (default 0.02)
    objective         : "max_sharpe" | "min_volatility" | "efficient_return" | "efficient_risk"
    target_return     : float (annual, for efficient_return)
    target_volatility : float (annual, for efficient_risk)
Output: { results: {...}, plot } (efficient frontier + chosen portfolio + assets).
"""
import sys, json, io, base64
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pypfopt import EfficientFrontier


def _fin(x, nd=6):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def _returns_df(df, cols, is_returns, rtype):
    out = {}
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if is_returns:
            out[c] = s
        elif rtype == "log":
            if (s <= 0).any():
                raise ValueError(f"Log returns require positive prices ({c}).")
            out[c] = np.log(s / s.shift(1))
        else:
            out[c] = s / s.shift(1) - 1.0
    return pd.DataFrame(out).dropna().reset_index(drop=True)


def main():
    try:
        p = json.load(sys.stdin)
        rows = p.get("data") or []
        if not rows:
            raise ValueError("No data provided.")
        df = pd.DataFrame(rows)
        cols = [c for c in (p.get("asset_cols") or []) if c in df.columns]
        if len(cols) < 2:
            raise ValueError("Select at least two asset columns.")
        is_returns = bool(p.get("is_returns", False))
        rtype = (p.get("return_type") or "simple").lower()
        ppy = int(p.get("periods_per_year") or 252)
        rf = float(p.get("rf_annual") or 0.02)
        objective = (p.get("objective") or "max_sharpe").lower()

        R = _returns_df(df, cols, is_returns, rtype)
        if len(R) < 5:
            raise ValueError("Need at least 5 aligned return observations.")
        mu = R.mean() * ppy                       # annualised expected returns
        S = R.cov() * ppy                          # annualised covariance
        n = len(cols)

        def solve(obj, **kw):
            ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
            if obj == "max_sharpe":
                ef.max_sharpe(risk_free_rate=rf)
            elif obj == "min_volatility":
                ef.min_volatility()
            elif obj == "efficient_return":
                ef.efficient_return(target_return=kw["target"])
            elif obj == "efficient_risk":
                ef.efficient_risk(target_volatility=kw["target"])
            w = ef.clean_weights()
            ret, vol, shr = ef.portfolio_performance(risk_free_rate=rf)
            return w, float(ret), float(vol), float(shr)

        if objective == "efficient_return":
            w, ret, vol, shr = solve("efficient_return", target=float(p.get("target_return") or float(mu.mean())))
        elif objective == "efficient_risk":
            tvol = float(p.get("target_volatility") or float(np.sqrt(np.diag(S)).mean()))
            w, ret, vol, shr = solve("efficient_risk", target=tvol)
        else:
            w, ret, vol, shr = solve(objective)

        weights = [{"asset": c, "weight": _fin(w.get(c, 0.0), 4)} for c in cols]
        weights_sorted = sorted(weights, key=lambda z: -(z["weight"] or 0))

        # efficient frontier
        frontier = []
        try:
            r_lo, r_hi = float(mu.min()), float(mu.max())
            for tr in np.linspace(r_lo, r_hi, 25):
                try:
                    ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
                    ef.efficient_return(target_return=tr)
                    rr, vv, _ = ef.portfolio_performance(risk_free_rate=rf)
                    frontier.append({"volatility": _fin(vv, 5), "return": _fin(rr, 5)})
                except Exception:
                    continue
        except Exception:
            frontier = []

        # per-asset points
        asset_pts = [{"asset": c, "volatility": _fin(float(np.sqrt(S.loc[c, c])), 5),
                      "return": _fin(float(mu[c]), 5)} for c in cols]

        plot = None
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=120,
                                           gridspec_kw={"width_ratios": [1.3, 1]})
            if frontier:
                fx = [f["volatility"]*100 for f in frontier]; fy = [f["return"]*100 for f in frontier]
                ax1.plot(fx, fy, "-", color="#2563eb", lw=2, label="Efficient frontier")
            ax1.scatter([a["volatility"]*100 for a in asset_pts], [a["return"]*100 for a in asset_pts],
                        color="#94a3b8", s=30, zorder=3)
            for a in asset_pts:
                ax1.annotate(a["asset"], (a["volatility"]*100, a["return"]*100),
                             fontsize=7, textcoords="offset points", xytext=(4, 3))
            ax1.scatter([vol*100], [ret*100], color="#dc2626", marker="*", s=260, zorder=5,
                        label=f"Chosen ({objective.replace('_',' ')})")
            ax1.set_xlabel("Volatility (ann. %)"); ax1.set_ylabel("Expected return (ann. %)")
            ax1.set_title("Efficient frontier"); ax1.legend(fontsize=8, frameon=False)
            # weights pie / bar
            nz = [(z["asset"], z["weight"]) for z in weights_sorted if (z["weight"] or 0) > 1e-4]
            ax2.barh([a for a, _ in nz][::-1], [wt*100 for _, wt in nz][::-1], color="#2563eb")
            ax2.set_xlabel("Weight (%)"); ax2.set_title("Optimal allocation")
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        n_held = sum(1 for z in weights if (z["weight"] or 0) > 1e-4)
        top = weights_sorted[0] if weights_sorted else None
        interpretation = (
            f"The {objective.replace('_', ' ')} portfolio holds {n_held} of {n} assets with an expected "
            f"return of {ret*100:.2f}% and volatility {vol*100:.2f}% per year, a Sharpe ratio of {shr:.2f}. "
            + (f"Its largest position is {top['asset']} at {top['weight']*100:.1f}%. " if top else "")
            + "Every point on the efficient frontier is the highest return achievable for its level of risk — "
            "portfolios below it are inefficient. Diversification lets the mix reach a better risk-return trade-off "
            "than any single asset because their movements partly offset."
        )

        results = {
            "status": "optimal", "unsolved": False, "n_assets": n, "n_obs": int(len(R)),
            "objective": objective, "periods_per_year": ppy, "rf_annual": _fin(rf, 6),
            "expected_return": _fin(ret, 6), "volatility": _fin(vol, 6), "sharpe": _fin(shr, 4),
            "weights": weights, "frontier": frontier, "assets": asset_pts,
            "interpretation": interpretation,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
