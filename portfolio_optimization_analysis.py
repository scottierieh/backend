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

Output: { results: {...}, plot,
          results.charts: {efficient_frontier, weights_bar, portfolio_comparison,
                            risk_contribution, backtest} }

Note: this page does not collect the user's *current* portfolio weights, so
sections that compare "current vs optimal" (allocation change, rebalancing
trades) are not built — results.current_portfolio_note explains this. An
equal-weight portfolio is used instead, purely as a labeled baseline for
comparison (never presented as the user's actual holdings).
"""
import sys, json, io, base64
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pypfopt import EfficientFrontier
from scipy.optimize import minimize

BLUE = "#2563eb"
RED = "#dc2626"
GREEN = "#16a34a"
AMBER = "#d97706"
GREY = "#94a3b8"
PURPLE = "#7c3aed"


def _fin(x, nd=6):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


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


def _perf(w, mu, S, rf):
    """Annualised return/vol/sharpe for a weight vector (numpy) given annualised mu/S."""
    w = np.asarray(w, dtype=float)
    ret = float(w @ mu.values)
    vol = float(np.sqrt(max(w @ S.values @ w, 0.0)))
    shr = (ret - rf) / vol if vol > 0 else 0.0
    return ret, vol, shr


def _risk_parity_weights(S):
    """Numerical equal-risk-contribution portfolio (long-only, sums to 1)."""
    n = S.shape[0]
    Sigma = S.values

    def rc_diff(w):
        port_var = w @ Sigma @ w
        if port_var <= 0:
            return 1e6
        sigma = np.sqrt(port_var)
        mcr = (Sigma @ w) / sigma
        rc = w * mcr
        target = rc.mean()
        return float(np.sum((rc - target) ** 2))

    w0 = np.ones(n) / n
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    bounds = [(1e-4, 1.0)] * n
    try:
        res = minimize(rc_diff, w0, method="SLSQP", bounds=bounds, constraints=cons,
                        options={"maxiter": 500, "ftol": 1e-12})
        if res.success:
            w = np.clip(res.x, 0, None)
            return w / w.sum(), True
    except Exception:
        pass
    return w0, False


def _drawdown_stats(port_returns, ppy, rf):
    """CAGR, annualised vol, sharpe, max drawdown from a period-return series (numpy array)."""
    port_returns = np.asarray(port_returns, dtype=float)
    n = len(port_returns)
    if n < 2:
        return None
    wealth = np.cumprod(1 + port_returns)
    total_return = wealth[-1] - 1.0
    years = n / ppy
    cagr = (wealth[-1] ** (1.0 / years) - 1.0) if years > 0 and wealth[-1] > 0 else None
    vol = float(np.std(port_returns, ddof=1) * np.sqrt(ppy))
    sharpe = ((cagr - rf) / vol) if (cagr is not None and vol > 0) else None
    peak = np.maximum.accumulate(wealth)
    dd = wealth / peak - 1.0
    max_dd = float(dd.min())
    return {"cagr": _fin(cagr, 5), "volatility": _fin(vol, 5), "sharpe": _fin(sharpe, 4),
            "max_drawdown": _fin(max_dd, 5), "total_return": _fin(total_return, 5)}, wealth, dd


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

        w_vec = np.array([w.get(c, 0.0) for c in cols])
        weights = [{"asset": c, "weight": _fin(w.get(c, 0.0), 4)} for c in cols]
        weights_sorted = sorted(weights, key=lambda z: -(z["weight"] or 0))

        # ---------------------------------------------------------------
        # efficient frontier
        # ---------------------------------------------------------------
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

        # ---------------------------------------------------------------
        # ⑤ alternative-objective portfolios (min-vol, max-sharpe, risk parity)
        # computed from the SAME mu/S regardless of the chosen `objective` above
        # ---------------------------------------------------------------
        try:
            w_minvol, ret_mv, vol_mv, shr_mv = solve("min_volatility")
        except Exception:
            w_minvol, ret_mv, vol_mv, shr_mv = None, None, None, None
        try:
            w_maxsh, ret_ms, vol_ms, shr_ms = solve("max_sharpe")
        except Exception:
            w_maxsh, ret_ms, vol_ms, shr_ms = None, None, None, None

        w_rp, rp_converged = _risk_parity_weights(S)
        ret_rp, vol_rp, shr_rp = _perf(w_rp, mu, S, rf)

        def _wvec(wd):
            if wd is None:
                return None
            return np.array([wd.get(c, 0.0) for c in cols])

        def _maxdd_for(weight_vector):
            if weight_vector is None:
                return None
            port = R[cols].values @ weight_vector
            stats, _, _ = _drawdown_stats(port, ppy, rf) if len(port) >= 2 else (None, None, None)
            return stats["max_drawdown"] if stats else None

        portfolio_comparison = [
            {"portfolio": "Minimum Volatility", "return": _fin(ret_mv, 5), "volatility": _fin(vol_mv, 5),
             "sharpe": _fin(shr_mv, 4), "max_drawdown": _maxdd_for(_wvec(w_minvol))},
            {"portfolio": "Maximum Sharpe", "return": _fin(ret_ms, 5), "volatility": _fin(vol_ms, 5),
             "sharpe": _fin(shr_ms, 4), "max_drawdown": _maxdd_for(_wvec(w_maxsh))},
            {"portfolio": "Risk Parity" + ("" if rp_converged else " (equal-weight approx.)"),
             "return": _fin(ret_rp, 5), "volatility": _fin(vol_rp, 5), "sharpe": _fin(shr_rp, 4),
             "max_drawdown": _maxdd_for(w_rp)},
            {"portfolio": f"Selected ({objective.replace('_', ' ')})", "return": _fin(ret, 5),
             "volatility": _fin(vol, 5), "sharpe": _fin(shr, 4), "max_drawdown": _maxdd_for(w_vec)},
        ]

        # ---------------------------------------------------------------
        # ⑦ risk contribution of the SELECTED optimal portfolio
        # rc_i = w_i * (Sigma w)_i / port_vol  (same formula as portfolio_risk_analysis.py)
        # ---------------------------------------------------------------
        Sigma = S.values
        port_var = float(w_vec @ Sigma @ w_vec)
        port_sd = np.sqrt(port_var) if port_var > 0 else 0.0
        if port_sd > 0:
            mcr = (Sigma @ w_vec) / port_sd
            rc = w_vec * mcr
            rc_pct = rc / rc.sum() if rc.sum() != 0 else rc
        else:
            rc_pct = np.zeros(n)
        risk_contributions = [{"asset": cols[i], "weight": _fin(w_vec[i], 4),
                                "risk_contribution": _fin(float(rc_pct[i]), 4)} for i in range(n)]
        risk_contributions_sorted = sorted(risk_contributions, key=lambda z: -(z["risk_contribution"] or 0))

        # ---------------------------------------------------------------
        # ① optimization summary (optimal only — no "current portfolio" input on this page)
        # ---------------------------------------------------------------
        n_held = sum(1 for z in weights if (z["weight"] or 0) > 1e-2)
        max_w = float(max((z["weight"] or 0) for z in weights)) if weights else None
        current_portfolio_note = (
            "This page does not collect the user's current portfolio weights, so no "
            "'current vs optimal' comparison, allocation-change, or rebalancing-trade table is "
            "shown. An equal-weight portfolio is used only as a neutral, clearly-labeled baseline "
            "in the portfolio comparison below — it is not presented as the user's actual holdings."
        )
        optimization_summary = {
            "optimal": {"expected_return": _fin(ret, 6), "volatility": _fin(vol, 6), "sharpe": _fin(shr, 4),
                        "max_weight": _fin(max_w, 4), "n_assets_held": n_held, "n_assets_total": n},
            "_note": ("Model estimate based on the historical expected-return and covariance inputs used for "
                      "this optimization — not a guaranteed future outcome. Past co-movement of assets may not persist."),
            "current_portfolio_note": current_portfolio_note,
        }

        # ---------------------------------------------------------------
        # ⑧ constraints actually applied (long-only, weight_bounds=(0,1))
        # ---------------------------------------------------------------
        constraints_summary = [
            {"label": "Position type", "value": "Long-only (no short selling)"},
            {"label": "Per-asset weight bounds", "value": "[0%, 100%]"},
            {"label": "Fully invested", "value": "Weights sum to 100%"},
            {"label": "Objective", "value": objective.replace("_", " ").title()},
        ]

        # ---------------------------------------------------------------
        # ⑫ historical backtest: optimal weights vs equal-weight, applied retroactively
        # to the SAME historical return series already loaded (not a future guarantee)
        # ---------------------------------------------------------------
        w_eq = np.ones(n) / n
        port_opt_returns = R[cols].values @ w_vec
        port_eq_returns = R[cols].values @ w_eq
        bt_opt, wealth_opt, dd_opt = _drawdown_stats(port_opt_returns, ppy, rf)
        bt_eq, wealth_eq, dd_eq = _drawdown_stats(port_eq_returns, ppy, rf)
        backtest = {
            "optimal": bt_opt, "equal_weight": bt_eq, "n_periods": int(len(R)),
            "_note": ("Historical backtest — applies the optimal weights retroactively to the same "
                      "historical return series used to estimate them (in-sample). Not a guarantee of "
                      "future performance; out-of-sample results are typically worse."),
        }

        # ---------------------------------------------------------------
        # Monte-Carlo random-portfolio cloud (context behind the frontier)
        # ---------------------------------------------------------------
        rng = np.random.default_rng(42)
        n_sim = 1500
        rand_w = rng.dirichlet(np.ones(n), size=n_sim)
        rand_ret = rand_w @ mu.values
        rand_vol = np.sqrt(np.einsum("ij,jk,ik->i", rand_w, Sigma, rand_w))

        # ---------------------------------------------------------------
        # charts (separate PNGs, rendered as VisualizationTabs)
        # ---------------------------------------------------------------
        charts = {}

        # ④ efficient frontier
        try:
            fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=120)
            ax.scatter(rand_vol * 100, rand_ret * 100, s=6, color=GREY, alpha=0.25, zorder=1, label="Random portfolios")
            if frontier:
                fx = [f["volatility"] * 100 for f in frontier]; fy = [f["return"] * 100 for f in frontier]
                ax.plot(fx, fy, "-", color=BLUE, lw=2, zorder=3, label="Efficient frontier")
            ax.scatter([a["volatility"] * 100 for a in asset_pts], [a["return"] * 100 for a in asset_pts],
                       color="#475569", s=35, zorder=4, marker="D", label="Individual assets")
            for a in asset_pts:
                ax.annotate(a["asset"], (a["volatility"] * 100, a["return"] * 100),
                            fontsize=7, textcoords="offset points", xytext=(4, 3))
            if vol_mv is not None:
                ax.scatter([vol_mv * 100], [ret_mv * 100], color=GREEN, marker="P", s=170, zorder=5, label="Min. volatility")
            if vol_ms is not None:
                ax.scatter([vol_ms * 100], [ret_ms * 100], color=AMBER, marker="^", s=170, zorder=5, label="Max. Sharpe")
            ax.scatter([vol * 100], [ret * 100], color=RED, marker="*", s=280, zorder=6,
                       label=f"Selected ({objective.replace('_', ' ')})")
            ax.set_xlabel("Volatility (ann. %)"); ax.set_ylabel("Expected return (ann. %)")
            ax.set_title("Efficient Frontier"); ax.legend(fontsize=7.5, frameon=False, loc="best")
            ax.grid(alpha=0.2)
            fig.tight_layout()
            charts["efficient_frontier"] = _png(fig)
        except Exception:
            plt.close("all"); charts["efficient_frontier"] = None

        # ⑥ weights bar (optimal allocation)
        try:
            fig, ax = plt.subplots(figsize=(6.5, 4.8), dpi=120)
            nz = [(z["asset"], z["weight"]) for z in weights_sorted if (z["weight"] or 0) > 1e-4]
            ax.barh([a for a, _ in nz][::-1], [wt * 100 for _, wt in nz][::-1], color=BLUE)
            ax.set_xlabel("Weight (%)"); ax.set_title("Optimal Allocation")
            ax.grid(alpha=0.2, axis="x")
            fig.tight_layout()
            charts["weights_bar"] = _png(fig)
        except Exception:
            plt.close("all"); charts["weights_bar"] = None

        # ⑤ risk-return comparison of alternative portfolios
        try:
            fig, ax = plt.subplots(figsize=(6.5, 5.2), dpi=120)
            colors = {"Minimum Volatility": GREEN, "Maximum Sharpe": AMBER, "Risk Parity": PURPLE}
            for row in portfolio_comparison:
                name = row["portfolio"]
                base = next((k for k in colors if name.startswith(k)), None)
                color = colors.get(base, RED)
                marker = "*" if name.startswith("Selected") else "o"
                size = 260 if name.startswith("Selected") else 140
                if row["volatility"] is not None and row["return"] is not None:
                    ax.scatter([row["volatility"] * 100], [row["return"] * 100], color=color, s=size,
                               marker=marker, zorder=4, edgecolor="white", linewidth=0.6, label=name)
            ax.set_xlabel("Volatility (ann. %)"); ax.set_ylabel("Expected return (ann. %)")
            ax.set_title("Risk-Return Comparison"); ax.legend(fontsize=7.5, frameon=False, loc="best")
            ax.grid(alpha=0.2)
            fig.tight_layout()
            charts["portfolio_comparison"] = _png(fig)
        except Exception:
            plt.close("all"); charts["portfolio_comparison"] = None

        # ⑦ risk contribution
        try:
            fig, ax = plt.subplots(figsize=(6.5, 4.8), dpi=120)
            order = sorted(range(n), key=lambda i: rc_pct[i])
            ax.barh([cols[i] for i in order], [rc_pct[i] * 100 for i in order], color=BLUE)
            ax.set_xlabel("Risk contribution (%)"); ax.set_title("Risk Contribution (selected portfolio)")
            ax.grid(alpha=0.2, axis="x")
            fig.tight_layout()
            charts["risk_contribution"] = _png(fig)
        except Exception:
            plt.close("all"); charts["risk_contribution"] = None

        # ⑫ backtest equity curve
        try:
            fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=120)
            ax.plot(wealth_opt, color=BLUE, lw=1.8, label=f"Optimal ({objective.replace('_', ' ')})")
            ax.plot(wealth_eq, color=GREY, lw=1.5, ls="--", label="Equal-weight")
            ax.set_xlabel("Period"); ax.set_ylabel("Growth of 1 unit")
            ax.set_title("Historical Backtest (in-sample)"); ax.legend(fontsize=8, frameon=False)
            ax.grid(alpha=0.2)
            fig.tight_layout()
            charts["backtest"] = _png(fig)
        except Exception:
            plt.close("all"); charts["backtest"] = None

        # legacy combined plot (kept for backward compatibility)
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
            nz = [(z["asset"], z["weight"]) for z in weights_sorted if (z["weight"] or 0) > 1e-4]
            ax2.barh([a for a, _ in nz][::-1], [wt*100 for _, wt in nz][::-1], color="#2563eb")
            ax2.set_xlabel("Weight (%)"); ax2.set_title("Optimal allocation")
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

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
            "optimization_summary": optimization_summary,
            "portfolio_comparison": portfolio_comparison,
            "risk_contributions": risk_contributions_sorted,
            "constraints_summary": constraints_summary,
            "backtest": backtest,
            "charts": charts,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
