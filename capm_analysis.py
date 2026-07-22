#!/usr/bin/env python3
"""CAPM — estimate a security's beta and alpha against the market.
OLS regression of excess asset return on excess market return (statsmodels).

Input (from capm-page.tsx):
    data              : list[dict]
    asset_col         : str    asset price or return column
    market_col        : str    market/benchmark price or return column
    is_returns        : bool   True = columns already hold period returns
    return_type       : "simple"|"log" (default simple)
    periods_per_year  : int    (default 252)
    rf_annual         : float  annual risk-free rate (default 0)
    compare_assets    : list[str]  optional additional asset return/price columns — CAPM is
                                    re-run for each (vs the same market_col) and reported in
                                    results.risk_return_table (backend-ready; no UI wiring yet).
Output: { results: {...}, plot, and results.charts: {...} } — six-section CAPM report:
    CAPM Summary, CAPM Regression, Beta Analysis, Security Market Line,
    Expected vs Actual Return, Risk-Return Relationship (multi-asset, optional).
"""
import sys, json, io, base64
import numpy as np
import pandas as pd
import statsmodels.api as sm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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


def _capm_fit(ra, rm, rf_p, ppy):
    """Run the CAPM excess-return OLS for one asset vs the market and return a dict of stats."""
    y = ra - rf_p
    x = rm - rf_p
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    alpha, beta = float(model.params[0]), float(model.params[1])
    mean_mkt_excess = float(np.mean(x))
    capm_expected = rf_p + beta * mean_mkt_excess
    capm_expected_ann = (1 + capm_expected) ** ppy - 1
    actual_ann = (1 + np.mean(ra)) ** ppy - 1
    alpha_ann = (1 + alpha) ** ppy - 1
    return {
        "model": model, "alpha": alpha, "beta": beta,
        "mean_mkt_excess": mean_mkt_excess,
        "capm_expected_ann": capm_expected_ann, "actual_ann": actual_ann, "alpha_ann": alpha_ann,
        "y": y, "x": x,
    }


def _to_returns(series, is_returns, return_type):
    s = pd.to_numeric(series, errors="coerce")
    if is_returns:
        return s.dropna().reset_index(drop=True)
    if return_type == "log":
        if (s <= 0).any():
            raise ValueError("Log returns require strictly positive prices.")
        return np.log(s / s.shift(1)).dropna().reset_index(drop=True)
    return (s / s.shift(1) - 1.0).dropna().reset_index(drop=True)


def main():
    try:
        p = json.load(sys.stdin)
        rows = p.get("data") or []
        if not rows:
            raise ValueError("No data provided.")
        df = pd.DataFrame(rows)
        acol = p.get("asset_col"); mcol = p.get("market_col")
        if not acol or acol not in df.columns or not mcol or mcol not in df.columns:
            raise ValueError("Select valid asset and market columns.")
        if acol == mcol:
            raise ValueError("Asset and market columns must be different.")
        is_returns = bool(p.get("is_returns", False))
        rtype = (p.get("return_type") or "simple").lower()
        ppy = int(p.get("periods_per_year") or 252)
        rf_annual = float(p.get("rf_annual") or 0.0)
        compare_assets = [c for c in (p.get("compare_assets") or []) if c and c in df.columns and c != mcol]

        ra = _to_returns(df[acol], is_returns, rtype)
        rm = _to_returns(df[mcol], is_returns, rtype)
        n = min(len(ra), len(rm))
        if n < 5:
            raise ValueError("Need at least 5 aligned return observations.")
        ra = ra.iloc[-n:].to_numpy(); rm = rm.iloc[-n:].to_numpy()
        rf_p = (1 + rf_annual) ** (1 / ppy) - 1

        y = ra - rf_p          # excess asset return
        x = rm - rf_p          # excess market return
        X = sm.add_constant(x)
        model = sm.OLS(y, X).fit()
        alpha, beta = float(model.params[0]), float(model.params[1])
        alpha_se, beta_se = float(model.bse[0]), float(model.bse[1])
        alpha_t, beta_t = float(model.tvalues[0]), float(model.tvalues[1])
        alpha_p, beta_p = float(model.pvalues[0]), float(model.pvalues[1])
        r2 = float(model.rsquared)
        adj_r2 = float(model.rsquared_adj)
        f_stat = float(model.fvalue)
        f_p_value = float(model.f_pvalue)
        ci = model.conf_int(alpha=0.05)
        alpha_ci_low, alpha_ci_high = float(ci[0][0]), float(ci[0][1])
        beta_ci_low, beta_ci_high = float(ci[1][0]), float(ci[1][1])

        mean_mkt_excess = float(np.mean(x))
        capm_expected = rf_p + beta * mean_mkt_excess          # per period
        capm_expected_ann = (1 + capm_expected) ** ppy - 1
        alpha_ann = (1 + alpha) ** ppy - 1
        actual_ann = (1 + np.mean(ra)) ** ppy - 1
        mkt_return_mean = float(np.mean(rm))
        mkt_return_ann = (1 + mkt_return_mean) ** ppy - 1
        mkt_risk_premium_ann = mkt_return_ann - rf_annual
        total_var = float(np.var(ra, ddof=1))
        systematic = float(beta ** 2 * np.var(rm, ddof=1))
        idiosyncratic = max(total_var - systematic, 0.0)
        sys_pct = 100 * systematic / total_var if total_var > 0 else 0.0

        # --- per-period series for time-series charts ---
        fitted = model.fittedvalues  # aligned with y (excess asset return)
        fitted_arr = np.asarray(fitted, dtype=float)
        actual_arr = np.asarray(y, dtype=float)
        cumulative_actual = np.cumprod(1 + actual_arr) - 1
        cumulative_fitted = np.cumprod(1 + fitted_arr) - 1
        resid = np.asarray(model.resid, dtype=float)

        # --- Security Market Line: beta grid from 0 to 2x this asset's beta ---
        beta_grid = np.linspace(0, max(2 * beta, 0.1), 40)
        sml_expected = rf_p + beta_grid * mean_mkt_excess
        sml_expected_ann = (1 + sml_expected) ** ppy - 1
        sml_points = [{"beta": _fin(b, 4), "expected_return": _fin(er, 6)}
                      for b, er in zip(beta_grid.tolist(), sml_expected_ann.tolist())]
        sml = {
            "points": sml_points,
            "asset": {"beta": _fin(beta, 4), "expected_return": _fin(capm_expected_ann, 6),
                      "actual_return": _fin((1 + np.mean(ra)) ** ppy - 1, 6)},
            "risk_free_annual": _fin(rf_annual, 6),
            "market_expected_annual": _fin((1 + mean_mkt_excess + rf_p) ** ppy - 1, 6),
        }

        # --- rolling beta (if enough observations) ---
        roll_window = min(max(n // 4, 10), 60)
        rolling_beta = []
        if n >= roll_window * 2:
            xs = pd.Series(x); ys = pd.Series(y)
            roll_cov = ys.rolling(roll_window).cov(xs)
            roll_var = xs.rolling(roll_window).var()
            rb = (roll_cov / roll_var).dropna()
            rolling_beta = [_fin(v, 4) for v in rb.tolist()]

        # --- Risk-Return Relationship: optional multi-asset comparison (§6) ---
        risk_return_table = [{
            "asset": acol, "beta": _fin(beta, 4),
            "expected_return": _fin(capm_expected_ann, 6), "actual_return": _fin(actual_ann, 6),
            "alpha": _fin(alpha_ann, 6),
        }]
        for cc in compare_assets:
            try:
                rc = _to_returns(df[cc], is_returns, rtype)
                nc = min(len(rc), len(rm))
                if nc < 5:
                    continue
                fit = _capm_fit(rc.iloc[-nc:].to_numpy(), rm[-nc:] if isinstance(rm, np.ndarray) else rm.iloc[-nc:].to_numpy(), rf_p, ppy)
                risk_return_table.append({
                    "asset": cc, "beta": _fin(fit["beta"], 4),
                    "expected_return": _fin(fit["capm_expected_ann"], 6),
                    "actual_return": _fin(fit["actual_ann"], 6),
                    "alpha": _fin(fit["alpha_ann"], 6),
                })
            except Exception:
                continue

        # --- charts: one PNG per panel (tabbed on the frontend) ---
        charts = {}
        try:
            fig, ax1 = plt.subplots(figsize=(7, 5), dpi=120)
            ax1.scatter(x * 100, y * 100, s=14, alpha=0.5, color="#2563eb")
            xr = np.linspace(x.min(), x.max(), 50)
            ax1.plot(xr * 100, (alpha + beta * xr) * 100, color="#dc2626", lw=2,
                      label=f"β={beta:.2f}, α={alpha*100:.3f}%")
            ax1.axhline(0, color="#cbd5e1", lw=0.6); ax1.axvline(0, color="#cbd5e1", lw=0.6)
            ax1.set_xlabel("Market excess return (%)"); ax1.set_ylabel("Asset excess return (%)")
            ax1.set_title("Beta vs Market Return (characteristic line)"); ax1.legend(fontsize=8, frameon=False)
            fig.tight_layout(); charts["beta_scatter"] = _png(fig)
        except Exception:
            plt.close("all")

        try:
            fig, ax2 = plt.subplots(figsize=(7, 5), dpi=120)
            ax2.plot(cumulative_actual * 100, color="#2563eb", lw=1.6, label="Actual (cum.)")
            ax2.plot(cumulative_fitted * 100, color="#dc2626", lw=1.6, ls="--", label="Expected/fitted (cum.)")
            ax2.set_xlabel("Period"); ax2.set_ylabel("Cumulative excess return (%)")
            ax2.set_title("Cumulative Actual vs Expected Return"); ax2.legend(fontsize=8, frameon=False)
            fig.tight_layout(); charts["cumulative"] = _png(fig)
        except Exception:
            plt.close("all")

        try:
            fig, ax3 = plt.subplots(figsize=(6, 5), dpi=120)
            ax3.bar(["Systematic", "Idiosyncratic"], [sys_pct, 100 - sys_pct],
                    color=["#2563eb", "#94a3b8"])
            ax3.set_ylabel("% of variance"); ax3.set_title("Risk Decomposition")
            fig.tight_layout(); charts["risk_decomposition"] = _png(fig)
        except Exception:
            plt.close("all")

        try:
            fig, ax4 = plt.subplots(figsize=(7, 5), dpi=120)
            if rolling_beta:
                ax4.plot(rolling_beta, color="#7c3aed", lw=1.6)
                ax4.axhline(beta, color="#94a3b8", lw=0.8, ls=":")
                ax4.set_title(f"Rolling Beta (window={roll_window})")
            else:
                ax4.text(0.5, 0.5, "Not enough observations\nfor rolling beta",
                          ha="center", va="center", fontsize=9, color="#64748b")
                ax4.set_title("Rolling Beta")
            ax4.set_xlabel("Period"); ax4.set_ylabel("Beta")
            fig.tight_layout(); charts["rolling_beta"] = _png(fig)
        except Exception:
            plt.close("all")

        try:
            fig, ax5 = plt.subplots(figsize=(7, 5), dpi=120)
            ax5.plot(beta_grid, sml_expected_ann * 100, color="#0f766e", lw=2, label="SML")
            ax5.scatter([beta], [capm_expected_ann * 100], color="#dc2626", s=60, zorder=5, label=f"{acol} (asset)")
            ax5.scatter([0], [rf_annual * 100], color="#334155", s=40, zorder=5, label="Risk-free (β=0)")
            ax5.scatter([1], [sml["market_expected_annual"] * 100], color="#2563eb", s=40, zorder=5, label="Market (β=1)")
            ax5.set_xlabel("Beta"); ax5.set_ylabel("Expected return (annual, %)")
            ax5.set_title("Security Market Line"); ax5.legend(fontsize=7, frameon=False)
            fig.tight_layout(); charts["sml"] = _png(fig)
        except Exception:
            plt.close("all")

        try:
            fig, ax6r = plt.subplots(figsize=(6, 5), dpi=120)
            rb_betas = [row["beta"] for row in risk_return_table]
            rb_rets = [row["actual_return"] * 100 if row["actual_return"] is not None else None for row in risk_return_table]
            colors = ["#dc2626"] + ["#2563eb"] * (len(risk_return_table) - 1)
            ax6r.scatter(rb_betas, rb_rets, s=60, c=colors, zorder=5)
            for row in risk_return_table:
                if row["beta"] is not None and row["actual_return"] is not None:
                    ax6r.annotate(row["asset"], (row["beta"], row["actual_return"] * 100),
                                   fontsize=8, xytext=(4, 4), textcoords="offset points")
            ax6r.set_xlabel("Beta"); ax6r.set_ylabel("Actual return (annual, %)")
            ax6r.set_title("Beta vs Actual Return")
            fig.tight_layout(); charts["risk_return"] = _png(fig)
        except Exception:
            plt.close("all")

        try:
            fig, ax7 = plt.subplots(figsize=(6, 5), dpi=120)
            try:
                import scipy.stats as _stats
                _stats.probplot(resid, dist="norm", plot=ax7)
                ax7.get_lines()[0].set_markerfacecolor("#2563eb")
                ax7.get_lines()[0].set_markeredgecolor("#2563eb")
                ax7.get_lines()[0].set_markersize(4)
                ax7.get_lines()[1].set_color("#dc2626")
            except Exception:
                sr = np.sort(resid)
                qn = np.linspace(0.01, 0.99, len(sr))
                theo = np.quantile(np.random.normal(0, np.std(resid), 5000), qn)
                ax7.scatter(theo, sr, s=10, color="#2563eb")
            ax7.set_title("Residual Q-Q Plot")
            fig.tight_layout(); charts["residual_qq"] = _png(fig)
        except Exception:
            plt.close("all")

        plot = charts.get("sml")  # legacy single-image field kept for older callers; charts dict is authoritative

        risk_word = ("more volatile than the market (aggressive)" if beta > 1.05
                     else "less volatile than the market (defensive)" if beta < 0.95
                     else "roughly as volatile as the market")
        alpha_word = ("a statistically significant positive alpha — it beat its CAPM-required return"
                      if (alpha > 0 and alpha_p < 0.05) else
                      "a statistically significant negative alpha — it lagged its CAPM-required return"
                      if (alpha < 0 and alpha_p < 0.05) else
                      "no statistically significant alpha — its return is explained by market risk alone")
        interpretation = (
            f"The asset's beta is {beta:.3f}, meaning it is {risk_word}: a 1% market move is associated with "
            f"about a {beta:.2f}% move in the asset. The regression shows {alpha_word} "
            f"(annualised alpha {alpha_ann*100:.2f}%). Market risk explains {sys_pct:.0f}% of the asset's "
            f"variance (R² = {r2:.2f}); the rest is idiosyncratic and diversifiable. CAPM's required "
            f"(fair) return for this beta is {capm_expected_ann*100:.2f}% annualised."
        )

        # --- ① CAPM Summary ---
        capm_summary = [
            {"metric": "Beta", "value": _fin(beta, 4)},
            {"metric": "Alpha (annualized)", "value": _fin(alpha_ann, 6)},
            {"metric": "Risk-Free Rate (annual)", "value": _fin(rf_annual, 6)},
            {"metric": "Market Return (mean, per period)", "value": _fin(mkt_return_mean, 6)},
            {"metric": "Market Return (annualized)", "value": _fin(mkt_return_ann, 6)},
            {"metric": "Market Risk Premium (annual)", "value": _fin(mkt_risk_premium_ann, 6)},
            {"metric": "Expected Return (CAPM, annual)", "value": _fin(capm_expected_ann, 6)},
            {"metric": "Actual Return (asset, annual)", "value": _fin(actual_ann, 6)},
            {"metric": "Jensen's Alpha (annual)", "value": _fin(alpha_ann, 6)},
        ]

        # --- ② CAPM Regression ---
        regression_table = [
            {"term": "Alpha (α)", "estimate": _fin(alpha, 6), "se": _fin(alpha_se, 6),
             "t": _fin(alpha_t, 3), "p": _fin(alpha_p, 5)},
            {"term": "Beta (β)", "estimate": _fin(beta, 4), "se": _fin(beta_se, 4),
             "t": _fin(beta_t, 3), "p": _fin(beta_p, 5)},
        ]
        regression_stats = {
            "r_squared": _fin(r2, 4), "adj_r_squared": _fin(adj_r2, 4),
            "n_obs": int(n), "f_stat": _fin(f_stat, 3), "f_p_value": _fin(f_p_value, 5),
        }

        # --- ③ Beta Analysis ---
        beta_analysis = {
            "beta": _fin(beta, 4), "beta_se": _fin(beta_se, 4),
            "t_stat": _fin(beta_t, 3), "p_value": _fin(beta_p, 5),
            "ci_low": _fin(beta_ci_low, 4), "ci_high": _fin(beta_ci_high, 4),
            "r_squared": _fin(r2, 4),
        }

        # --- ⑤ Expected vs Actual Return ---
        expected_vs_actual = {
            "risk_free_annual": _fin(rf_annual, 6),
            "market_risk_premium_annual": _fin(mkt_risk_premium_ann, 6),
            "beta": _fin(beta, 4),
            "expected_return_annual": _fin(capm_expected_ann, 6),
            "actual_return_annual": _fin(actual_ann, 6),
            "excess_return_annual": _fin(actual_ann - capm_expected_ann, 6),
        }

        results = {
            "status": "ok", "n_obs": int(n), "asset_col": acol, "market_col": mcol,
            "periods_per_year": ppy, "rf_annual": _fin(rf_annual, 6),
            "beta": _fin(beta, 4), "alpha_period": _fin(alpha, 6), "alpha_annual": _fin(alpha_ann, 6),
            "beta_se": _fin(beta_se, 4), "alpha_se": _fin(alpha_se, 6),
            "beta_t": _fin(beta_t, 3), "alpha_t": _fin(alpha_t, 3),
            "beta_p": _fin(beta_p, 5), "alpha_p": _fin(alpha_p, 5),
            "beta_ci_low": _fin(beta_ci_low, 4), "beta_ci_high": _fin(beta_ci_high, 4),
            "alpha_ci_low": _fin(alpha_ci_low, 6), "alpha_ci_high": _fin(alpha_ci_high, 6),
            "r_squared": _fin(r2, 4), "adj_r_squared": _fin(adj_r2, 4),
            "f_stat": _fin(f_stat, 3), "f_p_value": _fin(f_p_value, 5),
            "correlation": _fin(float(np.corrcoef(ra, rm)[0, 1]), 4),
            "capm_expected_annual": _fin(capm_expected_ann, 6),
            "asset_return_annual": _fin(actual_ann, 6),
            "market_return_annual": _fin(mkt_return_ann, 6),
            "market_risk_premium_annual": _fin(mkt_risk_premium_ann, 6),
            "systematic_pct": _fin(sys_pct, 2), "idiosyncratic_pct": _fin(100 - sys_pct, 2),
            "interpretation": interpretation,
            "sml": sml,
            "fitted_series": [_fin(v, 6) for v in fitted_arr.tolist()],
            "actual_series": [_fin(v, 6) for v in actual_arr.tolist()],
            "cumulative_actual": [_fin(v, 6) for v in cumulative_actual.tolist()],
            "cumulative_fitted": [_fin(v, 6) for v in cumulative_fitted.tolist()],
            "rolling_beta": rolling_beta, "rolling_beta_window": roll_window if rolling_beta else None,
            "capm_summary": capm_summary,
            "regression_table": regression_table, "regression_stats": regression_stats,
            "beta_analysis": beta_analysis,
            "expected_vs_actual": expected_vs_actual,
            "risk_return_table": risk_return_table,
            "charts": charts,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
