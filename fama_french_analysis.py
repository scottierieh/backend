#!/usr/bin/env python3
"""Fama-French Factor Model — regress an asset's excess return on risk factors.
statsmodels OLS.

Works for the 3-factor (Mkt-RF, SMB, HML), 5-factor (+ RMW, CMA) or any custom
factor set: the caller simply selects which columns are the factors.

Input (from fama-french-page.tsx):
    data             : list[dict]
    asset_col        : str     asset/portfolio return column
    factor_cols      : string[]  factor return columns (e.g. Mkt-RF, SMB, HML)
    rf_col           : str     (optional) risk-free column; excess = asset - rf
    periods_per_year : int     (default 252) to annualise alpha
Output: { results: {loadings, alpha, r_squared, factor_summary, factor_significance,
          factor_contribution_table, alpha_comparison, diagnostics, charts, ...}, plot }
"""
import sys, json, io, base64
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#2563eb"
GREEN = "#16a34a"
RED = "#dc2626"
GRAY = "#94a3b8"
PALETTE = ["#2563eb", "#dc2626", "#0f766e", "#7c3aed", "#d97706", "#059669"]


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


def _guess_market_col(factor_cols):
    """Identify which factor_col is the market factor (Mkt-RF or similar name)."""
    for c in factor_cols:
        cl = c.lower().replace(" ", "").replace("_", "")
        if cl in ("mktrf", "mkt-rf", "mkt", "market", "marketrf"):
            return c
    for c in factor_cols:
        if "mkt" in c.lower() or "market" in c.lower():
            return c
    return factor_cols[0]


def main():
    try:
        p = json.load(sys.stdin)
        rows = p.get("data") or []
        if not rows:
            raise ValueError("No data provided.")
        df = pd.DataFrame(rows)
        asset_col = p.get("asset_col")
        factor_cols = [c for c in (p.get("factor_cols") or []) if c in df.columns]
        rf_col = p.get("rf_col") or None
        ppy = int(p.get("periods_per_year") or 252)
        if not asset_col or asset_col not in df.columns:
            raise ValueError("Select the asset return column.")
        if len(factor_cols) < 1:
            raise ValueError("Select at least one factor column.")

        y = pd.to_numeric(df[asset_col], errors="coerce")
        if rf_col and rf_col in df.columns:
            rf = pd.to_numeric(df[rf_col], errors="coerce")
            y_excess = y - rf
            excess_label = f"{asset_col} - {rf_col}"
        else:
            y_excess = y
            excess_label = asset_col
        X = df[factor_cols].apply(pd.to_numeric, errors="coerce")

        reg = pd.concat([y_excess.rename("_y"), X], axis=1).dropna().reset_index(drop=True)
        if len(reg) < len(factor_cols) + 5:
            raise ValueError(f"Need at least {len(factor_cols) + 5} aligned observations for {len(factor_cols)} factors.")

        yv = reg["_y"].values
        Xv = sm.add_constant(reg[factor_cols].values)
        model = sm.OLS(yv, Xv).fit()

        names = ["alpha"] + factor_cols
        coefs = model.params
        std_errs = model.bse
        tvals = model.tvalues
        pvals = model.pvalues
        conf = model.conf_int()

        loadings = []
        for i, nm in enumerate(names):
            loadings.append({
                "name": nm,
                "coef": _fin(coefs[i], 6),
                "std_err": _fin(std_errs[i], 6),
                "t_stat": _fin(tvals[i], 4),
                "p_value": _fin(pvals[i], 6),
                "ci_low": _fin(conf[i][0], 6),
                "ci_high": _fin(conf[i][1], 6),
                "significant": bool(pvals[i] < 0.05),
            })

        alpha_period = float(coefs[0])
        alpha_annual = alpha_period * ppy
        alpha_p = float(pvals[0])

        # market factor identification (for section 1 and section 6)
        market_col = _guess_market_col(factor_cols)
        other_factors = [c for c in factor_cols if c != market_col]
        market_loading = next((l for l in loadings if l["name"] == market_col), None)

        # factor premium: mean period/annualised return of each factor itself
        factor_premium = []
        for c in factor_cols:
            mean_period = float(reg[c].mean())
            factor_premium.append({
                "name": c,
                "mean_period": _fin(mean_period, 6),
                "annualized": _fin(mean_period * ppy, 6),
            })
        factor_premium_map = {f["name"]: f["annualized"] for f in factor_premium}

        # predicted vs actual for plot
        fitted = model.fittedvalues
        resid = model.resid

        # cumulative growth of each factor (for "Cumulative Factor Return" chart)
        cumulative_factor_return = {}
        for c in factor_cols:
            fvals = reg[c].to_numpy(dtype=float)
            cumulative_factor_return[c] = [_fin(v, 6) for v in (np.cumprod(1 + fvals) - 1).tolist()]

        # per-period contribution of each factor to the fitted return: loading * factor value
        factor_loading_map = {f["name"]: f["coef"] for f in loadings[1:]}
        factor_contribution_series = {}
        for c in factor_cols:
            fvals = reg[c].to_numpy(dtype=float)
            contrib = factor_loading_map.get(c, 0.0) * fvals
            factor_contribution_series[c] = [_fin(v, 6) for v in contrib.tolist()]

        # ─────────────────────── ① Factor Model Summary ───────────────────────
        factor_summary = {
            "alpha_annual": _fin(alpha_annual, 6),
            "alpha_p_value": _fin(alpha_p, 6),
            "market_factor": market_col,
            "market_beta": market_loading["coef"] if market_loading else None,
            "other_factors": [
                {"name": l["name"], "coef": l["coef"], "significant": l["significant"]}
                for l in loadings[1:] if l["name"] != market_col
            ],
            "r_squared": _fin(model.rsquared, 4),
            "adj_r_squared": _fin(model.rsquared_adj, 4),
        }

        # ─────────────────────── ② Factor Regression summary block ───────────────────────
        regression_summary = {
            "r_squared": _fin(model.rsquared, 4),
            "adj_r_squared": _fin(model.rsquared_adj, 4),
            "f_stat": _fin(model.fvalue, 4),
            "f_p_value": _fin(model.f_pvalue, 6),
            "n_obs": int(len(reg)),
        }

        # ─────────────────────── ④ Factor Significance ───────────────────────
        factor_significance = [
            {"name": l["name"], "coef": l["coef"], "p_value": l["p_value"], "significant": l["significant"]}
            for l in loadings[1:]
        ]

        # ─────────────────────── ⑤ Factor Contribution ───────────────────────
        # contribution (return units, annualised) = loading * factor's annualised premium
        factor_contribution_table = []
        total_factor_contribution = 0.0
        for l in loadings[1:]:
            prem = factor_premium_map.get(l["name"])
            contrib = (l["coef"] * prem) if (l["coef"] is not None and prem is not None) else None
            if contrib is not None:
                total_factor_contribution += contrib
            factor_contribution_table.append({
                "name": l["name"], "coef": l["coef"], "factor_premium_annual": prem,
                "contribution": _fin(contrib, 6) if contrib is not None else None,
            })
        total_explained_return = total_factor_contribution + alpha_annual
        factor_contribution_table.append({
            "name": "Alpha", "coef": None, "factor_premium_annual": None,
            "contribution": _fin(alpha_annual, 6),
        })
        factor_contribution_table.append({
            "name": "Total Explained Return", "coef": None, "factor_premium_annual": None,
            "contribution": _fin(total_explained_return, 6),
        })
        # kept for backward compatibility with any earlier-referenced fields
        attribution = [
            {
                "name": row["name"], "beta": row["coef"],
                "factor_mean_annual": row["factor_premium_annual"],
                "contribution": row["contribution"],
                "pct_of_total": _fin(row["contribution"] / total_explained_return, 4)
                if row["contribution"] is not None and total_explained_return not in (None, 0) else None,
            }
            for row in factor_contribution_table if row["name"] not in ("Alpha", "Total Explained Return")
        ]

        # ─────────────────────── ⑥ Alpha Comparison: CAPM vs Fama-French ───────────────────────
        alpha_comparison = []
        try:
            mi = factor_cols.index(market_col)
            X_capm = sm.add_constant(reg[[market_col]].values)
            model_capm = sm.OLS(yv, X_capm).fit()
            capm_alpha_annual = float(model_capm.params[0]) * ppy
            alpha_comparison.append({
                "model": "CAPM", "alpha_annual": _fin(capm_alpha_annual, 6),
                "r_squared": _fin(model_capm.rsquared, 4),
            })
            # optional 3-factor row if a classic Mkt-RF/SMB/HML subset exists within a larger (e.g. 5-factor) set
            classic3 = []
            for pat in ["mkt", "smb", "hml"]:
                match = next((c for c in factor_cols if pat in c.lower()), None)
                if match:
                    classic3.append(match)
            if len(classic3) == 3 and len(factor_cols) > 3:
                X3 = sm.add_constant(reg[classic3].values)
                model3 = sm.OLS(yv, X3).fit()
                alpha_comparison.append({
                    "model": "3-Factor", "alpha_annual": _fin(float(model3.params[0]) * ppy, 6),
                    "r_squared": _fin(model3.rsquared, 4),
                })
            alpha_comparison.append({
                "model": "Fama-French", "alpha_annual": _fin(alpha_annual, 6),
                "r_squared": _fin(model.rsquared, 4),
            })
        except Exception:
            alpha_comparison = []

        # ─────────────────────── Diagnostics (result summary only) ───────────────────────
        diagnostics = {}
        try:
            dw = float(durbin_watson(resid))
            jb_stat, jb_p, skew, kurt = jarque_bera(resid)
            bp_lm, bp_p, _, _ = het_breuschpagan(resid, Xv)
            diagnostics = {
                "durbin_watson": _fin(dw, 3),
                "dw_note_en": "no strong autocorrelation" if 1.5 < dw < 2.5 else "possible autocorrelation",
                "dw_note_ko": "강한 자기상관 없음" if 1.5 < dw < 2.5 else "자기상관 가능성",
                "jarque_bera": _fin(jb_stat, 3), "jb_p_value": _fin(jb_p, 4),
                "residuals_normal": bool(jb_p >= 0.05),
                "skew": _fin(skew, 4), "kurtosis": _fin(kurt, 4),
                "breusch_pagan": _fin(bp_lm, 3), "bp_p_value": _fin(bp_p, 4),
                "homoskedastic": bool(bp_p >= 0.05),
            }
        except Exception:
            diagnostics = {}

        # ─────────────────────────────── Charts ───────────────────────────────
        # ③ Factor Exposure (standalone chart)
        chart_factor_exposure = None
        try:
            fl = loadings[1:]
            xs = [f["name"] for f in fl]
            ys = [f["coef"] for f in fl]
            cols = [BLUE if f["significant"] else GRAY for f in fl]
            fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=115)
            ax.barh(xs, ys, color=cols)
            ax.axvline(0, color="#111827", lw=0.8)
            for i, f in enumerate(fl):
                ax.text(f["coef"], i, f" {f['coef']:.3f}{'*' if f['significant'] else ''}",
                        va="center", ha="left" if f["coef"] >= 0 else "right", fontsize=8)
            ax.set_xlabel("Factor loading (β)")
            ax.set_title("Factor Exposure (blue = significant at 5%)")
            fig.tight_layout()
            chart_factor_exposure = _png(fig)
        except Exception:
            plt.close("all"); chart_factor_exposure = None

        # ⑤ Factor Contribution to Return (bar incl. alpha)
        chart_factor_contribution = None
        try:
            rows_c = [r for r in factor_contribution_table if r["name"] != "Total Explained Return"]
            xs = [r["name"] for r in rows_c]
            ys = [(r["contribution"] or 0) * 100 for r in rows_c]
            colors = [GREEN if v >= 0 else RED for v in ys]
            fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=115)
            ax.bar(xs, ys, color=colors)
            ax.axhline(0, color="#111827", lw=0.7)
            ax.set_ylabel("Contribution to annual return (%)")
            ax.set_title("Factor Contribution to Return")
            ax.tick_params(axis="x", rotation=25)
            fig.tight_layout()
            chart_factor_contribution = _png(fig)
        except Exception:
            plt.close("all"); chart_factor_contribution = None

        # ⑥ Alpha Comparison: CAPM vs Fama-French
        chart_alpha_comparison = None
        if alpha_comparison:
            try:
                models_lbl = [m["model"] for m in alpha_comparison]
                alphas = [(m["alpha_annual"] or 0) * 100 for m in alpha_comparison]
                r2s = [(m["r_squared"] or 0) * 100 for m in alpha_comparison]
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.5, 4.4), dpi=115)
                ax1.bar(models_lbl, alphas, color=PALETTE[:len(models_lbl)])
                ax1.axhline(0, color="#111827", lw=0.7)
                ax1.set_title("Alpha (annualised, %)"); ax1.set_ylabel("%")
                ax2.bar(models_lbl, r2s, color=PALETTE[:len(models_lbl)])
                ax2.set_title("R² (%)"); ax2.set_ylabel("%")
                fig.tight_layout()
                chart_alpha_comparison = _png(fig)
            except Exception:
                plt.close("all"); chart_alpha_comparison = None

        # cumulative factor return / factor contribution series charts, kept from the prior session
        chart_cumulative_factor = None
        try:
            fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=115)
            for i, c in enumerate(factor_cols):
                ax.plot(cumulative_factor_return[c], color=PALETTE[i % len(PALETTE)], lw=1.6, label=c)
            ax.axhline(0, color="#cbd5e1", lw=0.6)
            ax.set_xlabel("Period"); ax.set_ylabel("Cumulative return")
            ax.set_title("Cumulative Factor Return"); ax.legend(fontsize=8, frameon=False)
            fig.tight_layout()
            chart_cumulative_factor = _png(fig)
        except Exception:
            plt.close("all"); chart_cumulative_factor = None

        chart_contribution_series = None
        try:
            contrib_stack = np.array([factor_contribution_series[c] for c in factor_cols], dtype=float)
            xs_idx = np.arange(contrib_stack.shape[1]) if contrib_stack.size else np.array([])
            fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=115)
            if contrib_stack.size:
                ax.stackplot(xs_idx, contrib_stack, labels=factor_cols,
                              colors=[PALETTE[i % len(PALETTE)] for i in range(len(factor_cols))], alpha=0.8)
            ax.axhline(0, color="#111827", lw=0.6)
            ax.set_xlabel("Period"); ax.set_ylabel("Contribution to fitted return")
            ax.set_title("Factor Return Contribution (per period)"); ax.legend(fontsize=8, frameon=False)
            fig.tight_layout()
            chart_contribution_series = _png(fig)
        except Exception:
            plt.close("all"); chart_contribution_series = None

        charts = {
            "factor_exposure": chart_factor_exposure,
            "factor_contribution": chart_factor_contribution,
            "alpha_comparison": chart_alpha_comparison,
            "cumulative_factor_return": chart_cumulative_factor,
            "factor_contribution_series": chart_contribution_series,
        }

        # combined legacy 2x2 plot (fit + exposures + cumulative + contribution) — kept for the
        # existing "Full Factor Diagnostics" panel so older callers relying on top-level `plot` still work
        plot = None
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.5), dpi=120)
            (ax1, ax2), (ax3, ax4) = axes
            fl = loadings[1:]
            xs = [f["name"] for f in fl]
            ys = [f["coef"] for f in fl]
            cols = [BLUE if f["significant"] else GRAY for f in fl]
            ax1.bar(xs, ys, color=cols)
            ax1.axhline(0, color="#111827", lw=0.8)
            for i, f in enumerate(fl):
                ax1.text(i, f["coef"], f"{f['coef']:.2f}{'*' if f['significant'] else ''}",
                         ha="center", va="bottom" if f["coef"] >= 0 else "top", fontsize=8)
            ax1.set_ylabel("Factor loading (β)")
            ax1.set_title("Factor exposures (blue = significant)")
            ax1.tick_params(axis="x", rotation=25)
            ax2.scatter(fitted, yv, s=14, color=BLUE, alpha=0.5)
            lo, hi = float(min(fitted.min(), yv.min())), float(max(fitted.max(), yv.max()))
            ax2.plot([lo, hi], [lo, hi], "--", color=RED, lw=1)
            ax2.set_xlabel("Fitted excess return"); ax2.set_ylabel("Actual excess return")
            ax2.set_title(f"Fit (R² = {model.rsquared:.2f})")
            for i, c in enumerate(factor_cols):
                ax3.plot(cumulative_factor_return[c], color=PALETTE[i % len(PALETTE)], lw=1.6, label=c)
            ax3.axhline(0, color="#cbd5e1", lw=0.6)
            ax3.set_xlabel("Period"); ax3.set_ylabel("Cumulative return")
            ax3.set_title("Cumulative factor return"); ax3.legend(fontsize=8, frameon=False)
            contrib_stack = np.array([factor_contribution_series[c] for c in factor_cols], dtype=float)
            xs_idx = np.arange(contrib_stack.shape[1]) if contrib_stack.size else np.array([])
            if contrib_stack.size:
                ax4.stackplot(xs_idx, contrib_stack, labels=factor_cols,
                               colors=[PALETTE[i % len(PALETTE)] for i in range(len(factor_cols))], alpha=0.8)
            ax4.axhline(0, color="#111827", lw=0.6)
            ax4.set_xlabel("Period"); ax4.set_ylabel("Contribution to fitted return")
            ax4.set_title("Factor return contribution"); ax4.legend(fontsize=8, frameon=False)
            fig.tight_layout()
            plot = _png(fig)
        except Exception:
            plt.close("all"); plot = None

        # narrative on the significant factors
        sig_factors = [f["name"] for f in loadings[1:] if f["significant"]]
        interpretation = (
            f"Regressing {excess_label} on {len(factor_cols)} factor(s) explains {model.rsquared:.0%} of its "
            f"variation (R²). The alpha is {alpha_annual:.2%} annualised"
            + (f" and is statistically significant (p={alpha_p:.3f}), suggesting return not explained by these "
               "factors — genuine skill, a missing factor, or luck." if alpha_p < 0.05 else
               f" but is not statistically distinguishable from zero (p={alpha_p:.3f}), which is the usual finding: "
               "once the factors are accounted for, there is little unexplained return.")
            + (f" The significant exposures are {', '.join(sig_factors)}, describing the systematic risks that drive "
               "this asset's returns." if sig_factors else " None of the factor loadings are statistically significant.")
        )
        if alpha_comparison:
            capm_row = next((m for m in alpha_comparison if m["model"] == "CAPM"), None)
            ff_row = next((m for m in alpha_comparison if m["model"] == "Fama-French"), None)
            if capm_row and ff_row:
                interpretation += (
                    f" Compared with a single-factor CAPM (alpha {capm_row['alpha_annual']:.2%}, R² "
                    f"{capm_row['r_squared']:.0%}), adding the extra factors moves alpha to {ff_row['alpha_annual']:.2%} "
                    f"and R² to {ff_row['r_squared']:.0%} — the additional factors absorb return CAPM had attributed to skill."
                )

        results = {
            "status": "ok", "asset": asset_col, "excess_label": excess_label,
            "factors": factor_cols, "n_obs": int(len(reg)), "n_factors": len(factor_cols),
            "market_factor": market_col,
            "r_squared": _fin(model.rsquared, 4), "adj_r_squared": _fin(model.rsquared_adj, 4),
            "alpha_period": _fin(alpha_period, 6), "alpha_annual": _fin(alpha_annual, 6),
            "alpha_p_value": _fin(alpha_p, 6), "alpha_significant": bool(alpha_p < 0.05),
            "loadings": loadings,
            "factor_premium": factor_premium,
            "f_stat": _fin(model.fvalue, 4), "f_p_value": _fin(model.f_pvalue, 6),
            "resid_std": _fin(float(np.std(resid, ddof=1)), 6),
            "interpretation": interpretation,
            "cumulative_factor_return": cumulative_factor_return,
            "factor_contribution_series": factor_contribution_series,
            # ① Factor Model Summary
            "factor_summary": factor_summary,
            # ② Factor Regression summary block (loadings itself is the regression table)
            "regression_summary": regression_summary,
            # ④ Factor Significance
            "factor_significance": factor_significance,
            # ⑤ Factor Contribution (return units)
            "factor_contribution_table": factor_contribution_table,
            "attribution": attribution,
            "total_return_annual": _fin(total_explained_return, 6),
            "explained_return_annual": _fin(total_factor_contribution, 6),
            # ⑥ Alpha Comparison
            "alpha_comparison": alpha_comparison,
            # diagnostics result summary
            "diagnostics": diagnostics,
            # charts (tabbed) — plus the legacy combined `plot`
            "charts": charts,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
