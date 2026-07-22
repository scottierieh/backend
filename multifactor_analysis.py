#!/usr/bin/env python3
"""Multifactor Models — regress a return on an arbitrary, user-chosen set of
candidate factors (market return, rates, inflation, GDP, oil, FX, VIX,
industry factors, custom uploads, ...). Unlike Fama-French (which validates a
fixed theoretical factor set), this is an exploratory model-building tool: the
user picks any numeric columns as candidates and this analysis reports which
of them actually explain the target return. statsmodels OLS.

Sections produced (mirrors the approved 9-section spec):
  1 Model Summary        -> results.model_summary
  2 Factor Selection      -> (UI-only; the existing factor multi-select)
  3 Factor Regression     -> results.coefficients (unchanged)
  4 Factor Exposure       -> results.exposure_table + charts.factor_exposure
  5 Factor Significance   -> results.significance_table
  6 Factor Contribution   -> results.contribution_table + charts.factor_contribution
  7 Model Fit             -> results.fit_table + charts.actual_vs_fitted
  8 Residual Analysis     -> results.residual_table + charts.residual_timeseries (+ residuals_vs_fitted)
  9 Multicollinearity     -> results.vif_table + charts.correlation_matrix
  10 Model Comparison     -> results.model_comparison (nested ladder)

Input (from multifactor-page.tsx):
    data             : list[dict]
    target_col       : str        dependent return column
    factor_cols      : string[]   factor columns (>= 1), in the order selected
    rf_col           : str        (optional) risk-free; target -> excess
    periods_per_year : int        (default 252) to annualise alpha / factor means
Output: { results: {...}, plot, charts (also nested at results.charts) }
"""
import sys, json, io, base64
import numpy as np
import pandas as pd
import statsmodels.api as sm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#2563eb"
GREEN = "#16a34a"
RED = "#dc2626"
AMBER = "#f59e0b"
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


def _stars(p):
    if p is None:
        return "NS"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "NS"


def main():
    try:
        p = json.load(sys.stdin)
        rows = p.get("data") or []
        if not rows:
            raise ValueError("No data provided.")
        df = pd.DataFrame(rows)
        target_col = p.get("target_col")
        factor_cols = [c for c in (p.get("factor_cols") or []) if c in df.columns]
        rf_col = p.get("rf_col") or None
        ppy = int(p.get("periods_per_year") or 252)
        if not target_col or target_col not in df.columns:
            raise ValueError("Select the target return column.")
        if len(factor_cols) < 1:
            raise ValueError("Select at least one factor column.")

        y = pd.to_numeric(df[target_col], errors="coerce")
        if rf_col and rf_col in df.columns:
            rf = pd.to_numeric(df[rf_col], errors="coerce")
            y = y - rf
            target_label = f"{target_col} - {rf_col}"
        else:
            target_label = target_col
        X = df[factor_cols].apply(pd.to_numeric, errors="coerce")

        reg = pd.concat([y.rename("_y"), X], axis=1).dropna().reset_index(drop=True)
        k = len(factor_cols)
        if len(reg) < k + 5:
            raise ValueError(f"Need at least {k + 5} aligned observations for {k} factors.")

        yv = reg["_y"].values
        Xf = reg[factor_cols].values
        Xc = sm.add_constant(Xf)
        model = sm.OLS(yv, Xc).fit()

        # standardized betas: coef * sd(x)/sd(y)
        sd_y = float(np.std(yv, ddof=1))
        sd_x = np.std(Xf, axis=0, ddof=1)
        std_betas = {}
        for i, c in enumerate(factor_cols):
            b = float(model.params[i + 1])
            std_betas[c] = _fin((b * sd_x[i] / sd_y) if sd_y > 0 else 0.0, 4)

        # VIF for each factor (regress each factor on the others) — used both
        # for the coefficients table and the ⑨ multicollinearity table
        vif = {}
        vif_note = {}
        for i, c in enumerate(factor_cols):
            others = [j for j in range(k) if j != i]
            if not others:
                vif[c] = _fin(1.0, 3)
                vif_note[c] = "ok"
                continue
            try:
                Xi = Xf[:, i]
                Xo = sm.add_constant(Xf[:, others])
                r2i = sm.OLS(Xi, Xo).fit().rsquared
                v = 1.0 / (1.0 - r2i) if r2i < 0.9999 else 999.0
                vif[c] = _fin(v, 3)
                vif_note[c] = "very high (>50)" if v > 50 else ("elevated (>5)" if v > 5 else "ok")
            except Exception:
                vif[c] = None
                vif_note[c] = "failed to compute"

        # marginal R2 contribution: full R2 - R2 without that factor
        full_r2 = float(model.rsquared)
        contributions = {}
        for i, c in enumerate(factor_cols):
            others = [j for j in range(k) if j != i]
            if others:
                Xo = sm.add_constant(Xf[:, others])
                r2_wo = sm.OLS(yv, Xo).fit().rsquared
            else:
                r2_wo = 0.0
            contributions[c] = _fin(max(full_r2 - r2_wo, 0.0), 5)

        coefs, tvals, pvals = model.params, model.tvalues, model.pvalues
        conf = model.conf_int()
        names = ["alpha"] + factor_cols
        coefficients = []
        for i, nm in enumerate(names):
            coefficients.append({
                "name": nm, "coef": _fin(coefs[i], 6), "t_stat": _fin(tvals[i], 4),
                "p_value": _fin(pvals[i], 6), "ci_low": _fin(conf[i][0], 6), "ci_high": _fin(conf[i][1], 6),
                "std_beta": (std_betas.get(nm) if nm != "alpha" else None),
                "vif": (vif.get(nm) if nm != "alpha" else None),
                "contribution": (contributions.get(nm) if nm != "alpha" else None),
                "significant": bool(pvals[i] < 0.05),
            })

        alpha_period = float(coefs[0])
        alpha_annual = alpha_period * ppy
        max_vif = max([v for v in vif.values() if v is not None], default=0.0)
        collinear = max_vif >= 5.0

        fitted = model.fittedvalues
        resid = model.resid
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        mae = float(np.mean(np.abs(resid)))
        resid_mean = float(np.mean(resid))
        resid_std = float(np.std(resid, ddof=1))
        aic = float(model.aic)
        bic = float(model.bic)

        # ─────────────────────── ① Model Summary ───────────────────────
        model_summary = {
            "alpha_annual": _fin(alpha_annual, 6),
            "alpha_p_value": _fin(float(pvals[0]), 6),
            "r_squared": _fin(full_r2, 4),
            "adj_r_squared": _fin(model.rsquared_adj, 4),
            "n_obs": int(len(reg)),
            "f_stat": _fin(model.fvalue, 4),
            "f_p_value": _fin(model.f_pvalue, 6),
            "aic": _fin(aic, 2),
            "bic": _fin(bic, 2),
        }

        # ─────────────────────── ④ Factor Exposure ───────────────────────
        exposure_table = []
        for i, c in enumerate(factor_cols):
            coef_raw = float(coefs[i + 1])
            exposure_table.append({
                "name": c,
                "coef": _fin(coef_raw, 6),
                "std_beta": std_betas.get(c),
                "direction": "Positive" if coef_raw >= 0 else "Negative",
            })

        # ─────────────────────── ⑤ Factor Significance ───────────────────────
        significance_table = []
        for i, c in enumerate(factor_cols):
            pv = float(pvals[i + 1])
            significance_table.append({
                "name": c,
                "coef": _fin(float(coefs[i + 1]), 6),
                "p_value": _fin(pv, 6),
                "significance": _stars(pv),
                "significant": bool(pv < 0.05),
            })

        # ─────────────────────── ⑥ Factor Contribution ───────────────────────
        # Same pattern as fama_french_analysis.py's factor_premium / factor_contribution_table:
        # contribution = coefficient * the factor's realised (annualised) mean over the sample,
        # NOT the raw coefficient itself.
        contribution_table = []
        total_factor_contribution = 0.0
        for i, c in enumerate(factor_cols):
            coef_raw = float(coefs[i + 1])
            factor_mean_annual = float(reg[c].mean()) * ppy
            contrib = coef_raw * factor_mean_annual
            total_factor_contribution += contrib
            contribution_table.append({
                "name": c, "coef": _fin(coef_raw, 6),
                "factor_mean_annual": _fin(factor_mean_annual, 6),
                "contribution": _fin(contrib, 6),
            })
        total_return = total_factor_contribution + alpha_annual
        contribution_table.append({
            "name": "Alpha", "coef": None, "factor_mean_annual": None,
            "contribution": _fin(alpha_annual, 6),
        })
        contribution_table.append({
            "name": "Total", "coef": None, "factor_mean_annual": None,
            "contribution": _fin(total_return, 6),
        })

        # ─────────────────────── ⑦ Model Fit ───────────────────────
        fit_table = {
            "r_squared": _fin(full_r2, 4),
            "adj_r_squared": _fin(model.rsquared_adj, 4),
            "rmse": _fin(rmse, 6),
            "mae": _fin(mae, 6),
            "aic": _fin(aic, 2),
            "bic": _fin(bic, 2),
        }

        # ─────────────────────── ⑧ Residual Analysis ───────────────────────
        residual_table = {
            "resid_mean": _fin(resid_mean, 6),
            "resid_std": _fin(resid_std, 6),
            "rmse": _fin(rmse, 6),
            "mae": _fin(mae, 6),
        }

        # ─────────────────────── ⑨ Multicollinearity ───────────────────────
        vif_table = [
            {"name": c, "vif": vif.get(c), "note": vif_note.get(c, "ok")}
            for c in factor_cols
        ]
        corr_matrix_df = reg[factor_cols].corr()
        corr_matrix = {
            "labels": factor_cols,
            "matrix": [[_fin(v, 4) for v in row] for row in corr_matrix_df.values.tolist()],
        }

        # ─────────────────────── ⑩ Model Comparison (nested ladder) ───────────────────────
        # Model 1 = first factor alone, Model 2 = first two factors, ... Model N = all factors,
        # refitting OLS at each step in the order the user selected the factors. If there are
        # many factors (>6) showing every step makes an unwieldy table, so we cap the ladder to
        # step 1, the midpoint, and the full model (a reasonable compromise between granularity
        # and table size — full detail for small factor sets, a representative sample otherwise).
        if k <= 6:
            steps = list(range(1, k + 1))
        else:
            mid = max(2, k // 2)
            steps = sorted(set([1, mid, k]))
        model_comparison = []
        for s in steps:
            sub_cols = factor_cols[:s]
            Xs = sm.add_constant(reg[sub_cols].values)
            ms = sm.OLS(yv, Xs).fit()
            model_comparison.append({
                "model": f"Model {s}" + (" (full)" if s == k else ""),
                "factors": ", ".join(sub_cols),
                "n_factors": s,
                "r_squared": _fin(ms.rsquared, 4),
                "adj_r_squared": _fin(ms.rsquared_adj, 4),
                "aic": _fin(ms.aic, 2),
                "bic": _fin(ms.bic, 2),
            })

        # ─────────────────────────────── Charts (each its own PNG) ───────────────────────────────
        fl = [c for c in coefficients if c["name"] != "alpha"]
        xs_names = [c["name"] for c in fl]

        chart_factor_exposure = None
        try:
            sb = [c["std_beta"] or 0 for c in fl]
            colors = [BLUE if c["significant"] else GRAY for c in fl]
            fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=115)
            ax.barh(xs_names, sb, color=colors)
            ax.axvline(0, color="#111827", lw=0.8)
            ax.set_xlabel("Standardized beta (comparable scale)")
            ax.set_title("Factor Exposure — standardized beta (blue = significant)")
            fig.tight_layout()
            chart_factor_exposure = _png(fig)
        except Exception:
            plt.close("all"); chart_factor_exposure = None

        chart_r2_contribution = None
        try:
            contrib_pct = [(c["contribution"] or 0) * 100 for c in fl]
            fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=115)
            ax.barh(xs_names, contrib_pct, color=GREEN)
            ax.set_xlabel("Marginal contribution to R² (%)")
            ax.set_title(f"R² Contribution (total R² = {full_r2:.2f})")
            fig.tight_layout()
            chart_r2_contribution = _png(fig)
        except Exception:
            plt.close("all"); chart_r2_contribution = None

        chart_actual_vs_fitted = None
        try:
            fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=115)
            ax.scatter(fitted, yv, s=14, color=BLUE, alpha=0.5)
            lo, hi = float(min(fitted.min(), yv.min())), float(max(fitted.max(), yv.max()))
            ax.plot([lo, hi], [lo, hi], "--", color=RED, lw=1)
            ax.set_xlabel("Predicted return"); ax.set_ylabel("Actual return")
            ax.set_title(f"Actual vs Fitted Return (R² = {full_r2:.2f})")
            fig.tight_layout()
            chart_actual_vs_fitted = _png(fig)
        except Exception:
            plt.close("all"); chart_actual_vs_fitted = None

        chart_residuals_vs_fitted = None
        try:
            fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=115)
            ax.scatter(fitted, resid, s=14, color=AMBER, alpha=0.5)
            ax.axhline(0, color="#111827", lw=0.8, linestyle="--")
            ax.set_xlabel("Predicted return"); ax.set_ylabel("Residual")
            ax.set_title("Residuals vs Fitted")
            fig.tight_layout()
            chart_residuals_vs_fitted = _png(fig)
        except Exception:
            plt.close("all"); chart_residuals_vs_fitted = None

        chart_factor_contribution = None
        try:
            rows_c = [row for row in contribution_table if row["name"] != "Total"]
            xs_c = [row["name"] for row in rows_c]
            ys_c = [(row["contribution"] or 0) * 100 for row in rows_c]
            colors = [GREEN if v >= 0 else RED for v in ys_c]
            fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=115)
            ax.bar(xs_c, ys_c, color=colors)
            ax.axhline(0, color="#111827", lw=0.7)
            ax.set_ylabel("Contribution to annual return (%)")
            ax.set_title("Factor Contribution to Return")
            ax.tick_params(axis="x", rotation=25)
            fig.tight_layout()
            chart_factor_contribution = _png(fig)
        except Exception:
            plt.close("all"); chart_factor_contribution = None

        chart_residual_timeseries = None
        try:
            fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=115)
            ax.plot(np.arange(len(resid)), resid, color=BLUE, lw=1.1)
            ax.axhline(0, color="#111827", lw=0.8, linestyle="--")
            ax.set_xlabel("Period"); ax.set_ylabel("Residual")
            ax.set_title("Residual Time Series")
            fig.tight_layout()
            chart_residual_timeseries = _png(fig)
        except Exception:
            plt.close("all"); chart_residual_timeseries = None

        chart_correlation_matrix = None
        try:
            cm = corr_matrix_df.values
            fig, ax = plt.subplots(figsize=(max(5.5, 1.0 * k + 2), max(4.5, 1.0 * k + 1.5)), dpi=115)
            im = ax.imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1)
            ax.set_xticks(range(k)); ax.set_xticklabels(factor_cols, rotation=35, ha="right", fontsize=8)
            ax.set_yticks(range(k)); ax.set_yticklabels(factor_cols, fontsize=8)
            for i in range(k):
                for j in range(k):
                    ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                            color="white" if abs(cm[i, j]) > 0.6 else "black", fontsize=8)
            ax.set_title("Factor Correlation Matrix")
            fig.colorbar(im, ax=ax, shrink=0.8)
            fig.tight_layout()
            chart_correlation_matrix = _png(fig)
        except Exception:
            plt.close("all"); chart_correlation_matrix = None

        charts = {
            "factor_exposure": chart_factor_exposure,
            "r2_contribution": chart_r2_contribution,
            "actual_vs_fitted": chart_actual_vs_fitted,
            "residuals_vs_fitted": chart_residuals_vs_fitted,
            "factor_contribution": chart_factor_contribution,
            "residual_timeseries": chart_residual_timeseries,
            "correlation_matrix": chart_correlation_matrix,
        }

        # combined legacy 2x2 plot kept for backward compatibility with any earlier caller
        # relying on the top-level `plot` (built from the same 4 panels as before)
        plot = None
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12.5, 10), dpi=120)
            sb = [c["std_beta"] or 0 for c in fl]
            colors = [BLUE if c["significant"] else GRAY for c in fl]
            ax1.barh(xs_names, sb, color=colors)
            ax1.axvline(0, color="#111827", lw=0.8)
            ax1.set_xlabel("Standardized beta (comparable scale)")
            ax1.set_title("Factor importance (blue = significant)")
            contrib_pct = [(c["contribution"] or 0) * 100 for c in fl]
            ax2.barh(xs_names, contrib_pct, color=GREEN)
            ax2.set_xlabel("Marginal contribution to R² (%)")
            ax2.set_title(f"R² contribution (total R² = {full_r2:.2f})")
            ax3.scatter(fitted, yv, s=14, color=BLUE, alpha=0.5)
            lo, hi = float(min(fitted.min(), yv.min())), float(max(fitted.max(), yv.max()))
            ax3.plot([lo, hi], [lo, hi], "--", color=RED, lw=1)
            ax3.set_xlabel("Predicted return"); ax3.set_ylabel("Actual return")
            ax3.set_title(f"Actual vs predicted (R² = {full_r2:.2f})")
            ax4.scatter(fitted, resid, s=14, color=AMBER, alpha=0.5)
            ax4.axhline(0, color="#111827", lw=0.8, linestyle="--")
            ax4.set_xlabel("Predicted return"); ax4.set_ylabel("Residual")
            ax4.set_title("Residuals vs predicted")
            fig.tight_layout()
            plot = _png(fig)
        except Exception:
            plt.close("all"); plot = None

        sig = [c["name"] for c in coefficients if c["name"] != "alpha" and c["significant"]]
        interpretation = (
            f"The {k}-factor model explains {full_r2:.0%} of {target_label}'s variation. "
            + (f"Ranked by standardized beta, the factors that matter most are "
               f"{', '.join(sorted(sig, key=lambda c: -abs(std_betas.get(c, 0) or 0)))}. "
               if sig else "None of the factors are individually significant. ")
            + (f"Watch out for multicollinearity: the highest VIF is {max_vif:.1f}, so at least one factor is highly "
               "correlated with the others, which makes its individual coefficient unstable and hard to interpret. "
               if collinear else f"Multicollinearity is mild (max VIF {max_vif:.1f}), so the individual coefficients "
               "are reasonably stable. ")
            + f"The annualised alpha is {alpha_annual:.2%}."
        )

        results = {
            "status": "ok", "target": target_col, "target_label": target_label, "factors": factor_cols,
            "n_obs": int(len(reg)), "n_factors": k,
            "r_squared": _fin(full_r2, 4), "adj_r_squared": _fin(model.rsquared_adj, 4),
            "alpha_period": _fin(alpha_period, 6), "alpha_annual": _fin(alpha_annual, 6),
            "alpha_p_value": _fin(float(pvals[0]), 6), "alpha_significant": bool(pvals[0] < 0.05),
            "f_stat": _fin(model.fvalue, 4), "f_p_value": _fin(model.f_pvalue, 6),
            "aic": _fin(aic, 2), "bic": _fin(bic, 2),
            "max_vif": _fin(max_vif, 3), "multicollinearity": bool(collinear),
            "resid_std": _fin(resid_std, 6),
            "coefficients": coefficients,
            "interpretation": interpretation,
            # ① Model Summary
            "model_summary": model_summary,
            # ④ Factor Exposure
            "exposure_table": exposure_table,
            # ⑤ Factor Significance
            "significance_table": significance_table,
            # ⑥ Factor Contribution
            "contribution_table": contribution_table,
            "total_return_annual": _fin(total_return, 6),
            # ⑦ Model Fit
            "fit_table": fit_table,
            # ⑧ Residual Analysis
            "residual_table": residual_table,
            # ⑨ Multicollinearity
            "vif_table": vif_table,
            "corr_matrix": corr_matrix,
            # ⑩ Model Comparison (nested ladder)
            "model_comparison": model_comparison,
            # charts (tabbed) — plus the legacy combined `plot`
            "charts": charts,
        }
        print(json.dumps({"results": results, "plot": plot, "charts": charts}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
