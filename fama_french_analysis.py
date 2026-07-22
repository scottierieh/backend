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
Output: { results: {loadings, alpha, r_squared, ...}, plot }
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
        tvals = model.tvalues
        pvals = model.pvalues
        conf = model.conf_int()

        loadings = []
        for i, nm in enumerate(names):
            loadings.append({
                "name": nm,
                "coef": _fin(coefs[i], 6),
                "t_stat": _fin(tvals[i], 4),
                "p_value": _fin(pvals[i], 6),
                "ci_low": _fin(conf[i][0], 6),
                "ci_high": _fin(conf[i][1], 6),
                "significant": bool(pvals[i] < 0.05),
            })

        alpha_period = float(coefs[0])
        alpha_annual = alpha_period * ppy
        alpha_p = float(pvals[0])

        # factor premium: mean period/annualised return of each factor itself
        factor_premium = []
        for c in factor_cols:
            mean_period = float(reg[c].mean())
            factor_premium.append({
                "name": c,
                "mean_period": _fin(mean_period, 6),
                "annualized": _fin(mean_period * ppy, 6),
            })

        # predicted vs actual for plot
        fitted = model.fittedvalues
        resid = model.resid

        plot = None
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5), dpi=120)
            # factor loadings bar (exclude alpha)
            fl = loadings[1:]
            xs = [f["name"] for f in fl]
            ys = [f["coef"] for f in fl]
            cols = ["#2563eb" if f["significant"] else "#94a3b8" for f in fl]
            ax1.bar(xs, ys, color=cols)
            ax1.axhline(0, color="#111827", lw=0.8)
            for i, f in enumerate(fl):
                ax1.text(i, f["coef"], f"{f['coef']:.2f}{'*' if f['significant'] else ''}",
                         ha="center", va="bottom" if f["coef"] >= 0 else "top", fontsize=8)
            ax1.set_ylabel("Factor loading (β)")
            ax1.set_title("Factor exposures (blue = significant)")
            ax1.tick_params(axis="x", rotation=25)
            # actual vs fitted
            ax2.scatter(fitted, yv, s=14, color="#2563eb", alpha=0.5)
            lo, hi = float(min(fitted.min(), yv.min())), float(max(fitted.max(), yv.max()))
            ax2.plot([lo, hi], [lo, hi], "--", color="#dc2626", lw=1)
            ax2.set_xlabel("Fitted excess return"); ax2.set_ylabel("Actual excess return")
            ax2.set_title(f"Fit (R² = {model.rsquared:.2f})")
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
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

        results = {
            "status": "ok", "asset": asset_col, "excess_label": excess_label,
            "factors": factor_cols, "n_obs": int(len(reg)), "n_factors": len(factor_cols),
            "r_squared": _fin(model.rsquared, 4), "adj_r_squared": _fin(model.rsquared_adj, 4),
            "alpha_period": _fin(alpha_period, 6), "alpha_annual": _fin(alpha_annual, 6),
            "alpha_p_value": _fin(alpha_p, 6), "alpha_significant": bool(alpha_p < 0.05),
            "loadings": loadings,
            "factor_premium": factor_premium,
            "f_stat": _fin(model.fvalue, 4), "f_p_value": _fin(model.f_pvalue, 6),
            "resid_std": _fin(float(np.std(resid, ddof=1)), 6),
            "interpretation": interpretation,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
