#!/usr/bin/env python3
"""Multifactor Models — regress a return on an arbitrary set of factors, with
model-building diagnostics: standardized betas, multicollinearity (VIF), and
each factor's marginal contribution to R². statsmodels.

Input (from multifactor-page.tsx):
    data             : list[dict]
    target_col       : str        dependent return column
    factor_cols      : string[]   factor columns (>= 1)
    rf_col           : str        (optional) risk-free; target -> excess
    periods_per_year : int        (default 252) to annualise alpha
Output: { results: {coefficients, std_betas, vif, contributions, ...}, plot }
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

        # VIF for each factor (regress each factor on the others)
        vif = {}
        for i, c in enumerate(factor_cols):
            others = [j for j in range(k) if j != i]
            if not others:
                vif[c] = _fin(1.0, 3)
                continue
            Xi = Xf[:, i]
            Xo = sm.add_constant(Xf[:, others])
            r2i = sm.OLS(Xi, Xo).fit().rsquared
            vif[c] = _fin(1.0 / (1.0 - r2i) if r2i < 0.9999 else 999.0, 3)

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

        alpha_annual = float(coefs[0]) * ppy
        max_vif = max([v for v in vif.values() if v is not None], default=0.0)
        collinear = max_vif >= 5.0

        # plot: standardized betas + contribution + fit + residuals (2x2 grid)
        fitted = model.fittedvalues
        resid = model.resid
        plot = None
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12.5, 10), dpi=120)
            fl = [c for c in coefficients if c["name"] != "alpha"]
            xs = [c["name"] for c in fl]
            sb = [c["std_beta"] or 0 for c in fl]
            colors = ["#2563eb" if c["significant"] else "#94a3b8" for c in fl]
            ax1.barh(xs, sb, color=colors)
            ax1.axvline(0, color="#111827", lw=0.8)
            ax1.set_xlabel("Standardized beta (comparable scale)")
            ax1.set_title("Factor importance (blue = significant)")
            contrib = [(c["contribution"] or 0) * 100 for c in fl]
            ax2.barh(xs, contrib, color="#16a34a")
            ax2.set_xlabel("Marginal contribution to R² (%)")
            ax2.set_title(f"R² contribution (total R² = {full_r2:.2f})")
            # actual vs fitted
            ax3.scatter(fitted, yv, s=14, color="#2563eb", alpha=0.5)
            lo, hi = float(min(fitted.min(), yv.min())), float(max(fitted.max(), yv.max()))
            ax3.plot([lo, hi], [lo, hi], "--", color="#dc2626", lw=1)
            ax3.set_xlabel("Predicted return"); ax3.set_ylabel("Actual return")
            ax3.set_title(f"Actual vs predicted (R² = {full_r2:.2f})")
            # residuals vs fitted
            ax4.scatter(fitted, resid, s=14, color="#f59e0b", alpha=0.5)
            ax4.axhline(0, color="#111827", lw=0.8, linestyle="--")
            ax4.set_xlabel("Predicted return"); ax4.set_ylabel("Residual")
            ax4.set_title("Residuals vs predicted")
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
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
            "alpha_period": _fin(float(coefs[0]), 6), "alpha_annual": _fin(alpha_annual, 6),
            "alpha_p_value": _fin(float(pvals[0]), 6), "alpha_significant": bool(pvals[0] < 0.05),
            "f_stat": _fin(model.fvalue, 4), "f_p_value": _fin(model.f_pvalue, 6),
            "max_vif": _fin(max_vif, 3), "multicollinearity": bool(collinear),
            "resid_std": _fin(float(np.std(resid, ddof=1)), 6),
            "coefficients": coefficients,
            "interpretation": interpretation,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
