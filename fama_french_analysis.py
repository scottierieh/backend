#!/usr/bin/env python3
"""Fama-French Factor Model — regress an asset's excess return on risk factors.
statsmodels OLS with full attribution, rolling exposures and residual diagnostics.

Works for the 3-factor (Mkt-RF, SMB, HML), 5-factor (+ RMW, CMA) or any custom
factor set: the caller simply selects which columns are the factors.

Input (from fama-french-page.tsx):
    data             : list[dict]
    asset_col        : str     asset/portfolio return column
    factor_cols      : string[]  factor return columns (e.g. Mkt-RF, SMB, HML)
    rf_col           : str     (optional) risk-free column; excess = asset - rf
    periods_per_year : int     (default 252) to annualise alpha & attribution
Output: { results: {loadings, alpha, attribution, rolling, diagnostics, ...}, plot }
"""
import sys, json, io, base64
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.stattools import durbin_watson, jarque_bera
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy import stats as sps

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _fin(x, nd=6):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def _exposure_label(name, coef, significant):
    """Map a factor loading to a human-readable style tag."""
    n = (name or "").lower()
    c = coef
    if not significant:
        return {"level": "Neutral", "level_ko": "중립", "desc_en": "no significant exposure", "desc_ko": "유의한 노출 없음"}
    is_market = any(k in n for k in ["mkt", "market", "mktrf", "mkt-rf", "rm", "erm"])
    if is_market:
        if c > 1.1:
            return {"level": "High", "level_ko": "높음", "desc_en": "more volatile than the market", "desc_ko": "시장보다 변동성 큼"}
        if c < 0.9:
            return {"level": "Low", "level_ko": "낮음", "desc_en": "less volatile than the market", "desc_ko": "시장보다 변동성 작음"}
        return {"level": "Market-like", "level_ko": "시장 수준", "desc_en": "moves roughly with the market", "desc_ko": "시장과 유사하게 움직임"}
    pos_en, neg_en, pos_ko, neg_ko = "Positive", "Negative", "양(+)", "음(−)"
    hint = {
        "smb": ("small-cap tilt", "large-cap tilt", "소형주 성향", "대형주 성향"),
        "hml": ("value tilt", "growth tilt", "가치주 성향", "성장주 성향"),
        "rmw": ("robust profitability", "weak profitability", "높은 수익성", "낮은 수익성"),
        "cma": ("conservative investment", "aggressive investment", "보수적 투자", "공격적 투자"),
        "mom": ("momentum tilt", "contrarian tilt", "모멘텀 성향", "역행 성향"),
        "wml": ("momentum tilt", "contrarian tilt", "모멘텀 성향", "역행 성향"),
    }
    key = next((k for k in hint if k in n), None)
    if key:
        pe, ne, pk, nk = hint[key]
        return ({"level": pos_en, "level_ko": pos_ko, "desc_en": pe, "desc_ko": pk} if c > 0
                else {"level": neg_en, "level_ko": neg_ko, "desc_en": ne, "desc_ko": nk})
    return ({"level": pos_en, "level_ko": pos_ko, "desc_en": "positive exposure", "desc_ko": "양의 노출"} if c > 0
            else {"level": neg_en, "level_ko": neg_ko, "desc_en": "negative exposure", "desc_ko": "음의 노출"})


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

        yv = reg["_y"].values.astype(float)
        Fv = reg[factor_cols].values.astype(float)
        Xv = sm.add_constant(Fv)
        model = sm.OLS(yv, Xv).fit()

        names = ["alpha"] + factor_cols
        coefs = model.params
        tvals = model.tvalues
        pvals = model.pvalues
        ses = model.bse
        conf = model.conf_int()

        loadings = []
        for i, nm in enumerate(names):
            entry = {
                "name": nm,
                "coef": _fin(coefs[i], 6),
                "std_err": _fin(ses[i], 6),
                "t_stat": _fin(tvals[i], 4),
                "p_value": _fin(pvals[i], 6),
                "ci_low": _fin(conf[i][0], 6),
                "ci_high": _fin(conf[i][1], 6),
                "significant": bool(pvals[i] < 0.05),
            }
            if i > 0:
                entry["exposure"] = _exposure_label(nm, float(coefs[i]), entry["significant"])
            loadings.append(entry)

        alpha_period = float(coefs[0])
        alpha_annual = alpha_period * ppy
        alpha_p = float(pvals[0])

        # ---- Performance attribution: contribution_i = beta_i * mean(factor_i) ----
        factor_means = Fv.mean(axis=0)  # per-period mean factor return
        attribution = []
        explained_period = 0.0
        for j, nm in enumerate(factor_cols):
            contrib_period = float(coefs[j + 1]) * float(factor_means[j])
            explained_period += contrib_period
            attribution.append({
                "name": nm,
                "beta": _fin(coefs[j + 1], 4),
                "factor_mean_annual": _fin(float(factor_means[j]) * ppy, 6),
                "contribution": _fin(contrib_period * ppy, 6),  # annualised
            })
        attribution.append({"name": "Alpha", "beta": None, "factor_mean_annual": None,
                            "contribution": _fin(alpha_annual, 6)})
        total_return_annual = (explained_period + alpha_period) * ppy  # == mean(y)*ppy
        explained_annual = explained_period * ppy
        for a in attribution:
            c = a["contribution"]
            a["pct_of_total"] = _fin((c / total_return_annual) if (c is not None and abs(total_return_annual) > 1e-12) else None, 4)

        # ---- Variance decomposition (explained vs residual) ----
        r2 = float(model.rsquared)
        var_decomp = {"explained": _fin(r2, 4), "unexplained": _fin(1 - r2, 4)}

        # ---- Rolling factor exposures ----
        n = len(reg)
        window = int(max(min(len(factor_cols) + 15, n), min(60, max(20, n // 4))))
        window = min(window, n)
        rolling = {"window": window, "points": []}
        if n >= window + 5:
            step = max(1, (n - window) // 150)  # cap ~150 points
            Xr = sm.add_constant(Fv)
            for end in range(window, n + 1, step):
                s = end - window
                xb = Xr[s:end]
                yb = yv[s:end]
                try:
                    beta, *_ = np.linalg.lstsq(xb, yb, rcond=None)
                    pt = {"t": end - 1, "alpha": _fin(beta[0] * ppy, 6)}
                    for j, nm in enumerate(factor_cols):
                        pt[nm] = _fin(beta[j + 1], 4)
                    rolling["points"].append(pt)
                except Exception:
                    continue

        # ---- Residual diagnostics ----
        fitted = np.asarray(model.fittedvalues, dtype=float)
        resid = np.asarray(model.resid, dtype=float)
        dw = float(durbin_watson(resid))
        jb_stat, jb_p, jb_skew, jb_kurt = jarque_bera(resid)
        try:
            bp_stat, bp_p, _, _ = het_breuschpagan(resid, Xv)
        except Exception:
            bp_stat, bp_p = np.nan, np.nan
        diagnostics = {
            "durbin_watson": _fin(dw, 4),
            "dw_note_en": "no autocorrelation" if 1.5 <= dw <= 2.5 else ("positive autocorrelation" if dw < 1.5 else "negative autocorrelation"),
            "dw_note_ko": "자기상관 없음" if 1.5 <= dw <= 2.5 else ("양의 자기상관" if dw < 1.5 else "음의 자기상관"),
            "jarque_bera": _fin(jb_stat, 4), "jb_p_value": _fin(jb_p, 6),
            "residuals_normal": bool(jb_p > 0.05),
            "skew": _fin(jb_skew, 4), "kurtosis": _fin(jb_kurt, 4),
            "breusch_pagan": _fin(bp_stat, 4), "bp_p_value": _fin(bp_p, 6),
            "homoskedastic": bool(np.isfinite(bp_p) and bp_p > 0.05),
        }

        # ---- Comprehensive 2x3 figure ----
        plot = None
        try:
            fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), dpi=110)
            fl = loadings[1:]
            # (1) factor loadings bar
            ax = axes[0, 0]
            xs = [f["name"] for f in fl]; ys = [f["coef"] for f in fl]
            cols = ["#2563eb" if f["significant"] else "#94a3b8" for f in fl]
            ax.bar(xs, ys, color=cols)
            ax.axhline(0, color="#111827", lw=0.8)
            for i, f in enumerate(fl):
                ax.text(i, f["coef"], f"{f['coef']:.2f}{'*' if f['significant'] else ''}",
                        ha="center", va="bottom" if f["coef"] >= 0 else "top", fontsize=8)
            ax.set_ylabel("Loading (β)"); ax.set_title("1. Factor exposures (blue = sig.)")
            ax.tick_params(axis="x", rotation=25)
            # (2) performance attribution
            ax = axes[0, 1]
            an = [a["name"] for a in attribution]
            ac = [(a["contribution"] or 0) * 100 for a in attribution]
            acol = ["#16a34a" if v >= 0 else "#dc2626" for v in ac]
            acol[-1] = "#7c3aed"  # alpha
            ax.bar(an, ac, color=acol)
            ax.axhline(0, color="#111827", lw=0.8)
            for i, v in enumerate(ac):
                ax.text(i, v, f"{v:.1f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
            ax.set_ylabel("Annual contribution (%)"); ax.set_title("2. Return attribution")
            ax.tick_params(axis="x", rotation=25)
            # (3) explained vs unexplained variance pie
            ax = axes[0, 2]
            ax.pie([r2, max(1 - r2, 0)], labels=[f"Explained\n{r2*100:.0f}%", f"Idiosyncratic\n{(1-r2)*100:.0f}%"],
                   colors=["#2563eb", "#e5e7eb"], startangle=90, wedgeprops={"width": 0.42},
                   textprops={"fontsize": 9})
            ax.set_title("3. Variance explained (R²)")
            # (4) rolling betas
            ax = axes[1, 0]
            if rolling["points"]:
                tx = [pt["t"] for pt in rolling["points"]]
                for nm in factor_cols:
                    ax.plot(tx, [pt.get(nm) for pt in rolling["points"]], lw=1.4, label=nm)
                ax.axhline(0, color="#111827", lw=0.6)
                ax.legend(fontsize=7, ncol=2); ax.set_xlabel("Period")
                ax.set_ylabel("Rolling β"); ax.set_title(f"4. Rolling exposures (w={window})")
            else:
                ax.text(0.5, 0.5, "Not enough data\nfor rolling window", ha="center", va="center", fontsize=10)
                ax.set_title("4. Rolling exposures"); ax.axis("off")
            # (5) actual vs fitted
            ax = axes[1, 1]
            ax.scatter(fitted, yv, s=12, color="#2563eb", alpha=0.45)
            lo, hi = float(min(fitted.min(), yv.min())), float(max(fitted.max(), yv.max()))
            ax.plot([lo, hi], [lo, hi], "--", color="#dc2626", lw=1)
            ax.set_xlabel("Fitted excess return"); ax.set_ylabel("Actual excess return")
            ax.set_title(f"5. Actual vs predicted (R²={r2:.2f})")
            # (6) residual QQ plot
            ax = axes[1, 2]
            sps.probplot(resid, dist="norm", plot=ax)
            ax.get_lines()[0].set_markerfacecolor("#2563eb"); ax.get_lines()[0].set_markersize(4.0)
            ax.get_lines()[0].set_markeredgecolor("none"); ax.get_lines()[1].set_color("#dc2626")
            ax.set_title("6. Residual Q-Q plot")
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        # ---- exposure summary (headline tags) ----
        exposure_summary = [{"name": l["name"], "coef": l["coef"], "significant": l["significant"], **l["exposure"]}
                            for l in loadings[1:]]

        sig_factors = [f["name"] for f in loadings[1:] if f["significant"]]
        interpretation = (
            f"Regressing {excess_label} on {len(factor_cols)} factor(s) explains {r2:.0%} of its "
            f"variation (R²). The alpha is {alpha_annual:.2%} annualised"
            + (f" and is statistically significant (p={alpha_p:.3f}), suggesting return not explained by these "
               "factors — genuine skill, a missing factor, or luck." if alpha_p < 0.05 else
               f" but is not statistically distinguishable from zero (p={alpha_p:.3f}), which is the usual finding: "
               "once the factors are accounted for, there is little unexplained return.")
            + (f" The significant exposures are {', '.join(sig_factors)}, describing the systematic risks that drive "
               "this asset's returns." if sig_factors else " None of the factor loadings are statistically significant.")
            + f" Attribution assigns {explained_annual:.1%} of the {total_return_annual:.1%} annual return to factor "
              f"exposures and {alpha_annual:.1%} to alpha."
        )

        results = {
            "status": "ok", "asset": asset_col, "excess_label": excess_label,
            "factors": factor_cols, "n_obs": int(n), "n_factors": len(factor_cols),
            "periods_per_year": ppy,
            "r_squared": _fin(r2, 4), "adj_r_squared": _fin(model.rsquared_adj, 4),
            "alpha_period": _fin(alpha_period, 6), "alpha_annual": _fin(alpha_annual, 6),
            "alpha_p_value": _fin(alpha_p, 6), "alpha_significant": bool(alpha_p < 0.05),
            "alpha_std_err": _fin(float(ses[0]), 6),
            "loadings": loadings,
            "attribution": attribution,
            "total_return_annual": _fin(total_return_annual, 6),
            "explained_return_annual": _fin(explained_annual, 6),
            "var_decomp": var_decomp,
            "rolling": rolling,
            "diagnostics": diagnostics,
            "exposure_summary": exposure_summary,
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
