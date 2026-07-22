#!/usr/bin/env python3
"""CAPM — estimate a security's beta and alpha against the market.
OLS regression of excess asset return on excess market return (statsmodels),
with the full CAPM report: characteristic line, actual-vs-expected, risk
decomposition, Security Market Line, rolling beta and residual diagnostics.

Input (from capm-page.tsx):
    data              : list[dict]
    asset_col         : str    asset price or return column
    market_col        : str    market/benchmark price or return column
    is_returns        : bool   True = columns already hold period returns
    return_type       : "simple"|"log" (default simple)
    periods_per_year  : int    (default 252)
    rf_annual         : float  annual risk-free rate (default 0)
Output: { results: {...}, plot } (6-panel CAPM diagnostics).
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
        alpha_ci = model.conf_int()[0]; beta_ci = model.conf_int()[1]
        r2 = float(model.rsquared)

        mean_mkt_excess = float(np.mean(x))
        market_premium_ann = (1 + mean_mkt_excess) ** ppy - 1
        capm_expected = rf_p + beta * mean_mkt_excess          # per period (fair return)
        capm_expected_ann = (1 + capm_expected) ** ppy - 1
        # alpha at several horizons
        alpha_daily = alpha
        alpha_monthly = (1 + alpha) ** (ppy / 12.0) - 1
        alpha_ann = (1 + alpha) ** ppy - 1

        # risk decomposition
        total_var = float(np.var(ra, ddof=1))
        systematic = float(beta ** 2 * np.var(rm, ddof=1))
        idiosyncratic = max(total_var - systematic, 0.0)
        sys_pct = 100 * systematic / total_var if total_var > 0 else 0.0

        # required return breakdown
        required_return = {
            "risk_free_annual": _fin(rf_annual, 6),
            "market_premium_annual": _fin(market_premium_ann, 6),
            "beta": _fin(beta, 4),
            "required_return_annual": _fin(capm_expected_ann, 6),
        }

        # actual vs CAPM expected (per period, for plotting)
        capm_pred = rf_p + beta * x        # note: excludes alpha (pure CAPM fit line)
        fitted = np.asarray(model.fittedvalues, dtype=float)  # includes alpha
        resid = np.asarray(model.resid, dtype=float)

        # rolling beta
        rn = n
        window = int(min(max(20, rn // 4), rn))
        rolling = {"window": window, "points": []}
        if rn >= window + 5:
            step = max(1, (rn - window) // 150)
            Xr = sm.add_constant(x)
            for end in range(window, rn + 1, step):
                s = end - window
                try:
                    b, *_ = np.linalg.lstsq(Xr[s:end], y[s:end], rcond=None)
                    rolling["points"].append({"t": end - 1, "beta": _fin(b[1], 4), "alpha": _fin(b[0] * ppy, 6)})
                except Exception:
                    continue

        # residual diagnostics
        dw = float(durbin_watson(resid))
        jb_stat, jb_p, jb_skew, jb_kurt = jarque_bera(resid)
        try:
            bp_stat, bp_p, _, _ = het_breuschpagan(resid, X)
        except Exception:
            bp_stat, bp_p = np.nan, np.nan
        diagnostics = {
            "durbin_watson": _fin(dw, 4),
            "dw_note_en": "no autocorrelation" if 1.5 <= dw <= 2.5 else ("positive autocorrelation" if dw < 1.5 else "negative autocorrelation"),
            "dw_note_ko": "자기상관 없음" if 1.5 <= dw <= 2.5 else ("양의 자기상관" if dw < 1.5 else "음의 자기상관"),
            "jarque_bera": _fin(jb_stat, 4), "jb_p_value": _fin(jb_p, 6), "residuals_normal": bool(jb_p > 0.05),
            "skew": _fin(jb_skew, 4), "kurtosis": _fin(jb_kurt, 4),
            "breusch_pagan": _fin(bp_stat, 4), "bp_p_value": _fin(bp_p, 6),
            "homoskedastic": bool(np.isfinite(bp_p) and bp_p > 0.05),
        }

        # beta classification
        if beta < 0:
            beta_class = {"en": "Inverse (moves against the market)", "ko": "역방향 (시장과 반대)"}
        elif beta < 0.95:
            beta_class = {"en": "Defensive (less volatile than market)", "ko": "방어적 (시장보다 낮은 변동성)"}
        elif beta <= 1.05:
            beta_class = {"en": "Market-like (average volatility)", "ko": "시장 수준 (평균 변동성)"}
        else:
            beta_class = {"en": "Aggressive (more volatile than market)", "ko": "공격적 (시장보다 높은 변동성)"}

        # ---- 6-panel figure ----
        plot = None
        try:
            fig, axes = plt.subplots(2, 3, figsize=(16, 9.5), dpi=110)
            # (1) characteristic line
            ax = axes[0, 0]
            ax.scatter(x * 100, y * 100, s=13, alpha=0.5, color="#2563eb")
            xr = np.linspace(x.min(), x.max(), 50)
            ax.plot(xr * 100, (alpha + beta * xr) * 100, color="#dc2626", lw=2,
                    label=f"β={beta:.2f}, α={alpha*100:.3f}%")
            ax.axhline(0, color="#cbd5e1", lw=0.6); ax.axvline(0, color="#cbd5e1", lw=0.6)
            ax.set_xlabel("Market excess return (%)"); ax.set_ylabel("Asset excess return (%)")
            ax.set_title("1. Characteristic line"); ax.legend(fontsize=8, frameon=False)
            # (2) actual vs CAPM expected (time series)
            ax = axes[0, 1]
            tt = np.arange(n)
            ax.plot(tt, y * 100, color="#2563eb", lw=1.1, label="Actual excess")
            ax.plot(tt, capm_pred * 100, color="#dc2626", lw=1.1, alpha=0.8, label="CAPM expected")
            ax.axhline(0, color="#cbd5e1", lw=0.6)
            ax.set_xlabel("Period"); ax.set_ylabel("Excess return (%)")
            ax.set_title("2. Actual vs CAPM expected"); ax.legend(fontsize=8, frameon=False)
            # (3) risk decomposition pie
            ax = axes[0, 2]
            ax.pie([sys_pct, max(100 - sys_pct, 0)],
                   labels=[f"Systematic\n{sys_pct:.0f}%", f"Idiosyncratic\n{100-sys_pct:.0f}%"],
                   colors=["#2563eb", "#e5e7eb"], startangle=90, wedgeprops={"width": 0.42},
                   textprops={"fontsize": 9})
            ax.set_title(f"3. Risk decomposition (R²={r2:.2f})")
            # (4) rolling beta
            ax = axes[1, 0]
            if rolling["points"]:
                tx = [pt["t"] for pt in rolling["points"]]
                ax.plot(tx, [pt["beta"] for pt in rolling["points"]], color="#2563eb", lw=1.6)
                ax.axhline(beta, color="#dc2626", ls="--", lw=1, label=f"full-sample β={beta:.2f}")
                ax.axhline(1.0, color="#94a3b8", ls=":", lw=1, label="β=1")
                ax.set_xlabel("Period"); ax.set_ylabel("Rolling β")
                ax.set_title(f"4. Rolling beta (w={window})"); ax.legend(fontsize=7, frameon=False)
            else:
                ax.text(0.5, 0.5, "Not enough data\nfor rolling window", ha="center", va="center")
                ax.set_title("4. Rolling beta"); ax.axis("off")
            # (5) Security Market Line
            ax = axes[1, 1]
            bmax = max(1.4, beta * 1.3, 1.1)
            bx = np.linspace(0, bmax, 50)
            ax.plot(bx, (rf_annual + bx * market_premium_ann) * 100, color="#111827", lw=1.6, label="SML")
            ax.scatter([0], [rf_annual * 100], color="#16a34a", s=45, zorder=5, label="Risk-free")
            ax.scatter([1], [(rf_annual + market_premium_ann) * 100], color="#f59e0b", s=45, zorder=5, label="Market")
            actual_ann = (1 + float(np.mean(ra))) ** ppy - 1
            ax.scatter([beta], [actual_ann * 100], color="#dc2626", s=60, zorder=6, marker="D", label="Asset (actual)")
            ax.annotate("α>0" if actual_ann > capm_expected_ann else "α<0",
                        (beta, actual_ann * 100), textcoords="offset points", xytext=(6, 6), fontsize=8)
            ax.set_xlabel("Beta"); ax.set_ylabel("Expected return (%, ann.)")
            ax.set_title("5. Security Market Line"); ax.legend(fontsize=7, frameon=False)
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
            f"(fair) return for this beta is {capm_expected_ann*100:.2f}% annualised = risk-free "
            f"{rf_annual*100:.1f}% + β {beta:.2f} × market premium {market_premium_ann*100:.1f}%."
        )

        results = {
            "status": "ok", "n_obs": int(n), "asset_col": acol, "market_col": mcol,
            "periods_per_year": ppy, "rf_annual": _fin(rf_annual, 6),
            "return_type": rtype, "is_returns": is_returns,
            "beta": _fin(beta, 4), "beta_class": beta_class,
            "alpha_period": _fin(alpha, 6), "alpha_daily": _fin(alpha_daily, 6),
            "alpha_monthly": _fin(alpha_monthly, 6), "alpha_annual": _fin(alpha_ann, 6),
            "beta_se": _fin(beta_se, 4), "alpha_se": _fin(alpha_se, 6),
            "beta_t": _fin(beta_t, 3), "alpha_t": _fin(alpha_t, 3),
            "beta_p": _fin(beta_p, 5), "alpha_p": _fin(alpha_p, 5),
            "beta_ci_low": _fin(beta_ci[0], 4), "beta_ci_high": _fin(beta_ci[1], 4),
            "alpha_ci_low": _fin(alpha_ci[0], 6), "alpha_ci_high": _fin(alpha_ci[1], 6),
            "alpha_significant": bool(alpha_p < 0.05),
            "r_squared": _fin(r2, 4), "correlation": _fin(float(np.corrcoef(ra, rm)[0, 1]), 4),
            "f_stat": _fin(model.fvalue, 4), "f_p_value": _fin(model.f_pvalue, 6),
            "resid_std": _fin(float(np.std(resid, ddof=1)), 6), "df_resid": int(model.df_resid),
            "capm_expected_annual": _fin(capm_expected_ann, 6),
            "market_premium_annual": _fin(market_premium_ann, 6),
            "asset_return_annual": _fin((1 + float(np.mean(ra))) ** ppy - 1, 6),
            "required_return": required_return,
            "systematic_pct": _fin(sys_pct, 2), "idiosyncratic_pct": _fin(100 - sys_pct, 2),
            "total_variance": _fin(total_var, 8), "systematic_variance": _fin(systematic, 8),
            "idiosyncratic_variance": _fin(idiosyncratic, 8),
            "rolling": rolling, "diagnostics": diagnostics,
            "interpretation": interpretation,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
