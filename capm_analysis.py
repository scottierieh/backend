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
Output: { results: {...}, plot } (characteristic line scatter + SML).
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
        r2 = float(model.rsquared)

        mean_mkt_excess = float(np.mean(x))
        capm_expected = rf_p + beta * mean_mkt_excess          # per period
        capm_expected_ann = (1 + capm_expected) ** ppy - 1
        alpha_ann = (1 + alpha) ** ppy - 1
        total_var = float(np.var(ra, ddof=1))
        systematic = float(beta ** 2 * np.var(rm, ddof=1))
        idiosyncratic = max(total_var - systematic, 0.0)
        sys_pct = 100 * systematic / total_var if total_var > 0 else 0.0

        # plot: characteristic line
        plot = None
        try:
            fig, ax = plt.subplots(figsize=(7.2, 5.4), dpi=120)
            ax.scatter(x * 100, y * 100, s=14, alpha=0.5, color="#2563eb")
            xr = np.linspace(x.min(), x.max(), 50)
            ax.plot(xr * 100, (alpha + beta * xr) * 100, color="#dc2626", lw=2,
                    label=f"β = {beta:.2f}, α = {alpha*100:.3f}%")
            ax.axhline(0, color="#cbd5e1", lw=0.6); ax.axvline(0, color="#cbd5e1", lw=0.6)
            ax.set_xlabel("Market excess return (%)"); ax.set_ylabel("Asset excess return (%)")
            ax.set_title("Characteristic line (CAPM regression)")
            ax.legend(fontsize=9, frameon=False); fig.tight_layout()
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
            f"(fair) return for this beta is {capm_expected_ann*100:.2f}% annualised."
        )

        results = {
            "status": "ok", "n_obs": int(n), "asset_col": acol, "market_col": mcol,
            "periods_per_year": ppy, "rf_annual": _fin(rf_annual, 6),
            "beta": _fin(beta, 4), "alpha_period": _fin(alpha, 6), "alpha_annual": _fin(alpha_ann, 6),
            "beta_se": _fin(beta_se, 4), "alpha_se": _fin(alpha_se, 6),
            "beta_t": _fin(beta_t, 3), "alpha_t": _fin(alpha_t, 3),
            "beta_p": _fin(beta_p, 5), "alpha_p": _fin(alpha_p, 5),
            "r_squared": _fin(r2, 4), "correlation": _fin(float(np.corrcoef(ra, rm)[0, 1]), 4),
            "capm_expected_annual": _fin(capm_expected_ann, 6),
            "systematic_pct": _fin(sys_pct, 2), "idiosyncratic_pct": _fin(100 - sys_pct, 2),
            "interpretation": interpretation,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
