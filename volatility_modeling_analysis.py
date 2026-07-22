#!/usr/bin/env python3
"""Volatility Modeling — GARCH-family conditional volatility. arch.

Fits a GARCH(1,1) or EGARCH(1,1) model to a return series, extracting the
conditional (time-varying) volatility, persistence, long-run volatility, and a
forward volatility forecast.

Input (from volatility-modeling-page.tsx):
    data             : list[dict]
    asset_col        : str    return or price column
    is_returns       : bool
    return_type      : "simple"|"log"
    model            : "GARCH" | "EGARCH"
    dist             : "normal" | "t"
    periods_per_year : int    (default 252) to annualise volatility
    horizon          : int    forecast horizon (default 10)
Output: { results: {...}, plot } (returns + conditional vol + forecast).
"""
import sys, json, io, base64
import numpy as np
import pandas as pd
from arch import arch_model

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
        is_returns = bool(p.get("is_returns", False))
        rtype = (p.get("return_type") or "simple").lower()
        model_type = (p.get("model") or "GARCH").upper()
        dist = (p.get("dist") or "normal").lower()
        ppy = int(p.get("periods_per_year") or 252)
        horizon = int(p.get("horizon") or 10)
        if not asset_col or asset_col not in df.columns:
            raise ValueError("Select the return column.")
        horizon = max(1, min(horizon, 60))

        s = pd.to_numeric(df[asset_col], errors="coerce")
        if is_returns:
            ret = s.dropna()
        elif rtype == "log":
            if (s <= 0).any():
                raise ValueError("Log returns require positive prices.")
            ret = np.log(s / s.shift(1)).dropna()
        else:
            ret = (s / s.shift(1) - 1.0).dropna()
        ret = ret.reset_index(drop=True)
        if len(ret) < 50:
            raise ValueError("Need at least 50 return observations to fit a GARCH model.")

        # arch works best on percentage returns
        scale = 100.0
        y = ret.values * scale

        vol_model = "EGARCH" if model_type == "EGARCH" else "GARCH"
        am = arch_model(y, mean="Constant", vol=vol_model, p=1, q=1,
                        dist=("t" if dist == "t" else "normal"))
        res = am.fit(disp="off")

        params = {k: _fin(float(v), 6) for k, v in res.params.items()}
        cond_vol = np.asarray(res.conditional_volatility) / scale     # back to return units
        pvalues = {k: _fin(float(v), 6) for k, v in res.pvalues.items()}

        # persistence & long-run vol (GARCH only has clean closed form)
        alpha = float(res.params.get("alpha[1]", 0.0))
        beta = float(res.params.get("beta[1]", 0.0))
        omega = float(res.params.get("omega", 0.0))
        persistence = alpha + beta
        if vol_model == "GARCH" and persistence < 1 and persistence > 0:
            lr_var = omega / (1 - persistence) / (scale ** 2)
            lr_vol = float(np.sqrt(lr_var))
        else:
            lr_vol = float(np.std(ret, ddof=1))
        lr_vol_annual = lr_vol * np.sqrt(ppy)
        current_vol = float(cond_vol[-1])
        current_vol_annual = current_vol * np.sqrt(ppy)

        # forecast
        fc = res.forecast(horizon=horizon, reindex=False)
        fvar = np.asarray(fc.variance.values[-1]) / (scale ** 2)
        fvol = np.sqrt(fvar)
        forecast = [{"step": i + 1, "volatility": _fin(float(fvol[i]), 6),
                     "volatility_annual": _fin(float(fvol[i] * np.sqrt(ppy)), 5)} for i in range(len(fvol))]

        # plot: returns with conditional vol bands + forecast
        plot = None
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), dpi=115, sharex=False,
                                           gridspec_kw={"height_ratios": [1.1, 1]})
            xr = np.arange(len(ret))
            ax1.plot(xr, ret.values * 100, color="#94a3b8", lw=0.6, label="Returns")
            ax1.plot(xr, cond_vol * 100 * 2, color="#dc2626", lw=1, label="+2σ (cond.)")
            ax1.plot(xr, -cond_vol * 100 * 2, color="#dc2626", lw=1)
            ax1.set_ylabel("Return / band (%)"); ax1.legend(fontsize=8, frameon=False)
            ax1.set_title(f"{vol_model}(1,1) — returns and ±2 conditional-sigma band")
            # conditional vol + forecast
            ax2.plot(xr, cond_vol * 100, color="#2563eb", lw=1.2, label="Conditional volatility")
            fx = np.arange(len(ret), len(ret) + horizon)
            ax2.plot(fx, fvol * 100, "o-", color="#16a34a", lw=1.5, ms=3, label="Forecast")
            ax2.axhline(lr_vol * 100, color="#f59e0b", ls="--", lw=1, label="Long-run vol")
            ax2.set_ylabel("Volatility (%, per period)"); ax2.set_xlabel("Period")
            ax2.legend(fontsize=8, frameon=False); ax2.set_title("Conditional volatility & forecast")
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        persist_txt = ("very persistent — shocks to volatility decay slowly, so calm and turbulent periods cluster"
                       if persistence > 0.95 else
                       "moderately persistent — volatility clusters but reverts to its long-run level at a steady pace"
                       if persistence > 0.8 else
                       "low persistence — volatility reverts quickly and clusters weakly")
        interpretation = (
            f"The {vol_model}(1,1) model captures volatility clustering: the current conditional volatility is "
            f"{current_vol_annual:.1%} annualised, versus a long-run level of {lr_vol_annual:.1%}. The persistence "
            f"(alpha + beta) is {persistence:.3f}, which is {persist_txt}. "
            + (f"Because the current volatility is {'above' if current_vol > lr_vol else 'below'} the long-run level, "
               f"the forecast {'decays down toward' if current_vol > lr_vol else 'drifts up toward'} it over the next "
               f"{horizon} periods. " if vol_model == 'GARCH' else "")
            + "Conditional volatility models are the standard tool for risk that changes over time."
        )

        results = {
            "status": "ok", "asset": asset_col, "model": vol_model, "dist": dist, "n_obs": int(len(ret)),
            "periods_per_year": ppy, "horizon": horizon,
            "params": params, "pvalues": pvalues,
            "persistence": _fin(persistence, 5), "alpha": _fin(alpha, 5), "beta": _fin(beta, 5), "omega": _fin(omega, 8),
            "long_run_vol": _fin(lr_vol, 6), "long_run_vol_annual": _fin(lr_vol_annual, 5),
            "current_vol": _fin(current_vol, 6), "current_vol_annual": _fin(current_vol_annual, 5),
            "log_likelihood": _fin(float(res.loglikelihood), 3), "aic": _fin(float(res.aic), 3), "bic": _fin(float(res.bic), 3),
            "forecast": forecast,
            "interpretation": interpretation,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
