#!/usr/bin/env python3
"""Volatility Modeling — GARCH-family conditional volatility. arch.

Fits a GARCH(1,1) or EGARCH(1,1) model to a return series, extracting the
conditional (time-varying) volatility, persistence, long-run volatility, and a
forward volatility forecast. Also builds a full step-6 report with 9 additive
sections (ARCH-LM pre-test, model comparison, conditional volatility, volatility
persistence, asymmetric volatility, multi-horizon forecast, model diagnostics,
distribution comparison) — this page models ESTIMATED / conditional volatility,
distinct from the Volatility Analysis page's raw / rolling REALISED volatility.

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
from statsmodels.stats.diagnostic import het_arch, acorr_ljungbox

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#2563eb"
GREEN = "#16a34a"
RED = "#dc2626"
AMBER = "#d97706"
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
        arch_dist = "t" if dist == "t" else "normal"

        # ─────────────────────────── ① / primary fit (selected by user, kept for backward compat) ───────────────────────────
        vol_model = "EGARCH" if model_type == "EGARCH" else "GARCH"
        am = arch_model(y, mean="Constant", vol=vol_model, p=1, q=1, dist=arch_dist)
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

        # forecast (multi-horizon: reuse the max requested horizon, report a few steps)
        fc = res.forecast(horizon=horizon, reindex=False)
        fvar = np.asarray(fc.variance.values[-1]) / (scale ** 2)
        fvol = np.sqrt(fvar)
        forecast = [{"step": i + 1, "volatility": _fin(float(fvol[i]), 6),
                     "volatility_annual": _fin(float(fvol[i] * np.sqrt(ppy)), 5)} for i in range(len(fvol))]
        horizon_steps = sorted(set([h for h in (1, 5, 10, 20) if h <= len(fvol)] + [len(fvol)]))
        forecast_horizons = [{"horizon": h, "volatility": _fin(float(fvol[h - 1]), 6),
                               "volatility_annual": _fin(float(fvol[h - 1] * np.sqrt(ppy)), 5)} for h in horizon_steps]

        # ─────────────────────────── ② ARCH-LM pre-test (justifies fitting a GARCH-family model at all) ───────────────────────────
        arch_lm_table = []
        for lag in (5, 10):
            try:
                lm_stat, lm_p, f_stat, f_p = het_arch(ret.values, nlags=lag)
                arch_lm_table.append({
                    "test": f"ARCH-LM ({lag})", "statistic": _fin(lm_stat, 4), "pvalue": _fin(lm_p, 6),
                    "result": "Significant" if lm_p is not None and lm_p < 0.05 else "Not significant",
                })
            except Exception:
                continue
        arch_lm_significant = any(r["result"] == "Significant" for r in arch_lm_table)

        # ─────────────────────────── ③ Volatility Model Comparison ───────────────────────────
        model_specs = [
            ("ARCH(1)", dict(vol="ARCH", p=1)),
            ("GARCH(1,1)", dict(vol="GARCH", p=1, q=1)),
            ("EGARCH(1,1)", dict(vol="EGARCH", p=1, q=1)),
            ("GJR-GARCH(1,1)", dict(vol="GARCH", p=1, o=1, q=1)),
        ]
        model_comparison = []
        fitted_models = {}
        for name, spec in model_specs:
            try:
                m = arch_model(y, mean="Constant", dist=arch_dist, **spec)
                r = m.fit(disp="off")
                model_comparison.append({
                    "model": name, "aic": _fin(float(r.aic), 3), "bic": _fin(float(r.bic), 3),
                    "log_likelihood": _fin(float(r.loglikelihood), 3), "selected": False,
                })
                fitted_models[name] = r
            except Exception:
                continue
        if model_comparison:
            best_idx = int(np.argmin([row["aic"] if row["aic"] is not None else np.inf for row in model_comparison]))
            model_comparison[best_idx]["selected"] = True
            best_model_name = model_comparison[best_idx]["model"]
        else:
            best_model_name = None

        # ─────────────────────────── ④ Conditional Volatility summary ───────────────────────────
        conditional_vol_summary = {
            "average": _fin(float(np.mean(cond_vol)), 6),
            "current": _fin(current_vol, 6),
            "maximum": _fin(float(np.max(cond_vol)), 6),
            "minimum": _fin(float(np.min(cond_vol)), 6),
        }

        # ─────────────────────────── ⑤ Volatility Persistence + shock decay ───────────────────────────
        half_life = None
        if 0 < persistence < 1:
            half_life = float(np.log(0.5) / np.log(persistence))
        persistence_table = {
            "arch_coef": _fin(alpha, 5), "garch_coef": _fin(beta, 5),
            "persistence": _fin(persistence, 5), "half_life": _fin(half_life, 2),
        }
        decay_horizon = 60
        t_arr = np.arange(0, decay_horizon + 1)
        if 0 < persistence < 1:
            decay_curve = (persistence ** t_arr)
        else:
            decay_curve = np.zeros_like(t_arr, dtype=float)

        # ─────────────────────────── ⑥ Asymmetric Volatility (uses additionally-fit GJR-GARCH) ───────────────────────────
        asymmetric = None
        gjr = fitted_models.get("GJR-GARCH(1,1)")
        if gjr is not None and "gamma[1]" in gjr.params.index:
            a_g = float(gjr.params.get("alpha[1]", 0.0))
            g_g = float(gjr.params.get("gamma[1]", 0.0))
            b_g = float(gjr.params.get("beta[1]", 0.0))
            o_g = float(gjr.params.get("omega", 0.0))
            denom = 1 - a_g - g_g / 2.0 - b_g
            sigma2_lr = (o_g / denom) if denom > 0 else float(np.var(y))
            shock = 1.0  # ~1% shock in scaled (percentage-return) units
            sigma2_pos = o_g + a_g * shock ** 2 + b_g * sigma2_lr               # positive shock: indicator = 0
            sigma2_neg = o_g + a_g * shock ** 2 + g_g * shock ** 2 + b_g * sigma2_lr  # negative shock: indicator = 1
            vol_pos = float(np.sqrt(max(sigma2_pos, 0))) / scale
            vol_neg = float(np.sqrt(max(sigma2_neg, 0))) / scale
            asymmetric = {
                "asymmetry_param": _fin(g_g, 6),
                "rows": [
                    {"shock": "+1%", "volatility_impact": _fin(vol_pos, 6)},
                    {"shock": "-1%", "volatility_impact": _fin(vol_neg, 6)},
                ],
                "note": None,
            }
        else:
            asymmetric = {"asymmetry_param": None, "rows": [], "note": "GJR-GARCH could not be fit to this series, so asymmetric volatility (leverage effect) is not reported."}

        # ─────────────────────────── ⑧ Model Diagnostics (on standardised residuals of the PRIMARY fit) ───────────────────────────
        std_resid = np.asarray(res.resid) / np.asarray(res.conditional_volatility)
        std_resid = std_resid[np.isfinite(std_resid)]
        diagnostics = []
        try:
            lb = acorr_ljungbox(std_resid, lags=[10], return_df=True)
            stat_v = float(lb["lb_stat"].iloc[0]); p_v = float(lb["lb_pvalue"].iloc[0])
            diagnostics.append({"diagnostic": "Ljung-Box on standardized residuals (10)", "statistic": _fin(stat_v, 4),
                                 "pvalue": _fin(p_v, 6), "result": "Pass" if p_v > 0.05 else "Fail"})
        except Exception:
            pass
        try:
            lb2 = acorr_ljungbox(std_resid ** 2, lags=[10], return_df=True)
            stat_v = float(lb2["lb_stat"].iloc[0]); p_v = float(lb2["lb_pvalue"].iloc[0])
            diagnostics.append({"diagnostic": "Ljung-Box on squared standardized residuals (10)", "statistic": _fin(stat_v, 4),
                                 "pvalue": _fin(p_v, 6), "result": "Pass" if p_v > 0.05 else "Fail"})
        except Exception:
            pass
        try:
            lm_stat, lm_p, _, _ = het_arch(std_resid, nlags=5)
            diagnostics.append({"diagnostic": "ARCH-LM on standardized residuals (5)", "statistic": _fin(lm_stat, 4),
                                 "pvalue": _fin(lm_p, 6), "result": "Pass" if lm_p > 0.05 else "Fail"})
        except Exception:
            pass

        # ─────────────────────────── ⑨ Distribution Comparison (refit selected/best model with different dist) ───────────────────────────
        best_spec = dict(model_specs).get(best_model_name) if best_model_name else dict(vol=vol_model, p=1, q=1)
        distribution_comparison = []
        for dname, dkey in (("Normal", "normal"), ("Student-t", "t"), ("Skewed Student-t", "skewt"), ("GED", "ged")):
            try:
                m = arch_model(y, mean="Constant", dist=dkey, **best_spec)
                r = m.fit(disp="off")
                distribution_comparison.append({
                    "distribution": dname, "aic": _fin(float(r.aic), 3), "bic": _fin(float(r.bic), 3),
                })
            except Exception:
                continue

        # ─────────────────────────── plots ───────────────────────────
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
            ax2.plot(xr, cond_vol * 100, color=BLUE, lw=1.2, label="Conditional volatility")
            fx = np.arange(len(ret), len(ret) + horizon)
            ax2.plot(fx, fvol * 100, "o-", color=GREEN, lw=1.5, ms=3, label="Forecast")
            ax2.axhline(lr_vol * 100, color=AMBER, ls="--", lw=1, label="Long-run vol")
            ax2.set_ylabel("Volatility (%, per period)"); ax2.set_xlabel("Period")
            ax2.legend(fontsize=8, frameon=False); ax2.set_title("Conditional volatility & forecast")
            fig.tight_layout()
            plot = _png(fig)
        except Exception:
            plt.close("all"); plot = None

        chart_conditional_vol = None
        try:
            fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=115)
            xr = np.arange(len(ret))
            ax.plot(xr, cond_vol * 100, color=BLUE, lw=1.3, label="Conditional volatility")
            ax.axhline(lr_vol * 100, color=AMBER, ls="--", lw=1, label="Long-run vol")
            ax.set_title(f"Conditional Volatility — {vol_model}(1,1)")
            ax.set_xlabel("Period"); ax.set_ylabel("Volatility (%, per period)")
            ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.2)
            fig.tight_layout()
            chart_conditional_vol = _png(fig)
        except Exception:
            plt.close("all"); chart_conditional_vol = None

        chart_shock_decay = None
        try:
            fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=115)
            ax.plot(t_arr, decay_curve, color=PURPLE, lw=1.6)
            ax.axhline(0.5, color="#111827", lw=0.7, ls="--", label="Half impact")
            if half_life is not None:
                ax.axvline(half_life, color=RED, lw=0.9, ls="--", label=f"Half-life ≈ {half_life:.1f}")
            ax.set_title("Volatility Shock Decay")
            ax.set_xlabel("Periods ahead"); ax.set_ylabel("Fraction of initial shock remaining")
            ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.2)
            fig.tight_layout()
            chart_shock_decay = _png(fig)
        except Exception:
            plt.close("all"); chart_shock_decay = None

        chart_asymmetric = None
        if asymmetric["rows"]:
            try:
                fig, ax = plt.subplots(figsize=(6, 4.6), dpi=115)
                vals = [row["volatility_impact"] * 100 for row in asymmetric["rows"]]
                colors = [GREEN, RED]
                ax.bar(["+1% shock", "-1% shock"], vals, color=colors, width=0.55)
                ax.set_title("Asymmetric Volatility Response (GJR-GARCH)")
                ax.set_ylabel("Volatility impact (%, per period)")
                ax.grid(alpha=0.2, axis="y")
                fig.tight_layout()
                chart_asymmetric = _png(fig)
            except Exception:
                plt.close("all"); chart_asymmetric = None

        chart_forecast = None
        try:
            fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=115)
            xr = np.arange(len(ret))
            ax.plot(xr, cond_vol * 100, color=BLUE, lw=1.2, label="Historical conditional volatility")
            fx = np.arange(len(ret), len(ret) + horizon)
            ax.plot(fx, fvol * 100, "--", color=GREEN, lw=1.6, label="Forecast")
            ax.scatter(fx, fvol * 100, color=GREEN, s=14, zorder=3)
            ax.axhline(lr_vol * 100, color=AMBER, ls=":", lw=1, label="Long-run vol")
            ax.axvline(len(ret) - 0.5, color="#94a3b8", lw=0.8, ls="-")
            ax.set_title("Volatility Forecast")
            ax.set_xlabel("Period"); ax.set_ylabel("Volatility (%, per period)")
            ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.2)
            fig.tight_layout()
            chart_forecast = _png(fig)
        except Exception:
            plt.close("all"); chart_forecast = None

        charts = {
            "returns_and_vol": plot,
            "conditional_vol": chart_conditional_vol,
            "shock_decay": chart_shock_decay,
            "asymmetric": chart_asymmetric,
            "forecast": chart_forecast,
        }

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
            + ("The ARCH-LM pre-test found significant residual heteroskedasticity in the raw returns, which "
               "supports fitting a GARCH-family model in the first place. " if arch_lm_significant else
               "The ARCH-LM pre-test did NOT find significant heteroskedasticity in the raw returns at the tested "
               "lags — a GARCH-family model may not be well-motivated here, though the fit below is still reported. ")
            + "This model estimates conditional (forward-looking) volatility, which differs from the raw or "
              "rolling realised volatility shown on the Volatility Analysis page. "
              "Conditional volatility models are the standard tool for risk that changes over time."
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

            # ① Model Summary
            "model_summary": {
                "selected_model": f"{vol_model}(1,1)", "mean_model": "Constant",
                "distribution": "Student-t" if dist == "t" else "Normal",
                "aic": _fin(float(res.aic), 3), "bic": _fin(float(res.bic), 3),
                "persistence": _fin(persistence, 5),
                "long_run_volatility": _fin(lr_vol_annual, 5),
                "current_volatility": _fin(current_vol_annual, 5),
            },
            # ② ARCH-LM pre-test
            "arch_lm_table": arch_lm_table,
            "arch_lm_significant": arch_lm_significant,
            # ③ Model comparison
            "model_comparison": model_comparison,
            "best_model": best_model_name,
            # ④ Conditional volatility summary
            "conditional_vol_summary": conditional_vol_summary,
            # ⑤ Persistence + shock decay
            "persistence_table": persistence_table,
            "shock_decay": [{"t": int(t_arr[i]), "value": _fin(float(decay_curve[i]), 6)} for i in range(len(t_arr))],
            # ⑥ Asymmetric volatility
            "asymmetric": asymmetric,
            # ⑦ Multi-horizon forecast
            "forecast_horizons": forecast_horizons,
            # ⑧ Diagnostics
            "diagnostics": diagnostics,
            # ⑨ Distribution comparison
            "distribution_comparison": distribution_comparison,

            "charts": charts,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
