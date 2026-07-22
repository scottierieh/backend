#!/usr/bin/env python3
"""Financial Forecasting — ARIMA forecast of a financial series. statsmodels.

Selects an ARIMA(p,d,q) order by AIC over a small grid, fits it, and forecasts
forward with confidence intervals.

Input (from financial-forecasting-page.tsx):
    data        : list[dict]
    asset_col   : str
    series_type : "level"|"returns"|"log_returns"   (default level)
    horizon     : int   (default 10)
    max_order   : int   (default 2) max p and q searched
Output: { results: {order, forecast[], metrics}, plot }
"""
import sys, json, io, base64
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


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
        col = p.get("asset_col")
        stype = (p.get("series_type") or "level").lower()
        horizon = int(p.get("horizon") or 10)
        max_order = int(p.get("max_order") or 2)
        if not col or col not in df.columns:
            raise ValueError("Select a column.")
        horizon = max(1, min(horizon, 60))
        max_order = max(1, min(max_order, 4))

        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if stype == "returns":
            x = (s / s.shift(1) - 1.0).dropna(); label = "simple returns"
        elif stype == "log_returns":
            if (s <= 0).any():
                raise ValueError("Log returns require positive values.")
            x = np.log(s / s.shift(1)).dropna(); label = "log returns"
        else:
            x = s; label = "level"
        x = pd.Series(np.asarray(x, float)).reset_index(drop=True)
        n = len(x)
        if n < 30:
            raise ValueError("Need at least 30 observations to forecast.")

        # d: difference the level once by default (returns already stationary -> d=0)
        d_grid = [1, 0] if stype == "level" else [0, 1]

        best = None
        for d in d_grid:
            for pp in range(0, max_order + 1):
                for q in range(0, max_order + 1):
                    if pp == 0 and q == 0:
                        continue
                    try:
                        m = ARIMA(x.values, order=(pp, d, q)).fit()
                        if best is None or m.aic < best[1]:
                            best = ((pp, d, q), m.aic, m)
                    except Exception:
                        continue
        if best is None:
            raise ValueError("Could not fit any ARIMA model to this series.")
        order, aic, model = best

        fc = model.get_forecast(steps=horizon)
        mean_fc = np.asarray(fc.predicted_mean)
        ci = np.asarray(fc.conf_int(alpha=0.05))
        forecast = [{"step": i + 1, "forecast": _fin(float(mean_fc[i]), 6),
                     "lower": _fin(float(ci[i, 0]), 6), "upper": _fin(float(ci[i, 1]), 6)} for i in range(horizon)]

        resid = np.asarray(model.resid)
        rmse = float(np.sqrt(np.mean(resid[1:] ** 2)))
        mae = float(np.mean(np.abs(resid[1:])))

        # Ljung-Box on residuals (should be white noise if model captured structure)
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb = acorr_ljungbox(resid, lags=[min(10, n // 3)], return_df=True)
        lb_p = float(lb["lb_pvalue"].iloc[0])
        resid_white = bool(lb_p >= 0.05)

        plot = None
        try:
            fig, ax = plt.subplots(figsize=(11.5, 5.2), dpi=118)
            hist_x = np.arange(n)
            ax.plot(hist_x, x.values, color="#94a3b8", lw=0.9, label=f"Observed ({label})")
            fitted = np.asarray(model.fittedvalues)
            ax.plot(hist_x, fitted, color="#2563eb", lw=1, alpha=0.8, label="In-sample fit")
            fx = np.arange(n, n + horizon)
            ax.plot(fx, mean_fc, color="#16a34a", lw=2, label="Forecast")
            ax.fill_between(fx, ci[:, 0], ci[:, 1], color="#16a34a", alpha=0.18, label="95% interval")
            ax.axvline(n - 0.5, color="#111827", ls=":", lw=1)
            ax.set_xlabel("Period"); ax.set_ylabel(label)
            ax.set_title(f"ARIMA{order} forecast of {col}")
            ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.2)
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        next_val = mean_fc[0]
        interpretation = (
            f"An ARIMA{order} model, selected by AIC, best fit the {label} of {col}. It forecasts a next-period value "
            f"of {next_val:.4g}, with the 95% interval widening from [{ci[0,0]:.4g}, {ci[0,1]:.4g}] at step 1 to "
            f"[{ci[-1,0]:.4g}, {ci[-1,1]:.4g}] at step {horizon} as uncertainty compounds. "
            + ("The residuals pass the Ljung-Box test, so the model has captured the series' linear structure and the "
               "leftovers look like white noise. " if resid_white else
               "The residuals still show autocorrelation (Ljung-Box p < 0.05), so the model has not fully captured the "
               "structure — a different order or a nonlinear model may fit better. ")
            + ("Note that financial returns are usually close to unpredictable, so a returns forecast will typically "
               "revert quickly to the mean with wide intervals — the honest message is often how uncertain the future "
               "is, not a confident point prediction." if stype != "level" else
               "For a price level, the forecast mostly extrapolates the recent trend; the widening interval is the key "
               "output, showing how quickly confidence decays.")
        )

        results = {
            "status": "ok", "asset": col, "series_type": stype, "series_label": label, "n_obs": n,
            "order": {"p": order[0], "d": order[1], "q": order[2]}, "order_str": f"ARIMA{order}",
            "aic": _fin(aic, 3), "bic": _fin(float(model.bic), 3),
            "rmse": _fin(rmse, 6), "mae": _fin(mae, 6),
            "ljung_box_p": _fin(lb_p, 6), "residuals_white_noise": resid_white,
            "horizon": horizon, "forecast": forecast, "next_value": _fin(float(next_val), 6),
            "interpretation": interpretation,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
