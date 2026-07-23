#!/usr/bin/env python3
"""Financial Forecasting — ARIMA forecast of a financial series, with baseline
model comparison, rolling-origin backtesting and residual diagnostics.
statsmodels / numpy / pandas.

Selects an ARIMA(p,d,q) order by AIC over a small grid, fits it, and forecasts
forward with confidence intervals. Also fits Naive and Drift baselines, backtests
the primary model with a rolling-origin walk-forward scheme, and sweeps the
forecast horizon to show how uncertainty compounds.

Input (from financial-forecasting-page.tsx):
    data        : list[dict]
    asset_col   : str
    series_type : "level"|"returns"|"log_returns"   (default level)
    horizon     : int   (default 10)
    max_order   : int   (default 2) max p and q searched
Output: { results: {..., charts: {actual_vs_forecast, model_comparison, backtest,
          residual_ts, residual_dist, residual_acf, horizon_comparison}}, plot }
"""
import sys, json, io, base64
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

BLUE = "#2563eb"
GREEN = "#16a34a"
RED = "#dc2626"
AMBER = "#d97706"
GREY = "#94a3b8"
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


def _fit_arima_by_aic(x, d_grid, max_order):
    """Grid-search ARIMA order by AIC. Returns (order_tuple, aic, fitted_model) or None."""
    best = None
    for d in d_grid:
        for pp in range(0, max_order + 1):
            for q in range(0, max_order + 1):
                if pp == 0 and q == 0:
                    continue
                try:
                    m = ARIMA(x, order=(pp, d, q)).fit()
                    if best is None or m.aic < best[1]:
                        best = ((pp, d, q), m.aic, m)
                except Exception:
                    continue
    return best


def _dir_acc(actual_change, pred_change):
    """% of periods where sign(actual_change) == sign(predicted_change)."""
    a = np.asarray(actual_change, float)
    p = np.asarray(pred_change, float)
    mask = np.isfinite(a) & np.isfinite(p) & (a != 0)
    if mask.sum() == 0:
        return None
    return float(np.mean(np.sign(a[mask]) == np.sign(p[mask])) * 100.0)


def _rmse_mae_mape(actual, pred):
    a = np.asarray(actual, float); p = np.asarray(pred, float)
    err = a - p
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    nz = a != 0
    mape = float(np.mean(np.abs(err[nz] / a[nz])) * 100.0) if nz.sum() else None
    return rmse, mae, mape


def _naive_forecast(train, h):
    return np.full(h, train[-1], dtype=float)


def _drift_forecast(train, h):
    n = len(train)
    if n < 2:
        return np.full(h, train[-1], dtype=float)
    drift = (train[-1] - train[0]) / (n - 1)
    return train[-1] + drift * np.arange(1, h + 1)


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
        xv = x.values

        # ─────────────────────────── Primary model: ARIMA by AIC ───────────────────────────
        d_grid = [1, 0] if stype == "level" else [0, 1]
        best = _fit_arima_by_aic(xv, d_grid, max_order)
        if best is None:
            raise ValueError("Could not fit any ARIMA model to this series.")
        order, aic, model = best

        fc = model.get_forecast(steps=horizon)
        mean_fc = np.asarray(fc.predicted_mean)
        ci95 = np.asarray(fc.conf_int(alpha=0.05))
        ci80 = np.asarray(fc.conf_int(alpha=0.20))
        forecast = [{"step": i + 1, "forecast": _fin(float(mean_fc[i]), 6),
                     "lower": _fin(float(ci95[i, 0]), 6), "upper": _fin(float(ci95[i, 1]), 6),
                     "lower80": _fin(float(ci80[i, 0]), 6), "upper80": _fin(float(ci80[i, 1]), 6)} for i in range(horizon)]

        resid = np.asarray(model.resid)
        rmse = float(np.sqrt(np.mean(resid[1:] ** 2)))
        mae = float(np.mean(np.abs(resid[1:])))

        # Ljung-Box on residuals (should be white noise if model captured structure)
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb = acorr_ljungbox(resid, lags=[min(10, n // 3)], return_df=True)
        lb_p = float(lb["lb_pvalue"].iloc[0])
        resid_white = bool(lb_p >= 0.05)

        # Jarque-Bera normality test on residuals
        from scipy import stats as sstats
        jb_stat, jb_p = sstats.jarque_bera(resid)
        jb_p = float(jb_p)
        resid_normal = bool(jb_p >= 0.05)

        # In-sample directional accuracy: sign(actual change) vs sign(fitted change)
        fitted = np.asarray(model.fittedvalues)
        m_len = min(len(fitted), n)
        actual_change = np.diff(xv[:m_len])
        fitted_change = np.diff(fitted[:m_len])
        dir_acc_insample = _dir_acc(actual_change, fitted_change)

        # ═══════════════════════════ ② Forecast Summary ═══════════════════════════
        forecast_summary = {
            "model": f"ARIMA{order}", "horizon": horizon,
            "rmse": _fin(rmse, 6), "mae": _fin(mae, 6),
            "directional_accuracy": _fin(dir_acc_insample, 2),
        }

        # ═══════════════════════════ ⑤ Model Comparison (Naive / Drift / ARIMA) ═══════════════════════════
        # Train/test split for out-of-sample comparison: last `test_len` points held out.
        test_len = max(5, min(horizon, n // 5))
        test_len = min(test_len, n - 20)  # keep at least 20 train points
        test_len = max(test_len, 3)
        train_cmp = xv[:-test_len]
        test_cmp = xv[-test_len:]

        model_comparison = []

        # Naive
        naive_fc = _naive_forecast(train_cmp, test_len)
        r_, m_, mp_ = _rmse_mae_mape(test_cmp, naive_fc)
        da_ = _dir_acc(np.diff(np.concatenate([[train_cmp[-1]], test_cmp])), np.diff(np.concatenate([[train_cmp[-1]], naive_fc])))
        model_comparison.append({"model": "Naive", "mae": _fin(m_, 6), "rmse": _fin(r_, 6), "mape": _fin(mp_, 3), "directional_accuracy": _fin(da_, 2)})

        # Drift
        drift_fc = _drift_forecast(train_cmp, test_len)
        r_, m_, mp_ = _rmse_mae_mape(test_cmp, drift_fc)
        da_ = _dir_acc(np.diff(np.concatenate([[train_cmp[-1]], test_cmp])), np.diff(np.concatenate([[train_cmp[-1]], drift_fc])))
        model_comparison.append({"model": "Drift", "mae": _fin(m_, 6), "rmse": _fin(r_, 6), "mape": _fin(mp_, 3), "directional_accuracy": _fin(da_, 2)})

        # Primary ARIMA, refit on train_cmp only, forecast test_len steps
        arima_cmp_fc = None
        try:
            m_train = ARIMA(train_cmp, order=order).fit()
            arima_cmp_fc = np.asarray(m_train.get_forecast(steps=test_len).predicted_mean)
        except Exception:
            arima_cmp_fc = None
        if arima_cmp_fc is not None:
            r_, m_, mp_ = _rmse_mae_mape(test_cmp, arima_cmp_fc)
            da_ = _dir_acc(np.diff(np.concatenate([[train_cmp[-1]], test_cmp])), np.diff(np.concatenate([[train_cmp[-1]], arima_cmp_fc])))
            model_comparison.append({"model": f"ARIMA{order}", "mae": _fin(m_, 6), "rmse": _fin(r_, 6), "mape": _fin(mp_, 3), "directional_accuracy": _fin(da_, 2)})

        # Exponential Smoothing as an additional comparison point
        ets_cmp_fc = None
        try:
            trend_kind = "add" if stype == "level" else None
            ets_m = ExponentialSmoothing(train_cmp, trend=trend_kind, damped_trend=(trend_kind is not None)).fit()
            ets_cmp_fc = np.asarray(ets_m.forecast(test_len))
        except Exception:
            ets_cmp_fc = None
        if ets_cmp_fc is not None:
            r_, m_, mp_ = _rmse_mae_mape(test_cmp, ets_cmp_fc)
            da_ = _dir_acc(np.diff(np.concatenate([[train_cmp[-1]], test_cmp])), np.diff(np.concatenate([[train_cmp[-1]], ets_cmp_fc])))
            model_comparison.append({"model": "ExpSmoothing", "mae": _fin(m_, 6), "rmse": _fin(r_, 6), "mape": _fin(mp_, 3), "directional_accuracy": _fin(da_, 2)})

        chart_model_comparison = None
        try:
            names = [row["model"] for row in model_comparison]
            rmses = [row["rmse"] or 0 for row in model_comparison]
            maes = [row["mae"] or 0 for row in model_comparison]
            fig, ax = plt.subplots(figsize=(9, 4.8), dpi=115)
            xs = np.arange(len(names)); w = 0.35
            ax.bar(xs - w / 2, rmses, width=w, color=BLUE, label="RMSE")
            ax.bar(xs + w / 2, maes, width=w, color=AMBER, label="MAE")
            ax.set_xticks(xs); ax.set_xticklabels(names, rotation=15, ha="right")
            ax.set_title("Model Performance Comparison (out-of-sample)")
            ax.set_ylabel(label)
            ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.2, axis="y")
            fig.tight_layout()
            chart_model_comparison = _png(fig)
        except Exception:
            plt.close("all"); chart_model_comparison = None

        # ═══════════════════════════ ⑥ Backtesting (rolling-origin walk-forward) ═══════════════════════════
        n_windows = 4
        min_train = max(20, int(n * 0.5))
        remaining = n - min_train
        backtest_table = []
        backtest_segments = []  # (train_start, train_end, test_end) for the chart
        if remaining >= 5:
            step_size = max(1, remaining // n_windows)
            origins = []
            cursor = min_train
            for i in range(n_windows):
                t_start = cursor
                t_end = min(t_start + step_size, n)
                if t_end - t_start < 2 or t_start >= n - 1:
                    break
                origins.append((t_start, t_end))
                cursor = t_end
            for i, (t_start, t_end) in enumerate(origins):
                train_w = xv[:t_start]
                test_w = xv[t_start:t_end]
                if len(train_w) < 15 or len(test_w) < 2:
                    continue
                try:
                    mw = ARIMA(train_w, order=order).fit()
                    pred_w = np.asarray(mw.get_forecast(steps=len(test_w)).predicted_mean)
                except Exception:
                    continue
                r_, m_, _ = _rmse_mae_mape(test_w, pred_w)
                da_ = _dir_acc(np.diff(np.concatenate([[train_w[-1]], test_w])), np.diff(np.concatenate([[train_w[-1]], pred_w])))
                backtest_table.append({
                    "window": f"Test {i + 1}", "train_end": int(t_start), "test_start": int(t_start),
                    "test_end": int(t_end), "rmse": _fin(r_, 6), "mae": _fin(m_, 6),
                    "directional_accuracy": _fin(da_, 2),
                })
                backtest_segments.append((0, t_start, t_end))

        chart_backtest = None
        if backtest_table:
            try:
                fig, ax = plt.subplots(figsize=(11, 4.8), dpi=115)
                ax.plot(np.arange(n), xv, color=GREY, lw=1, label=f"Full series ({label})")
                colors = [BLUE, GREEN, PURPLE, AMBER, RED]
                for i, seg in enumerate(backtest_segments):
                    _, t_start, t_end = seg
                    ax.axvspan(t_start, t_end, color=colors[i % len(colors)], alpha=0.18)
                    ax.axvline(t_start, color=colors[i % len(colors)], ls=":", lw=1)
                    ax.text(t_start, ax.get_ylim()[1] * 0.97, f"T{i+1}", fontsize=7, color=colors[i % len(colors)])
                ax.set_title("Rolling Forecast Backtest — walk-forward test windows")
                ax.set_xlabel("Period"); ax.set_ylabel(label)
                ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.2)
                fig.tight_layout()
                chart_backtest = _png(fig)
            except Exception:
                plt.close("all"); chart_backtest = None

        # ═══════════════════════════ ⑦ Forecast Performance (aggregate of ⑥) ═══════════════════════════
        if backtest_table:
            agg_rmse = float(np.mean([r["rmse"] for r in backtest_table if r["rmse"] is not None]))
            agg_mae = float(np.mean([r["mae"] for r in backtest_table if r["mae"] is not None]))
            da_vals = [r["directional_accuracy"] for r in backtest_table if r["directional_accuracy"] is not None]
            agg_da = float(np.mean(da_vals)) if da_vals else None
            # MAPE aggregate: recompute from stored windows isn't stored; approximate via mean of per-window MAPE
            mape_vals = []
            for wi, (t_start, t_end) in enumerate(origins[:len(backtest_table)]):
                test_w = xv[t_start:t_end]
                nz = test_w != 0
                if nz.sum():
                    train_w = xv[:t_start]
                    try:
                        mw = ARIMA(train_w, order=order).fit()
                        pred_w = np.asarray(mw.get_forecast(steps=len(test_w)).predicted_mean)
                        mape_vals.append(float(np.mean(np.abs((test_w[nz] - pred_w[nz]) / test_w[nz])) * 100.0))
                    except Exception:
                        pass
            agg_mape = float(np.mean(mape_vals)) if mape_vals else None
            forecast_performance = {
                "mae": _fin(agg_mae, 6), "rmse": _fin(agg_rmse, 6),
                "mape": _fin(agg_mape, 3), "directional_accuracy": _fin(agg_da, 2),
                "n_windows": len(backtest_table),
            }
        else:
            forecast_performance = {"mae": None, "rmse": None, "mape": None, "directional_accuracy": None, "n_windows": 0}

        # ═══════════════════════════ ⑧ Residual Diagnostics ═══════════════════════════
        chart_residual_ts = None
        try:
            fig, ax = plt.subplots(figsize=(9, 4.2), dpi=115)
            ax.plot(np.arange(len(resid)), resid, color=BLUE, lw=0.9)
            ax.axhline(0, color="#111827", lw=0.7, ls="--")
            ax.set_title("Residual Time Series"); ax.set_xlabel("Period"); ax.set_ylabel("Residual")
            ax.grid(alpha=0.2)
            fig.tight_layout()
            chart_residual_ts = _png(fig)
        except Exception:
            plt.close("all"); chart_residual_ts = None

        chart_residual_dist = None
        try:
            fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=115)
            ax.hist(resid, bins=min(40, max(10, n // 8)), color="#93c5fd", edgecolor="white", density=True)
            xs = np.linspace(resid.min(), resid.max(), 100)
            ax.plot(xs, sstats.norm.pdf(xs, np.mean(resid), np.std(resid, ddof=1)), color=RED, lw=1.5, label="Normal")
            ax.legend(fontsize=8, frameon=False)
            ax.set_title("Residual Distribution"); ax.set_xlabel("Residual")
            ax.grid(alpha=0.2)
            fig.tight_layout()
            chart_residual_dist = _png(fig)
        except Exception:
            plt.close("all"); chart_residual_dist = None

        chart_residual_acf = None
        try:
            from statsmodels.tsa.stattools import acf as _acf
            nlags = min(20, n // 4)
            acf_vals = _acf(resid, nlags=nlags, fft=True)
            conf = 1.96 / np.sqrt(len(resid))
            fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=115)
            ax.bar(np.arange(len(acf_vals)), acf_vals, color=BLUE, width=0.6)
            ax.axhline(conf, color=RED, ls="--", lw=1)
            ax.axhline(-conf, color=RED, ls="--", lw=1)
            ax.axhline(0, color="#111827", lw=0.7)
            ax.set_title("Residual ACF"); ax.set_xlabel("Lag"); ax.set_ylabel("Autocorrelation")
            ax.grid(alpha=0.2)
            fig.tight_layout()
            chart_residual_acf = _png(fig)
        except Exception:
            plt.close("all"); chart_residual_acf = None

        residual_diagnostics = {
            "ljung_box_p": _fin(lb_p, 6), "residuals_white_noise": resid_white,
            "jarque_bera_p": _fin(jb_p, 6), "residuals_normal": resid_normal,
        }

        # ═══════════════════════════ ⑨ Forecast Horizon Comparison ═══════════════════════════
        candidate_h = [h for h in [1, 5, 10, 20, 60] if h <= min(60, max(1, n // 2))]
        if not candidate_h:
            candidate_h = [1]
        horizon_table = []
        for h in candidate_h:
            try:
                fc_h = model.get_forecast(steps=h)
                mean_h = np.asarray(fc_h.predicted_mean)
                ci_h = np.asarray(fc_h.conf_int(alpha=0.05))
                width = float(ci_h[-1, 1] - ci_h[-1, 0])
                horizon_table.append({
                    "horizon": h, "forecast": _fin(float(mean_h[-1]), 6),
                    "lower": _fin(float(ci_h[-1, 0]), 6), "upper": _fin(float(ci_h[-1, 1]), 6),
                    "ci_width": _fin(width, 6),
                })
            except Exception:
                continue

        chart_horizon_comparison = None
        if horizon_table:
            try:
                hs = [row["horizon"] for row in horizon_table]
                widths = [row["ci_width"] for row in horizon_table]
                fig, ax = plt.subplots(figsize=(8, 4.6), dpi=115)
                ax.plot(hs, widths, color=PURPLE, marker="o", lw=1.8)
                ax.set_title("Forecast Horizon Comparison — CI width grows with horizon")
                ax.set_xlabel("Horizon (steps ahead)"); ax.set_ylabel("95% CI width")
                ax.grid(alpha=0.2)
                fig.tight_layout()
                chart_horizon_comparison = _png(fig)
            except Exception:
                plt.close("all"); chart_horizon_comparison = None

        # ═══════════════════════════ ③ Actual vs Forecast chart (existing, kept) ═══════════════════════════
        plot = None
        try:
            fig, ax = plt.subplots(figsize=(11.5, 5.2), dpi=118)
            hist_x = np.arange(n)
            ax.plot(hist_x, x.values, color=GREY, lw=0.9, label=f"Observed ({label})")
            ax.plot(hist_x, fitted, color=BLUE, lw=1, alpha=0.8, label="In-sample fit")
            fx = np.arange(n, n + horizon)
            ax.plot(fx, mean_fc, color=GREEN, lw=2, label="Forecast")
            ax.fill_between(fx, ci95[:, 0], ci95[:, 1], color=GREEN, alpha=0.14, label="95% interval")
            ax.fill_between(fx, ci80[:, 0], ci80[:, 1], color=GREEN, alpha=0.28, label="80% interval")
            ax.axvline(n - 0.5, color="#111827", ls=":", lw=1)
            ax.set_xlabel("Period"); ax.set_ylabel(label)
            ax.set_title(f"ARIMA{order} forecast of {col}")
            ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.2)
            fig.tight_layout()
            plot = _png(fig)
        except Exception:
            plt.close("all"); plot = None

        next_val = mean_fc[0]
        interpretation = (
            f"An ARIMA{order} model, selected by AIC, best fit the {label} of {col}. It forecasts a next-period value "
            f"of {next_val:.4g}, with the 95% interval widening from [{ci95[0,0]:.4g}, {ci95[0,1]:.4g}] at step 1 to "
            f"[{ci95[-1,0]:.4g}, {ci95[-1,1]:.4g}] at step {horizon} as uncertainty compounds. "
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

        # ═══════════════════════════ ① Forecast Setup ═══════════════════════════
        forecast_setup = {
            "target_column": col, "series_type": stype, "series_label": label,
            "model": f"ARIMA{order}", "horizon": horizon, "training_period_n": n,
        }

        charts = {
            "actual_vs_forecast": plot,
            "model_comparison": chart_model_comparison,
            "backtest": chart_backtest,
            "residual_ts": chart_residual_ts,
            "residual_dist": chart_residual_dist,
            "residual_acf": chart_residual_acf,
            "horizon_comparison": chart_horizon_comparison,
        }

        results = {
            "status": "ok", "asset": col, "series_type": stype, "series_label": label, "n_obs": n,
            "order": {"p": order[0], "d": order[1], "q": order[2]}, "order_str": f"ARIMA{order}",
            "aic": _fin(aic, 3), "bic": _fin(float(model.bic), 3),
            "rmse": _fin(rmse, 6), "mae": _fin(mae, 6),
            "directional_accuracy": _fin(dir_acc_insample, 2),
            "ljung_box_p": _fin(lb_p, 6), "residuals_white_noise": resid_white,
            "jarque_bera_p": _fin(jb_p, 6), "residuals_normal": resid_normal,
            "horizon": horizon, "forecast": forecast, "next_value": _fin(float(next_val), 6),
            "interpretation": interpretation,
            # New sections
            "forecast_setup": forecast_setup,
            "forecast_summary": forecast_summary,
            "model_comparison": model_comparison,
            "backtest_table": backtest_table,
            "forecast_performance": forecast_performance,
            "residual_diagnostics": residual_diagnostics,
            "horizon_table": horizon_table,
            "charts": charts,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
