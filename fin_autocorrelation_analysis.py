#!/usr/bin/env python3
"""Autocorrelation Analysis — ACF/PACF and Ljung-Box for a return series.
statsmodels.

Tests whether a series (returns, or their absolute/squared values) is
serially correlated — i.e. predictable from its own past.

Also builds a full step-6 report with 7 additive sections (autocorrelation
summary, ACF, PACF, ACF/PACF comparison, Ljung-Box, return-vs-squared-return
ACF, and an optional rolling autocorrelation) each with its own chart where
applicable, rendered by the frontend via VisualizationTabs.

Input (from fin-autocorrelation-page.tsx):
    data        : list[dict]
    asset_col   : str
    is_returns  : bool
    return_type : "simple"|"log"
    nlags       : int   (default 20, always computed up to at least 10)
    transform   : "raw"|"abs"|"squared"   (default raw)
Output: { results: {acf_summary, lags[], ljung_box[], acf_pacf_table,
          rolling_acf_table, rolling_acf_note, charts: {acf, pacf,
          return_vs_squared, rolling_acf}}, plot }
"""
import sys, json, io, base64
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#2563eb"
GREEN = "#16a34a"
RED = "#dc2626"
AMBER = "#d97706"


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


def _acf_bar(ax, lags, vals, ci, color, title):
    ax.bar(lags, vals, color=color, width=0.6)
    ax.axhline(ci, color=RED, ls="--", lw=1)
    ax.axhline(-ci, color=RED, ls="--", lw=1)
    ax.axhline(0, color="#111827", lw=0.7)
    ax.set_title(title)
    ax.set_xlabel("Lag")
    ax.grid(alpha=0.2)


def main():
    try:
        p = json.load(sys.stdin)
        rows = p.get("data") or []
        if not rows:
            raise ValueError("No data provided.")
        df = pd.DataFrame(rows)
        col = p.get("asset_col")
        is_returns = bool(p.get("is_returns", False))
        rtype = (p.get("return_type") or "simple").lower()
        nlags_req = int(p.get("nlags") or 20)
        transform = (p.get("transform") or "raw").lower()
        if not col or col not in df.columns:
            raise ValueError("Select the return column.")

        s = pd.to_numeric(df[col], errors="coerce")
        if is_returns:
            ret = s.dropna()
        elif rtype == "log":
            if (s <= 0).any():
                raise ValueError("Log returns require positive prices.")
            ret = np.log(s / s.shift(1)).dropna()
        else:
            ret = (s / s.shift(1) - 1.0).dropna()
        ret = ret.reset_index(drop=True)
        n = len(ret)
        if n < 20:
            raise ValueError("Need at least 20 observations.")
        max_allowed = n // 3
        # Always compute at least up to lag 10 (up to 20 if cheap) so the
        # Autocorrelation Summary (lag 1/5/10) is always available.
        nlags = max(min(nlags_req, max_allowed), min(10, max_allowed))
        nlags = min(nlags, max_allowed)

        rv = ret.values
        series = rv
        label = "returns"
        if transform == "abs":
            series = np.abs(rv); label = "absolute returns"
        elif transform == "squared":
            series = rv ** 2; label = "squared returns"

        acf_vals, _ = acf(series, nlags=nlags, alpha=0.05, fft=True)
        pacf_vals, _ = pacf(series, nlags=nlags, alpha=0.05)
        ci = 1.96 / np.sqrt(n)

        lags_out = []
        for k in range(1, nlags + 1):
            av = float(acf_vals[k]); pv = float(pacf_vals[k])
            lags_out.append({
                "lag": k, "acf": _fin(av, 5), "pacf": _fin(pv, 5),
                "acf_significant": bool(abs(av) > ci), "pacf_significant": bool(abs(pv) > ci),
            })
        by_lag = {row["lag"]: row for row in lags_out}

        # Ljung-Box at a few horizons (existing computation, reused as-is).
        lb = acorr_ljungbox(series, lags=[min(10, nlags), min(20, nlags)], return_df=True)
        ljung = []
        for lag, row in lb.iterrows():
            ljung.append({"lag": int(lag), "stat": _fin(float(row["lb_stat"]), 4),
                          "p_value": _fin(float(row["lb_pvalue"]), 6),
                          "significant": bool(row["lb_pvalue"] < 0.05)})
        overall_sig = any(l["significant"] for l in ljung)
        n_sig_acf = sum(1 for l in lags_out if l["acf_significant"])

        # ─────────────────────── ① Autocorrelation Summary ───────────────────────
        acf_vals_signed = [row["acf"] for row in lags_out]
        max_acf = max(acf_vals_signed, key=lambda v: abs(v) if v is not None else 0) if acf_vals_signed else None
        lb10 = next((l for l in ljung if l["lag"] == min(10, nlags)), (ljung[0] if ljung else None))
        acf_summary = {
            "lag1_acf": by_lag.get(1, {}).get("acf"),
            "lag5_acf": by_lag.get(5, {}).get("acf") if nlags >= 5 else None,
            "lag10_acf": by_lag.get(10, {}).get("acf") if nlags >= 10 else None,
            "max_acf": _fin(max_acf, 5),
            "n_significant_lags": n_sig_acf,
            "ljung_box_p_value": lb10["p_value"] if lb10 else None,
            "ljung_box_lag": lb10["lag"] if lb10 else None,
        }

        # ─────────────────────────── ② / ③ ACF & PACF charts (separate PNGs) ───────────────────────────
        lags_axis = list(range(1, nlags + 1))
        chart_acf = None
        try:
            fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=115)
            _acf_bar(ax, lags_axis, [l["acf"] for l in lags_out], ci, BLUE, f"ACF of {label}")
            ax.set_ylabel("Autocorrelation")
            fig.tight_layout()
            chart_acf = _png(fig)
        except Exception:
            plt.close("all"); chart_acf = None

        chart_pacf = None
        try:
            fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=115)
            _acf_bar(ax, lags_axis, [l["pacf"] for l in lags_out], ci, GREEN, f"PACF of {label}")
            ax.set_ylabel("Partial autocorrelation")
            fig.tight_layout()
            chart_pacf = _png(fig)
        except Exception:
            plt.close("all"); chart_pacf = None

        # Combined 2-panel plot kept for backward compatibility (legacy `plot` field).
        plot = None
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), dpi=115)
            _acf_bar(ax1, lags_axis, [l["acf"] for l in lags_out], ci, BLUE, f"ACF of {label}")
            ax1.set_ylabel("Autocorrelation")
            _acf_bar(ax2, lags_axis, [l["pacf"] for l in lags_out], ci, GREEN, f"PACF of {label}")
            ax2.set_ylabel("Partial autocorrelation")
            fig.tight_layout()
            plot = _png(fig)
        except Exception:
            plt.close("all"); plot = None

        # ─────────────────────────── ④ ACF / PACF Comparison table ───────────────────────────
        # Identical underlying data to ②/③ — just re-surfaced as its own table.
        acf_pacf_table = [{"lag": l["lag"], "acf": l["acf"], "pacf": l["pacf"]} for l in lags_out]

        # ─────────────────────────── ⑥ Return vs Squared Return ACF (new) ───────────────────────────
        acf_ret_vals, _ = acf(rv, nlags=nlags, alpha=0.05, fft=True)
        sq = rv ** 2
        acf_sq_vals, _ = acf(sq, nlags=nlags, alpha=0.05, fft=True)
        return_vs_squared_table = []
        for k in range(1, nlags + 1):
            return_vs_squared_table.append({
                "lag": k,
                "acf_returns": _fin(float(acf_ret_vals[k]), 5),
                "acf_squared_returns": _fin(float(acf_sq_vals[k]), 5),
            })
        mean_abs_ret_acf = float(np.mean(np.abs(acf_ret_vals[1:nlags + 1])))
        mean_abs_sq_acf = float(np.mean(np.abs(acf_sq_vals[1:nlags + 1])))
        vol_clustering_signal = bool(mean_abs_sq_acf > mean_abs_ret_acf * 1.5 and mean_abs_sq_acf > ci)

        chart_return_vs_squared = None
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 7.0), dpi=115, sharex=True)
            _acf_bar(ax1, lags_axis, [float(v) for v in acf_ret_vals[1:nlags + 1]], ci, BLUE, "ACF of Returns")
            ax1.set_ylabel("Autocorrelation")
            _acf_bar(ax2, lags_axis, [float(v) for v in acf_sq_vals[1:nlags + 1]], ci, AMBER, "ACF of Squared Returns")
            ax2.set_ylabel("Autocorrelation")
            ax2.set_xlabel("Lag")
            fig.tight_layout()
            chart_return_vs_squared = _png(fig)
        except Exception:
            plt.close("all"); chart_return_vs_squared = None

        vol_clustering_note = (
            f"The average |ACF| of squared returns ({mean_abs_sq_acf:.4f}) is "
            + (f"notably higher than that of raw returns ({mean_abs_ret_acf:.4f})"
               if vol_clustering_signal else
               f"similar to that of raw returns ({mean_abs_ret_acf:.4f})")
            + (", which hints at volatility clustering — calm and turbulent periods tend to persist even though the "
               "sign of returns is close to unpredictable. See the Volatility Modeling page for a full ARCH/GARCH "
               "treatment of this effect."
               if vol_clustering_signal else
               ". There is little sign of volatility clustering at these lags in this sample.")
        )

        # ─────────────────────────── ⑦ Rolling Autocorrelation (advanced, best-effort) ───────────────────────────
        rolling_acf_table = None
        chart_rolling_acf = None
        rolling_acf_note = None
        windows = [w for w in (60, 90) if n >= w * 2]
        if windows:
            try:
                r_t = pd.Series(rv)
                r_lag1 = r_t.shift(1)
                r_lag5 = r_t.shift(5)
                roll_results = {}
                for w in windows:
                    lag1_roll = r_t.rolling(w).corr(r_lag1)
                    lag5_roll = r_t.rolling(w).corr(r_lag5)
                    roll_results[w] = (lag1_roll, lag5_roll)

                rolling_acf_table = []
                for w in windows:
                    lag1_roll, lag5_roll = roll_results[w]
                    rolling_acf_table.append({
                        "window": int(w),
                        "mean_lag1_acf": _fin(float(lag1_roll.mean(skipna=True)), 4),
                        "mean_lag5_acf": _fin(float(lag5_roll.mean(skipna=True)), 4),
                        "max_lag1_acf": _fin(float(lag1_roll.max(skipna=True)), 4),
                        "min_lag1_acf": _fin(float(lag1_roll.min(skipna=True)), 4),
                    })

                fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=115)
                colors = [BLUE, AMBER]
                for i, w in enumerate(windows):
                    lag1_roll, _ = roll_results[w]
                    ax.plot(lag1_roll.values, color=colors[i % len(colors)], lw=1.2, label=f"{w}-period window")
                ax.axhline(0, color="#111827", lw=0.7)
                ax.set_title("Rolling Lag-1 Autocorrelation")
                ax.set_xlabel("Period"); ax.set_ylabel("Rolling ACF (lag 1)")
                ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.2)
                fig.tight_layout()
                chart_rolling_acf = _png(fig)
            except Exception:
                plt.close("all"); chart_rolling_acf = None
        else:
            rolling_acf_note = (
                f"Not enough observations ({n}) for a stable rolling autocorrelation window "
                "(needs at least 120 for a 60-period window); this section is skipped."
            )

        interpretation = (
            f"Testing {n} {label} for serial correlation up to lag {nlags}, {n_sig_acf} of {nlags} autocorrelations "
            f"exceed the 95% significance band. "
            + (f"The Ljung-Box test rejects the null of no autocorrelation (p < 0.05), so the series is serially "
               "dependent — its past carries information about its future. "
               if overall_sig else
               "The Ljung-Box test does not reject the null of no autocorrelation, so the series behaves like white "
               "noise at these lags. ")
            + ("For raw returns, significant autocorrelation would hint at short-term predictability (or "
               "microstructure effects); its usual absence supports weak-form market efficiency. "
               if transform == "raw" else
               "Autocorrelation in absolute or squared returns is the signature of volatility clustering — the level "
               "of returns is unpredictable but their magnitude is not, which is exactly what GARCH models exploit. ")
        )

        charts = {
            "acf": chart_acf,
            "pacf": chart_pacf,
            "return_vs_squared": chart_return_vs_squared,
            "rolling_acf": chart_rolling_acf,
        }

        results = {
            "status": "ok", "asset": col, "n_obs": n, "nlags": nlags, "transform": transform, "series_label": label,
            "conf_band": _fin(ci, 5), "lags": lags_out, "ljung_box": ljung,
            "n_significant_acf": n_sig_acf, "autocorrelated": bool(overall_sig),
            "interpretation": interpretation,
            "acf_summary": acf_summary,
            "acf_pacf_table": acf_pacf_table,
            "return_vs_squared_table": return_vs_squared_table,
            "vol_clustering_signal": vol_clustering_signal,
            "vol_clustering_note": vol_clustering_note,
            "rolling_acf_table": rolling_acf_table,
            "rolling_acf_note": rolling_acf_note,
            "charts": charts,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
