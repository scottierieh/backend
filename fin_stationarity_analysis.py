#!/usr/bin/env python3
"""Stationarity Analysis — ADF and KPSS tests on a financial series. statsmodels.

Tests whether a series is stationary (stable mean/variance over time). Prices
are typically non-stationary (random walks); returns typically are stationary.

Input (from fin-stationarity-page.tsx):
    data        : list[dict]
    asset_col   : str
    series_type : "level"|"returns"|"log_returns"   what to test (default level)
Output: { results: {adf, kpss, pp, verdict}, plot }
"""
import sys, json, io, base64
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

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
        col = p.get("asset_col")
        stype = (p.get("series_type") or "level").lower()
        if not col or col not in df.columns:
            raise ValueError("Select a column.")

        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if stype == "returns":
            x = (s / s.shift(1) - 1.0).dropna(); label = "simple returns"
        elif stype == "log_returns":
            if (s <= 0).any():
                raise ValueError("Log returns require positive values.")
            x = np.log(s / s.shift(1)).dropna(); label = "log returns"
        else:
            x = s; label = "level"
        x = pd.Series(x).reset_index(drop=True)
        n = len(x)
        if n < 20:
            raise ValueError("Need at least 20 observations.")

        # ADF: null = unit root (non-stationary)
        adf_stat, adf_p, adf_lags, adf_nobs, adf_crit, _ = adfuller(x.values, autolag="AIC")
        # KPSS: null = stationary
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(x.values, regression="c", nlags="auto")

        # PP (Phillips-Perron): null = unit root (non-stationary), same interpretation as ADF.
        # Optional dependency (arch package) — degrade gracefully if unavailable.
        pp_stat = pp_p = None
        pp_stationary = None
        pp_note = None
        try:
            from arch.unitroot import PhillipsPerron
            pp_res = PhillipsPerron(x.values)
            pp_stat = float(pp_res.stat)
            pp_p = float(pp_res.pvalue)
            pp_stationary = bool(pp_p < 0.05)
        except ImportError:
            pp_note = "Phillips-Perron test skipped: the optional 'arch' package is not installed."
        except Exception as e:
            pp_note = f"Phillips-Perron test failed: {e}"

        adf_stationary = bool(adf_p < 0.05)      # reject unit root -> stationary
        kpss_stationary = bool(kpss_p >= 0.05)   # fail to reject stationarity -> stationary

        if adf_stationary and kpss_stationary:
            verdict = "stationary"; confidence = "both tests agree"
        elif not adf_stationary and not kpss_stationary:
            verdict = "non-stationary"; confidence = "both tests agree"
        elif adf_stationary and not kpss_stationary:
            verdict = "difference-stationary or trend-stationary (mixed signals)"; confidence = "tests disagree"
        else:
            verdict = "possibly trend-stationary (mixed signals)"; confidence = "tests disagree"

        # rolling mean/std for the plot
        w = max(5, n // 20)
        roll_mean = x.rolling(w).mean()
        roll_std = x.rolling(w).std()

        plot = None
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), dpi=115, sharex=True)
            ax1.plot(x.values, color="#94a3b8", lw=0.7, label=label)
            ax1.plot(roll_mean.values, color="#2563eb", lw=1.5, label=f"Rolling mean ({w})")
            ax1.set_title(f"{col} ({label}) with rolling mean"); ax1.legend(fontsize=8, frameon=False); ax1.grid(alpha=0.2)
            ax2.plot(roll_std.values, color="#f59e0b", lw=1.4, label=f"Rolling std ({w})")
            ax2.set_title("Rolling volatility"); ax2.set_xlabel("Period"); ax2.legend(fontsize=8, frameon=False); ax2.grid(alpha=0.2)
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        interpretation = (
            f"Testing the {label} of {col} ({n} observations): the ADF test {'rejects' if adf_stationary else 'does not reject'} "
            f"a unit root (p = {adf_p:.3f}), and the KPSS test {'does not reject' if kpss_stationary else 'rejects'} "
            f"stationarity (p = {kpss_p:.3f}). Together they indicate the series is {verdict} ({confidence}). "
            + ("A stationary series has a stable mean and variance over time, which is the precondition for most "
               "time-series models (ARMA, regression) to give valid results. "
               if verdict == "stationary" else
               "A non-stationary series drifts — its mean or variance changes over time — so it must be differenced "
               "(e.g. converted from prices to returns) before standard modelling. Regressing two non-stationary "
               "series against each other risks spurious correlation. "
               if verdict == "non-stationary" else
               "The tests disagree, which often points to a trend-stationary series or a borderline case; inspect the "
               "rolling mean and consider detrending or differencing. ")
        )

        results = {
            "status": "ok", "asset": col, "series_type": stype, "series_label": label, "n_obs": n,
            "adf": {"statistic": _fin(adf_stat, 4), "p_value": _fin(adf_p, 6), "lags": int(adf_lags),
                    "critical": {k: _fin(v, 4) for k, v in adf_crit.items()}, "stationary": adf_stationary},
            "kpss": {"statistic": _fin(kpss_stat, 4), "p_value": _fin(kpss_p, 6), "lags": int(kpss_lags),
                     "critical": {k: _fin(v, 4) for k, v in kpss_crit.items()}, "stationary": kpss_stationary},
            "pp": ({"statistic": _fin(pp_stat, 4), "p_value": _fin(pp_p, 6), "stationary": pp_stationary}
                   if pp_stat is not None else None),
            "pp_note": pp_note,
            "verdict": verdict, "confidence": confidence, "is_stationary": bool(verdict == "stationary"),
            "interpretation": interpretation,
            "test_comparison": [
                {"test": "ADF", "statistic": _fin(adf_stat, 4), "p_value": _fin(adf_p, 6), "stationary": adf_stationary},
                {"test": "KPSS", "statistic": _fin(kpss_stat, 4), "p_value": _fin(kpss_p, 6), "stationary": kpss_stationary},
            ] + ([{"test": "PP", "statistic": _fin(pp_stat, 4), "p_value": _fin(pp_p, 6), "stationary": pp_stationary}]
                 if pp_stat is not None else []),
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
