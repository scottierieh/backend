#!/usr/bin/env python3
"""Autocorrelation Analysis — ACF/PACF and Ljung-Box for a return series.
statsmodels.

Tests whether a series (returns, or their absolute/squared values) is
serially correlated — i.e. predictable from its own past.

Input (from fin-autocorrelation-page.tsx):
    data        : list[dict]
    asset_col   : str
    is_returns  : bool
    return_type : "simple"|"log"
    nlags       : int   (default 20)
    transform   : "raw"|"abs"|"squared"   (default raw)
Output: { results: {acf[], pacf[], ljung_box}, plot }
"""
import sys, json, io, base64
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

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
        is_returns = bool(p.get("is_returns", False))
        rtype = (p.get("return_type") or "simple").lower()
        nlags = int(p.get("nlags") or 20)
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
        nlags = max(1, min(nlags, n // 3))

        series = ret.values
        label = "returns"
        if transform == "abs":
            series = np.abs(series); label = "absolute returns"
        elif transform == "squared":
            series = series ** 2; label = "squared returns"

        acf_vals, acf_conf = acf(series, nlags=nlags, alpha=0.05, fft=True)
        pacf_vals, pacf_conf = pacf(series, nlags=nlags, alpha=0.05)
        ci = 1.96 / np.sqrt(n)

        lags_out = []
        for k in range(1, nlags + 1):
            av = float(acf_vals[k]); pv = float(pacf_vals[k])
            lags_out.append({
                "lag": k, "acf": _fin(av, 5), "pacf": _fin(pv, 5),
                "acf_significant": bool(abs(av) > ci), "pacf_significant": bool(abs(pv) > ci),
            })

        # Ljung-Box at a few horizons
        lb = acorr_ljungbox(series, lags=[min(10, nlags), min(20, nlags)], return_df=True)
        ljung = []
        for lag, row in lb.iterrows():
            ljung.append({"lag": int(lag), "stat": _fin(float(row["lb_stat"]), 4),
                          "p_value": _fin(float(row["lb_pvalue"]), 6),
                          "significant": bool(row["lb_pvalue"] < 0.05)})
        overall_sig = any(l["significant"] for l in ljung)
        n_sig_acf = sum(1 for l in lags_out if l["acf_significant"])

        plot = None
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), dpi=115)
            lags = list(range(1, nlags + 1))
            ax1.bar(lags, [l["acf"] for l in lags_out], color="#2563eb", width=0.6)
            ax1.axhline(ci, color="#dc2626", ls="--", lw=1); ax1.axhline(-ci, color="#dc2626", ls="--", lw=1)
            ax1.axhline(0, color="#111827", lw=0.7)
            ax1.set_title(f"ACF of {label}"); ax1.set_xlabel("Lag"); ax1.set_ylabel("Autocorrelation")
            ax2.bar(lags, [l["pacf"] for l in lags_out], color="#16a34a", width=0.6)
            ax2.axhline(ci, color="#dc2626", ls="--", lw=1); ax2.axhline(-ci, color="#dc2626", ls="--", lw=1)
            ax2.axhline(0, color="#111827", lw=0.7)
            ax2.set_title(f"PACF of {label}"); ax2.set_xlabel("Lag"); ax2.set_ylabel("Partial autocorrelation")
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

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

        results = {
            "status": "ok", "asset": col, "n_obs": n, "nlags": nlags, "transform": transform, "series_label": label,
            "conf_band": _fin(ci, 5), "lags": lags_out, "ljung_box": ljung,
            "n_significant_acf": n_sig_acf, "autocorrelated": bool(overall_sig),
            "interpretation": interpretation,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
