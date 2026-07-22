#!/usr/bin/env python3
"""Abnormal Return Analysis — abnormal returns over the whole sample by a chosen
benchmark model. numpy / pandas / statsmodels.

Unlike a single-event study, this profiles the abnormal-return series across the
entire sample: it computes AR each period under one of three benchmark models,
the running CAR, and flags the periods with statistically large abnormal moves.

Input (from abnormal-return-page.tsx):
    data        : list[dict]
    asset_col   : str
    market_col  : str   (required for market-adjusted / market-model)
    model       : "mean"|"market_adjusted"|"market_model"  (default market_model)
    is_returns  : bool
    return_type : "simple"|"log"
Output: { results: {ar stats, car, top_events}, plot }
"""
import sys, json, io, base64
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats as sstats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _fin(x, nd=6):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def _ret(s, is_returns, rtype):
    s = pd.to_numeric(s, errors="coerce")
    if is_returns:
        return s
    if rtype == "log":
        return np.log(s / s.shift(1))
    return s / s.shift(1) - 1.0


def main():
    try:
        p = json.load(sys.stdin)
        rows = p.get("data") or []
        if not rows:
            raise ValueError("No data provided.")
        df = pd.DataFrame(rows)
        asset_col = p.get("asset_col"); market_col = p.get("market_col") or None
        model = (p.get("model") or "market_model").lower()
        is_returns = bool(p.get("is_returns", False))
        rtype = (p.get("return_type") or "simple").lower()
        if not asset_col or asset_col not in df.columns:
            raise ValueError("Select the asset return column.")
        needs_mkt = model in ("market_adjusted", "market_model")
        if needs_mkt and (not market_col or market_col not in df.columns):
            raise ValueError("This benchmark model needs a market return column.")

        a = _ret(df[asset_col], is_returns, rtype)
        if market_col and market_col in df.columns:
            m = _ret(df[market_col], is_returns, rtype)
            reg = pd.concat([a.rename("_a"), m.rename("_m")], axis=1).dropna().reset_index(drop=True)
            av = reg["_a"].values; mv = reg["_m"].values
        else:
            av = a.dropna().reset_index(drop=True).values; mv = None
        n = len(av)
        if n < 20:
            raise ValueError("Need at least 20 aligned observations.")

        alpha = beta = None
        if model == "mean":
            expected = np.full(n, float(np.mean(av))); model_name = "Mean-adjusted"
        elif model == "market_adjusted":
            expected = mv.copy(); model_name = "Market-adjusted (beta = 1)"
        else:
            X = sm.add_constant(mv); fit = sm.OLS(av, X).fit()
            alpha = float(fit.params[0]); beta = float(fit.params[1])
            expected = alpha + beta * mv; model_name = "Market model"

        ar = av - expected
        ar_sd = float(np.std(ar, ddof=1))
        car = float(np.sum(ar))
        cum_ar = np.cumsum(ar)
        t_ar = ar / ar_sd if ar_sd > 0 else np.zeros(n)
        sig_mask = np.abs(t_ar) > 1.96
        n_sig = int(np.sum(sig_mask))
        # top abnormal events by |AR|
        order = np.argsort(-np.abs(ar))[:min(10, n)]
        top_events = [{"index": int(i), "abnormal_return": _fin(float(ar[i]), 6),
                       "t_stat": _fin(float(t_ar[i]), 4), "significant": bool(sig_mask[i])} for i in order]

        skew = float(sstats.skew(ar)); kurt = float(sstats.kurtosis(ar))
        mean_ar = float(np.mean(ar))
        # is mean AR significantly different from 0?
        t_mean = mean_ar / (ar_sd / np.sqrt(n)) if ar_sd > 0 else 0.0
        p_mean = 2 * (1 - sstats.t.cdf(abs(t_mean), df=n - 1))

        plot = None
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.5, 7), dpi=115)
            ax1.bar(range(n), ar * 100, color=np.where(sig_mask, "#dc2626", "#93c5fd"), width=1.0)
            ax1.axhline(1.96 * ar_sd * 100, color="#f59e0b", ls="--", lw=1)
            ax1.axhline(-1.96 * ar_sd * 100, color="#f59e0b", ls="--", lw=1)
            ax1.set_title(f"Abnormal returns ({model_name}) — red = significant")
            ax1.set_ylabel("AR (%)"); ax1.set_xlabel("Period")
            ax2.plot(cum_ar * 100, color="#16a34a", lw=1.6)
            ax2.axhline(0, color="#111827", lw=0.7)
            ax2.set_title(f"Cumulative abnormal return (CAR = {car*100:.2f}%)")
            ax2.set_ylabel("CAR (%)"); ax2.set_xlabel("Period")
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        interpretation = (
            f"Under the {model_name} benchmark, {n_sig} of {n} periods ({100*n_sig/n:.0f}%) show a statistically "
            f"significant abnormal return, and the cumulative abnormal return over the sample is {car:.2%}. "
            + (f"The market model estimated a beta of {beta:.2f} and an alpha of {alpha:.4f}. " if beta is not None else "")
            + (f"The average abnormal return is significantly different from zero (p = {p_mean:.3f}), suggesting a "
               "persistent drift the benchmark does not explain. " if p_mean < 0.05 else
               f"The average abnormal return is not significantly different from zero (p = {p_mean:.3f}), so there is "
               "no systematic drift away from the benchmark. ")
            + "Abnormal returns isolate the asset-specific component of performance — the part not explained by the "
            "chosen benchmark — which is where news, events and idiosyncratic risk show up."
        )

        results = {
            "status": "ok", "asset": asset_col, "market": market_col, "model": model, "model_name": model_name,
            "n_obs": n, "alpha": _fin(alpha, 6) if alpha is not None else None, "beta": _fin(beta, 6) if beta is not None else None,
            "car": _fin(car, 6), "mean_ar": _fin(mean_ar, 6), "ar_std": _fin(ar_sd, 6),
            "mean_ar_t": _fin(t_mean, 4), "mean_ar_p": _fin(float(p_mean), 6), "mean_ar_significant": bool(p_mean < 0.05),
            "n_significant": n_sig, "pct_significant": _fin(100 * n_sig / n, 2),
            "ar_skew": _fin(skew, 4), "ar_kurtosis": _fin(kurt, 4),
            "top_events": top_events,
            "interpretation": interpretation,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
