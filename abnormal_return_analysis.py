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


def _compute_ar(a, m, model):
    """Compute the AR series (and stats) for one asset series `a` against market `m`
    (m may be None). Mirrors the single-asset logic used in main()."""
    if m is not None:
        reg = pd.concat([a.rename("_a"), m.rename("_m")], axis=1).dropna().reset_index(drop=True)
        av = reg["_a"].values; mv = reg["_m"].values
        idx = reg.index
    else:
        s = a.dropna()
        av = s.reset_index(drop=True).values; mv = None
        idx = s.index
    n = len(av)
    if n < 20:
        return None
    if model == "mean":
        expected = np.full(n, float(np.mean(av)))
    elif model == "market_adjusted":
        expected = mv.copy()
    else:
        X = sm.add_constant(mv); fit = sm.OLS(av, X).fit()
        expected = fit.params[0] + fit.params[1] * mv
    ar = av - expected
    ar_sd = float(np.std(ar, ddof=1))
    t_ar = ar / ar_sd if ar_sd > 0 else np.zeros(n)
    sig_mask = np.abs(t_ar) > 1.96
    mean_ar = float(np.mean(ar))
    t_mean = mean_ar / (ar_sd / np.sqrt(n)) if ar_sd > 0 else 0.0
    p_mean = float(2 * (1 - sstats.t.cdf(abs(t_mean), df=n - 1)))
    return {"ar": ar, "index": idx, "n": n, "mean_ar": mean_ar, "t_mean": t_mean, "p_mean": p_mean,
            "cumulative_ar": float(np.sum(ar)), "n_significant": int(np.sum(sig_mask))}


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

        # ---- optional: multi-asset support (additive; does not affect single-asset path) ----
        per_asset = None
        try:
            asset_cols = p.get("asset_cols")
            if asset_cols and isinstance(asset_cols, list):
                rows_out = []
                for c in asset_cols:
                    if not c or c not in df.columns:
                        continue
                    a_i = _ret(df[c], is_returns, rtype)
                    m_i = _ret(df[market_col], is_returns, rtype) if (market_col and market_col in df.columns) else None
                    res_i = _compute_ar(a_i, m_i, model)
                    if res_i is None:
                        continue
                    ar_sd_i = float(np.std(res_i["ar"], ddof=1))
                    t_mean_i = res_i["mean_ar"] / (ar_sd_i / np.sqrt(res_i["n"])) if ar_sd_i > 0 else 0.0
                    rows_out.append({
                        "asset": c, "mean_ar": _fin(res_i["mean_ar"], 6), "mean_ar_t": _fin(t_mean_i, 4),
                        "mean_ar_p": _fin(res_i["p_mean"], 6), "cumulative_ar": _fin(res_i["cumulative_ar"], 6),
                        "n_significant": res_i["n_significant"],
                    })
                if rows_out:
                    per_asset = rows_out
        except Exception:
            per_asset = None

        # ---- optional: by-period aggregation (needs a usable date column) ----
        ar_by_period = None
        ar_period_note = None
        try:
            date_col = p.get("date_col")
            if date_col and date_col in df.columns:
                dser = pd.to_datetime(df[date_col], errors="coerce")
                # reconstruct aligned dates using same dropna logic as ar computation
                if market_col and market_col in df.columns:
                    tmp = pd.concat([a.rename("_a"), m.rename("_m"), dser.rename("_d")], axis=1).dropna(subset=["_a", "_m"]).reset_index(drop=True)
                else:
                    tmp = pd.concat([a.rename("_a"), dser.rename("_d")], axis=1).dropna(subset=["_a"]).reset_index(drop=True)
                dates_for_ar = pd.to_datetime(tmp["_d"], errors="coerce")
                if len(dates_for_ar) == n and dates_for_ar.notna().sum() >= n * 0.8:
                    per_df = pd.DataFrame({"ar": ar, "date": dates_for_ar.values})
                    per_df = per_df.dropna(subset=["date"])
                    per_df["period"] = pd.to_datetime(per_df["date"]).dt.to_period("M").astype(str)
                    grp = per_df.groupby("period")["ar"].agg(["mean", "count"]).reset_index()
                    ar_by_period = [{"period": row["period"], "mean_ar": _fin(row["mean"], 6), "n": int(row["count"])} for _, row in grp.iterrows()]
                else:
                    ar_period_note = "Date column present but could not be reliably aligned to the AR series."
            else:
                ar_period_note = "No date column provided; by-period aggregation skipped."
        except Exception:
            ar_by_period = None
            ar_period_note = "By-period aggregation failed on the provided date column."

        skew = float(sstats.skew(ar)); kurt = float(sstats.kurtosis(ar))
        mean_ar = float(np.mean(ar))
        # is mean AR significantly different from 0?
        t_mean = mean_ar / (ar_sd / np.sqrt(n)) if ar_sd > 0 else 0.0
        p_mean = 2 * (1 - sstats.t.cdf(abs(t_mean), df=n - 1))

        plot = None
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11.5, 10), dpi=115)
            ax1.bar(range(n), ar * 100, color=np.where(sig_mask, "#dc2626", "#93c5fd"), width=1.0)
            ax1.axhline(1.96 * ar_sd * 100, color="#f59e0b", ls="--", lw=1)
            ax1.axhline(-1.96 * ar_sd * 100, color="#f59e0b", ls="--", lw=1)
            ax1.set_title(f"Abnormal returns ({model_name}) — red = significant")
            ax1.set_ylabel("AR (%)"); ax1.set_xlabel("Period")
            ax2.plot(cum_ar * 100, color="#16a34a", lw=1.6)
            ax2.axhline(0, color="#111827", lw=0.7)
            ax2.set_title(f"Cumulative abnormal return (CAR = {car*100:.2f}%)")
            ax2.set_ylabel("CAR (%)"); ax2.set_xlabel("Period")
            ax3.hist(ar * 100, bins=min(30, max(10, n // 5)), color="#6366f1", edgecolor="white")
            ax3.axvline(0, color="#111827", lw=0.7)
            ax3.set_title("Abnormal return distribution")
            ax3.set_xlabel("AR (%)"); ax3.set_ylabel("Frequency")
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
        if per_asset is not None:
            results["per_asset"] = per_asset
        if ar_by_period is not None:
            results["ar_by_period"] = ar_by_period
        elif ar_period_note is not None:
            results["ar_by_period_note"] = ar_period_note
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
