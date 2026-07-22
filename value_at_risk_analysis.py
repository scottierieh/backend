#!/usr/bin/env python3
"""Value at Risk (VaR) — historical, parametric (normal) and Monte Carlo VaR/CVaR.
numpy / pandas / scipy, matplotlib for the return-distribution + backtest plot.

Input (from value-at-risk-page.tsx):
    data          : list[dict]
    value_col     : string          price or return column
    date_col      : string | None   optional ordering column
    is_returns    : bool
    log_returns   : bool            simple vs. log returns (ignored if is_returns)
    confidence    : float           chosen confidence level (0.90 / 0.95 / 0.99)
    position      : number          optional position size for currency VaR

Output: { results: {...}, plot }.

`results` field names are an exact drop-in replacement for what
value-at-risk-page.tsx used to compute client-side (n_returns, value_col,
freq, conf, mean, stdev, pos, byConf, chosen, nBreaches, expectedBreaches,
histogram, tailCounts, tailCount, actualRate, expectedRate) plus a handful of
additive extras (var_series / backtest table) guarded by optional-chaining on
the frontend so nothing existing breaks if they're absent.
"""
import sys, json, io, base64
import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONFS = ["0.90", "0.95", "0.99"]


def _fin(x, nd=6):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def _to_returns(values, is_returns, log_ret):
    values = [v for v in values if np.isfinite(v)]
    if is_returns:
        return np.array(values, dtype=float)
    out = []
    for i in range(1, len(values)):
        p0, p1 = values[i - 1], values[i]
        if not (np.isfinite(p0) and np.isfinite(p1)) or p0 <= 0 or p1 <= 0:
            continue
        out.append(np.log(p1 / p0) if log_ret else (p1 / p0 - 1))
    return np.array(out, dtype=float)


def _historical_var(returns, conf):
    sorted_r = np.sort(returns)
    n = len(sorted_r)
    idx = max(0, min(n - 1, int(np.floor(n * (1 - conf)))))
    var_v = float(sorted_r[idx])
    tail = sorted_r[: idx + 1]
    cvar_v = float(np.mean(tail)) if tail.size else var_v
    return var_v, cvar_v


def _monte_carlo_var(m, s, conf, sims=5000, seed=42):
    rng = np.random.default_rng(seed)
    draws = m + s * rng.standard_normal(sims)
    return _historical_var(draws, conf)


def main():
    try:
        p = json.load(sys.stdin)
        rows = p.get("data") or []
        if not rows:
            raise ValueError("No data provided.")
        df = pd.DataFrame(rows)

        value_col = p.get("value_col")
        if not value_col or value_col not in df.columns:
            raise ValueError("Select the price or return column.")
        date_col = p.get("date_col")
        if date_col and date_col in df.columns:
            df = df.sort_values(date_col, kind="stable").reset_index(drop=True)

        is_returns = bool(p.get("is_returns", False))
        log_ret = bool(p.get("log_returns", False))
        freq = str(p.get("freq") or "252")
        conf = float(p.get("confidence") or 0.95)
        if not (0.5 < conf < 1):
            raise ValueError("Confidence must be between 0.5 and 1 (e.g. 0.95).")
        pos = float(p.get("position") or 0)

        values = pd.to_numeric(df[value_col], errors="coerce").tolist()
        returns = _to_returns(values, is_returns, log_ret)
        if returns.size < 10:
            raise ValueError("Not enough valid observations to compute VaR.")

        m = float(np.mean(returns))
        s = float(np.std(returns, ddof=1)) if returns.size > 1 else 0.0

        conf_key = f"{conf:.2f}"

        by_conf = {}
        for cc in CONFS:
            c = float(cc)
            hist_v, hist_cv = _historical_var(returns, c)
            z = float(stats.norm.ppf(c))
            param_var = m - z * s
            a = 1 - c
            es_factor = stats.norm.pdf(z) / a if a > 0 else 0.0
            param_cvar = m - s * es_factor
            mc_v, mc_cv = _monte_carlo_var(m, s, c)
            by_conf[cc] = {
                "historical": {"varV": _fin(hist_v, 6), "cvarV": _fin(hist_cv, 6)},
                "parametric": {"varV": _fin(param_var, 6), "cvarV": _fin(param_cvar, 6)},
                "monteCarlo": {"varV": _fin(mc_v, 6), "cvarV": _fin(mc_cv, 6)},
            }

        # Make sure the chosen confidence is present even if it's a value
        # outside the standard 90/95/99 set.
        if conf_key not in by_conf:
            hist_v, hist_cv = _historical_var(returns, conf)
            z = float(stats.norm.ppf(conf))
            a = 1 - conf
            param_var = m - z * s
            param_cvar = m - s * (stats.norm.pdf(z) / a if a > 0 else 0.0)
            mc_v, mc_cv = _monte_carlo_var(m, s, conf)
            by_conf[conf_key] = {
                "historical": {"varV": _fin(hist_v, 6), "cvarV": _fin(hist_cv, 6)},
                "parametric": {"varV": _fin(param_var, 6), "cvarV": _fin(param_cvar, 6)},
                "monteCarlo": {"varV": _fin(mc_v, 6), "cvarV": _fin(mc_cv, 6)},
            }

        chosen = by_conf[conf_key]
        n = returns.size
        n_breaches = int(np.sum(returns <= chosen["historical"]["varV"]))
        expected_breaches = n * (1 - conf)

        tail_counts = {cc: int(round(n * (1 - float(cc)))) for cc in CONFS}
        if conf_key not in tail_counts:
            tail_counts[conf_key] = int(round(n * (1 - conf)))
        tail_count = tail_counts[conf_key]

        actual_rate = (n_breaches / n) if n > 0 else float("nan")
        expected_rate = 1 - conf

        sorted_r = np.sort(returns)
        bin_count = int(min(24, max(8, round(np.sqrt(n)))))
        lo, hi = float(sorted_r[0]), float(sorted_r[-1])
        bin_w = (hi - lo) / bin_count if bin_count else 1.0
        if bin_w == 0:
            bin_w = 1.0
        counts = np.zeros(bin_count, dtype=int)
        for x in returns:
            b = int(np.floor((x - lo) / bin_w))
            b = min(bin_count - 1, max(0, b))
            counts[b] += 1
        histogram = [
            {
                "bin": _fin(lo + (i + 0.5) * bin_w, 6),
                "count": int(counts[i]),
                "isVar": bool((lo + i * bin_w) <= chosen["historical"]["varV"]),
            }
            for i in range(bin_count)
        ]

        # --- Additive extras (guarded on the frontend; parity fields above are untouched) ---
        # Rolling backtest: VaR vs. actual return per period (using the chosen historical VaR
        # as a static threshold, the same convention used for nBreaches above).
        var_series = [
            {"i": i, "ret": _fin(float(x), 6), "breach": bool(x <= chosen["historical"]["varV"])}
            for i, x in enumerate(returns.tolist())
        ]
        backtest_table = [
            {
                "confidence": _fin(float(cc), 4),
                "var": by_conf[cc]["historical"]["varV"],
                "breaches": int(np.sum(returns <= by_conf[cc]["historical"]["varV"])),
                "expected": _fin(n * (1 - float(cc)), 2),
                "rate": _fin((np.sum(returns <= by_conf[cc]["historical"]["varV"]) / n) if n > 0 else None, 4),
            }
            for cc in CONFS
        ]

        # --- Plot: return distribution with VaR/CVaR lines, tail zoom, and VaR-vs-actual series ---
        plot = None
        try:
            fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), dpi=115)
            ax1, ax2, ax3 = axes

            ax1.hist(returns * 100, bins=bin_count, color="#93c5fd", edgecolor="white")
            ax1.axvline(chosen["historical"]["varV"] * 100, color="#dc2626", ls="--", lw=1.5,
                        label=f"VaR {conf:.0%}")
            ax1.axvline(chosen["historical"]["cvarV"] * 100, color="#7f1d1d", ls=":", lw=1.5,
                        label=f"CVaR {conf:.0%}")
            ax1.set_xlabel("Return (%)"); ax1.set_ylabel("Frequency")
            ax1.set_title("Return distribution & VaR"); ax1.legend(fontsize=8, frameon=False)

            tail_vals = sorted_r[: max(1, tail_count)]
            ax2.hist(tail_vals * 100, bins=min(15, max(5, len(tail_vals))), color="#fca5a5", edgecolor="white")
            ax2.axvline(chosen["historical"]["varV"] * 100, color="#dc2626", ls="--", lw=1.5)
            ax2.axvline(chosen["historical"]["cvarV"] * 100, color="#7f1d1d", ls=":", lw=1.5)
            ax2.set_xlabel("Return (%)"); ax2.set_title("Tail (CVaR) distribution")

            idx_arr = np.arange(n)
            colors = np.where(returns <= chosen["historical"]["varV"], "#dc2626", "#2563eb")
            ax3.scatter(idx_arr, returns * 100, s=10, c=colors, alpha=0.7)
            ax3.axhline(chosen["historical"]["varV"] * 100, color="#dc2626", ls="--", lw=1.2, label="VaR")
            ax3.set_xlabel("Period"); ax3.set_ylabel("Return (%)")
            ax3.set_title(f"Backtest: {n_breaches} breaches vs. {expected_breaches:.1f} expected")
            ax3.legend(fontsize=8, frameon=False)

            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        interpretation = (
            f"At {conf:.0%} confidence, the historical VaR of {value_col} is {chosen['historical']['varV']:.2%} "
            f"per period, with a CVaR (expected shortfall) of {chosen['historical']['cvarV']:.2%}. Parametric "
            f"VaR ({chosen['parametric']['varV']:.2%}) and Monte Carlo VaR ({chosen['monteCarlo']['varV']:.2%}) "
            f"are computed at the same confidence for comparison. The backtest found {n_breaches} breaches "
            f"against an expected {expected_breaches:.1f} across {n} observations."
        )

        results = {
            "status": "ok",
            "n_returns": int(n), "value_col": value_col, "freq": freq, "conf": _fin(conf, 4),
            "mean": _fin(m, 6), "stdev": _fin(s, 6), "pos": _fin(pos, 2),
            "byConf": by_conf, "chosen": chosen,
            "nBreaches": n_breaches, "expectedBreaches": _fin(expected_breaches, 4),
            "histogram": histogram, "tailCounts": tail_counts, "tailCount": tail_count,
            "actualRate": _fin(actual_rate, 6), "expectedRate": _fin(expected_rate, 6),
            # additive extras
            "varSeries": var_series, "backtestTable": backtest_table,
            "interpretation": interpretation,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
