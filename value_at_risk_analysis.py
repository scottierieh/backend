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


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


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

        # ═══════════════════════ Step-6 full report (additive) ═══════════════════════
        investment_amount = pos if pos > 0 else None

        # ── ① VaR Summary ─────────────────────────────────────────────────────
        var_summary = {
            "var_pct": chosen["historical"]["varV"],
            "var_amount": _fin(abs(chosen["historical"]["varV"]) * investment_amount, 2) if investment_amount else None,
            "investment_amount": _fin(investment_amount, 2) if investment_amount else None,
            "confidence": _fin(conf, 4),
            "holding_period": 1,
            "sample_size": int(n),
            "method": "Historical",
            "cvar_pct": chosen["historical"]["cvarV"],
        }

        # ── ② VaR Method Comparison ───────────────────────────────────────────
        c95, c99 = by_conf.get("0.95"), by_conf.get("0.99")
        method_comparison = []
        if c95 and c99:
            for label, key in [("Historical", "historical"), ("Parametric", "parametric"), ("Monte Carlo", "monteCarlo")]:
                method_comparison.append({
                    "method": label,
                    "var95": c95[key]["varV"], "var99": c99[key]["varV"],
                    "cvar95": c95[key]["cvarV"],
                })

        # ── ③ Confidence Level Analysis (table + chart) ───────────────────────
        conf_levels = [0.90, 0.95, 0.975, 0.99]
        confidence_table = []
        for c in conf_levels:
            v1, cv1 = _historical_var(returns, c)
            confidence_table.append({
                "confidence": _fin(c, 4),
                "var_1d": _fin(v1, 6),
                "var_10d": _fin(v1 * np.sqrt(10), 6),
                "cvar_1d": _fin(cv1, 6),
            })
        confidence_table_note = (
            "10-day VaR is scaled from 1-day VaR via VaR_T = VaR_1 * sqrt(T), the standard square-root-of-time "
            "rule. This assumes i.i.d. (independent, identically distributed) returns and is only an "
            "approximation for real financial data, which typically exhibits fat tails and autocorrelation."
        )

        chart_confidence_level = None
        try:
            fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=115)
            labels = [f"{row['confidence']*100:.1f}%" for row in confidence_table]
            vals = [abs(row["var_1d"]) * 100 if row["var_1d"] is not None else 0 for row in confidence_table]
            ax.bar(labels, vals, color="#2563eb", width=0.55)
            ax.plot(labels, vals, color="#dc2626", marker="o", lw=1.4)
            ax.set_title("VaR by Confidence Level")
            ax.set_xlabel("Confidence level"); ax.set_ylabel("VaR magnitude (%)")
            ax.grid(alpha=0.2, axis="y")
            fig.tight_layout()
            chart_confidence_level = _png(fig)
        except Exception:
            plt.close("all"); chart_confidence_level = None

        # ── ④ Holding Period Analysis (table + chart) ─────────────────────────
        var_1 = chosen["historical"]["varV"]
        horizons = [1, 5, 10, 20]
        horizon_table = [
            {"holding_period": hday, "var": _fin(var_1 * np.sqrt(hday), 6)}
            for hday in horizons
        ]
        horizon_table_note = (
            "Scaled from the 1-day VaR via the square-root-of-time rule (VaR_T = VaR_1 * sqrt(T)). This "
            "assumes i.i.d. returns; it can understate risk when real returns show fat tails or autocorrelation."
        )

        chart_holding_period = None
        try:
            fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=115)
            labels = [f"{row['holding_period']}D" for row in horizon_table]
            vals = [abs(row["var"]) * 100 if row["var"] is not None else 0 for row in horizon_table]
            ax.bar(labels, vals, color="#16a34a", width=0.55)
            ax.plot(labels, vals, color="#dc2626", marker="o", lw=1.4)
            ax.set_title("VaR by Holding Period")
            ax.set_xlabel("Holding period"); ax.set_ylabel("VaR magnitude (%)")
            ax.grid(alpha=0.2, axis="y")
            fig.tight_layout()
            chart_holding_period = _png(fig)
        except Exception:
            plt.close("all"); chart_holding_period = None

        # ── ⑤ VaR Distribution (chart + Tail Metrics table) ───────────────────
        v95, cv95 = (c95["historical"]["varV"], c95["historical"]["cvarV"]) if c95 else (None, None)
        v99, cv99 = (c99["historical"]["varV"], c99["historical"]["cvarV"]) if c99 else (None, None)
        max_loss = float(np.min(returns))
        tail_table = {
            "var95": _fin(v95, 6), "cvar95": _fin(cv95, 6),
            "var99": _fin(v99, 6), "cvar99": _fin(cv99, 6),
            "max_loss": _fin(max_loss, 6),
        }

        chart_distribution = None
        try:
            fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=115)
            ax.hist(returns * 100, bins=bin_count, color="#93c5fd", edgecolor="white")
            if v95 is not None:
                ax.axvline(v95 * 100, color="#dc2626", ls="--", lw=1.5, label="VaR 95%")
            if cv95 is not None:
                ax.axvline(cv95 * 100, color="#7f1d1d", ls=":", lw=1.5, label="CVaR 95%")
            if v99 is not None:
                ax.axvline(v99 * 100, color="#f59e0b", ls="--", lw=1.5, label="VaR 99%")
            if cv99 is not None:
                ax.axvline(cv99 * 100, color="#78350f", ls=":", lw=1.5, label="CVaR 99%")
            ax.set_xlabel("Return (%)"); ax.set_ylabel("Frequency")
            ax.set_title("Return Distribution & VaR/CVaR Thresholds")
            ax.legend(fontsize=8, frameon=False)
            fig.tight_layout()
            chart_distribution = _png(fig)
        except Exception:
            plt.close("all"); chart_distribution = None

        # ── ⑥ VaR Backtesting (chart + table) ─────────────────────────────────
        exceptions = (returns <= var_1).astype(int)
        backtest_summary = {
            "observations": int(n),
            "var_level": _fin(conf, 4),
            "expected_exceptions": _fin(n * (1 - conf), 4),
            "actual_exceptions": int(np.sum(exceptions)),
            "exception_rate": _fin(float(np.mean(exceptions)) if n > 0 else None, 6),
        }

        chart_backtest_timeline = None
        try:
            fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=115)
            idx_arr = np.arange(n)
            colors = np.where(exceptions == 1, "#dc2626", "#2563eb")
            ax.plot(idx_arr, returns * 100, color="#9ca3af", lw=0.8, zorder=1)
            ax.scatter(idx_arr, returns * 100, s=14, c=colors, alpha=0.8, zorder=2)
            ax.axhline(var_1 * 100, color="#dc2626", ls="--", lw=1.3,
                       label=f"VaR {conf:.0%} ({var_1 * 100:.2f}%)")
            ax.set_xlabel("Period"); ax.set_ylabel("Return (%)")
            ax.set_title(f"VaR Backtest Timeline — {int(np.sum(exceptions))} exceptions vs. "
                         f"{n * (1 - conf):.1f} expected")
            ax.legend(fontsize=8, frameon=False)
            fig.tight_layout()
            chart_backtest_timeline = _png(fig)
        except Exception:
            plt.close("all"); chart_backtest_timeline = None

        # ── ⑦ Statistical Backtesting (Kupiec POF, Christoffersen, cond. coverage) ──
        statistical_tests = []
        try:
            x = int(np.sum(exceptions))
            n_obs = int(n)
            p_level = 1 - conf
            if n_obs < 20 or x == 0:
                statistical_tests.append({
                    "test": "Kupiec POF (unconditional coverage)",
                    "statistic": None, "p_value": None, "result": None,
                    "note": "Sample size / exception count too small for a stable test.",
                })
            else:
                pi_hat = x / n_obs
                if x == n_obs:
                    logL_alt = 0.0
                else:
                    logL_alt = (n_obs - x) * np.log(1 - pi_hat) + x * np.log(pi_hat)
                logL_null = (n_obs - x) * np.log(1 - p_level) + x * np.log(p_level)
                lr_pof = float(-2 * (logL_null - logL_alt))
                p_pof = float(stats.chi2.sf(lr_pof, 1))
                statistical_tests.append({
                    "test": "Kupiec POF (unconditional coverage)",
                    "statistic": _fin(lr_pof, 4), "p_value": _fin(p_pof, 6),
                    "result": "Pass" if p_pof >= 0.05 else "Fail",
                })

                # Christoffersen independence test
                seq = exceptions
                n00 = int(np.sum((seq[:-1] == 0) & (seq[1:] == 0)))
                n01 = int(np.sum((seq[:-1] == 0) & (seq[1:] == 1)))
                n10 = int(np.sum((seq[:-1] == 1) & (seq[1:] == 0)))
                n11 = int(np.sum((seq[:-1] == 1) & (seq[1:] == 1)))
                if (n00 + n01) == 0 or (n10 + n11) == 0:
                    statistical_tests.append({
                        "test": "Christoffersen independence",
                        "statistic": None, "p_value": None, "result": None,
                        "note": "Too few exception transitions to estimate the independence test.",
                    })
                    lr_ind = None
                else:
                    pi01 = n01 / (n00 + n01)
                    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
                    pi_all = (n01 + n11) / (n00 + n01 + n10 + n11)

                    def _lt(a, b):
                        return a * np.log(b) if b > 0 else 0.0

                    logL_ind_null = _lt(n00 + n10, 1 - pi_all) + _lt(n01 + n11, pi_all)
                    logL_ind_alt = (_lt(n00, 1 - pi01) + _lt(n01, pi01) +
                                     _lt(n10, 1 - pi11) + _lt(n11, pi11))
                    lr_ind = float(-2 * (logL_ind_null - logL_ind_alt))
                    p_ind = float(stats.chi2.sf(lr_ind, 1))
                    statistical_tests.append({
                        "test": "Christoffersen independence",
                        "statistic": _fin(lr_ind, 4), "p_value": _fin(p_ind, 6),
                        "result": "Pass" if p_ind >= 0.05 else "Fail",
                    })

                if lr_ind is not None:
                    lr_cc = lr_pof + lr_ind
                    p_cc = float(stats.chi2.sf(lr_cc, 2))
                    statistical_tests.append({
                        "test": "Conditional coverage (Kupiec + Christoffersen)",
                        "statistic": _fin(lr_cc, 4), "p_value": _fin(p_cc, 6),
                        "result": "Pass" if p_cc >= 0.05 else "Fail",
                    })
        except Exception:
            statistical_tests = statistical_tests or [{
                "test": "Statistical backtesting",
                "statistic": None, "p_value": None, "result": None,
                "note": "Statistical backtests could not be computed for this sample.",
            }]

        charts = {
            "confidence_level": chart_confidence_level,
            "holding_period": chart_holding_period,
            "distribution": chart_distribution,
            "backtest_timeline": chart_backtest_timeline,
        }

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
            # step-6 full report (additive)
            "var_summary": var_summary,
            "method_comparison": method_comparison,
            "confidence_table": confidence_table,
            "confidence_table_note": confidence_table_note,
            "horizon_table": horizon_table,
            "horizon_table_note": horizon_table_note,
            "tail_table": tail_table,
            "backtest_summary": backtest_summary,
            "statistical_tests": statistical_tests,
            "charts": charts,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
