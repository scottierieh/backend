#!/usr/bin/env python3
"""Stationarity Analysis — ADF, KPSS, PP tests plus a full step-6 report.
statsmodels / arch (optional).

Tests whether a series is stable (constant mean/variance) over time. Prices
are typically non-stationary (random walks); returns typically are stationary.

Builds an additive 8-section report:
  1. Stationarity Summary   (badge + verdict table)
  2. Stationarity Tests     (ADF/KPSS/PP comparison table)
  3. Rolling Mean & Variance (2-panel chart)
  4. Original vs Differenced Series (chart + table)
  5. Differencing Analysis  (d=0,1,2 ladder + recommended_d)
  6. Trend Stationarity     (linear detrend, chart + table)
  7. Structural Break Detection (Zivot-Andrews, if available)
  8. Stationarity by Transformation (level/log/diff/log-diff table)

Input (from fin-stationarity-page.tsx):
    data        : list[dict]
    asset_col   : str
    series_type : "level"|"returns"|"log_returns"   what to test (default level)
Output: { results: {..., charts: {...}}, plot }
"""
import sys, json, io, base64, warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, kpss

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#2563eb"
GREEN = "#16a34a"
RED = "#dc2626"
AMBER = "#d97706"
GREY = "#94a3b8"


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


def run_adf_kpss(values, pp=True):
    """Run ADF (+ optional KPSS/PP) on a 1-D array. Returns a dict with
    statistic/p_value/stationary for each test that succeeded, plus a
    combined `stationary` flag (ADF & KPSS agreement rule used elsewhere
    in this file) — or None fields when a test could not be computed
    (e.g. too few observations)."""
    out = {"adf": None, "kpss": None, "pp": None, "pp_note": None}
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if len(v) < 8:
        out["_note"] = "Too few observations to run stationarity tests."
        return out
    try:
        adf_stat, adf_p, adf_lags, adf_nobs, adf_crit, _ = adfuller(v, autolag="AIC")
        adf_stationary = bool(adf_p < 0.05)
        out["adf"] = {"statistic": _fin(adf_stat, 4), "p_value": _fin(adf_p, 6), "lags": int(adf_lags),
                      "critical": {k: _fin(c, 4) for k, c in adf_crit.items()}, "stationary": adf_stationary}
    except Exception as e:
        out["adf"] = None
        out["_adf_error"] = str(e)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            kpss_stat, kpss_p, kpss_lags, kpss_crit = kpss(v, regression="c", nlags="auto")
        kpss_stationary = bool(kpss_p >= 0.05)
        out["kpss"] = {"statistic": _fin(kpss_stat, 4), "p_value": _fin(kpss_p, 6), "lags": int(kpss_lags),
                       "critical": {k: _fin(c, 4) for k, c in kpss_crit.items()}, "stationary": kpss_stationary}
    except Exception as e:
        out["kpss"] = None
        out["_kpss_error"] = str(e)
    if pp:
        try:
            from arch.unitroot import PhillipsPerron
            pp_res = PhillipsPerron(v)
            pp_stat = float(pp_res.stat); pp_p = float(pp_res.pvalue)
            out["pp"] = {"statistic": _fin(pp_stat, 4), "p_value": _fin(pp_p, 6), "stationary": bool(pp_p < 0.05)}
        except ImportError:
            out["pp_note"] = "Phillips-Perron test skipped: the optional 'arch' package is not installed."
        except Exception as e:
            out["pp_note"] = f"Phillips-Perron test failed: {e}"
    return out


def _verdict(adf_block, kpss_block):
    """ADF+KPSS agreement rule -> (verdict, confidence, is_stationary)."""
    if not adf_block or not kpss_block:
        return "unknown", "insufficient data", False
    a = adf_block["stationary"]; k = kpss_block["stationary"]
    if a and k:
        return "stationary", "both tests agree", True
    if (not a) and (not k):
        return "non-stationary", "both tests agree", False
    return "ambiguous — tests disagree", "tests disagree", False


def _test_row(name, block):
    if not block:
        return None
    return {"test": name, "statistic": block.get("statistic"), "p_value": block.get("p_value"),
            "stationary": block.get("stationary")}


def main():
    try:
        p = json.load(sys.stdin)
        rows = p.get("data") or []
        if not rows:
            raise ValueError("No data provided.")
        df = pd.DataFrame(rows)
        col = p.get("asset_col")
        stype = (p.get("series_type") or "level").lower()
        is_returns = stype in ("returns", "log_returns")
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
        xv = x.values

        # ─────────────────────── core ADF/KPSS/PP on the tested series ───────────────────────
        core = run_adf_kpss(xv, pp=True)
        adf_b, kpss_b, pp_b, pp_note = core["adf"], core["kpss"], core["pp"], core["pp_note"]
        verdict, confidence, is_stationary = _verdict(adf_b, kpss_b)

        differencing_required = not is_stationary

        if verdict == "ambiguous — tests disagree":
            final_note = ("ADF's null hypothesis is a unit root (non-stationary), while KPSS's null is "
                           "stationarity — so when they disagree it is itself informative: the series is "
                           "often trend-stationary or borderline rather than cleanly one or the other.")
        else:
            final_note = None

        interpretation = (
            f"Testing the {label} of {col} ({n} observations): the ADF test {'rejects' if (adf_b and adf_b['stationary']) else 'does not reject'} "
            f"a unit root, and the KPSS test {'does not reject' if (kpss_b and kpss_b['stationary']) else 'rejects'} "
            f"stationarity. Together they indicate the series is {verdict} ({confidence})."
        )

        # ═══════════════════════════ ① Stationarity Summary ═══════════════════════════
        stationarity_summary = {
            "adf_conclusion": (t_("Stationary") if adf_b and adf_b["stationary"] else t_("Non-Stationary")) if adf_b else None,
            "adf_p_value": adf_b["p_value"] if adf_b else None,
            "kpss_conclusion": (t_("Stationary") if kpss_b and kpss_b["stationary"] else t_("Non-Stationary")) if kpss_b else None,
            "kpss_p_value": kpss_b["p_value"] if kpss_b else None,
            "pp_conclusion": (t_("Stationary") if pp_b and pp_b["stationary"] else t_("Non-Stationary")) if pp_b else None,
            "pp_p_value": pp_b["p_value"] if pp_b else None,
            "differencing_required": bool(differencing_required),
            "final_conclusion": verdict,
            "final_conclusion_note": final_note,
        }

        # ═══════════════════════════ ② Stationarity Tests ═══════════════════════════
        test_comparison = [r for r in [_test_row("ADF", adf_b), _test_row("KPSS", kpss_b), _test_row("PP", pp_b)] if r]

        # ═══════════════════════════ ③ Rolling Mean & Variance ═══════════════════════════
        w = max(5, min(21, n // 4))
        roll_mean = x.rolling(w).mean()
        roll_var = x.rolling(w).var()

        chart_rolling = None
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), dpi=115, sharex=True)
            ax1.plot(xv, color=GREY, lw=0.7, label=label)
            ax1.plot(roll_mean.values, color=BLUE, lw=1.5, label=f"Rolling mean ({w})")
            ax1.set_title(f"{col} ({label}) with rolling mean"); ax1.legend(fontsize=8, frameon=False); ax1.grid(alpha=0.2)
            ax2.plot(roll_var.values, color=AMBER, lw=1.4, label=f"Rolling variance ({w})")
            ax2.set_title("Rolling variance"); ax2.set_xlabel("Period"); ax2.legend(fontsize=8, frameon=False); ax2.grid(alpha=0.2)
            fig.tight_layout()
            chart_rolling = _png(fig)
        except Exception:
            plt.close("all"); chart_rolling = None

        # legacy single `plot` key kept for backward-compat with existing frontend render
        plot = chart_rolling

        # ═══════════════════════════ ④/⑤ Differencing ladder (d=0,1,2) ═══════════════════════════
        diff_series = {"Level (d=0)": x}
        d1 = x.diff().dropna().reset_index(drop=True)
        diff_series["1st Difference (d=1)"] = d1
        d2 = x.diff().diff().dropna().reset_index(drop=True)
        diff_series["2nd Difference (d=2)"] = d2

        differencing_table = []
        recommended_d = None
        for i, (dname, dseries) in enumerate(diff_series.items()):
            block = run_adf_kpss(dseries.values, pp=False)
            a, k = block["adf"], block["kpss"]
            v, conf, stat = _verdict(a, k)
            differencing_table.append({
                "series": dname, "d": i,
                "adf_p_value": a["p_value"] if a else None,
                "kpss_p_value": k["p_value"] if k else None,
                "result": t_("Stationary") if stat else t_("Non-Stationary"),
                "stationary": bool(stat),
            })
            if recommended_d is None and stat:
                recommended_d = i
        recommended_d_note = None
        if recommended_d is None:
            recommended_d = 2
            recommended_d_note = ("Neither the level, 1st, nor 2nd difference achieved clean ADF/KPSS agreement "
                                   "on stationarity; d=2 is reported as a fallback upper bound — inspect the "
                                   "series further before modelling.")

        # ④ uses just the first two rows (Level, 1st Difference)
        original_vs_differenced_table = differencing_table[:2]

        chart_orig_diff = None
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), dpi=115)
            ax1.plot(xv, color=BLUE, lw=1.1)
            ax1.set_title(f"Original Series ({label})"); ax1.grid(alpha=0.2)
            ax2.plot(d1.values, color=GREEN, lw=1.0)
            ax2.axhline(0, color="#111827", lw=0.7, ls="--")
            ax2.set_title("First Difference"); ax2.set_xlabel("Period"); ax2.grid(alpha=0.2)
            fig.tight_layout()
            chart_orig_diff = _png(fig)
        except Exception:
            plt.close("all"); chart_orig_diff = None

        # ═══════════════════════════ ⑥ Trend Stationarity ═══════════════════════════
        t_idx = np.arange(n, dtype=float)
        slope, intercept = np.polyfit(t_idx, xv, 1)
        trend_line = intercept + slope * t_idx
        detrended = xv - trend_line

        # t-test on the OLS slope
        resid = xv - trend_line
        dof = n - 2
        trend_significant = False
        slope_p = None
        if dof > 0:
            x_mean = t_idx.mean()
            ssx = np.sum((t_idx - x_mean) ** 2)
            mse = np.sum(resid ** 2) / dof
            se_slope = np.sqrt(mse / ssx) if ssx > 0 else None
            if se_slope and se_slope > 0:
                from scipy import stats as sstats
                t_stat = slope / se_slope
                slope_p = float(2 * (1 - sstats.t.cdf(abs(t_stat), dof)))
                trend_significant = bool(slope_p < 0.05)

        detrended_block = run_adf_kpss(detrended, pp=False)
        detrended_adf = detrended_block["adf"]
        trend_stationary = bool(detrended_adf and detrended_adf["stationary"])

        trend_table = {
            "trend_slope": _fin(slope, 6),
            "trend_p_value": _fin(slope_p, 6) if slope_p is not None else None,
            "trend_significant": trend_significant,
            "trend": "Significant" if trend_significant else "Not Significant",
            "detrended_adf_statistic": detrended_adf["statistic"] if detrended_adf else None,
            "detrended_adf_p_value": detrended_adf["p_value"] if detrended_adf else None,
            "trend_stationary": trend_stationary,
        }

        chart_trend = None
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), dpi=115)
            ax1.plot(xv, color=GREY, lw=0.8, label=label)
            ax1.plot(trend_line, color=RED, lw=1.6, ls="--", label="Fitted trend")
            ax1.set_title("Original Series with Fitted Trend"); ax1.legend(fontsize=8, frameon=False); ax1.grid(alpha=0.2)
            ax2.plot(detrended, color=GREEN, lw=1.0)
            ax2.axhline(0, color="#111827", lw=0.7, ls="--")
            ax2.set_title("Detrended Series (residual)"); ax2.set_xlabel("Period"); ax2.grid(alpha=0.2)
            fig.tight_layout()
            chart_trend = _png(fig)
        except Exception:
            plt.close("all"); chart_trend = None

        # ═══════════════════════════ ⑦ Structural Break Detection (Zivot-Andrews) ═══════════════════════════
        structural_break = None
        structural_break_note = None
        chart_break = None
        try:
            from arch.unitroot import ZivotAndrews
            za_res = ZivotAndrews(xv)
            za_stat = float(za_res.stat)
            za_p = float(za_res.pvalue)
            # Newer `arch` releases expose a `.break_date`/`.break_idx` property directly;
            # older ones only expose the per-candidate statistics array `_all_stats`, whose
            # minimizer (most negative stat, matching `.stat`) is the detected break index.
            break_idx = getattr(za_res, "break_date", None) or getattr(za_res, "break_idx", None)
            if break_idx is None:
                try:
                    all_stats = np.asarray(za_res._all_stats, dtype=float)
                    break_idx = int(np.nanargmin(all_stats))
                except Exception:
                    break_idx = None
            try:
                break_idx = int(break_idx)
            except (TypeError, ValueError):
                break_idx = None
            significant = bool(za_p < 0.05)
            structural_break = {
                "break_index": break_idx,
                "statistic": _fin(za_stat, 4),
                "p_value": _fin(za_p, 6),
                "significant": significant,
            }
            try:
                fig, ax = plt.subplots(figsize=(11, 4.6), dpi=115)
                ax.plot(xv, color=BLUE, lw=1.1)
                if break_idx is not None and 0 <= break_idx < n:
                    ax.axvline(break_idx, color=RED, lw=1.6, ls="--", label=f"Break at index {break_idx}")
                    ax.legend(fontsize=8, frameon=False)
                ax.set_title("Structural Break (Zivot-Andrews)"); ax.set_xlabel("Period"); ax.grid(alpha=0.2)
                fig.tight_layout()
                chart_break = _png(fig)
            except Exception:
                plt.close("all"); chart_break = None
        except ImportError:
            structural_break_note = "Zivot-Andrews test unavailable in this environment (optional 'arch' package not installed)."
        except Exception as e:
            structural_break_note = f"Zivot-Andrews test unavailable in this environment: {e}"

        # ═══════════════════════════ ⑧ Stationarity by Transformation ═══════════════════════════
        transformation_table = []
        transformation_note = None
        if is_returns:
            transformation_note = ("Input is already a return series; level-vs-return transformation comparison "
                                    "requires raw price data.")
            v_stat, k_stat = adf_b, kpss_b
            transformation_table.append({
                "transformation": label.title(), "stationary": bool(is_stationary),
                "stationarity": t_("Stationary") if is_stationary else t_("Non-Stationary"),
                "recommended": True,
            })
        else:
            level = s.reset_index(drop=True)
            variants = {}
            variants["Original"] = level
            if (level > 0).all():
                variants["Log"] = np.log(level)
            variants["First Difference"] = level.diff().dropna().reset_index(drop=True)
            if (level > 0).all():
                variants["Log Difference"] = np.log(level).diff().dropna().reset_index(drop=True)

            stationary_flags = {}
            for name, series in variants.items():
                block = run_adf_kpss(pd.Series(series).values, pp=False)
                _, _, stat = _verdict(block["adf"], block["kpss"])
                stationary_flags[name] = stat

            # financial convention: prefer Log Difference, then First Difference, over raw Original/Log
            preference_order = ["Log Difference", "First Difference", "Log", "Original"]
            recommended_name = next((nm for nm in preference_order if stationary_flags.get(nm)), None)

            for name in variants:
                transformation_table.append({
                    "transformation": name,
                    "stationary": bool(stationary_flags.get(name, False)),
                    "stationarity": t_("Stationary") if stationary_flags.get(name) else t_("Non-Stationary"),
                    "recommended": bool(name == recommended_name),
                })

        charts = {
            "rolling_mean_var": chart_rolling,
            "original_vs_differenced": chart_orig_diff,
            "trend_detrend": chart_trend,
        }
        if chart_break is not None:
            charts["structural_break"] = chart_break

        results = {
            "status": "ok", "asset": col, "series_type": stype, "series_label": label, "n_obs": n,
            "roll_window": w,
            "adf": adf_b, "kpss": kpss_b, "pp": pp_b, "pp_note": pp_note,
            "verdict": verdict, "confidence": confidence, "is_stationary": bool(is_stationary),
            "interpretation": interpretation,
            "test_comparison": test_comparison,
            # ① Stationarity Summary
            "stationarity_summary": stationarity_summary,
            # ④/⑤ tables
            "original_vs_differenced_table": original_vs_differenced_table,
            "differencing_table": differencing_table,
            "recommended_d": recommended_d,
            "recommended_d_note": recommended_d_note,
            # ⑥ trend stationarity
            "trend_table": trend_table,
            # ⑦ structural break
            "structural_break": structural_break,
            "structural_break_note": structural_break_note,
            # ⑧ transformation table
            "transformation_table": transformation_table,
            "transformation_note": transformation_note,
            "charts": charts,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


def t_(x):
    """Placeholder passthrough — English label; frontend handles i18n."""
    return x


if __name__ == "__main__":
    main()
