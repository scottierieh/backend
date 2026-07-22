#!/usr/bin/env python3
"""Volatility Analysis — rolling & EWMA volatility, regimes, vol-of-vol.
numpy / pandas / scipy.

Input (from volatility-analysis-page.tsx):
    data        : list[dict]
    value_col   : string   price or return column
    date_col    : string   (optional) column to sort by, "__none__" = row order
    is_returns  : bool
    log_returns : bool     (ignored if is_returns)
    periods_per_year : int (default 252)
    window      : int      rolling window length (default 20)

Output: { results: {...}, plot } — reproduces the fields the client used to
compute in-browser: annVol, rollingVol, ewma, volOfVol, avgRv, maxRv, minRv,
highVolPct, currentVol, currentEwma, skewness, chartData, plus n_obs/n_returns/
value_col/freq/window. Extra fields (regime table, histogram stats) are additive.
"""
import sys, json, io, base64
import numpy as np
import pandas as pd
from scipy import stats as sstats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

EWMA_LAMBDA = 0.94


BLUE = "#2563eb"
GREEN = "#16a34a"
RED = "#dc2626"
AMBER = "#d97706"
PURPLE = "#7c3aed"
LIGHT_BLUE = "#93c5fd"
REGIME_COLORS = {"Low": "#16a34a", "Medium": "#2563eb", "High": "#d97706", "Extreme": "#dc2626"}


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
    values = np.asarray(values, dtype=float)
    if is_returns:
        return values[np.isfinite(values)]
    out = []
    for i in range(1, len(values)):
        p0, p1 = values[i - 1], values[i]
        if not (np.isfinite(p0) and np.isfinite(p1)) or p0 <= 0 or p1 <= 0:
            continue
        out.append(np.log(p1 / p0) if log_ret else (p1 / p0 - 1.0))
    return np.array(out, dtype=float)


def _rolling_vol(returns, window, ppy):
    out = []
    for i in range(window, len(returns) + 1):
        seg = returns[i - window:i]
        out.append(float(np.std(seg, ddof=1)) * np.sqrt(ppy))
    return np.array(out, dtype=float)


def _ewma_vol(returns, lam, ppy):
    if len(returns) == 0:
        return np.array([])
    out = np.zeros(len(returns))
    v2 = returns[0] ** 2
    out[0] = np.sqrt(v2) * np.sqrt(ppy)
    for i in range(1, len(returns)):
        v2 = lam * v2 + (1 - lam) * returns[i] ** 2
        out[i] = np.sqrt(v2) * np.sqrt(ppy)
    return out


def main():
    try:
        p = json.load(sys.stdin)
        rows = p.get("data") or []
        if not rows:
            raise ValueError("No data provided.")
        df = pd.DataFrame(rows)

        value_col = p.get("value_col")
        if not value_col or value_col not in df.columns:
            raise ValueError("Select a valid price/return column.")
        date_col = p.get("date_col") or "__none__"
        is_returns = bool(p.get("is_returns", False))
        log_ret = bool(p.get("log_returns", False))
        ppy = int(p.get("periods_per_year") or 252)
        window = max(5, int(p.get("window") or 20))

        if date_col != "__none__" and date_col in df.columns:
            df = df.sort_values(by=date_col, key=lambda s: s.astype(str)).reset_index(drop=True)

        values = pd.to_numeric(df[value_col], errors="coerce")
        values = values[np.isfinite(values)].values

        returns = _to_returns(values, is_returns, log_ret)
        if len(returns) < 5:
            raise ValueError("Not enough valid observations to compute volatility.")

        win_n = min(window, len(returns))
        ann_vol = float(np.std(returns, ddof=1)) * np.sqrt(ppy) if len(returns) > 1 else 0.0
        rv = _rolling_vol(returns, win_n, ppy)
        ewma = _ewma_vol(returns, EWMA_LAMBDA, ppy)

        vol_of_vol = float(np.std(rv, ddof=1)) if len(rv) > 1 else 0.0
        avg_rv = float(np.mean(rv)) if len(rv) else 0.0
        max_rv = float(np.max(rv)) if len(rv) else 0.0
        min_rv = float(np.min(rv)) if len(rv) else 0.0
        high_vol_bars = int(np.sum(rv > avg_rv * 1.5)) if len(rv) else 0
        high_vol_pct = (high_vol_bars / len(rv)) if len(rv) else 0.0
        current_vol = float(rv[-1]) if len(rv) else ann_vol
        current_ewma = float(ewma[-1]) if len(ewma) else ann_vol
        skewness = float(sstats.skew(returns, bias=False)) if len(returns) > 2 else 0.0

        chart_data = []
        for i, v in enumerate(rv):
            ewma_idx = i + win_n - 1
            chart_data.append({
                "idx": i + win_n,
                "vol": _fin(v, 6),
                "ewma": _fin(ewma[ewma_idx], 6) if 0 <= ewma_idx < len(ewma) else None,
            })

        # additive extras: regime breakdown table
        regimes = []
        if len(rv):
            low_th, high_th = avg_rv * 0.75, avg_rv * 1.5
            calm = rv[rv <= low_th]
            normal = rv[(rv > low_th) & (rv <= high_th)]
            stressed = rv[rv > high_th]
            for name, seg in [("Calm", calm), ("Normal", normal), ("Stressed", stressed)]:
                regimes.append({
                    "regime": name, "n": int(len(seg)),
                    "pct": _fin(len(seg) / len(rv), 4) if len(rv) else None,
                    "avg_vol": _fin(float(np.mean(seg)), 5) if len(seg) else None,
                })

        # ═══════════════════════ Step-6 full report (additive) ═══════════════════════

        # ── ① Volatility Summary ──────────────────────────────────────────────
        period_vol = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
        pos_ret = returns[returns > 0]
        neg_ret = returns[returns < 0]
        upside_vol = float(np.std(pos_ret, ddof=1)) * np.sqrt(ppy) if len(pos_ret) > 1 else 0.0
        downside_vol = float(np.std(neg_ret, ddof=1)) * np.sqrt(ppy) if len(neg_ret) > 1 else 0.0
        vol_summary = {
            "volatility": _fin(period_vol, 6),
            "annualized_volatility": _fin(ann_vol, 6),
            "upside_volatility": _fin(upside_vol, 6),
            "downside_volatility": _fin(downside_vol, 6),
            "min_volatility": _fin(min_rv, 6),
            "max_volatility": _fin(max_rv, 6),
            "average_volatility": _fin(avg_rv, 6),
            "volatility_of_volatility": _fin(vol_of_vol, 6),
        }

        # ── ② Rolling Volatility — 4 windows overlaid ─────────────────────────
        candidate_windows = [20, 60, 120, 252]
        n_avail = len(returns)
        active_windows = [w for w in candidate_windows if w <= n_avail // 2] or [win_n]
        rolling_windows = {}
        for w in active_windows:
            rv_w = _rolling_vol(returns, w, ppy)
            rolling_windows[str(w)] = [
                {"idx": int(i + w), "vol": _fin(v, 6)} for i, v in enumerate(rv_w)
            ]

        chart_rolling_vol = None
        try:
            fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=115)
            palette = [BLUE, GREEN, AMBER, RED]
            for i, w in enumerate(active_windows):
                series = rolling_windows[str(w)]
                if not series:
                    continue
                xs = [pt["idx"] for pt in series]
                ys = [pt["vol"] * 100 if pt["vol"] is not None else None for pt in series]
                ax.plot(xs, ys, color=palette[i % len(palette)], lw=1.4, label=f"{w}-period")
            ax.set_title("Rolling Volatility — Multiple Windows")
            ax.set_xlabel("Period"); ax.set_ylabel("Annualized volatility (%)")
            ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.2)
            fig.tight_layout()
            chart_rolling_vol = _png(fig)
        except Exception:
            plt.close("all"); chart_rolling_vol = None

        # ── ③ Volatility Dynamics — return series + rolling vol (win_n) ──────
        chart_dynamics = None
        try:
            fig, ax1 = plt.subplots(figsize=(9.5, 4.8), dpi=115)
            ax1.bar(range(len(returns)), returns * 100, color="#9ca3af", width=1.0, alpha=0.6, label="Return")
            ax1.axhline(0, color="#111827", lw=0.6)
            ax1.set_xlabel("Period"); ax1.set_ylabel("Return (%)")
            ax2 = ax1.twinx()
            vol_xs = [c["idx"] for c in chart_data]
            vol_ys = [c["vol"] * 100 if c["vol"] is not None else None for c in chart_data]
            ax2.plot(vol_xs, vol_ys, color=BLUE, lw=1.8, label=f"Rolling vol ({win_n}p)")
            ax2.set_ylabel("Annualized volatility (%)", color=BLUE)
            ax2.tick_params(axis="y", colors=BLUE)
            ax1.set_title("Volatility Dynamics — Return + Rolling Volatility")
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, frameon=False, loc="upper left")
            fig.tight_layout()
            chart_dynamics = _png(fig)
        except Exception:
            plt.close("all"); chart_dynamics = None

        # ── ④ Volatility Regime — fixed bands, fall back to quartiles ────────
        regime_table = []
        regime_series = []
        regime_method = "fixed"
        if len(rv):
            fixed_bands = [
                ("Low", 0.0, 0.10), ("Medium", 0.10, 0.20),
                ("High", 0.20, 0.30), ("Extreme", 0.30, np.inf),
            ]
            counts = [int(np.sum((rv >= lo) & (rv < hi))) for _, lo, hi in fixed_bands]
            if max(counts) >= len(rv) * 0.98:
                # fixed bands put (almost) everything in one bucket -> use quartiles instead
                regime_method = "quartile"
                qs = np.percentile(rv, [25, 50, 75])
                fixed_bands = [
                    ("Low", -np.inf, qs[0]), ("Medium", qs[0], qs[1]),
                    ("High", qs[1], qs[2]), ("Extreme", qs[2], np.inf),
                ]
            for name, lo, hi in fixed_bands:
                mask = (rv >= lo) & (rv < hi) if hi != np.inf else (rv >= lo)
                seg = rv[mask]
                lo_lbl = f"{lo * 100:.0f}%" if np.isfinite(lo) else "—"
                hi_lbl = f"{hi * 100:.0f}%" if np.isfinite(hi) else "+"
                regime_table.append({
                    "regime": name,
                    "range": f"{lo_lbl} – {hi_lbl}",
                    "n": int(len(seg)),
                    "period_share": _fin(len(seg) / len(rv), 4),
                })
            # per-period regime label, aligned to chart_data idx
            band_bounds = [(name, lo, hi) for name, lo, hi in fixed_bands]
            for i, v in enumerate(rv):
                lbl = next((name for name, lo, hi in band_bounds if (v >= lo) and (v < hi or hi == np.inf)), "Low")
                regime_series.append({"idx": int(i + win_n), "regime": lbl, "vol": _fin(float(v), 6)})

        chart_regime_timeline = None
        if regime_series:
            try:
                fig, ax = plt.subplots(figsize=(9.5, 3.2), dpi=115)
                order = ["Low", "Medium", "High", "Extreme"]
                y_of = {name: i for i, name in enumerate(order)}
                xs = [pt["idx"] for pt in regime_series]
                colors = [REGIME_COLORS.get(pt["regime"], "#9ca3af") for pt in regime_series]
                ax.bar(xs, [1] * len(xs), color=colors, width=1.0)
                ax.set_yticks([]); ax.set_xlabel("Period")
                ax.set_title("Volatility Regime Timeline")
                handles = [plt.Rectangle((0, 0), 1, 1, color=REGIME_COLORS[n]) for n in order]
                ax.legend(handles, order, fontsize=8, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.25))
                fig.tight_layout()
                chart_regime_timeline = _png(fig)
            except Exception:
                plt.close("all"); chart_regime_timeline = None

        # ── ⑤ Upside vs Downside Volatility ───────────────────────────────────
        downside_total_ratio = (downside_vol / ann_vol) if ann_vol else None
        updown_table = {
            "total_volatility": _fin(ann_vol, 6),
            "upside_volatility": _fin(upside_vol, 6),
            "downside_volatility": _fin(downside_vol, 6),
            "downside_total_ratio": _fin(downside_total_ratio, 4),
        }
        chart_upside_downside = None
        try:
            fig, ax = plt.subplots(figsize=(5.5, 4.6), dpi=115)
            ax.bar(["Upside", "Downside"], [upside_vol * 100, downside_vol * 100], color=[GREEN, RED], width=0.55)
            ax.set_title("Upside vs Downside Volatility")
            ax.set_ylabel("Annualized volatility (%)")
            ax.grid(alpha=0.2, axis="y")
            fig.tight_layout()
            chart_upside_downside = _png(fig)
        except Exception:
            plt.close("all"); chart_upside_downside = None

        # ── ⑥ Volatility Comparison — optional, needs compare_cols ───────────
        compare_cols = [c for c in (p.get("compare_cols") or []) if c in df.columns and c != value_col]
        comparison_table = None
        chart_comparison = None
        if compare_cols:
            comparison_table = []
            comp_series = {value_col: ann_vol}
            for c in compare_cols:
                try:
                    c_vals = pd.to_numeric(df[c], errors="coerce")
                    c_vals = c_vals[np.isfinite(c_vals)].values
                    c_ret = _to_returns(c_vals, is_returns, log_ret)
                    if len(c_ret) < 5:
                        continue
                    c_vol = float(np.std(c_ret, ddof=1)) * np.sqrt(ppy) if len(c_ret) > 1 else 0.0
                    comp_series[c] = c_vol
                except Exception:
                    continue
            for name, vol_v in comp_series.items():
                comparison_table.append({"asset": name, "annualized_volatility": _fin(vol_v, 6)})
            if len(comparison_table) > 1:
                try:
                    fig, ax = plt.subplots(figsize=(max(5.5, 1.2 * len(comparison_table)), 4.6), dpi=115)
                    names = [row["asset"] for row in comparison_table]
                    vals = [row["annualized_volatility"] * 100 if row["annualized_volatility"] is not None else 0 for row in comparison_table]
                    ax.bar(names, vals, color=PURPLE, width=0.55)
                    ax.set_title("Volatility Comparison")
                    ax.set_ylabel("Annualized volatility (%)")
                    ax.tick_params(axis="x", rotation=30)
                    ax.grid(alpha=0.2, axis="y")
                    fig.tight_layout()
                    chart_comparison = _png(fig)
                except Exception:
                    plt.close("all"); chart_comparison = None
            else:
                comparison_table = None

        charts = {
            "rolling_vol": chart_rolling_vol,
            "dynamics": chart_dynamics,
            "regime_timeline": chart_regime_timeline,
            "upside_downside": chart_upside_downside,
            "comparison": chart_comparison,
        }

        # plot: rolling + EWMA overlay, clustering, return-vs-vol scatter, histogram
        plot = None
        try:
            fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=110)
            ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

            xs = [c["idx"] for c in chart_data]
            vol_vals = [c["vol"] for c in chart_data]
            ewma_vals = [c["ewma"] for c in chart_data]
            ax1.fill_between(xs, vol_vals, 0, color="#2563eb", alpha=0.15)
            ax1.plot(xs, vol_vals, color="#2563eb", lw=1.3, label="Rolling vol")
            ax1.plot(xs, ewma_vals, color="#f59e0b", lw=1.6, label="EWMA")
            ax1.set_title("Rolling & EWMA volatility"); ax1.set_xlabel("Period"); ax1.set_ylabel("Ann. vol")
            ax1.legend(fontsize=8, frameon=False)

            ax2.plot(range(len(returns)), returns * 100, color="#6366f1", lw=0.8)
            ax2.axhline(0, color="#9ca3af", lw=0.7)
            ax2.set_title("Volatility clustering (returns)"); ax2.set_xlabel("Period"); ax2.set_ylabel("Return (%)")

            if len(rv) == len(returns[win_n - 1:]):
                ax3.scatter(returns[win_n - 1:] * 100, rv * 100, s=10, color="#10b981", alpha=0.6)
            ax3.set_title("Return vs. rolling volatility"); ax3.set_xlabel("Return (%)"); ax3.set_ylabel("Rolling vol (%)")

            if len(rv):
                ax4.hist(rv * 100, bins=min(30, max(8, len(rv) // 4)), color="#93c5fd", edgecolor="white")
            ax4.set_title("Volatility distribution"); ax4.set_xlabel("Rolling vol (%)"); ax4.set_ylabel("Frequency")

            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        results = {
            "status": "ok",
            "n_obs": int(len(values)), "n_returns": int(len(returns)),
            "value_col": value_col, "freq": str(p.get("freq") or ppy), "window": int(win_n),
            "annVol": _fin(ann_vol, 6),
            "rollingVol": [_fin(v, 6) for v in rv],
            "ewma": [_fin(v, 6) for v in ewma],
            "volOfVol": _fin(vol_of_vol, 6),
            "avgRv": _fin(avg_rv, 6),
            "maxRv": _fin(max_rv, 6),
            "minRv": _fin(min_rv, 6),
            "highVolPct": _fin(high_vol_pct, 6),
            "currentVol": _fin(current_vol, 6),
            "currentEwma": _fin(current_ewma, 6),
            "skewness": _fin(skewness, 6),
            "chartData": chart_data,
            "regimes": regimes,
            "vol_summary": vol_summary,
            "rolling_windows": rolling_windows,
            "regime_table": regime_table,
            "regime_series": regime_series,
            "regime_method": regime_method,
            "updown_table": updown_table,
            "comparison_table": comparison_table,
            "charts": charts,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
