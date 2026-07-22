#!/usr/bin/env python3
"""Portfolio Risk — volatility, VaR/CVaR, drawdown, risk contributions and a
full 9-section step-6 report (risk summary, risk decomposition, correlation
heatmap, diversification, marginal contribution to risk, optional factor
exposure, concentration, stress testing, rolling risk).
numpy / pandas.

Input (from portfolio-risk-page.tsx):
    data             : list[dict]
    asset_cols       : string[]   price or return columns (>= 1)
    weights          : number[]   (optional) portfolio weights; default equal
    is_returns       : bool
    return_type      : "simple"|"log"
    periods_per_year : int    (default 252)
    confidence       : float  (default 0.95) VaR/CVaR confidence
    benchmark_col    : str    (optional) benchmark price/return column, for beta
    factor_cols      : string[] (optional, backend-only) factor return columns for
                       an optional aggregate factor-exposure regression
    investment_amount: float  (optional) converts stress-test % loss to $ loss
Output: { results: {...}, plot } (return distribution + drawdown + risk contributions,
    plus results.charts: {risk_decomposition, correlation_heatmap, diversification,
    mcr, factor_exposure, concentration, stress_test, rolling_risk}).
"""
import sys, json, io, base64
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#2563eb"
GREEN = "#16a34a"
RED = "#dc2626"
AMBER = "#d97706"
LIGHT_BLUE = "#93c5fd"
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


def _to_returns(s, is_returns, rtype):
    s = pd.to_numeric(s, errors="coerce")
    if is_returns:
        return s
    if rtype == "log":
        if (s <= 0).any():
            raise ValueError("Log returns require positive prices.")
        return np.log(s / s.shift(1))
    return s / s.shift(1) - 1.0


def _returns_df(df, cols, is_returns, rtype):
    out = {}
    for c in cols:
        out[c] = _to_returns(df[c], is_returns, rtype)
    return pd.DataFrame(out).dropna().reset_index(drop=True)


def main():
    try:
        p = json.load(sys.stdin)
        rows = p.get("data") or []
        if not rows:
            raise ValueError("No data provided.")
        df = pd.DataFrame(rows)
        cols = [c for c in (p.get("asset_cols") or []) if c in df.columns]
        if len(cols) < 1:
            raise ValueError("Select at least one asset column.")
        is_returns = bool(p.get("is_returns", False))
        rtype = (p.get("return_type") or "simple").lower()
        ppy = int(p.get("periods_per_year") or 252)
        conf = float(p.get("confidence") or 0.95)
        if not (0.5 < conf < 1):
            raise ValueError("Confidence must be between 0.5 and 1 (e.g. 0.95).")
        benchmark_col = p.get("benchmark_col") or None
        if benchmark_col and benchmark_col not in df.columns:
            benchmark_col = None
        factor_cols = [c for c in (p.get("factor_cols") or []) if c in df.columns]
        investment_amount = p.get("investment_amount")
        try:
            investment_amount = float(investment_amount) if investment_amount not in (None, "") else None
        except (TypeError, ValueError):
            investment_amount = None

        w_in = p.get("weights")
        n = len(cols)
        if w_in and len(w_in) == n:
            w = np.array([float(x) for x in w_in], float)
            if w.sum() == 0:
                raise ValueError("Weights sum to zero.")
            w = w / w.sum()
        else:
            w = np.ones(n) / n

        R = _returns_df(df, cols, is_returns, rtype)
        if len(R) < 10:
            raise ValueError("Need at least 10 aligned return observations.")

        # Align optional benchmark / factor columns to the SAME row mask as R by
        # re-deriving returns on the full frame then inner-joining on index length.
        # Simplest robust approach: recompute a combined returns frame including
        # benchmark/factor columns together, then split — guarantees alignment.
        extra_cols = ([benchmark_col] if benchmark_col else []) + [c for c in factor_cols if c not in cols]
        bench = None
        factors_df = None
        if extra_cols:
            combo_cols = cols + extra_cols
            combo = _returns_df(df, combo_cols, is_returns, rtype)
            R = combo[cols].reset_index(drop=True)
            if benchmark_col:
                bench = combo[benchmark_col].reset_index(drop=True)
            if factor_cols:
                factors_df = combo[[c for c in factor_cols]].reset_index(drop=True)

        port = R.values @ w                              # portfolio return series
        mu_p = float(np.mean(port))
        vol_p = float(np.std(port, ddof=1))
        ann_return = mu_p * ppy
        ann_vol = vol_p * np.sqrt(ppy)
        sharpe = (ann_return / ann_vol) if ann_vol > 0 else None

        # VaR / CVaR (historical + parametric normal), as positive loss fractions
        a = 1 - conf
        var_hist = -np.percentile(port, 100 * a)
        tail = port[port <= -var_hist]
        cvar_hist = -float(np.mean(tail)) if tail.size > 0 else var_hist
        z = float(np.sqrt(2) * _erfinv(2 * a - 1))
        var_norm = -(mu_p + z * vol_p)
        # downside deviation & Sortino
        downside = port[port < 0]
        dd_dev = float(np.sqrt(np.mean(downside ** 2))) if downside.size else 0.0
        sortino = (ann_return / (dd_dev * np.sqrt(ppy))) if dd_dev > 0 else None

        # drawdown from cumulative wealth
        wealth = np.cumprod(1 + port)
        peak = np.maximum.accumulate(wealth)
        drawdown = wealth / peak - 1
        max_dd = float(drawdown.min())
        worst_period = float(np.min(port))

        # risk contributions: MCR = (Sigma w) / sigma_p ; contribution = w * MCR
        Sigma = R.cov().values
        port_var = float(w @ Sigma @ w)
        port_sd = np.sqrt(port_var) if port_var > 0 else 0.0
        if port_sd > 0:
            mcr = (Sigma @ w) / port_sd                 # marginal contribution to risk (per-period units)
            rc = w * mcr                                  # risk contribution in return-units (sums to port_sd)
            rc_pct = rc / rc.sum() if rc.sum() != 0 else rc
        else:
            mcr = np.zeros(n); rc = np.zeros(n); rc_pct = np.zeros(n)

        vols_annual = np.array([float(np.std(R[c], ddof=1) * np.sqrt(ppy)) for c in cols])
        weighted_avg_vol = float(np.sum(w * vols_annual))          # no-diversification benchmark
        diversification_ratio = (weighted_avg_vol / ann_vol) if ann_vol > 0 else None
        effective_n = float(1.0 / np.sum(w ** 2)) if np.sum(w ** 2) > 0 else None
        eff_ratio = (effective_n / n) if (effective_n is not None and n > 0) else None
        # Concentration threshold: effective-N / actual-N ratio (equivalently 1/(n*HHI)).
        # >=0.75 -> Low (close to equal-weight diversification), >=0.40 -> Medium, else High.
        if eff_ratio is None:
            risk_concentration = None
        elif eff_ratio >= 0.75:
            risk_concentration = "Low"
        elif eff_ratio >= 0.40:
            risk_concentration = "Medium"
        else:
            risk_concentration = "High"

        assets = [{"asset": cols[i], "weight": _fin(w[i], 4),
                   "vol_annual": _fin(float(vols_annual[i]), 5),
                   "mcr": _fin(float(mcr[i] * np.sqrt(ppy)), 5),
                   "risk_contribution_value": _fin(float(rc[i] * np.sqrt(ppy)), 5),
                   "risk_contribution": _fin(float(rc_pct[i]), 4)} for i in range(n)]
        assets_sorted = sorted(assets, key=lambda z: -(z["risk_contribution"] or 0))

        corr = R.corr().round(4).values.tolist() if n > 1 else [[1.0]]
        avg_pairwise_corr = None
        if n > 1:
            cm = np.array(corr)
            iu = np.triu_indices(n, k=1)
            avg_pairwise_corr = float(np.mean(cm[iu]))

        # ─────────────────────────── Beta vs benchmark (optional) ───────────────────────────
        beta = None
        asset_betas = None
        if bench is not None and len(bench) == len(R):
            bv = bench.values
            var_b = float(np.var(bv, ddof=1))
            if var_b > 0:
                beta = float(np.cov(port, bv, ddof=1)[0, 1] / var_b)
                asset_betas = np.array([float(np.cov(R[c].values, bv, ddof=1)[0, 1] / var_b) for c in cols])

        # ─────────────────────────── ① Risk Summary ───────────────────────────
        risk_summary = {
            "portfolio_volatility": _fin(ann_vol, 5),
            "downside_deviation": _fin(dd_dev * np.sqrt(ppy), 5),
            "max_drawdown": _fin(max_dd, 5),
            "beta": _fin(beta, 4) if beta is not None else None,
            "var_95": _fin(var_hist, 5),
            "cvar_95": _fin(cvar_hist, 5),
            "diversification_ratio": _fin(diversification_ratio, 4),
            "effective_n_assets": _fin(effective_n, 3),
            "risk_concentration": risk_concentration,
        }
        risk_summary_note = None if bench is not None else (
            "No benchmark column was supplied, so beta vs. benchmark is omitted."
        )

        # ─────────────────────────── ② Risk Decomposition (chart) ───────────────────────────
        chart_risk_decomposition = None
        try:
            order = np.argsort(rc_pct)[::-1]
            labels = [cols[i] for i in order]
            xs = np.arange(len(labels))
            fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=115)
            width = 0.38
            ax.bar(xs - width / 2, [w[i] * 100 for i in order], width, color=LIGHT_BLUE, label="Weight (%)")
            ax.bar(xs + width / 2, [rc_pct[i] * 100 for i in order], width, color=BLUE, label="Risk contribution (%)")
            ax.set_xticks(xs); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel("Percent"); ax.set_title("Risk Contribution vs. Weight")
            ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.2, axis="y")
            fig.tight_layout()
            chart_risk_decomposition = _png(fig)
        except Exception:
            plt.close("all"); chart_risk_decomposition = None

        # ─────────────────────────── ③ Correlation heatmap ───────────────────────────
        chart_correlation_heatmap = None
        if n > 1:
            try:
                cm = np.array(corr)
                fig, ax = plt.subplots(figsize=(max(5.5, 0.9 * n + 2), max(4.6, 0.9 * n + 1.5)), dpi=115)
                im = ax.imshow(cm, cmap="RdBu_r", vmin=-1, vmax=1)
                ax.set_xticks(range(n)); ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
                ax.set_yticks(range(n)); ax.set_yticklabels(cols, fontsize=8)
                for i in range(n):
                    for j in range(n):
                        val = cm[i, j]
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                                color="white" if abs(val) > 0.55 else "black", fontsize=8)
                ax.set_title("Correlation Heatmap")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                fig.tight_layout()
                chart_correlation_heatmap = _png(fig)
            except Exception:
                plt.close("all"); chart_correlation_heatmap = None

        # ─────────────────────────── ④ Diversification Analysis ───────────────────────────
        diversification_table = {
            "weighted_average_volatility": _fin(weighted_avg_vol, 5),
            "portfolio_volatility": _fin(ann_vol, 5),
            "diversification_ratio": _fin(diversification_ratio, 4),
            "effective_n_assets": _fin(effective_n, 3),
        }
        chart_diversification = None
        try:
            fig, ax = plt.subplots(figsize=(5.5, 4.6), dpi=115)
            bars = ax.bar(["Weighted-avg\nvolatility", "Portfolio\nvolatility"],
                           [weighted_avg_vol * 100, ann_vol * 100], color=[AMBER, GREEN], width=0.55)
            ax.set_ylabel("Annualized volatility (%)")
            ax.set_title("Diversification Benefit")
            ax.grid(alpha=0.2, axis="y")
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{b.get_height():.1f}%",
                        ha="center", va="bottom", fontsize=9)
            fig.tight_layout()
            chart_diversification = _png(fig)
        except Exception:
            plt.close("all"); chart_diversification = None

        # ─────────────────────────── ⑤ Marginal Contribution to Risk ───────────────────────────
        mcr_table = [{"asset": cols[i], "weight": _fin(w[i], 4),
                      "mcr": _fin(float(mcr[i] * np.sqrt(ppy)), 5),
                      "risk_contribution_pct": _fin(float(rc_pct[i]), 4)} for i in range(n)]
        mcr_table_sorted = sorted(mcr_table, key=lambda z: -(z["mcr"] or 0))
        chart_mcr = None
        try:
            order = np.argsort(mcr)[::-1]
            fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=115)
            ax.bar([cols[i] for i in order], [mcr[i] * np.sqrt(ppy) * 100 for i in order], color=PURPLE, width=0.6)
            ax.set_ylabel("Marginal contribution to risk (ann. %)")
            ax.set_title("Marginal Contribution to Risk")
            ax.set_xticklabels([cols[i] for i in order], rotation=30, ha="right", fontsize=8)
            ax.grid(alpha=0.2, axis="y")
            fig.tight_layout()
            chart_mcr = _png(fig)
        except Exception:
            plt.close("all"); chart_mcr = None

        # ─────────────────────────── ⑥ Factor Risk Exposure (optional) ───────────────────────────
        factor_exposure_table = None
        factor_exposure_note = None
        chart_factor_exposure = None
        if factor_cols:
            if factors_df is None or len(factors_df) != len(port):
                factor_exposure_note = "Factor columns could not be aligned with the asset returns; factor exposure was skipped."
            else:
                try:
                    k = len(factor_cols)
                    if len(port) < k + 5:
                        factor_exposure_note = f"Need at least {k + 5} aligned observations for {k} factor(s); factor exposure was skipped."
                    else:
                        X = factors_df[factor_cols].values.astype(float)
                        Xc = np.column_stack([np.ones(len(port)), X])
                        coef, *_ = np.linalg.lstsq(Xc, port, rcond=None)
                        fitted = Xc @ coef
                        resid = port - fitted
                        ss_res = float(np.sum(resid ** 2))
                        ss_tot = float(np.sum((port - np.mean(port)) ** 2))
                        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else None
                        factor_exposure_table = [{"factor": "alpha (intercept)", "beta": _fin(float(coef[0]), 6)}]
                        for i, fc in enumerate(factor_cols):
                            factor_exposure_table.append({"factor": fc, "beta": _fin(float(coef[i + 1]), 4)})
                        factor_exposure_note = (
                            f"OLS regression of the portfolio's aggregate return (weighted sum of asset returns) "
                            f"on {k} factor(s); R² = {r2:.1%}." if r2 is not None else
                            "OLS regression of the portfolio's aggregate return on the selected factors."
                        )
                        try:
                            fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=115)
                            fbetas = [row["beta"] for row in factor_exposure_table[1:]]
                            fnames = factor_cols
                            colors = [GREEN if b >= 0 else RED for b in fbetas]
                            ax.bar(fnames, fbetas, color=colors, width=0.55)
                            ax.axhline(0, color="#111827", lw=0.7)
                            ax.set_ylabel("Factor beta")
                            ax.set_title("Factor Risk Exposure")
                            ax.set_xticklabels(fnames, rotation=30, ha="right", fontsize=8)
                            ax.grid(alpha=0.2, axis="y")
                            fig.tight_layout()
                            chart_factor_exposure = _png(fig)
                        except Exception:
                            plt.close("all"); chart_factor_exposure = None
                except Exception as e:
                    factor_exposure_note = f"Factor exposure regression failed: {e}"
        else:
            factor_exposure_note = (
                "Skipped: factor exposure requires factor return columns (e.g. Fama-French style) as additional "
                "input beyond the asset weights/returns this page collects. Pass optional `factor_cols` to enable it."
            )

        # ─────────────────────────── ⑦ Concentration Risk ───────────────────────────
        # This page only accepts asset return/price columns — no sector/asset-class grouping
        # column is collected — so concentration is reported at the asset level.
        concentration_table = [{"category": cols[i], "weight": _fin(w[i], 4),
                                 "risk_contribution_pct": _fin(float(rc_pct[i]), 4)} for i in range(n)]
        concentration_table = sorted(concentration_table, key=lambda z: -(z["weight"] or 0))
        concentration_note = (
            "No sector/asset-class/country grouping column is collected by this page, so concentration is shown "
            "at the asset level. Add a categorical grouping input to get category-level concentration."
        )
        chart_concentration = None
        try:
            fig, ax = plt.subplots(figsize=(6.2, 4.6), dpi=115)
            wts = [row["weight"] * 100 for row in concentration_table]
            ax.pie(wts, labels=[row["category"] for row in concentration_table], autopct="%1.0f%%",
                   colors=plt.cm.Blues(np.linspace(0.35, 0.85, n)), textprops={"fontsize": 8})
            ax.set_title("Portfolio Concentration (by weight)")
            fig.tight_layout()
            chart_concentration = _png(fig)
        except Exception:
            plt.close("all"); chart_concentration = None

        # ─────────────────────────── ⑧ Stress Testing ───────────────────────────
        stress_scenarios = []
        # 1. Market shock: beta-scaled if a benchmark is available, else a uniform shock (noted).
        if beta is not None:
            mkt_ret = beta * -0.10
            mkt_note = "beta-scaled to a -10% market move"
        else:
            mkt_ret = -0.10
            mkt_note = "no benchmark supplied — applied as a uniform -10% shock to the portfolio"
        stress_scenarios.append({"scenario": "Market -10%", "portfolio_return": _fin(mkt_ret, 5), "_detail": mkt_note})
        # 2. Parallel volatility shock: VaR scales with the vol multiplier at a fixed z-score.
        vol_shock_ret = -var_hist * 1.5
        stress_scenarios.append({"scenario": "Volatility Shock (+50%)", "portfolio_return": _fin(vol_shock_ret, 5),
                                  "_detail": "VaR(95%) scaled by the 1.5x volatility multiplier"})
        # 3. Correlation -> 1 (diversification breakdown): loss scales toward the undiversified (weighted-avg) vol.
        if ann_vol > 0:
            corr_break_ret = -var_hist * (weighted_avg_vol / ann_vol)
        else:
            corr_break_ret = -var_hist
        stress_scenarios.append({"scenario": "Correlation -> 1 (Diversification Breakdown)",
                                  "portfolio_return": _fin(corr_break_ret, 5),
                                  "_detail": "VaR(95%) rescaled by weighted-avg-vol / portfolio-vol (loses diversification benefit)"})
        # 4. Historical worst-case replay: worst single observed period.
        stress_scenarios.append({"scenario": "Historical Worst-Case Replay", "portfolio_return": _fin(worst_period, 5),
                                  "_detail": "worst single historical period return in the sample"})
        for s in stress_scenarios:
            ret = s["portfolio_return"] or 0.0
            s["estimated_loss_pct"] = _fin(-ret, 5)
            s["estimated_loss_amount"] = _fin(-ret * investment_amount, 2) if investment_amount else None
        stress_test_note = (
            "Simplified, illustrative scenarios computed from the available return sample (beta, VaR, weighted "
            "volatility, historical worst period) — not a full historical-scenario risk system with macro factor "
            "shocks (e.g. rate or oil-price stress tests), which would require external scenario data this page "
            "does not collect."
        )
        chart_stress_test = None
        try:
            fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=115)
            labels = [s["scenario"] for s in stress_scenarios]
            vals = [(s["portfolio_return"] or 0) * 100 for s in stress_scenarios]
            colors = [RED if v < 0 else GREEN for v in vals]
            ax.bar(labels, vals, color=colors, width=0.55)
            ax.axhline(0, color="#111827", lw=0.7)
            ax.set_ylabel("Scenario portfolio return (%)")
            ax.set_title("Stress Test Impact")
            ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7.5)
            ax.grid(alpha=0.2, axis="y")
            fig.tight_layout()
            chart_stress_test = _png(fig)
        except Exception:
            plt.close("all"); chart_stress_test = None

        # ─────────────────────────── ⑨ Rolling Risk ───────────────────────────
        n_obs = len(R)
        short_w = max(5, min(21, n_obs // 3))
        long_w = max(short_w + 5, min(63, n_obs // 2)) if n_obs // 2 > short_w else short_w
        port_s = pd.Series(port)
        roll_vol_short = port_s.rolling(short_w).std(ddof=1) * np.sqrt(ppy)
        roll_vol_long = port_s.rolling(long_w).std(ddof=1) * np.sqrt(ppy) if long_w != short_w else None

        roll_avg_corr = None
        if n > 1:
            corr_vals = []
            for t in range(short_w, n_obs + 1):
                window = R.iloc[t - short_w:t]
                cm = window.corr().values
                iu = np.triu_indices(n, k=1)
                corr_vals.append(float(np.mean(cm[iu])))
            roll_avg_corr = pd.Series(corr_vals, index=range(short_w - 1, n_obs))

        roll_beta = None
        if bench is not None and len(bench) == n_obs:
            bv = bench.values
            betas = []
            for t in range(short_w, n_obs + 1):
                bw_ = bv[t - short_w:t]
                pw_ = port[t - short_w:t]
                var_bw = np.var(bw_, ddof=1)
                betas.append(float(np.cov(pw_, bw_, ddof=1)[0, 1] / var_bw) if var_bw > 0 else None)
            roll_beta = pd.Series(betas, index=range(short_w - 1, n_obs))

        rolling_risk_table = []
        for label, wlen in [("Short window", short_w), ("Long window", long_w if long_w != short_w else n_obs)]:
            wlen = min(wlen, n_obs)
            seg = port[-wlen:]
            vol_w = float(np.std(seg, ddof=1) * np.sqrt(ppy)) if wlen > 1 else None
            beta_w = None
            if bench is not None and len(bench) == n_obs and wlen > 1:
                bseg = bench.values[-wlen:]
                var_bseg = np.var(bseg, ddof=1)
                beta_w = float(np.cov(seg, bseg, ddof=1)[0, 1] / var_bseg) if var_bseg > 0 else None
            corr_w = None
            if n > 1 and wlen > 1:
                cm = R.iloc[-wlen:].corr().values
                iu = np.triu_indices(n, k=1)
                corr_w = float(np.mean(cm[iu]))
            rolling_risk_table.append({
                "window_label": label, "window_size": wlen,
                "volatility": _fin(vol_w, 5), "beta": _fin(beta_w, 4) if beta_w is not None else None,
                "avg_correlation": _fin(corr_w, 4) if corr_w is not None else None,
            })

        chart_rolling_risk = None
        try:
            n_panels = 1 + (1 if roll_avg_corr is not None else 0) + (1 if roll_beta is not None else 0)
            fig, axes = plt.subplots(n_panels, 1, figsize=(9.5, 3.1 * n_panels), dpi=115, sharex=True)
            axes = np.atleast_1d(axes)
            idx0 = 0
            ax = axes[idx0]
            ax.plot(roll_vol_short.index, roll_vol_short.values * 100, color=BLUE, lw=1.3, label=f"Rolling vol ({short_w}p, ann. %)")
            if roll_vol_long is not None:
                ax.plot(roll_vol_long.index, roll_vol_long.values * 100, color=AMBER, lw=1.1, ls="--", label=f"Rolling vol ({long_w}p, ann. %)")
            ax.set_ylabel("Vol (%)"); ax.legend(fontsize=7.5, frameon=False); ax.grid(alpha=0.2)
            ax.set_title("Rolling Risk Structure")
            idx0 += 1
            if roll_avg_corr is not None:
                ax = axes[idx0]
                ax.plot(roll_avg_corr.index, roll_avg_corr.values, color=PURPLE, lw=1.3, label=f"Rolling avg pairwise correlation ({short_w}p)")
                ax.axhline(0, color="#111827", lw=0.6)
                ax.set_ylabel("Avg corr"); ax.legend(fontsize=7.5, frameon=False); ax.grid(alpha=0.2)
                idx0 += 1
            if roll_beta is not None:
                ax = axes[idx0]
                ax.plot(roll_beta.index, roll_beta.values, color=GREEN, lw=1.3, label=f"Rolling beta ({short_w}p)")
                ax.axhline(1, color="#111827", lw=0.6, ls=":")
                ax.set_ylabel("Beta"); ax.legend(fontsize=7.5, frameon=False); ax.grid(alpha=0.2)
            axes[-1].set_xlabel("Period")
            fig.tight_layout()
            chart_rolling_risk = _png(fig)
        except Exception:
            plt.close("all"); chart_rolling_risk = None

        rolling_risk_note = None if bench is not None else "Rolling beta omitted — no benchmark column was supplied."

        charts = {
            "risk_decomposition": chart_risk_decomposition,
            "correlation_heatmap": chart_correlation_heatmap,
            "diversification": chart_diversification,
            "mcr": chart_mcr,
            "factor_exposure": chart_factor_exposure,
            "concentration": chart_concentration,
            "stress_test": chart_stress_test,
            "rolling_risk": chart_rolling_risk,
        }

        # original 3-panel plot (kept intact — other frontend parts rely on it)
        plot = None
        try:
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14.5, 4.6), dpi=115)
            ax1.hist(port * 100, bins=min(40, max(10, len(port) // 5)), color="#93c5fd", edgecolor="white")
            ax1.axvline(-var_hist * 100, color="#dc2626", ls="--", lw=1.5, label=f"VaR {conf:.0%}")
            ax1.axvline(-cvar_hist * 100, color="#7f1d1d", ls=":", lw=1.5, label=f"CVaR {conf:.0%}")
            ax1.set_xlabel("Period return (%)"); ax1.set_ylabel("Frequency")
            ax1.set_title("Return distribution"); ax1.legend(fontsize=8, frameon=False)
            ax2.fill_between(range(len(drawdown)), drawdown * 100, 0, color="#fca5a5")
            ax2.plot(drawdown * 100, color="#dc2626", lw=1)
            ax2.set_xlabel("Period"); ax2.set_ylabel("Drawdown (%)")
            ax2.set_title(f"Drawdown (max {max_dd:.1%})")
            order = np.argsort(rc_pct)
            ax3.barh([cols[i] for i in order], [rc_pct[i] * 100 for i in order], color="#2563eb")
            ax3.set_xlabel("Risk contribution (%)"); ax3.set_title("Risk contribution by asset")
            fig.tight_layout()
            plot = _png(fig)
        except Exception:
            plt.close("all"); plot = None

        top = assets_sorted[0] if assets_sorted else None
        interpretation = (
            f"The portfolio has an annualised volatility of {ann_vol:.1%} and, at {conf:.0%} confidence, a "
            f"historical Value-at-Risk of {var_hist:.2%} per period — on a bad day (the worst {a:.0%} of periods) "
            f"you would expect to lose at least that much, and {cvar_hist:.2%} on average once you are in that tail "
            f"(CVaR). The deepest peak-to-trough decline over the sample was {max_dd:.1%}. "
            + (f"Risk is concentrated in {top['asset']}, which carries {top['risk_contribution']:.0%} of the total "
               f"portfolio risk despite a {top['weight']:.0%} weight — a sign the risk is less diversified than the "
               f"weights suggest. " if top and n > 1 else "")
            + "VaR describes normal-times tail risk, not worst-case; stress scenarios can exceed it."
        )

        results = {
            "status": "ok", "n_assets": n, "n_obs": int(len(R)), "confidence": _fin(conf, 4),
            "periods_per_year": ppy, "weights_used": ("custom" if (w_in and len(w_in) == n) else "equal"),
            "annual_return": _fin(ann_return, 5), "annual_volatility": _fin(ann_vol, 5),
            "sharpe": _fin(sharpe, 4) if sharpe is not None else None,
            "sortino": _fin(sortino, 4) if sortino is not None else None,
            "var_hist": _fin(var_hist, 5), "cvar_hist": _fin(cvar_hist, 5), "var_normal": _fin(var_norm, 5),
            "downside_deviation": _fin(dd_dev * np.sqrt(ppy), 5), "max_drawdown": _fin(max_dd, 5),
            "assets": assets_sorted, "corr_labels": cols, "corr_matrix": corr,
            "interpretation": interpretation,

            # ① Risk Summary
            "risk_summary": risk_summary, "risk_summary_note": risk_summary_note,
            "benchmark_col": benchmark_col, "beta": _fin(beta, 4) if beta is not None else None,

            # ④ Diversification
            "diversification_table": diversification_table,

            # ⑤ MCR
            "mcr_table": mcr_table_sorted,

            # ⑥ Factor exposure (optional)
            "factor_exposure_table": factor_exposure_table, "factor_exposure_note": factor_exposure_note,

            # ⑦ Concentration
            "concentration_table": concentration_table, "concentration_note": concentration_note,

            # ⑧ Stress testing
            "stress_test_table": stress_scenarios, "stress_test_note": stress_test_note,
            "investment_amount": _fin(investment_amount, 2) if investment_amount else None,

            # ⑨ Rolling risk
            "rolling_risk_table": rolling_risk_table, "rolling_risk_note": rolling_risk_note,
            "rolling_windows": {"short": short_w, "long": long_w},

            "charts": charts,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


def _erfinv(y):
    # Winitzki approximation to the inverse error function (no scipy dependency).
    a = 0.147
    ln = np.log(1 - y * y)
    t = 2 / (np.pi * a) + ln / 2
    return np.sign(y) * np.sqrt(np.sqrt(t * t - ln / a) - t)


if __name__ == "__main__":
    main()
