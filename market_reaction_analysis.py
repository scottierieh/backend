#!/usr/bin/env python3
"""Market Reaction Analysis — how fast and asymmetrically an asset reacts to the
market. statsmodels.

Two lenses:
  1. Distributed-lag regression of the asset on contemporaneous and lagged market
     returns -> immediate vs delayed reaction (price-discovery speed / lead-lag).
  2. Up/down asymmetry: separate betas in rising vs falling markets.

Input (from market-reaction-page.tsx):
    data        : list[dict]
    asset_col   : str
    market_col  : str
    is_returns  : bool
    return_type : "simple"|"log"
    n_lags      : int   (default 3)
Output: { results: {lag_betas, asymmetry, speed}, plot }
"""
import sys, json, io, base64
import numpy as np
import pandas as pd
import statsmodels.api as sm

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
        asset_col = p.get("asset_col"); market_col = p.get("market_col")
        is_returns = bool(p.get("is_returns", False))
        rtype = (p.get("return_type") or "simple").lower()
        n_lags = int(p.get("n_lags") or 3)
        if not asset_col or asset_col not in df.columns:
            raise ValueError("Select the asset return column.")
        if not market_col or market_col not in df.columns:
            raise ValueError("Select the market return column.")
        n_lags = max(1, min(n_lags, 8))

        a = _ret(df[asset_col], is_returns, rtype)
        m = _ret(df[market_col], is_returns, rtype)
        base = pd.concat([a.rename("a"), m.rename("m")], axis=1).dropna().reset_index(drop=True)
        if len(base) < n_lags + 20:
            raise ValueError(f"Need at least {n_lags + 20} aligned observations.")

        # ---- distributed lag regression ----
        cols = {}
        cols["m_0"] = base["m"]
        for k in range(1, n_lags + 1):
            cols[f"m_{k}"] = base["m"].shift(k)
        X = pd.DataFrame(cols)
        reg = pd.concat([base["a"], X], axis=1).dropna().reset_index(drop=True)
        y = reg["a"].values
        Xv = sm.add_constant(reg[list(cols.keys())].values)
        fit = sm.OLS(y, Xv).fit()

        names = ["alpha"] + list(cols.keys())
        lag_betas = []
        contemp = float(fit.params[1])
        total_beta = float(np.sum(fit.params[1:]))
        for i, nm in enumerate(names):
            if nm == "alpha":
                continue
            lag = int(nm.split("_")[1])
            lag_betas.append({"lag": lag, "beta": _fin(float(fit.params[i]), 5),
                              "t_stat": _fin(float(fit.tvalues[i]), 4),
                              "p_value": _fin(float(fit.pvalues[i]), 6),
                              "significant": bool(fit.pvalues[i] < 0.05)})
        # speed of adjustment: share of total beta in the contemporaneous term
        speed = (contemp / total_beta) if total_beta != 0 else None
        delayed_share = (1 - speed) if speed is not None else None
        n_sig_lags = sum(1 for l in lag_betas if l["lag"] >= 1 and l["significant"])

        # ---- up/down asymmetry ----
        up = base["m"] > 0
        Xa = pd.DataFrame({
            "m": base["m"],
            "m_down": base["m"] * (~up).astype(float),   # extra slope in down markets
        })
        Xa_ = sm.add_constant(Xa.values)
        fit2 = sm.OLS(base["a"].values, Xa_).fit()
        beta_up = float(fit2.params[1])
        beta_down = float(fit2.params[1] + fit2.params[2])
        asym_coef = float(fit2.params[2]); asym_p = float(fit2.pvalues[2])
        asymmetric = bool(asym_p < 0.05)

        # ---- optional: volume reaction ----
        volume_reaction = None
        try:
            volume_col = p.get("volume_col")
            if volume_col and volume_col in df.columns:
                vol = pd.to_numeric(df[volume_col], errors="coerce")
                vol_aligned = vol.loc[base.index].reset_index(drop=True) if len(vol) == len(base) else vol.reindex(range(len(base)))
                vol_aligned = vol_aligned.dropna()
                if len(vol_aligned) >= 20:
                    mid_v = len(vol_aligned) // 2
                    baseline_avg = float(vol_aligned.iloc[:mid_v].mean())
                    event_avg = float(vol_aligned.iloc[mid_v:].mean())
                    pct_change = (event_avg / baseline_avg - 1.0) if baseline_avg else None
                    volume_reaction = {"baseline_avg": _fin(baseline_avg, 2), "event_avg": _fin(event_avg, 2),
                                       "pct_change": _fin(pct_change, 4) if pct_change is not None else None}
        except Exception:
            volume_reaction = None

        # ---- optional: volatility reaction (before vs after sample midpoint) ----
        volatility_reaction = None
        try:
            asset_series = base["a"].reset_index(drop=True)
            mid_vol = len(asset_series) // 2
            if mid_vol >= 10 and (len(asset_series) - mid_vol) >= 10:
                vol_before = float(asset_series.iloc[:mid_vol].std(ddof=1))
                vol_after = float(asset_series.iloc[mid_vol:].std(ddof=1))
                pct_change_v = (vol_after / vol_before - 1.0) if vol_before else None
                volatility_reaction = {"before": _fin(vol_before, 6), "after": _fin(vol_after, 6),
                                       "pct_change": _fin(pct_change_v, 4) if pct_change_v is not None else None}
        except Exception:
            volatility_reaction = None

        # ---- optional: industry/market grouping ----
        by_group = None
        try:
            group_col = p.get("group_col")
            if group_col and group_col in df.columns:
                grp_series = df[group_col]
                grp_aligned = grp_series.loc[base.index].reset_index(drop=True) if len(grp_series) == len(df) else None
                if grp_aligned is not None:
                    rows_out = []
                    for gval in pd.unique(grp_aligned.dropna()):
                        gmask = (grp_aligned == gval).values
                        sub = base.loc[gmask].reset_index(drop=True)
                        if len(sub) < n_lags + 20:
                            continue
                        gy = sub["a"].values
                        gX = sm.add_constant(sub["m"].values)
                        gfit = sm.OLS(gy, gX).fit()
                        g_beta0 = float(gfit.params[1]); g_p = float(gfit.pvalues[1])
                        rows_out.append({"group": str(gval), "beta_lag0": _fin(g_beta0, 5),
                                          "significant": bool(g_p < 0.05), "n": int(len(sub))})
                    if rows_out:
                        by_group = rows_out
        except Exception:
            by_group = None

        n_extra_panels = (1 if volume_reaction is not None else 0) + (1 if volatility_reaction is not None else 0)
        plot = None
        try:
            n_panels = 2 + n_extra_panels
            fig = plt.figure(figsize=(12.5 if n_panels <= 2 else 12.5, 5 if n_panels <= 2 else 5 * ((n_panels + 1) // 2)), dpi=120)
            ncols = 2
            nrows = (n_panels + 1) // 2
            ax1 = fig.add_subplot(nrows, ncols, 1)
            ax2 = fig.add_subplot(nrows, ncols, 2)
            lags = [l["lag"] for l in lag_betas]
            betas = [l["beta"] for l in lag_betas]
            cols_c = ["#2563eb" if l["significant"] else "#94a3b8" for l in lag_betas]
            ax1.bar(lags, betas, color=cols_c)
            ax1.axhline(0, color="#111827", lw=0.7)
            ax1.set_xlabel("Lag (0 = same period)"); ax1.set_ylabel("Reaction (beta)")
            ax1.set_title("Reaction to market by lag (blue = significant)")
            ax1.set_xticks(lags)
            # up/down betas
            ax2.bar(["Up market", "Down market"], [beta_up, beta_down], color=["#16a34a", "#dc2626"])
            for i, v in enumerate([beta_up, beta_down]):
                ax2.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
            ax2.axhline(0, color="#111827", lw=0.7)
            ax2.set_ylabel("Beta"); ax2.set_title(f"Up vs down market beta ({'asymmetric' if asymmetric else 'symmetric'})")

            panel_i = 3
            if volume_reaction is not None:
                ax3 = fig.add_subplot(nrows, ncols, panel_i); panel_i += 1
                ax3.bar(["Baseline", "Event window"], [volume_reaction["baseline_avg"], volume_reaction["event_avg"]], color=["#94a3b8", "#2563eb"])
                ax3.set_ylabel("Avg volume"); ax3.set_title("Trading volume reaction")
            if volatility_reaction is not None:
                ax4 = fig.add_subplot(nrows, ncols, panel_i); panel_i += 1
                ax4.bar(["Before", "After"], [volatility_reaction["before"], volatility_reaction["after"]], color=["#94a3b8", "#dc2626"])
                ax4.set_ylabel("Realized volatility"); ax4.set_title("Volatility reaction (before vs after midpoint)")

            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        speed_txt = ("almost all of the reaction happens in the same period (efficient, fast price discovery)"
                     if speed is not None and speed > 0.9 else
                     "most of the reaction is immediate, with some spilling into later periods"
                     if speed is not None and speed > 0.7 else
                     "a substantial part of the reaction is delayed, appearing at later lags — a sign of slow price "
                     "discovery, illiquidity, or that the asset lags the market")
        interpretation = (
            f"The asset's contemporaneous reaction to the market is a beta of {contemp:.2f}, and summing the lagged "
            f"reactions gives a total (long-run) beta of {total_beta:.2f}. "
            + (f"About {speed:.0%} of the total reaction occurs immediately — {speed_txt}. " if speed is not None else "")
            + (f"The reaction is asymmetric: the beta is {beta_up:.2f} in rising markets versus {beta_down:.2f} in "
               f"falling markets (difference significant, p = {asym_p:.3f}), so the asset "
               + ("falls harder than it rises with the market — a downside-amplifying profile. " if beta_down > beta_up else
                  "rises more than it falls with the market. ")
               if asymmetric else
               f"The up-market beta ({beta_up:.2f}) and down-market beta ({beta_down:.2f}) are not statistically "
               f"different (p = {asym_p:.3f}), so the reaction is symmetric. ")
        )

        results = {
            "status": "ok", "asset": asset_col, "market": market_col, "n_obs": int(len(reg)), "n_lags": n_lags,
            "contemporaneous_beta": _fin(contemp, 5), "total_beta": _fin(total_beta, 5),
            "speed_of_adjustment": _fin(speed, 4) if speed is not None else None,
            "delayed_share": _fin(delayed_share, 4) if delayed_share is not None else None,
            "n_significant_lags": n_sig_lags, "lag_betas": lag_betas,
            "beta_up": _fin(beta_up, 5), "beta_down": _fin(beta_down, 5),
            "asymmetry_coef": _fin(asym_coef, 5), "asymmetry_p": _fin(asym_p, 6), "asymmetric": asymmetric,
            "r_squared": _fin(float(fit.rsquared), 4),
            "interpretation": interpretation,
        }
        if volume_reaction is not None:
            results["volume_reaction"] = volume_reaction
        if volatility_reaction is not None:
            results["volatility_reaction"] = volatility_reaction
        if by_group is not None:
            results["by_group"] = by_group
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
