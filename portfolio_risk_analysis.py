#!/usr/bin/env python3
"""Portfolio Risk — volatility, VaR/CVaR, drawdown and risk contributions.
numpy / pandas.

Input (from portfolio-risk-page.tsx):
    data             : list[dict]
    asset_cols       : string[]   price or return columns (>= 1)
    weights          : number[]   (optional) portfolio weights; default equal
    is_returns       : bool
    return_type      : "simple"|"log"
    periods_per_year : int    (default 252)
    confidence       : float  (default 0.95) VaR/CVaR confidence
Output: { results: {...}, plot } (return distribution + drawdown + risk contributions).
"""
import sys, json, io, base64
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _fin(x, nd=6):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def _returns_df(df, cols, is_returns, rtype):
    out = {}
    for c in cols:
        s = pd.to_numeric(df[c], errors="coerce")
        if is_returns:
            out[c] = s
        elif rtype == "log":
            if (s <= 0).any():
                raise ValueError(f"Log returns require positive prices ({c}).")
            out[c] = np.log(s / s.shift(1))
        else:
            out[c] = s / s.shift(1) - 1.0
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
        from math import erf, sqrt
        # normal quantile via inverse erf approximation using numpy
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

        # risk contributions: MCR = (Sigma w) / sigma_p ; contribution = w * MCR
        Sigma = R.cov().values
        port_var = float(w @ Sigma @ w)
        port_sd = np.sqrt(port_var) if port_var > 0 else 0.0
        if port_sd > 0:
            mcr = (Sigma @ w) / port_sd
            rc = w * mcr
            rc_pct = rc / rc.sum() if rc.sum() != 0 else rc
        else:
            rc = np.zeros(n); rc_pct = np.zeros(n)

        assets = [{"asset": cols[i], "weight": _fin(w[i], 4),
                   "vol_annual": _fin(float(np.std(R[cols[i]], ddof=1) * np.sqrt(ppy)), 5),
                   "risk_contribution": _fin(float(rc_pct[i]), 4)} for i in range(n)]
        assets_sorted = sorted(assets, key=lambda z: -(z["risk_contribution"] or 0))

        corr = R.corr().round(4).values.tolist() if n > 1 else [[1.0]]

        # plot
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
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
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
