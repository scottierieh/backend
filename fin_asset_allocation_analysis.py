#!/usr/bin/env python3
"""Asset Allocation — compare allocation strategies. numpy / scipy / pandas.

Builds and compares several classic allocation schemes from historical returns:
equal-weight, inverse-volatility, minimum-variance, and risk parity (equal risk
contribution), reporting each portfolio's weights, return, volatility and Sharpe.

Input (from fin-asset-allocation-page.tsx):
    data             : list[dict]
    asset_cols       : string[]  >= 2 assets
    is_returns       : bool
    return_type      : "simple"|"log"
    periods_per_year : int   (default 252)
    rf_annual        : float (default 0.02)
Output: { results: {strategies[], risk_contributions}, plot }
"""
import sys, json, io, base64
import numpy as np
import pandas as pd
from scipy.optimize import minimize

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


def _risk_contrib(w, S):
    port_var = float(w @ S @ w)
    if port_var <= 0:
        return np.zeros_like(w)
    mrc = S @ w
    rc = w * mrc / np.sqrt(port_var)
    return rc / rc.sum() if rc.sum() != 0 else rc


def main():
    try:
        p = json.load(sys.stdin)
        rows = p.get("data") or []
        if not rows:
            raise ValueError("No data provided.")
        df = pd.DataFrame(rows)
        cols = [c for c in (p.get("asset_cols") or []) if c in df.columns]
        if len(cols) < 2:
            raise ValueError("Select at least two asset columns.")
        is_returns = bool(p.get("is_returns", False))
        rtype = (p.get("return_type") or "simple").lower()
        ppy = int(p.get("periods_per_year") or 252)
        rf = float(p.get("rf_annual") or 0.02)

        R = _returns_df(df, cols, is_returns, rtype)
        if len(R) < 10:
            raise ValueError("Need at least 10 aligned return observations.")
        n = len(cols)
        mu = R.mean().values * ppy
        S = R.cov().values * ppy
        vols = np.sqrt(np.diag(S))

        def perf(w):
            ret = float(w @ mu)
            vol = float(np.sqrt(w @ S @ w))
            shr = (ret - rf) / vol if vol > 0 else None
            return ret, vol, shr

        strategies = {}
        # equal weight
        strategies["equal_weight"] = np.ones(n) / n
        # inverse volatility
        iv = 1.0 / vols
        strategies["inverse_volatility"] = iv / iv.sum()
        # minimum variance (long-only)
        cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bnds = [(0, 1)] * n
        w0 = np.ones(n) / n
        try:
            res = minimize(lambda w: float(w @ S @ w), w0, method="SLSQP", bounds=bnds, constraints=cons)
            strategies["minimum_variance"] = res.x / res.x.sum()
        except Exception:
            strategies["minimum_variance"] = strategies["equal_weight"]
        # risk parity (equal risk contribution)
        def rp_obj(w):
            rc = _risk_contrib(w, S)
            return float(np.sum((rc - 1.0 / n) ** 2))
        try:
            res = minimize(rp_obj, w0, method="SLSQP", bounds=bnds, constraints=cons)
            strategies["risk_parity"] = res.x / res.x.sum()
        except Exception:
            strategies["risk_parity"] = strategies["inverse_volatility"]

        labels = {"equal_weight": "Equal weight", "inverse_volatility": "Inverse volatility",
                  "minimum_variance": "Minimum variance", "risk_parity": "Risk parity"}
        out = []
        for key, w in strategies.items():
            w = np.clip(w, 0, None); w = w / w.sum()
            ret, vol, shr = perf(w)
            rc = _risk_contrib(w, S)
            out.append({
                "key": key, "name": labels[key],
                "weights": [{"asset": cols[i], "weight": _fin(float(w[i]), 4),
                             "risk_contribution": _fin(float(rc[i]), 4)} for i in range(n)],
                "annual_return": _fin(ret, 5), "annual_volatility": _fin(vol, 5),
                "sharpe": _fin(shr, 4) if shr is not None else None,
                "max_weight": _fin(float(np.max(w)), 4),
                "effective_n": _fin(float(1.0 / np.sum(w ** 2)), 3),   # diversification (inverse HHI)
            })

        assets_info = [{"asset": cols[i], "vol_annual": _fin(float(vols[i]), 5),
                        "ann_return": _fin(float(mu[i]), 5)} for i in range(n)]

        plot = None
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=118,
                                           gridspec_kw={"width_ratios": [1.4, 1]})
            keys = [o["key"] for o in out]
            names = [o["name"] for o in out]
            bottoms = np.zeros(len(out))
            palette = plt.cm.tab10(np.linspace(0, 1, n))
            for i in range(n):
                vals = np.array([o["weights"][i]["weight"] * 100 for o in out])
                ax1.bar(names, vals, bottom=bottoms, color=palette[i], label=cols[i])
                bottoms += vals
            ax1.set_ylabel("Weight (%)"); ax1.set_title("Allocation by strategy")
            ax1.legend(fontsize=7, frameon=False, ncol=min(n, 4), loc="upper center", bbox_to_anchor=(0.5, -0.12))
            ax1.tick_params(axis="x", rotation=15)
            # risk-return scatter
            for o in out:
                ax2.scatter(o["annual_volatility"] * 100, o["annual_return"] * 100, s=90, zorder=5)
                ax2.annotate(o["name"], (o["annual_volatility"] * 100, o["annual_return"] * 100),
                             fontsize=7, textcoords="offset points", xytext=(5, 3))
            ax2.scatter([a["vol_annual"] * 100 for a in assets_info], [a["ann_return"] * 100 for a in assets_info],
                        color="#94a3b8", marker="x", s=40, label="Individual assets")
            ax2.set_xlabel("Volatility (ann. %)"); ax2.set_ylabel("Return (ann. %)")
            ax2.set_title("Risk vs return"); ax2.legend(fontsize=7, frameon=False); ax2.grid(alpha=0.2)
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        mv = next(o for o in out if o["key"] == "minimum_variance")
        rp = next(o for o in out if o["key"] == "risk_parity")
        ew = next(o for o in out if o["key"] == "equal_weight")
        interpretation = (
            f"Across {n} assets, the four allocation strategies trade off risk, return and diversification "
            f"differently. The minimum-variance portfolio reaches the lowest volatility ({mv['annual_volatility']:.1%}) "
            f"but often concentrates in the calmest assets, while equal weight is the simplest and most diversified by "
            f"capital ({ew['annual_volatility']:.1%} volatility). Risk parity ({rp['annual_volatility']:.1%}) aims for "
            "each asset to contribute the same share of risk rather than the same share of capital, which usually "
            "spreads risk more evenly than any capital-weighted scheme. None dominates on every axis — the right "
            "choice depends on whether you prioritise low risk, simplicity, or balanced risk contributions, and all "
            "are estimated from historical moments that shift over time."
        )

        results = {
            "status": "ok", "n_assets": n, "n_obs": int(len(R)), "periods_per_year": ppy, "rf_annual": _fin(rf, 5),
            "strategies": out, "assets": assets_info,
            "interpretation": interpretation,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
