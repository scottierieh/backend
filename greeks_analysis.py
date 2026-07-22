#!/usr/bin/env python3
"""Greeks Analysis — option sensitivities across the underlying. QuantLib-Python.

Prices a European option and profiles all five Greeks (delta, gamma, vega,
theta, rho) as the underlying moves, so you can see how the risk changes with
price — not just at the current spot.

Input (from greeks-analysis-page.tsx):
    option_type    : "call" | "put"
    spot           : float  underlying price S
    strike         : float  strike K
    expiry_years   : float  time to expiry T
    volatility     : float  annual sigma
    risk_free_rate : float  annual r
    dividend_yield : float  annual q (default 0)
Output: { results: {greeks at spot + profile arrays}, plot }
"""
import sys, json, io, base64
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import QuantLib as ql


def _fin(x, nd=6):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def main():
    try:
        p = json.load(sys.stdin)
        otype = (p.get("option_type") or "call").lower()
        is_call = otype != "put"
        S = float(p.get("spot")); K = float(p.get("strike"))
        T = float(p.get("expiry_years")); sigma = float(p.get("volatility"))
        r = float(p.get("risk_free_rate") or 0.0); q = float(p.get("dividend_yield") or 0.0)
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            raise ValueError("Spot, strike, expiry and volatility must all be positive.")

        val_date = ql.Date(1, 1, 2022)
        ql.Settings.instance().evaluationDate = val_date
        dc = ql.Actual365Fixed(); cal = ql.NullCalendar()
        maturity = val_date + int(round(T * 365))
        payoff = ql.PlainVanillaPayoff(ql.Option.Call if is_call else ql.Option.Put, K)
        exercise = ql.EuropeanExercise(maturity)
        rf_ts = ql.YieldTermStructureHandle(ql.FlatForward(val_date, r, dc))
        div_ts = ql.YieldTermStructureHandle(ql.FlatForward(val_date, q, dc))
        vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(val_date, cal, sigma, dc))

        def greeks_at(s):
            spot_h = ql.QuoteHandle(ql.SimpleQuote(float(s)))
            proc = ql.BlackScholesMertonProcess(spot_h, div_ts, rf_ts, vol_ts)
            opt = ql.VanillaOption(payoff, exercise)
            opt.setPricingEngine(ql.AnalyticEuropeanEngine(proc))
            return {
                "price": float(opt.NPV()), "delta": float(opt.delta()), "gamma": float(opt.gamma()),
                "vega": float(opt.vega()) / 100.0, "theta": float(opt.theta()) / 365.0,
                "rho": float(opt.rho()) / 100.0,
            }

        at = greeks_at(S)

        # profile across underlying
        srange = np.linspace(max(0.3 * S, 0.01), 1.7 * S, 60)
        prof = {k: [] for k in ("price", "delta", "gamma", "vega", "theta", "rho")}
        for s in srange:
            g = greeks_at(s)
            for k in prof:
                prof[k].append(_fin(g[k], 6))

        # plot: 2x3 panel of Greeks vs underlying
        plot = None
        try:
            fig, axes = plt.subplots(2, 3, figsize=(13.5, 7.5), dpi=115)
            panels = [("price", "Value", "#2563eb"), ("delta", "Delta", "#16a34a"),
                      ("gamma", "Gamma", "#9333ea"), ("vega", "Vega (per 1%)", "#0891b2"),
                      ("theta", "Theta (per day)", "#dc2626"), ("rho", "Rho (per 1%)", "#f59e0b")]
            for ax, (key, title, col) in zip(axes.ravel(), panels):
                ys = [np.nan if v is None else v for v in prof[key]]
                ax.plot(srange, ys, color=col, lw=2)
                ax.axvline(K, color="#94a3b8", ls=":", lw=1)
                ax.axvline(S, color="#111827", ls="--", lw=1)
                ax.scatter([S], [at[key]], color=col, zorder=5, s=40, edgecolor="white")
                ax.set_title(title, fontsize=10); ax.grid(alpha=0.2)
                ax.set_xlabel("Underlying", fontsize=8)
            fig.suptitle(f"{otype.capitalize()} option Greeks vs underlying (K={K:g}, spot marked)", fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        moneyness = S / K
        state = ("in-the-money" if (is_call and S > K) or (not is_call and S < K)
                 else "out-of-the-money" if (is_call and S < K) or (not is_call and S > K) else "at-the-money")

        interpretation = (
            f"At a spot of {S:g}, this {otype} option has a delta of {at['delta']:.3f} — it behaves like "
            f"{abs(at['delta']):.2f} shares of the underlying — and a gamma of {at['gamma']:.4f}, which is how "
            f"quickly that delta shifts as the price moves. Gamma (and therefore the risk of delta changing) peaks "
            f"near the strike {K:g} and fades deep in- or out-of-the-money, which is exactly what the curves show. "
            f"Vega {at['vega']:.4f} per 1% vol and theta {at['theta']:.4f} per day quantify the two forces that "
            f"trade off in every option: the value of uncertainty versus the cost of time passing."
        )

        results = {
            "status": "ok", "option_type": otype, "state": state, "moneyness": _fin(moneyness, 4),
            "spot": _fin(S, 4), "strike": _fin(K, 4), "expiry_years": _fin(T, 4),
            "volatility": _fin(sigma, 6), "risk_free_rate": _fin(r, 6), "dividend_yield": _fin(q, 6),
            "price": _fin(at["price"], 6), "delta": _fin(at["delta"], 6), "gamma": _fin(at["gamma"], 6),
            "vega": _fin(at["vega"], 6), "theta": _fin(at["theta"], 6), "rho": _fin(at["rho"], 6),
            "profile": {"underlying": [_fin(x, 4) for x in srange.tolist()], **prof},
            "interpretation": interpretation,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
