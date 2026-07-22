#!/usr/bin/env python3
"""Black-Scholes — European option price and Greeks. QuantLib-Python.

Input (from black-scholes-page.tsx):
    option_type      : "call" | "put"
    spot             : float  underlying price S
    strike           : float  strike K
    expiry_years     : float  time to expiry T (years)
    volatility       : float  annual sigma (e.g. 0.20 = 20%)
    risk_free_rate   : float  annual r
    dividend_yield   : float  annual q (default 0)
Output: { results: {price, delta, gamma, vega, theta, rho, d1, d2,
                    intrinsic_value, time_value, moneyness, ...}, plot }
"""
import sys, json, io, base64
import numpy as np
from scipy.stats import norm

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

        # ---- QuantLib European option (analytic Black-Scholes-Merton) ----
        val_date = ql.Date(1, 1, 2022)
        ql.Settings.instance().evaluationDate = val_date
        dc = ql.Actual365Fixed(); cal = ql.NullCalendar()
        maturity = val_date + int(round(T * 365))
        payoff = ql.PlainVanillaPayoff(ql.Option.Call if is_call else ql.Option.Put, K)
        exercise = ql.EuropeanExercise(maturity)
        option = ql.VanillaOption(payoff, exercise)
        spot_h = ql.QuoteHandle(ql.SimpleQuote(S))
        rf_ts = ql.YieldTermStructureHandle(ql.FlatForward(val_date, r, dc))
        div_ts = ql.YieldTermStructureHandle(ql.FlatForward(val_date, q, dc))
        vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(val_date, cal, sigma, dc))
        process = ql.BlackScholesMertonProcess(spot_h, div_ts, rf_ts, vol_ts)
        option.setPricingEngine(ql.AnalyticEuropeanEngine(process))

        price = float(option.NPV())
        delta = float(option.delta())
        gamma = float(option.gamma())
        vega_raw = float(option.vega())      # per 1.00 (100%) vol
        theta_raw = float(option.theta())    # per year
        rho_raw = float(option.rho())        # per 1.00 rate
        vega = vega_raw / 100.0              # per 1% vol
        theta_day = theta_raw / 365.0        # per calendar day
        rho = rho_raw / 100.0               # per 1% rate

        # d1, d2 for reference
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
        time_value = price - intrinsic
        moneyness = S / K
        state = ("in-the-money" if intrinsic > 0 else "out-of-the-money" if
                 (is_call and S < K) or (not is_call and S > K) else "at-the-money")

        # ---- sensitivity sweeps: price vs volatility, price vs time-to-maturity ----
        # (same sweep-and-reprice pattern as greeks_analysis.py, but over sigma and T)
        vol_range = np.linspace(max(0.5 * sigma, 0.01), 2.0 * sigma, 25)
        sensitivity_vol = []
        for v in vol_range:
            vol_ts_v = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(val_date, cal, float(v), dc))
            proc_v = ql.BlackScholesMertonProcess(spot_h, div_ts, rf_ts, vol_ts_v)
            o_v = ql.VanillaOption(payoff, exercise); o_v.setPricingEngine(ql.AnalyticEuropeanEngine(proc_v))
            sensitivity_vol.append({"vol": _fin(v, 6), "price": _fin(o_v.NPV(), 6)})

        t_range = np.linspace(max(T * 0.02, 1.0 / 365.0), T, 25)
        sensitivity_time = []
        for tt in t_range:
            maturity_t = val_date + int(round(float(tt) * 365))
            exercise_t = ql.EuropeanExercise(maturity_t)
            proc_t = ql.BlackScholesMertonProcess(spot_h, div_ts, rf_ts, vol_ts)
            o_t = ql.VanillaOption(payoff, exercise_t); o_t.setPricingEngine(ql.AnalyticEuropeanEngine(proc_t))
            sensitivity_time.append({"t": _fin(tt, 6), "price": _fin(o_t.NPV(), 6)})

        # ---- plot: option value & payoff vs spot, plus vol/time sensitivity ----
        plot = None
        try:
            fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=118)
            ax, ax_vol, ax_time, ax_spare = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
            srange = np.linspace(max(0.3 * S, 0.01), 1.7 * S, 120)
            payoffs = np.maximum(srange - K, 0) if is_call else np.maximum(K - srange, 0)
            values = []
            for s in srange:
                spot_h2 = ql.QuoteHandle(ql.SimpleQuote(float(s)))
                proc2 = ql.BlackScholesMertonProcess(spot_h2, div_ts, rf_ts, vol_ts)
                o2 = ql.VanillaOption(payoff, exercise); o2.setPricingEngine(ql.AnalyticEuropeanEngine(proc2))
                values.append(o2.NPV())
            ax.plot(srange, payoffs, "--", color="#94a3b8", label="Payoff at expiry")
            ax.plot(srange, values, color="#2563eb", lw=2, label=f"Value now (T={T:g}y)")
            ax.axvline(K, color="#f59e0b", ls=":", lw=1, label=f"Strike {K:g}")
            ax.axvline(S, color="#dc2626", ls=":", lw=1, label=f"Spot {S:g}")
            ax.scatter([S], [price], color="#dc2626", zorder=5, s=50)
            ax.set_xlabel("Underlying price"); ax.set_ylabel("Option value")
            ax.set_title(f"{otype.capitalize()} option value vs underlying")
            ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.2)

            ax_vol.plot(vol_range, [row["price"] for row in sensitivity_vol], color="#16a34a", lw=2)
            ax_vol.axvline(sigma, color="#111827", ls="--", lw=1, label=f"Input σ {sigma:g}")
            ax_vol.scatter([sigma], [price], color="#16a34a", zorder=5, s=50, edgecolor="white")
            ax_vol.set_xlabel("Volatility (σ)"); ax_vol.set_ylabel("Option value")
            ax_vol.set_title("Price vs Volatility")
            ax_vol.legend(fontsize=8, frameon=False); ax_vol.grid(alpha=0.2)

            ax_time.plot(t_range, [row["price"] for row in sensitivity_time], color="#9333ea", lw=2)
            ax_time.axvline(T, color="#111827", ls="--", lw=1, label=f"Input T {T:g}y")
            ax_time.scatter([T], [price], color="#9333ea", zorder=5, s=50, edgecolor="white")
            ax_time.set_xlabel("Time to maturity (years)"); ax_time.set_ylabel("Option value")
            ax_time.set_title("Price vs Time to Maturity")
            ax_time.legend(fontsize=8, frameon=False); ax_time.grid(alpha=0.2)

            ax_spare.axis("off")
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        interpretation = (
            f"The fair Black-Scholes value of this {otype} option is {price:,.4f}. It is {state} "
            f"(spot {S:g} vs strike {K:g}), so {intrinsic:,.2f} of the price is intrinsic value and "
            f"{time_value:,.2f} is time value that decays to zero by expiry. Delta {delta:.3f} means the "
            f"price moves about {delta:.3f} per 1 unit move in the underlying; gamma {gamma:.4f} is how fast "
            f"that delta itself changes. Vega {vega:.4f} is the gain per 1% rise in volatility, and theta "
            f"{theta_day:.4f} is the daily time decay — the price the holder pays for waiting."
        )

        results = {
            "status": "ok", "option_type": otype,
            "spot": _fin(S, 4), "strike": _fin(K, 4), "expiry_years": _fin(T, 4),
            "volatility": _fin(sigma, 6), "risk_free_rate": _fin(r, 6), "dividend_yield": _fin(q, 6),
            "price": _fin(price, 6), "intrinsic_value": _fin(intrinsic, 6), "time_value": _fin(time_value, 6),
            "moneyness": _fin(moneyness, 4), "state": state,
            "delta": _fin(delta, 6), "gamma": _fin(gamma, 6),
            "vega": _fin(vega, 6), "theta": _fin(theta_day, 6), "rho": _fin(rho, 6),
            "vega_per_100": _fin(vega_raw, 6), "theta_per_year": _fin(theta_raw, 6),
            "d1": _fin(d1, 4), "d2": _fin(d2, 4),
            "sensitivity_vol": sensitivity_vol, "sensitivity_time": sensitivity_time,
            "interpretation": interpretation,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
