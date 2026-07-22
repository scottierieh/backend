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
        if abs(S - K) / K < 0.005:
            state = "at-the-money"
        elif intrinsic > 0:
            state = "in-the-money"
        else:
            state = "out-of-the-money"
        break_even = K + price if is_call else K - price

        # both call & put via the same QuantLib engine, so they match `price`
        Nd1, Nd2 = norm.cdf(d1), norm.cdf(d2)
        def ql_price(call_flag):
            po = ql.PlainVanillaPayoff(ql.Option.Call if call_flag else ql.Option.Put, K)
            o = ql.VanillaOption(po, exercise); o.setPricingEngine(ql.AnalyticEuropeanEngine(process))
            return float(o.NPV())
        call_price = price if is_call else ql_price(True)
        put_price = price if not is_call else ql_price(False)

        # ---- sensitivity: price vs volatility and vs time-to-expiry ----
        def bs_price(s_, k_, t_, sig_):
            if t_ <= 0 or sig_ <= 0:
                iv = max(s_ - k_, 0.0) if is_call else max(k_ - s_, 0.0)
                return float(iv)
            dd1 = (np.log(s_ / k_) + (r - q + 0.5 * sig_ ** 2) * t_) / (sig_ * np.sqrt(t_))
            dd2 = dd1 - sig_ * np.sqrt(t_)
            if is_call:
                return float(s_ * np.exp(-q * t_) * norm.cdf(dd1) - k_ * np.exp(-r * t_) * norm.cdf(dd2))
            return float(k_ * np.exp(-r * t_) * norm.cdf(-dd2) - s_ * np.exp(-q * t_) * norm.cdf(-dd1))

        vol_grid = np.linspace(max(0.05, sigma * 0.3), sigma * 2.0, 40)
        sens_vol = [{"vol": _fin(float(v), 4), "price": _fin(bs_price(S, K, T, float(v)), 6)} for v in vol_grid]
        t_grid = np.linspace(max(T * 0.02, 1 / 365), T * 1.5, 40)
        sens_time = [{"t": _fin(float(tt), 4), "price": _fin(bs_price(S, K, float(tt), sigma), 6)} for tt in t_grid]

        # ---- plot: value vs spot, vs volatility, vs time ----
        plot = None
        try:
            fig, (ax, axv, axt) = plt.subplots(1, 3, figsize=(15.5, 5), dpi=118)
            srange = np.linspace(max(0.3 * S, 0.01), 1.7 * S, 120)
            payoffs = np.maximum(srange - K, 0) if is_call else np.maximum(K - srange, 0)
            values = [bs_price(float(s), K, T, sigma) for s in srange]
            ax.plot(srange, payoffs, "--", color="#94a3b8", label="Payoff at expiry")
            ax.plot(srange, values, color="#2563eb", lw=2, label=f"Value now (T={T:g}y)")
            ax.axvline(K, color="#f59e0b", ls=":", lw=1, label=f"Strike {K:g}")
            ax.axvline(S, color="#dc2626", ls=":", lw=1, label=f"Spot {S:g}")
            ax.scatter([S], [price], color="#dc2626", zorder=5, s=50)
            ax.set_xlabel("Underlying price"); ax.set_ylabel("Option value")
            ax.set_title("1. Value vs underlying (S)")
            ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.2)
            # vs volatility
            axv.plot([s["vol"] for s in sens_vol], [s["price"] for s in sens_vol], color="#0891b2", lw=2)
            axv.axvline(sigma, color="#dc2626", ls=":", lw=1, label=f"σ = {sigma:.0%}")
            axv.scatter([sigma], [price], color="#dc2626", zorder=5, s=45)
            axv.set_xlabel("Volatility (σ)"); axv.set_ylabel("Option value")
            axv.set_title("2. Value vs volatility (σ)"); axv.legend(fontsize=8, frameon=False); axv.grid(alpha=0.2)
            # vs time
            axt.plot([s["t"] for s in sens_time], [s["price"] for s in sens_time], color="#9333ea", lw=2)
            axt.axvline(T, color="#dc2626", ls=":", lw=1, label=f"T = {T:g}y")
            axt.scatter([T], [price], color="#dc2626", zorder=5, s=45)
            axt.set_xlabel("Time to expiry (years)"); axt.set_ylabel("Option value")
            axt.set_title("3. Value vs time (T)"); axt.legend(fontsize=8, frameon=False); axt.grid(alpha=0.2)
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
            "moneyness": _fin(moneyness, 4), "state": state, "break_even": _fin(break_even, 6),
            "call_price": _fin(call_price, 6), "put_price": _fin(put_price, 6),
            "delta": _fin(delta, 6), "gamma": _fin(gamma, 6),
            "vega": _fin(vega, 6), "theta": _fin(theta_day, 6), "rho": _fin(rho, 6),
            "vega_per_100": _fin(vega_raw, 6), "theta_per_year": _fin(theta_raw, 6),
            "d1": _fin(d1, 4), "d2": _fin(d2, 4),
            "nd1": _fin(Nd1, 4), "nd2": _fin(Nd2, 4),
            "sensitivity_vol": sens_vol, "sensitivity_time": sens_time,
            "interpretation": interpretation,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
