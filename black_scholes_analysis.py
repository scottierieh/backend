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
    market_price     : float  optional observed market price -> implied volatility (section 6)
Output: { results: {...an 8-section report...}, plot, results.charts: {...} }

Eight-section report:
    1. Pricing Summary (call & put both, always computed off the same d1/d2)
    2. Model Inputs (restatement)
    3. Black-Scholes Calculation (d1, d2, N(d1), N(d2), N(-d1), N(-d2) exposed explicitly)
    4. Price Sensitivity (5 sweeps: Spot / Strike / Volatility / Maturity / Rate)
    5. Call vs Put (across spot range, reusing the spot sweep)
    6. Implied Volatility (optional, needs market_price)
    7. Put-Call Parity (formula-consistency check, not a real arbitrage detector)
    8. Model Assumptions (static reference table + limitations note)
"""
import sys, json, io, base64
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

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


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _bs_price(S, K, T, sigma, r, q, is_call):
    """Closed-form Black-Scholes-Merton price (used for sweeps / implied vol, no QuantLib object churn)."""
    if T <= 0 or sigma <= 0:
        intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
        return intrinsic
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if is_call:
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def main():
    try:
        p = json.load(sys.stdin)
        otype = (p.get("option_type") or "call").lower()
        is_call = otype != "put"
        S = float(p.get("spot")); K = float(p.get("strike"))
        T = float(p.get("expiry_years")); sigma = float(p.get("volatility"))
        r = float(p.get("risk_free_rate") or 0.0); q = float(p.get("dividend_yield") or 0.0)
        market_price = p.get("market_price")
        market_price = float(market_price) if market_price not in (None, "") else None
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            raise ValueError("Spot, strike, expiry and volatility must all be positive.")

        # ---- QuantLib European option (analytic Black-Scholes-Merton) ----
        val_date = ql.Date(1, 1, 2022)
        ql.Settings.instance().evaluationDate = val_date
        dc = ql.Actual365Fixed(); cal = ql.NullCalendar()
        maturity = val_date + int(round(T * 365))
        payoff = ql.PlainVanillaPayoff(ql.Option.Call if is_call else ql.Option.Put, K)
        payoff_call = ql.PlainVanillaPayoff(ql.Option.Call, K)
        payoff_put = ql.PlainVanillaPayoff(ql.Option.Put, K)
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

        # ---- d1, d2, N(d1), N(d2) — section 3, the formula-transparency differentiator ----
        sqrtT = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT
        Nd1, Nd2 = float(norm.cdf(d1)), float(norm.cdf(d2))
        Nmd1, Nmd2 = float(norm.cdf(-d1)), float(norm.cdf(-d2))

        # ---- Call & Put both, off the closed-form formula using this SAME d1/d2 ----
        # (QuantLib's engine rounds T to whole days internally for its maturity Date, which would
        #  otherwise introduce a tiny day-count mismatch vs the d1/d2 shown above; using the closed
        #  form here keeps sections 1/3/5/7 exactly self-consistent with the exposed d1/d2.)
        call_price = float(S * np.exp(-q * T) * Nd1 - K * np.exp(-r * T) * Nd2)
        put_price = float(K * np.exp(-r * T) * Nmd2 - S * np.exp(-q * T) * Nmd1)
        price = call_price if is_call else put_price

        intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
        time_value = price - intrinsic
        moneyness = S / K
        break_even = K + price if is_call else K - price
        state = ("in-the-money" if intrinsic > 0 else "out-of-the-money" if
                 (is_call and S < K) or (not is_call and S > K) else "at-the-money")

        # ================= section 4: five-way sensitivity sweeps =================
        charts = {}

        # -- vs Spot (also feeds section 5 Call vs Put, and the payoff panel) --
        srange = np.linspace(max(0.3 * S, 0.01), 1.7 * S, 120)
        payoffs = np.maximum(srange - K, 0) if is_call else np.maximum(K - srange, 0)
        values_spot, call_vs_spot, put_vs_spot = [], [], []
        for s in srange:
            spot_h2 = ql.QuoteHandle(ql.SimpleQuote(float(s)))
            proc2 = ql.BlackScholesMertonProcess(spot_h2, div_ts, rf_ts, vol_ts)
            o2 = ql.VanillaOption(payoff, exercise); o2.setPricingEngine(ql.AnalyticEuropeanEngine(proc2))
            values_spot.append(float(o2.NPV()))
            oc = ql.VanillaOption(payoff_call, exercise); oc.setPricingEngine(ql.AnalyticEuropeanEngine(proc2))
            op = ql.VanillaOption(payoff_put, exercise); op.setPricingEngine(ql.AnalyticEuropeanEngine(proc2))
            call_vs_spot.append(float(oc.NPV())); put_vs_spot.append(float(op.NPV()))
        sensitivity_spot = [{"spot": _fin(s, 4), "price": _fin(v, 6), "call": _fin(c, 6), "put": _fin(pt, 6)}
                             for s, v, c, pt in zip(srange, values_spot, call_vs_spot, put_vs_spot)]

        # -- vs Strike (new) --
        k_range = np.linspace(max(0.7 * K, 0.01), 1.3 * K, 60)
        sensitivity_strike = []
        for kk in k_range:
            pay_k = ql.PlainVanillaPayoff(ql.Option.Call if is_call else ql.Option.Put, float(kk))
            o_k = ql.VanillaOption(pay_k, exercise); o_k.setPricingEngine(ql.AnalyticEuropeanEngine(process))
            sensitivity_strike.append({"strike": _fin(kk, 4), "price": _fin(o_k.NPV(), 6)})

        # -- vs Volatility (existing from prior session) --
        vol_range = np.linspace(max(0.5 * sigma, 0.01), 2.0 * sigma, 25)
        sensitivity_vol = []
        for v in vol_range:
            vol_ts_v = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(val_date, cal, float(v), dc))
            proc_v = ql.BlackScholesMertonProcess(spot_h, div_ts, rf_ts, vol_ts_v)
            o_v = ql.VanillaOption(payoff, exercise); o_v.setPricingEngine(ql.AnalyticEuropeanEngine(proc_v))
            sensitivity_vol.append({"vol": _fin(v, 6), "price": _fin(o_v.NPV(), 6)})

        # -- vs Time to Maturity (existing from prior session) --
        t_range = np.linspace(max(T * 0.02, 1.0 / 365.0), T, 25)
        sensitivity_time = []
        for tt in t_range:
            maturity_t = val_date + int(round(float(tt) * 365))
            exercise_t = ql.EuropeanExercise(maturity_t)
            proc_t = ql.BlackScholesMertonProcess(spot_h, div_ts, rf_ts, vol_ts)
            o_t = ql.VanillaOption(payoff, exercise_t); o_t.setPricingEngine(ql.AnalyticEuropeanEngine(proc_t))
            sensitivity_time.append({"t": _fin(tt, 6), "price": _fin(o_t.NPV(), 6)})

        # -- vs Risk-Free Rate (new) --
        r_range = np.linspace(0.0, max(0.10, r * 2.0), 25)
        sensitivity_rate = []
        for rr in r_range:
            rf_ts_r = ql.YieldTermStructureHandle(ql.FlatForward(val_date, float(rr), dc))
            proc_r = ql.BlackScholesMertonProcess(spot_h, div_ts, rf_ts_r, vol_ts)
            o_r = ql.VanillaOption(payoff, exercise); o_r.setPricingEngine(ql.AnalyticEuropeanEngine(proc_r))
            sensitivity_rate.append({"rate": _fin(rr, 6), "price": _fin(o_r.NPV(), 6)})

        # ================= section 6: implied volatility (optional) =================
        implied_vol = None
        iv_theoretical_price = None
        iv_round_trip = None
        if market_price is not None and market_price > 0:
            def f(vol_guess):
                return _bs_price(S, K, T, vol_guess, r, q, is_call) - market_price
            lo, hi = 1e-4, 3.0
            try:
                flo, fhi = f(lo), f(hi)
                if flo * fhi > 0:
                    implied_vol = None
                else:
                    implied_vol = float(brentq(f, lo, hi, xtol=1e-8, maxiter=200))
            except Exception:
                implied_vol = None
            iv_theoretical_price = price  # BS price at the *input* (historical) volatility
            if implied_vol is not None:
                iv_round_trip = _bs_price(S, K, T, implied_vol, r, q, is_call)

        # ================= section 7: put-call parity =================
        pv_strike = K * np.exp(-r * T)
        pv_spot_div = S * np.exp(-q * T)
        parity_lhs = call_price - put_price
        parity_rhs = pv_spot_div - pv_strike
        parity_diff = parity_lhs - parity_rhs
        arbitrage_signal = "No" if abs(parity_diff) < 1e-6 else "Yes"

        # ================= section 8: model assumptions (static) =================
        model_assumptions = [
            {"assumption": "European Exercise", "holds": "Yes", "note": "Can only be exercised at expiry, not earlier."},
            {"assumption": "Constant Volatility", "holds": "Yes", "note": "Sigma is fixed for the option's life; no smile/skew."},
            {"assumption": "Constant Risk-Free Rate", "holds": "Yes", "note": "A single flat rate applies over the whole horizon."},
            {"assumption": "Log-Normal Price Distribution", "holds": "Yes", "note": "The underlying follows geometric Brownian motion — no jumps."},
            {"assumption": "No Transaction Costs", "holds": "Yes", "note": "Frictionless, continuous hedging is assumed."},
            {"assumption": "Continuous Trading", "holds": "Yes", "note": "The underlying can be traded and hedged at any instant."},
        ]
        model_limitations = (
            "Black-Scholes assumes European exercise and cannot directly price American options with "
            "early-exercise value; it assumes constant volatility and cannot capture volatility smile/skew "
            "or sudden market shocks; and dividend treatment is simplified to a continuous yield."
        )

        # ---- charts: separate PNGs per sweep, tabbed on the frontend ----
        try:
            fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=118)
            ax.plot(srange, payoffs, "--", color="#94a3b8", label="Payoff at expiry")
            ax.plot(srange, values_spot, color="#2563eb", lw=2, label=f"Value now (T={T:g}y)")
            ax.axvline(K, color="#f59e0b", ls=":", lw=1, label=f"Strike {K:g}")
            ax.axvline(S, color="#dc2626", ls=":", lw=1, label=f"Spot {S:g}")
            ax.scatter([S], [price], color="#dc2626", zorder=5, s=50)
            ax.set_xlabel("Underlying price"); ax.set_ylabel("Option value")
            ax.set_title(f"{otype.capitalize()} option value vs underlying")
            ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.2)
            fig.tight_layout(); charts["vs_spot"] = _png(fig)
        except Exception:
            plt.close("all")

        try:
            fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=118)
            ax.plot(k_range, [row["price"] for row in sensitivity_strike], color="#ea580c", lw=2)
            ax.axvline(K, color="#111827", ls="--", lw=1, label=f"Input K {K:g}")
            ax.scatter([K], [price], color="#ea580c", zorder=5, s=50, edgecolor="white")
            ax.set_xlabel("Strike price"); ax.set_ylabel("Option value")
            ax.set_title(f"{otype.capitalize()} value vs strike")
            ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.2)
            fig.tight_layout(); charts["vs_strike"] = _png(fig)
        except Exception:
            plt.close("all")

        try:
            fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=118)
            ax.plot(vol_range, [row["price"] for row in sensitivity_vol], color="#16a34a", lw=2)
            ax.axvline(sigma, color="#111827", ls="--", lw=1, label=f"Input σ {sigma:g}")
            ax.scatter([sigma], [price], color="#16a34a", zorder=5, s=50, edgecolor="white")
            if market_price is not None and implied_vol is not None:
                ax.axhline(market_price, color="#dc2626", ls=":", lw=1, label=f"Market price {market_price:g}")
                ax.scatter([implied_vol], [market_price], color="#dc2626", zorder=6, s=70, marker="D", edgecolor="white", label=f"Implied σ {implied_vol:.4f}")
            ax.set_xlabel("Volatility (σ)"); ax.set_ylabel("Option value")
            ax.set_title("Price vs Volatility")
            ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.2)
            fig.tight_layout()
            charts["vs_vol"] = _png(fig)
            # section 6 reuses this same vol-sweep chart (identical content, separate tab key)
            if market_price is not None:
                charts["implied_vol"] = charts["vs_vol"]
        except Exception:
            plt.close("all")

        try:
            fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=118)
            ax.plot(t_range, [row["price"] for row in sensitivity_time], color="#9333ea", lw=2)
            ax.axvline(T, color="#111827", ls="--", lw=1, label=f"Input T {T:g}y")
            ax.scatter([T], [price], color="#9333ea", zorder=5, s=50, edgecolor="white")
            ax.set_xlabel("Time to maturity (years)"); ax.set_ylabel("Option value")
            ax.set_title("Price vs Time to Maturity")
            ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.2)
            fig.tight_layout(); charts["vs_maturity"] = _png(fig)
        except Exception:
            plt.close("all")

        try:
            fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=118)
            ax.plot(r_range, [row["price"] for row in sensitivity_rate], color="#0891b2", lw=2)
            ax.axvline(r, color="#111827", ls="--", lw=1, label=f"Input r {r:g}")
            ax.scatter([r], [price], color="#0891b2", zorder=5, s=50, edgecolor="white")
            ax.set_xlabel("Risk-free rate (r)"); ax.set_ylabel("Option value")
            ax.set_title("Price vs Risk-Free Rate")
            ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.2)
            fig.tight_layout(); charts["vs_rate"] = _png(fig)
        except Exception:
            plt.close("all")

        try:
            fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=118)
            ax.plot(srange, call_vs_spot, color="#2563eb", lw=2, label="Call")
            ax.plot(srange, put_vs_spot, color="#dc2626", lw=2, label="Put")
            ax.axvline(K, color="#f59e0b", ls=":", lw=1, label=f"Strike {K:g}")
            ax.axvline(S, color="#6b7280", ls=":", lw=1, label=f"Spot {S:g}")
            ax.set_xlabel("Underlying price"); ax.set_ylabel("Option value")
            ax.set_title("Call vs Put value across the underlying")
            ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.2)
            fig.tight_layout(); charts["call_vs_put"] = _png(fig)
        except Exception:
            plt.close("all")

        # legacy combined panel (kept for backward compatibility with any existing consumer of `plot`)
        plot = None
        try:
            fig, axes = plt.subplots(2, 2, figsize=(13, 10), dpi=118)
            ax, ax_vol, ax_time, ax_spare = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]
            ax.plot(srange, payoffs, "--", color="#94a3b8", label="Payoff at expiry")
            ax.plot(srange, values_spot, color="#2563eb", lw=2, label=f"Value now (T={T:g}y)")
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

        # ---- representative rows for the five sweep tables (5-7 points each) ----
        def _sample(rows, key, n=7):
            if not rows:
                return []
            idx = np.linspace(0, len(rows) - 1, min(n, len(rows))).round().astype(int)
            idx = sorted(set(int(i) for i in idx))
            return [rows[i] for i in idx]

        pricing_summary = [
            {"metric": "Option Type", "value": otype},
            {"metric": "Spot", "value": _fin(S, 4)},
            {"metric": "Strike", "value": _fin(K, 4)},
            {"metric": "Time to Maturity (years)", "value": _fin(T, 4)},
            {"metric": "Risk-Free Rate", "value": _fin(r, 6)},
            {"metric": "Volatility", "value": _fin(sigma, 6)},
            {"metric": "Dividend Yield", "value": _fin(q, 6)},
            {"metric": "Call Value", "value": _fin(call_price, 6)},
            {"metric": "Put Value", "value": _fin(put_price, 6)},
        ]

        bs_calculation = [
            {"quantity": "d1", "value": _fin(d1, 6)},
            {"quantity": "d2", "value": _fin(d2, 6)},
            {"quantity": "N(d1)", "value": _fin(Nd1, 6)},
            {"quantity": "N(d2)", "value": _fin(Nd2, 6)},
            {"quantity": "N(-d1)", "value": _fin(Nmd1, 6)},
            {"quantity": "N(-d2)", "value": _fin(Nmd2, 6)},
        ]

        call_vs_put_table = [
            {"spot": _fin(row["spot"], 4), "call": row["call"], "put": row["put"]}
            for row in _sample(sensitivity_spot, "spot")
        ]

        put_call_parity = [
            {"metric": "Call Price", "value": _fin(call_price, 6)},
            {"metric": "Put Price", "value": _fin(put_price, 6)},
            {"metric": "Spot Price", "value": _fin(S, 4)},
            {"metric": "Strike Price", "value": _fin(K, 4)},
            {"metric": "PV(Strike), dividend-adjusted", "value": _fin(pv_strike, 6)},
            {"metric": "C - P", "value": _fin(parity_lhs, 6)},
            {"metric": "S*e^(-qT) - K*e^(-rT)", "value": _fin(parity_rhs, 6)},
            {"metric": "Parity Difference", "value": _fin(parity_diff, 8)},
            {"metric": "Arbitrage Signal", "value": arbitrage_signal},
        ]

        implied_vol_table = None
        if market_price is not None:
            implied_vol_table = [
                {"metric": "Market Price", "value": _fin(market_price, 6)},
                {"metric": "BS Theoretical Price (input vol)", "value": _fin(iv_theoretical_price, 6)},
                {"metric": "Implied Volatility", "value": _fin(implied_vol, 6) if implied_vol is not None else None},
                {"metric": "Input (Historical) Volatility", "value": _fin(sigma, 6)},
                {"metric": "Round-trip BS Price at Implied Vol", "value": _fin(iv_round_trip, 6) if iv_round_trip is not None else None},
            ]

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
            "d1": _fin(d1, 6), "d2": _fin(d2, 6),
            "nd1": _fin(Nd1, 6), "nd2": _fin(Nd2, 6), "nmd1": _fin(Nmd1, 6), "nmd2": _fin(Nmd2, 6),
            # section 1
            "pricing_summary": pricing_summary,
            # section 3
            "bs_calculation": bs_calculation,
            # section 4 — five sweeps (full arrays kept for charting/back-compat, tables below sampled)
            "sensitivity_spot": sensitivity_spot,
            "sensitivity_strike": sensitivity_strike,
            "sensitivity_vol": sensitivity_vol,
            "sensitivity_time": sensitivity_time,
            "sensitivity_rate": sensitivity_rate,
            "sensitivity_spot_table": _sample(sensitivity_spot, "spot"),
            "sensitivity_strike_table": _sample(sensitivity_strike, "strike"),
            "sensitivity_vol_table": _sample(sensitivity_vol, "vol"),
            "sensitivity_time_table": _sample(sensitivity_time, "t"),
            "sensitivity_rate_table": _sample(sensitivity_rate, "rate"),
            # section 5
            "call_vs_put_table": call_vs_put_table,
            # section 6 (optional)
            "market_price": _fin(market_price, 6) if market_price is not None else None,
            "implied_volatility": _fin(implied_vol, 6) if implied_vol is not None else None,
            "implied_vol_theoretical_price": _fin(iv_theoretical_price, 6) if iv_theoretical_price is not None else None,
            "implied_vol_round_trip_price": _fin(iv_round_trip, 6) if iv_round_trip is not None else None,
            "implied_vol_table": implied_vol_table,
            # section 7
            "put_call_parity": put_call_parity,
            "parity_diff": _fin(parity_diff, 8), "arbitrage_signal": arbitrage_signal,
            # section 8
            "model_assumptions": model_assumptions,
            "model_limitations": model_limitations,
            "charts": charts,
            "interpretation": interpretation,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
