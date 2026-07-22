#!/usr/bin/env python3
"""Option Pricing — European & American options by multiple methods. QuantLib.

Compares the analytic Black-Scholes price (European), the binomial tree
(European & American), and Monte Carlo (European), shows the binomial
convergence, and isolates the early-exercise premium of American options.

Input (from option-pricing-page.tsx):
    option_type    : "call" | "put"
    exercise       : "european" | "american"
    spot, strike, expiry_years, volatility, risk_free_rate, dividend_yield
    steps          : int  binomial steps (default 200)
    mc_paths       : int  Monte Carlo paths (default 50000)
Output: { results: {prices by method, convergence, early_exercise}, plot }
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
        exercise_style = (p.get("exercise") or "european").lower()
        is_american = exercise_style == "american"
        S = float(p.get("spot")); K = float(p.get("strike"))
        T = float(p.get("expiry_years")); sigma = float(p.get("volatility"))
        r = float(p.get("risk_free_rate") or 0.0); q = float(p.get("dividend_yield") or 0.0)
        steps = int(p.get("steps") or 200)
        mc_paths = int(p.get("mc_paths") or 50000)
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            raise ValueError("Spot, strike, expiry and volatility must all be positive.")
        steps = max(10, min(steps, 2000))
        mc_paths = max(1000, min(mc_paths, 500000))

        val_date = ql.Date(1, 1, 2022)
        ql.Settings.instance().evaluationDate = val_date
        dc = ql.Actual365Fixed(); cal = ql.NullCalendar()
        maturity = val_date + int(round(T * 365))
        payoff = ql.PlainVanillaPayoff(ql.Option.Call if is_call else ql.Option.Put, K)
        spot_h = ql.QuoteHandle(ql.SimpleQuote(S))
        rf_ts = ql.YieldTermStructureHandle(ql.FlatForward(val_date, r, dc))
        div_ts = ql.YieldTermStructureHandle(ql.FlatForward(val_date, q, dc))
        vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(val_date, cal, sigma, dc))
        process = ql.BlackScholesMertonProcess(spot_h, div_ts, rf_ts, vol_ts)

        euro_exercise = ql.EuropeanExercise(maturity)
        amer_exercise = ql.AmericanExercise(val_date, maturity)

        methods = {}

        # analytic (European only)
        analytic_price = None
        opt_euro = ql.VanillaOption(payoff, euro_exercise)
        opt_euro.setPricingEngine(ql.AnalyticEuropeanEngine(process))
        analytic_price = float(opt_euro.NPV())
        methods["analytic"] = _fin(analytic_price, 6)

        # binomial (CRR) for the requested exercise style
        exercise = amer_exercise if is_american else euro_exercise
        opt = ql.VanillaOption(payoff, exercise)
        opt.setPricingEngine(ql.BinomialVanillaEngine(process, "crr", steps))
        binomial_price = float(opt.NPV())
        methods["binomial"] = _fin(binomial_price, 6)

        # Monte Carlo (European only; American MC needs Longstaff-Schwartz)
        mc_price = None; mc_stderr = None
        if not is_american:
            mc_engine = ql.MCEuropeanEngine(process, "pseudorandom", timeSteps=1,
                                            requiredSamples=mc_paths, seed=42)
            opt_mc = ql.VanillaOption(payoff, euro_exercise)
            opt_mc.setPricingEngine(mc_engine)
            mc_price = float(opt_mc.NPV())
            try:
                mc_stderr = float(opt_mc.errorEstimate())
            except Exception:
                mc_stderr = None
            methods["monte_carlo"] = _fin(mc_price, 6)

        # binomial convergence across step counts
        step_grid = [5, 10, 25, 50, 100, 200, 400, 800]
        convergence = []
        for n in step_grid:
            o = ql.VanillaOption(payoff, exercise)
            o.setPricingEngine(ql.BinomialVanillaEngine(process, "crr", n))
            convergence.append({"steps": n, "price": _fin(float(o.NPV()), 6)})

        # early-exercise premium (American binomial - European binomial, same steps)
        early_premium = None
        if is_american:
            oe = ql.VanillaOption(payoff, euro_exercise)
            oe.setPricingEngine(ql.BinomialVanillaEngine(process, "crr", steps))
            early_premium = binomial_price - float(oe.NPV())

        intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
        chosen = binomial_price
        time_value = chosen - intrinsic
        moneyness = S / K

        # plot: method comparison + convergence
        plot = None
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5), dpi=120)
            labels, vals, colors = [], [], []
            if analytic_price is not None and not is_american:
                labels.append("Analytic"); vals.append(analytic_price); colors.append("#2563eb")
            labels.append(f"Binomial\n({steps} steps)"); vals.append(binomial_price); colors.append("#16a34a")
            if mc_price is not None:
                labels.append(f"Monte Carlo\n({mc_paths:,})"); vals.append(mc_price); colors.append("#f59e0b")
            bars = ax1.bar(labels, vals, color=colors)
            for b, v in zip(bars, vals):
                ax1.text(b.get_x() + b.get_width()/2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
            ax1.set_ylabel("Option price")
            ax1.set_title(f"{('American' if is_american else 'European')} {otype} — price by method")
            # convergence
            cx = [c["steps"] for c in convergence]; cy = [c["price"] for c in convergence]
            ax2.plot(cx, cy, "o-", color="#16a34a", label="Binomial")
            if analytic_price is not None and not is_american:
                ax2.axhline(analytic_price, color="#2563eb", ls="--", lw=1.2, label="Analytic (exact)")
            ax2.set_xscale("log"); ax2.set_xlabel("Binomial steps (log)"); ax2.set_ylabel("Price")
            ax2.set_title("Binomial convergence"); ax2.legend(fontsize=8, frameon=False); ax2.grid(alpha=0.2)
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        method_note = (
            "For a European option the analytic Black-Scholes formula is exact; the binomial tree and Monte Carlo "
            "converge to it as steps and paths grow, which the convergence chart confirms."
            if not is_american else
            "American options have no closed-form price, so the binomial tree (which checks early exercise at every "
            "node) is the workhorse method here; Monte Carlo would need Longstaff-Schwartz regression."
        )
        interpretation = (
            f"This {('American' if is_american else 'European')} {otype} option is priced at {binomial_price:,.4f} "
            f"by the {steps}-step binomial tree. " + method_note + " "
            + (f"The early-exercise premium — the extra value from being able to exercise before expiry — is "
               f"{early_premium:,.4f} ({100*early_premium/binomial_price:.1f}% of the price), which is why an "
               "American option is never worth less than its European twin. " if (early_premium is not None and binomial_price) else "")
            + f"Of the price, {intrinsic:,.2f} is intrinsic value and {time_value:,.2f} is time value."
        )

        results = {
            "status": "ok", "option_type": otype, "exercise": exercise_style,
            "spot": _fin(S, 4), "strike": _fin(K, 4), "expiry_years": _fin(T, 4),
            "volatility": _fin(sigma, 6), "risk_free_rate": _fin(r, 6), "dividend_yield": _fin(q, 6),
            "steps": steps, "mc_paths": mc_paths,
            "price": _fin(binomial_price, 6), "methods": methods,
            "analytic_price": _fin(analytic_price, 6),
            "monte_carlo_stderr": _fin(mc_stderr, 6) if mc_stderr is not None else None,
            "early_exercise_premium": _fin(early_premium, 6) if early_premium is not None else None,
            "intrinsic_value": _fin(intrinsic, 6), "time_value": _fin(time_value, 6),
            "moneyness": _fin(moneyness, 4),
            "convergence": convergence,
            "interpretation": interpretation,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
