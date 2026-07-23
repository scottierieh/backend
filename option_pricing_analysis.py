#!/usr/bin/env python3
"""Option Pricing — European & American options by multiple methods. QuantLib.

Compares the analytic Black-Scholes price (European), the binomial tree
(European & American), Monte Carlo (European), and a Bachelier (normal-model)
price, shows the binomial convergence, isolates the early-exercise premium of
American options, and builds a full step-6 report with 7 additive sections:

  1. Pricing Summary            — the core contract + theoretical/intrinsic/time value
  2. Model Comparison            — Black-Scholes / Binomial / Monte Carlo / Bachelier, Call & Put
  3. Payoff Analysis              — long option P&L at expiry, break-even, max profit/loss
  4. Intrinsic vs Time Value      — decomposition (reuses section 1's numbers)
  5. Price Sensitivity            — Price vs Spot / Vol / Time sweeps (Call & Put)
  6. Put-Call Parity              — consistency check C - P vs S - K*e^-rT
  7. Implied Volatility (optional) — only if a market_price is supplied

Input (from option-pricing-page.tsx):
    option_type    : "call" | "put"
    exercise       : "european" | "american"
    spot, strike, expiry_years, volatility, risk_free_rate, dividend_yield
    steps          : int  binomial steps (default 200)
    mc_paths       : int  Monte Carlo paths (default 50000)
    market_price   : float, OPTIONAL — observed option price, solves implied vol
Output: { results: {..., charts: {model_comparison, payoff, price_vs_spot,
          price_vs_vol, price_vs_time, implied_vol?}}, plot }
"""
import sys, json, io, base64
import numpy as np
from scipy.stats import norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import QuantLib as ql

BLUE = "#2563eb"
GREEN = "#16a34a"
AMBER = "#f59e0b"
PURPLE = "#9333ea"
RED = "#dc2626"
CYAN = "#0891b2"
GRAY = "#94a3b8"


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


def _bachelier_price(S, K, T, sigma_n, r, is_call):
    """Bachelier (normal) model price, discounted variant:
        d = (S - K) / (sigma_n * sqrt(T))
        C = e^{-rT} [ (S-K)*N(d) + sigma_n*sqrt(T)*phi(d) ]
        P = e^{-rT} [ (K-S)*N(-d) + sigma_n*sqrt(T)*phi(d) ]
    Here sigma_n is treated as an absolute (price-unit) volatility; for
    comparability with the Black-Scholes lognormal input we approximate it
    as sigma_n = sigma * S (the usual local scaling at the current spot),
    which is the standard way to put a lognormal vol on the same footing as
    Bachelier's normal-model vol for a rough cross-model comparison.
    """
    sigma_abs = sigma_n * S
    if sigma_abs <= 0 or T <= 0:
        intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
        return intrinsic
    d = (S - K) / (sigma_abs * np.sqrt(T))
    disc = np.exp(-r * T)
    if is_call:
        return disc * ((S - K) * norm.cdf(d) + sigma_abs * np.sqrt(T) * norm.pdf(d))
    else:
        return disc * ((K - S) * norm.cdf(-d) + sigma_abs * np.sqrt(T) * norm.pdf(d))


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
        market_price_raw = p.get("market_price")
        market_price = float(market_price_raw) if market_price_raw not in (None, "") else None
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

        def analytic_price_for(_S, _K, _T, _sigma, _r, _q, _is_call):
            """Standalone Black-Scholes analytic price (for use in sweeps/solver)."""
            _maturity = val_date + max(1, int(round(_T * 365)))
            _payoff = ql.PlainVanillaPayoff(ql.Option.Call if _is_call else ql.Option.Put, _K)
            _exercise = ql.EuropeanExercise(_maturity)
            _spot_h = ql.QuoteHandle(ql.SimpleQuote(float(_S)))
            _rf_ts = ql.YieldTermStructureHandle(ql.FlatForward(val_date, _r, dc))
            _div_ts = ql.YieldTermStructureHandle(ql.FlatForward(val_date, _q, dc))
            _vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(val_date, cal, max(_sigma, 1e-6), dc))
            _proc = ql.BlackScholesMertonProcess(_spot_h, _div_ts, _rf_ts, _vol_ts)
            _opt = ql.VanillaOption(_payoff, _exercise)
            _opt.setPricingEngine(ql.AnalyticEuropeanEngine(_proc))
            return float(_opt.NPV())

        methods = {}

        # analytic (European only)
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

        bachelier_price_chosen = _bachelier_price(S, K, T, sigma, r, is_call)
        methods["bachelier"] = _fin(bachelier_price_chosen, 6)

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

        theoretical_price = binomial_price
        intrinsic = max(S - K, 0.0) if is_call else max(K - S, 0.0)
        time_value = theoretical_price - intrinsic
        moneyness = S / K

        # ---------------------------------------------------------------
        # ② Model Comparison — price BOTH call and put under each model
        # ---------------------------------------------------------------
        def price_both(sigma_use=sigma, r_use=r, T_use=T, S_use=S, K_use=K):
            out = {}
            _maturity = val_date + max(1, int(round(T_use * 365)))
            _rf = ql.YieldTermStructureHandle(ql.FlatForward(val_date, r_use, dc))
            _div = ql.YieldTermStructureHandle(ql.FlatForward(val_date, q, dc))
            _vol = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(val_date, cal, max(sigma_use, 1e-6), dc))
            _spot = ql.QuoteHandle(ql.SimpleQuote(float(S_use)))
            _proc = ql.BlackScholesMertonProcess(_spot, _div, _rf, _vol)
            _euro = ql.EuropeanExercise(_maturity)
            for kind, cflag in (("call", ql.Option.Call), ("put", ql.Option.Put)):
                pf = ql.PlainVanillaPayoff(cflag, K_use)
                o = ql.VanillaOption(pf, _euro)
                o.setPricingEngine(ql.AnalyticEuropeanEngine(_proc))
                out[kind] = float(o.NPV())
            return out

        both_analytic = price_both()

        def binomial_both(steps_use=steps):
            out = {}
            for kind, cflag in (("call", ql.Option.Call), ("put", ql.Option.Put)):
                pf = ql.PlainVanillaPayoff(cflag, K)
                o = ql.VanillaOption(pf, exercise)
                o.setPricingEngine(ql.BinomialVanillaEngine(process, "crr", steps_use))
                out[kind] = float(o.NPV())
            return out
        both_binomial = binomial_both()

        both_mc = {}
        for kind, cflag in (("call", ql.Option.Call), ("put", ql.Option.Put)):
            pf = ql.PlainVanillaPayoff(cflag, K)
            mc_eng = ql.MCEuropeanEngine(process, "pseudorandom", timeSteps=1,
                                         requiredSamples=min(mc_paths, 100000), seed=42)
            o = ql.VanillaOption(pf, euro_exercise)
            o.setPricingEngine(mc_eng)
            both_mc[kind] = float(o.NPV())

        both_bachelier = {
            "call": _bachelier_price(S, K, T, sigma, r, True),
            "put": _bachelier_price(S, K, T, sigma, r, False),
        }

        model_comparison = []
        baseline_call, baseline_put = both_analytic["call"], both_analytic["put"]
        for model_key, label_en, label_ko, prices in (
            ("black_scholes", "Black-Scholes (Analytic)", "블랙-숄즈 (해석해)", both_analytic),
            ("binomial", f"Binomial Tree ({steps} steps)", f"이항트리 ({steps}단계)", both_binomial),
            ("monte_carlo", "Monte Carlo", "몬테카를로", both_mc),
            ("bachelier", "Bachelier (Normal Model)", "바슐리에 (정규모형)", both_bachelier),
        ):
            model_comparison.append({
                "model": model_key, "label_en": label_en, "label_ko": label_ko,
                "call_price": _fin(prices["call"], 6), "put_price": _fin(prices["put"], 6),
                "difference": _fin((prices["call"] - prices["put"]) - (baseline_call - baseline_put), 6),
            })

        # ---------------------------------------------------------------
        # ③ Payoff Analysis — long option P&L at expiry
        # ---------------------------------------------------------------
        premium = theoretical_price
        break_even = K + premium if is_call else K - premium
        max_loss = premium
        max_profit = "unlimited" if is_call else _fin(K - premium, 4)

        s_grid = np.linspace(max(0.2 * S, 0.01), 1.8 * S, 160)
        payoff_at_expiry = np.maximum(s_grid - K, 0) if is_call else np.maximum(K - s_grid, 0)
        pnl_at_expiry = payoff_at_expiry - premium

        payoff_table = [
            {"strike": _fin(K, 4), "premium": _fin(premium, 4), "break_even": _fin(break_even, 4),
             "max_loss": _fin(max_loss, 4), "max_profit": max_profit, "option_type": otype},
        ]

        # ---------------------------------------------------------------
        # ⑤ Price Sensitivity — Spot / Vol / Time sweeps (Call & Put)
        # ---------------------------------------------------------------
        spot_range = np.linspace(max(0.7 * S, 0.01), 1.3 * S, 25)
        sens_spot = []
        for s in spot_range:
            both = price_both(S_use=float(s))
            sens_spot.append({"spot": _fin(s, 4), "call_price": _fin(both["call"], 6), "put_price": _fin(both["put"], 6)})

        vol_range = np.linspace(max(0.5 * sigma, 0.01), 2.0 * sigma, 25)
        sens_vol = []
        for v in vol_range:
            both = price_both(sigma_use=float(v))
            sens_vol.append({"vol": _fin(v, 6), "call_price": _fin(both["call"], 6), "put_price": _fin(both["put"], 6)})

        t_range = np.linspace(max(T * 0.02, 1.0 / 365.0), T, 25)
        sens_time = []
        for tt in t_range:
            both = price_both(T_use=float(tt))
            sens_time.append({"t_years": _fin(tt, 6), "days": int(round(tt * 365)),
                               "call_price": _fin(both["call"], 6), "put_price": _fin(both["put"], 6)})

        # representative rows for the tables (5-7 points)
        rep_spot_idx = np.linspace(0, len(sens_spot) - 1, 7).round().astype(int)
        rep_vol_idx = np.linspace(0, len(sens_vol) - 1, 7).round().astype(int)
        rep_days = [7, 30, 90, 180, 365]
        rep_time_rows = []
        for d in rep_days:
            tt = d / 365.0
            if tt > T * 1.05:
                continue
            both = price_both(T_use=tt)
            rep_time_rows.append({"t_years": _fin(tt, 6), "days": d,
                                    "call_price": _fin(both["call"], 6), "put_price": _fin(both["put"], 6)})
        if not rep_time_rows:
            rep_time_rows = [sens_time[i] for i in np.linspace(0, len(sens_time) - 1, 5).round().astype(int)]

        sensitivity_spot_table = [sens_spot[i] for i in rep_spot_idx]
        sensitivity_vol_table = [sens_vol[i] for i in rep_vol_idx]
        sensitivity_time_table = rep_time_rows

        # ---------------------------------------------------------------
        # ④ Intrinsic Value vs Time Value — decomposition (reuse §1)
        # ---------------------------------------------------------------
        decomposition = {
            "option_price": _fin(theoretical_price, 6),
            "intrinsic_value": _fin(intrinsic, 6),
            "time_value": _fin(time_value, 6),
        }

        # ---------------------------------------------------------------
        # ⑥ Put-Call Parity — consistency check
        # ---------------------------------------------------------------
        pv_strike = K * np.exp(-r * T)
        s_adj = S * np.exp(-q * T)
        call_p = both_analytic["call"]; put_p = both_analytic["put"]
        parity_lhs = call_p - put_p
        parity_rhs = s_adj - pv_strike
        parity_diff = parity_lhs - parity_rhs
        tol = max(0.01, 0.001 * max(abs(parity_lhs), 1.0))
        arbitrage_signal = "Yes" if abs(parity_diff) > tol else "No"
        put_call_parity = {
            "call_price": _fin(call_p, 6), "put_price": _fin(put_p, 6),
            "spot_price": _fin(S, 4), "pv_strike": _fin(pv_strike, 6),
            "parity_difference": _fin(parity_diff, 8), "arbitrage_signal": arbitrage_signal,
            "tolerance": _fin(tol, 6),
            "note": ("This checks internal consistency of THIS model's own Call and Put prices "
                     "(both from the same Black-Scholes analytic engine), not real-market arbitrage — "
                     "a genuine mismatch would only arise if Call and Put came from different models "
                     "or market quotes."),
        }

        # ---------------------------------------------------------------
        # ⑦ Implied Volatility — optional, only if market_price given
        # ---------------------------------------------------------------
        implied_vol_result = None
        if market_price is not None and market_price > 0:
            lo, hi = 1e-4, 3.0

            def f(vol_guess):
                return analytic_price_for(S, K, T, vol_guess, r, q, is_call) - market_price

            flo, fhi = f(lo), f(hi)
            solved_vol = None
            if flo * fhi <= 0:
                a, b = lo, hi
                fa = flo
                for _ in range(100):
                    mid = 0.5 * (a + b)
                    fm = f(mid)
                    if abs(fm) < 1e-8 or (b - a) < 1e-8:
                        solved_vol = mid
                        break
                    if fa * fm <= 0:
                        b = mid
                    else:
                        a = mid; fa = fm
                if solved_vol is None:
                    solved_vol = 0.5 * (a + b)
            model_price_at_input_vol = analytic_price_for(S, K, T, sigma, r, q, is_call)
            implied_vol_result = {
                "market_price": _fin(market_price, 6),
                "model_price": _fin(model_price_at_input_vol, 6),
                "implied_volatility": _fin(solved_vol, 6) if solved_vol is not None else None,
                "historical_volatility": _fin(sigma, 6),
                "reproduced_price": _fin(analytic_price_for(S, K, T, solved_vol, r, q, is_call), 6) if solved_vol is not None else None,
                "solved": solved_vol is not None,
            }

        # ---------------------------------------------------------------
        # Charts (separate PNGs, per VisualizationTabs pattern)
        # ---------------------------------------------------------------
        charts = {}

        # model_comparison: bar chart Call vs Put per model
        try:
            fig, ax = plt.subplots(figsize=(9, 5.5), dpi=120)
            labels = [row["label_en"] for row in model_comparison]
            x = np.arange(len(model_comparison))
            w = 0.35
            calls = [row["call_price"] for row in model_comparison]
            puts = [row["put_price"] for row in model_comparison]
            ax.bar(x - w / 2, calls, w, label="Call", color=BLUE)
            ax.bar(x + w / 2, puts, w, label="Put", color=AMBER)
            ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
            ax.set_ylabel("Option price"); ax.set_title("Model Comparison — Call vs Put")
            ax.legend(frameon=False); ax.grid(alpha=0.2, axis="y")
            fig.tight_layout()
            charts["model_comparison"] = _png(fig)
        except Exception:
            plt.close("all"); charts["model_comparison"] = None

        # payoff: hockey-stick P&L chart
        try:
            fig, ax = plt.subplots(figsize=(8.5, 5.5), dpi=120)
            ax.plot(s_grid, pnl_at_expiry, color=BLUE, lw=2.2, label=f"Long {otype} P&L")
            ax.axhline(0, color="#111827", lw=1)
            ax.axvline(K, color=GRAY, ls=":", lw=1.2, label=f"Strike {K:g}")
            ax.axvline(break_even, color=GREEN, ls="--", lw=1.2, label=f"Break-even {break_even:.2f}")
            ax.fill_between(s_grid, pnl_at_expiry, 0, where=(pnl_at_expiry >= 0), color=GREEN, alpha=0.15)
            ax.fill_between(s_grid, pnl_at_expiry, 0, where=(pnl_at_expiry < 0), color=RED, alpha=0.15)
            ax.set_xlabel("Underlying price at expiry"); ax.set_ylabel("Profit / Loss")
            ax.set_title(f"Payoff Diagram — Long {otype.capitalize()} (premium {premium:.4f})")
            ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.2)
            fig.tight_layout()
            charts["payoff"] = _png(fig)
        except Exception:
            plt.close("all"); charts["payoff"] = None

        # price_vs_spot
        try:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=118)
            ax.plot(spot_range, [row["call_price"] for row in sens_spot], color=BLUE, lw=2, label="Call")
            ax.plot(spot_range, [row["put_price"] for row in sens_spot], color=AMBER, lw=2, label="Put")
            ax.axvline(S, color="#111827", ls="--", lw=1, label=f"Spot {S:g}")
            ax.set_xlabel("Spot price"); ax.set_ylabel("Price"); ax.set_title("Price vs Underlying Price")
            ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.2)
            fig.tight_layout()
            charts["price_vs_spot"] = _png(fig)
        except Exception:
            plt.close("all"); charts["price_vs_spot"] = None

        # price_vs_vol
        try:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=118)
            ax.plot(vol_range, [row["call_price"] for row in sens_vol], color=BLUE, lw=2, label="Call")
            ax.plot(vol_range, [row["put_price"] for row in sens_vol], color=AMBER, lw=2, label="Put")
            ax.axvline(sigma, color="#111827", ls="--", lw=1, label=f"Input vol {sigma:g}")
            ax.set_xlabel("Volatility"); ax.set_ylabel("Price"); ax.set_title("Price vs Volatility")
            ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.2)
            fig.tight_layout()
            charts["price_vs_vol"] = _png(fig)
        except Exception:
            plt.close("all"); charts["price_vs_vol"] = None

        # price_vs_time
        try:
            fig, ax = plt.subplots(figsize=(8, 5), dpi=118)
            days_axis = [row["days"] for row in sens_time]
            ax.plot(days_axis, [row["call_price"] for row in sens_time], color=BLUE, lw=2, label="Call")
            ax.plot(days_axis, [row["put_price"] for row in sens_time], color=AMBER, lw=2, label="Put")
            ax.set_xlabel("Days to maturity"); ax.set_ylabel("Price"); ax.set_title("Price vs Time to Maturity")
            ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.2)
            fig.tight_layout()
            charts["price_vs_time"] = _png(fig)
        except Exception:
            plt.close("all"); charts["price_vs_time"] = None

        # implied_vol: 2-bar comparison (only if solved)
        if implied_vol_result and implied_vol_result.get("solved"):
            try:
                fig, ax = plt.subplots(figsize=(5.5, 5), dpi=120)
                vals = [implied_vol_result["implied_volatility"], implied_vol_result["historical_volatility"]]
                bars = ax.bar(["Implied", "Historical/Input"], vals, color=[PURPLE, CYAN])
                for b, v in zip(bars, vals):
                    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
                ax.set_ylabel("Volatility"); ax.set_title("Implied vs Historical Volatility")
                ax.grid(alpha=0.2, axis="y")
                fig.tight_layout()
                charts["implied_vol"] = _png(fig)
            except Exception:
                plt.close("all")

        # legacy combined plot (method comparison + convergence) — kept for backward compat
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
            cx = [c["steps"] for c in convergence]; cy = [c["price"] for c in convergence]
            ax2.plot(cx, cy, "o-", color="#16a34a", label="Binomial")
            if analytic_price is not None and not is_american:
                ax2.axhline(analytic_price, color="#2563eb", ls="--", lw=1.2, label="Analytic (exact)")
            ax2.set_xscale("log"); ax2.set_xlabel("Binomial steps (log)"); ax2.set_ylabel("Price")
            ax2.set_title("Binomial convergence"); ax2.legend(fontsize=8, frameon=False); ax2.grid(alpha=0.2)
            fig.tight_layout()
            plot = _png(fig)
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

            # ① Pricing Summary
            "theoretical_price": _fin(theoretical_price, 6),
            "pricing_summary": {
                "option_type": otype, "spot": _fin(S, 4), "strike": _fin(K, 4),
                "expiry_years": _fin(T, 4), "risk_free_rate": _fin(r, 6),
                "volatility": _fin(sigma, 6), "dividend_yield": _fin(q, 6),
                "theoretical_price": _fin(theoretical_price, 6),
                "intrinsic_value": _fin(intrinsic, 6), "time_value": _fin(time_value, 6),
            },

            # ② Model Comparison
            "model_comparison": model_comparison,

            # ③ Payoff Analysis
            "payoff_table": payoff_table,
            "break_even": _fin(break_even, 4), "premium": _fin(premium, 4),
            "max_loss": _fin(max_loss, 4), "max_profit": max_profit,

            # ④ Intrinsic vs Time Value
            "decomposition": decomposition,

            # ⑤ Price Sensitivity
            "sensitivity_spot_table": sensitivity_spot_table,
            "sensitivity_vol_table": sensitivity_vol_table,
            "sensitivity_time_table": sensitivity_time_table,
            "sensitivity_spot_full": sens_spot,
            "sensitivity_vol_full": sens_vol,
            "sensitivity_time_full": sens_time,

            # ⑥ Put-Call Parity
            "put_call_parity": put_call_parity,

            # ⑦ Implied Volatility (optional)
            "implied_volatility": implied_vol_result,

            "charts": charts,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
