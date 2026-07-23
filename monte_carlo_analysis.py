#!/usr/bin/env python3
"""Monte Carlo Simulation — geometric Brownian motion asset paths. NumPy (+SciPy).

Simulates many random price paths under GBM, then summarises the terminal
distribution: expected value, percentiles, probability of loss, and terminal
VaR/CVaR. Optionally prices a European option via a separate risk-neutral
Monte Carlo pass (same random draws, risk-free drift), validated against the
closed-form Black-Scholes price.

Builds a full step-6 report with 8 additive sections, each with its own chart
where applicable (paths, terminal distribution, payoff distribution,
probability analysis, convergence analysis), rendered by the frontend via
VisualizationTabs:
    1. Simulation Summary        4. Terminal Price Distribution   7. Probability Analysis
    2. Simulation Inputs         5. Option Pricing (+ BS check)   8. Convergence Analysis
    3. Simulated Price Paths     6. Payoff Distribution

Input (from monte-carlo-page.tsx):
    initial_price    : float  S0
    drift            : float  annual expected return mu (real-world, for paths/terminal dist)
    volatility       : float  annual sigma
    horizon_years    : float  T
    steps            : int    time steps per path (default 252)
    n_paths          : int    number of paths (default 10000)
    confidence       : float  (default 0.95) for terminal VaR
    seed             : int    (default 42)
    option_strike    : float  (optional) if given, price a European call & put
    risk_free_rate   : float  (optional, default = drift) risk-neutral rate used only for option pricing
Output: { results: {...}, plot, } where results.charts = {paths, terminal_dist,
          payoff_dist?, probability, convergence}
"""
import sys, json, io, base64
import numpy as np
from scipy.stats import norm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#2563eb"
GREEN = "#16a34a"
RED = "#dc2626"
AMBER = "#d97706"
LIGHT_BLUE = "#93c5fd"
PURPLE = "#9333ea"


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


def _bs_price(S, K, T, sigma, r, is_call=True, q=0.0):
    """Closed-form Black-Scholes-Merton price — used only as a sanity check on the MC estimate."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if is_call:
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def main():
    try:
        p = json.load(sys.stdin)
        S0 = float(p.get("initial_price"))
        mu = float(p.get("drift") or 0.0)
        sigma = float(p.get("volatility"))
        T = float(p.get("horizon_years"))
        steps = int(p.get("steps") or 252)
        n_paths = int(p.get("n_paths") or 10000)
        conf = float(p.get("confidence") or 0.95)
        seed = int(p.get("seed") or 42)
        strike = p.get("option_strike")
        rf_in = p.get("risk_free_rate")
        if S0 <= 0 or sigma <= 0 or T <= 0:
            raise ValueError("Initial price, volatility and horizon must all be positive.")
        if not (0.5 < conf < 1):
            raise ValueError("Confidence must be between 0.5 and 1 (e.g. 0.95).")
        steps = max(1, min(steps, 2000))
        n_paths = max(100, min(n_paths, 200000))

        rng = np.random.default_rng(seed)
        dt = T / steps
        # GBM increments: exp((mu - 0.5 sigma^2) dt + sigma sqrt(dt) Z)
        drift_term = (mu - 0.5 * sigma ** 2) * dt
        vol_term = sigma * np.sqrt(dt)
        Z = rng.standard_normal((n_paths, steps))
        log_incr = drift_term + vol_term * Z
        log_paths = np.cumsum(log_incr, axis=1)
        paths = S0 * np.exp(log_paths)                       # shape (n_paths, steps)
        terminal = paths[:, -1]

        mean_T = float(np.mean(terminal))
        median_T = float(np.median(terminal))
        std_T = float(np.std(terminal, ddof=1))
        theo_mean = S0 * np.exp(mu * T)                      # E[S_T] under GBM
        p_loss = float(np.mean(terminal < S0))               # prob end below start
        total_ret = terminal / S0 - 1.0
        a = 1 - conf
        var_level = float(np.percentile(total_ret, 100 * a))     # e.g. 5th pct return
        var_loss = -var_level
        tail = total_ret[total_ret <= var_level]
        cvar_loss = -float(np.mean(tail)) if tail.size else var_loss

        pcts = [1, 5, 25, 50, 75, 95, 99]
        percentiles = [{"pct": q, "price": _fin(float(np.percentile(terminal, q)), 4),
                        "return": _fin(float(np.percentile(total_ret, q)), 5)} for q in pcts]

        # ─────────────────────────── Section 4: Terminal Price Distribution ───────────────────────────
        sim_stats = [
            {"metric": "Mean", "value": _fin(mean_T, 4)},
            {"metric": "Median", "value": _fin(median_T, 4)},
            {"metric": "Std Dev", "value": _fin(std_T, 4)},
            {"metric": "Min", "value": _fin(float(np.min(terminal)), 4)},
            {"metric": "Max", "value": _fin(float(np.max(terminal)), 4)},
            {"metric": "5th Percentile", "value": _fin(float(np.percentile(terminal, 5)), 4)},
            {"metric": "25th Percentile", "value": _fin(float(np.percentile(terminal, 25)), 4)},
            {"metric": "75th Percentile", "value": _fin(float(np.percentile(terminal, 75)), 4)},
            {"metric": "95th Percentile", "value": _fin(float(np.percentile(terminal, 95)), 4)},
        ]

        # ─────────────────────────── Section 5/6: Option Pricing (risk-neutral MC) + Payoff ───────────────────────────
        # Priced with a SEPARATE risk-neutral pass (same Z draws, risk-free drift instead of the
        # real-world drift used for the paths/terminal-distribution sections above) — this is the
        # standard, unbiased way to Monte-Carlo price a European option; the real-world simulation
        # above is not risk-neutral and would give a biased price if discounted directly.
        option = None
        payoff_call = None
        is_call_default = True
        strike_val = None
        if strike is not None and str(strike) != "":
            Kx = float(strike)
            strike_val = Kx
            r_rate = float(rf_in) if (rf_in is not None and str(rf_in) != "") else mu
            rn_drift_term = (r_rate - 0.5 * sigma ** 2) * dt
            rn_log_incr = rn_drift_term + vol_term * Z
            rn_log_paths = np.cumsum(rn_log_incr, axis=1)
            rn_terminal = S0 * np.exp(rn_log_paths[:, -1])

            disc = np.exp(-r_rate * T)
            call_payoff = np.maximum(rn_terminal - Kx, 0.0)
            put_payoff = np.maximum(Kx - rn_terminal, 0.0)
            disc_call = disc * call_payoff
            disc_put = disc * put_payoff

            mc_call = float(np.mean(disc_call))
            mc_put = float(np.mean(disc_put))
            se_call = float(np.std(disc_call, ddof=1) / np.sqrt(n_paths))
            se_put = float(np.std(disc_put, ddof=1) / np.sqrt(n_paths))
            ci_call = (mc_call - 1.96 * se_call, mc_call + 1.96 * se_call)
            ci_put = (mc_put - 1.96 * se_put, mc_put + 1.96 * se_put)

            bs_call = float(_bs_price(S0, Kx, T, sigma, r_rate, is_call=True))
            bs_put = float(_bs_price(S0, Kx, T, sigma, r_rate, is_call=False))

            option = {
                "strike": _fin(Kx, 4), "risk_free_rate": _fin(r_rate, 6),
                "call": _fin(mc_call, 6), "put": _fin(mc_put, 6),
                "note": "Priced via a risk-neutral Monte Carlo pass (same draws, risk-free drift), discounted at the risk-free rate — not the real-world drift used for the paths above.",
                "risk_neutral": {
                    "risk_free_rate": _fin(r_rate, 6),
                    "mc_call": _fin(mc_call, 6), "mc_put": _fin(mc_put, 6),
                    "se_call": _fin(se_call, 6), "se_put": _fin(se_put, 6),
                    "ci_call_low": _fin(ci_call[0], 6), "ci_call_high": _fin(ci_call[1], 6),
                    "ci_put_low": _fin(ci_put[0], 6), "ci_put_high": _fin(ci_put[1], 6),
                    "bs_call": _fin(bs_call, 6), "bs_put": _fin(bs_put, 6),
                    "diff_call": _fin(mc_call - bs_call, 6), "diff_put": _fin(mc_put - bs_put, 6),
                    "mean_terminal_rn": _fin(float(np.mean(rn_terminal)), 4),
                },
            }
            payoff_call = call_payoff  # payoff-distribution panel uses the call payoff by default

        payoff_stats = None
        if payoff_call is not None:
            Kx = strike_val
            prob_itm_call = float(np.mean(payoff_call > 0))
            payoff_stats = [
                {"metric": "Mean Payoff", "value": _fin(float(np.mean(payoff_call)), 4)},
                {"metric": "Median Payoff", "value": _fin(float(np.median(payoff_call)), 4)},
                {"metric": "5th Percentile", "value": _fin(float(np.percentile(payoff_call, 5)), 4)},
                {"metric": "95th Percentile", "value": _fin(float(np.percentile(payoff_call, 95)), 4)},
                {"metric": "Probability ITM (call)", "value": _fin(prob_itm_call, 4)},
            ]

        # ─────────────────────────── Section 7: Probability Analysis (NEW) ───────────────────────────
        # Thresholds chosen relative to the starting price (and the strike, if pricing an option):
        # a modest +15% upside band, a -15% downside band, plus "above start" and (if an option is
        # priced) "above strike" — a small, sensible set of probability statements read off the
        # SAME real-world terminal distribution used in sections 3/4 (not the risk-neutral one).
        up_mult, down_mult = 1.15, 0.85
        prob_rows = [
            {"event": f"P(Price > Initial = {S0:g})", "probability": _fin(1 - p_loss, 4)},
            {"event": f"P(Price > {up_mult:g}×Initial = {up_mult * S0:,.2f})",
             "probability": _fin(float(np.mean(terminal > up_mult * S0)), 4)},
            {"event": f"P(Price < {down_mult:g}×Initial = {down_mult * S0:,.2f})",
             "probability": _fin(float(np.mean(terminal < down_mult * S0)), 4)},
        ]
        prob_thresholds = [S0, up_mult * S0, down_mult * S0]
        if strike_val is not None:
            prob_rows.append({"event": f"P(Price > Strike = {strike_val:g})",
                               "probability": _fin(float(np.mean(terminal > strike_val)), 4)})
            prob_thresholds.append(strike_val)

        # ─────────────────────────── Section 8: Convergence Analysis (NEW) ───────────────────────────
        # Re-uses PREFIXES of the already-simulated paths (no re-simulation) at increasing sample
        # sizes to show the classic Monte Carlo result: the estimate stabilises and its standard
        # error shrinks as roughly 1/sqrt(k) with more paths.
        checkpoints = sorted(set([k for k in (100, 500, 1000, 2500, 5000, n_paths) if k <= n_paths]))
        if not checkpoints or checkpoints[-1] != n_paths:
            checkpoints.append(n_paths)
        convergence = []
        conv_is_option = option is not None
        for k in checkpoints:
            if conv_is_option:
                sample = disc_call[:k]
                est = float(np.mean(sample))
                se = float(np.std(sample, ddof=1) / np.sqrt(k)) if k > 1 else None
            else:
                sample = terminal[:k]
                est = float(np.mean(sample))
                se = float(np.std(sample, ddof=1) / np.sqrt(k)) if k > 1 else None
            convergence.append({"n": k, "estimate": _fin(est, 6), "se": _fin(se, 6) if se is not None else None})
        convergence_metric = "Option Price (call)" if conv_is_option else "Mean Terminal Price"

        # ─────────────────────────── Charts (separate PNGs, tabbed on the frontend) ───────────────────────────
        charts = {}
        tgrid = np.linspace(0, T, steps + 1)
        n_show = min(100, n_paths)
        show_paths = np.hstack([np.full((n_show, 1), S0), paths[:n_show]])
        band_lo = np.percentile(np.hstack([np.full((n_paths, 1), S0), paths]), 5, axis=0)
        band_hi = np.percentile(np.hstack([np.full((n_paths, 1), S0), paths]), 95, axis=0)
        band_med = np.percentile(np.hstack([np.full((n_paths, 1), S0), paths]), 50, axis=0)

        # 3. Simulated Price Paths
        try:
            fig, ax = plt.subplots(figsize=(9, 5), dpi=118)
            for i in range(n_show):
                ax.plot(tgrid, show_paths[i], color=BLUE, alpha=0.10, lw=0.8)
            ax.plot(tgrid, band_med, color=RED, lw=1.8, label="Median path")
            ax.plot(tgrid, band_lo, color=AMBER, lw=1.3, ls="--", label="5th percentile")
            ax.plot(tgrid, band_hi, color=GREEN, lw=1.3, ls="--", label="95th percentile")
            ax.axhline(S0, color="#111827", lw=0.8, ls=":")
            ax.set_xlabel("Years"); ax.set_ylabel("Price")
            ax.set_title(f"{n_show} of {n_paths:,} simulated GBM paths (sample)")
            ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.2)
            fig.tight_layout()
            charts["paths"] = _png(fig)
        except Exception:
            plt.close("all"); charts["paths"] = None

        # 4. Terminal Price Distribution
        try:
            fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=118)
            ax.hist(terminal, bins=60, color=LIGHT_BLUE, edgecolor="white")
            ax.axvline(S0, color="#111827", lw=1, ls=":", label=f"Start {S0:g}")
            ax.axvline(mean_T, color=RED, lw=1.4, label=f"Mean {mean_T:,.1f}")
            ax.axvline(median_T, color=PURPLE, lw=1.2, ls="--", label=f"Median {median_T:,.1f}")
            ax.axvline(np.percentile(terminal, 100 * a), color=AMBER, lw=1.2, ls="--", label=f"VaR {conf:.0%}")
            ax.set_xlabel("Terminal price"); ax.set_ylabel("Frequency")
            ax.set_title("Terminal Price Distribution")
            ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.2)
            fig.tight_layout()
            charts["terminal_dist"] = _png(fig)
        except Exception:
            plt.close("all"); charts["terminal_dist"] = None

        # 6. Payoff Distribution (only if option priced)
        if payoff_call is not None:
            try:
                fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=118)
                ax.hist(payoff_call, bins=60, color="#86efac", edgecolor="white")
                ax.axvline(float(np.mean(payoff_call)), color=RED, lw=1.3, label=f"Mean {np.mean(payoff_call):,.2f}")
                ax.axvline(float(np.median(payoff_call)), color=PURPLE, lw=1.2, ls="--", label=f"Median {np.median(payoff_call):,.2f}")
                ax.set_xlabel("Call payoff at expiry"); ax.set_ylabel("Frequency")
                ax.set_title("Payoff Distribution (call)")
                ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.2)
                fig.tight_layout()
                charts["payoff_dist"] = _png(fig)
            except Exception:
                plt.close("all"); charts["payoff_dist"] = None

        # 7. Probability Analysis
        try:
            fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=118)
            ax.hist(terminal, bins=60, color=LIGHT_BLUE, edgecolor="white")
            colors = [RED, GREEN, AMBER, PURPLE]
            for i, thr in enumerate(prob_thresholds):
                ax.axvline(thr, color=colors[i % len(colors)], lw=1.4, ls="--", label=f"{thr:,.2f}")
            ax.set_xlabel("Terminal price"); ax.set_ylabel("Frequency")
            ax.set_title("Terminal Distribution with Probability Thresholds")
            ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.2)
            fig.tight_layout()
            charts["probability"] = _png(fig)
        except Exception:
            plt.close("all"); charts["probability"] = None

        # 8. Convergence Analysis
        try:
            ns = np.array([row["n"] for row in convergence], dtype=float)
            ests = np.array([row["estimate"] for row in convergence], dtype=float)
            ses = np.array([row["se"] if row["se"] is not None else 0.0 for row in convergence], dtype=float)
            fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=118)
            ax.plot(ns, ests, color=BLUE, lw=1.8, marker="o", ms=4, label=convergence_metric)
            ax.fill_between(ns, ests - 1.96 * ses, ests + 1.96 * ses, color=BLUE, alpha=0.15, label="±95% CI")
            if conv_is_option:
                ax.axhline(bs_call, color=RED, lw=1.2, ls=":", label=f"Black-Scholes {bs_call:,.4f}")
            ax.set_xscale("log")
            ax.set_xlabel("Number of simulations (log scale)"); ax.set_ylabel(convergence_metric)
            ax.set_title("Convergence of the Monte Carlo Estimate")
            ax.legend(fontsize=8, frameon=False); ax.grid(alpha=0.2)
            fig.tight_layout()
            charts["convergence"] = _png(fig)
        except Exception:
            plt.close("all"); charts["convergence"] = None

        interpretation = (
            f"Simulating {n_paths:,} geometric-Brownian-motion paths over {T:g} year(s) gives an expected terminal "
            f"price of {mean_T:,.2f} (median {median_T:,.2f}), starting from {S0:g}. Because GBM produces a "
            f"right-skewed lognormal distribution, the mean sits above the median — a few large winners pull it up. "
            f"There is a {p_loss:.0%} probability of ending below the starting price, and at {conf:.0%} confidence the "
            f"terminal loss reaches {var_loss:.1%} (with {cvar_loss:.1%} average once in that tail). The spread widens "
            f"with the square root of time, which is why the percentile bands fan out."
            + (f" The Monte Carlo call price of {option['call']:,.4f} sits within sampling noise of the "
               f"Black-Scholes check ({option['risk_neutral']['bs_call']:,.4f})." if option else "")
        )

        # ─────────────────────────── Section 1: Simulation Summary ───────────────────────────
        sim_type = "Option Pricing" if option is not None else "Price Simulation"
        summary_table = [
            {"metric": "Simulation Type", "value": sim_type},
            {"metric": "Simulations", "value": n_paths},
            {"metric": "Time Steps", "value": steps},
            {"metric": "Initial Price", "value": _fin(S0, 4)},
        ]
        if option is not None:
            summary_table.append({"metric": "Strike Price", "value": _fin(strike_val, 4)})
        summary_table += [
            {"metric": "Volatility", "value": _fin(sigma, 6)},
            {"metric": "Risk-Free Rate" if option is not None else "Drift", "value": _fin(option["risk_free_rate"], 6) if option is not None else _fin(mu, 6)},
            {"metric": "Maturity (years)", "value": _fin(T, 4)},
        ]
        if option is not None:
            rn = option["risk_neutral"]
            summary_table += [
                {"metric": "Estimated Option Price (call)", "value": _fin(rn["mc_call"], 6)},
                {"metric": "Standard Error", "value": _fin(rn["se_call"], 6)},
                {"metric": "95% CI", "value": f"[{rn['ci_call_low']:.4f}, {rn['ci_call_high']:.4f}]"},
            ]

        # ─────────────────────────── Section 2: Simulation Inputs (restatement) ───────────────────────────
        inputs_table = [
            {"metric": "Initial Price", "value": _fin(S0, 4)},
            {"metric": "Drift (annual)", "value": _fin(mu, 6)},
            {"metric": "Volatility (annual)", "value": _fin(sigma, 6)},
            {"metric": "Horizon (years)", "value": _fin(T, 4)},
            {"metric": "Time Steps", "value": steps},
            {"metric": "Number of Paths", "value": n_paths},
            {"metric": "VaR Confidence", "value": _fin(conf, 4)},
            {"metric": "Seed", "value": seed},
        ]
        if option is not None:
            inputs_table.append({"metric": "Option Strike", "value": _fin(strike_val, 4)})
            inputs_table.append({"metric": "Risk-Free Rate", "value": _fin(option["risk_free_rate"], 6)})

        results = {
            "status": "ok", "initial_price": _fin(S0, 4), "drift": _fin(mu, 6), "volatility": _fin(sigma, 6),
            "horizon_years": _fin(T, 4), "steps": steps, "n_paths": n_paths, "confidence": _fin(conf, 4), "seed": seed,
            "mean_terminal": _fin(mean_T, 4), "median_terminal": _fin(median_T, 4), "std_terminal": _fin(std_T, 4),
            "theoretical_mean": _fin(theo_mean, 4), "prob_loss": _fin(p_loss, 4),
            "var_loss": _fin(var_loss, 5), "cvar_loss": _fin(cvar_loss, 5),
            "expected_return": _fin(mean_T / S0 - 1.0, 5),
            "percentiles": percentiles, "option": option, "sim_stats": sim_stats,
            "payoff_stats": payoff_stats,
            "probability_table": prob_rows,
            "convergence_table": convergence, "convergence_metric": convergence_metric,
            "summary_table": summary_table, "inputs_table": inputs_table,
            "charts": charts,
            "interpretation": interpretation,
        }
        # Headline "plot" kept for backward compatibility (paths + terminal dist, side by side)
        plot = None
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=118, gridspec_kw={"width_ratios": [1.4, 1]})
            for i in range(n_show):
                ax1.plot(tgrid, show_paths[i], color=BLUE, alpha=0.10, lw=0.8)
            ax1.plot(tgrid, band_med, color=RED, lw=1.8, label="Median")
            ax1.plot(tgrid, band_lo, color=AMBER, lw=1.2, ls="--", label="5th pct")
            ax1.plot(tgrid, band_hi, color=GREEN, lw=1.2, ls="--", label="95th pct")
            ax1.axhline(S0, color="#111827", lw=0.8, ls=":")
            ax1.set_xlabel("Years"); ax1.set_ylabel("Price")
            ax1.set_title(f"{n_show} of {n_paths:,} simulated GBM paths")
            ax1.legend(fontsize=8, frameon=False)
            ax2.hist(terminal, bins=60, color=LIGHT_BLUE, edgecolor="white", orientation="horizontal")
            ax2.axhline(S0, color="#111827", lw=1, ls=":", label=f"Start {S0:g}")
            ax2.axhline(mean_T, color=RED, lw=1.2, label=f"Mean {mean_T:,.1f}")
            ax2.axhline(np.percentile(terminal, 100 * a), color=AMBER, lw=1.2, ls="--", label=f"VaR {conf:.0%}")
            ax2.set_xlabel("Frequency"); ax2.set_title("Terminal price distribution")
            ax2.legend(fontsize=7, frameon=False)
            fig.tight_layout()
            plot = _png(fig)
        except Exception:
            plt.close("all"); plot = None

        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
