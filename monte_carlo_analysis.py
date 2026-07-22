#!/usr/bin/env python3
"""Monte Carlo Simulation — geometric Brownian motion asset paths. NumPy (+SciPy).

Simulates many random price paths under GBM, then summarises the terminal
distribution: expected value, percentiles, probability of loss, and terminal
VaR/CVaR. Optionally prices a European option on the simulated terminal price.

Input (from monte-carlo-page.tsx):
    initial_price  : float  S0
    drift          : float  annual expected return mu
    volatility     : float  annual sigma
    horizon_years  : float  T
    steps          : int    time steps per path (default 252)
    n_paths        : int    number of paths (default 10000)
    confidence     : float  (default 0.95) for terminal VaR
    seed           : int    (default 42)
    option_strike  : float  (optional) if given, price a European call & put
Output: { results: {...}, plot } (sample paths + terminal histogram).
"""
import sys, json, io, base64
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _fin(x, nd=6):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


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

        # optional European option pricing via risk-neutral? Here we use the
        # simulated (real-world drift) terminal — clarify it's an expected payoff,
        # discounted at the drift as a simple estimate.
        option = None
        if strike is not None and str(strike) != "":
            Kx = float(strike)
            disc = np.exp(-mu * T)
            call = float(np.mean(np.maximum(terminal - Kx, 0.0)) * disc)
            put = float(np.mean(np.maximum(Kx - terminal, 0.0)) * disc)
            option = {"strike": _fin(Kx, 4), "call": _fin(call, 6), "put": _fin(put, 6),
                      "note": "Expected discounted payoff under the specified drift (not risk-neutral)."}

        # plot: sample paths + terminal histogram
        plot = None
        try:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=118,
                                           gridspec_kw={"width_ratios": [1.4, 1]})
            tgrid = np.linspace(0, T, steps + 1)
            n_show = min(200, n_paths)
            show_paths = np.hstack([np.full((n_show, 1), S0), paths[:n_show]])
            for i in range(n_show):
                ax1.plot(tgrid, show_paths[i], color="#2563eb", alpha=0.06, lw=0.8)
            # percentile bands over time
            band_lo = np.percentile(np.hstack([np.full((n_paths, 1), S0), paths]), 5, axis=0)
            band_hi = np.percentile(np.hstack([np.full((n_paths, 1), S0), paths]), 95, axis=0)
            band_med = np.percentile(np.hstack([np.full((n_paths, 1), S0), paths]), 50, axis=0)
            ax1.plot(tgrid, band_med, color="#dc2626", lw=1.8, label="Median")
            ax1.plot(tgrid, band_lo, color="#f59e0b", lw=1.2, ls="--", label="5th pct")
            ax1.plot(tgrid, band_hi, color="#16a34a", lw=1.2, ls="--", label="95th pct")
            ax1.axhline(S0, color="#111827", lw=0.8, ls=":")
            ax1.set_xlabel("Years"); ax1.set_ylabel("Price")
            ax1.set_title(f"{n_show} of {n_paths:,} simulated GBM paths")
            ax1.legend(fontsize=8, frameon=False)
            ax2.hist(terminal, bins=60, color="#93c5fd", edgecolor="white", orientation="horizontal")
            ax2.axhline(S0, color="#111827", lw=1, ls=":", label=f"Start {S0:g}")
            ax2.axhline(mean_T, color="#dc2626", lw=1.2, label=f"Mean {mean_T:,.1f}")
            ax2.axhline(np.percentile(terminal, 100 * a), color="#f59e0b", lw=1.2, ls="--",
                        label=f"VaR {conf:.0%}")
            ax2.set_xlabel("Frequency"); ax2.set_title("Terminal price distribution")
            ax2.legend(fontsize=7, frameon=False)
            fig.tight_layout()
            buf = io.BytesIO(); fig.savefig(buf, format="png"); plt.close(fig)
            plot = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception:
            plt.close("all"); plot = None

        interpretation = (
            f"Simulating {n_paths:,} geometric-Brownian-motion paths over {T:g} year(s) gives an expected terminal "
            f"price of {mean_T:,.2f} (median {median_T:,.2f}), starting from {S0:g}. Because GBM produces a "
            f"right-skewed lognormal distribution, the mean sits above the median — a few large winners pull it up. "
            f"There is a {p_loss:.0%} probability of ending below the starting price, and at {conf:.0%} confidence the "
            f"terminal loss reaches {var_loss:.1%} (with {cvar_loss:.1%} average once in that tail). The spread widens "
            f"with the square root of time, which is why the percentile bands fan out."
        )

        results = {
            "status": "ok", "initial_price": _fin(S0, 4), "drift": _fin(mu, 6), "volatility": _fin(sigma, 6),
            "horizon_years": _fin(T, 4), "steps": steps, "n_paths": n_paths, "confidence": _fin(conf, 4), "seed": seed,
            "mean_terminal": _fin(mean_T, 4), "median_terminal": _fin(median_T, 4), "std_terminal": _fin(std_T, 4),
            "theoretical_mean": _fin(theo_mean, 4), "prob_loss": _fin(p_loss, 4),
            "var_loss": _fin(var_loss, 5), "cvar_loss": _fin(cvar_loss, 5),
            "expected_return": _fin(mean_T / S0 - 1.0, 5),
            "percentiles": percentiles, "option": option,
            "interpretation": interpretation,
        }
        print(json.dumps({"results": results, "plot": plot}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
