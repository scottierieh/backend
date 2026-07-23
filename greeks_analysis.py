#!/usr/bin/env python3
"""Greeks Analysis — option sensitivities across the underlying. QuantLib-Python.

Prices a European option and profiles all five Greeks (delta, gamma, vega,
theta, rho) as the underlying moves, so you can see how the risk changes with
price — not just at the current spot. Also profiles vega vs volatility, theta
vs time-to-expiry, rho vs interest rate, a scenario table, and a Delta
surface (spot x time-to-expiry heatmap).

Input (from greeks-analysis-page.tsx):
    option_type    : "call" | "put"
    spot           : float  underlying price S
    strike         : float  strike K
    expiry_years   : float  time to expiry T
    volatility     : float  annual sigma
    risk_free_rate : float  annual r
    dividend_yield : float  annual q (default 0)
Output: { results: {greeks at spot + profile arrays + extra sweeps + scenarios}, charts: {...} }
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


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


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

        def _engine_for(spot_val, strike_val, T_val, sigma_val, r_val, q_val, is_call_val):
            maturity = val_date + max(int(round(T_val * 365)), 1)
            payoff = ql.PlainVanillaPayoff(ql.Option.Call if is_call_val else ql.Option.Put, strike_val)
            exercise = ql.EuropeanExercise(maturity)
            rf_ts = ql.YieldTermStructureHandle(ql.FlatForward(val_date, r_val, dc))
            div_ts = ql.YieldTermStructureHandle(ql.FlatForward(val_date, q_val, dc))
            vol_ts = ql.BlackVolTermStructureHandle(ql.BlackConstantVol(val_date, cal, sigma_val, dc))
            spot_h = ql.QuoteHandle(ql.SimpleQuote(float(spot_val)))
            proc = ql.BlackScholesMertonProcess(spot_h, div_ts, rf_ts, vol_ts)
            opt = ql.VanillaOption(payoff, exercise)
            opt.setPricingEngine(ql.AnalyticEuropeanEngine(proc))
            return opt

        def greeks_at(s, k=None, t=None, sig=None, rr=None, qq=None, call=None):
            """Compute Greeks at arbitrary (possibly shifted) inputs; defaults to base inputs."""
            k = K if k is None else k
            t = T if t is None else t
            sig = sigma if sig is None else sig
            rr = r if rr is None else rr
            qq = q if qq is None else qq
            call = is_call if call is None else call
            opt = _engine_for(s, k, t, sig, rr, qq, call)
            return {
                "price": float(opt.NPV()), "delta": float(opt.delta()), "gamma": float(opt.gamma()),
                "vega": float(opt.vega()) / 100.0, "theta": float(opt.theta()) / 365.0,
                "rho": float(opt.rho()) / 100.0,
            }

        at = greeks_at(S)

        # ---- profile across underlying (base 2x3 dataset, reused for ②-⑥ vs-Spot views) ----
        srange = np.linspace(max(0.3 * S, 0.01), 1.7 * S, 60)
        prof = {k: [] for k in ("price", "delta", "gamma", "vega", "theta", "rho")}
        for s in srange:
            g = greeks_at(s)
            for k in prof:
                prof[k].append(_fin(g[k], 6))

        # ---- ④ Vega vs Volatility sweep (10%-50%, other inputs fixed at base) ----
        vol_range = np.linspace(0.10, 0.50, 25)
        vega_vs_vol = [_fin(greeks_at(S, sig=v)["vega"], 6) for v in vol_range]

        # ---- ⑤ Theta vs Time-to-Expiry sweep (near-zero up to current T, spot fixed at S) ----
        t_range = np.linspace(max(T * 0.02, 1.0 / 365.0), T, 25)
        theta_vs_time = [_fin(greeks_at(S, t=tt)["theta"], 6) for tt in t_range]

        # ---- ⑥ Rho vs Interest-Rate sweep (0%-8%), both call & put rho for contrast ----
        rate_range = np.linspace(0.0, 0.08, 25)
        rho_call_vs_rate = [_fin(greeks_at(S, rr=rv, call=True)["rho"], 6) for rv in rate_range]
        rho_put_vs_rate = [_fin(greeks_at(S, rr=rv, call=False)["rho"], 6) for rv in rate_range]

        # ---- ⑦ Delta surface: Spot (x) x Time-to-Expiry (y) heatmap (advanced, Delta only) ----
        surf_spots = np.linspace(max(0.3 * S, 0.01), 1.7 * S, 20)
        surf_ts = np.linspace(max(T * 0.05, 1.0 / 365.0), T, 10)
        delta_surface = np.zeros((len(surf_ts), len(surf_spots)))
        for i, tt in enumerate(surf_ts):
            for j, sv in enumerate(surf_spots):
                delta_surface[i, j] = greeks_at(sv, t=tt)["delta"]

        # ---- ⑧ Scenario table: recompute Delta/Gamma/Vega/Theta under shifted inputs ----
        scenarios = [
            {"scenario": "Base", "s": S, "sig": sigma, "t": T, "rr": r},
            {"scenario": "Spot +10%", "s": S * 1.10, "sig": sigma, "t": T, "rr": r},
            {"scenario": "Vol +5pp", "s": S, "sig": sigma + 0.05, "t": T, "rr": r},
            {"scenario": "Time -30 Days", "s": S, "sig": sigma, "t": max(T - 30.0 / 365.0, 1.0 / 365.0), "rr": r},
            {"scenario": "Rate +1pp", "s": S, "sig": sigma, "t": T, "rr": r + 0.01},
        ]
        scenario_table = []
        for sc in scenarios:
            g = greeks_at(sc["s"], sig=sc["sig"], t=sc["t"], rr=sc["rr"])
            scenario_table.append({
                "scenario": sc["scenario"], "delta": _fin(g["delta"], 6), "gamma": _fin(g["gamma"], 6),
                "vega": _fin(g["vega"], 6), "theta": _fin(g["theta"], 6),
            })

        # ---- charts: split combined 2x3 panel into 6 separate PNGs, plus extra sweep charts + surface ----
        charts = {}
        try:
            panels = [("price", "Value", "#2563eb"), ("delta", "Delta", "#16a34a"),
                      ("gamma", "Gamma", "#9333ea"), ("vega", "Vega (per 1%)", "#0891b2"),
                      ("theta", "Theta (per day)", "#dc2626"), ("rho", "Rho (per 1%)", "#f59e0b")]
            for key, title, col in panels:
                fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=115)
                ys = [np.nan if v is None else v for v in prof[key]]
                ax.plot(srange, ys, color=col, lw=2)
                ax.axvline(K, color="#94a3b8", ls=":", lw=1, label="Strike")
                ax.axvline(S, color="#111827", ls="--", lw=1, label="Spot")
                ax.scatter([S], [at[key]], color=col, zorder=5, s=50, edgecolor="white")
                ax.set_title(f"{title} vs Underlying ({otype.capitalize()}, K={K:g})", fontsize=11)
                ax.set_xlabel("Underlying price"); ax.set_ylabel(title)
                ax.grid(alpha=0.2); ax.legend(fontsize=8)
                fig.tight_layout()
                charts[key] = _png(fig)
        except Exception:
            plt.close("all")

        try:
            fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=115)
            ax.plot(vol_range * 100, vega_vs_vol, color="#0891b2", lw=2, marker="o", ms=3)
            ax.axvline(sigma * 100, color="#111827", ls="--", lw=1, label="Current vol")
            ax.set_title(f"Vega vs Volatility ({otype.capitalize()}, S={S:g}, K={K:g})", fontsize=11)
            ax.set_xlabel("Volatility (%)"); ax.set_ylabel("Vega (per 1%)")
            ax.grid(alpha=0.2); ax.legend(fontsize=8)
            fig.tight_layout()
            charts["vega_vs_vol"] = _png(fig)
        except Exception:
            plt.close("all")

        try:
            fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=115)
            days = t_range * 365
            ax.plot(days, theta_vs_time, color="#dc2626", lw=2, marker="o", ms=3)
            ax.invert_xaxis()
            ax.set_title(f"Theta vs Days to Expiry ({otype.capitalize()}, S={S:g}, K={K:g})", fontsize=11)
            ax.set_xlabel("Days to expiry (expiry on the right)"); ax.set_ylabel("Theta (per day)")
            ax.grid(alpha=0.2)
            fig.tight_layout()
            charts["theta_vs_time"] = _png(fig)
        except Exception:
            plt.close("all")

        try:
            fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=115)
            ax.plot(rate_range * 100, rho_call_vs_rate, color="#16a34a", lw=2, marker="o", ms=3, label="Call rho")
            ax.plot(rate_range * 100, rho_put_vs_rate, color="#dc2626", lw=2, marker="o", ms=3, label="Put rho")
            ax.axhline(0, color="#94a3b8", lw=1)
            ax.axvline(r * 100, color="#111827", ls="--", lw=1, label="Current rate")
            ax.set_title(f"Rho vs Interest Rate (S={S:g}, K={K:g})", fontsize=11)
            ax.set_xlabel("Risk-free rate (%)"); ax.set_ylabel("Rho (per 1%)")
            ax.grid(alpha=0.2); ax.legend(fontsize=8)
            fig.tight_layout()
            charts["rho_vs_rate"] = _png(fig)
        except Exception:
            plt.close("all")

        try:
            fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=115)
            im = ax.pcolormesh(surf_spots, surf_ts * 365, delta_surface, shading="auto", cmap="RdYlGn")
            ax.axvline(K, color="#111827", ls=":", lw=1)
            ax.set_title(f"Delta Surface: Spot x Days-to-Expiry ({otype.capitalize()}, K={K:g})", fontsize=11)
            ax.set_xlabel("Underlying price"); ax.set_ylabel("Days to expiry")
            fig.colorbar(im, ax=ax, label="Delta")
            fig.tight_layout()
            charts["delta_surface"] = _png(fig)
        except Exception:
            plt.close("all")

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

        greeks_table = [
            {"greek": "Delta", "value": _fin(at["delta"], 6),
             "desc_en": "Change in option value per $1 move in the underlying.",
             "desc_ko": "기초자산이 $1 움직일 때 옵션 가치의 변화."},
            {"greek": "Gamma", "value": _fin(at["gamma"], 6),
             "desc_en": "Change in delta per $1 move in the underlying — how fast delta itself shifts.",
             "desc_ko": "기초자산이 $1 움직일 때 델타의 변화 — 델타 자체가 얼마나 빨리 변하는지."},
            {"greek": "Vega", "value": _fin(at["vega"], 6),
             "desc_en": "Change in option value per 1 percentage-point rise in volatility.",
             "desc_ko": "변동성이 1%p 상승할 때 옵션 가치의 변화."},
            {"greek": "Theta", "value": _fin(at["theta"], 6),
             "desc_en": "Change in option value per day of time decay, all else equal.",
             "desc_ko": "다른 조건이 동일할 때 하루가 지나며 옵션 가치가 줄어드는 정도(시간가치 소멸)."},
            {"greek": "Rho", "value": _fin(at["rho"], 6),
             "desc_en": "Change in option value per 1 percentage-point rise in the risk-free rate.",
             "desc_ko": "무위험이자율이 1%p 상승할 때 옵션 가치의 변화."},
        ]

        results = {
            "status": "ok", "option_type": otype, "state": state, "moneyness": _fin(moneyness, 4),
            "spot": _fin(S, 4), "strike": _fin(K, 4), "expiry_years": _fin(T, 4),
            "volatility": _fin(sigma, 6), "risk_free_rate": _fin(r, 6), "dividend_yield": _fin(q, 6),
            "price": _fin(at["price"], 6), "delta": _fin(at["delta"], 6), "gamma": _fin(at["gamma"], 6),
            "vega": _fin(at["vega"], 6), "theta": _fin(at["theta"], 6), "rho": _fin(at["rho"], 6),
            "profile": {"underlying": [_fin(x, 4) for x in srange.tolist()], **prof},
            "vega_vol_profile": {"volatility": [_fin(x, 6) for x in vol_range.tolist()], "vega": vega_vs_vol},
            "theta_time_profile": {"expiry_years": [_fin(x, 6) for x in t_range.tolist()],
                                    "days_to_expiry": [_fin(x * 365, 2) for x in t_range.tolist()],
                                    "theta": theta_vs_time},
            "rho_rate_profile": {"risk_free_rate": [_fin(x, 6) for x in rate_range.tolist()],
                                  "rho_call": rho_call_vs_rate, "rho_put": rho_put_vs_rate},
            "delta_surface": {"underlying": [_fin(x, 4) for x in surf_spots.tolist()],
                               "days_to_expiry": [_fin(x * 365, 2) for x in surf_ts.tolist()],
                               "delta": [[_fin(v, 6) for v in row] for row in delta_surface.tolist()]},
            "scenario_table": scenario_table,
            "greeks_table": greeks_table,
            "interpretation": interpretation,
            "advanced_notes": {
                "pnl_decomposition": "Skipped: a Greek-based P&L decomposition (delta P&L + gamma P&L + vega P&L + theta P&L) requires an actual realized move in spot/vol/time between two dates. This page is a single-point-in-time calculator with no historical position tracking, so there is no real move to decompose — a synthetic assumed move would not reflect what actually happened to a position.",
                "portfolio_greeks": "Skipped: portfolio-level Greeks require aggregating multiple option positions (different strikes, expiries, types and quantities). This page currently collects inputs for a single option only, so a true portfolio view would need a new multi-position input UI — a larger scope change than this analysis covers.",
            },
        }
        print(json.dumps({"results": results, "charts": charts}))
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
