#!/usr/bin/env python3
"""Marketing Mix Modeling (MMM) — aggregate weekly/monthly regression with
adstock (carry-over), saturation (diminishing returns), channel contribution
decomposition, ROI/ROAS, response curves and constrained budget optimization.
pandas / numpy / scipy / matplotlib.

Scope: aggregate time-series only (one row per period). No per-customer /
per-conversion attribution, no CLV/churn, no causal A/B inference — see
docs for the boundary with Attribution Analysis (per-conversion credit)
which this deliberately is NOT.

Builds a full step-6 report with additive sections:
  1. MMM Overview (KPI cards)
  2. Data & Model Setup (variable role table)
  3. Model Performance (actual vs predicted chart + fit table)
  4. Marketing Contribution (contribution chart + table)
  5. Channel Effectiveness (spend/ROI/ROAS table + ROAS bar chart)
  6. Adstock / Carryover Effect (decay chart + per-channel table)
  7. Diminishing Returns (saturation curve chart + marginal ROAS table)
  8. Response Curves (per-channel response curve chart)
  9. Incrementality Analysis (baseline vs actual chart + table)
  10. External Factors (driver decomposition — only if controls provided)
  11. Budget Optimization (scipy.optimize constrained allocation)
  12. Scenario Simulation (current / +10% / +20% / optimized)

Input (from mmm-page.tsx):
    data           : list[dict]           — one row per period
    target         : str                  — sales/revenue column
    channels       : list[str]            — marketing spend columns
    controls       : list[str] (optional) — price/promo/seasonality/etc.
    total_budget   : float (optional)     — for budget optimization; defaults
                                             to current total channel spend
    channel_bounds : dict[str, [min_pct, max_pct]] (optional) — bounds on
                                             each channel's share of the
                                             optimized budget, default [0, 1]

Output: { results: {...}, plot: <legacy single combined image, kept for
          backward compat>, charts: {key: base64 png, ...} }
"""
import sys
import json
import io
import base64
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import nnls, minimize

warnings.filterwarnings("ignore")

PALETTE = ['#a67b70', '#b5a888', '#c4956a', '#7a9471', '#8ba3a3', '#6b7565', '#d4c4a8', '#9a8471']


def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def adstock(x, decay):
    out = np.zeros_like(x, dtype=float)
    carry = 0.0
    for i, v in enumerate(x):
        carry = v + decay * carry
        out[i] = carry
    return out


def best_decay(spend, target_resid):
    """Grid-search the adstock decay that maximises correlation with the
    (control-adjusted) target — a standard, cheap MMM heuristic."""
    best_d, best_c = 0.0, -np.inf
    for d in np.arange(0.0, 0.91, 0.1):
        a = adstock(spend, d)
        if np.std(a) < 1e-9:
            continue
        c = abs(np.corrcoef(a, target_resid)[0, 1])
        if np.isfinite(c) and c > best_c:
            best_c, best_d = c, d
    return round(float(best_d), 2)


def safe_num(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except (TypeError, ValueError):
        return None


def main():
    try:
        payload = json.loads(sys.stdin.read())
        raw = payload.get("data", [])
        target_col = payload.get("target")
        channel_cols = list(payload.get("channels") or [])
        control_cols = [c for c in (payload.get("controls") or []) if c != target_col and c not in channel_cols]
        total_budget_in = payload.get("total_budget")
        bounds_in = payload.get("channel_bounds") or {}

        if not raw or not target_col or not channel_cols:
            print(json.dumps({"error": "Need target column and at least one channel column."}), file=sys.stderr)
            sys.exit(1)

        df = pd.DataFrame(raw)
        for c in [target_col] + channel_cols + control_cols:
            if c not in df.columns:
                print(json.dumps({"error": f"Column '{c}' not found in data."}), file=sys.stderr)
                sys.exit(1)
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=[target_col] + channel_cols).reset_index(drop=True)
        n = len(df)
        if n < max(8, len(channel_cols) + 3):
            print(json.dumps({"error": f"Not enough periods (n={n}) for {len(channel_cols)} channels."}), file=sys.stderr)
            sys.exit(1)
        for c in control_cols:
            df[c] = df[c].fillna(df[c].mean())

        y = df[target_col].to_numpy(dtype=float)

        # --- 1) pick adstock decay per channel (heuristic: correlate against
        #     target net of a naive control-only trend) -------------------
        control_trend = y - y.mean()
        decays = {}
        adstocked = {}
        for ch in channel_cols:
            s = df[ch].to_numpy(dtype=float)
            d = best_decay(s, control_trend)
            decays[ch] = d
            adstocked[ch] = adstock(s, d)

        # --- 2) saturation transform (log1p diminishing returns) ----------
        media_transform = {ch: np.log1p(adstocked[ch]) for ch in channel_cols}

        # --- 3) design matrix for NNLS: intercept + signed controls (as
        #     +/- pairs so effects can be negative, e.g. price) + nonneg
        #     media coefficients ----------------------------------------
        col_names = ["intercept"]
        X_cols = [np.ones(n)]
        for c in control_cols:
            v = df[c].to_numpy(dtype=float)
            col_names += [f"{c}__pos", f"{c}__neg"]
            X_cols += [v, -v]
        for ch in channel_cols:
            col_names.append(ch)
            X_cols.append(media_transform[ch])
        X = np.column_stack(X_cols)

        # scale each column to unit max for numeric stability, solve NNLS,
        # then rescale coefficients back
        scales = np.maximum(np.abs(X).max(axis=0), 1e-9)
        Xs = X / scales
        coef_s, _ = nnls(Xs, y)
        coef = coef_s / scales

        coef_map = dict(zip(col_names, coef))
        y_pred = X @ coef

        # --- fit metrics ---------------------------------------------------
        resid = y - y_pred
        sse = float(np.sum(resid ** 2))
        sst = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - sse / sst if sst > 0 else 0.0
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        mae = float(np.mean(np.abs(resid)))
        mape = float(np.mean(np.abs(resid / np.where(y == 0, np.nan, y)))) * 100
        if not np.isfinite(mape):
            mape = None

        # --- control (external factor) net effects -------------------------
        control_effect = {}
        for c in control_cols:
            eff = (coef_map.get(f"{c}__pos", 0.0) - coef_map.get(f"{c}__neg", 0.0)) * df[c].to_numpy(dtype=float)
            control_effect[c] = eff
        intercept_effect = np.full(n, coef_map["intercept"])
        baseline_series = intercept_effect + sum(control_effect.values()) if control_effect else intercept_effect

        media_contribution = {ch: coef_map[ch] * media_transform[ch] for ch in channel_cols}
        total_media = sum(v.sum() for v in media_contribution.values())
        total_actual = float(y.sum())
        total_baseline = float(baseline_series.sum())
        marketing_pct = (total_media / total_actual * 100) if total_actual else 0.0

        spend_total = {ch: float(df[ch].sum()) for ch in channel_cols}
        incr_total = {ch: float(media_contribution[ch].sum()) for ch in channel_cols}
        roi = {ch: ((incr_total[ch] - spend_total[ch]) / spend_total[ch] * 100) if spend_total[ch] > 0 else None for ch in channel_cols}
        roas = {ch: (incr_total[ch] / spend_total[ch]) if spend_total[ch] > 0 else None for ch in channel_cols}

        best_channel = max(channel_cols, key=lambda c: (roas[c] if roas[c] is not None else -1))

        # --- adstock / carryover table --------------------------------------
        carryover_rows = []
        weeks_axis = np.arange(1, 21)
        for ch in channel_cols:
            d = decays[ch]
            peak_effect = float(media_contribution[ch].max()) if len(media_contribution[ch]) else 0.0
            if d > 0:
                duration = int(np.ceil(np.log(0.05) / np.log(d))) if d < 1 else 20
            else:
                duration = 1
            carryover_rows.append({
                "channel": ch, "decay": round(d, 2), "peak_effect": round(peak_effect, 2),
                "carryover_duration_periods": min(duration, 20),
            })

        # --- diminishing returns / response curve -------------------------
        response_rows = []
        curve_points = {}
        for ch in channel_cols:
            avg_spend = float(df[ch].mean())
            max_spend = float(df[ch].max()) if df[ch].max() > 0 else avg_spend * 2 + 1
            xs = np.linspace(0, max(max_spend * 1.5, 1.0), 40)
            ys = coef_map[ch] * np.log1p(xs)
            curve_points[ch] = (xs.tolist(), ys.tolist())
            for level_pct in [50, 75, 100, 125, 150]:
                s_level = avg_spend * level_pct / 100
                incr = coef_map[ch] * np.log1p(s_level)
                marginal_roas = coef_map[ch] / (1 + s_level) if s_level >= 0 else None
                response_rows.append({
                    "channel": ch, "spend_level_pct": level_pct, "spend": round(s_level, 1),
                    "incremental_revenue": round(float(incr), 2),
                    "marginal_roas": round(float(marginal_roas), 3) if marginal_roas is not None else None,
                })

        # --- incrementality --------------------------------------------------
        incremental_series = y - baseline_series
        incrementality_table = []
        for i in range(n):
            incrementality_table.append({
                "period": df.iloc[i].get("week", i + 1) if "week" in df.columns else i + 1,
                "baseline": round(float(baseline_series[i]), 2),
                "actual": round(float(y[i]), 2),
                "incremental": round(float(incremental_series[i]), 2),
                "marketing_contribution_pct": round(float(incremental_series[i] / y[i] * 100), 1) if y[i] else None,
            })

        # --- external factors (only if controls provided) -------------------
        external_rows = []
        if control_cols:
            for c in control_cols:
                tot = float(control_effect[c].sum())
                external_rows.append({"factor": c, "contribution": round(tot, 2),
                                       "contribution_pct": round(tot / total_actual * 100, 2) if total_actual else None})
            external_rows.append({"factor": "baseline (intercept)", "contribution": round(float(intercept_effect.sum()), 2),
                                   "contribution_pct": round(float(intercept_effect.sum()) / total_actual * 100, 2) if total_actual else None})
            for ch in channel_cols:
                external_rows.append({"factor": ch, "contribution": round(float(incr_total[ch]), 2),
                                       "contribution_pct": round(incr_total[ch] / total_actual * 100, 2) if total_actual else None})

        # --- budget optimization ---------------------------------------------
        current_alloc = np.array([spend_total[ch] for ch in channel_cols], dtype=float)
        total_budget = float(total_budget_in) if safe_num(total_budget_in) else float(current_alloc.sum())

        def neg_revenue(x):
            return -sum(coef_map[ch] * np.log1p(max(v, 0)) for ch, v in zip(channel_cols, x))

        b_lo, b_hi = [], []
        for ch in channel_cols:
            lo_pct, hi_pct = bounds_in.get(ch, [0.0, 1.0])
            b_lo.append(max(0.0, lo_pct) * total_budget)
            b_hi.append(min(1.0, hi_pct if hi_pct else 1.0) * total_budget)
        bnds = list(zip(b_lo, b_hi))
        x0 = current_alloc / current_alloc.sum() * total_budget if current_alloc.sum() > 0 else np.full(len(channel_cols), total_budget / len(channel_cols))
        cons = ({"type": "eq", "fun": lambda x: np.sum(x) - total_budget},)
        opt = minimize(neg_revenue, x0, method="SLSQP", bounds=bnds, constraints=cons,
                        options={"maxiter": 300, "ftol": 1e-9})
        optimal_alloc = opt.x if opt.success else x0

        # Fair baseline for the optimizer's % gain: the *same* total_budget,
        # split in the current proportions (not the raw historical spend,
        # which may sum to a different total than total_budget) — otherwise
        # the comparison silently conflates "reallocating" with "spending
        # more/less overall".
        current_share = current_alloc / current_alloc.sum() if current_alloc.sum() > 0 else np.full(len(channel_cols), 1 / len(channel_cols))
        current_scaled = current_share * total_budget
        current_pred_media = sum(coef_map[ch] * np.log1p(max(current_scaled[i], 0)) for i, ch in enumerate(channel_cols))
        optimal_pred_media = sum(coef_map[ch] * np.log1p(max(optimal_alloc[i], 0)) for i, ch in enumerate(channel_cols))
        budget_gain_pct = ((optimal_pred_media - current_pred_media) / current_pred_media * 100) if current_pred_media > 0 else 0.0

        channels_table = []
        for i, ch in enumerate(channel_cols):
            cur = float(current_alloc[i])
            opt_v = float(optimal_alloc[i])
            channels_table.append({
                "channel": ch,
                "spend": round(spend_total[ch], 2),
                "incremental_revenue": round(incr_total[ch], 2),
                "contribution_pct": round(incr_total[ch] / total_actual * 100, 2) if total_actual else None,
                "roi": round(roi[ch], 1) if roi[ch] is not None else None,
                "roas": round(roas[ch], 3) if roas[ch] is not None else None,
                "decay": round(decays[ch], 2),
                "beta": round(float(coef_map[ch]), 4),
                "current_budget": round(cur, 2),
                "optimal_budget": round(opt_v, 2),
                "budget_change_pct": round((opt_v - cur) / cur * 100, 1) if cur > 0 else None,
            })

        # --- scenario simulation ---------------------------------------------
        def scaled_media_revenue(scale):
            alloc = current_alloc * scale
            return sum(coef_map[ch] * np.log1p(max(alloc[i], 0)) for i, ch in enumerate(channel_cols))

        scenarios = []
        for label, media_rev in [
            ("Current", current_pred_media),
            ("+10% budget", scaled_media_revenue(1.10)),
            ("+20% budget", scaled_media_revenue(1.20)),
            ("Optimized (same budget)", optimal_pred_media),
        ]:
            proj_revenue = total_baseline + media_rev
            scenarios.append({
                "scenario": label,
                "total_budget": round(total_budget * (1.10 if "10%" in label else 1.20 if "20%" in label else 1.0), 2),
                "projected_revenue": round(proj_revenue, 2),
                "incremental_revenue": round(media_rev, 2),
                "vs_current_pct": round((proj_revenue - (total_baseline + current_pred_media)) / (total_baseline + current_pred_media) * 100, 2) if (total_baseline + current_pred_media) else None,
            })

        # --- variable-role table (Data & Model Setup) --------------------------
        setup_rows = [{"role": "Target (Sales/Revenue)", "column": target_col}]
        for ch in channel_cols:
            setup_rows.append({"role": "Marketing variable", "column": ch})
        for c in control_cols:
            setup_rows.append({"role": "Control variable", "column": c})

        # ============================= CHARTS ==============================
        charts = {}

        # 3. Model performance
        fig, ax = plt.subplots(figsize=(9, 4.2))
        x_axis = df["week"] if "week" in df.columns else np.arange(1, n + 1)
        ax.plot(x_axis, y, label="Actual", color=PALETTE[0], marker="o", ms=3)
        ax.plot(x_axis, y_pred, label="Predicted", color=PALETTE[2], linestyle="--")
        ax.set_title(f"Actual vs Predicted {target_col} (R²={r2:.3f})")
        ax.set_xlabel("Period"); ax.set_ylabel(target_col); ax.legend(); ax.grid(alpha=0.25)
        charts["model_fit"] = fig_to_b64(fig)

        # 4. Marketing contribution (stacked bar: baseline + each channel, total only)
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        labels = ["Baseline"] + channel_cols
        values = [total_baseline] + [incr_total[ch] for ch in channel_cols]
        colors = [PALETTE[-1]] + PALETTE[: len(channel_cols)]
        ax.bar(labels, values, color=colors)
        ax.set_title("Revenue Contribution: Baseline vs Channels")
        ax.set_ylabel(target_col); ax.tick_params(axis="x", rotation=25)
        charts["contribution"] = fig_to_b64(fig)

        # 5. ROAS by channel
        fig, ax = plt.subplots(figsize=(7, 4))
        roas_vals = [roas[ch] if roas[ch] is not None else 0 for ch in channel_cols]
        ax.bar(channel_cols, roas_vals, color=PALETTE[: len(channel_cols)])
        ax.axhline(1.0, color="red", linestyle="--", linewidth=1, label="Break-even (ROAS=1)")
        ax.set_title("ROAS by Channel"); ax.set_ylabel("ROAS"); ax.legend()
        ax.tick_params(axis="x", rotation=25)
        charts["roas_by_channel"] = fig_to_b64(fig)

        # 6. Adstock / carryover decay curves
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        for i, ch in enumerate(channel_cols):
            d = decays[ch]
            carry = d ** weeks_axis
            ax.plot(weeks_axis, carry, label=f"{ch} (decay={d:.1f})", color=PALETTE[i % len(PALETTE)])
        ax.set_title("Adstock Carryover Effect Over Time")
        ax.set_xlabel("Periods since spend"); ax.set_ylabel("Remaining effect (fraction)")
        ax.legend(fontsize=8)
        charts["adstock_carryover"] = fig_to_b64(fig)

        # 7/8. Saturation / response curves (shared figure, reused for both sections)
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        for i, ch in enumerate(channel_cols):
            xs, ys = curve_points[ch]
            ax.plot(xs, ys, label=ch, color=PALETTE[i % len(PALETTE)])
            ax.scatter([spend_total[ch] / n], [coef_map[ch] * np.log1p(spend_total[ch] / n)],
                       color=PALETTE[i % len(PALETTE)], zorder=5, s=30)
        ax.set_title("Diminishing Returns / Response Curves (dot = current avg spend)")
        ax.set_xlabel("Spend per period"); ax.set_ylabel("Incremental revenue"); ax.legend(fontsize=8)
        charts["saturation_curves"] = fig_to_b64(fig)

        # 9. Incrementality: baseline vs actual
        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.plot(x_axis, y, label="Actual", color=PALETTE[0])
        ax.plot(x_axis, baseline_series, label="Baseline (no-marketing counterfactual)", color=PALETTE[5], linestyle="--")
        ax.fill_between(x_axis, baseline_series, y, color=PALETTE[2], alpha=0.25, label="Incremental (marketing)")
        ax.set_title("Incrementality: Baseline vs Actual"); ax.legend(fontsize=8)
        charts["incrementality"] = fig_to_b64(fig)

        # 10. External factors (only if controls provided)
        if control_cols:
            fig, ax = plt.subplots(figsize=(7.5, 4.2))
            ext_labels = [r["factor"] for r in external_rows]
            ext_vals = [r["contribution"] for r in external_rows]
            ax.bar(ext_labels, ext_vals, color=PALETTE[: len(ext_labels)])
            ax.set_title("Revenue Driver Decomposition"); ax.tick_params(axis="x", rotation=35)
            charts["external_factors"] = fig_to_b64(fig)

        # 11. Budget optimization: current vs optimal
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        width = 0.35
        xpos = np.arange(len(channel_cols))
        ax.bar(xpos - width / 2, current_alloc, width, label="Current", color=PALETTE[0])
        ax.bar(xpos + width / 2, optimal_alloc, width, label="Optimized", color=PALETTE[3])
        ax.set_xticks(xpos); ax.set_xticklabels(channel_cols, rotation=25)
        ax.set_title(f"Current vs Optimized Budget Allocation (+{budget_gain_pct:.1f}% media revenue)")
        ax.legend()
        charts["budget_optimization"] = fig_to_b64(fig)

        # 12. Scenario simulation
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        sc_labels = [s["scenario"] for s in scenarios]
        sc_vals = [s["projected_revenue"] for s in scenarios]
        ax.bar(sc_labels, sc_vals, color=PALETTE[: len(sc_labels)])
        ax.set_title("Scenario Simulation: Projected Revenue"); ax.tick_params(axis="x", rotation=15)
        charts["scenario_simulation"] = fig_to_b64(fig)

        # legacy single combined plot (kept so any older consumer doesn't break)
        plot_image = charts.get("model_fit")

        results = {
            "status": "ok",
            "target": target_col,
            "channels": channel_cols,
            "controls": control_cols,
            "n_periods": n,
            "n_channels": len(channel_cols),
            "r_squared": round(r2, 4),
            "rmse": round(rmse, 2),
            "mae": round(mae, 2),
            "mape": round(mape, 2) if mape is not None else None,
            "total_sales": round(total_actual, 2),
            "total_spend": round(float(sum(spend_total.values())), 2),
            "base_contribution": round(total_baseline, 2),
            "base_contribution_pct": round(total_baseline / total_actual, 4) if total_actual else None,
            "marketing_contribution_pct": round(marketing_pct, 2),
            "incremental_revenue": round(total_media, 2),
            "overall_roas": round(total_media / sum(spend_total.values()), 3) if sum(spend_total.values()) else None,
            "overall_roi": round((total_media - sum(spend_total.values())) / sum(spend_total.values()) * 100, 1) if sum(spend_total.values()) else None,
            "best_channel": best_channel,
            "total_budget": round(total_budget, 2),
            "optimal_total_revenue": round(total_baseline + optimal_pred_media, 2),
            "budget_gain_pct": round(budget_gain_pct, 2),
            "setup_table": setup_rows,
            "fit_table": [{"metric": "R²", "value": round(r2, 4)},
                          {"metric": "RMSE", "value": round(rmse, 2)},
                          {"metric": "MAE", "value": round(mae, 2)},
                          {"metric": "MAPE (%)", "value": round(mape, 2) if mape is not None else None}],
            "channels_table": channels_table,
            "carryover_table": carryover_rows,
            "response_curve_table": response_rows,
            "incrementality_table": incrementality_table[: min(n, 300)],
            "external_factors_table": external_rows,
            "scenario_table": scenarios,
            "charts": charts,
        }

        final_output = {"results": results, "plot": plot_image, "charts": charts}
        print(json.dumps(final_output))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
