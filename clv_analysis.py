#!/usr/bin/env python3
"""Customer Lifetime Value — historic + model-based predictive CLV.
pandas / numpy / matplotlib / lifetimes.

Aggregates transaction-level rows to one row per customer, computes the
existing simple heuristic (avg order value x frequency x lifespan x margin —
already used client-side), and additionally fits probabilistic CLV models
(BG/NBD, Pareto/NBD, Gamma-Gamma) via the `lifetimes` package to build a
10-section step-6 report:
  1. CLV Overview (KPI cards)
  2. CLV Calculation (formula/methodology breakdown)
  3. CLV Distribution (histogram + Pareto concentration chart)
  4. CLV by Customer (per-customer table, heuristic fields)
  5. CLV Model Comparison (Historical vs BG/NBD vs Pareto/NBD vs Gamma-Gamma vs ML placeholder)
  6. Predicted vs Actual CLV (train/holdout if data span allows, else proxy comparison — documented)
  7. CLV Forecast (cumulative predicted CLV over 6/12/24 months)
  8. CLV by Acquisition Channel (conditional on channel_col)
  9. CLV vs CAC (conditional on cac input)
  10. CLV Value Tier (terciles on model-based predictive CLV)

Scope note: this analysis does NOT compute RFM scores, customer segment
labels, or a standalone churn-prediction feature — alive-probability from
BG/NBD is used only as an internal input to the CLV calculation.

Input (from clv-page.tsx):
    data              : list[dict]
    customer_id_col   : str
    invoice_date_col  : str
    amount_col        : str
    margin_pct        : float   (0-100)
    lifespan_years    : float
    channel_col       : str | None   (optional)
    cac               : float | dict[str, float] | None   (optional)
Output: { results: {...}, plot: <same as one chart or null> }
"""
import sys
import json
import warnings
import io
import base64

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

BLUE = "#2563eb"
GREEN = "#16a34a"
RED = "#dc2626"
AMBER = "#d97706"
PURPLE = "#9333ea"


def _fin(x, nd=6):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return round(v, nd) if np.isfinite(v) else None


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches='tight')
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get('data')
        customer_id_col = payload.get('customer_id_col')
        invoice_date_col = payload.get('invoice_date_col')
        amount_col = payload.get('amount_col')
        margin_pct = float(payload.get('margin_pct', 30))
        lifespan_years = float(payload.get('lifespan_years', 3))
        channel_col = payload.get('channel_col') or None
        cac_input = payload.get('cac', None)

        if not all([data, customer_id_col, invoice_date_col, amount_col]):
            raise ValueError("Missing data or required column names.")

        margin = margin_pct / 100.0

        df = pd.DataFrame(data)
        df[invoice_date_col] = pd.to_datetime(df[invoice_date_col], errors='coerce')
        df[amount_col] = pd.to_numeric(df[amount_col], errors='coerce')
        df[customer_id_col] = df[customer_id_col].astype(str)

        subset_cols = [customer_id_col, invoice_date_col, amount_col]
        if channel_col and channel_col in df.columns:
            subset_cols.append(channel_col)
        df = df.dropna(subset=[customer_id_col, invoice_date_col, amount_col])
        df = df[df[amount_col] > 0]
        if df.empty:
            raise ValueError("No valid data for CLV analysis after cleaning.")

        n_transactions = int(len(df))
        snapshot_date = df[invoice_date_col].max()

        # ══════════════════════ Per-customer aggregation (heuristic base) ══════════════════════
        grp = df.groupby(customer_id_col)
        cust = grp.agg(
            orders=(invoice_date_col, 'count'),
            total_revenue=(amount_col, 'sum'),
            first_date=(invoice_date_col, 'min'),
            last_date=(invoice_date_col, 'max'),
        )
        cust['avg_order_value'] = cust['total_revenue'] / cust['orders']
        cust['tenure_days'] = (cust['last_date'] - cust['first_date']).dt.days.clip(lower=1)
        cust['orders_per_year'] = np.where(cust['orders'] <= 1, cust['orders'], cust['orders'] / (cust['tenure_days'] / 365.0))
        cust['historic_clv'] = cust['total_revenue'] * margin
        cust['predictive_clv_heuristic'] = cust['avg_order_value'] * cust['orders_per_year'] * lifespan_years * margin

        if channel_col and channel_col in df.columns:
            chan_mode = grp[channel_col].agg(lambda s: s.mode().iat[0] if not s.mode().empty else None)
            cust['channel'] = chan_mode

        n_customers = int(len(cust))
        total_revenue = float(cust['total_revenue'].sum())

        # ══════════════════════ Section 1: CLV Overview ══════════════════════
        overview = {
            "n_customers": n_customers,
            "n_transactions": n_transactions,
            "total_revenue": _fin(total_revenue, 2),
            "avg_historic_clv": _fin(cust['historic_clv'].mean(), 2),
            "avg_predictive_clv": _fin(cust['predictive_clv_heuristic'].mean(), 2),
            "margin_pct": _fin(margin_pct, 2),
            "lifespan_years": _fin(lifespan_years, 2),
        }

        # ══════════════════════ Section 2: CLV Calculation (methodology breakdown) ══════════════════════
        def stats5(series):
            s = series.dropna()
            if s.empty:
                return {"min": None, "median": None, "mean": None, "max": None}
            return {"min": _fin(s.min(), 2), "median": _fin(s.median(), 2), "mean": _fin(s.mean(), 2), "max": _fin(s.max(), 2)}

        calculation_breakdown = [
            {"component": "Average order value", "unit": "currency/order", **stats5(cust['avg_order_value'])},
            {"component": "Purchase frequency", "unit": "orders/year", **stats5(cust['orders_per_year'])},
            {"component": "Tenure (observed)", "unit": "days", **stats5(cust['tenure_days'])},
            {"component": "Historic CLV", "unit": "currency", **stats5(cust['historic_clv'])},
            {"component": "Predictive CLV (heuristic)", "unit": "currency", **stats5(cust['predictive_clv_heuristic'])},
        ]
        formula_note = (
            f"Historic CLV = total revenue x margin ({margin_pct:.0f}%). "
            f"Predictive CLV (heuristic) = avg order value x orders/year x lifespan "
            f"({lifespan_years:.1f}yr) x margin. Both are shown per-customer in the "
            f"CLV by Customer table below; this table summarises the distribution of "
            f"each input component across the customer base."
        )

        # ══════════════════════ Section 3: CLV Distribution ══════════════════════
        pred = cust['predictive_clv_heuristic'].values
        chart_hist = None
        try:
            fig, ax = plt.subplots(figsize=(6.5, 4.4), dpi=110)
            ax.hist(pred, bins=min(30, max(5, n_customers // 10 or 5)), color=BLUE, edgecolor='white', alpha=0.85)
            ax.set_title("Predictive CLV Distribution")
            ax.set_xlabel("Predictive CLV")
            ax.set_ylabel("Customers")
            ax.grid(alpha=0.2, axis='y')
            fig.tight_layout()
            chart_hist = _png(fig)
        except Exception:
            plt.close('all')

        sorted_desc = np.sort(pred)[::-1]
        cum = np.cumsum(sorted_desc)
        total_pred = cum[-1] if len(cum) and cum[-1] != 0 else 1.0
        cum_share = cum / total_pred
        pareto_x = (np.arange(1, len(cum) + 1)) / len(cum)
        top20_idx = max(0, int(0.2 * len(cum)) - 1)
        top20_share = float(cum_share[top20_idx]) if len(cum_share) else 0.0

        chart_pareto = None
        try:
            fig, ax = plt.subplots(figsize=(6.5, 4.4), dpi=110)
            ax.plot(pareto_x * 100, cum_share * 100, color=GREEN, linewidth=2.2)
            ax.plot([0, 100], [0, 100], color='#94a3b8', linestyle='--', linewidth=1.2)
            ax.axvline(20, color=RED, linestyle=':', linewidth=1)
            ax.set_title("Cumulative Predictive CLV (Pareto)")
            ax.set_xlabel("% of customers (top-value first)")
            ax.set_ylabel("% of total predicted value")
            ax.grid(alpha=0.2)
            fig.tight_layout()
            chart_pareto = _png(fig)
        except Exception:
            plt.close('all')

        distribution = {
            "top20_share_pct": _fin(top20_share * 100, 2),
            "note": f"Top 20% of customers hold {top20_share * 100:.0f}% of total predicted CLV.",
        }

        # ══════════════════════ Section 4: CLV by Customer ══════════════════════
        cust_reset = cust.reset_index().rename(columns={customer_id_col: 'customer_id'})
        cust_reset = cust_reset.sort_values('predictive_clv_heuristic', ascending=False)
        detail_cap = 500
        customer_table_full = []
        for _, row in cust_reset.iterrows():
            customer_table_full.append({
                "customer_id": str(row['customer_id']),
                "historic_clv": _fin(row['historic_clv'], 2),
                "predictive_clv": _fin(row['predictive_clv_heuristic'], 2),
                "orders": int(row['orders']),
                "avg_order_value": _fin(row['avg_order_value'], 2),
                "orders_per_year": _fin(row['orders_per_year'], 3),
                "tenure_days": int(row['tenure_days']),
            })
        customer_table_note = None
        if len(customer_table_full) > detail_cap:
            customer_table = customer_table_full[:detail_cap]
            customer_table_note = f"Showing top {detail_cap} of {len(customer_table_full)} customers by predictive CLV (preview)."
        else:
            customer_table = customer_table_full

        # ══════════════════════ lifetimes-based probabilistic models ══════════════════════
        model_comparison = []
        model_error_note = None
        cust_model = None  # frequency/recency/T/monetary_value + model CLV, indexed like cust
        bgf = None
        ggf_model = None
        try:
            from lifetimes.utils import summary_data_from_transaction_data
            from lifetimes import BetaGeoFitter, ParetoNBDFitter, GammaGammaFitter

            summary = summary_data_from_transaction_data(
                df, customer_id_col, invoice_date_col, monetary_value_col=amount_col,
                observation_period_end=snapshot_date, freq='D',
            )
            # Gamma-Gamma requires monetary_value > 0 and frequency > 0 (repeat purchasers)
            summary_pos = summary[(summary['frequency'] > 0) & (summary['monetary_value'] > 0)]

            bgf = BetaGeoFitter(penalizer_coef=0.01)
            bgf.fit(summary['frequency'], summary['recency'], summary['T'])

            horizon_days = lifespan_years * 365.0

            try:
                pnbf = ParetoNBDFitter(penalizer_coef=0.01)
                pnbf.fit(summary['frequency'], summary['recency'], summary['T'])
                pnbf_expected_txn = pnbf.conditional_expected_number_of_purchases_up_to_time(
                    horizon_days, summary['frequency'], summary['recency'], summary['T'])
            except Exception:
                pnbf = None
                pnbf_expected_txn = None

            ggf_available = len(summary_pos) >= 10
            if ggf_available:
                ggf_model = GammaGammaFitter(penalizer_coef=0.01)
                ggf_model.fit(summary_pos['frequency'], summary_pos['monetary_value'])
                summary['predicted_avg_value'] = np.nan
                summary.loc[summary_pos.index, 'predicted_avg_value'] = ggf_model.conditional_expected_average_profit(
                    summary_pos['frequency'], summary_pos['monetary_value'])

                clv_bgnbd_gg = ggf_model.customer_lifetime_value(
                    bgf, summary_pos['frequency'], summary_pos['recency'], summary_pos['T'],
                    summary_pos['monetary_value'], time=lifespan_years * 12, freq='D', discount_rate=0.01,
                ) * margin
                summary['model_clv'] = np.nan
                summary.loc[clv_bgnbd_gg.index, 'model_clv'] = clv_bgnbd_gg
            else:
                summary['predicted_avg_value'] = np.nan
                summary['model_clv'] = np.nan

            bgf_expected_txn = bgf.conditional_expected_number_of_purchases_up_to_time(
                horizon_days, summary['frequency'], summary['recency'], summary['T'])
            summary['bgf_expected_txn'] = bgf_expected_txn
            if pnbf_expected_txn is not None:
                summary['pnbf_expected_txn'] = pnbf_expected_txn

            cust_model = summary.copy()
            cust_model.index.name = 'customer_id'
            cust_model.index = cust_model.index.astype(str)

            avg_hist = float(cust['historic_clv'].mean())
            avg_pred_heuristic = float(cust['predictive_clv_heuristic'].mean())
            avg_model_clv = _fin(summary['model_clv'].mean(), 2) if ggf_available else None
            avg_bgf_txn = _fin(summary['bgf_expected_txn'].mean(), 3)
            avg_pnbf_txn = _fin(summary['pnbf_expected_txn'].mean(), 3) if pnbf_expected_txn is not None else None

            model_comparison = [
                {"model": "Historical (simple heuristic)", "estimate": _fin(avg_pred_heuristic, 2),
                 "metric": "avg predictive CLV", "description": "Avg order value x purchase frequency x expected lifespan x margin — deterministic, no statistical fit."},
                {"model": "BG/NBD", "estimate": avg_bgf_txn,
                 "metric": "avg expected transactions (horizon)", "description": "Beta-Geometric/NBD models purchase counts and each customer's probability of still being 'alive'; used as the transaction-count input to Gamma-Gamma CLV below."},
                {"model": "Pareto/NBD", "estimate": avg_pnbf_txn,
                 "metric": "avg expected transactions (horizon)", "description": "Alternative dropout-process model (continuous-time churn instead of BG/NBD's discrete geometric dropout). Compare to BG/NBD to see how sensitive the transaction forecast is to the churn assumption." if avg_pnbf_txn is not None else "Could not be fit on this data (see note)."},
                {"model": "Gamma-Gamma + BG/NBD", "estimate": avg_model_clv,
                 "metric": "avg predictive CLV" if avg_model_clv is not None else "unavailable",
                 "description": "Gamma-Gamma models each customer's average monetary value per transaction; combined with BG/NBD's expected transaction count to give a fully probabilistic predictive CLV." if ggf_available else "Requires at least 10 repeat customers (frequency > 0) with positive monetary value; not enough repeat purchasers in this dataset."},
                {"model": "ML (Model Lab)", "estimate": None, "metric": "n/a",
                 "description": "Placeholder — a trained ML regressor for CLV connects to the separate Model Lab feature and is not computed on this page."},
            ]
        except Exception as e:
            model_error_note = f"Probabilistic models (BG/NBD, Pareto/NBD, Gamma-Gamma) could not be fit: {e.__class__.__name__}: {e}"
            model_comparison = [
                {"model": "Historical (simple heuristic)", "estimate": _fin(cust['predictive_clv_heuristic'].mean(), 2),
                 "metric": "avg predictive CLV", "description": "Avg order value x purchase frequency x expected lifespan x margin — deterministic, no statistical fit."},
                {"model": "BG/NBD", "estimate": None, "metric": "unavailable", "description": model_error_note},
                {"model": "Pareto/NBD", "estimate": None, "metric": "unavailable", "description": model_error_note},
                {"model": "Gamma-Gamma + BG/NBD", "estimate": None, "metric": "unavailable", "description": model_error_note},
                {"model": "ML (Model Lab)", "estimate": None, "metric": "n/a",
                 "description": "Placeholder — a trained ML regressor for CLV connects to the separate Model Lab feature and is not computed on this page."},
            ]

        # ══════════════════════ Section 6: Predicted vs Actual CLV ══════════════════════
        predicted_vs_actual = None
        predicted_vs_actual_note = None
        chart_pred_vs_actual = None
        span_days = (snapshot_date - df[invoice_date_col].min()).days
        if cust_model is not None and 'model_clv' in cust_model.columns and span_days >= 90:
            try:
                from lifetimes.utils import calibration_and_holdout_data
                from lifetimes import BetaGeoFitter, GammaGammaFitter

                cal_end = df[invoice_date_col].min() + pd.Timedelta(days=int(span_days * 0.7))
                cal_hold = calibration_and_holdout_data(
                    df, customer_id_col, invoice_date_col,
                    calibration_period_end=cal_end, observation_period_end=snapshot_date,
                    freq='D', monetary_value_col=amount_col,
                )
                if len(cal_hold) >= 10:
                    bgf_cal = BetaGeoFitter(penalizer_coef=0.01)
                    bgf_cal.fit(cal_hold['frequency_cal'], cal_hold['recency_cal'], cal_hold['T_cal'])
                    cal_hold['predicted_holdout_txn'] = bgf_cal.predict(
                        cal_hold['duration_holdout'], cal_hold['frequency_cal'], cal_hold['recency_cal'], cal_hold['T_cal'])
                    # actual realised value in holdout vs predicted expected value (using calibration avg order value as proxy $/txn)
                    avg_val_cal = cal_hold['monetary_value_cal'].replace(0, np.nan)
                    cal_hold['predicted_holdout_value'] = (cal_hold['predicted_holdout_txn'] * avg_val_cal * margin).fillna(0)
                    cal_hold['actual_holdout_value'] = (cal_hold['frequency_holdout'] * avg_val_cal * margin).fillna(0)
                    sample = cal_hold.sample(n=min(300, len(cal_hold)), random_state=42) if len(cal_hold) > 300 else cal_hold
                    scatter_pts = [{"predicted": _fin(rw.predicted_holdout_value, 2), "actual": _fin(rw.actual_holdout_value, 2)}
                                   for rw in sample.itertuples()]
                    mae = float((cal_hold['predicted_holdout_value'] - cal_hold['actual_holdout_value']).abs().mean())
                    corr = float(np.corrcoef(cal_hold['predicted_holdout_value'], cal_hold['actual_holdout_value'])[0, 1]) if cal_hold['predicted_holdout_value'].std() > 0 and cal_hold['actual_holdout_value'].std() > 0 else None
                    predicted_vs_actual = {
                        "method": "train/holdout split",
                        "calibration_end": cal_end.isoformat(),
                        "mae": _fin(mae, 2),
                        "correlation": _fin(corr, 3) if corr is not None else None,
                        "points": scatter_pts,
                    }
                    predicted_vs_actual_note = (
                        f"BG/NBD was fit on the first 70% of the observed time span (calibration period, "
                        f"ending {cal_end.date().isoformat()}) and used to predict each customer's holdout-period "
                        f"value; compared against their actual realised value in the remaining 30%."
                    )
                    try:
                        fig, ax = plt.subplots(figsize=(6, 6), dpi=110)
                        xs = [p['predicted'] for p in scatter_pts]
                        ys = [p['actual'] for p in scatter_pts]
                        ax.scatter(xs, ys, alpha=0.5, color=PURPLE, s=18)
                        lims = [0, max(xs + ys + [1])]
                        ax.plot(lims, lims, color='#94a3b8', linestyle='--', linewidth=1.2)
                        ax.set_xlabel("Predicted holdout value")
                        ax.set_ylabel("Actual holdout value")
                        ax.set_title("Predicted vs Actual CLV (holdout)")
                        ax.grid(alpha=0.2)
                        fig.tight_layout()
                        chart_pred_vs_actual = _png(fig)
                    except Exception:
                        plt.close('all')
                else:
                    raise ValueError("Not enough customers with sufficient holdout history.")
            except Exception as e:
                predicted_vs_actual = None
                predicted_vs_actual_note = None

        if predicted_vs_actual is None:
            # Fallback: proxy comparison of model-based predictive CLV vs simple heuristic
            predicted_vs_actual_note = (
                "A clean train/holdout time split was not feasible for this dataset (insufficient date span or "
                "repeat-purchase history for a reliable calibration/holdout fit). As a documented proxy, this "
                "compares the Gamma-Gamma + BG/NBD model-based predictive CLV against the simple historic-CLV "
                "heuristic already used elsewhere on this page — not a true predicted-vs-realised-future comparison."
            )
            if cust_model is not None and 'model_clv' in cust_model.columns:
                joined = cust.join(cust_model[['model_clv']], how='inner')
                joined = joined.dropna(subset=['model_clv'])
                if not joined.empty:
                    sample = joined.sample(n=min(300, len(joined)), random_state=42) if len(joined) > 300 else joined
                    scatter_pts = [{"predicted": _fin(rw.model_clv, 2), "actual": _fin(rw.historic_clv, 2)} for rw in sample.itertuples()]
                    predicted_vs_actual = {"method": "proxy (model CLV vs historic heuristic)", "points": scatter_pts, "mae": None, "correlation": None}
                    try:
                        fig, ax = plt.subplots(figsize=(6, 6), dpi=110)
                        xs = [p['predicted'] for p in scatter_pts]
                        ys = [p['actual'] for p in scatter_pts]
                        ax.scatter(xs, ys, alpha=0.5, color=PURPLE, s=18)
                        lims = [0, max(xs + ys + [1])]
                        ax.plot(lims, lims, color='#94a3b8', linestyle='--', linewidth=1.2)
                        ax.set_xlabel("Model-based predictive CLV (Gamma-Gamma + BG/NBD)")
                        ax.set_ylabel("Historic CLV (heuristic, to date)")
                        ax.set_title("Model CLV vs Historic CLV (proxy)")
                        ax.grid(alpha=0.2)
                        fig.tight_layout()
                        chart_pred_vs_actual = _png(fig)
                    except Exception:
                        plt.close('all')

        # ══════════════════════ Section 7: CLV Forecast ══════════════════════
        forecast = None
        chart_forecast = None
        if bgf is not None and ggf_model is not None and cust_model is not None:
            try:
                summary_pos2 = cust_model[(cust_model['frequency'] > 0) & (cust_model['monetary_value'] > 0)]
                horizons_months = [1, 2, 3, 6, 9, 12, 18, 24]
                forecast_points = []
                for hm in horizons_months:
                    clv_h = ggf_model.customer_lifetime_value(
                        bgf, summary_pos2['frequency'], summary_pos2['recency'], summary_pos2['T'],
                        summary_pos2['monetary_value'], time=hm, freq='D', discount_rate=0.01,
                    ) * margin
                    forecast_points.append({"months": hm, "cumulative_clv": _fin(float(clv_h.sum()), 2)})
                forecast = {"points": forecast_points, "note": f"Cumulative portfolio CLV projected using Gamma-Gamma + BG/NBD over each horizon (0.01 monthly discount rate applied)."}
                try:
                    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=110)
                    xs = [p['months'] for p in forecast_points]
                    ys = [p['cumulative_clv'] for p in forecast_points]
                    ax.plot(xs, ys, marker='o', color=GREEN, linewidth=2.2)
                    ax.set_xlabel("Months ahead")
                    ax.set_ylabel("Cumulative predicted CLV (portfolio)")
                    ax.set_title("CLV Forecast (Gamma-Gamma + BG/NBD)")
                    ax.grid(alpha=0.2)
                    fig.tight_layout()
                    chart_forecast = _png(fig)
                except Exception:
                    plt.close('all')
            except Exception as e:
                forecast = None

        forecast_note = None
        if forecast is None:
            forecast_note = "CLV forecast requires the Gamma-Gamma + BG/NBD model to be available (see CLV Model Comparison); it could not be fit on this dataset."

        # ══════════════════════ Section 8: CLV by Acquisition Channel (conditional) ══════════════════════
        channel_table = None
        channel_note = None
        chart_channel = None
        if channel_col and channel_col in df.columns and 'channel' in cust.columns:
            try:
                by_chan = cust.dropna(subset=['channel']).groupby('channel')
                rows = []
                for chan, g in by_chan:
                    rows.append({
                        "channel": str(chan),
                        "customers": int(len(g)),
                        "avg_historic_clv": _fin(g['historic_clv'].mean(), 2),
                        "avg_predictive_clv": _fin(g['predictive_clv_heuristic'].mean(), 2),
                        "total_revenue": _fin(g['total_revenue'].sum(), 2),
                    })
                rows.sort(key=lambda r: r['avg_predictive_clv'] or 0, reverse=True)
                channel_table = rows
                try:
                    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=110)
                    chans = [r['channel'] for r in rows]
                    vals = [r['avg_predictive_clv'] or 0 for r in rows]
                    ax.barh(chans, vals, color=AMBER)
                    ax.set_xlabel("Avg predictive CLV")
                    ax.set_title("CLV by Acquisition Channel")
                    ax.grid(alpha=0.2, axis='x')
                    fig.tight_layout()
                    chart_channel = _png(fig)
                except Exception:
                    plt.close('all')
            except Exception as e:
                channel_note = f"Could not compute channel breakdown: {e}"
        else:
            channel_note = "No acquisition-channel column was provided — this section is skipped. Provide a categorical channel column to see CLV broken down by acquisition source."

        # ══════════════════════ Section 9: CLV vs CAC (conditional) ══════════════════════
        cac_table = None
        cac_note = None
        if cac_input is not None:
            try:
                if isinstance(cac_input, dict):
                    if channel_table is None:
                        cac_note = "Per-channel CAC was provided but no acquisition-channel column was supplied, so CLV:CAC cannot be joined by channel."
                    else:
                        rows = []
                        for r in channel_table:
                            chan = r['channel']
                            cac_val = cac_input.get(chan)
                            if cac_val is None:
                                continue
                            cac_val = float(cac_val)
                            ratio = (r['avg_predictive_clv'] / cac_val) if cac_val > 0 else None
                            payback_years = (cac_val / (r['avg_predictive_clv'] / lifespan_years)) if r['avg_predictive_clv'] else None
                            rows.append({
                                "channel": chan, "cac": _fin(cac_val, 2),
                                "avg_predictive_clv": r['avg_predictive_clv'],
                                "clv_cac_ratio": _fin(ratio, 2) if ratio is not None else None,
                                "payback_years": _fin(payback_years, 2) if payback_years else None,
                            })
                        cac_table = rows
                else:
                    cac_val = float(cac_input)
                    avg_pred = overview['avg_predictive_clv'] or 0
                    ratio = (avg_pred / cac_val) if cac_val > 0 else None
                    payback_years = (cac_val / (avg_pred / lifespan_years)) if avg_pred else None
                    cac_table = [{
                        "channel": "Overall", "cac": _fin(cac_val, 2),
                        "avg_predictive_clv": _fin(avg_pred, 2),
                        "clv_cac_ratio": _fin(ratio, 2) if ratio is not None else None,
                        "payback_years": _fin(payback_years, 2) if payback_years else None,
                    }]
            except Exception as e:
                cac_note = f"Could not compute CLV:CAC — {e}"
        else:
            cac_note = "No customer acquisition cost (CAC) was provided — this section is skipped. Provide an overall CAC (or per-channel CAC alongside a channel column) to see CLV:CAC ratio and payback period."

        # ══════════════════════ Section 10: CLV Value Tier (model-based, terciles) ══════════════════════
        tier_source = "heuristic predictive CLV (model-based CLV unavailable)"
        tier_series = cust['predictive_clv_heuristic']
        if cust_model is not None and 'model_clv' in cust_model.columns and cust_model['model_clv'].notna().sum() >= 10:
            tier_series = cust_model['model_clv'].dropna()
            tier_source = "Gamma-Gamma + BG/NBD model-based predictive CLV"

        n_tier = len(tier_series)
        sorted_tier = tier_series.sort_values(ascending=False)
        t1 = int(np.ceil(n_tier / 3))
        t2 = int(np.ceil(2 * n_tier / 3))
        tier_labels = np.array(['High'] * t1 + ['Medium'] * (t2 - t1) + ['Low'] * (n_tier - t2))
        total_tier_value = float(sorted_tier.sum()) or 1.0
        value_tier_table = []
        for label in ['High', 'Medium', 'Low']:
            mask = tier_labels == label
            vals = sorted_tier.values[mask]
            value_tier_table.append({
                "tier": label,
                "customers": int(mask.sum()),
                "total_value": _fin(float(vals.sum()), 2),
                "value_share_pct": _fin(float(vals.sum()) / total_tier_value * 100, 2),
                "avg_value": _fin(float(vals.mean()) if len(vals) else 0, 2),
            })
        value_tier_note = f"Tiers computed on {tier_source}."

        charts = {
            "clv_hist": chart_hist,
            "pareto": chart_pareto,
            "pred_vs_actual": chart_pred_vs_actual,
            "forecast": chart_forecast,
            "channel": chart_channel,
        }
        charts = {k: v for k, v in charts.items() if v is not None}

        results = {
            "status": "ok",
            "n_customers": n_customers,
            "n_transactions": n_transactions,
            "overview": overview,
            "calculation_breakdown": calculation_breakdown,
            "formula_note": formula_note,
            "distribution": distribution,
            "customer_table": customer_table,
            "customer_table_note": customer_table_note,
            "model_comparison": model_comparison,
            "model_error_note": model_error_note,
            "predicted_vs_actual": predicted_vs_actual,
            "predicted_vs_actual_note": predicted_vs_actual_note,
            "forecast": forecast,
            "forecast_note": forecast_note,
            "channel_table": channel_table,
            "channel_note": channel_note,
            "cac_table": cac_table,
            "cac_note": cac_note,
            "value_tier_table": value_tier_table,
            "value_tier_note": value_tier_note,
            "charts": charts,
        }

        final_output = {
            'plot': charts.get('clv_hist'),
            'results': results,
        }
        print(json.dumps(final_output))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
