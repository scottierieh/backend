#!/usr/bin/env python3
"""RFM Segmentation — Recency/Frequency/Monetary scoring & segment classification.
pandas / numpy / matplotlib.

Aggregates transaction-level rows to one row per customer, scores each of
R/F/M on a 1-5 scale (rank-based quintiles), assigns a standard RFM segment
label, and builds a full step-6 report with 8 additive sections:
  1. RFM Overview (KPI cards)
  2. RFM Score Distribution (table + 3 histograms)
  3. RFM Segment Distribution (table + bar chart)
  4. RFM Segment Matrix (R x F heatmap)
  5. RFM Customer Map (3D R/F/M scatter, or heatmap fallback)
  6. Segment Profile (table)
  7. Segment Migration (conditional — skipped, single-snapshot input only)
  8. Customer RFM Detail (table, capped preview if very large)

Scope note: this analysis is intentionally limited to classifying customers
by past R/F/M behaviour into named segments. It does NOT compute customer
lifetime value, churn probability, general multivariate clustering, or
revenue "what-if" projections — those live in separate analyses.

Input (from rfm-segmentation-page.tsx):
    data              : list[dict]
    customer_id_col   : str
    invoice_date_col  : str
    unit_price_col    : str
    quantity_col      : str
Output: { rfm_data, segment_distribution, plot, customer_id_col, results: {...} }
"""
import sys
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import warnings

warnings.filterwarnings('ignore')

BLUE = "#2563eb"
GREEN = "#16a34a"
RED = "#dc2626"
AMBER = "#d97706"
PURPLE = "#9333ea"
LIGHT_BLUE = "#93c5fd"


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


def get_rfm_segments(df):
    # Standard segmentation based on quintiles of R, F, and M scores
    segment_map = {
        r'555': 'Champions',
        r'[4-5][4-5][1-5]': 'Loyal Customers',
        r'[3-5]3[1-5]': 'Potential Loyalists',
        r'5[1-2][1-5]': 'New Customers',
        r'4[1-2][1-5]': 'Promising',
        r'3[1-2][1-5]': 'Needs Attention',
        r'2[1-5][1-5]': 'At Risk',
        r'1[3-5][1-5]': "Can't Lose Them",
        r'1[1-2][1-5]': 'Hibernating',
        r'111': 'Lost'
    }

    df['Segment'] = 'Others'
    for pattern, segment in segment_map.items():
        mask = df['RFM_Score'].str.match(pattern)
        df.loc[mask, 'Segment'] = segment

    return df


def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get('data')
        customer_id_col = payload.get('customer_id_col')
        invoice_date_col = payload.get('invoice_date_col')
        unit_price_col = payload.get('unit_price_col')
        quantity_col = payload.get('quantity_col')

        if not all([data, customer_id_col, invoice_date_col, unit_price_col, quantity_col]):
            raise ValueError("Missing data or required column names.")

        df = pd.DataFrame(data)

        # --- Data Cleaning and Preparation ---
        df[invoice_date_col] = pd.to_datetime(df[invoice_date_col], errors='coerce')
        df[unit_price_col] = pd.to_numeric(df[unit_price_col], errors='coerce')
        df[quantity_col] = pd.to_numeric(df[quantity_col], errors='coerce')

        # Calculate total amount
        df['total_amount'] = df[unit_price_col] * df[quantity_col]

        df.dropna(subset=[customer_id_col, invoice_date_col, 'total_amount'], inplace=True)
        df = df[df['total_amount'] > 0]

        if df.empty:
            raise ValueError("No valid data for RFM analysis after cleaning.")

        n_transactions = int(len(df))

        # --- RFM Calculation ---
        snapshot_date = df[invoice_date_col].max() + pd.DateOffset(days=1)

        rfm = df.groupby(customer_id_col).agg({
            invoice_date_col: lambda date: (snapshot_date - date.max()).days,
            customer_id_col: 'count',
            'total_amount': 'sum'
        })

        rfm.rename(columns={
            invoice_date_col: 'Recency',
            customer_id_col: 'Frequency',
            'total_amount': 'Monetary'
        }, inplace=True)

        # --- RFM Scoring - SIMPLIFIED AND ROBUST ---
        def score_rfm_column(series, ascending=True):
            """Create 1-5 scores using rank-based approach"""
            try:
                ranked = series.rank(method='first', ascending=ascending)
                scores = pd.qcut(ranked, q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
                return scores.astype(int)
            except Exception:
                percentiles = [0, 20, 40, 60, 80, 100]
                bins = np.percentile(series, percentiles)
                bins = np.unique(bins)
                if len(bins) < 2:
                    return pd.Series([3] * len(series), index=series.index)
                scores = pd.cut(series, bins=bins, labels=False, include_lowest=True, duplicates='drop')
                if scores is None or scores.isna().all():
                    return pd.Series([3] * len(series), index=series.index)
                min_score = scores.min()
                max_score = scores.max()
                if max_score == min_score:
                    return pd.Series([3] * len(series), index=series.index)
                normalized = ((scores - min_score) / (max_score - min_score) * 4 + 1).round()
                if not ascending:
                    normalized = 6 - normalized
                return normalized.astype(int)

        rfm['R_Score'] = score_rfm_column(rfm['Recency'], ascending=True)  # Lower recency = better
        rfm['R_Score'] = 6 - rfm['R_Score']  # Invert so 5 is best
        rfm['F_Score'] = score_rfm_column(rfm['Frequency'], ascending=True)
        rfm['M_Score'] = score_rfm_column(rfm['Monetary'], ascending=True)

        rfm['R_Score'] = rfm['R_Score'].clip(1, 5).astype(int)
        rfm['F_Score'] = rfm['F_Score'].clip(1, 5).astype(int)
        rfm['M_Score'] = rfm['M_Score'].clip(1, 5).astype(int)

        rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

        # --- Segmentation ---
        rfm = get_rfm_segments(rfm)

        segment_counts = rfm['Segment'].value_counts().reset_index()
        segment_counts.columns = ['Segment', 'Count']

        n_customers = int(len(rfm))

        # --- Original combined plot (kept intact — other frontend parts may rely on it) ---
        plot_image = None
        try:
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            sns.barplot(data=segment_counts, y='Segment', x='Count', ax=axes[0], palette='viridis')
            axes[0].set_title('Customer Segment Distribution', fontsize=14, fontweight='bold')
            axes[0].set_xlabel('Number of Customers')
            axes[0].set_ylabel('Segment')

            squarify_df = rfm.groupby('Segment')['Monetary'].sum().reset_index()
            squarify_df = squarify_df[squarify_df['Monetary'] > 0]

            if not squarify_df.empty:
                try:
                    import squarify
                    sizes = squarify_df['Monetary'].values
                    labels = [f'{row.Segment}\n${row.Monetary:,.0f}' for _, row in squarify_df.iterrows()]
                    colors = [plt.cm.viridis(i / float(len(sizes))) for i in range(len(sizes))]
                    squarify.plot(sizes=sizes, label=labels, ax=axes[1], alpha=0.8, color=colors, text_kwargs={'fontsize': 9})
                    axes[1].set_title('Segment Value (Monetary)', fontsize=14, fontweight='bold')
                    axes[1].axis('off')
                except ImportError:
                    axes[1].text(0.5, 0.5, 'squarify not installed', ha='center', va='center', fontsize=12)
                    axes[1].axis('off')
            else:
                axes[1].text(0.5, 0.5, 'No data to display', ha='center', va='center', fontsize=12)
                axes[1].axis('off')

            plt.tight_layout()
            plot_image_full = _png(fig)
            plot_image = plot_image_full
        except Exception:
            plt.close('all')
            plot_image = None

        # --- Final Results (legacy fields kept intact) ---
        rfm_reset = rfm.reset_index()

        rfm_data_clean = []
        for _, row in rfm_reset.iterrows():
            row_dict = {}
            for col in rfm_reset.columns:
                val = row[col]
                if pd.isna(val):
                    row_dict[col] = None
                elif isinstance(val, (np.integer, np.int64, np.int32)):
                    row_dict[col] = int(val)
                elif isinstance(val, (np.floating, np.float64, np.float32, float)):
                    if np.isnan(val) or np.isinf(val):
                        row_dict[col] = None
                    else:
                        row_dict[col] = float(val)
                elif isinstance(val, pd.Timestamp):
                    row_dict[col] = val.isoformat()
                else:
                    row_dict[col] = str(val)
            rfm_data_clean.append(row_dict)

        segment_dist_clean = []
        for _, row in segment_counts.iterrows():
            segment_dist_clean.append({
                'Segment': str(row['Segment']),
                'Count': int(row['Count'])
            })

        # ══════════════════════════ Section 1: RFM Overview ══════════════════════════
        total_revenue = float(rfm['Monetary'].sum())
        total_orders = int(rfm['Frequency'].sum())
        overview = {
            "total_customers": n_customers,
            "total_orders": total_orders,
            "total_revenue": _fin(total_revenue, 2),
            "avg_recency": _fin(rfm['Recency'].mean(), 2),
            "avg_frequency": _fin(rfm['Frequency'].mean(), 2),
            "avg_monetary": _fin(rfm['Monetary'].mean(), 2),
        }

        # ══════════════════════════ Section 2: RFM Score Distribution ══════════════════════════
        def score_table(score_col):
            vc = rfm[score_col].value_counts().sort_index()
            out = []
            for score in range(1, 6):
                cnt = int(vc.get(score, 0))
                out.append({
                    "score": score,
                    "customers": cnt,
                    "pct": _fin(cnt / n_customers * 100, 2) if n_customers else None,
                })
            return out

        r_score_table = score_table('R_Score')
        f_score_table = score_table('F_Score')
        m_score_table = score_table('M_Score')

        def score_hist(score_col, title, color):
            try:
                fig, ax = plt.subplots(figsize=(6.5, 4.4), dpi=110)
                counts = rfm[score_col].value_counts().sort_index()
                xs = list(range(1, 6))
                ys = [int(counts.get(s, 0)) for s in xs]
                ax.bar(xs, ys, color=color, width=0.65, edgecolor='white')
                ax.set_xticks(xs)
                ax.set_title(title)
                ax.set_xlabel("Score (1-5)")
                ax.set_ylabel("Customers")
                ax.grid(alpha=0.2, axis='y')
                fig.tight_layout()
                return _png(fig)
            except Exception:
                plt.close('all')
                return None

        chart_r_dist = score_hist('R_Score', 'Recency Score Distribution', BLUE)
        chart_f_dist = score_hist('F_Score', 'Frequency Score Distribution', GREEN)
        chart_m_dist = score_hist('M_Score', 'Monetary Score Distribution', AMBER)

        # ══════════════════════════ Section 3: RFM Segment Distribution ══════════════════════════
        # Table intentionally has only Segment/Customers/% — no revenue column (revenue-by-segment
        # belongs conceptually to CLV, kept out of scope here).
        segment_table = []
        for _, row in segment_counts.iterrows():
            cnt = int(row['Count'])
            segment_table.append({
                "segment": str(row['Segment']),
                "customers": cnt,
                "pct": _fin(cnt / n_customers * 100, 2) if n_customers else None,
            })

        chart_segment_bar = None
        try:
            fig, ax = plt.subplots(figsize=(8.5, 5), dpi=110)
            sc_sorted = segment_counts.sort_values('Count', ascending=True)
            colors = plt.cm.viridis(np.linspace(0.15, 0.9, len(sc_sorted)))
            ax.barh(sc_sorted['Segment'], sc_sorted['Count'], color=colors)
            ax.set_title("Customer Count by Segment")
            ax.set_xlabel("Number of Customers")
            ax.grid(alpha=0.2, axis='x')
            fig.tight_layout()
            chart_segment_bar = _png(fig)
        except Exception:
            plt.close('all')
            chart_segment_bar = None

        # ══════════════════════════ Section 4: RFM Segment Matrix (R x F heatmap) ══════════════════════════
        rf_matrix = np.zeros((5, 5), dtype=int)  # rows = R (1-5), cols = F (1-5)
        for r_score in range(1, 6):
            for f_score in range(1, 6):
                rf_matrix[r_score - 1, f_score - 1] = int(
                    ((rfm['R_Score'] == r_score) & (rfm['F_Score'] == f_score)).sum()
                )

        chart_rf_heatmap = None
        try:
            fig, ax = plt.subplots(figsize=(6.8, 6), dpi=110)
            im = ax.imshow(rf_matrix, cmap='YlOrRd', origin='lower')
            ax.set_xticks(range(5)); ax.set_xticklabels(range(1, 6))
            ax.set_yticks(range(5)); ax.set_yticklabels(range(1, 6))
            ax.set_xlabel("Frequency Score")
            ax.set_ylabel("Recency Score")
            ax.set_title("RFM Segment Matrix (R x F, count = customers, all M)")
            for i in range(5):
                for j in range(5):
                    val = rf_matrix[i, j]
                    ax.text(j, i, str(val), ha='center', va='center',
                            color='white' if val > rf_matrix.max() * 0.5 else 'black', fontsize=9)
            fig.colorbar(im, ax=ax, label='Customers')
            fig.tight_layout()
            chart_rf_heatmap = _png(fig)
        except Exception:
            plt.close('all')
            chart_rf_heatmap = None

        rf_matrix_table = []
        for r_score in range(1, 6):
            row_out = {"r_score": r_score}
            for f_score in range(1, 6):
                row_out[f"f_{f_score}"] = int(rf_matrix[r_score - 1, f_score - 1])
            rf_matrix_table.append(row_out)

        # ══════════════════════════ Section 5: RFM Customer Map (3D scatter, or fallback) ══════════════════════════
        chart_rfm_3d = None
        rfm_3d_note = None
        try:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

            sample_n = min(800, n_customers)
            if n_customers > 2000:
                sample_n = 600
            if n_customers > sample_n:
                sample_df = rfm.sample(n=sample_n, random_state=42)
                rfm_3d_note = f"Sampled {sample_n} of {n_customers} customers for legibility."
            else:
                sample_df = rfm

            fig = plt.figure(figsize=(7.5, 6.5), dpi=110)
            ax = fig.add_subplot(111, projection='3d')
            segments_unique = sample_df['Segment'].unique().tolist()
            palette = plt.cm.tab10(np.linspace(0, 1, max(len(segments_unique), 1)))
            seg_color = {seg: palette[i % len(palette)] for i, seg in enumerate(segments_unique)}
            colors = [seg_color[s] for s in sample_df['Segment']]
            ax.scatter(sample_df['Recency'], sample_df['Frequency'], sample_df['Monetary'],
                       c=colors, s=18, alpha=0.6, edgecolors='none')
            ax.set_xlabel('Recency (days)')
            ax.set_ylabel('Frequency')
            ax.set_zlabel('Monetary')
            title = "RFM Customer Map (3D)"
            if rfm_3d_note:
                title += f"\n({rfm_3d_note})"
            ax.set_title(title, fontsize=10)
            fig.tight_layout()
            chart_rfm_3d = _png(fig)
        except Exception as e:
            plt.close('all')
            chart_rfm_3d = None
            rfm_3d_note = f"3D customer map skipped ({e.__class__.__name__}); showing R x F heatmap instead. See RFM Segment Matrix above."

        # ══════════════════════════ Section 6: Segment Profile ══════════════════════════
        segment_profile = []
        for seg, grp in rfm.groupby('Segment'):
            segment_profile.append({
                "segment": str(seg),
                "customers": int(len(grp)),
                "avg_recency": _fin(grp['Recency'].mean(), 2),
                "avg_frequency": _fin(grp['Frequency'].mean(), 2),
                "avg_monetary": _fin(grp['Monetary'].mean(), 2),
                "typical_r_score": int(round(grp['R_Score'].mean())),
                "typical_f_score": int(round(grp['F_Score'].mean())),
                "typical_m_score": int(round(grp['M_Score'].mean())),
            })
        segment_profile.sort(key=lambda x: x['customers'], reverse=True)

        # ══════════════════════════ Section 7: Segment Migration (conditional) ══════════════════════════
        # This script computes RFM from a single snapshot of transaction data (one
        # recency/frequency/monetary value per customer as of `snapshot_date`). There
        # is no second time window or prior-period input, so period-over-period
        # segment migration cannot be computed in this pass.
        migration_note = (
            "Segment migration requires RFM computed at two or more points in time "
            "(e.g. monthly/quarterly snapshots) so a customer's previous segment can "
            "be compared to their current one. This analysis only receives a single "
            "transaction snapshot, so migration is skipped."
        )
        segment_migration_table = None
        chart_migration = None

        # ══════════════════════════ Section 8: Customer RFM Detail ══════════════════════════
        detail_cap = 500
        customer_detail_full = []
        for cid, row in rfm.iterrows():
            customer_detail_full.append({
                "customer_id": str(cid),
                "recency": _fin(row['Recency'], 2),
                "frequency": int(row['Frequency']),
                "monetary": _fin(row['Monetary'], 2),
                "r_score": int(row['R_Score']),
                "f_score": int(row['F_Score']),
                "m_score": int(row['M_Score']),
                "segment": str(row['Segment']),
            })
        customer_detail_note = None
        if len(customer_detail_full) > detail_cap:
            customer_detail = sorted(customer_detail_full, key=lambda x: x['monetary'] or 0, reverse=True)[:detail_cap]
            customer_detail_note = f"Showing top {detail_cap} of {len(customer_detail_full)} customers by monetary value (preview)."
        else:
            customer_detail = customer_detail_full

        charts = {
            "r_dist": chart_r_dist,
            "f_dist": chart_f_dist,
            "m_dist": chart_m_dist,
            "segment_bar": chart_segment_bar,
            "rf_heatmap": chart_rf_heatmap,
            "rfm_3d": chart_rfm_3d,
        }

        results = {
            "status": "ok",
            "n_customers": n_customers,
            "n_transactions": n_transactions,
            "overview": overview,
            "r_score_table": r_score_table,
            "f_score_table": f_score_table,
            "m_score_table": m_score_table,
            "segment_table": segment_table,
            "rf_matrix": rf_matrix.tolist(),
            "rf_matrix_table": rf_matrix_table,
            "rfm_3d_note": rfm_3d_note,
            "segment_profile": segment_profile,
            "segment_migration_table": segment_migration_table,
            "migration_note": migration_note,
            "customer_detail": customer_detail,
            "customer_detail_note": customer_detail_note,
            "charts": charts,
        }

        final_output = {
            'rfm_data': rfm_data_clean,
            'segment_distribution': segment_dist_clean,
            'plot': f"{plot_image}" if plot_image else None,
            'customer_id_col': customer_id_col,
            'results': results,
        }

        print(json.dumps(final_output))

    except Exception as e:
        error_msg = str(e)
        print(json.dumps({"error": error_msg}), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    try:
        import squarify  # noqa: F401
    except ImportError:
        import subprocess
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "squarify", "--break-system-packages"])
        except Exception:
            pass

    main()
