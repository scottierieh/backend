import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

def _to_native_type(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif pd.api.types.is_categorical_dtype(obj):
        return int(obj)
    return obj

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
                # Use rank to handle duplicates
                ranked = series.rank(method='first', ascending=ascending)
                # Use qcut on ranks
                scores = pd.qcut(ranked, q=5, labels=[1, 2, 3, 4, 5], duplicates='drop')
                # Convert categorical to int
                return scores.astype(int)
            except Exception as e:
                # Fallback: use percentile-based binning
                percentiles = [0, 20, 40, 60, 80, 100]
                bins = np.percentile(series, percentiles)
                bins = np.unique(bins)  # Remove duplicates
                if len(bins) < 2:
                    return pd.Series([3] * len(series), index=series.index)
                scores = pd.cut(series, bins=bins, labels=False, include_lowest=True, duplicates='drop')
                if scores is None or scores.isna().all():
                    return pd.Series([3] * len(series), index=series.index)
                # Normalize to 1-5 scale
                min_score = scores.min()
                max_score = scores.max()
                if max_score == min_score:
                    return pd.Series([3] * len(series), index=series.index)
                normalized = ((scores - min_score) / (max_score - min_score) * 4 + 1).round()
                if not ascending:
                    normalized = 6 - normalized
                return normalized.astype(int)
        
        # Score each metric
        rfm['R_Score'] = score_rfm_column(rfm['Recency'], ascending=True)  # Lower recency = better
        rfm['R_Score'] = 6 - rfm['R_Score']  # Invert so 5 is best
        rfm['F_Score'] = score_rfm_column(rfm['Frequency'], ascending=True)
        rfm['M_Score'] = score_rfm_column(rfm['Monetary'], ascending=True)
        
        # Ensure all scores are in 1-5 range
        rfm['R_Score'] = rfm['R_Score'].clip(1, 5).astype(int)
        rfm['F_Score'] = rfm['F_Score'].clip(1, 5).astype(int)
        rfm['M_Score'] = rfm['M_Score'].clip(1, 5).astype(int)
        
        rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)
        
        # --- Segmentation ---
        rfm = get_rfm_segments(rfm)
        
        segment_counts = rfm['Segment'].value_counts().reset_index()
        segment_counts.columns = ['Segment', 'Count']
        
        # --- Plotting ---
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Segment Distribution Bar Chart
        sns.barplot(data=segment_counts, y='Segment', x='Count', ax=axes[0], palette='viridis')
        axes[0].set_title('Customer Segment Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Number of Customers')
        axes[0].set_ylabel('Segment')

        # RFM Treemap
        squarify_df = rfm.groupby('Segment')['Monetary'].sum().reset_index()
        squarify_df = squarify_df[squarify_df['Monetary'] > 0]  # Remove zero values
        
        if not squarify_df.empty:
            import squarify
            sizes = squarify_df['Monetary'].values
            labels = [f'{row.Segment}\n${row.Monetary:,.0f}' for _, row in squarify_df.iterrows()]
            colors = [plt.cm.viridis(i/float(len(sizes))) for i in range(len(sizes))]
            
            squarify.plot(sizes=sizes, label=labels, ax=axes[1], alpha=0.8, color=colors, text_kwargs={'fontsize': 9})
            axes[1].set_title('Segment Value (Monetary)', fontsize=14, fontweight='bold')
            axes[1].axis('off')
        else:
            axes[1].text(0.5, 0.5, 'No data to display', ha='center', va='center', fontsize=12)
            axes[1].axis('off')
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plot_image = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)

        # --- Final Results ---
        rfm_reset = rfm.reset_index()
        
        # Convert all data to JSON-serializable types
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
        
        results = {
            'rfm_data': rfm_data_clean,
            'segment_distribution': segment_dist_clean,
            'plot': f"data:image/png;base64,{plot_image}",
            'customer_id_col': customer_id_col
        }
        
        print(json.dumps(results))

    except Exception as e:
        error_msg = str(e)
        print(json.dumps({"error": error_msg}), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    try:
        import squarify
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "squarify", "--break-system-packages"])
    
    main()

    