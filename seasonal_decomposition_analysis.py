import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
import io
import base64
import warnings
from calendar import month_name

warnings.filterwarnings('ignore')

def _to_native_type(obj):
    """Convert numpy/pandas types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        # Handle NaN and Inf values
        if np.isnan(obj):
            return None  # Convert NaN to None for JSON compatibility
        elif np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, np.ndarray):
        return [_to_native_type(item) for item in obj]
    elif isinstance(obj, dict):
        return {k: _to_native_type(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_native_type(item) for item in obj]
    return obj

def safe_variance(data):
    """Calculate variance safely, handling NaN values"""
    clean_data = data[~np.isnan(data)]
    if len(clean_data) == 0:
        return 0
    return np.var(clean_data)

def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get('data')
        time_col = payload.get('timeCol')
        value_col = payload.get('valueCol')
        model = payload.get('model', 'additive')
        period = int(payload.get('period', 12))

        if not all([data, time_col, value_col]):
            raise ValueError("Missing 'data', 'timeCol', or 'valueCol'")

        df = pd.DataFrame(data)
        
        # --- Data Preparation ---
        if time_col not in df.columns or value_col not in df.columns:
            raise ValueError(f"Columns '{time_col}' or '{value_col}' not found.")
            
        df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
        df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
        df = df.dropna(subset=[time_col, value_col]).set_index(time_col).sort_index()

        if len(df) < period * 2:
            raise ValueError(f"Not enough data for the given period. Need at least {period*2} data points, but have {len(df)}.")

        # --- Decomposition ---
        decomposition = seasonal_decompose(df[value_col], model=model, period=period)
        
        # Get components
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        resid = decomposition.resid
        
        # Clean components for variance calculation (remove NaN)
        trend_clean = trend.dropna()
        seasonal_clean = seasonal.dropna()
        resid_clean = resid.dropna()

        # --- Strength & Variance Calculation ---
        # Use clean data for calculations
        if len(resid_clean) > 0 and len(trend_clean) > 0:
            var_resid = np.var(resid_clean)
            # Align the indices before adding
            common_idx = trend_clean.index.intersection(resid_clean.index)
            if len(common_idx) > 0:
                var_trend_resid = np.var(trend_clean[common_idx] + resid_clean[common_idx])
                strength_trend = max(0, 1 - (var_resid / var_trend_resid)) if var_trend_resid > 0 else 0
            else:
                strength_trend = 0
        else:
            var_resid = 0
            strength_trend = 0
            
        if len(seasonal_clean) > 0 and len(resid_clean) > 0:
            # Align the indices before adding
            common_idx = seasonal_clean.index.intersection(resid_clean.index)
            if len(common_idx) > 0:
                var_seasonal_resid = np.var(seasonal_clean[common_idx] + resid_clean[common_idx])
                strength_seasonal = max(0, 1 - (var_resid / var_seasonal_resid)) if var_seasonal_resid > 0 else 0
            else:
                strength_seasonal = 0
        else:
            strength_seasonal = 0
        
        # Calculate variance explained
        total_var = np.var(df[value_col])
        var_explained = {
            'trend': (np.var(trend_clean) / total_var) * 100 if total_var > 0 and len(trend_clean) > 0 else 0,
            'seasonal': (np.var(seasonal_clean) / total_var) * 100 if total_var > 0 and len(seasonal_clean) > 0 else 0,
            'irregular': (np.var(resid_clean) / total_var) * 100 if total_var > 0 and len(resid_clean) > 0 else 0,
        }

        decomposition_summary = [
            {'component': 'Trend', 'strength': strength_trend, 'variance_explained': var_explained['trend']},
            {'component': 'Seasonal', 'strength': strength_seasonal, 'variance_explained': var_explained['seasonal']},
            {'component': 'Residual', 'strength': None, 'variance_explained': var_explained['irregular']}
        ]

        # --- Seasonal Pattern Analysis ---
        seasonal_pattern = []
        if period == 12 and len(seasonal) > 0:  # Monthly data
            # Get full seasonal component (including any NaN)
            seasonal_full = seasonal.copy()
            
            # For monthly pattern, we need at least one full year
            if len(seasonal_full) >= 12:
                # Calculate average for each month across all years
                monthly_values = {}
                for i in range(1, 13):
                    month_data = []
                    for idx in seasonal_full.index:
                        if hasattr(idx, 'month') and idx.month == i:
                            val = seasonal_full[idx]
                            if not np.isnan(val):
                                month_data.append(val)
                    
                    if month_data:
                        monthly_values[i] = np.mean(month_data)
                    else:
                        monthly_values[i] = 1 if model == 'multiplicative' else 0
                
                # Create seasonal pattern
                for i in range(1, 13):
                    month = month_name[i]
                    index_val = monthly_values.get(i, 1 if model == 'multiplicative' else 0)
                    
                    if model == 'multiplicative':
                        deviation = (index_val - 1) * 100 if index_val != 0 else 0
                    else:
                        deviation = index_val
                    
                    seasonal_pattern.append({
                        'month': month,
                        'seasonal_index': float(index_val),
                        'deviation': float(deviation)
                    })
        elif period == 7:  # Weekly data
            # Handle weekly patterns
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            for i in range(period):
                day_values = seasonal[i::period].dropna()
                if len(day_values) > 0:
                    avg_val = day_values.mean()
                    deviation = (avg_val - 1) * 100 if model == 'multiplicative' else avg_val
                    seasonal_pattern.append({
                        'month': days[i] if i < 7 else f'Period {i+1}',
                        'seasonal_index': float(avg_val),
                        'deviation': float(deviation)
                    })

        # --- Plotting ---
        fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
        
        # Original series
        df[value_col].plot(ax=axes[0], title='Original Series', color='blue', linewidth=1)
        axes[0].set_ylabel('Value')
        axes[0].grid(True, alpha=0.3)

        # Trend
        trend.plot(ax=axes[1], title='Trend Component', color='red', linewidth=2)
        axes[1].set_ylabel('Trend')
        axes[1].grid(True, alpha=0.3)
        
        # Seasonal
        seasonal.plot(ax=axes[2], title='Seasonal Component', color='green', linewidth=1)
        axes[2].set_ylabel('Seasonal')
        axes[2].grid(True, alpha=0.3)
        
        # Residuals
        resid.dropna().plot(ax=axes[3], title='Residual Component', marker='o', 
                            linestyle='None', alpha=0.6, color='purple', markersize=4)
        axes[3].set_ylabel('Residual')
        axes[3].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[3].grid(True, alpha=0.3)
        
        # Set common xlabel
        axes[3].set_xlabel('Date')
        
        plt.tight_layout()
        
        # Save plot
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        plot_image = base64.b64encode(buf.read()).decode('utf-8')

        # --- Prepare Results ---
        # Convert series to records with proper NaN handling
        trend_records = []
        seasonal_records = []
        resid_records = []
        
        for idx in df.index:
            # Trend
            trend_val = trend.get(idx, np.nan) if idx in trend.index else np.nan
            trend_records.append({
                time_col: idx.isoformat() if hasattr(idx, 'isoformat') else str(idx),
                'trend': _to_native_type(trend_val)
            })
            
            # Seasonal
            seasonal_val = seasonal.get(idx, np.nan) if idx in seasonal.index else np.nan
            seasonal_records.append({
                time_col: idx.isoformat() if hasattr(idx, 'isoformat') else str(idx),
                'seasonal': _to_native_type(seasonal_val)
            })
            
            # Residual
            resid_val = resid.get(idx, np.nan) if idx in resid.index else np.nan
            resid_records.append({
                time_col: idx.isoformat() if hasattr(idx, 'isoformat') else str(idx),
                'residual': _to_native_type(resid_val)
            })

        results = {
            'decomposition_summary': _to_native_type(decomposition_summary),
            'seasonal_pattern': _to_native_type(seasonal_pattern),
            'trend': trend_records,
            'seasonal': seasonal_records,
            'resid': resid_records,
        }

        response = {
            'results': results,
            'plot': f"data:image/png;base64,{plot_image}"
        }

        # Use custom JSON encoder to handle any remaining special values
        print(json.dumps(response, default=_to_native_type))

    except Exception as e:
        error_response = {"error": str(e)}
        sys.stderr.write(json.dumps(error_response) + "\n")
        sys.exit(1)

if __name__ == '__main__':
    main()

    