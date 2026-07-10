

import sys
import json
import numpy as np
import pandas as pd
import pingouin as pg
import math
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

def _to_native_type(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.bool_):
        return bool(obj)
    return obj

def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get('data')
        variables = payload.get('variables')
        control_vars = payload.get('controlVars')
        method = payload.get('method', 'pearson')

        if not data or not variables:
            raise ValueError("Missing 'data' or 'variables'")

        df = pd.DataFrame(data)
        
        all_cols = list(set(variables + (control_vars or [])))
        df_clean = df[all_cols].copy()
        for col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
        df_clean.dropna(inplace=True)
        
        if df_clean.shape[0] < 2:
            raise ValueError("Not enough valid data points for analysis.")

        # Perform partial correlation
        pcorr_matrix = pg.pcorr(df_clean, method=method, x=variables, y=variables, covar=control_vars).set_index('X')

        # Since pcorr gives a matrix of X vs Y, we need to reformat it into a square matrix
        # and extract the p-values and other stats if needed.
        # For simplicity, we will just return the main correlation matrix for now.
        
        # We'll re-run for p-values (this is not efficient but simpler to implement)
        p_values = pcorr_matrix.copy()
        for x in variables:
            for y in variables:
                if x == y:
                    p_values.loc[x, y] = 1.0
                else:
                    res = pg.pcorr(df_clean, x=x, y=y, covar=control_vars, method=method)
                    p_values.loc[x, y] = res['p-val'].iloc[0]

        # Generate Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(pcorr_matrix, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
        plt.title(f'Partial Correlation Matrix ({method.capitalize()})', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        heatmap_plot_img = base64.b64encode(buf.read()).decode('utf-8')

        response = {
            "correlation_matrix": pcorr_matrix.to_dict(),
            "p_value_matrix": p_values.to_dict(),
            "heatmap_plot": f"data:image/png;base64,{heatmap_plot_img}",
        }

        print(json.dumps(response, default=_to_native_type, ensure_ascii=False))

    except Exception as e:
        error_response = { "error": str(e), "error_type": type(e).__name__ }
        print(json.dumps(error_response, default=_to_native_type), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
