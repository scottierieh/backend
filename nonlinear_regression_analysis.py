
import sys
import json
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import io
import base64
import warnings

warnings.filterwarnings('ignore')

def _to_native_type(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, np.floating): return float(obj) if np.isfinite(obj) else None
    elif isinstance(obj, np.ndarray): return obj.tolist()
    return obj

# --- Model Functions ---
def model_exponential(x, a, b):
    return a * np.exp(b * x)

def model_logarithmic(x, a, b):
    return a + b * np.log(x)

def model_power(x, a, b):
    return a * np.power(x, b)
    
def model_sigmoid(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get('data')
        x_col = payload.get('x_col')
        y_col = payload.get('y_col')
        model_type = payload.get('model_type', 'exponential')

        if not all([data, x_col, y_col]):
            raise ValueError("Missing 'data', 'x_col', or 'y_col'")

        df = pd.DataFrame(data).dropna(subset=[x_col, y_col])
        x_data = pd.to_numeric(df[x_col], errors='coerce')
        y_data = pd.to_numeric(df[y_col], errors='coerce')
        
        valid_indices = ~np.isnan(x_data) & ~np.isnan(y_data)
        x_data = x_data[valid_indices]
        y_data = y_data[valid_indices]

        if len(x_data) < 3:
            raise ValueError("Not enough valid data points for analysis.")

        model_map = {
            'exponential': (model_exponential, [1.0, 0.1]),
            'logarithmic': (model_logarithmic, [1.0, 1.0]),
            'power': (model_power, [1.0, 1.0]),
            'sigmoid': (model_sigmoid, [np.max(y_data), 1.0, np.median(x_data)])
        }
        
        if model_type not in model_map:
            raise ValueError(f"Unknown model type: {model_type}")

        model_func, p0 = model_map[model_type]
        
        # Adjust for models that don't support non-positive values
        if model_type in ['logarithmic', 'power'] and np.any(x_data <= 0):
            raise ValueError(f"{model_type.capitalize()} model requires all x-values to be positive.")

        params, covariance = curve_fit(model_func, x_data, y_data, p0=p0, maxfev=5000)
        
        y_pred = model_func(x_data, *params)
        r2 = r2_score(y_data, y_pred)
        
        # --- Plotting ---
        plt.figure(figsize=(8, 6))
        plt.scatter(x_data, y_data, label='Data', alpha=0.6)
        
        x_fit = np.linspace(x_data.min(), x_data.max(), 200)
        y_fit = model_func(x_fit, *params)
        plt.plot(x_fit, y_fit, 'r-', label=f'Fit: {model_type}', linewidth=2)
        
        plt.title(f'Nonlinear Regression ({model_type.capitalize()})')
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        plot_image = base64.b64encode(buf.read()).decode('utf-8')

        param_names = {
            'exponential': ['a', 'b'],
            'logarithmic': ['a', 'b'],
            'power': ['a', 'b'],
            'sigmoid': ['L', 'k', 'x0']
        }[model_type]

        response = {
            "results": {
                "parameters": dict(zip(param_names, params)),
                "r_squared": r2
            },
            "plot": f"data:image/png;base64,{plot_image}"
        }

        print(json.dumps(response, default=_to_native_type))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
