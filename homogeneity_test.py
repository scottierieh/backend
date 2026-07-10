
import sys
import json
import pandas as pd
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

def _to_native_type(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj):
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
        value_var = payload.get('valueVar')
        group_var = payload.get('groupVar')
        alpha = payload.get('alpha', 0.05)

        if not all([data, value_var, group_var]):
            raise ValueError("Missing 'data', 'valueVar', or 'groupVar'")

        df = pd.DataFrame(data)
        
        # Prepare data
        clean_data = df[[value_var, group_var]].dropna()
        clean_data[value_var] = pd.to_numeric(clean_data[value_var], errors='coerce')
        clean_data.dropna(inplace=True)

        groups = clean_data[group_var].unique()
        if len(groups) < 2:
            raise ValueError("The grouping variable must have at least 2 groups.")

        samples = [clean_data[clean_data[group_var] == g][value_var] for g in groups]

        # --- Levene's Test ---
        levene_stat, levene_p = stats.levene(*samples)
        
        # --- Descriptive Statistics ---
        descriptives = {}
        for i, group in enumerate(groups):
            descriptives[str(group)] = {
                'n': len(samples[i]),
                'mean': np.mean(samples[i]),
                'variance': np.var(samples[i], ddof=1),
                'std_dev': np.std(samples[i], ddof=1)
            }

        # --- Interpretation ---
        assumption_met = levene_p > alpha
        interpretation_text = ""
        if assumption_met:
            interpretation_text = f"The test is not significant (p > {alpha}), so we assume the variances are equal across groups. The assumption of homogeneity of variances is met."
        else:
            interpretation_text = f"The test is significant (p <= {alpha}), indicating that the variances are not equal across groups. The assumption of homogeneity of variances is violated."

        # --- Plotting ---
        plt.figure(figsize=(8, 6))
        sns.boxplot(x=group_var, y=value_var, data=clean_data, palette='viridis')
        plt.title(f'Distribution of {value_var} by {group_var}')
        plt.xlabel(group_var)
        plt.ylabel(value_var)
        plt.tight_layout()
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        buf.seek(0)
        plot_image = base64.b64encode(buf.read()).decode('utf-8')
        
        results = {
            'levene_test': {
                'statistic': levene_stat,
                'p_value': levene_p
            },
            'descriptives': descriptives,
            'assumption_met': bool(assumption_met),
            'interpretation': interpretation_text,
            'plot': f"data:image/png;base64,{plot_image}"
        }
        
        print(json.dumps({'results': results}, default=_to_native_type))

    except Exception as e:
        error_response = {"error": str(e)}
        sys.stderr.write(json.dumps(error_response))
        sys.exit(1)

if __name__ == '__main__':
    main()
