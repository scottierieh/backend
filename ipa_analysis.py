

import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
import io
import base64

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

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

def create_main_plot(df_ipa, perf_mean, imp_mean, quadrant_colors):
    fig, ax1 = plt.subplots(figsize=(10, 8))
    for quadrant, color in quadrant_colors.items():
        data = df_ipa[df_ipa['quadrant'] == quadrant]
        if not data.empty:
            ax1.scatter(data['performance'], data['importance'], c=color, s=300, alpha=0.7, label=quadrant, edgecolors='black', linewidth=1.5)
    for _, row in df_ipa.iterrows():
        ax1.text(row['performance'], row['importance'], row['attribute'], fontsize=10, ha='center', va='center', fontweight='bold')
    
    ax1.axhline(y=imp_mean, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.axvline(x=perf_mean, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.set_xlabel('Performance (Mean Satisfaction)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Importance (Standardized Beta Coefficient)', fontsize=13, fontweight='bold')
    ax1.set_title('IPA Matrix - Regression-Based Importance', fontsize=15, fontweight='bold', pad=20)
    ax1.legend(loc='best', fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def create_dashboard_plot(df, df_ipa, perf_mean, quadrant_colors, attributes, dependent_var):
    fig = plt.figure(figsize=(12, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)

    # Importance Ranking
    ax2 = fig.add_subplot(gs[0, 0])
    df_imp_sorted = df_ipa.sort_values('relative_importance', ascending=True)
    colors_imp = [quadrant_colors.get(q, '#cccccc') for q in df_imp_sorted['quadrant']]
    ax2.barh(df_imp_sorted['attribute'], df_imp_sorted['relative_importance'], color=colors_imp, alpha=0.7, edgecolor='black')
    ax2.set_title('Attribute Importance Ranking (β-based)')
    ax2.set_xlabel('Relative Importance (%)')
    
    # Performance Ranking
    ax3 = fig.add_subplot(gs[1, 0])
    df_perf_sorted = df_ipa.sort_values('performance', ascending=True)
    colors_perf = [quadrant_colors.get(q, '#cccccc') for q in df_perf_sorted['quadrant']]
    ax3.barh(df_perf_sorted['attribute'], df_perf_sorted['performance'], color=colors_perf, alpha=0.7, edgecolor='black')
    ax3.axvline(perf_mean, color='r', ls='--');
    ax3.set_title('Attribute Performance Ranking')
    ax3.set_xlabel('Performance Score')

    # Bubble Chart
    ax4 = fig.add_subplot(gs[0, 1])
    scatter = sns.scatterplot(
        data=df_ipa,
        x='performance',
        y='importance',
        size='relative_importance',
        hue='quadrant',
        palette=quadrant_colors,
        sizes=(100, 2000),
        alpha=0.7,
        edgecolor='black',
        ax=ax4
    )
    for _, row in df_ipa.iterrows(): ax4.text(row['performance'], row['importance'], row['attribute'], ha='center', va='center', fontsize=8, weight='bold')
    ax4.axhline(df_ipa['importance'].mean(), ls='--', color='grey')
    ax4.axvline(df_ipa['performance'].mean(), ls='--', color='grey')
    ax4.set_title('Performance vs. Importance (Bubble size: Rel. Importance)')
    ax4.legend(title='Quadrant', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Gap Analysis
    ax5 = fig.add_subplot(gs[1, 1])
    df_gap_sorted = df_ipa.sort_values('gap')
    colors_gap = ['red' if g < 0 else 'green' for g in df_gap_sorted['gap']]
    ax5.barh(df_gap_sorted['attribute'], df_gap_sorted['gap'], color=colors_gap, alpha=0.7)
    ax5.axvline(0, color='k', lw=1); ax5.set_title('Performance-Importance Gap')

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get('data')
        dependent_var = payload.get('dependentVar', 'Overall_Satisfaction')
        independent_vars = payload.get('independentVars')

        if not data:
            raise ValueError("Data not provided.")

        df = pd.DataFrame(data)

        all_cols_for_analysis = [dependent_var] + (independent_vars or [])
        if not independent_vars:
            independent_vars = [col for col in df.columns if col != dependent_var]
            all_cols_for_analysis = df.columns.tolist()

        if dependent_var not in df.columns:
            raise ValueError(f"Dependent variable '{dependent_var}' not found in data.")
        missing_iv = [iv for iv in independent_vars if iv not in df.columns]
        if missing_iv:
             raise ValueError(f"Independent variables not found: {', '.join(missing_iv)}")

        df_analysis = df[all_cols_for_analysis].copy()
        for col in df_analysis.columns:
            df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')
        df_analysis.dropna(inplace=True)
        
        if df_analysis.shape[0] < len(independent_vars) + 2:
            raise ValueError(f"Not enough valid data points. Need at least {len(independent_vars) + 2} complete rows for regression.")

        # --- 1. Performance Calculation (Mean) ---
        performance = df_analysis[independent_vars].mean()

        # --- 2. Importance Calculation (Regression-based) ---
        X = df_analysis[independent_vars]
        y = df_analysis[dependent_var]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LinearRegression().fit(X_scaled, y)
        
        beta_coefficients = pd.DataFrame({'attribute': independent_vars, 'beta': model.coef_})
        
        total_beta_abs = beta_coefficients['beta'].abs().sum()
        beta_coefficients['relative_importance'] = (beta_coefficients['beta'].abs() / total_beta_abs) * 100 if total_beta_abs > 0 else 0
        
        # --- 3. IPA Matrix Data Preparation ---
        ipa_data = []
        for attr in independent_vars:
            perf = performance.get(attr, 0)
            beta_row = beta_coefficients[beta_coefficients['attribute'] == attr].iloc[0]
            ipa_data.append({
                'attribute': attr, 
                'performance': perf, 
                'importance': beta_row['beta'], 
                'relative_importance': beta_row['relative_importance']
            })
        
        df_ipa = pd.DataFrame(ipa_data)
        
        # --- 4. Quadrant Classification ---
        perf_mean = df_ipa['performance'].mean()
        imp_mean = 0 # With standardized Beta, 0 is the natural midpoint
        
        def classify_quadrant(row):
            if row['importance'] >= imp_mean and row['performance'] >= perf_mean: return 'Q1: Keep Up Good Work'
            elif row['importance'] >= imp_mean and row['performance'] < perf_mean: return 'Q2: Concentrate Here'
            elif row['importance'] < imp_mean and row['performance'] < perf_mean: return 'Q3: Low Priority'
            else: return 'Q4: Possible Overkill'
        
        df_ipa['quadrant'] = df_ipa.apply(classify_quadrant, axis=1)
        
        # --- 5. Advanced Metrics ---
        max_scale_value = df_analysis[independent_vars].max().max() if not df_analysis[independent_vars].empty else 7
        df_ipa['importance_scaled'] = (df_ipa['relative_importance'] / df_ipa['relative_importance'].max() * max_scale_value) if df_ipa['relative_importance'].max() > 0 else 0
        df_ipa['gap'] = df_ipa['performance'] - df_ipa['importance_scaled']
        df_ipa['priority_score'] = df_ipa['relative_importance'] * (max_scale_value - df_ipa['performance'])
        
        # --- 6. Statistical Validation ---
        r2 = model.score(X_scaled, y)
        adj_r2 = 1 - (1 - r2) * (len(y) - 1) / (len(y) - X.shape[1] - 1) if (len(y) - X.shape[1] - 1) > 0 else r2
        validation_results = {'r2': r2, 'adj_r2': adj_r2, 'beta_coefficients': beta_coefficients.to_dict('records')}

        quadrant_colors = {'Q1: Keep Up Good Work': '#4CAF50', 'Q2: Concentrate Here': '#F44336', 'Q3: Low Priority': '#9E9E9E', 'Q4: Possible Overkill': '#FF9800'}
        
        main_plot_img = create_main_plot(df_ipa, perf_mean, imp_mean, quadrant_colors)
        dashboard_plot_img = create_dashboard_plot(df_analysis, df_ipa, perf_mean, quadrant_colors, independent_vars, dependent_var)

        response = {
            'results': {
                'ipa_matrix': df_ipa.to_dict('records'),
                'regression_summary': validation_results,
            },
            'main_plot': main_plot_img,
            'dashboard_plot': dashboard_plot_img,
        }

        print(json.dumps(response, default=_to_native_type))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()
