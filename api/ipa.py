"""
IPA (Importance-Performance Analysis) Router for FastAPI
Key Driver Analysis using regression-based importance
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import io
import base64
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class IpaRequest(BaseModel):
    data: List[Dict[str, Any]]
    dependentVar: str
    independentVars: List[str]


class BetaCoefficient(BaseModel):
    attribute: str
    beta: float


class RegressionSummary(BaseModel):
    r2: float
    adj_r2: float
    beta_coefficients: List[Dict[str, Any]]


class IpaMatrixItem(BaseModel):
    attribute: str
    performance: float
    importance: float
    relative_importance: float
    quadrant: str
    gap: float
    priority_score: float


class IpaResults(BaseModel):
    ipa_matrix: List[Dict[str, Any]]
    regression_summary: Dict[str, Any]


class IpaResponse(BaseModel):
    results: IpaResults
    main_plot: str
    dashboard_plot: str


def _to_native_type(obj):
    """Convert numpy types to native Python types"""
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


def _convert_dict(d: dict) -> dict:
    """Recursively convert numpy types in dictionary"""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _convert_dict(v)
        elif isinstance(v, list):
            result[k] = [_convert_dict(i) if isinstance(i, dict) else _to_native_type(i) for i in v]
        else:
            result[k] = _to_native_type(v)
    return result


def create_main_plot(df_ipa: pd.DataFrame, perf_mean: float, imp_mean: float, quadrant_colors: dict) -> str:
    """Create main IPA matrix scatter plot"""
    fig, ax1 = plt.subplots(figsize=(10, 8))
    
    for quadrant, color in quadrant_colors.items():
        data = df_ipa[df_ipa['quadrant'] == quadrant]
        if not data.empty:
            ax1.scatter(
                data['performance'], 
                data['importance'], 
                c=color, 
                s=300, 
                alpha=0.7, 
                label=quadrant, 
                edgecolors='black', 
                linewidth=1.5
            )
    
    for _, row in df_ipa.iterrows():
        ax1.text(
            row['performance'], 
            row['importance'], 
            row['attribute'], 
            fontsize=10, 
            ha='center', 
            va='center', 
            fontweight='bold'
        )
    
    ax1.axhline(y=imp_mean, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.axvline(x=perf_mean, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax1.set_xlabel('Performance (Mean Satisfaction)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Importance (Standardized Beta Coefficient)', fontsize=13, fontweight='bold')
    ax1.set_title('IPA Matrix - Regression-Based Importance', fontsize=15, fontweight='bold', pad=20)
    ax1.legend(loc='best', fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=1)
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def create_dashboard_plot(
    df: pd.DataFrame, 
    df_ipa: pd.DataFrame, 
    perf_mean: float, 
    quadrant_colors: dict, 
    attributes: List[str], 
    dependent_var: str
) -> str:
    """Create comprehensive dashboard with multiple visualizations"""
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)

    # Importance Ranking
    ax2 = fig.add_subplot(gs[0, 0])
    df_imp_sorted = df_ipa.sort_values('relative_importance', ascending=True)
    colors_imp = [quadrant_colors.get(q, '#cccccc') for q in df_imp_sorted['quadrant']]
    ax2.barh(df_imp_sorted['attribute'], df_imp_sorted['relative_importance'], color=colors_imp, alpha=0.7, edgecolor='black')
    ax2.set_title('Attribute Importance Ranking (β-based)', fontweight='bold')
    ax2.set_xlabel('Relative Importance (%)')
    
    # Performance Ranking
    ax3 = fig.add_subplot(gs[1, 0])
    df_perf_sorted = df_ipa.sort_values('performance', ascending=True)
    colors_perf = [quadrant_colors.get(q, '#cccccc') for q in df_perf_sorted['quadrant']]
    ax3.barh(df_perf_sorted['attribute'], df_perf_sorted['performance'], color=colors_perf, alpha=0.7, edgecolor='black')
    ax3.axvline(perf_mean, color='r', ls='--', label=f'Mean: {perf_mean:.2f}')
    ax3.set_title('Attribute Performance Ranking', fontweight='bold')
    ax3.set_xlabel('Performance Score')
    ax3.legend()

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
    for _, row in df_ipa.iterrows():
        ax4.text(row['performance'], row['importance'], row['attribute'], ha='center', va='center', fontsize=8, weight='bold')
    ax4.axhline(df_ipa['importance'].mean(), ls='--', color='grey', alpha=0.7)
    ax4.axvline(df_ipa['performance'].mean(), ls='--', color='grey', alpha=0.7)
    ax4.set_title('Performance vs. Importance (Bubble size: Rel. Importance)', fontweight='bold')
    ax4.legend(title='Quadrant', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)

    # Gap Analysis
    ax5 = fig.add_subplot(gs[1, 1])
    df_gap_sorted = df_ipa.sort_values('gap')
    colors_gap = ['#F44336' if g < 0 else '#4CAF50' for g in df_gap_sorted['gap']]
    ax5.barh(df_gap_sorted['attribute'], df_gap_sorted['gap'], color=colors_gap, alpha=0.7, edgecolor='black')
    ax5.axvline(0, color='k', lw=1)
    ax5.set_title('Performance-Importance Gap', fontweight='bold')
    ax5.set_xlabel('Gap (Performance - Scaled Importance)')

    # Correlation Heatmap
    ax6 = fig.add_subplot(gs[:, 2])
    try:
        corr_matrix = df[[dependent_var] + attributes].corr(numeric_only=True)[[dependent_var]].sort_values(dependent_var, ascending=False)
        corr_matrix_display = corr_matrix.drop(dependent_var, errors='ignore')
        if not corr_matrix_display.empty:
            sns.heatmap(corr_matrix_display, annot=True, fmt='.3f', cmap='RdYlGn', center=0, ax=ax6, cbar_kws={'shrink': 0.8})
            ax6.set_title(f'Pearson Correlation with {dependent_var}', fontweight='bold')
        else:
            ax6.text(0.5, 0.5, 'Correlation data unavailable', ha='center', va='center')
            ax6.set_title('Correlation Analysis')
    except Exception:
        ax6.text(0.5, 0.5, 'Correlation calculation failed', ha='center', va='center')
        ax6.set_title('Correlation Analysis')
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


@router.post("/ipa")
async def run_ipa_analysis(request: IpaRequest) -> Dict[str, Any]:
    """
    Perform Importance-Performance Analysis (IPA) using regression-based importance.
    
    This endpoint:
    1. Calculates performance as mean ratings
    2. Derives importance from standardized regression beta coefficients
    3. Classifies attributes into four strategic quadrants
    4. Generates visualizations for strategic decision-making
    """
    try:
        data = request.data
        dependent_var = request.dependentVar
        independent_vars = request.independentVars

        if not data:
            raise HTTPException(status_code=400, detail="Data not provided.")
        
        if not dependent_var:
            raise HTTPException(status_code=400, detail="Dependent variable not specified.")
        
        if not independent_vars or len(independent_vars) < 1:
            raise HTTPException(status_code=400, detail="At least one independent variable required.")

        df = pd.DataFrame(data)

        # Validate columns exist
        if dependent_var not in df.columns:
            raise HTTPException(status_code=400, detail=f"Dependent variable '{dependent_var}' not found in data.")
        
        missing_iv = [iv for iv in independent_vars if iv not in df.columns]
        if missing_iv:
            raise HTTPException(status_code=400, detail=f"Independent variables not found: {', '.join(missing_iv)}")

        # Prepare analysis dataframe
        all_cols = [dependent_var] + independent_vars
        df_analysis = df[all_cols].copy()
        
        for col in df_analysis.columns:
            df_analysis[col] = pd.to_numeric(df_analysis[col], errors='coerce')
        
        df_analysis.dropna(inplace=True)
        
        min_rows = len(independent_vars) + 2
        if df_analysis.shape[0] < min_rows:
            raise HTTPException(
                status_code=400, 
                detail=f"Not enough valid data points. Need at least {min_rows} complete rows for regression, got {df_analysis.shape[0]}."
            )

        # --- 1. Performance Calculation (Mean) ---
        performance = df_analysis[independent_vars].mean()

        # --- 2. Importance Calculation (Regression-based) ---
        X = df_analysis[independent_vars]
        y = df_analysis[dependent_var]
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        model = LinearRegression().fit(X_scaled, y)
        
        beta_coefficients = pd.DataFrame({
            'attribute': independent_vars, 
            'beta': model.coef_
        })
        
        total_beta_abs = beta_coefficients['beta'].abs().sum()
        if total_beta_abs > 0:
            beta_coefficients['relative_importance'] = (beta_coefficients['beta'].abs() / total_beta_abs) * 100
        else:
            beta_coefficients['relative_importance'] = 0
        
        # --- 3. IPA Matrix Data Preparation ---
        ipa_data = []
        for attr in independent_vars:
            perf = performance.get(attr, 0)
            beta_row = beta_coefficients[beta_coefficients['attribute'] == attr].iloc[0]
            ipa_data.append({
                'attribute': attr, 
                'performance': float(perf), 
                'importance': float(beta_row['beta']), 
                'relative_importance': float(beta_row['relative_importance'])
            })
        
        df_ipa = pd.DataFrame(ipa_data)
        
        # --- 4. Quadrant Classification ---
        perf_mean = float(df_ipa['performance'].mean())
        imp_mean = 0  # With standardized Beta, 0 is the natural midpoint
        
        def classify_quadrant(row):
            if row['importance'] >= imp_mean and row['performance'] >= perf_mean:
                return 'Q1: Keep Up Good Work'
            elif row['importance'] >= imp_mean and row['performance'] < perf_mean:
                return 'Q2: Concentrate Here'
            elif row['importance'] < imp_mean and row['performance'] < perf_mean:
                return 'Q3: Low Priority'
            else:
                return 'Q4: Possible Overkill'
        
        df_ipa['quadrant'] = df_ipa.apply(classify_quadrant, axis=1)
        
        # --- 5. Advanced Metrics ---
        max_scale_value = float(df_analysis[independent_vars].max().max()) if not df_analysis[independent_vars].empty else 7.0
        
        if df_ipa['relative_importance'].max() > 0:
            df_ipa['importance_scaled'] = (df_ipa['relative_importance'] / df_ipa['relative_importance'].max() * max_scale_value)
        else:
            df_ipa['importance_scaled'] = 0
        
        df_ipa['gap'] = df_ipa['performance'] - df_ipa['importance_scaled']
        df_ipa['priority_score'] = df_ipa['relative_importance'] * (max_scale_value - df_ipa['performance'])
        
        # --- 6. Statistical Validation ---
        r2 = float(model.score(X_scaled, y))
        n = len(y)
        p = X.shape[1]
        if (n - p - 1) > 0:
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        else:
            adj_r2 = r2
        adj_r2 = float(adj_r2)
        
        beta_records = beta_coefficients.to_dict('records')
        beta_records = [_convert_dict(rec) for rec in beta_records]
        
        validation_results = {
            'r2': r2, 
            'adj_r2': adj_r2, 
            'beta_coefficients': beta_records
        }

        # --- 7. Generate Plots ---
        quadrant_colors = {
            'Q1: Keep Up Good Work': '#4CAF50', 
            'Q2: Concentrate Here': '#F44336', 
            'Q3: Low Priority': '#9E9E9E', 
            'Q4: Possible Overkill': '#FF9800'
        }
        
        main_plot_img = create_main_plot(df_ipa, perf_mean, imp_mean, quadrant_colors)
        dashboard_plot_img = create_dashboard_plot(
            df_analysis, df_ipa, perf_mean, quadrant_colors, independent_vars, dependent_var
        )

        # --- 8. Prepare Response ---
        ipa_matrix_records = df_ipa.to_dict('records')
        ipa_matrix_records = [_convert_dict(rec) for rec in ipa_matrix_records]

        response = {
            'results': {
                'ipa_matrix': ipa_matrix_records,
                'regression_summary': validation_results,
            },
            'main_plot': main_plot_img,
            'dashboard_plot': dashboard_plot_img,
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"IPA analysis failed: {str(e)}")
