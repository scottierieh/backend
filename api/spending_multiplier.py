"""
Public Spending Multiplier Analysis Router for FastAPI
Estimate fiscal multipliers via OLS regression: ΔY = α + β·ΔG + controls
β represents the spending multiplier (GDP change per unit of government spending)
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class SpendingMultiplierRequest(BaseModel):
    data: List[Dict[str, Any]]
    spending_col: str           # Government spending column
    outcome_col: str            # GDP / economic outcome column
    group_col: Optional[str] = None    # Region/state grouping (for panel)
    time_col: Optional[str] = None     # Time period column
    covariates: Optional[List[str]] = None  # Control variables
    use_differences: bool = True        # Use first-differences (ΔY on ΔG)
    lag_spending: bool = False          # Use lagged spending (t-1)
    robust_se: bool = True              # Use HC3 robust standard errors


def _to_native_type(obj):
    """Convert numpy/pandas types to JSON-serializable Python types"""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string"""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return image_base64


def compute_multiplier_ols(df: pd.DataFrame, spending_col: str, outcome_col: str,
                           covariates: List[str] = None, robust_se: bool = True) -> Dict[str, Any]:
    """Run OLS regression: Y = α + β·G + controls + ε"""
    try:
        # Build formula
        formula = f"{outcome_col} ~ {spending_col}"
        if covariates:
            valid_covs = [c for c in covariates if c in df.columns 
                         and c not in [outcome_col, spending_col]]
            if valid_covs:
                formula += " + " + " + ".join(valid_covs)

        # Fit model
        if robust_se:
            model = smf.ols(formula, data=df).fit(cov_type='HC3')
        else:
            model = smf.ols(formula, data=df).fit()

        # Extract coefficients
        coefficients = []
        param_names = {
            'Intercept': 'Intercept (α)',
            spending_col: f'Spending Multiplier (β)',
        }
        for param in model.params.index:
            display_name = param_names.get(param, param)
            coefficients.append({
                'term': display_name,
                'variable': param,
                'estimate': _to_native_type(model.params[param]),
                'std_error': _to_native_type(model.bse[param]),
                't_value': _to_native_type(model.tvalues[param]),
                'p_value': _to_native_type(model.pvalues[param]),
                'ci_lower': _to_native_type(model.conf_int().loc[param, 0]),
                'ci_upper': _to_native_type(model.conf_int().loc[param, 1]),
                'significant': bool(model.pvalues[param] < 0.05),
            })

        # The multiplier is the coefficient on spending
        multiplier = _to_native_type(model.params[spending_col])
        multiplier_se = _to_native_type(model.bse[spending_col])
        multiplier_p = _to_native_type(model.pvalues[spending_col])
        multiplier_ci = [
            _to_native_type(model.conf_int().loc[spending_col, 0]),
            _to_native_type(model.conf_int().loc[spending_col, 1]),
        ]

        # Residuals for diagnostics
        residuals = model.resid.tolist()
        fitted = model.fittedvalues.tolist()

        # Durbin-Watson test for autocorrelation
        from statsmodels.stats.stattools import durbin_watson
        dw_stat = _to_native_type(durbin_watson(model.resid))

        # Breusch-Pagan test for heteroscedasticity
        try:
            from statsmodels.stats.diagnostic import het_breuschpagan
            bp_stat, bp_p, _, _ = het_breuschpagan(model.resid, model.model.exog)
            bp_result = {'statistic': _to_native_type(bp_stat), 'p_value': _to_native_type(bp_p)}
        except:
            bp_result = None

        return {
            'multiplier': multiplier,
            'multiplier_se': multiplier_se,
            'multiplier_p': multiplier_p,
            'multiplier_ci': multiplier_ci,
            'significant': bool(multiplier_p < 0.05) if multiplier_p else False,
            'coefficients': coefficients,
            'r_squared': _to_native_type(model.rsquared),
            'adj_r_squared': _to_native_type(model.rsquared_adj),
            'f_statistic': _to_native_type(model.fvalue),
            'f_pvalue': _to_native_type(model.f_pvalue),
            'n_obs': _to_native_type(model.nobs),
            'aic': _to_native_type(model.aic),
            'bic': _to_native_type(model.bic),
            'durbin_watson': dw_stat,
            'breusch_pagan': bp_result,
            'residuals': residuals[:500],  # cap for JSON
            'fitted_values': fitted[:500],
            'formula': formula,
            'robust_se': robust_se,
        }
    except Exception as e:
        return {'error': str(e)}


def compute_by_group(df: pd.DataFrame, spending_col: str, outcome_col: str,
                     group_col: str, covariates: List[str] = None) -> List[Dict[str, Any]]:
    """Compute multiplier for each group (region/sector)"""
    results = []
    for group_name, gdf in df.groupby(group_col):
        if len(gdf) < 5:
            continue
        try:
            formula = f"{outcome_col} ~ {spending_col}"
            model = smf.ols(formula, data=gdf).fit(cov_type='HC3')
            mult = model.params.get(spending_col, None)
            p = model.pvalues.get(spending_col, None)
            ci = model.conf_int().loc[spending_col] if spending_col in model.conf_int().index else [None, None]
            results.append({
                'group': str(group_name),
                'multiplier': _to_native_type(mult),
                'std_error': _to_native_type(model.bse.get(spending_col, None)),
                'p_value': _to_native_type(p),
                'ci_lower': _to_native_type(ci[0]),
                'ci_upper': _to_native_type(ci[1]),
                'significant': bool(p < 0.05) if p else False,
                'n_obs': len(gdf),
                'r_squared': _to_native_type(model.rsquared),
            })
        except:
            continue

    # Sort by multiplier descending
    results.sort(key=lambda x: x['multiplier'] if x['multiplier'] is not None else 0, reverse=True)
    return results


def generate_scatter_plot(df: pd.DataFrame, spending_col: str, outcome_col: str,
                          model_result: Dict) -> str:
    """Generate spending vs outcome scatter with regression line"""
    fig, ax = plt.subplots(figsize=(12, 8))

    ax.scatter(df[spending_col], df[outcome_col], alpha=0.5, s=40, 
               color='#3b82f6', edgecolors='white', linewidths=0.5)

    # Regression line
    x_range = np.linspace(df[spending_col].min(), df[spending_col].max(), 100)
    intercept = model_result['coefficients'][0]['estimate'] if model_result.get('coefficients') else 0
    slope = model_result.get('multiplier', 0)
    y_pred = intercept + slope * x_range
    ax.plot(x_range, y_pred, color='#ef4444', linewidth=2.5, label=f'β = {slope:.4f}')

    # CI band
    if model_result.get('multiplier_ci'):
        ci_lo, ci_hi = model_result['multiplier_ci']
        y_lo = intercept + ci_lo * x_range
        y_hi = intercept + ci_hi * x_range
        ax.fill_between(x_range, y_lo, y_hi, alpha=0.1, color='#ef4444')

    ax.set_xlabel(spending_col, fontsize=12)
    ax.set_ylabel(outcome_col, fontsize=12)
    ax.set_title('Government Spending vs Economic Outcome', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_residual_plot(model_result: Dict) -> str:
    """Generate residual diagnostic plots"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    fitted = model_result.get('fitted_values', [])
    residuals = model_result.get('residuals', [])

    if fitted and residuals:
        # Residuals vs Fitted
        axes[0].scatter(fitted, residuals, alpha=0.5, s=30, color='#6366f1')
        axes[0].axhline(y=0, color='#ef4444', linestyle='--', linewidth=1.5)
        axes[0].set_xlabel('Fitted Values', fontsize=11)
        axes[0].set_ylabel('Residuals', fontsize=11)
        axes[0].set_title('Residuals vs Fitted', fontsize=13, fontweight='bold')
        axes[0].grid(True, linestyle='--', alpha=0.3)

        # QQ plot
        from scipy.stats import probplot
        probplot(residuals, plot=axes[1])
        axes[1].set_title('Q-Q Plot (Normality)', fontsize=13, fontweight='bold')
        axes[1].grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_group_multiplier_plot(group_results: List[Dict]) -> str:
    """Generate horizontal bar chart of multipliers by group"""
    if not group_results:
        return ""

    fig, ax = plt.subplots(figsize=(12, max(6, len(group_results) * 0.5)))

    groups = [r['group'] for r in group_results]
    multipliers = [r['multiplier'] for r in group_results]
    colors = ['#22c55e' if m and m > 1 else '#f59e0b' if m and m > 0 else '#ef4444' for m in multipliers]
    errors = [r.get('std_error', 0) or 0 for r in group_results]

    y_pos = range(len(groups))
    ax.barh(y_pos, multipliers, xerr=errors, capsize=3, color=colors, alpha=0.8, edgecolor='white')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(groups, fontsize=10)
    ax.axvline(x=1.0, color='#6366f1', linestyle='--', linewidth=2, alpha=0.7, label='Multiplier = 1.0')
    ax.axvline(x=0, color='#94a3b8', linestyle='-', linewidth=1)
    ax.set_xlabel('Spending Multiplier (β)', fontsize=12)
    ax.set_title('Spending Multiplier by Region/Group', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, axis='x', linestyle='--', alpha=0.3)
    ax.invert_yaxis()

    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_time_series_plot(df: pd.DataFrame, spending_col: str, outcome_col: str,
                               time_col: str) -> str:
    """Generate dual-axis time series of spending and outcome"""
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # Aggregate by time if needed
    ts = df.groupby(time_col)[[spending_col, outcome_col]].mean().reset_index()
    ts = ts.sort_values(time_col)

    color1 = '#3b82f6'
    color2 = '#ef4444'

    ax1.plot(ts[time_col], ts[spending_col], color=color1, linewidth=2.5, marker='o', 
             markersize=6, label=spending_col)
    ax1.set_xlabel(time_col, fontsize=12)
    ax1.set_ylabel(spending_col, fontsize=12, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    ax2.plot(ts[time_col], ts[outcome_col], color=color2, linewidth=2.5, marker='s', 
             markersize=6, label=outcome_col)
    ax2.set_ylabel(outcome_col, fontsize=12, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)

    ax1.set_title('Government Spending vs Economic Outcome Over Time', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.3)
    plt.xticks(rotation=45)

    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_interpretation(model_result: Dict, group_results: List[Dict],
                            use_differences: bool) -> Dict[str, Any]:
    """Generate comprehensive interpretation"""
    key_insights = []

    mult = model_result.get('multiplier')
    p = model_result.get('multiplier_p')
    sig = model_result.get('significant', False)
    ci = model_result.get('multiplier_ci', [None, None])

    if mult is None:
        return {'key_insights': [{'title': 'Error', 'description': model_result.get('error', 'Unknown'), 'status': 'negative'}]}

    # 1. Main multiplier result
    spec = "first-differenced" if use_differences else "levels"
    if sig:
        if mult > 1:
            key_insights.append({
                'title': 'Expansionary Multiplier',
                'description': f'The spending multiplier is {mult:.3f} (p = {p:.4f}), indicating that each unit of government spending generates {mult:.2f} units of economic output. Since β > 1, public spending has a net expansionary effect ({spec} specification).',
                'status': 'positive'
            })
        elif mult > 0:
            key_insights.append({
                'title': 'Positive but Sub-unitary Multiplier',
                'description': f'The multiplier is {mult:.3f} (p = {p:.4f}). While spending increases output, the effect is less than 1:1 (β < 1), suggesting partial crowding-out of private activity ({spec} specification).',
                'status': 'neutral'
            })
        else:
            key_insights.append({
                'title': 'Negative Multiplier',
                'description': f'The multiplier is {mult:.3f} (p = {p:.4f}). Government spending appears to reduce economic output, possibly through strong crowding-out effects ({spec} specification).',
                'status': 'negative'
            })
    else:
        key_insights.append({
            'title': 'Multiplier Not Statistically Significant',
            'description': f'β = {mult:.3f} (p = {p:.4f}). The spending-output relationship is not statistically significant at the 5% level. The 95% CI [{ci[0]:.3f}, {ci[1]:.3f}] includes both positive and negative values.',
            'status': 'warning'
        })

    # 2. Model fit
    r2 = model_result.get('r_squared')
    if r2 is not None:
        key_insights.append({
            'title': 'Model Fit',
            'description': f'R² = {r2*100:.1f}%, explaining {r2*100:.1f}% of outcome variation. F = {model_result.get("f_statistic", 0):.2f} (p = {model_result.get("f_pvalue", 1):.4f}). N = {model_result.get("n_obs", 0):.0f} observations.',
            'status': 'neutral'
        })

    # 3. Diagnostics
    dw = model_result.get('durbin_watson')
    if dw is not None:
        if dw < 1.5 or dw > 2.5:
            key_insights.append({
                'title': 'Autocorrelation Warning',
                'description': f'Durbin-Watson = {dw:.3f} (ideal ≈ 2.0). Values far from 2 suggest serial correlation in residuals, which can bias standard errors. Consider panel-corrected SE or Newey-West adjustment.',
                'status': 'warning'
            })

    bp = model_result.get('breusch_pagan')
    if bp and bp.get('p_value') is not None and bp['p_value'] < 0.05:
        key_insights.append({
            'title': 'Heteroscedasticity Detected',
            'description': f'Breusch-Pagan test: χ² = {bp["statistic"]:.2f}, p = {bp["p_value"]:.4f}. Heteroscedastic errors detected. HC3 robust standard errors are recommended (and used if enabled).',
            'status': 'warning'
        })

    # 4. Group heterogeneity
    if group_results and len(group_results) > 1:
        mults = [r['multiplier'] for r in group_results if r['multiplier'] is not None]
        if mults:
            max_g = max(group_results, key=lambda x: x['multiplier'] if x['multiplier'] else -999)
            min_g = min(group_results, key=lambda x: x['multiplier'] if x['multiplier'] else 999)
            key_insights.append({
                'title': 'Regional Heterogeneity',
                'description': f'Multipliers range from {min_g["multiplier"]:.3f} ({min_g["group"]}) to {max_g["multiplier"]:.3f} ({max_g["group"]}). The spread of {max_g["multiplier"] - min_g["multiplier"]:.3f} suggests significant regional variation in fiscal effectiveness.',
                'status': 'neutral'
            })

    # 5. Methodology caveat
    key_insights.append({
        'title': 'Endogeneity Caveat',
        'description': 'OLS multiplier estimates may suffer from endogeneity bias — government spending often responds to economic conditions (reverse causality). More rigorous approaches include IV/2SLS (using military spending or political shocks as instruments), structural VAR, or natural experiments.',
        'status': 'warning'
    })

    return {
        'key_insights': key_insights,
        'multiplier': _to_native_type(mult),
        'is_significant': sig,
        'recommendation': f'Estimated multiplier: {mult:.3f}' + (' (significant)' if sig else ' (not significant)'),
    }


@router.post("/spending-multiplier")
async def run_spending_multiplier(request: SpendingMultiplierRequest) -> Dict[str, Any]:
    """
    Estimate Public Spending Multiplier via OLS Regression.
    
    Computes the fiscal multiplier β from: Y = α + β·G + controls + ε
    where β measures the GDP response per unit of government spending.
    Supports first-differences, lagged spending, panel groups, and robust SE.
    """
    try:
        data = request.data
        spending_col = request.spending_col
        outcome_col = request.outcome_col
        group_col = request.group_col
        time_col = request.time_col
        covariates = request.covariates or []
        use_differences = request.use_differences
        lag_spending = request.lag_spending
        robust_se = request.robust_se

        if not data:
            raise HTTPException(status_code=400, detail="Data not provided.")

        df = pd.DataFrame(data)

        # Validate columns
        for col in [spending_col, outcome_col]:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{col}' not found.")

        # Convert to numeric
        df[spending_col] = pd.to_numeric(df[spending_col], errors='coerce')
        df[outcome_col] = pd.to_numeric(df[outcome_col], errors='coerce')
        df = df.dropna(subset=[spending_col, outcome_col])

        if len(df) < 10:
            raise HTTPException(status_code=400, detail="At least 10 observations required.")

        # Sort by time if available
        if time_col and time_col in df.columns:
            try:
                df[time_col] = pd.to_numeric(df[time_col], errors='coerce')
            except:
                pass
            df = df.sort_values(time_col)

        # Convert covariates to numeric
        for cov in covariates:
            if cov in df.columns:
                df[cov] = pd.to_numeric(df[cov], errors='coerce')

        # ── First-differences transformation ──
        original_df = df.copy()  # keep for plots
        diff_cols_used = []

        if use_differences:
            if group_col and group_col in df.columns:
                # Panel first-differences: within each group
                df = df.sort_values([group_col, time_col] if time_col and time_col in df.columns else [group_col])
                diff_df_parts = []
                for _, gdf in df.groupby(group_col):
                    gdf = gdf.copy()
                    gdf[f'd_{outcome_col}'] = gdf[outcome_col].diff()
                    gdf[f'd_{spending_col}'] = gdf[spending_col].diff()
                    for cov in covariates:
                        if cov in gdf.columns:
                            gdf[f'd_{cov}'] = gdf[cov].diff()
                    diff_df_parts.append(gdf)
                df = pd.concat(diff_df_parts, ignore_index=True)
            else:
                # Simple first-differences
                df[f'd_{outcome_col}'] = df[outcome_col].diff()
                df[f'd_{spending_col}'] = df[spending_col].diff()
                for cov in covariates:
                    if cov in df.columns:
                        df[f'd_{cov}'] = df[cov].diff()

            df = df.dropna(subset=[f'd_{outcome_col}', f'd_{spending_col}'])
            reg_outcome = f'd_{outcome_col}'
            reg_spending = f'd_{spending_col}'
            reg_covariates = [f'd_{cov}' for cov in covariates if f'd_{cov}' in df.columns]
            diff_cols_used = [reg_outcome, reg_spending] + reg_covariates
        else:
            reg_outcome = outcome_col
            reg_spending = spending_col
            reg_covariates = [c for c in covariates if c in df.columns]

        # ── Lag spending if requested ──
        if lag_spending:
            if group_col and group_col in df.columns:
                df[f'lag_{reg_spending}'] = df.groupby(group_col)[reg_spending].shift(1)
            else:
                df[f'lag_{reg_spending}'] = df[reg_spending].shift(1)
            df = df.dropna(subset=[f'lag_{reg_spending}'])
            reg_spending = f'lag_{reg_spending}'

        if len(df) < 5:
            raise HTTPException(status_code=400, detail="Not enough observations after transformations.")

        # ── Main OLS regression ──
        model_result = compute_multiplier_ols(df, reg_spending, reg_outcome, reg_covariates, robust_se)

        # ── Group-level multipliers ──
        group_results = []
        if group_col and group_col in df.columns:
            group_results = compute_by_group(df, reg_spending, reg_outcome, group_col, reg_covariates)

        # ── Visualizations ──
        scatter_plot = generate_scatter_plot(df, reg_spending, reg_outcome, model_result)
        residual_plot = generate_residual_plot(model_result) if 'error' not in model_result else ""
        group_plot = generate_group_multiplier_plot(group_results) if group_results else ""
        time_plot = generate_time_series_plot(original_df, spending_col, outcome_col, time_col) if time_col and time_col in df.columns else ""

        # ── Interpretation ──
        interpretation = generate_interpretation(model_result, group_results, use_differences)

        # ── Descriptive statistics ──
        desc_stats = {
            'spending_mean': _to_native_type(df[reg_spending].mean()),
            'spending_std': _to_native_type(df[reg_spending].std()),
            'spending_min': _to_native_type(df[reg_spending].min()),
            'spending_max': _to_native_type(df[reg_spending].max()),
            'outcome_mean': _to_native_type(df[reg_outcome].mean()),
            'outcome_std': _to_native_type(df[reg_outcome].std()),
            'outcome_min': _to_native_type(df[reg_outcome].min()),
            'outcome_max': _to_native_type(df[reg_outcome].max()),
            'correlation': _to_native_type(df[[reg_spending, reg_outcome]].corr().iloc[0, 1]),
            'n_obs': len(df),
            'n_groups': df[group_col].nunique() if group_col and group_col in df.columns else 1,
            'n_periods': df[time_col].nunique() if time_col and time_col in df.columns else None,
        }

        # ── Scatter data for frontend (Recharts) ──
        scatter_data = []
        sample_df = df.sample(min(500, len(df)), random_state=42) if len(df) > 500 else df
        for _, row in sample_df.iterrows():
            point = {
                'spending': _to_native_type(row[reg_spending]),
                'outcome': _to_native_type(row[reg_outcome]),
            }
            if group_col and group_col in df.columns:
                point['group'] = str(row[group_col])
            scatter_data.append(point)

        return {
            'model': model_result,
            'group_results': group_results,
            'descriptive_stats': desc_stats,
            'scatter_data': scatter_data,
            'scatter_plot': scatter_plot,
            'residual_plot': residual_plot,
            'group_plot': group_plot,
            'time_series_plot': time_plot,
            'interpretation': interpretation,
            'config': {
                'spending_col': spending_col,
                'outcome_col': outcome_col,
                'reg_spending': reg_spending,
                'reg_outcome': reg_outcome,
                'use_differences': use_differences,
                'lag_spending': lag_spending,
                'robust_se': robust_se,
                'group_col': group_col,
                'time_col': time_col,
                'covariates': covariates,
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Spending multiplier analysis failed: {str(e)}")
