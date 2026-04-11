"""
Regression Discontinuity Design (RDD) Router for FastAPI
Estimate causal effects at a threshold/cutoff point
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
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class RDDRequest(BaseModel):
    data: List[Dict[str, Any]]
    outcome_col: str  # Outcome variable (Y)
    running_col: str  # Running/forcing variable (X)
    cutoff: float  # Threshold value
    bandwidth: Optional[float] = None  # Bandwidth around cutoff
    polynomial_order: int = 1  # 1 for linear, 2 for quadratic
    kernel: str = "uniform"  # uniform, triangular, epanechnikov
    covariates: Optional[List[str]] = None  # Additional control variables


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


def calculate_optimal_bandwidth(running_var: np.ndarray, outcome: np.ndarray, 
                                 cutoff: float) -> float:
    """Calculate optimal bandwidth using Imbens-Kalyanaraman method (simplified)"""
    # Simplified IK bandwidth selector
    n = len(running_var)
    std_x = running_var.std()
    
    # Rule of thumb bandwidth
    h = 1.06 * std_x * (n ** (-1/5))
    
    # Scale to reasonable range around cutoff
    range_x = running_var.max() - running_var.min()
    h = min(h, range_x / 4)
    h = max(h, range_x / 20)
    
    return h


def get_kernel_weights(distance: np.ndarray, bandwidth: float, kernel: str) -> np.ndarray:
    """Calculate kernel weights for local regression"""
    u = distance / bandwidth
    
    if kernel == "uniform":
        weights = np.where(np.abs(u) <= 1, 1.0, 0.0)
    elif kernel == "triangular":
        weights = np.where(np.abs(u) <= 1, 1 - np.abs(u), 0.0)
    elif kernel == "epanechnikov":
        weights = np.where(np.abs(u) <= 1, 0.75 * (1 - u**2), 0.0)
    else:
        weights = np.where(np.abs(u) <= 1, 1.0, 0.0)
    
    return weights


def estimate_rdd_local_linear(df: pd.DataFrame, outcome_col: str, running_col: str,
                               cutoff: float, bandwidth: float, kernel: str,
                               polynomial_order: int = 1) -> Dict[str, Any]:
    """Estimate RDD using local linear regression"""
    
    df = df.copy()
    
    # Center running variable at cutoff
    df['centered'] = df[running_col] - cutoff
    df['treated'] = (df[running_col] >= cutoff).astype(int)
    
    # Filter to bandwidth
    in_bandwidth = np.abs(df['centered']) <= bandwidth
    df_bw = df[in_bandwidth].copy()
    
    if len(df_bw) < 10:
        return {'error': 'Insufficient observations within bandwidth'}
    
    # Calculate kernel weights
    weights = get_kernel_weights(df_bw['centered'].values, bandwidth, kernel)
    
    # Create polynomial features
    if polynomial_order == 1:
        df_bw['centered_treated'] = df_bw['centered'] * df_bw['treated']
        X = df_bw[['treated', 'centered', 'centered_treated']]
    else:
        df_bw['centered_sq'] = df_bw['centered'] ** 2
        df_bw['centered_treated'] = df_bw['centered'] * df_bw['treated']
        df_bw['centered_sq_treated'] = df_bw['centered_sq'] * df_bw['treated']
        X = df_bw[['treated', 'centered', 'centered_sq', 'centered_treated', 'centered_sq_treated']]
    
    X = sm.add_constant(X)
    y = df_bw[outcome_col]
    
    # Weighted least squares
    model = sm.WLS(y, X, weights=weights).fit()
    
    # RDD estimate is coefficient on 'treated'
    rdd_estimate = model.params['treated']
    std_error = model.bse['treated']
    t_stat = model.tvalues['treated']
    p_value = model.pvalues['treated']
    ci_lower = model.conf_int().loc['treated', 0]
    ci_upper = model.conf_int().loc['treated', 1]
    
    # Calculate means at cutoff
    left_of_cutoff = df_bw[df_bw['treated'] == 0][outcome_col]
    right_of_cutoff = df_bw[df_bw['treated'] == 1][outcome_col]
    
    return {
        'rdd_estimate': _to_native_type(rdd_estimate),
        'std_error': _to_native_type(std_error),
        't_statistic': _to_native_type(t_stat),
        'p_value': _to_native_type(p_value),
        'ci_lower': _to_native_type(ci_lower),
        'ci_upper': _to_native_type(ci_upper),
        'significant': bool(p_value < 0.05),
        'n_left': len(left_of_cutoff),
        'n_right': len(right_of_cutoff),
        'n_total': len(df_bw),
        'mean_left': _to_native_type(left_of_cutoff.mean()),
        'mean_right': _to_native_type(right_of_cutoff.mean()),
        'r_squared': _to_native_type(model.rsquared),
        'model_coefficients': {k: _to_native_type(v) for k, v in model.params.items()}
    }


def run_robustness_checks(df: pd.DataFrame, outcome_col: str, running_col: str,
                          cutoff: float, base_bandwidth: float, kernel: str) -> List[Dict]:
    """Run robustness checks with different bandwidths"""
    results = []
    
    bandwidth_multipliers = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    
    for mult in bandwidth_multipliers:
        bw = base_bandwidth * mult
        try:
            result = estimate_rdd_local_linear(df, outcome_col, running_col, 
                                                cutoff, bw, kernel, 1)
            if 'error' not in result:
                results.append({
                    'bandwidth': _to_native_type(bw),
                    'multiplier': mult,
                    'estimate': result['rdd_estimate'],
                    'std_error': result['std_error'],
                    'p_value': result['p_value'],
                    'n_obs': result['n_total'],
                    'significant': result['significant']
                })
        except:
            pass
    
    return results


def test_manipulation(running_var: np.ndarray, cutoff: float, 
                       n_bins: int = 20) -> Dict[str, Any]:
    """McCrary density test for manipulation at cutoff"""
    
    # Create bins
    bin_width = (running_var.max() - running_var.min()) / n_bins
    bins = np.arange(running_var.min(), running_var.max() + bin_width, bin_width)
    
    hist, bin_edges = np.histogram(running_var, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Find bin containing cutoff
    cutoff_bin_idx = np.searchsorted(bin_centers, cutoff)
    
    # Compare density left vs right of cutoff
    if cutoff_bin_idx > 0 and cutoff_bin_idx < len(hist):
        left_density = hist[:cutoff_bin_idx].mean()
        right_density = hist[cutoff_bin_idx:].mean()
        
        # Simple chi-square test
        expected = (left_density + right_density) / 2
        if expected > 0:
            chi_sq = ((left_density - expected)**2 + (right_density - expected)**2) / expected
            p_value = 1 - stats.chi2.cdf(chi_sq, df=1)
        else:
            chi_sq = 0
            p_value = 1
        
        manipulation_detected = bool(p_value < 0.05)
    else:
        left_density = right_density = chi_sq = p_value = None
        manipulation_detected = None
    
    return {
        'left_density': _to_native_type(left_density),
        'right_density': _to_native_type(right_density),
        'density_ratio': _to_native_type(right_density / left_density) if left_density and left_density > 0 else None,
        'chi_square': _to_native_type(chi_sq),
        'p_value': _to_native_type(p_value),
        'manipulation_detected': manipulation_detected,
        'message': 'No evidence of manipulation' if not manipulation_detected else 'Potential manipulation detected'
    }


def test_covariate_balance(df: pd.DataFrame, running_col: str, cutoff: float,
                            bandwidth: float, covariates: List[str]) -> List[Dict]:
    """Test covariate balance at cutoff"""
    balance_results = []
    
    df = df.copy()
    df['treated'] = (df[running_col] >= cutoff).astype(int)
    
    # Filter to bandwidth
    in_bandwidth = np.abs(df[running_col] - cutoff) <= bandwidth
    df_bw = df[in_bandwidth]
    
    for cov in covariates:
        if cov not in df_bw.columns:
            continue
        
        left = df_bw[df_bw['treated'] == 0][cov].dropna()
        right = df_bw[df_bw['treated'] == 1][cov].dropna()
        
        if len(left) < 2 or len(right) < 2:
            continue
        
        t_stat, p_value = stats.ttest_ind(left, right)
        
        balance_results.append({
            'covariate': cov,
            'mean_left': _to_native_type(left.mean()),
            'mean_right': _to_native_type(right.mean()),
            'difference': _to_native_type(right.mean() - left.mean()),
            't_statistic': _to_native_type(t_stat),
            'p_value': _to_native_type(p_value),
            'balanced': bool(p_value > 0.05)
        })
    
    return balance_results


def generate_rdd_plot(df: pd.DataFrame, outcome_col: str, running_col: str,
                       cutoff: float, bandwidth: float, rdd_result: Dict) -> str:
    """Generate main RDD visualization"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Scatter plot
    left_mask = df[running_col] < cutoff
    right_mask = df[running_col] >= cutoff
    
    ax.scatter(df.loc[left_mask, running_col], df.loc[left_mask, outcome_col],
               alpha=0.4, color='#3b82f6', s=40, label='Below Cutoff')
    ax.scatter(df.loc[right_mask, running_col], df.loc[right_mask, outcome_col],
               alpha=0.4, color='#ef4444', s=40, label='Above Cutoff')
    
    # Fit lines for each side
    for mask, color, side in [(left_mask, '#3b82f6', 'left'), (right_mask, '#ef4444', 'right')]:
        x = df.loc[mask, running_col].values.reshape(-1, 1)
        y = df.loc[mask, outcome_col].values
        
        if len(x) > 2:
            model = LinearRegression()
            model.fit(x, y)
            x_line = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
            y_line = model.predict(x_line)
            ax.plot(x_line, y_line, color=color, linewidth=3, alpha=0.8)
    
    # Cutoff line
    ax.axvline(x=cutoff, color='black', linestyle='--', linewidth=2, label=f'Cutoff = {cutoff}')
    
    # Bandwidth region
    ax.axvspan(cutoff - bandwidth, cutoff + bandwidth, alpha=0.1, color='gray', label='Bandwidth')
    
    # Discontinuity annotation
    if rdd_result and 'rdd_estimate' in rdd_result:
        jump = rdd_result['rdd_estimate']
        ax.annotate(f'Jump = {jump:.2f}', 
                   xy=(cutoff, (rdd_result['mean_left'] + rdd_result['mean_right']) / 2),
                   xytext=(cutoff + bandwidth * 0.5, rdd_result['mean_right']),
                   fontsize=12, fontweight='bold',
                   arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    ax.set_xlabel(running_col, fontsize=12)
    ax.set_ylabel(outcome_col, fontsize=12)
    ax.set_title('Regression Discontinuity Design', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_density_plot(running_var: np.ndarray, cutoff: float) -> str:
    """Generate McCrary density plot"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Histogram
    bins = 40
    ax.hist(running_var, bins=bins, density=True, alpha=0.6, color='#3b82f6', 
            edgecolor='white', linewidth=1)
    
    # KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(running_var)
    x_range = np.linspace(running_var.min(), running_var.max(), 200)
    ax.plot(x_range, kde(x_range), color='#1d4ed8', linewidth=2, label='Density')
    
    # Cutoff line
    ax.axvline(x=cutoff, color='red', linestyle='--', linewidth=2, label=f'Cutoff = {cutoff}')
    
    ax.set_xlabel('Running Variable', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('McCrary Density Test: Running Variable Distribution', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_robustness_plot(robustness_results: List[Dict]) -> str:
    """Generate robustness check plot"""
    if not robustness_results:
        return None
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bandwidths = [r['bandwidth'] for r in robustness_results]
    estimates = [r['estimate'] for r in robustness_results]
    std_errors = [r['std_error'] for r in robustness_results]
    
    # Point estimates with error bars
    ax.errorbar(bandwidths, estimates, yerr=[1.96 * se for se in std_errors],
                fmt='o-', color='#3b82f6', linewidth=2, markersize=10,
                capsize=5, capthick=2, label='RDD Estimate ± 95% CI')
    
    # Zero line
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    
    # Highlight base bandwidth
    base_bw = robustness_results[2]['bandwidth'] if len(robustness_results) > 2 else bandwidths[0]
    ax.axvline(x=base_bw, color='green', linestyle=':', linewidth=2, alpha=0.7, label='Base Bandwidth')
    
    ax.set_xlabel('Bandwidth', fontsize=12)
    ax.set_ylabel('RDD Estimate', fontsize=12)
    ax.set_title('Robustness Check: Sensitivity to Bandwidth Choice', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_local_plot(df: pd.DataFrame, outcome_col: str, running_col: str,
                         cutoff: float, bandwidth: float) -> str:
    """Generate local polynomial fit plot"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Filter to bandwidth
    df = df.copy()
    in_bw = np.abs(df[running_col] - cutoff) <= bandwidth * 1.5
    df_local = df[in_bw]
    
    left_mask = df_local[running_col] < cutoff
    right_mask = df_local[running_col] >= cutoff
    
    # Scatter
    ax.scatter(df_local.loc[left_mask, running_col], df_local.loc[left_mask, outcome_col],
               alpha=0.5, color='#3b82f6', s=60)
    ax.scatter(df_local.loc[right_mask, running_col], df_local.loc[right_mask, outcome_col],
               alpha=0.5, color='#ef4444', s=60)
    
    # Polynomial fits
    for mask, color in [(left_mask, '#3b82f6'), (right_mask, '#ef4444')]:
        x = df_local.loc[mask, running_col].values
        y = df_local.loc[mask, outcome_col].values
        
        if len(x) > 3:
            # Quadratic fit
            z = np.polyfit(x, y, 2)
            p = np.poly1d(z)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, p(x_line), color=color, linewidth=3)
    
    ax.axvline(x=cutoff, color='black', linestyle='--', linewidth=2)
    ax.axvspan(cutoff - bandwidth, cutoff + bandwidth, alpha=0.15, color='yellow')
    
    ax.set_xlabel(running_col, fontsize=12)
    ax.set_ylabel(outcome_col, fontsize=12)
    ax.set_title('Local Polynomial Regression (Quadratic)', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def generate_interpretation(rdd_result: Dict, manipulation_test: Dict,
                            robustness_results: List[Dict]) -> Dict[str, Any]:
    """Generate comprehensive interpretation"""
    key_insights = []
    
    # Main effect
    if rdd_result.get('significant'):
        key_insights.append({
            'title': 'Significant Discontinuity',
            'description': f"There is a significant jump of {rdd_result['rdd_estimate']:.3f} at the cutoff (p = {rdd_result['p_value']:.4f}).",
            'status': 'positive'
        })
    else:
        key_insights.append({
            'title': 'No Significant Discontinuity',
            'description': f"The estimated jump of {rdd_result['rdd_estimate']:.3f} is not statistically significant (p = {rdd_result['p_value']:.4f}).",
            'status': 'negative'
        })
    
    # Manipulation test
    if manipulation_test.get('manipulation_detected'):
        key_insights.append({
            'title': 'Warning: Potential Manipulation',
            'description': 'The density test suggests possible manipulation of the running variable at the cutoff.',
            'status': 'warning'
        })
    else:
        key_insights.append({
            'title': 'No Manipulation Detected',
            'description': 'The running variable shows no evidence of manipulation at the cutoff.',
            'status': 'positive'
        })
    
    # Robustness
    if robustness_results:
        sig_count = sum(1 for r in robustness_results if r.get('significant'))
        total = len(robustness_results)
        key_insights.append({
            'title': 'Robustness Check',
            'description': f"Estimate is significant in {sig_count}/{total} bandwidth specifications.",
            'status': 'positive' if sig_count >= total * 0.6 else 'warning'
        })
    
    # Sample info
    key_insights.append({
        'title': 'Sample Information',
        'description': f"Analysis uses {rdd_result['n_total']} observations within bandwidth ({rdd_result['n_left']} below, {rdd_result['n_right']} above cutoff).",
        'status': 'neutral'
    })
    
    return {
        'key_insights': key_insights,
        'recommendation': 'RDD assumptions appear satisfied. Causal interpretation is warranted.' if rdd_result.get('significant') and not manipulation_test.get('manipulation_detected') else 'Interpret with caution due to validity concerns.'
    }


@router.post("/regression-discontinuity")
async def run_rdd_analysis(request: RDDRequest) -> Dict[str, Any]:
    """
    Perform Regression Discontinuity Design (RDD) Analysis.
    
    Estimates causal effects at a threshold using local linear regression.
    """
    try:
        data = request.data
        outcome_col = request.outcome_col
        running_col = request.running_col
        cutoff = request.cutoff
        bandwidth = request.bandwidth
        polynomial_order = request.polynomial_order
        kernel = request.kernel
        covariates = request.covariates or []
        
        if not data:
            raise HTTPException(status_code=400, detail="Data not provided.")
        
        df = pd.DataFrame(data)
        
        # Validate columns
        for col in [outcome_col, running_col]:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Column '{col}' not found.")
        
        # Convert to numeric
        df[outcome_col] = pd.to_numeric(df[outcome_col], errors='coerce')
        df[running_col] = pd.to_numeric(df[running_col], errors='coerce')
        df = df.dropna(subset=[outcome_col, running_col])
        
        if len(df) < 20:
            raise HTTPException(status_code=400, detail="At least 20 observations required.")
        
        # Check cutoff validity
        if cutoff <= df[running_col].min() or cutoff >= df[running_col].max():
            raise HTTPException(status_code=400, detail="Cutoff must be within data range.")
        
        # Calculate optimal bandwidth if not provided
        if bandwidth is None:
            bandwidth = calculate_optimal_bandwidth(
                df[running_col].values, df[outcome_col].values, cutoff
            )
        
        # Main RDD estimation
        rdd_result = estimate_rdd_local_linear(
            df, outcome_col, running_col, cutoff, bandwidth, kernel, polynomial_order
        )
        
        if 'error' in rdd_result:
            raise HTTPException(status_code=400, detail=rdd_result['error'])
        
        # Robustness checks
        robustness_results = run_robustness_checks(df, outcome_col, running_col, 
                                                    cutoff, bandwidth, kernel)
        
        # Manipulation test
        manipulation_test = test_manipulation(df[running_col].values, cutoff)
        
        # Covariate balance (if covariates provided)
        covariate_balance = []
        if covariates:
            covariate_balance = test_covariate_balance(df, running_col, cutoff, 
                                                        bandwidth, covariates)
        
        # Generate visualizations
        rdd_plot = generate_rdd_plot(df, outcome_col, running_col, cutoff, bandwidth, rdd_result)
        density_plot = generate_density_plot(df[running_col].values, cutoff)
        robustness_plot = generate_robustness_plot(robustness_results)
        local_plot = generate_local_plot(df, outcome_col, running_col, cutoff, bandwidth)
        
        # Interpretation
        interpretation = generate_interpretation(rdd_result, manipulation_test, robustness_results)
        
        # Descriptive stats
        descriptive_stats = {
            'n_total': len(df),
            'n_below_cutoff': int((df[running_col] < cutoff).sum()),
            'n_above_cutoff': int((df[running_col] >= cutoff).sum()),
            'running_var_mean': _to_native_type(df[running_col].mean()),
            'running_var_std': _to_native_type(df[running_col].std()),
            'outcome_mean': _to_native_type(df[outcome_col].mean()),
            'outcome_std': _to_native_type(df[outcome_col].std())
        }
        
        return {
            'rdd_estimate': rdd_result,
            'bandwidth_used': _to_native_type(bandwidth),
            'cutoff': cutoff,
            'polynomial_order': polynomial_order,
            'kernel': kernel,
            'robustness_checks': robustness_results,
            'manipulation_test': manipulation_test,
            'covariate_balance': covariate_balance,
            'descriptive_stats': descriptive_stats,
            'rdd_plot': rdd_plot,
            'density_plot': density_plot,
            'robustness_plot': robustness_plot,
            'local_plot': local_plot,
            'interpretation': interpretation
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RDD analysis failed: {str(e)}")
