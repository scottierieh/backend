"""
Gage R&R (Repeatability and Reproducibility) Analysis FastAPI Endpoint
Measurement System Analysis (MSA) for evaluating measurement quality
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import base64
import warnings
from scipy import stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm

warnings.filterwarnings('ignore')
sns.set_style("darkgrid")
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

router = APIRouter()


class GageRRRequest(BaseModel):
    """Request model for Gage R&R Analysis"""
    data: List[Dict[str, Any]]
    part_col: str
    operator_col: str
    measurement_col: str
    tolerance: Optional[float] = None
    confidence_level: float = Field(default=0.95, ge=0.9, le=0.99)


def calculate_anova_components(df: pd.DataFrame, part_col: str, operator_col: str, measurement_col: str):
    """Calculate ANOVA variance components using statsmodels"""
    
    # Get counts
    parts = df[part_col].unique()
    operators = df[operator_col].unique()
    n_parts = len(parts)
    n_operators = len(operators)
    n_trials = len(df) // (n_parts * n_operators)
    
    # Overall mean
    grand_mean = df[measurement_col].mean()
    
    # Prepare data for statsmodels
    df_anova = df.copy()
    df_anova['Part'] = pd.Categorical(df_anova[part_col])
    df_anova['Operator'] = pd.Categorical(df_anova[operator_col])
    df_anova['Measurement'] = df_anova[measurement_col]
    
    # Fit ANOVA model using statsmodels
    try:
        model = ols('Measurement ~ C(Part) + C(Operator) + C(Part):C(Operator)', data=df_anova).fit()
        anova_table = anova_lm(model, typ=2)
        
        # Extract Mean Squares
        ms_part = anova_table.loc['C(Part)', 'mean_sq']
        ms_operator = anova_table.loc['C(Operator)', 'mean_sq']
        ms_interaction = anova_table.loc['C(Part):C(Operator)', 'mean_sq']
        ms_equipment = anova_table.loc['Residual', 'mean_sq']
        
        df_interaction = int(anova_table.loc['C(Part):C(Operator)', 'df'])
        df_equipment = int(anova_table.loc['Residual', 'df'])
        
    except Exception as e:
        print(f"statsmodels failed, using manual calculation: {e}")
        # Fallback
        part_means = df.groupby(part_col)[measurement_col].mean()
        operator_means = df.groupby(operator_col)[measurement_col].mean()
        ss_part = n_operators * n_trials * np.sum((part_means - grand_mean) ** 2)
        ss_operator = n_parts * n_trials * np.sum((operator_means - grand_mean) ** 2)
        df_part = n_parts - 1
        df_operator = n_operators - 1
        df_interaction = df_part * df_operator
        ms_part = ss_part / df_part if df_part > 0 else 0
        ms_operator = ss_operator / df_operator if df_operator > 0 else 0
        ms_interaction = 0.0001
        ms_equipment = df[measurement_col].var()
        df_equipment = len(df) - n_parts * n_operators
    
    # Calculate variance components
    var_equipment = ms_equipment
    
    if ms_interaction > ms_equipment:
        f_interaction = ms_interaction / ms_equipment
        p_interaction = 1 - stats.f.cdf(f_interaction, df_interaction, df_equipment)
    else:
        p_interaction = 1.0
    
    if p_interaction < 0.25:
        var_reproducibility = (ms_operator - ms_interaction) / (n_parts * n_trials)
        var_interaction = (ms_interaction - ms_equipment) / n_trials
        var_operator = max(0, var_reproducibility)
    else:
        var_operator = max(0, (ms_operator - ms_equipment) / (n_parts * n_trials))
        var_interaction = 0
    
    var_part = max(0, (ms_part - ms_interaction) / (n_operators * n_trials)) if ms_interaction > 0 else max(0, (ms_part - ms_equipment) / (n_operators * n_trials))
    
    var_repeatability = var_equipment
    var_reproducibility_total = var_operator + var_interaction
    var_gage_rr = var_repeatability + var_reproducibility_total
    var_total = var_gage_rr + var_part
    
    return {
        'var_part': var_part,
        'var_repeatability': var_repeatability,
        'var_operator': var_operator,
        'var_interaction': var_interaction,
        'var_reproducibility': var_reproducibility_total,
        'var_gage_rr': var_gage_rr,
        'var_total': var_total,
        'n_parts': n_parts,
        'n_operators': n_operators,
        'n_trials': n_trials,
        'grand_mean': grand_mean,
        'p_interaction': p_interaction
    }


def calculate_study_variation(variance_components: Dict[str, float], tolerance: Optional[float]):
    """Calculate %Study Variation and related metrics"""
    
    # Standard deviations (using 5.15 multiplier for 99% of distribution)
    k = 5.15
    
    sd_part = np.sqrt(variance_components['var_part'])
    sd_repeatability = np.sqrt(variance_components['var_repeatability'])
    sd_reproducibility = np.sqrt(variance_components['var_reproducibility'])
    sd_gage_rr = np.sqrt(variance_components['var_gage_rr'])
    sd_total = np.sqrt(variance_components['var_total'])
    
    # Study Variation (SV)
    sv_part = k * sd_part
    sv_repeatability = k * sd_repeatability
    sv_reproducibility = k * sd_reproducibility
    sv_gage_rr = k * sd_gage_rr
    sv_total = k * sd_total
    
    # %Study Variation (%SV) - percentage of total variation
    pct_sv_part = (variance_components['var_part'] / variance_components['var_total'] * 100) if variance_components['var_total'] > 0 else 0
    pct_sv_repeatability = (variance_components['var_repeatability'] / variance_components['var_total'] * 100) if variance_components['var_total'] > 0 else 0
    pct_sv_reproducibility = (variance_components['var_reproducibility'] / variance_components['var_total'] * 100) if variance_components['var_total'] > 0 else 0
    pct_sv_gage_rr = (variance_components['var_gage_rr'] / variance_components['var_total'] * 100) if variance_components['var_total'] > 0 else 0
    
    # %Tolerance (if tolerance provided)
    results = {
        'sd_part': sd_part,
        'sd_repeatability': sd_repeatability,
        'sd_reproducibility': sd_reproducibility,
        'sd_gage_rr': sd_gage_rr,
        'sd_total': sd_total,
        'sv_part': sv_part,
        'sv_repeatability': sv_repeatability,
        'sv_reproducibility': sv_reproducibility,
        'sv_gage_rr': sv_gage_rr,
        'sv_total': sv_total,
        'pct_sv_part': pct_sv_part,
        'pct_sv_repeatability': pct_sv_repeatability,
        'pct_sv_reproducibility': pct_sv_reproducibility,
        'pct_sv_gage_rr': pct_sv_gage_rr
    }
    
    if tolerance:
        pct_tol_repeatability = (sv_repeatability / tolerance * 100)
        pct_tol_reproducibility = (sv_reproducibility / tolerance * 100)
        pct_tol_gage_rr = (sv_gage_rr / tolerance * 100)
        
        results.update({
            'pct_tol_repeatability': pct_tol_repeatability,
            'pct_tol_reproducibility': pct_tol_reproducibility,
            'pct_tol_gage_rr': pct_tol_gage_rr,
            'tolerance': tolerance
        })
    
    return results


def calculate_ndc(variance_components: Dict[str, float]) -> int:
    """Calculate Number of Distinct Categories (NDC)"""
    
    var_part = variance_components['var_part']
    var_gage_rr = variance_components['var_gage_rr']
    
    if var_gage_rr > 0:
        ndc = int(np.floor(1.41 * np.sqrt(var_part / var_gage_rr)))
    else:
        ndc = 0
    
    return max(1, ndc)


def assess_measurement_system(pct_gage_rr: float, ndc: int) -> Dict[str, str]:
    """Assess measurement system acceptability"""
    
    # AIAG criteria
    if pct_gage_rr < 10:
        gage_status = "Acceptable"
        gage_color = "green"
    elif pct_gage_rr < 30:
        gage_status = "Marginal"
        gage_color = "yellow"
    else:
        gage_status = "Unacceptable"
        gage_color = "red"
    
    # NDC criteria
    if ndc >= 5:
        ndc_status = "Acceptable"
        ndc_color = "green"
    elif ndc >= 2:
        ndc_status = "Marginal"
        ndc_color = "yellow"
    else:
        ndc_status = "Unacceptable"
        ndc_color = "red"
    
    return {
        'gage_status': gage_status,
        'gage_color': gage_color,
        'ndc_status': ndc_status,
        'ndc_color': ndc_color
    }


def generate_visualizations(df: pd.DataFrame, part_col: str, operator_col: str, 
                           measurement_col: str, variance_components: Dict[str, float],
                           study_variation: Dict[str, float]):
    """Generate Gage R&R visualizations"""
    
    visualizations = {}
    
    # 1. Variance Components Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    components = ['Part-to-Part', 'Repeatability', 'Reproducibility', 'Gage R&R']
    variances = [
        variance_components['var_part'],
        variance_components['var_repeatability'],
        variance_components['var_reproducibility'],
        variance_components['var_gage_rr']
    ]
    percentages = [
        study_variation['pct_sv_part'],
        study_variation['pct_sv_repeatability'],
        study_variation['pct_sv_reproducibility'],
        study_variation['pct_sv_gage_rr']
    ]
    
    colors = ['#4A90E2', '#50C878', '#FFA500', '#E74C3C']
    x = np.arange(len(components))
    
    bars = ax.bar(x, percentages, color=colors, alpha=0.8, edgecolor='black')
    
    # Add percentage labels on bars
    for i, (bar, pct) in enumerate(zip(bars, percentages)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{pct:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('% Contribution to Total Variation', fontsize=11)
    ax.set_xlabel('Variance Component', fontsize=11)
    ax.set_title('Variance Components Analysis', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(components, rotation=45, ha='right')
    ax.axhline(y=10, color='green', linestyle='--', alpha=0.5, label='10% Threshold (Acceptable)')
    ax.axhline(y=30, color='orange', linestyle='--', alpha=0.5, label='30% Threshold (Marginal)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    visualizations['variance_components'] = fig_to_base64(fig)
    
    # 2. Xbar Chart by Operator
    fig, ax = plt.subplots(figsize=(12, 6))
    
    operators = sorted(df[operator_col].unique())
    parts = sorted(df[part_col].unique())
    
    for operator in operators:
        op_data = df[df[operator_col] == operator]
        part_means = op_data.groupby(part_col)[measurement_col].mean()
        ax.plot(parts, part_means, marker='o', label=operator, linewidth=2)
    
    grand_mean = df[measurement_col].mean()
    ax.axhline(y=grand_mean, color='red', linestyle='--', label=f'Grand Mean: {grand_mean:.3f}')
    
    ax.set_xlabel('Part', fontsize=11)
    ax.set_ylabel('Average Measurement', fontsize=11)
    ax.set_title('Xbar Chart by Operator', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    visualizations['xbar_chart'] = fig_to_base64(fig)
    
    # 3. Range Chart by Operator
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for operator in operators:
        op_data = df[df[operator_col] == operator]
        part_ranges = op_data.groupby(part_col)[measurement_col].apply(lambda x: x.max() - x.min())
        ax.plot(parts, part_ranges, marker='s', label=operator, linewidth=2)
    
    avg_range = df.groupby([part_col, operator_col])[measurement_col].apply(lambda x: x.max() - x.min()).mean()
    ax.axhline(y=avg_range, color='red', linestyle='--', label=f'Avg Range: {avg_range:.3f}')
    
    ax.set_xlabel('Part', fontsize=11)
    ax.set_ylabel('Range', fontsize=11)
    ax.set_title('Range Chart by Operator', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    visualizations['range_chart'] = fig_to_base64(fig)
    
    # 4. Operator Comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    operator_means = df.groupby(operator_col)[measurement_col].mean()
    operator_stds = df.groupby(operator_col)[measurement_col].std()
    
    x = np.arange(len(operators))
    ax.bar(x, operator_means, yerr=operator_stds, capsize=5, alpha=0.7, 
           color='#4A90E2', edgecolor='black')
    
    ax.axhline(y=grand_mean, color='red', linestyle='--', label=f'Grand Mean: {grand_mean:.3f}')
    ax.set_xlabel('Operator', fontsize=11)
    ax.set_ylabel('Average Measurement', fontsize=11)
    ax.set_title('Measurement by Operator', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(operators)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    visualizations['operator_comparison'] = fig_to_base64(fig)
    
    # 5. Part Variation
    fig, ax = plt.subplots(figsize=(12, 6))
    
    part_data = []
    part_labels = []
    for part in parts:
        part_measurements = df[df[part_col] == part][measurement_col]
        part_data.append(part_measurements)
        part_labels.append(str(part))
    
    bp = ax.boxplot(part_data, labels=part_labels, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('#4A90E2')
        patch.set_alpha(0.7)
    
    ax.set_xlabel('Part', fontsize=11)
    ax.set_ylabel('Measurement', fontsize=11)
    ax.set_title('Measurement by Part', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    visualizations['part_variation'] = fig_to_base64(fig)
    
    return visualizations


def fig_to_base64(fig):
    """Convert matplotlib figure to base64"""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64


def generate_insights(variance_components: Dict[str, float], study_variation: Dict[str, float], 
                     assessment: Dict[str, str], ndc: int, tolerance: Optional[float]):
    """Generate key insights"""
    
    insights = []
    
    pct_gage_rr = study_variation['pct_sv_gage_rr']
    
    # Overall system assessment
    if assessment['gage_status'] == 'Acceptable':
        insights.append({
            'title': 'Measurement System Acceptable',
            'description': f'Gage R&R of {pct_gage_rr:.1f}% is less than 10%, indicating excellent measurement system capability.',
            'status': 'positive'
        })
    elif assessment['gage_status'] == 'Marginal':
        insights.append({
            'title': 'Marginal Measurement System',
            'description': f'Gage R&R of {pct_gage_rr:.1f}% is between 10-30%. System may be acceptable depending on application criticality and improvement cost.',
            'status': 'warning'
        })
    else:
        insights.append({
            'title': 'Unacceptable Measurement System',
            'description': f'Gage R&R of {pct_gage_rr:.1f}% exceeds 30%. Measurement system requires improvement before use for decision-making.',
            'status': 'warning'
        })
    
    # Repeatability vs Reproducibility
    pct_repeat = study_variation['pct_sv_repeatability']
    pct_reprod = study_variation['pct_sv_reproducibility']
    
    if pct_repeat > pct_reprod * 1.5:
        insights.append({
            'title': 'Repeatability is Primary Issue',
            'description': f'Repeatability ({pct_repeat:.1f}%) exceeds reproducibility ({pct_reprod:.1f}%). Focus on equipment calibration, maintenance, and measurement procedure clarity.',
            'status': 'neutral'
        })
    elif pct_reprod > pct_repeat * 1.5:
        insights.append({
            'title': 'Reproducibility is Primary Issue',
            'description': f'Reproducibility ({pct_reprod:.1f}%) exceeds repeatability ({pct_repeat:.1f}%). Focus on operator training, standardized procedures, and potential fixture improvements.',
            'status': 'neutral'
        })
    
    # NDC assessment
    if ndc >= 5:
        insights.append({
            'title': 'Excellent Discrimination',
            'description': f'NDC of {ndc} indicates measurement system can distinguish {ndc} distinct categories. Excellent for process control and capability studies.',
            'status': 'positive'
        })
    elif ndc >= 2:
        insights.append({
            'title': 'Adequate Discrimination',
            'description': f'NDC of {ndc} is marginally acceptable. System can detect large process changes but may miss smaller variations.',
            'status': 'neutral'
        })
    else:
        insights.append({
            'title': 'Poor Discrimination',
            'description': f'NDC of {ndc} indicates system cannot adequately distinguish between parts. Measurement system improvement required.',
            'status': 'warning'
        })
    
    # Tolerance-based assessment
    if tolerance and 'pct_tol_gage_rr' in study_variation:
        pct_tol = study_variation['pct_tol_gage_rr']
        if pct_tol < 10:
            insights.append({
                'title': 'Excellent vs Tolerance',
                'description': f'Gage R&R is {pct_tol:.1f}% of tolerance, well below 10% threshold. Measurement precision is excellent relative to specification.',
                'status': 'positive'
            })
        elif pct_tol > 30:
            insights.append({
                'title': 'Poor vs Tolerance',
                'description': f'Gage R&R is {pct_tol:.1f}% of tolerance, exceeding 30% threshold. Measurement error consumes significant portion of tolerance band.',
                'status': 'warning'
            })
    
    return insights


@router.post("/gage-rr-analysis")
async def analyze_gage_rr(request: GageRRRequest):
    """
    Gage R&R Analysis Endpoint
    
    Evaluates measurement system capability using ANOVA method
    """
    try:
        if not request.data:
            raise HTTPException(400, "No data provided")
        
        df = pd.DataFrame(request.data)
        
        required_cols = [request.part_col, request.operator_col, request.measurement_col]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise HTTPException(400, f"Missing columns: {missing}")
        
        # Check data structure
        n_parts = df[request.part_col].nunique()
        n_operators = df[request.operator_col].nunique()
        n_measurements = len(df)
        
        if n_parts < 5:
            raise HTTPException(400, "Need at least 5 parts for reliable analysis")
        if n_operators < 2:
            raise HTTPException(400, "Need at least 2 operators for reproducibility analysis")
        
        # Calculate ANOVA variance components
        variance_components = calculate_anova_components(
            df, request.part_col, request.operator_col, request.measurement_col
        )
        
        # Calculate study variation
        study_variation = calculate_study_variation(variance_components, request.tolerance)
        
        # Calculate NDC
        ndc = calculate_ndc(variance_components)
        
        # Assess measurement system
        assessment = assess_measurement_system(study_variation['pct_sv_gage_rr'], ndc)
        
        # Generate visualizations
        visualizations = generate_visualizations(
            df, request.part_col, request.operator_col, request.measurement_col,
            variance_components, study_variation
        )
        
        # Generate insights
        insights = generate_insights(
            variance_components, study_variation, assessment, ndc, request.tolerance
        )
        
        # Prepare response
        response_data = {
            'success': True,
            'results': {
                'n_parts': variance_components['n_parts'],
                'n_operators': variance_components['n_operators'],
                'n_trials': variance_components['n_trials'],
                'grand_mean': float(variance_components['grand_mean']),
                'variance_components': {
                    'part': float(variance_components['var_part']),
                    'repeatability': float(variance_components['var_repeatability']),
                    'reproducibility': float(variance_components['var_reproducibility']),
                    'operator': float(variance_components['var_operator']),
                    'interaction': float(variance_components['var_interaction']),
                    'gage_rr': float(variance_components['var_gage_rr']),
                    'total': float(variance_components['var_total'])
                },
                'study_variation': {
                    'pct_part': float(study_variation['pct_sv_part']),
                    'pct_repeatability': float(study_variation['pct_sv_repeatability']),
                    'pct_reproducibility': float(study_variation['pct_sv_reproducibility']),
                    'pct_gage_rr': float(study_variation['pct_sv_gage_rr']),
                    'sd_gage_rr': float(study_variation['sd_gage_rr']),
                    'sv_gage_rr': float(study_variation['sv_gage_rr'])
                },
                'ndc': ndc,
                'assessment': assessment,
                'p_interaction': float(variance_components['p_interaction'])
            },
            'visualizations': visualizations,
            'key_insights': insights,
            'summary': {
                'analysis_type': 'gage_rr',
                'n_parts': n_parts,
                'n_operators': n_operators,
                'pct_gage_rr': round(study_variation['pct_sv_gage_rr'], 1),
                'ndc': ndc,
                'status': assessment['gage_status']
            }
        }
        
        # Add tolerance-based metrics if provided
        if request.tolerance:
            response_data['results']['study_variation'].update({
                'pct_tol_repeatability': float(study_variation['pct_tol_repeatability']),
                'pct_tol_reproducibility': float(study_variation['pct_tol_reproducibility']),
                'pct_tol_gage_rr': float(study_variation['pct_tol_gage_rr']),
                'tolerance': request.tolerance
            })
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")
