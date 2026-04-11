from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, List, Optional, Dict, Literal
import numpy as np
import pandas as pd
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

try:
    from scipy import stats
    from scipy.stats import chi2
    import pingouin as pg
    STATS_AVAILABLE = True
except ImportError:
    STATS_AVAILABLE = False

router = APIRouter()


# ============================================
# REQUEST MODELS
# ============================================

class TwoWayRMAnovaRequest(BaseModel):
    """Mixed design: Between × Within"""
    data: List[Dict[str, Any]] = Field(...)
    subject_col: str = Field(...)
    between_factor_col: str = Field(...)
    within_measure_cols: List[str] = Field(...)
    alpha: Optional[float] = 0.05
    post_hoc_method: Optional[Literal['bonferroni', 'tukey', 'scheffe']] = 'bonferroni'
    sphericity_correction: Optional[Literal['none', 'greenhouse-geisser', 'huynh-feldt']] = 'greenhouse-geisser'


class TwoWayWithinAnovaRequest(BaseModel):
    """Fully within-subjects: Factor1 × Factor2 (both within)"""
    data: List[Dict[str, Any]] = Field(...)
    subject_col: str = Field(...)
    within_factor1_cols: List[str] = Field(...)
    within_factor2_cols: List[str] = Field(...)
    factor1_name: Optional[str] = 'Factor1'
    factor2_name: Optional[str] = 'Factor2'
    alpha: Optional[float] = 0.05
    post_hoc_method: Optional[Literal['bonferroni', 'tukey', 'scheffe']] = 'bonferroni'
    sphericity_correction: Optional[Literal['none', 'greenhouse-geisser', 'huynh-feldt']] = 'greenhouse-geisser'


# ============================================
# HELPER FUNCTIONS
# ============================================

def _to_native(obj):
    """Convert numpy types to native Python types"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [_to_native(x) for x in obj.tolist()]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {str(k): _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_native(x) for x in obj]
    elif pd.isna(obj):
        return None
    return obj


def safe_float(val, default=0.0):
    """Safely convert to float"""
    try:
        if val is None or pd.isna(val) or np.isinf(val):
            return default
        return float(val)
    except:
        return default


def calculate_effect_size(ss_effect, ss_total, df_effect, df_error, ms_error):
    """Calculate partial eta-squared and partial omega-squared"""
    # Partial eta-squared
    partial_eta_sq = ss_effect / (ss_effect + (df_error * ms_error))
    
    # Partial omega-squared
    partial_omega_sq = (ss_effect - df_effect * ms_error) / (ss_effect + (df_error + 1) * ms_error)
    partial_omega_sq = max(0, partial_omega_sq)
    
    return safe_float(partial_eta_sq), safe_float(partial_omega_sq)


def mauchly_sphericity_test(data_wide, n_conditions):
    """
    Mauchly's test for sphericity
    Tests if variances of differences between conditions are equal
    """
    try:
        n_subjects = len(data_wide)
        
        # Calculate covariance matrix
        cov_matrix = np.cov(data_wide.T)
        
        # Calculate determinant
        det_cov = np.linalg.det(cov_matrix)
        
        # Calculate trace
        trace_cov = np.trace(cov_matrix)
        
        # Mauchly's W statistic
        k = n_conditions
        W = det_cov / ((trace_cov / k) ** k)
        
        # Chi-square approximation
        df = (k * (k - 1) / 2) - 1
        chi_sq = -(n_subjects - 1 - (2 * k + 1) / 6) * np.log(W)
        p_value = 1 - chi2.cdf(chi_sq, df)
        
        # Greenhouse-Geisser epsilon
        trace_cov_sq = np.trace(cov_matrix @ cov_matrix)
        epsilon_gg = (k * trace_cov ** 2) / ((k - 1) * (trace_cov_sq - (trace_cov ** 2 / k)))
        epsilon_gg = min(1.0, max(1 / (k - 1), epsilon_gg))
        
        # Huynh-Feldt epsilon
        epsilon_hf = (n_subjects * (k - 1) * epsilon_gg - 2) / ((k - 1) * (n_subjects - 1 - (k - 1) * epsilon_gg))
        epsilon_hf = min(1.0, epsilon_hf)
        
        return {
            'W': safe_float(W),
            'chi_square': safe_float(chi_sq),
            'df': int(df),
            'p_value': safe_float(p_value),
            'sphericity_met': bool(p_value > 0.05),
            'epsilon_gg': safe_float(epsilon_gg),
            'epsilon_hf': safe_float(epsilon_hf)
        }
    except Exception as e:
        return {
            'W': None,
            'chi_square': None,
            'df': None,
            'p_value': None,
            'sphericity_met': None,
            'epsilon_gg': None,
            'epsilon_hf': None,
            'error': str(e)
        }


def perform_pairwise_comparisons(data_long, dv_col, group_col, subject_col, method='bonferroni'):
    """Perform pairwise comparisons with correction"""
    try:
        # Use pingouin for pairwise t-tests
        pairwise = pg.pairwise_ttests(
            data=data_long,
            dv=dv_col,
            within=group_col,
            subject=subject_col,
            padjust=method,
            effsize='hedges'
        )
        
        results = []
        for _, row in pairwise.iterrows():
            results.append({
                'contrast': str(row['Contrast']),
                'a': str(row['A']),
                'b': str(row['B']),
                'mean_diff': safe_float(row.get('mean(A)', 0) - row.get('mean(B)', 0)),
                't_stat': safe_float(row['T']),
                'df': safe_float(row['dof']),
                'p_unc': safe_float(row['p-unc']),
                'p_adj': safe_float(row[f'p-{method}']),
                'effect_size': safe_float(row['hedges']),
                'significant': bool(row[f'p-{method}'] < 0.05)
            })
        
        return results
    except Exception as e:
        return []


def create_profile_plot(data_long, dv_col, within_col, between_col=None, subject_col=None):
    """Create profile plot for repeated measures"""
    fig, axes = plt.subplots(1, 2 if between_col else 1, figsize=(14 if between_col else 8, 6))
    
    if between_col:
        axes = [axes[0], axes[1]] if hasattr(axes, '__iter__') else [axes, axes]
    else:
        axes = [axes]
    
    # Plot 1: Individual trajectories + Mean
    ax1 = axes[0]
    
    if between_col:
        for group in data_long[between_col].unique():
            group_data = data_long[data_long[between_col] == group]
            
            # Individual lines (transparent)
            for subj in group_data[subject_col].unique():
                subj_data = group_data[group_data[subject_col] == subj]
                subj_data = subj_data.sort_values(within_col)
                ax1.plot(subj_data[within_col], subj_data[dv_col], 
                        alpha=0.2, linewidth=1, color='gray')
            
            # Mean line
            means = group_data.groupby(within_col)[dv_col].mean()
            sems = group_data.groupby(within_col)[dv_col].sem()
            ax1.errorbar(means.index, means.values, yerr=sems.values,
                        marker='o', linewidth=2.5, markersize=8, 
                        label=f'{between_col}={group}', capsize=5)
    else:
        # Individual lines
        for subj in data_long[subject_col].unique():
            subj_data = data_long[data_long[subject_col] == subj]
            subj_data = subj_data.sort_values(within_col)
            ax1.plot(subj_data[within_col], subj_data[dv_col], 
                    alpha=0.3, linewidth=1, color='gray')
        
        # Mean line
        means = data_long.groupby(within_col)[dv_col].mean()
        sems = data_long.groupby(within_col)[dv_col].sem()
        ax1.errorbar(means.index, means.values, yerr=sems.values,
                    marker='o', linewidth=3, markersize=10, 
                    color='#5B9BD5', label='Mean', capsize=5)
    
    ax1.set_xlabel(within_col, fontweight='bold')
    ax1.set_ylabel(dv_col, fontweight='bold')
    ax1.set_title('Profile Plot (Individual + Mean)', fontweight='bold', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Box plot by condition (if between factor exists)
    if between_col and len(axes) > 1:
        ax2 = axes[1]
        
        positions = []
        labels = []
        box_data = []
        
        within_levels = sorted(data_long[within_col].unique())
        between_levels = sorted(data_long[between_col].unique())
        
        pos = 0
        for i, w in enumerate(within_levels):
            for j, b in enumerate(between_levels):
                subset = data_long[(data_long[within_col] == w) & (data_long[between_col] == b)]
                box_data.append(subset[dv_col].values)
                positions.append(pos)
                labels.append(f'{w}\n{b}')
                pos += 1
            pos += 0.5  # Gap between within-factor levels
        
        bp = ax2.boxplot(box_data, positions=positions, labels=labels, patch_artist=True, widths=0.4)
        
        # Color by between-factor
        colors = plt.cm.Set2(np.linspace(0, 1, len(between_levels)))
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(colors[i % len(between_levels)])
            patch.set_alpha(0.7)
        
        ax2.set_ylabel(dv_col, fontweight='bold')
        ax2.set_title('Distribution by Condition', fontweight='bold', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"


def create_boxplot(data_long, dv_col, within_col, between_col=None):
    """Create detailed box plot"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if between_col:
        within_levels = sorted(data_long[within_col].unique())
        between_levels = sorted(data_long[between_col].unique())
        
        positions = []
        labels = []
        box_data = []
        
        pos = 0
        for i, w in enumerate(within_levels):
            for j, b in enumerate(between_levels):
                subset = data_long[(data_long[within_col] == w) & (data_long[between_col] == b)]
                box_data.append(subset[dv_col].values)
                positions.append(pos)
                labels.append(f'{w}\n{b}')
                pos += 1
            pos += 0.5
        
        bp = ax.boxplot(box_data, positions=positions, labels=labels, 
                       patch_artist=True, widths=0.4, showmeans=True)
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(between_levels)))
        for i, patch in enumerate(bp['boxes']):
            patch.set_facecolor(colors[i % len(between_levels)])
            patch.set_alpha(0.7)
    else:
        within_levels = sorted(data_long[within_col].unique())
        box_data = [data_long[data_long[within_col] == w][dv_col].values for w in within_levels]
        bp = ax.boxplot(box_data, labels=within_levels, patch_artist=True, 
                       widths=0.6, showmeans=True)
        
        for patch in bp['boxes']:
            patch.set_facecolor('#5B9BD5')
            patch.set_alpha(0.7)
    
    ax.set_xlabel(within_col, fontweight='bold', fontsize=12)
    ax.set_ylabel(dv_col, fontweight='bold', fontsize=12)
    ax.set_title('Box Plot - Distribution by Condition', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"


def create_mean_comparison_plot(data_long, dv_col, within_col, between_col=None):
    """Create mean comparison bar plot with error bars"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if between_col:
        within_levels = sorted(data_long[within_col].unique())
        between_levels = sorted(data_long[between_col].unique())
        
        x = np.arange(len(within_levels))
        width = 0.8 / len(between_levels)
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(between_levels)))
        
        for i, b in enumerate(between_levels):
            means = []
            sems = []
            for w in within_levels:
                subset = data_long[(data_long[within_col] == w) & (data_long[between_col] == b)]
                means.append(subset[dv_col].mean())
                sems.append(subset[dv_col].sem())
            
            offset = (i - len(between_levels)/2 + 0.5) * width
            ax.bar(x + offset, means, width, label=f'{between_col}={b}', 
                  yerr=sems, capsize=5, alpha=0.8, color=colors[i])
        
        ax.set_xlabel(within_col, fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(within_levels, rotation=45, ha='right')
        ax.legend()
    else:
        within_levels = sorted(data_long[within_col].unique())
        means = [data_long[data_long[within_col] == w][dv_col].mean() for w in within_levels]
        sems = [data_long[data_long[within_col] == w][dv_col].sem() for w in within_levels]
        
        x = np.arange(len(within_levels))
        ax.bar(x, means, yerr=sems, capsize=5, alpha=0.8, color='#5B9BD5')
        ax.set_xticks(x)
        ax.set_xticklabels(within_levels, rotation=45, ha='right')
        ax.set_xlabel(within_col, fontweight='bold', fontsize=12)
    
    ax.set_ylabel(f'Mean {dv_col}', fontweight='bold', fontsize=12)
    ax.set_title('Mean Comparison (±SE)', fontweight='bold', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"


def calculate_simple_effects(df_long, dv_col, within_col, between_col, subject_col, alpha=0.05):
    """Calculate simple effects for interaction follow-up"""
    simple_effects = []
    
    try:
        # Simple effect of Time within each Group
        for group in sorted(df_long[between_col].unique()):
            group_data = df_long[df_long[between_col] == group]
            
            # Run one-way RM ANOVA for this group
            aov = pg.rm_anova(
                data=group_data,
                dv=dv_col,
                within=within_col,
                subject=subject_col,
                detailed=True
            )
            
            if len(aov) > 0:
                row = aov.iloc[0]
                simple_effects.append({
                    'effect_type': 'Time within Group',
                    'level': str(group),
                    'F': safe_float(row.get('F', None)),
                    'df1': int(row.get('DF1', row.get('ddof1', 0))),
                    'df2': int(row.get('DF2', row.get('ddof2', 0))),
                    'p_value': safe_float(row.get('p-unc', None)),
                    'partial_eta_sq': safe_float(row.get('np2', None)),
                    'significant': bool(row.get('p-unc', 1) < alpha)
                })
        
        # Simple effect of Group at each Time point
        for time in sorted(df_long[within_col].unique()):
            time_data = df_long[df_long[within_col] == time]
            
            # Run independent t-test or one-way ANOVA
            groups = time_data[between_col].unique()
            if len(groups) == 2:
                # t-test for 2 groups
                g1_data = time_data[time_data[between_col] == groups[0]][dv_col]
                g2_data = time_data[time_data[between_col] == groups[1]][dv_col]
                
                t_stat, p_val = stats.ttest_ind(g1_data, g2_data)
                df_val = len(g1_data) + len(g2_data) - 2
                
                # Calculate eta-squared
                eta_sq = (t_stat ** 2) / ((t_stat ** 2) + df_val)
                
                simple_effects.append({
                    'effect_type': 'Group at Time',
                    'level': str(time),
                    'F': safe_float(t_stat ** 2),
                    'df1': 1,
                    'df2': int(df_val),
                    'p_value': safe_float(p_val),
                    'partial_eta_sq': safe_float(eta_sq),
                    'significant': bool(p_val < alpha)
                })
            else:
                # One-way ANOVA for >2 groups
                group_data_list = [time_data[time_data[between_col] == g][dv_col].values for g in groups]
                f_stat, p_val = stats.f_oneway(*group_data_list)
                
                # Calculate eta-squared
                ss_between = sum([len(g) * (np.mean(g) - time_data[dv_col].mean())**2 for g in group_data_list])
                ss_total = np.sum((time_data[dv_col] - time_data[dv_col].mean())**2)
                eta_sq = ss_between / ss_total if ss_total > 0 else 0
                
                simple_effects.append({
                    'effect_type': 'Group at Time',
                    'level': str(time),
                    'F': safe_float(f_stat),
                    'df1': len(groups) - 1,
                    'df2': len(time_data) - len(groups),
                    'p_value': safe_float(p_val),
                    'partial_eta_sq': safe_float(eta_sq),
                    'significant': bool(p_val < alpha)
                })
    
    except Exception as e:
        print(f"Simple effects calculation error: {e}")
    
    return simple_effects


def calculate_descriptives_by_cell(df_long, dv_col, within_col, between_col):
    """Calculate descriptive statistics for each Group × Time cell"""
    descriptives_by_cell = []
    
    for group in sorted(df_long[between_col].unique()):
        for time in sorted(df_long[within_col].unique()):
            cell_data = df_long[(df_long[between_col] == group) & (df_long[within_col] == time)][dv_col]
            
            if len(cell_data) > 0:
                descriptives_by_cell.append({
                    'group': str(group),
                    'time': str(time),
                    'n': int(len(cell_data)),
                    'mean': safe_float(cell_data.mean()),
                    'std': safe_float(cell_data.std()),
                    'se': safe_float(cell_data.sem()),
                    'min': safe_float(cell_data.min()),
                    'max': safe_float(cell_data.max()),
                    'ci_lower': safe_float(cell_data.mean() - 1.96 * cell_data.sem()),
                    'ci_upper': safe_float(cell_data.mean() + 1.96 * cell_data.sem())
                })
    
    return descriptives_by_cell


def generate_interpretation(results, design_type='mixed'):
    """Generate interpretation text"""
    parts = []
    
    parts.append("**Repeated Measures ANOVA Results**\n")
    
    if design_type == 'mixed':
        parts.append(f"Design: Mixed (Between × Within)\n")
    else:
        parts.append(f"Design: Fully Within-Subjects\n")
    
    # Sphericity
    sphericity = results.get('sphericity', {})
    if sphericity.get('p_value') is not None:
        if sphericity.get('sphericity_met'):
            parts.append(f"→ Sphericity assumption met (Mauchly's W = {sphericity['W']:.3f}, p = {sphericity['p_value']:.3f})")
        else:
            parts.append(f"→ ⚠️ Sphericity violated (Mauchly's W = {sphericity['W']:.3f}, p = {sphericity['p_value']:.3f})")
            parts.append(f"  Greenhouse-Geisser ε = {sphericity['epsilon_gg']:.3f}, Huynh-Feldt ε = {sphericity['epsilon_hf']:.3f}")
            parts.append(f"  Corrected p-values are provided in the ANOVA table.")
    
    parts.append("")
    
    # Main effects and interactions
    anova_table = results.get('anova_table', [])
    for row in anova_table:
        source = row['Source']
        f_val = row.get('F')
        p_val = row.get('p_value')
        eta_sq = row.get('partial_eta_sq')
        
        if f_val is not None and p_val is not None:
            sig = "**Significant**" if p_val < 0.05 else "Not significant"
            eta_str = f"{eta_sq:.3f}" if eta_sq else "N/A"
            parts.append(f"→ {source}: F = {f_val:.2f}, p = {p_val:.4f}, η²p = {eta_str} — {sig}")
    
    parts.append("")
    parts.append("**Recommendations**")
    parts.append("→ Examine pairwise comparisons for significant effects")
    parts.append("→ Check profile plots for visual interpretation")
    parts.append("→ Consider effect sizes (η²p) for practical significance")
    
    return "\n".join(parts)


# ============================================
# MIXED DESIGN (BETWEEN × WITHIN)
# ============================================

@router.post("/two-way-rm-anova")
def mixed_design_rm_anova(req: TwoWayRMAnovaRequest):
    """
    Mixed design repeated measures ANOVA
    One between-subjects factor × One within-subjects factor
    """
    
    if not STATS_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="Required packages not installed. Run: pip install scipy pingouin"
        )
    
    try:
        # Prepare data
        df = pd.DataFrame(req.data)
        subject_col = req.subject_col
        between_col = req.between_factor_col
        within_cols = req.within_measure_cols
        alpha = req.alpha
        post_hoc = req.post_hoc_method
        correction = req.sphericity_correction
        
        # Validate columns
        missing = [col for col in [subject_col, between_col] + within_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Columns not found: {missing}")
        
        if len(within_cols) < 2:
            raise ValueError("Need at least 2 within-subjects measures")
        
        # Clean data
        all_cols = [subject_col, between_col] + within_cols
        df_clean = df[all_cols].dropna()
        n_dropped = len(df) - len(df_clean)
        
        if len(df_clean) < 5:
            raise ValueError(f"Insufficient data: {len(df_clean)} subjects")
        
        # Convert to long format
        df_long = pd.melt(
            df_clean,
            id_vars=[subject_col, between_col],
            value_vars=within_cols,
            var_name='Time',
            value_name='Value'
        )
        
        # Run repeated measures ANOVA using pingouin
        # Check if we have a true mixed design or just repeated measures
        if df_clean[between_col].nunique() > 1:
            # Mixed design: between-subjects factor exists
            aov = pg.mixed_anova(
                data=df_long,
                dv='Value',
                within='Time',
                between=between_col,
                subject=subject_col,
                correction='auto' if correction == 'greenhouse-geisser' else False
            )
        else:
            # Pure repeated measures (no between factor)
            aov = pg.rm_anova(
                data=df_long,
                dv='Value',
                within='Time',
                subject=subject_col,
                correction='auto' if correction == 'greenhouse-geisser' else False,
                detailed=True
            )
        
        # Format ANOVA table
        anova_results = []
        for _, row in aov.iterrows():
            source = row['Source']
            
            anova_results.append({
                'Source': source,
                'SS': safe_float(row.get('SS', 0)),
                'df': int(row.get('DF', 0)),
                'MS': safe_float(row.get('MS', 0)),
                'F': safe_float(row.get('F', None)),
                'p_value': safe_float(row.get('p-unc', None)),
                'p_GG': safe_float(row.get('p-GG', None)) if 'p-GG' in row else None,
                'p_HF': safe_float(row.get('p-HF', None)) if 'p-HF' in row else None,
                'partial_eta_sq': safe_float(row.get('np2', None)),
                'epsilon': safe_float(row.get('eps', None)) if 'eps' in row else None
            })
        
        # Sphericity test
        within_data = df_clean[within_cols].values
        sphericity = mauchly_sphericity_test(within_data, len(within_cols))
        
        # Descriptive statistics
        descriptives = {}
        for col in within_cols:
            for group in df_clean[between_col].unique():
                group_data = df_clean[df_clean[between_col] == group][col]
                key = f"{group}_{col}"
                descriptives[key] = {
                    'n': int(len(group_data)),
                    'mean': safe_float(group_data.mean()),
                    'std': safe_float(group_data.std()),
                    'se': safe_float(group_data.sem()),
                    'ci_lower': safe_float(group_data.mean() - 1.96 * group_data.sem()),
                    'ci_upper': safe_float(group_data.mean() + 1.96 * group_data.sem())
                }
        
        # Pairwise comparisons (within-subjects)
        pairwise_within = perform_pairwise_comparisons(
            df_long, 'Value', 'Time', subject_col, post_hoc
        )
        
        # Pairwise comparisons (between-subjects) for each time point
        pairwise_between = []
        for time in within_cols:
            time_data = df_long[df_long['Time'] == time]
            
            groups = time_data[between_col].unique()
            for i, g1 in enumerate(groups):
                for g2 in groups[i+1:]:
                    d1 = time_data[time_data[between_col] == g1]['Value']
                    d2 = time_data[time_data[between_col] == g2]['Value']
                    
                    if len(d1) > 1 and len(d2) > 1:
                        t_stat, p_val = stats.ttest_ind(d1, d2)
                        
                        # Bonferroni correction
                        n_comparisons = len(groups) * (len(groups) - 1) / 2
                        p_adj = min(1.0, p_val * n_comparisons)
                        
                        pairwise_between.append({
                            'time': time,
                            'group1': str(g1),
                            'group2': str(g2),
                            'mean_diff': safe_float(d1.mean() - d2.mean()),
                            't_stat': safe_float(t_stat),
                            'p_unc': safe_float(p_val),
                            'p_adj': safe_float(p_adj),
                            'significant': bool(p_adj < alpha)
                        })
        
        # Create visualizations
        plot = create_profile_plot(df_long, 'Value', 'Time', between_col, subject_col)
        boxplot = create_boxplot(df_long, 'Value', 'Time', between_col)
        mean_plot = create_mean_comparison_plot(df_long, 'Value', 'Time', between_col)
        
        # Calculate simple effects (follow-up for interaction)
        simple_effects = calculate_simple_effects(df_long, 'Value', 'Time', between_col, subject_col, alpha)
        
        # Calculate descriptives by cell (Group × Time)
        descriptives_by_cell = calculate_descriptives_by_cell(df_long, 'Value', 'Time', between_col)
        
        # Interpretation
        results_dict = {
            'anova_table': anova_results,
            'sphericity': sphericity
        }
        interpretation = generate_interpretation(results_dict, 'mixed')
        
        # Compile response
        response = {
            'results': {
                'design_type': 'mixed',
                'anova_table': anova_results,
                'sphericity': sphericity,
                'descriptives': descriptives,
                'descriptives_by_cell': descriptives_by_cell,
                'simple_effects': simple_effects,
                'pairwise_within': pairwise_within,
                'pairwise_between': pairwise_between,
                'interpretation': interpretation,
                'settings': {
                    'alpha': alpha,
                    'post_hoc_method': post_hoc,
                    'sphericity_correction': correction
                },
                'n_subjects': len(df_clean),
                'n_dropped': n_dropped,
                'n_groups': df_clean[between_col].nunique(),
                'n_time_points': len(within_cols)
            },
            'plot': plot,
            'boxplot': boxplot,
            'mean_plot': mean_plot
        }
        
        return _to_native(response)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================
# FULLY WITHIN-SUBJECTS (FACTOR1 × FACTOR2)
# ============================================

@router.post("/two-way-within-anova")
def fully_within_rm_anova(req: TwoWayWithinAnovaRequest):
    """
    Fully within-subjects two-way ANOVA
    Both factors are within-subjects
    """
    
    if not STATS_AVAILABLE:
        raise HTTPException(
            status_code=500,
            detail="Required packages not installed. Run: pip install scipy pingouin"
        )
    
    try:
        # Prepare data
        df = pd.DataFrame(req.data)
        subject_col = req.subject_col
        factor1_cols = req.within_factor1_cols
        factor2_cols = req.within_factor2_cols
        factor1_name = req.factor1_name
        factor2_name = req.factor2_name
        alpha = req.alpha
        post_hoc = req.post_hoc_method
        correction = req.sphericity_correction
        
        # Validate
        all_measure_cols = factor1_cols + factor2_cols
        missing = [col for col in [subject_col] + all_measure_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Columns not found: {missing}")
        
        if len(factor1_cols) < 2 or len(factor2_cols) < 2:
            raise ValueError("Each factor needs at least 2 levels")
        
        # Clean data
        df_clean = df[[subject_col] + all_measure_cols].dropna()
        n_dropped = len(df) - len(df_clean)
        
        if len(df_clean) < 5:
            raise ValueError(f"Insufficient data: {len(df_clean)} subjects")
        
        # Determine factor structure (this is simplified - real implementation needs proper mapping)
        # For now, assume columns are named systematically
        # E.g., Drug1_Time1, Drug1_Time2, Drug2_Time1, Drug2_Time2
        
        # Create long format
        data_for_anova = []
        for idx, row in df_clean.iterrows():
            subject_id = row[subject_col]
            
            # Parse factor levels from column names
            for col in all_measure_cols:
                # Try to extract factor levels from column name
                # This is a simplified approach
                parts = col.split('_')
                if len(parts) >= 2:
                    level1 = parts[0]
                    level2 = parts[1] if len(parts) > 1 else parts[0]
                else:
                    level1 = col
                    level2 = col
                
                data_for_anova.append({
                    subject_col: subject_id,
                    factor1_name: level1,
                    factor2_name: level2,
                    'Value': row[col]
                })
        
        df_long = pd.DataFrame(data_for_anova)
        
        # Run two-way repeated measures ANOVA
        try:
            aov = pg.rm_anova2(
                data=df_long,
                dv='Value',
                within=[factor1_name, factor2_name],
                subject=subject_col,
                correction=correction != 'none'
            )
            
            # Format results
            anova_results = []
            for _, row in aov.iterrows():
                anova_results.append({
                    'Source': row['Source'],
                    'SS': safe_float(row.get('SS', 0)),
                    'df': int(row.get('DF', 0)),
                    'MS': safe_float(row.get('MS', 0)),
                    'F': safe_float(row.get('F', None)),
                    'p_value': safe_float(row.get('p-unc', None)),
                    'p_GG': safe_float(row.get('p-GG', None)) if 'p-GG' in row else None,
                    'partial_eta_sq': safe_float(row.get('np2', None)),
                    'epsilon': safe_float(row.get('eps', None)) if 'eps' in row else None
                })
        except Exception as e:
            # Fallback: use separate one-way ANOVAs
            anova_results = [{
                'Source': 'Error',
                'error': f"Two-way within ANOVA failed: {str(e)}"
            }]
        
        # Sphericity tests
        sphericity_f1 = mauchly_sphericity_test(
            df_clean[factor1_cols].values, len(factor1_cols)
        )
        sphericity_f2 = mauchly_sphericity_test(
            df_clean[factor2_cols].values, len(factor2_cols)
        )
        
        # Descriptive statistics
        descriptives = {}
        for col in all_measure_cols:
            data = df_clean[col]
            descriptives[col] = {
                'n': int(len(data)),
                'mean': safe_float(data.mean()),
                'std': safe_float(data.std()),
                'se': safe_float(data.sem()),
                'ci_lower': safe_float(data.mean() - 1.96 * data.sem()),
                'ci_upper': safe_float(data.mean() + 1.96 * data.sem())
            }
        
        # Pairwise comparisons for Factor 1
        pairwise_f1 = perform_pairwise_comparisons(
            df_long, 'Value', factor1_name, subject_col, post_hoc
        )
        
        # Pairwise comparisons for Factor 2
        pairwise_f2 = perform_pairwise_comparisons(
            df_long, 'Value', factor2_name, subject_col, post_hoc
        )
        
        # Create visualization
        plot = create_profile_plot(df_long, 'Value', factor1_name, factor2_name, subject_col)
        
        # Interpretation
        results_dict = {
            'anova_table': anova_results,
            'sphericity': sphericity_f1
        }
        interpretation = generate_interpretation(results_dict, 'fully_within')
        
        # Response
        response = {
            'results': {
                'design_type': 'fully_within',
                'anova_table': anova_results,
                'sphericity': {
                    factor1_name: sphericity_f1,
                    factor2_name: sphericity_f2
                },
                'descriptives': descriptives,
                'pairwise': {
                    factor1_name: pairwise_f1,
                    factor2_name: pairwise_f2
                },
                'interpretation': interpretation,
                'settings': {
                    'alpha': alpha,
                    'post_hoc_method': post_hoc,
                    'sphericity_correction': correction
                },
                'n_subjects': len(df_clean),
                'n_dropped': n_dropped,
                'factor1_levels': len(factor1_cols),
                'factor2_levels': len(factor2_cols)
            },
            'plot': plot
        }
        
        return _to_native(response)
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
