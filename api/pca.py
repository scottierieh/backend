from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import io, base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

try:
    from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
    FACTOR_ANALYZER_AVAILABLE = True
except ImportError:
    FACTOR_ANALYZER_AVAILABLE = False

router = APIRouter()

class PCARequest(BaseModel):
    data: list[dict[str, Any]] = Field(...)
    variables: List[str] = Field(...)
    nComponents: Optional[int] = None

def _to_native(obj):
    if isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj): return None
        return float(obj)
    elif isinstance(obj, np.ndarray): return [_to_native(x) for x in obj.tolist()]
    elif isinstance(obj, np.bool_): return bool(obj)
    elif isinstance(obj, dict): return {str(k): _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)): return [_to_native(x) for x in obj]
    return obj

def safe_float(val, default=0.0):
    try:
        if val is None or pd.isna(val) or np.isinf(val): return default
        return float(val)
    except: return default

def kmo_label(kmo):
    if kmo is None: return ''
    if kmo >= 0.90: return 'Marvelous'
    if kmo >= 0.80: return 'Meritorious'
    if kmo >= 0.70: return 'Middling'
    if kmo >= 0.60: return 'Mediocre'
    if kmo >= 0.50: return 'Miserable'
    return 'Unacceptable'

def compute_adequacy(scaled_df):
    """KMO and Bartlett's test of sphericity."""
    adequacy = {
        'kmo': None,
        'kmo_label': '',
        'bartlett_chi2': None,
        'bartlett_df': None,
        'bartlett_p': None,
        'bartlett_significant': False,
    }
    if not FACTOR_ANALYZER_AVAILABLE:
        return adequacy
    try:
        kmo_all, kmo_model = calculate_kmo(scaled_df)
        adequacy['kmo'] = float(kmo_model) if not np.isnan(kmo_model) else None
        adequacy['kmo_label'] = kmo_label(adequacy['kmo'])
    except Exception:
        pass
    try:
        chi2, p = calculate_bartlett_sphericity(scaled_df)
        p_val = float(p) if not np.isnan(p) else None
        n_vars = scaled_df.shape[1]
        adequacy['bartlett_chi2'] = float(chi2) if not np.isnan(chi2) else None
        adequacy['bartlett_df'] = int(n_vars * (n_vars - 1) / 2)
        adequacy['bartlett_p'] = p_val
        adequacy['bartlett_significant'] = bool(p_val < 0.05) if p_val is not None else False
    except Exception:
        pass
    return adequacy

def compute_warnings(clean_data, variables, n_components_used, adequacy):
    warnings = []
    n_obs = len(clean_data)
    ratio = n_obs / len(variables)

    if n_obs < 50:
        warnings.append(f"Very small sample (N = {n_obs}). PCA results may be unstable. N ≥ 100 recommended.")
    elif n_obs < 100:
        warnings.append(f"Small sample (N = {n_obs}). Consider collecting more data for stable PCA.")

    if ratio < 5:
        warnings.append(f"Subject-to-variable ratio is {ratio:.1f}:1 (minimum 5:1 recommended).")

    kmo = adequacy.get('kmo')
    if kmo is not None and kmo < 0.5:
        warnings.append(f"KMO = {kmo:.3f} is unacceptable. Variables may not be suitable for PCA.")
    elif kmo is not None and kmo < 0.6:
        warnings.append(f"KMO = {kmo:.3f} is miserable. PCA results should be interpreted with caution.")

    if adequacy.get('bartlett_significant') is False and adequacy.get('bartlett_p') is not None:
        warnings.append("Bartlett's test is not significant — variables may not be sufficiently correlated for PCA.")

    return warnings

def generate_interpretation(results, n_obs, variables):
    eigenvalues = results['eigenvalues']
    loadings = results['loadings']
    cumulative_variance = results['cumulative_variance_ratio']
    explained_variance = results['explained_variance_ratio']
    
    n_factors_kaiser = sum(1 for ev in eigenvalues if ev > 1)
    if n_factors_kaiser == 0 and len(eigenvalues) > 0:
        n_factors_kaiser = 1
    
    total_variance = cumulative_variance[n_factors_kaiser - 1] * 100 if n_factors_kaiser > 0 else 0
    
    parts = []
    parts.append("**Overall Analysis**")
    parts.append(f"→ A Principal Component Analysis (PCA) was conducted on {len(variables)} variables to explore the underlying structure of the data.")
    parts.append(f"→ Based on the Kaiser criterion (eigenvalues > 1), {n_factors_kaiser} component{'s were' if n_factors_kaiser != 1 else ' was'} extracted.")
    parts.append(f"→ Together, these components explain **{total_variance:.1f}%** of the total variance.")
    
    if total_variance >= 80:
        parts.append("→ This is an excellent level of variance explanation, indicating the components capture most of the data structure.")
    elif total_variance >= 70:
        parts.append("→ This is a good level of variance explanation, suitable for most analyses.")
    elif total_variance >= 60:
        parts.append("→ This is an acceptable level of variance explanation, though some information is lost.")
    else:
        parts.append("→ This is a relatively low level of variance explanation. Consider whether more components are needed.")

    parts.append("")
    parts.append("**Key Insights**")
    
    for i in range(min(n_factors_kaiser, 3)):
        var_pct = explained_variance[i] * 100
        parts.append(f"→ PC{i+1} explains {var_pct:.1f}% of variance (eigenvalue = {eigenvalues[i]:.2f}).")
        
        factor_loadings = np.array(loadings)[:, i]
        high_pos_indices = np.where(factor_loadings >= 0.5)[0]
        high_neg_indices = np.where(factor_loadings <= -0.5)[0]
        
        if len(high_pos_indices) > 0:
            high_pos_vars = [variables[j] for j in high_pos_indices]
            parts.append(f"  • Strong positive loadings: {', '.join(high_pos_vars)}")
        if len(high_neg_indices) > 0:
            high_neg_vars = [variables[j] for j in high_neg_indices]
            parts.append(f"  • Strong negative loadings: {', '.join(high_neg_vars)}")
    
    strong_loadings_count = np.sum(np.abs(loadings) >= 0.5)
    total_loadings = len(variables) * n_factors_kaiser
    parts.append(f"→ {strong_loadings_count} out of {total_loadings} loadings are strong (≥0.5).")

    parts.append("")
    parts.append("**Recommendations**")
    
    ratio = n_obs / len(variables)
    if ratio < 5:
        parts.append(f"→ Warning: The subject-to-variable ratio ({ratio:.1f}:1) is low. Consider collecting more data.")
    elif ratio < 10:
        parts.append(f"→ The subject-to-variable ratio ({ratio:.1f}:1) is adequate but could be improved.")
    else:
        parts.append(f"→ The subject-to-variable ratio ({ratio:.1f}:1) is good for reliable PCA results.")
    
    if total_variance < 70:
        parts.append("→ Consider extracting more components to capture at least 70% of variance.")
    
    if n_factors_kaiser == 1:
        parts.append("→ With only one component, the variables appear to measure a single underlying construct.")
    elif n_factors_kaiser == 2:
        parts.append("→ Two components suggest the data has two distinct underlying dimensions.")
    else:
        parts.append(f"→ Multiple components ({n_factors_kaiser}) indicate a complex, multidimensional structure.")
    
    parts.append("→ Use the component scores for subsequent analyses to reduce multicollinearity.")
    parts.append("→ Review the loadings plot to identify clusters of related variables.")

    return "\n".join(parts)

@router.post("/pca")
def pca_analysis(req: PCARequest):
    try:
        df = pd.DataFrame(req.data)
        variables = req.variables
        n_components = req.nComponents
        
        missing_cols = [col for col in variables if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found: {', '.join(missing_cols)}")
        
        clean_data = df[variables].apply(pd.to_numeric, errors='coerce').dropna()
        if len(clean_data) < 5:
            raise ValueError("Need at least 5 observations for PCA")
        
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(clean_data)
        scaled_df = pd.DataFrame(scaled_data, columns=variables)

        # KMO + Bartlett
        adequacy = compute_adequacy(scaled_df)

        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(scaled_data)

        # n_components_used: auto (kaiser) or user-specified
        kaiser_n = sum(1 for ev in pca.explained_variance_ if ev > 1)
        if kaiser_n == 0:
            kaiser_n = 1
        n_components_used = n_components if n_components else kaiser_n

        # parallel analysis (simplified: compare to mean eigenvalue of random)
        n_obs, n_vars = scaled_data.shape
        rng = np.random.default_rng(42)
        random_eigs = []
        for _ in range(100):
            rand = rng.standard_normal((n_obs, n_vars))
            rand_pca = PCA()
            rand_pca.fit(rand)
            random_eigs.append(rand_pca.explained_variance_)
        mean_random_eigs = np.mean(random_eigs, axis=0)
        n_components_parallel = int(np.sum(pca.explained_variance_ > mean_random_eigs))
        if n_components_parallel == 0:
            n_components_parallel = 1

        # elbow
        eigs = pca.explained_variance_
        if len(eigs) >= 3:
            diffs = np.diff(eigs)
            elbow_idx = int(np.argmin(diffs)) + 1
            n_components_elbow = max(1, elbow_idx)
        else:
            n_components_elbow = len(eigs)

        # cumvar (80% threshold)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        n_components_cumvar = int(np.searchsorted(cumvar, 0.80)) + 1
        n_components_cumvar = min(n_components_cumvar, len(eigs))

        # loadings (all components)
        loadings_full = pca.components_.T  # shape: (n_vars, n_components)

        # component scores
        score_cols = {f'PC{i+1}': principal_components[:, i].tolist() for i in range(pca.n_components_)}
        scores_df = pd.DataFrame(score_cols)
        records_omitted = len(df) > 500
        score_records = None if records_omitted else scores_df.round(4).to_dict(orient='records')
        score_descriptives = {
            col: {
                'mean': round(float(scores_df[col].mean()), 4),
                'std': round(float(scores_df[col].std()), 4),
                'min': round(float(scores_df[col].min()), 4),
                'max': round(float(scores_df[col].max()), 4),
            }
            for col in scores_df.columns
        }

        loading_threshold = 0.4

        results = {
            'eigenvalues': [safe_float(e) for e in pca.explained_variance_],
            'explained_variance_ratio': [safe_float(e) for e in pca.explained_variance_ratio_],
            'cumulative_variance_ratio': [safe_float(e) for e in np.cumsum(pca.explained_variance_ratio_)],
            'loadings': [[safe_float(loadings_full[i, j]) for j in range(pca.n_components_)] for i in range(len(variables))],
            'n_components_used': n_components_used,
            'n_components_kaiser': kaiser_n,
            'n_components_cumvar': n_components_cumvar,
            'n_components_elbow': n_components_elbow,
            'n_components_parallel': n_components_parallel,
            'auto_selected': n_components is None,
            'loading_threshold': loading_threshold,
            'variables': variables,
            'component_scores': {
                'records': score_records,
                'records_omitted': records_omitted,
                'descriptives': score_descriptives,
            },
        }

        results['interpretation'] = generate_interpretation(results, len(clean_data), variables)

        # warnings
        warnings = compute_warnings(clean_data, variables, n_components_used, adequacy)

        # Plot
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        line_color = '#C44E52'
        eigenvalues = results['eigenvalues']
        n_comps = len(eigenvalues)
        
        # 1. Scree Plot
        axes[0, 0].bar(range(1, n_comps + 1), eigenvalues, alpha=0.7, color='#5B9BD5', edgecolor='black')
        axes[0, 0].axhline(y=1, color=line_color, linestyle='--', lw=2, label='Kaiser Criterion')
        axes[0, 0].set_xlabel('Principal Component')
        axes[0, 0].set_ylabel('Eigenvalue')
        axes[0, 0].set_title('Scree Plot', fontweight='bold')
        axes[0, 0].legend()
        
        # 2. Cumulative Variance
        cumulative_var = np.array(results['cumulative_variance_ratio']) * 100
        axes[0, 1].plot(range(1, n_comps + 1), cumulative_var, 'o-', color='#5B9BD5', lw=2, ms=8)
        axes[0, 1].axhline(y=80, color=line_color, linestyle='--', lw=2, label='80% Threshold')
        axes[0, 1].set_xlabel('Number of Components')
        axes[0, 1].set_ylabel('Cumulative Variance (%)')
        axes[0, 1].set_title('Cumulative Variance Explained', fontweight='bold')
        axes[0, 1].set_ylim([0, 105])
        axes[0, 1].legend()
        
        # 3. Component Loadings
        loadings = np.array(results['loadings'])
        if loadings.shape[1] >= 2:
            axes[1, 0].scatter(loadings[:, 0], loadings[:, 1], alpha=0.8, s=100, color='#5B9BD5', edgecolors='black')
            axes[1, 0].axhline(0, color='grey', lw=1)
            axes[1, 0].axvline(0, color='grey', lw=1)
            axes[1, 0].set_xlabel(f'PC1 ({results["explained_variance_ratio"][0]*100:.1f}%)')
            axes[1, 0].set_ylabel(f'PC2 ({results["explained_variance_ratio"][1]*100:.1f}%)')
            axes[1, 0].set_title('Component Loadings (PC1 vs PC2)', fontweight='bold')
            for i, var in enumerate(variables):
                axes[1, 0].annotate(var, (loadings[i, 0], loadings[i, 1]), textcoords="offset points", xytext=(0, 5), ha='center', fontsize=9)
        else:
            axes[1, 0].text(0.5, 0.5, 'Only 1 component', ha='center', va='center')
            axes[1, 0].set_title('Component Loadings', fontweight='bold')
        
        # 4. Variance by Component
        var_explained = np.array(results['explained_variance_ratio']) * 100
        bars = axes[1, 1].bar(range(1, n_comps + 1), var_explained, alpha=0.7, color='#5B9BD5', edgecolor='black')
        axes[1, 1].axhline(y=var_explained.mean(), color=line_color, linestyle='--', lw=2, label=f'Mean: {var_explained.mean():.1f}%')
        axes[1, 1].set_xlabel('Principal Component')
        axes[1, 1].set_ylabel('Variance Explained (%)')
        axes[1, 1].set_title('Variance by Component', fontweight='bold')
        axes[1, 1].legend()
        
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        plot = f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"
        
        return _to_native({
            'results': results,
            'adequacy': adequacy,
            'warnings': warnings,
            'plot': plot,
        })
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
