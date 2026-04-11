from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, List, Dict, Optional
import numpy as np
import pandas as pd
import io, base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

try:
    from factor_analyzer import FactorAnalyzer
    from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
    FA_AVAILABLE = True
except ImportError:
    FA_AVAILABLE = False

router = APIRouter()

class ConstructValidityRequest(BaseModel):
    data: list[dict[str, Any]] = Field(...)
    factor_definitions: Dict[str, List[str]] = Field(...)
    rotation: Optional[str] = 'promax'

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

def calculate_cronbach_alpha(X):
    X = np.array(X)
    n_items = X.shape[1]
    if n_items < 2: return None
    item_variances = np.var(X, axis=0, ddof=1)
    total_variance = np.var(np.sum(X, axis=1), ddof=1)
    if total_variance == 0: return None
    alpha = (n_items / (n_items - 1)) * (1 - np.sum(item_variances) / total_variance)
    return float(alpha)

def calculate_item_total_correlation(X):
    X = np.array(X)
    n_items = X.shape[1]
    correlations = []
    for i in range(n_items):
        total_without = np.sum(np.delete(X, i, axis=1), axis=1)
        corr = np.corrcoef(X[:, i], total_without)[0, 1]
        correlations.append(safe_float(corr))
    return correlations

def calculate_alpha_if_deleted(X):
    X = np.array(X)
    n_items = X.shape[1]
    alphas = []
    for i in range(n_items):
        X_without = np.delete(X, i, axis=1)
        if X_without.shape[1] >= 2:
            alphas.append(calculate_cronbach_alpha(X_without))
        else:
            alphas.append(None)
    return alphas

def calculate_composite_reliability(loadings):
    loadings = np.array([l for l in loadings if l is not None and not np.isnan(l)])
    if len(loadings) == 0: return None
    sum_loadings = np.sum(loadings)
    sum_error_var = np.sum(1 - loadings ** 2)
    cr = (sum_loadings ** 2) / ((sum_loadings ** 2) + sum_error_var)
    return float(cr)

def calculate_ave(loadings):
    loadings = np.array([l for l in loadings if l is not None and not np.isnan(l)])
    if len(loadings) == 0: return None
    return float(np.mean(loadings ** 2))

def fornell_larcker_analysis(ave_values, correlation_matrix, factor_names):
    n_factors = len(factor_names)
    sqrt_ave = [np.sqrt(a) if a else 0 for a in ave_values]
    results = []
    valid = True
    for i in range(n_factors):
        for j in range(i + 1, n_factors):
            corr = correlation_matrix[i, j]
            is_valid = sqrt_ave[i] > abs(corr) and sqrt_ave[j] > abs(corr)
            if not is_valid: valid = False
            results.append({
                'factor_1': factor_names[i], 'factor_2': factor_names[j],
                'correlation': safe_float(corr),
                'sqrt_ave_1': safe_float(sqrt_ave[i]), 'sqrt_ave_2': safe_float(sqrt_ave[j]),
                'valid': is_valid
            })
    return results, valid

def htmt_analysis(data_groups, factor_names):
    n_factors = len(factor_names)
    results = []
    for i in range(n_factors):
        for j in range(i + 1, n_factors):
            items_i = data_groups[factor_names[i]]
            items_j = data_groups[factor_names[j]]
            
            heterotrait_corrs = []
            for col_i in items_i.columns:
                for col_j in items_j.columns:
                    corr = np.corrcoef(items_i[col_i], items_j[col_j])[0, 1]
                    if not np.isnan(corr): heterotrait_corrs.append(abs(corr))
            
            monotrait_i, monotrait_j = [], []
            for k, col1 in enumerate(items_i.columns):
                for col2 in list(items_i.columns)[k+1:]:
                    corr = np.corrcoef(items_i[col1], items_i[col2])[0, 1]
                    if not np.isnan(corr): monotrait_i.append(abs(corr))
            for k, col1 in enumerate(items_j.columns):
                for col2 in list(items_j.columns)[k+1:]:
                    corr = np.corrcoef(items_j[col1], items_j[col2])[0, 1]
                    if not np.isnan(corr): monotrait_j.append(abs(corr))
            
            if heterotrait_corrs and monotrait_i and monotrait_j:
                htmt = np.mean(heterotrait_corrs) / np.sqrt(np.mean(monotrait_i) * np.mean(monotrait_j))
                results.append({
                    'factor_1': factor_names[i], 'factor_2': factor_names[j],
                    'htmt': safe_float(htmt), 'valid_085': htmt < 0.85, 'valid_090': htmt < 0.90
                })
    return results

def create_plots(factor_results, loadings_matrix, all_items, factor_names, fornell_larcker_results, ave_values, htmt_results):
    n_factors = len(factor_names)
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Reliability Summary
    alphas = [f['cronbach_alpha'] or 0 for f in factor_results]
    crs = [f['composite_reliability'] or 0 for f in factor_results]
    aves = [f['ave'] or 0 for f in factor_results]
    x = np.arange(len(factor_names))
    width = 0.25
    
    axes[0, 0].bar(x - width, alphas, width, label="Cronbach's α", color='#5B9BD5')
    axes[0, 0].bar(x, crs, width, label='CR', color='#70AD47')
    axes[0, 0].bar(x + width, aves, width, label='AVE', color='#FFC000')
    axes[0, 0].axhline(y=0.7, color='red', linestyle='--', lw=1.5, label='α/CR Threshold')
    axes[0, 0].axhline(y=0.5, color='orange', linestyle='--', lw=1.5, label='AVE Threshold')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(factor_names, rotation=45, ha='right')
    axes[0, 0].set_ylabel('Value')
    axes[0, 0].set_title('Reliability & Validity Metrics', fontweight='bold')
    axes[0, 0].legend(loc='lower right', fontsize=8)
    axes[0, 0].set_ylim(0, 1.1)
    
    # 2. Factor Loadings Heatmap
    sns.heatmap(loadings_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                xticklabels=[f'F{i+1}' for i in range(n_factors)], yticklabels=all_items,
                vmin=-1, vmax=1, ax=axes[0, 1], cbar_kws={'shrink': 0.8})
    axes[0, 1].set_title('Factor Loadings', fontweight='bold')
    
    # 3. Fornell-Larcker
    if n_factors > 1 and fornell_larcker_results:
        matrix = np.zeros((n_factors, n_factors))
        for i in range(n_factors):
            matrix[i, i] = np.sqrt(ave_values[i]) if ave_values[i] else 0
        for r in fornell_larcker_results:
            i, j = factor_names.index(r['factor_1']), factor_names.index(r['factor_2'])
            matrix[i, j] = matrix[j, i] = r['correlation']
        sns.heatmap(matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                    xticklabels=factor_names, yticklabels=factor_names, ax=axes[1, 0], square=True)
        axes[1, 0].set_title('Fornell-Larcker\n(Diag=√AVE, Off-diag=r)', fontweight='bold')
    else:
        axes[1, 0].text(0.5, 0.5, 'Need >1 factor', ha='center', va='center')
        axes[1, 0].set_title('Fornell-Larcker', fontweight='bold')
    
    # 4. HTMT
    if n_factors > 1 and htmt_results:
        htmt_matrix = np.zeros((n_factors, n_factors))
        for r in htmt_results:
            i, j = factor_names.index(r['factor_1']), factor_names.index(r['factor_2'])
            htmt_matrix[i, j] = htmt_matrix[j, i] = r['htmt']
        mask = np.eye(n_factors, dtype=bool)
        sns.heatmap(htmt_matrix, annot=True, fmt='.3f', cmap='RdYlGn_r', mask=mask,
                    xticklabels=factor_names, yticklabels=factor_names, ax=axes[1, 1],
                    vmin=0, vmax=1, square=True)
        axes[1, 1].set_title('HTMT (<0.85=Good)', fontweight='bold')
    else:
        axes[1, 1].text(0.5, 0.5, 'Need >1 factor', ha='center', va='center')
        axes[1, 1].set_title('HTMT', fontweight='bold')
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"

def generate_interpretation(factor_results, fornell_larcker_valid, htmt_results):
    parts = []
    parts.append("**Overall Analysis**")
    
    all_alpha_valid = all(f.get('cronbach_alpha', 0) and f['cronbach_alpha'] >= 0.7 for f in factor_results)
    all_cr_valid = all(f.get('composite_reliability', 0) and f['composite_reliability'] >= 0.7 for f in factor_results)
    all_ave_valid = all(f.get('ave', 0) and f['ave'] >= 0.5 for f in factor_results)
    
    parts.append(f"→ Analyzed {len(factor_results)} constructs for reliability and validity.")
    parts.append(f"→ Internal consistency (α ≥ 0.7): {'✓ All passed' if all_alpha_valid else '⚠ Some issues'}")
    parts.append(f"→ Composite reliability (CR ≥ 0.7): {'✓ All passed' if all_cr_valid else '⚠ Some issues'}")
    parts.append(f"→ Convergent validity (AVE ≥ 0.5): {'✓ All passed' if all_ave_valid else '⚠ Some issues'}")
    
    parts.append("")
    parts.append("**Key Insights**")
    
    for f in factor_results:
        alpha = f.get('cronbach_alpha', 0) or 0
        cr = f.get('composite_reliability', 0) or 0
        ave = f.get('ave', 0) or 0
        status = '✓' if alpha >= 0.7 and cr >= 0.7 and ave >= 0.5 else '⚠'
        parts.append(f"→ {f['name']}: α={alpha:.3f}, CR={cr:.3f}, AVE={ave:.3f} {status}")
    
    if fornell_larcker_valid:
        parts.append("→ Fornell-Larcker criterion satisfied: √AVE > inter-construct correlations.")
    else:
        parts.append("→ ⚠ Fornell-Larcker criterion violated for some construct pairs.")
    
    if htmt_results:
        htmt_issues = [r for r in htmt_results if not r['valid_085']]
        if not htmt_issues:
            parts.append("→ HTMT criterion satisfied: All values < 0.85.")
        else:
            parts.append(f"→ ⚠ HTMT issues for {len(htmt_issues)} pair(s).")
    
    parts.append("")
    parts.append("**Recommendations**")
    
    low_alpha = [f['name'] for f in factor_results if not f.get('cronbach_alpha') or f['cronbach_alpha'] < 0.7]
    low_ave = [f['name'] for f in factor_results if not f.get('ave') or f['ave'] < 0.5]
    
    if low_alpha:
        parts.append(f"→ Improve internal consistency for: {', '.join(low_alpha)}. Review item-total correlations.")
    if low_ave:
        parts.append(f"→ Improve convergent validity for: {', '.join(low_ave)}. Consider removing items with loadings < 0.5.")
    if not fornell_larcker_valid:
        parts.append("→ Address discriminant validity issues by revising overlapping items.")
    
    parts.append("→ Report Cronbach's α, CR, AVE, and discriminant validity evidence in publications.")
    parts.append("→ Consider CFA for confirmatory validation of the factor structure.")
    
    return "\n".join(parts)

@router.post("/construct-validity")
def construct_validity(req: ConstructValidityRequest):
    if not FA_AVAILABLE:
        raise HTTPException(status_code=500, detail="factor_analyzer not installed")
    
    try:
        df = pd.DataFrame(req.data)
        factor_definitions = req.factor_definitions
        rotation = req.rotation if req.rotation != 'none' else None
        
        all_items = []
        for items in factor_definitions.values():
            for item in items:
                if item not in df.columns:
                    raise ValueError(f"Item '{item}' not found")
                all_items.append(item)
        all_items = list(dict.fromkeys(all_items))
        
        df_items = df[all_items].dropna()
        if len(df_items) < 30:
            raise ValueError(f"Need at least 30 observations, got {len(df_items)}")
        
        n_factors = len(factor_definitions)
        factor_names = list(factor_definitions.keys())
        
        # KMO & Bartlett
        try:
            _, kmo_model = calculate_kmo(df_items)
            chi_sq, p_val = calculate_bartlett_sphericity(df_items)
        except:
            kmo_model, chi_sq, p_val = 0.5, 0, 1
        
        # EFA
        fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation, method='ml')
        fa.fit(df_items.values)
        loadings_matrix = fa.loadings_
        
        # Factor analysis
        factor_results = []
        data_groups = {}
        
        for i, (factor_name, items) in enumerate(factor_definitions.items()):
            factor_data = df_items[items].values
            data_groups[factor_name] = df_items[items]
            
            item_indices = [all_items.index(item) for item in items]
            factor_loadings = [loadings_matrix[idx, i] for idx in item_indices]
            
            alpha = calculate_cronbach_alpha(factor_data)
            item_total = calculate_item_total_correlation(factor_data)
            alpha_if_del = calculate_alpha_if_deleted(factor_data)
            cr = calculate_composite_reliability(factor_loadings)
            ave = calculate_ave(factor_loadings)
            
            factor_results.append({
                'name': factor_name, 'items': items, 'n_items': len(items),
                'cronbach_alpha': safe_float(alpha) if alpha else None,
                'composite_reliability': safe_float(cr) if cr else None,
                'ave': safe_float(ave) if ave else None,
                'sqrt_ave': safe_float(np.sqrt(ave)) if ave else None,
                'factor_loadings': {item: safe_float(l) for item, l in zip(items, factor_loadings)},
                'item_total_correlations': {item: c for item, c in zip(items, item_total)},
                'alpha_if_deleted': {item: safe_float(a) if a else None for item, a in zip(items, alpha_if_del)},
                'valid_alpha': (alpha or 0) >= 0.7,
                'valid_cr': (cr or 0) >= 0.7,
                'valid_ave': (ave or 0) >= 0.5
            })
        
        # Factor correlations
        factor_scores = [df_items[items].mean(axis=1).values for items in factor_definitions.values()]
        factor_correlation = np.corrcoef(factor_scores)
        
        # Validity
        ave_values = [f['ave'] or 0 for f in factor_results]
        fornell_larcker_results, fornell_larcker_valid = fornell_larcker_analysis(ave_values, factor_correlation, factor_names)
        htmt_results = htmt_analysis(data_groups, factor_names) if n_factors > 1 else []
        
        # Plot
        plot = create_plots(factor_results, loadings_matrix, all_items, factor_names, 
                           fornell_larcker_results, ave_values, htmt_results)
        
        # Interpretation
        interpretation = generate_interpretation(factor_results, fornell_larcker_valid, htmt_results)
        
        response = {
            'n_observations': len(df_items),
            'n_factors': n_factors,
            'n_items': len(all_items),
            'kmo': safe_float(kmo_model),
            'bartlett_chi_square': safe_float(chi_sq),
            'bartlett_p_value': safe_float(p_val),
            'rotation': rotation or 'none',
            'factor_results': factor_results,
            'factor_correlation': [[safe_float(c) for c in row] for row in factor_correlation],
            'fornell_larcker': fornell_larcker_results,
            'fornell_larcker_valid': fornell_larcker_valid,
            'htmt': htmt_results,
            'overall_validity': {
                'internal_consistency': all(f['valid_alpha'] for f in factor_results),
                'composite_reliability': all(f['valid_cr'] for f in factor_results),
                'convergent_validity': all(f['valid_ave'] for f in factor_results),
                'discriminant_validity_fl': fornell_larcker_valid,
                'discriminant_validity_htmt': all(r['valid_085'] for r in htmt_results) if htmt_results else None
            },
            'interpretation': interpretation,
            'plot': plot
        }
        
        return _to_native(response)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
