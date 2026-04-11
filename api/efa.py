from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, List, Optional
import numpy as np
import pandas as pd
import io, base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn 1.6+ renamed force_all_finite → ensure_all_finite
# factor_analyzer 0.5.x uses the old name — patch the internal reference directly
try:
    import sklearn.utils.validation as _skval
    import inspect as _inspect
    if 'ensure_all_finite' in _inspect.signature(_skval.check_array).parameters:
        _orig_check = _skval.check_array
        def _patched_check_array(*args, **kwargs):
            if 'force_all_finite' in kwargs:
                kwargs['ensure_all_finite'] = kwargs.pop('force_all_finite')
            return _orig_check(*args, **kwargs)
        # patch both sklearn module and factor_analyzer's local reference
        _skval.check_array = _patched_check_array
        try:
            import factor_analyzer.factor_analyzer as _fa_mod
            _fa_mod.check_array = _patched_check_array
        except Exception:
            pass
except Exception:
    pass

try:
    from factor_analyzer import FactorAnalyzer
    from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
    FA_AVAILABLE = True
except ImportError:
    FA_AVAILABLE = False

router = APIRouter()

class EFARequest(BaseModel):
    data: list[dict[str, Any]] = Field(...)
    items: List[str] = Field(...)
    nFactors: Optional[int] = None          # None → auto-determine
    rotation: Optional[str] = 'varimax'
    method: Optional[str] = 'principal'
    loading_threshold: Optional[float] = None  # None → sample-size-based

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

def get_kmo_interpretation(kmo: float) -> str:
    if kmo >= 0.9: return "Marvelous"
    if kmo >= 0.8: return "Meritorious"
    if kmo >= 0.7: return "Middling"
    if kmo >= 0.6: return "Mediocre"
    if kmo >= 0.5: return "Miserable"
    return "Unacceptable"

# #4 sample-size-based loading threshold
def get_loading_threshold(n: int) -> float:
    if n >= 500: return 0.30
    if n >= 300: return 0.35
    if n >= 200: return 0.40
    if n >= 100: return 0.45
    return 0.55

# #1 parallel analysis: compare real eigenvalues vs random-data eigenvalues
def parallel_analysis(X: np.ndarray, n_iter: int = 100, percentile: int = 95) -> int:
    n, p = X.shape
    real_corr = np.corrcoef(X.T)
    real_eigs = np.sort(np.linalg.eigvalsh(real_corr))[::-1]
    rand_eigs = []
    rng = np.random.default_rng(42)
    for _ in range(n_iter):
        rand_data = rng.normal(size=(n, p))
        rand_corr = np.corrcoef(rand_data.T)
        rand_eigs.append(np.sort(np.linalg.eigvalsh(rand_corr))[::-1])
    rand_eigs = np.array(rand_eigs)
    threshold = np.percentile(rand_eigs, percentile, axis=0)
    n_factors = int(np.sum(real_eigs > threshold))
    return max(1, min(n_factors, p - 1))

# #1 Kaiser criterion
def kaiser_criterion(eigenvalues: np.ndarray) -> int:
    return max(1, int(np.sum(eigenvalues > 1)))

def generate_interpretation(n_factors, eigenvalues, variance_explained, cumulative_variance,
                            kmo, bartlett_p, loadings, items, communalities,
                            n_obs, threshold, factor_method, is_auto_factors, parallel_n):
    parts = []
    parts.append("**Overall Analysis**")

    kmo_interp = get_kmo_interpretation(kmo)
    parts.append(f"→ KMO = {kmo:.3f} ({kmo_interp}), Bartlett's test p {'< .001' if bartlett_p < 0.001 else f'= {bartlett_p:.3f}'}")
    parts.append(f"→ Extracted {n_factors} factor(s) explaining {cumulative_variance[-1]:.1f}% of total variance.")

    kaiser_count = sum(1 for e in eigenvalues if e > 1)
    if is_auto_factors:
        parts.append(f"→ Factor count auto-determined via parallel analysis: {parallel_n} factor(s) "
                     f"(Kaiser criterion suggested {kaiser_count}).")
    else:
        parts.append(f"→ Kaiser criterion suggests {kaiser_count} factor(s) (eigenvalue > 1).")

    # #3 ordinal data note
    parts.append(f"→ Note: Analysis uses Pearson correlations on the correlation matrix. "
                 "For ordinal/Likert data, polychoric correlations would be more appropriate.")

    parts.append("")
    parts.append("**Key Insights**")

    for i in range(n_factors):
        factor_loadings = loadings[:, i]
        high_loading_items = [
            (items[j], factor_loadings[j])
            for j in range(len(items)) if abs(factor_loadings[j]) >= threshold
        ]
        high_loading_items.sort(key=lambda x: abs(x[1]), reverse=True)
        if high_loading_items:
            items_str = ", ".join([f"{item} ({loading:.2f})" for item, loading in high_loading_items[:5]])
            parts.append(f"→ Factor {i+1} (Var: {variance_explained[i]:.1f}%): {items_str}")
        else:
            parts.append(f"→ Factor {i+1} (Var: {variance_explained[i]:.1f}%): no items above loading threshold ({threshold:.2f})")

    cross_loadings = []
    for j, item in enumerate(items):
        high_count = sum(1 for i in range(n_factors) if abs(loadings[j, i]) >= threshold)
        if high_count > 1:
            cross_loadings.append(item)
    if cross_loadings:
        parts.append(f"→ ⚠ Cross-loading items (≥{threshold:.2f} on 2+ factors): "
                     f"{', '.join(cross_loadings[:3])}{'...' if len(cross_loadings) > 3 else ''}")

    # #6 low communality warning
    low_comm = [(items[j], communalities[j]) for j in range(len(items)) if communalities[j] < 0.3]
    if low_comm:
        low_str = ", ".join([f"{item} ({c:.2f})" for item, c in low_comm[:5]])
        parts.append(f"→ ⚠ Low communality items (< 0.30): {low_str}. "
                     "These items share little variance with the extracted factors; consider removing them.")

    parts.append("")
    parts.append("**Recommendations**")

    if kmo < 0.6:
        parts.append("→ KMO is low; consider revising item selection or increasing sample size.")
    if cumulative_variance[-1] < 50:
        parts.append("→ Variance explained is low; consider adding more factors or revising items.")
    parts.append(f"→ Examine factor loadings ≥ |{threshold:.2f}| for item assignment "
                 f"(threshold based on N = {n_obs}).")
    parts.append("→ Consider confirmatory factor analysis (CFA) for validation.")
    parts.append("→ Report eigenvalues, variance explained, and rotation method used.")

    return "\n".join(parts)

def build_factor_interpretation(loadings, items, n_factors, threshold=0.4):
    interpretation = {}
    for f in range(n_factors):
        factor_name = f"Factor {f+1}"
        factor_vars = []
        factor_loads = []
        for i, item in enumerate(items):
            if abs(loadings[i, f]) >= threshold:
                factor_vars.append(item)
                factor_loads.append(float(loadings[i, f]))
        sorted_pairs = sorted(zip(factor_vars, factor_loads), key=lambda x: abs(x[1]), reverse=True)
        if sorted_pairs:
            factor_vars, factor_loads = zip(*sorted_pairs)
            interpretation[factor_name] = {
                "variables": list(factor_vars),
                "loadings": list(factor_loads)
            }
        else:
            interpretation[factor_name] = {
                "variables": [],
                "loadings": []
            }
    return interpretation

@router.post("/efa")
def efa_analysis(req: EFARequest):
    if not FA_AVAILABLE:
        raise HTTPException(status_code=500, detail="factor_analyzer not installed. Run: pip install factor_analyzer")

    try:
        df = pd.DataFrame(req.data)
        items = req.items
        rotation = req.rotation if req.rotation and req.rotation != 'none' else None
        method = req.method or 'principal'

        missing = [item for item in items if item not in df.columns]
        if missing:
            raise ValueError(f"Items not found: {missing}")

        if len(items) < 3:
            raise ValueError("Need at least 3 items for factor analysis")

        df_items = df[items].dropna()
        n_obs = len(df_items)

        if n_obs < 30:
            raise ValueError(f"Need at least 30 observations, got {n_obs}")

        warnings_list = []

        # #7 sample-to-variable ratio warnings
        ratio = n_obs / len(items)
        if ratio < 5:
            warnings_list.append(
                f"Sample-to-variable ratio is {ratio:.1f}:1 (N={n_obs}, items={len(items)}). "
                "A minimum of 5:1 is recommended; results may be unstable."
            )
        elif ratio < 10:
            warnings_list.append(
                f"Sample-to-variable ratio is {ratio:.1f}:1. "
                "A ratio of 10:1 or higher provides more stable factor solutions."
            )

        X = df_items.values

        # #4 loading threshold (user override or sample-size-based)
        threshold = req.loading_threshold if req.loading_threshold is not None else get_loading_threshold(n_obs)

        try:
            kmo_per_item, kmo_model = calculate_kmo(df_items)
            chi_sq, p_val = calculate_bartlett_sphericity(df_items)
        except Exception:
            kmo_model = 0.5
            chi_sq, p_val = 0, 1

        # #1 determine n_factors
        is_auto_factors = req.nFactors is None
        parallel_n = None
        if is_auto_factors:
            try:
                parallel_n = parallel_analysis(X)
            except Exception:
                parallel_n = None
            kaiser_n = kaiser_criterion(np.linalg.eigvalsh(np.corrcoef(X.T))[::-1])
            n_factors = parallel_n if parallel_n is not None else kaiser_n
            n_factors = max(1, min(n_factors, len(items) - 1))
        else:
            n_factors = req.nFactors
            if n_factors >= len(items):
                n_factors = len(items) - 1

        # #2 fit on correlation matrix
        corr_matrix = df_items.corr().values
        fa = FactorAnalyzer(n_factors=n_factors, rotation=rotation, method=method, is_corr_matrix=True)
        fa.fit(corr_matrix)

        loadings = fa.loadings_
        communalities = fa.get_communalities()
        eigenvalues, _ = fa.get_eigenvalues()
        variance = fa.get_factor_variance()

        variance_per_factor = [safe_float(v * 100) for v in variance[1][:n_factors]]
        cumulative_variance = [safe_float(v * 100) for v in variance[2][:n_factors]]

        # #6 low communality warnings
        low_comm_items = [items[j] for j in range(len(items)) if communalities[j] < 0.3]
        if low_comm_items:
            warnings_list.append(
                f"Low communality (< 0.30) detected for: {', '.join(low_comm_items)}. "
                "These items share little variance with the extracted factors and may need to be removed."
            )

        # plots with scoped style (#9)
        with plt.style.context('seaborn-v0_8-darkgrid'):
            fig, axes = plt.subplots(1, 3, figsize=(16, 5))

            n_eigen = min(len(eigenvalues), len(items))
            axes[0].bar(range(1, n_eigen + 1), eigenvalues[:n_eigen], alpha=0.7, color='#5B9BD5', edgecolor='black')
            axes[0].plot(range(1, n_eigen + 1), eigenvalues[:n_eigen], 'o-', color='#C44E52', lw=2)
            axes[0].axhline(y=1, color='green', linestyle='--', lw=2, label='Kaiser criterion (>1)')
            if parallel_n is not None:
                axes[0].axvline(x=parallel_n + 0.5, color='purple', linestyle=':', lw=2,
                                label=f'Parallel analysis ({parallel_n})')
            axes[0].set_xlabel('Factor')
            axes[0].set_ylabel('Eigenvalue')
            axes[0].set_title('Scree Plot', fontweight='bold')
            axes[0].legend(fontsize=8)

            loadings_df = pd.DataFrame(loadings, index=items, columns=[f'F{i+1}' for i in range(n_factors)])
            sns.heatmap(loadings_df, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                       vmin=-1, vmax=1, ax=axes[1], cbar_kws={'shrink': 0.8})
            axes[1].set_title(f'Factor Loadings (threshold={threshold:.2f})', fontweight='bold')

            comm_sorted = sorted(zip(items, communalities), key=lambda x: x[1], reverse=True)
            items_sorted, comm_vals = zip(*comm_sorted)
            colors = ['#2E7D32' if c >= 0.5 else '#FFA726' if c >= 0.3 else '#EF5350' for c in comm_vals]
            axes[2].barh(range(len(items_sorted)), comm_vals, color=colors, edgecolor='black', alpha=0.7)
            axes[2].set_yticks(range(len(items_sorted)))
            axes[2].set_yticklabels(items_sorted)
            axes[2].axvline(x=0.3, color='red', linestyle='--', lw=2, label='Low threshold (0.3)')
            axes[2].axvline(x=0.5, color='orange', linestyle='--', lw=1.5, label='Good threshold (0.5)')
            axes[2].set_xlabel('Communality')
            axes[2].set_title('Communalities', fontweight='bold')
            axes[2].legend(fontsize=8)
            axes[2].invert_yaxis()

            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            plot = f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"

        interpretation_text = generate_interpretation(
            n_factors, eigenvalues[:n_factors], variance_per_factor,
            cumulative_variance, kmo_model, p_val, loadings, items,
            communalities, n_obs, threshold, method,
            is_auto_factors, parallel_n
        )

        factor_interpretation = build_factor_interpretation(loadings, items, n_factors, threshold)

        response = {
            'n_observations': n_obs,
            'n_items': len(items),
            'n_factors': n_factors,
            'n_factors_auto': is_auto_factors,
            'rotation': rotation or 'none',
            'method': method,
            'loading_threshold': threshold,
            'variables': items,
            'eigenvalues': [safe_float(e) for e in eigenvalues[:len(items)]],
            'factor_loadings': [
                [safe_float(loadings[i, j]) for j in range(n_factors)]
                for i in range(len(items))
            ],
            'communalities': [safe_float(communalities[i]) for i in range(len(items))],
            'variance_explained': {
                'per_factor': variance_per_factor,
                'cumulative': cumulative_variance
            },
            'adequacy': {
                'kmo': safe_float(kmo_model),
                'kmo_interpretation': get_kmo_interpretation(kmo_model),
                'bartlett_statistic': safe_float(chi_sq),
                'bartlett_p_value': safe_float(p_val),
                'bartlett_significant': p_val < 0.05,
                'sample_to_variable_ratio': safe_float(ratio),
                'parallel_analysis_factors': parallel_n,
                'kaiser_factors': kaiser_criterion(eigenvalues),
            },
            'warnings': warnings_list,
            'interpretation': factor_interpretation,
            'full_interpretation': interpretation_text,
            'plot': plot
        }

        return _to_native(response)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
