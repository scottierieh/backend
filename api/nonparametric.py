from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats
import io, base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

router = APIRouter()

class NonparametricRequest(BaseModel):
    data: list[dict[str, Any]] = Field(...)
    test_type: str = Field(...)
    value_col: str = Field(...)
    group_col: Optional[str] = None
    paired_col: Optional[str] = None
    variables: Optional[List[str]] = None

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

def interpret_r_effect(r):
    r = abs(r)
    if r >= 0.5: return {'magnitude': 'Large', 'text': 'Large effect (r ≥ 0.5)'}
    elif r >= 0.3: return {'magnitude': 'Medium', 'text': 'Medium effect (0.3 ≤ r < 0.5)'}
    elif r >= 0.1: return {'magnitude': 'Small', 'text': 'Small effect (0.1 ≤ r < 0.3)'}
    else: return {'magnitude': 'Negligible', 'text': 'Negligible effect (r < 0.1)'}

def interpret_eta_effect(eta):
    if eta >= 0.14: return {'magnitude': 'Large', 'text': 'Large effect (η² ≥ 0.14)'}
    elif eta >= 0.06: return {'magnitude': 'Medium', 'text': 'Medium effect (0.06 ≤ η² < 0.14)'}
    else: return {'magnitude': 'Small', 'text': 'Small effect (η² < 0.06)'}

@router.post("/nonparametric")
def nonparametric_test(req: NonparametricRequest):
    try:
        df = pd.DataFrame(req.data)
        test_type = req.test_type.lower().replace('-', '_').replace(' ', '_')
        value_col = req.value_col
        group_col = req.group_col
        paired_col = req.paired_col
        variables = req.variables

        if test_type in ['mann_whitney', 'mannwhitney', 'mann_whitney_u']:
            if not group_col:
                raise ValueError("group_col required")
            clean = df[[value_col, group_col]].dropna()
            groups = clean[group_col].unique()
            if len(groups) != 2:
                raise ValueError("Exactly 2 groups required")
            g1 = clean[clean[group_col] == groups[0]][value_col].astype(float)
            g2 = clean[clean[group_col] == groups[1]][value_col].astype(float)
            stat, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')
            n1, n2 = len(g1), len(g2)
            r = 1 - (2 * stat) / (n1 * n2)
            effect_interp = interpret_r_effect(r)
            is_sig = p < 0.05
            group_stats = {
                str(groups[0]): {'n': n1, 'median': safe_float(g1.median()), 'mean': safe_float(g1.mean()), 'std': safe_float(g1.std()), 'min': safe_float(g1.min()), 'max': safe_float(g1.max())},
                str(groups[1]): {'n': n2, 'median': safe_float(g2.median()), 'mean': safe_float(g2.mean()), 'std': safe_float(g2.std()), 'min': safe_float(g2.min()), 'max': safe_float(g2.max())}
            }
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            sns.boxplot(x=group_col, y=value_col, data=clean, ax=axes[0], hue=group_col, palette='crest', legend=False)
            axes[0].set_title('Box Plot', fontweight='bold')
            sns.violinplot(x=group_col, y=value_col, data=clean, ax=axes[1], hue=group_col, palette='crest', legend=False)
            axes[1].set_title('Violin Plot', fontweight='bold')
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            plt.close(fig)
            buf.seek(0)
            plot = f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"
            sig_text = "significant" if is_sig else "not significant"
            interp = f"Mann-Whitney U = {stat:.2f}, p = {p:.4f}. The difference is {sig_text}. Effect size r = {r:.3f} ({effect_interp['magnitude']})."
            return _to_native({
                'test_name': 'Mann-Whitney U Test',
                'statistic': stat,
                'statistic_name': 'U',
                'p_value': p,
                'effect_size': r,
                'effect_size_interpretation': effect_interp,
                'is_significant': is_sig,
                'groups': [str(g) for g in groups],
                'group_stats': group_stats,
                'n1': n1,
                'n2': n2,
                'interpretation': {'conclusion': interp},
                'plot': plot
            })

        elif test_type in ['wilcoxon', 'wilcoxon_signed_rank']:
            if variables and len(variables) == 2:
                var1, var2 = variables[0], variables[1]
            elif paired_col:
                var1, var2 = value_col, paired_col
            else:
                raise ValueError("Two variables required")
            clean = df[[var1, var2]].dropna()
            x = clean[var1].astype(float)
            y = clean[var2].astype(float)
            stat, p = stats.wilcoxon(x, y)
            n = len(x)
            z = stats.norm.ppf(1 - p/2) if p > 0 and p < 1 else 0
            r = z / np.sqrt(n) if n > 0 else 0
            effect_interp = interpret_r_effect(r)
            is_sig = p < 0.05
            var_stats = {
                var1: {'n': n, 'median': safe_float(x.median()), 'mean': safe_float(x.mean()), 'std': safe_float(x.std())},
                var2: {'n': n, 'median': safe_float(y.median()), 'mean': safe_float(y.mean()), 'std': safe_float(y.std())}
            }
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.boxplot([x, y], labels=[var1, var2])
            ax.set_title('Paired Comparison', fontweight='bold')
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            plt.close(fig)
            buf.seek(0)
            plot = f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"
            sig_text = "significant" if is_sig else "not significant"
            interp = f"Wilcoxon W = {stat:.2f}, p = {p:.4f}. The difference is {sig_text}. Effect size r = {r:.3f} ({effect_interp['magnitude']})."
            return _to_native({
                'test_name': 'Wilcoxon Signed-Rank Test',
                'statistic': stat,
                'statistic_name': 'W',
                'p_value': p,
                'effect_size': r,
                'effect_size_interpretation': effect_interp,
                'is_significant': is_sig,
                'variables': [var1, var2],
                'var_stats': var_stats,
                'n': n,
                'interpretation': {'conclusion': interp},
                'plot': plot
            })

        elif test_type in ['kruskal', 'kruskal_wallis']:
            if not group_col:
                raise ValueError("group_col required")
            clean = df[[value_col, group_col]].dropna()
            groups = clean[group_col].unique()
            group_data = [clean[clean[group_col] == g][value_col].astype(float).values for g in groups]
            stat, p = stats.kruskal(*group_data)
            n = len(clean)
            k = len(groups)
            eta_sq = (stat - k + 1) / (n - k) if n > k else 0
            eta_sq = max(0, eta_sq)
            effect_interp = interpret_eta_effect(eta_sq)
            is_sig = p < 0.05
            group_stats = {}
            for g in groups:
                gd = clean[clean[group_col] == g][value_col]
                group_stats[str(g)] = {'n': len(gd), 'median': safe_float(gd.median()), 'mean': safe_float(gd.mean()), 'std': safe_float(gd.std())}
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.boxplot(x=group_col, y=value_col, data=clean, ax=ax, hue=group_col, palette='crest', legend=False)
            ax.set_title('Kruskal-Wallis Comparison', fontweight='bold')
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            plt.close(fig)
            buf.seek(0)
            plot = f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"
            sig_text = "significant" if is_sig else "not significant"
            interp = f"Kruskal-Wallis H = {stat:.2f}, p = {p:.4f}. The difference is {sig_text}. Effect size η² = {eta_sq:.3f} ({effect_interp['magnitude']})."
            return _to_native({
                'test_name': 'Kruskal-Wallis H Test',
                'statistic': stat,
                'statistic_name': 'H',
                'p_value': p,
                'effect_size': eta_sq,
                'effect_size_interpretation': effect_interp,
                'is_significant': is_sig,
                'groups': [str(g) for g in groups],
                'group_stats': group_stats,
                'n': n,
                'interpretation': {'conclusion': interp},
                'plot': plot
            })

        elif test_type in ['friedman']:
            if not variables or len(variables) < 3:
                raise ValueError("At least 3 repeated measures required")
            clean = df[variables].dropna()
            data_arrays = [clean[v].astype(float).values for v in variables]
            stat, p = stats.friedmanchisquare(*data_arrays)
            n = len(clean)
            k = len(variables)
            W = stat / (n * (k - 1)) if n > 0 and k > 1 else 0
            if W >= 0.5: effect_interp = {'magnitude': 'Large', 'text': 'Large effect (W ≥ 0.5)'}
            elif W >= 0.3: effect_interp = {'magnitude': 'Medium', 'text': 'Medium effect (0.3 ≤ W < 0.5)'}
            else: effect_interp = {'magnitude': 'Small', 'text': 'Small effect (W < 0.3)'}
            is_sig = p < 0.05
            cond_stats = {}
            for v in variables:
                cond_stats[v] = {'n': n, 'median': safe_float(clean[v].median()), 'mean': safe_float(clean[v].mean())}
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.boxplot([clean[v] for v in variables], labels=variables)
            ax.set_title('Friedman Test Comparison', fontweight='bold')
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            plt.close(fig)
            buf.seek(0)
            plot = f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"
            sig_text = "significant" if is_sig else "not significant"
            interp = f"Friedman χ² = {stat:.2f}, p = {p:.4f}. The difference is {sig_text}. Effect size W = {W:.3f} ({effect_interp['magnitude']})."
            return _to_native({
                'test_name': 'Friedman Test',
                'statistic': stat,
                'statistic_name': 'χ²',
                'p_value': p,
                'effect_size': W,
                'effect_size_interpretation': effect_interp,
                'is_significant': is_sig,
                'variables': variables,
                'condition_stats': cond_stats,
                'n': n,
                'interpretation': {'conclusion': interp},
                'plot': plot
            })

        else:
            raise ValueError(f"Unknown test type: {test_type}")

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
