from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, List, Optional
import numpy as np
import pandas as pd
import pingouin as pg
import math
import io, base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

router = APIRouter()

class ReliabilityRequest(BaseModel):
    data: list[dict[str, Any]] = Field(...)
    items: List[str] = Field(...)
    reverseCodeItems: Optional[List[str]] = []
    min_scale: Optional[float] = None   # #1 척도 최솟값 (예: 1)
    max_scale: Optional[float] = None   # #1 척도 최댓값 (예: 5)

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

def get_alpha_interpretation_level(alpha):
    if alpha >= 0.9: return 'Excellent'
    if alpha >= 0.8: return 'Good'
    if alpha >= 0.7: return 'Acceptable'
    if alpha >= 0.6: return 'Questionable'
    if alpha >= 0.5: return 'Poor'
    return 'Unacceptable'

def generate_interpretation(results):
    alpha = results['alpha']
    n_items = results['n_items']
    n_cases = results['n_cases']
    ci = results.get('confidence_interval', [None, None])
    sem = results.get('sem')
    item_stats = results.get('item_statistics', {})
    scale_stats = results.get('scale_statistics', {})

    alpha_if_deleted = item_stats.get('alpha_if_deleted', {})
    citc = item_stats.get('corrected_item_total_correlations', {})
    avg_inter_item = scale_stats.get('avg_inter_item_correlation')
    scale_mean = scale_stats.get('mean')
    scale_std = scale_stats.get('std')
    alpha_level = get_alpha_interpretation_level(alpha)
    omega = results.get('omega')
    ratio = n_cases / n_items if n_items > 0 else 0

    # CI / alpha string
    if ci[0] is not None and ci[1] is not None:
        alpha_str = f"α = {alpha:.3f}, 95% CI [{ci[0]:.3f}, {ci[1]:.3f}]"
    else:
        alpha_str = f"α = {alpha:.3f}"
    omega_str = f", McDonald's ω = {omega:.3f}" if omega is not None else ""

    # Alpha quality
    quality_map = {
        (0.95, 9): "excellent internal consistency; however, values above .95 may suggest item redundancy",
        (0.90, 0.95): "excellent internal consistency, suitable for research and high-stakes settings",
        (0.80, 0.90): "good internal consistency, suitable for most research and applied purposes",
        (0.70, 0.80): "acceptable internal consistency, meeting the threshold recommended by Nunnally (1978)",
        (0.60, 0.70): "questionable internal consistency; results should be interpreted with caution",
        (0.50, 0.60): "poor internal consistency; substantial scale revision is warranted",
        (-9, 0.50): "unacceptable internal consistency; the items do not form a reliable scale",
    }
    quality = next(v for (lo, hi), v in quality_map.items() if lo <= alpha < hi) if alpha >= 0 else \
        "a negative value, indicating a critical measurement problem (e.g., improperly reverse-coded items)"

    # Paragraph 1: reliability overview
    omega_note = ""
    if omega is not None:
        diff = abs(alpha - omega)
        omega_note = (f" McDonald's ω = {omega:.3f}; the negligible difference from alpha (Δ = {diff:.3f}) "
                      "supports tau-equivalence." if diff < 0.05 else
                      f" McDonald's ω = {omega:.3f} differs from alpha by {diff:.3f}, suggesting tau-equivalence "
                      "may be violated; omega is the preferred index.")

    p1 = (f"Internal consistency of the {n_items}-item scale was examined in a sample of {n_cases} participants. "
          f"Cronbach's alpha indicated {quality} ({alpha_str}{omega_str}).{omega_note}")

    # Paragraph 2: item-level evidence
    valid_citc = {k: v for k, v in citc.items() if v is not None}
    valid_aid  = {k: v for k, v in alpha_if_deleted.items() if v is not None}
    items_to_remove = [k for k, v in valid_aid.items() if v > alpha and v - alpha > 0.01]

    iic_part = ""
    if avg_inter_item is not None and not np.isnan(avg_inter_item):
        iic_qual = ("strong" if avg_inter_item >= 0.4 else "moderate" if avg_inter_item >= 0.2 else "weak")
        iic_part = f"The average inter-item correlation was r = {avg_inter_item:.3f}, indicating {iic_qual} item coherence."

    citc_part = ""
    if valid_citc:
        vals = list(valid_citc.values())
        low = [k for k, v in valid_citc.items() if v < 0.3]
        citc_part = (f" Corrected item-total correlations ranged from {min(vals):.3f} to {max(vals):.3f}; "
                     + ("all items met the r ≥ .30 criterion." if not low else
                        f"{len(low)} item(s) fell below r = .30 ({', '.join(low)}), warranting review."))

    aid_part = ""
    if items_to_remove:
        details = ", ".join(f"{k} (α = {valid_aid[k]:.3f})" for k in items_to_remove[:3])
        aid_part = f" Removal of {details} would increase alpha, suggesting these items reduce scale consistency."
    else:
        aid_part = " No item removal would meaningfully increase alpha, indicating all items contribute positively."

    p2 = iic_part + citc_part + aid_part

    # Paragraph 3: recommendations + APA
    if alpha >= 0.7:
        rec = "The scale is suitable for research use."
    elif alpha >= 0.6:
        rec = "Scale refinement is recommended before use; consider revising or removing poorly fitting items."
    else:
        rec = "Major revision is required; review item content and consider exploratory factor analysis."

    sem_part = f" SEM = {sem:.3f}." if sem is not None else ""
    apa = alpha_str + (f", ω = {omega:.3f}" if omega is not None else "")
    desc = f" (M = {scale_mean:.2f}, SD = {scale_std:.2f})" if scale_mean is not None and scale_std is not None else ""

    p3 = (f"{rec}{sem_part} "
          f"For APA 7th edition, report as: {apa}, N = {n_cases}, {n_items} items{desc}.")

    return f"**Overall Analysis**\n{p1}\n\n{p2}\n\n{p3}"

def create_plot(df_items, results):
    with plt.style.context('seaborn-v0_8-darkgrid'):
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle("Reliability Analysis Results", fontsize=16, fontweight='bold')

        item_stats = results['item_statistics']
        alpha = results['alpha']

        # 1. Item-Total Correlations
        citc = item_stats['corrected_item_total_correlations']
        items = list(citc.keys())
        correlations = list(citc.values())
        correlations_abs = [abs(c) for c in correlations]
        colors = ['#e74c3c' if (c is None or c < 0 or abs(c) < 0.3) else '#4C72B0' for c in correlations]

        axes[0, 0].barh(items, correlations_abs, color=colors)
        axes[0, 0].axvline(x=0.3, color='red', linestyle='--', lw=1.5, label='Threshold (r = 0.30)')
        axes[0, 0].set_xlabel('Corrected Item-Total Correlation')
        axes[0, 0].set_title('Item-Total Correlations')
        axes[0, 0].legend(loc='lower right')

        # 2. Alpha if Item Deleted — #4 음수 alpha 분기
        aid = item_stats['alpha_if_deleted']
        aid_items = list(aid.keys())
        raw_aid_values = [aid[k] for k in aid_items]
        alpha_abs = abs(alpha)

        if alpha < 0:
            # 음수 alpha: 막대 대신 warning 텍스트 박스
            axes[0, 1].set_visible(False)
            axes[0, 1].set_visible(True)
            axes[0, 1].set_xlim(0, 1)
            axes[0, 1].set_ylim(0, 1)
            axes[0, 1].set_xticks([])
            axes[0, 1].set_yticks([])
            axes[0, 1].set_title('Alpha if Item Deleted')
            axes[0, 1].text(
                0.5, 0.5,
                "⚠ Cronbach's α is negative.\n\n"
                "Alpha-if-deleted values are\nnot meaningful in this context.\n\n"
                "Check reverse coding and\nitem structure.",
                ha='center', va='center', fontsize=11,
                color='#c0392b', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='#fdecea', edgecolor='#e74c3c', linewidth=2),
                transform=axes[0, 1].transAxes
            )
        else:
            aid_values = [abs(v) if v is not None else 0 for v in raw_aid_values]
            colors_aid = ['#55A868' if v > alpha_abs else '#4C72B0' for v in aid_values]
            axes[0, 1].barh(aid_items, aid_values, color=colors_aid)
            axes[0, 1].axvline(x=alpha_abs, color='red', linestyle='--', lw=1.5,
                               label=f'Current α = {alpha_abs:.3f}')
            axes[0, 1].set_xlabel("Cronbach's α if Item Deleted")
            axes[0, 1].set_title('Alpha if Item Deleted')
            axes[0, 1].legend(loc='lower right')

        # 3. Inter-Item Correlation Heatmap
        corr_matrix = df_items.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='vlag', center=0,
                    ax=axes[1, 0], square=True, linewidths=0.5)
        axes[1, 0].set_title('Inter-Item Correlation Matrix')

        # 4. Scale Score Distribution Histogram (replaces Q-Q plot, #7)
        total_scores = df_items.sum(axis=1)
        axes[1, 1].hist(total_scores, bins='auto', color='#4C72B0', edgecolor='white', alpha=0.85)
        axes[1, 1].axvline(total_scores.mean(), color='red', linestyle='--', lw=1.5,
                           label=f'Mean = {total_scores.mean():.2f}')
        axes[1, 1].set_xlabel('Scale Score')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Scale Score Distribution')
        axes[1, 1].legend()

        fig.text(0.5, 0.02,
                 f"Overall Cronbach's α = {alpha:.3f} ({get_alpha_interpretation_level(alpha)})",
                 ha='center', fontsize=12, fontweight='bold',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])

        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100)
        plt.close(fig)
        buf.seek(0)
        return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"

@router.post("/reliability")
def reliability_analysis(req: ReliabilityRequest):
    try:
        df = pd.DataFrame(req.data)
        items = req.items
        reverse_code_items = req.reverseCodeItems or []

        missing_cols = [col for col in items if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Columns not found: {', '.join(missing_cols)}")

        # #2 item count guard
        if len(items) < 2:
            raise ValueError("At least 2 items are required for reliability analysis.")

        df_items = df[items].copy()

        # #3 coerce all columns to numeric
        non_numeric = []
        for col in df_items.columns:
            before = df_items[col].notna().sum()
            df_items[col] = pd.to_numeric(df_items[col], errors='coerce')
            after = df_items[col].notna().sum()
            if after < before:
                non_numeric.append(col)
        if non_numeric:
            raise ValueError(
                f"Non-numeric values found in item(s): {', '.join(non_numeric)}. "
                "All items must be numeric."
            )

        # #1 reverse coding with scale range
        for col in reverse_code_items:
            if col in df_items.columns:
                if req.min_scale is not None and req.max_scale is not None:
                    df_items[col] = req.min_scale + req.max_scale - df_items[col]
                else:
                    # fallback: observed min/max (warn in response)
                    max_val = df_items[col].max()
                    min_val = df_items[col].min()
                    df_items[col] = max_val + min_val - df_items[col]

        # #4 track missing cases
        original_n = len(df_items)
        df_items.dropna(inplace=True)
        analyzed_n = len(df_items)
        dropped_n = original_n - analyzed_n

        if analyzed_n < 2:
            raise ValueError("Not enough valid data after handling missing values (need at least 2 rows).")

        # item count warnings
        two_item_warning = len(items) == 2
        three_item_warning = len(items) == 3

        alpha_results = pg.cronbach_alpha(data=df_items, nan_policy='listwise')
        total_score = df_items.sum(axis=1)

        corrected_item_total_correlations = {}
        alpha_if_deleted = {}

        for item in df_items.columns:
            item_score = df_items[item]
            rest_score = total_score - item_score

            # #5 zero-variance guard
            if item_score.std() == 0 or rest_score.std() == 0:
                corrected_item_total_correlations[item] = None
            else:
                try:
                    correlation = pg.corr(item_score, rest_score)['r'].iloc[0]
                    corrected_item_total_correlations[item] = correlation
                except Exception:
                    corrected_item_total_correlations[item] = None

            # #6 alpha-if-deleted guard (unstable for ≤2 items)
            remaining = df_items.drop(columns=item)
            if remaining.shape[1] < 2:
                alpha_if_deleted[item] = None
            else:
                try:
                    alpha_if_deleted[item] = pg.cronbach_alpha(data=remaining)[0]
                except Exception:
                    alpha_if_deleted[item] = None

        sem_value = None
        if alpha_results[0] >= 0:
            sem_value = df_items.sum(axis=1).std() * (1 - alpha_results[0]) ** 0.5
            if math.isnan(sem_value):
                sem_value = None

        response = {
            'alpha': alpha_results[0],
            'n_items': df_items.shape[1],
            'n_cases': analyzed_n,
            # #4 missing case counts
            'original_n': original_n,
            'analyzed_n': analyzed_n,
            'dropped_n_missing': dropped_n,
            'confidence_interval': list(alpha_results[1]),
            'sem': sem_value,
            'item_statistics': {
                'means': df_items.mean().to_dict(),
                'stds': df_items.std().to_dict(),
                'corrected_item_total_correlations': corrected_item_total_correlations,
                'alpha_if_deleted': alpha_if_deleted,
            },
            'scale_statistics': {
                'mean': total_score.mean(),
                'std': total_score.std(),
                'variance': total_score.var(),
                'avg_inter_item_correlation': float(
                    df_items.corr().values[
                        np.triu_indices_from(df_items.corr().values, k=1)
                    ].mean()
                )
            },
            # #1 reverse coding method used
            'reverse_coding_method': (
                'scale_range' if (req.min_scale is not None and req.max_scale is not None)
                else ('observed_range' if reverse_code_items else 'none')
            ),
        }

        # #2 two-item warning
        if two_item_warning:
            response['warnings'] = response.get('warnings', [])
            response['warnings'].append(
                "Only 2 items provided. Cronbach's α with 2 items is less stable; "
                "interpret with caution and consider adding more items."
            )

        # 3-item warning
        if three_item_warning:
            response['warnings'] = response.get('warnings', [])
            response['warnings'].append(
                "Only 3 items provided. Reliability interpretation is limited with fewer than 3 items; "
                "4+ items are recommended for a stable and meaningful α estimate."
            )

        # #1 warn if observed-range fallback was used
        if reverse_code_items and (req.min_scale is None or req.max_scale is None):
            response['warnings'] = response.get('warnings', [])
            response['warnings'].append(
                "Reverse coding used observed min/max values because min_scale/max_scale were not provided. "
                "For accurate reverse coding, supply the intended scale range (e.g. min_scale=1, max_scale=5)."
            )

        # #4 warn if cases were dropped
        if dropped_n > 0:
            response['warnings'] = response.get('warnings', [])
            response['warnings'].append(
                f"{dropped_n} case(s) removed due to missing values (listwise deletion). "
                f"Analysis based on {analyzed_n} of {original_n} observations."
            )

        # #1 사례수 경고 (API 레벨)
        warnings_list = response.get('warnings', [])
        if analyzed_n < 20:
            warnings_list.append(
                f"Very small sample (N = {analyzed_n}). α estimate is highly unstable; "
                "results should not be used for decision-making. Collect at least 50+ cases."
            )
        elif analyzed_n < 50:
            warnings_list.append(
                f"Small sample (N = {analyzed_n}). α estimate may be unstable. "
                "A minimum of 50 cases is recommended for preliminary analysis."
            )
        elif analyzed_n < 100:
            warnings_list.append(
                f"Moderate sample (N = {analyzed_n}). Consider collecting 100+ cases "
                "for a more stable α estimate."
            )
        if warnings_list:
            response['warnings'] = warnings_list

        # #3 constant item warnings
        constant_items = [col for col in df_items.columns if df_items[col].std() == 0]
        near_constant_items = [
            col for col in df_items.columns
            if col not in constant_items and df_items[col].std() < 0.1
        ]
        for ci_item in constant_items:
            response.setdefault('warnings', []).append(
                f"Item '{ci_item}' has zero variance (all responses identical). "
                "This item cannot contribute to reliability and should be reviewed or removed."
            )
        for nc_item in near_constant_items:
            response.setdefault('warnings', []).append(
                f"Item '{nc_item}' may be near-constant (SD < 0.1). "
                "Very low variance can distort reliability estimates."
            )

        # #5 avg inter-item: exclude constant items
        valid_cols = [col for col in df_items.columns if df_items[col].std() > 0]
        if len(valid_cols) >= 2:
            corr_vals = df_items[valid_cols].corr().values
            avg_iic = float(corr_vals[np.triu_indices_from(corr_vals, k=1)].mean())
        else:
            avg_iic = None
        response['scale_statistics']['avg_inter_item_correlation'] = avg_iic

        # #6 McDonald's omega — correct formula: Σλ² / (Σλ² + Σθ)
        omega_value = None
        try:
            from sklearn.decomposition import FactorAnalysis
            if len(valid_cols) >= 2 and analyzed_n >= len(valid_cols) + 5:
                fa = FactorAnalysis(n_components=1, random_state=42)
                fa.fit(df_items[valid_cols].values)
                loadings = fa.components_[0]   # λ, shape: (n_items,)
                unique_var = fa.noise_variance_ # θ, shape: (n_items,)
                sum_lambda_sq = float(np.sum(loadings ** 2))   # Σλ²
                sum_theta = float(np.sum(unique_var))           # Σθ
                denom = sum_lambda_sq + sum_theta
                omega_value = sum_lambda_sq / denom if denom > 0 else None
        except ImportError:
            response.setdefault('warnings', []).append(
                "scikit-learn not available; McDonald's ω could not be computed."
            )
        except Exception as e:
            response.setdefault('warnings', []).append(
                f"McDonald's ω could not be computed: {str(e)}"
            )
        response['omega'] = omega_value

        response['interpretation'] = generate_interpretation(response)
        response['plot'] = create_plot(df_items, response)

        return _to_native(response)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
