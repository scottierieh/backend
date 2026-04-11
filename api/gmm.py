from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from scipy.spatial import ConvexHull
from scipy import stats
import logging
import io, base64

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
import seaborn as sns

sns.set_theme(style="darkgrid")

logger = logging.getLogger(__name__)
router = APIRouter()


class GMMRequest(BaseModel):
    data: list[dict[str, Any]] = Field(...)
    items: List[str] = Field(...)
    nComponents: int = Field(default=3)
    covarianceType: Optional[str] = 'full'
    boundaryType: Optional[str] = 'ellipse'
    scalerType: Optional[str] = 'standard'


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
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _validate_input(df: pd.DataFrame, items: List[str], n_components: int):
    """Comprehensive input validation matching KMeans/DBSCAN standards."""
    input_warnings = []

    # 1. Column existence
    missing = [c for c in items if c not in df.columns]
    if missing:
        raise ValueError(f"Column(s) not found: {', '.join(missing)}")

    # 2. Numeric check
    non_numeric = []
    for col in items:
        if not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric.append(col)
    if non_numeric:
        raise ValueError(
            f"Non-numeric column(s): {', '.join(non_numeric)}. "
            "GMM requires numeric variables only."
        )

    # 3. Categorical disguised as numeric (low unique ratio)
    for col in items:
        n_unique = df[col].nunique(dropna=True)
        n_total = df[col].count()
        if n_total > 20 and n_unique <= 5:
            input_warnings.append(
                f"'{col}' has only {n_unique} unique values — may be categorical. "
                "Consider excluding it from clustering."
            )

    # 4. Constant / near-zero variance
    for col in items:
        col_std = df[col].std(skipna=True)
        col_mean = abs(df[col].mean(skipna=True))
        if col_std == 0:
            raise ValueError(
                f"'{col}' is constant (zero variance). Remove it before clustering."
            )
        if col_mean > 1e-8 and (col_std / col_mean) < 0.01:
            input_warnings.append(
                f"'{col}' has near-zero variance (CV < 1%). "
                "It may not contribute meaningfully to clustering."
            )

    # 5. Duplicate rows warning
    cluster_data_raw = df[items].dropna()
    n_total = len(cluster_data_raw)
    n_dupes = n_total - cluster_data_raw.drop_duplicates().shape[0]
    if n_dupes > 0:
        dupe_pct = n_dupes / n_total * 100
        if dupe_pct > 10:
            input_warnings.append(
                f"{n_dupes} duplicate rows ({dupe_pct:.1f}%) detected. "
                "GMM may be distorted by duplicates — consider de-duplication."
            )

    # 6. Extreme outlier warning (per column, z-score > 5)
    for col in items:
        z = np.abs(stats.zscore(cluster_data_raw[col].dropna()))
        extreme = int((z > 5).sum())
        if extreme > 0:
            input_warnings.append(
                f"'{col}' has {extreme} extreme outlier(s) (|z| > 5). "
                "Consider using Robust scaler or removing outliers."
            )

    # 7. Minimum sample size
    if n_total < n_components * 5:
        raise ValueError(
            f"Too few samples ({n_total}) for {n_components} components. "
            f"Need at least {n_components * 5} observations."
        )
    if n_total < 30:
        input_warnings.append(
            f"Small sample size (n={n_total}). "
            "GMM results may not generalise well."
        )

    return input_warnings, cluster_data_raw


def _build_quality_warnings(
    profiles: dict,
    avg_probability: float,
    recommended_k: int,
    n_components: int,
    covariance_type: str,
    silhouette: float,
    converged: bool,
    n_samples: int,
):
    """Build structured quality warnings."""
    quality_warnings = []

    # Convergence
    if not converged:
        quality_warnings.append(
            "GMM did not converge. Try increasing max_iter or reducing n_components."
        )

    # Average probability too low
    if avg_probability < 0.6:
        quality_warnings.append(
            f"Average membership probability is low ({avg_probability:.1%}). "
            "Clusters may overlap significantly."
        )

    # Recommended k vs chosen k gap
    k_diff = abs(recommended_k - n_components)
    if k_diff >= 2:
        quality_warnings.append(
            f"BIC recommends k={recommended_k} but you chose k={n_components} "
            f"(difference: {k_diff}). Consider re-running with k={recommended_k}."
        )

    # Small component warning
    for name, p in profiles.items():
        if p['percentage'] < 5.0:
            quality_warnings.append(
                f"{name} is very small ({p['percentage']:.1f}% of data, n={p['size']}). "
                "May be unstable or represent noise."
            )

    # Covariance type overfitting risk
    if covariance_type == 'full' and n_samples < n_components * 20:
        quality_warnings.append(
            f"'full' covariance with small sample (n={n_samples}) risks overfitting. "
            "Consider 'diag' or 'tied' covariance type."
        )

    # Low silhouette
    if silhouette < 0.25 and silhouette > 0:
        quality_warnings.append(
            f"Silhouette score ({silhouette:.3f}) is low. "
            "Component separation is weak — clusters may heavily overlap."
        )

    return quality_warnings


def _soft_clustering_diagnostics(probabilities: np.ndarray, labels: np.ndarray, n_components: int):
    """GMM-specific soft clustering analysis."""
    n_samples = len(labels)

    # Max probability per point = confidence of assignment
    max_probs = probabilities.max(axis=1)
    avg_probability = float(max_probs.mean())

    # Low-confidence points: max prob < 0.6
    low_conf_mask = max_probs < 0.6
    low_conf_count = int(low_conf_mask.sum())
    low_conf_pct = low_conf_count / n_samples * 100

    # Ambiguous points: top-2 probs within 0.1 of each other
    if n_components >= 2:
        sorted_probs = np.sort(probabilities, axis=1)[:, ::-1]
        ambiguous_mask = (sorted_probs[:, 0] - sorted_probs[:, 1]) < 0.1
        ambiguous_count = int(ambiguous_mask.sum())
        ambiguous_pct = ambiguous_count / n_samples * 100
    else:
        ambiguous_count = 0
        ambiguous_pct = 0.0

    # Per-component soft stats
    per_component = {}
    for k in range(n_components):
        mask = labels == k
        if mask.sum() == 0:
            continue
        comp_probs = probabilities[mask, k]
        per_component[f'Component {k + 1}'] = {
            'mean_probability': safe_float(comp_probs.mean()),
            'min_probability': safe_float(comp_probs.min()),
            'soft_members': int((comp_probs < 0.5).sum()),
        }

    return {
        'avg_probability': avg_probability,
        'low_confidence_count': low_conf_count,
        'low_confidence_pct': safe_float(low_conf_pct),
        'ambiguous_count': ambiguous_count,
        'ambiguous_pct': safe_float(ambiguous_pct),
        'per_component': per_component,
        'note': (
            f"{low_conf_count} point(s) ({low_conf_pct:.1f}%) have low assignment confidence (<60%). "
            f"{ambiguous_count} point(s) ({ambiguous_pct:.1f}%) are ambiguous between components."
        )
    }


def _feature_drivers(cluster_data: pd.DataFrame, labels: np.ndarray, items: List[str]):
    """ANOVA-based feature importance across GMM components."""
    if len(np.unique(labels)) < 2:
        return None

    try:
        X = cluster_data[items].values
        f_stats, p_values = f_classif(X, labels)

        features = []
        for i, col in enumerate(items):
            f = safe_float(f_stats[i])
            p = safe_float(p_values[i], default=1.0)

            # Eta-squared effect size
            groups = [X[labels == k, i] for k in np.unique(labels)]
            grand_mean = X[:, i].mean()
            ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups if len(g) > 0)
            ss_total = ((X[:, i] - grand_mean) ** 2).sum()
            eta_sq = safe_float(ss_between / ss_total) if ss_total > 0 else 0.0

            effect = 'large' if eta_sq >= 0.14 else 'medium' if eta_sq >= 0.06 else 'small'
            features.append({
                'feature': col,
                'f_stat': f,
                'p_value': p,
                'eta_squared': eta_sq,
                'effect_size': effect,
                'is_significant': bool(p < 0.05),
                'rank': 0,
            })

        features.sort(key=lambda x: x['eta_squared'], reverse=True)
        for rank, feat in enumerate(features, 1):
            feat['rank'] = rank

        top_driver = features[0]['feature'] if features else None

        return {
            'features': features,
            'top_driver': top_driver,
            'note': 'ANOVA F-test: measures how well each variable separates GMM components.',
        }
    except Exception as e:
        logger.warning(f"Feature drivers failed: {e}")
        return None


def draw_convex_hull(ax, points, color, alpha=0.15):
    if len(points) < 3:
        return
    try:
        hull = ConvexHull(points)
        hull_points = np.append(hull.vertices, hull.vertices[0])
        ax.fill(
            points[hull_points, 0], points[hull_points, 1],
            color=color, alpha=alpha, edgecolor=color, linewidth=2, linestyle='-'
        )
    except Exception as e:
        logger.debug(f"ConvexHull skipped: {e}")


def draw_gmm_ellipse(ax, mean, cov, color, n_std=2.0, alpha=0.2):
    """Draw ellipse for GMM component based on covariance matrix."""
    try:
        if cov.ndim == 1:
            cov = np.diag(cov)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
        angle = np.degrees(np.arctan2(*eigenvectors[:, 0][::-1]))
        width = 2 * n_std * np.sqrt(max(eigenvalues[0], 1e-10))
        height = 2 * n_std * np.sqrt(max(eigenvalues[1], 1e-10))
        ellipse = Ellipse(
            xy=mean, width=width, height=height, angle=angle,
            facecolor=color, alpha=alpha, edgecolor=color,
            linewidth=2, linestyle='--'
        )
        ax.add_patch(ellipse)
    except Exception as e:
        logger.debug(f"Ellipse skipped: {e}")


@router.post("/gmm")
def gmm_clustering(req: GMMRequest):
    try:
        df = pd.DataFrame(req.data)
        items = req.items
        n_components = req.nComponents
        covariance_type = req.covarianceType or 'full'
        boundary_type = req.boundaryType or 'ellipse'
        scaler_type = req.scalerType or 'standard'

        # ── 1. Comprehensive input validation ──────────────────────────────
        input_warnings, cluster_data = _validate_input(df, items, n_components)

        # ── 2. Scaling ─────────────────────────────────────────────────────
        if scaler_type == 'robust':
            scaler = RobustScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()

        cluster_data_scaled = scaler.fit_transform(cluster_data)
        n_samples, n_features = cluster_data_scaled.shape

        # ── 3. Optimal k search (BIC / AIC / Silhouette) ──────────────────
        max_k = min(10, n_samples - 1)
        k_range = list(range(2, max_k + 1))
        bic_scores, aic_scores, silhouette_scores = [], [], []

        for k in k_range:
            try:
                gmm_k = GaussianMixture(
                    n_components=k, covariance_type=covariance_type,
                    n_init=10, random_state=42, max_iter=300
                )
                gmm_k.fit(cluster_data_scaled)
                bic_scores.append(safe_float(gmm_k.bic(cluster_data_scaled)))
                aic_scores.append(safe_float(gmm_k.aic(cluster_data_scaled)))
                lbl = gmm_k.predict(cluster_data_scaled)
                if len(np.unique(lbl)) > 1:
                    silhouette_scores.append(safe_float(silhouette_score(cluster_data_scaled, lbl)))
                else:
                    silhouette_scores.append(-1.0)
            except Exception as e:
                logger.warning(f"k={k} model fit failed: {e}")
                bic_scores.append(float('inf'))
                aic_scores.append(float('inf'))
                silhouette_scores.append(-1.0)

        recommended_k = k_range[int(np.argmin(bic_scores))] if bic_scores else 3

        # ── 4. Final GMM fit ───────────────────────────────────────────────
        gmm = GaussianMixture(
            n_components=n_components, covariance_type=covariance_type,
            n_init=10, random_state=42, max_iter=300
        )
        gmm.fit(cluster_data_scaled)
        labels = gmm.predict(cluster_data_scaled)
        probabilities = gmm.predict_proba(cluster_data_scaled)

        unique_labels, counts = np.unique(labels, return_counts=True)

        # ── 5. Quality metrics ─────────────────────────────────────────────
        if len(unique_labels) > 1:
            silhouette = safe_float(silhouette_score(cluster_data_scaled, labels))
            davies_bouldin = safe_float(davies_bouldin_score(cluster_data_scaled, labels))
            calinski = safe_float(calinski_harabasz_score(cluster_data_scaled, labels))
        else:
            silhouette, davies_bouldin, calinski = 0.0, 0.0, 0.0

        log_likelihood = safe_float(gmm.score(cluster_data_scaled) * n_samples)
        bic = safe_float(gmm.bic(cluster_data_scaled))
        aic = safe_float(gmm.aic(cluster_data_scaled))

        # ── 6. Profiles ────────────────────────────────────────────────────
        profiles = {}
        for i, label in enumerate(unique_labels):
            mask = labels == label
            subset = cluster_data.iloc[mask]
            avg_prob = safe_float(probabilities[mask, label].mean()) if mask.sum() > 0 else 0.0
            profiles[f'Component {label + 1}'] = {
                'size': int(counts[i]),
                'percentage': safe_float(counts[i] / n_samples * 100),
                'mean': {col: safe_float(subset[col].mean()) for col in items},
                'weight': safe_float(gmm.weights_[label]),
                'avg_probability': avg_prob,
            }

        # ── 7. Soft clustering diagnostics ─────────────────────────────────
        soft_diagnostics = _soft_clustering_diagnostics(probabilities, labels, n_components)

        # ── 8. Feature drivers (ANOVA) ─────────────────────────────────────
        feature_drivers = _feature_drivers(cluster_data, labels, items)

        # ── 9. Quality warnings ────────────────────────────────────────────
        quality_warnings = _build_quality_warnings(
            profiles=profiles,
            avg_probability=soft_diagnostics['avg_probability'],
            recommended_k=recommended_k,
            n_components=n_components,
            covariance_type=covariance_type,
            silhouette=silhouette,
            converged=bool(gmm.converged_),
            n_samples=n_samples,
        )

        # ── 10. Interpretations ────────────────────────────────────────────
        overall_means = cluster_data.mean()
        overall_std = cluster_data.std()

        cluster_profiles_text = []
        for name, profile in profiles.items():
            centroid = pd.Series(profile['mean'])
            deviations = (centroid - overall_means) / overall_std.replace(0, 1)
            top_features = deviations.nlargest(2).index.tolist()
            bottom_features = deviations.nsmallest(2).index.tolist()
            profile_desc = (
                f"<strong>{name} ({profile['percentage']:.1f}% of data, weight: {profile['weight']:.3f}):</strong> "
                f"Characterized by high values in <strong>{', '.join(top_features)}</strong> "
                f"and low values in <strong>{', '.join(bottom_features)}</strong>. "
                f"Average membership probability: {profile['avg_probability']:.1%}."
            )
            cluster_profiles_text.append(profile_desc)

        if silhouette >= 0.7:
            quality_desc = "strong and well-defined."
        elif silhouette >= 0.5:
            quality_desc = "reasonable and distinct."
        elif silhouette >= 0.25:
            quality_desc = "weak and could have some overlap."
        else:
            quality_desc = "not well-defined; results should be interpreted with caution."

        overall_quality = (
            f"The <strong>Silhouette Score of {silhouette:.3f}</strong> indicates the clustering structure is {quality_desc} "
            f"The <strong>BIC ({bic:.2f})</strong> and <strong>AIC ({aic:.2f})</strong> measure model fit (lower is better). "
            f"<strong>Log-Likelihood: {log_likelihood:.2f}</strong>. "
            f"Higher is better for the <strong>Calinski-Harabasz Score ({calinski:.2f})</strong>. "
            f"Lower is better for the <strong>Davies-Bouldin Score ({davies_bouldin:.3f})</strong>."
        )

        percentages = [p['percentage'] for p in profiles.values()]
        if len(percentages) > 1:
            max_p, min_p = max(percentages), min(percentages)
            if min_p > 0 and max_p / min_p > 3:
                dist_desc = "The component sizes are imbalanced — some components are significantly larger."
            else:
                dist_desc = "The components are relatively balanced in size."
        else:
            dist_desc = "Single component detected."

        # ── 11. Visualization ──────────────────────────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        colors = sns.color_palette('husl', n_colors=n_components)

        # Panel 1: BIC / AIC
        ax1 = axes[0, 0]
        ax1.set_facecolor('#f5f5f5')
        ax1.plot(k_range, bic_scores, 'o-', color='#2E86AB', linewidth=2.5, markersize=8,
                 markerfacecolor='white', markeredgewidth=2, label='BIC')
        ax1.plot(k_range, aic_scores, 's--', color='#E8505B', linewidth=2.5, markersize=8,
                 markerfacecolor='white', markeredgewidth=2, label='AIC')
        ax1.axvline(x=recommended_k, color='#F4A261', linewidth=2, linestyle=':', label=f'Recommended k={recommended_k}')
        ax1.set_xlabel('Number of Components (k)', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Information Criterion', fontsize=11, fontweight='bold')
        ax1.set_title('BIC/AIC for Optimal k', fontsize=13, fontweight='bold')
        ax1.legend(loc='best', frameon=True, facecolor='white')
        ax1.grid(True, alpha=0.4, linestyle='--')

        # Panel 2: Silhouette by k
        ax2 = axes[0, 1]
        ax2.set_facecolor('#f5f5f5')
        bar_colors = ['#E8505B' if s == max(silhouette_scores) else '#5B9BD5' for s in silhouette_scores]
        ax2.bar(range(len(k_range)), silhouette_scores, color=bar_colors, alpha=0.85,
                edgecolor='white', linewidth=1.5)
        ax2.set_xlabel('Number of Components (k)', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Silhouette Score', fontsize=11, fontweight='bold')
        ax2.set_title('Silhouette Scores by k', fontsize=13, fontweight='bold')
        ax2.set_xticks(range(len(k_range)))
        ax2.set_xticklabels(k_range)
        ax2.grid(True, alpha=0.4, axis='y', linestyle='--')

        # Panel 3: PCA scatter with ellipses
        ax3 = axes[1, 0]
        ax3.set_facecolor('#f0f0f0')

        if n_features >= 2:
            pca = PCA(n_components=2)
            pca_data = pca.fit_transform(cluster_data_scaled)
            means_pca = pca.transform(gmm.means_)

            for i, label in enumerate(unique_labels):
                if boundary_type == 'ellipse':
                    if covariance_type == 'full':
                        cov = gmm.covariances_[label]
                    elif covariance_type == 'tied':
                        cov = gmm.covariances_
                    elif covariance_type == 'diag':
                        cov = np.diag(gmm.covariances_[label])
                    else:  # spherical
                        cov = gmm.covariances_[label] * np.eye(n_features)
                    cov_pca = pca.components_ @ cov @ pca.components_.T
                    draw_gmm_ellipse(ax3, means_pca[label], cov_pca, colors[i], n_std=2.0, alpha=0.2)
                else:
                    mask = labels == label
                    draw_convex_hull(ax3, pca_data[mask], colors[i], alpha=0.2)

            for i, label in enumerate(unique_labels):
                mask = labels == label
                ax3.scatter(pca_data[mask, 0], pca_data[mask, 1], c=[colors[i]],
                            label=f'Component {label + 1}', alpha=0.7, s=60,
                            edgecolors='white', linewidth=0.8)

            ax3.scatter(means_pca[:, 0], means_pca[:, 1], s=120, c='white',
                        marker='o', edgecolors='black', linewidth=1.5, zorder=10)
            ax3.scatter(means_pca[:, 0], means_pca[:, 1], s=40, c='black',
                        marker='x', linewidth=1, zorder=11)

            ax3.set_title('Components with Gaussian Ellipses (PCA)', fontsize=13, fontweight='bold')
            ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=11)
            ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=11)

            handles, _ = ax3.get_legend_handles_labels()
            mean_marker = Line2D([0], [0], marker='o', color='w', markerfacecolor='white',
                                 markeredgecolor='black', markersize=8, label='Mean', linewidth=0)
            handles.append(mean_marker)
            ax3.legend(handles=handles, loc='best', frameon=True, facecolor='white', edgecolor='gray')
            ax3.grid(True, alpha=0.3, linestyle='--')

        # Panel 4: Component size donut
        ax4 = axes[1, 1]
        ax4.set_facecolor('white')
        sizes = [p['size'] for p in profiles.values()]
        labels_pie = list(profiles.keys())
        wedges, texts, autotexts = ax4.pie(
            sizes, labels=labels_pie, colors=colors, autopct='%1.1f%%',
            startangle=90, pctdistance=0.75, explode=[0.02] * len(sizes),
            wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2)
        )
        for autotext in autotexts:
            autotext.set_fontsize(10)
            autotext.set_fontweight('bold')
        ax4.text(0, 0, f'n={sum(sizes)}', ha='center', va='center', fontsize=14, fontweight='bold')
        ax4.set_title('Component Distribution', fontsize=13, fontweight='bold')

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        buf.seek(0)
        plot = f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"

        # ── 12. Response ───────────────────────────────────────────────────
        return _to_native({
            'results': {
                'optimal_k': {
                    'k_range': k_range,
                    'bic_scores': bic_scores,
                    'aic_scores': aic_scores,
                    'silhouette_scores': silhouette_scores,
                    'recommended_k': recommended_k,
                },
                'clustering_summary': {
                    'n_components': n_components,
                    'covariance_type': covariance_type,
                    'scaler': scaler_type,
                    'means': gmm.means_.tolist(),
                    'weights': gmm.weights_.tolist(),
                    'labels': labels.tolist(),
                    'converged': gmm.converged_,
                    'n_iter': gmm.n_iter_,
                    'n_samples': n_samples,
                },
                'profiles': profiles,
                'soft_diagnostics': soft_diagnostics,
                'feature_drivers': feature_drivers,
                'final_metrics': {
                    'silhouette': silhouette,
                    'davies_bouldin': davies_bouldin,
                    'calinski_harabasz': calinski,
                    'bic': bic,
                    'aic': aic,
                    'log_likelihood': log_likelihood,
                    'note': 'Silhouette: higher better. Davies-Bouldin: lower better. BIC/AIC: lower better.',
                },
                'interpretations': {
                    'overall_quality': overall_quality,
                    'cluster_profiles': cluster_profiles_text,
                    'cluster_distribution': dist_desc,
                },
                'warnings': {
                    'input': input_warnings,
                    'quality': quality_warnings,
                },
            },
            'plot': plot,
        })

    except ValueError as e:
        logger.error(f"GMM validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"GMM unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
