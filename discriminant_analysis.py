import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from scipy import stats
import io
import base64
import warnings

warnings.filterwarnings('ignore')

sns.set_theme(style="whitegrid")

def _to_native_type(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj

def clean_json(obj):
    if isinstance(obj, dict):
        return {k: clean_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json(v) for v in obj]
    else:
        return _to_native_type(obj)

def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def _generate_interpretation(results, group_var, n_obs, n_predictors):
    """Generate detailed interpretation for Discriminant Analysis results in APA format."""
    
    interpretation_parts = []
    
    wilks_lambda = results['wilks_lambda']['lambda']
    f_stat = results['wilks_lambda']['F']
    df1 = results['wilks_lambda']['df1']
    df2 = results['wilks_lambda']['df2']
    p_value = results['wilks_lambda']['p_value']
    accuracy = results['classification_metrics']['accuracy']
    n_groups = len(results['meta']['groups'])
    canonical_corr = results['canonical_correlations'][0] if results['canonical_correlations'] else 0
    
    is_significant = p_value < 0.05
    
    eta_squared = 1 - wilks_lambda
    if eta_squared >= 0.14:
        effect_size = "large"
    elif eta_squared >= 0.06:
        effect_size = "medium"
    elif eta_squared >= 0.01:
        effect_size = "small"
    else:
        effect_size = "negligible"
    
    # --- Overall Assessment (APA Format) ---
    interpretation_parts.append("**Overall Assessment**")
    
    interpretation_parts.append(
        f"→ A linear discriminant analysis was conducted to predict {group_var} membership "
        f"from {n_predictors} predictor variable(s) across {n_groups} groups (N = {n_obs})."
    )
    
    if is_significant:
        p_str = "p < .001" if p_value < 0.001 else f"p = {p_value:.3f}"
        interpretation_parts.append(
            f"→ The discriminant function was statistically significant, Λ = {wilks_lambda:.3f}, "
            f"F({df1:.0f}, {df2:.0f}) = {f_stat:.2f}, {p_str}, representing a {effect_size} effect size (η² = {eta_squared:.3f})."
        )
    else:
        interpretation_parts.append(
            f"→ The discriminant function was not statistically significant, Λ = {wilks_lambda:.3f}, "
            f"F({df1:.0f}, {df2:.0f}) = {f_stat:.2f}, p = {p_value:.3f}."
        )
    
    if canonical_corr > 0:
        interpretation_parts.append(
            f"→ The canonical correlation of {canonical_corr:.3f} indicates that "
            f"{(canonical_corr**2)*100:.1f}% of variance in discriminant scores is explained by group membership."
        )
    
    if accuracy >= 0.90:
        acc_desc = "excellent"
    elif accuracy >= 0.75:
        acc_desc = "good"
    elif accuracy >= 0.60:
        acc_desc = "fair"
    else:
        acc_desc = "poor"
    
    interpretation_parts.append(
        f"→ Overall classification accuracy was {accuracy*100:.1f}%, indicating {acc_desc} predictive performance."
    )
    
    # --- Statistical Insights ---
    interpretation_parts.append("")
    interpretation_parts.append("**Statistical Insights**")
    
    interpretation_parts.append(
        f"→ Wilks' Lambda (Λ = {wilks_lambda:.3f}) represents the proportion of total variance "
        f"in discriminant scores not explained by group differences."
    )
    
    if 'standardized_coeffs' in results and results['standardized_coeffs']:
        coeffs = np.array(results['standardized_coeffs'])
        if coeffs.size > 0:
            predictor_vars = results['meta']['predictor_vars']
            abs_coeffs = np.abs(coeffs[:, 0]) if coeffs.ndim > 1 else np.abs(coeffs)
            top_idx = np.argsort(abs_coeffs)[::-1][:3]
            
            interpretation_parts.append("→ Most influential predictors (by standardized coefficient magnitude):")
            for idx in top_idx:
                if idx < len(predictor_vars):
                    coef_val = coeffs[idx, 0] if coeffs.ndim > 1 else coeffs[idx]
                    direction = "positive" if coef_val > 0 else "negative"
                    interpretation_parts.append(
                        f"  • {predictor_vars[idx]}: {coef_val:.4f} ({direction} contribution)"
                    )
    
    precision = results['classification_metrics']['precision']
    recall = results['classification_metrics']['recall']
    f1 = results['classification_metrics']['f1_score']
    
    interpretation_parts.append(
        f"→ Classification metrics: Precision = {precision*100:.1f}%, Recall = {recall*100:.1f}%, F1-Score = {f1*100:.1f}%"
    )
    
    obs_per_group = n_obs / n_groups
    if obs_per_group < 20:
        interpretation_parts.append(
            f"→ Warning: Average {obs_per_group:.0f} observations per group is below recommended minimum of 20."
        )
    else:
        interpretation_parts.append(
            f"→ Sample size adequate with {obs_per_group:.0f} observations per group on average."
        )
    
    # --- Recommendations ---
    interpretation_parts.append("")
    interpretation_parts.append("**Recommendations**")
    
    if not is_significant:
        interpretation_parts.append(
            "→ The model is not statistically significant. Consider adding more discriminating predictors or increasing sample size."
        )
    elif accuracy < 0.70:
        interpretation_parts.append(
            "→ Classification accuracy is modest. Consider feature engineering or trying quadratic discriminant analysis (QDA)."
        )
    else:
        interpretation_parts.append(
            "→ The model shows good discrimination. Validate with cross-validation or holdout sample."
        )
    
    if 'box_m_test' in results and results['box_m_test'] and results['box_m_test'].get('statistic'):
        if not results['box_m_test'].get('homogeneous', True):
            interpretation_parts.append(
                "→ Box's M test indicates unequal covariance matrices. Consider using QDA instead of LDA."
            )
    
    interpretation_parts.append(
        "→ Examine the scatter plot to visually assess group separation in discriminant space."
    )
    
    interpretation_parts.append(
        "→ Report standardized coefficients and structure matrix for variable interpretation."
    )
    
    return "\n".join(interpretation_parts)

def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get('data')
        group_var = payload.get('groupVar')
        predictor_vars = payload.get('predictorVars')

        if not all([data, group_var, predictor_vars]):
            raise ValueError("Missing data, groupVar, or predictorVars")

        df = pd.DataFrame(data)
        
        # Clean data
        all_vars = [group_var] + predictor_vars
        df_clean = df[all_vars].dropna()
        
        if len(df_clean) < 10:
            raise ValueError("Not enough valid observations after removing missing values")
        
        X = df_clean[predictor_vars].values.astype(float)
        
        le = LabelEncoder()
        y = le.fit_transform(df_clean[group_var])
        group_names = le.classes_.tolist()
        n_groups = len(group_names)
        
        if n_groups < 2:
            raise ValueError("Need at least 2 groups for discriminant analysis")
        
        n_components = min(n_groups - 1, len(predictor_vars))
        
        # Fit LDA
        lda = LinearDiscriminantAnalysis(n_components=n_components, store_covariance=True)
        lda.fit(X, y)
        
        y_pred = lda.predict(X)
        X_lda = lda.transform(X)
        
        # Classification metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(y, y_pred)
        
        # Wilks' Lambda
        n = len(y)
        p = len(predictor_vars)
        k = n_groups
        
        eigenvalues = lda.explained_variance_ratio_ * np.sum(lda.explained_variance_ratio_) if hasattr(lda, 'explained_variance_ratio_') else []
        
        # Calculate Wilks' Lambda from eigenvalues
        if len(lda.explained_variance_ratio_) > 0:
            lambda_vals = 1 / (1 + lda.explained_variance_ratio_ * (n - k) / (k - 1))
            wilks_lambda = np.prod(lambda_vals)
        else:
            wilks_lambda = 1.0
        
        # F approximation for Wilks' Lambda
        df1 = p * (k - 1)
        df2 = n - k - p + 1
        
        if wilks_lambda < 1 and df2 > 0:
            f_stat = ((1 - wilks_lambda) / wilks_lambda) * (df2 / df1)
            p_value = 1 - stats.f.cdf(f_stat, df1, df2)
        else:
            f_stat = 0
            p_value = 1.0
        
        # Canonical correlations
        canonical_corrs = np.sqrt(lda.explained_variance_ratio_).tolist() if hasattr(lda, 'explained_variance_ratio_') else []
        
        # Standardized coefficients
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        lda_scaled = LinearDiscriminantAnalysis(n_components=n_components)
        lda_scaled.fit(X_scaled, y)
        std_coeffs = lda_scaled.scalings_.tolist()
        
        # Structure matrix (correlations between predictors and discriminant functions)
        structure_matrix = []
        for i in range(X.shape[1]):
            corrs = []
            for j in range(X_lda.shape[1]):
                corr = np.corrcoef(X[:, i], X_lda[:, j])[0, 1]
                corrs.append(corr if not np.isnan(corr) else 0)
            structure_matrix.append(corrs)
        
        # Group centroids
        centroids = []
        for g in range(n_groups):
            centroid = X_lda[y == g].mean(axis=0).tolist()
            centroids.append(centroid)
        
        # Eigenvalue details
        eigenvalue_details = []
        cumulative = 0
        for i, ev in enumerate(lda.explained_variance_ratio_):
            cumulative += ev
            eigenvalue_details.append({
                'function': f'LD{i+1}',
                'eigenvalue': float(ev * (n - k) / (k - 1)),
                'variance_explained': float(ev),
                'cumulative_variance': float(cumulative),
                'canonical_correlation': float(np.sqrt(ev)) if ev > 0 else 0
            })
        
        # Group statistics
        group_stats = {}
        for i, gname in enumerate(group_names):
            mask = y == i
            group_stats[str(gname)] = {
                'n': int(np.sum(mask)),
                'means': X[mask].mean(axis=0).tolist(),
                'stds': X[mask].std(axis=0).tolist(),
                'predictor_names': predictor_vars
            }
        
        # Prior probabilities
        priors = lda.priors_.tolist()
        
        # Classification function coefficients
        class_func_coeffs = {}
        class_func_intercepts = {}
        for i, gname in enumerate(group_names):
            class_func_coeffs[str(gname)] = lda.coef_[i].tolist() if len(lda.coef_) > i else lda.coef_[0].tolist()
            class_func_intercepts[str(gname)] = float(lda.intercept_[i]) if len(lda.intercept_) > i else float(lda.intercept_[0])
        
        # Box's M test (simplified approximation)
        try:
            cov_matrices = []
            for i in range(n_groups):
                cov_matrices.append(np.cov(X[y == i].T))
            
            pooled_cov = np.zeros_like(cov_matrices[0])
            ns = [np.sum(y == i) for i in range(n_groups)]
            for i, cov in enumerate(cov_matrices):
                pooled_cov += (ns[i] - 1) * cov
            pooled_cov /= (n - n_groups)
            
            M = 0
            for i, cov in enumerate(cov_matrices):
                if ns[i] > 1:
                    det_ratio = np.linalg.det(cov) / np.linalg.det(pooled_cov)
                    if det_ratio > 0:
                        M += (ns[i] - 1) * np.log(det_ratio)
            
            box_df = (n_groups - 1) * p * (p + 1) / 2
            box_p = 1 - stats.chi2.cdf(abs(M), box_df) if box_df > 0 else 1
            
            box_m_test = {
                'statistic': float(abs(M)),
                'df': float(box_df),
                'p_value': float(box_p),
                'homogeneous': box_p > 0.05
            }
        except:
            box_m_test = {'statistic': None, 'df': None, 'p_value': None, 'homogeneous': None}
        
        results = {
            'meta': {
                'groups': [str(g) for g in group_names],
                'n_components': n_components,
                'predictor_vars': predictor_vars
            },
            'classification_metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': conf_matrix.tolist(),
                'true_labels': y.tolist(),
                'predicted_labels': y_pred.tolist()
            },
            'eigenvalues': lda.explained_variance_ratio_.tolist(),
            'eigenvalue_details': eigenvalue_details,
            'canonical_correlations': canonical_corrs,
            'wilks_lambda': {
                'lambda': wilks_lambda,
                'F': f_stat,
                'df1': df1,
                'df2': df2,
                'p_value': p_value
            },
            'standardized_coeffs': std_coeffs,
            'structure_matrix': structure_matrix,
            'group_centroids': centroids,
            'lda_transformed_data': X_lda.tolist(),
            'true_labels_full': y.tolist(),
            'group_stats': group_stats,
            'priors': priors,
            'classification_function_coeffs': class_func_coeffs,
            'classification_function_intercepts': class_func_intercepts,
            'box_m_test': box_m_test
        }
        
        # Generate interpretation
        interpretation = _generate_interpretation(
            results=results,
            group_var=group_var,
            n_obs=len(df_clean),
            n_predictors=len(predictor_vars)
        )
        results['interpretation'] = interpretation
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Scatter plot of first two discriminant functions
        ax1 = axes[0, 0]
        colors = plt.cm.Set1(np.linspace(0, 1, n_groups))
        for i, gname in enumerate(group_names):
            mask = y == i
            if n_components >= 2:
                ax1.scatter(X_lda[mask, 0], X_lda[mask, 1], c=[colors[i]], label=str(gname), alpha=0.6, s=50)
            else:
                ax1.scatter(X_lda[mask, 0], np.zeros(np.sum(mask)), c=[colors[i]], label=str(gname), alpha=0.6, s=50)
        
        ax1.set_xlabel('LD1', fontsize=12)
        ax1.set_ylabel('LD2' if n_components >= 2 else '', fontsize=12)
        ax1.set_title('Discriminant Function Scores', fontsize=14, fontweight='bold')
        ax1.legend()
        
        # 2. Group centroids
        ax2 = axes[0, 1]
        centroid_df = pd.DataFrame(centroids, columns=[f'LD{i+1}' for i in range(n_components)], index=group_names)
        centroid_df.plot(kind='bar', ax=ax2, colormap='Set2')
        ax2.set_title('Group Centroids', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Group', fontsize=12)
        ax2.set_ylabel('Centroid Value', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend(title='Function')
        
        # 3. Confusion matrix
        ax3 = axes[1, 0]
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax3,
                    xticklabels=group_names, yticklabels=group_names)
        ax3.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Predicted', fontsize=12)
        ax3.set_ylabel('Actual', fontsize=12)
        
        # 4. Structure matrix heatmap
        ax4 = axes[1, 1]
        struct_df = pd.DataFrame(structure_matrix, 
                                  index=predictor_vars, 
                                  columns=[f'LD{i+1}' for i in range(n_components)])
        sns.heatmap(struct_df, annot=True, fmt='.3f', cmap='RdBu_r', center=0, ax=ax4)
        ax4.set_title('Structure Matrix (Loadings)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plot_image = fig_to_base64(fig)
        
        response = {
            'results': clean_json(results),
            'plots': {
                'lda_analysis': plot_image
            }
        }
        
        print(json.dumps(response))

    except Exception as e:
        import traceback
        print(json.dumps({"error": str(e), "traceback": traceback.format_exc()}), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()

    