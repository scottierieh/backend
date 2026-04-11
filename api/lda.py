from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, List
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedKFold
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from scipy.stats import f as f_dist, chi2
import io, base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

router = APIRouter()

class LDARequest(BaseModel):
    data: list[dict[str, Any]] = Field(...)
    groupVar: str = Field(...)
    predictorVars: List[str] = Field(...)

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
    return obj

def safe_float(val, default=0.0):
    """Safely convert to float, handling None, NaN, Inf"""
    try:
        if val is None or pd.isna(val) or np.isinf(val):
            return default
        return float(val)
    except:
        return default

def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

@router.post("/lda")
def lda_analysis(req: LDARequest):
    try:
        df = pd.DataFrame(req.data)
        group_var = req.groupVar
        predictor_vars = req.predictorVars

        # Validate inputs
        if group_var not in df.columns:
            raise ValueError(f"Group variable '{group_var}' not found in data")
        
        missing_predictors = [p for p in predictor_vars if p not in df.columns]
        if missing_predictors:
            raise ValueError(f"Predictor variables not found: {missing_predictors}")

        # Prepare data
        all_vars = [group_var] + predictor_vars
        df_clean = df[all_vars].dropna().copy()
        
        if len(df_clean) < 10:
            raise ValueError(f"Need at least 10 observations, got {len(df_clean)}")
        
        # Encode group labels
        le = LabelEncoder()
        y_encoded = le.fit_transform(df_clean[group_var])
        groups = [str(g) for g in le.classes_.tolist()]
        n_groups = len(groups)
        
        if n_groups < 2:
            raise ValueError("Need at least 2 groups for LDA")
        
        # Prepare features
        X = df_clean[predictor_vars].astype(float).values
        
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determine number of components
        n_components = min(n_groups - 1, len(predictor_vars))
        if n_components < 1:
            n_components = 1

        # Fit LDA
        lda = LinearDiscriminantAnalysis(n_components=n_components, store_covariance=True)
        lda.fit(X_scaled, y_encoded)
        X_lda = lda.transform(X_scaled)
        
        # Cross-validation with proper fold handling
        samples_per_group = np.bincount(y_encoded)
        min_samples = int(np.min(samples_per_group))
        
        # Determine safe number of CV folds
        cv_folds = min(3, min_samples)
        
        if cv_folds >= 2:
            # Use stratified k-fold for balanced splits
            skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            accuracy_scores = cross_val_score(lda, X_scaled, y_encoded, cv=skf)
            accuracy = float(np.mean(accuracy_scores))
            y_pred = cross_val_predict(lda, X_scaled, y_encoded, cv=skf)
        else:
            # Not enough samples for CV, use training predictions
            y_pred = lda.predict(X_scaled)
            accuracy = float(np.mean(y_pred == y_encoded))
            accuracy_scores = [accuracy]
        
        # Classification metrics
        conf_matrix = confusion_matrix(y_encoded, y_pred).tolist()
        
        avg_method = 'binary' if n_groups == 2 else 'weighted'
        precision = float(precision_score(y_encoded, y_pred, average=avg_method, zero_division=0))
        recall = float(recall_score(y_encoded, y_pred, average=avg_method, zero_division=0))
        f1 = float(f1_score(y_encoded, y_pred, average=avg_method, zero_division=0))
        
        class_report = classification_report(
            y_encoded, y_pred, 
            target_names=groups, 
            output_dict=True, 
            zero_division=0
        )
        
        # Eigenvalues and variance explained
        eigenvalues = lda.explained_variance_ratio_.tolist()
        canonical_correlations = [float(np.sqrt(ev)) for ev in lda.explained_variance_ratio_]

        # Wilks' Lambda calculation
        n = len(X_scaled)
        p = len(predictor_vars)
        g = n_groups
        
        try:
            # Wilks' Lambda using eigenvalues
            wilks_lambda = 1.0
            for ev in lda.explained_variance_ratio_:
                wilks_lambda *= 1 / (1 + ev)
            wilks_lambda = float(wilks_lambda)
        except:
            wilks_lambda = 0.0
        
        # F approximation for Wilks' Lambda
        if wilks_lambda > 0 and wilks_lambda < 1 and p > 0 and g > 1:
            try:
                df1 = p * (g - 1)
                df2 = n - p - g
                if df2 > 0:
                    f_approx = ((1 - wilks_lambda) / wilks_lambda) * (df2 / df1)
                    p_value_f = float(1 - f_dist.cdf(f_approx, df1, df2))
                else:
                    f_approx = 0.0
                    p_value_f = 1.0
            except:
                f_approx = 0.0
                df1, df2 = 0, 0
                p_value_f = 1.0
        else:
            f_approx = 0.0
            df1, df2 = p * (g - 1), n - p - g
            p_value_f = 1.0

        # Standardized coefficients (scalings)
        standardized_coeffs = lda.scalings_.tolist() if hasattr(lda, 'scalings_') else []
        
        # Structure matrix - correlation between variables and discriminant functions
        try:
            structure_matrix = []
            for i in range(len(predictor_vars)):
                row = []
                for j in range(n_components):
                    corr = np.corrcoef(X_scaled[:, i], X_lda[:, j])[0, 1]
                    row.append(safe_float(corr))
                structure_matrix.append(row)
        except:
            structure_matrix = standardized_coeffs

        # Group statistics
        group_stats = {}
        for i, group in enumerate(groups):
            mask = y_encoded == i
            group_data = X_scaled[mask]
            group_stats[group] = {
                'n': int(mask.sum()),
                'means': [safe_float(m) for m in np.mean(group_data, axis=0)],
                'stds': [safe_float(s) for s in np.std(group_data, axis=0, ddof=1)],
                'predictor_names': predictor_vars
            }

        # Box's M Test for homogeneity of covariance matrices
        try:
            S_pooled = lda.covariance_
            group_covs, group_ns = [], []
            for i in range(n_groups):
                mask = y_encoded == i
                gdata = X_scaled[mask]
                if len(gdata) > p:  # Need more samples than variables
                    group_covs.append(np.cov(gdata.T))
                    group_ns.append(len(gdata))
            
            if len(group_covs) >= 2 and all(np.linalg.det(cov) > 0 for cov in group_covs):
                n_total = sum(group_ns)
                det_pooled = np.linalg.det(S_pooled)
                if det_pooled > 0:
                    M = sum(
                        (ni - 1) * (np.log(det_pooled) - np.log(np.linalg.det(cov))) 
                        for cov, ni in zip(group_covs, group_ns) 
                        if np.linalg.det(cov) > 0
                    )
                    sum_inv = sum(1/(ni - 1) for ni in group_ns)
                    C = (2*p**2 + 3*p - 1) / (6*(p+1)*(g-1)) * (sum_inv - 1/(n_total - g))
                    chi2_stat = M * (1 - C)
                    df_box = p * (p + 1) * (g - 1) / 2
                    p_value_box = float(1 - chi2.cdf(chi2_stat, df_box))
                    box_m_test = {
                        'statistic': safe_float(chi2_stat), 
                        'df': safe_float(df_box), 
                        'p_value': p_value_box, 
                        'homogeneous': p_value_box > 0.05
                    }
                else:
                    box_m_test = {'statistic': None, 'df': None, 'p_value': None, 'homogeneous': None}
            else:
                box_m_test = {'statistic': None, 'df': None, 'p_value': None, 'homogeneous': None}
        except:
            box_m_test = {'statistic': None, 'df': None, 'p_value': None, 'homogeneous': None}

        # Classification function coefficients
        priors = lda.priors_.tolist()
        classification_coeffs = {}
        classification_intercepts = {}
        try:
            cov_inv = np.linalg.inv(lda.covariance_)
            for i, group in enumerate(groups):
                group_mean = lda.means_[i]
                coef = cov_inv @ group_mean
                intercept = -0.5 * group_mean @ cov_inv @ group_mean + np.log(lda.priors_[i])
                classification_coeffs[group] = [safe_float(c) for c in coef]
                classification_intercepts[group] = safe_float(intercept)
        except:
            pass

        # Eigenvalue details
        eigenvalue_details = []
        cumulative = 0
        for i, eig in enumerate(eigenvalues):
            cumulative += eig
            eigenvalue_details.append({
                'function': f'LD{i+1}',
                'eigenvalue': safe_float(eig),
                'variance_explained': safe_float(eig),
                'cumulative_variance': safe_float(cumulative),
                'canonical_correlation': safe_float(canonical_correlations[i]) if i < len(canonical_correlations) else 0
            })

        # Group means in original space
        group_means_original = {
            groups[i]: [safe_float(m) for m in lda.means_[i]] 
            for i in range(n_groups)
        }
        
        # Group centroids in LDA space
        group_centroids = lda.transform(lda.means_).tolist()

        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        line_color = '#C44E52'
        
        # 1. LDA Scatterplot
        plot_df = pd.DataFrame(X_lda, columns=[f'LD{i+1}' for i in range(n_components)])
        plot_df['group'] = le.inverse_transform(y_encoded)
        
        if n_components > 1:
            sns.scatterplot(
                data=plot_df, x='LD1', y='LD2', hue='group', 
                ax=axes[0, 0], palette='viridis', s=50, alpha=0.7
            )
            centroids = lda.transform(lda.means_)
            for i in range(n_groups):
                axes[0, 0].scatter(
                    centroids[i, 0], centroids[i, 1], 
                    marker='X', s=150, color=line_color, 
                    edgecolors='black', linewidths=1.5
                )
            axes[0, 0].set_ylabel('LD2')
        else:
            # 1D case - add jitter for visibility
            jitter = np.random.normal(0, 0.1, len(plot_df))
            sns.scatterplot(
                data=plot_df, x='LD1', y=jitter, hue='group', 
                ax=axes[0, 0], palette='viridis', s=50, alpha=0.7
            )
            axes[0, 0].set_ylabel('')
            axes[0, 0].set_yticks([])
        
        axes[0, 0].set_title('Discriminant Function Scatterplot', fontweight='bold')
        axes[0, 0].set_xlabel('LD1')
        axes[0, 0].legend()
        
        # 2. Confusion Matrix
        sns.heatmap(
            conf_matrix, annot=True, fmt='d', cmap='Blues', 
            ax=axes[0, 1], xticklabels=groups, yticklabels=groups
        )
        axes[0, 1].set_title('Confusion Matrix', fontweight='bold')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('True')
        
        # 3. Classification Metrics
        metrics_data = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1': f1}
        bars = axes[1, 0].bar(
            metrics_data.keys(), metrics_data.values(), 
            color='#5B9BD5', alpha=0.7, edgecolor='black'
        )
        axes[1, 0].set_ylim([0, 1])
        axes[1, 0].axhline(y=0.8, color=line_color, linestyle='--', lw=2, alpha=0.7)
        axes[1, 0].set_title('Classification Metrics', fontweight='bold')
        for bar in bars:
            axes[1, 0].text(
                bar.get_x() + bar.get_width()/2., bar.get_height(), 
                f'{bar.get_height():.3f}', ha='center', va='bottom'
            )
        
        # 4. Eigenvalues / Variance Explained
        if len(eigenvalues) > 0:
            x_eig = np.arange(1, len(eigenvalues) + 1)
            axes[1, 1].bar(x_eig, eigenvalues, color='#5B9BD5', alpha=0.7, edgecolor='black')
            axes[1, 1].plot(x_eig, eigenvalues, 'o-', color=line_color, lw=2, ms=8)
            axes[1, 1].set_xticks(x_eig)
            axes[1, 1].set_xlabel('Function')
            axes[1, 1].set_ylabel('Variance Ratio')
        else:
            axes[1, 1].text(0.5, 0.5, 'No eigenvalues', ha='center', va='center')
        axes[1, 1].set_title('Variance Explained by Function', fontweight='bold')
        
        plt.tight_layout()
        lda_plot = fig_to_base64(fig)

        # Correlation heatmap
        try:
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            corr = df_clean[predictor_vars].corr()
            sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax2, fmt='.2f')
            ax2.set_title('Predictor Correlation Heatmap', fontweight='bold')
            plt.tight_layout()
            heatmap = fig_to_base64(fig2)
        except:
            heatmap = None

        # Build response
        results = {
            'meta': {
                'groups': groups,
                'n_components': n_components,
                'predictor_vars': predictor_vars,
                'n_observations': len(df_clean)
            },
            'classification_metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'confusion_matrix': conf_matrix,
                'class_report': _to_native(class_report),
                'cv_folds': cv_folds,
                'cv_scores': [safe_float(s) for s in accuracy_scores]
            },
            'eigenvalues': eigenvalues,
            'eigenvalue_details': eigenvalue_details,
            'canonical_correlations': canonical_correlations,
            'wilks_lambda': {
                'lambda': safe_float(wilks_lambda),
                'F': safe_float(f_approx),
                'df1': df1,
                'df2': df2,
                'p_value': p_value_f
            },
            'standardized_coeffs': standardized_coeffs,
            'structure_matrix': structure_matrix,
            'group_centroids': group_centroids,
            'group_means_original': group_means_original,
            'group_stats': group_stats,
            'priors': priors,
            'classification_function_coeffs': classification_coeffs,
            'classification_function_intercepts': classification_intercepts,
            'pooled_covariance_matrix': lda.covariance_.tolist() if hasattr(lda, 'covariance_') else [],
            'box_m_test': box_m_test,
            'lda_transformed_data': X_lda.tolist(),
            'true_labels': y_encoded.tolist()
        }

        return _to_native({
            'results': results,
            'plots': {
                'lda_analysis': lda_plot,
                'heatmap': heatmap
            }
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
