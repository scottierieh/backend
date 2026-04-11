from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                              roc_curve, auc, precision_score, recall_score, f1_score)
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy import stats
import io, base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

router = APIRouter()

class LogisticRequest(BaseModel):
    data: list[dict[str, Any]] = Field(...)
    dependentVar: str = Field(...)
    independentVars: List[str] = Field(...)
    testSize: float = Field(default=0.3)
    standardize: bool = Field(default=False)

def _to_native(obj):
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


def youden_threshold(fpr, tpr, thresholds):
    """Optimal threshold via Youden Index = max(Sensitivity + Specificity - 1)."""
    j_scores = tpr - fpr
    idx = int(np.argmax(j_scores))
    return float(thresholds[idx]), float(tpr[idx]), float(1 - fpr[idx])


def hosmer_lemeshow_test(y_true, y_prob, n_groups=10):
    """
    Hosmer-Lemeshow Goodness of Fit Test
    Returns: statistic, p_value, table
    """
    try:
        data = pd.DataFrame({'y': y_true, 'prob': y_prob})
        data['decile'] = pd.qcut(data['prob'], q=n_groups, labels=False, duplicates='drop')
        
        observed_1 = data.groupby('decile')['y'].sum()
        observed_0 = data.groupby('decile')['y'].count() - observed_1
        expected_1 = data.groupby('decile')['prob'].sum()
        expected_0 = data.groupby('decile')['prob'].count() - expected_1
        
        # Chi-square statistic
        hl_stat = 0
        for d in observed_1.index:
            if expected_1[d] > 0:
                hl_stat += (observed_1[d] - expected_1[d])**2 / expected_1[d]
            if expected_0[d] > 0:
                hl_stat += (observed_0[d] - expected_0[d])**2 / expected_0[d]
        
        # Degrees of freedom = n_groups - 2
        from scipy import stats
        df = len(observed_1) - 2
        p_value = 1 - stats.chi2.cdf(hl_stat, df)
        
        return {
            'statistic': float(hl_stat),
            'p_value': float(p_value),
            'df': int(df),
            'n_groups': len(observed_1)
        }
    except Exception as e:
        return None


def calculate_vif(X):
    """
    Calculate Variance Inflation Factor for each predictor
    """
    try:
        X_with_const = sm.add_constant(X)
        vif_data = {}
        for i, col in enumerate(X.columns):
            vif_data[col] = float(variance_inflation_factor(X_with_const.values, i + 1))
        return vif_data
    except Exception:
        return {}


def influence_stats(model, X_const):
    """Cook's distance, leverage (hat values), standardised Pearson residuals."""
    try:
        influence = model.get_influence()
        hat       = influence.hat_matrix_diag
        resid_std = influence.resid_studentized
        # Cook's D approximation for logistic: D_i = h_i * r_i^2 / (p * (1-h_i))
        p = X_const.shape[1]
        cooks_d = (resid_std ** 2) * hat / (p * np.maximum(1 - hat, 1e-12))
        n       = len(cooks_d)
        cutoff  = 4.0 / n
        flagged = [int(i) for i in np.where(cooks_d > cutoff)[0]]
        return {
            "n_influential": len(flagged),
            "cutoff": round(cutoff, 4),
            "max_cooks_d": round(float(cooks_d.max()), 4),
            "mean_leverage": round(float(hat.mean()), 4),
            "flagged_indices": flagged[:20],
        }
    except Exception:
        return None


def compute_marginal_effects(model, X_full_const, features):
    """
    Average Marginal Effects (AME) for logistic regression.
    AME_j = mean over all obs of [ p_i * (1 - p_i) * beta_j ]
    For binary dummies: delta method (P(x=1) - P(x=0)) averaged.
    Returns dy/dx as probability change per unit increase.
    """
    try:
        p_hat   = model.predict(X_full_const)           # (n,) predicted probs
        weights = p_hat * (1 - p_hat)                   # logistic kernel
        params  = model.params                           # includes 'const'
        bse     = model.bse
        conf    = model.conf_int()
        n       = len(p_hat)

        ame = {}
        for feat in features:
            if feat not in params.index:
                continue
            beta = float(params[feat])
            # AME = mean(p*(1-p)) * beta
            ame_val = float(weights.mean() * beta)

            # Delta-method SE: se_AME ≈ mean(p*(1-p)) * se_beta
            se_ame  = float(weights.mean() * float(bse[feat]))
            z_ame   = ame_val / se_ame if se_ame > 0 else 0.0

            from scipy.stats import norm as _norm
            p_ame   = float(2 * (1 - _norm.cdf(abs(z_ame))))
            ci_lo   = ame_val - 1.96 * se_ame
            ci_hi   = ame_val + 1.96 * se_ame

            # MEM (Marginal Effect at Mean): evaluate at X̄
            x_mean  = X_full_const.mean()
            lp_mean = float(np.dot(x_mean.values, params.values))
            p_mean  = float(1 / (1 + np.exp(-lp_mean)))
            mem_val = float(p_mean * (1 - p_mean) * beta)

            ame[feat] = {
                "ame":       round(ame_val, 6),   # dy/dx averaged over all obs
                "mem":       round(mem_val, 6),   # dy/dx at sample means
                "se":        round(se_ame, 6),
                "z":         round(z_ame, 4),
                "p_value":   round(p_ame, 4),
                "ci_lower":  round(ci_lo, 6),
                "ci_upper":  round(ci_hi, 6),
                "pct_change": round(ame_val * 100, 4),  # as percentage points
                "significant": bool(p_ame < 0.05),
            }
        return ame
    except Exception as e:
        return {}


@router.post("/logistic-regression")
def logistic_regression(req: LogisticRequest):
    try:
        df = pd.DataFrame(req.data)
        dep_var = req.dependentVar
        indep_vars = req.independentVars
        
        all_vars = [dep_var] + indep_vars
        clean = df[all_vars].dropna()
        n_dropped = len(df) - len(clean)
        
        if clean.empty:
            raise ValueError("No valid data after removing missing values")
        
        le = LabelEncoder()
        y = le.fit_transform(clean[dep_var])
        classes = le.classes_.tolist()
        
        if len(classes) != 2:
            raise ValueError(f"Need exactly 2 classes, found {len(classes)}")
        
        X = pd.get_dummies(clean[indep_vars], drop_first=True, dtype=float)
        features = X.columns.tolist()
        
        if req.standardize:
            scaler = StandardScaler()
            X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=features, index=X.index)
        else:
            X_scaled = X
        
        # ── Full-sample statsmodels fit (statistical inference) ──────────
        X_full_const = sm.add_constant(X_scaled)
        model = sm.Logit(y, X_full_const).fit(disp=0)
        y_prob_full = model.predict(X_full_const)

        # ── Optimal threshold (Youden Index) on full-sample ROC ──────────
        fpr_full, tpr_full, thresh_full = roc_curve(y, y_prob_full)
        roc_auc_full = auc(fpr_full, tpr_full)
        opt_threshold, opt_sensitivity, opt_specificity = youden_threshold(fpr_full, tpr_full, thresh_full)
        y_pred_opt = (y_prob_full >= opt_threshold).astype(int)

        # ── Train/test split — ML-style held-out performance ─────────────
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=req.testSize, random_state=42, stratify=y
        )
        X_train_const = sm.add_constant(X_train)
        X_test_const  = sm.add_constant(X_test)
        model_split = sm.Logit(y_train, X_train_const).fit(disp=0)
        y_prob_test = model_split.predict(X_test_const)
        fpr, tpr, thresholds = roc_curve(y_test, y_prob_test)
        roc_auc = auc(fpr, tpr)
        opt_thr_test, _, _ = youden_threshold(fpr, tpr, thresholds)
        y_pred_test = (y_prob_test >= opt_thr_test).astype(int)

        acc = accuracy_score(y_test, y_pred_test)
        cm  = confusion_matrix(y_test, y_pred_test)
        report = classification_report(y_test, y_pred_test, target_names=[str(c) for c in classes], output_dict=True)
        
        # ── Detailed metrics at optimal threshold ────────────────────────
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        sensitivity   = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        specificity   = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        ppv           = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        npv           = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
        f1            = float(2 * ppv * sensitivity / (ppv + sensitivity)) if (ppv + sensitivity) > 0 else 0.0
        balanced_acc  = (sensitivity + specificity) / 2
        mcc_denom     = float(np.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)))
        mcc           = float((tp*tn - fp*fn) / mcc_denom) if mcc_denom > 0 else 0.0

        # ── Coefficients, odds ratios, standard errors, z-values ─────────
        coefficients = {}
        odds_ratios = {}
        odds_ratios_ci = {}
        p_values = {}
        std_errors = {}
        z_values = {}
        
        conf_int = model.conf_int()
        for i, feat in enumerate(model.params.index):
            if feat == 'const':
                continue
            coefficients[feat] = float(model.params[feat])
            odds_ratios[feat] = float(np.exp(model.params[feat]))
            p_values[feat] = float(model.pvalues[feat])
            std_errors[feat] = float(model.bse[feat])
            z_values[feat] = float(model.tvalues[feat])
            odds_ratios_ci[feat] = {
                '2.5%': float(np.exp(conf_int.loc[feat, 0])),
                '97.5%': float(np.exp(conf_int.loc[feat, 1]))
            }
        
        # Intercept info
        intercept = {
            'coefficient': float(model.params['const']),
            'std_error': float(model.bse['const']),
            'z_value': float(model.tvalues['const']),
            'p_value': float(model.pvalues['const'])
        }
        
        # Model summary
        model_summary = {
            'llf': float(model.llf),
            'llnull': float(model.llnull),
            'llr': float(model.llr),
            'llr_pvalue': float(model.llr_pvalue),
            'prsquared': float(model.prsquared),
            'df_model': float(model.df_model),
            'df_resid': float(model.df_resid),
            'aic': float(model.aic),
            'bic': float(model.bic)
        }
        
        # VIF (Variance Inflation Factor)
        vif = calculate_vif(X_scaled)
        
        # Hosmer-Lemeshow Test
        hl_test = hosmer_lemeshow_test(y, y_prob_full)

        # Influence statistics
        infl = influence_stats(model, X_full_const)

        # Marginal Effects (AME + MEM)
        marginal_effects = compute_marginal_effects(model, X_full_const, features)
        
        # Interpretation
        auc_desc = "excellent" if roc_auc >= 0.9 else "good" if roc_auc >= 0.8 else "fair" if roc_auc >= 0.7 else "poor"
        sig_vars = [f for f in coefficients if p_values[f] < 0.05]

        interp = "**Overall Analysis**\n"
        interp += f"→ Threshold: {opt_thr_test:.3f} (Youden Index, held-out test set)\n"
        interp += f"→ Accuracy: {acc*100:.1f}% | Balanced: {balanced_acc*100:.1f}% | AUC: {roc_auc:.3f} ({auc_desc})\n"
        interp += f"→ Sensitivity: {sensitivity:.3f} | Specificity: {specificity:.3f} | F1: {f1:.3f}\n"
        interp += f"→ Precision (PPV): {ppv:.3f} | NPV: {npv:.3f} | MCC: {mcc:.3f}\n"
        interp += f"→ Pseudo R²: {model.prsquared:.3f} | LLR p: {'<.001' if model.llr_pvalue < 0.001 else f'{model.llr_pvalue:.4f}'}\n"

        if hl_test:
            hl_sig = "good fit" if hl_test['p_value'] > 0.05 else "poor fit"
            interp += f"→ Hosmer-Lemeshow: χ²={hl_test['statistic']:.2f}, p={hl_test['p_value']:.4f} ({hl_sig})\n"

        interp += "\n**Key Insights**\n"
        if sig_vars:
            for v in sig_vars[:3]:
                or_val = odds_ratios[v]
                direction = "increases" if or_val > 1 else "decreases"
                pct = abs(or_val - 1) * 100
                interp += f"→ {v}: OR={or_val:.3f} ({direction} odds by {pct:.1f}%)\n"
        else:
            interp += "→ No significant predictors at α=0.05\n"

        if infl and infl['n_influential'] > 0:
            interp += f"→ {infl['n_influential']} influential observation(s) detected (Cook's D > {infl['cutoff']:.4f})\n"

        high_vif_vars = [v for v, val in vif.items() if val > 5]
        if high_vif_vars:
            interp += f"\n**Multicollinearity Warning**\n"
            interp += f"→ High VIF (>5): {', '.join(high_vif_vars[:3])}\n"

        interp += "\n**Recommendations**\n"
        if roc_auc < 0.7:
            interp += "→ Low AUC — consider adding more predictors\n"
        if model.prsquared < 0.1:
            interp += "→ Low pseudo R² — model has limited explanatory power\n"
        if hl_test and hl_test['p_value'] < 0.05:
            interp += "→ Hosmer-Lemeshow suggests poor fit — consider interaction terms or splines\n"
        if sensitivity < 0.7:
            interp += f"→ Low sensitivity ({sensitivity:.2f}) — model misses many positives; lower threshold or address class imbalance\n"
        interp += "→ Use calibration curve to assess probability reliability\n"
        if marginal_effects:
            top_me = sorted(marginal_effects.items(), key=lambda x: abs(x[1]['ame']), reverse=True)[:2]
            interp += "\n**Marginal Effects (AME)\n"
            for feat, me in top_me:
                direction = "increases" if me['ame'] > 0 else "decreases"
                interp += f"→ {feat}: +1 unit {direction} P(Y=1) by {me['pct_change']:+.2f} pp {'*' if me['significant'] else '(n.s.)'}\n"
        
        def _fig_to_b64(fig):
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close(fig)
            buf.seek(0)
            return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"

        # ── Plot 1: ROC + Confusion Matrix ───────────────────────────────
        fig1, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(fpr, tpr, color='#4C72B0', lw=2, label=f'Test AUC = {roc_auc:.3f}')
        axes[0].plot(fpr_full, tpr_full, color='#DD8452', lw=1.5, linestyle='--',
                     label=f'Full-sample AUC = {roc_auc_full:.3f}')
        axes[0].scatter([1 - opt_thr_test], [opt_sensitivity], marker='o', color='red',
                        s=80, zorder=5, label=f'Youden thr={opt_thr_test:.2f}')
        axes[0].plot([0, 1], [0, 1], 'gray', linestyle=':')
        axes[0].set_xlabel('False Positive Rate'); axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title('ROC Curve', fontweight='bold'); axes[0].legend(loc='lower right', fontsize=8)
        sns.heatmap(cm, annot=True, fmt='d', cmap='vlag', ax=axes[1],
                    xticklabels=classes, yticklabels=classes)
        axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')
        axes[1].set_title(f'Confusion Matrix (thr={opt_thr_test:.2f})', fontweight='bold')
        plot = _fig_to_b64(fig1)

        # ── Plot 2: Calibration curve ─────────────────────────────────────
        fig2, ax_cal = plt.subplots(figsize=(7, 6))
        try:
            from sklearn.calibration import calibration_curve
            prob_true, prob_pred = calibration_curve(y, y_prob_full, n_bins=10, strategy='quantile')
            ax_cal.plot(prob_pred, prob_true, marker='o', color='#4C72B0', lw=2, label='Model')
            ax_cal.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect calibration')
            ax_cal.set_xlabel('Mean Predicted Probability'); ax_cal.set_ylabel('Fraction of Positives')
            ax_cal.set_title('Calibration Curve (Reliability Diagram)', fontweight='bold')
            ax_cal.legend()
        except Exception as _ce:
            ax_cal.text(0.5, 0.5, f'Calibration unavailable: {_ce}',
                        ha='center', va='center', transform=ax_cal.transAxes, fontsize=9)
        plot_calibration = _fig_to_b64(fig2)

        # ── Plot 3: Coefficient forest plot ──────────────────────────────
        fig3, ax_forest = plt.subplots(figsize=(8, max(4, len(coefficients) * 0.5 + 2)))
        feat_names = list(coefficients.keys())
        or_vals  = [odds_ratios[f] for f in feat_names]
        ci_lo    = [odds_ratios_ci[f]['2.5%'] for f in feat_names]
        ci_hi    = [odds_ratios_ci[f]['97.5%'] for f in feat_names]
        colors   = ['#4C72B0' if p_values[f] < 0.05 else '#CCCCCC' for f in feat_names]
        y_pos    = range(len(feat_names))
        for i, (f, o, lo, hi, col) in enumerate(zip(feat_names, or_vals, ci_lo, ci_hi, colors)):
            ax_forest.plot([lo, hi], [i, i], color=col, lw=2)
            ax_forest.scatter(o, i, color=col, s=60, zorder=3)
        ax_forest.axvline(1.0, color='red', linestyle='--', lw=1)
        ax_forest.set_yticks(list(y_pos)); ax_forest.set_yticklabels(feat_names, fontsize=9)
        ax_forest.set_xlabel('Odds Ratio (95% CI)')
        ax_forest.set_title('Odds Ratio Forest Plot (blue = p<.05)', fontweight='bold')
        plot_forest = _fig_to_b64(fig3)
        
        return _to_native({
            'results': {
                'metrics': {
                    'accuracy':       acc,
                    'balanced_accuracy': balanced_acc,
                    'sensitivity':    sensitivity,
                    'specificity':    specificity,
                    'precision':      ppv,
                    'npv':            npv,
                    'f1':             f1,
                    'mcc':            mcc,
                    'optimal_threshold': opt_thr_test,
                    'optimal_threshold_note': 'Youden Index on held-out test set ROC',
                    'confusion_matrix': cm.tolist(),
                    'classification_report': report,
                },
                'full_sample': {
                    'roc_auc':         roc_auc_full,
                    'optimal_threshold': opt_threshold,
                    'sensitivity_at_opt': opt_sensitivity,
                    'specificity_at_opt': opt_specificity,
                    'fpr': fpr_full.tolist(),
                    'tpr': tpr_full.tolist(),
                    'note': 'ROC computed on full sample using full-sample model (optimistic — for calibration reference)',
                },
                'coefficients':   coefficients,
                'odds_ratios':    odds_ratios,
                'odds_ratios_ci': odds_ratios_ci,
                'p_values':       p_values,
                'std_errors':     std_errors,
                'z_values':       z_values,
                'intercept':      intercept,
                'model_summary':  model_summary,
                'vif':            vif,
                'hosmer_lemeshow': hl_test,
                'influence':      infl,
                'marginal_effects': marginal_effects,
                'roc_data': {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'auc': roc_auc,
                    'note': 'ROC on held-out test set',
                },
                'dependent_classes': [str(c) for c in classes],
                'interpretation': interp,
                'n_dropped': n_dropped,
                'n_total':   len(y),
                'n_train':   len(y_train),
                'n_test':    len(y_test),
            },
            'plot': plot,
            'plots': {
                'roc_cm':       plot,
                'calibration':  plot_calibration,
                'forest':       plot_forest,
            }
        })
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
