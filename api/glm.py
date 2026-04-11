from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, List, Optional
import numpy as np
import pandas as pd
import re
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.model_selection import KFold
from scipy import stats as _scipy_stats
import io, base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

router = APIRouter()

def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"


class GLMRequest(BaseModel):
    data: list[dict[str, Any]] = Field(...)
    target_var: str = Field(...)
    features: List[str] = Field(...)
    family: str = Field(default="gaussian")
    link_function: Optional[str] = Field(default=None)


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


def generate_interpretation(family, pseudo_r2, aic, bic, coefficients, target_var, n_obs, n_features):
    sig_coeffs = [c for c in coefficients if c['p_value'] < 0.05]
    
    effect_size = "large" if pseudo_r2 >= 0.26 else "medium" if pseudo_r2 >= 0.13 else "small" if pseudo_r2 >= 0.02 else "negligible"
    
    interpretation = "**Overall Analysis**\n"
    interpretation += f"→ GLM ({family}) predicting {target_var} with {n_features} predictors (N = {n_obs}).\n"
    interpretation += f"→ Pseudo R² = {pseudo_r2:.4f} ({effect_size} effect), AIC = {aic:.2f}, BIC = {bic:.2f}.\n"
    interpretation += f"→ {len(sig_coeffs)} of {len(coefficients)} coefficients significant at α = .05.\n\n"
    
    interpretation += "**Key Insights**\n"
    for c in sig_coeffs[:3]:
        if 'Intercept' not in c['variable']:
            if family in ['binomial', 'poisson'] and c.get('exp_coefficient'):
                interpretation += f"→ {c['variable']}: OR/RR = {c['exp_coefficient']:.3f}\n"
            else:
                interpretation += f"→ {c['variable']}: b = {c['coefficient']:.4f}\n"
    
    interpretation += "\n**Recommendations**\n"
    if pseudo_r2 < 0.1:
        interpretation += "→ Low pseudo R² - consider adding predictors.\n"
    else:
        interpretation += "→ Model appears reasonably specified.\n"

    return interpretation


def _append_diagnostics_to_interp(interp, cv_res, resid_diag, infl_diag):
    """Append CV / residual / influence summary lines to existing interpretation string."""
    if cv_res and cv_res.get('deviance_mean') is not None:
        interp += f"\n**Cross-Validation ({cv_res['n_folds']}-fold)**\n"
        interp += f"→ CV Deviance = {cv_res['deviance_mean']:.3f} ± {cv_res['deviance_std']:.3f}\n"
        if cv_res.get('r2_mean') is not None:
            interp += f"→ CV R² = {cv_res['r2_mean']:.3f} ± {cv_res['r2_std']:.3f}\n"
    if resid_diag:
        sw = resid_diag.get('shapiro_wilk', {})
        het = resid_diag.get('heteroscedasticity', {})
        if sw.get('p_value') is not None:
            label = 'normal' if sw.get('normal') else 'non-normal'
            interp += f"\n**Residual Diagnostics**\n→ Deviance residuals: Shapiro-Wilk p={sw['p_value']:.4f} ({label})\n"
        if het.get('detected'):
            interp += f"→ Heteroscedasticity detected (p={het['p_value']:.3f})\n"
    if infl_diag:
        nc = infl_diag.get('cooks_distance', {}).get('n_flagged', 0)
        if nc:
            interp += f"\n**Influence**: {nc} influential observation(s) (Cook's D > {infl_diag['cooks_distance']['cutoff']:.4f})\n"
    return interp


@router.post("/glm")
def glm_analysis(req: GLMRequest):
    try:
        df = pd.DataFrame(req.data)
        target_var = req.target_var
        features = req.features
        family_name = req.family.lower()
        link_function_name = req.link_function
        
        # Sanitize column names
        original_to_sanitized = {col: re.sub(r'[^A-Za-z0-9_]', '_', col) for col in df.columns}
        sanitized_to_original = {v: k for k, v in original_to_sanitized.items()}
        df_clean = df.rename(columns=original_to_sanitized)
        
        target_clean = original_to_sanitized.get(target_var, target_var)
        features_clean = [original_to_sanitized.get(f, f) for f in features]
        
        formula = f'Q("{target_clean}") ~ ' + ' + '.join([f'Q("{f}")' for f in features_clean])
        
        # Link functions
        link_map = {
            'logit': sm.families.links.Logit(),
            'probit': sm.families.links.Probit(),
            'log': sm.families.links.Log(),
        }
        link = link_map.get(link_function_name) if link_function_name else None
        
        # Family
        family_map = {
            'gaussian': sm.families.Gaussian(link=link) if link else sm.families.Gaussian(),
            'binomial': sm.families.Binomial(link=link) if link else sm.families.Binomial(),
            'poisson': sm.families.Poisson(link=link) if link else sm.families.Poisson(),
            'gamma': sm.families.Gamma(link=link if link else sm.families.links.Log()),
        }
        
        if family_name not in family_map:
            raise ValueError(f"Unsupported family: {family_name}")
        
        family = family_map[family_name]
        
        model = smf.glm(formula, data=df_clean, family=family)
        result = model.fit()
        
        pseudo_r2 = 1 - (result.deviance / result.null_deviance) if result.null_deviance > 0 else 0
        
        # Coefficients
        is_exp = family_name in ['binomial', 'poisson', 'gamma']
        coefficients = []
        
        for param_name in result.params.index:
            match = re.search(r'Q\("([^"]+)"\)', param_name)
            clean_name = sanitized_to_original.get(match.group(1), match.group(1)) if match else param_name
            
            row = {
                'variable': clean_name,
                'coefficient': float(result.params[param_name]),
                'p_value': float(result.pvalues[param_name]),
                'conf_int_lower': float(result.conf_int().loc[param_name, 0]),
                'conf_int_upper': float(result.conf_int().loc[param_name, 1]),
            }
            
            if is_exp:
                try:
                    row['exp_coefficient'] = float(np.exp(result.params[param_name]))
                except:
                    row['exp_coefficient'] = None
            
            coefficients.append(row)
        
        n_obs = len(df_clean.dropna(subset=[target_clean] + features_clean))

        # ── Residual diagnostics ─────────────────────────────────────────
        dev_resid   = result.resid_deviance
        pears_resid = result.resid_pearson
        fitted_vals = result.fittedvalues.values

        sw_stat, sw_p = (_scipy_stats.shapiro(dev_resid)
                         if len(dev_resid) <= 5000 else (None, None))
        bp_corr, bp_p = _scipy_stats.pearsonr(fitted_vals, np.abs(dev_resid))
        resid_diag = {
            'deviance_residuals': {
                'mean':     round(float(dev_resid.mean()),  4),
                'std':      round(float(dev_resid.std()),   4),
                'skewness': round(float(_scipy_stats.skew(dev_resid)),      4),
                'kurtosis': round(float(_scipy_stats.kurtosis(dev_resid)),  4),
            },
            'pearson_residuals': {
                'mean': round(float(pears_resid.mean()), 4),
                'std':  round(float(pears_resid.std()),  4),
            },
            'shapiro_wilk': {
                'statistic': round(float(sw_stat), 4) if sw_stat is not None else None,
                'p_value':   round(float(sw_p),   4) if sw_p   is not None else None,
                'normal':    bool(sw_p > 0.05)        if sw_p   is not None else None,
            },
            'heteroscedasticity': {
                'corr_fitted_abs_resid': round(float(bp_corr), 4),
                'p_value':              round(float(bp_p),    4),
                'detected':             bool(bp_p < 0.05),
            },
        }

        # ── Influence diagnostics ────────────────────────────────────────
        try:
            infl      = result.get_influence()
            hat       = infl.hat_matrix_diag
            cooks_d   = infl.cooks_distance[0]
            std_resid = infl.resid_studentized
            _n        = len(hat)
            _p        = len(features)
            cutoff_c  = 4.0 / _n
            cutoff_h  = 2 * (_p + 1) / _n
            flagged_c = [int(i) for i in np.where(cooks_d > cutoff_c)[0]]
            flagged_h = [int(i) for i in np.where(hat > cutoff_h)[0]]
            infl_diag = {
                'cooks_distance': {
                    'max':             round(float(cooks_d.max()), 4),
                    'cutoff':          round(cutoff_c, 4),
                    'n_flagged':       len(flagged_c),
                    'flagged_indices': flagged_c[:20],
                },
                'leverage': {
                    'mean':            round(float(hat.mean()), 4),
                    'max':             round(float(hat.max()),  4),
                    'cutoff':          round(cutoff_h, 4),
                    'n_flagged':       len(flagged_h),
                    'flagged_indices': flagged_h[:20],
                },
                'std_residuals': {
                    'max_abs':         round(float(np.abs(std_resid).max()), 4),
                    'n_outliers_3sd':  int(np.sum(np.abs(std_resid) > 3)),
                },
            }
        except Exception:
            infl_diag = None
            hat = cooks_d = std_resid = cutoff_c = cutoff_h = None

        # ── Cross-validation (manual KFold, deviance-based) ──────────────
        try:
            y_arr  = df_clean[target_clean].values.astype(float)
            X_cols = features_clean
            kf     = KFold(n_splits=5, shuffle=True, random_state=42)
            cv_dev, cv_r2 = [], []
            for tr, te in kf.split(y_arr):
                df_tr = df_clean.iloc[tr]
                df_te = df_clean.iloc[te]
                _m = smf.glm(formula, data=df_tr, family=family_map[family_name]).fit()
                _pred = _m.predict(df_te)
                _y    = df_te[target_clean].values.astype(float)
                # deviance per obs
                eps = 1e-12
                if family_name == 'gaussian':
                    _dev = float(np.mean((_y - _pred) ** 2))
                    ss_res = float(np.sum((_y - _pred)**2))
                    ss_tot = float(np.sum((_y - _y.mean())**2))
                    cv_r2.append(1 - ss_res/ss_tot if ss_tot > 0 else 0.0)
                elif family_name == 'binomial':
                    _pred_c = np.clip(_pred, eps, 1 - eps)
                    _dev = float(np.mean(-2 * (_y * np.log(_pred_c) + (1-_y)*np.log(1-_pred_c))))
                elif family_name == 'poisson':
                    _pred_c = np.clip(_pred, eps, None)
                    _dev = float(np.mean(2 * (_y * np.log(np.where(_y>0, _y/_pred_c, 1)) - (_y - _pred_c))))
                else:
                    _dev = float(np.mean((_y - _pred)**2))
                cv_dev.append(_dev)
            cv_res = {
                'deviance_mean': round(float(np.mean(cv_dev)), 4),
                'deviance_std':  round(float(np.std(cv_dev)),  4),
                'r2_mean':   round(float(np.mean(cv_r2)), 4) if cv_r2 else None,
                'r2_std':    round(float(np.std(cv_r2)),  4) if cv_r2 else None,
                'n_folds':   5,
            }
        except Exception:
            cv_res = None

        # ── Interpretation ───────────────────────────────────────────────
        interpretation = generate_interpretation(
            family_name, pseudo_r2, result.aic, result.bic,
            coefficients, target_var, n_obs, len(features))
        interpretation = _append_diagnostics_to_interp(
            interpretation, cv_res, resid_diag, infl_diag)

        # ── Plots ────────────────────────────────────────────────────────
        y_actual = df_clean[target_clean].values.astype(float)
        y_fitted = fitted_vals

        # Plot 1: Actual vs Predicted
        fig1, ax1 = plt.subplots(figsize=(7, 6))
        ax1.scatter(y_actual, y_fitted, alpha=0.55, color='#5B9BD5', s=28, edgecolors='none')
        lims = [min(y_actual.min(), y_fitted.min()), max(y_actual.max(), y_fitted.max())]
        ax1.plot(lims, lims, 'r--', lw=1.5, label='Perfect fit')
        ax1.set_xlabel('Actual'); ax1.set_ylabel('Predicted')
        ax1.set_title(f'Actual vs Predicted ({family_name})', fontweight='bold')
        ax1.legend(fontsize=8)
        plt.tight_layout()
        plot_actual_vs_pred = _fig_to_b64(fig1)

        # Plot 2: Residuals vs Fitted + Q-Q (deviance residuals)
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
        axes2[0].scatter(y_fitted, dev_resid, alpha=0.55, color='#5B9BD5', s=28, edgecolors='none')
        axes2[0].axhline(0, color='red', linestyle='--', lw=1.5)
        _xs = np.linspace(y_fitted.min(), y_fitted.max(), 200)
        axes2[0].plot(_xs, np.poly1d(np.polyfit(y_fitted, dev_resid, 2))(_xs),
                      color='orange', lw=1.5, label='Trend')
        axes2[0].set_xlabel('Fitted Values'); axes2[0].set_ylabel('Deviance Residuals')
        axes2[0].set_title('Residuals vs Fitted', fontweight='bold')
        axes2[0].legend(fontsize=8)
        _scipy_stats.probplot(dev_resid, dist='norm', plot=axes2[1])
        axes2[1].set_title('Normal Q-Q (Deviance Residuals)', fontweight='bold')
        axes2[1].get_lines()[0].set(markersize=4, alpha=0.6, color='#5B9BD5')
        axes2[1].get_lines()[1].set(color='red', lw=1.5)
        plt.tight_layout()
        plot_resid = _fig_to_b64(fig2)

        # Plot 3: Influence — Cook's D + Leverage vs Std Residual
        if infl_diag and cooks_d is not None:
            _n_plot = len(cooks_d)
            fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
            axes3[0].stem(range(_n_plot), cooks_d, linefmt='C0-', markerfmt='C0o',
                          basefmt='gray')
            axes3[0].axhline(cutoff_c, color='red', linestyle='--', lw=1.5,
                             label=f'4/n = {cutoff_c:.4f}')
            axes3[0].set_xlabel('Observation index'); axes3[0].set_ylabel("Cook's D")
            axes3[0].set_title("Cook's Distance", fontweight='bold')
            axes3[0].legend(fontsize=8)
            axes3[1].scatter(hat, std_resid, alpha=0.55, color='#5B9BD5', s=28)
            axes3[1].axhline( 3, color='red',    linestyle='--', lw=1.2, label='|z|=3')
            axes3[1].axhline(-3, color='red',    linestyle='--', lw=1.2)
            axes3[1].axvline(cutoff_h, color='orange', linestyle='--', lw=1.2,
                             label=f'h={cutoff_h:.3f}')
            axes3[1].set_xlabel('Leverage'); axes3[1].set_ylabel('Std Residual')
            axes3[1].set_title('Leverage vs Std Residual', fontweight='bold')
            axes3[1].legend(fontsize=8)
            plt.tight_layout()
            plot_influence = _fig_to_b64(fig3)
        else:
            plot_influence = None

        return _to_native({
            'results': {
                'aic':           float(result.aic),
                'bic':           float(result.bic),
                'log_likelihood':float(result.llf),
                'deviance':      float(result.deviance),
                'null_deviance': float(result.null_deviance),
                'pseudo_r2':     float(pseudo_r2),
                'df_model':      int(result.df_model),
                'df_resid':      int(result.df_resid),
                'coefficients':  coefficients,
                'family':        family_name,
                'cv_results':            cv_res,
                'residual_diagnostics':  resid_diag,
                'influence_diagnostics': infl_diag,
                'interpretation':        interpretation,
                'n_obs':    n_obs,
                'n_features': len(features),
            },
            'plot':           plot_actual_vs_pred,
            'residual_plot':  plot_resid,
            'influence_plot': plot_influence,
        })
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
