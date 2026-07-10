

import sys
import json
import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS, RandomEffects
import statsmodels.api as sm
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

def _to_native_type(obj):
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, (np.floating, float)): return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, (bool, np.bool_)): return bool(obj)
    return str(obj)

def generate_interpretation(fe_result, re_result, hausman_result, f_test_entity):
    """Generates an interpretation for the panel data regression results."""
    interp = "### Panel Data Regression Interpretation\n\n"
    
    # F-test Interpretation
    f_p = f_test_entity.get('p_value')
    if f_p is not None:
        if f_p < 0.05:
            interp += f"The **F-test for entity effects** is significant (p = {f_p:.4f}), which suggests that the Fixed Effects model is a better fit than a simple Pooled OLS model.\n\n"
        else:
            interp += f"The **F-test for entity effects** is not significant (p = {f_p:.4f}), suggesting that individual entity effects are not jointly significant. A Pooled OLS model might be sufficient.\n\n"

    # Hausman Test Interpretation
    hausman_p = hausman_result.get('p_value')
    if hausman_p is not None:
        if hausman_p < 0.05:
            interp += f"The **Hausman test** is significant (p = {hausman_p:.4f}), strongly suggesting that the **Fixed Effects (FE) model is more appropriate** than the Random Effects (RE) model. This indicates that unobserved entity-specific effects are likely correlated with the model's predictors.\n\n"
            preferred_model = fe_result
            model_name = "Fixed Effects"
        else:
            interp += f"The **Hausman test** is not significant (p = {hausman_p:.4f}), suggesting that the **Random Effects (RE) model is more efficient** and appropriate. This implies that the unobserved entity-specific effects are likely not correlated with the predictors.\n\n"
            preferred_model = re_result
            model_name = "Random Effects"
    else:
        interp += f"The Hausman test could not be performed ({hausman_result.get('error', 'reason unknown')}), so a definitive choice between FE and RE models cannot be made statistically. We will interpret the Fixed Effects model by default as it is generally more robust to omitted variable bias.\n\n"
        preferred_model = fe_result
        model_name = "Fixed Effects"
        
    # Interpretation of the preferred model
    interp += f"**Interpreting the {model_name} Model:**\n"
    
    significant_vars = []
    if preferred_model and 'pvalues' in preferred_model and preferred_model['pvalues']:
        for param, pval in preferred_model['pvalues'].items():
            if param != 'Intercept' and pval is not None and pval < 0.05:
                coef = preferred_model['params'][param]
                direction = "positively" if coef > 0 else "negatively"
                significant_vars.append(f"**{param}** (coefficient: {coef:.3f})")

    if significant_vars:
        interp += f"- The model shows that {', '.join(significant_vars)} significantly influence(s) the dependent variable, holding other factors constant.\n"
    else:
        interp += "- None of the independent variables showed a statistically significant effect on the dependent variable in this model.\n"
        
    rsquared_key = 'rsquared_within' if model_name == 'Fixed Effects' else 'rsquared_overall'
    if preferred_model and rsquared_key in preferred_model and preferred_model.get(rsquared_key) is not None:
        r2 = preferred_model.get(rsquared_key)
        if r2 is not None:
            interp += f"- The model explains approximately **{(r2 * 100):.1f}%** of the variance in the dependent variable ({'within entities' if rsquared_key == 'rsquared_within' else 'overall'}).\n"

    interp += "\n**Conclusion:** Based on the diagnostic tests, the analysis suggests the **{} model** is the most suitable. This approach provides more reliable estimates than simple regression by appropriately handling the panel structure of the data.".format(model_name)
    
    return interp

def manual_hausman_test(fe_model, re_model):
    """
    Manually calculates the Hausman test statistic.
    This is more robust than linearmodels.compare in some cases.
    """
    b_fe = fe_model.params
    b_re = re_model.params
    v_fe = fe_model.cov
    v_re = re_model.cov
    
    common_params = list(set(b_fe.index) & set(b_re.index))
    if 'Intercept' in common_params:
        common_params.remove('Intercept')

    if not common_params:
        raise ValueError("No common coefficients between models for Hausman test.")

    b_diff = (b_fe[common_params] - b_re[common_params]).values.reshape(-1, 1)
    v_diff = v_fe.loc[common_params, common_params] - v_re.loc[common_params, common_params]

    try:
        v_diff_inv = np.linalg.inv(v_diff)
        chi2_stat = b_diff.T @ v_diff_inv @ b_diff
        df = len(b_diff)
        p_value = 1 - stats.chi2.cdf(chi2_stat.item(), df)
        return chi2_stat.item(), p_value, df
    except np.linalg.LinAlgError:
        raise ValueError("Could not compute Hausman test. The difference in covariance matrices is singular. This can happen with robust standard errors or if models are too similar.")


def main():
    try:
        payload = json.load(sys.stdin)
        data = pd.DataFrame(payload.get('data'))
        dependent = payload.get('dependent')
        exog = payload.get('exog_cols') or payload.get('exog') # Handles both names
        entity_col = payload.get('entity_col')
        time_col = payload.get('time_col')

        if not all([data is not None, dependent, exog, entity_col, time_col]):
            raise ValueError("Missing required parameters.")
            
        df = data.copy()
        
        for col in [dependent] + exog:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=[dependent] + exog + [entity_col, time_col])
        
        if df.empty:
            raise ValueError("No valid data for panel analysis after cleaning.")

        df[time_col] = pd.to_datetime(df[time_col])
        df = df.set_index([entity_col, time_col])
        
        y = df[dependent]
        X = sm.add_constant(df[exog])

        # --- Pooled OLS ---
        pooled_ols = PanelOLS(y, X).fit(cov_type='robust')
        
        # --- Fixed Effects ---
        fe_model = PanelOLS(y, X, entity_effects=True).fit(cov_type='robust')
        
        # --- Random Effects ---
        re_model = RandomEffects(y, X).fit(cov_type='robust')

        # --- F-test for entity effects ---
        f_test_results = {}
        try:
            f_test_entity = fe_model.f_statistic_robust
            f_test_results = {
                'statistic': f_test_entity.stat, 'p_value': f_test_entity.pval, 'df': f_test_entity.df, 'df_resid': f_test_entity.df_resid,
                'interpretation': 'Fixed effects are significant (use FE over OLS)' if f_test_entity.pval < 0.05 else 'Fixed effects are not significant (OLS may be sufficient)'
            }
        except Exception as e:
            f_test_results = {'statistic': None, 'p_value': None, 'interpretation': f'F-test could not be computed: {e}'}

        # --- Hausman Test ---
        hausman_results = {}
        try:
            fe_nonrobust = PanelOLS(y, X, entity_effects=True).fit()
            re_nonrobust = RandomEffects(y, X).fit()
            chi2_stat, p_value, df_hausman = manual_hausman_test(fe_nonrobust, re_nonrobust)
            hausman_results = {
                'statistic': chi2_stat, 'p_value': p_value, 'df': df_hausman,
                'interpretation': 'Fixed Effects Recommended' if p_value < 0.05 else 'Random Effects Recommended'
            }
        except Exception as e:
            hausman_results = {'error': str(e), 'p_value': None, 'interpretation': f'Test failed: {e}'}
            
        def format_summary(model_fit, name):
            summary = {
                'name': name,
                'params': model_fit.params.to_dict(),
                'pvalues': model_fit.pvalues.to_dict(),
                'tstats': model_fit.tstats.to_dict(),
                'rsquared': None,
                'rsquared_within': None
            }
            try:
                summary['rsquared'] = getattr(model_fit, 'rsquared_overall', getattr(model_fit, 'rsquared', None))
            except Exception: pass
            try:
                summary['rsquared_within'] = getattr(model_fit, 'rsquared_within', None)
            except Exception: pass
            return summary

        pooled_res = format_summary(pooled_ols, 'Pooled OLS')
        fe_res = format_summary(fe_model, 'Fixed Effects')
        re_res = format_summary(re_model, 'Random Effects')
        
        interpretation = generate_interpretation(fe_res, re_res, hausman_results, f_test_results)

        response = {
            'results': {
                'pooled_ols': pooled_res,
                'fixed_effects': fe_res,
                'random_effects': re_res,
                'f_test_entity': f_test_results,
                'hausman_test': hausman_results,
                'interpretation': interpretation
            }
        }
        
        print(json.dumps(response, default=_to_native_type, ensure_ascii=False))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()

    