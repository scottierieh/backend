from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy import stats

router = APIRouter()

try:
    import semopy
    from semopy import Model
    SEMOPY_AVAILABLE = True
except ImportError:
    SEMOPY_AVAILABLE = False


class FactorDefinition(BaseModel):
    name: str
    indicators: List[str]


class CfaRequest(BaseModel):
    data: List[Dict[str, Any]]
    factors: List[FactorDefinition]
    estimator: str = "MLW"


def _to_native(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [_to_native(x) for x in obj.tolist()]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_native(x) for x in obj]
    return obj


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return 'data:image/png;base64,' + base64.b64encode(buf.read()).decode('utf-8')


def safe_get(value, default=0):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return default
    return value


def safe_float(val):
    """안전하게 float 변환 - '-' 같은 문자열 처리"""
    if val is None:
        return None
    if isinstance(val, (int, float)):
        if pd.isna(val):
            return None
        return float(val)
    if isinstance(val, str):
        val = val.strip()
        if val in ['-', '', 'NA', 'NaN', 'nan', 'N/A']:
            return None
        try:
            return float(val)
        except ValueError:
            return None
    return None


def validate_cfa_input(df, factors, all_indicators):
    """CFA 입력 검증. 오류는 ValueError, 경고는 리스트로 반환."""
    warnings_list = []

    # factor별 indicator 수 검증
    for factor in factors:
        n_ind = len(factor.indicators)
        if n_ind < 2:
            raise ValueError(
                f"Factor '{factor.name}' has only {n_ind} indicator(s). "
                "CFA requires at least 2 indicators per factor (3+ recommended)."
            )
        if n_ind == 2:
            warnings_list.append(
                f"Factor '{factor.name}' has only 2 indicators. "
                "A minimum of 3 is recommended for identification and stability."
            )

    # indicator 중복 검증
    all_inds_flat = [ind for f in factors for ind in f.indicators]
    duplicates = [ind for ind in set(all_inds_flat) if all_inds_flat.count(ind) > 1]
    if duplicates:
        warnings_list.append(
            f"Indicator(s) assigned to multiple factors: {', '.join(duplicates)}. "
            "Cross-loadings are not supported in standard CFA."
        )

    # constant / zero-variance indicator 검증
    X_check = df[all_indicators].apply(pd.to_numeric, errors='coerce').dropna()
    constant_items = [col for col in X_check.columns if X_check[col].std() == 0]
    if constant_items:
        raise ValueError(
            f"Indicator(s) with zero variance (constant): {', '.join(constant_items)}. "
            "Remove these before running CFA."
        )

    near_constant = [col for col in X_check.columns if 0 < X_check[col].std() < 0.01]
    if near_constant:
        warnings_list.append(
            f"Near-constant indicator(s) (SD < 0.01): {', '.join(near_constant)}. "
            "These may cause estimation problems."
        )

    # 표본 수 경고 — n_params_est 활용 (obs/parameter ratio)
    n_obs = len(X_check)
    n_params_est = sum(len(f.indicators) for f in factors)  # 최소 추정 파라미터 수
    obs_per_param = n_obs / n_params_est if n_params_est > 0 else n_obs

    if n_obs < 50:
        warnings_list.append(
            f"Very small sample (N = {n_obs}, ~{obs_per_param:.1f} obs/parameter). "
            "CFA results are likely unstable. N >= 200 is recommended for reliable estimates."
        )
    elif n_obs < 100:
        warnings_list.append(
            f"Small sample (N = {n_obs}, ~{obs_per_param:.1f} obs/parameter). "
            "Estimates may be unstable. N >= 200 is recommended."
        )
    elif n_obs < 200:
        warnings_list.append(
            f"Moderate sample (N = {n_obs}, ~{obs_per_param:.1f} obs/parameter). "
            "N >= 200 provides more stable CFA estimates."
        )

    # subjects-per-indicator ratio
    ratio = n_obs / len(all_indicators) if all_indicators else 0
    if ratio < 5:
        warnings_list.append(
            f"Subject-to-indicator ratio is {ratio:.1f}:1. "
            "A minimum of 10:1 is recommended for stable CFA."
        )

    return warnings_list


def interpret_fit(fit_indices):
    interpretations = []
    
    cfi = safe_get(fit_indices.get('CFI'), 0)
    if cfi >= 0.95:
        interpretations.append({'metric': 'CFI', 'value': f"{cfi:.3f}", 'status': 'excellent', 'interpretation': 'Excellent fit (≥0.95)'})
    elif cfi >= 0.90:
        interpretations.append({'metric': 'CFI', 'value': f"{cfi:.3f}", 'status': 'acceptable', 'interpretation': 'Acceptable fit (≥0.90)'})
    else:
        interpretations.append({'metric': 'CFI', 'value': f"{cfi:.3f}", 'status': 'poor', 'interpretation': 'Poor fit (<0.90)'})
    
    tli = safe_get(fit_indices.get('TLI'), safe_get(fit_indices.get('NNFI'), 0))
    if tli >= 0.95:
        interpretations.append({'metric': 'TLI', 'value': f"{tli:.3f}", 'status': 'excellent', 'interpretation': 'Excellent fit (≥0.95)'})
    elif tli >= 0.90:
        interpretations.append({'metric': 'TLI', 'value': f"{tli:.3f}", 'status': 'acceptable', 'interpretation': 'Acceptable fit (≥0.90)'})
    else:
        interpretations.append({'metric': 'TLI', 'value': f"{tli:.3f}", 'status': 'poor', 'interpretation': 'Poor fit (<0.90)'})
    
    rmsea = safe_get(fit_indices.get('RMSEA'), 1)
    if rmsea <= 0.05:
        interpretations.append({'metric': 'RMSEA', 'value': f"{rmsea:.3f}", 'status': 'excellent', 'interpretation': 'Close fit (≤0.05)'})
    elif rmsea <= 0.08:
        interpretations.append({'metric': 'RMSEA', 'value': f"{rmsea:.3f}", 'status': 'acceptable', 'interpretation': 'Reasonable fit (≤0.08)'})
    else:
        interpretations.append({'metric': 'RMSEA', 'value': f"{rmsea:.3f}", 'status': 'poor', 'interpretation': 'Poor fit (>0.08)'})
    
    srmr = safe_get(fit_indices.get('SRMR'), 1)
    if srmr <= 0.05:
        interpretations.append({'metric': 'SRMR', 'value': f"{srmr:.3f}", 'status': 'excellent', 'interpretation': 'Excellent fit (≤0.05)'})
    elif srmr <= 0.08:
        interpretations.append({'metric': 'SRMR', 'value': f"{srmr:.3f}", 'status': 'acceptable', 'interpretation': 'Acceptable fit (≤0.08)'})
    else:
        interpretations.append({'metric': 'SRMR', 'value': f"{srmr:.3f}", 'status': 'poor', 'interpretation': 'Poor fit (>0.08)'})
    
    return interpretations


def compute_srmr(model, obs_cov, var_names):
    """
    SRMR 직접 계산 (semopy calc_stats가 SRMR을 제공하지 않는 경우 대비).
    SRMR = sqrt( 2 * sum_{i>=j} (r_ij_obs - r_ij_implied)^2 / (p*(p+1)) )
    r_ij = cov_ij / sqrt(cov_ii * cov_jj)  (상관계수로 변환 후 비교)
    """
    try:
        sigma, _ = model.calc_sigma()
        p = len(var_names)
        total = 0.0
        count = 0
        for i in range(p):
            for j in range(i + 1):  # lower triangle including diagonal
                obs_ij = obs_cov[i, j]
                imp_ij = sigma[i, j]
                # 상관계수로 변환
                obs_denom = np.sqrt(abs(obs_cov[i, i] * obs_cov[j, j]))
                imp_denom = np.sqrt(abs(sigma[i, i] * sigma[j, j]))
                r_obs = obs_ij / obs_denom if obs_denom > 0 else 0.0
                r_imp = imp_ij / imp_denom if imp_denom > 0 else 0.0
                total += (r_obs - r_imp) ** 2
                count += 1
        srmr = np.sqrt(total / count) if count > 0 else None
        return float(srmr) if srmr is not None else None
    except Exception:
        return None


def compute_rmsea(chi2, df, n_obs):
    """
    RMSEA 직접 계산.
    RMSEA = sqrt(max((chi2 - df) / (df * (N - 1)), 0))
    chi2, df: 모델 카이제곱 통계량과 자유도
    n_obs: 표본 수
    """
    try:
        if df is None or df <= 0 or chi2 is None or n_obs is None or n_obs <= 1:
            return None
        rmsea = np.sqrt(max((chi2 - df) / (df * (n_obs - 1)), 0.0))
        return float(rmsea)
    except Exception:
        return None


def compute_standardized_residuals(model, obs_cov, var_names):
    """
    표준화 잔차 행렬: (obs_cov - sigma) / sqrt(sigma_ii * sigma_jj)
    |값| > 0.10 이면 해당 indicator pair에 문제 있을 수 있음.
    """
    try:
        sigma, _ = model.calc_sigma()
        resid = obs_cov - sigma
        p = len(var_names)
        std_resid = np.zeros((p, p))
        for i in range(p):
            for j in range(p):
                denom = np.sqrt(abs(sigma[i, i] * sigma[j, j]))
                std_resid[i, j] = resid[i, j] / denom if denom > 0 else 0.0
        df = pd.DataFrame(std_resid, index=var_names, columns=var_names)
        # upper-triangle only (i < j), flatten to list
        rows = []
        for i in range(p):
            for j in range(i + 1, p):
                rows.append({
                    'lhs': var_names[i],
                    'rhs': var_names[j],
                    'std_residual': round(float(std_resid[i, j]), 4),
                    'flagged': abs(std_resid[i, j]) > 0.10,
                })
        rows.sort(key=lambda x: abs(x['std_residual']), reverse=True)
        # also return full matrix as nested dict for heatmap
        matrix = {r: {c: round(float(df.loc[r, c]), 4) for c in var_names} for r in var_names}
        return rows, matrix
    except Exception:
        return [], {}


def compute_factor_scores(model, data, var_names, factor_names):
    """
    Bartlett / regression factor scores via semopy predict_factors.
    Returns descriptive stats (mean, sd, min, max) per factor.
    """
    try:
        fs = model.predict_factors(data[var_names])
        if fs is None or fs.empty:
            return None, None
        # rename columns to factor names if needed
        if list(fs.columns) != factor_names and len(fs.columns) == len(factor_names):
            fs.columns = factor_names
        desc = {}
        for col in fs.columns:
            desc[col] = {
                'mean': round(float(fs[col].mean()), 4),
                'std': round(float(fs[col].std()), 4),
                'min': round(float(fs[col].min()), 4),
                'max': round(float(fs[col].max()), 4),
            }
        return fs.round(4).to_dict(orient='records'), desc
    except Exception:
        return None, None


def compute_cross_loading_suggestions(base_model, data, factors, var_names, top_n=5):
    """
    각 indicator를 원래 factor 외 다른 factor에 추가했을 때 chi-sq 개선이 있는지 검정.
    delta_chi2 기준 상위 top_n 제안 반환.
    """
    try:
        s0 = semopy.calc_stats(base_model)
        chi2_0 = float(s0['chi2'].values[0])
        df_0 = float(s0['DoF'].values[0])
    except Exception:
        return []

    factor_map = {f.name: list(f.indicators) for f in factors}
    factor_names = [f.name for f in factors]
    suggestions = []

    for f_name in factor_names:
        assigned = factor_map[f_name]
        others = [v for v in var_names if v not in assigned]
        for ind in others:
            try:
                lines = []
                for fn in factor_names:
                    inds = factor_map[fn] + ([ind] if fn == f_name else [])
                    lines.append(f"{fn} =~ {' + '.join(inds)}")
                m2 = Model('\n'.join(lines))
                m2.fit(data[var_names])
                s2 = semopy.calc_stats(m2)
                chi2_1 = float(s2['chi2'].values[0])
                df_1 = float(s2['DoF'].values[0])
                delta_chi2 = chi2_0 - chi2_1
                delta_df = df_0 - df_1
                if delta_df <= 0:
                    continue
                p_val = float(1 - stats.chi2.cdf(delta_chi2, df=delta_df))
                # loading estimate
                est2 = m2.inspect()
                loading_val = None
                for op in ['=~', '~']:
                    mask = ((est2['lval'] == f_name) & (est2['rval'] == ind) & (est2['op'] == op)) | \
                           ((est2['lval'] == ind) & (est2['rval'] == f_name) & (est2['op'] == op))
                    if mask.any():
                        loading_val = round(float(est2[mask].iloc[0]['Estimate']), 3)
                        break
                suggestions.append({
                    'indicator': ind,
                    'suggested_factor': f_name,
                    'original_factor': next((fn for fn, inds in factor_map.items() if ind in inds), None),
                    'delta_chi2': round(delta_chi2, 3),
                    'delta_df': int(delta_df),
                    'p_value': round(p_val, 4),
                    'loading_estimate': loading_val,
                    'significant': bool(p_val < 0.05),
                })
            except Exception:
                continue

    suggestions.sort(key=lambda x: x['delta_chi2'], reverse=True)
    return suggestions[:top_n]


def compute_modification_indices(model, obs_cov, var_names, n_obs, top_n=10):
    """
    Approximate modification indices via standardized residual covariance.
    MI ≈ (n-1)/2 * (r_ij / sqrt(sigma_ii*sigma_jj + sigma_ij²))²
    Returns top_n residual covariance suggestions sorted by MI descending.
    """
    try:
        sigma, _ = model.calc_sigma()
        resid = obs_cov - sigma
        p = len(var_names)

        rows = []
        for i in range(p):
            for j in range(i + 1, p):
                denom = np.sqrt(sigma[i, i] * sigma[j, j] + sigma[i, j] ** 2)
                norm_r = resid[i, j] / denom if denom > 0 else 0
                mi_val = ((n_obs - 1) / 2) * norm_r ** 2
                p_val = float(1 - stats.chi2.cdf(mi_val, df=1))
                rows.append({
                    'lhs': var_names[i],
                    'op': '~~',
                    'rhs': var_names[j],
                    'mi': round(float(mi_val), 3),
                    'epc': round(float(resid[i, j]), 4),
                    'p_value': round(p_val, 4),
                    'significant': p_val < 0.05,
                })

        rows.sort(key=lambda x: x['mi'], reverse=True)
        return rows[:top_n]
    except Exception:
        return []


@router.post("/cfa")
async def confirmatory_factor_analysis(request: CfaRequest):
    if not SEMOPY_AVAILABLE:
        raise HTTPException(status_code=500, detail="semopy not installed")

    try:
        df = pd.DataFrame(request.data)
        factors = request.factors

        # Validate indicators exist
        all_indicators = []
        factor_names = []
        for factor in factors:
            factor_names.append(factor.name)
            for ind in factor.indicators:
                if ind not in df.columns:
                    raise HTTPException(status_code=400, detail=f"Indicator '{ind}' not found in data")
                if ind not in all_indicators:
                    all_indicators.append(ind)

        # #4 #5 강화된 검증
        try:
            validation_warnings = validate_cfa_input(df, factors, all_indicators)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))

        # Prepare data — #5 numeric validation with invalid column detection
        df_numeric = df[all_indicators].apply(pd.to_numeric, errors='coerce')
        non_numeric_cols = []
        for col in all_indicators:
            original_non_null = df[col].notna().sum()
            after_non_null = df_numeric[col].notna().sum()
            if after_non_null < original_non_null:
                n_invalid = original_non_null - after_non_null
                non_numeric_cols.append(f"{col} ({n_invalid} non-numeric values coerced to NaN)")
        if non_numeric_cols:
            validation_warnings.append(
                f"Non-numeric values detected and removed in: {'; '.join(non_numeric_cols)}"
            )

        X = df_numeric.dropna()
        dropped_rows = len(df) - len(X)
        if dropped_rows > 0:
            validation_warnings.append(
                f"{dropped_rows} row(s) dropped due to missing or non-numeric values "
                f"(N before: {len(df)}, N after: {len(X)})."
            )

        n_obs = len(X)
        n_vars = len(all_indicators)
        n_factors = len(factors)

        if n_obs < 10:
            raise HTTPException(status_code=400, detail=f"Need at least 10 observations after dropping missing values, got {n_obs}")

        # Build semopy model syntax
        model_lines = []
        for factor in factors:
            indicators_str = " + ".join(factor.indicators)
            model_lines.append(f"{factor.name} =~ {indicators_str}")
        model_syntax = "\n".join(model_lines)

        # #1 estimator applied — map request.estimator to semopy obj parameter
        estimator = (request.estimator or "MLW").upper()
        estimator_map = {
            "MLW": "MLW", "ML": "ML", "ULS": "ULS",
            "GLS": "GLS", "WLS": "WLS", "DWLS": "DWLS"
        }
        fit_obj = estimator_map.get(estimator, "MLW")

        # Fit CFA — capture semopy warnings instead of suppressing globally
        captured_warnings = []
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model = Model(model_syntax)
            try:
                model.fit(X, obj=fit_obj)
            except TypeError:
                # Older semopy versions may not support obj= keyword
                model.fit(X)
                validation_warnings.append(
                    f"Estimator '{estimator}' could not be applied (semopy version may not support it); "
                    "default estimator was used."
                )
            for warning in w:
                msg = str(warning.message)
                if any(kw in msg.lower() for kw in ['converge', 'heywood', 'singular', 'not positive', 'iteration']):
                    captured_warnings.append(msg)

        if captured_warnings:
            validation_warnings.extend(captured_warnings)

        estimates = model.inspect()

        # Get fit indices
        try:
            fit_stats = semopy.calc_stats(model)
            fit_indices_clean = {}
            target_keys = ['chi2', 'DoF', 'chi2 p-value', 'CFI', 'TLI', 'NNFI', 'RMSEA', 'SRMR', 'AIC', 'BIC', 'GFI', 'AGFI', 'NFI']

            # semopy 버전마다 DoF 컬럼명이 다를 수 있으므로 fallback 목록 준비
            dof_aliases = ['DoF', 'df', 'dof', 'Df', 'DF', 'degrees_of_freedom']

            def _extract_val(fs, key):
                """DataFrame/dict에서 값 추출. 없으면 None."""
                if isinstance(fs, pd.DataFrame):
                    if key in fs.columns:
                        v = fs[key].values[0] if len(fs) > 0 else None
                        return v
                    elif key in fs.index:
                        v = fs.loc[key]
                        return v.values[0] if hasattr(v, 'values') and len(v.values) > 0 else v
                elif isinstance(fs, dict):
                    return fs.get(key)
                return None

            for key in target_keys:
                raw = _extract_val(fit_stats, key)

                # DoF는 정수형으로 별도 처리 + alias fallback
                if key == 'DoF':
                    if raw is None:
                        for alias in dof_aliases:
                            raw = _extract_val(fit_stats, alias)
                            if raw is not None:
                                break
                    # 직접 계산 fallback: p*(p+1)/2 - t
                    if raw is None or (isinstance(raw, str) and raw.strip() in ['-', '', 'NA']):
                        try:
                            p = n_vars  # 관측변수 수
                            # semopy inspect()에서 자유 파라미터 수 추출
                            free_params = estimates[estimates['op'].isin(['=~', '~~', '~'])].copy()
                            # fixed params (fixed == True 또는 Estimate가 고정값) 제외
                            if 'Fixed' in free_params.columns:
                                free_params = free_params[free_params['Fixed'] == False]
                            t = len(free_params)
                            raw = int(p * (p + 1) / 2 - t)
                        except Exception:
                            raw = None
                    # 최종 int 변환
                    try:
                        fit_indices_clean[key] = int(float(raw)) if raw is not None else None
                    except (ValueError, TypeError):
                        fit_indices_clean[key] = None
                else:
                    fit_indices_clean[key] = safe_float(raw)

        except Exception as e:
            fit_indices_clean = {}
            validation_warnings.append(f"Fit indices could not be computed: {str(e)}")

        # fit_interpretation은 SRMR/RMSEA fallback 계산 후 아래에서 실행

        # Parse loadings
        loadings_table = []
        loadings_matrix = {}
        loading_ops = ['=~', '~', 'lambda']

        for var in all_indicators:
            row = {'indicator': var}
            for f_name in factor_names:
                found = False
                for op in loading_ops:
                    mask1 = (estimates['lval'] == f_name) & (estimates['rval'] == var) & (estimates['op'] == op)
                    mask2 = (estimates['lval'] == var) & (estimates['rval'] == f_name) & (estimates['op'] == op)
                    for mask in [mask1, mask2]:
                        if mask.any():
                            est_row = estimates[mask].iloc[0]
                            loading = est_row['Estimate']
                            se = est_row.get('Std. Err', est_row.get('SE', est_row.get('std', None)))
                            pval = est_row.get('p-value', est_row.get('pvalue', est_row.get('Pr(>|z|)', None)))
                            loading_val = safe_float(loading)
                            se_val = safe_float(se)
                            pval_val = safe_float(pval)
                            row[f_name] = {
                                'estimate': loading_val if loading_val is not None else 0,
                                'se': se_val,
                                'z': loading_val / se_val if loading_val is not None and se_val and se_val > 0 else None,
                                'pvalue': pval_val
                            }
                            if f_name not in loadings_matrix:
                                loadings_matrix[f_name] = {}
                            loadings_matrix[f_name][var] = loading_val if loading_val is not None else 0
                            found = True
                            break
                    if found:
                        break
                if not found:
                    row[f_name] = None
            loadings_table.append(row)

        # #2 Factor correlations — normalize covariance to correlation
        # semopy '~~' between factors returns covariance; convert to correlation
        factor_variances = {}
        for f_name in factor_names:
            mask_var = (estimates['lval'] == f_name) & (estimates['rval'] == f_name) & (estimates['op'] == '~~')
            if mask_var.any():
                var_val = safe_float(estimates[mask_var].iloc[0]['Estimate'])
                factor_variances[f_name] = var_val if var_val and var_val > 0 else 1.0
            else:
                factor_variances[f_name] = 1.0  # assume standardized

        factor_correlations = {}
        for f1 in factor_names:
            factor_correlations[f1] = {}
            for f2 in factor_names:
                if f1 == f2:
                    factor_correlations[f1][f2] = 1.0
                else:
                    mask = ((estimates['lval'] == f1) & (estimates['rval'] == f2) & (estimates['op'] == '~~')) | \
                           ((estimates['lval'] == f2) & (estimates['rval'] == f1) & (estimates['op'] == '~~'))
                    if mask.any():
                        cov_val = safe_float(estimates[mask].iloc[0]['Estimate'])
                        if cov_val is not None:
                            var1 = factor_variances.get(f1, 1.0)
                            var2 = factor_variances.get(f2, 1.0)
                            denom = np.sqrt(var1 * var2)
                            corr = cov_val / denom if denom > 0 else cov_val
                            # clip to [-1, 1]
                            factor_correlations[f1][f2] = float(np.clip(corr, -1.0, 1.0))
                        else:
                            factor_correlations[f1][f2] = 0.0
                    else:
                        factor_correlations[f1][f2] = 0.0

        # Reliability (CR, AVE)
        reliability = {}
        heywood_cases = []   # #3 collect Heywood cases

        for factor in factors:
            loadings_list = []
            error_vars = []
            for ind in factor.indicators:
                if factor.name in loadings_matrix and ind in loadings_matrix[factor.name]:
                    lam = loadings_matrix[factor.name][ind]
                    loadings_list.append(lam)

                    # #3 loading > 1 Heywood check
                    if abs(lam) > 1.0:
                        heywood_cases.append(
                            f"{ind} (loading = {lam:.3f} on {factor.name})"
                        )

                    mask = (estimates['lval'] == ind) & (estimates['rval'] == ind) & (estimates['op'] == '~~')
                    if mask.any():
                        err = estimates[mask].iloc[0]['Estimate']
                        err_val = safe_float(err)
                        # #3 negative error variance Heywood check
                        if err_val is not None and err_val < 0:
                            heywood_cases.append(
                                f"{ind} (negative error variance = {err_val:.4f})"
                            )
                        error_vars.append(err_val if err_val is not None and err_val >= 0 else abs(1 - lam**2))
                    else:
                        error_vars.append(abs(1 - lam**2))
            if loadings_list:
                sum_lam = sum(loadings_list)
                sum_lam_sq = sum([l**2 for l in loadings_list])
                sum_err = sum(error_vars)
                cr = (sum_lam ** 2) / (sum_lam ** 2 + sum_err) if (sum_lam ** 2 + sum_err) > 0 else 0
                ave = sum_lam_sq / (sum_lam_sq + sum_err) if (sum_lam_sq + sum_err) > 0 else 0
                reliability[factor.name] = {
                    'composite_reliability': float(np.clip(cr, 0, 1)),
                    'ave': float(np.clip(ave, 0, 1)),
                    'sqrt_ave': float(np.sqrt(np.clip(ave, 0, 1))),
                    'n_indicators': len(factor.indicators)
                }

        # #3 Heywood case warnings
        if heywood_cases:
            validation_warnings.append(
                f"⚠ Heywood case(s) detected — improper solution: {'; '.join(heywood_cases)}. "
                "This indicates estimation problems (possibly due to small N, too few indicators, or model misspecification). "
                "Reliability indices for affected factors may be unreliable."
            )
        # Modification indices + new analyses
        obs_cov_matrix = X[all_indicators].cov().values

        # SRMR fallback: semopy calc_stats가 제공하지 않으면 직접 계산
        if not fit_indices_clean.get('SRMR'):
            srmr_computed = compute_srmr(model, obs_cov_matrix, all_indicators)
            if srmr_computed is not None:
                fit_indices_clean['SRMR'] = round(srmr_computed, 4)

        # RMSEA fallback: semopy calc_stats가 제공하지 않으면 chi2/df/N으로 직접 계산
        if not fit_indices_clean.get('RMSEA'):
            rmsea_computed = compute_rmsea(
                fit_indices_clean.get('chi2'),
                fit_indices_clean.get('DoF'),
                n_obs
            )
            if rmsea_computed is not None:
                fit_indices_clean['RMSEA'] = round(rmsea_computed, 4)

        # SRMR/RMSEA fallback 완료 후 최종 fit_interpretation + overall_fit 계산
        fit_interpretation = interpret_fit(fit_indices_clean)
        good_count = sum(1 for fi in fit_interpretation if fi['status'] in ['excellent', 'acceptable'])
        total = len(fit_interpretation)
        if good_count == total:
            overall_fit = 'excellent'; overall_message = 'Model demonstrates excellent fit.'
        elif good_count >= total * 0.75:
            overall_fit = 'good'; overall_message = 'Model demonstrates good fit overall.'
        elif good_count >= total * 0.5:
            overall_fit = 'acceptable'; overall_message = 'Model fit is acceptable but could be improved.'
        else:
            overall_fit = 'poor'; overall_message = 'Model fit is poor. Consider revising the factor structure.'

        modification_indices = compute_modification_indices(
            model, obs_cov_matrix, all_indicators, n_obs, top_n=10
        )

        # #1 Standardized residual matrix
        std_resid_list, std_resid_matrix = compute_standardized_residuals(
            model, obs_cov_matrix, all_indicators
        )

        # #2 Factor scores
        factor_scores_records, factor_scores_desc = compute_factor_scores(
            model, X, all_indicators, factor_names
        )

        # #3 Cross-loading suggestions (chi-sq difference test)
        cross_loading_suggestions = compute_cross_loading_suggestions(
            model, X, factors, all_indicators, top_n=5
        )

        # Insights
        insights = []

        # Low loadings
        low_loadings = list(set([
            row['indicator'] for row in loadings_table
            for f_name in factor_names
            if row.get(f_name) and row[f_name].get('estimate', 0) < 0.5
        ]))
        if low_loadings:
            insights.append({'type': 'warning', 'title': 'Low Factor Loadings',
                             'description': f"Indicators with loadings < 0.5: {', '.join(sorted(low_loadings))}. "
                                            "Consider revising or removing these items."})

        # Convergent validity (AVE)
        low_ave = [f for f, r in reliability.items() if r['ave'] < 0.5]
        if low_ave:
            insights.append({'type': 'warning', 'title': 'Convergent Validity Concern',
                             'description': f"Factors with AVE < 0.5: {', '.join(low_ave)}. "
                                            "Items do not share sufficient variance with their factor."})

        # High factor correlations → discriminant validity
        high_corr_pairs = []
        for i, f1 in enumerate(factor_names):
            for f2 in factor_names[i+1:]:
                corr_val = abs(factor_correlations.get(f1, {}).get(f2, 0))
                if corr_val >= 0.85:
                    high_corr_pairs.append(f"{f1}–{f2} (r = {corr_val:.2f})")
        if high_corr_pairs:
            insights.append({'type': 'warning', 'title': 'High Factor Correlation',
                             'description': f"Highly correlated factor pairs: {', '.join(high_corr_pairs)}. "
                                            "Factors may not be empirically distinguishable."})

        # Discriminant validity: AVE > r² check (Fornell-Larcker)
        discriminant_failures = []
        for i, f1 in enumerate(factor_names):
            for f2 in factor_names[i+1:]:
                r = factor_correlations.get(f1, {}).get(f2, 0)
                ave1 = reliability.get(f1, {}).get('ave', 1.0)
                ave2 = reliability.get(f2, {}).get('ave', 1.0)
                if r ** 2 > min(ave1, ave2):
                    discriminant_failures.append(
                        f"{f1}–{f2} (r²={r**2:.2f} > AVE min={min(ave1,ave2):.2f})"
                    )
        if discriminant_failures:
            insights.append({'type': 'warning', 'title': 'Discriminant Validity Concern (Fornell-Larcker)',
                             'description': f"r² exceeds AVE for: {', '.join(discriminant_failures)}. "
                                            "Factors may overlap too much to be considered distinct constructs."})

        # Too many warnings summary
        if len(validation_warnings) >= 3:
            insights.append({'type': 'warning', 'title': 'Multiple Issues Detected',
                             'description': f"{len(validation_warnings)} warnings were raised. "
                                            "Review all warnings carefully before interpreting results."})

        # Significant modification indices suggestion
        sig_mi = [m for m in modification_indices if m['significant']]
        if sig_mi:
            top = sig_mi[0]
            insights.append({
                'type': 'info',
                'title': 'Modification Index Suggestion',
                'description': (
                    f"Largest significant MI: {top['lhs']} ~~ {top['rhs']} "
                    f"(MI = {top['mi']:.1f}, EPC = {top['epc']:+.3f}). "
                    "Adding this residual covariance path may improve model fit. "
                    "Only apply if theoretically justified."
                )
            })

        if not insights:
            insights.append({'type': 'info', 'title': 'Model Quality ✓',
                            'description': 'All validity measures are within acceptable ranges.'})

        # ============ PLOTS — scoped seaborn style (#3) ============
        with plt.style.context('seaborn-v0_8-whitegrid'):

            # Plot 1: Factor Loadings heatmap
            fig1, ax1 = plt.subplots(figsize=(10, max(6, n_vars * 0.5)))
            loading_data = [
                {'Indicator': var, 'Factor': f_name, 'Loading': loadings_matrix[f_name][var]}
                for var in all_indicators for f_name in factor_names
                if f_name in loadings_matrix and var in loadings_matrix[f_name]
            ]
            if loading_data:
                load_df = pd.DataFrame(loading_data)
                pivot = load_df.pivot(index='Indicator', columns='Factor', values='Loading')
                sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlBu_r', center=0,
                           ax=ax1, vmin=-1, vmax=1, cbar_kws={'label': 'Loading'})
                ax1.set_title('Factor Loadings', fontweight='bold', fontsize=14)
            else:
                ax1.text(0.5, 0.5, 'No loadings extracted', ha='center', va='center', fontsize=14)
                ax1.set_title('Factor Loadings (No Data)', fontweight='bold', fontsize=14)
            plt.tight_layout()
            plot_loadings = fig_to_base64(fig1)

            # Plot 2: Factor Correlations
            fig2, ax2 = plt.subplots(figsize=(8, 6))
            corr_matrix = np.array([[factor_correlations[f1][f2] for f2 in factor_names] for f1 in factor_names])
            sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
                       xticklabels=factor_names, yticklabels=factor_names, ax=ax2,
                       vmin=-1, vmax=1, square=True)
            ax2.set_title('Factor Correlations', fontweight='bold', fontsize=14)
            plt.tight_layout()
            plot_factor_corr = fig_to_base64(fig2)

            # #6 Plot 3: Fit Indices — split incremental vs badness-of-fit
            fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(12, 5))

            # Incremental fit: CFI, TLI (higher = better)
            inc_metrics = ['CFI', 'TLI', 'NFI']
            inc_vals = [safe_get(fit_indices_clean.get(m), 0) for m in inc_metrics]
            inc_colors = ['#2E7D32' if v >= 0.95 else '#FFA726' if v >= 0.90 else '#EF5350' for v in inc_vals]
            bars_a = ax3a.bar(inc_metrics, inc_vals, color=inc_colors, alpha=0.8, edgecolor='black')
            ax3a.axhline(y=0.95, color='green', linestyle='--', lw=1.5, label='Excellent (≥0.95)')
            ax3a.axhline(y=0.90, color='orange', linestyle='--', lw=1.5, label='Acceptable (≥0.90)')
            ax3a.set_ylim(0, 1.1)
            ax3a.set_ylabel('Value (higher = better)')
            ax3a.set_title('Incremental Fit Indices', fontweight='bold', fontsize=13)
            ax3a.legend(fontsize=8)
            for bar, val in zip(bars_a, inc_vals):
                ax3a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                         f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

            # Badness of fit: RMSEA, SRMR (lower = better)
            bad_metrics = ['RMSEA', 'SRMR']
            bad_vals = [safe_get(fit_indices_clean.get(m), 0) for m in bad_metrics]
            bad_colors = ['#2E7D32' if v <= 0.05 else '#FFA726' if v <= 0.08 else '#EF5350' for v in bad_vals]
            bars_b = ax3b.bar(bad_metrics, bad_vals, color=bad_colors, alpha=0.8, edgecolor='black')
            ax3b.axhline(y=0.05, color='green', linestyle='--', lw=1.5, label='Excellent (≤0.05)')
            ax3b.axhline(y=0.08, color='orange', linestyle='--', lw=1.5, label='Acceptable (≤0.08)')
            ax3b.set_ylim(0, max(0.15, max(bad_vals) * 1.2) if bad_vals else 0.15)
            ax3b.set_ylabel('Value (lower = better)')
            ax3b.set_title('Absolute Fit Indices', fontweight='bold', fontsize=13)
            ax3b.legend(fontsize=8)
            for bar, val in zip(bars_b, bad_vals):
                ax3b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                         f'{val:.3f}', ha='center', va='bottom', fontweight='bold')

            plt.tight_layout()
            plot_fit = fig_to_base64(fig3)

            # Plot 4: Reliability
            fig4, axes = plt.subplots(1, 2, figsize=(12, 5))
            if reliability:
                plot_factors = [f for f in factor_names if f in reliability]
                cr_vals = [reliability[f]['composite_reliability'] for f in plot_factors]
                ave_vals = [reliability[f]['ave'] for f in plot_factors]
                colors_cr = ['#2E7D32' if v >= 0.7 else '#EF5350' for v in cr_vals]
                bars1 = axes[0].bar(plot_factors, cr_vals, color=colors_cr, alpha=0.7, edgecolor='black')
                axes[0].axhline(y=0.7, color='blue', linestyle='--', label='Threshold (0.70)')
                axes[0].set_ylabel('CR'); axes[0].set_title('Composite Reliability', fontweight='bold')
                axes[0].set_ylim(0, 1); axes[0].legend()
                for bar, val in zip(bars1, cr_vals):
                    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                                f'{val:.3f}', ha='center', va='bottom')
                colors_ave = ['#2E7D32' if v >= 0.5 else '#EF5350' for v in ave_vals]
                bars2 = axes[1].bar(plot_factors, ave_vals, color=colors_ave, alpha=0.7, edgecolor='black')
                axes[1].axhline(y=0.5, color='blue', linestyle='--', label='Threshold (0.50)')
                axes[1].set_ylabel('AVE'); axes[1].set_title('Average Variance Extracted', fontweight='bold')
                axes[1].set_ylim(0, 1); axes[1].legend()
                for bar, val in zip(bars2, ave_vals):
                    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                                f'{val:.3f}', ha='center', va='bottom')
            else:
                axes[0].text(0.5, 0.5, 'No reliability data', ha='center', va='center')
                axes[1].text(0.5, 0.5, 'No reliability data', ha='center', va='center')
            plt.tight_layout()
            plot_reliability = fig_to_base64(fig4)

            # Plot 5: Path Diagram
            fig5, ax5 = plt.subplots(figsize=(14, 10))
            ax5.axis('off')
            factor_y = np.linspace(0.2, 0.8, n_factors) if n_factors > 1 else [0.5]
            factor_x = 0.2; indicator_x = 0.75
            for f_idx, f_name in enumerate(factor_names):
                y = factor_y[f_idx]
                circle = plt.Circle((factor_x, y), 0.07, fill=False, linewidth=2, color='blue')
                ax5.add_patch(circle)
                ax5.text(factor_x, y, f_name, ha='center', va='center', fontsize=10, fontweight='bold')
            for f_idx, factor in enumerate(factors):
                f_y = factor_y[f_idx]
                n_ind = len(factor.indicators)
                spread = min(0.15, 0.6 / max(n_factors, 1))
                ind_ys = np.linspace(f_y - spread, f_y + spread, n_ind) if n_ind > 1 else [f_y]
                for ind_idx, ind in enumerate(factor.indicators):
                    ind_y = ind_ys[ind_idx]
                    rect = plt.Rectangle((indicator_x - 0.08, ind_y - 0.025), 0.16, 0.05,
                                        fill=False, linewidth=1.5, color='black')
                    ax5.add_patch(rect)
                    ax5.text(indicator_x, ind_y, ind, ha='center', va='center', fontsize=8)
                    loading = loadings_matrix.get(factor.name, {}).get(ind, 0)
                    ax5.annotate('', xy=(indicator_x - 0.08, ind_y), xytext=(factor_x + 0.07, f_y),
                               arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
                    mid_x = (factor_x + 0.07 + indicator_x - 0.08) / 2
                    mid_y = (f_y + ind_y) / 2
                    ax5.text(mid_x, mid_y, f'{loading:.2f}', fontsize=8, ha='center',
                            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            if n_factors > 1:
                for i in range(n_factors):
                    for j in range(i+1, n_factors):
                        y1, y2 = factor_y[i], factor_y[j]
                        corr = factor_correlations[factor_names[i]][factor_names[j]]
                        ax5.annotate('', xy=(factor_x - 0.07, y1), xytext=(factor_x - 0.07, y2),
                                   arrowprops=dict(arrowstyle='<->', color='red', lw=1.5,
                                                 connectionstyle='arc3,rad=-0.3'))
                        ax5.text(factor_x - 0.15, (y1 + y2) / 2, f'{corr:.2f}', fontsize=9, color='red')
            ax5.set_xlim(0, 1); ax5.set_ylim(0, 1)
            ax5.set_title('Path Diagram (Standardized Loadings)', fontweight='bold', fontsize=14)
            plt.tight_layout()
            plot_path = fig_to_base64(fig5)

            # #4 Plot 6: Modification Index visualization
            plot_mi = None
            if modification_indices:
                mi_top = modification_indices[:min(10, len(modification_indices))]
                labels = [f"{r['lhs']} ~~ {r['rhs']}" for r in mi_top]
                mi_vals = [r['mi'] for r in mi_top]
                epc_vals = [r['epc'] for r in mi_top]
                sig_flags = [r['significant'] for r in mi_top]

                fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(14, max(5, len(mi_top) * 0.5 + 2)))

                # MI bar chart
                colors_mi = ['#EF5350' if s else '#90A4AE' for s in sig_flags]
                bars_mi = ax6a.barh(range(len(labels)), mi_vals, color=colors_mi, edgecolor='black', alpha=0.8)
                ax6a.set_yticks(range(len(labels)))
                ax6a.set_yticklabels(labels, fontsize=9)
                ax6a.axvline(x=3.84, color='red', linestyle='--', lw=1.5, label='χ²(df=1) p<.05 (3.84)')
                ax6a.set_xlabel('Modification Index (MI)')
                ax6a.set_title('Modification Indices\n(red = significant, suggest freeing path)', fontweight='bold')
                ax6a.legend(fontsize=8)
                ax6a.invert_yaxis()
                for bar, val in zip(bars_mi, mi_vals):
                    ax6a.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
                             f'{val:.2f}', va='center', fontsize=8)

                # EPC bar chart
                colors_epc = ['#EF5350' if e > 0 else '#42A5F5' for e in epc_vals]
                ax6b.barh(range(len(labels)), epc_vals, color=colors_epc, edgecolor='black', alpha=0.8)
                ax6b.set_yticks(range(len(labels)))
                ax6b.set_yticklabels(labels, fontsize=9)
                ax6b.axvline(x=0, color='black', lw=1)
                ax6b.set_xlabel('Expected Parameter Change (EPC)')
                ax6b.set_title('Expected Parameter Change\n(direction of suggested path)', fontweight='bold')
                ax6b.invert_yaxis()
                for i, val in enumerate(epc_vals):
                    ax6b.text(val + (0.002 if val >= 0 else -0.002),
                             i, f'{val:+.3f}', va='center',
                             ha='left' if val >= 0 else 'right', fontsize=8)

                plt.tight_layout()
                plot_mi = fig_to_base64(fig6)

            # Plot 7: Standardized residual heatmap
            plot_std_resid = None
            if std_resid_matrix:
                p = len(all_indicators)
                fig7, ax7 = plt.subplots(figsize=(max(6, p * 0.8), max(5, p * 0.7)))
                mat = np.array([[std_resid_matrix[r][c] for c in all_indicators] for r in all_indicators])
                sns.heatmap(mat, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                           vmin=-0.2, vmax=0.2,
                           xticklabels=all_indicators, yticklabels=all_indicators,
                           ax=ax7, cbar_kws={'label': 'Standardized Residual'})
                ax7.set_title('Standardized Residual Matrix\n(|value| > 0.10 flagged as problematic)',
                             fontweight='bold', fontsize=13)
                # mark flagged cells
                for i, r in enumerate(all_indicators):
                    for j, c in enumerate(all_indicators):
                        if i != j and abs(mat[i, j]) > 0.10:
                            ax7.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                                        edgecolor='gold', lw=2.5))
                plt.tight_layout()
                plot_std_resid = fig_to_base64(fig7)

        # Build response — #1 debug 제거, warnings 포함
        result = {
            'model_syntax': model_syntax,
            'warnings': validation_warnings,          # user-facing warnings
            'summary': {
                'n_observations': n_obs,
                'n_factors': n_factors,
                'n_indicators': n_vars,
                'estimator': request.estimator,
                'overall_fit': overall_fit,
                'overall_message': overall_message
            },
            'fit_indices': fit_indices_clean,
            'fit_interpretation': fit_interpretation,
            'loadings': loadings_table,
            'factor_correlations': factor_correlations,
            'reliability': reliability,
            'insights': insights,
            'modification_indices': modification_indices,
            'standardized_residuals': {
                'list': std_resid_list,       # sorted by |value|, flagged if >0.10
                'matrix': std_resid_matrix,   # full matrix for custom rendering
            },
            'factor_scores': {
                'records': factor_scores_records,   # per-subject scores (list of dicts)
                'descriptives': factor_scores_desc, # mean/std/min/max per factor
            },
            'cross_loading_suggestions': cross_loading_suggestions,
            'plots': {
                'loadings': plot_loadings,
                'factor_correlations': plot_factor_corr,
                'fit_indices': plot_fit,
                'reliability': plot_reliability,
                'path_diagram': plot_path,
                'modification_indices': plot_mi,         # #4 MI bar + EPC chart
                'standardized_residuals': plot_std_resid # standardized resid heatmap
            }
        }

        return _to_native(result)

    except HTTPException:
        raise
    except Exception as e:
        # #9 운영용 에러 — traceback은 로그에만, 사용자에겐 간결한 메시지
        import logging, traceback
        logging.error("CFA analysis error: %s\n%s", str(e), traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
