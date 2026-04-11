from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
import traceback

warnings.filterwarnings('ignore')

router = APIRouter()


# ══════════════════════════════════════════════════════════════
# Request Model
# ══════════════════════════════════════════════════════════════

class ConjointRequest(BaseModel):
    data: List[Dict[str, Any]]
    responseCol: str          # 1=chosen, 0=not chosen
    taskCol: str              # choice task / question ID
    respCol: str              # respondent ID
    altCol: Optional[str] = None   # alternative ID (optional)
    attributeCols: List[str]  # attribute level columns
    priceCol: Optional[str] = None  # price column for WTP (must be numeric-parseable)
    nSegments: int = 3              # number of segments for K-means
    noneOption: bool = False  # whether "none" / "no choice" option exists
    simulationProfiles: Optional[List[Dict[str, Any]]] = None


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

def _to_native(obj):
    """Recursively convert numpy types to Python native for JSON."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
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


def safe_float(val, default=0.0):
    try:
        if val is None:
            return default
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except Exception:
        return default


# ══════════════════════════════════════════════════════════════
# Data Parser — convert raw rows into design matrix
# ══════════════════════════════════════════════════════════════

def parse_conjoint_data(
    raw_data: List[Dict[str, Any]],
    response_col: str,
    task_col: str,
    resp_col: str,
    alt_col: Optional[str],
    attribute_cols: List[str],
    none_option: bool = False,
):
    """
    Parse CBC conjoint survey data into a design matrix suitable
    for conditional logit estimation.

    Returns:
        df: DataFrame with original data + dummy columns
        design_cols: list of dummy column names (for model matrix X)
        attribute_map: dict { attribute_name: [level_names...] }
        ref_levels: dict { attribute_name: reference_level }
    """
    df = pd.DataFrame(raw_data)

    # Validate required columns
    for col_name, col_val in [('response', response_col), ('task', task_col), ('respondent', resp_col)]:
        if col_val not in df.columns:
            raise HTTPException(status_code=400, detail=f"Column '{col_val}' not found. Available: {list(df.columns)}")

    missing_attrs = [c for c in attribute_cols if c not in df.columns]
    if missing_attrs:
        raise HTTPException(status_code=400, detail=f"Attribute columns not found: {missing_attrs}")

    # Ensure types
    df[response_col] = pd.to_numeric(df[response_col], errors='coerce').fillna(0).astype(int)
    df[task_col] = df[task_col].astype(str)
    df[resp_col] = df[resp_col].astype(str)

    # Create unique choice situation ID (respondent + task)
    df['_choice_id'] = df[resp_col] + '||' + df[task_col]

    # Build attribute map and dummy encode
    attribute_map = {}
    ref_levels = {}
    design_cols = []

    # If none_option: rows where all attribute values are empty/none = none alternative
    # Tag them and set attribute dummies to 0 (none has no attribute utility)
    if none_option:
        # Identify none rows: alternative == 0 or all attr cols empty
        def is_none_row(row):
            return all(str(row.get(a, '')).strip() in ('', 'none', 'None') for a in attribute_cols)
        df['_is_none'] = df.apply(is_none_row, axis=1)
    else:
        df['_is_none'] = False

    for attr in attribute_cols:
        df[attr] = df[attr].astype(str).str.strip()
        # Exclude none rows from level detection
        non_none_vals = df.loc[~df['_is_none'], attr]
        levels = sorted(non_none_vals.unique().tolist())
        # Remove empty/none strings
        levels = [l for l in levels if l not in ('', 'none', 'None')]

        if len(levels) < 2:
            continue  # skip constant attributes

        attribute_map[attr] = levels
        ref_levels[attr] = levels[0]

        # Dummy encoding: 0 for none rows, normal encoding for regular rows
        for lvl in levels[1:]:
            col_name = f"{attr}__{lvl}"
            df[col_name] = np.where(df['_is_none'], 0.0, (df[attr] == lvl).astype(float))
            design_cols.append(col_name)

    # ASC (Alternative Specific Constant) for none option
    if none_option and df['_is_none'].any():
        df['_ASC_none'] = df['_is_none'].astype(float)
        design_cols.append('_ASC_none')

    if len(design_cols) == 0:
        raise HTTPException(status_code=400, detail="No valid attribute levels found for modeling.")

    # Validate choice structure
    choice_counts = df.groupby('_choice_id')[response_col].sum()
    invalid = choice_counts[choice_counts != 1]
    if len(invalid) > len(choice_counts) * 0.1:
        pass  # proceed anyway, MLE handles it

    # None rate
    none_rate = float(df.loc[df['_is_none'], response_col].sum() / max(df['_choice_id'].nunique(), 1)) if none_option else 0.0

    n_respondents = df[resp_col].nunique()
    n_tasks = df['_choice_id'].nunique()
    n_alts_per_task = df.groupby('_choice_id').size().median()

    data_summary = {
        'n_respondents': int(n_respondents),
        'n_tasks': int(n_tasks),
        'n_observations': len(df),
        'n_alternatives_per_task': float(n_alts_per_task),
        'n_attributes': len(attribute_map),
        'n_parameters': len(design_cols),
        'none_rate': round(none_rate * 100, 1),
    }

    return df, design_cols, attribute_map, ref_levels, data_summary


# ══════════════════════════════════════════════════════════════
# Conditional Logit — MLE from scratch (scipy)
# ══════════════════════════════════════════════════════════════

def conditional_logit_fit(df, design_cols, response_col, choice_id_col='_choice_id'):
    """
    Fit conditional logit model via Maximum Likelihood Estimation.

    The log-likelihood for conditional logit:
        LL = Σ_n Σ_j  y_nj * [V_nj - log(Σ_k exp(V_nk))]
    where V_nj = X_nj @ beta

    Returns dict with coefficients, standard errors, z-values, p-values, fit stats.
    """
    X = df[design_cols].values.astype(np.float64)
    y = df[response_col].values.astype(np.float64)
    groups = df[choice_id_col].values

    # Build group indices for fast logsumexp
    unique_groups = np.unique(groups)
    group_indices = {g: np.where(groups == g)[0] for g in unique_groups}
    group_list = [group_indices[g] for g in unique_groups]

    n_params = X.shape[1]
    n_choices = len(unique_groups)

    def neg_log_likelihood(beta):
        V = X @ beta
        ll = 0.0
        for idx in group_list:
            v_group = V[idx]
            y_group = y[idx]
            # log-sum-exp trick for numerical stability
            max_v = np.max(v_group)
            log_denom = max_v + np.log(np.sum(np.exp(v_group - max_v)))
            ll += np.sum(y_group * (v_group - log_denom))
        return -ll

    def neg_log_likelihood_grad(beta):
        V = X @ beta
        grad = np.zeros(n_params)
        for idx in group_list:
            v_group = V[idx]
            y_group = y[idx]
            x_group = X[idx]
            max_v = np.max(v_group)
            exp_v = np.exp(v_group - max_v)
            probs = exp_v / np.sum(exp_v)
            grad -= (y_group - probs) @ x_group
        return grad

    # Initial values
    beta0 = np.zeros(n_params)

    # Optimize
    result = minimize(
        neg_log_likelihood,
        beta0,
        jac=neg_log_likelihood_grad,
        method='BFGS',
        options={'maxiter': 5000, 'disp': False},
    )

    if not result.success:
        # Try L-BFGS-B as fallback
        result = minimize(
            neg_log_likelihood,
            beta0,
            jac=neg_log_likelihood_grad,
            method='L-BFGS-B',
            options={'maxiter': 5000, 'disp': False},
        )

    beta_hat = result.x
    ll_model = -result.fun

    # Null model log-likelihood (all betas = 0)
    ll_null = 0.0
    for idx in group_list:
        n_alts = len(idx)
        chosen = np.sum(y[idx])
        ll_null += chosen * np.log(1.0 / n_alts)

    # Hessian for standard errors (numerical)
    from scipy.optimize import approx_fprime

    def hessian_approx(beta):
        n = len(beta)
        H = np.zeros((n, n))
        eps = 1e-5
        for i in range(n):
            def grad_i(b):
                return neg_log_likelihood_grad(b)[i]
            H[i, :] = approx_fprime(beta, grad_i, eps)
        return H

    H = hessian_approx(beta_hat)

    try:
        cov_matrix = np.linalg.inv(H)
        se = np.sqrt(np.maximum(np.diag(cov_matrix), 1e-12))
    except np.linalg.LinAlgError:
        se = np.full(n_params, np.nan)

    z_values = beta_hat / se
    p_values = 2 * (1 - norm.cdf(np.abs(z_values)))

    # Goodness of fit
    k = n_params
    n = n_choices
    aic = 2 * k - 2 * ll_model
    bic = k * np.log(n) - 2 * ll_model
    aicc = aic + (2 * k * (k + 1)) / max(n - k - 1, 1)
    mcfadden_r2 = 1 - (ll_model / ll_null) if ll_null != 0 else 0
    adj_mcfadden_r2 = 1 - ((ll_model - k) / ll_null) if ll_null != 0 else 0

    # Hit rate (prediction accuracy)
    V_all = X @ beta_hat
    hit = 0
    total = 0
    for idx in group_list:
        v_group = V_all[idx]
        y_group = y[idx]
        predicted = np.argmax(v_group)
        actual = np.argmax(y_group)
        if predicted == actual:
            hit += 1
        total += 1
    hit_rate = hit / total if total > 0 else 0

    return {
        'beta': beta_hat,
        'se': se,
        'z_values': z_values,
        'p_values': p_values,
        'll_model': ll_model,
        'll_null': ll_null,
        'aic': aic,
        'bic': bic,
        'aicc': aicc,
        'mcfadden_r2': mcfadden_r2,
        'adj_mcfadden_r2': adj_mcfadden_r2,
        'hit_rate': hit_rate,
        'n_params': n_params,
        'n_choices': n_choices,
        'converged': result.success,
    }


# ══════════════════════════════════════════════════════════════
# Part-worth & Relative Importance
# ══════════════════════════════════════════════════════════════

def compute_partworths_and_importance(
    beta: np.ndarray,
    se: np.ndarray,
    z_values: np.ndarray,
    p_values: np.ndarray,
    design_cols: List[str],
    attribute_map: Dict[str, List[str]],
    ref_levels: Dict[str, str],
):
    """
    Build structured part-worth utilities and relative importance scores.

    Part-worths include the reference level (fixed at 0) and all estimated levels.
    Importance = range of part-worths within attribute / sum of all ranges.
    """
    # Map coefficients to attribute->level
    coef_lookup = {}
    for i, col in enumerate(design_cols):
        coef_lookup[col] = {
            'coef': float(beta[i]),
            'se': float(se[i]),
            'z': float(z_values[i]),
            'p': float(p_values[i]),
        }

    partworths = {}  # attr -> [ {level, coef, se, z, p, significant} ]
    importance_ranges = {}

    for attr, levels in attribute_map.items():
        ref = ref_levels[attr]
        attr_pw = [{
            'level': ref,
            'coef': 0.0,
            'se': 0.0,
            'z': 0.0,
            'p': 1.0,
            'significant': False,
            'is_reference': True,
        }]

        for lvl in levels:
            if lvl == ref:
                continue
            col = f"{attr}__{lvl}"
            if col in coef_lookup:
                c = coef_lookup[col]
                attr_pw.append({
                    'level': lvl,
                    'coef': c['coef'],
                    'se': c['se'],
                    'z': c['z'],
                    'p': c['p'],
                    'significant': c['p'] < 0.05,
                    'is_reference': False,
                })
            else:
                attr_pw.append({
                    'level': lvl, 'coef': 0.0, 'se': 0.0,
                    'z': 0.0, 'p': 1.0, 'significant': False, 'is_reference': False,
                })

        partworths[attr] = attr_pw

        # Range for importance
        coefs = [pw['coef'] for pw in attr_pw]
        importance_ranges[attr] = max(coefs) - min(coefs)

    # Relative importance (%)
    total_range = sum(importance_ranges.values())
    importance = {}
    if total_range > 0:
        for attr in attribute_map:
            importance[attr] = (importance_ranges[attr] / total_range) * 100
    else:
        for attr in attribute_map:
            importance[attr] = 100.0 / len(attribute_map)

    return partworths, importance


# ══════════════════════════════════════════════════════════════
# Price Parser
# ══════════════════════════════════════════════════════════════

import re

def parse_price_value(val) -> Optional[float]:
    """Extract numeric value from price string like '$799', '1,200원', '€50.99'."""
    if val is None:
        return None
    s = str(val).strip()
    # Remove currency symbols, commas, spaces
    cleaned = re.sub(r'[^\d.]', '', s)
    try:
        f = float(cleaned)
        return f if f > 0 else None
    except (ValueError, TypeError):
        return None


# ══════════════════════════════════════════════════════════════
# WTP (Willingness to Pay)
# ══════════════════════════════════════════════════════════════

def compute_wtp_cbc(
    df, response_col, choice_id_col, price_col,
    attribute_cols_no_price, attribute_map, ref_levels,
):
    """
    Compute WTP by fitting a model with price as continuous variable.
    WTP_k = -beta_k / beta_price

    Returns:
        wtp_results: list of {attribute, level, wtp, wtp_ci_lower, wtp_ci_upper}
        beta_price: float
        price_col_name: str (the design column name for price)
    """
    # Build design matrix: dummies for non-price attrs + continuous price
    design_cols_wtp = []
    for attr in attribute_cols_no_price:
        if attr not in attribute_map:
            continue
        ref = ref_levels[attr]
        for lvl in attribute_map[attr]:
            if lvl == ref:
                continue
            col = f"{attr}__{lvl}"
            if col in df.columns:
                design_cols_wtp.append(col)

    # Parse price to numeric
    price_numeric = df[price_col].apply(parse_price_value)
    if price_numeric.isna().sum() > len(df) * 0.5:
        return None, None  # too many unparseable prices

    df['_price_numeric'] = price_numeric.fillna(price_numeric.median())
    design_cols_wtp.append('_price_numeric')

    # Fit conditional logit with continuous price
    fit_wtp = conditional_logit_fit(df, design_cols_wtp, response_col, choice_id_col)
    beta_wtp = fit_wtp['beta']
    se_wtp = fit_wtp['se']

    # Price coefficient is the last one
    beta_price = beta_wtp[-1]
    se_price = se_wtp[-1]

    if abs(beta_price) < 1e-10:
        return None, None  # price has no effect, can't compute WTP

    # Compute WTP for each non-price attribute level
    wtp_results = []
    idx = 0
    for attr in attribute_cols_no_price:
        if attr not in attribute_map:
            continue
        ref = ref_levels[attr]
        for lvl in attribute_map[attr]:
            if lvl == ref:
                wtp_results.append({
                    'attribute': attr, 'level': lvl,
                    'wtp': 0.0, 'wtp_ci_lower': 0.0, 'wtp_ci_upper': 0.0,
                    'is_reference': True,
                })
                continue
            b_k = beta_wtp[idx]
            se_k = se_wtp[idx]
            wtp = -b_k / beta_price
            # Delta method for WTP SE: se_wtp ≈ |1/beta_price| * sqrt(se_k² + (b_k/beta_price)² * se_price²)
            wtp_se = abs(1 / beta_price) * np.sqrt(se_k**2 + (b_k / beta_price)**2 * se_price**2)
            wtp_results.append({
                'attribute': attr, 'level': lvl,
                'wtp': safe_float(wtp),
                'wtp_ci_lower': safe_float(wtp - 1.96 * wtp_se),
                'wtp_ci_upper': safe_float(wtp + 1.96 * wtp_se),
                'is_reference': False,
            })
            idx += 1

    return wtp_results, safe_float(beta_price)


# ══════════════════════════════════════════════════════════════
# Segmentation (K-means on individual-level utilities)
# ══════════════════════════════════════════════════════════════

from sklearn.cluster import KMeans

def cbc_segmentation(
    df, design_cols, response_col, resp_col, choice_id_col,
    attribute_map, ref_levels, n_segments=3,
):
    """
    1. Estimate individual-level part-worths (per-respondent conditional logit)
    2. K-means clustering on part-worth vectors
    3. PCA for 2D cluster visualization
    4. Return segment profiles with importance, part-worths, and scatter data
    """
    from sklearn.decomposition import PCA

    respondents = df[resp_col].unique()
    if len(respondents) < n_segments * 2:
        return None  # not enough respondents

    # Aggregate model as fallback
    agg_fit = conditional_logit_fit(df, design_cols, response_col, choice_id_col)
    agg_beta = agg_fit['beta']

    # Individual-level estimation
    indiv_betas = []
    resp_ids = []

    for resp in respondents:
        rdf = df[df[resp_col] == resp]
        if len(rdf) < len(design_cols) + 2:
            indiv_betas.append(agg_beta.copy())  # fallback
            resp_ids.append(resp)
            continue
        try:
            rfit = conditional_logit_fit(rdf, design_cols, response_col, choice_id_col)
            beta_clipped = np.clip(rfit['beta'], -10, 10)
            indiv_betas.append(beta_clipped)
            resp_ids.append(resp)
        except Exception:
            indiv_betas.append(agg_beta.copy())
            resp_ids.append(resp)

    beta_matrix = np.array(indiv_betas)
    beta_matrix = np.nan_to_num(beta_matrix, nan=0.0, posinf=5.0, neginf=-5.0)

    # K-means
    n_segments = min(n_segments, len(beta_matrix) - 1)
    if n_segments < 2:
        return None

    kmeans = KMeans(n_clusters=n_segments, random_state=42, n_init=10)
    labels = kmeans.fit_predict(beta_matrix)

    # PCA for 2D scatter
    n_components = min(2, beta_matrix.shape[1])
    pca = PCA(n_components=n_components)
    coords = pca.fit_transform(beta_matrix)
    explained_var = pca.explained_variance_ratio_

    # Cluster centroids in PCA space
    centroids_pca = pca.transform(kmeans.cluster_centers_)

    # Scatter data for frontend
    scatter_data = []
    for i in range(len(resp_ids)):
        scatter_data.append({
            'respondent': str(resp_ids[i]),
            'pc1': safe_float(coords[i, 0]),
            'pc2': safe_float(coords[i, 1]) if n_components >= 2 else 0.0,
            'segment': int(labels[i] + 1),
        })

    centroid_data = []
    for seg_id in range(n_segments):
        centroid_data.append({
            'segment': seg_id + 1,
            'pc1': safe_float(centroids_pca[seg_id, 0]),
            'pc2': safe_float(centroids_pca[seg_id, 1]) if n_components >= 2 else 0.0,
        })

    # Build segment profiles
    segments = []
    for seg_id in range(n_segments):
        mask = labels == seg_id
        seg_betas = beta_matrix[mask]
        seg_mean = seg_betas.mean(axis=0)
        n_resp = int(mask.sum())

        seg_pw, seg_importance = compute_partworths_and_importance(
            seg_mean,
            np.zeros_like(seg_mean),
            np.zeros_like(seg_mean),
            np.ones_like(seg_mean),
            design_cols, attribute_map, ref_levels,
        )

        # Best level per attribute for this segment
        preferred = {}
        for attr, pw_list in seg_pw.items():
            best = max(pw_list, key=lambda x: x['coef'])
            preferred[attr] = best['level']

        segments.append({
            'segment_id': seg_id + 1,
            'n_respondents': n_resp,
            'pct_respondents': safe_float(n_resp / len(respondents) * 100),
            'importance': {attr: safe_float(v) for attr, v in seg_importance.items()},
            'partworths': {attr: [_to_native(pw) for pw in pw_list] for attr, pw_list in seg_pw.items()},
            'preferred': preferred,
        })

    return {
        'segments': segments,
        'scatter_data': scatter_data,
        'centroid_data': centroid_data,
        'pca_explained': [safe_float(v * 100) for v in explained_var],
    }


# ══════════════════════════════════════════════════════════════
# Market Simulation (What-if)
# ══════════════════════════════════════════════════════════════

def market_simulation(
    profiles: List[Dict[str, Any]],
    beta: np.ndarray,
    design_cols: List[str],
    attribute_map: Dict[str, List[str]],
    ref_levels: Dict[str, str],
):
    """
    Simulate market share for hypothetical product profiles.

    Each profile is a dict: { attribute_name: level_value, label: "Product A" }
    Uses logit share rule: share_j = exp(V_j) / Σ_k exp(V_k)
    """
    if not profiles or len(profiles) < 2:
        return None

    coef_map = {col: beta[i] for i, col in enumerate(design_cols)}
    utilities = []

    for prof in profiles:
        v = 0.0
        for attr, levels in attribute_map.items():
            level_val = str(prof.get(attr, ref_levels[attr])).strip()
            if level_val == ref_levels[attr]:
                continue  # reference = 0
            col = f"{attr}__{level_val}"
            if col in coef_map:
                v += coef_map[col]
        utilities.append(v)

    utilities = np.array(utilities)
    max_u = np.max(utilities)
    exp_u = np.exp(utilities - max_u)
    shares = exp_u / np.sum(exp_u)

    sim_results = []
    for i, prof in enumerate(profiles):
        sim_results.append({
            'label': prof.get('label', f'Profile {i+1}'),
            'utility': safe_float(utilities[i]),
            'share_pct': safe_float(shares[i] * 100),
            'attributes': {attr: str(prof.get(attr, ref_levels[attr])) for attr in attribute_map},
        })

    return sim_results


# ══════════════════════════════════════════════════════════════
# Main Endpoint
# ══════════════════════════════════════════════════════════════

@router.post("/conjoint")
async def conjoint_analysis(request: ConjointRequest):
    try:
        # ── 1. Parse Data ──
        df, design_cols, attribute_map, ref_levels, data_summary = parse_conjoint_data(
            raw_data=request.data,
            response_col=request.responseCol,
            task_col=request.taskCol,
            resp_col=request.respCol,
            alt_col=request.altCol,
            attribute_cols=request.attributeCols,
            none_option=request.noneOption,
        )

        # ── 2. Fit Model ──
        fit = conditional_logit_fit(
            df=df,
            design_cols=design_cols,
            response_col=request.responseCol,
            choice_id_col='_choice_id',
        )

        # ── 3. Part-worths & Importance ──
        partworths, importance = compute_partworths_and_importance(
            beta=fit['beta'],
            se=fit['se'],
            z_values=fit['z_values'],
            p_values=fit['p_values'],
            design_cols=design_cols,
            attribute_map=attribute_map,
            ref_levels=ref_levels,
        )

        # ── 4. WTP (if price column specified) ──
        wtp_results = None
        beta_price = None
        if request.priceCol and request.priceCol in df.columns:
            attr_cols_no_price = [a for a in request.attributeCols if a != request.priceCol]
            wtp_results, beta_price = compute_wtp_cbc(
                df, request.responseCol, '_choice_id', request.priceCol,
                attr_cols_no_price, attribute_map, ref_levels,
            )

        # ── 5. Segmentation ──
        seg_result = None
        if request.respCol:
            seg_result = cbc_segmentation(
                df, design_cols, request.responseCol, request.respCol,
                '_choice_id', attribute_map, ref_levels,
                n_segments=request.nSegments,
            )

        # ── 6. Market Simulation ──
        simulation = None
        if request.simulationProfiles and len(request.simulationProfiles) >= 2:
            simulation = market_simulation(
                profiles=request.simulationProfiles,
                beta=fit['beta'],
                design_cols=design_cols,
                attribute_map=attribute_map,
                ref_levels=ref_levels,
            )

        # ── 5. Chart Data for Frontend (recharts) ──

        # Part-worth chart data: flat list [{attribute, level, coef, se, significant}]
        pw_chart_data = []
        for attr, pw_list in partworths.items():
            for pw in pw_list:
                pw_chart_data.append({
                    'attribute': attr,
                    'level': pw['level'],
                    'coef': safe_float(pw['coef']),
                    'se': safe_float(pw['se']),
                    'ci_lower': safe_float(pw['coef'] - 1.96 * pw['se']),
                    'ci_upper': safe_float(pw['coef'] + 1.96 * pw['se']),
                    'significant': pw['significant'],
                    'is_reference': pw['is_reference'],
                    'label': f"{attr}: {pw['level']}",
                })

        # Importance chart data
        imp_chart_data = [
            {'attribute': attr, 'importance': safe_float(imp)}
            for attr, imp in sorted(importance.items(), key=lambda x: -x[1])
        ]

        # Coefficient table data
        coef_table = []
        for attr, pw_list in partworths.items():
            for pw in pw_list:
                coef_table.append({
                    'attribute': attr,
                    'level': pw['level'],
                    'coef': safe_float(pw['coef']),
                    'se': safe_float(pw['se']),
                    'z': safe_float(pw['z']),
                    'p': safe_float(pw['p']),
                    'significant': pw['significant'],
                    'is_reference': pw['is_reference'],
                    'ci_lower': safe_float(pw['coef'] - 1.96 * pw['se']),
                    'ci_upper': safe_float(pw['coef'] + 1.96 * pw['se']),
                })

        # WTP chart data
        wtp_chart_data = None
        if wtp_results:
            wtp_chart_data = [
                {
                    'attribute': w['attribute'],
                    'level': w['level'],
                    'wtp': w['wtp'],
                    'wtp_ci_lower': w['wtp_ci_lower'],
                    'wtp_ci_upper': w['wtp_ci_upper'],
                    'is_reference': w['is_reference'],
                    'label': f"{w['attribute']}: {w['level']}",
                }
                for w in wtp_results
            ]

        # Segmentation chart data
        seg_scatter_data = None
        seg_centroid_data = None
        seg_pca_explained = None
        segments_list = None
        if seg_result:
            segments_list = seg_result['segments']
            seg_scatter_data = seg_result['scatter_data']
            seg_centroid_data = seg_result['centroid_data']
            seg_pca_explained = seg_result['pca_explained']

        # ── 8. Build Response ──
        results = {
            'model_info': {
                'type': 'Conditional Logit (MLE)',
                'n_params': fit['n_params'],
                'converged': fit['converged'],
            },
            'fit_statistics': {
                'log_likelihood': safe_float(fit['ll_model']),
                'log_likelihood_null': safe_float(fit['ll_null']),
                'aic': safe_float(fit['aic']),
                'aicc': safe_float(fit['aicc']),
                'bic': safe_float(fit['bic']),
                'mcfadden_r2': safe_float(fit['mcfadden_r2']),
                'adj_mcfadden_r2': safe_float(fit['adj_mcfadden_r2']),
                'hit_rate': safe_float(fit['hit_rate']),
            },
            'data_summary': data_summary,
            'attribute_map': attribute_map,
            'ref_levels': ref_levels,
            'partworths': {attr: [_to_native(pw) for pw in pw_list] for attr, pw_list in partworths.items()},
            'importance': {attr: safe_float(v) for attr, v in importance.items()},
            'wtp': _to_native(wtp_results) if wtp_results else None,
            'beta_price': beta_price,
            'segments': _to_native(segments_list) if segments_list else None,
            'simulation': _to_native(simulation) if simulation else None,
            'charts': {
                'partworth_data': _to_native(pw_chart_data),
                'importance_data': _to_native(imp_chart_data),
                'coefficient_table': _to_native(coef_table),
                'wtp_data': _to_native(wtp_chart_data) if wtp_chart_data else None,
                'segment_scatter': _to_native(seg_scatter_data) if seg_scatter_data else None,
                'segment_centroids': _to_native(seg_centroid_data) if seg_centroid_data else None,
                'segment_pca_explained': _to_native(seg_pca_explained) if seg_pca_explained else None,
            },
        }

        return _to_native({'results': results})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")
