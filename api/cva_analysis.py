# ══════════════════════════════════════════════════════════════
# CVA (Traditional Conjoint) Analysis
# ══════════════════════════════════════════════════════════════
# Registration in main.py:
#   from api.cva_analysis import router as cva_router
#   app.include_router(cva_router, prefix="/api/analysis", tags=["CVA Conjoint Analysis"])
# ══════════════════════════════════════════════════════════════

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import numpy as np
from scipy import stats as scipy_stats
import warnings
import traceback

warnings.filterwarnings('ignore')

router = APIRouter()


# ══════════════════════════════════════════════════════════════
# Request Model
# ══════════════════════════════════════════════════════════════

class CVARequest(BaseModel):
    data: List[Dict[str, Any]]
    ratingCol: str              # column: rating / preference score
    attributeCols: List[str]    # columns: attribute level columns
    respCol: Optional[str] = None   # column: respondent ID (optional, for individual-level)
    profileCol: Optional[str] = None  # column: profile / card ID (optional)
    priceCol: Optional[str] = None    # price column for WTP (must be numeric-parseable)
    nSegments: int = 3                # number of segments for K-means
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
# Data Parser — build design matrix for OLS
# ══════════════════════════════════════════════════════════════

def parse_cva_data(
    raw_data: List[Dict[str, Any]],
    rating_col: str,
    attribute_cols: List[str],
    resp_col: Optional[str],
    profile_col: Optional[str],
):
    """
    Parse CVA survey data into design matrix for OLS regression.

    Each row = one profile rating from one respondent.
    Dummy-code attribute levels (first level = reference).

    Returns:
        df: DataFrame with dummy columns
        design_cols: list of dummy column names
        attribute_map: { attr: [levels...] }
        ref_levels: { attr: reference_level }
        data_summary: dict
    """
    df = pd.DataFrame(raw_data)

    # Validate
    if rating_col not in df.columns:
        raise HTTPException(status_code=400, detail=f"Rating column '{rating_col}' not found. Available: {list(df.columns)}")

    missing_attrs = [c for c in attribute_cols if c not in df.columns]
    if missing_attrs:
        raise HTTPException(status_code=400, detail=f"Attribute columns not found: {missing_attrs}")

    # Parse rating
    df[rating_col] = pd.to_numeric(df[rating_col], errors='coerce')
    df = df.dropna(subset=[rating_col])

    if len(df) < 5:
        raise HTTPException(status_code=400, detail="Need at least 5 rated profiles")

    # Build attribute map and dummies
    attribute_map = {}
    ref_levels = {}
    design_cols = []

    for attr in attribute_cols:
        df[attr] = df[attr].astype(str).str.strip()
        levels = sorted(df[attr].unique().tolist())

        if len(levels) < 2:
            continue

        attribute_map[attr] = levels
        ref_levels[attr] = levels[0]

        for lvl in levels[1:]:
            col_name = f"{attr}__{lvl}"
            df[col_name] = (df[attr] == lvl).astype(float)
            design_cols.append(col_name)

    if len(design_cols) == 0:
        raise HTTPException(status_code=400, detail="No valid attribute levels found for modeling.")

    # Data summary
    n_respondents = df[resp_col].nunique() if resp_col and resp_col in df.columns else 1
    n_profiles = df[profile_col].nunique() if profile_col and profile_col in df.columns else len(df)
    rating_min = float(df[rating_col].min())
    rating_max = float(df[rating_col].max())
    rating_mean = float(df[rating_col].mean())
    rating_std = float(df[rating_col].std())

    data_summary = {
        'n_respondents': int(n_respondents),
        'n_profiles': int(n_profiles),
        'n_observations': len(df),
        'n_attributes': len(attribute_map),
        'n_parameters': len(design_cols),
        'rating_min': safe_float(rating_min),
        'rating_max': safe_float(rating_max),
        'rating_mean': safe_float(rating_mean),
        'rating_std': safe_float(rating_std),
        'rating_scale': f"{rating_min:.0f}–{rating_max:.0f}",
    }

    return df, design_cols, attribute_map, ref_levels, data_summary

# ══════════════════════════════════════════════════════════════
# OLS / Ridge Regression for CVA
# ══════════════════════════════════════════════════════════════

def ols_fit(df, design_cols, rating_col):
    """
    Fit regression: rating = intercept + Σ beta_k * X_k + error

    Strategy:
    1. Check collinearity via condition number
    2. If clean (cond < 1e8): standard OLS
    3. If collinear: Ridge regression with GCV-optimized λ
       β_ridge = (X'X + λI)^-1 X'y

    Returns dict with coefficients, SE, t-values, p-values, R², F-test.
    """
    X = df[design_cols].values.astype(np.float64)
    y = df[rating_col].values.astype(np.float64)
    n, k = X.shape

    # Add intercept
    X_aug = np.column_stack([np.ones(n), X])
    k_aug = k + 1

    XtX = X_aug.T @ X_aug
    Xty = X_aug.T @ y

    # Collinearity detection
    rank = np.linalg.matrix_rank(X_aug)
    cond_number = np.linalg.cond(XtX)
    is_collinear = (rank < k_aug) or (cond_number > 1e8)
    collinear_warning = None
    ridge_lambda = 0.0
    method = 'OLS'

    if is_collinear:
        # Auto-select Ridge λ via GCV (Generalized Cross-Validation)
        # Try a range of λ values, pick the one with lowest GCV score
        lambdas = np.logspace(-4, 2, 30)
        I_aug = np.eye(k_aug)
        I_aug[0, 0] = 0  # Don't penalize intercept

        best_gcv = np.inf
        best_lam = lambdas[0]

        for lam in lambdas:
            try:
                A = XtX + lam * I_aug
                beta_try = np.linalg.solve(A, Xty)
                H = X_aug @ np.linalg.solve(A, X_aug.T)
                y_hat_try = X_aug @ beta_try
                resid = y - y_hat_try
                trace_H = np.trace(H)
                denom = (1 - trace_H / n) ** 2
                if denom > 1e-10:
                    gcv = np.mean(resid ** 2) / denom
                    if gcv < best_gcv:
                        best_gcv = gcv
                        best_lam = lam
            except Exception:
                continue

        ridge_lambda = best_lam
        method = 'Ridge'

        A = XtX + ridge_lambda * I_aug
        beta_hat = np.linalg.solve(A, Xty)

        collinear_warning = (
            f"Multicollinearity detected (condition number: {cond_number:.1e}, rank {rank}/{k_aug}). "
            f"Ridge regression applied (λ={ridge_lambda:.4f}) to stabilize estimates. "
            f"Coefficients are slightly shrunk but SE and p-values are reliable."
        )
    else:
        # Standard OLS
        beta_hat = np.linalg.solve(XtX, Xty)

    # Predictions and residuals
    y_hat = X_aug @ beta_hat
    residuals = y - y_hat

    # Sum of squares
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    ss_reg = ss_tot - ss_res

    # R² and adjusted R²
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0
    effective_df = max(n - k_aug, 1)
    adj_r_squared = 1 - (1 - r_squared) * (n - 1) / effective_df

    # Standard errors
    mse = ss_res / effective_df
    try:
        if is_collinear:
            I_aug = np.eye(k_aug)
            I_aug[0, 0] = 0
            A = XtX + ridge_lambda * I_aug
            A_inv = np.linalg.inv(A)
            # Ridge SE: Var(β_ridge) ≈ σ² * A_inv @ XtX @ A_inv
            cov_matrix = mse * (A_inv @ XtX @ A_inv)
        else:
            cov_matrix = mse * np.linalg.inv(XtX)
        diag_vals = np.maximum(np.diag(cov_matrix), 1e-12)
        se = np.sqrt(diag_vals)
        se = np.where(np.isfinite(se), se, 0.0)
    except Exception:
        se = np.full(k_aug, 0.0)

    t_values = np.where(se > 1e-10, beta_hat / se, 0.0)
    p_values = np.where(se > 1e-10,
        2 * (1 - scipy_stats.t.cdf(np.abs(t_values), df=effective_df)),
        1.0)

    # F-test (overall model)
    df_reg = k
    df_res = effective_df
    if df_reg > 0 and ss_res > 0:
        f_stat = (ss_reg / df_reg) / (ss_res / df_res)
        f_pvalue = 1 - scipy_stats.f.cdf(f_stat, df_reg, df_res)
    else:
        f_stat = 0
        f_pvalue = 1.0

    # AIC / BIC
    ll = -n / 2 * (np.log(2 * np.pi) + np.log(max(ss_res / n, 1e-12)) + 1)
    aic = 2 * k_aug - 2 * ll
    bic = k_aug * np.log(n) - 2 * ll
    aicc = aic + (2 * k_aug * (k_aug + 1)) / max(n - k_aug - 1, 1)

    # RMSE / MAE
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(residuals))

    return {
        'intercept': float(beta_hat[0]),
        'intercept_se': float(se[0]),
        'beta': beta_hat[1:],
        'se': se[1:],
        't_values': t_values[1:],
        'p_values': p_values[1:],
        'r_squared': safe_float(r_squared),
        'adj_r_squared': safe_float(adj_r_squared),
        'f_stat': safe_float(f_stat),
        'f_pvalue': safe_float(f_pvalue),
        'rmse': safe_float(rmse),
        'mae': safe_float(mae),
        'aic': safe_float(aic),
        'aicc': safe_float(aicc),
        'bic': safe_float(bic),
        'log_likelihood': safe_float(ll),
        'ss_reg': safe_float(ss_reg),
        'ss_res': safe_float(ss_res),
        'ss_tot': safe_float(ss_tot),
        'n': n,
        'k': k,
        'rank': rank,
        'method': method,
        'ridge_lambda': safe_float(ridge_lambda),
        'collinear_warning': collinear_warning,
        'residuals': residuals.tolist(),
        'y_hat': y_hat.tolist(),
    }


# ══════════════════════════════════════════════════════════════
# Part-worths & Relative Importance
# ══════════════════════════════════════════════════════════════

def compute_cva_partworths_and_importance(
    beta: np.ndarray,
    se: np.ndarray,
    t_values: np.ndarray,
    p_values: np.ndarray,
    design_cols: List[str],
    attribute_map: Dict[str, List[str]],
    ref_levels: Dict[str, str],
):
    """
    Build part-worth utilities and relative importance.
    Same logic as CBC but uses t-values instead of z-values.
    """
    coef_lookup = {}
    for i, col in enumerate(design_cols):
        coef_lookup[col] = {
            'coef': float(beta[i]),
            'se': float(se[i]),
            't': float(t_values[i]),
            'p': float(p_values[i]),
        }

    partworths = {}
    importance_ranges = {}

    for attr, levels in attribute_map.items():
        ref = ref_levels[attr]
        attr_pw = [{
            'level': ref,
            'coef': 0.0,
            'se': 0.0,
            't': 0.0,
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
                    't': c['t'],
                    'p': c['p'],
                    'significant': c['p'] < 0.05,
                    'is_reference': False,
                })
            else:
                attr_pw.append({
                    'level': lvl, 'coef': 0.0, 'se': 0.0,
                    't': 0.0, 'p': 1.0, 'significant': False, 'is_reference': False,
                })

        partworths[attr] = attr_pw
        coefs = [pw['coef'] for pw in attr_pw]
        importance_ranges[attr] = max(coefs) - min(coefs)

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
# Individual-level Analysis (if respondent column provided)
# ══════════════════════════════════════════════════════════════

def individual_level_analysis(
    df, design_cols, rating_col, resp_col, attribute_map, ref_levels,
):
    """
    Run OLS per respondent and compute individual part-worths.
    Returns summary stats of individual-level importance.
    """
    if not resp_col or resp_col not in df.columns:
        return None

    respondents = df[resp_col].unique()
    if len(respondents) < 3:
        return None

    individual_importances = {attr: [] for attr in attribute_map}

    for resp in respondents:
        rdf = df[df[resp_col] == resp]
        if len(rdf) < len(design_cols) + 2:
            continue  # not enough data for this respondent

        try:
            fit = ols_fit(rdf, design_cols, rating_col)
            _, imp = compute_cva_partworths_and_importance(
                fit['beta'], fit['se'], fit['t_values'], fit['p_values'],
                design_cols, attribute_map, ref_levels,
            )
            for attr, val in imp.items():
                individual_importances[attr].append(val)
        except Exception:
            continue

    if all(len(v) == 0 for v in individual_importances.values()):
        return None

    summary = {}
    for attr in attribute_map:
        vals = individual_importances[attr]
        if len(vals) > 0:
            summary[attr] = {
                'mean': safe_float(np.mean(vals)),
                'std': safe_float(np.std(vals)),
                'min': safe_float(np.min(vals)),
                'max': safe_float(np.max(vals)),
                'median': safe_float(np.median(vals)),
                'n': len(vals),
            }

    return summary

# ══════════════════════════════════════════════════════════════
# Price Parser
# ══════════════════════════════════════════════════════════════

import re

def parse_price_value(val) -> Optional[float]:
    """Extract numeric value from price string like '$799', '1,200원', '€50.99'."""
    if val is None:
        return None
    s = str(val).strip()
    cleaned = re.sub(r'[^\d.]', '', s)
    try:
        f = float(cleaned)
        return f if f > 0 else None
    except (ValueError, TypeError):
        return None


# ══════════════════════════════════════════════════════════════
# WTP (Willingness to Pay) — OLS-based
# ══════════════════════════════════════════════════════════════

def compute_wtp_cva(
    df, rating_col, price_col,
    attribute_cols_no_price, attribute_map, ref_levels,
):
    """
    Compute WTP with price as continuous variable in OLS.
    WTP_k = -beta_k / beta_price
    """
    # Build design cols without price dummies, add continuous price
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

    price_numeric = df[price_col].apply(parse_price_value)
    if price_numeric.isna().sum() > len(df) * 0.5:
        return None, None

    df['_price_numeric'] = price_numeric.fillna(price_numeric.median())
    design_cols_wtp.append('_price_numeric')

    fit_wtp = ols_fit(df, design_cols_wtp, rating_col)
    beta_wtp = fit_wtp['beta']
    se_wtp = fit_wtp['se']

    beta_price = beta_wtp[-1]
    se_price = se_wtp[-1]

    if abs(beta_price) < 1e-10:
        return None, None

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
# Segmentation (K-means on individual-level part-worths)
# ══════════════════════════════════════════════════════════════

from sklearn.cluster import KMeans

def cva_segmentation(
    df, design_cols, rating_col, resp_col,
    attribute_map, ref_levels, n_segments=3,
):
    """
    1. Per-respondent OLS → individual part-worth vectors
    2. K-means clustering
    3. PCA for 2D cluster visualization
    4. Segment-level importance, part-worths, and preferred levels
    """
    from sklearn.decomposition import PCA

    if not resp_col or resp_col not in df.columns:
        return None

    respondents = df[resp_col].unique()
    if len(respondents) < n_segments * 2:
        return None

    agg_fit = ols_fit(df, design_cols, rating_col)
    agg_beta = agg_fit['beta']

    indiv_betas = []
    resp_ids = []

    for resp in respondents:
        rdf = df[df[resp_col] == resp]
        if len(rdf) < len(design_cols) + 2:
            indiv_betas.append(agg_beta.copy())
            resp_ids.append(resp)
            continue
        try:
            rfit = ols_fit(rdf, design_cols, rating_col)
            beta_clipped = np.clip(rfit['beta'], -10, 10)
            indiv_betas.append(beta_clipped)
            resp_ids.append(resp)
        except Exception:
            indiv_betas.append(agg_beta.copy())
            resp_ids.append(resp)

    beta_matrix = np.array(indiv_betas)
    beta_matrix = np.nan_to_num(beta_matrix, nan=0.0, posinf=5.0, neginf=-5.0)

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
    centroids_pca = pca.transform(kmeans.cluster_centers_)

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

    segments = []
    for seg_id in range(n_segments):
        mask = labels == seg_id
        seg_betas = beta_matrix[mask]
        seg_mean = seg_betas.mean(axis=0)
        n_resp = int(mask.sum())

        seg_pw, seg_importance = compute_cva_partworths_and_importance(
            seg_mean,
            np.zeros_like(seg_mean),
            np.zeros_like(seg_mean),
            np.ones_like(seg_mean),
            design_cols, attribute_map, ref_levels,
        )

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
# Market Simulation (What-if) — OLS-based
# ══════════════════════════════════════════════════════════════

def cva_market_simulation(
    profiles: List[Dict[str, Any]],
    intercept: float,
    beta: np.ndarray,
    design_cols: List[str],
    attribute_map: Dict[str, List[str]],
    ref_levels: Dict[str, str],
):
    """
    Simulate market share for hypothetical product profiles using CVA utilities.

    Predicted rating for each profile, then share via logit transformation
    of predicted utilities: share_j = exp(V_j) / Σ exp(V_k)
    """
    if not profiles or len(profiles) < 2:
        return None

    coef_map = {col: beta[i] for i, col in enumerate(design_cols)}
    predicted_ratings = []

    for prof in profiles:
        v = intercept
        for attr, levels in attribute_map.items():
            level_val = str(prof.get(attr, ref_levels[attr])).strip()
            if level_val == ref_levels[attr]:
                continue
            col = f"{attr}__{level_val}"
            if col in coef_map:
                v += coef_map[col]
        predicted_ratings.append(v)

    ratings = np.array(predicted_ratings)

    # Logit-based share
    max_r = np.max(ratings)
    exp_r = np.exp(ratings - max_r)
    shares = exp_r / np.sum(exp_r)

    sim_results = []
    for i, prof in enumerate(profiles):
        sim_results.append({
            'label': prof.get('label', f'Profile {i+1}'),
            'predicted_rating': safe_float(ratings[i]),
            'share_pct': safe_float(shares[i] * 100),
            'attributes': {attr: str(prof.get(attr, ref_levels[attr])) for attr in attribute_map},
        })

    return sim_results


# ══════════════════════════════════════════════════════════════
# Main Endpoint
# ══════════════════════════════════════════════════════════════

@router.post("/cva")
async def cva_analysis(request: CVARequest):
    try:
        # ── 1. Parse Data ──
        df, design_cols, attribute_map, ref_levels, data_summary = parse_cva_data(
            raw_data=request.data,
            rating_col=request.ratingCol,
            attribute_cols=request.attributeCols,
            resp_col=request.respCol,
            profile_col=request.profileCol,
        )

        # ── 2. Fit OLS ──
        fit = ols_fit(df, design_cols, request.ratingCol)

        # ── 3. Part-worths & Importance ──
        partworths, importance = compute_cva_partworths_and_importance(
            beta=fit['beta'],
            se=fit['se'],
            t_values=fit['t_values'],
            p_values=fit['p_values'],
            design_cols=design_cols,
            attribute_map=attribute_map,
            ref_levels=ref_levels,
        )

        # ── 4. Individual-level (if respondent column) ──
        indiv = individual_level_analysis(
            df, design_cols, request.ratingCol,
            request.respCol, attribute_map, ref_levels,
        )

        # ── 5. WTP (if price column specified) ──
        wtp_results = None
        beta_price = None
        if request.priceCol and request.priceCol in df.columns:
            attr_cols_no_price = [a for a in request.attributeCols if a != request.priceCol]
            wtp_results, beta_price = compute_wtp_cva(
                df, request.ratingCol, request.priceCol,
                attr_cols_no_price, attribute_map, ref_levels,
            )

        # ── 6. Segmentation ──
        seg_result = None
        if request.respCol:
            seg_result = cva_segmentation(
                df, design_cols, request.ratingCol, request.respCol,
                attribute_map, ref_levels,
                n_segments=request.nSegments,
            )

        # ── 7. Market Simulation ──
        simulation = None
        if request.simulationProfiles and len(request.simulationProfiles) >= 2:
            simulation = cva_market_simulation(
                profiles=request.simulationProfiles,
                intercept=fit['intercept'],
                beta=fit['beta'],
                design_cols=design_cols,
                attribute_map=attribute_map,
                ref_levels=ref_levels,
            )

        # ── 6. Chart Data for Frontend (recharts) ──

        # Part-worth chart data
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

        # Residual data (for diagnostics chart)
        residual_data = []
        for i in range(min(len(fit['residuals']), 500)):  # cap at 500 points
            residual_data.append({
                'y_hat': safe_float(fit['y_hat'][i]),
                'residual': safe_float(fit['residuals'][i]),
            })

        # Coefficient table
        coef_table = []
        for attr, pw_list in partworths.items():
            for pw in pw_list:
                coef_table.append({
                    'attribute': attr,
                    'level': pw['level'],
                    'coef': safe_float(pw['coef']),
                    'se': safe_float(pw['se']),
                    't': safe_float(pw['t']),
                    'p': safe_float(pw['p']),
                    'significant': pw['significant'],
                    'is_reference': pw['is_reference'],
                    'ci_lower': safe_float(pw['coef'] - 1.96 * pw['se']),
                    'ci_upper': safe_float(pw['coef'] + 1.96 * pw['se']),
                })

        # Individual importance chart data (if available)
        indiv_chart_data = None
        if indiv:
            indiv_chart_data = [
                {
                    'attribute': attr,
                    'mean': vals['mean'],
                    'std': vals['std'],
                    'min': vals['min'],
                    'max': vals['max'],
                    'median': vals['median'],
                }
                for attr, vals in sorted(indiv.items(), key=lambda x: -x[1]['mean'])
            ]

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

        # ── 9. Build Response ──
        results = {
            'model_info': {
                'type': f"{fit['method']} Regression (Traditional Conjoint / CVA)",
                'method': fit['method'],
                'n_params': fit['k'],
                'intercept': safe_float(fit['intercept']),
                'intercept_se': safe_float(fit['intercept_se']),
                'ridge_lambda': fit.get('ridge_lambda', 0.0),
                'collinear_warning': fit.get('collinear_warning'),
            },
            'fit_statistics': {
                'r_squared': safe_float(fit['r_squared']),
                'adj_r_squared': safe_float(fit['adj_r_squared']),
                'f_stat': safe_float(fit['f_stat']),
                'f_pvalue': safe_float(fit['f_pvalue']),
                'rmse': safe_float(fit['rmse']),
                'mae': safe_float(fit['mae']),
                'aic': safe_float(fit['aic']),
                'aicc': safe_float(fit['aicc']),
                'bic': safe_float(fit['bic']),
                'log_likelihood': safe_float(fit['log_likelihood']),
            },
            'data_summary': data_summary,
            'attribute_map': attribute_map,
            'ref_levels': ref_levels,
            'partworths': {attr: [_to_native(pw) for pw in pw_list] for attr, pw_list in partworths.items()},
            'importance': {attr: safe_float(v) for attr, v in importance.items()},
            'individual_importance': _to_native(indiv) if indiv else None,
            'wtp': _to_native(wtp_results) if wtp_results else None,
            'beta_price': beta_price,
            'segments': _to_native(segments_list) if segments_list else None,
            'simulation': _to_native(simulation) if simulation else None,
            'charts': {
                'partworth_data': _to_native(pw_chart_data),
                'importance_data': _to_native(imp_chart_data),
                'coefficient_table': _to_native(coef_table),
                'residual_data': _to_native(residual_data),
                'individual_importance_data': _to_native(indiv_chart_data) if indiv_chart_data else None,
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
