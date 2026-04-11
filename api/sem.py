"""
Structural Equation Modeling (SEM) Router for FastAPI
Using semopy for proper Maximum Likelihood estimation
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import io
import base64
import logging
from contextlib import contextmanager
from scipy import stats

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================
# Fix 2: scoped matplotlib context manager
# (no global rcParams mutation)
# ============================================

@contextmanager
def _plot_context():
    """Apply plot style only within this context; restore original state after."""
    original = {
        'font.family': plt.rcParams.get('font.family'),
        'axes.unicode_minus': plt.rcParams.get('axes.unicode_minus'),
    }
    try:
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        yield
    finally:
        for key, val in original.items():
            plt.rcParams[key] = val


# ============================================
# Request / Response Models
# ============================================

class SEMRequest(BaseModel):
    data: List[Dict[str, Any]]
    model_spec: str             # lavaan-style syntax
    estimator: str = "MLW"     # MLW, ML, GLS, DWLS
    bootstrap_n: int = 0       # 0 = off; 200–1000 recommended for indirect CIs
    bootstrap_method: str = "ols_approximate"
    # "ols_approximate": fast, bivariate OLS per resample – NOT full SEM bootstrap.
    #                    Exploratory use only; do not cite as SEM bootstrap.
    # "sem_full":        full semopy re-fit per resample (Bollen & Stine 1990).
    #                    Publication-quality; much slower.


# ============================================
# Utility Functions
# ============================================

def _to_native(obj):
    """Convert numpy/pandas types to JSON-serializable Python types."""
    if obj is None:
        return None
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (float, np.floating, np.float64)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return [_to_native(x) for x in obj.tolist()]
    if isinstance(obj, str) and obj == '-':
        return None
    return obj


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return image_base64


def _safe_float(val) -> Optional[float]:
    """Safely convert to float."""
    if val is None or val == '-' or val == '':
        return None
    try:
        f = float(val)
        return None if (np.isnan(f) or np.isinf(f)) else f
    except (ValueError, TypeError):
        return None


# ============================================
# Model Syntax Parser
# ============================================

def parse_lavaan_syntax(model_spec: str) -> Dict[str, Any]:
    """Parse lavaan-style model specification."""
    latent_vars: Dict[str, List[str]] = {}
    regressions: List[Dict] = []
    covariances: List[Tuple] = []

    for raw_line in model_spec.strip().split('\n'):
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        if '#' in line:
            line = line[:line.index('#')].strip()

        if '=~' in line:
            parts = line.split('=~', 1)
            latent_name = parts[0].strip()
            indicators = [i.strip() for i in parts[1].split('+') if i.strip()]
            latent_vars[latent_name] = indicators
        elif '~~' in line:
            parts = line.split('~~', 1)
            covariances.append((parts[0].strip(), parts[1].strip()))
        elif '~' in line:
            parts = line.split('~', 1)
            dv = parts[0].strip()
            ivs = [iv.strip() for iv in parts[1].split('+') if iv.strip()]
            regressions.append({'dv': dv, 'ivs': ivs})

    return {
        'latent_vars': latent_vars,
        'regressions': regressions,
        'covariances': covariances,
    }


def convert_to_semopy_syntax(model_spec: str) -> str:
    """semopy supports lavaan syntax directly – no conversion needed."""
    return model_spec.strip()


# ============================================
# Fix 8: Comprehensive Validation
# ============================================

def _validate_sem_input(
    df: pd.DataFrame,
    parsed: Dict[str, Any],
    n_raw: int,
) -> Dict[str, List[str]]:
    """
    Run all pre-flight checks. Returns a dict with keys 'errors' and 'warnings'.
    Errors are fatal; warnings are surfaced to the caller but do not block execution.
    """
    errors: List[str] = []
    warnings: List[str] = []

    latent_vars = parsed['latent_vars']
    all_indicators = [ind for inds in latent_vars.values() for ind in inds]

    # ── minimum indicator count per latent ──────────────────────────────────
    for latent, indicators in latent_vars.items():
        n_ind = len(indicators)
        if n_ind == 0:
            errors.append(f"Latent variable '{latent}' has no indicators.")
        elif n_ind == 1:
            warnings.append(
                f"Latent variable '{latent}' has only 1 indicator (single-indicator latent). "
                "The factor loading is not identified without constraints."
            )
        elif n_ind == 2:
            warnings.append(
                f"Latent variable '{latent}' has only 2 indicators. "
                "Model is just-identified for this factor; fit indices may be uninformative."
            )

    # ── under-identification heuristic (ROUGH CHECK ONLY) ──────────────────
    #
    # IMPORTANT LIMITATION: True SEM identification requires rank conditions
    # on the information matrix (e.g. t-rule, rank rule, two-step rule) and
    # depends on the exact parameter structure, equality constraints, and
    # whether latent variances are fixed. The check below is a NECESSARY
    # but NOT SUFFICIENT condition: it catches obvious cases (more free
    # parameters than observed moments) but cannot guarantee that a model
    # passing this test is actually identified.
    #
    # Formal identification should be verified by:
    #   1. Checking whether semopy converges and returns a positive-definite
    #      information matrix (convergence failure often signals non-identification)
    #   2. Using dedicated tools (e.g. lavaan's lavTestIdentify() in R)
    #   3. Consulting SEM identification rules in Bollen (1989) or Kenny (2020)
    #
    # This heuristic counts:
    #   obs_moments  = p*(p+1)/2  (unique elements of the observed covariance matrix)
    #   free_params  = n_loadings + n_error_variances + n_structural_paths
    #                  (lower bound; covariances among latents are not counted)
    n_regressions = sum(len(r['ivs']) for r in parsed['regressions'])
    n_loadings = sum(max(len(v), 1) for v in latent_vars.values())
    n_error_vars = n_loadings  # one error variance per indicator (lower bound)
    free_params = n_loadings + n_error_vars + n_regressions
    obs_moments = len(all_indicators) * (len(all_indicators) + 1) // 2
    if obs_moments < free_params:
        warnings.append(
            f"Possible under-identification (heuristic only): "
            f"observed moments ({obs_moments}) < estimated free parameters ({free_params}). "
            "This is a rough necessary condition check – it does not guarantee identification. "
            "Confirm with semopy convergence diagnostics or a dedicated identification tool."
        )
    elif obs_moments == free_params:
        warnings.append(
            f"Model appears just-identified (obs_moments = free_params = {obs_moments}). "
            "Fit indices (CFI, RMSEA) will be undefined. "
            "Note: this is a heuristic check only; formal identification analysis is advised."
        )

    # ── zero variance ────────────────────────────────────────────────────────
    for ind in all_indicators:
        if ind not in df.columns:
            continue
        col = pd.to_numeric(df[ind], errors='coerce').dropna()
        if col.empty:
            errors.append(f"Indicator '{ind}' is entirely non-numeric.")
        elif col.std() == 0:
            errors.append(f"Indicator '{ind}' has zero variance (constant column).")
        elif col.std() < 1e-6:
            warnings.append(
                f"Indicator '{ind}' has near-zero variance (std={col.std():.2e}). "
                "This may cause numerical instability in estimation."
            )

    # ── Fix 3: sample size as warning, not hard error ───────────────────────
    N_WARN_SOFT = 100   # under this → informational note
    N_WARN_HARD = 30    # under this → strong warning
    n = len(df)
    if n < N_WARN_HARD:
        warnings.append(
            f"Very small sample (N={n}). SEM estimates will be highly unreliable. "
            "Interpret results with extreme caution."
        )
    elif n < N_WARN_SOFT:
        warnings.append(
            f"Small sample (N={n}; recommended ≥100 for SEM). "
            "Parameter estimates and fit indices may be unstable."
        )
    # Also warn when N dropped substantially after listwise deletion
    if n < n_raw * 0.7:
        warnings.append(
            f"Listwise deletion removed {n_raw - n} rows ({(n_raw-n)/n_raw*100:.1f}%). "
            "Consider multiple imputation if MAR/MCAR assumptions are questionable."
        )

    return {'errors': errors, 'warnings': warnings}


# ============================================
# Fix 4: Proper SRMR
# ============================================

def _calculate_srmr(model, df: pd.DataFrame, parsed: Dict) -> Optional[float]:
    """
    Calculate SRMR using the true model-implied covariance matrix Σ(θ)
    from semopy's `model.predict_cov()` (or `model.Sigma` attribute).
    Falls back to None if the model does not expose Σ(θ).
    """
    try:
        all_indicators = [
            ind for inds in parsed['latent_vars'].values()
            for ind in inds if ind in df.columns
        ]
        if len(all_indicators) < 2:
            return None

        obs_df = df[all_indicators].dropna()
        if len(obs_df) < 2:
            return None

        # Observed covariance / correlation
        S = obs_df.cov().values
        std_vec = np.sqrt(np.diag(S))
        # Observed correlation matrix
        with np.errstate(divide='ignore', invalid='ignore'):
            obs_corr = np.where(
                std_vec[:, None] * std_vec[None, :] == 0, 0,
                S / (std_vec[:, None] * std_vec[None, :])
            )

        # Model-implied covariance from semopy
        sigma = None
        if hasattr(model, 'predict_cov'):
            sigma_df = model.predict_cov(obs_df)
            # keep only the observed-variable sub-matrix
            try:
                sigma = sigma_df.loc[all_indicators, all_indicators].values
            except Exception:
                sigma = sigma_df.values
        elif hasattr(model, 'Sigma'):
            sigma = model.Sigma
        elif hasattr(model, 'mx_sigma'):
            sigma = model.mx_sigma

        if sigma is None or sigma.shape != S.shape:
            logger.warning("SRMR: model-implied Σ not available; returning None")
            return None

        # Model-implied correlation
        sigma_std = np.sqrt(np.diag(sigma))
        with np.errstate(divide='ignore', invalid='ignore'):
            impl_corr = np.where(
                sigma_std[:, None] * sigma_std[None, :] == 0, 0,
                sigma / (sigma_std[:, None] * sigma_std[None, :])
            )

        # SRMR = sqrt( mean of squared residuals over lower triangle )
        n = len(all_indicators)
        sq_resid = []
        for i in range(n):
            for j in range(i + 1, n):
                sq_resid.append((obs_corr[i, j] - impl_corr[i, j]) ** 2)

        return float(np.sqrt(np.mean(sq_resid))) if sq_resid else None

    except Exception as exc:
        logger.warning("SRMR calculation failed: %s", exc)
        return None


# ============================================
# Fix 5: Proper R² from variance components
# ============================================

def _estimate_r_squared(
    latent: str,
    structural_model: List[Dict],
    variances_covariances: Dict[str, Any],
) -> float:
    """
    Compute R² for an endogenous latent variable using the formula:

        R² = 1 − (ψ_latent / Var_total)

    where ψ_latent is the residual (disturbance) variance from the
    model estimates and Var_total is the model-implied total variance.

    Falls back to the squared-standardised-beta sum approximation only
    when residual variance is not available.
    """
    try:
        # ── primary path: residual variance approach ─────────────────────
        residual_var = None
        for item in variances_covariances.get('latent_variances', []):
            if item.get('variable') == latent or item.get('var1') == latent:
                residual_var = _safe_float(item.get('estimate'))
                break

        # Total variance = residual + Σ(β_i² * Var_Xi) + cross-terms
        # For the standardised solution: Var(η) = 1, ψ = 1 − R²
        # Use std_estimate for residual variance when possible
        std_residual = None
        for item in variances_covariances.get('latent_variances', []):
            if item.get('variable') == latent or item.get('var1') == latent:
                std_residual = _safe_float(item.get('std_estimate'))
                break

        if std_residual is not None and 0.0 <= std_residual <= 1.0:
            return round(1.0 - std_residual, 6)

        # ── fallback: OLS-style β² sum (acknowledges correlation) ────────
        # Collect standardised betas for paths pointing to `latent`
        paths = [p for p in structural_model if p['to'] == latent]
        if not paths:
            return 0.0

        std_betas = np.array([
            p.get('std_estimate') or p.get('estimate') or 0.0
            for p in paths
        ])
        # Upper bound: sum of |β|² (ignores predictor correlations – may exceed 1)
        r2_approx = float(np.sum(std_betas ** 2))
        return min(round(r2_approx, 6), 1.0)

    except Exception as exc:
        logger.warning("R² calculation failed for '%s': %s", latent, exc)
        return 0.0


# ============================================
# Fix 6: Multi-step + multiple-mediator effects
# Fix 7: Bootstrap indirect effect CIs
# ============================================

def _trace_paths(
    from_var: str,
    adjacency: Dict[str, List[Dict]],
    visited: Optional[set] = None,
    max_depth: int = 8,
) -> List[List[Dict]]:
    """
    Enumerate all simple paths (no cycles) starting at `from_var`
    with length ≥ 2 (i.e. at least one intermediate step).
    Returns a list of path segments, each segment being a list of edge dicts.
    """
    if visited is None:
        visited = set()
    if from_var in visited or max_depth == 0:
        return []
    visited = visited | {from_var}

    all_paths: List[List[Dict]] = []
    for edge in adjacency.get(from_var, []):
        next_var = edge['to']
        # Single-step (direct) – will be handled separately; only yield multi-step
        for sub_path in _trace_paths(next_var, adjacency, visited, max_depth - 1):
            all_paths.append([edge] + sub_path)
        # Also record the single outgoing edge so we can detect mediator chains
        all_paths.append([edge])

    return all_paths


# ── Bootstrap method classification ─────────────────────────────────────────
# This module offers two distinct bootstrap strategies for indirect effect CIs.
#
# FAST (default, bootstrap_method="ols_approximate"):
#   Each bootstrap resample re-estimates structural paths via bivariate OLS,
#   NOT by re-fitting the full SEM. This is fast but has known limitations:
#     • Ignores latent variable measurement error
#     • Ignores covariance constraints from the measurement model
#     • Does NOT replicate Sobel/semopy-level SEM bootstrap
#   Use for: quick exploration, large N, simple mediation chains.
#   Cite as: "OLS-approximate bootstrap (not full SEM bootstrap)"
#
# FULL SEM (bootstrap_method="sem_full"):
#   Re-fits the complete semopy model on each bootstrap resample.
#   This IS statistically equivalent to standard SEM bootstrap (Bollen & Stine 1990).
#   Much slower (seconds per resample); recommended bootstrap_n ≤ 500.
#   Use for: publication, complex models, small-to-medium N.
#   Cite as: "Full SEM bootstrap, percentile CI (Bollen & Stine 1990)"
# ─────────────────────────────────────────────────────────────────────────────

def _calculate_effects(
    parsed: Dict,
    structural_model: List[Dict],
    df: Optional[pd.DataFrame] = None,
    bootstrap_n: int = 0,
    bootstrap_method: str = "ols_approximate",
    model_spec: Optional[str] = None,
    estimator: str = "MLW",
) -> Dict[str, Any]:
    """
    Compute direct, indirect (multi-step, multiple-mediator), and total effects.

    bootstrap_n > 0 activates bootstrap 95% percentile CIs for each indirect
    effect. The method used depends on bootstrap_method:

    "ols_approximate" (default, fast):
        Structural paths are re-estimated via bivariate OLS on each bootstrap
        resample. This is NOT a full SEM bootstrap – measurement error and
        latent variable constraints are ignored. Results are labelled
        accordingly in the output and should not be cited as SEM bootstrap.

    "sem_full" (slow, statistically rigorous):
        The complete semopy model is re-fitted on each bootstrap resample.
        Equivalent to standard SEM bootstrap (Bollen & Stine 1990). Use for
        publication-quality indirect effect inference.
    """
    effects: Dict[str, Any] = {
        'direct': [],
        'indirect': [],
        'total': [],
        'bootstrap_note': None,
        'bootstrap_method': bootstrap_method if bootstrap_n > 0 else None,
    }

    if not structural_model:
        return effects

    # Build adjacency map
    adjacency: Dict[str, List[Dict]] = {}
    for p in structural_model:
        fv = p['from']
        adjacency.setdefault(fv, []).append({
            'to': p['to'],
            'estimate': p.get('estimate') or 0.0,
            'std_estimate': p.get('std_estimate') or 0.0,
        })

    # ── Direct effects ──────────────────────────────────────────────────────
    for p in structural_model:
        effects['direct'].append({
            'from': p['from'],
            'to': p['to'],
            'estimate': p.get('estimate'),
            'std_estimate': p.get('std_estimate'),
        })

    # ── Indirect effects (multi-step, multiple mediators) ──────────────────
    all_starts = list(adjacency.keys())
    for start in all_starts:
        all_path_chains = _trace_paths(start, adjacency)
        for chain in all_path_chains:
            if len(chain) < 2:
                continue  # direct only – skip

            mediators = [chain[i]['to'] for i in range(len(chain) - 1)]
            end_var = chain[-1]['to']

            ind_est = 1.0
            ind_std = 1.0
            for edge in chain:
                ind_est *= edge['estimate']
                ind_std *= edge['std_estimate']

            path_str = (
                start + ' → '
                + ' → '.join(mediators)
                + ' → ' + end_var
            )

            effects['indirect'].append({
                'from': start,
                'mediators': mediators,
                'to': end_var,
                'path': path_str,
                'n_steps': len(chain),
                'estimate': ind_est,
                'std_estimate': ind_std,
                'ci_lower': None,
                'ci_upper': None,
                'sig_bootstrap': None,
            })

    # ── Bootstrap CIs for indirect effects ─────────────────────────────────
    if bootstrap_n > 0 and df is not None and len(df) >= 30:
        _boot_indirect: Dict[str, List[float]] = {
            ie['path']: [] for ie in effects['indirect']
        }
        rng = np.random.default_rng(seed=42)
        n_rows = len(df)
        n_converged = 0

        if bootstrap_method == "sem_full":
            # ── Full SEM bootstrap (Bollen & Stine 1990) ─────────────────
            # Re-fit the complete semopy model on each resample.
            # Statistically equivalent to standard SEM bootstrap.
            if model_spec is None:
                effects['bootstrap_note'] = (
                    "sem_full bootstrap requires model_spec; fell back to ols_approximate."
                )
                bootstrap_method = "ols_approximate"  # fall through below
            else:
                try:
                    from semopy import Model as _Model
                    for _ in range(bootstrap_n):
                        boot_df = df.iloc[
                            rng.integers(0, n_rows, size=n_rows)
                        ].reset_index(drop=True)
                        try:
                            boot_model = _Model(model_spec)
                            boot_result = boot_model.fit(boot_df, obj=estimator)
                            if not boot_result.success:
                                continue
                            boot_est = boot_model.inspect()
                            # Rebuild adjacency from this resample's estimates
                            boot_adj: Dict[str, List[Dict]] = {}
                            for _, row in boot_est.iterrows():
                                if row.get('op') == '~':
                                    fv, tv = str(row.get('rval','')), str(row.get('lval',''))
                                    val = _safe_float(row.get('Estimate')) or 0.0
                                    boot_adj.setdefault(fv, []).append(
                                        {'to': tv, 'estimate': val, 'std_estimate': val}
                                    )
                            for ie in effects['indirect']:
                                chain_nodes = [ie['from']] + ie['mediators'] + [ie['to']]
                                prod = 1.0
                                for k in range(len(chain_nodes) - 1):
                                    fv, tv = chain_nodes[k], chain_nodes[k + 1]
                                    edges = [e for e in boot_adj.get(fv, []) if e['to'] == tv]
                                    prod *= edges[0]['estimate'] if edges else 0.0
                                _boot_indirect[ie['path']].append(prod)
                            n_converged += 1
                        except Exception:
                            pass

                    for ie in effects['indirect']:
                        samples = np.array(_boot_indirect.get(ie['path'], []))
                        if len(samples) >= 50:
                            ie['ci_lower'] = float(np.percentile(samples, 2.5))
                            ie['ci_upper'] = float(np.percentile(samples, 97.5))
                            ie['sig_bootstrap'] = not (ie['ci_lower'] <= 0 <= ie['ci_upper'])

                    effects['bootstrap_note'] = (
                        f"Full SEM bootstrap (Bollen & Stine 1990): {bootstrap_n} resamples "
                        f"requested, {n_converged} converged. "
                        "95% percentile CIs. "
                        "Cite as: full SEM bootstrap, not OLS-approximate."
                    )
                except Exception as exc:
                    logger.warning("Full SEM bootstrap failed: %s", exc)
                    effects['bootstrap_note'] = (
                        f"Full SEM bootstrap failed ({exc}); no CIs computed. "
                        "Try bootstrap_method='ols_approximate' or reduce bootstrap_n."
                    )

        if bootstrap_method == "ols_approximate":
            # ── OLS-approximate bootstrap ─────────────────────────────────
            # IMPORTANT LIMITATION: structural paths are re-estimated via
            # bivariate OLS (np.polyfit) on each resample – NOT by re-fitting
            # the full SEM. This ignores measurement error, factor loadings,
            # and latent variable constraints. Results are faster but less
            # accurate than full SEM bootstrap. Do NOT cite as SEM bootstrap.
            try:
                for _ in range(bootstrap_n):
                    boot_df = df.iloc[
                        rng.integers(0, n_rows, size=n_rows)
                    ].reset_index(drop=True)
                    try:
                        boot_adj_ols: Dict[str, List[Dict]] = {}
                        for p in structural_model:
                            fv, tv = p['from'], p['to']
                            if fv in boot_df.columns and tv in boot_df.columns:
                                x = pd.to_numeric(boot_df[fv], errors='coerce').dropna()
                                y = pd.to_numeric(boot_df[tv], errors='coerce').dropna()
                                idx_common = x.index.intersection(y.index)
                                if len(idx_common) >= 3:
                                    slope, *_ = np.polyfit(x[idx_common], y[idx_common], 1)
                                    boot_adj_ols.setdefault(fv, []).append(
                                        {'to': tv, 'estimate': slope, 'std_estimate': slope}
                                    )
                        for ie in effects['indirect']:
                            chain_nodes = [ie['from']] + ie['mediators'] + [ie['to']]
                            prod = 1.0
                            for k in range(len(chain_nodes) - 1):
                                fv, tv = chain_nodes[k], chain_nodes[k + 1]
                                edges = [e for e in boot_adj_ols.get(fv, []) if e['to'] == tv]
                                prod *= edges[0]['estimate'] if edges else 0.0
                            _boot_indirect[ie['path']].append(prod)
                        n_converged += 1
                    except Exception:
                        pass

                for ie in effects['indirect']:
                    samples = np.array(_boot_indirect.get(ie['path'], []))
                    if len(samples) >= 50:
                        ie['ci_lower'] = float(np.percentile(samples, 2.5))
                        ie['ci_upper'] = float(np.percentile(samples, 97.5))
                        ie['sig_bootstrap'] = not (ie['ci_lower'] <= 0 <= ie['ci_upper'])

                effects['bootstrap_note'] = (
                    f"OLS-approximate bootstrap: {bootstrap_n} resamples, "
                    f"{n_converged} completed. "
                    "WARNING: structural paths estimated via bivariate OLS, NOT full SEM re-fit. "
                    "Measurement error and latent constraints are ignored. "
                    "These CIs are exploratory only – do not cite as SEM bootstrap. "
                    "Use bootstrap_method='sem_full' for publication-quality inference."
                )
            except Exception as exc:
                logger.warning("OLS-approximate bootstrap failed: %s", exc)
                effects['bootstrap_note'] = f"Bootstrap failed: {exc}"

    elif bootstrap_n > 0:
        effects['bootstrap_note'] = (
            "Bootstrap requested but skipped (N < 30 or no data provided)."
        )

    # ── Total effects ───────────────────────────────────────────────────────
    effect_pairs: Dict[Tuple, Dict] = {}
    for d in effects['direct']:
        key = (d['from'], d['to'])
        effect_pairs.setdefault(key, {'direct': 0.0, 'direct_std': 0.0,
                                       'indirect': 0.0, 'indirect_std': 0.0})
        effect_pairs[key]['direct'] = d.get('estimate') or 0.0
        effect_pairs[key]['direct_std'] = d.get('std_estimate') or 0.0

    for ie in effects['indirect']:
        key = (ie['from'], ie['to'])
        effect_pairs.setdefault(key, {'direct': 0.0, 'direct_std': 0.0,
                                       'indirect': 0.0, 'indirect_std': 0.0})
        effect_pairs[key]['indirect'] += ie.get('estimate') or 0.0
        effect_pairs[key]['indirect_std'] += ie.get('std_estimate') or 0.0

    for (fv, tv), vals in effect_pairs.items():
        effects['total'].append({
            'from': fv,
            'to': tv,
            'direct_effect': vals['direct'] or None,
            'direct_effect_std': vals['direct_std'] or None,
            'indirect_effect': vals['indirect'] or None,
            'indirect_effect_std': vals['indirect_std'] or None,
            'total_effect': vals['direct'] + vals['indirect'],
            'total_effect_std': vals['direct_std'] + vals['indirect_std'],
        })

    return effects


# ============================================
# SEM Analysis with semopy
# ============================================

def run_semopy_analysis(
    df: pd.DataFrame,
    model_spec: str,
    estimator: str = "MLW",
    bootstrap_n: int = 0,
    bootstrap_method: str = "ols_approximate",
) -> Dict[str, Any]:
    """Run SEM analysis using semopy."""
    try:
        from semopy import Model, calc_stats
    except ImportError:
        raise HTTPException(
            status_code=500,
            detail="semopy package not installed. Run: pip install semopy",
        )

    semopy_syntax = convert_to_semopy_syntax(model_spec)
    parsed = parse_lavaan_syntax(model_spec)

    # ── Fit model ───────────────────────────────────────────────────────────
    try:
        model = Model(semopy_syntax)
        obj = {'MLW': 'MLW', 'ML': 'ML', 'GLS': 'GLS', 'DWLS': 'DWLS'}.get(estimator, 'MLW')
        result = model.fit(df, obj=obj)
        if not result.success:
            raise HTTPException(
                status_code=400,
                detail=f"Model did not converge: {result.message}",
            )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Model fitting failed: {exc}")

    # ── Estimates ───────────────────────────────────────────────────────────
    estimates_df = model.inspect()
    raw_estimates = estimates_df.to_dict('records')

    try:
        std_estimates_df = model.inspect(std_est=True)
    except Exception:
        std_estimates_df = estimates_df.copy()

    # ── Fit indices ─────────────────────────────────────────────────────────
    try:
        stats_df = calc_stats(model)

        def _fi(name):
            return _safe_float(stats_df.loc['Value', name]) if name in stats_df.columns else None

        fit_indices = {
            'chi2': _fi('chi2'),
            'DoF': _fi('DoF'),
            'chi2_pvalue': _fi('chi2 p-value'),
            'CFI': _fi('CFI'),
            'TLI': _fi('TLI'),
            'RMSEA': _fi('RMSEA'),
            'AIC': _fi('AIC'),
            'BIC': _fi('BIC'),
            'GFI': _fi('GFI'),
            'AGFI': _fi('AGFI'),
            'NFI': _fi('NFI'),
        }
    except Exception as exc:
        logger.warning("Fit statistics failed: %s", exc)
        fit_indices = {'error': str(exc)}

    # Fix 4: proper SRMR
    fit_indices['SRMR'] = _calculate_srmr(model, df, parsed)

    # ── Parse estimate rows ─────────────────────────────────────────────────
    measurement_model: Dict[str, Any] = {}
    structural_model: List[Dict] = []
    variances_covariances: Dict[str, Any] = {
        'latent_variances': [],
        'latent_covariances': [],
        'error_variances': [],
    }

    latent_names = set(parsed['latent_vars'].keys())
    all_indicators: set = set()
    for inds in parsed['latent_vars'].values():
        all_indicators.update(inds)

    for idx, row in estimates_df.iterrows():
        lval = row.get('lval', '')
        op = row.get('op', '')
        rval = row.get('rval', '')
        estimate = _safe_float(row.get('Estimate'))
        std_err = _safe_float(row.get('Std. Err'))
        z_val = _safe_float(row.get('z-value'))
        p_val = _safe_float(row.get('p-value'))

        try:
            std_estimate = _safe_float(std_estimates_df.iloc[idx].get('Estimate'))
        except Exception:
            std_estimate = estimate

        significant = p_val is not None and p_val < 0.05

        if op == '~':
            if rval in latent_names and lval in all_indicators:
                latent, indicator = rval, lval
                measurement_model.setdefault(latent, {'indicators': [], 'loadings': {}})
                if indicator not in measurement_model[latent]['indicators']:
                    measurement_model[latent]['indicators'].append(indicator)
                measurement_model[latent]['loadings'][indicator] = {
                    'estimate': estimate,
                    'std_estimate': std_estimate,
                    'std_error': std_err,
                    'z_value': z_val,
                    'p_value': p_val,
                    'significant': significant,
                }
            else:
                structural_model.append({
                    'path': f'{rval} → {lval}',
                    'from': rval,
                    'to': lval,
                    'estimate': estimate,
                    'std_estimate': std_estimate,
                    'std_error': std_err,
                    'z_value': z_val,
                    'p_value': p_val,
                    'significant': significant,
                })

        elif op == '~~':
            item = {
                'var1': lval, 'var2': rval,
                'estimate': estimate, 'std_estimate': std_estimate,
                'std_error': std_err, 'z_value': z_val, 'p_value': p_val,
            }
            if lval == rval:
                if lval in latent_names:
                    item['variable'] = lval
                    variances_covariances['latent_variances'].append(item)
                else:
                    item['indicator'] = lval
                    variances_covariances['error_variances'].append(item)
            else:
                variances_covariances['latent_covariances'].append(item)

    # ── R² (Fix 5) ──────────────────────────────────────────────────────────
    endogenous_latents = {r['dv'] for r in parsed['regressions'] if r['dv'] in latent_names}
    r_squared = {
        lat: _estimate_r_squared(lat, structural_model, variances_covariances)
        for lat in endogenous_latents
    }

    # ── Effects (Fix 6 + 7) ─────────────────────────────────────────────────
    effects = _calculate_effects(
        parsed, structural_model,
        df=df,
        bootstrap_n=bootstrap_n,
        bootstrap_method=bootstrap_method,
        model_spec=model_spec,
        estimator=estimator,
    )

    # ── Visualisations ──────────────────────────────────────────────────────
    path_diagram = _generate_path_diagram(parsed, measurement_model, structural_model)
    loading_heatmap = _generate_loading_heatmap(measurement_model)
    correlation_matrix = _generate_correlation_matrix(df, parsed)

    # ── Interpretation ──────────────────────────────────────────────────────
    interpretation = _generate_interpretation(
        measurement_model, structural_model, fit_indices,
        r_squared, effects, parsed,
    )

    return {
        'parsed_model': parsed,
        'raw_estimates': raw_estimates,
        'measurement_model': measurement_model,
        'structural_model': structural_model,
        'variances_covariances': variances_covariances,
        'r_squared': r_squared,
        'effects': effects,
        'fit_indices': fit_indices,
        'path_diagram': path_diagram,
        'loading_heatmap': loading_heatmap,
        'correlation_matrix': correlation_matrix,
        'interpretation': interpretation,
        'estimator': estimator,
        'n_observations': len(df),
    }


# ============================================
# Visualization Functions (Fix 2: use _plot_context)
# ============================================

def _auto_layout_latents(
    latent_names: List[str],
    regressions: List[Dict],
    canvas_w: float = 10.0,
    canvas_h: float = 8.0,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute latent variable (x, y) positions using a force-directed heuristic.

    Strategy (3 tiers):
      1. Exogenous-only latents (no incoming structural paths) → left column
      2. Mediator latents (both incoming and outgoing paths) → middle column
      3. Endogenous-only latents (no outgoing structural paths) → right column

    Within each column, nodes are spaced evenly.  Fallback: grid layout when
    the structural graph is empty or all latents are isolated.
    """
    if not latent_names:
        return {}

    # Build structural in/out degree per latent
    in_deg: Dict[str, int] = {n: 0 for n in latent_names}
    out_deg: Dict[str, int] = {n: 0 for n in latent_names}
    for reg in regressions:
        dv = reg['dv']
        for iv in reg['ivs']:
            if iv in in_deg:
                in_deg[dv] = in_deg.get(dv, 0) + 1
            if iv in out_deg:
                out_deg[iv] = out_deg.get(iv, 0) + 1

    exogenous = [n for n in latent_names if in_deg[n] == 0 and out_deg[n] > 0]
    mediators  = [n for n in latent_names if in_deg[n] > 0 and out_deg[n] > 0]
    endogenous = [n for n in latent_names if in_deg[n] > 0 and out_deg[n] == 0]
    isolated   = [n for n in latent_names
                  if n not in exogenous and n not in mediators and n not in endogenous]

    def _col_positions(nodes: List[str], x_frac: float) -> Dict[str, Tuple[float, float]]:
        n = len(nodes)
        if n == 0:
            return {}
        x = canvas_w * x_frac
        # vertical spacing: distribute evenly with margins
        ys = [canvas_h * (i + 1) / (n + 1) for i in range(n)]
        return {nodes[i]: (x, ys[i]) for i in range(n)}

    # Determine x fractions based on which tiers are occupied
    tiers = [t for t in [exogenous, mediators, endogenous, isolated] if t]
    n_tiers = max(len(tiers), 1)
    positions: Dict[str, Tuple[float, float]] = {}

    occupied_tiers = []
    if exogenous: occupied_tiers.append((exogenous, 0))
    if mediators:  occupied_tiers.append((mediators, 1 if exogenous else 0))
    if endogenous: occupied_tiers.append((endogenous, len(occupied_tiers)))
    if isolated:   occupied_tiers.append((isolated,   len(occupied_tiers)))

    n_cols = len(occupied_tiers) if occupied_tiers else 1
    for nodes, col_idx in occupied_tiers:
        x_frac = (col_idx + 1) / (n_cols + 1)
        positions.update(_col_positions(nodes, x_frac))

    # Fallback: if all isolated or no regressions, use grid
    if not positions or all(v == positions.get(latent_names[0]) for v in positions.values()):
        cols = max(1, int(np.ceil(np.sqrt(len(latent_names)))))
        for i, name in enumerate(latent_names):
            col = i % cols
            row = i // cols
            n_rows = int(np.ceil(len(latent_names) / cols))
            x = canvas_w * (col + 1) / (cols + 1)
            y = canvas_h * (row + 1) / (n_rows + 1)
            positions[name] = (x, y)

    return positions


def _generate_path_diagram(
    parsed: Dict, measurement_model: Dict, structural_model: List[Dict]
) -> Optional[str]:
    """
    Generate SEM path diagram with automatic latent variable layout.

    Layout algorithm: exogenous → mediator → endogenous column placement
    (see _auto_layout_latents). Scales figure size with model complexity.
    For very complex models (>8 latents or >6 indicators per latent),
    indicator labels are truncated and font sizes are reduced automatically.
    """
    try:
        with _plot_context():
            latent_names = list(parsed['latent_vars'].keys())
            n_latents = len(latent_names)
            if n_latents == 0:
                return None

            # ── Adaptive figure size ─────────────────────────────────────
            max_inds = max((len(v) for v in parsed['latent_vars'].values()), default=0)
            fig_w = max(14, n_latents * 3.5)
            fig_h = max(10, max_inds * 1.2 + 4)
            canvas_w, canvas_h = fig_w - 2, fig_h - 3  # drawing area margins

            fig, ax = plt.subplots(figsize=(fig_w, fig_h))
            ax.set_xlim(-1, fig_w - 1)
            ax.set_ylim(-1, fig_h - 1)
            ax.axis('off')

            # ── Auto-position latents ────────────────────────────────────
            latent_positions = _auto_layout_latents(
                latent_names, parsed['regressions'],
                canvas_w=canvas_w, canvas_h=canvas_h,
            )
            # Shift by margin offset so coordinates sit inside the axes
            latent_positions = {
                k: (x + 1.0, y + 1.5)
                for k, (x, y) in latent_positions.items()
            }

            # Ellipse half-sizes scale with label length
            ell_w = max(2.0, min(3.2, 0.25 * max((len(n) for n in latent_names), default=4)))
            ell_h = 0.9

            for name, (x, y) in latent_positions.items():
                ellipse = mpatches.Ellipse(
                    (x, y), ell_w, ell_h, fill=True,
                    facecolor='#E3F2FD', edgecolor='#1976D2', linewidth=2,
                )
                ax.add_patch(ellipse)
                ax.text(x, y, name, ha='center', va='center',
                        fontsize=max(7, min(10, 80 // max(n_latents, 1))),
                        fontweight='bold')

            # ── Indicators ───────────────────────────────────────────────
            ind_font = max(6, min(9, 70 // max(max_inds, 1)))
            max_label_len = 10 if n_latents <= 4 else 6

            for latent_name, indicators in parsed['latent_vars'].items():
                if latent_name not in latent_positions:
                    continue
                lx, ly = latent_positions[latent_name]
                n_inds = len(indicators)
                # Spacing adapts to how many indicators there are
                ind_spacing = min(0.95, 5.5 / max(n_inds, 1))

                for j, ind in enumerate(indicators):
                    offset = (j - (n_inds - 1) / 2) * ind_spacing
                    ix = lx + offset
                    iy = ly - 2.4

                    rect_w, rect_h = min(0.85, ind_spacing * 0.9), 0.55
                    rect = mpatches.Rectangle(
                        (ix - rect_w / 2, iy - rect_h / 2), rect_w, rect_h,
                        fill=True, facecolor='#FFF3E0', edgecolor='#F57C00', linewidth=1.5,
                    )
                    ax.add_patch(rect)
                    ax.text(ix, iy, ind[:max_label_len], ha='center', va='center',
                            fontsize=ind_font)
                    ax.annotate('', xy=(ix, iy + rect_h / 2), xytext=(lx, ly - ell_h / 2),
                                arrowprops=dict(arrowstyle='->', color='#BDBDBD', lw=0.9))

                    # Loading label
                    if latent_name in measurement_model:
                        load_data = measurement_model[latent_name].get('loadings', {}).get(ind)
                        if load_data:
                            load_val = (
                                load_data.get('std_estimate') or load_data.get('estimate')
                            )
                            if load_val is not None:
                                ax.text(
                                    (ix + lx) / 2 + 0.1, (iy + ly) / 2,
                                    f'{load_val:.2f}', fontsize=max(6, ind_font - 1),
                                    color='#1565C0',
                                )

            # ── Structural paths ─────────────────────────────────────────
            # Use varying arc radii to reduce overlap when many paths connect
            # the same pair or are parallel.
            path_pairs: Dict[Tuple, int] = {}  # count paths between same pair
            for path in structural_model:
                fv, tv = path['from'], path['to']
                if fv in latent_positions and tv in latent_positions:
                    key = (min(fv, tv), max(fv, tv))
                    path_pairs[key] = path_pairs.get(key, 0) + 1

            path_pair_counter: Dict[Tuple, int] = {}
            for path in structural_model:
                fv, tv = path['from'], path['to']
                if fv not in latent_positions or tv not in latent_positions:
                    continue
                fx, fy = latent_positions[fv]
                tx, ty = latent_positions[tv]
                is_sig = path.get('significant', False)
                color = '#2E7D32' if is_sig else '#9E9E9E'

                key = (min(fv, tv), max(fv, tv))
                path_pair_counter[key] = path_pair_counter.get(key, 0) + 1
                n_parallel = path_pairs.get(key, 1)
                # Alternate arc direction for parallel paths
                arc_rad = 0.15 + 0.12 * (path_pair_counter[key] - 1)
                if path_pair_counter[key] % 2 == 0:
                    arc_rad = -arc_rad

                ax.annotate(
                    '', xy=(tx - ell_w / 2, ty), xytext=(fx + ell_w / 2, fy),
                    arrowprops=dict(
                        arrowstyle='->', color=color,
                        lw=2.5 if is_sig else 1.5,
                        connectionstyle=f"arc3,rad={arc_rad}",
                    ),
                )
                beta = path.get('std_estimate') or path.get('estimate')
                if beta is not None:
                    mid_x = (fx + tx) / 2
                    mid_y = (fy + ty) / 2 + 0.35 + 0.2 * (path_pair_counter[key] - 1)
                    ax.text(
                        mid_x, mid_y, f"β={beta:.3f}",
                        fontsize=max(7, min(9, 80 // max(len(structural_model), 1))),
                        fontweight='bold', color=color,
                        bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.85),
                    )

            ax.set_title('SEM Path Diagram', fontsize=14, fontweight='bold', pad=20)
            legend_elements = [
                mpatches.Patch(facecolor='#E3F2FD', edgecolor='#1976D2', label='Latent Variable'),
                mpatches.Patch(facecolor='#FFF3E0', edgecolor='#F57C00', label='Observed Variable'),
                plt.Line2D([0], [0], color='#2E7D32', lw=2.5, label='Significant Path (p < .05)'),
                plt.Line2D([0], [0], color='#9E9E9E', lw=1.5, label='Non-significant Path'),
            ]
            ax.legend(handles=legend_elements, loc='lower right', fontsize=9)
            plt.tight_layout()
            return _fig_to_base64(fig)
    except Exception as exc:
        logger.warning("Path diagram error: %s", exc)
        return None


def _generate_loading_heatmap(measurement_model: Dict) -> Optional[str]:
    """Generate factor loadings heatmap."""
    try:
        if not measurement_model:
            return None
        all_indicators: List[str] = []
        all_latents = list(measurement_model.keys())
        for latent, data in measurement_model.items():
            for ind in data.get('indicators', []):
                if ind not in all_indicators:
                    all_indicators.append(ind)
        if not all_indicators or not all_latents:
            return None

        loading_matrix = np.zeros((len(all_indicators), len(all_latents)))
        for j, latent in enumerate(all_latents):
            for ind, load_data in measurement_model[latent].get('loadings', {}).items():
                if ind in all_indicators:
                    i = all_indicators.index(ind)
                    loading_matrix[i, j] = (
                        load_data.get('std_estimate') or load_data.get('estimate') or 0
                    )

        with _plot_context():
            fig, ax = plt.subplots(
                figsize=(max(8, len(all_latents) * 2), max(6, len(all_indicators) * 0.5))
            )
            sns.heatmap(
                loading_matrix, annot=True, fmt='.3f', cmap='RdBu_r', center=0,
                xticklabels=all_latents, yticklabels=all_indicators,
                vmin=-1, vmax=1, ax=ax,
                cbar_kws={'label': 'Standardized Loading'},
                annot_kws={'size': 9},
            )
            ax.set_title('Factor Loadings (Standardized)', fontsize=14, fontweight='bold')
            ax.set_xlabel('Latent Variables', fontsize=12)
            ax.set_ylabel('Indicators', fontsize=12)
            plt.tight_layout()
            return _fig_to_base64(fig)
    except Exception as exc:
        logger.warning("Loading heatmap error: %s", exc)
        return None


def _generate_correlation_matrix(df: pd.DataFrame, parsed: Dict) -> Optional[str]:
    """Generate indicator correlation matrix."""
    try:
        all_indicators = [
            ind for inds in parsed['latent_vars'].values()
            for ind in inds if ind in df.columns
        ]
        if len(all_indicators) < 2:
            return None
        corr_matrix = df[all_indicators].corr()
        with _plot_context():
            fig, ax = plt.subplots(
                figsize=(max(8, len(all_indicators) * 0.7), max(6, len(all_indicators) * 0.6))
            )
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            sns.heatmap(
                corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                center=0, vmin=-1, vmax=1, ax=ax,
                cbar_kws={'label': 'Correlation'}, annot_kws={'size': 8},
            )
            ax.set_title('Indicator Correlation Matrix', fontsize=14, fontweight='bold')
            plt.tight_layout()
            return _fig_to_base64(fig)
    except Exception as exc:
        logger.warning("Correlation matrix error: %s", exc)
        return None


# ============================================
# Interpretation Generation
# ============================================

def _generate_interpretation(
    measurement_model: Dict,
    structural_model: List[Dict],
    fit_indices: Dict,
    r_squared: Dict,
    effects: Dict,
    parsed: Dict,
) -> Dict[str, Any]:
    """Generate comprehensive English interpretation."""
    key_insights = []

    cfi = fit_indices.get('CFI') or 0
    rmsea = fit_indices.get('RMSEA') or 1
    srmr = fit_indices.get('SRMR')

    fit_quality = []
    fit_quality.append(
        f"CFI = {cfi:.3f} ({'excellent' if cfi >= .95 else 'acceptable' if cfi >= .90 else 'poor'})"
    )
    fit_quality.append(
        f"RMSEA = {rmsea:.3f} ({'excellent' if rmsea <= .05 else 'acceptable' if rmsea <= .08 else 'poor'})"
    )
    if srmr is not None:
        fit_quality.append(
            f"SRMR = {srmr:.3f} ({'good' if srmr <= .08 else 'acceptable' if srmr <= .10 else 'poor'})"
        )

    is_good_fit = cfi >= 0.90 and rmsea <= 0.08
    key_insights.append({
        'title': 'Model Fit',
        'description': '; '.join(fit_quality) + (
            '. The model shows adequate fit to the data.'
            if is_good_fit else '. Consider model modification.'
        ),
    })

    for latent, data in measurement_model.items():
        loadings = data.get('loadings', {})
        if not loadings:
            continue
        load_vals = [abs(l.get('std_estimate') or l.get('estimate') or 0) for l in loadings.values()]
        avg_loading = np.mean(load_vals) if load_vals else 0
        n_sig = sum(1 for l in loadings.values() if l.get('significant'))
        quality = 'excellent' if avg_loading >= 0.7 else 'acceptable' if avg_loading >= 0.5 else 'weak'
        key_insights.append({
            'title': f'Measurement: {latent}',
            'description': (
                f'Average loading = {avg_loading:.3f} ({quality}). '
                f'{n_sig}/{len(loadings)} indicators significant.'
            ),
        })

    sig_paths = [p for p in structural_model if p.get('significant')]
    nonsig_paths = [p for p in structural_model if not p.get('significant')]

    if sig_paths:
        strongest = max(sig_paths, key=lambda x: abs(x.get('std_estimate') or x.get('estimate') or 0))
        beta = strongest.get('std_estimate') or strongest.get('estimate')
        key_insights.append({
            'title': 'Strongest Path',
            'description': (
                f"{strongest['from']} → {strongest['to']} (β = {beta:.3f}, p < .05). "
                "This represents the strongest significant relationship in the model."
            ),
        })
        key_insights.append({
            'title': 'Significant Paths',
            'description': (
                f"{len(sig_paths)} path(s) significant at p < .05: "
                + ', '.join(f"{p['from']}→{p['to']}" for p in sig_paths[:3]) + '.'
            ),
        })

    if nonsig_paths:
        key_insights.append({
            'title': 'Non-significant Paths',
            'description': (
                f"{len(nonsig_paths)} path(s) not significant: "
                + ', '.join(f"{p['from']}→{p['to']}" for p in nonsig_paths[:2])
                + '. Consider removing or modifying.'
            ),
        })

    for var, r2 in r_squared.items():
        strength = 'substantial' if r2 >= 0.50 else 'moderate' if r2 >= 0.25 else 'weak'
        key_insights.append({
            'title': f'Variance Explained: {var}',
            'description': f'R² = {r2:.3f} ({r2*100:.1f}%). {strength.capitalize()} proportion explained.',
        })

    # Highlight multi-step indirect effects
    multi_step = [ie for ie in effects.get('indirect', []) if ie.get('n_steps', 1) >= 2]
    for ie in multi_step[:3]:
        boot_note = ''
        if ie.get('sig_bootstrap') is not None:
            boot_note = (
                f" Bootstrap 95% CI [{ie['ci_lower']:.3f}, {ie['ci_upper']:.3f}] – "
                f"{'significant' if ie['sig_bootstrap'] else 'not significant'}."
            )
        key_insights.append({
            'title': f"Indirect Effect ({ie['n_steps']}-step)",
            'description': (
                f"{ie['path']}: β = {ie.get('std_estimate', 0):.3f}.{boot_note}"
            ),
        })

    n_sig_paths = len(sig_paths)
    if is_good_fit and n_sig_paths > 0:
        overall = 'Good – The model fits well and shows significant relationships.'
    elif is_good_fit:
        overall = 'Acceptable – Good fit but no significant structural paths.'
    else:
        overall = 'Needs improvement – Consider modifying the model for better fit.'

    return {
        'key_insights': key_insights,
        'n_latent_vars': len(parsed['latent_vars']),
        'n_significant_paths': n_sig_paths,
        'overall_assessment': overall,
    }


# ============================================
# API Endpoint
# ============================================

@router.post("/sem")
async def run_sem_analysis(request: SEMRequest) -> Dict[str, Any]:
    """
    Perform Structural Equation Modeling (SEM) analysis.

    Uses semopy for proper Maximum Likelihood estimation.
    Supports lavaan-style model specification syntax.

    Returns measurement model, structural paths, fit indices,
    R², direct/indirect/total effects (with optional bootstrap CIs),
    visualisations, and English interpretation.
    """
    try:
        if not request.data:
            raise HTTPException(status_code=400, detail="Data not provided.")
        if not request.model_spec or not request.model_spec.strip():
            raise HTTPException(status_code=400, detail="Model specification required.")

        df_raw = pd.DataFrame(request.data)
        n_raw = len(df_raw)

        parsed = parse_lavaan_syntax(request.model_spec)
        if not parsed['latent_vars'] and not parsed['regressions']:
            raise HTTPException(
                status_code=400,
                detail="Invalid model specification. Use lavaan-style syntax.",
            )

        # Collect observed variable names
        all_obs_vars: set = set()
        for inds in parsed['latent_vars'].values():
            all_obs_vars.update(inds)
        for reg in parsed['regressions']:
            if reg['dv'] not in parsed['latent_vars']:
                all_obs_vars.add(reg['dv'])
            all_obs_vars.update(iv for iv in reg['ivs'] if iv not in parsed['latent_vars'])

        missing_cols = [c for c in all_obs_vars if c not in df_raw.columns]
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Columns not found in data: {', '.join(missing_cols)}",
            )

        # Convert to numeric + listwise deletion
        df = df_raw.copy()
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        relevant_cols = [c for c in all_obs_vars if c in df.columns]
        df = df.dropna(subset=relevant_cols).reset_index(drop=True)

        # Fix 8: comprehensive validation (errors fatal, warnings surfaced)
        validation = _validate_sem_input(df, parsed, n_raw)
        if validation['errors']:
            raise HTTPException(
                status_code=400,
                detail="Validation errors: " + " | ".join(validation['errors']),
            )

        # Run analysis
        result = run_semopy_analysis(
            df, request.model_spec, request.estimator,
            request.bootstrap_n, request.bootstrap_method,
        )

        # Attach validation warnings to result
        result['validation_warnings'] = validation['warnings']

        return _to_native(result)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error in run_sem_analysis")
        raise HTTPException(status_code=500, detail=f"SEM analysis failed: {exc}")
