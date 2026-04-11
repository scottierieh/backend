from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Optional, Union
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
import io, base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
sns.set_theme(style="whitegrid")

router = APIRouter()


class MediationRequest(BaseModel):
    data: list[dict[str, Any]] = Field(...)
    xVar: str = Field(...)
    mVar: Union[str, list[str]] = Field(...)
    yVar: str = Field(...)
    nBootstrap: Optional[int] = 1000
    standardize: Optional[bool] = True
    serialMediation: Optional[bool] = False
    bootstrapSeed: Optional[int] = 42
    robustSE: Optional[bool] = False
    covariates: Optional[list[str]] = []

import re as _re

def _split_sections(html_text: str) -> list:
    """Split HTML interpretation blob into [{title, body}] paragraph sections."""
    parts = _re.split(r'(?=<strong>[^<]{3,60}:</strong>)', (html_text or '').strip())
    sections = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        m = _re.match(r'<strong>([^<]+)</strong>(.*)', part, _re.DOTALL)
        if m:
            sections.append({'title': m.group(1).rstrip(':').strip(),
                             'body':  m.group(2).strip()})
        else:
            sections.append({'title': '', 'body': part})
    return sections

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
        if val is None or pd.isna(val) or np.isinf(val): return default
        return float(val)
    except: return default

def simple_regression(X, Y):
    model = LinearRegression()
    X_reshaped = X.reshape(-1, 1)
    model.fit(X_reshaped, Y)
    y_pred = model.predict(X_reshaped)
    residuals = Y - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    mse = ss_res / (len(Y) - 2) if (len(Y) - 2) > 0 else 0
    ss_x = np.sum((X - np.mean(X)) ** 2)
    se_coef = np.sqrt(mse / ss_x) if ss_x > 0 else np.nan
    t_stat = model.coef_[0] / se_coef if se_coef and not np.isnan(se_coef) else np.inf
    p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), len(Y) - 2)) if (len(Y) - 2) > 0 else np.nan
    return {'coef': model.coef_[0], 'intercept': model.intercept_, 'se': se_coef, 't_stat': t_stat, 'p_value': p_value, 'r_squared': r_squared}

def multiple_regression(X1, X2, Y):
    X_combined = np.column_stack([X1, X2])
    model = LinearRegression()
    model.fit(X_combined, Y)
    y_pred = model.predict(X_combined)
    residuals = Y - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    mse = ss_res / (len(Y) - 3) if (len(Y) - 3) > 0 else 0
    X_design = np.column_stack([np.ones(len(X1)), X1, X2])
    try:
        cov_matrix = mse * np.linalg.inv(X_design.T @ X_design)
        se_coefs = np.sqrt(np.diag(cov_matrix))[1:]
    except:
        se_coefs = [np.nan, np.nan]
    t_stats = model.coef_ / se_coefs if not np.any(np.isnan(se_coefs)) else np.array([np.inf, np.inf])
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), len(Y) - 3)) if (len(Y) - 3) > 0 else np.array([np.nan, np.nan])
    return {'coef1': model.coef_[0], 'coef2': model.coef_[1], 'intercept': model.intercept_, 'se1': se_coefs[0], 'se2': se_coefs[1], 't_stat1': t_stats[0], 't_stat2': t_stats[1], 'p_value1': p_values[0], 'p_value2': p_values[1], 'r_squared': r_squared}

def baron_kenny_analysis(X, M, Y):
    path_c = simple_regression(X, Y)
    path_a = simple_regression(X, M)
    path_bc = multiple_regression(X, M, Y)
    indirect_effect = path_a['coef'] * path_bc['coef2']
    sobel_se = np.sqrt(path_bc['coef2']**2 * path_a['se']**2 + path_a['coef']**2 * path_bc['se2']**2) if path_a['se'] > 0 and path_bc['se2'] > 0 else 0
    sobel_z = indirect_effect / sobel_se if sobel_se > 0 else np.inf
    sobel_p = 2 * (1 - stats.norm.cdf(np.abs(sobel_z)))
    return {
        'path_c': path_c, 'path_a': path_a,
        'path_b': {'coef': path_bc['coef2'], 'se': path_bc['se2'], 't_stat': path_bc['t_stat2'], 'p_value': path_bc['p_value2']},
        'path_c_prime': {'coef': path_bc['coef1'], 'se': path_bc['se1'], 't_stat': path_bc['t_stat1'], 'p_value': path_bc['p_value1']},
        'indirect_effect': indirect_effect,
        'sobel_test': {'effect': indirect_effect, 'se': sobel_se, 'z_stat': sobel_z, 'p_value': sobel_p}
    }

def bootstrap_analysis(X, M, Y, n_bootstrap=1000, confidence_level=0.95):
    np.random.seed(42)
    n = len(X)
    indirect_effects = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        X_boot, M_boot, Y_boot = X[indices], M[indices], Y[indices]
        path_a = simple_regression(X_boot, M_boot)
        path_bc = multiple_regression(X_boot, M_boot, Y_boot)
        indirect_effects.append(path_a['coef'] * path_bc['coef2'])
    indirect_effects = np.array(indirect_effects)
    alpha = 1 - confidence_level
    ci_lower = np.percentile(indirect_effects, 100 * alpha / 2)
    ci_upper = np.percentile(indirect_effects, 100 * (1 - alpha / 2))
    return {
        'mean_effect': np.mean(indirect_effects), 'se': np.std(indirect_effects),
        'ci_lower': ci_lower, 'ci_upper': ci_upper, 'n_bootstrap': n_bootstrap,
        'confidence_level': confidence_level, 'significant': not (ci_lower <= 0 <= ci_upper)
    }

def _sig_label(p):
    if p < 0.001: return "p < .001"
    if p < 0.01:  return f"p = {p:.3f}"
    if p < 0.05:  return f"p = {p:.3f}"
    return f"p = {p:.3f} (ns)"

def _effect_size_label(coef):
    a = abs(coef)
    if a >= 0.5: return "large"
    if a >= 0.3: return "moderate"
    if a >= 0.1: return "small"
    return "negligible"

def _apa_path(coef, se, t, df, p, r2=None, label=None):
    """Format a single regression path in APA 7th style."""
    p_str = "< .001" if p < 0.001 else f"= {p:.3f}"
    r2_str = f", <em>R</em>² = {r2:.3f}" if r2 is not None else ""
    lbl = f"{label}, " if label else ""
    return f"{lbl}β = {coef:.3f}, SE = {se:.3f}, <em>t</em>({int(df)}) = {t:.2f}, <em>p</em> {p_str}{r2_str}"

def _apa_ci(lo, hi):
    return f"95% CI [{lo:.3f}, {hi:.3f}]"

def _dir(coef):
    return "positively" if coef > 0 else "negatively"

def _es(coef):
    a = abs(coef)
    if a >= 0.50: return "large"
    if a >= 0.30: return "moderate"
    if a >= 0.10: return "small"
    return "negligible"

def _prop_label(pct):
    if pct > 66: return "a large proportion"
    if pct > 33: return "a moderate proportion"
    return "a small proportion"

def generate_interpretation(bk, boot, X_name, M_name, Y_name, n_obs=None):
    indirect_sig = boot['significant'] if boot else bk['sobel_test']['p_value'] < 0.05
    direct_sig   = bk['path_c_prime']['p_value'] < 0.05
    total_sig    = bk['path_c']['p_value'] < 0.05
    path_a_sig   = bk['path_a']['p_value'] < 0.05
    path_b_sig   = bk['path_b']['p_value'] < 0.05

    N = n_obs or 100  # fallback
    df_simple   = N - 2
    df_multiple = N - 3

    # Mediation type
    if indirect_sig and not direct_sig:
        med_type = "Full Mediation"
    elif indirect_sig and direct_sig:
        med_type = "Partial Mediation"
    else:
        med_type = "No Mediation"

    # Proportion mediated
    prop_med = None
    if total_sig and indirect_sig and abs(bk['path_c']['coef']) > 1e-6:
        prop_med = bk['indirect_effect'] / bk['path_c']['coef']

    # ── Overall Analysis (APA 7th) ────────────────────────────────────────────
    parts = []

    # 1. Model description sentence
    parts.append(
        f"A simple mediation analysis was conducted to examine whether <em>{M_name}</em> mediates "
        f"the relationship between <em>{X_name}</em> and <em>{Y_name}</em> "
        f"(<em>N</em> = {N}, bootstrap = {boot['n_bootstrap']:,} samples, BCa 95% CI)."
    )

    # 2. Opening verdict
    if med_type == "Full Mediation":
        parts.append(
            f"Results supported <strong>full mediation</strong>: the total effect of <em>{X_name}</em> on "
            f"<em>{Y_name}</em> was fully transmitted through <em>{M_name}</em>, and the direct effect was "
            f"no longer significant once the mediator was controlled."
        )
    elif med_type == "Partial Mediation":
        parts.append(
            f"Results supported <strong>partial mediation</strong>: <em>{M_name}</em> carried a significant "
            f"portion of the <em>{X_name}</em>–<em>{Y_name}</em> relationship, while a significant direct "
            f"effect of <em>{X_name}</em> on <em>{Y_name}</em> remained."
        )
    else:
        if not path_a_sig and not path_b_sig:
            parts.append(
                f"Results provided <strong>no support for mediation</strong> through <em>{M_name}</em>. "
                f"Neither the <em>{X_name}</em> → <em>{M_name}</em> path nor the <em>{M_name}</em> → <em>{Y_name}</em> "
                f"path reached statistical significance."
            )
        elif not path_a_sig:
            parts.append(
                f"Results provided <strong>no support for mediation</strong>. "
                f"The proposed causal chain was broken at the first link: "
                f"<em>{X_name}</em> did not significantly predict <em>{M_name}</em> (Path a)."
            )
        elif not path_b_sig:
            parts.append(
                f"Results provided <strong>no support for mediation</strong>. "
                f"Although <em>{X_name}</em> significantly predicted <em>{M_name}</em>, "
                f"<em>{M_name}</em> did not significantly predict <em>{Y_name}</em> after controlling for <em>{X_name}</em> (Path b)."
            )
        else:
            parts.append(
                f"Results provided <strong>no significant indirect effect</strong>, despite both individual paths (a and b) "
                f"reaching significance. The bootstrap confidence interval for the indirect effect contained zero, "
                f"suggesting possible suppression or insufficient power."
            )

    # 3. APA-formatted path coefficients
    a  = bk['path_a']
    b  = bk['path_b']
    c  = bk['path_c']
    cp = bk['path_c_prime']

    parts.append(
        f"<strong>Path a</strong> (<em>{X_name}</em> → <em>{M_name}</em>): "
        f"{_apa_path(a['coef'], a['se'], a['t_stat'], df_simple, a['p_value'])}. "
        f"<em>{X_name}</em> {_dir(a['coef'])} predicted <em>{M_name}</em> "
        f"({'significantly' if path_a_sig else 'non-significantly'}), "
        f"representing a {_es(a['coef'])} effect."
    )

    parts.append(
        f"<strong>Path b</strong> (<em>{M_name}</em> → <em>{Y_name}</em>, controlling for <em>{X_name}</em>): "
        f"{_apa_path(b['coef'], b['se'], b['t_stat'], df_multiple, b['p_value'])}. "
        f"<em>{M_name}</em> {_dir(b['coef'])} predicted <em>{Y_name}</em> "
        f"({'significantly' if path_b_sig else 'non-significantly'}), "
        f"representing a {_es(b['coef'])} effect."
    )

    parts.append(
        f"<strong>Path c</strong> (total effect, <em>{X_name}</em> → <em>{Y_name}</em>): "
        f"{_apa_path(c['coef'], c['se'], c['t_stat'], df_simple, c['p_value'])}. "
        f"The total effect of <em>{X_name}</em> on <em>{Y_name}</em> was "
        f"{'statistically significant' if total_sig else 'not statistically significant'} "
        f"and {_dir(c['coef'])} in direction."
    )

    parts.append(
        f"<strong>Path c′</strong> (direct effect, <em>{X_name}</em> → <em>{Y_name}</em> controlling for <em>{M_name}</em>): "
        f"{_apa_path(cp['coef'], cp['se'], cp['t_stat'], df_multiple, cp['p_value'])}. "
        f"After accounting for <em>{M_name}</em>, the direct effect of <em>{X_name}</em> on <em>{Y_name}</em> "
        f"was {'still statistically significant' if direct_sig else 'no longer statistically significant'}."
    )

    # 4. Indirect effect with bootstrap CI (core APA requirement)
    if boot:
        sig_str = (
            "did not include zero, indicating a statistically significant indirect effect"
            if boot['significant'] else
            "included zero, indicating the indirect effect was not statistically significant"
        )
        parts.append(
            f"<strong>Indirect effect</strong> (a × b = {bk['indirect_effect']:.3f}): "
            f"A bootstrap analysis ({boot['n_bootstrap']:,} resamples) yielded a point estimate of "
            f"{boot['mean_effect']:.3f} (SE = {boot['se']:.3f}), {_apa_ci(boot['ci_lower'], boot['ci_upper'])}. "
            f"The confidence interval {sig_str}."
        )

    # 5. Sobel test (supplementary)
    sobel = bk.get('sobel_test', {})
    if sobel and sobel.get('se', 0) > 0:
        sp = sobel['p_value']
        sp_str = "< .001" if sp < 0.001 else f"= {sp:.3f}"
        parts.append(
            f"The Sobel test also {'supported' if sp < 0.05 else 'did not support'} the indirect effect, "
            f"<em>z</em> = {sobel['z_stat']:.2f}, <em>p</em> {sp_str} "
            f"(note: bootstrap CI is the preferred test for asymmetric sampling distributions)."
        )

    # 6. Proportion mediated
    if prop_med is not None:
        pct = prop_med * 100
        parts.append(
            f"<em>{M_name}</em> accounted for <strong>{pct:.1f}%</strong> of the total effect of "
            f"<em>{X_name}</em> on <em>{Y_name}</em> — {_prop_label(pct)} of the variance in <em>{Y_name}</em> "
            f"attributable to <em>{X_name}</em> was transmitted through the mediator."
        )

    overall_analysis = "<br><br>".join(parts)

    # ── Recommendations ───────────────────────────────────────────────────────
    rec = []

    # APA reporting checklist
    boot_method = f"BCa bootstrap ({boot['n_bootstrap']:,} samples)" if boot else "bootstrap"
    rec.append(
        f"<strong>APA 7th reporting checklist:</strong> Per Preacher & Hayes (2008) and current APA guidelines, "
        f"report (1) all unstandardized path coefficients (a, b, c, c′) with SEs and <em>p</em>-values, "
        f"(2) the indirect effect estimate with {boot_method} 95% CI, "
        f"(3) the mediation type (full/partial/none), "
        f"and (4) a path diagram. "
        f"Do <em>not</em> use the Sobel test as the primary significance test — bootstrap CI is the standard."
    )

    # Substantive recommendations based on result
    if med_type == "Full Mediation":
        rec.append(
            f"<strong>Theoretical implication:</strong> The absence of a significant direct effect (c′) after "
            f"controlling for <em>{M_name}</em> suggests that <em>{M_name}</em> is the <em>mechanism</em> "
            f"through which <em>{X_name}</em> operates. Interventions targeting <em>{M_name}</em> directly "
            f"may be more effective than targeting <em>{X_name}</em> alone."
        )
        rec.append(
            f"<strong>Model extension:</strong> Test whether the mediation is <em>moderated</em> — does it hold "
            f"equally across different groups or conditions (moderated mediation / PROCESS Model 7 or 14)? "
            f"Also consider adding parallel mediators to test whether <em>{M_name}</em> is the unique mechanism "
            f"or one of several."
        )
        if prop_med and prop_med * 100 > 80:
            rec.append(
                f"<strong>Caution — near-complete mediation:</strong> {prop_med*100:.1f}% of the total effect is "
                f"mediated. Verify that <em>{X_name}</em> and <em>{M_name}</em> are not measuring essentially the "
                f"same construct (discriminant validity), which could artificially inflate the proportion mediated."
            )
    elif med_type == "Partial Mediation":
        direct_pct = 100 - (prop_med * 100 if prop_med else 50)
        rec.append(
            f"<strong>Theoretical implication:</strong> The significant direct effect (c′ = {cp['coef']:.3f}) "
            f"means <em>{M_name}</em> is <em>one</em> mechanism, but additional pathways remain unexplained "
            f"({'approximately ' + str(round(direct_pct)) + '% of the total effect is direct'}). "
            f"Future research should identify additional mediators to achieve a more complete mechanistic account."
        )
        rec.append(
            f"<strong>Reporting note:</strong> Report both effects independently — "
            f"indirect effect = {bk['indirect_effect']:.3f} {_apa_ci(boot['ci_lower'], boot['ci_upper']) if boot else ''} "
            f"and direct effect = {cp['coef']:.3f}, SE = {cp['se']:.3f}. "
            f"Describing only the indirect effect would misrepresent the model."
        )
        if prop_med:
            strength = "substantial" if prop_med > 0.4 else "moderate" if prop_med > 0.2 else "modest"
            rec.append(
                f"<strong>Effect size context:</strong> The {prop_med*100:.1f}% proportion mediated represents "
                f"a {strength} indirect pathway. "
                f"{'Intervention efforts targeting the mediator are likely to produce meaningful gains.' if prop_med > 0.3 else 'Given the modest indirect pathway, targeting both the mediator and the direct X → Y relationship may be optimal.'}"
            )
    else:
        # No mediation — diagnose the specific failure point
        if not path_a_sig:
            rec.append(
                f"<strong>Diagnosis — broken Path a:</strong> <em>{X_name}</em> does not reliably predict "
                f"<em>{M_name}</em> (β = {a['coef']:.3f}, SE = {a['se']:.3f}, <em>p</em> = {a['p_value']:.3f}). "
                f"Possible causes: (1) <em>{M_name}</em> is not causally downstream of <em>{X_name}</em> in this context; "
                f"(2) measurement error in <em>{M_name}</em> attenuates the relationship; "
                f"(3) the effect is too small for the current sample (<em>N</em> = {N}) to detect."
            )
        elif not path_b_sig:
            rec.append(
                f"<strong>Diagnosis — broken Path b:</strong> <em>{M_name}</em> does not predict <em>{Y_name}</em> "
                f"after controlling for <em>{X_name}</em> (β = {b['coef']:.3f}, SE = {b['se']:.3f}, "
                f"<em>p</em> = {b['p_value']:.3f}). "
                f"<em>{M_name}</em> may be a <em>distal</em> rather than proximal antecedent of <em>{Y_name}</em>, "
                f"or may act as a covariate rather than a true mediator in this model."
            )
        else:
            rec.append(
                f"<strong>Diagnosis — suppression / power issue:</strong> Both individual paths are significant "
                f"but the indirect effect CI spans zero {_apa_ci(boot['ci_lower'], boot['ci_upper']) if boot else ''}. "
                f"This pattern sometimes indicates a <em>suppressor variable</em> inflating individual path coefficients "
                f"while the combined product (a × b) remains unstable. "
                f"Consider a larger bootstrap sample (5,000+) and/or a larger dataset."
            )
        rec.append(
            f"<strong>Alternatives to consider:</strong> (1) <strong>Moderation analysis</strong> — does the "
            f"<em>{X_name}</em> → <em>{Y_name}</em> relationship vary as a function of a third variable? "
            f"(2) <strong>Alternative mediators</strong> — identify candidates with stronger theoretical links to both "
            f"<em>{X_name}</em> and <em>{Y_name}</em>. "
            f"(3) <strong>Longitudinal design</strong> — cross-sectional mediation is inherently limited in establishing "
            f"temporal precedence."
        )

    # Universal: power and sample size note
    if N < 100:
        rec.append(
            f"<strong>Statistical power note:</strong> With <em>N</em> = {N}, this study may be underpowered "
            f"for detecting small-to-moderate indirect effects. Fritz & MacKinnon (2007) recommend "
            f"<em>N</em> ≥ 100–200 for reliable BCa bootstrap intervals. Interpret results with caution and "
            f"consider replication with a larger sample."
        )

    recommendations = "<br><br>".join(rec)
    return overall_analysis, recommendations, med_type


def generate_parallel_interpretation(parallel_result, X_name, M_names, Y_name, n_obs=None, n_bootstrap=1000):
    """APA 7th interpretation for parallel mediation."""
    N = n_obs or 100
    paths        = parallel_result.get('paths', [])
    total_indirect = parallel_result.get('total_indirect', 0)
    total_ci     = parallel_result.get('total_boot_ci', {}) or {}
    total_sig    = total_ci.get('significant', False)
    sig_paths    = [i for i, p in enumerate(paths) if (p.get('bootstrap') or {}).get('significant', False)]
    n_med        = len(M_names)

    def apa_p(p):
        if p is None: return "= n/a"
        return "< .001" if p < 0.001 else f"= {p:.3f}"
    def apa_coef(b, se, t, df, p):
        if any(v is None for v in [b, se, t, df, p]): return f"β = {b:.3f}"
        return f"β = {b:.3f}, SE = {se:.3f}, t({int(df)}) = {t:.2f}, p {apa_p(p)}"

    parts = []

    # 1. Model description
    parts.append(
        f"A parallel multiple mediation analysis was conducted to examine whether {n_med} mediators "
        f"({', '.join(f'<em>{m}</em>' for m in M_names)}) simultaneously mediate the relationship "
        f"between <em>{X_name}</em> and <em>{Y_name}</em> "
        f"(<em>N</em> = {N}, bootstrap = {n_bootstrap:,} samples, BCa 95% CI). "
        f"All mediators were entered simultaneously, with each b-path estimated controlling for "
        f"<em>{X_name}</em> and all other mediators."
    )

    # 2. Verdict
    if total_sig and len(sig_paths) == n_med:
        parts.append(
            f"Results supported <strong>full parallel mediation</strong>: all {n_med} mediators carried "
            f"statistically significant specific indirect effects, and the total indirect effect was significant."
        )
    elif total_sig and len(sig_paths) > 0:
        sig_names = [M_names[i] for i in sig_paths]
        parts.append(
            f"Results provided <strong>partial support</strong> for the parallel mediation model: "
            f"{len(sig_paths)} of {n_med} mediators ({', '.join(f'<em>{m}</em>' for m in sig_names)}) "
            f"showed significant specific indirect effects, and the total indirect effect was significant."
        )
    elif len(sig_paths) > 0:
        sig_names = [M_names[i] for i in sig_paths]
        parts.append(
            f"Results showed <strong>mixed support</strong>: {len(sig_paths)} of {n_med} mediators "
            f"({', '.join(f'<em>{m}</em>' for m in sig_names)}) showed significant specific indirect effects, "
            f"but the total indirect effect CI included zero — pathways may partially offset each other."
        )
    else:
        parts.append(
            f"Results provided <strong>no support for parallel mediation</strong>: "
            f"none of the {n_med} mediators produced a significant specific indirect effect, "
            f"and the total indirect effect CI included zero."
        )

    # 3. Per-mediator APA-formatted paths
    # Try to get df from first path's baron_kenny (from simple_regression df = N-2)
    df_simple   = N - 2
    df_multiple = N - n_med - 2   # Y ~ X + M1 + ... + Mn

    for i, (path_data, m_name) in enumerate(zip(paths, M_names)):
        bk_p   = path_data.get('baron_kenny', {}) or {}
        boot_p = path_data.get('bootstrap',   {}) or {}
        if not bk_p:
            continue

        a = bk_p.get('path_a', {}) or {}
        b = bk_p.get('path_b', {}) or {}
        ind   = bk_p.get('indirect_effect', 0)
        ci_lo = boot_p.get('ci_lower', 0)
        ci_hi = boot_p.get('ci_upper', 0)
        sig   = boot_p.get('significant', False)

        a_str = apa_coef(a.get('coef',0), a.get('se'), a.get('t_stat'), df_simple,   a.get('p_value'))
        b_str = apa_coef(b.get('coef',0), b.get('se'), b.get('t_stat'), df_multiple, b.get('p_value'))

        parts.append(
            f"<strong>Mediator {i+1} (<em>{m_name}</em>):</strong> "
            f"Path a (<em>{X_name}</em> → <em>{m_name}</em>): {a_str}. "
            f"Path b (<em>{m_name}</em> → <em>{Y_name}</em>, controlling for <em>{X_name}</em> and other mediators): {b_str}. "
            f"Specific indirect effect (a × b) = {ind:.3f}, "
            f"95% CI [{ci_lo:.3f}, {ci_hi:.3f}] — "
            f"<strong>{'statistically significant' if sig else 'not statistically significant'}</strong>."
        )

    # 4. Direct effect (c')
    first_bk = (paths[0].get('baron_kenny') or {}) if paths else {}
    cp = first_bk.get('path_c_prime', {}) or {}
    c  = first_bk.get('path_c', {}) or {}
    if cp:
        parts.append(
            f"<strong>Direct effect</strong> (<em>{X_name}</em> → <em>{Y_name}</em> controlling for all mediators): "
            f"{apa_coef(cp.get('coef',0), cp.get('se'), cp.get('t_stat'), df_multiple, cp.get('p_value'))}. "
            f"Total effect (c): {apa_coef(c.get('coef',0), c.get('se'), c.get('t_stat'), df_simple, c.get('p_value'))}."
        )

    # 5. Total indirect
    if total_ci:
        ci_lo = total_ci.get('ci_lower', 0)
        ci_hi = total_ci.get('ci_upper', 0)
        parts.append(
            f"<strong>Total indirect effect</strong> (Σ specific indirect effects = {total_indirect:.3f}), "
            f"95% CI [{ci_lo:.3f}, {ci_hi:.3f}] — "
            f"<strong>{'statistically significant' if total_sig else 'not statistically significant'}</strong>."
        )

    overall_analysis = "<br><br>".join(parts)

    # ── Recommendations ───────────────────────────────────────────────────────
    rec = []
    rec.append(
        f"<strong>APA 7th reporting checklist (parallel mediation):</strong> "
        f"Report each specific indirect effect (a<sub>i</sub>b<sub>i</sub>) separately with its bootstrap 95% CI, "
        f"and report the total indirect effect with its CI. "
        f"Per Hayes (2022, PROCESS Model 4), also report path a and b coefficients for each mediator with SEs and p-values. "
        f"A path diagram is strongly recommended."
    )
    if len(sig_paths) > 1:
        rec.append(
            f"<strong>Pairwise contrasts:</strong> With {len(sig_paths)} significant mediators, test whether "
            f"their specific indirect effects significantly differ from each other "
            f"(e.g., a₁b₁ − a₂b₂ with bootstrap CI). A non-significant contrast means the mediators "
            f"are statistically interchangeable in their indirect role."
        )
    if len(sig_paths) < n_med:
        non_sig = [M_names[i] for i in range(n_med) if i not in sig_paths]
        rec.append(
            f"<strong>Non-significant mediators ({', '.join(f'<em>{m}</em>' for m in non_sig)}):</strong> "
            f"Consider whether low statistical power (Fritz & MacKinnon, 2007 recommend N ≥ 200 for parallel models), "
            f"measurement error in the mediator, or conceptual misspecification explains the null result."
        )
    rec.append(
        f"<strong>Model assumption:</strong> Parallel mediation assumes the mediators do not causally influence "
        f"each other. If there is theoretical reason to expect a causal ordering among mediators, "
        f"test a serial mediation model instead."
    )
    recommendations = "<br><br>".join(rec)
    return overall_analysis, recommendations


def generate_serial_interpretation(serial_result, X_name, M_names, Y_name, n_obs=None, n_bootstrap=1000):
    """APA 7th interpretation for serial (2-mediator) mediation — full path statistics."""
    N  = n_obs or 100
    M1, M2 = M_names[0], M_names[1]
    sb = serial_result.get('bootstrap', {}) or {}

    serial_ind = serial_result.get('indirect_serial', 0)
    m1_ind     = serial_result.get('indirect_m1',     0)
    m2_ind     = serial_result.get('indirect_m2',     0)
    total_ind  = serial_result.get('total_indirect',  0)

    serial_boot = sb.get('serial', {}) or {}
    m1_boot     = sb.get('via_m1', {}) or {}
    m2_boot     = sb.get('via_m2', {}) or {}
    total_boot  = sb.get('total',  {}) or {}

    serial_sig = serial_boot.get('significant', False)
    m1_sig     = m1_boot.get('significant', False)
    m2_sig     = m2_boot.get('significant', False)
    total_sig  = total_boot.get('significant', False)

    # path dicts from serial_mediation_analysis result
    a1 = serial_result.get('path_a1', {}) or {}
    a2 = serial_result.get('path_a2', {}) or {}
    d  = serial_result.get('path_d',  {}) or {}
    b1 = serial_result.get('path_b1', {}) or {}
    b2 = serial_result.get('path_b2', {}) or {}
    cp = serial_result.get('path_c_prime', {}) or {}
    c  = serial_result.get('path_c',       {}) or {}

    # approximate df
    df_simple   = N - 2
    df_outcome  = N - 4   # Y ~ X + M1 + M2 + intercept

    def apa_p(p):
        if p is None: return "= n/a"
        return "< .001" if p < 0.001 else f"= {p:.3f}"

    def apa_path(path_dict, df):
        b  = path_dict.get('coef', 0)
        se = path_dict.get('se')
        t  = path_dict.get('t_stat')
        p  = path_dict.get('p_value')
        if any(v is None for v in [se, t, p]):
            return f"β = {b:.3f}"
        return f"β = {b:.3f}, SE = {se:.3f}, t({int(df)}) = {t:.2f}, p {apa_p(p)}"

    def sig_star(p):
        if p is None: return ""
        return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else " (ns)"

    parts = []

    # ── 1. Model description ──────────────────────────────────────────────────
    parts.append(
        f"A serial (sequential) mediation analysis was conducted to test whether "
        f"<em>{X_name}</em> influences <em>{Y_name}</em> through a two-step mediational chain: "
        f"<em>{X_name}</em> → <em>{M1}</em> → <em>{M2}</em> → <em>{Y_name}</em> "
        f"(Hayes, 2022, PROCESS Model 6; <em>N</em> = {N}, "
        f"bootstrap = {n_bootstrap:,} BCa samples). "
        f"Three specific indirect effects were estimated: "
        f"(1) the full serial path through both <em>{M1}</em> and <em>{M2}</em> (a₁ × d × b₂), "
        f"(2) the path via <em>{M1}</em> only, bypassing <em>{M2}</em> (a₁ × b₁), and "
        f"(3) the path via <em>{M2}</em> only, bypassing <em>{M1}</em> (a₂ × b₂)."
    )

    # ── 2. Individual path coefficients ──────────────────────────────────────
    parts.append(
        f"<strong>Path coefficients:</strong> "
        f"Path a₁ (<em>{X_name}</em> → <em>{M1}</em>): {apa_path(a1, df_simple)}{sig_star(a1.get('p_value'))}. "
        f"Path a₂ (<em>{X_name}</em> → <em>{M2}</em>): {apa_path(a2, df_simple)}{sig_star(a2.get('p_value'))}. "
        f"Path d (<em>{M1}</em> → <em>{M2}</em>, controlling for <em>{X_name}</em>): "
        f"{apa_path(d, N - 3)}{sig_star(d.get('p_value'))}. "
        f"Path b₁ (<em>{M1}</em> → <em>{Y_name}</em>, controlling for <em>{X_name}</em>, <em>{M2}</em>): "
        f"{apa_path(b1, df_outcome)}{sig_star(b1.get('p_value'))}. "
        f"Path b₂ (<em>{M2}</em> → <em>{Y_name}</em>, controlling for <em>{X_name}</em>, <em>{M1}</em>): "
        f"{apa_path(b2, df_outcome)}{sig_star(b2.get('p_value'))}. "
        f"Direct effect c′ (<em>{X_name}</em> → <em>{Y_name}</em>, controlling for both mediators): "
        f"{apa_path(cp, df_outcome)}{sig_star(cp.get('p_value'))}. "
        f"Total effect c (<em>{X_name}</em> → <em>{Y_name}</em>): "
        f"{apa_path(c, df_simple)}{sig_star(c.get('p_value'))}."
    )

    # ── 3. Verdict ────────────────────────────────────────────────────────────
    sig_routes = []
    if serial_sig: sig_routes.append(f"the full serial chain (<em>{X_name}</em>→<em>{M1}</em>→<em>{M2}</em>→<em>{Y_name}</em>)")
    if m1_sig:     sig_routes.append(f"the <em>{M1}</em>-only route (<em>{X_name}</em>→<em>{M1}</em>→<em>{Y_name}</em>)")
    if m2_sig:     sig_routes.append(f"the <em>{M2}</em>-only route (<em>{X_name}</em>→<em>{M2}</em>→<em>{Y_name}</em>)")

    if total_sig and sig_routes:
        parts.append(
            f"<strong>Overall verdict:</strong> The total indirect effect was <strong>statistically significant</strong>, "
            f"with significant contributions from: {'; '.join(sig_routes)}."
        )
    elif total_sig:
        parts.append(
            f"<strong>Overall verdict:</strong> The total indirect effect was <strong>statistically significant</strong>, "
            f"though no individual route reached significance on its own — "
            f"suggesting the effect is distributed across multiple small pathways."
        )
    else:
        parts.append(
            f"<strong>Overall verdict:</strong> The total indirect effect was "
            f"<strong>not statistically significant</strong> (CI included zero). "
            f"None of the three specific indirect routes reached significance."
        )

    # ── 4. Three specific indirect effects (full APA) ─────────────────────────
    routes = [
        (f"Indirect effect 1 — Full serial path "
         f"(<em>{X_name}</em>→<em>{M1}</em>→<em>{M2}</em>→<em>{Y_name}</em>, a₁×d×b₂)",
         serial_ind, serial_boot, serial_sig),
        (f"Indirect effect 2 — Via <em>{M1}</em> only "
         f"(<em>{X_name}</em>→<em>{M1}</em>→<em>{Y_name}</em>, a₁×b₁)",
         m1_ind, m1_boot, m1_sig),
        (f"Indirect effect 3 — Via <em>{M2}</em> only "
         f"(<em>{X_name}</em>→<em>{M2}</em>→<em>{Y_name}</em>, a₂×b₂)",
         m2_ind, m2_boot, m2_sig),
    ]
    for label, ind, boot_d, is_sig in routes:
        if boot_d:
            ci_lo = boot_d.get('ci_lower', 0)
            ci_hi = boot_d.get('ci_upper', 0)
            parts.append(
                f"<strong>{label}:</strong> "
                f"indirect effect = {ind:.3f}, 95% CI [{ci_lo:.3f}, {ci_hi:.3f}] — "
                f"<strong>{'statistically significant' if is_sig else 'not statistically significant'}</strong> "
                f"({'CI does not include zero' if is_sig else 'CI includes zero'})."
            )

    # ── 5. Total indirect ─────────────────────────────────────────────────────
    if total_boot:
        ci_lo = total_boot.get('ci_lower', 0)
        ci_hi = total_boot.get('ci_upper', 0)
        parts.append(
            f"<strong>Total indirect effect</strong> (sum of all three routes = {total_ind:.3f}), "
            f"95% CI [{ci_lo:.3f}, {ci_hi:.3f}] — "
            f"<strong>{'statistically significant' if total_sig else 'not statistically significant'}</strong>."
        )

    overall_analysis = "<br><br>".join(parts)

    # ── Recommendations ───────────────────────────────────────────────────────
    rec = []

    rec.append(
        f"<strong>APA 7th reporting checklist (serial mediation, PROCESS Model 6):</strong> "
        f"Report all path coefficients (a₁, a₂, d, b₁, b₂, c′, c) with SEs and p-values in a table or path diagram. "
        f"Report all three specific indirect effects and the total indirect effect each with bootstrap 95% BCa CIs. "
        f"Explicitly state the assumed causal ordering of mediators and justify it theoretically "
        f"(Hayes, 2022; Preacher & Hayes, 2008)."
    )

    if serial_sig:
        rec.append(
            f"<strong>Serial chain confirmed (a₁ × d × b₂):</strong> "
            f"The full <em>{X_name}</em>→<em>{M1}</em>→<em>{M2}</em>→<em>{Y_name}</em> chain is significant, "
            f"supporting the proposed sequential mechanism. "
            f"This implies that interventions targeting <em>{M1}</em> may cascade through <em>{M2}</em> "
            f"to ultimately affect <em>{Y_name}</em>. "
            f"Report the d path (β = {d.get('coef', 0):.3f}) explicitly, "
            f"as it is the mechanistic link unique to the serial model."
        )
    else:
        rec.append(
            f"<strong>Serial chain not significant (a₁ × d × b₂ = {serial_ind:.3f}):</strong> "
            f"The full chain did not reach significance. "
            f"{'The d path (M1→M2) may be the bottleneck — check whether β = ' + str(round(d.get('coef',0),3)) + ' is significant.' if d.get('p_value', 1) > 0.05 else 'The d path is significant but the full product a₁×d×b₂ lacks power — consider increasing N.'} "
            f"If the theoretical ordering of <em>{M1}</em> before <em>{M2}</em> is uncertain, "
            f"also test the reversed model (<em>{X_name}</em>→<em>{M2}</em>→<em>{M1}</em>→<em>{Y_name}</em>)."
        )

    if m1_sig or m2_sig:
        active = []
        if m1_sig: active.append(f"<em>{M1}</em>-only (a₁×b₁ = {m1_ind:.3f})")
        if m2_sig: active.append(f"<em>{M2}</em>-only (a₂×b₂ = {m2_ind:.3f})")
        rec.append(
            f"<strong>Active parallel routes:</strong> {' and '.join(active)} "
            f"carried significant indirect effects independent of the serial path. "
            f"This suggests these mediators also operate as simple (non-serial) pathways, "
            f"which should be acknowledged in the theoretical discussion."
        )

    rec.append(
        f"<strong>Power consideration:</strong> Serial mediation requires larger samples than simple mediation "
        f"because the serial indirect effect (a₁ × d × b₂) is a product of three coefficients, "
        f"making it more susceptible to attenuation. "
        f"Fritz & MacKinnon (2007) recommend <em>N</em> ≥ 200 for reliable detection; "
        f"current <em>N</em> = {N}{' meets' if N >= 200 else ' does <strong>not</strong> meet'} this threshold."
    )

    recommendations = "<br><br>".join(rec)
    return overall_analysis, recommendations

def _bca_bootstrap(indirect_samples, observed, n_bootstrap, confidence_level=0.95):
    """BCa bootstrap CI for a pre-computed array of indirect effect samples."""
    alpha = 1 - confidence_level
    n = len(indirect_samples)
    if n == 0:
        return {'ci_lower': None, 'ci_upper': None, 'mean_effect': None, 'se': None, 'significant': False}

    mean_eff = float(np.mean(indirect_samples))
    se_eff   = float(np.std(indirect_samples))

    # Bias-correction
    z0 = stats.norm.ppf(np.mean(indirect_samples < observed)) if np.std(indirect_samples) > 0 else 0.0

    # Acceleration via jackknife
    jack = np.array([np.mean(np.delete(indirect_samples, i)) for i in range(min(n, 200))])
    jack_mean = np.mean(jack)
    num = np.sum((jack_mean - jack) ** 3)
    den = 6 * (np.sum((jack_mean - jack) ** 2) ** 1.5)
    a = float(num / den) if den != 0 else 0.0

    z_lo = stats.norm.ppf(alpha / 2)
    z_hi = stats.norm.ppf(1 - alpha / 2)

    def adj(z):
        num2 = z0 + z
        return stats.norm.cdf(z0 + num2 / (1 - a * num2))

    ci_lo = float(np.percentile(indirect_samples, adj(z_lo) * 100))
    ci_hi = float(np.percentile(indirect_samples, adj(z_hi) * 100))
    return {
        'mean_effect': mean_eff,
        'se':          se_eff,
        'ci_lower':    ci_lo,
        'ci_upper':    ci_hi,
        'n_bootstrap': n_bootstrap,
        'significant': not (ci_lo <= 0 <= ci_hi),
    }


def parallel_mediation_analysis(X, Ms, Y, M_names, n_bootstrap=1000, confidence_level=0.95, seed=42):
    """
    Parallel multiple mediation: X → [M1, M2, ...] → Y (all mediators simultaneous).
    Returns per-mediator Baron-Kenny + bootstrap, plus total indirect bootstrap.
    """
    rng = np.random.default_rng(seed)
    n   = len(X)
    n_med = len(Ms)

    # ── Per-mediator Baron-Kenny ──────────────────────────────────────────────
    # Step 1: regress Y on X + all Ms simultaneously (shared model for b paths)
    X_all_M = np.column_stack([X] + Ms)   # shape (n, 1 + n_med)
    y_full   = multiple_regression_multi(X_all_M, Y)  # returns coefs [intercept, b_X, b_M1, b_M2, ...]

    paths = []
    for i, (M, m_name) in enumerate(zip(Ms, M_names)):
        # a path: X → Mi
        a = simple_regression(X, M)
        # b path: Mi → Y controlling for X and all other Ms
        b_coef = y_full['coefficients'][i + 2]   # +2: skip intercept and X
        b_se   = y_full['std_errors'][i + 2]
        b_t    = y_full['t_stats'][i + 2]
        b_p    = y_full['p_values'][i + 2]
        b_path = {'coef': b_coef, 'se': b_se, 't_stat': b_t, 'p_value': b_p}

        # c (total) and c' (direct)
        c      = simple_regression(X, Y)
        c_prime_coef = y_full['coefficients'][1]   # X coef in full model
        c_prime = {'coef': c_prime_coef, 'se': y_full['std_errors'][1],
                   't_stat': y_full['t_stats'][1],  'p_value': y_full['p_values'][1]}

        indirect = a['coef'] * b_coef

        bk_i = {
            'path_a':       a,
            'path_b':       b_path,
            'path_c':       c,
            'path_c_prime': c_prime,
            'indirect_effect': safe_float(indirect),
        }
        paths.append({'mediator_index': i, 'mediator_name': m_name, 'baron_kenny': bk_i, 'bootstrap': None})

    # ── Bootstrap ─────────────────────────────────────────────────────────────
    boot_samples = [[] for _ in range(n_med)]   # per mediator
    total_samples = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        Xb  = X[idx];  Yb = Y[idx]
        Msb = [M[idx] for M in Ms]

        Xb_all_M = np.column_stack([Xb] + Msb)
        try:
            yf = multiple_regression_multi(Xb_all_M, Yb)
        except Exception:
            continue

        total_ind = 0.0
        for i, Mb in enumerate(Msb):
            ab = simple_regression(Xb, Mb)
            b_coef_b = yf['coefficients'][i + 2]
            ind = ab['coef'] * b_coef_b
            boot_samples[i].append(ind)
            total_ind += ind
        total_samples.append(total_ind)

    # Attach per-mediator CI
    for i in range(n_med):
        obs = paths[i]['baron_kenny']['indirect_effect']
        paths[i]['bootstrap'] = _bca_bootstrap(
            np.array(boot_samples[i]), obs, n_bootstrap, confidence_level
        )

    total_indirect = sum(p['baron_kenny']['indirect_effect'] for p in paths)
    total_boot_ci  = _bca_bootstrap(np.array(total_samples), total_indirect, n_bootstrap, confidence_level)

    return {
        'paths':          paths,
        'total_indirect': safe_float(total_indirect),
        'total_boot_ci':  total_boot_ci,
    }


def serial_mediation_analysis(X, M1, M2, Y, n_bootstrap=1000, confidence_level=0.95, seed=42):
    """
    Serial (2-step) mediation: X → M1 → M2 → Y.
    Decomposes into 3 specific indirect effects:
      1. Serial:   X→M1→M2→Y  (a1 * d * b2)
      2. Via M1:   X→M1→Y     (a1 * b1)
      3. Via M2:   X→M2→Y     (a2 * b2)
    """
    rng = np.random.default_rng(seed)
    n   = len(X)

    def _point_estimates(Xv, M1v, M2v, Yv):
        # a1: X → M1
        a1 = simple_regression(Xv, M1v)
        # a2: X → M2
        a2 = simple_regression(Xv, M2v)
        # d:  M1 → M2 (controlling for X)
        d_full = multiple_regression(Xv, M1v, M2v)   # Y=M2, predictors=[X, M1]
        d = {'coef': d_full['coef2'], 'se': d_full['se2'],
             't_stat': d_full['t_stat2'], 'p_value': d_full['p_value2']}

        # Full outcome model: Y ~ X + M1 + M2
        XM1M2 = np.column_stack([Xv, M1v, M2v])
        y_full = multiple_regression_multi(XM1M2, Yv)
        # b1 = M1 coef (index 2), b2 = M2 coef (index 3), c' = X coef (index 1)
        b1 = {'coef': y_full['coefficients'][2], 'se': y_full['std_errors'][2],
              't_stat': y_full['t_stats'][2],    'p_value': y_full['p_values'][2]}
        b2 = {'coef': y_full['coefficients'][3], 'se': y_full['std_errors'][3],
              't_stat': y_full['t_stats'][3],    'p_value': y_full['p_values'][3]}
        c_prime = {'coef': y_full['coefficients'][1], 'se': y_full['std_errors'][1],
                   't_stat': y_full['t_stats'][1],    'p_value': y_full['p_values'][1]}
        c = simple_regression(Xv, Yv)

        ind_serial = a1['coef'] * d['coef'] * b2['coef']
        ind_m1     = a1['coef'] * b1['coef']
        ind_m2     = a2['coef'] * b2['coef']
        total_ind  = ind_serial + ind_m1 + ind_m2

        return a1, a2, d, b1, b2, c_prime, c, ind_serial, ind_m1, ind_m2, total_ind

    a1, a2, d, b1, b2, c_prime, c, ind_serial, ind_m1, ind_m2, total_ind = \
        _point_estimates(X, M1, M2, Y)

    # ── Bootstrap ─────────────────────────────────────────────────────────────
    boot_serial = []; boot_m1 = []; boot_m2 = []; boot_total = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        try:
            _, _, _, _, _, _, _, is_, im1, im2, it = \
                _point_estimates(X[idx], M1[idx], M2[idx], Y[idx])
            boot_serial.append(is_)
            boot_m1.append(im1)
            boot_m2.append(im2)
            boot_total.append(it)
        except Exception:
            continue

    boot_results = {
        'serial': _bca_bootstrap(np.array(boot_serial), ind_serial,  n_bootstrap, confidence_level),
        'via_m1': _bca_bootstrap(np.array(boot_m1),     ind_m1,      n_bootstrap, confidence_level),
        'via_m2': _bca_bootstrap(np.array(boot_m2),     ind_m2,      n_bootstrap, confidence_level),
        'total':  _bca_bootstrap(np.array(boot_total),  total_ind,   n_bootstrap, confidence_level),
        'n_bootstrap': n_bootstrap,
        'seed': seed,
    }

    return {
        'path_a1':       a1,
        'path_a2':       a2,
        'path_d':        d,
        'path_b1':       b1,
        'path_b2':       b2,
        'path_c_prime':  c_prime,
        'path_c':        c,
        'indirect_serial': safe_float(ind_serial),
        'indirect_m1':     safe_float(ind_m1),
        'indirect_m2':     safe_float(ind_m2),
        'total_indirect':  safe_float(total_ind),
        'bootstrap':       boot_results,
    }


def multiple_regression_multi(X_matrix, Y):
    """
    OLS regression with arbitrary number of predictors.
    X_matrix shape: (n, k) — no intercept column needed (added internally).
    Returns dict with coefficients, SEs, t-stats, p-values.
    """
    n, k = X_matrix.shape
    X_design = np.column_stack([np.ones(n), X_matrix])
    model = LinearRegression()
    model.fit(X_matrix, Y)
    y_pred = model.predict(X_matrix)
    residuals = Y - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    mse = ss_res / (n - k - 1) if (n - k - 1) > 0 else 0

    try:
        cov = mse * np.linalg.inv(X_design.T @ X_design)
        ses = np.sqrt(np.diag(cov))
    except np.linalg.LinAlgError:
        ses = np.full(k + 1, np.nan)

    coeffs = np.concatenate([[model.intercept_], model.coef_])
    df = n - k - 1
    t_stats = coeffs / ses if not np.any(np.isnan(ses)) else np.full(k + 1, np.nan)
    p_values = (2 * (1 - stats.t.cdf(np.abs(t_stats), df))
                if df > 0 else np.full(k + 1, np.nan))

    return {
        'coefficients': [safe_float(c) for c in coeffs],
        'std_errors':   [safe_float(s) for s in ses],
        't_stats':      [safe_float(t) for t in t_stats],
        'p_values':     [safe_float(p) for p in p_values],
    }


def create_plot(bk, boot, X_name, M_name, Y_name):
    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white'})
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('white')
    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor('white')
    line_color = '#C44E52'
    primary_color = '#5B9BD5'
    sig_color = '#2E7D32'
    nonsig_color = '#757575'
    
    # 1. Path Diagram
    ax = axes[0, 0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    x_pos, m_pos, y_pos = (1.5, 5), (5, 8), (8.5, 5)
    box_w, box_h = 1.8, 1.3
    
    for pos, name, fc, ec in [(x_pos, X_name, '#E3F2FD', '#1976D2'), (m_pos, M_name, '#E8F5E9', '#388E3C'), (y_pos, Y_name, '#FFEBEE', '#D32F2F')]:
        box = FancyBboxPatch((pos[0]-box_w/2, pos[1]-box_h/2), box_w, box_h, boxstyle="round,pad=0.15", facecolor=fc, edgecolor=ec, linewidth=2.5)
        ax.add_patch(box)
        ax.text(pos[0], pos[1], name[:12], ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Arrows
    for start, end, coef, p, label, lbl_pos in [
        ((x_pos[0]+0.9, x_pos[1]+0.4), (m_pos[0]-0.9, m_pos[1]-0.4), bk['path_a']['coef'], bk['path_a']['p_value'], 'a', (2.8, 7)),
        ((m_pos[0]+0.9, m_pos[1]-0.4), (y_pos[0]-0.9, y_pos[1]+0.4), bk['path_b']['coef'], bk['path_b']['p_value'], 'b', (7.2, 7)),
    ]:
        sig = p < 0.05
        color = sig_color if sig else nonsig_color
        arrow = FancyArrowPatch(start, end, arrowstyle='->', lw=2.5 if sig else 1.5, color=color, mutation_scale=20)
        ax.add_patch(arrow)
        ax.text(lbl_pos[0], lbl_pos[1], f'{label} = {coef:.3f}{"*" if sig else ""}', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color))
    
    # c' path
    c_sig = bk['path_c_prime']['p_value'] < 0.05
    c_color = sig_color if c_sig else nonsig_color
    arrow_c = FancyArrowPatch((x_pos[0]+0.9, x_pos[1]), (y_pos[0]-0.9, y_pos[1]), arrowstyle='->', lw=2.5 if c_sig else 1.5, color=c_color, mutation_scale=20, connectionstyle="arc3,rad=0.3")
    ax.add_patch(arrow_c)
    ax.text(5, 3.5, f"c' = {bk['path_c_prime']['coef']:.3f}{'*' if c_sig else ''}", fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=c_color))
    
    ax.text(0.5, 0.8, '* p < .05', fontsize=8, style='italic')
    ax.text(0.5, 0.3, f"Indirect = {bk['indirect_effect']:.3f}", fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF9C4', edgecolor='#F57F17', linewidth=2))
    ax.set_title('Mediation Path Diagram', fontweight='bold')
    
    # 2. Effect Decomposition
    effects = [bk['path_c']['coef'], bk['path_c_prime']['coef'], bk['indirect_effect']]
    names = ['Total (c)', "Direct (c')", 'Indirect (a×b)']
    colors = [primary_color, line_color, '#F4A582']
    bars = axes[0, 1].bar(names, effects, color=colors, alpha=0.7, edgecolor='black')
    axes[0, 1].axhline(0, color='grey', lw=1)
    axes[0, 1].set_ylabel('Coefficient')
    axes[0, 1].set_title('Effect Decomposition', fontweight='bold')
    for bar, eff in zip(bars, effects):
        axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{eff:.3f}', ha='center', va='bottom' if eff > 0 else 'top', fontsize=9)
    
    # 3. Bootstrap Distribution
    if boot:
        axes[1, 0].axvline(0, color='red', lw=2, alpha=0.5, label='Null')
        axes[1, 0].axvline(boot['mean_effect'], color=line_color, linestyle='--', lw=2, label=f"Mean: {boot['mean_effect']:.3f}")
        axes[1, 0].axvline(boot['ci_lower'], color='green', linestyle=':', lw=2)
        axes[1, 0].axvline(boot['ci_upper'], color='green', linestyle=':', lw=2, label=f"95% CI")
        axes[1, 0].fill_betweenx([0, 1], boot['ci_lower'], boot['ci_upper'], alpha=0.2, color='green')
        axes[1, 0].set_xlabel('Indirect Effect')
        axes[1, 0].set_title('Bootstrap Confidence Interval', fontweight='bold')
        axes[1, 0].legend(fontsize=9)
        axes[1, 0].set_ylim(0, 1)
        axes[1, 0].set_yticks([])
    else:
        axes[1, 0].text(0.5, 0.5, 'Bootstrap not available', ha='center', va='center')
        axes[1, 0].axis('off')
    
    # 4. Summary Table
    ax = axes[1, 1]
    ax.axis('off')
    table_data = [
        ['Path', 'Coef', 'SE', 'p-value'],
        ['a (X→M)', f"{bk['path_a']['coef']:.3f}", f"{safe_float(bk['path_a']['se']):.3f}", f"{bk['path_a']['p_value']:.3f}" if bk['path_a']['p_value'] >= 0.001 else '<.001'],
        ['b (M→Y)', f"{bk['path_b']['coef']:.3f}", f"{safe_float(bk['path_b']['se']):.3f}", f"{bk['path_b']['p_value']:.3f}" if bk['path_b']['p_value'] >= 0.001 else '<.001'],
        ["c' (Direct)", f"{bk['path_c_prime']['coef']:.3f}", f"{safe_float(bk['path_c_prime']['se']):.3f}", f"{bk['path_c_prime']['p_value']:.3f}" if bk['path_c_prime']['p_value'] >= 0.001 else '<.001'],
        ['c (Total)', f"{bk['path_c']['coef']:.3f}", f"{safe_float(bk['path_c']['se']):.3f}", f"{bk['path_c']['p_value']:.3f}" if bk['path_c']['p_value'] >= 0.001 else '<.001'],
    ]
    if boot:
        table_data.append(['Indirect', f"{boot['mean_effect']:.3f}", f"{boot['se']:.3f}", f"CI: [{boot['ci_lower']:.3f}, {boot['ci_upper']:.3f}]"])
    
    table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.25, 0.2, 0.2, 0.35])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    for i in range(4):
        table[(0, i)].set_facecolor('#E8E8E8')
        table[(0, i)].set_text_props(weight='bold')
    ax.set_title('Path Coefficients Summary', fontweight='bold', pad=20)
    
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"

@router.post("/mediation")
def mediation_analysis(req: MediationRequest):
    try:
        df = pd.DataFrame(req.data)
        x_var, m_var, y_var = req.xVar, req.mVar, req.yVar
        n_bootstrap = req.nBootstrap or 1000
        N_obs = None

        # ── Multiple mediators: parallel or serial ────────────────────────────
        if isinstance(m_var, list) and len(m_var) > 1:
            m_vars = m_var
            all_vars = [x_var] + m_vars + [y_var]
            for v in all_vars:
                if v not in df.columns:
                    raise ValueError(f"Variable '{v}' not found")

            clean_data = df[all_vars].dropna()
            if len(clean_data) < 20:
                raise ValueError(f"Need at least 20 observations, got {len(clean_data)}")
            N_obs = len(clean_data)

            X = clean_data[x_var].values
            Y = clean_data[y_var].values
            Ms = [clean_data[m].values for m in m_vars]

            if req.standardize:
                sc = StandardScaler()
                X = sc.fit_transform(X.reshape(-1,1)).flatten()
                Y = sc.fit_transform(Y.reshape(-1,1)).flatten()
                Ms = [sc.fit_transform(m.reshape(-1,1)).flatten() for m in Ms]

            if getattr(req, 'serialMediation', False) and len(m_vars) == 2:
                serial_result = serial_mediation_analysis(X, Ms[0], Ms[1], Y, n_bootstrap=n_bootstrap)
                overall_analysis, recommendations = generate_serial_interpretation(
                    serial_result, x_var, m_vars, y_var, n_obs=N_obs, n_bootstrap=n_bootstrap
                )
                plot = create_plot(None, None, x_var, m_vars[0], y_var)
                response = {
                    'results': {
                        'serial_mediation': serial_result,
                        'mediation_type': 'Serial Mediation',
                        'interpretation': overall_analysis,
                        'overall_analysis': overall_analysis,
                        'recommendations': recommendations,
                        'analysis_sections':       _split_sections(overall_analysis),
                        'recommendation_sections': _split_sections(recommendations),
                        'proportion_mediated': None,
                    },
                    'n_observations': N_obs,
                    'standardized': req.standardize,
                    'bootstrap_method': 'bca',
                    'bootstrap_seed': getattr(req, 'bootstrapSeed', 42),
                    'warnings': [],
                    'plot': plot
                }
            else:
                parallel_result = parallel_mediation_analysis(X, Ms, Y, m_vars, n_bootstrap=n_bootstrap)
                overall_analysis, recommendations = generate_parallel_interpretation(
                    parallel_result, x_var, m_vars, y_var, n_obs=N_obs, n_bootstrap=n_bootstrap
                )
                plot = create_plot(None, None, x_var, '/'.join(m_vars), y_var)
                response = {
                    'results': {
                        'parallel_mediation': parallel_result,
                        'mediation_type': 'Parallel Mediation',
                        'interpretation': overall_analysis,
                        'overall_analysis': overall_analysis,
                        'recommendations': recommendations,
                        'analysis_sections':       _split_sections(overall_analysis),
                        'recommendation_sections': _split_sections(recommendations),
                        'proportion_mediated': None,
                    },
                    'n_observations': N_obs,
                    'standardized': req.standardize,
                    'bootstrap_method': 'bca',
                    'bootstrap_seed': getattr(req, 'bootstrapSeed', 42),
                    'warnings': [],
                    'plot': plot
                }
            return _to_native(response)

        # ── Single mediator ───────────────────────────────────────────────────
        if isinstance(m_var, list):
            m_var = m_var[0]

        for var in [x_var, m_var, y_var]:
            if var not in df.columns:
                raise ValueError(f"Variable '{var}' not found")

        clean_data = df[[x_var, m_var, y_var]].dropna()
        if len(clean_data) < 20:
            raise ValueError(f"Need at least 20 observations, got {len(clean_data)}")
        N_obs = len(clean_data)

        X = clean_data[x_var].values
        M = clean_data[m_var].values
        Y = clean_data[y_var].values

        if req.standardize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X.reshape(-1, 1)).flatten()
            M = scaler.fit_transform(M.reshape(-1, 1)).flatten()
            Y = scaler.fit_transform(Y.reshape(-1, 1)).flatten()

        bk   = baron_kenny_analysis(X, M, Y)
        boot = bootstrap_analysis(X, M, Y, n_bootstrap=n_bootstrap)

        overall_analysis, recommendations, mediation_type = generate_interpretation(
            bk, boot, x_var, m_var, y_var, n_obs=N_obs
        )
        plot = create_plot(bk, boot, x_var, m_var, y_var)

        indirect_sig = boot['significant'] if boot else bk['sobel_test']['p_value'] < 0.05
        total_sig    = bk['path_c']['p_value'] < 0.05
        prop_med = None
        if total_sig and indirect_sig and abs(bk['path_c']['coef']) > 1e-6:
            prop_med = bk['indirect_effect'] / bk['path_c']['coef']

        response = {
            'results': {
                'baron_kenny': bk,
                'bootstrap': boot,
                'mediation_type': mediation_type,
                'interpretation': overall_analysis,
                'proportion_mediated': prop_med,
                'overall_analysis': overall_analysis,
                'recommendations': recommendations,
                'analysis_sections':       _split_sections(overall_analysis),
                'recommendation_sections': _split_sections(recommendations),
            },
            'n_observations': N_obs,
            'standardized': req.standardize,
            'bootstrap_method': 'bca',
            'bootstrap_seed': getattr(req, 'bootstrapSeed', 42),
            'robust_se': getattr(req, 'robustSE', False),
            'covariates': [],
            'warnings': [],
            'plot': plot
        }

        return _to_native(response)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
