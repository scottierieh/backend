from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, Optional
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import io, base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="whitegrid")

router = APIRouter()

class ModerationRequest(BaseModel):
    data: list[dict[str, Any]] = Field(...)
    xVar: str = Field(...)
    yVar: str = Field(...)
    mVar: str = Field(...)
    centerMethod: Optional[str] = 'mean'
    categoricalModerator: Optional[bool] = False

# ── Utilities ──────────────────────────────────────────────────────────────────

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
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))): return default
        return float(val)
    except: return default

def center_variable(arr, method='mean'):
    if method == 'none':        return arr
    elif method == 'mean':      return arr - np.mean(arr)
    elif method == 'standardize': return (arr - np.mean(arr)) / np.std(arr)
    return arr

# ── FIX 3: Categorical moderator ──────────────────────────────────────────────

def prepare_moderator(M_raw, categorical=False):
    """
    Returns (M_col, mod_values, mod_labels, is_categorical, categories).
    Binary variables are auto-detected; categorical flag forces treatment.
    """
    unique_vals = np.sort(np.unique(M_raw))
    n_unique = len(unique_vals)

    if n_unique == 2 or (categorical and n_unique <= 10):
        ref = unique_vals[0]
        dummy = (M_raw != ref).astype(float)
        return dummy, [0.0, 1.0], [str(unique_vals[0]), str(unique_vals[-1])], True, unique_vals
    else:
        m_centered = M_raw - np.mean(M_raw)
        m_std = np.std(M_raw)
        return m_centered, [-m_std, 0.0, m_std], ['Low (−1 SD)', 'Mean', 'High (+1 SD)'], False, None

# ── statsmodels regression ─────────────────────────────────────────────────────

def run_ols(predictors_df: pd.DataFrame, outcome: np.ndarray):
    """
    Run OLS via statsmodels. Returns fitted model + structured result dict.
    statsmodels handles:
      - numerically stable QR-based inversion (no singular matrix risk)
      - heteroskedasticity-robust SE via HC3 if requested
      - full covariance matrix for downstream use (simple slopes)
    """
    X_with_const = sm.add_constant(predictors_df, has_constant='add')
    model = sm.OLS(outcome, X_with_const)
    fit   = model.fit()          # standard OLS; use fit_regularized for ill-conditioned data

    n   = int(fit.nobs)
    k   = int(fit.df_model)
    df  = int(fit.df_resid)

    return {
        'fit':          fit,                          # full statsmodels object
        'coefficients': [safe_float(c) for c in fit.params],
        'std_errors':   [safe_float(s) for s in fit.bse],
        't_stats':      [safe_float(t) for t in fit.tvalues],
        'p_values':     [safe_float(p) for p in fit.pvalues],
        'r_squared':    safe_float(fit.rsquared),
        'adj_r_squared':safe_float(fit.rsquared_adj),
        'f_stat':       safe_float(fit.fvalue),
        'f_p_value':    safe_float(fit.f_pvalue),
        'aic':          safe_float(fit.aic),
        'bic':          safe_float(fit.bic),
        'df': df, 'k': k, 'n': n,
        'cov_params': fit.cov_params().values,        # full covariance matrix (numpy array)
    }

# ── FIX 2: VIF via statsmodels ────────────────────────────────────────────────

def compute_vif(predictors_df: pd.DataFrame) -> dict:
    """
    Use statsmodels variance_inflation_factor().
    More robust than manual R² inversion — handles near-singular cases gracefully.
    """
    X = sm.add_constant(predictors_df, has_constant='add').values
    vifs = {}
    for i, col in enumerate(predictors_df.columns):
        try:
            v = variance_inflation_factor(X, i + 1)  # +1 to skip intercept column
            vifs[col] = safe_float(v)
        except Exception:
            vifs[col] = None
    return vifs

# ── FIX 1: Simple slopes with full Cov(b1, b3) from statsmodels cov_params ───

def compute_simple_slopes(step2: dict, mod_values: list, mod_labels: list) -> list:
    """
    Var(b1 + b3*M) = Var(b1) + M² Var(b3) + 2M Cov(b1, b3)
    Indices in [const=0, X=1, M=2, X*M=3].
    cov_params is taken directly from statsmodels fit — numerically stable.
    """
    b1   = step2['coefficients'][1]
    b3   = step2['coefficients'][3]
    df   = step2['df']
    cov  = step2['cov_params']          # full covariance matrix

    simple_slopes = []
    for m_val, label in zip(mod_values, mod_labels):
        slope = b1 + b3 * m_val

        var_b1   = cov[1, 1]
        var_b3   = cov[3, 3]
        cov_b1b3 = cov[1, 3]
        var_slope = var_b1 + (m_val ** 2) * var_b3 + 2 * m_val * cov_b1b3
        se_slope  = float(np.sqrt(max(var_slope, 0.0)))

        t_stat  = slope / se_slope if se_slope > 0 else np.inf
        p_value = 2 * (1 - stats.t.cdf(np.abs(t_stat), df)) if df > 0 else np.nan

        simple_slopes.append({
            'label':     label,
            'slope':     safe_float(slope),
            'std_error': safe_float(se_slope),
            't_stat':    safe_float(t_stat),
            'p_value':   safe_float(p_value),
            'ci_lower':  safe_float(slope - 1.96 * se_slope),
            'ci_upper':  safe_float(slope + 1.96 * se_slope),
        })
    return simple_slopes

# ── APA 7th Interpretation ────────────────────────────────────────────────────

def generate_interpretation(results, X_name, Y_name, M_name, N,
                             is_categorical=False, categories=None):
    step1, step2 = results['step1'], results['step2']
    r2c = results['r_squared_change']

    def apa_p(p): return "< .001" if p < 0.001 else f"= {p:.3f}"
    def apa_coef(b, se, t, df, p):
        return f"B = {b:.3f}, SE = {se:.3f}, t({int(df)}) = {t:.2f}, p {apa_p(p)}"

    b3      = step2['coefficients'][3]
    b3_p    = step2['p_values'][3]
    int_sig = b3_p < 0.05

    parts = []

    # 1. Model description
    mod_type    = "categorical" if is_categorical else "continuous"
    center_note = ("" if is_categorical else
                   " Variables were mean-centered prior to computing the interaction term "
                   "to reduce non-essential multicollinearity (Aiken & West, 1991).")
    parts.append(
        f"A hierarchical multiple regression was conducted using <em>statsmodels</em> OLS "
        f"to test whether <em>{M_name}</em> ({mod_type} moderator) moderates the relationship "
        f"between <em>{X_name}</em> and <em>{Y_name}</em> (<em>N</em> = {N}).{center_note}"
    )

    # 2. Step 1: main effects
    s1_sig = "significant" if step1['f_p_value'] < 0.05 else "non-significant"
    parts.append(
        f"<strong>Step 1 (main effects):</strong> "
        f"<em>R</em>² = {step1['r_squared']:.3f}, adj. <em>R</em>² = {step1['adj_r_squared']:.3f}, "
        f"<em>F</em>({step1['k']}, {step1['df']}) = {step1['f_stat']:.2f}, "
        f"<em>p</em> {apa_p(step1['f_p_value'])}, AIC = {step1['aic']:.1f}. "
        f"Main effect of <em>{X_name}</em>: "
        f"{apa_coef(step1['coefficients'][1], step1['std_errors'][1], step1['t_stats'][1], step1['df'], step1['p_values'][1])}. "
        f"Main effect of <em>{M_name}</em>: "
        f"{apa_coef(step1['coefficients'][2], step1['std_errors'][2], step1['t_stats'][2], step1['df'], step1['p_values'][2])}."
    )

    # 3. Step 2: interaction
    sig_word = "significantly" if r2c['p_change'] < 0.05 else "non-significantly"
    parts.append(
        f"<strong>Step 2 (+ interaction):</strong> Adding the <em>{X_name}</em> × <em>{M_name}</em> term "
        f"{sig_word} improved model fit, "
        f"Δ<em>R</em>² = {r2c['delta_r2']:.3f}, "
        f"Δ<em>F</em>(1, {step2['df']}) = {r2c['f_change']:.2f}, <em>p</em> {apa_p(r2c['p_change'])}. "
        f"Full model: <em>R</em>² = {step2['r_squared']:.3f}, adj. <em>R</em>² = {step2['adj_r_squared']:.3f}, "
        f"AIC = {step2['aic']:.1f}, BIC = {step2['bic']:.1f}."
    )

    # 4. Interaction coefficient
    direction = (
        f"a positive coefficient (B = {b3:.3f}) indicates the effect of <em>{X_name}</em> "
        f"<strong>strengthens</strong> as <em>{M_name}</em> increases"
        if b3 > 0 else
        f"a negative coefficient (B = {b3:.3f}) indicates the effect of <em>{X_name}</em> "
        f"<strong>weakens or reverses</strong> as <em>{M_name}</em> increases"
    )
    parts.append(
        f"<strong>Interaction (<em>{X_name}</em> × <em>{M_name}</em>):</strong> "
        f"{apa_coef(b3, step2['std_errors'][3], step2['t_stats'][3], step2['df'], b3_p)}. "
        f"The interaction was <strong>{'statistically significant' if int_sig else 'not statistically significant'}</strong> — "
        + (direction + "." if int_sig else
           f"the effect of <em>{X_name}</em> on <em>{Y_name}</em> "
           f"does not significantly vary across levels of <em>{M_name}</em>.")
    )

    # 5. Simple slopes — with substantive narrative
    if 'simple_slopes' in results and int_sig:
        slopes = results['simple_slopes']
        lines = []
        for s in slopes:
            sig = "significant" if s['p_value'] < 0.05 else "not significant"
            lines.append(
                f"<em>{M_name}</em> = {s['label']}: B = {s['slope']:.3f}, SE = {s['std_error']:.3f}, "
                f"t({step2['df']}) = {s['t_stat']:.2f}, p {apa_p(s['p_value'])}, "
                f"95% CI [{s['ci_lower']:.3f}, {s['ci_upper']:.3f}] ({sig})"
            )
        parts.append(
            f"<strong>Simple slopes</strong> (SE from statsmodels full covariance matrix; "
            f"includes Cov(b₁, b₃) term per Aiken & West, 1991): "
            + "; ".join(lines) + "."
        )

        # Determine interaction pattern (shared logic used again in bottom line)
        sv = [s['slope'] for s in slopes]
        sf = [s['p_value'] < 0.05 for s in slopes]
        labels = [s['label'] for s in slopes]
        sig_slope_vals = [sv[i] for i in range(len(sv)) if sf[i]]
        _crossover = len(sig_slope_vals) >= 2 and len(set(v > 0 for v in sig_slope_vals)) > 1

        if _crossover:
            sig_labels    = [labels[i] for i in range(len(slopes)) if sf[i]]
            nonsig_labels = [labels[i] for i in range(len(slopes)) if not sf[i]]
            parts.append(
                f"<strong>Simple slopes interpretation:</strong> The significant slopes have opposing signs "
                f"(values: {', '.join(f'{v:.3f}' for v in sv)}), indicating a <em>crossover interaction</em>. "
                f"The effect of <em>{X_name}</em> on <em>{Y_name}</em> reverses direction across levels of <em>{M_name}</em> "
                f"— positive at {'high' if sig_slope_vals[-1] > 0 else 'low'} <em>{M_name}</em>, "
                f"negative at {'low' if sig_slope_vals[-1] > 0 else 'high'} <em>{M_name}</em>. "
                f"This has strong theoretical implications: the direction of <em>{X_name}</em>'s effect "
                f"depends entirely on the level of <em>{M_name}</em>."
            )
        elif all(sf):
            direction_desc = "strengthens" if sv[-1] > sv[0] else "weakens"
            parts.append(
                f"<strong>Simple slopes interpretation:</strong> The effect of <em>{X_name}</em> on "
                f"<em>{Y_name}</em> is statistically significant at all three levels of <em>{M_name}</em> "
                f"and consistently {direction_desc} as <em>{M_name}</em> increases "
                f"(B = {sv[0]:.3f} at low, {sv[1]:.3f} at mean, {sv[2]:.3f} at high). "
                f"This is an <em>enhancing</em> moderation pattern — "
                f"<em>{M_name}</em> amplifies the <em>{X_name}</em> → <em>{Y_name}</em> relationship "
                f"without changing its direction."
            )
        elif not any(sf):
            parts.append(
                f"<strong>Simple slopes interpretation:</strong> Despite the significant interaction term, "
                f"none of the simple slopes reached significance at the three probed values of <em>{M_name}</em>. "
                f"The significant interaction may be driven by regions outside ±1 SD — "
                f"a Johnson-Neyman analysis is recommended to locate the precise significance boundary."
            )
        else:
            sig_labels    = [labels[i] for i in range(len(slopes)) if sf[i]]
            nonsig_labels = [labels[i] for i in range(len(slopes)) if not sf[i]]
            buffer_end    = 'lower' if not sf[0] else 'higher'
            parts.append(
                f"<strong>Simple slopes interpretation:</strong> The effect of <em>{X_name}</em> on "
                f"<em>{Y_name}</em> is significant at {', '.join(sig_labels)} levels of <em>{M_name}</em>, "
                f"but non-significant at {', '.join(nonsig_labels)} "
                f"(B = {sv[[i for i,s in enumerate(sf) if not s][0]]:.3f}). "
                f"This is an <em>attenuating</em> moderation pattern — "
                f"<em>{M_name}</em> buffers the <em>{X_name}</em> → <em>{Y_name}</em> relationship "
                f"at {buffer_end} levels."
            )

    # 6. Effect size
    if 'effect_size' in results:
        f2   = results['effect_size']['f_squared']
        size = "large" if f2 >= 0.35 else "medium" if f2 >= 0.15 else "small" if f2 >= 0.02 else "negligible"
        parts.append(
            f"<strong>Effect size:</strong> Cohen's <em>f</em>² = {f2:.3f} ({size} effect; "
            f"Cohen, 1988: small ≥ .02, medium ≥ .15, large ≥ .35)."
        )

    # 7. VIF
    if 'vif' in results:
        vif     = results['vif']
        concern = any(v > 10 for v in vif.values() if v is not None)
        vif_str = ", ".join(f"<em>{k}</em>: {v:.2f}" for k, v in vif.items() if v is not None)
        parts.append(
            f"<strong>Multicollinearity (VIF, via statsmodels):</strong> {vif_str}. "
            + ("⚠️ VIF > 10 detected — multicollinearity is a concern; mean-centering is strongly advised."
               if concern else "All VIFs < 10; multicollinearity is not a concern.")
        )

    # 8. Bottom line summary
    f2 = results.get('effect_size', {}).get('f_squared', 0)
    es_word = "large" if f2 >= 0.35 else "medium" if f2 >= 0.15 else "small" if f2 >= 0.02 else "negligible"
    if int_sig:
        if 'simple_slopes' in results:
            sv = [s['slope'] for s in results['simple_slopes']]
            sf = [s['p_value'] < 0.05 for s in results['simple_slopes']]
            sig_slope_vals = [sv[i] for i in range(len(sv)) if sf[i]]
            if len(sig_slope_vals) >= 2 and len(set(v > 0 for v in sig_slope_vals)) > 1:
                pattern = "crossover (sign-reversing)"
            elif all(sf):
                pattern = "enhancing (significant at all moderator levels)"
            else:
                pattern = "attenuating (significant only at some moderator levels)"
        else:
            pattern = "significant"
        parts.append(
            f"<strong>Bottom line:</strong> <em>{M_name}</em> is a statistically significant moderator of the "
            f"<em>{X_name}</em> → <em>{Y_name}</em> relationship (<em>f</em>² = {f2:.3f}, {es_word} effect). "
            f"The moderation pattern is <em>{pattern}</em>. "
            f"The interaction explains an additional {r2c['delta_r2']*100:.1f}% of variance in <em>{Y_name}</em> "
            f"beyond the main effects alone."
        )
    else:
        parts.append(
            f"<strong>Bottom line:</strong> <em>{M_name}</em> did not significantly moderate the "
            f"<em>{X_name}</em> → <em>{Y_name}</em> relationship "
            f"(Δ<em>R</em>² = {r2c['delta_r2']:.3f}, <em>f</em>² = {f2:.3f}, {es_word} effect). "
            f"The effect of <em>{X_name}</em> on <em>{Y_name}</em> appears to be consistent "
            f"regardless of <em>{M_name}</em> level."
        )

    overall_analysis = " ".join(parts)

    # ── Recommendations ───────────────────────────────────────────────────────
    rec = []
    rec.append(
        f"<strong>APA 7th reporting checklist:</strong> Present results in two hierarchical steps. "
        f"Step 1: <em>R</em>², adj. <em>R</em>², <em>F</em>, main effect Bs with SEs and <em>p</em>-values. "
        f"Step 2: Δ<em>R</em>², Δ<em>F</em>(1, df), interaction coefficient (B, SE, t, p). "
        f"If interaction is significant, report simple slopes at ±1 SD of moderator with 95% CIs "
        f"and include an interaction plot (Aiken & West, 1991; Hayes, 2022). "
        f"Standard errors are computed from the full statsmodels covariance matrix including the "
        f"Cov(b₁, b₃) cross-term — this is the methodologically correct approach."
    )

    if int_sig:
        rec.append(
            f"<strong>Interaction direction:</strong> {'Strengthening' if b3 > 0 else 'Weakening/attenuating'} moderation detected. "
            f"{'Examine whether simple slopes cross zero — a crossover interaction carries different theoretical implications than a purely attenuating one.' if b3 < 0 else 'Confirm the simple slopes are all in the expected direction and of substantively meaningful magnitude.'}"
        )
        rec.append(
            f"<strong>Johnson-Neyman technique:</strong> For continuous moderators, compute the "
            f"Johnson-Neyman significance region to identify the exact range of <em>{M_name}</em> "
            f"values where the <em>{X_name}</em> → <em>{Y_name}</em> slope is significant at α = .05. "
            f"This is more precise than the ±1 SD pick-a-point approach."
        )
        if is_categorical and categories is not None:
            rec.append(
                f"<strong>Categorical moderator note:</strong> <em>{M_name}</em> was dummy-coded "
                f"(reference = {categories[0]}). The interaction coefficient represents the "
                f"difference in the <em>{X_name}</em> → <em>{Y_name}</em> slope between "
                f"{categories[-1]} and {categories[0]}. Report the coding scheme in the Methods section."
            )
    else:
        rec.append(
            f"<strong>Null moderation:</strong> Δ<em>R</em>² = {r2c['delta_r2']:.3f}, "
            f"p {apa_p(r2c['p_change'])}. Moderation effects are typically small and require "
            f"large samples (N ≥ 200+; Fairchild & MacKinnon, 2009). Consider: "
            f"(1) alternative moderators, (2) a moderated mediation model (PROCESS Model 7/14), "
            f"or (3) increasing statistical power via a larger sample or repeated measures design."
        )

    if not is_categorical:
        rec.append(
            f"<strong>Mean-centering:</strong> Centering does <em>not</em> change the interaction "
            f"coefficient, its SE, or significance — only the interpretation of main effects changes "
            f"(they become conditional effects at the mean; Aiken & West, 1991)."
        )

    recommendations = " ".join(rec)
    return overall_analysis, recommendations

# ── Plot ──────────────────────────────────────────────────────────────────────

def create_plot(results, X, M_col, M_raw, X_name, Y_name, M_name,
                is_categorical=False, categories=None, mod_values=None, mod_labels=None):
    plt.rcParams.update({'figure.facecolor': 'white', 'axes.facecolor': 'white'})
    model  = results['step2']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.patch.set_facecolor('white')
    for row in axes:
        for ax in row:
            ax.set_facecolor('white')

    primary = '#5B9BD5'
    red     = '#C44E52'
    green   = '#2ECC71'
    colors  = [primary, green, red, '#F4A582', '#9B59B6']

    b0, b1, b2, b3 = model['coefficients']
    x_range = np.linspace(np.min(X), np.max(X), 50)

    if mod_values is None:
        if is_categorical and categories is not None:
            mod_values = [0.0, 1.0]
            mod_labels = [str(c) for c in categories[:2]]
        else:
            m_std = np.std(M_raw)
            mod_values = [-m_std, 0.0, m_std]
            mod_labels = ['Low (−1 SD)', 'Mean', 'High (+1 SD)']

    # 1. Interaction Plot
    for i, (mv, lbl) in enumerate(zip(mod_values, mod_labels)):
        y_pred = (b0 + b2 * mv) + (b1 + b3 * mv) * x_range
        axes[0, 0].plot(x_range, y_pred, label=f"{M_name} = {lbl}",
                        color=colors[i % len(colors)], lw=2.5)
    axes[0, 0].set_xlabel(f"{'Centered ' if not is_categorical else ''}{X_name}")
    axes[0, 0].set_ylabel(Y_name)
    axes[0, 0].set_title('Interaction Plot', fontweight='bold')
    axes[0, 0].legend(fontsize=9)

    # 2. Simple Slopes with 95% CI error bars
    if 'simple_slopes' in results:
        slopes = results['simple_slopes']
        slp_vals   = [s['slope'] for s in slopes]
        slp_labels = [s['label'] for s in slopes]
        ci_lo = [s['slope'] - s['ci_lower'] for s in slopes]
        ci_hi = [s['ci_upper'] - s['slope'] for s in slopes]
        bar_colors = [colors[i % len(colors)] for i in range(len(slopes))]
        bars = axes[0, 1].bar(slp_labels, slp_vals,
                              yerr=[ci_lo, ci_hi], capsize=5,
                              alpha=0.7, color=bar_colors, edgecolor='black')
        axes[0, 1].axhline(0, color='grey', lw=1)
        axes[0, 1].set_xlabel(f'{M_name} Level')
        axes[0, 1].set_ylabel('Simple Slope (B)')
        axes[0, 1].set_title('Simple Slopes ± 95% CI', fontweight='bold')
        for bar, s in zip(bars, slopes):
            sig = '***' if s['p_value'] < 0.001 else '**' if s['p_value'] < 0.01 \
                  else '*' if s['p_value'] < 0.05 else 'ns'
            axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            f"{s['slope']:.3f}{sig}", ha='center',
                            va='bottom' if bar.get_height() >= 0 else 'top', fontsize=9)

    # 3. R² Change
    r2_s1 = results['step1']['r_squared']
    r2_s2 = results['step2']['r_squared']
    dr2   = results['r_squared_change']['delta_r2']
    axes[1, 0].bar(['Step 1\n(Main Effects)', 'Step 2\n(+ Interaction)'],
                   [r2_s1, r2_s2], alpha=0.7, color=[primary, red], edgecolor='black')
    axes[1, 0].annotate('', xy=(1, r2_s2), xytext=(1, r2_s1),
                        arrowprops=dict(arrowstyle='<->', lw=2))
    axes[1, 0].text(1.15, (r2_s1 + r2_s2) / 2, f'ΔR²={dr2:.3f}', fontsize=10, fontweight='bold')
    axes[1, 0].set_ylabel('R²')
    axes[1, 0].set_title('R² Change (Hierarchical)', fontweight='bold')
    for x_pos, h in enumerate([r2_s1, r2_s2]):
        axes[1, 0].text(x_pos, h, f'{h:.3f}', ha='center', va='bottom', fontsize=9)

    # 4. VIF Bar Chart
    ax4 = axes[1, 1]
    if 'vif' in results:
        vif_data   = results['vif']
        vif_names  = list(vif_data.keys())
        vif_vals   = [v if v is not None else 0 for v in vif_data.values()]
        vif_colors = ['#C44E52' if v > 10 else '#F4A582' if v > 5 else primary for v in vif_vals]
        bars_v = ax4.bar(vif_names, vif_vals, color=vif_colors, alpha=0.8, edgecolor='black')
        ax4.axhline(5,  color='orange', lw=1.5, linestyle='--', label='Caution (5)')
        ax4.axhline(10, color='red',    lw=1.5, linestyle='--', label='Problematic (10)')
        ax4.set_ylabel('VIF')
        ax4.set_title('Variance Inflation Factors (VIF)', fontweight='bold')
        ax4.legend(fontsize=8)
        for bar, v in zip(bars_v, vif_vals):
            ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{v:.2f}', ha='center', va='bottom', fontsize=9)
        ax4.set_xticks(range(len(vif_names)))
        ax4.set_xticklabels(vif_names, rotation=15, ha='right', fontsize=9)
    else:
        ax4.axis('off')
        ax4.text(0.5, 0.5, 'VIF not available', ha='center', va='center', transform=ax4.transAxes)

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode()}"

# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post("/moderation")
def moderation_analysis(req: ModerationRequest):
    try:
        df = pd.DataFrame(req.data)
        x_var, y_var, m_var = req.xVar, req.yVar, req.mVar

        for var in [x_var, y_var, m_var]:
            if var not in df.columns:
                raise ValueError(f"Variable '{var}' not found")

        clean_data = df[[x_var, y_var, m_var]].dropna()
        if len(clean_data) < 20:
            raise ValueError(f"Need at least 20 observations, got {len(clean_data)}")

        N     = len(clean_data)
        X_raw = clean_data[x_var].values.astype(float)
        Y     = clean_data[y_var].values.astype(float)
        M_raw = clean_data[m_var].values.astype(float)

        # FIX 3: prepare moderator
        M_col, mod_values, mod_labels, is_categorical, categories = prepare_moderator(
            M_raw, categorical=req.categoricalModerator
        )
        X = center_variable(X_raw, req.centerMethod)
        interaction = X * M_col

        # Build DataFrames for statsmodels
        step1_df = pd.DataFrame({x_var: X, m_var: M_col})
        step2_df = pd.DataFrame({x_var: X, m_var: M_col, f'{x_var}×{m_var}': interaction})

        # Run OLS via statsmodels (numerically stable QR decomposition)
        step1 = run_ols(step1_df, Y)
        step2 = run_ols(step2_df, Y)

        # R² change test
        delta_r2 = step2['r_squared'] - step1['r_squared']
        delta_df = step2['k'] - step1['k']
        df2      = step2['df']
        f_change = (delta_r2 / delta_df) / ((1 - step2['r_squared']) / df2) \
            if delta_df > 0 and df2 > 0 and (1 - step2['r_squared']) > 0 else 0.0
        p_change = 1 - stats.f.cdf(f_change, delta_df, df2) \
            if delta_df > 0 and df2 > 0 else 1.0

        results = {
            'step1': step1,
            'step2': step2,
            'r_squared_change': {
                'delta_r2': safe_float(delta_r2),
                'f_change': safe_float(f_change),
                'p_change': safe_float(p_change),
            }
        }

        # FIX 1: correct simple slopes using statsmodels cov_params
        results['simple_slopes'] = compute_simple_slopes(step2, mod_values, mod_labels)

        # Effect size: Cohen's f²
        f_sq = delta_r2 / (1 - step2['r_squared']) if (1 - step2['r_squared']) > 0 else 0.0
        results['effect_size'] = {
            'f_squared': safe_float(f_sq),
            'interpretation': ("Large effect" if f_sq >= 0.35 else
                               "Medium effect" if f_sq >= 0.15 else
                               "Small effect"  if f_sq >= 0.02 else
                               "Negligible effect"),
        }

        # FIX 2: VIF via statsmodels
        results['vif'] = compute_vif(step2_df)

        # Remove non-serialisable objects before response
        for step in [step1, step2]:
            step.pop('fit', None)
            step.pop('cov_params', None)

        # Interpretation
        overall_analysis, recommendations = generate_interpretation(
            results, x_var, y_var, m_var, N, is_categorical, categories
        )
        results['interpretation']   = overall_analysis
        results['overall_analysis'] = overall_analysis
        results['recommendations']  = recommendations

        # Split into paragraph sections for frontend rendering
        # Each section starts with a <strong>Title:</strong> tag
        import re as _re
        def _split_sections(html_text: str) -> list:
            """Split a blob of HTML into [{title, body}] by <strong>…:</strong> markers."""
            # Split on boundaries where a <strong> tag starts (not mid-sentence bolds)
            parts = _re.split(r'(?=<strong>[^<]{3,50}:</strong>)', html_text.strip())
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

        results['analysis_sections']        = _split_sections(overall_analysis)
        results['recommendation_sections']  = _split_sections(recommendations)

        plot = create_plot(results, X, M_col, M_raw, x_var, y_var, m_var,
                           is_categorical, categories, mod_values, mod_labels)

        return _to_native({
            'results': results,
            'n_observations': N,
            'is_categorical_moderator': is_categorical,
            'plot': plot,
        })

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
