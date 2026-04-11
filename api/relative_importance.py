from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io, base64

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=130, bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64

router = APIRouter()

class RelativeImportanceRequest(BaseModel):
    data: list[dict[str, Any]] = Field(...)
    dependent_var: str = Field(...)
    independent_vars: List[str] = Field(...)
    n_bootstrap: int = Field(default=1000, ge=200, le=5000,
                             description="Bootstrap iterations for CI (200–5000)")
    ci_level: float = Field(default=0.95, ge=0.80, le=0.99,
                            description="Confidence level (0.80–0.99)")
    random_seed: int = Field(default=42)

def _to_native(obj):
    if isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    elif isinstance(obj, np.ndarray): return [_to_native(x) for x in obj.tolist()]
    elif isinstance(obj, dict): return {str(k): _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)): return [_to_native(x) for x in obj]
    return obj


def _rwa_core(X_s: np.ndarray, y_s: np.ndarray):
    """
    Relative Weight Analysis (Johnson 2000) on standardized inputs.
    Returns (weights_raw, r2) where weights_raw sums to r2.

    Fixes applied:
    1. Fit on standardized y  → correct RWA formulation
    2. np.clip(evals, 1e-12)  → eigenvalue stability for near-collinear data
    """
    n, p = X_s.shape
    R_xx = np.corrcoef(X_s, rowvar=False)

    # ❸ Eigenvalue stabilisation
    evals, evecs = np.linalg.eigh(R_xx)
    evals = np.clip(evals, 1e-12, None)          # stable inversion

    # Orthogonal basis Z
    Z = X_s @ evecs @ np.diag(1.0 / np.sqrt(evals))

    # ❷ Fit on standardized y
    model = LinearRegression(fit_intercept=True).fit(Z, y_s)
    beta_z = model.coef_

    # R² on standardised scale
    y_pred = model.predict(Z)
    ss_res = np.sum((y_s - y_pred) ** 2)
    ss_tot = np.sum((y_s - y_s.mean()) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # Raw relative weights: diagonal of  Λ^{-1/2} V' diag(β²) V Λ^{1/2}  reweighted
    sqrt_inv = np.diag(1.0 / np.sqrt(evals))
    sqrt_    = np.diag(np.sqrt(evals))
    raw = (evecs @ sqrt_inv @ np.diag(beta_z ** 2) @ sqrt_ @ evecs.T).diagonal()
    raw = np.maximum(raw, 0.0)

    # Rescale so weights sum to R²
    total = raw.sum()
    weights = raw * (r2 / total) if total > 0 else raw.copy()

    return weights, r2


def _multicollinearity(X_s: np.ndarray, var_names: list):
    """
    VIF + Condition Index for each predictor.
    Condition Index = sqrt(lambda_max / lambda_k); CI > 30 → serious collinearity.
    """
    from sklearn.linear_model import LinearRegression as _LR
    n, p = X_s.shape

    # VIF
    vifs = []
    for i in range(p):
        others = [j for j in range(p) if j != i]
        if not others:
            vifs.append(1.0)
            continue
        r2_i = _LR().fit(X_s[:, others], X_s[:, i]).score(X_s[:, others], X_s[:, i])
        vifs.append(float(1.0 / (1.0 - r2_i)) if r2_i < 1.0 else float('inf'))

    # Condition indices from eigenvalues of X'X (scaled)
    evals, _ = np.linalg.eigh(np.corrcoef(X_s, rowvar=False))
    evals = np.clip(evals, 1e-12, None)
    lambda_max = float(evals.max())
    cond_indices = np.sqrt(lambda_max / evals)   # sorted ascending (eigh gives ascending)

    out = {}
    for i, name in enumerate(var_names):
        out[name] = {
            'vif':            round(vifs[i], 4),
            'vif_flag':       'high (>10)' if vifs[i] > 10 else 'moderate (5–10)' if vifs[i] > 5 else 'ok',
        }
    return {
        'by_predictor':   out,
        'condition_index_max': round(float(cond_indices.max()), 4),
        'condition_index_flag': (
            'serious (>30)' if cond_indices.max() > 30
            else 'moderate (15–30)' if cond_indices.max() > 15
            else 'ok (<15)'
        ),
        'eigenvalues': [round(float(e), 6) for e in sorted(evals)],
        'condition_indices': [round(float(c), 4) for c in sorted(cond_indices)],
        'note': 'VIF > 10 or CI > 30 indicates serious multicollinearity',
    }


def _dominance_analysis(X_s: np.ndarray, y: np.ndarray, var_names: list):
    """
    Complete Dominance Analysis (Budescu 1993).
    For p predictors, fits all 2^p - 1 subset models.
    Returns general, conditional, and complete dominance.
    Practical limit: p <= 10 (1023 models).
    """
    from itertools import combinations
    p = len(var_names)
    if p > 10:
        return {'error': f'Dominance analysis skipped: p={p} > 10 (too many subset models)'}

    n_obs = X_s.shape[0]
    ss_tot = float(np.sum((y - y.mean()) ** 2))

    def _r2_subset(cols):
        if not cols:
            return 0.0
        from sklearn.linear_model import LinearRegression as _LR
        m = _LR().fit(X_s[:, list(cols)], y)
        ss_res = float(np.sum((y - m.predict(X_s[:, list(cols)])) ** 2))
        return max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

    # ── Additional R² for each predictor entering each subset ────────
    # additional_r2[i][subset_size] = mean additional R² when i added to
    # subsets of that size
    add_r2 = {i: {} for i in range(p)}

    for size in range(p):                          # 0 … p-1
        subsets = list(combinations(range(p), size))
        for sub in subsets:
            r2_base = _r2_subset(sub)
            for i in range(p):
                if i in sub:
                    continue
                r2_with = _r2_subset(tuple(sorted(sub + (i,))))
                delta = r2_with - r2_base
                add_r2[i].setdefault(size, []).append(delta)

    # General dominance weight = mean additional R² across all subset sizes
    general = {}
    for i in range(p):
        vals = [v for lst in add_r2[i].values() for v in lst]
        general[var_names[i]] = round(float(np.mean(vals)), 6) if vals else 0.0

    total_gd = sum(general.values())
    general_pct = {k: round(v / total_gd * 100, 4) if total_gd > 0 else 0.0
                   for k, v in general.items()}

    # Complete dominance: i completely dominates j if additional R²(i) >
    # additional R²(j) in ALL subsets that don't contain either
    complete = {}
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            key = f"{var_names[i]}_over_{var_names[j]}"
            dominates = True
            for size in range(p - 1):
                subs = [s for s in combinations(range(p), size) if i not in s and j not in s]
                for sub in subs:
                    di = _r2_subset(tuple(sorted(sub + (i,)))) - _r2_subset(sub)
                    dj = _r2_subset(tuple(sorted(sub + (j,)))) - _r2_subset(sub)
                    if di <= dj:
                        dominates = False
                        break
                if not dominates:
                    break
            complete[key] = dominates

    # Conditional dominance matrix (mean additional R² at each subset size)
    conditional = {}
    for i in range(p):
        row = {}
        for size in range(p):
            vals = add_r2[i].get(size, [])
            row[f'k={size}'] = round(float(np.mean(vals)), 6) if vals else None
        conditional[var_names[i]] = row

    return {
        'general_dominance':     general,
        'general_dominance_pct': general_pct,
        'complete_dominance':    complete,
        'conditional_dominance': conditional,
        'note': 'General dominance weights sum to R²; complete dominance = True if predictor i adds more R² than j in every possible subset',
    }


def _semi_partial_r2(X_s: np.ndarray, y: np.ndarray, full_r2: float):
    """Semi-partial R² for each predictor (unique contribution)."""
    p = X_s.shape[1]
    sp = []
    for i in range(p):
        others = [j for j in range(p) if j != i]
        if others:
            red = LinearRegression().fit(X_s[:, others], y)
            ss_res_red = np.sum((y - red.predict(X_s[:, others])) ** 2)
            ss_tot = np.sum((y - y.mean()) ** 2)
            r2_red = float(1 - ss_res_red / ss_tot) if ss_tot > 0 else 0.0
            sp.append(max(0.0, full_r2 - r2_red))
        else:
            sp.append(max(0.0, full_r2))
    return np.array(sp)


@router.post("/relative-importance")

def _plot_bar_chart(results_list: list, dep_var: str) -> str:
    """Bar chart: predictor relative weights %"""
    predictors = [r['predictor'] for r in results_list]
    weights = [r['relative_weight_pct'] for r in results_list]
    ci_low  = [r['rw_ci_lower'] for r in results_list]
    ci_high = [r['rw_ci_upper'] for r in results_list]

    n = len(predictors)
    colors = plt.cm.Blues(np.linspace(0.85, 0.35, n))

    fig, ax = plt.subplots(figsize=(max(7, n * 0.9 + 2), 5), facecolor='#fafafa')
    ax.set_facecolor('#fafafa')

    bars = ax.bar(predictors, weights, color=colors, edgecolor='white', linewidth=1.5, zorder=3)
    # Error bars for CI
    yerr_low  = [w - l for w, l in zip(weights, ci_low)]
    yerr_high = [h - w for w, h in zip(weights, ci_high)]
    ax.errorbar(predictors, weights, yerr=[yerr_low, yerr_high],
                fmt='none', color='#374151', capsize=5, capthick=1.5, linewidth=1.5, zorder=4)

    # Value labels
    for bar, w in zip(bars, weights):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(weights) * 0.02,
                f'{w:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#1f2937')

    ax.set_xlabel('Predictor', fontsize=11, color='#374151')
    ax.set_ylabel('Relative Weight (%)', fontsize=11, color='#374151')
    ax.set_title(f'Relative Importance of Predictors\nOutcome: {dep_var}', fontsize=13, fontweight='bold', color='#111827', pad=14)
    ax.set_ylim(0, max(weights) * 1.22)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(colors='#374151', labelsize=10)
    fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_ci_chart(results_list: list, ci_level: float) -> str:
    """Bootstrap CI plot — horizontal forest-plot style"""
    predictors = [r['predictor'] for r in results_list]
    weights    = [r['relative_weight_pct'] for r in results_list]
    ci_low     = [r['rw_ci_lower'] for r in results_list]
    ci_high    = [r['rw_ci_upper'] for r in results_list]
    n = len(predictors)

    fig, ax = plt.subplots(figsize=(8, max(4, n * 0.7 + 1.5)), facecolor='#fafafa')
    ax.set_facecolor('#fafafa')

    colors = plt.cm.Blues(np.linspace(0.8, 0.4, n))
    y_pos = list(range(n - 1, -1, -1))  # top = highest rank

    for i, (pred, w, lo, hi, yp, col) in enumerate(zip(predictors, weights, ci_low, ci_high, y_pos, colors)):
        sig = lo > 0
        ax.plot([lo, hi], [yp, yp], color=col, linewidth=3.5, solid_capstyle='round', zorder=3)
        ax.scatter([w], [yp], color=col, s=80, zorder=5, edgecolors='white', linewidths=1.5)
        if not sig:
            ax.scatter([w], [yp], color='none', s=100, zorder=6, edgecolors='#ef4444', linewidths=1.5)
        ax.text(hi + max(ci_high) * 0.03, yp, f'{w:.1f}%', va='center', fontsize=9.5,
                fontweight='bold', color='#1f2937')

    ax.axvline(0, color='#9ca3af', linestyle='--', linewidth=1, zorder=2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(predictors, fontsize=10, color='#374151')
    ax.set_xlabel('Relative Weight (%)', fontsize=11, color='#374151')
    ax.set_title(f'Bootstrap {int(ci_level*100)}% Confidence Intervals\nRelative Importance Estimates', 
                 fontsize=13, fontweight='bold', color='#111827', pad=14)
    ax.xaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(colors='#374151')

    sig_patch   = mpatches.Patch(color='#2563eb', label='Sig. (CI excludes 0)')
    insig_patch = mpatches.Patch(color='#93c5fd', label='Non-sig. (CI includes 0)')
    ax.legend(handles=[sig_patch, insig_patch], fontsize=9, loc='lower right')

    fig.tight_layout()
    return _fig_to_b64(fig)


def _plot_dominance(dom: dict, dep_var: str) -> str:
    """Dominance analysis — grouped bar chart of general dominance weights"""
    if not dom or 'general_dominance_pct' not in dom:
        return None

    gd = dom['general_dominance_pct']
    predictors = list(gd.keys())
    pcts = [gd[p] for p in predictors]
    n = len(predictors)

    # Sort descending
    order = np.argsort(pcts)[::-1]
    predictors = [predictors[i] for i in order]
    pcts = [pcts[i] for i in order]

    colors = plt.cm.Greens(np.linspace(0.8, 0.35, n))

    fig, ax = plt.subplots(figsize=(max(7, n * 0.9 + 2), 5), facecolor='#fafafa')
    ax.set_facecolor('#fafafa')

    bars = ax.bar(predictors, pcts, color=colors, edgecolor='white', linewidth=1.5, zorder=3)

    for bar, p in zip(bars, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(pcts) * 0.02,
                f'{p:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold', color='#1f2937')

    ax.set_xlabel('Predictor', fontsize=11, color='#374151')
    ax.set_ylabel('General Dominance Weight (%)', fontsize=11, color='#374151')
    ax.set_title(f'Dominance Analysis — General Dominance Weights\nOutcome: {dep_var}', 
                 fontsize=13, fontweight='bold', color='#111827', pad=14)
    ax.set_ylim(0, max(pcts) * 1.22)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(colors='#374151', labelsize=10)

    ax.annotate('General dominance weights represent average contribution\nacross all possible predictor subsets (Budescu, 1993)',
                xy=(0.5, -0.16), xycoords='axes fraction', ha='center', fontsize=8.5, color='#6b7280')

    fig.tight_layout()
    return _fig_to_b64(fig)


def relative_importance(req: RelativeImportanceRequest):
    try:
        if len(req.independent_vars) < 2:
            raise ValueError("Need at least 2 predictors")

        df = pd.DataFrame(req.data)
        all_vars = [req.dependent_var] + req.independent_vars
        df_c = df[all_vars].apply(pd.to_numeric, errors='coerce').dropna()
        n = len(df_c)

        if n < len(req.independent_vars) + 2:
            raise ValueError(f"Too few observations (n={n}) for {len(req.independent_vars)} predictors")

        X_raw = df_c[req.independent_vars].values
        y_raw = df_c[req.dependent_var].values

        # Standardise both X and y
        scaler_x = StandardScaler()
        X_s = scaler_x.fit_transform(X_raw)

        y_mean, y_std = float(y_raw.mean()), float(y_raw.std())
        y_s = (y_raw - y_mean) / y_std if y_std > 1e-12 else y_raw - y_mean

        # ── Point estimates ───────────────────────────────────────────────
        weights, r2 = _rwa_core(X_s, y_s)
        pct = (weights / r2 * 100.0) if r2 > 1e-12 else weights * 100.0

        # Standardised betas
        betas = LinearRegression(fit_intercept=False).fit(X_s, y_s).coef_

        # Semi-partial R²
        sp_r2 = _semi_partial_r2(X_s, y_raw, r2)

        # ── Multicollinearity diagnostics ────────────────────────────────
        mc_diag = _multicollinearity(X_s, req.independent_vars)

        # ── Dominance Analysis ────────────────────────────────────────────
        dom = _dominance_analysis(X_s, y_raw, req.independent_vars)

        # ── ❶ Bootstrap CI ────────────────────────────────────────────────
        rng = np.random.default_rng(req.random_seed)
        boot_pct  = np.zeros((req.n_bootstrap, len(req.independent_vars)))
        boot_beta = np.zeros((req.n_bootstrap, len(req.independent_vars)))
        boot_r2   = np.zeros(req.n_bootstrap)

        for b in range(req.n_bootstrap):
            idx   = rng.integers(0, n, size=n)
            Xb, yb = X_s[idx], y_s[idx]
            yb_std = yb.std()
            if yb_std < 1e-12:
                boot_pct[b]  = pct
                boot_beta[b] = betas
                boot_r2[b]   = r2
                continue
            try:
                w_b, r2_b = _rwa_core(Xb, yb)
                boot_pct[b]  = (w_b / r2_b * 100.0) if r2_b > 1e-12 else w_b * 100.0
                boot_beta[b] = LinearRegression(fit_intercept=False).fit(Xb, yb).coef_
                boot_r2[b]   = r2_b
            except Exception:
                boot_pct[b]  = pct
                boot_beta[b] = betas
                boot_r2[b]   = r2

        alpha = 1.0 - req.ci_level
        lo, hi = alpha / 2 * 100, (1 - alpha / 2) * 100

        ci_pct  = np.percentile(boot_pct,  [lo, hi], axis=0)   # (2, p)
        ci_beta = np.percentile(boot_beta, [lo, hi], axis=0)
        ci_r2   = np.percentile(boot_r2,   [lo, hi])

        # ── Results table ─────────────────────────────────────────────────
        results = []
        for i, var in enumerate(req.independent_vars):
            results.append({
                'predictor':             var,
                'rank':                  0,          # filled after sort
                'standardized_beta':     round(float(betas[i]), 6),
                'beta_ci_lower':         round(float(ci_beta[0, i]), 6),
                'beta_ci_upper':         round(float(ci_beta[1, i]), 6),
                'semi_partial_r2':       round(float(sp_r2[i]), 6),
                'relative_weight':       round(float(weights[i]), 6),
                'relative_weight_pct':   round(float(pct[i]), 4),
                'rw_ci_lower':           round(float(ci_pct[0, i]), 4),
                'rw_ci_upper':           round(float(ci_pct[1, i]), 4),
                'rw_se':                 round(float(boot_pct[:, i].std()), 4),
            })

        results.sort(key=lambda x: x['relative_weight_pct'], reverse=True)
        for i, r in enumerate(results):
            r['rank'] = i + 1

        # Bootstrap R² CI
        r2_ci = {
            'r2':       round(r2, 4),
            'ci_lower': round(float(ci_r2[0]), 4),
            'ci_upper': round(float(ci_r2[1]), 4),
            'ci_level': req.ci_level,
        }

        # ── Interpretation ────────────────────────────────────────────────
        top = results[0]
        interp  = f"**Relative Weight Analysis**\n"
        interp += f"→ R² = {r2:.3f} (95% CI: {r2_ci['ci_lower']:.3f}–{r2_ci['ci_upper']:.3f})\n"
        interp += f"→ Bootstrap n={req.n_bootstrap}, CI level={req.ci_level*100:.0f}%\n\n"
        interp += "**Predictor Rankings**\n"
        for r in results:
            sig = "" if r['rw_ci_lower'] > 0 else " [CI includes 0]"
            interp += (f"→ #{r['rank']} {r['predictor']}: "
                       f"{r['relative_weight_pct']:.1f}% "
                       f"(CI: {r['rw_ci_lower']:.1f}–{r['rw_ci_upper']:.1f}){sig}\n")
        interp += f"\n**Top Predictor**: {top['predictor']} explains {top['relative_weight_pct']:.1f}% of R²\n"

        # MC warnings in interpretation
        hi_vif = [k for k, v in mc_diag['by_predictor'].items() if v['vif'] > 5]
        if hi_vif:
            interp += f"\n**Multicollinearity Warning**\n"
            interp += f"→ High VIF (>5): {', '.join(hi_vif)}\n"
            interp += f"→ Condition Index max = {mc_diag['condition_index_max']:.1f} ({mc_diag['condition_index_flag']})\n"
            interp += "→ Relative weights may be unstable; interpret with caution\n"
        if 'general_dominance_pct' in dom:
            top_dom = max(dom['general_dominance_pct'], key=dom['general_dominance_pct'].get)
            interp += f"\n**Dominance Analysis**: {top_dom} generally dominates ({dom['general_dominance_pct'][top_dom]:.1f}% of R²)\n"

        # ── Generate plots ─────────────────────────────────────────────
        bar_plot       = _plot_bar_chart(results, req.dependent_var)
        ci_plot        = _plot_ci_chart(results, req.ci_level)
        dominance_plot = _plot_dominance(dom, req.dependent_var)

        return _to_native({
            'results':   results,
            'r_squared': r2_ci,
            'n_obs':     n,
            'n_bootstrap': req.n_bootstrap,
            'ci_level':  req.ci_level,
            'multicollinearity':   mc_diag,
            'dominance_analysis':  dom,
            'interpretation': interp,
            'bar_plot':        bar_plot,
            'ci_plot':         ci_plot,
            'dominance_plot':  dominance_plot,
        })

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
