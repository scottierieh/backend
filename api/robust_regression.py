from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any, List, Optional
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.robust.norms import HuberT, TukeyBiweight, RamsayE, AndrewWave, Hampel, LeastSquares
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_squared_error
from scipy import stats as _scipy_stats
import io, base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

router = APIRouter()

class RobustRequest(BaseModel):
    data: list[dict[str, Any]] = Field(...)
    y_col: str = Field(...)
    x_cols: List[str] = Field(...)              # 다변량 지원
    M: str = Field(default="HuberT")
    cv_folds: int = Field(default=5, ge=3, le=10)
    x_col: Optional[str] = Field(default=None)  # 하위호환

def _to_native(obj):
    if isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, (np.floating, float)):
        if np.isnan(obj) or np.isinf(obj): return None
        return float(obj)
    elif isinstance(obj, np.ndarray): return [_to_native(x) for x in obj.tolist()]
    elif isinstance(obj, dict): return {str(k): _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)): return [_to_native(x) for x in obj]
    return obj

def get_norm(name):
    return {
        'HuberT':       HuberT(),
        'TukeyBiweight':TukeyBiweight(),
        'RamsayE':      RamsayE(),
        'AndrewWave':   AndrewWave(),
        'Hampel':       Hampel(),
        'LeastSquares': LeastSquares(),
    }.get(name, HuberT())

def _fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def _coef_table(model, is_rlm=False):
    out = {}
    stat_key = 'z' if is_rlm else 't'
    for name in model.params.index:
        out[name] = {
            'coef':      round(float(model.params[name]),  6),
            'se':        round(float(model.bse[name]),     6),
            stat_key:    round(float(model.tvalues[name]), 4),
            'p_value':   round(float(model.pvalues[name]), 4),
            'significant': bool(model.pvalues[name] < 0.05),
        }
    return out


@router.post("/robust-regression")
def robust_regression(req: RobustRequest):
    try:
        # ── 하위호환: x_col → x_cols ─────────────────────────────────────
        x_cols = req.x_cols or ([req.x_col] if req.x_col else [])
        if not x_cols:
            raise ValueError("x_cols must have at least one variable")

        df = pd.DataFrame(req.data)[x_cols + [req.y_col]].dropna()
        n_dropped = len(pd.DataFrame(req.data)) - len(df)

        if len(df) < max(10, len(x_cols) + 2):
            raise ValueError(f"Not enough data (n={len(df)})")

        X_raw   = df[x_cols].astype(float)
        y       = df[req.y_col].astype(float)
        X_const = sm.add_constant(X_raw)
        n, p    = len(y), len(x_cols)

        # ── OLS ──────────────────────────────────────────────────────────
        ols = sm.OLS(y, X_const).fit()

        # ── RLM ──────────────────────────────────────────────────────────
        rlm      = sm.RLM(y, X_const, M=get_norm(req.M)).fit()
        rlm_pred = rlm.predict(X_const)
        ss_res   = float(np.sum((y - rlm_pred) ** 2))
        ss_tot   = float(np.sum((y - float(y.mean())) ** 2))
        rlm_r2   = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        # ── Coefficient comparison ────────────────────────────────────────
        coef_comparison = {}
        for feat in x_cols:
            ob = float(ols.params.get(feat, np.nan))
            rb = float(rlm.params.get(feat, np.nan))
            d  = abs(ob - rb)
            r  = d / abs(ob) * 100 if abs(ob) > 1e-6 else 0.0
            coef_comparison[feat] = {
                'ols_coef':      round(ob, 6),
                'rlm_coef':      round(rb, 6),
                'abs_diff':      round(d,  6),
                'pct_diff':      round(r,  2),
                'delta':         round(d,  6),   # alias for frontend
                'delta_pct':     round(r,  2),   # alias for frontend
                'outlier_impact':'high' if r > 20 else 'moderate' if r > 10 else 'low',
            }
        max_pct_diff = max((v['pct_diff'] for v in coef_comparison.values()), default=0.0)

        # ── RLM weights ───────────────────────────────────────────────────
        weights    = rlm.weights
        low_weight = [int(i) for i in np.where(weights < 0.5)[0]]
        weight_stats = {
            'mean':               round(float(weights.mean()), 4),
            'min':                round(float(weights.min()),  4),
            'max':                round(float(weights.max()),  4),
            'n_downweighted':     len(low_weight),
            'downweight_pct':     round(len(low_weight) / n * 100, 2),
            'downweighted_indices': low_weight[:20],
        }

        # ── Residual diagnostics ──────────────────────────────────────────
        ols_resid   = ols.resid.values
        fitted_vals = ols.fittedvalues.values
        sw_stat, sw_p = (_scipy_stats.shapiro(ols_resid)
                         if len(ols_resid) <= 5000 else (None, None))
        bp_corr, bp_p = _scipy_stats.pearsonr(fitted_vals, np.abs(ols_resid))
        residual_diagnostics = {
            'mean':     round(float(ols_resid.mean()), 4),
            'std':      round(float(ols_resid.std()),  4),
            'skewness': round(float(_scipy_stats.skew(ols_resid)), 4),
            'kurtosis': round(float(_scipy_stats.kurtosis(ols_resid)), 4),
            'shapiro_wilk': {
                'statistic': round(float(sw_stat), 4) if sw_stat is not None else None,
                'p_value':   round(float(sw_p),   4) if sw_p   is not None else None,
                'normal':    bool(sw_p > 0.05)        if sw_p   is not None else None,
            },
            'heteroscedasticity': {
                'corr_fitted_abs_resid': round(float(bp_corr), 4),
                'corr':                  round(float(bp_corr), 4),  # alias for frontend
                'p_value':              round(float(bp_p),    4),
                'detected':             bool(bp_p < 0.05),
            },
        }

        # ── Influence diagnostics (OLS) ───────────────────────────────────
        try:
            infl      = ols.get_influence()
            hat       = infl.hat_matrix_diag
            resid_std = infl.resid_studentized_internal
            cooks_d   = infl.cooks_distance[0]
            cutoff_c  = 4.0 / n
            cutoff_h  = 2 * (p + 1) / n
            flagged_c = [int(i) for i in np.where(cooks_d > cutoff_c)[0]]
            flagged_h = [int(i) for i in np.where(hat > cutoff_h)[0]]
            influence_diagnostics = {
                'cooks_distance': {
                    'max':               round(float(cooks_d.max()), 4),
                    'cutoff':            round(cutoff_c, 4),
                    'n_flagged':         len(flagged_c),
                    'flagged_indices':   flagged_c[:20],
                },
                'leverage': {
                    'mean':              round(float(hat.mean()), 4),
                    'max':               round(float(hat.max()),  4),
                    'cutoff':            round(cutoff_h, 4),
                    'n_flagged':         len(flagged_h),
                    'flagged_indices':   flagged_h[:20],
                },
                'studentized_residuals': {
                    'max_abs':           round(float(np.abs(resid_std).max()), 4),
                    'n_outliers_3sd':    int(np.sum(np.abs(resid_std) > 3)),
                },
            }
        except Exception:
            influence_diagnostics = None
            cooks_d = hat = resid_std = None
            cutoff_c = cutoff_h = None

        # ── Cross-validation (RLM, manual KFold) ─────────────────────────
        kf = KFold(n_splits=req.cv_folds, shuffle=True, random_state=42)
        X_arr, y_arr = X_const.values, y.values
        cv_r2s, cv_rmses = [], []
        for tr, te in kf.split(X_arr):
            try:
                _m    = sm.RLM(y_arr[tr], X_arr[tr], M=get_norm(req.M)).fit()
                _pred = X_arr[te] @ _m.params
                cv_r2s.append(float(r2_score(y_arr[te], _pred)))
                cv_rmses.append(float(np.sqrt(mean_squared_error(y_arr[te], _pred))))
            except Exception:
                pass
        cv_results = {
            'r2_mean':   round(float(np.mean(cv_r2s)),   4) if cv_r2s   else None,
            'r2_std':    round(float(np.std(cv_r2s)),    4) if cv_r2s   else None,
            'rmse_mean': round(float(np.mean(cv_rmses)), 4) if cv_rmses else None,
            'rmse_std':  round(float(np.std(cv_rmses)),  4) if cv_rmses else None,
            'n_folds':   req.cv_folds,
            'scores':    [round(s, 4) for s in cv_r2s],
        }

        # ── Interpretation ────────────────────────────────────────────────
        m_desc = {
            'HuberT':       'balanced efficiency and robustness',
            'TukeyBiweight':'strong outlier rejection (zero weight beyond threshold)',
            'RamsayE':      'smooth exponential downweighting',
            'AndrewWave':   'strong outlier resistance (sine-wave kernel)',
            'Hampel':       'three-part flexible outlier handling',
            'LeastSquares': 'no outlier protection — equivalent to OLS',
        }
        interp  = "**Robust Regression Analysis**\n"
        interp += f"→ Predictors: {', '.join(x_cols)}  |  n = {n}\n"
        interp += f"→ M-estimator: {req.M} — {m_desc.get(req.M, req.M)}\n\n"
        interp += "**Model Fit**\n"
        interp += f"→ OLS R² = {ols.rsquared:.4f} (adj = {ols.rsquared_adj:.4f})\n"
        interp += f"→ RLM Pseudo R² = {rlm_r2:.4f}\n"
        if cv_results['r2_mean'] is not None:
            interp += f"→ CV R² ({req.cv_folds}-fold) = {cv_results['r2_mean']:.3f} ± {cv_results['r2_std']:.3f}\n"
        interp += "\n**Coefficient Comparison (OLS vs RLM)**\n"
        for feat, v in coef_comparison.items():
            interp += f"→ {feat}: OLS={v['ols_coef']:.4f}, RLM={v['rlm_coef']:.4f}  Δ={v['pct_diff']:.1f}% ({v['outlier_impact']})\n"
        interp += f"\n**Downweighted Observations**: {weight_stats['n_downweighted']} (weight < 0.5)\n"
        if influence_diagnostics:
            interp += f"**Influential Points (Cook's D > {influence_diagnostics['cooks_distance']['cutoff']:.4f})**: {influence_diagnostics['cooks_distance']['n_flagged']}\n"
        interp += "\n**Recommendations**\n"
        if max_pct_diff > 20:
            interp += "→ Large OLS–RLM difference: outliers substantially distort OLS\n"
            interp += "→ Prefer RLM estimates; report both\n"
        elif max_pct_diff > 10:
            interp += "→ Moderate outlier influence — RLM preferred\n"
        else:
            interp += "→ OLS and RLM agree closely — OLS is appropriate\n"
        if residual_diagnostics['shapiro_wilk']['normal'] is False:
            interp += "→ Non-normal OLS residuals — robust methods preferable\n"
        if residual_diagnostics['heteroscedasticity']['detected']:
            interp += "→ Heteroscedasticity detected — consider WLS or HC standard errors\n"

        # ── Plots ─────────────────────────────────────────────────────────
        # Plot 1: scatter (univariate) or fitted vs actual (multivariate)
        if len(x_cols) == 1:
            fig1, ax1 = plt.subplots(figsize=(10, 7))
            x_s = X_raw.iloc[:, 0]
            sns.scatterplot(x=x_s, y=y, alpha=0.6, color='#5B9BD5', ax=ax1, label='Data')
            xr = np.linspace(float(x_s.min()), float(x_s.max()), 200)
            ax1.plot(xr, ols.params.iloc[0] + ols.params.iloc[1] * xr,
                     '--', color='#C44E52', lw=2, label=f'OLS (R²={ols.rsquared:.3f})')
            ax1.plot(xr, rlm.params.iloc[0] + rlm.params.iloc[1] * xr,
                     '-',  color='#4C72B0', lw=2, label=f'RLM-{req.M} (R²={rlm_r2:.3f})')
            if low_weight:
                ax1.scatter(x_s.iloc[low_weight], y.iloc[low_weight],
                            color='red', s=60, zorder=5, alpha=0.8, label='Downweighted')
            ax1.set_xlabel(x_cols[0]); ax1.set_ylabel(req.y_col)
            ax1.set_title(f'OLS vs Robust Regression ({req.M})', fontweight='bold')
            ax1.legend()
        else:
            fig1, axes1 = plt.subplots(1, 2, figsize=(12, 5))
            for ax, pred, label, color in [
                (axes1[0], ols.fittedvalues.values, f'OLS  R²={ols.rsquared:.3f}', '#C44E52'),
                (axes1[1], rlm_pred,                f'RLM-{req.M}  R²={rlm_r2:.3f}', '#4C72B0'),
            ]:
                ax.scatter(y, pred, alpha=0.55, color=color, s=28)
                lims = [min(float(y.min()), pred.min()), max(float(y.max()), pred.max())]
                ax.plot(lims, lims, 'k--', lw=1.2)
                ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
                ax.set_title(label, fontweight='bold')
        plot_main = _fig_to_b64(fig1)

        # Plot 2: Residuals vs Fitted + Q-Q
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
        axes2[0].scatter(fitted_vals, ols_resid, alpha=0.55, color='#5B9BD5', s=28, edgecolors='none')
        axes2[0].axhline(0, color='red', linestyle='--', lw=1.5)
        _xs = np.linspace(fitted_vals.min(), fitted_vals.max(), 200)
        axes2[0].plot(_xs, np.poly1d(np.polyfit(fitted_vals, ols_resid, 2))(_xs),
                      color='orange', lw=1.5, label='Trend')
        axes2[0].set_xlabel('Fitted Values'); axes2[0].set_ylabel('OLS Residuals')
        axes2[0].set_title('Residuals vs Fitted (OLS)', fontweight='bold')
        axes2[0].legend(fontsize=8)
        _scipy_stats.probplot(ols_resid, dist='norm', plot=axes2[1])
        axes2[1].set_title('Normal Q-Q (OLS Residuals)', fontweight='bold')
        axes2[1].get_lines()[0].set(markersize=4, alpha=0.6, color='#5B9BD5')
        axes2[1].get_lines()[1].set(color='red', lw=1.5)
        plt.tight_layout()
        plot_resid = _fig_to_b64(fig2)

        # Plot 3: Cook's D + Leverage vs Studentized Residual
        if influence_diagnostics and cooks_d is not None:
            fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
            axes3[0].stem(range(n), cooks_d, linefmt='C0-', markerfmt='C0o',
                          basefmt='gray')
            axes3[0].axhline(cutoff_c, color='red', linestyle='--', lw=1.5,
                             label=f'4/n = {cutoff_c:.4f}')
            axes3[0].set_xlabel('Index'); axes3[0].set_ylabel("Cook's D")
            axes3[0].set_title("Cook's Distance", fontweight='bold')
            axes3[0].legend(fontsize=8)
            axes3[1].scatter(hat, resid_std, alpha=0.55, color='#5B9BD5', s=28)
            axes3[1].axhline( 3, color='red',    linestyle='--', lw=1.2, label='|z|=3')
            axes3[1].axhline(-3, color='red',    linestyle='--', lw=1.2)
            axes3[1].axvline(cutoff_h, color='orange', linestyle='--', lw=1.2,
                             label=f'h={cutoff_h:.3f}')
            axes3[1].set_xlabel('Leverage'); axes3[1].set_ylabel('Studentized Residual')
            axes3[1].set_title('Leverage vs Studentized Residual', fontweight='bold')
            axes3[1].legend(fontsize=8)
            plt.tight_layout()
            plot_influence = _fig_to_b64(fig3)
        else:
            plot_influence = None

        # Plot 4: RLM weight distribution
        fig4, ax4 = plt.subplots(figsize=(7, 5))
        ax4.hist(weights, bins=30, color='#5B9BD5', edgecolor='white', alpha=0.85)
        ax4.axvline(0.5, color='red', linestyle='--', lw=1.5, label='weight = 0.5')
        ax4.set_xlabel('Observation Weight'); ax4.set_ylabel('Count')
        ax4.set_title(f'RLM Weight Distribution ({req.M})', fontweight='bold')
        ax4.legend()
        plt.tight_layout()
        plot_weights = _fig_to_b64(fig4)

        # ── Response ──────────────────────────────────────────────────────
        return _to_native({
            'results': {
                'ols': {
                    'coefficients':  _coef_table(ols,  is_rlm=False),
                    'r_squared':     round(float(ols.rsquared),     4),
                    'adj_r_squared': round(float(ols.rsquared_adj), 4),
                    'aic':           round(float(ols.aic),          4),
                    'bic':           round(float(ols.bic),          4),
                    'f_statistic':   round(float(ols.fvalue),       4),
                    'f_pvalue':      round(float(ols.f_pvalue),     4),
                },
                'rlm': {
                    'coefficients':     _coef_table(rlm, is_rlm=True),
                    'pseudo_r_squared': round(rlm_r2, 4),
                    'M_estimator':      req.M,
                    'weight_stats':     weight_stats,
                },
                'coef_comparison':       coef_comparison,
                'residual_diagnostics':  residual_diagnostics,
                'influence_diagnostics': influence_diagnostics,
                'cv_results':            cv_results,
                'interpretation':        interp,
                'n_obs':     n,
                'n_features': p,
                'n_dropped': n_dropped,
            },
            'plot':           plot_main,
            'residual_plot':  plot_resid,
            'influence_plot': plot_influence,
            'weights_plot':   plot_weights,
        })

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
