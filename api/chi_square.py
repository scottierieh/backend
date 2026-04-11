from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Any
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact
import io
import base64

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")

router = APIRouter()


class ChiSquareRequest(BaseModel):
    data: list[dict[str, Any]] = Field(...)
    rowVar: str = Field(...)
    colVar: str = Field(...)


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


def cramers_v_interpretation(v, df=1):
    """
    Cohen (1988) guidelines adjusted by df = min(r-1, c-1).
    df=1: small=.10, medium=.30, large=.50
    df=2: small=.07, medium=.21, large=.35
    df=3+: small=.06, medium=.17, large=.29
    """
    thresholds = {1: (0.10, 0.30, 0.50), 2: (0.07, 0.21, 0.35)}
    small, medium, large = thresholds.get(min(df, 2), (0.06, 0.17, 0.29))
    if v < small:  return "negligible"
    if v < medium: return "small"
    if v < large:  return "medium"
    return "large"


def generate_interpretation(chi2_stat, p_val, dof, cramers_v, n_total, row_var, col_var, std_residuals, contingency_table, expected_arr=None, fisher_result=None, test_used="chi2", min_dim=1, adj_residuals=None, bonferroni_z=1.96):
    p_text = "p < .001" if p_val < 0.001 else ("p < .01" if p_val < 0.01 else ("p < .05" if p_val < 0.05 else f"p = {p_val:.3f}"))
    sig_text = "significant" if p_val < 0.05 else "not significant"
    effect_text = cramers_v_interpretation(cramers_v, min_dim)
    
    sections = []
    sections.append("**Overall Analysis**")
    test_label = "Fishers Exact Test" if test_used == "fisher" else "Chi-square Test of Independence"
    sections.append(f"{test_label} between {row_var} and {col_var} using {n_total} observations.")
    if test_used == "fisher" and fisher_result:
        sections.append(f"Result: {sig_text} (Fisher p = {fisher_result['p_value']:.4f}, OR = {fisher_result['odds_ratio']:.4f}).")
    else:
        sections.append(f"Result: {sig_text}, chi2({dof}) = {chi2_stat:.2f}, {p_text}.")
    sections.append(f"Cramer's V = {cramers_v:.3f} ({effect_text} effect size).")
    if expected_arr is not None:
        n_cells = expected_arr.size
        n_low = int((expected_arr < 5).sum())
        pct = 100 * n_low / n_cells
        if n_low > 0:
            warn = f"WARNING: {n_low}/{n_cells} cells ({pct:.0f}%) have expected count < 5."
            warn += " >20% violated - results unreliable." if pct > 20 else " Interpret with caution."
            sections.append(warn)
    sections.append("")
    
    sections.append("**Key Insights**")
    if p_val < 0.05:
        sections.append(f"→ {row_var} and {col_var} are statistically dependent")
        
        # Find significant cells using adjusted residuals + Bonferroni threshold
        residual_src = adj_residuals if adj_residuals is not None else std_residuals
        for r_idx, r_name in enumerate(residual_src.index):
            for c_idx, c_name in enumerate(residual_src.columns):
                z = float(residual_src.iat[r_idx, c_idx])
                if abs(z) > bonferroni_z:
                    obs = contingency_table.iat[r_idx, c_idx]
                    direction = "more frequent" if z > 0 else "less frequent"
                    sections.append(f"→ **{r_name} × {c_name}**: Observed ({obs}) {direction} than expected (adj. z = {z:.2f}, Bonferroni p < .05)")
    else:
        sections.append(f"→ {row_var} and {col_var} appear statistically independent")
        sections.append("→ No evidence of association at α = 0.05")
    
    sections.append("")
    sections.append("**Recommendations**")
    if p_val < 0.05:
        sections.append("→ Examine contingency table to understand patterns")
        sections.append("→ Review standardized residuals for specific drivers")
        if cramers_v >= 0.3:
            sections.append(f"→ {effect_text.title()} effect size indicates meaningful relationship")
    else:
        sections.append("→ Variables can be treated as independent")
        sections.append("→ Consider larger sample or different categorization if association expected")
    
    return "\n".join(sections)


@router.post("/chi-square")
def chi_square_test(req: ChiSquareRequest):
    try:
        df = pd.DataFrame(req.data)
        row_var = req.rowVar
        col_var = req.colVar
        
        original_len = len(df)
        df_clean = df[[row_var, col_var]].dropna()
        dropped_rows = list(set(range(original_len)) - set(df_clean.index.tolist()))
        n_dropped = len(dropped_rows)
        
        if len(df_clean) < 2:
            raise ValueError("Not enough valid data")
        
        contingency_table = pd.crosstab(df_clean[row_var], df_clean[col_var])

        total = contingency_table.sum().sum()

        is_2x2_early = (contingency_table.shape[0] == 2 and contingency_table.shape[1] == 2)
        chi2_stat, p_val, dof, expected = chi2_contingency(contingency_table, correction=is_2x2_early)
        chi2_no_yates, p_no_yates, _, _ = chi2_contingency(contingency_table, correction=False)
        yates_applied = bool(is_2x2_early)

        residuals = contingency_table - expected
        std_residuals = residuals / np.sqrt(np.maximum(expected, 1e-12))

        # Adjusted standardized residuals (Haberman, 1973) — matches SPSS output
        row_totals = contingency_table.sum(axis=1).values
        col_totals = contingency_table.sum(axis=0).values
        adj_residuals = pd.DataFrame(index=contingency_table.index, columns=contingency_table.columns, dtype=float)
        for i, r in enumerate(contingency_table.index):
            for j, c in enumerate(contingency_table.columns):
                e_ij   = expected[i, j]
                row_p  = row_totals[i] / total
                col_p  = col_totals[j] / total
                denom  = np.sqrt(e_ij * (1 - row_p) * (1 - col_p))
                adj_residuals.loc[r, c] = float(residuals.iat[i, j]) / denom if denom > 1e-12 else 0.0

        # Bonferroni-corrected threshold for cell significance
        n_cells_total_early = contingency_table.size
        bonferroni_alpha = 0.05 / n_cells_total_early
        from scipy.stats import norm as _norm
        bonferroni_z = float(_norm.ppf(1 - bonferroni_alpha / 2))

        phi2 = chi2_stat / total if total > 0 else 0
        phi = np.sqrt(phi2)
        contingency_coeff = np.sqrt(chi2_stat / (chi2_stat + total)) if (chi2_stat + total) > 0 else 0
        
        n_rows, n_cols = contingency_table.shape
        min_dim = min(n_rows - 1, n_cols - 1)
        # Bias-corrected Cramér's V (Bergsma & Wicher, 2013)
        cramers_v_raw = np.sqrt(phi2 / min_dim) if min_dim > 0 else 0
        if min_dim > 0 and total > 1:
            phi2_corr = max(0, phi2 - (min_dim / (total - 1)))
            k_corr    = n_rows - (n_rows - 1) ** 2 / (total - 1)
            r_corr    = n_cols - (n_cols - 1) ** 2 / (total - 1)
            denom_corr = min(k_corr - 1, r_corr - 1)
            cramers_v = float(np.sqrt(phi2_corr / denom_corr)) if denom_corr > 0 else 0.0
        else:
            cramers_v = cramers_v_raw
        
        # Likelihood Ratio Chi-square (G-test)
        with np.errstate(divide='ignore', invalid='ignore'):
            obs_arr = contingency_table.values.astype(float)
            ratio = np.where(expected > 0, obs_arr / expected, 0)
            g_stat = float(2 * np.sum(obs_arr * np.where(ratio > 0, np.log(ratio), 0)))
        g_p_val = float(1 - __import__('scipy').stats.chi2.cdf(g_stat, dof))

        # Categorical Driver Detection
        drivers = []
        for r_idx, r in enumerate(contingency_table.index):
            for c_idx, c in enumerate(contingency_table.columns):
                z    = float(adj_residuals.loc[r, c])
                obs  = int(contingency_table.loc[r, c])
                exp  = float(expected[r_idx, c_idx])
                row_t = int(row_totals[r_idx])
                col_t = int(col_totals[c_idx])
                if abs(z) > bonferroni_z:
                    drivers.append({
                        "row":          str(r),
                        "col":          str(c),
                        "observed":     obs,
                        "expected":     round(exp, 2),
                        "adj_residual": round(z, 4),
                        "direction":    "overrepresented" if z > 0 else "underrepresented",
                        "lift":         round(obs / exp, 4) if exp > 0 else None,
                        "row_pct":      round(100 * obs / row_t, 1) if row_t > 0 else None,
                        "col_pct":      round(100 * obs / col_t, 1) if col_t > 0 else None,
                    })
        drivers_sorted = sorted(drivers, key=lambda x: abs(x["adj_residual"]), reverse=True)

        # Goodman-Kruskal Tau (asymmetric: row->col and col->row)
        def _gk_tau(ct):
            """Goodman-Kruskal tau: proportional reduction in error predicting cols from rows."""
            n = ct.values.sum()
            col_totals = ct.sum(axis=0).values
            row_totals = ct.sum(axis=1).values
            # E1: error predicting col ignoring row = 1 - sum(col_j/n)^2
            e1 = 1.0 - float(np.sum((col_totals / n) ** 2))
            # E2: weighted error using row info
            e2 = 0.0
            for i in range(len(row_totals)):
                if row_totals[i] > 0:
                    e2 += (row_totals[i] / n) * (1.0 - float(np.sum((ct.values[i] / row_totals[i]) ** 2)))
            return float((e1 - e2) / e1) if e1 > 1e-12 else 0.0

        tau_row_to_col = _gk_tau(contingency_table)
        tau_col_to_row = _gk_tau(contingency_table.T)

        # Expected count assumption check
        expected_arr = np.array(expected)
        n_cells_total = expected_arr.size
        n_cells_low   = int((expected_arr < 5).sum())
        pct_low       = float(100 * n_cells_low / n_cells_total)
        expected_count_check = {
            "n_cells": int(n_cells_total),
            "n_cells_below_5": n_cells_low,
            "pct_cells_below_5": round(pct_low, 1),
            "assumption_met": bool(n_cells_low == 0),
            "severe_violation": bool(pct_low > 20),
            "min_expected": float(round(expected_arr.min(), 4)),
            "note": (
                "All expected counts >= 5. Assumption satisfied."
                if n_cells_low == 0 else
                f"{n_cells_low}/{n_cells_total} cells ({pct_low:.0f}%) have expected < 5. " +
                ("Prefer Fishers Exact or merge categories." if pct_low > 20 else "Interpret with caution.")
            )
        }

        # Fisher exact test fallback (2x2 only)
        fisher_result = None
        test_used = "chi2"
        is_2x2 = is_2x2_early  # already computed above
        if is_2x2:
            _odds, _fp = fisher_exact(contingency_table.values)
            fisher_result = {
                "odds_ratio": float(_odds),
                "p_value":    float(_fp),
                "significant": bool(_fp < 0.05),
                "note": "2x2 table detected. Fishers Exact is preferred when any expected count < 5."
            }
            if n_cells_low > 0:
                test_used = "fisher"

        interpretation = generate_interpretation(chi2_stat, p_val, dof, cramers_v, total, row_var, col_var,
                                                 std_residuals, contingency_table,
                                                 expected_arr=expected_arr,
                                                 fisher_result=fisher_result,
                                                 test_used=test_used,
                                                 min_dim=min_dim,
                                                 adj_residuals=adj_residuals,
                                                 bonferroni_z=bonferroni_z)
        
        def _to_b64(fig):
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
            plt.close(fig)
            buf.seek(0)
            return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"

        # Tab 1: Grouped bar chart
        _fig1, _ax1 = plt.subplots(figsize=(10, 6))
        sns.countplot(data=df_clean, x=row_var, hue=col_var, palette="crest", ax=_ax1)
        _ax1.set_title("Grouped Bar Chart", fontweight="bold")
        _ax1.set_xlabel(row_var); _ax1.set_ylabel("Count")
        _ax1.tick_params(axis="x", rotation=45)
        _ax1.legend(title=col_var, fontsize=9)
        plt.tight_layout()
        plot_bar = _to_b64(_fig1)

        # Tab 2: Standardised residuals heatmap
        _fig2, _ax2 = plt.subplots(figsize=(10, 6))
        _adj_arr = adj_residuals.values.astype(float)
        _vmax = max(float(np.abs(_adj_arr).max()), 2.0)
        # Annotate: value + significance marker
        _annot = pd.DataFrame(index=adj_residuals.index, columns=adj_residuals.columns)
        for _r in adj_residuals.index:
            for _c in adj_residuals.columns:
                _z = float(adj_residuals.loc[_r, _c])
                _star = " ***" if abs(_z) > bonferroni_z else (" *" if abs(_z) > 1.96 else "")
                _annot.loc[_r, _c] = f"{_z:.2f}{_star}"
        sns.heatmap(adj_residuals.astype(float), annot=_annot, fmt="", center=0,
                    vmin=-_vmax, vmax=_vmax, cmap="RdBu_r", linewidths=0.5, ax=_ax2,
                    cbar_kws={"label": "Adjusted Residual"})
        _ax2.set_title(
            f"Adjusted Residuals Heatmap  (*** Bonferroni p<.05  |z|>{bonferroni_z:.2f},  * uncorrected p<.05  |z|>1.96)",
            fontweight="bold", fontsize=10)
        _ax2.set_xlabel(col_var); _ax2.set_ylabel(row_var)
        _ax2.tick_params(axis="x", rotation=45)
        plt.tight_layout()
        plot_heatmap = _to_b64(_fig2)

        # Tab 3: Mosaic plot
        _fig3, _ax3 = plt.subplots(figsize=(10, 6))
        try:
            from statsmodels.graphics.mosaicplot import mosaic as _mosaic
            import matplotlib.colors as _mc
            _cmap3 = plt.cm.RdBu_r
            _vm3   = max(float(np.abs(adj_residuals.values.astype(float)).max()), 1e-6)
            ct_d   = {(str(r), str(c)): int(contingency_table.loc[r, c])
                      for r in contingency_table.index for c in contingency_table.columns}
            def _props(key):
                try:
                    z = float(adj_residuals.loc[key[0], key[1]])
                except Exception:
                    z = 0.0
                return {"facecolor": _mc.to_hex(_cmap3(0.5 + 0.5 * np.clip(z / _vm3, -1, 1))),
                        "alpha": 0.85}
            _mosaic(ct_d, ax=_ax3, properties=_props,
                    labelizer=lambda k: str(ct_d[k]), gap=0.01)
            _ax3.set_title("Mosaic Plot (area=proportion, colour=residual)", fontweight="bold")
            _ax3.set_xlabel(col_var); _ax3.set_ylabel(row_var)
        except Exception as _me:
            _ax3.text(0.5, 0.5, f"Mosaic unavailable: {_me}",
                      ha="center", va="center", transform=_ax3.transAxes, fontsize=9)
            _ax3.set_title("Mosaic Plot", fontweight="bold")
        plt.tight_layout()
        plot_mosaic = _to_b64(_fig3)

        # Tab 4: Correspondence Analysis biplot
        _fig4, _ax4 = plt.subplots(figsize=(10, 6))
        try:
            _P  = contingency_table.values.astype(float); _P = _P / _P.sum()
            _rm = _P.sum(1); _cm = _P.sum(0)
            _S  = np.diag(1/np.sqrt(_rm+1e-15)) @ (_P - np.outer(_rm,_cm)) @ np.diag(1/np.sqrt(_cm+1e-15))
            _U, _D, _Vt = np.linalg.svd(_S, full_matrices=False)
            _iner = float((_D**2).sum())
            _p1   = float(_D[0]**2/_iner*100) if _iner>0 else 0
            _p2   = float(_D[1]**2/_iner*100) if len(_D)>1 and _iner>0 else 0
            _F = np.diag(1/np.sqrt(_rm+1e-15)) @ _U[:,:2] * _D[:2]
            _G = np.diag(1/np.sqrt(_cm+1e-15)) @ _Vt[:2,:].T * _D[:2]
            for i, lbl in enumerate([str(x) for x in contingency_table.index]):
                _ax4.scatter(_F[i,0], _F[i,1], color="#3B82F6", s=90, zorder=3)
                _ax4.annotate(lbl, (_F[i,0],_F[i,1]), xytext=(5,5),
                              textcoords="offset points", color="#1D4ED8", fontsize=9, fontweight="bold")
            for j, lbl in enumerate([str(x) for x in contingency_table.columns]):
                _ax4.scatter(_G[j,0], _G[j,1], color="#EF4444", s=90, marker="^", zorder=3)
                _ax4.annotate(lbl, (_G[j,0],_G[j,1]), xytext=(5,5),
                              textcoords="offset points", color="#B91C1C", fontsize=9, fontweight="bold")
            _ax4.axhline(0, color="gray", lw=0.5, ls="--")
            _ax4.axvline(0, color="gray", lw=0.5, ls="--")
            _ax4.set_xlabel(f"Dim 1 ({_p1:.1f}%)")
            _ax4.set_ylabel(f"Dim 2 ({_p2:.1f}%)")
            _ax4.set_title(f"Correspondence Analysis Biplot (inertia={_iner:.4f})", fontweight="bold")
            _ax4.legend(handles=[
                plt.Line2D([0],[0],marker="o",color="w",markerfacecolor="#3B82F6",markersize=8,label=row_var),
                plt.Line2D([0],[0],marker="^",color="w",markerfacecolor="#EF4444",markersize=8,label=col_var),
            ], fontsize=9)
        except Exception as _ca_e:
            _ax4.text(0.5, 0.5, f"CA biplot unavailable: {_ca_e}",
                      ha="center", va="center", transform=_ax4.transAxes, fontsize=9)
            _ax4.set_title("Correspondence Analysis", fontweight="bold")
        plt.tight_layout()
        plot_ca = _to_b64(_fig4)

        # legacy single plot = bar (하위 호환)
        plot = plot_bar
        
        # Percentage tables
        row_percent   = contingency_table.div(contingency_table.sum(axis=1), axis=0).mul(100).round(1)
        col_percent   = contingency_table.div(contingency_table.sum(axis=0), axis=1).mul(100).round(1)
        total_percent = contingency_table.div(total).mul(100).round(1)

        return _to_native({
            "results": {
                "test_used":             test_used,
                "contingency_table":     contingency_table.to_dict(),
                "expected_table":        pd.DataFrame(expected_arr,
                                             index=contingency_table.index,
                                             columns=contingency_table.columns).to_dict(),
                "std_residuals":         std_residuals.to_dict(),
                "adj_residuals":         adj_residuals.to_dict(),
                "bonferroni_z":          round(bonferroni_z, 4),
                "bonferroni_alpha_cell": round(bonferroni_alpha, 6),
                "drivers":               drivers_sorted[:10],
                "chi_squared": {
                    "statistic":          float(chi2_stat),
                    "p_value":            float(p_val),
                    "degrees_of_freedom": int(dof),
                    "yates_correction_applied": yates_applied,
                    "chi2_no_yates":      float(chi2_no_yates) if yates_applied else None,
                    "p_no_yates":         float(p_no_yates) if yates_applied else None,
                },
                "fisher_exact":          fisher_result,
                "expected_count_check":  expected_count_check,
                "likelihood_ratio": {
                    "statistic":    round(g_stat, 4),
                    "p_value":      round(g_p_val, 4),
                    "degrees_of_freedom": int(dof),
                    "significant":  bool(g_p_val < 0.05),
                },
                "goodman_kruskal_tau": {
                    "tau_row_to_col": round(tau_row_to_col, 4),
                    "tau_col_to_row": round(tau_col_to_row, 4),
                    "interpretation_row_to_col": cramers_v_interpretation(tau_row_to_col, min_dim),
                    "interpretation_col_to_row": cramers_v_interpretation(tau_col_to_row, min_dim),
                },
                "phi_coefficient":       float(phi),
                "contingency_coefficient": float(contingency_coeff),
                "cramers_v":             float(cramers_v),
                "cramers_v_raw":          float(cramers_v_raw),
                "cramers_v_interpretation": cramers_v_interpretation(cramers_v, min_dim),
                "row_percent":           row_percent.to_dict(),
                "col_percent":           col_percent.to_dict(),
                "total_percent":         total_percent.to_dict(),
                "interpretation":        interpretation,
                "row_var":               row_var,
                "col_var":               col_var,
                "row_levels":            contingency_table.index.tolist(),
                "col_levels":            contingency_table.columns.tolist(),
                "n_dropped":             n_dropped,
                "dropped_rows":          dropped_rows
            },
            "plot": plot,
                "plots": {
                    "bar":      plot_bar,
                    "heatmap":  plot_heatmap,
                    "mosaic":   plot_mosaic,
                    "ca":       plot_ca
                }
        })
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
