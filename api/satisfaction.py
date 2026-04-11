from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, pearsonr, f_oneway, ttest_ind
import statsmodels.api as sm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import io
import base64
import warnings

warnings.filterwarnings('ignore')
router = APIRouter()
sns.set_theme(style="darkgrid")

class SatisfactionRequest(BaseModel):
    data: List[Dict[str, Any]]
    satisfaction_var: str
    satisfaction_items: Optional[List[str]] = []
    group_vars: Optional[List[str]] = []
    scale_points: Optional[int] = 5

def _to_native(obj):
    if isinstance(obj, (np.bool_, bool)): return bool(obj)
    elif isinstance(obj, np.integer): return int(obj)
    elif isinstance(obj, np.floating):
        return None if np.isnan(obj) or np.isinf(obj) else float(obj)
    elif isinstance(obj, np.ndarray): return [_to_native(x) for x in obj.tolist()]
    elif isinstance(obj, dict): return {k: _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list): return [_to_native(x) for x in obj]
    elif isinstance(obj, (pd.Series, pd.DataFrame)): return _to_native(obj.to_dict())
    return obj

def safe_float(val, default=0.0):
    try:
        if val is None or (isinstance(val, float) and (np.isnan(val) or np.isinf(val))): return default
        return float(val)
    except: return default

def fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    buf.seek(0)
    return f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"

def get_effect_size_interpretation(d):
    d_abs = abs(d)
    if d_abs < 0.2: return "negligible"
    elif d_abs < 0.5: return "small"
    elif d_abs < 0.8: return "medium"
    return "large"

def get_correlation_strength(r):
    r_abs = abs(r)
    if r_abs < 0.2: return "negligible"
    elif r_abs < 0.4: return "weak"
    elif r_abs < 0.7: return "moderate"
    return "strong"

def analyze_descriptive(df, sat_var, scale_points):
    series = df[sat_var].dropna()
    desc = {
        'n': int(len(series)), 'mean': safe_float(series.mean()), 'std': safe_float(series.std()),
        'median': safe_float(series.median()), 'min': safe_float(series.min()), 'max': safe_float(series.max()),
        'q1': safe_float(series.quantile(0.25)), 'q3': safe_float(series.quantile(0.75)),
        'se': safe_float(series.std() / np.sqrt(len(series))),
        'skewness': safe_float(series.skew()), 'kurtosis': safe_float(series.kurtosis())
    }
    value_counts = series.value_counts().sort_index()
    distribution = [{'value': val, 'count': int(value_counts.get(val, 0)), 
                     'percent': safe_float((value_counts.get(val, 0) / len(series)) * 100 if len(series) > 0 else 0)}
                    for val in range(1, scale_points + 1)]
    desc['distribution'] = distribution
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].bar(range(1, scale_points + 1), [d['count'] for d in distribution], color='#4A90A4', alpha=0.8)
    axes[0].axvline(x=desc['mean'], color='#E74C3C', linestyle='--', linewidth=2, label=f'Mean: {desc["mean"]:.2f}')
    axes[0].set_xlabel('Response Value'); axes[0].set_ylabel('Frequency'); axes[0].set_title('Response Distribution'); axes[0].legend()
    bp = axes[1].boxplot(series, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('#4A90A4'); axes[1].set_title('Box Plot')
    plt.tight_layout()
    return {'overall': desc, 'by_item': [], 'plot': fig_to_base64(fig)}

def analyze_items(df, sat_items):
    return [{'item': item, 'n': int(len(s)), 'mean': safe_float(s.mean()), 'std': safe_float(s.std()), 'median': safe_float(s.median())}
            for item in sat_items if item in df.columns for s in [pd.to_numeric(df[item], errors='coerce').dropna()] if len(s) > 0]

def analyze_scale(df, sat_var, scale_points):
    series = df[sat_var].dropna()
    n = len(series)
    if scale_points == 5:
        top_values, bottom_values, middle_values = [4, 5], [1, 2], [3]
        top_label, bottom_label, middle_label = "4-5 (Satisfied)", "1-2 (Dissatisfied)", "3 (Neutral)"
    elif scale_points == 7:
        top_values, bottom_values, middle_values = [6, 7], [1, 2], [3, 4, 5]
        top_label, bottom_label, middle_label = "6-7 (Satisfied)", "1-2 (Dissatisfied)", "3-5 (Neutral)"
    elif scale_points == 10:
        top_values, bottom_values, middle_values = [9, 10], [1, 2, 3, 4], [5, 6, 7, 8]
        top_label, bottom_label, middle_label = "9-10 (Highly Satisfied)", "1-4 (Dissatisfied)", "5-8 (Moderate)"
    else:
        top_values, bottom_values, middle_values = [9, 10], [0,1,2,3,4,5,6], [7, 8]
        top_label, bottom_label, middle_label = "9-10 (Promoters)", "0-6 (Detractors)", "7-8 (Passives)"
    
    top_count, bottom_count, middle_count = int(series.isin(top_values).sum()), int(series.isin(bottom_values).sum()), int(series.isin(middle_values).sum())
    top_pct, bottom_pct, middle_pct = (top_count/n)*100 if n>0 else 0, (bottom_count/n)*100 if n>0 else 0, (middle_count/n)*100 if n>0 else 0
    sat_index = (series.mean()/10)*100 if scale_points==11 else ((series.mean()-1)/(scale_points-1))*100
    
    result = {'scale_type': f'{scale_points}-point',
              'top_box': {'count': top_count, 'percent': safe_float(top_pct), 'label': top_label},
              'bottom_box': {'count': bottom_count, 'percent': safe_float(bottom_pct), 'label': bottom_label},
              'middle_box': {'count': middle_count, 'percent': safe_float(middle_pct), 'label': middle_label},
              'satisfaction_index': safe_float(sat_index)}
    if scale_points == 11: result['nps_score'] = safe_float(top_pct - bottom_pct)
    return result

def analyze_item_ranking(df, sat_items, scale_points):
    if not sat_items: return {'rankings': [], 'strengths': [], 'improvements': [], 'plot': None}
    item_stats = [{'item': item, 'mean': s.mean(), 'std': s.std()} for item in sat_items if item in df.columns 
                  for s in [pd.to_numeric(df[item], errors='coerce').dropna()] if len(s) > 0]
    if not item_stats: return {'rankings': [], 'strengths': [], 'improvements': [], 'plot': None}
    item_stats.sort(key=lambda x: x['mean'], reverse=True)
    mean_of_means, std_of_means = np.mean([x['mean'] for x in item_stats]), np.std([x['mean'] for x in item_stats]) if len(item_stats)>1 else 0
    
    rankings, strengths, improvements = [], [], []
    for rank, item in enumerate(item_stats, 1):
        if item['mean'] >= mean_of_means + 0.5*std_of_means: category, strengths = 'strength', strengths + [item['item']]
        elif item['mean'] <= mean_of_means - 0.5*std_of_means: category, improvements = 'improvement', improvements + [item['item']]
        else: category = 'neutral'
        rankings.append({'rank': rank, 'item': item['item'], 'mean': safe_float(item['mean']), 'std': safe_float(item['std']), 'category': category})
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(rankings)*0.5)))
    items, means, cats = [r['item'] for r in rankings][::-1], [r['mean'] for r in rankings][::-1], [r['category'] for r in rankings][::-1]
    colors = ['#2ECC71' if c=='strength' else '#E74C3C' if c=='improvement' else '#95A5A6' for c in cats]
    ax.barh(items, means, color=colors, alpha=0.8)
    ax.axvline(x=mean_of_means, color='#3498DB', linestyle='--', linewidth=2, label=f'Avg: {mean_of_means:.2f}')
    ax.set_xlabel('Mean Score'); ax.set_title('Satisfaction Item Ranking'); ax.legend(); ax.set_xlim(0, scale_points+0.5)
    plt.tight_layout()
    return {'rankings': rankings, 'strengths': strengths, 'improvements': improvements, 'plot': fig_to_base64(fig)}

def analyze_crosstab(df, sat_var, group_vars, scale_points):
    if not group_vars: return []
    results = []
    for gv in group_vars:
        if gv not in df.columns: continue
        ct = pd.crosstab(df[gv], df[sat_var])
        try:
            chi2, p_value, _, _ = chi2_contingency(ct)
            cramers_v = np.sqrt(chi2/(ct.sum().sum()*max(1,min(ct.shape)-1)))
        except: chi2, p_value, cramers_v = 0, 1, 0
        group_means = {str(g): {'mean': safe_float(gd.mean()), 'n': int(len(gd)), 'std': safe_float(gd.std())}
                       for g in df[gv].dropna().unique() for gd in [df[df[gv]==g][sat_var].dropna()] if len(gd)>0}
        fig, ax = plt.subplots(figsize=(10, 6))
        groups, means = list(group_means.keys()), [group_means[g]['mean'] for g in group_means.keys()]
        ax.bar(groups, means, color='#4A90A4', alpha=0.8)
        ax.set_xlabel(gv); ax.set_ylabel(f'Mean {sat_var}'); ax.set_title(f'{sat_var} by {gv}'); ax.set_ylim(0, scale_points+0.5)
        for i, m in enumerate(means): ax.text(i, m+0.1, f'{m:.2f}', ha='center')
        plt.tight_layout()
        results.append({'group_var': gv, 'chi_square': safe_float(chi2), 'p_value': safe_float(p_value),
                       'cramers_v': safe_float(cramers_v), 'significant': bool(p_value<0.05), 'group_means': group_means, 'plot': fig_to_base64(fig)})
    return results

def analyze_mean_comparison(df, sat_var, group_vars):
    if not group_vars: return []
    results = []
    for gv in group_vars:
        if gv not in df.columns: continue
        groups_data = {str(g): gd for g in df[gv].dropna().unique() for gd in [df[df[gv]==g][sat_var].dropna()] if len(gd)>=2}
        if len(groups_data) < 2: continue
        group_names, group_values = list(groups_data.keys()), list(groups_data.values())
        groups_stats = {n: {'n': int(len(d)), 'mean': safe_float(d.mean()), 'std': safe_float(d.std()), 'se': safe_float(d.std()/np.sqrt(len(d)))} for n,d in groups_data.items()}
        
        if len(groups_data) == 2:
            test_type, (t_stat, p_val) = 'ttest', ttest_ind(group_values[0], group_values[1])
            test_stat, p_value = safe_float(t_stat), safe_float(p_val)
            ps = np.sqrt(((len(group_values[0])-1)*group_values[0].std()**2+(len(group_values[1])-1)*group_values[1].std()**2)/(len(group_values[0])+len(group_values[1])-2))
            effect_size = (group_values[0].mean()-group_values[1].mean())/ps if ps>0 else 0
            posthoc = None
        else:
            test_type, (f_stat, p_val) = 'anova', f_oneway(*group_values)
            test_stat, p_value = safe_float(f_stat), safe_float(p_val)
            all_data = pd.concat([pd.Series(v) for v in group_values])
            gm = all_data.mean()
            ssb, sst = sum(len(v)*(v.mean()-gm)**2 for v in group_values), sum((all_data-gm)**2)
            effect_size = ssb/sst if sst>0 else 0
            posthoc = []
            if p_value < 0.05:
                try:
                    av, ag = [], []
                    for n, d in groups_data.items(): av.extend(d.tolist()); ag.extend([n]*len(d))
                    tukey = pairwise_tukeyhsd(av, ag, alpha=0.05)
                    for i in range(len(tukey.summary().data)-1):
                        r = tukey.summary().data[i+1]
                        posthoc.append({'group1': str(r[0]), 'group2': str(r[1]), 'diff': safe_float(r[2]), 'p_value': safe_float(r[3]), 'significant': bool(r[5]) if len(r)>5 else safe_float(r[3])<0.05})
                except: pass
        
        fig, ax = plt.subplots(figsize=(10, 6))
        means, stds = [groups_stats[g]['mean'] for g in group_names], [groups_stats[g]['std'] for g in group_names]
        ax.bar(range(len(groups_stats)), means, yerr=stds, capsize=5, color='#4A90A4', alpha=0.8)
        ax.set_xticks(range(len(group_names))); ax.set_xticklabels(group_names)
        ax.set_xlabel(gv); ax.set_ylabel(f'Mean {sat_var}'); ax.set_title(f'Mean Comparison: {sat_var} by {gv}')
        plt.tight_layout()
        results.append({'test_type': test_type, 'group_var': gv, 'groups': groups_stats, 'test_statistic': test_stat,
                       'p_value': p_value, 'significant': bool(p_value<0.05), 'effect_size': safe_float(effect_size),
                       'effect_interpretation': get_effect_size_interpretation(effect_size), 'posthoc': posthoc, 'plot': fig_to_base64(fig)})
    return results

def analyze_correlation(df, sat_var, sat_items):
    if not sat_items: return None
    all_vars = [sat_var] + [v for v in sat_items if v != sat_var and v in df.columns]
    if len(all_vars) < 2: return None
    correlations = []
    for i, v1 in enumerate(all_vars):
        for v2 in all_vars[i+1:]:
            s1, s2 = pd.to_numeric(df[v1], errors='coerce').dropna(), pd.to_numeric(df[v2], errors='coerce').dropna()
            idx = s1.index.intersection(s2.index)
            if len(idx) >= 3:
                r, p = pearsonr(s1.loc[idx], s2.loc[idx])
                correlations.append({'var1': v1, 'var2': v2, 'r': safe_float(r), 'p_value': safe_float(p), 'significant': bool(p<0.05)})
    corr_matrix = df[all_vars].apply(pd.to_numeric, errors='coerce').corr()
    key_drivers = sorted([{'variable': c['var2'] if c['var1']==sat_var else c['var1'], 'correlation': c['r'], 'strength': get_correlation_strength(c['r'])}
                          for c in correlations if sat_var in [c['var1'], c['var2']]], key=lambda x: abs(x['correlation']), reverse=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', center=0, vmin=-1, vmax=1, ax=ax, square=True)
    ax.set_title('Correlation Matrix'); plt.tight_layout()
    return {'correlations': correlations, 'correlation_matrix': {str(k): {str(kk): safe_float(vv) for kk,vv in v.items()} for k,v in corr_matrix.to_dict().items()}, 'key_drivers': key_drivers, 'plot': fig_to_base64(fig)}

def analyze_regression(df, sat_var, sat_items):
    if not sat_items: return None
    predictors = [v for v in sat_items if v != sat_var and v in df.columns]
    if not predictors: return None
    analysis_df = df[[sat_var]+predictors].apply(pd.to_numeric, errors='coerce').dropna()
    if len(analysis_df) < len(predictors)+2: return None
    X, y = analysis_df[predictors], analysis_df[sat_var]
    try: model = sm.OLS(y, sm.add_constant(X)).fit()
    except: return None
    try:
        X_std, y_std = (X-X.mean())/X.std(), (y-y.mean())/y.std()
        betas = sm.OLS(y_std, sm.add_constant(X_std)).fit().params
    except: betas = pd.Series([0]*(len(predictors)+1), index=['const']+predictors)
    
    coefficients = [{'variable': v, 'coefficient': safe_float(model.params[v]), 'std_error': safe_float(model.bse[v]),
                     't_value': safe_float(model.tvalues[v]), 'p_value': safe_float(model.pvalues[v]),
                     'significant': bool(model.pvalues[v]<0.05), 'beta': safe_float(betas.get(v, 0))} for v in ['const']+predictors if v in model.params.index]
    total_beta = sum(abs(c['beta']) for c in coefficients if c['variable']!='const')
    importance = sorted([{'variable': c['variable'], 'importance': abs(c['beta']), 'percent': (abs(c['beta'])/total_beta)*100}
                        for c in coefficients if c['variable']!='const' and total_beta>0], key=lambda x: x['importance'], reverse=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    pred_coefs = [c for c in coefficients if c['variable']!='const']
    if pred_coefs:
        vars_n, betas_v = [c['variable'] for c in pred_coefs], [c['beta'] for c in pred_coefs]
        colors = ['#2ECC71' if b>0 else '#E74C3C' for b in betas_v]
        axes[0].barh(vars_n, betas_v, color=colors, alpha=0.8)
        axes[0].axvline(x=0, color='black', linewidth=0.5)
        axes[0].set_xlabel('Standardized Coefficient (β)'); axes[0].set_title('Regression Coefficients')
    if importance:
        labels, sizes = [f"{i['variable']}\n({i['percent']:.1f}%)" for i in importance[:5]], [i['percent'] for i in importance[:5]]
        axes[1].pie(sizes, labels=labels, colors=plt.cm.Blues(np.linspace(0.3, 0.9, len(sizes))), startangle=90, wedgeprops={'edgecolor': 'white'})
        axes[1].set_title('Relative Importance')
    plt.tight_layout()
    return {'r_squared': safe_float(model.rsquared), 'adjusted_r_squared': safe_float(model.rsquared_adj),
            'f_statistic': safe_float(model.fvalue), 'f_pvalue': safe_float(model.f_pvalue),
            'coefficients': coefficients, 'importance_ranking': importance, 'plot': fig_to_base64(fig), 'residual_plot': None}

def analyze_ipa(df, sat_var, sat_items, scale_points):
    if not sat_items or len(sat_items) < 2: return None
    predictors = [v for v in sat_items if v != sat_var and v in df.columns]
    if not predictors: return None
    analysis_df = df[[sat_var]+predictors].apply(pd.to_numeric, errors='coerce').dropna()
    if len(analysis_df) < len(predictors)+2: return None
    
    performance = analysis_df[predictors].mean()
    X, y = analysis_df[predictors], analysis_df[sat_var]
    X_scaled = StandardScaler().fit_transform(X)
    model = LinearRegression().fit(X_scaled, y)
    beta_df = pd.DataFrame({'attribute': predictors, 'beta': model.coef_})
    total_abs = beta_df['beta'].abs().sum()
    beta_df['relative_importance'] = (beta_df['beta'].abs()/total_abs)*100 if total_abs>0 else 0
    
    ipa_data = [{'attribute': attr, 'performance': float(performance.get(attr, 0)),
                 'importance': float(beta_df[beta_df['attribute']==attr].iloc[0]['beta']),
                 'relative_importance': float(beta_df[beta_df['attribute']==attr].iloc[0]['relative_importance'])} for attr in predictors]
    df_ipa = pd.DataFrame(ipa_data)
    perf_mean, imp_mean = float(df_ipa['performance'].mean()), 0
    
    def classify(row):
        if row['importance']>=imp_mean and row['performance']>=perf_mean: return 'Q1: Keep Up Good Work'
        elif row['importance']>=imp_mean and row['performance']<perf_mean: return 'Q2: Concentrate Here'
        elif row['importance']<imp_mean and row['performance']<perf_mean: return 'Q3: Low Priority'
        return 'Q4: Possible Overkill'
    df_ipa['quadrant'] = df_ipa.apply(classify, axis=1)
    
    qcolors = {'Q1: Keep Up Good Work': '#4CAF50', 'Q2: Concentrate Here': '#F44336', 'Q3: Low Priority': '#9E9E9E', 'Q4: Possible Overkill': '#FF9800'}
    fig, ax = plt.subplots(figsize=(10, 8))
    for q, c in qcolors.items():
        d = df_ipa[df_ipa['quadrant']==q]
        if not d.empty: ax.scatter(d['performance'], d['importance'], c=c, s=300, alpha=0.7, label=q, edgecolors='black', linewidth=1.5)
    for _, r in df_ipa.iterrows(): ax.text(r['performance'], r['importance'], r['attribute'][:10], fontsize=9, ha='center', va='center', fontweight='bold')
    ax.axhline(y=imp_mean, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.axvline(x=perf_mean, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    ax.set_xlabel('Performance (Mean Score)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Importance (β Coefficient)', fontsize=12, fontweight='bold')
    ax.set_title('IPA Matrix', fontsize=14, fontweight='bold'); ax.legend(loc='best', fontsize=9); ax.grid(True, alpha=0.3)
    plt.tight_layout(); ipa_plot = fig_to_base64(fig)
    
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    df_gap = df_ipa.copy()
    df_gap['gap'] = df_gap['performance'] - (df_gap['relative_importance']/df_gap['relative_importance'].max()*scale_points if df_gap['relative_importance'].max()>0 else 0)
    df_gap = df_gap.sort_values('gap')
    ax2.barh(df_gap['attribute'], df_gap['gap'], color=['#F44336' if g<0 else '#4CAF50' for g in df_gap['gap']], alpha=0.7, edgecolor='black')
    ax2.axvline(0, color='k', lw=1); ax2.set_title('Gap Analysis', fontweight='bold'); ax2.set_xlabel('Gap')
    plt.tight_layout(); gap_plot = fig_to_base64(fig2)
    
    ipa_matrix = df_ipa.to_dict('records')
    return {'ipa_matrix': ipa_matrix, 'performance_mean': perf_mean, 'importance_mean': imp_mean, 'r_squared': float(model.score(X_scaled, y)),
            'quadrant_summary': {'Q1': int(len(df_ipa[df_ipa['quadrant']=='Q1: Keep Up Good Work'])), 'Q2': int(len(df_ipa[df_ipa['quadrant']=='Q2: Concentrate Here'])),
                                 'Q3': int(len(df_ipa[df_ipa['quadrant']=='Q3: Low Priority'])), 'Q4': int(len(df_ipa[df_ipa['quadrant']=='Q4: Possible Overkill']))},
            'concentrate_here': [r['attribute'] for r in ipa_matrix if r['quadrant']=='Q2: Concentrate Here'],
            'keep_up': [r['attribute'] for r in ipa_matrix if r['quadrant']=='Q1: Keep Up Good Work'], 'ipa_plot': ipa_plot, 'gap_plot': gap_plot}

def generate_conclusion(results, sat_var, scale_points):
    desc, scale = results['descriptive_stats']['overall'], results['scale_analysis']
    sat_index = scale['satisfaction_index']
    level = 'HIGH' if sat_index>=70 else 'MODERATE' if sat_index>=50 else 'LOW'
    
    # Section summaries with plain language explanations
    section_summaries = {}
    
    # 1. Descriptive summary
    mean_pct = (desc['mean'] / scale_points) * 100
    section_summaries['descriptive'] = {
        'title': 'Basic Statistics',
        'summary': f"Average satisfaction score is {desc['mean']:.2f} out of {scale_points}.",
        'meaning': f"{'Most respondents are satisfied with the service/product.' if mean_pct >= 70 else 'Satisfaction is at a moderate level. There is room for improvement.' if mean_pct >= 50 else 'Overall dissatisfaction detected. Immediate improvement is needed.'}",
        'interpretation': 'POSITIVE' if mean_pct >= 70 else ('NEUTRAL' if mean_pct >= 50 else 'NEGATIVE')
    }
    
    # 2. Scale analysis summary
    top_pct, bottom_pct = scale['top_box']['percent'], scale['bottom_box']['percent']
    ratio = top_pct / bottom_pct if bottom_pct > 0 else 10
    section_summaries['scale'] = {
        'title': 'Satisfaction Ratio',
        'summary': f"Satisfied: {top_pct:.1f}% vs Dissatisfied: {bottom_pct:.1f}%",
        'meaning': f"{'Satisfied customers outnumber dissatisfied ones by ' + str(round(ratio, 1)) + 'x. This is a positive sign.' if ratio >= 2 else 'The ratio between satisfied and dissatisfied is close. Focus on reducing dissatisfaction.' if ratio >= 1 else 'More customers are dissatisfied than satisfied. Urgent action required.'}",
        'interpretation': 'POSITIVE' if ratio >= 2 else ('NEUTRAL' if ratio >= 1 else 'NEGATIVE')
    }
    
    # 3. Item ranking summary
    item_ranking = results.get('item_ranking', {})
    strengths = item_ranking.get('strengths', [])
    improvements = item_ranking.get('improvements', [])
    section_summaries['item_ranking'] = {
        'title': 'Item Rankings',
        'summary': f"{len(strengths)} strength(s), {len(improvements)} area(s) needing improvement",
        'meaning': f"{'Strengths: ' + ', '.join(strengths[:2]) + '. ' if strengths else ''}{'Priority improvements: ' + ', '.join(improvements[:2]) + '.' if improvements else 'No critical weak points identified.'}",
        'interpretation': 'BALANCED' if len(strengths) >= len(improvements) else 'NEEDS_WORK'
    }
    
    # 4. Group comparison summary
    mean_comp = results.get('mean_comparison', [])
    sig_diffs = [mc['group_var'] for mc in (mean_comp or []) if mc.get('significant')]
    section_summaries['comparison'] = {
        'title': 'Group Differences',
        'summary': f"{'Significant differences in: ' + ', '.join(sig_diffs) if sig_diffs else 'No significant differences between groups'}",
        'meaning': f"{'These groups show different satisfaction levels. Focus on the lower-scoring groups.' if sig_diffs else 'All groups have similar satisfaction. A unified improvement strategy will work.'}",
        'interpretation': 'ACTION_NEEDED' if sig_diffs else 'UNIFORM'
    }
    
    # 5. Correlation summary
    corr = results.get('correlation_analysis')
    if corr and corr.get('key_drivers'):
        top_corr = corr['key_drivers'][0] if corr['key_drivers'] else None
        r_val = abs(top_corr['correlation']) if top_corr else 0
        section_summaries['correlation'] = {
            'title': 'Correlation Analysis',
            'summary': f"Strongest correlate: {top_corr['variable']}" if top_corr else "No significant correlations",
            'meaning': f"{'Improving this item is likely to boost overall satisfaction.' if r_val >= 0.5 else 'This item has moderate influence on overall satisfaction.' if r_val >= 0.3 else 'Individual items show weak correlation with overall satisfaction.'}",
            'interpretation': 'STRONG' if r_val >= 0.5 else ('MODERATE' if r_val >= 0.3 else 'WEAK')
        }
    
    # 6. Regression summary (Drivers)
    reg = results.get('regression_analysis')
    if reg:
        r2 = reg.get('r_squared', 0)
        top_driver = reg['importance_ranking'][0] if reg.get('importance_ranking') else None
        section_summaries['regression'] = {
            'title': 'Key Drivers',
            'summary': f"#1 Driver: {top_driver['variable']} ({top_driver['percent']:.1f}% contribution)" if top_driver else "Driver analysis unavailable",
            'meaning': f"{'Improving this item will have the biggest impact on overall satisfaction.' if top_driver else ''} {'The model explains ' + str(round(r2*100)) + '% of satisfaction variance. ' + ('Results are reliable.' if r2 >= 0.5 else 'Use as reference.' if r2 >= 0.3 else 'Additional factors may be at play.')}",
            'interpretation': 'RELIABLE' if r2 >= 0.5 else ('MODERATE' if r2 >= 0.3 else 'LIMITED')
        }
    
    # 7. IPA summary
    ipa = results.get('ipa_analysis')
    if ipa:
        q_sum = ipa.get('quadrant_summary', {})
        concentrate = ipa.get('concentrate_here', [])
        keep_up = ipa.get('keep_up', [])
        section_summaries['ipa'] = {
            'title': 'IPA Priority Analysis',
            'summary': f"Urgent improvement: {q_sum.get('Q2', 0)} item(s), Maintain: {q_sum.get('Q1', 0)} item(s)",
            'meaning': f"{'⚠️ ' + ', '.join(concentrate[:2]) + ' - High importance but low performance. Prioritize these!' if concentrate else '✓ All important items are performing well.'} {' ★ ' + ', '.join(keep_up[:2]) + ' - Keep up the good work!' if keep_up else ''}",
            'interpretation': 'URGENT' if q_sum.get('Q2', 0) > 0 else 'GOOD'
        }
    
    # Numbered findings - Plain English
    numbered_findings = []
    
    # Finding 1: Current state
    if level == 'HIGH':
        numbered_findings.append(f"Satisfaction is strong. Average score is {desc['mean']:.1f} out of {scale_points} (Index: {sat_index:.0f}/100).")
    elif level == 'MODERATE':
        numbered_findings.append(f"Satisfaction is moderate. Average score is {desc['mean']:.1f}, indicating room for improvement.")
    else:
        numbered_findings.append(f"Satisfaction is low. Average score is {desc['mean']:.1f}, requiring urgent attention.")
    
    # Finding 2: Satisfaction ratio
    if ratio >= 2:
        numbered_findings.append(f"Satisfied customers ({top_pct:.0f}%) outnumber dissatisfied ones ({bottom_pct:.0f}%) by {ratio:.1f}x. This is healthy.")
    else:
        numbered_findings.append(f"Dissatisfied customer rate ({bottom_pct:.0f}%) is concerning. Risk of churn exists. Address pain points.")
    
    # Finding 3: Key driver
    if reg and reg.get('importance_ranking'):
        top3 = reg['importance_ranking'][:3]
        numbered_findings.append(f"To improve satisfaction, focus on '{top3[0]['variable']}'. It accounts for {top3[0]['percent']:.0f}% of satisfaction impact.")
    
    # Finding 4: IPA-based priorities
    if ipa and ipa.get('concentrate_here'):
        numbered_findings.append(f"Urgent action items: {', '.join(ipa['concentrate_here'][:3])}. These are important but underperforming.")
    elif improvements:
        numbered_findings.append(f"Areas needing improvement: {', '.join(improvements[:3])}. These scored below average.")
    
    # Finding 5: Group differences
    if sig_diffs:
        numbered_findings.append(f"Satisfaction varies across {', '.join(sig_diffs)}. Target the lower-scoring segments with tailored strategies.")
    
    # Final recommendation - Actionable advice
    if level == 'HIGH':
        final_rec = f"Performance is strong. Maintain focus on your strengths{' (' + ', '.join(keep_up[:2]) + ')' if ipa and keep_up else ''} while exploring opportunities to achieve excellence in {improvements[0] if improvements else 'remaining areas'}."
    elif level == 'MODERATE':
        priority = concentrate[0] if ipa and concentrate else (improvements[0] if improvements else 'low-scoring items')
        final_rec = f"Prioritize improving '{priority}'. Customers value this highly but current performance falls short. Improving it will yield the biggest satisfaction gains."
    else:
        final_rec = f"Urgent action required. Step 1: Fix '{concentrate[0] if ipa and concentrate else improvements[0] if improvements else 'critical items'}'. Step 2: Conduct root cause analysis for dissatisfied customers. Step 3: Re-survey in 3 months to measure progress."
    
    top_drivers = [f"{i['variable']} ({i['percent']:.1f}%)" for i in (reg or {}).get('importance_ranking', [])[:3]]
    improvement_areas = (ipa or {}).get('concentrate_here', [])[:3] or improvements[:3]
    
    return {
        'satisfaction_level': level,
        'satisfaction_index': safe_float(sat_index),
        'section_summaries': section_summaries,
        'numbered_findings': numbered_findings,
        'key_findings': numbered_findings[:3],
        'top_drivers': top_drivers,
        'improvement_areas': improvement_areas,
        'recommendation': final_rec
    }

@router.post("/satisfaction")
async def satisfaction_analysis(request: SatisfactionRequest):
    try:
        df = pd.DataFrame(request.data)
        sat_var, sat_items, group_vars, scale_points = request.satisfaction_var, request.satisfaction_items or [], request.group_vars or [], request.scale_points or 5
        if sat_var not in df.columns: raise HTTPException(status_code=400, detail=f"'{sat_var}' not found")
        df[sat_var] = pd.to_numeric(df[sat_var], errors='coerce')
        n_total, n_valid = len(df), int(df[sat_var].notna().sum())
        if n_valid < 3: raise HTTPException(status_code=400, detail="Need at least 3 valid observations")
        
        results = {'descriptive_stats': analyze_descriptive(df, sat_var, scale_points)}
        if sat_items: results['descriptive_stats']['by_item'] = analyze_items(df, sat_items)
        results['scale_analysis'] = analyze_scale(df, sat_var, scale_points)
        results['item_ranking'] = analyze_item_ranking(df, sat_items, scale_points)
        results['cross_tabulation'] = analyze_crosstab(df, sat_var, group_vars, scale_points) or None
        results['mean_comparison'] = analyze_mean_comparison(df, sat_var, group_vars) or None
        results['correlation_analysis'] = analyze_correlation(df, sat_var, sat_items)
        results['regression_analysis'] = analyze_regression(df, sat_var, sat_items)
        results['ipa_analysis'] = analyze_ipa(df, sat_var, sat_items, scale_points)
        results['overall_conclusion'] = generate_conclusion(results, sat_var, scale_points)
        results['summary_statistics'] = {'n_total': n_total, 'n_valid': n_valid, 'satisfaction_var': sat_var, 'satisfaction_items': sat_items, 'group_vars': group_vars, 'scale_points': scale_points}
        return _to_native(results)
    except HTTPException: raise
    except Exception as e:
        import traceback; traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
