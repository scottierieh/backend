"""
Organization Health Diagnosis API - 5-step framework
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64

router = APIRouter()

class OrgHealthRequest(BaseModel):
    data: List[Dict[str, Any]]
    engagement_col: str
    env_cols: List[str]
    satisfaction_col: Optional[str] = None
    dept_col: Optional[str] = None
    level_col: Optional[str] = None

def _to_native(obj):
    if obj is None: return None
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj) if not np.isnan(obj) else None
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64

def analyze_overview(df, engagement_col, satisfaction_col):
    engagement = pd.to_numeric(df[engagement_col], errors='coerce')
    n = len(df)
    avg_eng = engagement.mean()
    highly_engaged = (engagement >= 4).sum()
    moderate = ((engagement >= 3) & (engagement < 4)).sum()
    disengaged = (engagement < 3).sum()
    grade = 'A' if avg_eng >= 4.0 else 'B' if avg_eng >= 3.5 else 'C' if avg_eng >= 3.0 else 'D' if avg_eng >= 2.5 else 'F'
    result = {'n_employees': n, 'avg_engagement': _to_native(avg_eng), 'engagement_std': _to_native(engagement.std()), 'highly_engaged_pct': _to_native(highly_engaged / n * 100), 'moderate_engaged_pct': _to_native(moderate / n * 100), 'disengaged_pct': _to_native(disengaged / n * 100), 'health_grade': grade}
    if satisfaction_col and satisfaction_col in df.columns:
        sat = pd.to_numeric(df[satisfaction_col], errors='coerce')
        result['avg_satisfaction'] = _to_native(sat.mean())
        result['satisfaction_std'] = _to_native(sat.std())
    return result

def create_overview_chart(overview, df, engagement_col):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    engagement = pd.to_numeric(df[engagement_col], errors='coerce')
    axes[0].hist(engagement.dropna(), bins=20, color='#3b82f6', alpha=0.7, edgecolor='black')
    axes[0].axvline(x=overview['avg_engagement'], color='red', linestyle='--', label=f"Avg: {overview['avg_engagement']:.2f}")
    axes[0].axvline(x=3.5, color='green', linestyle=':', alpha=0.7, label='Target: 3.5')
    axes[0].set_xlabel('Engagement Score'); axes[0].set_ylabel('Frequency'); axes[0].set_title('Engagement Distribution', fontsize=11, fontweight='bold'); axes[0].legend()
    sizes = [overview['highly_engaged_pct'], overview['moderate_engaged_pct'], overview['disengaged_pct']]
    axes[1].pie(sizes, labels=['Highly Engaged', 'Moderate', 'Disengaged'], colors=['#10b981', '#3b82f6', '#ef4444'], autopct='%1.1f%%')
    axes[1].set_title('Engagement Segments', fontsize=11, fontweight='bold')
    plt.tight_layout()
    return _fig_to_base64(fig)

def analyze_comparison(df, engagement_col, satisfaction_col, dept_col, level_col):
    if not dept_col and not level_col: return {'error': 'No department or level column'}
    engagement = pd.to_numeric(df[engagement_col], errors='coerce')
    avg_eng = engagement.mean()
    result = {'avg_engagement': _to_native(avg_eng)}
    if dept_col and dept_col in df.columns:
        dept_stats = []
        for dept in df[dept_col].unique():
            mask = df[dept_col] == dept
            eng = engagement[mask].mean()
            sat = pd.to_numeric(df.loc[mask, satisfaction_col], errors='coerce').mean() if satisfaction_col and satisfaction_col in df.columns else None
            dept_stats.append({'dept': str(dept), 'engagement': _to_native(eng), 'satisfaction': _to_native(sat), 'count': int(mask.sum()), 'status': 'Below Avg' if eng < avg_eng * 0.95 else 'Above Avg' if eng > avg_eng * 1.05 else 'Average'})
        dept_stats = sorted(dept_stats, key=lambda x: x['engagement'] or 0, reverse=True)
        result['dept_stats'] = dept_stats
        result['highest_dept'], result['highest_dept_score'] = dept_stats[0]['dept'], dept_stats[0]['engagement']
        result['lowest_dept'], result['lowest_dept_score'] = dept_stats[-1]['dept'], dept_stats[-1]['engagement']
        result['dept_gap'] = _to_native(dept_stats[0]['engagement'] - dept_stats[-1]['engagement'])
    if level_col and level_col in df.columns:
        level_stats = [{'level': str(lvl), 'engagement': _to_native(engagement[df[level_col] == lvl].mean()), 'count': int((df[level_col] == lvl).sum())} for lvl in df[level_col].unique()]
        level_stats = sorted(level_stats, key=lambda x: x['engagement'] or 0, reverse=True)
        result['level_stats'] = level_stats
    return result

def create_comparison_chart(comparison, dept_col, level_col):
    if comparison.get('error'): return ""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if comparison.get('dept_stats'):
        depts = [d['dept'][:12] for d in comparison['dept_stats'][:8]]
        scores = [d['engagement'] for d in comparison['dept_stats'][:8]]
        colors = ['#10b981' if s >= comparison['avg_engagement'] else '#ef4444' for s in scores]
        axes[0].barh(depts, scores, color=colors, alpha=0.8, edgecolor='black')
        axes[0].axvline(x=comparison['avg_engagement'], color='blue', linestyle='--', label=f"Avg: {comparison['avg_engagement']:.2f}")
        axes[0].set_xlabel('Engagement'); axes[0].set_title('By Department', fontsize=11, fontweight='bold'); axes[0].legend()
    if comparison.get('level_stats'):
        levels = [l['level'][:12] for l in comparison['level_stats']]
        scores = [l['engagement'] for l in comparison['level_stats']]
        axes[1].bar(levels, scores, color='#3b82f6', alpha=0.8, edgecolor='black')
        axes[1].set_ylabel('Engagement'); axes[1].set_title('By Level', fontsize=11, fontweight='bold')
        plt.sca(axes[1]); plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return _fig_to_base64(fig)

def analyze_correlation(df, engagement_col, env_cols):
    engagement = pd.to_numeric(df[engagement_col], errors='coerce')
    correlations = []
    for col in env_cols:
        if col not in df.columns: continue
        vals = pd.to_numeric(df[col], errors='coerce')
        valid = engagement.notna() & vals.notna()
        if valid.sum() < 10: continue
        corr, p_val = stats.pearsonr(engagement[valid], vals[valid])
        avg_score = vals.mean()
        priority = 'High' if abs(corr) > 0.5 and avg_score < 3.5 else 'Medium' if abs(corr) > 0.3 else 'Low'
        correlations.append({'factor': col, 'correlation': _to_native(corr), 'p_value': _to_native(p_val), 'significant': p_val < 0.05, 'impact': 'Positive' if corr > 0 else 'Negative', 'avg_score': _to_native(avg_score), 'priority': priority, 'strength': 'Strong' if abs(corr) > 0.5 else 'Moderate' if abs(corr) > 0.3 else 'Weak'})
    correlations = sorted(correlations, key=lambda x: abs(x.get('correlation') or 0), reverse=True)
    return {'correlations': correlations, 'top_driver': correlations[0] if correlations else None, 'n_significant': sum(1 for c in correlations if c['significant'])}

def create_correlation_chart(correlation, df, engagement_col):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    corrs = correlation['correlations'][:8]
    axes[0].barh([c['factor'][:15] for c in corrs], [c['correlation'] for c in corrs], color=['#3b82f6' if c['significant'] else '#d1d5db' for c in corrs], alpha=0.8, edgecolor='black')
    axes[0].axvline(x=0, color='black', linewidth=0.5); axes[0].set_xlabel('Correlation with Engagement'); axes[0].set_title('Engagement Drivers', fontsize=11, fontweight='bold')
    top = correlation.get('top_driver')
    if top and top['factor'] in df.columns:
        engagement = pd.to_numeric(df[engagement_col], errors='coerce')
        driver = pd.to_numeric(df[top['factor']], errors='coerce')
        valid = engagement.notna() & driver.notna()
        axes[1].scatter(driver[valid], engagement[valid], alpha=0.5, color='#3b82f6')
        if valid.sum() > 5:
            z = np.polyfit(driver[valid], engagement[valid], 1); p = np.poly1d(z)
            axes[1].plot(np.linspace(driver[valid].min(), driver[valid].max(), 100), p(np.linspace(driver[valid].min(), driver[valid].max(), 100)), color='#ef4444', linestyle='--', linewidth=2)
        axes[1].set_xlabel(top['factor']); axes[1].set_ylabel('Engagement'); axes[1].set_title(f"Top Driver (r={top['correlation']:.3f})", fontsize=11, fontweight='bold')
    plt.tight_layout()
    return _fig_to_base64(fig)

def analyze_burnout(df, engagement_col, env_cols):
    engagement = pd.to_numeric(df[engagement_col], errors='coerce')
    risk_score = 5 - engagement
    at_risk = (engagement < 2.5) | (risk_score > 3)
    n_at_risk = int(at_risk.sum())
    burnout_rate = n_at_risk / len(df) * 100
    risk_factors = []
    for col in env_cols:
        if col not in df.columns: continue
        vals = pd.to_numeric(df[col], errors='coerce')
        valid = engagement.notna() & vals.notna()
        if valid.sum() < 10: continue
        corr, _ = stats.pearsonr(engagement[valid], vals[valid])
        if corr < -0.1: risk_factors.append({'factor': col, 'correlation': _to_native(corr), 'direction': 'Higher values increase risk'})
        elif corr > 0.2: risk_factors.append({'factor': col, 'correlation': _to_native(corr), 'direction': 'Lower values increase risk'})
    segments = [
        {'level': 'High Risk', 'count': int((engagement < 2.5).sum()), 'pct': _to_native((engagement < 2.5).sum() / len(df) * 100), 'action': 'Immediate'},
        {'level': 'Medium Risk', 'count': int(((engagement >= 2.5) & (engagement < 3.5)).sum()), 'pct': _to_native(((engagement >= 2.5) & (engagement < 3.5)).sum() / len(df) * 100), 'action': 'Monitor'},
        {'level': 'Low Risk', 'count': int((engagement >= 3.5).sum()), 'pct': _to_native((engagement >= 3.5).sum() / len(df) * 100), 'action': 'Maintain'}
    ]
    return {'burnout_rate': _to_native(burnout_rate), 'n_at_risk': n_at_risk, 'avg_risk_score': _to_native(risk_score.mean()), 'risk_factors': risk_factors[:5], 'segments': segments}

def create_burnout_chart(burnout, df, engagement_col):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    engagement = pd.to_numeric(df[engagement_col], errors='coerce')
    risk_score = 5 - engagement
    axes[0].hist(risk_score.dropna(), bins=20, color='#ef4444', alpha=0.7, edgecolor='black')
    axes[0].axvline(x=2.5, color='orange', linestyle='--', label='Medium Risk'); axes[0].axvline(x=3.5, color='red', linestyle='--', label='High Risk')
    axes[0].set_xlabel('Risk Score'); axes[0].set_ylabel('Frequency'); axes[0].set_title('Burnout Risk Distribution', fontsize=11, fontweight='bold'); axes[0].legend()
    segs = burnout['segments']
    axes[1].pie([s['pct'] for s in segs], labels=[s['level'] for s in segs], colors=['#ef4444', '#f97316', '#10b981'], autopct='%1.1f%%')
    axes[1].set_title('Risk Segments', fontsize=11, fontweight='bold')
    plt.tight_layout()
    return _fig_to_base64(fig)

def simulate_improvement(df, engagement_col, correlation, overview):
    current = overview['avg_engagement']
    top = correlation.get('top_driver')
    scenarios = []
    if top and top['correlation'] > 0:
        impact = abs(top['correlation']) * 0.5 * 10
        scenarios.append({'name': f"Improve {top['factor'][:15]}", 'description': 'Focus on top driver', 'engagement_gain': _to_native(impact), 'new_engagement': _to_native(min(5, current * (1 + impact/100))), 'productivity_gain': _to_native(impact * 1.5), 'recommended': True})
    scenarios.extend([
        {'name': 'Burnout Prevention', 'description': 'Workload balancing', 'engagement_gain': 8, 'new_engagement': _to_native(min(5, current * 1.08)), 'productivity_gain': 12, 'recommended': False},
        {'name': 'Comprehensive Program', 'description': 'Multi-factor intervention', 'engagement_gain': 15, 'new_engagement': _to_native(min(5, current * 1.15)), 'productivity_gain': 20, 'recommended': False}
    ])
    return {'current_engagement': current, 'scenarios': scenarios, 'best_scenario': max(scenarios, key=lambda x: x['productivity_gain']), 'recommendations': [f"Priority: Improve {top['factor']}" if top else "Implement engagement program", "Regular pulse surveys", "Manager training", "Recognition programs"]}

def create_simulation_chart(simulation):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    scenarios = simulation['scenarios']
    names = ['Current'] + [s['name'][:12] for s in scenarios]
    scores = [simulation['current_engagement']] + [s['new_engagement'] for s in scenarios]
    colors = ['#94a3b8'] + ['#10b981' if s['recommended'] else '#3b82f6' for s in scenarios]
    axes[0].bar(names, scores, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_ylabel('Engagement'); axes[0].set_title('Projected Engagement', fontsize=11, fontweight='bold'); axes[0].set_ylim(0, 5)
    plt.sca(axes[0]); plt.xticks(rotation=45, ha='right')
    axes[1].barh([s['name'][:12] for s in scenarios], [s['productivity_gain'] for s in scenarios], color=['#10b981' if s['recommended'] else '#3b82f6' for s in scenarios], alpha=0.8, edgecolor='black')
    axes[1].set_xlabel('Productivity Gain (%)'); axes[1].set_title('Expected Impact', fontsize=11, fontweight='bold')
    plt.tight_layout()
    return _fig_to_base64(fig)

def generate_report(overview, comparison, correlation, burnout, simulation):
    report = {}
    report['step1_overview'] = {
        'title': '1. Organization Health Score',
        'question': 'What is our organization health score and engagement level?',
        'finding': f"Engagement: {overview['avg_engagement']:.2f}/5.0 (Grade: {overview['health_grade']}), {overview['disengaged_pct']:.1f}% disengaged",
        'detail': (f"Organization health analysis of {overview['n_employees']:,} employees reveals average engagement of {overview['avg_engagement']:.2f}/5.0, "
                  f"earning a health grade of {overview['health_grade']}. "
                  f"Engagement distribution: {overview['highly_engaged_pct']:.1f}% highly engaged (≥4.0), {overview['moderate_engaged_pct']:.1f}% moderately engaged (3.0-3.9), {overview['disengaged_pct']:.1f}% disengaged (<3.0). "
                  + (f"Average satisfaction: {overview['avg_satisfaction']:.2f}/5.0. " if overview.get('avg_satisfaction') else "")
                  + ("Organization demonstrates strong employee engagement. Focus on maintaining current levels while addressing pockets of disengagement. " if overview['health_grade'] in ['A', 'B'] else "Moderate engagement suggests room for improvement. Targeted interventions recommended. " if overview['health_grade'] == 'C' else "Concerning engagement levels require urgent attention. Comprehensive wellness initiative needed."))
    }
    if comparison and not comparison.get('error'):
        report['step2_comparison'] = {
            'title': '2. Department/Level Comparison',
            'question': 'Which groups need attention?',
            'finding': f"Gap: {comparison.get('highest_dept', 'N/A')} ({comparison.get('highest_dept_score', 0):.2f}) vs {comparison.get('lowest_dept', 'N/A')} ({comparison.get('lowest_dept_score', 0):.2f})",
            'detail': (f"Cross-group analysis reveals {comparison.get('dept_gap', 0):.2f} point gap between highest ({comparison.get('highest_dept', 'N/A')}) and lowest ({comparison.get('lowest_dept', 'N/A')}) performing departments. "
                      + (f"Departments below average: {', '.join([d['dept'] for d in comparison.get('dept_stats', []) if d.get('status') == 'Below Avg'][:3])}. " if comparison.get('dept_stats') else "")
                      + ("Significant disparity indicates targeted intervention needed for underperforming groups. " if comparison.get('dept_gap', 0) > 0.5 else "Relatively consistent engagement across groups. "))
        }
    else:
        report['step2_comparison'] = {'title': '2. Group Comparison', 'question': 'Which groups need attention?', 'finding': 'Not configured', 'detail': 'Add department/level columns for group analysis.'}
    if correlation:
        top = correlation.get('top_driver', {})
        report['step3_correlation'] = {
            'title': '3. Engagement Drivers',
            'question': 'What drives engagement?',
            'finding': f"Top driver: {top.get('factor', 'N/A')} (r={top.get('correlation', 0):.3f}, {top.get('priority', 'N/A')} priority)",
            'detail': (f"Correlation analysis identifies {correlation['n_significant']} significant engagement drivers. "
                      f"'{top.get('factor', 'N/A')}' shows strongest relationship (r={top.get('correlation', 0):.3f}), meaning improvements in this area will have greatest impact on engagement. "
                      f"Current average score: {top.get('avg_score', 0):.2f}/5.0. "
                      "Investment prioritization: Address high-correlation, low-score factors first for maximum engagement lift.")
        }
    report['step4_burnout'] = {
        'title': '4. Burnout Risk Identification',
        'question': 'Who is at burnout risk?',
        'finding': f"Burnout rate: {burnout['burnout_rate']:.1f}% ({burnout['n_at_risk']} employees)",
        'detail': (f"Risk assessment identifies {burnout['n_at_risk']} employees ({burnout['burnout_rate']:.1f}%) at elevated burnout risk. "
                  f"Risk distribution: High risk {burnout['segments'][0]['pct']:.1f}% (immediate intervention), Medium risk {burnout['segments'][1]['pct']:.1f}% (monitor closely), Low risk {burnout['segments'][2]['pct']:.1f}%. "
                  + ("Critical burnout levels require immediate wellness intervention. " if burnout['burnout_rate'] > 20 else "Moderate burnout risk—proactive prevention recommended. " if burnout['burnout_rate'] > 10 else "Healthy burnout levels. Maintain supportive environment."))
    }
    if simulation:
        best = simulation.get('best_scenario', {})
        report['step5_simulation'] = {
            'title': '5. Improvement Impact Simulation',
            'question': 'What if we improve the work environment?',
            'finding': f"Best: {best.get('name', 'N/A')} → +{best.get('engagement_gain', 0):.0f}% engagement, +{best.get('productivity_gain', 0):.0f}% productivity",
            'detail': (f"Simulation models engagement interventions against current baseline of {simulation['current_engagement']:.2f}. "
                      f"'{best.get('name', 'N/A')}' offers highest impact: +{best.get('engagement_gain', 0):.1f}% engagement lifting score to {best.get('new_engagement', 0):.2f}, with projected +{best.get('productivity_gain', 0):.0f}% productivity gain. "
                      "Implementation: Start with recommended intervention, measure impact after 90 days, then iterate based on results.")
        }
    return report

def generate_insights(overview, comparison, correlation, burnout, simulation):
    insights = []
    if overview['avg_engagement'] >= 4.0:
        insights.append({'title': 'Strong Engagement', 'description': f"Score of {overview['avg_engagement']:.2f} indicates healthy organization.", 'status': 'positive'})
    elif overview['avg_engagement'] < 3.0:
        insights.append({'title': 'Low Engagement Alert', 'description': f"Score of {overview['avg_engagement']:.2f} requires urgent attention.", 'status': 'warning'})
    if overview['disengaged_pct'] > 20:
        insights.append({'title': 'High Disengagement', 'description': f"{overview['disengaged_pct']:.0f}% disengaged employees need intervention.", 'status': 'warning'})
    if burnout['burnout_rate'] > 15:
        insights.append({'title': 'Burnout Risk', 'description': f"{burnout['burnout_rate']:.1f}% at burnout risk. Implement wellness programs.", 'status': 'warning'})
    if correlation and correlation.get('top_driver'):
        insights.append({'title': f"Focus: {correlation['top_driver']['factor']}", 'description': f"Strongest engagement driver (r={correlation['top_driver']['correlation']:.3f}).", 'status': 'positive'})
    return insights

@router.post("/org-health")
async def analyze_org_health(request: OrgHealthRequest):
    try:
        df = pd.DataFrame(request.data)
        if len(df) < 20: raise HTTPException(status_code=400, detail="Need at least 20 employees")
        results, visualizations = {}, {}
        overview = analyze_overview(df, request.engagement_col, request.satisfaction_col)
        results['overview'] = overview
        visualizations['overview_chart'] = create_overview_chart(overview, df, request.engagement_col)
        comparison = analyze_comparison(df, request.engagement_col, request.satisfaction_col, request.dept_col, request.level_col)
        results['comparison'] = comparison
        if not comparison.get('error'): visualizations['comparison_chart'] = create_comparison_chart(comparison, request.dept_col, request.level_col)
        correlation = analyze_correlation(df, request.engagement_col, request.env_cols)
        results['correlation'] = correlation
        visualizations['correlation_chart'] = create_correlation_chart(correlation, df, request.engagement_col)
        burnout = analyze_burnout(df, request.engagement_col, request.env_cols)
        results['burnout'] = burnout
        visualizations['burnout_chart'] = create_burnout_chart(burnout, df, request.engagement_col)
        simulation = simulate_improvement(df, request.engagement_col, correlation, overview)
        results['simulation'] = simulation
        visualizations['simulation_chart'] = create_simulation_chart(simulation)
        report = generate_report(overview, comparison, correlation, burnout, simulation)
        insights = generate_insights(overview, comparison, correlation, burnout, simulation)
        summary = {'n_employees': overview['n_employees'], 'avg_engagement': overview['avg_engagement'], 'burnout_rate': burnout['burnout_rate'], 'top_driver': correlation.get('top_driver', {}).get('factor')}
        return {'success': True, 'results': results, 'visualizations': visualizations, 'report': report, 'key_insights': insights, 'summary': summary}
    except HTTPException: raise
    except Exception as e: raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
