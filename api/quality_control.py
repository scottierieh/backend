"""
Quality Control & Defect Cause Analysis API
5-step framework for defect root cause analysis
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

router = APIRouter()

class QualityRequest(BaseModel):
    data: List[Dict[str, Any]]
    defect_col: str
    env_cols: List[str]
    line_col: Optional[str] = None
    time_col: Optional[str] = None

def _to_native(obj):
    if obj is None: return None
    if isinstance(obj, (np.integer,)): return int(obj)
    if isinstance(obj, (np.floating,)): return float(obj) if not np.isnan(obj) else None
    if isinstance(obj, (np.bool_,)): return bool(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
    return obj

def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64


# Step 1: Defect Overview
def analyze_overview(df: pd.DataFrame, defect_col: str) -> Dict:
    defect = df[defect_col].apply(lambda x: 1 if str(x).lower() in ['1', 'yes', 'true', 'y'] else 0)
    
    n_records = len(df)
    n_defects = int(defect.sum())
    n_good = n_records - n_defects
    defect_rate = n_defects / n_records * 100 if n_records > 0 else 0
    yield_rate = 100 - defect_rate
    
    result = {
        'n_records': n_records,
        'n_defects': n_defects,
        'n_good': n_good,
        'defect_rate': _to_native(defect_rate),
        'yield_rate': _to_native(yield_rate)
    }
    
    # Check for defect type column
    if 'defect_type' in df.columns:
        defect_types = df[df[defect_col].apply(lambda x: str(x).lower() in ['1', 'yes', 'true', 'y'])]['defect_type'].value_counts().to_dict()
        result['defect_types'] = {str(k): int(v) for k, v in defect_types.items() if pd.notna(k)}
        result['top_defect_type'] = max(result['defect_types'], key=result['defect_types'].get) if result['defect_types'] else None
    
    return result


def create_overview_chart(overview: Dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Chart 1: Good vs Defect
    labels = ['Good', 'Defect']
    sizes = [overview['n_good'], overview['n_defects']]
    colors = ['#10b981', '#ef4444']
    axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[0].set_title('Production Quality', fontsize=11, fontweight='bold')
    
    # Chart 2: Defect types or rate bar
    if overview.get('defect_types'):
        types = list(overview['defect_types'].keys())
        counts = list(overview['defect_types'].values())
        axes[1].barh(types, counts, color='#ef4444', alpha=0.8, edgecolor='black')
        axes[1].set_xlabel('Count')
        axes[1].set_title('Defect by Type', fontsize=11, fontweight='bold')
    else:
        axes[1].bar(['Yield', 'Defect Rate'], [overview['yield_rate'], overview['defect_rate']], 
                   color=['#10b981', '#ef4444'], alpha=0.8, edgecolor='black')
        axes[1].set_ylabel('%')
        axes[1].set_title('Yield vs Defect Rate', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# Step 2: Line/Time Comparison
def analyze_comparison(df: pd.DataFrame, defect_col: str, line_col: Optional[str], time_col: Optional[str]) -> Dict:
    defect = df[defect_col].apply(lambda x: 1 if str(x).lower() in ['1', 'yes', 'true', 'y'] else 0)
    
    result = {}
    
    # Line comparison
    if line_col and line_col in df.columns:
        line_rates = {}
        for line in df[line_col].unique():
            line_data = defect[df[line_col] == line]
            if len(line_data) > 0:
                line_rates[str(line)] = {
                    'total': int(len(line_data)),
                    'defects': int(line_data.sum()),
                    'rate': _to_native(line_data.mean() * 100)
                }
        
        result['line_rates'] = line_rates
        result['avg_rate'] = _to_native(defect.mean() * 100)
        
        if line_rates:
            worst = max(line_rates, key=lambda x: line_rates[x]['rate'])
            best = min(line_rates, key=lambda x: line_rates[x]['rate'])
            result['worst_line'] = worst
            result['worst_line_rate'] = line_rates[worst]['rate']
            result['best_line'] = best
            result['best_line_rate'] = line_rates[best]['rate']
    
    # Time comparison
    if time_col and time_col in df.columns:
        time_rates = {}
        for period in df[time_col].unique():
            period_data = defect[df[time_col] == period]
            if len(period_data) > 0:
                time_rates[str(period)] = {
                    'total': int(len(period_data)),
                    'defects': int(period_data.sum()),
                    'rate': _to_native(period_data.mean() * 100)
                }
        
        result['time_rates'] = time_rates
        if time_rates:
            worst_time = max(time_rates, key=lambda x: time_rates[x]['rate'])
            result['worst_time'] = worst_time
            result['worst_time_rate'] = time_rates[worst_time]['rate']
    
    if not result:
        result['error'] = 'No line or time column provided'
    
    return result


def create_comparison_chart(comparison: Dict, line_col: str, time_col: str) -> str:
    if comparison.get('error'):
        return ""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Chart 1: Line comparison
    if comparison.get('line_rates'):
        lines = list(comparison['line_rates'].keys())
        rates = [comparison['line_rates'][l]['rate'] for l in lines]
        avg = comparison.get('avg_rate', np.mean(rates))
        colors = ['#ef4444' if r > avg else '#10b981' for r in rates]
        
        axes[0].barh(lines, rates, color=colors, alpha=0.8, edgecolor='black')
        axes[0].axvline(x=avg, color='blue', linestyle='--', label=f'Avg: {avg:.1f}%')
        axes[0].set_xlabel('Defect Rate (%)')
        axes[0].set_title('Defect Rate by Line', fontsize=11, fontweight='bold')
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, 'No line data', ha='center', va='center', transform=axes[0].transAxes)
    
    # Chart 2: Time comparison
    if comparison.get('time_rates'):
        periods = list(comparison['time_rates'].keys())
        rates = [comparison['time_rates'][p]['rate'] for p in periods]
        
        axes[1].bar(periods, rates, color='#3b82f6', alpha=0.8, edgecolor='black')
        axes[1].set_ylabel('Defect Rate (%)')
        axes[1].set_title('Defect Rate by Time Period', fontsize=11, fontweight='bold')
    else:
        axes[1].text(0.5, 0.5, 'No time data', ha='center', va='center', transform=axes[1].transAxes)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# Step 3: Environment-Defect Correlation
def analyze_correlation(df: pd.DataFrame, defect_col: str, env_cols: List[str]) -> Dict:
    defect = df[defect_col].apply(lambda x: 1 if str(x).lower() in ['1', 'yes', 'true', 'y'] else 0)
    
    correlations = []
    for col in env_cols:
        if col not in df.columns:
            continue
        
        vals = pd.to_numeric(df[col], errors='coerce')
        valid_idx = vals.notna() & defect.notna()
        
        if valid_idx.sum() < 10:
            continue
        
        corr, p_val = stats.pointbiserialr(defect[valid_idx], vals[valid_idx])
        
        correlations.append({
            'factor': col,
            'correlation': _to_native(corr),
            'p_value': _to_native(p_val),
            'significant': bool(p_val < 0.05),
            'direction': 'increases defects' if corr > 0 else 'decreases defects'
        })
    
    correlations = sorted(correlations, key=lambda x: abs(x.get('correlation') or 0), reverse=True)
    
    positive = [c for c in correlations if (c.get('correlation') or 0) > 0]
    negative = [c for c in correlations if (c.get('correlation') or 0) < 0]
    
    return {
        'correlations': correlations,
        'n_significant': sum(1 for c in correlations if c['significant']),
        'strongest_positive': positive[0] if positive else None,
        'strongest_negative': negative[0] if negative else None
    }


def create_correlation_chart(corr_data: Dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    corrs = corr_data['correlations'][:10]
    
    # Chart 1: Correlation bars
    factors = [c['factor'][:15] for c in corrs]
    values = [c['correlation'] for c in corrs]
    colors = ['#ef4444' if v > 0 else '#10b981' for v in values]
    
    axes[0].barh(factors, values, color=colors, alpha=0.8, edgecolor='black')
    axes[0].axvline(x=0, color='black', linewidth=0.5)
    axes[0].set_xlabel('Correlation with Defect')
    axes[0].set_title('Factor-Defect Correlation', fontsize=11, fontweight='bold')
    
    # Chart 2: Significance scatter
    abs_corr = [abs(c['correlation']) for c in corrs]
    sig = [1 if c['significant'] else 0.3 for c in corrs]
    colors2 = ['#3b82f6' if c['significant'] else '#d1d5db' for c in corrs]
    
    axes[1].scatter(abs_corr, range(len(corrs)), s=[s*200 for s in sig], c=colors2, alpha=0.7)
    axes[1].set_yticks(range(len(corrs)))
    axes[1].set_yticklabels(factors)
    axes[1].set_xlabel('|Correlation|')
    axes[1].set_title('Significance (blue = p<0.05)', fontsize=11, fontweight='bold')
    axes[1].axvline(x=0.2, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# Step 4: Root Cause Analysis (Logistic Regression)
def analyze_root_cause(df: pd.DataFrame, defect_col: str, env_cols: List[str]) -> Dict:
    defect = df[defect_col].apply(lambda x: 1 if str(x).lower() in ['1', 'yes', 'true', 'y'] else 0)
    
    valid_cols = [c for c in env_cols if c in df.columns]
    X = df[valid_cols].apply(pd.to_numeric, errors='coerce')
    
    valid_idx = X.notna().all(axis=1) & defect.notna()
    X = X[valid_idx]
    y = defect[valid_idx]
    
    if len(X) < 30 or y.sum() < 5 or (1-y).sum() < 5:
        return {'error': 'Insufficient data for root cause analysis'}
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_scaled, y)
    
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
    accuracy = cv_scores.mean()
    
    coefs = model.coef_[0]
    importance = np.abs(coefs) / np.abs(coefs).sum()
    
    causes = []
    for i, col in enumerate(valid_cols):
        causes.append({
            'factor': col,
            'coefficient': _to_native(coefs[i]),
            'importance': _to_native(importance[i]),
            'effect': 'increases defects' if coefs[i] > 0 else 'decreases defects',
            'odds_ratio': _to_native(np.exp(coefs[i]))
        })
    
    causes = sorted(causes, key=lambda x: abs(x.get('coefficient') or 0), reverse=True)
    
    return {
        'causes': causes,
        'top_cause': causes[0] if causes else None,
        'model_accuracy': _to_native(accuracy),
        'n_significant': sum(1 for c in causes if abs(c['coefficient']) > 0.1)
    }


def create_root_cause_chart(root_data: Dict) -> str:
    if root_data.get('error'):
        return ""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    causes = root_data['causes'][:8]
    
    # Chart 1: Coefficients
    names = [c['factor'][:15] for c in causes]
    coefs = [c['coefficient'] for c in causes]
    colors = ['#ef4444' if c > 0 else '#10b981' for c in coefs]
    
    axes[0].barh(names, coefs, color=colors, alpha=0.8, edgecolor='black')
    axes[0].axvline(x=0, color='black', linewidth=0.5)
    axes[0].set_xlabel('Coefficient')
    axes[0].set_title('Root Cause Coefficients', fontsize=11, fontweight='bold')
    
    # Chart 2: Importance
    importance = [c['importance'] * 100 for c in causes]
    axes[1].barh(names, importance, color='#3b82f6', alpha=0.8, edgecolor='black')
    axes[1].set_xlabel('Relative Importance (%)')
    axes[1].set_title('Factor Importance', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# Step 5: Yield Improvement Simulation
def simulate_improvement(df: pd.DataFrame, defect_col: str, root_data: Dict, overview: Dict) -> Dict:
    if root_data.get('error'):
        return {'error': 'Cannot simulate without root cause analysis'}
    
    current_defect_rate = overview['defect_rate']
    current_yield = overview['yield_rate']
    
    scenarios = []
    
    # Scenario 1: Fix top cause
    top_cause = root_data.get('top_cause')
    if top_cause:
        reduction = min(abs(top_cause['coefficient']) * 8, 25)
        new_defect = max(current_defect_rate * (1 - reduction/100), 0)
        new_yield = 100 - new_defect
        scenarios.append({
            'name': f"Optimize {top_cause['factor'][:15]}",
            'description': f"Control {top_cause['factor']} to optimal range",
            'focus_factor': top_cause['factor'],
            'defect_reduction': _to_native(reduction),
            'new_defect_rate': _to_native(new_defect),
            'yield_gain': _to_native(new_yield - current_yield),
            'new_yield': _to_native(new_yield)
        })
    
    # Scenario 2: Multi-factor improvement
    sig_causes = [c for c in root_data['causes'] if abs(c['coefficient']) > 0.1][:3]
    if sig_causes:
        total_impact = sum(abs(c['coefficient']) for c in sig_causes)
        reduction = min(total_impact * 6, 35)
        new_defect = max(current_defect_rate * (1 - reduction/100), 0)
        new_yield = 100 - new_defect
        scenarios.append({
            'name': 'Multi-Factor Improvement',
            'description': f"Optimize top {len(sig_causes)} factors together",
            'focus_factor': 'Multiple',
            'defect_reduction': _to_native(reduction),
            'new_defect_rate': _to_native(new_defect),
            'yield_gain': _to_native(new_yield - current_yield),
            'new_yield': _to_native(new_yield)
        })
    
    # Scenario 3: SPC implementation
    reduction = 15
    new_defect = current_defect_rate * 0.85
    new_yield = 100 - new_defect
    scenarios.append({
        'name': 'SPC Implementation',
        'description': 'Statistical process control monitoring',
        'focus_factor': 'Process Control',
        'defect_reduction': _to_native(reduction),
        'new_defect_rate': _to_native(new_defect),
        'yield_gain': _to_native(new_yield - current_yield),
        'new_yield': _to_native(new_yield)
    })
    
    best = max(scenarios, key=lambda x: x['yield_gain']) if scenarios else None
    
    recommendations = []
    if top_cause:
        if top_cause['coefficient'] > 0:
            recommendations.append(f"Reduce/control {top_cause['factor']} - main defect driver")
        else:
            recommendations.append(f"Increase {top_cause['factor']} - reduces defects")
    recommendations.append("Implement real-time monitoring for critical factors")
    recommendations.append("Establish control limits based on optimal factor ranges")
    
    return {
        'current_defect_rate': _to_native(current_defect_rate),
        'current_yield': _to_native(current_yield),
        'scenarios': scenarios,
        'best_scenario': best,
        'recommendations': recommendations
    }


def create_simulation_chart(simulation: Dict) -> str:
    if simulation.get('error'):
        return ""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    scenarios = simulation['scenarios']
    
    # Chart 1: Yield comparison
    names = ['Current'] + [s['name'][:15] for s in scenarios]
    yields = [simulation['current_yield']] + [s['new_yield'] for s in scenarios]
    colors = ['#94a3b8'] + ['#10b981'] * len(scenarios)
    
    axes[0].bar(names, yields, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_ylabel('Yield (%)')
    axes[0].set_title('Projected Yield', fontsize=11, fontweight='bold')
    axes[0].set_xticklabels(names, rotation=45, ha='right')
    axes[0].set_ylim(min(yields) - 5, 100)
    
    # Chart 2: Yield gain
    names2 = [s['name'][:15] for s in scenarios]
    gains = [s['yield_gain'] for s in scenarios]
    
    axes[1].barh(names2, gains, color='#10b981', alpha=0.8, edgecolor='black')
    axes[1].set_xlabel('Yield Gain (%)')
    axes[1].set_title('Improvement Potential', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# Report Generation
def generate_report(overview: Dict, comparison: Dict, correlation: Dict, root_cause: Dict, simulation: Dict) -> Dict:
    report = {}
    
    report['step1_overview'] = {
        'title': '1. Defect Overview',
        'question': 'What is our current defect rate?',
        'finding': f"Defect rate: {overview['defect_rate']:.1f}%, Yield: {overview['yield_rate']:.1f}%",
        'detail': f"Analysis of {overview['n_records']:,} production records shows {overview['n_defects']:,} defects ({overview['defect_rate']:.1f}% defect rate). "
                 f"Current yield stands at {overview['yield_rate']:.1f}%. "
                 + (f"Most common defect type is '{overview.get('top_defect_type', 'N/A')}'." if overview.get('top_defect_type') else "")
    }
    
    if comparison and not comparison.get('error'):
        report['step2_comparison'] = {
            'title': '2. Line/Time Comparison',
            'question': 'Which lines/shifts have the worst defect rates?',
            'finding': f"Worst: {comparison.get('worst_line', 'N/A')} ({comparison.get('worst_line_rate', 0):.1f}%), Best: {comparison.get('best_line', 'N/A')} ({comparison.get('best_line_rate', 0):.1f}%)",
            'detail': f"Line comparison shows significant variation in defect rates. "
                     f"'{comparison.get('worst_line', 'N/A')}' has the highest rate at {comparison.get('worst_line_rate', 0):.1f}%, "
                     f"while '{comparison.get('best_line', 'N/A')}' performs best at {comparison.get('best_line_rate', 0):.1f}%. "
                     + (f"Time analysis shows '{comparison.get('worst_time', 'N/A')}' as the most problematic period." if comparison.get('worst_time') else "")
        }
    else:
        report['step2_comparison'] = {'title': '2. Comparison', 'question': 'Line comparison', 'finding': 'Requires line/time columns', 'detail': ''}
    
    if correlation:
        strongest = correlation.get('strongest_positive') or correlation.get('strongest_negative') or {}
        report['step3_correlation'] = {
            'title': '3. Environment-Defect Correlation',
            'question': 'What factors correlate with defects?',
            'finding': f"Strongest factor: {strongest.get('factor', 'N/A')} (r={strongest.get('correlation', 0):.3f})",
            'detail': f"Correlation analysis found {correlation['n_significant']} statistically significant relationships. "
                     + (f"'{correlation['strongest_positive']['factor']}' positively correlates with defects (r={correlation['strongest_positive']['correlation']:.3f}), meaning higher values increase defect risk. " if correlation.get('strongest_positive') else "")
                     + (f"'{correlation['strongest_negative']['factor']}' negatively correlates (r={correlation['strongest_negative']['correlation']:.3f}), meaning higher values reduce defects." if correlation.get('strongest_negative') else "")
        }
    else:
        report['step3_correlation'] = {'title': '3. Correlation', 'question': 'Factor correlation', 'finding': 'Analysis not available', 'detail': ''}
    
    if root_cause and not root_cause.get('error'):
        top = root_cause.get('top_cause', {})
        report['step4_root_cause'] = {
            'title': '4. Root Cause Analysis',
            'question': 'What are the root causes of defects?',
            'finding': f"Top cause: {top.get('factor', 'N/A')} (importance: {top.get('importance', 0)*100:.1f}%)",
            'detail': f"Logistic regression model ({root_cause['model_accuracy']*100:.1f}% accuracy) identifies root causes. "
                     f"'{top.get('factor', 'N/A')}' is the primary driver with {top.get('importance', 0)*100:.1f}% importance. "
                     f"It {top.get('effect', 'affects defects')} with coefficient {top.get('coefficient', 0):.3f}."
        }
    else:
        report['step4_root_cause'] = {'title': '4. Root Cause', 'question': 'Root cause', 'finding': root_cause.get('error', 'Analysis not available'), 'detail': ''}
    
    if simulation and not simulation.get('error'):
        best = simulation.get('best_scenario', {})
        report['step5_simulation'] = {
            'title': '5. Yield Improvement Simulation',
            'question': 'How much can we improve yield?',
            'finding': f"Best scenario: +{best.get('yield_gain', 0):.1f}% yield ({best.get('name', 'N/A')})",
            'detail': f"Current yield is {simulation['current_yield']:.1f}%. "
                     f"'{best.get('name', 'N/A')}' scenario could improve yield to {best.get('new_yield', 0):.1f}%, "
                     f"reducing defects by {best.get('defect_reduction', 0):.0f}%."
        }
    else:
        report['step5_simulation'] = {'title': '5. Simulation', 'question': 'Improvement potential', 'finding': simulation.get('error', 'Simulation not available'), 'detail': ''}
    
    return report


def generate_insights(overview: Dict, comparison: Dict, correlation: Dict, root_cause: Dict, simulation: Dict) -> List[Dict]:
    insights = []
    
    if overview['defect_rate'] > 10:
        insights.append({'title': 'High Defect Rate', 'description': f"Defect rate of {overview['defect_rate']:.1f}% exceeds 10% threshold.", 'status': 'warning'})
    elif overview['defect_rate'] < 5:
        insights.append({'title': 'Good Quality', 'description': f"Defect rate of {overview['defect_rate']:.1f}% is under control.", 'status': 'positive'})
    
    if comparison and not comparison.get('error'):
        if comparison.get('worst_line_rate', 0) > comparison.get('avg_rate', 0) * 1.5:
            insights.append({'title': 'Problem Line', 'description': f"'{comparison['worst_line']}' has significantly higher defects.", 'status': 'warning'})
    
    if root_cause and not root_cause.get('error'):
        top = root_cause.get('top_cause', {})
        if top:
            insights.append({'title': f"Focus: {top['factor']}", 'description': f"Primary defect driver with {top['importance']*100:.0f}% importance.", 'status': 'warning'})
    
    if simulation and not simulation.get('error'):
        best = simulation.get('best_scenario', {})
        if best and best.get('yield_gain', 0) > 2:
            insights.append({'title': 'Improvement Potential', 'description': f"Can improve yield by +{best['yield_gain']:.1f}%.", 'status': 'positive'})
    
    return insights


@router.post("/quality-control")
async def analyze_quality(request: QualityRequest):
    try:
        df = pd.DataFrame(request.data)
        if len(df) < 20:
            raise HTTPException(status_code=400, detail="Need at least 20 records")
        
        results, visualizations = {}, {}
        
        # Step 1: Overview
        overview = analyze_overview(df, request.defect_col)
        results['overview'] = overview
        visualizations['overview_chart'] = create_overview_chart(overview)
        
        # Step 2: Comparison
        comparison = analyze_comparison(df, request.defect_col, request.line_col, request.time_col)
        results['comparison'] = comparison
        if not comparison.get('error'):
            visualizations['comparison_chart'] = create_comparison_chart(comparison, request.line_col, request.time_col)
        
        # Step 3: Correlation
        correlation = analyze_correlation(df, request.defect_col, request.env_cols)
        results['correlation'] = correlation
        visualizations['correlation_chart'] = create_correlation_chart(correlation)
        
        # Step 4: Root Cause
        root_cause = analyze_root_cause(df, request.defect_col, request.env_cols)
        results['root_cause'] = root_cause
        if not root_cause.get('error'):
            visualizations['root_cause_chart'] = create_root_cause_chart(root_cause)
        
        # Step 5: Simulation
        simulation = simulate_improvement(df, request.defect_col, root_cause, overview)
        results['simulation'] = simulation
        if not simulation.get('error'):
            visualizations['simulation_chart'] = create_simulation_chart(simulation)
        
        report = generate_report(overview, comparison, correlation, root_cause, simulation)
        insights = generate_insights(overview, comparison, correlation, root_cause, simulation)
        
        summary = {
            'n_records': overview['n_records'],
            'defect_rate': overview['defect_rate'],
            'n_defects': overview['n_defects'],
            'worst_line': comparison.get('worst_line') if not comparison.get('error') else None,
            'top_cause': root_cause.get('top_cause', {}).get('factor') if not root_cause.get('error') else None,
            'potential_improvement': simulation.get('best_scenario', {}).get('yield_gain', 0) if not simulation.get('error') else 0
        }
        
        return {'success': True, 'results': results, 'visualizations': visualizations, 'report': report, 'key_insights': insights, 'summary': summary}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
