"""
Churn Risk Management API
5-step framework for customer retention optimization
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

class ChurnRequest(BaseModel):
    data: List[Dict[str, Any]]
    churn_col: str
    activity_cols: List[str]
    customer_col: Optional[str] = None
    segment_col: Optional[str] = None

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


# Step 1: Churn Overview
def analyze_overview(df: pd.DataFrame, churn_col: str, segment_col: Optional[str] = None) -> Dict:
    # Convert churn to binary
    churn = df[churn_col].apply(lambda x: 1 if str(x).lower() in ['1', 'yes', 'true', 'y'] else 0)
    
    n_customers = len(df)
    n_churned = int(churn.sum())
    n_retained = n_customers - n_churned
    churn_rate = n_churned / n_customers * 100 if n_customers > 0 else 0
    
    result = {
        'n_customers': n_customers,
        'n_churned': n_churned,
        'n_retained': n_retained,
        'churn_rate': _to_native(churn_rate),
        'retention_rate': _to_native(100 - churn_rate)
    }
    
    # Segment analysis
    if segment_col and segment_col in df.columns:
        segment_rates = {}
        for seg in df[segment_col].unique():
            seg_data = churn[df[segment_col] == seg]
            if len(seg_data) > 0:
                segment_rates[str(seg)] = _to_native(seg_data.mean() * 100)
        result['segment_rates'] = segment_rates
        result['highest_churn_segment'] = max(segment_rates, key=segment_rates.get) if segment_rates else None
        result['lowest_churn_segment'] = min(segment_rates, key=segment_rates.get) if segment_rates else None
    
    return result


def create_overview_chart(overview: Dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Chart 1: Churn vs Retained
    labels = ['Retained', 'Churned']
    sizes = [overview['n_retained'], overview['n_churned']]
    colors = ['#10b981', '#ef4444']
    axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[0].set_title('Customer Distribution', fontsize=11, fontweight='bold')
    
    # Chart 2: Segment rates (if available)
    if overview.get('segment_rates'):
        segments = list(overview['segment_rates'].keys())
        rates = list(overview['segment_rates'].values())
        colors = ['#ef4444' if r > overview['churn_rate'] else '#10b981' for r in rates]
        axes[1].barh(segments, rates, color=colors, alpha=0.8, edgecolor='black')
        axes[1].axvline(x=overview['churn_rate'], color='blue', linestyle='--', label=f"Avg: {overview['churn_rate']:.1f}%")
        axes[1].set_xlabel('Churn Rate (%)')
        axes[1].set_title('Churn by Segment', fontsize=11, fontweight='bold')
        axes[1].legend()
    else:
        # Simple bar chart
        axes[1].bar(['Churn Rate', 'Retention Rate'], [overview['churn_rate'], overview['retention_rate']], 
                   color=['#ef4444', '#10b981'], alpha=0.8, edgecolor='black')
        axes[1].set_ylabel('%')
        axes[1].set_title('Churn vs Retention', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# Step 2: Retained vs Churned Comparison
def analyze_comparison(df: pd.DataFrame, churn_col: str, activity_cols: List[str]) -> Dict:
    churn = df[churn_col].apply(lambda x: 1 if str(x).lower() in ['1', 'yes', 'true', 'y'] else 0)
    
    retained = df[churn == 0]
    churned = df[churn == 1]
    
    comparisons = []
    for col in activity_cols:
        if col not in df.columns:
            continue
        
        ret_vals = pd.to_numeric(retained[col], errors='coerce').dropna()
        churn_vals = pd.to_numeric(churned[col], errors='coerce').dropna()
        
        if len(ret_vals) < 5 or len(churn_vals) < 5:
            continue
        
        ret_mean = ret_vals.mean()
        churn_mean = churn_vals.mean()
        diff_pct = (churn_mean - ret_mean) / ret_mean * 100 if ret_mean != 0 else 0
        
        # T-test
        t_stat, p_val = stats.ttest_ind(ret_vals, churn_vals)
        
        comparisons.append({
            'metric': col,
            'retained_mean': _to_native(ret_mean),
            'retained_std': _to_native(ret_vals.std()),
            'churned_mean': _to_native(churn_mean),
            'churned_std': _to_native(churn_vals.std()),
            'diff_pct': _to_native(diff_pct),
            't_statistic': _to_native(t_stat),
            'p_value': _to_native(p_val),
            'significant': bool(p_val < 0.05)
        })
    
    comparisons = sorted(comparisons, key=lambda x: abs(x.get('diff_pct') or 0), reverse=True)
    
    return {
        'comparisons': comparisons,
        'n_metrics': len(comparisons),
        'n_significant': sum(1 for c in comparisons if c['significant']),
        'biggest_difference': comparisons[0] if comparisons else None
    }


def create_comparison_chart(comparison: Dict, df: pd.DataFrame, churn_col: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    comps = comparison['comparisons'][:6]  # Top 6
    
    # Chart 1: Mean comparison
    metrics = [c['metric'][:15] for c in comps]
    retained = [c['retained_mean'] for c in comps]
    churned = [c['churned_mean'] for c in comps]
    
    x = np.arange(len(metrics))
    width = 0.35
    axes[0].bar(x - width/2, retained, width, label='Retained', color='#10b981', alpha=0.8)
    axes[0].bar(x + width/2, churned, width, label='Churned', color='#ef4444', alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(metrics, rotation=45, ha='right')
    axes[0].set_ylabel('Mean Value')
    axes[0].set_title('Retained vs Churned', fontsize=11, fontweight='bold')
    axes[0].legend()
    
    # Chart 2: Difference percentage
    diffs = [c['diff_pct'] for c in comps]
    colors = ['#ef4444' if d > 0 else '#10b981' for d in diffs]
    axes[1].barh(metrics, diffs, color=colors, alpha=0.8, edgecolor='black')
    axes[1].axvline(x=0, color='black', linewidth=0.5)
    axes[1].set_xlabel('Difference %')
    axes[1].set_title('Churned vs Retained Difference', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# Step 3: Activity-Churn Correlation
def analyze_correlation(df: pd.DataFrame, churn_col: str, activity_cols: List[str]) -> Dict:
    churn = df[churn_col].apply(lambda x: 1 if str(x).lower() in ['1', 'yes', 'true', 'y'] else 0)
    
    correlations = []
    for col in activity_cols:
        if col not in df.columns:
            continue
        
        vals = pd.to_numeric(df[col], errors='coerce')
        valid_idx = vals.notna() & churn.notna()
        
        if valid_idx.sum() < 10:
            continue
        
        corr, p_val = stats.pointbiserialr(churn[valid_idx], vals[valid_idx])
        
        correlations.append({
            'metric': col,
            'correlation': _to_native(corr),
            'p_value': _to_native(p_val),
            'significant': bool(p_val < 0.05),
            'direction': 'increases churn' if corr > 0 else 'decreases churn',
            'strength': 'strong' if abs(corr) > 0.5 else 'moderate' if abs(corr) > 0.3 else 'weak'
        })
    
    correlations = sorted(correlations, key=lambda x: abs(x.get('correlation') or 0), reverse=True)
    
    positive = [c for c in correlations if (c.get('correlation') or 0) > 0]
    negative = [c for c in correlations if (c.get('correlation') or 0) < 0]
    
    return {
        'correlations': correlations,
        'n_correlations': len(correlations),
        'n_significant': sum(1 for c in correlations if c['significant']),
        'strongest_positive': positive[0] if positive else None,
        'strongest_negative': negative[0] if negative else None
    }


def create_correlation_chart(corr_data: Dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    corrs = corr_data['correlations'][:10]
    
    # Chart 1: Correlation bars
    metrics = [c['metric'][:15] for c in corrs]
    values = [c['correlation'] for c in corrs]
    colors = ['#ef4444' if v > 0 else '#10b981' for v in values]
    
    axes[0].barh(metrics, values, color=colors, alpha=0.8, edgecolor='black')
    axes[0].axvline(x=0, color='black', linewidth=0.5)
    axes[0].set_xlabel('Correlation with Churn')
    axes[0].set_title('Activity-Churn Correlation', fontsize=11, fontweight='bold')
    
    # Chart 2: Significance
    sig = [1 if c['significant'] else 0.3 for c in corrs]
    abs_corr = [abs(c['correlation']) for c in corrs]
    colors2 = ['#3b82f6' if c['significant'] else '#d1d5db' for c in corrs]
    
    axes[1].scatter(abs_corr, range(len(corrs)), s=[s*200 for s in sig], c=colors2, alpha=0.7)
    axes[1].set_yticks(range(len(corrs)))
    axes[1].set_yticklabels(metrics)
    axes[1].set_xlabel('|Correlation|')
    axes[1].set_title('Significance (blue = p<0.05)', fontsize=11, fontweight='bold')
    axes[1].axvline(x=0.3, color='red', linestyle='--', alpha=0.5, label='Moderate threshold')
    axes[1].legend()
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# Step 4: Churn Drivers (Logistic Regression)
def analyze_drivers(df: pd.DataFrame, churn_col: str, activity_cols: List[str]) -> Dict:
    churn = df[churn_col].apply(lambda x: 1 if str(x).lower() in ['1', 'yes', 'true', 'y'] else 0)
    
    # Prepare features
    valid_cols = [c for c in activity_cols if c in df.columns]
    X = df[valid_cols].apply(pd.to_numeric, errors='coerce')
    
    # Remove rows with missing values
    valid_idx = X.notna().all(axis=1) & churn.notna()
    X = X[valid_idx]
    y = churn[valid_idx]
    
    if len(X) < 30 or y.sum() < 5 or (1-y).sum() < 5:
        return {'error': 'Insufficient data for driver analysis'}
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Logistic Regression
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_scaled, y)
    
    # Cross-validation accuracy
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
    accuracy = cv_scores.mean()
    
    # Feature importance
    coefs = model.coef_[0]
    importance = np.abs(coefs) / np.abs(coefs).sum()
    
    drivers = []
    for i, col in enumerate(valid_cols):
        drivers.append({
            'driver': col,
            'coefficient': _to_native(coefs[i]),
            'importance': _to_native(importance[i]),
            'effect': 'increases churn' if coefs[i] > 0 else 'decreases churn',
            'odds_ratio': _to_native(np.exp(coefs[i]))
        })
    
    drivers = sorted(drivers, key=lambda x: abs(x.get('coefficient') or 0), reverse=True)
    
    return {
        'drivers': drivers,
        'top_driver': drivers[0] if drivers else None,
        'model_accuracy': _to_native(accuracy),
        'model_accuracy_std': _to_native(cv_scores.std()),
        'n_significant': sum(1 for d in drivers if abs(d['coefficient']) > 0.1),
        'intercept': _to_native(model.intercept_[0])
    }


def create_drivers_chart(drivers_data: Dict) -> str:
    if drivers_data.get('error'):
        return ""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    drivers = drivers_data['drivers'][:8]
    
    # Chart 1: Coefficients
    names = [d['driver'][:15] for d in drivers]
    coefs = [d['coefficient'] for d in drivers]
    colors = ['#ef4444' if c > 0 else '#10b981' for c in coefs]
    
    axes[0].barh(names, coefs, color=colors, alpha=0.8, edgecolor='black')
    axes[0].axvline(x=0, color='black', linewidth=0.5)
    axes[0].set_xlabel('Coefficient (Standardized)')
    axes[0].set_title('Churn Driver Coefficients', fontsize=11, fontweight='bold')
    
    # Chart 2: Importance
    importance = [d['importance'] * 100 for d in drivers]
    axes[1].barh(names, importance, color='#3b82f6', alpha=0.8, edgecolor='black')
    axes[1].set_xlabel('Relative Importance (%)')
    axes[1].set_title('Driver Importance', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# Step 5: Prevention Impact Simulation
def simulate_prevention(df: pd.DataFrame, churn_col: str, drivers_data: Dict, overview: Dict) -> Dict:
    if drivers_data.get('error'):
        return {'error': 'Cannot simulate without driver analysis'}
    
    current_rate = overview['churn_rate']
    n_customers = overview['n_customers']
    n_churned = overview['n_churned']
    
    scenarios = []
    
    # Scenario 1: Target top driver
    top_driver = drivers_data.get('top_driver')
    if top_driver:
        # Simulate 20% improvement in top driver
        reduction = min(abs(top_driver['coefficient']) * 10, 30)  # Cap at 30%
        new_rate = max(current_rate * (1 - reduction/100), 0)
        saved = int(n_churned * reduction / 100)
        scenarios.append({
            'name': f"Improve {top_driver['driver']}",
            'description': f"20% improvement in {top_driver['driver']}",
            'reduction': _to_native(reduction),
            'new_churn_rate': _to_native(new_rate),
            'customers_saved': saved
        })
    
    # Scenario 2: Target all significant drivers
    sig_drivers = [d for d in drivers_data['drivers'] if abs(d['coefficient']) > 0.1]
    if sig_drivers:
        total_impact = sum(abs(d['coefficient']) for d in sig_drivers[:3])
        reduction = min(total_impact * 8, 40)
        new_rate = max(current_rate * (1 - reduction/100), 0)
        saved = int(n_churned * reduction / 100)
        scenarios.append({
            'name': 'Multi-factor Intervention',
            'description': f"Address top {min(3, len(sig_drivers))} churn drivers",
            'reduction': _to_native(reduction),
            'new_churn_rate': _to_native(new_rate),
            'customers_saved': saved
        })
    
    # Scenario 3: Proactive retention program
    reduction = 15
    new_rate = current_rate * 0.85
    saved = int(n_churned * 0.15)
    scenarios.append({
        'name': 'Proactive Retention',
        'description': 'Early warning system + targeted outreach',
        'reduction': _to_native(reduction),
        'new_churn_rate': _to_native(new_rate),
        'customers_saved': saved
    })
    
    # Find best scenario
    best = max(scenarios, key=lambda x: x['reduction']) if scenarios else None
    
    # Recommendations
    recommendations = []
    if top_driver:
        if top_driver['coefficient'] > 0:
            recommendations.append(f"Focus on reducing {top_driver['driver']} as it increases churn risk")
        else:
            recommendations.append(f"Increase {top_driver['driver']} engagement as it reduces churn")
    recommendations.append("Implement early warning system for at-risk customers")
    recommendations.append("Create personalized retention offers based on customer segment")
    
    return {
        'current_churn_rate': _to_native(current_rate),
        'scenarios': scenarios,
        'best_scenario': best,
        'recommendations': recommendations,
        'potential_savings': {
            'customers': best['customers_saved'] if best else 0,
            'percentage': best['reduction'] if best else 0
        }
    }


def create_prevention_chart(prevention: Dict) -> str:
    if prevention.get('error'):
        return ""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    scenarios = prevention['scenarios']
    
    # Chart 1: Churn rate comparison
    names = ['Current'] + [s['name'][:15] for s in scenarios]
    rates = [prevention['current_churn_rate']] + [s['new_churn_rate'] for s in scenarios]
    colors = ['#ef4444'] + ['#10b981'] * len(scenarios)
    
    axes[0].bar(names, rates, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_ylabel('Churn Rate (%)')
    axes[0].set_title('Projected Churn Rate by Scenario', fontsize=11, fontweight='bold')
    axes[0].set_xticklabels(names, rotation=45, ha='right')
    
    # Chart 2: Customers saved
    names2 = [s['name'][:15] for s in scenarios]
    saved = [s['customers_saved'] for s in scenarios]
    
    axes[1].barh(names2, saved, color='#3b82f6', alpha=0.8, edgecolor='black')
    axes[1].set_xlabel('Customers Saved')
    axes[1].set_title('Prevention Impact', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# Report Generation
def generate_report(overview: Dict, comparison: Dict, correlation: Dict, drivers: Dict, prevention: Dict) -> Dict:
    report = {}
    
    report['step1_overview'] = {
        'title': '1. Churn Overview',
        'question': 'What is our current churn rate?',
        'finding': f"Churn rate is {overview['churn_rate']:.1f}% ({overview['n_churned']:,} of {overview['n_customers']:,} customers)",
        'detail': f"Analysis of {overview['n_customers']:,} customers shows {overview['n_churned']:,} have churned, representing a {overview['churn_rate']:.1f}% churn rate. "
                 f"This means {overview['n_retained']:,} customers ({overview['retention_rate']:.1f}%) have been retained. "
                 + (f"Segment analysis reveals highest churn in '{overview.get('highest_churn_segment', 'N/A')}' and lowest in '{overview.get('lowest_churn_segment', 'N/A')}'." if overview.get('segment_rates') else "")
    }
    
    if comparison and not comparison.get('error'):
        biggest = comparison.get('biggest_difference', {})
        report['step2_comparison'] = {
            'title': '2. Retained vs Churned Comparison',
            'question': 'How do retained and churned customers differ?',
            'finding': f"Biggest difference: {biggest.get('metric', 'N/A')} ({biggest.get('diff_pct', 0):.1f}% difference)",
            'detail': f"Comparison of {comparison['n_metrics']} metrics between retained and churned customers reveals {comparison['n_significant']} statistically significant differences. "
                     f"The most notable difference is in '{biggest.get('metric', 'N/A')}' where churned customers average {biggest.get('churned_mean', 0):.2f} vs {biggest.get('retained_mean', 0):.2f} for retained customers."
        }
    else:
        report['step2_comparison'] = {'title': '2. Comparison', 'question': 'How do they differ?', 'finding': 'Comparison not available', 'detail': ''}
    
    if correlation and not correlation.get('error'):
        strongest = correlation.get('strongest_positive') or correlation.get('strongest_negative') or {}
        report['step3_correlation'] = {
            'title': '3. Activity-Churn Correlation',
            'question': 'What activities relate to churn?',
            'finding': f"Strongest signal: {strongest.get('metric', 'N/A')} (r={strongest.get('correlation', 0):.3f})",
            'detail': f"Correlation analysis of {correlation['n_correlations']} activity metrics found {correlation['n_significant']} with significant relationships to churn. "
                     + (f"'{correlation['strongest_positive']['metric']}' shows positive correlation (r={correlation['strongest_positive']['correlation']:.3f}), indicating it increases churn risk. " if correlation.get('strongest_positive') else "")
                     + (f"'{correlation['strongest_negative']['metric']}' shows negative correlation (r={correlation['strongest_negative']['correlation']:.3f}), indicating it reduces churn." if correlation.get('strongest_negative') else "")
        }
    else:
        report['step3_correlation'] = {'title': '3. Correlation', 'question': 'What relates to churn?', 'finding': 'Correlation not available', 'detail': ''}
    
    if drivers and not drivers.get('error'):
        top = drivers.get('top_driver', {})
        report['step4_drivers'] = {
            'title': '4. Churn Drivers',
            'question': 'What causes churn?',
            'finding': f"Top driver: {top.get('driver', 'N/A')} (importance: {top.get('importance', 0)*100:.1f}%)",
            'detail': f"Logistic regression model with {drivers['model_accuracy']*100:.1f}% accuracy identifies key churn drivers. "
                     f"'{top.get('driver', 'N/A')}' is the strongest predictor with coefficient {top.get('coefficient', 0):.3f}, meaning it {top.get('effect', 'affects churn')}. "
                     f"{drivers['n_significant']} factors show substantial impact on churn probability."
        }
    else:
        report['step4_drivers'] = {'title': '4. Drivers', 'question': 'What causes churn?', 'finding': drivers.get('error', 'Driver analysis not available'), 'detail': ''}
    
    if prevention and not prevention.get('error'):
        best = prevention.get('best_scenario', {})
        report['step5_prevention'] = {
            'title': '5. Prevention Impact',
            'question': 'What if we intervene?',
            'finding': f"Best scenario: {best.get('name', 'N/A')} can reduce churn by {best.get('reduction', 0):.1f}%",
            'detail': f"Simulation of prevention strategies shows potential to reduce churn from {prevention['current_churn_rate']:.1f}% to {best.get('new_churn_rate', 0):.1f}%. "
                     f"This would save approximately {best.get('customers_saved', 0):,} customers. "
                     f"Key recommendations: {'; '.join(prevention.get('recommendations', [])[:2])}"
        }
    else:
        report['step5_prevention'] = {'title': '5. Prevention', 'question': 'What if we intervene?', 'finding': prevention.get('error', 'Prevention simulation not available'), 'detail': ''}
    
    return report


def generate_insights(overview: Dict, comparison: Dict, correlation: Dict, drivers: Dict, prevention: Dict) -> List[Dict]:
    insights = []
    
    # Churn rate insight
    if overview['churn_rate'] > 20:
        insights.append({'title': 'High Churn Rate', 'description': f"Churn rate of {overview['churn_rate']:.1f}% exceeds typical benchmark of 20%.", 'status': 'warning'})
    elif overview['churn_rate'] < 10:
        insights.append({'title': 'Healthy Retention', 'description': f"Churn rate of {overview['churn_rate']:.1f}% indicates strong customer retention.", 'status': 'positive'})
    
    # Comparison insight
    if comparison and not comparison.get('error') and comparison['n_significant'] > 0:
        insights.append({'title': 'Clear Differentiation', 'description': f"{comparison['n_significant']} metrics significantly differ between retained and churned customers.", 'status': 'neutral'})
    
    # Driver insight
    if drivers and not drivers.get('error'):
        top = drivers.get('top_driver', {})
        if top:
            insights.append({'title': f"Focus on {top['driver']}", 'description': f"Top churn driver with {top['importance']*100:.0f}% importance.", 'status': 'warning'})
    
    # Prevention insight
    if prevention and not prevention.get('error'):
        best = prevention.get('best_scenario', {})
        if best and best.get('reduction', 0) > 15:
            insights.append({'title': 'High Prevention Potential', 'description': f"Can reduce churn by {best['reduction']:.0f}% with targeted intervention.", 'status': 'positive'})
    
    return insights


@router.post("/churn-risk")
async def analyze_churn_risk(request: ChurnRequest):
    try:
        df = pd.DataFrame(request.data)
        if len(df) < 20:
            raise HTTPException(status_code=400, detail="Need at least 20 records")
        
        results, visualizations = {}, {}
        
        # Step 1: Overview
        overview = analyze_overview(df, request.churn_col, request.segment_col)
        results['overview'] = overview
        visualizations['overview_chart'] = create_overview_chart(overview)
        
        # Step 2: Comparison
        comparison = analyze_comparison(df, request.churn_col, request.activity_cols)
        results['comparison'] = comparison
        visualizations['comparison_chart'] = create_comparison_chart(comparison, df, request.churn_col)
        
        # Step 3: Correlation
        correlation = analyze_correlation(df, request.churn_col, request.activity_cols)
        results['correlation'] = correlation
        visualizations['correlation_chart'] = create_correlation_chart(correlation)
        
        # Step 4: Drivers
        drivers = analyze_drivers(df, request.churn_col, request.activity_cols)
        results['drivers'] = drivers
        if not drivers.get('error'):
            visualizations['drivers_chart'] = create_drivers_chart(drivers)
        
        # Step 5: Prevention
        prevention = simulate_prevention(df, request.churn_col, drivers, overview)
        results['prevention'] = prevention
        if not prevention.get('error'):
            visualizations['prevention_chart'] = create_prevention_chart(prevention)
        
        report = generate_report(overview, comparison, correlation, drivers, prevention)
        insights = generate_insights(overview, comparison, correlation, drivers, prevention)
        
        summary = {
            'n_customers': overview['n_customers'],
            'churn_rate': overview['churn_rate'],
            'n_churned': overview['n_churned'],
            'n_retained': overview['n_retained'],
            'top_driver': drivers.get('top_driver', {}).get('driver') if not drivers.get('error') else None,
            'prevention_impact': prevention.get('best_scenario', {}).get('reduction', 0) if not prevention.get('error') else 0
        }
        
        return {'success': True, 'results': results, 'visualizations': visualizations, 'report': report, 'key_insights': insights, 'summary': summary}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
