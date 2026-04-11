"""
Risk & Safety Analysis API
5-step framework for accident prevention
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

class RiskRequest(BaseModel):
    data: List[Dict[str, Any]]
    incident_col: str
    safety_cols: List[str]
    location_col: Optional[str] = None
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


# Step 1: Incident Overview
def analyze_overview(df: pd.DataFrame, incident_col: str) -> Dict:
    incident = df[incident_col].apply(lambda x: 1 if str(x).lower() in ['1', 'yes', 'true', 'y'] else 0)
    
    n_records = len(df)
    n_incidents = int(incident.sum())
    n_safe = n_records - n_incidents
    incident_rate = n_incidents / n_records * 100 if n_records > 0 else 0
    safe_rate = 100 - incident_rate
    
    result = {
        'n_records': n_records,
        'n_incidents': n_incidents,
        'n_safe': n_safe,
        'incident_rate': _to_native(incident_rate),
        'safe_rate': _to_native(safe_rate)
    }
    
    # Check for incident type column
    if 'incident_type' in df.columns:
        incident_types = df[df[incident_col].apply(lambda x: str(x).lower() in ['1', 'yes', 'true', 'y'])]['incident_type'].value_counts().to_dict()
        result['incident_types'] = {str(k): int(v) for k, v in incident_types.items() if pd.notna(k)}
        result['top_incident_type'] = max(result['incident_types'], key=result['incident_types'].get) if result['incident_types'] else None
    
    # Check for severity column
    if 'severity' in df.columns:
        severity_dist = df[df[incident_col].apply(lambda x: str(x).lower() in ['1', 'yes', 'true', 'y'])]['severity'].value_counts().to_dict()
        result['severity_distribution'] = {str(k): int(v) for k, v in severity_dist.items() if pd.notna(k)}
    
    return result


def create_overview_chart(overview: Dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Chart 1: Safe vs Incident
    labels = ['Safe', 'Incident']
    sizes = [overview['n_safe'], overview['n_incidents']]
    colors = ['#10b981', '#ef4444']
    axes[0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    axes[0].set_title('Safety Overview', fontsize=11, fontweight='bold')
    
    # Chart 2: Incident types or severity
    if overview.get('incident_types'):
        types = list(overview['incident_types'].keys())
        counts = list(overview['incident_types'].values())
        axes[1].barh(types, counts, color='#ef4444', alpha=0.8, edgecolor='black')
        axes[1].set_xlabel('Count')
        axes[1].set_title('Incident by Type', fontsize=11, fontweight='bold')
    elif overview.get('severity_distribution'):
        sev = list(overview['severity_distribution'].keys())
        counts = list(overview['severity_distribution'].values())
        colors = ['#fbbf24' if 'Minor' in s else '#f97316' if 'Mod' in s else '#ef4444' for s in sev]
        axes[1].bar(sev, counts, color=colors, alpha=0.8, edgecolor='black')
        axes[1].set_ylabel('Count')
        axes[1].set_title('Incident by Severity', fontsize=11, fontweight='bold')
    else:
        axes[1].bar(['Safe Rate', 'Incident Rate'], [overview['safe_rate'], overview['incident_rate']], 
                   color=['#10b981', '#ef4444'], alpha=0.8, edgecolor='black')
        axes[1].set_ylabel('%')
        axes[1].set_title('Safety vs Incident Rate', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# Step 2: Location/Situation Comparison
def analyze_comparison(df: pd.DataFrame, incident_col: str, location_col: Optional[str], time_col: Optional[str]) -> Dict:
    incident = df[incident_col].apply(lambda x: 1 if str(x).lower() in ['1', 'yes', 'true', 'y'] else 0)
    
    result = {}
    
    # Location comparison
    if location_col and location_col in df.columns:
        location_rates = {}
        for loc in df[location_col].unique():
            loc_data = incident[df[location_col] == loc]
            if len(loc_data) > 0:
                location_rates[str(loc)] = {
                    'total': int(len(loc_data)),
                    'incidents': int(loc_data.sum()),
                    'rate': _to_native(loc_data.mean() * 100)
                }
        
        result['location_rates'] = location_rates
        result['avg_rate'] = _to_native(incident.mean() * 100)
        
        if location_rates:
            worst = max(location_rates, key=lambda x: location_rates[x]['rate'])
            safest = min(location_rates, key=lambda x: location_rates[x]['rate'])
            result['worst_location'] = worst
            result['worst_location_rate'] = location_rates[worst]['rate']
            result['safest_location'] = safest
            result['safest_location_rate'] = location_rates[safest]['rate']
    
    # Time/shift comparison
    if time_col and time_col in df.columns:
        time_rates = {}
        for period in df[time_col].unique():
            period_data = incident[df[time_col] == period]
            if len(period_data) > 0:
                time_rates[str(period)] = {
                    'total': int(len(period_data)),
                    'incidents': int(period_data.sum()),
                    'rate': _to_native(period_data.mean() * 100)
                }
        
        result['time_rates'] = time_rates
        if time_rates:
            worst_time = max(time_rates, key=lambda x: time_rates[x]['rate'])
            result['worst_time'] = worst_time
            result['worst_time_rate'] = time_rates[worst_time]['rate']
    
    if not result:
        result['error'] = 'No location or time column provided'
    
    return result


def create_comparison_chart(comparison: Dict) -> str:
    if comparison.get('error'):
        return ""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Chart 1: Location comparison
    if comparison.get('location_rates'):
        locs = list(comparison['location_rates'].keys())
        rates = [comparison['location_rates'][l]['rate'] for l in locs]
        avg = comparison.get('avg_rate', np.mean(rates))
        colors = ['#ef4444' if r > avg else '#10b981' for r in rates]
        
        axes[0].barh(locs, rates, color=colors, alpha=0.8, edgecolor='black')
        axes[0].axvline(x=avg, color='blue', linestyle='--', label=f'Avg: {avg:.1f}%')
        axes[0].set_xlabel('Incident Rate (%)')
        axes[0].set_title('Risk by Location', fontsize=11, fontweight='bold')
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, 'No location data', ha='center', va='center', transform=axes[0].transAxes)
    
    # Chart 2: Time comparison
    if comparison.get('time_rates'):
        periods = list(comparison['time_rates'].keys())
        rates = [comparison['time_rates'][p]['rate'] for p in periods]
        
        axes[1].bar(periods, rates, color='#f97316', alpha=0.8, edgecolor='black')
        axes[1].set_ylabel('Incident Rate (%)')
        axes[1].set_title('Risk by Time Period', fontsize=11, fontweight='bold')
    else:
        axes[1].text(0.5, 0.5, 'No time data', ha='center', va='center', transform=axes[1].transAxes)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# Step 3: Safety Factor Correlation
def analyze_correlation(df: pd.DataFrame, incident_col: str, safety_cols: List[str]) -> Dict:
    incident = df[incident_col].apply(lambda x: 1 if str(x).lower() in ['1', 'yes', 'true', 'y'] else 0)
    
    correlations = []
    for col in safety_cols:
        if col not in df.columns:
            continue
        
        vals = pd.to_numeric(df[col], errors='coerce')
        valid_idx = vals.notna() & incident.notna()
        
        if valid_idx.sum() < 10:
            continue
        
        corr, p_val = stats.pointbiserialr(incident[valid_idx], vals[valid_idx])
        
        correlations.append({
            'factor': col,
            'correlation': _to_native(corr),
            'p_value': _to_native(p_val),
            'significant': bool(p_val < 0.05),
            'effect': 'increases risk' if corr > 0 else 'reduces risk'
        })
    
    correlations = sorted(correlations, key=lambda x: abs(x.get('correlation') or 0), reverse=True)
    
    risk_factors = [c for c in correlations if (c.get('correlation') or 0) > 0]
    protection_factors = [c for c in correlations if (c.get('correlation') or 0) < 0]
    
    return {
        'correlations': correlations,
        'n_significant': sum(1 for c in correlations if c['significant']),
        'strongest_risk': risk_factors[0] if risk_factors else None,
        'strongest_protection': protection_factors[0] if protection_factors else None
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
    axes[0].set_xlabel('Correlation with Incidents')
    axes[0].set_title('Safety Factor Correlation', fontsize=11, fontweight='bold')
    
    # Chart 2: Risk vs Protection
    risk = [c for c in corrs if c['correlation'] > 0]
    protect = [c for c in corrs if c['correlation'] < 0]
    
    if risk or protect:
        labels = []
        vals = []
        cols = []
        for r in risk[:3]:
            labels.append(r['factor'][:12])
            vals.append(abs(r['correlation']))
            cols.append('#ef4444')
        for p in protect[:3]:
            labels.append(p['factor'][:12])
            vals.append(abs(p['correlation']))
            cols.append('#10b981')
        
        axes[1].barh(labels, vals, color=cols, alpha=0.8, edgecolor='black')
        axes[1].set_xlabel('|Correlation|')
        axes[1].set_title('Risk (red) vs Protection (green)', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# Step 4: Risk Patterns (Logistic Regression)
def analyze_patterns(df: pd.DataFrame, incident_col: str, safety_cols: List[str]) -> Dict:
    incident = df[incident_col].apply(lambda x: 1 if str(x).lower() in ['1', 'yes', 'true', 'y'] else 0)
    
    valid_cols = [c for c in safety_cols if c in df.columns]
    X = df[valid_cols].apply(pd.to_numeric, errors='coerce')
    
    valid_idx = X.notna().all(axis=1) & incident.notna()
    X = X[valid_idx]
    y = incident[valid_idx]
    
    if len(X) < 30 or y.sum() < 5 or (1-y).sum() < 5:
        return {'error': 'Insufficient data for pattern analysis'}
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_scaled, y)
    
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
    accuracy = cv_scores.mean()
    
    coefs = model.coef_[0]
    importance = np.abs(coefs) / np.abs(coefs).sum()
    
    indicators = []
    for i, col in enumerate(valid_cols):
        indicators.append({
            'factor': col,
            'coefficient': _to_native(coefs[i]),
            'importance': _to_native(importance[i]),
            'effect': 'increases risk' if coefs[i] > 0 else 'reduces risk',
            'odds_ratio': _to_native(np.exp(coefs[i]))
        })
    
    indicators = sorted(indicators, key=lambda x: abs(x.get('coefficient') or 0), reverse=True)
    
    # Risk thresholds
    thresholds = []
    for col in valid_cols[:3]:
        vals = pd.to_numeric(df[col], errors='coerce')
        mean_val = vals.mean()
        std_val = vals.std()
        
        # Check if high or low values correlate with risk
        corr = indicators[valid_cols.index(col)]['coefficient'] if col in valid_cols else 0
        
        if corr > 0:  # Higher values = higher risk
            threshold = mean_val + std_val
            direction = 'above'
        else:  # Lower values = higher risk
            threshold = mean_val - std_val
            direction = 'below'
        
        # Calculate risk multiplier
        if direction == 'above':
            high_risk = incident[vals > threshold].mean()
            low_risk = incident[vals <= threshold].mean()
        else:
            high_risk = incident[vals < threshold].mean()
            low_risk = incident[vals >= threshold].mean()
        
        multiplier = high_risk / low_risk if low_risk > 0 else 1
        
        thresholds.append({
            'factor': col,
            'threshold': _to_native(threshold),
            'direction': direction,
            'risk_multiplier': _to_native(multiplier)
        })
    
    return {
        'indicators': indicators,
        'top_indicator': indicators[0] if indicators else None,
        'model_accuracy': _to_native(accuracy),
        'n_risk_factors': sum(1 for i in indicators if i['coefficient'] > 0),
        'risk_thresholds': thresholds
    }


def create_patterns_chart(patterns: Dict) -> str:
    if patterns.get('error'):
        return ""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    indicators = patterns['indicators'][:8]
    
    # Chart 1: Coefficients
    names = [ind['factor'][:15] for ind in indicators]
    coefs = [ind['coefficient'] for ind in indicators]
    colors = ['#ef4444' if c > 0 else '#10b981' for c in coefs]
    
    axes[0].barh(names, coefs, color=colors, alpha=0.8, edgecolor='black')
    axes[0].axvline(x=0, color='black', linewidth=0.5)
    axes[0].set_xlabel('Coefficient')
    axes[0].set_title('Risk Pattern Coefficients', fontsize=11, fontweight='bold')
    
    # Chart 2: Importance
    importance = [ind['importance'] * 100 for ind in indicators]
    axes[1].barh(names, importance, color='#3b82f6', alpha=0.8, edgecolor='black')
    axes[1].set_xlabel('Relative Importance (%)')
    axes[1].set_title('Factor Importance', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# Step 5: Prevention Simulation
def simulate_prevention(df: pd.DataFrame, incident_col: str, patterns: Dict, overview: Dict) -> Dict:
    if patterns.get('error'):
        return {'error': 'Cannot simulate without pattern analysis'}
    
    current_rate = overview['incident_rate']
    n_incidents = overview['n_incidents']
    
    scenarios = []
    
    # Scenario 1: Address top risk factor
    top = patterns.get('top_indicator')
    if top and top['coefficient'] > 0:
        reduction = min(abs(top['coefficient']) * 15, 30)
        new_rate = max(current_rate * (1 - reduction/100), 0)
        prevented = int(n_incidents * reduction / 100)
        scenarios.append({
            'name': f"Control {top['factor'][:15]}",
            'description': f"Address top risk factor: {top['factor']}",
            'focus_factor': top['factor'],
            'reduction': _to_native(reduction),
            'new_rate': _to_native(new_rate),
            'incidents_prevented': prevented
        })
    
    # Scenario 2: Enhance protection factors
    protection = [i for i in patterns['indicators'] if i['coefficient'] < 0][:2]
    if protection:
        total_impact = sum(abs(p['coefficient']) for p in protection)
        reduction = min(total_impact * 12, 25)
        new_rate = max(current_rate * (1 - reduction/100), 0)
        prevented = int(n_incidents * reduction / 100)
        scenarios.append({
            'name': 'Enhance Safety Measures',
            'description': f"Strengthen {', '.join([p['factor'][:10] for p in protection])}",
            'focus_factor': 'Safety Measures',
            'reduction': _to_native(reduction),
            'new_rate': _to_native(new_rate),
            'incidents_prevented': prevented
        })
    
    # Scenario 3: Comprehensive safety program
    reduction = 35
    new_rate = current_rate * 0.65
    prevented = int(n_incidents * 0.35)
    scenarios.append({
        'name': 'Comprehensive Safety Program',
        'description': 'Full safety overhaul: training, equipment, monitoring',
        'focus_factor': 'All Factors',
        'reduction': _to_native(reduction),
        'new_rate': _to_native(new_rate),
        'incidents_prevented': prevented
    })
    
    best = max(scenarios, key=lambda x: x['reduction']) if scenarios else None
    
    recommendations = []
    if top and top['coefficient'] > 0:
        recommendations.append(f"Priority: Address {top['factor']} - strongest risk factor")
    if protection:
        recommendations.append(f"Strengthen: {protection[0]['factor']} - reduces incidents")
    recommendations.append("Implement early warning system based on identified thresholds")
    recommendations.append("Regular safety audits focusing on high-risk locations")
    
    return {
        'current_incident_rate': _to_native(current_rate),
        'scenarios': scenarios,
        'best_scenario': best,
        'recommendations': recommendations
    }


def create_prevention_chart(prevention: Dict) -> str:
    if prevention.get('error'):
        return ""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    scenarios = prevention['scenarios']
    
    # Chart 1: Incident rate comparison
    names = ['Current'] + [s['name'][:15] for s in scenarios]
    rates = [prevention['current_incident_rate']] + [s['new_rate'] for s in scenarios]
    colors = ['#ef4444'] + ['#10b981'] * len(scenarios)
    
    axes[0].bar(names, rates, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_ylabel('Incident Rate (%)')
    axes[0].set_title('Projected Incident Rate', fontsize=11, fontweight='bold')
    axes[0].set_xticklabels(names, rotation=45, ha='right')
    
    # Chart 2: Incidents prevented
    names2 = [s['name'][:15] for s in scenarios]
    prevented = [s['incidents_prevented'] for s in scenarios]
    
    axes[1].barh(names2, prevented, color='#10b981', alpha=0.8, edgecolor='black')
    axes[1].set_xlabel('Incidents Prevented')
    axes[1].set_title('Prevention Impact', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# Report Generation
def generate_report(overview: Dict, comparison: Dict, correlation: Dict, patterns: Dict, prevention: Dict) -> Dict:
    report = {}
    
    report['step1_overview'] = {
        'title': '1. Incident Overview',
        'question': 'What is our current incident rate?',
        'finding': f"Incident rate: {overview['incident_rate']:.1f}%, Safety rate: {overview['safe_rate']:.1f}%",
        'detail': f"Analysis of {overview['n_records']:,} records shows {overview['n_incidents']:,} incidents ({overview['incident_rate']:.1f}%). "
                 f"Safety rate stands at {overview['safe_rate']:.1f}%. "
                 + (f"Most common incident type is '{overview.get('top_incident_type', 'N/A')}'." if overview.get('top_incident_type') else "")
    }
    
    if comparison and not comparison.get('error'):
        report['step2_comparison'] = {
            'title': '2. Location/Situation Comparison',
            'question': 'Where are incidents most common?',
            'finding': f"Highest risk: {comparison.get('worst_location', 'N/A')} ({comparison.get('worst_location_rate', 0):.1f}%), Safest: {comparison.get('safest_location', 'N/A')}",
            'detail': f"Location analysis reveals significant variation in incident rates. "
                     f"'{comparison.get('worst_location', 'N/A')}' has the highest risk at {comparison.get('worst_location_rate', 0):.1f}%, "
                     f"while '{comparison.get('safest_location', 'N/A')}' is safest at {comparison.get('safest_location_rate', 0):.1f}%. "
                     + (f"Time analysis shows '{comparison.get('worst_time', 'N/A')}' as the highest-risk period." if comparison.get('worst_time') else "")
        }
    else:
        report['step2_comparison'] = {'title': '2. Comparison', 'question': 'Location comparison', 'finding': 'Requires location column', 'detail': ''}
    
    if correlation:
        report['step3_correlation'] = {
            'title': '3. Safety Factor Correlation',
            'question': 'What safety factors affect incidents?',
            'finding': f"Risk factor: {correlation.get('strongest_risk', {}).get('factor', 'N/A')}, Protection: {correlation.get('strongest_protection', {}).get('factor', 'N/A')}",
            'detail': f"Correlation analysis found {correlation['n_significant']} significant relationships. "
                     + (f"'{correlation['strongest_risk']['factor']}' increases risk (r={correlation['strongest_risk']['correlation']:.3f}). " if correlation.get('strongest_risk') else "")
                     + (f"'{correlation['strongest_protection']['factor']}' reduces risk (r={correlation['strongest_protection']['correlation']:.3f})." if correlation.get('strongest_protection') else "")
        }
    
    if patterns and not patterns.get('error'):
        top = patterns.get('top_indicator', {})
        report['step4_patterns'] = {
            'title': '4. Risk Patterns & Warning Signs',
            'question': 'What are the warning signs?',
            'finding': f"Top indicator: {top.get('factor', 'N/A')} (importance: {top.get('importance', 0)*100:.1f}%)",
            'detail': f"Pattern analysis model ({patterns['model_accuracy']*100:.1f}% accuracy) identifies {patterns['n_risk_factors']} risk factors. "
                     f"'{top.get('factor', 'N/A')}' is the strongest predictor with {top.get('importance', 0)*100:.1f}% importance. "
                     f"Risk thresholds have been established for early warning."
        }
    else:
        report['step4_patterns'] = {'title': '4. Patterns', 'question': 'Warning signs', 'finding': patterns.get('error', 'Analysis not available'), 'detail': ''}
    
    if prevention and not prevention.get('error'):
        best = prevention.get('best_scenario', {})
        report['step5_prevention'] = {
            'title': '5. Prevention Impact Simulation',
            'question': 'How many incidents can we prevent?',
            'finding': f"Best scenario: -{best.get('reduction', 0):.0f}% incidents ({best.get('name', 'N/A')})",
            'detail': f"Current incident rate is {prevention['current_incident_rate']:.1f}%. "
                     f"'{best.get('name', 'N/A')}' could reduce this to {best.get('new_rate', 0):.1f}%, "
                     f"preventing approximately {best.get('incidents_prevented', 0)} incidents."
        }
    else:
        report['step5_prevention'] = {'title': '5. Prevention', 'question': 'Prevention potential', 'finding': prevention.get('error', 'Simulation not available'), 'detail': ''}
    
    return report


def generate_insights(overview: Dict, comparison: Dict, correlation: Dict, patterns: Dict, prevention: Dict) -> List[Dict]:
    insights = []
    
    if overview['incident_rate'] > 10:
        insights.append({'title': 'High Incident Rate', 'description': f"Incident rate of {overview['incident_rate']:.1f}% requires immediate attention.", 'status': 'warning'})
    elif overview['incident_rate'] < 5:
        insights.append({'title': 'Good Safety Record', 'description': f"Incident rate of {overview['incident_rate']:.1f}% is well controlled.", 'status': 'positive'})
    
    if comparison and not comparison.get('error'):
        if comparison.get('worst_location_rate', 0) > comparison.get('avg_rate', 0) * 1.5:
            insights.append({'title': 'High-Risk Location', 'description': f"'{comparison['worst_location']}' has significantly elevated risk.", 'status': 'warning'})
    
    if patterns and not patterns.get('error'):
        top = patterns.get('top_indicator', {})
        if top:
            insights.append({'title': f"Focus: {top['factor']}", 'description': f"Primary risk indicator with {top['importance']*100:.0f}% importance.", 'status': 'warning'})
    
    if prevention and not prevention.get('error'):
        best = prevention.get('best_scenario', {})
        if best and best.get('reduction', 0) > 20:
            insights.append({'title': 'High Prevention Potential', 'description': f"Can reduce incidents by {best['reduction']:.0f}%.", 'status': 'positive'})
    
    return insights


@router.post("/risk-safety")
async def analyze_risk_safety(request: RiskRequest):
    try:
        df = pd.DataFrame(request.data)
        if len(df) < 20:
            raise HTTPException(status_code=400, detail="Need at least 20 records")
        
        results, visualizations = {}, {}
        
        # Step 1: Overview
        overview = analyze_overview(df, request.incident_col)
        results['overview'] = overview
        visualizations['overview_chart'] = create_overview_chart(overview)
        
        # Step 2: Comparison
        comparison = analyze_comparison(df, request.incident_col, request.location_col, request.time_col)
        results['comparison'] = comparison
        if not comparison.get('error'):
            visualizations['comparison_chart'] = create_comparison_chart(comparison)
        
        # Step 3: Correlation
        correlation = analyze_correlation(df, request.incident_col, request.safety_cols)
        results['correlation'] = correlation
        visualizations['correlation_chart'] = create_correlation_chart(correlation)
        
        # Step 4: Patterns
        patterns = analyze_patterns(df, request.incident_col, request.safety_cols)
        results['patterns'] = patterns
        if not patterns.get('error'):
            visualizations['patterns_chart'] = create_patterns_chart(patterns)
        
        # Step 5: Prevention
        prevention = simulate_prevention(df, request.incident_col, patterns, overview)
        results['prevention'] = prevention
        if not prevention.get('error'):
            visualizations['prevention_chart'] = create_prevention_chart(prevention)
        
        report = generate_report(overview, comparison, correlation, patterns, prevention)
        insights = generate_insights(overview, comparison, correlation, patterns, prevention)
        
        summary = {
            'n_records': overview['n_records'],
            'incident_rate': overview['incident_rate'],
            'n_incidents': overview['n_incidents'],
            'worst_location': comparison.get('worst_location') if not comparison.get('error') else None,
            'top_risk_factor': patterns.get('top_indicator', {}).get('factor') if not patterns.get('error') else None,
            'prevention_potential': prevention.get('best_scenario', {}).get('reduction', 0) if not prevention.get('error') else 0
        }
        
        return {'success': True, 'results': results, 'visualizations': visualizations, 'report': report, 'key_insights': insights, 'summary': summary}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
