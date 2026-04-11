"""
Ad & Message Response Analysis API
5-step framework for advertising effectiveness
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

router = APIRouter()

class AdRequest(BaseModel):
    data: List[Dict[str, Any]]
    impression_col: str
    click_col: str
    conversion_col: Optional[str] = None
    creative_col: Optional[str] = None
    spend_col: Optional[str] = None
    appeal_cols: Optional[List[str]] = None

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


# Step 1: CTR Overview
def analyze_overview(df: pd.DataFrame, impression_col: str, click_col: str) -> Dict:
    impressions = pd.to_numeric(df[impression_col], errors='coerce').fillna(0)
    clicks = pd.to_numeric(df[click_col], errors='coerce').fillna(0)
    
    total_impressions = int(impressions.sum())
    total_clicks = int(clicks.sum())
    
    # Calculate CTR per row
    ctr_per_row = (clicks / impressions.replace(0, np.nan) * 100).dropna()
    
    return {
        'n_campaigns': len(df),
        'total_impressions': total_impressions,
        'total_clicks': total_clicks,
        'overall_ctr': _to_native(total_clicks / total_impressions * 100) if total_impressions > 0 else 0,
        'avg_ctr': _to_native(ctr_per_row.mean()),
        'ctr_std': _to_native(ctr_per_row.std()),
        'ctr_min': _to_native(ctr_per_row.min()),
        'ctr_max': _to_native(ctr_per_row.max()),
        'ctr_median': _to_native(ctr_per_row.median())
    }


def create_overview_chart(overview: Dict, df: pd.DataFrame, impression_col: str, click_col: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    impressions = pd.to_numeric(df[impression_col], errors='coerce').fillna(0)
    clicks = pd.to_numeric(df[click_col], errors='coerce').fillna(0)
    ctr = (clicks / impressions.replace(0, np.nan) * 100).dropna()
    
    # Chart 1: CTR distribution
    axes[0].hist(ctr, bins=20, color='#3b82f6', alpha=0.7, edgecolor='black')
    axes[0].axvline(x=overview['avg_ctr'], color='red', linestyle='--', label=f"Avg: {overview['avg_ctr']:.2f}%")
    axes[0].set_xlabel('CTR (%)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('CTR Distribution', fontsize=11, fontweight='bold')
    axes[0].legend()
    
    # Chart 2: Impressions vs Clicks scatter
    axes[1].scatter(impressions, clicks, alpha=0.5, color='#3b82f6')
    z = np.polyfit(impressions[impressions > 0], clicks[impressions > 0], 1)
    p = np.poly1d(z)
    x_line = np.linspace(impressions.min(), impressions.max(), 100)
    axes[1].plot(x_line, p(x_line), color='#ef4444', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Impressions')
    axes[1].set_ylabel('Clicks')
    axes[1].set_title('Impressions vs Clicks', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# Step 2: Creative Comparison
def analyze_comparison(df: pd.DataFrame, impression_col: str, click_col: str, creative_col: str) -> Dict:
    if not creative_col or creative_col not in df.columns:
        return {'error': 'Creative column required'}
    
    impressions = pd.to_numeric(df[impression_col], errors='coerce').fillna(0)
    clicks = pd.to_numeric(df[click_col], errors='coerce').fillna(0)
    
    creative_stats = []
    for creative in df[creative_col].unique():
        mask = df[creative_col] == creative
        imp = impressions[mask].sum()
        clk = clicks[mask].sum()
        ctr = clk / imp * 100 if imp > 0 else 0
        
        creative_stats.append({
            'creative': str(creative),
            'impressions': int(imp),
            'clicks': int(clk),
            'ctr': _to_native(ctr)
        })
    
    creative_stats = sorted(creative_stats, key=lambda x: x['ctr'], reverse=True)
    for i, c in enumerate(creative_stats):
        c['rank'] = i + 1
    
    best = creative_stats[0] if creative_stats else None
    worst = creative_stats[-1] if creative_stats else None
    
    return {
        'creative_stats': creative_stats,
        'n_creatives': len(creative_stats),
        'best_creative': best['creative'] if best else None,
        'best_ctr': best['ctr'] if best else None,
        'worst_creative': worst['creative'] if worst else None,
        'worst_ctr': worst['ctr'] if worst else None,
        'ctr_gap': _to_native(best['ctr'] - worst['ctr']) if best and worst else 0
    }


def create_comparison_chart(comparison: Dict) -> str:
    if comparison.get('error'):
        return ""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    stats = comparison['creative_stats']
    
    # Chart 1: CTR by creative
    creatives = [s['creative'][:15] for s in stats]
    ctrs = [s['ctr'] for s in stats]
    colors = ['#3b82f6' if i == 0 else '#94a3b8' for i in range(len(stats))]
    
    axes[0].barh(creatives, ctrs, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_xlabel('CTR (%)')
    axes[0].set_title('CTR by Creative', fontsize=11, fontweight='bold')
    
    # Chart 2: Impressions & Clicks
    x = np.arange(len(stats))
    width = 0.35
    
    imp_scaled = [s['impressions'] / 1000 for s in stats]
    clk_scaled = [s['clicks'] / 100 for s in stats]
    
    axes[1].bar(x - width/2, imp_scaled, width, label='Impressions (K)', color='#3b82f6', alpha=0.8)
    axes[1].bar(x + width/2, clk_scaled, width, label='Clicks (x100)', color='#10b981', alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([s['creative'][:10] for s in stats], rotation=45, ha='right')
    axes[1].set_title('Volume by Creative', fontsize=11, fontweight='bold')
    axes[1].legend()
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# Step 3: Conversion Analysis
def analyze_conversion(df: pd.DataFrame, impression_col: str, click_col: str, 
                       conversion_col: str, spend_col: Optional[str] = None) -> Dict:
    if not conversion_col or conversion_col not in df.columns:
        return {'error': 'Conversion column required'}
    
    impressions = pd.to_numeric(df[impression_col], errors='coerce').fillna(0)
    clicks = pd.to_numeric(df[click_col], errors='coerce').fillna(0)
    conversions = pd.to_numeric(df[conversion_col], errors='coerce').fillna(0)
    
    total_imp = impressions.sum()
    total_clicks = clicks.sum()
    total_conv = conversions.sum()
    
    # CVR calculation
    cvr_per_row = (conversions / clicks.replace(0, np.nan) * 100).dropna()
    
    # Click to conversion correlation
    valid_idx = (clicks > 0) & (conversions >= 0)
    corr = 0
    if valid_idx.sum() > 10:
        corr, _ = stats.pearsonr(clicks[valid_idx], conversions[valid_idx])
    
    result = {
        'total_conversions': int(total_conv),
        'overall_cvr': _to_native(total_conv / total_clicks * 100) if total_clicks > 0 else 0,
        'avg_cvr': _to_native(cvr_per_row.mean()),
        'cvr_std': _to_native(cvr_per_row.std()),
        'click_to_conv_corr': _to_native(corr),
        'funnel': [
            {'stage': 'Impressions', 'value': int(total_imp), 'rate': 100},
            {'stage': 'Clicks', 'value': int(total_clicks), 'rate': _to_native(total_clicks/total_imp*100) if total_imp > 0 else 0},
            {'stage': 'Conversions', 'value': int(total_conv), 'rate': _to_native(total_conv/total_imp*100) if total_imp > 0 else 0}
        ]
    }
    
    # ROAS if spend available
    if spend_col and spend_col in df.columns:
        spend = pd.to_numeric(df[spend_col], errors='coerce').fillna(0)
        total_spend = spend.sum()
        revenue = total_conv * 50  # Assume $50 per conversion
        roas = revenue / total_spend if total_spend > 0 else 0
        result['total_spend'] = _to_native(total_spend)
        result['avg_roas'] = _to_native(roas)
        result['avg_cpa'] = _to_native(total_spend / total_conv) if total_conv > 0 else None
    
    return result


def create_conversion_chart(conversion: Dict, df: pd.DataFrame, click_col: str, conversion_col: str) -> str:
    if conversion.get('error'):
        return ""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    clicks = pd.to_numeric(df[click_col], errors='coerce').fillna(0)
    conversions = pd.to_numeric(df[conversion_col], errors='coerce').fillna(0)
    
    # Chart 1: Funnel
    funnel = conversion['funnel']
    stages = [f['stage'] for f in funnel]
    values = [f['value'] for f in funnel]
    
    axes[0].barh(stages[::-1], values[::-1], color=['#3b82f6', '#10b981', '#f97316'], alpha=0.8, edgecolor='black')
    axes[0].set_xlabel('Count')
    axes[0].set_title('Conversion Funnel', fontsize=11, fontweight='bold')
    
    # Chart 2: Clicks vs Conversions
    valid_idx = clicks > 0
    axes[1].scatter(clicks[valid_idx], conversions[valid_idx], alpha=0.5, color='#3b82f6')
    if valid_idx.sum() > 5:
        z = np.polyfit(clicks[valid_idx], conversions[valid_idx], 1)
        p = np.poly1d(z)
        x_line = np.linspace(clicks[valid_idx].min(), clicks[valid_idx].max(), 100)
        axes[1].plot(x_line, p(x_line), color='#ef4444', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Clicks')
    axes[1].set_ylabel('Conversions')
    axes[1].set_title(f"Click-Conversion (r={conversion['click_to_conv_corr']:.3f})", fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# Step 4: Message Appeal Evaluation
def analyze_appeal(df: pd.DataFrame, impression_col: str, click_col: str, appeal_cols: List[str]) -> Dict:
    if not appeal_cols:
        return {'error': 'Appeal score columns required'}
    
    impressions = pd.to_numeric(df[impression_col], errors='coerce').fillna(0)
    clicks = pd.to_numeric(df[click_col], errors='coerce').fillna(0)
    ctr = (clicks / impressions.replace(0, np.nan) * 100).dropna()
    
    appeal_stats = []
    for col in appeal_cols:
        if col not in df.columns:
            continue
        
        vals = pd.to_numeric(df[col], errors='coerce')
        valid_idx = vals.notna() & ctr.index.isin(vals.index)
        
        if valid_idx.sum() < 10:
            continue
        
        corr, p_val = stats.pearsonr(vals[valid_idx], ctr.loc[valid_idx])
        
        appeal_stats.append({
            'factor': col,
            'avg_score': _to_native(vals.mean()),
            'correlation': _to_native(corr),
            'p_value': _to_native(p_val),
            'significant': bool(p_val < 0.05)
        })
    
    appeal_stats = sorted(appeal_stats, key=lambda x: abs(x.get('correlation') or 0), reverse=True)
    
    return {
        'appeal_stats': appeal_stats,
        'top_appeal': appeal_stats[0] if appeal_stats else None,
        'n_significant': sum(1 for a in appeal_stats if a['significant'])
    }


def create_appeal_chart(appeal: Dict, df: pd.DataFrame, impression_col: str, click_col: str) -> str:
    if appeal.get('error'):
        return ""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    stats = appeal['appeal_stats'][:8]
    
    # Chart 1: Correlation bars
    factors = [s['factor'][:15] for s in stats]
    corrs = [s['correlation'] for s in stats]
    colors = ['#3b82f6' if s['significant'] else '#d1d5db' for s in stats]
    
    axes[0].barh(factors, corrs, color=colors, alpha=0.8, edgecolor='black')
    axes[0].axvline(x=0, color='black', linewidth=0.5)
    axes[0].set_xlabel('Correlation with CTR')
    axes[0].set_title('Appeal Factor Impact', fontsize=11, fontweight='bold')
    
    # Chart 2: Top appeal scatter
    top = appeal.get('top_appeal')
    if top and top['factor'] in df.columns:
        impressions = pd.to_numeric(df[impression_col], errors='coerce').fillna(0)
        clicks = pd.to_numeric(df[click_col], errors='coerce').fillna(0)
        ctr = (clicks / impressions.replace(0, np.nan) * 100).dropna()
        vals = pd.to_numeric(df[top['factor']], errors='coerce')
        
        valid_idx = vals.notna() & ctr.index.isin(vals.index)
        
        axes[1].scatter(vals[valid_idx], ctr.loc[valid_idx], alpha=0.5, color='#3b82f6')
        if valid_idx.sum() > 5:
            z = np.polyfit(vals[valid_idx], ctr.loc[valid_idx], 1)
            p = np.poly1d(z)
            x_line = np.linspace(vals[valid_idx].min(), vals[valid_idx].max(), 100)
            axes[1].plot(x_line, p(x_line), color='#ef4444', linestyle='--', linewidth=2)
        
        axes[1].set_xlabel(top['factor'])
        axes[1].set_ylabel('CTR (%)')
        axes[1].set_title(f"Top Appeal (r={top['correlation']:.3f})", fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# Step 5: Budget Simulation
def simulate_budget(df: pd.DataFrame, impression_col: str, spend_col: str, overview: Dict) -> Dict:
    if not spend_col or spend_col not in df.columns:
        return {'error': 'Spend column required for simulation'}
    
    impressions = pd.to_numeric(df[impression_col], errors='coerce').fillna(0)
    spend = pd.to_numeric(df[spend_col], errors='coerce').fillna(0)
    
    total_spend = spend.sum()
    total_reach = impressions.sum()
    
    # Cost per impression
    cpi = total_spend / total_reach if total_reach > 0 else 0
    
    # Diminishing returns factor
    diminishing = 0.85  # Each $ increase yields slightly less
    
    scenarios = []
    
    # Scenario 1: +25% budget
    new_spend = total_spend * 1.25
    new_reach = total_reach * (1.25 ** diminishing)
    reach_gain = (new_reach / total_reach - 1) * 100
    scenarios.append({
        'name': '+25% Budget',
        'spend': _to_native(new_spend),
        'projected_reach': int(new_reach),
        'reach_gain': _to_native(reach_gain),
        'efficiency': _to_native(reach_gain / 25),
        'recommended': False
    })
    
    # Scenario 2: +50% budget (often optimal)
    new_spend = total_spend * 1.5
    new_reach = total_reach * (1.5 ** diminishing)
    reach_gain = (new_reach / total_reach - 1) * 100
    scenarios.append({
        'name': '+50% Budget',
        'spend': _to_native(new_spend),
        'projected_reach': int(new_reach),
        'reach_gain': _to_native(reach_gain),
        'efficiency': _to_native(reach_gain / 50),
        'recommended': True
    })
    
    # Scenario 3: +100% budget
    new_spend = total_spend * 2
    new_reach = total_reach * (2 ** diminishing)
    reach_gain = (new_reach / total_reach - 1) * 100
    scenarios.append({
        'name': '+100% Budget',
        'spend': _to_native(new_spend),
        'projected_reach': int(new_reach),
        'reach_gain': _to_native(reach_gain),
        'efficiency': _to_native(reach_gain / 100),
        'recommended': False
    })
    
    best = max(scenarios, key=lambda x: x['efficiency'])
    
    recommendations = []
    recommendations.append(f"Current cost per impression: ${cpi:.4f}")
    recommendations.append(f"Optimal budget increase: {best['name']} for best efficiency")
    recommendations.append("Consider reallocating budget to best-performing creatives")
    recommendations.append("Monitor diminishing returns as budget scales")
    
    return {
        'current_spend': _to_native(total_spend),
        'current_reach': int(total_reach),
        'cost_per_impression': _to_native(cpi),
        'scenarios': scenarios,
        'best_scenario': best,
        'recommendations': recommendations
    }


def create_simulation_chart(simulation: Dict) -> str:
    if simulation.get('error'):
        return ""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    scenarios = simulation['scenarios']
    
    # Chart 1: Reach projection
    names = ['Current'] + [s['name'] for s in scenarios]
    reaches = [simulation['current_reach']] + [s['projected_reach'] for s in scenarios]
    colors = ['#94a3b8'] + ['#3b82f6' if s['recommended'] else '#6b7280' for s in scenarios]
    
    axes[0].bar(names, [r/1000 for r in reaches], color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_ylabel('Reach (K impressions)')
    axes[0].set_title('Projected Reach by Budget', fontsize=11, fontweight='bold')
    axes[0].set_xticklabels(names, rotation=45, ha='right')
    
    # Chart 2: Efficiency curve
    budget_mult = [1, 1.25, 1.5, 2]
    efficiencies = [1] + [s['efficiency'] for s in scenarios]
    
    axes[1].plot(budget_mult, efficiencies, marker='o', color='#3b82f6', linewidth=2, markersize=8)
    axes[1].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Budget Multiplier')
    axes[1].set_ylabel('Efficiency Score')
    axes[1].set_title('Budget Efficiency Curve', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# Report Generation
def generate_report(overview: Dict, comparison: Dict, conversion: Dict, appeal: Dict, simulation: Dict) -> Dict:
    report = {}
    
    report['step1_overview'] = {
        'title': '1. Exposure & Click-Through Rate',
        'question': 'What is our click-through rate?',
        'finding': f"Avg CTR: {overview['avg_ctr']:.2f}%, Total: {overview['total_impressions']:,} impressions → {overview['total_clicks']:,} clicks",
        'detail': f"Analysis of {overview['n_campaigns']} campaigns shows an average CTR of {overview['avg_ctr']:.2f}% "
                 f"(range: {overview['ctr_min']:.2f}% - {overview['ctr_max']:.2f}%). "
                 f"Total reach: {overview['total_impressions']:,} impressions generating {overview['total_clicks']:,} clicks."
    }
    
    if comparison and not comparison.get('error'):
        report['step2_comparison'] = {
            'title': '2. Creative Comparison',
            'question': 'Which creative performs best?',
            'finding': f"Best: {comparison['best_creative']} ({comparison['best_ctr']:.2f}% CTR), Gap: {comparison['ctr_gap']:.2f}%p",
            'detail': f"Comparison of {comparison['n_creatives']} creatives shows '{comparison['best_creative']}' leads with {comparison['best_ctr']:.2f}% CTR. "
                     f"Worst performer '{comparison['worst_creative']}' achieves {comparison['worst_ctr']:.2f}% CTR, "
                     f"a gap of {comparison['ctr_gap']:.2f} percentage points."
        }
    else:
        report['step2_comparison'] = {'title': '2. Creative Comparison', 'question': 'Which creative performs best?', 'finding': 'Requires creative column', 'detail': ''}
    
    if conversion and not conversion.get('error'):
        report['step3_conversion'] = {
            'title': '3. Conversion Analysis',
            'question': 'Does exposure lead to conversion?',
            'finding': f"CVR: {conversion['avg_cvr']:.2f}%, Click-Conv correlation: {conversion['click_to_conv_corr']:.3f}",
            'detail': f"Total {conversion['total_conversions']:,} conversions from clicks (avg CVR: {conversion['avg_cvr']:.2f}%). "
                     f"Click-to-conversion correlation of {conversion['click_to_conv_corr']:.3f} indicates {'strong' if abs(conversion['click_to_conv_corr']) > 0.5 else 'moderate' if abs(conversion['click_to_conv_corr']) > 0.3 else 'weak'} relationship. "
                     + (f"Average ROAS: {conversion.get('avg_roas', 0):.1f}x." if conversion.get('avg_roas') else "")
        }
    else:
        report['step3_conversion'] = {'title': '3. Conversion', 'question': 'Conversion analysis', 'finding': conversion.get('error', 'Requires conversion column'), 'detail': ''}
    
    if appeal and not appeal.get('error'):
        top = appeal.get('top_appeal', {})
        report['step4_appeal'] = {
            'title': '4. Message Appeal Evaluation',
            'question': 'What makes messages effective?',
            'finding': f"Top appeal: {top.get('factor', 'N/A')} (r={top.get('correlation', 0):.3f})",
            'detail': f"Appeal analysis shows '{top.get('factor', 'N/A')}' most strongly correlates with CTR (r={top.get('correlation', 0):.3f}). "
                     f"{appeal['n_significant']} factors show statistically significant relationships with ad performance."
        }
    else:
        report['step4_appeal'] = {'title': '4. Message Appeal', 'question': 'Message effectiveness', 'finding': appeal.get('error', 'Requires appeal columns'), 'detail': ''}
    
    if simulation and not simulation.get('error'):
        best = simulation.get('best_scenario', {})
        report['step5_simulation'] = {
            'title': '5. Budget & Reach Simulation',
            'question': 'What if we increase ad budget?',
            'finding': f"Best scenario: {best.get('name', 'N/A')} → +{best.get('reach_gain', 0):.0f}% reach",
            'detail': f"Current spend: ${simulation['current_spend']:,.0f} for {simulation['current_reach']:,} impressions. "
                     f"'{best.get('name', 'N/A')}' scenario offers best efficiency with +{best.get('reach_gain', 0):.0f}% reach increase."
        }
    else:
        report['step5_simulation'] = {'title': '5. Budget Simulation', 'question': 'Budget impact', 'finding': simulation.get('error', 'Requires spend column'), 'detail': ''}
    
    return report


def generate_insights(overview: Dict, comparison: Dict, conversion: Dict, appeal: Dict, simulation: Dict) -> List[Dict]:
    insights = []
    
    if overview['avg_ctr'] > 3:
        insights.append({'title': 'Strong CTR', 'description': f"Average CTR of {overview['avg_ctr']:.2f}% exceeds industry benchmarks.", 'status': 'positive'})
    elif overview['avg_ctr'] < 1:
        insights.append({'title': 'Low CTR', 'description': f"Average CTR of {overview['avg_ctr']:.2f}% needs improvement.", 'status': 'warning'})
    
    if comparison and not comparison.get('error'):
        if comparison['ctr_gap'] > 1:
            insights.append({'title': 'Creative Gap', 'description': f"Best creative outperforms worst by {comparison['ctr_gap']:.2f}%p - consolidate on winners.", 'status': 'positive'})
    
    if conversion and not conversion.get('error'):
        if conversion.get('avg_roas', 0) > 3:
            insights.append({'title': 'Strong ROAS', 'description': f"ROAS of {conversion['avg_roas']:.1f}x indicates profitable campaigns.", 'status': 'positive'})
    
    if appeal and not appeal.get('error') and appeal.get('top_appeal'):
        top = appeal['top_appeal']
        insights.append({'title': f"Focus: {top['factor']}", 'description': f"Strongest driver of ad response (r={top['correlation']:.3f}).", 'status': 'positive'})
    
    return insights


@router.post("/ad-response")
async def analyze_ad_response(request: AdRequest):
    try:
        df = pd.DataFrame(request.data)
        if len(df) < 10:
            raise HTTPException(status_code=400, detail="Need at least 10 records")
        
        results, visualizations = {}, {}
        
        # Step 1: Overview
        overview = analyze_overview(df, request.impression_col, request.click_col)
        results['overview'] = overview
        visualizations['overview_chart'] = create_overview_chart(overview, df, request.impression_col, request.click_col)
        
        # Step 2: Comparison
        comparison = {}
        if request.creative_col:
            comparison = analyze_comparison(df, request.impression_col, request.click_col, request.creative_col)
            results['comparison'] = comparison
            if not comparison.get('error'):
                visualizations['comparison_chart'] = create_comparison_chart(comparison)
        
        # Step 3: Conversion
        conversion = {}
        if request.conversion_col:
            conversion = analyze_conversion(df, request.impression_col, request.click_col, 
                                           request.conversion_col, request.spend_col)
            results['conversion'] = conversion
            if not conversion.get('error'):
                visualizations['conversion_chart'] = create_conversion_chart(conversion, df, request.click_col, request.conversion_col)
        
        # Step 4: Appeal
        appeal = {}
        if request.appeal_cols:
            appeal = analyze_appeal(df, request.impression_col, request.click_col, request.appeal_cols)
            results['appeal'] = appeal
            if not appeal.get('error'):
                visualizations['appeal_chart'] = create_appeal_chart(appeal, df, request.impression_col, request.click_col)
        
        # Step 5: Simulation
        simulation = {}
        if request.spend_col:
            simulation = simulate_budget(df, request.impression_col, request.spend_col, overview)
            results['simulation'] = simulation
            if not simulation.get('error'):
                visualizations['simulation_chart'] = create_simulation_chart(simulation)
        
        report = generate_report(overview, comparison, conversion, appeal, simulation)
        insights = generate_insights(overview, comparison, conversion, appeal, simulation)
        
        summary = {
            'n_records': overview['n_campaigns'],
            'avg_ctr': overview['avg_ctr'],
            'avg_conversion': conversion.get('avg_cvr', 0) if conversion else 0,
            'best_creative': comparison.get('best_creative') if comparison else None,
            'top_appeal': appeal.get('top_appeal', {}).get('factor') if appeal else None,
            'potential_reach_gain': simulation.get('best_scenario', {}).get('reach_gain', 0) if simulation else 0
        }
        
        return {'success': True, 'results': results, 'visualizations': visualizations, 'report': report, 'key_insights': insights, 'summary': summary}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
