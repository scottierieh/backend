"""
Brand Image & Competitiveness API
5-step framework for brand perception analysis
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

router = APIRouter()

class BrandRequest(BaseModel):
    data: List[Dict[str, Any]]
    brand_col: str
    image_cols: List[str]
    our_brand: str
    loyalty_col: Optional[str] = None
    awareness_col: Optional[str] = None

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


# Step 1: Awareness & Image Distribution
def analyze_awareness(df: pd.DataFrame, brand_col: str, image_cols: List[str], 
                      our_brand: str, awareness_col: Optional[str] = None) -> Dict:
    our_data = df[df[brand_col] == our_brand]
    
    result = {
        'n_respondents': len(df),
        'n_our_brand': len(our_data),
        'brands': df[brand_col].unique().tolist()
    }
    
    # Awareness rate
    if awareness_col and awareness_col in df.columns:
        awareness = pd.to_numeric(our_data[awareness_col], errors='coerce').dropna()
        result['awareness_rate'] = _to_native(awareness.mean())
        result['awareness_std'] = _to_native(awareness.std())
    else:
        result['awareness_rate'] = None
    
    # Image scores for our brand
    our_scores = {}
    for col in image_cols:
        if col in df.columns:
            vals = pd.to_numeric(our_data[col], errors='coerce').dropna()
            if len(vals) > 0:
                our_scores[col] = _to_native(vals.mean())
    
    result['our_brand_scores'] = our_scores
    result['avg_image_score'] = _to_native(np.mean(list(our_scores.values()))) if our_scores else None
    
    # Strongest and weakest attributes
    if our_scores:
        result['strongest_attribute'] = max(our_scores, key=our_scores.get)
        result['weakest_attribute'] = min(our_scores, key=our_scores.get)
    
    # Distribution stats
    distributions = {}
    for col in image_cols:
        if col in df.columns:
            vals = pd.to_numeric(our_data[col], errors='coerce').dropna()
            if len(vals) > 0:
                distributions[col] = {
                    'mean': _to_native(vals.mean()),
                    'std': _to_native(vals.std()),
                    'min': _to_native(vals.min()),
                    'max': _to_native(vals.max())
                }
    result['distributions'] = distributions
    
    return result


def create_awareness_chart(awareness: Dict, df: pd.DataFrame, brand_col: str, 
                           image_cols: List[str], our_brand: str) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Chart 1: Our brand image profile (radar-like bar chart)
    scores = awareness.get('our_brand_scores', {})
    if scores:
        attrs = list(scores.keys())
        values = list(scores.values())
        colors = ['#3b82f6' if v >= np.mean(values) else '#94a3b8' for v in values]
        
        axes[0].barh(attrs, values, color=colors, alpha=0.8, edgecolor='black')
        axes[0].axvline(x=np.mean(values), color='red', linestyle='--', label=f'Avg: {np.mean(values):.2f}')
        axes[0].set_xlabel('Score')
        axes[0].set_title(f'{our_brand} Image Profile', fontsize=11, fontweight='bold')
        axes[0].legend()
    
    # Chart 2: Score distribution comparison
    our_data = df[df[brand_col] == our_brand]
    for i, col in enumerate(image_cols[:4]):
        if col in df.columns:
            vals = pd.to_numeric(our_data[col], errors='coerce').dropna()
            if len(vals) > 0:
                axes[1].hist(vals, bins=10, alpha=0.5, label=col[:15])
    
    axes[1].set_xlabel('Score')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Score Distributions', fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=8)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# Step 2: Competitive Comparison
def analyze_comparison(df: pd.DataFrame, brand_col: str, image_cols: List[str], our_brand: str) -> Dict:
    brands = df[brand_col].unique()
    
    brand_scores = []
    for brand in brands:
        brand_data = df[df[brand_col] == brand]
        scores = {}
        for col in image_cols:
            if col in df.columns:
                vals = pd.to_numeric(brand_data[col], errors='coerce').dropna()
                if len(vals) > 0:
                    scores[col] = _to_native(vals.mean())
        
        avg_score = np.mean(list(scores.values())) if scores else 0
        strongest = max(scores, key=scores.get) if scores else None
        weakest = min(scores, key=scores.get) if scores else None
        
        brand_scores.append({
            'brand': str(brand),
            'is_ours': brand == our_brand,
            'scores': scores,
            'avg_score': _to_native(avg_score),
            'strongest_attr': strongest,
            'weakest_attr': weakest
        })
    
    # Rank brands
    brand_scores = sorted(brand_scores, key=lambda x: x['avg_score'], reverse=True)
    for i, b in enumerate(brand_scores):
        b['rank'] = i + 1
    
    # Our brand info
    our_info = next((b for b in brand_scores if b['is_ours']), None)
    
    return {
        'brand_rankings': brand_scores,
        'n_brands': len(brand_scores),
        'our_rank': our_info['rank'] if our_info else None,
        'our_strength': our_info['strongest_attr'] if our_info else None,
        'our_weakness': our_info['weakest_attr'] if our_info else None,
        'leader': brand_scores[0]['brand'] if brand_scores else None
    }


def create_comparison_chart(comparison: Dict, image_cols: List[str]) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    rankings = comparison['brand_rankings']
    
    # Chart 1: Overall brand ranking
    brands = [r['brand'][:12] for r in rankings]
    scores = [r['avg_score'] for r in rankings]
    colors = ['#3b82f6' if r['is_ours'] else '#94a3b8' for r in rankings]
    
    axes[0].barh(brands, scores, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_xlabel('Average Image Score')
    axes[0].set_title('Brand Ranking', fontsize=11, fontweight='bold')
    
    # Chart 2: Attribute comparison (spider-like grouped bar)
    n_brands = min(len(rankings), 4)
    n_attrs = min(len(image_cols), 5)
    
    x = np.arange(n_attrs)
    width = 0.8 / n_brands
    
    for i, brand_info in enumerate(rankings[:n_brands]):
        scores = [brand_info['scores'].get(col, 0) for col in image_cols[:n_attrs]]
        color = '#3b82f6' if brand_info['is_ours'] else f'C{i}'
        axes[1].bar(x + i * width, scores, width, label=brand_info['brand'][:10], 
                   color=color, alpha=0.8)
    
    axes[1].set_xticks(x + width * (n_brands - 1) / 2)
    axes[1].set_xticklabels([c[:12] for c in image_cols[:n_attrs]], rotation=45, ha='right')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Attribute Comparison', fontsize=11, fontweight='bold')
    axes[1].legend(fontsize=8)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# Step 3: Image-Loyalty Relationship
def analyze_loyalty(df: pd.DataFrame, brand_col: str, image_cols: List[str], 
                    our_brand: str, loyalty_col: str) -> Dict:
    if not loyalty_col or loyalty_col not in df.columns:
        return {'error': 'Loyalty column required'}
    
    our_data = df[df[brand_col] == our_brand].copy()
    loyalty = pd.to_numeric(our_data[loyalty_col], errors='coerce')
    
    correlations = []
    for col in image_cols:
        if col not in df.columns:
            continue
        
        img = pd.to_numeric(our_data[col], errors='coerce')
        valid_idx = img.notna() & loyalty.notna()
        
        if valid_idx.sum() < 10:
            continue
        
        corr, p_val = stats.pearsonr(img[valid_idx], loyalty[valid_idx])
        
        correlations.append({
            'attribute': col,
            'correlation': _to_native(corr),
            'p_value': _to_native(p_val),
            'significant': bool(p_val < 0.05)
        })
    
    correlations = sorted(correlations, key=lambda x: abs(x.get('correlation') or 0), reverse=True)
    
    # Multiple regression for R²
    valid_cols = [c for c in image_cols if c in df.columns]
    X = our_data[valid_cols].apply(pd.to_numeric, errors='coerce')
    valid_idx = X.notna().all(axis=1) & loyalty.notna()
    
    r_squared = 0
    if valid_idx.sum() > 20:
        X_clean = X[valid_idx]
        y_clean = loyalty[valid_idx]
        model = LinearRegression().fit(X_clean, y_clean)
        r_squared = model.score(X_clean, y_clean)
    
    # Overall correlation
    all_img = our_data[valid_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)
    valid_idx = all_img.notna() & loyalty.notna()
    overall_corr = 0
    if valid_idx.sum() > 10:
        overall_corr, _ = stats.pearsonr(all_img[valid_idx], loyalty[valid_idx])
    
    return {
        'attribute_correlations': correlations,
        'top_driver': correlations[0] if correlations else None,
        'overall_correlation': _to_native(overall_corr),
        'r_squared': _to_native(r_squared),
        'n_significant': sum(1 for c in correlations if c['significant'])
    }


def create_loyalty_chart(loyalty: Dict, df: pd.DataFrame, brand_col: str, 
                         image_cols: List[str], our_brand: str, loyalty_col: str) -> str:
    if loyalty.get('error'):
        return ""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    our_data = df[df[brand_col] == our_brand]
    loy = pd.to_numeric(our_data[loyalty_col], errors='coerce')
    
    # Chart 1: Correlation bars
    corrs = loyalty['attribute_correlations'][:8]
    attrs = [c['attribute'][:15] for c in corrs]
    values = [c['correlation'] for c in corrs]
    colors = ['#3b82f6' if c['significant'] else '#d1d5db' for c in corrs]
    
    axes[0].barh(attrs, values, color=colors, alpha=0.8, edgecolor='black')
    axes[0].axvline(x=0, color='black', linewidth=0.5)
    axes[0].set_xlabel('Correlation with Loyalty')
    axes[0].set_title('Image-Loyalty Correlations', fontsize=11, fontweight='bold')
    
    # Chart 2: Scatter of top driver
    top = loyalty.get('top_driver')
    if top and top['attribute'] in df.columns:
        img = pd.to_numeric(our_data[top['attribute']], errors='coerce')
        valid_idx = img.notna() & loy.notna()
        
        axes[1].scatter(img[valid_idx], loy[valid_idx], alpha=0.5, color='#3b82f6')
        
        # Trend line
        z = np.polyfit(img[valid_idx], loy[valid_idx], 1)
        p = np.poly1d(z)
        x_line = np.linspace(img[valid_idx].min(), img[valid_idx].max(), 100)
        axes[1].plot(x_line, p(x_line), color='#ef4444', linestyle='--', linewidth=2)
        
        axes[1].set_xlabel(top['attribute'])
        axes[1].set_ylabel(loyalty_col)
        axes[1].set_title(f"Top Driver (r={top['correlation']:.3f})", fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# Step 4: Gap Analysis
def analyze_gap(df: pd.DataFrame, brand_col: str, image_cols: List[str], our_brand: str) -> Dict:
    brands = df[brand_col].unique()
    
    # Calculate scores for each brand and attribute
    brand_attr_scores = {}
    for brand in brands:
        brand_data = df[df[brand_col] == brand]
        brand_attr_scores[str(brand)] = {}
        for col in image_cols:
            if col in df.columns:
                vals = pd.to_numeric(brand_data[col], errors='coerce').dropna()
                if len(vals) > 0:
                    brand_attr_scores[str(brand)][col] = vals.mean()
    
    our_scores = brand_attr_scores.get(our_brand, {})
    
    # Find gaps
    gaps = []
    for col in image_cols:
        our_score = our_scores.get(col, 0)
        
        # Find best competitor score
        best_score = our_score
        best_brand = our_brand
        for brand, scores in brand_attr_scores.items():
            if brand != our_brand and scores.get(col, 0) > best_score:
                best_score = scores.get(col, 0)
                best_brand = brand
        
        gap = our_score - best_score
        
        gaps.append({
            'attribute': col,
            'our_score': _to_native(our_score),
            'best_score': _to_native(best_score),
            'vs_brand': best_brand,
            'gap': _to_native(gap),
            'priority': 'high' if gap < -0.3 else 'medium' if gap < 0 else 'low'
        })
    
    gaps = sorted(gaps, key=lambda x: x['gap'])
    
    # Priority gaps (negative gaps = we're behind)
    priority_gaps = [g for g in gaps if g['gap'] < 0]
    
    return {
        'gaps': gaps,
        'priority_gaps': priority_gaps[:5],
        'biggest_gap': gaps[0] if gaps else None,
        'n_gaps_behind': sum(1 for g in gaps if g['gap'] < 0),
        'n_gaps_ahead': sum(1 for g in gaps if g['gap'] > 0)
    }


def create_gap_chart(gap: Dict, image_cols: List[str]) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    gaps = gap['gaps']
    
    # Chart 1: Gap bars
    attrs = [g['attribute'][:15] for g in gaps]
    values = [g['gap'] for g in gaps]
    colors = ['#10b981' if v >= 0 else '#ef4444' for v in values]
    
    axes[0].barh(attrs, values, color=colors, alpha=0.8, edgecolor='black')
    axes[0].axvline(x=0, color='black', linewidth=1)
    axes[0].set_xlabel('Gap (+ = ahead, - = behind)')
    axes[0].set_title('Competitive Gap by Attribute', fontsize=11, fontweight='bold')
    
    # Chart 2: Our score vs best competitor
    our_scores = [g['our_score'] for g in gaps]
    best_scores = [g['best_score'] for g in gaps]
    
    x = np.arange(len(gaps))
    width = 0.35
    
    axes[1].bar(x - width/2, our_scores, width, label='Our Brand', color='#3b82f6', alpha=0.8)
    axes[1].bar(x + width/2, best_scores, width, label='Best Competitor', color='#94a3b8', alpha=0.8)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(attrs, rotation=45, ha='right')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Our Brand vs Best Competitor', fontsize=11, fontweight='bold')
    axes[1].legend()
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# Step 5: Market Share Simulation
def simulate_share(df: pd.DataFrame, brand_col: str, image_cols: List[str], 
                   our_brand: str, gap: Dict, loyalty: Optional[Dict] = None) -> Dict:
    brands = df[brand_col].unique()
    n_brands = len(brands)
    
    # Estimate current "share" from sample proportion
    current_share = (df[brand_col] == our_brand).mean() * 100
    
    scenarios = []
    
    # Scenario 1: Close biggest gap
    biggest_gap = gap.get('biggest_gap')
    if biggest_gap and biggest_gap['gap'] < 0:
        improvement = abs(biggest_gap['gap'])
        share_gain = improvement * 2  # Simplified model
        scenarios.append({
            'name': f"Close {biggest_gap['attribute'][:15]} Gap",
            'description': f"Match best competitor in {biggest_gap['attribute']}",
            'focus_attribute': biggest_gap['attribute'],
            'improvement': _to_native(improvement),
            'share_gain': _to_native(min(share_gain, 10)),
            'new_share': _to_native(current_share + min(share_gain, 10))
        })
    
    # Scenario 2: Improve top loyalty driver
    if loyalty and not loyalty.get('error') and loyalty.get('top_driver'):
        driver = loyalty['top_driver']
        share_gain = abs(driver['correlation']) * 5
        scenarios.append({
            'name': f"Boost {driver['attribute'][:15]}",
            'description': f"Improve top loyalty driver by 20%",
            'focus_attribute': driver['attribute'],
            'improvement': 0.2,
            'share_gain': _to_native(min(share_gain, 8)),
            'new_share': _to_native(current_share + min(share_gain, 8))
        })
    
    # Scenario 3: Multi-attribute improvement
    priority_gaps = gap.get('priority_gaps', [])[:3]
    if priority_gaps:
        total_gap = sum(abs(g['gap']) for g in priority_gaps)
        share_gain = total_gap * 1.5
        scenarios.append({
            'name': 'Multi-Attribute Improvement',
            'description': f"Close gaps in top {len(priority_gaps)} attributes",
            'focus_attribute': 'Multiple',
            'improvement': _to_native(total_gap / len(priority_gaps)),
            'share_gain': _to_native(min(share_gain, 12)),
            'new_share': _to_native(current_share + min(share_gain, 12))
        })
    
    # Best scenario
    best = max(scenarios, key=lambda x: x['share_gain']) if scenarios else None
    
    # Recommendations
    recommendations = []
    if biggest_gap and biggest_gap['gap'] < 0:
        recommendations.append(f"Priority: Close gap in {biggest_gap['attribute']} (gap: {biggest_gap['gap']:.2f})")
    if loyalty and loyalty.get('top_driver'):
        recommendations.append(f"Leverage: Strengthen {loyalty['top_driver']['attribute']} to drive loyalty")
    recommendations.append("Monitor competitor movements and adjust strategy accordingly")
    
    return {
        'current_share': _to_native(current_share),
        'scenarios': scenarios,
        'best_scenario': best,
        'potential_share_gain': best['share_gain'] if best else 0,
        'recommendations': recommendations
    }


def create_simulation_chart(simulation: Dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    scenarios = simulation['scenarios']
    
    # Chart 1: Share comparison
    names = ['Current'] + [s['name'][:15] for s in scenarios]
    shares = [simulation['current_share']] + [s['new_share'] for s in scenarios]
    colors = ['#94a3b8'] + ['#3b82f6'] * len(scenarios)
    
    axes[0].bar(names, shares, color=colors, alpha=0.8, edgecolor='black')
    axes[0].set_ylabel('Market Share (%)')
    axes[0].set_title('Projected Market Share', fontsize=11, fontweight='bold')
    axes[0].set_xticklabels(names, rotation=45, ha='right')
    
    # Chart 2: Share gain
    names2 = [s['name'][:15] for s in scenarios]
    gains = [s['share_gain'] for s in scenarios]
    
    axes[1].barh(names2, gains, color='#10b981', alpha=0.8, edgecolor='black')
    axes[1].set_xlabel('Share Gain (%)')
    axes[1].set_title('Potential Share Gain by Scenario', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# Report Generation
def generate_report(awareness: Dict, comparison: Dict, loyalty: Optional[Dict], 
                    gap: Dict, simulation: Dict, our_brand: str) -> Dict:
    report = {}
    
    report['step1_awareness'] = {
        'title': '1. Awareness & Image Distribution',
        'question': 'How is our brand perceived?',
        'finding': f"Avg image score: {awareness['avg_image_score']:.2f}, Strongest: {awareness.get('strongest_attribute', 'N/A')}",
        'detail': f"Analysis of {awareness['n_our_brand']} responses for {our_brand} shows an average image score of {awareness['avg_image_score']:.2f}. "
                 f"The strongest attribute is '{awareness.get('strongest_attribute', 'N/A')}' while '{awareness.get('weakest_attribute', 'N/A')}' shows room for improvement. "
                 + (f"Brand awareness stands at {awareness['awareness_rate']:.0f}%." if awareness.get('awareness_rate') else "")
    }
    
    report['step2_comparison'] = {
        'title': '2. Competitive Comparison',
        'question': 'How do we compare to competitors?',
        'finding': f"Rank #{comparison['our_rank']} of {comparison['n_brands']} brands, Leader: {comparison['leader']}",
        'detail': f"{our_brand} ranks #{comparison['our_rank']} among {comparison['n_brands']} brands. "
                 f"Our competitive strength is in '{comparison['our_strength']}' while '{comparison['our_weakness']}' is where competitors outperform us. "
                 f"The market leader is '{comparison['leader']}'."
    }
    
    if loyalty and not loyalty.get('error'):
        top = loyalty.get('top_driver', {})
        report['step3_loyalty'] = {
            'title': '3. Image-Loyalty Relationship',
            'question': 'Does image drive loyalty?',
            'finding': f"Top driver: {top.get('attribute', 'N/A')} (r={top.get('correlation', 0):.3f}), R²={loyalty['r_squared']:.3f}",
            'detail': f"Image attributes explain {loyalty['r_squared']*100:.1f}% of loyalty variance. "
                     f"'{top.get('attribute', 'N/A')}' has the strongest impact on loyalty (r={top.get('correlation', 0):.3f}). "
                     f"{loyalty['n_significant']} attributes show statistically significant relationships."
        }
    else:
        report['step3_loyalty'] = {
            'title': '3. Image-Loyalty Relationship',
            'question': 'Does image drive loyalty?',
            'finding': 'Loyalty analysis requires loyalty column',
            'detail': 'Configure a loyalty metric to analyze the image-loyalty relationship.'
        }
    
    biggest = gap.get('biggest_gap', {})
    report['step4_gap'] = {
        'title': '4. Gap Analysis',
        'question': 'Where should we improve?',
        'finding': f"Biggest gap: {biggest.get('attribute', 'N/A')} ({biggest.get('gap', 0):.2f} vs {biggest.get('vs_brand', 'N/A')})",
        'detail': f"We trail competitors in {gap['n_gaps_behind']} attributes and lead in {gap['n_gaps_ahead']}. "
                 f"The largest gap is in '{biggest.get('attribute', 'N/A')}' where we score {biggest.get('our_score', 0):.2f} vs {biggest.get('best_score', 0):.2f} for {biggest.get('vs_brand', 'N/A')}. "
                 f"Priority improvement areas have been identified."
    }
    
    best = simulation.get('best_scenario', {})
    report['step5_simulation'] = {
        'title': '5. Market Share Simulation',
        'question': 'What if we improve our image?',
        'finding': f"Best scenario: +{best.get('share_gain', 0):.1f}% share ({best.get('name', 'N/A')})",
        'detail': f"Current estimated share is {simulation['current_share']:.1f}%. "
                 f"The best improvement scenario '{best.get('name', 'N/A')}' could increase share to {best.get('new_share', 0):.1f}%. "
                 f"Key recommendations: {'; '.join(simulation.get('recommendations', [])[:2])}"
    }
    
    return report


def generate_insights(awareness: Dict, comparison: Dict, loyalty: Optional[Dict], 
                      gap: Dict, simulation: Dict) -> List[Dict]:
    insights = []
    
    # Ranking insight
    if comparison['our_rank'] == 1:
        insights.append({'title': 'Market Leader', 'description': 'We rank #1 in overall brand image.', 'status': 'positive'})
    elif comparison['our_rank'] <= 2:
        insights.append({'title': 'Strong Position', 'description': f"Rank #{comparison['our_rank']} - close to market leader.", 'status': 'positive'})
    else:
        insights.append({'title': 'Improvement Needed', 'description': f"Rank #{comparison['our_rank']} - significant gap to leaders.", 'status': 'warning'})
    
    # Gap insight
    if gap['n_gaps_behind'] > gap['n_gaps_ahead']:
        insights.append({'title': 'Competitive Gap', 'description': f"Behind in {gap['n_gaps_behind']} attributes, ahead in {gap['n_gaps_ahead']}.", 'status': 'warning'})
    
    # Loyalty insight
    if loyalty and not loyalty.get('error') and loyalty.get('top_driver'):
        top = loyalty['top_driver']
        if top['correlation'] > 0.5:
            insights.append({'title': 'Strong Loyalty Driver', 'description': f"{top['attribute']} strongly drives loyalty (r={top['correlation']:.2f}).", 'status': 'positive'})
    
    # Simulation insight
    if simulation.get('best_scenario') and simulation['best_scenario'].get('share_gain', 0) > 3:
        insights.append({'title': 'Growth Potential', 'description': f"Can gain +{simulation['best_scenario']['share_gain']:.1f}% share with focused improvements.", 'status': 'positive'})
    
    return insights


@router.post("/brand-image")
async def analyze_brand_image(request: BrandRequest):
    try:
        df = pd.DataFrame(request.data)
        if len(df) < 20:
            raise HTTPException(status_code=400, detail="Need at least 20 records")
        
        if request.our_brand not in df[request.brand_col].values:
            raise HTTPException(status_code=400, detail=f"Brand '{request.our_brand}' not found in data")
        
        results, visualizations = {}, {}
        
        # Step 1: Awareness
        awareness = analyze_awareness(df, request.brand_col, request.image_cols, 
                                       request.our_brand, request.awareness_col)
        results['awareness'] = awareness
        visualizations['awareness_chart'] = create_awareness_chart(
            awareness, df, request.brand_col, request.image_cols, request.our_brand)
        
        # Step 2: Comparison
        comparison = analyze_comparison(df, request.brand_col, request.image_cols, request.our_brand)
        results['comparison'] = comparison
        visualizations['comparison_chart'] = create_comparison_chart(comparison, request.image_cols)
        
        # Step 3: Loyalty
        loyalty = None
        if request.loyalty_col:
            loyalty = analyze_loyalty(df, request.brand_col, request.image_cols, 
                                      request.our_brand, request.loyalty_col)
            results['loyalty'] = loyalty
            if not loyalty.get('error'):
                visualizations['loyalty_chart'] = create_loyalty_chart(
                    loyalty, df, request.brand_col, request.image_cols, 
                    request.our_brand, request.loyalty_col)
        
        # Step 4: Gap
        gap = analyze_gap(df, request.brand_col, request.image_cols, request.our_brand)
        results['gap'] = gap
        visualizations['gap_chart'] = create_gap_chart(gap, request.image_cols)
        
        # Step 5: Simulation
        simulation = simulate_share(df, request.brand_col, request.image_cols, 
                                    request.our_brand, gap, loyalty)
        results['simulation'] = simulation
        visualizations['simulation_chart'] = create_simulation_chart(simulation)
        
        report = generate_report(awareness, comparison, loyalty, gap, simulation, request.our_brand)
        insights = generate_insights(awareness, comparison, loyalty, gap, simulation)
        
        summary = {
            'n_respondents': awareness['n_respondents'],
            'brand_awareness': awareness.get('awareness_rate', 0) or 0,
            'avg_image_score': awareness['avg_image_score'],
            'loyalty_correlation': loyalty['overall_correlation'] if loyalty and not loyalty.get('error') else 0,
            'main_gap': gap['biggest_gap']['attribute'] if gap.get('biggest_gap') else None,
            'potential_share_gain': simulation.get('best_scenario', {}).get('share_gain', 0)
        }
        
        return {'success': True, 'results': results, 'visualizations': visualizations, 
                'report': report, 'key_insights': insights, 'summary': summary}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
