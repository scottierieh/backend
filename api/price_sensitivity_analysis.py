"""
Price Sensitivity & Resistance Analysis API
5-step framework for comprehensive price analysis
1. Price Acceptance Status (현황 파악) - 가격 수용도 분석
2. Segment Price Response (집단 비교) - 소득/성향별 가격 반응
3. Price-Brand Value Relationship (관계성/원인) - 가격과 브랜드 가치 관계
4. Price Resistance Threshold (심층 진단) - 심리적 가격 저항선
5. Revenue Optimization (최적화/예측) - 가격 조정 시 수익 변화
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
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class PriceSensitivityRequest(BaseModel):
    data: List[Dict[str, Any]]
    max_price_col: str  # Maximum acceptable price (too expensive)
    min_price_col: Optional[str] = None  # Minimum acceptable price (too cheap)
    ideal_price_col: Optional[str] = None  # Ideal/expected price
    purchase_intent_col: Optional[str] = None  # Purchase intent at current price
    segment_col: Optional[str] = None  # Income/segment column
    brand_value_col: Optional[str] = None  # Brand value perception
    current_price: Optional[float] = None  # Current market price


def _to_native(obj):
    if obj is None:
        return None
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj) if not np.isnan(obj) else None
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64


# =============================================================================
# Step 1: Price Acceptance Status (현황 파악)
# =============================================================================
def analyze_price_acceptance(df: pd.DataFrame, max_price_col: str,
                            min_price_col: Optional[str], ideal_price_col: Optional[str],
                            current_price: Optional[float]) -> Dict:
    max_prices = pd.to_numeric(df[max_price_col], errors='coerce').dropna()
    
    result = {
        'n_responses': len(max_prices),
        'max_price': {
            'mean': _to_native(max_prices.mean()),
            'median': _to_native(max_prices.median()),
            'std': _to_native(max_prices.std()),
            'min': _to_native(max_prices.min()),
            'max': _to_native(max_prices.max()),
            'q1': _to_native(max_prices.quantile(0.25)),
            'q3': _to_native(max_prices.quantile(0.75))
        }
    }
    
    # Min price (too cheap) analysis
    if min_price_col and min_price_col in df.columns:
        min_prices = pd.to_numeric(df[min_price_col], errors='coerce').dropna()
        result['min_price'] = {
            'mean': _to_native(min_prices.mean()),
            'median': _to_native(min_prices.median()),
            'std': _to_native(min_prices.std())
        }
        result['acceptable_range'] = {
            'lower': _to_native(min_prices.mean()),
            'upper': _to_native(max_prices.mean()),
            'spread': _to_native(max_prices.mean() - min_prices.mean())
        }
    
    # Ideal price analysis
    if ideal_price_col and ideal_price_col in df.columns:
        ideal_prices = pd.to_numeric(df[ideal_price_col], errors='coerce').dropna()
        result['ideal_price'] = {
            'mean': _to_native(ideal_prices.mean()),
            'median': _to_native(ideal_prices.median()),
            'std': _to_native(ideal_prices.std())
        }
    
    # Current price evaluation
    if current_price:
        acceptance_rate = (max_prices >= current_price).mean() * 100
        result['current_price_analysis'] = {
            'current_price': current_price,
            'acceptance_rate': _to_native(acceptance_rate),
            'above_max_pct': _to_native((current_price > max_prices).mean() * 100),
            'position': 'acceptable' if acceptance_rate > 70 else 'marginal' if acceptance_rate > 50 else 'too_high'
        }
    
    # Price distribution buckets
    price_range = max_prices.max() - max_prices.min()
    n_buckets = 5
    bucket_size = price_range / n_buckets
    buckets = []
    for i in range(n_buckets):
        lower = max_prices.min() + i * bucket_size
        upper = lower + bucket_size
        count = ((max_prices >= lower) & (max_prices < upper)).sum()
        buckets.append({
            'range': f"{lower:.0f}-{upper:.0f}",
            'lower': _to_native(lower),
            'upper': _to_native(upper),
            'count': _to_native(count),
            'pct': _to_native(count / len(max_prices) * 100)
        })
    result['distribution'] = buckets
    
    return result


def create_acceptance_chart(acc_data: Dict, current_price: Optional[float]) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Chart 1: Price distribution
    ax1 = axes[0]
    dist = acc_data.get('distribution', [])
    if dist:
        ranges = [d['range'] for d in dist]
        counts = [d['count'] for d in dist]
        colors = ['#3b82f6'] * len(dist)
        
        if current_price:
            for i, d in enumerate(dist):
                if d['lower'] <= current_price < d['upper']:
                    colors[i] = '#f59e0b'
        
        ax1.bar(ranges, counts, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Price Range')
        ax1.set_ylabel('Count')
        ax1.set_title('Max Acceptable Price Distribution', fontsize=11, fontweight='bold')
        ax1.tick_params(axis='x', rotation=45)
    
    # Chart 2: Price metrics summary
    ax2 = axes[1]
    metrics = []
    values = []
    colors = []
    
    if acc_data.get('min_price'):
        metrics.append('Too Cheap')
        values.append(acc_data['min_price']['mean'])
        colors.append('#ef4444')
    
    if acc_data.get('ideal_price'):
        metrics.append('Ideal')
        values.append(acc_data['ideal_price']['mean'])
        colors.append('#10b981')
    
    metrics.append('Max Accept')
    values.append(acc_data['max_price']['mean'])
    colors.append('#3b82f6')
    
    if current_price:
        metrics.append('Current')
        values.append(current_price)
        colors.append('#f59e0b')
    
    ax2.barh(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Price')
    ax2.set_title('Price Points Overview', fontsize=11, fontweight='bold')
    
    for i, v in enumerate(values):
        ax2.text(v + max(values) * 0.02, i, f'{v:.0f}', va='center')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Step 2: Segment Price Response (집단 비교)
# =============================================================================
def analyze_segment_response(df: pd.DataFrame, max_price_col: str, segment_col: str,
                            purchase_intent_col: Optional[str]) -> Dict:
    segments = df[segment_col].unique()
    segment_stats = []
    
    for seg in segments:
        seg_data = df[df[segment_col] == seg]
        max_prices = pd.to_numeric(seg_data[max_price_col], errors='coerce').dropna()
        
        seg_stat = {
            'segment': _to_native(seg),
            'n': len(seg_data),
            'max_price_mean': _to_native(max_prices.mean()),
            'max_price_median': _to_native(max_prices.median()),
            'max_price_std': _to_native(max_prices.std())
        }
        
        if purchase_intent_col and purchase_intent_col in df.columns:
            intent = pd.to_numeric(seg_data[purchase_intent_col], errors='coerce').dropna()
            seg_stat['purchase_intent_mean'] = _to_native(intent.mean())
            
            # Price sensitivity = std/mean (coefficient of variation)
            if max_prices.mean() > 0:
                seg_stat['price_sensitivity'] = _to_native(max_prices.std() / max_prices.mean())
            else:
                seg_stat['price_sensitivity'] = None
        
        segment_stats.append(seg_stat)
    
    # Sort by max price mean
    segment_stats = sorted(segment_stats, key=lambda x: x['max_price_mean'], reverse=True)
    
    # ANOVA test
    groups = [pd.to_numeric(df[df[segment_col] == seg][max_price_col], errors='coerce').dropna() 
              for seg in segments]
    groups = [g for g in groups if len(g) > 0]
    
    if len(groups) >= 2:
        try:
            f_stat, p_value = stats.f_oneway(*groups)
            significant = bool(p_value < 0.05)
        except:
            f_stat, p_value, significant = None, None, False
    else:
        f_stat, p_value, significant = None, None, False
    
    # Price elasticity proxy
    highest_seg = segment_stats[0] if segment_stats else None
    lowest_seg = segment_stats[-1] if segment_stats else None
    
    price_gap = None
    if highest_seg and lowest_seg:
        price_gap = highest_seg['max_price_mean'] - lowest_seg['max_price_mean']
    
    return {
        'segments': segment_stats,
        'n_segments': len(segment_stats),
        'anova': {
            'f_statistic': _to_native(f_stat),
            'p_value': _to_native(p_value),
            'significant': significant
        },
        'highest_acceptance': highest_seg,
        'lowest_acceptance': lowest_seg,
        'segment_price_gap': _to_native(price_gap)
    }


def create_segment_chart(seg_data: Dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    segments = seg_data.get('segments', [])
    
    # Chart 1: Max price by segment
    ax1 = axes[0]
    if segments:
        names = [str(s['segment'])[:15] for s in segments]
        means = [s['max_price_mean'] for s in segments]
        colors = ['#10b981' if i == 0 else '#ef4444' if i == len(segments)-1 else '#3b82f6' 
                  for i in range(len(segments))]
        
        ax1.barh(names, means, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Max Acceptable Price (Mean)')
        ax1.set_title('Price Acceptance by Segment', fontsize=11, fontweight='bold')
    
    # Chart 2: Price sensitivity by segment
    ax2 = axes[1]
    if segments and segments[0].get('price_sensitivity') is not None:
        sensitivities = [s.get('price_sensitivity', 0) or 0 for s in segments]
        colors = ['#ef4444' if s > 0.3 else '#f59e0b' if s > 0.2 else '#10b981' for s in sensitivities]
        
        ax2.barh(names, sensitivities, color=colors, alpha=0.7, edgecolor='black')
        ax2.axvline(x=0.25, color='gray', linestyle='--', alpha=0.5, label='High sensitivity')
        ax2.set_xlabel('Price Sensitivity (CV)')
        ax2.set_title('Price Sensitivity by Segment', fontsize=11, fontweight='bold')
        ax2.legend()
    else:
        # Alternative: show purchase intent
        if segments[0].get('purchase_intent_mean') is not None:
            intents = [s.get('purchase_intent_mean', 0) for s in segments]
            ax2.barh(names, intents, color='#3b82f6', alpha=0.7, edgecolor='black')
            ax2.set_xlabel('Purchase Intent (Mean)')
            ax2.set_title('Purchase Intent by Segment', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Step 3: Price-Brand Value Relationship (관계성/원인)
# =============================================================================
def analyze_price_brand_relationship(df: pd.DataFrame, max_price_col: str,
                                     brand_value_col: str, purchase_intent_col: Optional[str]) -> Dict:
    df_clean = df[[max_price_col, brand_value_col]].apply(pd.to_numeric, errors='coerce').dropna()
    
    if len(df_clean) < 10:
        return {'error': 'Insufficient data'}
    
    max_prices = df_clean[max_price_col]
    brand_values = df_clean[brand_value_col]
    
    # Correlation
    corr, p_value = stats.pearsonr(max_prices, brand_values)
    
    # Regression
    slope, intercept, r_value, p_val, std_err = stats.linregress(brand_values, max_prices)
    
    result = {
        'correlation': _to_native(corr),
        'p_value': _to_native(p_value),
        'significant': bool(p_value < 0.05),
        'r_squared': _to_native(r_value ** 2),
        'regression': {
            'slope': _to_native(slope),
            'intercept': _to_native(intercept),
            'interpretation': f"1 unit increase in brand value = {slope:.2f} price acceptance increase"
        },
        'relationship_strength': 'strong' if abs(corr) > 0.5 else 'moderate' if abs(corr) > 0.3 else 'weak',
        'n_observations': len(df_clean)
    }
    
    # Brand value segments
    q33, q66 = brand_values.quantile([0.33, 0.66])
    brand_segments = {
        'low_brand': max_prices[brand_values <= q33].mean(),
        'mid_brand': max_prices[(brand_values > q33) & (brand_values <= q66)].mean(),
        'high_brand': max_prices[brand_values > q66].mean()
    }
    result['price_by_brand_level'] = {k: _to_native(v) for k, v in brand_segments.items()}
    
    # Brand premium
    if brand_segments['low_brand'] > 0:
        brand_premium = (brand_segments['high_brand'] - brand_segments['low_brand']) / brand_segments['low_brand'] * 100
        result['brand_premium_pct'] = _to_native(brand_premium)
    
    # Purchase intent relationship if available
    if purchase_intent_col and purchase_intent_col in df.columns:
        df_intent = df[[max_price_col, purchase_intent_col]].apply(pd.to_numeric, errors='coerce').dropna()
        if len(df_intent) >= 10:
            intent_corr, intent_p = stats.pearsonr(df_intent[max_price_col], df_intent[purchase_intent_col])
            result['price_intent_correlation'] = _to_native(intent_corr)
            result['price_intent_significant'] = bool(intent_p < 0.05)
    
    return result


def create_relationship_chart(rel_data: Dict, df: pd.DataFrame, max_price_col: str, brand_value_col: str) -> str:
    if rel_data.get('error'):
        return ""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Chart 1: Scatter plot with regression
    ax1 = axes[0]
    df_clean = df[[max_price_col, brand_value_col]].apply(pd.to_numeric, errors='coerce').dropna()
    
    ax1.scatter(df_clean[brand_value_col], df_clean[max_price_col], alpha=0.5, color='#3b82f6')
    
    # Regression line
    reg = rel_data.get('regression', {})
    if reg:
        x_line = np.linspace(df_clean[brand_value_col].min(), df_clean[brand_value_col].max(), 100)
        y_line = reg['slope'] * x_line + reg['intercept']
        ax1.plot(x_line, y_line, color='#ef4444', linestyle='--', linewidth=2,
                label=f"r={rel_data['correlation']:.3f}")
    
    ax1.set_xlabel('Brand Value Perception')
    ax1.set_ylabel('Max Acceptable Price')
    ax1.set_title('Price-Brand Value Relationship', fontsize=11, fontweight='bold')
    ax1.legend()
    
    # Chart 2: Price by brand level
    ax2 = axes[1]
    brand_levels = rel_data.get('price_by_brand_level', {})
    if brand_levels:
        levels = ['Low Brand', 'Mid Brand', 'High Brand']
        prices = [brand_levels['low_brand'], brand_levels['mid_brand'], brand_levels['high_brand']]
        colors = ['#ef4444', '#f59e0b', '#10b981']
        
        ax2.bar(levels, prices, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('Avg Max Acceptable Price')
        ax2.set_title('Price Acceptance by Brand Level', fontsize=11, fontweight='bold')
        
        # Add brand premium annotation
        if rel_data.get('brand_premium_pct'):
            ax2.annotate(f"+{rel_data['brand_premium_pct']:.1f}%\nBrand Premium",
                        xy=(2, prices[2]), xytext=(2.3, prices[2] * 0.9),
                        fontsize=9, color='#10b981')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Step 4: Price Resistance Threshold (심층 진단)
# =============================================================================
def analyze_price_resistance(df: pd.DataFrame, max_price_col: str,
                            min_price_col: Optional[str], ideal_price_col: Optional[str],
                            current_price: Optional[float]) -> Dict:
    max_prices = pd.to_numeric(df[max_price_col], errors='coerce').dropna()
    
    # Van Westendorp Price Sensitivity Meter approach
    price_points = np.linspace(max_prices.min() * 0.8, max_prices.max() * 1.1, 50)
    
    # "Too expensive" curve - cumulative % who find price too high
    too_expensive = [(max_prices < p).mean() * 100 for p in price_points]
    
    # "Not too expensive" curve
    not_too_expensive = [100 - te for te in too_expensive]
    
    result = {
        'price_points': [_to_native(p) for p in price_points],
        'too_expensive_curve': [_to_native(te) for te in too_expensive],
        'not_too_expensive_curve': [_to_native(nte) for nte in not_too_expensive]
    }
    
    # Find resistance thresholds
    # Point where 50% find it too expensive
    idx_50 = np.argmin(np.abs(np.array(too_expensive) - 50))
    result['resistance_50pct'] = _to_native(price_points[idx_50])
    
    # Point where 75% find it too expensive
    idx_75 = np.argmin(np.abs(np.array(too_expensive) - 75))
    result['resistance_75pct'] = _to_native(price_points[idx_75])
    
    # Point where 25% find it too expensive (75% accept)
    idx_25 = np.argmin(np.abs(np.array(too_expensive) - 25))
    result['acceptance_75pct'] = _to_native(price_points[idx_25])
    
    # Van Westendorp analysis if min_price available
    if min_price_col and min_price_col in df.columns:
        min_prices = pd.to_numeric(df[min_price_col], errors='coerce').dropna()
        
        # "Too cheap" curve
        too_cheap = [(min_prices > p).mean() * 100 for p in price_points]
        result['too_cheap_curve'] = [_to_native(tc) for tc in too_cheap]
        
        # Point of Marginal Cheapness (PMC): too_cheap crosses not_too_expensive
        pmc_idx = np.argmin(np.abs(np.array(too_cheap) - np.array(not_too_expensive)))
        result['point_marginal_cheapness'] = _to_native(price_points[pmc_idx])
        
        # Point of Marginal Expensiveness (PME): too_expensive crosses not_too_cheap
        not_too_cheap = [100 - tc for tc in too_cheap]
        pme_idx = np.argmin(np.abs(np.array(too_expensive) - np.array(not_too_cheap)))
        result['point_marginal_expensiveness'] = _to_native(price_points[pme_idx])
        
        # Optimal Price Point (OPP): too_cheap crosses too_expensive
        opp_idx = np.argmin(np.abs(np.array(too_cheap) - np.array(too_expensive)))
        result['optimal_price_point'] = _to_native(price_points[opp_idx])
        
        # Acceptable price range
        result['acceptable_range'] = {
            'lower': result['point_marginal_cheapness'],
            'upper': result['point_marginal_expensiveness'],
            'optimal': result['optimal_price_point']
        }
    
    # Psychological price points
    psychological_points = []
    common_endings = [9, 99, 95, 90, 0]
    
    for price in [max_prices.mean(), max_prices.median(), result.get('resistance_50pct', max_prices.mean())]:
        base = int(price)
        for ending in [9, 99, 95]:
            if base > 100:
                psych_price = (base // 100) * 100 + ending
            else:
                psych_price = (base // 10) * 10 + (ending % 10)
            
            if psych_price not in [p['price'] for p in psychological_points]:
                acceptance = (max_prices >= psych_price).mean() * 100
                psychological_points.append({
                    'price': _to_native(psych_price),
                    'acceptance_rate': _to_native(acceptance)
                })
    
    psychological_points = sorted(psychological_points, key=lambda x: x['price'])[:5]
    result['psychological_prices'] = psychological_points
    
    # Current price position
    if current_price:
        current_acceptance = (max_prices >= current_price).mean() * 100
        result['current_price_position'] = {
            'price': current_price,
            'acceptance_rate': _to_native(current_acceptance),
            'vs_optimal': _to_native(current_price - result.get('optimal_price_point', current_price)),
            'position': 'optimal' if abs(current_price - result.get('optimal_price_point', current_price)) < max_prices.std() * 0.5 
                       else 'below_optimal' if current_price < result.get('optimal_price_point', current_price)
                       else 'above_optimal'
        }
    
    return result


def create_resistance_chart(res_data: Dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    price_points = res_data.get('price_points', [])
    too_expensive = res_data.get('too_expensive_curve', [])
    
    # Chart 1: Price sensitivity curves
    ax1 = axes[0]
    if price_points and too_expensive:
        ax1.plot(price_points, too_expensive, 'r-', linewidth=2, label='Too Expensive')
        ax1.plot(price_points, res_data.get('not_too_expensive_curve', []), 'g-', linewidth=2, label='Not Too Expensive')
        
        if res_data.get('too_cheap_curve'):
            ax1.plot(price_points, res_data['too_cheap_curve'], 'b-', linewidth=2, label='Too Cheap')
        
        # Mark key points
        if res_data.get('resistance_50pct'):
            ax1.axvline(x=res_data['resistance_50pct'], color='orange', linestyle='--', alpha=0.7, label=f"50% Resistance: {res_data['resistance_50pct']:.0f}")
        
        if res_data.get('optimal_price_point'):
            ax1.axvline(x=res_data['optimal_price_point'], color='purple', linestyle='--', alpha=0.7, label=f"Optimal: {res_data['optimal_price_point']:.0f}")
        
        ax1.set_xlabel('Price')
        ax1.set_ylabel('Cumulative %')
        ax1.set_title('Price Sensitivity Curves', fontsize=11, fontweight='bold')
        ax1.legend(loc='best', fontsize=8)
        ax1.set_ylim(0, 100)
    
    # Chart 2: Key thresholds
    ax2 = axes[1]
    thresholds = []
    values = []
    colors = []
    
    if res_data.get('acceptance_75pct'):
        thresholds.append('75% Accept')
        values.append(res_data['acceptance_75pct'])
        colors.append('#10b981')
    
    if res_data.get('optimal_price_point'):
        thresholds.append('Optimal')
        values.append(res_data['optimal_price_point'])
        colors.append('#8b5cf6')
    
    if res_data.get('resistance_50pct'):
        thresholds.append('50% Resist')
        values.append(res_data['resistance_50pct'])
        colors.append('#f59e0b')
    
    if res_data.get('resistance_75pct'):
        thresholds.append('75% Resist')
        values.append(res_data['resistance_75pct'])
        colors.append('#ef4444')
    
    if thresholds:
        ax2.barh(thresholds, values, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Price')
        ax2.set_title('Price Resistance Thresholds', fontsize=11, fontweight='bold')
        
        for i, v in enumerate(values):
            ax2.text(v + max(values) * 0.02, i, f'{v:.0f}', va='center')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Step 5: Revenue Optimization (최적화/예측)
# =============================================================================
def simulate_revenue_optimization(df: pd.DataFrame, max_price_col: str,
                                  current_price: Optional[float], purchase_intent_col: Optional[str]) -> Dict:
    max_prices = pd.to_numeric(df[max_price_col], errors='coerce').dropna()
    
    # Generate price scenarios
    price_range = np.linspace(max_prices.min() * 0.8, max_prices.max(), 20)
    
    simulations = []
    for price in price_range:
        # Demand estimation: % who would accept this price
        acceptance_rate = (max_prices >= price).mean()
        
        # Revenue index = price × acceptance_rate (normalized)
        revenue_index = price * acceptance_rate
        
        simulations.append({
            'price': _to_native(price),
            'acceptance_rate': _to_native(acceptance_rate * 100),
            'demand_index': _to_native(acceptance_rate * 100),
            'revenue_index': _to_native(revenue_index)
        })
    
    # Find optimal price (max revenue)
    optimal_idx = np.argmax([s['revenue_index'] for s in simulations])
    optimal_price = simulations[optimal_idx]['price']
    optimal_revenue = simulations[optimal_idx]['revenue_index']
    
    result = {
        'simulations': simulations,
        'optimal_price': _to_native(optimal_price),
        'optimal_revenue_index': _to_native(optimal_revenue),
        'optimal_acceptance': _to_native(simulations[optimal_idx]['acceptance_rate'])
    }
    
    # Current vs optimal comparison
    if current_price:
        current_acceptance = (max_prices >= current_price).mean()
        current_revenue = current_price * current_acceptance
        
        result['current_analysis'] = {
            'price': current_price,
            'acceptance_rate': _to_native(current_acceptance * 100),
            'revenue_index': _to_native(current_revenue)
        }
        
        # Revenue change scenarios
        price_changes = [-20, -10, -5, 0, 5, 10, 20]
        scenarios = []
        
        for pct_change in price_changes:
            new_price = current_price * (1 + pct_change / 100)
            new_acceptance = (max_prices >= new_price).mean()
            new_revenue = new_price * new_acceptance
            revenue_change = (new_revenue - current_revenue) / current_revenue * 100 if current_revenue > 0 else 0
            
            scenarios.append({
                'price_change_pct': pct_change,
                'new_price': _to_native(new_price),
                'new_acceptance_pct': _to_native(new_acceptance * 100),
                'revenue_change_pct': _to_native(revenue_change),
                'recommendation': 'optimal' if abs(new_price - optimal_price) < max_prices.std() * 0.3 else 
                                 'consider' if revenue_change > 0 else 'avoid'
            })
        
        result['scenarios'] = scenarios
        
        # Gap to optimal
        result['gap_to_optimal'] = {
            'price_diff': _to_native(optimal_price - current_price),
            'price_diff_pct': _to_native((optimal_price - current_price) / current_price * 100),
            'potential_revenue_gain': _to_native((optimal_revenue - current_revenue) / current_revenue * 100) if current_revenue > 0 else 0
        }
    
    # Price elasticity estimate
    # Using arc elasticity between two points
    mid_idx = len(simulations) // 2
    if mid_idx > 0 and mid_idx < len(simulations) - 1:
        p1, q1 = simulations[mid_idx - 2]['price'], simulations[mid_idx - 2]['acceptance_rate']
        p2, q2 = simulations[mid_idx + 2]['price'], simulations[mid_idx + 2]['acceptance_rate']
        
        if p1 != p2 and (q1 + q2) > 0:
            elasticity = ((q2 - q1) / ((q1 + q2) / 2)) / ((p2 - p1) / ((p1 + p2) / 2))
            result['price_elasticity'] = _to_native(elasticity)
            result['elasticity_interpretation'] = 'elastic' if abs(elasticity) > 1 else 'inelastic'
    
    return result


def create_optimization_chart(opt_data: Dict, current_price: Optional[float]) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    simulations = opt_data.get('simulations', [])
    
    # Chart 1: Revenue curve
    ax1 = axes[0]
    if simulations:
        prices = [s['price'] for s in simulations]
        revenues = [s['revenue_index'] for s in simulations]
        acceptances = [s['acceptance_rate'] for s in simulations]
        
        ax1_twin = ax1.twinx()
        
        line1, = ax1.plot(prices, revenues, 'b-', linewidth=2, label='Revenue Index')
        line2, = ax1_twin.plot(prices, acceptances, 'g--', linewidth=2, label='Acceptance %')
        
        # Mark optimal
        if opt_data.get('optimal_price'):
            ax1.axvline(x=opt_data['optimal_price'], color='purple', linestyle='--', alpha=0.7)
            ax1.scatter([opt_data['optimal_price']], [opt_data['optimal_revenue_index']], 
                       color='purple', s=100, zorder=5, label=f"Optimal: {opt_data['optimal_price']:.0f}")
        
        # Mark current
        if current_price:
            ax1.axvline(x=current_price, color='orange', linestyle='--', alpha=0.7)
        
        ax1.set_xlabel('Price')
        ax1.set_ylabel('Revenue Index', color='blue')
        ax1_twin.set_ylabel('Acceptance Rate (%)', color='green')
        ax1.set_title('Price Optimization Curve', fontsize=11, fontweight='bold')
        ax1.legend(loc='upper right')
    
    # Chart 2: Scenario comparison
    ax2 = axes[1]
    scenarios = opt_data.get('scenarios', [])
    if scenarios:
        changes = [f"{s['price_change_pct']:+d}%" for s in scenarios]
        revenue_changes = [s['revenue_change_pct'] for s in scenarios]
        colors = ['#10b981' if r > 0 else '#ef4444' if r < 0 else '#9ca3af' for r in revenue_changes]
        
        ax2.bar(changes, revenue_changes, color=colors, alpha=0.7, edgecolor='black')
        ax2.axhline(y=0, color='gray', linestyle='-', linewidth=1)
        ax2.set_xlabel('Price Change')
        ax2.set_ylabel('Revenue Change (%)')
        ax2.set_title('Revenue Impact by Price Change', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Report & Insights
# =============================================================================
def generate_report(acc: Dict, seg: Optional[Dict], rel: Optional[Dict], 
                   res: Dict, opt: Dict, current_price: Optional[float]) -> Dict:
    report = {}
    
    report['step1_acceptance'] = {
        'title': '1. Price Acceptance Status',
        'question': 'What is the current price acceptance level?',
        'finding': f"Max acceptable price: avg {acc['max_price']['mean']:.0f}, median {acc['max_price']['median']:.0f}",
        'detail': f"Analysis of {acc['n_responses']} responses shows max acceptable price ranges from {acc['max_price']['min']:.0f} to {acc['max_price']['max']:.0f}. "
                 + (f"At current price {current_price:.0f}, acceptance rate is {acc['current_price_analysis']['acceptance_rate']:.1f}%." if acc.get('current_price_analysis') else "")
    }
    
    if seg and not seg.get('error'):
        report['step2_segment'] = {
            'title': '2. Segment Price Response',
            'question': 'How do segments differ in price acceptance?',
            'finding': f"{seg['n_segments']} segments, price gap of {seg['segment_price_gap']:.0f} between highest/lowest",
            'detail': f"Highest acceptance: {seg['highest_acceptance']['segment']} ({seg['highest_acceptance']['max_price_mean']:.0f}). "
                     f"Lowest: {seg['lowest_acceptance']['segment']} ({seg['lowest_acceptance']['max_price_mean']:.0f}). "
                     f"Difference is {'significant' if seg['anova']['significant'] else 'not significant'} (p={seg['anova']['p_value']:.4f})."
        }
    else:
        report['step2_segment'] = {
            'title': '2. Segment Price Response',
            'question': 'How do segments differ in price acceptance?',
            'finding': 'Segment analysis not performed',
            'detail': seg.get('error', 'No segment column specified.')
        }
    
    if rel and not rel.get('error'):
        report['step3_relationship'] = {
            'title': '3. Price-Brand Value Relationship',
            'question': 'How does brand perception affect price acceptance?',
            'finding': f"Correlation r={rel['correlation']:.3f} ({rel['relationship_strength']}), brand premium {rel.get('brand_premium_pct', 0):.1f}%",
            'detail': f"Brand value shows {rel['relationship_strength']} {'positive' if rel['correlation'] > 0 else 'negative'} relationship with price acceptance. "
                     f"High brand perception customers accept {rel['price_by_brand_level']['high_brand']:.0f} vs {rel['price_by_brand_level']['low_brand']:.0f} for low."
        }
    else:
        report['step3_relationship'] = {
            'title': '3. Price-Brand Value Relationship',
            'question': 'How does brand perception affect price acceptance?',
            'finding': 'Relationship analysis not performed',
            'detail': rel.get('error', 'No brand value column specified.')
        }
    
    report['step4_resistance'] = {
        'title': '4. Price Resistance Threshold',
        'question': 'Where are the psychological price barriers?',
        'finding': f"50% resistance at {res['resistance_50pct']:.0f}, 75% resistance at {res['resistance_75pct']:.0f}",
        'detail': f"Price sensitivity analysis shows 50% of customers resist at {res['resistance_50pct']:.0f} and 75% at {res['resistance_75pct']:.0f}. "
                 + (f"Optimal price point identified at {res['optimal_price_point']:.0f}." if res.get('optimal_price_point') else "")
    }
    
    report['step5_optimization'] = {
        'title': '5. Revenue Optimization',
        'question': 'What price maximizes revenue?',
        'finding': f"Optimal price: {opt['optimal_price']:.0f} with {opt['optimal_acceptance']:.1f}% acceptance",
        'detail': f"Revenue optimization suggests {opt['optimal_price']:.0f} as optimal price point. "
                 + (f"Current price is {opt['gap_to_optimal']['price_diff_pct']:+.1f}% from optimal with {opt['gap_to_optimal']['potential_revenue_gain']:.1f}% revenue gain potential." 
                    if opt.get('gap_to_optimal') else "")
    }
    
    return report


def generate_insights(acc: Dict, seg: Optional[Dict], rel: Optional[Dict],
                     res: Dict, opt: Dict) -> List[Dict]:
    insights = []
    
    # Acceptance insight
    if acc.get('current_price_analysis'):
        if acc['current_price_analysis']['acceptance_rate'] < 50:
            insights.append({
                'title': 'Low Price Acceptance',
                'description': f"Only {acc['current_price_analysis']['acceptance_rate']:.1f}% accept current price. Consider reduction.",
                'status': 'warning'
            })
        elif acc['current_price_analysis']['acceptance_rate'] > 80:
            insights.append({
                'title': 'High Price Acceptance',
                'description': f"{acc['current_price_analysis']['acceptance_rate']:.1f}% accept current price. Room for increase.",
                'status': 'positive'
            })
    
    # Segment insight
    if seg and seg.get('segment_price_gap') and seg['segment_price_gap'] > acc['max_price']['mean'] * 0.2:
        insights.append({
            'title': 'Significant Segment Difference',
            'description': f"Price acceptance varies {seg['segment_price_gap']:.0f} between segments. Consider tiered pricing.",
            'status': 'positive'
        })
    
    # Brand insight
    if rel and not rel.get('error') and rel.get('brand_premium_pct', 0) > 20:
        insights.append({
            'title': 'Strong Brand Premium',
            'description': f"Brand value drives {rel['brand_premium_pct']:.1f}% price premium. Invest in brand building.",
            'status': 'positive'
        })
    
    # Optimization insight
    if opt.get('gap_to_optimal') and abs(opt['gap_to_optimal']['price_diff_pct']) > 10:
        direction = 'increase' if opt['gap_to_optimal']['price_diff'] > 0 else 'decrease'
        insights.append({
            'title': 'Price Optimization Opportunity',
            'description': f"Optimal price is {abs(opt['gap_to_optimal']['price_diff_pct']):.1f}% {direction}. Potential {opt['gap_to_optimal']['potential_revenue_gain']:.1f}% revenue gain.",
            'status': 'warning' if direction == 'decrease' else 'positive'
        })
    
    return insights


# =============================================================================
# Main API Endpoint
# =============================================================================
@router.post("/price-sensitivity-analysis")
async def analyze_price_sensitivity(request: PriceSensitivityRequest):
    try:
        df = pd.DataFrame(request.data)
        max_price_col = request.max_price_col
        min_price_col = request.min_price_col
        ideal_price_col = request.ideal_price_col
        purchase_intent_col = request.purchase_intent_col
        segment_col = request.segment_col
        brand_value_col = request.brand_value_col
        current_price = request.current_price
        
        if len(df) < 10:
            raise HTTPException(status_code=400, detail="Need at least 10 responses")
        
        # Convert columns to numeric
        df[max_price_col] = pd.to_numeric(df[max_price_col], errors='coerce')
        
        results = {}
        visualizations = {}
        
        # Step 1: Price Acceptance
        acc = analyze_price_acceptance(df, max_price_col, min_price_col, ideal_price_col, current_price)
        results['acceptance'] = acc
        visualizations['acceptance_chart'] = create_acceptance_chart(acc, current_price)
        
        # Step 2: Segment Response
        seg = None
        if segment_col and segment_col in df.columns:
            seg = analyze_segment_response(df, max_price_col, segment_col, purchase_intent_col)
            results['segment'] = seg
            visualizations['segment_chart'] = create_segment_chart(seg)
        
        # Step 3: Price-Brand Relationship
        rel = None
        if brand_value_col and brand_value_col in df.columns:
            df[brand_value_col] = pd.to_numeric(df[brand_value_col], errors='coerce')
            rel = analyze_price_brand_relationship(df, max_price_col, brand_value_col, purchase_intent_col)
            results['relationship'] = rel
            if not rel.get('error'):
                visualizations['relationship_chart'] = create_relationship_chart(rel, df, max_price_col, brand_value_col)
        
        # Step 4: Price Resistance
        res = analyze_price_resistance(df, max_price_col, min_price_col, ideal_price_col, current_price)
        results['resistance'] = res
        visualizations['resistance_chart'] = create_resistance_chart(res)
        
        # Step 5: Revenue Optimization
        opt = simulate_revenue_optimization(df, max_price_col, current_price, purchase_intent_col)
        results['optimization'] = opt
        visualizations['optimization_chart'] = create_optimization_chart(opt, current_price)
        
        report = generate_report(acc, seg, rel, res, opt, current_price)
        insights = generate_insights(acc, seg, rel, res, opt)
        
        summary = {
            'n_responses': acc['n_responses'],
            'max_price_avg': acc['max_price']['mean'],
            'optimal_price': opt['optimal_price'],
            'current_acceptance': acc['current_price_analysis']['acceptance_rate'] if acc.get('current_price_analysis') else None,
            'resistance_50pct': res['resistance_50pct']
        }
        
        return {
            'success': True,
            'results': results,
            'visualizations': visualizations,
            'report': report,
            'key_insights': insights,
            'summary': summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
