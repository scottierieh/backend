"""
Demand & Response Elasticity Analysis API
5-step framework for comprehensive elasticity analysis
1. Response Status (현황 파악) - 조건별 반응 현황
2. Promotion Comparison (집단 비교) - 프로모션 유/무 비교
3. Condition-Demand Relationship (관계성/원인) - 조건 변화와 수요 관계
4. Elasticity Analysis (심층 진단) - 탄력성/반응도 분석
5. Performance Simulation (최적화/예측) - 조건 변경 시 성과
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
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

router = APIRouter()


class ElasticityRequest(BaseModel):
    data: List[Dict[str, Any]]
    demand_col: str  # Demand/Response variable (sales, clicks, conversions)
    condition_cols: List[str]  # Condition variables (price, promotion, etc.)
    promotion_col: Optional[str] = None  # Binary promotion indicator
    time_col: Optional[str] = None  # Time period column
    segment_col: Optional[str] = None  # Segment column


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
# Step 1: Response Status (현황 파악)
# =============================================================================
def analyze_response_status(df: pd.DataFrame, demand_col: str, condition_cols: List[str]) -> Dict:
    demand = pd.to_numeric(df[demand_col], errors='coerce').dropna()
    
    result = {
        'n_observations': len(demand),
        'demand': {
            'mean': _to_native(demand.mean()),
            'median': _to_native(demand.median()),
            'std': _to_native(demand.std()),
            'min': _to_native(demand.min()),
            'max': _to_native(demand.max()),
            'cv': _to_native(demand.std() / demand.mean() * 100) if demand.mean() > 0 else None
        }
    }
    
    # Condition-wise response summary
    condition_stats = []
    for col in condition_cols:
        values = pd.to_numeric(df[col], errors='coerce').dropna()
        
        # Create condition buckets (Low, Medium, High)
        q33, q66 = values.quantile([0.33, 0.66])
        
        low_demand = demand[values <= q33].mean()
        mid_demand = demand[(values > q33) & (values <= q66)].mean()
        high_demand = demand[values > q66].mean()
        
        condition_stats.append({
            'condition': col,
            'mean': _to_native(values.mean()),
            'std': _to_native(values.std()),
            'demand_at_low': _to_native(low_demand),
            'demand_at_mid': _to_native(mid_demand),
            'demand_at_high': _to_native(high_demand),
            'demand_change': _to_native(high_demand - low_demand),
            'demand_change_pct': _to_native((high_demand - low_demand) / low_demand * 100) if low_demand > 0 else None
        })
    
    result['conditions'] = condition_stats
    result['n_conditions'] = len(condition_stats)
    
    # Identify most impactful condition
    if condition_stats:
        most_impact = max(condition_stats, key=lambda x: abs(x['demand_change_pct'] or 0))
        result['most_impactful'] = most_impact
    
    return result


def create_status_chart(status_data: Dict, df: pd.DataFrame, demand_col: str, condition_cols: List[str]) -> str:
    n_cols = min(len(condition_cols), 3)
    fig, axes = plt.subplots(1, n_cols + 1, figsize=(4 * (n_cols + 1), 4))
    if n_cols == 0:
        axes = [axes]
    
    # Chart 1: Demand distribution
    ax1 = axes[0]
    demand = pd.to_numeric(df[demand_col], errors='coerce').dropna()
    ax1.hist(demand, bins=20, color='#3b82f6', alpha=0.7, edgecolor='black')
    ax1.axvline(demand.mean(), color='#ef4444', linestyle='--', linewidth=2, label=f'Mean: {demand.mean():.1f}')
    ax1.set_xlabel(demand_col)
    ax1.set_ylabel('Frequency')
    ax1.set_title('Demand Distribution', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=8)
    
    # Charts 2+: Demand by condition level
    for i, col in enumerate(condition_cols[:n_cols]):
        ax = axes[i + 1]
        values = pd.to_numeric(df[col], errors='coerce')
        q33, q66 = values.quantile([0.33, 0.66])
        
        levels = ['Low', 'Mid', 'High']
        demands = [
            demand[values <= q33].mean(),
            demand[(values > q33) & (values <= q66)].mean(),
            demand[values > q66].mean()
        ]
        colors = ['#3b82f6', '#f59e0b', '#10b981']
        
        ax.bar(levels, demands, color=colors, alpha=0.7, edgecolor='black')
        ax.set_xlabel(f'{col[:15]} Level')
        ax.set_ylabel('Avg Demand')
        ax.set_title(f'Demand by {col[:15]}', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Step 2: Promotion Comparison (집단 비교)
# =============================================================================
def analyze_promotion_comparison(df: pd.DataFrame, demand_col: str, promotion_col: str,
                                 segment_col: Optional[str] = None) -> Dict:
    df_clean = df[[demand_col, promotion_col]].copy()
    df_clean[demand_col] = pd.to_numeric(df_clean[demand_col], errors='coerce')
    
    # Handle promotion column (binary or convert)
    promo_values = df[promotion_col].unique()
    if len(promo_values) == 2:
        # Binary promotion
        promo_map = {v: i for i, v in enumerate(sorted(promo_values))}
        df_clean['promo_binary'] = df[promotion_col].map(promo_map)
    else:
        # Numeric - convert to binary by median
        df_clean['promo_binary'] = (pd.to_numeric(df[promotion_col], errors='coerce') > 
                                     pd.to_numeric(df[promotion_col], errors='coerce').median()).astype(int)
    
    df_clean = df_clean.dropna()
    
    promo_off = df_clean[df_clean['promo_binary'] == 0][demand_col]
    promo_on = df_clean[df_clean['promo_binary'] == 1][demand_col]
    
    # T-test
    if len(promo_off) > 0 and len(promo_on) > 0:
        t_stat, p_value = stats.ttest_ind(promo_off, promo_on)
        significant = bool(p_value < 0.05)
    else:
        t_stat, p_value, significant = None, None, False
    
    # Effect calculation
    lift = (promo_on.mean() - promo_off.mean()) / promo_off.mean() * 100 if promo_off.mean() > 0 else 0
    
    # Cohen's d effect size
    pooled_std = np.sqrt((promo_off.std()**2 + promo_on.std()**2) / 2)
    cohens_d = (promo_on.mean() - promo_off.mean()) / pooled_std if pooled_std > 0 else 0
    
    result = {
        'promo_off': {
            'n': len(promo_off),
            'mean': _to_native(promo_off.mean()),
            'std': _to_native(promo_off.std()),
            'median': _to_native(promo_off.median())
        },
        'promo_on': {
            'n': len(promo_on),
            'mean': _to_native(promo_on.mean()),
            'std': _to_native(promo_on.std()),
            'median': _to_native(promo_on.median())
        },
        'comparison': {
            'lift': _to_native(lift),
            'absolute_diff': _to_native(promo_on.mean() - promo_off.mean()),
            't_statistic': _to_native(t_stat),
            'p_value': _to_native(p_value),
            'significant': significant,
            'cohens_d': _to_native(cohens_d),
            'effect_size': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'
        }
    }
    
    # Segment-wise comparison
    if segment_col and segment_col in df.columns:
        segment_comparison = []
        for seg in df[segment_col].unique():
            seg_data = df_clean[df[segment_col] == seg]
            seg_off = seg_data[seg_data['promo_binary'] == 0][demand_col]
            seg_on = seg_data[seg_data['promo_binary'] == 1][demand_col]
            
            if len(seg_off) > 0 and len(seg_on) > 0:
                seg_lift = (seg_on.mean() - seg_off.mean()) / seg_off.mean() * 100 if seg_off.mean() > 0 else 0
                segment_comparison.append({
                    'segment': _to_native(seg),
                    'promo_off_mean': _to_native(seg_off.mean()),
                    'promo_on_mean': _to_native(seg_on.mean()),
                    'lift': _to_native(seg_lift)
                })
        
        segment_comparison = sorted(segment_comparison, key=lambda x: x['lift'], reverse=True)
        result['segment_comparison'] = segment_comparison
        result['best_response_segment'] = segment_comparison[0] if segment_comparison else None
    
    return result


def create_promotion_chart(promo_data: Dict) -> str:
    has_segments = 'segment_comparison' in promo_data and promo_data['segment_comparison']
    n_charts = 3 if has_segments else 2
    fig, axes = plt.subplots(1, n_charts, figsize=(5 * n_charts, 4))
    
    # Chart 1: Promo On vs Off
    ax1 = axes[0]
    labels = ['Promo OFF', 'Promo ON']
    means = [promo_data['promo_off']['mean'], promo_data['promo_on']['mean']]
    stds = [promo_data['promo_off']['std'], promo_data['promo_on']['std']]
    colors = ['#6b7280', '#10b981']
    
    bars = ax1.bar(labels, means, yerr=stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_ylabel('Avg Demand')
    ax1.set_title(f"Promotion Effect (+{promo_data['comparison']['lift']:.1f}%)", fontsize=11, fontweight='bold')
    
    # Add significance marker
    if promo_data['comparison']['significant']:
        ax1.annotate('*', xy=(0.5, max(means) * 1.1), fontsize=20, ha='center', color='#ef4444')
    
    # Chart 2: Effect size visualization
    ax2 = axes[1]
    metrics = ['Lift (%)', "Cohen's d"]
    values = [promo_data['comparison']['lift'], promo_data['comparison']['cohens_d']]
    colors = ['#3b82f6' if v > 0 else '#ef4444' for v in values]
    
    ax2.barh(metrics, values, color=colors, alpha=0.7, edgecolor='black')
    ax2.axvline(x=0, color='gray', linestyle='-', linewidth=1)
    ax2.set_xlabel('Value')
    ax2.set_title('Effect Metrics', fontsize=11, fontweight='bold')
    
    # Chart 3: Segment comparison (if available)
    if has_segments:
        ax3 = axes[2]
        segs = [s['segment'][:12] for s in promo_data['segment_comparison']]
        lifts = [s['lift'] for s in promo_data['segment_comparison']]
        colors = ['#10b981' if l > 0 else '#ef4444' for l in lifts]
        
        ax3.barh(segs, lifts, color=colors, alpha=0.7, edgecolor='black')
        ax3.axvline(x=0, color='gray', linestyle='-', linewidth=1)
        ax3.set_xlabel('Lift (%)')
        ax3.set_title('Promo Lift by Segment', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Step 3: Condition-Demand Relationship (관계성/원인)
# =============================================================================
def analyze_condition_relationship(df: pd.DataFrame, demand_col: str, condition_cols: List[str]) -> Dict:
    df_num = df[[demand_col] + condition_cols].apply(pd.to_numeric, errors='coerce').dropna()
    
    if len(df_num) < 10:
        return {'error': 'Insufficient data'}
    
    demand = df_num[demand_col]
    
    # Correlations
    correlations = []
    for col in condition_cols:
        corr, p_value = stats.pearsonr(df_num[col], demand)
        correlations.append({
            'condition': col,
            'correlation': _to_native(corr),
            'p_value': _to_native(p_value),
            'significant': bool(p_value < 0.05),
            'direction': 'positive' if corr > 0 else 'negative',
            'strength': 'strong' if abs(corr) > 0.5 else 'moderate' if abs(corr) > 0.3 else 'weak'
        })
    
    correlations = sorted(correlations, key=lambda x: abs(x['correlation']), reverse=True)
    
    # Multiple regression
    X = df_num[condition_cols]
    y = demand
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    r2 = model.score(X_scaled, y)
    
    # Standardized coefficients (beta)
    drivers = []
    total_abs = sum(abs(c) for c in model.coef_)
    for col, coef in zip(condition_cols, model.coef_):
        rel_imp = (abs(coef) / total_abs * 100) if total_abs > 0 else 0
        drivers.append({
            'condition': col,
            'beta': _to_native(coef),
            'relative_importance': _to_native(rel_imp),
            'direction': 'positive' if coef > 0 else 'negative'
        })
    
    drivers = sorted(drivers, key=lambda x: abs(x['beta']), reverse=True)
    
    return {
        'correlations': correlations,
        'top_correlate': correlations[0] if correlations else None,
        'drivers': drivers,
        'top_driver': drivers[0] if drivers else None,
        'r_squared': _to_native(r2),
        'model_quality': 'good' if r2 > 0.5 else 'moderate' if r2 > 0.3 else 'low',
        'n_observations': len(df_num),
        'n_significant': sum(1 for c in correlations if c['significant'])
    }


def create_relationship_chart(rel_data: Dict, df: pd.DataFrame, demand_col: str, condition_cols: List[str]) -> str:
    if rel_data.get('error'):
        return ""
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Chart 1: Correlation bars
    ax1 = axes[0]
    corrs = rel_data.get('correlations', [])
    if corrs:
        names = [c['condition'][:12] for c in corrs]
        vals = [c['correlation'] for c in corrs]
        colors = ['#10b981' if v > 0 else '#ef4444' for v in vals]
        
        ax1.barh(names, vals, color=colors, alpha=0.7, edgecolor='black')
        ax1.axvline(x=0, color='gray', linestyle='-', linewidth=1)
        ax1.set_xlabel('Correlation with Demand')
        ax1.set_title('Condition-Demand Correlation', fontsize=11, fontweight='bold')
    
    # Chart 2: Relative importance
    ax2 = axes[1]
    drivers = rel_data.get('drivers', [])
    if drivers:
        names = [d['condition'][:12] for d in drivers]
        imps = [d['relative_importance'] for d in drivers]
        colors = ['#10b981' if d['direction'] == 'positive' else '#ef4444' for d in drivers]
        
        ax2.barh(names, imps, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Relative Importance (%)')
        ax2.set_title(f"Demand Drivers (R²={rel_data['r_squared']:.3f})", fontsize=11, fontweight='bold')
    
    # Chart 3: Scatter of top driver
    ax3 = axes[2]
    top = rel_data.get('top_correlate')
    if top:
        df_num = df[[demand_col, top['condition']]].apply(pd.to_numeric, errors='coerce').dropna()
        ax3.scatter(df_num[top['condition']], df_num[demand_col], alpha=0.5, color='#3b82f6')
        
        # Regression line
        z = np.polyfit(df_num[top['condition']], df_num[demand_col], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df_num[top['condition']].min(), df_num[top['condition']].max(), 100)
        ax3.plot(x_line, p(x_line), color='#ef4444', linestyle='--', linewidth=2)
        
        ax3.set_xlabel(top['condition'])
        ax3.set_ylabel(demand_col)
        ax3.set_title(f"Top Driver (r={top['correlation']:.3f})", fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Step 4: Elasticity Analysis (심층 진단)
# =============================================================================
def analyze_elasticity(df: pd.DataFrame, demand_col: str, condition_cols: List[str]) -> Dict:
    df_num = df[[demand_col] + condition_cols].apply(pd.to_numeric, errors='coerce').dropna()
    
    if len(df_num) < 10:
        return {'error': 'Insufficient data'}
    
    elasticities = []
    
    for col in condition_cols:
        cond = df_num[col]
        demand = df_num[demand_col]
        
        # Point elasticity using regression
        # E = (dQ/dP) * (P/Q) ≈ β * (mean_P / mean_Q)
        slope, intercept, r_value, p_val, std_err = stats.linregress(cond, demand)
        
        mean_cond = cond.mean()
        mean_demand = demand.mean()
        
        if mean_demand > 0:
            elasticity = slope * (mean_cond / mean_demand)
        else:
            elasticity = 0
        
        # Arc elasticity between quartiles
        q1_cond = cond.quantile(0.25)
        q3_cond = cond.quantile(0.75)
        q1_demand = demand[cond <= q1_cond].mean()
        q3_demand = demand[cond >= q3_cond].mean()
        
        if (q1_cond + q3_cond) > 0 and (q1_demand + q3_demand) > 0:
            arc_elasticity = ((q3_demand - q1_demand) / ((q1_demand + q3_demand) / 2)) / \
                            ((q3_cond - q1_cond) / ((q1_cond + q3_cond) / 2))
        else:
            arc_elasticity = elasticity
        
        # Classification
        abs_e = abs(elasticity)
        if abs_e > 1:
            elasticity_type = 'elastic'
            interpretation = 'Demand is highly responsive'
        elif abs_e == 1:
            elasticity_type = 'unit_elastic'
            interpretation = 'Proportional response'
        else:
            elasticity_type = 'inelastic'
            interpretation = 'Demand is less responsive'
        
        elasticities.append({
            'condition': col,
            'elasticity': _to_native(elasticity),
            'arc_elasticity': _to_native(arc_elasticity),
            'elasticity_type': elasticity_type,
            'interpretation': interpretation,
            'slope': _to_native(slope),
            'r_squared': _to_native(r_value ** 2),
            'direction': 'positive' if elasticity > 0 else 'negative'
        })
    
    # Sort by absolute elasticity
    elasticities = sorted(elasticities, key=lambda x: abs(x['elasticity']), reverse=True)
    
    # Identify most elastic condition
    most_elastic = elasticities[0] if elasticities else None
    most_inelastic = min(elasticities, key=lambda x: abs(x['elasticity'])) if elasticities else None
    
    return {
        'elasticities': elasticities,
        'most_elastic': most_elastic,
        'most_inelastic': most_inelastic,
        'n_elastic': sum(1 for e in elasticities if e['elasticity_type'] == 'elastic'),
        'n_inelastic': sum(1 for e in elasticities if e['elasticity_type'] == 'inelastic'),
        'n_observations': len(df_num)
    }


def create_elasticity_chart(elast_data: Dict) -> str:
    if elast_data.get('error'):
        return ""
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    elasticities = elast_data.get('elasticities', [])
    
    # Chart 1: Elasticity values
    ax1 = axes[0]
    if elasticities:
        names = [e['condition'][:15] for e in elasticities]
        vals = [e['elasticity'] for e in elasticities]
        colors = ['#ef4444' if abs(v) > 1 else '#f59e0b' if abs(v) > 0.5 else '#10b981' for v in vals]
        
        ax1.barh(names, vals, color=colors, alpha=0.7, edgecolor='black')
        ax1.axvline(x=-1, color='gray', linestyle='--', alpha=0.5)
        ax1.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
        ax1.axvline(x=0, color='gray', linestyle='-', linewidth=1)
        ax1.set_xlabel('Elasticity')
        ax1.set_title('Demand Elasticity by Condition', fontsize=11, fontweight='bold')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#ef4444', alpha=0.7, label='Elastic (|E|>1)'),
            Patch(facecolor='#f59e0b', alpha=0.7, label='Moderate'),
            Patch(facecolor='#10b981', alpha=0.7, label='Inelastic (|E|<0.5)')
        ]
        ax1.legend(handles=legend_elements, loc='best', fontsize=8)
    
    # Chart 2: Elasticity type distribution
    ax2 = axes[1]
    types = ['Elastic', 'Inelastic']
    counts = [elast_data['n_elastic'], elast_data['n_inelastic']]
    colors = ['#ef4444', '#10b981']
    
    if sum(counts) > 0:
        ax2.pie(counts, labels=types, colors=colors, autopct='%1.0f%%', startangle=90)
        ax2.set_title('Elasticity Distribution', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Step 5: Performance Simulation (최적화/예측)
# =============================================================================
def simulate_performance(df: pd.DataFrame, demand_col: str, condition_cols: List[str],
                        elast_data: Dict) -> Dict:
    df_num = df[[demand_col] + condition_cols].apply(pd.to_numeric, errors='coerce').dropna()
    
    if len(df_num) < 10 or elast_data.get('error'):
        return {'error': 'Insufficient data or elasticity analysis failed'}
    
    current_demand = df_num[demand_col].mean()
    
    # Simulate for each condition
    simulations = []
    
    for e in elast_data.get('elasticities', []):
        col = e['condition']
        elasticity = e['elasticity']
        current_value = df_num[col].mean()
        
        # Simulate various changes
        changes = [-20, -10, -5, 5, 10, 20]
        scenarios = []
        
        for pct_change in changes:
            new_value = current_value * (1 + pct_change / 100)
            
            # Demand change = Elasticity × % change in condition
            demand_pct_change = elasticity * pct_change
            new_demand = current_demand * (1 + demand_pct_change / 100)
            
            scenarios.append({
                'condition_change_pct': pct_change,
                'new_condition_value': _to_native(new_value),
                'demand_change_pct': _to_native(demand_pct_change),
                'new_demand': _to_native(new_demand),
                'impact': 'high' if abs(demand_pct_change) > 10 else 'medium' if abs(demand_pct_change) > 5 else 'low'
            })
        
        # Find optimal change (maximize demand for positive elasticity, minimize for negative)
        if elasticity > 0:
            optimal = max(scenarios, key=lambda x: x['new_demand'])
        else:
            # For negative elasticity (like price), decreasing condition increases demand
            optimal = max(scenarios, key=lambda x: x['new_demand'])
        
        simulations.append({
            'condition': col,
            'elasticity': _to_native(elasticity),
            'elasticity_type': e['elasticity_type'],
            'current_value': _to_native(current_value),
            'scenarios': scenarios,
            'optimal_change': optimal
        })
    
    # Overall recommendations
    recommendations = []
    for sim in simulations:
        if sim['elasticity_type'] == 'elastic':
            if sim['elasticity'] < 0:
                recommendations.append({
                    'condition': sim['condition'],
                    'action': 'decrease',
                    'reason': f"Highly elastic (E={sim['elasticity']:.2f}). Small decrease yields large demand increase.",
                    'expected_impact': f"+{abs(sim['elasticity'] * 5):.1f}% demand for -5% {sim['condition']}"
                })
            else:
                recommendations.append({
                    'condition': sim['condition'],
                    'action': 'increase',
                    'reason': f"Highly elastic (E={sim['elasticity']:.2f}). Increase drives demand.",
                    'expected_impact': f"+{abs(sim['elasticity'] * 5):.1f}% demand for +5% {sim['condition']}"
                })
        else:
            recommendations.append({
                'condition': sim['condition'],
                'action': 'maintain',
                'reason': f"Inelastic (E={sim['elasticity']:.2f}). Changes have limited demand impact.",
                'expected_impact': 'Low sensitivity'
            })
    
    return {
        'current_demand': _to_native(current_demand),
        'simulations': simulations,
        'recommendations': recommendations,
        'n_conditions': len(simulations)
    }


def create_simulation_chart(sim_data: Dict) -> str:
    if sim_data.get('error'):
        return ""
    
    simulations = sim_data.get('simulations', [])
    n_sims = min(len(simulations), 3)
    
    if n_sims == 0:
        return ""
    
    fig, axes = plt.subplots(1, n_sims, figsize=(5 * n_sims, 4))
    if n_sims == 1:
        axes = [axes]
    
    for i, sim in enumerate(simulations[:n_sims]):
        ax = axes[i]
        scenarios = sim['scenarios']
        
        changes = [s['condition_change_pct'] for s in scenarios]
        demands = [s['demand_change_pct'] for s in scenarios]
        colors = ['#10b981' if d > 0 else '#ef4444' for d in demands]
        
        ax.bar([f"{c:+d}%" for c in changes], demands, color=colors, alpha=0.7, edgecolor='black')
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)
        ax.set_xlabel(f'{sim["condition"][:12]} Change')
        ax.set_ylabel('Demand Change (%)')
        ax.set_title(f'{sim["condition"][:12]} (E={sim["elasticity"]:.2f})', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Report & Insights
# =============================================================================
def generate_report(status: Dict, promo: Optional[Dict], rel: Optional[Dict],
                   elast: Dict, sim: Dict) -> Dict:
    report = {}
    
    report['step1_status'] = {
        'title': '1. Response Status',
        'question': 'What is the current demand response?',
        'finding': f"Avg demand: {status['demand']['mean']:.1f}, CV: {status['demand']['cv']:.1f}%",
        'detail': f"Analysis of {status['n_observations']} observations shows demand ranging from {status['demand']['min']:.1f} to {status['demand']['max']:.1f}. "
                 f"Most impactful condition: {status['most_impactful']['condition']} ({status['most_impactful']['demand_change_pct']:.1f}% change)."
    }
    
    if promo and not promo.get('error'):
        report['step2_promotion'] = {
            'title': '2. Promotion Comparison',
            'question': 'Does promotion affect demand?',
            'finding': f"Lift: {promo['comparison']['lift']:+.1f}%, Effect: {promo['comparison']['effect_size']}",
            'detail': f"Promotion {'significantly' if promo['comparison']['significant'] else 'does not significantly'} affects demand (p={promo['comparison']['p_value']:.4f}). "
                     f"Average demand: {promo['promo_off']['mean']:.1f} (off) vs {promo['promo_on']['mean']:.1f} (on)."
        }
    else:
        report['step2_promotion'] = {
            'title': '2. Promotion Comparison',
            'question': 'Does promotion affect demand?',
            'finding': 'Promotion analysis not performed',
            'detail': promo.get('error', 'No promotion column specified.')
        }
    
    if rel and not rel.get('error'):
        report['step3_relationship'] = {
            'title': '3. Condition-Demand Relationship',
            'question': 'What conditions drive demand?',
            'finding': f"Top driver: {rel['top_driver']['condition']} (β={rel['top_driver']['beta']:.3f}), R²={rel['r_squared']:.3f}",
            'detail': f"{rel['n_significant']} of {len(rel['correlations'])} conditions show significant correlation. "
                     f"Model explains {rel['r_squared']*100:.1f}% of demand variance."
        }
    else:
        report['step3_relationship'] = {
            'title': '3. Condition-Demand Relationship',
            'question': 'What conditions drive demand?',
            'finding': 'Relationship analysis failed',
            'detail': rel.get('error', 'Insufficient data.')
        }
    
    if elast and not elast.get('error'):
        report['step4_elasticity'] = {
            'title': '4. Elasticity Analysis',
            'question': 'How responsive is demand?',
            'finding': f"Most elastic: {elast['most_elastic']['condition']} (E={elast['most_elastic']['elasticity']:.2f})",
            'detail': f"{elast['n_elastic']} elastic and {elast['n_inelastic']} inelastic conditions identified. "
                     f"Most elastic: {elast['most_elastic']['interpretation']}."
        }
    else:
        report['step4_elasticity'] = {
            'title': '4. Elasticity Analysis',
            'question': 'How responsive is demand?',
            'finding': 'Elasticity analysis failed',
            'detail': elast.get('error', 'Insufficient data.')
        }
    
    if sim and not sim.get('error'):
        top_rec = sim['recommendations'][0] if sim['recommendations'] else None
        report['step5_simulation'] = {
            'title': '5. Performance Simulation',
            'question': 'What changes optimize demand?',
            'finding': f"Recommendation: {top_rec['action']} {top_rec['condition']}" if top_rec else "No clear recommendation",
            'detail': f"Current demand: {sim['current_demand']:.1f}. " + (top_rec['reason'] if top_rec else "")
        }
    else:
        report['step5_simulation'] = {
            'title': '5. Performance Simulation',
            'question': 'What changes optimize demand?',
            'finding': 'Simulation failed',
            'detail': sim.get('error', 'Insufficient data.')
        }
    
    return report


def generate_insights(status: Dict, promo: Optional[Dict], rel: Optional[Dict],
                     elast: Dict, sim: Dict) -> List[Dict]:
    insights = []
    
    # Promotion insight
    if promo and not promo.get('error'):
        if promo['comparison']['significant'] and promo['comparison']['lift'] > 10:
            insights.append({
                'title': 'Strong Promotion Effect',
                'description': f"Promotions drive {promo['comparison']['lift']:.1f}% demand lift. Highly effective.",
                'status': 'positive'
            })
        elif promo['comparison']['lift'] < 5:
            insights.append({
                'title': 'Weak Promotion Effect',
                'description': f"Only {promo['comparison']['lift']:.1f}% lift from promotions. Consider optimization.",
                'status': 'warning'
            })
    
    # Elasticity insight
    if elast and not elast.get('error'):
        if elast['n_elastic'] > 0:
            insights.append({
                'title': 'High Elasticity Opportunity',
                'description': f"{elast['most_elastic']['condition']} shows E={elast['most_elastic']['elasticity']:.2f}. High leverage point.",
                'status': 'positive'
            })
    
    # Simulation insight
    if sim and not sim.get('error') and sim.get('recommendations'):
        top_rec = sim['recommendations'][0]
        if top_rec['action'] != 'maintain':
            insights.append({
                'title': 'Optimization Opportunity',
                'description': f"{top_rec['action'].title()} {top_rec['condition']}: {top_rec['expected_impact']}",
                'status': 'positive'
            })
    
    # Variability insight
    if status['demand']['cv'] and status['demand']['cv'] > 30:
        insights.append({
            'title': 'High Demand Variability',
            'description': f"Demand CV of {status['demand']['cv']:.1f}% indicates volatility. Stabilization needed.",
            'status': 'warning'
        })
    
    return insights


# =============================================================================
# Main API Endpoint
# =============================================================================
@router.post("/elasticity-analysis")
async def analyze_elasticity_endpoint(request: ElasticityRequest):
    try:
        df = pd.DataFrame(request.data)
        demand_col = request.demand_col
        condition_cols = request.condition_cols
        promotion_col = request.promotion_col
        segment_col = request.segment_col
        
        if len(df) < 10:
            raise HTTPException(status_code=400, detail="Need at least 10 observations")
        
        # Convert columns to numeric
        df[demand_col] = pd.to_numeric(df[demand_col], errors='coerce')
        for col in condition_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        results = {}
        visualizations = {}
        
        # Step 1: Response Status
        status = analyze_response_status(df, demand_col, condition_cols)
        results['status'] = status
        visualizations['status_chart'] = create_status_chart(status, df, demand_col, condition_cols)
        
        # Step 2: Promotion Comparison
        promo = None
        if promotion_col and promotion_col in df.columns:
            promo = analyze_promotion_comparison(df, demand_col, promotion_col, segment_col)
            results['promotion'] = promo
            visualizations['promotion_chart'] = create_promotion_chart(promo)
        
        # Step 3: Condition-Demand Relationship
        rel = analyze_condition_relationship(df, demand_col, condition_cols)
        results['relationship'] = rel
        if not rel.get('error'):
            visualizations['relationship_chart'] = create_relationship_chart(rel, df, demand_col, condition_cols)
        
        # Step 4: Elasticity Analysis
        elast = analyze_elasticity(df, demand_col, condition_cols)
        results['elasticity'] = elast
        if not elast.get('error'):
            visualizations['elasticity_chart'] = create_elasticity_chart(elast)
        
        # Step 5: Performance Simulation
        sim = simulate_performance(df, demand_col, condition_cols, elast)
        results['simulation'] = sim
        if not sim.get('error'):
            visualizations['simulation_chart'] = create_simulation_chart(sim)
        
        report = generate_report(status, promo, rel, elast, sim)
        insights = generate_insights(status, promo, rel, elast, sim)
        
        summary = {
            'n_observations': status['n_observations'],
            'avg_demand': status['demand']['mean'],
            'most_elastic': elast['most_elastic']['condition'] if elast.get('most_elastic') else None,
            'top_elasticity': elast['most_elastic']['elasticity'] if elast.get('most_elastic') else None,
            'promo_lift': promo['comparison']['lift'] if promo else None
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
