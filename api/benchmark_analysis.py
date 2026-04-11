"""
Benchmark & Competitive Analysis API
5-step framework for comprehensive competitive positioning analysis
1. Market Position (vs Market Average)
2. Strength vs Weakness Indicators
3. Competitive Advantage Factors
4. Relative Competitiveness Diagnosis
5. Improvement Simulation (Rank Change)
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


class BenchmarkRequest(BaseModel):
    data: List[Dict[str, Any]]
    entity_col: str  # Company/Product name column
    metric_cols: List[str]  # Performance metrics
    target_entity: Optional[str] = None  # Focus entity (default: first)
    weight_cols: Optional[Dict[str, float]] = None  # Metric weights for scoring
    higher_better: Optional[Dict[str, bool]] = None  # Direction of metrics


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
# Step 1: Market Position
# =============================================================================
def analyze_market_position(df: pd.DataFrame, entity_col: str, metric_cols: List[str], 
                           target_entity: str, higher_better: Dict[str, bool]) -> Dict:
    target_data = df[df[entity_col] == target_entity].iloc[0]
    
    metrics = []
    for col in metric_cols:
        values = pd.to_numeric(df[col], errors='coerce').dropna()
        target_val = pd.to_numeric(target_data[col], errors='coerce')
        
        market_avg = values.mean()
        market_std = values.std()
        market_min = values.min()
        market_max = values.max()
        
        # Gap from average
        gap = target_val - market_avg
        gap_pct = (gap / market_avg * 100) if market_avg != 0 else 0
        
        # Z-score
        z_score = (target_val - market_avg) / market_std if market_std > 0 else 0
        
        # Percentile rank
        percentile = stats.percentileofscore(values, target_val)
        
        # Rank
        is_higher_better = higher_better.get(col, True)
        if is_higher_better:
            rank = (values > target_val).sum() + 1
        else:
            rank = (values < target_val).sum() + 1
        
        # Position assessment
        if is_higher_better:
            position = 'above' if target_val > market_avg else 'below' if target_val < market_avg else 'average'
        else:
            position = 'above' if target_val < market_avg else 'below' if target_val > market_avg else 'average'
        
        metrics.append({
            'metric': col,
            'target_value': _to_native(target_val),
            'market_avg': _to_native(market_avg),
            'market_std': _to_native(market_std),
            'market_min': _to_native(market_min),
            'market_max': _to_native(market_max),
            'gap': _to_native(gap),
            'gap_pct': _to_native(gap_pct),
            'z_score': _to_native(z_score),
            'percentile': _to_native(percentile),
            'rank': _to_native(rank),
            'total_entities': len(values),
            'position': position,
            'higher_better': is_higher_better
        })
    
    # Overall position
    above_count = sum(1 for m in metrics if m['position'] == 'above')
    below_count = sum(1 for m in metrics if m['position'] == 'below')
    avg_percentile = np.mean([m['percentile'] for m in metrics])
    
    return {
        'target_entity': target_entity,
        'metrics': metrics,
        'n_metrics': len(metrics),
        'n_entities': len(df),
        'above_average_count': above_count,
        'below_average_count': below_count,
        'avg_percentile': _to_native(avg_percentile),
        'overall_position': 'leader' if avg_percentile > 75 else 'above average' if avg_percentile > 50 else 'below average' if avg_percentile > 25 else 'laggard'
    }


def create_position_chart(pos_data: Dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    metrics = pos_data.get('metrics', [])
    
    # Chart 1: Gap from market average
    ax1 = axes[0]
    if metrics:
        names = [m['metric'][:15] for m in metrics]
        gaps = [m['gap_pct'] for m in metrics]
        colors = ['#10b981' if g > 0 else '#ef4444' for g in gaps]
        ax1.barh(names, gaps, color=colors, alpha=0.7, edgecolor='black')
        ax1.axvline(x=0, color='gray', linestyle='-', linewidth=2)
        ax1.set_xlabel('Gap from Market Average (%)')
        ax1.set_title(f"{pos_data['target_entity']} vs Market", fontsize=11, fontweight='bold')
    
    # Chart 2: Percentile ranks
    ax2 = axes[1]
    if metrics:
        percentiles = [m['percentile'] for m in metrics]
        colors = ['#10b981' if p > 50 else '#ef4444' for p in percentiles]
        ax2.barh(names, percentiles, color=colors, alpha=0.7, edgecolor='black')
        ax2.axvline(x=50, color='gray', linestyle='--', linewidth=2, label='Median')
        ax2.set_xlabel('Percentile Rank')
        ax2.set_xlim(0, 100)
        ax2.set_title('Market Percentile Position', fontsize=11, fontweight='bold')
        ax2.legend()
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Step 2: Strength vs Weakness
# =============================================================================
def analyze_strength_weakness(df: pd.DataFrame, entity_col: str, metric_cols: List[str],
                             target_entity: str, higher_better: Dict[str, bool]) -> Dict:
    target_data = df[df[entity_col] == target_entity].iloc[0]
    
    analysis = []
    for col in metric_cols:
        values = pd.to_numeric(df[col], errors='coerce').dropna()
        target_val = pd.to_numeric(target_data[col], errors='coerce')
        
        market_avg = values.mean()
        market_std = values.std()
        
        z_score = (target_val - market_avg) / market_std if market_std > 0 else 0
        is_higher_better = higher_better.get(col, True)
        
        # Adjust z-score for direction
        effective_z = z_score if is_higher_better else -z_score
        
        # Classification
        if effective_z > 1.5:
            classification = 'major_strength'
        elif effective_z > 0.5:
            classification = 'strength'
        elif effective_z > -0.5:
            classification = 'neutral'
        elif effective_z > -1.5:
            classification = 'weakness'
        else:
            classification = 'major_weakness'
        
        analysis.append({
            'metric': col,
            'value': _to_native(target_val),
            'market_avg': _to_native(market_avg),
            'z_score': _to_native(z_score),
            'effective_z': _to_native(effective_z),
            'classification': classification,
            'higher_better': is_higher_better
        })
    
    # Separate strengths and weaknesses
    strengths = [a for a in analysis if a['classification'] in ['strength', 'major_strength']]
    weaknesses = [a for a in analysis if a['classification'] in ['weakness', 'major_weakness']]
    neutrals = [a for a in analysis if a['classification'] == 'neutral']
    
    # Sort by effective z-score
    strengths = sorted(strengths, key=lambda x: x['effective_z'], reverse=True)
    weaknesses = sorted(weaknesses, key=lambda x: x['effective_z'])
    
    return {
        'analysis': analysis,
        'strengths': strengths,
        'weaknesses': weaknesses,
        'neutrals': neutrals,
        'n_strengths': len(strengths),
        'n_weaknesses': len(weaknesses),
        'n_neutral': len(neutrals),
        'top_strength': strengths[0] if strengths else None,
        'top_weakness': weaknesses[0] if weaknesses else None,
        'balance': 'strength-heavy' if len(strengths) > len(weaknesses) else 'weakness-heavy' if len(weaknesses) > len(strengths) else 'balanced'
    }


def create_sw_chart(sw_data: Dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    analysis = sw_data.get('analysis', [])
    
    # Chart 1: Z-score spectrum
    ax1 = axes[0]
    if analysis:
        sorted_analysis = sorted(analysis, key=lambda x: x['effective_z'], reverse=True)
        names = [a['metric'][:15] for a in sorted_analysis]
        z_scores = [a['effective_z'] for a in sorted_analysis]
        
        colors = []
        for z in z_scores:
            if z > 1.5:
                colors.append('#059669')  # major strength
            elif z > 0.5:
                colors.append('#10b981')  # strength
            elif z > -0.5:
                colors.append('#9ca3af')  # neutral
            elif z > -1.5:
                colors.append('#ef4444')  # weakness
            else:
                colors.append('#b91c1c')  # major weakness
        
        ax1.barh(names, z_scores, color=colors, alpha=0.7, edgecolor='black')
        ax1.axvline(x=0, color='gray', linestyle='-', linewidth=2)
        ax1.axvline(x=0.5, color='green', linestyle='--', alpha=0.5)
        ax1.axvline(x=-0.5, color='red', linestyle='--', alpha=0.5)
        ax1.set_xlabel('Effective Z-Score (Strength ↔ Weakness)')
        ax1.set_title('Strength-Weakness Spectrum', fontsize=11, fontweight='bold')
    
    # Chart 2: Classification pie
    ax2 = axes[1]
    labels = ['Major Strength', 'Strength', 'Neutral', 'Weakness', 'Major Weakness']
    counts = [
        sum(1 for a in analysis if a['classification'] == 'major_strength'),
        sum(1 for a in analysis if a['classification'] == 'strength'),
        sum(1 for a in analysis if a['classification'] == 'neutral'),
        sum(1 for a in analysis if a['classification'] == 'weakness'),
        sum(1 for a in analysis if a['classification'] == 'major_weakness')
    ]
    colors = ['#059669', '#10b981', '#9ca3af', '#ef4444', '#b91c1c']
    
    # Only include non-zero segments
    filtered = [(l, c, col) for l, c, col in zip(labels, counts, colors) if c > 0]
    if filtered:
        labels_f, counts_f, colors_f = zip(*filtered)
        ax2.pie(counts_f, labels=labels_f, colors=colors_f, autopct='%1.0f%%', startangle=90)
        ax2.set_title('Metric Classification', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Step 3: Competitive Advantage
# =============================================================================
def analyze_competitive_advantage(df: pd.DataFrame, entity_col: str, metric_cols: List[str],
                                  target_entity: str, higher_better: Dict[str, bool],
                                  weights: Dict[str, float]) -> Dict:
    target_data = df[df[entity_col] == target_entity].iloc[0]
    
    advantages = []
    for col in metric_cols:
        values = pd.to_numeric(df[col], errors='coerce').dropna()
        target_val = pd.to_numeric(target_data[col], errors='coerce')
        
        is_higher_better = higher_better.get(col, True)
        weight = weights.get(col, 1.0)
        
        # Normalize to 0-100 scale
        min_val, max_val = values.min(), values.max()
        if max_val > min_val:
            if is_higher_better:
                normalized = (target_val - min_val) / (max_val - min_val) * 100
            else:
                normalized = (max_val - target_val) / (max_val - min_val) * 100
        else:
            normalized = 50
        
        # Weighted score contribution
        weighted_contribution = normalized * weight
        
        # Competitive position
        if is_higher_better:
            rank = (values > target_val).sum() + 1
            is_leader = target_val == values.max()
        else:
            rank = (values < target_val).sum() + 1
            is_leader = target_val == values.min()
        
        advantages.append({
            'metric': col,
            'value': _to_native(target_val),
            'normalized_score': _to_native(normalized),
            'weight': weight,
            'weighted_contribution': _to_native(weighted_contribution),
            'rank': _to_native(rank),
            'is_leader': is_leader,
            'is_top3': rank <= 3,
            'higher_better': is_higher_better
        })
    
    # Overall competitive score
    total_weight = sum(weights.get(col, 1.0) for col in metric_cols)
    overall_score = sum(a['weighted_contribution'] for a in advantages) / total_weight if total_weight > 0 else 0
    
    # Sort by contribution
    advantages = sorted(advantages, key=lambda x: x['weighted_contribution'], reverse=True)
    
    # Key advantages (top 3 where we lead)
    key_advantages = [a for a in advantages if a['is_top3']][:3]
    
    return {
        'advantages': advantages,
        'overall_score': _to_native(overall_score),
        'n_leader_positions': sum(1 for a in advantages if a['is_leader']),
        'n_top3_positions': sum(1 for a in advantages if a['is_top3']),
        'key_advantages': key_advantages,
        'competitive_level': 'dominant' if overall_score > 80 else 'strong' if overall_score > 60 else 'moderate' if overall_score > 40 else 'weak'
    }


def create_advantage_chart(adv_data: Dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    advantages = adv_data.get('advantages', [])
    
    # Chart 1: Weighted contribution
    ax1 = axes[0]
    if advantages:
        names = [a['metric'][:15] for a in advantages]
        contributions = [a['weighted_contribution'] for a in advantages]
        colors = ['#10b981' if a['is_top3'] else '#3b82f6' for a in advantages]
        ax1.barh(names, contributions, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Weighted Score Contribution')
        ax1.set_title('Competitive Advantage Factors', fontsize=11, fontweight='bold')
    
    # Chart 2: Radar-like comparison (normalized scores)
    ax2 = axes[1]
    if advantages:
        names = [a['metric'][:12] for a in advantages[:8]]  # Limit to 8
        scores = [a['normalized_score'] for a in advantages[:8]]
        
        x = np.arange(len(names))
        ax2.bar(x, scores, color='#3b82f6', alpha=0.7, edgecolor='black')
        ax2.axhline(y=50, color='gray', linestyle='--', label='Market Median')
        ax2.set_xticks(x)
        ax2.set_xticklabels(names, rotation=45, ha='right')
        ax2.set_ylabel('Normalized Score (0-100)')
        ax2.set_ylim(0, 100)
        ax2.set_title(f"Competitive Score: {adv_data['overall_score']:.1f}/100", fontsize=11, fontweight='bold')
        ax2.legend()
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Step 4: Relative Competitiveness
# =============================================================================
def analyze_relative_competitiveness(df: pd.DataFrame, entity_col: str, metric_cols: List[str],
                                     target_entity: str, higher_better: Dict[str, bool],
                                     weights: Dict[str, float]) -> Dict:
    # Calculate composite score for all entities
    entity_scores = []
    
    for _, row in df.iterrows():
        entity = row[entity_col]
        scores = []
        
        for col in metric_cols:
            values = pd.to_numeric(df[col], errors='coerce').dropna()
            val = pd.to_numeric(row[col], errors='coerce')
            
            is_higher_better = higher_better.get(col, True)
            weight = weights.get(col, 1.0)
            
            min_val, max_val = values.min(), values.max()
            if max_val > min_val:
                if is_higher_better:
                    normalized = (val - min_val) / (max_val - min_val) * 100
                else:
                    normalized = (max_val - val) / (max_val - min_val) * 100
            else:
                normalized = 50
            
            scores.append(normalized * weight)
        
        total_weight = sum(weights.get(col, 1.0) for col in metric_cols)
        composite_score = sum(scores) / total_weight if total_weight > 0 else 0
        
        entity_scores.append({
            'entity': _to_native(entity),
            'composite_score': _to_native(composite_score),
            'is_target': entity == target_entity
        })
    
    # Rank entities
    entity_scores = sorted(entity_scores, key=lambda x: x['composite_score'], reverse=True)
    for i, es in enumerate(entity_scores):
        es['rank'] = i + 1
    
    # Find target position
    target_score = next((es for es in entity_scores if es['is_target']), None)
    target_rank = target_score['rank'] if target_score else None
    
    # Gap to leader
    leader = entity_scores[0]
    gap_to_leader = leader['composite_score'] - target_score['composite_score'] if target_score else 0
    
    # Nearest competitors
    target_idx = next((i for i, es in enumerate(entity_scores) if es['is_target']), 0)
    competitors_above = entity_scores[max(0, target_idx-3):target_idx]
    competitors_below = entity_scores[target_idx+1:target_idx+4]
    
    return {
        'rankings': entity_scores,
        'target_rank': target_rank,
        'target_score': target_score['composite_score'] if target_score else None,
        'total_entities': len(entity_scores),
        'leader': leader,
        'gap_to_leader': _to_native(gap_to_leader),
        'competitors_above': competitors_above,
        'competitors_below': competitors_below,
        'percentile_rank': _to_native((1 - (target_rank - 1) / len(entity_scores)) * 100) if target_rank else None
    }


def create_competitiveness_chart(comp_data: Dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    rankings = comp_data.get('rankings', [])
    
    # Chart 1: Full ranking
    ax1 = axes[0]
    if rankings:
        # Show top 10 and target
        top_rankings = rankings[:10]
        target_in_top = any(r['is_target'] for r in top_rankings)
        
        if not target_in_top:
            target_entry = next((r for r in rankings if r['is_target']), None)
            if target_entry:
                top_rankings = rankings[:9] + [target_entry]
        
        names = [f"#{r['rank']} {str(r['entity'])[:12]}" for r in top_rankings]
        scores = [r['composite_score'] for r in top_rankings]
        colors = ['#f59e0b' if r['is_target'] else '#3b82f6' for r in top_rankings]
        
        ax1.barh(names, scores, color=colors, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Composite Score')
        ax1.set_title('Competitive Rankings', fontsize=11, fontweight='bold')
        ax1.invert_yaxis()
    
    # Chart 2: Position in market
    ax2 = axes[1]
    target_rank = comp_data.get('target_rank', 1)
    total = comp_data.get('total_entities', 1)
    
    # Create position indicator
    percentile = comp_data.get('percentile_rank', 50)
    ax2.barh(['Position'], [100], color='#e5e7eb', alpha=0.5)
    ax2.barh(['Position'], [percentile], color='#3b82f6', alpha=0.7)
    ax2.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlim(0, 100)
    ax2.set_xlabel('Percentile Position')
    ax2.set_title(f"Rank: #{target_rank} of {total} ({percentile:.0f}th percentile)", fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Step 5: Improvement Simulation
# =============================================================================
def simulate_improvement(df: pd.DataFrame, entity_col: str, metric_cols: List[str],
                        target_entity: str, higher_better: Dict[str, bool],
                        weights: Dict[str, float], sw_data: Dict) -> Dict:
    # Current ranking
    current_scores = []
    for _, row in df.iterrows():
        entity = row[entity_col]
        score = 0
        total_weight = 0
        
        for col in metric_cols:
            values = pd.to_numeric(df[col], errors='coerce').dropna()
            val = pd.to_numeric(row[col], errors='coerce')
            is_higher_better = higher_better.get(col, True)
            weight = weights.get(col, 1.0)
            
            min_val, max_val = values.min(), values.max()
            if max_val > min_val:
                if is_higher_better:
                    normalized = (val - min_val) / (max_val - min_val) * 100
                else:
                    normalized = (max_val - val) / (max_val - min_val) * 100
            else:
                normalized = 50
            
            score += normalized * weight
            total_weight += weight
        
        current_scores.append({
            'entity': entity,
            'score': score / total_weight if total_weight > 0 else 0
        })
    
    current_scores = sorted(current_scores, key=lambda x: x['score'], reverse=True)
    current_rank = next((i+1 for i, s in enumerate(current_scores) if s['entity'] == target_entity), None)
    current_score = next((s['score'] for s in current_scores if s['entity'] == target_entity), 0)
    
    # Simulation scenarios
    scenarios = []
    weaknesses = sw_data.get('weaknesses', [])
    
    # Scenario 1: Fix top weakness to market average
    if weaknesses:
        top_weakness = weaknesses[0]
        scenario1_scores = current_scores.copy()
        
        # Recalculate target score with improved weakness
        target_idx = next((i for i, s in enumerate(scenario1_scores) if s['entity'] == target_entity), None)
        if target_idx is not None:
            improvement = abs(top_weakness['effective_z']) * 10  # Approximate improvement
            new_score = current_score + improvement * weights.get(top_weakness['metric'], 1.0) / sum(weights.values())
            scenario1_scores[target_idx] = {'entity': target_entity, 'score': new_score}
            scenario1_scores = sorted(scenario1_scores, key=lambda x: x['score'], reverse=True)
            new_rank = next((i+1 for i, s in enumerate(scenario1_scores) if s['entity'] == target_entity), current_rank)
            
            scenarios.append({
                'name': f"Fix {top_weakness['metric'][:20]}",
                'description': f"Improve {top_weakness['metric']} to market average",
                'metric': top_weakness['metric'],
                'current_rank': current_rank,
                'new_rank': new_rank,
                'rank_change': current_rank - new_rank,
                'current_score': _to_native(current_score),
                'new_score': _to_native(new_score),
                'score_change': _to_native(new_score - current_score)
            })
    
    # Scenario 2: Fix all weaknesses to neutral
    if len(weaknesses) > 1:
        total_improvement = sum(abs(w['effective_z']) * 10 * weights.get(w['metric'], 1.0) for w in weaknesses)
        new_score = current_score + total_improvement / sum(weights.values()) / len(weaknesses)
        
        scenario2_scores = [{'entity': s['entity'], 'score': s['score']} for s in current_scores]
        target_idx = next((i for i, s in enumerate(scenario2_scores) if s['entity'] == target_entity), None)
        if target_idx is not None:
            scenario2_scores[target_idx] = {'entity': target_entity, 'score': new_score}
            scenario2_scores = sorted(scenario2_scores, key=lambda x: x['score'], reverse=True)
            new_rank = next((i+1 for i, s in enumerate(scenario2_scores) if s['entity'] == target_entity), current_rank)
            
            scenarios.append({
                'name': 'Fix All Weaknesses',
                'description': f"Improve all {len(weaknesses)} weakness metrics to neutral",
                'metric': 'Multiple',
                'current_rank': current_rank,
                'new_rank': new_rank,
                'rank_change': current_rank - new_rank,
                'current_score': _to_native(current_score),
                'new_score': _to_native(new_score),
                'score_change': _to_native(new_score - current_score)
            })
    
    # Scenario 3: Become leader in top weighted metric
    top_weighted_metric = max(metric_cols, key=lambda x: weights.get(x, 1.0))
    values = pd.to_numeric(df[top_weighted_metric], errors='coerce').dropna()
    max_improvement = (values.max() - df[df[entity_col] == target_entity][top_weighted_metric].values[0])
    
    if max_improvement > 0:
        improvement = 20 * weights.get(top_weighted_metric, 1.0) / sum(weights.values())
        new_score = current_score + improvement
        
        scenario3_scores = [{'entity': s['entity'], 'score': s['score']} for s in current_scores]
        target_idx = next((i for i, s in enumerate(scenario3_scores) if s['entity'] == target_entity), None)
        if target_idx is not None:
            scenario3_scores[target_idx] = {'entity': target_entity, 'score': new_score}
            scenario3_scores = sorted(scenario3_scores, key=lambda x: x['score'], reverse=True)
            new_rank = next((i+1 for i, s in enumerate(scenario3_scores) if s['entity'] == target_entity), current_rank)
            
            scenarios.append({
                'name': f"Lead in {top_weighted_metric[:15]}",
                'description': f"Become market leader in {top_weighted_metric}",
                'metric': top_weighted_metric,
                'current_rank': current_rank,
                'new_rank': new_rank,
                'rank_change': current_rank - new_rank,
                'current_score': _to_native(current_score),
                'new_score': _to_native(new_score),
                'score_change': _to_native(new_score - current_score)
            })
    
    # Best scenario
    best_scenario = max(scenarios, key=lambda x: x['rank_change']) if scenarios else None
    
    return {
        'current_rank': current_rank,
        'current_score': _to_native(current_score),
        'total_entities': len(df),
        'scenarios': scenarios,
        'best_scenario': best_scenario,
        'max_possible_rank': 1,
        'improvement_potential': current_rank - 1
    }


def create_simulation_chart(sim_data: Dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    scenarios = sim_data.get('scenarios', [])
    
    # Chart 1: Rank change by scenario
    ax1 = axes[0]
    if scenarios:
        names = [s['name'][:20] for s in scenarios]
        rank_changes = [s['rank_change'] for s in scenarios]
        colors = ['#10b981' if r > 0 else '#ef4444' if r < 0 else '#9ca3af' for r in rank_changes]
        
        ax1.barh(names, rank_changes, color=colors, alpha=0.7, edgecolor='black')
        ax1.axvline(x=0, color='gray', linestyle='-', linewidth=2)
        ax1.set_xlabel('Rank Improvement (positions)')
        ax1.set_title('Improvement Scenarios', fontsize=11, fontweight='bold')
    
    # Chart 2: Score improvement
    ax2 = axes[1]
    if scenarios:
        current = sim_data.get('current_score', 0)
        x = np.arange(len(scenarios) + 1)
        scores = [current] + [s['new_score'] for s in scenarios]
        labels = ['Current'] + [s['name'][:15] for s in scenarios]
        colors = ['#6b7280'] + ['#10b981' if s['score_change'] > 0 else '#ef4444' for s in scenarios]
        
        ax2.bar(x, scores, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_xticks(x)
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.set_ylabel('Composite Score')
        ax2.set_title('Score Impact', fontsize=11, fontweight='bold')
        ax2.axhline(y=current, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Report & Insights
# =============================================================================
def generate_report(pos: Dict, sw: Dict, adv: Dict, comp: Dict, sim: Dict, target: str) -> Dict:
    report = {}
    
    report['step1_position'] = {
        'title': '1. Market Position',
        'question': 'Where do we stand vs market average?',
        'finding': f"{target} is {pos['overall_position']} (avg percentile: {pos['avg_percentile']:.0f}%)",
        'detail': f"Analysis of {pos['n_metrics']} metrics shows {target} is above average in {pos['above_average_count']} and below in {pos['below_average_count']}. "
                 f"Overall market position: {pos['overall_position']}."
    }
    
    report['step2_sw'] = {
        'title': '2. Strengths vs Weaknesses',
        'question': 'What are our key strengths and weaknesses?',
        'finding': f"{sw['n_strengths']} strengths, {sw['n_weaknesses']} weaknesses ({sw['balance']})",
        'detail': f"Top strength: {sw['top_strength']['metric'] if sw['top_strength'] else 'None'} (z={sw['top_strength']['effective_z']:.2f if sw['top_strength'] else 0}). "
                 f"Top weakness: {sw['top_weakness']['metric'] if sw['top_weakness'] else 'None'} (z={sw['top_weakness']['effective_z']:.2f if sw['top_weakness'] else 0})."
    }
    
    report['step3_advantage'] = {
        'title': '3. Competitive Advantage',
        'question': 'What drives our competitive position?',
        'finding': f"Overall score: {adv['overall_score']:.1f}/100 ({adv['competitive_level']})",
        'detail': f"Leading in {adv['n_leader_positions']} metrics, top 3 in {adv['n_top3_positions']}. "
                 f"Key advantages: {', '.join([a['metric'][:15] for a in adv['key_advantages'][:3]])}."
    }
    
    report['step4_competitiveness'] = {
        'title': '4. Relative Competitiveness',
        'question': 'How do we rank against competitors?',
        'finding': f"Rank #{comp['target_rank']} of {comp['total_entities']} ({comp['percentile_rank']:.0f}th percentile)",
        'detail': f"Gap to leader ({comp['leader']['entity']}): {comp['gap_to_leader']:.1f} points. "
                 f"Composite score: {comp['target_score']:.1f}/100."
    }
    
    if sim['best_scenario']:
        report['step5_simulation'] = {
            'title': '5. Improvement Simulation',
            'question': 'What if we fix our weaknesses?',
            'finding': f"Best scenario: {sim['best_scenario']['name']} (+{sim['best_scenario']['rank_change']} ranks)",
            'detail': f"Improving {sim['best_scenario']['metric']} could move from #{sim['current_rank']} to #{sim['best_scenario']['new_rank']}. "
                     f"Score would increase by {sim['best_scenario']['score_change']:.1f} points."
        }
    else:
        report['step5_simulation'] = {
            'title': '5. Improvement Simulation',
            'question': 'What if we fix our weaknesses?',
            'finding': 'No improvement scenarios available',
            'detail': 'Current position may already be optimal or no clear weaknesses identified.'
        }
    
    return report


def generate_insights(pos: Dict, sw: Dict, adv: Dict, comp: Dict, sim: Dict) -> List[Dict]:
    insights = []
    
    # Position insight
    if pos['overall_position'] in ['leader', 'above average']:
        insights.append({
            'title': 'Strong Market Position',
            'description': f"Above average in {pos['above_average_count']} of {pos['n_metrics']} metrics.",
            'status': 'positive'
        })
    elif pos['overall_position'] == 'laggard':
        insights.append({
            'title': 'Market Position Concern',
            'description': f"Below average in {pos['below_average_count']} metrics. Improvement needed.",
            'status': 'warning'
        })
    
    # Weakness insight
    if sw['n_weaknesses'] > sw['n_strengths']:
        insights.append({
            'title': 'More Weaknesses Than Strengths',
            'description': f"{sw['n_weaknesses']} weaknesses vs {sw['n_strengths']} strengths. Focus on improvement.",
            'status': 'warning'
        })
    
    # Improvement potential
    if sim.get('best_scenario') and sim['best_scenario']['rank_change'] >= 3:
        insights.append({
            'title': 'High Improvement Potential',
            'description': f"Fixing {sim['best_scenario']['metric']} could improve rank by {sim['best_scenario']['rank_change']} positions.",
            'status': 'positive'
        })
    
    # Competitive advantage
    if adv['n_leader_positions'] > 0:
        insights.append({
            'title': 'Market Leadership',
            'description': f"Leading the market in {adv['n_leader_positions']} metrics.",
            'status': 'positive'
        })
    
    return insights


# =============================================================================
# Main API Endpoint
# =============================================================================
@router.post("/benchmark-analysis")
async def analyze_benchmark(request: BenchmarkRequest):
    try:
        df = pd.DataFrame(request.data)
        entity_col = request.entity_col
        metric_cols = request.metric_cols
        target_entity = request.target_entity or df[entity_col].iloc[0]
        weights = request.weight_cols or {col: 1.0 for col in metric_cols}
        higher_better = request.higher_better or {col: True for col in metric_cols}
        
        if len(df) < 2:
            raise HTTPException(status_code=400, detail="Need at least 2 entities to compare")
        
        if target_entity not in df[entity_col].values:
            raise HTTPException(status_code=400, detail=f"Target entity '{target_entity}' not found")
        
        # Convert metric columns to numeric
        for col in metric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        results = {}
        visualizations = {}
        
        # Step 1: Market Position
        pos = analyze_market_position(df, entity_col, metric_cols, target_entity, higher_better)
        results['position'] = pos
        visualizations['position_chart'] = create_position_chart(pos)
        
        # Step 2: Strength vs Weakness
        sw = analyze_strength_weakness(df, entity_col, metric_cols, target_entity, higher_better)
        results['strength_weakness'] = sw
        visualizations['sw_chart'] = create_sw_chart(sw)
        
        # Step 3: Competitive Advantage
        adv = analyze_competitive_advantage(df, entity_col, metric_cols, target_entity, higher_better, weights)
        results['advantage'] = adv
        visualizations['advantage_chart'] = create_advantage_chart(adv)
        
        # Step 4: Relative Competitiveness
        comp = analyze_relative_competitiveness(df, entity_col, metric_cols, target_entity, higher_better, weights)
        results['competitiveness'] = comp
        visualizations['competitiveness_chart'] = create_competitiveness_chart(comp)
        
        # Step 5: Improvement Simulation
        sim = simulate_improvement(df, entity_col, metric_cols, target_entity, higher_better, weights, sw)
        results['simulation'] = sim
        visualizations['simulation_chart'] = create_simulation_chart(sim)
        
        report = generate_report(pos, sw, adv, comp, sim, target_entity)
        insights = generate_insights(pos, sw, adv, comp, sim)
        
        summary = {
            'target_entity': target_entity,
            'n_entities': len(df),
            'n_metrics': len(metric_cols),
            'current_rank': comp['target_rank'],
            'overall_score': adv['overall_score'],
            'n_strengths': sw['n_strengths'],
            'n_weaknesses': sw['n_weaknesses']
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
