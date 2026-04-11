"""
ROI Analysis API
비용 대비 성과(ROI) 분석을 위한 5단계 프레임워크
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


# =============================================================================
# Request/Response Models
# =============================================================================
class ROIAnalysisRequest(BaseModel):
    data: List[Dict[str, Any]]
    cost_col: str  # 비용 컬럼
    performance_col: str  # 성과 컬럼
    channel_col: Optional[str] = None  # 채널/그룹 컬럼
    channel_name_col: Optional[str] = None  # 채널 이름 컬럼
    additional_cost_cols: Optional[List[str]] = None  # 추가 비용 컬럼들


# =============================================================================
# Utility Functions
# =============================================================================
def _to_native_type(obj):
    """Convert numpy types to native Python types"""
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
    if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
        return str(obj)
    return obj


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64 string"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return b64


# =============================================================================
# Step 1: Cost & Performance Overview (총 비용 및 성과 현황)
# =============================================================================
def analyze_cost_performance_overview(df: pd.DataFrame, cost_col: str, performance_col: str,
                                       additional_cost_cols: Optional[List[str]] = None) -> Dict:
    """Analyze overall cost and performance status"""
    
    # 기본 통계
    total_cost = df[cost_col].sum()
    total_performance = df[performance_col].sum()
    avg_cost = df[cost_col].mean()
    avg_performance = df[performance_col].mean()
    
    # ROI 계산 (성과/비용)
    overall_roi = (total_performance / total_cost * 100) if total_cost > 0 else 0
    
    # 개별 ROI
    df_temp = df.copy()
    df_temp['_roi'] = df_temp.apply(
        lambda x: (x[performance_col] / x[cost_col] * 100) if x[cost_col] > 0 else 0, axis=1
    )
    
    avg_roi = df_temp['_roi'].mean()
    median_roi = df_temp['_roi'].median()
    std_roi = df_temp['_roi'].std()
    min_roi = df_temp['_roi'].min()
    max_roi = df_temp['_roi'].max()
    
    # ROI 분포 (Tier)
    def categorize_roi(roi):
        if roi >= 150:
            return 'Excellent (≥150%)'
        elif roi >= 100:
            return 'Good (100-150%)'
        elif roi >= 50:
            return 'Moderate (50-100%)'
        else:
            return 'Low (<50%)'
    
    df_temp['_roi_tier'] = df_temp['_roi'].apply(categorize_roi)
    tier_distribution = df_temp['_roi_tier'].value_counts().to_dict()
    
    # 추가 비용 컬럼 합계
    additional_costs = {}
    if additional_cost_cols:
        for col in additional_cost_cols:
            if col in df.columns:
                additional_costs[col] = _to_native_type(df[col].sum())
    
    # Top/Bottom performers
    top_performers = df_temp.nlargest(5, '_roi')[[cost_col, performance_col, '_roi']].to_dict('records')
    bottom_performers = df_temp.nsmallest(5, '_roi')[[cost_col, performance_col, '_roi']].to_dict('records')
    
    return {
        'total_cost': _to_native_type(total_cost),
        'total_performance': _to_native_type(total_performance),
        'avg_cost': _to_native_type(avg_cost),
        'avg_performance': _to_native_type(avg_performance),
        'overall_roi': _to_native_type(overall_roi),
        'avg_roi': _to_native_type(avg_roi),
        'median_roi': _to_native_type(median_roi),
        'std_roi': _to_native_type(std_roi),
        'min_roi': _to_native_type(min_roi),
        'max_roi': _to_native_type(max_roi),
        'n_records': len(df),
        'tier_distribution': {k: _to_native_type(v) for k, v in tier_distribution.items()},
        'additional_costs': additional_costs,
        'top_performers': [{k: _to_native_type(v) for k, v in r.items()} for r in top_performers],
        'bottom_performers': [{k: _to_native_type(v) for k, v in r.items()} for r in bottom_performers]
    }


def create_overview_chart(overview: Dict, cost_col: str, performance_col: str) -> str:
    """Create overview visualization"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Chart 1: Cost vs Performance (Bar)
    ax1 = axes[0]
    categories = ['Total Cost', 'Total Performance']
    values = [overview['total_cost'], overview['total_performance']]
    colors = ['#3b82f6', '#10b981']
    bars = ax1.bar(categories, values, color=colors)
    ax1.set_title('Cost vs Performance', fontsize=11, fontweight='bold')
    ax1.ticklabel_format(style='plain', axis='y')
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{val:,.0f}',
                ha='center', va='bottom', fontsize=9)
    
    # Chart 2: ROI Distribution (Pie)
    ax2 = axes[1]
    tier_dist = overview['tier_distribution']
    if tier_dist:
        tier_order = ['Excellent (≥150%)', 'Good (100-150%)', 'Moderate (50-100%)', 'Low (<50%)']
        tier_colors = ['#10b981', '#3b82f6', '#f59e0b', '#ef4444']
        labels = []
        sizes = []
        pie_colors = []
        for tier, color in zip(tier_order, tier_colors):
            if tier in tier_dist:
                labels.append(tier)
                sizes.append(tier_dist[tier])
                pie_colors.append(color)
        if sizes:
            ax2.pie(sizes, labels=labels, colors=pie_colors, autopct='%1.1f%%', startangle=90)
    ax2.set_title('ROI Distribution', fontsize=11, fontweight='bold')
    
    # Chart 3: ROI Summary Stats
    ax3 = axes[2]
    stats_labels = ['Average', 'Median', 'Min', 'Max']
    stats_values = [overview['avg_roi'], overview['median_roi'], overview['min_roi'], overview['max_roi']]
    y_pos = range(len(stats_labels))
    colors = ['#3b82f6', '#3b82f6', '#ef4444', '#10b981']
    bars = ax3.barh(y_pos, stats_values, color=colors)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(stats_labels)
    ax3.set_xlabel('ROI (%)')
    ax3.set_title('ROI Statistics', fontsize=11, fontweight='bold')
    ax3.axvline(x=100, color='gray', linestyle='--', alpha=0.5)
    for bar, val in zip(bars, stats_values):
        ax3.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
                va='center', fontsize=9)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Step 2: ROI by Channel (채널별 ROI 비교)
# =============================================================================
def analyze_channel_roi(df: pd.DataFrame, cost_col: str, performance_col: str, 
                        channel_col: str) -> Dict:
    """Analyze ROI by channel/group"""
    
    # 채널별 집계
    channel_stats = df.groupby(channel_col).agg({
        cost_col: ['sum', 'mean', 'count'],
        performance_col: ['sum', 'mean']
    }).reset_index()
    
    channel_stats.columns = ['channel', 'total_cost', 'avg_cost', 'count', 'total_performance', 'avg_performance']
    
    # ROI 계산
    channel_stats['roi'] = channel_stats.apply(
        lambda x: (x['total_performance'] / x['total_cost'] * 100) if x['total_cost'] > 0 else 0, axis=1
    )
    channel_stats['cost_share'] = channel_stats['total_cost'] / channel_stats['total_cost'].sum() * 100
    channel_stats['performance_share'] = channel_stats['total_performance'] / channel_stats['total_performance'].sum() * 100
    
    # 효율성 점수 (성과 점유율 / 비용 점유율)
    channel_stats['efficiency'] = channel_stats.apply(
        lambda x: x['performance_share'] / x['cost_share'] if x['cost_share'] > 0 else 0, axis=1
    )
    
    # 순위
    channel_stats = channel_stats.sort_values('roi', ascending=False)
    channel_stats['rank'] = range(1, len(channel_stats) + 1)
    
    # Best/Worst
    best = channel_stats.iloc[0]
    worst = channel_stats.iloc[-1]
    
    # 통계적 검정 (ANOVA)
    groups = [group[performance_col].values / group[cost_col].values * 100 
              for name, group in df.groupby(channel_col) if len(group) > 0]
    groups = [g[~np.isnan(g) & ~np.isinf(g)] for g in groups]
    groups = [g for g in groups if len(g) > 0]
    
    if len(groups) >= 2:
        try:
            f_stat, p_value = stats.f_oneway(*groups)
            significant = bool(p_value < 0.05)
        except:
            f_stat, p_value, significant = None, None, False
    else:
        f_stat, p_value, significant = None, None, False
    
    # ROI 갭
    roi_gap = best['roi'] - worst['roi']
    
    channel_data = []
    for _, row in channel_stats.iterrows():
        channel_data.append({
            'channel': _to_native_type(row['channel']),
            'rank': _to_native_type(row['rank']),
            'total_cost': _to_native_type(row['total_cost']),
            'total_performance': _to_native_type(row['total_performance']),
            'avg_cost': _to_native_type(row['avg_cost']),
            'avg_performance': _to_native_type(row['avg_performance']),
            'roi': _to_native_type(row['roi']),
            'cost_share': _to_native_type(row['cost_share']),
            'performance_share': _to_native_type(row['performance_share']),
            'efficiency': _to_native_type(row['efficiency']),
            'count': _to_native_type(row['count'])
        })
    
    return {
        'channel_data': channel_data,
        'n_channels': len(channel_stats),
        'best_channel': {
            'channel': _to_native_type(best['channel']),
            'roi': _to_native_type(best['roi']),
            'efficiency': _to_native_type(best['efficiency'])
        },
        'worst_channel': {
            'channel': _to_native_type(worst['channel']),
            'roi': _to_native_type(worst['roi']),
            'efficiency': _to_native_type(worst['efficiency'])
        },
        'roi_gap': _to_native_type(roi_gap),
        'statistical_test': {
            'method': 'ANOVA',
            'f_statistic': _to_native_type(f_stat),
            'p_value': _to_native_type(p_value),
            'significant_difference': significant
        }
    }


def create_channel_chart(channel_data: Dict) -> str:
    """Create channel comparison visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    channels = [c['channel'] for c in channel_data['channel_data']]
    rois = [c['roi'] for c in channel_data['channel_data']]
    cost_shares = [c['cost_share'] for c in channel_data['channel_data']]
    perf_shares = [c['performance_share'] for c in channel_data['channel_data']]
    
    # Chart 1: ROI by Channel
    ax1 = axes[0]
    colors = ['#10b981' if r >= 100 else '#f59e0b' if r >= 50 else '#ef4444' for r in rois]
    bars = ax1.barh(channels, rois, color=colors)
    ax1.axvline(x=100, color='gray', linestyle='--', alpha=0.5, label='Break-even')
    ax1.set_xlabel('ROI (%)')
    ax1.set_title('ROI by Channel', fontsize=11, fontweight='bold')
    for bar, val in zip(bars, rois):
        ax1.text(val + 1, bar.get_y() + bar.get_height()/2, f'{val:.1f}%',
                va='center', fontsize=9)
    
    # Chart 2: Cost vs Performance Share
    ax2 = axes[1]
    x = np.arange(len(channels))
    width = 0.35
    bars1 = ax2.bar(x - width/2, cost_shares, width, label='Cost Share', color='#3b82f6')
    bars2 = ax2.bar(x + width/2, perf_shares, width, label='Performance Share', color='#10b981')
    ax2.set_ylabel('Share (%)')
    ax2.set_title('Cost vs Performance Share', fontsize=11, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(channels, rotation=45, ha='right')
    ax2.legend()
    
    plt.tight_layout()
    return _fig_to_base64(fig)
# =============================================================================
# Step 3: Cost-Performance Correlation (비용과 성과 상관성)
# =============================================================================
def analyze_cost_performance_correlation(df: pd.DataFrame, cost_col: str, performance_col: str,
                                          additional_cost_cols: Optional[List[str]] = None) -> Dict:
    """Analyze correlation between cost and performance"""
    
    correlations = []
    
    # 메인 비용-성과 상관관계
    cost_vals = df[cost_col].values
    perf_vals = df[performance_col].values
    
    # Remove NaN
    mask = ~(np.isnan(cost_vals) | np.isnan(perf_vals))
    cost_clean = cost_vals[mask]
    perf_clean = perf_vals[mask]
    
    if len(cost_clean) >= 3:
        # Pearson
        pearson_r, pearson_p = stats.pearsonr(cost_clean, perf_clean)
        # Spearman
        spearman_r, spearman_p = stats.spearmanr(cost_clean, perf_clean)
        
        # 상관계수 해석
        abs_r = abs(pearson_r)
        if abs_r >= 0.7:
            strength = 'strong'
        elif abs_r >= 0.4:
            strength = 'moderate'
        elif abs_r >= 0.2:
            strength = 'weak'
        else:
            strength = 'very weak'
        
        direction = 'positive' if pearson_r > 0 else 'negative'
        
        correlations.append({
            'cost_variable': cost_col,
            'pearson_r': _to_native_type(pearson_r),
            'pearson_p': _to_native_type(pearson_p),
            'spearman_r': _to_native_type(spearman_r),
            'spearman_p': _to_native_type(spearman_p),
            'strength': strength,
            'direction': direction,
            'significant': bool(pearson_p < 0.05),
            'n_samples': len(cost_clean)
        })
    
    # 추가 비용 컬럼들
    if additional_cost_cols:
        for col in additional_cost_cols:
            if col in df.columns and col != cost_col:
                col_vals = pd.to_numeric(df[col], errors='coerce').values
                mask = ~(np.isnan(col_vals) | np.isnan(perf_vals))
                col_clean = col_vals[mask]
                perf_clean2 = perf_vals[mask]
                
                if len(col_clean) >= 3:
                    pearson_r, pearson_p = stats.pearsonr(col_clean, perf_clean2)
                    spearman_r, spearman_p = stats.spearmanr(col_clean, perf_clean2)
                    
                    abs_r = abs(pearson_r)
                    if abs_r >= 0.7:
                        strength = 'strong'
                    elif abs_r >= 0.4:
                        strength = 'moderate'
                    elif abs_r >= 0.2:
                        strength = 'weak'
                    else:
                        strength = 'very weak'
                    
                    direction = 'positive' if pearson_r > 0 else 'negative'
                    
                    correlations.append({
                        'cost_variable': col,
                        'pearson_r': _to_native_type(pearson_r),
                        'pearson_p': _to_native_type(pearson_p),
                        'spearman_r': _to_native_type(spearman_r),
                        'spearman_p': _to_native_type(spearman_p),
                        'strength': strength,
                        'direction': direction,
                        'significant': bool(pearson_p < 0.05),
                        'n_samples': len(col_clean)
                    })
    
    # 가장 강한 상관관계
    strongest = None
    if correlations:
        strongest = max(correlations, key=lambda x: abs(x['pearson_r']))
    
    # 유의미한 상관관계 수
    significant_count = sum(1 for c in correlations if c['significant'])
    
    return {
        'correlations': correlations,
        'strongest_correlation': strongest,
        'n_cost_variables': len(correlations),
        'significant_count': significant_count,
        'main_cost_col': cost_col,
        'performance_col': performance_col
    }


def create_correlation_chart(df: pd.DataFrame, cost_col: str, performance_col: str,
                              correlation_data: Dict) -> str:
    """Create correlation visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Chart 1: Scatter plot
    ax1 = axes[0]
    ax1.scatter(df[cost_col], df[performance_col], alpha=0.6, color='#3b82f6', edgecolor='white')
    
    # Trend line
    z = np.polyfit(df[cost_col].dropna(), df[performance_col].dropna(), 1)
    p = np.poly1d(z)
    x_line = np.linspace(df[cost_col].min(), df[cost_col].max(), 100)
    ax1.plot(x_line, p(x_line), color='#ef4444', linestyle='--', linewidth=2, label='Trend')
    
    strongest = correlation_data.get('strongest_correlation', {})
    r_val = strongest.get('pearson_r', 0) if strongest else 0
    ax1.set_xlabel(cost_col)
    ax1.set_ylabel(performance_col)
    ax1.set_title(f'Cost vs Performance (r={r_val:.3f})', fontsize=11, fontweight='bold')
    ax1.legend()
    
    # Chart 2: Correlation bars
    ax2 = axes[1]
    correlations = correlation_data.get('correlations', [])
    if correlations:
        vars_list = [c['cost_variable'] for c in correlations]
        r_values = [c['pearson_r'] for c in correlations]
        colors = ['#10b981' if r > 0 else '#ef4444' for r in r_values]
        bars = ax2.barh(vars_list, r_values, color=colors)
        ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        ax2.set_xlabel('Correlation (r)')
        ax2.set_title('Cost Variables Correlation', fontsize=11, fontweight='bold')
        ax2.set_xlim(-1, 1)
        for bar, val in zip(bars, r_values):
            ax2.text(val + 0.02 if val >= 0 else val - 0.1, bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Step 4: Efficiency Blockers (가성비 저해 요인)
# =============================================================================
def analyze_efficiency_blockers(df: pd.DataFrame, cost_col: str, performance_col: str,
                                 channel_col: Optional[str] = None,
                                 additional_cost_cols: Optional[List[str]] = None) -> Dict:
    """Identify factors that block efficiency (ROI)"""
    
    try:
        from statsmodels.regression.linear_model import OLS
        from statsmodels.tools import add_constant
    except ImportError:
        return {'error': 'statsmodels not installed'}
    
    # ROI 계산
    df_temp = df.copy()
    df_temp['_roi'] = df_temp.apply(
        lambda x: (x[performance_col] / x[cost_col] * 100) if x[cost_col] > 0 else np.nan, axis=1
    )
    df_temp = df_temp.dropna(subset=['_roi'])
    
    if len(df_temp) < 5:
        return {'error': 'Insufficient data for analysis'}
    
    # 특징 변수 준비
    feature_cols = [cost_col]
    if additional_cost_cols:
        feature_cols.extend([c for c in additional_cost_cols if c in df_temp.columns])
    
    # 채널 더미 변수
    if channel_col and channel_col in df_temp.columns:
        channel_dummies = pd.get_dummies(df_temp[channel_col], prefix='channel', drop_first=True)
        df_temp = pd.concat([df_temp, channel_dummies], axis=1)
        feature_cols.extend(channel_dummies.columns.tolist())
    
    # 회귀 분석
    X = df_temp[feature_cols].apply(pd.to_numeric, errors='coerce')
    X = X.fillna(X.mean())
    y = df_temp['_roi']
    
    try:
        X_const = add_constant(X)
        model = OLS(y, X_const).fit()
        r_squared = model.rsquared
        
        # 계수 분석
        blockers = []
        for col in feature_cols:
            if col in model.params.index:
                coef = model.params[col]
                pval = model.pvalues[col]
                
                # 음수 계수 = 효율성 저해 요인
                if coef < 0 and pval < 0.1:
                    blockers.append({
                        'factor': col.replace('channel_', ''),
                        'coefficient': _to_native_type(coef),
                        'p_value': _to_native_type(pval),
                        'significant': bool(pval < 0.05),
                        'impact': 'negative'
                    })
        
        # 잔차 분석 - 비효율적인 항목 식별
        df_temp['_predicted_roi'] = model.predict(X_const)
        df_temp['_residual'] = df_temp['_roi'] - df_temp['_predicted_roi']
        
        # 음의 잔차 = 예상보다 ROI가 낮음
        inefficient_items = df_temp[df_temp['_residual'] < -df_temp['_residual'].std()]
        
        inefficient_summary = []
        if channel_col:
            inefficient_by_channel = inefficient_items.groupby(channel_col).size().to_dict()
            inefficient_summary = [{'channel': _to_native_type(k), 'count': _to_native_type(v)} 
                                   for k, v in inefficient_by_channel.items()]
        
        # ROI 분산 원인 분해
        total_variance = df_temp['_roi'].var()
        explained_variance = model.predict(X_const).var()
        
        return {
            'blockers': sorted(blockers, key=lambda x: x['coefficient']),
            'n_blockers': len(blockers),
            'r_squared': _to_native_type(r_squared),
            'model_quality': 'good' if r_squared > 0.5 else 'moderate' if r_squared > 0.2 else 'low',
            'inefficient_count': len(inefficient_items),
            'inefficient_by_channel': inefficient_summary,
            'variance_explained': _to_native_type(explained_variance / total_variance * 100) if total_variance > 0 else 0,
            'n_observations': len(df_temp)
        }
    except Exception as e:
        return {'error': str(e)}


def create_blockers_chart(blockers_data: Dict) -> str:
    """Create efficiency blockers visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Chart 1: Blockers impact
    ax1 = axes[0]
    blockers = blockers_data.get('blockers', [])
    if blockers:
        factors = [b['factor'][:15] for b in blockers]
        impacts = [b['coefficient'] for b in blockers]
        colors = ['#ef4444' if b['significant'] else '#f59e0b' for b in blockers]
        bars = ax1.barh(factors, impacts, color=colors)
        ax1.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        ax1.set_xlabel('Impact on ROI')
        ax1.set_title('Efficiency Blockers', fontsize=11, fontweight='bold')
        for bar, val in zip(bars, impacts):
            ax1.text(val - 0.5, bar.get_y() + bar.get_height()/2, f'{val:.2f}',
                    va='center', fontsize=9)
    else:
        ax1.text(0.5, 0.5, 'No significant blockers found', ha='center', va='center',
                transform=ax1.transAxes, fontsize=12)
        ax1.set_title('Efficiency Blockers', fontsize=11, fontweight='bold')
    
    # Chart 2: Inefficient by channel
    ax2 = axes[1]
    inefficient = blockers_data.get('inefficient_by_channel', [])
    if inefficient:
        channels = [i['channel'] for i in inefficient]
        counts = [i['count'] for i in inefficient]
        ax2.bar(channels, counts, color='#ef4444', alpha=0.7)
        ax2.set_ylabel('Count')
        ax2.set_title('Inefficient Items by Channel', fontsize=11, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
    else:
        ax2.text(0.5, 0.5, 'No channel data', ha='center', va='center',
                transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Inefficient Items by Channel', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# =============================================================================
# Step 5: Budget Optimization (예산 최적화 시나리오)
# =============================================================================
def simulate_budget_optimization(df: pd.DataFrame, cost_col: str, performance_col: str,
                                  channel_col: Optional[str] = None,
                                  blockers_data: Optional[Dict] = None) -> Dict:
    """Simulate budget optimization scenarios"""
    
    current_total_cost = df[cost_col].sum()
    current_total_performance = df[performance_col].sum()
    current_roi = (current_total_performance / current_total_cost * 100) if current_total_cost > 0 else 0
    
    scenarios = []
    
    # Scenario 1: 10% 예산 증가
    new_cost_1 = current_total_cost * 1.1
    # 한계 수익 체감 가정 (90% 효율)
    expected_perf_1 = current_total_performance * 1.09
    expected_roi_1 = (expected_perf_1 / new_cost_1 * 100)
    scenarios.append({
        'name': '10% Budget Increase',
        'cost_change': '+10%',
        'new_total_cost': _to_native_type(new_cost_1),
        'expected_performance': _to_native_type(expected_perf_1),
        'expected_roi': _to_native_type(expected_roi_1),
        'roi_change': _to_native_type(expected_roi_1 - current_roi),
        'performance_change_pct': _to_native_type((expected_perf_1 / current_total_performance - 1) * 100)
    })
    
    # Scenario 2: 10% 예산 감소
    new_cost_2 = current_total_cost * 0.9
    expected_perf_2 = current_total_performance * 0.88  # 비선형 감소
    expected_roi_2 = (expected_perf_2 / new_cost_2 * 100)
    scenarios.append({
        'name': '10% Budget Cut',
        'cost_change': '-10%',
        'new_total_cost': _to_native_type(new_cost_2),
        'expected_performance': _to_native_type(expected_perf_2),
        'expected_roi': _to_native_type(expected_roi_2),
        'roi_change': _to_native_type(expected_roi_2 - current_roi),
        'performance_change_pct': _to_native_type((expected_perf_2 / current_total_performance - 1) * 100)
    })
    
    # Scenario 3: 채널 최적화 (있는 경우)
    if channel_col:
        channel_stats = df.groupby(channel_col).agg({
            cost_col: 'sum',
            performance_col: 'sum'
        }).reset_index()
        channel_stats['roi'] = channel_stats[performance_col] / channel_stats[cost_col] * 100
        
        # 상위 ROI 채널로 예산 재분배
        avg_roi = channel_stats['roi'].mean()
        high_roi_channels = channel_stats[channel_stats['roi'] > avg_roi]
        low_roi_channels = channel_stats[channel_stats['roi'] <= avg_roi]
        
        if len(high_roi_channels) > 0 and len(low_roi_channels) > 0:
            # 하위 채널 예산 20% 상위 채널로 이동
            budget_shift = low_roi_channels[cost_col].sum() * 0.2
            
            # 예상 성과 계산
            high_roi_avg = high_roi_channels['roi'].mean()
            additional_perf = budget_shift * high_roi_avg / 100
            lost_perf = budget_shift * low_roi_channels['roi'].mean() / 100
            
            new_total_perf = current_total_performance + additional_perf - lost_perf
            expected_roi_3 = (new_total_perf / current_total_cost * 100)
            
            scenarios.append({
                'name': 'Channel Reallocation',
                'cost_change': '0% (reallocated)',
                'new_total_cost': _to_native_type(current_total_cost),
                'expected_performance': _to_native_type(new_total_perf),
                'expected_roi': _to_native_type(expected_roi_3),
                'roi_change': _to_native_type(expected_roi_3 - current_roi),
                'performance_change_pct': _to_native_type((new_total_perf / current_total_performance - 1) * 100),
                'detail': f"Shift 20% budget from low ROI to high ROI channels"
            })
    
    # 최적 시나리오
    best_scenario = max(scenarios, key=lambda x: x['expected_roi']) if scenarios else None
    
    # 추천 사항
    recommendations = []
    if best_scenario:
        recommendations.append(f"Consider '{best_scenario['name']}' for best ROI improvement")
    
    if channel_col and blockers_data and blockers_data.get('inefficient_by_channel'):
        inefficient = blockers_data['inefficient_by_channel']
        if inefficient:
            worst = max(inefficient, key=lambda x: x['count'])
            recommendations.append(f"Review efficiency in '{worst['channel']}' channel")
    
    return {
        'current_state': {
            'total_cost': _to_native_type(current_total_cost),
            'total_performance': _to_native_type(current_total_performance),
            'current_roi': _to_native_type(current_roi)
        },
        'scenarios': scenarios,
        'best_scenario': best_scenario,
        'recommendations': recommendations,
        'model_reliability': 'moderate'  # 단순 시뮬레이션이므로
    }


def create_optimization_chart(optimization_data: Dict) -> str:
    """Create budget optimization visualization"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    scenarios = optimization_data.get('scenarios', [])
    current = optimization_data.get('current_state', {})
    
    # Chart 1: ROI Comparison
    ax1 = axes[0]
    names = ['Current'] + [s['name'] for s in scenarios]
    rois = [current.get('current_roi', 0)] + [s['expected_roi'] for s in scenarios]
    colors = ['#6b7280'] + ['#10b981' if s['roi_change'] > 0 else '#ef4444' for s in scenarios]
    
    bars = ax1.bar(names, rois, color=colors)
    ax1.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Break-even')
    ax1.set_ylabel('ROI (%)')
    ax1.set_title('ROI by Scenario', fontsize=11, fontweight='bold')
    ax1.tick_params(axis='x', rotation=45)
    for bar, val in zip(bars, rois):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f'{val:.1f}%',
                ha='center', fontsize=9)
    
    # Chart 2: Performance Change
    ax2 = axes[1]
    if scenarios:
        names = [s['name'] for s in scenarios]
        changes = [s['performance_change_pct'] for s in scenarios]
        colors = ['#10b981' if c > 0 else '#ef4444' for c in changes]
        bars = ax2.barh(names, changes, color=colors)
        ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.3)
        ax2.set_xlabel('Performance Change (%)')
        ax2.set_title('Expected Performance Impact', fontsize=11, fontweight='bold')
        for bar, val in zip(bars, changes):
            ax2.text(val + 0.2 if val >= 0 else val - 1.5, bar.get_y() + bar.get_height()/2,
                    f'{val:+.1f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    return _fig_to_base64(fig)
# =============================================================================
# Insight Generation
# =============================================================================
def generate_roi_insights(overview: Dict, channel: Optional[Dict], correlation: Optional[Dict],
                          blockers: Optional[Dict], optimization: Optional[Dict]) -> List[Dict]:
    """Generate key insights from ROI analysis"""
    insights = []
    
    # Overview insights
    if overview:
        overall_roi = overview.get('overall_roi', 0)
        if overall_roi >= 150:
            insights.append({
                'title': 'Excellent ROI Performance',
                'description': f"Overall ROI is {overall_roi:.1f}%, significantly above break-even.",
                'status': 'positive'
            })
        elif overall_roi >= 100:
            insights.append({
                'title': 'Positive ROI',
                'description': f"Overall ROI is {overall_roi:.1f}%, above break-even point.",
                'status': 'positive'
            })
        else:
            insights.append({
                'title': 'ROI Below Break-even',
                'description': f"Overall ROI is {overall_roi:.1f}%, below 100% break-even threshold.",
                'status': 'warning'
            })
    
    # Channel insights
    if channel and not channel.get('error'):
        best = channel.get('best_channel', {})
        worst = channel.get('worst_channel', {})
        gap = channel.get('roi_gap', 0)
        
        if gap > 50:
            insights.append({
                'title': 'Significant Channel Disparity',
                'description': f"ROI gap of {gap:.1f}% between {best.get('channel')} and {worst.get('channel')}.",
                'status': 'warning'
            })
        
        if channel.get('statistical_test', {}).get('significant_difference'):
            insights.append({
                'title': 'Statistically Different Channels',
                'description': "Channel performance differences are statistically significant.",
                'status': 'neutral'
            })
    
    # Correlation insights
    if correlation and correlation.get('strongest_correlation'):
        strongest = correlation['strongest_correlation']
        if strongest.get('significant'):
            insights.append({
                'title': f"Cost-Performance Link: {strongest['cost_variable']}",
                'description': f"{strongest['strength'].capitalize()} {strongest['direction']} correlation (r={strongest['pearson_r']:.3f}).",
                'status': 'positive' if strongest['direction'] == 'positive' else 'neutral'
            })
    
    # Blockers insights
    if blockers and not blockers.get('error'):
        n_blockers = blockers.get('n_blockers', 0)
        if n_blockers > 0:
            top_blocker = blockers['blockers'][0] if blockers.get('blockers') else None
            if top_blocker:
                insights.append({
                    'title': f"Key Efficiency Blocker: {top_blocker['factor']}",
                    'description': f"Negatively impacts ROI with coefficient of {top_blocker['coefficient']:.2f}.",
                    'status': 'warning'
                })
        
        inefficient_count = blockers.get('inefficient_count', 0)
        if inefficient_count > 0:
            insights.append({
                'title': 'Underperforming Items Identified',
                'description': f"{inefficient_count} items performing below expected ROI levels.",
                'status': 'warning'
            })
    
    # Optimization insights
    if optimization and optimization.get('best_scenario'):
        best = optimization['best_scenario']
        if best['roi_change'] > 0:
            insights.append({
                'title': 'Optimization Opportunity',
                'description': f"'{best['name']}' could improve ROI by {best['roi_change']:.1f}%.",
                'status': 'positive'
            })
    
    return insights


# =============================================================================
# Report Generation
# =============================================================================
def generate_roi_report(overview: Dict, channel: Optional[Dict], correlation: Optional[Dict],
                        blockers: Optional[Dict], optimization: Optional[Dict]) -> Dict[str, Any]:
    """Generate structured report for ROI analysis"""
    report = {}
    
    # Step 1: Cost & Performance Overview
    if overview:
        total_cost = overview.get('total_cost', 0)
        total_perf = overview.get('total_performance', 0)
        overall_roi = overview.get('overall_roi', 0)
        avg_roi = overview.get('avg_roi', 0)
        
        finding = f"Overall ROI is {overall_roi:.1f}% with total cost of {total_cost:,.0f}"
        
        detail = f"The analysis covers {overview.get('n_records', 0)} records with a combined cost of {total_cost:,.0f} "
        detail += f"generating total performance of {total_perf:,.0f}. "
        detail += f"This yields an overall ROI of {overall_roi:.1f}%, "
        detail += f"{'exceeding' if overall_roi >= 100 else 'falling below'} the break-even threshold. "
        detail += f"Individual ROI values average {avg_roi:.1f}% with a range from {overview.get('min_roi', 0):.1f}% "
        detail += f"to {overview.get('max_roi', 0):.1f}%. "
        
        tier_dist = overview.get('tier_distribution', {})
        excellent = tier_dist.get('Excellent (≥150%)', 0)
        low = tier_dist.get('Low (<50%)', 0)
        if excellent > 0:
            detail += f"{excellent} items show excellent ROI above 150%. "
        if low > 0:
            detail += f"However, {low} items have concerning ROI below 50%, requiring attention."
        
        report['step1_overview'] = {
            'title': '1. Cost & Performance Overview',
            'question': 'What is the overall cost-performance status?',
            'finding': finding,
            'detail': detail
        }
    
    # Step 2: Channel ROI
    if channel and not channel.get('error'):
        best = channel.get('best_channel', {})
        worst = channel.get('worst_channel', {})
        gap = channel.get('roi_gap', 0)
        n_channels = channel.get('n_channels', 0)
        
        finding = f"{best.get('channel', 'N/A')} leads with {best.get('roi', 0):.1f}% ROI, "
        finding += f"while {worst.get('channel', 'N/A')} trails at {worst.get('roi', 0):.1f}%"
        
        detail = f"Analysis across {n_channels} channels reveals varying ROI performance. "
        detail += f"{best.get('channel', 'N/A')} achieves the highest ROI at {best.get('roi', 0):.1f}% "
        detail += f"with an efficiency score of {best.get('efficiency', 0):.2f}. "
        detail += f"In contrast, {worst.get('channel', 'N/A')} shows the lowest ROI at {worst.get('roi', 0):.1f}%. "
        detail += f"The ROI gap of {gap:.1f}% between top and bottom performers "
        
        stat_test = channel.get('statistical_test', {})
        if stat_test.get('significant_difference'):
            detail += f"is statistically significant (ANOVA p={stat_test.get('p_value', 0):.4f}), "
            detail += "indicating systematic differences rather than random variation. "
            detail += "This suggests targeted intervention could yield meaningful improvements."
        else:
            detail += "is not statistically significant, though operational improvements may still be beneficial."
        
        report['step2_channel'] = {
            'title': '2. ROI by Channel',
            'question': 'Which channels deliver the best return on investment?',
            'finding': finding,
            'detail': detail
        }
    else:
        report['step2_channel'] = {
            'title': '2. ROI by Channel',
            'question': 'Which channels deliver the best return on investment?',
            'finding': 'Channel analysis not available',
            'detail': 'Channel column was not specified. To compare ROI across channels, segments, or teams, include a grouping variable in the configuration.'
        }
    
    # Step 3: Correlation
    if correlation and correlation.get('correlations'):
        strongest = correlation.get('strongest_correlation', {})
        n_vars = correlation.get('n_cost_variables', 0)
        sig_count = correlation.get('significant_count', 0)
        
        if strongest:
            finding = f"{strongest.get('cost_variable', 'N/A')} shows {strongest.get('strength', 'N/A')} "
            finding += f"{strongest.get('direction', 'N/A')} correlation with performance (r={strongest.get('pearson_r', 0):.3f})"
            
            detail = f"Correlation analysis examined {n_vars} cost variable(s) against performance outcomes. "
            detail += f"{sig_count} showed statistically significant relationships. "
            detail += f"The strongest correlation was found with {strongest.get('cost_variable', 'N/A')} "
            detail += f"(r={strongest.get('pearson_r', 0):.3f}, p={strongest.get('pearson_p', 0):.4f}), "
            detail += f"indicating a {strongest.get('strength', 'N/A')} {strongest.get('direction', 'N/A')} relationship. "
            
            if strongest.get('direction') == 'positive':
                detail += "This positive relationship suggests that increased investment tends to drive higher performance returns. "
            else:
                detail += "This inverse relationship warrants investigation into potential diminishing returns or inefficiencies. "
            
            detail += "Understanding these correlations helps optimize budget allocation for maximum impact."
        else:
            finding = "No significant cost-performance correlations identified"
            detail = "The analysis did not find statistically significant correlations between cost variables and performance."
        
        report['step3_correlation'] = {
            'title': '3. Cost-Performance Correlation',
            'question': 'How does spending relate to performance outcomes?',
            'finding': finding,
            'detail': detail
        }
    else:
        report['step3_correlation'] = {
            'title': '3. Cost-Performance Correlation',
            'question': 'How does spending relate to performance outcomes?',
            'finding': 'Correlation analysis not available',
            'detail': 'Insufficient data or cost variables for correlation analysis.'
        }
    
    # Step 4: Blockers
    if blockers and not blockers.get('error'):
        n_blockers = blockers.get('n_blockers', 0)
        r_squared = blockers.get('r_squared', 0)
        inefficient_count = blockers.get('inefficient_count', 0)
        
        if n_blockers > 0:
            top_blocker = blockers['blockers'][0]
            finding = f"{top_blocker['factor']} identified as primary efficiency blocker"
            
            detail = f"Regression analysis (R²={r_squared:.3f}) identified {n_blockers} factor(s) that negatively impact ROI efficiency. "
            detail += f"The primary blocker is {top_blocker['factor']} with a coefficient of {top_blocker['coefficient']:.3f} "
            detail += f"(p={top_blocker['p_value']:.4f}). "
            detail += f"Additionally, {inefficient_count} individual items were identified as performing below expected ROI levels "
            detail += "based on their cost profile. "
            
            if blockers.get('inefficient_by_channel'):
                worst_channel = max(blockers['inefficient_by_channel'], key=lambda x: x['count'])
                detail += f"The channel with most inefficient items is {worst_channel['channel']} ({worst_channel['count']} items). "
            
            detail += "Addressing these blockers through process improvements or resource reallocation could improve overall ROI."
        else:
            finding = "No significant efficiency blockers identified"
            detail = f"The regression model (R²={r_squared:.3f}) did not identify significant negative factors affecting ROI. "
            detail += "This suggests current operations are reasonably optimized, though continuous monitoring is recommended."
        
        report['step4_blockers'] = {
            'title': '4. Efficiency Blockers',
            'question': 'What factors are hurting our cost efficiency?',
            'finding': finding,
            'detail': detail
        }
    else:
        error_msg = blockers.get('error', 'Not available') if blockers else 'Not configured'
        report['step4_blockers'] = {
            'title': '4. Efficiency Blockers',
            'question': 'What factors are hurting our cost efficiency?',
            'finding': 'Blocker analysis not available',
            'detail': f'Could not perform blocker analysis. {error_msg}'
        }
    
    # Step 5: Optimization
    if optimization and optimization.get('scenarios'):
        best = optimization.get('best_scenario', {})
        current = optimization.get('current_state', {})
        
        finding = f"'{best.get('name', 'N/A')}' scenario could achieve {best.get('expected_roi', 0):.1f}% ROI"
        
        detail = f"Starting from current ROI of {current.get('current_roi', 0):.1f}%, "
        detail += f"simulation of {len(optimization['scenarios'])} scenarios identifies optimization opportunities. "
        
        for scenario in optimization['scenarios']:
            detail += f"'{scenario['name']}' ({scenario['cost_change']}) would result in expected ROI of {scenario['expected_roi']:.1f}% "
            detail += f"with performance change of {scenario['performance_change_pct']:+.1f}%. "
        
        detail += f"The recommended scenario is '{best.get('name', 'N/A')}' which shows the most favorable ROI outcome. "
        
        if optimization.get('recommendations'):
            detail += "Key recommendations: " + "; ".join(optimization['recommendations']) + "."
        
        report['step5_optimization'] = {
            'title': '5. Budget Optimization',
            'question': 'How can we optimize budget allocation for better ROI?',
            'finding': finding,
            'detail': detail
        }
    else:
        report['step5_optimization'] = {
            'title': '5. Budget Optimization',
            'question': 'How can we optimize budget allocation for better ROI?',
            'finding': 'Optimization scenarios not available',
            'detail': 'Insufficient data to generate optimization scenarios.'
        }
    
    return report


# =============================================================================
# Main API Endpoint
# =============================================================================
@router.post("/roi-analysis")
async def analyze_roi(request: ROIAnalysisRequest):
    try:
        df = pd.DataFrame(request.data)
        
        if len(df) < 3:
            raise HTTPException(status_code=400, detail="Need at least 3 data points")
        
        if request.cost_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Cost column '{request.cost_col}' not found")
        
        if request.performance_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Performance column '{request.performance_col}' not found")
        
        # Convert to numeric
        df[request.cost_col] = pd.to_numeric(df[request.cost_col], errors='coerce')
        df[request.performance_col] = pd.to_numeric(df[request.performance_col], errors='coerce')
        
        results = {}
        visualizations = {}
        
        # Step 1: Overview
        overview = analyze_cost_performance_overview(
            df, request.cost_col, request.performance_col, request.additional_cost_cols
        )
        results['overview'] = overview
        visualizations['overview_chart'] = create_overview_chart(overview, request.cost_col, request.performance_col)
        
        # Step 2: Channel ROI
        channel = None
        if request.channel_col and request.channel_col in df.columns:
            channel = analyze_channel_roi(df, request.cost_col, request.performance_col, request.channel_col)
            results['channel'] = channel
            visualizations['channel_chart'] = create_channel_chart(channel)
        
        # Step 3: Correlation
        correlation = analyze_cost_performance_correlation(
            df, request.cost_col, request.performance_col, request.additional_cost_cols
        )
        results['correlation'] = correlation
        visualizations['correlation_chart'] = create_correlation_chart(
            df, request.cost_col, request.performance_col, correlation
        )
        
        # Step 4: Blockers
        blockers = analyze_efficiency_blockers(
            df, request.cost_col, request.performance_col, 
            request.channel_col, request.additional_cost_cols
        )
        results['blockers'] = blockers
        if not blockers.get('error'):
            visualizations['blockers_chart'] = create_blockers_chart(blockers)
        
        # Step 5: Optimization
        optimization = simulate_budget_optimization(
            df, request.cost_col, request.performance_col, request.channel_col, blockers
        )
        results['optimization'] = optimization
        visualizations['optimization_chart'] = create_optimization_chart(optimization)
        
        # Generate Insights & Report
        insights = generate_roi_insights(overview, channel, correlation, blockers, optimization)
        report = generate_roi_report(overview, channel, correlation, blockers, optimization)
        
        # Summary
        summary = {
            'total_cost': overview['total_cost'],
            'total_performance': overview['total_performance'],
            'overall_roi': overview['overall_roi'],
            'n_records': overview['n_records'],
            'n_channels': channel['n_channels'] if channel else None,
            'best_channel': channel['best_channel']['channel'] if channel else None,
            'primary_blocker': blockers['blockers'][0]['factor'] if blockers and blockers.get('blockers') else None
        }
        
        return {
            'success': True,
            'results': results,
            'visualizations': visualizations,
            'key_insights': insights,
            'report': report,
            'summary': summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ROI analysis failed: {str(e)}")
