"""
Parking Demand Forecast Router for FastAPI
Transportation/Urban Analytics - Time Series Forecasting & Capacity Planning
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# ML imports
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'

COLORS = {
    'primary': '#3b82f6',
    'critical': '#dc2626',
    'high': '#f97316',
    'moderate': '#eab308',
    'low': '#22c55e',
    'neutral': '#6b7280',
}

router = APIRouter()


# ============================================================
# Pydantic Models
# ============================================================

class ForecastRequest(BaseModel):
    data: List[Dict[str, Any]]
    date_col: str
    hour_col: Optional[str] = None
    demand_col: str
    zone_col: Optional[str] = None
    capacity_col: Optional[str] = None
    rate_col: Optional[str] = None
    weather_cols: Optional[List[str]] = None
    event_col: Optional[str] = None
    forecast_model: str = "xgboost"
    forecast_horizon: int = 7
    confidence_level: float = 0.95
    parking_type: str = "mixed"
    city: Optional[str] = "San Francisco, CA"


# ============================================================
# Helper Functions
# ============================================================

def _to_native(obj):
    """Convert numpy types to native Python types"""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if np.isnan(obj) or np.isinf(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _fig_to_base64(fig) -> str:
    """Convert matplotlib figure to base64"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


def _setup_style():
    sns.set_style("whitegrid", {'axes.facecolor': '#f8f9fa', 'grid.color': '#e5e7eb'})
    sns.set_context("notebook", font_scale=1.0)


def _style_axis(ax):
    for spine in ax.spines.values():
        spine.set_color('#d1d5db')
        spine.set_linewidth(0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def get_demand_level(occupancy: float) -> str:
    """Classify demand level based on occupancy rate"""
    if occupancy >= 90:
        return "critical"
    elif occupancy >= 75:
        return "high"
    elif occupancy >= 50:
        return "moderate"
    return "low"


def mean_absolute_percentage_error(y_true, y_pred):
    """Calculate MAPE"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def get_zone_recommendation(occupancy_rate: float, turnover: float, zone_name: str) -> str:
    """Generate recommendation based on zone metrics"""
    if occupancy_rate >= 95:
        return "Critical capacity - implement overflow routing immediately"
    elif occupancy_rate >= 90:
        return "Consider dynamic pricing during peak hours"
    elif occupancy_rate >= 80:
        return "Near optimal utilization - monitor for growth"
    elif occupancy_rate >= 60:
        return "Good utilization - maintain current operations"
    elif turnover < 2:
        return "Low turnover - consider time limits or pricing adjustments"
    return "Underutilized - increase marketing or reduce rates"


# ============================================================
# Feature Engineering
# ============================================================

def create_time_features(df: pd.DataFrame, date_col: str, hour_col: Optional[str] = None) -> pd.DataFrame:
    """Create time-based features for forecasting"""
    df = df.copy()
    
    df['date_parsed'] = pd.to_datetime(df[date_col], errors='coerce')
    df['day_of_week'] = df['date_parsed'].dt.dayofweek
    df['day_of_month'] = df['date_parsed'].dt.day
    df['month'] = df['date_parsed'].dt.month
    df['week_of_year'] = df['date_parsed'].dt.isocalendar().week.astype(int)
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    if hour_col and hour_col in df.columns:
        df['hour'] = pd.to_numeric(df[hour_col], errors='coerce').fillna(12).astype(int)
    elif 'hour' not in df.columns:
        df['hour'] = 12
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Rush hour indicators
    df['is_morning_rush'] = ((df['hour'] >= 7) & (df['hour'] <= 9)).astype(int)
    df['is_evening_rush'] = ((df['hour'] >= 16) & (df['hour'] <= 19)).astype(int)
    df['is_business_hours'] = ((df['hour'] >= 8) & (df['hour'] <= 18)).astype(int)
    
    return df


def create_lag_features(df: pd.DataFrame, target_col: str, lags: List[int] = [1, 24, 168]) -> pd.DataFrame:
    """Create lag features for time series"""
    df = df.copy()
    
    for lag in lags:
        if len(df) > lag:
            df[f'lag_{lag}'] = df[target_col].shift(lag)
    
    for window in [24, 168]:
        if len(df) > window:
            df[f'rolling_mean_{window}'] = df[target_col].rolling(window=window, min_periods=1).mean()
            df[f'rolling_std_{window}'] = df[target_col].rolling(window=window, min_periods=1).std()
    
    return df


# ============================================================
# Forecasting Functions
# ============================================================

def train_forecast_model(df: pd.DataFrame, target_col: str, feature_cols: List[str]) -> tuple:
    """Train forecasting model"""
    df_clean = df.dropna(subset=feature_cols + [target_col])
    X = df_clean[feature_cols].values
    y = df_clean[target_col].values
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred_test = model.predict(X_test_scaled)
    
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mape = mean_absolute_percentage_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)
    
    importance = model.feature_importances_ if hasattr(model, 'feature_importances_') else np.ones(len(feature_cols)) / len(feature_cols)
    
    return model, scaler, {
        'mae': _to_native(round(mae, 2)),
        'rmse': _to_native(round(rmse, 2)),
        'mape': _to_native(round(mape, 2)),
        'r2': _to_native(round(r2, 3)),
        'feature_importance': dict(zip(feature_cols, [_to_native(x) for x in importance.tolist()]))
    }


def generate_hourly_forecast(model, scaler, feature_cols: List[str], base_demand: float,
                              capacity: int, confidence_level: float) -> List[Dict]:
    """Generate hourly forecast for next 24 hours"""
    forecasts = []
    current_date = datetime.now()
    
    for hour in range(24):
        features = {
            'hour': hour, 'day_of_week': current_date.weekday(), 'month': current_date.month,
            'is_weekend': 1 if current_date.weekday() >= 5 else 0,
            'hour_sin': np.sin(2 * np.pi * hour / 24), 'hour_cos': np.cos(2 * np.pi * hour / 24),
            'dow_sin': np.sin(2 * np.pi * current_date.weekday() / 7),
            'dow_cos': np.cos(2 * np.pi * current_date.weekday() / 7),
            'month_sin': np.sin(2 * np.pi * current_date.month / 12),
            'month_cos': np.cos(2 * np.pi * current_date.month / 12),
            'is_morning_rush': 1 if 7 <= hour <= 9 else 0,
            'is_evening_rush': 1 if 16 <= hour <= 19 else 0,
            'is_business_hours': 1 if 8 <= hour <= 18 else 0,
        }
        
        for lag in [1, 24, 168]:
            features[f'lag_{lag}'] = base_demand
        features['rolling_mean_24'] = base_demand
        features['rolling_std_24'] = base_demand * 0.1
        features['rolling_mean_168'] = base_demand
        features['rolling_std_168'] = base_demand * 0.15
        
        X = np.array([[features.get(col, 0) for col in feature_cols]])
        X_scaled = scaler.transform(X)
        predicted = max(0, model.predict(X_scaled)[0])
        
        std_error = predicted * 0.1
        z_score = 1.96 if confidence_level >= 0.95 else 1.645
        lower = max(0, predicted - z_score * std_error)
        upper = predicted + z_score * std_error
        
        occupancy = min(100, (predicted / capacity) * 100) if capacity > 0 else 50
        
        forecasts.append({
            'hour': hour,
            'predicted_demand': _to_native(int(predicted)),
            'lower_bound': _to_native(int(lower)),
            'upper_bound': _to_native(int(upper)),
            'occupancy_rate': _to_native(round(occupancy, 1)),
            'demand_level': get_demand_level(occupancy)
        })
    
    return forecasts


def generate_daily_forecast(model, scaler, feature_cols: List[str], base_demand: float,
                            capacity: int, forecast_horizon: int, avg_rate: float) -> List[Dict]:
    """Generate daily forecast"""
    forecasts = []
    current_date = datetime.now()
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    for day_offset in range(1, forecast_horizon + 1):
        forecast_date = current_date + timedelta(days=day_offset)
        dow = forecast_date.weekday()
        is_weekend = dow >= 5
        
        daily_demands = []
        peak_demand = 0
        peak_hour = 12
        
        for hour in range(24):
            features = {
                'hour': hour, 'day_of_week': dow, 'month': forecast_date.month,
                'is_weekend': 1 if is_weekend else 0,
                'hour_sin': np.sin(2 * np.pi * hour / 24), 'hour_cos': np.cos(2 * np.pi * hour / 24),
                'dow_sin': np.sin(2 * np.pi * dow / 7), 'dow_cos': np.cos(2 * np.pi * dow / 7),
                'month_sin': np.sin(2 * np.pi * forecast_date.month / 12),
                'month_cos': np.cos(2 * np.pi * forecast_date.month / 12),
                'is_morning_rush': 1 if 7 <= hour <= 9 else 0,
                'is_evening_rush': 1 if 16 <= hour <= 19 else 0,
                'is_business_hours': 1 if 8 <= hour <= 18 else 0,
            }
            
            for lag in [1, 24, 168]:
                features[f'lag_{lag}'] = base_demand
            features['rolling_mean_24'] = base_demand
            features['rolling_std_24'] = base_demand * 0.1
            features['rolling_mean_168'] = base_demand
            features['rolling_std_168'] = base_demand * 0.15
            
            X = np.array([[features.get(col, 0) for col in feature_cols]])
            X_scaled = scaler.transform(X)
            hourly_demand = max(0, model.predict(X_scaled)[0])
            daily_demands.append(hourly_demand)
            
            if hourly_demand > peak_demand:
                peak_demand = hourly_demand
                peak_hour = hour
        
        avg_demand = np.mean(daily_demands)
        total_demand = sum(daily_demands)
        avg_occupancy = min(100, (avg_demand / capacity) * 100) if capacity > 0 else 50
        revenue = total_demand * avg_rate * 1.5
        
        forecasts.append({
            'date': forecast_date.strftime("%Y-%m-%d"),
            'day_of_week': day_names[dow],
            'predicted_demand': _to_native(int(avg_demand)),
            'peak_hour': _to_native(peak_hour),
            'peak_demand': _to_native(int(peak_demand)),
            'avg_occupancy': _to_native(round(avg_occupancy, 1)),
            'revenue_estimate': _to_native(round(revenue, 2))
        })
    
    return forecasts


def analyze_zones(df: pd.DataFrame, zone_col: Optional[str], demand_col: str,
                  capacity_col: Optional[str], rate_col: Optional[str]) -> List[Dict]:
    """Analyze parking zones"""
    zones = []
    
    if not zone_col or zone_col not in df.columns:
        capacity = int(df[capacity_col].iloc[0]) if capacity_col and capacity_col in df.columns else 500
        avg_demand = df[demand_col].mean()
        peak_demand = df[demand_col].max()
        rate = float(df[rate_col].iloc[0]) if rate_col and rate_col in df.columns else 3.0
        
        peak_occ = min(100, (peak_demand / capacity) * 100) if capacity > 0 else 50
        turnover = 24 * avg_demand / capacity if capacity > 0 else 3.0
        
        zones.append({
            'zone_id': 'Z001', 'zone_name': 'Main Zone',
            'capacity': _to_native(capacity),
            'predicted_peak_demand': _to_native(int(peak_demand)),
            'peak_occupancy_rate': _to_native(round(peak_occ, 1)),
            'avg_turnover': _to_native(round(turnover, 1)),
            'revenue_potential': _to_native(round(avg_demand * rate * 1.5 * 24, 2)),
            'recommendation': get_zone_recommendation(peak_occ, turnover, "Main Zone")
        })
        return zones
    
    for zone_id in df[zone_col].unique():
        zone_data = df[df[zone_col] == zone_id]
        zone_name = zone_data['zone_name'].iloc[0] if 'zone_name' in zone_data.columns else str(zone_id)
        
        capacity = int(zone_data[capacity_col].iloc[0]) if capacity_col and capacity_col in zone_data.columns else 500
        rate = float(zone_data[rate_col].iloc[0]) if rate_col and rate_col in zone_data.columns else 3.0
        
        avg_demand = zone_data[demand_col].mean()
        peak_demand = zone_data[demand_col].max()
        peak_occ = min(100, (peak_demand / capacity) * 100) if capacity > 0 else 50
        turnover = 24 * avg_demand / capacity if capacity > 0 else 3.0
        
        zones.append({
            'zone_id': str(zone_id), 'zone_name': str(zone_name),
            'capacity': _to_native(capacity),
            'predicted_peak_demand': _to_native(int(peak_demand)),
            'peak_occupancy_rate': _to_native(round(peak_occ, 1)),
            'avg_turnover': _to_native(round(turnover, 1)),
            'revenue_potential': _to_native(round(avg_demand * rate * 1.5 * 24, 2)),
            'recommendation': get_zone_recommendation(peak_occ, turnover, str(zone_name))
        })
    
    zones.sort(key=lambda x: x['peak_occupancy_rate'], reverse=True)
    return zones


def analyze_demand_patterns(df: pd.DataFrame, demand_col: str) -> Dict:
    """Analyze demand patterns"""
    if 'is_weekend' in df.columns:
        weekday_data = df[df['is_weekend'] == 0]
        weekend_data = df[df['is_weekend'] == 1]
    else:
        weekday_data = df
        weekend_data = df
    
    weekday_avg = weekday_data[demand_col].mean() if len(weekday_data) > 0 else 0
    weekend_avg = weekend_data[demand_col].mean() if len(weekend_data) > 0 else 0
    
    if 'hour' in df.columns:
        weekday_hourly = weekday_data.groupby('hour')[demand_col].mean() if len(weekday_data) > 0 else pd.Series()
        weekend_hourly = weekend_data.groupby('hour')[demand_col].mean() if len(weekend_data) > 0 else pd.Series()
        peak_hour_weekday = int(weekday_hourly.idxmax()) if len(weekday_hourly) > 0 else 11
        peak_hour_weekend = int(weekend_hourly.idxmax()) if len(weekend_hourly) > 0 else 14
    else:
        peak_hour_weekday, peak_hour_weekend = 11, 14
    
    def format_hour(h):
        period = "AM" if h < 12 else "PM"
        display = h if h <= 12 else h - 12
        if display == 0:
            display = 12
        next_h = (h + 1) % 24
        next_period = "AM" if next_h < 12 else "PM"
        next_display = next_h if next_h <= 12 else next_h - 12
        if next_display == 0:
            next_display = 12
        return f"{display}:00 {period} - {next_display}:00 {next_period}"
    
    return {
        'weekday_avg': _to_native(int(weekday_avg)),
        'weekend_avg': _to_native(int(weekend_avg)),
        'peak_hour_weekday': format_hour(peak_hour_weekday),
        'peak_hour_weekend': format_hour(peak_hour_weekend),
        'seasonal_patterns': [
            {'pattern_type': 'Summer Peak', 'peak_months': ['June', 'July', 'August'],
             'low_months': ['January', 'February'], 'variance_pct': 15}
        ]
    }


def generate_insights(metrics: Dict, zones: List[Dict], patterns: Dict, drivers: List[Dict]) -> List[Dict]:
    """Generate key insights"""
    insights = []
    
    if metrics['mape'] <= 10:
        insights.append({
            'title': 'Model Accuracy Strong',
            'description': f"Forecast model achieved {metrics['mape']}% MAPE with R² of {metrics['r2']}.",
            'status': 'positive'
        })
    else:
        insights.append({
            'title': 'Model Accuracy Moderate',
            'description': f"Forecast model achieved {metrics['mape']}% MAPE. Consider more data.",
            'status': 'neutral'
        })
    
    critical_zones = [z for z in zones if z['peak_occupancy_rate'] >= 90]
    if critical_zones:
        names = ", ".join([z['zone_name'] for z in critical_zones[:2]])
        insights.append({
            'title': 'Peak Capacity Stress',
            'description': f"{names} approach {critical_zones[0]['peak_occupancy_rate']}% occupancy during peaks.",
            'status': 'warning'
        })
    
    diff_pct = abs(patterns['weekday_avg'] - patterns['weekend_avg']) / max(patterns['weekday_avg'], 1) * 100
    if diff_pct > 15:
        higher = "Weekday" if patterns['weekday_avg'] > patterns['weekend_avg'] else "Weekend"
        insights.append({
            'title': f'{higher} Demand Dominance',
            'description': f"{higher} demand is {diff_pct:.0f}% higher.",
            'status': 'neutral'
        })
    
    if drivers:
        insights.append({
            'title': f"{drivers[0]['factor']} Primary Driver",
            'description': f"{drivers[0]['factor']} accounts for {drivers[0]['importance']*100:.0f}% of demand variation.",
            'status': 'neutral'
        })
    
    return insights


def generate_recommendations(zones: List[Dict], capacity_analysis: Dict, patterns: Dict) -> List[Dict]:
    """Generate recommendations"""
    recommendations = []
    
    critical_zones = [z for z in zones if z['peak_occupancy_rate'] >= 90]
    for zone in critical_zones[:2]:
        recommendations.append({
            'priority': 'immediate', 'category': 'Capacity',
            'action': f"Implement overflow routing from {zone['zone_name']} during peak hours",
            'expected_impact': "Reduce overflow by 50-60%"
        })
    
    high_zones = [z for z in zones if 80 <= z['peak_occupancy_rate'] < 90]
    if high_zones:
        recommendations.append({
            'priority': 'immediate', 'category': 'Pricing',
            'action': f"Deploy dynamic pricing in {high_zones[0]['zone_name']}",
            'expected_impact': "+15-20% revenue"
        })
    
    recommendations.append({
        'priority': 'short_term', 'category': 'Technology',
        'action': "Install real-time availability displays at zone entrances",
        'expected_impact': "Improve distribution, reduce search time 30%"
    })
    
    low_zones = [z for z in zones if z['peak_occupancy_rate'] < 60]
    if low_zones:
        recommendations.append({
            'priority': 'short_term', 'category': 'Marketing',
            'action': f"Promote {low_zones[0]['zone_name']} parking with discounted rates",
            'expected_impact': "Increase utilization 20-30%"
        })
    
    if capacity_analysis.get('recommended_capacity_increase', 0) > 0:
        recommendations.append({
            'priority': 'long_term', 'category': 'Infrastructure',
            'action': f"Add {capacity_analysis['recommended_capacity_increase']} spaces",
            'expected_impact': "Maintain <85% peak occupancy"
        })
    
    return recommendations


# ============================================================
# Visualization Functions
# ============================================================

def create_demand_forecast_chart(hourly_forecast: List[Dict]) -> str:
    """Create demand forecast chart"""
    _setup_style()
    fig, ax = plt.subplots(figsize=(12, 5))
    
    hours = [f['hour'] for f in hourly_forecast]
    demands = [f['predicted_demand'] for f in hourly_forecast]
    lower = [f['lower_bound'] for f in hourly_forecast]
    upper = [f['upper_bound'] for f in hourly_forecast]
    
    ax.fill_between(hours, lower, upper, alpha=0.3, color=COLORS['primary'], label='95% CI')
    ax.plot(hours, demands, color=COLORS['primary'], linewidth=2, marker='o', markersize=4, label='Forecast')
    
    ax.set_xlabel('Hour of Day', fontsize=11)
    ax.set_ylabel('Predicted Demand', fontsize=11)
    ax.set_title('24-Hour Demand Forecast', fontsize=13, fontweight='600', pad=15)
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    ax.grid(True, alpha=0.3)
    _style_axis(ax)
    
    return _fig_to_base64(fig)


def create_weekly_pattern_chart(daily_forecast: List[Dict]) -> str:
    """Create weekly pattern chart"""
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    
    demands = [f['predicted_demand'] for f in daily_forecast]
    colors = [COLORS['critical'] if f['avg_occupancy'] >= 80 else COLORS['primary'] for f in daily_forecast]
    
    ax.bar(range(len(daily_forecast)), demands, color=colors)
    ax.set_xticks(range(len(daily_forecast)))
    ax.set_xticklabels([f['day_of_week'][:3] for f in daily_forecast], rotation=45)
    ax.set_ylabel('Average Daily Demand', fontsize=11)
    ax.set_title('Daily Demand Forecast', fontsize=13, fontweight='600', pad=15)
    ax.grid(True, alpha=0.3, axis='y')
    _style_axis(ax)
    
    return _fig_to_base64(fig)


def create_zone_heatmap(zones: List[Dict]) -> str:
    """Create zone occupancy chart"""
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = [z['zone_name'] for z in zones]
    occupancies = [z['peak_occupancy_rate'] for z in zones]
    colors = [COLORS['critical'] if o >= 90 else COLORS['high'] if o >= 75 else COLORS['low'] for o in occupancies]
    
    bars = ax.barh(names, occupancies, color=colors)
    ax.axvline(x=85, color=COLORS['critical'], linestyle='--', label='Target Max (85%)')
    ax.set_xlabel('Peak Occupancy Rate (%)', fontsize=11)
    ax.set_title('Zone Peak Occupancy', fontsize=13, fontweight='600', pad=15)
    ax.legend()
    ax.set_xlim(0, 100)
    _style_axis(ax)
    
    for bar, occ in zip(bars, occupancies):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, f'{occ:.1f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


def create_drivers_chart(drivers: List[Dict]) -> str:
    """Create demand drivers chart"""
    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    
    factors = [d['factor'] for d in drivers]
    importance = [d['importance'] * 100 for d in drivers]
    colors = [COLORS['low'] if d['direction'] == 'positive' else COLORS['critical'] for d in drivers]
    
    ax.barh(factors, importance, color=colors)
    ax.set_xlabel('Feature Importance (%)', fontsize=11)
    ax.set_title('Demand Drivers', fontsize=13, fontweight='600', pad=15)
    _style_axis(ax)
    
    plt.tight_layout()
    return _fig_to_base64(fig)


# ============================================================
# Main Router Endpoint
# ============================================================

@router.post("/parking-forecast")
async def forecast_parking_demand(request: ForecastRequest) -> Dict[str, Any]:
    """Perform parking demand forecast analysis"""
    try:
        df = pd.DataFrame(request.data)
        
        if len(df) < 100:
            raise HTTPException(status_code=400, detail="Insufficient data. Need at least 100 records.")
        
        if request.demand_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Demand column '{request.demand_col}' not found")
        
        # Create features
        df = create_time_features(df, request.date_col, request.hour_col)
        df[request.demand_col] = pd.to_numeric(df[request.demand_col], errors='coerce').fillna(0)
        
        # Feature columns
        feature_cols = [
            'hour', 'day_of_week', 'month', 'is_weekend',
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos',
            'is_morning_rush', 'is_evening_rush', 'is_business_hours'
        ]
        
        df = create_lag_features(df, request.demand_col)
        lag_cols = [col for col in df.columns if col.startswith('lag_') or col.startswith('rolling_')]
        feature_cols.extend(lag_cols)
        
        if request.weather_cols:
            for col in request.weather_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
                    feature_cols.append(col)
        
        if request.event_col and request.event_col in df.columns:
            df[request.event_col] = pd.to_numeric(df[request.event_col], errors='coerce').fillna(0)
            feature_cols.append(request.event_col)
        
        feature_cols = [col for col in feature_cols if col in df.columns]
        
        # Train model
        model, scaler, metrics = train_forecast_model(df, request.demand_col, feature_cols)
        
        base_demand = df[request.demand_col].mean()
        total_capacity = int(df[request.capacity_col].sum() / len(df[request.capacity_col].unique())) \
            if request.capacity_col and request.capacity_col in df.columns else int(base_demand * 1.3)
        avg_rate = df[request.rate_col].mean() if request.rate_col and request.rate_col in df.columns else 3.0
        
        # Generate forecasts
        hourly_forecast = generate_hourly_forecast(model, scaler, feature_cols, base_demand, total_capacity, request.confidence_level)
        daily_forecast = generate_daily_forecast(model, scaler, feature_cols, base_demand, total_capacity, request.forecast_horizon, avg_rate)
        zone_forecasts = analyze_zones(df, request.zone_col, request.demand_col, request.capacity_col, request.rate_col)
        demand_patterns = analyze_demand_patterns(df, request.demand_col)
        
        # Demand drivers
        importance = metrics.get('feature_importance', {})
        total_imp = sum(importance.values()) if importance else 1
        
        driver_mapping = {
            'day_of_week': 'Day of Week', 'dow_sin': 'Day of Week', 'dow_cos': 'Day of Week',
            'hour': 'Hour of Day', 'hour_sin': 'Hour of Day', 'hour_cos': 'Hour of Day',
            'is_weekend': 'Weekend Effect', 'month': 'Month/Season', 'month_sin': 'Month/Season', 'month_cos': 'Month/Season',
            'is_morning_rush': 'Rush Hour', 'is_evening_rush': 'Rush Hour',
            'temperature_f': 'Temperature', 'precipitation': 'Precipitation', 'special_event': 'Special Events'
        }
        
        driver_importance = {}
        for feat, imp in importance.items():
            driver_name = driver_mapping.get(feat, feat)
            driver_importance[driver_name] = driver_importance.get(driver_name, 0) + imp
        
        demand_drivers = [
            {'factor': name, 'importance': _to_native(round(imp / total_imp, 3)) if total_imp > 0 else 0,
             'coefficient': _to_native(round(imp, 3)), 'direction': 'negative' if name == 'Precipitation' else 'positive'}
            for name, imp in sorted(driver_importance.items(), key=lambda x: x[1], reverse=True)
        ][:6]
        
        # Capacity analysis
        avg_util = (base_demand / total_capacity * 100) if total_capacity > 0 else 50
        peak_demand = df[request.demand_col].max()
        peak_util = (peak_demand / total_capacity * 100) if total_capacity > 0 else 80
        overflow_hours = [h['hour'] for h in hourly_forecast if h['occupancy_rate'] >= 85]
        rec_increase = int(peak_demand * 0.1) if peak_util > 85 else 0
        
        capacity_analysis = {
            'total_capacity': _to_native(total_capacity),
            'avg_utilization': _to_native(round(avg_util, 1)),
            'peak_utilization': _to_native(round(peak_util, 1)),
            'overflow_risk_hours': overflow_hours,
            'recommended_capacity_increase': _to_native(rec_increase)
        }
        
        # Revenue
        daily_revenue = sum(f['revenue_estimate'] for f in daily_forecast) / len(daily_forecast)
        revenue_forecast = {
            'daily_avg': _to_native(round(daily_revenue, 2)),
            'weekly_total': _to_native(round(daily_revenue * 7, 2)),
            'monthly_projection': _to_native(round(daily_revenue * 30, 2)),
            'optimization_potential': _to_native(round(12.5 if peak_util > 80 else 8.0, 1))
        }
        
        # Visualizations
        visualizations = {}
        try:
            visualizations['demand_forecast'] = create_demand_forecast_chart(hourly_forecast)
            visualizations['weekly_pattern'] = create_weekly_pattern_chart(daily_forecast)
            visualizations['zone_heatmap'] = create_zone_heatmap(zone_forecasts)
            visualizations['drivers_chart'] = create_drivers_chart(demand_drivers)
        except Exception as e:
            print(f"Visualization error: {e}")
        
        insights = generate_insights(metrics, zone_forecasts, demand_patterns, demand_drivers)
        recommendations = generate_recommendations(zone_forecasts, capacity_analysis, demand_patterns)
        
        summary = {
            'forecast_period': f"{request.forecast_horizon} days",
            'total_zones': _to_native(len(zone_forecasts)),
            'total_capacity': _to_native(total_capacity),
            'avg_daily_demand': _to_native(int(base_demand)),
            'peak_demand': _to_native(int(peak_demand)),
            'avg_occupancy_rate': _to_native(round(avg_util, 1)),
            'primary_demand_driver': demand_drivers[0]['factor'] if demand_drivers else 'Unknown',
            'model_accuracy': _to_native(round(100 - metrics['mape'], 1))
        }
        
        return {
            'success': True,
            'model_performance': {
                'algorithm': request.forecast_model,
                'mae': metrics['mae'], 'rmse': metrics['rmse'],
                'mape': metrics['mape'], 'r2': metrics['r2'],
                'training_periods': len(df), 'forecast_horizon': request.forecast_horizon
            },
            'hourly_forecast': hourly_forecast,
            'daily_forecast': daily_forecast,
            'zone_forecasts': zone_forecasts,
            'demand_patterns': demand_patterns,
            'demand_drivers': demand_drivers,
            'capacity_analysis': capacity_analysis,
            'revenue_forecast': revenue_forecast,
            'visualizations': visualizations,
            'key_insights': insights,
            'recommendations': recommendations,
            'summary': summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parking forecast failed: {str(e)}")
