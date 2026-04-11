"""
Statistica API - Main FastAPI Application
Complete main.py with all routers and CORS configuration
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import io
import base64
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from datetime import datetime, timedelta
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# ============================================
# SCHEDULER IMPORTS
# ============================================

import os
import asyncio
import aiohttp
import firebase_admin
from firebase_admin import credentials, firestore as fs
from typing import Optional as _Opt

_firebase_db = None

def _get_db():
    global _firebase_db
    if _firebase_db is None:
        if not firebase_admin._apps:
            cred_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
            if cred_path and os.path.exists(cred_path):
                firebase_admin.initialize_app(credentials.Certificate(cred_path))
            else:
                firebase_admin.initialize_app(credentials.ApplicationDefault())
        _firebase_db = fs.client()
    return _firebase_db

def _get_schedule_config(job_id: str, org_id: str = 'default_org') -> _Opt[dict]:
    db = _get_db()
    doc = db.collection('orgs').document(org_id).collection('schedules').document(job_id).get()
    if not doc.exists:
        return None
    data = doc.to_dict()
    if not data.get('enabled', False):
        return None
    return data.get('config', {})

def _save_csv(db, file_id: str, file_name: str, csv: str, data_type: str,
              description: str, columns: list, column_types: list,
              source_platform: str, org_id: str = 'default_org'):
    from datetime import timezone
    db.collection('shared-files').document(file_id).set({
        'fileName': file_name, 'fileSize': len(csv), 'fileType': '.csv',
        'orgId': org_id, 'uploadedBy': 'system_scheduler',
        'uploadedByEmail': 'scheduler@statistica.ai',
        'description': description,
        'createdAt': fs.SERVER_TIMESTAMP,
        'downloadURL': 'data:text/csv;charset=utf-8,' + csv,
        'autoMapped': True, 'dataType': data_type,
        'columns': columns, 'columnTypes': column_types,
        'sourcePlatform': source_platform,
        'syncedAt': datetime.now(timezone.utc).isoformat(),
        'scheduledSync': True,
    })

def _update_status(job_id: str, status: str, error: str = None, org_id: str = 'default_org'):
    db = _get_db()
    data = {'lastSyncAt': fs.SERVER_TIMESTAMP, 'lastSyncStatus': status}
    if error:
        data['lastSyncError'] = error
    db.collection('orgs').document(org_id).collection('schedules').document(job_id).set(data, merge=True)

# ML imports
from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ── Text Analysis router ──────────────────────────────────────
from api.text_analysis import router as text_analysis_router

# ── SCM router ───────────────────────────────────────────────
from api.scm_router import router as scm_router

# ── Investment Intelligence routers ──────────────────────────────────────────
from routers.statistics import router as inv_statistics_router
from routers.factor import router as inv_factor_router
from routers.signal import router as inv_signal_router
from routers.valuation import router as inv_valuation_router
from routers.portfolio import router as inv_portfolio_router
from routers.backtest import router as inv_backtest_router
from routers.risk import router as inv_risk_router
from routers.derivatives import router as inv_derivatives_router
from routers.screening import router as inv_screening_router

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# ============================================
# APP INITIALIZATION
# ============================================

app = FastAPI(
    title="Statistica API",
    description="Statistical Analysis API for Business Intelligence",
    version="2.0.0"
)

# ============================================
# CORS CONFIGURATION
# ============================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=86400,
)

# ── Router registration ───────────────────────────────────────

# ── Investment Intelligence (/router/investment/...) ─────────
app.include_router(inv_statistics_router,  prefix="/router/investment/statistics",  tags=["Investment - Statistics"])
app.include_router(inv_factor_router,      prefix="/router/investment/factor",      tags=["Investment - Factor"])
app.include_router(inv_signal_router,      prefix="/router/investment/signal",      tags=["Investment - Signal"])
app.include_router(inv_valuation_router,   prefix="/router/investment/valuation",   tags=["Investment - Valuation"])
app.include_router(inv_portfolio_router,   prefix="/router/investment/portfolio",   tags=["Investment - Portfolio"])
app.include_router(inv_backtest_router,    prefix="/router/investment/backtest",     tags=["Investment - Backtest"])
app.include_router(inv_risk_router,        prefix="/router/investment/risk",         tags=["Investment - Risk"])
app.include_router(inv_derivatives_router, prefix="/router/investment/derivatives",  tags=["Investment - Derivatives"])
app.include_router(inv_screening_router,   prefix="/router/investment/screening",    tags=["Investment - Screening"])

# ============================================
# SHARED HELPER FUNCTIONS
# ============================================

def _to_native(obj):
    """Convert numpy types to native Python types"""
    if isinstance(obj, np.integer): return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if np.isnan(obj) or np.isinf(obj) else float(obj)
    if isinstance(obj, np.ndarray): return obj.tolist()
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
    sns.set_style("darkgrid", {'axes.facecolor': '#f8f9fa', 'grid.color': '#dee2e6'})
    sns.set_context("notebook", font_scale=1.0)


def _style_axis(ax):
    for spine in ax.spines.values():
        spine.set_color('#cccccc')
        spine.set_linewidth(0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)






# Color Palettes
COLORS_DEMOGRAPHIC = {'birth': '#4299e1', 'death': '#e53e3e', 'natural': '#48bb78', 'primary': '#4a5568', 'secondary': '#718096', 'warning': '#ed8936'}
COLORS_MIGRATION = {'in': '#48bb78', 'out': '#e53e3e', 'net_pos': '#38a169', 'net_neg': '#c53030', 'primary': '#4a5568', 'secondary': '#718096'}
COLORS_EMPLOYMENT = {'unemployment': '#e53e3e', 'employment': '#48bb78', 'participation': '#4299e1', 'primary': '#4a5568', 'secondary': '#718096', 'warning': '#ed8936'}
COLORS_INCOME = {'income': '#48bb78', 'median': '#4299e1', 'poverty': '#e53e3e', 'primary': '#4a5568', 'secondary': '#718096', 'warning': '#ed8936'}
COLORS_INFLATION = {'inflation': '#e53e3e', 'core': '#4299e1', 'target': '#48bb78', 'primary': '#4a5568', 'secondary': '#718096', 'warning': '#ed8936'}
COLORS_INDUSTRY = {'primary': '#48bb78', 'secondary': '#4299e1', 'tertiary': '#9f7aea', 'quaternary': '#ed8936', 'growth': '#38a169', 'decline': '#e53e3e', 'neutral': '#718096'}
COLORS_TRAFFIC = {'primary': '#3b82f6', 'critical': '#dc2626', 'high': '#f97316', 'medium': '#eab308', 'low': '#22c55e', 'neutral': '#6b7280'}
COLORS_BUDGET = {'primary': '#4a5568', 'secondary': '#718096', 'accent': '#2d3748', 'positive': '#48bb78', 'warning': '#ed8936', 'danger': '#e53e3e', 'excellent': '#38a169', 'good': '#4299e1', 'fair': '#ecc94b', 'poor': '#e53e3e'}
PALETTE = ['#4a5568', '#718096', '#a0aec0', '#4299e1', '#48bb78', '#ed8936', '#e53e3e', '#9f7aea']


# ============================================
# BIRTH & MORTALITY ANALYSIS
# ============================================

class BirthMortalityRequest(BaseModel):
    data: List[Dict[str, Any]]
    period_col: str
    population_col: Optional[str] = None
    births_col: Optional[str] = None
    deaths_col: Optional[str] = None
    birth_rate_col: Optional[str] = None
    death_rate_col: Optional[str] = None
    tfr_col: Optional[str] = None
    imr_col: Optional[str] = None
    region_col: Optional[str] = None
    age_group_col: Optional[str] = None
    analysis_focus: str = "trend"
    analysis_period: str = "Analysis Period"


@app.post("/api/analysis/birth-mortality")
async def run_birth_mortality_analysis(request: BirthMortalityRequest) -> Dict[str, Any]:
    try:
        df = pd.DataFrame(request.data)
        if request.period_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Period column '{request.period_col}' not found")
        
        # Calculate rates
        if request.birth_rate_col and request.birth_rate_col in df.columns:
            df['_birth_rate'] = pd.to_numeric(df[request.birth_rate_col], errors='coerce')
        elif request.births_col and request.population_col:
            births = pd.to_numeric(df[request.births_col], errors='coerce')
            pop = pd.to_numeric(df[request.population_col], errors='coerce')
            df['_birth_rate'] = (births / pop) * 1000
        else:
            df['_birth_rate'] = np.nan
        
        if request.death_rate_col and request.death_rate_col in df.columns:
            df['_death_rate'] = pd.to_numeric(df[request.death_rate_col], errors='coerce')
        elif request.deaths_col and request.population_col:
            deaths = pd.to_numeric(df[request.deaths_col], errors='coerce')
            pop = pd.to_numeric(df[request.population_col], errors='coerce')
            df['_death_rate'] = (deaths / pop) * 1000
        else:
            df['_death_rate'] = np.nan
        
        df['_natural_increase'] = df['_birth_rate'] - df['_death_rate']
        df['_tfr'] = pd.to_numeric(df[request.tfr_col], errors='coerce') if request.tfr_col and request.tfr_col in df.columns else df['_birth_rate'] / 8
        df['_imr'] = pd.to_numeric(df[request.imr_col], errors='coerce') if request.imr_col and request.imr_col in df.columns else 3.0
        
        # Overall metrics
        total_pop = df[request.population_col].sum() if request.population_col else 0
        total_births = df[request.births_col].sum() if request.births_col else 0
        total_deaths = df[request.deaths_col].sum() if request.deaths_col else 0
        birth_rate, death_rate = df['_birth_rate'].mean(), df['_death_rate'].mean()
        natural_increase, tfr, imr = df['_natural_increase'].mean(), df['_tfr'].mean(), df['_imr'].mean()
        
        overall = {'birth_rate': _to_native(birth_rate), 'death_rate': _to_native(death_rate), 'natural_increase_rate': _to_native(natural_increase), 'total_fertility_rate': _to_native(tfr), 'infant_mortality_rate': _to_native(imr), 'total_births': _to_native(total_births), 'total_deaths': _to_native(total_deaths), 'total_population': _to_native(total_pop)}
        
        temporal = df.groupby(request.period_col).agg({'_birth_rate': 'mean', '_death_rate': 'mean', '_natural_increase': 'mean', '_tfr': 'mean'}).reset_index().sort_values(request.period_col)
        temporal_result = {'periods': [str(p) for p in temporal[request.period_col].tolist()], 'birth_rates': [_to_native(x) for x in temporal['_birth_rate'].tolist()], 'death_rates': [_to_native(x) for x in temporal['_death_rate'].tolist()], 'natural_increase_rates': [_to_native(x) for x in temporal['_natural_increase'].tolist()], 'tfr': [_to_native(x) for x in temporal['_tfr'].tolist()]}
        
        regional = []
        if request.region_col and request.region_col in df.columns:
            for region in df[request.region_col].unique():
                rdf = df[df[request.region_col] == region]
                regional.append({'region': str(region), 'birth_rate': _to_native(rdf['_birth_rate'].mean()), 'death_rate': _to_native(rdf['_death_rate'].mean()), 'natural_increase': _to_native(rdf['_natural_increase'].mean()), 'tfr': _to_native(rdf['_tfr'].mean())})
            regional = sorted(regional, key=lambda x: x['birth_rate'] or 0, reverse=True)
        
        _setup_style()
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        ax = axes[0, 0]
        ax.plot(temporal_result['periods'], temporal_result['birth_rates'], color=COLORS_DEMOGRAPHIC['birth'], linewidth=2.5, marker='o', markersize=6, label='Birth Rate')
        ax.plot(temporal_result['periods'], temporal_result['death_rates'], color=COLORS_DEMOGRAPHIC['death'], linewidth=2.5, marker='s', markersize=6, label='Death Rate')
        ax.fill_between(temporal_result['periods'], temporal_result['birth_rates'], temporal_result['death_rates'], alpha=0.15, color=COLORS_DEMOGRAPHIC['natural'])
        ax.set_ylabel('Rate (per 1,000)', fontsize=11)
        ax.set_title('Birth and Death Rate Trends', fontsize=13, fontweight='600', pad=15)
        ax.legend(fontsize=9)
        _style_axis(ax)
        
        ax = axes[0, 1]
        colors_ni = [COLORS_DEMOGRAPHIC['natural'] if x >= 0 else COLORS_DEMOGRAPHIC['death'] for x in temporal_result['natural_increase_rates']]
        ax.bar(temporal_result['periods'], temporal_result['natural_increase_rates'], color=colors_ni, alpha=0.8, edgecolor='white')
        ax.axhline(0, color=COLORS_DEMOGRAPHIC['primary'], linestyle='--', linewidth=1)
        ax.set_ylabel('Natural Increase (per 1,000)', fontsize=11)
        ax.set_title('Natural Population Increase', fontsize=13, fontweight='600', pad=15)
        _style_axis(ax)
        
        ax = axes[1, 0]
        ax.plot(temporal_result['periods'], temporal_result['tfr'], color=COLORS_DEMOGRAPHIC['birth'], linewidth=2.5, marker='D', markersize=6)
        ax.axhline(2.1, color=COLORS_DEMOGRAPHIC['warning'], linestyle='--', linewidth=2, label='Replacement Level (2.1)')
        ax.fill_between(temporal_result['periods'], temporal_result['tfr'], 2.1, alpha=0.2, color=COLORS_DEMOGRAPHIC['birth'])
        ax.set_ylabel('Total Fertility Rate', fontsize=11)
        ax.set_title('Total Fertility Rate Trend', fontsize=13, fontweight='600', pad=15)
        ax.legend(fontsize=9)
        _style_axis(ax)
        
        ax = axes[1, 1]
        if regional:
            regions_names = [r['region'] for r in regional[:10]]
            birth_vals = [r['birth_rate'] or 0 for r in regional[:10]]
            death_vals = [r['death_rate'] or 0 for r in regional[:10]]
            x = np.arange(len(regions_names))
            ax.barh(x - 0.2, birth_vals, 0.4, label='Birth Rate', color=COLORS_DEMOGRAPHIC['birth'])
            ax.barh(x + 0.2, death_vals, 0.4, label='Death Rate', color=COLORS_DEMOGRAPHIC['death'])
            ax.set_yticks(x)
            ax.set_yticklabels(regions_names)
            ax.set_xlabel('Rate (per 1,000)', fontsize=11)
            ax.set_title('Regional Comparison', fontsize=13, fontweight='600', pad=15)
            ax.legend(fontsize=9)
        _style_axis(ax)
        
        plt.tight_layout()
        combined_chart = _fig_to_base64(fig)
        
        tfr_val = overall['total_fertility_rate'] or 0
        fertility_level = "Very Low" if tfr_val < 1.3 else "Low" if tfr_val < 1.8 else "Below Replacement" if tfr_val < 2.1 else "Above Replacement"
        growth = "Natural Decrease" if (overall['natural_increase_rate'] or 0) < 0 else "Natural Increase"
        
        summary = {'analysis_period': request.analysis_period, 'birth_rate': overall['birth_rate'], 'death_rate': overall['death_rate'], 'natural_increase': overall['natural_increase_rate'], 'tfr': tfr_val, 'imr': overall['infant_mortality_rate'], 'highest_region': regional[0]['region'] if regional else 'N/A', 'lowest_region': regional[-1]['region'] if regional else 'N/A', 'fertility_level': fertility_level, 'population_growth_type': growth}
        
        insights = []
        if tfr_val < 1.3:
            insights.append({'title': 'Very Low Fertility Crisis', 'description': f'TFR of {tfr_val:.2f} indicates severe demographic challenges ahead.', 'status': 'critical'})
        elif tfr_val < 2.1:
            insights.append({'title': 'Below Replacement Fertility', 'description': f'TFR of {tfr_val:.2f} is below replacement level of 2.1.', 'status': 'warning'})
        
        if (overall['natural_increase_rate'] or 0) < 0:
            insights.append({'title': 'Population Decline', 'description': 'Deaths exceed births, indicating natural population decline.', 'status': 'warning'})
        
        return {'success': True, 'overall_metrics': overall, 'temporal_analysis': temporal_result, 'regional_analysis': regional, 'visualizations': {'combined_chart': combined_chart}, 'key_insights': insights, 'summary': summary}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Birth-mortality analysis failed: {str(e)}")


# ============================================
# MIGRATION ANALYSIS
# ============================================

class MigrationRequest(BaseModel):
    data: List[Dict[str, Any]]
    period_col: str
    region_col: str
    in_migration_col: Optional[str] = None
    out_migration_col: Optional[str] = None
    net_migration_col: Optional[str] = None
    population_col: Optional[str] = None
    age_group_col: Optional[str] = None
    origin_col: Optional[str] = None
    destination_col: Optional[str] = None
    migration_type_col: Optional[str] = None
    analysis_focus: str = "trend"
    analysis_period: str = "Analysis Period"


@app.post("/api/analysis/migration")
async def run_migration_analysis(request: MigrationRequest) -> Dict[str, Any]:
    try:
        df = pd.DataFrame(request.data)
        if request.period_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Period column '{request.period_col}' not found")
        if request.region_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Region column '{request.region_col}' not found")
        
        df['_in'] = pd.to_numeric(df[request.in_migration_col], errors='coerce') if request.in_migration_col and request.in_migration_col in df.columns else 0
        df['_out'] = pd.to_numeric(df[request.out_migration_col], errors='coerce') if request.out_migration_col and request.out_migration_col in df.columns else 0
        
        if request.net_migration_col and request.net_migration_col in df.columns:
            df['_net'] = pd.to_numeric(df[request.net_migration_col], errors='coerce')
        else:
            df['_net'] = df['_in'] - df['_out']
        
        df['_pop'] = pd.to_numeric(df[request.population_col], errors='coerce') if request.population_col and request.population_col in df.columns else 100000
        df['_net_rate'] = (df['_net'] / df['_pop']) * 1000
        
        total_in, total_out, total_net = df['_in'].sum(), df['_out'].sum(), df['_net'].sum()
        overall = {'total_in_migration': _to_native(total_in), 'total_out_migration': _to_native(total_out), 'net_migration': _to_native(total_net), 'migration_rate': _to_native(df['_net_rate'].mean()), 'gross_migration': _to_native(total_in + total_out)}
        
        temporal = df.groupby(request.period_col).agg({'_in': 'sum', '_out': 'sum', '_net': 'sum', '_net_rate': 'mean'}).reset_index().sort_values(request.period_col)
        temporal_result = {'periods': [str(p) for p in temporal[request.period_col].tolist()], 'in_migration': [_to_native(x) for x in temporal['_in'].tolist()], 'out_migration': [_to_native(x) for x in temporal['_out'].tolist()], 'net_migration': [_to_native(x) for x in temporal['_net'].tolist()], 'net_rate': [_to_native(x) for x in temporal['_net_rate'].tolist()]}
        
        regional = []
        for region in df[request.region_col].unique():
            rdf = df[df[request.region_col] == region]
            regional.append({'region': str(region), 'in_migration': _to_native(rdf['_in'].sum()), 'out_migration': _to_native(rdf['_out'].sum()), 'net_migration': _to_native(rdf['_net'].sum()), 'net_rate': _to_native(rdf['_net_rate'].mean())})
        regional = sorted(regional, key=lambda x: x['net_migration'] or 0, reverse=True)
        
        _setup_style()
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        ax = axes[0, 0]
        ax.plot(temporal_result['periods'], temporal_result['in_migration'], color=COLORS_MIGRATION['in'], linewidth=2.5, marker='o', markersize=6, label='In-Migration')
        ax.plot(temporal_result['periods'], temporal_result['out_migration'], color=COLORS_MIGRATION['out'], linewidth=2.5, marker='s', markersize=6, label='Out-Migration')
        ax.set_ylabel('Number of Migrants', fontsize=11)
        ax.set_title('Migration Flow Trends', fontsize=13, fontweight='600', pad=15)
        ax.legend(fontsize=9)
        _style_axis(ax)
        
        ax = axes[0, 1]
        colors_net = [COLORS_MIGRATION['net_pos'] if x >= 0 else COLORS_MIGRATION['net_neg'] for x in temporal_result['net_migration']]
        ax.bar(temporal_result['periods'], temporal_result['net_migration'], color=colors_net, alpha=0.8, edgecolor='white')
        ax.axhline(0, color=COLORS_MIGRATION['primary'], linestyle='--', linewidth=1)
        ax.set_ylabel('Net Migration', fontsize=11)
        ax.set_title('Net Migration by Period', fontsize=13, fontweight='600', pad=15)
        _style_axis(ax)
        
        ax = axes[1, 0]
        ax.plot(temporal_result['periods'], temporal_result['net_rate'], color=COLORS_MIGRATION['primary'], linewidth=2.5, marker='D', markersize=6)
        ax.fill_between(temporal_result['periods'], temporal_result['net_rate'], 0, alpha=0.2, color=COLORS_MIGRATION['in'])
        ax.axhline(0, color=COLORS_MIGRATION['secondary'], linestyle='--', linewidth=1)
        ax.set_ylabel('Net Migration Rate (per 1,000)', fontsize=11)
        ax.set_title('Net Migration Rate Trend', fontsize=13, fontweight='600', pad=15)
        _style_axis(ax)
        
        ax = axes[1, 1]
        regions_names = [r['region'] for r in regional[:10]]
        net_vals = [r['net_migration'] or 0 for r in regional[:10]]
        colors_regional = [COLORS_MIGRATION['net_pos'] if x >= 0 else COLORS_MIGRATION['net_neg'] for x in net_vals]
        ax.barh(regions_names, net_vals, color=colors_regional, edgecolor='white', height=0.6)
        ax.axvline(0, color=COLORS_MIGRATION['primary'], linestyle='--', linewidth=1)
        ax.set_xlabel('Net Migration', fontsize=11)
        ax.set_title('Regional Net Migration', fontsize=13, fontweight='600', pad=15)
        _style_axis(ax)
        
        plt.tight_layout()
        combined_chart = _fig_to_base64(fig)
        
        net = overall['net_migration'] or 0
        flow_type = "Net In-Migration" if net > 0 else "Net Out-Migration" if net < 0 else "Balanced"
        
        summary = {'analysis_period': request.analysis_period, 'total_in': overall['total_in_migration'], 'total_out': overall['total_out_migration'], 'net_migration': net, 'migration_rate': overall['migration_rate'], 'top_gaining': regional[0]['region'] if regional else 'N/A', 'top_losing': regional[-1]['region'] if regional else 'N/A', 'flow_type': flow_type}
        
        insights = []
        if net > 0:
            insights.append({'title': 'Net Population Gain', 'description': f'Region experienced net gain of {net:,.0f} through migration.', 'status': 'positive'})
        elif net < 0:
            insights.append({'title': 'Net Population Loss', 'description': f'Region experienced net loss of {abs(net):,.0f} through migration.', 'status': 'warning'})
        
        return {'success': True, 'overall_metrics': overall, 'temporal_analysis': temporal_result, 'regional_analysis': regional, 'visualizations': {'combined_chart': combined_chart}, 'key_insights': insights, 'summary': summary}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Migration analysis failed: {str(e)}")


# ============================================
# UNEMPLOYMENT ANALYSIS
# ============================================

class UnemploymentRequest(BaseModel):
    data: List[Dict[str, Any]]
    period_col: str
    unemployment_rate_col: Optional[str] = None
    labor_force_col: Optional[str] = None
    unemployed_col: Optional[str] = None
    employed_col: Optional[str] = None
    region_col: Optional[str] = None
    age_group_col: Optional[str] = None
    gender_col: Optional[str] = None
    education_col: Optional[str] = None
    duration_col: Optional[str] = None
    analysis_focus: str = "trend"
    analysis_period: str = "Analysis Period"


@app.post("/api/analysis/unemployment")
async def run_unemployment_analysis(request: UnemploymentRequest) -> Dict[str, Any]:
    try:
        df = pd.DataFrame(request.data)
        if request.period_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Period column '{request.period_col}' not found")
        
        if request.unemployment_rate_col and request.unemployment_rate_col in df.columns:
            df['_unemp_rate'] = pd.to_numeric(df[request.unemployment_rate_col], errors='coerce')
        elif request.unemployed_col and request.labor_force_col:
            unemployed = pd.to_numeric(df[request.unemployed_col], errors='coerce')
            labor_force = pd.to_numeric(df[request.labor_force_col], errors='coerce')
            df['_unemp_rate'] = (unemployed / labor_force.replace(0, np.nan)) * 100
        else:
            df['_unemp_rate'] = np.nan
        
        df['_labor_force'] = pd.to_numeric(df[request.labor_force_col], errors='coerce') if request.labor_force_col else 0
        df['_unemployed'] = pd.to_numeric(df[request.unemployed_col], errors='coerce') if request.unemployed_col else df['_labor_force'] * df['_unemp_rate'] / 100
        df['_employed'] = pd.to_numeric(df[request.employed_col], errors='coerce') if request.employed_col else df['_labor_force'] - df['_unemployed']
        
        total_lf = df['_labor_force'].sum()
        total_unemployed = df['_unemployed'].sum()
        total_employed = df['_employed'].sum()
        unemp_rate = (total_unemployed / total_lf * 100) if total_lf > 0 else df['_unemp_rate'].mean()
        emp_rate = (total_employed / total_lf * 100) if total_lf > 0 else 100 - unemp_rate
        
        overall = {'unemployment_rate': _to_native(unemp_rate), 'labor_force': _to_native(total_lf), 'employed': _to_native(total_employed), 'unemployed': _to_native(total_unemployed), 'employment_rate': _to_native(emp_rate), 'yoy_change': None}
        
        temporal = df.groupby(request.period_col).agg({'_unemp_rate': 'mean', '_labor_force': 'sum', '_employed': 'sum', '_unemployed': 'sum'}).reset_index().sort_values(request.period_col)
        temporal_result = {'periods': [str(p) for p in temporal[request.period_col].tolist()], 'unemployment_rates': [_to_native(x) for x in temporal['_unemp_rate'].tolist()], 'labor_force': [_to_native(x) for x in temporal['_labor_force'].tolist()]}
        
        regional = []
        if request.region_col and request.region_col in df.columns:
            for region in df[request.region_col].unique():
                rdf = df[df[request.region_col] == region]
                regional.append({'region': str(region), 'unemployment_rate': _to_native(rdf['_unemp_rate'].mean()), 'labor_force': _to_native(rdf['_labor_force'].sum()), 'employed': _to_native(rdf['_employed'].sum()), 'unemployed': _to_native(rdf['_unemployed'].sum())})
            regional = sorted(regional, key=lambda x: x['unemployment_rate'] or 0, reverse=True)
        
        _setup_style()
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(temporal_result['periods'], temporal_result['unemployment_rates'], color=COLORS_EMPLOYMENT['unemployment'], linewidth=2.5, marker='o', markersize=6)
        ax.fill_between(temporal_result['periods'], temporal_result['unemployment_rates'], alpha=0.2, color=COLORS_EMPLOYMENT['unemployment'])
        ax.axhline(5, color=COLORS_EMPLOYMENT['secondary'], linestyle='--', linewidth=1, label='Natural Rate (5%)')
        ax.set_ylabel('Unemployment Rate (%)', fontsize=11)
        ax.set_title('Unemployment Rate Trend', fontsize=13, fontweight='600', pad=15)
        ax.legend(fontsize=9)
        _style_axis(ax)
        plt.tight_layout()
        trend_chart = _fig_to_base64(fig)
        
        rate = overall['unemployment_rate'] or 0
        status = "Tight labor market" if rate < 4 else "Full employment" if rate < 5 else "Moderate slack" if rate < 7 else "Significant slack"
        
        summary = {'analysis_period': request.analysis_period, 'unemployment_rate': rate, 'labor_force': overall['labor_force'], 'unemployed': overall['unemployed'], 'highest_region': regional[0]['region'] if regional else 'N/A', 'lowest_region': regional[-1]['region'] if regional else 'N/A', 'labor_market_status': status}
        
        insights = []
        if rate < 4:
            insights.append({'title': 'Low Unemployment', 'description': f'Unemployment rate of {rate:.1f}% indicates a tight labor market.', 'status': 'positive'})
        elif rate > 6:
            insights.append({'title': 'High Unemployment', 'description': f'Unemployment rate of {rate:.1f}% indicates labor market slack.', 'status': 'warning'})
        
        return {'success': True, 'overall_metrics': overall, 'temporal_analysis': temporal_result, 'regional_analysis': regional, 'visualizations': {'trend_chart': trend_chart}, 'key_insights': insights, 'summary': summary}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unemployment analysis failed: {str(e)}")

# ============================================
# INCOME ANALYSIS
# ============================================

class IncomeRequest(BaseModel):
    data: List[Dict[str, Any]]
    period_col: Optional[str] = None
    region_col: str
    gross_income_col: Optional[str] = None
    net_income_col: Optional[str] = None
    population_col: Optional[str] = None
    households_col: Optional[str] = None
    quintile_col: Optional[str] = None
    income_type_col: Optional[str] = None
    cost_of_living_col: Optional[str] = None
    analysis_focus: str = "regional"
    analysis_period: str = "Analysis Period"


def calculate_gini(incomes: np.ndarray, weights: np.ndarray = None) -> float:
    if len(incomes) == 0:
        return 0.0
    incomes = np.array(incomes)
    weights = np.ones(len(incomes)) if weights is None else np.array(weights)
    sorted_idx = np.argsort(incomes)
    incomes, weights = incomes[sorted_idx], weights[sorted_idx]
    cum_weights, cum_income = np.cumsum(weights), np.cumsum(incomes * weights)
    total_weight, total_income = cum_weights[-1], cum_income[-1]
    if total_income == 0 or total_weight == 0:
        return 0.0
    return max(0, min(1, 1 - 2 * np.sum(cum_income * weights) / (total_weight * total_income)))


@app.post("/api/analysis/income")
async def run_income_analysis(request: IncomeRequest) -> Dict[str, Any]:
    try:
        df = pd.DataFrame(request.data)
        if request.region_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Region column '{request.region_col}' not found")
        
        income_col = request.gross_income_col or request.net_income_col
        if not income_col or income_col not in df.columns:
            raise HTTPException(status_code=400, detail="Income column not found")
        
        df[income_col] = pd.to_numeric(df[income_col], errors='coerce')
        pop_col = request.population_col or request.households_col
        incomes = df[income_col].dropna()
        weights = df[pop_col].dropna() if pop_col and pop_col in df.columns else None
        
        mean_income, median_income = incomes.mean(), incomes.median()
        gini = calculate_gini(incomes.values, weights.values if weights is not None else None)
        poverty_rate = (incomes < median_income * 0.5).mean() * 100
        
        overall = {'total_income': _to_native(incomes.sum()), 'mean_income': _to_native(mean_income), 'median_income': _to_native(median_income), 'gini_coefficient': _to_native(gini), 'poverty_rate': _to_native(poverty_rate)}
        
        regional = [{'region': str(region), 'mean_income': _to_native(df[df[request.region_col] == region][income_col].mean()), 'median_income': _to_native(df[df[request.region_col] == region][income_col].median()), 'gini': _to_native(calculate_gini(df[df[request.region_col] == region][income_col].dropna().values))} for region in df[request.region_col].unique()]
        regional = sorted(regional, key=lambda x: x['mean_income'] or 0, reverse=True)
        
        _setup_style()
        fig, ax = plt.subplots(figsize=(12, 6))
        regions_names = [r['region'] for r in regional[:12]]
        mean_vals = [r['mean_income'] / 10000 if r['mean_income'] else 0 for r in regional[:12]]
        median_vals = [r['median_income'] / 10000 if r['median_income'] else 0 for r in regional[:12]]
        x = np.arange(len(regions_names))
        ax.bar(x - 0.2, mean_vals, 0.4, label='Mean Income', color=COLORS_INCOME['income'])
        ax.bar(x + 0.2, median_vals, 0.4, label='Median Income', color=COLORS_INCOME['median'])
        ax.set_xticks(x)
        ax.set_xticklabels(regions_names, rotation=45, ha='right')
        ax.set_ylabel('Income (만원)', fontsize=11)
        ax.set_title('Regional Income Comparison', fontsize=13, fontweight='600', pad=15)
        ax.legend(fontsize=9)
        _style_axis(ax)
        plt.tight_layout()
        regional_chart = _fig_to_base64(fig)
        
        gini_val = overall['gini_coefficient'] or 0
        ineq_level = "Low" if gini_val < 0.30 else "Moderate" if gini_val < 0.35 else "High" if gini_val < 0.40 else "Very High"
        
        summary = {'analysis_period': request.analysis_period, 'mean_income': overall['mean_income'], 'median_income': overall['median_income'], 'gini': gini_val, 'poverty_rate': overall['poverty_rate'], 'highest_region': regional[0]['region'] if regional else 'N/A', 'lowest_region': regional[-1]['region'] if regional else 'N/A', 'inequality_level': ineq_level}
        
        insights = []
        if gini_val < 0.30:
            insights.append({'title': 'Low Inequality', 'description': f'Gini of {gini_val:.3f} indicates relatively equal income distribution.', 'status': 'positive'})
        elif gini_val > 0.40:
            insights.append({'title': 'High Inequality', 'description': f'Gini of {gini_val:.3f} suggests significant income disparity.', 'status': 'warning'})
        
        return {'success': True, 'overall_metrics': overall, 'regional_analysis': regional, 'visualizations': {'regional_map': regional_chart}, 'key_insights': insights, 'summary': summary}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Income analysis failed: {str(e)}")


# ============================================
# INFLATION ANALYSIS
# ============================================

class InflationRequest(BaseModel):
    data: List[Dict[str, Any]]
    period_col: str
    cpi_col: str
    category_col: Optional[str] = None
    weight_col: Optional[str] = None
    region_col: Optional[str] = None
    is_core_col: Optional[str] = None
    base_year: int = 2020
    inflation_target: float = 2.0
    analysis_focus: str = "trend"
    analysis_period: str = "Analysis Period"


@app.post("/api/analysis/inflation")
async def run_inflation_analysis(request: InflationRequest) -> Dict[str, Any]:
    try:
        df = pd.DataFrame(request.data)
        if request.period_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Period column '{request.period_col}' not found")
        if request.cpi_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"CPI column '{request.cpi_col}' not found")
        
        df[request.cpi_col] = pd.to_numeric(df[request.cpi_col], errors='coerce')
        agg = df.groupby(request.period_col)[request.cpi_col].mean().reset_index().sort_values(request.period_col)
        agg.columns = [request.period_col, '_cpi']
        agg['_mom'] = agg['_cpi'].pct_change() * 100
        agg['_yoy'] = agg['_cpi'].pct_change(periods=12) * 100
        
        current, previous = agg['_cpi'].iloc[-1], agg['_cpi'].iloc[-2] if len(agg) > 1 else agg['_cpi'].iloc[-1]
        yoy_cpi = agg['_cpi'].iloc[-13] if len(agg) > 12 else agg['_cpi'].iloc[0]
        mom = ((current - previous) / previous * 100) if previous > 0 else 0
        yoy = ((current - yoy_cpi) / yoy_cpi * 100) if yoy_cpi > 0 else 0
        
        overall = {'headline_inflation': _to_native(yoy), 'core_inflation': _to_native(yoy * 0.85), 'current_cpi': _to_native(current), 'mom_change': _to_native(mom), 'yoy_change': _to_native(yoy)}
        temporal_result = {'periods': agg[request.period_col].astype(str).tolist(), 'cpi_values': [_to_native(x) for x in agg['_cpi'].tolist()], 'yoy_rates': [_to_native(x) if not pd.isna(x) else 0 for x in agg['_yoy'].tolist()], 'mom_rates': [_to_native(x) if not pd.isna(x) else 0 for x in agg['_mom'].tolist()]}
        
        categories = []
        if request.category_col and request.category_col in df.columns:
            for cat in df[request.category_col].unique():
                cdf = df[df[request.category_col] == cat].sort_values(request.period_col)
                cat_yoy = 0
                if len(cdf) >= 2:
                    current_cpi, yoy_cpi = cdf[request.cpi_col].iloc[-1], cdf[request.cpi_col].iloc[-13] if len(cdf) > 12 else cdf[request.cpi_col].iloc[0]
                    cat_yoy = ((current_cpi - yoy_cpi) / yoy_cpi * 100) if yoy_cpi > 0 else 0
                categories.append({'category': str(cat), 'yoy_rate': _to_native(cat_yoy)})
            categories = sorted(categories, key=lambda x: x['yoy_rate'] or 0, reverse=True)
        
        _setup_style()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        ax1.plot(temporal_result['periods'], temporal_result['yoy_rates'], color=COLORS_INFLATION['inflation'], linewidth=2.5, marker='o', markersize=5, label='YoY Inflation')
        ax1.axhline(request.inflation_target, color=COLORS_INFLATION['target'], linestyle='--', linewidth=2, label=f'Target ({request.inflation_target}%)')
        ax1.fill_between(temporal_result['periods'], temporal_result['yoy_rates'], request.inflation_target, alpha=0.2, color=COLORS_INFLATION['inflation'])
        ax1.set_ylabel('Inflation Rate (%)', fontsize=11)
        ax1.set_title('Inflation Rate Trend (YoY)', fontsize=13, fontweight='600', pad=15)
        ax1.legend(fontsize=9)
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
        _style_axis(ax1)
        
        ax2.plot(temporal_result['periods'], temporal_result['cpi_values'], color=COLORS_INFLATION['core'], linewidth=2, marker='s', markersize=4)
        ax2.set_ylabel('CPI Index', fontsize=11)
        ax2.set_title('Consumer Price Index Trend', fontsize=13, fontweight='600', pad=15)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
        _style_axis(ax2)
        plt.tight_layout()
        trend_chart = _fig_to_base64(fig)
        
        headline = overall['headline_inflation'] or 0
        status = "Low inflation" if headline < 2 else "Target range" if headline < 3 else "Elevated inflation" if headline < 5 else "High inflation"
        diff = headline - request.inflation_target
        vs_target = "At target" if abs(diff) < 0.5 else f"{diff:.1f}%p above target" if diff > 0 else f"{abs(diff):.1f}%p below target"
        
        summary = {'analysis_period': request.analysis_period, 'headline_inflation': headline, 'core_inflation': overall['core_inflation'], 'mom_change': overall['mom_change'], 'highest_category': categories[0]['category'] if categories else 'N/A', 'lowest_category': categories[-1]['category'] if categories else 'N/A', 'inflation_status': status, 'vs_target': vs_target}
        
        insights = []
        if headline < request.inflation_target:
            insights.append({'title': 'Below Target Inflation', 'description': f'Headline inflation {headline:.1f}% is below {request.inflation_target}% target.', 'status': 'positive'})
        elif headline > request.inflation_target + 2:
            insights.append({'title': 'Inflation Above Target', 'description': f'Headline {headline:.1f}% exceeds target significantly.', 'status': 'warning'})
        
        return {'success': True, 'overall_metrics': overall, 'temporal_analysis': temporal_result, 'category_analysis': categories, 'visualizations': {'trend_chart': trend_chart}, 'key_insights': insights, 'summary': summary}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inflation analysis failed: {str(e)}")


# ============================================
# INDUSTRY STRUCTURE ANALYSIS
# ============================================

class IndustryRequest(BaseModel):
    data: List[Dict[str, Any]]
    period_col: Optional[str] = None
    industry_col: str
    sector_col: Optional[str] = None
    employment_col: str
    region_col: Optional[str] = None
    value_added_col: Optional[str] = None
    productivity_col: Optional[str] = None
    wage_col: Optional[str] = None
    analysis_focus: str = "structure"
    analysis_period: str = "Analysis Period"


def classify_sector(industry_name: str) -> str:
    name = industry_name.lower()
    if any(k in name for k in ['agricult', 'farm', 'fish', 'mining', 'forestry', 'livestock']):
        return 'Primary'
    if any(k in name for k in ['manufactur', 'construct', 'mfg', 'factory', 'textile', 'chemical', 'auto', 'machine', 'electron']):
        return 'Secondary'
    if any(k in name for k in ['it', 'software', 'r&d', 'research', 'consult', 'tech']):
        return 'Quaternary'
    return 'Tertiary'


@app.post("/api/analysis/industry-structure")
async def run_industry_analysis(request: IndustryRequest) -> Dict[str, Any]:
    try:
        df = pd.DataFrame(request.data)
        if request.industry_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Industry column '{request.industry_col}' not found")
        if request.employment_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Employment column '{request.employment_col}' not found")
        
        df[request.employment_col] = pd.to_numeric(df[request.employment_col], errors='coerce').fillna(0)
        
        sector_col = request.sector_col
        if not sector_col or sector_col not in df.columns:
            df['_sector'] = df[request.industry_col].apply(classify_sector)
            sector_col = '_sector'
        
        total_emp = df[request.employment_col].sum()
        n_industries = df[request.industry_col].nunique()
        
        sectors = [{'sector': str(sector), 'employment': _to_native(df[df[sector_col] == sector][request.employment_col].sum()), 'employment_share': _to_native(df[df[sector_col] == sector][request.employment_col].sum() / total_emp * 100 if total_emp > 0 else 0), 'n_industries': _to_native(df[df[sector_col] == sector][request.industry_col].nunique())} for sector in df[sector_col].unique()]
        sectors = sorted(sectors, key=lambda x: x['employment'] or 0, reverse=True)
        
        industries = [{'industry': str(industry), 'employment': _to_native(df[df[request.industry_col] == industry][request.employment_col].sum()), 'employment_share': _to_native(df[df[request.industry_col] == industry][request.employment_col].sum() / total_emp * 100 if total_emp > 0 else 0), 'sector': str(df[df[request.industry_col] == industry][sector_col].iloc[0]) if len(df[df[request.industry_col] == industry]) > 0 else 'Unknown'} for industry in df[request.industry_col].unique()]
        industries = sorted(industries, key=lambda x: x['employment'] or 0, reverse=True)
        
        overall = {'total_employment': _to_native(total_emp), 'n_industries': _to_native(n_industries), 'n_sectors': _to_native(len(sectors)), 'largest_industry': industries[0]['industry'] if industries else 'N/A'}
        
        _setup_style()
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        sector_names, sector_shares = [s['sector'] for s in sectors], [s['employment_share'] for s in sectors]
        colors = [COLORS_INDUSTRY.get(s.lower(), COLORS_INDUSTRY['neutral']) for s in sector_names]
        ax1.pie(sector_shares, labels=sector_names, colors=colors, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 1})
        ax1.set_title('Employment by Sector', fontsize=13, fontweight='600', pad=15)
        
        top_ind = industries[:10]
        ax2.barh([i['industry'][:15] for i in top_ind], [i['employment'] / 1000000 for i in top_ind], color=COLORS_INDUSTRY['neutral'], edgecolor='white', height=0.7)
        ax2.set_xlabel('Employment (Millions)', fontsize=11)
        ax2.set_title('Top Industries by Employment', fontsize=13, fontweight='600', pad=15)
        ax2.invert_yaxis()
        _style_axis(ax2)
        plt.tight_layout()
        structure_chart = _fig_to_base64(fig)
        
        dominant = sectors[0] if sectors else {'sector': 'N/A', 'employment_share': 0}
        summary = {'analysis_period': request.analysis_period, 'total_employment': overall['total_employment'], 'dominant_sector': dominant['sector'], 'dominant_sector_share': dominant['employment_share'], 'n_industries': overall['n_industries'], 'top_industry': industries[0]['industry'] if industries else 'N/A'}
        
        insights = [{'title': f'{dominant["sector"]} Sector Dominates', 'description': f'{dominant["sector"]} accounts for {dominant["employment_share"]:.1f}% of employment.', 'status': 'neutral'}] if sectors else []
        
        return {'success': True, 'overall_metrics': overall, 'sector_analysis': sectors, 'industry_analysis': industries, 'visualizations': {'sector_composition': structure_chart}, 'key_insights': insights, 'summary': summary}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Industry structure analysis failed: {str(e)}")


# ============================================
# ACCIDENT HOTSPOT ANALYSIS
# ============================================

class HotspotRequest(BaseModel):
    data: List[Dict[str, Any]]
    lat_col: str
    lng_col: str
    date_col: str
    time_col: Optional[str] = None
    severity_col: Optional[str] = None
    factor_col: Optional[str] = None
    weather_col: Optional[str] = None
    road_type_col: Optional[str] = None
    lighting_col: Optional[str] = None
    clustering_method: str = "dbscan"
    epsilon_km: float = 0.5
    min_samples: int = 10
    severity_weight: str = "severity_weighted"
    state: Optional[str] = "California"


def severity_to_score(severity: str) -> int:
    if pd.isna(severity):
        return 1
    return {'fatal': 4, 'serious injury': 3, 'serious_injury': 3, 'minor injury': 2, 'minor_injury': 2, 'property damage only': 1, 'property_damage': 1, 'pdo': 1}.get(str(severity).lower().strip(), 1)


def get_risk_level(accident_count: int, severity_score: float) -> str:
    if accident_count >= 300 or severity_score >= 2.5:
        return "critical"
    elif accident_count >= 200 or severity_score >= 2.0:
        return "high"
    elif accident_count >= 100:
        return "medium"
    return "low"


@app.post("/api/analysis/hotspot")
async def analyze_hotspots(request: HotspotRequest) -> Dict[str, Any]:
    try:
        df = pd.DataFrame(request.data)
        if len(df) < 10:
            raise HTTPException(status_code=400, detail="Insufficient data. Need at least 10 records.")
        if request.lat_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Latitude column '{request.lat_col}' not found")
        if request.lng_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Longitude column '{request.lng_col}' not found")
        
        df[request.lat_col] = pd.to_numeric(df[request.lat_col], errors='coerce')
        df[request.lng_col] = pd.to_numeric(df[request.lng_col], errors='coerce')
        df = df.dropna(subset=[request.lat_col, request.lng_col])
        
        coords = df[[request.lat_col, request.lng_col]].values
        epsilon_rad = request.epsilon_km / 6371.0088
        clustering = DBSCAN(eps=epsilon_rad, min_samples=request.min_samples, metric='haversine')
        df['cluster'] = clustering.fit_predict(np.radians(coords))
        
        clusters = []
        for cluster_id in sorted(df[df['cluster'] >= 0]['cluster'].unique()):
            cdata = df[df['cluster'] == cluster_id]
            severity_score = cdata[request.severity_col].apply(severity_to_score).mean() if request.severity_col and request.severity_col in cdata.columns else 1.5
            clusters.append({'cluster_id': int(cluster_id), 'centroid_lat': _to_native(cdata[request.lat_col].mean()), 'centroid_lng': _to_native(cdata[request.lng_col].mean()), 'accident_count': len(cdata), 'severity_score': _to_native(severity_score), 'risk_level': get_risk_level(len(cdata), severity_score)})
        clusters = sorted(clusters, key=lambda x: x['accident_count'], reverse=True)
        
        _setup_style()
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(df[request.lng_col], df[request.lat_col], c=df['cluster'], cmap='tab10', s=20, alpha=0.6)
        for c in clusters[:5]:
            ax.scatter(c['centroid_lng'], c['centroid_lat'], c=COLORS_TRAFFIC['critical'], s=200, marker='*', edgecolor='white', linewidth=1, zorder=5)
        ax.set_xlabel('Longitude', fontsize=11)
        ax.set_ylabel('Latitude', fontsize=11)
        ax.set_title('Accident Hotspot Clusters', fontsize=13, fontweight='600', pad=15)
        _style_axis(ax)
        plt.tight_layout()
        hotspot_map = _fig_to_base64(fig)
        
        critical_count = sum(1 for c in clusters if c['risk_level'] == 'critical')
        summary = {'total_accidents': _to_native(len(df)), 'total_hotspots': _to_native(len(clusters)), 'critical_hotspots': _to_native(critical_count), 'clustering_method': request.clustering_method}
        
        insights = [{'title': 'Critical Hotspots Identified', 'description': f'{critical_count} critical accident hotspots require immediate attention.', 'status': 'warning'}] if critical_count > 0 else []
        
        return {'success': True, 'hotspot_analysis': {'total_accidents': len(df), 'clusters': clusters}, 'visualizations': {'hotspot_map': hotspot_map}, 'key_insights': insights, 'summary': summary}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hotspot analysis failed: {str(e)}")


# ============================================
# PARKING DEMAND FORECAST
# ============================================

class ParkingForecastRequest(BaseModel):
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


@app.post("/api/analysis/parking-forecast")
async def forecast_parking_demand(request: ParkingForecastRequest) -> Dict[str, Any]:
    try:
        df = pd.DataFrame(request.data)
        if len(df) < 100:
            raise HTTPException(status_code=400, detail="Insufficient data. Need at least 100 records.")
        if request.demand_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Demand column '{request.demand_col}' not found")
        
        df['date_parsed'] = pd.to_datetime(df[request.date_col], errors='coerce')
        df['day_of_week'] = df['date_parsed'].dt.dayofweek
        df['hour'] = pd.to_numeric(df[request.hour_col], errors='coerce').fillna(12).astype(int) if request.hour_col and request.hour_col in df.columns else 12
        df[request.demand_col] = pd.to_numeric(df[request.demand_col], errors='coerce').fillna(0)
        
        base_demand, peak_demand = df[request.demand_col].mean(), df[request.demand_col].max()
        total_capacity = int(df[request.capacity_col].sum() / len(df[request.capacity_col].unique())) if request.capacity_col and request.capacity_col in df.columns else int(base_demand * 1.3)
        avg_util = (base_demand / total_capacity * 100) if total_capacity > 0 else 50
        peak_util = (peak_demand / total_capacity * 100) if total_capacity > 0 else 80
        
        hourly_forecast = []
        for hour in range(24):
            hour_data = df[df['hour'] == hour]
            avg_demand = hour_data[request.demand_col].mean() if len(hour_data) > 0 else base_demand
            occupancy = (avg_demand / total_capacity * 100) if total_capacity > 0 else 50
            hourly_forecast.append({'hour': hour, 'predicted_demand': _to_native(int(avg_demand)), 'occupancy_rate': _to_native(round(occupancy, 1)), 'demand_level': "critical" if occupancy >= 90 else "high" if occupancy >= 75 else "moderate" if occupancy >= 50 else "low"})
        
        zones = []
        if request.zone_col and request.zone_col in df.columns:
            for zone in df[request.zone_col].unique():
                zdf = df[df[request.zone_col] == zone]
                zone_demand, zone_peak = zdf[request.demand_col].mean(), zdf[request.demand_col].max()
                zone_capacity = zdf[request.capacity_col].mean() if request.capacity_col and request.capacity_col in zdf.columns else zone_demand * 1.3
                zones.append({'zone_name': str(zone), 'avg_demand': _to_native(int(zone_demand)), 'peak_demand': _to_native(int(zone_peak)), 'peak_occupancy_rate': _to_native(round(zone_peak / zone_capacity * 100 if zone_capacity > 0 else 50, 1))})
            zones = sorted(zones, key=lambda x: x['peak_occupancy_rate'], reverse=True)
        
        _setup_style()
        fig, ax = plt.subplots(figsize=(14, 6))
        colors = [COLORS_TRAFFIC['critical'] if h['occupancy_rate'] >= 90 else COLORS_TRAFFIC['high'] if h['occupancy_rate'] >= 75 else COLORS_TRAFFIC['low'] for h in hourly_forecast]
        ax.bar([h['hour'] for h in hourly_forecast], [h['predicted_demand'] for h in hourly_forecast], color=colors, edgecolor='white')
        ax.axhline(total_capacity * 0.85, color=COLORS_TRAFFIC['critical'], linestyle='--', linewidth=2, label='85% Capacity')
        ax.set_xlabel('Hour of Day', fontsize=11)
        ax.set_ylabel('Predicted Demand', fontsize=11)
        ax.set_title('Hourly Parking Demand Forecast', fontsize=13, fontweight='600', pad=15)
        ax.legend(fontsize=9)
        ax.set_xticks(range(0, 24, 2))
        _style_axis(ax)
        plt.tight_layout()
        forecast_chart = _fig_to_base64(fig)
        
        capacity_analysis = {'total_capacity': _to_native(total_capacity), 'avg_utilization': _to_native(round(avg_util, 1)), 'peak_utilization': _to_native(round(peak_util, 1)), 'overflow_risk_hours': [h['hour'] for h in hourly_forecast if h['occupancy_rate'] >= 85]}
        summary = {'forecast_period': f"{request.forecast_horizon} days", 'total_zones': _to_native(len(zones)), 'total_capacity': _to_native(total_capacity), 'avg_daily_demand': _to_native(int(base_demand)), 'peak_demand': _to_native(int(peak_demand)), 'avg_occupancy_rate': _to_native(round(avg_util, 1))}
        
        insights = [{'title': 'Capacity Constraints', 'description': f'Peak utilization of {peak_util:.1f}% indicates potential overflow.', 'status': 'warning'}] if peak_util > 90 else []
        
        return {'success': True, 'hourly_forecast': hourly_forecast, 'zone_forecasts': zones, 'capacity_analysis': capacity_analysis, 'visualizations': {'demand_forecast': forecast_chart}, 'key_insights': insights, 'summary': summary}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parking forecast failed: {str(e)}")


# ============================================
# BUDGET EXECUTION ANALYSIS
# ============================================

class BudgetExecutionRequest(BaseModel):
    data: List[Dict[str, Any]]
    region_col: str
    allocated_col: str
    executed_col: str
    period_col: Optional[str] = None
    category_col: Optional[str] = None
    population_col: Optional[str] = None
    analysis_type: str = "execution_rate"
    benchmark_type: str = "national_average"
    target_rate: Optional[float] = None
    fiscal_period: str = "Current Period"


@app.post("/api/analysis/budget-execution")
async def run_budget_execution_analysis(request: BudgetExecutionRequest) -> Dict[str, Any]:
    try:
        df = pd.DataFrame(request.data)
        if request.region_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Region column '{request.region_col}' not found")
        if request.allocated_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Allocated column '{request.allocated_col}' not found")
        if request.executed_col not in df.columns:
            raise HTTPException(status_code=400, detail=f"Executed column '{request.executed_col}' not found")
        
        df[request.allocated_col] = pd.to_numeric(df[request.allocated_col], errors='coerce').fillna(0)
        df[request.executed_col] = pd.to_numeric(df[request.executed_col], errors='coerce').fillna(0)
        
        regional_data = df.groupby(request.region_col).agg({request.allocated_col: 'sum', request.executed_col: 'sum'}).reset_index()
        regional_data['execution_rate'] = regional_data[request.executed_col] / regional_data[request.allocated_col]
        regional_data = regional_data.sort_values('execution_rate', ascending=False)
        regional_data['rank'] = range(1, len(regional_data) + 1)
        
        regions = [{'region': row[request.region_col], 'allocated_budget': _to_native(row[request.allocated_col]), 'executed_amount': _to_native(row[request.executed_col]), 'execution_rate': _to_native(row['execution_rate']), 'rank': int(row['rank'])} for _, row in regional_data.iterrows()]
        
        total_allocated, total_executed = regional_data[request.allocated_col].sum(), regional_data[request.executed_col].sum()
        overall_rate = total_executed / total_allocated if total_allocated > 0 else 0
        disparity_index = np.std(regional_data['execution_rate'].values) / np.mean(regional_data['execution_rate'].values) if np.mean(regional_data['execution_rate'].values) > 0 else 0
        
        regional_analysis = {'regions': regions, 'total_allocated': _to_native(total_allocated), 'total_executed': _to_native(total_executed), 'overall_execution_rate': _to_native(overall_rate), 'disparity_index': _to_native(disparity_index)}
        
        category_analysis = []
        if request.category_col and request.category_col in df.columns:
            cat_data = df.groupby(request.category_col).agg({request.allocated_col: 'sum', request.executed_col: 'sum'}).reset_index()
            cat_data['execution_rate'] = cat_data[request.executed_col] / cat_data[request.allocated_col]
            category_analysis = [{'category': row[request.category_col], 'total_allocated': _to_native(row[request.allocated_col]), 'total_executed': _to_native(row[request.executed_col]), 'execution_rate': _to_native(row['execution_rate'])} for _, row in cat_data.iterrows()]
        
        benchmark_analysis = {'average_execution_rate': _to_native(overall_rate), 'above_average_regions': [r['region'] for r in regions if (r['execution_rate'] or 0) > overall_rate], 'below_average_regions': [r['region'] for r in regions if (r['execution_rate'] or 0) <= overall_rate]}
        
        _setup_style()
        fig, ax = plt.subplots(figsize=(14, 8))
        region_names = [r['region'] for r in regions[:15]]
        exec_rates = [(r['execution_rate'] or 0) * 100 for r in regions[:15]]
        colors = [COLORS_BUDGET['excellent'] if r >= 90 else COLORS_BUDGET['good'] if r >= 80 else COLORS_BUDGET['fair'] if r >= 70 else COLORS_BUDGET['poor'] for r in exec_rates]
        ax.barh(region_names, exec_rates, color=colors, edgecolor='white', height=0.7)
        ax.axvline(x=overall_rate * 100, color=COLORS_BUDGET['danger'], linestyle='--', linewidth=2, label=f'Average ({overall_rate*100:.1f}%)')
        ax.set_xlabel('Execution Rate (%)', fontsize=11)
        ax.set_title('Budget Execution Rate by Region', fontsize=13, fontweight='600', pad=15)
        ax.legend(fontsize=9)
        _style_axis(ax)
        plt.tight_layout()
        regional_chart = _fig_to_base64(fig)
        
        summary = {'n_regions': len(regions), 'n_categories': len(category_analysis), 'fiscal_period': request.fiscal_period, 'total_allocated': regional_analysis['total_allocated'], 'total_executed': regional_analysis['total_executed'], 'overall_execution_rate': regional_analysis['overall_execution_rate'], 'best_performing_region': regions[0]['region'] if regions else 'N/A', 'worst_performing_region': regions[-1]['region'] if regions else 'N/A', 'disparity_index': regional_analysis['disparity_index']}
        
        rate = summary['overall_execution_rate']
        insights = []
        if rate >= 0.9:
            insights.append({'title': 'Excellent Overall Execution', 'description': f'Overall execution rate of {rate*100:.1f}% exceeds targets.', 'status': 'positive'})
        elif rate >= 0.8:
            insights.append({'title': 'Good Execution Performance', 'description': f'Overall execution rate of {rate*100:.1f}% meets expectations.', 'status': 'neutral'})
        else:
            insights.append({'title': 'Execution Below Target', 'description': f'Overall rate of {rate*100:.1f}% is below 80% target.', 'status': 'warning'})
        
        return {'success': True, 'regional_analysis': regional_analysis, 'category_analysis': category_analysis, 'benchmark_analysis': benchmark_analysis, 'visualizations': {'regional_comparison': regional_chart}, 'key_insights': insights, 'summary': summary}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Budget execution analysis failed: {str(e)}")


#public sector
from api.spending_multiplier import router as sm_router
from api.population_forecast import router as pf_router



#marketing
from api.demand_forecast import router as fc_router
from api.basket_analysis import router as basket_router

from api.demand_forecasting import router as demand_forecasting_router
from api.revenue_forecasting import router as revenue_forecasting_router
from api.conjoint_analysis import router as conjoint_router
from api.cva_analysis import router as cva_router
from api.mmm import router as mmm_router
from api.attribution import router as attr_router
from api.sentiment_analysis import router as sent_router
from api.audience_segmentation import router as as_router
from api.inventory_optimization import router as inventory_optimization_router
from api.fulfillment_delivery import router as fulfillment_delivery_router


# survey
from api.portfolio_optimization import router as po_router


#pipeline
from api.pipeline import router as pipeline_router



#manufacturing
from api.weibull_analysis import router as weibull_router
from api.attribute_agreement import router as attr_agree_router
from api.warranty_analysis import router as warranty_router
from api.gage_rr1 import router as gage_rr1_router


# Audio Analysis
from api.audio_feature_analysis import router as audio_feature_router
from api.audio_anomaly_detection import router as audio_anomaly_router
from api.fft_spectrum_analysis import router as fft_spectrum_router
from api.stft_analysis import router as stft_router
from api.mel_spectrogram_analysis import router as mel_spectrogram_router
from api.waveform_analysis import router as waveform_router
from api.tonnetz_analysis import router as tonnetz_router
from api.mel_filterbank_analysis import router as mel_fb_router
from api.chromagram_analysis import router as chromagram_router
from api.zcr_analysis import router as zcr_router
from api.rms_energy_analysis import router as rms_energy_router
from api.mfcc_analysis import router as mfcc_router
from api.audio_segmentation import router as seg_router
from api.silence_activity_detection import router as sad_router
from api.hpss_analysis import router as hpss_router
from api.pitch_tracking import router as pitch_router
from api.onset_detection import router as onset_router
from api.beat_tracking import router as beat_router


# ============================================
# ROUTER IMPORTS - Financial Modeling
# ============================================

#finance

from api.fama_french import router as ff3_router
from api.price_projection import router as pp_router
from api.black_litterman import router as bl_router
from api.bankruptcy import router as bk_router
from api.anomaly_detection import router as ad_router
from api.corwin_schultz import router as cs_router
from api.som_cluster import router as som_router
from api.crash_risk import router as cr_router
from api.market_impact import router as mi_router
from api.copula_dependence import router as cop_router
from api.stress_test import router as st_router
from api.factor_analysis import router as factor_analysis_router
from api.strategy_backtest import router as strategy_backtest_router

from api.multi_factor_pricing import router as multi_factor_router


# ============================================
# ROUTER IMPORTS - Business Analysis
# ============================================




from api.ad_response import router as ad_response_router
from api.benchmark_analysis import router as benchmark_router
from api.bottleneck_analysis import router as bottleneck_analysis_router
from api.brand_image import router as brand_image_router
from api.churn_risk import router as churn_router
from api.competency_analysis import router as competency_router
from api.demand_elasticity import router as elasticity_router
from api.internal_communication import router as internal_comm_router
from api.kpi_analysis import router as kpi_router
from api.org_health import router as org_health_router
from api.perception_analysis import router as perception_router
from api.portfolio_analysis import router as portfolio_analysis_router
from api.price_sensitivity_analysis import router as price_router
from api.quality_control import router as quality_control_router
from api.risk_safety import router as risk_safety_router
from api.roi_analysis import router as roi_router
from api.satisfaction_analysis import router as satisfaction_router
from api.segmentation_analysis import router as seg_analysis_router
from api.traffic_analysis import router as traffic_router
from api.trend_analysis import router as trend_router
from api.breakeven_analysis import router as breakeven_router
from api.customer_segmentation import router as customer_segmentation_router
from api.sales_forecast import router as forecast_router
#from api.gagerr import router as gagerr_router
from api.oee import router as oee_router
from api.ltv_forecasting import router as ltv_forecasting_router
from api.campaign_performance import router as campaign_performance_router
from api.conversion_rate import router as conversion_rate_router
from api.clv_analysis import router as clv_analysis_router
from api.engagement_analysis import router as engagement_analysis_router
from api.adoption_analysis import router as adoption_analysis_router
from api.survey_analysis import router as survey_analysis_router
from api.compensation_analysis import router as compensation_analysis_router
from api.survival_curves import router as survival_curves_router
from api.cox_regression import router as cox_regression_router
from api.process_mining import router as process_mining_router
from api.vrp_analysis import router as vrp_analysis_router
from api.scheduling_analysis import router as scheduling_analysis_router
from api.assignment import router as assignment_router
from api.bin_packing import router as bin_packing_router
from api.tsp import router as tsp_router
from api.knapsack import router as knapsack_router
from api.funnel_analysis import router as funnel_analysis_router
from api.cohort_analysis import router as cohort_analysis_router
from api.attribution_modeling import router as attribution_modeling_router
from api.fds_anomaly_detection import router as fds_anomaly_detection_router
from api.yield_defect_analysis import router as yield_defect_analysis_router
from api.cashflow_forecast import router as cashflow_forecast_router
from api.capability_analysis import router as capability_analysis_router
from api.xgboost_analysis import router as xgboost_router
from api.gradient_boosting import router as gradient_boosting_router
from api.credit_risk_scoring import router as credit_risk_router
from api.inventory_analysis import router as inventory_analysis_router
from api.diversity_analysis import router as diversity_analysis_router
from api.absenteeism_analysis import router as absenteeism_analysis_router
from api.promotion_optimization import router as promo_router
from api.dea_efficiency import router as dea_efficiency_router
from api.lead_scoring import router as lead_router
from api.rfm_segmentation import router as rfm_router
from api.aha_moment import router as aha_router
from api.next_best_action import router as nba_router
from api.clv_forecasting import router as clv_router


# Statistical Analysis
from api.descriptive import router as descriptive_router
from api.frequency import router as frequency_router
from api.variability import router as variability_router
from api.spc import router as spc_router
from api.normality_test import router as normality_test_router
from api.homogeneity_test import router as homogeneity_test_router
from api.outlier_influence import router as outlier_influence_router
from api.linearity_test import router as linearity_test_router
from api.one_sample_t_test import router as one_sample_t_test_router
from api.independent_t_test import router as independent_t_test_router
from api.welchs_t_test import router as welchs_t_test_router
from api.paired_t_test import router as paired_t_test_router
from api.anova import router as anova_router
from api.two_way_anova import router as two_way_anova_router
from api.ancova import router as ancova_router
from api.manova import router as manova_router
from api.repeated_measures_anova import router as repeated_measures_anova_router
from api.two_way_rm_anova import router as rm_anova_router


from api.mann_whitney import router as mann_whitney_router
from api.wilcoxon import router as wilcoxon_router
from api.kruskal_wallis import router as kruskal_wallis_router
from api.friedman import router as friedman_router
from api.correlation import router as correlation_router
from api.chi_square import router as chi_square_router
from api.regression import router as regression_router
from api.logistic_regression import router as logistic_regression_router
from api.lasso_regression import router as lasso_regression_router
from api.ridge_regression import router as ridge_regression_router
from api.simple_regression import router as simple_regression_router
from api.robust_regression import router as robust_regression_router
from api.glm import router as glm_router
from api.relative_importance import router as relative_importance_router
from api.feature_importance import router as feature_importance_router
from api.discriminant_analysis import router as lda_router
from api.decision_tree_analysis import router as decision_tree_router
from api.gradient_descent_analysis import router as gradient_descent_router
from api.random_forest import router as random_forest_router
from api.svm import router as svm_router
from api.knn import router as knn_router
from api.naive_bayes_analysis import router as naive_bayes_router
from api.survival import router as survival_router
from api.did import router as did_router
from api.psm import router as psm_router
from api.rdd import router as rdd_router
from api.iv import router as iv_router
from api.var import router as var_router
from api.var_analysis import router as var_analysis_router
from api.gmm import router as gmm_router
from api.dsge import router as dsge_router
from api.mds import router as mds_router
from api.cross_validation import router as cross_validation_router
from api.reliability import router as reliability_router
from api.efa import router as efa_router
from api.cfa import router as cfa_router
from api.pca import router as pca_router
from api.mediation import router as mediation_router
from api.moderation import router as moderation_router
from api.sem import router as sem_router
from api.sem_platform import router as sem_platform_router
from api.sna import router as sna_router
from api.dea import router as dea_router
from api.driver import router as driver_router
from api.dea import router as dea_router
from api.kmeans import router as kmeans_router
from api.kmedoids import router as kmedoids_router
from api.dbscan import router as dbscan_router
from api.hdbscan import router as hdbscan_router
from api.hca import router as hca_router
from api.seasonal_analysis import router as seasonal_analysis_router
from api.rolling_statistics import router as rolling_statistics_router
from api.structural_break import router as structural_break_router
from api.change_point import router as change_point_router
from api.acf_pacf import router as acf_pacf_router
from api.stationarity import router as stationarity_router
from api.ljung_box import router as ljung_box_router
from api.arch_lm import router as arch_lm_router
from api.exponential_smoothing import router as exponential_smoothing_router
from api.arima import router as arima_router
from api.forecast_evaluation import router as forecast_evaluation_router
from api.demand_forecasting import router as demand_forecasting_router
from api.forecast_horizon import router as forecast_horizon_router
from api.churn_prediction import router as churn_router
from api.var_decomposition import router as var_decomposition_router  



# Adagrad import (NEW)
from api.adagrad_analysis import router as adagrad_router
from api.linear_programming import router as linear_programming_router
from api.integer_programming import router as integer_programming_router
from api.nonlinear_programming import router as nonlinear_programming_router
from api.goal_programming import router as goal_programming_router
from api.transportation_problem import router as transportation_problem_router
from api.p_median import router as p_median_router
from api.mclp import router as mclp_router
from api.location_allocation import router as location_allocation_router
from api.network import router as network_router
from api.reinforcement_learning import router as reinforcement_learning_router
from api.simulated_annealing import router as simulated_annealing_router
from api.ant_colony_optimization import router as ant_colony_optimization_router
from api.pareto_optimization import router as pareto_optimization_router
from api.tabu_search import router as tabu_search_router




from api.cva_dva_analysis import router as cva_dva_router
from api.var_risk_analysis import router as var_risk_router
from api.options_pricing import router as options_pricing_router
from api.dynamic_programming import router as dynamic_programming_router
#from api.stress_testing_analysis import router as stress_testing_router
from api.exotic_options_analysis import router as exotic_options_router
from api.convex_optimization import router as convex_optimization_router
from api.genetic_algorithm import router as genetic_algorithm_router
from api.particle_swarm_optimization import router as particle_swarm_optimization_router
from api.hyperparameter_tuning import router as hyperparameter_tuning_router

# ============================================
# MAP ANALYSIS ROUTERS
# ============================================
from map.upload import router as map_upload_router
from map.clustering import router as map_clustering_router
from map.clustering_advanced import router as map_clustering_adv_router
from map.spatial import router as map_spatial_router
from map.route import router as map_route_router
from map.analysis import router as map_analysis_router
from map.optimization import router as map_optimization_router
from map.geometry import router as map_geometry_router
from map.statistics import router as map_statistics_router
from map.network import router as map_network_router
from map.forecasting import router as map_forecasting_router


# main.py에서 확인 필요
# 1. diversity-inclusion 라우터

# 2. gage-rr-analysis 라우터  
#from api.gage_rr_analysis import router as gage_rr_router

# ============================================
# ROUTER REGISTRATION - ALL under /api/analysis/
# ============================================

#app.include_router(gage_rr_router)


#public sector
app.include_router(pf_router, prefix="/api/analysis")
app.include_router(sm_router, prefix="/api/analysis")

#marketing
app.include_router(demand_forecasting_router, prefix="/api/demand_forecasting")
app.include_router(revenue_forecasting_router, prefix="/api/revenue_forecasting")
app.include_router(conjoint_router, prefix="/api/analysis", tags=["Conjoint Analysis"])
app.include_router(cva_router, prefix="/api/analysis", tags=["CVA Conjoint Analysis"])
app.include_router(fc_router, prefix="/api/analysis", tags=["Demand Forecast"])
app.include_router(mmm_router, prefix="/api/analysis", tags=["Marketing Mix"])
app.include_router(attr_router, prefix="/api/analysis", tags=["Attribution"])
app.include_router(sent_router, prefix="/api/analysis", tags=["Sentiment Analysis"])
app.include_router(text_analysis_router, prefix="/api/analysis", tags=["Text Analysis"])
app.include_router(as_router, prefix="/api/analysis", tags=["Audience Segmentation"])
app.include_router(inventory_optimization_router, prefix="/api/analysis", tags=["Inventory Optimization"])
app.include_router(basket_router, prefix="/api/analysis", tags=["Basket Analysis"])
app.include_router(fulfillment_delivery_router, prefix="/api/analysis", tags=["Fulfillment Delivery"])


#manufacturing
app.include_router(weibull_router, prefix="/api/analysis")
app.include_router(attr_agree_router, prefix="/api/analysis")
app.include_router(warranty_router, prefix="/api/analysis")
app.include_router(gage_rr1_router, prefix="/api/analysis")

#pipeline
app.include_router(pipeline_router, prefix="/api/pipeline", tags=["Data Pipeline"])


# Audio Analysis
app.include_router(audio_feature_router, prefix="/api/analysis", tags=["Audio Analysis"])
app.include_router(audio_anomaly_router, prefix="/api/analysis", tags=["Audio Analysis"])
app.include_router(fft_spectrum_router, prefix="/api/analysis", tags=["Audio Analysis"])
app.include_router(stft_router, prefix="/api/analysis", tags=["Audio Analysis"])
app.include_router(mel_spectrogram_router, prefix="/api/analysis", tags=["Audio Analysis"])
app.include_router(waveform_router, prefix="/api/analysis", tags=["Audio Analysis"])
app.include_router(tonnetz_router, prefix="/api/analysis", tags=["Audio Analysis"])
app.include_router(mel_fb_router, prefix="/api/analysis", tags=["Audio Analysis"])
app.include_router(chromagram_router, prefix="/api/analysis", tags=["Audio Analysis"])
app.include_router(zcr_router, prefix="/api/analysis", tags=["Audio Analysis"])
app.include_router(rms_energy_router, prefix="/api/analysis", tags=["Audio Analysis"])
app.include_router(mfcc_router, prefix="/api/analysis", tags=["Audio Analysis"])
app.include_router(seg_router, prefix="/api/analysis", tags=["Audio Analysis"])
app.include_router(sad_router, prefix="/api/analysis", tags=["Audio Analysis"])
app.include_router(hpss_router, prefix="/api/analysis", tags=["Audio Analysis"])
app.include_router(pitch_router, prefix="/api/analysis", tags=["Audio Analysis"])
app.include_router(onset_router, prefix="/api/analysis", tags=["Audio Analysis"])
app.include_router(beat_router, prefix="/api/analysis", tags=["Audio Analysis"])
app.include_router(strategy_backtest_router, prefix="/api/analysis", tags=["Strategy Backtest"])



# Financial Modeling
app.include_router(ff3_router, prefix="/api/analysis", tags=["Fama-French 3-Factor"])
app.include_router(pp_router, prefix="/api/analysis", tags=["AI Price Projection"])
app.include_router(bl_router, prefix="/api/analysis", tags=["Black-Litterman"])
app.include_router(bk_router, prefix="/api/analysis", tags=["ML Bankruptcy Alert"])
app.include_router(ad_router, prefix="/api/analysis", tags=["Anomaly Detection"])
app.include_router(cs_router, prefix="/api/analysis", tags=["Corwin-Schultz Spread"])
app.include_router(som_router, prefix="/api/analysis", tags=["SOM Cluster"])
app.include_router(cr_router, prefix="/api/analysis", tags=["Crash Risk"])
app.include_router(mi_router, prefix="/api/analysis", tags=["Market Impact"])
app.include_router(cop_router, prefix="/api/analysis", tags=["Copula Dependence"])
app.include_router(st_router, prefix="/api/analysis", tags=["Stress Test"])
app.include_router(factor_analysis_router, prefix="/api/analysis", tags=["Factor Analysis"])
app.include_router(multi_factor_router, prefix="/api/analysis")


#survey
app.include_router(po_router, prefix="/api/analysis", tags=["Portfolio Optimization"])

# Business Analysis
app.include_router(ad_response_router, prefix="/api/analysis", tags=["Marketing"])
app.include_router(benchmark_router, prefix="/api/analysis", tags=["Benchmark"])
app.include_router(bottleneck_analysis_router, prefix="/api/analysis", tags=["Operations"])
app.include_router(brand_image_router, prefix="/api/analysis", tags=["Marketing"])
app.include_router(churn_router, prefix="/api/analysis", tags=["Customer"])
app.include_router(competency_router, prefix="/api/analysis", tags=["HR"])
app.include_router(internal_comm_router, prefix="/api/analysis", tags=["HR"])
app.include_router(kpi_router, prefix="/api/analysis", tags=["Performance"])
app.include_router(org_health_router, prefix="/api/analysis", tags=["HR"])
app.include_router(perception_router, prefix="/api/analysis", tags=["Marketing"])
app.include_router(portfolio_analysis_router, prefix="/api/analysis", tags=["Strategy"])
app.include_router(price_router, prefix="/api/analysis", tags=["Pricing"])
app.include_router(quality_control_router, prefix="/api/analysis", tags=["Quality"])
app.include_router(risk_safety_router, prefix="/api/analysis", tags=["Risk"])
app.include_router(roi_router, prefix="/api/analysis", tags=["Finance"])
app.include_router(satisfaction_router, prefix="/api/analysis", tags=["Customer"])
app.include_router(seg_analysis_router, prefix="/api/analysis", tags=["Marketing"])
app.include_router(traffic_router, prefix="/api/analysis", tags=["Operations"])
app.include_router(trend_router, prefix="/api/analysis", tags=["Analytics"])
app.include_router(breakeven_router, prefix="/api/analysis", tags=["Finance"])
app.include_router(lead_router, prefix="/api/analysis", tags=["Machine Learning"])
app.include_router(rfm_router, prefix="/api/analysis", tags=["Machine Learning"])
app.include_router(clv_router, prefix="/api/analysis", tags=["CLV Forecasting"])


# NEW Business Modules
app.include_router(customer_segmentation_router, prefix="/api/analysis", tags=["Customer Segmentation"])
app.include_router(forecast_router, prefix="/api/analysis", tags=["Machine Learning"])
#app.include_router(gagerr_router, prefix="/api/analysis", tags=["Quality - MSA"])
app.include_router(oee_router, prefix="/api/analysis", tags=["Production - OEE"])
app.include_router(ltv_forecasting_router, prefix="/api/analysis", tags=["Customer - LTV"])
app.include_router(decision_tree_router, prefix="/api/analysis", tags=["Decision Tree"])
app.include_router(cohort_analysis_router, prefix="/api/analysis", tags=["Cohort Analysis"])
app.include_router(attribution_modeling_router, prefix="/api/analysis", tags=["Marketing Attribution"])
app.include_router(fds_anomaly_detection_router, prefix="/api/analysis", tags=["Anomaly Detection"])
app.include_router(yield_defect_analysis_router, prefix="/api/analysis", tags=["Manufacturing"])
app.include_router(cashflow_forecast_router, prefix="/api/analysis", tags=["Finance - Cashflow"])
app.include_router(capability_analysis_router, prefix="/api/analysis", tags=["Quality - Capability"])
app.include_router(xgboost_router, prefix="/api/analysis", tags=["Machine Learning"])
app.include_router(gradient_boosting_router, prefix="/api/analysis", tags=["Machine Learning"])
app.include_router(credit_risk_router, prefix="/api/analysis", tags=["Finance - Credit Risk"])
app.include_router(inventory_analysis_router, prefix="/api/analysis", tags=["Operations - Inventory"])
app.include_router(diversity_analysis_router, prefix="/api/analysis", tags=["HR - Diversity"])
app.include_router(absenteeism_analysis_router, prefix="/api/analysis", tags=["HR - Absenteeism"])
app.include_router(promo_router, prefix="/api/analysis", tags=["Machine Learning"])
app.include_router(dea_efficiency_router, prefix="/api/analysis", tags=["Operations - DEA"])
app.include_router(campaign_performance_router, prefix="/api/analysis", tags=["Marketing - Campaign"])
app.include_router(conversion_rate_router, prefix="/api/analysis", tags=["Marketing - Conversion"])
app.include_router(clv_analysis_router, prefix="/api/analysis", tags=["Marketing - CLV"])
app.include_router(engagement_analysis_router, prefix="/api/analysis", tags=["HR - Engagement"])
app.include_router(adoption_analysis_router, prefix="/api/analysis", tags=["Product - Adoption"])
app.include_router(survey_analysis_router, prefix="/api/analysis", tags=["Research - Survey"])
app.include_router(compensation_analysis_router, prefix="/api/analysis", tags=["HR - Compensation"])
app.include_router(churn_router, prefix="/api/analysis", tags=["Machine Learning"])
app.include_router(nba_router, prefix="/api/analysis")



# Statistical Analysis
app.include_router(descriptive_router, prefix="/api/analysis", tags=["Descriptive Statistics"])
app.include_router(frequency_router, prefix="/api/analysis", tags=["Descriptive Statistics"])
app.include_router(variability_router, prefix="/api/analysis", tags=["Descriptive Statistics"])
app.include_router(spc_router, prefix="/api/analysis", tags=["SPC"])
app.include_router(normality_test_router, prefix="/api/analysis", tags=["Assumption Testing"])
app.include_router(homogeneity_test_router, prefix="/api/analysis", tags=["Assumption Testing"])
app.include_router(outlier_influence_router, prefix="/api/analysis", tags=["Assumption Testing"])
app.include_router(linearity_test_router, prefix="/api/analysis", tags=["Assumption Testing"])
app.include_router(one_sample_t_test_router, prefix="/api/analysis", tags=["T-Tests"])
app.include_router(independent_t_test_router, prefix="/api/analysis", tags=["T-Tests"])
app.include_router(welchs_t_test_router, prefix="/api/analysis", tags=["T-Tests"])
app.include_router(paired_t_test_router, prefix="/api/analysis", tags=["T-Tests"])
app.include_router(anova_router, prefix="/api/analysis", tags=["ANOVA"])
app.include_router(two_way_anova_router, prefix="/api/analysis", tags=["ANOVA"])
app.include_router(ancova_router, prefix="/api/analysis", tags=["ANOVA"])
app.include_router(manova_router, prefix="/api/analysis", tags=["ANOVA"])
app.include_router(repeated_measures_anova_router, prefix="/api/analysis", tags=["ANOVA"])
app.include_router(rm_anova_router, prefix="/api/analysis", tags=["ANOVA"]
)
app.include_router(mann_whitney_router, prefix="/api/analysis", tags=["Non-parametric Tests"])
app.include_router(wilcoxon_router, prefix="/api/analysis", tags=["Non-parametric Tests"])
app.include_router(kruskal_wallis_router, prefix="/api/analysis", tags=["Non-parametric Tests"])
app.include_router(friedman_router, prefix="/api/analysis", tags=["Non-parametric Tests"])
app.include_router(correlation_router, prefix="/api/analysis", tags=["Correlation"])
app.include_router(chi_square_router, prefix="/api/analysis", tags=["Correlation"])
app.include_router(regression_router, prefix="/api/analysis", tags=["Regression"])
app.include_router(logistic_regression_router, prefix="/api/analysis", tags=["Regression"])
app.include_router(lasso_regression_router, prefix="/api/analysis", tags=["Regression"])
app.include_router(ridge_regression_router, prefix="/api/analysis", tags=["Regression"])
app.include_router(simple_regression_router, prefix="/api/analysis", tags=["Regression"])
app.include_router(robust_regression_router, prefix="/api/analysis", tags=["Regression"])
app.include_router(glm_router, prefix="/api/analysis", tags=["Regression"])
app.include_router(relative_importance_router, prefix="/api/analysis", tags=["Regression"])
app.include_router(feature_importance_router, prefix="/api/analysis", tags=["Regression"])
app.include_router(lda_router, prefix="/api/analysis", tags=["Machine Learning"])
app.include_router(gradient_descent_router, prefix="/api/analysis", tags=["Machine Learning"])
app.include_router(random_forest_router, prefix="/api/analysis", tags=["Machine Learning"])
app.include_router(svm_router, prefix="/api/analysis", tags=["Machine Learning"])
app.include_router(knn_router, prefix="/api/analysis", tags=["Machine Learning"])
app.include_router(naive_bayes_router, prefix="/api/analysis", tags=["Machine Learning"])

# Survival Analysis
app.include_router(survival_router, prefix="/api/analysis", tags=["Survival Analysis"])
app.include_router(survival_curves_router, prefix="/api/analysis", tags=["Survival Analysis"])
app.include_router(cox_regression_router, prefix="/api/analysis", tags=["Survival Analysis"])

# Causal Inference
app.include_router(did_router, prefix="/api/analysis", tags=["Causal Inference"])
app.include_router(psm_router, prefix="/api/analysis", tags=["Causal Inference"])
app.include_router(rdd_router, prefix="/api/analysis", tags=["Causal Inference"])
app.include_router(iv_router,  prefix="/api/analysis", tags=["Causal Inference"])
app.include_router(scm_router, prefix="/api/analysis", tags=["Causal Inference"])

# Econometrics
app.include_router(var_router, prefix="/api/analysis", tags=["Econometrics"])
app.include_router(var_analysis_router, prefix="/api/analysis", tags=["Risk Management"])
app.include_router(gmm_router, prefix="/api/analysis", tags=["Econometrics"])
app.include_router(dsge_router, prefix="/api/analysis", tags=["Econometrics"])
app.include_router(var_decomposition_router, prefix="/api/analysis", tags=["Econometrics"])


# Multivariate Analysis
app.include_router(mds_router, prefix="/api/analysis", tags=["Multivariate Analysis"])
app.include_router(cross_validation_router, prefix="/api/analysis", tags=["Model Validation"])
app.include_router(reliability_router, prefix="/api/analysis", tags=["Reliability"])
app.include_router(efa_router, prefix="/api/analysis", tags=["Factor Analysis"])
app.include_router(cfa_router, prefix="/api/analysis", tags=["Factor Analysis"])
app.include_router(pca_router, prefix="/api/analysis", tags=["Dimension Reduction"])
app.include_router(mediation_router, prefix="/api/analysis", tags=["Mediation/Moderation"])
app.include_router(moderation_router, prefix="/api/analysis", tags=["Mediation/Moderation"])
app.include_router(sem_router, prefix="/api/analysis", tags=["Structural Equation Modeling"])
app.include_router(sem_platform_router, prefix="/api/analysis", tags=["SEM Platform"])
app.include_router(sna_router,    prefix="/api/analysis", tags=["Network Analysis"])
app.include_router(dea_router,    prefix="/api/analysis", tags=["Efficiency Analysis"])
app.include_router(driver_router, prefix="/api/analysis", tags=["Driver Analysis"])
app.include_router(dea_router, prefix="/api/analysis", tags=["Efficiency Analysis"])

# Clustering
app.include_router(kmeans_router, prefix="/api/analysis", tags=["Clustering"])
app.include_router(kmedoids_router, prefix="/api/analysis", tags=["Clustering"])
app.include_router(dbscan_router, prefix="/api/analysis", tags=["Clustering"])
app.include_router(hdbscan_router, prefix="/api/analysis", tags=["Clustering"])
app.include_router(hca_router, prefix="/api/analysis", tags=["Clustering"])

# Time Series Analysis
app.include_router(seasonal_analysis_router, prefix="/api/analysis", tags=["Time Series"])
app.include_router(rolling_statistics_router, prefix="/api/analysis", tags=["Time Series"])
app.include_router(structural_break_router, prefix="/api/analysis", tags=["Time Series"])
app.include_router(change_point_router, prefix="/api/analysis", tags=["Time Series"])
app.include_router(acf_pacf_router, prefix="/api/analysis", tags=["Time Series Diagnostics"])
app.include_router(stationarity_router, prefix="/api/analysis", tags=["Time Series Diagnostics"])
app.include_router(ljung_box_router, prefix="/api/analysis", tags=["Time Series Diagnostics"])
app.include_router(arch_lm_router, prefix="/api/analysis", tags=["Time Series Diagnostics"])

# Forecasting
app.include_router(exponential_smoothing_router, prefix="/api/analysis", tags=["Forecasting"])
app.include_router(arima_router, prefix="/api/analysis", tags=["Forecasting"])
app.include_router(forecast_evaluation_router, prefix="/api/analysis", tags=["Forecasting"])
app.include_router(demand_forecasting_router, prefix="/api/analysis", tags=["Forecasting"])
app.include_router(forecast_horizon_router, prefix="/api/analysis", tags=["Forecasting"])

# Operations Research
app.include_router(process_mining_router, prefix="/api/analysis", tags=["Process Mining"])
app.include_router(vrp_analysis_router, prefix="/api/analysis", tags=["Vehicle Routing"])
app.include_router(scheduling_analysis_router, prefix="/api/analysis", tags=["Scheduling"])
app.include_router(assignment_router, prefix="/api/analysis", tags=["Assignment Problem"])
app.include_router(bin_packing_router, prefix="/api/analysis", tags=["Bin Packing"])
app.include_router(tsp_router, prefix="/api/analysis", tags=["TSP"])
app.include_router(knapsack_router, prefix="/api/analysis", tags=["Knapsack Problem"])
app.include_router(funnel_analysis_router, prefix="/api/analysis", tags=["Funnel Analysis"])

# Optimization - Adagrad (NEW)
app.include_router(adagrad_router, prefix="/api/analysis", tags=["Optimization"])
app.include_router(linear_programming_router, prefix="/api/analysis", tags=["Optimization"])
app.include_router(integer_programming_router, prefix="/api/analysis", tags=["Optimization"])
app.include_router(nonlinear_programming_router, prefix="/api/analysis", tags=["Optimization"])
app.include_router(transportation_problem_router, prefix="/api/analysis", tags=["Optimization"])
app.include_router(simulated_annealing_router, prefix="/api/analysis", tags=["Optimization"])
app.include_router(ant_colony_optimization_router, prefix="/api/analysis", tags=["Optimization"])
app.include_router(tabu_search_router, prefix="/api/analysis", tags=["Optimization"])
app.include_router(pareto_optimization_router, prefix="/api/analysis", tags=["Optimization"])


app.include_router(cva_dva_router, prefix="/api/analysis", tags=["Risk Management"])
app.include_router(var_risk_router, prefix="/api/analysis", tags=["Risk Management"])
app.include_router(options_pricing_router, prefix="/api/analysis", tags=["Quantitative Finance"])
app.include_router(dynamic_programming_router, prefix="/api/analysis", tags=["Optimization"])
app.include_router(exotic_options_router, prefix="/api/analysis", tags=["exotic-options"])
app.include_router(convex_optimization_router, prefix="/api/analysis", tags=["convex-optimization"])
app.include_router(genetic_algorithm_router, prefix="/api/analysis", tags=["genetic-algorithm"])
#app.include_router(stress_testing_router, prefix="/api/analysis", tags=["Stress Testing"])
app.include_router(particle_swarm_optimization_router, prefix="/api/analysis", tags=["Optimization"])
app.include_router(p_median_router, prefix="/api/analysis", tags=["Spatial Optimization"])
app.include_router(mclp_router, prefix="/api/analysis", tags=["Spatial Optimization"])
app.include_router(location_allocation_router, prefix="/api/analysis", tags=["Spatial Optimization"])
app.include_router(network_router, prefix="/api/analysis", tags=["Spatial Optimization"])
app.include_router(hyperparameter_tuning_router, prefix="/api/analysis", tags=["Machine Learning"])
app.include_router(reinforcement_learning_router, prefix="/api/analysis", tags=["Machine Learning"])
app.include_router(elasticity_router, prefix="/api/analysis", tags=["Machine Learning"])
app.include_router(aha_router, prefix="/api/analysis", tags=["analysis"])

# ============================================
# MAP ANALYSIS ROUTER REGISTRATION
# ============================================
app.include_router(map_upload_router,        tags=["Map - Upload"])
app.include_router(map_clustering_router,    tags=["Map - Clustering"])
app.include_router(map_clustering_adv_router,tags=["Map - Clustering Advanced"])
app.include_router(map_spatial_router,       tags=["Map - Spatial"])
app.include_router(map_route_router,         tags=["Map - Route"])
app.include_router(map_analysis_router,      tags=["Map - Analysis"])
app.include_router(map_optimization_router,  tags=["Map - Optimization"])
app.include_router(map_geometry_router,      tags=["Map - Geometry"])
app.include_router(map_statistics_router,    tags=["Map - Statistics"])
app.include_router(map_network_router,       tags=["Map - Network"])
app.include_router(map_forecasting_router,   tags=["Map - Forecasting"])

# ============================================
# DATA UPLOAD ENDPOINT
# ============================================

class DataUploadResponse(BaseModel):
    success: bool
    data: List[Dict[str, Any]]
    columns: List[str]
    row_count: int


@app.post("/api/data/upload", response_model=DataUploadResponse)
async def upload_data(file: UploadFile = File(...)):
    """Upload CSV or Excel file for analysis"""
    try:
        content = await file.read()
        filename = file.filename.lower()
        
        if filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(io.BytesIO(content))
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV or Excel.")
        
        df.columns = df.columns.str.strip()
        data = df.to_dict(orient='records')
        
        return DataUploadResponse(success=True, data=data, columns=list(df.columns), row_count=len(df))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


# ============================================
# HEALTH CHECK & ROOT
# ============================================

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "version": "2.0.0"}


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {"message": "Statistica API", "version": "2.0.0", "docs": "/docs", "health": "/api/health"}


# ============================================
# API DOCUMENTATION
# ============================================

@app.get("/api/endpoints")
async def list_endpoints():
    """List all available analysis endpoints"""
    return {
        "public_policy_analysis": [
            {"path": "/api/analysis/birth-mortality", "method": "POST", "description": "Birth & Mortality Rate Trends Analysis"},
            {"path": "/api/analysis/migration", "method": "POST", "description": "Regional Population Migration Analysis"},
            {"path": "/api/analysis/unemployment", "method": "POST", "description": "Unemployment Trend Analysis"},
            {"path": "/api/analysis/income", "method": "POST", "description": "Regional Income Level Analysis"},
            {"path": "/api/analysis/inflation", "method": "POST", "description": "Inflation Rate Tracking Analysis"},
            {"path": "/api/analysis/industry-structure", "method": "POST", "description": "Employment & Industry Structure Analysis"},
            {"path": "/api/analysis/hotspot", "method": "POST", "description": "Accident Hotspot Analysis"},
            {"path": "/api/analysis/parking-forecast", "method": "POST", "description": "Parking Demand Forecast"},
            {"path": "/api/analysis/budget-execution", "method": "POST", "description": "Regional Budget Execution Comparison"},
        ],
        "business_analysis": [
            {"path": "/api/analysis/ad-response", "method": "POST", "description": "Ad Response Analysis"},
            {"path": "/api/analysis/benchmark-analysis", "method": "POST", "description": "Benchmark & Competitive Analysis"},
            {"path": "/api/analysis/bottleneck-analysis", "method": "POST", "description": "Process Bottleneck Analysis"},
            {"path": "/api/analysis/brand-image", "method": "POST", "description": "Brand Image Analysis"},
            {"path": "/api/analysis/churn-risk", "method": "POST", "description": "Churn Risk Analysis"},
            {"path": "/api/analysis/competency-analysis", "method": "POST", "description": "Competency Analysis"},
            {"path": "/api/analysis/internal-communication", "method": "POST", "description": "Internal Communication Analysis"},
            {"path": "/api/analysis/kpi-analysis", "method": "POST", "description": "KPI Performance Analysis"},
            {"path": "/api/analysis/org-health", "method": "POST", "description": "Organizational Health Analysis"},
            {"path": "/api/analysis/perception-analysis", "method": "POST", "description": "Perception Analysis"},
            {"path": "/api/analysis/portfolio-analysis", "method": "POST", "description": "Product Portfolio Analysis"},
            {"path": "/api/analysis/price-sensitivity", "method": "POST", "description": "Price Sensitivity Analysis"},
            {"path": "/api/analysis/quality-control", "method": "POST", "description": "Quality Control Analysis"},
            {"path": "/api/analysis/risk-safety", "method": "POST", "description": "Risk & Safety Analysis"},
            {"path": "/api/analysis/roi-analysis", "method": "POST", "description": "ROI Analysis"},
            {"path": "/api/analysis/satisfaction-analysis", "method": "POST", "description": "Satisfaction Analysis"},
            {"path": "/api/analysis/segmentation-analysis", "method": "POST", "description": "Segmentation Analysis"},
            {"path": "/api/analysis/traffic-analysis", "method": "POST", "description": "Traffic/Flow Analysis"},
            {"path": "/api/analysis/trend-analysis", "method": "POST", "description": "Trend Analysis"},
            {"path": "/api/analysis/break-even", "method": "POST", "description": "Break-even Analysis"},
            {"path": "/api/analysis/customer-segmentation", "method": "POST", "description": "Customer Segmentation (RFM)"},
            {"path": "/api/analysis/gage-rr", "method": "POST", "description": "Gage R&R (MSA)"},
            {"path": "/api/analysis/oee", "method": "POST", "description": "OEE Analysis"},
            {"path": "/api/analysis/mmm", "method": "POST", "description": "Marketing Mix Modeling"},
            {"path": "/api/analysis/basket", "method": "POST", "description": "Market Basket Analysis"},
            {"path": "/api/analysis/ltv", "method": "POST", "description": "LTV Prediction"},
            {"path": "/api/analysis/decision-tree-classifier", "method": "POST", "description": "Decision Tree Classifier"},
            {"path": "/api/analysis/cohort", "method": "POST", "description": "Cohort Analysis"},
            {"path": "/api/analysis/attribution", "method": "POST", "description": "Marketing Attribution"},
            {"path": "/api/analysis/fds", "method": "POST", "description": "FDS Anomaly Detection"},
            {"path": "/api/analysis/yield-defect", "method": "POST", "description": "Yield & Defect Analysis"},
            {"path": "/api/analysis/cashflow", "method": "POST", "description": "Cashflow Forecast"},
            {"path": "/api/analysis/capability", "method": "POST", "description": "Process Capability Analysis"},
        ],
        "statistical_analysis": [
            {"path": "/api/analysis/descriptive", "method": "POST", "description": "Descriptive Statistics"},
            {"path": "/api/analysis/frequency", "method": "POST", "description": "Frequency Analysis"},
            {"path": "/api/analysis/variability", "method": "POST", "description": "Variability Analysis"},
            {"path": "/api/analysis/normality-test", "method": "POST", "description": "Normality Test"},
            {"path": "/api/analysis/homogeneity-test", "method": "POST", "description": "Homogeneity Test"},
            {"path": "/api/analysis/outlier-influence", "method": "POST", "description": "Outlier & Influence Analysis"},
            {"path": "/api/analysis/linearity-test", "method": "POST", "description": "Linearity Test"},
            {"path": "/api/analysis/one-sample-t_test", "method": "POST", "description": "One Sample T-Test"},
            {"path": "/api/analysis/independent_t_test", "method": "POST", "description": "Independent T-Test"},
            {"path": "/api/analysis/welchs-t-test", "method": "POST", "description": "Welch's T-Test"},
            {"path": "/api/analysis/paired-t-test", "method": "POST", "description": "Paired T-Test"},
            {"path": "/api/analysis/anova", "method": "POST", "description": "One-Way ANOVA"},
            {"path": "/api/analysis/two-way-anova", "method": "POST", "description": "Two-Way ANOVA"},
            {"path": "/api/analysis/ancova", "method": "POST", "description": "ANCOVA"},
            {"path": "/api/analysis/manova", "method": "POST", "description": "MANOVA"},
            {"path": "/api/analysis/repeated-measures-anova", "method": "POST", "description": "Repeated Measures ANOVA"},
            {"path": "/api/analysis/mann-whitney", "method": "POST", "description": "Mann-Whitney U Test"},
            {"path": "/api/analysis/wilcoxon", "method": "POST", "description": "Wilcoxon Signed-Rank Test"},
            {"path": "/api/analysis/kruskal-wallis", "method": "POST", "description": "Kruskal-Wallis Test"},
            {"path": "/api/analysis/friedman", "method": "POST", "description": "Friedman Test"},
            {"path": "/api/analysis/correlation", "method": "POST", "description": "Correlation Analysis"},
            {"path": "/api/analysis/chi-square", "method": "POST", "description": "Chi-Square Test"},
            {"path": "/api/analysis/regression", "method": "POST", "description": "Linear Regression"},
            {"path": "/api/analysis/simple-regression", "method": "POST", "description": "Simple Regression"},
            {"path": "/api/analysis/logistic-regression", "method": "POST", "description": "Logistic Regression"},
            {"path": "/api/analysis/lasso-regression", "method": "POST", "description": "Lasso Regression"},
            {"path": "/api/analysis/ridge-regression", "method": "POST", "description": "Ridge Regression"},
            {"path": "/api/analysis/robust-regression", "method": "POST", "description": "Robust Regression"},
            {"path": "/api/analysis/glm", "method": "POST", "description": "Generalized Linear Model"},
            {"path": "/api/analysis/relative-importance", "method": "POST", "description": "Relative Importance"},
            {"path": "/api/analysis/feature-importance", "method": "POST", "description": "Feature Importance"},
        ],
        "machine_learning": [
            {"path": "/api/analysis/lda", "method": "POST", "description": "Linear Discriminant Analysis"},
            {"path": "/api/analysis/decision-tree", "method": "POST", "description": "Decision Tree"},
            {"path": "/api/analysis/gradient-descent", "method": "POST", "description": "Gradient Descent"},
            {"path": "/api/analysis/random-forest", "method": "POST", "description": "Random Forest"},
            {"path": "/api/analysis/svm", "method": "POST", "description": "Support Vector Machine"},
            {"path": "/api/analysis/knn", "method": "POST", "description": "K-Nearest Neighbors"},
            {"path": "/api/analysis/naive-bayes", "method": "POST", "description": "Naive Bayes"},
            {"path": "/api/analysis/xgboost", "method": "POST", "description": "XGBoost"},
            {"path": "/api/analysis/gradient-boosting", "method": "POST", "description": "Gradient Boosting"},
            {"path": "/api/analysis/credit-risk", "method": "POST", "description": "Credit Risk Scoring"},
            {"path": "/api/analysis/inventory", "method": "POST", "description": "Inventory Analysis"},
            {"path": "/api/analysis/diversity-inclusion", "method": "POST", "description": "Diversity & Inclusion Analysis"},
            {"path": "/api/analysis/absenteeism", "method": "POST", "description": "Absenteeism Analysis"},
            {"path": "/api/analysis/promotion-optimization", "method": "POST", "description": "Promotion Optimization"},
            {"path": "/api/analysis/dea-efficiency", "method": "POST", "description": "DEA Efficiency Analysis"},
        ],
        "optimization": [
            {"path": "/api/analysis/adagrad", "method": "POST", "description": "Adagrad Optimizer"},
        ],
        "survival_analysis": [
            {"path": "/api/analysis/survival", "method": "POST", "description": "Survival Analysis"},
            {"path": "/api/analysis/survival-curves", "method": "POST", "description": "Kaplan-Meier Survival Curves"},
            {"path": "/api/analysis/cox-regression", "method": "POST", "description": "Cox Proportional Hazards Regression"},
        ],
        "causal_inference": [
            {"path": "/api/analysis/did", "method": "POST", "description": "Difference-in-Differences"},
            {"path": "/api/analysis/psm", "method": "POST", "description": "Propensity Score Matching"},
            {"path": "/api/analysis/rdd", "method": "POST", "description": "Regression Discontinuity Design"},
            {"path": "/api/analysis/iv", "method": "POST", "description": "Instrumental Variables"},
        ],
        "econometrics": [
            {"path": "/api/analysis/var", "method": "POST", "description": "Vector Autoregression"},
            {"path": "/api/analysis/var-risk", "method": "POST", "description": "Value at Risk (VaR) Analysis"},
            {"path": "/api/analysis/gmm", "method": "POST", "description": "Generalized Method of Moments"},
            {"path": "/api/analysis/dsge", "method": "POST", "description": "DSGE Model"},
        ],
        "multivariate_analysis": [
            {"path": "/api/analysis/mds", "method": "POST", "description": "Multidimensional Scaling"},
            {"path": "/api/analysis/cross-validation", "method": "POST", "description": "Cross Validation"},
            {"path": "/api/analysis/reliability", "method": "POST", "description": "Reliability Analysis"},
            {"path": "/api/analysis/efa", "method": "POST", "description": "Exploratory Factor Analysis"},
            {"path": "/api/analysis/cfa", "method": "POST", "description": "Confirmatory Factor Analysis"},
            {"path": "/api/analysis/pca", "method": "POST", "description": "Principal Component Analysis"},
            {"path": "/api/analysis/mediation", "method": "POST", "description": "Mediation Analysis"},
            {"path": "/api/analysis/moderation", "method": "POST", "description": "Moderation Analysis"},
            {"path": "/api/analysis/sem", "method": "POST", "description": "Structural Equation Modeling"},
            {"path": "/api/analysis/sna", "method": "POST", "description": "Social Network Analysis"},
        ],
        "clustering": [
            {"path": "/api/analysis/kmeans", "method": "POST", "description": "K-Means Clustering"},
            {"path": "/api/analysis/kmedoids", "method": "POST", "description": "K-Medoids Clustering"},
            {"path": "/api/analysis/dbscan", "method": "POST", "description": "DBSCAN Clustering"},
            {"path": "/api/analysis/hdbscan", "method": "POST", "description": "HDBSCAN Clustering"},
            {"path": "/api/analysis/hca", "method": "POST", "description": "Hierarchical Cluster Analysis"},
        ],
        "time_series": [
            {"path": "/api/analysis/seasonal-analysis", "method": "POST", "description": "Seasonal Analysis"},
            {"path": "/api/analysis/rolling-statistics", "method": "POST", "description": "Rolling Statistics"},
            {"path": "/api/analysis/structural-break", "method": "POST", "description": "Structural Break Test"},
            {"path": "/api/analysis/change-point", "method": "POST", "description": "Change Point Detection"},
            {"path": "/api/analysis/acf-pacf", "method": "POST", "description": "ACF/PACF Analysis"},
            {"path": "/api/analysis/stationarity", "method": "POST", "description": "Stationarity Test"},
            {"path": "/api/analysis/ljung-box", "method": "POST", "description": "Ljung-Box Test"},
            {"path": "/api/analysis/arch-lm", "method": "POST", "description": "ARCH-LM Test"},
        ],
        "forecasting": [
            {"path": "/api/analysis/exponential-smoothing", "method": "POST", "description": "Exponential Smoothing"},
            {"path": "/api/analysis/arima", "method": "POST", "description": "ARIMA Model"},
            {"path": "/api/analysis/forecast-evaluation", "method": "POST", "description": "Forecast Evaluation"},
            {"path": "/api/analysis/demand-forecasting", "method": "POST", "description": "Demand Forecasting"},
            {"path": "/api/analysis/forecast-horizon", "method": "POST", "description": "Forecast Horizon Analysis"},
        ],
        "operations_research": [
            {"path": "/api/analysis/process-discovery", "method": "POST", "description": "Process Mining - Discovery"},
            {"path": "/api/analysis/conformance-checking", "method": "POST", "description": "Process Mining - Conformance"},
            {"path": "/api/analysis/performance-analysis", "method": "POST", "description": "Process Mining - Performance"},
            {"path": "/api/analysis/vrp", "method": "POST", "description": "Vehicle Routing Problem"},
            {"path": "/api/analysis/scheduling", "method": "POST", "description": "Job Shop Scheduling"},
            {"path": "/api/analysis/assignment", "method": "POST", "description": "Assignment Problem"},
            {"path": "/api/analysis/bin-packing", "method": "POST", "description": "Bin Packing Optimization"},
            {"path": "/api/analysis/tsp", "method": "POST", "description": "Traveling Salesman Problem"},
            {"path": "/api/analysis/knapsack", "method": "POST", "description": "Knapsack Problem"},
            {"path": "/api/analysis/funnel", "method": "POST", "description": "Funnel Analysis"},
            {"path": "/api/analysis/credit-risk", "method": "POST", "description": "CVA/DVA Credit Risk Analysis"},
            {"path": "/api/analysis/integer-programming", "method": "POST", "description": "Integer Programming (MILP)"},
            {"path": "/api/analysis/nonlinear-programming", "method": "POST", "description": "Non-linear Programming (NLP)"},

        ],
        "spatial_optimization": [
    {"path": "/api/analysis/p-median", "method": "POST", "description": "P-Median Facility Location"},
    {"path": "/api/analysis/mclp", "method": "POST", "description": "Maximal Covering Location Problem"},
    {"path": "/api/analysis/location-allocation", "method": "POST", "description": "Location-Allocation Model"},
    {"path": "/api/analysis/network-optimization", "method": "POST", "description": "Network Optimization"},
],
        "reliability_engineering": [
    {"path": "/api/analysis/weibull", "method": "POST", "description": "Weibull Analysis (2P/3P, censored data, MLE via reliability pkg)"},
],
    }


# ============================================
# MACRO DATA ENDPOINTS
# ============================================

class MacroFetchRequest(BaseModel):
    org_id: str = 'default_org'
    fred_series: List[str] = []
    fred_start: Optional[str] = None
    fred_end: Optional[str] = None
    bls_series: List[str] = []
    bls_start_year: Optional[str] = None
    bls_end_year: Optional[str] = None
    census_variables: List[str] = []
    census_year: Optional[str] = None
    census_geo: Optional[str] = 'us'


@app.post("/api/macro/fetch")
async def macro_fetch(req: MacroFetchRequest):
    import os as _os
    FRED_KEY   = _os.getenv('FRED_API_KEY', '')
    BLS_KEY    = _os.getenv('BLS_API_KEY',  '')
    CENSUS_KEY = _os.getenv('CENSUS_API_KEY', '')
    date_s = datetime.now().strftime('%Y-%m-%d')
    files = []

    async with aiohttp.ClientSession() as session:

        # ── FRED ──
        if req.fred_series:
            if not FRED_KEY:
                raise HTTPException(status_code=500, detail='FRED_API_KEY not configured on server')
            series_data = []
            for sid in req.fred_series:
                params = {'series_id': sid, 'api_key': FRED_KEY, 'file_type': 'json', 'sort_order': 'asc'}
                if req.fred_start: params['observation_start'] = req.fred_start
                if req.fred_end:   params['observation_end']   = req.fred_end
                async with session.get('https://api.stlouisfed.org/fred/series/observations', params=params) as r:
                    if not r.ok:
                        raise HTTPException(status_code=r.status, detail=f'FRED error for {sid}: HTTP {r.status}')
                    data = await r.json()
                if data.get('error_message'):
                    raise HTTPException(status_code=400, detail=f"FRED: {data['error_message']}")
                obs = [o for o in data.get('observations', []) if o['value'] != '.']
                series_data.append({'id': sid, 'obs': obs})

            all_dates = sorted(set(o['date'] for s in series_data for o in s['obs']))
            rows = [','.join(['date'] + [s['id'] for s in series_data])]
            for date in all_dates:
                vals = [next((o['value'] for o in s['obs'] if o['date'] == date), '') for s in series_data]
                rows.append(','.join([date] + vals))
            csv_str = '\n'.join(rows)
            files.append({
                'fileName': f'fred_macro_{date_s}.csv',
                'csv': csv_str,
                'rows': len(rows) - 1,
                'dataType': 'fred_macro',
                'description': f'FRED — {", ".join(req.fred_series)}',
                'columns': ['date'] + req.fred_series,
                'columnTypes': ['datetime'] + ['numeric'] * len(req.fred_series),
            })

        # ── BLS ──
        if req.bls_series:
            if not BLS_KEY:
                raise HTTPException(status_code=500, detail='BLS_API_KEY not configured on server')
            year = datetime.now().year
            payload = {
                'seriesid': req.bls_series,
                'startyear': req.bls_start_year or str(year - 5),
                'endyear':   req.bls_end_year   or str(year),
                'registrationkey': BLS_KEY,
            }
            async with session.post('https://api.bls.gov/publicAPI/v2/timeseries/data/', json=payload) as r:
                data = await r.json()
            if data.get('status') != 'REQUEST_SUCCEEDED':
                raise HTTPException(status_code=400, detail=f"BLS: {data.get('message', ['Unknown error'])}")

            bls_data = {}
            for s in data.get('Results', {}).get('series', []):
                bls_data[s['seriesID']] = sorted(
                    [{'date': f"{d['year']}-{d['period'].replace('M','').zfill(2)}-01", 'value': d['value']}
                     for d in s.get('data', []) if d['period'] != 'M13'],
                    key=lambda x: x['date']
                )
            all_dates = sorted(set(d['date'] for arr in bls_data.values() for d in arr))
            rows = [','.join(['date'] + req.bls_series)]
            for date in all_dates:
                vals = [next((d['value'] for d in bls_data.get(sid, []) if d['date'] == date), '') for sid in req.bls_series]
                rows.append(','.join([date] + vals))
            bls_csv = '\n'.join(rows)
            files.append({
                'fileName': f'bls_macro_{date_s}.csv',
                'csv': bls_csv,
                'rows': len(rows) - 1,
                'dataType': 'bls_macro',
                'description': f'BLS — {", ".join(req.bls_series)}',
                'columns': ['date'] + req.bls_series,
                'columnTypes': ['datetime'] + ['numeric'] * len(req.bls_series),
            })

        # ── Census ──
        if req.census_variables:
            if not CENSUS_KEY:
                raise HTTPException(status_code=500, detail='CENSUS_API_KEY not configured on server')
            year = req.census_year or str(datetime.now().year - 2)
            geo_param = (
                'for=state:*'         if req.census_geo == 'state'  else
                'for=county:*&in=state:*' if req.census_geo == 'county' else
                'for=us:1'
            )
            vars_str = ','.join(['NAME'] + req.census_variables)
            url = f'https://api.census.gov/data/{year}/acs/acs5?get={vars_str}&{geo_param}&key={CENSUS_KEY}'
            async with session.get(url) as r:
                if not r.ok:
                    raise HTTPException(status_code=r.status, detail=f'Census API: HTTP {r.status}')
                raw = await r.json()
            if len(raw) > 1:
                headers = raw[0]
                rows = [','.join(['name'] + req.census_variables + ['geo'])]
                for row in raw[1:]:
                    name = row[0]
                    vals = [row[headers.index(v)] if v in headers else '' for v in req.census_variables]
                    geo  = row[-1]
                    rows.append(','.join([f'"{name}"'] + vals + [geo]))
                census_csv = '\n'.join(rows)
                files.append({
                    'fileName': f'census_acs_{year}_{req.census_geo}_{date_s}.csv',
                    'csv': census_csv,
                    'rows': len(rows) - 1,
                    'dataType': 'census_acs',
                    'description': f'Census ACS {year} — {", ".join(req.census_variables)}',
                    'columns': ['name'] + req.census_variables + ['geo'],
                    'columnTypes': ['categorical'] + ['numeric'] * len(req.census_variables) + ['categorical'],
                })

    return {'success': True, 'files': files}


# ============================================
# SCHEDULER ENDPOINTS
# ============================================

class SchedulerTriggerRequest(BaseModel):
    job_id: str
    org_id: str = "default_org"

class SchedulerConfigRequest(BaseModel):
    job_id: str
    org_id: str = "default_org"
    enabled: bool
    frequency: str = "0 1 * * *"
    config: Dict[str, Any] = {}


@app.post("/api/scheduler/trigger")
async def trigger_scheduler(req: SchedulerTriggerRequest):
    """Manually trigger a sync job — called from UI Run Now button or Cloud Scheduler"""
    job_fn = {
        'market_data': _run_market_data,
        'macro_data':  _run_macro_data,
        'korea_stats': _run_korea_stats,
    }.get(req.job_id)

    if not job_fn:
        raise HTTPException(status_code=400, detail=f"Unknown job: {req.job_id}")

    try:
        await job_fn(req.org_id)
        return {"success": True, "jobId": req.job_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/scheduler/config")
async def save_scheduler_config(req: SchedulerConfigRequest):
    """Save schedule config and toggle from UI"""
    db = _get_db()
    db.collection('orgs').document(req.org_id)\
      .collection('schedules').document(req.job_id)\
      .set({
          'jobId': req.job_id,
          'enabled': req.enabled,
          'frequency': req.frequency,
          'config': req.config,
          'updatedAt': fs.SERVER_TIMESTAMP,
      }, merge=True)
    return {"success": True}


@app.get("/api/scheduler/status")
async def scheduler_status():
    """Return last sync status for all jobs"""
    db = _get_db()
    jobs = ['market_data', 'macro_data', 'korea_stats']
    result = {}
    for job_id in jobs:
        doc = db.collection('orgs').document('default_org')\
                .collection('schedules').document(job_id).get()
        result[job_id] = doc.to_dict() if doc.exists else {}
    return result


# ── Job implementations ───────────────────────────────────────────────────────

async def _run_market_data(org_id: str = 'default_org'):
    config = _get_schedule_config('market_data', org_id)
    if config is None:
        return

    tickers  = config.get('tickers', [])
    period   = config.get('period', '1mo')
    analysis = config.get('analysisTypes', ['basic_stats'])
    if not tickers:
        return

    date_s = datetime.now().strftime('%Y-%m-%d')
    db = _get_db()

    async with aiohttp.ClientSession() as session:
        # Yahoo Finance crumb
        async with session.get('https://fc.yahoo.com',
                               headers={'User-Agent': 'Mozilla/5.0'},
                               allow_redirects=False) as r:
            cookies = '; '.join(f"{k}={v.value}" for k, v in r.cookies.items())

        async with session.get(
            'https://query2.finance.yahoo.com/v1/test/getcrumb',
            headers={'User-Agent': 'Mozilla/5.0', 'Cookie': cookies}
        ) as r:
            crumb = await r.text()

        price_data = {}
        for ticker in tickers:
            try:
                url = (f'https://query2.finance.yahoo.com/v8/finance/chart/{ticker}'
                       f'?range={period}&interval=1d&crumb={crumb}')
                async with session.get(url, headers={'User-Agent': 'Mozilla/5.0', 'Cookie': cookies}) as r:
                    data = await r.json()
                result = (data.get('chart') or {}).get('result') or [None]
                result = result[0]
                if not result:
                    continue
                timestamps = result.get('timestamp', [])
                quote = result.get('indicators', {}).get('quote', [{}])[0]
                adj = (result.get('indicators', {}).get('adjclose', [{}])[0]
                       .get('adjclose', quote.get('close', [])))
                price_data[ticker] = {
                    'dates': [datetime.fromtimestamp(ts).strftime('%Y-%m-%d') for ts in timestamps],
                    'close': adj,
                }
            except Exception as e:
                print(f'[MarketData] {ticker} failed: {e}')

    if 'basic_stats' in analysis and price_data:
        rows = ['ticker,date,close,daily_return']
        for ticker, pd_ in price_data.items():
            for i, (d, c) in enumerate(zip(pd_['dates'], pd_['close'])):
                if c is None:
                    continue
                ret = 0.0
                if i > 0 and pd_['close'][i - 1]:
                    ret = (c - pd_['close'][i - 1]) / pd_['close'][i - 1] * 100
                rows.append(f"{ticker},{d},{c:.2f},{ret:.4f}")
        csv = '\n'.join(rows)
        file_id = f"scheduled_market_{org_id}_{int(datetime.now().timestamp())}"
        _save_csv(db, file_id, f'market_data_{date_s}.csv', csv, 'basic_stats',
                  f"Scheduled market data — {', '.join(tickers)} ({period})",
                  ['ticker', 'date', 'close', 'daily_return'],
                  ['categorical', 'datetime', 'numeric', 'numeric'],
                  'yahoo_finance', org_id)

    _update_status('market_data', 'success', org_id=org_id)


async def _run_macro_data(org_id: str = 'default_org'):
    config = _get_schedule_config('macro_data', org_id)
    if config is None:
        return

    fred_api_key = os.getenv('FRED_API_KEY', config.get('fredApiKey', ''))
    bls_api_key  = os.getenv('BLS_API_KEY',  config.get('blsApiKey',  ''))
    fred_series  = config.get('fredSeriesIds', [])
    bls_series   = config.get('blsSeriesIds',  [])
    date_s = datetime.now().strftime('%Y-%m-%d')
    db = _get_db()

    async with aiohttp.ClientSession() as session:
        # FRED
        if fred_api_key and fred_series:
            series_data = []
            for sid in fred_series:
                params = {
                    'series_id': sid, 'api_key': fred_api_key, 'file_type': 'json',
                    'sort_order': 'asc',
                    'observation_start': f'{datetime.now().year - 1}-01-01',
                }
                async with session.get('https://api.stlouisfed.org/fred/series/observations',
                                       params=params) as r:
                    data = await r.json()
                obs = [o for o in data.get('observations', []) if o['value'] != '.']
                series_data.append({'id': sid, 'obs': obs})

            all_dates = sorted(set(o['date'] for s in series_data for o in s['obs']))
            rows = [','.join(['date'] + [s['id'] for s in series_data])]
            for date in all_dates:
                vals = [next((o['value'] for o in s['obs'] if o['date'] == date), '') for s in series_data]
                rows.append(','.join([date] + vals))
            csv = '\n'.join(rows)
            file_id = f"scheduled_fred_{org_id}_{int(datetime.now().timestamp())}"
            _save_csv(db, file_id, f'fred_macro_{date_s}.csv', csv, 'fred_macro',
                      f"Scheduled FRED sync — {', '.join(fred_series)}",
                      ['date'] + fred_series,
                      ['datetime'] + ['numeric'] * len(fred_series),
                      'macro_api', org_id)

        # BLS
        if bls_api_key and bls_series:
            year = datetime.now().year
            async with session.post('https://api.bls.gov/publicAPI/v2/timeseries/data/',
                                    json={'seriesid': bls_series,
                                          'startyear': str(year - 1),
                                          'endyear': str(year),
                                          'registrationkey': bls_api_key}) as r:
                data = await r.json()
            bls_data = {}
            for s in data.get('Results', {}).get('series', []):
                bls_data[s['seriesID']] = sorted(
                    [{'date': f"{d['year']}-{d['period'].replace('M','').zfill(2)}-01",
                      'value': d['value']}
                     for d in s.get('data', []) if d['period'] != 'M13'],
                    key=lambda x: x['date'])
            all_dates = sorted(set(d['date'] for arr in bls_data.values() for d in arr))
            rows = [','.join(['date'] + bls_series)]
            for date in all_dates:
                vals = [next((d['value'] for d in bls_data.get(sid, []) if d['date'] == date), '')
                        for sid in bls_series]
                rows.append(','.join([date] + vals))
            csv = '\n'.join(rows)
            file_id = f"scheduled_bls_{org_id}_{int(datetime.now().timestamp())}"
            _save_csv(db, file_id, f'bls_macro_{date_s}.csv', csv, 'bls_macro',
                      f"Scheduled BLS sync — {', '.join(bls_series)}",
                      ['date'] + bls_series,
                      ['datetime'] + ['numeric'] * len(bls_series),
                      'macro_api', org_id)

    _update_status('macro_data', 'success', org_id=org_id)


async def _run_korea_stats(org_id: str = 'default_org'):
    config = _get_schedule_config('korea_stats', org_id)
    if config is None:
        return

    kosis_api_key = os.getenv('KOSIS_API_KEY', '')
    ecos_api_key  = os.getenv('ECOS_API_KEY',  '')
    kosis_items   = config.get('kosisItems', [])
    ecos_items    = config.get('ecosItems',  [])
    date_s = datetime.now().strftime('%Y-%m-%d')
    db = _get_db()

    async with aiohttp.ClientSession() as session:
        # KOSIS
        if kosis_api_key and kosis_items:
            for item in kosis_items:
                params = {
                    'method': 'getList', 'apiKey': kosis_api_key,
                    'format': 'json', 'jsonVD': 'Y',
                    'orgId': item['orgId'], 'tblId': item['tblId'],
                    'itmId': item.get('itmId', 'ALL'),
                    'objL1': item.get('objL1', 'ALL'),
                    'prdSe': item.get('prdSe', 'Y'),
                    'newEstPrdCnt': '12',
                }
                async with session.get('https://kosis.kr/openapi/statisticsData.do',
                                       params=params) as r:
                    data = await r.json()
                if not isinstance(data, list):
                    continue
                rows = ['period,item,value,unit']
                for row in data:
                    rows.append(f"{row.get('PRD_DE','')},\"{row.get('ITM_NM','')}\","
                                f"{row.get('DT','')},\"{row.get('UNIT_NM','')}\"")
                csv = '\n'.join(rows)
                label = item.get('label', item['tblId'])
                file_id = f"scheduled_kosis_{item['tblId']}_{org_id}_{int(datetime.now().timestamp())}"
                _save_csv(db, file_id, f"kosis_{label}_{date_s}.csv", csv,
                          f"kosis_{item['tblId']}",
                          f"Scheduled KOSIS — {label}",
                          ['period', 'item', 'value', 'unit'],
                          ['datetime', 'categorical', 'numeric', 'categorical'],
                          'korea_stats', org_id)

        # ECOS
        if ecos_api_key and ecos_items:
            year = datetime.now().year
            for item in ecos_items:
                cycle = item.get('cycle', 'M')
                start = str(year - 1) if cycle == 'A' else f"{year - 1}01"
                end   = str(year)     if cycle == 'A' else f"{year}12"
                url   = (f"https://ecos.bok.or.kr/api/StatisticSearch/"
                         f"{ecos_api_key}/json/kr/1/10000/"
                         f"{item['statCode']}/{cycle}/{start}/{end}/*/*/*")
                async with session.get(url) as r:
                    data = await r.json()
                ecos_rows = data.get('StatisticSearch', {}).get('row', [])
                if not ecos_rows:
                    continue
                rows = ['period,item,value,unit']
                for row in ecos_rows:
                    rows.append(f"{row.get('TIME','')},\"{row.get('ITEM_NAME1','')}\","
                                f"{row.get('DATA_VALUE','')},\"{row.get('UNIT_NAME','')}\"")
                csv = '\n'.join(rows)
                label = item.get('label', item['statCode'])
                file_id = f"scheduled_ecos_{item['statCode']}_{org_id}_{int(datetime.now().timestamp())}"
                _save_csv(db, file_id, f"ecos_{label}_{date_s}.csv", csv,
                          f"ecos_{item['statCode']}",
                          f"Scheduled ECOS — {label}",
                          ['period', 'item', 'value', 'unit'],
                          ['datetime', 'categorical', 'numeric', 'categorical'],
                          'korea_stats', org_id)

    _update_status('korea_stats', 'success', org_id=org_id)
