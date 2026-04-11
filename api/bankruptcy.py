from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from scipy import stats
import warnings
import traceback

warnings.filterwarnings('ignore')

router = APIRouter()


# ══════════════════════════════════════════════════════════════
# Request Model
# ══════════════════════════════════════════════════════════════

class BankruptcyRequest(BaseModel):
    data: Optional[List[Dict[str, Any]]] = None
    # Generate mode
    generate: bool = False
    nCompanies: int = 200
    seed: Optional[int] = None
    # Model config
    testSize: float = 0.25
    xgbEstimators: int = 100
    xgbMaxDepth: int = 5


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

def _to_native(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return [_to_native(x) for x in obj.tolist()]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_native(x) for x in obj]
    return obj


def safe_float(val, default=0.0):
    try:
        if val is None:
            return default
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return default
        return f
    except Exception:
        return default


# ══════════════════════════════════════════════════════════════
# Financial Ratio Data Generator
# ══════════════════════════════════════════════════════════════

INDUSTRIES = ['Technology', 'Manufacturing', 'Retail', 'Healthcare', 'Energy', 'Financial', 'Consumer']

def generate_bankruptcy_data(
    n_companies: int = 200,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate realistic financial ratio data for bankruptcy prediction.

    Features based on classic Altman Z-Score components + additional ratios:
    - X1: Working Capital / Total Assets
    - X2: Retained Earnings / Total Assets
    - X3: EBIT / Total Assets
    - X4: Market Value Equity / Total Liabilities
    - X5: Sales / Total Assets
    - Additional: Current Ratio, Debt/Equity, ROA, Cash Flow/Debt, Interest Coverage

    ~25-30% bankruptcy rate (imbalanced like real data).
    """
    rng = np.random.default_rng(seed)

    bankrupt_ratio = 0.28
    n_bankrupt = int(n_companies * bankrupt_ratio)
    n_healthy = n_companies - n_bankrupt

    records = []

    for i in range(n_companies):
        is_bankrupt = i < n_bankrupt
        industry = rng.choice(INDUSTRIES)

        if is_bankrupt:
            # Distressed company — poor financial ratios
            wc_ta = rng.normal(-0.05, 0.15)           # Working Capital / Total Assets
            re_ta = rng.normal(-0.10, 0.20)            # Retained Earnings / Total Assets
            ebit_ta = rng.normal(-0.02, 0.08)          # EBIT / Total Assets
            mve_tl = rng.normal(0.4, 0.3)              # Market Value Equity / Total Liabilities
            sales_ta = rng.normal(0.8, 0.4)            # Sales / Total Assets
            current_ratio = rng.normal(0.8, 0.3)       # Current Ratio
            debt_equity = rng.normal(3.5, 2.0)         # Debt / Equity
            roa = rng.normal(-0.05, 0.08)              # Return on Assets
            cf_debt = rng.normal(0.02, 0.08)           # Cash Flow / Total Debt
            interest_cov = rng.normal(0.5, 1.0)        # Interest Coverage
            net_margin = rng.normal(-0.08, 0.10)       # Net Profit Margin
            asset_turnover = rng.normal(0.6, 0.3)      # Asset Turnover
            log_assets = rng.normal(5.5, 1.2)          # Log(Total Assets) — size proxy
        else:
            # Healthy company — strong financial ratios
            wc_ta = rng.normal(0.25, 0.15)
            re_ta = rng.normal(0.30, 0.15)
            ebit_ta = rng.normal(0.10, 0.06)
            mve_tl = rng.normal(2.5, 1.5)
            sales_ta = rng.normal(1.2, 0.5)
            current_ratio = rng.normal(2.0, 0.6)
            debt_equity = rng.normal(0.8, 0.5)
            roa = rng.normal(0.08, 0.05)
            cf_debt = rng.normal(0.20, 0.10)
            interest_cov = rng.normal(5.0, 3.0)
            net_margin = rng.normal(0.08, 0.06)
            asset_turnover = rng.normal(1.0, 0.4)
            log_assets = rng.normal(7.0, 1.5)

        records.append({
            'company_id': f'C{str(i + 1).zfill(4)}',
            'industry': industry,
            'wc_ta': round(wc_ta, 4),
            're_ta': round(re_ta, 4),
            'ebit_ta': round(ebit_ta, 4),
            'mve_tl': round(max(mve_tl, 0.01), 4),
            'sales_ta': round(max(sales_ta, 0.05), 4),
            'current_ratio': round(max(current_ratio, 0.1), 4),
            'debt_equity': round(max(debt_equity, 0.01), 4),
            'roa': round(roa, 4),
            'cf_debt': round(cf_debt, 4),
            'interest_coverage': round(interest_cov, 4),
            'net_margin': round(net_margin, 4),
            'asset_turnover': round(max(asset_turnover, 0.05), 4),
            'log_total_assets': round(log_assets, 4),
            'bankrupt': 1 if is_bankrupt else 0,
        })

    rng.shuffle(records)
    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════
# Altman Z-Score
# ══════════════════════════════════════════════════════════════

def compute_altman_zscore(df: pd.DataFrame) -> pd.Series:
    """
    Altman Z-Score = 1.2×X1 + 1.4×X2 + 3.3×X3 + 0.6×X4 + 1.0×X5

    Interpretation:
        Z > 2.99 → Safe zone
        1.81 < Z < 2.99 → Grey zone
        Z < 1.81 → Distress zone
    """
    z = (1.2 * df['wc_ta'] +
         1.4 * df['re_ta'] +
         3.3 * df['ebit_ta'] +
         0.6 * df['mve_tl'] +
         1.0 * df['sales_ta'])
    return z


def zscore_zone(z: float) -> str:
    if z > 2.99:
        return 'Safe'
    elif z > 1.81:
        return 'Grey'
    else:
        return 'Distress'


# ══════════════════════════════════════════════════════════════
# ML Models — Logistic Regression + XGBoost
# ══════════════════════════════════════════════════════════════

FEATURE_COLS = [
    'wc_ta', 're_ta', 'ebit_ta', 'mve_tl', 'sales_ta',
    'current_ratio', 'debt_equity', 'roa', 'cf_debt',
    'interest_coverage', 'net_margin', 'asset_turnover', 'log_total_assets',
]


def train_models(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'bankrupt',
    test_size: float = 0.25,
    xgb_n_estimators: int = 100,
    xgb_max_depth: int = 5,
):
    """
    Train Logistic Regression and XGBoost classifiers.
    Returns metrics, feature importance, predictions, ROC data.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, roc_curve, confusion_matrix, classification_report,
    )
    from xgboost import XGBClassifier

    X = df[feature_cols].values
    y = df[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y,
    )

    # Scale for logistic regression
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # ── Logistic Regression ──
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr.fit(X_train_scaled, y_train)
    lr_proba = lr.predict_proba(X_test_scaled)[:, 1]
    lr_pred = lr.predict(X_test_scaled)

    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_proba)
    lr_cm = confusion_matrix(y_test, lr_pred)

    results['logistic'] = {
        'accuracy': safe_float(accuracy_score(y_test, lr_pred)),
        'precision': safe_float(precision_score(y_test, lr_pred, zero_division=0)),
        'recall': safe_float(recall_score(y_test, lr_pred, zero_division=0)),
        'f1': safe_float(f1_score(y_test, lr_pred, zero_division=0)),
        'auc': safe_float(roc_auc_score(y_test, lr_proba)),
        'roc_curve': [{'fpr': safe_float(f), 'tpr': safe_float(t)} for f, t in zip(lr_fpr, lr_tpr)],
        'confusion_matrix': lr_cm.tolist(),
        'coefficients': {feature_cols[i]: safe_float(lr.coef_[0][i]) for i in range(len(feature_cols))},
    }

    # ── XGBoost ──
    xgb = XGBClassifier(
        n_estimators=xgb_n_estimators,
        max_depth=xgb_max_depth,
        learning_rate=0.1,
        random_state=42,
        scale_pos_weight=len(y_train[y_train == 0]) / max(len(y_train[y_train == 1]), 1),
        use_label_encoder=False,
        eval_metric='logloss',
    )
    xgb.fit(X_train, y_train)
    xgb_proba = xgb.predict_proba(X_test)[:, 1]
    xgb_pred = xgb.predict(X_test)

    xgb_fpr, xgb_tpr, _ = roc_curve(y_test, xgb_proba)
    xgb_cm = confusion_matrix(y_test, xgb_pred)

    # Feature importance
    importance = xgb.feature_importances_
    feat_imp = sorted(
        [{'feature': feature_cols[i], 'importance': safe_float(importance[i])}
         for i in range(len(feature_cols))],
        key=lambda x: -x['importance'],
    )

    results['xgboost'] = {
        'accuracy': safe_float(accuracy_score(y_test, xgb_pred)),
        'precision': safe_float(precision_score(y_test, xgb_pred, zero_division=0)),
        'recall': safe_float(recall_score(y_test, xgb_pred, zero_division=0)),
        'f1': safe_float(f1_score(y_test, xgb_pred, zero_division=0)),
        'auc': safe_float(roc_auc_score(y_test, xgb_proba)),
        'roc_curve': [{'fpr': safe_float(f), 'tpr': safe_float(t)} for f, t in zip(xgb_fpr, xgb_tpr)],
        'confusion_matrix': xgb_cm.tolist(),
        'feature_importance': feat_imp,
        'n_estimators': xgb_n_estimators,
        'max_depth': xgb_max_depth,
    }

    # ── Full dataset predictions (XGBoost) for company-level output ──
    X_all_scaled = scaler.transform(X)
    all_proba_xgb = xgb.predict_proba(X)[:, 1]
    all_proba_lr = lr.predict_proba(X_all_scaled)[:, 1]

    results['predictions'] = {
        'xgb_proba': all_proba_xgb.tolist(),
        'lr_proba': all_proba_lr.tolist(),
    }

    results['data_split'] = {
        'train_size': len(X_train),
        'test_size': len(X_test),
        'train_bankrupt_pct': safe_float(y_train.mean() * 100),
        'test_bankrupt_pct': safe_float(y_test.mean() * 100),
    }

    return results


# ══════════════════════════════════════════════════════════════
# Main Endpoint
# ══════════════════════════════════════════════════════════════

@router.post("/bankruptcy")
async def bankruptcy_analysis(request: BankruptcyRequest):
    try:
        # ── 1. Get Data ──
        if request.generate or not request.data:
            df = generate_bankruptcy_data(
                n_companies=request.nCompanies,
                seed=request.seed,
            )
        else:
            df = pd.DataFrame(request.data)
            for col in FEATURE_COLS:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna(subset=[c for c in FEATURE_COLS if c in df.columns])

        n = len(df)
        if n < 30:
            raise HTTPException(status_code=400, detail=f"Need at least 30 companies, got {n}")

        available_features = [c for c in FEATURE_COLS if c in df.columns]
        if len(available_features) < 3:
            raise HTTPException(status_code=400, detail=f"Need at least 3 feature columns. Found: {available_features}")

        # ── 2. Altman Z-Score ──
        has_zscore_cols = all(c in df.columns for c in ['wc_ta', 're_ta', 'ebit_ta', 'mve_tl', 'sales_ta'])
        if has_zscore_cols:
            df['z_score'] = compute_altman_zscore(df)
            df['z_zone'] = df['z_score'].apply(zscore_zone)

        # ── 3. Train ML Models ──
        ml_results = train_models(
            df=df,
            feature_cols=available_features,
            target_col='bankrupt',
            test_size=request.testSize,
            xgb_n_estimators=request.xgbEstimators,
            xgb_max_depth=request.xgbMaxDepth,
        )

        # ── 4. Company-Level Results ──
        company_data = []
        for i in range(n):
            row = df.iloc[i]
            entry = {
                'company_id': str(row.get('company_id', f'C{i+1}')),
                'industry': str(row.get('industry', 'Unknown')),
                'actual': int(row['bankrupt']),
                'xgb_prob': safe_float(ml_results['predictions']['xgb_proba'][i]),
                'lr_prob': safe_float(ml_results['predictions']['lr_proba'][i]),
                'xgb_alert': ml_results['predictions']['xgb_proba'][i] > 0.5,
            }
            if has_zscore_cols:
                entry['z_score'] = safe_float(row['z_score'])
                entry['z_zone'] = row['z_zone']
            company_data.append(entry)

        # Sort by XGB probability (highest risk first)
        company_data.sort(key=lambda x: -x['xgb_prob'])

        # ── 5. Chart Data ──

        # ROC curves — subsample for chart
        roc_lr = ml_results['logistic']['roc_curve']
        roc_xgb = ml_results['xgboost']['roc_curve']
        # Merge into single chart
        max_pts = 100
        step_lr = max(1, len(roc_lr) // max_pts)
        step_xgb = max(1, len(roc_xgb) // max_pts)
        roc_chart = []
        for i in range(0, len(roc_lr), step_lr):
            roc_chart.append({'fpr': roc_lr[i]['fpr'], 'lr_tpr': roc_lr[i]['tpr']})
        roc_xgb_sub = [roc_xgb[i] for i in range(0, len(roc_xgb), step_xgb)]

        # Feature importance chart
        feat_imp_chart = ml_results['xgboost']['feature_importance']

        # Z-Score distribution chart
        zscore_chart = []
        if has_zscore_cols:
            for i in range(n):
                row = df.iloc[i]
                zscore_chart.append({
                    'company_id': str(row.get('company_id', '')),
                    'z_score': safe_float(row['z_score']),
                    'bankrupt': int(row['bankrupt']),
                    'zone': row['z_zone'],
                })

        # Probability distribution chart
        prob_dist = []
        bins = np.linspace(0, 1, 21)
        for j in range(len(bins) - 1):
            lo, hi = bins[j], bins[j + 1]
            mask = (np.array(ml_results['predictions']['xgb_proba']) >= lo) & (np.array(ml_results['predictions']['xgb_proba']) < hi)
            actual = np.array([c['actual'] for c in company_data])
            prob_dist.append({
                'range': f'{lo:.0%}-{hi:.0%}',
                'count': int(mask.sum()),
                'bankrupt_count': int((mask & (df['bankrupt'].values == 1)).sum()),
                'healthy_count': int((mask & (df['bankrupt'].values == 0)).sum()),
            })

        # Model comparison
        model_comparison = [
            {'metric': 'Accuracy', 'logistic': ml_results['logistic']['accuracy'], 'xgboost': ml_results['xgboost']['accuracy']},
            {'metric': 'Precision', 'logistic': ml_results['logistic']['precision'], 'xgboost': ml_results['xgboost']['precision']},
            {'metric': 'Recall', 'logistic': ml_results['logistic']['recall'], 'xgboost': ml_results['xgboost']['recall']},
            {'metric': 'F1 Score', 'logistic': ml_results['logistic']['f1'], 'xgboost': ml_results['xgboost']['f1']},
            {'metric': 'AUC-ROC', 'logistic': ml_results['logistic']['auc'], 'xgboost': ml_results['xgboost']['auc']},
        ]

        # Logistic regression coefficients chart
        lr_coef_chart = sorted(
            [{'feature': k, 'coefficient': v} for k, v in ml_results['logistic']['coefficients'].items()],
            key=lambda x: abs(x['coefficient']),
            reverse=True,
        )

        # ── 6. Summary Statistics ──
        n_bankrupt = int(df['bankrupt'].sum())
        n_healthy = n - n_bankrupt

        alert_counts = {
            'high_risk': len([c for c in company_data if c['xgb_prob'] > 0.7]),
            'medium_risk': len([c for c in company_data if 0.3 < c['xgb_prob'] <= 0.7]),
            'low_risk': len([c for c in company_data if c['xgb_prob'] <= 0.3]),
        }

        zscore_summary = None
        if has_zscore_cols:
            zscore_summary = {
                'safe': int((df['z_zone'] == 'Safe').sum()),
                'grey': int((df['z_zone'] == 'Grey').sum()),
                'distress': int((df['z_zone'] == 'Distress').sum()),
                'mean': safe_float(df['z_score'].mean()),
                'median': safe_float(df['z_score'].median()),
            }

        # ── 7. Build Response ──
        results = {
            'summary': {
                'n_companies': n,
                'n_bankrupt': n_bankrupt,
                'n_healthy': n_healthy,
                'bankrupt_pct': safe_float(n_bankrupt / n * 100),
            },
            'alert_counts': alert_counts,
            'zscore_summary': zscore_summary,
            'logistic': {
                'accuracy': ml_results['logistic']['accuracy'],
                'precision': ml_results['logistic']['precision'],
                'recall': ml_results['logistic']['recall'],
                'f1': ml_results['logistic']['f1'],
                'auc': ml_results['logistic']['auc'],
                'confusion_matrix': ml_results['logistic']['confusion_matrix'],
            },
            'xgboost': {
                'accuracy': ml_results['xgboost']['accuracy'],
                'precision': ml_results['xgboost']['precision'],
                'recall': ml_results['xgboost']['recall'],
                'f1': ml_results['xgboost']['f1'],
                'auc': ml_results['xgboost']['auc'],
                'confusion_matrix': ml_results['xgboost']['confusion_matrix'],
                'n_estimators': ml_results['xgboost']['n_estimators'],
                'max_depth': ml_results['xgboost']['max_depth'],
            },
            'data_split': ml_results['data_split'],
            'company_data': company_data[:100],  # top 100 by risk
            'charts': {
                'feature_importance': feat_imp_chart,
                'model_comparison': model_comparison,
                'lr_coefficients': lr_coef_chart,
                'roc_lr': [roc_lr[i] for i in range(0, len(roc_lr), step_lr)],
                'roc_xgb': roc_xgb_sub,
                'zscore_chart': zscore_chart[:200],
                'prob_distribution': prob_dist,
            },
        }

        return _to_native({'results': results})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")
