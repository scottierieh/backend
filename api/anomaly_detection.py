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

class AnomalyRequest(BaseModel):
    data: Optional[List[Dict[str, Any]]] = None
    generate: bool = False
    nSamples: int = 1000
    nFeatures: int = 8
    anomalyRatio: float = 0.05
    seed: Optional[int] = None
    # Autoencoder config
    encodingDim: int = 4
    epochs: int = 50
    batchSize: int = 32
    # Threshold
    thresholdPct: float = 95.0  # percentile for anomaly threshold
    # Isolation Forest
    iforestContamination: float = 0.05


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
# Data Generator
# ══════════════════════════════════════════════════════════════

FEATURE_NAMES = [
    'cpu_usage', 'memory_pct', 'disk_io', 'network_in',
    'network_out', 'latency_ms', 'error_rate', 'request_count',
    'cache_hit_ratio', 'thread_count', 'gc_pause_ms', 'queue_depth',
]


def generate_anomaly_data(
    n_samples: int = 1000,
    n_features: int = 8,
    anomaly_ratio: float = 0.05,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Generate multivariate system metrics data with injected anomalies.
    Normal: correlated Gaussian clusters.
    Anomalies: point anomalies, contextual shifts, collective bursts.
    """
    rng = np.random.default_rng(seed)
    n_features = min(n_features, len(FEATURE_NAMES))
    features = FEATURE_NAMES[:n_features]

    n_anomaly = int(n_samples * anomaly_ratio)
    n_normal = n_samples - n_anomaly

    # Normal data — correlated multivariate Gaussian
    mean_normal = rng.uniform(30, 70, size=n_features)
    # Build correlation structure
    A = rng.uniform(0.3, 0.7, size=(n_features, n_features))
    cov = A @ A.T + np.eye(n_features) * 2.0
    X_normal = rng.multivariate_normal(mean_normal, cov, size=n_normal)
    X_normal = np.clip(X_normal, 0, 100)

    # Anomaly data — various types
    anomalies = []
    types = []
    for i in range(n_anomaly):
        atype = rng.choice(['point', 'shift', 'spike'])
        if atype == 'point':
            # Random extreme values
            row = mean_normal + rng.uniform(3, 6, size=n_features) * np.sqrt(np.diag(cov)) * rng.choice([-1, 1], size=n_features)
            row = np.clip(row, 0, 100)
        elif atype == 'shift':
            # Shifted mean
            row = rng.multivariate_normal(mean_normal + rng.uniform(15, 30, size=n_features), cov * 0.5)
            row = np.clip(row, 0, 100)
        else:  # spike
            # One or two features spike
            row = rng.multivariate_normal(mean_normal, cov)
            spike_idx = rng.choice(n_features, size=rng.integers(1, 3), replace=False)
            row[spike_idx] = rng.uniform(90, 100, size=len(spike_idx))
            row = np.clip(row, 0, 100)
        anomalies.append(row)
        types.append(atype)

    X_anomaly = np.array(anomalies) if anomalies else np.empty((0, n_features))

    # Combine
    X = np.vstack([X_normal, X_anomaly])
    labels = np.array([0] * n_normal + [1] * n_anomaly)
    anomaly_types = ['normal'] * n_normal + types

    # Shuffle
    idx = rng.permutation(n_samples)
    X = X[idx]
    labels = labels[idx]
    anomaly_types = [anomaly_types[i] for i in idx]

    # Timestamps
    end = pd.Timestamp('2025-04-30 23:59:00')
    timestamps = pd.date_range(end=end, periods=n_samples, freq='1min')

    df = pd.DataFrame(X, columns=features)
    df.insert(0, 'timestamp', timestamps.strftime('%Y-%m-%d %H:%M'))
    df['_label'] = labels
    df['_anomaly_type'] = anomaly_types

    return df, features


# ══════════════════════════════════════════════════════════════
# Autoencoder Model (TensorFlow/Keras)
# ══════════════════════════════════════════════════════════════

def train_autoencoder(
    X_train: np.ndarray,
    X_all: np.ndarray,
    encoding_dim: int = 4,
    epochs: int = 50,
    batch_size: int = 32,
) -> Dict[str, Any]:
    """
    Train an autoencoder for anomaly detection.
    Architecture: Input → Dense(encoding_dim*2) → Dense(encoding_dim) → Dense(encoding_dim*2) → Output
    Anomalies have high reconstruction error.
    """
    import tensorflow as tf
    tf.get_logger().setLevel('ERROR')
    from tensorflow import keras

    input_dim = X_train.shape[1]
    mid_dim = max(encoding_dim * 2, input_dim // 2)

    # Build model
    encoder_input = keras.Input(shape=(input_dim,))
    x = keras.layers.Dense(mid_dim, activation='relu')(encoder_input)
    x = keras.layers.Dropout(0.2)(x)
    encoded = keras.layers.Dense(encoding_dim, activation='relu', name='bottleneck')(x)
    x = keras.layers.Dense(mid_dim, activation='relu')(encoded)
    x = keras.layers.Dropout(0.2)(x)
    decoded = keras.layers.Dense(input_dim, activation='linear')(x)

    autoencoder = keras.Model(encoder_input, decoded)
    encoder = keras.Model(encoder_input, encoded)

    autoencoder.compile(optimizer='adam', loss='mse')

    history = autoencoder.fit(
        X_train, X_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.15,
        verbose=0,
        shuffle=True,
    )

    # Reconstruction on full dataset
    X_pred = autoencoder.predict(X_all, verbose=0)
    recon_errors = np.mean((X_all - X_pred) ** 2, axis=1)

    # Encoded representations
    encoded_repr = encoder.predict(X_all, verbose=0)

    # Training history
    loss_history = []
    for i in range(len(history.history['loss'])):
        entry = {'epoch': i + 1, 'train_loss': safe_float(history.history['loss'][i])}
        if 'val_loss' in history.history:
            entry['val_loss'] = safe_float(history.history['val_loss'][i])
        loss_history.append(entry)

    return {
        'recon_errors': recon_errors,
        'encoded': encoded_repr,
        'loss_history': loss_history,
        'final_train_loss': safe_float(history.history['loss'][-1]),
        'final_val_loss': safe_float(history.history.get('val_loss', [0])[-1]),
        'architecture': f'{input_dim} → {mid_dim} → {encoding_dim} → {mid_dim} → {input_dim}',
    }


# ══════════════════════════════════════════════════════════════
# Isolation Forest
# ══════════════════════════════════════════════════════════════

def run_isolation_forest(
    X: np.ndarray,
    contamination: float = 0.05,
) -> Dict[str, Any]:
    from sklearn.ensemble import IsolationForest

    iso = IsolationForest(
        contamination=contamination,
        random_state=42,
        n_estimators=100,
    )
    iso.fit(X)
    scores = iso.decision_function(X)  # lower = more anomalous
    predictions = iso.predict(X)  # -1 = anomaly, 1 = normal

    return {
        'scores': scores,
        'predictions': (predictions == -1).astype(int),  # 1 = anomaly
    }


# ══════════════════════════════════════════════════════════════
# Main Endpoint
# ══════════════════════════════════════════════════════════════

@router.post("/anomaly-detection")
async def anomaly_detection(request: AnomalyRequest):
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, f1_score,
            roc_auc_score, roc_curve, confusion_matrix,
        )

        # ── 1. Get Data ──
        features = None
        labels = None

        if request.generate or not request.data:
            df, features = generate_anomaly_data(
                n_samples=request.nSamples,
                n_features=request.nFeatures,
                anomaly_ratio=request.anomalyRatio,
                seed=request.seed,
            )
            labels = df['_label'].values
        else:
            df = pd.DataFrame(request.data)
            features = [c for c in df.columns if c not in ('timestamp', '_label', '_anomaly_type') and not c.startswith('_')]
            for f in features:
                df[f] = pd.to_numeric(df[f], errors='coerce')
            df = df.dropna(subset=features)
            if '_label' in df.columns:
                labels = df['_label'].values.astype(int)

        n = len(df)
        n_features = len(features)
        if n < 30:
            raise HTTPException(status_code=400, detail=f"Need at least 30 samples, got {n}")

        X = df[features].values.astype(np.float64)
        has_labels = labels is not None and len(labels) == n

        # ── 2. Scale ──
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Use "normal" subset for training if labels exist
        if has_labels:
            X_train = X_scaled[labels == 0]
        else:
            X_train = X_scaled  # unsupervised — use all

        # ── 3. Autoencoder ──
        ae_results = train_autoencoder(
            X_train=X_train,
            X_all=X_scaled,
            encoding_dim=min(request.encodingDim, n_features - 1),
            epochs=request.epochs,
            batch_size=request.batchSize,
        )

        recon_errors = ae_results['recon_errors']
        threshold = float(np.percentile(recon_errors, request.thresholdPct))
        ae_predictions = (recon_errors > threshold).astype(int)

        # ── 4. Isolation Forest ──
        iforest = run_isolation_forest(X_scaled, contamination=request.iforestContamination)

        # ── 5. Metrics (if labels available) ──
        ae_metrics = {}
        iforest_metrics = {}
        ae_roc = []
        iforest_roc = []

        if has_labels and labels.sum() > 0:
            # Autoencoder
            ae_metrics = {
                'accuracy': safe_float(accuracy_score(labels, ae_predictions)),
                'precision': safe_float(precision_score(labels, ae_predictions, zero_division=0)),
                'recall': safe_float(recall_score(labels, ae_predictions, zero_division=0)),
                'f1': safe_float(f1_score(labels, ae_predictions, zero_division=0)),
                'auc': safe_float(roc_auc_score(labels, recon_errors)),
            }
            fpr, tpr, _ = roc_curve(labels, recon_errors)
            step = max(1, len(fpr) // 100)
            ae_roc = [{'fpr': safe_float(fpr[i]), 'tpr': safe_float(tpr[i])} for i in range(0, len(fpr), step)]

            ae_cm = confusion_matrix(labels, ae_predictions).tolist()
            ae_metrics['confusion_matrix'] = ae_cm

            # Isolation Forest
            iforest_metrics = {
                'accuracy': safe_float(accuracy_score(labels, iforest['predictions'])),
                'precision': safe_float(precision_score(labels, iforest['predictions'], zero_division=0)),
                'recall': safe_float(recall_score(labels, iforest['predictions'], zero_division=0)),
                'f1': safe_float(f1_score(labels, iforest['predictions'], zero_division=0)),
                'auc': safe_float(roc_auc_score(labels, -iforest['scores'])),
            }
            fpr2, tpr2, _ = roc_curve(labels, -iforest['scores'])
            step2 = max(1, len(fpr2) // 100)
            iforest_roc = [{'fpr': safe_float(fpr2[i]), 'tpr': safe_float(tpr2[i])} for i in range(0, len(fpr2), step2)]

            iforest_cm = confusion_matrix(labels, iforest['predictions']).tolist()
            iforest_metrics['confusion_matrix'] = iforest_cm

        # ── 6. Chart Data ──

        # Reconstruction error distribution
        error_bins = np.linspace(recon_errors.min(), min(recon_errors.max(), np.percentile(recon_errors, 99.5)), 40)
        error_dist = []
        for j in range(len(error_bins) - 1):
            lo, hi = error_bins[j], error_bins[j + 1]
            mask = (recon_errors >= lo) & (recon_errors < hi)
            entry = {'range': f'{lo:.4f}', 'count': int(mask.sum())}
            if has_labels:
                entry['normal'] = int((mask & (labels == 0)).sum())
                entry['anomaly'] = int((mask & (labels == 1)).sum())
            error_dist.append(entry)

        # Time series chart (reconstruction error over time)
        ts_chart = []
        timestamps = df['timestamp'].values if 'timestamp' in df.columns else [str(i) for i in range(n)]
        step_ts = max(1, n // 500)
        for i in range(0, n, step_ts):
            entry = {
                'timestamp': str(timestamps[i]),
                'recon_error': safe_float(recon_errors[i]),
                'ae_anomaly': int(ae_predictions[i]),
                'iforest_anomaly': int(iforest['predictions'][i]),
            }
            if has_labels:
                entry['actual'] = int(labels[i])
            ts_chart.append(entry)

        # 2D scatter of encoded representation (first 2 dims)
        encoded = ae_results['encoded']
        scatter_data = []
        step_sc = max(1, n // 500)
        for i in range(0, n, step_sc):
            entry = {
                'x': safe_float(encoded[i, 0]) if encoded.shape[1] > 0 else 0,
                'y': safe_float(encoded[i, 1]) if encoded.shape[1] > 1 else 0,
                'recon_error': safe_float(recon_errors[i]),
                'ae_anomaly': int(ae_predictions[i]),
            }
            if has_labels:
                entry['actual'] = int(labels[i])
            scatter_data.append(entry)

        # Feature correlation with anomaly score
        feat_corr = []
        for j, fname in enumerate(features):
            corr_val = float(np.corrcoef(X[:, j], recon_errors)[0, 1])
            feat_corr.append({'feature': fname, 'correlation': safe_float(corr_val)})
        feat_corr.sort(key=lambda x: abs(x['correlation']), reverse=True)

        # Feature stats by group
        feature_stats = []
        for j, fname in enumerate(features):
            entry = {
                'feature': fname,
                'mean': safe_float(X[:, j].mean()),
                'std': safe_float(X[:, j].std()),
                'min': safe_float(X[:, j].min()),
                'max': safe_float(X[:, j].max()),
            }
            if has_labels:
                entry['normal_mean'] = safe_float(X[labels == 0, j].mean())
                entry['anomaly_mean'] = safe_float(X[labels == 1, j].mean()) if labels.sum() > 0 else 0
            feature_stats.append(entry)

        # Model comparison chart
        model_comparison = []
        if ae_metrics and iforest_metrics:
            for m in ['accuracy', 'precision', 'recall', 'f1', 'auc']:
                model_comparison.append({
                    'metric': m.upper() if m != 'auc' else 'AUC-ROC',
                    'autoencoder': ae_metrics.get(m, 0),
                    'isolation_forest': iforest_metrics.get(m, 0),
                })

        # ── 7. Summary ──
        n_ae_anomalies = int(ae_predictions.sum())
        n_if_anomalies = int(iforest['predictions'].sum())
        n_actual = int(labels.sum()) if has_labels else None

        results = {
            'summary': {
                'n_samples': n,
                'n_features': n_features,
                'features': features,
                'has_labels': has_labels,
                'n_actual_anomalies': n_actual,
                'n_ae_anomalies': n_ae_anomalies,
                'n_if_anomalies': n_if_anomalies,
                'ae_anomaly_pct': safe_float(n_ae_anomalies / n * 100),
                'if_anomaly_pct': safe_float(n_if_anomalies / n * 100),
                'threshold': safe_float(threshold),
                'threshold_pct': request.thresholdPct,
            },
            'autoencoder': {
                'metrics': ae_metrics,
                'architecture': ae_results['architecture'],
                'encoding_dim': min(request.encodingDim, n_features - 1),
                'epochs': request.epochs,
                'final_train_loss': ae_results['final_train_loss'],
                'final_val_loss': ae_results['final_val_loss'],
            },
            'isolation_forest': {
                'metrics': iforest_metrics,
                'contamination': request.iforestContamination,
            },
            'feature_stats': feature_stats,
            'charts': {
                'error_distribution': error_dist,
                'time_series': ts_chart,
                'encoded_scatter': scatter_data,
                'feature_correlation': feat_corr,
                'loss_history': ae_results['loss_history'],
                'roc_ae': ae_roc,
                'roc_iforest': iforest_roc,
                'model_comparison': model_comparison,
            },
        }

        return _to_native({'results': results})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")
