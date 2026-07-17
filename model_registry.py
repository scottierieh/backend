"""
Model Registry — training + storage + inference for Model Lab 2 Predict (RC-2).

Design (docs/model-lab2-rc2-predict.md): the FRONTEND owns the registry METADATA in
Firestore (status pending→training→ready→failed, sourceRunId, versions, …). This
backend is STATELESS about the registry — it only trains, stores the joblib Pipeline,
and predicts. So the sync→async switch later changes nothing on the frontend: the
Repository keeps doing register()/observe(status), and only the executor here changes.

Two abstractions keep infra from blocking development (both have a dev impl now):
  - ModelStorage    : save/load/delete artifacts. LocalStorage now, GCSStorage later.
  - TrainingExecutor: run training. ImmediateExecutor (sync, dev) now,
                      CloudTasksExecutor (async, prod) later.

⚠️ DEVELOPMENT MODE: ImmediateExecutor trains inside the HTTP request — fine for small
datasets / API validation only. Production must swap in CloudTasksExecutor + a
Cloud Run Job so large-data training never runs in the request path.
"""

import os
import io
import json
import uuid
import datetime
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Versioning — bump when the training/preprocessing changes so a stored model can be
# traced ("why did the prediction change?"). Surfaced in the registry metadata.
PREPROCESSING_VERSION = "1.0.0"
PIPELINE_VERSION = "1.0.0"
TRAINING_CODE_VERSION = "1.0.0"


# ── Storage abstraction ──────────────────────────────────────────────────────
class ModelStorage(ABC):
    @abstractmethod
    def save(self, model_id: str, obj: Any) -> str: ...   # returns artifact_uri
    @abstractmethod
    def load(self, artifact_uri: str) -> Any: ...
    @abstractmethod
    def delete(self, artifact_uri: str) -> None: ...


class LocalStorage(ModelStorage):
    """Dev/test storage under a local dir. artifact_uri = file://<path>."""
    def __init__(self, root: Optional[str] = None):
        self.root = root or os.path.join(os.path.dirname(os.path.abspath(__file__)), "_model_store")
        os.makedirs(self.root, exist_ok=True)

    def save(self, model_id: str, obj: Any) -> str:
        path = os.path.join(self.root, f"{model_id}.joblib")
        joblib.dump(obj, path)
        return f"file://{path}"

    def load(self, artifact_uri: str) -> Any:
        path = artifact_uri.replace("file://", "", 1)
        return joblib.load(path)

    def delete(self, artifact_uri: str) -> None:
        path = artifact_uri.replace("file://", "", 1)
        if os.path.exists(path):
            os.remove(path)


class GCSStorage(ModelStorage):
    """Production storage. artifact_uri = gs://<bucket>/models/<model_id>.joblib.
    Lazy-imports google-cloud-storage so dev doesn't need the dependency."""
    def __init__(self, bucket: str):
        self.bucket_name = bucket

    def _bucket(self):
        from google.cloud import storage  # lazy
        return storage.Client().bucket(self.bucket_name)

    def save(self, model_id: str, obj: Any) -> str:
        blob_path = f"models/{model_id}.joblib"
        buf = io.BytesIO()
        joblib.dump(obj, buf)
        buf.seek(0)
        self._bucket().blob(blob_path).upload_from_file(buf)
        return f"gs://{self.bucket_name}/{blob_path}"

    def load(self, artifact_uri: str) -> Any:
        blob_path = artifact_uri.split(f"gs://{self.bucket_name}/", 1)[-1]
        data = self._bucket().blob(blob_path).download_as_bytes()
        return joblib.load(io.BytesIO(data))

    def delete(self, artifact_uri: str) -> None:
        blob_path = artifact_uri.split(f"gs://{self.bucket_name}/", 1)[-1]
        self._bucket().blob(blob_path).delete()


def default_storage() -> ModelStorage:
    bucket = os.environ.get("MODEL_STORE_BUCKET")
    return GCSStorage(bucket) if bucket else LocalStorage()


# ── Training executor abstraction ────────────────────────────────────────────
class TrainingExecutor(ABC):
    @abstractmethod
    def enqueue_training(self, model_id: str, spec: Dict[str, Any]) -> Dict[str, Any]: ...


class ImmediateExecutor(TrainingExecutor):
    """DEV ONLY — trains synchronously in-request. API-validation use, small data."""
    def __init__(self, storage: ModelStorage):
        self.storage = storage

    def enqueue_training(self, model_id: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        return train_and_store(model_id, spec, self.storage)


class CloudTasksExecutor(TrainingExecutor):
    """PROD (stub) — enqueue a Cloud Task that triggers a Cloud Run Job, returns
    immediately with status='training'. Frontend keeps polling Firestore status."""
    def enqueue_training(self, model_id: str, spec: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError("CloudTasksExecutor: wire Cloud Tasks + Cloud Run Job in prod.")


# ── Algorithm registry (id → estimator factory) ──────────────────────────────
def _estimator(algorithm: str, task: str):
    a = algorithm.lower().replace(" ", "_")
    if "random_forest" in a or a == "randomforest":
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        return RandomForestClassifier() if task == "classification" else RandomForestRegressor()
    if "xgboost" in a or a == "xgb":
        from xgboost import XGBClassifier, XGBRegressor
        return XGBClassifier() if task == "classification" else XGBRegressor()
    if "lightgbm" in a or a == "lgbm":
        from lightgbm import LGBMClassifier, LGBMRegressor
        return LGBMClassifier() if task == "classification" else LGBMRegressor()
    if "decision_tree" in a:
        from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
        return DecisionTreeClassifier() if task == "classification" else DecisionTreeRegressor()
    if "gbm" in a or "gradient_boosting" in a:
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        return GradientBoostingClassifier() if task == "classification" else GradientBoostingRegressor()
    if "svm" in a or "support_vector" in a:
        from sklearn.svm import SVC, SVR
        return SVC(probability=True) if task == "classification" else SVR()
    if "knn" in a or "neighbors" in a:
        from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
        return KNeighborsClassifier() if task == "classification" else KNeighborsRegressor()
    if "naive_bayes" in a:
        from sklearn.naive_bayes import GaussianNB
        return GaussianNB()
    if "discriminant" in a or a == "lda":
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        return LinearDiscriminantAnalysis()
    if "logistic" in a:
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(max_iter=1000)
    # Fallback: sensible default per task.
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    return RandomForestClassifier() if task == "classification" else RandomForestRegressor()


def build_feature_schema(df: pd.DataFrame, features: List[str]) -> List[Dict[str, Any]]:
    schema = []
    for col in features:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            schema.append({"name": col, "type": "numeric"})
        else:
            cats = [str(v) for v in s.dropna().unique().tolist()][:200]
            schema.append({"name": col, "type": "categorical", "categories": cats})
    return schema


def _build_pipeline(df: pd.DataFrame, features: List[str], estimator) -> Pipeline:
    num_cols = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    cat_cols = [c for c in features if c not in num_cols]
    pre = ColumnTransformer(transformers=[
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num_cols),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ], remainder="drop")
    # The whole Pipeline (preprocessing + model) is serialized, so inference reproduces
    # the exact training-time encoding — the reason we save a Pipeline, not a bare model.
    return Pipeline([("pre", pre), ("model", estimator)])


def train_and_store(model_id: str, spec: Dict[str, Any], storage: ModelStorage) -> Dict[str, Any]:
    """Train the chosen algorithm on the FULL data as a Pipeline, store it, return
    the registry outcome the frontend persists to Firestore."""
    data = spec.get("data")
    target = spec.get("target")
    features = spec.get("features") or []
    task = spec.get("task") or "classification"
    algorithm = spec.get("algorithm") or "random_forest"
    if not data or not target or not features:
        raise ValueError("Missing data, target, or features.")

    df = pd.DataFrame(data)
    missing = [c for c in [target] + features if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found: {', '.join(missing)}")

    feature_schema = build_feature_schema(df, features)
    X = df[features].copy()
    y = df[target].copy()
    valid = ~y.isna()
    X, y = X[valid].reset_index(drop=True), y[valid].reset_index(drop=True)
    if len(X) < 20:
        raise ValueError("At least 20 valid rows required to register a model.")

    pipe = _build_pipeline(df, features, _estimator(algorithm, task))
    pipe.fit(X, y)

    # Full-data training metric (train fit — indicative; the honest CV metric lives on
    # the Run). Kept small and non-authoritative on purpose.
    from sklearn.metrics import accuracy_score, r2_score
    if task == "classification":
        metrics = {"train_accuracy": float(accuracy_score(y, pipe.predict(X)))}
    else:
        metrics = {"train_r2": float(r2_score(y, pipe.predict(X)))}

    artifact_uri = storage.save(model_id, {"pipeline": pipe, "features": features,
                                           "target": target, "task": task,
                                           "feature_schema": feature_schema})
    return {
        "modelId": model_id,
        "status": "ready",
        "artifactUri": artifact_uri,
        "featureSchema": feature_schema,
        "metrics": metrics,
        "preprocessingVersion": PREPROCESSING_VERSION,
        "pipelineVersion": PIPELINE_VERSION,
        "trainingCodeVersion": TRAINING_CODE_VERSION,
        "readyAt": datetime.datetime.utcnow().isoformat() + "Z",
    }


def predict(artifact_uri: str, rows: List[Dict[str, Any]], storage: ModelStorage) -> Dict[str, Any]:
    bundle = storage.load(artifact_uri)
    pipe, features, task = bundle["pipeline"], bundle["features"], bundle["task"]
    df = pd.DataFrame(rows)
    for c in features:
        if c not in df.columns:
            df[c] = np.nan
    X = df[features]
    preds = pipe.predict(X)
    out: Dict[str, Any] = {"predictions": [(_native(p)) for p in preds]}
    if task == "classification" and hasattr(pipe, "predict_proba"):
        try:
            proba = pipe.predict_proba(X)
            out["probabilities"] = [float(np.max(row)) for row in proba]
        except Exception:
            pass
    return out


def new_model_id() -> str:
    return uuid.uuid4().hex


def _native(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return None if (np.isnan(v) or np.isinf(v)) else float(v)
    return v
