"""
models_api.py — Model Lab's model registry backend: train, predict, explain,
delete, by model_id.

THE GAP THIS CLOSES: the frontend (src/lib/model-lab2/register-model.ts and
predict-client.ts) has called POST /api/models/{id}/train, POST
/api/models/{id}/predict, and DELETE /api/models/{id} since those files were
written — but until now nothing in this backend served those routes at all.
register-model.ts's own comment says so: "Until the backend is deployed this
ends at 'failed' (honest)." No fitted model was ever persisted anywhere
either — every *_analysis.py script fits a model, reports metrics, and lets
the object be garbage-collected on process exit (zero joblib/pickle usage
in this repo before this file).

This is a genuinely different shape from the rest of the backend: every
other analysis is a one-shot CLI script (stdin JSON -> stdout JSON, wired
through main.py's SCRIPT_ROUTES subprocess table) with no state between
calls. A model registry is inherently stateful — train once, predict many
times later — so this is a real APIRouter module (mirroring the
conjoint/survey family's pattern in main.py, not the subprocess pattern),
persisting fitted pipelines to Cloud Storage via model_store.py and
reloading them by model_id on every predict/explain call.

CONTRACT (fixed by the frontend, which was written first — see
register-model.ts:84-103 and predict-client.ts): train's request/response
shape and predict's base {predictions, probabilities} shape are matched
exactly here so NO frontend change is needed for those to start working.
predict's `explain`/`shapContributions` fields are new and additive.

Design choices worth knowing before touching this file:
  - SHAP is computed via a callable wrapping pipeline.predict/predict_proba,
    not shap.TreeExplainer directly on the fitted estimator. That trades
    TreeExplainer's speed for correctness: TreeExplainer needs the transformed
    (post-ColumnTransformer) feature space, which explodes a one-hot-encoded
    categorical into several dummy columns and would report SHAP per dummy
    column instead of per original feature. The callable approach runs in
    the ORIGINAL raw feature space, so SHAP values line up with the same
    feature names the user picked and sees everywhere else. Slower
    (Permutation, not Tree), acceptable for a "runs once per registration /
    once per explain click" workload with typical tabular column counts.
  - A small background sample of training rows is persisted in the artifact
    itself (not recomputed at predict time) so per-row explain doesn't need
    the original training data around.
  - Every persisted artifact is a dict {pipeline, label_encoder, task,
    features, background}, not a bare sklearn object — label_encoder is
    needed to turn predicted class indices back into the original label
    strings the user actually chose.
"""

import time
import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import Any, Literal, Optional

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error

from analysis_common import _to_native_type
from algorithm_registry import build_estimator
import model_store

router = APIRouter()

Task = Literal['classification', 'regression']

BACKGROUND_SAMPLE_SIZE = 50
BEESWARM_SAMPLE_SIZE = 100
MIN_TRAINING_ROWS = 10


class TrainRequest(BaseModel):
    data: list[dict[str, Any]]
    algorithm: str
    target: str
    features: list[str]
    task: Task


class PredictRequest(BaseModel):
    artifactUri: str
    rows: list[dict[str, Any]]
    explain: Optional[bool] = False


def _fail(status: int, detail: str):
    raise HTTPException(status_code=status, detail=detail)


def _numeric_ratio(series: pd.Series) -> float:
    non_null = series.notna().sum()
    if non_null == 0:
        return 0.0
    return pd.to_numeric(series, errors='coerce').notna().sum() / non_null


def _split_feature_types(df: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    numeric = [c for c in features if _numeric_ratio(df[c]) >= 0.8]
    categorical = [c for c in features if c not in numeric]
    return numeric, categorical


def _build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    transformers = []
    if numeric_features:
        transformers.append(('num', Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
        ]), numeric_features))
    if categorical_features:
        transformers.append(('cat', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(handle_unknown='ignore')),
        ]), categorical_features))
    return ColumnTransformer(transformers, remainder='drop')


def _build_pipeline(algorithm: str, task: Task, numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    return Pipeline([
        ('pre', _build_preprocessor(numeric_features, categorical_features)),
        ('est', build_estimator(algorithm, task)),
    ])


def _prepare_xy(df: pd.DataFrame, target: str, features: list[str], task: Task, numeric_features: list[str]):
    """Coerce numeric feature columns, drop rows with a missing/non-numeric
    target, and encode y for classification. Returns (X, y, label_encoder)."""
    df = df.copy()
    for c in numeric_features:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    if task == 'classification':
        valid = df[target].notna()
        X = df.loc[valid, features]
        y_raw = df.loc[valid, target].astype(str)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)
        return X, y, label_encoder

    y_num = pd.to_numeric(df[target], errors='coerce')
    valid = y_num.notna()
    X = df.loc[valid, features]
    y = y_num.loc[valid].values
    return X, y, None


def _encode_for_shap(X: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.Index]]:
    """shap's default masker perturbs data numerically (np.isclose etc.) and
    breaks outright on a raw string column, so every non-numeric feature gets
    factorized to integer codes first. Returns the coded frame plus the
    per-column category index the codes were drawn from — callers use that
    same index both as the chart's numeric 'value' (a categorical feature's
    SHAP beeswarm point is its code, same convention as most SHAP UIs) and,
    via _decode_row, to turn codes back into real category strings before
    calling the pipeline."""
    X_numeric = pd.DataFrame(index=X.index)
    categories: dict[str, pd.Index] = {}
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            X_numeric[col] = pd.to_numeric(X[col], errors='coerce').astype(float)
        else:
            codes, uniques = pd.factorize(X[col])
            X_numeric[col] = codes.astype(float)
            categories[col] = uniques
    return X_numeric, categories


def _encode_with(X: pd.DataFrame, categories: dict[str, pd.Index]) -> pd.DataFrame:
    """Encode X using an EXISTING categories mapping (from _encode_for_shap
    on the background), so codes stay consistent with what the explainer's
    background was built against, rather than each call inventing its own
    code assignment. A category unseen in that background maps to -1 (which
    _decode_row then clips into range rather than erroring on)."""
    X_numeric = pd.DataFrame(index=X.index)
    for col in X.columns:
        if col in categories:
            mapping = {v: i for i, v in enumerate(categories[col])}
            X_numeric[col] = X[col].map(mapping).astype('float64').where(X[col].isin(mapping), -1.0)
        else:
            X_numeric[col] = pd.to_numeric(X[col], errors='coerce').astype(float)
    return X_numeric


def _decode_row(arr, columns: list[str], categories: dict[str, pd.Index]) -> pd.DataFrame:
    frame = pd.DataFrame(np.asarray(arr), columns=columns)
    for col, uniques in categories.items():
        codes = frame[col].round().astype(int).clip(0, len(uniques) - 1)
        frame[col] = np.asarray(uniques)[codes.values]
    return frame


def _predict_fn_for(pipeline: Pipeline, task: Task, columns: list[str], categories: dict[str, pd.Index]):
    """A callable over the shap-encoded (numeric-coded) feature space that
    decodes back to real categories before calling the pipeline — see
    _encode_for_shap for why the coding step exists at all."""
    def fn(data):
        frame = _decode_row(data, columns, categories)
        if task == 'classification':
            proba = pipeline.predict_proba(frame)
            return proba[:, 1] if proba.shape[1] == 2 else proba.max(axis=1)
        return pipeline.predict(frame)
    return fn


def _compute_beeswarm(pipeline: Pipeline, task: Task, X: pd.DataFrame) -> Optional[dict]:
    try:
        import shap
        background_n = min(len(X), BACKGROUND_SAMPLE_SIZE)
        background_raw = X.sample(n=background_n, random_state=42) if len(X) > background_n else X
        background, categories = _encode_for_shap(background_raw)

        n = min(len(X), BEESWARM_SAMPLE_SIZE)
        sample_raw = X.sample(n=n, random_state=1) if len(X) > n else X
        sample = _encode_with(sample_raw, categories)

        explainer = shap.Explainer(_predict_fn_for(pipeline, task, list(X.columns), categories), background)
        sv = explainer(sample, silent=True)
        values = np.asarray(sv.values)

        features_out = []
        for j, feat in enumerate(X.columns):
            shap_col = values[:, j]
            value_col = sample[feat].values
            features_out.append({
                'feature': feat,
                'meanAbsShap': _to_native_type(float(np.abs(shap_col).mean())),
                'points': [
                    {'shap': _to_native_type(float(shap_col[i])), 'value': _to_native_type(float(value_col[i]))}
                    for i in range(len(sample))
                ],
            })
        features_out.sort(key=lambda f: f['meanAbsShap'], reverse=True)
        return {'features': features_out}
    except Exception:
        return None


def _compute_row_contributions(pipeline: Pipeline, task: Task, background_raw: pd.DataFrame, rows_raw: pd.DataFrame) -> Optional[list]:
    try:
        import shap
        background, categories = _encode_for_shap(background_raw)
        rows = _encode_with(rows_raw, categories)

        explainer = shap.Explainer(_predict_fn_for(pipeline, task, list(rows_raw.columns), categories), background)
        sv = explainer(rows, silent=True)
        values = np.asarray(sv.values)
        out = []
        for i in range(len(rows)):
            out.append([
                {
                    'feature': feat,
                    'value': _to_native_type(float(rows[feat].iloc[i])),
                    'shap': _to_native_type(float(values[i, j])),
                }
                for j, feat in enumerate(rows_raw.columns)
            ])
        return out
    except Exception:
        return None


@router.post("/models/{model_id}/train")
def train_model(model_id: str, req: TrainRequest):
    if not req.data:
        _fail(400, "No training data provided")
    if req.target not in (req.data[0].keys() if req.data else []):
        _fail(400, f"Target column '{req.target}' not found in data")
    df = pd.DataFrame(req.data)
    missing = [f for f in req.features if f not in df.columns]
    if missing:
        _fail(400, f"Feature column(s) not found: {missing}")
    if req.target not in df.columns:
        _fail(400, f"Target column '{req.target}' not found in data")

    numeric_features, categorical_features = _split_feature_types(df, req.features)
    X, y, label_encoder = _prepare_xy(df, req.target, req.features, req.task, numeric_features)
    if len(X) < MIN_TRAINING_ROWS:
        _fail(400, f"Not enough valid rows to train ({len(X)} after dropping missing targets, need >= {MIN_TRAINING_ROWS})")

    try:
        estimator = build_estimator(req.algorithm, req.task)
    except ValueError as e:
        _fail(400, str(e))

    # Honest metrics from a held-out split first...
    metrics: dict[str, float] = {}
    try:
        stratify = y if req.task == 'classification' and len(set(y)) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify,
        )
        eval_pipeline = _build_pipeline(req.algorithm, req.task, numeric_features, categorical_features)
        eval_pipeline.fit(X_train, y_train)
        y_pred = eval_pipeline.predict(X_test)
        if req.task == 'classification':
            metrics['accuracy'] = _to_native_type(accuracy_score(y_test, y_pred))
            metrics['f1'] = _to_native_type(f1_score(y_test, y_pred, average='weighted'))
        else:
            metrics['r2'] = _to_native_type(r2_score(y_test, y_pred))
            metrics['mae'] = _to_native_type(mean_absolute_error(y_test, y_pred))
    except Exception:
        pass  # best-effort — register-model.ts falls back to the run's own metrics if this is empty

    # ...then a fresh fit on ALL valid rows for the model actually served.
    pipeline = _build_pipeline(req.algorithm, req.task, numeric_features, categorical_features)
    try:
        pipeline.fit(X, y)
    except Exception as e:
        _fail(422, f"Training failed: {e}")

    shap_beeswarm = _compute_beeswarm(pipeline, req.task, X)

    background_n = min(len(X), BACKGROUND_SAMPLE_SIZE)
    background = X.sample(n=background_n, random_state=42) if len(X) > background_n else X

    artifact = {
        'pipeline': pipeline,
        'label_encoder': label_encoder,
        'task': req.task,
        'features': req.features,
        'background': background,
    }
    try:
        artifact_uri = model_store.save_pipeline(model_id, artifact)
    except Exception as e:
        _fail(502, f"Could not save model artifact: {e}")

    return {'artifactUri': artifact_uri, 'metrics': metrics or None, 'shapBeeswarm': shap_beeswarm}


@router.post("/models/{model_id}/predict")
def predict_model(model_id: str, req: PredictRequest):
    if not req.rows:
        _fail(400, "No rows to predict")
    try:
        artifact = model_store.load_pipeline(req.artifactUri)
    except Exception as e:
        _fail(404, f"Could not load model artifact: {e}")

    pipeline: Pipeline = artifact['pipeline']
    label_encoder: Optional[LabelEncoder] = artifact.get('label_encoder')
    task: Task = artifact['task']
    features: list[str] = artifact['features']
    background: pd.DataFrame = artifact.get('background')

    rows_df = pd.DataFrame(req.rows)
    missing = [f for f in features if f not in rows_df.columns]
    if missing:
        _fail(400, f"Row(s) missing required feature(s): {missing}")
    X = rows_df[features]

    try:
        raw_pred = pipeline.predict(X)
    except Exception as e:
        _fail(422, f"Prediction failed: {e}")

    probabilities = None
    if task == 'classification':
        predictions = list(label_encoder.inverse_transform(raw_pred)) if label_encoder is not None else list(raw_pred)
        try:
            proba = pipeline.predict_proba(X)
            probabilities = [_to_native_type(float(row.max())) for row in proba]
        except Exception:
            probabilities = None
    else:
        predictions = [_to_native_type(float(v)) for v in raw_pred]

    shap_contributions = None
    if req.explain and background is not None:
        shap_contributions = _compute_row_contributions(pipeline, task, background, X)

    return {
        'predictions': [_to_native_type(p) for p in predictions],
        'probabilities': probabilities,
        'shapContributions': shap_contributions,
    }


@router.delete("/models/{model_id}")
def delete_model(model_id: str, artifactUri: Optional[str] = Query(default=None)):
    if artifactUri:
        model_store.delete_artifact(artifactUri)
    return {'ok': True}
