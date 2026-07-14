"""
model_store.py — shared model persistence for the Predict / What-if features.

Every *_analysis.py script currently trains a model and throws it away once the
process exits — there was no way to run the same model again on new input, so
the frontend's Predict/What-if buttons had nothing real to call (they always
hit predict-schema/predict, which never existed). This module gives scripts a
single place to save a fitted model + everything needed to reproduce its
input transform, and gives main.py's predict routes a single place to load it
back.

Bundle format (joblib-pickled dict):
  model         : the fitted sklearn-compatible estimator
  feature_cols  : raw feature column names, in the order the UI should collect them
  dummy_columns : post-pd.get_dummies() column order the model was actually
                  trained on (None if the script didn't one-hot encode)
  target_col    : target column name
  task_type     : 'classification' | 'regression'

Storage is GCS, bucket name from the MODEL_BUCKET env var. Every function here
is defensive — save failures are caught by the caller (a model that can't be
persisted should never fail the analysis itself), and load failures raise
FileNotFoundError / RuntimeError with a clear message for the API layer to
turn into an HTTP error.
"""

import io
import os
import uuid

import joblib

_BUCKET_NAME = os.environ.get("MODEL_BUCKET")
_client = None


def _bucket():
    global _client
    if not _BUCKET_NAME:
        raise RuntimeError(
            "MODEL_BUCKET env var not set — model persistence is disabled on this deploy"
        )
    if _client is None:
        from google.cloud import storage
        _client = storage.Client()
    return _client.bucket(_BUCKET_NAME)


def save_model_bundle(model, feature_cols, target_col, task_type, analysis_type,
                       num_cols=None, dummy_columns=None, **extra) -> str:
    """Persist a fitted model + its input contract. Returns the model_id to hand back
    to the frontend as modelArtifactRef. Raises on failure — callers should wrap this
    in try/except so a storage hiccup never crashes the analysis itself.

    num_cols: which of feature_cols are numeric (drives number vs text input rendering
    in Predict/What-if). Defaults to all of feature_cols if not given.
    """
    model_id = uuid.uuid4().hex
    feature_cols = list(feature_cols)
    num_cols = list(num_cols) if num_cols is not None else list(feature_cols)
    bundle = {
        "model": model,
        "feature_cols": feature_cols,
        "num_cols": num_cols,
        "cat_cols": [c for c in feature_cols if c not in num_cols],
        "dummy_columns": list(dummy_columns) if dummy_columns is not None else None,
        "target_col": target_col,
        "task_type": task_type,
        "analysis_type": analysis_type,
        "classes": [str(c) for c in model.classes_] if hasattr(model, "classes_") else None,
        **extra,
    }
    buf = io.BytesIO()
    joblib.dump(bundle, buf)
    buf.seek(0)
    blob = _bucket().blob(f"model_bundles/{model_id}.joblib")
    blob.upload_from_file(buf, content_type="application/octet-stream")
    return model_id


def load_model_bundle(model_id: str) -> dict:
    blob = _bucket().blob(f"model_bundles/{model_id}.joblib")
    if not blob.exists():
        raise FileNotFoundError(f"Model {model_id!r} not found")
    buf = io.BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)
    return joblib.load(buf)


def align_features(df, feature_cols, dummy_columns):
    """Reproduce the training-time one-hot encoding on new prediction input: encode,
    then reindex to the exact training-time dummy column set (missing dummy columns
    filled with 0, unseen categories/extra columns dropped)."""
    X = df[feature_cols]
    if dummy_columns is None:
        return X
    import pandas as pd
    X = pd.get_dummies(X)
    return X.reindex(columns=dummy_columns, fill_value=0)
