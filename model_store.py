"""
model_store.py — GCS-backed persistence for trained Model Lab pipelines.

Model Lab 2's frontend (register-model.ts / predict-client.ts) has called
POST /api/models/{id}/train and /predict since it was written, expecting a
real persisted, re-loadable model behind `artifactUri` — until now nothing
in this backend actually served that contract (see models_api.py's module
docstring for the full picture). This module is the storage half of that:
save a fitted pipeline once at train time, reload it by URI on every
predict/explain call.

Bucket defaults to this project's Firebase Storage bucket (same one the
frontend already uses for other file storage), overridable via
MODEL_ARTIFACTS_BUCKET for local/dev use against a different bucket.
"""

import os
import io
import time
import threading
import joblib

_BUCKET_NAME = os.environ.get('MODEL_ARTIFACTS_BUCKET', 'restart2-98207181-3e3a5.firebasestorage.app')

# Process-local cache: a warm Cloud Run instance shouldn't re-download the
# same artifact on every predict call in a burst. Small and unbounded by
# design — one instance serves a handful of active models at a time, not
# thousands; if that ever changes, cap it with an LRU eviction instead.
_cache_lock = threading.Lock()
_pipeline_cache: dict[str, object] = {}


def _client():
    # Imported lazily so a backend that never touches model-serving routes
    # doesn't pay the google-cloud-storage import cost or need credentials
    # configured just to boot.
    from google.cloud import storage
    return storage.Client()


def _blob_path_from_uri(artifact_uri: str) -> str:
    """artifact_uri is gs://<bucket>/<path> — return just <path>, validating
    the bucket matches what this instance is configured to serve (a URI from
    a different bucket is refused rather than silently trusted)."""
    if not artifact_uri.startswith('gs://'):
        raise ValueError(f'Not a gs:// artifact URI: {artifact_uri!r}')
    rest = artifact_uri[len('gs://'):]
    bucket, _, path = rest.partition('/')
    if bucket != _BUCKET_NAME:
        raise ValueError(f'Artifact URI bucket {bucket!r} does not match configured bucket {_BUCKET_NAME!r}')
    if not path:
        raise ValueError(f'Artifact URI has no object path: {artifact_uri!r}')
    return path


def save_pipeline(model_id: str, artifact: dict) -> str:
    """Serialize `artifact` (the {pipeline, label_encoder, task, features, target}
    dict models_api.py builds) and upload it. Returns the gs:// URI to store as
    ModelRegistryEntry.artifactUri."""
    buf = io.BytesIO()
    joblib.dump(artifact, buf)
    buf.seek(0)
    path = f'models/{model_id}/{int(time.time() * 1000)}/pipeline.joblib'
    blob = _client().bucket(_BUCKET_NAME).blob(path)
    blob.upload_from_file(buf, content_type='application/octet-stream')
    return f'gs://{_BUCKET_NAME}/{path}'


def load_pipeline(artifact_uri: str) -> dict:
    """Load (and cache) the artifact dict previously written by save_pipeline."""
    with _cache_lock:
        cached = _pipeline_cache.get(artifact_uri)
    if cached is not None:
        return cached
    path = _blob_path_from_uri(artifact_uri)
    blob = _client().bucket(_BUCKET_NAME).blob(path)
    buf = io.BytesIO()
    blob.download_to_file(buf)
    buf.seek(0)
    artifact = joblib.load(buf)
    with _cache_lock:
        _pipeline_cache[artifact_uri] = artifact
    return artifact


def delete_artifact(artifact_uri: str) -> None:
    """Best-effort delete — register-model.ts's unregisterModel already treats
    the DELETE call as fire-and-forget and removes the Firestore doc regardless,
    so a missing/already-gone blob is not an error here."""
    try:
        path = _blob_path_from_uri(artifact_uri)
    except ValueError:
        return
    try:
        _client().bucket(_BUCKET_NAME).blob(path).delete()
    except Exception:
        pass
    with _cache_lock:
        _pipeline_cache.pop(artifact_uri, None)
