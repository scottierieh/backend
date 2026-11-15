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

import base64
import hashlib
import hmac
import io
import logging
import os
import re
import secrets
import shutil
import tempfile
import threading
import time
from collections import OrderedDict

import joblib

logger = logging.getLogger("skari.model_store")

_BUCKET_NAME = os.environ.get('MODEL_ARTIFACTS_BUCKET', 'restart2-98207181-3e3a5.firebasestorage.app')

# ---------------------------------------------------------------------------
# Trust model. joblib.load is pickle: loading an attacker-chosen object is
# remote code execution. Before this, /predict loaded ANY gs://<bucket>/<path>
# the caller named — including objects users can upload themselves (storage.rules
# lets a signed-in user write monitoring/{uid}/*.csv and skari/{uid}/**, and a
# pickle does not care what the file is called). So now:
#
#   1. the object path must be models/{model_id}/{digits}/pipeline.joblib with
#      model_id = the route's {model_id} (storage.rules gives clients no write
#      access under models/ — the default deny), AND
#   2. the object must carry a server-written HMAC-SHA256 signature in its GCS
#      custom metadata, over (path, owner uid, sha256 of the bytes), keyed with
#      SKARI_MODEL_SIGNING_KEY (falling back to SKARI_BACKEND_KEY). The digest
#      is recomputed over the bytes actually downloaded, so a swapped object
#      fails even if its metadata was copied.
#
# Predict and delete need the same signature AND the caller must be the uid
# recorded as owner at train time, or the service key (the Next.js server: the
# public-form route, which checks the form and the model record itself, and
# account deletion). The owner is checked from the verified metadata BEFORE
# the bytes are downloaded or unpickled. An artifact signed without an owner
# (trained while authentication was off) is service-key only, except in
# explicit local-dev mode (SKARI_AUTH_DISABLED=1), where the anonymous caller
# may use those and only those.
#
# WHAT AN ARTIFACT HOLDS (2026-11 audit E-06): the fitted pipeline, the label
# encoder, the feature list, the threshold, the fitted Feature Engineering
# recipe and `background` -- up to 50 RAW training rows (feature columns only,
# no target) that SHAP needs as its reference distribution for per-row
# explanations; the AutoGluon kind also carries the packed predictor. Some
# fitted models hold training data themselves (k-NN, SVM support vectors,
# several AutoGluon members). So an artifact is personal data of the account
# that trained it: delete_owner_artifacts() removes all of an account's
# artifacts for account deletion.
# ---------------------------------------------------------------------------
_SIG_VERSION = 'v1'
_META_SIG = 'skari-sig'
_META_OWNER = 'skari-owner'
_META_SHA = 'skari-sha256'

_MODEL_ID_RE = re.compile(r'^[A-Za-z0-9_-]{1,128}$')

_ephemeral_key = None


_warned_fallback = False


def _signing_key() -> bytes:
    key = os.environ.get('SKARI_MODEL_SIGNING_KEY') or ''
    if not key:
        # Fallback kept so artifacts already signed with the service key keep
        # loading; the Next.js server also holds that key, so set a separate
        # SKARI_MODEL_SIGNING_KEY (and re-sign with scripts/sign_legacy_models.py).
        key = os.environ.get('SKARI_BACKEND_KEY') or ''
        global _warned_fallback
        if key and not _warned_fallback:
            _warned_fallback = True
            logger.warning('SKARI_MODEL_SIGNING_KEY is not set: model artifacts are signed '
                           'with SKARI_BACKEND_KEY. Set a separate signing key.')
    if key:
        return key.encode('utf-8')
    # Local dev without any key: a per-process random key. Models trained in
    # this process are servable by it; nothing else (no earlier artifact, no
    # foreign object) will ever verify.
    global _ephemeral_key
    if _ephemeral_key is None:
        logger.warning('No SKARI_MODEL_SIGNING_KEY / SKARI_BACKEND_KEY set: model '
                       'artifacts are signed with a per-process key and will not '
                       'load after a restart or on another instance.')
        _ephemeral_key = secrets.token_bytes(32)
    return _ephemeral_key


def _sign(path: str, owner: str, sha256_hex: str) -> str:
    msg = '\n'.join([_SIG_VERSION, path, owner, sha256_hex]).encode('utf-8')
    mac = hmac.new(_signing_key(), msg, hashlib.sha256).digest()
    return _SIG_VERSION + ':' + base64.urlsafe_b64encode(mac).decode('ascii')


def _verify(path: str, metadata, data_sha256_hex: str = None) -> str:
    """Return the signed owner uid ('' if none) or raise ValueError."""
    metadata = metadata or {}
    sig = metadata.get(_META_SIG) or ''
    owner = metadata.get(_META_OWNER) or ''
    sha = metadata.get(_META_SHA) or ''
    if not sig or not sha:
        raise ValueError('Model artifact is not signed by this service.')
    if data_sha256_hex is not None and not hmac.compare_digest(sha, data_sha256_hex):
        raise ValueError('Model artifact content does not match its signature.')
    if not hmac.compare_digest(sig, _sign(path, owner, sha)):
        raise ValueError('Model artifact signature is invalid.')
    return owner


def validate_model_id(model_id: str) -> str:
    if not isinstance(model_id, str) or not _MODEL_ID_RE.match(model_id):
        raise ValueError('Invalid model id.')
    return model_id


def _expected_path_re(model_id: str):
    return re.compile(r'^models/' + re.escape(model_id) + r'/[0-9]{1,20}/pipeline\.joblib$')


# Process-local cache: a warm Cloud Run instance shouldn't re-download the
# same artifact on every predict call in a burst. Least-recently-used, at most
# _CACHE_MAX artifacts (SKARI_MODEL_CACHE_MAX, default 16): unbounded, one
# account could fill the instance's memory by training and scoring many small
# models. Evicting an AutoGluon artifact also removes the ag_serve_* directory
# its predictor was unpacked into (models_api._ag_unpack records it as
# `_ag_dir`). Holds (artifact, signed owner); only ever artifacts that passed
# signature verification.
_CACHE_MAX = max(1, int(os.environ.get('SKARI_MODEL_CACHE_MAX', '16')))
_cache_lock = threading.Lock()
_pipeline_cache: 'OrderedDict[str, tuple]' = OrderedDict()


def _release(artifact):
    """Free what an evicted artifact holds outside the Python heap."""
    d = artifact.get('_ag_dir') if isinstance(artifact, dict) else None
    if isinstance(d, str) and os.path.basename(d).startswith('ag_serve_') \
            and os.path.dirname(os.path.abspath(d)) == os.path.abspath(tempfile.gettempdir()):
        shutil.rmtree(d, ignore_errors=True)


def _cache_get(path):
    with _cache_lock:
        hit = _pipeline_cache.get(path)
        if hit is not None:
            _pipeline_cache.move_to_end(path)
        return hit


def _cache_put(path, artifact, owner):
    evicted = []
    with _cache_lock:
        _pipeline_cache[path] = (artifact, owner)
        _pipeline_cache.move_to_end(path)
        while len(_pipeline_cache) > _CACHE_MAX:
            evicted.append(_pipeline_cache.popitem(last=False)[1][0])
    for a in evicted:
        _release(a)


def _cache_pop(path):
    with _cache_lock:
        hit = _pipeline_cache.pop(path, None)
    if hit is not None:
        _release(hit[0])


def _client():
    # Imported lazily so a backend that never touches model-serving routes
    # doesn't pay the google-cloud-storage import cost or need credentials
    # configured just to boot.
    from google.cloud import storage
    return storage.Client()


def _bucket():
    return _client().bucket(_BUCKET_NAME)


def _blob_path_from_uri(artifact_uri: str, model_id: str = None) -> str:
    """artifact_uri is gs://<bucket>/<path> — return just <path>, validating
    the bucket matches what this instance is configured to serve (a URI from
    a different bucket is refused rather than silently trusted) and, when
    model_id is given, that the path is that model's own artifact path."""
    if not isinstance(artifact_uri, str) or not artifact_uri.startswith('gs://'):
        raise ValueError(f'Not a gs:// artifact URI: {artifact_uri!r}')
    rest = artifact_uri[len('gs://'):]
    bucket, _, path = rest.partition('/')
    if bucket != _BUCKET_NAME:
        raise ValueError(f'Artifact URI bucket {bucket!r} does not match configured bucket {_BUCKET_NAME!r}')
    if not path:
        raise ValueError(f'Artifact URI has no object path: {artifact_uri!r}')
    if model_id is not None:
        validate_model_id(model_id)
        if not _expected_path_re(model_id).match(path):
            raise ValueError('Artifact URI does not belong to this model.')
    return path


def save_pipeline(model_id: str, artifact: dict, owner_uid: str = None) -> str:
    """Serialize `artifact` (the {pipeline, label_encoder, task, features, target}
    dict models_api.py builds), sign it, and upload it. Returns the gs:// URI to
    store as ModelRegistryEntry.artifactUri."""
    validate_model_id(model_id)
    buf = io.BytesIO()
    joblib.dump(artifact, buf)
    data = buf.getvalue()
    path = f'models/{model_id}/{int(time.time() * 1000)}/pipeline.joblib'
    owner = owner_uid or ''
    sha = hashlib.sha256(data).hexdigest()
    blob = _bucket().blob(path)
    blob.metadata = {_META_OWNER: owner, _META_SHA: sha, _META_SIG: _sign(path, owner, sha)}
    blob.upload_from_file(io.BytesIO(data), content_type='application/octet-stream')
    return f'gs://{_BUCKET_NAME}/{path}'


class NotOwner(Exception):
    pass


def _check_owner(owner: str, caller_uid=None, is_service=False, auth_enforced=True):
    """Raise NotOwner unless the caller may use an artifact signed for `owner`.

    `auth_enforced=False` is passed only in explicit local-dev mode
    (auth.auth_disabled()); even then an artifact that names an owner stays
    that owner's."""
    if is_service:
        return
    if not auth_enforced and not caller_uid and not owner:
        return
    if not caller_uid or not owner or not hmac.compare_digest(owner, caller_uid):
        raise NotOwner()


def load_pipeline(artifact_uri: str, model_id: str, caller_uid: str = None,
                  is_service: bool = True, auth_enforced: bool = True) -> dict:
    """Load (and cache) an artifact previously written by save_pipeline for
    `model_id`. Raises ValueError for anything this service did not sign and
    NotOwner when the caller is neither its signed owner nor the service key.

    The default (is_service=True) is for in-process callers (scripts, checks);
    the HTTP route passes the real caller."""
    path = _blob_path_from_uri(artifact_uri, model_id)
    cached = _cache_get(path)
    if cached is not None:
        _check_owner(cached[1], caller_uid, is_service, auth_enforced)
        return cached[0]
    blob = _bucket().get_blob(path)
    if blob is None:
        raise ValueError('Model artifact not found.')
    # Owner first, from the signed metadata alone: a caller who may not use
    # this model never makes us download or unpickle it.
    owner = _verify(path, blob.metadata)
    _check_owner(owner, caller_uid, is_service, auth_enforced)
    data = blob.download_as_bytes()
    _verify(path, blob.metadata, hashlib.sha256(data).hexdigest())
    artifact = joblib.load(io.BytesIO(data))
    _cache_put(path, artifact, owner)
    return artifact


def delete_artifact(artifact_uri: str, model_id: str, caller_uid: str = None,
                    is_service: bool = False, auth_enforced: bool = True) -> bool:
    """Delete a signed artifact of `model_id`. Raises NotOwner when the caller
    is not the uid it was trained by. Returns False (not an error) for a URI
    that is not this model's or an object already gone — register-model.ts
    treats the DELETE as fire-and-forget and removes the Firestore doc anyway."""
    try:
        path = _blob_path_from_uri(artifact_uri, model_id)
    except ValueError:
        return False
    try:
        blob = _bucket().get_blob(path)
    except Exception:
        return False
    if blob is None:
        return False
    if not is_service:
        # The trusted Next.js server (service key) may delete any artifact of
        # this model, signed or legacy; everyone else must be the signed owner
        # -- also when authentication is off for local dev (_check_owner).
        try:
            owner = _verify(path, blob.metadata)  # metadata only; never deserialized
        except ValueError:
            raise NotOwner()
        _check_owner(owner, caller_uid, False, auth_enforced)
    try:
        blob.delete(if_generation_match=blob.generation)
    except Exception:
        pass
    _cache_pop(path)
    return True


_ANY_ARTIFACT_RE = re.compile(r'^models/[A-Za-z0-9_-]{1,128}/[0-9]{1,20}/pipeline\.joblib$')


def delete_owner_artifacts(owner_uid: str) -> dict:
    """Delete every artifact whose signed owner is `owner_uid` (account
    deletion). Service-key only -- the route checks that. Lists models/ and
    reads each object's metadata; nothing is downloaded or deserialized. An
    object whose signature does not verify is skipped: its owner field cannot
    be trusted, and it is not servable anyway.

    Returns {'deleted': n, 'failed': n}."""
    if not isinstance(owner_uid, str) or not owner_uid or len(owner_uid) > 128:
        raise ValueError('Invalid uid.')
    deleted = failed = 0
    for blob in _bucket().list_blobs(prefix='models/'):
        name = getattr(blob, 'name', '')
        if not _ANY_ARTIFACT_RE.match(name or ''):
            continue
        try:
            owner = _verify(name, blob.metadata)
        except ValueError:
            continue
        if not hmac.compare_digest(owner, owner_uid):
            continue
        try:
            blob.delete()
            deleted += 1
        except Exception:
            logger.exception('could not delete model artifact %s', name)
            failed += 1
            continue
        _cache_pop(name)
    return {'deleted': deleted, 'failed': failed}
