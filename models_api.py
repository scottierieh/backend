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
import io
import shutil
import tarfile
import tempfile
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
from feature_pipeline import FeatureEngineer
import model_store

router = APIRouter()

Task = Literal['classification', 'regression']

BACKGROUND_SAMPLE_SIZE = 50
BEESWARM_SAMPLE_SIZE = 100
MIN_TRAINING_ROWS = 10
# A score on a handful of rows is noise wearing a decimal point. The AutoML
# screen seals 20% of the table, so this only bites on very small uploads.
MIN_HOLDOUT_ROWS = 20


class TrainRequest(BaseModel):
    data: list[dict[str, Any]]
    algorithm: str
    target: str
    features: list[str]
    task: Task
    # Optional Feature Engineering Lab recipe (RegisteredModel.pipeline on the
    # frontend) -- same {id, kind, columns, keepSource} shape as TransformStep.
    # When present, `features` names columns AFTER this recipe runs (e.g.
    # 'income_log'), and `data` carries the RAW columns the recipe reads from.
    # See feature_pipeline.py for why this has to be fit here rather than
    # trusting whatever the frontend already applied client-side.
    pipeline: Optional[list[dict[str, Any]]] = None
    # Rows to score the trained model on, instead of splitting `data`
    # internally. Model Lab's AutoML screen seals a final test set before it
    # starts, chooses a model without ever sending those rows, and then asks
    # for one score on them -- so the number it reports was not the maximum of
    # thirteen tries on the same split. Same columns as `data`, target
    # included. Omitted keeps the internal 80/20 split, so existing callers
    # are unaffected.
    holdout: Optional[list[dict[str, Any]]] = None

    # ---- AutoGluon ---------------------------------------------------------
    # An explicit field, not a string match on `algorithm`. That field carries
    # a display label ("AutoGluon · WeightedEnsemble_L2") chosen by the screen,
    # and deciding what to fit from what a label happens to start with is the
    # mistake this repo keeps paying for -- 'LightGBM' contains 'gbm', so the
    # registry stopped matching labels and started matching ids for the same
    # reason. Absent means the fourteen-estimator path, exactly as before.
    engine: Optional[str] = None
    # Which model on AutoGluon's leaderboard to serve. The predictor holds all
    # of them; this is the one the reader picked off the board. None serves
    # AutoGluon's own best, which is normally the weighted ensemble.
    agModel: Optional[str] = None
    # Seconds the refit may spend. The save is a real fit, not a copy of the
    # run -- see the branch in train_model for why it is one fit and not two.
    timeLimit: Optional[int] = None
    evalMetric: Optional[str] = None
    preset: Optional[str] = None


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


def _compute_feature_baseline(X: pd.DataFrame, numeric_features: list[str], categorical_features: list[str]) -> dict:
    """Per-feature training-time distribution summary, persisted on the
    ModelRegistryEntry (not the artifact) so Feature Drift (deployment-
    section.tsx's Monitoring tab) can compare it against live prediction-log
    inputs without reloading the model. Deliberately simple — mean/std for
    numeric, top category frequencies for categorical — a first cut, not a
    PSI/KS-test-grade drift statistic."""
    baseline: dict[str, dict] = {}
    for c in numeric_features:
        col = pd.to_numeric(X[c], errors='coerce').dropna()
        if len(col) == 0:
            continue
        baseline[c] = {
            'type': 'numeric',
            'mean': _to_native_type(float(col.mean())),
            'std': _to_native_type(float(col.std(ddof=0))),
        }
    for c in categorical_features:
        counts = X[c].astype(str).value_counts(normalize=True)
        baseline[c] = {
            'type': 'categorical',
            'frequencies': {str(k): _to_native_type(float(v)) for k, v in counts.head(20).items()},
        }
    return baseline


# ---- AutoGluon as a served model -------------------------------------------
#
# A TabularPredictor is a DIRECTORY, and model_store writes a single joblib
# blob. Rather than teach the store about directories -- which would change
# the artifactUri contract and with it the registry, deploy and monitoring
# screens that read it -- the directory is packed into bytes and carried
# inside the artifact dict the store already writes.
#
# clone_for_deployment first, and it is not an optimisation. Measured on the
# example data with best_quality at a 60s budget: 59.5MB of training
# directory becomes 16.3MB of deployment clone, and 4.1MB gzipped. Skipping it
# means every prediction instance downloads the training leftovers too.

AG_BUNDLE_ARCNAME = '.'


def _ag_pack(predictor) -> bytes:
    """A deployment clone of `predictor`, as one gzipped tar."""
    clone_dir = tempfile.mkdtemp(prefix='ag_depl_')
    try:
        # dirs_exist_ok because mkdtemp already made the path; return_clone
        # False because the clone is only wanted on disk, and loading it back
        # here would cost a second load for nothing.
        predictor.clone_for_deployment(path=clone_dir, return_clone=False, dirs_exist_ok=True)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode='w:gz') as tar:
            tar.add(clone_dir, arcname=AG_BUNDLE_ARCNAME)
        return buf.getvalue()
    finally:
        shutil.rmtree(clone_dir, ignore_errors=True)


def _ag_unpack(bundle: bytes):
    """The predictor back out of a bundle, into a directory that stays put.

    The directory is NOT removed: TabularPredictor.load reads from it lazily,
    so deleting it after loading breaks the first prediction rather than the
    load. It lives as long as the instance, which is the same lifetime as the
    artifact cache that holds the predictor.
    """
    out = tempfile.mkdtemp(prefix='ag_serve_')
    with tarfile.open(fileobj=io.BytesIO(bundle), mode='r:gz') as tar:
        # `filter='data'` refuses absolute paths, '..' and device files. The
        # bundle is written by this service, but a tar that is trusted because
        # of where it is expected to come from is how path traversal gets in.
        try:
            tar.extractall(out, filter='data')
        except TypeError:
            # Python < 3.12 has no `filter`; the default is the old behaviour.
            tar.extractall(out)
    from autogluon.tabular import TabularPredictor
    return TabularPredictor.load(out)


def _ag_predictor(artifact: dict):
    """The loaded predictor for an artifact, unpacked once per instance.

    Cached inside the artifact dict itself, which model_store already keeps per
    URI, so a burst of predictions untars once rather than once each.
    """
    loaded = artifact.get('_ag_loaded')
    if loaded is None:
        loaded = _ag_unpack(artifact['bundle'])
        artifact['_ag_loaded'] = loaded
    return loaded


@router.post("/models/{model_id}/train")
def train_model(model_id: str, req: TrainRequest):
    if not req.data:
        _fail(400, "No training data provided")
    if req.target not in (req.data[0].keys() if req.data else []):
        _fail(400, f"Target column '{req.target}' not found in data")
    df = pd.DataFrame(req.data)
    if req.target not in df.columns:
        _fail(400, f"Target column '{req.target}' not found in data")

    feature_engineer: Optional[FeatureEngineer] = None
    raw_baseline: Optional[dict] = None
    if req.pipeline:
        # `features` names columns the recipe produces (e.g. 'income_log'),
        # not columns present in the raw `data` -- validated after the recipe
        # runs, below, instead of here. Fit on the feature columns only: the
        # target must never be an input the recipe can touch, and excluding
        # it here is also what keeps feature_engineer.input_columns_ (below,
        # persisted as `raw_columns`) from demanding the target back at
        # predict time, when no caller has it.
        feature_engineer = FeatureEngineer(req.pipeline)
        raw_features = df.drop(columns=[req.target], errors='ignore')
        try:
            feature_engineer.fit(raw_features)
            engineered = feature_engineer.transform(raw_features)
        except Exception as e:
            _fail(422, f"Feature pipeline failed: {e}")
        # Kept for the response. /predict demands rows in these columns -- the
        # recipe's INPUTS -- and the caller has no way to know what they are:
        # `features` names the recipe's outputs ('income_log', 'city_seoul'),
        # and asking for those at predict time earns a 400 naming the raw
        # columns it wanted instead. Any model with a recipe was unservable
        # for exactly that reason.
        raw_numeric, raw_categorical = _split_feature_types(
            raw_features, list(raw_features.columns))
        raw_baseline = _compute_feature_baseline(raw_features, raw_numeric, raw_categorical)
        engineered[req.target] = df[req.target]
        df = engineered

    missing = [f for f in req.features if f not in df.columns]
    if missing:
        _fail(400, f"Feature column(s) not found: {missing}")

    # ---- AutoGluon -------------------------------------------------------
    #
    # A different shape of save, for a reason worth stating. The fourteen
    # estimators are fitted twice here on purpose: once on a train split to
    # score honestly, then again on every valid row for the model actually
    # served. Each fit costs under a second, so the second one is free.
    #
    # An AutoGluon fit costs the caller's whole budget. Doing it twice turns a
    # 300-second save into a 600-second one, and buys less than it does for a
    # single estimator: AutoGluon already holds out its own validation split
    # inside the rows it is given, and its ensemble weights are fitted on that
    # split. So it is fitted ONCE, on the training rows, and that same
    # predictor is what gets served. The caller's sealed holdout, when sent,
    # scores that predictor without ever having been part of it -- which is
    # the claim the two-stage split exists to make, and it survives.
    if (req.engine or '').lower() == 'autogluon':
        try:
            from autogluon.tabular import TabularPredictor
        except ImportError:
            _fail(503, 'autogluon.tabular is not installed on the server.')

        ag_metric = req.evalMetric or ('accuracy' if req.task == 'classification' else 'root_mean_squared_error')
        ag_frame = df[req.features + [req.target]].dropna(subset=[req.target]).reset_index(drop=True)
        work_dir = tempfile.mkdtemp(prefix='ag_train_')
        try:
            try:
                predictor = TabularPredictor(
                    label=req.target,
                    problem_type=('regression' if req.task == 'regression' else None),
                    eval_metric=ag_metric,
                    path=work_dir,
                    verbosity=0,
                ).fit(
                    ag_frame,
                    time_limit=int(req.timeLimit or 300),
                    presets=req.preset or 'best_quality',
                    excluded_model_types=['NN_TORCH', 'FASTAI'],
                )
            except Exception as e:
                _fail(422, f'AutoGluon training failed: {e}')

            # The model the reader picked off the board. An unknown name is the
            # caller's mistake and is worth saying so, rather than quietly
            # serving a different model than the one they chose.
            available = list(predictor.model_names())
            ag_model = req.agModel if req.agModel in available else None
            if req.agModel and ag_model is None:
                _fail(400, f"Model '{req.agModel}' is not on this predictor. Have: {available}")

            metrics: dict[str, float] = {}
            if req.holdout:
                h = pd.DataFrame(req.holdout)
                if req.target not in h.columns:
                    _fail(400, f"Holdout rows are missing the target column '{req.target}'")
                if feature_engineer is not None:
                    try:
                        h_eng = feature_engineer.transform(h.drop(columns=[req.target], errors='ignore'))
                    except Exception as e:
                        _fail(422, f'Feature pipeline failed on holdout rows: {e}')
                    h_eng[req.target] = h[req.target]
                    h = h_eng
                h_missing = [f for f in req.features if f not in h.columns]
                if h_missing:
                    _fail(400, f'Holdout rows are missing feature column(s): {h_missing}')
                h = h[req.features + [req.target]].dropna(subset=[req.target])
                if len(h) >= MIN_HOLDOUT_ROWS:
                    try:
                        scored = predictor.evaluate(h, silent=True)
                        metrics = {k: _to_native_type(float(v)) for k, v in scored.items()
                                   if isinstance(v, (int, float)) and not isinstance(v, bool)}
                        metrics['n_eval'] = float(len(h))
                    except Exception:
                        pass  # best-effort, same as the estimator path

            bundle = _ag_pack(predictor)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

        artifact = {
            'kind': 'autogluon',
            'bundle': bundle,
            'ag_model': ag_model,
            'task': req.task,
            'features': req.features,
            'feature_engineer': feature_engineer,
            'raw_columns': feature_engineer.input_columns_ if feature_engineer else req.features,
            # Read by /predict to turn a probability frame into a fixed column
            # order the caller can rely on, rather than whatever the frame
            # happens to be ordered by.
            'class_labels': [_to_native_type(c) for c in (predictor.class_labels or [])]
                            if req.task == 'classification' else None,
        }
        try:
            artifact_uri = model_store.save_pipeline(model_id, artifact)
        except Exception as e:
            _fail(502, f'Could not save model artifact: {e}')

        return {
            'artifactUri': artifact_uri,
            'metrics': metrics or None,
            # Not computed for this engine yet. The SHAP path wraps
            # predict_proba and is model-agnostic, so it CAN be done here; it
            # is a separate change with its own cost (a background sample and
            # an explainer over a predictor that is slower per call than one
            # estimator), and reporting a beeswarm this response did not
            # compute would be worse than reporting none.
            'shapBeeswarm': None,
            'featureBaseline': _compute_feature_baseline(
                ag_frame[req.features], *_split_feature_types(ag_frame, req.features)),
            'rawColumns': artifact['raw_columns'] if feature_engineer else None,
            'rawBaseline': raw_baseline if feature_engineer else None,
            'evaluatedOn': 'holdout' if metrics.get('n_eval') else 'internal_split',
        }


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
    holdout_used = False
    try:
        if req.holdout:
            # Caller supplied the evaluation rows. They go through the same
            # recipe and the same _prepare_xy as training, or the score would
            # be measured on differently shaped data than the model saw.
            hdf = pd.DataFrame(req.holdout)
            if req.target not in hdf.columns:
                _fail(400, f"Holdout rows are missing the target column '{req.target}'")
            if feature_engineer is not None:
                raw_holdout = hdf.drop(columns=[req.target], errors='ignore')
                try:
                    engineered_holdout = feature_engineer.transform(raw_holdout)
                except Exception as e:
                    _fail(422, f"Feature pipeline failed on holdout rows: {e}")
                engineered_holdout[req.target] = hdf[req.target]
                hdf = engineered_holdout
            h_missing = [f for f in req.features if f not in hdf.columns]
            if h_missing:
                _fail(400, f"Holdout rows are missing feature column(s): {h_missing}")
            X_test, y_test, _ = _prepare_xy(hdf, req.target, req.features, req.task, numeric_features)
            if len(X_test) < MIN_HOLDOUT_ROWS:
                _fail(400, f"Not enough valid holdout rows to score ({len(X_test)}, need >= {MIN_HOLDOUT_ROWS})")
            X_train, y_train = X, y
            holdout_used = True
        else:
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
            # The AutoML leaderboard ranks an imbalanced target on macro F1, so
            # a final score reported in a different average would not be
            # comparable to the ranking it is meant to confirm.
            metrics['f1_macro'] = _to_native_type(f1_score(y_test, y_pred, average='macro'))
        else:
            metrics['r2'] = _to_native_type(r2_score(y_test, y_pred))
            metrics['mae'] = _to_native_type(mean_absolute_error(y_test, y_pred))
        metrics['n_eval'] = float(len(X_test))
    except HTTPException:
        raise  # a bad holdout is the caller's error, not a metric we can skip
    except Exception:
        pass  # best-effort — register-model.ts falls back to the run's own metrics if this is empty

    # ...then a fresh fit on ALL valid rows for the model actually served.
    pipeline = _build_pipeline(req.algorithm, req.task, numeric_features, categorical_features)
    try:
        pipeline.fit(X, y)
    except Exception as e:
        _fail(422, f"Training failed: {e}")

    shap_beeswarm = _compute_beeswarm(pipeline, req.task, X)
    feature_baseline = _compute_feature_baseline(X, numeric_features, categorical_features)

    background_n = min(len(X), BACKGROUND_SAMPLE_SIZE)
    background = X.sample(n=background_n, random_state=42) if len(X) > background_n else X

    artifact = {
        'pipeline': pipeline,
        'label_encoder': label_encoder,
        'task': req.task,
        'features': req.features,
        'background': background,
        # None when no Feature Engineering recipe was configured -- predict
        # then reads `features` straight off incoming rows, same as before
        # this existed.
        'feature_engineer': feature_engineer,
        'raw_columns': feature_engineer.input_columns_ if feature_engineer else req.features,
    }
    try:
        artifact_uri = model_store.save_pipeline(model_id, artifact)
    except Exception as e:
        _fail(502, f"Could not save model artifact: {e}")

    return {
        'artifactUri': artifact_uri, 'metrics': metrics or None, 'shapBeeswarm': shap_beeswarm,
        'featureBaseline': feature_baseline,
        # The columns /predict will demand, and their distributions, so the
        # caller can build an input form for the right columns. Absent when
        # there was no recipe, and then `features` already is that list.
        'rawColumns': artifact['raw_columns'] if feature_engineer else None,
        'rawBaseline': raw_baseline if feature_engineer else None,
        # Lets the caller tell "scored on the rows I sealed" from "scored on an
        # internal split", which are not the same claim.
        'evaluatedOn': 'holdout' if holdout_used else 'internal_split',
    }


@router.post("/models/{model_id}/predict")
def predict_model(model_id: str, req: PredictRequest):
    if not req.rows:
        _fail(400, "No rows to predict")
    try:
        artifact = model_store.load_pipeline(req.artifactUri)
    except Exception as e:
        _fail(404, f"Could not load model artifact: {e}")

    # AutoGluon artifacts carry a packed predictor rather than an sklearn
    # Pipeline. Everything below this branch -- the label encoder, the
    # ColumnTransformer, the SHAP background -- belongs to the estimator path
    # and has no counterpart here, so this answers and returns rather than
    # trying to share it.
    if artifact.get('kind') == 'autogluon':
        raw_columns: list[str] = artifact.get('raw_columns') or artifact.get('features') or []
        rows_df = pd.DataFrame(req.rows)
        missing = [f for f in raw_columns if f not in rows_df.columns]
        if missing:
            _fail(400, f"Row(s) missing required feature(s): {missing}")

        fe: Optional[FeatureEngineer] = artifact.get('feature_engineer')
        X = fe.transform(rows_df[raw_columns]) if fe is not None else rows_df[raw_columns]
        features: list[str] = artifact.get('features') or list(X.columns)
        missing_engineered = [f for f in features if f not in X.columns]
        if missing_engineered:
            _fail(422, f"Feature pipeline did not produce required column(s): {missing_engineered}")
        X = X[features]

        try:
            predictor = _ag_predictor(artifact)
        except Exception as e:
            _fail(422, f'Could not load the AutoGluon predictor: {e}')

        # `model=None` serves AutoGluon's own best, which is what a caller who
        # never picked one gets.
        ag_model = artifact.get('ag_model')
        try:
            predictions = [_to_native_type(v) for v in predictor.predict(X, model=ag_model).tolist()]
        except Exception as e:
            _fail(422, f'Prediction failed: {e}')

        probabilities = None
        if artifact.get('task') == 'classification':
            try:
                proba = predictor.predict_proba(X, model=ag_model)
                # Column order pinned to what was stored at train time rather
                # than to whatever this frame is ordered by: the caller reads
                # position i as class i and has nothing else to go on, so an
                # order that moves between calls silently relabels every
                # probability. Only applied when the stored labels name exactly
                # this frame's columns -- a mismatch means the assumption is
                # wrong, and reordering on a wrong assumption is worse than
                # leaving the frame alone.
                labels = artifact.get('class_labels') or []
                if labels and set(map(str, labels)) == set(map(str, proba.columns)):
                    by_str = {str(c): c for c in proba.columns}
                    proba = proba[[by_str[str(v)] for v in labels]]
                probabilities = [[_to_native_type(float(v)) for v in row] for row in proba.values]
            except Exception:
                probabilities = None

        return {
            'predictions': predictions,
            'probabilities': probabilities,
            # Row-level SHAP over a predictor is a separate change -- see the
            # note on shapBeeswarm in train_model. None is what the contract
            # already means by "this engine does not report it".
            'shapContributions': None,
        }


    pipeline: Pipeline = artifact['pipeline']
    label_encoder: Optional[LabelEncoder] = artifact.get('label_encoder')
    task: Task = artifact['task']
    features: list[str] = artifact['features']
    background: pd.DataFrame = artifact.get('background')
    feature_engineer: Optional[FeatureEngineer] = artifact.get('feature_engineer')
    raw_columns: list[str] = artifact.get('raw_columns') or features

    rows_df = pd.DataFrame(req.rows)
    missing = [f for f in raw_columns if f not in rows_df.columns]
    if missing:
        _fail(400, f"Row(s) missing required feature(s): {missing}")

    # A model trained with a Feature Engineering recipe expects rows in that
    # recipe's RAW columns, not its output columns -- re-derive the same
    # engineered columns here, with the training-time fitted parameters
    # (medians, box-cox lambdas, category codes, ...) feature_engineer already
    # holds, rather than the caller trying to reproduce them.
    engineered = feature_engineer.transform(rows_df[raw_columns]) if feature_engineer is not None else rows_df
    missing_engineered = [f for f in features if f not in engineered.columns]
    if missing_engineered:
        _fail(422, f"Feature pipeline did not produce required column(s): {missing_engineered}")
    X = engineered[features]

    try:
        raw_pred = pipeline.predict(X)
    except Exception as e:
        _fail(422, f"Prediction failed: {e}")

    probabilities = None
    if task == 'classification':
        predictions = list(label_encoder.inverse_transform(raw_pred)) if label_encoder is not None else list(raw_pred)
        try:
            proba = pipeline.predict_proba(X)
            # Every row's FULL per-class distribution, not just the winning
            # class's own probability -- predict-client.ts already declares
            # `probabilities: number[][]` and every consumer (deploy-section's
            # Math.max(...probs), explain-section's p[p.length-1]) already
            # reads it as one array per row. Collapsing to a single float here
            # made those correct only by accident for two classes (row.max()
            # is trivially the winning probability regardless of class count,
            # but downstream code expects to index the array), and wrong for
            # three or more. Column order matches pipeline.predict_proba's own
            # (== label_encoder.classes_, sorted), unchanged from before.
            probabilities = [[_to_native_type(float(v)) for v in row] for row in proba]
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
