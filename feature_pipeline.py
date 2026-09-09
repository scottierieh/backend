"""
feature_pipeline.py — server-side re-implementation of the Feature
Engineering Lab's transform catalog (src/lib/feature-engineering/transforms.ts
on the frontend), as a fittable sklearn transformer.

WHY THIS EXISTS: the frontend's `applyPipeline` is a pure function of
whatever table you hand it — every step (median, mean, box-cox lambda,
category list, min/max, ...) is recomputed from that table on every call.
Called once on the training table, that is a normal fit. Called again on a
handful of new predict-time rows, it would recompute fresh (and wrong)
statistics from just those rows instead of reusing the training-time ones —
a new min/max, a different category-to-code assignment, a different box-cox
lambda. There is nothing to hand-wave here: median/mean/std/min/max, box-cox
and Yeo-Johnson lambdas, and category-to-code maps all have to be fit once
on the training data and re-applied unchanged at predict time, which is
exactly what an sklearn transformer's fit/transform split is for.

FeatureEngineer below fits each configured step in turn (mirroring
transforms.ts's TransformStep semantics and naming exactly — same derived
column names, same shift/lambda search, same category ordering) and stores
per-step parameters; transform() re-applies those stored parameters rather
than recomputing them. It is meant to sit as the first step of the same
sklearn Pipeline models_api.py already builds and persists, ahead of the
existing ColumnTransformer, so one fitted object handles both training-time
and predict-time consistently.

Naming/semantics kept identical to transforms.ts on purpose, so that
`features` (the list of post-engineering column names chosen on the
frontend before this pipeline existed) still resolves to the same columns:
  log            -> f'{col}_log'          (log(x) if x>0 else NaN)
  boxcox         -> f'{col}_boxcox'
  yeojohnson     -> f'{col}_yj'
  onehot         -> f'{col}_{category}'   (<=20 categories, sorted)
  label/ordinal  -> f'{col}_label' / f'{col}_ordinal'
  standard       -> f'{col}_std'
  robust         -> f'{col}_robust'
  minmax         -> f'{col}_minmax'
  polynomial     -> f'{col}^2'
  interaction    -> f'{colA}×{colB}'
  impute_*       -> in place, no new column

A step's `keepSource: False` drops its source column(s) after a *generating*
step only (matching transforms.ts's dropSources: an in-place step like
imputation grows no columns, so there is nothing to drop).
"""

from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

ONEHOT_CAP = 20
LAMBDA_GRID = [round(-2.0 + 0.1 * i, 1) for i in range(41)]  # -2.0 .. 2.0 step 0.1


def _is_missing(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return v == ''
    try:
        return bool(pd.isna(v))
    except (TypeError, ValueError):
        return False


def _numeric_values(s: pd.Series) -> np.ndarray:
    vals = pd.to_numeric(s, errors='coerce').dropna().to_numpy(dtype=float)
    return vals


def _skewness(vals: np.ndarray) -> float:
    if len(vals) == 0:
        return 0.0
    m = vals.mean()
    s = vals.std(ddof=0) or 1.0
    return float(np.mean(((vals - m) / s) ** 3))


def _boxcox_value(x: np.ndarray, lam: float) -> np.ndarray:
    if abs(lam) < 1e-6:
        return np.log(x)
    return (np.power(x, lam) - 1) / lam


def _best_boxcox_lambda(shifted: np.ndarray) -> float:
    best_l, best_score = 1.0, float('inf')
    for l in LAMBDA_GRID:
        score = abs(_skewness(_boxcox_value(shifted, l)))
        if score < best_score:
            best_score, best_l = score, l
    return best_l


def _yeojohnson_value(x: np.ndarray, lam: float) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    if abs(lam) < 1e-6:
        out[pos] = np.log(x[pos] + 1)
    else:
        out[pos] = (np.power(x[pos] + 1, lam) - 1) / lam
    neg = ~pos
    if abs(lam - 2) < 1e-6:
        out[neg] = -np.log(-x[neg] + 1)
    else:
        out[neg] = -((np.power(-x[neg] + 1, 2 - lam) - 1) / (2 - lam))
    return out


def _best_yeojohnson_lambda(vals: np.ndarray) -> float:
    best_l, best_score = 1.0, float('inf')
    for l in LAMBDA_GRID:
        score = abs(_skewness(_yeojohnson_value(vals, l)))
        if score < best_score:
            best_score, best_l = score, l
    return best_l


def _unique_sorted(s: pd.Series) -> list:
    vals = {str(v) for v in s if not _is_missing(v)}
    return sorted(vals)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Fits and re-applies the Feature Engineering Lab's TransformStep
    pipeline. `steps` is the same JSON the frontend already persists on the
    model registry entry (RegisteredModel.pipeline) -- a list of
    {id, kind, columns, keepSource}."""

    def __init__(self, steps: Optional[list[dict]] = None):
        self.steps = steps or []

    def fit(self, X: pd.DataFrame, y=None):
        X = X.copy()
        self.input_columns_ = list(X.columns)
        self.fitted_steps_: list[dict] = []

        for step in self.steps:
            kind = step.get('kind')
            cols = step.get('columns') or []
            if not cols or cols[0] not in X.columns:
                continue
            col = cols[0]
            keep_source = step.get('keepSource')
            keep_source = True if keep_source is None else bool(keep_source)
            before_cols = set(X.columns)
            params: dict = {}

            if kind == 'impute_median':
                nums = _numeric_values(X[col])
                if len(nums):
                    params['value'] = float(np.median(nums))
                    X[col] = X[col].where(~X[col].map(_is_missing), params['value'])
            elif kind == 'impute_mean':
                nums = _numeric_values(X[col])
                if len(nums):
                    params['value'] = float(nums.mean())
                    X[col] = X[col].where(~X[col].map(_is_missing), params['value'])
            elif kind == 'impute_zero':
                params['value'] = 0.0
                X[col] = X[col].where(~X[col].map(_is_missing), 0.0)
            elif kind == 'impute_mode':
                non_missing = X[col][~X[col].map(_is_missing)]
                if len(non_missing):
                    mode = non_missing.astype(str).mode()
                    fill = mode.iloc[0] if len(mode) else None
                    # recover the original (non-stringified) value for that mode
                    match = non_missing[non_missing.astype(str) == fill]
                    params['value'] = match.iloc[0] if len(match) else fill
                    X[col] = X[col].where(~X[col].map(_is_missing), params['value'])

            elif kind == 'log':
                name = f'{col}_log'
                vals = pd.to_numeric(X[col], errors='coerce')
                X[name] = np.where(vals > 0, np.log(vals.where(vals > 0)), np.nan)
            elif kind == 'boxcox':
                nums = _numeric_values(X[col])
                shift = abs(nums.min()) + 1 if len(nums) and nums.min() <= 0 else 0.0
                lam = _best_boxcox_lambda(nums + shift) if len(nums) else 1.0
                params['shift'], params['lambda'] = float(shift), float(lam)
                name = f'{col}_boxcox'
                vals = pd.to_numeric(X[col], errors='coerce')
                shifted = (vals + shift).to_numpy(dtype=float)
                with np.errstate(invalid='ignore', divide='ignore'):
                    transformed = _boxcox_value(shifted, lam)
                X[name] = np.where(vals.notna(), transformed, np.nan)
            elif kind == 'yeojohnson':
                nums = _numeric_values(X[col])
                lam = _best_yeojohnson_lambda(nums) if len(nums) else 1.0
                params['lambda'] = float(lam)
                name = f'{col}_yj'
                vals = pd.to_numeric(X[col], errors='coerce')
                filled = vals.fillna(0).to_numpy(dtype=float)
                transformed = _yeojohnson_value(filled, lam)
                X[name] = np.where(vals.notna(), transformed, np.nan)

            elif kind == 'onehot':
                cats = _unique_sorted(X[col])[:ONEHOT_CAP]
                params['categories'] = cats
                for c in cats:
                    X[f'{col}_{c}'] = (X[col].astype(str) == c).astype(float)
            elif kind in ('label', 'ordinal'):
                cats = _unique_sorted(X[col])
                code_of = {c: i for i, c in enumerate(cats)}
                params['categories'] = cats
                name = f'{col}_{kind}'
                X[name] = X[col].map(lambda v, m=code_of: (
                    np.nan if _is_missing(v) else m.get(str(v), np.nan)
                ))

            elif kind == 'standard':
                nums = _numeric_values(X[col])
                m = float(nums.mean()) if len(nums) else 0.0
                s = float(nums.std(ddof=0)) if len(nums) else 1.0
                s = s or 1.0
                params['mean'], params['std'] = m, s
                name = f'{col}_std'
                vals = pd.to_numeric(X[col], errors='coerce')
                X[name] = (vals - m) / s
            elif kind == 'robust':
                nums = np.sort(_numeric_values(X[col]))
                med = float(np.median(nums)) if len(nums) else 0.0
                iqr = float(np.percentile(nums, 75) - np.percentile(nums, 25)) if len(nums) else 1.0
                iqr = iqr or 1.0
                params['median'], params['iqr'] = med, iqr
                name = f'{col}_robust'
                vals = pd.to_numeric(X[col], errors='coerce')
                X[name] = (vals - med) / iqr
            elif kind == 'minmax':
                nums = _numeric_values(X[col])
                lo = float(nums.min()) if len(nums) else 0.0
                hi = float(nums.max()) if len(nums) else 1.0
                rng = (hi - lo) or 1.0
                params['min'], params['range'] = lo, rng
                name = f'{col}_minmax'
                vals = pd.to_numeric(X[col], errors='coerce')
                X[name] = (vals - lo) / rng

            elif kind == 'polynomial':
                name = f'{col}^2'
                vals = pd.to_numeric(X[col], errors='coerce')
                X[name] = vals ** 2
            elif kind == 'interaction':
                if len(cols) < 2 or cols[1] not in X.columns:
                    continue
                col_b = cols[1]
                name = f'{col}×{col_b}'
                a = pd.to_numeric(X[col], errors='coerce')
                b = pd.to_numeric(X[col_b], errors='coerce')
                X[name] = a * b
            else:
                continue

            after_cols = set(X.columns)
            added = list(after_cols - before_cols)
            if not keep_source and added:
                drop_cols = [c for c in cols if c in X.columns]
                if drop_cols:
                    X = X.drop(columns=drop_cols)

            self.fitted_steps_.append({
                'kind': kind, 'columns': cols, 'keepSource': keep_source, 'params': params,
            })

        self.output_columns_ = list(X.columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for entry in self.fitted_steps_:
            kind, cols, keep_source, params = (
                entry['kind'], entry['columns'], entry['keepSource'], entry['params']
            )
            col = cols[0]
            if col not in X.columns:
                continue
            before_cols = set(X.columns)

            if kind in ('impute_median', 'impute_mean', 'impute_zero', 'impute_mode'):
                if 'value' in params:
                    X[col] = X[col].where(~X[col].map(_is_missing), params['value'])
            elif kind == 'log':
                name = f'{col}_log'
                vals = pd.to_numeric(X[col], errors='coerce')
                X[name] = np.where(vals > 0, np.log(vals.where(vals > 0)), np.nan)
            elif kind == 'boxcox':
                name = f'{col}_boxcox'
                vals = pd.to_numeric(X[col], errors='coerce')
                shifted = (vals + params['shift']).to_numpy(dtype=float)
                with np.errstate(invalid='ignore', divide='ignore'):
                    transformed = _boxcox_value(shifted, params['lambda'])
                X[name] = np.where(vals.notna(), transformed, np.nan)
            elif kind == 'yeojohnson':
                name = f'{col}_yj'
                vals = pd.to_numeric(X[col], errors='coerce')
                filled = vals.fillna(0).to_numpy(dtype=float)
                transformed = _yeojohnson_value(filled, params['lambda'])
                X[name] = np.where(vals.notna(), transformed, np.nan)
            elif kind == 'onehot':
                cats = params.get('categories', [])
                for c in cats:
                    X[f'{col}_{c}'] = (X[col].astype(str) == c).astype(float)
            elif kind in ('label', 'ordinal'):
                cats = params.get('categories', [])
                code_of = {c: i for i, c in enumerate(cats)}
                name = f'{col}_{kind}'
                X[name] = X[col].map(lambda v, m=code_of: (
                    np.nan if _is_missing(v) else m.get(str(v), np.nan)
                ))
            elif kind == 'standard':
                name = f'{col}_std'
                vals = pd.to_numeric(X[col], errors='coerce')
                X[name] = (vals - params['mean']) / params['std']
            elif kind == 'robust':
                name = f'{col}_robust'
                vals = pd.to_numeric(X[col], errors='coerce')
                X[name] = (vals - params['median']) / params['iqr']
            elif kind == 'minmax':
                name = f'{col}_minmax'
                vals = pd.to_numeric(X[col], errors='coerce')
                X[name] = (vals - params['min']) / params['range']
            elif kind == 'polynomial':
                name = f'{col}^2'
                vals = pd.to_numeric(X[col], errors='coerce')
                X[name] = vals ** 2
            elif kind == 'interaction':
                if len(cols) < 2 or cols[1] not in X.columns:
                    continue
                col_b = cols[1]
                name = f'{col}×{col_b}'
                a = pd.to_numeric(X[col], errors='coerce')
                b = pd.to_numeric(X[col_b], errors='coerce')
                X[name] = a * b

            after_cols = set(X.columns)
            added = list(after_cols - before_cols)
            if not keep_source and added:
                drop_cols = [c for c in cols if c in X.columns]
                if drop_cols:
                    X = X.drop(columns=drop_cols)

        # Predict-time rows may lack a column no step ever touches but that
        # this fit never saw missing either -- nothing to do for those; any
        # genuinely required column absent here is caught by the caller
        # checking input_columns_ before calling transform.
        return X
