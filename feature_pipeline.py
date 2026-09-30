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

import datetime as dt
import re
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


def _is_numeric_column(s: pd.Series) -> bool:
    """Is this column numbers? Judged on the values, the way the frontend
    engine judges it (transforms.ts isNumericColumn): every non-missing value
    has to be a number. A column of numeric-looking STRINGS is not numeric on
    either side, so the two agree about which of a pair is the group."""
    seen = False
    for v in s:
        if _is_missing(v):
            continue
        if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)):
            return False
        seen = True
    return seen


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


# Only ISO-ish YYYY-MM-DD (optional time) and YYYY/MM/DD. pandas' own parser
# reads "3" and a product code as dates, which would turn any short-string
# column into five columns of nonsense; the frontend engine uses this exact
# pattern (transforms.ts DATE_RE) and the two must agree character for
# character, or a recipe saved on the screen produces different columns here.
_DATE_RE = re.compile(r'^(\d{4})[-/](\d{2})[-/](\d{2})([T ](\d{2}):(\d{2})(:\d{2})?)?')

_DATE_PARTS = ('year', 'month', 'day', 'dow', 'hour')


def _parse_date(v: Any):
    if isinstance(v, (dt.datetime, dt.date)):
        return dt.datetime(v.year, v.month, v.day,
                           getattr(v, 'hour', 0), getattr(v, 'minute', 0))
    if not isinstance(v, str):
        return None
    m = _DATE_RE.match(v.strip())
    if not m:
        return None
    try:
        return dt.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                           int(m.group(5) or 0), int(m.group(6) or 0))
    except ValueError:
        return None


def _date_part(d: dt.datetime, part: str) -> int:
    if part == 'year':
        return d.year
    if part == 'month':
        return d.month
    if part == 'day':
        return d.day
    if part == 'dow':
        # JavaScript's getUTCDay() is 0=Sunday; Python's weekday() is
        # 0=Monday. Matching JS here is not a style choice -- the same recipe
        # has to put the same number in the same column on both sides.
        return (d.weekday() + 1) % 7
    if part == 'hour':
        return d.hour
    return 0


def _text_length(s: str) -> int:
    # JavaScript's String.length counts UTF-16 code units, Python's len()
    # counts code points. They agree for every character in the BMP and differ
    # for emoji and other astral characters, so this counts the way the
    # frontend does rather than the way Python would.
    return len(s.encode('utf-16-le')) // 2


# Enough components to carry this much of the variance between them. The
# frontend engine uses the same constant (transforms.ts PCA_VARIANCE_TARGET);
# a different number on the two sides is a different number of columns.
_PCA_VARIANCE_TARGET = 0.95


# Target encoding. Both numbers are shared with the frontend engine
# (transforms.ts TARGET_ENCODE_FOLDS / TARGET_ENCODE_SMOOTHING); a different
# value on either side is a different column.
#
# Replacing a category with the target's mean in that category puts the answer
# into the question. Fitted on the training rows and applied to those same rows,
# every row's feature contains that row's own label -- a level seen once becomes
# exactly its label, and a model learns to read it back. So a fitting row's
# value is computed from the OTHER folds only, which is what makes
# fit_transform() and fit().transform() different here. sklearn's own
# TargetEncoder documents the same asymmetry for the same reason.
#
# The folds are position modulo K rather than a shuffle: the two engines must
# produce the same folds and cannot share a random number generator, and
# interleaving spreads a target-sorted table where contiguous blocks would not.
_TARGET_ENCODE_FOLDS = 5

# The m in the m-estimate `(n*mean + m*prior) / (n + m)`: how many rows a level
# needs before its own average outweighs the overall one. Not sklearn's
# smooth='auto' (an empirical-Bayes shrinkage on each level's variance), which
# is a better estimator and a worse thing to implement twice -- a stated
# constant is explainable in one sentence and cannot drift between engines.
_TARGET_ENCODE_SMOOTHING = 20


def _prepare_target_y(y, task):
    """The target as numbers, by the rule transforms.ts follows.

    Regression takes the value. Classification needs exactly two classes: the
    distinct labels sorted as strings, and the LAST one counts as 1. Which of
    the two that is does not matter to a model (`p` and `1 - p` carry the same
    information); which rule is used does, because both engines have to pick
    the same one.

    More than two classes returns None. One column cannot hold "how does the
    target behave here" when the target has three answers, and declining is
    better than averaging class codes as if they were a quantity.
    """
    if y is None:
        return None
    vals = list(y)
    if task == 'regression':
        nums = pd.to_numeric(pd.Series(vals), errors='coerce')
        return [None if pd.isna(v) else float(v) for v in nums]
    labels = sorted({str(v) for v in vals if not _is_missing(v)})
    if len(labels) != 2:
        return None
    return [None if _is_missing(v) else (1.0 if str(v) == labels[1] else 0.0)
            for v in vals]


def _te_smoothed(total, n, prior):
    m = _TARGET_ENCODE_SMOOTHING
    return (total + m * prior) / (n + m)


def _te_sums(levels, ys, admit):
    """Per-level totals over the rows `admit` accepts, plus their prior."""
    sums = {}
    total = 0.0
    count = 0
    for i, yv in enumerate(ys):
        if yv is None or not admit(i):
            continue
        total += yv
        count += 1
        key = levels[i]
        if key is None:
            continue
        hit = sums.get(key)
        if hit:
            hit[0] += yv
            hit[1] += 1
        else:
            sums[key] = [yv, 1]
    return sums, (total / count if count else 0.0), count > 0


def _fix_sign(vec: np.ndarray) -> np.ndarray:
    """An eigenvector and its negative describe the same axis, and which one
    an implementation returns is arbitrary -- so two implementations of PCA
    agree about the axes and disagree about the SIGN of every score, silently.
    Both engines fix it the same way: largest-magnitude loading positive.
    (sklearn's own `svd_flip` exists for this reason.)"""
    at = int(np.argmax(np.abs(vec)))
    return -vec if vec[at] < 0 else vec


def _pca_names(headers, k: int) -> list:
    """`pc1, pc2, ...`, moved out of the way of anything already there.

    The suffix belongs to the STEP, not to each name: naming them one at a
    time gave a second PCA `pc1_2, pc2` -- its second component called `pc2`,
    which reads as the first PCA's second component and is a different
    variable. transforms.ts applies the same rule, so the two engines agree
    about which run a column belongs to."""
    taken = set(headers)
    suffix = 1
    while True:
        names = [f'pc{i}' if suffix == 1 else f'pc{i}_{suffix}' for i in range(1, k + 1)]
        if not any(n in taken for n in names):
            return names
        suffix += 1


def _unique_sorted(s: pd.Series) -> list:
    vals = {str(v) for v in s if not _is_missing(v)}
    return sorted(vals)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Fits and re-applies the Feature Engineering Lab's TransformStep
    pipeline. `steps` is the same JSON the frontend already persists on the
    model registry entry (RegisteredModel.pipeline) -- a list of
    {id, kind, columns, keepSource}."""

    def __init__(self, steps: Optional[list[dict]] = None, task: Optional[str] = None):
        self.steps = steps or []
        # Only `target_encode` reads this, and it is a constructor parameter
        # rather than an argument to fit() so that get_params/clone carry it and
        # the pickled engineer remembers what it was fitted for.
        self.task = task

    def fit(self, X: pd.DataFrame, y=None):
        self._fit_pass(X, y)
        return self

    def fit_transform(self, X: pd.DataFrame, y=None, **fit_params) -> pd.DataFrame:
        """Fit, and return the FITTING rows transformed.

        Deliberately not `fit(X, y).transform(X)`. With a `target_encode` step
        those two differ: this one gives each row its out-of-fold value, and
        transform() gives it the lookup built from every training row -- which
        for these rows includes their own labels. Anything that TRAINS on the
        fitting rows wants this one. (sklearn's TargetEncoder draws the same
        distinction, and a sklearn Pipeline calls fit_transform on its
        intermediate steps, so this is also the behaviour a Pipeline expects.)
        """
        return self._fit_pass(X, y)

    def _fit_pass(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
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
            elif kind == 'winsorize':
                vals = _numeric_values(X[col])
                if len(vals) >= 4:
                    # numpy's default quantile is linear interpolation, which
                    # is what the browser engine's quantile() does and what
                    # data_profile_analysis.py used to measure the outliers
                    # this step is acting on.
                    q1 = float(np.quantile(vals, 0.25))
                    q3 = float(np.quantile(vals, 0.75))
                    iqr = q3 - q1
                    # A column whose middle half is one value has no spread to
                    # measure a tail against. The profile counts zero outliers
                    # here too, and this learns nothing rather than clipping
                    # the column down to its median.
                    if iqr > 0:
                        params['lower'] = q1 - 1.5 * iqr
                        params['upper'] = q3 + 1.5 * iqr
                        X[col] = pd.to_numeric(X[col], errors='coerce').clip(
                            params['lower'], params['upper'])
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
            elif kind == 'date_parts':
                parsed = X[col].map(_parse_date)
                seen = parsed.dropna()
                # Only the parts that vary. A file covering one month would
                # otherwise get a `month` column holding one value.
                parts = [p for p in _DATE_PARTS
                         if seen.map(lambda d, _p=p: _date_part(d, _p)).nunique() > 1] if len(seen) else []
                params['parts'] = parts
                for p in parts:
                    X[f'{col}_{p}'] = parsed.map(
                        lambda d, _p=p: _date_part(d, _p) if d is not None else np.nan)
            elif kind == 'text_stats':
                text = X[col].map(lambda v: None if _is_missing(v) else str(v))
                X[f'{col}_len'] = text.map(lambda s: np.nan if s is None else _text_length(s))
                X[f'{col}_words'] = text.map(
                    lambda s: np.nan if s is None else (0 if not s.strip() else len(s.split())))
            elif kind == 'pca':
                usable = [c for c in cols if c in X.columns and _is_numeric_column(X[c])]
                if len(usable) < 2:
                    continue
                block = X[usable].apply(pd.to_numeric, errors='coerce')
                # Complete rows only: a blank would otherwise contribute a
                # zero to that column's centred value, which is a data point
                # the data does not contain.
                complete = block.dropna()
                if len(complete) < len(usable) + 1:
                    continue
                centre = complete.mean(axis=0).to_numpy(dtype=float)
                # The correlation matrix, not the covariance one. On raw units
                # the first component is whichever column happens to be
                # measured in the largest numbers.
                scale = complete.std(axis=0, ddof=0).to_numpy(dtype=float)
                scale = np.where(scale == 0, 1.0, scale)
                z = (complete.to_numpy(dtype=float) - centre) / scale
                cov = np.cov(z, rowvar=False, ddof=1)
                values, vectors = np.linalg.eigh(np.atleast_2d(cov))
                order = np.argsort(values)[::-1]
                values = values[order]
                vectors = vectors[:, order]
                total = float(np.clip(values, 0, None).sum()) or 1.0

                carried, keep = 0.0, 0
                while keep < len(values) and carried < _PCA_VARIANCE_TARGET:
                    carried += float(max(values[keep], 0.0)) / total
                    keep += 1

                kept = [_fix_sign(vectors[:, i]) for i in range(keep)]
                params.update({
                    'pcaCols': usable,
                    'pcaCentre': [float(v) for v in centre],
                    'pcaScale': [float(v) for v in scale],
                    'pcaVectors': [[float(w) for w in vec] for vec in kept],
                    'pcaExplained': [float(max(values[i], 0.0)) / total for i in range(keep)],
                })
                names = _pca_names(X.columns, keep)
                zi = (block.to_numpy(dtype=float) - centre) / scale
                scores = zi @ np.array(kept).T if keep else np.empty((len(X), 0))
                for i, name in enumerate(names):
                    X[name] = scores[:, i]
            elif kind == 'target_encode':
                # The only step that reads the target, and it reads it from `y`
                # rather than from a column: models_api fits the recipe on the
                # feature columns with the target dropped, so a step looking for
                # a column would find nothing and silently do nothing.
                ys = _prepare_target_y(y, self.task)
                if ys is None or len(ys) != len(X):
                    continue
                levels = [None if _is_missing(v) else str(v) for v in X[col].tolist()]
                sums, prior, has_y = _te_sums(levels, ys, lambda i: True)
                if not has_y:
                    continue
                params.update({
                    'teMeans': {k: _te_smoothed(t, n, prior) for k, (t, n) in sums.items()},
                    'tePrior': prior,
                })
                # One pass per fold, not one per row: a fold's complement is the
                # same set for every row in it.
                k_folds = _TARGET_ENCODE_FOLDS
                folds = [_te_sums(levels, ys, lambda i, f=f: i % k_folds != f)
                         for f in range(k_folds)]
                oof = []
                for i, key in enumerate(levels):
                    f_sums, f_prior, f_has = folds[i % k_folds]
                    if not f_has:
                        # Nothing outside this row's fold is labelled; the
                        # overall prior is the most that can be said.
                        oof.append(prior)
                    elif key is None:
                        oof.append(f_prior)
                    else:
                        hit = f_sums.get(key)
                        # A level appearing only inside this row's own fold gets
                        # the prior: the alternative is its own rows' mean,
                        # which is the leak.
                        oof.append(_te_smoothed(hit[0], hit[1], f_prior) if hit else f_prior)
                X[f'{col}_te'] = np.asarray(oof, dtype=float)
            elif kind == 'group_stats':
                if len(cols) < 2 or cols[1] not in X.columns:
                    continue
                col_b = cols[1]
                # The roles come from the TYPES, not from the order the two
                # were picked in -- the frontend decides the same way, and a
                # pair that is two numbers or two categories is no pair.
                a_num = _is_numeric_column(X[col])
                b_num = _is_numeric_column(X[col_b])
                if a_num == b_num:
                    continue
                group_col = col_b if a_num else col
                value_col = col if a_num else col_b
                values = pd.to_numeric(X[value_col], errors='coerce')
                groups = X[group_col].map(lambda v: None if _is_missing(v) else str(v))
                ok = values.notna() & groups.notna()
                if not ok.any():
                    continue
                means = values[ok].groupby(groups[ok]).mean()
                overall = float(values[ok].mean())
                params.update({
                    'groupCol': group_col, 'valueCol': value_col,
                    'groupMeans': {str(k): float(v) for k, v in means.items()},
                    'overall': overall,
                })
                mean_name = f'{value_col}_by_{group_col}_mean'
                diff_name = f'{value_col}_by_{group_col}_diff'
                mapped = groups.map(lambda g: params['groupMeans'].get(g, overall)
                                    if g is not None else overall)
                X[mean_name] = mapped.astype(float)
                X[diff_name] = values - mapped.astype(float)
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
        return X

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
            elif kind == 'winsorize':
                if 'lower' in params and 'upper' in params:
                    # In place, like imputation: a clipped value is the same
                    # variable with its tail pulled in.
                    X[col] = pd.to_numeric(X[col], errors='coerce').clip(
                        params['lower'], params['upper'])
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
            elif kind == 'date_parts':
                parsed = X[col].map(_parse_date)
                for p in params.get('parts', []):
                    X[f'{col}_{p}'] = parsed.map(
                        lambda d, _p=p: _date_part(d, _p) if d is not None else np.nan)
            elif kind == 'text_stats':
                text = X[col].map(lambda v: None if _is_missing(v) else str(v))
                X[f'{col}_len'] = text.map(lambda s: np.nan if s is None else _text_length(s))
                X[f'{col}_words'] = text.map(
                    lambda s: np.nan if s is None else (0 if not s.strip() else len(s.split())))
            elif kind == 'pca':
                pca_cols = params.get('pcaCols') or []
                vectors = params.get('pcaVectors') or []
                if not pca_cols or not vectors or not all(c in X.columns for c in pca_cols):
                    continue
                centre = np.array(params['pcaCentre'], dtype=float)
                scale = np.array(params['pcaScale'], dtype=float)
                block = X[pca_cols].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
                # A row missing any input scores null rather than a number
                # computed from a centre standing in for the value: the
                # component is a weighted sum of ALL the inputs, and one
                # absent makes it a different sum, not a slightly worse one.
                z = (block - centre) / np.where(scale == 0, 1.0, scale)
                scores = z @ np.array(vectors, dtype=float).T
                for i, name in enumerate(_pca_names(X.columns, len(vectors))):
                    X[name] = scores[:, i]
            elif kind == 'target_encode':
                te_means = params.get('teMeans')
                prior = params.get('tePrior')
                if te_means is None or prior is None:
                    continue
                # An unseen level and a missing one say the same thing --
                # nothing is known about this row's category -- and the prior is
                # what that says. A null would drop the row at predict time.
                X[f'{col}_te'] = X[col].map(
                    lambda v, m=te_means, p=float(prior): (
                        p if _is_missing(v) else float(m.get(str(v), p)))).astype(float)
            elif kind == 'group_stats':
                group_col = params.get('groupCol')
                value_col = params.get('valueCol')
                if not group_col or not value_col or group_col not in X.columns:
                    continue
                means = params.get('groupMeans') or {}
                overall = float(params.get('overall', 0.0))
                groups = X[group_col].map(lambda v: None if _is_missing(v) else str(v))
                # A group unseen in training falls back to the overall
                # training mean. A null here would drop the row at predict
                # time, and the overall mean is what "nothing known about this
                # group" actually says.
                mapped = groups.map(lambda g: means.get(g, overall) if g is not None else overall)
                values = (pd.to_numeric(X[value_col], errors='coerce')
                          if value_col in X.columns else pd.Series(np.nan, index=X.index))
                X[f'{value_col}_by_{group_col}_mean'] = mapped.astype(float)
                X[f'{value_col}_by_{group_col}_diff'] = values - mapped.astype(float)

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
