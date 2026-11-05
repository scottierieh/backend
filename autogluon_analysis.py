"""
autogluon_analysis.py — AutoGluon as one more Auto Compare engine.

Route: /api/analysis/autogluon. Design: statistica-frontend
docs/automl-autogluon-design.md.

WHY THIS EXISTS, in one paragraph. The other fourteen scripts each train one
algorithm with library defaults, and the frontend fans out to all fourteen and
ranks what comes back. Nothing tunes, nothing ensembles, and every correctness
fix -- the train/holdout split order, class weighting, the row counts -- had to
be made fourteen times. AutoGluon does tuning, bagging, stacking and a weighted
ensemble behind one `fit()`, and its leaderboard has exactly the shape this
app's Compare screen already draws: several models, ranked. So this script is
one request that answers with many rows, rather than many requests that answer
with one each.

WHAT IT DOES NOT DO. It does not split off the final-test rows: the caller
already sealed those in the browser and sends only its training half, so
AutoGluon never sees them and the app's two-stage discipline survives. It does
not preprocess -- that is the point of using AutoGluon -- beyond dropping rows
with no target, which are unscoreable either way and have to be counted before
anything else touches them.

CLI contract, same as every *_analysis.py here: one JSON object on stdin, one
JSON object on stdout; on failure print {"error": ...} to stderr and exit(1).
"""

import os
import sys
import json
import time
import shutil
import tempfile
import contextlib

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    accuracy_score, f1_score, average_precision_score,
    precision_score, recall_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix as sk_confusion_matrix,
)


def _to_native_type(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


def _finite(x):
    """None for anything that would serialise as NaN/Infinity -- the frontend
    reads a missing metric as 'not reported' and draws an em dash, which is
    true, where NaN would print as a number that is not one."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


# The caller names the metric it will rank on, and AutoGluon optimises that same
# metric. Ranking on one measure while searching on another is how tune_analysis
# ended up hill-climbing weighted F1 for a board sorted by PR-AUC; this mapping
# is the one place that stays in step with automl-client.ts's rankingPolicy.
#
# Keys are AutoGluon/sklearn scorer names; values are the ModelMetrics field the
# frontend reads (src/lib/types/model-result.ts).
_METRIC_FIELD = {
    'accuracy': 'accuracy',
    'f1': 'f1',
    'f1_macro': 'f1',
    'f1_weighted': 'f1',
    'roc_auc': 'auc',
    'average_precision': 'pr_auc',
    'r2': 'r2',
    'root_mean_squared_error': 'rmse',
    'mean_absolute_error': 'mae',
}

# Regression scorers AutoGluon reports as negated (higher is better), which the
# frontend's own columns do not -- it prints RMSE and MAE as errors and sorts
# them ascending itself.
_NEGATED = {'root_mean_squared_error', 'mean_absolute_error'}

_DEFAULT_METRIC = {'classification': 'accuracy', 'regression': 'root_mean_squared_error'}


# AutoGluon's default lineup includes two torch models. requirements.txt
# deliberately carries the GBM extras only -- the torch wheels are large, the
# service has no GPU, and on tabular data of this size the boosted trees win
# anyway -- so asking for them spends part of the budget failing to import and
# then prints nine lines about it. Naming them here is how the exclusion stays
# a decision rather than a side effect of what happens to be installed.
_NO_TORCH = ['NN_TORCH', 'FASTAI']

# Sealed (score, outcome) pairs. A pair costs about 13 bytes of JSON.
#
# Two limits, because one is not enough. The per-model cap keeps a single
# enormous holdout from filling the response on its own. The total cap exists
# because the first one is multiplied by however many models the search
# returns, and that number is not small: a 120s best_quality run on 720 rows
# came back with 44. At ten thousand each that is 5.7 MB of pairs for a board
# the reader will look at one model of at a time.
#
# 150,000 pairs is about 1.9 MB -- large for a response and small next to the
# alternative, which is the screens falling back to nine coarse cuts. A board
# of four models still gets the full ten thousand each; one of forty-four gets
# about 3,400, which is far more than the ten slices a gain curve draws and
# enough for the profit curve to step at every count a team would act on.
SEALED_SCORED_MAX = 10000
SEALED_SCORED_TOTAL = 150000

# Regularisation strengths for the linear model, as fixed values rather than a
# search space. AutoGluon trains one model per dict in the list, so this walks
# C without turning on HPO -- and a search space would not merely be ignored
# here, it raises: "Any provided search spaces will error if
# hyperparameter_tune_kwargs=None", which is this script's call. Each value
# becomes its own leaderboard row, which is also what the screen wants: how the
# score moves with C is the readable part.
#
# The range reaches well below sklearn's default C=1 on purpose. On a small,
# low-dimensional table a strongly regularised logistic regression is often
# the model to beat, and AutoGluon's own default search space starts at 0.1 --
# above where such a winner tends to sit.
_LR_GRID = [{'C': c, 'penalty': 'L2'} for c in (0.003, 0.01, 0.03, 0.1, 0.3, 1, 3)]


def _hyperparameters_for(preset: str):
    """The preset's own portfolio, plus a linear model.

    No AutoGluon portfolio carries one: `best_quality` maps to `zeroshot`,
    which is CAT/FASTAI/GBM/NN_TORCH/RF/XGB/XT, and the `default` config is the
    same list. So every board this engine has ever produced was chosen from
    trees and nets only -- on a small table that can exclude the best model
    before the search starts.

    Returns None when the preset's portfolio cannot be resolved, which leaves
    AutoGluon to do exactly what it did before.
    """
    try:
        from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
        from autogluon.tabular.configs.presets_configs import (
            tabular_presets_alias, tabular_presets_dict,
        )
    except ImportError:
        return None
    # 'best_quality' is a key; 'best' and 'bq' are aliases for it.
    resolved = tabular_presets_alias.get(preset, preset)
    if resolved not in tabular_presets_dict:
        return None
    # A preset that names no portfolio (medium_quality is one) leaves AutoGluon
    # on its own 'default' config, so that is what has to be extended -- not
    # nothing, which would quietly skip the linear model for that preset.
    name = tabular_presets_dict[resolved].get('hyperparameters', 'default')
    if not isinstance(name, str):
        return None
    try:
        config = get_hyperparameter_config(name)
    except ValueError:
        return None
    if not isinstance(config, dict):
        return None
    config = dict(config)
    config.setdefault('LR', _LR_GRID)
    return config


class _StderrSink:
    """Holds whatever AutoGluon writes to stderr while it is fitting.

    Every *_analysis.py here has the same CLI contract: stdout carries the
    answer, and anything on stderr means the run failed. AutoGluon does not
    work that way -- it reports skipped models and third-party import warnings
    on stderr and still returns a leaderboard -- so a successful run left
    nineteen lines behind that read, to the caller and to the logs, as a
    failure that also happened to succeed.

    The stream is taken at the file-descriptor level rather than by rebinding
    `sys.stderr`, because the noise comes from C extensions and from logging
    handlers that captured the stream when they were imported, and
    `contextlib.redirect_stderr` reaches neither.

    Nothing is thrown away: on failure the tail goes into the error message,
    which is the one place it was ever worth reading.
    """

    def __init__(self):
        self._file = tempfile.TemporaryFile()

    def tail(self, limit: int = 1500) -> str:
        try:
            self._file.flush()
            self._file.seek(0)
            return self._file.read().decode('utf-8', 'replace').strip()[-limit:]
        except Exception:
            return ''

    def close(self):
        try:
            self._file.close()
        except Exception:
            pass

    @contextlib.contextmanager
    def capturing(self):
        saved = os.dup(2)
        try:
            sys.stderr.flush()
            os.dup2(self._file.fileno(), 2)
            yield
        finally:
            sys.stderr.flush()
            os.dup2(saved, 2)
            os.close(saved)


# How many features get a curve, and how finely. The 14 scripts draw the top
# six at sklearn's default resolution; the number of points is smaller here
# because each one costs a prediction over the whole sample and AutoGluon's
# bagged ensembles are slower per row than one estimator. Measured on the
# example data: 20 points over 200 rows is 1.3s per feature, 7.6s for six —
# which is what a hundred points would cost for one.
PDP_FEATURES = 6
PDP_GRID = 20
PDP_SAMPLE = 200
PDP_ICE = 30


def _pdp_grid(col: pd.Series) -> list:
    """The values to hold a column at.

    Quantiles rather than a linear span, so a right-tailed column spends its
    points where the rows are instead of stretching most of the line across a
    tail two rows live in. Duplicates are dropped, which is what makes a
    column with six distinct values draw six points rather than twenty copies.
    """
    if pd.api.types.is_numeric_dtype(col):
        vals = pd.to_numeric(col, errors='coerce').dropna()
        if vals.empty:
            return []
        qs = np.quantile(vals, np.linspace(0.05, 0.95, PDP_GRID))
        return sorted({float(v) for v in qs})
    # Categorical: its own values, the common ones first, and no more than the
    # numeric budget so one high-cardinality column cannot cost minutes.
    return [v for v in col.astype(str).value_counts().head(PDP_GRID).index]


def _compute_pdp(predictor, df: pd.DataFrame, features: list, task_type: str,
                 order: list, positive_class, inverse=None):
    """Partial dependence, computed from the definition rather than through
    sklearn.inspection.

    `partial_dependence` wants an estimator that passes scikit-learn's own
    checks -- `fit`, `classes_`, `_estimator_type`, `check_is_fitted` -- and a
    TabularPredictor is not one. Standing in for all of that would be more
    code, and more fragile code, than the definition itself: hold one column
    at a value, predict every row, average. The individual rows are the ICE
    curves the chart already draws, and they come out of the same call.

    One request per feature, with the grid and the sample crossed into a
    single frame, because a per-row loop over a bagged ensemble is the
    difference between a second and several minutes.
    """
    # A curve is ONE class's probability across the grid. For two classes that
    # is well defined. Past two there is no single class to follow: the
    # winning class's probability stops describing one class at the point the
    # prediction flips, which draws a V where the real curve falls straight
    # through -- the same reason What-if refuses more than two classes rather
    # than drawing them wrong.
    if task_type == 'classification' and positive_class is None:
        return None

    sample = df[features]
    if len(sample) > PDP_SAMPLE:
        sample = sample.sample(n=PDP_SAMPLE, random_state=42)

    ranked = [f for f in order if f in features] + [f for f in features if f not in order]
    out = []
    for feat in ranked[:PDP_FEATURES]:
        grid = _pdp_grid(df[feat])
        if len(grid) < 2:
            continue
        stacked = pd.concat([sample.assign(**{feat: v}) for v in grid], ignore_index=True)
        try:
            if task_type == 'classification':
                proba = predictor.predict_proba(stacked)
                col = proba[positive_class] if positive_class in proba.columns else proba.iloc[:, -1]
                y = np.asarray(col, dtype=float)
            else:
                y = np.asarray(predictor.predict(stacked), dtype=float)
                if inverse is not None:
                    y = inverse(y)
        except Exception:
            continue

        # Back to (grid, row): the stack was built grid-major.
        curves = y.reshape(len(grid), len(sample))
        out.append({
            'feature': feat,
            # NOT _to_native_type: that converts numpy scalars and sends
            # everything else through str(), so a plain Python float -- which
            # is what a quantile comes back as here -- landed in the payload
            # as "3.0". The screen's Number() rescued it, and the type was
            # wrong all the way down the wire. A category is already a string
            # and stays one; a number stays a number.
            'grid': [v if isinstance(v, str) else _finite(v) for v in grid],
            'average': [_finite(v) for v in curves.mean(axis=1)],
            # ICE rows, transposed so each is one row across the whole grid.
            'individual': [[_finite(v) for v in row] for row in curves.T[:PDP_ICE]],
        })
    return out or None



def _sealed_cap(n_models: int) -> int:
    """Pairs each model may send, once the board's size is known.

    Dividing the total by the model count rather than handing it out first
    come, first served: the screens compare models against each other, so a
    board where the first few carry every row and the rest carry none would
    draw curves that are not comparable.
    """
    return max(1, min(SEALED_SCORED_MAX, SEALED_SCORED_TOTAL // max(1, n_models)))


def _sealed_scored(y_true, proba, cap: int = SEALED_SCORED_MAX) -> dict:
    """The sealed (score, outcome) pairs themselves, not just cuts through them.

    `sealed_points` above is nine measured cuts, which cannot answer "how much
    of the answer sits in the top 10%" -- that needs a count at every row --
    and cannot be resampled for a range, because resampling nine points gives
    a range that steps between them and reads far steadier than the data is.
    The screens that read this (gain and lift, the profit curve, the margin
    spread) were written against this field; until it was sent they drew
    nothing or fell back to the nine.

    Thinned by an even stride past the cap rather than a random draw: the same
    response twice gives the same curve, and a file that arrives grouped by
    outcome still thins to the same mix. `sampled` travels with it, so the
    screen can say the curve is off a thinned sample instead of implying it
    read every row.
    """
    n_total = int(len(proba))
    step = 1 if n_total <= cap else int(np.ceil(n_total / cap))
    pred = [round(float(v), 6) for v in proba[::step]]
    actual = [int(v) for v in y_true[::step]]
    return {
        'n': len(pred),
        'n_total': n_total,
        'sampled': step > 1,
        'pred': pred,
        'actual': actual,
    }

def _points_at(y_true, proba, cuts) -> list:
    """Operating points on the sealed rows, in counts as well as ratios.

    The counts are the point. "정밀도 0.50" is a ratio nobody budgets in;
    "놓침 12건 · 잘못 부름 40건" multiplies by a cost and becomes money, which
    is what decides which model is actually cheapest to run -- a question the
    ranking metric cannot answer, because it does not know what a miss costs.
    """
    out = []
    for cut in cuts:
        pred = (proba >= cut).astype(int)
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        fn = int(((pred == 0) & (y_true == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        out.append({
            'threshold': _finite(cut),
            'precision': _finite(precision),
            'recall': _finite(recall),
            'f1': _finite(f1),
            'predicted_positive': tp + fp,
            'missed': fn,
            'false_alarms': fp,
        })
    return out


# Cuts reported per model on the sealed rows. Quantiles of that model's own
# predicted probabilities rather than an even 0..1 grid: a model whose scores
# all sit between .02 and .35 has nothing to say about a cut at .9, and an even
# grid spends most of its rows there.
SEALED_CUTS = 9
# Below this many sealed rows the per-level and per-bin figures below are noise
# wearing decimal places.
MIN_SEALED_FOR_CHECKS = 30
# Fewest rows a subgroup needs before its recall is worth printing.
MIN_LEVEL_ROWS = 10


def _calibration(y_true, proba, bins: int = 10):
    """Does a predicted 30% actually happen 30% of the time?

    A model can rank perfectly and still be wrong about the number. Ranking is
    all PR-AUC measures, and every screen that prints a probability -- the
    prediction panel, the threshold table, any cost in won -- reads the number
    instead. Equal-width bins across the predicted range, each with how often
    the thing actually happened, plus ECE as the one-figure summary.
    """
    if len(y_true) < MIN_SEALED_FOR_CHECKS or len(np.unique(y_true)) < 2:
        return None
    edges = np.linspace(0.0, 1.0, bins + 1)
    idx = np.clip(np.digitize(proba, edges[1:-1], right=False), 0, bins - 1)
    out = []
    ece = 0.0
    for b in range(bins):
        sel = idx == b
        n = int(sel.sum())
        if n == 0:
            continue
        predicted = float(proba[sel].mean())
        actual = float(y_true[sel].mean())
        ece += (n / len(y_true)) * abs(predicted - actual)
        out.append({
            'from': _finite(edges[b]), 'to': _finite(edges[b + 1]),
            'n': n, 'predicted': _finite(predicted), 'actual': _finite(actual),
        })
    return {'bins': out, 'ece': _finite(ece), 'n': int(len(y_true))} if out else None


def _subgroups(frame: pd.DataFrame, features: list, y_true, pred, max_levels: int = 6) -> list:
    """Per-level recall and rates on the sealed rows, for columns that name
    groups rather than measure something.

    An overall score hides a model that finds 60% of one group's positives and
    20% of another's. This does not judge that -- a real difference in the data
    looks identical to an unfair model from here -- it puts the numbers where
    someone can see them before deploying.
    """
    out = []
    for name in features:
        if name not in frame.columns:
            continue
        col = frame[name].astype(str)
        if col.nunique() > max_levels * 2:
            continue
        levels = []
        for value in col.unique():
            mask = (col == value).to_numpy()
            n = int(mask.sum())
            if n < MIN_LEVEL_ROWS:
                continue
            yt = y_true[mask]
            yp = pred[mask]
            pos = int(yt.sum())
            tp = int(((yp == 1) & (yt == 1)).sum())
            levels.append({
                'value': str(value),
                'n': n,
                'n_positive': pos,
                'actual_rate': _finite(yt.mean()),
                'predicted_rate': _finite(yp.mean()),
                'recall': _finite(tp / pos) if pos else None,
            })
        if len(levels) >= 2:
            levels.sort(key=lambda d: d['n'], reverse=True)
            out.append({'feature': name, 'levels': levels[:max_levels]})
    return out


# ---- the one-page report ------------------------------------------------
#
# Everything from here to _detect_task_type feeds the Auto Lab report
# (statistica-frontend docs/backend-requests-auto-lab-report.md). All of it is
# additive: a caller that reads none of the new keys gets the response it
# always got.

# Out-of-fold predictions are cut into this many disjoint slices to give each
# model a score per slice. The same slices for every model, so the scores pair.
CV_PARTITION_K = 5
INTERVAL_LEVEL = 0.9
SHAP_CLASS_ROWS = 60
SHAP_CLASS_BACKGROUND = 30
LEARNING_FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)
LINEAR_MAX_LEVELS = 12
LINEAR_KEEP_LEVELS = 8
# Wall-clock seconds the whole script may use. Cloud Run cuts the request at
# 600, so the optional extras below are skipped rather than started when they
# could run past it -- a report without a learning curve beats no report.
REQUEST_BUDGET = 560


def _jsonable(v):
    """A label or value as JSON wants it. Not _to_native_type: that sends a
    plain Python int through str(), and a class label 1 must stay 1."""
    if isinstance(v, np.generic):
        return v.item()
    return v


def _score(metric: str, y_true, pred=None, pos_score=None, positive=None):
    """The ranking metric from predictions already in hand, in the units the
    screen prints -- RMSE and MAE as errors, not AutoGluon's negated form."""
    try:
        y = np.asarray(y_true)
        if metric == 'root_mean_squared_error':
            return _finite(np.sqrt(mean_squared_error(y.astype(float), np.asarray(pred, dtype=float))))
        if metric == 'mean_absolute_error':
            return _finite(mean_absolute_error(y.astype(float), np.asarray(pred, dtype=float)))
        if metric == 'r2':
            return _finite(r2_score(y.astype(float), np.asarray(pred, dtype=float)))
        if metric == 'accuracy':
            return _finite(accuracy_score(y, np.asarray(pred)))
        if metric == 'f1':
            return _finite(f1_score((y == positive).astype(int),
                                    (np.asarray(pred) == positive).astype(int), zero_division=0))
        if metric in ('f1_macro', 'f1_weighted'):
            return _finite(f1_score(y, np.asarray(pred), average=metric.split('_')[1], zero_division=0))
        if metric in ('roc_auc', 'average_precision'):
            yb = (y == positive).astype(int)
            if pos_score is None or len(np.unique(yb)) < 2:
                return None
            s = np.asarray(pos_score, dtype=float)
            return _finite(roc_auc_score(yb, s) if metric == 'roc_auc' else average_precision_score(yb, s))
    except Exception:
        return None
    return None


def _cv_form(value, metric: str):
    """A score in the higher-is-better form `cv_mean` travels in."""
    if value is None:
        return None
    return -value if metric in _NEGATED else value


def _cv_partition(y: pd.Series, task_type: str) -> np.ndarray:
    """Slice id per training row, stratified for classification, fixed seed."""
    from sklearn.model_selection import StratifiedKFold, KFold
    n = len(y)
    fold = np.zeros(n, dtype=int)
    blank = np.zeros(n)
    if task_type == 'classification' and y.astype(str).value_counts().min() >= CV_PARTITION_K:
        parts = StratifiedKFold(CV_PARTITION_K, shuffle=True, random_state=0).split(blank, y.astype(str))
    else:
        parts = KFold(CV_PARTITION_K, shuffle=True, random_state=0).split(blank)
    for k, (_, test) in enumerate(parts):
        fold[test] = k
    return fold


def _fold_scores(metric, folds, y_true, pred, pos_score, positive):
    """The ranking metric on each slice of one model's out-of-fold predictions.

    Not the scores of AutoGluon's own bagging folds: those are not exposed, and
    a model a stack level up was fitted on different folds anyway. Every
    prediction used here is still out-of-fold, so each slice score is an honest
    held-out score, and because every model is cut by the same slices the
    scores pair -- which is what a "how often does the leader stay first"
    resample needs.
    """
    out = []
    for k in range(CV_PARTITION_K):
        sel = folds == k
        if not sel.any():
            continue
        out.append(_score(metric, y_true[sel],
                          None if pred is None else pred[sel],
                          None if pos_score is None else pos_score[sel], positive))
    if len(out) < 2 or any(v is None for v in out):
        return None
    return out


def _conformal_half_width(abs_residuals, level: float = INTERVAL_LEVEL):
    """Split-conformal: the ceil((n+1)·level)-th smallest absolute out-of-fold
    residual. A band of that half-width holds `level` of new rows on average,
    with no assumption about the shape of the errors -- which matters on the
    right-tailed targets this lab sees, where a normal band undercovers the
    large values."""
    r = np.sort(np.abs(np.asarray(abs_residuals, dtype=float)))
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 20:
        return None
    k = int(np.ceil((n + 1) * level))
    if k > n:
        return None
    return float(r[k - 1])


def _subgroup_errors(frame: pd.DataFrame, features: list, y_true, pred, task_type: str,
                     max_levels: int = 6) -> list:
    """_subgroups for the cases it does not cover: accuracy per level for more
    than two classes, error and bias per level for regression."""
    out = []
    for name in features:
        if name not in frame.columns:
            continue
        col = frame[name].astype(str)
        if col.nunique() > max_levels * 2:
            continue
        levels = []
        for value in col.unique():
            mask = (col == value).to_numpy()
            n = int(mask.sum())
            if n < MIN_LEVEL_ROWS:
                continue
            yt, yp = y_true[mask], pred[mask]
            if task_type == 'regression':
                yt, yp = yt.astype(float), yp.astype(float)
                levels.append({
                    'value': str(value), 'n': n,
                    'mae': _finite(np.mean(np.abs(yp - yt))),
                    'bias': _finite(np.mean(yp - yt)),
                    'mean_actual': _finite(np.mean(yt)),
                })
            else:
                levels.append({'value': str(value), 'n': n,
                               'accuracy': _finite(np.mean(yt == yp))})
        if len(levels) >= 2:
            levels.sort(key=lambda d: d['n'], reverse=True)
            out.append({'feature': name, 'levels': levels[:max_levels]})
    return out


def _per_class(y_true, pred, labels) -> list:
    p, r, f, s = (precision_score(y_true, pred, labels=labels, average=None, zero_division=0),
                  recall_score(y_true, pred, labels=labels, average=None, zero_division=0),
                  f1_score(y_true, pred, labels=labels, average=None, zero_division=0),
                  [int((np.asarray(y_true) == c).sum()) for c in labels])
    return [{'label': _jsonable(c), 'precision': _finite(p[i]), 'recall': _finite(r[i]),
             'f1': _finite(f[i]), 'support': s[i]} for i, c in enumerate(labels)]


def _shap_by_class(predictor, model_name, explain: pd.DataFrame, background: pd.DataFrame,
                   features: list, labels: list) -> dict:
    """Mean |SHAP| per feature, per class, from the permutation explainer.

    shap perturbs numbers, so categories travel as integer codes and are
    decoded back before the predictor sees them -- the same trick models_api
    uses for its row explanations.
    """
    import shap
    cats, kinds = {}, {}
    for c in features:
        both = pd.concat([background[c], explain[c]])
        if pd.api.types.is_numeric_dtype(both):
            kinds[c] = background[c].dtype
        else:
            cats[c] = pd.Index(sorted(both.dropna().astype(str).unique()))

    def encode(frame):
        cols = {}
        for c in features:
            if c in cats:
                codes = pd.Categorical(frame[c].astype(str).where(frame[c].notna()),
                                       categories=cats[c]).codes.astype(float)
                codes[codes < 0] = np.nan
                cols[c] = codes
            else:
                cols[c] = pd.to_numeric(frame[c], errors='coerce').to_numpy(float)
        return pd.DataFrame(cols, columns=features).to_numpy(float)

    def decode(arr):
        frame = pd.DataFrame(np.asarray(arr, dtype=float), columns=features)
        for c, idx in cats.items():
            codes = frame[c].to_numpy()
            ok = np.isfinite(codes)
            vals = np.empty(len(codes), dtype=object)
            vals[ok] = idx.take(codes[ok].astype(int))
            vals[~ok] = None
            frame[c] = vals
        for c, dtype in kinds.items():
            if pd.api.types.is_integer_dtype(dtype) and frame[c].notna().all():
                frame[c] = frame[c].round().astype(dtype)
        return frame

    def f(arr):
        proba = predictor.predict_proba(decode(arr), model=model_name)
        return proba[labels].to_numpy(dtype=float)

    explainer = shap.PermutationExplainer(f, encode(background))
    values = np.asarray(explainer(encode(explain), max_evals=2 * len(features) + 1, silent=True).values)
    if values.ndim != 3:
        return None
    mean_abs = np.abs(values).mean(axis=0)  # (features, classes)
    out = {}
    for k, label in enumerate(labels):
        order = np.argsort(-mean_abs[:, k])
        out[str(label)] = [{'feature': features[j], 'importance': _finite(mean_abs[j, k])}
                           for j in order]
    return out


def _learning_curve(frame: pd.DataFrame, target: str, task_type: str, eval_metric: str,
                    deadline: float, positive, inverse=None) -> dict:
    """Train and out-of-fold score at growing shares of the training rows.

    One LightGBM at its defaults, bagged five ways, per share -- not the whole
    search again, which would cost five times the budget. The question it
    answers ("is the score still climbing?") is about the data more than the
    model, and the label says which model it was.
    """
    from autogluon.tabular import TabularPredictor
    remaining = deadline - time.time()
    if remaining < 40:
        return {'skipped': 'time', 'seconds_left': _finite(remaining)}
    per_fit = max(8, min(45, int(remaining / len(LEARNING_FRACTIONS)) - 3))
    # train/cv are the metric in its own form (RMSE as a positive error), not
    # the higher-is-better form cv_mean and fold_scores use -- so say which
    # way is up, as cv_results does, instead of leaving the screen to guess.
    out = {'metric': eval_metric, 'model': 'LightGBM', 'scope': 'oof_5fold',
           'higher_is_better': eval_metric not in _NEGATED,
           'fractions': [], 'n_rows': [], 'train': [], 'cv': []}
    for frac in LEARNING_FRACTIONS:
        if time.time() + per_fit > deadline:
            out['stopped_early'] = True
            break
        if frac >= 1.0:
            sub = frame
        elif task_type == 'classification':
            sub = frame.groupby(target, group_keys=False).apply(
                lambda g: g.sample(frac=frac, random_state=0))
        else:
            sub = frame.sample(frac=frac, random_state=0)
        sub = sub.reset_index(drop=True)
        if len(sub) < 30:
            continue
        work = tempfile.mkdtemp(prefix='ag_lc_')
        try:
            p = TabularPredictor(
                label=target,
                problem_type=('regression' if task_type == 'regression' else None),
                eval_metric=eval_metric, path=work, verbosity=0,
            ).fit(sub, hyperparameters={'GBM': {}}, num_bag_folds=5, num_bag_sets=1,
                  num_stack_levels=0, fit_weighted_ensemble=False, time_limit=per_fit)
            name = p.model_names()[0]
            y = sub[target].to_numpy()
            if task_type == 'regression':
                tr = np.asarray(p.predict(sub, model=name), dtype=float)
                cv = np.asarray(p.predict_oof(model=name), dtype=float)
                if inverse is not None:
                    y, tr, cv = inverse(y.astype(float)), inverse(tr), inverse(cv)
                s_tr, s_cv = _score(eval_metric, y, tr), _score(eval_metric, y, cv)
            else:
                tr_p, cv_p = p.predict_proba(sub, model=name), p.predict_proba_oof(model=name)
                tr_l, cv_l = p.predict(sub, model=name), p.predict_oof(model=name)
                pos_tr = tr_p[positive].to_numpy() if positive is not None else None
                pos_cv = cv_p[positive].to_numpy() if positive is not None else None
                s_tr = _score(eval_metric, y, tr_l.to_numpy(), pos_tr, positive)
                s_cv = _score(eval_metric, y[cv_l.index.to_numpy()], cv_l.to_numpy(), pos_cv, positive)
            out['fractions'].append(frac)
            out['n_rows'].append(int(len(sub)))
            out['train'].append(s_tr)
            out['cv'].append(s_cv)
        except Exception:
            continue
        finally:
            shutil.rmtree(work, ignore_errors=True)
    return out if out['fractions'] else {'skipped': 'failed'}


def _linear_coefficients(frame: pd.DataFrame, target: str, features: list, task_type: str,
                         positive):
    """Coefficients and 95% intervals from a plain logistic or OLS fit.

    AutoGluon's own LinearModel does not expose standard errors, and its
    preprocessing renames the columns, so the "simple model" panel reads this
    separate, standardised fit instead -- a numeric coefficient is per one
    standard deviation, a category's is against the most common level. The
    meta says so, so the screen can.
    """
    import statsmodels.api as sm
    parts, refs, skipped = [], {}, []
    for c in features:
        s = frame[c]
        if pd.api.types.is_numeric_dtype(s):
            v = pd.to_numeric(s, errors='coerce')
            v = v.fillna(v.median())
            sd = float(v.std(ddof=0))
            if not np.isfinite(sd) or sd == 0:
                skipped.append(c)
                continue
            parts.append(((v - v.mean()) / sd).rename(c))
            continue
        s = s.astype(str).where(s.notna(), '(결측)')
        counts = s.value_counts()
        if len(counts) < 2 or len(counts) > LINEAR_MAX_LEVELS:
            skipped.append(c)
            continue
        ref = counts.index[0]
        refs[c] = ref
        keep = list(counts.index[1:1 + LINEAR_KEEP_LEVELS])
        for level in keep:
            parts.append((s == level).astype(float).rename(f'{c}: {level}'))
        rest = counts.index[1 + LINEAR_KEEP_LEVELS:]
        if len(rest):
            parts.append(s.isin(rest).astype(float).rename(f'{c}: (기타)'))
    if not parts:
        return None, {'reason': 'no_usable_columns', 'skipped': skipped}
    X = sm.add_constant(pd.concat(parts, axis=1), has_constant='add')
    if task_type == 'regression':
        y = pd.to_numeric(frame[target], errors='coerce').astype(float)
        res = sm.OLS(y, X).fit()
        kind, converged = 'ols', True
    else:
        y = (frame[target] == positive).astype(float)
        res = sm.Logit(y, X).fit(disp=0, maxiter=200)
        kind = 'logistic'
        converged = bool(res.mle_retvals.get('converged', True))
    ci = res.conf_int(0.05)
    rows = []
    for term in X.columns:
        if term == 'const':
            continue
        feature, _, level = term.partition(': ')
        coef, lo, hi = float(res.params[term]), float(ci.loc[term, 0]), float(ci.loc[term, 1])
        row = {'feature': feature, 'term': term, 'level': level or None,
               'coef': _finite(coef), 'ci_low': _finite(lo), 'ci_high': _finite(hi),
               'p_value': _finite(res.pvalues[term])}
        if kind == 'logistic':
            row.update({'odds_ratio': _finite(np.exp(coef)),
                        'or_low': _finite(np.exp(lo)), 'or_high': _finite(np.exp(hi))})
        rows.append(row)
    meta = {
        'model': kind, 'standardized': True, 'n': int(len(y)),
        'reference_levels': {k: str(v) for k, v in refs.items()},
        'skipped': skipped, 'converged': converged,
        # A log-odds this large on standardised inputs is separation, not an
        # effect: the intervals that come with it are not worth printing.
        # Regression coefficients are in the target's units, so no such rule.
        'unstable': bool(kind == 'logistic' and any(abs(r['coef'] or 0) > 10 for r in rows)),
        'positive_class': _jsonable(positive) if kind == 'logistic' else None,
        'fit': _finite(res.rsquared if kind == 'ols' else res.prsquared),
    }
    return rows, meta


def _library_versions() -> dict:
    """Installed versions of the packages that decide what the board holds."""
    from importlib import metadata
    out = {'python': sys.version.split()[0]}
    for pkg in ('autogluon.tabular', 'lightgbm', 'catboost', 'xgboost',
                'scikit-learn', 'numpy', 'pandas'):
        try:
            out[pkg] = metadata.version(pkg)
        except Exception:
            out[pkg] = None
    return out


def _detect_task_type(y: pd.Series) -> str:
    vals = y.dropna()
    if vals.empty:
        return 'classification'
    if pd.api.types.is_numeric_dtype(vals) and vals.nunique() > 20:
        return 'regression'
    return 'classification'


def main():
    script_started = time.time()
    try:
        payload = json.load(sys.stdin)

        data = payload.get('data')
        target = payload.get('target_col') or payload.get('target')
        features = payload.get('feature_cols') or payload.get('features')
        if not data or not target or not features:
            raise ValueError('Missing data, target_col, or feature_cols')

        task_req = payload.get('task_type', 'auto')
        # Default kept well inside the Cloud Run request timeout. The caller
        # sends the real budget; see the design doc's staging of this (a longer
        # limit needs the request timeout raised, and a much longer one needs a
        # job queue, neither of which this script decides).
        time_limit = int(payload.get('time_limit') or 300)
        # 'best_quality', not AutoGluon's gentler default, because the budget
        # has to be real. Measured on the app's own example (971 training rows,
        # PR-AUC, this script, this machine):
        #
        #   medium_quality, 300s budget -> 20.5s used, hit_limit False, PR-AUC .419
        #   best_quality,   120s budget -> 120.3s used, hit_limit True,  PR-AUC .497
        #   best_quality,   300s budget -> 300.2s used, hit_limit True,  PR-AUC .505
        #
        # Two things follow. The smallest budget under best_quality beats the
        # largest under the old default, so this is not a trade. And under the
        # old default the budget was decoration: the search stopped after a
        # fifteenth of it whatever the caller asked for, which makes the lab's
        # "you say how long it may look" untrue and leaves `hit_limit` always
        # False -- the very flag the design says should decide whether to raise
        # the request timeout.
        #
        # What this does change is that a 300-second budget now takes 300
        # seconds of request. The design already assumes that (stage 1 is
        # 240-500s inside the current Cloud Run timeout); it was simply not
        # happening. A caller that needs the old behaviour sends preset itself.
        preset = payload.get('preset') or 'best_quality'

        df_all = pd.DataFrame(data)
        missing_cols = [c for c in [target] + list(features) if c not in df_all.columns]
        if missing_cols:
            raise ValueError(f'Columns not in the data: {missing_cols}')

        # Count before anything else drops a row, so row_counts describes what
        # was sent rather than what survived.
        n_input = len(df_all)
        df = df_all[list(features) + [target]].dropna(subset=[target]).reset_index(drop=True)
        n_target_missing_dropped = n_input - len(df)

        # Repeated rows, dropped only when the caller asks. A duplicate is
        # counted twice by every fold and every score, so leaving them in
        # quietly weights those rows double -- but removing them changes the
        # data a previous run was ranked on, so this script does not decide it.
        # The Data screen shows the count and offers the choice; this flag is
        # that choice arriving.
        n_duplicates_dropped = 0
        if bool(payload.get('drop_duplicate_rows')):
            before = len(df)
            df = df.drop_duplicates().reset_index(drop=True)
            n_duplicates_dropped = before - len(df)
        if len(df) < 50:
            raise ValueError(
                f'At least 50 rows with a target are required; {len(df)} usable of {n_input}.'
            )

        # The caller's own sealed rows, if it sent them -- never fit on, only
        # scored against afterward. Optional: a caller that predates this
        # field gets exactly the response it used to (holdout_df stays None,
        # n_holdout stays 0 below).
        holdout = payload.get('holdout')
        holdout_df = None
        # Where each kept sealed row sat in the request's own `holdout` array,
        # so row-level predictions can be joined back to the rows the caller
        # holds (rows without a target are dropped, which shifts positions).
        holdout_pos = None
        if holdout:
            h_all = pd.DataFrame(holdout)
            h_missing = [c for c in [target] + list(features) if c not in h_all.columns]
            if h_missing:
                raise ValueError(f'Holdout rows are missing column(s): {h_missing}')
            h_sub = h_all[list(features) + [target]]
            keep = h_sub[target].notna().to_numpy()
            holdout_pos = np.flatnonzero(keep)
            holdout_df = h_sub[keep].reset_index(drop=True)

        task_type = task_req if task_req in ('classification', 'regression') else _detect_task_type(df[target])

        eval_metric = payload.get('eval_metric') or _DEFAULT_METRIC[task_type]
        if eval_metric not in _METRIC_FIELD:
            raise ValueError(
                f"Unsupported eval_metric '{eval_metric}' — "
                f"expected one of {sorted(_METRIC_FIELD)}"
            )
        metric_field = _METRIC_FIELD[eval_metric]

        # Train on log(1+y) and report everything back in the target's own
        # units. Sending a log target from the browser instead makes every
        # score a log-unit score, and RMSE does not survive the trip back.
        requested_transform = payload.get('target_transform') or 'none'
        if requested_transform not in ('none', 'log1p'):
            raise ValueError(f"Unsupported target_transform '{requested_transform}' — expected 'none' or 'log1p'")
        transform = None
        transform_note = None
        if requested_transform == 'log1p':
            if task_type != 'regression':
                transform_note = 'ignored_for_classification'
            else:
                y_num = pd.to_numeric(df[target], errors='coerce')
                if y_num.isna().any() or (y_num < 0).any():
                    raise ValueError('target_transform log1p needs a numeric target with no negative values.')
                transform = 'log1p'
        inverse = np.expm1 if transform else None
        df_fit = df
        if transform:
            df_fit = df.copy()
            df_fit[target] = np.log1p(pd.to_numeric(df[target]).astype(float))

        # Imported here, not at the top and not before the checks above: a
        # caller who sent the wrong column names should be told that, and an
        # import placed earlier answered every bad payload with "autogluon is
        # not installed" instead.
        try:
            from autogluon.tabular import TabularPredictor
        except ImportError:
            raise ValueError(
                'autogluon.tabular is not installed on the server. '
                'Add autogluon.tabular to requirements.txt (the GBM extras '
                'only -- the torch models are not used here).'
            )

        work_dir = tempfile.mkdtemp(prefix='ag_')
        sink = _StderrSink()
        started = time.time()
        try:
            with sink.capturing():
                predictor = TabularPredictor(
                    label=target,
                    problem_type=('regression' if task_type == 'regression' else None),
                    eval_metric=eval_metric,
                    path=work_dir,
                    verbosity=0,
                ).fit(
                    df_fit,
                    time_limit=time_limit,
                    presets=preset,
                    hyperparameters=_hyperparameters_for(preset),
                    excluded_model_types=_NO_TORCH,
                )

                seconds_used = round(time.time() - started, 1)
                board = predictor.leaderboard(silent=True)

                # Every kept model's score on rows it never trained or tuned
                # against -- no refit, since the predictor already holds all
                # of them; this is one inference pass over (usually small)
                # sealed rows, not another fit. What the search's own
                # score_val cannot rule out is the one AutoGluon picks having
                # won by luck among dozens of candidates; this is the check
                # that rules it out, for every row on the board at once.
                # Not under a target transform: the leaderboard would score
                # log-unit predictions against the sealed rows' own units. The
                # sealed scores are computed from predictions further down.
                sealed_board = None
                if holdout_df is not None and len(holdout_df) > 0 and not transform:
                    try:
                        sealed_board = predictor.leaderboard(holdout_df, silent=True)
                    except Exception:
                        sealed_board = None

            # One row per model AutoGluon kept, in its own ranking order.
            #
            # The metric it optimised is filled in here; for binary
            # classification the other Compare columns and the ROC/PR curves
            # are filled in just below, from AutoGluon's own internal
            # validation fold -- not from the training rows, so the honesty
            # this comment used to promise ("an empty cell rather than a
            # training score mislabelled as held-out") still holds. Multiclass
            # and regression runs still leave those cells empty: there is no
            # single ROC/PR curve for more than two classes, and accuracy/f1
            # are not regression measures.
            models = []
            for _, r in board.iterrows():
                raw_score = _finite(r.get('score_val'))
                score = (-raw_score if (raw_score is not None and eval_metric in _NEGATED)
                         else raw_score)
                name = str(r.get('model'))
                models.append({
                    'name': name,
                    'is_ensemble': name.startswith('WeightedEnsemble'),
                    'metrics': {
                        metric_field: score,
                        # AutoGluon's score_val IS a cross-validated figure when
                        # the preset bags; reporting it here lets the board show
                        # stability beside the score, as it does for the other
                        # fourteen.
                        #
                        # Higher-is-better, as the other fourteen send it and as
                        # automl-client.ts reads it: an error metric comes back
                        # negated (RMSE 3.0 as -3.0), which is score_val's own
                        # form. Sending the positive error here is what printed
                        # "RMSE -148308" on the board -- the screen flipped a
                        # sign that had already been flipped.
                        'cv_mean': _cv_form(score, eval_metric),
                    },
                    'fit_seconds': _finite(r.get('fit_time')),
                    'predict_seconds': _finite(r.get('pred_time_val')),
                })
                if sealed_board is not None:
                    sealed_row = sealed_board[sealed_board['model'] == name]
                    if len(sealed_row):
                        raw_sealed = _finite(sealed_row.iloc[0].get('score_test'))
                        sealed_score = (-raw_sealed if (raw_sealed is not None
                                        and eval_metric in _NEGATED) else raw_sealed)
                        if sealed_score is not None:
                            models[-1]['sealed_metrics'] = {metric_field: sealed_score}

            labels = list(predictor.class_labels or []) if task_type == 'classification' else []
            positive = None
            if len(labels) == 2:
                positive = (predictor.positive_class
                            if predictor.positive_class in labels else labels[-1])

            # ---- every model's out-of-fold predictions, once ----------------
            #
            # The bagged preset already holds a prediction for every training
            # row from a model that never saw it. Scoring those by hand fills
            # what the leaderboard leaves empty for more than two classes and
            # for regression, gives each model a score per slice (the ± on the
            # board and the "does the leader stay first" resample), and is the
            # calibration set for the regression band below -- with no refit.
            folds = _cv_partition(df[target], task_type)
            y_all = df[target].to_numpy()
            oof = {}
            if transform:
                # score_val above was measured in log units; nothing on the
                # board may carry that number under the target's own name.
                for m in models:
                    m['metrics'][metric_field] = None
                    m['metrics']['cv_mean'] = None
            for m in models:
                try:
                    pos_o = None
                    raw_o = None
                    if task_type == 'classification':
                        proba_o = predictor.predict_proba_oof(model=m['name'])
                        pred_s = predictor.predict_oof(model=m['name']).loc[proba_o.index]
                        pred_o = pred_s.to_numpy()
                        if positive is not None and positive in proba_o.columns:
                            pos_o = proba_o[positive].to_numpy(dtype=float)
                    else:
                        pred_s = predictor.predict_oof(model=m['name'])
                        raw_o = pred_s.to_numpy(dtype=float)
                        pred_o = inverse(raw_o) if inverse is not None else raw_o
                    idx = pred_s.index.to_numpy()
                    y_o = y_all[idx]
                    oof[m['name']] = (idx, y_o, pred_o, pos_o, raw_o)

                    if transform:
                        cv_score = _score(eval_metric, y_o, pred_o)
                        m['metrics'][metric_field] = cv_score
                        m['metrics']['cv_mean'] = _cv_form(cv_score, eval_metric)
                    if task_type == 'regression':
                        m['metrics'].setdefault('rmse', _score('root_mean_squared_error', y_o, pred_o))
                        m['metrics'].setdefault('mae', _score('mean_absolute_error', y_o, pred_o))
                        m['metrics'].setdefault('r2', _score('r2', y_o, pred_o))
                    elif len(labels) > 2:
                        m['metrics'].setdefault('accuracy', _score('accuracy', y_o, pred_o))
                        m['metrics'].setdefault('f1', _score('f1_macro', y_o, pred_o))
                        cm = sk_confusion_matrix(y_o, pred_o, labels=labels)
                        m['confusion_matrix'] = [[int(v) for v in row] for row in cm]

                    fold_scores = _fold_scores(eval_metric, folds[idx], y_o, pred_o, pos_o, positive)
                    if fold_scores:
                        # Same form as cv_mean (higher is better), so their
                        # mean sits beside it and a resample can rank on it.
                        m['cv_results'] = {
                            'cv_scoring': eval_metric,
                            'fold_scores': [_cv_form(v, eval_metric) for v in fold_scores],
                            'higher_is_better': True,
                        }
                        m['metrics']['cv_std'] = _finite(np.std(fold_scores, ddof=1))
                except Exception:
                    continue

            if transform:
                # Re-rank in the target's own units. The search ranked on log
                # error, which weights a miss on a small value like a miss on
                # a large one; the screen ranks on the error it prints.
                lower_better = eval_metric in _NEGATED
                models.sort(key=lambda m: (
                    m['metrics'].get(metric_field) is None,
                    (m['metrics'].get(metric_field) or 0.0) * (1 if lower_better else -1)))
            recommended = models[0]['name'] if models else None

            # ---- the regression band, calibrated out-of-fold ----------------
            interval_q = None
            if task_type == 'regression' and recommended in oof:
                _, y_o, pred_o, _, raw_o = oof[recommended]
                if inverse is not None:
                    residuals = np.log1p(y_o.astype(float)) - raw_o
                else:
                    residuals = y_o.astype(float) - pred_o
                interval_q = _conformal_half_width(residuals)
            prediction_interval = None
            if interval_q is not None:
                prediction_interval = {
                    'model': recommended,
                    'level': INTERVAL_LEVEL,
                    'method': 'split_conformal_oof',
                    # Under log1p the band is symmetric in log units, so it is
                    # wider above the prediction than below it in the
                    # target's own units, as the errors are.
                    'space': transform or 'identity',
                    'half_width': _finite(interval_q),
                    'n_calibration': int(len(oof[recommended][0])),
                    'coverage': None,
                    'n_holdout': 0,
                }

            # ---- every model on the sealed rows -----------------------------
            #
            # The ranking metric cannot answer "which model is cheapest",
            # because it does not know what a miss costs against a wrong call.
            # That answer needs counts at a cut, for every model, on rows none
            # of them were chosen on -- so it is computed here, where the
            # predictor and the sealed rows are both in hand, rather than
            # refitting anything later. The screen multiplies by the two
            # amounts the reader enters; nothing here assumes a currency or a
            # price. The other sealed figures -- every metric the report's
            # evaluation section prints, for every task type -- come out of
            # the same predictions.
            sealed_checks = None
            sealed_predictions = None
            if holdout_df is not None and len(holdout_df) > 0:
                y_h = holdout_df[target].to_numpy()
                if task_type == 'regression':
                    y_h = pd.to_numeric(holdout_df[target], errors='coerce').to_numpy(dtype=float)
                y_bin = (y_h == positive).astype(int) if positive is not None else None
                pairs_ok = y_bin is not None and len(np.unique(y_bin)) == 2
                cap = _sealed_cap(len(models))
                rec = None
                for m in models:
                    name = m['name']
                    try:
                        proba_h = None
                        pos_h = None
                        raw_h = None
                        if task_type == 'classification':
                            proba_h = predictor.predict_proba(holdout_df, model=name)
                            pred_h = predictor.predict(holdout_df, model=name).to_numpy()
                            if positive is not None and positive in proba_h.columns:
                                pos_h = proba_h[positive].to_numpy(dtype=float)
                        else:
                            raw_h = predictor.predict(holdout_df, model=name).to_numpy(dtype=float)
                            pred_h = inverse(raw_h) if inverse is not None else raw_h
                    except Exception:
                        continue
                    if name == recommended:
                        rec = (pred_h, proba_h, pos_h, raw_h)
                    sm_ = dict(m.get('sealed_metrics') or {})
                    if transform:
                        sm_[metric_field] = _score(eval_metric, y_h, pred_h)
                    if task_type == 'regression':
                        sm_.setdefault('rmse', _score('root_mean_squared_error', y_h, pred_h))
                        sm_.setdefault('mae', _score('mean_absolute_error', y_h, pred_h))
                        sm_.setdefault('r2', _score('r2', y_h, pred_h))
                    else:
                        sm_.setdefault('accuracy', _score('accuracy', y_h, pred_h))
                        if positive is not None:
                            sm_.setdefault('f1', _score('f1', y_h, pred_h, positive=positive))
                            pb = (pred_h == positive).astype(int)
                            sm_.setdefault('precision', _finite(precision_score(y_bin, pb, zero_division=0)))
                            sm_.setdefault('recall', _finite(recall_score(y_bin, pb, zero_division=0)))
                            sm_.setdefault('pr_auc', _score('average_precision', y_h, None, pos_h, positive))
                            sm_.setdefault('auc', _score('roc_auc', y_h, None, pos_h, positive))
                            if pairs_ok and pos_h is not None:
                                cuts = sorted({round(float(c), 4) for c in
                                               np.quantile(pos_h, np.linspace(0.05, 0.95, SEALED_CUTS))
                                               if 0.0 < float(c) < 1.0})
                                if cuts:
                                    m['sealed_points'] = _points_at(y_bin, pos_h, cuts)
                                m['sealed_scored'] = _sealed_scored(y_bin, pos_h, cap)
                        else:
                            sm_.setdefault('f1', _score('f1_macro', y_h, pred_h))
                            cm_h = sk_confusion_matrix(y_h, pred_h, labels=labels)
                            m['sealed_confusion_matrix'] = [[int(v) for v in row] for row in cm_h]
                            m['sealed_per_class'] = _per_class(y_h, pred_h, labels)
                    sm_ = {k: v for k, v in sm_.items() if v is not None}
                    if sm_:
                        m['sealed_metrics'] = sm_

                # The checks describe the model that would be deployed, so
                # they are computed once, for the recommended model, rather
                # than for every row on the board.
                if rec is not None:
                    pred_r, proba_r, pos_r, raw_r = rec
                    try:
                        if positive is not None:
                            if pairs_ok and pos_r is not None:
                                sealed_checks = {
                                    'model': recommended,
                                    'n': int(len(y_h)),
                                    # Same conversion as before this block was
                                    # generalised, so the type the screen
                                    # compares against does not change.
                                    'positive_class': _to_native_type(positive),
                                    'calibration': _calibration(y_bin, pos_r),
                                    'subgroups': _subgroups(holdout_df, list(features), y_bin,
                                                            (pred_r == positive).astype(int)),
                                }
                        else:
                            sealed_checks = {
                                'model': recommended,
                                'n': int(len(y_h)),
                                'subgroups': _subgroup_errors(holdout_df, list(features), y_h,
                                                              pred_r, task_type),
                            }
                            if labels:
                                sealed_checks['class_labels'] = [_jsonable(c) for c in labels]
                    except Exception:
                        sealed_checks = None

                    # Row by row, for the recommended model only: what the
                    # report's threshold table, gain curve, residual plot and
                    # bootstrap ranges are all computed from in the browser.
                    try:
                        n_h = len(y_h)
                        row_cap = (SEALED_SCORED_MAX if len(labels) <= 2
                                   else max(1000, min(SEALED_SCORED_MAX, 40000 // len(labels))))
                        step = 1 if n_h <= row_cap else int(np.ceil(n_h / row_cap))
                        sel = np.arange(0, n_h, step)
                        sealed_predictions = {
                            'model': recommended,
                            'row_index': [int(holdout_pos[i]) for i in sel],
                            'y_true': [_jsonable(y_h[i]) for i in sel],
                            'y_pred': [_jsonable(pred_r[i]) for i in sel],
                            'n': int(len(sel)),
                            'n_total': int(n_h),
                            'sampled': bool(step > 1),
                        }
                        if task_type == 'classification':
                            sealed_predictions['class_labels'] = [_jsonable(c) for c in labels]
                            if positive is not None and pos_r is not None:
                                sealed_predictions['positive_class'] = _jsonable(positive)
                                sealed_predictions['proba'] = [round(float(pos_r[i]), 6) for i in sel]
                            elif proba_r is not None:
                                block = proba_r[labels].to_numpy(dtype=float)
                                sealed_predictions['proba'] = [[round(float(v), 6) for v in block[i]]
                                                               for i in sel]
                        elif interval_q is not None:
                            base = raw_r if inverse is not None else pred_r
                            lo, hi = base - interval_q, base + interval_q
                            if inverse is not None:
                                lo, hi = inverse(lo), inverse(hi)
                            sealed_predictions['lower'] = [_finite(lo[i]) for i in sel]
                            sealed_predictions['upper'] = [_finite(hi[i]) for i in sel]
                            covered = (y_h >= lo) & (y_h <= hi)
                            prediction_interval['coverage'] = _finite(covered.mean())
                            prediction_interval['n_holdout'] = int(n_h)
                    except Exception:
                        sealed_predictions = None

            # ---- the other Compare columns, and ROC/PR, for binary models --
            #
            # leaderboard(extra_metrics=...) looked like the way to get these,
            # but AutoGluon raises unless a `data=` frame is also given -- it
            # scores extra metrics on data the caller supplies, not on its own
            # internal validation fold the way score_val above is. The caller
            # here has no such frame (the design note above the model loop
            # says why), so these come from predict_oof/predict_proba_oof
            # instead: the same out-of-fold predictions on df's own rows that
            # the curves below are already built from, scored by hand with the
            # identical sklearn functions the other fourteen scripts use on
            # their own y_test/y_pred. One source of truth, not two.
            #
            # Only binary classification gets a curve, for the reason
            # _compute_pdp above gives: past two classes there is no single
            # probability column to draw one from.
            #
            # setdefault, not assignment: the metric a model was optimised on
            # is already in `metrics` from score_val above, and that is the
            # figure AutoGluon actually searched on -- recomputing it here by
            # hand would be a second, possibly-conflicting source of truth for
            # the one column that matters most on this row.
            if task_type == 'classification':
                labels = list(predictor.class_labels or [])
                if len(labels) == 2:
                    pos_label = (predictor.positive_class if predictor.positive_class in labels
                                 else labels[-1])
                    for m in models:
                        try:
                            oof_proba = predictor.predict_proba_oof(model=m['name'])
                            if pos_label not in oof_proba.columns:
                                continue
                            oof_pred = predictor.predict_oof(model=m['name'])
                            idx = oof_proba.index
                            y_true = (df[target].loc[idx] == pos_label).astype(int)
                            y_score = oof_proba[pos_label]
                            y_pred_bin = (oof_pred.loc[idx] == pos_label).astype(int)
                            if y_true.nunique() < 2 or len(y_true) < 10:
                                continue

                            m['metrics'].setdefault('accuracy', _finite(accuracy_score(y_true, y_pred_bin)))
                            m['metrics'].setdefault(
                                'f1', _finite(f1_score(y_true, y_pred_bin, zero_division=0)))
                            m['metrics'].setdefault(
                                'pr_auc', _finite(average_precision_score(y_true, y_score)))

                            fpr, tpr, _ = roc_curve(y_true, y_score)
                            roc_auc_val = _finite(auc(fpr, tpr))
                            # Compare table reads metrics.auc, not roc_data.auc.
                            m['metrics'].setdefault('auc', roc_auc_val)
                            m['roc_data'] = {'binary': {
                                'fpr': [_finite(x) for x in fpr],
                                'tpr': [_finite(x) for x in tpr],
                                'auc': roc_auc_val,
                            }}
                            precision, recall, _ = precision_recall_curve(y_true, y_score)
                            m['pr_data'] = {'binary': {
                                'precision': [_finite(x) for x in precision],
                                'recall': [_finite(x) for x in recall],
                            }}
                            # [[tn, fp], [fn, tp]] -- labels=[0, 1] pins row 0
                            # to the actual-negative row regardless of which
                            # label value happens to sort first, matching what
                            # the Compare screen's cmOf() reads this shape as.
                            cm = sk_confusion_matrix(y_true, y_pred_bin, labels=[0, 1])
                            m['confusion_matrix'] = [[int(v) for v in row] for row in cm]
                        except Exception:
                            # Not every kept model supports OOF predictions
                            # (e.g. a refit-full variant) -- that row simply
                            # keeps the empty cells it already had.
                            pass

            # ---- which columns it leans on ------------------------------
            #
            # Permutation importance, in the shape the Explain screen already
            # reads from the other fourteen scripts ({feature,
            # importance_mean, importance_std}). AutoGluon reports it as
            # `importance` and `stddev` over five shuffle sets, which is what
            # those two names mean.
            #
            # Measured on the rows this script was given, and that is not the
            # same claim the other scripts make. They split their input again
            # and permute on the half they held back; AutoGluon refuses to
            # compute importance without a dataset once it is bagging, and the
            # preset here bags. Holding rows back inside this script instead
            # would shrink the training set the board then ranks -- distorting
            # the comparison this whole engine exists to make.
            #
            # So it is measured on the training rows and says so, and the
            # screen says so too. On rows a model has fitted, shuffling a
            # column it memorised costs more than shuffling one it generalised
            # from, so these numbers run high and run highest exactly where
            # they would mislead. A ranking is still worth having; a ranking
            # presented as held-out evidence is not.
            # ---- the data checks every other script already runs -------
            #
            # Sixteen scripts call compute_guardrails; this one never did, and
            # Auto Lab's only engine is this one. So the leaderboard's "any
            # column that gives the answer away?" row has been reporting
            # "clear" for every Auto Lab run while checking nothing -- a false
            # all-clear, which is worse than an empty panel.
            #
            # Run on the frame that was fitted, after the duplicate drop, so it
            # describes the data the board was ranked on.
            try:
                from guardrails import compute_guardrails
                guardrails = compute_guardrails(
                    df[list(features)], df[target], list(features), task_type,
                    {'accuracy': None, 'r2': None},
                )
            except Exception:
                guardrails = []
            redundant_pairs = [
                g['pair'] for g in guardrails
                if g.get('id') == 'redundant_pair' and isinstance(g.get('pair'), list)
                and len(g['pair']) == 2
            ]

            perm_importance = None
            perm_scope = None
            try:
                # Features that say the same thing are measured together.
                # Shuffling one of a correlated pair leaves the other to answer,
                # so both come back near zero and the information they share
                # looks unimportant. A tuple makes AutoGluon shuffle the group
                # in one go: "This importance will differ from adding the
                # individual importances of each feature in feature_list, and
                # will be more accurate to the overall group importance."
                #
                # The pairs come from the guardrail that found them, so this
                # does not re-derive the statistic. Date parts need no group:
                # at feature_stage='original' (the default) the raw column is
                # shuffled and the engine's own expansion runs afterwards, so
                # the parts it derives from a date already move together.
                groups, grouped = [], set()
                for pair in (redundant_pairs or []):
                    a, b = pair
                    if a in grouped or b in grouped:
                        continue
                    groups.append((f'{a} + {b}', [a, b]))
                    grouped.update((a, b))
                ask = [c for c in features if c not in grouped] + groups

                # Five shuffle sets is AutoGluon's default without a time
                # limit, and its own docstring asks for at least ten before
                # trusting the spread -- which this response reports.
                fi = predictor.feature_importance(
                    df_fit, features=ask or None, num_shuffle_sets=10, silent=True,
                )
                perm_importance = [
                    {
                        'feature': str(name),
                        'importance_mean': _finite(row.get('importance')),
                        'importance_std': _finite(row.get('stddev')),
                        # Already computed: include_confidence_band defaults to
                        # True at a 99% level, and the screen was drawing only
                        # the standard deviation while these sat unread.
                        'importance_p': _finite(row.get('p_value')),
                        'importance_low': _finite(row.get('p99_low')),
                        'importance_high': _finite(row.get('p99_high')),
                        'shuffles': _finite(row.get('n')),
                        'grouped': str(name) in {g[0] for g in groups},
                    }
                    for name, row in fi.iterrows()
                ]
                perm_scope = 'train'
            except Exception:
                perm_importance = None

            # ---- how the prediction moves with each column ---------------
            #
            # Same contract the other fourteen scripts emit
            # ({feature, grid, average, individual}), so the Explain screen's
            # second block draws it with no change. Ordered by the importance
            # computed just above, which is what "the top six" means there.
            pdp = _compute_pdp(
                predictor, df, list(features), task_type,
                [d['feature'] for d in (perm_importance or [])], positive, inverse,
            )

            singles = [m for m in models if not m['is_ensemble']]
            best_single = singles[0]['name'] if singles else None
            deadline = script_started + REQUEST_BUDGET

            # ---- which columns move each class (more than two classes) ------
            #
            # On the best first-level single model rather than the pick: the
            # permutation explainer calls the model (2·features+1) times per
            # row, and an ensemble -- or a second-level model, which runs every
            # first-level model underneath it -- is far slower per call. The
            # meta names the model.
            shap_by_class = None
            shap_by_class_meta = None
            level_one = [m['name'] for m in singles if m['name'].endswith('_L1')]
            shap_model = level_one[0] if level_one else best_single
            if len(labels) > 2 and shap_model and time.time() < deadline - 60:
                shap_started = time.time()
                try:
                    source = holdout_df if holdout_df is not None and len(holdout_df) >= 20 else df
                    explain = source[list(features)]
                    if len(explain) > SHAP_CLASS_ROWS:
                        explain = explain.sample(n=SHAP_CLASS_ROWS, random_state=0)
                    background = df[list(features)]
                    if len(background) > SHAP_CLASS_BACKGROUND:
                        background = background.sample(n=SHAP_CLASS_BACKGROUND, random_state=1)
                    with sink.capturing():
                        shap_by_class = _shap_by_class(predictor, shap_model, explain, background,
                                                       list(features), labels)
                    shap_by_class_meta = {
                        'model': shap_model,
                        'n_rows': int(len(explain)),
                        'rows_from': 'holdout' if source is holdout_df else 'train',
                        'background': int(len(background)),
                        'statistic': 'mean_abs_shap_probability',
                        'seconds': round(time.time() - shap_started, 1),
                    }
                except Exception:
                    shap_by_class = None
                    shap_by_class_meta = None

            # ---- does more data still help? (only when asked) ---------------
            learning_curve = None
            if bool(payload.get('learning_curve')):
                try:
                    with sink.capturing():
                        learning_curve = _learning_curve(df_fit, target, task_type, eval_metric,
                                                         deadline, positive, inverse)
                except Exception:
                    learning_curve = {'skipped': 'failed'}

            # ---- the simple model, read as coefficients ----------------------
            linear_coefficients = None
            linear_meta = None
            if task_type == 'regression' or positive is not None:
                try:
                    with sink.capturing():
                        linear_coefficients, linear_meta = _linear_coefficients(
                            df, target, list(features), task_type, positive)
                except Exception as e:
                    linear_coefficients = None
                    linear_meta = {'reason': 'fit_failed', 'detail': str(e)[:200]}

            # What AutoGluon made of the columns, so the Features screen can
            # stop being a task and start being a record of what happened.
            try:
                used = list(predictor.feature_metadata_in.get_features())
                produced = list(predictor.feature_metadata.get_features())
            except Exception:
                used, produced = list(features), list(features)
            generated = [c for c in produced if c not in features]
            dropped = [c for c in features if c not in used]

            response = {
                'engine': 'autogluon',
                'task_type': task_type,
                'eval_metric': eval_metric,
                'models': models,
                'best_single': best_single,
                'best_overall': models[0]['name'] if models else None,
                # The contract the Explain screen already reads. None when
                # AutoGluon could not compute it -- the screen draws nothing
                # rather than a ranking that is not there.
                'guardrails': guardrails,
                'perm_importance': perm_importance,
                # None for three or more classes: a single curve cannot follow
                # one class there, and the screen draws nothing rather than a
                # line whose meaning changes at the crossing point.
                'pdp': pdp,
                # What those numbers were measured on. 'train' is not the
                # held-out measurement the other scripts report, and the
                # screen has to be able to tell the difference before it
                # repeats their wording.
                'perm_importance_scope': perm_scope,
                # Calibration and per-group figures for the model that would
                # be deployed, measured on the sealed rows. None when there
                # were none to measure on, or too few to mean anything.
                'sealed_checks': sealed_checks,
                # ---- the one-page report (docs/backend-requests-auto-lab-report.md)
                # The model the report recommends: AutoGluon's pick, or, under
                # a target transform, the best in the target's own units.
                'recommended': recommended,
                'class_labels': [_jsonable(c) for c in labels] or None,
                # A. Row by row on the sealed rows, recommended model only.
                'sealed_predictions': sealed_predictions,
                # B. The 90% band /predict will draw, and how often it held.
                'prediction_interval': prediction_interval,
                # G. How models[].cv_results.fold_scores were cut.
                'cv_partition': {
                    'k': CV_PARTITION_K,
                    'method': 'oof_slices',
                    'stratified': task_type == 'classification',
                    'seed': 0,
                },
                # D. More than two classes only.
                'shap_importance_by_class': shap_by_class,
                'shap_importance_by_class_meta': shap_by_class_meta,
                # E. Only when the request asked for it.
                'learning_curve': learning_curve,
                # F.
                'target_transform': {
                    'requested': requested_transform,
                    'applied': transform or 'none',
                    'note': transform_note,
                    # Importance was measured on the scale the models were
                    # fitted on; its order is what carries over, not its size.
                    'importance_scale': transform or 'identity',
                },
                # H. Binary and regression only.
                'linear_coefficients': linear_coefficients,
                'linear_coefficients_meta': linear_meta,
                'preprocessing': {
                    'generated': generated,
                    'dropped': dropped,
                    'final_feature_count': len(produced),
                },
                # AutoGluon optimises the metric it was given, which for an
                # uneven target is PR-AUC rather than accuracy. That is the
                # balancing, and it is why this path needs no class_weight of
                # its own -- but the screen should still be able to say so.
                'class_weighting': {
                    'applied': True,
                    'method': f'AutoGluon eval_metric={eval_metric}',
                    'reason': None,
                },
                'row_counts': {
                    'n_input': n_input,
                    'n_target_missing_dropped': n_target_missing_dropped,
                    'n_duplicates_dropped': n_duplicates_dropped,
                    # AutoGluon owns its own train/validation split inside the
                    # rows it was handed, and does not report the sizes, so
                    # these describe what it received rather than what it made
                    # of it.
                    'n_train': len(df),
                    'n_holdout': len(holdout) if holdout else 0,
                    'n_train_used': len(df),
                    'n_holdout_used': len(holdout_df) if holdout_df is not None else 0,
                },
                'time': {
                    'limit': time_limit,
                    'used': seconds_used,
                    # Within a few seconds of the budget means the search was
                    # still improving when the clock stopped -- worth saying,
                    # because it is the one case where more time would help.
                    'hit_limit': bool(seconds_used >= time_limit - 5),
                },
                # What produced this board, so a rerun can be told whether it
                # ran on the same stack -- the report otherwise has to say it
                # cannot promise the same pick under the same settings.
                'versions': _library_versions(),
                'error': None,
            }
            print(json.dumps(response, default=_to_native_type))
        except Exception as e:
            # AutoGluon's own account of what went wrong was captured rather
            # than printed, so carry the tail of it into the one line the
            # caller does read. Without this the message is whatever the
            # Python exception says, which for a fit that ran out of memory or
            # found no usable model is far less specific than what it logged.
            detail = sink.tail()
            raise RuntimeError(f'{e}\n{detail}' if detail else str(e)) from e
        finally:
            sink.close()
            shutil.rmtree(work_dir, ignore_errors=True)

    except Exception as e:  # noqa: BLE001 — CLI contract: any failure -> stderr + exit(1)
        print(json.dumps({'error': str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
