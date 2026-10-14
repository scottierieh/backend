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
                 order: list, positive_class):
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


def _detect_task_type(y: pd.Series) -> str:
    vals = y.dropna()
    if vals.empty:
        return 'classification'
    if pd.api.types.is_numeric_dtype(vals) and vals.nunique() > 20:
        return 'regression'
    return 'classification'


def main():
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
        if holdout:
            h_all = pd.DataFrame(holdout)
            h_missing = [c for c in [target] + list(features) if c not in h_all.columns]
            if h_missing:
                raise ValueError(f'Holdout rows are missing column(s): {h_missing}')
            holdout_df = h_all[list(features) + [target]].dropna(subset=[target]).reset_index(drop=True)

        task_type = task_req if task_req in ('classification', 'regression') else _detect_task_type(df[target])

        eval_metric = payload.get('eval_metric') or _DEFAULT_METRIC[task_type]
        if eval_metric not in _METRIC_FIELD:
            raise ValueError(
                f"Unsupported eval_metric '{eval_metric}' — "
                f"expected one of {sorted(_METRIC_FIELD)}"
            )
        metric_field = _METRIC_FIELD[eval_metric]

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
                    df,
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
                sealed_board = None
                if holdout_df is not None and len(holdout_df) > 0:
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
                        'cv_mean': score,
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

            # ---- what each model would cost to run, on the sealed rows -----
            #
            # The ranking metric cannot answer "which model is cheapest",
            # because it does not know what a miss costs against a wrong call.
            # That answer needs counts at a cut, for every model, on rows none
            # of them were chosen on -- so it is computed here, where the
            # predictor and the sealed rows are both in hand, rather than
            # refitting anything later. The screen multiplies by the two
            # amounts the reader enters; nothing here assumes a currency or a
            # price.
            sealed_checks = None
            if (holdout_df is not None and len(holdout_df) > 0
                    and task_type == 'classification'):
                try:
                    labels_h = list(predictor.class_labels or [])
                    if len(labels_h) == 2:
                        pos_h = (predictor.positive_class
                                 if predictor.positive_class in labels_h else labels_h[-1])
                        y_h = (holdout_df[target] == pos_h).astype(int).to_numpy()
                        if len(np.unique(y_h)) == 2:
                            cap = _sealed_cap(len(models))
                            for m in models:
                                try:
                                    pr = predictor.predict_proba(holdout_df, model=m['name'])
                                    if pos_h not in pr.columns:
                                        continue
                                    score_h = pr[pos_h].to_numpy()
                                    cuts = sorted({round(float(c), 4) for c in
                                                   np.quantile(score_h, np.linspace(0.05, 0.95, SEALED_CUTS))
                                                   if 0.0 < float(c) < 1.0})
                                    if cuts:
                                        m['sealed_points'] = _points_at(y_h, score_h, cuts)
                                    m['sealed_scored'] = _sealed_scored(y_h, score_h, cap)
                                except Exception:
                                    continue
                            # The calibration and subgroup checks describe the
                            # model that would be deployed, so they are
                            # computed once for AutoGluon's own pick rather
                            # than for every row on the board.
                            best_name = str(board.iloc[0].get('model')) if len(board) else None
                            if best_name:
                                pr = predictor.predict_proba(holdout_df, model=best_name)
                                if pos_h in pr.columns:
                                    score_b = pr[pos_h].to_numpy()
                                    pred_b = (predictor.predict(holdout_df, model=best_name)
                                              == pos_h).astype(int).to_numpy()
                                    sealed_checks = {
                                        'model': best_name,
                                        'n': int(len(y_h)),
                                        'positive_class': _to_native_type(pos_h),
                                        'calibration': _calibration(y_h, score_b),
                                        'subgroups': _subgroups(holdout_df, list(features), y_h, pred_b),
                                    }
                except Exception:
                    sealed_checks = None

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
                    df, features=ask or None, num_shuffle_sets=10, silent=True,
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
            positive = None
            if task_type == 'classification':
                labels = list(predictor.class_labels or [])
                if len(labels) == 2:
                    positive = getattr(predictor, 'positive_class', None)
                    if positive is None or positive not in labels:
                        positive = labels[-1]
            pdp = _compute_pdp(
                predictor, df, list(features), task_type,
                [d['feature'] for d in (perm_importance or [])], positive,
            )

            singles = [m for m in models if not m['is_ensemble']]
            best_single = singles[0]['name'] if singles else None

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
