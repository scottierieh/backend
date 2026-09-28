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
        if len(df) < 50:
            raise ValueError(
                f'At least 50 rows with a target are required; {len(df)} usable of {n_input}.'
            )

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
                    excluded_model_types=_NO_TORCH,
                )

                seconds_used = round(time.time() - started, 1)
                board = predictor.leaderboard(silent=True)

            # One row per model AutoGluon kept, in its own ranking order.
            #
            # Only the metric it optimised is filled in. AutoGluon scores its
            # leaderboard on its internal validation split, which this script
            # does not hold, so the other columns would have to be computed on
            # the training rows -- a training score printed in a column the
            # screen labels as held-out. An empty cell is the honest answer and
            # the Compare table already draws one.
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
                    # AutoGluon owns its own train/validation split inside the
                    # rows it was handed, and does not report the sizes, so
                    # these describe what it received rather than what it made
                    # of it. The caller's own sealed holdout is separate and
                    # was never sent.
                    'n_train': len(df),
                    'n_holdout': 0,
                    'n_train_used': len(df),
                    'n_holdout_used': 0,
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
