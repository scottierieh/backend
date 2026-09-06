"""
tune_analysis.py — Model Lab "Tune the winner" backend (route: /api/analysis/tune).

Powers src/components/model-lab-tune-panel.tsx: after Auto Compare finds the top
model, this re-tunes THAT model's hyperparameters within a time-budget preset and
re-runs the leakage guardrails so a higher score can't quietly hide leakage.

Distinct from hyperparameter_tuning_analysis.py (the standalone wizard page): this
one speaks the TunePanel contract (model key + preset in, best_score/improvement/
guardrails out) and supports the three tunable Auto-Compare winners — XGBoost,
Random Forest, GBM — including XGBoost, which the standalone sklearn-only script
does not cover.

CLI-script contract (like every *_analysis.py here): read one JSON object from
stdin, print one JSON object to stdout; on error print {"error": ...} to stderr
and exit(1).
"""

import sys
import json
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, r2_score
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    GradientBoostingClassifier, GradientBoostingRegressor,
)

try:
    from xgboost import XGBClassifier, XGBRegressor
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

# Reuse the shared leakage/imbalance/perfect-score guardrails (guardrails.py in
# repo root) — the same module wired into all 14 Auto Compare scripts, so the
# post-tuning re-check is identical to the pre-tuning one.
from guardrails import compute_guardrails


def _to_native(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# TunePanel sends the Auto Compare def.route as `model`, mapped there to these keys:
#   'xgboost' -> 'xgboost', 'randomforest' -> 'random-forest', 'gradient-boosting' -> 'gbm'
# Accept those exact strings (plus a few natural aliases). Each entry builds a fresh
# default estimator (the baseline) and a search space for RandomizedSearchCV.
def _model_space(model_key, task_type):
    key = (model_key or '').lower()
    is_clf = task_type == 'classification'

    if key in ('random-forest', 'random_forest', 'randomforest', 'rf'):
        est = RandomForestClassifier(random_state=42) if is_clf else RandomForestRegressor(random_state=42)
        space = {
            'n_estimators': [100, 200, 300, 400, 500],
            'max_depth': [3, 5, 10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
        }
        return est, space

    if key in ('gbm', 'gradient-boosting', 'gradient_boosting', 'gradientboosting'):
        est = GradientBoostingClassifier(random_state=42) if is_clf else GradientBoostingRegressor(random_state=42)
        space = {
            'n_estimators': [100, 200, 300, 400],
            'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
            'max_depth': [2, 3, 4, 5],
            'subsample': [0.7, 0.85, 1.0],
        }
        return est, space

    if key in ('xgboost', 'xgb'):
        if not _HAS_XGB:
            raise ValueError("XGBoost is not installed on the server")
        common = dict(random_state=42, n_jobs=-1, verbosity=0)
        est = (XGBClassifier(eval_metric='logloss', **common) if is_clf
               else XGBRegressor(**common))
        space = {
            'n_estimators': [100, 200, 300, 400, 500],
            'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2],
            'max_depth': [3, 4, 5, 6, 8],
            'subsample': [0.7, 0.85, 1.0],
            'colsample_bytree': [0.7, 0.85, 1.0],
            'min_child_weight': [1, 3, 5],
        }
        return est, space

    raise ValueError(f"Unsupported model '{model_key}' — tunable models are xgboost, random-forest, gbm")


# Preset -> search effort. RandomizedSearchCV has no native wall-clock budget, so the
# preset maps to (n_iter candidates, cv folds); the "~N min" the UI shows is the rough
# envelope these produce on typical Model Lab datasets. Kept well within the backend's
# 600s Cloud Run request timeout even for 'thorough'.
_PRESETS = {
    'fast':     {'n_iter': 15, 'cv': 3, 'label': 'Fast'},
    'balanced': {'n_iter': 40, 'cv': 5, 'label': 'Balanced'},
    'thorough': {'n_iter': 80, 'cv': 5, 'label': 'Thorough'},
}


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
        model_key = payload.get('model') or payload.get('model_type')
        preset_key = (payload.get('preset') or 'balanced').lower()
        task_req = payload.get('task_type', 'auto')

        if not all([data, features, target, model_key]):
            raise ValueError("Missing data, features, target, or model")

        preset = _PRESETS.get(preset_key, _PRESETS['balanced'])

        df = pd.DataFrame(data).dropna(subset=[target])
        task_type = task_req if task_req in ('classification', 'regression') else _detect_task_type(df[target])

        # Raw feature frame + target — kept for the guardrail leakage check, which wants
        # original columns vs the original target (it does its own numeric coercion).
        X_raw = df[features].copy()
        y_raw = df[target]
        if task_type == 'regression':
            y_num = pd.to_numeric(y_raw, errors='coerce')
            mask = y_num.notna()
            X_raw, y_raw, y_num = X_raw[mask], y_raw[mask], y_num[mask]
            y = y_num.to_numpy(dtype=float)
        else:
            # Label-encode so XGBoost (which requires numeric labels) works alongside sklearn.
            y = LabelEncoder().fit_transform(y_raw.astype(str))

        X_raw = X_raw.reset_index(drop=True)
        y_raw = y_raw.reset_index(drop=True)
        X = pd.get_dummies(X_raw, drop_first=True)
        if X.shape[1] == 0:
            raise ValueError("No usable features after encoding")

        scoring = 'f1_weighted' if task_type == 'classification' else 'r2'
        report_scoring = 'f1' if task_type == 'classification' else 'r2'

        def _score(y_true, y_pred):
            return (f1_score(y_true, y_pred, average='weighted', zero_division=0)
                    if task_type == 'classification' else r2_score(y_true, y_pred))

        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.25, random_state=42,
                stratify=y if task_type == 'classification' else None,
            )
        except ValueError:
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42)

        estimator, space = _model_space(model_key, task_type)

        # Baseline = the model's default hyperparameters on the same split, so
        # "before -> after" is an apples-to-apples comparison on identical data.
        baseline_est, _ = _model_space(model_key, task_type)
        baseline_est.fit(X_tr, y_tr)
        baseline_score = float(_score(y_te, baseline_est.predict(X_te)))

        n_iter = preset['n_iter']
        started = time.time()
        search = RandomizedSearchCV(
            estimator, space, n_iter=n_iter, cv=preset['cv'],
            scoring=scoring, random_state=42, n_jobs=-1, refit=True,
        )
        search.fit(X_tr, y_tr)
        seconds_used = round(time.time() - started, 1)

        best_model = search.best_estimator_
        tuned_score = float(_score(y_te, best_model.predict(X_te)))

        # Post-tuning guardrail re-check — identical logic to Auto Compare's pre-tuning
        # pass, on the raw features/target, so a leakage-inflated gain can't slip through.
        metrics = {'accuracy': tuned_score} if task_type == 'classification' else {'r2': tuned_score}
        guardrails = compute_guardrails(X_raw, y_raw, features, task_type, metrics)

        response = {
            'model': model_key,
            'task_type': task_type,
            'scoring': report_scoring,
            'preset_label': preset['label'],
            'baseline_score': baseline_score,
            'best_score': tuned_score,
            'improvement': tuned_score - baseline_score,
            'best_params': {k: _to_native(v) for k, v in search.best_params_.items()},
            'n_trials': int(len(search.cv_results_['params'])),
            'seconds_used': seconds_used,
            'model_id': None,  # no model persistence yet — panel treats this as optional
            'guardrails': guardrails,
        }
        print(json.dumps(response, default=_to_native))

    except Exception as e:  # noqa: BLE001 — CLI contract: any failure -> stderr + exit(1)
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
