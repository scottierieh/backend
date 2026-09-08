"""
feature_engineering_analysis.py — leakage check + score-based feature
selection for Model Lab's Features step (route: /api/analysis/feature-
engineering).

CLI-script contract (like every *_analysis.py here): read one JSON object
from stdin, print one JSON object to stdout; on error print {"error": ...}
to stderr and exit(1).

Leakage detection reuses guardrails.compute_guardrails() as-is — the same
function every *_analysis.py calls AFTER training — with an empty metrics
dict. Its two structural checks (target_duplicate, leakage_suspect) only
need X and y, not a trained model, so they run just as well before
training; the one check that DOES need a trained score (perfect_score)
simply never fires here, since metrics={}.get(...) always returns None.
Same warning shape as every model script's `guardrails` field, so the
frontend's existing rendering for that shape works here unchanged.

Feature selection uses mutual information (sklearn.feature_selection), not
any one model's importance — it doesn't commit to a particular algorithm's
bias, and the same function family covers both classification and
regression with a task-appropriate variant.
"""

import sys
import json
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder

from analysis_common import _to_native_type
from guardrails import compute_guardrails


def _prepare(df: pd.DataFrame, features: list, task: str):
    """Numeric-coerce + label-encode X for mutual_info_*, which needs a
    fully numeric matrix and an explicit discrete/continuous mask per
    column. Returns (X, is_categorical) — NaNs are median/mode-filled here
    only for the MI calculation; the leakage check below uses the original
    columns, not this filled copy, so a genuinely missing value there still
    reads as missing rather than a filled-in guess."""
    X = pd.DataFrame(index=df.index)
    is_categorical: dict[str, bool] = {}
    for col in features:
        if pd.api.types.is_numeric_dtype(df[col]):
            X[col] = pd.to_numeric(df[col], errors='coerce')
            is_categorical[col] = False
        else:
            X[col] = LabelEncoder().fit_transform(df[col].astype(str))
            is_categorical[col] = True
        if X[col].isna().any():
            fill = X[col].median() if not is_categorical[col] else X[col].mode().iloc[0]
            X[col] = X[col].fillna(fill)
    return X, is_categorical


def main():
    try:
        payload = json.load(sys.stdin)
        data = payload.get('data')
        features = payload.get('features')
        target = payload.get('target')
        task = payload.get('task', 'classification')

        if not data or not features or not target:
            raise ValueError("Missing data, features, or target")

        df = pd.DataFrame(data)
        missing = [c for c in list(features) + [target] if c not in df.columns]
        if missing:
            raise ValueError(f"Column(s) not found: {missing}")

        X, is_categorical = _prepare(df, features, task)

        # Leakage — same function/shape every model script's response
        # already carries under `guardrails`, called with the RAW target
        # column (compute_guardrails does its own numeric coercion).
        leakage_warnings = compute_guardrails(X, df[target], features, task, {})

        if task == 'classification':
            y = LabelEncoder().fit_transform(df[target].astype(str))
        else:
            y = pd.to_numeric(df[target], errors='coerce').fillna(0).values

        discrete_mask = [is_categorical[c] for c in features]
        mi_func = mutual_info_classif if task == 'classification' else mutual_info_regression
        try:
            mi = mi_func(X[features].values, y, discrete_features=discrete_mask, random_state=42)
        except Exception:
            mi = np.zeros(len(features))

        order = np.argsort(mi)[::-1]
        feature_scores = [
            {
                'feature': features[i],
                'mutual_info': _to_native_type(float(mi[i])),
                'rank': int(rank) + 1,
            }
            for rank, i in enumerate(order)
        ]

        print(json.dumps({
            'leakage_warnings': leakage_warnings,
            'feature_scores': feature_scores,
        }, default=_to_native_type))

    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
