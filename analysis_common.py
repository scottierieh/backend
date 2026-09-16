"""
Shared helpers duplicated (byte-for-byte, in this cluster of files) across
several *_analysis.py CLI scripts. Extracted per audit finding H3/M2 —
jscpd measured 24.33% duplication across the 58 top-level analysis
scripts, with these four functions among the most-repeated blocks.

Not every *_analysis.py file uses the exact same implementation of each of
these — some have their own variant (different rounding, different
fallback behavior, etc.), so this module is only imported by files whose
version was verified byte-identical to the one here. Don't blanket-replace
a file's local definition with an import from here without diffing first;
see docs/model-lab-python-backend-updates.md §3 for why that matters
("SAME = fine, DIFFERENT = inspect manually").
"""

import io
import base64
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _compute_multiclass_auc(y_true, y_pred_proba):
    """Macro-average ROC-AUC: binary uses the positive-class column; multiclass uses One-vs-Rest macro averaging."""
    try:
        n_classes = y_pred_proba.shape[1]
        if n_classes == 2:
            return float(roc_auc_score(y_true, y_pred_proba[:, 1]))
        else:
            return float(roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro'))
    except Exception:
        return None


def _to_native_type(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return float(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _fig_to_base64(fig) -> str:
    import matplotlib.pyplot as plt
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', dpi=120, bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close(fig)
    return image_base64


def build_error_examples(X_test, y_test, y_pred, feature_names, y_pred_proba=None, max_examples: int = 20):
    """Misclassified test rows for the Error Analysis view (Model Lab 2's
    Evaluate > Explain tab reads this via the existing errorExamples field —
    src/lib/types/model-result.ts:145 — which was already wired end to end
    from auto-compare-engine.ts down to Firestore, just never populated by
    any backend script until now: `r.norm?.error_examples`).

    Classification only — regression's analog is the already-shipped
    actual-vs-predicted scatter (regressionDiag), not a misclassification
    list. X_test may be a DataFrame or ndarray; y_test/y_pred are 1D
    arrays/Series of the (label-encoded or raw) class values actually
    compared, so 'actual'/'predicted' are whatever those values are —
    callers that label-encoded y should pass the decoded (original-label)
    versions in, not the encoded ints, so the UI shows real class names.
    """
    try:
        y_test_arr = np.asarray(y_test)
        y_pred_arr = np.asarray(y_pred)
        wrong = np.where(y_test_arr != y_pred_arr)[0]
        if len(wrong) == 0:
            return []
        if len(wrong) > max_examples:
            # Deterministic, evenly-spread sample rather than just the first N,
            # so a long run of one confused class near the top of the test set
            # doesn't crowd out every other kind of mistake.
            idx = np.linspace(0, len(wrong) - 1, max_examples).round().astype(int)
            wrong = wrong[idx]

        X_arr = X_test.values if hasattr(X_test, 'values') else np.asarray(X_test)
        out = []
        for i in wrong:
            row = {
                'index': int(i),
                'actual': _to_native_type(y_test_arr[i]),
                'predicted': _to_native_type(y_pred_arr[i]),
            }
            if y_pred_proba is not None:
                proba = np.asarray(y_pred_proba)
                row['confidence'] = _to_native_type(float(proba[i].max())) if proba.ndim == 2 else _to_native_type(float(proba[i]))
            if feature_names is not None:
                for j, name in enumerate(feature_names):
                    row[name] = _to_native_type(X_arr[i, j])
            out.append(row)
        return out
    except Exception:
        return None


def detect_task_type(y: pd.Series) -> str:
    unique_ratio = len(y.unique()) / len(y)
    if not pd.api.types.is_numeric_dtype(y) or y.dtype.name == 'category':
        return 'classification'
    elif len(y.unique()) <= 10 or unique_ratio < 0.05:
        return 'classification'
    else:
        return 'regression'


# The space `base_value + sum(shap)` lands in, per script. Measured rather than
# assumed: statistica-frontend's scripts/check-shap-space.py rebuilds each
# explainer the way its script builds it and checks whether that sum equals
# predict_proba or its logit.
#
#   probability  TreeExplainer on sklearn's own trees (random forest, decision
#                tree), and shap.Explainer over predict_proba (adaboost,
#                stacking, mlp) -- the sum IS the predicted probability.
#   log_odds     TreeExplainer on a boosted raw margin (xgboost, lightgbm,
#                gradient boosting) and CatBoost's native ShapValues.
#
# This matters because a caller cannot tell from the numbers. A waterfall bar
# labelled "+21%p" is only honest in probability space, and in log-odds space a
# per-variable percentage point is not merely mis-scaled but undefined: convert
# the contributions one at a time and the answer changes with the ordering. So
# the consumer needs to be told, and inferring it from the algorithm's name
# goes stale silently the day one of these scripts changes explainer.
SHAP_SPACE_PROBABILITY = 'probability'
SHAP_SPACE_LOG_ODDS = 'log_odds'
SHAP_SPACE_RAW = 'raw'


def shap_contract(space: str, task_type: str, class_labels=None, explained_index: int = 1):
    """The three fields that say how to read this script's SHAP values.

    `space` is one of the SHAP_SPACE_* constants above, naming what the
    contributions are denominated in for a CLASSIFIER. A regression model's
    SHAP is in the target's own units whatever the explainer, so `task_type`
    overrides it.

    `class_labels` and `explained_index` say WHICH class the contributions
    explain. Every script here takes the positive class's slice for binary and
    skips 3+ classes outright, so the index is a fact the script knows and the
    caller would otherwise be guessing from array order -- a guess that breaks
    the day multiclass lands. The label is what a UI actually prints.
    """
    if task_type != 'classification':
        return {
            'shap_output_space': SHAP_SPACE_RAW,
            'shap_explained_class': None,
            'shap_explained_label': None,
        }

    label = None
    if class_labels is not None:
        try:
            labels = list(class_labels)
            if 0 <= explained_index < len(labels):
                label = str(labels[explained_index])
        except TypeError:
            label = None

    return {
        'shap_output_space': space,
        'shap_explained_class': explained_index,
        'shap_explained_label': label,
    }


def shap_matrix(sv, X_arr, feature_names, base_value, max_rows: int = 200):
    """The per-sample SHAP matrix a summary beeswarm and a dependence plot need.

    `shap_samples` carries up to 8 rows, which is one waterfall and nothing
    else: a beeswarm of 8 dots says nothing about a distribution, and a
    dependence cloud of 8 points is not a cloud. This returns the same
    information for up to `max_rows` rows, as columns rather than per-row
    dicts so the payload stays small:

        {features: [...], shap: [[...], ...], values: [[...], ...],
         base_value: float, n_rows: int}

    `shap[i][j]` and `values[i][j]` are row i's contribution and own value for
    features[j]. Rows are subsampled deterministically when there are more than
    max_rows, so two calls on the same fit agree.

    Returns None rather than a partial payload when the shapes do not line up.
    """
    try:
        sv = np.asarray(sv, dtype=float)
        if sv.ndim == 3:
            # binary classification: the positive class's slice, the same
            # convention as the confusion matrix and the ROC curve
            if sv.shape[2] != 2:
                return None
            sv = sv[:, :, 1]
        if sv.ndim != 2 or sv.shape[1] != len(feature_names):
            return None

        X_arr = np.asarray(getattr(X_arr, 'values', X_arr))
        if X_arr.ndim != 2 or X_arr.shape[1] != len(feature_names):
            return None
        n = min(sv.shape[0], X_arr.shape[0])
        if n == 0:
            return None

        idx = np.arange(n)
        if n > max_rows:
            idx = np.random.RandomState(42).choice(n, size=max_rows, replace=False)
            idx.sort()

        base = np.ravel(np.asarray(base_value, dtype=object))
        base = float(base[-1]) if base.size > 1 else float(base[0])

        return {
            'features': list(feature_names),
            'base_value': _to_native_type(base),
            'n_rows': int(len(idx)),
            'shap': [[_to_native_type(v) for v in sv[i]] for i in idx],
            'values': [[_to_native_type(v) for v in X_arr[i]] for i in idx],
        }
    except Exception:
        return None


def ale_1d(predict_fn, X_arr, feature_names, feature_indices, n_bins: int = 20):
    """Accumulated Local Effects, one feature at a time.

    Why this and not just PDP: a partial dependence plot moves one feature
    across its range while holding the others at the values they actually have,
    which manufactures rows the data never contained -- a 30-year-old with 40
    years of experience -- and averages the model's answer over them. When the
    predictors are correlated, and in research data they usually are, that
    average is partly about combinations that do not exist.

    ALE avoids it by only ever asking about a row against the edges of the bin
    it is already in: for each bin, the model is evaluated with the feature set
    to that bin's lower and upper edge, the differences are averaged over the
    rows in that bin, and those local differences are accumulated across bins.
    Nothing is evaluated outside the data's own joint distribution.

    The curve is centred on its own weighted mean, so it reads as a deviation
    ("this value moves the prediction +0.08 relative to average"), not as a
    predicted level. Returns [{feature, grid, ale, counts}] -- `counts` is the
    rows per bin, so a frontend can mark the sparse end of a curve rather than
    drawing it with the same confidence as the middle.
    """
    out = []
    try:
        X_arr = np.asarray(getattr(X_arr, 'values', X_arr), dtype=float)
    except (TypeError, ValueError):
        return out

    for j in feature_indices:
        try:
            col = X_arr[:, j]
            finite = col[np.isfinite(col)]
            if finite.size < 20:
                continue
            # Quantile edges, deduplicated: a feature with few distinct values
            # (a count, a Likert item) would otherwise get empty bins.
            qs = np.linspace(0, 100, n_bins + 1)
            edges = np.unique(np.percentile(finite, qs))
            if edges.size < 3:
                continue

            # Each row's bin, by its own value. Rows below the first edge or
            # above the last belong to the end bins.
            bin_of = np.clip(np.searchsorted(edges, col, side='left'), 1, edges.size - 1)

            deltas = np.zeros(edges.size - 1, dtype=float)
            counts = np.zeros(edges.size - 1, dtype=int)
            for b in range(1, edges.size):
                rows = np.where(bin_of == b)[0]
                counts[b - 1] = rows.size
                if rows.size == 0:
                    continue
                lo = X_arr[rows].copy()
                hi = X_arr[rows].copy()
                lo[:, j] = edges[b - 1]
                hi[:, j] = edges[b]
                p_lo = np.asarray(predict_fn(lo), dtype=float)
                p_hi = np.asarray(predict_fn(hi), dtype=float)
                if p_lo.ndim == 2:
                    # predict_proba: the positive class, as everywhere else
                    p_lo, p_hi = p_lo[:, -1], p_hi[:, -1]
                deltas[b - 1] = float(np.mean(p_hi - p_lo))

            # Accumulate, then centre on the weighted mean so the curve is a
            # deviation from the average prediction rather than a level.
            acc = np.concatenate([[0.0], np.cumsum(deltas)])
            weights = np.concatenate([[counts[0]], counts])
            if weights.sum() > 0:
                acc = acc - float(np.average(acc, weights=weights))

            out.append({
                'feature': feature_names[j],
                'grid': [_to_native_type(v) for v in edges],
                'ale': [_to_native_type(v) for v in acc],
                'counts': [int(c) for c in counts],
            })
        except Exception:
            continue
    return out


def shap_interaction_top(explainer, X_arr, feature_names, top_n: int = 10,
                         max_rows: int = 60):
    """Pairwise SHAP interaction strength, strongest pairs first.

    `shap_interaction_values` returns an (n, p, p) tensor whose off-diagonal
    (i, a, b) is how much of row i's prediction is attributable to features a
    and b *jointly*, beyond what either contributes alone. The strength of a
    pair is the mean absolute off-diagonal over rows.

    It costs O(n · p²) tree traversals, hence max_rows: 60 rows is enough to
    rank pairs and keeps a wide model from timing out. Returns
    [{feature_1, feature_2, score, rank}] -- deliberately the same shape
    CatBoost's native type='Interaction' already produces, so a caller ranks
    both the same way, and `method` says which one it is looking at.
    """
    try:
        X_arr = np.asarray(getattr(X_arr, 'values', X_arr))
        n = min(max_rows, X_arr.shape[0])
        if n < 5 or len(feature_names) < 2:
            return []
        iv = np.asarray(explainer.shap_interaction_values(X_arr[:n]), dtype=float)
        if iv.ndim == 4:
            if iv.shape[3] != 2:
                return []
            iv = iv[:, :, :, 1]
        if iv.ndim != 3 or iv.shape[1] != iv.shape[2] != len(feature_names):
            return []

        strength = np.abs(iv).mean(axis=0)
        rows = []
        for a in range(len(feature_names)):
            for b in range(a + 1, len(feature_names)):
                # (a,b) and (b,a) each hold half the pair's attribution
                rows.append({
                    'feature_1': feature_names[a],
                    'feature_2': feature_names[b],
                    'score': _to_native_type(float(strength[a, b] + strength[b, a])),
                })
        rows.sort(key=lambda r: r['score'] or 0.0, reverse=True)
        rows = rows[:top_n]
        for i, r in enumerate(rows):
            r['rank'] = i + 1
        return rows
    except Exception:
        return []
