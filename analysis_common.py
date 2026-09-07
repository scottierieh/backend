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
