"""
model_diagnostics.py — shared Tier-B diagnostics for Model Lab scripts.

Like guardrails.py / cv_strategy.py: one place computing the "deeper" diagnostics
so every model script gets them by calling in, instead of copy-pasting into 14
files. All functions are defensive (return {} / [] / None on any failure) so a
diagnostic never crashes an analysis.

Functions:
  bootstrap_ci(...)     — #14 confidence interval on the test metric (resampling)
  calibration_curve(...) — #4  are predicted probabilities well-calibrated?
  pr_curve(...)          — #4  precision-recall curve points
  error_examples(...)    — #9  the rows the model got most confidently wrong
"""

import numpy as np


def _to_native(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


def bootstrap_ci(y_true, y_pred, task_type, n_boot=500, alpha=0.05, seed=42):
    """95% CI for the primary metric (accuracy / r2) via bootstrap resampling of the
    test set. Returns {'metric', 'point', 'low', 'high'} or {} on failure."""
    try:
        from sklearn.metrics import accuracy_score, r2_score
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        n = len(y_true)
        if n < 20:
            return {}
        metric = 'accuracy' if task_type == 'classification' else 'r2'
        score = accuracy_score if task_type == 'classification' else r2_score
        rng = np.random.RandomState(seed)
        stats = []
        for _ in range(n_boot):
            idx = rng.randint(0, n, n)
            try:
                stats.append(score(y_true[idx], y_pred[idx]))
            except Exception:
                continue
        if not stats:
            return {}
        lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return {
            'metric': metric,
            'point': _to_native(float(score(y_true, y_pred))),
            'low': _to_native(float(lo)),
            'high': _to_native(float(hi)),
            'n_boot': int(n_boot),
        }
    except Exception:
        return {}


def calibration_curve(y_true, y_proba, n_bins=10):
    """Reliability curve for binary classification: mean predicted prob vs actual
    positive rate per bin. Returns [{'pred', 'actual', 'count'}] or [] if not binary."""
    try:
        y_true = np.asarray(y_true).astype(int)
        y_proba = np.asarray(y_proba, dtype=float)
        if y_proba.ndim == 2:
            if y_proba.shape[1] != 2:
                return []           # calibration curve is a binary concept
            y_proba = y_proba[:, 1]
        if len(np.unique(y_true)) != 2:
            return []
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        out = []
        for b in range(n_bins):
            mask = (y_proba >= bins[b]) & (y_proba < bins[b + 1] if b < n_bins - 1 else y_proba <= bins[b + 1])
            cnt = int(mask.sum())
            if cnt == 0:
                continue
            out.append({
                'pred': _to_native(float(y_proba[mask].mean())),
                'actual': _to_native(float(y_true[mask].mean())),
                'count': cnt,
            })
        return out
    except Exception:
        return []


def pr_curve(y_true, y_proba, max_points=50):
    """Precision-recall curve for binary classification, thinned to max_points."""
    try:
        from sklearn.metrics import precision_recall_curve
        y_true = np.asarray(y_true).astype(int)
        y_proba = np.asarray(y_proba, dtype=float)
        if y_proba.ndim == 2:
            if y_proba.shape[1] != 2:
                return []
            y_proba = y_proba[:, 1]
        if len(np.unique(y_true)) != 2:
            return []
        prec, rec, _ = precision_recall_curve(y_true, y_proba)
        step = max(1, len(prec) // max_points)
        return [{'recall': _to_native(float(rec[i])), 'precision': _to_native(float(prec[i]))}
                for i in range(0, len(prec), step)]
    except Exception:
        return []


def error_examples(y_true, y_pred, y_proba=None, feature_names=None, X=None, top_k=10):
    """The rows the model got most confidently wrong (#9). For classification, ranks
    misclassified rows by the probability the model assigned to its (wrong) prediction.
    Returns [{'index', 'actual', 'predicted', 'confidence', <features...>}]."""
    try:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        wrong = np.where(y_true != y_pred)[0]
        if len(wrong) == 0:
            return []
        conf = None
        if y_proba is not None:
            proba = np.asarray(y_proba, dtype=float)
            if proba.ndim == 2:
                conf = proba.max(axis=1)          # confidence in the predicted class
        order = wrong[np.argsort(-conf[wrong])] if conf is not None else wrong
        out = []
        for i in order[:top_k]:
            row = {
                'index': int(i),
                'actual': _to_native(y_true[i]),
                'predicted': _to_native(y_pred[i]),
                'confidence': _to_native(float(conf[i])) if conf is not None else None,
            }
            if X is not None and feature_names is not None:
                try:
                    vals = X.iloc[int(i)] if hasattr(X, 'iloc') else X[int(i)]
                    for fn in list(feature_names)[:6]:
                        row[str(fn)] = _to_native(vals[fn] if hasattr(vals, '__getitem__') else None)
                except Exception:
                    pass
            out.append(row)
        return out
    except Exception:
        return []
