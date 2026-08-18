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


def detect_task_type(y: pd.Series) -> str:
    unique_ratio = len(y.unique()) / len(y)
    if not pd.api.types.is_numeric_dtype(y) or y.dtype.name == 'category':
        return 'classification'
    elif len(y.unique()) <= 10 or unique_ratio < 0.05:
        return 'classification'
    else:
        return 'regression'
