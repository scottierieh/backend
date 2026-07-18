# ─────────────────────────────────────────────────────────────────────────────
# Shared regression-diagnostic sampler.
#
# Regression analyses already render actual-vs-predicted / residual plots as
# base64 PNGs, but the frontend (Model Lab 2 Evaluate) needs the raw (actual,
# predicted) pairs to draw clean interactive Predicted-vs-Actual / Residual / Q–Q
# charts. This returns a capped, JSON-safe sample of those pairs so the response
# stays small (Firestore 1MB doc limit downstream).
# ─────────────────────────────────────────────────────────────────────────────

import numpy as np
from typing import Any, Dict, Optional


def regression_sample(y_true: Any, y_pred: Any, cap: int = 500) -> Optional[Dict[str, list]]:
    """Return {'y_true_sample': [...], 'y_pred_sample': [...]} evenly downsampled to
    at most `cap` points, or None if the inputs are unusable. Non-finite pairs are
    dropped; both arrays are returned as plain Python floats."""
    try:
        yt = np.asarray(y_true, dtype=float).ravel()
        yp = np.asarray(y_pred, dtype=float).ravel()
    except (ValueError, TypeError):
        return None
    if yt.size < 2 or yt.size != yp.size:
        return None
    mask = np.isfinite(yt) & np.isfinite(yp)
    yt, yp = yt[mask], yp[mask]
    if yt.size < 2:
        return None
    if yt.size > cap:
        idx = np.linspace(0, yt.size - 1, cap).astype(int)
        yt, yp = yt[idx], yp[idx]
    return {
        'y_true_sample': [float(x) for x in yt],
        'y_pred_sample': [float(x) for x in yp],
    }
