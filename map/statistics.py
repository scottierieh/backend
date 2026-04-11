"""
routers/statistics.py

POST /api/stats/hotspot        — Getis-Ord Gi* hot/cold spot detection
POST /api/stats/autocorr       — Moran's I spatial autocorrelation
POST /api/stats/regression     — Spatial OLS regression
POST /api/stats/mlscore        — XGBoost location scoring + SHAP explanations
"""

import math
import statistics as pystats
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from map.utils import haversine, centroid, to_native

router = APIRouter()


class BaseMapRequest(BaseModel):
    data: List[Dict[str, Any]]


# ══════════════════════════════════════════════════════════
# Getis-Ord Gi* Hot Spot Analysis
# ══════════════════════════════════════════════════════════

class HotspotRequest(BaseMapRequest):
    valueCol: str
    distanceM: float = 1000    # spatial weight threshold
    significance: float = 0.05


@router.post("/api/stats/hotspot")
def run_hotspot(req: HotspotRequest):
    from scipy import stats

    pts = [r for r in req.data
           if r.get("lat") and r.get("lng")
           and isinstance(r.get(req.valueCol), (int, float))]

    if len(pts) < 4:
        raise HTTPException(400, "Need at least 4 points with numeric values.")

    n = len(pts)
    values = np.array([float(p[req.valueCol]) for p in pts])
    mean_v = float(np.mean(values))
    std_v = float(np.std(values))
    if std_v == 0:
        raise HTTPException(400, "All values are identical — cannot compute Gi*.")

    # Spatial weight matrix (binary distance threshold)
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                d = haversine(pts[i]["lat"], pts[i]["lng"],
                              pts[j]["lat"], pts[j]["lng"])
                if d <= req.distanceM:
                    W[i][j] = 1.0

    # Gi* statistic
    results = []
    for i in range(n):
        w_i = W[i]
        sum_w = float(np.sum(w_i))
        if sum_w == 0:
            gi_star = 0.0
            p_value = 1.0
        else:
            num = float(np.sum(w_i * values)) - mean_v * sum_w
            s = float(np.sqrt((np.sum(values ** 2) / n) - mean_v ** 2))
            if s == 0:
                gi_star = 0.0
                p_value = 1.0
            else:
                denom = s * math.sqrt(
                    (n * np.sum(w_i ** 2) - sum_w ** 2) / (n - 1)
                )
                gi_star = num / denom if denom != 0 else 0.0
                p_value = float(2 * (1 - stats.norm.cdf(abs(gi_star))))

        significant = p_value < req.significance
        if gi_star > 0 and significant:
            spot_type = "hot"
        elif gi_star < 0 and significant:
            spot_type = "cold"
        else:
            spot_type = "neutral"

        results.append({
            "row": pts[i],
            "giStar": round(gi_star, 4),
            "pValue": round(p_value, 4),
            "significant": significant,
            "spotType": spot_type,
            "value": float(pts[i][req.valueCol]),
        })

    hot_count = sum(1 for r in results if r["spotType"] == "hot")
    cold_count = sum(1 for r in results if r["spotType"] == "cold")

    # Color mapping for LeafletMap
    for r in results:
        if r["spotType"] == "hot":
            intensity = min(abs(r["giStar"]) / 3, 1.0)
            r["color"] = f"rgba(239,68,68,{0.4 + intensity * 0.6:.2f})"
        elif r["spotType"] == "cold":
            intensity = min(abs(r["giStar"]) / 3, 1.0)
            r["color"] = f"rgba(59,130,246,{0.4 + intensity * 0.6:.2f})"
        else:
            r["color"] = "rgba(156,163,175,0.4)"

    return to_native({
        "results": {
            "points": results,
            "hotCount": hot_count,
            "coldCount": cold_count,
            "neutralCount": n - hot_count - cold_count,
            "valueCol": req.valueCol,
            "distanceM": req.distanceM,
            "significance": req.significance,
            "visible": True,
        }
    })


# ══════════════════════════════════════════════════════════
# Moran's I — Spatial Autocorrelation
# ══════════════════════════════════════════════════════════

class MoranRequest(BaseMapRequest):
    valueCol: str
    distanceM: float = 1000


@router.post("/api/stats/autocorr")
def run_morans_i(req: MoranRequest):
    from scipy import stats

    pts = [r for r in req.data
           if r.get("lat") and r.get("lng")
           and isinstance(r.get(req.valueCol), (int, float))]

    if len(pts) < 4:
        raise HTTPException(400, "Need at least 4 points.")

    n = len(pts)
    values = np.array([float(p[req.valueCol]) for p in pts])
    mean_v = np.mean(values)
    z = values - mean_v

    # Weight matrix
    W = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                d = haversine(pts[i]["lat"], pts[i]["lng"],
                              pts[j]["lat"], pts[j]["lng"])
                if d <= req.distanceM:
                    W[i][j] = 1.0

    W_sum = float(np.sum(W))
    if W_sum == 0:
        raise HTTPException(400, "No point pairs within distance threshold.")

    # Row-standardize
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    W_std = W / row_sums

    # Moran's I
    numerator = float(np.dot(z, np.dot(W_std, z)))
    denominator = float(np.sum(z ** 2))
    moran_i = (n / W_sum) * (numerator / denominator) if denominator != 0 else 0.0

    # Expected value & variance under normality assumption
    E_i = -1.0 / (n - 1)
    var_i = (n ** 2 * float(np.sum(W_std ** 2))
             - n * float(np.sum(np.sum(W_std, axis=0) ** 2))
             + float(np.sum(np.sum(W_std, axis=0)) ** 2)) / ((n - 1) ** 2 * float(np.sum(W_std ** 2)))

    z_score = (moran_i - E_i) / math.sqrt(max(var_i, 1e-10))
    p_value = float(2 * (1 - stats.norm.cdf(abs(z_score))))

    if moran_i > 0 and p_value < 0.05:
        interpretation = "Clustered — similar values tend to be near each other."
    elif moran_i < 0 and p_value < 0.05:
        interpretation = "Dispersed — dissimilar values tend to be near each other."
    else:
        interpretation = "Random — no significant spatial pattern detected."

    # Local Moran for each point
    local_results = []
    for i in range(n):
        local_i = z[i] * float(np.dot(W_std[i], z)) / (denominator / n) if denominator else 0.0
        local_results.append({
            "row": pts[i],
            "localI": round(local_i, 4),
            "value": float(pts[i][req.valueCol]),
            "zScore": round(float(z[i]), 4),
            "color": "#ef4444" if local_i > 1 else "#3b82f6" if local_i < -1 else "#9ca3af",
        })

    return to_native({
        "results": {
            "moranI": round(moran_i, 4),
            "expectedI": round(E_i, 4),
            "zScore": round(z_score, 4),
            "pValue": round(p_value, 4),
            "significant": p_value < 0.05,
            "interpretation": interpretation,
            "localPoints": local_results,
            "valueCol": req.valueCol,
            "visible": True,
        }
    })


# ══════════════════════════════════════════════════════════
# Spatial OLS Regression
# ══════════════════════════════════════════════════════════

class SpatialRegressionRequest(BaseMapRequest):
    targetCol: str
    featureCols: List[str]


@router.post("/api/stats/regression")
def run_spatial_regression(req: SpatialRegressionRequest):
    import statsmodels.api as sm

    pts = [r for r in req.data if r.get("lat") and r.get("lng")]
    valid = [p for p in pts
             if isinstance(p.get(req.targetCol), (int, float))
             and all(isinstance(p.get(c), (int, float)) for c in req.featureCols)]

    if len(valid) < len(req.featureCols) + 2:
        raise HTTPException(400, "Not enough valid rows for regression.")

    y = np.array([float(p[req.targetCol]) for p in valid])
    X_raw = np.array([[float(p[c]) for c in req.featureCols] for p in valid])
    X = sm.add_constant(X_raw)

    model = sm.OLS(y, X).fit()

    coefs = []
    param_names = ["intercept"] + req.featureCols
    for name, coef, pval, std_err in zip(
        param_names, model.params, model.pvalues, model.bse
    ):
        coefs.append({
            "name": name,
            "coef": round(float(coef), 4),
            "stdErr": round(float(std_err), 4),
            "pValue": round(float(pval), 4),
            "significant": float(pval) < 0.05,
        })

    residuals = model.resid
    # Color points by residual
    r_std = float(np.std(residuals)) or 1.0
    point_results = []
    for i, p in enumerate(valid):
        r = float(residuals[i])
        norm_r = r / r_std
        color = (
            f"rgba(239,68,68,{min(abs(norm_r)/3, 1):.2f})" if norm_r > 0
            else f"rgba(59,130,246,{min(abs(norm_r)/3, 1):.2f})"
        )
        point_results.append({
            "row": p,
            "predicted": round(float(model.fittedvalues[i]), 3),
            "actual": float(p[req.targetCol]),
            "residual": round(r, 3),
            "color": color,
        })

    return to_native({
        "results": {
            "coefficients": coefs,
            "rSquared": round(float(model.rsquared), 4),
            "adjRSquared": round(float(model.rsquared_adj), 4),
            "fStatistic": round(float(model.fvalue), 3),
            "fPValue": round(float(model.f_pvalue), 4),
            "nObs": int(model.nobs),
            "points": point_results,
            "targetCol": req.targetCol,
            "featureCols": req.featureCols,
            "visible": True,
        }
    })


# ══════════════════════════════════════════════════════════
# XGBoost Location ML Score + SHAP
# ══════════════════════════════════════════════════════════

class MLScoreRequest(BaseMapRequest):
    targetCol: str
    featureCols: List[str]
    topN: int = 10   # top locations to highlight


@router.post("/api/stats/mlscore")
def run_ml_score(req: MLScoreRequest):
    try:
        import xgboost as xgb
        import shap
    except ImportError:
        raise HTTPException(500, "xgboost/shap not installed")

    pts = [r for r in req.data if r.get("lat") and r.get("lng")]
    valid = [p for p in pts
             if isinstance(p.get(req.targetCol), (int, float))
             and all(isinstance(p.get(c), (int, float)) for c in req.featureCols)]

    if len(valid) < 5:
        raise HTTPException(400, "Need at least 5 valid rows.")

    y = np.array([float(p[req.targetCol]) for p in valid])
    X = np.array([[float(p[c]) for c in req.featureCols] for p in valid])

    model = xgb.XGBRegressor(n_estimators=100, max_depth=4,
                              learning_rate=0.1, random_state=42,
                              verbosity=0)
    model.fit(X, y)
    preds = model.predict(X)

    # SHAP values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Normalize predictions to 0-100 score
    pred_min, pred_max = float(preds.min()), float(preds.max())
    pred_range = pred_max - pred_min or 1.0
    scores = [(float(p) - pred_min) / pred_range * 100 for p in preds]

    # Feature importance (mean |SHAP|)
    feature_importance = []
    for j, col in enumerate(req.featureCols):
        mean_shap = float(np.mean(np.abs(shap_values[:, j])))
        feature_importance.append({
            "feature": col,
            "importance": round(mean_shap, 4),
            "importancePct": 0.0,  # filled below
        })
    total_imp = sum(f["importance"] for f in feature_importance) or 1.0
    for f in feature_importance:
        f["importancePct"] = round(f["importance"] / total_imp * 100, 1)
    feature_importance.sort(key=lambda x: -x["importance"])

    # Per-point results
    point_results = []
    for i, p in enumerate(valid):
        score = scores[i]
        shap_row = {req.featureCols[j]: round(float(shap_values[i, j]), 4)
                    for j in range(len(req.featureCols))}
        top_drivers = sorted(shap_row.items(), key=lambda x: -abs(x[1]))[:3]

        point_results.append({
            "row": p,
            "score": round(score, 1),
            "predicted": round(float(preds[i]), 3),
            "actual": float(p[req.targetCol]),
            "shapValues": shap_row,
            "topDrivers": [{"feature": k, "impact": v} for k, v in top_drivers],
            "rank": 0,  # filled below
        })

    point_results.sort(key=lambda x: -x["score"])
    for rank, r in enumerate(point_results, 1):
        r["rank"] = rank
        score = r["score"]
        r["color"] = (
            "#22c55e" if score >= 80 else
            "#84cc16" if score >= 60 else
            "#eab308" if score >= 40 else
            "#f97316" if score >= 20 else
            "#ef4444"
        )

    return to_native({
        "results": {
            "points": point_results,
            "featureImportance": feature_importance,
            "topN": point_results[:req.topN],
            "targetCol": req.targetCol,
            "featureCols": req.featureCols,
            "visible": True,
        }
    })
