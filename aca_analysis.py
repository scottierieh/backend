"""
FastAPI router for Adaptive Conjoint Analysis (ACA).

ACA combines two data sources per respondent:
  1. Self-explicated priors — level desirability ratings + attribute importance.
  2. Adaptive paired comparisons — preference on partial profiles.

Estimation: pooled OLS over dummy-coded part-worths (reference level = 0 per
attribute, no intercept) using both equation sources:
  • Self-explicated row:  β(k,l) = importance_k · desirability_k,l        (direct anchor)
  • Paired row:           Σ β(left) − Σ β(right) = centered preference     (trade-off)
Both sources are scaled to a comparable range so neither dominates. Dummy-coded
betas are then centered within each attribute to give zero-sum part-worths for
display (matching the RC/CBC convention), with importance = range / Σranges.

Input:
{
  "attributes": [{"id","name","levels":[{"id","value"}]}],
  "responses": [
    {
      "respondentId": "r1",
      "selfExplicated": {
        "levels": {"a1": {"A": 8, "B": 3}, ...},        # desirability per level
        "importance": {"a1": 5, "a2": 3, ...}            # importance per attribute
      },
      "pairs": [
        {"leftLevels": {"a1":"A","a2":"X"}, "rightLevels": {"a1":"B","a2":"Y"},
         "response": 7, "scale": 9}
      ],
      "calibration": [ {"levels": {...}, "rating": 80} ]   # optional, currently informational
    }
  ],
  "priceAttributeId": "a3"   # optional
}
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np

router = APIRouter()


class ACAAnalysisPayload(BaseModel):
    attributes: List[Dict[str, Any]]
    responses: List[Dict[str, Any]]
    priceAttributeId: Optional[str] = None


# ── numpy → native helpers ──────────────────────────────────────────────────
def _to_native(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj

def _native(obj):
    if isinstance(obj, dict):
        return {k: _native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_native(v) for v in obj]
    return _to_native(obj)


# ── t-distribution helpers (no scipy) ──────────────────────────────────────
def _betacf(a, b, x, itmax=200, eps=3e-12):
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0; d = 1.0 - qab * x / qap
    if abs(d) < 1e-30: d = 1e-30
    d = 1.0 / d; h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30: d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d; h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30: d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d; delta = d * c; h *= delta
        if abs(delta - 1.0) < eps: break
    return h


def _betai(a, b, x):
    import math
    if x <= 0.0: return 0.0
    if x >= 1.0: return 1.0
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _t_two_sided_p(t, df):
    if df <= 0: return float("nan")
    t = abs(float(t)); x = df / (df + t * t)
    return float(_betai(df / 2.0, 0.5, x))


def _t_crit_95(df):
    if df <= 0: return float("nan")
    z = 1.959963985
    g1 = (z ** 3 + z) / 4.0
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96.0
    g3 = (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384.0
    return float(z + g1 / df + g2 / df ** 2 + g3 / df ** 3)


def _column_index(attributes):
    """Map (attribute_id, level_value) → dummy-coded column index.
    Reference level (first level of each attribute) is dropped (implicit 0)."""
    col = {}
    idx = 0
    ref = {}
    for attr in attributes:
        levels = [l["value"] for l in attr["levels"]]
        ref[attr["id"]] = levels[0]
        for lv in levels[1:]:
            col[(attr["id"], lv)] = idx
            idx += 1
    return col, ref, idx


def _row_for_profile(levels_map, col_index, ref, n_cols):
    """Dummy-coded design row for a (partial) profile."""
    row = np.zeros(n_cols)
    for attr_id, lv in levels_map.items():
        if lv == ref.get(attr_id):
            continue
        c = col_index.get((attr_id, lv))
        if c is not None:
            row[c] = 1.0
    return row


def _fit_dummy(X, y, n_cols):
    lam = 1e-6
    return np.linalg.solve(X.T @ X + lam * np.eye(n_cols), X.T @ y)


def _centered_pw_aca(beta, attributes, col_index, ref):
    """Dummy betas → {attrId: {value: utility}} centred, plus importance dict."""
    util_map, ranges = {}, {}
    for attr in attributes:
        aid = attr["id"]
        lv_values = [l["value"] for l in attr["levels"]]
        raw = [0.0 if lv == ref[aid] else float(beta[col_index[(aid, lv)]]) for lv in lv_values]
        m = float(np.mean(raw))
        cen = [v - m for v in raw]
        util_map[aid] = {lv_values[i]: cen[i] for i in range(len(lv_values))}
        ranges[aid] = max(cen) - min(cen)
    tot = sum(ranges.values()) or 1.0
    imp = {aid: ranges[aid] / tot * 100 for aid in ranges}
    return util_map, imp


def _bootstrap_aca(resp_blocks, attributes, col_index, ref, n_cols, B=400, seed=20240601):
    n = len(resp_blocks)
    if n < 8:
        return None
    rng = np.random.default_rng(seed)
    attr_ids = [a["id"] for a in attributes]
    names = {a["id"]: a["name"] for a in attributes}
    imp_s = {aid: [] for aid in attr_ids}
    rank1 = {aid: 0 for aid in attr_ids}
    rankdist = {aid: [] for aid in attr_ids}
    lvl_s = {}
    valid = 0
    for _ in range(B):
        idx = rng.integers(0, n, n)
        X = np.vstack([resp_blocks[i][0] for i in idx])
        y = np.concatenate([resp_blocks[i][1] for i in idx])
        try:
            beta = _fit_dummy(X, y, n_cols)
        except Exception:
            continue
        util_map, imp = _centered_pw_aca(beta, attributes, col_index, ref)
        valid += 1
        order = sorted(attr_ids, key=lambda a: -imp[a])
        rank1[order[0]] += 1
        for rank, aid in enumerate(order, start=1):
            rankdist[aid].append(rank)
        for aid in attr_ids:
            imp_s[aid].append(imp[aid])
            for val, u in util_map[aid].items():
                lvl_s.setdefault((aid, val), []).append(u)
    if valid == 0:
        return None
    importance_ci = []
    for a in attributes:
        s = imp_s[a["id"]]
        importance_ci.append({
            "attributeId": a["id"], "attribute": names[a["id"]],
            "mean": float(np.mean(s)), "lo": float(np.percentile(s, 2.5)), "hi": float(np.percentile(s, 97.5)),
            "rank1Pct": float(rank1[a["id"]] / valid * 100), "medianRank": float(np.median(rankdist[a["id"]])),
        })
    importance_ci.sort(key=lambda d: -d["mean"])
    partworth_ci = {}
    for (aid, val), s in lvl_s.items():
        partworth_ci.setdefault(aid, []).append({"value": val, "lo": float(np.percentile(s, 2.5)), "hi": float(np.percentile(s, 97.5))})
    return {"iterations": valid, "importanceCI": importance_ci, "partWorthCI": partworth_ci}


def _individual_aca(resp_blocks, attributes, col_index, ref, n_cols):
    out = []
    for X, y in resp_blocks:
        if len(y) < n_cols + 1:
            continue
        try:
            beta = _fit_dummy(X, y, n_cols)
        except Exception:
            continue
        util_map, _ = _centered_pw_aca(beta, attributes, col_index, ref)
        out.append(util_map)
    return out


def _segment_importance_aca(resp_blocks, resp_segments, attributes, col_index, ref, n_cols):
    if not any(s is not None for s in resp_segments):
        return None
    groups = {}
    for blk, seg in zip(resp_blocks, resp_segments):
        if seg is None:
            continue
        groups.setdefault(str(seg), []).append(blk)
    out = []
    for seg, blks in groups.items():
        if len(blks) < 3:
            continue
        X = np.vstack([b[0] for b in blks]); y = np.concatenate([b[1] for b in blks])
        if len(y) < n_cols + 1:
            continue
        try:
            beta = _fit_dummy(X, y, n_cols)
        except Exception:
            continue
        _, imp = _centered_pw_aca(beta, attributes, col_index, ref)
        out.append({"segment": seg, "n": len(blks),
                    "importance": {a["name"]: float(imp[a["id"]]) for a in attributes}})
    out.sort(key=lambda d: -d["n"])
    return out if out else None


def run_aca_analysis(attributes, responses, price_attribute_id=None):
    col_index, ref, n_cols = _column_index(attributes)
    if n_cols == 0:
        raise ValueError("Attributes must have at least two levels each.")

    X_rows: List[np.ndarray] = []
    y_vals: List[float] = []
    n_respondents = 0
    n_self_eq = 0
    n_pair_eq = 0
    resp_blocks = []          # per-respondent (rows, ys) for bootstrap & individual fits
    resp_segments = []        # parallel segment label per block

    for resp in responses:
        used = False
        rx_start = len(X_rows)
        se = resp.get("selfExplicated") or {}
        levels_rating = se.get("levels") or {}
        importance = se.get("importance") or {}

        # ── Self-explicated anchoring equations (dummy-coded) ──
        # For each attribute, normalise desirability to 0..1 (min..max within
        # attribute), scale by normalised importance, and anchor each non-ref
        # level's part-worth to importance·(desir_l − desir_ref).
        imp_vals = [float(v) for v in importance.values() if v is not None]
        imp_max = max(imp_vals) if imp_vals else 1.0
        for attr in attributes:
            aid = attr["id"]
            lv_values = [l["value"] for l in attr["levels"]]
            ratings = levels_rating.get(aid) or {}
            d = [float(ratings.get(lv, np.nan)) for lv in lv_values]
            if all(np.isnan(x) for x in d):
                continue
            dmin = np.nanmin(d)
            dmax = np.nanmax(d)
            span = (dmax - dmin) or 1.0
            imp_norm = (float(importance.get(aid, imp_max)) / imp_max) if imp_max else 1.0
            ref_lv = ref[aid]
            ref_idx = lv_values.index(ref_lv)
            ref_d = d[ref_idx] if not np.isnan(d[ref_idx]) else dmin
            for i, lv in enumerate(lv_values):
                if lv == ref_lv or np.isnan(d[i]):
                    continue
                c = col_index.get((aid, lv))
                if c is None:
                    continue
                target = imp_norm * ((d[i] - ref_d) / span)
                row = np.zeros(n_cols)
                row[c] = 1.0
                X_rows.append(row)
                y_vals.append(float(target))
                n_self_eq += 1
                used = True

        # ── Paired-comparison trade-off equations ──
        for pair in resp.get("pairs") or []:
            left = pair.get("leftLevels") or {}
            right = pair.get("rightLevels") or {}
            response = pair.get("response")
            scale = int(pair.get("scale", 9))
            if response is None:
                continue
            mid = (scale + 1) / 2.0
            half = (scale - 1) / 2.0 or 1.0
            # Map preference to a utility difference in the same 0..1-ish range as
            # the self-explicated anchors: (response − mid)/half ∈ [−1, 1].
            centered = (float(response) - mid) / half
            row = (_row_for_profile(left, col_index, ref, n_cols)
                   - _row_for_profile(right, col_index, ref, n_cols))
            if not np.any(row):
                continue
            X_rows.append(row)
            y_vals.append(float(centered))
            n_pair_eq += 1
            used = True

        if used:
            n_respondents += 1
            resp_blocks.append((np.array(X_rows[rx_start:]), np.array(y_vals[rx_start:])))
            resp_segments.append(resp.get("segment"))

    if len(y_vals) < n_cols:
        raise ValueError(
            f"Insufficient equations ({len(y_vals)}) for {n_cols} parameters. "
            "Collect more responses or enable more ACA phases."
        )

    X = np.array(X_rows)
    y = np.array(y_vals)
    # Ridge-stabilised least squares (tiny λ) so sparse self-explicated-only
    # designs stay solvable.
    lam = 1e-6
    XtX = X.T @ X + lam * np.eye(n_cols)
    XtXinv = np.linalg.inv(XtX)
    beta = XtXinv @ (X.T @ y)

    y_hat = X @ beta
    resid = y - y_hat
    n_obs = len(y)
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    df_resid = max(1, n_obs - n_cols)
    sigma2 = ss_res / df_resid
    cov = sigma2 * XtXinv                       # coefficient covariance

    # ── Dummy → centred part-worths per attribute (with inference) ──
    part_worths = []
    total_ranges = []
    for attr in attributes:
        aid = attr["id"]
        lv_values = [l["value"] for l in attr["levels"]]
        K = len(lv_values)
        cols = [col_index[(aid, lv)] for lv in lv_values if lv != ref[aid]]
        raw = [0.0 if lv == ref[aid] else float(beta[col_index[(aid, lv)]]) for lv in lv_values]
        mean = float(np.mean(raw))
        centred = [v - mean for v in raw]

        level_utils = []
        for i, lv in enumerate(lv_values):
            # centred_i = a^T beta, a nonzero only on this attribute's columns.
            a = np.zeros(n_cols)
            for c in cols:
                a[c] = -1.0 / K
            if lv != ref[aid]:
                a[col_index[(aid, lv)]] += 1.0
            se = float(np.sqrt(max(0.0, a @ cov @ a)))
            entry = {"value": lv, "utility": centred[i]}
            if se > 0:
                tval = centred[i] / se
                tcrit = _t_crit_95(df_resid)
                entry.update({"se": se, "t": float(tval),
                              "pValue": _t_two_sided_p(tval, df_resid),
                              "ciLow": float(centred[i] - tcrit * se),
                              "ciHigh": float(centred[i] + tcrit * se)})
            level_utils.append(entry)

        rng = max(centred) - min(centred)
        total_ranges.append(rng)
        part_worths.append({
            "attributeId": aid, "attributeName": attr["name"],
            "levels": level_utils, "range": float(rng),
        })
    total = sum(total_ranges)
    for i, pw in enumerate(part_worths):
        pw["importance"] = float(total_ranges[i] / total * 100) if total > 0 else 0.0

    # ── Fit statistics ──
    import math
    adj_r2 = 1 - (1 - r2) * (n_obs - 1) / df_resid if df_resid > 0 else r2
    rmse = math.sqrt(ss_res / n_obs) if n_obs > 0 else float("nan")
    ll = -0.5 * n_obs * (math.log(2 * math.pi) + math.log(ss_res / n_obs if ss_res > 0 else 1e-12) + 1) if n_obs > 0 else float("nan")
    fit_stats = {
        "rSquared": r2, "adjRSquared": adj_r2, "rmse": rmse,
        "aic": 2 * n_cols - 2 * ll, "bic": n_cols * math.log(n_obs) - 2 * ll if n_obs > 0 else float("nan"),
        "nObs": n_obs, "nParams": n_cols,
    }

    # ── WTP (reuse RC convention: price slope of level utilities) ──
    wtp = None
    if price_attribute_id:
        price_pw = next((p for p in part_worths if p["attributeId"] == price_attribute_id), None)
        if price_pw:
            prices, utils = [], []
            for lvl in price_pw["levels"]:
                try:
                    pval = float("".join(c for c in lvl["value"] if c.isdigit() or c in ".,"))
                    prices.append(pval)
                    utils.append(lvl["utility"])
                except (ValueError, TypeError):
                    continue
            if len(prices) >= 2:
                Xp = np.vstack([np.ones(len(prices)), np.array(prices)]).T
                slope = np.linalg.lstsq(Xp, np.array(utils), rcond=None)[0][1]
                if abs(slope) > 1e-9:
                    wtp = {}
                    for pw in part_worths:
                        if pw["attributeId"] == price_attribute_id:
                            continue
                        for lvl in pw["levels"]:
                            wtp[f"{pw['attributeName']}_{lvl['value']}"] = float(-lvl["utility"] / slope)

    bootstrap = _bootstrap_aca(resp_blocks, attributes, col_index, ref, n_cols)
    individual = _individual_aca(resp_blocks, attributes, col_index, ref, n_cols)
    segment_importance = _segment_importance_aca(resp_blocks, resp_segments, attributes, col_index, ref, n_cols)

    result = {
        "partWorths": part_worths,
        "rSquared": r2,
        "fitStats": fit_stats,
        "n": n_respondents,
        "nObs": len(y_vals),
        "nSelfExplicated": n_self_eq,
        "nPaired": n_pair_eq,
    }
    if bootstrap is not None:
        result["bootstrap"] = bootstrap
    if individual:
        result["individualPartWorths"] = individual
    if segment_importance:
        result["segmentImportance"] = segment_importance
    if wtp is not None:
        result["wtp"] = wtp
    return _native(result)


@router.post("/adaptive-conjoint")
async def analyze_adaptive_conjoint(payload: ACAAnalysisPayload):
    try:
        results = run_aca_analysis(
            attributes=payload.attributes,
            responses=payload.responses,
            price_attribute_id=payload.priceAttributeId,
        )
        return {"results": results}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
