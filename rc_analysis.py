from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np

router = APIRouter()


class RCAnalysisPayload(BaseModel):
    profiles: List[Dict[str, Any]]
    responses: List[Dict[str, Any]]
    attributes: List[Dict[str, Any]]
    priceAttributeId: Optional[str] = None


# ── helpers ──────────────────────────────────────────────────────────────────

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


def build_effects_matrix(profile_levels, attributes):
    rows = []
    for pl in profile_levels:
        row = []
        for attr in attributes:
            attr_id = attr["id"]
            levels = [l["value"] for l in attr["levels"]]
            K = len(levels)
            chosen = pl.get(attr_id, levels[-1])
            if chosen == levels[-1]:
                row.extend([-1] * (K - 1))
            else:
                idx = levels.index(chosen) if chosen in levels else 0
                coding = [0] * (K - 1)
                coding[idx] = 1
                row.extend(coding)
        rows.append(row)
    return np.array(rows, dtype=float)


# ── t-distribution p-value (no scipy) ─────────────────────────────────────────
def _betacf(a, b, x, itmax=200, eps=3e-12):
    """Continued fraction for the incomplete beta function (Numerical Recipes)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betai(a, b, x):
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    import math
    lbeta = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    bt = math.exp(lbeta + a * math.log(x) + b * math.log(1.0 - x))
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _t_two_sided_p(t, df):
    """Two-sided p-value for a t-statistic with df degrees of freedom."""
    if df <= 0:
        return float("nan")
    t = abs(float(t))
    x = df / (df + t * t)
    return float(_betai(df / 2.0, 0.5, x))


def _t_crit_95(df):
    """Approximate two-sided 95% t critical value via Cornish-Fisher-style fit."""
    if df <= 0:
        return float("nan")
    # Accurate small-sample expansion around z=1.959964 (Normal 97.5%).
    z = 1.959963985
    g1 = (z ** 3 + z) / 4.0
    g2 = (5 * z ** 5 + 16 * z ** 3 + 3 * z) / 96.0
    g3 = (3 * z ** 7 + 19 * z ** 5 + 17 * z ** 3 - 15 * z) / 384.0
    return float(z + g1 / df + g2 / df ** 2 + g3 / df ** 3)


def solve_ols(X, y):
    """β = (X'X)⁻¹X'y with intercept. Returns beta, r2, and inference inputs."""
    n = len(y)
    Xb = np.hstack([np.ones((n, 1)), X])
    p = Xb.shape[1]
    XtX = Xb.T @ Xb
    try:
        XtXinv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtXinv = np.linalg.pinv(XtX)
    beta = XtXinv @ Xb.T @ y
    y_hat = Xb @ beta
    residuals = y - y_hat
    ss_res = float(residuals @ residuals)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    df_resid = max(1, n - p)
    sigma2 = ss_res / df_resid                       # unbiased residual variance
    cov = sigma2 * XtXinv                             # coefficient covariance matrix
    return beta, r2, {
        "cov": cov, "ss_res": ss_res, "ss_tot": ss_tot,
        "n": n, "p": p, "df_resid": df_resid, "sigma2": sigma2,
    }


def compute_fit_stats(info, r2):
    """Adjusted R², RMSE, log-likelihood, AIC/AICc/BIC, and the overall F-test."""
    import math
    n, p = info["n"], info["p"]
    ss_res, ss_tot = info["ss_res"], info["ss_tot"]
    df_resid = info["df_resid"]
    adj_r2 = 1 - (1 - r2) * (n - 1) / df_resid if df_resid > 0 else r2
    rmse = math.sqrt(ss_res / n) if n > 0 else float("nan")
    # Gaussian log-likelihood at the MLE.
    ll = -0.5 * n * (math.log(2 * math.pi) + math.log(ss_res / n if ss_res > 0 else 1e-12) + 1) if n > 0 else float("nan")
    aic = 2 * p - 2 * ll
    aicc = aic + (2 * p * (p + 1)) / (n - p - 1) if (n - p - 1) > 0 else float("nan")
    bic = p * math.log(n) - 2 * ll if n > 0 else float("nan")
    # Overall F-test (model vs intercept-only).
    df_model = max(1, p - 1)
    ss_reg = ss_tot - ss_res
    f_stat = (ss_reg / df_model) / (ss_res / df_resid) if ss_res > 0 and df_resid > 0 else float("nan")
    f_x = df_resid / (df_resid + df_model * f_stat) if not math.isnan(f_stat) else 1.0
    f_p = float(_betai(df_resid / 2.0, df_model / 2.0, f_x)) if not math.isnan(f_stat) else float("nan")
    return {
        "adjRSquared": adj_r2, "rmse": rmse, "logLikelihood": ll,
        "aic": aic, "aicc": aicc, "bic": bic,
        "fStat": f_stat, "fPValue": f_p,
        "dfModel": df_model, "dfResid": df_resid,
    }


def derive_part_worths(beta, attributes, info=None):
    """
    Convert effects-coded betas to part-worth utilities, attaching per-level
    inference (SE, t, p, 95% CI) when the OLS covariance matrix is provided.
    The reference level utility is -Σβ for the attribute; its variance is
    a'Σa with a = vector of -1s over that attribute's columns (sum of the
    covariance block).
    """
    cov = info["cov"] if info else None
    df_resid = info["df_resid"] if info else None
    t_crit = _t_crit_95(df_resid) if df_resid else None

    col_idx = 1
    attr_pws = []
    total_ranges = []
    for attr in attributes:
        levels = [l["value"] for l in attr["levels"]]
        K = len(levels)
        cols = list(range(col_idx, col_idx + K - 1))
        attr_betas = beta[col_idx: col_idx + K - 1]
        col_idx += K - 1

        utilities = list(attr_betas)
        utilities.append(-float(np.sum(attr_betas)))

        # Standard error per level.
        ses = [None] * K
        if cov is not None:
            for j, c in enumerate(cols):
                ses[j] = float(np.sqrt(max(0.0, cov[c, c])))
            # Reference level: var = sum of the attribute's covariance block.
            if cols:
                block = cov[np.ix_(cols, cols)]
                ses[K - 1] = float(np.sqrt(max(0.0, block.sum())))

        level_utils = []
        for i in range(K):
            entry = {"value": levels[i], "utility": float(utilities[i])}
            se = ses[i]
            if se is not None and se > 0 and df_resid:
                tval = utilities[i] / se
                entry["se"] = se
                entry["t"] = float(tval)
                entry["pValue"] = _t_two_sided_p(tval, df_resid)
                if t_crit is not None:
                    entry["ciLow"] = float(utilities[i] - t_crit * se)
                    entry["ciHigh"] = float(utilities[i] + t_crit * se)
            level_utils.append(entry)

        rng = max(u["utility"] for u in level_utils) - min(u["utility"] for u in level_utils)
        total_ranges.append(rng)
        attr_pws.append({
            "attributeId": attr["id"],
            "attributeName": attr["name"],
            "levels": level_utils,
            "range": float(rng),
        })
    total = sum(total_ranges)
    for i, pw in enumerate(attr_pws):
        pw["importance"] = float(total_ranges[i] / total * 100) if total > 0 else 0.0
    return attr_pws


def compute_wtp(part_worths, price_attribute_id):
    price_pw = next((p for p in part_worths if p["attributeId"] == price_attribute_id), None)
    if not price_pw:
        return None
    prices, utilities = [], []
    for lvl in price_pw["levels"]:
        try:
            pval = float("".join(c for c in lvl["value"] if c.isdigit() or c in ".,"))
            prices.append(pval)
            utilities.append(lvl["utility"])
        except (ValueError, TypeError):
            continue
    if len(prices) < 2:
        return None
    Xp = np.vstack([np.ones(len(prices)), np.array(prices)]).T
    try:
        slope = np.linalg.lstsq(Xp, np.array(utilities), rcond=None)[0][1]
    except Exception:
        return None
    if abs(slope) < 1e-9:
        return None
    wtp_map = {}
    for pw in part_worths:
        if pw["attributeId"] == price_attribute_id:
            continue
        for lvl in pw["levels"]:
            wtp_map[f"{pw['attributeName']}_{lvl['value']}"] = float(-lvl["utility"] / slope)
    return wtp_map


def validate_holdout(holdout_profiles, holdout_obs, part_worths, intercept):
    if not holdout_profiles or not holdout_obs:
        return None
    pw_map = {pw["attributeId"]: {l["value"]: l["utility"] for l in pw["levels"]} for pw in part_worths}
    profile_map = {p["id"]: p for p in holdout_profiles}
    predicted, actual = [], []
    for obs in holdout_obs:
        pid, rating = obs.get("profileId"), obs.get("rating")
        if pid not in profile_map or rating is None:
            continue
        pred = intercept
        for attr_id, lvl_map in pw_map.items():
            chosen = profile_map[pid]["levels"].get(attr_id)
            if chosen and chosen in lvl_map:
                pred += lvl_map[chosen]
        predicted.append(pred)
        actual.append(float(rating))
    if len(actual) < 2:
        return None
    actual_np, pred_np = np.array(actual), np.array(predicted)
    residuals = actual_np - pred_np
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((actual_np - actual_np.mean()) ** 2))
    return {
        "mse": float(np.mean(residuals ** 2)),
        "mae": float(np.mean(np.abs(residuals))),
        "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0,
    }


def _centered_pw(beta, attributes):
    """Effects-coded beta → {attrId: {value: utility}} and importance dict."""
    col_idx = 1
    util_map, ranges = {}, {}
    for attr in attributes:
        levels = [l["value"] for l in attr["levels"]]
        K = len(levels)
        ab = beta[col_idx: col_idx + K - 1]
        col_idx += K - 1
        u = list(ab) + [-float(np.sum(ab))]
        util_map[attr["id"]] = {levels[i]: float(u[i]) for i in range(K)}
        ranges[attr["id"]] = max(u) - min(u)
    tot = sum(ranges.values()) or 1.0
    imp = {aid: ranges[aid] / tot * 100 for aid in ranges}
    return util_map, imp


def _bootstrap_rc(resp_blocks, attributes, B=500, seed=20240601):
    """Respondent-level bootstrap → importance CIs, rank-1 stability, part-worth CIs."""
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
            beta, _, _ = solve_ols(X, y)
        except Exception:
            continue
        util_map, imp = _centered_pw(beta, attributes)
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
            "rank1Pct": float(rank1[a["id"]] / valid * 100),
            "medianRank": float(np.median(rankdist[a["id"]])),
        })
    importance_ci.sort(key=lambda d: -d["mean"])
    partworth_ci = {}
    for (aid, val), s in lvl_s.items():
        partworth_ci.setdefault(aid, []).append({
            "value": val, "lo": float(np.percentile(s, 2.5)), "hi": float(np.percentile(s, 97.5)),
        })
    return {"iterations": valid, "importanceCI": importance_ci, "partWorthCI": partworth_ci}


def _individual_rc(resp_blocks, attributes):
    """Per-respondent OLS → centred part-worths, for distribution/heterogeneity views."""
    n_params = sum(len(a["levels"]) - 1 for a in attributes) + 1
    out = []
    for X, y in resp_blocks:
        if len(y) < n_params + 1 or np.unique(y).size < 2:
            continue
        try:
            beta, _, _ = solve_ols(X, y)
        except Exception:
            continue
        util_map, _ = _centered_pw(beta, attributes)
        out.append(util_map)
    return out


def _segment_importance_rc(resp_blocks, resp_segments, attributes):
    """Per segment-value OLS → importance per attribute, for the segment heatmap."""
    if not any(s is not None for s in resp_segments):
        return None
    groups = {}
    for blk, seg in zip(resp_blocks, resp_segments):
        if seg is None:
            continue
        groups.setdefault(str(seg), []).append(blk)
    n_params = sum(len(a["levels"]) - 1 for a in attributes) + 1
    out = []
    for seg, blks in groups.items():
        if len(blks) < 3:
            continue
        X = np.vstack([b[0] for b in blks]); y = np.concatenate([b[1] for b in blks])
        if len(y) < n_params + 1:
            continue
        try:
            beta, _, _ = solve_ols(X, y)
        except Exception:
            continue
        _, imp = _centered_pw(beta, attributes)
        out.append({
            "segment": seg, "n": len(blks),
            "importance": {a["name"]: float(imp[a["id"]]) for a in attributes},
        })
    out.sort(key=lambda d: -d["n"])
    return out if out else None


def run_rc_analysis(profiles, responses, attributes, price_attribute_id=None):
    estimation_profiles = [p for p in profiles if not p.get("isHoldout", False)]
    holdout_profiles = [p for p in profiles if p.get("isHoldout", False)]
    holdout_ids = {p["id"] for p in holdout_profiles}

    X_profile = build_effects_matrix([p["levels"] for p in estimation_profiles], attributes)
    profile_id_to_row = {p["id"]: i for i, p in enumerate(estimation_profiles)}

    X_rows, y_vals = [], []
    resp_blocks = []          # per-respondent (X, y) for bootstrap & individual fits
    resp_segments = []        # parallel segment label per block (or None)
    n_respondents = 0
    for resp in responses:
        ratings = resp.get("ratings", {})
        if not any(pid in ratings and ratings[pid] is not None for pid in profile_id_to_row):
            continue
        n_respondents += 1
        rx, ry = [], []
        for pid, row_idx in profile_id_to_row.items():
            rating = ratings.get(pid)
            if rating is None:
                continue
            X_rows.append(X_profile[row_idx]); y_vals.append(float(rating))
            rx.append(X_profile[row_idx]); ry.append(float(rating))
        if rx:
            resp_blocks.append((np.array(rx), np.array(ry)))
            resp_segments.append(resp.get("segment"))

    if len(y_vals) < len(attributes) + 2:
        raise ValueError(f"Insufficient observations ({len(y_vals)}) for {len(attributes)} attributes.")

    beta, r2, info = solve_ols(np.array(X_rows), np.array(y_vals))
    intercept = float(beta[0])
    part_worths = derive_part_worths(beta, attributes, info)
    fit_stats = compute_fit_stats(info, r2)

    # Bootstrap (respondent-level) → importance CIs + rank stability + part-worth CIs.
    bootstrap = _bootstrap_rc(resp_blocks, attributes)
    # Individual-level part-worths (per-respondent OLS) for distribution views.
    individual = _individual_rc(resp_blocks, attributes)
    # Per-segment importance (when a segment variable was supplied).
    segment_importance = _segment_importance_rc(resp_blocks, resp_segments, attributes)

    # Intercept inference.
    intercept_se = float(np.sqrt(max(0.0, info["cov"][0, 0])))
    intercept_stats = {"value": intercept, "se": intercept_se}
    if intercept_se > 0:
        it = intercept / intercept_se
        intercept_stats["t"] = float(it)
        intercept_stats["pValue"] = _t_two_sided_p(it, info["df_resid"])

    wtp = compute_wtp(part_worths, price_attribute_id) if price_attribute_id else None

    holdout_validation = None
    if holdout_profiles:
        holdout_obs = []
        for resp in responses:
            ratings = resp.get("ratings", {})
            for pid in holdout_ids:
                if pid in ratings and ratings[pid] is not None:
                    holdout_obs.append({"profileId": pid, "rating": float(ratings[pid])})
        holdout_validation = validate_holdout(holdout_profiles, holdout_obs, part_worths, intercept)

    result = {
        "partWorths": part_worths,
        "intercept": intercept,
        "interceptStats": intercept_stats,
        "rSquared": r2,
        "fitStats": fit_stats,
        "n": n_respondents,
        "nObs": len(y_vals),
    }
    if bootstrap is not None:
        result["bootstrap"] = bootstrap
    if individual:
        result["individualPartWorths"] = individual
    if segment_importance:
        result["segmentImportance"] = segment_importance
    if wtp is not None:
        result["wtp"] = wtp
    if holdout_validation is not None:
        result["holdoutValidation"] = holdout_validation
    return _native(result)


# ── endpoint ───────────────────────────────────────────────────────────────

@router.post("/rating-conjoint")
async def analyze_rating_conjoint(payload: RCAnalysisPayload):
    try:
        results = run_rc_analysis(
            profiles=payload.profiles,
            responses=payload.responses,
            attributes=payload.attributes,
            price_attribute_id=payload.priceAttributeId,
        )
        return {"results": results}
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
