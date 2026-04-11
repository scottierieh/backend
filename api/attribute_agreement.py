from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
import math
import traceback

router = APIRouter()


# ══════════════════════════════════════════════════════════════
# Request Model
# ══════════════════════════════════════════════════════════════

class AttributeAgreementRequest(BaseModel):
    data: List[Dict[str, Any]]
    colSample: Optional[str] = None
    colAppraiser: Optional[str] = None
    colResult: Optional[str] = None
    colReference: Optional[str] = None
    generate: bool = False


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

def _to_native(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    elif isinstance(obj, np.ndarray):
        return [_to_native(x) for x in obj.tolist()]
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_native(x) for x in obj]
    return obj


def safe_float(val, default=0.0):
    try:
        if val is None:
            return default
        f = float(val)
        return default if (math.isnan(f) or math.isinf(f)) else f
    except Exception:
        return default


def kappa_label(k: float) -> str:
    if k < 0:
        return "Poor"
    if k < 0.20:
        return "Slight"
    if k < 0.40:
        return "Fair"
    if k < 0.60:
        return "Moderate"
    if k < 0.80:
        return "Substantial"
    return "Almost Perfect"


# ══════════════════════════════════════════════════════════════
# Fleiss' Kappa
# ══════════════════════════════════════════════════════════════

def fleiss_kappa(rating_matrix: np.ndarray) -> Dict[str, Any]:
    n, k = rating_matrix.shape
    N = int(rating_matrix[0].sum())
    if N < 2 or n < 2:
        return {"kappa": 0.0, "pe": 0.0, "po": 0.0, "label": "N/A", "se": 0.0, "ci_lower": 0.0, "ci_upper": 0.0}

    p_j = rating_matrix.sum(axis=0) / (n * N)
    P_i = (np.sum(rating_matrix ** 2, axis=1) - N) / (N * (N - 1))
    P_bar = float(np.mean(P_i))
    P_e = float(np.sum(p_j ** 2))

    if abs(1 - P_e) < 1e-15:
        kappa = 1.0 if P_bar >= P_e else 0.0
    else:
        kappa = (P_bar - P_e) / (1 - P_e)

    se = 0.0
    if abs(1 - P_e) > 1e-15:
        num = 2 * (P_e - sum(float(pj) ** 2 * (1 - float(pj)) for pj in p_j))
        den = n * N * (N - 1) * (1 - P_e) ** 2
        if den > 0:
            se = math.sqrt(abs(num / den))

    ci_lower = kappa - 1.96 * se
    ci_upper = kappa + 1.96 * se

    return {
        "kappa": safe_float(kappa),
        "po": safe_float(P_bar),
        "pe": safe_float(P_e),
        "label": kappa_label(kappa),
        "se": safe_float(se),
        "ci_lower": safe_float(ci_lower),
        "ci_upper": safe_float(ci_upper),
    }


# ══════════════════════════════════════════════════════════════
# Cohen's Kappa (pairwise)
# ══════════════════════════════════════════════════════════════

def cohens_kappa(ratings_a: list, ratings_b: list, categories: list) -> Dict[str, Any]:
    n = len(ratings_a)
    if n == 0:
        return {"kappa": 0.0, "po": 0.0, "pe": 0.0, "label": "N/A"}

    k = len(categories)
    cat_idx = {c: i for i, c in enumerate(categories)}

    matrix = np.zeros((k, k), dtype=int)
    for a, b in zip(ratings_a, ratings_b):
        if a in cat_idx and b in cat_idx:
            matrix[cat_idx[a], cat_idx[b]] += 1

    po = float(np.trace(matrix)) / n
    row_sums = matrix.sum(axis=1) / n
    col_sums = matrix.sum(axis=0) / n
    pe = float(np.sum(row_sums * col_sums))

    if abs(1 - pe) < 1e-15:
        kappa = 1.0 if po >= pe else 0.0
    else:
        kappa = (po - pe) / (1 - pe)

    return {
        "kappa": safe_float(kappa),
        "po": safe_float(po),
        "pe": safe_float(pe),
        "label": kappa_label(kappa),
    }


# ══════════════════════════════════════════════════════════════
# Example Data
# ══════════════════════════════════════════════════════════════

def generate_example_data() -> List[Dict[str, Any]]:
    rng = np.random.default_rng(42)
    rows = []
    n_samples, n_appraisers, n_trials = 30, 3, 2
    reference = ["Pass" if rng.random() > 0.3 else "Fail" for _ in range(n_samples)]
    names = ["Alice", "Bob", "Charlie"]
    accuracy = [0.95, 0.88, 0.92]

    for s in range(n_samples):
        for a in range(n_appraisers):
            for t in range(n_trials):
                if rng.random() < accuracy[a]:
                    result = reference[s]
                else:
                    result = "Fail" if reference[s] == "Pass" else "Pass"
                rows.append({
                    "sample": s + 1, "appraiser": names[a], "trial": t + 1,
                    "result": result, "reference": reference[s],
                })
    return rows


# ══════════════════════════════════════════════════════════════
# Column Auto-Detection
# ══════════════════════════════════════════════════════════════

def _detect_column(headers: List[str], keywords: List[str]) -> Optional[str]:
    lower_headers = [h.lower() for h in headers]
    for kw in keywords:
        for i, lh in enumerate(lower_headers):
            if lh == kw:
                return headers[i]
    for kw in keywords:
        if len(kw) <= 3:
            continue
        for i, lh in enumerate(lower_headers):
            if kw in lh:
                return headers[i]
    return None


# ══════════════════════════════════════════════════════════════
# Main Computation
# ══════════════════════════════════════════════════════════════

def compute_attribute_agreement(
    data: List[Dict[str, Any]],
    sample_col: str, appraiser_col: str,
    result_col: str, reference_col: Optional[str],
) -> Dict[str, Any]:

    obs = []
    for row in data:
        s = str(row.get(sample_col, "")).strip()
        a = str(row.get(appraiser_col, "")).strip()
        r = str(row.get(result_col, "")).strip()
        ref = str(row.get(reference_col, "")).strip() if reference_col else None
        if s and a and r:
            obs.append({"sample": s, "appraiser": a, "result": r, "reference": ref})

    if len(obs) < 4:
        raise HTTPException(status_code=400, detail=f"Need at least 4 observations. Got {len(obs)}.")

    samples = sorted(set(o["sample"] for o in obs), key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else 0, x))
    appraisers = sorted(set(o["appraiser"] for o in obs))
    categories = sorted(set(o["result"] for o in obs))

    n_samples = len(samples)
    n_appraisers = len(appraisers)
    n_categories = len(categories)

    if n_appraisers < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 appraisers.")
    if n_categories < 2:
        raise HTTPException(status_code=400, detail="Need at least 2 result categories.")

    cell_data: Dict[str, Dict[str, List[str]]] = {s: {a: [] for a in appraisers} for s in samples}
    for o in obs:
        cell_data[o["sample"]][o["appraiser"]].append(o["result"])

    trial_counts = [len(cell_data[s][a]) for s in samples for a in appraisers if len(cell_data[s][a]) > 0]
    n_trials = min(trial_counts) if trial_counts else 1

    # 1. Within-Appraiser Repeatability
    appraiser_repeatability = []
    for a in appraisers:
        agree_count, total_count = 0, 0
        for s in samples:
            trials = cell_data[s][a]
            if len(trials) >= 2:
                if len(set(trials)) == 1:
                    agree_count += 1
                total_count += 1
        pct = (agree_count / total_count * 100) if total_count > 0 else 0
        appraiser_repeatability.append({"appraiser": a, "matched": agree_count, "total": total_count, "pct": safe_float(pct)})

    overall_repeatability = sum(r["matched"] for r in appraiser_repeatability)
    overall_repeat_total = sum(r["total"] for r in appraiser_repeatability)
    overall_repeat_pct = (overall_repeatability / overall_repeat_total * 100) if overall_repeat_total > 0 else 0

    # 2. Appraiser vs Reference
    reference_map = {}
    has_reference = False
    if reference_col:
        for o in obs:
            if o["reference"] and o["reference"] not in ("", "None", "nan"):
                reference_map[o["sample"]] = o["reference"]
                has_reference = True

    appraiser_vs_reference = []
    if has_reference:
        for a in appraisers:
            correct, total = 0, 0
            for s in samples:
                if s not in reference_map:
                    continue
                ref = reference_map[s]
                for t in cell_data[s][a]:
                    if t == ref:
                        correct += 1
                    total += 1
            pct = (correct / total * 100) if total > 0 else 0
            appraiser_vs_reference.append({"appraiser": a, "correct": correct, "total": total, "pct": safe_float(pct)})

    # 3. Between-Appraiser Agreement
    between_agree, between_total = 0, 0
    for s in samples:
        first_trials = [cell_data[s][a][0] for a in appraisers if cell_data[s][a]]
        if len(first_trials) == n_appraisers:
            between_total += 1
            if len(set(first_trials)) == 1:
                between_agree += 1
    between_pct = (between_agree / between_total * 100) if between_total > 0 else 0

    between_vs_ref_agree, between_vs_ref_total = 0, 0
    if has_reference:
        for s in samples:
            if s not in reference_map:
                continue
            ref = reference_map[s]
            first_trials = [cell_data[s][a][0] for a in appraisers if cell_data[s][a]]
            if len(first_trials) == n_appraisers:
                between_vs_ref_total += 1
                if len(set(first_trials)) == 1 and first_trials[0] == ref:
                    between_vs_ref_agree += 1
    between_vs_ref_pct = (between_vs_ref_agree / between_vs_ref_total * 100) if between_vs_ref_total > 0 else 0

    # 4. Fleiss' Kappa
    cat_idx = {c: i for i, c in enumerate(categories)}
    rating_matrix = np.zeros((n_samples, n_categories), dtype=int)
    for si, s in enumerate(samples):
        for a in appraisers:
            if cell_data[s][a]:
                r = cell_data[s][a][0]
                if r in cat_idx:
                    rating_matrix[si, cat_idx[r]] += 1
    fleiss = fleiss_kappa(rating_matrix)

    # 5. Pairwise Cohen's Kappa
    pairwise_kappa = []
    for i in range(n_appraisers):
        for j in range(i + 1, n_appraisers):
            a1, a2 = appraisers[i], appraisers[j]
            r1 = [cell_data[s][a1][0] for s in samples if cell_data[s][a1] and cell_data[s][a2]]
            r2 = [cell_data[s][a2][0] for s in samples if cell_data[s][a1] and cell_data[s][a2]]
            ck = cohens_kappa(r1, r2, categories)
            pairwise_kappa.append({"appraiser1": a1, "appraiser2": a2, **ck})

    # 6. Confusion Matrices
    confusion_matrices = []
    if has_reference:
        for a in appraisers:
            matrix = {tc: {pc: 0 for pc in categories} for tc in categories}
            for s in samples:
                if s not in reference_map:
                    continue
                ref = reference_map[s]
                for tr in cell_data[s][a]:
                    if ref in matrix and tr in matrix[ref]:
                        matrix[ref][tr] += 1
            rows = []
            for tc in categories:
                row = {"reference": tc}
                for pc in categories:
                    row[pc] = matrix[tc][pc]
                rows.append(row)
            confusion_matrices.append({"appraiser": a, "matrix": rows})

    # 7. Per-category Kappa
    per_category = []
    for ci, cat in enumerate(categories):
        binary = np.zeros((n_samples, 2), dtype=int)
        for si in range(n_samples):
            binary[si, 0] = rating_matrix[si, ci]
            binary[si, 1] = n_appraisers - rating_matrix[si, ci]
        fk = fleiss_kappa(binary)
        per_category.append({"category": cat, "kappa": fk["kappa"], "label": fk["label"]})

    return {
        "nSamples": n_samples, "nAppraisers": n_appraisers, "nTrials": n_trials,
        "nTotal": len(obs), "categories": categories, "appraisers": appraisers,
        "fleissKappa": fleiss, "pairwiseKappa": pairwise_kappa, "perCategoryKappa": per_category,
        "appraiserRepeatability": appraiser_repeatability,
        "overallRepeatability": {"matched": overall_repeatability, "total": overall_repeat_total, "pct": safe_float(overall_repeat_pct)},
        "betweenAppraiserAgreement": {"matched": between_agree, "total": between_total, "pct": safe_float(between_pct)},
        "hasReference": has_reference, "appraiserVsReference": appraiser_vs_reference,
        "betweenVsReference": {"matched": between_vs_ref_agree, "total": between_vs_ref_total, "pct": safe_float(between_vs_ref_pct)},
        "confusionMatrices": confusion_matrices,
    }


# ══════════════════════════════════════════════════════════════
# Main Endpoint
# ══════════════════════════════════════════════════════════════

@router.post("/attribute-agreement")
async def attribute_agreement(request: AttributeAgreementRequest):
    try:
        if request.generate or not request.data:
            data = generate_example_data()
            col_sample, col_appraiser, col_result, col_reference = "sample", "appraiser", "result", "reference"
        else:
            data = request.data
            if not data:
                raise HTTPException(status_code=400, detail="No data provided.")
            headers = list(data[0].keys())
            col_sample = request.colSample or _detect_column(headers, ["sample", "part", "item", "unit", "specimen", "sample_id"])
            col_appraiser = request.colAppraiser or _detect_column(headers, ["appraiser", "operator", "inspector", "rater", "judge", "technician"])
            col_result = request.colResult or _detect_column(headers, ["result", "rating", "decision", "judgment", "classification", "grade", "response", "pass_fail"])
            col_reference = request.colReference or _detect_column(headers, ["reference", "standard", "known", "true_value", "actual", "correct", "master"])

            if not col_sample:
                raise HTTPException(status_code=400, detail="Cannot find sample column.")
            if not col_appraiser:
                raise HTTPException(status_code=400, detail="Cannot find appraiser column.")
            if not col_result:
                raise HTTPException(status_code=400, detail="Cannot find result column.")

        result = compute_attribute_agreement(data, col_sample, col_appraiser, col_result, col_reference)
        return _to_native({"results": result})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")
