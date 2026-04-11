from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import numpy as np
import traceback
import math

router = APIRouter()


# ══════════════════════════════════════════════════════════════
# Request / Response Models
# ══════════════════════════════════════════════════════════════

class GageRRRequest(BaseModel):
    data: List[Dict[str, Any]]
    colPart: Optional[str] = None
    colOperator: Optional[str] = None
    colMeasurement: Optional[str] = None
    modelType: str = "crossed"       # always crossed random ANOVA per AIAG MSA
    tolerance: Optional[float] = None
    generate: bool = False


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

def _to_native(obj):
    """Recursively convert numpy types to Python native types."""
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


# ══════════════════════════════════════════════════════════════
# F-distribution p-value  (Regularized Incomplete Beta)
# ══════════════════════════════════════════════════════════════

def _lgamma(x: float) -> float:
    c = [
        76.18009172947146, -86.50532032941677, 24.01409824083091,
        -1.231739572450155, 0.001208650973866179, -0.000005395239384953,
    ]
    y = x
    tmp = x + 5.5
    tmp -= (x + 0.5) * math.log(tmp)
    ser = 1.000000000190015
    for j in range(6):
        y += 1
        ser += c[j] / y
    return -tmp + math.log(2.5066282746310005 * ser / x)


def _regularized_beta(x: float, a: float, b: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    ln_beta = _lgamma(a) + _lgamma(b) - _lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - ln_beta)
    f_cf = 1.0
    c = 1.0
    d = 1.0 - (a + b) * x / (a + 1)
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    f_cf = d
    for m in range(1, 301):
        num = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m))
        d = 1.0 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        f_cf *= d * c
        num = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        if abs(d) < 1e-30:
            d = 1e-30
        d = 1.0 / d
        c = 1.0 + num / c
        if abs(c) < 1e-30:
            c = 1e-30
        delta = d * c
        f_cf *= delta
        if abs(delta - 1.0) < 1e-12:
            break
    return front * f_cf / a


def f_pvalue(f: float, df1: int, df2: int) -> float:
    if f <= 0 or df1 <= 0 or df2 <= 0:
        return 1.0
    x = df2 / (df2 + df1 * f)
    return _regularized_beta(x, df2 / 2.0, df1 / 2.0)


# ══════════════════════════════════════════════════════════════
# Example Data Generation  (10 parts × 3 operators × 3 trials)
# ══════════════════════════════════════════════════════════════

def generate_example_data() -> List[Dict[str, Any]]:
    rng = np.random.default_rng(42)
    part_nominals = [10.2, 10.5, 10.1, 10.8, 10.3, 10.6, 10.0, 10.9, 10.4, 10.7]
    operator_bias = [0.0, 0.03, -0.02]
    equip_sigma = 0.05
    rows = []
    for p in range(10):
        for o in range(3):
            for t in range(3):
                z = rng.standard_normal()
                value = part_nominals[p] + operator_bias[o] + equip_sigma * z
                rows.append({
                    "part": f"P{p + 1}",
                    "operator": f"Op{o + 1}",
                    "trial": t + 1,
                    "measurement": round(value, 4),
                })
    return rows


# ══════════════════════════════════════════════════════════════
# Core Computation  — Two-way ANOVA Gage R&R
# ══════════════════════════════════════════════════════════════

def compute_gage_rr(
    data: List[Dict[str, Any]],
    part_col: str,
    operator_col: str,
    measure_col: str,
    model_type: str,           # "fixed" | "mixed"
    tolerance_val: Optional[float],
) -> Dict[str, Any]:
    """
    Compute crossed Gage R&R with Two-way ANOVA.
    Mirrors the TypeScript computeGageRR() logic exactly.
    """

    # ── Parse observations ──
    obs = []
    for row in data:
        p = str(row.get(part_col, "")).strip()
        o = str(row.get(operator_col, "")).strip()
        try:
            v = float(row.get(measure_col, ""))
        except (ValueError, TypeError):
            continue
        if p and o and not math.isnan(v):
            obs.append({"part": p, "operator": o, "value": v})

    if len(obs) < 6:
        raise HTTPException(status_code=400, detail=f"Need at least 6 valid observations. Got {len(obs)}.")

    parts = sorted(set(o["part"] for o in obs))
    operators = sorted(set(o["operator"] for o in obs))
    a = len(parts)      # nParts
    b = len(operators)   # nOperators

    if a < 2 or b < 2:
        raise HTTPException(status_code=400, detail=f"Need at least 2 parts and 2 operators. Got {a} parts, {b} operators.")

    # ── Build cell structure [part][operator] → values[] ──
    cells: Dict[str, Dict[str, List[float]]] = {
        p: {o: [] for o in operators} for p in parts
    }
    for o in obs:
        if o["part"] in cells and o["operator"] in cells[o["part"]]:
            cells[o["part"]][o["operator"]].append(o["value"])

    # nTrials = minimum replications across all cells
    r = min(len(cells[p][o]) for p in parts for o in operators)
    if r < 1:
        raise HTTPException(status_code=400, detail="Not all part×operator combinations have at least 1 measurement.")

    # Trim to balanced design
    for p in parts:
        for o in operators:
            cells[p][o] = cells[p][o][:r]

    N = a * b * r

    # ── Grand mean ──
    grand_sum = sum(v for p in parts for o in operators for v in cells[p][o])
    grand_mean = grand_sum / N

    # ── Part means ──
    part_mean: Dict[str, float] = {}
    for p in parts:
        vals = [v for o in operators for v in cells[p][o]]
        part_mean[p] = sum(vals) / len(vals)

    # ── Operator means ──
    op_mean: Dict[str, float] = {}
    for o in operators:
        vals = [v for p in parts for v in cells[p][o]]
        op_mean[o] = sum(vals) / len(vals)

    # ── Cell means ──
    cell_mean: Dict[str, Dict[str, float]] = {}
    for p in parts:
        cell_mean[p] = {}
        for o in operators:
            cell_mean[p][o] = sum(cells[p][o]) / len(cells[p][o])

    # ── Sum of Squares ──
    ss_part = b * r * sum((part_mean[p] - grand_mean) ** 2 for p in parts)
    ss_op = a * r * sum((op_mean[o] - grand_mean) ** 2 for o in operators)
    ss_int = r * sum(
        (cell_mean[p][o] - part_mean[p] - op_mean[o] + grand_mean) ** 2
        for p in parts for o in operators
    )
    ss_error = sum(
        (v - cell_mean[p][o]) ** 2
        for p in parts for o in operators for v in cells[p][o]
    )
    ss_total = sum(
        (v - grand_mean) ** 2
        for p in parts for o in operators for v in cells[p][o]
    )

    # ── Degrees of Freedom ──
    df_part = a - 1
    df_op = b - 1
    df_int = df_part * df_op
    df_error = a * b * (r - 1)
    df_total = N - 1

    # ── Mean Squares ──
    ms_part = ss_part / df_part if df_part > 0 else 0.0
    ms_op = ss_op / df_op if df_op > 0 else 0.0
    ms_int = ss_int / df_int if df_int > 0 else 0.0
    ms_error = ss_error / df_error if df_error > 0 else 0.0

    # ── F-ratios & p-values (Step 1: with interaction) ──
    # For crossed random/mixed ANOVA per AIAG MSA:
    # F_part = MS_part / MS_interaction
    # F_operator = MS_operator / MS_interaction
    # F_interaction = MS_interaction / MS_error
    f_int = ms_int / ms_error if ms_error > 0 else 0.0
    p_int = f_pvalue(f_int, df_int, df_error) if df_int > 0 and df_error > 0 else None

    f_part = ms_part / ms_int if ms_int > 0 else 0.0
    f_op = ms_op / ms_int if ms_int > 0 else 0.0
    p_part = f_pvalue(f_part, df_part, df_int) if df_part > 0 and df_int > 0 else None
    p_op = f_pvalue(f_op, df_op, df_int) if df_op > 0 and df_int > 0 else None

    # ── Auto-pooling per AIAG MSA Manual ──
    # If interaction p-value > 0.25, pool interaction into repeatability
    pooling_threshold = 0.25
    interaction_pooled = False

    if p_int is not None and p_int > pooling_threshold:
        interaction_pooled = True
        # Pool: combine interaction SS and error SS
        ss_error_pooled = ss_error + ss_int
        df_error_pooled = df_error + df_int
        ms_error_pooled = ss_error_pooled / df_error_pooled if df_error_pooled > 0 else 0.0

        # Recompute F-ratios against pooled error
        f_part = ms_part / ms_error_pooled if ms_error_pooled > 0 else 0.0
        f_op = ms_op / ms_error_pooled if ms_error_pooled > 0 else 0.0
        p_part = f_pvalue(f_part, df_part, df_error_pooled) if df_part > 0 and df_error_pooled > 0 else None
        p_op = f_pvalue(f_op, df_op, df_error_pooled) if df_op > 0 and df_error_pooled > 0 else None

        # Update error terms
        ss_error = ss_error_pooled
        df_error = df_error_pooled
        ms_error = ms_error_pooled

        anova = [
            {"source": "Part", "df": df_part, "ss": ss_part, "ms": ms_part, "f": f_part, "p": p_part},
            {"source": "Operator", "df": df_op, "ss": ss_op, "ms": ms_op, "f": f_op, "p": p_op},
            {"source": "Repeatability (Equipment)", "df": df_error, "ss": ss_error, "ms": ms_error, "f": None, "p": None},
            {"source": "Total", "df": df_total, "ss": ss_total, "ms": ss_total / df_total if df_total > 0 else 0.0, "f": None, "p": None},
        ]
    else:
        anova = [
            {"source": "Part", "df": df_part, "ss": ss_part, "ms": ms_part, "f": f_part, "p": p_part},
            {"source": "Operator", "df": df_op, "ss": ss_op, "ms": ms_op, "f": f_op, "p": p_op},
            {"source": "Part × Operator", "df": df_int, "ss": ss_int, "ms": ms_int, "f": f_int, "p": p_int},
            {"source": "Repeatability (Equipment)", "df": df_error, "ss": ss_error, "ms": ms_error, "f": None, "p": None},
            {"source": "Total", "df": df_total, "ss": ss_total, "ms": ss_total / df_total if df_total > 0 else 0.0, "f": None, "p": None},
        ]

    # ── Variance Components (EMS method) ──
    var_repeatability = ms_error

    if interaction_pooled:
        # No interaction term — use pooled MS_error
        var_interaction = 0.0
        var_operator = (ms_op - ms_error) / (a * r)
        var_part = (ms_part - ms_error) / (b * r)
    else:
        var_interaction = (ms_int - ms_error) / r
        if var_interaction < 0:
            var_interaction = 0.0

        if var_interaction > 0:
            var_operator = (ms_op - ms_int) / (a * r)
        else:
            var_operator = (ms_op - ms_error) / (a * r)

        if var_interaction > 0:
            var_part = (ms_part - ms_int) / (b * r)
        else:
            var_part = (ms_part - ms_error) / (b * r)

    if var_operator < 0:
        var_operator = 0.0
    if var_part < 0:
        var_part = 0.0

    var_reproducibility = var_operator + var_interaction
    var_grr = var_repeatability + var_reproducibility
    var_total = var_grr + var_part

    if var_total <= 0:
        raise HTTPException(status_code=400, detail="Total variance is zero or negative. Check data quality.")

    # ── Variance component table ──
    sd_total = math.sqrt(var_total)

    def mk_comp(src: str, v: float) -> Dict[str, Any]:
        sd = math.sqrt(v)
        return {
            "source": src,
            "variance": safe_float(v),
            "pctContribution": safe_float((v / var_total) * 100),
            "stdDev": safe_float(sd),
            "studyVar": safe_float(6 * sd),
            "pctStudyVar": safe_float((sd / sd_total) * 100) if sd_total > 0 else 0.0,
            "pctTolerance": safe_float((6 * sd / tolerance_val) * 100) if tolerance_val and tolerance_val > 0 else None,
        }

    components = [
        mk_comp("Total Gage R&R", var_grr),
        mk_comp("  Repeatability", var_repeatability),
        mk_comp("  Reproducibility", var_reproducibility),
        mk_comp("    Operator", var_operator),
        mk_comp("    Operator × Part", var_interaction),
        mk_comp("Part-to-Part", var_part),
        mk_comp("Total Variation", var_total),
    ]

    # ── %GRR ──
    pct_grr = (math.sqrt(var_grr) / sd_total) * 100
    pct_repeatability = (math.sqrt(var_repeatability) / sd_total) * 100
    pct_reproducibility = (math.sqrt(var_reproducibility) / sd_total) * 100
    pct_part = (math.sqrt(var_part) / sd_total) * 100

    # ── ndc ──
    ndc = int(math.sqrt(2) * (math.sqrt(var_part) / math.sqrt(var_grr))) if var_grr > 0 else 0

    # ── Per-operator stats ──
    operator_stats = []
    for o in operators:
        vals = [v for p in parts for v in cells[p][o]]
        mean_val = sum(vals) / len(vals)
        sd_val = math.sqrt(sum((v - mean_val) ** 2 for v in vals) / (len(vals) - 1)) if len(vals) > 1 else 0.0
        operator_stats.append({
            "operator": o,
            "mean": safe_float(mean_val),
            "range": safe_float(max(vals) - min(vals)),
            "stdDev": safe_float(sd_val),
            "n": len(vals),
        })

    # ── Per-part stats ──
    part_stats = []
    for p in parts:
        vals = [v for o in operators for v in cells[p][o]]
        mean_val = sum(vals) / len(vals)
        part_stats.append({
            "part": p,
            "mean": safe_float(mean_val),
            "range": safe_float(max(vals) - min(vals)),
            "n": len(vals),
        })

    # ── By operator×part (for charts) ──
    by_operator_part = []
    for o in operators:
        for p in parts:
            vals = cells[p][o]
            by_operator_part.append({
                "operator": o,
                "part": p,
                "values": [safe_float(v) for v in vals],
                "mean": safe_float(cell_mean[p][o]),
            })

    return {
        "nParts": a,
        "nOperators": b,
        "nTrials": r,
        "nTotal": N,
        "modelType": model_type,
        "interactionPooled": interaction_pooled,
        "anova": anova,
        "varRepeatability": safe_float(var_repeatability),
        "varReproducibility": safe_float(var_reproducibility),
        "varOperator": safe_float(var_operator),
        "varInteraction": safe_float(var_interaction),
        "varGRR": safe_float(var_grr),
        "varPart": safe_float(var_part),
        "varTotal": safe_float(var_total),
        "components": components,
        "pctGRR": safe_float(pct_grr),
        "pctRepeatability": safe_float(pct_repeatability),
        "pctReproducibility": safe_float(pct_reproducibility),
        "pctPart": safe_float(pct_part),
        "ndc": ndc,
        "tolerance": tolerance_val,
        "operatorStats": operator_stats,
        "partStats": part_stats,
        "byOperatorPart": by_operator_part,
    }


# ══════════════════════════════════════════════════════════════
# Auto-detect column names
# ══════════════════════════════════════════════════════════════

def _detect_column(headers: List[str], keywords: List[str]) -> Optional[str]:
    """Find a column matching keywords (exact match first, then substring)."""
    lower_headers = [h.lower() for h in headers]
    # Exact match
    for kw in keywords:
        for i, lh in enumerate(lower_headers):
            if lh == kw:
                return headers[i]
    # Substring match (only for keywords > 3 chars)
    for kw in keywords:
        if len(kw) <= 3:
            continue
        for i, lh in enumerate(lower_headers):
            if kw in lh:
                return headers[i]
    return None


# ══════════════════════════════════════════════════════════════
# Main Endpoint
# ══════════════════════════════════════════════════════════════

@router.post("/gage-rr")
async def gage_rr(request: GageRRRequest):
    try:
        # ── 1. Data ──
        if request.generate or not request.data:
            data = generate_example_data()
            col_part = "part"
            col_operator = "operator"
            col_measurement = "measurement"
        else:
            data = request.data
            if not data or len(data) == 0:
                raise HTTPException(status_code=400, detail="No data provided.")

            headers = list(data[0].keys())

            # Auto-detect or use provided column names
            col_part = request.colPart or _detect_column(
                headers, ["part", "part_id", "sample", "specimen"]
            )
            col_operator = request.colOperator or _detect_column(
                headers, ["operator", "appraiser", "inspector", "technician"]
            )
            col_measurement = request.colMeasurement or _detect_column(
                headers, ["measurement", "value", "reading", "result", "dimension"]
            )

            if not col_part:
                raise HTTPException(status_code=400, detail="Cannot find part column. Provide colPart or use standard naming (part, sample, specimen).")
            if not col_operator:
                raise HTTPException(status_code=400, detail="Cannot find operator column. Provide colOperator or use standard naming (operator, appraiser, inspector).")
            if not col_measurement:
                raise HTTPException(status_code=400, detail="Cannot find measurement column. Provide colMeasurement or use standard naming (measurement, value, reading).")

        # ── 2. Tolerance ──
        tolerance_val = request.tolerance
        if tolerance_val is not None and tolerance_val <= 0:
            tolerance_val = None

        # ── 3. Model type ──
        model_type = request.modelType if request.modelType in ("fixed", "mixed") else "mixed"

        # ── 4. Compute ──
        result = compute_gage_rr(
            data=data,
            part_col=col_part,
            operator_col=col_operator,
            measure_col=col_measurement,
            model_type=model_type,
            tolerance_val=tolerance_val,
        )

        return _to_native({"results": result})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")
