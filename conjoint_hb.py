"""
FastAPI router: Hierarchical Bayes estimation for choice-based conjoint.

Exposes POST /api/analysis/conjoint-hb. Accepts the SAME flat-row payload the
aggregate CBC endpoint (conjoint_analysis.py) uses, but estimates
individual-level part-worths via HB-MNL (hb_mnl.py — Gibbs/Metropolis,
pure numpy/scipy). Shares its estimation core with ACBC (acbc_analysis.py).
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from hb_mnl import run_hb_mnl
from price_levels import price_slope

router = APIRouter()


class ConjointHBPayload(BaseModel):
    data: List[Dict[str, Any]]
    attributeCols: List[str]
    respCol: str = "respondent_id"
    taskCol: str = "task_id"
    responseCol: str = "choice"
    priceCol: Optional[str] = None
    draws: int = 1000
    tune: int = 1000
    chains: int = 2


def _compute_wtp(partworth_rows, price_col):
    """WTP = -(part-worth / price slope); price slope from OLS of the price
    levels' part-worths on their amounts (price_levels.py).

    Takes the `charts.partworth_data` rows and returns WTP rows in the same
    shape the aggregate endpoint sends as `wtp` (conjoint_analysis.py:
    attribute, level, wtp, wtp_ci_lower, wtp_ci_upper, is_reference, label),
    because the CBC screen reads both endpoints' results with the same code
    and walks `wtp` as a list. The range is each level's 95% interval divided
    by the same slope; the slope's own uncertainty is not included.

    Returns (rows, slope), or (None, None) when the price levels give no slope.
    """
    slope = price_slope((p["level"], p["coef"]) for p in partworth_rows if p["attribute"] == price_col)
    if slope is None:
        return None, None
    rows = []
    for p in partworth_rows:
        if p["attribute"] == price_col:
            continue
        ref = bool(p.get("is_reference"))
        lo, hi = -p["ci_lower"] / slope, -p["ci_upper"] / slope
        rows.append({
            "attribute": p["attribute"], "level": p["level"],
            "wtp": 0.0 if ref else float(-p["coef"] / slope),
            "wtp_ci_lower": 0.0 if ref else float(min(lo, hi)),
            "wtp_ci_upper": 0.0 if ref else float(max(lo, hi)),
            "is_reference": ref,
            "label": f"{p['attribute']}: {p['level']}",
        })
    return rows, float(slope)


@router.post("/conjoint-hb")
async def analyze_conjoint_hb(payload: ConjointHBPayload):
    try:
        res = run_hb_mnl(
            data=payload.data,
            attribute_cols=payload.attributeCols,
            resp_col=payload.respCol,
            task_col=payload.taskCol,
            choice_col=payload.responseCol,
            draws=payload.draws,
            tune=payload.tune,
            chains=payload.chains,
        )
        res["price_col"] = None
        if payload.priceCol:
            wtp, slope = _compute_wtp(res["charts"]["partworth_data"], payload.priceCol)
            if wtp is not None:
                res["wtp"] = wtp
                res["beta_price"] = slope
                res["charts"]["wtp_data"] = wtp
                res["price_col"] = payload.priceCol
        return {"results": res}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
