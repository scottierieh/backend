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


def _compute_wtp(part_worths, price_col):
    """WTP = -(part-worth / price slope); price slope from OLS of numeric price levels."""
    import numpy as np
    price_levels = [p for p in part_worths if p["attribute"] == price_col]
    if len(price_levels) < 2:
        return None
    prices, utils = [], []
    for p in price_levels:
        try:
            prices.append(float("".join(c for c in str(p["level"]) if c.isdigit() or c in ".,")))
            utils.append(p["value"])
        except (ValueError, TypeError):
            continue
    if len(prices) < 2:
        return None
    Xp = np.vstack([np.ones(len(prices)), np.array(prices)]).T
    slope = np.linalg.lstsq(Xp, np.array(utils), rcond=None)[0][1]
    if abs(slope) < 1e-9:
        return None
    return {
        f"{p['attribute']}_{p['level']}": float(-p["value"] / slope)
        for p in part_worths if p["attribute"] != price_col
    }


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
            wtp = _compute_wtp(res["partWorths"], payload.priceCol)
            if wtp is not None:
                res["wtp"] = wtp
                res["price_col"] = payload.priceCol
        return {"results": res}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
