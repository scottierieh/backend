"""
FastAPI router for Adaptive Choice-Based Conjoint (ACBC).

ACBC interviews each respondent in three stages:
  1. BYO (Build Your Own) — pick the preferred level on each attribute.
  2. Screening — near-neighbour concepts marked a "possibility" or not;
     must-have / unacceptable rules emerge from the pattern.
  3. Choice Tournament — surviving concepts compete; winners reveal trade-offs.

Estimation: the tournament choices (clean multinomial) are fed to the shared
HB-MNL core (hb_mnl.py) for individual-level part-worths — the same engine
CBC's HB option uses. Screening drives must-have / unacceptable rule
detection (reported separately), and BYO gives each respondent's ideal.

Input:
{
  "attributes": [{"id","name","levels":[{"id","value"}]}],
  "responses": [
    {
      "respondentId": "r1",
      "byo": {"levels": {"a1":"A","a2":"20"}},
      "screening": [{"concept": {"a1":"A","a2":"30"}, "possibility": true}, ...],
      "tournament": [{"concepts": [{"a1":..},{..},{..}], "winnerIndex": 1}, ...]
    }
  ],
  "priceAttributeId": "a3",
  "draws": 800, "tune": 800, "chains": 2
}
"""

import math
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from hb_mnl import run_hb_mnl

router = APIRouter()


class ACBCPayload(BaseModel):
    attributes: List[Dict[str, Any]]
    responses: List[Dict[str, Any]]
    priceAttributeId: Optional[str] = None
    draws: int = 800
    tune: int = 800
    chains: int = 2


def _sanitize(obj):
    """Recursively replace NaN/Inf floats with None for JSON compliance."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    return obj


def _tournament_to_rows(attributes, responses):
    """Flatten tournament choice tasks into hb_mnl's flat-row format."""
    attr_ids = [a["id"] for a in attributes]
    id_to_name = {a["id"]: a["name"] for a in attributes}
    rows = []
    for ri, resp in enumerate(responses):
        rid = resp.get("respondentId", f"R{ri}")
        for ti, task in enumerate(resp.get("tournament") or []):
            concepts = task.get("concepts") or []
            win = task.get("winnerIndex")
            if not concepts or win is None or win >= len(concepts):
                continue
            for ci, concept in enumerate(concepts):
                row = {"respondent_id": rid, "task_id": f"{rid}_T{ti}",
                       "choice": 1 if ci == win else 0}
                for aid in attr_ids:
                    row[id_to_name[aid]] = concept.get(aid, "")
                rows.append(row)
    return rows, [id_to_name[a] for a in attr_ids]


def _detect_rules(attributes, responses):
    """Must-have / unacceptable detection from screening possibilities.

    - unacceptable: a level that, whenever it appeared in a screened concept,
      was almost always rejected (low possibility rate).
    - must-have: a level whose presence almost always coincides with acceptance.
    Reported as rates with the supporting counts so the UI can rank them.
    """
    # counts[(attr,level)] = [appearances, possibility_count]
    counts: Dict[tuple, List[int]] = {}
    for resp in responses:
        for s in resp.get("screening") or []:
            concept = s.get("concept") or {}
            poss = 1 if s.get("possibility") else 0
            for aid, lv in concept.items():
                key = (aid, lv)
                if key not in counts:
                    counts[key] = [0, 0]
                counts[key][0] += 1
                counts[key][1] += poss

    id_to_name = {a["id"]: a["name"] for a in attributes}
    unacceptable, must_have = [], []
    for (aid, lv), (n, pos) in counts.items():
        if n < 5:
            continue
        rate = pos / n
        rec = {"attributeId": aid, "attribute": id_to_name.get(aid, aid),
               "level": lv, "appearances": n, "possibilityRate": round(rate, 3)}
        if rate <= 0.15:
            unacceptable.append(rec)
        elif rate >= 0.85:
            must_have.append(rec)
    unacceptable.sort(key=lambda d: d["possibilityRate"])
    must_have.sort(key=lambda d: -d["possibilityRate"])
    return must_have, unacceptable


def _byo_summary(attributes, responses):
    """Most-frequent BYO level per attribute (the modal ideal configuration)."""
    from collections import Counter
    id_to_name = {a["id"]: a["name"] for a in attributes}
    per_attr: Dict[str, Counter] = {a["id"]: Counter() for a in attributes}
    n = 0
    for resp in responses:
        byo = (resp.get("byo") or {}).get("levels") or {}
        if not byo:
            continue
        n += 1
        for aid, lv in byo.items():
            if aid in per_attr:
                per_attr[aid][lv] += 1
    summary = []
    for a in attributes:
        c = per_attr[a["id"]]
        if not c:
            continue
        top, cnt = c.most_common(1)[0]
        summary.append({"attributeId": a["id"], "attribute": id_to_name[a["id"]],
                        "topLevel": top, "share": round(cnt / n, 3) if n else 0.0})
    return summary, n


def _wtp(part_worths_flat, price_name):
    import numpy as np
    price = [p for p in part_worths_flat if p["attribute"] == price_name]
    if len(price) < 2:
        return None
    prices, utils = [], []
    for p in price:
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
    return {f"{p['attribute']}_{p['level']}": float(-p["value"] / slope)
            for p in part_worths_flat if p["attribute"] != price_name}


@router.post("/adaptive-cbc")
async def analyze_acbc(payload: ACBCPayload):
    try:
        rows, attr_cols = _tournament_to_rows(payload.attributes, payload.responses)
        if len(rows) < 4:
            raise ValueError("Not enough tournament choices to estimate utilities. "
                             "Collect more responses or enable the tournament phase.")

        hb = run_hb_mnl(
            data=rows, attribute_cols=attr_cols,
            resp_col="respondent_id", task_col="task_id", choice_col="choice",
            draws=payload.draws, tune=payload.tune, chains=payload.chains,
        )

        # Price WTP (reuse the part-worth list)
        if payload.priceAttributeId:
            price_name = next((a["name"] for a in payload.attributes
                               if a["id"] == payload.priceAttributeId), None)
            if price_name:
                wtp = _wtp(hb.get("partWorthsFlat", []), price_name)
                if wtp:
                    hb["wtp"] = wtp

        must_have, unacceptable = _detect_rules(payload.attributes, payload.responses)
        byo, n_byo = _byo_summary(payload.attributes, payload.responses)

        hb["mustHave"] = must_have
        hb["unacceptable"] = unacceptable
        hb["byoSummary"] = byo
        hb["nByo"] = n_byo
        hb["estimator"] = "ACBC — Hierarchical Bayes (tournament) + screening rules"
        return {"results": _sanitize(hb)}
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=str(e))
