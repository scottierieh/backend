"""
process_mining.py
FastAPI router for Process Mining analysis.

Endpoints:
    POST /api/analysis/process-mining        — DFG + variants + stats
    POST /api/analysis/process-mining/conformance — Conformance checking
    POST /api/analysis/process-mining/resource    — Resource analysis
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import traceback
import warnings

import pandas as pd
import numpy as np
import pm4py
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.algo.discovery.dfg import algorithm as dfg_discovery
from pm4py.algo.discovery.inductive import algorithm as inductive_miner
from pm4py.statistics.variants.log import get as variants_module
from pm4py.statistics.end_activities.log import get as end_activities_module
from pm4py.statistics.start_activities.log import get as start_activities_module
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay

warnings.filterwarnings("ignore")

router = APIRouter()


# ══════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════

def _to_native(obj):
    """numpy/pandas 타입 → Python 기본 타입 변환."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if (np.isnan(obj) or np.isinf(obj)) else float(obj)
    if isinstance(obj, np.ndarray):
        return [_to_native(x) for x in obj.tolist()]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, dict):
        return {k: _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_native(x) for x in obj]
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj


def _fmt_ms(ms: float) -> str:
    """milliseconds → 사람이 읽기 쉬운 문자열."""
    if not ms or ms <= 0:
        return "—"
    s = ms / 1000
    m = s / 60
    h = m / 60
    d = h / 24
    if d >= 30:
        return f"{d/30:.1f}mo"
    if d >= 7:
        return f"{d/7:.1f}wk"
    if d >= 1:
        return f"{d:.1f}d"
    if h >= 1:
        return f"{h:.1f}h"
    if m >= 1:
        return f"{m:.0f}m"
    return f"{s:.0f}s"


def _rows_to_eventlog(
    rows: List[Dict[str, Any]],
    case_col: str,
    activity_col: str,
    timestamp_col: str,
) -> EventLog:
    """raw rows → PM4Py EventLog 변환."""
    df = pd.DataFrame(rows)
    df = df.rename(columns={
        case_col:      "case:concept:name",
        activity_col:  "concept:name",
        timestamp_col: "time:timestamp",
    })
    df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["time:timestamp"])
    df = df.sort_values(["case:concept:name", "time:timestamp"])
    log = log_converter.apply(df, variant=log_converter.Variants.TO_EVENT_LOG)
    return log


def _percentile(arr: List[float], p: float) -> float:
    if not arr:
        return 0.0
    sorted_arr = sorted(arr)
    idx = int(len(sorted_arr) * p)
    idx = min(idx, len(sorted_arr) - 1)
    return sorted_arr[idx]


# ══════════════════════════════════════════════════════════════
# Request / Response Models
# ══════════════════════════════════════════════════════════════

class ProcessMiningRequest(BaseModel):
    data:          List[Dict[str, Any]]
    caseCol:       str
    activityCol:   str
    timestampCol:  str
    # 선택적 컬럼
    resourceCol:   Optional[str] = None
    # 분석 옵션
    maxVariants:   int = 20
    minActivityFreq: int = 1


class ConformanceRequest(BaseModel):
    data:         List[Dict[str, Any]]
    caseCol:      str
    activityCol:  str
    timestampCol: str
    referencePath: List[str]   # 이상적인 활동 순서
    threshold:    float = 0.7


class ResourceRequest(BaseModel):
    data:         List[Dict[str, Any]]
    caseCol:      str
    activityCol:  str
    timestampCol: str
    resourceCol:  str


# ══════════════════════════════════════════════════════════════
# Main Process Mining Endpoint
# ══════════════════════════════════════════════════════════════

@router.post("/api/analysis/process-mining")
async def run_process_mining(request: ProcessMiningRequest):
    try:
        rows = request.data
        if not rows:
            raise HTTPException(status_code=400, detail="No data provided.")

        # ── 1. EventLog 생성 ──────────────────────────────────
        log = _rows_to_eventlog(
            rows,
            request.caseCol,
            request.activityCol,
            request.timestampCol,
        )

        n_cases  = len(log)
        n_events = sum(len(t) for t in log)

        # ── 2. DFG (Directly Follows Graph) ───────────────────
        dfg, start_acts, end_acts = pm4py.discover_dfg(log)

        # 노드 (활동) 통계
        activity_freq: Dict[str, int] = {}
        activity_wait: Dict[str, List[float]] = {}

        for trace in log:
            for i, event in enumerate(trace):
                act = event["concept:name"]
                activity_freq[act] = activity_freq.get(act, 0) + 1
                if i > 0:
                    prev_ts  = trace[i-1]["time:timestamp"]
                    curr_ts  = event["time:timestamp"]
                    wait_ms  = (curr_ts - prev_ts).total_seconds() * 1000
                    if wait_ms >= 0:
                        if act not in activity_wait:
                            activity_wait[act] = []
                        activity_wait[act].append(wait_ms)

        # 엣지 (transition) 통계
        edge_wait: Dict[str, List[float]] = {}
        for trace in log:
            for i in range(len(trace) - 1):
                src  = trace[i]["concept:name"]
                tgt  = trace[i+1]["concept:name"]
                key  = f"{src}|||{tgt}"
                wait = (trace[i+1]["time:timestamp"] - trace[i]["time:timestamp"]).total_seconds() * 1000
                if wait >= 0:
                    if key not in edge_wait:
                        edge_wait[key] = []
                    edge_wait[key].append(wait)

        # 노드 정렬 (빈도순)
        sorted_acts = sorted(activity_freq.items(), key=lambda x: -x[1])
        rank_map    = {act: i for i, (act, _) in enumerate(sorted_acts)}

        nodes = []
        for act, freq in sorted_acts:
            if freq < request.minActivityFreq:
                continue
            waits = sorted(activity_wait.get(act, []))
            avg_w = float(np.mean(waits)) if waits else 0.0
            nodes.append({
                "id":          act,
                "label":       act,
                "frequency":   freq,
                "caseCount":   freq,  # PM4Py dfg는 case-level; 이벤트 빈도로 근사
                "avgWaitMs":   avg_w,
                "medianWaitMs": _percentile(waits, 0.5),
                "maxWaitMs":   waits[-1] if waits else 0.0,
                "isStart":     act in start_acts and act not in end_acts,
                "isEnd":       act in end_acts   and act not in start_acts,
                "x": 0, "y": 0,  # 레이아웃은 프론트 Dagre가 처리
            })

        # 엣지
        max_freq_edge = max(dfg.values()) if dfg else 1
        edges = []
        total_impact = 0.0

        for (src, tgt), freq in dfg.items():
            key    = f"{src}|||{tgt}"
            waits  = sorted(edge_wait.get(key, []))
            avg_w  = float(np.mean(waits)) if waits else 0.0
            impact = avg_w * freq

            src_rank = rank_map.get(src, 0)
            tgt_rank = rank_map.get(tgt, 0)
            is_rework = (src == tgt) or (tgt_rank < src_rank)

            edges.append({
                "id":           f"{src}|||{tgt}",
                "source":       src,
                "target":       tgt,
                "frequency":    freq,
                "avgWaitMs":    avg_w,
                "medianWaitMs": _percentile(waits, 0.5),
                "maxWaitMs":    waits[-1] if waits else 0.0,
                "isRework":     is_rework,
                "impactScore":  impact,
            })
            total_impact += impact

        edges.sort(key=lambda e: -e["impactScore"])

        # impactPct 계산
        for e in edges:
            e["impactPct"] = round(e["impactScore"] / total_impact * 100, 2) if total_impact > 0 else 0.0

        # ── 3. Variants ────────────────────────────────────────
        variants_raw = variants_module.get_variants(log)
        variant_list = sorted(variants_raw.items(), key=lambda x: -len(x[1]))

        total_cases = n_cases or 1
        cumul       = 0.0
        variants    = []

        for i, (path_key, case_ids) in enumerate(variant_list[:request.maxVariants]):
            path    = list(path_key)
            count   = len(case_ids)
            pct     = round(count / total_cases * 100, 1)
            cumul  += pct

            # variant 케이스들의 duration
            durs = []
            for trace in log:
                if trace.attributes.get("concept:name") in {t.attributes.get("concept:name") for t in case_ids}:
                    if len(trace) > 1:
                        dur = (trace[-1]["time:timestamp"] - trace[0]["time:timestamp"]).total_seconds() * 1000
                        if dur >= 0:
                            durs.append(dur)

            seen  = set()
            has_rework = False
            for act in path:
                if act in seen:
                    has_rework = True
                    break
                seen.add(act)

            variants.append({
                "id":        f"V{i+1}",
                "path":      path,
                "count":     count,
                "pct":       pct,
                "cumulPct":  round(cumul, 1),
                "avgDurMs":  float(np.mean(durs)) if durs else 0.0,
                "isHappy":   i == 0,
                "hasRework": has_rework,
            })

        # ── 4. Case Durations ──────────────────────────────────
        case_durations = []
        all_dur_ms     = []

        for trace in log:
            cid  = trace.attributes.get("concept:name", "")
            path = [e["concept:name"] for e in trace]
            if len(trace) > 1:
                dur = (trace[-1]["time:timestamp"] - trace[0]["time:timestamp"]).total_seconds() * 1000
            else:
                dur = 0.0

            # 첫 이벤트 attrs (resource, region 등)
            first_attrs = {}
            if trace:
                for k, v in trace[0].items():
                    if k not in ("concept:name", "time:timestamp", "case:concept:name"):
                        try:
                            first_attrs[k] = str(v)
                        except Exception:
                            pass

            case_durations.append({
                "caseId":     cid,
                "durationMs": dur,
                "eventCount": len(trace),
                "variant":    path,
                "attrs":      first_attrs,
            })
            if dur >= 0:
                all_dur_ms.append(dur)

        all_dur_sorted = sorted(all_dur_ms)
        avg_dur   = float(np.mean(all_dur_sorted)) if all_dur_sorted else 0.0
        med_dur   = _percentile(all_dur_sorted, 0.5)
        p90_dur   = _percentile(all_dur_sorted, 0.9)

        # ── 5. Stats ───────────────────────────────────────────
        rework_events = sum(e["frequency"] for e in edges if e["isRework"])
        rework_rate   = rework_events / n_events * 100 if n_events > 0 else 0.0

        date_range_start = None
        date_range_end   = None
        all_ts = []
        for trace in log:
            for event in trace:
                all_ts.append(event["time:timestamp"].timestamp() * 1000)
        if all_ts:
            date_range_start = min(all_ts)
            date_range_end   = max(all_ts)

        stats = {
            "totalEvents":       n_events,
            "totalCases":        n_cases,
            "uniqueActivities":  len(nodes),
            "uniqueVariants":    len(variants),
            "avgCaseDurationMs": avg_dur,
            "medianCaseDurMs":   med_dur,
            "p90CaseDurMs":      p90_dur,
            "happyPathPct":      variants[0]["pct"] if variants else 0.0,
            "reworkRate":        round(rework_rate, 2),
            "bottleneckEdge":    edges[0]["id"]    if edges else "—",
            "bottleneckNode":    sorted(nodes, key=lambda n: -n["avgWaitMs"])[0]["id"] if nodes else "—",
            "dateRange":         [date_range_start, date_range_end],
        }

        # ── 6. Case Timelines (샘플 200개) ─────────────────────
        case_timelines = {}
        for trace in log[:200]:
            cid    = trace.attributes.get("concept:name", "")
            events = []
            for event in trace:
                events.append({
                    "activity":  event["concept:name"],
                    "timestamp": event["time:timestamp"].timestamp() * 1000,
                    "resource":  str(event.get(request.resourceCol or "org:resource", "")),
                    "attrs":     {},
                })
            case_timelines[cid] = events

        # ── 7. Result ──────────────────────────────────────────
        result = {
            "nodes":          nodes,
            "edges":          edges,
            "variants":       variants,
            "caseDurations":  case_durations,
            "caseTimelines":  case_timelines,
            "stats":          stats,
            "nodeMap":        {n["id"]: n for n in nodes},
            "edgeMap":        {e["id"]: e for e in edges},
        }

        return _to_native({"results": result})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════
# Conformance Checking Endpoint
# ══════════════════════════════════════════════════════════════

@router.post("/api/analysis/process-mining/conformance")
async def run_conformance(request: ConformanceRequest):
    try:
        log = _rows_to_eventlog(
            request.data,
            request.caseCol,
            request.activityCol,
            request.timestampCol,
        )

        ref = request.referencePath
        results = []
        total_cases = len(log)

        for trace in log:
            cid    = trace.attributes.get("concept:name", "")
            actual = [e["concept:name"] for e in trace]

            # Greedy matching — reference 순서대로 actual에서 찾기
            path_match = [False] * len(ref)
            deviations = []
            ai = 0
            matched = 0
            visited = set()

            for ri, ref_act in enumerate(ref):
                found = False
                for a in range(ai, len(actual)):
                    if actual[a] == ref_act and a not in visited:
                        path_match[ri] = True
                        visited.add(a)
                        ai = a + 1
                        matched += 1
                        found = True
                        break
                if not found:
                    deviations.append({"type": "skip", "activity": ref_act})

            for a, act in enumerate(actual):
                if act not in ref:
                    deviations.append({"type": "extra", "activity": act})

            fitness = matched / len(ref) if ref else 1.0

            dur = 0.0
            if len(trace) > 1:
                dur = (trace[-1]["time:timestamp"] - trace[0]["time:timestamp"]).total_seconds() * 1000

            results.append({
                "caseId":     cid,
                "fitness":    round(fitness, 4),
                "deviations": deviations,
                "pathMatch":  path_match,
                "durationMs": dur,
            })

        results.sort(key=lambda r: r["fitness"])

        conforming     = [r for r in results if r["fitness"] >= request.threshold]
        non_conforming = [r for r in results if r["fitness"] <  request.threshold]
        avg_fitness    = float(np.mean([r["fitness"] for r in results])) if results else 0.0

        # deviation 집계
        dev_counts: Dict[str, Dict] = {}
        for r in results:
            for d in r["deviations"]:
                key = f"{d['type']}:{d['activity']}"
                if key not in dev_counts:
                    dev_counts[key] = {"type": d["type"], "activity": d["activity"], "count": 0}
                dev_counts[key]["count"] += 1

        dev_summary = sorted(dev_counts.values(), key=lambda x: -x["count"])[:10]

        return _to_native({
            "results": {
                "cases":          results,
                "conforming":     conforming,
                "nonConforming":  non_conforming,
                "avgFitness":     round(avg_fitness, 4),
                "conformanceRate": round(len(conforming) / total_cases * 100, 1) if total_cases else 0.0,
                "devSummary":     dev_summary,
                "threshold":      request.threshold,
                "totalCases":     total_cases,
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════
# Resource Analysis Endpoint
# ══════════════════════════════════════════════════════════════

@router.post("/api/analysis/process-mining/resource")
async def run_resource_analysis(request: ResourceRequest):
    try:
        log = _rows_to_eventlog(
            request.data,
            request.caseCol,
            request.activityCol,
            request.timestampCol,
        )

        res_col = request.resourceCol

        # resource별 통계 수집
        stats: Dict[str, Dict] = {}

        for trace in log:
            cid = trace.attributes.get("concept:name", "")
            for i, event in enumerate(trace):
                res = str(event.get(res_col, "Unknown") or "Unknown")
                act = event["concept:name"]

                if res not in stats:
                    stats[res] = {
                        "resource":   res,
                        "eventCount": 0,
                        "cases":      set(),
                        "activities": set(),
                        "waits":      [],
                    }

                stats[res]["eventCount"] += 1
                stats[res]["cases"].add(cid)
                stats[res]["activities"].add(act)

                if i > 0:
                    wait = (event["time:timestamp"] - trace[i-1]["time:timestamp"]).total_seconds() * 1000
                    if wait >= 0:
                        stats[res]["waits"].append(wait)

        resource_stats = []
        for res, s in stats.items():
            waits = sorted(s["waits"])
            resource_stats.append({
                "resource":    res,
                "eventCount":  s["eventCount"],
                "caseCount":   len(s["cases"]),
                "activities":  list(s["activities"]),
                "avgWaitMs":   float(np.mean(waits)) if waits else 0.0,
                "medianWaitMs": _percentile(waits, 0.5),
                "totalWaitMs": float(np.sum(waits)) if waits else 0.0,
            })

        resource_stats.sort(key=lambda x: -x["eventCount"])

        # Activity × Resource 히트맵
        all_acts = sorted(set(
            e["concept:name"] for trace in log for e in trace
        ))[:10]
        all_res = [r["resource"] for r in resource_stats[:6]]

        matrix: Dict[str, Dict[str, int]] = {act: {res: 0 for res in all_res} for act in all_acts}
        for trace in log:
            for event in trace:
                act = event["concept:name"]
                res = str(event.get(res_col, "Unknown") or "Unknown")
                if act in matrix and res in matrix[act]:
                    matrix[act][res] += 1

        return _to_native({
            "results": {
                "resourceStats": resource_stats,
                "heatmap": {
                    "activities": all_acts,
                    "resources":  all_res,
                    "matrix":     matrix,
                },
                "totalResources": len(resource_stats),
            }
        })

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{str(e)}\n{traceback.format_exc()}")
