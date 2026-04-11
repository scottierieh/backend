"""
routers/forecasting.py

POST /api/forecast/demand      — Per-location ARIMA demand forecast
POST /api/forecast/anomaly     — Temporal anomaly detection per location
"""

import math
import statistics as pystats
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from map.utils import haversine, to_native

router = APIRouter()


class BaseMapRequest(BaseModel):
    data: List[Dict[str, Any]]


# ══════════════════════════════════════════════════════════
# Per-location ARIMA Demand Forecast
# ══════════════════════════════════════════════════════════

class DemandForecastRequest(BaseModel):
    timeseries: List[Dict[str, Any]]  # [{locationId, lat, lng, date, value}, ...]
    dateCol: str
    valueCol: str
    locationIdCol: str
    horizonPeriods: int = 6           # periods to forecast ahead
    granularity: str = "month"        # day | week | month


@router.post("/api/forecast/demand")
def run_demand_forecast(req: DemandForecastRequest):
    try:
        import pmdarima as pm
    except ImportError:
        raise HTTPException(500, "pmdarima not installed")

    import re
    from datetime import datetime

    def parse_date(s: str) -> str:
        """Truncate date to granularity label."""
        match = re.search(r"(\d{4})-(\d{2})-(\d{2})", str(s))
        if not match:
            return str(s)[:10]
        y, m, d = match.groups()
        if req.granularity == "year":  return y
        if req.granularity == "month": return f"{y}-{m}"
        if req.granularity == "week":
            from datetime import date, timedelta
            dt = date(int(y), int(m), int(d))
            week_start = dt - timedelta(days=dt.weekday())
            return str(week_start)
        return f"{y}-{m}-{d}"

    # Group by location
    loc_groups: dict[str, list] = {}
    for row in req.timeseries:
        loc_id = str(row.get(req.locationIdCol, "unknown"))
        date_key = parse_date(str(row.get(req.dateCol, "")))
        val = row.get(req.valueCol)
        if isinstance(val, (int, float)):
            loc_groups.setdefault(loc_id, []).append({
                "date": date_key,
                "value": float(val),
                "lat": row.get("lat"),
                "lng": row.get("lng"),
                "row": row,
            })

    results = []
    for loc_id, entries in loc_groups.items():
        entries.sort(key=lambda x: x["date"])
        # Aggregate by date (in case multiple rows per date)
        date_vals: dict[str, list] = {}
        for e in entries:
            date_vals.setdefault(e["date"], []).append(e["value"])
        dates = sorted(date_vals.keys())
        values = [pystats.mean(date_vals[d]) for d in dates]

        if len(values) < 4:
            continue  # not enough history

        y = np.array(values)

        try:
            model = pm.auto_arima(
                y,
                seasonal=False,
                stepwise=True,
                suppress_warnings=True,
                error_action="ignore",
                max_p=3, max_q=3, max_d=2,
                information_criterion="aic",
            )
            forecast, conf_int = model.predict(
                n_periods=req.horizonPeriods,
                return_conf_int=True,
                alpha=0.2,  # 80% CI
            )
            order = model.order
            aic = round(float(model.aic()), 2)
        except Exception:
            # Fallback: simple linear trend
            x = np.arange(len(y))
            coef = np.polyfit(x, y, 1)
            future_x = np.arange(len(y), len(y) + req.horizonPeriods)
            forecast = np.polyval(coef, future_x)
            conf_int = np.column_stack([forecast * 0.85, forecast * 1.15])
            order = (1, 0, 0)
            aic = None

        # Generate future date labels
        last_date = dates[-1] if dates else "unknown"
        future_dates = _generate_future_dates(last_date, req.horizonPeriods, req.granularity)

        # First row of this location for lat/lng
        lat = entries[0].get("lat")
        lng = entries[0].get("lng")

        # Trend direction
        if len(values) >= 2:
            trend_pct = (values[-1] - values[0]) / abs(values[0]) * 100 if values[0] != 0 else 0
        else:
            trend_pct = 0.0

        forecast_color = "#22c55e" if float(forecast[-1]) > values[-1] else "#ef4444"

        results.append({
            "locationId": loc_id,
            "lat": lat,
            "lng": lng,
            "history": [{"date": d, "value": round(v, 2)} for d, v in zip(dates, values)],
            "forecast": [
                {
                    "date": future_dates[i],
                    "value": round(float(forecast[i]), 2),
                    "lower": round(float(conf_int[i][0]), 2),
                    "upper": round(float(conf_int[i][1]), 2),
                }
                for i in range(req.horizonPeriods)
            ],
            "lastValue": round(float(values[-1]), 2),
            "forecastFinal": round(float(forecast[-1]), 2),
            "trendPct": round(trend_pct, 1),
            "arimaOrder": list(order),
            "aic": aic,
            "color": forecast_color,
        })

    return to_native({
        "results": {
            "locations": results,
            "horizonPeriods": req.horizonPeriods,
            "granularity": req.granularity,
            "visible": True,
        }
    })


def _generate_future_dates(last_date: str, n: int, granularity: str) -> list[str]:
    """Generate n future date labels after last_date."""
    import re
    from datetime import date, timedelta

    match = re.search(r"(\d{4})-(\d{2})(?:-(\d{2}))?", last_date)
    if not match:
        return [f"T+{i+1}" for i in range(n)]

    y, m = int(match.group(1)), int(match.group(2))
    d = int(match.group(3)) if match.group(3) else 1

    results = []
    current = date(y, m, d)
    for _ in range(n):
        if granularity == "day":
            current = current + timedelta(days=1)
            results.append(str(current))
        elif granularity == "week":
            current = current + timedelta(weeks=1)
            results.append(str(current))
        elif granularity == "month":
            m2 = current.month + 1
            y2 = current.year + (m2 - 1) // 12
            m2 = ((m2 - 1) % 12) + 1
            current = date(y2, m2, 1)
            results.append(f"{y2}-{m2:02d}")
        else:  # year
            current = date(current.year + 1, 1, 1)
            results.append(str(current.year))
    return results


# ══════════════════════════════════════════════════════════
# Temporal Anomaly Detection
# ══════════════════════════════════════════════════════════

class TemporalAnomalyRequest(BaseModel):
    timeseries: List[Dict[str, Any]]
    dateCol: str
    valueCol: str
    locationIdCol: str
    zThreshold: float = 2.5   # z-score threshold


@router.post("/api/forecast/anomaly")
def run_temporal_anomaly(req: TemporalAnomalyRequest):
    import re

    loc_groups: dict[str, list] = {}
    for row in req.timeseries:
        loc_id = str(row.get(req.locationIdCol, "unknown"))
        val = row.get(req.valueCol)
        date_str = str(row.get(req.dateCol, ""))
        if isinstance(val, (int, float)):
            loc_groups.setdefault(loc_id, []).append({
                "date": date_str,
                "value": float(val),
                "lat": row.get("lat"),
                "lng": row.get("lng"),
                "row": row,
            })

    results = []
    all_anomalies = []

    for loc_id, entries in loc_groups.items():
        entries.sort(key=lambda x: x["date"])
        values = [e["value"] for e in entries]
        if len(values) < 3:
            continue

        mean_v = pystats.mean(values)
        std_v = pystats.stdev(values) if len(values) > 1 else 0
        if std_v == 0:
            continue

        loc_anomalies = []
        for e in entries:
            z = abs(e["value"] - mean_v) / std_v
            is_anomaly = z >= req.zThreshold
            direction = "high" if e["value"] > mean_v else "low"
            entry = {
                **e,
                "zScore": round(z, 3),
                "isAnomaly": is_anomaly,
                "direction": direction if is_anomaly else "normal",
                "deviation": round((e["value"] - mean_v) / mean_v * 100, 1) if mean_v != 0 else 0,
            }
            loc_anomalies.append(entry)
            if is_anomaly:
                all_anomalies.append({**entry, "locationId": loc_id})

        n_anomalies = sum(1 for e in loc_anomalies if e["isAnomaly"])

        lat = entries[0].get("lat")
        lng = entries[0].get("lng")

        results.append({
            "locationId": loc_id,
            "lat": lat,
            "lng": lng,
            "series": loc_anomalies,
            "nAnomalies": n_anomalies,
            "mean": round(mean_v, 2),
            "std": round(std_v, 2),
            "color": "#ef4444" if n_anomalies > 0 else "#22c55e",
        })

    results.sort(key=lambda x: -x["nAnomalies"])

    return to_native({
        "results": {
            "locations": results,
            "allAnomalies": sorted(all_anomalies, key=lambda x: -x["zScore"]),
            "totalAnomalies": len(all_anomalies),
            "zThreshold": req.zThreshold,
            "visible": True,
        }
    })
